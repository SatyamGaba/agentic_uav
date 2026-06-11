from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from agentic_uav.communication import Message
from agentic_uav.models import Cell, Sector, manhattan, neighborhood
from agentic_uav.planning import (
    Action,
    BeliefMap,
    MethodState,
    nearest_open_urgent,
    nearest_uncovered,
)

if TYPE_CHECKING:
    from agentic_uav.llm_agent import AgentDecider


class SwarmMethod(Protocol):
    method_id: str

    def initialize_mission(self, simulation: object) -> MethodState:
        ...

    def decide_tick(
        self,
        simulation: object,
        observations: dict[str, dict[str, object]],
        method_state: MethodState,
    ) -> list[Action]:
        ...

    def handle_event(self, event: object, method_state: MethodState) -> None:
        ...


@dataclass
class StaticPartitionMethod:
    method_id: str = "static"

    def initialize_mission(self, simulation: object) -> MethodState:
        uav_ids = sorted(simulation.uavs)
        if not uav_ids:
            return MethodState()

        cells = _serpentine_cells(simulation)
        assignments: dict[str, list[Cell]] = {}
        start = 0
        base_size, remainder = divmod(len(cells), len(uav_ids))
        for index, uav_id in enumerate(uav_ids):
            chunk_size = base_size + (1 if index < remainder else 0)
            assignments[uav_id] = cells[start : start + chunk_size]
            start += chunk_size
        return MethodState(assignments=assignments)

    def decide_tick(
        self,
        simulation: object,
        observations: dict[str, dict[str, object]],
        method_state: MethodState,
    ) -> list[Action]:
        actions: list[Action] = []
        for uav_id, observation in observations.items():
            uav = observation["self"]
            belief = method_state.beliefs.setdefault(uav_id, BeliefMap())
            assigned = method_state.assignments.get(uav_id, [])
            index = method_state.assignment_indices.get(uav_id, 0)
            while index < len(assigned) and not _needs_static_visit(
                simulation, assigned[index], belief
            ):
                index += 1
            method_state.assignment_indices[uav_id] = index
            target = assigned[index] if index < len(assigned) else uav.cell
            action_type = "continue_assignment" if uav.target_cell == target else "retarget_sector"
            actions.append(Action(uav_id=uav_id, action_type=action_type, target_cell=target))
        return actions

    def handle_event(self, event: object, method_state: MethodState) -> None:
        return None


@dataclass
class RuleAdaptiveMethod:
    method_id: str = "rules"

    def initialize_mission(self, simulation: object) -> MethodState:
        return MethodState()

    def decide_tick(
        self,
        simulation: object,
        observations: dict[str, dict[str, object]],
        method_state: MethodState,
    ) -> list[Action]:
        actions: list[Action] = []
        for uav_id, observation in observations.items():
            uav = observation["self"]
            _ingest_messages(uav_id, observation, method_state)
            belief = method_state.beliefs.setdefault(uav_id, BeliefMap())
            local_urgent = _local_urgent_cells(observation)
            method_state.known_urgent.update(local_urgent)
            belief.known_urgent.update(local_urgent)

            urgent_candidates = [
                cell for cell in sorted(belief.known_urgent) if _is_open_cell(simulation, cell)
            ]
            target = _choose_unclaimed(
                urgent_candidates,
                uav_id,
                uav.cell,
                method_state.peer_intents,
            )
            role = "priority_responder" if target is not None else "coverage"

            if target is None:
                current_target = method_state.targets_by_uav.get(uav_id)
                if current_target is not None and _needs_local_visit(simulation, current_target, belief):
                    target = current_target

            if target is None:
                local_uncovered = _local_uncovered_cells(observation)
                target = _choose_unclaimed(
                    local_uncovered,
                    uav_id,
                    uav.cell,
                    method_state.peer_intents,
                )

            if target is None:
                target = _patrol_target(simulation, uav_id)

            messages = _rule_messages(simulation, uav_id, target, role, belief)
            method_state.targets_by_uav[uav_id] = target
            method_state.roles_by_uav[uav_id] = role
            actions.append(
                Action(
                    uav_id=uav_id,
                    action_type="switch_role",
                    target_cell=target,
                    new_role=role,
                    messages=messages,
                )
            )
        return actions

    def handle_event(self, event: object, method_state: MethodState) -> None:
        return None


@dataclass
class GreedyMethod:
    method_id: str = "greedy"

    def initialize_mission(self, simulation: object) -> MethodState:
        return MethodState()

    def decide_tick(
        self,
        simulation: object,
        observations: dict[str, dict[str, object]],
        method_state: MethodState,
    ) -> list[Action]:
        actions: list[Action] = []
        for uav_id, observation in observations.items():
            _ingest_messages(uav_id, observation, method_state)
            actions.append(greedy_decision(simulation, observation, method_state))
        return actions

    def handle_event(self, event: object, method_state: MethodState) -> None:
        return None


def greedy_decision(
    simulation: object,
    observation: dict[str, object],
    method_state: MethodState,
) -> Action:
    """Greedy heuristic decision for a single UAV from its own observation.

    Reused as the deterministic fallback for the LLM-driven AgenticMethod.
    """

    uav = observation["self"]
    uav_id = uav.uav_id
    target = nearest_open_urgent(simulation, observation)
    messages: list[Message] = []
    role = "priority_responder"
    if target is not None:
        messages.append(
            Message(
                sender_id=uav_id,
                message_type="intent_summary",
                payload={"target_cell": target, "role": role},
                ttl=1,
                urgency="routine",
            )
        )
    else:
        target = nearest_uncovered(simulation, observation)
        role = "coverage"
    return Action(
        uav_id=uav_id,
        action_type="switch_role",
        target_cell=target,
        new_role=role,
        messages=messages,
    )


@dataclass
class AgenticMethod:
    method_id: str = "agentic"
    decider: "AgentDecider | None" = None

    def __post_init__(self) -> None:
        if self.decider is None:
            from agentic_uav.llm_agent import MockAgentDecider

            self.decider = MockAgentDecider()

    def initialize_mission(self, simulation: object) -> MethodState:
        return MethodState(replan_pending=True)

    def handle_event(self, event: object, method_state: MethodState) -> None:
        method_state.replan_pending = True
        reason = getattr(event, "event_type", "event")
        for uav_id in method_state.plan_by_uav:
            method_state.replan_reason_by_uav[uav_id] = reason
        method_state.replan_reason_by_uav["_last"] = reason

    def decide_tick(
        self,
        simulation: object,
        observations: dict[str, dict[str, object]],
        method_state: MethodState,
    ) -> list[Action]:
        for uav_id, observation in observations.items():
            _ingest_messages(uav_id, observation, method_state)

        if method_state.replan_pending:
            actions = self._replan(simulation, observations, method_state)
            method_state.replan_pending = False
            method_state.last_replan_tick = getattr(simulation, "tick", None)
            reason = method_state.replan_reason_by_uav.get("_last", "mission_start")
            simulation.metrics.log_replan(
                simulation.tick, reason, list(observations.keys())
            )
            return actions

        return self._execute_plan(simulation, observations, method_state)

    def _replan(
        self,
        simulation: object,
        observations: dict[str, dict[str, object]],
        method_state: MethodState,
    ) -> list[Action]:
        from agentic_uav.llm_agent import decide_proposal

        # Decentralized: each active UAV proposes a {target, role} from its OWN view.
        proposals: dict[str, Action] = {}
        ranked_by_uav: dict[str, list[Cell]] = {}
        for uav_id, observation in observations.items():
            action, ranked = decide_proposal(
                self.decider, simulation, observation, method_state
            )
            proposals[uav_id] = action
            ranked_by_uav[uav_id] = ranked

        # Negotiation: deconflict colliding targets synchronously, in-tick.
        self._deconflict(simulation, observations, method_state, proposals, ranked_by_uav)

        actions: list[Action] = []
        for uav_id, observation in observations.items():
            action = proposals[uav_id]
            method_state.plan_by_uav[uav_id] = {
                "target_cell": action.target_cell,
                "role": action.new_role,
            }
            method_state.targets_by_uav[uav_id] = action.target_cell
            method_state.roles_by_uav[uav_id] = action.new_role
            actions.append(action)
        return actions

    def _deconflict(
        self,
        simulation: object,
        observations: dict[str, dict[str, object]],
        method_state: MethodState,
        proposals: dict[str, Action],
        ranked_by_uav: dict[str, list[Cell]],
    ) -> None:
        claimed: dict[Cell, str] = {}
        for uav_id in sorted(proposals):
            target = proposals[uav_id].target_cell
            if target is None:
                continue
            holder = claimed.get(target)
            if holder is None:
                claimed[target] = uav_id
                continue
            # Contest: nearest UAV wins, tie broken by lexicographic uav_id.
            holder_dist = manhattan(observations[holder]["self"].cell, target)
            uav_dist = manhattan(observations[uav_id]["self"].cell, target)
            if (uav_dist, uav_id) < (holder_dist, holder):
                loser, winner = holder, uav_id
            else:
                loser, winner = uav_id, holder
            claimed[target] = winner
            proposals[loser] = self._repick(
                simulation,
                observations[loser],
                method_state,
                proposals[loser],
                ranked_by_uav.get(loser, []),
                claimed,
            )
            new_target = proposals[loser].target_cell
            if new_target is not None and new_target not in claimed:
                claimed[new_target] = loser

    def _repick(
        self,
        simulation: object,
        observation: dict[str, object],
        method_state: MethodState,
        action: Action,
        ranked: list[Cell],
        claimed: dict[Cell, str],
    ) -> Action:
        # Prefer the loser's next-best ranked target, else fall back to greedy.
        for cell in ranked:
            if cell not in claimed and _is_open_cell(simulation, cell):
                return Action(
                    uav_id=action.uav_id,
                    action_type="switch_role",
                    target_cell=cell,
                    new_role=action.new_role,
                    messages=action.messages,
                )
        fallback = greedy_decision(simulation, observation, method_state)
        if fallback.target_cell is not None and fallback.target_cell in claimed:
            fallback = Action(
                uav_id=fallback.uav_id,
                action_type="switch_role",
                target_cell=observation["self"].cell,
                new_role=fallback.new_role,
                messages=fallback.messages,
            )
        return fallback

    def _execute_plan(
        self,
        simulation: object,
        observations: dict[str, dict[str, object]],
        method_state: MethodState,
    ) -> list[Action]:
        actions: list[Action] = []
        for uav_id, observation in observations.items():
            plan = method_state.plan_by_uav.get(uav_id)
            if plan is None:
                action = greedy_decision(simulation, observation, method_state)
                method_state.plan_by_uav[uav_id] = {
                    "target_cell": action.target_cell,
                    "role": action.new_role,
                }
                actions.append(action)
                continue
            target = plan.get("target_cell")
            role = plan.get("role") or "coverage"
            messages = [
                Message(
                    sender_id=uav_id,
                    message_type="intent_summary",
                    payload={"target_cell": target, "role": role},
                    ttl=1,
                    urgency="routine",
                )
            ]
            actions.append(
                Action(
                    uav_id=uav_id,
                    action_type="execute_plan",
                    target_cell=target,
                    new_role=role,
                    messages=messages,
                )
            )
        return actions


@dataclass
class TaskConsiderationMethod:
    method_id: str = "task_consideration"

    def initialize_mission(self, simulation: object) -> MethodState:
        return MethodState()

    def decide_tick(
        self,
        simulation: object,
        observations: dict[str, dict[str, object]],
        method_state: MethodState,
    ) -> list[Action]:
        actions: list[Action] = []
        for uav_id, observation in observations.items():
            uav = observation["self"]
            _ingest_messages(uav_id, observation, method_state)
            belief = method_state.beliefs.setdefault(uav_id, BeliefMap())
            local_urgent = _local_urgent_cells(observation)
            method_state.known_urgent.update(local_urgent)
            belief.known_urgent.update(local_urgent)

            candidates = set(_local_uncovered_cells(observation))
            candidates.update(cell for cell in belief.known_urgent if _is_open_cell(simulation, cell))
            if not candidates:
                candidates.add(nearest_uncovered(simulation, observation))

            target = min(
                candidates,
                key=lambda cell: _task_consideration_rank(
                    simulation,
                    uav_id,
                    uav.cell,
                    cell,
                    method_state,
                    belief,
                ),
            )
            role = "priority_responder" if _believes_urgent(simulation, target, belief) else "coverage"
            score = _task_consideration_score(
                simulation,
                uav_id,
                uav.cell,
                target,
                method_state,
                belief,
                include_conflict=False,
            )
            method_state.targets_by_uav[uav_id] = target
            method_state.roles_by_uav[uav_id] = role
            method_state.task_commitments[uav_id] = target

            actions.append(
                Action(
                    uav_id=uav_id,
                    action_type="switch_role",
                    target_cell=target,
                    new_role=role,
                    messages=[
                        Message(
                            sender_id=uav_id,
                            message_type="task_commitment",
                            payload={"target_cell": target, "role": role, "score": score},
                            ttl=1,
                            urgency="routine",
                        )
                    ],
                )
            )
        return actions

    def handle_event(self, event: object, method_state: MethodState) -> None:
        return None


def build_method(method_name: str) -> SwarmMethod:
    from agentic_uav.llm_agent import build_decider

    methods: dict[str, SwarmMethod] = {
        "static": StaticPartitionMethod(),
        "rules": RuleAdaptiveMethod(),
        "task_consideration": TaskConsiderationMethod(),
        "greedy": GreedyMethod(),
        "agentic": AgenticMethod(decider=build_decider()),
    }
    if method_name not in methods:
        raise ValueError(f"Unknown swarm method: {method_name}")
    return methods[method_name]


def _serpentine_cells(simulation: object) -> list[Cell]:
    cells: list[Cell] = []
    for y in range(simulation.world.height):
        x_values = range(simulation.world.width)
        if y % 2:
            x_values = reversed(range(simulation.world.width))
        for x in x_values:
            cell = (x, y)
            if _is_open_cell(simulation, cell):
                cells.append(cell)
    return cells


def _is_open_cell(simulation: object, cell: Cell | None) -> bool:
    if cell is None or cell not in simulation.world.sectors:
        return False
    return not simulation.world.sectors[cell].blocked


def _needs_static_visit(simulation: object, cell: Cell, belief: BeliefMap) -> bool:
    # Layout/blocked stays global; coverage progress comes from the UAV's belief.
    return _is_open_cell(simulation, cell) and cell not in belief.covered


def _needs_local_visit(simulation: object, cell: Cell, belief: BeliefMap) -> bool:
    # Layout/blocked stays global; coverage progress comes from the UAV's belief.
    return _is_open_cell(simulation, cell) and cell not in belief.covered


def _ingest_messages(
    uav_id: str,
    observation: dict[str, object],
    method_state: MethodState,
) -> None:
    belief = method_state.beliefs.setdefault(uav_id, BeliefMap())
    for message in observation.get("messages", []):
        if not isinstance(message, Message) or message.sender_id == uav_id:
            continue
        target = _message_target(message)
        if target is None:
            continue
        if message.message_type == "urgent_sector":
            method_state.known_urgent.add(target)
            belief.known_urgent.add(target)
        elif message.message_type == "intent_summary":
            method_state.peer_intents[message.sender_id] = target
            belief.peer_intents[message.sender_id] = target
        elif message.message_type == "task_commitment":
            method_state.task_commitments[message.sender_id] = target
            method_state.peer_intents[message.sender_id] = target
            belief.peer_intents[message.sender_id] = target
    # Surface the UAV's belief of peer intents on its own observation so the
    # serialized (LLM-facing) view is populated and negotiation can use it as a
    # prior. Each entry is JSON-safe and local to this UAV.
    observation["peer_intents"] = [
        {"peer_id": peer_id, "target_cell": list(cell)}
        for peer_id, cell in sorted(belief.peer_intents.items())
    ]


def _message_target(message: Message) -> Cell | None:
    raw_cell = message.payload.get("target_cell", message.payload.get("cell"))
    if not isinstance(raw_cell, (list, tuple)) or len(raw_cell) != 2:
        return None
    return (int(raw_cell[0]), int(raw_cell[1]))


def _local_urgent_cells(observation: dict[str, object]) -> list[Cell]:
    return sorted(
        sector.cell
        for sector in observation.get("nearby", [])
        if isinstance(sector, Sector) and sector.priority == "urgent" and not sector.blocked
    )


def _local_uncovered_cells(observation: dict[str, object]) -> list[Cell]:
    return sorted(
        sector.cell
        for sector in observation.get("nearby", [])
        if isinstance(sector, Sector) and not sector.blocked and sector.coverage < 1.0
    )


def _choose_unclaimed(
    candidates: list[Cell],
    uav_id: str,
    current_cell: Cell,
    peer_intents: dict[str, Cell],
) -> Cell | None:
    if not candidates:
        return None
    open_candidates = sorted(set(candidates), key=lambda cell: (manhattan(current_cell, cell), cell))
    unclaimed = [
        cell
        for cell in open_candidates
        if not any(peer_id < uav_id and peer_target == cell for peer_id, peer_target in peer_intents.items())
    ]
    return (unclaimed or open_candidates)[0]


def _rule_messages(
    simulation: object, uav_id: str, target: Cell, role: str, belief: BeliefMap
) -> list[Message]:
    messages: list[Message] = []
    if _believes_urgent(simulation, target, belief):
        messages.append(
            Message(
                sender_id=uav_id,
                message_type="urgent_sector",
                payload={"cell": target},
                ttl=simulation.config.urgent_message_ttl,
                urgency="urgent",
            )
        )
    if simulation.tick % simulation.config.heartbeat_interval == 0:
        messages.append(
            Message(
                sender_id=uav_id,
                message_type="intent_summary",
                payload={"target_cell": target, "role": role},
                ttl=1,
                urgency="routine",
            )
        )
    return messages


def _patrol_target(simulation: object, uav_id: str) -> Cell:
    cells = _serpentine_cells(simulation)
    if not cells:
        return simulation.uavs[uav_id].cell
    uav_ids = sorted(simulation.uavs)
    offset = uav_ids.index(uav_id) if uav_id in uav_ids else 0
    return cells[(simulation.tick + offset) % len(cells)]


def _believes_urgent(simulation: object, cell: Cell | None, belief: BeliefMap) -> bool:
    # Urgency is *dynamic* state: a UAV may only act on cells it has sensed or
    # learned about (belief.known_urgent), never ground-truth world priority of
    # an unsensed cell. Layout/blocked stays a global read.
    return _is_open_cell(simulation, cell) and cell in belief.known_urgent


def _task_consideration_rank(
    simulation: object,
    uav_id: str,
    current_cell: Cell,
    candidate: Cell,
    method_state: MethodState,
    belief: BeliefMap,
) -> tuple[float, int, Cell]:
    score = _task_consideration_score(
        simulation,
        uav_id,
        current_cell,
        candidate,
        method_state,
        belief,
        include_conflict=True,
    )
    return (-score, manhattan(current_cell, candidate), candidate)


def _task_consideration_score(
    simulation: object,
    uav_id: str,
    current_cell: Cell,
    candidate: Cell,
    method_state: MethodState,
    belief: BeliefMap,
    *,
    include_conflict: bool,
) -> float:
    distance = manhattan(current_cell, candidate)
    score = 40.0 if _believes_urgent(simulation, candidate, belief) else 0.0
    score -= float(distance)
    score += 2.0 * _uncovered_neighbor_count(simulation, candidate, belief)
    if method_state.targets_by_uav.get(uav_id) == candidate:
        score += 8.0
    if include_conflict:
        score -= _peer_conflict_penalty(
            simulation, uav_id, current_cell, candidate, method_state, belief, score
        )
    return score


def _uncovered_neighbor_count(simulation: object, cell: Cell, belief: BeliefMap) -> int:
    # Layout/blocked stays global; coverage comes from the deciding UAV's belief
    # (it cannot read ground-truth coverage of cells it has not sensed).
    return sum(
        1
        for neighbor in neighborhood(cell, radius=1)
        if _is_open_cell(simulation, neighbor) and neighbor not in belief.covered
    )


def _peer_conflict_penalty(
    simulation: object,
    uav_id: str,
    current_cell: Cell,
    candidate: Cell,
    method_state: MethodState,
    belief: BeliefMap,
    local_score: float,
) -> float:
    for peer_id, peer_target in method_state.task_commitments.items():
        if peer_id == uav_id or peer_target != candidate:
            continue
        peer_cell = simulation.uavs[peer_id].cell if peer_id in simulation.uavs else candidate
        # The deciding UAV estimates the peer's score from its OWN belief (it has
        # no access to the peer's private belief); this avoids any global read.
        peer_score = _task_consideration_score(
            simulation,
            peer_id,
            peer_cell,
            candidate,
            method_state,
            belief,
            include_conflict=False,
        )
        if peer_score > local_score or (peer_score == local_score and peer_id < uav_id):
            return 1000.0
    return 0.0
