from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from agentic_uav.communication import Message
from agentic_uav.models import Cell, Sector, manhattan, neighborhood
from agentic_uav.planning import (
    Action,
    MethodState,
    nearest_open_urgent,
    nearest_uncovered,
)


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
            assigned = method_state.assignments.get(uav_id, [])
            index = method_state.assignment_indices.get(uav_id, 0)
            while index < len(assigned) and not _needs_static_visit(simulation, assigned[index]):
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
            local_urgent = _local_urgent_cells(observation)
            method_state.known_urgent.update(local_urgent)

            urgent_candidates = [
                cell for cell in sorted(method_state.known_urgent) if _is_open_cell(simulation, cell)
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
                if current_target is not None and _needs_local_visit(simulation, current_target):
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

            messages = _rule_messages(simulation, uav_id, target, role)
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
class AgenticMethod:
    method_id: str = "agentic"

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
            method_state.known_urgent.update(_local_urgent_cells(observation))

            candidates = set(_local_uncovered_cells(observation))
            candidates.update(cell for cell in method_state.known_urgent if _is_open_cell(simulation, cell))
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
                ),
            )
            role = "priority_responder" if _is_urgent_cell(simulation, target) else "coverage"
            score = _task_consideration_score(
                simulation,
                uav_id,
                uav.cell,
                target,
                method_state,
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
    methods: dict[str, SwarmMethod] = {
        "static": StaticPartitionMethod(),
        "rules": RuleAdaptiveMethod(),
        "task_consideration": TaskConsiderationMethod(),
        "agentic": AgenticMethod(),
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


def _needs_static_visit(simulation: object, cell: Cell) -> bool:
    return _is_open_cell(simulation, cell) and simulation.world.sectors[cell].coverage < 1.0


def _needs_local_visit(simulation: object, cell: Cell) -> bool:
    return _is_open_cell(simulation, cell) and simulation.world.sectors[cell].coverage < 1.0


def _ingest_messages(
    uav_id: str,
    observation: dict[str, object],
    method_state: MethodState,
) -> None:
    for message in observation.get("messages", []):
        if not isinstance(message, Message) or message.sender_id == uav_id:
            continue
        target = _message_target(message)
        if target is None:
            continue
        if message.message_type == "urgent_sector":
            method_state.known_urgent.add(target)
        elif message.message_type == "intent_summary":
            method_state.peer_intents[message.sender_id] = target
        elif message.message_type == "task_commitment":
            method_state.task_commitments[message.sender_id] = target
            method_state.peer_intents[message.sender_id] = target


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


def _rule_messages(simulation: object, uav_id: str, target: Cell, role: str) -> list[Message]:
    messages: list[Message] = []
    if _is_urgent_cell(simulation, target):
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


def _is_urgent_cell(simulation: object, cell: Cell | None) -> bool:
    return _is_open_cell(simulation, cell) and simulation.world.sectors[cell].priority == "urgent"


def _task_consideration_rank(
    simulation: object,
    uav_id: str,
    current_cell: Cell,
    candidate: Cell,
    method_state: MethodState,
) -> tuple[float, int, Cell]:
    score = _task_consideration_score(
        simulation,
        uav_id,
        current_cell,
        candidate,
        method_state,
        include_conflict=True,
    )
    return (-score, manhattan(current_cell, candidate), candidate)


def _task_consideration_score(
    simulation: object,
    uav_id: str,
    current_cell: Cell,
    candidate: Cell,
    method_state: MethodState,
    *,
    include_conflict: bool,
) -> float:
    distance = manhattan(current_cell, candidate)
    score = 40.0 if _is_urgent_cell(simulation, candidate) else 0.0
    score -= float(distance)
    score += 2.0 * _uncovered_neighbor_count(simulation, candidate)
    if method_state.targets_by_uav.get(uav_id) == candidate:
        score += 8.0
    if include_conflict:
        score -= _peer_conflict_penalty(simulation, uav_id, current_cell, candidate, method_state, score)
    return score


def _uncovered_neighbor_count(simulation: object, cell: Cell) -> int:
    return sum(
        1
        for neighbor in neighborhood(cell, radius=1)
        if _is_open_cell(simulation, neighbor) and simulation.world.sectors[neighbor].coverage < 1.0
    )


def _peer_conflict_penalty(
    simulation: object,
    uav_id: str,
    current_cell: Cell,
    candidate: Cell,
    method_state: MethodState,
    local_score: float,
) -> float:
    for peer_id, peer_target in method_state.task_commitments.items():
        if peer_id == uav_id or peer_target != candidate:
            continue
        peer_cell = simulation.uavs[peer_id].cell if peer_id in simulation.uavs else candidate
        peer_score = _task_consideration_score(
            simulation,
            peer_id,
            peer_cell,
            candidate,
            method_state,
            include_conflict=False,
        )
        if peer_score > local_score or (peer_score == local_score and peer_id < uav_id):
            return 1000.0
    return 0.0
