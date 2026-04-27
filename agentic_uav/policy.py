from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from agentic_uav.communication import Message
from agentic_uav.models import Cell
from agentic_uav.planning import (
    Action,
    MethodState,
    first_uncovered,
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
        assignments: dict[str, list[Cell]] = {}
        uav_ids = list(simulation.uavs.keys())
        for index, cell in enumerate(sorted(simulation.world.sectors)):
            assignments.setdefault(uav_ids[index % len(uav_ids)], []).append(cell)
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
            target = first_uncovered(simulation, assigned) or uav.cell
            actions.append(Action(uav_id=uav_id, action_type="retarget_sector", target_cell=target))
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
            target = nearest_open_urgent(simulation, observation)
            if target is None:
                target = nearest_uncovered(simulation, observation)
            actions.append(Action(uav_id=uav_id, action_type="retarget_sector", target_cell=target))
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


def build_method(method_name: str) -> SwarmMethod:
    methods: dict[str, SwarmMethod] = {
        "static": StaticPartitionMethod(),
        "rules": RuleAdaptiveMethod(),
        "agentic": AgenticMethod(),
    }
    if method_name not in methods:
        raise ValueError(f"Unknown swarm method: {method_name}")
    return methods[method_name]
