from __future__ import annotations

from random import Random
from typing import Any

from agentic_uav.communication import Message, NetworkModel
from agentic_uav.models import (
    Cell,
    CommunicationEvent,
    ScenarioConfig,
    Sector,
    UavConfig,
    UavState,
    WorldConfig,
    WorldState,
    coverage_ratio,
    manhattan,
    neighborhood,
)
from agentic_uav.planning import Action, MethodState, ObservationBuilder


class MetricsLogger:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self.messages_sent = 0
        self.urgent_targets: set[Cell] = set()

    def log_tick(self, tick: int, world: WorldState, uavs: dict[str, UavState]) -> None:
        self.records.append(
            {
                "tick": tick,
                "coverage_ratio": coverage_ratio(world),
                "active_uavs": sum(1 for uav in uavs.values() if uav.active),
                "messages_sent": self.messages_sent,
                "uav_cells": {uav_id: uav.cell for uav_id, uav in uavs.items()},
                "uav_roles": {uav_id: uav.role for uav_id, uav in uavs.items()},
                "uav_targets": {uav_id: uav.target_cell for uav_id, uav in uavs.items()},
                "urgent_targets": sorted(self.urgent_targets),
            }
        )

    def summary(self, ticks_run: int, world: WorldState, max_ticks: int) -> dict[str, Any]:
        coverage = coverage_ratio(world)
        is_solved = coverage >= 1.0
        termination_reason = "solved" if is_solved else "max_ticks"
        if not is_solved and ticks_run < max_ticks:
            termination_reason = "running"
        return {
            "ticks_run": ticks_run,
            "coverage_ratio": coverage,
            "messages_sent": self.messages_sent,
            "urgent_targets": sorted(self.urgent_targets),
            "is_solved": is_solved,
            "termination_reason": termination_reason,
        }


class EventInjector:
    def __init__(self, events: list[CommunicationEvent]) -> None:
        self.events = events

    def apply(self, tick: int, world: WorldState, uavs: dict[str, UavState]) -> None:
        for event in self.events:
            if event.tick != tick:
                continue
            if event.event_type == "block_sector":
                cell = tuple(event.payload["cell"])
                world.sectors[cell].blocked = True
            elif event.event_type == "dropout":
                uav_id = event.payload["uav_id"]
                if uav_id in uavs:
                    uavs[uav_id].active = False
                    uavs[uav_id].health = "dropped"
            elif event.event_type == "urgent_sector":
                cell = tuple(event.payload["cell"])
                world.sectors[cell].priority = "urgent"


class Simulation:
    def __init__(self, config: ScenarioConfig, method: Any) -> None:
        self.config = config
        self.method = method
        self.world = WorldState.from_config(config.world)
        self.uavs = {
            uav.uav_id: UavState(
                uav_id=uav.uav_id,
                cell=uav.cell,
                role=uav.role,
                energy=uav.energy,
                health=uav.health,
            )
            for uav in config.uavs
        }
        self.tick = 0
        self.random = Random(config.seed)
        self.network = NetworkModel(config.communication_range)
        self.events = EventInjector(config.events)
        self.observations = ObservationBuilder(config.sensing_radius)
        self.metrics = MetricsLogger()
        self.method_state = self.method.initialize_mission(self)

    @classmethod
    def from_config(cls, config: ScenarioConfig) -> Simulation:
        from agentic_uav.policy import build_method

        return cls(config, build_method(config.method_name))

    def run(self) -> dict[str, Any]:
        while not self.is_finished:
            self.step()
        return self.metrics.summary(self.tick, self.world, self.config.ticks)

    @property
    def is_solved(self) -> bool:
        return coverage_ratio(self.world) >= 1.0

    @property
    def is_finished(self) -> bool:
        return self.is_solved or self.tick >= self.config.ticks

    @property
    def termination_reason(self) -> str:
        if self.is_solved:
            return "solved"
        if self.tick >= self.config.ticks:
            return "max_ticks"
        return "running"

    def step(self) -> None:
        if self.is_finished:
            return
        self.events.apply(self.tick, self.world, self.uavs)
        for uav in self.uavs.values():
            uav.inbox.clear()
        self.network.deliver(self.uavs)

        observations = self.observations.build(self.world, self.uavs)
        actions = self.method.decide_tick(self, observations, self.method_state)
        self.resolve_actions(actions)

        self._apply_sensing()
        self.metrics.log_tick(self.tick, self.world, self.uavs)
        self.tick += 1

    def send_messages(self, messages: list[Message]) -> None:
        self.metrics.messages_sent += len(messages)
        self.network.enqueue(messages)

    def resolve_actions(self, actions: list[Action]) -> None:
        outgoing: list[Message] = []
        for action in actions:
            uav = self.uavs.get(action.uav_id)
            if uav is None or not uav.active:
                continue
            if action.new_role is not None:
                uav.role = action.new_role
            if action.target_cell is not None and self._is_valid_target(action.target_cell):
                uav.target_cell = action.target_cell
                if self.world.sectors[action.target_cell].priority == "urgent":
                    self.metrics.urgent_targets.add(action.target_cell)
                self._move_toward(uav, action.target_cell)
            outgoing.extend(action.messages)
        self.send_messages(outgoing)

    def _is_valid_target(self, cell: Cell) -> bool:
        return self.world.in_bounds(cell) and not self.world.sectors[cell].blocked

    def _move_toward(self, uav: UavState, target: Cell) -> None:
        if uav.cell == target:
            return
        candidates = [
            (uav.cell[0] + 1, uav.cell[1]),
            (uav.cell[0] - 1, uav.cell[1]),
            (uav.cell[0], uav.cell[1] + 1),
            (uav.cell[0], uav.cell[1] - 1),
        ]
        candidates = [cell for cell in candidates if self._is_valid_target(cell)]
        if not candidates:
            return
        uav.cell = min(candidates, key=lambda cell: manhattan(cell, target))

    def _apply_sensing(self) -> None:
        for uav in self.uavs.values():
            if not uav.active:
                continue
            for cell in neighborhood(uav.cell, radius=self.config.sensing_radius):
                if cell in self.world.sectors and not self.world.sectors[cell].blocked:
                    self.world.sectors[cell].coverage = 1.0


__all__ = [
    "Action",
    "Cell",
    "CommunicationEvent",
    "EventInjector",
    "Message",
    "MethodState",
    "MetricsLogger",
    "NetworkModel",
    "ObservationBuilder",
    "ScenarioConfig",
    "Sector",
    "Simulation",
    "UavConfig",
    "UavState",
    "WorldConfig",
    "WorldState",
    "coverage_ratio",
    "manhattan",
    "neighborhood",
]
