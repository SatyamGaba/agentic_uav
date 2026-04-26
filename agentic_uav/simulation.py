from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import Any


Cell = tuple[int, int]


@dataclass
class Sector:
    cell: Cell
    coverage: float = 0.0
    priority: str = "normal"
    blocked: bool = False
    hazard: str = "none"
    visibility: float = 1.0
    comm_quality: float = 1.0


@dataclass
class WorldConfig:
    width: int
    height: int
    sectors: list[Sector] = field(default_factory=list)


@dataclass
class UavConfig:
    uav_id: str
    cell: Cell
    role: str = "coverage"
    energy: float = 1.0
    health: str = "nominal"


@dataclass
class CommunicationEvent:
    tick: int
    event_type: str
    payload: dict[str, Any]


@dataclass
class ScenarioConfig:
    method_name: str
    ticks: int
    communication_range: int
    sensing_radius: int
    heartbeat_interval: int
    urgent_message_ttl: int
    world: WorldConfig
    uavs: list[UavConfig]
    events: list[CommunicationEvent] = field(default_factory=list)
    seed: int = 0


@dataclass
class Message:
    sender_id: str
    message_type: str
    payload: dict[str, Any]
    ttl: int
    urgency: str = "routine"
    recipient_id: str | None = None


@dataclass
class UavState:
    uav_id: str
    cell: Cell
    role: str = "coverage"
    energy: float = 1.0
    health: str = "nominal"
    target_cell: Cell | None = None
    inbox: list[Message] = field(default_factory=list)
    outbox: list[Message] = field(default_factory=list)
    active: bool = True


@dataclass
class WorldState:
    width: int
    height: int
    sectors: dict[Cell, Sector]

    @classmethod
    def from_config(cls, config: WorldConfig) -> WorldState:
        sectors = {
            (x, y): Sector(cell=(x, y))
            for y in range(config.height)
            for x in range(config.width)
        }
        for sector in config.sectors:
            sectors[sector.cell] = sector
        return cls(width=config.width, height=config.height, sectors=sectors)

    def in_bounds(self, cell: Cell) -> bool:
        x, y = cell
        return 0 <= x < self.width and 0 <= y < self.height


@dataclass
class Action:
    uav_id: str
    action_type: str
    target_cell: Cell | None = None
    new_role: str | None = None
    messages: list[Message] = field(default_factory=list)


@dataclass
class MethodState:
    assignments: dict[str, list[Cell]] = field(default_factory=dict)


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
            }
        )

    def summary(self, ticks_run: int, world: WorldState) -> dict[str, Any]:
        return {
            "ticks_run": ticks_run,
            "coverage_ratio": coverage_ratio(world),
            "messages_sent": self.messages_sent,
            "urgent_targets": sorted(self.urgent_targets),
        }


class NetworkModel:
    def __init__(self, communication_range: int) -> None:
        self.communication_range = communication_range
        self.pending: list[Message] = []

    def enqueue(self, messages: list[Message]) -> None:
        self.pending.extend(messages)

    def deliver(self, uavs: dict[str, UavState]) -> None:
        current = self.pending
        self.pending = []
        forwarded: list[Message] = []

        for message in current:
            delivered_to = self._neighbors(message.sender_id, uavs, message.recipient_id)
            for uav_id in delivered_to:
                received = Message(
                    sender_id=message.sender_id,
                    message_type=message.message_type,
                    payload=dict(message.payload),
                    ttl=message.ttl,
                    urgency=message.urgency,
                    recipient_id=uav_id,
                )
                uavs[uav_id].inbox.append(received)
                if message.urgency == "urgent" and message.ttl > 1:
                    forwarded.append(
                        Message(
                            sender_id=uav_id,
                            message_type=message.message_type,
                            payload=dict(message.payload),
                            ttl=message.ttl - 1,
                            urgency=message.urgency,
                        )
                    )
        self.pending.extend(forwarded)

    def _neighbors(
        self,
        sender_id: str,
        uavs: dict[str, UavState],
        recipient_id: str | None = None,
    ) -> list[str]:
        if sender_id not in uavs:
            return []
        sender = uavs[sender_id]
        if not sender.active:
            return []

        neighbors: list[str] = []
        for uav_id, uav in uavs.items():
            if uav_id == sender_id or not uav.active:
                continue
            if recipient_id is not None and uav_id != recipient_id:
                continue
            if manhattan(sender.cell, uav.cell) <= self.communication_range:
                neighbors.append(uav_id)
        return neighbors


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


class ObservationBuilder:
    def __init__(self, sensing_radius: int) -> None:
        self.sensing_radius = sensing_radius

    def build(self, world: WorldState, uavs: dict[str, UavState]) -> dict[str, dict[str, Any]]:
        observations: dict[str, dict[str, Any]] = {}
        urgent_cells = [
            cell for cell, sector in world.sectors.items() if sector.priority == "urgent"
        ]
        for uav_id, uav in uavs.items():
            if not uav.active:
                continue
            observations[uav_id] = {
                "self": uav,
                "urgent_cells": urgent_cells,
                "nearby": self._nearby_sectors(world, uav.cell),
            }
        return observations

    def _nearby_sectors(self, world: WorldState, center: Cell) -> list[Sector]:
        cells = neighborhood(center, radius=self.sensing_radius)
        return [world.sectors[cell] for cell in cells if cell in world.sectors]


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
        from agentic_uav.methods import build_method

        return cls(config, build_method(config.method_name))

    def run(self) -> dict[str, Any]:
        for _ in range(self.config.ticks):
            self.step()
        return self.metrics.summary(self.tick, self.world)

    @property
    def is_finished(self) -> bool:
        return self.tick >= self.config.ticks

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


def coverage_ratio(world: WorldState) -> float:
    sectors = [sector for sector in world.sectors.values() if not sector.blocked]
    if not sectors:
        return 0.0
    return sum(1 for sector in sectors if sector.coverage >= 1.0) / len(sectors)


def neighborhood(cell: Cell, radius: int) -> list[Cell]:
    x0, y0 = cell
    return [
        (x, y)
        for y in range(y0 - radius, y0 + radius + 1)
        for x in range(x0 - radius, x0 + radius + 1)
    ]


def manhattan(left: Cell, right: Cell) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])
