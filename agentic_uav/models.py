from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agentic_uav.communication import Message


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
    mission_type: str = "disaster_mapping"


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
