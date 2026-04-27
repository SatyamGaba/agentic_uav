from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentic_uav.communication import Message
from agentic_uav.models import Cell, Sector, UavState, WorldState, manhattan, neighborhood


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


def first_uncovered(simulation: object, cells: list[Cell]) -> Cell | None:
    for cell in cells:
        sector = simulation.world.sectors[cell]
        if not sector.blocked and sector.coverage < 1.0:
            return cell
    return None


def nearest_open_urgent(simulation: object, observation: dict[str, object]) -> Cell | None:
    uav = observation["self"]
    urgent_cells = observation["urgent_cells"]
    candidates = [
        cell
        for cell in urgent_cells
        if not simulation.world.sectors[cell].blocked
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda cell: manhattan(uav.cell, cell))


def nearest_uncovered(simulation: object, observation: dict[str, object]) -> Cell | None:
    uav = observation["self"]
    candidates = [
        cell
        for cell, sector in simulation.world.sectors.items()
        if not sector.blocked and sector.coverage < 1.0
    ]
    if not candidates:
        return uav.cell
    return min(candidates, key=lambda cell: manhattan(uav.cell, cell))
