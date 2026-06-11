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
class BeliefMap:
    """Per-UAV memory of *dynamic* world state discovered locally or via messages.

    Static layout (bounds, blocked cells) stays global; coverage progress and
    urgent events must be discovered, so each UAV keeps its own belief here.
    """

    covered: set[Cell] = field(default_factory=set)
    known_urgent: set[Cell] = field(default_factory=set)
    cleared_urgent: set[Cell] = field(default_factory=set)
    peer_intents: dict[str, Cell] = field(default_factory=dict)


@dataclass
class MethodState:
    assignments: dict[str, list[Cell]] = field(default_factory=dict)
    assignment_indices: dict[str, int] = field(default_factory=dict)
    targets_by_uav: dict[str, Cell] = field(default_factory=dict)
    roles_by_uav: dict[str, str] = field(default_factory=dict)
    known_urgent: set[Cell] = field(default_factory=set)
    peer_intents: dict[str, Cell] = field(default_factory=dict)
    task_commitments: dict[str, Cell] = field(default_factory=dict)
    beliefs: dict[str, BeliefMap] = field(default_factory=dict)
    replan_pending: bool = False
    plan_by_uav: dict[str, dict[str, Any]] = field(default_factory=dict)
    replan_reason_by_uav: dict[str, str] = field(default_factory=dict)
    last_replan_tick: int | None = None


class ObservationBuilder:
    def __init__(self, sensing_radius: int) -> None:
        self.sensing_radius = sensing_radius

    def build(
        self,
        world: WorldState,
        uavs: dict[str, UavState],
        beliefs: dict[str, BeliefMap],
    ) -> dict[str, dict[str, Any]]:
        observations: dict[str, dict[str, Any]] = {}
        for uav_id, uav in uavs.items():
            if not uav.active:
                continue
            belief = beliefs.setdefault(uav_id, BeliefMap())
            nearby = self._nearby_sectors(world, uav.cell)
            self._fold_into_belief(belief, nearby)
            observations[uav_id] = {
                "self": uav,
                "nearby": nearby,
                "known_urgent": sorted(belief.known_urgent),
                "messages": list(uav.inbox),
            }
        return observations

    def _fold_into_belief(self, belief: BeliefMap, nearby: list[Sector]) -> None:
        for sector in nearby:
            if sector.blocked:
                continue
            if sector.coverage >= 1.0:
                belief.covered.add(sector.cell)
            if sector.priority == "urgent":
                belief.known_urgent.add(sector.cell)

    def _nearby_sectors(self, world: WorldState, center: Cell) -> list[Sector]:
        cells = neighborhood(center, radius=self.sensing_radius)
        return [world.sectors[cell] for cell in cells if cell in world.sectors]


def nearest_open_urgent(simulation: object, observation: dict[str, object]) -> Cell | None:
    """Nearest urgent cell the UAV *believes* is open and uncleared.

    Reads the UAV's belief (sensed nearby + learned via messages), never a
    global world scan. Layout/blocked checks stay global (the UAV "has a map").
    """

    uav = observation["self"]
    candidates = [
        cell
        for cell in observation.get("known_urgent", [])
        if _is_open(simulation, cell)
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda cell: (manhattan(uav.cell, cell), cell))


def nearest_uncovered(simulation: object, observation: dict[str, object]) -> Cell | None:
    """Nearest cell the UAV believes is uncovered, from sensed neighbours.

    Falls back to a blind patrol target when belief offers no uncovered lead,
    so no global coverage knowledge leaks into the decision.
    """

    uav = observation["self"]
    # Exclude the UAV's own cell: standing still gets covered by sensing anyway,
    # so an uncovered lead should move the UAV outward to explore.
    candidates = [
        sector.cell
        for sector in observation.get("nearby", [])
        if isinstance(sector, Sector)
        and not sector.blocked
        and sector.coverage < 1.0
        and sector.cell != uav.cell
    ]
    if not candidates:
        return _blind_patrol_target(simulation, uav)
    return min(candidates, key=lambda cell: (manhattan(uav.cell, cell), cell))


def _is_open(simulation: object, cell: Cell | None) -> bool:
    if cell is None or cell not in simulation.world.sectors:
        return False
    return not simulation.world.sectors[cell].blocked


def _blind_patrol_target(simulation: object, uav: UavState) -> Cell:
    """A layout-only exploration target (no coverage knowledge leaked)."""

    open_cells = [
        cell for cell, sector in simulation.world.sectors.items() if not sector.blocked
    ]
    if not open_cells:
        return uav.cell
    uav_ids = sorted(simulation.uavs)
    offset = uav_ids.index(uav.uav_id) if uav.uav_id in uav_ids else 0
    ordered = sorted(open_cells)
    return ordered[(simulation.tick + offset) % len(ordered)]
