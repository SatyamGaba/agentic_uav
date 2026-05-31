"""Shared type definitions for the agentic_uav package.

Centralises type aliases and typed dictionaries to avoid circular imports
and give every module proper IDE support and type-checker coverage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

if TYPE_CHECKING:
    from agentic_uav.communication import Message
    from agentic_uav.models import Cell, Sector, UavState


class Observation(TypedDict):
    """Per-UAV observation produced by ``ObservationBuilder`` each tick.

    Keys:
        self: The observing UAV's current state.
        urgent_cells: Global list of cells with ``priority == "urgent"``.
        nearby: Sectors within the UAV's sensing radius.
        messages: Incoming messages delivered this tick.
    """

    self: UavState
    urgent_cells: list[Cell]
    nearby: list[Sector]
    messages: list[Message]
