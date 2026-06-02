from __future__ import annotations

from dataclasses import dataclass
from typing import Set

from agentic_uav.models import Cell

@dataclass
class BlackoutZone:
    """A region where messages cannot be delivered."""
    cells: Set[Cell]
    start_tick: int
    end_tick: int | None = None

    def is_active(self, tick: int) -> bool:
        """Check if the blackout zone is active at the given tick."""
        if tick < self.start_tick:
            return False
        if self.end_tick is not None and tick >= self.end_tick:
            return False
        return True
