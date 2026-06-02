from __future__ import annotations
import copy
from dataclasses import dataclass

from agentic_uav.models import UavState, WorldState
from agentic_uav.planning import Action


@dataclass
class SafetyGovernor:
    # Define a safety threshold for battery (e.g., 15% remaining)
    LOW_ENERGY_THRESHOLD: float = 0.15 

    def validate_action(self, proposed: Action, uav: UavState, world: WorldState) -> Action:
        """Check proposed action against safety rules. 
        Returns a safe, authorized version of the action without mutating inputs.
        """
        # 1. Clone the proposed action to avoid unintended side effects upstream
        safe_action = copy.deepcopy(proposed)
        
        # 2. Critical Energy Check (Failsafe triggered before total depletion)
        if uav.energy <= self.LOW_ENERGY_THRESHOLD:
            # Force emergency landing or return-to-home behavior
            safe_action.target_cell = uav.cell  # Command immediate halt/hover
            safe_action.new_role = "emergency_land"
            return safe_action

        # 3. Geofence & Obstacle Validation
        if safe_action.target_cell is not None:
            is_out_of_bounds = not world.in_bounds(safe_action.target_cell)
            
            # Use safe navigation to check if sector is blocked
            sector = world.sectors.get(safe_action.target_cell)
            is_blocked = sector is not None and sector.blocked

            if is_out_of_bounds or is_blocked:
                # Instead of None, command the UAV to stay exactly where it currently is
                safe_action.target_cell = uav.cell 
                # Optional: log a safety violation warning here

        return safe_action
