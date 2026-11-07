"""
Safety Fallback State Machine
=============================
Veto layer restricting FSM logic exclusively to emergency collision repulsion,
3D target re-acquisition, and communication timeout failsafes.
"""

from __future__ import annotations
import logging
from typing import Tuple, Optional
import numpy as np

from contracts import SafetyState, TargetRecoveryAction

logger = logging.getLogger("MavicVLM")


class SafetyFallbackFSM:
    """Evaluates low-level invariants and issues safety overrides when necessary."""

    def __init__(self, emergency_clearance_m: float = 1.5):
        self.emergency_clearance_m = emergency_clearance_m
        self.state = SafetyState.NORMAL

    def step(
        self,
        d_frontal: float,
        tracker_action: TargetRecoveryAction,
        comm_timeout: bool = False,
    ) -> Tuple[SafetyState, Optional[np.ndarray], Optional[float]]:
        """Evaluates safety hierarchy and returns: (SafetyState, override_velocity_cmd, override_yaw_rate).
        
        If override_velocity_cmd is None, System 1 proceeds with normal APF execution.
        """
        # Priority 1: Emergency Obstacle Repulsion (Hard Collision Prevention)
        if d_frontal <= self.emergency_clearance_m:
            self.state = SafetyState.EMERGENCY_REPULSION
            logger.warning(f"[SafetyFSM] ⚠️ Proximity breach! d_frontal={d_frontal:.2f}m <= {self.emergency_clearance_m}m. Forcing hard brake & climb.")
            # Zero horizontal velocity, climb vertically by 1.0 m/s
            override_v = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            return self.state, override_v, 0.0

        # Priority 2: System 2 Communication / Heartbeat Timeout
        if comm_timeout:
            self.state = SafetyState.HOVER_REPLAN
            logger.warning("[SafetyFSM] ⚠️ System 2 communication timeout. Forcing failsafe hover.")
            override_v = np.zeros(3, dtype=np.float64)
            return self.state, override_v, 0.0

        # Priority 3: Target Lost / Re-planning trigger
        if tracker_action.trigger_replan:
            self.state = SafetyState.HOVER_REPLAN
            override_v = np.zeros(3, dtype=np.float64)
            return self.state, override_v, 0.0

        # Priority 4: Active 3D Target Re-acquisition or Budgeted Scan
        if tracker_action.velocity_cmd is not None:
            if np.linalg.norm(tracker_action.velocity_cmd) > 0.1:
                self.state = SafetyState.REACQUISITION
            else:
                self.state = SafetyState.BUDGETED_SEARCH
            return self.state, tracker_action.velocity_cmd, tracker_action.yaw_rate_cmd

        # Normal Flight
        self.state = SafetyState.NORMAL
        return self.state, None, None
