"""
Test suite for Master Dual-Rate Controller & Compulsory Feedback Invariant
==========================================================================
Verifies Phase 7 exit criteria:
- Fast continuous loop drives UAV toward waypoint
- Compulsory feedback invariant: forcibly stalling System 2 causes System 1
  to force HOVER_REPLAN within expiry_s seconds.
"""

import math
import time
import asyncio
import numpy as np
import pytest

from contracts import (
    DroneTelemetry,
    SubgoalIntent,
    Subgoal,
    SafetyState,
)
from adapters.webots_adapter import WebotsAdapter
from cognitive.vlm_client import MockVLMClient, SelfConsistentVLMWrapper
from cognitive.baseline_agents import NaiveVLMVelocityAgent
from main_controller import (
    SharedBlackboard,
    FastContinuousController,
    DualRateMasterController,
)


def test_fast_controller_tracking():
    """Verifies that System 1 advances physics and drives toward active subgoal."""
    adapter = WebotsAdapter(initial_pos=np.array([0.0, 0.0, 5.0]))
    initial_subgoal = Subgoal(
        subgoal_id=1,
        intent=SubgoalIntent.CORRIDOR_TRANSIT,
        waypoint_xyz=(10.0, 0.0, 5.0),
        v_target=3.0,
        issued_at=0.0,
        expiry_s=10.0,
    )
    blackboard = SharedBlackboard(initial_subgoal)
    fast_ctrl = FastContinuousController(adapter, blackboard, tick_hz=20.0)

    # Run 10 ticks (0.5 seconds of flight)
    for _ in range(10):
        fast_ctrl.tick()

    tel = adapter.get_telemetry()
    assert tel.position[0] > 0.0  # Has advanced forward along +X toward waypoint (10, 0, 5)
    assert tel.timestamp == pytest.approx(0.5, abs=1e-3)
    assert blackboard.active_safety_state == SafetyState.NORMAL


def test_compulsory_feedback_invariant_stall():
    """Crucial Phase 7 Test:

    When System 2 stalls and fails to refresh Subgoal within expiry_s,
    System 1 MUST enforce HOVER_REPLAN and awaken System 2 via trigger_event.
    """
    adapter = WebotsAdapter(initial_pos=np.array([0.0, 0.0, 5.0]))

    # Short expiry of 0.2 seconds
    expired_subgoal = Subgoal(
        subgoal_id=1,
        intent=SubgoalIntent.TARGET_APPROACH,
        waypoint_xyz=(20.0, 0.0, 5.0),
        v_target=3.0,
        issued_at=0.0,
        expiry_s=0.2, # Expires after 0.2s of sim-time
    )
    blackboard = SharedBlackboard(expired_subgoal)
    fast_ctrl = FastContinuousController(adapter, blackboard, tick_hz=20.0)

    # 1. First 3 ticks: t = 0.05, 0.10, 0.15s (not yet expired)
    for _ in range(3):
        fast_ctrl.tick()
    assert blackboard.active_safety_state == SafetyState.NORMAL

    # 2. Advance 4 more ticks: t reaches 0.35s > 0.2s expiry!
    for _ in range(4):
        fast_ctrl.tick()

    # Invariant Assertion: Safety FSM must trigger HOVER_REPLAN!
    assert blackboard.active_safety_state == SafetyState.HOVER_REPLAN

    # Invariant Assertion: Trigger event must be set to awaken System 2!
    assert blackboard.trigger_event.is_set() is True

    # Allow smooth jerk-limited deceleration to bring UAV to rest (25 ticks = 1.25s, 5 time constants)
    for _ in range(25):
        fast_ctrl.tick()

    # Invariant Assertion: Commanded velocity has smoothly brought UAV to complete rest!
    assert np.allclose(adapter.vel, np.zeros(3), atol=1e-2)
