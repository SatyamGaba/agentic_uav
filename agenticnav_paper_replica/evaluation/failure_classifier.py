"""Waterfall failure classification taxonomy for AgenticNav replication (v7 Specification)."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


def classify_failure(
    success: bool,
    oracle_success: bool,
    called_stop: bool,
    final_distance_to_goal: float,
    action_budget_exceeded: bool,
    deadlock_terminated: bool,
    protocol_violation: bool = False,
    infrastructure_error: bool = False,
    success_distance_threshold: float = 3.0,
) -> Optional[str]:
    """Classifies an episode outcome according to the strict priority waterfall (Section 10.2).

    Priority:
        1. Infrastructure failure: excluded and re-run.
        2. Safety-Deadlock: R_max reached (consecutive reselect exhaustion).
        3. Stop-Premature: called_stop=True but OSR = 0 (stopped before reaching goal).
        4. Stop-After-Reach: called_stop=True and OSR = 1 (stopped outside goal radius).
        5. Protocol-Violation: M_max exceeded or invalid tool protocol.
        6. Missed-Stop: OSR = 1 but agent never called stop before budget ran out.
        7. Budget-Exceeded: OSR = 0 and A_max simulator action budget exhausted.
        8. Unclassified: All other failure cases.

    Returns:
        Failure category string or None if successful.
    """
    if success:
        return None

    # Priority 1: Infrastructure failure
    if infrastructure_error:
        return "infrastructure_failure"

    # Priority 2: Safety-Deadlock (R_max reached)
    if deadlock_terminated:
        return "safety_deadlock"

    # Priority 3 & 4: Explicit Stop called
    if called_stop:
        if oracle_success:
            return "stop_after_reach"
        return "stop_premature"

    # Priority 5: Protocol-Violation
    if protocol_violation:
        return "protocol_violation"

    # Priority 6: Missed-Stop (Reached goal area during trajectory, but never called stop)
    if oracle_success:
        return "missed_stop"

    # Priority 7: Budget-Exceeded (Never reached goal and exhausted budget)
    if action_budget_exceeded:
        return "budget_exceeded"

    # Priority 8: Unclassified
    return "unclassified"
