"""Privilege Sentinel Canary & Firewall Assertion Layer.

Enforces zero leakage of privileged oracle data into VLM prompt payloads.
Asserts against full outgoing HTTP request bodies at the network client level.
"""

import json
from typing import Any, Dict, List, Optional, Sequence


FORBIDDEN_ORACLE_KEYS = [
    "geodesic_distance",
    "shortest_path_distance",
    "shortest_path_len",
    "distance_to_goal",
    "goal_position",
    "goal_coordinates",
    "oracle_path",
    "oracle_distance",
    "ground_truth_goal",
    "reference_path",
]


class PrivilegeFirewallBreach(RuntimeError):
    """Raised when an outgoing VLM request payload contains forbidden oracle data."""
    pass


def assert_firewall_integrity(
    payload_str: str,
    sentinel_canaries: Optional[Sequence[str]] = None,
    current_oracle_metrics: Optional[Dict[str, Any]] = None,
) -> None:
    """Scans raw serialized outgoing HTTP request payload for oracle leaks.

    Parameters:
      payload_str: Raw JSON or text payload sent to LLM/VLM backend.
      sentinel_canaries: Explicit canary strings (e.g. 'CANARY_GEOD_987654') registered for the episode.
      current_oracle_metrics: Dictionary of exact float values (e.g. geodesic_dist, goal_x, goal_z)
                              to verify they do not appear as substrings in the prompt.
    """
    lower_payload = payload_str.lower()

    # 1. Check forbidden oracle parameter names
    for key in FORBIDDEN_ORACLE_KEYS:
        if key in lower_payload:
            raise PrivilegeFirewallBreach(
                f"PRIVILEGE FIREWALL BREACH: Forbidden oracle key '{key}' found in outgoing VLM request payload!"
            )

    # 2. Check explicit sentinel canary strings
    if sentinel_canaries:
        for canary in sentinel_canaries:
            if canary in payload_str:
                raise PrivilegeFirewallBreach(
                    f"PRIVILEGE FIREWALL BREACH: Sentinel canary token '{canary}' detected in outgoing VLM request payload!"
                )

    # 3. Check exact numeric oracle metrics if provided
    if current_oracle_metrics:
        for metric_name, val in current_oracle_metrics.items():
            if val is None:
                continue
            if isinstance(val, (int, float)):
                # Match formatted floats like "7.423"
                s_val = f"{val:.3f}"
                if len(s_val) >= 4 and s_val in payload_str:
                    raise PrivilegeFirewallBreach(
                        f"PRIVILEGE FIREWALL BREACH: Oracle numeric metric '{metric_name}={s_val}' leaked into outgoing prompt payload!"
                    )

