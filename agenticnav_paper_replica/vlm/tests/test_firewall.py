"""Unit tests for the Privilege Sentinel Canary and Firewall Layer."""

import json
import pytest

from agenticnav_paper_replica.vlm.firewall import (
    PrivilegeFirewallBreach,
    assert_firewall_integrity,
)


def test_clean_payload_passes():
    clean_payload = json.dumps({
        "model": "qwen3.5:4b",
        "messages": [
            {"role": "system", "content": "You are a navigation robot."},
            {"role": "user", "content": "Walk into the kitchen and stop near the table."},
        ],
        "options": {"temperature": 0.0},
    })
    # Should pass without raising exception
    assert_firewall_integrity(clean_payload)


def test_geodesic_distance_blocked():
    leaked_payload = json.dumps({
        "messages": [
            {"role": "user", "content": "Your geodesic_distance to goal is 4.52m."},
        ]
    })
    with pytest.raises(PrivilegeFirewallBreach, match="Forbidden oracle key 'geodesic_distance'"):
        assert_firewall_integrity(leaked_payload)


def test_shortest_path_blocked():
    leaked_payload = json.dumps({
        "messages": [
            {"role": "user", "content": "Target shortest_path_distance is 8.12m."},
        ]
    })
    with pytest.raises(PrivilegeFirewallBreach, match="Forbidden oracle key 'shortest_path_distance'"):
        assert_firewall_integrity(leaked_payload)


def test_sentinel_canary_blocked():
    canary = "CANARY_TOKEN_XYZ_987654"
    leaked_payload = json.dumps({
        "messages": [
            {"role": "user", "content": f"Instruction: Follow path past landmark {canary}."},
        ]
    })
    with pytest.raises(PrivilegeFirewallBreach, match="Sentinel canary token 'CANARY_TOKEN_XYZ_987654'"):
        assert_firewall_integrity(leaked_payload, sentinel_canaries=[canary])


def test_oracle_numeric_metric_blocked():
    # If the true distance is 7.423m and appears in the prompt
    metrics = {"true_distance": 7.423}
    leaked_payload = json.dumps({
        "messages": [
            {"role": "user", "content": "You are 7.423 meters from goal."},
        ]
    })
    with pytest.raises(PrivilegeFirewallBreach, match="Oracle numeric metric 'true_distance=7.423' leaked"):
        assert_firewall_integrity(leaked_payload, current_oracle_metrics=metrics)


def test_positive_control_intentional_leak_trips_firewall():
    """Positive control: Proves that injecting oracle distance into prompt unconditionally aborts execution."""
    canary = "ORACLE_CANARY_TEST_BREACH"
    compromised_prompt = f"Target located at 2.57 m ≤ 3.00 m (within goal radius) {canary}"
    payload = json.dumps({
        "model": "qwen3.5:4b",
        "messages": [{"role": "user", "content": compromised_prompt}]
    })
    # Both canary token check and oracle key check should independently flag this
    with pytest.raises(PrivilegeFirewallBreach) as exc_info:
        assert_firewall_integrity(payload, sentinel_canaries=[canary])
    assert "Sentinel canary token" in str(exc_info.value)


def test_vllm_client_firewall_trips_on_leak():
    """Verify OpenAIVLLMNavClient triggers PrivilegeFirewallBreach before network dispatch."""
    from agenticnav_paper_replica.vlm.vllm_client import OpenAIVLLMNavClient
    client = OpenAIVLLMNavClient(base_url="http://fake-host:8000/v1")
    messages = [
        {"role": "user", "content": "Here is the oracle distance_to_goal: 3.5m"}
    ]
    with pytest.raises(PrivilegeFirewallBreach, match="Forbidden oracle key 'distance_to_goal'"):
        client(messages)



