"""Provider-agnostic LLM decision seam for the agentic swarm method.

This module defines the contract between the simulator and a (future) LLM:
an :class:`AgentDecider` turns a *local* serialized observation into an action
JSON dict. A real vendor adapter is a drop-in ``AgentDecider``; for tests and
smoke runs we ship a deterministic, dependency-free :class:`MockAgentDecider`.

No ``anthropic``/``openai`` dependency is added here. All decider output is
validated against the world (in-bounds, not blocked, role in the allowed set)
and falls back to the deterministic greedy heuristic on any failure.
"""

from __future__ import annotations

import os
from typing import Any, Protocol, runtime_checkable

from agentic_uav.communication import Message
from agentic_uav.models import Cell, Sector
from agentic_uav.planning import Action, MethodState
from agentic_uav.policy import greedy_decision

ALLOWED_ROLES = ("coverage", "priority_responder", "relay")


@runtime_checkable
class AgentDecider(Protocol):
    def decide(self, observation: dict[str, Any]) -> dict[str, Any]:
        ...


def build_decider(name: str | None = None) -> AgentDecider:
    """Select the agentic decider backend.

    The explicit ``name`` argument overrides the ``AGENTIC_UAV_DECIDER`` env
    var; the default (neither set) is the dependency-free ``MockAgentDecider``,
    so tests, CI, and the experiment sweeps stay offline unless a real backend
    is opted into. This factory is the ONLY place that branches on backend:
    ``method_name`` still selects the method, the env only selects which decider
    the ``agentic`` method drives.

    Real providers are imported lazily so importing this module never requires
    ``boto3`` and so no import cycle is introduced (``llm_providers`` imports
    back from this module).
    """

    choice = (name or os.environ.get("AGENTIC_UAV_DECIDER") or "mock").strip().lower()
    if choice == "mock":
        return MockAgentDecider()
    if choice == "bedrock":
        from agentic_uav.llm_providers import BedrockAgentDecider

        return BedrockAgentDecider()
    if choice == "qgenie":
        from agentic_uav.llm_providers import QGenieAgentDecider

        return QGenieAgentDecider()
    raise ValueError(
        f"Unknown decider backend: {choice!r}; choose from 'mock', 'bedrock', 'qgenie'"
    )


def serialize_observation(observation: dict[str, Any]) -> dict[str, Any]:
    """Project a UAV's observation into the JSON-only LOCAL view for the LLM.

    Emits only locally-knowable fields: self, sensed nearby sectors, the UAV's
    believed urgent cells, inbox messages, and peer intents. It NEVER emits
    global coverage or a global urgent list (observation symmetry).
    """

    uav = observation["self"]
    nearby = [
        {
            "cell": list(sector.cell),
            "coverage": sector.coverage,
            "priority": sector.priority,
            "blocked": sector.blocked,
        }
        for sector in observation.get("nearby", [])
        if isinstance(sector, Sector)
    ]
    messages = [
        {
            "sender_id": message.sender_id,
            "message_type": message.message_type,
            "payload": dict(message.payload),
        }
        for message in observation.get("messages", [])
        if isinstance(message, Message)
    ]
    return {
        "self": {
            "uav_id": uav.uav_id,
            "cell": list(uav.cell),
            "role": uav.role,
            "target_cell": list(uav.target_cell) if uav.target_cell is not None else None,
        },
        "nearby": nearby,
        "known_urgent": [list(cell) for cell in observation.get("known_urgent", [])],
        "messages": messages,
        "peer_intents": observation.get("peer_intents", []),
    }


def validate_action(action_json: dict[str, Any], simulation: object) -> dict[str, Any] | None:
    """Validate decider output; return a normalized dict or ``None`` on violation."""

    if not isinstance(action_json, dict):
        return None
    role = action_json.get("role")
    if role not in ALLOWED_ROLES:
        return None
    target = _coerce_cell(action_json.get("target_cell"))
    if target is not None and not _is_open_in_bounds(simulation, target):
        return None
    ranked = [
        cell
        for cell in (_coerce_cell(item) for item in action_json.get("ranked_targets", []))
        if cell is not None and _is_open_in_bounds(simulation, cell)
    ]
    return {"target_cell": target, "role": role, "ranked_targets": ranked}


def decide_proposal(
    decider: AgentDecider,
    simulation: object,
    observation: dict[str, Any],
    method_state: MethodState,
) -> tuple[Action, list[Cell]]:
    """Run serialize -> decider.decide -> validate; fall back to greedy on failure.

    Returns the chosen :class:`Action` together with the validated ranked
    alternative targets (used by in-tick negotiation when a target collides).
    """

    uav = observation["self"]
    try:
        raw = decider.decide(serialize_observation(observation))
        validated = validate_action(raw, simulation)
    except Exception:
        validated = None

    if validated is None:
        return greedy_decision(simulation, observation, method_state), []

    target = validated["target_cell"]
    role = validated["role"]
    messages: list[Message] = []
    if target is not None:
        messages.append(
            Message(
                sender_id=uav.uav_id,
                message_type="intent_summary",
                payload={"target_cell": target, "role": role},
                ttl=1,
                urgency="routine",
            )
        )
    action = Action(
        uav_id=uav.uav_id,
        action_type="switch_role",
        target_cell=target,
        new_role=role,
        messages=messages,
    )
    return action, validated["ranked_targets"]


def decide_action(
    decider: AgentDecider,
    simulation: object,
    observation: dict[str, Any],
    method_state: MethodState,
) -> Action:
    action, _ = decide_proposal(decider, simulation, observation, method_state)
    return action


class MockAgentDecider:
    """Deterministic, dependency-free decider that mirrors the greedy heuristic.

    Under this mock, ``agentic`` behaves like ``greedy`` while exercising the
    full serialize -> validate -> action pipeline. Swapping in a real LLM
    adapter is the only change needed for a genuine evaluation.
    """

    def decide(self, observation: dict[str, Any]) -> dict[str, Any]:
        self_cell = tuple(observation["self"]["cell"])
        # Negotiation prior: cells peers have already announced intent toward.
        # Prefer leaving those to the peer (deconfliction settles real ties).
        peer_cells = {
            tuple(entry["target_cell"])
            for entry in observation.get("peer_intents", [])
            if isinstance(entry, dict) and entry.get("target_cell") is not None
        }
        known_urgent = [tuple(cell) for cell in observation.get("known_urgent", [])]
        if known_urgent:
            ordered = _prefer_unclaimed(
                sorted(known_urgent, key=lambda cell: (_manhattan(self_cell, cell), cell)),
                peer_cells,
            )
            return {
                "target_cell": list(ordered[0]),
                "role": "priority_responder",
                "reason": "respond to nearest believed urgent sector (avoiding peer intents)",
                "ranked_targets": [list(cell) for cell in ordered],
            }

        uncovered = sorted(
            (
                tuple(sector["cell"])
                for sector in observation.get("nearby", [])
                if not sector["blocked"]
                and sector["coverage"] < 1.0
                and tuple(sector["cell"]) != self_cell
            ),
            key=lambda cell: (_manhattan(self_cell, cell), cell),
        )
        if uncovered:
            ordered = _prefer_unclaimed(uncovered, peer_cells)
            return {
                "target_cell": list(ordered[0]),
                "role": "coverage",
                "reason": "cover nearest sensed uncovered sector (avoiding peer intents)",
                "ranked_targets": [list(cell) for cell in ordered],
            }
        return {
            "target_cell": None,
            "role": "coverage",
            "reason": "no local lead; patrol",
            "ranked_targets": [],
        }


def _prefer_unclaimed(ordered: list[Cell], peer_cells: set[Cell]) -> list[Cell]:
    """Reorder candidates so cells no peer has announced intent toward come first.

    Relative order within each group is preserved, so distance ranking still
    decides among equally-(un)claimed cells. Never drops candidates, so a UAV
    still acts when every lead is contested (negotiation resolves the tie).
    """

    if not peer_cells:
        return ordered
    unclaimed = [cell for cell in ordered if cell not in peer_cells]
    claimed = [cell for cell in ordered if cell in peer_cells]
    return unclaimed + claimed


def _coerce_cell(value: Any) -> Cell | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        return (int(value[0]), int(value[1]))
    except (TypeError, ValueError):
        return None


def _is_open_in_bounds(simulation: object, cell: Cell) -> bool:
    if not simulation.world.in_bounds(cell):
        return False
    sector = simulation.world.sectors.get(cell)
    return sector is not None and not sector.blocked


def _manhattan(left: Cell, right: Cell) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


__all__ = [
    "ALLOWED_ROLES",
    "AgentDecider",
    "MockAgentDecider",
    "build_decider",
    "decide_action",
    "decide_proposal",
    "serialize_observation",
    "validate_action",
]
