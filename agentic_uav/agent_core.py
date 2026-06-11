from __future__ import annotations

from dataclasses import dataclass, field

from agentic_uav.communication import (
    Message,
    MSG_HEARTBEAT,
    MSG_TASK_COMMITMENT,
    MSG_INTENT_SUMMARY,
    MSG_HAZARD_ALERT,
    MSG_FAILURE_NOTICE,
    MSG_COVERAGE_UPDATE,
)
from agentic_uav.models import Cell, UavState


@dataclass
class SectorBelief:
    """What a single UAV believes about a sector."""
    cell: Cell
    coverage: float
    priority: str
    blocked: bool
    visibility: float
    last_observed_tick: int
    observed_by_self: bool


@dataclass
class PeerState:
    """Last-known state of a peer UAV, from messages."""
    uav_id: str
    cell: Cell
    role: str
    target_cell: Cell | None
    energy: float
    health: str
    last_heard_tick: int


@dataclass
class LocalWorldModel:
    """§4.5 — Per-UAV belief state, NOT the global world."""
    known_sectors: dict[Cell, SectorBelief] = field(default_factory=dict)
    peer_states: dict[str, PeerState] = field(default_factory=dict)
    known_hazards: set[Cell] = field(default_factory=set)
    task_status: dict[Cell, str] = field(default_factory=dict)
    mission_progress: float = 0.0
    last_updated: int = 0

    def update_from_messages(self, messages: list[Message], current_tick: int) -> None:
        self.last_updated = current_tick
        for msg in messages:
            if msg.message_type == MSG_HEARTBEAT:
                self.peer_states[msg.sender_id] = PeerState(
                    uav_id=msg.sender_id,
                    cell=tuple(msg.payload["cell"]),
                    role=msg.payload["role"],
                    target_cell=None,
                    energy=msg.payload["energy"],
                    health=msg.payload["health"],
                    last_heard_tick=current_tick,
                )
            elif msg.message_type == MSG_INTENT_SUMMARY or msg.message_type == MSG_TASK_COMMITMENT:
                if msg.sender_id in self.peer_states:
                    self.peer_states[msg.sender_id].target_cell = tuple(msg.payload["target_cell"])
            elif msg.message_type == MSG_HAZARD_ALERT:
                cell = tuple(msg.payload["cell"])
                self.known_hazards.add(cell)
                if cell not in self.known_sectors:
                    self.known_sectors[cell] = SectorBelief(
                        cell=cell,
                        coverage=0.0,
                        priority="urgent",
                        blocked=False,
                        visibility=1.0,
                        last_observed_tick=current_tick,
                        observed_by_self=False,
                    )
                else:
                    self.known_sectors[cell].priority = "urgent"
            elif msg.message_type == MSG_FAILURE_NOTICE:
                failed_id = msg.payload.get("uav_id")
                if failed_id and failed_id in self.peer_states:
                    self.peer_states[failed_id].health = "dropped"
            elif msg.message_type == MSG_COVERAGE_UPDATE:
                cell = tuple(msg.payload["cell"])
                coverage = msg.payload["coverage"]
                if cell in self.known_sectors:
                    self.known_sectors[cell].coverage = max(self.known_sectors[cell].coverage, coverage)
                    self.known_sectors[cell].last_observed_tick = current_tick
                    self.known_sectors[cell].observed_by_self = False
                if coverage >= 1.0:
                    self.known_hazards.discard(cell)


@dataclass
class HealthStatus:
    """§4.4 — Internal capability assessment."""
    energy: float
    health: str
    sensing_degraded: bool
    comm_degraded: bool
    availability: str


@dataclass
class InboxSummary:
    """§4.3 — Processed communication inputs."""
    peer_intents: dict[str, Cell] = field(default_factory=dict)
    peer_commitments: dict[str, Cell] = field(default_factory=dict)
    urgent_alerts: list[Cell] = field(default_factory=list)
    hazard_alerts: list[Cell] = field(default_factory=list)
    failure_notices: list[str] = field(default_factory=list)
    coverage_updates: list[tuple[Cell, float]] = field(default_factory=list)

    @classmethod
    def from_messages(cls, messages: list[Message]) -> InboxSummary:
        summary = cls()
        for msg in messages:
            if msg.message_type == MSG_INTENT_SUMMARY:
                summary.peer_intents[msg.sender_id] = tuple(msg.payload["target_cell"])
            elif msg.message_type == MSG_TASK_COMMITMENT:
                summary.peer_commitments[msg.sender_id] = tuple(msg.payload["target_cell"])
            elif msg.message_type == MSG_HAZARD_ALERT:
                cell = tuple(msg.payload["cell"])
                summary.hazard_alerts.append(cell)
                summary.urgent_alerts.append(cell)
            elif msg.message_type == "urgent_sector":
                summary.urgent_alerts.append(tuple(msg.payload["cell"]))
            elif msg.message_type == MSG_FAILURE_NOTICE:
                summary.failure_notices.append(msg.payload["uav_id"])
            elif msg.message_type == MSG_COVERAGE_UPDATE:
                summary.coverage_updates.append((tuple(msg.payload["cell"]), msg.payload["coverage"]))
        return summary


@dataclass
class MissionContext:
    """§4.1 — Mission-level parameters."""
    mission_type: str
    success_threshold: float
    max_ticks: int
    current_tick: int


@dataclass
class AgentContext:
    """Complete context package for the agent's think phase."""
    self_state: UavState
    world_model: LocalWorldModel
    health: HealthStatus
    inbox_summary: InboxSummary
    mission_config: MissionContext
    current_tick: int

