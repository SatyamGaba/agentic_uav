from __future__ import annotations

from dataclasses import dataclass, field

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

