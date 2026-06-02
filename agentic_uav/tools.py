from __future__ import annotations

from typing import Any

from agentic_uav.agent_core import AgentContext, PeerState
from agentic_uav.models import Cell
from agentic_uav.planning import Action


def tool_find_best_task(context: AgentContext) -> Cell | None:
    """Task allocation — select highest-value uncovered/urgent cell considering peers."""
    return None


def tool_plan_route(context: AgentContext, target: Cell) -> list[Cell]:
    """Route planner — BFS/A* path avoiding blocked cells."""
    return []


def tool_assess_hazards(context: AgentContext) -> list[Cell]:
    """Hazard assessment — identify and rank known hazards."""
    return []


def tool_check_mission_progress(context: AgentContext) -> float:
    """Mission progress — estimated coverage ratio from local belief."""
    return 0.0


def tool_generate_coverage_plan(context: AgentContext, peers: dict[str, PeerState]) -> Cell | None:
    """Collaborative coverage — pick uncovered area avoiding peer targets."""
    return None


def tool_contingency_replan(context: AgentContext, trigger: str) -> Action | None:
    """Contingency — respond to energy depletion, comm loss, peer failure."""
    return None

