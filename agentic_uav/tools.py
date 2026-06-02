from __future__ import annotations

from typing import Any

from agentic_uav.agent_core import AgentContext, PeerState
from agentic_uav.models import Cell, manhattan
from agentic_uav.planning import Action


def tool_find_best_task(context: AgentContext, previous_target: Cell | None = None) -> Cell | None:
    """Task allocation — select highest-value uncovered/urgent cell considering peers."""
    candidates = set()
    urgent_candidates = set()
    
    for cell, belief in context.world_model.known_sectors.items():
        if belief.blocked or belief.coverage >= 1.0:
            continue
        candidates.add(cell)
        if belief.priority == "urgent":
            urgent_candidates.add(cell)
            
    for cell in context.inbox_summary.urgent_alerts + context.inbox_summary.hazard_alerts:
        if cell not in context.world_model.known_sectors or (context.world_model.known_sectors[cell].coverage < 1.0 and not context.world_model.known_sectors[cell].blocked):
            urgent_candidates.add(cell)

    target_pool = urgent_candidates if urgent_candidates else candidates
    if not target_pool:
        return None
        
    my_id = context.self_state.uav_id
    current_cell = context.self_state.cell
    
    # Peer conflict avoidance based on physical distance & ID tie-breaker
    peer_targets = set()
    for peer in context.world_model.peer_states.values():
        if peer.target_cell is not None and peer.health != "dropped":
            peer_dist = manhattan(peer.cell, peer.target_cell)
            my_dist_to_peer_target = manhattan(current_cell, peer.target_cell)
            if peer_dist < my_dist_to_peer_target or (peer_dist == my_dist_to_peer_target and peer.uav_id < my_id):
                peer_targets.add(peer.target_cell)
    
    unclaimed = [c for c in target_pool if c not in peer_targets]
    if not unclaimed:
        unclaimed = list(target_pool)
        
    def distance_key(c: Cell) -> float:
        dist = float(manhattan(current_cell, c))
        if previous_target is not None and c == previous_target:
            dist -= 3.0  # Hysteresis bonus: prefer current target unless another is at least 3 steps closer
        return dist

    return min(unclaimed, key=lambda c: (distance_key(c), c))


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

