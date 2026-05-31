from __future__ import annotations

from typing import Any

from agentic_uav.simulation import Simulation, coverage_ratio, manhattan


ROLE_COLORS = {
    "coverage": "#E8F7F1",
    "priority_responder": "#FF6B4A",
    "relay": "#35C7B8",
}

UAV_COLORS = [
    "#4F8EF7",
    "#F97316",
    "#22C55E",
    "#E11D48",
    "#8B5CF6",
    "#06B6D4",
    "#F59E0B",
    "#EC4899",
    "#84CC16",
    "#14B8A6",
    "#A855F7",
    "#64748B",
]

CELL_STYLES = {
    "uncovered": {"fill": "rgba(31, 41, 51, 0.24)", "label": "Uncovered"},
    "covered": {"fill": "rgba(61, 220, 151, 0.34)", "label": "Covered"},
    "urgent": {"fill": "rgba(231, 184, 74, 0.42)", "label": "Urgent"},
    "blocked": {"fill": "rgba(87, 60, 54, 0.42)", "label": "Blocked"},
}


def build_grid_portrayal(simulation: Simulation) -> dict[str, Any]:
    cells: dict[tuple[int, int], dict[str, Any]] = {}
    for cell, sector in simulation.world.sectors.items():
        state = _sector_state(sector)
        cells[cell] = {
            "state": state,
            "fill": CELL_STYLES[state]["fill"],
            "coverage": sector.coverage,
            "priority": sector.priority,
            "blocked": sector.blocked,
        }

    uav_colors = _uav_colors(simulation)
    paths = [
        {
            "id": uav_id,
            "color": uav_colors[uav_id],
            "points": _uav_path_points(simulation, uav_id),
            "active": simulation.uavs[uav_id].active,
        }
        for uav_id in simulation.uavs
    ]
    uavs = [
        {
            "id": uav.uav_id,
            "cell": uav.cell,
            "role": uav.role,
            "color": uav_colors[uav.uav_id],
            "active": uav.active,
            "status": "active" if uav.active else "dropped",
            "target_cell": uav.target_cell,
        }
        for uav in simulation.uavs.values()
    ]

    return {
        "width": simulation.world.width,
        "height": simulation.world.height,
        "cells": cells,
        "uavs": uavs,
        "paths": paths,
        "communication_links": _communication_links(simulation),
    }


def build_metric_series(simulation: Simulation) -> dict[str, list[Any]]:
    ticks: list[int] = []
    coverage: list[float] = []
    active_uavs: list[int] = []
    messages_sent: list[int] = []

    for record in simulation.metrics.records:
        ticks.append(record["tick"])
        coverage.append(record["coverage_ratio"])
        active_uavs.append(record["active_uavs"])
        messages_sent.append(record["messages_sent"])

    return {
        "tick": ticks,
        "coverage_ratio": coverage,
        "active_uavs": active_uavs,
        "messages_sent": messages_sent,
    }


def build_dashboard_state(simulation: Simulation) -> dict[str, Any]:
    active_uavs = sum(1 for uav in simulation.uavs.values() if uav.active)
    total_uavs = len(simulation.uavs)
    return {
        "tick": simulation.tick,
        "is_finished": simulation.is_finished,
        "is_solved": simulation.is_solved,
        "termination_reason": simulation.termination_reason,
        "coverage_ratio": coverage_ratio(simulation.world),
        "messages_sent": simulation.metrics.messages_sent,
        "active_uavs": active_uavs,
        "total_uavs": total_uavs,
        "dropped_uavs": total_uavs - active_uavs,
        "urgent_targets": list(simulation.metrics.urgent_targets),
        "urgent_target_count": len(simulation.metrics.urgent_targets),
    }


def build_event_timeline(simulation: Simulation) -> list[dict[str, Any]]:
    timeline: list[dict[str, Any]] = []
    for event in sorted(simulation.config.events, key=lambda item: item.tick):
        timeline.append(
            {
                "tick": event.tick,
                "event_type": event.event_type,
                "label": _event_label(event.event_type),
                "detail": _event_detail(event.payload),
                "tone": _event_tone(event.event_type),
                "state": _event_state(event.tick, simulation.tick),
            }
        )
    return timeline


def run_to_end(simulation: Simulation) -> None:
    while not simulation.is_finished:
        simulation.step()


def _uav_colors(simulation: Simulation) -> dict[str, str]:
    return {
        uav_id: UAV_COLORS[index % len(UAV_COLORS)]
        for index, uav_id in enumerate(simulation.uavs)
    }


def _uav_path_points(simulation: Simulation, uav_id: str) -> list[tuple[int, int]]:
    starts = {uav.uav_id: uav.cell for uav in simulation.config.uavs}
    points: list[tuple[int, int]] = []
    if uav_id in starts:
        points.append(starts[uav_id])

    for record in simulation.metrics.records:
        cell = record["uav_cells"].get(uav_id)
        if cell is not None:
            points.append(cell)

    current = simulation.uavs[uav_id].cell
    if not points or points[-1] != current:
        points.append(current)

    deduped: list[tuple[int, int]] = []
    for point in points:
        if not deduped or deduped[-1] != point:
            deduped.append(point)
    return deduped


def _communication_links(simulation: Simulation) -> list[dict[str, Any]]:
    active_uavs = [uav for uav in simulation.uavs.values() if uav.active]
    links: list[dict[str, Any]] = []
    for index, source in enumerate(active_uavs):
        for target in active_uavs[index + 1 :]:
            distance = manhattan(source.cell, target.cell)
            if distance <= simulation.config.communication_range:
                links.append(
                    {
                        "source_id": source.uav_id,
                        "target_id": target.uav_id,
                        "source_cell": source.cell,
                        "target_cell": target.cell,
                        "distance": distance,
                    }
                )
    return links


def _sector_state(sector: Any) -> str:
    if sector.blocked:
        return "blocked"
    if sector.priority == "urgent":
        return "urgent"
    if sector.coverage >= 1.0:
        return "covered"
    return "uncovered"


def _event_label(event_type: str) -> str:
    labels = {
        "block_sector": "Sector blocked",
        "dropout": "UAV dropout",
        "urgent_sector": "Urgent sector",
    }
    return labels.get(event_type, event_type.replace("_", " ").title())


def _event_detail(payload: dict[str, Any]) -> str:
    if "cell" in payload:
        cell = tuple(payload["cell"])
        return f"cell {cell[0]},{cell[1]}"
    if "uav_id" in payload:
        return str(payload["uav_id"])
    return ""


def _event_tone(event_type: str) -> str:
    tones = {
        "block_sector": "blocked",
        "dropout": "dropout",
        "urgent_sector": "urgent",
    }
    return tones.get(event_type, "neutral")


def _event_state(event_tick: int, current_tick: int) -> str:
    if event_tick < current_tick:
        return "past"
    if event_tick == current_tick:
        return "active"
    return "upcoming"
