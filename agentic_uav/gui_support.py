from __future__ import annotations

from typing import Any

from agentic_uav.simulation import Simulation, coverage_ratio


ROLE_COLORS = {
    "coverage": "#E8F7F1",
    "priority_responder": "#FF6B4A",
    "relay": "#35C7B8",
}

CELL_STYLES = {
    "uncovered": {"fill": "#1B2120", "label": "Uncovered"},
    "covered": {"fill": "#2E8F67", "label": "Covered"},
    "urgent": {"fill": "#E7B84A", "label": "Urgent"},
    "blocked": {"fill": "#573C36", "label": "Blocked"},
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

    uavs = [
        {
            "id": uav.uav_id,
            "cell": uav.cell,
            "role": uav.role,
            "color": ROLE_COLORS.get(uav.role, "#CBD5E1"),
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
