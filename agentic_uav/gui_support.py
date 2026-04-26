from __future__ import annotations

from typing import Any

from agentic_uav.simulation import Simulation, coverage_ratio


ROLE_COLORS = {
    "coverage": "#F8FAFC",
    "priority_responder": "#FF5A3D",
    "relay": "#18A999",
}

CELL_STYLES = {
    "uncovered": {"fill": "#1F2937", "label": "Uncovered"},
    "covered": {"fill": "#23A455", "label": "Covered"},
    "urgent": {"fill": "#F59E0B", "label": "Urgent"},
    "blocked": {"fill": "#5B3A32", "label": "Blocked"},
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
    return {
        "tick": simulation.tick,
        "is_finished": simulation.is_finished,
        "coverage_ratio": coverage_ratio(simulation.world),
        "messages_sent": simulation.metrics.messages_sent,
        "active_uavs": sum(1 for uav in simulation.uavs.values() if uav.active),
        "urgent_targets": list(simulation.metrics.urgent_targets),
    }


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
