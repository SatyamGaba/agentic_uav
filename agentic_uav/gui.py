from __future__ import annotations

import html
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import solara

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_uav.gui_support import (
    CELL_STYLES,
    build_dashboard_state,
    build_grid_portrayal,
    build_metric_series,
    run_to_end,
)
from agentic_uav.scenarios import ScenarioParams, build_demo_scenario
from agentic_uav.simulation import Simulation


METHODS = ["static", "rules", "agentic"]

method_name = solara.reactive("agentic")
seed = solara.reactive(7)
grid_size = solara.reactive(8)
uav_count = solara.reactive(4)
sensing_radius = solara.reactive(1)
communication_range = solara.reactive(3)
tick_horizon = solara.reactive(12)
heartbeat_interval = solara.reactive(3)
urgent_message_ttl = solara.reactive(2)
version = solara.reactive(0)
simulation = solara.reactive(Simulation.from_config(build_demo_scenario(params=ScenarioParams())))


@solara.component
def Page() -> None:
    solara.Title("Agentic UAV Swarm")
    solara.Style(_CSS)

    refresh_key = version.value
    state = build_dashboard_state(simulation.value)
    portrayal = build_grid_portrayal(simulation.value)

    with solara.Column(gap="18px", style={"padding": "18px", "background": "#EEF2F6"}):
        solara.Markdown("# Agentic UAV Swarm")
        with solara.Columns([0.9, 1.6, 1.1], gutters=True):
            _Controls(state)
            _GridPanel(portrayal, state, refresh_key)
            _MetricsPanel(state)


@solara.component
def _Controls(state: dict[str, object]) -> None:
    with solara.Column(gap="12px"):
        with solara.Card(title="Configuration", elevation=2, margin=0):
            solara.Select("Method", values=METHODS, value=method_name, on_value=_set_method)
            solara.SliderInt("Grid size", value=grid_size, min=4, max=14, step=1, on_value=_set_grid_size)
            solara.SliderInt("UAV count", value=uav_count, min=1, max=12, step=1, on_value=_set_uav_count)
            solara.SliderInt("Sensing radius", value=sensing_radius, min=0, max=4, step=1, on_value=_set_sensing_radius)
            solara.SliderInt("Communication distance", value=communication_range, min=1, max=8, step=1, on_value=_set_communication_range)
            solara.InputInt("Seed", value=seed, on_value=_set_seed)

            with solara.Details(summary="Advanced", expand=False):
                solara.SliderInt("Tick horizon", value=tick_horizon, min=1, max=60, step=1, on_value=_set_tick_horizon)
                solara.SliderInt("Heartbeat interval", value=heartbeat_interval, min=1, max=12, step=1, on_value=_set_heartbeat_interval)
                solara.SliderInt("Urgent message TTL", value=urgent_message_ttl, min=1, max=8, step=1, on_value=_set_urgent_message_ttl)

        with solara.Card(title="Controls", elevation=2, margin=0):
            with solara.Row(gap="10px"):
                solara.Button("Reset", on_click=_reset, color="primary", outlined=True)
                solara.Button("Next Step", on_click=_step_once, color="primary", disabled=state["is_finished"])
                solara.Button("End", on_click=_end, color="success", disabled=state["is_finished"])
            solara.Markdown(
                f"""
                **Tick:** {state["tick"]} / {simulation.value.config.ticks}

                **Active UAVs:** {state["active_uavs"]}

                **Messages:** {state["messages_sent"]}
                """
            )


@solara.component
def _GridPanel(portrayal: dict[str, object], state: dict[str, object], refresh_key: int) -> None:
    with solara.Card(title="Mission State Space", subtitle=f"Coverage {state['coverage_ratio']:.0%}", elevation=2, margin=0):
        solara.HTML(tag="div", unsafe_innerHTML=_grid_html(portrayal, refresh_key), classes=["uav-grid-wrap"])
        solara.HTML(tag="div", unsafe_innerHTML=_legend_html(), classes=["legend"])


@solara.component
def _MetricsPanel(state: dict[str, object]) -> None:
    with solara.Card(title="Metrics", elevation=2, margin=0):
        with solara.Row(gap="10px"):
            _MetricCard("Coverage", f"{state['coverage_ratio']:.0%}")
            _MetricCard("Active", str(state["active_uavs"]))
            _MetricCard("Messages", str(state["messages_sent"]))
        figure = _metric_figure(simulation.value)
        solara.FigureMatplotlib(figure, dependencies=[version.value])
        plt.close(figure)


@solara.component
def _MetricCard(label: str, value: str) -> None:
    solara.HTML(
        tag="div",
        unsafe_innerHTML=f"<div class='metric-label'>{html.escape(label)}</div><div class='metric-value'>{html.escape(value)}</div>",
        classes=["metric-card"],
    )


def _set_method(value: str) -> None:
    method_name.value = value
    _reset()


def _set_seed(value: int | None) -> None:
    seed.value = 0 if value is None else value
    _reset()


def _set_grid_size(value: int) -> None:
    grid_size.value = value
    _reset()


def _set_uav_count(value: int) -> None:
    uav_count.value = value
    _reset()


def _set_sensing_radius(value: int) -> None:
    sensing_radius.value = value
    _reset()


def _set_communication_range(value: int) -> None:
    communication_range.value = value
    _reset()


def _set_tick_horizon(value: int) -> None:
    tick_horizon.value = value
    _reset()


def _set_heartbeat_interval(value: int) -> None:
    heartbeat_interval.value = value
    _reset()


def _set_urgent_message_ttl(value: int) -> None:
    urgent_message_ttl.value = value
    _reset()


def _reset() -> None:
    simulation.value = Simulation.from_config(build_demo_scenario(params=_scenario_params()))
    version.value += 1


def _step_once() -> None:
    simulation.value.step()
    version.value += 1


def _end() -> None:
    run_to_end(simulation.value)
    version.value += 1


def _scenario_params() -> ScenarioParams:
    return ScenarioParams(
        method_name=method_name.value,
        grid_size=grid_size.value,
        uav_count=uav_count.value,
        sensing_radius=sensing_radius.value,
        communication_range=communication_range.value,
        seed=seed.value,
        ticks=tick_horizon.value,
        heartbeat_interval=heartbeat_interval.value,
        urgent_message_ttl=urgent_message_ttl.value,
    )


def _grid_html(portrayal: dict[str, object], refresh_key: int = 0) -> str:
    width = portrayal["width"]
    height = portrayal["height"]
    cells = portrayal["cells"]
    uavs = portrayal["uavs"]
    by_cell: dict[tuple[int, int], list[dict[str, object]]] = {}
    for uav in uavs:
        if uav["active"]:
            by_cell.setdefault(uav["cell"], []).append(uav)

    items: list[str] = []
    for y in range(height):
        for x in range(width):
            cell = (x, y)
            sector = cells[cell]
            badges = "".join(_uav_badge(uav) for uav in by_cell.get(cell, []))
            items.append(
                "<div class='grid-cell {state}' style='background:{fill}'>"
                "<span class='cell-coord'>{coord}</span>{badges}</div>".format(
                    state=sector["state"],
                    fill=sector["fill"],
                    coord=f"{x},{y}",
                    badges=badges,
                )
            )
    return "<div class='uav-grid' data-refresh='{refresh_key}' style='grid-template-columns: repeat({width}, 1fr)'>{items}</div>".format(
        refresh_key=refresh_key,
        width=width,
        items="".join(items),
    )


def _uav_badge(uav: dict[str, object]) -> str:
    return (
        "<span class='uav-marker' title='{title}' style='background:{color}'>{label}</span>"
    ).format(
        title=html.escape(f"{uav['id']} | {uav['role']}"),
        color=uav["color"],
        label=html.escape(str(uav["id"]).replace("u", "")),
    )


def _legend_html() -> str:
    cells = "".join(
        f"<span class='legend-item'><span style='background:{style['fill']}'></span>{style['label']}</span>"
        for style in CELL_STYLES.values()
    )
    roles = (
        "<span class='legend-item'><span class='role-dot coverage'></span>Coverage UAV</span>"
        "<span class='legend-item'><span class='role-dot priority'></span>Priority responder</span>"
        "<span class='legend-item'><span class='role-dot relay'></span>Relay</span>"
    )
    return cells + roles


def _metric_figure(sim: Simulation):
    series = build_metric_series(sim)
    figure, axis = plt.subplots(figsize=(5.4, 3.2))
    ticks = series["tick"]
    if ticks:
        axis.plot(ticks, series["coverage_ratio"], color="#13795B", linewidth=2.6, label="Coverage")
        axis.plot(ticks, [value / max(1, len(sim.uavs)) for value in series["active_uavs"]], color="#2563EB", linewidth=2, label="Active UAVs")
        max_messages = max(series["messages_sent"]) or 1
        axis.plot(ticks, [value / max_messages for value in series["messages_sent"]], color="#E11D48", linewidth=2, label="Messages")
        axis.legend(loc="lower right")
    else:
        axis.text(0.5, 0.5, "Next Step starts metrics", ha="center", va="center", color="#64748B")
    axis.set_ylim(0, 1.05)
    axis.set_xlabel("Tick")
    axis.set_ylabel("Normalized value")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    return figure


_CSS = """
.v-application {
  font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.uav-grid-wrap {
  width: min(72vh, 100%);
  margin: 0 auto;
}
.uav-grid {
  display: grid;
  gap: 5px;
  padding: 12px;
  background: #0F172A;
  border: 1px solid rgba(15, 23, 42, 0.2);
  border-radius: 8px;
  box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.05);
}
.grid-cell {
  aspect-ratio: 1;
  position: relative;
  border-radius: 5px;
  border: 1px solid rgba(255, 255, 255, 0.25);
  overflow: hidden;
}
.grid-cell.urgent {
  box-shadow: inset 0 0 0 3px rgba(255, 255, 255, 0.55);
}
.grid-cell.blocked {
  background-image: repeating-linear-gradient(135deg, rgba(255,255,255,0.12) 0 4px, transparent 4px 8px);
}
.cell-coord {
  position: absolute;
  left: 4px;
  top: 2px;
  font-size: 9px;
  color: rgba(255, 255, 255, 0.52);
}
.uav-marker {
  position: absolute;
  left: 50%;
  top: 50%;
  transform: translate(-50%, -50%);
  width: 24px;
  height: 24px;
  border-radius: 50%;
  display: grid;
  place-items: center;
  border: 2px solid #0F172A;
  color: #0F172A;
  font-size: 11px;
  font-weight: 800;
  box-shadow: 0 5px 12px rgba(0, 0, 0, 0.25);
}
.legend {
  display: flex;
  flex-wrap: wrap;
  gap: 10px 14px;
  margin-top: 14px;
}
.legend-item {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  color: #334155;
  font-size: 13px;
}
.legend-item span:first-child,
.role-dot {
  width: 14px;
  height: 14px;
  display: inline-block;
  border-radius: 3px;
}
.role-dot {
  border-radius: 50%;
  border: 1px solid #0F172A;
}
.role-dot.coverage { background: #F8FAFC; }
.role-dot.priority { background: #FF5A3D; }
.role-dot.relay { background: #18A999; }
.metric-card {
  flex: 1;
  padding: 12px;
  background: #F8FAFC;
  border: 1px solid #E2E8F0;
  border-radius: 8px;
}
.metric-label {
  color: #64748B;
  font-size: 12px;
}
.metric-value {
  color: #0F172A;
  font-size: 24px;
  font-weight: 800;
}
"""
