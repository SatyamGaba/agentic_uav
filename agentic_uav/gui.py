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
    build_event_timeline,
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
    timeline = build_event_timeline(simulation.value)

    with solara.Column(gap="18px", style={"padding": "22px", "background": "#FFFFFF", "min-height": "100vh"}):
        _MissionHeader(state)
        with solara.ColumnsResponsive(default=12, medium=[3, 6, 3], gutters=True, classes=["mission-layout"]):
            _Controls(state)
            _GridPanel(portrayal, state, refresh_key)
            _MetricsPanel(state, timeline)


@solara.component
def _MissionHeader(state: dict[str, object]) -> None:
    progress = int(float(state["coverage_ratio"]) * 100)
    solara.HTML(
        tag="div",
        unsafe_innerHTML=(
            "<div class='mission-header'>"
            "<div>"
            "<div class='eyebrow'>Decentralized UAV Evaluation</div>"
            "<h1>Agentic Swarm Mission Console</h1>"
            "<div class='header-subtitle'>"
            f"{html.escape(method_name.value)} method | seed {seed.value} | "
            f"{state['active_uavs']} of {state['total_uavs']} UAVs active"
            "</div>"
            "</div>"
            "<div class='mission-progress' title='Mission coverage'>"
            f"<span>{progress}%</span>"
            "<small>coverage</small>"
            "</div>"
            "</div>"
        ),
        classes=["mission-header-wrap"],
    )


@solara.component
def _Controls(state: dict[str, object]) -> None:
    with solara.Column(gap="14px"):
        with solara.Card(title="Mission Setup", elevation=0, margin=0):
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

        with solara.Card(title="Run Control", elevation=0, margin=0):
            with solara.Row(gap="8px", style={"flex-wrap": "wrap"}):
                solara.Button("Reset", on_click=_reset, color="primary", outlined=True)
                solara.Button("Next Step", on_click=_step_once, color="primary", disabled=state["is_finished"])
                solara.Button("End", on_click=_end, color="success", disabled=state["is_finished"])
            solara.HTML(tag="div", unsafe_innerHTML=_run_status_html(state), classes=["run-status"])


@solara.component
def _GridPanel(portrayal: dict[str, object], state: dict[str, object], refresh_key: int) -> None:
    with solara.Card(title="Operating Area", subtitle=f"Tick {state['tick']} of {simulation.value.config.ticks}", elevation=0, margin=0):
        solara.HTML(tag="div", unsafe_innerHTML=_grid_html(portrayal, refresh_key), classes=["uav-grid-wrap"])
        solara.HTML(tag="div", unsafe_innerHTML=_grid_footer_html(state), classes=["grid-footer"])


@solara.component
def _MetricsPanel(state: dict[str, object], timeline: list[dict[str, object]]) -> None:
    with solara.Card(title="Mission Telemetry", elevation=0, margin=0):
        with solara.Column(gap="10px"):
            with solara.Row(gap="10px"):
                _MetricCard("Coverage", f"{state['coverage_ratio']:.0%}", "green")
                _MetricCard("Active", f"{state['active_uavs']}/{state['total_uavs']}", "cyan")
            with solara.Row(gap="10px"):
                _MetricCard("Messages", str(state["messages_sent"]), "amber")
                _MetricCard("Urgent", str(state["urgent_target_count"]), "coral")
        figure = _metric_figure(simulation.value)
        solara.FigureMatplotlib(figure, dependencies=[version.value])
        plt.close(figure)
        solara.HTML(tag="div", unsafe_innerHTML=_timeline_html(timeline), classes=["event-timeline"])


@solara.component
def _MetricCard(label: str, value: str, tone: str = "green") -> None:
    solara.HTML(
        tag="div",
        unsafe_innerHTML=(
            f"<div class='metric-label'>{html.escape(label)}</div>"
            f"<div class='metric-value'>{html.escape(value)}</div>"
        ),
        classes=["metric-card", f"metric-{tone}"],
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
    target_cells = {uav["target_cell"] for uav in uavs if uav["target_cell"] is not None and uav["active"]}
    for uav in uavs:
        by_cell.setdefault(uav["cell"], []).append(uav)

    items: list[str] = []
    for y in range(height):
        for x in range(width):
            cell = (x, y)
            sector = cells[cell]
            classes = ["grid-cell", str(sector["state"])]
            if cell in target_cells:
                classes.append("targeted")
            badges = "".join(_uav_badge(uav, index) for index, uav in enumerate(by_cell.get(cell, [])))
            items.append(
                "<div class='{classes}' style='background:{fill}'>"
                "{badges}</div>".format(
                    classes=" ".join(classes),
                    fill=sector["fill"],
                    badges=badges,
                )
            )
    return "<div class='uav-grid' data-refresh='{refresh_key}' style='grid-template-columns: repeat({width}, 1fr)'>{items}</div>".format(
        refresh_key=refresh_key,
        width=width,
        items="".join(items),
    )


def _uav_badge(uav: dict[str, object], index: int = 0) -> str:
    status = "active" if uav["active"] else "dropped"
    target = uav.get("target_cell")
    target_text = ""
    if target is not None:
        target_text = f" -> {target[0]},{target[1]}"
    return (
        "<span class='uav-marker {status}' title='{title}' "
        "style='--uav-color:{color}; --uav-offset:{offset}px'></span>"
    ).format(
        status=status,
        title=html.escape(f"{uav['id']} | {uav['role']} | {status}{target_text}"),
        color=uav["color"],
        offset=index * 8,
    )


def _grid_footer_html(state: dict[str, object]) -> str:
    return (
        "<div class='grid-stats'>"
        f"<span>Coverage <strong>{state['coverage_ratio']:.0%}</strong></span>"
        f"<span>Dropped <strong>{state['dropped_uavs']}</strong></span>"
        f"<span>Messages <strong>{state['messages_sent']}</strong></span>"
        "</div>"
        f"<div class='legend'>{_legend_html()}</div>"
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
    target = "<span class='legend-item'><span class='target-dot'></span>Targeted sector</span>"
    return cells + roles + target


def _run_status_html(state: dict[str, object]) -> str:
    status = "Complete" if state["is_finished"] else "Running"
    return (
        "<div class='status-grid'>"
        f"<div><span>Tick</span><strong>{state['tick']}/{simulation.value.config.ticks}</strong></div>"
        f"<div><span>Status</span><strong>{status}</strong></div>"
        f"<div><span>UAVs</span><strong>{state['active_uavs']}/{state['total_uavs']}</strong></div>"
        f"<div><span>Msgs</span><strong>{state['messages_sent']}</strong></div>"
        "</div>"
    )


def _timeline_html(timeline: list[dict[str, object]]) -> str:
    if not timeline:
        return "<div class='timeline-title'>Scenario Events</div><div class='timeline-empty'>No scheduled events</div>"
    items = []
    for event in timeline:
        items.append(
            "<div class='timeline-item {tone} {state}'>"
            "<div class='timeline-marker'></div>"
            "<div>"
            "<div class='timeline-row'><strong>{label}</strong><span>t={tick}</span></div>"
            "<div class='timeline-detail'>{detail}</div>"
            "</div>"
            "</div>".format(
                tone=html.escape(str(event["tone"])),
                state=html.escape(str(event["state"])),
                label=html.escape(str(event["label"])),
                tick=html.escape(str(event["tick"])),
                detail=html.escape(str(event["detail"])),
            )
        )
    return "<div class='timeline-title'>Scenario Events</div>" + "".join(items)


def _metric_figure(sim: Simulation):
    series = build_metric_series(sim)
    figure, axis = plt.subplots(figsize=(5.4, 3.0))
    figure.patch.set_facecolor("#FFFFFF")
    axis.set_facecolor("#FFFFFF")
    ticks = series["tick"]
    if ticks:
        axis.plot(ticks, series["coverage_ratio"], color="#3DDC97", linewidth=2.8, label="Coverage")
        axis.plot(ticks, [value / max(1, len(sim.uavs)) for value in series["active_uavs"]], color="#35C7B8", linewidth=2.2, label="Active UAVs")
        max_messages = max(series["messages_sent"]) or 1
        axis.plot(ticks, [value / max_messages for value in series["messages_sent"]], color="#E7B84A", linewidth=2.2, label="Messages")
        for event in sim.config.events:
            if ticks[0] <= event.tick <= ticks[-1]:
                axis.axvline(event.tick, color="#FF6B4A", alpha=0.28, linewidth=1.4)
        legend = axis.legend(loc="lower right", frameon=True)
        legend.get_frame().set_facecolor("#FFFFFF")
        legend.get_frame().set_edgecolor("#D8E1DE")
        for text in legend.get_texts():
            text.set_color("#1F2933")
    else:
        axis.text(0.5, 0.5, "Next Step starts telemetry", ha="center", va="center", color="#66756F")
    axis.set_ylim(0, 1.05)
    axis.set_xlabel("Tick")
    axis.set_ylabel("Normalized value")
    axis.tick_params(colors="#40504A")
    axis.xaxis.label.set_color("#40504A")
    axis.yaxis.label.set_color("#40504A")
    for spine in axis.spines.values():
        spine.set_color("#D8E1DE")
    axis.grid(color="#D8E1DE", alpha=0.85, linewidth=0.8)
    figure.tight_layout()
    return figure


_CSS = """
.v-application {
  font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: #FFFFFF;
  color: #1F2933;
}
.v-card {
  background: #FFFFFF !important;
  color: #1F2933 !important;
  border: 1px solid #D8E1DE !important;
  border-radius: 8px !important;
  box-shadow: 0 12px 28px rgba(31, 41, 51, 0.08) !important;
}
.v-card-title {
  color: #1F2933 !important;
  font-weight: 800 !important;
  letter-spacing: 0 !important;
}
.v-card-subtitle {
  color: #66756F !important;
}
.v-label,
.v-input,
.v-select,
.v-field,
.v-slider,
.v-expansion-panel-title {
  color: #1F2933 !important;
}
.v-field,
.v-input__control {
  background: #F7FAF9;
}
.mission-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 18px;
  padding: 18px 20px;
  background: #FFFFFF;
  border: 1px solid #D8E1DE;
  border-radius: 8px;
  box-shadow: 0 12px 28px rgba(31, 41, 51, 0.08);
}
.mission-header h1 {
  margin: 4px 0 6px;
  color: #1F2933;
  font-size: 30px;
  line-height: 1.05;
  font-weight: 850;
  letter-spacing: 0;
}
.eyebrow {
  color: #35C7B8;
  font-size: 12px;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 0;
}
.header-subtitle {
  color: #66756F;
  font-size: 14px;
}
.mission-progress {
  width: 92px;
  height: 92px;
  border-radius: 50%;
  display: grid;
  place-items: center;
  background: radial-gradient(circle at center, #FFFFFF 48%, transparent 49%),
    conic-gradient(#3DDC97, #35C7B8, #E7B84A);
  border: 1px solid #D8E1DE;
  flex: 0 0 auto;
}
.mission-progress span,
.mission-progress small {
  grid-area: 1 / 1;
}
.mission-progress span {
  color: #1F2933;
  font-size: 23px;
  font-weight: 850;
  transform: translateY(-7px);
}
.mission-progress small {
  color: #66756F;
  font-size: 11px;
  transform: translateY(14px);
}
.mission-layout {
  align-items: stretch;
}
.status-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(74px, 1fr));
  gap: 8px;
  margin-top: 14px;
}
.status-grid div {
  min-width: 0;
  min-height: 54px;
  padding: 9px;
  background: #F7FAF9;
  border: 1px solid #D8E1DE;
  border-radius: 8px;
}
.status-grid span {
  display: block;
  color: #66756F;
  font-size: 10px;
  line-height: 1.1;
  white-space: nowrap;
}
.status-grid strong {
  display: block;
  color: #1F2933;
  font-size: clamp(13px, 1.1vw, 16px);
  line-height: 1.15;
  margin-top: 3px;
  overflow-wrap: anywhere;
}
.uav-grid-wrap {
  width: min(78vh, 100%);
  margin: 0 auto;
}
.uav-grid {
  display: grid;
  gap: 4px;
  padding: 14px;
  background: #0B0F0D;
  border: 1px solid #31423B;
  border-radius: 8px;
  box-shadow: 0 12px 26px rgba(31, 41, 51, 0.12);
}
.grid-cell {
  aspect-ratio: 1;
  position: relative;
  border-radius: 4px;
  border: 1px solid rgba(220, 231, 224, 0.12);
  overflow: hidden;
}
.grid-cell.covered {
  box-shadow: inset 0 0 18px rgba(61, 220, 151, 0.18);
}
.grid-cell.urgent {
  box-shadow: inset 0 0 0 2px rgba(255, 255, 255, 0.56), 0 0 18px rgba(231, 184, 74, 0.28);
}
.grid-cell.blocked {
  background-image: repeating-linear-gradient(135deg, rgba(255,255,255,0.13) 0 4px, transparent 4px 8px);
}
.grid-cell.targeted::after {
  content: "";
  position: absolute;
  inset: 17%;
  border: 2px solid rgba(255, 107, 74, 0.92);
  border-radius: 50%;
  box-shadow: 0 0 16px rgba(255, 107, 74, 0.34);
  pointer-events: none;
}
.uav-marker {
  position: absolute;
  left: 50%;
  top: 50%;
  transform: translate(calc(-50% + var(--uav-offset, 0px)), calc(-50% + var(--uav-offset, 0px)));
  width: 24px;
  height: 24px;
  border-radius: 50%;
  display: grid;
  place-items: center;
  background: rgba(255, 255, 255, 0.9);
  border: 2px solid var(--uav-color);
  color: var(--uav-color);
  box-shadow: 0 7px 16px rgba(0, 0, 0, 0.38);
  z-index: 2;
}
.uav-marker::before {
  content: "";
  width: 0;
  height: 0;
  border-left: 6px solid transparent;
  border-right: 6px solid transparent;
  border-bottom: 13px solid var(--uav-color);
  transform: translateY(-1px);
}
.uav-marker.dropped {
  background: rgba(255, 255, 255, 0.74);
  border-color: #66756F;
  color: #66756F;
  filter: grayscale(0.7);
}
.uav-marker.dropped::before,
.uav-marker.dropped::after {
  content: "";
  position: absolute;
  width: 15px;
  height: 3px;
  background: #FF6B4A;
  border-radius: 999px;
}
.uav-marker.dropped::before {
  transform: rotate(45deg);
  border: 0;
}
.uav-marker.dropped::after {
  transform: rotate(-45deg);
}
.grid-footer {
  margin-top: 16px;
}
.grid-stats {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-bottom: 12px;
}
.grid-stats span {
  padding: 8px 10px;
  background: #F7FAF9;
  border: 1px solid #D8E1DE;
  border-radius: 8px;
  color: #66756F;
  font-size: 12px;
}
.grid-stats strong {
  color: #1F2933;
}
.legend {
  display: flex;
  flex-wrap: wrap;
  gap: 10px 14px;
}
.legend-item {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  color: #40504A;
  font-size: 12px;
}
.legend-item span:first-child,
.role-dot,
.target-dot {
  width: 14px;
  height: 14px;
  display: inline-block;
  border-radius: 3px;
}
.role-dot {
  border-radius: 50%;
  border: 1px solid #0B0F0D;
}
.role-dot.coverage { background: #E8F7F1; }
.role-dot.priority { background: #FF6B4A; }
.role-dot.relay { background: #35C7B8; }
.target-dot {
  border-radius: 50%;
  border: 2px solid #FF6B4A;
}
.metric-card {
  flex: 1;
  min-height: 78px;
  padding: 13px;
  background: #F7FAF9;
  border: 1px solid #D8E1DE;
  border-radius: 8px;
  position: relative;
  overflow: hidden;
}
.metric-card::before {
  content: "";
  position: absolute;
  left: 0;
  top: 0;
  bottom: 0;
  width: 4px;
  background: #3DDC97;
}
.metric-cyan::before { background: #35C7B8; }
.metric-amber::before { background: #E7B84A; }
.metric-coral::before { background: #FF6B4A; }
.metric-green::before { background: #3DDC97; }
.metric-card div {
  position: relative;
}
.metric-label {
  color: #66756F;
  font-size: 12px;
}
.metric-value {
  color: #1F2933;
  font-size: 26px;
  font-weight: 800;
  line-height: 1.1;
  margin-top: 6px;
}
.event-timeline {
  margin-top: 12px;
}
.timeline-title {
  margin: 8px 0 10px;
  color: #1F2933;
  font-size: 14px;
  font-weight: 800;
}
.timeline-empty {
  color: #66756F;
  padding: 12px;
  background: #F7FAF9;
  border: 1px solid #D8E1DE;
  border-radius: 8px;
}
.timeline-item {
  display: grid;
  grid-template-columns: 14px 1fr;
  gap: 10px;
  padding: 11px 0;
  border-top: 1px solid #D8E1DE;
  opacity: 0.74;
}
.timeline-item.active,
.timeline-item.past {
  opacity: 1;
}
.timeline-marker {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  margin-top: 5px;
  background: #98A8A0;
  box-shadow: 0 0 0 4px rgba(152, 168, 160, 0.12);
}
.timeline-item.urgent .timeline-marker { background: #E7B84A; box-shadow: 0 0 0 4px rgba(231, 184, 74, 0.16); }
.timeline-item.dropout .timeline-marker { background: #FF6B4A; box-shadow: 0 0 0 4px rgba(255, 107, 74, 0.16); }
.timeline-item.blocked .timeline-marker { background: #8F6A5F; box-shadow: 0 0 0 4px rgba(143, 106, 95, 0.16); }
.timeline-row {
  display: flex;
  justify-content: space-between;
  gap: 8px;
  color: #1F2933;
  font-size: 13px;
}
.timeline-row span,
.timeline-detail {
  color: #66756F;
}
.timeline-detail {
  font-size: 12px;
  margin-top: 2px;
}
@media (max-width: 980px) {
  .mission-header {
    align-items: flex-start;
  }
  .mission-header h1 {
    font-size: 24px;
  }
  .mission-progress {
    width: 76px;
    height: 76px;
  }
  .status-grid {
    grid-template-columns: 1fr;
  }
  .uav-grid-wrap {
    width: min(92vw, 620px);
  }
}
"""
