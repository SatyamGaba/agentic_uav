from __future__ import annotations

import html
import math
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
from agentic_uav.scenarios import MISSION_TYPES, ScenarioParams, build_demo_scenario
from agentic_uav.simulation import Simulation
from agentic_uav.ui_config import LAST_CONFIG_PATH, METHODS as UI_METHODS, load_ui_params, save_ui_params


METHODS = list(UI_METHODS)
DEFAULT_PARAMS = ScenarioParams()
STARTUP_PARAMS = load_ui_params()
PAGE_STYLE = {"padding": "22px", "background": "#FFFFFF", "min-height": "100vh"}
TICK_HORIZON_MIN = 1
TICK_HORIZON_MAX = 10_000
TICK_HORIZON_SCALE_MAX = 1_000


def _tick_to_log_slider(ticks: int) -> int:
    bounded = min(max(ticks, TICK_HORIZON_MIN), TICK_HORIZON_MAX)
    ratio = math.log(bounded / TICK_HORIZON_MIN) / math.log(TICK_HORIZON_MAX / TICK_HORIZON_MIN)
    return round(ratio * TICK_HORIZON_SCALE_MAX)


def _log_slider_to_tick(scale: int) -> int:
    ratio = min(max(scale, 0), TICK_HORIZON_SCALE_MAX) / TICK_HORIZON_SCALE_MAX
    value = TICK_HORIZON_MIN * ((TICK_HORIZON_MAX / TICK_HORIZON_MIN) ** ratio)
    return int(round(value))


method_name = solara.reactive(STARTUP_PARAMS.method_name)
mission_type = solara.reactive(STARTUP_PARAMS.mission_type)
seed = solara.reactive(STARTUP_PARAMS.seed)
grid_size = solara.reactive(STARTUP_PARAMS.grid_size)
uav_count = solara.reactive(STARTUP_PARAMS.uav_count)
sensing_radius = solara.reactive(STARTUP_PARAMS.sensing_radius)
communication_range = solara.reactive(STARTUP_PARAMS.communication_range)
tick_horizon = solara.reactive(STARTUP_PARAMS.ticks)
tick_horizon_scale = solara.reactive(_tick_to_log_slider(STARTUP_PARAMS.ticks))
heartbeat_interval = solara.reactive(STARTUP_PARAMS.heartbeat_interval)
urgent_message_ttl = solara.reactive(STARTUP_PARAMS.urgent_message_ttl)
show_communication_links = solara.reactive(False)
version = solara.reactive(0)
simulation = solara.reactive(Simulation.from_config(build_demo_scenario(params=STARTUP_PARAMS)))
_has_loaded_persisted_config = False


@solara.component
def Page() -> None:
    solara.Title("Agentic UAV Swarm")
    solara.Style(_CSS)
    solara.use_effect(_load_persisted_config, [])
    solara.use_effect(_persist_current_config, [_config_dependency()])

    refresh_key = version.value
    state = build_dashboard_state(simulation.value)
    portrayal = build_grid_portrayal(simulation.value)
    timeline = build_event_timeline(simulation.value)

    with solara.Column(gap="18px", style=PAGE_STYLE):
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
            f"{html.escape(mission_type.value)} mission | {html.escape(method_name.value)} method | seed {seed.value} | "
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
        _MissionSetupCard()
        _RunControlCard(state)


@solara.component
def _MissionSetupCard() -> None:
    with solara.Card(title="Mission Setup", elevation=0, margin=0):
        solara.Select("Mission", values=MISSION_TYPES, value=mission_type, on_value=_set_mission_type)
        solara.Select("Method", values=METHODS, value=method_name, on_value=_set_method)
        solara.SliderInt("Grid size", value=grid_size, min=4, max=64, step=1, on_value=_set_grid_size)
        solara.SliderInt("UAV count", value=uav_count, min=1, max=12, step=1, on_value=_set_uav_count)
        solara.SliderInt("Sensing radius", value=sensing_radius, min=0, max=4, step=1, on_value=_set_sensing_radius)
        solara.SliderInt("Communication distance", value=communication_range, min=1, max=8, step=1, on_value=_set_communication_range)
        solara.InputInt("Seed", value=seed, on_value=_set_seed, continuous_update=True)
        _AdvancedControls()


@solara.component
def _AdvancedControls() -> None:
    with solara.Details(summary="Advanced", expand=False):
        solara.SliderInt(
            f"Max ticks ({tick_horizon.value:,})",
            value=tick_horizon_scale,
            min=0,
            max=TICK_HORIZON_SCALE_MAX,
            step=1,
            on_value=_set_tick_horizon_scale,
        )
        solara.SliderInt("Heartbeat interval", value=heartbeat_interval, min=1, max=12, step=1, on_value=_set_heartbeat_interval)
        solara.SliderInt("Urgent message TTL", value=urgent_message_ttl, min=1, max=8, step=1, on_value=_set_urgent_message_ttl)


@solara.component
def _RunControlCard(state: dict[str, object]) -> None:
    with solara.Card(title="Run Control", elevation=0, margin=0):
        with solara.Row(gap="8px", style={"flex-wrap": "wrap"}):
            solara.Button("Reset", on_click=_reset, color="primary", outlined=True)
            solara.Button("Next Step", on_click=_step_once, color="primary", disabled=state["is_finished"])
            solara.Button("End", on_click=_end, color="success", disabled=state["is_finished"])
        solara.HTML(tag="div", unsafe_innerHTML=_run_status_html(state), classes=["run-status"])


@solara.component
def _GridPanel(portrayal: dict[str, object], state: dict[str, object], refresh_key: int) -> None:
    with solara.Card(title="Operating Area", subtitle=f"Tick {state['tick']} of {simulation.value.config.ticks}", elevation=0, margin=0):
        solara.Checkbox(label="Show communication links", value=show_communication_links)
        solara.HTML(
            tag="div",
            unsafe_innerHTML=_grid_html(portrayal, refresh_key, show_communication_links.value),
            classes=["uav-grid-wrap"],
        )
        solara.HTML(tag="div", unsafe_innerHTML=_grid_footer_html(state), classes=["grid-footer"])


@solara.component
def _MetricsPanel(state: dict[str, object], timeline: list[dict[str, object]]) -> None:
    with solara.Card(title="Mission Telemetry", elevation=0, margin=0):
        _MetricSummary(state)
        _MetricChart()
        solara.HTML(tag="div", unsafe_innerHTML=_timeline_html(timeline), classes=["event-timeline"])


@solara.component
def _MetricSummary(state: dict[str, object]) -> None:
    with solara.Column(gap="10px"):
        with solara.Row(gap="10px"):
            _MetricCard("Coverage", f"{state['coverage_ratio']:.0%}", "green")
            _MetricCard("Active", f"{state['active_uavs']}/{state['total_uavs']}", "cyan")
        with solara.Row(gap="10px"):
            _MetricCard("Messages", str(state["messages_sent"]), "amber")
            _MetricCard("Urgent", str(state["urgent_target_count"]), "coral")


@solara.component
def _MetricChart() -> None:
    figure = _metric_figure(simulation.value)
    solara.FigureMatplotlib(figure, dependencies=[version.value])
    plt.close(figure)


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
    _persist_and_reset()


def _set_mission_type(value: str) -> None:
    mission_type.value = value
    _persist_and_reset()


def _set_seed(value: int | None) -> None:
    seed.value = 0 if value is None else value
    _persist_and_reset()


def _set_grid_size(value: int) -> None:
    grid_size.value = value
    _persist_and_reset()


def _set_uav_count(value: int) -> None:
    uav_count.value = value
    _persist_and_reset()


def _set_sensing_radius(value: int) -> None:
    sensing_radius.value = value
    _persist_and_reset()


def _set_communication_range(value: int) -> None:
    communication_range.value = value
    _persist_and_reset()


def _set_tick_horizon(value: int) -> None:
    tick_horizon.value = min(max(value, TICK_HORIZON_MIN), TICK_HORIZON_MAX)
    tick_horizon_scale.value = _tick_to_log_slider(tick_horizon.value)
    _persist_and_reset()


def _set_tick_horizon_scale(value: int) -> None:
    tick_horizon_scale.value = value
    tick_horizon.value = _log_slider_to_tick(value)
    _persist_and_reset()


def _set_heartbeat_interval(value: int) -> None:
    heartbeat_interval.value = value
    _persist_and_reset()


def _set_urgent_message_ttl(value: int) -> None:
    urgent_message_ttl.value = value
    _persist_and_reset()


def _persist_and_reset() -> None:
    save_ui_params(_scenario_params(), LAST_CONFIG_PATH)
    _reset()


def _load_persisted_config() -> None:
    global _has_loaded_persisted_config
    params = load_ui_params(last_path=LAST_CONFIG_PATH)
    if params != _scenario_params():
        _apply_scenario_params(params)
        _reset()
    _has_loaded_persisted_config = True


def _persist_current_config() -> None:
    if not _has_loaded_persisted_config:
        return
    params = _scenario_params()
    save_ui_params(params, LAST_CONFIG_PATH)
    if params != _simulation_params():
        _reset()


def _apply_scenario_params(params: ScenarioParams) -> None:
    method_name.value = params.method_name
    mission_type.value = params.mission_type
    grid_size.value = params.grid_size
    uav_count.value = params.uav_count
    sensing_radius.value = params.sensing_radius
    communication_range.value = params.communication_range
    seed.value = params.seed
    tick_horizon.value = params.ticks
    tick_horizon_scale.value = _tick_to_log_slider(params.ticks)
    heartbeat_interval.value = params.heartbeat_interval
    urgent_message_ttl.value = params.urgent_message_ttl


def _config_dependency() -> tuple[object, ...]:
    params = _scenario_params()
    return (
        params.method_name,
        params.mission_type,
        params.grid_size,
        params.uav_count,
        params.sensing_radius,
        params.communication_range,
        params.seed,
        params.ticks,
        params.heartbeat_interval,
        params.urgent_message_ttl,
    )


def _simulation_params() -> ScenarioParams:
    config = simulation.value.config
    return ScenarioParams(
        method_name=config.method_name,
        mission_type=config.mission_type,
        grid_size=config.world.width,
        uav_count=len(config.uavs),
        sensing_radius=config.sensing_radius,
        communication_range=config.communication_range,
        seed=config.seed,
        ticks=config.ticks,
        heartbeat_interval=config.heartbeat_interval,
        urgent_message_ttl=config.urgent_message_ttl,
    )


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
        mission_type=mission_type.value,
        grid_size=grid_size.value,
        uav_count=uav_count.value,
        sensing_radius=sensing_radius.value,
        communication_range=communication_range.value,
        seed=seed.value,
        ticks=tick_horizon.value,
        heartbeat_interval=heartbeat_interval.value,
        urgent_message_ttl=urgent_message_ttl.value,
    )


def _grid_html(
    portrayal: dict[str, object],
    refresh_key: int = 0,
    show_links: bool = False,
) -> str:
    width = portrayal["width"]
    height = portrayal["height"]
    cells = portrayal["cells"]
    uavs = portrayal["uavs"]
    paths = portrayal.get("paths", [])
    links = portrayal.get("communication_links", []) if show_links else []
    by_cell = _uavs_by_cell(uavs)
    target_cells = _target_cells(uavs)
    path_layer = _path_layer_html(paths, width, height)
    link_layer = _communication_layer_html(links, width, height)

    items = [
        _grid_cell_html((x, y), cells[(x, y)], by_cell.get((x, y), []), target_cells)
        for y in range(height)
        for x in range(width)
    ]
    return "<div class='uav-grid' data-refresh='{refresh_key}' style='grid-template-columns: repeat({width}, 1fr)'>{path_layer}{link_layer}{items}</div>".format(
        refresh_key=refresh_key,
        width=width,
        path_layer=path_layer,
        link_layer=link_layer,
        items="".join(items),
    )


def _path_layer_html(paths: list[dict[str, object]], width: int, height: int) -> str:
    if not paths:
        return ""

    path_items = "".join(_uav_path_html(path) for path in paths)
    if not path_items:
        return ""
    return (
        "<svg class='uav-path-layer' viewBox='0 0 {width} {height}' "
        "preserveAspectRatio='none' aria-hidden='true'>{path_items}</svg>"
    ).format(width=width, height=height, path_items=path_items)


def _uav_path_html(path: dict[str, object]) -> str:
    points = path.get("points", [])
    if not points:
        return ""

    color = html.escape(str(path["color"]))
    uav_id = html.escape(str(path["id"]))
    point_text = _path_points_attribute(points)
    start_x, start_y = _svg_point(points[0])
    end_x, end_y = _svg_point(points[-1])
    polyline = ""
    if len(points) > 1:
        polyline = (
            "<polyline class='uav-path-line' points='{points}' stroke='{color}'>"
            "<title>{uav_id} path</title>"
            "</polyline>"
        ).format(points=point_text, color=color, uav_id=uav_id)
    current = (
        "<circle class='uav-path-current' cx='{x:.3f}' cy='{y:.3f}' r='0.100' fill='{color}'>"
        "<title>{uav_id} current position</title>"
        "</circle>"
    ).format(x=end_x, y=end_y, color=color, uav_id=uav_id)
    start = (
        "<circle class='uav-path-start' cx='{x:.3f}' cy='{y:.3f}' r='0.075' fill='{color}'></circle>"
    ).format(x=start_x, y=start_y, color=color)
    return polyline + start + current


def _communication_layer_html(links: list[dict[str, object]], width: int, height: int) -> str:
    if not links:
        return ""

    link_items = "".join(_communication_link_html(link) for link in links)
    return (
        "<svg class='communication-link-layer' viewBox='0 0 {width} {height}' "
        "preserveAspectRatio='none' aria-hidden='true'>{link_items}</svg>"
    ).format(width=width, height=height, link_items=link_items)


def _communication_link_html(link: dict[str, object]) -> str:
    source_x, source_y = _svg_point(link["source_cell"])
    target_x, target_y = _svg_point(link["target_cell"])
    title = html.escape(f"{link['source_id']} to {link['target_id']} | d={link['distance']}")
    return (
        "<line class='communication-link-line' x1='{source_x:.3f}' y1='{source_y:.3f}' "
        "x2='{target_x:.3f}' y2='{target_y:.3f}'>"
        "<title>{title}</title>"
        "</line>"
    ).format(source_x=source_x, source_y=source_y, target_x=target_x, target_y=target_y, title=title)


def _path_points_attribute(points: list[tuple[int, int]]) -> str:
    return " ".join(f"{x:.3f},{y:.3f}" for x, y in (_svg_point(point) for point in points))


def _svg_point(point: tuple[int, int]) -> tuple[float, float]:
    return point[0] + 0.5, point[1] + 0.5


def _uavs_by_cell(uavs: list[dict[str, object]]) -> dict[tuple[int, int], list[dict[str, object]]]:
    by_cell: dict[tuple[int, int], list[dict[str, object]]] = {}
    for uav in uavs:
        by_cell.setdefault(uav["cell"], []).append(uav)
    return by_cell


def _target_cells(uavs: list[dict[str, object]]) -> set[tuple[int, int]]:
    return {uav["target_cell"] for uav in uavs if uav["target_cell"] is not None and uav["active"]}


def _grid_cell_html(
    cell: tuple[int, int],
    sector: dict[str, object],
    uavs: list[dict[str, object]],
    target_cells: set[tuple[int, int]],
) -> str:
    classes = ["grid-cell", str(sector["state"])]
    if cell in target_cells:
        classes.append("targeted")
    badges = "".join(_uav_badge(uav, index) for index, uav in enumerate(uavs))
    return "<div class='{classes}' style='background:{fill}'>{badges}</div>".format(
        classes=" ".join(classes),
        fill=sector["fill"],
        badges=badges,
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
    communication = "<span class='legend-item'><span class='communication-dot'></span>Communication link</span>"
    path = "<span class='legend-item'><span class='path-dot'></span>Path taken</span>"
    return cells + roles + target + communication + path


def _run_status_html(state: dict[str, object]) -> str:
    status = _run_status_label(state)
    return (
        "<div class='status-grid'>"
        f"<div><span>Tick</span><strong>{state['tick']}/{simulation.value.config.ticks}</strong></div>"
        f"<div><span>Status</span><strong>{status}</strong></div>"
        f"<div><span>UAVs</span><strong>{state['active_uavs']}/{state['total_uavs']}</strong></div>"
        f"<div><span>Msgs</span><strong>{state['messages_sent']}</strong></div>"
        "</div>"
    )


def _run_status_label(state: dict[str, object]) -> str:
    if state["termination_reason"] == "solved":
        return "Solved"
    if state["termination_reason"] == "max_ticks":
        return "Unsolved"
    return "Running"


def _timeline_html(timeline: list[dict[str, object]]) -> str:
    if not timeline:
        return "<div class='timeline-title'>Scenario Events</div><div class='timeline-empty'>No scheduled events</div>"
    return "<div class='timeline-title'>Scenario Events</div>" + "".join(
        _timeline_item_html(event) for event in timeline
    )


def _timeline_item_html(event: dict[str, object]) -> str:
    return (
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
  background: #FFFFFF;
  border: 1px solid #D8E1DE;
  border-radius: 8px;
  box-shadow: 0 12px 26px rgba(31, 41, 51, 0.08);
  position: relative;
}
.uav-path-layer {
  position: absolute;
  inset: 14px;
  width: calc(100% - 28px);
  height: calc(100% - 28px);
  pointer-events: none;
  z-index: 1;
  overflow: visible;
}
.uav-path-line {
  fill: none;
  stroke-width: 2.2;
  stroke-linecap: round;
  stroke-linejoin: round;
  opacity: 0.58;
  vector-effect: non-scaling-stroke;
  filter: drop-shadow(0 1px 3px rgba(31, 41, 51, 0.12));
}
.uav-path-start {
  opacity: 0.72;
  stroke: rgba(255, 255, 255, 0.92);
  stroke-width: 1.2;
  vector-effect: non-scaling-stroke;
}
.uav-path-current {
  opacity: 0.84;
  stroke: rgba(255, 255, 255, 0.96);
  stroke-width: 1.5;
  vector-effect: non-scaling-stroke;
}
.communication-link-layer {
  position: absolute;
  inset: 14px;
  width: calc(100% - 28px);
  height: calc(100% - 28px);
  pointer-events: none;
  z-index: 2;
  overflow: visible;
}
.communication-link-line {
  stroke: rgba(20, 184, 166, 0.64);
  stroke-width: 1.9;
  stroke-linecap: round;
  stroke-dasharray: 6 5;
  vector-effect: non-scaling-stroke;
  filter: drop-shadow(0 1px 3px rgba(20, 184, 166, 0.20));
}
.grid-cell {
  aspect-ratio: 1;
  position: relative;
  border-radius: 4px;
  border: 1px solid rgba(49, 66, 59, 0.16);
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
  border: 2px solid rgba(255, 107, 74, 0.86);
  border-radius: 50%;
  box-shadow: 0 0 14px rgba(255, 107, 74, 0.24);
  pointer-events: none;
  z-index: 2;
}
.uav-marker {
  position: absolute;
  left: 50%;
  top: 50%;
  transform: translate(calc(-50% + var(--uav-offset, 0px)), calc(-50% + var(--uav-offset, 0px)));
  width: 22px;
  height: 22px;
  display: block;
  background:
    radial-gradient(circle at 4px 4px, rgba(255,255,255,0.96) 0 2px, var(--uav-color) 2px 4px, transparent 4px),
    radial-gradient(circle at 18px 4px, rgba(255,255,255,0.96) 0 2px, var(--uav-color) 2px 4px, transparent 4px),
    radial-gradient(circle at 4px 18px, rgba(255,255,255,0.96) 0 2px, var(--uav-color) 2px 4px, transparent 4px),
    radial-gradient(circle at 18px 18px, rgba(255,255,255,0.96) 0 2px, var(--uav-color) 2px 4px, transparent 4px),
    radial-gradient(ellipse at center, rgba(255,255,255,0.98) 0 4px, var(--uav-color) 4px 7px, transparent 7px);
  color: var(--uav-color);
  filter: drop-shadow(0 7px 10px rgba(0, 0, 0, 0.42));
  z-index: 3;
}
.uav-marker::before {
  content: "";
  position: absolute;
  left: 5px;
  right: 5px;
  top: 10px;
  height: 3px;
  background: var(--uav-color);
  border-radius: 999px;
}
.uav-marker::after {
  content: "";
  position: absolute;
  top: 5px;
  bottom: 5px;
  left: 10px;
  width: 3px;
  background: var(--uav-color);
  border-radius: 999px;
}
.uav-marker.dropped {
  background: rgba(255, 255, 255, 0.5);
  color: #66756F;
  filter: grayscale(0.85) drop-shadow(0 5px 8px rgba(0, 0, 0, 0.32));
  opacity: 0.76;
}
.uav-marker.dropped::before,
.uav-marker.dropped::after {
  content: "";
  position: absolute;
  left: 5px;
  right: auto;
  top: 10px;
  bottom: auto;
  width: 12px;
  height: 3px;
  background: #FF6B4A;
  border-radius: 999px;
}
.uav-marker.dropped::before {
  transform: rotate(45deg);
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
.communication-dot {
  position: relative;
  background: transparent;
  border: 2px solid #35C7B8;
  border-radius: 50%;
}
.communication-dot::before,
.communication-dot::after {
  content: "";
  position: absolute;
  top: 5px;
  width: 4px;
  height: 4px;
  background: #35C7B8;
  border-radius: 50%;
}
.communication-dot::before {
  left: 1px;
}
.communication-dot::after {
  right: 1px;
}
.path-dot {
  position: relative;
  background: transparent;
  border: 2px solid #4F8EF7;
  border-radius: 50%;
}
.path-dot::after {
  content: "";
  position: absolute;
  left: 2px;
  right: 2px;
  top: 5px;
  height: 2px;
  background: #4F8EF7;
  border-radius: 999px;
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
