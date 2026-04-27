import inspect
import json
import tempfile
import unittest
from pathlib import Path

import agentic_uav.gui as gui
from agentic_uav.gui_support import (
    build_dashboard_state,
    build_event_timeline,
    build_grid_portrayal,
    build_metric_series,
    run_to_end,
)
from agentic_uav.simulation import ScenarioConfig, Sector, Simulation, UavConfig, WorldConfig
from agentic_uav.simulation import CommunicationEvent


def build_gui_scenario() -> ScenarioConfig:
    return ScenarioConfig(
        method_name="agentic",
        ticks=3,
        communication_range=2,
        sensing_radius=1,
        heartbeat_interval=3,
        urgent_message_ttl=2,
        world=WorldConfig(
            width=4,
            height=4,
            sectors=[
                Sector(cell=(1, 1), blocked=True),
                Sector(cell=(2, 2), priority="urgent"),
            ],
        ),
        uavs=[UavConfig(uav_id="u0", cell=(0, 0))],
        events=[],
        seed=4,
    )


class GuiSupportTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self._original_last_config_path = gui.LAST_CONFIG_PATH
        self._original_has_loaded_persisted_config = gui._has_loaded_persisted_config
        gui.LAST_CONFIG_PATH = Path(self._tempdir.name) / "last_config.json"
        gui._has_loaded_persisted_config = False

    def tearDown(self) -> None:
        gui.LAST_CONFIG_PATH = self._original_last_config_path
        gui._has_loaded_persisted_config = self._original_has_loaded_persisted_config
        self._tempdir.cleanup()

    def test_gui_module_exposes_solara_page(self) -> None:
        self.assertTrue(callable(gui.Page))

    def test_step_once_advances_grid_refresh_key(self) -> None:
        gui.method_name.value = "agentic"
        gui.mission_type.value = "disaster_mapping"
        gui.seed.value = 7
        gui._reset()
        before_tick = gui.simulation.value.tick
        before_version = gui.version.value

        gui._step_once()

        self.assertEqual(gui.simulation.value.tick, before_tick + 1)
        self.assertEqual(gui.version.value, before_version + 1)

    def test_config_change_resets_simulation_to_tick_zero(self) -> None:
        gui._reset()
        gui._step_once()
        before_version = gui.version.value

        gui._set_grid_size(6)

        self.assertEqual(gui.simulation.value.tick, 0)
        self.assertEqual(gui.simulation.value.config.world.width, 6)
        self.assertEqual(gui.simulation.value.config.world.height, 6)
        self.assertEqual(gui.version.value, before_version + 1)
        saved_config = json.loads(gui.LAST_CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertEqual(saved_config["grid_size"], 6)

    def test_page_reload_loads_persisted_config_into_controls(self) -> None:
        gui.save_ui_params(gui.ScenarioParams(grid_size=9, seed=31), gui.LAST_CONFIG_PATH)
        gui._apply_scenario_params(gui.ScenarioParams())
        gui._reset()

        gui._load_persisted_config()

        self.assertEqual(gui.grid_size.value, 9)
        self.assertEqual(gui.seed.value, 31)
        self.assertEqual(gui.simulation.value.config.world.width, 9)
        self.assertEqual(gui.simulation.value.config.seed, 31)

    def test_reactive_config_change_persists_and_resets_without_setter(self) -> None:
        gui._has_loaded_persisted_config = True
        gui._apply_scenario_params(gui.ScenarioParams())
        gui._reset()

        gui.grid_size.value = 7
        gui._persist_current_config()

        saved_config = json.loads(gui.LAST_CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertEqual(saved_config["grid_size"], 7)
        self.assertEqual(gui.simulation.value.config.world.width, 7)

    def test_scenario_params_include_mission_type(self) -> None:
        gui.mission_type.value = "survey"

        params = gui._scenario_params()

        self.assertEqual(params.mission_type, "survey")

    def test_scenario_params_default_to_five_hundred_max_ticks(self) -> None:
        gui.tick_horizon.value = gui.DEFAULT_PARAMS.ticks

        params = gui._scenario_params()

        self.assertEqual(params.ticks, 500)

    def test_log_tick_slider_maps_to_full_tick_range(self) -> None:
        self.assertEqual(gui._log_slider_to_tick(0), 1)
        self.assertEqual(gui._log_slider_to_tick(gui.TICK_HORIZON_SCALE_MAX), 10_000)
        self.assertEqual(gui._log_slider_to_tick(gui.TICK_HORIZON_SCALE_MAX // 2), 100)

    def test_tick_horizon_scale_updates_actual_max_ticks(self) -> None:
        before_version = gui.version.value

        gui._set_tick_horizon_scale(gui.TICK_HORIZON_SCALE_MAX)

        self.assertEqual(gui.tick_horizon.value, 10_000)
        self.assertEqual(gui.simulation.value.config.ticks, 10_000)
        self.assertEqual(gui.version.value, before_version + 1)

    def test_mission_type_change_resets_simulation_to_tick_zero(self) -> None:
        gui._set_mission_type("disaster_mapping")
        gui._step_once()
        before_version = gui.version.value

        gui._set_mission_type("survey")

        self.assertEqual(gui.simulation.value.tick, 0)
        self.assertEqual(gui.simulation.value.config.mission_type, "survey")
        self.assertEqual(gui.simulation.value.config.world.sectors, [])
        self.assertEqual(gui.simulation.value.config.events, [])
        self.assertEqual(gui.version.value, before_version + 1)

    def test_next_step_and_end_actions_advance_expected_ticks(self) -> None:
        gui.tick_horizon.value = 4
        gui._reset()

        gui._step_once()
        self.assertEqual(gui.simulation.value.tick, 1)

        gui._end()
        self.assertEqual(gui.simulation.value.tick, 4)
        self.assertTrue(gui.simulation.value.is_finished)

    def test_end_stops_when_goal_is_solved(self) -> None:
        gui.simulation.value = Simulation.from_config(
            ScenarioConfig(
                method_name="static",
                ticks=500,
                communication_range=1,
                sensing_radius=0,
                heartbeat_interval=3,
                urgent_message_ttl=2,
                world=WorldConfig(width=1, height=1),
                uavs=[UavConfig(uav_id="u0", cell=(0, 0))],
                events=[],
                seed=1,
            )
        )

        gui._end()

        self.assertEqual(gui.simulation.value.tick, 1)
        self.assertTrue(gui.simulation.value.is_solved)
        self.assertEqual(gui.simulation.value.termination_reason, "solved")

    def test_grid_panel_accepts_refresh_key(self) -> None:
        self.assertIn("refresh_key", inspect.signature(gui._GridPanel).parameters)

    def test_grid_portrayal_marks_sector_and_uav_layers(self) -> None:
        simulation = Simulation.from_config(build_gui_scenario())

        portrayal = build_grid_portrayal(simulation)

        self.assertEqual(portrayal["width"], 4)
        self.assertEqual(portrayal["height"], 4)
        self.assertEqual(portrayal["cells"][(1, 1)]["state"], "blocked")
        self.assertEqual(portrayal["cells"][(2, 2)]["state"], "urgent")
        self.assertTrue(portrayal["cells"][(0, 0)]["fill"].startswith("rgba("))
        self.assertEqual(portrayal["uavs"][0]["id"], "u0")
        self.assertEqual(portrayal["uavs"][0]["role"], "coverage")
        self.assertIn("paths", portrayal)
        self.assertIn("communication_links", portrayal)

    def test_grid_portrayal_tracks_uav_paths_from_initial_cell(self) -> None:
        simulation = Simulation.from_config(build_gui_scenario())
        initial_cell = simulation.uavs["u0"].cell
        simulation.step()
        simulation.step()

        portrayal = build_grid_portrayal(simulation)

        path = [path for path in portrayal["paths"] if path["id"] == "u0"][0]
        self.assertEqual(path["points"][0], initial_cell)
        self.assertEqual(path["points"][-1], simulation.uavs["u0"].cell)
        self.assertEqual(path["color"], portrayal["uavs"][0]["color"])
        self.assertTrue(
            all(left != right for left, right in zip(path["points"], path["points"][1:]))
        )

    def test_grid_portrayal_assigns_distinct_uav_identity_colors(self) -> None:
        scenario = build_gui_scenario()
        scenario.uavs.extend(
            [
                UavConfig(uav_id="u1", cell=(0, 0)),
                UavConfig(uav_id="u2", cell=(0, 0)),
            ]
        )
        simulation = Simulation.from_config(scenario)

        portrayal = build_grid_portrayal(simulation)
        colors = [uav["color"] for uav in portrayal["uavs"]]

        self.assertEqual(len(colors), len(set(colors)))

    def test_grid_portrayal_marks_uavs_within_communication_range(self) -> None:
        scenario = build_gui_scenario()
        scenario.communication_range = 2
        scenario.uavs = [
            UavConfig(uav_id="u0", cell=(0, 0)),
            UavConfig(uav_id="u1", cell=(2, 0)),
            UavConfig(uav_id="u2", cell=(3, 3)),
        ]
        simulation = Simulation.from_config(scenario)
        simulation.uavs["u2"].active = False

        portrayal = build_grid_portrayal(simulation)

        self.assertEqual(
            portrayal["communication_links"],
            [
                {
                    "source_id": "u0",
                    "target_id": "u1",
                    "source_cell": (0, 0),
                    "target_cell": (2, 0),
                    "distance": 2,
                }
            ],
        )

    def test_grid_portrayal_marks_dropped_uavs(self) -> None:
        scenario = build_gui_scenario()
        scenario.uavs.append(UavConfig(uav_id="u1", cell=(3, 3)))
        simulation = Simulation.from_config(scenario)
        simulation.uavs["u1"].active = False

        portrayal = build_grid_portrayal(simulation)

        dropped = [uav for uav in portrayal["uavs"] if uav["id"] == "u1"][0]
        self.assertFalse(dropped["active"])
        self.assertEqual(dropped["status"], "dropped")

    def test_grid_html_renders_uavs_as_icon_overlays(self) -> None:
        simulation = Simulation.from_config(build_gui_scenario())
        portrayal = build_grid_portrayal(simulation)

        html = gui._grid_html(portrayal)

        self.assertIn("grid-cell uncovered", html)
        self.assertIn("uav-marker active", html)
        self.assertIn("--uav-color:", html)
        self.assertIn("uav-path-layer", html)
        self.assertNotIn("communication-link-layer", html)
        self.assertIn("rgba(", html)
        self.assertNotIn(">0</span>", html)
        self.assertNotIn(">u0</span>", html)

    def test_grid_html_renders_uav_paths_as_svg_polylines(self) -> None:
        simulation = Simulation.from_config(build_gui_scenario())
        simulation.step()
        portrayal = build_grid_portrayal(simulation)

        html = gui._grid_html(portrayal)

        self.assertIn("<svg class='uav-path-layer'", html)
        self.assertIn("<polyline class='uav-path-line'", html)
        self.assertIn("u0 path", html)

    def test_grid_html_renders_communication_links_when_enabled(self) -> None:
        scenario = build_gui_scenario()
        scenario.communication_range = 2
        scenario.uavs.append(UavConfig(uav_id="u1", cell=(2, 0)))
        simulation = Simulation.from_config(scenario)
        portrayal = build_grid_portrayal(simulation)

        hidden_html = gui._grid_html(portrayal, show_links=False)
        visible_html = gui._grid_html(portrayal, show_links=True)

        self.assertNotIn("communication-link-layer", hidden_html)
        self.assertIn("<svg class='communication-link-layer'", visible_html)
        self.assertIn("<line class='communication-link-line'", visible_html)
        self.assertIn("u0 to u1", visible_html)

    def test_metric_series_tracks_coverage_messages_and_active_uavs(self) -> None:
        simulation = Simulation.from_config(build_gui_scenario())
        simulation.step()
        simulation.step()

        series = build_metric_series(simulation)

        self.assertEqual(series["tick"], [0, 1])
        self.assertEqual(len(series["coverage_ratio"]), 2)
        self.assertEqual(series["active_uavs"], [1, 1])
        self.assertEqual(len(series["messages_sent"]), 2)

    def test_metric_series_keeps_historical_active_uav_counts(self) -> None:
        scenario = build_gui_scenario()
        scenario.uavs.append(UavConfig(uav_id="u1", cell=(3, 3)))
        scenario.events.append(CommunicationEvent(tick=1, event_type="dropout", payload={"uav_id": "u1"}))
        simulation = Simulation.from_config(scenario)
        simulation.step()
        simulation.step()

        series = build_metric_series(simulation)

        self.assertEqual(series["active_uavs"], [2, 1])

    def test_event_timeline_describes_scheduled_events(self) -> None:
        scenario = build_gui_scenario()
        scenario.events.extend(
            [
                CommunicationEvent(tick=1, event_type="urgent_sector", payload={"cell": (3, 3)}),
                CommunicationEvent(tick=2, event_type="dropout", payload={"uav_id": "u0"}),
            ]
        )
        simulation = Simulation.from_config(scenario)

        timeline = build_event_timeline(simulation)

        self.assertEqual([event["tick"] for event in timeline], [1, 2])
        self.assertEqual(timeline[0]["label"], "Urgent sector")
        self.assertEqual(timeline[0]["detail"], "cell 3,3")
        self.assertEqual(timeline[0]["state"], "upcoming")

        simulation.step()
        timeline = build_event_timeline(simulation)
        self.assertEqual(timeline[0]["state"], "active")

    def test_dashboard_state_is_small_summary_for_ui(self) -> None:
        simulation = Simulation.from_config(build_gui_scenario())
        simulation.step()

        state = build_dashboard_state(simulation)

        self.assertEqual(state["tick"], 1)
        self.assertFalse(state["is_finished"])
        self.assertFalse(state["is_solved"])
        self.assertEqual(state["termination_reason"], "running")
        self.assertIn("coverage_ratio", state)
        self.assertIn("messages_sent", state)

    def test_run_status_html_renders_solved_and_unsolved_states(self) -> None:
        base_state = {
            "tick": 1,
            "active_uavs": 1,
            "total_uavs": 1,
            "messages_sent": 0,
        }

        solved = gui._run_status_html({**base_state, "termination_reason": "solved"})
        unsolved = gui._run_status_html({**base_state, "termination_reason": "max_ticks"})

        self.assertIn("<strong>Solved</strong>", solved)
        self.assertIn("<strong>Unsolved</strong>", unsolved)

    def test_run_to_end_advances_until_horizon(self) -> None:
        simulation = Simulation.from_config(build_gui_scenario())

        run_to_end(simulation)

        self.assertEqual(simulation.tick, 3)
        self.assertTrue(simulation.is_finished)


if __name__ == "__main__":
    unittest.main()
