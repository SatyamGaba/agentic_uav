import inspect
import unittest

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
    def test_gui_module_exposes_solara_page(self) -> None:
        self.assertTrue(callable(gui.Page))

    def test_step_once_advances_grid_refresh_key(self) -> None:
        gui.method_name.value = "agentic"
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

    def test_next_step_and_end_actions_advance_expected_ticks(self) -> None:
        gui.tick_horizon.value = 4
        gui._reset()

        gui._step_once()
        self.assertEqual(gui.simulation.value.tick, 1)

        gui._end()
        self.assertEqual(gui.simulation.value.tick, 4)
        self.assertTrue(gui.simulation.value.is_finished)

    def test_grid_panel_accepts_refresh_key(self) -> None:
        self.assertIn("refresh_key", inspect.signature(gui._GridPanel).parameters)

    def test_grid_portrayal_marks_sector_and_uav_layers(self) -> None:
        simulation = Simulation.from_config(build_gui_scenario())

        portrayal = build_grid_portrayal(simulation)

        self.assertEqual(portrayal["width"], 4)
        self.assertEqual(portrayal["height"], 4)
        self.assertEqual(portrayal["cells"][(1, 1)]["state"], "blocked")
        self.assertEqual(portrayal["cells"][(2, 2)]["state"], "urgent")
        self.assertEqual(portrayal["uavs"][0]["id"], "u0")
        self.assertEqual(portrayal["uavs"][0]["role"], "coverage")

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
        self.assertNotIn(">0</span>", html)

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
        self.assertIn("coverage_ratio", state)
        self.assertIn("messages_sent", state)

    def test_run_to_end_advances_until_horizon(self) -> None:
        simulation = Simulation.from_config(build_gui_scenario())

        run_to_end(simulation)

        self.assertEqual(simulation.tick, 3)
        self.assertTrue(simulation.is_finished)


if __name__ == "__main__":
    unittest.main()
