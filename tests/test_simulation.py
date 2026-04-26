import tempfile
import unittest
from pathlib import Path

from agentic_uav.methods import AgenticMethod, RuleAdaptiveMethod, StaticPartitionMethod
from agentic_uav.rendering import Renderer
from agentic_uav.scenarios import build_demo_scenario
from agentic_uav.simulation import (
    CommunicationEvent,
    Message,
    ScenarioConfig,
    Sector,
    Simulation,
    UavConfig,
    WorldConfig,
)


def build_scenario(method_name: str) -> ScenarioConfig:
    world = WorldConfig(
        width=4,
        height=4,
        sectors=[
            Sector(cell=(3, 3), priority="urgent"),
            Sector(cell=(1, 1), blocked=True),
        ],
    )
    return ScenarioConfig(
        method_name=method_name,
        ticks=3,
        communication_range=1,
        sensing_radius=1,
        heartbeat_interval=3,
        urgent_message_ttl=2,
        world=world,
        uavs=[
            UavConfig(uav_id="u0", cell=(0, 0)),
            UavConfig(uav_id="u1", cell=(3, 0)),
        ],
        events=[],
        seed=7,
    )


class SwappableMethodsTest(unittest.TestCase):
    def test_methods_are_swappable_via_config(self) -> None:
        expected = {
            "static": StaticPartitionMethod,
            "rules": RuleAdaptiveMethod,
            "agentic": AgenticMethod,
        }
        for method_name, method_type in expected.items():
            with self.subTest(method_name=method_name):
                simulation = Simulation.from_config(build_scenario(method_name))
                self.assertIsInstance(simulation.method, method_type)

    def test_methods_share_same_simulation_backbone(self) -> None:
        for method_name in ("static", "rules", "agentic"):
            with self.subTest(method_name=method_name):
                simulation = Simulation.from_config(build_scenario(method_name))
                summary = simulation.run()
                self.assertEqual(summary["ticks_run"], 3)
                self.assertIn("coverage_ratio", summary)
                self.assertIn("messages_sent", summary)

    def test_static_vs_adaptive_behavior_differs_on_urgent_sector(self) -> None:
        static_summary = Simulation.from_config(build_scenario("static")).run()
        rule_summary = Simulation.from_config(build_scenario("rules")).run()
        agentic_summary = Simulation.from_config(build_scenario("agentic")).run()

        self.assertNotIn((3, 3), static_summary["urgent_targets"])
        self.assertIn((3, 3), rule_summary["urgent_targets"])
        self.assertIn((3, 3), agentic_summary["urgent_targets"])

    def test_agentic_method_uses_coverage_role_without_urgent_work(self) -> None:
        scenario = ScenarioConfig(
            method_name="agentic",
            ticks=1,
            communication_range=1,
            sensing_radius=1,
            heartbeat_interval=3,
            urgent_message_ttl=2,
            world=WorldConfig(width=3, height=3),
            uavs=[UavConfig(uav_id="u0", cell=(0, 0))],
            events=[],
            seed=3,
        )
        simulation = Simulation.from_config(scenario)

        simulation.step()

        self.assertEqual(simulation.uavs["u0"].role, "coverage")


class CommunicationModelTest(unittest.TestCase):
    def test_messages_travel_one_hop_per_tick_with_ttl(self) -> None:
        scenario = ScenarioConfig(
            method_name="static",
            ticks=2,
            communication_range=1,
            sensing_radius=1,
            heartbeat_interval=10,
            urgent_message_ttl=2,
            world=WorldConfig(width=3, height=1),
            uavs=[
                UavConfig(uav_id="u0", cell=(0, 0)),
                UavConfig(uav_id="u1", cell=(1, 0)),
                UavConfig(uav_id="u2", cell=(2, 0)),
            ],
            events=[],
            seed=1,
        )
        simulation = Simulation.from_config(scenario)

        simulation.send_messages(
            [
                Message(
                    sender_id="u0",
                    message_type="urgent_sector",
                    payload={"cell": (2, 0)},
                    ttl=2,
                    urgency="urgent",
                )
            ]
        )

        simulation.step()
        self.assertEqual(
            [message.sender_id for message in simulation.uavs["u1"].inbox],
            ["u0"],
        )
        self.assertEqual(simulation.uavs["u2"].inbox, [])

        simulation.step()
        self.assertEqual(
            [(message.sender_id, message.payload["cell"]) for message in simulation.uavs["u2"].inbox],
            [("u1", (2, 0))],
        )


class ScenarioBuilderTest(unittest.TestCase):
    def test_builder_applies_ui_parameters_and_deterministic_uav_positions(self) -> None:
        scenario = build_demo_scenario(
            method_name="rules",
            grid_size=6,
            uav_count=5,
            sensing_radius=2,
            communication_range=4,
            seed=11,
            ticks=9,
            heartbeat_interval=5,
            urgent_message_ttl=3,
        )
        repeat = build_demo_scenario(
            method_name="rules",
            grid_size=6,
            uav_count=5,
            sensing_radius=2,
            communication_range=4,
            seed=11,
            ticks=9,
            heartbeat_interval=5,
            urgent_message_ttl=3,
        )

        self.assertEqual(scenario.method_name, "rules")
        self.assertEqual((scenario.world.width, scenario.world.height), (6, 6))
        self.assertEqual(len(scenario.uavs), 5)
        self.assertEqual(scenario.sensing_radius, 2)
        self.assertEqual(scenario.communication_range, 4)
        self.assertEqual(scenario.seed, 11)
        self.assertEqual(scenario.ticks, 9)
        self.assertEqual(scenario.heartbeat_interval, 5)
        self.assertEqual(scenario.urgent_message_ttl, 3)
        self.assertEqual([uav.cell for uav in scenario.uavs], [uav.cell for uav in repeat.uavs])
        self.assertEqual(
            [uav.cell for uav in scenario.uavs],
            [(0, 0), (5, 0), (0, 5), (5, 5), (3, 0)],
        )


class SensingRadiusTest(unittest.TestCase):
    def test_sensing_radius_changes_covered_cells_after_one_step(self) -> None:
        small_radius = ScenarioConfig(
            method_name="static",
            ticks=1,
            communication_range=1,
            sensing_radius=0,
            heartbeat_interval=3,
            urgent_message_ttl=2,
            world=WorldConfig(width=5, height=5),
            uavs=[UavConfig(uav_id="u0", cell=(2, 2))],
            events=[],
            seed=1,
        )
        large_radius = ScenarioConfig(
            method_name="static",
            ticks=1,
            communication_range=1,
            sensing_radius=2,
            heartbeat_interval=3,
            urgent_message_ttl=2,
            world=WorldConfig(width=5, height=5),
            uavs=[UavConfig(uav_id="u0", cell=(2, 2))],
            events=[],
            seed=1,
        )

        small_simulation = Simulation.from_config(small_radius)
        large_simulation = Simulation.from_config(large_radius)
        small_simulation.step()
        large_simulation.step()

        small_covered = {
            cell for cell, sector in small_simulation.world.sectors.items() if sector.coverage
        }
        large_covered = {
            cell for cell, sector in large_simulation.world.sectors.items() if sector.coverage
        }

        self.assertEqual(len(small_covered), 1)
        self.assertGreater(len(large_covered), len(small_covered))


class RenderingTest(unittest.TestCase):
    def test_renderer_saves_snapshot(self) -> None:
        simulation = Simulation.from_config(build_scenario("agentic"))
        simulation.step()

        renderer = Renderer()
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "snapshot.png"
            renderer.render_snapshot(simulation.world, simulation.uavs, output_path)
            self.assertTrue(output_path.exists())


class EventHandlingTest(unittest.TestCase):
    def test_event_injector_blocks_sector_at_expected_tick(self) -> None:
        scenario = build_scenario("rules")
        scenario.events.append(
            CommunicationEvent(
                tick=1,
                event_type="block_sector",
                payload={"cell": (0, 1)},
            )
        )

        simulation = Simulation.from_config(scenario)
        simulation.step()
        self.assertFalse(simulation.world.sectors[(0, 1)].blocked)

        simulation.step()
        self.assertTrue(simulation.world.sectors[(0, 1)].blocked)


if __name__ == "__main__":
    unittest.main()
