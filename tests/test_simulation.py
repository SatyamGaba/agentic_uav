import tempfile
import unittest
from pathlib import Path

from agentic_uav.methods import (
    AgenticMethod,
    GreedyMethod,
    RuleAdaptiveMethod,
    StaticPartitionMethod,
    TaskConsiderationMethod,
)
from agentic_uav.planning import BeliefMap
from agentic_uav.rendering import Renderer
from agentic_uav.scenarios import ScenarioParams, build_demo_scenario
from agentic_uav.simulation import (
    Cell,
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
            "task_consideration": TaskConsiderationMethod,
            "greedy": GreedyMethod,
            "agentic": AgenticMethod,
        }
        for method_name, method_type in expected.items():
            with self.subTest(method_name=method_name):
                simulation = Simulation.from_config(build_scenario(method_name))
                self.assertIsInstance(simulation.method, method_type)

    def test_methods_share_same_simulation_backbone(self) -> None:
        for method_name in ("static", "rules", "task_consideration", "greedy", "agentic"):
            with self.subTest(method_name=method_name):
                simulation = Simulation.from_config(build_scenario(method_name))
                summary = simulation.run()
                self.assertEqual(summary["ticks_run"], 3)
                self.assertIn("coverage_ratio", summary)
                self.assertIn("messages_sent", summary)
                self.assertIn("is_solved", summary)
                self.assertIn("termination_reason", summary)

    def test_greedy_responds_to_locally_sensed_urgent_sector(self) -> None:
        # With the global oracle removed, the UAV must be within sensing range
        # of the urgent cell to know about it. Placed at (3, 2) with radius 1,
        # u0 senses the urgent (3, 3) and the greedy method targets it; static
        # follows its serpentine partition and never targets the urgent cell.
        def scenario(method_name: str) -> ScenarioConfig:
            return ScenarioConfig(
                method_name=method_name,
                ticks=3,
                communication_range=1,
                sensing_radius=1,
                heartbeat_interval=3,
                urgent_message_ttl=2,
                world=WorldConfig(
                    width=4,
                    height=4,
                    sectors=[Sector(cell=(3, 3), priority="urgent")],
                ),
                uavs=[UavConfig(uav_id="u0", cell=(3, 2))],
                events=[],
                seed=7,
            )

        greedy_summary = Simulation.from_config(scenario("greedy")).run()
        static_summary = Simulation.from_config(scenario("static")).run()

        self.assertIn((3, 3), greedy_summary["urgent_targets"])
        self.assertNotIn((3, 3), static_summary["urgent_targets"])

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


class StaticPartitionBaselineTest(unittest.TestCase):
    def test_static_assignments_are_balanced_deterministic_and_skip_blocked_cells(self) -> None:
        scenario = ScenarioConfig(
            method_name="static",
            ticks=1,
            communication_range=1,
            sensing_radius=0,
            heartbeat_interval=3,
            urgent_message_ttl=2,
            world=WorldConfig(width=4, height=2, sectors=[Sector(cell=(1, 0), blocked=True)]),
            uavs=[
                UavConfig(uav_id="u0", cell=(0, 0)),
                UavConfig(uav_id="u1", cell=(0, 0)),
                UavConfig(uav_id="u2", cell=(0, 0)),
            ],
            events=[],
            seed=1,
        )

        simulation = Simulation.from_config(scenario)
        assignments = simulation.method_state.assignments
        assigned_cells = [cell for cells in assignments.values() for cell in cells]

        self.assertEqual(assignments["u0"], [(0, 0), (2, 0), (3, 0)])
        self.assertEqual(assignments["u1"], [(3, 1), (2, 1)])
        self.assertEqual(assignments["u2"], [(1, 1), (0, 1)])
        self.assertNotIn((1, 0), assigned_cells)
        self.assertEqual(len(assigned_cells), len(set(assigned_cells)))
        self.assertLessEqual(max(map(len, assignments.values())) - min(map(len, assignments.values())), 1)

    def test_static_route_progresses_through_assignment(self) -> None:
        scenario = ScenarioConfig(
            method_name="static",
            ticks=2,
            communication_range=1,
            sensing_radius=0,
            heartbeat_interval=3,
            urgent_message_ttl=2,
            world=WorldConfig(width=3, height=1),
            uavs=[UavConfig(uav_id="u0", cell=(0, 0))],
            events=[],
            seed=1,
        )
        simulation = Simulation.from_config(scenario)

        simulation.step()
        simulation.step()

        self.assertEqual(simulation.method_state.assignment_indices["u0"], 1)
        self.assertEqual(simulation.uavs["u0"].target_cell, (1, 0))
        self.assertEqual(simulation.uavs["u0"].cell, (1, 0))

    def test_static_does_not_reassign_for_urgent_or_dropout_events(self) -> None:
        scenario = ScenarioConfig(
            method_name="static",
            ticks=2,
            communication_range=1,
            sensing_radius=0,
            heartbeat_interval=3,
            urgent_message_ttl=2,
            world=WorldConfig(width=4, height=1, sectors=[Sector(cell=(3, 0), priority="urgent")]),
            uavs=[
                UavConfig(uav_id="u0", cell=(0, 0)),
                UavConfig(uav_id="u1", cell=(2, 0)),
            ],
            events=[
                CommunicationEvent(tick=0, event_type="dropout", payload={"uav_id": "u1"}),
                CommunicationEvent(tick=0, event_type="urgent_sector", payload={"cell": (3, 0)}),
            ],
            seed=1,
        )
        simulation = Simulation.from_config(scenario)
        original_assignments = {
            uav_id: list(cells) for uav_id, cells in simulation.method_state.assignments.items()
        }

        summary = simulation.run()

        self.assertEqual(simulation.method_state.assignments, original_assignments)
        self.assertFalse(simulation.uavs["u1"].active)
        self.assertNotIn((3, 0), summary["urgent_targets"])

    def test_static_with_zero_uavs_does_not_crash(self) -> None:
        scenario = ScenarioConfig(
            method_name="static",
            ticks=1,
            communication_range=1,
            sensing_radius=0,
            heartbeat_interval=3,
            urgent_message_ttl=2,
            world=WorldConfig(width=1, height=1),
            uavs=[],
            events=[],
            seed=1,
        )

        summary = Simulation.from_config(scenario).run()

        self.assertEqual(summary["ticks_run"], 1)
        self.assertEqual(summary["coverage_ratio"], 0.0)
        self.assertEqual(summary["termination_reason"], "max_ticks")


class RuleAdaptiveBaselineTest(unittest.TestCase):
    def test_rules_use_local_urgent_work_and_emit_urgent_message(self) -> None:
        scenario = ScenarioConfig(
            method_name="rules",
            ticks=1,
            communication_range=2,
            sensing_radius=1,
            heartbeat_interval=3,
            urgent_message_ttl=4,
            world=WorldConfig(width=3, height=1, sectors=[Sector(cell=(1, 0), priority="urgent")]),
            uavs=[UavConfig(uav_id="u0", cell=(0, 0))],
            events=[],
            seed=1,
        )
        simulation = Simulation.from_config(scenario)

        simulation.step()

        self.assertEqual(simulation.uavs["u0"].target_cell, (1, 0))
        self.assertEqual(simulation.uavs["u0"].role, "priority_responder")
        urgent_messages = [
            message for message in simulation.network.pending if message.message_type == "urgent_sector"
        ]
        self.assertEqual([(message.payload["cell"], message.ttl) for message in urgent_messages], [((1, 0), 4)])

    def test_rules_ignore_far_urgent_work_without_local_observation_or_message(self) -> None:
        scenario = ScenarioConfig(
            method_name="rules",
            ticks=1,
            communication_range=1,
            sensing_radius=1,
            heartbeat_interval=3,
            urgent_message_ttl=2,
            world=WorldConfig(width=5, height=5, sectors=[Sector(cell=(4, 4), priority="urgent")]),
            uavs=[UavConfig(uav_id="u0", cell=(0, 0))],
            events=[],
            seed=1,
        )
        simulation = Simulation.from_config(scenario)

        simulation.step()

        self.assertNotEqual(simulation.uavs["u0"].target_cell, (4, 4))
        self.assertNotIn((4, 4), simulation.metrics.urgent_targets)

    def test_rules_use_inbox_urgent_message(self) -> None:
        scenario = ScenarioConfig(
            method_name="rules",
            ticks=1,
            communication_range=1,
            sensing_radius=0,
            heartbeat_interval=3,
            urgent_message_ttl=2,
            world=WorldConfig(width=5, height=1, sectors=[Sector(cell=(4, 0), priority="urgent")]),
            uavs=[
                UavConfig(uav_id="u0", cell=(0, 0)),
                UavConfig(uav_id="u1", cell=(1, 0)),
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
                    payload={"cell": (4, 0)},
                    ttl=2,
                    urgency="urgent",
                )
            ]
        )

        simulation.step()

        self.assertEqual(simulation.uavs["u1"].target_cell, (4, 0))
        self.assertEqual(simulation.uavs["u1"].role, "priority_responder")

    def test_rules_emit_heartbeat_intent_messages(self) -> None:
        scenario = ScenarioConfig(
            method_name="rules",
            ticks=1,
            communication_range=1,
            sensing_radius=0,
            heartbeat_interval=1,
            urgent_message_ttl=2,
            world=WorldConfig(width=2, height=1),
            uavs=[UavConfig(uav_id="u0", cell=(0, 0))],
            events=[],
            seed=1,
        )
        simulation = Simulation.from_config(scenario)

        simulation.step()

        self.assertIn("intent_summary", {message.message_type for message in simulation.network.pending})


class TaskConsiderationBaselineTest(unittest.TestCase):
    def test_task_consideration_emits_commitment_messages(self) -> None:
        simulation = Simulation.from_config(
            ScenarioConfig(
                method_name="task_consideration",
                ticks=1,
                communication_range=1,
                sensing_radius=0,
                heartbeat_interval=3,
                urgent_message_ttl=2,
                world=WorldConfig(width=2, height=1),
                uavs=[UavConfig(uav_id="u0", cell=(0, 0))],
                events=[],
                seed=1,
            )
        )

        simulation.step()

        commitments = [
            message for message in simulation.network.pending if message.message_type == "task_commitment"
        ]
        self.assertEqual(len(commitments), 1)
        self.assertEqual(commitments[0].payload["target_cell"], simulation.uavs["u0"].target_cell)

    def test_task_consideration_prefers_urgent_candidates(self) -> None:
        simulation = Simulation.from_config(
            ScenarioConfig(
                method_name="task_consideration",
                ticks=1,
                communication_range=1,
                sensing_radius=1,
                heartbeat_interval=3,
                urgent_message_ttl=2,
                world=WorldConfig(width=3, height=1, sectors=[Sector(cell=(1, 0), priority="urgent")]),
                uavs=[UavConfig(uav_id="u0", cell=(0, 0))],
                events=[],
                seed=1,
            )
        )

        simulation.step()

        self.assertEqual(simulation.uavs["u0"].target_cell, (1, 0))
        self.assertEqual(simulation.uavs["u0"].role, "priority_responder")

    def test_task_consideration_uses_commitments_to_avoid_conflicts(self) -> None:
        simulation = Simulation.from_config(
            ScenarioConfig(
                method_name="task_consideration",
                ticks=1,
                communication_range=1,
                sensing_radius=1,
                heartbeat_interval=3,
                urgent_message_ttl=2,
                world=WorldConfig(width=3, height=1),
                uavs=[
                    UavConfig(uav_id="u0", cell=(1, 0)),
                    UavConfig(uav_id="u1", cell=(0, 0)),
                ],
                events=[],
                seed=1,
            )
        )
        simulation.method_state.task_commitments["u0"] = (1, 0)
        observations = simulation.observations.build(
            simulation.world, simulation.uavs, simulation.method_state.beliefs
        )

        actions = simulation.method.decide_tick(simulation, {"u1": observations["u1"]}, simulation.method_state)

        self.assertNotEqual(actions[0].target_cell, (1, 0))

    def test_task_consideration_keeps_current_target_with_hysteresis(self) -> None:
        simulation = Simulation.from_config(
            ScenarioConfig(
                method_name="task_consideration",
                ticks=1,
                communication_range=1,
                sensing_radius=1,
                heartbeat_interval=3,
                urgent_message_ttl=2,
                world=WorldConfig(width=3, height=1),
                uavs=[UavConfig(uav_id="u0", cell=(1, 0))],
                events=[],
                seed=1,
            )
        )
        simulation.method_state.targets_by_uav["u0"] = (2, 0)
        observations = simulation.observations.build(
            simulation.world, simulation.uavs, simulation.method_state.beliefs
        )

        actions = simulation.method.decide_tick(simulation, observations, simulation.method_state)

        self.assertEqual(actions[0].target_cell, (2, 0))


class CommunicationModelTest(unittest.TestCase):
    def test_messages_travel_one_hop_per_tick_with_ttl(self) -> None:
        scenario = ScenarioConfig(
            method_name="static",
            ticks=2,
            communication_range=1,
            sensing_radius=0,
            heartbeat_interval=10,
            urgent_message_ttl=2,
            world=WorldConfig(width=5, height=1),
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


class GoalBasedCompletionTest(unittest.TestCase):
    def test_run_stops_early_when_full_coverage_is_reached(self) -> None:
        scenario = ScenarioConfig(
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

        summary = Simulation.from_config(scenario).run()

        self.assertEqual(summary["ticks_run"], 1)
        self.assertEqual(summary["coverage_ratio"], 1.0)
        self.assertTrue(summary["is_solved"])
        self.assertEqual(summary["termination_reason"], "solved")

    def test_run_reports_unsolved_when_max_ticks_are_exhausted(self) -> None:
        scenario = ScenarioConfig(
            method_name="static",
            ticks=2,
            communication_range=1,
            sensing_radius=0,
            heartbeat_interval=3,
            urgent_message_ttl=2,
            world=WorldConfig(width=4, height=1),
            uavs=[UavConfig(uav_id="u0", cell=(0, 0))],
            events=[],
            seed=1,
        )

        summary = Simulation.from_config(scenario).run()

        self.assertEqual(summary["ticks_run"], 2)
        self.assertLess(summary["coverage_ratio"], 1.0)
        self.assertFalse(summary["is_solved"])
        self.assertEqual(summary["termination_reason"], "max_ticks")


class ScenarioBuilderTest(unittest.TestCase):
    def test_builder_defaults_to_twenty_five_by_twenty_five_grid(self) -> None:
        scenario = build_demo_scenario()

        self.assertEqual((scenario.world.width, scenario.world.height), (25, 25))
        self.assertEqual(scenario.ticks, 500)
        self.assertEqual(scenario.mission_type, "disaster_mapping")

    def test_default_disaster_mapping_keeps_demo_disruptions(self) -> None:
        scenario = build_demo_scenario()

        self.assertEqual(scenario.mission_type, "disaster_mapping")
        self.assertIn("urgent", {sector.priority for sector in scenario.world.sectors})
        self.assertTrue(any(sector.blocked for sector in scenario.world.sectors))
        self.assertEqual(
            [event.event_type for event in scenario.events],
            ["urgent_sector", "dropout"],
        )

    def test_survey_mission_has_clean_coverage_setup(self) -> None:
        scenario = build_demo_scenario(params=ScenarioParams(mission_type="survey"))

        self.assertEqual(scenario.mission_type, "survey")
        self.assertEqual(scenario.world.sectors, [])
        self.assertEqual(scenario.events, [])

    def test_builder_applies_ui_parameters_and_deterministic_uav_positions(self) -> None:
        scenario = build_demo_scenario(
            method_name="rules",
            mission_type="survey",
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
            mission_type="survey",
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
        self.assertEqual(scenario.mission_type, "survey")
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
            [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0)],
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


class ObservationSymmetryTest(unittest.TestCase):
    def _observation_for(self, method_name: str) -> dict:
        simulation = Simulation.from_config(build_scenario(method_name))
        return simulation.observations.build(
            simulation.world, simulation.uavs, simulation.method_state.beliefs
        )["u0"]

    def test_all_methods_receive_identical_observation_payloads(self) -> None:
        reference = self._observation_for("static")
        for method_name in ("rules", "task_consideration", "greedy", "agentic"):
            with self.subTest(method_name=method_name):
                observation = self._observation_for(method_name)
                self.assertEqual(set(observation), set(reference))
                self.assertEqual(observation["known_urgent"], reference["known_urgent"])
                self.assertEqual(observation["nearby"], reference["nearby"])
                self.assertEqual(observation["self"].cell, reference["self"].cell)

    def test_observation_has_no_global_urgent_cells_leak(self) -> None:
        observation = self._observation_for("agentic")

        self.assertNotIn("urgent_cells", observation)
        self.assertIn("known_urgent", observation)


class BeliefLocalityTest(unittest.TestCase):
    def test_uav_beyond_sensing_range_does_not_know_far_urgent_cell(self) -> None:
        # u0 at (0, 0), sensing radius 1, can never sense the urgent (4, 4) cell
        # and receives no message about it, so it must not target it.
        scenario = ScenarioConfig(
            method_name="greedy",
            ticks=1,
            communication_range=1,
            sensing_radius=1,
            heartbeat_interval=3,
            urgent_message_ttl=2,
            world=WorldConfig(width=5, height=5, sectors=[Sector(cell=(4, 4), priority="urgent")]),
            uavs=[UavConfig(uav_id="u0", cell=(0, 0))],
            events=[],
            seed=1,
        )
        simulation = Simulation.from_config(scenario)

        simulation.step()

        belief = simulation.method_state.beliefs["u0"]
        self.assertNotIn((4, 4), belief.known_urgent)
        self.assertNotEqual(simulation.uavs["u0"].target_cell, (4, 4))
        self.assertNotIn((4, 4), simulation.metrics.urgent_targets)


class DecisionDynamicSymmetryTest(unittest.TestCase):
    """A method's decision must not change based on global coverage of cells the
    deciding UAV has neither sensed nor heard about (observation symmetry). Each
    case marks a specific *unsensed* cell covered in ground truth and asserts the
    target is identical to the un-marked run; it then confirms the same cell in
    the UAV's own belief DOES change the decision (so the test can actually fail
    if a method reads global coverage)."""

    def _decide_target(
        self,
        method_name: str,
        *,
        commit_target: Cell | None = None,
        covered_in_world: Cell | None = None,
        covered_in_belief: Cell | None = None,
    ) -> object:
        scenario = ScenarioConfig(
            method_name=method_name,
            ticks=1,
            communication_range=1,
            sensing_radius=1,
            heartbeat_interval=3,
            urgent_message_ttl=2,
            world=WorldConfig(width=6, height=1),
            uavs=[UavConfig(uav_id="u0", cell=(0, 0))],
            events=[],
            seed=1,
        )
        simulation = Simulation.from_config(scenario)
        if commit_target is not None:
            simulation.method_state.targets_by_uav["u0"] = commit_target
        if covered_in_world is not None:
            simulation.world.sectors[covered_in_world].coverage = 1.0
        if covered_in_belief is not None:
            simulation.method_state.beliefs.setdefault("u0", BeliefMap()).covered.add(
                covered_in_belief
            )
        observations = simulation.observations.build(
            simulation.world, simulation.uavs, simulation.method_state.beliefs
        )
        actions = simulation.method.decide_tick(simulation, observations, simulation.method_state)
        return actions[0].target_cell

    def test_rules_decision_ignores_unsensed_global_coverage(self) -> None:
        # rules' hysteresis re-checks the committed target (5,0), which u0@(0,0)
        # (radius 1) cannot sense. Global coverage of it must not flip the choice.
        baseline = self._decide_target("rules", commit_target=(5, 0))
        with_global = self._decide_target(
            "rules", commit_target=(5, 0), covered_in_world=(5, 0)
        )
        self.assertEqual(baseline, with_global)
        # Sanity: believing (5,0) is covered DOES change the decision, so the
        # equality above is a real symmetry guarantee, not a no-op.
        with_belief = self._decide_target(
            "rules", commit_target=(5, 0), covered_in_belief=(5, 0)
        )
        self.assertNotEqual(baseline, with_belief)

    def test_task_consideration_decision_ignores_unsensed_global_coverage(self) -> None:
        # (2,0) is a neighbour of local candidate (1,0) but outside u0's sensing
        # range, so it feeds _uncovered_neighbor_count only if global coverage leaks.
        baseline = self._decide_target("task_consideration")
        with_global = self._decide_target("task_consideration", covered_in_world=(2, 0))
        self.assertEqual(baseline, with_global)
        with_belief = self._decide_target("task_consideration", covered_in_belief=(2, 0))
        self.assertNotEqual(baseline, with_belief)


class UnsensedUrgentSymmetryTest(unittest.TestCase):
    """Urgency is dynamic state: a method must not read the global urgent priority
    of a cell the deciding UAV has neither sensed nor heard about. When a blind
    patrol fallback lands a UAV's target on an unsensed urgent cell, the UAV may
    move there (layout is global) but must NOT act on its urgency."""

    def _run(self, method_name: str) -> Simulation:
        # u0 at (4,0), radius 1, senses only (3,0)/(4,0); urgent at (0,0) is unsensed.
        # Covering (3,0)/(4,0) forces the only candidate to be the blind-patrol
        # fallback, whose first open cell is (0,0) — the urgent one.
        scenario = ScenarioConfig(
            method_name=method_name,
            ticks=1,
            communication_range=1,
            sensing_radius=1,
            heartbeat_interval=1,
            urgent_message_ttl=3,
            world=WorldConfig(width=5, height=1, sectors=[Sector(cell=(0, 0), priority="urgent")]),
            uavs=[UavConfig(uav_id="u0", cell=(4, 0))],
            events=[],
            seed=1,
        )
        simulation = Simulation.from_config(scenario)
        simulation.world.sectors[(3, 0)].coverage = 1.0
        simulation.world.sectors[(4, 0)].coverage = 1.0
        simulation.step()
        return simulation

    def test_rules_does_not_broadcast_urgent_for_unsensed_patrol_target(self) -> None:
        simulation = self._run("rules")
        urgent_msgs = [
            m for m in simulation.network.pending if m.message_type == "urgent_sector"
        ]
        self.assertEqual(urgent_msgs, [])
        self.assertEqual(simulation.uavs["u0"].role, "coverage")
        self.assertNotIn((0, 0), simulation.method_state.beliefs["u0"].known_urgent)

    def test_task_consideration_does_not_flag_priority_for_unsensed_patrol_target(self) -> None:
        simulation = self._run("task_consideration")
        self.assertEqual(simulation.uavs["u0"].role, "coverage")
        self.assertNotIn((0, 0), simulation.method_state.beliefs["u0"].known_urgent)


class AgenticMethodTest(unittest.TestCase):
    def test_handle_event_marks_replan_pending_per_fired_event(self) -> None:
        scenario = ScenarioConfig(
            method_name="agentic",
            ticks=3,
            communication_range=1,
            sensing_radius=1,
            heartbeat_interval=3,
            urgent_message_ttl=2,
            world=WorldConfig(width=4, height=4),
            uavs=[UavConfig(uav_id="u0", cell=(0, 0))],
            events=[CommunicationEvent(tick=1, event_type="urgent_sector", payload={"cell": (2, 2)})],
            seed=1,
        )
        simulation = Simulation.from_config(scenario)
        self.assertTrue(simulation.method_state.replan_pending)  # mission-start plan

        simulation.step()  # tick 0: consumes mission-start replan
        self.assertFalse(simulation.method_state.replan_pending)

        simulation.step()  # tick 1: event fires -> handle_event sets replan_pending
        # The replan for the event was consumed in the same tick.
        self.assertEqual(simulation.method_state.replan_reason_by_uav.get("_last"), "urgent_sector")
        self.assertGreaterEqual(len(simulation.metrics.replans), 2)

    def test_handle_event_invoked_per_fired_event(self) -> None:
        captured: list[str] = []

        class RecordingMethod(AgenticMethod):
            def handle_event(self, event, method_state) -> None:  # type: ignore[override]
                captured.append(event.event_type)
                super().handle_event(event, method_state)

        scenario = ScenarioConfig(
            method_name="agentic",
            ticks=2,
            communication_range=1,
            sensing_radius=1,
            heartbeat_interval=3,
            urgent_message_ttl=2,
            world=WorldConfig(width=4, height=4),
            uavs=[UavConfig(uav_id="u0", cell=(0, 0))],
            events=[
                CommunicationEvent(tick=0, event_type="urgent_sector", payload={"cell": (2, 2)}),
                CommunicationEvent(tick=0, event_type="block_sector", payload={"cell": (3, 3)}),
            ],
            seed=1,
        )
        simulation = Simulation(scenario, RecordingMethod())

        simulation.step()

        self.assertEqual(captured, ["urgent_sector", "block_sector"])

    def test_agentic_negotiation_separates_colliding_proposals(self) -> None:
        # Both UAVs sense the same urgent cell and would propose it; in-tick
        # negotiation must split them onto different targets, closer/lower-id wins.
        scenario = ScenarioConfig(
            method_name="agentic",
            ticks=1,
            communication_range=4,
            sensing_radius=2,
            heartbeat_interval=3,
            urgent_message_ttl=2,
            world=WorldConfig(
                width=4,
                height=4,
                sectors=[Sector(cell=(2, 2), priority="urgent")],
            ),
            uavs=[
                UavConfig(uav_id="u0", cell=(2, 1)),
                UavConfig(uav_id="u1", cell=(0, 2)),
            ],
            seed=1,
        )
        simulation = Simulation.from_config(scenario)
        observations = simulation.observations.build(
            simulation.world, simulation.uavs, simulation.method_state.beliefs
        )

        actions = simulation.method.decide_tick(simulation, observations, simulation.method_state)
        targets = {action.uav_id: action.target_cell for action in actions}

        self.assertNotEqual(targets["u0"], targets["u1"])
        # u0 is the nearer UAV (distance 1 vs 2) so it wins the urgent cell.
        self.assertEqual(targets["u0"], (2, 2))

    def test_dropout_records_replan_and_plan_change_traces(self) -> None:
        from agentic_uav.experiments import build_disaster_urgent_dropout_scenario

        scenario = build_disaster_urgent_dropout_scenario(
            "agentic",
            seed=7,
            swarm_size=6,
            ticks=40,
            grid_size=10,
        )
        simulation = Simulation.from_config(scenario)
        simulation.run()

        self.assertTrue(simulation.metrics.replans)
        self.assertTrue(simulation.metrics.plan_changes)
        reasons = {replan["reason"] for replan in simulation.metrics.replans}
        self.assertIn("dropout", reasons)


if __name__ == "__main__":
    unittest.main()
