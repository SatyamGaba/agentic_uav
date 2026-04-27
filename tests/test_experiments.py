import json
import tempfile
import unittest
from pathlib import Path

from agentic_uav.experiments import (
    DISASTER_URGENT_DROPOUT,
    METHODS,
    SURVEY_DROPOUT,
    ExperimentSuiteConfig,
    build_disaster_urgent_dropout_scenario,
    build_survey_dropout_scenario,
    count_record_changes,
    dropout_count,
    dropped_uav_ids,
    normalized_coverage_auc,
    run_experiment_suite,
)


class ExperimentScenarioTest(unittest.TestCase):
    def test_dropout_selection_is_fractional_deterministic_and_method_paired(self) -> None:
        self.assertEqual(dropout_count(5, 0.0), 0)
        self.assertEqual(dropout_count(5, 0.1), 1)
        self.assertEqual(dropout_count(10, 0.25), 3)

        static = build_survey_dropout_scenario(
            "static",
            seed=7,
            swarm_size=10,
            dropout_fraction=0.25,
            ticks=100,
            grid_size=12,
        )
        agentic = build_survey_dropout_scenario(
            "agentic",
            seed=7,
            swarm_size=10,
            dropout_fraction=0.25,
            ticks=100,
            grid_size=12,
        )

        self.assertEqual(
            [event.payload["uav_id"] for event in static.events],
            [event.payload["uav_id"] for event in agentic.events],
        )
        self.assertEqual({event.tick for event in static.events}, {30})
        self.assertEqual([event.payload["uav_id"] for event in static.events], dropped_uav_ids(7, 10, 0.25))

    def test_disaster_scenario_has_urgent_task_and_dropout(self) -> None:
        scenario = build_disaster_urgent_dropout_scenario(
            "task_consideration",
            seed=3,
            swarm_size=5,
            ticks=100,
            grid_size=12,
        )

        self.assertEqual([event.event_type for event in scenario.events], ["urgent_sector", "dropout"])
        self.assertEqual(scenario.events[0].tick, 25)
        self.assertEqual(scenario.events[1].tick, 30)


class ExperimentMetricTest(unittest.TestCase):
    def test_coverage_auc_extends_solved_runs_to_deadline(self) -> None:
        records = [
            {"tick": 0, "coverage_ratio": 0.25},
            {"tick": 1, "coverage_ratio": 1.0},
        ]

        self.assertEqual(normalized_coverage_auc(records, 4, 1.0), 0.8125)

    def test_record_change_counter_tracks_roles_and_targets(self) -> None:
        records = [
            {"uav_roles": {"u0": "coverage"}, "uav_targets": {"u0": (0, 0)}},
            {"uav_roles": {"u0": "coverage"}, "uav_targets": {"u0": (1, 0)}},
            {"uav_roles": {"u0": "priority_responder"}, "uav_targets": {"u0": (1, 0)}},
        ]

        self.assertEqual(count_record_changes(records, "uav_roles"), 1)
        self.assertEqual(count_record_changes(records, "uav_targets"), 1)


class ExperimentSuiteTest(unittest.TestCase):
    def test_small_survey_suite_writes_trials_and_aggregates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_experiment_suite(
                ExperimentSuiteConfig(
                    output_dir=Path(tmp),
                    seed_count=1,
                    swarm_sizes=(5,),
                    dropout_fractions=(0.0, 0.25),
                    ticks=20,
                    grid_size=8,
                    family=SURVEY_DROPOUT,
                    make_plots=False,
                )
            )

            self.assertEqual(len(result.trial_rows), len(METHODS) * 2)
            self.assertEqual(len(result.aggregate_rows), len(METHODS) * 2)
            self.assertTrue((Path(tmp) / "trials.csv").exists())
            self.assertTrue((Path(tmp) / "aggregates.csv").exists())
            trials = json.loads((Path(tmp) / "trials.json").read_text(encoding="utf-8"))
            self.assertEqual(trials[0]["family"], SURVEY_DROPOUT)
            self.assertIn("coverage_auc", trials[0])
            self.assertIn("recovery_slope", trials[-1])

    def test_small_all_suite_can_generate_plots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_experiment_suite(
                ExperimentSuiteConfig(
                    output_dir=Path(tmp),
                    seed_count=1,
                    swarm_sizes=(5,),
                    dropout_fractions=(0.25,),
                    ticks=20,
                    grid_size=8,
                    make_plots=True,
                )
            )

            self.assertEqual(
                {row["family"] for row in result.trial_rows},
                {SURVEY_DROPOUT, DISASTER_URGENT_DROPOUT},
            )
            self.assertTrue(result.output_paths["representative_timeline"].exists())
            self.assertTrue(result.output_paths["survey_dropout_final_coverage_swarm_5"].exists())


if __name__ == "__main__":
    unittest.main()
