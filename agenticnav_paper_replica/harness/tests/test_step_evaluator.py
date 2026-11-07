import unittest
import numpy as np
from pathlib import Path
import tempfile
import shutil

from agenticnav_paper_replica.harness.step_evaluator import StepEvaluator, StepEvaluation


class TestStepEvaluator(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_step_evaluator_diagnostics(self):
        evaluator = StepEvaluator(
            mode="verbose",
            out_dir=self.tmp_dir,
            save_intermediate_bev=True,
            interactive_pause=False,
        )

        dummy_depths = {
            0: np.full((120, 160), 2.5, dtype=np.float32),
            1: np.full((120, 160), 1.6, dtype=np.float32),
            2: np.full((120, 160), 3.0, dtype=np.float32),
            3: np.full((120, 160), 1.6, dtype=np.float32),
        }
        dummy_bev = np.full((100, 100, 3), 128, dtype=np.uint8)

        turn_log = [
            {
                "role": "assistant",
                "thought": (
                    "Milestone Breakdown:\n"
                    "1. Exit the room: In Progress\n"
                    "2. Turn left into corridor: Not started\n"
                    "Active Milestone: Milestone 1\n"
                    "Moving forward towards the exit doorway."
                ),
            }
        ]

        action_payload = {"view": 0, "u": 0.5, "v": 0.55, "reason": "Advance forward"}

        res = evaluator.evaluate_step(
            step_idx=1,
            node_idx=1,
            instruction="Exit the room and turn left. Walk forward and stop near the table.",
            pre_pose=(0.0, 0.0, 0.0),
            post_pose=(0.0, -2.0, 0.0),
            pre_goal_dist_m=8.5,
            post_goal_dist_m=6.5,
            accumulated_dist_m=2.0,
            depth_views=dummy_depths,
            bev_image=dummy_bev,
            occupancy_grid=None,
            turn_log=turn_log,
            action_type="move_to",
            action_payload=action_payload,
            actual_moved_m=2.0,
            collided=False,
        )

        self.assertTrue(res.movement_consistent)
        self.assertTrue(res.parsing_valid)
        self.assertEqual(res.goal_progress_m, 2.0)
        self.assertEqual(res.reasoning_quality, "high")
        self.assertFalse(res.premature_stop_detected)
        self.assertEqual(res.view_clearances_m["Front (0°)"], 2.5)
        self.assertEqual(res.view_clearances_m["Right (-90°)"], 1.6)

        # Verify audit file export
        evaluator.save_episode_audit(ep_id="ep_test")
        audit_file = self.tmp_dir / "audit_eval_ep_test.json"
        self.assertTrue(audit_file.exists())


if __name__ == "__main__":
    unittest.main()

