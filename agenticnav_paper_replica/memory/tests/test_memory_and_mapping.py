"""Unit tests for memory and BEV mapping components."""

import numpy as np
import pytest

from agenticnav_paper_replica.memory.history import ActionHistory
from agenticnav_paper_replica.memory.image_cache import EpisodicImageCache
from agenticnav_paper_replica.mapping.occupancy_grid import OccupancyGrid2D
from agenticnav_paper_replica.mapping.bev_renderer import BEVRenderer


class TestActionHistory:
    def test_sliding_window_limit(self):
        hist = ActionHistory(max_history=6)
        for i in range(10):
            hist.append(
                step_idx=i + 1,
                from_node=i + 1,
                to_node=i + 2,
                action_type="move_to",
                action_params={"view": 0, "u": 0.5, "v": 0.5},
                reason=f"Moving step {i+1}",
                outcome={"status": "executed", "quantized_rho_m": 0.5, "quantized_theta_deg": 0.0},
            )
        assert len(hist) == 6
        records = hist.get_records()
        # Should contain steps 5 through 10
        assert records[0].step_idx == 5
        assert records[-1].step_idx == 10

    def test_prompt_formatting_and_ablations(self):
        hist = ActionHistory(max_history=6, mode="full")
        hist.append(
            step_idx=1,
            from_node=1,
            to_node=2,
            action_type="move_to",
            action_params={"view": 1, "u": 0.52, "v": 0.61},
            reason="Explore right corridor",
            outcome={"status": "executed", "quantized_rho_m": 1.25, "quantized_theta_deg": -90.0},
        )
        text_full = hist.format_for_prompt()
        assert "Node 1 -> Node 2" in text_full
        assert "Explore right corridor" in text_full
        assert "Right (-90°)" in text_full

        # Ablation: without_history
        hist_none = ActionHistory(mode="without_history")
        hist_none.append(
            step_idx=1, from_node=1, to_node=2,
            action_type="move_to", action_params={}, reason="Test",
        )
        assert hist_none.format_for_prompt() == "No previous actions recorded."

        # Ablation: reasoning_to_task_progress
        hist_prog = ActionHistory(mode="reasoning_to_task_progress")
        hist_prog.append(
            step_idx=1, from_node=1, to_node=2,
            action_type="move_to", action_params={"view": 0, "u": 0.5, "v": 0.5},
            reason="Internal rationale that should be hidden",
            task_progress_summary="Moved 1.0m down the hallway.",
        )
        text_prog = hist_prog.format_for_prompt()
        assert "Moved 1.0m down the hallway." in text_prog
        assert "Internal rationale that should be hidden" not in text_prog

    def test_prompt_formatting_collision_shortfall_and_pure_rotation(self):
        hist = ActionHistory(max_history=6, mode="full")
        # 1. Pure in-place rotation
        hist.append(
            step_idx=1,
            from_node=1,
            to_node=2,
            action_type="move_to",
            action_params={"view": 3, "u": 0.5, "v": 0.85, "theta_deg": 90.0, "rho_m": 0.0, "forward_steps": 0},
            reason="Turn left toward archway",
            outcome={"status": "executed", "moved_m": 0.0, "collided": False},
        )
        prompt_rot = hist.format_for_prompt()
        assert "ROTATED +90° in place" in prompt_rot
        assert "View 0 now faces your new heading" in prompt_rot

        # 2. Collision shortfall
        hist.append(
            step_idx=2,
            from_node=2,
            to_node=3,
            action_type="move_to",
            action_params={"view": 0, "u": 0.5, "v": 0.7, "theta_deg": 0.0, "rho_m": 1.0, "forward_steps": 4},
            reason="Move along billiard table",
            outcome={"status": "executed", "moved_m": 0.14, "collided": True},
        )
        prompt_coll = hist.format_for_prompt()
        assert "PHYSICAL COLLISION" in prompt_coll
        assert "shortened displacement from 1.00m to 0.14m" in prompt_coll


class TestEpisodicImageCache:
    def test_store_and_recall_valid_node(self):
        cache = EpisodicImageCache()
        # Store observations at node 1
        dummy_views = {v: np.full((100, 100, 3), v * 50, dtype=np.uint8) for v in range(4)}
        cache.store_node_observations(node=1, rgb_views=dummy_views)

        # Agent is now at node 2
        cache.set_current_node(node=2)

        # Recall node 1 view 0
        res = cache.recall(node=1, view=0)
        assert res.success is True
        assert res.image is not None
        assert res.image.shape == (100, 100, 3)

    def test_recall_current_or_future_node_rejected(self):
        cache = EpisodicImageCache()
        cache.set_current_node(node=2)

        # Recall current node 2 must fail
        res_cur = cache.recall(node=2, view=0)
        assert res_cur.success is False
        assert "past nodes" in res_cur.error_message

        # Recall future node 3 must fail
        res_fut = cache.recall(node=3, view=0)
        assert res_fut.success is False
        assert "past nodes" in res_fut.error_message

        # Recall node 0 must fail
        res_zero = cache.recall(node=0, view=0)
        assert res_zero.success is False
        assert "1-indexed" in res_zero.error_message

    def test_recall_invalid_view_rejected(self):
        cache = EpisodicImageCache()
        cache.set_current_node(node=3)
        res = cache.recall(node=1, view=5)
        assert res.success is False
        assert "Invalid view=5" in res.error_message


class TestOccupancyGridAndBEVRenderer:
    def test_grid_world_transforms(self):
        grid = OccupancyGrid2D(grid_size=256, resolution_m=0.15)
        # Center is (0.0, 0.0) -> (128, 128)
        c, r = grid.world_to_grid(0.0, 0.0)
        assert c == 128
        assert r == 128

        # Roundtrip
        wx, wz = grid.grid_to_world(c, r)
        assert abs(wx) < 1e-4
        assert abs(wz) < 1e-4

    def test_grid_raycast_update(self):
        grid = OccupancyGrid2D(grid_size=256, resolution_m=0.15)
        # Obstacle at (0.0, -2.0)
        obs = np.array([[0.0, -2.0]])
        grid.update(agent_x_m=0.0, agent_z_m=0.0, obstacle_points_world_xz=obs)

        # Obstacle cell should be occupied (value 0 in uint8 map)
        img = grid.to_uint8_image()
        c_obs, r_obs = grid.world_to_grid(0.0, -2.0)
        assert img[r_obs, c_obs] == 0

    def test_free_space_clearing_and_dilation(self):
        grid = OccupancyGrid2D(grid_size=256, resolution_m=0.15)
        # Free space endpoint at (0.0, -3.0) (e.g. max range or floor hit)
        free_pts = np.array([[0.0, -3.0]])
        grid.update(
            agent_x_m=0.0,
            agent_z_m=0.0,
            obstacle_points_world_xz=np.empty((0, 2)),
            free_endpoints_world_xz=free_pts,
        )

        img = grid.to_uint8_image(dilate_obstacles=False)
        # Intermediate cell along ray should be free (255)
        c_mid, r_mid = grid.world_to_grid(0.0, -1.5)
        assert img[r_mid, c_mid] == 255
        # Endpoint itself should NOT be marked occupied
        c_end, r_end = grid.world_to_grid(0.0, -3.0)
        assert img[r_end, c_end] != 0

        # Test dilation on an isolated occupied point
        grid_dil = OccupancyGrid2D(grid_size=256, resolution_m=0.15)
        obs = np.array([[0.0, -1.0]])
        grid_dil.update(agent_x_m=0.0, agent_z_m=0.0, obstacle_points_world_xz=obs)
        c_o, r_o = grid_dil.world_to_grid(0.0, -1.0)

        img_undil = grid_dil.to_uint8_image(dilate_obstacles=False)
        img_dil = grid_dil.to_uint8_image(dilate_obstacles=True)
        # 1-pixel dilation must cause neighbor to become 0 (occupied)
        assert img_undil[r_o + 1, c_o] != 0
        assert img_dil[r_o + 1, c_o] == 0

    def test_dynamic_grid_recentering(self):
        grid = OccupancyGrid2D(grid_size=256, resolution_m=0.15)
        # Move agent far to the right (+20.0m)
        assert grid.origin_x_m == 0.0
        grid.check_and_recenter(agent_x_m=20.0, agent_z_m=0.0, margin_px=32)
        assert grid.origin_x_m > 0.0
        # Agent should now be mapped near center_px
        c_new, _ = grid.world_to_grid(20.0, 0.0)
        assert abs(c_new - grid.center_px) < 5

    def test_bev_renderer_full_and_ablation(self):
        grid = OccupancyGrid2D(grid_size=256, resolution_m=0.15)
        renderer = BEVRenderer(mode="full")

        poses = [
            (0.0, 0.0, 0.0),      # Node 1
            (0.0, -1.5, 0.0),     # Node 2
            (1.0, -2.5, 1.57),    # Node 3
        ]
        bev_img = renderer.render(grid, node_poses=poses, current_node_idx=3)
        assert bev_img is not None
        assert bev_img.shape == (256, 256, 3)
        assert bev_img.dtype == np.uint8

        # Ablation mode
        renderer_none = BEVRenderer(mode="without_bev")
        bev_none = renderer_none.render(grid, node_poses=poses, current_node_idx=3)
        assert bev_none is None


