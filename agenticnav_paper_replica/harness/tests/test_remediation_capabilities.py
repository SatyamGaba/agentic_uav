"""Unit tests for Arm Remediation Capabilities:
- Near-plane clipping recovery and upward ray scanning
- Azimuth ground projection fallback for Oracle actuation
- Tool call argument sanitization and coordinate normalization
- Runtime loop/oscillation warning detection and injection
"""

import json
import math
import numpy as np
import pytest

from agenticnav_paper_replica.geometry.pinhole import (
    CameraConfig,
    scan_upward_for_valid_depth,
    compute_azimuth_ground_projection,
    deproject_pixel,
    camera_to_agent_frame,
)
from agenticnav_paper_replica.geometry.safety_region import GroundCorridorChecker
from agenticnav_paper_replica.harness.executor import ToolExecutor
from agenticnav_paper_replica.harness.prompts import (
    build_system_prompt,
    build_step_user_prompt,
)
from agenticnav_paper_replica.harness.step_loop import (
    sanitize_tool_call,
    StepDecisionLoop,
    StepLoopResult,
)
from agenticnav_paper_replica.memory.image_cache import EpisodicImageCache
from agenticnav_paper_replica.memory.history import ActionHistory


class TestRemediationNearPlaneClipping:
    def setup_method(self):
        self.config = CameraConfig()
        self.checker = GroundCorridorChecker()
        self.image_cache = EpisodicImageCache()
        self.executor_standard = ToolExecutor(self.config, self.checker, self.image_cache, oracle_local_actuation=False)
        self.executor_oracle = ToolExecutor(self.config, self.checker, self.image_cache, oracle_local_actuation=True)

    def test_scan_upward_for_valid_depth(self):
        # Create a depth image where lower half (v >= 0.6) has near-plane clip (0.2m <= d_min=0.5m)
        # but upper half has valid open space (2.5m)
        depth_img = np.full((480, 640), 2.5, dtype=np.float32)
        depth_img[288:, :] = 0.2  # v >= 0.60 is clipped

        res = scan_upward_for_valid_depth(depth_img, u=0.5, v=0.75, config=self.config, min_v=0.35, step_v=0.04)
        assert res is not None
        v_cand, x0, y0, z_c = res
        assert v_cand < 0.60
        assert z_c == 2.5

    def test_compute_azimuth_ground_projection(self):
        # Front view (view 0), center u=0.5: heading straight ahead (p_x=0, p_z=-1.25)
        p_fwd = compute_azimuth_ground_projection(view_idx=0, u=0.5, config=self.config, distance_m=1.25)
        assert abs(p_fwd[0]) < 5e-3
        assert abs(p_fwd[1]) < 1e-3  # Floor elevation
        assert abs(p_fwd[2] - (-1.25)) < 1e-3

        # Right view (view 1), center u=0.5: heading to the right (p_x=+1.25, p_z=0)
        p_right = compute_azimuth_ground_projection(view_idx=1, u=0.5, config=self.config, distance_m=1.25)
        assert abs(p_right[0] - 1.25) < 1e-3
        assert abs(p_right[1]) < 1e-3
        assert abs(p_right[2]) < 5e-3

    def test_oracle_recovers_from_near_plane_clip(self):
        # Depth view entirely clipped at 0.3m (e.g. facing a table or wall closely)
        depth_views = {0: np.full((480, 640), 0.3, dtype=np.float32)}

        # In Oracle mode, execute_move_to uses azimuth ground projection fallback
        res = self.executor_oracle.execute_move_to(
            view=0, u=0.5, v=0.75, reason="Move past table",
            depth_views=depth_views, obstacle_points_agent_xz=np.empty((0, 2)),
        )
        assert res["status"] == "executed"
        assert res["rho_m"] == 1.25
        assert res["forward_steps"] == 5

        # In standard non-oracle mode, returns informative reselect with near_plane_clip diagnostic
        res_std = self.executor_standard.execute_move_to(
            view=0, u=0.5, v=0.75, reason="Move past table",
            depth_views=depth_views, obstacle_points_agent_xz=np.empty((0, 2)),
        )
        assert res_std["status"] == "reselect"
        assert res_std["cause"] == "near_field_occlusion"
        assert res_std["detail"] == "near_field_occlusion"
        assert "NEAR_FIELD_OCCLUSION" in res_std["message"]


class TestRemediationToolCallSanitizer:
    def test_sanitize_markdown_wrapped_arguments(self):
        raw_call = {
            "name": "move_to",
            "arguments": '```json\n{"view": 0, "u": 0.5, "v": 0.65, "reason": "Advance forward"}\n```',
        }
        clean = sanitize_tool_call(raw_call)
        assert clean["name"] == "move_to"
        assert clean["arguments"]["view"] == 0
        assert clean["arguments"]["u"] == 0.5
        assert clean["arguments"]["v"] == 0.65

    def test_sanitize_pixel_coordinates_and_string_view(self):
        # Raw pixel coordinates (e.g. u=320, v=240 on 640x480) and string view "View 1"
        raw_call = {
            "name": "move_to",
            "arguments": {"view": "View 1", "u": 320, "v": 240, "reason": "Turn right"},
        }
        clean = sanitize_tool_call(raw_call)
        assert clean["name"] == "move_to"
        assert clean["arguments"]["view"] == 1
        assert abs(clean["arguments"]["u"] - 0.5) < 1e-3
        assert abs(clean["arguments"]["v"] - 0.5) < 1e-3

    def test_sanitize_query_depth_json_string_points(self):
        raw_call = {
            "name": "query_depth",
            "arguments": {"points": '[{"view": "front", "u": 0.5, "v": 0.7}]'},
        }
        clean = sanitize_tool_call(raw_call)
        assert clean["name"] == "query_depth"
        pts = clean["arguments"]["points"]
        assert len(pts) == 1
        assert pts[0]["view"] == 0
        assert pts[0]["u"] == 0.5
        assert pts[0]["v"] == 0.7


class TestRemediationLoopWarningAndPrompts:
    def test_loop_warning_rendered_in_prompt(self):
        warning = "[LOOP WARNING: You are currently within 0.85m of Step 2.]"
        prompt = build_step_user_prompt(
            instruction="Go to the kitchen and stop at the island.",
            current_node=5,
            history_text="Step 4: moved 1.0m",
            accumulated_distance_m=5.0,
            loop_warning=warning,
        )
        assert "### Loop / Oscillation Warning:" in prompt
        assert warning in prompt

    def test_system_prompt_contains_3_stage_protocol(self):
        sys_prompt = build_system_prompt()
        assert "3-STAGE DECISION PROTOCOL" in sys_prompt
        assert "STAGE 1: Mandatory Pre-Stride Arrival Check" in sys_prompt
        assert "STAGE 2: Counterfactual Dual-Hypothesis Gate" in sys_prompt
        assert "STAGE 3: Anti-Loop & BEV Trajectory Discipline" in sys_prompt
        assert "H_ARRIVED" in sys_prompt
        assert "H_PREMATURE" in sys_prompt
