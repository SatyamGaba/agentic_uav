import math
import numpy as np
import pytest
from agenticnav_paper_replica.geometry.pinhole import (
    CameraConfig,
    VIEW_AZIMUTHS_DEG,
    camera_to_agent_frame,
    agent_to_camera_frame,
    deproject_pixel,
    unproject_depth_image,
    extract_view_obstacle_and_free_points,
    parse_view_index,
    parse_view_index_ex,
    InvalidViewError,
)

cfg = CameraConfig()


def test_physical_direction_of_each_view():
    """Verify that optical axis (0, 0, 2) in each view points in the correct physical direction."""
    opt_center = np.array([0.0, 0.0, 2.0], dtype=np.float64)
    expected_signs = {
        0: (0.0, -2.0),   # Front: Z_ag negative
        1: (2.0, 0.0),    # Right: X_ag positive
        2: (0.0, 2.0),    # Back:  Z_ag positive
        3: (-2.0, 0.0),   # Left:  X_ag negative
    }
    for view, (exp_x, exp_z) in expected_signs.items():
        p_ag = camera_to_agent_frame(opt_center, view, cfg)
        assert abs(p_ag[0] - exp_x) < 1e-6, f"View {view} X_ag failed: expected {exp_x}, got {p_ag[0]}"
        assert abs(p_ag[2] - exp_z) < 1e-6, f"View {view} Z_ag failed: expected {exp_z}, got {p_ag[2]}"


def circ_diff(a: float, b: float) -> float:
    return (a - b + 180.0) % 360.0 - 180.0


@pytest.mark.parametrize("view,expected_deg", [
    (0, 0.0),
    (1, -90.0),
    (2, 180.0),
    (3, 90.0),
])
def test_heading_values(view: int, expected_deg: float):
    """Verify that center pixel heading matches expected view azimuth within 0.15 deg."""
    opt_point = np.array([0.0, 0.0, 2.0], dtype=np.float64)
    p_ag = camera_to_agent_frame(opt_point, view, cfg)
    heading_deg = math.degrees(math.atan2(-p_ag[0], -p_ag[2]))
    assert abs(circ_diff(heading_deg, expected_deg)) < 0.15, (
        f"View {view} heading {heading_deg} failed to match {expected_deg}"
    )


def test_agent_camera_roundtrip():
    """Verify exact roundtrip between camera_to_agent_frame and agent_to_camera_frame."""
    test_points = [
        np.array([0.0, 0.0, 2.0], dtype=np.float64),
        np.array([1.5, -0.3, 3.2], dtype=np.float64),
        np.array([-0.8, 0.5, 1.2], dtype=np.float64),
        np.array([0.1, 0.1, 4.8], dtype=np.float64),
    ]
    for view in range(4):
        for p_c in test_points:
            p_ag = camera_to_agent_frame(p_c, view, cfg)
            p_c_rec = agent_to_camera_frame(p_ag, view, cfg)
            assert np.allclose(p_c, p_c_rec, atol=1e-9), f"Roundtrip failed for view {view}, point {p_c}"


def test_vectorized_path_matches_scalar_path():
    """Verify that unproject_depth_image and extract_view_obstacle_and_free_points match camera_to_agent_frame."""
    depth_img = np.full((cfg.image_height, cfg.image_width), 2.0, dtype=np.float32)
    stride = 16
    for view in range(4):
        cloud_unproj = unproject_depth_image(depth_img, view_idx=view, config=cfg, stride=stride, y_min=-10.0, y_max=10.0)
        cloud_extract, _ = extract_view_obstacle_and_free_points(depth_img, view_idx=view, config=cfg, stride=stride, y_min=-10.0, y_max=10.0)

        ys = np.arange(0, cfg.image_height, stride)
        xs = np.arange(0, cfg.image_width, stride)
        grid_x, grid_y = np.meshgrid(xs, ys)
        gx_flat = grid_x.flatten()
        gy_flat = grid_y.flatten()

        for idx in [0, len(gx_flat) // 2, len(gx_flat) - 1]:
            x0 = gx_flat[idx]
            y0 = gy_flat[idx]
            p_c = deproject_pixel(x0, y0, 2.0, cfg, is_normalized=False)
            p_ag_scalar = camera_to_agent_frame(p_c, view, cfg)

            diff_unproj = np.min(np.linalg.norm(cloud_unproj - p_ag_scalar, axis=1))
            assert diff_unproj < 1e-6, f"Vector unproject mismatch at pixel ({x0}, {y0}) in view {view}"

            diff_extract = np.min(np.linalg.norm(cloud_extract - p_ag_scalar, axis=1))
            assert diff_extract < 1e-6, f"Vector extract mismatch at pixel ({x0}, {y0}) in view {view}"


def test_side_view_clouds_lie_on_correct_side():
    """Verify that full point clouds for side views lie strictly on the expected side of the agent."""
    depth_img = np.full((cfg.image_height, cfg.image_width), 2.0, dtype=np.float32)
    stride = 8

    # View 1 is Right (-90 deg): all optical rays ahead in view 1 have X_agent > 0
    cloud_right = unproject_depth_image(depth_img, view_idx=1, config=cfg, stride=stride, y_min=-10.0, y_max=10.0)
    assert np.all(cloud_right[:, 0] > 0.0), f"Right view cloud has non-positive X coordinates: min={np.min(cloud_right[:, 0])}"

    # View 3 is Left (+90 deg): all optical rays ahead in view 3 have X_agent < 0
    cloud_left = unproject_depth_image(depth_img, view_idx=3, config=cfg, stride=stride, y_min=-10.0, y_max=10.0)
    assert np.all(cloud_left[:, 0] < 0.0), f"Left view cloud has non-negative X coordinates: max={np.max(cloud_left[:, 0])}"



def _nearest_neighbor_mean_dist(pts_a: np.ndarray, pts_b: np.ndarray, max_samples: int = 500) -> float:
    """Compute mean nearest-neighbor distance from pts_a to pts_b using Euclidean distance."""
    if len(pts_a) == 0 or len(pts_b) == 0:
        return float("inf")
    rng = np.random.RandomState(42)
    if len(pts_a) > max_samples:
        idx_a = rng.choice(len(pts_a), max_samples, replace=False)
        pts_a = pts_a[idx_a]
    if len(pts_b) > max_samples:
        idx_b = rng.choice(len(pts_b), max_samples, replace=False)
        pts_b = pts_b[idx_b]

    diff = pts_a[:, np.newaxis, :] - pts_b[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    min_dists = np.sqrt(np.min(dist_sq, axis=1))
    return float(np.mean(min_dists))


def test_pointcloud_physical_overlap_with_negative_control():
    """Simulator test verifying physical overlap (< 0.08m) and mirrored negative control (> 0.30m)."""
    import os
    scene_path = "/home/krutik/data/scene_datasets/mp3d/gZ6f7yhEvPG/gZ6f7yhEvPG.glb"
    if not os.path.exists(scene_path):
        pytest.skip(f"Habitat scene not found on disk at {scene_path}")

    from agenticnav_paper_replica.embodiment.habitat_ground import HabitatGroundEmbodiment
    emb = HabitatGroundEmbodiment(scene_id=scene_path, max_actions=100, gpu_device_id=-1)
    ep_data = {
        "start_position": [-2.79, 0.26, 1.38],
        "start_rotation": [0.0, 0.0, 0.0, 1.0],
    }

    try:
        # Check View 3 (Left +90 deg) vs physical 90 deg left turn
        obs_init = emb.reset(ep_data)
        pts_v3, _ = extract_view_obstacle_and_free_points(obs_init.depth_views[3], view_idx=3, config=cfg, stride=8)

        for _ in range(6):
            emb.sim.step("turn_left")
        obs_after_left = emb._extract_obs()

        pts_front_left, _ = extract_view_obstacle_and_free_points(obs_after_left.depth_views[0], view_idx=0, config=cfg, stride=8)

        psi_l = math.radians(90.0)
        cos_l, sin_l = math.cos(psi_l), math.sin(psi_l)
        x_rot = cos_l * pts_front_left[:, 0] + sin_l * pts_front_left[:, 2]
        z_rot = -sin_l * pts_front_left[:, 0] + cos_l * pts_front_left[:, 2]
        pts_front_left_orig = np.column_stack([x_rot, pts_front_left[:, 1], z_rot])

        mean_nn = _nearest_neighbor_mean_dist(pts_v3, pts_front_left_orig)
        assert mean_nn < 0.08, f"True Left view NN distance too high: {mean_nn:.4f}m"

        # Negative control: mirror side cloud in X
        pts_v3_mirrored = pts_v3.copy()
        pts_v3_mirrored[:, 0] *= -1.0
        mean_nn_neg = _nearest_neighbor_mean_dist(pts_v3_mirrored, pts_front_left_orig)
        assert mean_nn_neg > 0.30, f"Mirrored negative control failed to separate: {mean_nn_neg:.4f}m <= 0.30m"

        # Check View 1 (Right -90 deg) vs physical 90 deg right turn
        obs_init = emb.reset(ep_data)
        pts_v1, _ = extract_view_obstacle_and_free_points(obs_init.depth_views[1], view_idx=1, config=cfg, stride=8)

        for _ in range(6):
            emb.sim.step("turn_right")
        obs_after_right = emb._extract_obs()

        pts_front_right, _ = extract_view_obstacle_and_free_points(obs_after_right.depth_views[0], view_idx=0, config=cfg, stride=8)

        psi_r = math.radians(-90.0)
        cos_r, sin_r = math.cos(psi_r), math.sin(psi_r)
        x_rot_r = cos_r * pts_front_right[:, 0] + sin_r * pts_front_right[:, 2]
        z_rot_r = -sin_r * pts_front_right[:, 0] + cos_r * pts_front_right[:, 2]
        pts_front_right_orig = np.column_stack([x_rot_r, pts_front_right[:, 1], z_rot_r])

        mean_nn_r = _nearest_neighbor_mean_dist(pts_v1, pts_front_right_orig)
        assert mean_nn_r < 0.08, f"True Right view NN distance too high: {mean_nn_r:.4f}m"

        pts_v1_mirrored = pts_v1.copy()
        pts_v1_mirrored[:, 0] *= -1.0
        mean_nn_neg_r = _nearest_neighbor_mean_dist(pts_v1_mirrored, pts_front_right_orig)
        assert mean_nn_neg_r > 0.30, f"Mirrored Right negative control failed to separate: {mean_nn_neg_r:.4f}m <= 0.30m"
    finally:
        emb.close()



def test_parse_view_index_ex_valid_and_rewritten():
    """Verify parse_view_index_ex returns expected (view, rewritten) pairs."""
    # Exact ints: not rewritten
    assert parse_view_index_ex(0) == (0, False)
    assert parse_view_index_ex(1) == (1, False)
    assert parse_view_index_ex(2) == (2, False)
    assert parse_view_index_ex(3) == (3, False)

    # Strings: rewritten
    assert parse_view_index_ex("0") == (0, True)
    assert parse_view_index_ex("1") == (1, True)
    assert parse_view_index_ex("2") == (2, True)
    assert parse_view_index_ex("3") == (3, True)

    # Keywords: rewritten
    assert parse_view_index_ex("Front") == (0, True)
    assert parse_view_index_ex("forward") == (0, True)
    assert parse_view_index_ex("right") == (1, True)
    assert parse_view_index_ex("Back") == (2, True)
    assert parse_view_index_ex("rear") == (2, True)
    assert parse_view_index_ex("left") == (3, True)
    assert parse_view_index_ex("view 1") == (1, True)
    assert parse_view_index_ex("cam: 2") == (2, True)
    assert parse_view_index_ex(1.0) == (1, True)


def test_parse_view_index_ex_invalid_inputs_raise():
    """Verify parse_view_index_ex raises InvalidViewError on invalid/ambiguous inputs."""
    invalid_inputs = [
        None,
        True,
        False,
        -1,
        4,
        5,
        "banana",
        "",
        "   ",
        "front-right",  # ambiguous multiple keyword match
        "left_and_right",  # ambiguous
        "camera 9",
        1.5,
        float("nan"),
    ]
    for inp in invalid_inputs:
        with pytest.raises(InvalidViewError):
            parse_view_index_ex(inp)
