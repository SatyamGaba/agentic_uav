"""Habitat-Sim ground embodiment implementation and mock embodiment."""

import math
import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from agenticnav_paper_replica.embodiment.base import EmbodimentBase, StepObservation
from agenticnav_paper_replica.geometry.pinhole import CameraConfig, VIEW_AZIMUTHS_DEG

try:
    import habitat_sim
    HABITAT_AVAILABLE = True
except ImportError:
    HABITAT_AVAILABLE = False


def resolve_scene_path(scene_id: str, search_roots: Optional[List[str]] = None) -> str:
    """Resolves raw scene ID (e.g. 'gZ6f7yhEvPG' or 'mp3d/gZ6f7yhEvPG/gZ6f7yhEvPG.glb') to a valid .glb file path."""
    if os.path.isfile(scene_id):
        return os.path.abspath(scene_id)

    # Clean raw id if in format "mp3d/SCENE/SCENE.glb"
    base_id = scene_id
    if "/" in base_id:
        for part in base_id.split("/"):
            if part and not part.endswith(".glb") and part != "mp3d":
                base_id = part
                break
    base_id = base_id.replace(".glb", "")
    roots = search_roots or [
        "/home/krutik/data/scene_datasets/mp3d",
        "data/scene_datasets/mp3d",
        "agenticnav_paper_replica/data/scene_datasets/mp3d",
        "/home/jovyan/scratch/agenticnav_paper_replica/data/scene_datasets/mp3d",
        "/home/jovyan/scratch/scene_datasets/mp3d",
        "/data/scene_datasets/mp3d",
        "/datasets/mp3d",
        "/datasets/scene_datasets/mp3d",
    ]
    for r in roots:
        p1 = os.path.join(r, base_id, f"{base_id}.glb")
        if os.path.isfile(p1):
            return p1
        p2 = os.path.join(r, f"{base_id}.glb")
        if os.path.isfile(p2):
            return p2
    return scene_id


class MockGroundEmbodiment(EmbodimentBase):
    """Mock embodiment with geometric raycasting against indoor corridor and room layouts.
    
    Provides realistic 2D depth pointclouds and stylized indoor camera views for development
    and benchmarking when Habitat-Sim or Matterport3D meshes are not locally available.
    """

    def __init__(self, camera_config: Optional[CameraConfig] = None, max_actions: int = 500):
        self.config = camera_config or CameraConfig()
        self.max_actions = max_actions
        self.action_count = 0
        self.x = 0.0
        self.z = 0.0
        self.yaw_rad = 0.0
        self.goal_x = 0.0
        self.goal_z = -10.0
        self.shortest_path_len = 10.0
        self.trajectory: List[Tuple[float, float, float]] = []
        self.walls: List[Tuple[np.ndarray, np.ndarray]] = []
        self._precompute_camera_rays()

    def _precompute_camera_rays(self) -> None:
        """Precomputes intrinsic optical ray angles for fast vectorized raycasting."""
        W, H = self.config.image_width, self.config.image_height
        self._cols = np.arange(W)
        self._rows = np.arange(H)
        self._ray_angles = np.arctan2(self._cols + 0.5 - self.config.cx, self.config.fx)  # (W,)
        self._pitches = np.arctan2(self._rows + 0.5 - self.config.cy, self.config.fy)      # (H,)
        # Floor depth per row (points downward when pitch > 0.02 rad)
        self._floor_dist_ground = np.where(self._pitches > 0.02, self.config.camera_height / np.tan(self._pitches), 100.0)
        self._floor_opt = self._floor_dist_ground[:, None] * np.cos(self._ray_angles)[None, :]  # (H, W)

    def _build_room_geometry(self) -> None:
        """Constructs indoor corridor and room boundary wall segments connecting start to goal."""
        gx, gz = self.goal_x, self.goal_z
        walls = []
        w_half = 1.6  # Corridor half-width (3.2m wide corridor)

        min_z = min(gz - 2.5, -14.0)
        max_z = 2.5
        min_x = min(gx - 2.5, -w_half)
        max_x = max(gx + 2.5, w_half)

        # 1. Main corridor segments
        if abs(gx) < 0.8:
            # Straight corridor
            walls.append((np.array([-w_half, max_z]), np.array([-w_half, min_z])))
            walls.append((np.array([w_half, max_z]), np.array([w_half, min_z])))
            walls.append((np.array([-w_half, max_z]), np.array([w_half, max_z])))
            walls.append((np.array([-w_half, min_z]), np.array([w_half, min_z])))
        else:
            # L-shaped / turning layout: corridor down to gz, then branch toward gx
            # Main corridor
            walls.append((np.array([-w_half, max_z]), np.array([w_half, max_z])))
            if gx > 0:
                walls.append((np.array([-w_half, max_z]), np.array([-w_half, min_z])))
                walls.append((np.array([w_half, max_z]), np.array([w_half, gz + 1.2])))
                walls.append((np.array([w_half, gz - 1.2]), np.array([w_half, min_z])))
                # Target room on the right
                walls.append((np.array([w_half, gz + 1.2]), np.array([max_x, gz + 1.2])))
                walls.append((np.array([w_half, gz - 1.2]), np.array([max_x, gz - 1.2])))
                walls.append((np.array([max_x, gz - 1.2]), np.array([max_x, gz + 1.2])))
            else:
                walls.append((np.array([w_half, max_z]), np.array([w_half, min_z])))
                walls.append((np.array([-w_half, max_z]), np.array([-w_half, gz + 1.2])))
                walls.append((np.array([-w_half, gz - 1.2]), np.array([-w_half, min_z])))
                # Target room on the left
                walls.append((np.array([-w_half, gz + 1.2]), np.array([min_x, gz + 1.2])))
                walls.append((np.array([-w_half, gz - 1.2]), np.array([min_x, gz - 1.2])))
                walls.append((np.array([min_x, gz - 1.2]), np.array([min_x, gz + 1.2])))

            walls.append((np.array([-w_half, min_z]), np.array([w_half, min_z])))

        # 2. Add goal landmark object (a 0.8m x 0.8m table/counter at the goal position)
        hw = 0.4
        walls.append((np.array([gx - hw, gz - hw]), np.array([gx + hw, gz - hw])))
        walls.append((np.array([gx + hw, gz - hw]), np.array([gx + hw, gz + hw])))
        walls.append((np.array([gx + hw, gz + hw]), np.array([gx - hw, gz + hw])))
        walls.append((np.array([gx - hw, gz + hw]), np.array([gx - hw, gz - hw])))

        self.walls = walls

    def reset(self, episode_data: Dict[str, Any]) -> StepObservation:
        self.action_count = 0
        self.x = 0.0
        self.z = 0.0
        self.yaw_rad = 0.0

        goal_pos = episode_data.get("goal_position")
        if goal_pos and isinstance(goal_pos, (list, tuple)) and len(goal_pos) >= 3:
            self.goal_x = float(goal_pos[0])
            self.goal_z = float(goal_pos[2])
        else:
            self.goal_x = float(episode_data.get("goal_x", 0.0))
            self.goal_z = float(episode_data.get("goal_z", -10.0))

        if "shortest_path_distance" in episode_data:
            self.shortest_path_len = float(episode_data["shortest_path_distance"])
        else:
            self.shortest_path_len = math.sqrt((self.goal_x - self.x) ** 2 + (self.goal_z - self.z) ** 2)

        self.trajectory = [(self.x, self.z, self.yaw_rad)]
        self._build_room_geometry()
        obs = self._make_obs()
        self.first_observation = obs
        self.last_observation = obs
        return obs

    def step_discrete(self, turn_steps_15deg: int, forward_steps_025m: int) -> StepObservation:
        x_before = self.x
        z_before = self.z
        prim_traj: List[Tuple[float, float, float]] = []
        prim_geod: List[float] = []

        turn_dir = 1 if turn_steps_15deg > 0 else -1
        for _ in range(abs(turn_steps_15deg)):
            self.yaw_rad += math.radians(15.0 * turn_dir)
            self.action_count += 1
            cur_pose = (self.x, self.z, self.yaw_rad)
            self.trajectory.append(cur_pose)
            prim_traj.append(cur_pose)
            prim_geod.append(self.get_geodesic_distance_to_goal())

        dx = -math.sin(self.yaw_rad) * 0.25
        dz = -math.cos(self.yaw_rad) * 0.25
        for _ in range(max(0, forward_steps_025m)):
            self.x += dx
            self.z += dz
            self.action_count += 1
            cur_pose = (self.x, self.z, self.yaw_rad)
            self.trajectory.append(cur_pose)
            prim_traj.append(cur_pose)
            prim_geod.append(self.get_geodesic_distance_to_goal())

        actual_dist = math.sqrt((self.x - x_before) ** 2 + (self.z - z_before) ** 2)
        obs = self._make_obs()
        obs.primitive_trajectory = prim_traj
        obs.primitive_geodesic_distances = prim_geod
        obs.actual_distance_moved = actual_dist
        obs.collided = False
        self.last_observation = obs
        return obs

    def _make_obs(self) -> StepObservation:
        """Generates realistic metric depth views and stylized RGB observations via vectorized raycasting."""
        W, H = self.config.image_width, self.config.image_height
        depth_views = {}
        rgb_views = {}

        for v_idx, psi_deg in VIEW_AZIMUTHS_DEG.items():
            v_yaw = self.yaw_rad + math.radians(psi_deg)
            tot_yaw = v_yaw + self._ray_angles  # (W,)
            dx = -np.sin(tot_yaw)  # (W,)
            dz = -np.cos(tot_yaw)  # (W,)

            # Raycast against all walls
            min_dists = np.full(W, 100.0, dtype=np.float64)
            for w1, w2 in self.walls:
                v1_0 = w1[0] - self.x
                v1_1 = w1[1] - self.z
                v2_0 = w2[0] - w1[0]
                v2_1 = w2[1] - w1[1]

                cross = dx * v2_1 - dz * v2_0
                valid = np.abs(cross) > 1e-6
                t = np.where(valid, (v1_0 * v2_1 - v1_1 * v2_0) / (cross + 1e-12), 100.0)
                u = np.where(valid, (v1_0 * dz - v1_1 * dx) / (cross + 1e-12), -1.0)
                hit = valid & (t > 0.05) & (u >= 0.0) & (u <= 1.0)
                min_dists = np.where(hit & (t < min_dists), t, min_dists)

            # Optical depth to walls for all pixels (H, W)
            wall_opt = min_dists[None, :] * np.cos(self._ray_angles)[None, :]
            # Combine floor depth with wall depth
            depth_raw = np.minimum(self._floor_opt, wall_opt)
            depth_img = np.clip(depth_raw, self.config.d_min, self.config.d_max).astype(np.float32)

            # Generate stylized RGB view
            rgb_img = np.zeros((H, W, 3), dtype=np.uint8)
            is_floor = (self._floor_opt < wall_opt) & (depth_img < self.config.d_max)
            is_wall = (~is_floor) & (depth_img < self.config.d_max - 0.05)
            is_open = depth_img >= self.config.d_max - 0.05

            rgb_img[is_floor] = [185, 170, 150]    # Warm wood / floor tone
            rgb_img[is_wall] = [225, 222, 215]     # Clean indoor wall
            rgb_img[is_open] = [75, 80, 95]        # Open corridor / doorway depth

            # If facing goal landmark within 3.5m, render distinctive landmark marker
            dist_to_goal = math.sqrt((self.goal_x - self.x) ** 2 + (self.goal_z - self.z) ** 2)
            if dist_to_goal < 4.0:
                angle_to_goal = math.atan2(-(self.goal_x - self.x), -(self.goal_z - self.z))
                rel_angle = (angle_to_goal - v_yaw + math.pi) % (2 * math.pi) - math.pi
                if abs(rel_angle) < math.radians(40.0):
                    # Landmark is inside camera FOV
                    center_col = int(round(self.config.cx + self.config.fx * math.tan(rel_angle)))
                    col_span = max(10, int(round(120 / max(0.5, dist_to_goal))))
                    c0 = max(0, center_col - col_span)
                    c1 = min(W, center_col + col_span)
                    r0 = int(round(H * 0.45))
                    r1 = int(round(H * 0.75))
                    rgb_img[r0:r1, c0:c1] = [140, 70, 20]  # Distinctive furniture / table tone

            depth_views[v_idx] = depth_img
            rgb_views[v_idx] = rgb_img

        return StepObservation(
            rgb_views=rgb_views,
            depth_views=depth_views,
            allocentric_pose=(self.x, self.z, self.yaw_rad),
            simulator_actions_taken=self.action_count,
            is_done=(self.action_count >= self.max_actions),
        )

    def get_geodesic_distance_to_goal(self) -> float:
        return float(math.sqrt((self.goal_x - self.x) ** 2 + (self.goal_z - self.z) ** 2))

    def get_shortest_path_distance(self) -> float:
        return float(self.shortest_path_len)

    def close(self) -> None:
        pass


class HabitatGroundEmbodiment(EmbodimentBase):
    """Production Habitat-Sim embodiment for Matterport3D ground robot evaluation."""

    def __init__(
        self,
        scene_id: str,
        camera_config: Optional[CameraConfig] = None,
        max_actions: int = 500,
        gpu_device_id: int = -1,
    ):
        if not HABITAT_AVAILABLE:
            raise RuntimeError("habitat_sim is not installed in this environment.")

        self.config = camera_config or CameraConfig()
        self.max_actions = max_actions
        self.scene_id = scene_id
        self.gpu_device_id = gpu_device_id
        self.sim: Optional[habitat_sim.Simulator] = None
        self.action_count = 0
        self.current_goal_pos: Optional[np.ndarray] = None
        self.shortest_path_len = 0.0

        self._init_simulator()

    def _init_simulator(self) -> None:
        """Configures Habitat-Sim with 4 RGB and 4 Depth sensors mounted at 1.25m."""
        resolved_scene = resolve_scene_path(self.scene_id)
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = resolved_scene
        sim_cfg.gpu_device_id = self.gpu_device_id
        sim_cfg.enable_physics = False

        sensor_specs = []
        for v_idx, psi_deg in VIEW_AZIMUTHS_DEG.items():
            # RGB sensor
            rgb_spec = habitat_sim.CameraSensorSpec()
            rgb_spec.uuid = f"rgb_view_{v_idx}"
            rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
            rgb_spec.resolution = [self.config.image_height, self.config.image_width]
            rgb_spec.hfov = self.config.hfov_deg
            rgb_spec.position = [0.0, self.config.camera_height, 0.0]
            rgb_spec.orientation = [0.0, math.radians(psi_deg), 0.0]
            sensor_specs.append(rgb_spec)

            # Depth sensor
            depth_spec = habitat_sim.CameraSensorSpec()
            depth_spec.uuid = f"depth_view_{v_idx}"
            depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
            depth_spec.resolution = [self.config.image_height, self.config.image_width]
            depth_spec.hfov = self.config.hfov_deg
            depth_spec.position = [0.0, self.config.camera_height, 0.0]
            depth_spec.orientation = [0.0, math.radians(psi_deg), 0.0]
            sensor_specs.append(depth_spec)

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = sensor_specs
        agent_cfg.action_space = {
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
            ),
            "turn_left": habitat_sim.agent.ActionSpec(
                "turn_left", habitat_sim.agent.ActuationSpec(amount=15.0)
            ),
            "turn_right": habitat_sim.agent.ActionSpec(
                "turn_right", habitat_sim.agent.ActuationSpec(amount=15.0)
            ),
        }

        cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
        self.sim = habitat_sim.Simulator(cfg)
        pf = self.sim.pathfinder
        if not pf.is_loaded:
            navmesh_path = resolved_scene.replace(".glb", ".navmesh")
            if os.path.exists(navmesh_path):
                pf.load_nav_mesh(navmesh_path)

    def reset(self, episode_data: Dict[str, Any]) -> StepObservation:
        self.action_count = 0
        agent = self.sim.get_agent(0)
        pf = self.sim.pathfinder

        start_pos = episode_data.get("start_position", [0.0, 0.0, 0.0])
        start_rot = episode_data.get("start_rotation", [0.0, 0.0, 0.0, 1.0])
        if "goals" in episode_data and len(episode_data["goals"]) > 0:
            goal_pos = episode_data["goals"][0]["position"]
        elif "goal_position" in episode_data:
            goal_pos = episode_data["goal_position"]
        elif "reference_path" in episode_data and len(episode_data["reference_path"]) > 0:
            goal_pos = episode_data["reference_path"][-1]
        else:
            goal_pos = [0.0, 0.0, -10.0]

        # Robust snapping to navmesh
        if pf.is_loaded:
            sp_cand = pf.snap_point(np.array(start_pos, dtype=np.float32))
            if not np.isnan(sp_cand[0]):
                start_pos = [float(sp_cand[0]), float(sp_cand[1]), float(sp_cand[2])]

            gp_cand = pf.snap_point(np.array(goal_pos, dtype=np.float32))
            if np.isnan(gp_cand[0]):
                bounds = pf.get_bounds()
                clamped = np.array(goal_pos, dtype=np.float32)
                clamped[0] = np.clip(clamped[0], bounds[0][0] + 0.3, bounds[1][0] - 0.3)
                clamped[1] = np.clip(clamped[1], bounds[0][1], bounds[1][1])
                clamped[2] = np.clip(clamped[2], bounds[0][2] + 0.3, bounds[1][2] - 0.3)
                gp_cand = pf.snap_point(clamped)
            if not np.isnan(gp_cand[0]):
                goal_pos = [float(gp_cand[0]), float(gp_cand[1]), float(gp_cand[2])]

        self.current_goal_pos = np.array(goal_pos, dtype=np.float64)
        agent.set_state(habitat_sim.agent.AgentState(position=start_pos, rotation=start_rot))

        # Shortest path distance via PathFinder (oracle evaluation only)
        path = habitat_sim.ShortestPath()
        path.requested_start = np.array(start_pos, dtype=np.float32)
        path.requested_end = np.array(goal_pos, dtype=np.float32)
        found = pf.find_path(path)
        self.shortest_path_len = float(path.geodesic_distance) if found else float(np.linalg.norm(np.array(goal_pos) - np.array(start_pos)))

        return self._extract_obs()

    def step_discrete(self, turn_steps_15deg: int, forward_steps_025m: int) -> StepObservation:
        pos_before = np.array(self.sim.get_agent(0).get_state().position, copy=True)
        prim_traj: List[Tuple[float, float, float]] = []
        prim_geod: List[float] = []
        collided = False

        def record_substep():
            state = self.sim.get_agent(0).get_state()
            pos = state.position
            rot = state.rotation
            yaw = 2.0 * math.atan2(rot.y, rot.w)
            prim_traj.append((float(pos[0]), float(pos[2]), float(yaw)))
            prim_geod.append(self.get_geodesic_distance_to_goal())

        # Execute turn actions
        if turn_steps_15deg != 0:
            turn_action = "turn_left" if turn_steps_15deg > 0 else "turn_right"
            for _ in range(abs(turn_steps_15deg)):
                self.sim.step(turn_action)
                self.action_count += 1
                record_substep()

        # Execute forward actions
        for _ in range(max(0, forward_steps_025m)):
            pos_pre = np.array(self.sim.get_agent(0).get_state().position, copy=True)
            self.sim.step("move_forward")
            self.action_count += 1
            pos_post = self.sim.get_agent(0).get_state().position
            if np.linalg.norm(pos_post - pos_pre) < 0.05:
                collided = True
            record_substep()

        pos_after = self.sim.get_agent(0).get_state().position
        actual_dist = float(np.linalg.norm(pos_after - pos_before))

        obs = self._extract_obs()
        obs.primitive_trajectory = prim_traj
        obs.primitive_geodesic_distances = prim_geod
        obs.actual_distance_moved = actual_dist
        obs.collided = collided
        return obs

    def _extract_obs(self) -> StepObservation:
        raw_obs = self.sim.get_sensor_observations()
        rgb_views = {}
        depth_views = {}

        for v_idx in VIEW_AZIMUTHS_DEG:
            rgb_views[v_idx] = raw_obs[f"rgb_view_{v_idx}"][:, :, :3]
            # Depth sensor returns planar z_c in meters
            d = raw_obs[f"depth_view_{v_idx}"]
            depth_views[v_idx] = np.clip(d, self.config.d_min, self.config.d_max)

        state = self.sim.get_agent(0).get_state()
        pos = state.position
        # Extract yaw from quaternion
        rot = state.rotation
        yaw = 2.0 * math.atan2(rot.y, rot.w)

        return StepObservation(
            rgb_views=rgb_views,
            depth_views=depth_views,
            allocentric_pose=(float(pos[0]), float(pos[2]), float(yaw)),
            simulator_actions_taken=self.action_count,
            is_done=(self.action_count >= self.max_actions),
        )

    def get_geodesic_distance_to_goal(self) -> float:
        if self.current_goal_pos is None:
            return 0.0
        agent_pos = self.sim.get_agent(0).get_state().position
        path = habitat_sim.ShortestPath()
        path.requested_start = np.array(agent_pos, dtype=np.float32)
        path.requested_end = np.array(self.current_goal_pos, dtype=np.float32)
        self.sim.pathfinder.find_path(path)
        return float(path.geodesic_distance)

    def get_shortest_path_distance(self) -> float:
        return float(self.shortest_path_len)

    def close(self) -> None:
        if self.sim is not None:
            self.sim.close()
            self.sim = None

