"""End-to-end episode execution pipeline for AgenticNav replication (v7 Specification)."""

from dataclasses import dataclass
import math
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from agenticnav_paper_replica.configs.default_config import ReplicaConfig
from agenticnav_paper_replica.embodiment.base import EmbodimentBase, StepObservation
from agenticnav_paper_replica.geometry.pinhole import (
    CameraConfig,
    unproject_depth_image,
    extract_view_obstacle_and_free_points,
)
from agenticnav_paper_replica.geometry.safety_region import GroundCorridorChecker
from agenticnav_paper_replica.harness.executor import ToolExecutor
from agenticnav_paper_replica.harness.step_loop import StepDecisionLoop, StepLoopResult
from agenticnav_paper_replica.mapping.occupancy_grid import OccupancyGrid2D
from agenticnav_paper_replica.mapping.bev_renderer import BEVRenderer
from agenticnav_paper_replica.memory.history import ActionHistory
from agenticnav_paper_replica.memory.image_cache import EpisodicImageCache
from agenticnav_paper_replica.evaluation.metrics import compute_episode_metrics, EpisodeMetrics
from agenticnav_paper_replica.evaluation.failure_classifier import classify_failure
from agenticnav_paper_replica.traces.trace_logger import TraceLogger, EpisodeTrace, StepTrace


@dataclass
class EpisodeRunResult:
    episode_id: str
    metrics: EpisodeMetrics
    failure_category: Optional[str]
    total_steps: int
    called_stop: bool
    terminated_deadlock: bool
    terminated_protocol_violation: bool = False
    trace: Optional[EpisodeTrace] = None
    final_bev_image: Optional[np.ndarray] = None


def run_episode(
    episode_data: Dict[str, Any],
    embodiment: EmbodimentBase,
    config: ReplicaConfig,
    vlm_client_fn: Callable[[List[Dict[str, Any]], List[Dict[str, Any]]], Dict[str, Any]],
    trace_logger: Optional[TraceLogger] = None,
    step_evaluator: Optional[Any] = None,
    step_sensor_callback: Optional[Callable[[int, Dict[int, np.ndarray], Dict[int, np.ndarray], np.ndarray], None]] = None,
) -> EpisodeRunResult:
    """Runs a single navigation episode end-to-end with strict privilege firewall enforcement.

    Privilege Firewall Guarantees:
        - vlm_client_fn receives only RGB views, BEV map, and recent action history.
        - Neither the agent nor the decision loop accesses the goal coordinates or NavMesh.
        - Embodiment oracle distances are queried strictly at the end for evaluation.
    """
    episode_id = str(episode_data.get("episode_id", "ep_000"))
    scene_id = str(episode_data.get("scene_id", "scene"))
    instruction = str(episode_data.get("instruction", ""))

    # 1. Initialize sensor and embodiment configuration
    camera_cfg = CameraConfig(
        image_width=config.image_width,
        image_height=config.image_height,
        hfov_deg=config.hfov_deg,
        d_min=config.d_min,
        d_max=config.d_max,
        min_obstacle_depth=getattr(config, "min_obstacle_depth", 0.05),
        camera_height=config.camera_height,
    )
    corridor_checker = GroundCorridorChecker(
        robot_radius=config.robot_radius,
        base_margin=config.base_margin,
        rho_max=config.rho_max,
        turn_step_deg=config.turn_step_deg,
        forward_step_m=config.forward_step_m,
        zero_rho_policy=config.zero_rho_policy,
        allow_adjacent_bin_snap=config.adjacent_bin_snap,
        max_snap_deg=config.max_snap_deg,
        max_snap_drift_m=config.max_snap_drift_m,
        obstacle_point_threshold=getattr(config, "obstacle_point_threshold", 5),
    )

    # Memory, mapping, and harness
    image_cache = EpisodicImageCache()
    history = ActionHistory(max_history=config.max_history_steps, mode=config.history_mode)
    occupancy_grid = OccupancyGrid2D(grid_size=config.bev_grid_size, resolution_m=config.bev_resolution_m)
    bev_renderer = BEVRenderer(mode="without_bev" if not config.bev_image else "full")
    tool_executor = ToolExecutor(
        camera_cfg,
        corridor_checker,
        image_cache,
        standoff_mode=config.standoff_mode,
        oracle_local_actuation=getattr(config, "oracle_local_actuation", False),
    )
    step_loop = StepDecisionLoop(
        executor=tool_executor,
        max_evidence_queries=config.max_evidence_queries_per_step,
        max_consecutive_reselects=config.max_consecutive_reselects,
        max_malformed_calls=config.max_malformed_calls_per_step,
        enable_depth=(config.depth_tool != "off"),
        enable_recall=config.recall_tool,
        enable_arrival_verification=getattr(config, "arrival_verification", True),
        floor_grounding_prompt=getattr(config, "floor_grounding_prompt", True),
    )

    # 2. Reset simulator to start pose
    obs: StepObservation = embodiment.reset(episode_data)
    current_node = 1
    image_cache.set_current_node(current_node)
    image_cache.store_node_observations(current_node, obs.rgb_views)

    trajectory: List[Tuple[float, float, float]] = [obs.position]
    node_poses: List[Tuple[float, float, float]] = [obs.allocentric_pose]
    geodesic_distances: List[float] = [embodiment.get_geodesic_distance_to_goal()]

    called_stop = False
    terminated_deadlock = False
    terminated_protocol_violation = False
    consecutive_reselects = 0
    step_traces: List[Dict[str, Any]] = []

    # 3. Main decision loop
    max_decision_steps = getattr(config, "max_decision_steps", 25)
    while (
        not obs.is_done
        and not called_stop
        and not terminated_deadlock
        and not terminated_protocol_violation
        and current_node <= max_decision_steps
    ):
        # Extract 3D obstacle pointcloud and 2D free-space endpoints from current depth views
        obstacle_pts_list = []
        free_pts_list = []
        for v_idx, d_img in obs.depth_views.items():
            pts_obs, pts_free = extract_view_obstacle_and_free_points(
                d_img,
                view_idx=v_idx,
                config=camera_cfg,
                stride=4,
                y_min=config.chassis_y_min,
                y_max=getattr(config, "chassis_y_max_3d", 2.50),
            )
            if len(pts_obs) > 0:
                obstacle_pts_list.append(pts_obs)
            if len(pts_free) > 0:
                free_pts_list.append(pts_free)

        ax, az, ayaw = obs.allocentric_pose
        cos_a = math.cos(ayaw)
        sin_a = math.sin(ayaw)

        if obstacle_pts_list:
            cur_obs_agent_xyz = np.vstack(obstacle_pts_list)
            ox_agent = cur_obs_agent_xyz[:, 0]
            oy_agent = cur_obs_agent_xyz[:, 1]
            oz_agent = cur_obs_agent_xyz[:, 2]

            # 2D BEV map: filter strictly to current floor horizontal slice [chassis_y_min, chassis_y_max]
            bev_mask = (oy_agent >= config.chassis_y_min) & (oy_agent <= config.chassis_y_max)
            if np.any(bev_mask):
                ox_bev = ox_agent[bev_mask]
                oz_bev = oz_agent[bev_mask]
                ox_world = ax + cos_a * ox_bev + sin_a * oz_bev
                oz_world = az - sin_a * ox_bev + cos_a * oz_bev
                obs_world_xz = np.column_stack([ox_world, oz_world])
                obs_agent_xz = np.column_stack([ox_bev, oz_bev])
            else:
                obs_world_xz = np.empty((0, 2), dtype=np.float64)
                obs_agent_xz = np.empty((0, 2), dtype=np.float64)

            # Full 3D point cloud for 3D swept cylinder collision checking
            obs_agent_xyz = cur_obs_agent_xyz
        else:
            obs_world_xz = np.empty((0, 2), dtype=np.float64)
            obs_agent_xz = np.empty((0, 2), dtype=np.float64)
            obs_agent_xyz = np.empty((0, 3), dtype=np.float64)

        if free_pts_list:
            cur_free_agent_xz = np.vstack(free_pts_list)
            fx_agent = cur_free_agent_xz[:, 0]
            fz_agent = cur_free_agent_xz[:, 1]
            fx_world = ax + cos_a * fx_agent + sin_a * fz_agent
            fz_world = az - sin_a * fx_agent + cos_a * fz_agent
            free_world_xz = np.column_stack([fx_world, fz_world])
        else:
            free_world_xz = None

        occupancy_grid.update(
            agent_x_m=ax,
            agent_z_m=az,
            obstacle_points_world_xz=obs_world_xz,
            free_endpoints_world_xz=free_world_xz,
        )

        # Render BEV map
        bev_img = bev_renderer.render(occupancy_grid, node_poses=node_poses, current_node_idx=current_node)

        # Compute accumulated odometry distance traversed along trajectory
        total_dist_m = 0.0
        if len(trajectory) >= 2:
            for ti in range(len(trajectory) - 1):
                tdx = trajectory[ti + 1][0] - trajectory[ti][0]
                tdz = trajectory[ti + 1][1] - trajectory[ti][1]
                total_dist_m += math.sqrt(tdx * tdx + tdz * tdz)

        # Check for trajectory loop / spatial revisitation (Task 4)
        loop_warning: Optional[str] = None
        if len(node_poses) >= 3:
            cur_x, cur_z = obs.allocentric_pose[0], obs.allocentric_pose[1]
            min_prev_dist = float("inf")
            revisited_step = -1
            for prev_idx in range(len(node_poses) - 2):
                px, pz, _ = node_poses[prev_idx]
                dist_prev = math.sqrt((cur_x - px) ** 2 + (cur_z - pz) ** 2)
                if dist_prev < min_prev_dist:
                    min_prev_dist = dist_prev
                    revisited_step = prev_idx

            if min_prev_dist < 1.20:
                loop_warning = (
                    f"[LOOP WARNING: You are currently within {min_prev_dist:.2f}m of where you were at Step {revisited_step}. "
                    "You are pacing or oscillating in an area you have already visited. "
                    "(1) If you are at/near the destination landmark described in the instruction, CALL STOP() NOW instead of pacing. "
                    "(2) If not at the destination, DO NOT retrace your prior path: pick a completely different, unexplored direction/doorway from the BEV map!]"
                )

        pre_pose = obs.allocentric_pose
        pre_goal_dist = embodiment.get_geodesic_distance_to_goal()

        if step_sensor_callback:
            step_sensor_callback(current_node, obs.rgb_views, obs.depth_views, bev_img)

        # Run multi-turn step decision
        step_result: StepLoopResult = step_loop.run_step(
            instruction=instruction,
            current_node=current_node,
            history=history,
            rgb_views=obs.rgb_views,
            depth_views=obs.depth_views,
            obstacle_points_agent_xz=obs_agent_xz,
            bev_image=bev_img,
            vlm_client_fn=vlm_client_fn,
            initial_reselect_count=consecutive_reselects,
            accumulated_distance_m=total_dist_m,
            obstacle_points_agent_xyz=obs_agent_xyz,
            loop_warning=loop_warning,
        )

        step_traces.append({
            "step_idx": current_node,
            "node_idx": current_node,
            "agent_pose": {"x": obs.allocentric_pose[0], "z": obs.allocentric_pose[1], "yaw_rad": obs.allocentric_pose[2]},
            "geodesic_distance_to_goal_m": embodiment.get_geodesic_distance_to_goal(),
            "view_zero_depth_fractions": step_result.view_zero_depth_fractions,
            "turns": step_result.turn_log,
            "action_type": step_result.action_type,
            "action_payload": step_result.action_payload,
            "arrival_verification_triggered": step_result.arrival_verification_triggered,
        })

        if step_result.action_type == "stop":
            called_stop = True
            history.append(
                step_idx=current_node,
                from_node=current_node,
                to_node=current_node,
                action_type="stop",
                action_params=step_result.action_payload,
                reason=step_result.action_payload.get("reason", ""),
            )
            if step_evaluator:
                step_evaluator.evaluate_step(
                    step_idx=current_node,
                    node_idx=current_node,
                    instruction=instruction,
                    pre_pose=pre_pose,
                    post_pose=obs.allocentric_pose,
                    pre_goal_dist_m=pre_goal_dist,
                    post_goal_dist_m=embodiment.get_geodesic_distance_to_goal(),
                    accumulated_dist_m=total_dist_m,
                    depth_views=obs.depth_views,
                    bev_image=bev_img,
                    occupancy_grid=occupancy_grid,
                    turn_log=step_result.turn_log,
                    action_type="stop",
                    action_payload=step_result.action_payload,
                    actual_moved_m=0.0,
                    collided=False,
                )
            break

        elif step_result.action_type == "deadlock":
            terminated_deadlock = True
            if step_evaluator:
                step_evaluator.evaluate_step(
                    step_idx=current_node,
                    node_idx=current_node,
                    instruction=instruction,
                    pre_pose=pre_pose,
                    post_pose=obs.allocentric_pose,
                    pre_goal_dist_m=pre_goal_dist,
                    post_goal_dist_m=embodiment.get_geodesic_distance_to_goal(),
                    accumulated_dist_m=total_dist_m,
                    depth_views=obs.depth_views,
                    bev_image=bev_img,
                    occupancy_grid=occupancy_grid,
                    turn_log=step_result.turn_log,
                    action_type="deadlock",
                    action_payload=step_result.action_payload,
                    actual_moved_m=0.0,
                    collided=False,
                )
            break

        elif step_result.action_type == "protocol_violation":
            terminated_protocol_violation = True
            if step_evaluator:
                step_evaluator.evaluate_step(
                    step_idx=current_node,
                    node_idx=current_node,
                    instruction=instruction,
                    pre_pose=pre_pose,
                    post_pose=obs.allocentric_pose,
                    pre_goal_dist_m=pre_goal_dist,
                    post_goal_dist_m=embodiment.get_geodesic_distance_to_goal(),
                    accumulated_dist_m=total_dist_m,
                    depth_views=obs.depth_views,
                    bev_image=bev_img,
                    occupancy_grid=occupancy_grid,
                    turn_log=step_result.turn_log,
                    action_type="protocol_violation",
                    action_payload=step_result.action_payload,
                    actual_moved_m=0.0,
                    collided=False,
                )
            break

        elif step_result.action_type == "move_to":
            if getattr(config, "oracle_local_actuation", False):
                # Habitat Oracle Local Actuation (Arm O)
                p_agent = step_result.action_payload.get("p_agent")
                if p_agent is not None:
                    px, py, pz = p_agent
                    ax, az, ayaw = obs.allocentric_pose
                    cur_pos = obs.position
                    x_world = cur_pos[0] + math.cos(ayaw) * px + math.sin(ayaw) * pz
                    y_world = cur_pos[1] + py
                    z_world = cur_pos[2] - math.sin(ayaw) * px + math.cos(ayaw) * pz
                    target_world = np.array([x_world, y_world, z_world], dtype=np.float32)
                else:
                    theta_deg = step_result.action_payload.get("theta_deg", 0.0)
                    rho_m = step_result.action_payload.get("rho_m", 1.0)
                    th_rad = math.radians(theta_deg)
                    px = -rho_m * math.sin(th_rad)
                    pz = -rho_m * math.cos(th_rad)
                    ax, az, ayaw = obs.allocentric_pose
                    cur_pos = obs.position
                    x_world = cur_pos[0] + math.cos(ayaw) * px + math.sin(ayaw) * pz
                    y_world = cur_pos[1]
                    z_world = cur_pos[2] - math.sin(ayaw) * px + math.cos(ayaw) * pz
                    target_world = np.array([x_world, y_world, z_world], dtype=np.float32)

                max_step = step_result.action_payload.get("rho_m", 1.75)
                obs = embodiment.step_waypoint_navmesh(target_world_pos=target_world, max_step_distance_m=max_step)
            else:
                turn_steps = step_result.action_payload.get("turn_steps", 0)
                forward_steps = step_result.action_payload.get("forward_steps", 0)
                obs = embodiment.step_discrete(turn_steps_15deg=turn_steps, forward_steps_025m=forward_steps)

            # Record per-primitive steps for standard Habitat metric evaluation (OSR, TL, SPL)
            if obs.primitive_trajectory:
                trajectory.extend(obs.primitive_trajectory)
                geodesic_distances.extend(obs.primitive_geodesic_distances)
            else:
                trajectory.append(obs.position)
                geodesic_distances.append(embodiment.get_geodesic_distance_to_goal())

            # Ground move outcome with actual simulator physics
            outcome_dict = dict(step_result.action_payload)
            outcome_dict["moved_m"] = round(obs.actual_distance_moved, 2)
            outcome_dict["collided"] = obs.collided

            if step_traces:
                step_traces[-1]["post_geodesic_distance_to_goal_m"] = embodiment.get_geodesic_distance_to_goal()
                step_traces[-1]["outcome"] = outcome_dict
                step_traces[-1]["post_agent_pose"] = {
                    "x": obs.allocentric_pose[0],
                    "z": obs.allocentric_pose[1],
                    "yaw_rad": obs.allocentric_pose[2],
                }

            if step_evaluator:
                step_evaluator.evaluate_step(
                    step_idx=current_node,
                    node_idx=current_node,
                    instruction=instruction,
                    pre_pose=pre_pose,
                    post_pose=obs.allocentric_pose,
                    pre_goal_dist_m=pre_goal_dist,
                    post_goal_dist_m=embodiment.get_geodesic_distance_to_goal(),
                    accumulated_dist_m=total_dist_m + obs.actual_distance_moved,
                    depth_views=obs.depth_views,
                    bev_image=bev_img,
                    occupancy_grid=occupancy_grid,
                    turn_log=step_result.turn_log,
                    action_type="move_to",
                    action_payload=step_result.action_payload,
                    actual_moved_m=obs.actual_distance_moved,
                    collided=obs.collided,
                )

            next_node = current_node + 1
            history.append(
                step_idx=current_node,
                from_node=current_node,
                to_node=next_node,
                action_type="move_to",
                action_params=step_result.action_payload,
                reason=step_result.action_payload.get("reason", ""),
                outcome=outcome_dict,
            )

            current_node = next_node
            image_cache.set_current_node(current_node)
            image_cache.store_node_observations(current_node, obs.rgb_views)
            node_poses.append(obs.allocentric_pose)
            consecutive_reselects = 0

        else:
            break

    # 4. Final metrics and failure evaluation (Section 10)
    metrics: EpisodeMetrics = compute_episode_metrics(
        trajectory=trajectory,
        distances_to_goal=geodesic_distances,
        shortest_path_length=embodiment.get_shortest_path_distance(),
        simulator_action_count=obs.simulator_actions_taken,
        called_stop=called_stop,
        success_distance_threshold=config.success_distance_m,
    )

    failure_cat = classify_failure(
        success=metrics.success,
        oracle_success=metrics.oracle_success,
        called_stop=called_stop,
        final_distance_to_goal=metrics.navigation_error_m,
        action_budget_exceeded=(
            obs.simulator_actions_taken >= config.max_simulator_actions
            or current_node > max_decision_steps
        ),
        deadlock_terminated=terminated_deadlock,
        protocol_violation=terminated_protocol_violation,
        infrastructure_error=False,
        success_distance_threshold=config.success_distance_m,
    )

    # 5. Log trace
    trace = None
    if trace_logger:
        trace = EpisodeTrace(
            episode_id=episode_id,
            scene_id=scene_id,
            instruction=instruction,
            shortest_path_distance_m=embodiment.get_shortest_path_distance(),
            split=str(episode_data.get("split", "unknown")),
            trajectory_id=episode_data.get("trajectory_id"),
            start_position=episode_data.get("start_position"),
            steps=step_traces,
            metrics={
                "success": metrics.success,
                "spl": metrics.spl,
                "oracle_success": metrics.oracle_success,
                "navigation_error_m": metrics.navigation_error_m,
                "trajectory_length_m": metrics.trajectory_length_m,
                "simulator_action_count": metrics.simulator_action_count,
            },
            failure_category=failure_cat,
        )
        trace_logger.log_episode(trace)

    final_bev = bev_renderer.render(occupancy_grid, node_poses=node_poses, current_node_idx=current_node)

    return EpisodeRunResult(
        episode_id=episode_id,
        metrics=metrics,
        failure_category=failure_cat,
        total_steps=current_node,
        called_stop=called_stop,
        terminated_deadlock=terminated_deadlock,
        terminated_protocol_violation=terminated_protocol_violation,
        trace=trace,
        final_bev_image=final_bev,
    )
