"""
Iteration 6 Master Dual-Rate Controller
=======================================
Coordinates high-rate physical control (System 1 at 20-25 Hz) with
deliberative cognitive reasoning (System 2 at 0.1-0.5 Hz).

Enforces the COMPULSORY FEEDBACK INVARIANT:
- System 2 cannot be bypassed or overridden.
- If a Subgoal expires (now - issued_at > expiry_s), System 1 forces HOVER_REPLAN
  and awakens System 2.
"""

from __future__ import annotations
import math
import time
import asyncio
import threading
import logging
from typing import Optional, List, Dict, Any
import numpy as np

from contracts import (
    DroneTelemetry,
    CameraIntrinsics,
    SubgoalIntent,
    Subgoal,
    SafetyState,
)
from adapters.base_adapter import ContinuousUAVController
from cognitive.baseline_agents import BaseVLNAgent
from core.depth_regulator import DepthOdometryRegulator
from core.kinematics import JerkLimitedIntegrator, APFPlanner
from core.tracker_memory import TargetMemoryTrackerV6
from core.safety_fsm import SafetyFallbackFSM

logger = logging.getLogger("MavicVLM")


class SharedBlackboard:
    """Thread-safe state blackboard shared between Fast and Slow loops."""

    def __init__(self, initial_subgoal: Subgoal):
        self.lock = threading.Lock()
        self.current_subgoal: Subgoal = initial_subgoal
        self.latest_telemetry: Optional[DroneTelemetry] = None
        self.latest_rgb: Optional[np.ndarray] = None
        self.latest_depth: Optional[np.ndarray] = None
        self.active_safety_state = SafetyState.NORMAL
        self.trigger_event = threading.Event()
        self.stop_requested = False

    def update_telemetry_and_perception(
        self,
        telemetry: DroneTelemetry,
        rgb: np.ndarray,
        depth: np.ndarray,
    ) -> None:
        with self.lock:
            self.latest_telemetry = telemetry
            self.latest_rgb = rgb
            self.latest_depth = depth

    def get_current_subgoal(self) -> Subgoal:
        with self.lock:
            return self.current_subgoal

    def set_subgoal(self, subgoal: Subgoal) -> None:
        with self.lock:
            self.current_subgoal = subgoal

    def request_stop(self) -> None:
        with self.lock:
            self.stop_requested = True
            self.trigger_event.set()

    def is_stop_requested(self) -> bool:
        with self.lock:
            return self.stop_requested


class FastContinuousController:
    """System 1: High-rate (20-25 Hz) continuous reactive flight loop."""

    def __init__(
        self,
        adapter: ContinuousUAVController,
        blackboard: SharedBlackboard,
        scene_graph: Optional[Any] = None,
        tick_hz: float = 20.0,
    ):
        self.adapter = adapter
        self.blackboard = blackboard
        self.scene_graph = scene_graph if scene_graph is not None else getattr(adapter, "scene_graph", None)
        self.dt = 1.0 / tick_hz

        # Core physical infrastructure
        self.depth_regulator = DepthOdometryRegulator(d_margin=1.5, d_slowdown=8.0)
        self.apf_planner = APFPlanner(k_att=1.0, k_rep=8.0, d_0=6.0, v_max=4.0)
        self.integrator = JerkLimitedIntegrator(a_max=2.5, j_max=6.0)
        self.tracker_memory = TargetMemoryTrackerV6()
        self.safety_fsm = SafetyFallbackFSM(emergency_clearance_m=1.5)

        # Kinematic state
        self.current_acc = np.zeros(3, dtype=np.float64)
        self.last_keyframe_pos = np.zeros(3, dtype=np.float64)
        self.last_keyframe_yaw = 0.0

    def tick(self) -> None:
        """Executes a single high-frequency tick of System 1."""
        telemetry = self.adapter.get_telemetry()
        rgb, depth, intrinsics = self.adapter.get_camera_frame()
        self.blackboard.update_telemetry_and_perception(telemetry, rgb, depth)

        subgoal = self.blackboard.get_current_subgoal()

        # 1. COMPULSORY FEEDBACK INVARIANT ENFORCEMENT
        # Check heartbeat expiry of active subgoal
        now = telemetry.timestamp
        elapsed = now - subgoal.issued_at
        comm_timeout = False
        if elapsed > subgoal.expiry_s:
            comm_timeout = True
            self.blackboard.trigger_event.set()

        # 2. In-Loop Depth-Odometry Regulation
        zones = self.depth_regulator.compute_zone_depths(depth)
        d_frontal = self.depth_regulator.frontal_clearance(zones)

        # 3. Target Memory Tracking (active only when tracking an explicit target node)
        sg = self.scene_graph if self.scene_graph is not None else getattr(self.adapter, "scene_graph", None)
        if subgoal.target_node_id is not None and sg is not None:
            if self.tracker_memory.active_target_id != subgoal.target_node_id:
                self.tracker_memory.set_active_target(subgoal.target_node_id, sg)
            tracker_action = self.tracker_memory.update(
                target_visible=False,
                bbox_norm=None,
                scene_graph=sg,
                telemetry=telemetry,
            )
        else:
            self.tracker_memory.active_target_id = None
            empty_sg = sg if sg is not None else type("EmptySceneGraph", (), {"nodes": {}})()
            tracker_action = self.tracker_memory.update(False, None, empty_sg, telemetry) # type: ignore

        # 4. Safety Fallback FSM
        safety_state, override_v, override_yaw = self.safety_fsm.step(
            d_frontal=d_frontal,
            tracker_action=tracker_action,
            comm_timeout=comm_timeout,
        )
        self.blackboard.active_safety_state = safety_state

        if override_v is not None:
            # Safety override engaged (emergency repulsion, reacquisition, or failsafe hover)
            v_cmd = override_v
            yaw_rate_cmd = override_yaw if override_yaw is not None else 0.0
        else:
            # 5. Normal Flight Planning via 3D APF & Depth Scaling
            waypoint = np.array(subgoal.waypoint_xyz, dtype=np.float64)
            raw_v_cmd = self.apf_planner.compute_force(
                current_pos=telemetry.position,
                target_waypoint=waypoint,
                zone_depths=zones,
                uav_yaw=telemetry.yaw,
            )

            # Scale forward speed based on physical clearance
            speed_desired = float(np.linalg.norm(raw_v_cmd))
            speed_scaled = self.depth_regulator.scale_forward_velocity(
                v_desired=speed_desired,
                d_frontal=d_frontal,
                current_speed=telemetry.speed,
            )
            v_cmd = (raw_v_cmd / max(1e-3, speed_desired)) * speed_scaled if speed_desired > 1e-3 else raw_v_cmd

            # Gimbal pitch servoing toward waypoint
            dz = waypoint[2] - telemetry.altitude
            d_horiz = math.hypot(waypoint[0] - telemetry.position[0], waypoint[1] - telemetry.position[1])
            gimbal_pitch = float(np.clip(-math.atan2(dz, max(1.0, d_horiz)), 0.0, 0.8))
            self.adapter.set_gimbal_angles(pitch=gimbal_pitch)

            # Align yaw with waypoint direction
            desired_yaw = math.atan2(waypoint[1] - telemetry.position[1], waypoint[0] - telemetry.position[0])
            yaw_diff = (desired_yaw - telemetry.yaw + math.pi) % (2.0 * math.pi) - math.pi
            yaw_rate_cmd = float(np.clip(yaw_diff * 1.5, -1.0, 1.0))

        # 6. Apply Jerk-Limited Integration & Command Adapter
        _, new_vel, self.current_acc = self.integrator.step(
            pos=telemetry.position,
            vel=telemetry.velocity,
            acc=self.current_acc,
            v_cmd=v_cmd,
            dt=self.dt,
        )
        self.adapter.set_continuous_velocity(new_vel[0], new_vel[1], new_vel[2], yaw_rate=yaw_rate_cmd)
        self.adapter.step_simulation(self.dt)

        # 7. Check Keyframe Triggers for System 2
        dist_moved = float(np.linalg.norm(telemetry.position - self.last_keyframe_pos))
        yaw_rotated = abs((telemetry.yaw - self.last_keyframe_yaw + math.pi) % (2.0 * math.pi) - math.pi)
        dist_to_wpt = float(np.linalg.norm(telemetry.position - np.array(subgoal.waypoint_xyz)))

        if dist_moved >= 8.0 or yaw_rotated >= math.radians(45.0) or dist_to_wpt <= 2.0:
            self.last_keyframe_pos = telemetry.position.copy()
            self.last_keyframe_yaw = telemetry.yaw
            self.blackboard.trigger_event.set()


class DualRateMasterController:
    """Master controller coordinating System 1 and System 2 threads."""

    def __init__(
        self,
        agent: BaseVLNAgent,
        adapter: ContinuousUAVController,
        mission_instruction: str,
        initial_subgoal: Optional[Subgoal] = None,
    ):
        self.agent = agent
        self.adapter = adapter
        self.instruction = mission_instruction

        if initial_subgoal is None:
            init_tel = adapter.get_telemetry()
            initial_subgoal = Subgoal(
                subgoal_id=0,
                intent=SubgoalIntent.SEARCH_AND_ORIENT,
                waypoint_xyz=(init_tel.position[0], init_tel.position[1], init_tel.position[2] + 2.0),
                issued_at=init_tel.timestamp,
                expiry_s=6.0,
            )

        self.blackboard = SharedBlackboard(initial_subgoal)
        agent_sg = getattr(agent, "scene_graph", None)
        self.fast_controller = FastContinuousController(adapter, self.blackboard, scene_graph=agent_sg)
        self.fast_thread: Optional[threading.Thread] = None

    def _fast_loop_worker(self) -> None:
        while not self.blackboard.is_stop_requested():
            self.fast_controller.tick()
            time.sleep(self.fast_controller.dt)

    def start_fast_loop(self) -> None:
        self.fast_thread = threading.Thread(target=self._fast_loop_worker, daemon=True)
        self.fast_thread.start()

    def stop(self) -> None:
        self.blackboard.request_stop()
        if self.fast_thread and self.fast_thread.is_alive():
            self.fast_thread.join(timeout=1.0)

    async def run_slow_step(self) -> Subgoal:
        """Executes a single asynchronous System 2 deliberative reasoning step."""
        with self.blackboard.lock:
            tel = self.blackboard.latest_telemetry
            rgb = self.blackboard.latest_rgb
            depth = self.blackboard.latest_depth

        if tel is None or rgb is None or depth is None:
            tel = self.adapter.get_telemetry()
            rgb, depth, _ = self.adapter.get_camera_frame()

        try:
            new_subgoal = await self.agent.step(rgb, depth, tel, self.instruction)
        except Exception as e:
            logger.error(f"[DualRateMasterController] System 2 deliberative step failed: {e}. Enforcing safe HOVER_REPLAN fallback.")
            curr_sg = self.blackboard.get_current_subgoal()
            new_subgoal = Subgoal(
                subgoal_id=curr_sg.subgoal_id + 1,
                intent=SubgoalIntent.SEARCH_AND_ORIENT,
                waypoint_xyz=(tel.position[0], tel.position[1], tel.altitude),
                v_target=0.0,  # Station-keeping hover failsafe
                caution_level="high",
                termination_condition="replan_timeout",
                issued_at=tel.timestamp,
                expiry_s=3.0,
            )
        self.blackboard.set_subgoal(new_subgoal)
        return new_subgoal
