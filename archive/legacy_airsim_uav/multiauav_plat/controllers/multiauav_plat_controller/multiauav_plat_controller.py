import os
import sys
import math
import time
import logging
import threading
from typing import Optional, List, Tuple, Set, Dict, Any
from controller import Supervisor
from config import (
    FlightState, ActionStep, StateTransition, MAX_SPEED, CLIMB_SPEED, TURN_SPEED,
    MIN_ALTITUDE, MAX_ALTITUDE, TARGET_PRECISION, YAW_PRECISION, GEOFENCE_RADIUS,
    MAX_ACTION_QUEUE_LEN, IMAGE_PATH, SEARCH_ALTITUDE, VLM_COOLDOWN, SEARCH_VLM_COOLDOWN,
    IMAGE_SAVE_INTERVAL, SCAN_SECTOR_SIZE, MAX_HISTORY_LEN, MOVE_SPEED,
    COORD_TURN_THRESHOLD, BATTERY_DRAIN_RATE, BATTERY_CRITICAL_PCT, IMAGE_TMP_PATH,
    JPEG_QUALITY, image_file_lock
)
from utils import sign, is_frame_uniform, normalize_angle, heading_to_cardinal
from perception import SceneGraph, VLMClient, ActionProjector, TargetBelief, ActionVerifier, VerificationResult, is_valid_target_match, TargetMemoryTracker, TargetMemory
from planning import MissionPhase, SearchPhase, NavigatePhase, ManeuverPhase, MissionPlanner, MissionPhaseType
from server import run_web_server, telemetry, mission, HistoryEntry, manual_queue_lock, manual_command_queue

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("multiauav_plat.log", mode="a")
    ]
)
logger = logging.getLogger("MultiUAVPlat")

class MavicVLM(Supervisor):
    """Webots Supervisor-based Mavic drone with VLM-driven autonomy for MultiUAV-Plat benchmark."""
    def __init__(self):
        super().__init__()
        self.time_step = int(self.getBasicTimeStep())
        self.dt = self.time_step / 1000.0
        self.robot_node = self.getSelf()
        self.trans_field = self.robot_node.getField("translation")
        self.rot_field = self.robot_node.getField("rotation")
        self.memory_tracker = TargetMemoryTracker()
        
        self.camera = self.getDevice("camera")
        self.camera.enable(self.time_step)
        self.imu = self.getDevice("inertial unit")
        self.imu.enable(self.time_step)
        self.gps = self.getDevice("gps")
        self.gps.enable(self.time_step)
        
        self.camera_pitch_motor = self.getDevice("camera pitch")
        self.cam_node = self.getFromDef("CAM")
        self.cam_rot_field = self.cam_node.getField("rotation") if self.cam_node else None
        self._gimbal_pitch = 0.0
        self.set_gimbal_pitch(0.0)

        self.motors = [
            self.getDevice("front left propeller"),
            self.getDevice("front right propeller"),
            self.getDevice("rear left propeller"),
            self.getDevice("rear right propeller")
        ]
        for m in self.motors:
            m.setPosition(float('inf'))
            m.setVelocity(0.0)

        self.x = self.y = self.alt = 0.0
        self.yaw = 0.0
        self.target_x = self.target_y = self.target_alt = self.target_yaw = 0.0
        self.battery = 100.0
        self.flight_state = FlightState.DORMANT

        self.mission_start_x = self.mission_start_y = 0.0
        self.mission_start_time = 0.0
        self.known_objects: List[str] = []
        self.step_counter = 0
        self.history: List[HistoryEntry] = []
        self.target_beliefs: Dict[str, TargetBelief] = {}
        self.scene_graph = SceneGraph()
        self.mission_phases: List[MissionPhase] = []
        self.current_phase: Optional[MissionPhase] = None
        self.revert_to_search = False
        self.mission_start_yaw = 0.0
        self.completed_targets: Set[str] = set()
        self.vlm_driven: bool = False

        self.action_queue: List[ActionStep] = []
        self.action_queue_lock = threading.Lock()
        self.vlm = VLMClient()
        self.last_vlm_call_time = 0.0
        self.current_prompt = ""
        self.current_api_provider = "lmstudio"
        self.current_api_key = ""
        self.current_model_name = ""
        self.current_prompt_id = 0

        self.arc = {
            "distance": 0.0,
            "angle": 0.0,
            "start_x": 0.0,
            "start_y": 0.0,
            "start_yaw": 0.0,
            "elapsed": 0.0,
            "duration": 1.0
        }
        self._last_image_save_time = 0.0
        self._init_trail()
        logger.info("[MultiUAV-Plat Controller] Initialized")

    def _init_trail(self) -> None:
        """Dynamically inject IndexedLineSet node to draw real-time 3D flight trajectory in Webots."""
        try:
            root_children = self.getRoot().getField("children")
            trail_wbt = """
            DEF UAV_PATH Shape {
              appearance Appearance {
                material Material {
                  diffuseColor 1 0 0
                  emissiveColor 1 0.2 0.2
                }
              }
              geometry DEF UAV_PATH_LINE IndexedLineSet {
                coord DEF UAV_PATH_COORD Coordinate { point [] }
                coordIndex []
              }
            }
            """
            root_children.importMFNodeFromString(-1, trail_wbt)
            self.path_coord_field = self.getFromDef("UAV_PATH_COORD").getField("point")
            self.path_index_field = self.getFromDef("UAV_PATH_LINE").getField("coordIndex")
            self.path_points: List[List[float]] = []
            logger.info("[MultiUAV-Plat Controller] 3D path visualization node injected")
        except Exception as e:
            logger.warning(f"[MultiUAV-Plat Controller] Could not inject 3D path node: {e}")
            self.path_coord_field = None
            self.path_index_field = None
            self.path_points = []

    def _update_trail(self) -> None:
        if not self.path_coord_field or not self.path_index_field:
            return
        if self.flight_state == FlightState.DORMANT:
            return
        current_pos = [self.x, self.y, self.alt]
        if not self.path_points or math.dist(self.path_points[-1], current_pos) >= 0.2:
            idx = len(self.path_points)
            self.path_points.append(current_pos)
            self.path_coord_field.insertMFVec3f(-1, current_pos)
            if idx > 0:
                self.path_index_field.insertMFInt32(-1, idx - 1)
                self.path_index_field.insertMFInt32(-1, idx)
                self.path_index_field.insertMFInt32(-1, -1)
                if idx % 3 == 0:
                    try:
                        sphere_wbt = f"""
                        Transform {{
                          translation {current_pos[0]:.3f} {current_pos[1]:.3f} {current_pos[2]:.3f}
                          children [
                            Shape {{
                              appearance Appearance {{
                                material Material {{ emissiveColor 1 0 0 }}
                              }}
                              geometry Sphere {{ radius 0.15 }}
                            }}
                          ]
                        }}
                        """
                        self.getRoot().getField("children").importMFNodeFromString(-1, sphere_wbt)
                    except Exception:
                        pass



    def set_gimbal_pitch(self, pitch: float) -> None:
        pitch = max(0.0, min(0.8, pitch))
        self._gimbal_pitch = pitch
        if self.cam_rot_field:
            self.cam_rot_field.setSFRotation([0.0, 1.0, 0.0, pitch])
        else:
            self.camera_pitch_motor.setPosition(pitch)

    @property
    def gimbal_pitch(self) -> float:
        return self._gimbal_pitch

    def _is_search_mission(self, prompt: str) -> bool:
        p = prompt.lower()
        search_kws = ["find", "search", "look for", "locate", "scan", "survey", "identify", "spot", "detect"]
        return any(k in p for k in search_kws)

    def _build_context_block(self, prompt: str) -> str:
        target_name = self.current_phase.target_class if self.current_phase else ""
        mem = self.memory_tracker.get_memory(target_name) if target_name else None

        mem_str = "No prior visual tracking in memory."
        if mem and mem.last_seen_timestamp > 0:
            ago = self.getTime() - mem.last_seen_timestamp
            mem_str = f"Target was last seen {ago:.1f}s ago near '{mem.last_seen_edge}' (misses: {mem.consecutive_misses})."

        cardinal = heading_to_cardinal(self.yaw)
        return (
            f"TELEMETRY: Drone at X={self.x:.1f}m, Y={self.y:.1f}m, Alt={self.alt:.1f}m, Heading={math.degrees(self.yaw):.0f}° ({cardinal}), GimbalPitch={self.gimbal_pitch:.2f} rad.\n"
            f"TARGET MEMORY CONTEXT: {mem_str}\n"
        )

    def _get_current_frame_base64(self) -> str:
        try:
            from PIL import Image
            import base64
            from io import BytesIO
            width = self.camera.getWidth()
            height = self.camera.getHeight()
            image_bytes = self.camera.getImage()
            if not image_bytes:
                return ""
            img = Image.frombytes("RGBA", (width, height), image_bytes, "raw", "BGRA")
            img = img.convert("RGB")
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=80)
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
        except Exception as e:
            logger.error(f"Failed to get in-memory frame: {e}")
            return ""

    def _query_vlm_and_queue(self, prompt: str, provider: str, key: str, model_name: str) -> FlightState:
        vlm_prompt = prompt
        if self.current_phase and self.current_phase.target_class:
            target = self.current_phase.target_class
            vlm_prompt = f"Active Sub-Goal: Locate and inspect '{target}'. Full Mission Prompt: '{prompt}'"
            if self.completed_targets:
                vlm_prompt += f" (Already completed sub-goals: {list(self.completed_targets)})"
        logger.info(f"[MultiUAV-Plat] Querying VLM (step #{self.step_counter + 1}) with focus: '{vlm_prompt}'")
        context = self._build_context_block(prompt)
        frame_b64 = self._get_current_frame_base64()
        action_list, reasoning, target_visible, detected_object, box_center_x, box_center_y, confidence, expected_obs = self.vlm.query(
            vlm_prompt, provider, key, model_name, IMAGE_PATH, context, default_gimbal_pitch=self.gimbal_pitch, image_base64=frame_b64
        )

        active_target = self.current_phase.target_class if self.current_phase else ""
        verif = ActionVerifier.verify(expected_obs, target_visible, detected_object, active_target)
        logger.info(f"[MultiUAV-Plat] Stage 5 Verification: {verif.reasoning}")
        if verif.recommended_replan and self.current_phase:
            logger.warning("[MultiUAV-Plat] Stage 6 Reflection: Distraction detected during verification. Re-syncing active search phase.")
            self.revert_to_search = True

        obj_label = detected_object.strip().lower() if detected_object else ""
        if self.current_phase and self.current_phase.target_class:
            target = self.current_phase.target_class.lower()
            if target_visible and not obj_label:
                obj_label = target
            if obj_label and is_valid_target_match(obj_label, target) and confidence >= 0.3:
                if not target_visible:
                    logger.info(f"[MultiUAV-Plat] Target/Feature '{obj_label}' identified in vision feed (confidence: {confidence:.2f}). Setting target_visible=True for active phase '{target}'.")
                    target_visible = True
            elif obj_label and not is_valid_target_match(obj_label, target):
                logger.warning(f"[MultiUAV-Plat] VLM distracted by '{obj_label}' while searching for '{target}'. Forcing target_visible=False for active phase.")
                target_visible = False
        if not obj_label:
            obj_label = self.current_phase.target_class if self.current_phase else "target"
        if target_visible or detected_object:
            belief = self.scene_graph.update(
                obj_label, self.x, self.y, self.yaw,
                box_center_x, box_center_y, self.gimbal_pitch, self.alt,
                confidence, self.getTime()
            )
            self.target_beliefs = self.scene_graph.beliefs
            logger.info(f"[MultiUAV-Plat] 🔍 Target spotted: '{obj_label}' @ absolute coordinates X={belief.absolute_position[0]:.2f}, Y={belief.absolute_position[1]:.2f} (est distance: {belief.estimated_distance:.1f}m, confidence: {confidence:.2f})")
            display_entry = f"{obj_label} @ X={belief.absolute_position[0]:.1f},Y={belief.absolute_position[1]:.1f}"
            if display_entry not in self.known_objects:
                self.known_objects.append(display_entry)
            telemetry.add_vlm_log(f"🎯 TARGET SPOTTED: {obj_label} @ [{belief.absolute_position[0]:.1f}, {belief.absolute_position[1]:.1f}]", "PERCEPTION_UPDATE", confidence)
            if not self.current_phase and target_visible:
                logger.info(f"[MultiUAV-Plat] ✅ Target confirmed via reactive perception. Mission complete.")
                mission.update(prompt="")
                self.current_prompt = ""
                return FlightState.REASONING

        target_name = self.current_phase.target_class if self.current_phase else "target"
        mem = self.memory_tracker.update(target_name, target_visible, box_center_x, box_center_y, self.getTime())

        if mem.coasting_active and (self.getTime() - mem.coast_start_time) <= 3.0:
            coast_elapsed = self.getTime() - mem.coast_start_time
            logger.info(f"[Coast Mode] 🛡️ Target lost {coast_elapsed:.1f}s ago near '{mem.last_seen_edge}'. Executing Coast & Gimbal Reacquisition. Locking body translations.")
            action_list = [step for step in action_list if step.action.startswith("TURN_") or step.action == "HOVER"]
            if not action_list:
                if coast_elapsed <= 1.5:
                    new_pitch = 0.2 if coast_elapsed < 0.7 else 0.4
                    self.set_gimbal_pitch(new_pitch)
                    action_list = [ActionStep("HOVER", 0.0, gimbal_pitch=new_pitch)]
                else:
                    turn_act = "TURN_LEFT" if mem.last_seen_edge == "left_edge" else ("TURN_RIGHT" if mem.last_seen_edge == "right_edge" else "TURN_LEFT")
                    action_list = [ActionStep(turn_act, 8.0, gimbal_pitch=self.gimbal_pitch)]

        if action_list and action_list[0].action != "HOVER":
            self.vlm_driven = True
            logger.info(f"[MultiUAV-Plat] VLM returned action '{action_list[0].action}' for direct execution")
            telemetry.add_vlm_log(reasoning, f"ACTION({action_list[0].action})", 0.0)
            with self.action_queue_lock:
                self.action_queue.extend(action_list)
                if len(self.action_queue) > MAX_ACTION_QUEUE_LEN:
                    self.action_queue = self.action_queue[:MAX_ACTION_QUEUE_LEN]
        else:
            self.vlm_driven = False
            telemetry.add_vlm_log(reasoning, "PERCEPTION_ONLY", confidence)
        return FlightState.REASONING

    def _state_dormant(self) -> None:
        self.battery = 100.0
        telemetry.update(x=self.x, y=self.y, altitude=self.alt, roll=0.0, pitch=0.0, yaw=self.yaw, battery=self.battery, state="DORMANT", gimbal_pitch=self.gimbal_pitch)

    def _state_takeoff(self, next_x: float, next_y: float, next_alt: float, next_yaw: float) -> StateTransition:
        if abs(self.alt - self.target_alt) < 0.1:
            logger.info(f"[MultiUAV-Plat] Takeoff complete at alt={self.alt:.2f}m. Entering REASONING.")
            return StateTransition(FlightState.REASONING, next_x, next_y, next_yaw)
        return StateTransition(FlightState.TAKEOFF, next_x, next_y, next_yaw)

    def _state_navigating(self, next_x: float, next_y: float, next_alt: float, next_yaw: float) -> StateTransition:
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        dist = math.sqrt(dx * dx + dy * dy)
        desired_yaw = math.atan2(dy, dx) if dist > 0.05 else self.target_yaw
        yaw_diff = normalize_angle(desired_yaw - self.yaw)

        if abs(yaw_diff) > COORD_TURN_THRESHOLD:
            yaw_step = sign(yaw_diff) * TURN_SPEED * self.dt
            if abs(yaw_step) > abs(yaw_diff):
                yaw_step = yaw_diff
            next_yaw = self.yaw + yaw_step
            return StateTransition(FlightState.NAVIGATING, next_x, next_y, next_yaw)

        if dist > 0.05:
            move_step = min(dist, MOVE_SPEED * self.dt)
            next_x = self.x + move_step * math.cos(desired_yaw)
            next_y = self.y + move_step * math.sin(desired_yaw)
            return StateTransition(FlightState.NAVIGATING, next_x, next_y, desired_yaw)

        final_yaw_diff = normalize_angle(self.target_yaw - self.yaw)
        if abs(final_yaw_diff) > YAW_PRECISION:
            yaw_step = sign(final_yaw_diff) * TURN_SPEED * self.dt
            if abs(yaw_step) > abs(final_yaw_diff):
                yaw_step = final_yaw_diff
            next_yaw = self.yaw + yaw_step
            return StateTransition(FlightState.NAVIGATING, next_x, next_y, next_yaw)

        logger.info(f"[MultiUAV-Plat] Waypoint reached: ({self.x:.2f}, {self.y:.2f}, {self.alt:.2f}m, yaw={math.degrees(self.yaw):.1f}°)")
        return StateTransition(FlightState.REASONING, self.x, self.y, self.target_yaw)


    def _state_curving(self, next_x: float, next_y: float, next_yaw: float) -> StateTransition:
        self.arc["elapsed"] += self.dt
        frac = min(1.0, self.arc["elapsed"] / self.arc["duration"])
        dist = self.arc["distance"] * frac
        angle_rad = math.radians(self.arc["angle"]) * frac
        start_x = self.arc["start_x"]
        start_y = self.arc["start_y"]
        start_yaw = self.arc["start_yaw"]

        if abs(angle_rad) < 0.001:
            cur_x = start_x + dist * math.cos(start_yaw)
            cur_y = start_y + dist * math.sin(start_yaw)
            cur_yaw = start_yaw
        else:
            R = dist / angle_rad if abs(angle_rad) > 0.001 else dist
            C = 2.0 * R * math.sin(angle_rad / 2.0)
            chord = start_yaw + angle_rad / 2.0
            cur_x = start_x + C * math.cos(chord)
            cur_y = start_y + C * math.sin(chord)
            cur_yaw = normalize_angle(start_yaw + angle_rad)

        if frac >= 1.0:
            logger.info(f"[MultiUAV-Plat] Curve complete. End pose: ({cur_x:.2f}, {cur_y:.2f}, {math.degrees(cur_yaw):.1f}°)")
            self.target_x = cur_x
            self.target_y = cur_y
            self.target_yaw = cur_yaw
            return StateTransition(FlightState.REASONING, cur_x, cur_y, cur_yaw)

        return StateTransition(FlightState.CURVING, cur_x, cur_y, cur_yaw)

    def _state_landing(self, next_yaw: float) -> StateTransition:
        if self.alt <= 0.1:
            logger.info("[MultiUAV-Plat] Landed. Entering DORMANT.")
            return StateTransition(FlightState.DORMANT, self.x, self.y, next_yaw)
        return StateTransition(FlightState.LANDING, self.x, self.y, next_yaw)

    def _state_reasoning(self, next_yaw: float) -> StateTransition:
        next_x, next_y = self.x, self.y
        yaw_diff = normalize_angle(self.target_yaw - self.yaw)
        if abs(yaw_diff) > 0.005:
            yaw_step = sign(yaw_diff) * TURN_SPEED * self.dt
            if abs(yaw_step) > abs(yaw_diff):
                yaw_step = yaw_diff
            next_yaw = self.yaw + yaw_step

        with self.action_queue_lock:
            has_actions = len(self.action_queue) > 0
        if has_actions:
            return self._process_next_action(next_x, next_y, next_yaw)

        if self.current_phase:
            if self.revert_to_search:
                self.revert_to_search = False
                target = self.current_phase.target_class
                logger.info(f"[Planner] Target lost. Re-planning search for target: '{target}'")
                self.mission_phases = [
                    SearchPhase(target_class=target),
                    NavigatePhase(target_class=target, stop_distance=6.0),
                    ManeuverPhase(maneuver_type="hover", target_class=target)
                ]
                self.current_phase = self.mission_phases.pop(0)
                self.current_phase.enter(self)
            if self.current_phase.completed:
                if self.current_phase.phase_type == MissionPhaseType.NAVIGATE and self.current_phase.target_class:
                    self.completed_targets.add(self.current_phase.target_class)
                    logger.info(f"[Planner] Target '{self.current_phase.target_class}' marked completed.")
                if self.mission_phases:
                    self.current_phase = self.mission_phases.pop(0)
                    self.current_phase.enter(self)
                    logger.info(f"[Planner] Transitioning to next phase: {self.current_phase.phase_type.value}")
                else:
                    logger.info("[Planner] All mission phases completed successfully.")
                    self.current_phase = None
                    mission.update(prompt="")
                    self.current_prompt = ""
                    return StateTransition(FlightState.REASONING, next_x, next_y, next_yaw)

            next_state = self.current_phase.execute(self)
            if next_state == FlightState.REASONING:
                # If phase populated the action queue (e.g. VLMActionSequencePhase), defer VLM query
                with self.action_queue_lock:
                    if len(self.action_queue) > 0:
                        return StateTransition(FlightState.REASONING, next_x, next_y, next_yaw)

                current_time = self.getTime()
                active_cooldown = SEARCH_VLM_COOLDOWN if isinstance(self.current_phase, SearchPhase) else VLM_COOLDOWN
                if self.current_prompt and (current_time - self.last_vlm_call_time > active_cooldown):
                    if is_frame_uniform(IMAGE_PATH) and abs(self.alt - self.target_alt) > 0.3:
                        return StateTransition(FlightState.REASONING, next_x, next_y, next_yaw)
                    self.last_vlm_call_time = current_time
                    self._query_vlm_and_queue(self.current_prompt, self.current_api_provider, self.current_api_key, self.current_model_name)
                return StateTransition(FlightState.REASONING, next_x, next_y, next_yaw)
            else:
                return StateTransition(next_state, self.target_x, self.target_y, next_yaw)

        with self.action_queue_lock:
            has_actions = len(self.action_queue) > 0
        if has_actions:
            return self._process_next_action(next_x, next_y, next_yaw)

        current_time = self.getTime()
        if self.current_prompt and (current_time - self.last_vlm_call_time > VLM_COOLDOWN):
            if is_frame_uniform(IMAGE_PATH) and abs(self.alt - self.target_alt) > 0.3:
                return StateTransition(FlightState.REASONING, next_x, next_y, next_yaw)
            self.last_vlm_call_time = current_time
            self._query_vlm_and_queue(self.current_prompt, self.current_api_provider, self.current_api_key, self.current_model_name)

        return StateTransition(FlightState.REASONING, next_x, next_y, next_yaw)

    def _process_next_action(self, next_x: float, next_y: float, next_yaw: float) -> StateTransition:
        with self.action_queue_lock:
            step = self.action_queue.pop(0)

        proj_x, proj_y, proj_yaw = ActionProjector.project(step.action, step.value, self.x, self.y, self.yaw)
        dist_from_origin = math.sqrt((proj_x - self.mission_start_x)**2 + (proj_y - self.mission_start_y)**2)
        if dist_from_origin > GEOFENCE_RADIUS:
            logger.warning(f"[GEOFENCE] Step {step.action} {step.value} would exceed geofence limit ({dist_from_origin:.1f}m). Converting to HOVER.")
            step.action = "HOVER"
            step.value = 0.0

        if abs(step.gimbal_pitch - self.gimbal_pitch) > 0.05:
            self.set_gimbal_pitch(step.gimbal_pitch)

        self.step_counter += 1
        self._set_target(step.action, step.value)

        if step.action == "LAND":
            return StateTransition(FlightState.LANDING, next_x, next_y, next_yaw)
        elif step.action in ("CLIMB", "DESCENT"):
            return StateTransition(FlightState.TAKEOFF, next_x, next_y, next_yaw)
        elif step.action == "HOVER":
            return StateTransition(FlightState.REASONING, next_x, next_y, next_yaw)
        elif step.action == "MOVE_CURVE":
            return StateTransition(FlightState.CURVING, next_x, next_y, next_yaw)
        else:
            return StateTransition(FlightState.NAVIGATING, next_x, next_y, next_yaw)

    def _set_target(self, action: str, value: Any) -> None:
        if action == "HOVER" or action == "LAND":
            self.target_x = self.x
            self.target_y = self.y
            self.target_yaw = self.yaw
            return
        if action == "MOVE_CURVE":
            dist, angle_deg = value
            self.arc["distance"] = dist
            self.arc["angle"] = angle_deg
            self.arc["start_x"] = self.x
            self.arc["start_y"] = self.y
            self.arc["start_yaw"] = self.yaw
            self.arc["elapsed"] = 0.0
            self.arc["duration"] = max(0.5, dist / MOVE_SPEED)
            return
        if action.startswith("MOVE_"):
            move_yaw = self.yaw
            if action == "MOVE_BACKWARD": move_yaw += math.pi
            elif action == "MOVE_LEFT": move_yaw += math.pi / 2.0
            elif action == "MOVE_RIGHT": move_yaw -= math.pi / 2.0
            dist = float(value)
            self.target_x = self.x + dist * math.cos(move_yaw)
            self.target_y = self.y + dist * math.sin(move_yaw)
            self.target_yaw = self.yaw
            return
        if action.startswith("TURN_"):
            angle_rad = math.radians(float(value))
            if action == "TURN_LEFT": self.target_yaw = normalize_angle(self.yaw + angle_rad)
            else: self.target_yaw = normalize_angle(self.yaw - angle_rad)
            self.target_x = self.x
            self.target_y = self.y
            return
        if action == "CLIMB":
            self.target_alt = min(MAX_ALTITUDE, self.alt + float(value))
            return
        if action == "DESCENT":
            self.target_alt = max(MIN_ALTITUDE, self.alt - float(value))
            return

    def run(self) -> None:
        logger.info("[MultiUAV-Plat Controller] Main loop started")
        while self.step(self.time_step) != -1:
            try:
                # 1. Update pose from sensors
                gps_pos = self.gps.getValues()
                rpy = self.imu.getRollPitchYaw()
                self.x = gps_pos[0]
                self.y = gps_pos[1]
                self.alt = gps_pos[2]
                self.yaw = rpy[2]

                # 2. Check for manual commands
                with manual_queue_lock:
                    has_manual = len(manual_command_queue) > 0
                if has_manual:
                    with manual_queue_lock:
                        m_cmd = manual_command_queue.pop(0)
                    new_st = self._process_manual_command(m_cmd)
                    if new_st is not None:
                        self.flight_state = new_st

                # 3. Check for mission updates
                has_update, p_text, prov, key, m_name, pid = mission.get_prompt_update(self.current_prompt_id)
                if has_update:
                    self.current_prompt_id = pid
                    self._init_mission(p_text, prov, key, m_name)

                # 4. State machine execution
                next_x, next_y, next_alt, next_yaw = self.x, self.y, self.alt, self.yaw
                if self.flight_state != FlightState.DORMANT:
                    alt_diff = self.target_alt - self.alt
                    if abs(alt_diff) > 0.01:
                        alt_step = sign(alt_diff) * CLIMB_SPEED * self.dt
                        if abs(alt_step) > abs(alt_diff): alt_step = alt_diff
                        next_alt = self.alt + alt_step
                    else: next_alt = self.target_alt
                    self.motors[0].setVelocity(25.0)
                    self.motors[1].setVelocity(-25.0)
                    self.motors[2].setVelocity(-25.0)
                    self.motors[3].setVelocity(25.0)

                if self.flight_state == FlightState.DORMANT:
                    self._state_dormant()
                    continue
                elif self.flight_state == FlightState.TAKEOFF:
                    transition = self._state_takeoff(next_x, next_y, next_alt, next_yaw)
                    next_x, next_y, next_yaw = transition.x, transition.y, transition.yaw
                    self.flight_state = transition.next_state
                elif self.flight_state == FlightState.NAVIGATING:
                    transition = self._state_navigating(next_x, next_y, next_alt, next_yaw)
                    next_x, next_y, next_yaw = transition.x, transition.y, transition.yaw
                    self.flight_state = transition.next_state
                elif self.flight_state == FlightState.CURVING:
                    transition = self._state_curving(next_x, next_y, next_yaw)
                    next_x, next_y, next_yaw = transition.x, transition.y, transition.yaw
                    self.flight_state = transition.next_state
                elif self.flight_state == FlightState.LANDING:
                    transition = self._state_landing(next_yaw)
                    next_x, next_y, next_yaw = transition.x, transition.y, transition.yaw
                    self.flight_state = transition.next_state
                    if self.flight_state == FlightState.DORMANT: next_alt = 0.065
                elif self.flight_state == FlightState.REASONING:
                    transition = self._state_reasoning(next_yaw)
                    next_x, next_y, next_yaw = transition.x, transition.y, transition.yaw
                    self.flight_state = transition.next_state

                # 5. Apply pose & update telemetry
                self.trans_field.setSFVec3f([next_x, next_y, next_alt])
                self.rot_field.setSFRotation([0.0, 0.0, 1.0, next_yaw])
                self.robot_node.setVelocity([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                self._update_trail()


                telemetry.update(
                    x=self.x, y=self.y, altitude=self.alt,
                    roll=rpy[0], pitch=rpy[1], yaw=self.yaw,
                    battery=self.battery, state=self.flight_state.value,
                    gimbal_pitch=self.gimbal_pitch
                )
            except Exception as e:
                logger.error(f"[MultiUAV-Plat Controller] Error in flight loop: {e}", exc_info=True)

    def _process_manual_command(self, cmd: str) -> Optional[FlightState]:
        logger.info(f"[MultiUAV-Plat] Manual command: '{cmd}'")
        mission.update(prompt="", new_request_queued=False)
        with self.action_queue_lock:
            self.action_queue.clear()
        if cmd.startswith("gimbal_pitch_"):
            try:
                pitch = float(cmd.split("_")[-1])
                self.set_gimbal_pitch(pitch)
            except ValueError: pass
            return None
        if cmd == "takeoff":
            if self.flight_state == FlightState.DORMANT:
                self.target_x = self.x
                self.target_y = self.y
                self.target_alt = 3.0
                self.target_yaw = self.yaw
                return FlightState.TAKEOFF
        elif cmd == "land":
            return FlightState.LANDING
        elif cmd == "hover":
            self.target_x = self.x
            self.target_y = self.y
            self.target_yaw = self.yaw
            return FlightState.REASONING
        return None

    def _init_mission(self, prompt: str, provider: str = "lmstudio", key: str = "", model_name: str = "") -> None:
        self.current_prompt = prompt
        self.current_api_provider = provider
        self.current_api_key = key
        self.current_model_name = model_name
        self.mission_start_x = self.x
        self.mission_start_y = self.y
        self.mission_start_yaw = self.yaw
        self.mission_start_time = self.getTime()
        self.known_objects.clear()
        self.target_beliefs.clear()
        self.scene_graph.beliefs.clear()
        self.step_counter = 0
        self.history.clear()
        self.revert_to_search = False
        self.completed_targets.clear()
        with self.action_queue_lock:
            self.action_queue.clear()
        self.mission_phases = MissionPlanner.plan(
            prompt,
            vlm_client=self.vlm,
            provider=provider,
            api_key=key,
            model_name=model_name
        )
        if self.mission_phases:
            self.current_phase = self.mission_phases.pop(0)
            self.current_phase.enter(self)
            logger.info(f"[MissionPlanner] Starting phase: {self.current_phase.phase_type.value}")
        else:
            self.current_phase = None

if __name__ == "__main__":
    server_thread = threading.Thread(target=run_web_server, args=(8081,), daemon=True)
    server_thread.start()
    robot = MavicVLM()
    robot.run()
