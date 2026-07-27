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
        logging.FileHandler("mavic_vlm.log", mode="a")
    ]
)
logger = logging.getLogger("MavicVLM")
class MavicVLM(Supervisor):
    """Webots Supervisor-based Mavic drone with VLM-driven autonomy."""
    def __init__(self):
        super().__init__()
        self.time_step = int(self.getBasicTimeStep())
        self.dt = self.time_step / 1000.0
        # Supervisor handles for absolute pose control
        self.robot_node = self.getSelf()
        self.trans_field = self.robot_node.getField("translation")
        self.rot_field = self.robot_node.getField("rotation")
        # Target memory tracker (Fix 1)
        self.memory_tracker = TargetMemoryTracker()
        # Sensors
        self.camera = self.getDevice("camera")
        self.camera.enable(self.time_step)
        self.imu = self.getDevice("inertial unit")
        self.imu.enable(self.time_step)
        self.gps = self.getDevice("gps")
        self.gps.enable(self.time_step)
        # Gimbal
        self.camera_pitch_motor = self.getDevice("camera pitch")
        self.cam_node = self.getFromDef("CAM")
        self.cam_rot_field = self.cam_node.getField("rotation") if self.cam_node else None
        self._gimbal_pitch = 0.0
        self.set_gimbal_pitch(0.0)
        # Propellers (visual only — Supervisor bypasses physics)
        self.motors = [
            self.getDevice("front left propeller"),
            self.getDevice("front right propeller"),
            self.getDevice("rear left propeller"),
            self.getDevice("rear right propeller")
        ]
        for m in self.motors:
            m.setPosition(float('inf'))
            m.setVelocity(0.0)
        # State
        self.x = self.y = self.alt = 0.0
        self.yaw = 0.0
        self.target_x = self.target_y = self.target_alt = self.target_yaw = 0.0
        self.battery = 100.0
        self.flight_state = FlightState.DORMANT
        # Mission context
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
        # Action queue (thread-safe)
        self.action_queue: List[ActionStep] = []
        self.action_queue_lock = threading.Lock()
        # VLM
        self.vlm = VLMClient()
        self.last_vlm_call_time = 0.0
        self.current_prompt = ""
        self.current_api_provider = "lmstudio"
        self.current_api_key = ""
        self.current_model_name = ""
        self.current_prompt_id = 0
        # Arc trajectory state
        self.arc = {
            "distance": 0.0,
            "angle": 0.0,
            "start_x": 0.0,
            "start_y": 0.0,
            "start_yaw": 0.0,
            "elapsed": 0.0,
            "duration": 1.0
        }
        # Image saving
        self._last_image_save_time = 0.0
        self._init_trail()
        logger.info("[Mavic VLM Controller] Initialized")

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
            logger.info("[Mavic VLM Controller] 3D path visualization node injected")
        except Exception as e:
            logger.warning(f"[Mavic VLM Controller] Could not inject 3D path node: {e}")
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
            # Webots MFField insertion requires insertMFVec3f / insertMFInt32
            self.path_coord_field.insertMFVec3f(-1, current_pos)
            if idx > 0:
                self.path_index_field.insertMFInt32(-1, idx - 1)
                self.path_index_field.insertMFInt32(-1, idx)
                self.path_index_field.insertMFInt32(-1, -1)
                # Inject a visible glowing marker sphere every 1.0m step
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


    # -----------------------------------------------------------------------
    # Gimbal Control
    # -----------------------------------------------------------------------
    def set_gimbal_pitch(self, pitch: float) -> None:
        """Unified gimbal control — always use this method."""
        pitch = max(0.0, min(0.8, pitch))
        self._gimbal_pitch = pitch
        if self.cam_rot_field:
            self.cam_rot_field.setSFRotation([0.0, 1.0, 0.0, pitch])
        else:
            self.camera_pitch_motor.setPosition(pitch)
    @property
    def gimbal_pitch(self) -> float:
        return self._gimbal_pitch
    # -----------------------------------------------------------------------
    # Context Building
    # -----------------------------------------------------------------------
    def _is_search_mission(self, prompt: str) -> bool:
        p = prompt.lower()
        search_kws = ["find", "search", "look for", "locate", "scan", "survey", "identify", "spot", "detect"]
        return any(k in p for k in search_kws)
    def _build_context_block(self, prompt: str) -> str:
        yaw_deg = math.degrees(self.yaw) % 360
        cardinal = heading_to_cardinal(yaw_deg)
        dx = self.x - self.mission_start_x
        dy = self.y - self.mission_start_y
        dist_from_origin = math.sqrt(dx * dx + dy * dy)
        mission_elapsed = self.getTime() - self.mission_start_time
        # Camera label
        if self.gimbal_pitch <= 0.1:
            cam_label = "Horizon (far view) ✓"
        elif self.gimbal_pitch <= 0.5:
            cam_label = f"Oblique ~{int(self.gimbal_pitch * 90 / 0.8)}° down"
        else:
            cam_label = "Nadir (floor view)"
        # History display
        if self.history:
            hist_lines = "\n".join(f"  {h.to_display()}" for h in self.history[-10:])
        else:
            hist_lines = "  (no actions yet — first step)"
        # Known objects
        if self.known_objects:
            objects_str = "\n".join(f"  - {o}" for o in self.known_objects)
        else:
            objects_str = "  None confirmed yet."
        # Target beliefs
        if self.target_beliefs:
            beliefs_str = "\n".join(f"  - {name}: conf={tb.confidence:.2f} @ absolute X={tb.absolute_position[0]:.1f}, Y={tb.absolute_position[1]:.1f} (dist: {tb.estimated_distance:.1f}m)" for name, tb in self.target_beliefs.items())
        else:
            beliefs_str = "  No active beliefs."
        # Scan coverage from structured history
        scanned_sectors = set()
        for h in self.history:
            if h.action in ("HOVER", "TURN_LEFT", "TURN_RIGHT"):
                sector = (round(h.heading / SCAN_SECTOR_SIZE) * SCAN_SECTOR_SIZE) % 360
                scanned_sectors.add(sector)
        pct_covered = int(len(scanned_sectors) / 6 * 100)
        if scanned_sectors:
            sector_labels = ", ".join(f"{s}°" for s in sorted(scanned_sectors))
            scan_status = f"{pct_covered}% of horizon scanned — sectors: {sector_labels}"
        else:
            scan_status = "0% — no sectors scanned yet"
        # Advisory
        is_search = self._is_search_mission(prompt)
        advisory = ""
        if is_search:
            if self.gimbal_pitch > 0.1:
                advisory = "⚠ SEARCH MODE: Camera NOT at horizon. Output HOVER with gimbal_pitch=0.0 immediately."
            elif pct_covered < 100:
                all_sectors = set(range(0, 360, SCAN_SECTOR_SIZE))
                remaining = sorted(all_sectors - scanned_sectors)
                if remaining:
                    next_hdg = remaining[0]
                    delta = next_hdg - int(yaw_deg)
                    direction = "TURN_LEFT" if delta > 0 else "TURN_RIGHT"
                    advisory = f"▶ SEARCH MODE: Scan {pct_covered}% done. Next: {direction} to reach {next_hdg}° sector."
            else:
                advisory = "▶ SEARCH MODE: Full 360° scan complete. MOVE_FORWARD 12m to reposition."
        return (
            "\n=== DRONE SITUATIONAL CONTEXT ===\n"
            f"Position   : X={self.x:.1f}m, Y={self.y:.1f}m  "
            f"(Δ from start: {dx:+.1f}m X, {dy:+.1f}m Y, {dist_from_origin:.1f}m total)\n"
            f"Altitude   : {self.alt:.1f}m\n"
            f"Heading    : {yaw_deg:.0f}° ({cardinal})\n"
            f"Camera     : {cam_label} (gimbal_pitch={self.gimbal_pitch:.2f})\n"
            f"Scan Status: {scan_status}\n"
            f"Mission    : '{prompt}'  [Step #{self.step_counter + 1}, elapsed {mission_elapsed:.1f}s]\n"
            + (f"Advisory   : {advisory}\n" if advisory else "")
            + f"Known Objects This Session:\n{objects_str}\n"
            f"Target Beliefs:\n{beliefs_str}\n"
            f"Action History (last 10):\n{hist_lines}\n"
            "=================================\n"
        )
    # -----------------------------------------------------------------------
    # Telemetry
    # -----------------------------------------------------------------------
    def _update_telemetry(self) -> None:
        target = None
        if self.flight_state in (FlightState.NAVIGATING, FlightState.TAKEOFF,
                                  FlightState.LANDING, FlightState.CURVING):
            target = [self.target_x, self.target_y, self.target_alt]
        from dataclasses import asdict
        beliefs_list = [asdict(b) for b in self.target_beliefs.values()]
        telemetry.update(
            x=self.x, y=self.y, altitude=self.alt,
            yaw=self.yaw, battery=self.battery,
            state=self.flight_state.value,
            target=target,
            gimbal_pitch=self.gimbal_pitch,
            target_beliefs=beliefs_list
        )
    def _add_history(self, action: str, distance: float = 0.0, yaw_angle: float = 0.0) -> None:
        """Add structured history entry."""
        entry = HistoryEntry(
            step=self.step_counter,
            action=action,
            distance=distance,
            yaw_angle=yaw_angle,
            heading=math.degrees(self.yaw) % 360,
            x=self.x, y=self.y, altitude=self.alt
        )
        self.history.append(entry)
        if len(self.history) > MAX_HISTORY_LEN:
            self.history.pop(0)
    # -----------------------------------------------------------------------
    # Navigation Math
    # -----------------------------------------------------------------------
    def _set_target(self, action: str, value: float | Tuple[float, float]) -> None:
        """Compute absolute target from relative action."""
        logger.info(f"[Navigation] Setting target: {action} ({value})")
        if action == "LAND":
            self.target_alt = 0.0
            return
        elif action == "CLIMB":
            self.target_alt = min(MAX_ALTITUDE, float(value))
            return
        elif action == "DESCEND":
            self.target_alt = max(MIN_ALTITUDE, float(value))
            return
        elif action == "HOVER":
            self.target_x = self.x
            self.target_y = self.y
            self.target_alt = self.alt
            return
        if action == "MOVE_CURVE":
            dist, angle_deg = value  # type: ignore
            angle_rad = math.radians(angle_deg)
            self.arc["distance"] = dist
            self.arc["angle"] = angle_rad
            self.arc["start_x"] = self.x
            self.arc["start_y"] = self.y
            self.arc["start_yaw"] = self.yaw
            self.arc["elapsed"] = 0.0
            # Duration = max(translation_time, rotation_time) for synchronization
            trans_time = dist / MOVE_SPEED if MOVE_SPEED > 0 else 0.1
            rot_time = abs(angle_rad) / TURN_SPEED if TURN_SPEED > 0 else 0.1
            self.arc["duration"] = max(trans_time, rot_time, 0.1)
            # Chord endpoint
            if abs(angle_rad) < 0.001:
                self.target_x = self.x + dist * math.cos(self.yaw)
                self.target_y = self.y + dist * math.sin(self.yaw)
            else:
                R = dist / angle_rad
                C = 2.0 * R * math.sin(angle_rad / 2.0)
                chord_angle = self.yaw + angle_rad / 2.0
                self.target_x = self.x + C * math.cos(chord_angle)
                self.target_y = self.y + C * math.sin(chord_angle)
            self.target_yaw = normalize_angle(self.yaw + angle_rad)
            self.target_alt = self.alt
            return
        if action.startswith("MOVE_"):
            move_yaw = self.yaw
            if action == "MOVE_BACKWARD":
                move_yaw += math.pi
            elif action == "MOVE_LEFT":
                move_yaw += math.pi / 2.0
            elif action == "MOVE_RIGHT":
                move_yaw -= math.pi / 2.0
            dist = float(value)  # type: ignore
            self.target_x = self.x + dist * math.cos(move_yaw)
            self.target_y = self.y + dist * math.sin(move_yaw)
        elif action == "CLIMB":
            dist = float(value)  # type: ignore
            self.target_alt = min(MAX_ALTITUDE, self.alt + dist)
        elif action == "DESCENT":
            dist = float(value)  # type: ignore
            self.target_alt = max(MIN_ALTITUDE, self.alt - dist)
        elif action.startswith("TURN_"):
            angle_rad = math.radians(float(value))  # type: ignore
            if action == "TURN_LEFT":
                self.target_yaw = normalize_angle(self.yaw + angle_rad)
            else:
                self.target_yaw = normalize_angle(self.yaw - angle_rad)
    # -----------------------------------------------------------------------
    # State Machine Handlers
    # -----------------------------------------------------------------------
    def _state_dormant(self) -> None:
        self._update_telemetry()
        for m in self.motors:
            m.setVelocity(0.0)
    def _state_takeoff(self, next_x: float, next_y: float, next_alt: float, next_yaw: float) -> StateTransition:
        self._update_telemetry()
        # Yaw alignment during climb
        yaw_diff = normalize_angle(self.target_yaw - self.yaw)
        if abs(yaw_diff) > 0.005:
            yaw_step = sign(yaw_diff) * TURN_SPEED * self.dt
            if abs(yaw_step) > abs(yaw_diff):
                yaw_step = yaw_diff
            next_yaw = self.yaw + yaw_step
        # Check altitude reached
        if abs(self.alt - self.target_alt) < 0.15:
            dist_to_target = math.sqrt((self.target_x - self.x)**2 + (self.target_y - self.y)**2)
            if dist_to_target > TARGET_PRECISION:
                return StateTransition(FlightState.NAVIGATING, next_x, next_y, next_yaw)
            else:
                return StateTransition(FlightState.REASONING, next_x, next_y, next_yaw)
        return StateTransition(FlightState.TAKEOFF, next_x, next_y, next_yaw)
    def _state_navigating(self, next_x: float, next_y: float, next_alt: float, next_yaw: float) -> StateTransition:
        self._update_telemetry()
        dist_to_target = math.sqrt((self.target_x - self.x)**2 + (self.target_y - self.y)**2)
        if dist_to_target < TARGET_PRECISION:
            # Position reached — check yaw
            yaw_diff = normalize_angle(self.target_yaw - self.yaw)
            if abs(yaw_diff) < YAW_PRECISION:
                # Use actual GPS position (self.x/y), NOT self.target_x/y, to prevent
                # the drone teleporting to the target coordinate on the next setSFVec3f call.
                return StateTransition(FlightState.REASONING, self.x, self.y, self.target_yaw)
            else:
                yaw_step = sign(yaw_diff) * TURN_SPEED * self.dt
                if abs(yaw_step) > abs(yaw_diff):
                    yaw_step = yaw_diff
                next_yaw = self.yaw + yaw_step
            return StateTransition(FlightState.NAVIGATING, self.x, self.y, next_yaw)

        # Move toward target with coordinated turn
        target_angle = math.atan2(self.target_y - self.y, self.target_x - self.x)
        angle_diff = normalize_angle(target_angle - self.yaw)
        # Turn toward target
        if abs(angle_diff) > 0.005:
            yaw_step = sign(angle_diff) * TURN_SPEED * self.dt
            if abs(yaw_step) > abs(angle_diff):
                yaw_step = angle_diff
            next_yaw = self.yaw + yaw_step
        # Move if roughly aligned (coordinated turn for small errors)
        if abs(angle_diff) < COORD_TURN_THRESHOLD:
            speed_factor = max(0.3, 1.0 - abs(angle_diff) / COORD_TURN_THRESHOLD)
            pos_step = MOVE_SPEED * speed_factor * self.dt
            if pos_step > dist_to_target:
                pos_step = dist_to_target
            next_x = self.x + pos_step * math.cos(target_angle)
            next_y = self.y + pos_step * math.sin(target_angle)
        return StateTransition(FlightState.NAVIGATING, next_x, next_y, next_yaw)
    def _state_curving(self, next_x: float, next_y: float, next_yaw: float) -> StateTransition:
        self._update_telemetry()
        self.arc["elapsed"] += self.dt
        t = min(1.0, self.arc["elapsed"] / self.arc["duration"])
        theta_t = t * self.arc["angle"]
        if abs(self.arc["angle"]) < 0.001:
            C_t = t * self.arc["distance"]
            chord_angle_t = self.arc["start_yaw"]
        else:
            R = self.arc["distance"] / self.arc["angle"]
            C_t = 2.0 * R * math.sin(theta_t / 2.0)
            chord_angle_t = self.arc["start_yaw"] + theta_t / 2.0
        next_x = self.arc["start_x"] + C_t * math.cos(chord_angle_t)
        next_y = self.arc["start_y"] + C_t * math.sin(chord_angle_t)
        next_yaw = normalize_angle(self.arc["start_yaw"] + theta_t)
        if t >= 1.0:
            logger.info("[Navigation] Curved arc trajectory complete")
            self.target_x = next_x
            self.target_y = next_y
            self.target_yaw = next_yaw
            return StateTransition(FlightState.REASONING, next_x, next_y, next_yaw)
        return StateTransition(FlightState.CURVING, next_x, next_y, next_yaw)
    def _state_landing(self, next_yaw: float) -> StateTransition:
        self._update_telemetry()
        yaw_diff = normalize_angle(self.target_yaw - self.yaw)
        if abs(yaw_diff) > 0.005:
            yaw_step = sign(yaw_diff) * TURN_SPEED * self.dt
            if abs(yaw_step) > abs(yaw_diff):
                yaw_step = yaw_diff
            next_yaw = self.yaw + yaw_step
        if self.alt < 0.15:
            logger.info("[Flight Machine] Landing complete. Motors off.")
            return StateTransition(FlightState.DORMANT, self.x, self.y, next_yaw)
        return StateTransition(FlightState.LANDING, self.x, self.y, next_yaw)
    def _state_reasoning(self, next_yaw: float) -> StateTransition:
        self._update_telemetry()
        self.target_x = self.x
        self.target_y = self.y
        next_x = self.x
        next_y = self.y
        # Yaw alignment
        yaw_diff = normalize_angle(self.target_yaw - self.yaw)
        if abs(yaw_diff) > 0.005:
            yaw_step = sign(yaw_diff) * TURN_SPEED * self.dt
            if abs(yaw_step) > abs(yaw_diff):
                yaw_step = yaw_diff
            next_yaw = self.yaw + yaw_step
        # Prioritize queued VLM action sequence execution (long-sequence latency reduction)
        with self.action_queue_lock:
            has_actions = len(self.action_queue) > 0
        if has_actions:
            return self._process_next_action(next_x, next_y, next_yaw)

        # Phased planner execution loop
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
                # Target navigated to successfully — register to completed ignore list
                if self.current_phase.phase_type == MissionPhaseType.NAVIGATE and self.current_phase.target_class:
                    self.completed_targets.add(self.current_phase.target_class)
                    logger.info(f"[Planner] Target '{self.current_phase.target_class}' marked completed (IGNORE in subsequent sweeps).")
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
            # Execute active phase
            next_state = self.current_phase.execute(self)
            if next_state == FlightState.REASONING:
                # If phase populated the action queue (e.g. VLMActionSequencePhase), defer VLM query
                with self.action_queue_lock:
                    if len(self.action_queue) > 0:
                        return StateTransition(FlightState.REASONING, next_x, next_y, next_yaw)

                # Settle and query VLM perception
                current_time = self.getTime()
                # Use shorter cooldown during active search to avoid stalling sector sweeps
                active_cooldown = SEARCH_VLM_COOLDOWN if isinstance(self.current_phase, SearchPhase) else VLM_COOLDOWN
                if self.current_prompt and (current_time - self.last_vlm_call_time > active_cooldown):
                    if is_frame_uniform(IMAGE_PATH) and abs(self.alt - self.target_alt) > 0.3:
                        return StateTransition(FlightState.REASONING, next_x, next_y, next_yaw)
                    self.last_vlm_call_time = current_time
                    self._query_vlm_and_queue(self.current_prompt, self.current_api_provider, self.current_api_key, self.current_model_name)
                return StateTransition(FlightState.REASONING, next_x, next_y, next_yaw)
            else:
                return StateTransition(next_state, self.target_x, self.target_y, next_yaw)
        # Fallback to standard reactive execution (for manual commands / backward compatibility)
        with self.action_queue_lock:
            has_actions = len(self.action_queue) > 0
        if has_actions:
            return self._process_next_action(next_x, next_y, next_yaw)
        # Check VLM cooldown for general reactive loops
        current_time = self.getTime()
        if self.current_prompt and (current_time - self.last_vlm_call_time > VLM_COOLDOWN):
            if is_frame_uniform(IMAGE_PATH) and abs(self.alt - self.target_alt) > 0.3:
                return StateTransition(FlightState.REASONING, next_x, next_y, next_yaw)
            self.last_vlm_call_time = current_time
            next_state = self._query_vlm_and_queue(self.current_prompt, self.current_api_provider, self.current_api_key, self.current_model_name)
            return StateTransition(next_state, next_x, next_y, next_yaw)
        return StateTransition(FlightState.REASONING, next_x, next_y, next_yaw)
    def _process_next_action(self, next_x: float, next_y: float, next_yaw: float) -> StateTransition:
        with self.action_queue_lock:
            if not self.action_queue:
                return StateTransition(FlightState.REASONING, next_x, next_y, next_yaw)
            step = self.action_queue.pop(0)
        # Safety Geofence pre-check using ActionProjector
        proj_x, proj_y, proj_yaw = ActionProjector.project(step.action, step.value, self.x, self.y, self.yaw)
        dist_from_origin = math.sqrt((proj_x - self.mission_start_x)**2 + (proj_y - self.mission_start_y)**2)
        if dist_from_origin > GEOFENCE_RADIUS:
            logger.warning(f"[GEOFENCE] Step {step.action} {step.value} would exceed geofence limit ({dist_from_origin:.1f}m). Converting to HOVER.")
            step.action = "HOVER"
            step.value = 0.0
        # Apply gimbal if changed
        if abs(step.gimbal_pitch - self.gimbal_pitch) > 0.05:
            self.set_gimbal_pitch(step.gimbal_pitch)
            cam_desc = "Horizon" if step.gimbal_pitch <= 0.1 else ("Down" if step.gimbal_pitch >= 0.7 else "Oblique")
            self._add_history("GIMBAL", yaw_angle=step.gimbal_pitch)
            logger.info(f"[Flight Machine] Gimbal adjusted to {step.gimbal_pitch:.2f} ({cam_desc})")
        # Record history
        self.step_counter += 1
        if step.action == "MOVE_CURVE":
            dist, angle = step.value  # type: ignore
            self._add_history(step.action, distance=dist, yaw_angle=angle)
        elif step.action in ("TURN_LEFT", "TURN_RIGHT"):
            self._add_history(step.action, yaw_angle=float(step.value))  # type: ignore
        elif step.action.startswith("MOVE_"):
            self._add_history(step.action, distance=float(step.value))  # type: ignore
        else:
            self._add_history(step.action)
        # Execute
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
    def _get_current_frame_base64(self) -> str:
        """Encodes the current camera view directly from Webots memory to Base64 to bypass disk locks."""
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
    def _build_context_block(self, prompt: str) -> str:
        """Constructs rich telemetry and inter-frame target memory context for the VLM (Fix 4)."""
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

    def _query_vlm_and_queue(self, prompt: str, provider: str, key: str, model_name: str) -> FlightState:
        # Preserve sub-intent context: pass active sub-goal AND full mission prompt to VLM
        vlm_prompt = prompt
        if self.current_phase and self.current_phase.target_class:
            target = self.current_phase.target_class
            vlm_prompt = f"Active Sub-Goal: Locate and inspect '{target}'. Full Mission Prompt: '{prompt}'"
            if self.completed_targets:
                vlm_prompt += f" (Already completed sub-goals: {list(self.completed_targets)})"
        logger.info(f"[Flight Machine] Querying VLM (step #{self.step_counter + 1}) with focus: '{vlm_prompt}'")
        context = self._build_context_block(prompt)
        frame_b64 = self._get_current_frame_base64()
        action_list, reasoning, target_visible, detected_object, box_center_x, box_center_y, confidence, expected_obs = self.vlm.query(
            vlm_prompt, provider, key, model_name, IMAGE_PATH, context, default_gimbal_pitch=self.gimbal_pitch, image_base64=frame_b64
        )

        # MultiUAV-Plat Stage 5: Verification Check
        active_target = self.current_phase.target_class if self.current_phase else ""
        verif = ActionVerifier.verify(expected_obs, target_visible, detected_object, active_target)
        logger.info(f"[Flight Machine] Stage 5 Verification: {verif.reasoning}")
        if verif.recommended_replan and self.current_phase:
            logger.warning("[Flight Machine] Stage 6 Reflection: Distraction detected during verification. Re-syncing active search phase.")
            self.revert_to_search = True

        # 1. Determine what the VLM actually saw
        obj_label = detected_object.strip().lower() if detected_object else ""
        # 2. Prevent Attention Bias / Fixation & Promote Valid Detections
        if self.current_phase and self.current_phase.target_class:
            target = self.current_phase.target_class.lower()
            # If the VLM claims it saw the target but forgot to name it, assume it meant the active target
            if target_visible and not obj_label:
                obj_label = target
            # If the VLM detected the target object/feature (e.g. "house in distance"), promote target_visible=True
            if obj_label and is_valid_target_match(obj_label, target) and confidence >= 0.3:
                if not target_visible:
                    logger.info(f"[Flight Machine] Target/Feature '{obj_label}' identified in vision feed (confidence: {confidence:.2f}). Setting target_visible=True for active phase '{target}'.")
                    target_visible = True
            # If the VLM got distracted and reported an unrelated object...
            elif obj_label and not is_valid_target_match(obj_label, target):
                logger.warning(f"[Flight Machine] VLM distracted by '{obj_label}' while searching for '{target}'. Forcing target_visible=False for active phase.")
                target_visible = False
        # Fallback if entirely empty
        if not obj_label:
            obj_label = self.current_phase.target_class if self.current_phase else "target"
        if target_visible or detected_object:
            # Update SceneGraph belief with absolute coordinates
            belief = self.scene_graph.update(
                obj_label, self.x, self.y, self.yaw,
                box_center_x, box_center_y, self.gimbal_pitch, self.alt,
                confidence, self.getTime()
            )
            # Sync target_beliefs dict for UI serialization
            self.target_beliefs = self.scene_graph.beliefs
            logger.info(f"[Flight Machine] 🔍 Target spotted: '{obj_label}' @ absolute coordinates X={belief.absolute_position[0]:.2f}, Y={belief.absolute_position[1]:.2f} (est distance: {belief.estimated_distance:.1f}m, confidence: {confidence:.2f})")
            display_entry = f"{obj_label} @ X={belief.absolute_position[0]:.1f},Y={belief.absolute_position[1]:.1f}"
            if display_entry not in self.known_objects:
                self.known_objects.append(display_entry)
            telemetry.add_vlm_log(f"🎯 TARGET SPOTTED: {obj_label} @ [{belief.absolute_position[0]:.1f}, {belief.absolute_position[1]:.1f}]", "PERCEPTION_UPDATE", confidence)
            # If not a phased mission, treat spotting target as instant reactive completion
            if not self.current_phase and target_visible:
                logger.info(f"[Flight Machine] ✅ Target confirmed via reactive perception. Mission complete.")
                mission.update(prompt="")
                self.current_prompt = ""
                return FlightState.REASONING
        # Update Target Memory Tracker (Fix 1 & Fix 3)
        target_name = self.current_phase.target_class if self.current_phase else "target"
        mem = self.memory_tracker.update(target_name, target_visible, box_center_x, box_center_y, self.getTime())

        # Check 3-Second Coast Mode & Structured Reacquisition (Fix 2 & Fix 3)
        if mem.coasting_active and (self.getTime() - mem.coast_start_time) <= 3.0:
            coast_elapsed = self.getTime() - mem.coast_start_time
            logger.info(f"[Coast Mode] 🛡️ Target lost {coast_elapsed:.1f}s ago near '{mem.last_seen_edge}'. Executing Coast & Gimbal Reacquisition. Locking body translations.")
            # Lock out body translations during coasting
            action_list = [step for step in action_list if step.action.startswith("TURN_") or step.action == "HOVER"]
            if not action_list:
                if coast_elapsed <= 1.5:
                    # Phase 1: Gimbal pitch sweep
                    new_pitch = 0.2 if coast_elapsed < 0.7 else 0.4
                    self.set_gimbal_pitch(new_pitch)
                    action_list = [ActionStep("HOVER", 0.0, gimbal_pitch=new_pitch)]
                else:
                    # Phase 2: Small turn toward last_seen_edge
                    turn_act = "TURN_LEFT" if mem.last_seen_edge == "left_edge" else ("TURN_RIGHT" if mem.last_seen_edge == "right_edge" else "TURN_LEFT")
                    action_list = [ActionStep(turn_act, 8.0, gimbal_pitch=self.gimbal_pitch)]

        # Queue actions if returned (e.g. for direct backward compatible movement commands)
        if action_list and action_list[0].action != "HOVER":
            self.vlm_driven = True
            logger.info(f"[Flight Machine] VLM returned action '{action_list[0].action}' for direct execution")
            telemetry.add_vlm_log(reasoning, f"ACTION({action_list[0].action})", 0.0)
            with self.action_queue_lock:
                self.action_queue.extend(action_list)
                if len(self.action_queue) > MAX_ACTION_QUEUE_LEN:
                    self.action_queue = self.action_queue[:MAX_ACTION_QUEUE_LEN]
        else:
            self.vlm_driven = False
            telemetry.add_vlm_log(reasoning, "PERCEPTION_ONLY", confidence)
        return FlightState.REASONING
    # -----------------------------------------------------------------------
    # Manual Commands
    # -----------------------------------------------------------------------
    def _process_manual_command(self, cmd: str) -> Optional[FlightState]:
        """Process a single manual command. Returns new flight state or None if no change."""
        logger.info(f"[Flight Machine] Manual command: '{cmd}'")
        # Cancel active mission
        mission.update(prompt="", new_request_queued=False)
        with self.action_queue_lock:
            self.action_queue.clear()
        # Gimbal override
        if cmd.startswith("gimbal_pitch_"):
            try:
                pitch = float(cmd.split("_")[-1])
                self.set_gimbal_pitch(pitch)
            except ValueError:
                logger.error(f"Invalid gimbal pitch command: {cmd}")
            return None
        # State-dependent commands
        if cmd == "takeoff":
            if self.flight_state == FlightState.DORMANT:
                self.target_x = self.x
                self.target_y = self.y
                self.target_alt = 3.0
                self.target_yaw = self.yaw
                return FlightState.TAKEOFF
            return None
        if cmd == "land":
            if self.flight_state != FlightState.DORMANT:
                self.target_alt = 0.0
                return FlightState.LANDING
            return None
        if cmd == "hover":
            if self.flight_state != FlightState.DORMANT:
                self._set_target("HOVER", 0.0)
                return FlightState.REASONING
            return None
        # Movement commands — auto-takeoff if dormant
        if self.flight_state == FlightState.DORMANT:
            self.target_alt = 3.0
            self.target_yaw = self.yaw
            # Return TAKEOFF; movement will be processed next tick after takeoff completes
            return FlightState.TAKEOFF
        if self.flight_state in (FlightState.DORMANT, FlightState.TAKEOFF):
            return None  # Wait for takeoff
        # Navigation commands
        nav_map = {
            "forward": ("MOVE_FORWARD", 4.0),
            "backward": ("MOVE_BACKWARD", 4.0),
            "left": ("MOVE_LEFT", 4.0),
            "right": ("MOVE_RIGHT", 4.0),
            "yaw_left": ("TURN_LEFT", 90.0),
            "yaw_right": ("TURN_RIGHT", 90.0),
        }
        if cmd in nav_map:
            action, val = nav_map[cmd]
            self._set_target(action, val)
            return FlightState.NAVIGATING
        if cmd == "up":
            self.target_alt = min(MAX_ALTITUDE, self.target_alt + 1.0)
            self.target_x = self.x
            self.target_y = self.y
            return FlightState.NAVIGATING
        if cmd == "down":
            self.target_alt = max(MIN_ALTITUDE, self.target_alt - 1.0)
            self.target_x = self.x
            self.target_y = self.y
            return FlightState.NAVIGATING
        logger.warning(f"Unknown manual command: '{cmd}'")
        return None
    # -----------------------------------------------------------------------
    # Main Run Loop
    # -----------------------------------------------------------------------
    def run(self) -> None:
        logger.info("[Mavic VLM Controller] Starting flight control loop...")
        while self.step(self.time_step) != -1:
            # 1. Sensor reading with NaN fallback
            x_pos, y_pos, altitude = self.gps.getValues()
            _, _, yaw = self.imu.getRollPitchYaw()
            if any(math.isnan(v) for v in (x_pos, y_pos, altitude, yaw)):
                logger.warning("[SENSOR] NaN detected, using last valid values")
            else:
                self.x, self.y, self.alt, self.yaw = x_pos, y_pos, altitude, yaw
            # 2. Battery drain & failsafe
            if self.flight_state != FlightState.DORMANT:
                self.battery = max(0.0, self.battery - self.dt * BATTERY_DRAIN_RATE)
                if self.battery <= BATTERY_CRITICAL_PCT and self.flight_state != FlightState.LANDING:
                    logger.critical("[BATTERY] Critical level reached — emergency landing!")
                    self.flight_state = FlightState.LANDING
                    self.target_alt = 0.0
                    mission.update(prompt="")
                    self.current_prompt = ""
                    with self.action_queue_lock:
                        self.action_queue.clear()
            # 3. Atomic image save
            current_time = self.getTime()
            if current_time - self._last_image_save_time >= IMAGE_SAVE_INTERVAL:
                self._last_image_save_time = current_time
                try:
                    self.camera.saveImage(IMAGE_TMP_PATH, JPEG_QUALITY)
                    if os.path.exists(IMAGE_TMP_PATH):
                        with image_file_lock:
                            os.replace(IMAGE_TMP_PATH, IMAGE_PATH)
                except Exception as e:
                    # Ignore occasional Windows file locking access denied warnings
                    if "PermissionError" not in str(e) and "Access is denied" not in str(e):
                        logger.error(f"Image save failed: {e}")
            # 4. Process manual commands
            with manual_queue_lock:
                if manual_command_queue:
                    manual_cmd = manual_command_queue.pop(0)
                else:
                    manual_cmd = None
            if manual_cmd:
                new_state = self._process_manual_command(manual_cmd)
                if new_state is not None:
                    self.flight_state = new_state
            # 5. Process new mission prompt atomically
            was_queued, prompt_text, provider, key, model_name, new_pid = mission.get_prompt_update(self.current_prompt_id)
            if was_queued and prompt_text:
                self.current_prompt_id = new_pid
                self._init_mission(prompt_text, provider, key, model_name)
                if self.flight_state == FlightState.DORMANT:
                    self.flight_state = FlightState.TAKEOFF
                    self.target_x = self.x
                    self.target_y = self.y
                    self.target_yaw = self.yaw
                is_search = self._is_search_mission(prompt_text)
                if is_search and self.mission_phases:
                    # Use the first SearchPhase's altitude profile for initial climb
                    first_phase = self.current_phase
                    takeoff_alt = getattr(first_phase, "search_altitude", SEARCH_ALTITUDE)
                    self.set_gimbal_pitch(0.0)  # Keep horizontal during takeoff climb
                elif is_search:
                    takeoff_alt = SEARCH_ALTITUDE
                    self.set_gimbal_pitch(0.0)
                else:
                    takeoff_alt = 3.0
                    self.set_gimbal_pitch(0.0)
                self.target_alt = takeoff_alt
                logger.info(f"[Flight Machine] Mission '{prompt_text}' (ID: {new_pid}) initialized. Takeoff to {takeoff_alt}m")
            # 6. Geofencing check
            dist_from_origin = math.sqrt((self.x - self.mission_start_x)**2 + (self.y - self.mission_start_y)**2)
            if dist_from_origin > GEOFENCE_RADIUS and self.flight_state not in (FlightState.LANDING, FlightState.DORMANT):
                logger.warning(f"[GEOFENCE] Drone at {dist_from_origin:.1f}m exceeds limit. Returning to origin.")
                self.target_x = self.mission_start_x
                self.target_y = self.mission_start_y
                self.flight_state = FlightState.NAVIGATING
            # 7. State machine execution
            next_x, next_y, next_alt, next_yaw = self.x, self.y, self.alt, self.yaw
            # Vertical control (common to all airborne states)
            if self.flight_state != FlightState.DORMANT:
                alt_diff = self.target_alt - self.alt
                if abs(alt_diff) > 0.01:
                    alt_step = sign(alt_diff) * CLIMB_SPEED * self.dt
                    if abs(alt_step) > abs(alt_diff):
                        alt_step = alt_diff
                    next_alt = self.alt + alt_step
                else:
                    next_alt = self.target_alt
                # Spin propellers visually
                self.motors[0].setVelocity(25.0)
                self.motors[1].setVelocity(-25.0)
                self.motors[2].setVelocity(-25.0)
                self.motors[3].setVelocity(25.0)
            # State-specific logic with StateTransition assignments
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
                if self.flight_state == FlightState.DORMANT:
                    next_alt = 0.065
            elif self.flight_state == FlightState.REASONING:
                transition = self._state_reasoning(next_yaw)
                next_x, next_y, next_yaw = transition.x, transition.y, transition.yaw
                self.flight_state = transition.next_state
            else:
                logger.error(f"Unknown flight state: {self.flight_state}")
                self.flight_state = FlightState.REASONING
            # 8. Apply absolute pose override
            self.trans_field.setSFVec3f([next_x, next_y, next_alt])
            self.rot_field.setSFRotation([0.0, 0.0, 1.0, next_yaw])
            self.robot_node.setVelocity([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            self._update_trail()

    def _init_mission(self, prompt: str, provider: str = "lmstudio", key: str = "", model_name: str = "") -> None:
        """Reset mission context for a new prompt."""
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
        # Decompose prompt into structured mission phases
        # Pass the VLM client so each SearchPhase gets an LLM-determined altitude/gimbal profile
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
            logger.info(f"[MissionPlanner] Decomposed mission into {len(self.mission_phases) + 1} phases. Starting: {self.current_phase.phase_type.value}")
        else:
            self.current_phase = None
# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    server_thread = threading.Thread(target=run_web_server, args=(8080,), daemon=True)
    server_thread.start()
    robot = MavicVLM()
    robot.run()
