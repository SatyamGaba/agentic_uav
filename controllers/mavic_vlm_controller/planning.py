import math
import logging
import re
from enum import Enum
from typing import Optional, List, Dict, Any
from config import FlightState, ActionStep, SCAN_SECTOR_SIZE, SEARCH_MAX_LAPS, REPOSITION_DISTANCE, FALLBACK_SEARCH_PROFILES, DEFAULT_SEARCH_PROFILE, YAW_PRECISION

from utils import normalize_angle
from perception import SceneGraph, VLMClient

logger = logging.getLogger("MavicVLM")

class MissionPhaseType(Enum):
    TAKEOFF = "TAKEOFF"
    SEARCH = "SEARCH"
    NAVIGATE = "NAVIGATE"
    MANEUVER = "MANEUVER"
    LAND = "LAND"
    HOVER = "HOVER"


class MissionPhase:
    """Abstract base class representing a structured mission flight phase."""
    def __init__(self, phase_type: MissionPhaseType, target_class: str = ""):
        self.phase_type = phase_type
        self.target_class = target_class
        self.initialized = False
        self.completed = False

    def enter(self, drone) -> None:
        self.initialized = True

    def execute(self, drone) -> FlightState:
        raise NotImplementedError()


class SearchPhase(MissionPhase):
    """Executes a structured search sweep in SCAN_SECTOR_SIZE° increments."""
    def __init__(self, target_class: str, max_laps: int = SEARCH_MAX_LAPS):
        super().__init__(MissionPhaseType.SEARCH, target_class)
        self.sector_offsets = list(range(0, 360, SCAN_SECTOR_SIZE))  # [0, 60, 120, 180, 240, 300]
        self.current_sector_idx = 0
        self.at_sector = False           # True once yaw is aligned to current sector
        self.queried_at_sector = False   # True once VLM was confirmed dispatched at this sector
        self.sector_entry_time = -1.0    # sim-time when we first aligned to current sector
        self.laps_completed = 0
        self.max_laps = max_laps
        profile = FALLBACK_SEARCH_PROFILES.get(target_class, DEFAULT_SEARCH_PROFILE)
        self.search_altitude: float = profile["altitude"]
        self.search_gimbal_pitch: float = profile["gimbal_pitch"]

    def enter(self, drone) -> None:
        super().enter(drone)
        drone.target_alt = self.search_altitude
        logger.info(
            f"[SearchPhase] Profile for '{self.target_class}': "
            f"alt={self.search_altitude}m, gimbal={self.search_gimbal_pitch:.2f}rad"
        )
        drone.target_x = drone.x
        drone.target_y = drone.y
        self.current_sector_idx = 0
        self.at_sector = False
        self.queried_at_sector = False
        self.sector_entry_time = -1.0
        self._set_sector_yaw(drone)

    def _set_sector_yaw(self, drone) -> None:
        if getattr(drone, "vlm_driven", False):
            drone.target_yaw = drone.yaw
            return
        offset_rad = math.radians(self.sector_offsets[self.current_sector_idx])
        drone.target_yaw = normalize_angle(drone.mission_start_yaw + offset_rad)

    def execute(self, drone) -> FlightState:
        drone.target_alt = self.search_altitude
        drone.set_gimbal_pitch(self.search_gimbal_pitch)

        # 1. Check if target already located in SceneGraph
        if self.target_class in drone.scene_graph.beliefs:
            self.completed = True
            logger.info(f"[SearchPhase] ✅ Target '{self.target_class}' found. Phase complete.")
            return FlightState.REASONING

        yaw_diff = normalize_angle(drone.target_yaw - drone.yaw)

        # 2. Still rotating — wait for yaw alignment
        if abs(yaw_diff) > YAW_PRECISION:
            self.at_sector = False
            self.queried_at_sector = False
            self.sector_entry_time = -1.0
            return FlightState.NAVIGATING

        # 3. First time aligned
        if not self.at_sector:
            self.at_sector = True
            self.sector_entry_time = drone.getTime()
            self.queried_at_sector = False
            logger.info(f"[SearchPhase] Aligned to sector {self.sector_offsets[self.current_sector_idx]}° — awaiting VLM query")

        # 4. Wait until VLM query dispatched
        if not self.queried_at_sector:
            if drone.last_vlm_call_time >= self.sector_entry_time:
                self.queried_at_sector = True
                logger.info(f"[SearchPhase] VLM confirmed at sector {self.sector_offsets[self.current_sector_idx]}° ✓")
            else:
                return FlightState.REASONING

        # 5. Advance sector
        self.current_sector_idx += 1
        self.at_sector = False
        self.queried_at_sector = False
        self.sector_entry_time = -1.0

        if self.current_sector_idx < len(self.sector_offsets):
            self._set_sector_yaw(drone)
            logger.info(f"[SearchPhase] Rotating to sector {self.sector_offsets[self.current_sector_idx]}°")
            return FlightState.NAVIGATING

        # 6. Full 360° lap complete
        self.laps_completed += 1
        self.current_sector_idx = 0

        if self.laps_completed >= self.max_laps:
            logger.warning(
                f"[SearchPhase] Target '{self.target_class}' not found after "
                f"{self.max_laps} lap(s). Aborting search phase."
            )
            self.completed = True
            return FlightState.REASONING

        logger.info(f"[SearchPhase] Lap {self.laps_completed} done. Repositioning {REPOSITION_DISTANCE}m forward.")
        drone.target_x = drone.x + REPOSITION_DISTANCE * math.cos(drone.yaw)
        drone.target_y = drone.y + REPOSITION_DISTANCE * math.sin(drone.yaw)
        self._set_sector_yaw(drone)
        return FlightState.NAVIGATING


class NavigatePhase(MissionPhase):
    """Flies toward a SceneGraph target belief, stopping at a specified proximity."""
    def __init__(self, target_class: str, stop_distance: float = 6.0):
        super().__init__(MissionPhaseType.NAVIGATE, target_class)
        self.stop_distance = stop_distance
        self._cached_wp_x: Optional[float] = None
        self._cached_wp_y: Optional[float] = None

    def execute(self, drone) -> FlightState:
        if self.target_class not in drone.scene_graph.beliefs:
            logger.warning(f"[NavigatePhase] Target '{self.target_class}' lost from SceneGraph. Reverting to SEARCH.")
            self.completed = True
            drone.revert_to_search = True
            return FlightState.REASONING

        belief = drone.scene_graph.beliefs[self.target_class]
        target_x, target_y = belief.absolute_position

        dist_to_target = math.sqrt((target_x - drone.x)**2 + (target_y - drone.y)**2)
        if dist_to_target <= self.stop_distance:
            self.completed = True
            logger.info(f"[NavigatePhase] Arrived near '{self.target_class}' (dist: {dist_to_target:.1f}m).")
            drone.target_x = drone.x
            drone.target_y = drone.y
            self._cached_wp_x = None
            self._cached_wp_y = None
            return FlightState.REASONING

        bearing = math.atan2(target_y - drone.y, target_x - drone.x)
        wp_x = target_x - self.stop_distance * math.cos(bearing)
        wp_y = target_y - self.stop_distance * math.sin(bearing)

        if (self._cached_wp_x is None or
                math.sqrt((wp_x - self._cached_wp_x)**2 + (wp_y - self._cached_wp_y)**2) > 1.0):
            self._cached_wp_x = wp_x
            self._cached_wp_y = wp_y
            drone.target_x = wp_x
            drone.target_y = wp_y

        drone.target_yaw = bearing

        # DYNAMIC GIMBAL TRACKING
        horizontal_dist = math.sqrt((target_x - drone.x)**2 + (target_y - drone.y)**2)
        if horizontal_dist > 0.1:
            dynamic_pitch = math.atan2(drone.alt, horizontal_dist)
            drone.set_gimbal_pitch(dynamic_pitch)

        return FlightState.NAVIGATING


class ManeuverPhase(MissionPhase):
    """Executes a high-level spatial maneuver (orbit, tilt camera scan, hover)."""
    def __init__(self, maneuver_type: str, target_class: str = "", duration: float = 4.0, is_final: bool = False):
        super().__init__(MissionPhaseType.MANEUVER, target_class)
        self.maneuver_type = maneuver_type.lower()
        self.duration = duration
        self.is_final = is_final
        self.phase_elapsed = 0.0
        self.orbit_angle = 0.0
        self.orbit_radius = 5.0

    def enter(self, drone) -> None:
        super().enter(drone)
        self.phase_elapsed = 0.0
        self.orbit_angle = drone.yaw
        kind = "Final maneuver" if self.is_final else "Waypoint pause"
        logger.info(f"[ManeuverPhase] {kind} '{self.maneuver_type}' for target '{self.target_class}' (duration: {self.duration:.1f}s)")

    def execute(self, drone) -> FlightState:
        self.phase_elapsed += drone.dt
        if self.phase_elapsed >= self.duration:
            self.completed = True
            if self.is_final:
                logger.info(f"[ManeuverPhase] Final maneuver '{self.maneuver_type}' complete. Mission finished.")
            else:
                logger.info(f"[ManeuverPhase] Waypoint pause at '{self.target_class}' complete. Proceeding to next target phase.")
            return FlightState.REASONING

        if "orbit" in self.maneuver_type or "circle" in self.maneuver_type:
            if self.target_class in drone.scene_graph.beliefs:
                belief = drone.scene_graph.beliefs[self.target_class]
                center_x, center_y = belief.absolute_position
            else:
                center_x, center_y = drone.x, drone.y

            self.orbit_angle += 1.0 * drone.dt  # Speed: 1.0 rad/s
            drone.target_x = center_x + self.orbit_radius * math.cos(self.orbit_angle)
            drone.target_y = center_y + self.orbit_radius * math.sin(self.orbit_angle)
            drone.target_yaw = math.atan2(center_y - drone.y, center_x - drone.x)
            return FlightState.NAVIGATING

        elif "scan" in self.maneuver_type or "tilt" in self.maneuver_type:
            pitch = 0.4 + 0.4 * math.sin(self.phase_elapsed * 0.5)
            drone.set_gimbal_pitch(pitch)
            drone.target_x = drone.x
            drone.target_y = drone.y
            return FlightState.NAVIGATING

class GeometricManeuverPhase(MissionPhase):
    """Executes pure 3D geometric flight paths (triangle, square, circle) without visual target dependencies."""
    def __init__(self, pattern_type: str, side_length: float = 4.0, target_alt: float = 8.0):
        super().__init__(MissionPhaseType.MANEUVER, "geometric_pattern")
        self.pattern_type = pattern_type.lower()
        self.side_length = side_length
        self.target_alt = target_alt
        self.waypoints: List[Tuple[float, float]] = []
        self.current_wp_idx = 0
        self.start_x = 0.0
        self.start_y = 0.0

    def enter(self, drone) -> None:
        super().enter(drone)
        self.start_x = drone.x
        self.start_y = drone.y
        drone.target_alt = self.target_alt
        L = self.side_length

        if "triangle" in self.pattern_type:
            # Equilateral triangle relative to start position
            h = L * math.sqrt(3.0) / 2.0
            self.waypoints = [
                (self.start_x + L * math.cos(drone.yaw), self.start_y + L * math.sin(drone.yaw)),
                (self.start_x + (L/2.0) * math.cos(drone.yaw) - h * math.sin(drone.yaw), self.start_y + (L/2.0) * math.sin(drone.yaw) + h * math.cos(drone.yaw)),
                (self.start_x, self.start_y)
            ]
        elif "square" in self.pattern_type:
            cos_y, sin_y = math.cos(drone.yaw), math.sin(drone.yaw)
            self.waypoints = [
                (self.start_x + L * cos_y, self.start_y + L * sin_y),
                (self.start_x + L * cos_y - L * sin_y, self.start_y + L * sin_y + L * cos_y),
                (self.start_x - L * sin_y, self.start_y + L * cos_y),
                (self.start_x, self.start_y)
            ]
        else:
            self.waypoints = [(self.start_x + L * math.cos(drone.yaw), self.start_y + L * math.sin(drone.yaw)), (self.start_x, self.start_y)]

        self.current_wp_idx = 0
        logger.info(f"[GeometricManeuverPhase] Initialized {len(self.waypoints)} waypoints for '{self.pattern_type}' (side={L}m, alt={self.target_alt}m)")

    def execute(self, drone) -> FlightState:
        drone.target_alt = self.target_alt
        
        # 1. Smooth altitude climb/descent prior to horizontal waypoint navigation
        if abs(drone.alt - self.target_alt) > 0.4:
            drone.target_x = drone.x
            drone.target_y = drone.y
            logger.info(f"[GeometricManeuverPhase] Altitude climb/descent in progress to {self.target_alt:.1f}m (current: {drone.alt:.1f}m)")
            return FlightState.NAVIGATING

        # 2. Sequential waypoint navigation
        if self.current_wp_idx >= len(self.waypoints):
            self.completed = True
            logger.info("[GeometricManeuverPhase] Complete! Geometric pattern finished.")
            return FlightState.REASONING

        wp_x, wp_y = self.waypoints[self.current_wp_idx]
        dist = math.sqrt((wp_x - drone.x)**2 + (wp_y - drone.y)**2)
        if dist < 0.4:
            logger.info(f"[GeometricManeuverPhase] Waypoint {self.current_wp_idx + 1}/{len(self.waypoints)} reached.")
            self.current_wp_idx += 1
            if self.current_wp_idx >= len(self.waypoints):
                self.completed = True
                return FlightState.REASONING

        drone.target_x, drone.target_y = self.waypoints[self.current_wp_idx]
        drone.target_yaw = math.atan2(drone.target_y - drone.y, drone.target_x - drone.x)
        return FlightState.NAVIGATING



class VLMActionSequencePhase(MissionPhase):
    """Executes a sequence of VLM-planned action steps sequentially without single-step oscillation."""
    def __init__(self, action_steps: List[ActionStep]):
        super().__init__(MissionPhaseType.MANEUVER, "vlm_action_sequence")
        self.action_steps = action_steps
        self.current_idx = 0

    def enter(self, drone) -> None:
        super().enter(drone)
        self.current_idx = 0
        logger.info(f"[VLMActionSequencePhase] Initialized {len(self.action_steps)} VLM action steps")

    def execute(self, drone) -> FlightState:
        if self.current_idx >= len(self.action_steps):
            self.completed = True
            logger.info("[VLMActionSequencePhase] Complete! VLM action sequence finished.")
            return FlightState.REASONING

        step = self.action_steps[self.current_idx]
        self.current_idx += 1
        with drone.action_queue_lock:
            drone.action_queue.append(step)
            logger.info(f"[VLMActionSequencePhase] Step {self.current_idx}/{len(self.action_steps)} queued: {step.action} {step.value}")

        return FlightState.REASONING


class MissionPlanner:
    """General-purpose LLM-native mission planner.
    
    Uses two-stage LLM reasoning:
    1. stage 1 (planning): Decomposes any prompt (e.g. 'move to manor and focus on top left window')
       into a structured sequence of sub-goals without hardcoded candidate lists.
    2. stage 2 (execution): Resolves optimal physics search profiles for each sub-goal.
    """
    @staticmethod
    def plan(
        prompt: str,
        vlm_client: Optional[Any] = None,
        provider: str = "",
        api_key: str = "",
        model_name: str = ""
    ) -> List[MissionPhase]:
        p = prompt.lower()
        phases: List[MissionPhase] = []
        ordered_targets: List[str] = []

        # 1. Query LLM for general-purpose sub-goal or action decomposition
        if vlm_client is not None:
            subgoals = vlm_client.plan_mission(prompt, provider, api_key, model_name)
            if isinstance(subgoals, list) and subgoals:
                vlm_actions: List[ActionStep] = []
                for sg in subgoals:
                    if isinstance(sg, dict) and ("action" in sg or "action_sequence" in sg):
                        raw_acts = sg.get("action_sequence", [sg])
                        for raw_act in raw_acts:
                            act = raw_act.get("action", "HOVER").upper()
                            if act == "MOVE_CURVE":
                                val = (float(raw_act.get("distance", 1.0)), float(raw_act.get("degrees", 15.0)))
                            elif act in ("TURN_LEFT", "TURN_RIGHT"):
                                val = float(raw_act.get("degrees", raw_act.get("distance", raw_act.get("value", 90.0))))
                            else:
                                val = float(raw_act.get("distance", raw_act.get("value", 3.0)))
                            gimbal = float(raw_act.get("gimbal_pitch", 0.0))
                            vlm_actions.append(ActionStep(action=act, value=val, gimbal_pitch=gimbal))
                    elif isinstance(sg, dict):
                        t_cls = sg.get("target_class", sg.get("target", sg.get("description", "")))
                        if t_cls:
                            ordered_targets.append(t_cls.strip().lower())

                if vlm_actions:
                    logger.info(f"[MissionPlanner] VLM plan_mission generated {len(vlm_actions)} sequential flight action(s). Executing VLM plan phase.")
                    return [VLMActionSequencePhase(vlm_actions)]


        # 2. Fallback heuristic parsing if LLM planning offline or unreturned
        if not ordered_targets:
            clauses = re.split(r'\b(?:then|and|after that|finally|,)\b', p)
            for clause in clauses:
                match = re.search(
                    r'(?:find|search for|look for|locate|scan for|identify|fly towards|fly to|focus on|inspect)\s+(?:a |an |the )?([\w\s]+)',
                    clause.strip()
                )
                if match:
                    target_phrase = match.group(1).strip()
                    if target_phrase and target_phrase not in ordered_targets:
                        ordered_targets.append(target_phrase)

        # Final fallback if still empty
        if not ordered_targets:
            clean_p = re.sub(r'^(?:fly to|fly towards|search for|find|locate)\s+(?:a |an |the )?', '', p).strip()
            ordered_targets = [clean_p if clean_p else "target"]

        num_targets = len(ordered_targets)
        logger.info(f"[MissionPlanner] Generated {num_targets} LLM sub-goal(s): {ordered_targets}")

        for i, target in enumerate(ordered_targets):
            is_last_target = (i == num_targets - 1)
            
            profile = MissionPlanner._get_search_profile(
                target, vlm_client, provider, api_key, model_name
            )
            
            # 1. Search phase
            search_phase = SearchPhase(target_class=target)
            search_phase.search_altitude = profile["altitude"]
            search_phase.search_gimbal_pitch = profile["gimbal_pitch"]
            phases.append(search_phase)

                # 2. Navigate phase
            phases.append(NavigatePhase(target_class=target, stop_distance=6.0))

            # 3. Maneuver / Pause phase
            if not is_last_target:
                phases.append(ManeuverPhase(maneuver_type="hover", target_class=target, duration=4.0, is_final=False))
            else:
                if "orbit" in p or "circle" in p:
                    phases.append(ManeuverPhase(maneuver_type="orbit", target_class=target, duration=10.0, is_final=True))
                elif "scan" in p or "inspect" in p or "survey" in p or "focus" in p:
                    phases.append(ManeuverPhase(maneuver_type="scan", target_class=target, duration=8.0, is_final=True))
                else:
                    phases.append(ManeuverPhase(maneuver_type="hover", target_class=target, duration=6.0, is_final=True))

        return phases

    @staticmethod
    def _get_search_profile(
        target_class: str,
        vlm_client: Optional[Any],
        provider: str,
        api_key: str,
        model_name: str
    ) -> Dict[str, float]:
        fallback = FALLBACK_SEARCH_PROFILES.get(target_class, DEFAULT_SEARCH_PROFILE)

        if vlm_client is None:
            logger.info(f"[MissionPlanner] No VLM client — using fallback profile for '{target_class}'")
            return fallback

        try:
            profile = vlm_client.query_search_profile(target_class, provider, api_key, model_name)
            logger.info(
                f"[MissionPlanner] LLM profile for '{target_class}': "
                f"alt={profile['altitude']:.1f}m, gimbal={profile['gimbal_pitch']:.2f}rad — "
                f"reason: {profile.get('reason', 'n/a')}"
            )
            return profile
        except Exception as e:
            logger.warning(f"[MissionPlanner] Profile query failed for '{target_class}': {e}. Using fallback.")
            return fallback
