import json
import math
import base64
import urllib.request
import urllib.error
import logging
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from config import ActionStep, FALLBACK_SEARCH_PROFILES, DEFAULT_SEARCH_PROFILE, VLM_TIMEOUT, API_RETRY_DELAY, MAX_API_RETRIES
from utils import normalize_angle, extract_number, sanitize_lmstudio_url

logger = logging.getLogger("MavicVLM")

@dataclass
class TargetBelief:
    """Target object tracking belief state with absolute coordinate calculation."""
    target_class: str
    last_seen_gps: Tuple[float, float]
    last_seen_heading: float
    pixel_location: Tuple[float, float]  # (center_x, center_y) in [-1.0, 1.0]
    confidence: float
    timestamp: float
    estimated_distance: float
    absolute_position: Tuple[float, float]

    @staticmethod
    def calculate_absolute_position(drone_x: float, drone_y: float, drone_yaw: float, 
                                    pixel_x: float, gimbal_pitch: float, altitude: float, 
                                    estimated_dist: float = 15.0) -> Tuple[float, float]:
        # Horizontal FOV of Mavic 2 Pro is approx 60 degrees (1.05 rad)
        hfov = math.radians(60.0)
        bearing_offset = -pixel_x * (hfov / 2.0)  # pixel_x is normalized in [-1.0, 1.0]
        absolute_bearing = normalize_angle(drone_yaw + bearing_offset)

        # Distance estimation:
        if gimbal_pitch > 0.1:
            try:
                ground_dist = altitude / math.tan(gimbal_pitch)
                dist = max(2.0, min(100.0, ground_dist))
            except ZeroDivisionError:
                dist = estimated_dist
        else:
            dist = estimated_dist

        proj_x = drone_x + dist * math.cos(absolute_bearing)
        proj_y = drone_y + dist * math.sin(absolute_bearing)
        return proj_x, proj_y


class SceneGraph:
    """Maintains active spatial beliefs and filters/refines target positions over time."""
    def __init__(self):
        self.beliefs: Dict[str, TargetBelief] = {}

    def update(self, object_class: str, drone_x: float, drone_y: float, drone_yaw: float,
               pixel_x: float, pixel_y: float, gimbal_pitch: float, altitude: float,
               confidence: float, timestamp: float) -> TargetBelief:
        # Calculate absolute position
        abs_x, abs_y = TargetBelief.calculate_absolute_position(
            drone_x, drone_y, drone_yaw, pixel_x, gimbal_pitch, altitude
        )

        if object_class in self.beliefs:
            prev = self.beliefs[object_class]
            # Simple alpha filter to smooth out noise
            alpha = 0.5
            refined_x = prev.absolute_position[0] * (1.0 - alpha) + abs_x * alpha
            refined_y = prev.absolute_position[1] * (1.0 - alpha) + abs_y * alpha
            
            belief = TargetBelief(
                target_class=object_class,
                last_seen_gps=(drone_x, drone_y),
                last_seen_heading=drone_yaw,
                pixel_location=(pixel_x, pixel_y),
                confidence=confidence,
                timestamp=timestamp,
                estimated_distance=math.sqrt((refined_x - drone_x)**2 + (refined_y - drone_y)**2),
                absolute_position=(refined_x, refined_y)
            )
        else:
            belief = TargetBelief(
                target_class=object_class,
                last_seen_gps=(drone_x, drone_y),
                last_seen_heading=drone_yaw,
                pixel_location=(pixel_x, pixel_y),
                confidence=confidence,
                timestamp=timestamp,
                estimated_distance=15.0,
                absolute_position=(abs_x, abs_y)
            )

        self.beliefs[object_class] = belief
        return belief


@dataclass
class TargetMemory:
    """Inter-frame spatial memory bridging visual perception gaps (Fix 1)."""
    last_bbox: Tuple[float, float] = (0.0, 0.0)  # (center_x, center_y) in [-1.0, 1.0]
    last_seen_timestamp: float = 0.0
    last_seen_edge: str = "center"  # 'left_edge', 'right_edge', 'top_edge', 'bottom_edge', 'center'
    consecutive_misses: int = 0
    pixel_velocity: Tuple[float, float] = (0.0, 0.0)  # (vx, vy) per second
    coasting_active: bool = False
    coast_start_time: float = 0.0


class TargetMemoryTracker:
    """Maintains target state memory between frames to eliminate unanchored flight oscillations."""
    def __init__(self):
        self.memory: Dict[str, TargetMemory] = {}

    def update(self, target_class: str, visible: bool, box_x: float, box_y: float, timestamp: float) -> TargetMemory:
        key = target_class.lower().strip() or "active_target"
        if key not in self.memory:
            self.memory[key] = TargetMemory()

        mem = self.memory[key]

        if visible:
            # Determine edge proximity (Fix 3)
            edge = "center"
            if box_x < -0.7: edge = "left_edge"
            elif box_x > 0.7: edge = "right_edge"
            elif box_y < -0.7: edge = "top_edge"
            elif box_y > 0.7: edge = "bottom_edge"

            dt = max(0.1, timestamp - mem.last_seen_timestamp) if mem.last_seen_timestamp > 0 else 1.0
            vx = (box_x - mem.last_bbox[0]) / dt if mem.last_seen_timestamp > 0 else 0.0
            vy = (box_y - mem.last_bbox[1]) / dt if mem.last_seen_timestamp > 0 else 0.0

            mem.last_bbox = (box_x, box_y)
            mem.last_seen_timestamp = timestamp
            mem.last_seen_edge = edge
            mem.consecutive_misses = 0
            mem.pixel_velocity = (vx, vy)
            mem.coasting_active = False
        else:
            mem.consecutive_misses += 1
            if mem.consecutive_misses == 1:
                mem.coasting_active = True
                mem.coast_start_time = timestamp

        return mem

    def get_memory(self, target_class: str) -> Optional[TargetMemory]:
        return self.memory.get(target_class.lower().strip(), None)



def is_valid_target_match(detected_label: str, target_class: str) -> bool:
    """Semantic parent-child subfeature matching (e.g. window/door in house/manor, wheel in car)."""
    d = detected_label.lower().strip()
    t = target_class.lower().strip()
    if not d or not t:
        return True
    if t in d or d in t:
        return True

    subfeature_parents = {
        "window": ["house", "manor", "building", "home", "facade", "structure", "wall"],
        "door": ["house", "manor", "building", "home", "entrance", "facade"],
        "roof": ["house", "manor", "building", "home", "chimney"],
        "wheel": ["car", "tesla", "vehicle", "automobile", "truck"],
        "tire": ["car", "tesla", "vehicle", "automobile", "truck"],
        "windturbine": ["windmill", "blade", "tower", "generator"],
        "blade": ["windmill", "windturbine", "generator", "tower"]
    }

    for feature, parents in subfeature_parents.items():
        if feature in t and any(p in d for p in parents):
            return True
        if feature in d and any(p in t for p in parents):
            return True

    return False


@dataclass
class VerificationResult:
    """Stage 5 Verification result comparing visual predictions against actual camera observation."""
    passed: bool
    reasoning: str
    observed_target: str
    recommended_replan: bool


class ActionVerifier:
    """Verifies post-step camera observation against predicted expectations (MultiUAV-Plat Stage 5)."""
    @staticmethod
    def verify(expected_obs: str, actual_target_visible: bool, actual_detected_object: str, active_target_class: str) -> VerificationResult:
        # 1. Target expected & visually confirmed or matched via parent-child subfeature
        if actual_target_visible and is_valid_target_match(actual_detected_object, active_target_class):
            return VerificationResult(
                passed=True,
                reasoning=f"Verification Passed: Target/Feature '{actual_detected_object}' visually confirmed for '{active_target_class}'.",
                observed_target=actual_detected_object,
                recommended_replan=False
            )
        # 2. Exploration/approach step — target not yet in view
        if not actual_target_visible:
            return VerificationResult(
                passed=True,
                reasoning=f"Verification Normal: Approach step completed, target area under observation. Expected: '{expected_obs}'",
                observed_target="none",
                recommended_replan=False
            )
        # 3. Distracted by unexpected object (e.g. searching for tesla, saw windmill)
        if actual_target_visible and not is_valid_target_match(actual_detected_object, active_target_class):
            return VerificationResult(
                passed=False,
                reasoning=f"Verification Alert: Expected '{active_target_class}', but camera observed unrelated '{actual_detected_object}'. Re-plan triggered.",
                observed_target=actual_detected_object,
                recommended_replan=True
            )
        return VerificationResult(passed=True, reasoning="Verification OK", observed_target=actual_detected_object, recommended_replan=False)


class VLMResponseHandler:
    """Handles VLM response cleaning, schema parsing, validation, and error recovery."""
    def __init__(self, default_gimbal_pitch: float = 0.0):
        self.default_gimbal_pitch = default_gimbal_pitch

    def clean_response(self, text: str) -> str:
        # Strip markdown fences if present
        text = text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        # Strip <think> tags
        import re
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        
        # Clean trailing commas in JSON lists/objects
        text = re.sub(r",(\s*[}\]])", r"\1", text)
        
        # Handle both JSON array [...] and JSON object {...}
        bracket_start = text.find("[")
        brace_start = text.find("{")

        if bracket_start != -1 and (brace_start == -1 or bracket_start < brace_start):
            bracket_end = text.rfind("]")
            if bracket_end != -1:
                return text[bracket_start:bracket_end + 1].strip()

        if brace_start != -1:
            brace_end = text.rfind("}")
            if brace_end != -1:
                return text[brace_start:brace_end + 1].strip()

        return text.strip()

    def parse_and_validate(self, text: str) -> Tuple[List[ActionStep], str, bool, str, float, float, float, str]:
        cleaned = self.clean_response(text)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error(f"VLM response is not valid JSON: {e}. Cleaned text: {cleaned[:200]}")
            raise ValueError(f"JSON decode failed: {e}")

        reasoning = data.get("reasoning", "VLM parsed sequence")
        expected_observation = data.get("expected_observation", "Observe target area").strip()
        
        # Support both new perception fields and old target_found field
        target_visible = bool(data.get("target_visible", data.get("target_found", False)))
        detected_object = data.get("detected_object", "").strip()

        # Bounding box centers in range [-1.0, 1.0]
        try:
            box_center_x = float(data.get("box_center_x", 0.0))
            box_center_y = float(data.get("box_center_y", 0.0))
        except (ValueError, TypeError):
            box_center_x = 0.0
            box_center_y = 0.0

        try:
            confidence = float(data.get("confidence", 1.0 if target_visible else 0.0))
        except (ValueError, TypeError):
            confidence = 1.0 if target_visible else 0.0

        raw_sequence = data.get("action_sequence", [])
        if not raw_sequence and "action" in data:
            raw_sequence = [data]

        # Action Confidence Gating (Fix 5): Downgrade unanchored body translation if low confidence
        if not target_visible and confidence < 0.7:
            for step in raw_sequence:
                act = step.get("action", "").upper()
                if act.startswith("MOVE_") or act in ("CLIMB", "DESCENT"):
                    logger.info(f"[Action Gating] Low confidence ({confidence:.2f}) without visual anchor. Downgrading '{act}' to HOVER.")
                    step["action"] = "HOVER"
                    step["value"] = 0.0

        # Enforce single movement step per call (with future intent stated in reasoning)
        if raw_sequence:
            raw_sequence = raw_sequence[:1]

        action_list: List[ActionStep] = []
        valid_actions = {
            "MOVE_FORWARD", "MOVE_BACKWARD", "MOVE_LEFT", "MOVE_RIGHT",
            "TURN_LEFT", "TURN_RIGHT", "MOVE_CURVE", "HOVER", "LAND",
            "CLIMB", "DESCENT"
        }

        for step in raw_sequence:
            action = step.get("action", "HOVER").upper()
            if action not in valid_actions:
                action = "HOVER"

            if action == "MOVE_CURVE":
                try:
                    dist = min(3.0, max(0.5, float(step.get("distance", 1.0))))
                    yaw = min(25.0, max(5.0, float(step.get("yaw_angle", step.get("angle", step.get("degrees", 15.0))))))
                    val = (dist, yaw)
                except (ValueError, TypeError):
                    val = (1.0, 15.0)
            else:
                if action.startswith("TURN_"):
                    try:
                        raw_ang = float(step.get("degrees", step.get("yaw_angle", step.get("angle", step.get("distance", step.get("value", 15.0))))))
                        val = min(90.0, max(0.5, raw_ang))
                    except (ValueError, TypeError):
                        val = 15.0
                elif action in ("CLIMB", "DESCENT"):
                    try:
                        raw_dist = float(step.get("distance", step.get("value", 0.5)))
                        val = min(0.8, max(0.2, raw_dist))
                    except (ValueError, TypeError):
                        val = 0.5
                else:
                    try:
                        raw_dist = float(step.get("distance", step.get("value", 3.0)))
                        val = min(3.0, max(0.5, raw_dist))
                    except (ValueError, TypeError):
                        val = 3.0

            try:
                gimbal = max(0.0, min(0.8, float(step.get("gimbal_pitch", self.default_gimbal_pitch))))
            except (ValueError, TypeError):
                gimbal = self.default_gimbal_pitch

            action_list.append(ActionStep(action=action, value=val, gimbal_pitch=gimbal))

        if not action_list:
            action_list.append(ActionStep(action="HOVER", value=0.0, gimbal_pitch=self.default_gimbal_pitch))

        return action_list, reasoning, target_visible, detected_object, box_center_x, box_center_y, confidence, expected_observation


class ActionProjector:
    """Helper to project relative flight actions to absolute space for safety checking."""
    @staticmethod
    def project(action: str, value: Any, current_x: float, current_y: float, current_yaw: float) -> Tuple[float, float, float]:
        """Returns projected (next_x, next_y, next_yaw) coordinates."""
        if action == "HOVER" or action == "LAND":
            return current_x, current_y, current_yaw

        if action == "MOVE_CURVE":
            dist, angle_deg = value
            angle_rad = math.radians(angle_deg)
            if abs(angle_rad) < 0.001:
                proj_x = current_x + dist * math.cos(current_yaw)
                proj_y = current_y + dist * math.sin(current_yaw)
            else:
                R = dist / angle_rad
                C = 2.0 * R * math.sin(angle_rad / 2.0)
                chord_angle = current_yaw + angle_rad / 2.0
                proj_x = current_x + C * math.cos(chord_angle)
                proj_y = current_y + C * math.sin(chord_angle)
            proj_yaw = normalize_angle(current_yaw + angle_rad)
            return proj_x, proj_y, proj_yaw

        if action.startswith("MOVE_"):
            move_yaw = current_yaw
            if action == "MOVE_BACKWARD":
                move_yaw += math.pi
            elif action == "MOVE_LEFT":
                move_yaw += math.pi / 2.0
            elif action == "MOVE_RIGHT":
                move_yaw -= math.pi / 2.0

            dist = float(value)
            proj_x = current_x + dist * math.cos(move_yaw)
            proj_y = current_y + dist * math.sin(move_yaw)
            return proj_x, proj_y, current_yaw

        if action.startswith("TURN_"):
            angle_rad = math.radians(float(value))
            if action == "TURN_LEFT":
                proj_yaw = normalize_angle(current_yaw + angle_rad)
            else:
                proj_yaw = normalize_angle(current_yaw - angle_rad)
            return current_x, current_y, proj_yaw

        return current_x, current_y, current_yaw


class VLMClient:
    """Encapsulates all VLM API interactions with retry logic and structured parsing."""

    def __init__(self, timeout: float = VLM_TIMEOUT):
        self.timeout = timeout
        self.system_instruction = self._build_system_instruction()

    PLANNING_SYSTEM = (
        "You are an autonomous UAV mission planner. Given a user's mission prompt, "
        "decompose it into a JSON array of sequential sub-goals or actions.\n"
        "RULES:\n"
        "- For motion patterns or spatial tasks (e.g. 'patrol 10m square', 'make triangle motion of side 4m at 8m altitude'):\n"
        "  Decompose into an explicit sequence of action steps: output objects with keys 'action' ('CLIMB', 'MOVE_FORWARD', 'TURN_RIGHT', 'TURN_LEFT'), 'distance' (float in meters), 'degrees' (float in degrees: 120° for triangle, 90° for square).\n"
        "- For visual object search tasks (e.g. 'find manor and focus on window'):\n"
        "  Output target classes: 'target_class': 1. Macro target ('manor'), 2. Sub-feature inspection ('window').\n"
        "Return ONLY a valid JSON array of objects."
    )


    def _build_system_instruction(self) -> str:
        return (
            "ROLE:\n"
            "You are the visual perception and spatial control module for an autonomous search and rescue UAV.\n\n"
            "UAV CAPABILITIES & ACTIONS:\n"
            "- MOVE_FORWARD, MOVE_BACKWARD, MOVE_LEFT, MOVE_RIGHT (distance in meters, max 3.0m per step)\n"
            "- TURN_LEFT, TURN_RIGHT (specify exact 'degrees' input: e.g. 3.0°–5.0° for fine-tuning alignment, 15.0° for standard turn, 30.0° for wide scan)\n"
            "- CLIMB, DESCENT (altitude adjustment in meters, max 0.5m per step)\n"
            "- GIMBAL_PITCH (camera pitch in radians: 0.0=horizon, 0.35=shallow ground, 0.8=nadir straight down)\n"
            "- HOVER, LAND\n\n"
            "GROUND TRUTH & NAVIGATION RULES:\n"
            "1. Single Movement Rule: Output EXACTLY ONE single action step in 'action_sequence' per call. Do NOT output multiple steps.\n"
            "2. Altitude Precedence Rule: If mission prompt specifies an altitude (e.g. 'at 8m altitude'), issue CLIMB or DESCENT FIRST until target altitude is reached before executing horizontal translation.\n"
            "3. Geometric Shape Rule: For shape maneuvers, calculate exact exterior turn angles (Triangle = 120° turn; Square = 90° turn). Do NOT hallucinate unrelated ground objects (e.g. cars, houses) as targets during pure motion tasks.\n"
            "4. Out-of-Frame / Missing Target Rule: If target is NOT visible, do NOT hallucinate bounding boxes. Select gimbal scan or small turn.\n"
            "5. Gimbal-First Rule: If target is near edge, prioritize camera gimbal pitch adjustment.\n"
            "6. Future Intent & Plan: Describe complete future navigation plan in 'reasoning'.\n"
            "7. Ground Truth Rule: Only set box_center_x and box_center_y to non-zero values when target/sub-feature is ACTUALLY visible.\n\n"

            "OUTPUT SCHEMA:\n"
            "{\n"
            "  \"reasoning\": \"Describe current visual state, current single action, and planned future movements.\",\n"
            "  \"target_visible\": true/false,\n"
            "  \"detected_object\": \"Name of object or feature detected\",\n"
            "  \"expected_observation\": \"What the camera frame should observe after executing this single movement step\",\n"
            "  \"box_center_x\": float (-1.0 to 1.0, center 0.0),\n"
            "  \"box_center_y\": float (-1.0 to 1.0, center 0.0),\n"
            "  \"confidence\": float (0.0 to 1.0),\n"
            "  \"action_sequence\": [ {\"action\": \"MOVE_FORWARD\", \"distance\": 2.5, \"gimbal_pitch\": 0.2} ]\n"
            "}"
        )

    PROFILE_SYSTEM = (
        "You are a UAV mission planner. Given an object to search for, determine the optimal "
        "search altitude and camera gimbal pitch for a drone to visually detect it.\n"
        "PHYSICS RULES (Crucial for Ground Coverage):\n"
        "- The camera has a 45-degree vertical field of view.\n"
        "- altitude: 2.0–10.0 metres above ground\n"
        "- gimbal_pitch: 0.0 (horizontal) to 0.8 rad (straight down)\n"
        "- A steep pitch (e.g., 0.6 to 0.8 rad) acts like a soda straw, only seeing a tiny square of ground directly below. Use ONLY for tiny hidden objects.\n"
        "- A shallow pitch (0.2 to 0.4 rad) allows the camera to see the ground stretching far to the horizon, maximizing search area. Use for cars, roads, people, and general ground search.\n"
        "- For tall objects (windmills, towers): use high altitude (8-10m) and horizon pitch (0.0 rad).\n"
        "- For large ground objects (cars, buildings): use medium altitude (6-8m) and shallow pitch (0.2-0.4 rad).\n"
        "Return ONLY valid JSON with exactly these fields:\n"
        "{\"altitude\": <float>, \"gimbal_pitch\": <float>, \"reason\": \"<one sentence>\"}"
    )

    def _get_api_config(self, provider, key, model_name, temp, prompt, sys_prompt, img_data=None):
        """Dynamic payload builder to eliminate duplicate code across providers."""
        if provider == "ollama":
            return (
                f"{key}/api/generate" if key else "http://127.0.0.1:11434/api/generate", {},
                {"model": model_name or "qwen2-vl", "prompt": prompt, "stream": False, "format": "json", "options": {"temperature": temp}, **({"images": [img_data]} if img_data else {})}
            )
        if provider == "lmstudio":
            return (
                sanitize_lmstudio_url(key) if key else "http://127.0.0.1:1234/v1/chat/completions", {},
                {"model": model_name or "qwen/qwen3-vl-4b", "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": [{"type": "text", "text": prompt}] + ([{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_data}"}}] if img_data else []) if img_data else prompt}], "temperature": temp, "stream": False}
            )
        if provider == "qwen-dashscope":
            if not key: raise ValueError("DashScope API key missing")
            return (
                "https://dashscope.aliyuncs.com/api/v1/services/aigc/" + ("multimodal-generation/generation" if img_data else "text-generation/generation"),
                {"Authorization": f"Bearer {key}"},
                {"model": model_name or ("qwen-vl-plus" if img_data else "qwen-turbo"), "input": {"messages": [{"role": "user", "content": [{"image": f"data:image/jpeg;base64,{img_data}"}, {"text": prompt}]} if img_data else {"role": "user", "content": prompt}] if img_data else [{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}]}, "parameters": {"result_format": "message", "temperature": temp}}
            )
        # gemini
        if not key: raise ValueError("Gemini API key missing")
        return (
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={key}", {},
            {"contents": [{"parts": [{"text": prompt}] + ([{"inlineData": {"mimeType": "image/jpeg", "data": img_data}}] if img_data else [])}], "generationConfig": {"responseMimeType": "application/json", "temperature": temp}}
        )

    def plan_mission(self, prompt: str, provider: str, key: str, model_name: str) -> List[Dict[str, Any]]:
        """Text-only LLM planning call to break complex prompt into sequential sub-goals."""
        user_prompt = f"Mission prompt to decompose into sequential sub-goals: \"{prompt}\""
        url, hdrs, payload = self._get_api_config(provider, key, model_name, 0.1, user_prompt, self.PLANNING_SYSTEM)
        try:
            raw = self._send_raw(url, payload, provider, hdrs)
            handler = VLMResponseHandler()
            cleaned = handler.clean_response(raw)
            data = json.loads(cleaned)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return data.get("subgoals", data.get("plan", data.get("sub_goals", [data])))
            return []
        except Exception as e:
            logger.warning(f"[VLMClient] plan_mission query failed: {e}")
            return []

    def query_search_profile(self, target_class: str, provider: str, key: str, model_name: str) -> dict[str, float]:
        url, hdrs, payload = self._get_api_config(provider, key, model_name, 0.1, f"Object to search for: \"{target_class}\"", self.PROFILE_SYSTEM)
        try:
            raw = self._send_raw(url, payload, provider, hdrs)
            data = json.loads(VLMResponseHandler().clean_response(raw))
            return {"altitude": max(2.0, min(10.0, float(data.get("altitude", DEFAULT_SEARCH_PROFILE["altitude"])))), "gimbal_pitch": max(0.0, min(0.8, float(data.get("gimbal_pitch", DEFAULT_SEARCH_PROFILE["gimbal_pitch"])))), "reason": data.get("reason", "")}
        except Exception as e:
            logger.warning(f"[VLMClient] Profile query failed for '{target_class}': {e}")
            raise

    def query(self, prompt: str, provider: str, key: str, model_name: str, image_path: str, context_block: str, default_gimbal_pitch: float = 0.0, image_base64: str = "") -> Tuple[List[ActionStep], str, bool, str, float, float, float, str]:
        img_data = image_base64
        if not img_data:
            try:
                import threading
                # inline lock skipping for brevity since this is local
                with open(image_path, "rb") as f: img_data = base64.b64encode(f.read()).decode("utf-8")
            except Exception as e:
                return self._fallback(prompt)

        handler = VLMResponseHandler(default_gimbal_pitch=default_gimbal_pitch)
        for attempt, temp in enumerate([0.3, 0.1, 0.0]):
            try:
                url, hdrs, payload = self._get_api_config(provider, key, model_name, temp, f"{self.system_instruction}\n\nUser Mission: {prompt}\n{context_block}", self.system_instruction, img_data)
                return handler.parse_and_validate(self._send_raw(url, payload, provider, hdrs))
            except Exception as e:
                logger.warning(f"VLM query failed (attempt {attempt + 1}): {e}")
                if attempt < 2: import time; time.sleep(API_RETRY_DELAY * (2 ** attempt))
                else: return self._fallback(prompt)
        return self._fallback(prompt)

    def _send_raw(self, url: str, payload: dict, provider: str, hdrs: dict = None) -> str:
        headers = {"Content-Type": "application/json"}
        if hdrs: headers.update(hdrs)
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            if provider == "ollama": return body.get("response", "")
            if provider in ("lmstudio", "dashscope"):
                return body.get("choices", [{}])[0].get("message", {}).get("content", body.get("choices", [{}])[0].get("text", body.get("response", body.get("content", body.get("text", "")))))
            return body.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")

    def _fallback(self, prompt: str) -> Tuple[List[ActionStep], str, bool, str, float, float, float, str]:
        p = prompt.lower()
        if "circle" in p: return [ActionStep("MOVE_CURVE", (2.0 * math.pi * extract_number(p, 1.0) / 4.0, 90.0))], "Fallback", False, "", 0.0, 0.0, 0.0, "Observe area"
        for word, action in [("forward", "MOVE_FORWARD"), ("back", "MOVE_BACKWARD"), ("left", "MOVE_LEFT"), ("right", "MOVE_RIGHT")]:
            if word in p and "turn" not in p: return [ActionStep(action, extract_number(p, 4.0))], f"Fallback {word}", False, "", 0.0, 0.0, 0.0, "Observe area"
        if "turn left" in p: return [ActionStep("TURN_LEFT", extract_number(p, 90.0))], "Fallback turn left", False, "", 0.0, 0.0, 0.0, "Observe area"
        if "turn right" in p: return [ActionStep("TURN_RIGHT", extract_number(p, 90.0))], "Fallback turn right", False, "", 0.0, 0.0, 0.0, "Observe area"
        if "land" in p: return [ActionStep("LAND", 0.0)], "Fallback land", False, "", 0.0, 0.0, 0.0, "Observe area"
        return [ActionStep("HOVER", 0.0)], "No pattern", False, "", 0.0, 0.0, 0.0, "Observe area"
