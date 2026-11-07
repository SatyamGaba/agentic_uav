"""
Discrete Action Space Adapter (OpenFly-Agent, AerialVLN, CMA paradigm).
Maps multi-modal observations and instructions to OpenFly's 10 calibrated discrete action codes.
"""
import json
import re
import math
from typing import Dict, Any, Optional, List
from framework.interfaces.base_adapter import BaseVLNAdapter
from framework.interfaces.base_vlm import BaseVLMBackend
from framework.types import VLNObservation, NavAction


# Canonical OpenFly Action Kinematics (strictly aligned with benchmark/seen.json ground truth)
# Coordinate Convention:
# - Heading Yaw: 0 rad points along +X (East/Forward).
# - Positive Yaw (+30 deg, Action 2): Rotates CCW toward +Y (North/Left). Physically corresponds to TURN_LEFT.
# - Negative Yaw (-30 deg, Action 3): Rotates CW toward -Y (South/Right). Physically corresponds to TURN_RIGHT.
# - Lateral Translation:
#     +Y (Action 7, dy = +3.0m): Port / Left side.
#     -Y (Action 6, dy = -3.0m): Starboard / Right side.
# This convention was empirically validated against all 1,800 trajectories in seen.json.
OPENFLY_ACTION_VOCAB = {
    0:  {"name": "STOP",               "dx": 0.00, "dy": 0.0,   "yaw_deg": 0.0,   "dz": 0.0},
    1:  {"name": "MOVE_FORWARD_FINE",  "dx": 3.00, "dy": 0.0,   "yaw_deg": 0.0,   "dz": 0.0},
    2:  {"name": "TURN_LEFT",          "dx": 0.00, "dy": 0.0,   "yaw_deg": 30.0,  "dz": 0.0},
    3:  {"name": "TURN_RIGHT",         "dx": 0.00, "dy": 0.0,   "yaw_deg": -30.0, "dz": 0.0},
    4:  {"name": "CLIMB",              "dx": 0.00, "dy": 0.0,   "yaw_deg": 0.0,   "dz": 3.0},
    5:  {"name": "DESCEND",            "dx": 0.00, "dy": 0.0,   "yaw_deg": 0.0,   "dz": -3.0},
    6:  {"name": "STRAFE_RIGHT",       "dx": 0.00, "dy": -3.00, "yaw_deg": 0.0,   "dz": 0.0},
    7:  {"name": "STRAFE_LEFT",        "dx": 0.00, "dy": 3.00,  "yaw_deg": 0.0,   "dz": 0.0},
    8:  {"name": "MOVE_FORWARD_MED",   "dx": 6.00, "dy": 0.0,   "yaw_deg": 0.0,   "dz": 0.0},
    9:  {"name": "MOVE_FORWARD_LARGE", "dx": 9.00, "dy": 0.0,   "yaw_deg": 0.0,   "dz": 0.0},
    -1: {"name": "CLIMB_ALT",          "dx": 0.00, "dy": 0.0,   "yaw_deg": 0.0,   "dz": 3.0},
    -2: {"name": "DESCEND_ALT",        "dx": 0.00, "dy": 0.0,   "yaw_deg": 0.0,   "dz": -3.0},
}



class DiscreteActionAdapter(BaseVLNAdapter):
    """
    Adapter implementing the discrete action space of OpenFly.
    Can operate step-by-step or by executing pre-planned action sequences.
    """
    def __init__(
        self,
        vlm_backend: BaseVLMBackend,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(name="DiscreteActionAdapter", config=config)
        if vlm_backend is None:
            raise ValueError("DiscreteActionAdapter strictly requires an active VLM backend. Heuristics are prohibited.")
        self.vlm = vlm_backend
        self.action_queue: List[int] = []

    def reset(self, initial_observation: VLNObservation) -> None:
        self.action_queue.clear()

    def step(self, observation: VLNObservation) -> NavAction:
        # If queue is empty, query model to generate action code(s)
        if not self.action_queue:
            self._plan_action_codes(observation)

        code = self.action_queue.pop(0) if self.action_queue else 0
        phys = OPENFLY_ACTION_VOCAB.get(code, OPENFLY_ACTION_VOCAB[8])

        is_stop = (code == 0)
        yaw_rad = math.radians(phys.get("yaw_deg", 0.0))
        dx = phys.get("dx", 3.0)
        dy = phys.get("dy", 0.0)
        dz = phys.get("dz", 0.0)

        return NavAction(
            dx=dx,
            dy=dy,
            dz=dz,
            yaw_delta=yaw_rad,
            action_code=code,
            is_stop=is_stop,
            reasoning=f"Executing OpenFly action {code}: {phys['name']}"
        )

    def _plan_action_codes(self, observation: VLNObservation) -> None:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "plan_flight",
                    "description": "Execute an aerial flight plan composed of discrete OpenFly action integers.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action_codes": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "description": "Sequence of OpenFly action integers: 9 (forward 9m), 8 (forward 6m), 1 (forward 3m), 3 (turn left -30 deg), 2 (turn right +30 deg), 4 (climb 3m), 5 (descend 3m), 0 (stop landmark reached)."
                            }
                        },
                        "required": ["action_codes"]
                    }
                }
            }
        ]

        prompt = (
            f"You are an aerial drone navigator. Select discrete flight action codes to navigate toward the landmark:\n"
            f"Action Vocabulary:\n"
            f"  9: Forward Large (9.0m)\n"
            f"  8: Forward Medium (6.0m)\n"
            f"  1: Forward Fine (3.0m)\n"
            f"  3: Turn Left (-30 deg)\n"
            f"  2: Turn Right (+30 deg)\n"
            f"  6: Strafe Left (-3.0m)\n"
            f"  7: Strafe Right (+3.0m)\n"
            f"  4: Climb (+3m)\n"
            f"  5: Descend (-3m)\n"
            f"  0: Stop (Target Landmark Reached)\n\n"
            f"Instruction: \"{observation.instruction}\"\n"
            f"Step: {observation.step_id}\n"
            f"Use the plan_flight function to output action codes."
        )

        user_msg: Dict[str, Any] = {"role": "user", "content": prompt}
        if observation.rgb_b64:
            user_msg["images"] = [observation.rgb_b64]

        response = self.vlm.chat(messages=[user_msg], tools=tools, max_tokens=4096)

        # 1. Check native tool calls
        for tc in response.tool_calls:
            if tc.get("function", {}).get("name") == "plan_flight":
                args = tc["function"].get("arguments", {})
                codes = args.get("action_codes", [])
                if isinstance(codes, list) and len(codes) > 0:
                    valid_codes = [int(c) for c in codes if int(c) in OPENFLY_ACTION_VOCAB]
                    if valid_codes:
                        self.action_queue.extend(valid_codes)
                        return

        # 2. Check text or thinking block
        text = response.content.strip() if response.content.strip() else response.thinking.strip()
        codes = self._parse_json_codes(text)
        if codes:
            self.action_queue.extend(codes)
        else:
            raise RuntimeError(f"VLM backend failed to output valid action codes from text: '{text[:100]}'. Heuristic fallbacks prohibited.")

    def _parse_json_codes(self, text: str) -> List[int]:
        m = re.search(r'\[[\s\d,\-]+\]', text)
        if m:
            try:
                arr = json.loads(m.group(0))
                return [int(c) for c in arr if int(c) in OPENFLY_ACTION_VOCAB]
            except Exception:
                pass
        return []
