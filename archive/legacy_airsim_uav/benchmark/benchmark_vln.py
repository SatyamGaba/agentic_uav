import os
import sys
import json
import math
import argparse
import logging
from typing import List, Dict, Any, Tuple

# Add controller path to sys.path
CONTROLLER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "controllers", "mavic_vlm_controller")
if CONTROLLER_DIR not in sys.path:
    sys.path.insert(0, CONTROLLER_DIR)

from config import ActionStep
from utils import normalize_angle

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("OpenFlyBenchmark")

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# OPENFLY GROUND-TRUTH ACTION CODE PHYSICS DEFINITIONS
# ============================================================================
# Decoded directly from OpenFly dataset pose.jsonl and empirical trajectory deltas:
ACTION_PHYSICS = {
    9:  {"name": "MOVE_FORWARD_LARGE", "dist": 7.64, "yaw_delta_deg": 0.0,   "dz": 0.0},
    8:  {"name": "MOVE_FORWARD_MED",   "dist": 3.00, "yaw_delta_deg": 0.0,   "dz": 0.0},
    1:  {"name": "MOVE_FORWARD_FINE",  "dist": 3.00, "yaw_delta_deg": -2.0,  "dz": 0.0},
    3:  {"name": "TURN_LEFT_ARC",      "dist": 3.34, "yaw_delta_deg": -30.0, "dz": 0.0},  # Left is -yaw in OpenFly coordinate frame
    2:  {"name": "TURN_RIGHT_ARC",     "dist": 3.32, "yaw_delta_deg": 30.0,  "dz": 0.0},  # Right is +yaw in OpenFly coordinate frame
    -1: {"name": "CLIMB",              "dist": 0.00, "yaw_delta_deg": 0.0,   "dz": 3.0},
    4:  {"name": "CLIMB_ALT",          "dist": 0.00, "yaw_delta_deg": 0.0,   "dz": 3.0},
    -2: {"name": "DESCENT",            "dist": 0.00, "yaw_delta_deg": 0.0,   "dz": -3.0},
    5:  {"name": "DESCENT_ALT",        "dist": 0.00, "yaw_delta_deg": 0.0,   "dz": -3.0},
    0:  {"name": "STOP",               "dist": 0.00, "yaw_delta_deg": 0.0,   "dz": 0.0},
}

def calculate_path_length(points: List[List[float]]) -> float:
    """Calculates total 3D Euclidean length along a sequence of waypoints."""
    if len(points) < 2:
        return 0.0
    total = 0.0
    for i in range(len(points) - 1):
        dx = points[i+1][0] - points[i][0]
        dy = points[i+1][1] - points[i][1]
        dz = points[i+1][2] - points[i][2]
        total += math.sqrt(dx*dx + dy*dy + dz*dz)
    return total

def reconstruct_true_start_pos(pos_0: List[float], yaw_0: float, action_0: int) -> List[float]:
    """
    Reconstructs the true pre-flight initial position P0 before action_0 was executed.
    In OpenFly, pos[0] is the position AFTER executing action[0].
    """
    phys = ACTION_PHYSICS.get(action_0, ACTION_PHYSICS[8])
    dist = phys["dist"]
    dz = phys["dz"]
    
    # Invert the first movement
    x0 = pos_0[0] - dist * math.cos(yaw_0)
    y0 = pos_0[1] - dist * math.sin(yaw_0)
    z0 = pos_0[2] - dz
    return [x0, y0, z0]

def rollout_openfly_physics(
    true_start_pos: List[float],
    true_start_yaw: float,
    action_codes: List[int],
    max_steps: int = 45
) -> Tuple[List[List[float]], List[float], float]:
    """
    Simulates the drone kinematics adhering strictly to OpenFly's discrete action physics.
    """
    curr_x, curr_y, curr_z = true_start_pos[0], true_start_pos[1], true_start_pos[2]
    curr_yaw = true_start_yaw
    
    trajectory = [[curr_x, curr_y, curr_z]]
    
    for i, code in enumerate(action_codes[:max_steps]):
        if code == 0 and i > 0:
            break
            
        phys = ACTION_PHYSICS.get(code, ACTION_PHYSICS[8])
        dist = phys["dist"]
        yaw_delta = math.radians(phys["yaw_delta_deg"])
        dz = phys["dz"]
        
        # In OpenFly arc turns, yaw shifts while moving
        mid_yaw = curr_yaw + yaw_delta * 0.5
        curr_x += dist * math.cos(mid_yaw)
        curr_y += dist * math.sin(mid_yaw)
        curr_z += dz
        curr_yaw = normalize_angle(curr_yaw + yaw_delta)
        
        trajectory.append([curr_x, curr_y, curr_z])
        
    return trajectory, [curr_x, curr_y, curr_z], curr_yaw

def compute_wilson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Computes 95% Wilson Score confidence interval for success proportions."""
    if n == 0:
        return 0.0, 0.0
    z = 1.96  # 95% confidence
    p_hat = k / n
    denom = 1 + (z**2) / n
    center = (p_hat + (z**2) / (2 * n)) / denom
    margin = (z * math.sqrt((p_hat * (1 - p_hat) / n) + ((z**2) / (4 * (n**2))))) / denom
    return max(0.0, round((center - margin) * 100, 2)), min(100.0, round((center + margin) * 100, 2))

class ValidatedOpenFlyEvaluator:
    def __init__(self, provider: str = "heuristic", api_key: str = "", model_name: str = ""):
        self.provider = provider
        self.api_key = api_key
        self.model_name = model_name
        self.vlm_client = None
        
        if provider not in ("heuristic", "mock", "fallback"):
            try:
                from perception import VLMClient
                self.vlm_client = VLMClient()
            except Exception as e:
                logger.warning(f"Could not initialize VLMClient: {e}. Defaulting to heuristic parser.")

    def plan_action_codes(self, instruction: str) -> List[int]:
        """
        Translates a natural language instruction into the official 10-class OpenFly action vocabulary:
        {9: Forward-Large(7.6m), 8: Forward-Med(3m), 1: Forward-Fine(3m), 3: Turn-Left, 2: Turn-Right, -1: Climb, -2: Descent, 0: Stop}
        """
        # 1. VLM Planning Mode
        if self.vlm_client and self.provider not in ("heuristic", "mock", "fallback"):
            prompt_grounded = (
                f"You are an aerial drone navigator. Translate this flight instruction into an array of discrete action codes:\n"
                f"Action Code Vocabulary:\n"
                f"  9: Move Forward Far / Long Stretch (7.6m)\n"
                f"  8: Move Forward Medium (3m)\n"
                f"  1: Move Forward Short / Alignment (3m)\n"
                f"  3: Turn Left (~30 deg arc)\n"
                f"  2: Turn Right (~30 deg arc)\n"
                f" -1: Climb / Ascend (3m)\n"
                f" -2: Descend (3m)\n"
                f"  0: Stop (Target Reached)\n\n"
                f"Instruction: \"{instruction}\"\n"
                f"Return ONLY valid JSON array of integer codes, e.g. [9, 9, 8, 3, 9, 9, 0]"
            )
            try:
                subgoals = self.vlm_client.plan_mission(prompt_grounded, self.provider, self.api_key, self.model_name)
                if isinstance(subgoals, list) and subgoals:
                    codes = []
                    for item in subgoals:
                        if isinstance(item, int) and item in ACTION_PHYSICS:
                            codes.append(item)
                        elif isinstance(item, dict):
                            # extract code or convert action name
                            c = item.get("code", item.get("action_code", None))
                            if c is not None and int(c) in ACTION_PHYSICS:
                                codes.append(int(c))
                            else:
                                act_name = str(item.get("action", "")).upper()
                                if "FORWARD" in act_name or "STRAIGHT" in act_name:
                                    dist = float(item.get("distance", item.get("value", 6.0)))
                                    codes.extend([9] * max(1, int(round(dist / 7.6))))
                                elif "LEFT" in act_name:
                                    deg = float(item.get("degrees", 30.0))
                                    codes.extend([3] * max(1, int(round(deg / 30.0))))
                                elif "RIGHT" in act_name:
                                    deg = float(item.get("degrees", 30.0))
                                    codes.extend([2] * max(1, int(round(deg / 30.0))))
                                elif "CLIMB" in act_name or "UP" in act_name:
                                    codes.append(-1)
                                elif "DESCENT" in act_name or "DOWN" in act_name:
                                    codes.append(-2)
                    if codes:
                        if codes[-1] != 0:
                            codes.append(0)
                        return codes
            except Exception as e:
                logger.warning(f"VLM action code generation failed: {e}")

        # 2. Calibrated Heuristic Action Parser
        import re
        p = instruction.lower()
        codes: List[int] = []
        tokens = re.split(r'[,;.\n]|\band\b|\bthen\b|\bnext\b|\bfinally\b|\bafter that\b', p)
        
        for clause in tokens:
            clause = clause.strip()
            if not clause:
                continue
            
            # Climb / Descend
            if any(k in clause for k in ("climb", "ascend", "higher", "go up")):
                codes.extend([-1, -1])
            if any(k in clause for k in ("descend", "lower", "go down")):
                codes.extend([-2, -2])
                
            # Turns
            if "turn left" in clause or "veer left" in clause or "left" in clause:
                if "slightly" in clause or "little" in clause:
                    codes.append(3)
                elif "sharp" in clause or "around" in clause:
                    codes.extend([3, 3])
                else:
                    codes.append(3)
            elif "turn right" in clause or "veer right" in clause or "right" in clause:
                if "slightly" in clause or "little" in clause:
                    codes.append(2)
                elif "sharp" in clause or "around" in clause:
                    codes.extend([2, 2])
                else:
                    codes.append(2)
                    
            # Forward movement
            if any(k in clause for k in ("proceed", "continue straight", "move ahead", "move forward", "fly forward", "fly straight", "advance", "straight", "keep going")):
                if "slightly" in clause or "short" in clause:
                    codes.append(8)
                elif "far" in clause or "long" in clause or "distance" in clause:
                    codes.extend([9, 9, 9])
                else:
                    codes.extend([9, 9])
                    
        if not codes:
            codes = [9, 9, 8, 0]
        else:
            if codes[-1] != 0:
                codes.append(0)
                
        return codes

    def evaluate(self, split_name: str = "seen", max_episodes: int = 50) -> Dict[str, Any]:
        data_file = os.path.join(DATA_DIR, f"{split_name}.json")
        if not os.path.exists(data_file):
            from download_dataset import download_annotations
            download_annotations()
            
        with open(data_file, "r", encoding="utf-8") as f:
            dataset = json.load(f)
            
        total_episodes = min(len(dataset), max_episodes)
        logger.info(f"Running Validated OpenFly Benchmark on '{split_name}' split ({total_episodes}/{len(dataset)} episodes)...")
        
        errors: List[float] = []
        successes: List[int] = []
        oracle_successes: List[int] = []
        spl_scores: List[float] = []
        detailed_records = []
        
        for idx in range(total_episodes):
            item = dataset[idx]
            instruction = item["gpt_instruction"]
            gt_positions = item["pos"]
            gt_actions = item["action"]
            gt_yaw = item["yaw"]
            
            # 1. Reconstruct true initial starting position before action 0
            true_start_pos = reconstruct_true_start_pos(gt_positions[0], gt_yaw[0], gt_actions[0])
            true_start_yaw = gt_yaw[0]
            target_pos = gt_positions[-1]
            
            # Ground truth path length
            gt_len = calculate_path_length([true_start_pos] + gt_positions)
            
            # 2. Predict discrete action codes
            pred_codes = self.plan_action_codes(instruction)
            
            # 3. Rollout using validated OpenFly physics
            pred_trajectory, final_pos, final_yaw = rollout_openfly_physics(true_start_pos, true_start_yaw, pred_codes)
            pred_len = calculate_path_length(pred_trajectory)
            
            # 4. Metric Calculations
            # NE (Navigation Error)
            dx = final_pos[0] - target_pos[0]
            dy = final_pos[1] - target_pos[1]
            dz = final_pos[2] - target_pos[2]
            ne = math.sqrt(dx*dx + dy*dy + dz*dz)
            errors.append(ne)
            
            # SR (Success Rate @ 20m)
            is_success = 1 if ne <= 20.0 else 0
            successes.append(is_success)
            
            # OSR (Oracle Success Rate @ 20m)
            min_dist = min(
                math.sqrt((p[0]-target_pos[0])**2 + (p[1]-target_pos[1])**2 + (p[2]-target_pos[2])**2)
                for p in pred_trajectory
            )
            is_oracle = 1 if min_dist <= 20.0 else 0
            oracle_successes.append(is_oracle)
            
            # SPL (Success weighted by Path Length)
            denom = max(pred_len, gt_len, 1e-4)
            spl = (is_success * (gt_len / denom))
            spl_scores.append(spl)
            
            detailed_records.append({
                "episode": idx,
                "instruction": instruction[:120] + "...",
                "gt_len_m": round(gt_len, 2),
                "pred_len_m": round(pred_len, 2),
                "ne_m": round(ne, 2),
                "success": bool(is_success),
                "oracle": bool(is_oracle),
                "spl": round(spl, 4)
            })
            
            if (idx + 1) % 10 == 0 or idx == total_episodes - 1:
                logger.info(f"Progress: [{idx+1}/{total_episodes}] | Current SR: {sum(successes)/(idx+1)*100:.1f}% | Avg NE: {sum(errors)/(idx+1):.1f}m")
                
        # Statistical Aggregation
        avg_ne = sum(errors) / total_episodes if total_episodes > 0 else 0.0
        k_succ = sum(successes)
        sr_pct = (k_succ / total_episodes * 100.0) if total_episodes > 0 else 0.0
        k_ora = sum(oracle_successes)
        osr_pct = (k_ora / total_episodes * 100.0) if total_episodes > 0 else 0.0
        spl_pct = (sum(spl_scores) / total_episodes * 100.0) if total_episodes > 0 else 0.0
        
        ci_sr_low, ci_sr_high = compute_wilson_ci(k_succ, total_episodes)
        ci_osr_low, ci_osr_high = compute_wilson_ci(k_ora, total_episodes)
        
        results = {
            "benchmark": "OpenFly (ICLR 2026) - Validated Physics",
            "split": split_name,
            "episodes_evaluated": total_episodes,
            "provider": self.provider,
            "metrics": {
                "NE (meters) ↓": round(avg_ne, 2),
                "SR (%) ↑": round(sr_pct, 2),
                "SR 95% CI (%)": f"[{ci_sr_low}%, {ci_sr_high}%]",
                "OSR (%) ↑": round(osr_pct, 2),
                "OSR 95% CI (%)": f"[{ci_osr_low}%, {ci_osr_high}%]",
                "SPL (%) ↑": round(spl_pct, 2)
            },
            "detailed_samples": detailed_records[:10]
        }
        
        output_path = os.path.join(DATA_DIR, f"validated_results_{split_name}_{self.provider}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
            
        logger.info("=" * 70)
        logger.info(f"VALIDATED BENCHMARK RESULTS ({split_name.upper()} SPLIT)")
        logger.info(f"Navigation Error (NE ↓) : {avg_ne:.2f} meters")
        logger.info(f"Success Rate (SR ↑)     : {sr_pct:.2f} % (95% CI: [{ci_sr_low}%, {ci_sr_high}%])")
        logger.info(f"Oracle Success (OSR ↑)  : {osr_pct:.2f} % (95% CI: [{ci_osr_low}%, {ci_osr_high}%])")
        logger.info(f"Path Length Wtd (SPL ↑) : {spl_pct:.2f} %")
        logger.info(f"Saved full results to: {output_path}")
        logger.info("=" * 70)
        
        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Validated OpenFly Aerial VLN Benchmark")
    parser.add_argument("--split", type=str, default="seen", choices=["seen", "unseen"])
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--provider", type=str, default="heuristic", help="lmstudio, gemini, heuristic")
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--model", type=str, default="")
    args = parser.parse_args()
    
    evaluator = ValidatedOpenFlyEvaluator(provider=args.provider, api_key=args.api_key, model_name=args.model)
    evaluator.evaluate(split_name=args.split, max_episodes=args.episodes)
