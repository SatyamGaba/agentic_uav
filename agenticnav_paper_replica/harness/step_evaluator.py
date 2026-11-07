"""
Step-by-Step Evaluation and Tracking Harness for AgenticNav / UAV Navigation.
Evaluates reasoning quality, visual/depth perception, movement consistency,
and parsing validity for each step before advancing to the next step.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import math
import json
import re
import numpy as np
from PIL import Image

from agenticnav_paper_replica.geometry.pinhole import VIEW_LABELS


@dataclass
class StepEvaluation:
    """Detailed evaluation telemetry for a single decision step."""
    step_idx: int
    node_idx: int
    instruction: str

    # Poses & Metrics
    pre_pose: Dict[str, float]
    post_pose: Dict[str, float]
    pre_goal_dist_m: float
    post_goal_dist_m: float
    goal_progress_m: float  # pre - post: positive means got closer
    actual_moved_m: float
    collided: bool
    stalled: bool

    # Perception & Visual Feedback
    view_clearances_m: Dict[str, float]
    bev_stats: Dict[str, Any]

    # Reasoning Evaluation
    raw_thought: str
    detected_milestones: List[str]
    active_milestone: str
    reasoning_quality: str  # "high", "acceptable", "lacks_milestones", "hallucination_warning"

    # Action & Parsing
    action_type: str
    action_payload: Dict[str, Any]
    parsing_valid: bool
    parsing_format: str

    # Movement Consistency
    movement_consistent: bool
    premature_stop_detected: bool
    warnings: List[str] = field(default_factory=list)


class StepEvaluator:
    """
    Evaluates each step of the UAV before moving to the next step.
    Supports interactive pilot mode (step-by-step terminal telemetry & pause)
    and autonomous mode (machine-readable audit logging).
    """

    def __init__(
        self,
        mode: str = "verbose",  # "verbose", "summary", "autonomous"
        out_dir: Optional[Path] = None,
        save_intermediate_bev: bool = True,
        interactive_pause: bool = False,
    ):
        self.mode = mode
        self.out_dir = Path(out_dir) if out_dir else None
        self.save_intermediate_bev = save_intermediate_bev
        self.interactive_pause = interactive_pause
        self.evaluations: List[StepEvaluation] = []

        if self.out_dir:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            if self.save_intermediate_bev:
                (self.out_dir / "step_bevs").mkdir(parents=True, exist_ok=True)

    def evaluate_step(
        self,
        step_idx: int,
        node_idx: int,
        instruction: str,
        pre_pose: Tuple[float, float, float],
        post_pose: Tuple[float, float, float],
        pre_goal_dist_m: float,
        post_goal_dist_m: float,
        accumulated_dist_m: float,
        depth_views: Dict[int, np.ndarray],
        bev_image: Optional[np.ndarray],
        occupancy_grid: Any,
        turn_log: List[Dict[str, Any]],
        action_type: str,
        action_payload: Dict[str, Any],
        actual_moved_m: float,
        collided: bool,
    ) -> StepEvaluation:
        """
        Runs full diagnostic inspection on the step before advancing to the next step.
        """
        warnings: List[str] = []

        # 1. Visual & Depth Feedback Analysis
        view_clearances: Dict[str, float] = {}
        for v_idx in (0, 1, 2, 3):
            label = VIEW_LABELS.get(v_idx, f"View_{v_idx}")
            d_img = depth_views.get(v_idx)
            if d_img is not None and d_img.size > 0:
                # Metric clearance measured in central horizontal corridor band
                h, w = d_img.shape[:2]
                band = d_img[int(h * 0.4):int(h * 0.6), int(w * 0.3):int(w * 0.7)]
                valid = band[band > 0.05]
                clearance = float(np.min(valid)) if len(valid) > 0 else 0.0
                view_clearances[label] = round(clearance, 2)
            else:
                view_clearances[label] = 0.0

        # BEV occupancy stats
        bev_stats: Dict[str, Any] = {}
        if occupancy_grid is not None and hasattr(occupancy_grid, "grid"):
            g = occupancy_grid.grid
            free_cells = int(np.sum(g == 255))
            obstacle_cells = int(np.sum(g == 0))
            unexplored_cells = int(np.sum(g == 127))
            cell_area = (occupancy_grid.resolution_m) ** 2
            bev_stats = {
                "explored_free_m2": round(free_cells * cell_area, 2),
                "obstacle_m2": round(obstacle_cells * cell_area, 2),
                "unexplored_m2": round(unexplored_cells * cell_area, 2),
                "total_obstacle_pixels": obstacle_cells,
            }

        # 2. Reasoning & Milestone Analysis
        raw_thought = ""
        parsing_format = "native_tool_call"
        for turn in turn_log:
            if turn.get("role") == "assistant" or "thought" in turn:
                raw_thought = turn.get("thought") or turn.get("content", "")
                if "```json" in raw_thought:
                    parsing_format = "markdown_json_fence"
                elif "<tool_call>" in raw_thought:
                    parsing_format = "xml_tool_call"
                elif "{" in raw_thought:
                    parsing_format = "balanced_json"

        # Detect sequential milestones
        detected_milestones = []
        milestone_matches = re.findall(r"(?:Milestone\s*\d+|Step\s*\d+[:\-\.]|[0-9]+\.\s+[A-Z][a-z]+)", raw_thought)
        if milestone_matches:
            detected_milestones = [m.strip() for m in milestone_matches[:6]]

        # Active milestone check
        active_milestone = "Not specified"
        act_match = re.search(r"(?:In Progress|Active Milestone|Current Milestone)[:\s*]+([^\n\r]+)", raw_thought, re.IGNORECASE)
        if act_match:
            active_milestone = act_match.group(1).strip()
        elif detected_milestones:
            active_milestone = detected_milestones[0]

        # Reasoning quality classification
        if detected_milestones and len(raw_thought) > 40:
            reasoning_quality = "high"
        elif len(raw_thought) > 30:
            reasoning_quality = "acceptable"
        else:
            reasoning_quality = "lacks_milestones"
            warnings.append("Model provided minimal or no milestone decomposition in thought trace.")

        # 3. Action Parsing Validation
        parsing_valid = True
        if action_type == "move_to":
            v = action_payload.get("view")
            u = action_payload.get("u")
            v_coord = action_payload.get("v")
            if v not in (0, 1, 2, 3):
                parsing_valid = False
                warnings.append(f"Invalid view index: {v}")
            if u is None or not (0.0 <= u <= 1.0):
                parsing_valid = False
                warnings.append(f"Invalid u coordinate: {u}")
            if v_coord is None or not (0.0 <= v_coord <= 1.0):
                parsing_valid = False
                warnings.append(f"Invalid v coordinate: {v_coord}")
        elif action_type == "stop":
            parsing_valid = True
        elif action_type in ("deadlock", "protocol_violation"):
            parsing_valid = False
            warnings.append(f"Action terminated with failure status: {action_type}")

        # 4. Movement Consistency & Physics Validation
        goal_progress_m = round(pre_goal_dist_m - post_goal_dist_m, 2)
        stalled = False
        if action_type == "move_to" and actual_moved_m < 0.10:
            stalled = True
            warnings.append(f"Agent stalled: commanded move_to but only moved {actual_moved_m:.2f}m.")

        if collided:
            warnings.append("Embodiment reported collision with physical obstacle during execution.")

        # Premature stop check
        premature_stop_detected = False
        if action_type == "stop":
            if post_goal_dist_m > 3.0:
                premature_stop_detected = True
                warnings.append(
                    f"PREMATURE STOP DETECTED: Agent invoked stop() {post_goal_dist_m:.2f}m from goal (> 3.0m threshold)."
                )
            if accumulated_dist_m < 4.0:
                warnings.append(
                    f"SUSPICIOUS EARLY STOP: Agent called stop after only {accumulated_dist_m:.2f}m total traversal."
                )

        # Overall movement consistency
        movement_consistent = parsing_valid and not stalled and (not premature_stop_detected)
        if goal_progress_m < -1.0:
            warnings.append(f"Negative progress: Agent moved {abs(goal_progress_m):.2f}m away from the goal.")

        eval_res = StepEvaluation(
            step_idx=step_idx,
            node_idx=node_idx,
            instruction=instruction,
            pre_pose={"x": round(pre_pose[0], 2), "z": round(pre_pose[1], 2), "yaw_deg": round(math.degrees(pre_pose[2]), 1)},
            post_pose={"x": round(post_pose[0], 2), "z": round(post_pose[1], 2), "yaw_deg": round(math.degrees(post_pose[2]), 1)},
            pre_goal_dist_m=round(pre_goal_dist_m, 2),
            post_goal_dist_m=round(post_goal_dist_m, 2),
            goal_progress_m=goal_progress_m,
            actual_moved_m=round(actual_moved_m, 2),
            collided=collided,
            stalled=stalled,
            view_clearances_m=view_clearances,
            bev_stats=bev_stats,
            raw_thought=raw_thought.strip(),
            detected_milestones=detected_milestones,
            active_milestone=active_milestone,
            reasoning_quality=reasoning_quality,
            action_type=action_type,
            action_payload=action_payload,
            parsing_valid=parsing_valid,
            parsing_format=parsing_format,
            movement_consistent=movement_consistent,
            premature_stop_detected=premature_stop_detected,
            warnings=warnings,
        )

        self.evaluations.append(eval_res)

        # 5. Output Reporting & Visual Feedback
        if self.mode in ("verbose", "interactive"):
            self._print_step_dashboard(eval_res, accumulated_dist_m)

        # Save intermediate BEV map
        if self.save_intermediate_bev and self.out_dir and bev_image is not None:
            bev_file = self.out_dir / "step_bevs" / f"step_{step_idx:02d}_node_{node_idx:02d}.png"
            Image.fromarray(bev_image).save(bev_file)

        # Interactive pause if enabled
        if self.interactive_pause:
            input("\n[PILOT PAUSE] Press Enter to evaluate and advance to next step (or Ctrl+C to abort)...")

        return eval_res

    def _print_step_dashboard(self, ev: StepEvaluation, accumulated_dist_m: float):
        """Renders rich terminal telemetry for the step."""
        banner = "=" * 80
        print("\n" + banner)
        print(f" [STEP EVALUATION] Step {ev.step_idx:02d} @ Node {ev.node_idx:02d} | Accumulated Odometry: {accumulated_dist_m:.2f}m")
        print(banner)

        # Visual clearances
        c = ev.view_clearances_m
        print(f" 1. VISUAL & DEPTH PERCEPTION:")
        print(f"    - Front: {c.get('Front', 0):.2f}m | Right: {c.get('Right', 0):.2f}m | Back: {c.get('Back', 0):.2f}m | Left: {c.get('Left', 0):.2f}m")
        if ev.bev_stats:
            print(f"    - BEV Explored Area: {ev.bev_stats.get('explored_free_m2', 0)} m² | Obstacles: {ev.bev_stats.get('total_obstacle_pixels', 0)} px")

        # Reasoning
        print(f" 2. VLM REASONING ANALYSIS:")
        print(f"    - Reasoning Quality: [{ev.reasoning_quality.upper()}] | Format: {ev.parsing_format}")
        if ev.detected_milestones:
            print(f"    - Detected Milestones: {', '.join(ev.detected_milestones)}")
        print(f"    - Active Milestone   : {ev.active_milestone}")
        if ev.raw_thought:
            first_line = ev.raw_thought.split("\n")[0][:90]
            print(f"    - Model Thought      : \"{first_line}...\"")

        # Action & Parsing
        print(f" 3. ACTION & PARSING:")
        print(f"    - Action Type  : {ev.action_type.upper()}")
        print(f"    - Parsing Valid: {ev.parsing_valid}")
        print(f"    - Payload      : {json.dumps(ev.action_payload)}")

        # Movement Consistency
        print(f" 4. MOVEMENT CONSISTENCY & PROGRESS:")
        p0 = ev.pre_pose
        p1 = ev.post_pose
        print(f"    - Pose Shift   : ({p0['x']:+5.2f}, {p0['z']:+5.2f}, {p0['yaw_deg']:+5.1f}°) -> ({p1['x']:+5.2f}, {p1['z']:+5.2f}, {p1['yaw_deg']:+5.1f}°)")
        print(f"    - Distance Moved: {ev.actual_moved_m:.2f}m | Collided: {ev.collided} | Stalled: {ev.stalled}")
        prog_symbol = "CLOSER (✓)" if ev.goal_progress_m > 0 else ("NO CHANGE (=)" if ev.goal_progress_m == 0 else "FARTHER (⚠️)")
        print(f"    - Goal Distance: {ev.pre_goal_dist_m:.2f}m -> {ev.post_goal_dist_m:.2f}m (Δ: {ev.goal_progress_m:+.2f}m {prog_symbol})")

        # Warnings / Evaluation Status
        if ev.warnings:
            print(f" ⚠️  DIAGNOSTIC WARNINGS ({len(ev.warnings)}):")
            for w in ev.warnings:
                print(f"    - [WARNING] {w}")
        else:
            print(f" ✓  STEP CONSISTENCY: ALL CHECKS PASSED")
        print("-" * 80)

    def save_episode_audit(self, ep_id: str, out_file: Optional[Path] = None):
        """Saves machine-readable step evaluation history to JSON."""
        target = out_file or (self.out_dir / f"audit_eval_{ep_id}.json" if self.out_dir else None)
        if not target:
            return

        data = {
            "episode_id": ep_id,
            "total_steps_evaluated": len(self.evaluations),
            "summary": {
                "all_consistent": all(e.movement_consistent for e in self.evaluations),
                "total_collisions": sum(1 for e in self.evaluations if e.collided),
                "total_stalls": sum(1 for e in self.evaluations if e.stalled),
                "premature_stop_detected": any(e.premature_stop_detected for e in self.evaluations),
                "high_quality_reasoning_ratio": round(
                    sum(1 for e in self.evaluations if e.reasoning_quality == "high") / max(1, len(self.evaluations)),
                    2,
                ),
            },
            "steps": [asdict(e) for e in self.evaluations],
        }

        with open(target, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"[EVALUATOR] Step audit trace saved to: {target}")

