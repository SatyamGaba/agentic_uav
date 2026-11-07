"""Episode trace logging and SFT dataset export for AgenticNav."""

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class TurnTrace:
    turn_idx: int
    prompt_text: str
    tool_calls: List[Dict[str, Any]]
    tool_results: List[Dict[str, Any]]


@dataclass
class StepTrace:
    step_idx: int
    node_idx: int
    agent_pose: Dict[str, float]  # {"x": ..., "z": ..., "yaw_rad": ...}
    geodesic_distance_to_goal_m: float
    turns: List[Dict[str, Any]]
    action_type: str
    action_payload: Dict[str, Any]


@dataclass
class EpisodeTrace:
    episode_id: str
    scene_id: str
    instruction: str
    shortest_path_distance_m: float
    steps: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Optional[Dict[str, Any]] = None
    failure_category: Optional[str] = None


class TraceLogger:
    """Logs full evaluation traces into JSONL files and provides SFT export."""

    def __init__(self, log_dir: Union[str, Path]):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.log_dir / "evaluation_traces.jsonl"

    def log_episode(self, episode_trace: EpisodeTrace) -> None:
        """Appends episode trace to the JSONL log file."""
        data = asdict(episode_trace)
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, default=str) + "\n")

    def export_to_sft_format(self, output_file: Union[str, Path], only_successful: bool = True) -> int:
        """Exports logged episodes into standard multi-turn ShareGPT format for VLM fine-tuning."""
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.jsonl_path.exists():
            return 0

        exported_count = 0
        with open(self.jsonl_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
            for line in fin:
                if not line.strip():
                    continue
                ep = json.loads(line)
                if only_successful and not ep.get("metrics", {}).get("success", False):
                    continue

                # Build multi-turn conversations
                conversations = []
                # System prompt
                conversations.append({
                    "from": "system",
                    "value": f"Goal: {ep.get('instruction', '')}"
                })

                for step in ep.get("steps", []):
                    for turn in step.get("turns", []):
                        resp = turn.get("response", {})
                        if "tool_calls" in resp:
                            conversations.append({
                                "from": "gpt",
                                "value": json.dumps(resp["tool_calls"])
                            })
                        elif "content" in resp and resp["content"]:
                            conversations.append({
                                "from": "gpt",
                                "value": resp["content"]
                            })

                fout.write(json.dumps({"id": ep["episode_id"], "conversations": conversations}) + "\n")
                exported_count += 1

        return exported_count

