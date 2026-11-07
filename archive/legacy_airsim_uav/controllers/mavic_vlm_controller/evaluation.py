import os
import json
import time
import threading
import logging
from typing import List, Dict, Any

logger = logging.getLogger("MavicVLM.Evaluation")

class MetricsTracker:
    def __init__(self, log_file: str = "metrics_log.json"):
        # Make the log file relative to this file's directory
        self.log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), log_file)
        self.lock = threading.Lock()
        
        # Single-UAV Agentic Metrics
        self.metrics = {
            "total_missions": 0,
            "successful_missions": 0,
            
            "total_subtasks": 0,
            "successful_subtasks": 0,
            
            "total_vlm_calls": 0,
            "tool_calling_failures": 0,
            
            "decision_latencies_sec": [],
            
            "replanning_count": 0,
            "hallucination_count": 0
        }
        self.load()

    def load(self):
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for k in self.metrics.keys():
                        if k in data:
                            self.metrics[k] = data[k]
            except Exception as e:
                logger.warning(f"Failed to load metrics from {self.log_file}: {e}")

    def save(self):
        with self.lock:
            try:
                with open(self.log_file, "w", encoding="utf-8") as f:
                    json.dump(self.metrics, f, indent=4)
            except Exception as e:
                logger.warning(f"Failed to save metrics to {self.log_file}: {e}")

    def get_summary(self) -> Dict[str, float]:
        with self.lock:
            m = self.metrics
            msr = (m["successful_missions"] / m["total_missions"]) * 100 if m["total_missions"] > 0 else 0.0
            scr = (m["successful_subtasks"] / m["total_subtasks"]) * 100 if m["total_subtasks"] > 0 else 0.0
            tca = ((m["total_vlm_calls"] - m["tool_calling_failures"]) / m["total_vlm_calls"]) * 100 if m["total_vlm_calls"] > 0 else 0.0
            avg_latency = sum(m["decision_latencies_sec"]) / len(m["decision_latencies_sec"]) if m["decision_latencies_sec"] else 0.0
            
            return {
                "Mission Success Rate (MSR) %": round(msr, 2),
                "Sub-Task Completion Rate (SCR) %": round(scr, 2),
                "Tool Calling Accuracy (TCA) %": round(tca, 2),
                "Avg Decision Latency (s)": round(avg_latency, 2),
                "Total Replannings": m["replanning_count"],
                "Total Hallucinations": m["hallucination_count"]
            }

    def record_mission(self, success: bool):
        with self.lock:
            self.metrics["total_missions"] += 1
            if success:
                self.metrics["successful_missions"] += 1
        self.save()

    def record_subtask(self, success: bool):
        with self.lock:
            self.metrics["total_subtasks"] += 1
            if success:
                self.metrics["successful_subtasks"] += 1
        self.save()

    def record_vlm_call(self, latency: float, success: bool = True, hallucination: bool = False):
        with self.lock:
            self.metrics["total_vlm_calls"] += 1
            self.metrics["decision_latencies_sec"].append(latency)
            if not success:
                self.metrics["tool_calling_failures"] += 1
            if hallucination:
                self.metrics["hallucination_count"] += 1
        self.save()

    def record_replanning(self):
        with self.lock:
            self.metrics["replanning_count"] += 1
        self.save()

# Global instance for easy import
global_metrics = MetricsTracker()
