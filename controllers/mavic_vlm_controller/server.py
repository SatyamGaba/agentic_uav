import os
import sys
CONTROLLER_DIR = os.path.dirname(os.path.abspath(__file__))
if CONTROLLER_DIR not in sys.path:
    sys.path.insert(0, CONTROLLER_DIR)

import json
import base64
import logging
import threading
from typing import Optional, List, Dict, Any, Tuple
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from dataclasses import dataclass, field
import time
from config import FlightState, IMAGE_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("MavicVLM")

MAX_LOG_ENTRIES = 50
MAX_MANUAL_COMMANDS = 10
VALID_PROVIDERS = {"gemini", "ollama", "lmstudio", "qwen-dashscope"}
CONTROLLER_DIR = os.path.dirname(os.path.abspath(__file__))

@dataclass
class HistoryEntry:
    """Structured flight history entry — no string parsing needed."""
    step: int
    action: str
    distance: float = 0.0
    yaw_angle: float = 0.0
    heading: float = 0.0
    x: float = 0.0
    y: float = 0.0
    altitude: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_display(self) -> str:
        if self.action == "MOVE_CURVE":
            return (f"Step {self.step}: MOVE_CURVE arc={self.distance:.2f}m, "
                    f"sweep={self.yaw_angle:.0f}° [X={self.x:.1f}, Y={self.y:.1f}, hdg={self.heading:.0f}°]")
        elif self.action in ("TURN_LEFT", "TURN_RIGHT"):
            return (f"Step {self.step}: {self.action} {self.yaw_angle:.0f}° "
                    f"[from hdg={self.heading:.0f}°, X={self.x:.1f}, Y={self.y:.1f}]")
        else:
            return (f"Step {self.step}: {self.action} {self.distance:.1f}m "
                    f"[X={self.x:.1f}, Y={self.y:.1f}, hdg={self.heading:.0f}°]")


@dataclass
class MissionConfig:
    """Thread-safe mission configuration container with allowlist validation."""
    prompt: str = ""
    api_provider: str = "lmstudio"
    api_key: str = "http://127.0.0.1:1234/v1/chat/completions"
    model_name: str = "qwen/qwen3-vl-4b"
    prompt_id: int = 0
    new_request_queued: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    ALLOWED_FIELDS = {"prompt", "api_provider", "api_key", "model_name"}

    def update(self, **kwargs) -> None:
        with self._lock:
            for k, v in kwargs.items():
                if k in self.ALLOWED_FIELDS:
                    setattr(self, k, v)
            if "prompt" in kwargs:
                self.prompt_id += 1
                self.new_request_queued = True

    def read(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "prompt": self.prompt,
                "api_provider": self.api_provider,
                "api_key": self.api_key,
                "model_name": self.model_name,
                "prompt_id": self.prompt_id,
                "new_request_queued": self.new_request_queued
            }

    def get_prompt_update(self, current_prompt_id: int) -> Tuple[bool, str, str, str, str, int]:
        """Atomically read prompt details if prompt_id is newer than current_prompt_id."""
        with self._lock:
            if self.prompt_id > current_prompt_id:
                self.new_request_queued = False
                return True, self.prompt, self.api_provider, self.api_key, self.model_name, self.prompt_id
            return False, "", "", "", "", current_prompt_id


@dataclass
class TelemetryData:
    """Live telemetry exposed via HTTP /telemetry endpoint."""
    x: float = 0.0
    y: float = 0.0
    altitude: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    battery: float = 100.0
    state: str = "DORMANT"
    target: Optional[List[float]] = None
    gimbal_pitch: float = 0.0
    vlm_logs: List[Dict[str, Any]] = field(default_factory=list)
    target_beliefs: List[Dict[str, Any]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def update(self, **kwargs) -> None:
        with self._lock:
            for k, v in kwargs.items():
                if hasattr(self, k) and not k.startswith("_"):
                    setattr(self, k, v)

    def add_vlm_log(self, reasoning: str, action: str, distance_or_angle: float) -> None:
        with self._lock:
            self.vlm_logs.append({
                "time": time.strftime("%H:%M:%S", time.localtime()),
                "reasoning": reasoning,
                "action": action,
                "distance_or_angle": distance_or_angle
            })
            if len(self.vlm_logs) > MAX_LOG_ENTRIES:
                self.vlm_logs.pop(0)

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "x": self.x,
                "y": self.y,
                "altitude": self.altitude,
                "roll": self.roll,
                "pitch": self.pitch,
                "yaw": self.yaw,
                "battery": self.battery,
                "state": self.state,
                "target": self.target,
                "gimbal_pitch": self.gimbal_pitch,
                "vlm_logs": list(self.vlm_logs),
                "target_beliefs": list(self.target_beliefs)
            }


# ---------------------------------------------------------------------------
# Global Shared State (with proper synchronization)
# ---------------------------------------------------------------------------

telemetry = TelemetryData()
mission = MissionConfig()
manual_command_queue: List[str] = []
manual_queue_lock = threading.Lock()
image_file_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------


class DashboardHTTPHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the drone dashboard."""

    def log_message(self, format: str, *args) -> None:
        logger.info(f"[HTTP] {format % args}")

    def _send_json(self, status_code: int, data: Dict[str, Any]) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status_code: int, message: str) -> None:
        self._send_json(status_code, {"status": "error", "message": message})

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            index_path = os.path.join(CONTROLLER_DIR, "index.html")
            if os.path.exists(index_path):
                with open(index_path, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(content)
            else:
                self._send_error(404, "index.html not found")

        elif self.path.startswith("/image"):
            try:
                img_data = None
                with image_file_lock:
                    if os.path.exists(IMAGE_PATH):
                        with open(IMAGE_PATH, "rb") as f:
                            img_data = f.read()
                if not img_data:
                    img_data = base64.b64decode("R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7")
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(img_data)))
                self.send_header("Cache-Control", "no-store, must-revalidate")
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(img_data)
            except Exception as e:
                logger.error(f"Error serving image: {e}")

        elif self.path == "/telemetry":
            self._send_json(200, telemetry.to_dict())
        else:
            self._send_error(404, "Endpoint not found")

    def do_POST(self) -> None:
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode("utf-8")) if post_data else {}
        except (json.JSONDecodeError, ValueError) as e:
            self._send_error(400, f"Invalid JSON: {e}")
            return

        if self.path == "/command":
            self._handle_command(data)
        elif self.path == "/manual":
            self._handle_manual(data)
        elif self.path == "/land":
            self._handle_land()
        else:
            self.send_response(404)
            self.end_headers()

    def _handle_command(self, data: Dict[str, Any]) -> None:
        prompt = data.get("prompt", "").strip()
        api_provider = data.get("api_provider", "gemini").strip().lower()
        api_key = data.get("api_key", "").strip()
        model_name = data.get("model_name", "qwen2-vl").strip()

        # Validation
        if not prompt:
            self._send_error(400, "Prompt cannot be empty")
            return
        if api_provider not in VALID_PROVIDERS:
            self._send_error(400, f"Invalid provider. Valid: {VALID_PROVIDERS}")
            return
        if api_provider in ("gemini", "qwen-dashscope") and not api_key:
            self._send_error(400, f"API key required for {api_provider}")
            return

        mission.update(
            prompt=prompt,
            api_provider=api_provider,
            api_key=api_key,
            model_name=model_name,
            new_request_queued=True
        )

        # Auto-elevate from DORMANT
        with telemetry._lock:
            if telemetry.state == "DORMANT":
                telemetry.update(state="TAKEOFF")

        logger.info(f"[Dashboard] Mission prompt received: '{prompt}' (via {api_provider})")
        self._send_json(200, {"status": "success", "message": "Mission updated"})

    def _handle_manual(self, data: Dict[str, Any]) -> None:
        action = data.get("action", "").strip()
        if not action:
            self._send_error(400, "Action cannot be empty")
            return

        with manual_queue_lock:
            if len(manual_command_queue) >= MAX_MANUAL_COMMANDS:
                manual_command_queue.pop(0)
            manual_command_queue.append(action)

        self._send_json(200, {"status": "success", "message": "Manual command queued"})

    def _handle_land(self) -> None:
        mission.update(prompt="", new_request_queued=False)
        with telemetry._lock:
            telemetry.update(state="LANDING", target=None)
        logger.info("[Dashboard] Emergency Land command received")
        self._send_json(200, {"status": "success", "message": "Landing triggered"})


def run_web_server(port: int = 8080) -> None:
    try:
        server = ThreadingHTTPServer(("0.0.0.0", port), DashboardHTTPHandler)
        logger.info(f"[Web Server] Multi-threaded server started on http://127.0.0.1:{port}")
        server.serve_forever()
    except Exception as e:
        logger.error(f"[Web Server] Failed to start on port {port}: {e}")

if __name__ == "__main__":
    run_web_server(8080)


# ---------------------------------------------------------------------------
# VLM API Client
# ---------------------------------------------------------------------------

