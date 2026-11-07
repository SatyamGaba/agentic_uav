import os
import sys
import math
import time
import json
import base64
import logging
import threading
from enum import Enum, StrEnum, auto
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple

# VLM settings
VLM_COOLDOWN = 4.0         # seconds between queries (general)
SEARCH_VLM_COOLDOWN = 2.0  # seconds between queries during active search
VLM_TIMEOUT = 20           # seconds per API call
MAX_API_RETRIES = 2
API_RETRY_DELAY = 1.0      # seconds

# Camera
IMAGE_SAVE_INTERVAL = 0.1  # seconds
JPEG_QUALITY = 80
IMAGE_PATH = "temp_frame.jpg"
IMAGE_TMP_PATH = "temp_frame_tmp.jpg"
image_file_lock = threading.Lock()

# Search behavior
SEARCH_ALTITUDE = 8.0      # m — fallback for unknown objects
SCAN_SECTOR_SIZE = 60      # degrees per sector (6 sectors = 360° sweep)
SEARCH_MAX_LAPS = 2        # number of 360° laps before aborting search
REPOSITION_DISTANCE = 12.0 # m to reposition forward after a full lap

FALLBACK_SEARCH_PROFILES: Dict[str, Dict[str, float]] = {
    "windmill":     {"altitude": 8.0, "gimbal_pitch": 0.0},
    "tesla":        {"altitude": 6.0, "gimbal_pitch": 0.35},
    "cardboard box":{"altitude": 4.0, "gimbal_pitch": 0.45},
    "manor":        {"altitude": 6.0, "gimbal_pitch": 0.2},
}
DEFAULT_SEARCH_PROFILE: Dict[str, float] = {"altitude": 5.0, "gimbal_pitch": 0.35}

# Webots constants
MAX_SPEED = 10.0
MOVE_SPEED = 2.0
CLIMB_SPEED = 2.0
TURN_SPEED = 1.5
MIN_ALTITUDE = 0.5
MAX_ALTITUDE = 15.0
TARGET_PRECISION = 0.5
YAW_PRECISION = 0.05
GEOFENCE_RADIUS = 30.0
COORD_TURN_THRESHOLD = 0.1

BATTERY_DRAIN_RATE = 0.05
BATTERY_CRITICAL_PCT = 10.0
MAX_HISTORY_LEN = 100
MAX_ACTION_QUEUE_LEN = 10

class FlightState(StrEnum):
    """Canonical flight states."""
    DORMANT = auto()
    TAKEOFF = auto()
    NAVIGATING = auto()
    REASONING = auto()
    LANDING = auto()
    CURVING = auto()


@dataclass
class ActionStep:
    """A single step in an action sequence returned by the VLM."""
    action: str
    value: float | Tuple[float, float]  # scalar for moves/turns, tuple for curves
    gimbal_pitch: float = 0.0

    def is_movement(self) -> bool:
        return self.action.startswith("MOVE_") and self.action != "MOVE_CURVE"

    def is_turn(self) -> bool:
        return self.action.startswith("TURN_")

    def is_curve(self) -> bool:
        return self.action == "MOVE_CURVE"


@dataclass(frozen=True)
class StateTransition:
    """Canonical representation of a state machine step transition."""
    next_state: FlightState
    x: float
    y: float
    yaw: float
