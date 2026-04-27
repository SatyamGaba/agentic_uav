from __future__ import annotations

import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

from agentic_uav.scenarios import MISSION_TYPES, ScenarioParams


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default_config.json"
LAST_CONFIG_PATH = PROJECT_ROOT / ".agentic_uav" / "last_config.json"
METHODS = ("static", "rules", "task_consideration", "agentic")

_PARAM_FIELDS = {field.name for field in fields(ScenarioParams)}
_INT_RANGES = {
    "grid_size": (4, 64),
    "uav_count": (1, 12),
    "sensing_radius": (0, 4),
    "communication_range": (1, 8),
    "ticks": (1, 10_000),
    "heartbeat_interval": (1, 12),
    "urgent_message_ttl": (1, 8),
}


def load_ui_params(
    default_path: Path | str = DEFAULT_CONFIG_PATH,
    last_path: Path | str = LAST_CONFIG_PATH,
) -> ScenarioParams:
    default_params = load_params_file(default_path, ScenarioParams())
    return load_params_file(last_path, default_params)


def load_params_file(path: Path | str, fallback: ScenarioParams) -> ScenarioParams:
    config_path = Path(path)
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fallback
    if not isinstance(data, dict):
        return fallback
    return params_from_dict(data, fallback)


def save_ui_params(params: ScenarioParams, path: Path | str = LAST_CONFIG_PATH) -> None:
    config_path = Path(path)
    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(params_to_dict(params), indent=2) + "\n", encoding="utf-8")
    except OSError:
        return


def params_to_dict(params: ScenarioParams) -> dict[str, Any]:
    return {key: value for key, value in asdict(params).items() if key in _PARAM_FIELDS}


def params_from_dict(data: dict[str, Any], fallback: ScenarioParams | None = None) -> ScenarioParams:
    base = fallback or ScenarioParams()
    values = params_to_dict(base)

    method_name = data.get("method_name")
    if method_name in METHODS:
        values["method_name"] = method_name

    mission_type = data.get("mission_type")
    if mission_type in MISSION_TYPES:
        values["mission_type"] = mission_type

    seed = _coerce_int(data.get("seed"), values["seed"])
    values["seed"] = seed

    for key, bounds in _INT_RANGES.items():
        values[key] = _clamp(_coerce_int(data.get(key), values[key]), bounds)

    return ScenarioParams(**values)


def _coerce_int(value: Any, fallback: int) -> int:
    if isinstance(value, bool):
        return fallback
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _clamp(value: int, bounds: tuple[int, int]) -> int:
    lower, upper = bounds
    return min(max(value, lower), upper)
