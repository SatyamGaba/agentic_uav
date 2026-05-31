import json
import tempfile
import unittest
from pathlib import Path

from agentic_uav.scenarios import ScenarioParams
from agentic_uav.ui_config import (
    load_ui_params,
    params_from_dict,
    params_to_dict,
    save_ui_params,
)


class UiConfigTest(unittest.TestCase):
    def test_loads_default_config_when_last_config_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            default_path = tmp_path / "default_config.json"
            last_path = tmp_path / "last_config.json"
            default_params = ScenarioParams(method_name="rules", grid_size=10, seed=42)
            default_path.write_text(json.dumps(params_to_dict(default_params)), encoding="utf-8")

            params = load_ui_params(default_path, last_path)

            self.assertEqual(params.method_name, "rules")
            self.assertEqual(params.grid_size, 10)
            self.assertEqual(params.seed, 42)

    def test_last_config_overrides_defaults_and_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            default_path = tmp_path / "default_config.json"
            last_path = tmp_path / ".agentic_uav" / "last_config.json"
            default_path.write_text(json.dumps(params_to_dict(ScenarioParams())), encoding="utf-8")
            last_params = ScenarioParams(
                method_name="task_consideration",
                mission_type="survey",
                grid_size=6,
                uav_count=2,
                seed=99,
            )

            save_ui_params(last_params, last_path)
            params = load_ui_params(default_path, last_path)

            self.assertEqual(params, last_params)

    def test_malformed_last_config_falls_back_to_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            default_path = tmp_path / "default_config.json"
            last_path = tmp_path / "last_config.json"
            default_params = ScenarioParams(method_name="rules", grid_size=8)
            default_path.write_text(json.dumps(params_to_dict(default_params)), encoding="utf-8")
            last_path.write_text("{not json", encoding="utf-8")

            params = load_ui_params(default_path, last_path)

            self.assertEqual(params, default_params)

    def test_partial_config_fills_defaults_and_clamps_values(self) -> None:
        params = params_from_dict(
            {
                "method_name": "unknown",
                "mission_type": "also_unknown",
                "grid_size": "99",
                "uav_count": 0,
                "sensing_radius": "bad",
                "communication_range": 50,
                "seed": "12",
                "ticks": 20_000,
                "heartbeat_interval": 99,
                "urgent_message_ttl": True,
            },
            ScenarioParams(method_name="rules", mission_type="survey", sensing_radius=2),
        )

        self.assertEqual(params.method_name, "rules")
        self.assertEqual(params.mission_type, "survey")
        self.assertEqual(params.grid_size, 64)
        self.assertEqual(params.uav_count, 1)
        self.assertEqual(params.sensing_radius, 2)
        self.assertEqual(params.communication_range, 8)
        self.assertEqual(params.seed, 12)
        self.assertEqual(params.ticks, 10_000)
        self.assertEqual(params.heartbeat_interval, 12)
        self.assertEqual(params.urgent_message_ttl, 2)


if __name__ == "__main__":
    unittest.main()
