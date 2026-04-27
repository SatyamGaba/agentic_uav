import unittest
from pathlib import Path

import main
from main import build_demo_scenario, run_scenario


class CliTest(unittest.TestCase):
    def test_no_args_defaults_to_ui_command(self) -> None:
        args = main.parse_args([])

        self.assertEqual(args.command, "ui")
        self.assertEqual(args.host, "127.0.0.1")
        self.assertEqual(args.port, 8765)
        self.assertFalse(args.open_browser)

    def test_ui_command_builds_solara_launcher(self) -> None:
        args = main.parse_args(["ui", "--port", "8766", "--open"])

        command = main.build_ui_command(args)

        self.assertIn("solara", command)
        self.assertIn("agentic_uav/gui.py", command)
        self.assertIn("--port", command)
        self.assertIn("8766", command)
        self.assertNotIn("--no-open", command)

    def test_main_uses_injected_runner_for_ui(self) -> None:
        calls = []

        exit_code = main.main(["ui", "--port", "8767"], run_command=calls.append)

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(calls), 1)
        self.assertIn("--no-open", calls[0])

    def test_main_handles_keyboard_interrupt_from_ui(self) -> None:
        def interrupted_runner(command: list[str]) -> None:
            raise KeyboardInterrupt

        self.assertEqual(main.main(["ui"], run_command=interrupted_runner), 130)

    def test_demo_scenario_runs_for_each_method(self) -> None:
        for method_name in ("static", "rules", "agentic"):
            with self.subTest(method_name=method_name):
                summary = run_scenario(build_demo_scenario(method_name))
                self.assertLessEqual(summary["ticks_run"], 50)
                self.assertIn("coverage_ratio", summary)
                self.assertIn("is_solved", summary)
                self.assertIn("termination_reason", summary)

    def test_run_command_executes_headless_scenario(self) -> None:
        summary = main.run_headless(main.parse_args(["run", "--method", "agentic"]))

        self.assertLessEqual(summary["ticks_run"], 50)
        self.assertIn("coverage_ratio", summary)
        self.assertIn("is_solved", summary)
        self.assertIn("termination_reason", summary)

    def test_readme_documents_uv_run_main_as_default_ui_launch(self) -> None:
        readme = Path("README.md").read_text()

        self.assertIn("uv run main.py", readme)


if __name__ == "__main__":
    unittest.main()
