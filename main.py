#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Callable, Sequence

from agentic_uav.rendering import Renderer
from agentic_uav.scenarios import build_demo_scenario
from agentic_uav.simulation import ScenarioConfig, Simulation


def run_scenario(config: ScenarioConfig) -> dict[str, object]:
    simulation = Simulation.from_config(config)
    return simulation.run()


def run_headless(args: argparse.Namespace) -> dict[str, object]:
    config = build_demo_scenario(args.method)
    simulation = Simulation.from_config(config)
    summary = simulation.run()

    if args.snapshot is not None:
        Renderer().render_snapshot(simulation.world, simulation.uavs, args.snapshot)

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the swappable UAV swarm simulator.")
    subparsers = parser.add_subparsers(dest="command")

    ui_parser = subparsers.add_parser("ui", help="Launch the Solara browser UI.")
    ui_parser.add_argument("--host", default="127.0.0.1")
    ui_parser.add_argument("--port", type=int, default=8765)
    ui_parser.add_argument("--open", dest="open_browser", action="store_true", default=False)
    ui_parser.add_argument("--no-open", dest="open_browser", action="store_false")

    run_parser = subparsers.add_parser("run", help="Run a headless simulation.")
    run_parser.add_argument("--method", choices=["static", "rules", "agentic"], default="agentic")
    run_parser.add_argument("--snapshot", type=Path, default=None)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        args = ["ui"]
    elif args[0].startswith("-"):
        args = ["run", *args]
    return build_parser().parse_args(args)


def build_ui_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "solara",
        "run",
        "agentic_uav/gui.py",
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    if not args.open_browser:
        command.append("--no-open")
    return command


def main(
    argv: Sequence[str] | None = None,
    run_command: Callable[[list[str]], object] = subprocess.run,
) -> int:
    args = parse_args(argv)

    if args.command == "run":
        print(run_headless(args))
        return 0

    try:
        result = run_command(build_ui_command(args))
    except KeyboardInterrupt:
        return 130
    return int(getattr(result, "returncode", 0) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
