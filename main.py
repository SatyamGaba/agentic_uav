#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Callable, Sequence

from agentic_uav.experiments import ExperimentSuiteConfig, run_experiment_suite
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


def run_experiments(args: argparse.Namespace) -> dict[str, object]:
    result = run_experiment_suite(
        ExperimentSuiteConfig(
            output_dir=args.output_dir,
            seed_count=args.seed_count,
            base_seed=args.base_seed,
            swarm_sizes=_parse_int_tuple(args.swarm_sizes),
            dropout_fractions=_parse_float_tuple(args.dropout_fractions),
            ticks=args.ticks,
            grid_size=args.grid_size,
            success_threshold=args.success_threshold,
            family=args.family,
            make_plots=not args.no_plots,
        )
    )
    return {
        "trials": len(result.trial_rows),
        "aggregates": len(result.aggregate_rows),
        "output_dir": str(args.output_dir),
        "outputs": {key: str(path) for key, path in result.output_paths.items()},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the swappable UAV swarm simulator.")
    subparsers = parser.add_subparsers(dest="command")

    ui_parser = subparsers.add_parser("ui", help="Launch the Solara browser UI.")
    ui_parser.add_argument("--host", default="127.0.0.1")
    ui_parser.add_argument("--port", type=int, default=8765)
    ui_parser.add_argument("--open", dest="open_browser", action="store_true", default=False)
    ui_parser.add_argument("--no-open", dest="open_browser", action="store_false")

    run_parser = subparsers.add_parser("run", help="Run a headless simulation.")
    run_parser.add_argument(
        "--method",
        choices=["static", "rules", "task_consideration", "agentic"],
        default="agentic",
    )
    run_parser.add_argument("--snapshot", type=Path, default=None)

    experiment_parser = subparsers.add_parser(
        "experiment",
        help="Run paired baseline-vs-agentic experiment sweeps.",
    )
    experiment_parser.add_argument("--output-dir", type=Path, default=Path("runs/experiments"))
    experiment_parser.add_argument("--seed-count", type=int, default=20)
    experiment_parser.add_argument("--base-seed", type=int, default=0)
    experiment_parser.add_argument("--swarm-sizes", default="5,10,15")
    experiment_parser.add_argument("--dropout-fractions", default="0,0.1,0.25,0.4")
    experiment_parser.add_argument("--ticks", type=int, default=500)
    experiment_parser.add_argument("--grid-size", type=int, default=25)
    experiment_parser.add_argument("--success-threshold", type=float, default=0.9)
    experiment_parser.add_argument(
        "--family",
        choices=["all", "survey_dropout", "disaster_urgent_dropout"],
        default="all",
    )
    experiment_parser.add_argument("--no-plots", action="store_true", default=False)
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


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _parse_float_tuple(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def main(
    argv: Sequence[str] | None = None,
    run_command: Callable[[list[str]], object] = subprocess.run,
) -> int:
    args = parse_args(argv)

    if args.command == "run":
        print(run_headless(args))
        return 0
    if args.command == "experiment":
        print(run_experiments(args))
        return 0

    try:
        result = run_command(build_ui_command(args))
    except KeyboardInterrupt:
        return 130
    return int(getattr(result, "returncode", 0) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
