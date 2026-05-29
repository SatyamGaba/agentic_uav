#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

from agentic_uav.experiments import (
    METHODS,
    ExperimentSuiteConfig,
    _write_csv,
    count_record_changes,
    normalized_coverage_auc,
    run_experiment_suite,
)
from agentic_uav.rendering import Renderer
from agentic_uav.scenarios import build_demo_scenario
from agentic_uav.simulation import ScenarioConfig, Simulation


def run_scenario(config: ScenarioConfig) -> dict[str, object]:
    simulation = Simulation.from_config(config)
    return simulation.run()


def run_headless(args: argparse.Namespace) -> dict[str, object]:
    methods = _parse_method_list(args.method)
    grid_sizes = _parse_int_tuple(args.grid_size)
    if not grid_sizes:
        raise ValueError("--grid-size must list at least one grid size")
    if any(size < 1 for size in grid_sizes):
        raise ValueError("--grid-size values must be >= 1")

    combos = [(method, size) for method in methods for size in grid_sizes]

    results: list[dict[str, Any]] = []
    for method, grid_size in combos:
        config = build_demo_scenario(
            method,
            mission_type=args.mission,
            grid_size=grid_size,
            uav_count=args.uav_count,
            seed=args.seed,
            ticks=args.ticks,
        )
        simulation = Simulation.from_config(config)
        summary = simulation.run()
        results.append(
            {
                "method": method,
                "grid_size": grid_size,
                "simulation": simulation,
                "summary": summary,
                "records": simulation.metrics.records,
                "replans": simulation.metrics.replans,
                "plan_changes": simulation.metrics.plan_changes,
            }
        )

    if len(combos) == 1:
        result = results[0]
        simulation = result["simulation"]
        if args.snapshot is not None:
            Renderer().render_snapshot(simulation.world, simulation.uavs, args.snapshot)
        if args.output_dir is not None:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            _write_per_tick(
                args.output_dir,
                result["method"],
                result["grid_size"],
                args.seed,
                result["records"],
            )
        return result["summary"]

    output_dir = args.output_dir if args.output_dir is not None else Path("runs/run")
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, str] = {}
    for result in results:
        per_tick = _write_per_tick(
            output_dir,
            result["method"],
            result["grid_size"],
            args.seed,
            result["records"],
        )
        outputs.update({key: str(path) for key, path in per_tick.items()})

    aggregate_rows = _build_run_aggregates(results, args)
    comparison_csv = output_dir / "comparison.csv"
    comparison_json = output_dir / "comparison.json"
    _write_csv(comparison_csv, aggregate_rows)
    comparison_json.write_text(json.dumps(aggregate_rows, indent=2) + "\n", encoding="utf-8")
    outputs["comparison_csv"] = str(comparison_csv)
    outputs["comparison_json"] = str(comparison_json)

    if not args.no_plots:
        plot_paths = _write_run_plots(output_dir, results, aggregate_rows)
        outputs.update({key: str(path) for key, path in plot_paths.items()})

    return {
        "runs": len(combos),
        "output_dir": str(output_dir),
        "outputs": outputs,
    }


def _parse_method_list(value: str) -> list[str]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if parts == ["all"]:
        return list(METHODS)
    if not parts:
        raise ValueError("--method must name at least one method")
    unknown = [part for part in parts if part not in METHODS]
    if unknown:
        raise ValueError(
            f"unknown method(s): {', '.join(unknown)}; choose from {', '.join(METHODS)} or 'all'"
        )
    return parts


def _write_per_tick(
    output_dir: Path,
    method: str,
    grid_size: int,
    seed: int,
    records: list[dict[str, Any]],
) -> dict[str, Path]:
    stem = f"metrics_{method}_grid{grid_size}_seed{seed}"
    csv_path = output_dir / f"{stem}.csv"
    json_path = output_dir / f"{stem}.json"
    nested_keys = ("uav_cells", "uav_roles", "uav_targets", "urgent_targets")
    csv_rows = [
        {
            "tick": record["tick"],
            "coverage_ratio": record["coverage_ratio"],
            "active_uavs": record["active_uavs"],
            "messages_sent": record["messages_sent"],
            **{key: json.dumps(record.get(key)) for key in nested_keys},
        }
        for record in records
    ]
    _write_csv(csv_path, csv_rows)
    json_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    return {f"{stem}_csv": csv_path, f"{stem}_json": json_path}


def _build_run_aggregates(
    results: list[dict[str, Any]], args: argparse.Namespace
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        summary = result["summary"]
        records = result["records"]
        rows.append(
            {
                "method": result["method"],
                "grid_size": result["grid_size"],
                "uav_count": args.uav_count,
                "ticks": args.ticks,
                "seed": args.seed,
                "mission_type": args.mission,
                "ticks_run": summary["ticks_run"],
                "final_coverage_ratio": summary["coverage_ratio"],
                "is_solved": summary["is_solved"],
                "termination_reason": summary["termination_reason"],
                "messages_sent": summary["messages_sent"],
                "urgent_targets": len(summary["urgent_targets"]),
                "coverage_auc": normalized_coverage_auc(
                    records, args.ticks, summary["coverage_ratio"]
                ),
                "replan_count": len(result["replans"]),
                "plan_change_count": len(result["plan_changes"]),
                "role_switches": count_record_changes(records, "uav_roles"),
                "target_changes": count_record_changes(records, "uav_targets"),
            }
        )
    return rows


def _write_run_plots(
    output_dir: Path,
    results: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
) -> dict[str, Path]:
    try:
        from matplotlib import pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plots", file=sys.stderr)
        return {}

    timeline_path = output_dir / "coverage_vs_time.png"
    figure, axis = plt.subplots(figsize=(8, 5))
    for result in results:
        records = result["records"]
        ticks = [record["tick"] for record in records]
        coverage = [record["coverage_ratio"] for record in records]
        axis.plot(ticks, coverage, label=f"{result['method']} g{result['grid_size']}")
    axis.set_xlabel("Tick")
    axis.set_ylabel("Coverage ratio")
    axis.set_ylim(0, 1.05)
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize="small")
    figure.tight_layout()
    figure.savefig(timeline_path, dpi=180)

    grid_path = output_dir / "coverage_vs_grid_size.png"
    figure2, axis2 = plt.subplots(figsize=(8, 5))
    methods = sorted({row["method"] for row in aggregate_rows})
    for method in methods:
        points = sorted(
            ((row["grid_size"], row["final_coverage_ratio"]) for row in aggregate_rows if row["method"] == method),
            key=lambda item: item[0],
        )
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        axis2.plot(xs, ys, marker="o", label=method)
    axis2.set_xlabel("Grid size")
    axis2.set_ylabel("Final coverage ratio")
    axis2.set_ylim(0, 1.05)
    axis2.grid(True, alpha=0.25)
    axis2.legend(fontsize="small")
    figure2.tight_layout()
    figure2.savefig(grid_path, dpi=180)

    plt.close("all")
    return {"coverage_vs_time": timeline_path, "coverage_vs_grid_size": grid_path}


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
        default="agentic",
        help="Comma-separated method(s) or 'all'. One method -> single run; "
        "multiple methods sweep the cartesian product with --grid-size.",
    )
    run_parser.add_argument(
        "--grid-size",
        default="25",
        help="Comma-separated grid size(s); the world is square (size x size).",
    )
    run_parser.add_argument("--uav-count", type=int, default=4)
    run_parser.add_argument("--ticks", type=int, default=500)
    run_parser.add_argument("--seed", type=int, default=7)
    run_parser.add_argument(
        "--mission",
        choices=["survey", "disaster_mapping"],
        default="disaster_mapping",
    )
    run_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for per-tick metrics, comparison table, and plots. "
        "Defaults to runs/run when sweeping multiple combos.",
    )
    run_parser.add_argument("--no-plots", action="store_true", default=False)
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
