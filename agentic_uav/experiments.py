from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from random import Random
from statistics import mean, stdev
from typing import Any, Iterable

from agentic_uav.models import CommunicationEvent, ScenarioConfig
from agentic_uav.scenarios import build_demo_scenario
from agentic_uav.simulation import Simulation


METHODS = ("static", "rules", "task_consideration", "agentic")
SURVEY_DROPOUT = "survey_dropout"
DISASTER_URGENT_DROPOUT = "disaster_urgent_dropout"


@dataclass(frozen=True)
class ExperimentSuiteConfig:
    output_dir: Path
    seed_count: int = 20
    base_seed: int = 0
    swarm_sizes: tuple[int, ...] = (5, 10, 15)
    dropout_fractions: tuple[float, ...] = (0.0, 0.1, 0.25, 0.4)
    ticks: int = 500
    grid_size: int = 25
    sensing_radius: int = 1
    communication_range: int = 3
    success_threshold: float = 0.9
    family: str = "all"
    make_plots: bool = True


@dataclass(frozen=True)
class ExperimentSuiteResult:
    trial_rows: list[dict[str, Any]]
    aggregate_rows: list[dict[str, Any]]
    output_paths: dict[str, Path]


def run_experiment_suite(config: ExperimentSuiteConfig) -> ExperimentSuiteResult:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    trial_rows: list[dict[str, Any]] = []

    if config.family in ("all", SURVEY_DROPOUT):
        trial_rows.extend(_run_survey_dropout_trials(config))
    if config.family in ("all", DISASTER_URGENT_DROPOUT):
        trial_rows.extend(_run_disaster_trials(config))
    if config.family not in ("all", SURVEY_DROPOUT, DISASTER_URGENT_DROPOUT):
        raise ValueError(f"Unknown experiment family: {config.family}")

    aggregate_rows = aggregate_trials(trial_rows)
    output_paths = write_experiment_outputs(config.output_dir, trial_rows, aggregate_rows)
    if config.make_plots:
        output_paths.update(write_experiment_plots(config, trial_rows, aggregate_rows))
    return ExperimentSuiteResult(
        trial_rows=trial_rows,
        aggregate_rows=aggregate_rows,
        output_paths=output_paths,
    )


def build_survey_dropout_scenario(
    method_name: str,
    *,
    seed: int,
    swarm_size: int,
    dropout_fraction: float,
    ticks: int,
    grid_size: int,
    sensing_radius: int = 1,
    communication_range: int = 3,
) -> ScenarioConfig:
    scenario = build_demo_scenario(
        method_name,
        mission_type="survey",
        grid_size=grid_size,
        uav_count=swarm_size,
        sensing_radius=sensing_radius,
        communication_range=communication_range,
        seed=seed,
        ticks=ticks,
    )
    dropout_tick = dropout_event_tick(ticks)
    scenario.events = [
        CommunicationEvent(tick=dropout_tick, event_type="dropout", payload={"uav_id": uav_id})
        for uav_id in dropped_uav_ids(seed, swarm_size, dropout_fraction)
    ]
    return scenario


def build_disaster_urgent_dropout_scenario(
    method_name: str,
    *,
    seed: int,
    swarm_size: int,
    ticks: int,
    grid_size: int,
    sensing_radius: int = 1,
    communication_range: int = 3,
) -> ScenarioConfig:
    scenario = build_demo_scenario(
        method_name,
        mission_type="disaster_mapping",
        grid_size=grid_size,
        uav_count=swarm_size,
        sensing_radius=sensing_radius,
        communication_range=communication_range,
        seed=seed,
        ticks=ticks,
    )
    urgent_tick = max(1, int(ticks * 0.25))
    dropout_tick = dropout_event_tick(ticks)
    urgent_cell = (max(0, grid_size - 2), max(0, grid_size // 2))
    dropped = dropped_uav_ids(seed, swarm_size, 1 / max(1, swarm_size))
    scenario.events = [
        CommunicationEvent(
            tick=urgent_tick,
            event_type="urgent_sector",
            payload={"cell": urgent_cell},
        ),
        *[
            CommunicationEvent(tick=dropout_tick, event_type="dropout", payload={"uav_id": uav_id})
            for uav_id in dropped
        ],
    ]
    return scenario


def run_trial(
    *,
    family: str,
    scenario: ScenarioConfig,
    dropout_fraction: float,
    dropout_count: int,
    dropout_tick: int | None,
    success_threshold: float,
) -> dict[str, Any]:
    simulation = Simulation.from_config(scenario)
    summary = simulation.run()
    records = simulation.metrics.records
    urgent_response_time = urgent_response_time_from_records(scenario, records)
    completion_tick = summary["ticks_run"] if summary["coverage_ratio"] >= 1.0 else None
    return {
        "family": family,
        "method": scenario.method_name,
        "mission_type": scenario.mission_type,
        "seed": scenario.seed,
        "swarm_size": len(scenario.uavs),
        "dropout_fraction": round(dropout_fraction, 4),
        "dropout_count": dropout_count,
        "dropout_tick": dropout_tick,
        "ticks": scenario.ticks,
        "ticks_run": summary["ticks_run"],
        "final_coverage_ratio": summary["coverage_ratio"],
        "success": summary["coverage_ratio"] >= success_threshold,
        "completion_tick": completion_tick,
        "coverage_auc": normalized_coverage_auc(records, scenario.ticks, summary["coverage_ratio"]),
        "recovery_slope": recovery_slope(records, dropout_tick, scenario.ticks),
        "messages_sent": summary["messages_sent"],
        "role_switches": count_record_changes(records, "uav_roles"),
        "target_changes": count_record_changes(records, "uav_targets"),
        "urgent_response_time": urgent_response_time,
        "urgent_targets": len(summary["urgent_targets"]),
        "termination_reason": summary["termination_reason"],
    }


def aggregate_trials(trial_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in trial_rows:
        key = (
            row["family"],
            row["method"],
            row["mission_type"],
            row["swarm_size"],
            row["dropout_fraction"],
        )
        groups.setdefault(key, []).append(row)

    aggregates: list[dict[str, Any]] = []
    for key, rows in sorted(groups.items()):
        family, method, mission_type, swarm_size, dropout_fraction = key
        success_values = [1.0 if row["success"] else 0.0 for row in rows]
        final_coverage_values = [row["final_coverage_ratio"] for row in rows]
        aggregates.append(
            {
                "family": family,
                "method": method,
                "mission_type": mission_type,
                "swarm_size": swarm_size,
                "dropout_fraction": dropout_fraction,
                "trials": len(rows),
                "success_rate": mean(success_values),
                "success_rate_sem": sem(success_values),
                "mean_final_coverage_ratio": mean(final_coverage_values),
                "final_coverage_sem": sem(final_coverage_values),
                "mean_coverage_auc": _mean_numeric(row["coverage_auc"] for row in rows),
                "mean_recovery_slope": _mean_numeric(row["recovery_slope"] for row in rows),
                "mean_completion_tick": _mean_numeric(row["completion_tick"] for row in rows),
                "mean_messages_sent": _mean_numeric(row["messages_sent"] for row in rows),
                "mean_role_switches": _mean_numeric(row["role_switches"] for row in rows),
                "mean_target_changes": _mean_numeric(row["target_changes"] for row in rows),
                "mean_urgent_response_time": _mean_numeric(row["urgent_response_time"] for row in rows),
            }
        )
    return aggregates


def write_experiment_outputs(
    output_dir: Path,
    trial_rows: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
) -> dict[str, Path]:
    paths = {
        "trials_csv": output_dir / "trials.csv",
        "aggregates_csv": output_dir / "aggregates.csv",
        "trials_json": output_dir / "trials.json",
        "aggregates_json": output_dir / "aggregates.json",
    }
    _write_csv(paths["trials_csv"], trial_rows)
    _write_csv(paths["aggregates_csv"], aggregate_rows)
    paths["trials_json"].write_text(json.dumps(trial_rows, indent=2) + "\n", encoding="utf-8")
    paths["aggregates_json"].write_text(json.dumps(aggregate_rows, indent=2) + "\n", encoding="utf-8")
    return paths


def write_experiment_plots(
    config: ExperimentSuiteConfig,
    trial_rows: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
) -> dict[str, Path]:
    from matplotlib import pyplot as plt

    paths: dict[str, Path] = {}
    if config.family in ("all", SURVEY_DROPOUT):
        for swarm_size in config.swarm_sizes:
            for metric, label, path_name in (
                ("mean_final_coverage_ratio", "Final coverage ratio", "final_coverage"),
                ("success_rate", "Mission success rate", "success_rate"),
            ):
                rows = [
                    row
                    for row in aggregate_rows
                    if row["family"] == SURVEY_DROPOUT and row["swarm_size"] == swarm_size
                ]
                if not rows:
                    continue
                path = config.output_dir / f"survey_dropout_{path_name}_swarm_{swarm_size}.png"
                _plot_dropout_metric(rows, metric, label, path)
                paths[f"survey_dropout_{path_name}_swarm_{swarm_size}"] = path

    representative_path = config.output_dir / "representative_timeline.png"
    _plot_representative_timeline(config, representative_path)
    paths["representative_timeline"] = representative_path
    plt.close("all")
    return paths


def normalized_coverage_auc(
    records: list[dict[str, Any]],
    max_ticks: int,
    final_coverage: float,
) -> float:
    if max_ticks <= 0:
        return 0.0
    values = [record["coverage_ratio"] for record in records]
    if len(values) < max_ticks:
        values.extend([final_coverage] * (max_ticks - len(values)))
    return sum(values[:max_ticks]) / max_ticks


def recovery_slope(
    records: list[dict[str, Any]],
    dropout_tick: int | None,
    max_ticks: int,
) -> float | None:
    if dropout_tick is None or not records:
        return None
    window = max(1, int(max_ticks * 0.2))
    start_tick = max(0, dropout_tick - 1)
    end_tick = min(max_ticks - 1, dropout_tick + window)
    if end_tick <= start_tick:
        return None
    start_coverage = coverage_at_tick(records, start_tick)
    end_coverage = coverage_at_tick(records, end_tick)
    return (end_coverage - start_coverage) / (end_tick - start_tick)


def urgent_response_time_from_records(
    scenario: ScenarioConfig,
    records: list[dict[str, Any]],
) -> int | None:
    urgent_events = [event for event in scenario.events if event.event_type == "urgent_sector"]
    if not urgent_events:
        return None
    response_times: list[int] = []
    for event in urgent_events:
        cell = tuple(event.payload["cell"])
        for record in records:
            if record["tick"] >= event.tick and cell in record.get("urgent_targets", []):
                response_times.append(record["tick"] - event.tick)
                break
    return min(response_times) if response_times else None


def count_record_changes(records: list[dict[str, Any]], key: str) -> int:
    changes = 0
    previous: dict[str, Any] = {}
    for record in records:
        current = record.get(key, {})
        for uav_id, value in current.items():
            if uav_id in previous and previous[uav_id] != value:
                changes += 1
        previous = dict(current)
    return changes


def coverage_at_tick(records: list[dict[str, Any]], tick: int) -> float:
    value = 0.0
    for record in records:
        if record["tick"] > tick:
            break
        value = record["coverage_ratio"]
    return value


def dropped_uav_ids(seed: int, swarm_size: int, dropout_fraction: float) -> list[str]:
    count = dropout_count(swarm_size, dropout_fraction)
    if count == 0:
        return []
    ids = [f"u{index}" for index in range(swarm_size)]
    return sorted(Random(seed).sample(ids, count), key=lambda uav_id: int(uav_id[1:]))


def dropout_count(swarm_size: int, dropout_fraction: float) -> int:
    if swarm_size <= 0 or dropout_fraction <= 0:
        return 0
    return min(swarm_size, max(1, math.ceil(swarm_size * dropout_fraction)))


def dropout_event_tick(ticks: int) -> int:
    return min(max(0, int(ticks * 0.3)), max(0, ticks - 1))


def _run_survey_dropout_trials(config: ExperimentSuiteConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for swarm_size in config.swarm_sizes:
        for dropout_fraction in config.dropout_fractions:
            for seed in _seeds(config):
                dropped = dropped_uav_ids(seed, swarm_size, dropout_fraction)
                dropout_tick = dropout_event_tick(config.ticks) if dropped else None
                for method in METHODS:
                    scenario = build_survey_dropout_scenario(
                        method,
                        seed=seed,
                        swarm_size=swarm_size,
                        dropout_fraction=dropout_fraction,
                        ticks=config.ticks,
                        grid_size=config.grid_size,
                        sensing_radius=config.sensing_radius,
                        communication_range=config.communication_range,
                    )
                    rows.append(
                        run_trial(
                            family=SURVEY_DROPOUT,
                            scenario=scenario,
                            dropout_fraction=dropout_fraction,
                            dropout_count=len(dropped),
                            dropout_tick=dropout_tick,
                            success_threshold=config.success_threshold,
                        )
                    )
    return rows


def _run_disaster_trials(config: ExperimentSuiteConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for swarm_size in config.swarm_sizes:
        dropout_fraction = round(1 / max(1, swarm_size), 4)
        for seed in _seeds(config):
            dropped = dropped_uav_ids(seed, swarm_size, dropout_fraction)
            for method in METHODS:
                scenario = build_disaster_urgent_dropout_scenario(
                    method,
                    seed=seed,
                    swarm_size=swarm_size,
                    ticks=config.ticks,
                    grid_size=config.grid_size,
                    sensing_radius=config.sensing_radius,
                    communication_range=config.communication_range,
                )
                rows.append(
                    run_trial(
                        family=DISASTER_URGENT_DROPOUT,
                        scenario=scenario,
                        dropout_fraction=dropout_fraction,
                        dropout_count=len(dropped),
                        dropout_tick=dropout_event_tick(config.ticks),
                        success_threshold=config.success_threshold,
                    )
                )
    return rows


def _plot_dropout_metric(rows: list[dict[str, Any]], metric: str, label: str, path: Path) -> None:
    from matplotlib import pyplot as plt

    plt.figure(figsize=(7, 4.2))
    for method in METHODS:
        method_rows = sorted(
            (row for row in rows if row["method"] == method),
            key=lambda row: row["dropout_fraction"],
        )
        if not method_rows:
            continue
        plt.plot(
            [row["dropout_fraction"] for row in method_rows],
            [row[metric] for row in method_rows],
            marker="o",
            label=method,
        )
    plt.xlabel("Dropped UAV fraction")
    plt.ylabel(label)
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_representative_timeline(config: ExperimentSuiteConfig, path: Path) -> None:
    from matplotlib import pyplot as plt

    swarm_size = config.swarm_sizes[len(config.swarm_sizes) // 2]
    dropout_fraction = 0.25 if 0.25 in config.dropout_fractions else config.dropout_fractions[-1]
    seed = config.base_seed
    plt.figure(figsize=(8, 4.4))
    for method in METHODS:
        if config.family == DISASTER_URGENT_DROPOUT:
            scenario = build_disaster_urgent_dropout_scenario(
                method,
                seed=seed,
                swarm_size=swarm_size,
                ticks=config.ticks,
                grid_size=config.grid_size,
                sensing_radius=config.sensing_radius,
                communication_range=config.communication_range,
            )
        else:
            scenario = build_survey_dropout_scenario(
                method,
                seed=seed,
                swarm_size=swarm_size,
                dropout_fraction=dropout_fraction,
                ticks=config.ticks,
                grid_size=config.grid_size,
                sensing_radius=config.sensing_radius,
                communication_range=config.communication_range,
            )
        simulation = Simulation.from_config(scenario)
        simulation.run()
        plt.plot(
            [record["tick"] for record in simulation.metrics.records],
            [record["coverage_ratio"] for record in simulation.metrics.records],
            label=method,
        )
    if config.family == DISASTER_URGENT_DROPOUT:
        plt.axvline(max(1, int(config.ticks * 0.25)), color="#A66F00", linestyle=":", linewidth=1, label="urgent")
    plt.axvline(dropout_event_tick(config.ticks), color="#444444", linestyle="--", linewidth=1, label="dropout")
    plt.xlabel("Tick")
    plt.ylabel("Coverage ratio")
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if value is None else value for key, value in row.items()})


def _mean_numeric(values: Iterable[Any]) -> float | None:
    numeric = [value for value in values if isinstance(value, (int, float))]
    return mean(numeric) if numeric else None


def sem(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return stdev(values) / math.sqrt(len(values))


def _seeds(config: ExperimentSuiteConfig) -> range:
    return range(config.base_seed, config.base_seed + config.seed_count)
