from __future__ import annotations

from dataclasses import dataclass

from agentic_uav.simulation import (
    CommunicationEvent,
    ScenarioConfig,
    Sector,
    UavConfig,
    WorldConfig,
)

MISSION_TYPES = ["survey", "disaster_mapping"]


@dataclass(frozen=True)
class ScenarioParams:
    method_name: str = "agentic"
    mission_type: str = "disaster_mapping"
    grid_size: int = 12
    uav_count: int = 4
    sensing_radius: int = 1
    communication_range: int = 3
    seed: int = 7
    ticks: int = 50
    heartbeat_interval: int = 3
    urgent_message_ttl: int = 2


def build_demo_scenario(
    method_name: str | None = None,
    *,
    mission_type: str = "disaster_mapping",
    grid_size: int = 12,
    uav_count: int = 4,
    sensing_radius: int = 1,
    communication_range: int = 3,
    seed: int = 7,
    ticks: int = 50,
    heartbeat_interval: int = 3,
    urgent_message_ttl: int = 2,
    params: ScenarioParams | None = None,
) -> ScenarioConfig:
    if params is None:
        params = ScenarioParams(
            method_name=method_name or "agentic",
            mission_type=mission_type,
            grid_size=grid_size,
            uav_count=uav_count,
            sensing_radius=sensing_radius,
            communication_range=communication_range,
            seed=seed,
            ticks=ticks,
            heartbeat_interval=heartbeat_interval,
            urgent_message_ttl=urgent_message_ttl,
        )
    elif method_name is not None:
        params = ScenarioParams(
            method_name=method_name,
            mission_type=params.mission_type,
            grid_size=params.grid_size,
            uav_count=params.uav_count,
            sensing_radius=params.sensing_radius,
            communication_range=params.communication_range,
            seed=params.seed,
            ticks=params.ticks,
            heartbeat_interval=params.heartbeat_interval,
            urgent_message_ttl=params.urgent_message_ttl,
        )

    if params.mission_type not in MISSION_TYPES:
        raise ValueError(f"Unknown mission type: {params.mission_type}")

    sectors = _mission_sectors(params.mission_type, params.grid_size)
    events = _mission_events(params.mission_type, params.grid_size, params.ticks)
    return ScenarioConfig(
        method_name=params.method_name,
        ticks=params.ticks,
        communication_range=params.communication_range,
        sensing_radius=params.sensing_radius,
        heartbeat_interval=params.heartbeat_interval,
        urgent_message_ttl=params.urgent_message_ttl,
        world=WorldConfig(
            width=params.grid_size,
            height=params.grid_size,
            sectors=sectors,
        ),
        uavs=[
            UavConfig(uav_id=f"u{index}", cell=cell)
            for index, cell in enumerate(_uav_start_cells(params.grid_size, params.uav_count))
        ],
        events=events,
        seed=params.seed,
        mission_type=params.mission_type,
    )


def _mission_sectors(mission_type: str, grid_size: int) -> list[Sector]:
    if mission_type == "survey":
        return []
    blocked = _demo_blocked_cells(grid_size)
    return [
        Sector(cell=_clamp_cell((6, 6), grid_size), priority="urgent"),
        *[Sector(cell=cell, blocked=True) for cell in blocked],
    ]


def _mission_events(
    mission_type: str,
    grid_size: int,
    ticks: int,
) -> list[CommunicationEvent]:
    if mission_type == "survey":
        return []
    return [
        CommunicationEvent(
            tick=min(4, max(0, ticks - 1)),
            event_type="urgent_sector",
            payload={"cell": _clamp_cell((1, 6), grid_size)},
        ),
        CommunicationEvent(
            tick=min(6, max(0, ticks - 1)),
            event_type="dropout",
            payload={"uav_id": "u1"},
        ),
    ]


def _demo_blocked_cells(grid_size: int) -> list[tuple[int, int]]:
    candidates = [_clamp_cell((3, 3), grid_size), _clamp_cell((4, 3), grid_size)]
    return list(dict.fromkeys(candidates))


def _clamp_cell(cell: tuple[int, int], grid_size: int) -> tuple[int, int]:
    limit = max(0, grid_size - 1)
    return (min(cell[0], limit), min(cell[1], limit))


def _uav_start_cells(grid_size: int, uav_count: int) -> list[tuple[int, int]]:
    if uav_count <= 0:
        return []

    last = max(0, grid_size - 1)
    mid = grid_size // 2
    anchors = [
        (0, 0),
        (last, 0),
        (0, last),
        (last, last),
        (mid, 0),
        (mid, last),
        (0, mid),
        (last, mid),
    ]

    starts: list[tuple[int, int]] = []
    for cell in anchors + _perimeter_cells(grid_size) + _interior_cells(grid_size):
        if cell not in starts:
            starts.append(cell)
        if len(starts) == uav_count:
            return starts
    return starts


def _perimeter_cells(grid_size: int) -> list[tuple[int, int]]:
    last = max(0, grid_size - 1)
    cells: list[tuple[int, int]] = []
    for x in range(grid_size):
        cells.append((x, 0))
        cells.append((x, last))
    for y in range(1, last):
        cells.append((0, y))
        cells.append((last, y))
    return cells


def _interior_cells(grid_size: int) -> list[tuple[int, int]]:
    return [
        (x, y)
        for y in range(1, max(1, grid_size - 1))
        for x in range(1, max(1, grid_size - 1))
    ]
