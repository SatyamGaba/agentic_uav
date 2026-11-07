"""
Spatial Prompt Grounder
=======================
Serializes confirmed 3D Scene Graph nodes into structured, token-bounded
spatial context blocks for VLM reasoning.
"""

from __future__ import annotations
import math
from typing import List, Tuple
import numpy as np

from contracts import DroneTelemetry, SceneNode


def compute_relative_bearing_elevation(telemetry: DroneTelemetry, node: SceneNode) -> Tuple[float, float, float]:
    """Computes Euclidean distance, relative bearing in degrees, and elevation angle in degrees.
    
    Returns:
        (distance_m, relative_bearing_deg, elevation_deg)
        - relative_bearing_deg: in [-180, 180], where 0° is directly ahead along UAV heading,
                                positive is right, negative is left.
        - elevation_deg: in [-90, 90], positive is above UAV, negative is below.
    """
    dx = float(node.centroid_world[0] - telemetry.position[0])
    dy = float(node.centroid_world[1] - telemetry.position[1])
    dz = float(node.centroid_world[2] - telemetry.altitude)

    dist_horiz = math.hypot(dx, dy)
    dist_3d = math.hypot(dist_horiz, dz)

    # Angle to landmark in world frame (CCW from East)
    world_bearing = math.atan2(dy, dx)
    # Relative bearing: difference from UAV yaw (heading)
    rel_bearing_rad = (world_bearing - telemetry.yaw + math.pi) % (2.0 * math.pi) - math.pi
    # In camera/cockpit view convention: positive bearing is right, negative is left
    rel_bearing_deg = -math.degrees(rel_bearing_rad)

    elevation_rad = math.atan2(dz, max(0.1, dist_horiz))
    elevation_deg = math.degrees(elevation_rad)

    return dist_3d, rel_bearing_deg, elevation_deg


def serialize_scene_graph(
    nodes: List[SceneNode],
    telemetry: DroneTelemetry,
    max_nodes: int = 15,
) -> str:
    """Serializes confirmed nodes into a high-density, token-efficient text prompt."""
    if not nodes:
        return "[CONFIRMED 3D SCENE GRAPH]\n  (No confirmed landmarks mapped yet - proceed with visual exploration)"

    # Prioritize landmarks by proximity to UAV
    scored_nodes = []
    for node in nodes:
        dist, bearing, elev = compute_relative_bearing_elevation(telemetry, node)
        scored_nodes.append((dist, bearing, elev, node))
    scored_nodes.sort(key=lambda item: item[0])

    lines = [
        "[CONFIRMED 3D SCENE GRAPH - WORLD ENU FRAME]",
        f"UAV State: Pos=(X={telemetry.position[0]:.1f}m, Y={telemetry.position[1]:.1f}m, Alt={telemetry.altitude:.1f}m) | "
        f"Heading={math.degrees(telemetry.yaw):.0f}° | Speed={telemetry.speed:.1f}m/s",
        "Confirmed Landmarks:",
    ]

    for dist, bearing, elev, node in scored_nodes[:max_nodes]:
        lines.append(
            f"  - Node #{node.node_id:02d} [{node.label}]: "
            f"Pos=(X={node.centroid_world[0]:.1f}, Y={node.centroid_world[1]:.1f}, Z={node.centroid_world[2]:.1f}m) | "
            f"Rel: Dist={dist:.1f}m, Bearing={bearing:+.1f}°, Elev={elev:+.1f}° | "
            f"Dim: ({node.dimensions_3d[0]:.1f}m x {node.dimensions_3d[1]:.1f}m x {node.dimensions_3d[2]:.1f}m) | "
            f"Conf={node.confidence:.2f}"
        )

    return "\n".join(lines)
