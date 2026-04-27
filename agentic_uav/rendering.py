from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
from matplotlib import colors as mcolors
from matplotlib.patches import Circle
from matplotlib import pyplot as plt

from agentic_uav.gui_support import CELL_STYLES, ROLE_COLORS


class Renderer:
    def render_snapshot(self, world, uavs, output_path: Path) -> None:
        grid = []
        for y in range(world.height):
            row = []
            for x in range(world.width):
                sector = world.sectors[(x, y)]
                row.append(_sector_rgba(sector))
            grid.append(row)

        figure, axis = plt.subplots(figsize=(6, 6))
        figure.patch.set_facecolor("#FFFFFF")
        axis.set_facecolor("#0B0F0D")
        axis.imshow(grid)
        axis.set_xticks(range(world.width))
        axis.set_yticks(range(world.height))
        axis.grid(color=(1, 1, 1, 0.32), linewidth=0.5)
        axis.set_xlim(-0.5, world.width - 0.5)
        axis.set_ylim(world.height - 0.5, -0.5)

        for uav in uavs.values():
            _draw_uav_icon(axis, uav)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.tight_layout()
        figure.savefig(output_path)
        plt.close(figure)


def _sector_rgba(sector) -> tuple[float, float, float, float]:
    if sector.blocked:
        state = "blocked"
    elif sector.priority == "urgent":
        state = "urgent"
    elif sector.coverage >= 1.0:
        state = "covered"
    else:
        state = "uncovered"
    return _to_matplotlib_rgba(CELL_STYLES[state]["fill"])


def _to_matplotlib_rgba(color: str) -> tuple[float, float, float, float]:
    if not color.startswith("rgba("):
        return mcolors.to_rgba(color)

    values = color.removeprefix("rgba(").removesuffix(")").split(",")
    red, green, blue = (int(value.strip()) / 255 for value in values[:3])
    alpha = float(values[3].strip())
    return red, green, blue, alpha


def _draw_uav_icon(axis, uav) -> None:
    x, y = uav.cell
    color = ROLE_COLORS.get(uav.role, "#CBD5E1")
    if not uav.active:
        color = "#66756F"

    arm_kwargs = {
        "color": color,
        "linewidth": 2.0,
        "solid_capstyle": "round",
        "zorder": 4,
    }
    axis.plot([x - 0.22, x + 0.22], [y, y], **arm_kwargs)
    axis.plot([x, x], [y - 0.22, y + 0.22], **arm_kwargs)

    for rotor_x, rotor_y in (
        (x - 0.26, y - 0.26),
        (x + 0.26, y - 0.26),
        (x - 0.26, y + 0.26),
        (x + 0.26, y + 0.26),
    ):
        axis.add_patch(
            Circle(
                (rotor_x, rotor_y),
                0.085,
                facecolor="#FFFFFF",
                edgecolor=color,
                linewidth=1.7,
                zorder=5,
            )
        )

    axis.add_patch(
        Circle((x, y), 0.11, facecolor="#FFFFFF", edgecolor=color, linewidth=1.8, zorder=6)
    )
    axis.text(x, y, uav.uav_id, ha="center", va="center", fontsize=6.5, color="#1F2933", zorder=7)

    if not uav.active:
        axis.plot(
            [x - 0.16, x + 0.16],
            [y - 0.16, y + 0.16],
            color="#FF6B4A",
            linewidth=2.0,
            zorder=8,
        )
        axis.plot(
            [x - 0.16, x + 0.16],
            [y + 0.16, y - 0.16],
            color="#FF6B4A",
            linewidth=2.0,
            zorder=8,
        )
