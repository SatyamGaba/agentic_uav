from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt


class Renderer:
    def render_snapshot(self, world, uavs, output_path: Path) -> None:
        grid = []
        for y in range(world.height):
            row = []
            for x in range(world.width):
                sector = world.sectors[(x, y)]
                if sector.blocked:
                    row.append(0.0)
                elif sector.priority == "urgent":
                    row.append(0.5)
                else:
                    row.append(1.0 if sector.coverage >= 1.0 else 0.2)
            grid.append(row)

        figure, axis = plt.subplots(figsize=(6, 6))
        axis.imshow(grid, cmap="viridis", vmin=0.0, vmax=1.0)
        axis.set_xticks(range(world.width))
        axis.set_yticks(range(world.height))
        axis.grid(color="white", linewidth=0.5)
        axis.set_xlim(-0.5, world.width - 0.5)
        axis.set_ylim(world.height - 0.5, -0.5)

        colors = {"coverage": "white", "priority_responder": "red", "relay": "cyan"}
        for uav in uavs.values():
            if not uav.active:
                continue
            axis.scatter(
                uav.cell[0],
                uav.cell[1],
                s=160,
                color=colors.get(uav.role, "white"),
                edgecolors="black",
            )
            axis.text(uav.cell[0], uav.cell[1], uav.uav_id, ha="center", va="center", fontsize=8)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.tight_layout()
        figure.savefig(output_path)
        plt.close(figure)
