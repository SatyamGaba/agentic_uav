"""Allocentric BEV map renderer with trajectory polyline, numbered recall nodes, and heading indicator."""

import math
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from agenticnav_paper_replica.mapping.occupancy_grid import OccupancyGrid2D


class BEVRenderer:
    """Renders the paper-style 256x256 allocentric BEV map for the VLM."""

    def __init__(self, mode: str = "full"):
        self.mode = mode  # "full", "without_bev"

    def render(
        self,
        grid: OccupancyGrid2D,
        node_poses: List[Tuple[float, float, float]],  # List of (x_m, z_m, heading_rad) for nodes 1..t
        current_node_idx: int,
    ) -> Optional[np.ndarray]:
        """Render allocentric BEV map.

        Args:
            grid: Current OccupancyGrid2D instance.
            node_poses: Allocentric poses [(x, z, yaw), ...] for each decision node 1..t.
            current_node_idx: Index of current decision node (1-based).

        Returns:
            (256, 256, 3) uint8 RGB image, or None if mode == "without_bev".
        """
        if self.mode == "without_bev":
            return None

        # Convert occupancy grid to RGB image
        gray_map = grid.to_uint8_image()
        # Create 3-channel RGB image
        rgb_map = np.stack([gray_map, gray_map, gray_map], axis=-1)
        pil_img = Image.fromarray(rgb_map)
        draw = ImageDraw.Draw(pil_img)

        # Try to get default font or fallback
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        # If there are node poses, render trajectory
        if len(node_poses) > 0:
            # Convert poses to grid pixel coordinates
            pixel_points = [grid.world_to_grid(x, z) for x, z, _ in node_poses]

            # 1. Draw trajectory polyline (cyan / light blue)
            if len(pixel_points) >= 2:
                for i in range(len(pixel_points) - 1):
                    p1 = pixel_points[i]
                    p2 = pixel_points[i + 1]
                    draw.line([p1, p2], fill=(0, 150, 255), width=2)

            # 2. Draw past numbered recall nodes (nodes 1 .. t-1)
            for idx, (px, pz, _) in enumerate(node_poses[:-1], start=1):
                col, row = pixel_points[idx - 1]
                # Draw circular node marker
                r = 6
                draw.ellipse(
                    [(col - r, row - r), (col + r, row + r)],
                    fill=(255, 200, 0),  # Amber/Yellow
                    outline=(0, 0, 0),
                    width=1,
                )
                # Node number label
                text = str(idx)
                if font:
                    draw.text((col - 3, row - 5), text, fill=(0, 0, 0), font=font)

            # 3. Draw current agent pose (node t)
            cur_x, cur_z, cur_yaw_rad = node_poses[-1]
            cur_col, cur_row = pixel_points[-1]

            # Draw current position circle
            draw.ellipse(
                [(cur_col - 5, cur_row - 5), (cur_col + 5, cur_row + 5)],
                fill=(255, 0, 0),  # Red
                outline=(255, 255, 255),
                width=1,
            )

            # View specifications: (view_idx, rel_azimuth_deg, label, color, arrow_len)
            view_specs = [
                (0, 0.0, "0", (255, 0, 0), 18),      # Front (Red)
                (1, -90.0, "1", (0, 100, 255), 15),  # Right (Blue)
                (2, 180.0, "2", (180, 0, 200), 15),  # Back (Purple)
                (3, 90.0, "3", (0, 180, 50), 15),    # Left (Green)
            ]

            # Draw 4-view directional indicators
            for v_idx, rel_deg, lbl, color, v_len in view_specs:
                v_yaw = cur_yaw_rad + math.radians(rel_deg)
                d_col = -math.sin(v_yaw)
                d_row = -math.cos(v_yaw)

                tip_col = cur_col + int(round(v_len * d_col))
                tip_row = cur_row + int(round(v_len * d_row))
                draw.line([(cur_col, cur_row), (tip_col, tip_row)], fill=color, width=2)

                # Label near tip
                lbl_col = tip_col + int(round(5 * d_col)) - 3
                lbl_row = tip_row + int(round(5 * d_row)) - 4
                if font:
                    draw.text((lbl_col, lbl_row), lbl, fill=color, font=font)

            # Draw FOV wedge for front camera (90 deg total, +/- 45 deg)
            fov_half_rad = math.radians(45.0)
            wedge_len = 24
            for sign in [-1.0, +1.0]:
                w_yaw = cur_yaw_rad + sign * fov_half_rad
                w_col = -math.sin(w_yaw)
                w_row = -math.cos(w_yaw)
                w_tip_col = cur_col + int(round(wedge_len * w_col))
                w_tip_row = cur_row + int(round(wedge_len * w_row))
                draw.line([(cur_col, cur_row), (w_tip_col, w_tip_row)], fill=(255, 100, 100), width=1)

        # 4. Draw 5m metric scale bar in bottom-left corner
        bar_len_px = int(round(5.0 / grid.resolution))  # ~33 px at 0.15 m/px
        bar_x0 = 12
        bar_y = grid.grid_size - 14
        bar_x1 = bar_x0 + bar_len_px

        # Background box for crisp legibility
        draw.rectangle([(bar_x0 - 4, bar_y - 12), (bar_x1 + 6, bar_y + 6)], fill=(255, 255, 255), outline=(150, 150, 150))
        # Horizontal bar
        draw.line([(bar_x0, bar_y), (bar_x1, bar_y)], fill=(0, 0, 0), width=2)
        # End tick marks
        draw.line([(bar_x0, bar_y - 3), (bar_x0, bar_y + 3)], fill=(0, 0, 0), width=2)
        draw.line([(bar_x1, bar_y - 3), (bar_x1, bar_y + 3)], fill=(0, 0, 0), width=2)
        # Label
        if font:
            draw.text((bar_x0 + bar_len_px // 2 - 6, bar_y - 11), "5m", fill=(0, 0, 0), font=font)

        return np.array(pil_img, dtype=np.uint8)


