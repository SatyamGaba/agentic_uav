"""2D occupancy grid mapping from multi-view depth pointclouds."""

import math
from typing import Optional, Tuple
import numpy as np


class OccupancyGrid2D:
    """Allocentric 2D occupancy grid centered on the start pose with dynamic recentering.

    Grid coordinates:
        - Image size: (grid_size, grid_size), default 256x256
        - Resolution: meters_per_pixel, default 0.15 m/px
        - Spans: grid_size * meters_per_pixel = 38.4m x 38.4m
        - Center (start position): pixel (grid_size // 2, grid_size // 2)

    Values:
        - Unknown: 127
        - Free: 255
        - Occupied: 0
    """

    def __init__(
        self,
        grid_size: int = 256,
        resolution_m: float = 0.15,
        hit_prob: float = 0.7,
        miss_prob: float = 0.4,
    ):
        self.grid_size = grid_size
        self.resolution = resolution_m
        self.center_px = grid_size // 2
        self.origin_x_m = 0.0
        self.origin_z_m = 0.0

        # Log-odds representation for robust fusion
        self.l_free = math.log(miss_prob / (1.0 - miss_prob))
        self.l_occ = math.log(hit_prob / (1.0 - hit_prob))
        self.log_odds = np.zeros((grid_size, grid_size), dtype=np.float32)

    def world_to_grid(self, x_m: float, z_m: float) -> Tuple[int, int]:
        """Convert allocentric ground coordinates (x_m, z_m) to grid pixel (col, row).

        x_m maps to col (horizontal axis).
        z_m maps to row (vertical axis, -Z forward maps up / smaller row index).
        """
        col = int(round(self.center_px + ((x_m - self.origin_x_m) / self.resolution)))
        row = int(round(self.center_px + ((z_m - self.origin_z_m) / self.resolution)))
        return col, row

    def world_to_grid_array(self, x_m: np.ndarray, z_m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized conversion of allocentric coordinates arrays to grid pixels."""
        cols = np.round(self.center_px + ((x_m - self.origin_x_m) / self.resolution)).astype(int)
        rows = np.round(self.center_px + ((z_m - self.origin_z_m) / self.resolution)).astype(int)
        return cols, rows

    def grid_to_world(self, col: int, row: int) -> Tuple[float, float]:
        """Convert grid pixel (col, row) to allocentric coordinates (x_m, z_m)."""
        x_m = self.origin_x_m + (col - self.center_px) * self.resolution
        z_m = self.origin_z_m + (row - self.center_px) * self.resolution
        return x_m, z_m

    def check_and_recenter(self, agent_x_m: float, agent_z_m: float, margin_px: int = 32) -> None:
        """Dynamically shifts grid buffer if agent nears canvas boundary."""
        gc, gr = self.world_to_grid(agent_x_m, agent_z_m)
        if (
            gc < margin_px
            or gc >= self.grid_size - margin_px
            or gr < margin_px
            or gr >= self.grid_size - margin_px
        ):
            shift_col = gc - self.center_px
            shift_row = gr - self.center_px

            # Roll and zero out newly exposed areas
            new_log_odds = np.zeros_like(self.log_odds)
            r_src_start = max(0, shift_row)
            r_src_end = min(self.grid_size, self.grid_size + shift_row)
            r_dst_start = max(0, -shift_row)
            r_dst_end = min(self.grid_size, self.grid_size - shift_row)

            c_src_start = max(0, shift_col)
            c_src_end = min(self.grid_size, self.grid_size + shift_col)
            c_dst_start = max(0, -shift_col)
            c_dst_end = min(self.grid_size, self.grid_size - shift_col)

            if r_src_end > r_src_start and c_src_end > c_src_start:
                new_log_odds[r_dst_start:r_dst_end, c_dst_start:c_dst_end] = (
                    self.log_odds[r_src_start:r_src_end, c_src_start:c_src_end]
                )

            self.log_odds = new_log_odds
            self.origin_x_m += shift_col * self.resolution
            self.origin_z_m += shift_row * self.resolution

    def update(
        self,
        agent_x_m: float,
        agent_z_m: float,
        obstacle_points_world_xz: np.ndarray,
        free_endpoints_world_xz: Optional[np.ndarray] = None,
        free_ray_step_m: float = 0.15,
    ) -> None:
        """Update occupancy grid with raycasting for obstacles and free space endpoints.

        Args:
            agent_x_m, agent_z_m: Current agent allocentric ground position.
            obstacle_points_world_xz: (N, 2) array of allocentric [x, z] obstacle points.
            free_endpoints_world_xz: Optional (M, 2) array of free space endpoints (floor/max-range).
            free_ray_step_m: Step size along rays for marking free space.
        """
        self.check_and_recenter(agent_x_m, agent_z_m)

        # 1. Clear free space along rays to free endpoints (floor hits and saturated max range)
        if free_endpoints_world_xz is not None and free_endpoints_world_xz.shape[0] > 0:
            num_free = free_endpoints_world_xz.shape[0]
            free_indices = np.linspace(0, num_free - 1, min(num_free, 600), dtype=int)
            for idx in free_indices:
                fx, fz = free_endpoints_world_xz[idx]
                dist = math.sqrt((fx - agent_x_m) ** 2 + (fz - agent_z_m) ** 2)
                if dist < 0.1:
                    continue

                num_steps = max(1, int(dist / free_ray_step_m))
                step_fracs = np.linspace(0.1, 1.0, num_steps)
                for frac in step_fracs:
                    rx = agent_x_m + frac * (fx - agent_x_m)
                    rz = agent_z_m + frac * (fz - agent_z_m)
                    gc, gr = self.world_to_grid(rx, rz)
                    if 0 <= gc < self.grid_size and 0 <= gr < self.grid_size:
                        self.log_odds[gr, gc] = max(-4.0, self.log_odds[gr, gc] + self.l_free)

        # 2. Raycast to obstacles: free along ray, occupied at endpoint
        if obstacle_points_world_xz.shape[0] > 0:
            num_pts = obstacle_points_world_xz.shape[0]
            sample_indices = np.linspace(0, num_pts - 1, min(num_pts, 600), dtype=int)

            for idx in sample_indices:
                ox, oz = obstacle_points_world_xz[idx]
                dist = math.sqrt((ox - agent_x_m) ** 2 + (oz - agent_z_m) ** 2)
                if dist < 0.1:
                    continue

                num_steps = max(1, int(dist / free_ray_step_m))
                step_fracs = np.linspace(0.1, 0.9, num_steps)
                for frac in step_fracs:
                    rx = agent_x_m + frac * (ox - agent_x_m)
                    rz = agent_z_m + frac * (oz - agent_z_m)
                    gc, gr = self.world_to_grid(rx, rz)
                    if 0 <= gc < self.grid_size and 0 <= gr < self.grid_size:
                        self.log_odds[gr, gc] = max(-4.0, self.log_odds[gr, gc] + self.l_free)

                # Obstacle point itself (occupied)
                gc, gr = self.world_to_grid(ox, oz)
                if 0 <= gc < self.grid_size and 0 <= gr < self.grid_size:
                    self.log_odds[gr, gc] = min(4.0, self.log_odds[gr, gc] + self.l_occ)

            # Vectorized obstacle stamping for all obstacle points
            cols, rows = self.world_to_grid_array(obstacle_points_world_xz[:, 0], obstacle_points_world_xz[:, 1])
            valid = (cols >= 0) & (cols < self.grid_size) & (rows >= 0) & (rows < self.grid_size)
            if np.any(valid):
                self.log_odds[rows[valid], cols[valid]] = np.minimum(4.0, self.log_odds[rows[valid], cols[valid]] + self.l_occ)

    def to_uint8_image(self, dilate_obstacles: bool = True) -> np.ndarray:
        """Convert log-odds map to uint8 grayscale image (0=occ, 127=unknown, 255=free).

        If dilate_obstacles=True, performs 1-pixel dilation on occupied obstacles so thin
        walls are clearly visible on the 256x256 map.
        """
        img = np.full((self.grid_size, self.grid_size), 127, dtype=np.uint8)
        img[self.log_odds < -0.4] = 255  # Free
        img[self.log_odds > 0.4] = 0     # Occupied

        if dilate_obstacles:
            occ = (img == 0)
            if np.any(occ):
                dilated_occ = occ.copy()
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        shifted = np.roll(np.roll(occ, dr, axis=0), dc, axis=1)
                        if dr == -1:
                            shifted[-1, :] = False
                        elif dr == 1:
                            shifted[0, :] = False
                        if dc == -1:
                            shifted[:, -1] = False
                        elif dc == 1:
                            shifted[:, 0] = False
                        dilated_occ |= shifted
                img[dilated_occ] = 0

        return img


