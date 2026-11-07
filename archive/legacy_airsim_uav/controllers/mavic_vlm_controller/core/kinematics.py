"""
Continuous 6-DoF Kinematics & 3D Artificial Potential Field (APF)
================================================================
Deterministic high-rate kinematic velocity integrator with dynamic
jerk limiting, ray clamping, and 3D artificial potential field repulsion.
"""

from __future__ import annotations
import math
import numpy as np
from typing import Tuple, Optional


class JerkLimitedIntegrator:
    """Second-order kinematic velocity and acceleration integrator with jerk constraints."""

    def __init__(
        self,
        a_max: float = 3.0,     # max acceleration in m/s^2
        j_max: float = 8.0,     # max jerk in m/s^3
        tau_v: float = 0.25,    # velocity tracking time constant in seconds
    ):
        self.a_max = a_max
        self.j_max = j_max
        self.tau_v = tau_v

    def step(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        acc: np.ndarray,
        v_cmd: np.ndarray,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Integrates velocity command forward by dt seconds with acceleration/jerk limits."""
        # Desired raw acceleration to match target velocity
        a_desired = (v_cmd - vel) / max(0.01, self.tau_v)

        # Jerk limiting: da = clamp((a_desired - acc) / dt, -j_max, j_max) * dt
        da = a_desired - acc
        jerk = da / max(1e-4, dt)
        jerk_norm = np.linalg.norm(jerk)
        if jerk_norm > self.j_max:
            jerk = (jerk / jerk_norm) * self.j_max
        new_acc = acc + jerk * dt

        # Acceleration limiting
        acc_norm = np.linalg.norm(new_acc)
        if acc_norm > self.a_max:
            new_acc = (new_acc / acc_norm) * self.a_max

        # Integrate velocity and position
        new_vel = vel + new_acc * dt
        new_pos = pos + new_vel * dt
        return new_pos, new_vel, new_acc


class APFPlanner:
    """3D Artificial Potential Field (APF) motion planner for continuous reactive flight."""

    def __init__(
        self,
        k_att: float = 1.0,
        k_rep: float = 8.0,
        d_0: float = 6.0,          # obstacle influence distance in meters
        v_max: float = 4.0,        # max flight speed in m/s
        z_min: float = 1.5,        # min flight ceiling clearance in meters
        z_max: float = 35.0,       # max flight ceiling in meters
    ):
        self.k_att = k_att
        self.k_rep = k_rep
        self.d_0 = d_0
        self.v_max = v_max
        self.z_min = z_min
        self.z_max = z_max

    def compute_force(
        self,
        current_pos: np.ndarray,
        target_waypoint: np.ndarray,
        zone_depths: np.ndarray,
        uav_yaw: float,
        hfov_rad: float = math.radians(65.0),
    ) -> np.ndarray:
        """Computes net 3D force vector (attractive + repulsive) in World ENU."""
        # 1. Attractive force toward target waypoint
        diff = target_waypoint - current_pos
        dist = float(np.linalg.norm(diff))
        if dist > 1e-3:
            f_att = self.k_att * (diff / dist) * min(self.v_max, dist)
        else:
            f_att = np.zeros(3, dtype=np.float64)

        # 2. Repulsive force from 3x3 depth zones
        f_rep = np.zeros(3, dtype=np.float64)
        rows, cols = zone_depths.shape

        for r in range(rows):
            for c in range(cols):
                d = zone_depths[r, c]
                if d < self.d_0 and d > 0.1:
                    # Calculate horizontal angle offset relative to UAV heading
                    col_offset = (c - (cols - 1) / 2.0) / ((cols - 1) / 2.0)
                    azimuth_offset = -col_offset * (hfov_rad / 2.0) # right is negative yaw
                    sector_heading = uav_yaw + azimuth_offset

                    # Obstacle unit vector from UAV in World ENU (assuming horizontal plane for simplicity)
                    r_hat = np.array([math.cos(sector_heading), math.sin(sector_heading), 0.0])

                    # Khatib repulsion formula: k_rep * (1/d - 1/d0) * (1/d^2) * (-r_hat)
                    rep_mag = self.k_rep * (1.0 / d - 1.0 / self.d_0) * (1.0 / (d * d))
                    f_rep += rep_mag * (-r_hat)

        # 3. Vertical boundary envelope repulsion
        f_alt = np.zeros(3, dtype=np.float64)
        curr_z = current_pos[2]
        if curr_z < self.z_min:
            f_alt[2] = 2.0 * (self.z_min - curr_z)
        elif curr_z > self.z_max:
            f_alt[2] = -2.0 * (curr_z - self.z_max)

        # Net force
        f_net = f_att + f_rep + f_alt

        # Clamp net speed magnitude to v_max
        speed = float(np.linalg.norm(f_net))
        if speed > self.v_max:
            f_net = (f_net / speed) * self.v_max

        return f_net
