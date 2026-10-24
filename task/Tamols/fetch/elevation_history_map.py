from __future__ import annotations

import math
from dataclasses import dataclass, field
import numpy as np
from scipy import ndimage


@dataclass
class ElevationHistoryMapCfg:
    """Configuration for the local rolling elevation history map."""
    cell_size: float = 0.05
    """Grid resolution in meters."""

    x_range: tuple[float, float] = (-1.0, 1.4)
    """Body-frame X bounds [min_x, max_x] in meters (-1.0m rear, +1.4m front)."""

    y_range: tuple[float, float] = (-0.9, 0.9)
    """Body-frame Y bounds [min_y, max_y] in meters (-0.9m right, +0.9m left)."""

    decay_steps: int = 100
    """Number of steps before an unobserved cell confidence decays."""

    default_height: float = 0.0
    """Default floor height before first observation."""


class ElevationHistoryMap:
    """Robot-centric rolling 2D elevation history map.
    
    Integrates forward radar point cloud / ray hits with robot odometry,
    accumulating a continuous 2D height map under and around the robot's body
    and foot landing zones as the robot advances.
    """

    def __init__(self, cfg: ElevationHistoryMapCfg | None = None):
        self.cfg = cfg or ElevationHistoryMapCfg()
        self.cell_size = self.cfg.cell_size
        self.x_min, self.x_max = self.cfg.x_range
        self.y_min, self.y_max = self.cfg.y_range

        self.nx = int(round((self.x_max - self.x_min) / self.cell_size))
        self.ny = int(round((self.y_max - self.y_min) / self.cell_size))

        # Coordinate axes for grid cell centers in body yaw frame
        self.x_coords = np.linspace(self.x_min, self.x_max, self.nx, endpoint=False) + 0.5 * self.cell_size
        self.y_coords = np.linspace(self.y_min, self.y_max, self.ny, endpoint=False) + 0.5 * self.cell_size

        # Grid state buffers
        self.current_default_height = self.cfg.default_height
        self.height_map = np.full((self.nx, self.ny), self.current_default_height, dtype=np.float32)
        self.valid_mask = np.zeros((self.nx, self.ny), dtype=bool)
        self.age_steps = np.full((self.nx, self.ny), self.cfg.decay_steps, dtype=np.int32)

        # Odometry tracking
        self.last_base_pos_xy: np.ndarray | None = None
        self.last_base_yaw: float | None = None

    def reset(self, initial_height: float | None = None):
        """Reset the history map with the current local ground height."""
        init_h = self.cfg.default_height if initial_height is None else float(initial_height)
        self.current_default_height = init_h
        self.height_map.fill(init_h)
        self.valid_mask.fill(False)
        self.age_steps.fill(self.cfg.decay_steps)
        self.last_base_pos_xy = None
        self.last_base_yaw = None

    def _shift_map_odometry(self, delta_x_body: float, delta_y_body: float, delta_yaw: float):
        """Shift the existing height map according to body displacement."""
        shift_x = -delta_x_body / self.cell_size
        shift_y = -delta_y_body / self.cell_size

        # Sub-pixel shift using ndimage
        if abs(shift_x) > 0.01 or abs(shift_y) > 0.01:
            self.height_map = ndimage.shift(
                self.height_map,
                shift=[shift_x, shift_y],
                mode="nearest",
                order=1,
            )
            valid_float = self.valid_mask.astype(np.float32)
            valid_shifted = ndimage.shift(
                valid_float,
                shift=[shift_x, shift_y],
                mode="constant",
                cval=0.0,
                order=0,
            )
            self.valid_mask = valid_shifted > 0.5
            self.age_steps += 1
            self.valid_mask[self.age_steps > self.cfg.decay_steps] = False

    def update(
        self,
        radar_hit_pos_w: np.ndarray,
        base_pos_w: np.ndarray,
        base_yaw: float,
        radar_distances: np.ndarray | None = None,
        max_valid_distance: float = 5.0,
        ground_estimate: float | None = None,
    ) -> np.ndarray:
        """Update elevation history map with new radar measurements and robot pose.
        
        Args:
            radar_hit_pos_w: [N, 3] world coordinates of radar ray intersections.
            base_pos_w: [3] current robot base position in world frame.
            base_yaw: Current robot heading angle in radians.
            radar_distances: [N] ray intersection distances (-1 for misses).
            max_valid_distance: Maximum valid ray distance.
            ground_estimate: Current estimated ground height under the robot.
            
        Returns:
            Processed height_map array [Nx, Ny].
        """
        if ground_estimate is not None:
            self.current_default_height = float(ground_estimate)
        curr_xy = np.asarray(base_pos_w[:2], dtype=np.float64)

        # 1. Odometry shift if previous pose exists
        if self.last_base_pos_xy is not None:
            delta_w = curr_xy - self.last_base_pos_xy
            cos_y = math.cos(base_yaw)
            sin_y = math.sin(base_yaw)
            delta_x_b = cos_y * delta_w[0] + sin_y * delta_w[1]
            delta_y_b = -sin_y * delta_w[0] + cos_y * delta_w[1]
            delta_yaw = base_yaw - (self.last_base_yaw if self.last_base_yaw is not None else base_yaw)
            self._shift_map_odometry(delta_x_b, delta_y_b, delta_yaw)

        self.last_base_pos_xy = curr_xy.copy()
        self.last_base_yaw = base_yaw

        # 2. Project new radar points into body yaw frame
        hits = np.asarray(radar_hit_pos_w, dtype=np.float32)
        if hits.ndim == 3:
            hits = hits.reshape(-1, 3)

        if radar_distances is not None:
            dists = np.asarray(radar_distances).reshape(-1)
            valid_rays = (dists > 0.0) & (dists <= max_valid_distance)
            hits = hits[valid_rays]

        if len(hits) > 0:
            rel_w = hits[:, :2] - curr_xy
            cos_y = math.cos(base_yaw)
            sin_y = math.sin(base_yaw)
            rel_x_b = cos_y * rel_w[:, 0] + sin_y * rel_w[:, 1]
            rel_y_b = -sin_y * rel_w[:, 0] + cos_y * rel_w[:, 1]
            rel_z_w = hits[:, 2]

            # Convert to grid indices
            idx_x = np.round((rel_x_b - self.x_min) / self.cell_size).astype(int)
            idx_y = np.round((rel_y_b - self.y_min) / self.cell_size).astype(int)

            in_bounds = (
                (idx_x >= 0) & (idx_x < self.nx) &
                (idx_y >= 0) & (idx_y < self.ny)
            )

            idx_x = idx_x[in_bounds]
            idx_y = idx_y[in_bounds]
            z_vals = rel_z_w[in_bounds]

            # Accumulate into grid
            for i, j, z in zip(idx_x, idx_y, z_vals):
                self.height_map[i, j] = z
                self.valid_mask[i, j] = True
                self.age_steps[i, j] = 0

        # 3. Fill unobserved/missing regions to prevent discontinuities
        self._inpaint_unobserved()

        return self.height_map.copy()

    def _inpaint_unobserved(self):
        """Inpaint missing cells using nearest valid cell heights, falling back to current ground height."""
        if not np.any(self.valid_mask):
            self.height_map.fill(self.current_default_height)
            return

        if not np.all(self.valid_mask):
            # Compute distance transform and nearest indices to valid cells
            indices = ndimage.distance_transform_edt(
                ~self.valid_mask, return_distances=False, return_indices=True
            )
            nearest_i = indices[0]
            nearest_j = indices[1]
            self.height_map[~self.valid_mask] = self.height_map[
                nearest_i[~self.valid_mask], nearest_j[~self.valid_mask]
            ]

    def query_height(self, x_body: float, y_body: float) -> float:
        """Query interpolated height at (x, y) in body yaw frame."""
        u = (x_body - self.x_min) / self.cell_size
        v = (y_body - self.y_min) / self.cell_size

        u = np.clip(u, 0, self.nx - 1.001)
        v = np.clip(v, 0, self.ny - 1.001)

        u0 = int(math.floor(u))
        u1 = min(u0 + 1, self.nx - 1)
        v0 = int(math.floor(v))
        v1 = min(v0 + 1, self.ny - 1)

        du = u - u0
        dv = v - v0

        h00 = self.height_map[u0, v0]
        h01 = self.height_map[u0, v1]
        h10 = self.height_map[u1, v0]
        h11 = self.height_map[u1, v1]

        h0 = h00 * (1 - dv) + h01 * dv
        h1 = h10 * (1 - dv) + h11 * dv
        return float(h0 * (1 - du) + h1 * du)
