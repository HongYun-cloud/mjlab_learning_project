from __future__ import annotations

import math
from dataclasses import dataclass
import mujoco
import torch


@dataclass
class SectorPatternCfg:
  """Sector pattern - forward-looking radar/lidar sector pattern.

  Rays are distributed in polar coordinates (radius, angle) across a horizontal sector
  in front of the robot, pointing downwards for height scanning.
  """

  fov_deg: float = 70.0
  """Field of view in degrees."""

  min_radius: float = 0.2
  """Inner radius of the sector in meters."""

  max_radius: float = 2.0
  """Outer radius of the sector in meters."""

  num_radii: int = 12
  """Number of radial distance samples."""

  num_angles: int = 15
  """Number of angular samples across the FOV."""

  direction: tuple[float, float, float] = (0.0, 0.0, -1.0)
  """Ray direction in frame-local coordinates (defaults to downward for height scanning)."""

  def generate_rays(
    self, mj_model: mujoco.MjModel | None, device: str
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate ray pattern.

    Args:
      mj_model: MuJoCo model (unused for sector pattern).
      device: Device for tensor operations.

    Returns:
      Tuple of (local_offsets [N, 3], local_directions [N, 3]).
    """
    del mj_model
    half_fov_rad = math.radians(self.fov_deg / 2.0)
    r = torch.linspace(
      self.min_radius,
      self.max_radius,
      self.num_radii,
      device=device,
      dtype=torch.float32,
    )
    theta = torch.linspace(
      -half_fov_rad,
      half_fov_rad,
      self.num_angles,
      device=device,
      dtype=torch.float32,
    )

    grid_r, grid_theta = torch.meshgrid(r, theta, indexing="ij")

    # In robot frame: +X is forward, +Y is left
    x = grid_r * torch.cos(grid_theta)
    y = grid_r * torch.sin(grid_theta)
    z = torch.zeros_like(x)

    num_rays = grid_r.numel()
    local_offsets = torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=-1)

    dir_t = torch.tensor(self.direction, device=device, dtype=torch.float32)
    dir_t = dir_t / dir_t.norm()
    local_directions = dir_t.unsqueeze(0).expand(num_rays, 3).clone()

    return local_offsets, local_directions

