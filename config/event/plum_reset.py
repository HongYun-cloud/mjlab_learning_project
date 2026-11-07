from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def randomize_plum_target(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str = "Goal_direction",
    inner_margin: float = 0.3,
    inner_band: float = 0.7,
) -> None:
    """在梅花桩外侧平地的靠内区域随机采样目标点。

    参数：
    - ``inner_margin``：目标点到梅花桩区域外边缘的最小净距。
    - ``inner_band``：从内侧边缘向外使用的平地宽度。
      默认 2m 外侧平地中只使用靠内的 0.3m ~ 1.0m 范围。
    """
    if len(env_ids) == 0:
        return

    terrain = getattr(env.scene, "terrain", None)
    generator_cfg = getattr(
        getattr(terrain, "cfg", None), "terrain_generator", None
    )
    if generator_cfg is None:
        return

    sub = next(
        (
            cfg
            for cfg in generator_cfg.sub_terrains.values()
            if hasattr(cfg, "field_size")
        ),
        None,
    )
    if sub is None:
        return

    field_half = float(min(sub.field_size)) / 2.0
    flat_border = float(sub.flat_border_width)
    inner_min = field_half + float(inner_margin)
    inner_max = min(
        field_half + float(inner_margin) + float(inner_band),
        field_half + flat_border,
    )
    if inner_max <= inner_min:
        return

    n = len(env_ids)
    side = torch.randint(0, 4, (n,), device=env.device)
    along_unit = torch.rand(n, device=env.device) * 2.0 - 1.0
    depth_unit = torch.rand(n, device=env.device)

    along = along_unit * inner_max
    depth = inner_min + depth_unit * (inner_max - inner_min)

    pos = torch.zeros((n, 2), device=env.device, dtype=torch.float32)
    top = side == 0
    bottom = side == 1
    right = side == 2
    left = side == 3

    pos[top, 0] = along[top]
    pos[top, 1] = depth[top]
    pos[bottom, 0] = along[bottom]
    pos[bottom, 1] = -depth[bottom]
    pos[right, 0] = depth[right]
    pos[right, 1] = along[right]
    pos[left, 0] = -depth[left]
    pos[left, 1] = along[left]

    if command_name not in env.command_manager.active_terms:
        return
    command = env.command_manager.get_term(command_name)
    if not hasattr(command, "target_positions"):
        command.target_positions = torch.zeros(
            (env.num_envs, 2), device=env.device, dtype=torch.float32
        )
    command.target_positions[env_ids] = pos
