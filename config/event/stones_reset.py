from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def randomize_stones_target(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str = "Goal_direction",
) -> None:
    """在每个 reset 从 E/W/N/S 四个外侧平台中心随机选一个目标点。"""
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
            if hasattr(cfg, "num_pillars_per_path")
        ),
        None,
    )
    if sub is None:
        return

    levels = terrain.terrain_levels[env_ids].long()
    device = env.device

    platform = torch.tensor(
        sub.platform_size_schedule, device=device, dtype=torch.float32
    )[levels]
    size = torch.tensor(
        sub.pillar_size_schedule, device=device, dtype=torch.float32
    )[levels]
    gap = torch.tensor(
        sub.pillar_gap_schedule, device=device, dtype=torch.float32
    )[levels]

    pitch = size + gap
    stagger_levels = max(int(getattr(sub, "stagger_levels", 9)), 1)
    stagger_t = torch.clamp(levels.float(), 0.0, float(stagger_levels))
    stagger_t = stagger_t / float(stagger_levels)
    stagger = stagger_t * pitch
    per_row = sub.num_pillars_per_path // sub.num_rows
    distance = (
        platform
        + 2.0 * gap
        + size
        + stagger
        + 2.0 * (per_row - 1) * pitch
    )

    directions = torch.tensor(
        ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)),
        device=device,
    )
    choice = torch.randint(0, 4, (len(env_ids),), device=device)
    target_pos = directions[choice] * distance.unsqueeze(-1)

    if command_name not in env.command_manager.active_terms:
        return
    command = env.command_manager.get_term(command_name)
    if not hasattr(command, "target_positions"):
        command.target_positions = torch.zeros(
            (env.num_envs, 2), device=device, dtype=torch.float32
        )
    command.target_positions[env_ids] = target_pos
