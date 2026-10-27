"""Custom curriculum functions for stair/velocity training."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")


def terrain_levels_forward_dist(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    move_up_dist: float = 6.0,
    move_down_dist: float = 1.5,
    asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
) -> dict[str, torch.Tensor]:
    """Curriculum: advance terrain level when the robot walks ``move_up_dist`` metres
    along its *forward* direction from its spawn position in a single episode.

    升阶条件
    --------
    机器人从本次 Episode 出生点出发，沿着出生时的前进朝向实际走过的位移 ≥ ``move_up_dist``（默认 6.0 m）。
    （例如从 y=3.0 出生，翻越楼梯中心 y=0.0 到达对侧 y=-3.0，位移为 6.0 m）。

    降阶条件
    --------
    机器人在前进方向上位移 < ``move_down_dist``（默认 1.5 m，未能成功上楼或卡住倒地）。

    保持条件
    --------
    前进位移在 [1.5m, 6.0m) 之间（如走了一半或正在尝试登顶），保持当前难度继续训练。

    Parameters
    ----------
    env:
        当前环境实例。
    env_ids:
        本次 reset 涉及的 env 索引。
    command_name:
        速度命令名称，对应 ``env.command_manager`` 中的 key。
    move_up_dist:
        升阶所需的**前进方向位移**，单位米，默认 6.0。
    move_down_dist:
        降阶阈值位移，单位米，默认 1.5。
    asset_cfg:
        机器人实体配置，默认为 ``SceneEntityCfg("robot")``。
    """
    asset: Entity = env.scene[asset_cfg.name]

    terrain = env.scene.terrain
    assert terrain is not None
    terrain_generator = terrain.cfg.terrain_generator
    assert terrain_generator is not None

    # ---------- 1. 获取出生起点与当前终点 ----------
    curr_pos_w = asset.data.root_link_pos_w[env_ids, :2]  # (N, 2) 当前（终点）世界位置

    if hasattr(env, "spawn_root_pos_w"):
        spawn_pos_w = env.spawn_root_pos_w[env_ids]
    else:
        # 兼容性降级：若未记录则默认距 origin y=3.0 出生
        spawn_pos_w = env.scene.env_origins[env_ids, :2].clone()
        spawn_pos_w[:, 1] += 3.0

    if hasattr(env, "spawn_heading_w"):
        spawn_heading = env.spawn_heading_w[env_ids]
    else:
        # 默认面朝 -Y 方向 (yaw = -pi/2)
        spawn_heading = -1.57 * torch.ones(len(env_ids), device=env.device)

    # ---------- 2. 计算沿出生前进方向的位移 ----------
    # 位移矢量 (世界系)
    delta_pos = curr_pos_w - spawn_pos_w  # (N, 2)

    # 出生时的前进方向单位向量 (世界系)
    fwd_dir = torch.stack([torch.cos(spawn_heading), torch.sin(spawn_heading)], dim=-1)  # (N, 2)

    # 在前进方向上的投影标量位移（沿初始朝向走多远）
    forward_dist = (delta_pos * fwd_dir).sum(dim=1)  # (N,)

    # ---------- 3. 升 / 降阶判断 ----------
    # 升阶：前进位移达到 move_up_dist (6.0m)
    move_up = forward_dist >= move_up_dist

    # 降阶：前进位移不到 move_down_dist (1.5m，卡死或倒地)
    move_down = forward_dist < move_down_dist
    move_down = move_down & ~move_up

    # 初始 reset（step_counter == 0）时机器人还在出生点，冻结不升不降
    if env.common_step_counter == 0:
        move_up = torch.zeros_like(move_up)
        move_down = torch.zeros_like(move_down)

    # ---------- 4. 更新地形等级 ----------
    terrain.update_env_origins(env_ids, move_up, move_down)

    # ---------- 5. 统计与监控指标 ----------
    levels = terrain.terrain_levels.float()
    result: dict[str, torch.Tensor] = {
        "mean": torch.mean(levels),
        "max": torch.max(levels),
        "mean_forward_dist": forward_dist.mean(),
        "move_up_rate": move_up.float().mean(),
        "move_down_rate": move_down.float().mean(),
    }

    sub_terrain_names = list(terrain_generator.sub_terrains.keys())
    terrain_origins = terrain.terrain_origins
    assert terrain_origins is not None
    num_cols = terrain_origins.shape[1]
    if num_cols == len(sub_terrain_names):
        types = terrain.terrain_types
        for i, name in enumerate(sub_terrain_names):
            mask = types == i
            if mask.any():
                result[name] = torch.mean(levels[mask])

    return result


def terrain_levels_reach_target(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str = "Goal_direction",
    distance_threshold: float = 0.05,
    required_reach_count: int = 3,
    num_fails_to_level_down: int = 5,
    allow_downgrade: bool = True,
    increase_distance_on_flat: bool = False,
    distance_step: float = 0.5,
    max_distance: float = 5.0,
    min_distance: float = 2.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
) -> dict[str, torch.Tensor]:
    """Curriculum: advance terrain level (or target distance) when the robot reaches
    the target point (within ``distance_threshold``, default 0.05m = 5cm) multiple times
    (``required_reach_count``, default 3).

    升阶规则 (Level Up)
    -------------------
    环境在当前难度下成功到达目标点 5cm 以内累计达到 ``required_reach_count`` 次（默认 3 次），
    判定为已经掌握当前难度，触发升阶（楼梯高度变高 / 平地目标距离增大）。

    降阶规则 (Level Down)
    -------------------
    在当前难度连续失败达到 ``num_fails_to_level_down`` 次（默认 5 次），
    触发降阶以帮助恢复运动策略。

    重置与初始状态
    -------------
    初始 reset（common_step_counter == 0）时冻结升降阶。
    升阶或降阶后，该环境的计数器清零，重新开始统计。
    """
    from myrobot_mjlab.config.reward.Trackingreward import _get_goal_pos

    if len(env_ids) == 0:
        return {}

    # 1. 判断本次 reset 的环境是否成功到达目标点 5cm 以内
    if hasattr(env, "termination_manager") and "reach_target" in env.termination_manager.active_terms:
        reached = env.termination_manager.get_term("reach_target")[env_ids]
    else:
        asset: Entity = env.scene[asset_cfg.name]
        robot_pos = asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2]
        goal_pos = _get_goal_pos(env, command_name=command_name)[env_ids, :2]
        dist = torch.linalg.norm(robot_pos - goal_pos, dim=-1)
        reached = dist <= distance_threshold

    # 2. 状态计数器初始化
    if not hasattr(env, "target_reach_count"):
        env.target_reach_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    if not hasattr(env, "target_fail_count"):
        env.target_fail_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    if not hasattr(env, "total_reach_count"):
        env.total_reach_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    # 3. 统计到达次数与失败次数
    reached_mask = reached.bool()
    failed_mask = ~reached_mask

    env.target_reach_count[env_ids[reached_mask]] += 1
    env.total_reach_count[env_ids[reached_mask]] += 1
    env.target_fail_count[env_ids[reached_mask]] = 0

    env.target_fail_count[env_ids[failed_mask]] += 1

    # 4. 判断升阶与降阶
    # 到达多次目标点（默认 3 次）后升级
    move_up = env.target_reach_count[env_ids] >= required_reach_count

    if allow_downgrade:
        move_down = env.target_fail_count[env_ids] >= num_fails_to_level_down
        move_down = move_down & ~move_up
    else:
        move_down = torch.zeros_like(move_up)

    # 初始 reset（step_counter == 0）时冻结不升不降
    if env.common_step_counter == 0:
        move_up = torch.zeros_like(move_up)
        move_down = torch.zeros_like(move_down)

    # 升阶或降阶后清零计数器
    reset_cnt_mask = move_up | move_down
    if reset_cnt_mask.any():
        env.target_reach_count[env_ids[reset_cnt_mask]] = 0
        env.target_fail_count[env_ids[reset_cnt_mask]] = 0

    # 5. 更新地形难度或平地目标距离
    terrain = env.scene.terrain
    has_terrain_gen = (
        terrain is not None
        and getattr(terrain.cfg, "terrain_generator", None) is not None
        and getattr(terrain, "terrain_origins", None) is not None
    )

    if has_terrain_gen:
        terrain.update_env_origins(env_ids, move_up, move_down)
    elif increase_distance_on_flat and command_name in env.command_manager.active_terms:
        cmd_term = env.command_manager.get_term(command_name)
        if not hasattr(cmd_term, "target_distances"):
            cmd_term.target_distances = torch.full(
                (env.num_envs,), float(cmd_term.cfg.target_distance), device=env.device, dtype=torch.float32
            )
        if move_up.any():
            cmd_term.target_distances[env_ids[move_up]] = torch.clamp(
                cmd_term.target_distances[env_ids[move_up]] + distance_step,
                max=max_distance,
            )
        if allow_downgrade and move_down.any():
            cmd_term.target_distances[env_ids[move_down]] = torch.clamp(
                cmd_term.target_distances[env_ids[move_down]] - distance_step,
                min=min_distance,
            )

    # 6. 统计与监控指标
    result: dict[str, torch.Tensor] = {
        "reach_rate": reached.float().mean(),
        "move_up_rate": move_up.float().mean(),
        "move_down_rate": move_down.float().mean(),
        "mean_reach_count": env.target_reach_count[env_ids].float().mean(),
        "total_reaches": env.total_reach_count.float().sum(),
    }

    if has_terrain_gen and getattr(terrain, "terrain_levels", None) is not None:
        levels = terrain.terrain_levels.float()
        result["mean"] = torch.mean(levels)
        result["max"] = torch.max(levels)
        result["mean_terrain_level"] = torch.mean(levels)
        result["max_terrain_level"] = torch.max(levels)

        terrain_generator = terrain.cfg.terrain_generator
        if terrain_generator is not None:
            sub_terrain_names = list(terrain_generator.sub_terrains.keys())
            terrain_origins = terrain.terrain_origins
            if terrain_origins is not None and terrain_origins.shape[1] == len(sub_terrain_names):
                types = terrain.terrain_types
                for i, name in enumerate(sub_terrain_names):
                    mask = types == i
                    if mask.any():
                        result[name] = torch.mean(levels[mask])
    elif increase_distance_on_flat and command_name in env.command_manager.active_terms:
        cmd_term = env.command_manager.get_term(command_name)
        if hasattr(cmd_term, "target_distances"):
            result["mean_target_distance"] = cmd_term.target_distances.mean()

    return result




def terrain_levels_stair_peak(
    env,
    env_ids,
    command_name: str = "twist",
    up_streak: int = 5,
    down_streak: int = 3,
) -> dict:
    """楼梯课程: 连续 ``up_streak`` 次到达楼梯顶端升一级, 连续 ``down_streak``
    次未到达降一级 (DTC 楼梯任务专用)。

    成败判据
    --------
    读速度命令项 (StairVelocityCommand, north_peak 模式) 的
    ``peak_reached_episode``: 本 episode 是否曾到达楼梯顶端 (目标点半径
    center_threshold 内)。配合 "reached_peak" 成功终止 (到顶即结束) 与
    10 s 超时终止 (未到顶结束), 每个 episode 恰好产生一次成败判定。

    升降规则
    --------
    - 连续 ``up_streak`` (默认 5) 个成功 episode -> 难度 +1 级;
    - 连续 ``down_streak`` (默认 3) 个失败 episode -> 难度 -1 级;
    - 升级或降级后该环境的成败计数清零, 重新开始统计;
    - 初次 reset (common_step_counter == 0) 冻结, 避免出生即误判。

    底层调用 :meth:`TerrainEntity.update_env_origins` 更新 terrain_levels 与
    env_origins (到顶 max_terrain_level 后随机重置等级)。
    """
    if len(env_ids) == 0:
        return {}

    device = env.device
    cmd = env.command_manager.get_term(command_name)
    reached = getattr(cmd, "peak_reached_episode", None)
    if reached is None:
        reached = torch.zeros(env.num_envs, dtype=torch.bool, device=device)

    if not hasattr(env, "stair_peak_success_streak"):
        env.stair_peak_success_streak = torch.zeros(
            env.num_envs, dtype=torch.long, device=device
        )
        env.stair_peak_fail_streak = torch.zeros(
            env.num_envs, dtype=torch.long, device=device
        )

    ok = reached[env_ids].bool()
    env.stair_peak_success_streak[env_ids] = torch.where(
        ok, env.stair_peak_success_streak[env_ids] + 1,
        torch.zeros_like(env.stair_peak_success_streak[env_ids]),
    )
    env.stair_peak_fail_streak[env_ids] = torch.where(
        ok, torch.zeros_like(env.stair_peak_fail_streak[env_ids]),
        env.stair_peak_fail_streak[env_ids] + 1,
    )

    move_up = env.stair_peak_success_streak[env_ids] >= up_streak
    move_down = (env.stair_peak_fail_streak[env_ids] >= down_streak) & ~move_up

    # 初次 reset 时机器人还没走过任何 episode, 冻结升降
    if env.common_step_counter == 0:
        move_up = torch.zeros_like(move_up)
        move_down = torch.zeros_like(move_down)

    # 升/降级后重新开始计数
    changed = move_up | move_down
    if changed.any():
        env.stair_peak_success_streak[env_ids[changed]] = 0
        env.stair_peak_fail_streak[env_ids[changed]] = 0

    terrain = env.scene.terrain
    has_levels = (
        terrain is not None
        and getattr(terrain, "terrain_origins", None) is not None
    )
    if has_levels:
        terrain.update_env_origins(env_ids, move_up, move_down)

    result: dict = {
        "reach_rate": ok.float().mean(),
        "move_up_rate": move_up.float().mean(),
        "move_down_rate": move_down.float().mean(),
        "success_streak_mean": env.stair_peak_success_streak[env_ids].float().mean(),
        "fail_streak_mean": env.stair_peak_fail_streak[env_ids].float().mean(),
    }
    if has_levels and getattr(terrain, "terrain_levels", None) is not None:
        levels = terrain.terrain_levels.float()
        result["mean"] = torch.mean(levels)
        result["max"] = torch.max(levels)
    return result
