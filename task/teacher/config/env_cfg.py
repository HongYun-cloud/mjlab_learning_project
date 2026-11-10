"""Teacher environment configs.

The teacher is trained with privileged observations.  All task-specific
terrains/rewards are shared with the Track tasks; only the actor observation
group is replaced by the privileged critic group.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg

from myrobot_mjlab.task.track.config.env_cfg import (
    my_go2_flat_env_cfg as _track_flat_env_cfg,
    my_go2_low_clearance_bar_env_cfg as _track_low_clearance_bar_env_cfg,
    my_go2_platform_env_cfg as _track_platform_env_cfg,
    my_go2_plum_env_cfg as _track_plum_env_cfg,
    my_go2_stair_down_env_cfg as _track_stair_down_env_cfg,
    my_go2_stair_up_env_cfg as _track_stair_up_env_cfg,
    my_go2_stones_2rows_env_cfg as _track_stones_2rows_env_cfg,
    my_go2_tire_env_cfg as _track_tire_env_cfg,
)


def _make_teacher_cfg(
    env_cfg_fn, play: bool = False
) -> ManagerBasedRlEnvCfg:
    """Build a Track env cfg and promote critic observations to the actor."""
    cfg = env_cfg_fn(play=play)
    # Teacher actor sees the privileged group (base_lin_vel, base_height,
    # foot forces, friction, etc.).  The group is intentionally shared with
    # the critic so both models receive the same privileged observation.
    cfg.observations["actor"] = cfg.observations["critic"]
    return cfg


def my_go2_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    return _make_teacher_cfg(_track_flat_env_cfg, play)


def my_go2_stair_up_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    return _make_teacher_cfg(_track_stair_up_env_cfg, play)


def my_go2_stair_down_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    return _make_teacher_cfg(_track_stair_down_env_cfg, play)


def my_go2_tire_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    return _make_teacher_cfg(_track_tire_env_cfg, play)


def my_go2_platform_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    return _make_teacher_cfg(_track_platform_env_cfg, play)


def my_go2_plum_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    return _make_teacher_cfg(_track_plum_env_cfg, play)


def my_go2_stones_2rows_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    return _make_teacher_cfg(_track_stones_2rows_env_cfg, play)


def my_go2_low_clearance_bar_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    return _make_teacher_cfg(_track_low_clearance_bar_env_cfg, play)


__all__ = [
    "my_go2_flat_env_cfg",
    "my_go2_stair_up_env_cfg",
    "my_go2_stair_down_env_cfg",
    "my_go2_tire_env_cfg",
    "my_go2_platform_env_cfg",
    "my_go2_plum_env_cfg",
    "my_go2_stones_2rows_env_cfg",
    "my_go2_low_clearance_bar_env_cfg",
]
