"""Teacher environment configs.

The teacher uses exactly the same terrain, rewards, events, terminations and
curriculum as the Track task.  The privileged observation split lives in
``task/teacher/config/rl_cfg.py``:

    obs_groups = {"actor": ("critic",), "critic": ("critic",)}

No environment-level observation override is needed.
"""

from myrobot_mjlab.task.track.config.env_cfg import (
    my_go2_flat_env_cfg,
    my_go2_low_clearance_bar_env_cfg,
    my_go2_platform_env_cfg,
    my_go2_plum_env_cfg,
    my_go2_stair_down_env_cfg,
    my_go2_stair_up_env_cfg,
    my_go2_stones_2rows_env_cfg,
    my_go2_tire_env_cfg,
)

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
