"""Student environment configs.

The student uses the deployable Track actor group and the privileged Track
critic group as the teacher observation group.  No environment changes are
needed relative to the Track task configs.
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
    "my_go2_platform_env_cfg",
    "my_go2_plum_env_cfg",
    "my_go2_stones_2rows_env_cfg",
    "my_go2_tire_env_cfg",
    "my_go2_low_clearance_bar_env_cfg",
]
