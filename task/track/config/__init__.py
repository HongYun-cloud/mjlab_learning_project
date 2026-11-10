from mjlab.tasks.registry import register_mjlab_task
from .rl_cfg import my_go2_ppo_runner_cfg
from .runner import TrackOnPolicyRunner
from .env_cfg import my_go2_flat_env_cfg
from .env_cfg import my_go2_stair_env_cfg
from .env_cfg import my_go2_stair_up_env_cfg
from .env_cfg import my_go2_stair_down_env_cfg
from .env_cfg import my_go2_tire_env_cfg
from .env_cfg import my_go2_platform_env_cfg
from .env_cfg import my_go2_plum_env_cfg
from .env_cfg import my_go2_stones_2rows_env_cfg
from .env_cfg import my_go2_low_clearance_bar_env_cfg

register_mjlab_task(
    task_id="Mjlab-Tracking-Flat-Unitree-Go2",
    env_cfg=my_go2_flat_env_cfg(),
    play_env_cfg=my_go2_flat_env_cfg(play=True),
    rl_cfg=my_go2_ppo_runner_cfg(),
    runner_cls=TrackOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Tracking-Stair-Unitree-Go2",
    env_cfg=my_go2_stair_env_cfg(),
    play_env_cfg=my_go2_stair_env_cfg(play=True),
    rl_cfg=my_go2_ppo_runner_cfg(),
    runner_cls=TrackOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Tracking-StairUp-Unitree-Go2",
    env_cfg=my_go2_stair_up_env_cfg(),
    play_env_cfg=my_go2_stair_up_env_cfg(play=True),
    rl_cfg=my_go2_ppo_runner_cfg(),
    runner_cls=TrackOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Tracking-StairDown-Unitree-Go2",
    env_cfg=my_go2_stair_down_env_cfg(),
    play_env_cfg=my_go2_stair_down_env_cfg(play=True),
    rl_cfg=my_go2_ppo_runner_cfg(),
    runner_cls=TrackOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Tracking-TireArray-Unitree-Go2",
    env_cfg=my_go2_tire_env_cfg(),
    play_env_cfg=my_go2_tire_env_cfg(play=True),
    rl_cfg=my_go2_ppo_runner_cfg(),
    runner_cls=TrackOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Tracking-Platform-Unitree-Go2",
    env_cfg=my_go2_platform_env_cfg(),
    play_env_cfg=my_go2_platform_env_cfg(play=True),
    rl_cfg=my_go2_ppo_runner_cfg(),
    runner_cls=TrackOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Tracking-PlumBlossom-Unitree-Go2",
    env_cfg=my_go2_plum_env_cfg(),
    play_env_cfg=my_go2_plum_env_cfg(play=True),
    rl_cfg=my_go2_ppo_runner_cfg(),
    runner_cls=TrackOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Tracking-Stones2Rows-Unitree-Go2",
    env_cfg=my_go2_stones_2rows_env_cfg(),
    play_env_cfg=my_go2_stones_2rows_env_cfg(play=True),
    rl_cfg=my_go2_ppo_runner_cfg(),
    runner_cls=TrackOnPolicyRunner,
)
register_mjlab_task(
    task_id="Mjlab-Tracking-LowClearanceBar-Unitree-Go2",
    env_cfg=my_go2_low_clearance_bar_env_cfg(),
    play_env_cfg=my_go2_low_clearance_bar_env_cfg(play=True),
    rl_cfg=my_go2_ppo_runner_cfg(),
    runner_cls=TrackOnPolicyRunner,
)
