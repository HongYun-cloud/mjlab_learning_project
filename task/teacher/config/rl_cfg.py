"""RL configuration for privileged teacher training.

Teacher PPO uses the privileged critic observation group as its actor input.
The value function also uses the privileged group.  RND/symmetry are disabled
by default because the privileged observation set is not the deployable actor
set and the symmetry mirror rules are defined for the deployable actor group.
"""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from myrobot_mjlab.task.track.config.rl_cfg import (
    my_go2_ppo_runner_cfg as _track_ppo_runner_cfg,
)


def my_go2_teacher_ppo_runner_cfg(
    use_rnd: bool = False,
    use_symmetry: bool = False,
) -> RslRlOnPolicyRunnerCfg:
    """Teacher PPO config: actor and critic both use the privileged group."""
    cfg = _track_ppo_runner_cfg(
        use_rnd=use_rnd,
        use_symmetry=use_symmetry,
    )
    cfg.obs_groups = {
        "actor": ("critic",),
        "critic": ("critic",),
    }
    if use_rnd:
        cfg.obs_groups["rnd_state"] = ("rnd_state",)
    return cfg


# Backward-compatible alias for callers that used the copied Track name.
my_go2_ppo_runner_cfg = my_go2_teacher_ppo_runner_cfg
