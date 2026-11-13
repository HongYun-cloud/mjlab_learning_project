"""RL configuration for the Track tasks (Unitree Go2)."""

from dataclasses import dataclass

from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


# 深度相机使用独立的 2D 观测组，actor 用 CNN 编码器处理。
# mjlab 的 SpatialSoftmaxCNNModel = CNN + spatial softmax，适合低分辨率深度图。
_DEPTH_CNN_CFG = {
    "output_channels": [16, 32],
    "kernel_size": [5, 3],
    "stride": [2, 2],
    "padding": "zeros",
    "activation": "elu",
    "max_pool": False,
    "global_pool": "none",
    "spatial_softmax": True,
    "spatial_softmax_temperature": 1.0,
}
_DEPTH_MODEL_CLS = "mjlab.rl.spatial_softmax:SpatialSoftmaxCNNModel"


@dataclass
class TrackPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO config with the optional RSL-RL RND / symmetry extensions."""

    rnd_cfg: dict | None = None
    """Random Network Distillation (curiosity) config.

    RSL-RL scales ``weight`` by the environment step time (0.02 s here), so the
    effective intrinsic-reward coefficient per control step is
    ``weight * 0.02``.
    """

    symmetry_cfg: dict | None = None
    """Symmetry data-augmentation config (RSL-RL ``extensions.Symmetry``)."""


def my_go2_ppo_runner_cfg(
    use_rnd: bool = True,
    use_symmetry: bool = True,
    use_depth: bool = True,
) -> RslRlOnPolicyRunnerCfg:
    """Create RL runner configuration for the Track tasks.

    The two extensions are enabled by default.  Set ``use_rnd=False`` /
    ``use_symmetry=False`` to reproduce the plain PPO baseline.

    ``use_depth=True`` feeds the forward-facing depth camera group into the
    actor.  The depth group is a 2D image, so the actor switches to a CNN
    encoder; set ``use_depth=False`` to keep the original MLP-only actor (the
    camera sensor and depth group are still created by the environment but are
    not consumed by the policy).
    """

    # RND sees the clean (s, a) group defined in tracking_env_cfg.py.
    obs_groups = {"actor": ("actor",), "critic": ("critic",)}
    if use_depth:
        obs_groups["actor"] = ("actor", "depth")
    if use_rnd:
        obs_groups["rnd_state"] = ("rnd_state",)

    rnd_cfg = None
    if use_rnd:
        rnd_cfg = dict(
            # rsl_rl 会再乘以 env.step_dt (0.02s)：实际每步系数 = weight * 0.02。
            # 5.0 约等于每步 0.1 的内在奖励。
            weight=5.0,
            learning_rate=1.0e-3,
            num_outputs=64,
            predictor_hidden_dims=(256, 128),
            target_hidden_dims=(256, 128),
            activation="elu",
            state_normalization=True,
            reward_normalization=True,
            # 常数权重：论文的经验是好奇心奖励主要提升鲁棒性，而不是
            # 靠退火提高样本效率。需要退火时可改为:
            # {"mode": "linear", "initial_step": 0,
            #  "final_step": 10000, "final_value": 0.0}
            weight_schedule=None,
        )

    symmetry_cfg = None
    if use_symmetry:
        symmetry_cfg = dict(
            # 左右镜像：原始 + 1 个镜像配对（2 倍数据）。
            # 论文的四向（左右+前后）版本可改为:
            # "myrobot_mjlab.task.track.config.symmetry:mirror_obs_actions_4way"
            # 注意：Track 的台阶/沟壑只在一侧，前后镜像并不严格成立，
            # 默认只用左右镜像。
            data_augmentation_func=(
                "myrobot_mjlab.task.track.config.symmetry:mirror_obs_actions"
            ),
            use_data_augmentation=True,
            # 论文只做数据增强；镜像损失作为可选的额外正则项。
            use_mirror_loss=False,
            mirror_loss_coeff=0.0,
        )

    actor_kwargs = dict(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    )
    if use_depth:
        actor_kwargs["cnn_cfg"] = _DEPTH_CNN_CFG
        actor_kwargs["class_name"] = _DEPTH_MODEL_CLS

    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(**actor_kwargs),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=TrackPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
            rnd_cfg=rnd_cfg,
            symmetry_cfg=symmetry_cfg,
        ),
        obs_groups=obs_groups,
        experiment_name="go2_velocity",
        save_interval=50,
        num_steps_per_env=24,
        max_iterations=10_000,
    )
