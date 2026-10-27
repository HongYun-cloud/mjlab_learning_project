"""DTC 论文对齐版速度任务 (baseline 实验入口).

这个配置刻意把“论文 DTC 的接口/奖励”放在一个独立 task 中，避免和当前
已经加入大量 task-specific workaround 的 Stair/Platform 配置混在一起：

- 速度指令使用 UniformVelocityCommand (不是 north_peak 目标点);
- policy 观测: base twist / gravity / joint pos-vel / actions /
  reference twist / planar footholds xy / desired joint at touchdown /
  contact state / time-left / foot→foothold height scan;
- actor 不看 base pose; critic 保留 TAMOLS base pose/twist/acc 特权量;
- reward 使用论文 Table 3 的主体: base pose 6 项、hard foothold、
  consistency、foot power、joint power、action rate、joint acceleration;
- TAMOLS 每只脚触地重规划, stride 0.93 s, map cell 0.04 m;
- 终止条件为 base 接触地形 (论文), 不加 goal/stop/last-2s。

注意: 当前 active planner 仍是 task/Tamols/fetch 的轻量
TorchTAMOLSMotionPlanner. 完整 IntegratedNLP/GN-SQP 的接入是下一步单独
工作; 本文件先把 policy 侧和训练信号对齐论文, 便于做 A/B。
"""

from __future__ import annotations

import math
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from myrobot_mjlab.config.event.stair_reset import (
    out_of_patch_bounds,
    reset_root_state_from_stair,
)
from myrobot_mjlab.config.terrain.stair_terrain import STAIRS_TERRAINS_CFG
from myrobot_mjlab.task.DTC.config.env_cfg import my_go2_rough_env_cfg
from myrobot_mjlab.task.DTC.observation import terms as dtc_obs
from myrobot_mjlab.task.DTC.tamols.dtc_command import DTCFootholdCommandCfg


def my_go2_dtc_baseline_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = my_go2_rough_env_cfg(play=play)

    # ---- terrain: DTC 论文式连续地形训练 (这里先用 stairs 做 A/B) ----
    cfg.scene.terrain = TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=replace(STAIRS_TERRAINS_CFG),
        max_init_terrain_level=0,
    )
    # Same reset as the stair task: spawn in the center hollow, random yaw;
    # velocity commands decide where to go. No goal/reached-peak termination.
    cfg.events = {
        "reset_base": EventTermCfg(
            func=reset_root_state_from_stair,
            mode="reset",
            params={
                "pose_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (0.0, 0.0),
                    "yaw": (-math.pi, math.pi),
                },
                "velocity_range": {},
                "nominal_height": 0.28,
            },
        ),
        "reset_robot_joints": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (0.0, 0.0),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            },
        ),
    }

    # ---- termination: base contact (paper), plus training patch bounds ----
    base_contact = ContactSensorCfg(
        name="base_ground_contact",
        primary=ContactMatch(mode="body", pattern="base_link", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
    cfg.scene.sensors = tuple(cfg.scene.sensors or ()) + (base_contact,)
    cfg.terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "out_of_patch_bounds": TerminationTermCfg(
            func=out_of_patch_bounds,
            time_out=True,
            params={"half_size": 4.0, "margin": 0.1},
        ),
        "base_contact": TerminationTermCfg(
            func=mdp.illegal_contact,
            params={"sensor_name": base_contact.name},
        ),
    }

    # ---- velocity command: reference twist, no goal target ----
    twist_cmd = cfg.commands["twist"]
    twist_cmd.ranges.lin_vel_x = (-0.8, 0.8)
    twist_cmd.ranges.lin_vel_y = (-0.4, 0.4)
    twist_cmd.ranges.ang_vel_z = (-0.6, 0.6)
    twist_cmd.resampling_time_range = (3.0, 8.0)

    # ---- TAMOLS: paper update rate/gait; lightweight planner for now ----
    foothold_cmd = cfg.commands["Tamols_footholds"]
    assert isinstance(foothold_cmd, DTCFootholdCommandCfg)
    foothold_cmd.gait_cfg.stride_duration = 0.93
    foothold_cmd.gait_cfg.duty_factor = 0.5
    foothold_cmd.cell_size = 0.04
    foothold_cmd.x_range = (-1.0, 1.1)
    foothold_cmd.y_range = (-0.6, 0.6)
    foothold_cmd.swing_obs_enable = True
    foothold_cmd.swing_path_samples = 10
    foothold_cmd.in_place_gait_enable = False
    foothold_cmd.replan_after_all_four_touchdowns = False
    foothold_cmd.replan_per_foot_touchdown = True
    foothold_cmd.debug_vis = play
    foothold_cmd.plan_time_log = play
    for opt in (
        foothold_cmd.optimizer_cfg,
        foothold_cmd.planner_cfg.optimizer_cfg,
    ):
        # Paper feeds the reference twist to TAMOLS; keep the command as the
        # nominal feed-forward (no measured-velocity blend).
        opt.nominal_cmd_blend = 1.0
        opt.max_step_offset = 0.35

    # ---- observations: DTC paper policy interface ----
    actor_terms = cfg.observations["actor"].terms
    critic_terms = cfg.observations["critic"].terms
    drop = ("tamols_footholds", "tamols_gait_phase", "height_scan")
    for key in drop:
        actor_terms.pop(key, None)
        critic_terms.pop(key, None)

    actor_terms["tamols_footholds_xy"] = ObservationTermCfg(
        func=dtc_obs.tamols_footholds_xy,
        params={"command_name": "Tamols_footholds", "entity_name": "robot"},
        noise=Unoise(n_min=-0.02, n_max=0.02),
    )
    actor_terms["tamols_time_left"] = ObservationTermCfg(
        func=dtc_obs.tamols_time_left,
        params={"command_name": "Tamols_footholds"},
    )
    actor_terms["tamols_path_height"] = ObservationTermCfg(
        func=dtc_obs.tamols_path_height,
        params={"command_name": "Tamols_footholds"},
    )
    critic_terms["tamols_footholds_xy"] = ObservationTermCfg(
        func=dtc_obs.tamols_footholds_xy,
        params={"command_name": "Tamols_footholds", "entity_name": "robot"},
    )
    critic_terms["tamols_time_left"] = ObservationTermCfg(
        func=dtc_obs.tamols_time_left,
        params={"command_name": "Tamols_footholds"},
    )
    critic_terms["tamols_path_height"] = ObservationTermCfg(
        func=dtc_obs.tamols_path_height,
        params={"command_name": "Tamols_footholds"},
    )
    critic_terms["tamols_base_lin_acc_desired"] = ObservationTermCfg(
        func=dtc_obs.tamols_base_lin_acc_desired,
        params={"command_name": "Tamols_footholds", "entity_name": "robot"},
    )
    critic_terms["tamols_base_ang_acc_desired"] = ObservationTermCfg(
        func=dtc_obs.tamols_base_ang_acc_desired,
        params={"command_name": "Tamols_footholds", "entity_name": "robot"},
    )

    # ---- rewards: DTC paper Table 3 main terms only ----
    keep = (
        "base_pose_tracking_pos",
        "base_pose_tracking_rot",
        "base_pose_tracking_vel_pos",
        "base_pose_tracking_vel_rot",
        "base_pose_tracking_acc_pos",
        "base_pose_tracking_acc_rot",
        "foothold_hard_tracking",
        "action_consistency",
        "foot_contact_power",
        "joint_power",
        "dtc_action_rate",
        "dtc_joint_acceleration",
    )
    cfg.rewards = {k: cfg.rewards[k] for k in keep if k in cfg.rewards}

    pose_cfg = {
        "base_pose_tracking_pos": (1200.0, 1.0, 0, "pos"),
        "base_pose_tracking_rot": (90.0, 1.0, 0, "rot"),
        "base_pose_tracking_vel_pos": (10.0, 1.0, 1, "pos"),
        "base_pose_tracking_vel_rot": (1.0, 1.0, 1, "rot"),
        "base_pose_tracking_acc_pos": (0.05, 1.0, 2, "pos"),
        "base_pose_tracking_acc_rot": (0.005, 1.0, 2, "rot"),
    }
    for key, (sigma, weight, order, component) in pose_cfg.items():
        term = cfg.rewards.get(key)
        if term is None:
            continue
        term.params["sigma"] = sigma
        term.params["order"] = order
        term.params["component"] = component
        term.weight = weight

    foot = cfg.rewards.get("foothold_hard_tracking")
    if foot is not None:
        foot.weight = 6.0
        foot.params.update(
            eps=1e-5,
            force_threshold=1.0,
            d_min=1e-6,
        )
    cons = cfg.rewards.get("action_consistency")
    if cons is not None:
        cons.weight = 20.0
        cons.params.update(w_p=1.0, dt_compare=0.01, err_cap=0.1)
    for key, weight in (
        ("foot_contact_power", 0.02),
        ("joint_power", 0.025),
        ("dtc_action_rate", 0.02),
        ("dtc_joint_acceleration", 1e-6),
    ):
        if key in cfg.rewards:
            cfg.rewards[key].weight = weight

    # ---- paper training defaults ----
    if not play:
        cfg.episode_length_s = 20.0

    return cfg
