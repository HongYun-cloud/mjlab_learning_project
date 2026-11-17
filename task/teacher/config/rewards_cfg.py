"""Reward configuration for the teacher task."""

import math

from mjlab.envs import mdp as envs_mdp
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp
from myrobot_mjlab.config.reward.Trackingreward import (
  aggressive_motion_penalty,
  base_accel_penalty,
  collision_penalty,
  end_position,
  face_target,
  feet_too_close_penalty,
  foot_contact_pattern_penalty,
  FootAccelPenalty,
  JointPowerBalancePenalty,
  joint_power_penalty,
  joint_vel_limit,
  move_in_direction,
  normal_pose,
  reach_target,
  stand_pose,
  standstill,
  stall_penalty,
  track_target_bias,
  track_target_position,
)


def make_rewards() -> dict[str, RewardTermCfg]:
  return {
    "track_target_position": RewardTermCfg(
      func=track_target_position,
      weight=5.0,
      params={
        "command_name": "Goal_direction",
        "t_r": 2.0,
        "dense": False,
        "foot_body_names": ("FR_foot", "FL_foot", "RR_foot", "RL_foot"),
        "height_command_name": "Base_height_target",
      },
    ),
    "track_target_bias": RewardTermCfg(
      func=track_target_bias,
      weight=5.0,
      params={
        "command_name": "Goal_direction",
        "distance_threshold": 2.0,
        "t_r": 4.0,
        "goal_deadzone": 0.15,
      },
    ),
    # 提前终止惩罚：只对非 timeout 的 termination 生效。
    "termination_penalty": RewardTermCfg(
      func=envs_mdp.is_terminated,
      weight=-200.0,
    ),
    # 非足端 contact sensor 命中即惩罚。
    "collision_penalty": RewardTermCfg(
      func=collision_penalty,
      weight=-1.0,
    ),
    # 允许 1~4 脚触地，禁止 0 脚。
    "foot_contact_pattern_penalty": RewardTermCfg(
      func=foot_contact_pattern_penalty,
      weight=-10.0,
      params={
        "sensor_name": "feet_ground_contact",
        "min_contacts": 1,
        "max_contacts": 4,
      },
    ),
    "normal_pose": RewardTermCfg(
      func=normal_pose,
      weight=-1.0,
      params={"command_name": "Goal_direction"},
    ),
    # 终点 0.2m 内：四足中心 XY error 越小，exp(-error/0.05) 越大。
    "end_position": RewardTermCfg(
      func=end_position,
      weight=0.5,
      params={
        "command_name": "Goal_direction",
        "distance_threshold": 0.2,
        "scale": 0.05,
        "foot_body_names": ("FR_foot", "FL_foot", "RR_foot", "RL_foot"),
      },
    ),
    # 进入成功半径的全局正奖励（四足中心 XY + base 高度判定）。
    "goal_arrival_bonus": RewardTermCfg(
      func=reach_target,
      weight=5.0,
      params={
        "command_name": "Goal_direction",
        "distance_threshold": 0.15,
        "foot_body_names": ("FR_foot", "FL_foot", "RR_foot", "RL_foot"),
        "height_command_name": "Base_height_target",
      },
    ),
    "joint_vel": RewardTermCfg(
      func=envs_mdp.joint_vel_l2,
      weight=-0.001,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "joint_power": RewardTermCfg(
      func=joint_power_penalty,
      weight=-0.125,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "vel_norm": 30.0,
        "torque_norm": 25.0,
      },
    ),
    "base_accel_penalty": RewardTermCfg(
      func=base_accel_penalty,
      weight=-1.0e-3,
      params={"asset_cfg": SceneEntityCfg("robot")},
    ),
    "foot_accel_penalty": RewardTermCfg(
      func=FootAccelPenalty,
      weight=-5e-4,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot",
          body_names=("FR_foot", "FL_foot", "RR_foot", "RL_foot"),
        )
      },
    ),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.05),
    "stall_penalty": RewardTermCfg(
      func=stall_penalty,
      weight=-1.0,
      params={
        "penalty_value": 1.0,
        "vel_threshold": 0.2,
        "dist_threshold": 0.15,
        "foot_body_names": ("FR_foot", "FL_foot", "RR_foot", "RL_foot"),
      },
    ),
    "aggressive_motion": RewardTermCfg(
      func=aggressive_motion_penalty,
      weight=-5.0,
      params={"max_speed": 0.8},
    ),
    "moveindirection": RewardTermCfg(
      func=move_in_direction,
      weight=1.0,
      params={
        "command_name": "Goal_direction",
        "goal_deadzone": 0.25,
      },
    ),
    "feet_too_close": RewardTermCfg(
      func=feet_too_close_penalty,
      weight=-0.5,
      params={
        "min_distance": 0.18,
        "foot_pairs": (
          ("FL_foot", "FR_foot"),
          ("RL_foot", "RR_foot"),
        ),
      },
    ),
    "face_target": RewardTermCfg(
      func=face_target,
      weight=0.8,
      params={"min_heading_error": math.pi / 2},
    ),
    "standstill": RewardTermCfg(
      func=standstill,
      weight=-1.0,
      params={
        "command_name": "Goal_direction",
        "distance_threshold": 0.25,
        "foot_body_names": ("FR_foot", "FL_foot", "RR_foot", "RL_foot"),
      },
    ),
    "stand_pose": RewardTermCfg(
      func=stand_pose,
      weight=-10.0,
      params={
        "command_name": "Goal_direction",
        "distance_threshold": 0.25,
        "contact_distance_threshold": 0.15,
        "contact_vel_threshold": 0.05,
        "t_r": 2.0,
        "height_command_name": "Base_height_target",
        "sensor_name": "feet_ground_contact",
        "min_contacts": 4,
        "contact_penalty": 1.0,
        "foot_body_names": ("FR_foot", "FL_foot", "RR_foot", "RL_foot"),
      },
    ),
    "pose": RewardTermCfg(
      func=envs_mdp.posture,
      weight=1.0,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "std": {
          r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.35,
          r".*(FR|FL|RR|RL)_calf_joint.*": 0.75,
        },
      },
    ),
    "joint_power_balance": RewardTermCfg(
      func=JointPowerBalancePenalty,
      weight=-0.2,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "window_s": 2.0,
        "vel_norm": 30.0,
        "torque_norm": 25.0,
        "min_speed": 0.1,
        "eps": 1e-8,
      },
    ),
  }
