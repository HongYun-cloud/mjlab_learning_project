import math
import os
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.sensor import (
  GridPatternCfg,
  ObjRef,
  RayCastSensorCfg,
  TerrainHeightSensorCfg,
)
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity.mdp import terminations as velocity_terminations
from mjlab.tasks.velocity import mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig
from myrobot_mjlab.config.command.StairVelocityCommand import StairVelocityCommandCfg
from myrobot_mjlab.config.curriculum.stair_curriculum import (
  terrain_levels_forward_dist,
  terrain_levels_reach_target,
)
from myrobot_mjlab.config.command.AllTrackingCommand import (
  GoalDirectionCommandCfg,
  HeightTargetCommandCfg,
  HeadErrorCommandCfg,
)
from myrobot_mjlab.config.observation.Trackingobservation import (
  base_height,
  direction_real,
  direction_to_goal,
  friction_coefficient,
  heading_to_goal_error,
  target_distance,
  terrain_type,
  time_left,
)
from myrobot_mjlab.config.reward.Trackingreward import (
  airborne_feet_penalty,
  aggressive_motion_penalty,
  base_accel_penalty,
  base_angular_velocity_penalty,
  base_vertical_velocity_penalty,
  body_lateral_velocity_penalty,
  collision_penalty,
  diagonal_trot_reward,
  end_position,
  face_target,
  feet_slip_penalty,
  foot_contact_pattern_penalty,
  FootAccelPenalty,
  joint_power_penalty,
  joint_vel_limit,
  move_in_direction,
  MoveInDirectionFirstN,
  normal_pose,
  reach_target,
  stand_pose,
  standstill,
  stall_penalty,
  speed_limit_penalty,
  track_base_height,
  track_target_bias,
  track_target_position,
  trot_contact_reward,
)
# 通用速度任务环境cfg

def make_velocity_env_cfg() -> ManagerBasedRlEnvCfg:

    terrain_scan = RayCastSensorCfg(
      name="terrain_scan",
      frame=ObjRef(type="body", name="base_link", entity="robot"),  # Set per-robot.
      ray_alignment="yaw",
      pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),
      max_distance=5.0,
      exclude_parent_body=True,
      include_geom_groups=(0,),  # Terrain only.
      debug_vis=True,
    )

    ##
    # Observations
    ##

    actor_terms = {
        "base_ang_vel": ObservationTermCfg(
          func=mdp.builtin_sensor,
          params={"sensor_name": "robot/imu_gyro"},
          noise=Unoise(n_min=-0.2, n_max=0.2),
        ),
        "direction_goal": ObservationTermCfg(
          func=direction_to_goal,
          params={"command_name": "Goal_direction", "in_body_frame": True},
        ),
        "head_error": ObservationTermCfg(
          func=heading_to_goal_error,
          params={"command_name": "Goal_direction"},
        ),
        "target_distance": ObservationTermCfg(
          func=target_distance,
          params={"command_name": "Goal_direction"},
        ),
        "time_left": ObservationTermCfg(
          func=time_left,
          params={"normalize": True},
        ),
        # 地形选项 one-hot: [平地, 楼梯, 轮胎阵, 沟壑/平台, 单木桥]
        # 传 None 时根据场景地形自动推断；各任务可在 env_cfg 中显式覆盖。
        "terrain_type": ObservationTermCfg(
          func=terrain_type,
          params={"terrain_type": None},
        ),
        "height_command": ObservationTermCfg(
          func=mdp.generated_commands,
          params={"command_name": "Base_height_target"},
        ),
        "projected_gravity": ObservationTermCfg(
          func=mdp.projected_gravity,
          noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
        "joint_pos": ObservationTermCfg(
          func=mdp.joint_pos_rel,
          params={"biased": True},
          noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "joint_vel": ObservationTermCfg(
          func=mdp.joint_vel_rel,
          noise=Unoise(n_min=-1.5, n_max=1.5),
        ),
        "actions": ObservationTermCfg(func=mdp.last_action),
        "height_scan": ObservationTermCfg(
          func=envs_mdp.height_scan,
          params={"sensor_name": "terrain_scan"},
          noise=Unoise(n_min=-0.1, n_max=0.1),
          scale=1 / terrain_scan.max_distance,
        ),
      }

    critic_terms = {
        **actor_terms,
        # Critic sees the true (unbiased) joint positions as privileged information.
        "base_lin_vel": ObservationTermCfg(
          func=mdp.builtin_sensor,
          params={"sensor_name": "robot/frame_vel"},
          noise=Unoise(n_min=-0.5, n_max=0.5),
        ),
        "base_height": ObservationTermCfg(
          func=base_height,
        ),
        "direction_real": ObservationTermCfg(
          func=direction_real,
        ),
        "Friction_coefficient": ObservationTermCfg(
          func=friction_coefficient,
        ),
        "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
        "foot_air_time": ObservationTermCfg(
          func=mdp.foot_air_time,
          params={"sensor_name": "feet_ground_contact"},
        ),
        "foot_contact": ObservationTermCfg(
          func=mdp.foot_contact,
          params={"sensor_name": "feet_ground_contact"},
        ),
        "foot_contact_forces": ObservationTermCfg(
          func=mdp.foot_contact_forces,
          params={"sensor_name": "feet_ground_contact"},
        ),
      }

    # Teacher actor uses the privileged observation set.
    actor_terms = critic_terms

    ##
    # RND (Random Network Distillation) 内在好奇心奖励输入
    ##
    # 论文中的好奇心奖励使用状态-动作对 (s, a) 作为输入。这里复用 actor
    # 观测项，但去掉噪声/延迟/历史，并把上一时刻动作 "actions" 也纳入，
    # 因此 RND 实际看到的是 (s_t, a_{t-1})。
    # terrain_type 是任务常量（同一任务内始终相同），不提供新颖性，故排除。
    def _rnd_term(name: str) -> ObservationTermCfg:
        return replace(
          actor_terms[name],
          noise=None,
          delay_min_lag=0,
          delay_max_lag=0,
          history_length=0,
        )

    rnd_terms = {
        name: _rnd_term(name)
        for name in (
          "base_ang_vel",
          "direction_goal",
          "head_error",
          "height_command",
          "projected_gravity",
          "joint_pos",
          "joint_vel",
          "actions",
        )
      }

    ## enable_corruption 扰动开关
    observations = {
        "actor": ObservationGroupCfg(
          terms=actor_terms,
          concatenate_terms=True,
          enable_corruption=True,
        ),
        "critic": ObservationGroupCfg(
          terms=critic_terms,
          concatenate_terms=True,
          enable_corruption=False,
        ),
        # RND 输入组：不经过观测噪声，由 PPO 的 RND 扩展单独使用。
        "rnd_state": ObservationGroupCfg(
          terms=rnd_terms,
          concatenate_terms=True,
          enable_corruption=False,
        ),
      }


    ##
    # Metrics
    ## 这只是一个训练指标，不会进入任何计算

    metrics = {
      "mean_action_acc": MetricsTermCfg(
        func=mdp.mean_action_acc,
      ),
      "reach_target_success": MetricsTermCfg(
        func=reach_target,
        reduce="max",
        params={
          "command_name": "Goal_direction",
          "distance_threshold": 0.15,
          "foot_body_names": ("FR_foot", "FL_foot", "RR_foot", "RL_foot"),
          "height_command_name": "Base_height_target",
        },
      ),
    }


    ##
    # Actions
    ##
    actions: dict[str, ActionTermCfg] = {
        "joint_pos": JointPositionActionCfg(
          entity_name="robot",
          actuator_names=(".*",),
          scale=0.5,  # Override per-robot.
          use_default_offset=True,
        )
      }

    commands: dict[str, CommandTermCfg] = {
        "Goal_direction": GoalDirectionCommandCfg(
          entity_name="robot",
          target_distance=3.2,
          cardinal_directions=("E", "N", "W", "S"),
          resampling_time_range=(10.0, 10.0),
          resample_on_timer=False,
          debug_vis=False,
        ),
        "Base_height_target": HeightTargetCommandCfg(
          entity_name="robot",
          nominal_height=0.28,
          foot_body_names=("FR_foot", "FL_foot", "RR_foot", "RL_foot"),
          height_smooth_alpha=0.05,
          resampling_time_range=(4.0, 10.0),
          debug_vis=True,
        ),
        "Heading_error": HeadErrorCommandCfg(
          entity_name="robot",
          resampling_time_range=(10.0, 10.0),
          debug_vis=False,
        ),
      }

    ##
    # Events
    ##

    events = {
        "reset_base": EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
            "x": (-0.5, 0.5),
            "y": (3.5, 4.0),
            "z": (-0.01, -0.05),
            "yaw": (-1.57, -1.57),
            },
            "velocity_range": {},
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

    rewards = {
      "track_target_position": RewardTermCfg(
        func=track_target_position,
        # (1 / t_r) * 10，t_r = 2.0s -> 5.0
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
      # 提前终止惩罚：非 timeout 的 termination 生效，权重 -200。
      "termination_penalty": RewardTermCfg(
        func=envs_mdp.is_terminated,
        weight=-200.0,
      ),

      # 碰撞惩罚：非足端 contact sensor 命中即返回 1，权重 -1。
      "collision_penalty": RewardTermCfg(
        func=collision_penalty,
        weight=-1.0,
      ),
      # 禁止 0 脚或 4 脚同时触地：允许 1~3 脚触地。
      "foot_contact_pattern_penalty": RewardTermCfg(
        func=foot_contact_pattern_penalty,
        weight=-1.0,
        params={
          "sensor_name": "feet_ground_contact",
          "min_contacts": 1,
          "max_contacts": 4,
        },
      ),
      # 姿态惩罚：|pitch| + |roll|，distance_threshold=None 时全程生效。
      "normal_pose": RewardTermCfg(
        func=normal_pose,
        weight=-1.0,
        params={
          "command_name": "Goal_direction",
          "distance_threshold": 0.25,
        },
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
      # Joint Velocity Penalty: ||dot{q}||^2
      "joint_vel": RewardTermCfg(
        func=envs_mdp.joint_vel_l2,
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
      ),
      # Joint velocity soft limit: sum_i max(|dot{q}_i| - 0.9*limit_i, 0)
      "joint_vel_limit": RewardTermCfg(
        func=joint_vel_limit,
        weight=-1.0,
        params={
          "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
          "vel_limit": 10.0,
        },
      ),

      # Normalized squared joint power:
      # sum_i ((qdot_i / 30) * (tau_i / 25))^2
      "joint_power": RewardTermCfg(
        func=joint_power_penalty,
        weight=-0.025,
        params={
          "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
          "vel_norm": 30.0,
          "torque_norm": 25.0,
        },
      ),

      # Torque penalty: ||tau||^2
      "torque_penalty": RewardTermCfg(
        func=envs_mdp.joint_torques_l2,
        weight=-1.0e-3,
        params={"asset_cfg": SceneEntityCfg("robot")},
      ),
      # Base acceleration penalty: ||a_lin||^2 + 0.02 * ||a_ang||^2
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
      # Speed soft limit: zero below 0.5 m/s; exp((v-0.5)/e)-1 above.
      "speed_limit_penalty": RewardTermCfg(
        func=speed_limit_penalty,
        weight=-1.0,
        params={"max_speed": 1.5},
      ),
      # 论文 Aggressive Motion: (|v_H| - 1)^2 * 1(|v_H| > 1)
      "aggressive_motion": RewardTermCfg(
        func=aggressive_motion_penalty,
        weight=-5.0,
        params={"max_speed": 1.0},
      ),
      "moveindirection": RewardTermCfg(
        func=move_in_direction,
        weight=1.0,
        params={"command_name": "Goal_direction"},
      ),
      # 航向误差 > 90° 时才惩罚，其余情况返回 0。
      "face_target": RewardTermCfg(
        func=face_target,
        weight=0.8,
        params={"min_heading_error": math.pi / 2},
      ),
      # 终点附近生效
      "standstill": RewardTermCfg(
        func=standstill,
        weight=-1.0,
        params={
          "command_name": "Goal_direction",
          "distance_threshold": 0.25,
          "heading_threshold": 0.5,
          "t_r": 2.0,
        },
      ),
      # 终点附近生效
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
        weight=0.5,
        params={
          "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
          "std": {
            r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.35,
            r".*(FR|FL|RR|RL)_calf_joint.*": 0.75,
          },
        },
      ),
      "angular_momentum": RewardTermCfg(
        func=mdp.angular_momentum_penalty,
        weight=0.0,  # Override per-robot
        params={"sensor_name": "robot/root_angmom"},
      ),



      # Foot acceleration penalty: sum_4 ||a_foot||

    }

    ##
    # Terminations
    ##

    terminations = {
      "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
      "fell_over": TerminationTermCfg(
        func=mdp.bad_orientation,
        params={"limit_angle": math.radians(70.0)},
      ),
      "reached_edge": TerminationTermCfg(
        func=velocity_terminations.terrain_edge_reached,
        params={"threshold_fraction": 3.5125},
        time_out=True,
      ),
    }

    ##
    # Curriculum
    ##

    curriculum_mode = os.getenv(
      "MYROBOT_CURRICULUM_MODE", "curriculum"
    ).strip().lower()
    direct_random = curriculum_mode in ("random", "direct", "1", "true")

    curriculum = {
      "terrain_levels": CurriculumTermCfg(
        func=terrain_levels_reach_target,
        params={
          "command_name": "Goal_direction",
          "distance_threshold": 0.15,
          "required_reach_count": 3,
          "max_level": 6,
          "direct_random": direct_random,
        },
      ),
    }

    return ManagerBasedRlEnvCfg(
      scene=SceneCfg(
        terrain=TerrainEntityCfg(
            terrain_type="generator",
            terrain_generator=replace(ROUGH_TERRAINS_CFG),
            max_init_terrain_level=0,
        ),

        sensors=(terrain_scan,),
        num_envs=1,
        extent=2.0,
      ),
      observations=observations,
      actions=actions,
      commands=commands,
      events=events,
      rewards=rewards,
      terminations=terminations,
      curriculum=curriculum,
      metrics=metrics,
      viewer=ViewerConfig(
        origin_type=ViewerConfig.OriginType.ASSET_BODY,
        entity_name="robot",
        body_name="",  # Set per-robot.
        distance=3.0,
        elevation=-5.0,
        azimuth=90.0,
      ),
      sim=SimulationCfg(
        nconmax=35,
        njmax=1500,
        mujoco=MujocoCfg(
          timestep=0.005,
          iterations=10,
          ls_iterations=20,
        ),
      ),
      decimation=4,
      episode_length_s=6.0,
  )