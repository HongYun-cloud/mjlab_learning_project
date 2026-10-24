import math
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
  stair_boolean,
)
from myrobot_mjlab.config.reward.Trackingreward import (
  airborne_feet_penalty,
  base_angular_velocity_penalty,
  base_vertical_velocity_penalty,
  body_lateral_velocity_penalty,
  diagonal_trot_reward,
  face_target,
  feet_slip_penalty,
  reach_target,
  stall_penalty,
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
        "stair_boolean": ObservationTermCfg(
          func=stair_boolean,
          params={"is_stair": False},
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
        "height_scan": ObservationTermCfg(
          func=envs_mdp.height_scan,
          params={"sensor_name": "terrain_scan"},
          noise=Unoise(n_min=-0.1, n_max=0.1),
          scale=1 / terrain_scan.max_distance,
        ),
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
          "distance_threshold": 0.05,
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
          debug_vis=False,
        ),
        "Base_height_target": HeightTargetCommandCfg(
          entity_name="robot",
          nominal_height=0.28,
          foot_body_names=("FR_foot", "FL_foot", "RR_foot", "RL_foot"),
          is_stair=False,
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
        weight=10.0,
        params={"command_name": "Goal_direction", "dense": False},
      ),
      "track_target_bias": RewardTermCfg(
        func=track_target_bias,
        weight=1.5,
        params={"command_name": "Goal_direction"},
      ),
      "track_base_height": RewardTermCfg(
        func=track_base_height,
        weight=1.5,
        params={"command_name": "Base_height_target", "std": 0.04},
      ),
      "stall_penalty": RewardTermCfg(
        func=stall_penalty,
        weight=2.0,
        params={"penalty_value": -5.0},
      ),
      "face_target": RewardTermCfg(
        func=face_target,
        weight=0.8,
      ),
      "lateral_vel": RewardTermCfg(
        func=body_lateral_velocity_penalty,
        weight=-0.1,
      ),
      "body_ang_vel": RewardTermCfg(
        func=base_angular_velocity_penalty,
        weight=-0.15,
      ),
      "base_vertical_vel": RewardTermCfg(
        func=base_vertical_velocity_penalty,
        weight=-0.5,
      ),
      "upright": RewardTermCfg(
        func=mdp.upright,
        weight=1.0,
        params={                
          "std": math.sqrt(0.2),
          "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
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
      "angular_momentum": RewardTermCfg(
        func=mdp.angular_momentum_penalty,
        weight=0.0,  # Override per-robot
        params={"sensor_name": "robot/root_angmom"},
      ),
      "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
      "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.05),
      "air_time": RewardTermCfg(
        func=mdp.feet_air_time,
        weight=0.25,
        params={
          "sensor_name": "feet_ground_contact",
          "threshold_min": 0.05,
          "threshold_max": 0.5,
          "command_name": None,
        },
      ),

      "soft_landing": RewardTermCfg(
        func=mdp.soft_landing,
        weight=-1e-5,
        params={
          "sensor_name": "feet_ground_contact",
          "command_name": None,
        },
      ),
      "foot_slip": RewardTermCfg(
        func=feet_slip_penalty,
        weight=-0.1,
        params={
          "sensor_name": "feet_ground_contact",
        },
      ),
      # 空中腿数惩罚：三条或四条腿同时在空中（即支撑腿 <= 1 条）时扣分。
      # 不强求对角步态，允许自然行走/漫步/对角走，但杜绝三腿起跳、全腾空跃进的投机跳跃步态。
      "airborne_feet_penalty": RewardTermCfg(
        func=airborne_feet_penalty,
        weight=-1.0,
        params={
          "sensor_name": "feet_ground_contact",
          "max_airborne_feet": 2,
        },
      ),
      "joint_acc": RewardTermCfg(
        func=envs_mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
      ),
    }

    ##
    # Terminations
    ##
    
    terminations = {
      "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
      "reach_target": TerminationTermCfg(
        func=reach_target,
        time_out=True,
        params={
          "command_name": "Goal_direction",
          "distance_threshold": 0.05,
        },
      ),
      "fell_over": TerminationTermCfg(
        func=mdp.bad_orientation,
        params={"limit_angle": math.radians(70.0)},
      ),
      "out_of_terrain_bounds": TerminationTermCfg(
        func=mdp.out_of_terrain_bounds,
        time_out=True,
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
    
    curriculum = {
      "terrain_levels": CurriculumTermCfg(
        func=terrain_levels_reach_target,
        params={
          "command_name": "Goal_direction",
          "distance_threshold": 0.05,
          "required_reach_count": 3,
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
      episode_length_s=10.0,
  )