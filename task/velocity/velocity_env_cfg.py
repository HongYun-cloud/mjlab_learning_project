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
# 通用速度任务环境cfg

def make_velocity_env_cfg() -> ManagerBasedRlEnvCfg:

   

    ##
    # Observations
    ##

    actor_terms = {
        "base_lin_vel": ObservationTermCfg(
          func=mdp.builtin_sensor,
          params={"sensor_name": "robot/frame_vel"},
          noise=Unoise(n_min=-0.5, n_max=0.5),
        ),
        "base_ang_vel": ObservationTermCfg(
          func=mdp.builtin_sensor,
          params={"sensor_name": "robot/imu_gyro"},
          noise=Unoise(n_min=-0.2, n_max=0.2),
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
        "command": ObservationTermCfg(
          func=mdp.generated_commands,
          params={"command_name": "twist"},
        ),
      }

    critic_terms = {
        **actor_terms,
        # Critic sees the true (unbiased) joint positions as privileged information.
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
        "twist": StairVelocityCommandCfg(
          entity_name="robot",
          resampling_time_range=(3.0, 8.0),
          debug_vis=True,
          forward_speed=1.0,
          switch_steps=200,
        )
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
      "track_linear_velocity": RewardTermCfg(
        func=mdp.track_linear_velocity,
        weight=2.0,
        params={"command_name": "twist", "std": math.sqrt(0.25)},
      ),
      "track_angular_velocity": RewardTermCfg(
        func=mdp.track_angular_velocity,
        weight=2.0,
        params={"command_name": "twist", "std": math.sqrt(0.5)},
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
        func=mdp.variable_posture,
        weight=1.0,
        params={
          "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
          "command_name": "twist",
          "std_standing": {},  # Set per-robot.
          "std_walking": {},  # Set per-robot.
          "std_running": {},  # Set per-robot.
          "walking_threshold": 0.05,
          "running_threshold": 1.5,
        },
      ),
      "body_ang_vel": RewardTermCfg(
        func=mdp.body_angular_velocity_penalty,
        weight=0.0,  # Override per-robot
        params={"asset_cfg": SceneEntityCfg("robot", body_names=())},  # Set per-robot.
      ),
      "angular_momentum": RewardTermCfg(
        func=mdp.angular_momentum_penalty,
        weight=0.0,  # Override per-robot
        params={"sensor_name": "robot/root_angmom"},
      ),
      "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
      "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1),
      "air_time": RewardTermCfg(
        func=mdp.feet_air_time,
        weight=0.0,  # Override per-robot.
        params={
          "sensor_name": "feet_ground_contact",
          "threshold_min": 0.05,
          "threshold_max": 0.5,
          "command_name": "twist",
          "command_threshold": 0.5,
        },
      ),
      "soft_landing": RewardTermCfg(
        func=mdp.soft_landing,
        weight=-1e-5,
        params={
          "sensor_name": "feet_ground_contact",
          "command_name": "twist",
          "command_threshold": 0.05,
        },
      ),
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
      "out_of_terrain_bounds": TerminationTermCfg(
        func=mdp.out_of_terrain_bounds,
        time_out=True,
      ),
      "reached_edge": TerminationTermCfg(
        func=velocity_terminations.terrain_edge_reached,
        params={"threshold_fraction": 0.9125},
        time_out=True,
      ),
    }
    
    ##
    # Curriculum
    ##
    
    curriculum = {
      "terrain_levels": CurriculumTermCfg(
        func=mdp.terrain_levels_vel,
        params={"command_name": "twist"},
      ),
    }

    return ManagerBasedRlEnvCfg(
      scene=SceneCfg(
        terrain=TerrainEntityCfg(
            terrain_type="generator",
            terrain_generator=replace(ROUGH_TERRAINS_CFG),
            max_init_terrain_level=0,
        ),

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
      episode_length_s=20.0,
  )