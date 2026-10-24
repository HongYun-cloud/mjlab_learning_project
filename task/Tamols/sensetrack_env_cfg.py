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
  GoalAlignedVelocityCommandCfg,
  GoalDirectionCommandCfg,
  HeightTargetCommandCfg,
  HeadErrorCommandCfg,
)
from myrobot_mjlab.task.Tamols.fetch import TamolsFootholdCommandCfg
from myrobot_mjlab.config.observation import (
  base_height,
  base_twist,
  desired_base_acc,
  desired_base_pose,
  desired_contact_state,
  desired_joint_positions,
  desired_contact_state,
  desired_joint_positions,
  direction_real,
  direction_to_goal,
  friction_coefficient,
  heading_to_goal_error,
  planar_footholds,
  reference_twist,
  stair_boolean,
  time_left_in_phase,
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
from myrobot_mjlab.config.reward.tamols_reward import (
  tamols_foot_power_penalty,
  tamols_joint_power_penalty,
  track_tamols_base_angular_acceleration,
  track_tamols_base_angular_velocity,
  track_tamols_base_linear_acceleration,
  track_tamols_base_linear_velocity,
  track_tamols_base_orientation,
  track_tamols_base_position,
  track_tamols_contact_schedule,
  track_tamols_desired_joint_positions,
  track_tamols_footholds,
)
from myrobot_mjlab.config.sensor import SectorPatternCfg



# 通用速度任务环境cfg

def make_velocity_env_cfg() -> ManagerBasedRlEnvCfg:

    terrain_scan = RayCastSensorCfg(
      name="terrain_scan",
      frame=ObjRef(type="body", name="base_link", entity="robot"),  # Set per-robot.
      ray_alignment="yaw",
      # 扇区半角 45°、内径 0.1m：原 70°/0.2m 时，Go2 前髋标称落脚点
      # (±0.193, ±0.145) 的极角约 37° 恰好落在扇区外，近身区域只能靠足底雷达补。
      # 射线总数仍是 num_radii*num_angles=180，观测维度不变。
      pattern=SectorPatternCfg(fov_deg=90.0, min_radius=0.1),
      max_distance=5.0,
      exclude_parent_body=True,
      include_geom_groups=(0,),  # Terrain only.
      debug_vis=True,
    )

    # 4 足端局部 3x3 高程微扫描 (每足 9 射线，4 足共 36 维，用于台阶边缘感知与抬腿避障)
    foot_height_scan = RayCastSensorCfg(
      name="foot_height_scan",
      frame=(
        ObjRef(type="body", name="FL_foot", entity="robot"),
        ObjRef(type="body", name="FR_foot", entity="robot"),
        ObjRef(type="body", name="RL_foot", entity="robot"),
        ObjRef(type="body", name="RR_foot", entity="robot"),
      ),
      pattern=GridPatternCfg(size=(0.2, 0.2), resolution=0.1),
      ray_alignment="yaw",
      max_distance=1.0,
      exclude_parent_body=True,
      include_geom_groups=(0,),  # Terrain only.
      debug_vis=False,
      viz=RayCastSensorCfg.VizCfg(
        show_rays=False,
        hit_color=(1.0, 0.0, 1.0, 0.8),  # Magenta 射线
        hit_sphere_color=(1.0, 0.0, 1.0, 1.0),
      ),
    )

    ##
    # Observations
    ##

    actor_terms = {
        "base_twist": ObservationTermCfg(
          func=base_twist,
          params={"in_body_frame": True},
        ),
        "stair_boolean": ObservationTermCfg(
          func=stair_boolean,
          params={"is_stair": False},
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
        "planar_footholds": ObservationTermCfg(
          func=planar_footholds,
          params={
            "command_name": "Tamols_footholds",
            "in_body_frame": True,
            "planar": True,
          },
          noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
        "desired_joint_pos": ObservationTermCfg(
          func=desired_joint_positions,
          params={"command_name": "Tamols_footholds"},
        ),
        "desired_contact_state": ObservationTermCfg(
          func=desired_contact_state,
          params={"command_name": "Tamols_footholds"},
        ),
        "time_left_in_phase": ObservationTermCfg(
          func=time_left_in_phase,
          params={"command_name": "Tamols_footholds"},
        ),
        "reference_twist": ObservationTermCfg(
          func=reference_twist,
          params={"command_name": "Tamols_footholds"},
        ),
        "actions": ObservationTermCfg(func=mdp.last_action),
        "foot_height_scan": ObservationTermCfg(
          func=envs_mdp.height_scan,
          params={"sensor_name": "foot_height_scan"},
          noise=Unoise(n_min=-0.05, n_max=0.05),
          scale=1 / foot_height_scan.max_distance,
        ),
        "terrain_scan": ObservationTermCfg(
          func=envs_mdp.height_scan,
          params={"sensor_name": "terrain_scan"},
          noise=Unoise(n_min=-0.1, n_max=0.1),
          scale=1 / terrain_scan.max_distance,
        ),
      }

    critic_terms = {
        **actor_terms,
        # Critic sees the true (unbiased) joint positions as privileged information.
        # Note: base linear and angular velocities are already included via base_twist in actor_terms.
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
        "desired_base_pose": ObservationTermCfg(
          func=desired_base_pose,
          params={"command_name": "Tamols_footholds"},
        ),
        "desired_base_acc": ObservationTermCfg(
          func=desired_base_acc,
          params={"command_name": "Tamols_footholds"},
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
        # 期望速度指令: TAMOLS 的 Raibert 标称落脚点、期望基座线/角速度与线/角加速度
        # 五项计算全部直接使用它 (见 TamolsMotionCommandCfg.velocity_command_name)。
        # 默认 (未启用 Viser 摇杆) 时方向自动对齐 Goal_direction:
        #   vxy_b = speed * [cos(Δψ), sin(Δψ)]，wz = gain * Δψ
        # 因此 TAMOLS 规划出的落脚点方向始终指向目标点。
        # Viser 中勾选 Twist -> Enable 后，滑条值覆盖该默认行为。
        "twist": GoalAlignedVelocityCommandCfg(
          entity_name="robot",
          resampling_time_range=(6.0, 10.0),
          heading_command=False,
          debug_vis=True,
          goal_command_name="Goal_direction",
          align_speed_range=(0.25, 0.6),
          align_heading=True,
          align_heading_gain=1.5,
          ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.3, 0.6),
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-0.8, 0.8),
          ),
        ),
        "Goal_direction": GoalDirectionCommandCfg(
          entity_name="robot",
          target_distance=3.8,
          cardinal_directions=("E",),
          resampling_time_range=(10.0, 10.0),
          debug_vis=False,
        ),
        "Tamols_footholds": TamolsFootholdCommandCfg(
          entity_name="robot",
          sensor_name="terrain_scan",
          nominal_speed=0.45,
          debug_vis=True,
          dot_radius=0.035,
          dot_color=(1.0, 0.05, 0.05, 0.95),  # 鲜艳小红点 (TAMOLS最优落脚点)
          show_nominal_dots=True,              # 同步显示淡蓝色初猜测点，直观对比规避边缘效果
          nominal_dot_color=(0.2, 0.6, 1.0, 0.7),
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
      # ========================================================================
      # TAMOLS 运动规划与轨迹跟踪奖励 (参考 DTC Table 3)
      # ========================================================================
      "track_tamols_footholds": RewardTermCfg(
        func=track_tamols_footholds,
        weight=6.0,
        params={
          "command_name": "Tamols_footholds",
          "sensor_name": "feet_ground_contact",
          "mode": "exp",
          "std": 0.05,
          "reach_threshold": 0.05,
          "once_per_touchdown": True,
          "only_on_contact": True,
        },
      ),

      "track_tamols_base_orientation": RewardTermCfg(
        func=track_tamols_base_orientation,
        weight=2.0,
        params={"command_name": "Tamols_footholds", "sigma": 90.0},
      ),
      "track_tamols_base_position": RewardTermCfg(
        func=track_tamols_base_position,
        weight=1.5,
        params={
          "command_name": "Tamols_footholds",
          "std": 0.05,
          "reach_threshold": 0.05,
          "once_per_reach": True,
          "track_z_only": True,
        },
      ),
      "track_tamols_base_linear_vel": RewardTermCfg(
        func=track_tamols_base_linear_velocity,
        weight=1.5,
        params={"command_name": "Tamols_footholds", "std": 0.25},
      ),
      "track_tamols_base_angular_vel": RewardTermCfg(
        func=track_tamols_base_angular_velocity,
        weight=0.5,
        params={"command_name": "Tamols_footholds", "sigma": 1.0},
      ),
      # "track_tamols_desired_joint_pos": RewardTermCfg(
      #   func=track_tamols_desired_joint_positions,
      #   weight=0.5,
      #   params={"command_name": "Tamols_footholds", "std": 0.35},
      # ),
      "track_tamols_contact_schedule": RewardTermCfg(
        func=track_tamols_contact_schedule,
        weight=0.5,
        params={"command_name": "Tamols_footholds", "sensor_name": "feet_ground_contact"},
      ),
      "tamols_foot_power": RewardTermCfg(
        func=tamols_foot_power_penalty,
        weight=-0.02,
        params={"sensor_name": "feet_ground_contact"},
      ),
      "tamols_joint_power": RewardTermCfg(
        func=tamols_joint_power_penalty,
        weight=-0.025,
      ),

      # ========================================================================
      # 目标导航与任务奖励
      # ========================================================================
      # "track_base_height": RewardTermCfg(
      #   func=track_base_height,
      #   weight=1.5,
      #   params={"command_name": "Base_height_target", "std": 0.04},
      # ),
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
        weight=0.7,
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
      # "foot_slip": RewardTermCfg(
      #   func=feet_slip_penalty,
      #   weight=-0.1,
      #   params={
      #     "sensor_name": "feet_ground_contact",
      #   },
      # ),
      # # 空中腿数惩罚：三条或四条腿同时在空中（即支撑腿 <= 1 条）时扣分。
      # # 不强求对角步态，允许自然行走/漫步/对角走，但杜绝三腿起跳、全腾空跃进的投机跳跃步态。
      # "airborne_feet_penalty": RewardTermCfg(
      #   func=airborne_feet_penalty,
      #   weight=-1.0,
      #   params={
      #     "sensor_name": "feet_ground_contact",
      #     "max_airborne_feet": 2,
      #   },
      # ),
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

        sensors=(terrain_scan, foot_height_scan),
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