import math
import os
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import GridPatternCfg, ObjRef, RayCastSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.mdp import terminations as velocity_terminations
from mjlab.tasks.velocity import mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG
from mjlab.viewer import ViewerConfig
from myrobot_mjlab.config.command.AllTrackingCommand import (
  GoalAlignedVelocityCommandCfg,
  GoalDirectionCommandCfg,
  HeightTargetCommandCfg,
)
from myrobot_mjlab.config.curriculum.stair_curriculum import (
  terrain_levels_reach_target,
)
from myrobot_mjlab.config.reward.Trackingreward import reach_target
from myrobot_mjlab.task.teacher.config.actor_cfg import (
  make_actor_group,
  make_actor_terms,
)
from myrobot_mjlab.task.teacher.config.critic_cfg import (
  make_critic_group,
  make_critic_terms,
)
from myrobot_mjlab.task.teacher.config.rewards_cfg import make_rewards
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

    base_actor_terms = make_actor_terms(
      height_scan_max_distance=terrain_scan.max_distance
    )
    critic_terms = make_critic_terms(base_actor_terms)
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
          "height_error",
          "projected_gravity",
          "joint_pos",
          "joint_vel",
          "actions",
        )
      }

    observations = {
        "actor": make_actor_group(actor_terms),
        "critic": make_critic_group(critic_terms),
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
        # 标量目标速度：方向跟随 Goal_direction，速度固定为 0.8 m/s。
        "track_base_velocity": GoalAlignedVelocityCommandCfg(
          entity_name="robot",
          goal_command_name="Goal_direction",
          align_speed_range=(0.8, 0.8),
          align_heading=False,
          resampling_time_range=(10.0, 10.0),
          ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
          ),
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

    rewards = make_rewards()

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
      episode_length_s=10.0,
  )