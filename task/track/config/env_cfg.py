import math
from typing import Literal
from dataclasses import replace

from myrobot_mjlab.robot_cfg import (
  GO2_ACTION_SCALE,
  get_go2_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from myrobot_mjlab.task.track.tracking_env_cfg import make_velocity_env_cfg
from mjlab.scene import SceneCfg
from mjlab.terrains import TerrainEntityCfg
from myrobot_mjlab.config.terrain.stair_terrain import STAIRS_TERRAINS_CFG
from myrobot_mjlab.config.terrain.tire_array import TIRE_ARRAY_TERRAINS_CFG
from myrobot_mjlab.config.terrain.step_platform_terrain import STEP_PLATFORM_TERRAINS_CFG
from myrobot_mjlab.config.curriculum.stair_curriculum import terrain_levels_reach_target
from myrobot_mjlab.config.event.stair_reset import reset_root_state_from_stair,out_of_patch_bounds
from myrobot_mjlab.config.event.tire_reset import reset_root_state_from_tire
from myrobot_mjlab.config.reward.stair_reward import track_linear_velocity_stair
from myrobot_mjlab.config.command.AllTrackingCommand import (
  GoalDirectionCommandCfg,
  HeightTargetCommandCfg,
  HeadErrorCommandCfg,
)
from myrobot_mjlab.config.command.TireVelocityCommand import TireTraverseVelocityCommandCfg
from mjlab.managers.command_manager import CommandTermCfg
import traceback




def my_go2_rough_env_cfg(play: bool = False,) -> ManagerBasedRlEnvCfg:
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500       # 当 MuJoCo 进行连续碰撞检测时，允许它最多迭代 500 次来求解碰撞
  cfg.sim.mujoco.impratio = 10              # 接触阻抗（impedance）中法向和切向方向的相对比例
  cfg.sim.mujoco.cone = "elliptic"          # 椭圆摩擦锥
  cfg.sim.contact_sensor_maxmatch = 500     # 最多处理多少个接触
  cfg.sim.mujoco.disableflags = ("multiccd","nativeccd") # 由于mjlab的MULTICCD和xml的margin这两种碰撞机制冲突现在关闭multiccd

  cfg.scene.entities = {"robot": get_go2_robot_cfg()} 


  foot_names = ("FR", "FL", "RR", "RL")
  geom_names = ("FR", "FL", "RR", "RL")

  # 足端和场景的碰撞
  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),
    secondary=ContactMatch(mode="body", pattern=r"terrain",),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )

  thigh_geom_names = tuple(
    f"{leg}_thigh_collision" for leg in foot_names
  )
    # 大腿和场景的碰撞
  thigh_ground_cfg = ContactSensorCfg(
    name="thigh_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      pattern=thigh_geom_names,
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )

  calf_geom_names = tuple(
    f"{leg}_calf_collision" for leg in foot_names
  )
  # 大腿和场景碰撞
  shank_ground_cfg = ContactSensorCfg(
    name="shank_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      pattern=calf_geom_names,
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  

  # trunk_head_ground_cfg = ContactSensorCfg(
  #   name="trunk_ground_touch",
  #   primary=ContactMatch(
  #     mode="geom",
  #     entity="robot",
  #     pattern=("trunk_collision", "head_collision"),
  #   ),
  #   secondary=ContactMatch(mode="body", pattern="terrain"),
  #   fields=("found", "force"),
  #   reduce="none",
  #   num_slots=1,
  #   history_length=4,
  # )
  

  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
    thigh_ground_cfg,
    shank_ground_cfg,
    # trunk_head_ground_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = GO2_ACTION_SCALE

  cfg.viewer.body_name = "base_link"
  cfg.viewer.distance = 1.5
  cfg.viewer.elevation = -10.0

  cfg.rewards.pop("track_base_height",None)
  cfg.rewards["upright"].params["asset_cfg"].body_names = ("base_link",)

  cfg.rewards["angular_momentum"].weight = 0.0
  cfg.rewards["air_time"].weight = 0.0

  # Per-body-group collision penalties.
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-0.1,
    params={"sensor_name": self_collision_cfg.name},
  )
  cfg.rewards["shank_collision"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-0.1,
    params={"sensor_name": shank_ground_cfg.name},
  )
  # cfg.rewards["trunk_head_collision"] = RewardTermCfg(
  #   func=mdp.self_collision_cost,
  #   weight=-0.1,
  #   params={"sensor_name": trunk_head_ground_cfg.name},
  # )

  # On rough terrain the quadruped tilts significantly; don't terminate on
  # orientation alone. Let out_of_terrain_bounds handle resets.
  cfg.terminations.pop("fell_over", None)

  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": thigh_ground_cfg.name},
  )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    # cfg.episode_length_s = int(1e9)
    cfg.scene.num_envs = 1
    cfg.observations["actor"].enable_corruption = False

    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum = {}


    if "Goal_direction" in cfg.commands:
      cfg.commands["Goal_direction"].debug_vis = True

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg

def my_go2_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go2 flat terrain tracking configuration."""
  cfg = my_go2_stair_env_cfg(play=play)
  cfg.observations["actor"].terms["stair_boolean"].params["is_stair"] = False
  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # 1. 切换为平地地形
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # 2. 移除平地不需要的传感器与特权高度扫描
  remove_sensors = {
    "terrain_scan",
    # "self_collision",
    # "thigh_ground_touch",
    # "shank_ground_touch",
    # "trunk_ground_touch",
  }

  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name not in remove_sensors
  )

  cfg.rewards["upright"].params.pop("terrain_sensor_names", None)
  cfg.observations["critic"].terms.pop("height_scan", None)

  # 3. 碰撞惩罚：平地保留 shank_collision 避免小腿触地拖拽
  for key in ("self_collisions", "trunk_head_collision"):
    cfg.rewards.pop(key, None)
  cfg.rewards["shank_collision"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-0.1,
    params={"sensor_name": "shank_ground_touch"},
  )

  # 4. 终止条件：平地加入 fall_over，大腿触地 illegal_contact，以及最低高度 base_too_low（低于 0.20m 即重置）
  cfg.terminations.pop("out_of_terrain_bounds", None)
  cfg.terminations["fell_over"] = TerminationTermCfg(
    func=mdp.bad_orientation,
    params={"limit_angle": math.radians(70.0)},
  )


  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": "thigh_ground_touch"},
  )
  # 小腿 (shank) 触地 = 拖腿行走。正常步态下小腿永远不会着地，
  # 一旦检测到就终止，直接封死"三条腿擦地"这种退化解。
  # cfg.terminations["shank_contact"] = TerminationTermCfg(
  #   func=mdp.illegal_contact,
  #   params={"sensor_name": "shank_ground_touch", "force_threshold": 10.0},
  # )
  cfg.terminations["base_too_low"] = TerminationTermCfg(
    func=mdp.root_height_below_minimum,
    params={"minimum_height": 0.20},
  )

  cfg.terminations["out_of_patch_bounds"] = TerminationTermCfg(
    func=out_of_patch_bounds,
    time_out=True,
    params={"half_size": 6.0, "margin": 0.2},
  )
  
  # 5. 平地步态与速度约束（目标速度设为 0.8 m/s，提升脚掌离地与站姿高度稳定性）

  cfg.rewards["track_target_bias"].params["speed_std"] = 0.25
  cfg.rewards["air_time"].weight = 0.05
  cfg.rewards["pose"].weight = 1.0
  cfg.rewards["pose"].params["std"] = {
    r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.45,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.75,
  }


  # 5. 平地课程：到达多次目标点后升级（提升目标距离）
  cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
    func=terrain_levels_reach_target,
    params={
      "command_name": "Goal_direction",
      "distance_threshold": 0.05,
      "required_reach_count": 3,
      "increase_distance_on_flat": True,
      "distance_step": 0.5,
      "max_distance": 5.0,
      "min_distance": 2.0,
    },
  )

  # 6. 设置平地任务标志位（is_stair = False）
  if "stair_boolean" in cfg.observations["actor"].terms:
    cfg.observations["actor"].terms["stair_boolean"].params["is_stair"] = False

  # 7. 修改 command 参数（保持各项 command 不变，仅修改参数）
  cmd_goal = cfg.commands["Goal_direction"]
  assert isinstance(cmd_goal, GoalDirectionCommandCfg)
  cmd_goal.target_distance = 4.0  # 平地目标距离增大为 4.0m
  cmd_goal.resampling_time_range = (10.0, 10.0)

  cmd_height = cfg.commands["Base_height_target"]
  assert isinstance(cmd_height, HeightTargetCommandCfg)
  cmd_height.nominal_height = 0.28
  cmd_height.is_stair = False
  cmd_height.height_smooth_alpha = 0.05

  cmd_head = cfg.commands["Heading_error"]
  assert isinstance(cmd_head, HeadErrorCommandCfg)
  cmd_head.resampling_time_range = (10.0, 10.0)

  if play:
    cfg.scene.num_envs = 1

  return cfg

def my_go2_stair_env_cfg(
    play: bool = False,) -> ManagerBasedRlEnvCfg:
  cfg = my_go2_rough_env_cfg(play=play)
  cfg.observations["actor"].terms["stair_boolean"].params["is_stair"] = True
  cfg.observations["critic"].terms.pop("height_scan", None)
  remove_sensors = {"terrain_scan"}
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name not in remove_sensors
  )
  cfg.scene.terrain = TerrainEntityCfg(
            terrain_type="generator",
            terrain_generator=replace(STAIRS_TERRAINS_CFG),
            max_init_terrain_level=0,
        )

  for key in ("self_collisions", "trunk_head_collision"):
    cfg.rewards.pop(key, None)

  cfg.rewards.pop("track_base_height", None)
  
  cfg.rewards["upright"].params.pop("terrain_sensor_names", None)
  
  # 1. 碰撞与步态约束（同步平地有效参数）
  cfg.rewards["shank_collision"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-0.1,
    params={"sensor_name": "shank_ground_touch"},
  )
  cfg.rewards["track_target_bias"].params["speed_std"] = 0.25
  cfg.rewards["air_time"].weight = 0.10
  cfg.rewards["pose"].weight = 0.5
  cfg.rewards["pose"].params["std"] = {
    r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.75,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.75,
  }

  # 2. 终止条件：大腿触地 illegal_contact，以及摔倒翻车限制 fell_over (70度)
  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": "thigh_ground_touch"},
  )
  # cfg.terminations["fell_over"] = TerminationTermCfg(
  #   func=mdp.bad_orientation,
  #   params={"limit_angle": math.radians(70.0)},
  # )

  # 楼梯地形 border_width=20m，内置 out_of_terrain_bounds 边界太大不会触发；
  # 用 out_of_patch_bounds 代替：机器人离开自己的 8×8 patch 即终止（time_out）
  cfg.terminations.pop("out_of_terrain_bounds", None)
  cfg.terminations.pop("reached_edge", None)
  cfg.terminations.pop("fell_over",None)
  cfg.terminations["out_of_patch_bounds"] = TerminationTermCfg(
    func=out_of_patch_bounds,
    time_out=True,
    params={"half_size": 4.5, "margin": 0.2},  # limit=4.3m，为 3.8m 目标点预留 0.5m 缓冲
  )
  cfg.events = {
      "reset_base": EventTermCfg(
        func=reset_root_state_from_stair,
        mode="reset",
        params={
          "pose_range": {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),          # 倒金字塔中心凹底平地 (3m×3m 范围中央)
            "z": (0.0, 0.0),
            "yaw": (-3.14, 3.14),      # 凹底随机朝向四周
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

  
  if play:
    cfg.scene.num_envs = 1
    cfg.scene.terrain.max_init_terrain_level = 5

  # 3. 楼梯任务指令参数平滑与重采样
  cmd_height = cfg.commands["Base_height_target"]
  assert isinstance(cmd_height, HeightTargetCommandCfg)
  cmd_height.nominal_height = 0.28
  cmd_height.is_stair = True
  cmd_height.height_smooth_alpha = 0.05

  cmd_head = cfg.commands["Heading_error"]
  assert isinstance(cmd_head, HeadErrorCommandCfg)
  cmd_head.resampling_time_range = (10.0, 10.0)

  cmd_goal = cfg.commands["Goal_direction"]
  assert isinstance(cmd_goal, GoalDirectionCommandCfg)
  cmd_goal.target_distance = 3.8  # 目标点外移至最高台阶外沿后方 0.8m，确保四足完全登顶平地
  cmd_goal.resampling_time_range = (10.0, 10.0)

  return cfg

def my_go2_tire_env_cfg(
    play: bool = False,) -> ManagerBasedRlEnvCfg:
  cfg = my_go2_rough_env_cfg(play=play)

  cfg.scene.terrain = TerrainEntityCfg(
      terrain_type="generator",
      terrain_generator=replace(TIRE_ARRAY_TERRAINS_CFG),
      max_init_terrain_level=0,
  )

  for key in ("self_collisions", "shank_collision", "trunk_head_collision"):
    cfg.rewards.pop(key, None)

  cfg.terminations.pop("fell_over", None)

  # 8×8 patch 边界检测
  cfg.terminations.pop("out_of_terrain_bounds", None)
  cfg.terminations.pop("reached_edge", None)
  cfg.terminations["out_of_patch_bounds"] = TerminationTermCfg(
    func=out_of_patch_bounds,
    time_out=True,
    params={"half_size": 4.0, "margin": 0.1},
  )
  
  cfg.commands["twist"] = TireTraverseVelocityCommandCfg(
      entity_name="robot",
      forward_speed=1.0,
      target_y=-3.0,
      resampling_time_range=(20.0, 20.0),
      debug_vis=True,
  )

  cfg.events = {
      "reset_base": EventTermCfg(
          func=reset_root_state_from_tire,
          mode="reset",
          params={
              "pose_range": {
                  "x": (-0.5, 0.5),
                  "y": (2.5, 3.2),           # 出生在平坦起跑平台
                  "z": (0.0, 0.0),
                  "yaw": (-1.57, -1.57),     # 面朝轮胎阵 (-Y 方向)
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

  if "stair_boolean" in cfg.observations["actor"].terms:
    cfg.observations["actor"].terms["stair_boolean"].params["is_stair"] = False

  if play:
    cfg.scene.terrain.max_init_terrain_level = 5

  return cfg

def my_go2_platform_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = my_go2_rough_env_cfg(play=play)
  cfg.observations["actor"].terms["stair_boolean"].params["is_stair"] = True
  cfg.observations["critic"].terms.pop("height_scan", None)
  cfg.scene.terrain = TerrainEntityCfg(
            terrain_type="generator",
            terrain_generator=replace(STEP_PLATFORM_TERRAINS_CFG),
            max_init_terrain_level=0,
        )
  
  for key in ("self_collisions", "trunk_head_collision"):
    cfg.rewards.pop(key, None)
  
  cfg.rewards.pop("track_base_height", None)
  
  cfg.rewards["upright"].params.pop("terrain_sensor_names", None)
  
  # 1. 碰撞与步态约束（同步平地有效参数）
  cfg.rewards["shank_collision"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-0.1,
    params={"sensor_name": "shank_ground_touch"},
  )
  cfg.rewards["track_target_bias"].params["speed_std"] = 0.25
  cfg.rewards["air_time"].weight = 1.0
  cfg.rewards["pose"].weight = 1.0
  cfg.rewards["pose"].params["std"] = {
    r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.45,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.75,
  }
  
  # 2. 终止条件：大腿触地 illegal_contact，以及摔倒翻车限制 fell_over (70度)
  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": "thigh_ground_touch"},
  )
  cfg.terminations["fell_over"] = TerminationTermCfg(
    func=mdp.bad_orientation,
    params={"limit_angle": math.radians(70.0)},
  )
  
  # 楼梯地形 border_width=20m，内置 out_of_terrain_bounds 边界太大不会触发；
  # 用 out_of_patch_bounds 代替：机器人离开自己的 8×8 patch 即终止（time_out）
  cfg.terminations.pop("out_of_terrain_bounds", None)
  cfg.terminations.pop("reached_edge", None)
  cfg.terminations["out_of_patch_bounds"] = TerminationTermCfg(
    func=out_of_patch_bounds,
    time_out=True,
    params={"half_size": 4.0, "margin": 0.1},  # patch 8×8, limit=3.9m  )
  )
  cfg.events = {
      "reset_base": EventTermCfg(
        func=reset_root_state_from_stair,
        mode="reset",
        params={
          "pose_range": {
            "x": (-0.1, 0.1),
            "y": (-0.2, 0.2),
            "z": (0.0, 0.0),
            "yaw": (-0.1, 0.1),          # 面朝台阶方向 (+X)
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
  
  
  if play:
    cfg.scene.num_envs = 1
    cfg.scene.terrain.max_init_terrain_level = 5
  
  # 3. 楼梯任务指令参数平滑与重采样
  cmd_height = cfg.commands["Base_height_target"]
  assert isinstance(cmd_height, HeightTargetCommandCfg)
  cmd_height.nominal_height = 0.28
  cmd_height.is_stair = True
  cmd_height.height_smooth_alpha = 0.05
  
  cmd_head = cfg.commands["Heading_error"]
  assert isinstance(cmd_head, HeadErrorCommandCfg)
  cmd_head.resampling_time_range = (10.0, 10.0)
  
  cmd_goal = cfg.commands["Goal_direction"]
  assert isinstance(cmd_goal, GoalDirectionCommandCfg)
  cmd_goal.cardinal_directions = ("E",)
  # 出生点在第一平台顶面中心 (见 STEP_PLATFORM_TERRAINS_CFG)，目标点取第二平台中心：
  # 两平台中心距 = platform_length/2 + gap + second_platform_length/2 = 0.8 + gap，
  # gap 随课程在 0.2~0.6m 变化 → 中心距 1.0~1.4m。取 1.2m 时目标点始终落在
  # 第二平台内 (距中心 ±0.2m，平台半长 0.4m)。
  cmd_goal.target_distance = 1.2
  cmd_goal.resampling_time_range = (10.0, 10.0)
  

  
  return cfg





