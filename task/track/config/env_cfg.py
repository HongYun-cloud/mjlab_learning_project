import math
import os
from typing import Literal
from dataclasses import replace

import torch

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
from myrobot_mjlab.config.terrain.stones_2rows_terrain import STONES_2ROWS_TERRAINS_CFG
from myrobot_mjlab.config.terrain.plum_blossom_field_terrain import (
    PLUM_BLOSSOM_FIELD_TERRAINS_CFG,
)
from myrobot_mjlab.config.terrain.stones_2rows_terrain import (
    STONES_2ROWS_TERRAINS_CFG,
)
from myrobot_mjlab.config.curriculum.stair_curriculum import terrain_levels_reach_target, anneal_reward_weight
from myrobot_mjlab.config.event.stair_reset import reset_root_state_from_stair,out_of_patch_bounds
from myrobot_mjlab.config.event.tire_reset import reset_root_state_from_tire
from myrobot_mjlab.config.event.plum_reset import randomize_plum_target
from myrobot_mjlab.config.event.stones_reset import (
  base_below_height,
  randomize_stones_target,
)
from myrobot_mjlab.config.reward.Trackingreward import goal_forward_bonus
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

  # 高度奖励保留在配置中：track_base_height 跟踪 Base_height_target 指令。
  if "upright" in cfg.rewards:
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("base_link",)

  cfg.rewards["angular_momentum"].weight = 0.0
  if "air_time" in cfg.rewards:
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
  # orientation alone.
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
  cfg = my_go2_rough_env_cfg(play=play)
  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # 1. 切换为平地地形
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # 2. Actor 和 Critic 都使用 height_scan，保留 terrain_scan 传感器。
  if "upright" in cfg.rewards:
    cfg.rewards["upright"].params.pop("terrain_sensor_names", None)

  # 3. 碰撞惩罚：平地保留 shank_collision 避免小腿触地拖拽
  for key in ("self_collisions", "trunk_head_collision"):
    cfg.rewards.pop(key, None)
  cfg.rewards["shank_collision"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-0.1,
    params={"sensor_name": "shank_ground_touch"},
  )

  # 4. 终止条件：平地加入 fell_over、大腿触地 illegal_contact。
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

  cfg.terminations["out_of_patch_bounds"] = TerminationTermCfg(
    func=out_of_patch_bounds,
    time_out=True,
    params={"half_size": 6.0, "margin": 0.2},
  )
  
  # 5. 平地步态与速度约束（目标速度设为 0.8 m/s，提升脚掌离地与站姿高度稳定性）

  cfg.rewards["track_target_bias"].params["speed_std"] = 0.25
  if "air_time" in cfg.rewards:
    cfg.rewards["air_time"].weight = 0.05
  if "pose" in cfg.rewards:
    cfg.rewards["pose"].weight = 1.0
    cfg.rewards["pose"].params["std"] = {
      r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.45,
      r".*(FR|FL|RR|RL)_calf_joint.*": 0.75,
    }

  # 5. 平地课程：到达多次目标点后升级（提升目标距离）
  curriculum_mode = os.getenv(
    "MYROBOT_CURRICULUM_MODE", "curriculum"
  ).strip().lower()
  cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
    func=terrain_levels_reach_target,
    params={
      "command_name": "Goal_direction",
      "distance_threshold": 0.05,
      "required_reach_count": 3,
      "direct_random": curriculum_mode in ("random", "direct", "1", "true"),
      "increase_distance_on_flat": True,
      "distance_step": 0.5,
      "max_distance": 5.0,
      "min_distance": 2.0,
    },
  )

  # 6. 设置地形选项（平地）
  if "terrain_type" in cfg.observations["actor"].terms:
    cfg.observations["actor"].terms["terrain_type"].params["terrain_type"] = "flat"

  # 7. 修改 command 参数（保持各项 command 不变，仅修改参数）
  cmd_goal = cfg.commands["Goal_direction"]
  assert isinstance(cmd_goal, GoalDirectionCommandCfg)
  cmd_goal.target_distance = 4.0  # 平地目标距离增大为 4.0m
  cmd_goal.resampling_time_range = (10.0, 10.0)

  cmd_height = cfg.commands["Base_height_target"]
  assert isinstance(cmd_height, HeightTargetCommandCfg)
  cmd_height.nominal_height = 0.25
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
  cfg.observations["actor"].terms["terrain_type"].params["terrain_type"] = "stairs"
  cfg.scene.terrain = TerrainEntityCfg(
            terrain_type="generator",
            terrain_generator=replace(STAIRS_TERRAINS_CFG),
            max_init_terrain_level=0,
        )

  # 头部/躯干与地形的接触传感器：只用于终止，不进奖励。
  trunk_head_ground_cfg = ContactSensorCfg(
    name="trunk_head_ground_touch",
    primary=ContactMatch(mode="body", pattern="base_link", entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = tuple(cfg.scene.sensors or ()) + (trunk_head_ground_cfg,)

  for key in ("self_collisions", "trunk_head_collision","air_time"):
    cfg.rewards.pop(key, None)
  
  # 1. 碰撞与步态约束（同步平地有效参数）
  cfg.rewards["shank_collision"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-0.1,
    params={"sensor_name": "shank_ground_touch"},
  )


  # 2. 终止条件：大腿触地 illegal_contact，以及摔倒翻车限制 fell_over (70度)
  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": "thigh_ground_touch"},
  )
  cfg.terminations["trunk_head_collision"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": "trunk_head_ground_touch"},
  )

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

  
  # 目标点向外移到最高台阶外沿附近，避免落在楼梯内侧。
  cmd_goal = cfg.commands["Goal_direction"]
  assert isinstance(cmd_goal, GoalDirectionCommandCfg)
  cmd_goal.target_distance = 3.8

  # Stair 不使用 track_target_bias 的终点死区。
  cfg.rewards["track_target_bias"].params["goal_deadzone"] = 0.0

  # Stair 使用 15s episode；奖励时间窗按 15/6 = 2.5 倍等比例拉长。
  cfg.episode_length_s = 15.0
  cfg.rewards["track_target_position"].weight = 2.0  # (1 / t_r) * 10, t_r = 5
  cfg.rewards["track_target_position"].params["t_r"] = 5.0
  cfg.rewards["track_target_bias"].params["t_r"] = 10.0
  cfg.rewards["standstill"].params["t_r"] = 5.0
  cfg.rewards["stand_pose"].params["t_r"] = 5.0

  if play:
    cfg.scene.num_envs = 1
    cfg.scene.terrain.max_init_terrain_level = 5

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

  # 目标点放到轮胎阵对面（-Y 方向，即机器人出生平台的正前方）。
  # 与 twist 的 target_y=-3.0 对齐：相对 env origin 的局部坐标 y=-3.0，
  # 对应轮胎阵远端边缘外的平坦区域。
  cmd_goal = cfg.commands["Goal_direction"]
  assert isinstance(cmd_goal, GoalDirectionCommandCfg)
  cmd_goal.cardinal_directions = ("S",)
  cmd_goal.target_distance = 3.0
  cmd_goal.resampling_time_range = (20.0, 20.0)

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

  if "terrain_type" in cfg.observations["actor"].terms:
    cfg.observations["actor"].terms["terrain_type"].params["terrain_type"] = "tire_array"

  if play:
    # 单环境 + 关闭观测扰动与课程。
    cfg.scene.num_envs = 1
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}

    # 注意：上面重新赋值了 cfg.scene.terrain，覆盖掉了 rough play 中对
    # 原地形生成器做的 play 设置，因此这里需要重新应用。
    # 轮胎地形由大量 torus mesh 组成，patch 越多 Viser 首次加载越慢；
    # play 只保留单列 5 个难度等级，训练时仍使用完整的 10 个等级。
    cfg.scene.terrain.max_init_terrain_level = 4
    terrain_generator = cfg.scene.terrain.terrain_generator
    if terrain_generator is not None:
      terrain_generator.curriculum = True
      terrain_generator.num_cols = 1
      terrain_generator.num_rows = 5
      terrain_generator.border_width = 20.0

    # Play 可视化：显示轮胎阵穿越速度指令、目标方向与高度指令。
    if "twist" in cfg.commands:
      cfg.commands["twist"].debug_vis = True
    if "Goal_direction" in cfg.commands:
      cfg.commands["Goal_direction"].debug_vis = True
    if "Base_height_target" in cfg.commands:
      cfg.commands["Base_height_target"].debug_vis = True

  return cfg

def _schedule_tensor(
  schedule: tuple[float, ...], difficulty: torch.Tensor
) -> torch.Tensor:
  """Map per-env difficulty in [0, 1] to schedule entries."""
  values = torch.tensor(schedule, device=difficulty.device, dtype=difficulty.dtype)
  if values.numel() == 1:
    return values[0].expand_as(difficulty)
  index = torch.round(
    difficulty.clamp(0.0, 1.0) * (values.numel() - 1)
  ).long()
  return values[index]


def update_platform_goal_distance(
  env,
  env_ids,
  command_name: str = "Goal_direction",
) -> None:
  """根据当前地形难度更新 Platform 任务的目标距离。

  中间小平台的间距随 difficulty 变化，因此第二平台中心到出生点的距离
  也会变化；这里在 reset 时把每个 env 的目标距离写入 Goal_direction 的
  ``target_distances``，命令项随后会使用它重新采样目标点。
  """
  if len(env_ids) == 0:
    return

  terrain = getattr(env.scene, "terrain", None)
  if terrain is None or getattr(terrain, "terrain_origins", None) is None:
    return

  generator_cfg = getattr(terrain.cfg, "terrain_generator", None)
  if generator_cfg is None:
    return

  sub = next(
    (
      cfg
      for cfg in generator_cfg.sub_terrains.values()
      if hasattr(cfg, "num_mid_platforms")
    ),
    None,
  )
  if sub is None:
    return

  num_rows = terrain.terrain_origins.shape[0]
  difficulty = terrain.terrain_levels[env_ids].float() / max(num_rows - 1, 1)

  if getattr(sub, "gap_distance_schedule", None) is not None:
    spacing = _schedule_tensor(sub.gap_distance_schedule, difficulty)
  elif sub.gap_distance is not None:
    spacing = torch.full_like(difficulty, float(sub.gap_distance))
  elif sub.vary_gap_with_difficulty:
    gap_min, gap_max = sub.gap_distance_range
    spacing = gap_min + difficulty * (gap_max - gap_min)
  else:
    spacing = torch.full_like(difficulty, float(sub.gap_distance_range[0]))

  if getattr(sub, "mid_platform_length_schedule", None) is not None:
    mid_platform_length = _schedule_tensor(
      sub.mid_platform_length_schedule, difficulty
    )
  else:
    mid_platform_length = torch.full_like(
      difficulty, float(sub.mid_platform_length)
    )

  total_gap = (
    (sub.num_mid_platforms + 1) * spacing
    + sub.num_mid_platforms * mid_platform_length
  )
  up_stairs_length = sub.num_steps * sub.step_width
  if sub.spawn_on_platform:
    start_distance = sub.platform_length / 2.0
  else:
    start_distance = sub.spawn_distance_before + up_stairs_length + sub.platform_length
  distance = (
    start_distance
    + total_gap
    + sub.second_platform_length / 2.0
  )

  if command_name not in env.command_manager.active_terms:
    return
  command = env.command_manager.get_term(command_name)
  if not hasattr(command, "target_distances"):
    command.target_distances = torch.full(
      (env.num_envs,),
      float(command.cfg.target_distance),
      device=env.device,
      dtype=torch.float32,
    )
  command.target_distances[env_ids] = distance.to(
    device=env.device, dtype=torch.float32
  )


def _apply_platform_like_config(
  cfg: ManagerBasedRlEnvCfg,
  *,
  out_of_patch_params: dict[str, float],
) -> None:
  """Apply the Platform-style rewards, terminations, and episode settings."""
  # 头部/躯干与地形的接触传感器：只用于终止，不进奖励。
  trunk_head_ground_cfg = ContactSensorCfg(
    name="trunk_head_ground_touch",
    primary=ContactMatch(mode="body", pattern="base_link", entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = tuple(cfg.scene.sensors or ()) + (trunk_head_ground_cfg,)

  for key in ("self_collisions", "trunk_head_collision", "air_time"):
    cfg.rewards.pop(key, None)

  cfg.rewards["shank_collision"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-0.1,
    params={"sensor_name": "shank_ground_touch"},
  )

  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": "thigh_ground_touch"},
  )
  cfg.terminations["trunk_head_collision"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": "trunk_head_ground_touch"},
  )

  cfg.terminations.pop("reached_edge", None)
  cfg.terminations.pop("fell_over", None)
  cfg.terminations["out_of_patch_bounds"] = TerminationTermCfg(
    func=out_of_patch_bounds,
    time_out=True,
    params=out_of_patch_params,
  )

  # 与 Platform/Stair 对齐的 episode 和奖励时间窗。
  cfg.rewards["track_target_bias"].params["goal_deadzone"] = 0.0
  cfg.episode_length_s = 15.0
  cfg.rewards["track_target_position"].weight = 2.0
  cfg.rewards["track_target_position"].params["t_r"] = 5.0
  cfg.rewards["track_target_bias"].params["t_r"] = 10.0
  cfg.rewards["standstill"].params["t_r"] = 5.0
  cfg.rewards["stand_pose"].params["t_r"] = 5.0

  # Platform 的额外权重调整。
  cfg.rewards["torque_penalty"].weight = -1.0e-4
  cfg.rewards["aggressive_motion"].weight = -4.0


def my_go2_platform_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = my_go2_rough_env_cfg(play=play)
  cfg.observations["actor"].terms["terrain_type"].params["terrain_type"] = "platform"

  cfg.scene.terrain = TerrainEntityCfg(
            terrain_type="generator",
            terrain_generator=replace(STEP_PLATFORM_TERRAINS_CFG),
            max_init_terrain_level=0,
        )

  # Platform 的目标点在第二平台中心，方向必须沿地形行进方向固定，
  # 不能像 Stair 一样在四个基本方位里随机采样。
  platform_sub = next(
    sub
    for sub in cfg.scene.terrain.terrain_generator.sub_terrains.values()
    if hasattr(sub, "num_mid_platforms")
  )
  direction_to_cardinal = {"+x": "E", "-x": "W", "+y": "N", "-y": "S"}
  cfg.commands["Goal_direction"].cardinal_directions = (
    direction_to_cardinal[platform_sub.direction],
  )
  _apply_platform_like_config(
    cfg,
    out_of_patch_params={"half_size_x": 8.0, "half_size_y": 4.0, "margin": 0.2},
  )
  cfg.events = {
      "reset_base": EventTermCfg(
        func=reset_root_state_from_stair,
        mode="reset",
        params={
          "pose_range": {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "z": (0.0, 0.0),
            "yaw": (-3.14, 3.14),        # 与 Stair 一致：随机朝向
          },
            "velocity_range": {},
            "nominal_height": 0.28,
          },
      ),
      "update_goal_distance": EventTermCfg(
        func=update_platform_goal_distance,
        mode="reset",
        params={"command_name": "Goal_direction"},
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

  # Platform 课程划分为 level 0~12 共 13 个难度点。
  # 到 level 12 后不再继续加难，reset 时在 0~12 之间随机。
  if "terrain_levels" in cfg.curriculum:
    cfg.curriculum["terrain_levels"].params["max_level"] = 12
    cfg.curriculum["terrain_levels"].params["randomize_at_max"] = True

  
  if play:
    cfg.scene.num_envs = 1
    cfg.scene.terrain.max_init_terrain_level = 5

  
  return cfg


def my_go2_plum_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Track task on the plum blossom field terrain.

  - 从 ``my_go2_rough_env_cfg`` 继承；
  - 奖励、终止、episode、传感器使用 Platform 的基本配置；
  - 地形使用 15m 梅花桩区域 + 2m 外侧平地；
  - 出生在中心平台，目标在 +X 方向外侧平地的中心。
  """
  cfg = my_go2_rough_env_cfg(play=play)
  cfg.observations["actor"].terms["terrain_type"].params["terrain_type"] = "single beam"

  # 梅花桩地形有很多小桩，接触数量比普通地形大。
  cfg.sim.njmax = 300
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = 128

  cfg.scene.terrain = TerrainEntityCfg(
    terrain_type="generator",
    terrain_generator=replace(PLUM_BLOSSOM_FIELD_TERRAINS_CFG),
    max_init_terrain_level=0,
  )

  # 1. 碰撞与步态约束（同步平地有效参数）
  cfg.rewards["shank_collision"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-0.1,
    params={"sensor_name": "shank_ground_touch"},
  )
  # 2. 终止条件：大腿触地 illegal_contact，以及摔倒翻车限制 fell_over (70度)
  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": "thigh_ground_touch"},
  )
  cfg.terminations["trunk_head_collision"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": "trunk_head_ground_touch"},
  )
  plum_sub = next(
    iter(cfg.scene.terrain.terrain_generator.sub_terrains.values())
  )
  # 中心平台 -> 外侧平地中心的距离。
  goal_distance = (
    float(plum_sub.field_size[0]) / 2.0
    + float(plum_sub.flat_border_width) / 2.0
  )
  cfg.commands["Goal_direction"].cardinal_directions = ("E",)
  cfg.commands["Goal_direction"].target_distance = goal_distance

  _apply_platform_like_config(
    cfg,
    out_of_patch_params={
      "half_size": float(PLUM_BLOSSOM_FIELD_TERRAINS_CFG.size[0]) / 2.0,
      "margin": 0.2,
    },
  )

  # 姿态惩罚全程生效，不再只在终点附近。
  cfg.rewards["normal_pose"].params["distance_threshold"] = None

  # Plum 专项：终止惩罚和前期前进引导。
  cfg.rewards["termination_penalty"].weight = -200.0
  cfg.rewards["track_target_position"].dense = True
  cfg.rewards["goal_forward_bonus"] = RewardTermCfg(
    func=goal_forward_bonus,
    weight=1.5,
    params={
      "command_name": "Goal_direction",
      "min_forward_speed": 0.05,
    },
  )

  cfg.events = {
    "reset_base": EventTermCfg(
      func=reset_root_state_from_stair,
      mode="reset",
      params={
        "pose_range": {
          "x": (-0.1, 0.1),
          "y": (-0.1, 0.1),
          "z": (0.0, 0.0),
          "yaw": (-0.1, 0.1),
        },
        "velocity_range": {},
        "nominal_height": 0.28,
      },
    ),
    "randomize_goal": EventTermCfg(
      func=randomize_plum_target,
      mode="reset",
      params={
        "command_name": "Goal_direction",
        "inner_margin": 0.3,
        "inner_band": 0.7,
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

  # 梅花桩地形共 13 个难度（0~12）。
  if "terrain_levels" in cfg.curriculum:
    cfg.curriculum["terrain_levels"].params["max_level"] = 12
    cfg.curriculum["terrain_levels"].params["randomize_at_max"] = True

  # 前进奖励从 1.5 线性退火到 1e-3，覆盖前 1000 个 iteration。
  # 1000 iter × 24 steps = 24000 control steps。
  if not play:
    cfg.curriculum["goal_forward_anneal"] = CurriculumTermCfg(
      func=anneal_reward_weight,
      params={
        "reward_name": "goal_forward_bonus",
        "total_steps": 1000 * 24,
        "schedule": ((0.0, 1.5), (1.0, 1.0e-3)),
      },
    )

  if play:
    cfg.scene.num_envs = 1
    cfg.scene.terrain.max_init_terrain_level = 12

  return cfg

def my_go2_stones_2rows_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Track task on the Stones-2Rows terrain.

  - 中心平台 + 四个方向的双排交错梅花桩
  - 目标点固定在 E/W/N/S 四个外侧平台中心，每次 reset 随机选一个
  - 奖励/终止使用 Platform/Stair 的基本配置；前期保留前进奖励
  - 机身世界 z < -3m（掉出世界）时终止
  """
  cfg = my_go2_rough_env_cfg(play=play)
  cfg.observations["actor"].terms["terrain_type"].params["terrain_type"] = "single beam"

  cfg.sim.njmax = 300
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = 128

  cfg.scene.terrain = TerrainEntityCfg(
    terrain_type="generator",
    terrain_generator=replace(STONES_2ROWS_TERRAINS_CFG),
    max_init_terrain_level=0,
  )

  stones_sub = next(
    iter(cfg.scene.terrain.terrain_generator.sub_terrains.values())
  )
  goal_distance = float(stones_sub._target_distance(0))
  cfg.commands["Goal_direction"].cardinal_directions = ("E",)
  cfg.commands["Goal_direction"].target_distance = goal_distance

  _apply_platform_like_config(
    cfg,
    out_of_patch_params={
      "half_size": float(STONES_2ROWS_TERRAINS_CFG.size[0]) / 2.0,
      "margin": 0.2,
    },
  )

  # 掉出世界：机身世界 z < -3m 时终止（只用于 Stones-2Rows）。
  cfg.terminations["base_below_world"] = TerminationTermCfg(
    func=base_below_height,
    params={"min_height": -3.0},
  )

  # 姿态惩罚全程生效，不再只在终点附近。
  cfg.rewards["normal_pose"].params["distance_threshold"] = None

  # 奖励配置与 Plum 的实际行为对齐：
  # aggressive_motion / torque_penalty 沿用 helper 的 Platform 默认值。
  # Plum 的 track_target_position.dense=True 写在 cfg 属性上，函数实际读
  # params["dense"]；这里用 params 显式打开 dense。
  cfg.rewards["track_target_position"].params["dense"] = True
  cfg.rewards["goal_forward_bonus"] = RewardTermCfg(
    func=goal_forward_bonus,
    weight=2.5,
    params={
      "command_name": "Goal_direction",
      "min_forward_speed": 0.05,
    },
  )
  cfg.rewards["termination_penalty"].weight = -100.0

  cfg.events = {
    "reset_base": EventTermCfg(
      func=reset_root_state_from_stair,
      mode="reset",
      params={
        "pose_range": {
          "x": (-0.1, 0.1),
          "y": (-0.1, 0.1),
          "z": (0.0, 0.0),
          "yaw": (-3.14, 3.14),
        },
        "velocity_range": {},
        "nominal_height": 0.28,
      },
    ),
    "randomize_goal": EventTermCfg(
      func=randomize_stones_target,
      mode="reset",
      params={"command_name": "Goal_direction"},
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

  # Stones-2Rows 共 13 个难度（0~12）。
  if "terrain_levels" in cfg.curriculum:
    cfg.curriculum["terrain_levels"].params["max_level"] = 12
    cfg.curriculum["terrain_levels"].params["randomize_at_max"] = True

  if not play:
    cfg.curriculum["goal_forward_anneal"] = CurriculumTermCfg(
      func=anneal_reward_weight,
      params={
        "reward_name": "goal_forward_bonus",
        "total_steps": 1000 * 24,
        "schedule": ((0.0, 1.5), (1.0, 1.0e-3)),
      },
    )

  if play:
    cfg.scene.num_envs = 1
    cfg.scene.terrain.max_init_terrain_level = 12

  return cfg
