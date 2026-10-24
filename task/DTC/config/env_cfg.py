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
from myrobot_mjlab.task.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.scene import SceneCfg
from mjlab.terrains import TerrainEntityCfg
from myrobot_mjlab.config.terrain.stair_terrain import STAIRS_TERRAINS_CFG
from myrobot_mjlab.config.terrain.tire_array import TIRE_ARRAY_TERRAINS_CFG
from myrobot_mjlab.config.event.stair_reset import reset_root_state_from_stair,out_of_patch_bounds
from myrobot_mjlab.config.event.tire_reset import reset_root_state_from_tire
from myrobot_mjlab.config.reward.stair_reward import track_linear_velocity_stair
from myrobot_mjlab.config.command.StairVelocityCommand import StairVelocityCommandCfg
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


  cfg.rewards["pose"].params["std_standing"] = {
    r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.05,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.1,
  }
  cfg.rewards["pose"].params["std_walking"] = {
    r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.3,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.6,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.3,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.6,
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base_link",)

#   for reward_name in ["foot_clearance", "foot_slip"]:
#     cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  cfg.rewards["body_ang_vel"].weight = 0.0
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
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False

    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum = {}


    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg

def my_go2_stair_env_cfg(
    play: bool = False,) -> ManagerBasedRlEnvCfg:
  cfg = my_go2_rough_env_cfg(play=play)

  cfg.scene.terrain = TerrainEntityCfg(
            terrain_type="generator",
            terrain_generator=replace(STAIRS_TERRAINS_CFG),
            max_init_terrain_level=0,
        )

  for key in ("self_collisions", "shank_collision", "trunk_head_collision"):
    cfg.rewards.pop(key, None)

  # cfg.rewards.pop("track_linear_velocity",None)
  # # 注意 weight 必须为正：函数返回 exp(-e/std^2) - 1.0，跟踪越差值越负。
  # # 若 weight 为负，"站着不动"反而拿到正奖励（-1.0 * -1.0 = +1.0）。
  # cfg.rewards["track_linear_velocity"]=RewardTermCfg(
  #       func=track_linear_velocity_stair,
  #       weight=0.5,
  #       params={"command_name": "twist", "std": math.sqrt(0.25)},
  # )
  
  cfg.terminations["fell_over"] = TerminationTermCfg(
    func=mdp.bad_orientation,
    params={"limit_angle": math.radians(70.0)},
  )
  # cfg.terminations.pop("illegal_contact", None)

  # 楼梯地形 border_width=20m，内置 out_of_terrain_bounds 边界太大不会触发；
  # 用 out_of_patch_bounds 代替：机器人离开自己的 8×8 patch 即终止（time_out）
  cfg.terminations.pop("out_of_terrain_bounds", None)
  cfg.terminations.pop("reached_edge", None)
  cfg.terminations["out_of_patch_bounds"] = TerminationTermCfg(
    func=out_of_patch_bounds,
    time_out=True,
    params={"half_size": 4.0, "margin": 0.1},  # patch 8×8, limit=3.9m  )
  )
  # 命令：增大 switch_steps 给机器人足够时间上/下楼
  # 机器人出生在 y≈4m 处，以 1m/s 走到中心约 4~6s，
  # 台阶 + 转向实际需要更多时间，设 750 steps ≈ 15s
  cfg.commands["twist"] = StairVelocityCommandCfg(
          entity_name="robot",
          resampling_time_range=(20.0, 20.0),  # 与 episode_length_s 对齐，整个 episode 不随机重采
          debug_vis=True,
          forward_speed=1.0,
          switch_steps=750,    # 500 steps × 0.02s/step = 10s，足够走上/下楼梯
          center_threshold=0.8,
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
          ),}

  
  if play:
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

  if play:
    cfg.scene.terrain.max_init_terrain_level = 5

  return cfg

def my_go2_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go1 flat terrain velocity configuration."""
  cfg = my_go2_rough_env_cfg(play=play)



  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensors and collision sensors not needed on flat.
  remove_sensors = {
    "terrain_scan",
    "self_collision",
    "thigh_ground_touch",
    "shank_ground_touch",
    "trunk_ground_touch",
  }

  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name not in remove_sensors
  )

  cfg.rewards["upright"].params.pop("terrain_sensor_names", None)

  # Remove granular collision rewards (not useful on flat ground).
  for key in ("self_collisions", "shank_collision", "trunk_head_collision"):
    cfg.rewards.pop(key, None)

  # On flat terrain fell_over is sufficient; thigh contact implies fallen.
  cfg.terminations.pop("illegal_contact", None)
  cfg.terminations.pop("out_of_terrain_bounds", None)
  cfg.terminations["fell_over"] = TerminationTermCfg(
    func=mdp.bad_orientation,
    params={"limit_angle": math.radians(70.0)},
  )

  # Disable terrain curriculum (not present in play mode since rough clears all).
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.5, 2.0)
    twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

  return cfg