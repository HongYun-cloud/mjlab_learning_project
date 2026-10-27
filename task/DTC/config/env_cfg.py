import math
import os
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
from myrobot_mjlab.task.DTC.velocity_env_cfg import make_velocity_env_cfg
from mjlab.scene import SceneCfg
from mjlab.terrains import TerrainEntityCfg
from myrobot_mjlab.config.terrain.stair_terrain import STAIRS_TERRAINS_CFG
from myrobot_mjlab.config.terrain.tire_array import TIRE_ARRAY_TERRAINS_CFG
from myrobot_mjlab.config.event.stair_reset import reset_root_state_from_stair,out_of_patch_bounds
from myrobot_mjlab.config.event.tire_reset import reset_root_state_from_tire
from myrobot_mjlab.config.reward.stair_reward import (
    reach_peak_bonus,
    track_linear_velocity_stair,
)
from myrobot_mjlab.config.command.StairVelocityCommand import (
    StairVelocityCommandCfg,
    reached_stair_peak,
)
from myrobot_mjlab.config.command.TireVelocityCommand import TireTraverseVelocityCommandCfg
from myrobot_mjlab.config.curriculum.stair_curriculum import terrain_levels_stair_peak
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from myrobot_mjlab.task.DTC.tamols.dtc_command import DTCFootholdCommandCfg
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

  # pose (variable_posture) 的分段姿态标准差 (per-robot)
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



#   for reward_name in ["foot_clearance", "foot_slip"]:
#     cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

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

    # play 打开全部调试可视化 (训练默认全关: velocity_env_cfg 基础 cfg 与
    # 各命令项 cfg 的 debug_vis 均为 False, 仅此处恢复)。
    for _sensor in (cfg.scene.sensors or ()):
      if hasattr(_sensor, "debug_vis"):
        _sensor.debug_vis = True
    # terrain_scan_map 是 TAMOLS 内部使用的 325 线前向高程扫描, 不需要在
    # Viser 里画射线; 打开它会让 debug 图元翻倍, 明显拖慢 viewer.
    for _sensor in (cfg.scene.sensors or ()):
      if getattr(_sensor, "name", None) == "terrain_scan_map":
        _sensor.debug_vis = False
    for _cmd_cfg in cfg.commands.values():
      _cmd_cfg.debug_vis = True

    # TAMOLS 单次规划耗时打印 (仅 play; 训练配置保持默认关闭)
    _foothold_cmd = cfg.commands.get("Tamols_footholds")
    if _foothold_cmd is not None:
      _foothold_cmd.plan_time_log = True
      _foothold_cmd.map_ui_enable = True

    cfg.observations["actor"].enable_corruption = False

    # play 只看运动/规划, 不需要 reward 反向传播. base_pose_tracking 六项
    # 每步约 7 ms, 加上 action_consistency/正则项约 10 ms, 是 Viser 帧率
    # 的主要瓶颈之一. 默认在 play 中关闭这些高开销项; 需要完整 reward
    # 面板时设置环境变量 DTC_PLAY_FULL_REWARDS=1.
    if os.getenv("DTC_PLAY_FULL_REWARDS", "0") != "1":
      for _key in (
        "base_pose_tracking_pos",
        "base_pose_tracking_rot",
        "base_pose_tracking_vel_pos",
        "base_pose_tracking_vel_rot",
        "base_pose_tracking_acc_pos",
        "base_pose_tracking_acc_rot",
        "action_consistency",
        "foot_contact_power",
        "joint_power",
        "dtc_action_rate",
        "dtc_joint_acceleration",
        "dof_pos_limits",
      ):
        cfg.rewards.pop(_key, None)

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
            # 从平地 checkpoint 微调时不必再从 Level 0 (0 台阶) 重新爬课程;
            # Level 1 ≈ 3.3 cm 台阶, 仍留出适应空间.
            max_init_terrain_level=1,
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
  cfg.rewards.pop("pose")
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
          debug_vis=False,  # 训练默认关; play 分支统一打开
          # 楼梯踏面只有 0.3 m; 步幅 = forward_speed × stride_duration.
           # 0.40 m/s × stride_duration=0.8 s ≈ 0.32 m; 落点再限制到
           # 0.30 m, 避免跨过 0.3 m 踏面后落在台阶边缘/立面.
           forward_speed=0.40,
          switch_steps=750,    # 500 steps × 0.02s/step = 10s，足够走上/下楼梯
          center_threshold=0.8,
          # north_peak 模式: 速度恒指向 patch 正北方向台阶区外 1 m 的平地
          # (origin + (0, 3.85): 台阶区外缘 ~3.15 m, 终点在最高面 z=0 上),
          # 头部朝向恒北 (+y 为北, yaw=π/2, wz P 控制对齐),
          # 到达目标 (< 0.8 m) 后原地站立, 不再切下楼。
          # 注意 3.85 贴着 out_of_patch_bounds 上限 3.9 (half 4.0 - margin 0.1),
          # 若再外移需同步放宽 margin。
          target_mode="north_peak",
          target_offset_w=(0.0, 3.85),
          face_heading=math.pi / 2,
          stop_at_target=True,
          # 速度方向 = 当前基座位置 -> 目标终点, 大小恒为 forward_speed.
          forward_only=False,
      )

  # ---- 楼梯上的 TAMOLS 规划修正 ----------------------------------------
  # 1) 标称落点前馈混入 50% 实测速度: 策略爬楼时实际速度低于指令,
  #    纯指令前馈会让落点系统性偏前 0.1~0.3 m;
  # 2) 相对髋部水平步幅上限 0.30 m: 和 0.3 m 踏面匹配, 减少跨边/卡脚;
  # 3) 每对对角腿 (FL+RR / FR+RL) 都落地一次就重规划, 即每半个
  #    步态周期 (~0.4 s) 更新一次; 整周期模式 (0.8 s) 在楼梯上会让
  #    冻结计划迅速过时.
  foothold_cmd = cfg.commands["Tamols_footholds"]
  assert isinstance(foothold_cmd, DTCFootholdCommandCfg)
  foothold_cmd.replan_after_all_four_touchdowns = False
  foothold_cmd.in_place_gait_enable = True
  for _opt in (foothold_cmd.optimizer_cfg, foothold_cmd.planner_cfg.optimizer_cfg):
      _opt.nominal_cmd_blend = 0.5
      _opt.max_step_offset = 0.30

  # 步频: 默认 trot 周期 0.6 s (每腿 1.67 Hz) 在 play 里显得很快;
  # 拉长到 0.8 s (每腿 1.25 Hz), 落脚节奏更慢.
  # 0.40 m/s × 0.8 s = 0.32 m, 由 max_step_offset=0.30 限制到踏面长度.
  foothold_cmd.gait_cfg.stride_duration = 0.8

  # 3~10 cm 的台阶上, ±0.1 m 的高度扫描噪声会把台阶完全淹没;
  # 只对楼梯任务收紧 (不改变 checkpoint 的观测维度).
  cfg.observations["actor"].terms["height_scan"].noise = Unoise(
      n_min=-0.05, n_max=0.05
  )

  # 大腿 10 N 接触就终止过于严格: 楼梯上正常擦碰台阶立面也会误判,
  # 日志中 illegal_contact 占终止的 38%. 抬高到 50 N, 仍能抓住真实卡死.
  cfg.terminations["illegal_contact"].params["force_threshold"] = 50.0

  # 课程: 连续 5 个 episode 到达楼梯顶端 -> 难度 +1 级;
  #       连续 3 个 episode 未到达 -> 难度 -1 级 (替换继承的 forward_dist)。
  cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
      func=terrain_levels_stair_peak,
      params={"command_name": "twist", "up_streak": 5, "down_streak": 3},
  )
  # 到达终点不再终止 episode: 固定跑满 10 s, 最后 2 s 在终点停稳/
  # 姿态良好才给奖励. 课程仍用 peak_reached_episode 记"到达".
  cfg.terminations.pop("reached_peak", None)
  # 到终点后的分项保持奖励: 到达终点/线速度为 0/姿态良好 各占 1/3,
  # 连续评分, 越好越高; 只保持在最后 2 s 窗口内时结算.
  cfg.rewards["reach_peak_bonus"] = RewardTermCfg(
      func=reach_peak_bonus,
      weight=1.0,
      params={
          "command_name": "twist",
          "foothold_term": "foothold_hard_tracking",
          "bonus_scale": 1.5,
          "only_last_seconds": 2.0,
          "goal_weight": 1.0 / 3.0,
          "velocity_weight": 1.0 / 3.0,
          "posture_weight": 1.0 / 3.0,
          "goal_sigma": 0.40,
          "velocity_sigma": 0.10,
          "yaw_rate_sigma": 0.15,
          "posture_sigma": 0.15,
      },
  )
  # 训练 episode 上限 10 秒: 0.45 m/s 爬 3 m 约需 6.8 s, 留出起步/抬腿
  # 和最后 2 秒奖励窗口; 未到顶会被 time_out 计失败.
  if not play:
    cfg.episode_length_s = 10.0

  cfg.events = {
          "reset_base": EventTermCfg(
          func=reset_root_state_from_stair,
          mode="reset",
          params={
              "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),          # 倒金字塔中心凹底平地 (3m×3m 范围中央)
                "z": (0.0, 0.0),
                "yaw": (math.pi / 2, math.pi / 2),  # 出生即朝北 (与 face_heading 一致)
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
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.5, 2.0)
    twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)
    twist_cmd.debug_vis = True
    cfg.commands["Tamols_footholds"].debug_vis = True
                                  
                          
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
    # "terrain_scan",
    "self_collision",
    "thigh_ground_touch",
    "shank_ground_touch",
    "trunk_ground_touch",
  }

  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name not in remove_sensors
  )

  # Remove granular collision rewards (not useful on flat ground).
  for key in ("self_collisions", "shank_collision", "trunk_head_collision"):
    cfg.rewards.pop(key, None)

  # flat 无 terrain_scan 传感器: 同步移除 height_scan 观测 (其余
  # TAMOLS 观测项保留 —— 落脚点/期望关节/期望接触不依赖地形传感器).
  # for group in (cfg.observations["actor"], cfg.observations["critic"]):
  #   group.terms.pop("height_scan", None)

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

def my_go2_platform_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """DTC 平台任务: 起始大平台 -> 5 个小平台 -> 目标大平台.

  地形复用 ``config/terrain/plum_blossom_terrain.py`` 的“大平台 +
  中间小平台 + 大平台”结构, 但把中间小平台数量设为 5、横向偏置设为 0,
  形成一条直线平台链。出生点在起始平台中心, 终点在目标平台中心。

  1. 起始/目标大平台: 0.8 m × 0.8 m, 顶面 z=0.20 m
  2. 中间小平台: 0.8 m × 0.2 m (沿行进方向 × 横向), 数量 5, 间距 0.2 m
  3. 出生点: 起始平台中心, 朝 +X
  4. 终点: 目标平台中心; 起始/目标中心距 = 5.8 m
  """
  from myrobot_mjlab.config.terrain.plum_blossom_terrain import (
      PLUM_BLOSSOM_TERRAINS_CFG,
      plum_blossom_terrain,
  )

  cfg = my_go2_stair_env_cfg(play=play)

  # ---- 1. 地形: 5 个中间小平台 (直线排列, 不是梅花桩交错) ----
  num_pillars = 5
  pillar_gap = 0.20
  platform_terrain = plum_blossom_terrain(
      proportion=1.0,
      num_pillars=num_pillars,
      pillar_length=0.8,    # 沿行进方向 (X)
      pillar_width=0.2,     # 横向 (Y)
      pillar_gap=pillar_gap,
      vary_gap_with_difficulty=False,
      lateral_offset=0.0,   # 5 个小平台排成直线
      spawn_on_platform=True,
  )
  platform_generator = replace(
      PLUM_BLOSSOM_TERRAINS_CFG,
      sub_terrains={"platform": platform_terrain},
  )
  cfg.scene.terrain = TerrainEntityCfg(
      terrain_type="generator",
      terrain_generator=platform_generator,
      max_init_terrain_level=0,
  )

  # ---- 2. 起始/终点中心距 ----
  # 起始平台半长 + 平台到首个小平台的 gap + 5 个小平台 + 4 个间隙
  # + 末个小平台到目标平台的 gap + 目标平台半长
  goal_distance = (
      platform_terrain.platform_length / 2.0
      + platform_terrain.platform_to_pillar_gap
      + platform_terrain.num_pillars * platform_terrain.pillar_length
      + max(0, platform_terrain.num_pillars - 1) * float(pillar_gap)
      + platform_terrain.platform_to_pillar_gap
      + platform_terrain.second_platform_length / 2.0
  )

  # ---- 3. 速度命令: 从起始平台指向目标平台 ----
  cfg.commands["twist"] = StairVelocityCommandCfg(
      entity_name="robot",
      # episode=30 s, 重采样必须晚于 episode, 避免中途重置 peak 状态.
      resampling_time_range=(40.0, 40.0),
      debug_vis=False,
      # 小平台沿行进方向 0.8 m、横向 0.2 m, 间距 0.2 m; 步幅要小于
      # 一个平台长度, 保持 0.30 m/s.
      forward_speed=0.30,
      center_threshold=0.35,
      target_mode="north_peak",
      target_offset_w=(goal_distance, 0.0),
      face_heading=0.0,
      stop_at_target=True,
      forward_only=True,
  )

  # ---- 4. 出生在起始平台中心, 朝 +X ----
  cfg.events["reset_base"].params["pose_range"] = {
      "x": (-0.10, 0.10),
      "y": (-0.10, 0.10),
      "z": (0.0, 0.0),
      "yaw": (0.0, 0.0),
  }

  # 平台链总长 5.8 m, 出生点在起始平台; 原来的 patch 边界 half_size=4.0
  # 会在机器人到达终点前就触发 out_of_patch_bounds, 这里按目标距离放宽.
  if "out_of_patch_bounds" in cfg.terminations:
      cfg.terminations["out_of_patch_bounds"].params["half_size"] = goal_distance + 1.0
      cfg.terminations["out_of_patch_bounds"].params["margin"] = 0.1

  # 固定平台链, 不做坡度/间隙课程.
  cfg.curriculum = {}

  # 平台任务沿用楼梯的“最后 2 秒”窗口: 到目标平台后停稳/姿态良好,
  # 在最后 2 s 内按 到达/线速度/姿态 三个分项持续拿分.

  if not play:
      # 5.8 m / 0.30 m/s ≈ 19.3 s, 留出起步/落脚时间.
      cfg.episode_length_s = 30.0
  else:
      cfg.commands["twist"].debug_vis = True
      cfg.commands["Tamols_footholds"].debug_vis = True

  return cfg
