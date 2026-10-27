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
from myrobot_mjlab.task.DTC.observation import terms as dtc_obs
from myrobot_mjlab.task.DTC.reward import terms as dtc_reward
from myrobot_mjlab.task.DTC.tamols.dtc_command import (
  DTCFootholdCommandCfg,
)
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig
from myrobot_mjlab.config.command.StairVelocityCommand import StairVelocityCommandCfg
from myrobot_mjlab.config.curriculum.stair_curriculum import terrain_levels_forward_dist
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
      debug_vis=False,  # 训练默认关; play 分支统一打开
    )

    # TAMOLS 专用前向高程扫描: 观测范围必须覆盖规划落点 (最大约 base+0.6 m),
    # 否则 map 中未被观测的区域会用 ground_default (当前台阶高度) 填平,
    # 规划器会把落点放到"虚拟平地"上, z 比真实台阶低, 脚就会踢到台阶立面.
    # 保持 terrain_scan (actor height_scan) 尺寸不变, 避免改变策略观测维度.
    terrain_scan_map = RayCastSensorCfg(
      name="terrain_scan_map",
      frame=ObjRef(type="body", name="base_link", entity="robot"),
      ray_alignment="yaw",
      pattern=GridPatternCfg(size=(2.4, 1.2), resolution=0.1),
      max_distance=5.0,
      exclude_parent_body=True,
      include_geom_groups=(0,),
      debug_vis=False,
    )

    # 足底垂直高度传感器: 同时用于
    #   1) TAMOLS 高程图 (足下真实地形, 补前向雷达的空洞);
    #   2) 摆动腿抬脚高度奖励 (feet_swing_height).
    # 帧顺序与 feet_ground_contact 的 primary 顺序一致 (FR, FL, RR, RL).
    foot_height_scan = TerrainHeightSensorCfg(
      name="foot_height_scan",
      frame=tuple(
        ObjRef(type="body", name=f"{leg}_foot", entity="robot")
        for leg in ("FR", "FL", "RR", "RL")
      ),
      ray_alignment="world",
      pattern=GridPatternCfg(size=(0.08, 0.08), resolution=0.04),
      max_distance=1.0,
      exclude_parent_body=True,
      include_geom_groups=(0,),
      debug_vis=False,
      reduction="min",
    )

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
        # ---- DTC: TAMOLS 相关观测 (见 tamols/obs.py 文档) ----
        "tamols_footholds": ObservationTermCfg(
          func=dtc_obs.tamols_footholds,
          params={"command_name": "Tamols_footholds", "entity_name": "robot"},
          # 落点观测是控制目标, 噪声过大会让 3~10 cm 的台阶高度完全淹没.
          noise=Unoise(n_min=-0.02, n_max=0.02),
        ),
        "tamols_desired_joint_pos": ObservationTermCfg(
          func=dtc_obs.tamols_desired_joint_pos,
          params={"command_name": "Tamols_footholds", "entity_name": "robot"},
        ),
        "tamols_desired_contacts": ObservationTermCfg(
          func=dtc_obs.tamols_desired_contacts,
          params={"command_name": "Tamols_footholds"},
        ),
        # 固定 trot 的每腿相位 (sin/cos 编码, 连续无跳变) + 摆动掩码 [12]
        "tamols_gait_phase": ObservationTermCfg(
          func=dtc_obs.tamols_gait_phase,
          params={"command_name": "Tamols_footholds"},
        ),
        "height_scan": ObservationTermCfg(
          func=mdp.height_scan,
          params={"sensor_name": "terrain_scan", "offset": 0.5},
          noise=Unoise(n_min=-0.1, n_max=0.1),
        ),
      }

    critic_terms = {
        **actor_terms,
        # Critic sees the true (unbiased) joint positions as privileged information.
        "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
        # ---- DTC: critic 特权信息 —— TAMOLS 期望基座状态 (actor 不含) ----
        # 注意 tamols_base_twist_desired 是 TAMOLS 规划输出的基座速度,
        # 与 actor 的 "command" (velocity 命令项指令) 是两个不同量.
        "tamols_base_pos_desired": ObservationTermCfg(
          func=dtc_obs.tamols_base_pos_desired,
          params={"command_name": "Tamols_footholds", "entity_name": "robot"},
        ),
        "tamols_base_quat_desired": ObservationTermCfg(
          func=dtc_obs.tamols_base_quat_desired,
          params={"command_name": "Tamols_footholds"},
        ),
        "tamols_base_twist_desired": ObservationTermCfg(
          func=dtc_obs.tamols_base_twist_desired,
          params={"command_name": "Tamols_footholds", "entity_name": "robot"},
        ),
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
        "twist": UniformVelocityCommandCfg(
          entity_name="robot",
          resampling_time_range=(3.0, 8.0),
          rel_standing_envs=0.1,
          rel_heading_envs=0.3,
          rel_forward_envs=0.2,
          heading_command=True,
          heading_control_stiffness=0.5,
          debug_vis=False,  # 训练默认关; play 分支统一打开
          ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-0.5, 0.5),
            heading=(-math.pi, math.pi),
          ),
        ),
        # DTC: TAMOLS 落脚点规划命令项 (继承 fetch 现有方案, 绘制走
        # DTC debug_viz). 仅供仿真视图观测落足点与后续 obs/reward 接入,
        # 不参与速度任务的观测/奖励; 平地变体无 terrain_scan 时命令项
        # 内部容错 (足点退化为平地 z=0).
        # sensor_name 用前向更远的 terrain_scan_map; actor 的 height_scan
        # 仍用 1.6×1.0 的 terrain_scan (保持观测维度/旧 checkpoint 兼容).
        # 地图范围收窄到扫描覆盖内, 避免规划器在未观测区域当平地采样.
        "Tamols_footholds": DTCFootholdCommandCfg(
          entity_name="robot",
          sensor_name="terrain_scan_map",
          foot_sensor_name="foot_height_scan",
          x_range=(-1.0, 1.1),
          y_range=(-0.6, 0.6),
          # swing_obs 目前没有接入 actor 观测/奖励, 每步组装会白耗 ~1 ms;
          # 需要时再打开 (见 DTCFootholdCommandCfg.swing_obs_enable).
          swing_obs_enable=False,
          debug_vis=False,  # 训练默认关; play 分支统一打开
          dot_radius=0.035,
          show_nominal_dots=True,
        ),
      }

    ##
    # Events
    ##
    
    events = {
        # 出生位姿 (rough/flat 共用; stair/tire 变体在 config/env_cfg.py
        # 里整体覆写 events): 绕 env 原点小范围随机, yaw 全域随机.
        # 旧值 y=(3.5,4.0)+yaw=-1.57 是楼梯/轮胎任务的进场位, 在 rough
        # 上会出生到相邻子地形块 (高度不匹配); z=(-0.01,-0.05) 为负偏移,
        # 默认根高 0.278 时脚底本就只离地 5mm, 每次重生穿地 1~5cm.
        "reset_base": EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "z": (0.0, 0.0),
            "yaw": (-3.14, 3.14),
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
      # ---- DTC: Base Pose Tracking r_{B_n} (对 TAMOLS 参考轨迹) ----
      # 参考随 time_since_replan 流动 (见 reward/terms._advance_reference);
      # σ 标定: exp(-σ·e), e=5cm -> 0.37 (pos), e=3° -> 0.35 (rot).
      # 旧值 σ_pos=1200/σ_rot=90 要求 mm/1° 级精度, 奖励恒为 0 (无梯度).
      "base_pose_tracking_pos": RewardTermCfg(
        func=dtc_reward.base_pose_tracking,
        weight=1.0,
        params={
          "command_name": "Tamols_footholds",
          "sigma": 20.0,
          "order": 0,
          "component": "pos",   # 仅位置 (向量差)
        },
      ),
      "base_pose_tracking_rot": RewardTermCfg(
        func=dtc_reward.base_pose_tracking,
        weight=1.0,
        params={
          "command_name": "Tamols_footholds",
          "sigma": 20.0,
          "order": 0,
          "component": "rot",   # 仅姿态 (四元数差)
        },
      ),
      "base_pose_tracking_vel_pos": RewardTermCfg(
        func=dtc_reward.base_pose_tracking,
        weight=1.0,
        params={
          "command_name": "Tamols_footholds",
          "sigma": 10.0,
          "order": 1,
          "component": "pos",   # 仅线速度 (向量差)
        },
      ),
      "base_pose_tracking_vel_rot": RewardTermCfg(
        func=dtc_reward.base_pose_tracking,
        weight=1.0,
        params={
          "command_name": "Tamols_footholds",
          "sigma": 1.0,
          "order": 1,
          "component": "rot",   # 仅角速度 (向量差)
        },
      ),
      "base_pose_tracking_acc_pos": RewardTermCfg(
        func=dtc_reward.base_pose_tracking,
        weight=0.5,
        params={
          "command_name": "Tamols_footholds",
          "sigma": 0.005,
          "order": 2,
          "component": "pos",   # 仅线加速度 (向量差)
        },
      ),
      "base_pose_tracking_acc_rot": RewardTermCfg(
        func=dtc_reward.base_pose_tracking,
        weight=0.5,
        params={
          "command_name": "Tamols_footholds",
          "sigma": 0.05,
          "order": 2,
          "component": "rot",   # 仅角加速度 (向量差)
        },
      ),
      # ---- DTC: 硬落脚跟踪 r_{p_i} = -ln(max(d, d_min)^2 + eps) ----
      # 门控: 期望接触态 且 ||f_i|| > 1 N 且 本支撑相未给过分;
      # 每步态周期每脚至多一次; 出生/reset 时支撑相腿不送分 (需真实
      # 摆动->触地才得分); d_min 封顶单脚得分 (去掉 d->0 时无界彩票).
      "foothold_hard_tracking": RewardTermCfg(
        func=dtc_reward.foothold_hard_tracking,
        weight=6.0,
        params={
          "command_name": "Tamols_footholds",
          "sensor_name": "feet_ground_contact",
          "eps": 1e-5,
          "force_threshold": 1.0,
          "d_min": 0.02,
        },
      ),
      # ---- DTC: 摆动腿抬脚高度 ----
      # 在触地瞬间检查本摆动相的峰值离地高度, 低于 target_height 时罚.
      # 平地策略常学出 2~5 cm 的低擦地摆动; 上楼梯时这个高度会直接撞上
      # 台阶立面 (日志里 illegal_contact 占终止的 38%), 因此显式给一个
      # 10 cm 的抬脚目标. 传感器为足底 3×3 垂直射线 (foot_height_scan).
      # 之前 15 cm / -2.0 在 0.55 m/s 下会把前腿抬得过高, play 看起来
      # 很“狂野”; 降到 10 cm 并减弱权重, 让摆动更贴近地面.
      "foot_swing_height": RewardTermCfg(
        func=mdp.feet_swing_height,
        weight=-1.0,
        params={
          "sensor_name": "feet_ground_contact",
          "height_sensor_name": "foot_height_scan",
          "target_height": 0.10,
          "command_name": "twist",
          "command_threshold": 0.1,
        },
      ),
      # ---- DTC: 动作一致性 r_c (相邻两次规划解差异惩罚) ----
      # 按 env 触地沿捕获解快照, 绝对时间对齐后在重叠窗采样: 以指令速度
      # 行走时两解重合 (差≈0), 站桩不动时差 ≈ |v_cmd|·Δ·采样数 (实测
      # raw≈-4.7) —— 这是"没跟上指令"的跟踪误差, 方向与速度跟踪一致.
      # weight 20 -> 0.5: 旧语义是每步恒定移动惩罚; 新语义下 0.5 已足够
      # 约束规划抖动, 且保证早期"站住"的净回报为正 (杜绝早死最优).
      "action_consistency": RewardTermCfg(
        func=dtc_reward.action_consistency,
        weight=0.5,
        params={
          "command_name": "Tamols_footholds",
          "w_p": 1.0,
          "dt_compare": 0.01,
          "err_cap": 0.1,   # 单样本差上界: 指令突变/跟踪失败由跟踪项负责
        },
      ),
      # ---- DTC: 原地滞留/打转惩罚 (仿 track 的 stall_penalty) ----
      # 指令要求移动 (|v_cmd_xy| > 0.1) 而实际沿指令方向的投影速度
      # < 0.1 m/s 时固定罚分; 原地左右徘徊/打转也会被罚.
      # 命令为零的环境 (standing 采样、stair north_peak 到达中心) 不罚.
      # raw -5.0 × weight 1.0 × dt 0.02 = -0.1/step (-5/s), 能压住
      # "原地徘徊也能拿速度跟踪分"的局部最优.
      "stall_penalty": RewardTermCfg(
        func=dtc_reward.stall_penalty,
        weight=1.0,
        params={
          "command_name": "twist",
          "vel_threshold": 0.1,
          "cmd_threshold": 0.1,
          "penalty_value": -5.0,
        },
      ),
      # ---- DTC: 正则化惩罚 (论文补充材料) ----
      "foot_contact_power": RewardTermCfg(
        func=dtc_reward.foot_contact_power,
        weight=0.02,
        params={
          "sensor_name": "feet_ground_contact",
          "entity_name": "robot",
        },
      ),
      # r_r2 = -(Σ q̇·τ)²: 正常行走 Σq̇τ ~ 40-200 W, 平方后 1.6e3-4e4,
      # 旧权重 0.025 让该项达 -250/步 (第0轮实测比率 -21/s), 淹没所有
      # 正项并推动"早死"; 1e-3 下正常行走 -2~-5/步, 只罚暴力扑腾.
      "joint_power": RewardTermCfg(
        func=dtc_reward.joint_power,
        weight=1e-3,
        params={"entity_name": "robot"},
      ),
      "dtc_action_rate": RewardTermCfg(
        func=dtc_reward.action_rate,
        weight=0.02,
      ),
      "dtc_joint_acceleration": RewardTermCfg(
        func=dtc_reward.joint_acceleration,
        weight=1e-6,
        params={"entity_name": "robot"},
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
      "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
      "air_time": RewardTermCfg(
        func=mdp.feet_air_time,
        weight=1.0,  # Override per-robot.
        params={
          "sensor_name": "feet_ground_contact",
          "threshold_min": 0.05,
          "threshold_max": 0.5,
          "command_name": "twist",
          "command_threshold": 0.5,
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
        params={"threshold_fraction": 3.5125},
        time_out=True,
      ),
    }
    
    ##
    # Curriculum
    ##
    
    curriculum = {
      "terrain_levels": CurriculumTermCfg(
        func=terrain_levels_forward_dist,
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
        sensors=(terrain_scan, terrain_scan_map, foot_height_scan),
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