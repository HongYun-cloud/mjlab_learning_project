from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import torch
import torch.nn.functional as F
import math

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

from .map_processing import TorchTAMOLSMapProcessor
from .tamols_foothold import TAMOLSOptimizerCfg
from .tamols_motion_planner import TorchTAMOLSMotionPlanner, TAMOLSMotionPlannerCfg
from .gait_scheduler import GaitScheduler, GaitSchedulerCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer


FOOT_NAMES = ("FL", "FR", "RL", "RR")


@dataclass(kw_only=True)
class TamolsMotionCommandCfg(CommandTermCfg):
    """TAMOLS 完整运动规划命令配置 (含基座姿态、加速度、接触计划、相位与触地关节角)。"""
    entity_name: str = "robot"
    sensor_name: str = "terrain_scan"
    foot_sensor_name: str = "foot_height_scan"
    resampling_time_range: tuple[float, float] = (1000.0, 1000.0)
    debug_vis: bool = True

    # 3D 可视化小红点参数
    dot_radius: float = 0.035
    """足端规划小红点球体半径 (m)"""

    dot_color: tuple[float, float, float, float] = (1.0, 0.05, 0.05, 0.95)
    """RGBA 鲜艳亮红色 (小红点)"""

    show_nominal_dots: bool = False
    """是否同时显示未优化的标称落脚点 (用于对比)"""

    nominal_dot_color: tuple[float, float, float, float] = (0.2, 0.6, 1.0, 0.7)
    """标称落脚点颜色 (淡蓝)"""

    nominal_speed: float = 0.45
    """期望标称前向行进速度 (m/s)，适中速度有利于稳定爬楼与避障"""

    velocity_command_name: str = "twist"
    """提供期望速度 (机体系 vx, vy, wz) 的指令项名称。

    找到该指令项时，TAMOLS 的五项速度相关计算全部直接使用它：
        1. Raibert 标称落脚点 (半支撑时长前馈 + 速度反馈);
        2. 期望基座线速度 desired_base_lin_vel_w;
        3. 期望基座角速度 desired_base_ang_vel_w;
        4. 期望基座线加速度 desired_base_lin_acc_w (向指令速度加速);
        5. 期望基座角加速度 desired_base_ang_acc_w (向指令角速度加速)。
    UniformVelocityCommand 自带 Viser 摇杆滑条，可在仿真中直接给入期望速度。
    找不到该指令项时才回退到 Goal_direction 航向误差推导 (旧行为)。
    """

    # 极速默认旁路模式 (若需要跳过计算直接输出默认姿态，设为 True)
    bypass_tamols: bool = False
    """设为 False 以启用全 GPU PyTorch Conv2D TAMOLS 落脚点与运动规划"""

    # 重新求解触发模式
    replan_on_touchdown: bool = True
    """主要在腿触地 (Touchdown / 相切换) 时触发重新规划求解"""

    replan_on_command_change: bool = True
    """当目标航向或速度指令发生显著改变时触发重新计算"""

    command_change_threshold: float = 0.15
    """期望速度 (vx/vy/wz 任一分量) 变化超过该阈值时重新规划 (m/s 或 rad/s)"""

    # 本地高程图尺寸 (m)
    x_range: tuple[float, float] = (-1.0, 1.4)
    y_range: tuple[float, float] = (-0.9, 0.9)
    cell_size: float = 0.05

    optimizer_cfg: TAMOLSOptimizerCfg = field(default_factory=TAMOLSOptimizerCfg)
    planner_cfg: TAMOLSMotionPlannerCfg = field(default_factory=TAMOLSMotionPlannerCfg)
    gait_cfg: GaitSchedulerCfg = field(default_factory=GaitSchedulerCfg)

    def build(self, env: ManagerBasedRlEnv) -> TamolsMotionCommand:
        return TamolsMotionCommand(self, env)


# 保留别名兼容原有配置
TamolsFootholdCommandCfg = TamolsMotionCommandCfg


class TamolsMotionCommand(CommandTerm):
    """TAMOLS 完整运动规划与低层控制接口执行项 (纯 PyTorch GPU 2D 卷积与张量并行实现)。
    
    在每个控制步：
    1. 步态时钟调度器更新当前步态相位 phi_i、接触计划 c_i 以及当前相剩余时间 t_rem, i；
    2. 将前向雷达 (terrain_scan) 与足底微扫描点云投影写入 GPU 局部高程图张量；
    3. 仅在触地 (Touchdown)、指令显著改变或重置时，通过 PyTorch 2D 卷积与形态学并行求解最优落脚点；
    4. 根据落脚点与地形倾角，计算地形自适应基座期望姿态 (Roll, Pitch, Yaw, Quat) 与高度；
    5. 计算基座参考线加速度与角加速度；
    6. 利用 Go2 解析逆运动学闭式求解触地时刻的 12 维期望关节位置；
    7. 在 Viser 界面中渲染落脚点小红点。
    """
    cfg: TamolsMotionCommandCfg

    def __init__(self, cfg: TamolsMotionCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.robot: Entity = env.scene[cfg.entity_name]

        # 1. 步态调度器
        self.gait_scheduler = GaitScheduler(
            cfg=cfg.gait_cfg,
            num_envs=self.num_envs,
            device=self.device,
        )

        # 2. 地图尺寸与 GPU 处理器
        self.cell_size = float(cfg.cell_size)
        self.x_min, self.x_max = float(cfg.x_range[0]), float(cfg.x_range[1])
        self.y_min, self.y_max = float(cfg.y_range[0]), float(cfg.y_range[1])
        self.nx = int(round((self.x_max - self.x_min) / self.cell_size))
        self.ny = int(round((self.y_max - self.y_min) / self.cell_size))

        self.map_processor = TorchTAMOLSMapProcessor(
            cell_size=self.cell_size,
            device=self.device,
        )
        self.torch_planner = TorchTAMOLSMotionPlanner(
            cfg=cfg.planner_cfg,
            cell_size=self.cell_size,
            x_min=self.x_min,
            x_max=self.x_max,
            y_min=self.y_min,
            y_max=self.y_max,
            device=self.device,
        )

        # 3. GPU 本地高程图缓冲区 [num_envs, 1, Nx, Ny]
        #    h_raw: 世界系 z 高度; h_valid: 该格是否被真实雷达命中过。
        #    地图随里程计滚动 (见 _roll_map_odometry)，未观测格子用 ground_default 填充。
        self.h_raw = torch.zeros((self.num_envs, 1, self.nx, self.ny), device=self.device, dtype=torch.float32)
        self.h_valid = torch.zeros((self.num_envs, 1, self.nx, self.ny), device=self.device, dtype=torch.float32)
        """1.0 = 该格在当前机身系下持有有效的世界高度采样。"""

        self.ground_default = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        """未观测格子的回退高度 (地形估计，不跟随机身瞬时机高)。"""

        self._last_base_xy = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float32)
        self._last_base_yaw = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self._pose_valid = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        """False 表示该环境刚 reset，本步不做里程计滚动 (避免用瞬移前的位姿污染地图)。"""

        self._map_needs_init = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        """True 表示地图需要在 sim.forward() 之后用当帧雷达点重新初始化。

        reset 时机器人位姿数据尚未刷新 (reset 回调先于 write_data_to_sim/forward)，
        不能在那里读 root_link_pos_w 来估地高，否则会把上一帧 (甚至出生默认位姿) 的机高
        当成地面，导致所有未观测格子的 z 悬在空中。
        """

        # 4. 规划输出张量缓冲区
        self.optimal_footholds = torch.zeros((self.num_envs, 4, 3), device=self.device, dtype=torch.float32)
        self.nominal_footholds = torch.zeros((self.num_envs, 4, 3), device=self.device, dtype=torch.float32)
        self.planar_footholds = torch.zeros((self.num_envs, 8), device=self.device, dtype=torch.float32)

        self.desired_base_pos_w = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self.desired_base_quat_w = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.float32)
        self.desired_base_euler = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self.desired_base_lin_vel_w = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self.desired_base_ang_vel_w = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self.desired_base_lin_acc_w = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self.desired_base_ang_acc_w = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self.desired_joint_pos = torch.zeros((self.num_envs, 12), device=self.device, dtype=torch.float32)
        self._command = torch.zeros((self.num_envs, 12), device=self.device, dtype=torch.float32)

        # 5. 事件驱动重新规划跟踪与到达一次性给分状态跟踪
        self.footholds_initialized = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.last_cmd_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        """上一次重新规划时的期望速度 (vx, vy, wz)，用于检测速度指令突变。"""
        self._velocity_cmd_term: CommandTerm | None = None
        """提供期望速度的指令项缓存 (如 twist)，首次使用时惰性解析。"""

        # 到达参考点后仅给一次分的状态掩码 (True 代表当前周期已给过分，False 代表可触发给分)
        # 初始时设为 True，防止机器人出生瞬间在原地不迈步直接吃分
        self.foothold_awarded = torch.ones((self.num_envs, 4), device=self.device, dtype=torch.bool)
        self.base_pos_awarded = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)

        # 每条腿"正在飞向、并将在触地时落到"的目标世界坐标。
        # optimal_footholds 对支撑腿表示"下一次触地"的位置 (一整个步幅之前方)，
        # 在触地瞬间会被重新规划成再下一次触地的目标；若奖励直接用它，
        # 触地时永远对不上，跟踪奖励会完全失效。因此在摆动相持续记录该腿的
        # 落点，触地后冻结，供落脚点跟踪奖励在触地瞬间比对。
        self.landing_footholds = torch.zeros((self.num_envs, 4, 3), device=self.device, dtype=torch.float32)

        # 6. 名义常数配置 (Unitree Go2)
        self.nominal_foot_b = torch.tensor([
            [0.1934, 0.142, -0.278],   # FL
            [0.1934, -0.142, -0.278],  # FR
            [-0.1934, 0.142, -0.278],  # RL
            [-0.1934, -0.142, -0.278], # RR
        ], device=self.device, dtype=torch.float32)

        self.nominal_joint_pos = torch.tensor([
            -0.1, 0.9, -1.8,  # FL
             0.1, 0.9, -1.8,  # FR
            -0.1, 0.9, -1.8,  # RL
             0.1, 0.9, -1.8,  # RR
        ], device=self.device, dtype=torch.float32).repeat(self.num_envs, 1)

        self.nominal_planar_footholds = self.nominal_foot_b[:, :2].flatten().repeat(self.num_envs, 1)
        self.planar_footholds.copy_(self.nominal_planar_footholds)
        self.desired_joint_pos.copy_(self.nominal_joint_pos)

    @property
    def command(self) -> torch.Tensor:
        """返回 4 腿期望落脚点 [num_envs, 12] (FL_xyz, FR_xyz, RL_xyz, RR_xyz)。"""
        return self._command

    @property
    def desired_contact_state(self) -> torch.Tensor:
        """返回每条腿的接触计划 [num_envs, 4] (1: Stance, 0: Swing)。"""
        return self.gait_scheduler.contact_states

    @property
    def time_left_in_phase(self) -> torch.Tensor:
        """返回每条腿当前步态相的剩余时间 [num_envs, 4] (秒)。"""
        return self.gait_scheduler.time_left

    @property
    def gait_phase(self) -> torch.Tensor:
        """返回每条腿的归一化步态相位 [num_envs, 4] (0.0 到 1.0)。"""
        return self.gait_scheduler.phases

    def _init_reset_maps(
        self,
        hit_pos_w: torch.Tensor | None,
        hits_ok: torch.Tensor | None,
    ) -> None:
        """用当帧观测重置刚 reset 环境的高程图。

        地高取本帧所有有效命中点 z 的中位数 (平地≈0，台阶上≈当前台阶面)；
        没有命中点时才退回 "当前机高 - h_des"。这里读到的机器人位姿
        已经是 sim.forward() 之后的新位姿。
        """
        init = self._map_needs_init
        if not bool(init.any()):
            return

        fallback = self.robot.data.root_link_pos_w[:, 2] - self.cfg.optimizer_cfg.h_des
        if hit_pos_w is not None and hits_ok is not None:
            z = hit_pos_w[..., 2].masked_fill(~hits_ok, float("nan"))
            median_z = z.nanmedian(dim=1).values
            # 中位数离"机高 - h_des"太远说明本帧命中大多是噪声，退回机高估计
            sane = torch.isfinite(median_z) & ((median_z - fallback).abs() < 0.5)
            ground = torch.where(sane, median_z, fallback)
        else:
            ground = fallback

        self.ground_default[init] = ground[init]
        self.h_raw[init] = ground[init].view(-1, 1, 1, 1)
        self.h_valid[init] = 0.0
        self._pose_valid[init] = False
        self._map_needs_init[init] = False

    def _roll_map_odometry(self) -> None:
        """按机身位姿增量把高程图内容搬回"世界不动"的位置。

        高程图是机身偏航系下的栅格，机器人前进/转向后旧格子的索引含义已经变了。
        这里用一次双线性重采样把内容平移 (Δx, Δy) 并旋转 Δyaw：
            world_point_body_current = R(-Δyaw) * world_point_body_last - d_b
        等价于对新图上每个格点 p 去旧图采样 R(Δyaw) * (p + d_b)。
        """
        base_xy = self.robot.data.root_link_pos_w[:, :2]
        yaw = self.robot.data.heading_w

        delta_w = base_xy - self._last_base_xy
        dyaw = yaw - self._last_base_yaw

        cos_y = torch.cos(yaw)
        sin_y = torch.sin(yaw)
        dx_b = cos_y * delta_w[:, 0] + sin_y * delta_w[:, 1]
        dy_b = -sin_y * delta_w[:, 0] + cos_y * delta_w[:, 1]

        # 刚 reset 的环境位姿是瞬移来的，本步不做滚动
        has_pose = self._pose_valid
        dx_b = torch.where(has_pose, dx_b, torch.zeros_like(dx_b))
        dy_b = torch.where(has_pose, dy_b, torch.zeros_like(dy_b))
        dyaw = torch.where(has_pose, dyaw, torch.zeros_like(dyaw))

        self._last_base_xy.copy_(base_xy)
        self._last_base_yaw.copy_(yaw)
        self._pose_valid.fill_(True)

        if bool((dx_b.abs().max() < 1e-5) and (dy_b.abs().max() < 1e-5) and (dyaw.abs().max() < 1e-5)):
            return

        # 输出栅格中心坐标 (机身系)，[1, nx, 1] / [1, 1, ny]
        xs = self.x_min + torch.arange(self.nx, device=self.device, dtype=torch.float32) * self.cell_size
        ys = self.y_min + torch.arange(self.ny, device=self.device, dtype=torch.float32) * self.cell_size
        px = xs.view(1, self.nx, 1)
        py = ys.view(1, 1, self.ny)

        c = torch.cos(dyaw).view(-1, 1, 1)
        s = torch.sin(dyaw).view(-1, 1, 1)
        dx = dx_b.view(-1, 1, 1)
        dy = dy_b.view(-1, 1, 1)

        # 旧机身系下的采样点 p_prev = R(dyaw) @ (p_cur + d_b)
        src_x = c * (px + dx) - s * (py + dy)
        src_y = s * (px + dx) + c * (py + dy)

        # align_corners=True: 归一化坐标 -1/+1 对应首/末格中心
        norm_x = 2.0 * (src_x - self.x_min) / (self.cell_size * max(self.nx - 1, 1)) - 1.0
        norm_y = 2.0 * (src_y - self.y_min) / (self.cell_size * max(self.ny - 1, 1)) - 1.0
        # h_raw 布局 [B, 1, nx, ny]: H=nx(最后一维的前一维), W=ny → grid 分量 0 沿 W=y，分量 1 沿 H=x
        grid = torch.stack([norm_y, norm_x], dim=-1).view(-1, self.nx * self.ny, 1, 2)

        shifted_h = F.grid_sample(self.h_raw, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        shifted_v = F.grid_sample(self.h_valid, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        shifted_h = shifted_h.view(-1, 1, self.nx, self.ny)
        shifted_v = (shifted_v.view(-1, 1, self.nx, self.ny) > 0.5).to(self.h_raw.dtype)
        # 插值到空白区域的结果丢弃，改用地面默认高度
        self.h_raw = torch.where(shifted_v > 0.5, shifted_h, self.ground_default.view(-1, 1, 1, 1))
        self.h_valid = shifted_v

    def _deposit_sensor_hits(self) -> None:
        """将前向雷达点云与足底扫描点投影到本地高程图张量中 (纯 GPU 索引赋值，<0.02ms)。"""
        sensors = self._env.scene.sensors
        all_hits = []
        all_ok = []

        for sensor_name in (self.cfg.sensor_name, getattr(self.cfg, "foot_sensor_name", None)):
            if not sensor_name or sensor_name not in sensors:
                continue
            sensor = sensors[sensor_name]
            sensor_dists = sensor.data.distances
            max_dist = float(getattr(sensor.cfg, "max_distance", 5.0))
            all_hits.append(sensor.data.hit_pos_w)
            # 打空射线并不总是被标成 -1：warp 会把距离退化成略小于 max_distance 的
            # 饱和值，命中点落在射线末端。足底垂直射线打空时就会得到 z≈-1.0 m 的
            # 假地面 (脚陷进地形时尤其常见)，直接写进地图会让最优点"钻到地下"。
            all_ok.append((sensor_dists > 0.0) & (sensor_dists < 0.95 * max_dist))

        if all_hits:
            hit_pos_w = torch.cat(all_hits, dim=1)   # [num_envs, N_total, 3]
            hits_ok = torch.cat(all_ok, dim=1)        # [num_envs, N_total]
        else:
            hit_pos_w = None
            hits_ok = None

        # 1. 刚 reset 的环境：用当帧命中点的中位数估计地面 (此时机器人位姿已刷新)
        self._init_reset_maps(hit_pos_w, hits_ok)

        # 2. 先把旧内容按里程计滚动，再写入本帧观测，这样新观测总是覆盖在正确的位置上
        self._roll_map_odometry()

        if hit_pos_w is None:
            return

        base_pos_w = self.robot.data.root_link_pos_w
        base_yaw = self.robot.data.heading_w

        rel_w = hit_pos_w - base_pos_w.unsqueeze(1)
        cos_y = torch.cos(base_yaw).unsqueeze(1)
        sin_y = torch.sin(base_yaw).unsqueeze(1)

        rel_x_b = cos_y * rel_w[..., 0] + sin_y * rel_w[..., 1]
        rel_y_b = -sin_y * rel_w[..., 0] + cos_y * rel_w[..., 1]
        rel_z_w = hit_pos_w[..., 2]

        idx_x = torch.round((rel_x_b - self.x_min) / self.cell_size).long()
        idx_y = torch.round((rel_y_b - self.y_min) / self.cell_size).long()

        # 第二道防线：地图只覆盖机身附近约 2 m，真实地面不会离当前地高差 0.6 m 以上
        z_lo = (self.ground_default - 0.6).unsqueeze(1)
        z_hi = (self.ground_default + 0.6).unsqueeze(1)
        plausible_z = (rel_z_w > z_lo) & (rel_z_w < z_hi)

        valid = (
            hits_ok & plausible_z &
            (idx_x >= 0) & (idx_x < self.nx) &
            (idx_y >= 0) & (idx_y < self.ny)
        )

        if valid.any():
            b_idx = torch.arange(self.num_envs, device=self.device).unsqueeze(1).expand_as(idx_x)
            self.h_raw[b_idx[valid], 0, idx_x[valid], idx_y[valid]] = rel_z_w[valid]
            self.h_valid[b_idx[valid], 0, idx_x[valid], idx_y[valid]] = 1.0

        # 缓慢更新"地面默认高度"：只用已观测格子的均值做低通，绝不跟随机身瞬时高度
        count = self.h_valid.sum(dim=(1, 2, 3))
        mean_h = (self.h_raw * self.h_valid).sum(dim=(1, 2, 3)) / count.clamp(min=1.0)
        enough = count > 100
        self.ground_default = torch.where(
            enough, 0.98 * self.ground_default + 0.02 * mean_h, self.ground_default
        )

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        """环境重置：重置步态时钟并标记地图待重建 (计时由基类 _resample 负责)。

        注意不要在这里读 root_link_pos_w 估地高：reset 回调早于 write_data_to_sim/forward，
        读到的是旧位姿。真正的地图初始化放在 _init_reset_maps 里。
        """
        if len(env_ids) == 0:
            return

        self.gait_scheduler.reset(env_ids)
        self.footholds_initialized[env_ids] = False

        # 运行期绝不再整体覆写地图，否则高程图退化成"跟随当前机高"
        self._map_needs_init[env_ids] = True
        self._pose_valid[env_ids] = False

        # 在 reset 时将已给分标记锁定为 True，杜绝出生瞬间在原地白领一次满分
        self.foothold_awarded[env_ids] = True
        self.base_pos_awarded[env_ids] = True

    def _desired_velocity(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回机体系期望速度 (vx [m/s], vy [m/s], wz [rad/s])。

        优先读取 ``cfg.velocity_command_name`` 指定的速度指令项 (默认 "twist")，
        该指令项直接提供期望速度 (UniformVelocityCommand 还带 Viser 摇杆滑条)。
        只有找不到速度指令项时，才回退到旧的 Goal_direction 航向误差推导。
        """
        term = self._velocity_cmd_term
        if term is None:
            name = getattr(self.cfg, "velocity_command_name", None)
            if name and name in self._env.command_manager.active_terms:
                candidate = self._env.command_manager.get_term(name)
                if hasattr(candidate, "command"):
                    term = candidate
                    self._velocity_cmd_term = candidate

        if term is not None:
            # UniformVelocityCommand / StairVelocityCommand 等都把机体系速度存在
            # vel_command_b [B, 3] 中；command 属性通常返回同一张量。
            cmd = getattr(term, "vel_command_b", None)
            if cmd is None:
                cmd = term.command
            cmd = cmd.to(device=self.device, dtype=torch.float32)
            if cmd.dim() == 1:
                cmd = cmd.unsqueeze(0).expand(self.num_envs, -1)
            return cmd[:, 0], cmd[:, 1], cmd[:, 2]

        # ---- 回退：由 Goal_direction 航向误差推导期望速度 (旧行为) ----
        target_speed = float(self.cfg.nominal_speed)
        if "Goal_direction" in self._env.command_manager.active_terms:
            goal_term = self._env.command_manager.get_term("Goal_direction")
            if hasattr(goal_term, "heading_error"):
                head_err = goal_term.heading_error[:, 0]
                return (
                    target_speed * torch.clamp(torch.cos(head_err), min=0.0),
                    target_speed * torch.sin(head_err),
                    torch.clamp(1.5 * head_err, min=-1.5, max=1.5),
                )

        return (
            torch.full((self.num_envs,), target_speed, device=self.device),
            torch.zeros(self.num_envs, device=self.device),
            torch.zeros(self.num_envs, device=self.device),
        )

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        """纯 PyTorch GPU 向量化的 TAMOLS 运动规划与落脚点更新。
        
        执行流程：
        1. 步进步态调度器时钟；
        2. 若开启 bypass_tamols，直接输出名义几何解并返回；
        3. 将传感器射线击中点直接投影并写入 GPU 局部高程图；
        4. 检查触发事件 (仅在触地 Touchdown、指令显著改变或复位时重解)；
        5. 调用 PyTorch GPU 2D 卷积 (Conv2D) 与形态学算子并行优化候选落脚点与基座轨迹；
        6. 全向量化更新机身局部坐标系平面落脚点与 command。
        """
        # ======================================================================
        # 1. 推进步态调度器时钟
        # ======================================================================
        prev_contacts = self.gait_scheduler.contact_states.clone()
        dt = float(self._env.step_dt)
        self.gait_scheduler.step(dt=dt, env_ids=env_ids)
        curr_contacts = self.gait_scheduler.contact_states

        # 当某条腿处于腾空摆动相 (curr_contacts < 0.5) 时，重置该腿到达给分标记为 False，使其可为即将到来的下一次触地给分
        is_swing = curr_contacts < 0.5
        self.foothold_awarded[is_swing] = False

        # 期望速度 (机体系, 来自速度指令项 twist；无该指令项时才回退到 Goal_direction)
        cmd_vx, cmd_vy, cmd_wz = self._desired_velocity()

        # ======================================================================
        # 2. 极速名义默认旁路模式 (若用户配置为 bypass_tamols)
        # ======================================================================
        if getattr(self.cfg, "bypass_tamols", False):
            base_pos_w = self.robot.data.root_link_pos_w
            base_yaw = self.robot.data.heading_w
            cos_y = torch.cos(base_yaw)
            sin_y = torch.sin(base_yaw)

            self.desired_base_pos_w[:, 0] = base_pos_w[:, 0]
            self.desired_base_pos_w[:, 1] = base_pos_w[:, 1]
            self.desired_base_pos_w[:, 2] = 0.278

            half_yaw = base_yaw * 0.5
            self.desired_base_quat_w[:, 0] = torch.cos(half_yaw)
            self.desired_base_quat_w[:, 1] = 0.0
            self.desired_base_quat_w[:, 2] = 0.0
            self.desired_base_quat_w[:, 3] = torch.sin(half_yaw)
            self.desired_base_euler[:, 0] = 0.0
            self.desired_base_euler[:, 1] = 0.0
            self.desired_base_euler[:, 2] = base_yaw

            self.desired_base_lin_vel_w[:, 0] = cos_y * cmd_vx - sin_y * cmd_vy
            self.desired_base_lin_vel_w[:, 1] = sin_y * cmd_vx + cos_y * cmd_vy
            self.desired_base_lin_vel_w[:, 2] = 0.0
            self.desired_base_ang_vel_w[:, 0] = 0.0
            self.desired_base_ang_vel_w[:, 1] = 0.0
            self.desired_base_ang_vel_w[:, 2] = cmd_wz

            self.desired_base_lin_acc_w.zero_()
            self.desired_base_ang_acc_w.zero_()

            foot_b = self.nominal_foot_b
            foot_b_xy = foot_b[:, :2].unsqueeze(0)
            f_w_x = base_pos_w[:, 0:1] + cos_y.unsqueeze(-1) * foot_b_xy[..., 0] - sin_y.unsqueeze(-1) * foot_b_xy[..., 1]
            f_w_y = base_pos_w[:, 1:2] + sin_y.unsqueeze(-1) * foot_b_xy[..., 0] + cos_y.unsqueeze(-1) * foot_b_xy[..., 1]
            f_w_z = base_pos_w[:, 2:3] + foot_b[:, 2].unsqueeze(0)

            self.optimal_footholds[..., 0] = f_w_x
            self.optimal_footholds[..., 1] = f_w_y
            self.optimal_footholds[..., 2] = f_w_z
            self.nominal_footholds.copy_(self.optimal_footholds)

            self.planar_footholds.copy_(self.nominal_planar_footholds)
            self.desired_joint_pos.copy_(self.nominal_joint_pos)
            self._command = self.optimal_footholds.reshape(self.num_envs, 12)
            self.landing_footholds.copy_(self.optimal_footholds)
            self.footholds_initialized[:] = True
            return

        # ======================================================================
        # 3. 将雷达点云写入 GPU 局部高程图
        # ======================================================================
        self._deposit_sensor_hits()

        # ======================================================================
        # 4. 判断触发事件 (腿触地 / 指令突变 / 环境重置)
        # ======================================================================
        if env_ids is not None:
            replan_mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
            replan_mask[env_ids] = True
        else:
            replan_mask = ~self.footholds_initialized
            if self.cfg.replan_on_touchdown:
                touchdown = (prev_contacts < 0.5) & (curr_contacts > 0.5)
                replan_mask |= torch.any(touchdown, dim=-1)

            if self.cfg.replan_on_command_change:
                # 期望速度突变 (vx/vy/wz 任一分量变化超过阈值) 时重新规划
                cur_cmd_vel = torch.stack([cmd_vx, cmd_vy, cmd_wz], dim=-1)
                cmd_changed = (
                    (cur_cmd_vel - self.last_cmd_vel).abs().amax(dim=-1)
                    > self.cfg.command_change_threshold
                )
                replan_mask |= cmd_changed
                self.last_cmd_vel[replan_mask] = cur_cmd_vel[replan_mask]

        # ======================================================================
        # 5. 执行 PyTorch GPU 2D 卷积与落脚点最优化规划
        # ======================================================================
        if replan_mask.any():
            # 为新规划步骤重置基座到达给分标记
            self.base_pos_awarded[replan_mask] = False

            cmd_vel_b = torch.stack([cmd_vx, cmd_vy], dim=-1)

            # 每条腿距其"下次触地"的时间:
            #   摆动相 → time_left 就是剩余摆动时间;
            #   支撑相 → time_left 是剩余支撑时间，之后还要再经历整个摆动相。
            # 标称落脚点目标是"下次触地后半个支撑相"时的髋部位置，所以前馈提前时间
            # 必须包含这段等待时间；否则机器人以 v 前进时，目标会系统性地落在身后
            # (原来固定用 0.5*支撑相 = 0.15s，而实际要等 0.3~0.6s)。
            gait = self.gait_scheduler
            stride = float(self.cfg.gait_cfg.stride_duration)
            duty = float(self.cfg.gait_cfg.duty_factor)
            stance_time = stride * duty
            swing_time = stride * (1.0 - duty)
            time_to_touchdown = torch.where(
                gait.contact_states < 0.5,
                gait.time_left,
                gait.time_left + swing_time,
            )
            lead_time = time_to_touchdown + 0.5 * stance_time

            # a. 标称落脚点 (Raibert 启发式 + 旋转补偿)
            nom_footholds = self.torch_planner.foothold_planner.compute_nominal_footholds(
                base_pos_w=self.robot.data.root_link_pos_w,
                base_yaw=self.robot.data.heading_w,
                base_vel_w=self.robot.data.root_link_lin_vel_w,
                cmd_vel_b=cmd_vel_b,
                base_ang_vel_w=self.robot.data.root_link_ang_vel_w,
                cmd_yaw_rate=cmd_wz,
                stance_time=stance_time,
                lead_time=lead_time,
            )

            # b. GPU 2D 卷积与形态学高程图处理 (h_s1, h_s2, gradients)
            h_s1, h_s2, grads = self.map_processor(self.h_raw)
            gx, gy = grads["h"]
            gx_s1, gy_s1 = grads["h_s1"]
            grad_sq_raw = gx**2 + gy**2
            grad_sq_s1 = gx_s1**2 + gy_s1**2

            terrain_maps = torch.cat([self.h_raw, h_s2, grad_sq_raw, grad_sq_s1], dim=1)  # [num_envs, 4, Nx, Ny]

            # c. GPU 落脚点优化、自适应基座姿态与解析逆运动学
            plan_res = self.torch_planner.plan_motion(
                nominal_footholds_w=nom_footholds,
                base_pos_w=self.robot.data.root_link_pos_w,
                base_yaw=self.robot.data.heading_w,
                base_vel_w=self.robot.data.root_link_lin_vel_w,
                cmd_vel_b=cmd_vel_b,
                base_ang_vel_w=self.robot.data.root_link_ang_vel_w,
                cmd_yaw_rate=cmd_wz,
                terrain_maps=terrain_maps,
                lead_time=lead_time,
            )

            # d. 仅更新发生触地或指令改变的环境 (其余环境落脚点世界坐标保持稳定)
            self.optimal_footholds[replan_mask] = plan_res["optimal_footholds_w"][replan_mask]
            self.nominal_footholds[replan_mask] = nom_footholds[replan_mask]
            self.desired_base_pos_w[replan_mask] = plan_res["desired_base_pos_w"][replan_mask]
            self.desired_base_quat_w[replan_mask] = plan_res["desired_base_quat_w"][replan_mask]
            self.desired_base_euler[replan_mask] = plan_res["desired_base_euler"][replan_mask]
            self.desired_base_lin_vel_w[replan_mask] = plan_res["desired_base_lin_vel_w"][replan_mask]
            self.desired_base_ang_vel_w[replan_mask] = plan_res["desired_base_ang_vel_w"][replan_mask]
            self.desired_base_lin_acc_w[replan_mask] = plan_res["desired_base_lin_acc_w"][replan_mask]
            self.desired_base_ang_acc_w[replan_mask] = plan_res["desired_base_ang_acc_w"][replan_mask]
            self.desired_joint_pos[replan_mask] = plan_res["desired_joint_pos"][replan_mask]
            self.footholds_initialized[replan_mask] = True

        # ======================================================================
        # 6. 全环境向量化更新机身航向坐标系平面落脚点与 command (耗时 < 0.01ms)
        # ======================================================================
        all_pos_w = self.robot.data.root_link_pos_w[:, :2]
        all_yaw = self.robot.data.heading_w
        cos_y = torch.cos(all_yaw).unsqueeze(-1)
        sin_y = torch.sin(all_yaw).unsqueeze(-1)

        rel_w = self.optimal_footholds[:, :, :2] - all_pos_w.unsqueeze(1)
        p_b_x = cos_y * rel_w[..., 0] + sin_y * rel_w[..., 1]
        p_b_y = -sin_y * rel_w[..., 0] + cos_y * rel_w[..., 1]

        self.planar_footholds = torch.stack([p_b_x, p_b_y], dim=-1).flatten(start_dim=1)
        self._command = self.optimal_footholds.reshape(self.num_envs, 12)

        # ======================================================================
        # 7. 记录每条腿的"落点目标" (用于触地瞬间的落脚点跟踪奖励)
        # ======================================================================
        is_swing_now = curr_contacts < 0.5
        if env_ids is not None:
            # 复位环境先用当前规划初始化，避免沿用上一回合的旧值
            self.landing_footholds[env_ids] = self.optimal_footholds[env_ids]
        # 摆动腿: 规划目标就是它正在飞向的落点，持续跟随最新的重规划结果;
        # 支撑腿: 保持摆动相结束时冻结的落点不变。
        self.landing_footholds[is_swing_now] = self.optimal_footholds[is_swing_now]

    def _update_metrics(self) -> None:
        pass

    def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
        """在 Viser 3D 视图中渲染 TAMOLS 优化落脚点小红点及期望基座标杆。"""
        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return

        for env_id in env_indices:
            # 绘制 4 腿期望落脚点 (小红点)
            for leg_i, foot_name in enumerate(FOOT_NAMES):
                pt = self.optimal_footholds[env_id, leg_i]
                if isinstance(pt, torch.Tensor):
                    pt = pt.detach().cpu().numpy()
                visualizer.add_sphere(
                    center=pt,
                    radius=self.cfg.dot_radius,
                    color=self.cfg.dot_color,
                    label=f"tamols_opt_{foot_name}_{env_id}",
                )

                if self.cfg.show_nominal_dots:
                    nom_pt = self.nominal_footholds[env_id, leg_i]
                    if isinstance(nom_pt, torch.Tensor):
                        nom_pt = nom_pt.detach().cpu().numpy()
                    visualizer.add_sphere(
                        center=nom_pt,
                        radius=self.cfg.dot_radius * 0.75,
                        color=self.cfg.nominal_dot_color,
                        label=f"tamols_nom_{foot_name}_{env_id}",
                    )


# 保留原有类名以兼容已有代码导入
TamolsFootholdCommand = TamolsMotionCommand
