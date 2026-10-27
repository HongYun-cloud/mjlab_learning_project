from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import dataclass, field

import math

import numpy as np
import torch

from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand, UniformVelocityCommandCfg
from mjlab.managers.command_manager import CommandTermCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


class StairVelocityCommand(UniformVelocityCommand):
    """楼梯速度命令。

    上楼阶段 (going_up=True)
    -----------------------
    每个 step 重新计算 world-frame 中 "机器人 → env中心" 方向，
    投影到 body frame 得到 [vx_b, vy_b]，乘以 forward_speed。
    这样无论机器人朝向如何，命令始终拉向中心。

    下楼阶段 (going_up=False)
    -----------------------
    切换到下楼时一次性采样一个随机的 world-frame 方向（远离中心的半球内），
    之后每 step 将该 world 方向投影到 body frame 得到命令。
    """

    cfg: StairVelocityCommandCfg

    def __init__(self, cfg: StairVelocityCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)

        self.going_up = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._step_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # 下楼时的 world-frame 目标方向单位向量 (N, 2)
        self._down_dir_w = torch.zeros(self.num_envs, 2, device=self.device)

        # -- north_peak 模式的到达状态 (供课程/终止项消费) --
        self.arrived_target = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        """当前一步是否处于目标点半径 (center_threshold) 内。"""
        self.peak_reached_episode = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        """本 episode 内是否曾到达过目标点 (到达后保持 True, reset 时清零)。"""
        self.peak_reached_now = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        """本步"首次到达"上升沿 (到达当步 True 一次; 供到达稀疏奖励使用)。"""

    # ------------------------------------------------------------------
    # episode reset
    # ------------------------------------------------------------------
    def _resample_command(self, env_ids: torch.Tensor) -> None:
        # 禁用父类所有特殊模式
        self.is_heading_env[env_ids] = False
        self.is_standing_env[env_ids] = False
        self.is_world_env[env_ids] = False
        self.is_forward_env[env_ids] = False

        # 重置为上楼
        self.going_up[env_ids] = True
        self._step_count[env_ids] = 0
        self.arrived_target[env_ids] = False
        self.peak_reached_episode[env_ids] = False
        self.peak_reached_now[env_ids] = False
        self._record_reset_reason(env_ids)

        # 初始命令占位（compute 里会立刻覆盖）
        self.vel_command_b[env_ids, 0] = self.cfg.forward_speed
        self.vel_command_b[env_ids, 1] = 0.0
        self.vel_command_b[env_ids, 2] = 0.0

    # ------------------------------------------------------------------
    # 辅助：world→body 旋转（与父类 _update_command 完全一致的公式）
    # ------------------------------------------------------------------
    def _world_to_body(self, dir_w_x: torch.Tensor, dir_w_y: torch.Tensor,
                       heading: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """将 world-frame 2D 方向旋转到 body frame。"""
        cos_h = torch.cos(heading)
        sin_h = torch.sin(heading)
        vx_b =  cos_h * dir_w_x + sin_h * dir_w_y
        vy_b = -sin_h * dir_w_x + cos_h * dir_w_y
        return vx_b, vy_b

    # ------------------------------------------------------------------
    # Viser 面板: reset/终止条件状态 (仅 viewer, 训练时零开销)
    # ------------------------------------------------------------------
    def create_debug_vis_gui(self, server, on_change=None) -> None:
        """官方 CommandTerm GUI 钩子: 在 viewer 侧栏加终止条件面板。"""
        super().create_debug_vis_gui(server, on_change)
        with server.gui.add_folder("Reset / 终止条件"):
            self._term_status_html = server.gui.add_html("等待仿真…")
            self._reset_reason_html = server.gui.add_html("上次 reset: —")

    def _update_reset_gui(self) -> None:
        """刷新终止条件面板 (每 10 控制步调用一次; 未建 GUI 时早退)。"""
        html = getattr(self, "_term_status_html", None)
        if html is None:
            return
        tm = self._env.termination_manager
        rows = []
        for name in list(getattr(tm, "active_terms", [])):
            try:
                val = tm.get_term(name)
            except Exception:
                continue
            if torch.is_tensor(val) and val.numel():
                fired = bool(val[0])
            else:
                fired = bool(val)
            rows.append(f"{'🔴' if fired else '⚪'} <code>{name}</code>")
        html.content = "<br>".join(rows) if rows else "(无终止条件)"

    def _record_reset_reason(self, env_ids: torch.Tensor) -> None:
        """reset 时记录触发原因 (此时 termination 触发值尚未被清)。"""
        hits = []
        try:
            tm = self._env.termination_manager
            for name in list(getattr(tm, "active_terms", [])):
                val = tm.get_term(name)
                if torch.is_tensor(val) and val.numel() and bool(val[env_ids].any()):
                    hits.append(name)
        except Exception:
            pass
        self._last_reset_reason = " + ".join(hits) if hits else "手动 / 初始 reset"
        el = getattr(self, "_reset_reason_html", None)
        if el is not None:
            el.content = f"上次 reset: <b>{self._last_reset_reason}</b>"

    def _gui_tick(self) -> None:
        """终止条件面板节流刷新 (每 10 控制步一次 = 5 Hz; 训练态无 GUI 早退)。"""
        self._gui_frame = getattr(self, "_gui_frame", 0) + 1
        if self._gui_frame % 10 == 0:
            self._update_reset_gui()

    # ------------------------------------------------------------------
    # Viser 绘制: 终点标识 (viewer Debug Viz 勾选后生效, 训练零开销)
    # ------------------------------------------------------------------
    def _debug_vis_impl(self, visualizer) -> None:
        """终点标识, 用于核对目标点是否设对。

        - 金色大球 = 终点 (north_peak: env_origin + target_offset_w;
          center 模式: env_origin), 竖串小球 = 目标立柱 (远处可见);
        - 绿色小点环 = 到达半径 (center_threshold) 边界;
        - 到达后 (arrived_target) 金球变绿;
        - center 模式下楼阶段: 机器人上方红箭头 = 随机下楼方向.
        """
        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return

        env_origins = self._env.scene.env_origins.detach().cpu().numpy()
        arrived = self.arrived_target.detach().cpu().numpy()
        radius = float(self.cfg.center_threshold)

        for env_id in env_indices:
            origin = env_origins[env_id]
            # 线速度命令(青色) / 实际速度(红色) / 当前与目标朝向(蓝/绿色)
            # 同时显示, 用来确认“转圈”是 command 还是策略/机身实际运动.
            base = self.robot.data.root_link_pos_w[env_id].detach().cpu().numpy()
            heading = float(self.robot.data.heading_w[env_id].item())
            cmd_b = self.vel_command_b[env_id, :2].detach().cpu().numpy()
            c, s = math.cos(heading), math.sin(heading)
            cmd_w = np.array(
                [c * cmd_b[0] - s * cmd_b[1], s * cmd_b[0] + c * cmd_b[1]]
            )
            arrow_start = base + np.array([0.0, 0.0, 0.18])
            if np.linalg.norm(cmd_w) > 1e-3:
                visualizer.add_arrow(
                    start=arrow_start,
                    end=arrow_start + np.array([cmd_w[0], cmd_w[1], 0.0]) * 0.8,
                    color=(0.0, 0.85, 1.0, 0.95),
                    width=0.025,
                    label=f"stair_cmd_vel_{env_id}",
                )

            actual_w = (
                self.robot.data.root_link_lin_vel_w[env_id, :2]
                .detach().cpu().numpy()
            )
            if np.linalg.norm(actual_w) > 0.015:
                visualizer.add_arrow(
                    start=arrow_start,
                    end=arrow_start + np.array([actual_w[0], actual_w[1], 0.0]) * 0.8,
                    color=(1.0, 0.15, 0.15, 0.95),
                    width=0.018,
                    label=f"stair_actual_vel_{env_id}",
                )

            # 当前朝向蓝色, 目标朝向绿色; 两箭头方向不同即可看出航向误差.
            visualizer.add_arrow(
                start=arrow_start + np.array([0.0, 0.0, 0.06]),
                end=arrow_start + np.array([0.35 * c, 0.35 * s, 0.06]),
                color=(0.2, 0.45, 1.0, 0.85),
                width=0.012,
                label=f"stair_heading_{env_id}",
            )
            if self.cfg.face_heading is not None:
                desired = float(self.cfg.face_heading)
                visualizer.add_arrow(
                    start=arrow_start + np.array([0.0, 0.0, 0.12]),
                    end=arrow_start + np.array([
                        0.35 * math.cos(desired),
                        0.35 * math.sin(desired),
                        0.12,
                    ]),
                    color=(0.2, 1.0, 0.35, 0.9),
                    width=0.012,
                    label=f"stair_heading_target_{env_id}",
                )

            if self.cfg.target_mode == "north_peak":
                tx, ty = origin[0] + self.cfg.target_offset_w[0], origin[1] + self.cfg.target_offset_w[1]
                # 标识高度 = 台阶最高面。倒金字塔地形 (BoxInvertedPyramidStairs)
                # 的 env_origin 钉在**凹底出生面** (随难度降至 -1.8 m), 而
                # 周围 border/最高台阶面恒在世界 z=0 —— 用 origin.z 画会把
                # 标识沉到凹底。目标 XY 处的台阶面本身略低于 0 (L9 时
                # -0.30), 画在 z=0 便于与 border 平面对照。
                z = 0.0
            else:
                tx, ty = origin[0], origin[1]
                z = float(origin[2])     # center 模式终点即凹底表面
            goal_color = (
                (0.2, 1.0, 0.3, 0.95) if bool(arrived[env_id]) else (1.0, 0.85, 0.0, 0.95)
            )
            visualizer.add_sphere(
                center=np.array([tx, ty, z + 0.08]),
                radius=0.12,
                color=goal_color,
                label=f"goal_pt_{env_id}",
            )
            for h in (0.3, 0.6, 0.9):     # 目标立柱
                visualizer.add_sphere(
                    center=np.array([tx, ty, z + h]),
                    radius=0.05,
                    color=goal_color,
                    label=f"goal_post_{h}_{env_id}",
                )
            for k in range(8):            # 到达半径圈
                ang = 2.0 * math.pi * k / 8.0
                visualizer.add_sphere(
                    center=np.array([
                        tx + radius * math.cos(ang),
                        ty + radius * math.sin(ang),
                        z + 0.03,
                    ]),
                    radius=0.04,
                    color=(0.2, 1.0, 0.3, 0.55),
                    label=f"goal_ring_{k}_{env_id}",
                )
            if self.cfg.target_mode != "north_peak" and bool((~self.going_up)[env_id]):
                d = self._down_dir_w[env_id].detach().cpu().numpy()
                rp = self.robot.data.root_link_pos_w[env_id, :2].detach().cpu().numpy()
                zr = float(self.robot.data.root_link_pos_w[env_id, 2])
                visualizer.add_arrow(
                    start=np.array([rp[0], rp[1], zr + 0.2]),
                    end=np.array([rp[0] + d[0], rp[1] + d[1], zr + 0.2]),
                    color=(1.0, 0.1, 0.1, 0.9),
                    width=0.02,
                    label=f"down_dir_{env_id}",
                )

    # ------------------------------------------------------------------
    # 每 step 计算命令
    # ------------------------------------------------------------------
    def compute(self, dt: float | torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        # 父类：处理 resampling 计时、metrics 等
        # is_heading_env / is_standing_env / is_world_env 全 False → 不会覆盖 vel_command_b
        super().compute(dt, env_ids)

        # ---- 位置信息 ----
        robot_pos_w = self.robot.data.root_link_pos_w          # (N, 3)
        env_origins = self._env.scene.env_origins              # (N, 3)
        rel_pos = robot_pos_w[:, :2] - env_origins[:, :2]     # (N, 2) 机器人 - 中心
        dist = torch.norm(rel_pos, dim=-1)                     # (N,)

        # ---- north_peak 模式: 恒定目标点 + 恒定朝向 -----------------------
        # 目标 = patch 中心 + target_offset_w (世界系水平偏移; +y 为北,
        # 默认 2.85 m ≈ 倒金字塔台阶区外缘 = 北方向台阶最高点)。
        # 速度方向恒指向该点; 期望朝向恒为 face_heading (P 控制对齐,
        # 受 ranges.ang_vel_z 限幅); forward_only=True 时不因目标横向偏差
        # 产生侧移. 到达 (< center_threshold) 后 vx/vy/wz 命令归零; DTC
        # 可继续用 gait scheduler 做原地 trot, 而不是强制冻结步态.
        if self.cfg.target_mode == "north_peak":
            target_w = env_origins[:, :2] + torch.tensor(
                self.cfg.target_offset_w, device=self.device, dtype=torch.float32
            ).unsqueeze(0)
            to_target = target_w - robot_pos_w[:, :2]          # (N, 2)
            dist_t = torch.norm(to_target, dim=-1, keepdim=True).clamp(min=0.01)
            dir_w = to_target / dist_t

            heading = self.robot.data.heading_w
            vx_b, vy_b = self._world_to_body(dir_w[:, 0], dir_w[:, 1], heading)
            if self.cfg.forward_only:
                # Fixed-heading stair/platform traverses should advance along
                # body X. Projecting a slightly offset target creates lateral
                # commands and can make the policy crab-walk.
                self.vel_command_b[:, 0] = self.cfg.forward_speed
                self.vel_command_b[:, 1] = 0.0
            else:
                self.vel_command_b[:, 0] = self.cfg.forward_speed * vx_b
                self.vel_command_b[:, 1] = self.cfg.forward_speed * vy_b

            if self.cfg.stop_at_target:
                arrived = dist_t.squeeze(-1) < self.cfg.center_threshold
                self.vel_command_b[arrived, 0] = 0.0
                self.vel_command_b[arrived, 1] = 0.0

            # 到达状态缓存 (课程升降级 / 到顶终止判定 / 到达奖励用):
            # arrived_target = 当前是否在峰值半径内; peak_reached_episode =
            # 本 episode 是否曾到达; peak_reached_now = 本步首次到达上升沿。
            self.arrived_target = dist_t.squeeze(-1) < self.cfg.center_threshold
            self.peak_reached_now = self.arrived_target & ~self.peak_reached_episode
            self.peak_reached_episode |= self.arrived_target

            if self.cfg.face_heading is not None:
                yaw_err = self.cfg.face_heading - heading
                yaw_err = torch.atan2(torch.sin(yaw_err), torch.cos(yaw_err))
                wz = self.cfg.heading_control_stiffness * yaw_err
                lo, hi = self.cfg.ranges.ang_vel_z or (-1.0, 1.0)
                self.vel_command_b[:, 2] = wz.clamp(min=lo, max=hi)
                # 到终点后停掉平移, 但保留低增益、带死区的 heading hold.
                # 完全把 wz 置零会让策略看不到航向纠正目标; 原地 gait 的
                # 左右微小不对称会积累成持续转圈. 小误差进入死区后命令归零.
                if self.cfg.stop_at_target:
                    target_wz = (
                        float(self.cfg.target_heading_stiffness) * yaw_err
                    ).clamp(
                        min=-float(self.cfg.target_yaw_rate_limit),
                        max=float(self.cfg.target_yaw_rate_limit),
                    )
                    target_wz = torch.where(
                        yaw_err.abs() <= float(self.cfg.target_heading_deadband),
                        torch.zeros_like(target_wz),
                        target_wz,
                    )
                    self.vel_command_b[arrived, 2] = target_wz[arrived]

            self._gui_tick()
            return

        # ---- 步数 & 方向切换 ----
        ids = slice(None) if env_ids is None else env_ids
        self._step_count[ids] += 1

        at_center = dist[ids] < self.cfg.center_threshold
        timed_out = self._step_count[ids] >= self.cfg.switch_steps
        should_switch = timed_out | (self.going_up[ids] & at_center)

        if isinstance(ids, torch.Tensor):
            switch_ids = ids[should_switch]
        else:
            switch_ids = should_switch.nonzero(as_tuple=False).flatten()

        if len(switch_ids) > 0:
            self.going_up[switch_ids] = ~self.going_up[switch_ids]
            self._step_count[switch_ids] = 0

            # 新切换到下楼的 env：采样随机 world 方向
            new_down_mask = ~self.going_up[switch_ids]
            new_down_ids = switch_ids[new_down_mask]
            if len(new_down_ids) > 0:
                angles = torch.empty(len(new_down_ids), device=self.device).uniform_(
                    0, 2 * math.pi
                )
                self._down_dir_w[new_down_ids, 0] = torch.cos(angles)
                self._down_dir_w[new_down_ids, 1] = torch.sin(angles)

        # ---- 上楼 env：速度指向中心 ----
        up_mask = self.going_up
        if up_mask.any():
            up_ids = up_mask.nonzero(as_tuple=False).flatten()

            # world-frame 方向：机器人 → 中心 = -rel_pos，归一化
            to_center = -rel_pos[up_ids]                         # (M, 2)
            to_center_norm = torch.norm(to_center, dim=-1, keepdim=True).clamp(min=0.01)
            dir_w = to_center / to_center_norm                   # 单位向量

            # world → body
            heading = self.robot.data.heading_w[up_ids]
            vx_b, vy_b = self._world_to_body(dir_w[:, 0], dir_w[:, 1], heading)

            self.vel_command_b[up_ids, 0] = self.cfg.forward_speed * vx_b
            self.vel_command_b[up_ids, 1] = self.cfg.forward_speed * vy_b
            self.vel_command_b[up_ids, 2] = 0.0

        # ---- 下楼 env：沿保存的随机 world 方向 ----
        down_mask = ~self.going_up
        if down_mask.any():
            down_ids = down_mask.nonzero(as_tuple=False).flatten()

            dir_w = self._down_dir_w[down_ids]                   # (M, 2)
            heading = self.robot.data.heading_w[down_ids]
            vx_b, vy_b = self._world_to_body(dir_w[:, 0], dir_w[:, 1], heading)

            self.vel_command_b[down_ids, 0] = self.cfg.forward_speed * vx_b
            self.vel_command_b[down_ids, 1] = self.cfg.forward_speed * vy_b
            self.vel_command_b[down_ids, 2] = 0.0

        self._gui_tick()
        # down_mask = ~self.going_up
        # if down_mask.any():
        #     down_ids = down_mask.nonzero(as_tuple=False).flatten()

        #     self.vel_command_b[down_ids, 0] = self.cfg.forward_speed
        #     self.vel_command_b[down_ids, 1] = 0.0
        #     self.vel_command_b[down_ids, 2] = 0.0


@dataclass(kw_only=True)
class StairVelocityCommandCfg(UniformVelocityCommandCfg):
    """楼梯场景速度命令配置。

    - forward_speed    : 命令速度大小 (m/s)
    - switch_steps     : 上楼超时后切换方向的步数
    - center_threshold : 距 env_origin 小于此距离视为"到顶"
    """
    forward_speed: float = 1.0
    switch_steps: int = 750
    center_threshold: float = 0.8

    target_mode: str = "center"
    """center: 速度指向 env_origin, 到达切随机方向下楼 (旧行为);
    north_peak: 速度恒指向 env_origin + target_offset_w (北方向台阶最高点),
    不再上/下楼切换。"""

    target_offset_w: tuple[float, float] = (0.0, 2.85)
    """north_peak 目标点相对 patch 中心的世界系水平偏移 (+y 为北)。
    倒金字塔 patch 半径 4 m、平台 3 m、台阶区外缘 ≈3 m, 取 2.85 留半格余量。"""

    face_heading: float | None = None
    """非 None 时期望朝向恒为该 yaw (rad, 世界系), wz 用 P 控制对齐并受
    ranges.ang_vel_z 限幅。north_peak + 朝北 = math.pi / 2。"""

    stop_at_target: bool = False
    """north_peak 模式: 距目标 < center_threshold 时线速度命令置零 (原地站立)。"""

    forward_only: bool = False
    """north_peak 模式固定朝向时只给 body-x 速度, 不因目标点横向误差产生侧移."""

    target_heading_stiffness: float = 0.6
    """到达 north_peak 后航向保持的比例增益 (比行进时 heading gain 更低)."""

    target_heading_deadband: float = 0.03
    """到达目标后的航向死区 (rad), 小于约 1.7° 时不再下发转向命令."""

    target_yaw_rate_limit: float = 0.25
    """到达目标后的最大 yaw 角速度命令 (rad/s), 避免原地过度转向."""

    ranges: UniformVelocityCommandCfg.Ranges = field(
        default_factory=lambda: UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.0, 1.0),
            heading=None,
        )
    )

    def build(self, env: ManagerBasedRlEnv) -> StairVelocityCommand:
        return StairVelocityCommand(self, env)

    def __post_init__(self):
        self.rel_standing_envs = 0.0
        self.rel_heading_envs  = 0.0
        self.rel_world_envs    = 0.0
        self.rel_forward_envs  = 0.0
        self.heading_command   = False
        self.init_velocity_prob = 0.0
        self.heading_control_stiffness = 2.0


def reached_stair_peak(env, command_name: str = "twist") -> torch.Tensor:
    """终止条件: 到达楼梯顶端 (north_peak 目标点 center_threshold 半径内)。

    配 TerminationTermCfg(time_out=True) 作为**成功终止**: 到顶即结束 episode
    并由课程项计入"到达 1 次"。命令项非 north_peak 模式时恒为 False。
    """
    cmd = env.command_manager.get_term(command_name)
    arrived = getattr(cmd, "arrived_target", None)
    if arrived is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return arrived.clone()
