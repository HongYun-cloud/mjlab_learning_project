from __future__ import annotations

import collections
from typing import TYPE_CHECKING, Callable, Any
from dataclasses import dataclass
import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.tasks.velocity.mdp.velocity_command import (
    UniformVelocityCommand,
    UniformVelocityCommandCfg,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer
    import viser


# 候选基本方位单位向量
CARDINAL_OFFSETS: dict[str, tuple[float, float]] = {
    "E": (1.0, 0.0),   # +X (东)
    "N": (0.0, 1.0),   # +Y (北)
    "W": (-1.0, 0.0),  # -X (西)
    "S": (0.0, -1.0),  # -Y (南)
}


# ==============================================================================
# 1. Goal Direction Command (目标方向命令)
# ==============================================================================

@dataclass(kw_only=True)
class GoalDirectionCommandCfg(CommandTermCfg):
    """目标行进方向命令配置。"""
    entity_name: str = "robot"
    target_distance: float = 3.2
    cardinal_directions: tuple[str, ...] = ("E", "N", "W", "S")
    resampling_time_range: tuple[float, float] = (10.0, 10.0)
    debug_vis: bool = True
    print_reset_reason: bool = False
    """是否在终端打印每次重置原因 (默认 False，关闭终端打印，仅在 View 界面可见)"""

    def build(self, env: ManagerBasedRlEnv) -> GoalDirectionCommand:
        return GoalDirectionCommand(self, env)


class GoalDirectionCommand(CommandTerm):
    """目标行进方向命令执行项。

    在倒金字塔地形重置时，从 (东/南/西/北) 边缘顶端采样目标点 r_goal，
    并在每步计算朝向该目标点的机身局部航向系单位方向向量：
        d_base = [cos(Delta psi), sin(Delta psi)]
    同时维护 target_pos 和 heading_error 供奖励函数与关联命令使用。
    """
    cfg: GoalDirectionCommandCfg

    def __init__(self, cfg: GoalDirectionCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.robot: Entity = env.scene[cfg.entity_name]

        # 候选方位单位向量表 (K, 2)
        candidate_list = [CARDINAL_OFFSETS[d] for d in cfg.cardinal_directions]
        self._candidate_offsets = torch.tensor(
            candidate_list, device=self.device, dtype=torch.float32
        )

        # 维护目标点在环境局部坐标系下的绝对坐标 (num_envs, 2)
        self.target_pos = torch.zeros((self.num_envs, 2), device=self.device)

        # 维护当前步机身航向系下的单位方向命令 (num_envs, 2)
        self._command = torch.zeros((self.num_envs, 2), device=self.device)

        # 维护当前航向误差角 (num_envs, 1)
        self.heading_error = torch.zeros((self.num_envs, 1), device=self.device)

        # 维护终止与重置信息
        self.last_reset_reason: list[str] = ["Ready (初始准备)"] * self.num_envs
        self.reset_count: list[int] = [0] * self.num_envs
        self.reset_stats: list[dict[str, int]] = [collections.defaultdict(int) for _ in range(self.num_envs)]
        self._term_info_html = None
        self._last_html_content: str = ""
        self._get_env_idx: Callable[[], int] | None = None

    @property
    def command(self) -> torch.Tensor:
        """返回当前步目标方向指令张量 (num_envs, 2)。"""
        return self._command

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> dict[str, float]:
        """重置环境时捕获终止原因并更新统计。"""
        if env_ids is None:
            ids_tensor = torch.arange(self.num_envs, device=self.device)
        elif isinstance(env_ids, slice):
            ids_tensor = torch.arange(self.num_envs, device=self.device)[env_ids]
        else:
            ids_tensor = env_ids

        self._record_termination_reasons(ids_tensor)
        return super().reset(env_ids)

    def _record_termination_reasons(self, ids_tensor: torch.Tensor) -> None:
        """从 termination_manager 中解析导致重置的具体终止项。"""
        term_map = {
            "reach_target": ("🎯", "到达目标点 (reach_target)"),
            "illegal_contact": ("🦵", "大腿触地 (illegal_contact)"),
            "fell_over": ("🔄", "机身翻倒 (fell_over)"),
            "base_too_low": ("⬇️", "机身过低 (base_too_low)"),
            "out_of_patch_bounds": ("🚧", "越界终止 (out_of_patch_bounds)"),
            "out_of_terrain_bounds": ("🚧", "地形出界 (out_of_terrain_bounds)"),
            "reached_edge": ("⛰️", "到达边缘 (reached_edge)"),
            "time_out": ("⏱️", "超时重置 (time_out)"),
        }

        has_term_mgr = hasattr(self._env, "termination_manager") and hasattr(
            self._env.termination_manager, "_term_dones"
        )
        term_dones = self._env.termination_manager._term_dones if has_term_mgr else {}

        for env_id in ids_tensor:
            idx = int(env_id.item())
            matched = []
            for term_name, dones in term_dones.items():
                if idx < len(dones) and dones[idx].item():
                    matched.append(term_name)

            if not matched:
                if (
                    hasattr(self._env, "reset_time_outs")
                    and idx < len(self._env.reset_time_outs)
                    and self._env.reset_time_outs[idx].item()
                ):
                    matched.append("time_out")
                elif (
                    hasattr(self._env, "reset_terminated")
                    and idx < len(self._env.reset_terminated)
                    and self._env.reset_terminated[idx].item()
                ):
                    matched.append("terminated")

            if matched:
                display_parts = [
                    term_map.get(m, ("⚠️", m))[0] + " " + term_map.get(m, ("⚠️", m))[1]
                    for m in matched
                ]
                reason_str = " + ".join(display_parts)
                for m in matched:
                    self.reset_stats[idx][m] += 1
            else:
                reason_str = "🚀 初始/手动重置 (manual_reset)"
                self.reset_stats[idx]["manual_or_init"] += 1

            self.last_reset_reason[idx] = reason_str
            self.reset_count[idx] += 1

            # 记录重置信息供 Viser/Viewer 界面渲染，终端输出默认关闭
            if getattr(self.cfg, "print_reset_reason", False):
                ep_step = (
                    int(self._env.episode_length_buf[idx].item())
                    if hasattr(self._env, "episode_length_buf")
                    else 0
                )
                print(f"[Env {idx} RESET #{self.reset_count[idx]}] 原因: {reason_str} | 步数: {ep_step}")

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        """重置时在四向台阶边缘采样目标点。"""
        if len(env_ids) == 0:
            return
        num_resets = len(env_ids)
        k = len(self._candidate_offsets)
        choice_idx = torch.randint(0, k, (num_resets,), device=self.device)
        chosen_offsets = self._candidate_offsets[choice_idx]  # (num_resets, 2)

        # 目标点 xy 坐标 = 候选单位向量 * 目标距离
        if hasattr(self, "target_distances"):
            dist = self.target_distances[env_ids].unsqueeze(-1)
            self.target_pos[env_ids] = chosen_offsets * dist
        else:
            self.target_pos[env_ids] = chosen_offsets * self.cfg.target_distance

        # 立即更新一次方向指令
        self._update_command(env_ids)

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        """每步计算指向目标点的机身航向系单位方向与航向偏差。"""
        ids = slice(None) if env_ids is None else env_ids

        # 机器人当前位置 (在局部 env frame 下)
        robot_pos_w = self.robot.data.root_link_pos_w[ids]
        env_origins = self._env.scene.env_origins[ids]
        rel_pos = robot_pos_w[:, :2] - env_origins[:, :2]  # (M, 2)

        # 目标相对向量: 目标点 - 机器人当前位置
        to_target = self.target_pos[ids] - rel_pos  # (M, 2)
        dist = torch.norm(to_target, dim=-1, keepdim=True).clamp(min=1e-4)
        dir_w = to_target / dist  # (M, 2)

        # 目标航向角与机身航向角
        target_yaw = torch.atan2(dir_w[:, 1], dir_w[:, 0])  # (M,)
        robot_yaw = self.robot.data.heading_w[ids]          # (M,)

        # 航向偏差角 Delta psi in [-pi, pi]
        yaw_diff = target_yaw - robot_yaw
        head_err = torch.atan2(torch.sin(yaw_diff), torch.cos(yaw_diff))  # (M,)
        self.heading_error[ids, 0] = head_err

        # 投影到机身局部航向系 (Yaw-aligned base frame)
        self._command[ids, 0] = torch.cos(head_err)
        self._command[ids, 1] = torch.sin(head_err)

    def _update_metrics(self) -> None:
        pass

    def _render_term_html(self, env_idx: int) -> None:
        """在 Viser 控制面板中渲染当前环境的重置与终止状态。"""
        if self._term_info_html is None:
            return
        if env_idx >= self.num_envs:
            env_idx = 0

        last_reason = self.last_reset_reason[env_idx]
        count = self.reset_count[env_idx]
        stats = self.reset_stats[env_idx]

        if "reach_target" in last_reason:
            badge_color = "#22c55e"
        elif "time_out" in last_reason:
            badge_color = "#eab308"
        elif "illegal_contact" in last_reason or "fell_over" in last_reason:
            badge_color = "#ef4444"
        elif "out_of" in last_reason:
            badge_color = "#f97316"
        else:
            badge_color = "#38bdf8"

        stats_lines = []
        for k, v in sorted(stats.items(), key=lambda x: -x[1]):
            stats_lines.append(f"• {k}: {v} 次")
        stats_str = "<br/>".join(stats_lines) if stats_lines else "暂无"

        ep_step = (
            int(self._env.episode_length_buf[env_idx].item())
            if hasattr(self._env, "episode_length_buf")
            else 0
        )

        html = f"""
        <div style="font-size: 0.88em; line-height: 1.45; padding: 8px 10px; background: rgba(15, 23, 42, 0.6); border-radius: 6px; border: 1px solid rgba(255,255,255,0.08);">
          <div style="font-weight: 600; color: #f1f5f9; margin-bottom: 4px;">
            环境 #{env_idx} 终止状态监控
          </div>
          <div style="margin-bottom: 4px;">
            <strong>上次终止原因:</strong><br/>
            <span style="color: {badge_color}; font-weight: bold;">{last_reason}</span>
          </div>
          <div><strong>当前步数:</strong> {ep_step} 步</div>
          <div><strong>总重置次数:</strong> {count} 次</div>
          <hr style="border: 0; border-top: 1px solid rgba(255,255,255,0.1); margin: 6px 0;"/>
          <div style="font-size: 0.82em; color: #94a3b8;">
            <strong>历史重置统计:</strong><br/>
            {stats_str}
          </div>
        </div>
        """
        if html != self._last_html_content:
            self._term_info_html.content = html
            self._last_html_content = html

    def create_gui(
        self,
        name: str,
        server: viser.ViserServer,
        get_env_idx: Callable[[], int],
        on_change: Callable[[], None] | None = None,
        request_action: Callable[[str, Any], None] | None = None,
    ) -> None:
        """在 Viser 控制面板中添加终止状态监控与地形难度交互控件。"""
        self._get_env_idx = get_env_idx

        # 1. 终止原因与重置监控面板
        with server.gui.add_folder("Termination & Reset (终止原因监控)"):
            self._term_info_html = server.gui.add_html("")
            self._render_term_html(get_env_idx())

        # 2. 地形难度选择
        terrain = getattr(self._env.scene, "terrain", None)
        if terrain is not None and getattr(terrain, "terrain_origins", None) is not None:
            num_levels = terrain.terrain_origins.shape[0]
            options = [
                f"Level {i} ({'Flat/0m' if i == 0 else f'0.{i*3:02d}m steps'})"
                for i in range(num_levels)
            ]
            options.append("Random (随机难度)")

            init_lvl = 0
            if hasattr(self._env, "specified_terrain_level") and self._env.specified_terrain_level is not None:
                if str(self._env.specified_terrain_level).lower() == "random":
                    init_lvl = len(options) - 1
                else:
                    try:
                        init_lvl = min(max(0, int(self._env.specified_terrain_level)), num_levels - 1)
                    except (ValueError, TypeError):
                        pass
            elif hasattr(terrain, "terrain_levels"):
                env_idx = min(get_env_idx(), len(terrain.terrain_levels) - 1)
                init_lvl = min(max(0, int(terrain.terrain_levels[env_idx].item())), num_levels - 1)

            with server.gui.add_folder("Terrain Difficulty (楼梯难度)"):
                dropdown = server.gui.add_dropdown(
                    "Difficulty Level",
                    options=options,
                    initial_value=options[init_lvl],
                )

                def _on_dropdown_change(_) -> None:
                    selected_str = dropdown.value
                    if selected_str == "Random (随机难度)":
                        self._env.specified_terrain_level = "random"
                    elif selected_str in options:
                        lvl = options.index(selected_str)
                        self._env.specified_terrain_level = lvl
                    if request_action is not None:
                        request_action("RESET")

                dropdown.on_update(_on_dropdown_change)

    def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
        """在播放或可视化时，在目标位置绘制高亮标志物与行进方向指示线，并在机身正上方显示终止原因。"""
        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return

        import numpy as np
        robot_pos_ws = self.robot.data.root_link_pos_w.cpu().numpy()
        env_origins = self._env.scene.env_origins.cpu().numpy()
        targets_local = self.target_pos.cpu().numpy()

        # target_heights 等于 command 里 Base_height_target 里的 nominal_height
        if "Base_height_target" in self._env.command_manager.active_terms:
            height_cmd_term = self._env.command_manager.get_term("Base_height_target")
            target_heights = float(height_cmd_term.cfg.nominal_height)
        else:
            target_heights = None

        server = getattr(visualizer, "server", None)

        for batch in env_indices:
            robot_pos = robot_pos_ws[batch]
            origin = env_origins[batch]

            # 目标点世界坐标 (x, y)
            target_x = float(origin[0] + targets_local[batch, 0])
            target_y = float(origin[1] + targets_local[batch, 1])

            # 目标落点 z：优先用 target_heights (nominal_height)，否则 fallback
            base_z = float(origin[2])
            if target_heights is not None:
                ref_z = target_heights
            else:
                ref_z = max(float(robot_pos[2]), base_z)

            # 1. 竖立的鲜艳标杆 (从地面延伸至空中 1.4m，无论地形起伏都清晰可见)
            pole_bottom = np.array([target_x, target_y, base_z - 0.2], dtype=np.float32)
            pole_top = np.array([target_x, target_y, base_z + ref_z + 1.2], dtype=np.float32)
            visualizer.add_cylinder(
                start=pole_bottom,
                end=pole_top,
                radius=0.035,
                color=(1.0, 0.25, 0.0, 0.9),  # 亮红橙色标杆
                label=f"goal_pole_{batch}",
            )

            # 2. 标杆顶部的发光红色大标志球 (半径 0.2m)
            visualizer.add_sphere(
                center=pole_top,
                radius=0.20,
                color=(1.0, 0.05, 0.05, 0.95),  # 鲜艳红色信号球
                label=f"goal_top_{batch}",
            )

            # 3. 目标落点标记球，z = Base_height_target 目标高度 (半径 0.15m)
            visualizer.add_sphere(
                center=np.array([target_x, target_y, ref_z], dtype=np.float32),
                radius=0.15,
                color=(1.0, 0.7, 0.0, 0.85),  # 金黄色落点球
                label=f"goal_base_{batch}",
            )

            # 4. 从机身中心直接指向目标落点的荧光绿导航箭头
            arrow_start = np.array([robot_pos[0], robot_pos[1], robot_pos[2]], dtype=np.float32)
            arrow_end = np.array([target_x, target_y, ref_z + 0.15], dtype=np.float32)
            visualizer.add_arrow(
                start=arrow_start,
                end=arrow_end,
                color=(0.1, 0.95, 0.2, 0.75),  # 荧光绿色
                width=0.025,
                label=f"goal_dir_{batch}",
            )

            # 5. 上次终止原因指示 (3D View 悬浮文字与颜色状态球)
            last_reason = self.last_reset_reason[batch]
            dx = target_x - robot_pos[0]
            dy = target_y - robot_pos[1]
            dist_to_goal = float(np.sqrt(dx * dx + dy * dy))
            ep_step = (
                int(self._env.episode_length_buf[batch].item())
                if hasattr(self._env, "episode_length_buf")
                else 0
            )

            # 状态球颜色
            if "reach_target" in last_reason:
                status_color = (0.1, 0.95, 0.2, 0.85)  # 绿: 成功
            elif "time_out" in last_reason:
                status_color = (1.0, 0.85, 0.1, 0.85)  # 黄: 超时
            elif "manual" in last_reason or "初始" in last_reason:
                status_color = (0.2, 0.6, 1.0, 0.85)   # 蓝: 准备
            else:
                status_color = (1.0, 0.2, 0.2, 0.85)   # 红: 失败终止

            # 在机身正上方 0.35m 放置状态指示光球
            status_pos = np.array([robot_pos[0], robot_pos[1], robot_pos[2] + 0.35], dtype=np.float32)
            visualizer.add_sphere(
                center=status_pos,
                radius=0.06,
                color=status_color,
                label=f"term_status_sphere_{batch}",
            )

            # 在 Viser 3D 视图中添加悬浮文本标签 (深度测试关闭，保证始终清晰可见)
            if server is not None:
                label_pos = np.array([robot_pos[0], robot_pos[1], robot_pos[2] + 0.45], dtype=np.float32)
                label_text = f"[{last_reason}]\n距目标: {dist_to_goal:.2f}m | 步数: {ep_step}"
                server.scene.add_label(
                    name=f"/robot_term_info_{batch}",
                    text=label_text,
                    position=label_pos,
                    depth_test=False,
                    font_screen_scale=1.0,
                    anchor="bottom-center",
                )

        # 刷新 Viser 控制面板 HTML 内容
        if hasattr(self, "_get_env_idx") and self._get_env_idx is not None:
            curr_env = self._get_env_idx()
            self._render_term_html(curr_env)


# ==============================================================================
# 2. Base Height Target Command (机身高度目标命令)
# ==============================================================================

@dataclass(kw_only=True)
class HeightTargetCommandCfg(CommandTermCfg):
    """机身高度目标命令配置。

    目标身体高度基于地面高度动态计算：
        height_cmd = ground_z + nominal_height
    其中 ground_z 由足端 body 在世界坐标系下的 z 值估算：
        - 平地模式（is_stair=False）：取四足 z 的最小值（最稳定的接地估计）
        - 楼梯模式（is_stair=True）：取四足 z 的最大值（跟随最高台阶高度）
    """
    entity_name: str = "robot"
    nominal_height: float = 0.28
    """狗站立时机身相对地面的高度（米）。"""
    foot_body_names: tuple[str, ...] = ("FR_foot", "FL_foot", "RR_foot", "RL_foot")
    """四个足端 body 的名称，用于估算地面高度。"""
    is_stair: bool = False
    """是否为楼梯任务。平地取足端最小 z，楼梯取足端最大 z。"""
    height_smooth_alpha: float = 0.05
    """低通滤波系数 (0, 1]：越小越平滑，越大跟随越快。默认 0.05 较为平滑。"""
    resampling_time_range: tuple[float, float] = (10.0, 10.0)
    debug_vis: bool = False

    def build(self, env: ManagerBasedRlEnv) -> HeightTargetCommand:
        return HeightTargetCommand(self, env)


class HeightTargetCommand(CommandTerm):
    """机身高度目标命令执行项。

    每步根据足端 body 世界坐标的 z 值动态估算当前地面高度，
    加上 nominal_height 后作为期望机身高度指令输出。
    低通滤波避免楼梯台阶边缘造成的高度突变。
    """
    cfg: HeightTargetCommandCfg

    def __init__(self, cfg: HeightTargetCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.robot: Entity = env.scene[cfg.entity_name]

        # 初始化：相对地面高度从 0.0 开始，加上 nominal_height 作为初始高度命令 (0.28m)
        self._ground_z = torch.zeros(self.num_envs, device=self.device)
        self._command = torch.full((self.num_envs, 1), cfg.nominal_height, device=self.device)

        # 解析足端 body 的 body_id 索引
        self._foot_body_ids: list[int] | None = None

    def _resolve_foot_ids(self) -> list[int]:
        """延迟解析足端 body ids（第一次使用时）。"""
        if self._foot_body_ids is not None:
            return self._foot_body_ids
        body_names: list[str] = list(self.robot.body_names)
        ids = []
        for name in self.cfg.foot_body_names:
            try:
                ids.append(body_names.index(name))
            except ValueError:
                # 兼容：如果找不到，fallback 到 root body
                pass
        if not ids:
            ids = [0]  # fallback：用 root body
        self._foot_body_ids = ids
        return ids

    @property
    def command(self) -> torch.Tensor:
        """返回当前步期望机身高度指令张量 (num_envs, 1)。"""
        return self._command

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        """重置时，重置对应环境相对地面高度估计为 0.0。"""
        if len(env_ids) == 0:
            return
        self._ground_z[env_ids] = 0.0
        self._command[env_ids, 0] = self.cfg.nominal_height

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        """每步用足端相对环境原点的 z 坐标估算地面高度，低通滤波后更新目标高度命令。"""
        ids = slice(None) if env_ids is None else env_ids

        foot_ids = self._resolve_foot_ids()

        # 获取足端相对当前环境 Patch 原点的相对 z 坐标 (num_envs, num_feet)
        # 彻底消除不同难度倒金字塔凹底绝对落差（-0.18m ~ -0.90m）造成的指令负数畸变！
        foot_pos_w = self.robot.data.body_link_pos_w  # (num_envs, num_bodies, 3)
        origin_z = self._env.scene.env_origins[:, 2:3]  # (num_envs, 1)
        foot_z_rel = foot_pos_w[:, foot_ids, 2] - origin_z

        # 楼梯：取最大相对 z（跟随已踏上的最高台阶，最低为凹底 0.0）
        # 平地：取最小相对 z（最接近地面的足端）
        if self.cfg.is_stair:
            raw_ground_z = foot_z_rel.max(dim=-1).values.clamp(min=0.0)
        else:
            raw_ground_z = foot_z_rel.min(dim=-1).values

        # 低通滤波平滑地面相对高度估计
        alpha = self.cfg.height_smooth_alpha
        self._ground_z[ids] = (
            (1.0 - alpha) * self._ground_z[ids] + alpha * raw_ground_z[ids]
        )

        # 输出 = 相对地面高度 + 标称站立高度 (在凹底出生时严格为 nominal_height=0.28)
        self._command[ids, 0] = self._ground_z[ids] + self.cfg.nominal_height

    def _update_metrics(self) -> None:
        pass


# ==============================================================================
# 3. Heading Error Command (航向误差命令)
# ==============================================================================

@dataclass(kw_only=True)
class HeadErrorCommandCfg(CommandTermCfg):
    """航向误差命令配置。"""
    entity_name: str = "robot"
    resampling_time_range: tuple[float, float] = (10.0, 10.0)
    debug_vis: bool = False

    def build(self, env: ManagerBasedRlEnv) -> HeadErrorCommand:
        return HeadErrorCommand(self, env)


class HeadErrorCommand(CommandTerm):
    """航向误差命令执行项。"""
    cfg: HeadErrorCommandCfg

    def __init__(self, cfg: HeadErrorCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._command = torch.zeros((self.num_envs, 1), device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """返回当前步航向角误差指令张量 (num_envs, 1)。"""
        return self._command

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        self._update_command(env_ids)

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        """从 Goal_direction 共享航向误差值。"""
        ids = slice(None) if env_ids is None else env_ids
        if "Goal_direction" in self._env.command_manager.active_terms:
            goal_term = self._env.command_manager.get_term("Goal_direction")
            if hasattr(goal_term, "heading_error"):
                self._command[ids] = goal_term.heading_error[ids]

    def _update_metrics(self) -> None:
        pass


# ==============================================================================
# 4. Goal-Aligned Velocity Command (与目标方向对齐的速度命令)
# ==============================================================================

@dataclass(kw_only=True)
class GoalAlignedVelocityCommandCfg(UniformVelocityCommandCfg):
    """默认 (未在 Viser 中启用摇杆) 时把速度指令方向对齐到 Goal_direction 的配置。

    - 线速度方向：直接使用 Goal_direction 的机体系单位方向向量
      ``[cos(Δψ), sin(Δψ)]``，因此指令始终指向目标点；
    - 线速度大小：在 ``align_speed_range`` 内随机重采样；
    - 角速度：``align_heading_gain * 航向误差``，按 ``ranges.ang_vel_z`` 裁剪。

    在 Viser 中勾选 Twist -> Enable 后，滑条值会覆盖这里的对齐结果。
    """
    goal_command_name: str = "Goal_direction"
    """用于对齐方向的目标方向指令项名称。"""

    align_speed_range: tuple[float, float] = (0.25, 0.6)
    """对齐时线速度大小的采样区间 (m/s)。设为相同值即为固定速度。"""

    align_heading: bool = True
    """是否由航向误差生成角速度指令；关闭则保留父类的随机角速度。"""

    align_heading_gain: float = 1.5
    """角速度 = align_heading_gain * 航向误差。"""

    def build(self, env: ManagerBasedRlEnv) -> "GoalAlignedVelocityCommand":
        return GoalAlignedVelocityCommand(self, env)


class GoalAlignedVelocityCommand(UniformVelocityCommand):
    """默认把线速度方向与角速度对齐到 Goal_direction 的速度指令项。"""

    cfg: GoalAlignedVelocityCommandCfg

    def __init__(self, cfg: GoalAlignedVelocityCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        # 对齐后的线速度大小 (每次重采样更新一次)
        self.align_speed = torch.zeros(self.num_envs, device=self.device)
        self._goal_term = None

    def _goal_command_term(self) -> GoalDirectionCommand | None:
        """懒加载 Goal_direction 指令项；该任务未配置时返回 None (退化为随机指令)。"""
        if self._goal_term is None:
            name = self.cfg.goal_command_name
            if name in self._env.command_manager.active_terms:
                self._goal_term = self._env.command_manager.get_term(name)
        return self._goal_term

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        if len(env_ids) == 0:
            return
        r = torch.empty(len(env_ids), device=self.device)
        self.align_speed[env_ids] = r.uniform_(*self.cfg.align_speed_range)

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        super()._update_command(env_ids)

        goal = self._goal_command_term()
        if goal is None or not hasattr(goal, "command"):
            return

        # 保留父类 standing / forward-only / world-frame 环境各自的语义
        active = ~(self.is_standing_env | self.is_forward_env | self.is_world_env)
        ids = active.nonzero(as_tuple=False).flatten()
        if len(ids) == 0:
            return

        # Goal_direction 已给出机体系单位方向 [cos(Δψ), sin(Δψ)]
        dir_b = goal.command[ids].to(device=self.device, dtype=torch.float32)
        speed = self.align_speed[ids]
        self.vel_command_b[ids, 0] = speed * dir_b[:, 0]
        self.vel_command_b[ids, 1] = speed * dir_b[:, 1]

        if self.cfg.align_heading and hasattr(goal, "heading_error"):
            head_err = goal.heading_error[ids, 0]
            lo, hi = self.cfg.ranges.ang_vel_z
            self.vel_command_b[ids, 2] = torch.clamp(
                self.cfg.align_heading_gain * head_err, min=lo, max=hi
            )


# 别名兼容
HeadingErrorCommandCfg = HeadErrorCommandCfg
HeadingErrorCommand = HeadErrorCommand

__all__ = [
    "GoalDirectionCommandCfg",
    "GoalDirectionCommand",
    "HeightTargetCommandCfg",
    "HeightTargetCommand",
    "HeadErrorCommandCfg",
    "HeadErrorCommand",
    "HeadingErrorCommandCfg",
    "HeadingErrorCommand",
    "GoalAlignedVelocityCommandCfg",
    "GoalAlignedVelocityCommand",
]
