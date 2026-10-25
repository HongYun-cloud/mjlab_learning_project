"""DTC/TAMOLS 规划的仿真视图可视化 (mjlab DebugVisualizer 协议).

定位 (与用户约定一致)
=====================
- 可视化的对象是 **task/DTC/tamols 求解器生成的计划** (:class:`TAMOLSPlan`,
  由 :mod:`rl_bridge` 的 worker 发布): 有限时域内的期望落脚点、基座
  参考轨迹与触地事件; 不是全局路径。
- RL 训练暂用 fetch 侧现有配置方案; 本模块只依赖 duck-typed 的
  mjlab ``DebugVisualizer`` 协议 (``add_sphere``/``add_arrow``/
  ``get_env_indices``), **不 import mjlab** (仅 TYPE_CHECKING), 因此
  无 viewer 环境 (训练/单测) 也可运行与测试。

接线方式
========
:class:`DTCPlanVizObserver` 实现了 :class:`rl_bridge.PlanObserver`:
worker 每发布一份计划就调用 ``on_plan`` 更新内部快照; MJLab 侧的命令
项/管理器每帧调用 ``debug_vis(visualizer)`` (与 mjlab CommandTerm 的
``_debug_vis_impl`` 约定一致) 把快照画出来::

    worker = PlanningWorker(solve_fn)
    viz = DTCPlanVizObserver(env_id=0)
    worker.attach_observer(viz)
    ...
    # mjlab 侧每帧:
    my_term.debug_vis = lambda v: viz.debug_vis(v)

也可脱离 worker 直接使用 :func:`draw_plan` 绘制任意计划快照.

绘制内容 (可逐项开关, 见 :class:`DTCVizConfig`)
================================================
- ``draw_footholds``: 期望落脚点 (红, label ``dtc_foothold_{leg}_{env}``);
- ``draw_trajectory``: 基座参考轨迹采样点 (青, ``dtc_base_{i}_{env}``);
- ``draw_horizon_end``: 时域终点坐标架 (label ``dtc_end_{env}``);
- ``draw_touchdown``: 触地腿在触地时刻的足点标记 (橙, ``dtc_td_{leg}_{env}``).

单位: 位置 m (世界系), 与 TAMOLSPlan 一致.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from myrobot_mjlab.task.DTC.tamols.rl_bridge import (
    PlanObserver,
    TAMOLSPlan,
)

if TYPE_CHECKING:  # 仅类型检查; 运行时 duck-typing, 不依赖 mjlab
    from mjlab.viewer.debug_visualizer import DebugVisualizer

LEG_NAMES = ("FL", "FR", "RL", "RR")


@dataclass
class DTCVizConfig:
    draw_footholds: bool = True
    draw_trajectory: bool = True
    draw_touchdown: bool = True
    draw_horizon_end: bool = False
    trajectory_samples: int = 12
    """基座轨迹采样点数 (沿时域均匀取)."""
    foothold_radius: float = 0.035
    trajectory_radius: float = 0.02
    foothold_color: tuple[float, float, float, float] = (1.0, 0.05, 0.05, 0.95)
    trajectory_color: tuple[float, float, float, float] = (0.1, 0.8, 0.9, 0.8)
    touchdown_color: tuple[float, float, float, float] = (1.0, 0.55, 0.0, 0.95)
    end_frame_color: tuple[float, float, float, float] = (0.2, 1.0, 0.3, 0.9)


def draw_plan(
    visualizer: "DebugVisualizer",
    plan: TAMOLSPlan | None,
    cfg: DTCVizConfig | None = None,
) -> int:
    """把一份计划快照画到 ``visualizer`` 上, 返回绘制的图元数.

    ``plan=None`` 或未成功计划时不绘制任何图元 (由 worker 语义保证
    latest_plan 保留旧解, 调用方无需清理——mjlab 场景每帧清空队列).
    """
    cfg = cfg or DTCVizConfig()
    if plan is None or not plan.success:
        return 0
    drawn = 0
    env = getattr(visualizer, "env_idx", 0)
    if cfg.draw_footholds and plan.footholds_w.size:
        for leg in range(4):
            pt = plan.footholds_w[leg]
            if not np.isfinite(pt).all():
                continue
            visualizer.add_sphere(
                center=np.asarray(pt, dtype=float),
                radius=cfg.foothold_radius,
                color=cfg.foothold_color,
                label=f"dtc_foothold_{LEG_NAMES[leg]}_{env}",
            )
            drawn += 1
    if cfg.draw_trajectory and plan.t.size:
        n = max(2, min(cfg.trajectory_samples, plan.t.size))
        for i in np.linspace(0, len(plan.t) - 1, n).astype(int):
            t = float(plan.t[i])
            ref = plan.base_reference_at(t)
            visualizer.add_sphere(
                center=ref["pos_w"],
                radius=cfg.trajectory_radius,
                color=cfg.trajectory_color,
                label=f"dtc_base_{i}_{env}",
            )
            drawn += 1
    if cfg.draw_touchdown and plan.touchdown_times.size:
        for leg in range(4):
            t_td = plan.touchdown_times[leg]
            if not np.isfinite(t_td):
                continue
            ref = plan.base_reference_at(float(t_td))
            visualizer.add_arrow(
                start=ref["pos_w"] + np.array([0.0, 0.0, 0.05]),
                end=plan.footholds_w[leg] + np.array([0.0, 0.0, 0.01]),
                color=cfg.touchdown_color,
                width=0.012,
                label=f"dtc_td_{LEG_NAMES[leg]}_{env}",
            )
            drawn += 1
    if cfg.draw_horizon_end and plan.success:
        ref = plan.base_reference_at(plan.horizon[1])
        visualizer.add_frame(
            position=ref["pos_w"],
            rotation_matrix=_rot_from_euler(ref["euler_zyx"]),
            scale=0.15,
            label=f"dtc_end_{env}",
        )
        drawn += 1
    return drawn


def _rot_from_euler(euler: np.ndarray) -> np.ndarray:
    cr, sr = np.cos(euler[0]), np.sin(euler[0])
    cp, sp = np.cos(euler[1]), np.sin(euler[1])
    cy, sy = np.cos(euler[2]), np.sin(euler[2])
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def draw_command_footholds(
    visualizer: "DebugVisualizer",
    cmd: Any,
    cfg: DTCVizConfig | None = None,
) -> int:
    """把 fetch 侧 ``TamolsMotionCommand`` 生成的足点画出来 (适配层).

    ``cmd`` 只需 duck-type 以下属性 (torch 张量或 numpy 均可)::

        optimal_footholds [N,4,3]  重规划最优落点 (红)
        landing_footholds [N,4,3]  摆动腿当前落点目标 (橙, 奖励跟踪量)
        nominal_footholds [N,4,3]  名义落点 (蓝, 可选)
        num_envs, cfg.show_nominal_dots / dot_radius / dot_color /
        nominal_dot_color        (getattr 带默认, fetch cfg 无需为此修改)

    所有绘制逻辑与 label 约定 (``tamols_opt_/land_/nom_{leg}_{env}``)
    都在本模块 (DTC/tamols); fetch 命令项的 ``_debug_vis_impl`` 只是
    两行委托. 返回绘制图元数.
    """
    cfg = cfg or DTCVizConfig()
    drawn = 0
    env = getattr(visualizer, "env_idx", 0)
    for env_id in visualizer.get_env_indices(int(getattr(cmd, "num_envs", 1))):
        dot_radius = float(getattr(cmd.cfg, "dot_radius", cfg.foothold_radius))
        dot_color = getattr(cmd.cfg, "dot_color", cfg.foothold_color)
        for leg in range(4):
            name = LEG_NAMES[leg] if leg < len(LEG_NAMES) else str(leg)
            opt = cmd.optimal_footholds[env_id, leg]
            opt = opt.detach().cpu().numpy() if hasattr(opt, "detach") else np.asarray(opt)
            visualizer.add_sphere(
                center=np.asarray(opt, dtype=float),
                radius=dot_radius,
                color=dot_color,
                label=f"tamols_opt_{name}_{env_id}",
            )
            drawn += 1
        if getattr(cmd.cfg, "show_landing_dots", True) and hasattr(
            cmd, "landing_footholds"
        ):
            landing_color = getattr(cmd.cfg, "landing_dot_color",
                                    (1.0, 0.55, 0.0, 0.95))
            for leg in range(4):
                name = LEG_NAMES[leg] if leg < len(LEG_NAMES) else str(leg)
                land = cmd.landing_footholds[env_id, leg]
                land = (land.detach().cpu().numpy()
                        if hasattr(land, "detach") else np.asarray(land))
                visualizer.add_sphere(
                    center=np.asarray(land, dtype=float),
                    radius=dot_radius * 0.9,
                    color=landing_color,
                    label=f"tamols_land_{name}_{env_id}",
                )
                drawn += 1
        if getattr(cmd.cfg, "show_nominal_dots", False) and hasattr(
            cmd, "nominal_footholds"
        ):
            nom_color = getattr(cmd.cfg, "nominal_dot_color",
                                (0.2, 0.6, 1.0, 0.7))
            for leg in range(4):
                name = LEG_NAMES[leg] if leg < len(LEG_NAMES) else str(leg)
                nom = cmd.nominal_footholds[env_id, leg]
                nom = (nom.detach().cpu().numpy()
                       if hasattr(nom, "detach") else np.asarray(nom))
                visualizer.add_sphere(
                    center=np.asarray(nom, dtype=float),
                    radius=dot_radius * 0.75,
                    color=nom_color,
                    label=f"tamols_nom_{name}_{env_id}",
                )
                drawn += 1
    return drawn


class DTCPlanVizObserver(PlanObserver):
    """把 worker 发布的最新计划存下来, 供每帧 ``debug_vis`` 绘制.

    同时实现 mjlab 风格的 ``debug_vis(visualizer)`` 与属性
    ``env_idx`` / ``show_all_envs`` (与 CommandTerm._debug_vis_impl
    的调用约定一致), 因此可直接挂进任何管理器/播放脚本的绘制路径.
    """

    def __init__(
        self,
        env_idx: int = 0,
        show_all_envs: bool = False,
        cfg: DTCVizConfig | None = None,
    ):
        self.env_idx = env_idx
        self.show_all_envs = show_all_envs
        self.cfg = cfg or DTCVizConfig()
        self.plan: TAMOLSPlan | None = None
        self.stamps_seen: list[int] = []

    # -- rl_bridge.PlanObserver 协议 ------------------------------------- #
    def on_plan(self, plan: TAMOLSPlan) -> None:
        self.plan = plan
        self.stamps_seen.append(plan.stamp)

    def observe(self, t: float, plan: TAMOLSPlan | None) -> dict[str, Any]:
        return {"dtc_viz_stamp": plan.stamp if plan else None}

    # -- mjlab 每帧绘制入口 ----------------------------------------------- #
    def debug_vis(self, visualizer: "DebugVisualizer") -> None:
        draw_plan(visualizer, self.plan, self.cfg)
