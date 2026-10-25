"""DTC 速度任务用的 TAMOLS 落脚点命令项 (仅覆写可视化, 规划逻辑零改动).

与用户约定的关系
================
- 命令项的**规划计算**完全继承 fetch 侧现有方案
  (``TamolsMotionCommand``: 触地触发的 TorchTAMOLSMotionPlanner 重规划),
  本模块不改任何规划行为;
- 唯一覆写是 ``_debug_vis_impl``: 绘制委托给
  :func:`myrobot_mjlab.task.DTC.tamols.debug_viz.draw_command_footholds`
  (红 = optimal 重规划落点 / 橙 = landing 摆动腿当前目标 / 蓝 = nominal
  初值), 在速度任务的 Viser 视图里观测 TAMOLS 计算的落足点是否正确;
- 本文件位于 task/DTC 内, 不修改 fetch/velocity 的任何文件.
"""

from __future__ import annotations

from dataclasses import dataclass

from myrobot_mjlab.task.DTC.tamols.debug_viz import draw_command_footholds
from myrobot_mjlab.task.Tamols.fetch.tamols_command import (
    TamolsMotionCommand,
    TamolsMotionCommandCfg,
)


@dataclass(kw_only=True)
class DTCFootholdCommandCfg(TamolsMotionCommandCfg):
    """DTC 版 TAMOLS 落脚点命令项配置 (默认开启三色足点可视化)."""

    debug_vis: bool = True
    show_nominal_dots: bool = True
    """ nominal 蓝点默认显示 (对比重规划是否偏离初值)."""


class DTCFootholdCommand(TamolsMotionCommand):
    """继承 fetch 的 TAMOLS 命令项 (规划计算不变), 只换可视化绘制."""

    def _debug_vis_impl(self, visualizer) -> None:
        # 绘制逻辑位于 task/DTC/tamols/debug_viz.py (红/橙/蓝足点)
        draw_command_footholds(visualizer, self)
