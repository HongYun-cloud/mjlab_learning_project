"""DTC 摆动腿观测层: TAMOLS 落足点 + 固定 trot + 路径局部高度.

用户约定 (第一步)
================
- **接收 TAMOLS 输出的落足点**: :func:`footholds_from` 同时接受
  fetch 命令项的 ``optimal_footholds`` 张量 [N,4,3] (世界系, FL/FR/
  RL/RR) 与 DTC NLP 桥接的 :class:`~myrobot_mjlab.task.DTC.tamols.
  rl_bridge.TAMOLSPlan` (``footholds_w``)。
- **固定 trot 步态**: 复用 fetch 的 ``GaitScheduler`` (相位偏移
  (0, 0.5, 0.5, 0) = 对角 trot, duty 0.5), 只 import 不修改;
  ``SwingObsBuilder`` 可**共享**命令项已有的调度器实例 (时钟由命令项
  推进, builder 不重复 step), 也可独立自建 (测试/其他接入)。
- **控制摆腿的观测** (供 RL, 摆动轨迹由策略自己学, 本模块**不生成
  任何规划轨迹**):
    - ``goal_rel_b``  [N,4,3]  目标落点相对当前脚的位置 (机体系 xy,
      z 为高度差; 支撑腿同样给出, RL 自行决定使用方式);
    - ``path_rel_h``  [N,4,S]  当前脚→落点 xy 直线路径上 S 个采样点
      的局部高度 (相对当前脚高度; 采样点只是地形查询位置, 不是参考
      轨迹);
    - ``swing_mask``  [N,4]    摆动相掩码; ``swing_progress`` [N,4]
      摆动进度 0..1 (支撑相为 0);
    - ``phase`` [N,4] 步态相位; ``touchdown_events`` [N,4] 本步
      摆动→支撑切换沿 (供触发 TAMOLS 重规划 / rl_bridge 事件)。
- **局部高度来源** (机体系 xy 查询, 批量, 可插拔):
    - :class:`LocalGridHeights`: fetch 风格本地高程图
      ``[N,1,Ny,Nx]`` (x 前向, y 左侧, 原点=机体), 双线性, 越界 clamp;
    - :class:`CallableHeights`: 任意函数 (机体系 xy → z);
    - :class:`FlatHeights`: 平地容错 (常数 z, 默认 0)。

单位: 位置/高度 m, 相位无量纲, 时间 s。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from myrobot_mjlab.task.Tamols.fetch.gait_scheduler import (
    GaitScheduler,
    GaitSchedulerCfg,
)

LEG_ORDER = ("FL", "FR", "RL", "RR")
TROT_PHASE_OFFSETS = (0.0, 0.5, 0.5, 0.0)
"""对角 trot: FL+RR 一组, FR+RL 一组 (fetch GaitScheduler 默认)."""


# ---------------------------------------------------------------------- #
# 落足点接收 (TAMOLS 输出 → 张量)
# ---------------------------------------------------------------------- #
def footholds_from(source: Any, device: str | torch.device = "cpu") -> torch.Tensor:
    """从 TAMOLS 输出源取落足点张量 ``[N,4,3]`` (世界系, FL/FR/RL/RR).

    支持:
      - 已是 ``[N,4,3]`` 张量 (fetch 命令项的 ``optimal_footholds``);
      - :class:`TAMOLSPlan` (DTC NLP 桥接, 取 ``footholds_w``);
      - 带 ``optimal_footholds`` 属性的命令项.
    """
    if isinstance(source, torch.Tensor):
        footholds = source
    else:
        plan_footholds = getattr(source, "footholds_w", None)
        if plan_footholds is None:
            plan_footholds = getattr(source, "optimal_footholds", None)
        if plan_footholds is None:
            raise ValueError(
                "footholds_from: 源既不是 [N,4,3] 张量, 也没有 "
                "footholds_w (TAMOLSPlan) 或 optimal_footholds (命令项)"
            )
        footholds = plan_footholds
        if hasattr(footholds, "detach"):
            footholds = footholds.detach()
        footholds = torch.as_tensor(footholds, dtype=torch.float32, device=device)
    if footholds.shape == (4, 3):
        footholds = footholds.unsqueeze(0)  # TAMOLSPlan 是单环境计划
    if footholds.dim() != 3 or footholds.shape[1:] != (4, 3):
        raise ValueError(
            f"落足点形状应为 [N,4,3] 或 [4,3] (FL/FR/RL/RR), "
            f"实际 {tuple(footholds.shape)}"
        )
    return footholds


# ---------------------------------------------------------------------- #
# 局部高度提供者 (机体系 xy → z, 批量)
# ---------------------------------------------------------------------- #
class FlatHeights:
    """平地容错: 常数高度 (无地图时的 fallback)."""

    def __init__(self, z: float = 0.0):
        self.z = float(z)

    def query(self, body_xy: torch.Tensor) -> torch.Tensor:
        shape = body_xy.shape[:-1]
        return torch.full(shape, self.z, dtype=body_xy.dtype, device=body_xy.device)


class CallableHeights:
    """任意高度函数 (机体系 xy [...,2] -> z [...]); 用于测试/自定义地形."""

    def __init__(self, fn: Callable[[torch.Tensor], torch.Tensor]):
        self.fn = fn

    def query(self, body_xy: torch.Tensor) -> torch.Tensor:
        return self.fn(body_xy)


class LocalGridHeights:
    """fetch 风格本地高程图的双线性查询 (机体系网格).

    网格约定与 ``TamolsMotionCommand`` 的 ``h_raw`` 一致:
    ``[N,1,Ny,Nx]`` 或 ``[N,Ny,Nx]``, x 前向 (j 列), y 左侧 (i 行),
    ``x[j] = x_min + j*dx``; 高度单位 m。越界 clamp (与
    :mod:`map_query` 的符号路径语义一致)。输入为机体系 xy ``[...,2]``,
    输出同形状前导维的 z。
    """

    def __init__(
        self,
        h_grid: torch.Tensor,
        x_min: float,
        y_min: float,
        dx: float,
        dy: float | None = None,
    ):
        if h_grid.dim() == 4:
            h_grid = h_grid[:, 0]
        if h_grid.dim() != 3:
            raise ValueError(
                f"h_grid 需为 [N,Ny,Nx] 或 [N,1,Ny,Nx], 实际 {tuple(h_grid.shape)}"
            )
        self.h = h_grid
        self.x_min = float(x_min)
        self.y_min = float(y_min)
        self.dx = float(dx)
        self.dy = float(dy if dy is not None else dx)

    def query(self, body_xy: torch.Tensor) -> torch.Tensor:
        *lead, _ = body_xy.shape
        n = self.h.shape[0]
        if lead[:1] and lead[0] == n:
            env = torch.arange(n, device=body_xy.device)
        else:
            env = torch.zeros(lead, dtype=torch.long, device=body_xy.device)
        fx = (body_xy[..., 0] - self.x_min) / self.dx
        fy = (body_xy[..., 1] - self.y_min) / self.dy
        fx = fx.clamp(0.0, float(self.h.shape[2] - 1))
        fy = fy.clamp(0.0, float(self.h.shape[1] - 1))
        j0 = torch.clamp(fx.floor().long(), max=self.h.shape[2] - 2)
        i0 = torch.clamp(fy.floor().long(), max=self.h.shape[1] - 2)
        tx = fx - j0
        ty = fy - i0
        if env.dim() and env.dim() > 0 and lead[:1] and lead[0] == n:
            idx = env[..., None, None].expand_as(j0)
        else:
            idx = env
        def g(i, j):
            return self.h[idx, i, j]
        return (
            (1 - ty) * (1 - tx) * g(i0, j0)
            + (1 - ty) * tx * g(i0, j0 + 1)
            + ty * (1 - tx) * g(i0 + 1, j0)
            + ty * tx * g(i0 + 1, j0 + 1)
        )


# ---------------------------------------------------------------------- #
# 摆动观测构造器
# ---------------------------------------------------------------------- #
@dataclass
class SwingObsCfg:
    path_samples: int = 5
    """当前脚→落点 xy 直线路径的采样点数 S (含两端; 仅地形查询位置)."""
    gait: GaitSchedulerCfg = field(default_factory=GaitSchedulerCfg)
    """固定 trot (默认相位偏移 = 对角 trot); 已由调用方共享调度器时忽略."""


class SwingObsBuilder:
    """组装摆动腿观测 (每步一次 :meth:`update`).

    Args:
      cfg: 采样数与步态配置.
      num_envs / device: 批量.
      gait: 可注入的**共享**调度器 (命令项已推进时钟时传入, builder
            不再自建/推进); None 时自建固定 trot 调度器并由 ``step`` 推进.
    """

    def __init__(
        self,
        cfg: SwingObsCfg | None = None,
        num_envs: int = 1,
        device: str | torch.device = "cpu",
        gait: GaitScheduler | None = None,
    ):
        self.cfg = cfg or SwingObsCfg()
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        if gait is not None:
            self.gait = gait
            self._owns_gait = False
        else:
            trot = GaitSchedulerCfg(
                stride_duration=self.cfg.gait.stride_duration,
                duty_factor=self.cfg.gait.duty_factor,
                phase_offsets=self.cfg.gait.phase_offsets,
            )
            self.gait = GaitScheduler(trot, num_envs=self.num_envs, device=self.device)
            self._owns_gait = True
        self._prev_swing = torch.zeros(
            (self.num_envs, 4), dtype=torch.bool, device=self.device
        )

    # -- 时钟 (仅自建调度器时; 共享时钟由命令项推进) -------------------- #
    def step(self, dt: float, env_ids: torch.Tensor | None = None) -> None:
        if not self._owns_gait:
            raise RuntimeError(
                "共享调度器模式: 时钟由命令项推进, builder.step() 禁用 "
                "(避免双倍推进)"
            )
        self.gait.step(dt=dt, env_ids=env_ids)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if self._owns_gait:
            self.gait.reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self._prev_swing[ids] = False

    # -- 观测 -------------------------------------------------------------- #
    @torch.no_grad()
    def update(
        self,
        foot_pos_w: torch.Tensor,
        footholds_w: torch.Tensor,
        base_pos_w: torch.Tensor,
        base_yaw_w: torch.Tensor,
        heights: Any,
    ) -> dict[str, torch.Tensor]:
        """组装并返回观测字典 (形状/语义见模块文档).

        Args:
          foot_pos_w [N,4,3] / footholds_w [N,4,3]: 世界系 (后者可经
            :func:`footholds_from` 得到).
          base_pos_w [N,3] / base_yaw_w [N]: 机体系参考.
          heights: 机体系 xy 查询对象 (FlatHeights/CallableHeights/
            LocalGridHeights).
        """
        foot = foot_pos_w.to(dtype=torch.float32)
        goal = footholds_w.to(dtype=torch.float32)
        base = base_pos_w.to(dtype=torch.float32).reshape(-1, 3)
        yaw = base_yaw_w.to(dtype=torch.float32).reshape(-1)
        n = foot.shape[0]
        cos, sin = torch.cos(yaw), torch.sin(yaw)

        def to_body(vec_w: torch.Tensor) -> torch.Tensor:
            d = vec_w - base[:, :2].unsqueeze(1)
            bx = cos.unsqueeze(1) * d[..., 0] + sin.unsqueeze(1) * d[..., 1]
            by = -sin.unsqueeze(1) * d[..., 0] + cos.unsqueeze(1) * d[..., 1]
            return torch.stack((bx, by), dim=-1)

        foot_b = to_body(foot[..., :2])          # [N,4,2]
        goal_b = to_body(goal[..., :2])          # [N,4,2]
        goal_rel_b = torch.cat(
            (goal_b - foot_b, (goal[..., 2] - foot[..., 2]).unsqueeze(-1)),
            dim=-1,
        )                                        # [N,4,3]

        s = torch.linspace(
            0.0, 1.0, self.cfg.path_samples,
            device=foot.device, dtype=torch.float32,
        ).view(1, 1, -1, 1)                      # [1,1,S,1]
        path_b = foot_b.unsqueeze(2) + s * (
            goal_b.unsqueeze(2) - foot_b.unsqueeze(2)
        )                                        # [N,4,S,2] (仅地形查询位置)
        h = heights.query(path_b)                # [N,4,S]
        path_rel_h = h - foot[..., 2].unsqueeze(-1)

        phase = self.gait.phases[:n]             # [N,4]
        swing_mask = self.gait.is_swing[:n]
        duty = self.gait.cfg.duty_factor
        progress = torch.where(
            swing_mask,
            (phase - duty).clamp(min=0.0) / max(1.0 - duty, 1e-9),
            torch.zeros_like(phase),
        )
        touchdown = swing_mask.logical_not() & self._prev_swing[:n]
        self._prev_swing[:n] = swing_mask

        return {
            "goal_rel_b": goal_rel_b,
            "path_rel_h": path_rel_h,
            "path_xy_b": path_b,
            "swing_mask": swing_mask,
            "swing_progress": progress,
            "phase": phase,
            "touchdown_events": touchdown,
        }
