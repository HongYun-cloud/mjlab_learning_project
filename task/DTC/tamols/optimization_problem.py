"""TAMOLS 轨迹优化问题: 变量布局与约束组装 (求解器无关).

决策变量与 :class:`~myrobot_mjlab.task.DTC.tamols.optimization_state.OptimizationStateLayout`
一一对应::

    spline_coeffs[p]       [6, K]  第 p 个相位的样条系数 a0..a_{K-1}
    landing_footholds      [L, 3]  落地的脚点
    desired_footholds      [L, 3]  期望的落脚点
    slack                  [S]     松弛变量

优化问题的数学形式在这里描述, 具体的求解器调用全部经过
:class:`~myrobot_mjlab.task.DTC.tamols.solver_backend.SolverBackend`.
当前默认后端是 CasADi + IPOPT; 之后替换自定义求解器时, 本文件的变量布局
和约束定义不需要修改.

已实现的约束
------------
1. 初始位姿约束 (:meth:`TrajectoryOptimizationProblem.add_initial_pose_constraint`)

   第一段轨迹 (phase 0) 的起点等于机身当前位姿::

       pose(0) = a_0 = [x, y, z, roll, pitch, yaw]^T

   因为样条在相位内按 ``pose(tau) = sum_i a_i * tau^i`` 定义, ``tau = 0``
   时只剩常数项 a0, 即系数矩阵的第 0 列.

2. 相邻段连接约束 (:meth:`TrajectoryOptimizationProblem.add_junction_constraints`)

   对每一对相邻相位 (k, k+1), **只添加一个等式**::

       sum_i a_i^(k) * T_k^i == a_0^(k+1)

   即上一段的终点与下一段的起点是同一个表达式, 而不是分别对中间点写两条约束.
   所有分量 (或 ``dims`` 选择的若干分量) 放在同一个向量等式中.

后续的初始速度约束、动力学约束等可以继续以
``self.backend.add_equality(...)`` / ``add_inequality(...)`` 的形式追加.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from myrobot_mjlab.task.DTC.tamols.optimization_state import (
    OptimizationStateLayout,
)
from myrobot_mjlab.task.DTC.tamols.solver_backend import (
    SolverBackend,
    make_backend,
)

BASE_DIMS = 6
POSE_COMPONENT_NAMES = ("x", "y", "z", "roll", "pitch", "yaw")


@dataclass
class DecisionVariables:
    """优化问题的决策变量 (后端表达式, 不是数值张量)."""

    spline_coeffs: list[Any]
    """长度 P 的列表, 每个元素为 [6, K] 的样条系数矩阵."""

    landing_footholds: Any
    """[L, 3] 落地的脚点."""

    desired_footholds: Any
    """[L, 3] 期望的落脚点."""

    slack: Any
    """[S] 松弛变量."""


class TrajectoryOptimizationProblem:
    """TAMOLS 基础姿态 + 落脚点的轨迹优化问题.

    Args:
      layout: 变量布局; 默认 ``OptimizationStateLayout()``
        (P=1, L=4, K=5, S=1).
      backend: 求解器后端; None 时用 ``make_backend("casadi", ...)`` 创建.
      plugin: 仅在后端为 None 时使用, 默认 ``"ipopt"``.
      solver_options: 仅在后端为 None 时使用, 传给求解器.
    """

    def __init__(
        self,
        layout: OptimizationStateLayout | None = None,
        backend: SolverBackend | None = None,
        *,
        plugin: str = "ipopt",
        solver_options: dict[str, Any] | None = None,
    ):
        self.layout = (
            layout if layout is not None else OptimizationStateLayout()
        )
        self.backend = (
            backend
            if backend is not None
            else make_backend(
                "casadi",
                plugin=plugin,
                solver_options=solver_options or {},
            )
        )
        self.vars = self._create_variables()

        self.initial_pose: np.ndarray | None = None
        """最近一次添加的初始位姿约束目标 [6]."""

    # ------------------------------------------------------------------ #
    # 变量
    # ------------------------------------------------------------------ #
    def _create_variables(self) -> DecisionVariables:
        layout = self.layout
        spline_coeffs = [
            self.backend.variable(
                (BASE_DIMS, layout.num_coeffs), name=f"spline_coeffs_{phase}"
            )
            for phase in range(layout.num_phases)
        ]
        landing_footholds = self.backend.variable(
            (layout.num_legs, 3), name="landing_footholds"
        )
        desired_footholds = self.backend.variable(
            (layout.num_legs, 3), name="desired_footholds"
        )
        slack = self.backend.variable((layout.num_slack,), name="slack")
        return DecisionVariables(
            spline_coeffs=spline_coeffs,
            landing_footholds=landing_footholds,
            desired_footholds=desired_footholds,
            slack=slack,
        )

    # ------------------------------------------------------------------ #
    # 约束
    # ------------------------------------------------------------------ #
    def add_initial_pose_constraint(
        self,
        pose: Sequence[float] | np.ndarray,
        *,
        phase: int = 0,
        dims: Sequence[int] | None = None,
    ) -> None:
        """约束指定相位起点等于机身当前位姿.

        对应 ``pose(tau=0) = a_0``. 默认约束全部 6 个分量::

            spline_coeffs[phase][:, 0] == pose

        Args:
          pose: 长度 6 的当前位姿 ``[x, y, z, roll, pitch, yaw]``,
            位置单位 m, 角度单位 rad, ZYX 欧拉角.
          phase: 相位索引, 默认 0 (第一段轨迹).
          dims: 可选的分量选择, 例如 ``(0, 1, 2)`` 只约束位置、
            ``(3, 4, 5)`` 只约束姿态; None 表示全部 6 维.

        Raises:
          ValueError: pose 长度/数值非法, phase 越界, dims 非法.
        """
        target = self._validate_pose(pose)
        if not 0 <= phase < self.layout.num_phases:
            raise ValueError(
                f"phase 必须在 [0, {self.layout.num_phases}) 内, 实际为 {phase}"
            )

        index = self._resolve_dims(dims)

        # a0: 系数矩阵 [6, K] 的第 0 列; 只取需要约束的分量.
        lhs = self.vars.spline_coeffs[phase][index, 0]
        rhs = target[index].reshape(-1, 1)
        self.backend.add_equality(lhs, rhs, name="initial_pose")
        self.initial_pose = target

    def add_junction_constraints(
        self,
        durations: float | Sequence[float] | np.ndarray,
        *,
        dims: Sequence[int] | None = None,
    ) -> int:
        """约束相邻相位的连接: 下一段起点 == 上一段终点.

        对每个相位边界 k (0 <= k < P-1) 只添加一个向量等式::

            sum_{i=0}^{K-1} a_i^(k) * T_k^i == a_0^(k+1)

        其中 ``T_k`` 是第 k 个相位的持续时间. 左边是 "这一段的终点",
        右边是 "下一段的起点", 两者写在同一个等式里.

        Args:
          durations: 各相位持续时间 (s). 标量表示所有相位相同;
            序列长度必须为 P.
          dims: 需要满足连接的分量, 默认全部 6 维.

        Returns:
          实际添加的等式个数, 即 ``P - 1`` (P=1 时为 0).

        Raises:
          ValueError: 持续时间非法 (非正/非有限/个数不符) 或 dims 非法.
        """
        num_phases = self.layout.num_phases
        phase_durations = self._resolve_durations(durations, num_phases)
        index = self._resolve_dims(dims)

        count = 0
        for phase in range(num_phases - 1):
            end = self._spline_end_value(
                phase, phase_durations[phase], index
            )
            start = self.vars.spline_coeffs[phase + 1][index, 0]
            self.backend.add_equality(
                end, start, name=f"junction_{phase}_{phase + 1}"
            )
            count += 1
        return count

    def _spline_end_value(
        self, phase: int, duration: float, index: list[int]
    ):
        """第 ``phase`` 段在 ``tau=duration`` 处的取值 (Horner 形式)."""
        coeffs = self.vars.spline_coeffs[phase]
        value = coeffs[index, self.layout.num_coeffs - 1]
        for i in range(self.layout.num_coeffs - 2, -1, -1):
            value = value * duration + coeffs[index, i]
        return value

    # ------------------------------------------------------------------ #
    # 求解
    # ------------------------------------------------------------------ #
    def solve(self) -> Any:
        """调用后端求解当前问题."""
        return self.backend.solve()

    def value(self, expr: Any) -> np.ndarray:
        """读取后端表达式在当前解处的数值."""
        return self.backend.value(expr)

    # ------------------------------------------------------------------ #
    # 工具
    # ------------------------------------------------------------------ #
    @staticmethod
    def _resolve_dims(dims: Sequence[int] | None) -> list[int]:
        """把可选的 dims 规范化为去重后的分量索引列表."""
        if dims is None:
            return list(range(BASE_DIMS))
        index = [int(d) for d in dims]
        if not index:
            raise ValueError("dims 不能为空")
        for d in index:
            if not 0 <= d < BASE_DIMS:
                raise ValueError(
                    f"dims 中的分量必须在 [0, {BASE_DIMS}) 内, 实际为 {d}"
                )
        if len(set(index)) != len(index):
            raise ValueError(f"dims 中有重复分量: {index}")
        return index

    @staticmethod
    def _resolve_durations(
        durations: float | Sequence[float] | np.ndarray, num_phases: int
    ) -> list[float]:
        """把标量/序列相位时长规范化为长度 P 的列表."""
        if np.isscalar(durations):
            values = [float(durations)] * num_phases
        else:
            arr = np.asarray(durations, dtype=float).reshape(-1)
            if arr.size == 1:
                values = [float(arr[0])] * num_phases
            elif arr.size == num_phases:
                values = [float(v) for v in arr]
            else:
                raise ValueError(
                    f"durations 需为标量或长度 {num_phases} 的序列, "
                    f"实际 size={arr.size}"
                )
        for value in values:
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"相位持续时间必须为有限正数, 实际为 {values}"
                )
        return values

    @staticmethod
    def _validate_pose(pose: Sequence[float] | np.ndarray) -> np.ndarray:
        arr = np.asarray(pose, dtype=float)
        if arr.size != BASE_DIMS:
            raise ValueError(
                f"pose 必须包含 {BASE_DIMS} 个分量 "
                f"{POSE_COMPONENT_NAMES}, 实际 size={arr.size}"
            )
        arr = arr.reshape(-1)
        if not np.isfinite(arr).all():
            raise ValueError(f"pose 中存在非有限值: {arr}")
        return arr
