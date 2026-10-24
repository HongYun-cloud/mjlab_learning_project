"""TAMOLS 论文 V-B 节的八项软目标 (objective / soft task).

本模块实现论文 V-B 的全部目标函数, 与已完成的约束模块
(:mod:`giac_constraints`, :mod:`v_a_constraints`, :mod:`optimization_problem`)
**互不修改**: 只从 ``v_a_constraints`` 复用符号/数值通用的小工具
(``point3`` / ``matvec3`` / ``spline_value`` / ``rotation_matrix_zyx``),
不改动其中任何约束行为.

硬约束与软目标的界限 (用户要求 3)
--------------------------------
论文 V-B1 明确把 "落脚点贴地" 写成**软任务** ``min (h(p_i) - ez.p_i)^2``
(正文: "such a formulation can lead to a suboptimal solution or even local
infeasibility. We therefore soften the task"), 本模块**不**提供任何把它
变成等式约束的接口; 装配时它只进入目标函数, 不进入
``backend.add_equality``. 其余七项同理, 全部是加权代价.

公式 -> 代码映射 (残差均为**未加权**标量代价, 最小化)
----------------------------------------------------
::

    V-B1 落脚点贴地      (h(p_i) - p_i_z)^2
                         -> foothold_ground_residual / foothold_ground_records
    V-B2 腿间防撞        z = ||(I-ez ez^T)(p_i-p_j)||^2   (式 32, 平方距离)
                         barrier = (max(eps_min - z, 0))^2  (式 32 下单侧二次 barrier)
                         -> leg_collision_barrier / leg_collision_records
    V-B3 名义运动学      ||p_B + R_B r_i - p_i - l_des||^2,  l_des=[0,0,h_des]  (式 33)
                         每条腿每个支撑相取一次, 时刻 = midstance
                         -> nominal_kinematics_residual / nominal_kinematics_records
    V-B4 机身对齐        sum_i (ez.(p_B + R_B r_i - l_des) - h_s2(p_B))^2      (式 34)
                         沿轨迹采样 (sampled)
                         -> base_pose_alignment_term / base_pose_alignment_records
    V-B5 避边缘          ||∇h(p_i)||^2 + ||∇h_s1(p_i)||^2                      (式 35)
                         -> edge_avoidance_residual / edge_avoidance_records
    V-B6 上一轮解        ||p_i - p_i_prev||^2
                         -> previous_solution_residual / previous_solution_records
    V-B7 动量跟踪        ||P^B - P_des^B||^2/m^2 + ||L^B - L_des^B||^2         (式 36)
                         = ||v^B - v_des^B||^2 + ||I_B (w^B - w_des^B)||^2
                         沿轨迹采样 (sampled)
                         -> tracking_residual / tracking_records
    V-B8 角动量平滑      ||L_dot_B||^2,  L_B = R_B I_B w^B (世界系角动量)
                         沿轨迹采样 (sampled)
                         -> smoothness_residual / smoothness_records

权重 (论文 Table I; 连续任务按 ``T_k = tau_k / num_samples`` 缩放,
即对时间做矩形积分, ``num_samples`` 默认 6, 与论文 "sampled ... for each
Tk = τk/6 seconds" 一致)::

    task                     sampled   weight
    footholds on ground                1e4
    leg collision avoidance            1e-3
    nominal kinematics                 7
    base pose alignment      X         100 * T_k
    edge avoidance                     3
    previous solution                  1e-2
    tracking                 X         2 * T_k
    smoothness               X         1e-3 * T_k

单位与缩放
----------
- V-B1/V-B3/V-B4/V-B6: 残差单位 m^2; V-B1 权重 1e4 使 kinematically
  feasible 问题的贴地误差保持在建图精度内 (<5 mm, 论文 V-C.3);
- V-B2: ``z`` 是**平方** xy 距离 (m^2), barrier 残差单位 m^4; ``eps_min``
  因此也是 m^2 (默认 0.01 = (0.1 m)^2, 论文未给数值, 见假设);
- V-B5: 梯度无量纲 (m/m), 残差无量纲;
- V-B7: 线动量项除以 m^2 后为 (m/s)^2, 与论文 "weight I_3x3 for the
  linear and I_B^T I_B for the angular velocity" 一致; 角动量项为
  (kg m^2/s)^2 量级, 由 I_B^T I_B 加权;
- V-B8: (kg m^2/s^2)^2 量级 (I_B 参与), 权重 1e-3 * T_k.

地图查询
--------
所有 h / h_s1 / h_s2 查询经过 :mod:`map_query` 的双线性插值层
(:class:`~myrobot_mjlab.task.DTC.tamols.map_query.TerrainMaps`), 边界
截断、缺失值填充与有效性诊断都在该层处理; 本模块不直接读网格, 也不
假设梯度在任意位置有效 (V-B5 的梯度场在缺失格附近由填充策略给出,
``filled_fraction`` / ``validity_at`` 可供调用方判断可信度).

动量约定
--------
- 机体系线速度 ``v^B = R_B^T ṗ_B``; 机体系角速度 ``w^B`` 由 ZYX 欧拉角
  及其导数经 :func:`body_omega_zyx` 得到 (与 ``base_spline.Spline6D`` 同
  公式);
- ``P^B = m v^B``, ``L^B = I_B w^B`` (机体系), 世界系角动量
  ``L_B^W = R_B I_B w^B``;
- ``L_dot_B`` (V-B8) 取世界系角动量的时间导数
  ``d/dt(R_B I_B w^B) = R_B (w^B x (I_B w^B) + I_B ẇ^B)``,
  其中 ``ẇ^B`` 由 :func:`body_omega_rate_zyx` 解析给出; 单测中与
  CasADi 符号时间求导和数值差分三方对照.

默认常量
--------
- ``DEFAULT_H_DES = 0.28`` m, 与工程 ``TAMOLSOptimizerCfg.h_des`` 一致;
- ``DEFAULT_EPS_MIN = 0.01`` m^2 (即脚间 xy 距离下限 0.1 m); 论文未给数
  值, 属待确认假设;
- ``GO2_MASS = 15.2064`` kg, ``GO2_INERTIA_B`` 为 go2.xml 名义站姿
  (thigh=0.9, calf=-1.8) 下用 MuJoCo 合成的整机惯量 (关于整机质心,
  机体系): 由 ``body_mass/body_inertia`` 并联公式合成, 对角约
  (0.2554, 0.4826, 0.4513) kg m^2, 含 xz 耦合 -0.0187. 论文假设机身
  几何中心与质心重合; 实际质心在几何中心下方约 2.5 cm, 该偏移按论文
  假设忽略 (待确认假设).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from myrobot_mjlab.task.DTC.tamols.map_query import TerrainMaps
from myrobot_mjlab.task.DTC.tamols.v_a_constraints import (
    GO2_STANCE_HIP_OFFSETS,
    LEG_ORDER,
    leg_index,
    matvec3,
    point3,
    rotation_matrix_zyx,
    spline_value,
)

EZ = (0.0, 0.0, 1.0)

DEFAULT_H_DES = 0.28
"""名义腿伸展高度 h_des [m], 与工程 ``TAMOLSOptimizerCfg.h_des`` 一致."""

DEFAULT_EPS_MIN = 0.01
"""V-B2 的 xy 平方距离下限 eps_min [m^2] (对应脚间距 0.1 m). 论文未给数值."""

GO2_MASS = 15.2064
"""Go2 整机质量 [kg] (go2.xml 全部 body 质量之和, MuJoCo 读取)."""

GO2_INERTIA_B = (
    (0.25544167, 0.0, -0.01867286),
    (0.0, 0.48264498, 0.0),
    (-0.01867286, 0.0, 0.45131243),
)
"""Go2 名义站姿整机惯量 I_B [kg m^2] (机体系, 关于整机质心). 见模块文档."""

CATEGORY_NAMES = {
    "V-B1": "footholds on ground",
    "V-B2": "leg collision avoidance",
    "V-B3": "nominal kinematics",
    "V-B4": "base pose alignment",
    "V-B5": "edge avoidance",
    "V-B6": "previous solution",
    "V-B7": "tracking",
    "V-B8": "smoothness",
    "29": "robustness margin (GIAC slack, 式 29)",
}

SAMPLED_CATEGORIES = ("V-B4", "V-B7", "V-B8")
"""沿轨迹采样的连续任务 (权重乘 T_k)."""


def _is_symbolic(value: Any) -> bool:
    module = type(value).__module__
    return module == "casadi" or module.startswith("casadi.")


def _maximum(a: Any, b: Any) -> Any:
    if _is_symbolic(a) or _is_symbolic(b):
        import casadi as ca

        return ca.fmax(a, b)
    return max(a, b)


def _dot3(a: Sequence[Any], b: Sequence[Any]) -> Any:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _sub3(a: Sequence[Any], b: Sequence[Any]) -> tuple[Any, Any, Any]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross3(a: Sequence[Any], b: Sequence[Any]) -> tuple[Any, Any, Any]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _squared_norm3(a: Sequence[Any]) -> Any:
    return _dot3(a, a)


def _foothold_at(footholds: Any, index: int) -> tuple[Any, Any, Any]:
    """取第 ``index`` 个脚点 (numpy / CasADi MX / 序列均可)."""
    shape = getattr(footholds, "shape", None)
    if shape is not None and len(shape) == 2:
        return point3(footholds[index, :])
    return point3(footholds[index])


# ---------------------------------------------------------------------- #
# 权重配置
# ---------------------------------------------------------------------- #
@dataclass(frozen=True)
class VBWeights:
    """论文 Table I 的软任务权重 (未含采样缩放).

    采样类任务 (``SAMPLED_CATEGORIES``) 的有效权重为
    ``weight * T_k``, ``T_k = phase_duration / num_samples``; 其余任务
    每个记录直接用表中权重.
    """

    footholds_on_ground: float = 1e4
    leg_collision: float = 1e-3
    nominal_kinematics: float = 7.0
    base_pose_alignment: float = 100.0
    edge_avoidance: float = 3.0
    previous_solution: float = 1e-2
    tracking: float = 2.0
    smoothness: float = 1e-3
    robustness_margin: float = 7e-3
    """GIAC 松弛 eps 的权重 (Table I "robustness margin", 每相位一项)."""
    num_samples: int = 6
    """每个相位的采样段数; T_k = tau_k / num_samples (论文 τk/6)."""

    _TABLE = {
        "V-B1": "footholds_on_ground",
        "V-B2": "leg_collision",
        "V-B3": "nominal_kinematics",
        "V-B4": "base_pose_alignment",
        "V-B5": "edge_avoidance",
        "V-B6": "previous_solution",
        "V-B7": "tracking",
        "V-B8": "smoothness",
        "29": "robustness_margin",
    }

    def base_weight(self, category: str) -> float:
        if category not in self._TABLE:
            raise ValueError(
                f"未知目标类别 {category!r}; 可选 {sorted(self._TABLE)}"
            )
        return float(getattr(self, self._TABLE[category]))

    def weight_for(self, category: str, phase_duration: float | None = None) -> float:
        """类别的有效权重; 采样类必须给 ``phase_duration`` [s]."""
        base = self.base_weight(category)
        if category not in SAMPLED_CATEGORIES:
            return base
        if phase_duration is None:
            raise ValueError(
                f"采样类目标 {category} 需要 phase_duration 以计算 "
                f"T_k = tau_k / {self.num_samples}"
            )
        if not (math.isfinite(phase_duration) and phase_duration > 0.0):
            raise ValueError(
                f"phase_duration 必须为有限正数, 实际为 {phase_duration}"
            )
        return base * phase_duration / self.num_samples


# ---------------------------------------------------------------------- #
# 记录
# ---------------------------------------------------------------------- #
@dataclass(frozen=True)
class ObjectiveRecord:
    """一项 (已定位到具体脚/相位/采样时刻的) 未加权代价.

    Attributes:
      name: 唯一名称, 例如 ``"VB1_ground_leg0"``.
      category: ``"V-B1"`` .. ``"V-B8"``.
      residual: 未加权标量代价 (数值或 CasADi MX).
      weight: 有效权重 (采样类已含 T_k 因子).
      sampled: 是否沿轨迹采样 (Table I 的 "sampled" 列).
      phase / tau: 采样所在相位与相位内时刻 [s]; 非采样任务为 None.
      foot_indices: 参与的脚在输入数组中的下标.
      note: 备注 (单位、来源等).
    """

    name: str
    category: str
    residual: Any
    weight: float
    sampled: bool = False
    phase: int | None = None
    tau: float | None = None
    foot_indices: tuple[int, ...] = ()
    note: str = ""
    roots: tuple[Any, ...] = ()
    """根残差 s_k, 满足 ``residual = sum_k s_k^2`` (加性字段, 默认空).

    供高斯-牛顿 SQP (:mod:`gn_sqp_solver`) 构造 PSD 近似 Hessian
    ``H = sum 2 w ∇s_k s_k^T``; 线性项 (式 29 的 eps) 没有根残差,
    保持空元组, 其曲率按 0 处理 (梯度仍精确进入 QP 线性项).
    """

    def __post_init__(self) -> None:
        if self.category not in CATEGORY_NAMES:
            raise ValueError(
                f"未知目标类别 {self.category!r}; 可选 {sorted(CATEGORY_NAMES)}"
            )
        if not math.isfinite(self.weight):
            raise ValueError(f"权重必须有限, 实际为 {self.weight}")

    def weighted_residual(self) -> Any:
        """``weight * residual`` (进入总目标的项)."""
        return self.weight * self.residual

    def value(self, evaluate: Callable[[Any], Any] | None = None) -> float:
        """未加权残差的数值; 符号残差需传 ``evaluate``."""
        residual = self.residual
        if evaluate is not None:
            residual = evaluate(residual)
        if hasattr(residual, "full"):
            residual = residual.full()
        return float(np.asarray(residual, dtype=float).reshape(-1)[0])

    def weighted_value(self, evaluate: Callable[[Any], Any] | None = None) -> float:
        return self.weight * self.value(evaluate)


@dataclass(frozen=True)
class ObjectiveSet:
    """一组目标记录."""

    records: tuple[ObjectiveRecord, ...]
    note: str = ""

    def __iter__(self):
        return iter(self.records)

    def __len__(self) -> int:
        return len(self.records)

    def by_category(self) -> dict[str, tuple[ObjectiveRecord, ...]]:
        grouped: dict[str, list[ObjectiveRecord]] = {}
        for record in self.records:
            grouped.setdefault(record.category, []).append(record)
        return {key: tuple(value) for key, value in grouped.items()}

    def values(
        self, evaluate: Callable[[Any], Any] | None = None
    ) -> dict[str, float]:
        """``{name: 未加权残差数值}``."""
        return {record.name: record.value(evaluate) for record in self.records}

    def weighted_values(
        self, evaluate: Callable[[Any], Any] | None = None
    ) -> dict[str, float]:
        """``{name: 加权残差数值}``."""
        return {
            record.name: record.weighted_value(evaluate)
            for record in self.records
        }

    def total(self, evaluate: Callable[[Any], Any] | None = None) -> Any:
        """加权残差之和 (符号或数值)."""
        if not self.records:
            return 0.0
        total = self.records[0].weighted_residual()
        for record in self.records[1:]:
            total = total + record.weighted_residual()
        if evaluate is not None:
            total = evaluate(total)
        if hasattr(total, "full"):
            total = total.full()
        return total

    def describe(self) -> str:
        lines = [f"ObjectiveSet: {len(self.records)} records"]
        for category, records in self.by_category().items():
            lines.append(
                f"  {category} {CATEGORY_NAMES[category]:<26s} "
                f"n={len(records):<3d} weight={records[0].weight:.4g}"
                f"{' (sampled, xT_k)' if records[0].sampled else ''}"
            )
        if self.note:
            lines.append(f"  note: {self.note}")
        return "\n".join(lines)


def merge_objective_sets(*sets: ObjectiveSet) -> ObjectiveSet:
    """合并多个 ObjectiveSet (记录按输入顺序拼接)."""
    records: list[ObjectiveRecord] = []
    notes: list[str] = []
    for item in sets:
        records.extend(item.records)
        if item.note:
            notes.append(item.note)
    return ObjectiveSet(records=tuple(records), note="; ".join(notes))


def assemble_total_objective(*sets: ObjectiveSet) -> Any:
    """总目标函数表达式: 所有集合的加权残差之和 (CasADi 组装入口).

    返回的表达式可以直接 ``backend.add_objective(...)`` 或
    ``opti.minimize(...)``; 每个记录项已含权重.
    """
    return merge_objective_sets(*sets).total()


# ---------------------------------------------------------------------- #
# 运动学小工具 (numpy / CasADi MX 通用)
# ---------------------------------------------------------------------- #
def desired_leg_extension(h_des: Any) -> tuple[Any, Any, Any]:
    """``l_des = [0, 0, h_des]^T`` (式 33). 数值输入时校验有限性."""
    if isinstance(h_des, (int, float, np.integer, np.floating)):
        if not math.isfinite(float(h_des)):
            raise ValueError(f"h_des 必须有限, 实际为 {h_des}")
    return (0.0, 0.0, h_des)


def hip_position_world(
    p_B: Sequence[Any], R_B: Any, r_i: Sequence[Any]
) -> tuple[Any, Any, Any]:
    """``p_B + R_B r_i^B`` (髋/大腿中心的世界系位置)."""
    rotated = matvec3(R_B, r_i)
    return (
        p_B[0] + rotated[0],
        p_B[1] + rotated[1],
        p_B[2] + rotated[2],
    )


def body_omega_zyx(
    euler: Sequence[Any], euler_rates: Sequence[Any]
) -> tuple[Any, Any, Any]:
    """ZYX 欧拉角 -> 机体系角速度 (与 ``base_spline`` 同公式)."""
    phi, theta, _ = euler[0], euler[1], euler[2]
    phi_dot, theta_dot, psi_dot = euler_rates[0], euler_rates[1], euler_rates[2]
    sin_phi, cos_phi = math_sin(phi), math_cos(phi)
    sin_theta, cos_theta = math_sin(theta), math_cos(theta)
    return (
        phi_dot - psi_dot * sin_theta,
        theta_dot * cos_phi + psi_dot * sin_phi * cos_theta,
        -theta_dot * sin_phi + psi_dot * cos_phi * cos_theta,
    )


def body_omega_rate_zyx(
    euler: Sequence[Any],
    euler_rates: Sequence[Any],
    euler_accels: Sequence[Any],
) -> tuple[Any, Any, Any]:
    """``body_omega_zyx`` 对时间的导数 ẇ^B (解析式)."""
    phi, theta, _ = euler[0], euler[1], euler[2]
    phi_dot, theta_dot, psi_dot = euler_rates[0], euler_rates[1], euler_rates[2]
    phi_dd, theta_dd, psi_dd = euler_accels[0], euler_accels[1], euler_accels[2]
    sin_phi, cos_phi = math_sin(phi), math_cos(phi)
    sin_theta, cos_theta = math_sin(theta), math_cos(theta)
    wx = phi_dd - psi_dd * sin_theta - psi_dot * theta_dot * cos_theta
    wy = (
        theta_dd * cos_phi
        - theta_dot * phi_dot * sin_phi
        + psi_dd * sin_phi * cos_theta
        + psi_dot * phi_dot * cos_phi * cos_theta
        - psi_dot * theta_dot * sin_phi * sin_theta
    )
    wz = (
        -theta_dd * sin_phi
        - theta_dot * phi_dot * cos_phi
        + psi_dd * cos_phi * cos_theta
        - psi_dot * phi_dot * sin_phi * cos_theta
        - psi_dot * theta_dot * cos_phi * sin_theta
    )
    return (wx, wy, wz)


def math_sin(value: Any) -> Any:
    if _is_symbolic(value):
        import casadi as ca

        return ca.sin(value)
    return math.sin(float(value))


def math_cos(value: Any) -> Any:
    if _is_symbolic(value):
        import casadi as ca

        return ca.cos(value)
    return math.cos(float(value))


def linear_velocity_base(
    R_B: Any, p_dot_world: Sequence[Any]
) -> tuple[Any, Any, Any]:
    """``v^B = R_B^T ṗ_B`` (机体系线速度)."""
    return (
        R_B[0][0] * p_dot_world[0]
        + R_B[1][0] * p_dot_world[1]
        + R_B[2][0] * p_dot_world[2],
        R_B[0][1] * p_dot_world[0]
        + R_B[1][1] * p_dot_world[1]
        + R_B[2][1] * p_dot_world[2],
        R_B[0][2] * p_dot_world[0]
        + R_B[1][2] * p_dot_world[1]
        + R_B[2][2] * p_dot_world[2],
    )


def angular_momentum_world(
    R_B: Any, inertia_B: Any, omega_b: Sequence[Any]
) -> tuple[Any, Any, Any]:
    """世界系角动量 ``L_B = R_B I_B w^B``."""
    return matvec3(R_B, matvec3(inertia_B, omega_b))


def angular_momentum_rate_world(
    R_B: Any,
    inertia_B: Any,
    omega_b: Sequence[Any],
    omega_dot_b: Sequence[Any],
) -> tuple[Any, Any, Any]:
    """``d/dt (R_B I_B w^B) = R_B (w^B x (I_B w^B) + I_B ẇ^B)``."""
    i_omega = matvec3(inertia_B, omega_b)
    inner = _cross3(omega_b, i_omega)
    i_omega_dot = matvec3(inertia_B, omega_dot_b)
    return matvec3(
        R_B,
        (
            inner[0] + i_omega_dot[0],
            inner[1] + i_omega_dot[1],
            inner[2] + i_omega_dot[2],
        ),
    )


# ---------------------------------------------------------------------- #
# 八项残差 (未加权标量代价)
# ---------------------------------------------------------------------- #
def foothold_ground_residual(h_at_p: Any, p_i: Sequence[Any]) -> Any:
    """V-B1: ``(h(p_i) - ez.p_i)^2`` [m^2]. 软任务, 不作等式约束."""
    p_i = point3(p_i, "p_i")
    return (h_at_p - p_i[2]) ** 2


def leg_collision_barrier(
    p_i: Sequence[Any], p_j: Sequence[Any], eps_min: float = DEFAULT_EPS_MIN
) -> Any:
    """V-B2: 单侧二次 barrier [m^4].

    ``z = ||(I - ez ez^T)(p_i - p_j)||^2`` (式 32, **平方** xy 距离)::

        barrier = (eps_min - z)^2,  z < eps_min
                  0,                z >= eps_min

    用 ``max(eps_min - z, 0)^2`` 统一表达: C^1 连续, 在 z >= eps_min 区
    间梯度恒为 0.
    """
    if isinstance(eps_min, (int, float, np.integer, np.floating)):
        if not (math.isfinite(float(eps_min)) and float(eps_min) > 0.0):
            raise ValueError(f"eps_min 必须为有限正数 [m^2], 实际为 {eps_min}")
    p_i = point3(p_i, "p_i")
    p_j = point3(p_j, "p_j")
    z = (p_i[0] - p_j[0]) ** 2 + (p_i[1] - p_j[1]) ** 2
    return _maximum(eps_min - z, 0.0) ** 2


def nominal_kinematics_residual(
    p_B: Sequence[Any],
    R_B: Any,
    r_i: Sequence[Any],
    p_i: Sequence[Any],
    h_des: float = DEFAULT_H_DES,
) -> Any:
    """V-B3: ``||p_B + R_B r_i - p_i - l_des||^2`` [m^2] (式 33)."""
    hip = hip_position_world(p_B, R_B, r_i)
    diff = _sub3(hip, point3(p_i, "p_i"))
    l_des = desired_leg_extension(h_des)
    return _squared_norm3(_sub3(diff, l_des))


def base_pose_alignment_term(
    p_B: Sequence[Any],
    R_B: Any,
    r_i: Sequence[Any],
    hs2_at_pB: Any,
    h_des: float = DEFAULT_H_DES,
) -> Any:
    """V-B4 的单腿项 ``(ez.(p_B + R_B r_i - l_des) - h_s2(p_B))^2`` [m^2].

    论文式 (34) 是四条腿求和; 本函数给出求和中的单项, 记录装配时对
    i=1..4 累加.
    """
    hip = hip_position_world(p_B, R_B, r_i)
    l_des = desired_leg_extension(h_des)
    height = hip[2] - l_des[2]
    return (height - hs2_at_pB) ** 2


def edge_avoidance_residual(
    grad_h: Sequence[Any], grad_hs1: Sequence[Any]
) -> Any:
    """V-B5: ``∇h^T ∇h + ∇h_s1^T ∇h_s1`` [无量纲] (式 35).

    梯度是 xy 平面上的 2 维向量, 因此这里是 2 分量的内积.
    """
    for label, value in (("grad_h", grad_h), ("grad_hs1", grad_hs1)):
        if len(value) != 2:
            raise ValueError(
                f"{label} 必须是 2 维平面梯度 (gx, gy), 实际为 {len(value)} 个分量"
            )
    return (
        grad_h[0] * grad_h[0]
        + grad_h[1] * grad_h[1]
        + grad_hs1[0] * grad_hs1[0]
        + grad_hs1[1] * grad_hs1[1]
    )


def previous_solution_residual(
    p_i: Sequence[Any], p_prev: Sequence[Any]
) -> Any:
    """V-B6: ``||p_i - p_i_prev||^2`` [m^2]."""
    return _squared_norm3(_sub3(point3(p_i, "p_i"), point3(p_prev, "p_prev")))


def tracking_residual(
    v_b: Sequence[Any],
    v_des_b: Sequence[Any],
    omega_b: Sequence[Any],
    omega_des_b: Sequence[Any],
    inertia_B: Any = GO2_INERTIA_B,
) -> Any:
    """V-B7: ``||P-P_des||^2/m^2 + ||L-L_des||^2`` (式 36).

    机体系下 ``P = m v``, ``L = I_B w``, 故等价于
    ``||v-v_des||^2 + ||I_B (w-w_des)||^2`` (论文的权重 I_3 与 I_B^T I_B).
    """
    dv = _sub3(v_b, v_des_b)
    dw = _sub3(omega_b, omega_des_b)
    return _squared_norm3(dv) + _squared_norm3(matvec3(inertia_B, dw))


def smoothness_residual(
    L_dot_world: Sequence[Any],
) -> Any:
    """V-B8: ``||L_dot_B||^2`` [(kg m^2/s^2)^2]."""
    return _squared_norm3(L_dot_world)


# ---------------------------------------------------------------------- #
# 采样时刻 / midstance
# ---------------------------------------------------------------------- #
def default_sample_times(duration: float, num_samples: int = 6) -> tuple[float, ...]:
    """相位内采样时刻: ``linspace(0, T, n+1)[:-1]`` (与 V-A 模块一致)."""
    if not (math.isfinite(duration) and duration > 0.0):
        raise ValueError(f"duration 必须为有限正数, 实际为 {duration}")
    if not isinstance(num_samples, int) or num_samples < 1:
        raise ValueError(f"num_samples 必须为正整数, 实际为 {num_samples!r}")
    step = duration / num_samples
    return tuple(step * k for k in range(num_samples))


def midstance_from_contacts(
    contact_sets: Sequence[Sequence[int | str]],
    durations: Sequence[float],
) -> dict[int, tuple[int, float]]:
    """由每相位支撑腿集合推每条腿的 midstance ``(phase, tau)``.

    取该腿**连续**支撑相位段 (第一次出现到第一次中断) 的全局中点, 再
    换算到所在相位的局部时刻. 论文 V-B3: "the objective is applied once
    per stance ... we define this time as midstance". 从未支撑的腿不出
    现在返回值中 (调用方应跳过该腿的 V-B3).
    """
    durations = [float(d) for d in durations]
    for value in durations:
        if not (math.isfinite(value) and value > 0.0):
            raise ValueError(f"durations 必须为有限正数, 实际为 {durations}")
    starts = [0.0]
    for value in durations[:-1]:
        starts.append(starts[-1] + value)

    result: dict[int, tuple[int, float]] = {}
    for leg in range(len(LEG_ORDER)):
        run: list[int] = []
        for phase, contacts in enumerate(contact_sets):
            inside = any(leg_index(c) == leg for c in contacts)
            if inside:
                if run and run[-1] != phase - 1:
                    break
                run.append(phase)
            elif run:
                break
        if not run:
            continue
        t_start = starts[run[0]]
        t_end = starts[run[-1]] + durations[run[-1]]
        t_mid = 0.5 * (t_start + t_end)
        phase = run[-1]
        for candidate in run:
            if t_mid < starts[candidate] + durations[candidate]:
                phase = candidate
                break
        result[leg] = (phase, t_mid - starts[phase])
    return result


# ---------------------------------------------------------------------- #
# 记录装配: 每项目标一个独立函数
# ---------------------------------------------------------------------- #
def _resolve_legs(legs: Sequence[int | str] | None) -> tuple[int, ...]:
    if legs is None:
        return tuple(range(len(LEG_ORDER)))
    return tuple(leg_index(leg) for leg in legs)


def foothold_ground_records(
    maps: TerrainMaps,
    footholds: Any,
    legs: Sequence[int | str] | None = None,
    weights: VBWeights | None = None,
    phase: int | None = None,
) -> ObjectiveSet:
    """V-B1 记录: 每个期望落脚点一条 ``(h(p_i) - p_i_z)^2``."""
    weights = weights or VBWeights()
    records = []
    for leg in _resolve_legs(legs):
        p_i = _foothold_at(footholds, leg)
        h_at_p = maps.height_h(p_i[0], p_i[1])
        records.append(
            ObjectiveRecord(
                name=f"VB1_ground_leg{leg}",
                category="V-B1",
                residual=foothold_ground_residual(h_at_p, p_i),
                weight=weights.base_weight("V-B1"),
                phase=phase,
                foot_indices=(leg,),
                note="soft task, unit m^2",
                roots=(h_at_p - p_i[2],),
            )
        )
    return ObjectiveSet(records=tuple(records))


def leg_collision_records(
    footholds: Any,
    eps_min: float = DEFAULT_EPS_MIN,
    pairs: Sequence[Sequence[int]] | None = None,
    weights: VBWeights | None = None,
) -> ObjectiveSet:
    """V-B2 记录: 每对脚一条单侧二次 barrier (默认所有无序对 i<j)."""
    weights = weights or VBWeights()
    if pairs is None:
        pairs = tuple(combinations(range(len(LEG_ORDER)), 2))
    records = []
    for i, j in pairs:
        i, j = int(i), int(j)
        if i == j:
            raise ValueError(f"pairs 中不允许 i == j, 实际为 ({i}, {j})")
        p_i = _foothold_at(footholds, i)
        p_j = _foothold_at(footholds, j)
        z = (p_i[0] - p_j[0]) ** 2 + (p_i[1] - p_j[1]) ** 2
        records.append(
            ObjectiveRecord(
                name=f"VB2_collision_{i}_{j}",
                category="V-B2",
                residual=leg_collision_barrier(p_i, p_j, eps_min),
                weight=weights.base_weight("V-B2"),
                foot_indices=(i, j),
                note="one-sided quadratic barrier, unit m^4",
                roots=(_maximum(eps_min - z, 0.0),),
            )
        )
    return ObjectiveSet(records=tuple(records))


def nominal_kinematics_records(
    spline_coeffs: Sequence[Any],
    footholds: Any,
    midstance: Mapping[int | str, tuple[int, float]],
    r_hips: Any = GO2_STANCE_HIP_OFFSETS,
    h_des: float = DEFAULT_H_DES,
    weights: VBWeights | None = None,
) -> ObjectiveSet:
    """V-B3 记录: 每条腿在 midstance 时刻一条式 (33).

    Args:
      spline_coeffs: 每相位一个 ``(6, K)`` 系数 (数值或 MX).
      midstance: ``{腿: (phase, tau)}``, 见 :func:`midstance_from_contacts`.
    """
    weights = weights or VBWeights()
    records = []
    for leg, (phase, tau) in midstance.items():
        index = leg_index(leg)
        phase = int(phase)
        if not 0 <= phase < len(spline_coeffs):
            raise ValueError(
                f"midstance 的相位 {phase} 越界 (共 {len(spline_coeffs)} 段)"
            )
        pose = spline_value(spline_coeffs[phase], float(tau), order=0)
        R_B = rotation_matrix_zyx(pose[3], pose[4], pose[5])
        hip = hip_position_world(pose[0:3], R_B, _foothold_at(r_hips, index))
        p_i = _foothold_at(footholds, index)
        l_des = desired_leg_extension(h_des)
        records.append(
            ObjectiveRecord(
                name=f"VB3_nomkin_leg{index}_phase{phase}_tau{float(tau):.6g}",
                category="V-B3",
                residual=nominal_kinematics_residual(
                    pose[0:3], R_B, _foothold_at(r_hips, index), p_i, h_des,
                ),
                weight=weights.base_weight("V-B3"),
                phase=phase,
                tau=float(tau),
                foot_indices=(index,),
                note="once per stance at midstance, unit m^2",
                roots=_sub3(_sub3(hip, p_i), l_des),
            )
        )
    return ObjectiveSet(records=tuple(records))


def base_pose_alignment_records(
    spline_coeffs: Sequence[Any],
    durations: Sequence[float],
    maps: TerrainMaps,
    r_hips: Any = GO2_STANCE_HIP_OFFSETS,
    h_des: float = DEFAULT_H_DES,
    weights: VBWeights | None = None,
    sample_times: Sequence[Sequence[float]] | None = None,
) -> ObjectiveSet:
    """V-B4 记录: 每相位每采样时刻四条腿各一条式 (34) 单项.

    权重为 ``100 * T_k``, ``T_k = tau_k / num_samples`` (sampled 任务).
    """
    weights = weights or VBWeights()
    records = []
    for phase, coeffs in enumerate(spline_coeffs):
        times = (
            tuple(float(t) for t in sample_times[phase])
            if sample_times is not None
            else default_sample_times(durations[phase], weights.num_samples)
        )
        weight = weights.weight_for("V-B4", float(durations[phase]))
        for tau in times:
            pose = spline_value(coeffs, tau, order=0)
            p_B = pose[0:3]
            R_B = rotation_matrix_zyx(pose[3], pose[4], pose[5])
            hs2 = maps.height_hs2(p_B[0], p_B[1])
            for leg in range(len(LEG_ORDER)):
                hip = hip_position_world(p_B, R_B, _foothold_at(r_hips, leg))
                records.append(
                    ObjectiveRecord(
                        name=f"VB4_align_phase{phase}_tau{tau:.6g}_leg{leg}",
                        category="V-B4",
                        residual=base_pose_alignment_term(
                            p_B, R_B, _foothold_at(r_hips, leg), hs2, h_des
                        ),
                        weight=weight,
                        sampled=True,
                        phase=phase,
                        tau=float(tau),
                        foot_indices=(leg,),
                        note="sampled, unit m^2",
                        roots=(hip[2] - h_des - hs2,),
                    )
                )
    return ObjectiveSet(records=tuple(records))


def edge_avoidance_records(
    maps: TerrainMaps,
    footholds: Any,
    legs: Sequence[int | str] | None = None,
    weights: VBWeights | None = None,
) -> ObjectiveSet:
    """V-B5 记录: 每个期望落脚点一条式 (35)."""
    weights = weights or VBWeights()
    records = []
    for leg in _resolve_legs(legs):
        p_i = _foothold_at(footholds, leg)
        grad_h = maps.grad_h(p_i[0], p_i[1])
        grad_hs1 = maps.grad_hs1(p_i[0], p_i[1])
        records.append(
            ObjectiveRecord(
                name=f"VB5_edge_leg{leg}",
                category="V-B5",
                residual=edge_avoidance_residual(grad_h, grad_hs1),
                weight=weights.base_weight("V-B5"),
                foot_indices=(leg,),
                note="dimensionless (m/m)^2",
                roots=(grad_h[0], grad_h[1], grad_hs1[0], grad_hs1[1]),
            )
        )
    return ObjectiveSet(records=tuple(records))


def previous_solution_records(
    footholds: Any,
    previous_footholds: Any,
    legs: Sequence[int | str] | None = None,
    weights: VBWeights | None = None,
) -> ObjectiveSet:
    """V-B6 记录: 每个期望落脚点相对上一轮最优解的平方距离."""
    weights = weights or VBWeights()
    records = []
    for leg in _resolve_legs(legs):
        p_i = _foothold_at(footholds, leg)
        p_prev = _foothold_at(previous_footholds, leg)
        records.append(
            ObjectiveRecord(
                name=f"VB6_prev_leg{leg}",
                category="V-B6",
                residual=previous_solution_residual(p_i, p_prev),
                weight=weights.base_weight("V-B6"),
                foot_indices=(leg,),
                note="unit m^2",
                roots=_sub3(p_i, p_prev),
            )
        )
    return ObjectiveSet(records=tuple(records))


def _spline_pose_rates(
    coeffs: Any, tau: float
) -> tuple[tuple[Any, ...], tuple[Any, ...], tuple[Any, ...]]:
    return (
        spline_value(coeffs, tau, order=0),
        spline_value(coeffs, tau, order=1),
        spline_value(coeffs, tau, order=2),
    )


def tracking_records(
    spline_coeffs: Sequence[Any],
    durations: Sequence[float],
    ref_twist: Callable[[int, float], tuple[Sequence[float], Sequence[float]]],
    inertia_B: Any = GO2_INERTIA_B,
    weights: VBWeights | None = None,
    sample_times: Sequence[Sequence[float]] | None = None,
) -> ObjectiveSet:
    """V-B7 记录: 每相位每采样时刻一条式 (36).

    Args:
      ref_twist: ``(phase, tau) -> (v_des_b, omega_des_b)``, 机体系参考
        线速度 [m/s] 与角速度 [rad/s].
    """
    weights = weights or VBWeights()
    records = []
    for phase, coeffs in enumerate(spline_coeffs):
        times = (
            tuple(float(t) for t in sample_times[phase])
            if sample_times is not None
            else default_sample_times(durations[phase], weights.num_samples)
        )
        weight = weights.weight_for("V-B7", float(durations[phase]))
        for tau in times:
            pose, rate, _ = _spline_pose_rates(coeffs, tau)
            R_B = rotation_matrix_zyx(pose[3], pose[4], pose[5])
            v_b = linear_velocity_base(R_B, rate[0:3])
            omega_b = body_omega_zyx(pose[3:6], rate[3:6])
            v_des, omega_des = ref_twist(phase, float(tau))
            v_des = point3(v_des, "v_des_b")
            omega_des = point3(omega_des, "omega_des_b")
            dv = _sub3(v_b, v_des)
            i_dw = matvec3(inertia_B, _sub3(omega_b, omega_des))
            records.append(
                ObjectiveRecord(
                    name=f"VB7_track_phase{phase}_tau{tau:.6g}",
                    category="V-B7",
                    residual=tracking_residual(
                        v_b, v_des, omega_b, omega_des, inertia_B,
                    ),
                    weight=weight,
                    sampled=True,
                    phase=phase,
                    tau=float(tau),
                    note="sampled, (m/s)^2 + (kg m^2/s)^2",
                    roots=tuple(dv) + tuple(i_dw),
                )
            )
    return ObjectiveSet(records=tuple(records))


def smoothness_records(
    spline_coeffs: Sequence[Any],
    durations: Sequence[float],
    inertia_B: Any = GO2_INERTIA_B,
    weights: VBWeights | None = None,
    sample_times: Sequence[Sequence[float]] | None = None,
    l_dot_provider: Callable[[int, float], Sequence[Any]] | None = None,
) -> ObjectiveSet:
    """V-B8 记录: 每相位每采样时刻一条 ``||L_dot_B||^2``.

    Args:
      l_dot_provider: 可选 ``(phase, tau) -> L_dot 三元组``. 集成层用它把
        **同一个表达式节点**同时喂给 GIAC 约束与本目标 (见
        :mod:`nlp_integration`); None 时在本函数内部按样条现算 (独立使用
        时的默认行为, 公式与 provider 版本完全相同).
    """
    weights = weights or VBWeights()
    records = []
    for phase, coeffs in enumerate(spline_coeffs):
        times = (
            tuple(float(t) for t in sample_times[phase])
            if sample_times is not None
            else default_sample_times(durations[phase], weights.num_samples)
        )
        weight = weights.weight_for("V-B8", float(durations[phase]))
        for tau in times:
            if l_dot_provider is not None:
                L_dot = l_dot_provider(phase, float(tau))
            else:
                pose, rate, accel = _spline_pose_rates(coeffs, tau)
                R_B = rotation_matrix_zyx(pose[3], pose[4], pose[5])
                omega_b = body_omega_zyx(pose[3:6], rate[3:6])
                omega_dot_b = body_omega_rate_zyx(
                    pose[3:6], rate[3:6], accel[3:6]
                )
                L_dot = angular_momentum_rate_world(
                    R_B, inertia_B, omega_b, omega_dot_b
                )
            records.append(
                ObjectiveRecord(
                    name=f"VB8_smooth_phase{phase}_tau{tau:.6g}",
                    category="V-B8",
                    residual=smoothness_residual(L_dot),
                    weight=weight,
                    sampled=True,
                    phase=phase,
                    tau=float(tau),
                    note="sampled, (kg m^2/s^2)^2",
                    roots=tuple(L_dot),
                )
            )
    return ObjectiveSet(records=tuple(records))


# ---------------------------------------------------------------------- #
# 一键装配
# ---------------------------------------------------------------------- #
def build_v_b_objective_sets(
    *,
    maps: TerrainMaps,
    spline_coeffs: Sequence[Any],
    durations: Sequence[float],
    footholds: Any,
    previous_footholds: Any | None = None,
    contact_sets: Sequence[Sequence[int | str]] | None = None,
    midstance: Mapping[int | str, tuple[int, float]] | None = None,
    r_hips: Any = GO2_STANCE_HIP_OFFSETS,
    h_des: float = DEFAULT_H_DES,
    eps_min: float = DEFAULT_EPS_MIN,
    inertia_B: Any = GO2_INERTIA_B,
    weights: VBWeights | None = None,
    ref_twist: Callable[[int, float], tuple[Sequence[float], Sequence[float]]]
    | None = None,
    sample_times: Sequence[Sequence[float]] | None = None,
    collision_pairs: Sequence[Sequence[int]] | None = None,
) -> ObjectiveSet:
    """按输入可用性装配全部八项目标, 返回合并后的 :class:`ObjectiveSet`.

    缺输入的项被跳过并在 ``note`` 中说明 (不伪造数据):
    ``previous_footholds=None`` 跳过 V-B6; ``ref_twist=None`` 跳过 V-B7;
    ``midstance`` 与 ``contact_sets`` 都为 None 时跳过 V-B3.
    """
    weights = weights or VBWeights()
    durations = [float(d) for d in durations]
    if len(durations) != len(spline_coeffs):
        raise ValueError(
            f"durations 个数 {len(durations)} 与样条段数 "
            f"{len(spline_coeffs)} 不一致"
        )
    notes: list[str] = []
    sets: list[ObjectiveSet] = [
        foothold_ground_records(maps, footholds, weights=weights),
        leg_collision_records(
            footholds, eps_min=eps_min, pairs=collision_pairs, weights=weights
        ),
        base_pose_alignment_records(
            spline_coeffs, durations, maps,
            r_hips=r_hips, h_des=h_des, weights=weights,
            sample_times=sample_times,
        ),
        edge_avoidance_records(maps, footholds, weights=weights),
        smoothness_records(
            spline_coeffs, durations, inertia_B=inertia_B,
            weights=weights, sample_times=sample_times,
        ),
    ]
    if midstance is None and contact_sets is not None:
        midstance = midstance_from_contacts(contact_sets, durations)
    if midstance is not None:
        sets.insert(
            2,
            nominal_kinematics_records(
                spline_coeffs, footholds, midstance,
                r_hips=r_hips, h_des=h_des, weights=weights,
            ),
        )
    else:
        notes.append("V-B3 skipped (no midstance/contact_sets)")
    if previous_footholds is not None:
        sets.append(
            previous_solution_records(footholds, previous_footholds, weights=weights)
        )
    else:
        notes.append("V-B6 skipped (no previous_footholds)")
    if ref_twist is not None:
        sets.append(
            tracking_records(
                spline_coeffs, durations, ref_twist,
                inertia_B=inertia_B, weights=weights,
                sample_times=sample_times,
            )
        )
    else:
        notes.append("V-B7 skipped (no ref_twist)")
    merged = merge_objective_sets(*sets)
    return ObjectiveSet(records=merged.records, note="; ".join(notes))


def add_v_b_objectives(
    backend: Any, objective_set: ObjectiveSet
) -> ObjectiveSet:
    """把目标集合的加权项累加进 ``backend`` (CasADi 组装入口).

    调用方在添加完全部目标后调用 ``backend.minimize()`` 一次. 本函数
    **不**添加任何约束: V-B 全部是软目标 (用户要求 3).
    """
    for record in objective_set.records:
        backend.add_objective(record.weighted_residual(), name=record.name)
    return objective_set
