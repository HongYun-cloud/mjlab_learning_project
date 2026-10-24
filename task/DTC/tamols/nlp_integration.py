"""TAMOLS 完整 NLP 集成层: 统一变量布局 + 统一 L_dot + 全约束/全目标装配.

本模块**不修改**任何已通过测试的模块 (``giac_constraints`` /
``v_a_constraints`` / ``v_b_objectives`` / ``optimization_problem`` /
``optimization_state`` / ``map_query``), 只在它们之上做装配:

调用流程::

    params = NLPParameters(...)                 # 参数 (测量/地图/惯量/参考)
    assumptions = AssumptionsConfig(...)        # 待核实假设 (进入求解日志)
    nlp = IntegratedNLP(durations=..., schedules=..., maps=..., params=...,
                        assumptions=..., weights=...)
    nlp.add_initial_state_constraints()         # 等式: 初始位姿/速度
    nlp.add_junction_constraints()              # 等式: 相位连接
    nlp.add_dynamic_stability_constraints()     # GIAC (统一 L_dot) + 摩擦/推地
    nlp.add_kinematic_contact_constraints()     # 式 (30) 凸性 + 式 (31) 可达
    nlp.add_objectives()                        # V-B1..V-B8 (+ 式 29 鲁棒裕度)
    nlp.finalize()                              # backend.add_objective + minimize
    nlp.set_initial_guess(...)                  # 冷启动初值 / 或 set_warm_start
    report = nlp.solve(scenario_name="...")     # SolveReport (诊断 + 独立复算入口)

决策变量 vs 参数
----------------
决策变量 (见 :class:`IntegratedNLPLayout.describe`)::

    spline_coeffs      (P, 6, K)   机身样条系数 a0..a_{K-1} (m, rad 及其时间导数量纲)
    desired_footholds  (4, 3)       horizon 内触地腿的期望落脚点 [m]
    slack              (P,)        GIAC 鲁棒裕度 eps (式 29), use_giac_slack 时存在

参数 (不做自由变量)::

    base_pose0 / base_twist0      当前测量位姿/速度 (经初始等式约束使用, 不重复建变量)
    measured_stance_feet          已着地脚的世界系位置 [m]
    maps (h/h_s1/h_s2 + 梯度场)    高程图查询数据 (map_query.TerrainMaps)
    mass / inertia_B / mu         质量 [kg] / 机体系惯量 [kg m^2] / 摩擦系数
    hip_offsets / l_min / l_max / h_des   腿部几何参数 [m]
    ref_twist(phase, tau)         速度/姿态跟踪参考 (机体系线速度 [m/s] + 角速度 [rad/s])
    previous_footholds            上一轮解 [m]
    durations                     各相位时长 [s]

已着地脚**不是**自由变量: 它们是测量量, 直接以参数进入 GIAC/可达/凸性
表达式; horizon 内不触地的腿, 其 ``desired_footholds`` 分量用等式约束
钉到测量值 (避免悬挂自由变量, 见 ``add_initial_state_constraints``).

统一 L_dot (本次集成优先事项)
------------------------------
GIAC (17b)-(17f) 与 V-B8 使用**同一个** CasADi 表达式树:
:func:`trajectory_angular_momentum_rate` 由样条系数解析给出
``L_dot_B(tau) = d/dt (R_B I_B omega_B^B)`` (公式见
:mod:`v_b_objectives`), :meth:`IntegratedNLP.add_dynamic_stability_constraints`
**不接受**外部 ``l_dot_override`` (传入即抛 ``ValueError``), 从接口上排除
"独立 L_dot 与轨迹导出不一致" 的可能; 对应测试见
``test_nlp_integration.TestLDotUnification``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np

from myrobot_mjlab.task.DTC.tamols.assumptions_config import (
    AssumptionsConfig,
    DEFAULT_ASSUMPTIONS,
)
from myrobot_mjlab.task.DTC.tamols import friction_sign
from myrobot_mjlab.task.DTC.tamols.giac_constraints import (
    EQUALITY,
    INEQUALITY,
    ballistic_residuals,
    double_support_line_residual,
    double_support_moment_residual,
    giac_residual,
    single_support_residuals,
)
from myrobot_mjlab.task.DTC.tamols.map_query import TerrainMaps
from myrobot_mjlab.task.DTC.tamols.solver_backend import SolverBackend, make_backend
from myrobot_mjlab.task.DTC.tamols.v_a_constraints import (
    CCW_LEG_ORDER,
    GO2_HIP_JOINT_OFFSETS,
    GO2_STANCE_HIP_OFFSETS,
    LEG_ORDER,
    friction_push_records,
    leg_index,
    reachability_residuals,
    rotation_matrix_zyx,
    spline_value,
)
from myrobot_mjlab.task.DTC.tamols.v_b_objectives import (
    DEFAULT_EPS_MIN,
    DEFAULT_H_DES,
    GO2_INERTIA_B,
    GO2_MASS,
    ObjectiveRecord,
    ObjectiveSet,
    VBWeights,
    angular_momentum_rate_world,
    angular_momentum_world,
    base_pose_alignment_records,
    body_omega_rate_zyx,
    body_omega_zyx,
    edge_avoidance_records,
    foothold_ground_records,
    leg_collision_records,
    merge_objective_sets,
    midstance_from_contacts,
    nominal_kinematics_records,
    previous_solution_records,
    smoothness_records,
    tracking_records,
)

BASE_DIMS = 6


# ---------------------------------------------------------------------- #
# 轨迹导出的角动量及其变化率 (GIAC 与 V-B8 共用)
# ---------------------------------------------------------------------- #
def trajectory_angular_momentum(
    coeffs: Any, tau: float, inertia_B: Any = GO2_INERTIA_B
) -> tuple[Any, Any, Any]:
    """``L_B(tau) = R_B I_B omega_B^B`` (世界系), 由样条系数解析给出."""
    pose = spline_value(coeffs, tau, order=0)
    rate = spline_value(coeffs, tau, order=1)
    R_B = rotation_matrix_zyx(pose[3], pose[4], pose[5])
    omega = body_omega_zyx(pose[3:6], rate[3:6])
    return angular_momentum_world(R_B, inertia_B, omega)


def trajectory_angular_momentum_rate(
    coeffs: Any, tau: float, inertia_B: Any = GO2_INERTIA_B
) -> tuple[Any, Any, Any]:
    """``d/dt L_B(tau)`` (世界系), 与 V-B8 / GIAC 共用同一表达式树.

    ``L_dot = R_B (omega x (I_B omega) + I_B omega_dot)``, 其中
    ``omega`` / ``omega_dot`` 由 ZYX 欧拉角及其 1/2 阶导数经
    :func:`v_b_objectives.body_omega_zyx` /
    :func:`v_b_objectives.body_omega_rate_zyx` 映射 (二者**不相等**,
    映射矩阵非单位阵, 单测覆盖).
    """
    pose = spline_value(coeffs, tau, order=0)
    rate = spline_value(coeffs, tau, order=1)
    accel = spline_value(coeffs, tau, order=2)
    R_B = rotation_matrix_zyx(pose[3], pose[4], pose[5])
    omega = body_omega_zyx(pose[3:6], rate[3:6])
    omega_dot = body_omega_rate_zyx(pose[3:6], rate[3:6], accel[3:6])
    return angular_momentum_rate_world(R_B, inertia_B, omega, omega_dot)


# ---------------------------------------------------------------------- #
# 变量布局
# ---------------------------------------------------------------------- #
@dataclass(frozen=True)
class IntegratedNLPLayout:
    """集成 NLP 的决策变量布局 (参数不在此表, 见 :data:`PARAMETER_SPECS`).

    展平顺序 (决策向量 x)::

        [ spline_coeffs (P*6*K, 相位优先, 相位内 C 序 dim 外 k 内)
        | desired_footholds (12, 腿优先)
        | slack (P, 仅 use_giac_slack) ]

    与 ``base_spline`` / ``v_a_constraints.spline_value`` 的系数索引
    ``coeffs[dim, k]`` 一致 (见 ``test_nlp_integration`` 的索引测试).
    """

    num_phases: int
    num_coeffs: int = 5
    num_legs: int = 4
    use_slack: bool = True

    def __post_init__(self) -> None:
        if self.num_phases < 1:
            raise ValueError(f"num_phases 必须 >= 1, 实际为 {self.num_phases}")
        if self.num_coeffs < 1:
            raise ValueError(f"num_coeffs 必须 >= 1, 实际为 {self.num_coeffs}")

    @property
    def block_shapes(self) -> dict[str, tuple[int, ...]]:
        shapes = {
            "spline_coeffs": (self.num_phases, BASE_DIMS, self.num_coeffs),
            "desired_footholds": (self.num_legs, 3),
        }
        if self.use_slack:
            shapes["slack"] = (self.num_phases,)
        return shapes

    @property
    def block_sizes(self) -> dict[str, int]:
        return {k: int(np.prod(s)) for k, s in self.block_shapes.items()}

    @property
    def slices(self) -> dict[str, slice]:
        out: dict[str, slice] = {}
        start = 0
        for name, size in self.block_sizes.items():
            out[name] = slice(start, start + size)
            start += size
        return out

    @property
    def total_size(self) -> int:
        return sum(self.block_sizes.values())

    def pack(self, spline_coeffs: Sequence[np.ndarray], desired: np.ndarray,
             slack: np.ndarray | None = None) -> np.ndarray:
        parts = [np.asarray(c, dtype=float).reshape(-1) for c in spline_coeffs]
        parts.append(np.asarray(desired, dtype=float).reshape(-1))
        if self.use_slack:
            if slack is None:
                raise ValueError("use_slack=True 时 pack 需要 slack")
            parts.append(np.asarray(slack, dtype=float).reshape(-1))
        elif slack is not None:
            raise ValueError("use_slack=False 时不能 pack slack")
        return np.concatenate(parts)

    def unpack(self, x: np.ndarray) -> dict[str, np.ndarray]:
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size != self.total_size:
            raise ValueError(
                f"x 长度 {x.size} 与布局 total_size={self.total_size} 不一致"
            )
        out: dict[str, np.ndarray] = {}
        out["spline_coeffs"] = x[self.slices["spline_coeffs"]].reshape(
            self.num_phases, BASE_DIMS, self.num_coeffs
        )
        out["desired_footholds"] = x[self.slices["desired_footholds"]].reshape(
            self.num_legs, 3
        )
        if self.use_slack:
            out["slack"] = x[self.slices["slack"]]
        return out

    def describe(self) -> str:
        lines = [
            f"IntegratedNLPLayout: P={self.num_phases}, K={self.num_coeffs}, "
            f"L={self.num_legs}, slack={'yes' if self.use_slack else 'no'}, "
            f"total={self.total_size}",
            "decision variables:",
        ]
        units = {
            "spline_coeffs": "a_k: m/s^k (位置维), rad/s^k (欧拉维)",
            "desired_footholds": "m (世界系)",
            "slack": "与 GIAC 残差同量纲 (kg m^3/s^2), <= 0",
        }
        for name, shape in self.block_shapes.items():
            sl = self.slices[name]
            lines.append(
                f"  {name:<18s} shape={str(shape):<14s} x[{sl.start}:{sl.stop}) "
                f"units: {units[name]}"
            )
        lines.append("parameters (not decision variables):")
        for name, shape, unit, note in PARAMETER_SPECS:
            lines.append(f"  {name:<22s} shape={str(shape):<10s} units: {unit:<16s} {note}")
        return "\n".join(lines)


PARAMETER_SPECS: tuple[tuple[str, Any, str, str], ...] = (
    ("base_pose0", (6,), "m, rad", "测量位姿 [x,y,z,roll,pitch,yaw], 初始等式使用"),
    ("base_twist0", (6,), "m/s, rad/s", "测量速度 [线速度世界系, 欧拉角导数]"),
    ("measured_stance_feet", ("<=4", 3), "m", "已着地脚位置, 参数不进变量块"),
    ("maps h/hs1/hs2+grads", ("Ny", "Nx"), "m, m/m", "map_query 双线性查询数据"),
    ("mass", (), "kg", "整机质量"),
    ("inertia_B", (3, 3), "kg m^2", "机体系惯量 (名义位形)"),
    ("mu", (), "-", "摩擦系数 (assumptions)"),
    ("hip_offsets", (4, 3), "m", "r_i^B (assumptions.hip_offsets 选择)"),
    ("l_min/l_max", (), "m", "可达球半径"),
    ("h_des", (), "m", "名义腿伸展高度"),
    ("ref_twist", (3, 3), "m/s, rad/s", "机体系参考线速度/角速度 (回调)"),
    ("previous_footholds", (4, 3), "m", "上一轮解 (V-B6)"),
    ("durations", ("P",), "s", "各相位时长"),
)


# ---------------------------------------------------------------------- #
# 步态时序
# ---------------------------------------------------------------------- #
@dataclass(frozen=True)
class LegSchedule:
    """一条腿在 horizon 内的接触时序.

    - ``grounded_at_start=True``: t=0 着地, 位置为测量参数
      ``measured_position``; ``lift_phase`` (可选) 之前一直支撑;
      若之后 ``touchdown_phase`` 再次触地, 该次落点为决策变量.
    - ``grounded_at_start=False``: 必须给出 ``touchdown_phase``.
    """

    grounded_at_start: bool
    lift_phase: int | None = None
    touchdown_phase: int | None = None
    measured_position: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        if self.grounded_at_start and self.measured_position is None:
            raise ValueError("grounded_at_start 的腿必须给 measured_position")
        if not self.grounded_at_start and self.touchdown_phase is None:
            raise ValueError("horizon 内触地的腿必须给 touchdown_phase")
        if not self.grounded_at_start and self.lift_phase is not None:
            raise ValueError("未着地的腿在 horizon 内不再抬腿 (lift_phase 必为 None)")
        if (
            self.lift_phase is not None
            and self.touchdown_phase is not None
            and self.touchdown_phase <= self.lift_phase
        ):
            raise ValueError("touchdown_phase 必须晚于 lift_phase")

    def in_stance(self, phase: int) -> bool:
        if self.grounded_at_start and (
            self.lift_phase is None or phase < self.lift_phase
        ):
            return True
        return self.touchdown_phase is not None and phase >= self.touchdown_phase

    def foot_is_measured(self, phase: int) -> bool:
        return (
            self.grounded_at_start
            and (self.lift_phase is None or phase < self.lift_phase)
        )

    @property
    def touchdown_leg(self) -> bool:
        return self.touchdown_phase is not None


# ---------------------------------------------------------------------- #
# 参数容器
# ---------------------------------------------------------------------- #
@dataclass
class NLPParameters:
    """集成 NLP 的全部参数 (测量量/地图/模型常数/参考)."""

    base_pose0: np.ndarray
    base_twist0: np.ndarray
    measured_feet: Mapping[int | str, Sequence[float]]
    mass: float = GO2_MASS
    inertia_B: Any = GO2_INERTIA_B
    hip_offsets: np.ndarray | None = None
    l_min: float = 0.15
    l_max: float = 0.40
    h_des: float = DEFAULT_H_DES
    min_foot_distance_m: float = 0.1
    """V-B2 的脚间**距离**下限 [m]; 内部平方后作为 eps_min [m^2]."""
    ref_twist: Callable[[int, float], tuple[Sequence[float], Sequence[float]]] | None = None
    previous_footholds: np.ndarray | None = None
    solver_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.base_pose0 = np.asarray(self.base_pose0, dtype=float).reshape(6)
        self.base_twist0 = np.asarray(self.base_twist0, dtype=float).reshape(6)
        if not np.isfinite(self.base_pose0).all():
            raise ValueError(f"base_pose0 含非有限值: {self.base_pose0}")
        if not np.isfinite(self.base_twist0).all():
            raise ValueError(f"base_twist0 含非有限值: {self.base_twist0}")
        if self.min_foot_distance_m <= 0:
            raise ValueError(
                "min_foot_distance_m 必须为正 [m]; 平方距离阈值 eps_min [m^2] "
                f"由其导出, 实际为 {self.min_foot_distance_m}"
            )
        self.measured_feet = {
            leg_index(k): np.asarray(v, dtype=float).reshape(3)
            for k, v in self.measured_feet.items()
        }

    @property
    def eps_min_m2(self) -> float:
        """V-B2 的**平方**距离阈值 [m^2] (z 的定义见论文式 32)."""
        return self.min_foot_distance_m ** 2


# ---------------------------------------------------------------------- #
# 约束/目标记录
# ---------------------------------------------------------------------- #
@dataclass(frozen=True)
class ConstraintEntry:
    """一条已加入 backend 的约束及其诊断信息."""

    name: str
    category: str
    kind: str  # EQUALITY / INEQUALITY
    residual: Any
    ub: float = 0.0
    phase: int | None = None
    tau: float | None = None
    source: str = ""

    def violation(self, value: np.ndarray) -> float:
        arr = np.asarray(value, dtype=float).reshape(-1)
        if self.kind == EQUALITY:
            return float(np.max(np.abs(arr))) if arr.size else 0.0
        return float(np.max(arr - self.ub)) if arr.size else 0.0


@dataclass(frozen=True)
class ObjectiveLine:
    name: str
    category: str
    phase: int | None
    tau: float | None
    unweighted: float
    weight: float
    weighted: float
    fraction: float


@dataclass(frozen=True)
class ConstraintLine:
    name: str
    category: str
    kind: str
    value: float
    violation: float
    phase: int | None
    tau: float | None
    source: str


@dataclass
class SolveReport:
    """一次求解的完整诊断 (状态/迭代/目标分项/约束违反/尺度/假设)."""

    scenario: str
    status: str
    success: bool
    iterations: int | None
    solve_seconds: float
    layout_text: str
    assumptions_text: str
    objective_total: float
    objective_lines: tuple[ObjectiveLine, ...]
    constraint_lines: tuple[ConstraintLine, ...]
    max_violation_by_category: dict[str, float]
    variable_ranges: dict[str, tuple[float, float]]
    warnings: tuple[str, ...]
    solution: dict[str, np.ndarray]
    versions: dict[str, str]
    solution_source: str = "solver_solution"
    """``solution``/约束取值的来源: 成功时为求解器解; 失败时为
    "last_iterate_not_a_solution" (最后迭代点, 只供诊断, 不是解)."""
    expected_dec_dim: int = 0
    """该次求解的决策向量总维数 (= layout.total_size)."""

    def state_vector(self, layout: IntegratedNLPLayout) -> np.ndarray:
        """最优状态向量 x (决策变量的完整输出).

        拼接顺序与 :class:`IntegratedNLPLayout` 一致::

            x = [ spline_coeffs | desired_footholds | slack ]

        测量位姿/速度、已着地脚、地图、惯量等**参数不在 x 中**: 它们是
        求解的输入, 不是优化结果。由 x 可派生轨迹 (样条求值)、落脚点、
        基座速度/加速度、L_dot 等; ``layout.unpack(x)`` 可还原各块。
        """
        if layout.total_size != self.expected_dec_dim:
            raise ValueError(
                f"layout.total_size={layout.total_size} 与报告的决策维数 "
                f"{self.expected_dec_dim} 不一致"
            )
        blocks = [
            self.solution["spline_coeffs"].reshape(-1),
            self.solution["desired_footholds"].reshape(-1),
        ]
        if "slack" in self.solution:
            blocks.append(self.solution["slack"].reshape(-1))
        return np.concatenate(blocks)

    def format(self) -> str:
        lines = [
            f"=== SolveReport: {self.scenario} ===",
            f"status={self.status} success={self.success} "
            f"iterations={self.iterations} solve_time={self.solve_seconds:.3f}s",
            f"solution_source={self.solution_source}",
            f"objective total = {self.objective_total:.6g}",
            "objective breakdown (unweighted | weight | weighted | fraction):",
        ]
        by_cat: dict[str, list[ObjectiveLine]] = {}
        for row in self.objective_lines:
            by_cat.setdefault(row.category, []).append(row)
        for cat, rows in by_cat.items():
            cat_sum = sum(r.weighted for r in rows)
            lines.append(
                f"  [{cat}] n={len(rows)} sum_weighted={cat_sum:.6g} "
                f"fraction={cat_sum / self.objective_total if self.objective_total else 0.0:.3%}"
            )
            for r in rows[:3]:
                lines.append(
                    f"      {r.name:<44s} u={r.unweighted:.4g} w={r.weight:.4g} "
                    f"w*u={r.weighted:.4g} frac={r.fraction:.3%}"
                )
            if len(rows) > 3:
                lines.append(f"      ... ({len(rows) - 3} more)")
        lines.append("constraint violations by category (max):")
        for cat, viol in self.max_violation_by_category.items():
            lines.append(f"  {cat:<10s} {viol:.4g}")
        lines.append("variable ranges (min, max):")
        for name, (lo, hi) in self.variable_ranges.items():
            lines.append(f"  {name:<18s} [{lo:.4g}, {hi:.4g}]")
        if self.warnings:
            lines.append("warnings:")
            lines.extend(f"  ! {w}" for w in self.warnings)
        lines.append("versions: " + ", ".join(f"{k}={v}" for k, v in self.versions.items()))
        lines.append(self.assumptions_text)
        return "\n".join(lines)


# ---------------------------------------------------------------------- #
# 集成 NLP
# ---------------------------------------------------------------------- #
class IntegratedNLP:
    """把 V-A 约束 + GIAC + V-B 目标装配到同一个 CasADi NLP."""

    def __init__(
        self,
        *,
        durations: Sequence[float],
        schedules: Mapping[int | str, LegSchedule],
        maps: TerrainMaps,
        params: NLPParameters,
        assumptions: AssumptionsConfig | None = None,
        weights: VBWeights | None = None,
        backend: SolverBackend | None = None,
        sample_times: Sequence[Sequence[float]] | None = None,
    ):
        self.durations = [float(d) for d in durations]
        for d in self.durations:
            if not (math.isfinite(d) and d > 0):
                raise ValueError(f"durations 必须为有限正数, 实际为 {durations}")
        self.schedules = {leg_index(k): v for k, v in schedules.items()}
        if set(self.schedules) != set(range(len(LEG_ORDER))):
            raise ValueError("schedules 必须覆盖全部 4 条腿")
        for leg, sched in self.schedules.items():
            if sched.grounded_at_start and leg not in params.measured_feet:
                raise ValueError(f"腿 {leg} grounded_at_start 但 params 缺测量脚位置")
        self.maps = maps
        self.params = params
        self.assumptions = assumptions or DEFAULT_ASSUMPTIONS
        self.weights = weights or VBWeights()
        self.layout = IntegratedNLPLayout(
            num_phases=len(self.durations),
            num_coeffs=5,
            use_slack=self.assumptions.use_giac_slack,
        )
        self.backend = backend or make_backend(
            "casadi", solver_options=dict(params.solver_options)
        )
        self._sample_times_override = sample_times

        self.vars = self._create_variables()
        self.constraint_entries: list[ConstraintEntry] = []
        self.assembly_warnings: list[str] = []
        self.assembly_infeasible: list[str] = []
        self.objective_set: ObjectiveSet | None = None
        self._finalized = False
        self._struct_version = 0
        """结构版本: 每次 add_* / finalize 递增; extract 缓存键的一部分."""
        self._extract_cache: dict[tuple, Any] = {}
        """按 (jit, 结构版本, 地图身份, 时长, 采样数) 缓存的提取结果.

        符号图与 Jacobian 构建占单次求解 ~44% 耗时 (profile 实测), 而
        同一结构下它们完全相同: 结构不变时重复求解零提取成本.
        地图数据 (插值查表) 内嵌在图中, 更新地图需要重新提取."""
        self._ldot_cache: dict[tuple[int, float], tuple[Any, Any, Any]] = {}

        if params.hip_offsets is not None:
            self.hip_offsets = np.asarray(params.hip_offsets, dtype=float)
        elif self.assumptions.hip_offsets == "stance":
            self.hip_offsets = GO2_STANCE_HIP_OFFSETS
        else:
            self.hip_offsets = GO2_HIP_JOINT_OFFSETS

    # -- 变量 ------------------------------------------------------------ #
    def _create_variables(self) -> dict[str, Any]:
        b = self.backend
        spline = [
            b.variable((BASE_DIMS, self.layout.num_coeffs), f"spline_coeffs_{p}")
            for p in range(self.layout.num_phases)
        ]
        desired = b.variable((self.layout.num_legs, 3), "desired_footholds")
        slack = (
            b.variable((self.layout.num_phases,), "slack")
            if self.layout.use_slack
            else None
        )
        return {"spline_coeffs": spline, "desired_footholds": desired, "slack": slack}

    # -- 小工具 ---------------------------------------------------------- #
    def sample_times(self, phase: int) -> tuple[float, ...]:
        if self._sample_times_override is not None:
            return tuple(float(t) for t in self._sample_times_override[phase])
        step = self.durations[phase] / self.weights.num_samples
        return tuple(step * k for k in range(self.weights.num_samples))

    def stance_legs(self, phase: int) -> list[int]:
        return [leg for leg in range(4) if self.schedules[leg].in_stance(phase)]

    def touchdown_legs(self) -> list[int]:
        return [leg for leg in range(4) if self.schedules[leg].touchdown_leg]

    def foot_expr(self, phase: int, leg: int):
        """该相位该腿支撑点表达式: 测量参数或 desired 变量; 摆动返回 None."""
        sched = self.schedules[leg]
        if not sched.in_stance(phase):
            return None
        if sched.foot_is_measured(phase):
            return tuple(float(v) for v in self.params.measured_feet[leg])
        return tuple(self.vars["desired_footholds"][leg, k] for k in range(3))

    def _ordered_stance(self, phase: int, winding: Literal["ccw", "cw"]):
        """按绕行方向返回 (腿下标序列, 支撑点表达式序列)."""
        cycle = [leg_index(name) for name in CCW_LEG_ORDER]
        stance = self.stance_legs(phase)
        ordered = [leg for leg in cycle if leg in stance]
        if winding == "cw":
            ordered = list(reversed(ordered))
        return ordered, [self.foot_expr(phase, leg) for leg in ordered]

    def _add_entry(self, entry: ConstraintEntry) -> None:
        residual = entry.residual
        is_const = isinstance(residual, (int, float, np.integer, np.floating)) or (
            getattr(residual, "is_constant", lambda: False)()
        )
        if is_const:
            # 常数约束 (例如全部支撑脚都是测量参数时的凸性): CasADi 的
            # subject_to 不接受常数. 装配时直接数值判定: 满足则跳过
            # (冗余), 违反则记录为装配期违反 (问题不可行), 都不进求解器.
            value = float(np.asarray(residual, dtype=float).reshape(-1)[0])
            viol = entry.violation(np.asarray(value))
            if viol > 0.0:
                message = (
                    f"常数约束 {entry.name} 在装配时即违反 (残差 {value:.6g}, "
                    f"上界 {entry.ub}): 该问题与求解器无关地不可行"
                )
                self.assembly_warnings.append(message)
                self.assembly_infeasible.append(message)
            entry = ConstraintEntry(
                name=entry.name, category=entry.category, kind=entry.kind,
                residual=value, ub=entry.ub, phase=entry.phase, tau=entry.tau,
                source=entry.source + " [constant at assembly]",
            )
            self.constraint_entries.append(entry)
            return
        if entry.kind == EQUALITY:
            self.backend.add_equality(residual, 0.0, name=entry.name)
        else:
            self.backend.add_inequality(
                residual, -float("inf"), entry.ub, name=entry.name
            )
        self.constraint_entries.append(entry)

    def _suff(self, phase: int, tau: float | None) -> str:
        s = f"_phase{phase}"
        if tau is not None:
            s += f"_tau{tau:.6g}"
        return s

    def cached_extract(self, jit: bool = False):
        """按结构版本缓存的 GN-SQP 提取结果 (含 JIT 求值函数).

        缓存键 = (jit, 结构版本, 地图对象身份, 相位时长, 采样数).
        符号图/Jacobian 构建占单次求解 ~44% (profile 实测), 结构不变时
        重复求解零提取成本; 地图数据内嵌于插值查表, 更新地图后需重新
        提取 (结构版本不变时以地图身份区分). JIT 编译 (~30 s/图) 仅在
        缓存未命中时发生, 由调用方摊销.
        """
        key = (
            bool(jit), self._struct_version, id(self.maps),
            tuple(self.durations), self.weights.num_samples,
        )
        cached = self._extract_cache.get(key)
        if cached is None:
            from myrobot_mjlab.task.DTC.tamols.gn_sqp_solver import extract_nlp

            cached = extract_nlp(self, jit=jit)
            self._extract_cache[key] = cached
        return cached

    def l_dot_expression(self, phase: int, tau: float) -> tuple[Any, Any, Any]:
        """(phase, tau) 处轨迹导出的 L_dot_B 表达式 (缓存, 全 NLP 共用).

        GIAC 约束 (17b)-(17f) 与 V-B8 平滑目标通过本方法拿到**同一个
        表达式节点**, 从结构上排除 "约束用一套 L_dot、目标用另一套" 的
        不一致; 见 ``test_nlp_integration.TestIntegrationConsistency``.
        """
        key = (int(phase), round(float(tau), 12))
        if key not in self._ldot_cache:
            self._ldot_cache[key] = trajectory_angular_momentum_rate(
                self.vars["spline_coeffs"][phase], key[1],
                self.params.inertia_B,
            )
        return self._ldot_cache[key]

    def _accel_sign(self, p_ddot: Sequence[Any]) -> tuple[Any, Any, Any]:
        """按 assumptions.inertial_accel_sign 给出 a_B (不私自翻转)."""
        g_z = -9.81
        if self.assumptions.inertial_accel_sign == "paper_literal":
            return (0.0 - p_ddot[0], 0.0 - p_ddot[1], g_z - p_ddot[2])
        return (p_ddot[0] - 0.0, p_ddot[1] - 0.0, p_ddot[2] - g_z)

    # -- 等式: 初始状态 / 连接 ------------------------------------------- #
    
    def add_initial_state_constraints(self) -> list[ConstraintEntry]:
        """初始位姿/速度等式 + 不触地腿的 desired 钉到测量值."""
        self._struct_version += 1
        added: list[ConstraintEntry] = []
        coeffs0 = self.vars["spline_coeffs"][0]
        for dim in range(BASE_DIMS):
            entry = ConstraintEntry(
                name=f"initial_pose_{dim}",
                category="V-A-1",
                kind=EQUALITY,
                residual=coeffs0[dim, 0] - self.params.base_pose0[dim],
                phase=0,
                tau=0.0,
                source="measured base pose",
            )
            self._add_entry(entry)
            added.append(entry)
        rate0 = spline_value(coeffs0, 0.0, order=1)
        for dim in range(BASE_DIMS):
            entry = ConstraintEntry(
                name=f"initial_twist_{dim}",
                category="V-A-1",
                kind=EQUALITY,
                residual=rate0[dim] - self.params.base_twist0[dim],
                phase=0,
                tau=0.0,
                source="measured base twist",
            )
            self._add_entry(entry)
            added.append(entry)
        for leg in range(4):
            if not self.schedules[leg].touchdown_leg:
                # 该腿 horizon 内不触地: desired 分量不是自由量, 钉到测量值
                for k in range(3):
                    entry = ConstraintEntry(
                        name=f"pin_desired_leg{leg}_{k}",
                        category="layout",
                        kind=EQUALITY,
                        residual=self.vars["desired_footholds"][leg, k]
                        - float(self.params.measured_feet[leg][k]),
                        source="measured foot (no touchdown in horizon)",
                    )
                    self._add_entry(entry)
                    added.append(entry)
        return added

    
    def add_junction_constraints(self) -> list[ConstraintEntry]:
        self._struct_version += 1
        added: list[ConstraintEntry] = []
        for phase in range(self.layout.num_phases - 1):
            coeffs = self.vars["spline_coeffs"][phase]
            T = self.durations[phase]
            end = [
                self._horner(coeffs[dim, :], T) for dim in range(BASE_DIMS)
            ]
            nxt = self.vars["spline_coeffs"][phase + 1]
            for dim in range(BASE_DIMS):
                entry = ConstraintEntry(
                    name=f"junction_{phase}_{phase+1}_{dim}",
                    category="V-A-1",
                    kind=EQUALITY,
                    residual=end[dim] - nxt[dim, 0],
                    phase=phase,
                    tau=T,
                    source="spline junction C0",
                )
                self._add_entry(entry)
                added.append(entry)
            # 一阶导连接 (论文: 光滑到一阶导)
            end1 = [self._horner_deriv(coeffs[dim, :], T) for dim in range(BASE_DIMS)]
            nxt1 = [nxt[dim, 1] for dim in range(BASE_DIMS)]
            for dim in range(BASE_DIMS):
                entry = ConstraintEntry(
                    name=f"junction_d1_{phase}_{phase+1}_{dim}",
                    category="V-A-1",
                    kind=EQUALITY,
                    residual=end1[dim] - nxt1[dim],
                    phase=phase,
                    tau=T,
                    source="spline junction C1",
                )
                self._add_entry(entry)
                added.append(entry)
        return added

    def _horner(self, coeffs_row: Any, T: float) -> Any:
        n = self.layout.num_coeffs
        value = coeffs_row[n - 1]
        for k in range(n - 2, -1, -1):
            value = value * T + coeffs_row[k]
        return value

    def _horner_deriv(self, coeffs_row: Any, T: float) -> Any:
        # d/dtau sum a_k tau^k = sum k a_k tau^{k-1}
        n = self.layout.num_coeffs
        value = (n - 1) * coeffs_row[n - 1]
        for k in range(n - 2, 0, -1):
            value = value * T + k * coeffs_row[k]
        return value

    # -- GIAC + 摩擦/推地 (统一 L_dot) ------------------------------------ #
    
    def add_dynamic_stability_constraints(
        self, l_dot_override: Any = None
    ) -> list[ConstraintEntry]:
        """按接触相位分派 GIAC (17b)-(17f) + 摩擦/推地.

        Raises:
          ValueError: ``l_dot_override`` 非 None. L_dot 必须来自
            :func:`trajectory_angular_momentum_rate` (与 V-B8 同表达式),
            接口不允许外部独立 L_dot.
        """
        self._struct_version += 1
        if l_dot_override is not None:
            raise ValueError(
                "GIAC 的 L_dot 必须与机身轨迹导出值共用同一表达式 "
                "(trajectory_angular_momentum_rate); 不接受外部独立 L_dot. "
                "如确需对比, 请在测试中直接调用 giac_constraints 底层函数."
            )
        added: list[ConstraintEntry] = []
        winding = self.assumptions.resolved_winding()
        if self.layout.use_slack:
            # 式 (29) 要求每相位 eps <= 0. 缺了它, 正的 eps 会把 GIAC
            # 松弛成 "允许违反" (回归: test_slack_upper_bound_enforced).
            for phase in range(self.layout.num_phases):
                entry = ConstraintEntry(
                    name=f"29_eps_upper_phase{phase}", category="29",
                    kind=INEQUALITY, residual=self.vars["slack"][phase],
                    phase=phase, source="式 (29): eps <= 0 (robustness margin)",
                )
                self._add_entry(entry)
                added.append(entry)
        for phase in range(self.layout.num_phases):
            coeffs = self.vars["spline_coeffs"][phase]
            for tau in self.sample_times(phase):
                pose = spline_value(coeffs, tau, order=0)
                accel = spline_value(coeffs, tau, order=2)
                p_B = pose[0:3]
                p_ddot = accel[0:3]
                a_B = self._accel_sign(p_ddot)
                L_dot = self.l_dot_expression(phase, tau)
                ordered, feet = self._ordered_stance(phase, winding)
                n = len(feet)
                suff = self._suff(phase, tau)
                if n >= 3:
                    if self.assumptions.giac_pairs == "cyclic_edges":
                        pairs = [(i, (i + 1) % n) for i in range(n)]
                    else:
                        pairs = list(combinations(range(n), 2))
                    for i, j in pairs:
                        res = giac_residual(
                            self.params.mass, p_B, a_B, L_dot, feet[i], feet[j]
                        )
                        added.append(
                            self._add_giac_ineq(
                                f"17b_pair_{ordered[i]}_{ordered[j]}{suff}",
                                res, phase, tau,
                                f"GIAC N={n} winding={winding} pair=({ordered[i]},{ordered[j]})",
                            )
                        )
                elif n == 2:
                    res_eq = double_support_line_residual(
                        self.params.mass, p_B, a_B, L_dot, feet[0], feet[1]
                    )
                    entry = ConstraintEntry(
                        name=f"17c_line{self._suff(phase, tau)}",
                        category="17c", kind=EQUALITY, residual=res_eq,
                        phase=phase, tau=tau,
                        source=f"GIAC N=2 winding={winding}",
                    )
                    self._add_entry(entry)
                    added.append(entry)
                    res_ineq = double_support_moment_residual(
                        self.params.mass, p_B, a_B, L_dot, feet[0], feet[1]
                    )
                    added.append(
                        self._add_giac_ineq(
                            f"17d_moment{suff}", res_ineq, phase, tau,
                            f"GIAC N=2 winding={winding}",
                        )
                    )
                elif n == 1:
                    for axis, res in zip(
                        "xyz",
                        single_support_residuals(
                            self.params.mass, p_B, a_B, L_dot, feet[0]
                        ),
                    ):
                        entry = ConstraintEntry(
                            name=f"17e_{axis}{suff}", category="17e",
                            kind=EQUALITY, residual=res, phase=phase, tau=tau,
                            source="GIAC N=1",
                        )
                        self._add_entry(entry)
                        added.append(entry)
                else:
                    for axis, res in zip("xyz", ballistic_residuals(a_B, L_dot)[:3]):
                        entry = ConstraintEntry(
                            name=f"17f_aB_{axis}{suff}", category="17f",
                            kind=EQUALITY, residual=res, phase=phase, tau=tau,
                            source="GIAC N=0",
                        )
                        self._add_entry(entry)
                        added.append(entry)
                    for axis, res in zip("xyz", ballistic_residuals(a_B, L_dot)[3:]):
                        entry = ConstraintEntry(
                            name=f"17f_Ldot_{axis}{suff}", category="17f",
                            kind=EQUALITY, residual=res, phase=phase, tau=tau,
                            source="GIAC N=0",
                        )
                        self._add_entry(entry)
                        added.append(entry)

                if n > 0 and self.assumptions.friction_mode == "enabled":
                    if (
                        self.assumptions.friction_cone_form
                        == "force_balance_derived"
                    ):
                        if self.assumptions.friction_form != "squared":
                            raise ValueError(
                                "force_balance_derived 只提供平方形式 "
                                "(friction_sign.derived_cone_residuals), "
                                "请保持 friction_form='squared'"
                            )
                        fric_records = friction_sign.friction_records_derived(
                            a_B, p_ddot, self.assumptions.mu,
                            include_push=self.assumptions.include_push,
                            phase=phase,
                        )
                        source = (
                            "friction cone=force_balance_derived "
                            "(paper-internal evidence, UNVERIFIED externally)"
                        )
                    else:
                        fric_records = friction_push_records(
                            a_B, p_ddot, self.assumptions.mu,
                            form=self.assumptions.friction_form,
                            include_push=self.assumptions.include_push,
                            phase=phase,
                        ).records
                        source = (
                            f"friction cone=paper_literal(5) "
                            f"sign={self.assumptions.inertial_accel_sign}"
                        )
                    for record in fric_records:
                        entry = ConstraintEntry(
                            name=f"{record.name}{suff}",
                            category=record.category,
                            kind=record.kind,
                            residual=record.residual,
                            phase=phase, tau=tau,
                            source=source,
                        )
                        self._add_entry(entry)
                        added.append(entry)
        return added

    def _add_giac_ineq(
        self, name: str, residual: Any, phase: int, tau: float, source: str
    ) -> ConstraintEntry:
        """GIAC 不等式; use_giac_slack 时写成 ``res - eps_p <= 0`` (式 29)."""
        if self.layout.use_slack:
            expr = residual - self.vars["slack"][phase]
        else:
            expr = residual
        entry = ConstraintEntry(
            name=name, category="17b-17f", kind=INEQUALITY, residual=expr,
            phase=phase, tau=tau, source=source,
        )
        self._add_entry(entry)
        return entry

    # -- 凸性 (30) + 可达 (31) -------------------------------------------- #
    def _convexity_residuals(self, feet4: Sequence[Any], fourth: str):
        p1, p2, p3, p4 = feet4
        def sub(a, b):
            return (a[0] - b[0], a[1] - b[1], a[2] - b[2])
        def crossz(a, b):
            return a[0] * b[1] - a[1] * b[0]
        p12, p13, p14 = sub(p2, p1), sub(p3, p1), sub(p4, p1)
        p23, p24 = sub(p3, p2), sub(p4, p2)
        first3 = (
            crossz(p13, p12),
            crossz(p14, p13),
            crossz(p24, p23),
        )
        if fourth == "p21_engineering":
            p21 = sub(p1, p2)
            fourth_res = crossz(p21, p24)
        else:  # p20_literal: p0 取世界原点
            p20 = (0.0 - p2[0], 0.0 - p2[1], 0.0 - p2[2])
            fourth_res = crossz(p20, p24)
        return first3 + (fourth_res,)

    
    def add_kinematic_contact_constraints(self) -> list[ConstraintEntry]:
        self._struct_version += 1
        added: list[ConstraintEntry] = []
        names = ("30_convex_p13_p12", "30_convex_p14_p13",
                 "30_convex_p24_p23", "30_convex_fourth")
        for phase in range(self.layout.num_phases):
            stance = self.stance_legs(phase)
            if len(stance) == 4:
                ordered, feet = self._ordered_stance(phase, "ccw")
                residuals = self._convexity_residuals(
                    feet, self.assumptions.convexity_fourth_term
                )
                for name, res in zip(names, residuals):
                    entry = ConstraintEntry(
                        name=f"{name}_phase{phase}", category="30",
                        kind=INEQUALITY, residual=res, phase=phase,
                        source=f"convexity 4th={self.assumptions.convexity_fourth_term} "
                               f"order={tuple(ordered)}",
                    )
                    self._add_entry(entry)
                    added.append(entry)
            # 可达 (31): 支撑腿在采样时刻
            coeffs = self.vars["spline_coeffs"][phase]
            for tau in self.sample_times(phase):
                pose = spline_value(coeffs, tau, order=0)
                R_B = rotation_matrix_zyx(pose[3], pose[4], pose[5])
                for leg in stance:
                    foot = self.foot_expr(phase, leg)
                    lower, upper = reachability_residuals(
                        pose[0:3], R_B, foot, self.hip_offsets[leg],
                        self.params.l_min, self.params.l_max,
                    )
                    for tag, res in (("31_lower", lower), ("31_upper", upper)):
                        entry = ConstraintEntry(
                            name=f"{tag}_leg{leg}{self._suff(phase, tau)}",
                            category=tag, kind=INEQUALITY, residual=res,
                            phase=phase, tau=tau,
                            source=f"reachability l=[{self.params.l_min},{self.params.l_max}] m",
                        )
                        self._add_entry(entry)
                        added.append(entry)
        return added

    # -- 目标 -------------------------------------------------------------- #
    
    def add_objectives(self) -> ObjectiveSet:
        self._struct_version += 1
        p = self.params
        desired = self.vars["desired_footholds"]
        td = self.touchdown_legs()
        coeffs = self.vars["spline_coeffs"]
        per_phase_times = [self.sample_times(k) for k in range(self.layout.num_phases)]
        sets: list[ObjectiveSet] = []
        if td:
            sets.append(
                foothold_ground_records(self.maps, desired, legs=td, weights=self.weights)
            )
            sets.append(
                leg_collision_records(
                    desired, eps_min=p.eps_min_m2,
                    pairs=list(combinations(td, 2)), weights=self.weights,
                )
            )
            sets.append(
                edge_avoidance_records(self.maps, desired, legs=td, weights=self.weights)
            )
        contact_sets = [self.stance_legs(k) for k in range(self.layout.num_phases)]
        midstance = midstance_from_contacts(contact_sets, self.durations)
        midstance = {leg: v for leg, v in midstance.items() if leg in td}
        if midstance:
            sets.append(
                nominal_kinematics_records(
                    coeffs, desired, midstance,
                    r_hips=self.hip_offsets, h_des=p.h_des, weights=self.weights,
                )
            )
        sets.append(
            base_pose_alignment_records(
                coeffs, self.durations, self.maps,
                r_hips=self.hip_offsets, h_des=p.h_des,
                weights=self.weights, sample_times=per_phase_times,
            )
        )
        if p.previous_footholds is not None and td:
            sets.append(
                previous_solution_records(
                    desired, p.previous_footholds, legs=td, weights=self.weights
                )
            )
        if p.ref_twist is not None:
            sets.append(
                tracking_records(
                    coeffs, self.durations, p.ref_twist,
                    inertia_B=p.inertia_B, weights=self.weights,
                    sample_times=per_phase_times,
                )
            )
        sets.append(
            smoothness_records(
                coeffs, self.durations, inertia_B=p.inertia_B,
                weights=self.weights, sample_times=per_phase_times,
                l_dot_provider=self.l_dot_expression,
            )
        )
        if self.layout.use_slack:
            slack_records = []
            for phase in range(self.layout.num_phases):
                slack_records.append(
                    ObjectiveRecord(
                        name=f"29_slack_phase{phase}", category="29",
                        residual=self.vars["slack"][phase],
                        weight=self.weights.robustness_margin,
                        phase=phase,
                        note="robustness margin (式 29), unit = GIAC residual unit",
                    )
                )
            sets.append(ObjectiveSet(records=tuple(slack_records)))
        self.objective_set = merge_objective_sets(*sets)
        return self.objective_set

    
    def finalize(self) -> None:
        self._struct_version += 1
        if self.objective_set is None:
            raise RuntimeError("finalize() 前必须先 add_objectives()")
        for record in self.objective_set.records:
            self.backend.add_objective(record.weighted_residual(), name=record.name)
        self.backend.minimize()
        self._finalized = True

    @staticmethod
    def check_finite(solution: Mapping[str, np.ndarray]) -> list[str]:
        """NaN/Inf 检查: 返回警告字符串列表 (空列表表示全部有限)."""
        out: list[str] = []
        for name, arr in solution.items():
            arr = np.asarray(arr, dtype=float)
            if arr.size and not np.isfinite(arr).all():
                out.append(f"solution block '{name}' 含 NaN/Inf")
        return out

    # -- 初值 / 热启动 ------------------------------------------------------ #
    def set_initial_guess(
        self,
        spline_coeffs: Sequence[np.ndarray],
        desired_footholds: np.ndarray,
        slack: np.ndarray | None = None,
    ) -> None:
        coeffs = list(spline_coeffs)
        if len(coeffs) != self.layout.num_phases:
            raise ValueError("spline_coeffs 段数与相位不一致")
        for phase, c in enumerate(coeffs):
            self.backend.set_initial(self.vars["spline_coeffs"][phase], np.asarray(c))
        self.backend.set_initial(
            self.vars["desired_footholds"], np.asarray(desired_footholds, dtype=float)
        )
        if self.layout.use_slack:
            value = (
                np.zeros(self.layout.num_phases)
                if slack is None
                else np.asarray(slack, dtype=float)
            )
            self.backend.set_initial(self.vars["slack"], value)

    def set_warm_start(self, report: SolveReport) -> None:
        """热启动: 用上一次报告中的解作为初值."""
        sol = report.solution
        for phase, c in enumerate(sol["spline_coeffs"]):
            self.backend.set_initial(self.vars["spline_coeffs"][phase], c)
        self.backend.set_initial(
            self.vars["desired_footholds"], sol["desired_footholds"]
        )
        if self.layout.use_slack and "slack" in sol:
            self.backend.set_initial(self.vars["slack"], sol["slack"])

    # -- 场景辅助 ---------------------------------------------------------- #
    def add_constant_base_pin(self, phase: int = 0) -> list[ConstraintEntry]:

        """场景专用等式: 把指定相位样条钉为常速度位姿 (k>=2 系数全 0).

        用于合成场景中精确表达 "静止机身" (配合 base_twist0=0): 只有
        初始等式时, 样条仍可在相位内加速 (字面符号约定下求解器甚至会
        用它"自由落体"来满足摩擦锥), 钉住加速度及以上阶后 a_B 才真正
        等于 g - p_B_ddot = g. 一阶系数留给初始速度等式, 避免过约束.
        """
        self._struct_version += 1
        added: list[ConstraintEntry] = []
        coeffs = self.vars["spline_coeffs"][phase]
        for dim in range(BASE_DIMS):
            for k in range(2, self.layout.num_coeffs):
                entry = ConstraintEntry(
                    name=f"scenario_pin_p{phase}_{dim}_{k}",
                    category="scenario",
                    kind=EQUALITY,
                    residual=coeffs[dim, k],
                    phase=phase,
                    source="scenario: static base (constant pose spline)",
                )
                self._add_entry(entry)
                added.append(entry)
        return added

    # -- 求解与报告 ---------------------------------------------------------- #
    def solve(self, scenario: str = "") -> SolveReport:
        if not self._finalized:
            raise RuntimeError("solve() 前必须先 finalize()")
        if self.assembly_infeasible:
            # 常数约束在装配时即违反: 问题与求解器无关地不可行, 不求解.
            # 此分支只报告数值 (常数) 残差; 符号残差未求解没有数值意义.
            con_lines: list[ConstraintLine] = []
            by_cat: dict[str, float] = {}
            for entry in self.constraint_entries:
                if not isinstance(
                    entry.residual, (int, float, np.integer, np.floating)
                ):
                    continue
                val = np.asarray(entry.residual, dtype=float).reshape(-1)
                viol = entry.violation(val)
                con_lines.append(
                    ConstraintLine(
                        entry.name, entry.category, entry.kind,
                        float(val[0]) if val.size else 0.0, viol,
                        entry.phase, entry.tau, entry.source,
                    )
                )
                by_cat[entry.category] = max(by_cat.get(entry.category, 0.0), viol)
            import casadi as ca

            return SolveReport(
                scenario=scenario,
                status="Infeasible_At_Assembly",
                success=False,
                iterations=0,
                solve_seconds=0.0,
                layout_text=self.layout.describe(),
                assumptions_text=self.assumptions.describe(),
                objective_total=float("nan"),
                objective_lines=(),
                constraint_lines=tuple(con_lines),
                max_violation_by_category=by_cat,
                variable_ranges={},
                warnings=tuple(self.assembly_warnings),
                solution={},
                versions={"casadi": ca.__version__},
                expected_dec_dim=self.layout.total_size,
            )
        t0 = time.perf_counter()
        status, success, iterations = "solver_error", False, None
        try:
            self.backend.solve()
            diag = self.backend.solve_diagnostics()
            status = str(diag.get("status"))
            success = bool(diag.get("success"))
            iterations = diag.get("iterations")
        except Exception as exc:  # IPOPT 抛异常 (infeasible/unbounded 等)
            status = f"solver_error: {type(exc).__name__}: {exc}"
        dt = time.perf_counter() - t0

        evaluate = self.backend.value
        solution: dict[str, np.ndarray] = {}
        warnings: list[str] = list(self.assembly_warnings)
        try:
            solution["spline_coeffs"] = np.stack(
                [np.asarray(evaluate(c), dtype=float) for c in self.vars["spline_coeffs"]]
            )
            solution["desired_footholds"] = np.asarray(
                evaluate(self.vars["desired_footholds"]), dtype=float
            )
            if self.layout.use_slack:
                solution["slack"] = np.asarray(
                    evaluate(self.vars["slack"]), dtype=float
                ).reshape(-1)
            warnings.extend(self.check_finite(solution))
        except Exception as exc:
            warnings.append(f"读取解失败: {type(exc).__name__}: {exc}")

        # 目标分项
        obj_lines: list[ObjectiveLine] = []
        total = 0.0
        weighted_values: list[float] = []
        if self.objective_set is not None and solution:
            for record in self.objective_set.records:
                u = record.value(evaluate)
                wv = record.weight * u
                obj_lines.append(
                    ObjectiveLine(
                        record.name, record.category, record.phase, record.tau,
                        u, record.weight, wv, 0.0,
                    )
                )
                weighted_values.append(wv)
            total = float(sum(weighted_values))
            obj_lines = [
                ObjectiveLine(
                    r.name, r.category, r.phase, r.tau, r.unweighted, r.weight,
                    r.weighted, (r.weighted / total if total else 0.0),
                )
                for r in obj_lines
            ]
        if not np.isfinite(total):
            warnings.append("总目标非有限值")

        # 约束违反
        con_lines: list[ConstraintLine] = []
        by_cat: dict[str, float] = {}
        if solution:
            for entry in self.constraint_entries:
                try:
                    val = np.asarray(evaluate(entry.residual), dtype=float)
                except Exception:
                    continue
                viol = entry.violation(val)
                con_lines.append(
                    ConstraintLine(
                        entry.name, entry.category, entry.kind,
                        float(val.reshape(-1)[0]) if val.size else 0.0,
                        viol, entry.phase, entry.tau, entry.source,
                    )
                )
                by_cat[entry.category] = max(by_cat.get(entry.category, 0.0), viol)

        # 变量范围
        ranges = {
            name: (float(arr.min()), float(arr.max()))
            for name, arr in solution.items()
            if arr.size
        }

        # 地图覆盖检查 (解出的落脚点)
        if solution:
            for leg in self.touchdown_legs():
                foot = solution["desired_footholds"][leg]
                if not np.isfinite(foot).all():
                    continue
                if not self.maps.h.covers(foot[0], foot[1]):
                    warnings.append(
                        f"leg {leg} 解落脚点 ({foot[0]:.3f}, {foot[1]:.3f}) 超出地图范围"
                    )
                else:
                    validity = self.maps.validity_at(foot[0], foot[1])
                    if min(validity.values()) < 0.999:
                        warnings.append(
                            f"leg {leg} 解落脚点处地图有效性不足: {validity}"
                        )

        import casadi as ca  # 版本信息

        versions = {"casadi": ca.__version__, "ipopt": status if not success else "bundled"}
        return SolveReport(
            scenario=scenario,
            status=status,
            success=success,
            iterations=iterations,
            solve_seconds=dt,
            layout_text=self.layout.describe(),
            assumptions_text=self.assumptions.describe(),
            objective_total=total,
            objective_lines=tuple(obj_lines),
            constraint_lines=tuple(con_lines),
            max_violation_by_category=by_cat,
            variable_ranges=ranges,
            warnings=tuple(warnings),
            solution=solution,
            versions=versions,
            expected_dec_dim=self.layout.total_size,
            solution_source=(
                "solver_solution" if success else "last_iterate_not_a_solution"
            ),
        )
