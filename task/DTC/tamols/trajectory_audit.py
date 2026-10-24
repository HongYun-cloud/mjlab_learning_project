"""求解后轨迹的物理合理性审计 (eps<=0 修复后的复核工具).

输入是 :class:`~myrobot_mjlab.task.DTC.tamols.nlp_integration.SolveReport`
(成功求解) 与产生它的 :class:`IntegratedNLP`, 输出各物理指标的数值与
``violations`` 列表. 审计**不修改**任何约束/目标, 只读解.

指标定义
--------
- 摩擦利用率 ``util = ||F_xy|| / (μ F_z)``, 其中合力
  ``F = m (p̈ - g)`` (论文式 (1) 线性部分, 与符号约定无关):
  ``util <= 1`` 且 ``F_z > 0`` 是合力不滑移/可推地的必要条件
  (逐接触力分配无法由本模型验证, 见 friction_sign.py 文档).
- 贴地误差 ``|h(p_i) - p_i_z|`` (V-B1 软任务, 论文期望 < 5 mm 量级).
- 可达裕度: 触地腿在触地相位采样点的 ``d = ||p_B + R r_i - p_i||``
  相对 ``[l_min, l_max]`` 的裕度.
- GIAC 裕度: 每相位松弛 ``eps_p`` (式 29, 修复后必须 <= 0);
  ``-eps_p`` 越大鲁棒裕度越大.
- 连接连续性: 相位边界位姿/速度跳变 (C0/C1 等式约束的残差).
- 基座运动范围: z 区间、水平/竖直加速度峰值.

所有阈值都是**审计阈值** (物理合理性), 与求解器容差无关; 违反只进入
``violations`` 列表, 由调用方/测试决定如何处理.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from myrobot_mjlab.task.DTC.tamols.nlp_integration import IntegratedNLP, SolveReport
from myrobot_mjlab.task.DTC.tamols.v_a_constraints import (
    rotation_matrix_zyx,
    spline_value,
)
from myrobot_mjlab.task.DTC.tamols.v_b_objectives import body_omega_zyx


@dataclass
class AuditBounds:
    max_friction_utilization: float = 1.0 + 1e-6
    max_ground_error_m: float = 0.01
    base_z_range: tuple[float, float] = (0.15, 0.45)
    max_horizontal_acc: float = 8.0
    vertical_acc_range: tuple[float, float] = (-12.0, 12.0)
    max_junction_jump: float = 1e-8
    max_slack: float = 1e-9


@dataclass
class TrajectoryAudit:
    t: np.ndarray
    base_pos: np.ndarray
    base_euler: np.ndarray
    base_lin_acc_w: np.ndarray
    friction_utilization: dict[tuple[int, float], float]
    net_contact_force_z: dict[tuple[int, float], float]
    ground_error: dict[int, float]
    reach: dict[int, tuple[float, float, float]]  # leg: (d, l_min, l_max)
    slack: np.ndarray
    junction_jumps: dict[str, float]
    violations: list[str] = field(default_factory=list)

    def format(self) -> str:
        def fmt(values, spec, empty="n/a"):
            if not values:
                return empty
            return format(max(values), spec)

        lines = [
            f"TrajectoryAudit: T={self.t[-1]:.3f}s, samples={self.t.size}",
            f"  base z range      : [{self.base_pos[:,2].min():.3f}, {self.base_pos[:,2].max():.3f}] m",
            f"  |a_horiz| max     : {np.max(np.linalg.norm(self.base_lin_acc_w[:, :2], axis=1)):.3f} m/s^2",
            f"  a_z range         : [{self.base_lin_acc_w[:,2].min():.3f}, {self.base_lin_acc_w[:,2].max():.3f}] m/s^2",
            f"  friction util max : {fmt(list(self.friction_utilization.values()), '.4f')}",
            f"  F_z min           : {fmt(list(self.net_contact_force_z.values()), '.3f') + (' N' if self.net_contact_force_z else '')}",
            f"  ground error max  : {fmt(list(self.ground_error.values()), '.5f') + (' m' if self.ground_error else '')}",
            f"  slack (eps<=0)    : {np.round(self.slack, 5).tolist()}",
            f"  junction jumps    : {self.junction_jumps}",
        ]
        if self.violations:
            lines.append("  VIOLATIONS:")
            lines.extend(f"    ! {v}" for v in self.violations)
        else:
            lines.append("  violations: none")
        return "\n".join(lines)


def sample_times_global(
    nlp: IntegratedNLP, samples_per_phase: int | None = None
) -> list[tuple[int, float, float]]:
    """返回 ``[(phase, tau_local, t_global), ...]`` (每相位等距采样)."""
    n = samples_per_phase or nlp.weights.num_samples
    starts = np.concatenate(([0.0], np.cumsum(nlp.durations)))
    out = []
    for phase, T in enumerate(nlp.durations):
        for k in range(n):
            tau = T * k / n
            out.append((phase, tau, float(starts[phase] + tau)))
    return out


def audit_trajectory(
    nlp: IntegratedNLP,
    report: SolveReport,
    bounds: AuditBounds | None = None,
    samples_per_phase: int | None = None,
) -> TrajectoryAudit:
    """对成功求解的报告做物理合理性审计 (只读)."""
    if not report.success or not report.solution:
        raise ValueError(
            "audit_trajectory 只审计成功求解的报告; 当前 status="
            f"{report.status}, solution_source={report.solution_source}"
        )
    b = bounds or AuditBounds()
    coeffs = report.solution["spline_coeffs"]
    desired = report.solution["desired_footholds"]
    mass = nlp.params.mass
    mu = nlp.assumptions.mu

    samples = sample_times_global(nlp, samples_per_phase)
    t = np.array([s[2] for s in samples])
    pos = np.zeros((len(samples), 3))
    euler = np.zeros((len(samples), 3))
    acc = np.zeros((len(samples), 3))
    util: dict[tuple[int, float], float] = {}
    fz: dict[tuple[int, float], float] = {}
    for idx, (phase, tau, _) in enumerate(samples):
        p0 = np.array([float(v) for v in spline_value(coeffs[phase], tau, 0)])
        p2 = np.array([float(v) for v in spline_value(coeffs[phase], tau, 2)])
        pos[idx] = p0[0:3]
        euler[idx] = p0[3:6]
        acc[idx] = p2[0:3]
        F = mass * (p2[0:3] - np.array([0.0, 0.0, -9.81]))
        fz[(phase, tau)] = float(F[2])
        denom = mu * F[2]
        util[(phase, tau)] = (
            float(np.hypot(F[0], F[1]) / denom) if denom > 0 else float("inf")
        )

    ground_error: dict[int, float] = {}
    reach: dict[int, tuple[float, float, float]] = {}
    for leg in nlp.touchdown_legs():
        p = desired[leg]
        h_at = float(nlp.maps.height_h(p[0], p[1]))
        ground_error[leg] = abs(h_at - p[2])
        phase = nlp.schedules[leg].touchdown_phase
        p0 = np.array([float(v) for v in spline_value(coeffs[phase], 0.0, 0)])
        R = np.array(rotation_matrix_zyx(p0[3], p0[4], p0[5]))
        hip = p0[0:3] + R @ np.asarray(nlp.hip_offsets[leg])
        d = float(np.linalg.norm(hip - p))
        reach[leg] = (d, nlp.params.l_min, nlp.params.l_max)

    slack = report.solution.get("slack", np.zeros(0))
    jumps: dict[str, float] = {}
    for phase in range(nlp.layout.num_phases - 1):
        T = nlp.durations[phase]
        e0 = np.array([float(v) for v in spline_value(coeffs[phase], T, 0)])
        s0 = np.array([float(v) for v in spline_value(coeffs[phase + 1], 0.0, 0)])
        e1 = np.array([float(v) for v in spline_value(coeffs[phase], T, 1)])
        s1 = np.array([float(v) for v in spline_value(coeffs[phase + 1], 0.0, 1)])
        jumps[f"pose_{phase}_{phase+1}"] = float(np.max(np.abs(e0 - s0)))
        jumps[f"rate_{phase}_{phase+1}"] = float(np.max(np.abs(e1 - s1)))

    violations: list[str] = []
    for key, u in util.items():
        if u > b.max_friction_utilization:
            violations.append(f"friction utilization {u:.4f} > 1 at {key}")
    for key, value in fz.items():
        if value <= 0:
            violations.append(f"net contact force F_z={value:.3f} <= 0 at {key}")
    for leg, err in ground_error.items():
        if err > b.max_ground_error_m:
            violations.append(f"leg {leg} ground error {err:.4f} m > tol")
    for leg, (d, lo, hi) in reach.items():
        if d > hi or d < lo:
            violations.append(f"leg {leg} reach d={d:.3f} outside [{lo},{hi}]")
    zmin, zmax = pos[:, 2].min(), pos[:, 2].max()
    if not (b.base_z_range[0] <= zmin and zmax <= b.base_z_range[1]):
        violations.append(f"base z [{zmin:.3f},{zmax:.3f}] outside {b.base_z_range}")
    a_horiz = np.linalg.norm(acc[:, :2], axis=1).max()
    if a_horiz > b.max_horizontal_acc:
        violations.append(f"|a_horiz| max {a_horiz:.3f} > {b.max_horizontal_acc}")
    if not (b.vertical_acc_range[0] <= acc[:, 2].min()
            and acc[:, 2].max() <= b.vertical_acc_range[1]):
        violations.append(f"a_z outside {b.vertical_acc_range}")
    for name, jump in jumps.items():
        if jump > b.max_junction_jump:
            violations.append(f"junction {name} jump {jump:.3e}")
    if slack.size and slack.max() > b.max_slack:
        violations.append(f"slack max {slack.max():.4g} > 0 (式 29 违反)")

    return TrajectoryAudit(
        t=t, base_pos=pos, base_euler=euler, base_lin_acc_w=acc,
        friction_utilization=util, net_contact_force_z=fz,
        ground_error=ground_error, reach=reach,
        slack=slack, junction_jumps=jumps, violations=violations,
    )


def compare_solutions(
    report_a: SolveReport, report_b: SolveReport
) -> dict[str, float]:
    """两个成功解的差异 (落脚点最大位移 / 基座轨迹最大偏差 / slack 差)."""
    da = float(
        np.max(
            np.linalg.norm(
                report_a.solution["desired_footholds"]
                - report_b.solution["desired_footholds"],
                axis=1,
            )
        )
    )
    ca = report_a.solution["spline_coeffs"]
    cb = report_b.solution["spline_coeffs"]
    db = float(np.max(np.abs(ca - cb)))
    ds = float(
        np.max(
            np.abs(
                report_a.solution.get("slack", np.zeros(1))
                - report_b.solution.get("slack", np.zeros(1))
            )
        )
    )
    return {"max_foothold_delta_m": da, "max_coeff_delta": db, "max_slack_delta": ds}
