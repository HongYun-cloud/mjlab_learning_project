"""最优状态向量 x 的下游导出: 基座轨迹 / 触地事件 / 触地关节角 (任务 A).

输入是 :class:`SolveReport` (成功求解) 与 :class:`IntegratedNLP`; 导出
内容全部由 x = [spline_coeffs | desired_footholds | slack] 派生:

- 基座轨迹: 逐相位样条求值 (位置/欧拉角/世界系线速度/线加速度/机体系
  角速度), 相位边界只取一次 (C0/C1 等式保证边界连续, 见
  :mod:`trajectory_audit` 的 junction 检查);
- 触地事件: 每腿触地全局时刻 = 其 ``touchdown_phase`` 的起始时刻;
  horizon 内不触地的腿为 NaN;
- 触地关节角: 在触地时刻用**该时刻的基座位姿**把世界系落脚点变换到
  髋关节系 (``GO2_HIP_OFFSETS_TRUNK``, 运动学真值; 注意与 NLP 可达约束
  用的名义支撑中心 ``hip_offsets`` 不同, 见 v_a_constraints 文档), 再调用
  工程现有解析 IK ``go2_leg_analytical_ik_np``; 可用
  ``go2_leg_forward_kinematics_np`` 闭合验证 (测试覆盖).

单位/坐标系: 位置 m (世界系, z 向上), 角度 rad (ZYX 欧拉角), 速度 m/s,
加速度 m/s^2, 角速度 rad/s (机体系); 关节角 rad, 顺序 [hip, thigh, calf]
逐腿拼接即 12 维 (FL, FR, RL, RR), 与工程 RL 观测/动作顺序一致.

关节限位 (go2_description.urdf, 左右对称)::

    hip   [-1.0472,  1.0472]
    thigh [-1.5708,  3.4907]
    calf  [-2.7227, -0.83776]

本模块不做 RL 观测接入 (观测构建在 ``config/observation`` 侧, 需要环境
状态); 导出的基座/关节量与其字段一一对应, 接入时直接取值即可.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from myrobot_mjlab.task.DTC.tamols.nlp_integration import IntegratedNLP, SolveReport
from myrobot_mjlab.task.DTC.tamols.v_a_constraints import (
    rotation_matrix_zyx,
    spline_value,
)
from myrobot_mjlab.task.DTC.tamols.v_b_objectives import body_omega_zyx
from myrobot_mjlab.task.Tamols.fetch.go2_kinematics import (
    GO2_HIP_OFFSETS_TRUNK,
    GO2_L_HIP_SIGNS,
    go2_leg_analytical_ik_np,
)

JOINT_LIMITS = {
    "hip": (-1.0472, 1.0472),
    "thigh": (-1.5708, 3.4907),
    "calf": (-2.7227, -0.83776),
}
"""Go2 关节限位 [rad] (go2_description.urdf, 左右对称)."""


def phase_start_times(nlp: IntegratedNLP) -> np.ndarray:
    starts = np.concatenate(([0.0], np.cumsum(nlp.durations)))
    return starts[:-1].copy()


@dataclass
class TrajectoryExport:
    t: np.ndarray
    base_pos_w: np.ndarray
    base_euler_zyx: np.ndarray
    base_lin_vel_w: np.ndarray
    base_lin_vel_b: np.ndarray
    base_lin_acc_w: np.ndarray
    base_ang_vel_b: np.ndarray
    footholds_w: np.ndarray
    touchdown_times: np.ndarray
    touchdown_joint_pos: np.ndarray
    contact_sets: list[tuple[int, ...]]

    def joint_positions_flat(self, leg: int) -> np.ndarray:
        return self.touchdown_joint_pos[leg]


def export_trajectory(
    nlp: IntegratedNLP,
    report: SolveReport,
    samples_per_phase: int = 25,
) -> TrajectoryExport:
    """由成功求解的报告导出基座轨迹与触地事件/关节角."""
    if not report.success or not report.solution:
        raise ValueError(
            f"export_trajectory 只接受成功求解的报告; status={report.status}, "
            f"solution_source={report.solution_source}"
        )
    coeffs = report.solution["spline_coeffs"]
    desired = report.solution["desired_footholds"]
    starts = phase_start_times(nlp)

    ts: list[float] = []
    pos: list[np.ndarray] = []
    euler: list[np.ndarray] = []
    vel_w: list[np.ndarray] = []
    vel_b: list[np.ndarray] = []
    acc_w: list[np.ndarray] = []
    omega_b: list[np.ndarray] = []
    for phase, T in enumerate(nlp.durations):
        grid = np.linspace(0.0, T, samples_per_phase)
        if phase > 0:
            grid = grid[1:]  # 边界点只取一次 (C0/C1 连续)
        for tau in grid:
            p0 = np.array([float(v) for v in spline_value(coeffs[phase], tau, 0)])
            p1 = np.array([float(v) for v in spline_value(coeffs[phase], tau, 1)])
            p2 = np.array([float(v) for v in spline_value(coeffs[phase], tau, 2)])
            R = np.array(rotation_matrix_zyx(p0[3], p0[4], p0[5]))
            ts.append(float(starts[phase] + tau))
            pos.append(p0[0:3])
            euler.append(p0[3:6])
            vel_w.append(p1[0:3])
            vel_b.append(R.T @ p1[0:3])
            acc_w.append(p2[0:3])
            omega_b.append(
                np.array([float(v) for v in body_omega_zyx(p0[3:6], p1[3:6])])
            )

    touchdown_times = np.full(4, np.nan)
    joints = np.full((4, 3), np.nan)
    for leg in range(4):
        sched = nlp.schedules[leg]
        if sched.touchdown_phase is None:
            continue
        phase = sched.touchdown_phase
        t_td = float(starts[phase])
        touchdown_times[leg] = t_td
        p0 = np.array([float(v) for v in spline_value(coeffs[phase], 0.0, 0)])
        R = np.array(rotation_matrix_zyx(p0[3], p0[4], p0[5]))
        p_foot_w = desired[leg]
        p_hip_frame = R.T @ (p_foot_w - p0[0:3]) - GO2_HIP_OFFSETS_TRUNK[leg]
        q = go2_leg_analytical_ik_np(
            p_hip_frame, is_left=GO2_L_HIP_SIGNS[leg] > 0
        )
        joints[leg] = q

    return TrajectoryExport(
        t=np.array(ts),
        base_pos_w=np.array(pos),
        base_euler_zyx=np.array(euler),
        base_lin_vel_w=np.array(vel_w),
        base_lin_vel_b=np.array(vel_b),
        base_lin_acc_w=np.array(acc_w),
        base_ang_vel_b=np.array(omega_b),
        footholds_w=desired.copy(),
        touchdown_times=touchdown_times,
        touchdown_joint_pos=joints,
        contact_sets=[tuple(nlp.stance_legs(p)) for p in range(nlp.layout.num_phases)],
    )


def check_export(
    nlp: IntegratedNLP, export: TrajectoryExport
) -> list[str]:
    """导出量的健全性检查 (限位/有限性/触地高度), 返回违反列表."""
    violations: list[str] = []
    for name, idx in (("hip", 0), ("thigh", 1), ("calf", 2)):
        lo, hi = JOINT_LIMITS[name]
        for leg in range(4):
            q = export.touchdown_joint_pos[leg, idx]
            if np.isfinite(q) and not (lo - 1e-9 <= q <= hi + 1e-9):
                violations.append(
                    f"leg {leg} {name} joint {q:.4f} outside [{lo}, {hi}]"
                )
    if not np.isfinite(export.base_pos_w).all():
        violations.append("base trajectory contains non-finite values")
    for leg in nlp.touchdown_legs():
        p = export.footholds_w[leg]
        h_at = float(nlp.maps.height_h(p[0], p[1]))
        if abs(h_at - p[2]) > 0.01:
            violations.append(
                f"leg {leg} touchdown foot z={p[2]:.4f} vs terrain {h_at:.4f}"
            )
    return violations
