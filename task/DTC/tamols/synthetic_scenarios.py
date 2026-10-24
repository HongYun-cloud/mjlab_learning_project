"""合成场景构造 + 独立 NumPy 残差复算 + 可复现 CLI.

运行 (可复现命令)::

    cd /home/hong/myrobot_mjlab/src && \
    /home/hong/myrobot_mjlab/.venv/bin/python -m \
        myrobot_mjlab.task.DTC.tamols.synthetic_scenarios

五个场景 (固定构造, 无随机过程; 初值由解析公式给出, 种子仅用于
不需要随机的确定性构造)::

    S1 static_flat        平地 / 静止机身 / 四足静态支撑
                          (a) 字面符号: 预期摩擦锥不可行 -> 报告, 不宣称求解器 bug
                          (b) contact_normal_explicit: 预期数值可行
    S2 crawl_flat         平地低速前进 (crawl 三相位, 两个未来落脚点)
    S3 crawl_step         简单台阶 (前腿登上 +0.15 m 平台)
    S4 two_stones         两个候选凸起平台的落脚选择 (局部最优, 报告选择)
    S5 infeasible_reach   故意冲突: 初始位姿等式 + l_max=0.06 -> 解析证明无解

每个场景输出: IPOPT 状态/迭代数/目标分项/约束最大违反/变量范围/
待核实假设清单, 以及**独立 NumPy 代码**复算的关键约束残差对照.

"不可行" 的区分 (S5): 报告同时给出 (i) IPOPT 状态, (ii) 解析论证
(在 tau=0 处基座位姿被等式固定、脚位置为测量参数, 故可达残差为正常数
且必然违反), 因此结论是 "问题确实无解", 而不是 "求解器没收敛".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from myrobot_mjlab.task.DTC.tamols.assumptions_config import AssumptionsConfig
from myrobot_mjlab.task.DTC.tamols.map_query import TerrainMaps
from myrobot_mjlab.task.DTC.tamols.nlp_integration import (
    IntegratedNLP,
    LegSchedule,
    NLPParameters,
    SolveReport,
    trajectory_angular_momentum_rate,
)
from myrobot_mjlab.task.DTC.tamols.v_b_objectives import VBWeights

SEED = 0  # 场景构造完全确定; 保留常量以备将来引入随机扰动
SOLVER_OPTIONS = {
    "print_time": False,
    # max_wall_time: 求解超时保护; 超时按求解失败处理 (report.status 记录)
    "ipopt": {"print_level": 0, "max_iter": 300, "tol": 1e-6, "max_wall_time": 60.0},
}

GO2_FEET_STANCE = np.array(
    [
        [0.1934, 0.145, 0.0],
        [0.1934, -0.145, 0.0],
        [-0.1934, 0.145, 0.0],
        [-0.1934, -0.145, 0.0],
    ]
)


# ---------------------------------------------------------------------- #
# 独立 NumPy 参考实现 (不复用被测模块公式)
# ---------------------------------------------------------------------- #
def ref_rot_zyx(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def ref_omega(euler, rates):
    phi, theta, _ = euler
    pd, td, yd = rates
    return np.array(
        [
            pd - yd * math.sin(theta),
            td * math.cos(phi) + yd * math.sin(phi) * math.cos(theta),
            -td * math.sin(phi) + yd * math.cos(phi) * math.cos(theta),
        ]
    )


def ref_spline(coeffs_phase, tau, order=0):
    """pose^(order)(tau) = sum_k k!/(k-order)! a_k tau^(k-order)."""
    out = np.zeros(6)
    for dim in range(6):
        acc = 0.0
        for k in range(order, coeffs_phase.shape[1]):
            acc += math.factorial(k) / math.factorial(k - order) * coeffs_phase[dim, k] * tau ** (k - order)
        out[dim] = acc
    return out


def ref_L_dot_fd(coeffs_phase, I, tau, h=1e-5):
    """独立 FD: L(t) = R(φ(t)) I ω(φ, φ̇) 的中心差分."""
    def L(t):
        pose = ref_spline(coeffs_phase, t, 0)
        rate = ref_spline(coeffs_phase, t, 1)
        R = ref_rot_zyx(pose[3], pose[4], pose[5])
        return R @ np.asarray(I) @ ref_omega(pose[3:6], rate[3:6])
    return (L(tau + h) - L(tau - h)) / (2 * h)


def ref_giac(mass, p_B, a_B, L_dot, p_i, p_j):
    p_ij = np.asarray(p_j) - np.asarray(p_i)
    return float(
        mass * np.dot(p_ij, np.cross(np.asarray(p_B) - np.asarray(p_i), a_B))
        - np.dot(p_ij, L_dot)
    )


def ref_reach(p_B, R, r, p, l_min, l_max):
    d2 = float(np.sum((np.asarray(p_B) + R @ np.asarray(r) - np.asarray(p)) ** 2))
    return l_min ** 2 - d2, d2 - l_max ** 2


def bilinear_manual(grid, fx, fy):
    ny, nx = grid.shape
    fx = min(max(fx, 0.0), nx - 1.0)
    fy = min(max(fy, 0.0), ny - 1.0)
    j0 = min(int(math.floor(fx)), nx - 2)
    i0 = min(int(math.floor(fy)), ny - 2)
    tx, ty = fx - j0, fy - i0
    return float(
        (1 - ty) * (1 - tx) * grid[i0, j0]
        + (1 - ty) * tx * grid[i0, j0 + 1]
        + ty * (1 - tx) * grid[i0 + 1, j0]
        + ty * tx * grid[i0 + 1, j0 + 1]
    )


# ---------------------------------------------------------------------- #
# 地图构造
# ---------------------------------------------------------------------- #
def _grid(nx, ny, dx, origin):
    """世界系坐标网格: X[i,j] = origin[0] + j*dx, Y[i,j] = origin[1] + i*dy."""
    X, Y = np.meshgrid(
        origin[0] + np.arange(nx) * dx, origin[1] + np.arange(ny) * dx
    )
    return X, Y


def flat_maps(dx=0.05, n=41, origin=(-1.0, -1.0), h0=0.0):
    X, Y = _grid(n, n, dx, origin)
    h = np.full_like(X, h0)
    hs2 = np.full_like(X, h0)
    return TerrainMaps(h, h.copy(), hs2, dx=dx, dy=dx, origin=origin)


def step_maps(dx=0.05, n=61, origin=(-1.2, -1.2), x_step=0.30, h_high=0.15):
    X, Y = _grid(n, n, dx, origin)
    h = np.where(X > x_step, h_high, 0.0)
    # hs1: 3x3 盒滤波 (仅用于梯度第二项)
    hs1 = _box(h, 1)
    hs2 = _virtual_floor(h)
    return TerrainMaps(h, hs1, hs2, dx=dx, dy=dx, origin=origin)


def stones_maps(dx=0.05, n=81, origin=(-1.0, -0.8), h_pit=-0.25, r=0.10):
    """论文 stepping-stones 式地形: 后场平台 + 窄沟 (-0.25) + 三个石台 (h=0).

    石台 A(0.30,0.28) / B(0.30,0.06) 是 FL 的两个候选平台, C(0.30,-0.20)
    供 FR. 沟宽 0.02-0.10 m: 论文式 (21)-(23) 的掩码膨胀能架桥 (h_s2
    保持平台高度, 基座不下沉); 落在沟底则 |dz| = 0.53 > l_max = 0.40
    违反可达球, 因此落脚选择被限制在石台之间, 构成真正的候选平台选择
    问题 (A/B 都可达, 选择由初值/代价决定, 属局部最优).
    """
    X, Y = _grid(n, n, dx, origin)
    h = np.full_like(X, h_pit)
    h = np.where(X < 0.10, 0.0, h)  # 后场平台 (测量脚所在)
    for cx, cy in ((0.30, 0.28), (0.30, 0.06), (0.30, -0.20)):
        h = np.where((X - cx) ** 2 + (Y - cy) ** 2 <= r**2, 0.0, h)
    hs1 = _box(h, 1)
    hs2 = _virtual_floor(h)
    return TerrainMaps(h, hs1, hs2, dx=dx, dy=dx, origin=origin)


def _box(grid, radius):
    if radius <= 0:
        return grid.copy()
    out = np.zeros_like(grid)
    cnt = 0
    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            out += np.roll(np.roll(grid, di, axis=0), dj, axis=1)
            cnt += 1
    return out / cnt


def _max_filter(grid, radius):
    out = grid.copy()
    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            out = np.maximum(out, np.roll(np.roll(grid, di, axis=0), dj, axis=1))
    return out


def _virtual_floor(h, delta_thr=0.05, dilate_r=3, smooth_r=4):
    """论文式 (21)-(23) 的简化版虚拟地板 h_s2.

    候选格 (|h - 局部均值| > thr) 掩码膨胀后, 被覆盖格的高度替换为邻域
    最大值 (间隙被抬到周围地表 / 落脚台加宽), 再做大核平滑. 效果:
    h_s2 在坑/间隙上方"架桥", 使机身对齐目标 (V-B4) 不会把基座拉进坑里
    (论文 Fig.6: "hs2 spawns above the stepping stones").
    """
    med = _box(h, 2)
    mask = np.abs(h - med) > delta_thr
    covered = _max_filter(mask.astype(float), dilate_r) > 0
    h_max = _max_filter(h, dilate_r)
    h_d = np.where(covered, h_max, h)
    return _box(h_d, smooth_r)


# ---------------------------------------------------------------------- #
# 场景定义
# ---------------------------------------------------------------------- #
@dataclass
class ScenarioResult:
    name: str
    report: SolveReport
    independent_checks: dict[str, tuple[float, float]]
    notes: str = ""


def _crawl_schedules():
    """crawl: phase0 {FR,RL,RR}, phase1 {FL,RL,RR}+FL 触地, phase2 全支撑."""
    return {
        0: LegSchedule(True, lift_phase=0, touchdown_phase=1,
                       measured_position=tuple(GO2_FEET_STANCE[0])),
        1: LegSchedule(True, lift_phase=1, touchdown_phase=2,
                       measured_position=tuple(GO2_FEET_STANCE[1])),
        2: LegSchedule(True, measured_position=tuple(GO2_FEET_STANCE[2])),
        3: LegSchedule(True, measured_position=tuple(GO2_FEET_STANCE[3])),
    }


def _crawl_guess(nlp: IntegratedNLP, v_cmd: np.ndarray):
    """解析初值: 基座匀速直线, 姿态水平; 落脚点 = 名义脚位 + v * 触地时刻."""
    p0 = nlp.params.base_pose0[0:3]
    v_cmd = np.array([v_cmd[0], v_cmd[1], 0.0], dtype=float)
    # 先构造落脚点初值 (名义脚位 + v * 触地时刻), 供 GIAC 友好加速度使用
    desired = np.array(GO2_FEET_STANCE, dtype=float).copy()
    desired_xy = {}
    for leg, sched in nlp.schedules.items():
        if sched.touchdown_phase is not None:
            t_td = sum(nlp.durations[: sched.touchdown_phase])
            desired[leg] = GO2_FEET_STANCE[leg] + np.array(
                [v_cmd[0] * t_td, v_cmd[1] * t_td, 0.0]
            )
            desired[leg, 2] = 0.0
        desired_xy[leg] = desired[leg][:2].copy()
    coeffs = []
    t0 = 0.0
    for phase, T in enumerate(nlp.durations):
        c = np.zeros((6, 5))
        c[0:3, 0] = p0 + v_cmd * t0
        c[0:3, 1] = v_cmd
        # GIAC 友好初值: 水平加速度把 "有效重力" g - p_ddot 指向支撑多边形
        # 质心, 否则 N=3 相位基座在三角形外, 零加速度初值必然违反 GIAC,
        # IPOPT 的 restoration 阶段会把解甩到无关的局部最优.
        stance = nlp.stance_legs(phase)
        feet_xy = []
        for leg in stance:
            sched = nlp.schedules[leg]
            if sched.foot_is_measured(phase):
                feet_xy.append(np.asarray(nlp.params.measured_feet[leg])[:2])
            else:
                feet_xy.append(desired_xy[leg])
        if feet_xy:
            centroid = np.mean(feet_xy, axis=0)
            p_B_mid = (p0 + v_cmd * (t0 + 0.5 * T))[:2]
            acc_h = 32.7 * (p_B_mid - centroid)  # ~ g / 0.3 m 的倾斜增益
            acc_h = np.clip(acc_h, -6.0, 6.0)
            c[0:3, 2] = np.array([acc_h[0], acc_h[1], 0.0]) / 2.0
        coeffs.append(c)
        t0 += T
    return coeffs, desired


def build_scenario(name: str, assumptions: AssumptionsConfig):
    """返回 (IntegratedNLP, 初值(coeffs, desired), 场景说明)."""
    if name == "static_flat":
        maps = flat_maps()
        schedules = {
            leg: LegSchedule(True, measured_position=tuple(GO2_FEET_STANCE[leg]))
            for leg in range(4)
        }
        params = NLPParameters(
            base_pose0=[0, 0, 0.30, 0, 0, 0],
            base_twist0=np.zeros(6),
            measured_feet={leg: GO2_FEET_STANCE[leg] for leg in range(4)},
            ref_twist=lambda p, t: (np.zeros(3), np.zeros(3)),
            solver_options=SOLVER_OPTIONS,
        )
        nlp = IntegratedNLP(
            durations=[0.4], schedules=schedules, maps=maps,
            params=params, assumptions=assumptions,
        )
        coeffs = [np.zeros((6, 5))]
        coeffs[0][0:3, 0] = [0, 0, 0.30]
        desired = GO2_FEET_STANCE.copy()
        return nlp, (coeffs, desired), "flat, static base, 4-foot stance, P=1"

    if name == "crawl_flat":
        maps = flat_maps()
        params = NLPParameters(
            base_pose0=[0, 0, 0.30, 0, 0, 0],
            base_twist0=[0.25, 0, 0, 0, 0, 0],
            measured_feet={leg: GO2_FEET_STANCE[leg] for leg in range(4)},
            ref_twist=lambda p, t: (np.array([0.25, 0, 0.0]), np.zeros(3)),
            solver_options=SOLVER_OPTIONS,
        )
        nlp = IntegratedNLP(
            durations=[0.3, 0.3, 0.3], schedules=_crawl_schedules(), maps=maps,
            params=params, assumptions=assumptions,
        )
        return nlp, _crawl_guess(nlp, np.array([0.25, 0.0])), \
            "flat, vx=0.25 m/s crawl, P=3, touchdowns FL@1 FR@2"

    if name == "crawl_step":
        maps = step_maps()
        feet = GO2_FEET_STANCE.copy()
        params = NLPParameters(
            base_pose0=[0, 0, 0.32, 0, 0, 0],
            base_twist0=[0.2, 0, 0, 0, 0, 0],
            measured_feet={leg: feet[leg] for leg in range(4)},
            ref_twist=lambda p, t: (np.array([0.2, 0, 0.0]), np.zeros(3)),
            solver_options=SOLVER_OPTIONS,
        )
        nlp = IntegratedNLP(
            durations=[0.3, 0.3, 0.3], schedules=_crawl_schedules(), maps=maps,
            params=params, assumptions=assumptions,
        )
        coeffs, desired = _crawl_guess(nlp, np.array([0.2, 0.0]))
        # 前腿初值放到上平台
        desired[0, 2] = 0.15
        desired[1, 2] = 0.15
        return nlp, (coeffs, desired), \
            "step +0.15 m at x=0.30, front legs touchdown on upper tread"

    if name == "two_stones":
        maps = stones_maps()
        params = NLPParameters(
            base_pose0=[0, 0, 0.30, 0, 0, 0],
            base_twist0=[0.2, 0, 0, 0, 0, 0],
            measured_feet={leg: GO2_FEET_STANCE[leg] for leg in range(4)},
            ref_twist=lambda p, t: (np.array([0.2, 0, 0.0]), np.zeros(3)),
            solver_options=SOLVER_OPTIONS,
        )
        nlp = IntegratedNLP(
            durations=[0.3, 0.3, 0.3], schedules=_crawl_schedules(), maps=maps,
            params=params, assumptions=assumptions,
        )
        coeffs, desired = _crawl_guess(nlp, np.array([0.2, 0.0]))
        # 初值放在石台 A 顶: 检验局部落脚选择 (预期停留在 A, 局部最优)
        desired[0] = [0.30, 0.28, 0.0]
        desired[1] = [0.30, -0.20, 0.0]
        return nlp, (coeffs, desired), \
            "narrow gaps + stones A(0.30,0.28)/B(0.30,0.06)/C(0.30,-0.20); FL guess on A"

    if name == "infeasible_reach":
        maps = flat_maps()
        schedules = {
            leg: LegSchedule(True, measured_position=tuple(GO2_FEET_STANCE[leg]))
            for leg in range(4)
        }
        params = NLPParameters(
            base_pose0=[0, 0, 0.30, 0, 0, 0],
            base_twist0=np.zeros(6),
            measured_feet={leg: GO2_FEET_STANCE[leg] for leg in range(4)},
            l_min=0.02,
            l_max=0.06,
            ref_twist=lambda p, t: (np.zeros(3), np.zeros(3)),
            solver_options=SOLVER_OPTIONS,
        )
        nlp = IntegratedNLP(
            durations=[0.4], schedules=schedules, maps=maps,
            params=params, assumptions=assumptions,
        )
        coeffs = [np.zeros((6, 5))]
        coeffs[0][0:3, 0] = [0, 0, 0.30]
        return nlp, (coeffs, GO2_FEET_STANCE.copy()), \
            "conflict: initial pose pin z=0.30 + l_max=0.06 m -> provably infeasible"

    raise ValueError(f"未知场景 {name!r}")


# ---------------------------------------------------------------------- #
# 独立残差复算
# ---------------------------------------------------------------------- #
def independent_residual_check(
    nlp: IntegratedNLP, report: SolveReport, tol: float = 1e-6
) -> dict[str, tuple[float, float]]:
    """用独立 NumPy 公式复算关键约束/目标残差, 返回 {名称: (报告值, 复算值)}.

    覆盖: 初始位姿/速度等式, 连接等式, GIAC (17b, L_dot 用独立 FD),
    摩擦 (17a), 可达 (31), 凸性 (30), V-B1 贴地, V-B8 平滑.
    """
    if not report.solution:
        return {}
    coeffs = report.solution["spline_coeffs"]
    desired = report.solution["desired_footholds"]
    by_name = {line.name: line.value for line in report.constraint_lines}
    obj_by_name = {line.name: line.unweighted for line in report.objective_lines}
    out: dict[str, tuple[float, float]] = {}
    I = np.asarray(nlp.params.inertia_B)
    mass = nlp.params.mass
    winding = nlp.assumptions.resolved_winding()

    def foot_at(phase, leg):
        sched = nlp.schedules[leg]
        if sched.foot_is_measured(phase):
            return np.asarray(nlp.params.measured_feet[leg], dtype=float)
        return desired[leg]

    for phase in range(nlp.layout.num_phases):
        c = coeffs[phase]
        for tau in nlp.sample_times(phase):
            pose = ref_spline(c, tau, 0)
            accel = ref_spline(c, tau, 2)
            p_B, p_ddot = pose[0:3], accel[0:3]
            if nlp.assumptions.inertial_accel_sign == "paper_literal":
                a_B = np.array([0, 0, -9.81]) - p_ddot
            else:
                a_B = p_ddot - np.array([0, 0, -9.81])
            L_dot = ref_L_dot_fd(c, I, tau)
            stance = [leg for leg in range(4) if nlp.schedules[leg].in_stance(phase)]
            cycle = [0, 2, 3, 1]
            ordered = [leg for leg in cycle if leg in stance]
            if winding == "cw":
                ordered = list(reversed(ordered))
            feet = [foot_at(phase, leg) for leg in ordered]
            n = len(feet)
            if n >= 3:
                pairs = [(i, (i + 1) % n) for i in range(n)] if \
                    nlp.assumptions.giac_pairs == "cyclic_edges" else \
                    [(i, j) for i in range(n) for j in range(i + 1, n)]
                for i, j in pairs:
                    name = f"17b_pair_{ordered[i]}_{ordered[j]}_phase{phase}_tau{tau:.6g}"
                    if name in by_name:
                        out[name] = (
                            by_name[name],
                            ref_giac(mass, p_B, a_B, L_dot, feet[i], feet[j])
                            - (report.solution.get("slack", np.zeros(1))[phase]
                               if nlp.layout.use_slack else 0.0),
                        )
            # 摩擦 (norm/squared 的 squared 形式独立复算)
            if nlp.assumptions.friction_mode == "enabled" and n > 0:
                name = f"17a_cone_phase{phase}_tau{tau:.6g}"
                if name in by_name:
                    if nlp.assumptions.friction_form == "squared":
                        refv = float(a_B[0] ** 2 + a_B[1] ** 2
                                     - (nlp.assumptions.mu * a_B[2]) ** 2)
                    else:
                        refv = float(math.hypot(a_B[0], a_B[1]) - nlp.assumptions.mu * a_B[2])
                    out[name] = (by_name[name], refv)
            # 可达
            R = ref_rot_zyx(pose[3], pose[4], pose[5])
            for leg in stance:
                lo, up = ref_reach(
                    p_B, R, nlp.hip_offsets[leg], foot_at(phase, leg),
                    nlp.params.l_min, nlp.params.l_max,
                )
                for tag, refv in (("31_lower", lo), ("31_upper", up)):
                    name = f"{tag}_leg{leg}_phase{phase}_tau{tau:.6g}"
                    if name in by_name:
                        out[name] = (by_name[name], refv)
        # V-B8: 取每相位第一个采样点
        tau = nlp.sample_times(phase)[0]
        name = f"VB8_smooth_phase{phase}_tau{tau:.6g}"
        if name in obj_by_name:
            L_dot = ref_L_dot_fd(coeffs[phase], I, tau)
            out[name] = (obj_by_name[name], float(L_dot @ L_dot))

    # V-B1: 每个触地腿
    maps = nlp.maps
    h_grid = maps.h.data
    dx, dy = maps.h.dx, maps.h.dy
    for leg in nlp.touchdown_legs():
        name = f"VB1_ground_leg{leg}"
        if name in obj_by_name:
            p = desired[leg]
            fx = (p[0] - maps.h.origin[0]) / dx
            fy = (p[1] - maps.h.origin[1]) / dy
            h_val = bilinear_manual(h_grid, fx, fy)
            out[name] = (obj_by_name[name], (h_val - p[2]) ** 2)
    return out


def check_consistency(checks: dict[str, tuple[float, float]], tol: float = 1e-5) -> list[str]:
    bad = []
    for name, (rep, ref) in checks.items():
        scale = max(1.0, abs(rep), abs(ref))
        if abs(rep - ref) > tol * scale:
            bad.append(f"{name}: report={rep:.6g} ref={ref:.6g}")
    return bad


# ---------------------------------------------------------------------- #
# 运行
# ---------------------------------------------------------------------- #
CONTACT_NORMAL = AssumptionsConfig(inertial_accel_sign="contact_normal_explicit")
LITERAL = AssumptionsConfig()  # 默认: 论文字面


def run_scenario(name: str, assumptions: AssumptionsConfig, verbose: bool = True):
    nlp, guess, note = build_scenario(name, assumptions)
    if name == "static_flat":
        # "静止机身" 的严格表达: 样条钉为常值位姿, 否则字面符号下求解器
        # 会用基座自由落体加速度去"满足"摩擦锥 (数学自洽但物理荒谬).
        nlp.add_constant_base_pin(0)
    nlp.add_initial_state_constraints()
    nlp.add_junction_constraints()
    nlp.add_dynamic_stability_constraints()
    nlp.add_kinematic_contact_constraints()
    nlp.add_objectives()
    nlp.finalize()
    coeffs, desired = guess
    nlp.set_initial_guess(coeffs, desired)
    report = nlp.solve(scenario=f"{name}[{assumptions.inertial_accel_sign}]")
    checks = independent_residual_check(nlp, report)
    if verbose:
        print(report.format())
        print("independent NumPy residual checks (report vs ref):")
        for key, (rep, ref) in list(checks.items())[:12]:
            print(f"  {key:<52s} {rep:>14.6g} vs {ref:>14.6g}")
        if len(checks) > 12:
            print(f"  ... ({len(checks) - 12} more)")
        bad = check_consistency(checks)
        print("consistency:", "OK" if not bad else f"MISMATCH {bad}")
        print("scenario note:", note)
        print()
    return ScenarioResult(name, report, checks, note)


def main() -> int:
    results = []
    # S1 双配置: 字面 (预期不可行, 静止基准) 与 contact_normal (预期可行)
    results.append(run_scenario("static_flat", LITERAL))
    results.append(run_scenario("static_flat", CONTACT_NORMAL))
    for name in ("crawl_flat", "crawl_step", "two_stones", "infeasible_reach"):
        results.append(run_scenario(name, CONTACT_NORMAL))

    print("=== summary ===")
    for r in results:
        bad = check_consistency(r.independent_checks)
        status_short = r.report.status.splitlines()[0][:44]
        print(
            f"{r.report.scenario:<46s} status={status_short:<44s} "
            f"iters={r.report.iterations} obj={r.report.objective_total:.4g} "
            f"maxviol={max(r.report.max_violation_by_category.values(), default=0.0):.2e} "
            f"checks={'OK' if not bad else 'MISMATCH'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
