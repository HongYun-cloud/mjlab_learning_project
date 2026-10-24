"""V-B 目标函数 (式 32-36 / Table I) 与地图查询层的单元测试.

运行:
    python -m unittest myrobot_mjlab.task.DTC.tamols.test_v_b_objectives -v

覆盖 (对应用户要求 6):
  1. 地图查询: 双线性插值与手写 numpy 公式互相对照; 节点值精确;
     边界 nan/clamp/raise 三种策略; 缺失值 median/value/raise 三种填充
     与 validity 诊断; yaw/origin 坐标变换; 梯度场与解析梯度对照;
     符号 (CasADi) 查询与数值查询一致.
  2. 每项目标: 数值与测试文件内**独立**的 NumPy 参考公式复算一致;
     梯度方向用有限差分验证 (推离/拉近/零梯度区间).
  3. 符号表达式 (CasADi MX) 与数值路径在同一点取值一致; 总目标的
     CasADi 精确梯度与独立 numpy 总目标的有限差分一致.
  4. 权重: Table I 数值与 sampled 任务的 T_k 缩放.
  5. 装配: V-B1 保持软目标 (不产生等式约束); 小型 NLP 可解且解是
     **局部**最优 (碗形地图中心), 不声称全局.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from myrobot_mjlab.task.DTC.tamols import solver_backend
from myrobot_mjlab.task.DTC.tamols.map_query import GridMapField, TerrainMaps
from myrobot_mjlab.task.DTC.tamols.v_b_objectives import (
    DEFAULT_EPS_MIN,
    DEFAULT_H_DES,
    GO2_INERTIA_B,
    VBWeights,
    add_v_b_objectives,
    angular_momentum_rate_world,
    assemble_total_objective,
    base_pose_alignment_records,
    base_pose_alignment_term,
    body_omega_rate_zyx,
    body_omega_zyx,
    build_v_b_objective_sets,
    default_sample_times,
    edge_avoidance_records,
    edge_avoidance_residual,
    foothold_ground_records,
    foothold_ground_residual,
    leg_collision_barrier,
    leg_collision_records,
    midstance_from_contacts,
    nominal_kinematics_records,
    nominal_kinematics_residual,
    previous_solution_records,
    previous_solution_residual,
    smoothness_records,
    smoothness_residual,
    tracking_records,
    tracking_residual,
)

CASADI_AVAILABLE = solver_backend.ca is not None

RNG = np.random.default_rng(7)


# ---------------------------------------------------------------------- #
# 独立 NumPy 参考实现 (不复用被测模块的公式)
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


def ref_vb1(h_val, p):
    return (h_val - p[2]) ** 2


def ref_vb2(p_i, p_j, eps):
    z = (p_i[0] - p_j[0]) ** 2 + (p_i[1] - p_j[1]) ** 2
    d = eps - z
    return d * d if d > 0.0 else 0.0


def ref_vb3(p_B, R, r, p, h_des):
    hip = np.asarray(p_B) + np.asarray(R) @ np.asarray(r)
    d = hip - np.asarray(p) - np.array([0.0, 0.0, h_des])
    return float(d @ d)


def ref_vb4_term(p_B, R, r, h_des, hs2):
    hip = np.asarray(p_B) + np.asarray(R) @ np.asarray(r)
    return (hip[2] - h_des - hs2) ** 2


def ref_vb5(grad_h, grad_hs1):
    g = np.asarray(grad_h)
    g1 = np.asarray(grad_hs1)
    return float(g @ g + g1 @ g1)


def ref_vb6(p, p_prev):
    d = np.asarray(p) - np.asarray(p_prev)
    return float(d @ d)


def ref_vb7(v, v_des, w, w_des, I):
    dv = np.asarray(v) - np.asarray(v_des)
    dw = np.asarray(w) - np.asarray(w_des)
    Iw = np.asarray(I) @ dw
    return float(dv @ dv + Iw @ Iw)


def ref_vb8(L_dot):
    L = np.asarray(L_dot)
    return float(L @ L)


def bilinear_manual(grid, fx, fy):
    """手写双线性插值 (网格索引坐标), 独立于被测模块."""
    ny, nx = grid.shape
    fx = min(max(fx, 0.0), nx - 1.0)
    fy = min(max(fy, 0.0), ny - 1.0)
    j0 = min(int(math.floor(fx)), nx - 2)
    i0 = min(int(math.floor(fy)), ny - 2)
    tx, ty = fx - j0, fy - i0
    return (
        (1 - ty) * (1 - tx) * grid[i0, j0]
        + (1 - ty) * tx * grid[i0, j0 + 1]
        + ty * (1 - tx) * grid[i0 + 1, j0]
        + ty * tx * grid[i0 + 1, j0 + 1]
    )


def num_grad(fn, x0, eps=1e-6):
    """标量函数 fn (np.ndarray -> float) 在 x0 的中心差分梯度."""
    x0 = np.asarray(x0, dtype=float)
    grad = np.zeros_like(x0)
    for i in range(x0.size):
        plus = x0.copy()
        minus = x0.copy()
        plus[i] += eps
        minus[i] -= eps
        grad[i] = (fn(plus) - fn(minus)) / (2 * eps)
    return grad


# ---------------------------------------------------------------------- #
# 地图夹具
# ---------------------------------------------------------------------- #
def bowl_maps(size=25, dx=0.05, origin=(-0.6, -0.6), a=2.0):
    """碗形 h (中心最低) + 同形状 hs1 + 平坦 hs2."""
    nx = ny = size
    X, Y = np.meshgrid(np.arange(nx) * dx, np.arange(ny) * dx)
    cx = cy = 0.5 * (size - 1) * dx
    h = a * ((X - cx) ** 2 + (Y - cy) ** 2)
    hs1 = 0.5 * h
    hs2 = np.full_like(h, 0.3)
    return TerrainMaps(h, hs1, hs2, dx=dx, dy=dx, origin=origin), (cx, cy)


def plane_maps(size=21, dx=0.05, origin=(-0.5, -0.5), gx=0.1, gy=0.05):
    nx = ny = size
    X, Y = np.meshgrid(np.arange(nx) * dx, np.arange(ny) * dx)
    h = gx * X + gy * Y
    hs1 = h.copy()
    hs2 = 0.3 + 0.02 * X
    return TerrainMaps(h, hs1, hs2, dx=dx, dy=dx, origin=origin)


def random_coeffs(num_phases=2, num_coeffs=5, scale=1.0):
    coeffs = []
    for _ in range(num_phases):
        c = RNG.normal(0.0, 0.05 * scale, size=(6, num_coeffs))
        c[0, 0], c[1, 0], c[2, 0] = 0.0, 0.0, 0.30  # 起始位置
        c[3:, 0] = 0.0  # 起始姿态水平
        coeffs.append(c)
    return coeffs


# ---------------------------------------------------------------------- #
# 地图查询层
# ---------------------------------------------------------------------- #
class TestGridMapField(unittest.TestCase):
    def setUp(self):
        self.nx, self.ny = 5, 4
        self.dx, self.dy = 0.25, 0.5
        self.grid = np.array(
            [[10.0 * j + i for j in range(self.nx)] for i in range(self.ny)]
        )
        self.field = GridMapField(self.grid, self.dx, self.dy, name="t")

    def test_node_values_exact(self):
        for i in range(self.ny):
            for j in range(self.nx):
                value = self.field.query(j * self.dx, i * self.dy)
                self.assertAlmostEqual(value, self.grid[i, j], places=12)

    def test_bilinear_matches_manual(self):
        for x, y in [(0.1, 0.7), (0.63, 1.21), (0.99, 0.02)]:
            fx, fy = x / self.dx, y / self.dy
            expect = bilinear_manual(self.grid, fx, fy)
            self.assertAlmostEqual(self.field.query(x, y), expect, places=12)

    def test_out_of_bounds_modes(self):
        corner = self.grid[-1, -1]
        self.assertAlmostEqual(
            self.field.query(10.0, 10.0), corner, places=12
        )  # 默认 clamp
        nan_field = GridMapField(
            self.grid, self.dx, self.dy, out_of_bounds="nan", name="n"
        )
        self.assertTrue(math.isnan(nan_field.query(-1.0, 0.0)))
        raise_field = GridMapField(
            self.grid, self.dx, self.dy, out_of_bounds="raise", name="r"
        )
        with self.assertRaises(ValueError):
            raise_field.query(0.0, 100.0)
        self.assertTrue(self.field.covers(0.5, 0.5))
        self.assertFalse(self.field.covers(5.0, 0.5))

    def test_missing_fill_and_validity(self):
        grid = self.grid.copy()
        grid[1, 1] = float("nan")
        with self.assertRaises(ValueError):
            GridMapField(grid, self.dx, self.dy, fill="raise", name="r")
        median_field = GridMapField(grid, self.dx, self.dy, name="m")
        self.assertEqual(median_field.invalid_cells, 1)
        # 填充值 = 有效格的中位数 (与 map_process.fill_invalid 同策略)
        expect_fill = float(np.nanmedian(grid))
        self.assertAlmostEqual(median_field.fill_value_used, expect_fill, places=12)
        # 缺失格处查询 == 填充值; 有效性 < 1
        self.assertAlmostEqual(
            median_field.query(1 * self.dx, 1 * self.dy), expect_fill, places=12
        )
        self.assertLess(median_field.validity(1 * self.dx, 1 * self.dy), 1.0)
        self.assertAlmostEqual(
            self.field.validity(1 * self.dx, 1 * self.dy), 1.0, places=12
        )
        value_field = GridMapField(
            grid, self.dx, self.dy, fill="value", fill_value=-7.0, name="v"
        )
        self.assertAlmostEqual(
            value_field.query(1 * self.dx, 1 * self.dy), -7.0, places=12
        )

    def test_origin_and_yaw(self):
        field = GridMapField(
            self.grid, self.dx, self.dy, origin=(1.0, 2.0), yaw=math.pi / 2,
            name="rot",
        )
        # 世界点 p 满足 R(yaw)^T (p - origin) = (0.25, 0.5) -> grid[1, 1]
        # R(pi/2) = [[0,-1],[1,0]]; p = origin + R(yaw) @ (0.25, 0.5)
        p = np.array([1.0, 2.0]) + ref_rot_zyx(0, 0, math.pi / 2)[:2, :2] @ np.array(
            [0.25, 0.5]
        )
        self.assertAlmostEqual(field.query(p[0], p[1]), self.grid[1, 1], places=10)

    @unittest.skipUnless(CASADI_AVAILABLE, "CasADi 未安装")
    def test_symbolic_matches_numeric(self):
        import casadi as ca

        x = ca.MX.sym("x")
        y = ca.MX.sym("y")
        f = ca.Function("f", [x, y], [self.field.query(x, y)])
        for px, py in [(0.1, 0.7), (0.63, 1.21), (0.0, 0.0)]:
            self.assertAlmostEqual(
                float(f(px, py)), self.field.query(px, py), places=12
            )
        # 符号查询越界 = clamp 到角点
        self.assertAlmostEqual(
            float(f(50.0, 50.0)), self.grid[-1, -1], places=12
        )

    @unittest.skipUnless(CASADI_AVAILABLE, "CasADi 未安装")
    def test_symbolic_jacobian_is_cell_gradient(self):
        import casadi as ca

        x = ca.MX.sym("x")
        y = ca.MX.sym("y")
        expr = self.field.query(x, y)
        J = ca.jacobian(expr, ca.vertcat(x, y))
        f = ca.Function("f", [x, y], [J])
        # 格 (0,0)-(1,1) 内双线性: d/dx = (1-ty)(v01-v00)+ty(v11-v10) / dx
        px, py = 0.1, 0.7
        ty = (py / self.dy) - 0.0
        expect_dx = ((1 - ty) * (self.grid[0, 1] - self.grid[0, 0])
                     + ty * (self.grid[1, 1] - self.grid[1, 0])) / self.dx
        expect_dy = self.dx * 0 + (self.grid[1, 0] - self.grid[0, 0]) / self.dy
        out = f(px, py).full().ravel()
        self.assertAlmostEqual(out[0], expect_dx, places=10)
        self.assertAlmostEqual(out[1], expect_dy, places=10)


class TestTerrainMaps(unittest.TestCase):
    def test_gradient_matches_analytic_plane(self):
        maps = plane_maps(gx=0.1, gy=0.05)
        gx, gy = maps.grad_h(0.0, 0.0)
        self.assertAlmostEqual(gx, 0.1, places=8)
        self.assertAlmostEqual(gy, 0.05, places=8)

    def test_gradient_matches_analytic_bowl(self):
        maps, (cx, cy) = bowl_maps(a=2.0)
        origin = np.array(maps.h.origin)
        p_world = origin + np.array([cx + 0.1, cy - 0.05])
        gx, gy = maps.grad_h(p_world[0], p_world[1])
        # h = a((X-cx)^2+(Y-cy)^2): ∇h = 2a(X-cx, Y-cy) = 2a(0.1, -0.05)
        self.assertAlmostEqual(gx, 2 * 2.0 * 0.1, places=6)
        self.assertAlmostEqual(gy, 2 * 2.0 * -0.05, places=6)

    def test_hs2_and_validity(self):
        maps, _ = bowl_maps()
        self.assertAlmostEqual(maps.height_hs2(0.0, 0.0), 0.3, places=12)
        validity = maps.validity_at(0.0, 0.0)
        self.assertTrue(all(v == 1.0 for v in validity.values()))

    def test_missing_hs1_or_hs2_raises(self):
        nx = 10
        h = np.zeros((nx, nx))
        maps = TerrainMaps(h, dx=0.1, dy=0.1)
        with self.assertRaises(RuntimeError):
            maps.height_hs2(0.0, 0.0)
        with self.assertRaises(RuntimeError):
            maps.grad_hs1(0.0, 0.0)

    def test_batch_input_squeezed_and_rejected(self):
        h = np.zeros((1, 1, 8, 8))
        maps = TerrainMaps(h, dx=0.1, dy=0.1)
        self.assertEqual(maps.h.ny, 8)
        with self.assertRaises(ValueError):
            TerrainMaps(np.zeros((2, 1, 8, 8)), dx=0.1, dy=0.1)


# ---------------------------------------------------------------------- #
# 残差数值 + 独立复算 + 梯度方向
# ---------------------------------------------------------------------- #
class TestResidualValues(unittest.TestCase):
    def test_vb1_value_and_gradient(self):
        h_val, p = 0.3, np.array([0.1, -0.2, 0.2])
        value = foothold_ground_residual(h_val, p)
        self.assertAlmostEqual(value, ref_vb1(h_val, p), places=12)
        self.assertAlmostEqual(value, 0.01, places=12)
        grad = num_grad(lambda q: foothold_ground_residual(h_val, q), p)
        # d/dp = (0, 0, -2(h - p_z)) = (0, 0, -0.2): 抬高 p_z 代价下降
        np.testing.assert_allclose(grad, [0.0, 0.0, -0.2], atol=1e-6)
        on_ground = np.array([0.1, -0.2, 0.3])
        grad0 = num_grad(lambda q: foothold_ground_residual(h_val, q), on_ground)
        np.testing.assert_allclose(grad0, 0.0, atol=1e-6)

    def test_vb2_barrier_value_and_directions(self):
        eps = 0.04  # (0.2 m)^2
        close = np.array([0.05, 0.0, 0.0])
        other = np.array([0.0, 0.0, 0.0])
        value = leg_collision_barrier(close, other, eps)
        self.assertAlmostEqual(value, ref_vb2(close, other, eps), places=12)
        z = 0.05**2
        self.assertAlmostEqual(value, (eps - z) ** 2, places=12)
        grad = num_grad(lambda q: leg_collision_barrier(q, other, eps), close)
        # 沿分离方向 (close - other) 移动应使代价下降
        direction = (close - other) / np.linalg.norm(close - other)
        self.assertLess(grad @ direction, 0.0)
        far = np.array([0.5, 0.0, 0.0])
        self.assertEqual(leg_collision_barrier(far, other, eps), 0.0)
        grad_far = num_grad(lambda q: leg_collision_barrier(q, other, eps), far)
        np.testing.assert_allclose(grad_far, 0.0, atol=1e-10)
        with self.assertRaises(ValueError):
            leg_collision_barrier(close, other, -1.0)

    def test_vb3_value_and_gradient(self):
        p_B = np.array([0.0, 0.0, 0.3])
        R = ref_rot_zyx(0.1, -0.2, 0.7)
        r = np.array([0.19, 0.15, 0.0])
        p = np.array([0.3, 0.2, 0.05])
        value = nominal_kinematics_residual(p_B, R, r, p, DEFAULT_H_DES)
        self.assertAlmostEqual(
            value, ref_vb3(p_B, R, r, p, DEFAULT_H_DES), places=12
        )
        grad = num_grad(
            lambda q: nominal_kinematics_residual(p_B, R, r, q, DEFAULT_H_DES), p
        )
        hip = p_B + R @ r
        expect = -2 * (hip - p - np.array([0, 0, DEFAULT_H_DES]))
        np.testing.assert_allclose(grad, expect, atol=1e-6)

    def test_vb4_term_value_and_gradient(self):
        p_B = np.array([0.0, 0.0, 0.32])
        R = ref_rot_zyx(0.0, 0.0, 0.3)
        r = np.array([0.19, 0.15, 0.0])
        hs2 = 0.3
        value = base_pose_alignment_term(p_B, R, r, hs2, DEFAULT_H_DES)
        self.assertAlmostEqual(
            value, ref_vb4_term(p_B, R, r, DEFAULT_H_DES, hs2), places=12
        )
        grad = num_grad(
            lambda q: base_pose_alignment_term(q, R, r, hs2, DEFAULT_H_DES), p_B
        )
        delta = (p_B + R @ r)[2] - DEFAULT_H_DES - hs2
        # hs2 平坦: 只 z 方向有梯度, 2*delta
        np.testing.assert_allclose(grad, [0.0, 0.0, 2 * delta], atol=1e-6)

    def test_vb5_value_and_gradient_direction(self):
        maps, (cx, cy) = bowl_maps(a=2.0)
        origin = np.array(maps.h.origin)
        offset = np.array([0.15, -0.1])
        p_map = np.array([cx, cy]) + offset
        p_world = origin + p_map
        gh = maps.grad_h(p_world[0], p_world[1])
        gh1 = maps.grad_hs1(p_world[0], p_world[1])
        value = edge_avoidance_residual(gh, gh1)
        self.assertAlmostEqual(value, ref_vb5(gh, gh1), places=10)
        # 碗形: 代价 ~ r^2, 沿 -offset (朝碗底) 方向应下降
        def cost(q):
            w = origin + q
            g0 = maps.grad_h(w[0], w[1])
            g1 = maps.grad_hs1(w[0], w[1])
            return edge_avoidance_residual(g0, g1)

        grad = num_grad(cost, p_map, eps=1e-5)
        radial_in = -offset / np.linalg.norm(offset)
        self.assertLess(grad @ radial_in, 0.0)
        # 平面地图梯度恒定 -> 代价与位置无关, 梯度为 0
        flat = plane_maps()
        g0 = flat.grad_h(0.0, 0.0)
        g1 = flat.grad_hs1(0.0, 0.0)

        def cost_flat(q):
            a = flat.grad_h(q[0], q[1])
            b = flat.grad_hs1(q[0], q[1])
            return edge_avoidance_residual(a, b)

        grad_flat = num_grad(cost_flat, np.array([0.1, 0.2]), eps=1e-5)
        np.testing.assert_allclose(grad_flat, 0.0, atol=1e-8)

    def test_vb6_value_and_gradient(self):
        p = np.array([0.4, 0.1, 0.1])
        prev = np.array([0.3, 0.2, 0.05])
        value = previous_solution_residual(p, prev)
        self.assertAlmostEqual(value, ref_vb6(p, prev), places=12)
        grad = num_grad(lambda q: previous_solution_residual(q, prev), p)
        np.testing.assert_allclose(grad, 2 * (p - prev), atol=1e-6)

    def test_vb7_value_and_gradient(self):
        I = np.array(GO2_INERTIA_B)
        v = np.array([0.4, -0.1, 0.0])
        v_des = np.array([0.5, 0.0, 0.0])
        w = np.array([0.0, 0.0, 0.3])
        w_des = np.array([0.0, 0.0, 0.1])
        value = tracking_residual(v, v_des, w, w_des, I)
        self.assertAlmostEqual(value, ref_vb7(v, v_des, w, w_des, I), places=12)
        grad_w = num_grad(lambda q: tracking_residual(v, v_des, q, w_des, I), w)
        dw = w - w_des
        np.testing.assert_allclose(grad_w, 2 * (I.T @ I @ dw), atol=1e-6)
        grad_v = num_grad(lambda q: tracking_residual(q, v_des, w, w_des, I), v)
        np.testing.assert_allclose(grad_v, 2 * (v - v_des), atol=1e-6)
        # 参考值处梯度为零, 代价为零
        zero = tracking_residual(v_des, v_des, w_des, w_des, I)
        self.assertAlmostEqual(zero, 0.0, places=12)

    def test_vb8_value_and_gradient(self):
        I = np.array(GO2_INERTIA_B)
        R = ref_rot_zyx(0.1, 0.2, -0.3)
        omega = np.array([0.1, -0.2, 0.3])
        omega_dot = np.array([0.5, 0.4, -0.6])
        L_dot = angular_momentum_rate_world(R, I, omega, omega_dot)
        value = smoothness_residual(L_dot)
        self.assertAlmostEqual(value, ref_vb8(L_dot), places=12)
        # omega = 0 时 L_dot = R I ω̇, 梯度 wrt ω̇ = 2 I^T I ω̇
        L0 = angular_momentum_rate_world(R, I, (0.0, 0.0, 0.0), omega_dot)
        grad = num_grad(
            lambda q: smoothness_residual(
                angular_momentum_rate_world(R, I, (0.0, 0.0, 0.0), q)
            ),
            omega_dot,
        )
        np.testing.assert_allclose(grad, 2 * (I.T @ I @ omega_dot), atol=1e-6)
        self.assertAlmostEqual(
            smoothness_residual(L0), ref_vb8(R @ I @ omega_dot), places=10
        )


class TestAngularMomentumRate(unittest.TestCase):
    """L_dot 解析式 vs 数值差分 vs CasADi 符号时间求导 (三方对照)."""

    def _euler_poly(self, t):
        # 任意光滑欧拉角轨迹: phi=0.1 t + 0.05 t^2, theta=..., psi=...
        phi = 0.1 * t + 0.05 * t**2
        theta = -0.2 * t + 0.03 * t**2
        psi = 0.4 * t - 0.02 * t**2
        return phi, theta, psi

    def _euler_rates(self, t):
        return (
            0.1 + 0.1 * t,
            -0.2 + 0.06 * t,
            0.4 - 0.04 * t,
        )

    def _euler_accels(self, t):
        return (0.1, 0.06, -0.04)

    def _L_world(self, t):
        I = np.array(GO2_INERTIA_B)
        phi, theta, psi = self._euler_poly(t)
        R = ref_rot_zyx(phi, theta, psi)
        w = ref_omega((phi, theta, psi), self._euler_rates(t))
        return R @ I @ w

    def test_matches_finite_difference(self):
        I = GO2_INERTIA_B
        for t in (0.0, 0.13, 0.37):
            phi, theta, psi = self._euler_poly(t)
            R = ref_rot_zyx(phi, theta, psi)
            w = ref_omega((phi, theta, psi), self._euler_rates(t))
            w_dot = body_omega_rate_zyx(
                (phi, theta, psi), self._euler_rates(t), self._euler_accels(t)
            )
            L_dot = np.array(angular_momentum_rate_world(R, I, w, w_dot))
            h = 1e-5
            fd = (self._L_world(t + h) - self._L_world(t - h)) / (2 * h)
            np.testing.assert_allclose(L_dot, fd, atol=1e-5)

    def test_body_omega_rate_matches_finite_difference(self):
        for t in (0.0, 0.21, 0.44):
            phi, theta, psi = self._euler_poly(t)
            w_dot = np.array(
                body_omega_rate_zyx(
                    (phi, theta, psi), self._euler_rates(t), self._euler_accels(t)
                )
            )
            h = 1e-5
            wp = ref_omega(self._euler_poly(t + h), self._euler_rates(t + h))
            wm = ref_omega(self._euler_poly(t - h), self._euler_rates(t - h))
            np.testing.assert_allclose(w_dot, (wp - wm) / (2 * h), atol=1e-5)

    @unittest.skipUnless(CASADI_AVAILABLE, "CasADi 未安装")
    def test_matches_casadi_symbolic_time(self):
        import casadi as ca

        I = ca.DM(GO2_INERTIA_B)
        t = ca.MX.sym("t")
        # 与 _euler_poly 相同的多项式, 用符号时间构造
        phi = 0.1 * t + 0.05 * t**2
        theta = -0.2 * t + 0.03 * t**2
        psi = 0.4 * t - 0.02 * t**2
        rates = (
            ca.jacobian(phi, t).T,
            ca.jacobian(theta, t).T,
            ca.jacobian(psi, t).T,
        )
        from myrobot_mjlab.task.DTC.tamols.v_b_objectives import (
            angular_momentum_world,
            rotation_matrix_zyx as rm,
        )

        R = rm(phi, theta, psi)
        omega = body_omega_zyx((phi, theta, psi), rates)
        L = angular_momentum_world(R, I, omega)
        L_dot = ca.jacobian(ca.vertcat(*L), t).T
        f = ca.Function("f", [t], [L_dot])
        I_np = GO2_INERTIA_B
        for tv in (0.0, 0.13, 0.37):
            phi_n, theta_n, psi_n = self._euler_poly(tv)
            R_n = ref_rot_zyx(phi_n, theta_n, psi_n)
            w_n = ref_omega((phi_n, theta_n, psi_n), self._euler_rates(tv))
            w_dot_n = body_omega_rate_zyx(
                (phi_n, theta_n, psi_n),
                self._euler_rates(tv),
                self._euler_accels(tv),
            )
            mine = np.array(
                angular_momentum_rate_world(R_n, I_np, w_n, w_dot_n)
            )
            np.testing.assert_allclose(
                mine, np.array(f(tv)).ravel(), atol=1e-9
            )


# ---------------------------------------------------------------------- #
# 符号 vs 数值
# ---------------------------------------------------------------------- #
@unittest.skipUnless(CASADI_AVAILABLE, "CasADi 未安装")
class TestSymbolicMatchesNumeric(unittest.TestCase):
    def setUp(self):
        import casadi as ca

        self.ca = ca
        self.rng = np.random.default_rng(11)

    def test_residuals_symbolic_equal_numeric(self):
        ca = self.ca
        p = ca.MX.sym("p", 3)
        q = ca.MX.sym("q", 3)
        h_val = ca.MX.sym("h")
        eps = ca.MX.sym("eps")
        rng = self.rng
        pv = rng.normal(size=3)
        qv = rng.normal(size=3)
        hv = 0.27
        epsv = 0.05

        cases = [
            (
                foothold_ground_residual(h_val, p),
                [h_val, p],
                [hv, pv],
                foothold_ground_residual(hv, pv),
            ),
            (
                leg_collision_barrier(p, q, eps),
                [eps, p, q],
                [epsv, pv, qv],
                leg_collision_barrier(pv, qv, epsv),
            ),
            (
                previous_solution_residual(p, q),
                [p, q],
                [pv, qv],
                previous_solution_residual(pv, qv),
            ),
        ]
        for expr, syms, vals, expect in cases:
            f = ca.Function("f", syms, [expr])
            got = float(f(*vals))
            self.assertAlmostEqual(got, float(expect), places=10)

    def test_map_query_inside_records_symbolic(self):
        ca = self.ca
        maps, _ = bowl_maps()
        p = ca.MX.sym("p", 3)
        expr = foothold_ground_residual(maps.height_h(p[0], p[1]), p)
        f = ca.Function("f", [p], [expr])
        pv = np.array([0.1, -0.05, 0.12])
        expect = foothold_ground_residual(
            maps.height_h(pv[0], pv[1]), pv
        )
        self.assertAlmostEqual(float(f(pv)), float(expect), places=10)


# ---------------------------------------------------------------------- #
# 权重与记录
# ---------------------------------------------------------------------- #
class TestWeights(unittest.TestCase):
    def test_table_values(self):
        w = VBWeights()
        self.assertEqual(w.base_weight("V-B1"), 1e4)
        self.assertEqual(w.base_weight("V-B2"), 1e-3)
        self.assertEqual(w.base_weight("V-B3"), 7.0)
        self.assertEqual(w.base_weight("V-B4"), 100.0)
        self.assertEqual(w.base_weight("V-B5"), 3.0)
        self.assertEqual(w.base_weight("V-B6"), 1e-2)
        self.assertEqual(w.base_weight("V-B7"), 2.0)
        self.assertEqual(w.base_weight("V-B8"), 1e-3)

    def test_sampled_scaling(self):
        w = VBWeights()
        T = 0.3
        self.assertAlmostEqual(w.weight_for("V-B4", T), 100.0 * T / 6, places=12)
        self.assertAlmostEqual(w.weight_for("V-B7", T), 2.0 * T / 6, places=12)
        self.assertAlmostEqual(w.weight_for("V-B8", T), 1e-3 * T / 6, places=12)
        self.assertEqual(w.weight_for("V-B1"), 1e4)  # 非采样不需要 duration
        with self.assertRaises(ValueError):
            w.weight_for("V-B4")
        with self.assertRaises(ValueError):
            w.weight_for("V-B4", -1.0)
        with self.assertRaises(ValueError):
            w.base_weight("V-B9")

    def test_sample_times(self):
        times = default_sample_times(0.3, 6)
        self.assertEqual(len(times), 6)
        self.assertAlmostEqual(times[0], 0.0, places=12)
        self.assertAlmostEqual(times[-1], 0.25, places=12)
        with self.assertRaises(ValueError):
            default_sample_times(0.0)


class TestRecords(unittest.TestCase):
    def setUp(self):
        self.maps, self.center = bowl_maps()
        self.footholds = np.array(
            [
                [0.2, 0.15, 0.05],
                [0.2, -0.15, 0.05],
                [-0.2, 0.15, 0.05],
                [-0.2, -0.15, 0.05],
            ]
        )
        self.coeffs = random_coeffs(2)
        self.durations = [0.3, 0.3]

    def test_counts(self):
        sets = foothold_ground_records(self.maps, self.footholds)
        self.assertEqual(len(sets), 4)
        coll = leg_collision_records(self.footholds)
        self.assertEqual(len(coll), 6)
        edge = edge_avoidance_records(self.maps, self.footholds)
        self.assertEqual(len(edge), 4)
        prev = previous_solution_records(self.footholds, self.footholds * 0.9)
        self.assertEqual(len(prev), 4)
        align = base_pose_alignment_records(
            self.coeffs, self.durations, self.maps
        )
        self.assertEqual(len(align), 2 * 6 * 4)
        self.assertTrue(all(r.sampled for r in align))
        smooth = smoothness_records(self.coeffs, self.durations)
        self.assertEqual(len(smooth), 2 * 6)
        track = tracking_records(
            self.coeffs,
            self.durations,
            lambda phase, tau: (np.zeros(3), np.zeros(3)),
        )
        self.assertEqual(len(track), 2 * 6)

    def test_sampled_weights_in_records(self):
        align = base_pose_alignment_records(
            self.coeffs, self.durations, self.maps
        )
        for record in align:
            self.assertAlmostEqual(record.weight, 100.0 * 0.3 / 6, places=12)

    def test_midstance(self):
        contacts = [[0, 1], [2, 3]]
        mid = midstance_from_contacts(contacts, self.durations)
        self.assertEqual(set(mid.keys()), {0, 1, 2, 3})
        self.assertEqual(mid[0][0], 0)
        self.assertAlmostEqual(mid[0][1], 0.15, places=12)
        self.assertEqual(mid[2][0], 1)
        self.assertAlmostEqual(mid[2][1], 0.15, places=12)
        records = nominal_kinematics_records(
            self.coeffs, self.footholds, mid
        )
        self.assertEqual(len(records), 4)
        for record in records:
            self.assertEqual(record.category, "V-B3")
        # 从未支撑的腿被跳过
        mid_partial = midstance_from_contacts([[0], [0]], self.durations)
        self.assertEqual(set(mid_partial.keys()), {0})

    def test_total_equals_weighted_sum(self):
        sets = foothold_ground_records(self.maps, self.footholds)
        total = sets.total()
        expect = sum(r.weight * r.value() for r in sets)
        self.assertAlmostEqual(float(total), expect, places=10)

    def test_build_skips_missing(self):
        merged = build_v_b_objective_sets(
            maps=self.maps,
            spline_coeffs=self.coeffs,
            durations=self.durations,
            footholds=self.footholds,
            contact_sets=[[0, 1], [2, 3]],
        )
        categories = set(merged.by_category().keys())
        self.assertEqual(
            categories, {"V-B1", "V-B2", "V-B3", "V-B4", "V-B5", "V-B8"}
        )
        self.assertIn("V-B6 skipped", merged.note)
        self.assertIn("V-B7 skipped", merged.note)
        full = build_v_b_objective_sets(
            maps=self.maps,
            spline_coeffs=self.coeffs,
            durations=self.durations,
            footholds=self.footholds,
            previous_footholds=self.footholds,
            contact_sets=[[0, 1], [2, 3]],
            ref_twist=lambda phase, tau: (np.zeros(3), np.zeros(3)),
        )
        self.assertEqual(
            set(full.by_category().keys()),
            {"V-B1", "V-B2", "V-B3", "V-B4", "V-B5", "V-B6", "V-B7", "V-B8"},
        )


# ---------------------------------------------------------------------- #
# CasADi 组装与求解
# ---------------------------------------------------------------------- #
@unittest.skipUnless(CASADI_AVAILABLE, "CasADi 未安装")
class TestAssembly(unittest.TestCase):
    def setUp(self):
        self.maps, self.center = bowl_maps()
        self.backend = solver_backend.make_backend(
            "casadi", solver_options={"ipopt": {"print_level": 0, "max_iter": 200}}
        )

    def test_ground_task_is_soft_not_equality(self):
        backend = self.backend
        p = backend.variable((4, 3), name="footholds")
        sets = foothold_ground_records(self.maps, p)
        add_v_b_objectives(backend, sets)
        backend.minimize()
        # 软目标: 没有任何等式/不等式约束被加入
        self.assertEqual(backend.constraints, [])
        backend.set_initial(p, np.array([[0.1, 0.1, 0.4]] * 4))
        backend.solve()
        sol = backend.value(p)
        # 碗底高度 h=0; 只有贴地代价时解应贴到 h(p) (局部最优即碗底附近)
        for i in range(4):
            h_at = self.maps.height_h(sol[i, 0], sol[i, 1])
            self.assertLess(abs(h_at - sol[i, 2]), 1e-4)

    def test_casadi_gradient_matches_independent_fd(self):
        import casadi as ca

        backend = self.backend
        # jacobian 要求自变量是原始符号向量: 用 12 维 sym + reshape 构造
        # p (reshape 为列优先: p[i, j] = x[4*j + i]).
        x = ca.MX.sym("x", 12)
        p = ca.reshape(x, 4, 3)
        prev = np.array(
            [
                [0.05, 0.05, 0.0],
                [-0.05, 0.05, 0.0],
                [0.05, -0.05, 0.0],
                [-0.05, -0.05, 0.0],
            ]
        )
        from myrobot_mjlab.task.DTC.tamols.v_b_objectives import (
            edge_avoidance_records as edge_rec,
            foothold_ground_records as ground_rec,
            leg_collision_records as coll_rec,
            merge_objective_sets,
            previous_solution_records as prev_rec,
        )

        # 只组装与落脚点相关的四项, 参考公式逐项独立复算
        sets = merge_objective_sets(
            ground_rec(self.maps, p),
            coll_rec(p),
            edge_rec(self.maps, p),
            prev_rec(p, prev),
        )
        total = assemble_total_objective(sets)
        J = ca.jacobian(total, x)
        f = ca.Function("f", [x], [total, J])
        # 评估点刻意避开网格节点/格线: 双线性插值的梯度在格线上不连续,
        # 节点处 CasADi 取右格单侧导数, 中心差分会平均两侧, 无法比较.
        pts0 = np.array(
            [
                [0.213, 0.161, 0.05],
                [0.213, -0.139, 0.05],
                [-0.187, 0.161, 0.05],
                [-0.187, -0.139, 0.05],
            ]
        )
        x0 = pts0.T.ravel()  # 与 reshape 的列优先布局一致

        # 独立 numpy 总目标: 手写双线性查 h 与梯度场
        maps = self.maps
        h_grid = maps.h.data
        gx_grid = maps.grad_h_x.data
        gy_grid = maps.grad_h_y.data
        g1x_grid = maps.grad_hs1_x.data
        g1y_grid = maps.grad_hs1_y.data
        dx, dy = maps.h.dx, maps.h.dy
        w = VBWeights()

        def ref_total(xv):
            pts = xv.reshape(3, 4).T  # 列优先布局的逆
            cost = 0.0
            for i in range(4):
                fx = (pts[i, 0] - maps.h.origin[0]) / dx
                fy = (pts[i, 1] - maps.h.origin[1]) / dy
                h_val = bilinear_manual(h_grid, fx, fy)
                cost += w.base_weight("V-B1") * ref_vb1(h_val, pts[i])
                gh = (
                    bilinear_manual(gx_grid, fx, fy),
                    bilinear_manual(gy_grid, fx, fy),
                )
                gh1 = (
                    bilinear_manual(g1x_grid, fx, fy),
                    bilinear_manual(g1y_grid, fx, fy),
                )
                cost += w.base_weight("V-B5") * ref_vb5(gh, gh1)
                cost += w.base_weight("V-B6") * ref_vb6(pts[i], prev[i])
            for i, j in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]:
                cost += w.base_weight("V-B2") * ref_vb2(
                    pts[i], pts[j], DEFAULT_EPS_MIN
                )
            return cost

        value, jac = f(x0)
        fd = num_grad(ref_total, x0, eps=1e-6)
        self.assertAlmostEqual(float(value), ref_total(x0), places=8)
        np.testing.assert_allclose(np.array(jac).ravel(), fd, rtol=1e-4, atol=1e-6)

    def test_small_nlp_local_optimum_at_bowl_center(self):
        backend = self.backend
        p = backend.variable((4, 3), name="footholds")
        sets = foothold_ground_records(self.maps, p)
        edge = edge_avoidance_records(self.maps, p)
        from myrobot_mjlab.task.DTC.tamols.v_b_objectives import (
            merge_objective_sets,
        )

        add_v_b_objectives(backend, merge_objective_sets(sets, edge))
        backend.minimize()
        origin = np.array(self.maps.h.origin)
        center_world = origin + np.array(self.center)
        offsets = np.array(
            [
                [0.1, 0.05],
                [0.1, -0.05],
                [-0.1, 0.05],
                [-0.1, -0.05],
            ]
        )
        init = np.column_stack(
            [center_world[0] + offsets[:, 0],
             center_world[1] + offsets[:, 1],
             np.full(4, 0.2)]
        )
        backend.set_initial(p, init)
        backend.solve()
        sol = backend.value(p)
        # 碗底同时是贴地与避边缘的局部最优; 只声称局部
        for i in range(4):
            self.assertLess(
                np.linalg.norm(sol[i, :2] - center_world), 0.02,
                msg=f"leg {i} 未收敛到碗底附近的局部最优",
            )
            self.assertLess(abs(sol[i, 2]), 1e-3)


if __name__ == "__main__":
    unittest.main()
