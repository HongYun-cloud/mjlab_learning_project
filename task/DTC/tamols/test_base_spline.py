"""base_spline 的单元测试.

覆盖: 系数布局、已知多项式求值、各阶导数、batch 维、欧拉角到机体系角速度、
旋转矩阵正交性、相位拼接残差与异常路径.

运行:
    python -m unittest myrobot_mjlab.task.DTC.tamols.test_base_spline -v
"""

from __future__ import annotations

import math
import unittest

import torch

from myrobot_mjlab.task.DTC.tamols.base_spline import Spline6D

DTYPE = torch.float64


def zero_coeffs(num_phases: int = 1, k: int = 5) -> torch.Tensor:
    return torch.zeros(num_phases, 6, k, dtype=DTYPE)


class TestSplineBasics(unittest.TestCase):
    def test_defaults(self):
        spline = Spline6D()
        self.assertEqual(spline.num_coeffs, 5)
        self.assertEqual(spline.degree, 4)
        self.assertEqual(spline.coefficient_shape(1), (1, 6, 5))
        self.assertEqual(spline.coefficient_shape(3), (3, 6, 5))

    def test_zeros_helper(self):
        spline = Spline6D()
        coeffs = spline.zeros(2, dtype=DTYPE)
        self.assertEqual(tuple(coeffs.shape), (2, 6, 5))
        self.assertTrue(bool((coeffs == 0).all()))

    def test_configurable_quintic(self):
        spline = Spline6D(num_coeffs=6)
        self.assertEqual(spline.degree, 5)
        coeffs = zero_coeffs(k=6)
        coeffs[0, 0, 5] = 1.0
        self.assertAlmostEqual(
            float(spline.evaluate(coeffs, 2.0)[0, 0, 0]), 32.0, places=12
        )

    def test_bad_num_coeffs(self):
        with self.assertRaises(ValueError):
            Spline6D(0)
        with self.assertRaises(TypeError):
            Spline6D(4.0)  # type: ignore[arg-type]

    def test_coefficient_shape_validation(self):
        spline = Spline6D()
        with self.assertRaises(ValueError):
            spline.coefficient_shape(0)


class TestEvaluation(unittest.TestCase):
    def setUp(self):
        self.spline = Spline6D()
        # p_d(tau) = (d+1) * (1 + 2 tau + 3 tau^2 + 4 tau^3 + 5 tau^4)
        self.coeffs = zero_coeffs()
        for d in range(6):
            for i in range(5):
                self.coeffs[0, d, i] = (d + 1) * (i + 1)

    def test_known_polynomial(self):
        tau = torch.tensor([0.0, 0.5, 1.0], dtype=DTYPE)
        pose = self.spline.evaluate(self.coeffs, tau)
        self.assertEqual(tuple(pose.shape), (1, 3, 6))

        # s(tau) = 1 + 2tau + 3tau^2 + 4tau^3 + 5tau^4
        s = 1 + 2 * tau + 3 * tau**2 + 4 * tau**3 + 5 * tau**4
        for d in range(6):
            self.assertTrue(
                torch.allclose(pose[0, :, d], (d + 1) * s, atol=1e-12)
            )

    def test_position_and_euler_split(self):
        tau = torch.tensor([0.25, 0.75], dtype=DTYPE)
        pose = self.spline.evaluate(self.coeffs, tau)
        self.assertTrue(
            torch.equal(self.spline.position(self.coeffs, tau), pose[..., :3])
        )
        self.assertTrue(
            torch.equal(self.spline.euler(self.coeffs, tau), pose[..., 3:])
        )

    def test_derivatives_of_known_polynomial(self):
        # d0: 1 + 2t + 3t^2 + 4t^3 + 5t^4
        tau = torch.tensor([0.0, 0.3, 1.2], dtype=DTYPE)
        coeffs = zero_coeffs()
        coeffs[0, 0, :] = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=DTYPE)

        expected = {
            0: 1 + 2 * tau + 3 * tau**2 + 4 * tau**3 + 5 * tau**4,
            1: 2 + 6 * tau + 12 * tau**2 + 20 * tau**3,
            2: 6 + 24 * tau + 60 * tau**2,
            3: 24 + 120 * tau,
            4: torch.full_like(tau, 120.0),
            5: torch.zeros_like(tau),
        }
        for order, want in expected.items():
            got = self.spline.evaluate(coeffs, tau, order)[0, :, 0]
            self.assertTrue(torch.allclose(got, want, atol=1e-12), order)

    def test_velocity_acceleration_jerk_aliases(self):
        tau = torch.tensor([0.4], dtype=DTYPE)
        vel = self.spline.velocity(self.coeffs, tau)
        acc = self.spline.acceleration(self.coeffs, tau)
        jerk = self.spline.jerk(self.coeffs, tau)
        for got, order in ((vel, 1), (acc, 2), (jerk, 3)):
            self.assertTrue(
                torch.equal(got, self.spline.evaluate(self.coeffs, tau, order))
            )

    def test_scalar_tau_and_batch(self):
        # 标量 tau -> N=1
        out = self.spline.evaluate(self.coeffs, 0.5)
        self.assertEqual(tuple(out.shape), (1, 1, 6))
        # [B, P, 6, K] -> [B, P, N, 6]
        batch = self.coeffs.expand(2, 3, 6, 5).clone()
        tau = torch.tensor([0.0, 0.5], dtype=DTYPE)
        out = self.spline.evaluate(batch, tau)
        self.assertEqual(tuple(out.shape), (2, 3, 2, 6))
        self.assertTrue(
            torch.allclose(out[1, 2], self.spline.evaluate(self.coeffs, tau)[0])
        )

    def test_higher_order_than_degree_is_zero(self):
        tau = torch.tensor([0.0, 1.0], dtype=DTYPE)
        out = self.spline.evaluate(self.coeffs, tau, order=5)
        self.assertEqual(tuple(out.shape), (1, 2, 6))
        self.assertTrue(bool((out == 0).all()))
        out = self.spline.evaluate(self.coeffs, tau, order=9)
        self.assertEqual(tuple(out.shape), (1, 2, 6))
        self.assertTrue(bool((out == 0).all()))

    def test_invalid_inputs(self):
        tau = torch.tensor([0.0], dtype=DTYPE)
        with self.assertRaises(TypeError):
            self.spline.evaluate([[0.0] * 5] * 6, tau)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            self.spline.evaluate(torch.zeros(6, 5, dtype=torch.int64), tau)
        with self.assertRaises(ValueError):
            self.spline.evaluate(torch.zeros(5, 5, dtype=DTYPE), tau)
        with self.assertRaises(ValueError):
            self.spline.evaluate(torch.zeros(6, 4, dtype=DTYPE), tau)
        with self.assertRaises(ValueError):
            self.spline.evaluate(self.coeffs, tau, order=-1)
        with self.assertRaises(TypeError):
            self.spline.evaluate(self.coeffs, tau, order=1.0)  # type: ignore[arg-type]


class TestOrientation(unittest.TestCase):
    def setUp(self):
        self.spline = Spline6D()

    def test_single_axis_angular_velocity(self):
        # 纯 yaw: psi = 0.5 tau
        coeffs = zero_coeffs()
        coeffs[0, 5, 1] = 0.5
        omega = self.spline.body_angular_velocity(coeffs, 0.7)[0, 0]
        self.assertTrue(
            torch.allclose(omega, torch.tensor([0.0, 0.0, 0.5], dtype=DTYPE))
        )

        # 纯 pitch: theta = 0.3 tau
        coeffs = zero_coeffs()
        coeffs[0, 4, 1] = 0.3
        omega = self.spline.body_angular_velocity(coeffs, 0.7)[0, 0]
        self.assertTrue(
            torch.allclose(omega, torch.tensor([0.0, 0.3, 0.0], dtype=DTYPE))
        )

        # 纯 roll: phi = 0.2 tau
        coeffs = zero_coeffs()
        coeffs[0, 3, 1] = 0.2
        omega = self.spline.body_angular_velocity(coeffs, 0.7)[0, 0]
        self.assertTrue(
            torch.allclose(omega, torch.tensor([0.2, 0.0, 0.0], dtype=DTYPE))
        )

    def test_combined_against_rotation_derivative(self):
        # phi = 0.2 t, theta = 0.7 t^2, psi = -0.4 t
        coeffs = zero_coeffs()
        coeffs[0, 3, 1] = 0.2
        coeffs[0, 4, 2] = 0.7
        coeffs[0, 5, 1] = -0.4

        tau = 0.3
        omega = self.spline.body_angular_velocity(coeffs, tau)[0, 0]

        h = 1e-6
        r_plus = self.spline.rotation_matrix(coeffs, tau + h)[0, 0]
        r_minus = self.spline.rotation_matrix(coeffs, tau - h)[0, 0]
        r = self.spline.rotation_matrix(coeffs, tau)[0, 0]
        r_dot = (r_plus - r_minus) / (2 * h)
        omega_hat = r.transpose(-1, -2) @ r_dot
        vee = torch.stack(
            (omega_hat[2, 1], omega_hat[0, 2], omega_hat[1, 0])
        )
        self.assertTrue(torch.allclose(omega, vee, atol=1e-6))

    def test_rotation_matrix_is_orthonormal(self):
        coeffs = zero_coeffs()
        coeffs[0, 3, 1] = 0.2
        coeffs[0, 4, 2] = 0.7
        coeffs[0, 5, 1] = -0.4
        r = self.spline.rotation_matrix(coeffs, 0.4)[0, 0]
        eye = torch.eye(3, dtype=DTYPE)
        self.assertTrue(torch.allclose(r.T @ r, eye, atol=1e-12))
        self.assertAlmostEqual(float(torch.linalg.det(r)), 1.0, places=12)

    def test_known_rotation(self):
        # 只有 yaw = pi/2: 机身 x 轴应指向世界 y 轴
        coeffs = zero_coeffs()
        coeffs[0, 5, 0] = math.pi / 2
        r = self.spline.rotation_matrix(coeffs, 0.0)[0, 0]
        expected = torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=DTYPE
        )
        self.assertTrue(torch.allclose(r, expected, atol=1e-12))

    def test_euler_rate_mapping_shape_validation(self):
        with self.assertRaises(ValueError):
            Spline6D.body_omega_from_euler_rates(
                torch.zeros(2, dtype=DTYPE), torch.zeros(3, dtype=DTYPE)
            )
        with self.assertRaises(ValueError):
            Spline6D.body_omega_from_euler_rates(
                torch.zeros(1, 3, dtype=DTYPE), torch.zeros(2, 3, dtype=DTYPE)
            )


class TestJunctions(unittest.TestCase):
    def setUp(self):
        self.spline = Spline6D()
        self.T = 0.5
        # 相位 0: p = 1 + 2t + 0.5t^2; 相位 1 起点与速度严格接上
        self.coeffs = zero_coeffs(num_phases=2)
        for d in range(6):
            self.coeffs[0, d, 0] = 1.0
            self.coeffs[0, d, 1] = 2.0
            self.coeffs[0, d, 2] = 0.5
            p_end = 1.0 + 2.0 * self.T + 0.5 * self.T**2
            v_end = 2.0 + 1.0 * self.T
            self.coeffs[1, d, 0] = p_end
            self.coeffs[1, d, 1] = v_end
            self.coeffs[1, d, 2] = -0.3

    def test_continuous_junction(self):
        for order in (0, 1):
            residual = self.spline.junction_residual(
                self.coeffs, [self.T, 0.4], order
            )
            self.assertEqual(tuple(residual.shape), (1, 6))
            self.assertTrue(
                torch.allclose(residual, torch.zeros_like(residual), atol=1e-12),
                order,
            )

    def test_discontinuous_junction(self):
        self.coeffs[1, 0, 0] += 0.1  # x 位置跳变
        residual = self.spline.junction_residual(self.coeffs, [self.T, 0.4], 0)
        self.assertAlmostEqual(float(residual[0, 0]), -0.1, places=12)
        self.assertTrue(
            torch.allclose(residual[0, 1:], torch.zeros(5, dtype=DTYPE), atol=1e-12)
        )

    def test_batched_junction(self):
        coeffs = self.coeffs.expand(2, 2, 6, 5).clone()
        residual = self.spline.junction_residual(coeffs, [self.T, 0.4], 0)
        self.assertEqual(tuple(residual.shape), (2, 1, 6))

    def test_single_phase(self):
        coeffs = zero_coeffs(num_phases=1)
        residual = self.spline.junction_residual(coeffs, [0.5], 0)
        self.assertEqual(tuple(residual.shape), (0, 6))

    def test_bad_durations(self):
        with self.assertRaises(ValueError):
            self.spline.junction_residual(self.coeffs, [0.5], 0)
        with self.assertRaises(ValueError):
            self.spline.junction_residual(self.coeffs, [0.0, 0.4], 0)
        with self.assertRaises(ValueError):
            self.spline.junction_residual(self.coeffs, [0.5, float("nan")], 0)
        with self.assertRaises(ValueError):
            self.spline.junction_residual(zero_coeffs(num_phases=2), [0.5], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
