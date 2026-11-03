"""NLP 规划适配器的小工具单测 (不启动 MuJoCo/环境)."""

from __future__ import annotations

import math
import unittest

import numpy as np

from myrobot_mjlab.task.DTC.tamols.nlp_planner import (
    euler_rates_from_body_omega,
    euler_zyx_to_quat_wxyz,
    quat_wxyz_to_euler_zyx,
)
from myrobot_mjlab.task.DTC.tamols.v_b_objectives import body_omega_zyx


class TestEulerQuat(unittest.TestCase):
    def test_quat_roundtrip(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            euler = rng.uniform(-0.6, 0.6, size=3)
            q = euler_zyx_to_quat_wxyz(euler)
            back = quat_wxyz_to_euler_zyx(q)
            np.testing.assert_allclose(back, euler, atol=1e-10)

    def test_quat_unit_norm(self):
        for euler in ([0.0, 0.0, 0.0], [0.2, -0.3, 1.1], [-0.4, 0.5, -2.0]):
            q = euler_zyx_to_quat_wxyz(euler)
            self.assertAlmostEqual(float(np.linalg.norm(q)), 1.0, places=12)


class TestEulerRates(unittest.TestCase):
    def test_inverse_of_body_omega(self):
        rng = np.random.default_rng(1)
        for _ in range(50):
            euler = rng.uniform(-0.5, 0.5, size=3)
            rates = rng.uniform(-1.0, 1.0, size=3)
            omega = np.array(
                [float(v) for v in body_omega_zyx(euler, rates)]
            )
            recovered = euler_rates_from_body_omega(euler, omega)
            np.testing.assert_allclose(recovered, rates, atol=1e-10)

    def test_singular_pitch_is_finite(self):
        euler = [0.1, math.pi / 2, -0.3]
        rates = euler_rates_from_body_omega(euler, [0.1, 0.2, 0.3])
        self.assertTrue(np.isfinite(rates).all())


if __name__ == "__main__":
    unittest.main()
