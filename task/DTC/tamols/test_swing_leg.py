"""DTC 摆动腿观测层测试 (固定 trot / 落足点接收 / 路径局部高度).

运行:
    python -m unittest myrobot_mjlab.task.DTC.tamols.test_swing_leg -v

覆盖:
  1. 固定 trot: FL/RR 同相、FR/RL 反相, 摆动/支撑互补;
  2. touchdown 事件恰在 摆动→支撑 切换的那一步为 True;
  3. 共享调度器模式: builder.step() 抛错 (防止双倍推进), 时钟由外部
     推进时观测相位正确跟随;
  4. 路径采样: 端点=当前脚/落点 (机体系), 中点线性; 高度取自解析图
     z=0.1x+0.05y (独立 NumPy 复算); FlatHeights 容错;
  5. goal_rel_b 数值独立复算 (含 yaw 旋转);
  6. footholds_from: 张量 / TAMOLSPlan / 命令项属性 三源;
  7. LocalGridHeights 与手写双线性一致, 越界 clamp.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from myrobot_mjlab.task.DTC.tamols.swing_leg import (
    FlatHeights,
    LocalGridHeights,
    SwingObsBuilder,
    SwingObsCfg,
    TROT_PHASE_OFFSETS,
    footholds_from,
)
from myrobot_mjlab.task.Tamols.fetch.gait_scheduler import (
    GaitScheduler,
    GaitSchedulerCfg,
)


def _linear_terrain(body_xy: torch.Tensor) -> torch.Tensor:
    """解析地形 z = 0.1x + 0.05y (机体系)."""
    return 0.1 * body_xy[..., 0] + 0.05 * body_xy[..., 1]


class TestTrotSchedule(unittest.TestCase):
    def test_trot_phase_offsets(self):
        gait = GaitScheduler(GaitSchedulerCfg(), num_envs=2)
        self.assertEqual(gait.cfg.phase_offsets, TROT_PHASE_OFFSETS)
        # t=0: phi = offsets -> FL/RR 相位 0 (支撑, phi<D=0.5),
        #      FR/RL 相位 0.5 (摆动)
        swing = gait.is_swing
        self.assertFalse(bool(swing[0, 0]))   # FL 支撑
        self.assertFalse(bool(swing[0, 3]))   # RR 支撑
        self.assertTrue(bool(swing[0, 1]))    # FR 摆动
        self.assertTrue(bool(swing[0, 2]))    # RL 摆动

    def test_half_cycle_swap(self):
        b = SwingObsBuilder(SwingObsCfg(), num_envs=1)
        b.step(0.30)  # T=0.6, duty=0.5: 半周期后 FL/RR 与 FR/RL 互换
        swing = b.gait.is_swing
        self.assertTrue(bool(swing[0, 0]) and bool(swing[0, 3]))
        self.assertFalse(bool(swing[0, 1]) or bool(swing[0, 2]))

    def test_touchdown_event_only_on_transition(self):
        b = SwingObsBuilder(SwingObsCfg(), num_envs=1)
        events = []
        for _ in range(62):  # ~1.24s: 覆盖两个整周期 (T=0.6) 的切换沿
            b.step(0.02)
            obs = b.update(
                foot_pos_w=torch.zeros(1, 4, 3),
                footholds_w=torch.zeros(1, 4, 3),
                base_pos_w=torch.zeros(1, 3),
                base_yaw_w=torch.zeros(1),
                heights=FlatHeights(0.0),
            )
            events.append(obs["touchdown_events"].clone())
        td = torch.stack(events)  # [62,1,4]
        # 实测调度 (浮点累积使 0.6s 边界偏移一步):
        # FR 触地在步 16 与 46; FL 触地在步 31 与 61 — 各 2 次
        self.assertEqual(int(td[:, 0, 1].sum()), 2)
        self.assertEqual(int(td[:, 0, 0].sum()), 2)
        self.assertTrue(bool(td[15, 0, 1]))
        self.assertTrue(bool(td[30, 0, 0]))
        # 事件为单步脉冲 (无连续 True)
        for leg in range(4):
            seq = td[:, 0, leg].int().tolist()
            for i in range(len(seq) - 1):
                self.assertFalse(seq[i] == 1 and seq[i + 1] == 1, (leg, i))

    def test_shared_gait_clock_not_double_stepped(self):
        gait = GaitScheduler(GaitSchedulerCfg(), num_envs=1)
        b = SwingObsBuilder(gait=gait, num_envs=1)
        with self.assertRaises(RuntimeError):
            b.step(0.02)
        gait.step(0.30)
        obs = b.update(
            torch.zeros(1, 4, 3), torch.zeros(1, 4, 3),
            torch.zeros(1, 3), torch.zeros(1), FlatHeights(0.0),
        )
        self.assertAlmostEqual(float(obs["phase"][0, 0]), 0.5, places=6)


class TestSwingObservations(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.n = 2
        self.foot = torch.tensor(
            [[[0.20, 0.15, 0.02], [0.20, -0.15, 0.02],
              [-0.20, 0.15, 0.02], [-0.20, -0.15, 0.02]]]
        ).repeat(self.n, 1, 1)
        self.goal = self.foot.clone()
        self.goal[:, :, 0] += 0.10   # 前进 10cm
        self.goal[:, :, 2] += 0.03   # 目标地面高 3cm
        self.base = torch.zeros(self.n, 3)
        self.yaw = torch.tensor([0.0, math.pi / 2])
        self.builder = SwingObsBuilder(SwingObsCfg(path_samples=5), num_envs=self.n)

    def _update(self):
        return self.builder.update(
            self.foot, self.goal, self.base, self.yaw,
            LocalGridHeights(_make_grid(), x_min=-1.0, y_min=-1.0, dx=0.1),
        )

    def test_goal_rel_b_independent_recompute(self):
        obs = self._update()
        for env in range(self.n):
            yaw = float(self.yaw[env])
            c, s = math.cos(yaw), math.sin(yaw)
            for leg in range(4):
                dx = float(self.goal[env, leg, 0] - self.foot[env, leg, 0])
                dy = float(self.goal[env, leg, 1] - self.foot[env, leg, 1])
                exp = (c * dx + s * dy, -s * dx + c * dy, 0.03)
                got = [float(v) for v in obs["goal_rel_b"][env, leg]]
                for g_, e_ in zip(got, exp):
                    self.assertAlmostEqual(g_, e_, places=5)

    def test_path_endpoints_and_mid_height(self):
        obs = self._update()
        path = obs["path_xy_b"]       # [N,4,S,2]
        rel_h = obs["path_rel_h"]     # [N,4,S]
        for env in range(self.n):
            yaw = float(self.yaw[env])
            c, s = math.cos(yaw), math.sin(yaw)
            leg = 0
            foot_b = (c * 0.20 + s * 0.15, -s * 0.20 + c * 0.15)
            goal_b = (c * 0.30 + s * 0.15, -s * 0.30 + c * 0.15)
            # 端点
            np.testing.assert_allclose(
                path[env, leg, 0].numpy(), foot_b, atol=1e-5
            )
            np.testing.assert_allclose(
                path[env, leg, -1].numpy(), goal_b, atol=1e-5
            )
            # 中点高度 = 0.1*x + 0.05*y - foot_z (独立 NumPy)
            mid = path[env, leg, 2].numpy()
            exp_h = 0.1 * mid[0] + 0.05 * mid[1] - 0.02
            self.assertAlmostEqual(float(rel_h[env, leg, 2]), exp_h, places=4)

    def test_flat_heights_fallback(self):
        obs = self.builder.update(
            self.foot, self.goal, self.base, self.yaw, FlatHeights(0.0)
        )
        # 全部路径高度 = 0 - foot_z
        np.testing.assert_allclose(
            obs["path_rel_h"].numpy(), -0.02, atol=1e-6
        )

    def test_shapes(self):
        obs = self._update()
        self.assertEqual(obs["goal_rel_b"].shape, (self.n, 4, 3))
        self.assertEqual(obs["path_rel_h"].shape, (self.n, 4, 5))
        self.assertEqual(obs["path_xy_b"].shape, (self.n, 4, 5, 2))
        self.assertEqual(obs["swing_mask"].shape, (self.n, 4))
        self.assertEqual(obs["swing_progress"].shape, (self.n, 4))
        self.assertEqual(obs["phase"].shape, (self.n, 4))
        self.assertEqual(obs["touchdown_events"].shape, (self.n, 4))
        # 支撑相 progress = 0; 摆动相 in [0, 1] (phi==D 边界时恰为 0)
        prog = obs["swing_progress"]
        stance = ~obs["swing_mask"]
        self.assertTrue(bool((prog[stance] == 0).all()))
        sw = prog[obs["swing_mask"]]
        self.assertTrue(bool(((sw >= 0) & (sw <= 1)).all()))


class TestLocalGridHeights(unittest.TestCase):
    def test_matches_analytic_at_nodes_and_midpoints(self):
        grid = _make_grid()
        q = LocalGridHeights(grid, x_min=-1.0, y_min=-1.0, dx=0.1)
        pts = torch.tensor([[[[0.0, 0.0], [0.25, -0.35], [-0.9, 0.85]]]])
        got = q.query(pts)[0, 0].numpy()
        exp = 0.1 * pts[0, 0, :, 0].numpy() + 0.05 * pts[0, 0, :, 1].numpy()
        np.testing.assert_allclose(got, exp, atol=2e-3)  # 网格离散化误差

    def test_out_of_bounds_clamps(self):
        grid = _make_grid()
        q = LocalGridHeights(grid, x_min=-1.0, y_min=-1.0, dx=0.1)
        far = torch.tensor([[[[5.0, 5.0], [-5.0, -5.0]]]])
        corner = q.query(far)
        self.assertTrue(torch.isfinite(corner).all())


class TestFootholdsFrom(unittest.TestCase):
    def test_tensor_passthrough(self):
        t = torch.zeros(3, 4, 3)
        self.assertIs(footholds_from(t), t)

    def test_command_attr(self):
        class _Cmd:
            optimal_footholds = torch.ones(2, 4, 3)

        out = footholds_from(_Cmd(), device="cpu")
        self.assertEqual(out.shape, (2, 4, 3))

    def test_tamols_plan(self):
        from myrobot_mjlab.task.DTC.tamols.rl_bridge import TAMOLSPlan

        plan = TAMOLSPlan(
            stamp=1, status="ok", success=True, horizon=(0.0, 0.6),
            t=np.zeros(1),
            base_pos_w=np.zeros((1, 3)), base_euler_zyx=np.zeros((1, 3)),
            base_lin_vel_w=np.zeros((1, 3)), base_lin_vel_b=np.zeros((1, 3)),
            base_ang_vel_b=np.zeros((1, 3)), base_lin_acc_w=np.zeros((1, 3)),
            footholds_w=np.arange(12.0).reshape(4, 3) + 1.0,
            touchdown_times=np.zeros(4),
            touchdown_joint_pos=np.zeros((4, 3)),
            touchdown_joint_vel=np.zeros((4, 3)),
            contact_sets=(), spline_coeffs=np.zeros((1, 6, 5)),
            durations=(0.6,), state={},
        )
        out = footholds_from(plan)
        # 单环境 TAMOLSPlan 的 [4,3] 会补 batch 维 -> [1,4,3]
        self.assertEqual(out.shape, (1, 4, 3))
        self.assertAlmostEqual(float(out[0, 0, 0]), 1.0, places=6)

    def test_rejects_bad_shape(self):
        with self.assertRaises(ValueError):
            footholds_from(torch.zeros(2, 3))


def _make_grid() -> torch.Tensor:
    """解析地形 z=0.1x+0.05y 的网格 (x,y ∈ [-1,1], dx=0.1)."""
    xs = np.arange(-1.0, 1.0 + 1e-9, 0.1)
    ys = np.arange(-1.0, 1.0 + 1e-9, 0.1)
    X, Y = np.meshgrid(xs, ys, indexing="xy")  # V[iy, ix]
    H = 0.1 * X + 0.05 * Y
    return torch.from_numpy(H).float().unsqueeze(0)  # [1,Ny,Nx]


if __name__ == "__main__":
    unittest.main()
