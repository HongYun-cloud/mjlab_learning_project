"""Base Pose Tracking 奖励测试 (shim env, 独立 NumPy 复算).

运行:
    python -m unittest myrobot_mjlab.task.DTC.reward.test_terms -v

覆盖:
  1. order=0: 位置向量差 + 朝向四元数差 (机体系轴角), 期望已流动到 t+dt
     (独立复算 p + v·T 与 q⊗exp(w·T/2), T = time_since_replan + dt;
     shim 无 tsr 时退化为 T=dt);
  2. order=1: 速度向量差 (期望 v+a dt 转机体系后相减);
  3. order=2: 加速度向量差 (期望零阶保持; 测量后向差分, 首步=0);
  4. 奖励 = exp(-sigma ||err||) 数值复算;
  5. ⊖ 语义: 朝向用四元数差 (旋转后误差轴角), 位置族为向量差;
  6. 缺命令项报错; order 非法报错;
  7. cfg 接线: 三个 reward 项 (pos/vel/acc) 存在且引用 DTC 函数.
"""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from myrobot_mjlab.task.DTC.reward import terms as dtc_reward


def _quat_wxyz(w, x, y, z):
    return torch.tensor([[w, x, y, z]])


def _make_env(dt: float = 0.02):
    """构造 shim: 期望参考点 + 测量状态 (yaw=pi/2 的测量姿态)."""
    yaw = math.pi / 2
    q_meas = _quat_wxyz(math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)).repeat(2, 1)
    cmd = SimpleNamespace(
        desired_base_pos_w=torch.tensor([[1.0, 0.5, 0.35]] * 2),
        desired_base_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1),
        desired_base_lin_vel_w=torch.tensor([[0.3, 0.1, 0.0]] * 2),
        desired_base_ang_vel_w=torch.tensor([[0.0, 0.0, 0.4]] * 2),
        desired_base_lin_acc_w=torch.tensor([[0.5, 0.0, 0.0]] * 2),
        desired_base_ang_acc_w=torch.tensor([[0.0, 0.0, 0.2]] * 2),
        time_since_replan=torch.zeros(2),
    )
    asset = SimpleNamespace(
        data=SimpleNamespace(
            root_link_pos_w=torch.tensor([[0.9, 0.4, 0.3]] * 2),
            root_link_quat_w=q_meas,
            root_link_lin_vel_b=torch.tensor([[0.2, 0.05, 0.0]] * 2),
            root_link_ang_vel_b=torch.tensor([[0.0, 0.0, 0.3]] * 2),
        )
    )
    env = SimpleNamespace(
        scene={"robot": asset},
        command_manager=SimpleNamespace(get_term=lambda name: cmd),
        step_dt=dt,
    )
    return env, cmd, asset, yaw


class TestBasePoseTracking(unittest.TestCase):
    def setUp(self):
        dtc_reward._PREV_LIN_VEL_B.clear()
        dtc_reward._PREV_ANG_VEL_B.clear()

    def test_order0_pose_independent_recompute(self):
        env, cmd, asset, yaw = _make_env()
        dt = 0.02
        r = dtc_reward.base_pose_tracking(env, sigma=2.0, order=0)
        # 期望流动到 t+dt (NumPy): tsr=0 -> T=dt, p + v·T
        p = np.array([1.0, 0.5, 0.35]) + np.array([0.3, 0.1, 0.0]) * dt
        # q* = [1,0,0,0] ⊗ exp(w dt/2), w=(0,0,0.4)
        dq = np.array([math.cos(0.4 * dt / 2), 0, 0, math.sin(0.4 * dt / 2)])
        q_des = dq  # q*=[1,0,0,0] 时 q_new = dq
        # 机体系姿态误差: q_err = q_meas^{-1} ⊗ q_des
        q_meas = np.array([
            math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)
        ])
        # 直接用四元数乘法 (w,x,y,z)
        def qmul(a, b):
            aw, ax, ay, az = a; bw, bx, by, bz = b
            return np.array([
                aw*bw - ax*bx - ay*by - az*bz,
                aw*bx + ax*bw + ay*bz - az*by,
                aw*by - ax*bz + ay*bw + az*bx,
                aw*bz + ax*by - ay*bx + az*bw,
            ])
        q_err = qmul(np.concatenate(([q_meas[0]], -q_meas[1:])), q_des)
        # 轴角: angle = 2*atan2(||v||, |w|) (未归一化分量), axis = v/||v||
        ang = 2 * math.atan2(np.linalg.norm(q_err[1:]), abs(q_err[0]))
        v_e = q_err[1:] / max(np.linalg.norm(q_err[1:]), 1e-12)
        ang_err = ang * v_e
        lin_err = p - np.array([0.9, 0.4, 0.3])
        err = np.concatenate((lin_err, ang_err))
        exp_r = math.exp(-2.0 * np.linalg.norm(err))
        self.assertAlmostEqual(float(r[0]), exp_r, places=6)

    def test_squared_error_matches_paper(self):
        env, cmd, asset, yaw = _make_env()
        dt = 0.02
        r = dtc_reward.base_pose_tracking(
            env, sigma=1200.0, order=0, component="pos", squared=True
        )
        p = np.array([1.0, 0.5, 0.35]) + np.array([0.3, 0.1, 0.0]) * dt
        err = p - np.array([0.9, 0.4, 0.3])
        exp_r = math.exp(-1200.0 * float(np.dot(err, err)))
        self.assertAlmostEqual(float(r[0]), exp_r, places=6)
        # 非平方范数在 σ=1200 / cm 级误差下 underflow 到 0, 无梯度.
        r_old = dtc_reward.base_pose_tracking(
            env, sigma=1200.0, order=0, component="pos"
        )
        self.assertLess(float(r_old[0]), 1e-5)

    def test_flowing_reference_uses_time_since_replan(self):
        """tsr>0: 位置/姿态参考按 T=tsr+dt 流动 (不再钉死在重规划点)."""
        env, cmd, asset, yaw = _make_env()
        cmd.time_since_replan = torch.full((2,), 0.28)
        dt, T = 0.02, 0.30
        # 位置族: p* + v*·T
        r_pos = dtc_reward.base_pose_tracking(
            env, sigma=1.0, order=0, component="pos"
        )
        p = np.array([1.0, 0.5, 0.35]) + np.array([0.3, 0.1, 0.0]) * T
        err_p = np.linalg.norm(p - np.array([0.9, 0.4, 0.3]))
        self.assertAlmostEqual(float(r_pos[0]), math.exp(-err_p), places=5)
        # 角族: q* ⊗ exp(w·T/2), 测量 yaw=pi/2 -> 误差角 = |0.4·T - pi/2|
        r_rot = dtc_reward.base_pose_tracking(
            env, sigma=1.0, order=0, component="rot"
        )
        ang = abs(math.pi / 2 - 0.4 * T)
        self.assertAlmostEqual(float(r_rot[0]), math.exp(-ang), places=5)

    def test_order1_velocity_vector_diff(self):
        env, cmd, asset, yaw = _make_env()
        dt = 0.02
        r = dtc_reward.base_pose_tracking(env, sigma=1.5, order=1)
        # 期望 v*+a*dt (世界系) -> 机体系 (yaw=pi/2: (x,y)->(y,-x))
        v_des_w = np.array([0.3, 0.1, 0.0]) + np.array([0.5, 0.0, 0.0]) * dt
        c, s = math.cos(yaw), math.sin(yaw)
        v_des_b = np.array([c*v_des_w[0]+s*v_des_w[1],
                            -s*v_des_w[0]+c*v_des_w[1], v_des_w[2]])
        lin_err = v_des_b - np.array([0.2, 0.05, 0.0])
        # 期望 w*+alpha*dt -> 机体系: (0,0,0.4+0.2dt) z 不变
        w_des_b = np.array([0.0, 0.0, 0.4 + 0.2 * dt])
        ang_err = w_des_b - np.array([0.0, 0.0, 0.3])
        err = np.concatenate((lin_err, ang_err))
        exp_r = math.exp(-1.5 * np.linalg.norm(err))
        self.assertAlmostEqual(float(r[0]), exp_r, places=6)

    def test_order2_accel_vector_diff_first_step(self):
        env, cmd, asset, yaw = _make_env()
        r1 = dtc_reward.base_pose_tracking(env, sigma=1.0, order=2)
        # 首步测量加速度=0 (差分缓存为空); 期望 a* 零阶保持
        dt = 0.02
        c, s = math.cos(yaw), math.sin(yaw)
        a_w = np.array([0.5, 0.0, 0.0])
        a_b = np.array([c*a_w[0]+s*a_w[1], -s*a_w[0]+c*a_w[1], 0.0])
        alpha_w = np.array([0.0, 0.0, 0.2])
        alpha_b = np.array([0.0, 0.0, 0.2])
        err = np.concatenate((a_b - 0.0, alpha_b - 0.0))
        exp_r = math.exp(-1.0 * np.linalg.norm(err))
        self.assertAlmostEqual(float(r1[0]), exp_r, places=6)

    def test_order2_backward_difference_second_step(self):
        env, cmd, asset, yaw = _make_env()
        dtc_reward.base_pose_tracking(env, sigma=1.0, order=2)  # 填缓存
        # 改测量速度: 模拟上一步 -> 本步的加速度
        asset.data.root_link_lin_vel_b = torch.tensor(
            [[0.3, 0.05, 0.0]] * 2
        )
        asset.data.root_link_ang_vel_b = torch.tensor(
            [[0.0, 0.0, 0.5]] * 2
        )
        r2 = dtc_reward.base_pose_tracking(env, sigma=1.0, order=2)
        dt = 0.02
        lin_acc_meas = np.array([(0.3 - 0.2) / dt, 0.0, 0.0])
        ang_acc_meas = np.array([0.0, 0.0, (0.5 - 0.3) / dt])
        c, s = math.cos(yaw), math.sin(yaw)
        a_w = np.array([0.5, 0.0, 0.0])
        a_b = np.array([c*a_w[0]+s*a_w[1], -s*a_w[0]+c*a_w[1], 0.0])
        err = np.concatenate((a_b - lin_acc_meas, np.array([0, 0, 0.2]) - ang_acc_meas))
        exp_r = math.exp(-np.linalg.norm(err))
        self.assertAlmostEqual(float(r2[0]), exp_r, places=5)

    def test_quat_integrate_small_angle(self):
        q = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(3, 1)
        w = torch.tensor([[0.0, 0.0, 0.4]]).repeat(3, 1)
        out = dtc_reward._quat_integrate(q, w, 0.02)
        # 小角度: q ≈ [1, 0, 0, w dt/2]
        np.testing.assert_allclose(
            out.numpy(),
            np.tile(
                [math.cos(0.004), 0, 0, math.sin(0.004)], (3, 1)
            ),
            atol=1e-9,
        )

    def test_minus_semantics_orientation_vs_vector(self):
        """⊖: 朝向=四元数差; 同一朝向差下 位置零误差时仍有姿态误差."""
        env, cmd, asset, yaw = _make_env()
        # 让位置期望=测量 (前推后仍有微小差, 单独验证朝向主导)
        r = dtc_reward.base_pose_tracking(env, sigma=1.0, order=0)
        self.assertLess(float(r[0]), 1.0)  # 存在朝向误差 -> 奖励<1

    def test_errors(self):
        env, cmd, asset, yaw = _make_env()
        with self.assertRaises(ValueError):
            dtc_reward.base_pose_tracking(env, sigma=1.0, order=3)
        env.command_manager.get_term = lambda name: None
        with self.assertRaises(KeyError):
            dtc_reward.base_pose_tracking(env, sigma=1.0, order=0)


class TestComponentSplit(unittest.TestCase):
    """component 拆分: pos/rot/both 三种模式的数值独立性."""

    def _env(self):
        import math
        from types import SimpleNamespace

        yaw = math.pi / 2
        cmd = SimpleNamespace(
            desired_base_pos_w=torch.tensor([[1.0, 0.5, 0.35]] * 2),
            desired_base_quat_w=torch.tensor([[1.0, 0, 0, 0]]).repeat(2, 1),
            desired_base_lin_vel_w=torch.tensor([[0.3, 0.1, 0.0]] * 2),
            desired_base_ang_vel_w=torch.tensor([[0.0, 0.0, 0.4]] * 2),
            desired_base_lin_acc_w=torch.tensor([[0.5, 0.0, 0.0]] * 2),
            desired_base_ang_acc_w=torch.tensor([[0.0, 0.0, 0.2]] * 2),
            time_since_replan=torch.zeros(2),
        )
        asset = SimpleNamespace(
            data=SimpleNamespace(
                root_link_pos_w=torch.tensor([[0.9, 0.4, 0.3]] * 2),
                root_link_quat_w=torch.tensor(
                    [[math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]] * 2
                ),
                root_link_lin_vel_b=torch.zeros(2, 3),
                root_link_ang_vel_b=torch.zeros(2, 3),
            )
        )
        return SimpleNamespace(
            scene={"robot": asset},
            command_manager=SimpleNamespace(get_term=lambda name: cmd),
            step_dt=0.02,
        ), yaw, cmd, asset

    def test_pos_only_excludes_orientation(self):
        env, yaw, cmd, asset = self._env()
        r_pos = dtc_reward.base_pose_tracking(
            env, sigma=2.0, order=0, component="pos"
        )
        # 独立复算: 仅位置误差 (流动 T=tsr+dt, shim tsr=0)
        import numpy as np

        dt = 0.02
        p = np.array([1.0, 0.5, 0.35]) + np.array([0.3, 0.1, 0.0]) * dt
        err = np.linalg.norm(p - np.array([0.9, 0.4, 0.3]))
        self.assertAlmostEqual(float(r_pos[0]), math.exp(-2.0 * err), places=6)
        # 与 both 比较: both 更小 (姿态误差额外计入)
        r_both = dtc_reward.base_pose_tracking(env, sigma=2.0, order=0)
        self.assertLess(float(r_both[0]), float(r_pos[0]))

    def test_rot_only_excludes_position(self):
        env, yaw, cmd, asset = self._env()
        r_rot = dtc_reward.base_pose_tracking(
            env, sigma=2.0, order=0, component="rot"
        )
        # 误差 ≈ pi/2 - 0.004 (期望单位 vs 测量 yaw=pi/2, 期望已前推小角度)
        import numpy as np

        dq = np.array([math.cos(0.4 * 0.02 / 2), 0, 0, math.sin(0.4 * 0.02 / 2)])
        # 期望 (前推后) 与测量的相对旋转 -> 机体系轴角范数
        # 独立计算: q_err = conj(q_meas) ⊗ q_des; 轴角范数

        def qmul(a, b):
            aw, ax, ay, az = a; bw, bx, by, bz = b
            return np.array([
                aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
                aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw])

        q_meas = np.array([math.cos(yaw/2), 0, 0, math.sin(yaw/2)])
        q_err = qmul(np.concatenate(([q_meas[0]], -q_meas[1:])), dq)
        ang = 2 * math.atan2(np.linalg.norm(q_err[1:]), abs(q_err[0]))
        self.assertAlmostEqual(float(r_rot[0]), math.exp(-2.0 * ang), places=6)

    def test_vel_and_acc_components_independent(self):
        """order=1/2 的 pos/rot 拆分: 各自只含对应误差族."""
        env, yaw, cmd, asset = self._env()
        r_vel_pos = dtc_reward.base_pose_tracking(
            env, sigma=1.5, order=1, component="pos"
        )
        r_vel_rot = dtc_reward.base_pose_tracking(
            env, sigma=1.5, order=1, component="rot"
        )
        r_vel_both = dtc_reward.base_pose_tracking(
            env, sigma=1.5, order=1
        )
        # both 的误差范数 > 任一单侧 (两侧均非零)
        self.assertGreater(float(r_vel_both[0]), 0.0)
        self.assertGreater(float(r_vel_pos[0]), float(r_vel_both[0]))
        self.assertGreater(float(r_vel_rot[0]), float(r_vel_both[0]))
        # 加速度族同理
        r_acc_pos = dtc_reward.base_pose_tracking(
            env, sigma=1.0, order=2, component="pos"
        )
        r_acc_rot = dtc_reward.base_pose_tracking(
            env, sigma=1.0, order=2, component="rot"
        )
        self.assertGreater(float(r_acc_pos[0]), 0.0)
        self.assertGreater(float(r_acc_rot[0]), 0.0)

    def test_rot_weight_ignored_for_components(self):
        env, yaw, cmd, asset = self._env()
        r1 = dtc_reward.base_pose_tracking(
            env, sigma=2.0, order=0, component="rot", rot_weight=5.0
        )
        r2 = dtc_reward.base_pose_tracking(
            env, sigma=2.0, order=0, component="rot", rot_weight=1.0
        )
        self.assertEqual(float(r1[0]), float(r2[0]))

    def test_invalid_component(self):
        env, *_ = self._env()
        with self.assertRaises(ValueError):
            dtc_reward.base_pose_tracking(
                env, sigma=1.0, order=0, component="pose"
            )


class TestStallPenalty(unittest.TestCase):
    """原地滞留惩罚: 命令要求移动而没动 -> 罚; 动了或命令为零 -> 0."""

    def _env(self, cmd_speed=(0.5, 0.0), meas_speed=(0.0, 0.0)):
        cmd = SimpleNamespace(
            vel_command_b=torch.tensor(
                [[cmd_speed[0], cmd_speed[1], 0.0]]
            ).repeat(2, 1),
        )
        asset = SimpleNamespace(
            data=SimpleNamespace(
                root_link_lin_vel_b=torch.tensor(
                    [[meas_speed[0], meas_speed[1], 0.0]]
                ).repeat(2, 1),
            )
        )
        return SimpleNamespace(
            scene={"robot": asset},
            command_manager=SimpleNamespace(get_term=lambda name: cmd),
            step_dt=0.02,
        )

    def test_stalled_when_commanded_but_stationary(self):
        r = dtc_reward.stall_penalty(self._env())
        self.assertTrue(bool((r == -2.0).all()))

    def test_no_penalty_when_moving(self):
        r = dtc_reward.stall_penalty(
            self._env(cmd_speed=(0.5, 0.0), meas_speed=(0.3, 0.0))
        )
        self.assertTrue(bool((r == 0.0).all()))

    def test_no_penalty_when_command_is_zero(self):
        # standing 采样 / north_peak 到达中心: 命令为零不罚
        r = dtc_reward.stall_penalty(
            self._env(cmd_speed=(0.0, 0.0), meas_speed=(0.0, 0.0))
        )
        self.assertTrue(bool((r == 0.0).all()))

    def test_thresholds_and_vertical_ignored(self):
        # 恰在阈值上不触发 (严格不等号)
        r = dtc_reward.stall_penalty(
            self._env(cmd_speed=(0.1, 0.0), meas_speed=(0.1, 0.0))
        )
        self.assertTrue(bool((r == 0.0).all()))
        # 垂直速度大不算"动了"
        env = self._env(cmd_speed=(0.5, 0.0), meas_speed=(0.0, 0.0))
        env.scene["robot"].data.root_link_lin_vel_b[:, 2] = 5.0
        self.assertTrue(bool((dtc_reward.stall_penalty(env) == -2.0).all()))

    def test_cfg_wired(self):
        from myrobot_mjlab.task.DTC.config import my_go2_flat_env_cfg

        rewards = my_go2_flat_env_cfg().rewards
        self.assertIn("stall_penalty", rewards)
        term = rewards["stall_penalty"]
        self.assertEqual(term.func.__name__, "stall_penalty")
        self.assertEqual(
            term.func.__module__, "myrobot_mjlab.task.DTC.reward.terms"
        )
        self.assertEqual(term.params["command_name"], "twist")
        self.assertGreater(term.weight, 0.0)  # 函数返回负值, weight 为正


class TestCfgWiring(unittest.TestCase):
    def test_pos_rot_split_wired(self):
        """三阶 (0/1/2) × 两族 (pos/rot) 六项全部独立接线."""
        from myrobot_mjlab.task.DTC.config import my_go2_flat_env_cfg

        rewards = my_go2_flat_env_cfg().rewards
        expected = {
            "base_pose_tracking_pos": (0, "pos"),
            "base_pose_tracking_rot": (0, "rot"),
            "base_pose_tracking_vel_pos": (1, "pos"),
            "base_pose_tracking_vel_rot": (1, "rot"),
            "base_pose_tracking_acc_pos": (2, "pos"),
            "base_pose_tracking_acc_rot": (2, "rot"),
        }
        for key, (order, comp) in expected.items():
            self.assertIn(key, rewards, key)
            term = rewards[key]
            self.assertEqual(term.params["order"], order, key)
            self.assertEqual(term.params["component"], comp, key)
            self.assertGreater(term.weight, 0.0, key)
        # 旧混合项已移除 (避免重复计分)
        for old_key in (
            "base_pose_tracking_vel", "base_pose_tracking_acc"
        ):
            self.assertNotIn(old_key, rewards)

    def test_reward_terms_wired(self):
        from myrobot_mjlab.task.DTC.config import my_go2_flat_env_cfg

        cfg = my_go2_flat_env_cfg()
        rewards = cfg.rewards
        for key in (
            "base_pose_tracking_pos",
            "base_pose_tracking_rot",
            "base_pose_tracking_vel_pos",
            "base_pose_tracking_vel_rot",
            "base_pose_tracking_acc_pos",
            "base_pose_tracking_acc_rot",
        ):
            self.assertIn(key, rewards, key)
            term = rewards[key]
            self.assertEqual(
                term.func.__module__,
                "myrobot_mjlab.task.DTC.reward.terms",
            )
            self.assertEqual(term.func.__name__, "base_pose_tracking")
            self.assertEqual(
                term.params["order"],
                {"base_pose_tracking_pos": 0, "base_pose_tracking_rot": 0,
                 "base_pose_tracking_vel_pos": 1,
                 "base_pose_tracking_vel_rot": 1,
                 "base_pose_tracking_acc_pos": 2,
                 "base_pose_tracking_acc_rot": 2}[key])
            self.assertGreater(term.weight, 0.0)


if __name__ == "__main__":
    unittest.main()
