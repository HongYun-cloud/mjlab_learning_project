"""DTC 观测项测试 (shim 环境, 无需真实 mjlab env).

运行:
    python -m unittest myrobot_mjlab.task.DTC.observation.test_terms -v

覆盖:
  1. 三个 DTC obs 函数的数值 (独立 NumPy 复算, 含 yaw 旋转/默认关节
     参考偏移/接触状态直读);
  2. 缺命令项时报清晰错误;
  3. cfg 接线: 四个变体的 actor terms 含 tamols_footholds /
     desired_joint_pos / desired_contacts; rough/stair/tire 含
     height_scan (terrain_scan 存在), flat 不含 (传感器被移除);
  4. critic 继承 actor 项 (含 TAMOLS 观测).
"""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from myrobot_mjlab.task.DTC.observation import terms as dtc_obs


def _make_env(num_envs: int = 2, yaw: float = math.pi / 2):
    foot = torch.tensor(
        [[[0.30, 0.15, 0.05], [0.30, -0.15, 0.0],
          [-0.10, 0.15, 0.0], [-0.10, -0.15, 0.0]]]
    ).repeat(num_envs, 1, 1)
    quat = torch.tensor(
        [[math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]] * num_envs
    )
    cmd = SimpleNamespace(
        optimal_footholds=foot,
        desired_joint_pos=torch.tensor(
            [[0.0, 0.9, -1.8] * 4]
        ).repeat(num_envs, 1),
        gait_scheduler=SimpleNamespace(
            contact_states=torch.tensor([[1.0, 0.0, 0.0, 1.0]]).repeat(
                num_envs, 1
            )
        ),
    )
    default = torch.tensor([[0.0, 0.8, -1.6] * 4]).repeat(num_envs, 1)
    env = SimpleNamespace(
        scene={
            "robot": SimpleNamespace(
                data=SimpleNamespace(
                    root_link_pos_w=torch.tensor([[0.2, 0.0, 0.3]] * num_envs),
                    root_link_quat_w=quat,
                    default_joint_pos=default,
                )
            )
        },
        command_manager=SimpleNamespace(
            get_term=lambda name: cmd if name == "Tamols_footholds" else None,
            _terms={"Tamols_footholds": cmd},
        ),
    )
    return env, foot, default, cmd


class TestObsFunctions(unittest.TestCase):
    def test_tamols_footholds_relative_body_frame(self):
        env, foot, _, _ = _make_env()
        out = dtc_obs.tamols_footholds(env)
        self.assertEqual(out.shape, (2, 12))
        # 独立复算 (yaw=pi/2): rel = foot - base; 旋转到机体系
        yaw = math.pi / 2
        c, s = math.cos(yaw), math.sin(yaw)
        for leg in range(4):
            dx = float(foot[0, leg, 0]) - 0.2
            dy = float(foot[0, leg, 1]) - 0.0
            dz = float(foot[0, leg, 2]) - 0.3
            exp = (c * dx + s * dy, -s * dx + c * dy, dz)
            got = [float(out[0, leg * 3 + k]) for k in range(3)]
            for g_, e_ in zip(got, exp):
                self.assertAlmostEqual(g_, e_, places=5)

    def test_desired_joint_pos_relative_default(self):
        env, _, default, _ = _make_env()
        out = dtc_obs.tamols_desired_joint_pos(env)
        self.assertEqual(out.shape, (2, 12))
        np.testing.assert_allclose(
            out.numpy(),
            (env.command_manager._terms["Tamols_footholds"]
             .desired_joint_pos - default).numpy(),
            atol=1e-6,
        )
        # 与 joint_pos_rel 同参考: 差值应为 0.1 (0.9-0.8) / -0.2
        self.assertAlmostEqual(float(out[0, 1]), 0.1, places=6)
        self.assertAlmostEqual(float(out[0, 2]), -0.2, places=6)

    def test_desired_contacts_passthrough(self):
        env, _, _, cmd = _make_env()
        out = dtc_obs.tamols_desired_contacts(env)
        self.assertEqual(out.shape, (2, 4))
        np.testing.assert_allclose(
            out.numpy(), cmd.gait_scheduler.contact_states.numpy()
        )

    def test_missing_command_term_raises(self):
        env, _, _, _ = _make_env()
        env.command_manager.get_term = lambda name: None
        with self.assertRaises(KeyError):
            dtc_obs.tamols_footholds(env)


class TestGaitPhaseObs(unittest.TestCase):
    """trot 相位观测: sin/cos 编码数值 + 摆动掩码 + cfg 接线."""

    def _make_env(self, t: float = 0.0):
        from types import SimpleNamespace

        # trot: FL/RR offset=0, FR/RL offset=0.5, T=0.6
        from myrobot_mjlab.task.Tamols.fetch.gait_scheduler import (
            GaitScheduler,
        )

        gait = GaitScheduler(num_envs=2)
        gait.current_time.fill_(t)
        gait._update_all()
        cmd = SimpleNamespace(gait_scheduler=gait)
        return SimpleNamespace(
            scene={"robot": object()},
            command_manager=SimpleNamespace(get_term=lambda name: cmd),
        )

    def test_shape_and_encoding(self):
        import math

        env = self._make_env(t=0.15)  # phi_FL=0.25, phi_FR=0.75
        out = dtc_obs.tamols_gait_phase(env)
        self.assertEqual(out.shape, (2, 12))
        phi_fl = 0.25
        np.testing.assert_allclose(
            out[0, :2].numpy(),
            [math.sin(2 * math.pi * phi_fl), math.cos(2 * math.pi * phi_fl)],
            atol=1e-6,
        )
        phi_fr = 0.75
        np.testing.assert_allclose(
            out[0, 2:4].numpy(),
            [math.sin(2 * math.pi * phi_fr), math.cos(2 * math.pi * phi_fr)],
            atol=1e-6,
        )
        # t=0.15: FL 支撑(phi<D=0.5)->mask 0; FR 摆动(phi>=0.5)->1
        self.assertEqual(float(out[0, 8]), 0.0)
        self.assertEqual(float(out[0, 9]), 1.0)

    def test_without_mask(self):
        env = self._make_env()
        out = dtc_obs.tamols_gait_phase(env, include_swing_mask=False)
        self.assertEqual(out.shape, (2, 8))

    def test_diagonal_pairs_antisymmetric(self):
        # trot 对角组: FL 与 FR 相位差 0.5 -> sin 反号, cos 同号
        env = self._make_env(t=0.1)
        out = dtc_obs.tamols_gait_phase(env, include_swing_mask=False)
        self.assertAlmostEqual(
            float(out[0, 0] + out[0, 2]), 0.0, places=6
        )
        # 相位差 0.5: sin/cos 均反号 -> 和为 0
        self.assertAlmostEqual(
            float(out[0, 1] + out[0, 3]), 0.0, places=6
        )

    def test_cfg_wired_all_variants(self):
        from myrobot_mjlab.task.DTC.config import (
            my_go2_flat_env_cfg,
            my_go2_stair_env_cfg,
        )

        for fn in (my_go2_flat_env_cfg, my_go2_stair_env_cfg):
            terms = fn().observations["actor"].terms
            self.assertIn("tamols_gait_phase", terms)
            self.assertTrue(
                terms["tamols_gait_phase"].params.get(
                    "include_swing_mask", True
                )
            )


class TestCriticObsFunctions(unittest.TestCase):
    """critic 专用三项: TAMOLS 期望位置/姿态/twist (数值独立复算)."""

    def _make_env(self):
        import math

        yaw = math.pi / 2
        quat = torch.tensor(
            [[math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]] * 2
        )
        cmd = SimpleNamespace(
            desired_base_pos_w=torch.tensor([[0.5, 0.1, 0.32]] * 2),
            desired_base_quat_w=torch.tensor(
                [[0.9806, 0.0, 0.1987, 0.0]]  # ~pitch 0.4, 未归一化比例
            ).repeat(2, 1),
            desired_base_lin_vel_w=torch.tensor([[0.3, 0.1, 0.0]] * 2),
            desired_base_ang_vel_w=torch.tensor([[0.0, 0.0, 0.5]] * 2),
        )
        env = SimpleNamespace(
            scene={
                "robot": SimpleNamespace(
                    data=SimpleNamespace(
                        root_link_pos_w=torch.tensor(
                            [[0.2, 0.0, 0.3]] * 2
                        ),
                        root_link_quat_w=quat,
                    )
                )
            },
            command_manager=SimpleNamespace(
                get_term=lambda name: cmd,
            ),
        )
        return env, yaw, cmd

    def test_base_pos_desired_body_frame(self):
        import math

        env, yaw, _ = self._make_env()
        out = dtc_obs.tamols_base_pos_desired(env)
        self.assertEqual(out.shape, (2, 3))
        c, s = math.cos(yaw), math.sin(yaw)
        dx, dy, dz = 0.5 - 0.2, 0.1 - 0.0, 0.32 - 0.3
        np.testing.assert_allclose(
            out[0].numpy(),
            [c * dx + s * dy, -s * dx + c * dy, dz],
            atol=1e-6,
        )

    def test_base_quat_desired_normalized(self):
        env, _, cmd = self._make_env()
        out = dtc_obs.tamols_base_quat_desired(env)
        self.assertEqual(out.shape, (2, 4))
        norm = out.norm(dim=-1)
        np.testing.assert_allclose(norm.numpy(), 1.0, atol=1e-6)
        # 方向保持 (与原始未归一化四元数同向)
        np.testing.assert_allclose(
            out.numpy(),
            (cmd.desired_base_quat_w
             / cmd.desired_base_quat_w.norm(dim=-1, keepdim=True)).numpy(),
            atol=1e-6,
        )

    def test_base_twist_desired_body_frame(self):
        import math

        env, yaw, _ = self._make_env()
        out = dtc_obs.tamols_base_twist_desired(env)
        self.assertEqual(out.shape, (2, 6))
        c, s = math.cos(yaw), math.sin(yaw)
        exp_lin = (c * 0.3 + s * 0.1, -s * 0.3 + c * 0.1, 0.0)
        # 角速度只有 z 分量 (0,0,0.5): yaw 旋转不改变 z 分量
        exp_ang = (0.0, 0.0, 0.5)
        np.testing.assert_allclose(
            out[0, :3].numpy(), exp_lin, atol=1e-6
        )
        np.testing.assert_allclose(
            out[0, 3:].numpy(), exp_ang, atol=1e-6
        )


class TestCfgWiring(unittest.TestCase):
    def _actor(self, factory):
        cfg = factory()
        return cfg.observations["actor"].terms

    def test_all_variants_have_tamols_terms(self):
        from myrobot_mjlab.task.DTC.config import (
            my_go2_flat_env_cfg,
            my_go2_stair_env_cfg,
            my_go2_tire_env_cfg,
        )
        from myrobot_mjlab.task.DTC.config.env_cfg import my_go2_rough_env_cfg

        for name, fn in (
            ("rough", my_go2_rough_env_cfg),
            ("flat", my_go2_flat_env_cfg),
            ("stair", my_go2_stair_env_cfg),
            ("tire", my_go2_tire_env_cfg),
        ):
            terms = self._actor(fn)
            for key in (
                "tamols_footholds",
                "tamols_desired_joint_pos",
                "tamols_desired_contacts",
            ):
                self.assertIn(key, terms, f"{name}: 缺 {key}")
            # 既有项未被破坏
            for key in (
                "base_lin_vel", "base_ang_vel", "projected_gravity",
                "joint_pos", "joint_vel", "actions", "command",
            ):
                self.assertIn(key, terms, f"{name}: 既有项 {key} 丢失")

    def test_height_scan_variants(self):
        from myrobot_mjlab.task.DTC.config import (
            my_go2_flat_env_cfg,
            my_go2_stair_env_cfg,
        )

        stair = self._actor(my_go2_stair_env_cfg)
        self.assertIn("height_scan", stair)
        self.assertEqual(stair["height_scan"].params["sensor_name"], "terrain_scan")
        # 用户已保留 flat 的 terrain_scan (原移除代码被注释):
        # flat 现同样带 height_scan 观测
        flat = self._actor(my_go2_flat_env_cfg)
        self.assertIn("height_scan", flat)
        self.assertEqual(
            flat["height_scan"].params["sensor_name"], "terrain_scan"
        )

    def test_critic_privileged_terms(self):
        from myrobot_mjlab.task.DTC.config import my_go2_flat_env_cfg

        cfg = my_go2_flat_env_cfg()
        actor = cfg.observations["actor"].terms
        critic = cfg.observations["critic"].terms
        for key in (
            "tamols_base_pos_desired",
            "tamols_base_quat_desired",
            "tamols_base_twist_desired",
        ):
            self.assertIn(key, critic)
            # 特权信息: actor 不得包含 (actor 的 "command" 是 twist
            # 指令, 与 tamols_base_twist_desired 是不同量)
            self.assertNotIn(key, actor)

    def test_critic_inherits_actor_terms(self):
        from myrobot_mjlab.task.DTC.config import my_go2_flat_env_cfg

        cfg = my_go2_flat_env_cfg()
        critic = cfg.observations["critic"].terms
        self.assertIn("tamols_footholds", critic)
        self.assertIn("tamols_desired_contacts", critic)


if __name__ == "__main__":
    unittest.main()
