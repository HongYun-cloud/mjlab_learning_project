"""正则化惩罚测试 (shim env, 独立 NumPy 复算).

运行:
    python -m unittest myrobot_mjlab.task.DTC.reward.test_regularization -v

覆盖:
  1. r_{r1} 足端功率: |v^T f| 求和 (含传感器 primary 顺序映射);
     无力传感器 -> 0;
  2. r_{r2} 关节功率: -(sum qdot*tau)^2;
  3. 动作变化率: -(a_t - a_{t-1})^2 (action_manager 内建缓冲);
  4. 关节加速度: -||qddot||^2;
  5. cfg 接线: 四项均接入 DTC 工厂.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from myrobot_mjlab.task.DTC.reward import terms as dtc_reward


def _make_asset():
    foot = torch.tensor(
        [[[0.2, 0.15, 0.0], [0.2, -0.15, 0.0],
          [-0.2, 0.15, 0.0], [-0.2, -0.15, 0.0]]]
    ).repeat(2, 1, 1)
    foot_vel = torch.tensor(
        [[[0.1, 0.0, 0.0], [0.0, -0.2, 0.0],
          [0.3, 0.1, 0.0], [0.0, 0.0, 0.0]]]
    ).repeat(2, 1, 1)
    data = SimpleNamespace(
        body_link_pos_w=torch.zeros(2, 5, 3),
        body_link_vel_w=torch.zeros(2, 5, 6),
        joint_vel=torch.tensor([[0.1, -0.2, 0.3] + [0.0] * 9]).repeat(2, 1),
        qfrc_actuator=torch.tensor([[5.0, 2.0, -1.0] + [0.0] * 9]).repeat(2, 1),
        joint_acc=torch.tensor([[0.5, -0.1, 0.2] + [0.0] * 9]).repeat(2, 1),
    )
    data.body_link_pos_w[:, 1:5, :] = foot
    data.body_link_vel_w[:, 1:5, :3] = foot_vel
    return SimpleNamespace(
        data=data,
        body_names=["base", "FL_foot", "FR_foot", "RL_foot", "RR_foot"],
    ), foot_vel


def _make_env(force=None, primary=("FR", "FL", "RR", "RL")):
    asset, foot_vel = _make_asset()

    class _Scene(dict):
        sensors: dict = {}

    scene = _Scene({"robot": asset})
    if force is not None:
        scene["feet_ground_contact"] = SimpleNamespace(
            data=SimpleNamespace(force=force),
            primary_names=primary,
        )
        scene.sensors = {"feet_ground_contact": scene["feet_ground_contact"]}
    else:
        scene.sensors = {}
    am = SimpleNamespace(
        action=torch.tensor([[0.1, 0.2, -0.3]]),
        prev_action=torch.tensor([[0.0, 0.2, -0.1]]),
    )
    return SimpleNamespace(scene=scene, action_manager=am), asset, foot_vel


class TestFootContactPower(unittest.TestCase):
    def test_value_with_primary_mapping(self):
        # primaries (FR,FL,RR,RL): FL 在 index 1, 力 (0,0,10)
        force = torch.zeros(2, 4, 3)
        force[:, 1, 2] = 10.0                     # FL: f=(0,0,10)
        env, _, foot_vel = _make_env(force=force)
        r = dtc_reward.foot_contact_power(env)
        # v_FL=(0.1,0,0) · f_FL=(0,0,10) = 0 -> |0|; 其他腿无力
        self.assertEqual(float(r.abs().sum()), 0.0)

    def test_sliding_foot_penalized(self):
        # FL 力沿 x (与速度同向): |v·f| = 0.1*4 = 0.4
        force = torch.zeros(2, 4, 3)
        force[:, 1, 0] = 4.0
        env, _, _ = _make_env(force=force)
        r = dtc_reward.foot_contact_power(env)
        self.assertAlmostEqual(float(r[0]), -0.4, places=6)

    def test_multi_leg_sum(self):
        # FL (idx1) 力 (0,0,2), RL (idx3) 力 (0,1,0)
        force = torch.zeros(2, 4, 3)
        force[:, 1, 2] = 2.0
        force[:, 3, 1] = 1.0
        env, _, foot_vel = _make_env(force=force)
        r = dtc_reward.foot_contact_power(env)
        # v_FL=(0.1,0,0)·(0,0,2)=0 ; v_RL=(0.3,0.1,0)·(0,1,0)=0.1
        self.assertAlmostEqual(float(r[0]), -0.1, places=6)

    def test_no_sensor_returns_zero(self):
        env, _, _ = _make_env(force=None)
        r = dtc_reward.foot_contact_power(env)
        self.assertEqual(float(r.abs().sum()), 0.0)


class TestJointPower(unittest.TestCase):
    def test_value(self):
        env, asset, _ = _make_env()
        r = dtc_reward.joint_power(env)
        power = float((asset.data.joint_vel[0] * asset.data.qfrc_actuator[0]).sum())
        self.assertAlmostEqual(float(r[0]), -(power**2), places=6)

    def test_negative_reward(self):
        env, _, _ = _make_env()
        r = dtc_reward.joint_power(env)
        self.assertLessEqual(float(r[0]), 0.0)


class TestActionRateAndAccel(unittest.TestCase):
    def test_action_rate_value(self):
        env, _, _ = _make_env()
        r = dtc_reward.action_rate(env)
        diff = np.array([0.1, 0.0, -0.2])
        self.assertAlmostEqual(
            float(r[0]), -float((diff**2).sum()), places=6
        )

    def test_joint_acceleration_value(self):
        env, asset, _ = _make_env()
        r = dtc_reward.joint_acceleration(env)
        exp = -float((asset.data.joint_acc[0] ** 2).sum())
        self.assertAlmostEqual(float(r[0]), exp, places=6)


class TestCfgWiring(unittest.TestCase):
    def test_regularization_wired(self):
        from myrobot_mjlab.task.DTC.config import my_go2_flat_env_cfg

        rewards = my_go2_flat_env_cfg().rewards
        for key, fname in (
            ("foot_contact_power", "foot_contact_power"),
            ("joint_power", "joint_power"),
            ("dtc_action_rate", "action_rate"),
            ("dtc_joint_acceleration", "joint_acceleration"),
        ):
            self.assertIn(key, rewards, key)
            term = rewards[key]
            self.assertEqual(term.func.__name__, fname, key)
            self.assertEqual(
                term.func.__module__,
                "myrobot_mjlab.task.DTC.reward.terms",
                key,
            )
            self.assertGreater(term.weight, 0.0, key)


if __name__ == "__main__":
    unittest.main()
