"""硬落脚跟踪奖励测试 (shim env, 独立 NumPy 复算).

运行:
    python -m unittest myrobot_mjlab.task.DTC.reward.test_hard_tracking -v

覆盖 (新语义):
  1. 数值: -ln(max(d, d_min)^2 + eps) 独立复算 (含 landing_footholds 优先);
  2. 门控: 期望接触(支撑) & ||f||>1 才给分;
  3. 每周期每脚一次: 连续满足门控的下一步 0 分;
  4. 触地沿重置: swing->stance 后可再次得分;
  5. **出生不送分**: 状态初建时支撑相腿即使有力也不得分, 需先经历摆动;
  6. **时钟回绕 reset 检测**: current_time 回退 => 支撑相腿重新锁定;
  7. **d_min 封顶**: d=0 时得分为 -ln(d_min^2+eps), 不再是 -ln(eps);
  8. 接触传感器 primary 顺序映射 (FR,FL,RR,RL -> FL/FR/RL/RR);
  9. 无力传感器退化 (仅期望门控); eps/d_min 非法报错;
 10. cfg 接线断言.
"""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import torch

from myrobot_mjlab.task.DTC.reward import terms as dtc_reward

_EPS = 1e-3
_D_MIN = 0.02


def _expected(d: float, eps: float = _EPS, d_min: float = _D_MIN) -> float:
    """单脚期望得分: -ln(max(d, d_min)^2 + eps)."""
    de = max(d, d_min)
    return -math.log(de * de + eps)


def _make_env(
    foot_z_offset: float = 0.0,
    force: torch.Tensor | None = None,
    primary_names=("FR", "FL", "RR", "RL"),
    contact=(1.0, 1.0, 1.0, 1.0),
    landing_offset_x: float = 0.1,
):
    """shim: 4 腿, 期望落点 = 当前脚 + (landing_offset_x, 0, z_offset).

    gait_scheduler 带可写的 contact_states 与 current_time (时钟回绕
    检测依赖后者)。
    """
    foot = torch.tensor(
        [[[0.20, 0.15, 0.0], [0.20, -0.15, 0.0],
          [-0.20, 0.15, 0.0], [-0.20, -0.15, 0.0]]]
    ).repeat(2, 1, 1)
    foot[:, :, 2] += foot_z_offset
    landing = foot.clone()
    landing[:, :, 0] += landing_offset_x
    cmd = SimpleNamespace(
        optimal_footholds=landing + 5.0,          # 应被 landing 覆盖
        landing_footholds=landing,
        gait_scheduler=SimpleNamespace(
            contact_states=torch.tensor(contact).unsqueeze(0).repeat(2, 1),
            current_time=torch.zeros(2),
        ),
    )
    data = SimpleNamespace(
        body_link_pos_w=torch.zeros(2, 6, 3),
        root_link_pos_w=torch.zeros(2, 3),
        root_link_quat_w=torch.tensor([[1.0, 0, 0, 0]]).repeat(2, 1),
        root_link_lin_vel_b=torch.zeros(2, 3),
        root_link_ang_vel_b=torch.zeros(2, 3),
    )
    data.body_link_pos_w[:, 1:5, :] = foot  # FL/FR/RL/RR = 下标 1..4
    asset = SimpleNamespace(body_names=[
        "base", "FL_foot", "FR_foot", "RL_foot", "RR_foot", "extra",
    ], data=data)
    sensor = None
    if force is not None:
        sensor = SimpleNamespace(
            data=SimpleNamespace(force=force),
            primary_names=primary_names,
        )

    class _Scene(dict):
        sensors: dict = {}

    scene = _Scene({"robot": asset, "feet_ground_contact": sensor})
    scene.sensors = {"feet_ground_contact": sensor} if sensor else {}
    env = SimpleNamespace(
        scene=scene,
        command_manager=SimpleNamespace(get_term=lambda name: cmd),
        step_dt=0.02,
    )
    return env, foot, landing, cmd


def _forces(norm_per_leg, primary_names=("FR", "FL", "RR", "RL")):
    """构造 [B, P, 3] 力张量, norm_per_leg 按 FL/FR/RL/RR 顺序给范数."""
    f = torch.zeros(2, len(primary_names), 3)
    for i, name in enumerate(primary_names):
        f[:, i, 2] = norm_per_leg[("FL", "FR", "RL", "RR").index(name)]
    return f


def _touchdown_all(env, cmd, eps=_EPS, d_min=_D_MIN):
    """全体腿走一步摆动再触地, 返回触地步的奖励 (合法得分路径)."""
    cmd.gait_scheduler.contact_states[:] = 0.0     # 全摆动
    dtc_reward.foothold_hard_tracking(env, eps=eps, d_min=d_min)
    cmd.gait_scheduler.contact_states[:] = 1.0     # 触地沿
    return dtc_reward.foothold_hard_tracking(env, eps=eps, d_min=d_min)


class TestHardTracking(unittest.TestCase):
    def setUp(self):
        dtc_reward._HARD_TRACK_STATE.clear()

    def test_value_and_landing_priority(self):
        env, foot, landing, cmd = _make_env(force=_forces([2.0] * 4))
        r = _touchdown_all(env, cmd)
        d = float((landing[0, 0] - foot[0, 0]).norm())      # 0.1 m
        self.assertAlmostEqual(float(r[0]), 4 * _expected(d), delta=1e-5)
        # 若用 optimal(+5m) 会得到截然不同的值 -> 验证 landing 被优先
        self.assertNotAlmostEqual(
            float(r[0]), 4 * _expected(d + 5.0), places=2
        )

    def test_force_gating(self):
        # 力 <= 1: 不给分
        env, _, _, cmd = _make_env(force=_forces([0.5] * 4))
        r = _touchdown_all(env, cmd)
        self.assertEqual(float(r.abs().sum()), 0.0)
        # 混合: 仅力 > 1 的腿给分 (FL=2, RL=5 给; FR=0, RR=1.0 不给)
        env2, foot, landing, cmd2 = _make_env(
            force=_forces([2.0, 0.0, 5.0, 1.0])
        )
        r2 = _touchdown_all(env2, cmd2)
        d = float((landing[0, 0] - foot[0, 0]).norm())
        self.assertAlmostEqual(float(r2[0]), 2 * _expected(d), places=6)

    def test_once_per_stance_then_reset_on_touchdown(self):
        env, foot, landing, cmd = _make_env(force=_forces([2.0] * 4))
        r1 = _touchdown_all(env, cmd)
        self.assertGreater(float(r1[0]), 0.0)
        # 同一支撑相第二步: 已给分 -> 0
        r2 = dtc_reward.foothold_hard_tracking(env, eps=_EPS, d_min=_D_MIN)
        self.assertEqual(float(r2[0]), 0.0)
        # 再次 swing->stance -> 重新得分
        r3 = _touchdown_all(env, cmd)
        self.assertGreater(float(r3[0]), 0.0)

    def test_spawn_stance_legs_get_no_free_reward(self):
        """出生不送分: 初建状态时支撑相腿即使有力也不得分."""
        env, foot, landing, cmd = _make_env(force=_forces([2.0] * 4))
        # 首次调用即支撑+接触: 旧版此处会直接发 4 腿的分
        r0 = dtc_reward.foothold_hard_tracking(env, eps=_EPS, d_min=_D_MIN)
        self.assertEqual(float(r0.abs().sum()), 0.0)
        # 经历过摆动->触地后才得分
        r1 = _touchdown_all(env, cmd)
        d = float((landing[0, 0] - foot[0, 0]).norm())
        self.assertAlmostEqual(float(r1[0]), 4 * _expected(d), delta=1e-5)

    def test_swing_leg_never_rewards(self):
        """时钟摆动相的腿即使有力也不给分."""
        env, foot, landing, cmd = _make_env(force=_forces([2.0] * 4))
        # 初始化于全支撑 (锁定), 然后 FL/RL/RR 触地, FR 保持摆动
        dtc_reward.foothold_hard_tracking(env, eps=_EPS, d_min=_D_MIN)
        cmd.gait_scheduler.contact_states[:] = 0.0
        dtc_reward.foothold_hard_tracking(env, eps=_EPS, d_min=_D_MIN)
        cmd.gait_scheduler.contact_states[:] = torch.tensor(
            [[1.0, 0.0, 1.0, 1.0]]
        ).repeat(2, 1)
        r = dtc_reward.foothold_hard_tracking(env, eps=_EPS, d_min=_D_MIN)
        d = float((landing[0, 0] - foot[0, 0]).norm())
        self.assertAlmostEqual(float(r[0]), 3 * _expected(d), delta=1e-5)

    def test_clock_rewind_reset_blocks_free_reward(self):
        """时钟回绕 => env reset: 支撑相腿重新锁定, 不送分."""
        env, foot, landing, cmd = _make_env(force=_forces([2.0] * 4))
        clock = cmd.gait_scheduler.current_time
        # 正常走一个周期拿到分
        r1 = _touchdown_all(env, cmd)
        self.assertGreater(float(r1[0]), 0.0)
        # 时钟前进若干步 (支撑相保持 -> 已 awarded, 无分)
        clock[:] = 0.4
        r2 = dtc_reward.foothold_hard_tracking(env, eps=_EPS, d_min=_D_MIN)
        self.assertEqual(float(r2.abs().sum()), 0.0)
        # 模拟 reset: 时钟回绕到 0.02, 全体支撑 (旧版这里会再送一次分)
        clock[:] = 0.02
        r3 = dtc_reward.foothold_hard_tracking(env, eps=_EPS, d_min=_D_MIN)
        self.assertEqual(float(r3.abs().sum()), 0.0)
        # reset 后走真实摆动->触地: 恢复给分
        r4 = _touchdown_all(env, cmd)
        self.assertGreater(float(r4[0]), 0.0)

    def test_d_min_caps_jackpot(self):
        """d=0 (脚正好在落点上) 时得分被 d_min 封顶, 不再是 -ln(eps)."""
        env, _, _, cmd = _make_env(
            force=_forces([2.0] * 4), landing_offset_x=0.0
        )
        r = _touchdown_all(env, cmd)
        capped = 4 * _expected(0.0)
        uncapped = 4 * -math.log(_EPS)
        self.assertAlmostEqual(float(r[0]), capped, delta=1e-5)
        self.assertLess(capped, uncapped)

    def test_sensor_primary_order_mapping(self):
        # primaries 顺序 (FR,FL,RR,RL): 只给 FL(列表第 0) 力
        env, foot, landing, cmd = _make_env(force=_forces([3.0, 0.0, 0.0, 0.0]))
        r = _touchdown_all(env, cmd)
        d = float((landing[0, 0] - foot[0, 0]).norm())
        self.assertAlmostEqual(float(r[0]), _expected(d), delta=1e-5)

    def test_no_sensor_degrades_to_stance_gate(self):
        env, foot, landing, cmd = _make_env(force=None)
        r = _touchdown_all(env, cmd)
        d = float((landing[0, 0] - foot[0, 0]).norm())
        self.assertAlmostEqual(float(r[0]), 4 * _expected(d), delta=1e-5)

    def test_invalid_params(self):
        env, *_ = _make_env(force=_forces([2.0] * 4))
        with self.assertRaises(ValueError):
            dtc_reward.foothold_hard_tracking(env, eps=0.0)
        with self.assertRaises(ValueError):
            dtc_reward.foothold_hard_tracking(env, eps=1.5)
        with self.assertRaises(ValueError):
            dtc_reward.foothold_hard_tracking(env, d_min=0.0)


class TestCfgWiring(unittest.TestCase):
    def test_hard_tracking_wired(self):
        from myrobot_mjlab.task.DTC.config import my_go2_flat_env_cfg

        rewards = my_go2_flat_env_cfg().rewards
        self.assertIn("foothold_hard_tracking", rewards)
        term = rewards["foothold_hard_tracking"]
        self.assertEqual(term.func.__name__, "foothold_hard_tracking")
        self.assertEqual(
            term.func.__module__, "myrobot_mjlab.task.DTC.reward.terms"
        )
        self.assertEqual(term.params["command_name"], "Tamols_footholds")
        self.assertEqual(term.params["sensor_name"], "feet_ground_contact")
        self.assertLess(term.params["eps"], 0.1)
        self.assertGreater(term.params.get("d_min", 0.02), 0.0)
        self.assertGreater(term.weight, 0.0)


if __name__ == "__main__":
    unittest.main()
