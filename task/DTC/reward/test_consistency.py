"""动作一致性奖励测试 (shim env, 独立 NumPy 复算).

运行:
    python -m unittest myrobot_mjlab.task.DTC.reward.test_consistency -v

覆盖 (整周期重规划语义: 捕获 = 命令项 replan_event):
  1. 首次捕获 (只有解 a): 奖励 0;
  2. 两解数值独立复算: 解 a 整个时域 (0, H] 上绝对时间采样, 解 b 外推
     Δ+s; 四元数差; 落脚点项按可比腿掩码 comp;
  3. **整周期 comp 恒空**: 间隔内摆动过的腿被排除, 落脚点项为 0
     (相邻两解的落点目标指向不同触地事件, 计入会罚正常行走);
  4. **匀速流动解 ≈ 0**: 整周期 (Δ=H) 下两解轨迹物理重合;
  5. **逐触地兼容 (fallback)**: 无 replan_event 的命令项按触地沿捕获,
     间隔内未摆动的悬空腿 comp=1, 落脚点差计入;
  6. **err_cap 封顶**: 大跳变时单样本差被封顶, 总量有界;
  7. **按 env 事件隔离**: 只有 replan_event 为 True 的 env 捕获;
  8. **非重规划的触地沿不捕获**: 奖励值保持;
  9. 时钟回绕 (env reset): 两解作废 -> 0, 重新累计后恢复;
 10. 参数校验 (w_p/dt_compare/err_cap) + reset 钩子 + cfg 接线.
"""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import torch

from myrobot_mjlab.task.DTC.reward import terms as dtc_reward

H = 0.6        # stride_duration
DT_C = 0.01    # dt_compare
W_Z = 0.1      # 期望偏航角速度
V_X = 0.2      # 期望前向速度


def _quat_yaw(psi: float) -> list[float]:
    return [math.cos(psi / 2), 0.0, 0.0, math.sin(psi / 2)]


def _make_cmd(with_event: bool = True):
    """shim 命令项: 可写 desired_* / optimal_footholds / 步态时钟."""
    foot = torch.tensor(
        [[[0.20, 0.15, 0.0], [0.20, -0.15, 0.0],
          [-0.20, 0.15, 0.0], [-0.20, -0.15, 0.0]]]
    ).repeat(2, 1, 1)
    cmd = SimpleNamespace(
        desired_base_pos_w=torch.tensor([[0.0, 0.0, 0.3]] * 2),
        desired_base_quat_w=torch.tensor([_quat_yaw(0.0)]).repeat(2, 1),
        desired_base_lin_vel_w=torch.tensor([[V_X, 0.0, 0.0]] * 2),
        desired_base_ang_vel_w=torch.tensor([[0.0, 0.0, W_Z]] * 2),
        desired_base_lin_acc_w=torch.zeros(2, 3),
        desired_base_ang_acc_w=torch.zeros(2, 3),
        optimal_footholds=foot,
        gait_scheduler=SimpleNamespace(
            cfg=SimpleNamespace(stride_duration=H),
            is_swing=torch.zeros(2, 4, dtype=torch.bool),
            current_time=torch.zeros(2),
        ),
    )
    if with_event:
        cmd.replan_event = torch.zeros(2, dtype=torch.bool)
    cmd._last_dx = [0.0, 0.0]
    return cmd


def _make_env(cmd):
    class _Scene(dict):
        sensors: dict = {}

    return SimpleNamespace(
        scene=_Scene({"robot": object()}),
        command_manager=SimpleNamespace(get_term=lambda name: cmd),
        step_dt=0.02,
    )


def _set_solution(
    cmd, pos_x: float, yaw: float, foot_dx: float = 0.0,
    vx: float = V_X, envs=(0, 1),
):
    """把 shim 命令项设成 "捕获时刻参考 (pos_x, yaw)、速度 vx 的规划解"."""
    for i in envs:
        cmd.desired_base_pos_w[i] = torch.tensor([pos_x, 0.0, 0.3])
        cmd.desired_base_quat_w[i] = torch.tensor(_quat_yaw(yaw))
        cmd.desired_base_lin_vel_w[i] = torch.tensor([vx, 0.0, 0.0])
        cmd.optimal_footholds[i, :, 0] += foot_dx - cmd._last_dx[i]
        cmd._last_dx[i] = foot_dx


def _step(env, cmd, t: float, swing, event: bool = False) -> torch.Tensor:
    """推进时钟到 t、设置摆动掩码与重规划事件, 调一次奖励."""
    cmd.gait_scheduler.current_time.fill_(t)
    cmd.gait_scheduler.is_swing[:] = torch.tensor(swing)
    if hasattr(cmd, "replan_event"):
        cmd.replan_event.fill_(event)
    return dtc_reward.action_consistency(env, w_p=1.0, dt_compare=DT_C)


_ALL_STANCE = (False, False, False, False)
_ALL_SWING = (True, True, True, True)
_TOUCH = (False, True, False, True)   # FL/RR 触地相, FR/RL 摆动相


class TestActionConsistency(unittest.TestCase):
    def setUp(self):
        dtc_reward._CONSIST_STATE.clear()

    def test_first_capture_zero(self):
        cmd = _make_cmd()
        env = _make_env(cmd)
        r = _step(env, cmd, 0.6, _TOUCH, event=True)   # 首次捕获: 只有 a
        self.assertEqual(float(r.abs().sum()), 0.0)

    def test_full_cycle_recompute(self):
        """整周期 Δ=H: 解 a 整时域采样; FR/RL 间隔内落地再起摆 (有沿)
        -> comp 恒空, 落脚点项为 0."""
        cmd = _make_cmd()
        env = _make_env(cmd)
        _set_solution(cmd, pos_x=0.0, yaw=0.0)              # 解 b @ t0=0.6
        _step(env, cmd, 0.6, _TOUCH, event=True)
        # 间隔步 @0.9: FR/RL 落地 (产生触地沿 -> 目标已指向新事件)
        _step(env, cmd, 0.9, _ALL_STANCE, event=False)
        _set_solution(                                       # 解 a @ t0=1.2
            cmd, pos_x=0.06, yaw=0.05, foot_dx=0.05, vx=0.25
        )
        r = _step(env, cmd, 1.2, _TOUCH, event=True)

        delta = 0.6
        total = 0.0
        j = 1
        while j * DT_C < H - 1e-9:                           # s ∈ (0, 0.6)
            s = j * DT_C
            pa_x = 0.06 + 0.25 * s
            pb_x = 0.0 + V_X * (delta + s)
            # yaw_a(s) = 0.05 + W_Z·s; yaw_b(Δ+s) = 0 + W_Z·(Δ+s)
            d_yaw = abs((0.05 + W_Z * s) - (W_Z * (delta + s)))
            e = math.sqrt((pa_x - pb_x) ** 2 + d_yaw ** 2)
            total += min(e, 0.1)
            j += 1
        # comp 恒空: FR/RL 间隔内落过地 -> 落脚点项 0
        expected = -total
        self.assertAlmostEqual(float(r[0]), expected, delta=2e-4)
        self.assertAlmostEqual(float(r[0]), float(r[1]), places=6)

    def test_consistent_walking_scores_zero(self):
        """整周期流动解: Δ=H 时两解轨迹处处重合 -> 惩罚 ≈ 0."""
        cmd = _make_cmd()
        env = _make_env(cmd)
        _set_solution(cmd, pos_x=V_X * 0.6, yaw=W_Z * 0.6)   # b @ 0.6
        _step(env, cmd, 0.6, _TOUCH, event=True)
        _set_solution(cmd, pos_x=V_X * 1.2, yaw=W_Z * 1.2)   # a @ 1.2
        r = _step(env, cmd, 1.2, _TOUCH, event=True)
        self.assertAlmostEqual(float(r.abs().max()), 0.0, delta=1e-5)

    def test_comp_includes_pending_swing_leg(self):
        """逐触地模式可比腿: 全程悬空未落地的腿 (无沿) 计入落脚点项."""
        cmd = _make_cmd()
        env = _make_env(cmd)
        _set_solution(cmd, pos_x=0.0, yaw=0.0)               # 解 b @ t0=0.3
        _step(env, cmd, 0.3, _TOUCH, event=True)
        # 间隔步: FR/RL 保持悬空 (无触地沿 -> 目标仍指向同一次落地)
        _step(env, cmd, 0.45, _TOUCH, event=False)
        # 捕获步 t0=0.6: FR/RL 仍悬空 -> comp = FR/RL
        _set_solution(cmd, pos_x=0.06, yaw=0.05, foot_dx=0.05, vx=0.25)
        r = _step(env, cmd, 0.6, _TOUCH, event=True)

        delta = 0.3
        total = 0.0
        j = 1
        while j * DT_C < H - 1e-9:
            s = j * DT_C
            pa_x = 0.06 + 0.25 * s
            pb_x = 0.0 + V_X * (delta + s)
            d_yaw = abs((0.05 + W_Z * s) - (W_Z * (delta + s)))
            e = math.sqrt((pa_x - pb_x) ** 2 + d_yaw ** 2)
            total += min(e, 0.1)
            j += 1
        foot_diff = math.sqrt(2 * 0.05 ** 2)                 # FR/RR 两腿差 0.05
        expected = -(total + foot_diff)
        self.assertAlmostEqual(float(r[0]), expected, delta=2e-4)

    def test_err_cap_bounds_total(self):
        """单样本差超 err_cap 被封顶: 大跳变时总量有界."""
        cmd = _make_cmd()
        env = _make_env(cmd)
        _set_solution(cmd, pos_x=0.0, yaw=0.0)
        _step(env, cmd, 0.6, _TOUCH, event=True)
        _set_solution(cmd, pos_x=5.0, yaw=2.0)               # 5 m 级跳变
        r = _step(env, cmd, 1.2, _TOUCH, event=True)
        max_total = (H / DT_C - 1) * 0.1                     # 59 采样 × cap
        self.assertGreater(float(r[0]), -max_total - 1e-6)
        self.assertLess(float(r[0]), -max_total + 0.01)      # 每样本都顶到 cap

    def test_per_env_event_isolation(self):
        """只有 replan_event 为 True 的 env 捕获, 其余 env 奖励不变."""
        cmd = _make_cmd()
        env = _make_env(cmd)
        _set_solution(cmd, pos_x=0.0, yaw=0.0)
        _step(env, cmd, 0.6, _TOUCH, event=True)
        _set_solution(cmd, pos_x=0.08, yaw=W_Z * 0.6)        # 与流差 0.02 m
        r1 = _step(env, cmd, 1.2, _TOUCH, event=True)
        self.assertNotEqual(float(r1[0]), 0.0)
        # 只给 env0 下一次捕获 (env1 event=False, 解不更新)
        _set_solution(cmd, pos_x=0.30, yaw=W_Z * 1.8, envs=(0,))
        cmd.gait_scheduler.current_time.fill_(1.8)
        cmd.gait_scheduler.is_swing[:] = torch.tensor(
            [[False, True, False, True], [False, True, False, True]]
        )
        cmd.replan_event[:] = torch.tensor([True, False])
        r2 = dtc_reward.action_consistency(env, w_p=1.0, dt_compare=DT_C)
        self.assertNotEqual(float(r2[0]), float(r1[0]))      # env0 已更新
        self.assertAlmostEqual(float(r2[1]), float(r1[1]), places=6)

    def test_non_replan_touchdown_keeps_value(self):
        """非重规划的触地沿 (event=False) 不捕获 -> 奖励值保持不变."""
        cmd = _make_cmd()
        env = _make_env(cmd)
        _set_solution(cmd, pos_x=0.0, yaw=0.0)
        _step(env, cmd, 0.6, _TOUCH, event=True)
        _set_solution(cmd, pos_x=0.08, yaw=W_Z * 0.6)
        r1 = _step(env, cmd, 1.2, _TOUCH, event=True)
        # 抬腿步 + 触地沿步 (event=False): 快照不动, 值相同
        _step(env, cmd, 1.38, _ALL_SWING, event=False)
        r2 = _step(env, cmd, 1.4, _ALL_STANCE, event=False)
        self.assertEqual(float(r1[0]), float(r2[0]))

    def test_clock_rewind_invalidates(self):
        """时钟回绕 = env reset: 两解作废, 奖励归零, 重新累计后恢复."""
        cmd = _make_cmd()
        env = _make_env(cmd)
        _set_solution(cmd, pos_x=0.0, yaw=0.0)
        _step(env, cmd, 0.6, _TOUCH, event=True)
        _set_solution(cmd, pos_x=0.08, yaw=W_Z * 0.6)
        r1 = _step(env, cmd, 1.2, _TOUCH, event=True)
        self.assertNotEqual(float(r1[0]), 0.0)
        # reset: 时钟回绕 (全摆动掩码, 无捕获)
        r_reset = _step(env, cmd, 0.02, _ALL_SWING, event=False)
        self.assertEqual(float(r_reset.abs().sum()), 0.0)
        # 重新累计两次捕获 -> 恢复
        _set_solution(cmd, pos_x=0.0, yaw=0.0, foot_dx=0.0)
        _step(env, cmd, 0.6, _TOUCH, event=True)
        r_mid = _step(env, cmd, 0.62, _TOUCH, event=False)
        self.assertEqual(float(r_mid.abs().sum()), 0.0)      # 只有 a
        _set_solution(cmd, pos_x=0.08, yaw=W_Z * 1.2, foot_dx=0.0)
        r_back = _step(env, cmd, 1.2, _TOUCH, event=True)
        self.assertNotEqual(float(r_back[0]), 0.0)

    def test_fallback_touchdown_capture(self):
        """无 replan_event 属性的命令项: 回退到触地沿捕获 (兼容旧行为)."""
        cmd = _make_cmd(with_event=False)
        env = _make_env(cmd)
        # 抬腿步 (初始化) -> 落地步 (触地沿) 捕获
        r = _step(env, cmd, 0.3, _ALL_SWING, event=False)
        self.assertEqual(float(r.abs().sum()), 0.0)
        _set_solution(cmd, pos_x=0.0, yaw=0.0)
        r = _step(env, cmd, 0.6, _TOUCH, event=False)        # 沿: 捕获 a
        self.assertEqual(float(r.abs().sum()), 0.0)
        _set_solution(cmd, pos_x=0.06, yaw=W_Z * 0.3)
        # 第二次沿: FL/RR 落地 (掩码翻转产生沿) -> 捕获 b+新 a
        r = _step(env, cmd, 0.9, (True, False, True, False), event=False)
        self.assertNotEqual(float(r[0]), 0.0)

    def test_params_and_reset(self):
        cmd = _make_cmd()
        env = _make_env(cmd)
        with self.assertRaises(ValueError):
            dtc_reward.action_consistency(env, w_p=0.0)
        with self.assertRaises(ValueError):
            dtc_reward.action_consistency(env, w_p=1.0, dt_compare=0.0)
        with self.assertRaises(ValueError):
            dtc_reward.action_consistency(env, w_p=1.0, err_cap=0.0)
        _step(env, cmd, 0.6, _TOUCH, event=True)
        self.assertIn(id(env), dtc_reward._CONSIST_STATE)
        dtc_reward.reset_consistency_state(id(env))
        self.assertNotIn(id(env), dtc_reward._CONSIST_STATE)


class TestCfgWiring(unittest.TestCase):
    def test_consistency_wired(self):
        from myrobot_mjlab.task.DTC.config import my_go2_flat_env_cfg

        rewards = my_go2_flat_env_cfg().rewards
        self.assertIn("action_consistency", rewards)
        term = rewards["action_consistency"]
        self.assertEqual(term.func.__name__, "action_consistency")
        self.assertEqual(
            term.func.__module__, "myrobot_mjlab.task.DTC.reward.terms"
        )
        self.assertEqual(term.params["command_name"], "Tamols_footholds")
        self.assertAlmostEqual(term.params["dt_compare"], 0.01, places=9)
        self.assertGreater(term.params["w_p"], 0.0)
        self.assertGreater(term.params.get("err_cap", 0.1), 0.0)
        self.assertGreater(term.weight, 0.0)

    def test_full_cycle_replan_default(self):
        """整周期重规划默认开启, DTC stair cfg 沿用."""
        from myrobot_mjlab.task.Tamols.fetch.tamols_command import (
            TamolsMotionCommandCfg,
        )

        self.assertTrue(TamolsMotionCommandCfg.replan_after_all_four_touchdowns)


if __name__ == "__main__":
    unittest.main()
