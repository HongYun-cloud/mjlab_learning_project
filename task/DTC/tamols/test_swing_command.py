"""DTC 命令项的摆动腿观测集成测试 (shim, 无需真实环境).

运行:
    python -m unittest myrobot_mjlab.task.DTC.tamols.test_swing_command -v

覆盖:
  1. cfg 默认开启 swing_obs, 路径采样数可配;
  2. 规划方法仍未覆写 (只有观测/绘制扩展);
  3. _update_swing_obs 端到端 (shim 注入 robot.data / h_raw / trot
     时钟): 观测字典形状、数值与独立 NumPy 复算一致 (含 yaw 旋转与
     本地高程图查询);
  4. 共享时钟: 命令项推进 gait_scheduler 一次, 观测相位跟随, builder
     不重复 step;
  5. 关闭开关时 _update_swing_obs 为 no-op.
"""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from myrobot_mjlab.task.DTC.tamols.dtc_command import (
    DTCFootholdCommand,
    DTCFootholdCommandCfg,
)
from myrobot_mjlab.task.Tamols.fetch.gait_scheduler import GaitScheduler


def _make_cmd(num_envs: int = 2, enable: bool = True):
    cmd = object.__new__(DTCFootholdCommand)
    cmd.cfg = DTCFootholdCommandCfg(
        entity_name="robot", sensor_name="terrain_scan",
        swing_obs_enable=enable, swing_path_samples=5,
    )
    cmd._env = SimpleNamespace(num_envs=num_envs, step_dt=0.02, device="cpu")
    cmd.gait_scheduler = GaitScheduler(num_envs=num_envs)
    # 本地高程图 (机体系网格, x∈[-1,1] dx=0.1): z = 0.1x + 0.05y
    # 实际 TamolsMotionCommand.h_raw 布局为 [N,1,Nx,Ny] (x=第2维,
    # y=第3维), 不是 [N,1,Ny,Nx]; 这里按真实布局构造, _update_swing_obs
    # 内部再转置给 LocalGridHeights.
    xs = np.arange(-1.0, 1.0 + 1e-9, 0.1)
    ys = np.arange(-1.0, 1.0 + 1e-9, 0.1)
    H = 0.1 * xs[:, None] + 0.05 * ys[None, :]  # V[ix, iy]
    cmd.h_raw = torch.from_numpy(H).float()[None, None].repeat(num_envs, 1, 1, 1)
    cmd.x_min, cmd.y_min, cmd.cell_size = -1.0, -1.0, 0.1
    # robot.data: 脚位置 / 基座位姿
    foot = torch.tensor(
        [[[0.20, 0.15, 0.02], [0.20, -0.15, 0.02],
          [-0.20, 0.15, 0.02], [-0.20, -0.15, 0.02]]]
    ).repeat(num_envs, 1, 1)
    yaw = math.pi / 2
    qw, qx, qy, qz = math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)
    cmd.robot = SimpleNamespace(
        body_names=["base_link", "FL_foot", "FR_foot", "RL_foot", "RR_foot"],
        data=SimpleNamespace(
            body_link_pos_w=torch.zeros(num_envs, 5, 3),
            root_link_pos_w=torch.zeros(num_envs, 3),
            root_link_quat_w=torch.tensor(
                [[qw, qx, qy, qz]] * num_envs
            ),
        ),
    )
    cmd.robot.data.body_link_pos_w[:, 1:5, :] = foot
    cmd.optimal_footholds = foot.clone()
    cmd.optimal_footholds[:, :, 0] += 0.10
    cmd.optimal_footholds[:, :, 2] += 0.03
    cmd._post_super_init()
    return cmd, foot


class TestSwingObsIntegration(unittest.TestCase):
    def test_cfg_defaults_and_no_planning_override(self):
        cfg = DTCFootholdCommandCfg(entity_name="robot", sensor_name="s")
        self.assertTrue(cfg.swing_obs_enable)
        self.assertEqual(cfg.swing_path_samples, 5)
        # 核心规划方法不得覆写
        self.assertNotIn("_update_command", DTCFootholdCommand.__dict__)
        # _resample_command 允许覆写但必须先调 super (保留原重采样行为)
        import inspect

        src = inspect.getsource(
            DTCFootholdCommand.__dict__["_resample_command"]
        )
        self.assertIn("super()._resample_command", src)

    def test_update_swing_obs_end_to_end(self):
        cmd, foot = _make_cmd()
        DTCFootholdCommand._update_swing_obs(cmd)
        obs = cmd.swing_obs
        self.assertIsNotNone(obs)
        self.assertEqual(obs["goal_rel_b"].shape, (2, 4, 3))
        self.assertEqual(obs["path_rel_h"].shape, (2, 4, 5))
        # 独立 NumPy 复算 (yaw=pi/2): goal_rel_b xy
        yaw = math.pi / 2
        c, s = math.cos(yaw), math.sin(yaw)
        leg = 0
        dx, dy = 0.10, 0.0
        exp_xy = (c * dx + s * dy, -s * dx + c * dy)
        got = [float(v) for v in obs["goal_rel_b"][0, leg, :]]
        self.assertAlmostEqual(got[0], exp_xy[0], places=5)
        self.assertAlmostEqual(got[1], exp_xy[1], places=5)
        self.assertAlmostEqual(got[2], 0.03, places=5)
        # 路径中点高度 (机体系): 脚在机体系 (0.15, -0.20) (yaw=pi/2),
        # 落点 (0.15, -0.30); 中点 (0.15, -0.25) -> h=0.1*0.15+0.05*(-0.25)
        mid_b = obs["path_xy_b"][0, leg, 2].numpy()
        np.testing.assert_allclose(mid_b, [0.15, -0.25], atol=1e-4)
        exp_h = 0.1 * 0.15 + 0.05 * (-0.25) - 0.02
        self.assertAlmostEqual(float(obs["path_rel_h"][0, leg, 2]), exp_h, places=4)

    def test_shared_gait_clock_follows_command(self):
        cmd, _ = _make_cmd()
        cmd.gait_scheduler.step(0.30)   # 命令项推进 (半周期)
        DTCFootholdCommand._update_swing_obs(cmd)
        self.assertAlmostEqual(
            float(cmd.swing_obs["phase"][0, 0]), 0.5, places=5
        )
        with self.assertRaises(RuntimeError):
            cmd._swing_builder.step(0.02)   # builder 禁止自行推进

    def test_disabled_is_noop(self):
        cmd, _ = _make_cmd(enable=False)
        DTCFootholdCommand._update_swing_obs(cmd)
        self.assertIsNone(cmd.swing_obs)

    def test_update_metrics_hook_calls_swing_obs(self):
        cmd, _ = _make_cmd()
        called = []
        cmd._update_swing_obs = lambda: called.append(1)
        DTCFootholdCommand._update_metrics(cmd)
        self.assertEqual(called, [1])


if __name__ == "__main__":
    unittest.main()
