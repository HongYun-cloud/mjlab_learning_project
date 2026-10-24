"""Unit tests for TAMOLS & DTC reward functions."""

import math
import numpy as np
import torch
from unittest.mock import MagicMock

from myrobot_mjlab.config.reward.tamols_reward import (
    track_tamols_footholds,
    track_tamols_base_position,
    track_tamols_base_orientation,
    track_tamols_base_linear_velocity,
    track_tamols_base_angular_velocity,
    track_tamols_base_linear_acceleration,
    track_tamols_base_angular_acceleration,
    track_tamols_desired_joint_positions,
    track_tamols_contact_schedule,
    tamols_foot_power_penalty,
    tamols_joint_power_penalty,
)


def create_mock_env(num_envs: int = 4):
    env = MagicMock()
    env.step_dt = 0.02
    env.extras = {}

    # Mock TAMOLS Command Term
    cmd_term = MagicMock()
    cmd_term.optimal_footholds = torch.zeros(num_envs, 4, 3, dtype=torch.float32)
    # Set realistic default footholds
    cmd_term.optimal_footholds[:, 0] = torch.tensor([ 0.1934,  0.142, 0.0])
    cmd_term.optimal_footholds[:, 1] = torch.tensor([ 0.1934, -0.142, 0.0])
    cmd_term.optimal_footholds[:, 2] = torch.tensor([-0.1934,  0.142, 0.0])
    cmd_term.optimal_footholds[:, 3] = torch.tensor([-0.1934, -0.142, 0.0])

    cmd_term.desired_base_pos_w = torch.tensor([[0.0, 0.0, 0.28]] * num_envs, dtype=torch.float32)
    cmd_term.desired_base_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * num_envs, dtype=torch.float32)
    cmd_term.desired_base_lin_vel_w = torch.tensor([[0.4, 0.0, 0.0]] * num_envs, dtype=torch.float32)
    cmd_term.desired_base_ang_vel_w = torch.tensor([[0.0, 0.0, 0.0]] * num_envs, dtype=torch.float32)
    cmd_term.desired_base_lin_acc_w = torch.tensor([[0.1, 0.0, 0.0]] * num_envs, dtype=torch.float32)
    cmd_term.desired_base_ang_acc_w = torch.tensor([[0.0, 0.0, 0.0]] * num_envs, dtype=torch.float32)
    cmd_term.desired_joint_pos = torch.zeros(num_envs, 12, dtype=torch.float32)
    cmd_term.desired_contact_state = torch.tensor([[1.0, 0.0, 0.0, 1.0]] * num_envs, dtype=torch.float32)
    cmd_term.time_left_in_phase = torch.tensor([[0.15, 0.15, 0.15, 0.15]] * num_envs, dtype=torch.float32)
    cmd_term.gait_phase = torch.tensor([[0.2, 0.7, 0.7, 0.2]] * num_envs, dtype=torch.float32)
    # 每条腿正在飞向的落点 (真实命令项中由摆动相持续更新、触地后冻结)
    cmd_term.landing_footholds = cmd_term.optimal_footholds.clone()
    # 一次性给分状态掩码
    cmd_term.foothold_awarded = torch.zeros(num_envs, 4, dtype=torch.bool)
    cmd_term.base_pos_awarded = torch.zeros(num_envs, dtype=torch.bool)

    env.command_manager.active_terms = {"Tamols_footholds": cmd_term}
    env.command_manager.get_term.side_effect = lambda name: cmd_term

    # Mock Robot Entity
    robot = MagicMock()
    robot.body_names = [
        "base_link",
        "FL_hip", "FL_thigh", "FL_calf", "FL_foot",
        "FR_hip", "FR_thigh", "FR_calf", "FR_foot",
        "RL_hip", "RL_thigh", "RL_calf", "RL_foot",
        "RR_hip", "RR_thigh", "RR_calf", "RR_foot",
    ]

    robot_data = MagicMock()
    # Foot positions matching desired with small 1cm error
    foot_pos = torch.zeros(num_envs, len(robot.body_names), 3, dtype=torch.float32)
    foot_pos[:, robot.body_names.index("FL_foot")] = cmd_term.optimal_footholds[:, 0] + 0.01
    foot_pos[:, robot.body_names.index("FR_foot")] = cmd_term.optimal_footholds[:, 1] + 0.01
    foot_pos[:, robot.body_names.index("RL_foot")] = cmd_term.optimal_footholds[:, 2] + 0.01
    foot_pos[:, robot.body_names.index("RR_foot")] = cmd_term.optimal_footholds[:, 3] + 0.01
    robot_data.body_link_pos_w = foot_pos

    # Foot velocity: 0.1 m/s
    foot_vel = torch.zeros(num_envs, len(robot.body_names), 3, dtype=torch.float32)
    foot_vel[:, robot.body_names.index("FL_foot")] = torch.tensor([0.05, 0.0, 0.0])
    foot_vel[:, robot.body_names.index("FR_foot")] = torch.tensor([0.05, 0.0, 0.0])
    foot_vel[:, robot.body_names.index("RL_foot")] = torch.tensor([0.05, 0.0, 0.0])
    foot_vel[:, robot.body_names.index("RR_foot")] = torch.tensor([0.05, 0.0, 0.0])
    robot_data.body_link_vel_w = foot_vel

    robot_data.root_link_pos_w = torch.tensor([[0.0, 0.0, 0.28]] * num_envs, dtype=torch.float32)
    robot_data.root_link_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * num_envs, dtype=torch.float32)
    robot_data.root_link_lin_vel_w = torch.tensor([[0.4, 0.0, 0.0]] * num_envs, dtype=torch.float32)
    robot_data.root_link_ang_vel_w = torch.tensor([[0.0, 0.0, 0.0]] * num_envs, dtype=torch.float32)

    robot_data.joint_pos = torch.zeros(num_envs, 12, dtype=torch.float32)
    robot_data.joint_vel = torch.ones(num_envs, 12, dtype=torch.float32) * 0.5
    robot_data.joint_torques = torch.ones(num_envs, 12, dtype=torch.float32) * 2.0
    robot_data.qfrc_actuator = torch.ones(num_envs, 12, dtype=torch.float32) * 2.0

    robot.data = robot_data

    # Mock Contact Sensor
    contact_sensor = MagicMock()
    contact_sensor.primary_names = ("FL", "FR", "RL", "RR")
    contact_sensor_data = MagicMock()
    contact_sensor_data.found = torch.tensor([[1, 0, 0, 1]] * num_envs, dtype=torch.int32)
    contact_sensor_data.force = torch.zeros(num_envs, 4, 3, dtype=torch.float32)
    contact_sensor_data.force[:, 0, 2] = 25.0
    contact_sensor_data.force[:, 3, 2] = 25.0
    contact_sensor.data = contact_sensor_data

    env.scene = {
        "robot": robot,
        "feet_ground_contact": contact_sensor,
    }

    return env


def test_tamols_rewards():
    print("Testing TAMOLS reward functions...")
    env = create_mock_env(num_envs=4)

    # 1. Foothold tracking
    r_foot_exp = track_tamols_footholds(env, mode="exp")
    assert r_foot_exp.shape == (4,)
    assert (r_foot_exp > 0.0).all()
    print(f"[OK] track_tamols_footholds (exp): {r_foot_exp}")

    r_foot_log = track_tamols_footholds(env, mode="log")
    assert r_foot_log.shape == (4,)
    print(f"[OK] track_tamols_footholds (log): {r_foot_log}")

    # 2. Base position tracking
    r_base_pos = track_tamols_base_position(env)
    assert r_base_pos.shape == (4,)
    assert torch.allclose(r_base_pos, torch.ones(4), atol=1e-3)
    print(f"[OK] track_tamols_base_position: {r_base_pos}")

    # 3. Base orientation tracking
    r_base_rot = track_tamols_base_orientation(env)
    assert r_base_rot.shape == (4,)
    assert torch.allclose(r_base_rot, torch.ones(4), atol=1e-3)
    print(f"[OK] track_tamols_base_orientation: {r_base_rot}")

    # 4. Base linear velocity tracking
    r_base_lin_vel = track_tamols_base_linear_velocity(env)
    assert r_base_lin_vel.shape == (4,)
    assert torch.allclose(r_base_lin_vel, torch.ones(4), atol=1e-3)
    print(f"[OK] track_tamols_base_linear_velocity: {r_base_lin_vel}")

    # 5. Base angular velocity tracking
    r_base_ang_vel = track_tamols_base_angular_velocity(env)
    assert r_base_ang_vel.shape == (4,)
    assert torch.allclose(r_base_ang_vel, torch.ones(4), atol=1e-3)
    print(f"[OK] track_tamols_base_angular_velocity: {r_base_ang_vel}")

    # 6. Base accelerations
    r_lin_acc = track_tamols_base_linear_acceleration(env)
    assert r_lin_acc.shape == (4,)
    print(f"[OK] track_tamols_base_linear_acceleration: {r_lin_acc}")

    r_ang_acc = track_tamols_base_angular_acceleration(env)
    assert r_ang_acc.shape == (4,)
    print(f"[OK] track_tamols_base_angular_acceleration: {r_ang_acc}")

    # 7. Joint position guidance
    r_joint_pos = track_tamols_desired_joint_positions(env)
    assert r_joint_pos.shape == (4,)
    assert (r_joint_pos > 0.0).all()
    print(f"[OK] track_tamols_desired_joint_positions: {r_joint_pos}")

    # 8. Contact schedule tracking
    r_contact = track_tamols_contact_schedule(env)
    assert r_contact.shape == (4,)
    assert torch.allclose(r_contact, torch.ones(4), atol=1e-3)
    print(f"[OK] track_tamols_contact_schedule: {r_contact}")

    # 9. Penalties
    p_foot = tamols_foot_power_penalty(env)
    assert p_foot.shape == (4,)
    print(f"[OK] tamols_foot_power_penalty: {p_foot}")

    p_joint = tamols_joint_power_penalty(env)
    assert p_joint.shape == (4,)
    print(f"[OK] tamols_joint_power_penalty: {p_joint}")

    print("\nALL TAMOLS REWARD TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_tamols_rewards()
