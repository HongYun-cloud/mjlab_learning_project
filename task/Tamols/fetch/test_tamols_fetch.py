from __future__ import annotations

import math
import numpy as np
import torch

from myrobot_mjlab.task.Tamols.fetch import (
    ElevationHistoryMap,
    ElevationHistoryMapCfg,
    process_height_maps,
    TAMOLSFootholdPlanner,
    TAMOLSOptimizerCfg,
    TAMOLSMotionPlanner,
    TAMOLSMotionPlannerCfg,
    GaitScheduler,
    GaitSchedulerCfg,
    go2_leg_forward_kinematics_np,
    go2_leg_analytical_ik_np,
    go2_analytical_ik_batch_torch,
)


def test_go2_analytical_ik():
    print("------------------------------------------------------------")
    print("1. Testing Unitree Go2 Analytical Inverse Kinematics")
    print("------------------------------------------------------------")
    # Test Left Leg (FL/RL) and Right Leg (FR/RR)
    test_joint_configs = [
        (-0.1, 0.9, -1.8),   # Nominal standing pose
        (0.15, 0.65, -1.4),  # Crouch pose
        (-0.2, 1.1, -2.1),   # Extended step pose
        (0.0, 0.8, -1.6),    # Neutral roll pose
    ]

    for is_left in [True, False]:
        leg_str = "Left" if is_left else "Right"
        for q_orig in test_joint_configs:
            # Adjust hip roll sign for right leg if needed
            q_test = (q_orig[0] if is_left else -q_orig[0], q_orig[1], q_orig[2])
            # FK -> p_foot
            p_foot = go2_leg_forward_kinematics_np(*q_test, is_left=is_left)
            # IK -> q_rec
            q_rec = go2_leg_analytical_ik_np(p_foot, is_left=is_left)
            err = np.max(np.abs(np.array(q_test) - np.array(q_rec)))
            assert err < 1e-4, f"IK roundtrip error too large for {leg_str} leg: {err:.6e}"

    print("[OK] Go2 Analytical IK roundtrip exactness verified (error < 1e-4 rad).")

    # Test PyTorch vectorized batch IK
    p_batch = torch.tensor([
        [[0.0, 0.0686, -0.273], [0.0, -0.0686, -0.273], [0.0, 0.0686, -0.273], [0.0, -0.0686, -0.273]],
        [[0.08, 0.07, -0.25], [0.08, -0.07, -0.25], [-0.08, 0.07, -0.25], [-0.08, -0.07, -0.25]],
    ], dtype=torch.float32)

    q_batch = go2_analytical_ik_batch_torch(p_batch)
    assert q_batch.shape == (2, 12), f"Expected shape (2, 12), got {q_batch.shape}"
    # Check that nominal standing yields approximately [-0.1, 0.9, -1.8, 0.1, 0.9, -1.8, ...]
    assert abs(q_batch[0, 1].item() - 0.9) < 0.05
    assert abs(q_batch[0, 2].item() - (-1.8)) < 0.05
    print("[OK] Go2 Vectorized PyTorch IK batch execution verified.")


def test_gait_scheduler():
    print("------------------------------------------------------------")
    print("2. Testing Gait Scheduler (Trot Phase, Contact, Time Left)")
    print("------------------------------------------------------------")
    cfg = GaitSchedulerCfg(stride_duration=0.60, duty_factor=0.50)
    scheduler = GaitScheduler(cfg=cfg, num_envs=2, device="cpu")

    # At t = 0.0:
    # Phase offsets: FL=0.0, FR=0.5, RL=0.5, RR=0.0
    # FL & RR have phi = 0.0 (< 0.5 -> Stance, c=1.0)
    # FR & RL have phi = 0.5 (>= 0.5 -> Swing, c=0.0)
    assert torch.allclose(scheduler.contact_states[0], torch.tensor([1.0, 0.0, 0.0, 1.0]))
    assert torch.allclose(scheduler.time_left[0], torch.tensor([0.30, 0.30, 0.30, 0.30]))
    print(f"[OK] t=0.00s Contact States: {scheduler.contact_states[0].tolist()} (FL+RR Stance, FR+RL Swing)")

    # Step forward by 0.30s (half stride, phase swap)
    scheduler.step(dt=0.30)
    # Now FL & RR have phi = 0.5 (Swing, c=0.0)
    # FR & RL have phi = 0.0 (Stance, c=1.0)
    assert torch.allclose(scheduler.contact_states[0], torch.tensor([0.0, 1.0, 1.0, 0.0]))
    assert torch.allclose(scheduler.time_left[0], torch.tensor([0.30, 0.30, 0.30, 0.30]))
    print(f"[OK] t=0.30s Contact States: {scheduler.contact_states[0].tolist()} (FR+RL Stance, FL+RR Swing)")

    # Step forward by 0.15s (quarter phase within swing/stance)
    scheduler.step(dt=0.15)
    # Time left should be 0.30 - 0.15 = 0.15s
    assert torch.allclose(scheduler.time_left[0], torch.tensor([0.15, 0.15, 0.15, 0.15]), atol=1e-3)
    print(f"[OK] t=0.45s Remaining Time countdown verified: {scheduler.time_left[0].tolist()}")


def test_elevation_history_and_tamols():
    print("------------------------------------------------------------")
    print("3. Testing TAMOLS Elevation Map & Foothold Optimizer")
    print("------------------------------------------------------------")

    # 1. Initialize Elevation History Map
    map_cfg = ElevationHistoryMapCfg(
        cell_size=0.05,
        x_range=(-1.0, 1.4),
        y_range=(-0.9, 0.9),
    )
    history_map = ElevationHistoryMap(map_cfg)
    assert history_map.nx == 48
    assert history_map.ny == 36

    # 2. Simulate terrain with a step at x = 0.6m in world frame
    def terrain_height_func(xw, yw):
        return 0.15 if xw >= 0.6 else 0.0

    r_samples = np.linspace(0.2, 1.4, 12)
    theta_samples = np.linspace(math.radians(-35), math.radians(35), 15)
    r_grid, th_grid = np.meshgrid(r_samples, theta_samples)

    base_pos_0 = np.array([0.0, 0.0, 0.28], dtype=np.float32)
    base_yaw_0 = 0.0

    radar_x_w0 = base_pos_0[0] + r_grid * np.cos(th_grid)
    radar_y_w0 = base_pos_0[1] + r_grid * np.sin(th_grid)
    radar_z_w0 = np.vectorize(terrain_height_func)(radar_x_w0, radar_y_w0)
    radar_hits_0 = np.stack([radar_x_w0.flatten(), radar_y_w0.flatten(), radar_z_w0.flatten()], axis=-1)

    h_0 = history_map.update(radar_hits_0, base_pos_0, base_yaw_0)

    # Robot advances forward by 0.35m
    base_pos_1 = np.array([0.35, 0.0, 0.28], dtype=np.float32)
    base_yaw_1 = 0.0
    radar_x_w1 = base_pos_1[0] + r_grid * np.cos(th_grid)
    radar_y_w1 = base_pos_1[1] + r_grid * np.sin(th_grid)
    radar_z_w1 = np.vectorize(terrain_height_func)(radar_x_w1, radar_y_w1)
    radar_hits_1 = np.stack([radar_x_w1.flatten(), radar_y_w1.flatten(), radar_z_w1.flatten()], axis=-1)

    h_1 = history_map.update(radar_hits_1, base_pos_1, base_yaw_1)

    # Terrain under front feet query
    h_at_step = history_map.query_height(0.35, 0.0)
    assert abs(h_at_step - 0.15) < 0.03, f"Expected ~0.15, got {h_at_step}"

    # Map Processing
    h_s1, h_s2, gradients = process_height_maps(h_1, cell_size=history_map.cell_size)
    assert h_s1.shape == h_1.shape
    assert h_s2.shape == h_1.shape
    print("[OK] Elevation Map & Feature processing verified.")

    # 4. TAMOLS Full Motion Planner Test
    print("------------------------------------------------------------")
    print("4. Testing Full TAMOLS Motion Planner (Pose, Accel, IK)")
    print("------------------------------------------------------------")
    motion_planner = TAMOLSMotionPlanner(TAMOLSMotionPlannerCfg(
        optimizer_cfg=TAMOLSOptimizerCfg(
            w_kinematic=15.0,
            w_edge_raw=5.0,
            w_edge_smooth=2.0,
            search_radius=0.12,
            search_resolution=0.02,
        )
    ))

    cmd_vel_b = np.array([0.3, 0.0], dtype=np.float32)
    base_vel_w = np.array([0.3, 0.0, 0.0], dtype=np.float32)
    base_ang_vel_w = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    nominal_footholds = motion_planner.foothold_planner.compute_nominal_footholds(
        base_pos_w=base_pos_1,
        base_yaw=base_yaw_1,
        base_vel_w=base_vel_w,
        cmd_vel_b=cmd_vel_b,
        stance_time=0.3,
    )

    plan_res = motion_planner.plan_motion(
        nominal_footholds_w=nominal_footholds,
        base_pos_w=base_pos_1,
        base_yaw=base_yaw_1,
        base_vel_w=base_vel_w,
        cmd_vel_b=cmd_vel_b,
        base_ang_vel_w=base_ang_vel_w,
        cmd_yaw_rate=0.0,
        elevation_map=history_map,
    )

    # 1. Check Footholds
    opt_footholds = plan_res["optimal_footholds_w"]
    assert opt_footholds.shape == (4, 3)
    print(f"[OK] Planned Optimal Footholds:\n{opt_footholds}")

    # 2. Check Base Pose
    des_pos = plan_res["desired_base_pos_w"]
    des_quat = plan_res["desired_base_quat_w"]
    des_euler = plan_res["desired_base_euler"]
    assert des_pos.shape == (3,)
    assert des_quat.shape == (4,)
    # Base height should be elevated above ground by ~h_des (0.28m)
    assert des_pos[2] > 0.28
    # Front feet are on the 15cm step while rear feet are on 0cm ground -> Pitch should be positive (tilting up)
    pitch_deg = math.degrees(des_euler[1])
    print(f"[OK] Desired Base Pose: z={des_pos[2]:.3f}m, Pitch={pitch_deg:.2f} deg, Quat={des_quat}")
    assert des_euler[1] > 0.05, f"Pitch should adapt positively to forward step: {des_euler[1]}"

    # 3. Check Base Linear & Angular Accelerations
    des_lin_acc = plan_res["desired_base_lin_acc_w"]
    des_ang_acc = plan_res["desired_base_ang_acc_w"]
    assert des_lin_acc.shape == (3,)
    assert des_ang_acc.shape == (3,)
    print(f"[OK] Desired Base Lin Acc: {des_lin_acc.tolist()} m/s^2")
    print(f"[OK] Desired Base Ang Acc: {des_ang_acc.tolist()} rad/s^2")

    # 4. Check Touchdown Desired Joint Positions via Analytical IK
    q_des = plan_res["desired_joint_pos"]
    assert q_des.shape == (12,)
    print(f"[OK] Desired Touchdown Joint Angles (12D via Analytical IK):\n{q_des.reshape(4, 3)}")

    # Ensure no NaN or Inf
    assert not np.isnan(q_des).any()
    assert not np.isinf(q_des).any()


if __name__ == "__main__":
    print("============================================================")
    print("TAMOLS & DTC Comprehensive Verification Suite")
    print("============================================================")
    test_go2_analytical_ik()
    test_gait_scheduler()
    test_elevation_history_and_tamols()
    print("\n>>> ALL 4 VERIFICATION SUITES PASSED SUCCESSFULLY! <<<")
