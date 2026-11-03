"""把 TAMOLS C++ 求解器接入 DTC 命令项的落脚点规划适配器.

设计目标
========
- **不改 fetch 的 grid 规划器**: 本类只替换 ``TamolsMotionCommand`` 中的
  ``self.torch_planner`` 的 ``plan_motion`` 行为；``compute_nominal_footholds``
  仍委托给原 grid 规划器，作为 NLP 的初值/热启动，而不是最终解。
- **C++ direct builder**: 真实 NLP / Layer 1 都从 ``durations`` /
  ``schedules`` / ``TerrainMaps`` / ``NLPParameters`` 直接打包成
  :class:`TamolsSolverProblem`，默认路径不再构建 ``IntegratedNLP``
  (CasADi 表达式仅保留给显式 ``nlp_solver_backend="python"``).
- **异步求解**: worker 内完成 direct builder、Layer 1、Layer 2、真实 GN-SQP
  与轨迹导出；主线程只打包数组并取上一份有效解，不阻塞控制循环。
- **失败可回退**: NLP 不可行/超时/未收敛时，按配置回退到原 grid 规划器，
  保证 play 不中断。
"""

from __future__ import annotations

import math
import queue
import threading
import time
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import torch

from myrobot_mjlab.task.DTC.tamols.assumptions_config import AssumptionsConfig
from myrobot_mjlab.task.DTC.tamols.gn_sqp_solver import (
    GNSQPOptions,
    solve_gn_sqp,
)
from myrobot_mjlab.task.DTC.tamols.map_query import TerrainMaps
from myrobot_mjlab.task.DTC.tamols.nlp_integration import LegSchedule, NLPParameters
from myrobot_mjlab.task.DTC.tamols.rl_bridge import plan_from_report
from myrobot_mjlab.task.DTC.tamols.trajectory_export import (
    export_trajectory,
    touchdown_joint_velocities,
)
from myrobot_mjlab.task.DTC.tamols.v_a_constraints import rotation_matrix_zyx


class _AsyncPending(Exception):
    """异步求解尚未返回；调用方应继续使用 grid 计划。"""


@dataclass
class PlanContext:
    """轻量计划上下文: 只保留导出/解包需要的时序与布局.

    替代 ``IntegratedNLP`` 作为下游 ``export_trajectory`` /
    ``plan_from_report`` 的输入，避免为了导出而构建 CasADi 表达式。
    """

    durations: list[float]
    schedules: dict[int, LegSchedule]
    num_coeffs: int = 5
    use_slack: bool = True

    @property
    def num_phases(self) -> int:
        return len(self.durations)

    @property
    def layout(self) -> SimpleNamespace:
        return SimpleNamespace(
            num_phases=self.num_phases,
            num_coeffs=self.num_coeffs,
            use_slack=self.use_slack,
        )

    def touchdown_legs(self) -> list[int]:
        return [
            leg for leg in range(4)
            if self.schedules[leg].touchdown_leg
        ]

    def stance_legs(self, phase: int) -> list[int]:
        return [
            leg for leg in range(4)
            if self.schedules[leg].in_stance(phase)
        ]


_CPP_SOLVE_LOCK = threading.Lock()
"""CppAD 的 tape 不是线程安全的；所有 C++ 求解串行进入."""

_CASADI_BUILD_LOCK = threading.Lock()
"""仅 Python/CasADi 后端使用；默认 cpp 路径不构建 CasADi 表达式."""


def _accepted_status(status: str, max_viol: float, limit: float = 5e-3) -> bool:
    # QP 失败时当前点可能仍是可行/可接受的；只要约束违反足够小就保留，
    # 避免因为一次 QP 数值失败丢掉整份计划.
    return status.startswith("GN-SQP: converged") or max_viol <= limit


# ---------------------------------------------------------------------- #
# 小工具: 四元数/欧拉角
# ---------------------------------------------------------------------- #
def quat_wxyz_to_euler_zyx(q: Sequence[float]) -> np.ndarray:
    """``[w,x,y,z]`` -> ``[roll,pitch,yaw]`` (ZYX)."""
    w, x, y, z = [float(v) for v in q]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, sinp)))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw], dtype=float)


def euler_zyx_to_quat_wxyz(euler: Sequence[float]) -> np.ndarray:
    """``[roll,pitch,yaw]`` -> ``[w,x,y,z]`` (ZYX)."""
    roll, pitch, yaw = [float(v) for v in euler]
    cr, sr = math.cos(0.5 * roll), math.sin(0.5 * roll)
    cp, sp = math.cos(0.5 * pitch), math.sin(0.5 * pitch)
    cy, sy = math.cos(0.5 * yaw), math.sin(0.5 * yaw)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=float,
    )


def euler_rates_from_body_omega(
    euler: Sequence[float], omega_b: Sequence[float]
) -> np.ndarray:
    """机体系角速度 -> ZYX 欧拉角导数 (``body_omega_zyx`` 的逆)."""
    roll, pitch, _ = [float(v) for v in euler]
    wx, wy, wz = [float(v) for v in omega_b]
    cos_p = math.cos(pitch)
    if abs(cos_p) < 1e-4:
        cos_p = 1e-4 if cos_p >= 0.0 else -1e-4
    pitch_dot = wy * math.cos(roll) - wz * math.sin(roll)
    yaw_dot = (wy * math.sin(roll) + wz * math.cos(roll)) / cos_p
    roll_dot = wx + yaw_dot * math.sin(pitch)
    return np.array([roll_dot, pitch_dot, yaw_dot], dtype=float)


class NLPFootholdPlanner:
    """``TorchTAMOLSMotionPlanner`` 的 NLP 后端适配器.

    只覆写 ``plan_motion``；``foothold_planner`` 继续指向原 grid 规划器，
    这样 ``TamolsMotionCommand`` 仍能先算 Raibert 标称点作为 NLP 初值。
    """

    def __init__(self, command: Any, base_planner: Any, cfg: Any):
        self.command = command
        self.base_planner = base_planner
        self.cfg = cfg
        self.foothold_planner = base_planner.foothold_planner
        self._last_footholds: dict[int, np.ndarray] = {}
        self._last_state: dict[int, dict[str, np.ndarray]] = {}
        self._fail_count = 0
        self._last_status: str = ""
        self._last_solve_s: float = 0.0

        # 异步求解: 单 worker, 队列只保留最新任务; 主线程不阻塞.
        self._async_enabled = bool(getattr(cfg, "nlp_async", True))
        self._async_queue: queue.Queue = queue.Queue(maxsize=1)
        self._async_lock = threading.Lock()
        self._async_latest: dict[str, Any] | None = None
        self._async_last_row: dict[str, np.ndarray] | None = None
        self._async_applied_gen = 0
        self._async_gen = 0
        self._async_stop = threading.Event()
        self._async_thread = None
        if self._async_enabled:
            self._async_thread = threading.Thread(
                target=self._async_worker,
                name="tamols-nlp-async",
                daemon=True,
            )
            self._async_thread.start()

    def stop(self) -> None:
        self._async_stop.set()
        try:
            self._async_queue.put_nowait(None)
        except queue.Full:
            pass
        if self._async_thread is not None:
            self._async_thread.join(timeout=1.0)

    def reset(self, env_ids: Any) -> None:
        """环境 reset 时清掉该环境的上一轮落脚点，避免 V-B6 用旧位置。"""
        ids = env_ids.detach().cpu().tolist() if hasattr(env_ids, "detach") else env_ids
        for i in ids:
            self._last_footholds.pop(int(i), None)
            self._last_state.pop(int(i), None)
        with self._async_lock:
            self._async_gen += 1
            self._async_latest = None
            self._async_last_row = None
            self._async_applied_gen = self._async_gen


    # -- 异步求解 ------------------------------------------------------- #
    def _async_worker(self) -> None:
        while not self._async_stop.is_set():
            try:
                job = self._async_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if job is None:
                break
            try:
                from myrobot_mjlab.task.DTC.tamols.cpp_solver import unpack_x

                t_solve = time.perf_counter()
                ctx = PlanContext(
                    durations=[float(v) for v in job["durations"]],
                    schedules=job["schedules"],
                    num_coeffs=5,
                    use_slack=bool(job["assumptions"].use_giac_slack),
                )
                cpp = self._make_cpp_problem(
                    maps=job["maps"],
                    ctx=ctx,
                    params=job["params"],
                    assumptions=job["assumptions"],
                    ref_vx=job["ref_vx"],
                    ref_vy=job["ref_vy"],
                    ref_wz=job["ref_wz"],
                    previous_footholds=job["params"].previous_footholds,
                )
                layer1_ok = False
                sol_v = None
                cpp_v = None
                x0_v = None
                if bool(getattr(self.cfg, "nlp_layer1", True)):
                    try:
                        cpp_v = self._make_cpp_problem(
                            maps=self._virtual_maps(job["maps"]),
                            ctx=ctx,
                            params=job["params"],
                            assumptions=job["assumptions"],
                            ref_vx=job["ref_vx"],
                            ref_vy=job["ref_vy"],
                            ref_wz=job["ref_wz"],
                            previous_footholds=job["params"].previous_footholds,
                            include_dynamics=True,
                            include_kinematic=True,
                        )
                        x0_v = np.asarray(job["x0"], dtype=float).copy()
                    except Exception as exc:
                        if bool(getattr(self.cfg, "nlp_verbose", False)):
                            print(f"[NLP] async layer1 build skipped: {exc}")
                        cpp_v = None
                        x0_v = None

                with _CPP_SOLVE_LOCK:
                    # ---- Layer 1: h_s2 虚拟地面简化 TO ----
                    if cpp_v is not None and x0_v is not None:
                        xv, fv, itv, stv, violv = cpp_v.solve(
                            x0_v,
                            max_iter=int(getattr(self.cfg, "nlp_layer1_max_iter", 20)),
                            tol_step=1e-6,
                            tol_viol=1e-3,
                            tol_stationary=1e-3,
                            qp_lin_tol=1e-3,
                            hessian_reg=1e-8,
                        )
                        layer1_ok = stv.startswith("GN-SQP: converged") or (
                            "qp_failed" not in stv
                            and "qp_infeasible" not in stv
                            and violv <= 5e-2
                        )
                        if bool(getattr(self.cfg, "nlp_verbose", False)):
                            print(
                                f"[NLP] async layer1: it={itv} status={stv} "
                                f"viol={violv:.2e} ok={layer1_ok}"
                            )
                        if layer1_ok:
                            sol_v = unpack_x(ctx, xv)

                    # ---- Layer 2: 真实地图 0.4 m batch search ----
                    center = (
                        np.asarray(sol_v["desired_footholds"], dtype=float)
                        if sol_v is not None
                        else np.asarray(job["center"], dtype=float)
                    )
                    refined = None
                    if bool(getattr(self.cfg, "nlp_layer2", True)):
                        refined = self._layer2_refine(
                            job["maps"],
                            job["env_idx"],
                            center,
                            job["legs"],
                            h_valid=job.get("h_valid"),
                        )
                    elif sol_v is not None:
                        refined = np.asarray(sol_v["desired_footholds"], dtype=float)

                    # ---- 真实 NLP 热启动 ----
                    if sol_v is not None:
                        P = len(ctx.durations)
                        x0_real = np.concatenate(
                            [np.asarray(c, dtype=float).reshape(-1)
                             for c in sol_v["spline_coeffs"]]
                            + [np.asarray(refined, dtype=float).reshape(-1)]
                            + (
                                [np.asarray(sol_v["slack"], dtype=float).reshape(-1)]
                                if "slack" in sol_v
                                else [np.zeros(P, dtype=float)]
                            )
                        )
                    else:
                        x0_real = np.asarray(job["x0"], dtype=float).copy()
                        if refined is not None:
                            P = len(ctx.durations)
                            K = int(ctx.num_coeffs)
                            off = P * 6 * K
                            x0_real[off : off + 12] = np.asarray(
                                refined, dtype=float
                            ).reshape(-1)

                    x, f, it, status, max_viol = self._solve_real_truncated(
                        cpp, x0_real, job["x0_init"]
                    )
                accepted = _accepted_status(
                    status,
                    max_viol,
                    float(getattr(self.cfg, "nlp_max_violation", 5e-3)),
                )
                row = None
                if accepted:
                    row = self._row_from_solution(
                        ctx=ctx,
                        solution=unpack_x(ctx, x),
                        status=status,
                        f=f,
                        solve_seconds=time.perf_counter() - t_solve,
                        env_idx=job["env_idx"],
                        joint_pos=job["joint_pos"],
                        durations=job["durations"],
                    )
                result = {
                    "generation": job["generation"],
                    "x": x,
                    "f": f,
                    "iters": it,
                    "status": status,
                    "max_viol": max_viol,
                    "env_idx": job["env_idx"],
                    "joint_pos": job["joint_pos"],
                    "durations": job["durations"],
                    "seconds": job["seconds"],
                    "solve_seconds": time.perf_counter() - t_solve,
                    "layer1_ok": layer1_ok,
                    "accepted": accepted,
                    "row": row,
                }
            except Exception as exc:
                result = {
                    "generation": job["generation"],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            if bool(getattr(self.cfg, "nlp_verbose", False)) and "error" not in result:
                print(
                    f"[NLP] async gen={result['generation']} layer1={result.get('layer1_ok')} "
                    f"it={result['iters']} status={result['status']} "
                    f"solve={result['solve_seconds']*1e3:.0f} ms"
                )
            with self._async_lock:
                if result.get("generation", 0) >= self._async_applied_gen:
                    self._async_latest = result

    def _submit_async(
        self,
        *,
        maps: Any,
        params: Any,
        schedules: Any,
        assumptions: Any,
        x0: np.ndarray,
        x0_init: np.ndarray,
        ref_vx: float,
        ref_vy: float,
        ref_wz: float,
        center: np.ndarray,
        legs: Sequence[int],
        h_valid: np.ndarray | None,
        env_idx: int,
        joint_pos: np.ndarray,
        durations: Sequence[float],
    ) -> None:
        with self._async_lock:
            self._async_gen += 1
            gen = self._async_gen
        job = {
            "generation": gen,
            "maps": maps,
            "params": params,
            "schedules": schedules,
            "assumptions": assumptions,
            "x0": np.asarray(x0, dtype=float).copy(),
            "x0_init": np.asarray(x0_init, dtype=float).copy(),
            "ref_vx": ref_vx,
            "ref_vy": ref_vy,
            "ref_wz": ref_wz,
            "center": np.asarray(center, dtype=float).copy(),
            "legs": tuple(int(v) for v in legs),
            "h_valid": None if h_valid is None else np.asarray(h_valid, dtype=float).copy(),
            "env_idx": env_idx,
            "joint_pos": np.asarray(joint_pos, dtype=float).copy(),
            "durations": tuple(float(v) for v in durations),
            "seconds": time.perf_counter(),
        }
        try:
            self._async_queue.put_nowait(job)
        except queue.Full:
            try:
                self._async_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._async_queue.put_nowait(job)
            except queue.Full:
                pass

    def _latest_async_row(self) -> dict[str, torch.Tensor] | None:
        with self._async_lock:
            latest = self._async_latest
            applied = self._async_applied_gen
            cached = self._async_last_row
        if latest is None or latest.get("generation", 0) <= applied:
            return cached
        if "error" in latest:
            with self._async_lock:
                self._async_applied_gen = latest["generation"]
            if bool(getattr(self.cfg, "nlp_verbose", False)):
                print(f"[NLP] async failed: {latest['error']}")
            return cached
        status = latest["status"]
        accepted = _accepted_status(
            status,
            float(latest["max_viol"]),
            float(getattr(self.cfg, "nlp_max_violation", 5e-3)),
        )
        with self._async_lock:
            self._async_applied_gen = latest["generation"]
        if not accepted:
            if bool(getattr(self.cfg, "nlp_verbose", False)):
                print(
                    f"[NLP] async rejected: {status}, "
                    f"max_viol={latest['max_viol']:.3e}"
                )
            return cached
        if latest.get("row") is not None:
            with self._async_lock:
                self._async_applied_gen = latest["generation"]
                self._async_last_row = latest["row"]
            return latest["row"]
        return cached



    @staticmethod
    def _bilinear_vec(grid: np.ndarray, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
        ny, nx = grid.shape
        gx = np.clip(gx, 0.0, float(nx - 1))
        gy = np.clip(gy, 0.0, float(ny - 1))
        i0 = np.floor(gx).astype(np.int64)
        j0 = np.floor(gy).astype(np.int64)
        i1 = np.minimum(i0 + 1, nx - 1)
        j1 = np.minimum(j0 + 1, ny - 1)
        fx = gx - i0
        fy = gy - j0
        v00 = grid[j0, i0]
        v10 = grid[j0, i1]
        v01 = grid[j1, i0]
        v11 = grid[j1, i1]
        v0 = v00 * (1.0 - fx) + v10 * fx
        v1 = v01 * (1.0 - fx) + v11 * fx
        return v0 * (1.0 - fy) + v1 * fy

    def _layer2_refine(
        self,
        maps: TerrainMaps,
        env_idx: int,
        center: np.ndarray,
        legs: Sequence[int],
        h_valid: np.ndarray | None = None,
    ) -> np.ndarray:
        """论文 V-D3: 上一轮落脚点周围 0.4 m batch search.

        候选点很少时 (4 条腿 × 0.4 m/2 cm ≈ 5.5k) GPU kernel 固定开销反而
        更慢，因此按 ``nlp_layer2_gpu_min_points`` 自适应选择 CPU/GPU。
        """
        t0 = time.perf_counter()
        r = float(getattr(self.cfg, "nlp_layer2_radius", 0.4))
        res = float(getattr(self.cfg, "nlp_layer2_resolution", 0.02))
        off = torch.arange(-r, r + 0.5 * res, res, dtype=torch.float32)
        num_offsets = int(off.numel())
        min_gpu_points = int(
            getattr(self.cfg, "nlp_layer2_gpu_min_points", 20000)
        )
        cfg_device = getattr(self.command, "device", torch.device("cpu"))
        device = (
            cfg_device
            if int(len(legs)) * num_offsets * num_offsets >= min_gpu_points
            else torch.device("cpu")
        )

        h = torch.as_tensor(np.asarray(maps.h.data), dtype=torch.float32, device=device)
        hs2 = torch.as_tensor(np.asarray(maps.hs2.data), dtype=torch.float32, device=device)
        ghx = torch.as_tensor(np.asarray(maps.grad_h_x.data), dtype=torch.float32, device=device)
        ghy = torch.as_tensor(np.asarray(maps.grad_h_y.data), dtype=torch.float32, device=device)
        gs1x = torch.as_tensor(np.asarray(maps.grad_hs1_x.data), dtype=torch.float32, device=device)
        gs1y = torch.as_tensor(np.asarray(maps.grad_hs1_y.data), dtype=torch.float32, device=device)
        dx = float(maps.h.dx)
        dy = float(maps.h.dy)
        ox0, oy0 = maps.h.origin
        yaw = float(maps.h.yaw)
        c, s = math.cos(yaw), math.sin(yaw)

        from myrobot_mjlab.task.DTC.tamols.map_derivatives import compute_derivatives

        hs1 = torch.as_tensor(np.asarray(maps.hs1.data), dtype=torch.float32, device=device)
        curv = compute_derivatives(hs1.unsqueeze(0), dx, dy).mean_curvature.squeeze(0)

        if h_valid is not None:
            valid = torch.as_tensor(np.asarray(h_valid), dtype=torch.float32, device=device)
        else:
            try:
                valid = (
                    self.command.h_valid[env_idx, 0]
                    .detach()
                    .to(device)
                    .T.to(torch.float32)
                )
            except Exception:
                valid = torch.ones_like(h)

        off = off.to(device)
        oy, ox = torch.meshgrid(off, off, indexing="xy")
        ox = ox.reshape(-1)
        oy = oy.reshape(-1)

        leg_idx = torch.as_tensor(list(legs), dtype=torch.long, device=device)
        centers = torch.as_tensor(center, dtype=torch.float32, device=device)[leg_idx]
        # [L, K]
        xw = centers[:, 0:1] + ox.unsqueeze(0)
        yw = centers[:, 1:2] + oy.unsqueeze(0)
        dxw = xw - ox0
        dyw = yw - oy0
        xm = c * dxw + s * dyw
        ym = -s * dxw + c * dyw
        gx = xm / dx
        gy = ym / dy

        def sample(grid: torch.Tensor) -> torch.Tensor:
            ny, nx = grid.shape
            gxc = torch.clamp(gx, 0.0, float(nx - 1))
            gyc = torch.clamp(gy, 0.0, float(ny - 1))
            i0 = torch.floor(gxc).long().clamp(0, nx - 1)
            j0 = torch.floor(gyc).long().clamp(0, ny - 1)
            i1 = (i0 + 1).clamp(max=nx - 1)
            j1 = (j0 + 1).clamp(max=ny - 1)
            fx = gxc - i0.to(torch.float32)
            fy = gyc - j0.to(torch.float32)
            v00 = grid[j0, i0]
            v10 = grid[j0, i1]
            v01 = grid[j1, i0]
            v11 = grid[j1, i1]
            v0 = v00 * (1.0 - fx) + v10 * fx
            v1 = v01 * (1.0 - fx) + v11 * fx
            return v0 * (1.0 - fy) + v1 * fy

        with torch.no_grad():
            hh = sample(h)
            hh2 = sample(hs2)
            gh_x = sample(ghx)
            gh_y = sample(ghy)
            gs1_x = sample(gs1x)
            gs1_y = sample(gs1y)
            cc = sample(curv)
            vv = sample(valid)
            cost = (
                20.0 * (ox.unsqueeze(0) ** 2 + oy.unsqueeze(0) ** 2)
                + 3.0 * (gh_x * gh_x + gh_y * gh_y + gs1_x * gs1_x + gs1_y * gs1_y)
                + torch.clamp(cc, min=0.0) ** 2
                + 2.0 * (hh - hh2) ** 2
                + 1e3 * (vv < 0.5).to(torch.float32)
            )
            best = torch.argmin(cost, dim=1)
            rows = torch.arange(len(legs), device=device)
            bx = xw[rows, best].detach().cpu().numpy()
            by = yw[rows, best].detach().cpu().numpy()
            bz = hh[rows, best].detach().cpu().numpy()

        refined = np.asarray(center, dtype=float).copy()
        for k, leg in enumerate(legs):
            refined[int(leg)] = (float(bx[k]), float(by[k]), float(bz[k]))
        if bool(getattr(self.cfg, "nlp_verbose", False)):
            print(f"[NLP] layer2 search: {(time.perf_counter()-t0)*1e3:.2f} ms")
        return refined

    def _row_from_solution(
        self,
        *,
        ctx: PlanContext,
        solution: dict[str, np.ndarray],
        status: str,
        f: float,
        solve_seconds: float,
        env_idx: int,
        joint_pos: np.ndarray,
        durations: Sequence[float],
    ) -> dict[str, np.ndarray]:
        report = SimpleNamespace(
            success=True,
            status=status,
            solution=solution,
            warnings=(),
            solve_seconds=solve_seconds,
            objective_total=f,
            solution_source="cpp_gn_sqp",
        )
        export = export_trajectory(
            ctx,
            report,
            samples_per_phase=int(
                getattr(self.cfg, "nlp_samples_per_phase", 25)
            ),
        )
        jvel = touchdown_joint_velocities(ctx, report, export)
        plan = plan_from_report(ctx, report, export, jvel, stamp=0)

        footholds = np.asarray(plan.footholds_w, dtype=float).copy()
        if not np.isfinite(footholds).all():
            raise RuntimeError("NLP 导出的 footholds 含 NaN")

        desired_joint = joint_pos.copy()
        for leg in ctx.touchdown_legs():
            q = np.asarray(plan.touchdown_joint_pos[leg], dtype=float)
            if np.isfinite(q).all():
                desired_joint[3 * leg : 3 * leg + 3] = q

        ref0 = plan.base_reference_at(0.0)
        dt = min(0.02, max(float(sum(durations)), 1e-3))
        ref1 = plan.base_reference_at(dt)
        R0 = np.asarray(rotation_matrix_zyx(*ref0["euler_zyx"]), dtype=float)
        R1 = np.asarray(rotation_matrix_zyx(*ref1["euler_zyx"]), dtype=float)
        ang_vel_w0 = R0 @ np.asarray(ref0["ang_vel_b"], dtype=float)
        ang_vel_w1 = R1 @ np.asarray(ref1["ang_vel_b"], dtype=float)
        ang_acc_w = (ang_vel_w1 - ang_vel_w0) / dt

        self._last_footholds[env_idx] = footholds
        self._last_state[env_idx] = {
            key: np.asarray(value, dtype=float).copy()
            for key, value in plan.state.items()
        }
        return {
            "optimal_footholds_w": footholds,
            "desired_base_pos_w": np.asarray(ref0["pos_w"], dtype=float),
            "desired_base_quat_w": euler_zyx_to_quat_wxyz(ref0["euler_zyx"]),
            "desired_base_euler": np.asarray(ref0["euler_zyx"], dtype=float),
            "desired_base_lin_vel_w": np.asarray(ref0["lin_vel_w"], dtype=float),
            "desired_base_ang_vel_w": ang_vel_w0,
            "desired_base_lin_acc_w": np.asarray(ref0["lin_acc_w"], dtype=float),
            "desired_base_ang_acc_w": ang_acc_w,
            "desired_joint_pos": desired_joint,
        }

    # -- 与原 planner 相同的接口 ---------------------------------------- #
    def plan_motion(
        self,
        nominal_footholds_w: torch.Tensor,
        base_pos_w: torch.Tensor,
        base_yaw: torch.Tensor,
        base_vel_w: torch.Tensor,
        cmd_vel_b: torch.Tensor,
        base_ang_vel_w: torch.Tensor | None,
        cmd_yaw_rate: torch.Tensor | float,
        terrain_maps: torch.Tensor,
        lead_time: torch.Tensor | float | None = None,
    ) -> dict[str, torch.Tensor]:
        """对 ``replan_idx`` 子集逐个求解 NLP，返回与原 planner 同结构的 dict."""
        m = int(base_pos_w.shape[0])
        device = base_pos_w.device
        dtype = base_pos_w.dtype

        env_ids = getattr(self.command, "_last_replan_idx", None)
        if env_ids is None or len(env_ids) != m:
            raise RuntimeError(
                "NLP 适配器拿不到 replan_idx；请确认 TamolsMotionCommand "
                "在调用 plan_motion 前写入了 self._last_replan_idx"
            )

        rows: list[dict[str, np.ndarray]] = []
        solve_times: list[float] = []
        try:
            for j in range(m):
                env_idx = int(env_ids[j])
                t0 = time.perf_counter()
                row = self._solve_one(
                    j=j,
                    env_idx=env_idx,
                    nominal_w=nominal_footholds_w[j],
                    base_pos_w=base_pos_w[j],
                    base_yaw=float(base_yaw[j]),
                    base_vel_w=base_vel_w[j],
                    cmd_vel_b=cmd_vel_b[j],
                    cmd_yaw_rate=cmd_yaw_rate,
                    terrain_maps=terrain_maps[j : j + 1],
                )
                solve_times.append(time.perf_counter() - t0)
                rows.append(row)
        except _AsyncPending:
            return self.base_planner.plan_motion(
                nominal_footholds_w=nominal_footholds_w,
                base_pos_w=base_pos_w,
                base_yaw=base_yaw,
                base_vel_w=base_vel_w,
                cmd_vel_b=cmd_vel_b,
                base_ang_vel_w=base_ang_vel_w,
                cmd_yaw_rate=cmd_yaw_rate,
                terrain_maps=terrain_maps,
                lead_time=lead_time,
            )
        except Exception as exc:
            self._fail_count += 1
            self._last_status = f"{type(exc).__name__}: {exc}"
            if not bool(getattr(self.cfg, "nlp_fallback_to_grid", True)):
                raise
            print(
                f"[NLP] 求解失败，回退 grid: {self._last_status} "
                f"(累计失败 {self._fail_count} 次)"
            )
            return self.base_planner.plan_motion(
                nominal_footholds_w=nominal_footholds_w,
                base_pos_w=base_pos_w,
                base_yaw=base_yaw,
                base_vel_w=base_vel_w,
                cmd_vel_b=cmd_vel_b,
                base_ang_vel_w=base_ang_vel_w,
                cmd_yaw_rate=cmd_yaw_rate,
                terrain_maps=terrain_maps,
                lead_time=lead_time,
            )

        self._last_solve_s = float(np.mean(solve_times)) if solve_times else 0.0
        if bool(getattr(self.cfg, "nlp_verbose", False)):
            print(
                f"[NLP] batch={m} mean={self._last_solve_s * 1e3:.0f} ms "
                f"last={solve_times[-1] * 1e3 if solve_times else 0:.0f} ms"
            )

        def stack(key: str, shape: tuple[int, ...]) -> torch.Tensor:
            arr = np.stack([r[key] for r in rows], axis=0)
            return torch.as_tensor(arr, device=device, dtype=dtype).reshape(
                (m, *shape)
            )

        return {
            "optimal_footholds_w": stack("optimal_footholds_w", (4, 3)),
            "desired_base_pos_w": stack("desired_base_pos_w", (3,)),
            "desired_base_quat_w": stack("desired_base_quat_w", (4,)),
            "desired_base_euler": stack("desired_base_euler", (3,)),
            "desired_base_lin_vel_w": stack("desired_base_lin_vel_w", (3,)),
            "desired_base_ang_vel_w": stack("desired_base_ang_vel_w", (3,)),
            "desired_base_lin_acc_w": stack("desired_base_lin_acc_w", (3,)),
            "desired_base_ang_acc_w": stack("desired_base_ang_acc_w", (3,)),
            "desired_joint_pos": stack("desired_joint_pos", (12,)),
        }

    # -- 单环境求解 ------------------------------------------------------- #


    @staticmethod
    def _virtual_maps(maps: TerrainMaps) -> TerrainMaps:
        """Layer 1 的虚拟地面: h = h_s1 = h_s2 = 原 h_s2."""
        hs2_np = np.asarray(maps.hs2.data, dtype=float)
        return TerrainMaps(
            hs2_np,
            hs2_np,
            hs2_np,
            dx=float(maps.h.dx),
            dy=float(maps.h.dy),
            origin=tuple(float(v) for v in maps.h.origin),
            yaw=float(maps.h.yaw),
        )

    @staticmethod
    def _make_cpp_problem(
        *,
        maps: TerrainMaps,
        ctx: PlanContext,
        params: Any,
        assumptions: AssumptionsConfig,
        ref_vx: float,
        ref_vy: float,
        ref_wz: float,
        previous_footholds: np.ndarray | None,
        include_dynamics: bool = True,
        include_kinematic: bool = True,
    ) -> Any:
        from myrobot_mjlab.task.DTC.tamols.cpp_solver import (
            make_problem_from_data,
        )

        return make_problem_from_data(
            durations=ctx.durations,
            schedules=ctx.schedules,
            maps=maps,
            params=params,
            assumptions=assumptions,
            td_legs=ctx.touchdown_legs(),
            ref_vx=ref_vx,
            ref_vy=ref_vy,
            ref_wz=ref_wz,
            previous_footholds=previous_footholds,
            include_dynamics=include_dynamics,
            include_kinematic=include_kinematic,
            num_coeffs=ctx.num_coeffs,
            use_slack=ctx.use_slack,
        )

    def _solve_real_truncated(
        self,
        cpp: Any,
        x0: np.ndarray,
        x0_init: np.ndarray,
    ) -> tuple[np.ndarray, float, int, str, float]:
        """论文式截断 GN-SQP: 先 1~5 iter，不够再从当前点继续."""
        limit = float(getattr(self.cfg, "nlp_max_violation", 5e-3))
        first = max(
            1,
            min(
                int(getattr(self.cfg, "nlp_real_first_iter", 20)),
                int(getattr(self.cfg, "nlp_real_max_iter", 20)),
            ),
        )

        def solve(start: np.ndarray, n_iter: int):
            return cpp.solve(
                start,
                max_iter=n_iter,
                tol_step=1e-6,
                tol_viol=1e-4,
                tol_stationary=1e-4,
                qp_lin_tol=1e-4,
                hessian_reg=1e-8,
            )

        x, f, it, status, max_viol = solve(x0, first)
        if "qp_infeasible" in status and not np.allclose(x0, x0_init):
            x, f, it, status, max_viol = solve(x0_init, first)
        if _accepted_status(status, max_viol, limit):
            return x, f, it, status, max_viol

        if "qp_failed" in status or "qp_infeasible" in status:
            # QP 数值失败后从当前点继续通常立刻再次失败；换解析初值重试.
            if np.allclose(x0, x0_init):
                return x, f, it, status, max_viol
            x, f, it, status, max_viol = solve(
                x0_init, int(getattr(self.cfg, "nlp_real_max_iter", 20))
            )
        elif first < int(getattr(self.cfg, "nlp_real_max_iter", 20)):
            # 首轮截断不够：从当前点继续跑满首轮预算.
            x2, f2, it2, st2, v2 = solve(
                x, int(getattr(self.cfg, "nlp_real_max_iter", 20))
            )
            x, f, status, max_viol = x2, f2, st2, v2
            it += it2
        if _accepted_status(status, max_viol, limit):
            return x, f, it, status, max_viol

        x2, f2, it2, st2, v2 = solve(
            x, int(getattr(self.cfg, "nlp_real_max_iter2", 40))
        )
        return x2, f2, it + it2, st2, v2

    def _solve_one(
        self,
        *,
        j: int,
        env_idx: int,
        nominal_w: torch.Tensor,
        base_pos_w: torch.Tensor,
        base_yaw: float,
        base_vel_w: torch.Tensor,
        cmd_vel_b: torch.Tensor,
        cmd_yaw_rate: torch.Tensor | float,
        terrain_maps: torch.Tensor,
    ) -> dict[str, np.ndarray]:
        cmd = self.command
        robot = cmd.robot
        gait = cmd.gait_scheduler

        # ---- 当前状态 ---------------------------------------------------
        base_pos = base_pos_w.detach().cpu().numpy().astype(float)
        base_quat = robot.data.root_link_quat_w[env_idx].detach().cpu().numpy()
        euler = quat_wxyz_to_euler_zyx(base_quat)
        omega_b = robot.data.root_link_ang_vel_b[env_idx].detach().cpu().numpy()
        euler_rates = euler_rates_from_body_omega(euler, omega_b)
        base_twist0 = np.concatenate(
            [
                base_vel_w.detach().cpu().numpy().astype(float),
                euler_rates,
            ]
        )

        foot_ids = cmd._foot_body_ids()
        foot_pos = (
            robot.data.body_link_pos_w[env_idx, foot_ids, :]
            .detach()
            .cpu()
            .numpy()
            .astype(float)
        )
        joint_pos = robot.data.joint_pos[env_idx].detach().cpu().numpy().astype(float)

        contact = gait.contact_states[env_idx].detach().cpu().numpy() > 0.5

        stride = float(cmd.cfg.gait_cfg.stride_duration)
        duty = float(cmd.cfg.gait_cfg.duty_factor)
        stance_time = max(stride * duty, 0.02)
        swing_time = max(stride - stance_time, 0.02)
        # 用完整的一个 trot 半周期作为两个相位，避免 time_left 接近 0 时
        # 生成 0.02 s 的极短相位导致 NLP 病态。NLP 只负责一个周期内的
        # 局部规划，当前相位剩余时间由上层 gait scheduler 继续推进。
        durations = [stance_time, swing_time]

        # 2 相 trot: 当前支撑腿在 phase0 支撑、phase1 摆动；
        # 当前摆动腿在 phase0 摆动、phase1 触地。
        schedules: dict[int, LegSchedule] = {}
        measured_feet: dict[int, tuple[float, float, float]] = {}
        for leg in range(4):
            if contact[leg]:
                schedules[leg] = LegSchedule(
                    grounded_at_start=True,
                    lift_phase=1,
                    measured_position=tuple(float(v) for v in foot_pos[leg]),
                )
                measured_feet[leg] = tuple(float(v) for v in foot_pos[leg])
            else:
                schedules[leg] = LegSchedule(
                    grounded_at_start=False,
                    touchdown_phase=1,
                )

        # ---- 局部高程图 -> TerrainMaps ---------------------------------
        h_use = terrain_maps[0:1, 0:1]  # [1,1,Nx,Ny]
        h_s2 = terrain_maps[0, 1]       # [Nx,Ny]
        h_s1_b, _, _ = cmd.map_processor(h_use)
        h_np = h_use[0, 0].T.detach().cpu().numpy().astype(float)
        hs1_np = h_s1_b[0, 0].T.detach().cpu().numpy().astype(float)
        hs2_np = h_s2.T.detach().cpu().numpy().astype(float)

        cos_y = math.cos(base_yaw)
        sin_y = math.sin(base_yaw)
        origin = (
            float(base_pos[0] + cos_y * cmd.x_min - sin_y * cmd.y_min),
            float(base_pos[1] + sin_y * cmd.x_min + cos_y * cmd.y_min),
        )
        maps = TerrainMaps(
            h_np,
            hs1_np,
            hs2_np,
            dx=float(cmd.cell_size),
            dy=float(cmd.cell_size),
            origin=origin,
            yaw=base_yaw,
        )

        # ---- 参考速度 ---------------------------------------------------
        cmd_vx = float(cmd_vel_b[0])
        cmd_vy = float(cmd_vel_b[1]) if cmd_vel_b.shape[-1] > 1 else 0.0
        if isinstance(cmd_yaw_rate, (int, float)):
            cmd_wz = float(cmd_yaw_rate)
        else:
            cmd_wz = float(cmd_yaw_rate[j])

        def ref_twist(_phase: int, _tau: float):
            return (
                np.array([cmd_vx, cmd_vy, 0.0], dtype=float),
                np.array([0.0, 0.0, cmd_wz], dtype=float),
            )

        # ---- 初值: Raibert 标称点 + 当前机身状态 ------------------------
        nominal_np = nominal_w.detach().cpu().numpy().astype(float).copy()
        for leg in range(4):
            if schedules[leg].touchdown_phase is None:
                nominal_np[leg] = foot_pos[leg]

        v_cmd_w = np.array(
            [
                cos_y * cmd_vx - sin_y * cmd_vy,
                sin_y * cmd_vx + cos_y * cmd_vy,
                0.0,
            ],
            dtype=float,
        )
        coeffs: list[np.ndarray] = []
        t_start = 0.0
        for phase, T in enumerate(durations):
            c = np.zeros((6, 5), dtype=float)
            c[0:3, 0] = base_pos + v_cmd_w * t_start
            c[0:3, 1] = v_cmd_w
            c[3:6, 0] = euler
            c[3:6, 1] = euler_rates
            coeffs.append(c)
            t_start += T

        blocks = [c for c in coeffs] + [
            nominal_np,
            np.zeros(len(durations), dtype=float),
        ]

        # 论文 V-C/D 的热启动: 上一次成功解的完整决策向量作为本次初值。
        # 样条是相位的世界系系数，随基座移动会略有偏差，但比零加速度
        # 初值更接近最优，能显著减少 SQP 迭代。
        prev_state = self._last_state.get(env_idx)
        if prev_state is not None:
            try:
                prev_coeffs = [
                    np.asarray(c, dtype=float).reshape(6, 5)
                    for c in prev_state["spline_coeffs"]
                ]
                if len(prev_coeffs) == len(durations):
                    blocks = prev_coeffs + [
                        np.asarray(
                            prev_state["desired_footholds"], dtype=float
                        ).reshape(4, 3),
                    ]
                    if "slack" in prev_state:
                        blocks.append(
                            np.asarray(
                                prev_state["slack"], dtype=float
                            ).reshape(len(durations))
                        )
                    else:
                        blocks.append(np.zeros(len(durations), dtype=float))
            except Exception:
                pass

        previous = self._last_footholds.get(env_idx)
        if previous is None and bool(cmd.footholds_initialized[env_idx]):
            previous = cmd.optimal_footholds[env_idx].detach().cpu().numpy().astype(float)


        params = NLPParameters(
            base_pose0=np.concatenate([base_pos, euler]),
            base_twist0=base_twist0,
            measured_feet=measured_feet,
            ref_twist=ref_twist,
            previous_footholds=previous,
            solver_options={
                "print_time": False,
                "ipopt": {
                    "print_level": 0,
                    "max_iter": 120,
                    "max_wall_time": 3.0,
                },
            },
        )
        assumptions = AssumptionsConfig(
            inertial_accel_sign=str(
                getattr(
                    self.cfg,
                    "nlp_inertial_accel_sign",
                    "contact_normal_explicit",
                )
            ),
            friction_mode=str(
                getattr(self.cfg, "nlp_friction_mode", "enabled")
            ),
            friction_cone_form=str(
                getattr(
                    self.cfg,
                    "nlp_friction_cone_form",
                    "force_balance_derived",
                )
            ),
        )
        backend = str(getattr(self.cfg, "nlp_solver_backend", "cpp"))
        async_cpp = self._async_enabled and backend == "cpp"
        ctx = PlanContext(
            durations=durations,
            schedules=schedules,
            num_coeffs=5,
            use_slack=bool(assumptions.use_giac_slack),
        )
        nlp = None
        if backend != "cpp":
            from myrobot_mjlab.task.DTC.tamols.nlp_integration import (
                IntegratedNLP,
            )

            with _CASADI_BUILD_LOCK:
                nlp = IntegratedNLP(
                    durations=durations,
                    schedules=schedules,
                    maps=maps,
                    params=params,
                    assumptions=assumptions,
                )
                nlp.add_initial_state_constraints()
                nlp.add_junction_constraints()
                nlp.add_dynamic_stability_constraints()
                nlp.add_kinematic_contact_constraints()
                nlp.add_objectives()
                nlp.finalize()

        # ---- Layer 1 (论文 V-D2): h_s2 虚拟地面上的简化 TO ----
        layer1_solution = None
        # Layer 1 目前只在同步模式下运行；异步模式下 CppAD tape 与 worker
        # 并发会冲突，后续应把 Layer 1 也移入 worker.
        if bool(getattr(self.cfg, "nlp_layer1", True)) and not self._async_enabled:
            try:
                from myrobot_mjlab.task.DTC.tamols.cpp_solver import (
                    unpack_x as _unpack_x,
                )

                cpp_v = self._make_cpp_problem(
                    maps=self._virtual_maps(maps),
                    ctx=ctx,
                    params=params,
                    assumptions=assumptions,
                    ref_vx=cmd_vx,
                    ref_vy=cmd_vy,
                    ref_wz=cmd_wz,
                    previous_footholds=previous,
                    include_dynamics=True,
                    include_kinematic=True,
                )
                x0_v = np.concatenate(
                    [np.asarray(b, dtype=float).reshape(-1) for b in blocks]
                )
                with _CPP_SOLVE_LOCK:
                    xv, fv, itv, stv, violv = cpp_v.solve(
                        x0_v,
                        max_iter=int(getattr(self.cfg, "nlp_layer1_max_iter", 20)),
                        tol_step=1e-6,
                        tol_viol=1e-3,
                        tol_stationary=1e-3,
                        qp_lin_tol=1e-3,
                        hessian_reg=1e-8,
                    )
                okv = stv.startswith("GN-SQP: converged") or (
                    "qp_failed" not in stv
                    and "qp_infeasible" not in stv
                    and violv <= 5e-3
                )
                if okv:
                    layer1_solution = _unpack_x(ctx, xv)
                    if bool(getattr(self.cfg, "nlp_verbose", False)):
                        print(
                            f"[NLP] layer1 virtual TO: it={itv} "
                            f"status={stv} viol={violv:.2e}"
                        )
            except Exception as exc:
                if bool(getattr(self.cfg, "nlp_verbose", False)):
                    print(f"[NLP] layer1 skipped: {exc}")

        # ---- Layer 2 (论文 V-D3): 上次 NLP/层1 落脚点周围 0.4 m 细化 ----
        refined = None
        if bool(getattr(self.cfg, "nlp_layer2", True)) and not self._async_enabled:
            center = (
                np.asarray(layer1_solution["desired_footholds"], dtype=float)
                if layer1_solution is not None
                else (previous if previous is not None else nominal_np)
            )
            refined = self._layer2_refine(
                maps, env_idx, center, ctx.touchdown_legs()
            )
        elif layer1_solution is not None:
            refined = np.asarray(
                layer1_solution["desired_footholds"], dtype=float
            )

        # ---- 截断 graduated sequence: 只用 Layer 1 + Layer 2 热启动真实 NLP ----
        if layer1_solution is not None:
            blocks = [
                np.asarray(c, dtype=float).reshape(6, 5)
                for c in layer1_solution["spline_coeffs"]
            ]
            blocks.append(
                refined
                if refined is not None
                else np.asarray(
                    layer1_solution["desired_footholds"], dtype=float
                ).reshape(4, 3)
            )
            if "slack" in layer1_solution:
                blocks.append(
                    np.asarray(layer1_solution["slack"], dtype=float).reshape(
                        len(durations)
                    )
                )
            else:
                blocks.append(np.zeros(len(durations), dtype=float))
        elif refined is not None:
            blocks[len(ctx.durations)] = refined

        if backend == "cpp":
            from myrobot_mjlab.task.DTC.tamols.cpp_solver import unpack_x

            # 论文 V-C/D 的 warm start: blocks 已在上面优先用上一轮
            # spline_coeffs / desired_footholds / slack 覆盖解析初值.
            x0 = np.concatenate(
                [np.asarray(b, dtype=float).reshape(-1) for b in blocks]
            )
            x0_init = np.concatenate(
                [np.asarray(c, dtype=float).reshape(-1) for c in coeffs]
                + [np.asarray(nominal_np, dtype=float).reshape(-1),
                   np.zeros(len(durations), dtype=float)]
            )
            if async_cpp:
                # 首次 reset 还没有 grid 初始解，NLP 往往不可行；先让 grid
                # 完成初始化，下一次 replan 再投递异步 NLP 任务。
                if not bool(self.command.footholds_initialized[env_idx]):
                    raise _AsyncPending()
                try:
                    h_valid = (
                        self.command.h_valid[env_idx, 0]
                        .detach()
                        .cpu()
                        .numpy()
                        .T.astype(float)
                        .copy()
                    )
                except Exception:
                    h_valid = None
                self._submit_async(
                    maps=maps,
                    params=params,
                    schedules=schedules,
                    assumptions=assumptions,
                    x0=x0,
                    x0_init=x0_init,
                    ref_vx=cmd_vx,
                    ref_vy=cmd_vy,
                    ref_wz=cmd_wz,
                    center=previous if previous is not None else nominal_np,
                    legs=tuple(
                        leg for leg in range(4)
                        if schedules[leg].touchdown_leg
                    ),
                    h_valid=h_valid,
                    env_idx=env_idx,
                    joint_pos=joint_pos,
                    durations=durations,
                )
                row = self._latest_async_row()
                if row is not None:
                    return row
                raise _AsyncPending()

            cpp = self._make_cpp_problem(
                maps=maps,
                ctx=ctx,
                params=params,
                assumptions=assumptions,
                ref_vx=cmd_vx,
                ref_vy=cmd_vy,
                ref_wz=cmd_wz,
                previous_footholds=previous,
            )
            t_solve = time.perf_counter()
            x_out, f_out, iters, status, max_viol = (
                self._solve_real_truncated(cpp, x0, x0_init)
            )
            solve_seconds = time.perf_counter() - t_solve
            accepted = _accepted_status(
                status,
                max_viol,
                float(getattr(self.cfg, "nlp_max_violation", 5e-3)),
            )
            if not accepted:
                raise RuntimeError(f"C++ {status}, max_viol={max_viol:.3e}")
            if bool(getattr(self.cfg, "nlp_compare_python", False)):
                try:
                    from myrobot_mjlab.task.DTC.tamols.gn_sqp_solver import (
                        GNSQPOptions,
                        solve_gn_sqp,
                    )

                    rep_py = solve_gn_sqp(
                        nlp,
                        blocks,
                        GNSQPOptions(
                            max_iter=int(getattr(self.cfg, "nlp_max_iter", 60)),
                            tol_viol=1e-4,
                            qp_lin_tol=1e-4,
                            tol_stationary=1e-4,
                            tol_step=1e-6,
                        ),
                    )
                    py_x = np.concatenate(
                        [np.asarray(c).reshape(-1) for c in rep_py.solution["spline_coeffs"]]
                        + [np.asarray(rep_py.solution["desired_footholds"]).reshape(-1)]
                        + (
                            [np.asarray(rep_py.solution["slack"]).reshape(-1)]
                            if "slack" in rep_py.solution
                            else []
                        )
                    )
                    cc = len(durations) * 6 * ctx.num_coeffs
                    dd = np.abs(x_out - py_x)
                    print(
                        f"[NLP] compare cpp={status} py={rep_py.status} "
                        f"max|x|={dd.max():.3e} "
                        f"coeff={dd[:cc].max():.3e} "
                        f"foot={dd[cc:cc+12].max():.3e} "
                        f"slack={dd[cc+12:].max():.3e} "
                        f"f_cpp={f_out:.6g} f_py={rep_py.objective_total:.6g}"
                    )
                except Exception as exc:
                    print(f"[NLP] compare python failed: {exc}")

            return self._row_from_solution(
                ctx=ctx,
                solution=unpack_x(ctx, x_out),
                status=status,
                f=f_out,
                solve_seconds=solve_seconds,
                env_idx=env_idx,
                joint_pos=joint_pos,
                durations=durations,
            )
        else:
            solver = str(getattr(self.cfg, "nlp_solver", "gn_sqp"))
            if solver == "ipopt":
                nlp.set_initial_guess(coeffs, nominal_np, blocks[-1])
                report = nlp.solve("dtc")
            else:
                opts = GNSQPOptions(
                    max_iter=int(getattr(self.cfg, "nlp_max_iter", 60)),
                    jit=bool(getattr(self.cfg, "nlp_jit", False)),
                    use_extract_cache=True,
                    tol_viol=1e-4,
                    qp_lin_tol=1e-4,
                    tol_stationary=1e-4,
                    tol_step=1e-6,
                )
                report = solve_gn_sqp(nlp, blocks, opts)

        if not report.success and bool(
            getattr(self.cfg, "nlp_accept_last_iterate", True)
        ):
            # GN-SQP 在真实地图上常在接近可行时触发 max_iter；若最后迭代点
            # 的约束违反已经很小，接受它作为数值解，避免整步回退 grid。
            max_viol = max(
                report.max_violation_by_category.values(), default=float("inf")
            )
            limit = float(getattr(self.cfg, "nlp_max_violation", 5e-3))
            if report.solution and max_viol <= limit:
                report = replace(
                    report,
                    success=True,
                    solution_source="last_iterate_accepted",
                )
                if bool(getattr(self.cfg, "nlp_verbose", False)):
                    print(
                        f"[NLP] 接受最后迭代点: {report.status}, "
                        f"max_viol={max_viol:.3g} <= {limit:.3g}"
                    )

        if not report.success:
            if bool(getattr(self.cfg, "nlp_verbose", False)):
                print("[NLP] status:", report.status)
                print("[NLP] violations:", report.max_violation_by_category)
                print("[NLP] warnings:", report.warnings)
                worst = sorted(
                    report.constraint_lines,
                    key=lambda x: x.violation,
                    reverse=True,
                )[:10]
                for line in worst:
                    print(
                        f"[NLP]   {line.name}: viol={line.violation:.4g} "
                        f"cat={line.category} src={line.source}"
                    )
            raise RuntimeError(f"NLP 未收敛: {report.status}")

        with _CASADI_BUILD_LOCK:
            export = export_trajectory(
                nlp,
                report,
                samples_per_phase=int(
                    getattr(self.cfg, "nlp_samples_per_phase", 25)
                ),
            )
            jvel = touchdown_joint_velocities(nlp, report, export)
            plan = plan_from_report(nlp, report, export, jvel, stamp=0)

        footholds = np.asarray(plan.footholds_w, dtype=float).copy()
        if not np.isfinite(footholds).all():
            raise RuntimeError("NLP 导出的 footholds 含 NaN")

        desired_joint = joint_pos.copy()
        for leg in nlp.touchdown_legs():
            q = np.asarray(plan.touchdown_joint_pos[leg], dtype=float)
            if np.isfinite(q).all():
                desired_joint[3 * leg : 3 * leg + 3] = q

        ref0 = plan.base_reference_at(0.0)
        dt = min(0.02, max(float(sum(durations)), 1e-3))
        ref1 = plan.base_reference_at(dt)
        R0 = np.asarray(
            rotation_matrix_zyx(*ref0["euler_zyx"]), dtype=float
        )
        R1 = np.asarray(
            rotation_matrix_zyx(*ref1["euler_zyx"]), dtype=float
        )
        ang_vel_w0 = R0 @ np.asarray(ref0["ang_vel_b"], dtype=float)
        ang_vel_w1 = R1 @ np.asarray(ref1["ang_vel_b"], dtype=float)
        ang_acc_w = (ang_vel_w1 - ang_vel_w0) / dt

        self._last_footholds[env_idx] = footholds
        self._last_state[env_idx] = {
            key: np.asarray(value, dtype=float).copy()
            for key, value in plan.state.items()
        }

        return {
            "optimal_footholds_w": footholds,
            "desired_base_pos_w": np.asarray(ref0["pos_w"], dtype=float),
            "desired_base_quat_w": euler_zyx_to_quat_wxyz(ref0["euler_zyx"]),
            "desired_base_euler": np.asarray(ref0["euler_zyx"], dtype=float),
            "desired_base_lin_vel_w": np.asarray(ref0["lin_vel_w"], dtype=float),
            "desired_base_ang_vel_w": ang_vel_w0,
            "desired_base_lin_acc_w": np.asarray(ref0["lin_acc_w"], dtype=float),
            "desired_base_ang_acc_w": ang_acc_w,
            "desired_joint_pos": desired_joint,
        }
