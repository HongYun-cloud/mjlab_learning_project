"""TAMOLS 与 RL 训练回路的桥接层 (接口先行, obs/reward 之后接入).

边界与定位 (用户约定, 实现必须遵守)
==================================
1. TAMOLS 的输出是**局部、有限时域**的运动规划: 一个步态周期内的
   基座轨迹与位姿 + 四条腿的落脚点 + 触地时刻的期望关节位置/速度。
   它**不是**从起点到终点的全局路径规划器; 全局导航由上游 (RL 策略/
   命令层) 负责。
2. 在 RL 中, TAMOLS 求解运行在**并行且独立**的线程 (每个环境实例一个
   :class:`PlanningWorker`); 训练主循环**永不阻塞等待求解**。
3. 更新触发: 以**腿触地**为主; 跌倒 / 推搡 (外力扰动) / 速度指令变化
   等事件也触发重解 (事件由 RL 侧通过 :meth:`PlanningWorker.notify`
   投递, 事件队列合并)。
4. **两次更新之间, 策略使用最新的规划解**:
   :meth:`PlanningWorker.latest_plan` 无锁语义上返回最近一次**完成**的
   快照 (单调 stamp), 没有新解时保持不变; 求解失败时保留上一份有效解
   并在 ``status`` 记录失败原因。

发布契约 (:class:`TAMOLSPlan`)
==============================
字段 (全部为世界系/机体系明确标注的 numpy 数组, 单位见下)::

    基座:  t [s], pos_w [m], euler_zyx [rad], lin_vel_w [m/s],
           lin_vel_b [m/s], ang_vel_b [rad/s], lin_acc_w [m/s^2]
           —— 可由 :meth:`TAMOLSPlan.base_reference_at(t)` 在时域内
           任意时刻**精确**求值 (样条, 非网格插值);
    落脚:  footholds_w [4,3] [m], touchdown_times [4] [s] (NaN=horizon 内不触地),
    关节:  touchdown_joint_pos [4,3] [rad], touchdown_joint_vel [4,3] [rad/s]
           (关节顺序 [hip, thigh, calf] 逐腿, FL/FR/RL/RR, 与工程动作空间一致);
    接触:  contact_sets (逐相位的支撑腿集合);
    元数据: stamp / status / success / warnings / horizon;
    热启动: state (决策变量块, 供下一次求解 warm start)。

obs / reward 预留接口 (本次不接入)
==================================
- :class:`PlanObserver`: ``on_plan(plan)`` (新解发布时) 与
  ``observe(t, plan) -> dict`` (每步构造观测项) —— RL 侧实现后
  ``worker.attach_observer(...)`` 即可, 桥接层不做任何策略耦合;
- :class:`RewardHook`: ``on_event(event, plan)`` (事件发生时, 含产生
  的计划或 None) —— 同上, 仅登记与回调。

线程模型
========
重负载 (符号求值/QP) 在 CasADi/QuadProg++ 的 C 代码中执行并释放 GIL,
因此每环境一个 :class:`PlanningWorker` 线程可有效并行; 策略线程与
worker 之间只通过 ``latest_plan`` 快照通信 (双缓冲语义, 无共享可变
训练状态).
"""

from __future__ import annotations

import queue
import threading
from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Sequence

import numpy as np

from myrobot_mjlab.task.DTC.tamols.gn_sqp_solver import (
    GNSQPOptions,
    SolveReport,
    solve_gn_sqp,
)
from myrobot_mjlab.task.DTC.tamols.nlp_integration import IntegratedNLP
from myrobot_mjlab.task.DTC.tamols.trajectory_export import (
    TrajectoryExport,
    export_trajectory,
    touchdown_joint_velocities,
)
from myrobot_mjlab.task.DTC.tamols.v_a_constraints import (
    rotation_matrix_zyx,
    spline_value,
)
from myrobot_mjlab.task.DTC.tamols.v_b_objectives import body_omega_zyx

PlanEventKind = Literal["touchdown", "fall", "push", "cmd_change"]

EVENT_KINDS: tuple[str, ...] = (
    "touchdown", "fall", "push", "cmd_change",
)
"""合法的求解触发事件: 触地(主) / 跌倒 / 推搡 / 速度指令变化."""


# ---------------------------------------------------------------------- #
# 发布契约
# ---------------------------------------------------------------------- #
@dataclass(frozen=True)
class TAMOLSPlan:
    """一次求解发布的最新的局部有限时域规划 (不可变快照)."""

    stamp: int
    """单调递增发布序号; 策略线程以此判断是否有新解."""

    status: str
    success: bool
    horizon: tuple[float, float]
    """时域 [t0, t1] [s] (相对求解发布时刻, 长度 = 一个步态周期)."""

    t: np.ndarray
    base_pos_w: np.ndarray
    base_euler_zyx: np.ndarray
    base_lin_vel_w: np.ndarray
    base_lin_vel_b: np.ndarray
    base_ang_vel_b: np.ndarray
    base_lin_acc_w: np.ndarray

    footholds_w: np.ndarray
    touchdown_times: np.ndarray
    touchdown_joint_pos: np.ndarray
    touchdown_joint_vel: np.ndarray

    contact_sets: tuple[tuple[int, ...], ...]
    spline_coeffs: np.ndarray
    durations: tuple[float, ...]

    state: dict[str, np.ndarray]
    """决策变量块 (spline_coeffs / desired_footholds / slack), 供下次热启动."""

    warnings: tuple[str, ...] = ()
    event: str = ""
    solve_seconds: float = 0.0

    @property
    def t_end(self) -> float:
        return self.horizon[1]

    def base_reference_at(self, t: float) -> dict[str, np.ndarray]:
        """时域内任意时刻的基座参考 (样条精确求值, 越界截断到端点).

        返回 ``{"pos_w", "euler_zyx", "lin_vel_w", "lin_vel_b",
        "ang_vel_b", "lin_acc_w"}`` (单位见模块文档).
        """
        tc = min(max(float(t), self.horizon[0]), self.horizon[1])
        starts = np.concatenate(([0.0], np.cumsum(self.durations)))
        phase = int(np.searchsorted(starts[1:], tc, side="right"))
        phase = min(phase, len(self.durations) - 1)
        tau = tc - starts[phase]
        c = self.spline_coeffs[phase]
        p0 = np.array([float(v) for v in spline_value(c, tau, 0)])
        p1 = np.array([float(v) for v in spline_value(c, tau, 1)])
        p2 = np.array([float(v) for v in spline_value(c, tau, 2)])
        R = np.array(rotation_matrix_zyx(p0[3], p0[4], p0[5]))
        return {
            "pos_w": p0[0:3],
            "euler_zyx": p0[3:6],
            "lin_vel_w": p1[0:3],
            "lin_vel_b": R.T @ p1[0:3],
            "ang_vel_b": np.array(
                [float(v) for v in body_omega_zyx(p0[3:6], p1[3:6])]
            ),
            "lin_acc_w": p2[0:3],
        }

    def foothold_for_leg(self, leg: int) -> np.ndarray:
        """该腿在 horizon 内的期望落脚点 [3] [m] (不触地腿 = 当前测量位)."""
        return self.footholds_w[leg]

    def joint_target_for_leg(self, leg: int) -> tuple[np.ndarray, np.ndarray]:
        """触地时刻期望关节 (位置 [rad], 速度 [rad/s]); 不触地腿为 NaN."""
        return self.touchdown_joint_pos[leg], self.touchdown_joint_vel[leg]


def plan_from_report(
    nlp: IntegratedNLP,
    report: SolveReport,
    export: TrajectoryExport | None,
    joint_vel: np.ndarray | None,
    *,
    stamp: int,
    event: str = "",
) -> TAMOLSPlan:
    """由求解报告 (+导出/关节速度) 构造发布快照; 失败时构造占位快照.

    失败 (或未成功) 时不携带轨迹数据 (调用方/worker 保留上一份有效解),
    但 ``state`` 仍给出 (最后迭代点仅用于诊断, 不作热启动来源——由
    worker 侧保证只把成功解作为下一次 warm start).
    """
    t0, t1 = 0.0, float(sum(nlp.durations))
    if report.success and export is not None:
        return TAMOLSPlan(
            stamp=stamp,
            status=report.status,
            success=True,
            horizon=(t0, t1),
            t=export.t,
            base_pos_w=export.base_pos_w,
            base_euler_zyx=export.base_euler_zyx,
            base_lin_vel_w=export.base_lin_vel_w,
            base_lin_vel_b=export.base_lin_vel_b,
            base_ang_vel_b=export.base_ang_vel_b,
            base_lin_acc_w=export.base_lin_acc_w,
            footholds_w=export.footholds_w,
            touchdown_times=export.touchdown_times,
            touchdown_joint_pos=export.touchdown_joint_pos,
            touchdown_joint_vel=joint_vel,
            contact_sets=tuple(export.contact_sets),
            spline_coeffs=report.solution["spline_coeffs"],
            durations=tuple(nlp.durations),
            state=dict(report.solution),
            warnings=tuple(report.warnings),
            event=event,
            solve_seconds=report.solve_seconds,
        )
    return TAMOLSPlan(
        stamp=stamp,
        status=report.status,
        success=False,
        horizon=(t0, t1),
        t=np.zeros(0),
        base_pos_w=np.zeros((0, 3)),
        base_euler_zyx=np.zeros((0, 3)),
        base_lin_vel_w=np.zeros((0, 3)),
        base_lin_vel_b=np.zeros((0, 3)),
        base_ang_vel_b=np.zeros((0, 3)),
        base_lin_acc_w=np.zeros((0, 3)),
        footholds_w=np.full((4, 3), np.nan),
        touchdown_times=np.full(4, np.nan),
        touchdown_joint_pos=np.full((4, 3), np.nan),
        touchdown_joint_vel=np.full((4, 3), np.nan),
        contact_sets=tuple(),
        spline_coeffs=np.zeros((nlp.layout.num_phases, 6, nlp.layout.num_coeffs)),
        durations=tuple(nlp.durations),
        state={},
        warnings=tuple(report.warnings),
        event=event,
        solve_seconds=report.solve_seconds,
    )


# ---------------------------------------------------------------------- #
# 求解函数工厂 (桥接层不关心 NLP 如何构造; 这里提供默认接线)
# ---------------------------------------------------------------------- #
def make_default_solve_fn(
    nlp: IntegratedNLP,
    initial_blocks: Sequence[np.ndarray],
    opts: GNSQPOptions | None = None,
    samples_per_phase: int = 25,
) -> Callable[[TAMOLSPlan | None, str, Any], TAMOLSPlan]:
    """构造 ``solve_fn(warm_plan, event, payload) -> TAMOLSPlan``.

    - ``warm_plan`` 为上一份**成功**计划时, 用其 ``state`` 热启动
      (重规划场景, 论文 V-C/D 的 warm start);
    - 求解成功: 导出基座轨迹 + 落脚点 + 触地关节位置/速度 -> 计划;
    - 求解失败: 返回 ``success=False`` 占位计划 (worker 保留旧解).
    """
    initial = [np.asarray(b, dtype=float) for b in initial_blocks]

    def solve_fn(warm: TAMOLSPlan | None, event: str = "", payload: Any = None) -> TAMOLSPlan:
        if warm is not None and warm.success and warm.state:
            blocks = [
                np.asarray(c, dtype=float)
                for c in warm.state["spline_coeffs"]
            ] + [
                np.asarray(warm.state["desired_footholds"], dtype=float)
            ]
            if nlp.layout.use_slack:
                blocks.append(
                    np.asarray(warm.state["slack"], dtype=float)
                )
        else:
            blocks = list(initial)
        report = solve_gn_sqp(nlp, blocks, opts)
        if report.success:
            export = export_trajectory(nlp, report, samples_per_phase)
            jvel = touchdown_joint_velocities(nlp, report, export)
        else:
            export, jvel = None, None
        return plan_from_report(
            nlp, report, export, jvel, stamp=0, event=event
        )

    return solve_fn


# ---------------------------------------------------------------------- #
# obs / reward 预留接口 (本次不接入; RL 侧实现后 attach 即可)
# ---------------------------------------------------------------------- #
class PlanObserver(ABC):
    """RL 观测接口 (预留): 从最新计划构造观测项."""

    def on_plan(self, plan: TAMOLSPlan) -> None:
        """新计划发布时回调 (默认无操作)."""

    def observe(
        self, t: float, plan: TAMOLSPlan | None
    ) -> dict[str, Any]:
        """每个控制步由策略线程调用, 返回观测项 (默认空)."""
        return {}


class RewardHook(ABC):
    """RL 奖励接口 (预留): 事件/计划驱动的奖励项登记."""

    def on_event(self, event: str, plan: TAMOLSPlan | None) -> None:
        """求解事件回调 (默认无操作): 'touchdown'/'fall'/'push'/
        'cmd_change'; ``plan`` 为事件处理后的最新计划 (可能为 None)."""


# ---------------------------------------------------------------------- #
# 并行独立线程的求解工作器
# ---------------------------------------------------------------------- #
class PlanningWorker:
    """每环境一个的 TAMOLS 求解线程 (事件触发, 双缓冲发布).

    用法::

        worker = PlanningWorker(make_default_solve_fn(nlp, blocks, opts))
        worker.start()
        worker.notify("touchdown")            # 腿触地 (训练中主要触发)
        worker.notify("push", {"f": ...})     # 推搡/跌倒/指令变化同理
        plan = worker.latest_plan()           # 策略线程: 两次更新之间
                                              # 始终拿到最近一次完成的解
        ...
        worker.stop()

    语义:
      - 事件队列**合并**: worker 处理一个事件期间到达的同类事件被吸收,
        处理完后只保留一次重解 (求解 ~10ms 级, 触地频率 ~Hz, 不堆积);
      - 失败保留旧解: 求解异常/失败时 ``latest_plan`` 不变,
        失败信息进入 worker.last_error 与失败计划的 status;
      - ``latest_plan`` 在锁内返回引用 (TAMOLSPlan 不可变), 策略线程
        可安全持有.
    """

    def __init__(
        self,
        solve_fn: Callable[[TAMOLSPlan | None, str, Any], TAMOLSPlan],
        *,
        name: str = "tamols-planner",
        max_pending_events: int = 16,
    ):
        self._solve_fn = solve_fn
        self._events: "queue.Queue[tuple[str, Any]]" = queue.Queue(
            maxsize=max_pending_events
        )
        self._lock = threading.Lock()
        self._latest: TAMOLSPlan | None = None
        self._stamp = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_error: str | None = None
        self.observers: list[PlanObserver] = []
        self.reward_hooks: list[RewardHook] = []
        self.name = name

    # -- 生命周期 -------------------------------------------------------- #
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("worker 已启动")
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name=self.name, daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # -- 策略线程接口 ----------------------------------------------------- #
    def notify(self, event: PlanEventKind | str, payload: Any = None) -> None:
        """投递重解触发事件 (非阻塞; 队列满时丢弃——最新状态总是由
        下一个事件补上, 见类文档的合并语义)."""
        if event not in EVENT_KINDS:
            raise ValueError(
                f"未知事件 {event!r}; 可选 {EVENT_KINDS}"
            )
        try:
            self._events.put_nowait((event, payload))
        except queue.Full:
            pass

    def latest_plan(self) -> TAMOLSPlan | None:
        """最近一次**完成**的计划 (两次更新之间策略使用它; 无则 None)."""
        with self._lock:
            return self._latest

    @property
    def stamp(self) -> int:
        with self._lock:
            return self._stamp

    # -- obs/reward 登记 -------------------------------------------------- #
    def attach_observer(self, observer: PlanObserver) -> None:
        self.observers.append(observer)

    def attach_reward_hook(self, hook: RewardHook) -> None:
        self.reward_hooks.append(hook)

    # -- worker 线程主体 --------------------------------------------------- #
    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                event, payload = self._events.get(timeout=0.02)
            except queue.Empty:
                continue
            if self._stop.is_set():
                break
            plan = self._solve_once(event, payload)
            with self._lock:
                self._stamp += 1
                plan = _restamp(plan, self._stamp)
                if plan.success:
                    self._latest = plan
                self.last_error = None if plan.success else plan.status
            for hook in self.reward_hooks:
                hook.on_event(event, self._latest)
            if plan.success:
                for obs in self.observers:
                    obs.on_plan(plan)
            # 合并处理期间堆积的后续事件 (只保留最新一次触发)
            self._drain_events()

    def _solve_once(self, event: str, payload: Any) -> TAMOLSPlan:
        warm = self._latest if (self._latest and self._latest.success) else None
        try:
            return self._solve_fn(warm, event, payload)
        except Exception as exc:  # 求解器异常: 保留旧解, 记录错误
            import time as _time

            return TAMOLSPlan(
                stamp=0,
                status=f"worker_error: {type(exc).__name__}: {exc}",
                success=False,
                horizon=(0.0, 0.0),
                t=np.zeros(0),
                base_pos_w=np.zeros((0, 3)),
                base_euler_zyx=np.zeros((0, 3)),
                base_lin_vel_w=np.zeros((0, 3)),
                base_lin_vel_b=np.zeros((0, 3)),
                base_ang_vel_b=np.zeros((0, 3)),
                base_lin_acc_w=np.zeros((0, 3)),
                footholds_w=np.full((4, 3), np.nan),
                touchdown_times=np.full(4, np.nan),
                touchdown_joint_pos=np.full((4, 3), np.nan),
                touchdown_joint_vel=np.full((4, 3), np.nan),
                contact_sets=tuple(),
                spline_coeffs=np.zeros((0, 0, 0)),
                durations=(),
                state={},
                warnings=(),
                event=event,
                solve_seconds=0.0,
            )

    def _drain_events(self) -> None:
        drained = 0
        while True:
            try:
                self._events.get_nowait()
                drained += 1
            except queue.Empty:
                break


def _restamp(plan: TAMOLSPlan, stamp: int) -> TAMOLSPlan:
    if plan.stamp == stamp:
        return plan
    from dataclasses import replace

    return replace(plan, stamp=stamp)
