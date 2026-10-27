"""DTC 奖励项: Base Pose Tracking (对 TAMOLS 参考轨迹).

用户给定公式与语义
==================::

    r_{B_n} = exp( -sigma_{B_n} * || b*(t + dt) ⊖ b(t) ||_2 )

- ``n = order ∈ {0, 1, 2}`` 为导数阶数: 0 = 位姿, 1 = 速度, 2 = 加速度;
- ``b*(t + dt)``: 参考轨迹在当前时刻 + dt 处的期望基座状态。TAMOLS 命令项
  (fetch 规划器) 只提供上次重规划时刻的冻结解 (``desired_base_*``), 本模块
  用命令项维护的 ``time_since_replan`` (重规划清零, 每步累加) 把它外推成
  **随时间流动的参考** (``T = time_since_replan + dt``)::
      p*(t+dt) = p* + v* T                         (xy 常速流动, z 保持地形高度)
      q*(t+dt) = q* ⊗ exp(omega* T / 2)            (yaw 随指令角速度流动)
      v*(t+dt) = v* + a* dt                        (一阶)
      w*(t+dt) = w* + alpha* dt                    (一阶)
      a*(t+dt) = a*,  alpha*(t+dt) = alpha*        (零阶保持)
  旧实现只前推固定 dt, 位置参考在两次重规划之间钉死在原地 (前进被罚,
  站桩满分); 命令项无 ``time_since_replan`` 时仍退化为该旧行为.
- ``⊖``: **仅基座朝向是四元数差** (机体系下相对旋转的轴角向量, 弧度);
  其他情况 (位置/线速度/线加速度/角速度/角加速度) 一律为**向量差**
  (同一坐标系下相减; 位置族在世界系, 速度族在机体系 —— 范数旋转不变,
  机体系便于与测量 ``root_link_*_vel_b`` 直接相减).
- ``b(t)``: 测量值 —— 位置/姿态 ``root_link_pos_w / quat_w``, 线/角
  速度 ``root_link_lin_vel_b / ang_vel_b``; 线/角加速度 mjlab 不提供,
  用机体系速度的后向差分 (模块级缓存, 首步为 0 避免瞬态).

奖励 = exp(-sigma * || [平移误差(3); rot_weight * 角误差(3)] ||_2);
``rot_weight`` 平衡米与弧度量纲 (默认 1.0).
"""

from __future__ import annotations

import torch

_PREV_LIN_VEL_B: dict[int, torch.Tensor] = {}
_PREV_ANG_VEL_B: dict[int, torch.Tensor] = {}


def _command_term(env, command_name: str):
    term = env.command_manager.get_term(command_name)
    if term is None:
        raise KeyError(
            f"奖励依赖命令项 {command_name!r} (TAMOLS desired_base_* 缓冲)"
        )
    return term


# ---------------------------------------------------------------------- #
# 四元数工具 (w,x,y,z)
# ---------------------------------------------------------------------- #
def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dim=-1,
    )


def _quat_conj(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def _quat_from_rotvec(rotvec: torch.Tensor) -> torch.Tensor:
    """旋转向量 (axis*angle) -> 四元数 [N,3]->[N,4]."""
    half = rotvec * 0.5
    angle = half.norm(dim=-1, keepdim=True)
    axis = half / angle.clamp(min=1e-12)
    return torch.cat(
        (torch.cos(angle), torch.sin(angle) * axis), dim=-1
    )


def _quat_integrate_batch(
    q: torch.Tensor, omega_w: torch.Tensor, dt: torch.Tensor
) -> torch.Tensor:
    """批量四元数积分: ``q [N,4]``, ``omega [N,3]``, ``dt [S]`` 或 ``[N,S]``.

    返回 ``[N,S,4]``。用于把 action_consistency 的 60 次逐样本循环合并成
    一次张量运算 (原来每步 60 次小 kernel launch 在 play 中占 ~37 ms)。
    """
    if dt.dim() == 1:
        dt = dt.view(1, -1, 1)
    elif dt.dim() == 2:
        dt = dt.unsqueeze(-1)
    dq = _quat_from_rotvec(omega_w.unsqueeze(1) * dt)
    out = _quat_mul(q.unsqueeze(1), dq)
    return out / out.norm(dim=-1, keepdim=True).clamp(min=1e-9)


def _quat_integrate(
    q: torch.Tensor, omega_w: torch.Tensor, dt: float | torch.Tensor
) -> torch.Tensor:
    """世界系角速度 ``omega_w`` 前向积分四元数: ``q ⊗ exp(w dt/2)``.

    ``dt`` 可为标量或 [N,1] 张量 (每 env 不同的外推时长).
    """
    dq = _quat_from_rotvec(omega_w * dt)
    out = _quat_mul(q, dq)
    return out / out.norm(dim=-1, keepdim=True).clamp(min=1e-9)


def _quat_axisangle_body(
    q_des_w: torch.Tensor, q_meas_w: torch.Tensor
) -> torch.Tensor:
    """机体系姿态误差 (⊖ 的朝向部分): 轴角向量 [N,3] [rad]."""
    q_err = _quat_mul(_quat_conj(q_meas_w), q_des_w)
    q_err = q_err / q_err.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    w = q_err[..., 0:1].clamp(-1.0, 1.0)
    v = q_err[..., 1:]
    v_norm = v.norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(v_norm, w.abs())
    return v / v_norm.clamp(min=1e-9) * angle


def _rotate_world_to_body(
    quat_w: torch.Tensor, vec_w: torch.Tensor
) -> torch.Tensor:
    """``R^T v``: 世界系向量转机体系 (四元数共轭作用)."""
    q_v = torch.cat(
        (torch.zeros_like(vec_w[..., :1]), vec_w), dim=-1
    )
    return _quat_mul(_quat_conj(quat_w), _quat_mul(q_v, quat_w))[..., 1:]


# ---------------------------------------------------------------------- #
# 加速度测量 (后向差分)
# ---------------------------------------------------------------------- #
def _lin_accel_b(asset, key: int, dt: float) -> torch.Tensor:
    vel = asset.data.root_link_lin_vel_b
    prev = _PREV_LIN_VEL_B.get(key)
    _PREV_LIN_VEL_B[key] = vel.detach().clone()
    if prev is None or prev.shape != vel.shape:
        return torch.zeros_like(vel)
    return (vel - prev) / max(dt, 1e-6)


def _ang_accel_b(asset, key: int, dt: float) -> torch.Tensor:
    vel = asset.data.root_link_ang_vel_b
    prev = _PREV_ANG_VEL_B.get(key)
    _PREV_ANG_VEL_B[key] = vel.detach().clone()
    if prev is None or prev.shape != vel.shape:
        return torch.zeros_like(vel)
    return (vel - prev) / max(dt, 1e-6)


def reset_reward_caches(env_id_key: int) -> None:
    """环境 reset 时清差分缓存 (调用方在 reset 钩子里使用)."""
    _PREV_LIN_VEL_B.pop(env_id_key, None)
    _PREV_ANG_VEL_B.pop(env_id_key, None)


# ---------------------------------------------------------------------- #
# 期望状态前推: b*(t) -> b*(t + dt), 从命令项参考点采样
# ---------------------------------------------------------------------- #
def _advance_reference(term, dt: float) -> dict[str, torch.Tensor]:
    """采样期望参考 b*(t + dt) (见模块文档).

    **流动参考**: 命令项的 ``desired_base_*`` 是上次重规划时刻的冻结解,
    只前推固定 dt 会让位置参考在两次重规划之间 (trot 下 0.3 s) 钉死在
    原地 —— 前进被线性罚、站桩满分。这里用命令项维护的
    ``time_since_replan`` (重规划时清零, 每步累加) 把参考外推到当前
    时刻 + dt::

        T = time_since_replan + dt
        p*(t+dt) = p* + v* T        (xy 常速流动; v*_z=0, z 保持地形高度)
        q*(t+dt) = q* ⊗ exp(w* T/2) (yaw 随指令角速度流动)

    速度/加速度参考本身是常值指令, 保持原一阶前推/零阶保持。
    命令项无 ``time_since_replan`` 时退化为旧行为 (只前推一个 dt),
    兼容 shim/旧命令项。
    """
    p = term.desired_base_pos_w
    q = term.desired_base_quat_w
    v = term.desired_base_lin_vel_w
    w = term.desired_base_ang_vel_w
    a = term.desired_base_lin_acc_w
    alpha = term.desired_base_ang_acc_w
    tsr = getattr(term, "time_since_replan", None)
    if isinstance(tsr, torch.Tensor) and tsr.shape[:1] == p.shape[:1]:
        T = (tsr + dt).unsqueeze(-1)                      # [N,1]
    else:
        T = torch.full(
            (p.shape[0], 1), dt, device=p.device, dtype=p.dtype
        )
    q_n = _quat_integrate(q, w, T)
    return {
        "pos": p + v * T,
        "quat": q_n,
        "lin_vel": v + a * dt,
        "ang_vel": w + alpha * dt,
        "lin_acc": a,
        "ang_acc": alpha,
    }


# ---------------------------------------------------------------------- #
# 奖励
# ---------------------------------------------------------------------- #
def base_pose_tracking(
    env,
    sigma: float,
    order: int = 0,
    component: str = "both",
    command_name: str = "Tamols_footholds",
    entity_name: str = "robot",
    rot_weight: float = 1.0,
    squared: bool = False,
) -> torch.Tensor:
    """``r_{B_n} = exp(-sigma || b*(t+dt) ⊖ b(t) ||_2^2)``.

    Args:
      sigma: ``sigma_{B_n}``; cfg 中分多个 reward 项各配各的.
      order: 导数阶数 0(位姿)/1(速度)/2(加速度).
      component: 误差成分. ``"both"`` = 6 维 [Δp; w_r*Δθ] 混合
        (原公式整体); ``"pos"`` 仅平移族 (位置/线速度/线加速度);
        ``"rot"`` 仅角族 (姿态/角速度/角加速度) —— 拆分后可对
        位置与姿态**分别调权重** (cfg 中各配 sigma/weight).
      command_name: TAMOLS 命令项 (期望参考来源).
      rot_weight: 角误差与平移误差的量纲平衡 (仅 component="both"
        时有意义; 默认 1.0).
      squared: 是否使用论文的平方范数 ``||e||^2``. 论文 Table 的
        σ=1200/90/10/... 是按平方误差标定的; 旧任务使用非平方范数
        并手动把 σ 调小. 新 DTC baseline 应设为 True.
    """
    if order not in (0, 1, 2):
        raise ValueError(f"order 需为 0/1/2, 实际 {order}")
    if component not in ("both", "pos", "rot"):
        raise ValueError(
            f"component 需为 'both'/'pos'/'rot', 实际 {component!r}"
        )
    term = _command_term(env, command_name)
    asset = env.scene[entity_name]
    data = asset.data
    dt = float(getattr(env, "step_dt", 0.02))
    ref = _advance_reference(term, dt)

    if order == 0:
        # ⊖: 位置=向量差(世界系); 朝向=四元数差(机体系轴角)
        lin = ref["pos"] - data.root_link_pos_w
        ang = _quat_axisangle_body(ref["quat"], data.root_link_quat_w)
    elif order == 1:
        # 速度族: 向量差 (机体系; 期望由世界系转到机体系)
        lin = _rotate_world_to_body(
            data.root_link_quat_w, ref["lin_vel"]
        ) - data.root_link_lin_vel_b
        ang = _rotate_world_to_body(
            data.root_link_quat_w, ref["ang_vel"]
        ) - data.root_link_ang_vel_b
    else:
        key = id(env)
        lin = _rotate_world_to_body(
            data.root_link_quat_w, ref["lin_acc"]
        ) - _lin_accel_b(asset, key, dt)
        ang = _rotate_world_to_body(
            data.root_link_quat_w, ref["ang_acc"]
        ) - _ang_accel_b(asset, key, dt)

    if component == "pos":
        err = lin
    elif component == "rot":
        err = ang
    else:
        err = torch.cat((lin, rot_weight * ang), dim=-1)
    if squared:
        # 论文式: exp(-σ ||e||²). 非平方版本在 σ=1200 时对 cm 级
        # 误差已经 underflow 到 0, 会让整个 base-pose 奖励失去梯度.
        norm_sq = err.square().sum(dim=-1)
        return torch.exp(-float(sigma) * norm_sq)
    return torch.exp(-float(sigma) * err.norm(dim=-1))


# ---------------------------------------------------------------------- #
# 硬落脚跟踪 r_{p_i} (Foothold Hard Tracking)
# ---------------------------------------------------------------------- #
# 每脚每步态周期至多一次的状态缓存:
#   awarded    = 本支撑相已给分 [N,4] bool
#   prev_swing = 上一步摆动掩码 (检测 swing->stance 触地沿) [N,4] bool
#   prev_time  = 上一步的步态时钟读数 (回绕 => 该 env 被 reset) [N] | None
_HARD_TRACK_STATE: dict[
    int, tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]
] = {}

LEG_ORDER = ("FL", "FR", "RL", "RR")


def reset_hard_track_state(env_id_key: int) -> None:
    """清硬跟踪缓存 (环境 reset 钩子调用)."""
    _HARD_TRACK_STATE.pop(env_id_key, None)


def _contact_force_norms(env, sensor_name: str) -> torch.Tensor | None:
    """各腿接触力范数 [N,4] (按 FL/FR/RL/RR 重排).

    接触传感器 primaries 顺序任意 (velocity cfg 中为 FR,FL,RR,RL),
    通过 ``primary_names`` 映射; force 为净力 [B,P*S,3], 取 slot 0.
    """
    sensor = env.scene.get(sensor_name) if hasattr(env.scene, "get") else None
    if sensor is None and sensor_name in getattr(env.scene, "sensors", {}):
        sensor = env.scene.sensors[sensor_name]
    if sensor is None or getattr(sensor.data, "force", None) is None:
        return None
    primaries = list(getattr(sensor, "primary_names", ()) or ())
    if not primaries:  # 无名字信息: 假定已是 FL,FR,RL,RR
        order = list(range(4))
    else:
        order = []
        for leg in LEG_ORDER:
            for idx, name in enumerate(primaries):
                if name == leg or name.startswith(f"{leg}_") or name == f"{leg}_foot":
                    order.append(idx)
                    break
            else:
                return None  # 无法对齐腿序: 视为无传感器 (奖励退化为
                # 仅期望接触门控, 见 base 调用方)
    force = sensor.data.force  # [B, P*S, 3]
    num_slots = max(force.shape[1] // max(len(primaries), 1), 1)
    norms = force.norm(dim=-1)  # [B, P*S]
    if num_slots > 1:
        norms = norms.view(force.shape[0], -1, num_slots)[..., 0]
    return norms[:, order]


def foothold_hard_tracking(
    env,
    eps: float = 1e-3,
    command_name: str = "Tamols_footholds",
    sensor_name: str = "feet_ground_contact",
    entity_name: str = "robot",
    force_threshold: float = 1.0,
    d_min: float = 0.02,
) -> torch.Tensor:
    """硬落脚跟踪 ``r_{p_i} = -ln(max(||p_i* - p_i||, d_min)^2 + eps)`` [N].

    - ``p_i``: 当前足部位置 (body_link_pos_w, FL/FR/RL/RR);
    - ``p_i*``: 期望落脚点 —— 优先 ``landing_footholds`` (支撑相冻结的
      本次触地目标; ``optimal_footholds`` 在触地时已重规划为下一步,
      直接用会导致支撑相永远对不上), 无该属性时回退;
    - 门控 (同一步全部满足才给分): ① 期望接触状态 (gait trot 的
      contact_states); ② 接触力范数 ``||f_i|| > force_threshold``;
      ③ 本支撑相尚未给过分;
    - **每个步态周期每脚至多一次**: 触地沿 (swing->stance) 清除该脚
      的已给分标志, 支撑相内首次满足 ①② 的一步以**当时的落脚误差**
      结账, 之后本相不再给;
    - **出生不送分**: 状态初建 (进程内首次调用) 或 env 被 reset
      (由步态时钟 ``current_time`` 回绕检测, 无需外部接线) 时, 已处于
      支撑相的腿直接置为已给分 —— 只有经历过真实摆动->触地的腿才
      能得分 (修复: 旧版 reset 后第 0 步脚压在地上即白领一次分);
    - **距离下限 ``d_min``**: d < d_min 按 d_min 结账, 单脚上限
      ``-ln(d_min^2 + eps)`` (默认 ≈7.8) —— 去掉旧版 d→0 时
      ``-ln(eps)`` 的无界站桩彩票;
    - 返回各腿之和 (每腿独立得分).
    """
    if not (0.0 < eps < 1.0):
        raise ValueError(f"eps 需满足 0 < eps << 1, 实际 {eps}")
    if d_min <= 0.0:
        raise ValueError(f"d_min 需 > 0, 实际 {d_min}")
    term = _command_term(env, command_name)
    asset = env.scene[entity_name]
    key = id(env)

    stance = term.gait_scheduler.contact_states >= 0.5      # [N,4] bool
    swing = ~stance
    clock = getattr(term.gait_scheduler, "current_time", None)

    state = _HARD_TRACK_STATE.get(key)
    if state is not None and state[0].shape != stance.shape:
        state = None
    if state is None:
        # 初建: 支撑相腿锁定为已给分 (出生不送分)
        awarded = stance.clone()
        prev_swing = swing.clone()
        prev_time = None if clock is None else clock.detach().clone()
    else:
        awarded, prev_swing, prev_time = state
        if clock is not None and prev_time is not None:
            reset_envs = clock < prev_time - 1e-6           # 时钟回绕=reset
            if bool(reset_envs.any()):
                awarded = awarded.clone()
                prev_swing = prev_swing.clone()
                awarded[reset_envs] = stance[reset_envs]
                prev_swing[reset_envs] = swing[reset_envs]
        prev_time = None if clock is None else clock.detach().clone()

    touchdown = prev_swing & stance                         # 本步触地沿
    awarded = awarded & ~touchdown                          # 触地沿清标志

    force = _contact_force_norms(env, sensor_name)          # [N,4] | None
    in_contact = (
        (force > float(force_threshold))
        if force is not None else stance  # 无传感器: 退化为仅期望门控
    )
    gate = stance & in_contact & ~awarded
    _HARD_TRACK_STATE[key] = (awarded | gate, swing.clone(), prev_time)

    if not bool(gate.any()):
        return torch.zeros(stance.shape[0], device=stance.device)

    names = list(asset.body_names)
    foot_ids = []
    for wanted in (f"{l}_foot" for l in LEG_ORDER):
        for cand in (wanted, wanted.replace("_foot", ""), f"{wanted}_link"):
            if cand in names:
                foot_ids.append(names.index(cand))
                break
        else:
            raise ValueError(f"无法解析脚 body {wanted!r}; 可用: {names}")
    p = asset.data.body_link_pos_w[:, foot_ids, :]          # [N,4,3]
    des = getattr(term, "landing_footholds", None)
    if not isinstance(des, torch.Tensor):
        des = term.optimal_footholds
    d = (des - p).norm(dim=-1).clamp(min=float(d_min))      # [N,4]
    r_per_foot = -torch.log(d * d + float(eps))             # [N,4]

    return (r_per_foot * gate).sum(dim=-1)


# ---------------------------------------------------------------------- #
# 动作一致性 r_c (Action / Plan Consistency)
# ---------------------------------------------------------------------- #
# 相邻两次规划解的快照 (batched 张量, 按 env 各自推进):
#   {基座参考状态, footholds,
#    "t0":    捕获时刻的步态时钟读数 [N] (绝对时间对齐用),
#    "comp":  捕获时刻的摆动掩码 [N,4] (落脚点项可比腿),
#    "valid": 是否已捕获有效解 [N]}
_CONSIST_STATE: dict[int, dict] = {}


def reset_consistency_state(env_id_key: int) -> None:
    """清一致性缓存 (环境 reset 钩子调用)."""
    _CONSIST_STATE.pop(env_id_key, None)


def _snapshot(
    term, clock: torch.Tensor, comp: torch.Tensor
) -> dict[str, torch.Tensor]:
    """重规划时刻捕获规划解快照 (基座参考状态 + 落脚点 + 时间戳).

    ``comp``: 可比腿掩码 [N,4] —— 见 action_consistency 文档.
    """
    return {
        "pos": term.desired_base_pos_w.detach().clone(),
        "quat": term.desired_base_quat_w.detach().clone(),
        "lin_vel": term.desired_base_lin_vel_w.detach().clone(),
        "ang_vel": term.desired_base_ang_vel_w.detach().clone(),
        "footholds": term.optimal_footholds.detach().clone(),
        "t0": clock.detach().clone(),
        "comp": comp.detach().clone(),
        "valid": torch.ones(
            clock.shape[0], dtype=torch.bool, device=clock.device
        ),
    }


def _flow(
    snap: dict, s: float | torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """快照在其自身相对时刻 ``s`` (秒, 标量或 [N,1]) 处的基座参考.

    与 ``_advance_reference`` 的流动参考同一常速外推 + 常角速度四元数
    积分模型 —— 两次比较必须用同一模型, 差异才只反映规划抖动.
    """
    pos = snap["pos"] + snap["lin_vel"] * s
    quat = _quat_integrate(snap["quat"], snap["ang_vel"], s)
    return pos, quat


def action_consistency(
    env,
    w_p: float = 1.0,
    command_name: str = "Tamols_footholds",
    dt_compare: float = 0.01,
    horizon: float | None = None,
    rot_weight: float = 1.0,
    err_cap: float = 0.1,
) -> torch.Tensor:
    """动作一致性 ``r_c`` (<=0, 相邻两解差异惩罚).

    ``r_c = -Σ_j ‖ b_a*(t_j) ⊖ b_b*(t_j) ‖ - w_p ‖ p_a* - p_b* ‖_comp``

    捕获时机与语义 (整周期重规划模式):

    - **捕获 = 真实重规划**: 命令项每步维护 ``replan_event`` (整周期
      模式下触地沿不再等于重规划), 按它捕获快照; 无该缓冲的 shim/旧
      命令项回退到触地沿检测。快照在两次重规划之间冻结 -> 奖励值保持.
    - **绝对时间对齐**: 解 a/b 各自从捕获时刻 (``t0_a``/``t0_b``) 外推,
      在解 a 的整个时域 (0, horizon] 上按绝对时间采样, 解 b 外推到
      ``Δ + s`` (整周期下 Δ≈horizon, 两解时域不重叠, 这是把论文的重叠窗
      退化后的等价形式 —— 一致行走时两轨迹仍处处重合, Δ 小于 horizon
      时与论文重叠窗严格一致)。
    - **落脚点项可比腿**: 只比较"捕获时刻仍悬空 **且** 自上次捕获以来
      未发生过触地沿"的腿 (per-leg 触地沿累计器) —— 这样的腿两次捕获
      的目标指向同一次触地事件。整周期重规划下相邻两解的所有目标指向
      不同事件 (comp 恒空, 项为 0 —— 计划在周期内冻结, 本无中途改写
      可言); 逐触地模式下则恰好选出横跨两次重规划仍悬空的腿。
    - **按 env 触发/隔离** + env reset 由步态时钟回绕检测后作废两解.
    - **``err_cap``**: 单样本差上界 —— 超 cap 的差属于跟踪失败或指令
      突变 (跟踪项已覆盖), 封顶避免一致性项在重计划瞬间爆量.

    任一解无效时返回 0.
    """
    if w_p <= 0.0:
        raise ValueError(f"w_p 需 > 0, 实际 {w_p}")
    if dt_compare <= 0.0:
        raise ValueError(f"dt_compare 需 > 0, 实际 {dt_compare}")
    if err_cap <= 0.0:
        raise ValueError(f"err_cap 需 > 0, 实际 {err_cap}")
    term = _command_term(env, command_name)
    key = id(env)
    if horizon is None:
        horizon = float(term.gait_scheduler.cfg.stride_duration)

    swing = term.gait_scheduler.is_swing              # [N,4] bool
    clock = term.gait_scheduler.current_time          # [N]
    n_envs = swing.shape[0]
    zeros = torch.zeros(n_envs, device=swing.device)

    state = _CONSIST_STATE.get(key)
    if state is not None and state["prev_swing"].shape != swing.shape:
        state = None
    if state is None:
        # 初建: prev_swing/prev_clock 置为当前值 (fallback 模式下触地沿
        # 检测自然为 False); swung 初始化为当前摆动掩码 —— 初始已在空中的
        # 腿视为"间隔内摆动过", 其目标不参与落脚点比较。初始化后继续执行,
        # 首次调用即可按 replan_event 捕获。
        state = {
            "a": None, "b": None,
            "prev_swing": swing.detach().clone(),
            "prev_clock": clock.detach().clone(),
            "touched": torch.zeros_like(swing),
            "value": None,
        }
        _CONSIST_STATE[key] = state

    reset_envs = clock < state["prev_clock"] - 1e-6        # 时钟回绕=reset
    touchdown_edge = state["prev_swing"] & ~swing          # [N,4] per-leg 沿
    touchdown = touchdown_edge.any(dim=-1)                 # [N] 按 env
    state["prev_swing"] = swing.detach().clone()
    state["prev_clock"] = clock.detach().clone()
    touched_before = state["touched"].clone()              # 至上次捕获的累计
    state["touched"] = state["touched"] | touchdown_edge

    event = getattr(term, "replan_event", None)
    if isinstance(event, torch.Tensor) and event.shape[0] == n_envs:
        capture = event.bool()                    # 真实重规划事件 (优先)
    else:
        capture = touchdown                       # 兼容: 触地沿检测

    a, b = state["a"], state["b"]
    if bool(capture.any()):
        comp = (swing & ~touched_before).float()
        snap = _snapshot(term, clock, comp)
        if a is None:
            a = snap
            a["valid"] = capture.clone()
        else:
            if b is None:
                b = {k: torch.zeros_like(v) for k, v in a.items()}
            for k in b:
                if k != "valid":
                    b[k][capture] = a[k][capture]
            b["valid"][capture] = a["valid"][capture]
            for k in a:
                if k != "valid":
                    a[k][capture] = snap[k][capture]
            a["valid"][capture] = True
        state["a"], state["b"] = a, b
        # 沿累计器: 已捕获 env 从本步重新计数 (捕获步的沿属于刚结束的间隔)
        state["touched"][capture] = False
    if a is not None and bool(reset_envs.any()):
        a["valid"][reset_envs] = False
        if b is not None:
            b["valid"][reset_envs] = False
        state["touched"][reset_envs] = False

    # 快照/时钟都没变化时, 奖励值完全由冻结的两解决定, 直接返回缓存.
    # play (单 env) 下这能把 action_consistency 从每步 37 ms 降到只在
    # 重规划那一步计算; 训练 batch 中通常每个 step 都有 env 重规划, 缓存
    # 不命中, 由下面的向量化计算承担.
    if (
        state["value"] is not None
        and not bool(capture.any())
        and not bool(reset_envs.any())
    ):
        return state["value"]

    if a is None or b is None:
        state["value"] = zeros
        return zeros
    both = a["valid"] & b["valid"]
    if not bool(both.any()):
        state["value"] = zeros
        return zeros

    delta_t = (a["t0"] - b["t0"]).clamp(min=0.0)           # [N]
    n_samples = int((horizon - 1e-9) / dt_compare)
    if n_samples > 0:
        # 一次向量化采样 [N,S], 避免 60 次 Python/kernel launch (play 中的
        # 主要卡顿来源).
        s = torch.arange(
            1, n_samples + 1, device=swing.device, dtype=torch.float32
        ) * float(dt_compare)                               # [S]
        s_a = s.view(1, -1, 1)                              # [1,S,1]
        s_b = (delta_t.unsqueeze(1) + s.unsqueeze(0))       # [N,S]
        pos_a = a["pos"].unsqueeze(1) + a["lin_vel"].unsqueeze(1) * s_a
        quat_a = _quat_integrate_batch(a["quat"], a["ang_vel"], s)
        pos_b = b["pos"].unsqueeze(1) + b["lin_vel"].unsqueeze(1) * s_b.unsqueeze(-1)
        quat_b = _quat_integrate_batch(b["quat"], b["ang_vel"], s_b)
        lin = pos_a - pos_b
        ang = _quat_axisangle_body(quat_a, quat_b)
        err = torch.cat((lin, rot_weight * ang), dim=-1).norm(dim=-1)  # [N,S]
        total = (
            err.clamp(max=float(err_cap)) * both.unsqueeze(-1)
        ).sum(dim=-1)
    else:
        total = torch.zeros(n_envs, device=swing.device)

    comp = a["comp"].unsqueeze(-1)                          # [N,4,1]
    d_foot = (
        ((a["footholds"] - b["footholds"]) * comp)
        .reshape(n_envs, -1)
        .norm(dim=-1)
    )
    value = (-total - float(w_p) * d_foot) * both
    state["value"] = value
    return value


# ---------------------------------------------------------------------- #
# 正则化惩罚 (论文补充材料)
# ---------------------------------------------------------------------- #
_PREV_ACTION: dict[int, torch.Tensor] = {}
_PREV_JOINT_VEL: dict[int, torch.Tensor] = {}


def reset_regularization_state(env_id_key: int) -> None:
    """清正则化差分缓存 (reset 钩子)."""
    _PREV_ACTION.pop(env_id_key, None)
    _PREV_JOINT_VEL.pop(env_id_key, None)


def _foot_body_ids(asset) -> list[int]:
    names = list(asset.body_names)
    ids = []
    for wanted in (f"{l}_foot" for l in LEG_ORDER):
        for cand in (wanted, wanted.replace("_foot", ""), f"{wanted}_link"):
            if cand in names:
                ids.append(names.index(cand))
                break
        else:
            raise ValueError(f"无法解析脚 body {wanted!r}; 可用: {names}")
    return ids


def foot_contact_power(
    env,
    sensor_name: str = "feet_ground_contact",
    entity_name: str = "robot",
) -> torch.Tensor:
    """足端接触功率惩罚 ``r_{r1} = -sum_i |v_i^T f_i|``.

    ``v_i``: 脚的世界系线速度 (body_link_vel_w 前 3 维, FL/FR/RL/RR);
    ``f_i``: 接触传感器净力 (按 primary 名映射到腿序). 脚在大力接触下
    仍快速滑动 (打滑/拖动) 时惩罚增大. 无力数据时返回 0 (无接触).
    """
    asset = env.scene[entity_name]
    vel = asset.data.body_link_vel_w[:, _foot_body_ids(asset), :3]  # [N,4,3]
    force = _contact_force_norms_aligned(env, sensor_name, per_axis=True)
    if force is None:
        return torch.zeros(vel.shape[0], device=vel.device)
    power = (vel * force).sum(dim=-1)          # v^T f  [N,4]
    return -power.abs().sum(dim=-1)


def _contact_force_norms_aligned(
    env, sensor_name: str, per_axis: bool = False
):
    """接触力 [N,4,3] (按 FL/FR/RL/RR); per_axis=False 时为范数 [N,4]."""
    sensor = None
    if hasattr(env.scene, "get"):
        sensor = env.scene.get(sensor_name)
    if sensor is None and sensor_name in getattr(env.scene, "sensors", {}):
        sensor = env.scene.sensors[sensor_name]
    if sensor is None or getattr(sensor.data, "force", None) is None:
        return None
    force = sensor.data.force  # [B, P*S, 3]
    primaries = list(getattr(sensor, "primary_names", ()) or ())
    num_slots = max(force.shape[1] // max(len(primaries), 1), 1)
    if num_slots > 1:
        force = force.view(force.shape[0], -1, num_slots, 3)[..., 0, :]
    if not primaries:
        order = list(range(min(4, force.shape[1])))
    else:
        order = []
        for leg in LEG_ORDER:
            for idx, name in enumerate(primaries):
                if name == leg or name.startswith(f"{leg}_") or name == f"{leg}_foot":
                    order.append(idx)
                    break
        if len(order) != 4:
            return None
    force = force[:, order]                    # [N,4,3]
    return force if per_axis else force.norm(dim=-1)


def joint_power(
    env,
    entity_name: str = "robot",
) -> torch.Tensor:
    """关节功率惩罚 ``r_{r2} = -sum_i (qdot_i^T tau_i)^2``.

    ``qdot``: 关节速度 (joint_vel); ``tau``: 关节力矩 —— mjlab 无
    ``joint_torques`` (歧义), 用 ``qfrc_actuator`` (执行器投影到关节
    空间的力, 电机即指令力矩×齿比, 文档见 EntityData). 惩罚大关节
    功率, 抑制摆动期不必要的剧烈动作.
    """
    asset = env.scene[entity_name]
    vel = asset.data.joint_vel                     # [N, nv]
    tau = asset.data.qfrc_actuator                 # [N, nv]
    power = (vel * tau).sum(dim=-1)                # [N]
    return -(power * power)


def action_rate(env) -> torch.Tensor:
    """动作变化率惩罚 ``-||a_t - a_{t-1}||^2`` (补充材料).

    直接用 ``action_manager.action / prev_action`` (mjlab 内建缓冲,
    与 mdp.action_rate_l2 同源; 本函数只是加负号并保持 DTC 命名).
    """
    diff = env.action_manager.action - env.action_manager.prev_action
    return -(diff * diff).sum(dim=-1)


def joint_acceleration(
    env,
    entity_name: str = "robot",
) -> torch.Tensor:
    """关节加速度惩罚 ``-||qddot||^2`` (EntityData.joint_acc)."""
    asset = env.scene[entity_name]
    acc = asset.data.joint_acc
    return -(acc * acc).sum(dim=-1)


# ---------------------------------------------------------------------- #
# 原地滞留惩罚 (仿 track/Trackingreward.stall_penalty)
# ---------------------------------------------------------------------- #
def stall_penalty(
    env,
    command_name: str = "twist",
    entity_name: str = "robot",
    vel_threshold: float = 0.1,
    cmd_threshold: float = 0.1,
    penalty_value: float = -2.0,
) -> torch.Tensor:
    """指令要求移动而机器人没有沿指令方向前进时的固定罚分.

    track 原版 (``Trackingreward.stall_penalty``) 判据是
    "距目标点 > 0.3 m 且 ``|v| < 0.1``"; DTC 是速度指令任务, 没有目标
    点, 判据换成指令方向上的投影速度::

        r = penalty_value  if ``|v_cmd_xy| > cmd_threshold`` 且
                              ``v_meas_xy · v_cmd_dir < vel_threshold``
          = 0              otherwise

    - 只看水平面 (xy): 垂直速度与偏航不触发;
    - 用**指令方向投影**而不是速度范数: 原地左右徘徊/打转时速度范数
      可能 > 0.1, 旧判据不会罚; 现在投影速度低就会被罚;
    - 命令为零的环境 (standing 采样、stair north_peak 到达中心后的原地
      站立) **不**触发 —— 只有"被要求走而没走"才罚;
    - 沿指令方向真正走起来后立即解除; 横向/后退运动仍会被罚.
    """
    cmd_term = _command_term(env, command_name)
    cmd = getattr(cmd_term, "vel_command_b", None)
    if cmd is None:
        cmd = cmd_term.command
    cmd = cmd.to(dtype=torch.float32)
    asset = env.scene[entity_name]
    cmd_xy = cmd[:, :2]
    cmd_speed = cmd_xy.norm(dim=-1)                          # [N]
    cmd_dir = cmd_xy / cmd_speed.clamp(min=1e-6).unsqueeze(-1)
    vel_xy = asset.data.root_link_lin_vel_b[:, :2]
    forward_speed = (vel_xy * cmd_dir).sum(dim=-1)           # [N]
    is_stalled = (cmd_speed > float(cmd_threshold)) & (
        forward_speed < float(vel_threshold)
    )
    return torch.where(
        is_stalled,
        torch.full_like(forward_speed, float(penalty_value)),
        torch.zeros_like(forward_speed),
    )
