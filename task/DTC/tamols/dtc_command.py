"""DTC 速度任务用的 TAMOLS 落脚点命令项与独立地图 UI.

与用户约定的关系
================
- 命令项的**规划计算**完全继承 fetch 侧现有方案
  (``TamolsMotionCommand``: 触地触发的 TorchTAMOLSMotionPlanner 重规划),
  本模块不改任何规划行为;
- 规划计算继承 fetch 侧实现; 独立 ``ElevationMapUI`` 在单独的网页端口显示
  机体系局部高程图与落点, 不在 MuJoCo/Viser 仿真场景中叠加图元;
- 本文件位于 task/DTC 内, 不修改 fetch/velocity 的任何文件.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Literal

import torch

from myrobot_mjlab.task.Tamols.fetch.tamols_command import (
    TamolsMotionCommand,
    TamolsMotionCommandCfg,
)
from myrobot_mjlab.task.DTC.tamols.debug_viz import draw_command_footholds

if TYPE_CHECKING:
    import viser


@dataclass(kw_only=True)
class DTCFootholdCommandCfg(TamolsMotionCommandCfg):
    """DTC 版 TAMOLS 落脚点命令项配置 (默认开启三色足点可视化)."""

    debug_vis: bool = False
    map_ui_enable: bool = False
    """是否单独启动高程图与落点查看 UI (不会叠加到仿真场景)。"""

    map_ui_port: int = 8081
    """独立高程图 UI 的本地网页端口; 0 表示自动分配。"""

    map_ui_stride: int = 1
    """3D 地形网格降采样倍数 (1 表示按原始高程图分辨率绘制)。"""

    map_ui_update_interval: float = 0.1
    """独立地图 UI 刷新间隔 (秒), 限制传图与绘制开销。"""

    show_nominal_dots: bool = True
    """ nominal 蓝点默认显示 (对比重规划是否偏离初值)."""

    replan_on_any_touchdown: bool = True
    """论文 TAMOLS: 每个 leg touch-down 事件求解一次。

    trot 中两条对角腿同时触地，实际触发频率与“每个对角对一次”相同；
    单腿提前触地时不再等待同对的另一条腿。
    """

    replan_after_all_four_touchdowns: bool = False
    """保留的整周期开关；当 ``replan_on_any_touchdown=True`` 时不生效。

    DTC 速度任务默认由 ``replan_on_any_touchdown`` 在每次触地时重规划；
    指令突变/环境重置仍即时触发。
    """

    swing_obs_enable: bool = True
    """是否组装摆动腿观测 (第一步: 见 swing_leg 模块文档).

    观测 = 当前脚位置/目标落点(机体系相对量) + 摆动路径局部高度 +
    步态相位/进度 + 触地事件; **不生成摆动轨迹** (RL 自己学)."""

    swing_path_samples: int = 5
    """摆动路径地形采样点数 (含两端; 仅地形查询位置, 不是参考轨迹)."""

    swing_foot_bodies: tuple[str, ...] = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")

    in_place_gait_enable: bool = False
    """水平/角速度命令为零时, 固定名义站姿落点但保留共享 gait 相位,
    让策略能在原地持续 trot, 避免地图优化把足点推成内八/偏斜站姿."""

    in_place_cmd_threshold: float = 0.04
    """判定原地踏步模式的最大水平命令速度 (m/s)."""

    # ---- 规划器后端: 轻量 grid 或完整 NLP ----
    planner_mode: Literal["grid", "nlp"] = "grid"
    """落脚点规划后端。

    - ``grid``: fetch 的 Raibert 标称点 + 邻域离散网格地形优化（轻量后端）；
    - ``nlp``: 用 C++/CppAD 求解器把 ``desired_footholds`` 直接作为
      决策变量，与机身样条轨迹联合优化。默认异步，主循环不阻塞；
      超过 ``nlp_max_envs`` 时自动回退 grid。
    """

    nlp_max_envs: int = 4
    """启用 NLP 模式允许的最大环境数；超过时自动回退到 grid 并打印警告。"""

    nlp_solver: Literal["gn_sqp", "ipopt"] = "gn_sqp"
    """Python 侧 NLP 求解后端（仅 nlp_solver_backend='python' 时使用）。"""

    nlp_solver_backend: Literal["python", "cpp"] = "cpp"
    """求解器实现：``cpp`` 为 C++/Eigen+CppAD+QuadProg++，``python`` 为
    原 CasADi 实现。默认 C++。"""

    nlp_samples_per_phase: int = 25
    """轨迹导出时每个相位的采样点数。"""

    nlp_max_iter: int = 60
    """Python/CasADi 路径的最大迭代次数。"""

    nlp_real_max_iter: int = 20
    """真实 NLP 在 Layer 1/2 热启动后的首轮最大迭代数（截断求解）。"""

    nlp_real_first_iter: int = 20
    """首轮截断迭代数：先跑这么多次，已可行则不再继续。

    对应论文 graduated optimization 的“先少量迭代看是否足够”。
    注意：取 5 时虽然更快，但零速度站立时也会接受未收敛的落脚点，
    造成 xy 抖动；默认取满首轮预算，数值优先。
    """

    nlp_real_max_iter2: int = 40
    """首轮未接受时从当前点继续的最大迭代数。"""

    nlp_jit: bool = False
    """是否启用 GN-SQP 的 CasADi JIT（首次编译约 30 s，之后更快）。"""

    nlp_fallback_to_grid: bool = True
    """NLP 求解失败时是否回退到 grid 规划器。"""

    nlp_accept_last_iterate: bool = True
    """GN-SQP 达到 max_iter 但最后迭代点已接近可行时，是否接受该点。"""

    nlp_max_violation: float = 5e-3
    """接受最后迭代点允许的最大约束违反量。"""

    nlp_friction_mode: Literal["enabled", "disabled"] = "disabled"
    """NLP 摩擦锥约束。论文摩擦符号在本仓库仍是待核实项，默认关闭；
    需要严格摩擦可行性时设为 enabled，并配合 nlp_friction_cone_form。"""

    nlp_friction_cone_form: Literal[
        "paper_literal", "force_balance_derived"
    ] = "force_balance_derived"
    """摩擦锥公式。``force_balance_derived`` 是论文内部证据派生的自洽形式；
    ``paper_literal`` 保留论文印刷式（静止点不可行，仅用于对照）。"""

    nlp_inertial_accel_sign: Literal[
        "paper_literal", "contact_normal_explicit"
    ] = "contact_normal_explicit"
    """GIAC/摩擦的惯性加速度符号约定。``contact_normal_explicit`` 为
    物理自洽的接触法向形式；``paper_literal`` 为论文印刷式。"""

    nlp_verbose: bool = False
    """是否打印每次 NLP 求解的状态/耗时。"""

    nlp_compare_python: bool = False
    """调试: 每次 C++ 求解后同时跑 Python GN-SQP，并打印解差异。"""

    nlp_async: bool = True
    """是否用后台线程异步求解 C++ NLP；主循环立即用上一份结果/grid 继续。"""

    nlp_layer1: bool = True
    """论文 V-D2: 先用 h_s2 虚拟地面解一层简化 TO，作为真实 NLP 的热启动。"""

    nlp_layer1_max_iter: int = 20
    """Layer 1 简化 TO 的最大 GN-SQP 迭代数。"""

    nlp_layer2: bool = True
    """论文 V-D3: 在上一轮落脚点周围做 0.4 m batch search 细化。"""

    nlp_layer2_radius: float = 0.4
    """Layer 2 搜索半径 [m]。"""

    nlp_layer2_resolution: float = 0.02
    """Layer 2 搜索分辨率 [m]。"""

    nlp_layer2_gpu_min_points: int = 20000
    """候选点总数低于该值时 Layer 2 用 CPU：小批量 GPU kernel 反而更慢。"""

    def build(self, env) -> "DTCFootholdCommand":
        """构建 DTC 子类实例 (父类 build 返回 TamolsMotionCommand,
        不覆写则 DTC 的 create_gui/三色绘制/摆动观测全部静默失效)."""
        return DTCFootholdCommand(self, env)


class DTCFootholdCommand(TamolsMotionCommand):
    """继承 fetch 的 TAMOLS 命令项, 增加独立高程图 UI 与摆动观测.

    第一步扩展 (本类内, 不改 fetch): 组装摆动腿观测
    (``self.swing_obs``, 见 :mod:`swing_leg`), 数据源全部复用父类
    已有状态 —— 固定 trot 时钟 = 父类 ``gait_scheduler`` (builder 共享,
    不重复推进), 落足点 = ``optimal_footholds`` (TAMOLS 输出),
    局部高度 = 父类 ``h_raw`` 本地高程图 (机体系网格).
    """

    def _post_super_init(self) -> None:
        """在 super().__init__ 之后调用 (由 __init__ 包装负责)."""
        from myrobot_mjlab.task.DTC.tamols.swing_leg import (
            SwingObsBuilder,
            SwingObsCfg,
        )

        if self.cfg.swing_obs_enable:
            self._swing_builder = SwingObsBuilder(
                SwingObsCfg(path_samples=self.cfg.swing_path_samples),
                num_envs=self.num_envs,
                device=self.device,
                gait=self.gait_scheduler,   # 共享父类固定 trot 时钟
            )
        else:
            self._swing_builder = None
        self.swing_obs: dict | None = None

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._post_super_init()

        # 规划器后端: 默认 fetch 的 grid；NLP 只替换 plan_motion，
        # Raibert 标称点仍由原 grid 规划器提供，作为 NLP 初值。
        self._grid_planner = self.torch_planner
        self._nlp_planner = None
        if getattr(self.cfg, "planner_mode", "grid") == "nlp":
            max_envs = int(getattr(self.cfg, "nlp_max_envs", 4))
            if self.num_envs > max_envs:
                print(
                    f"[NLP] planner_mode='nlp' 但 num_envs={self.num_envs} > "
                    f"nlp_max_envs={max_envs}，继续使用 grid planner。"
                    " 如需强制 NLP，请增大 nlp_max_envs 并接受 ~0.5 s/次的求解耗时。"
                )
            else:
                from myrobot_mjlab.task.DTC.tamols.nlp_planner import (
                    NLPFootholdPlanner,
                )

                self._nlp_planner = NLPFootholdPlanner(
                    command=self,
                    base_planner=self.torch_planner,
                    cfg=self.cfg,
                )
                self.torch_planner = self._nlp_planner
                print(
                    f"[NLP] 启用 IntegratedNLP 落脚点规划 "
                    f"(num_envs={self.num_envs}, backend={self.cfg.nlp_solver_backend}, qp=quadprog_cpp)"
                )

        # “每次触地”只对 NLP 模式生效；grid 训练仍用原来的对角对/整周期
        # 触发，避免 1024 env 的 grid 回退路径被改频率。
        if self._nlp_planner is None:
            self.cfg.replan_on_any_touchdown = False

        self._elevation_map_ui = None
        if self.cfg.map_ui_enable:
            try:
                from myrobot_mjlab.task.DTC.tamols.elevation_map_ui import (
                    ElevationMapUI,
                )

                self._elevation_map_ui = ElevationMapUI(
                    robot=self.robot,
                    port=self.cfg.map_ui_port,
                    stride=self.cfg.map_ui_stride,
                    update_interval=self.cfg.map_ui_update_interval,
                    show_nominal=self.cfg.show_nominal_dots,
                )
                # Render the robot and a waiting/empty-map status immediately,
                # so the dedicated page is not blank before the first sim step.
                self._elevation_map_ui.update(self, force=True)
            except Exception as exc:
                print(f"[ElevationMapUI] 独立 UI 启动失败: {exc}")

    # -- 摆动腿观测 (第一步) ------------------------------------------- #
    def _update_swing_obs(self) -> None:
        """每步组装摆动观测到 ``self.swing_obs`` (无轨迹生成)."""
        if not self.cfg.swing_obs_enable or self._swing_builder is None:
            return
        from myrobot_mjlab.task.DTC.tamols.swing_leg import (
            LocalGridHeights,
            footholds_from,
        )

        data = self.robot.data
        foot_ids = self._foot_body_ids()
        foot_pos_w = data.body_link_pos_w[:, foot_ids, :]      # [N,4,3]
        base_pos_w = data.root_link_pos_w                       # [N,3]
        quat = data.root_link_quat_w                            # [N,4] w,x,y,z
        yaw = torch.atan2(
            2.0 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
            1.0 - 2.0 * (quat[:, 2] ** 2 + quat[:, 3] ** 2),
        )
        heights = LocalGridHeights(
            # h_raw 的布局是 [N, 1, Nx, Ny] (x=第2维, y=第3维);
            # LocalGridHeights 的约定是 [N, 1, Ny, Nx] (x=最后一维),
            # 必须转置, 否则 x/y 会被当成错误的网格轴 (非方阵时完全错位).
            self.h_raw.transpose(2, 3),
            x_min=self.x_min, y_min=self.y_min, dx=self.cell_size,
        )
        self.swing_obs = self._swing_builder.update(
            foot_pos_w=foot_pos_w,
            footholds_w=footholds_from(self.optimal_footholds, device=self.device),
            base_pos_w=base_pos_w,
            base_yaw_w=yaw,
            heights=heights,
        )

    def _foot_body_ids(self) -> list[int]:
        """解析脚 body 下标 (顺序 FL/FR/RL/RR)."""
        names = list(self.robot.body_names)
        ids = []
        for wanted in self.cfg.swing_foot_bodies:
            for cand in (wanted, wanted.replace("_foot", ""), f"{wanted}_link"):
                if cand in names:
                    ids.append(names.index(cand))
                    break
            else:
                raise ValueError(
                    f"无法解析脚 body {wanted!r}; 可用: {names}"
                )
        return ids

    # -- 生命周期钩子 (父类对应方法保持原行为) --------------------------- #
    def _update_metrics(self) -> None:
        """更新摆动观测和独立高程图 UI, 不改规划行为."""
        super()._update_metrics()
        self._update_swing_obs()
        elevation_map_ui = getattr(self, "_elevation_map_ui", None)
        if elevation_map_ui is not None:
            elevation_map_ui.update(self)

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        if self.cfg.swing_obs_enable and getattr(
            self, "_swing_builder", None
        ) is not None:
            self._swing_builder.reset(env_ids)
        if getattr(self, "_nlp_planner", None) is not None:
            self._nlp_planner.reset(env_ids)

    # -- viewer 课程难度控制 (复制自 track 的 AllTrackingCommand 方案) ---- #
    def create_gui(
        self,
        name: str,
        server: viser.ViserServer,
        get_env_idx: Callable[[], int],
        on_change: Callable[[], None] | None = None,
        request_action: Callable[[str, Any], None] | None = None,
    ) -> None:
        """Viser 面板中的地形课程难度下拉框 (stair/tire/rough 变体生效).

        机制与 track 的 ``AllTrackingCommand.create_gui`` 相同:
        下拉选择写入 ``env.specified_terrain_level`` (int 或 "random"),
        随后请求一次 RESET; ``reset_root_state_from_stair`` 在 reset 时
        读取该值, 把 ``terrain.terrain_levels`` 钉在所选难度并同步
        ``env_origins`` (见 config/event/stair_reset.py 第 0 步)。
        plane 平地 (无 terrain_origins) 不创建控件。
        """
        terrain = getattr(self._env.scene, "terrain", None)
        if terrain is None or getattr(terrain, "terrain_origins", None) is None:
            return
        num_levels = terrain.terrain_origins.shape[0]
        # 通用等级标签: DTC 楼梯为 10 级 0~0.30 m (STAIRS_TERRAINS_CFG),
        # 与 track 的 0.03 m/级不同, 不照搬其硬编码米数
        options = [f"Level {i} / {num_levels - 1}" for i in range(num_levels)]
        options.append("Random (随机难度)")

        init_lvl = 0
        specified = getattr(self._env, "specified_terrain_level", None)
        if specified is not None:
            if str(specified).lower() == "random":
                init_lvl = len(options) - 1
            else:
                try:
                    init_lvl = min(max(0, int(specified)), num_levels - 1)
                except (ValueError, TypeError):
                    pass
        elif hasattr(terrain, "terrain_levels"):
            env_idx = min(get_env_idx(), len(terrain.terrain_levels) - 1)
            init_lvl = min(
                max(0, int(terrain.terrain_levels[env_idx].item())),
                num_levels - 1,
            )

        with server.gui.add_folder("Terrain Difficulty (课程难度)"):
            dropdown = server.gui.add_dropdown(
                "Difficulty Level",
                options=options,
                initial_value=options[init_lvl],
            )

            def _on_dropdown_change(_) -> None:
                selected = dropdown.value
                if selected == options[-1]:
                    self._env.specified_terrain_level = "random"
                else:
                    self._env.specified_terrain_level = options.index(selected)
                if request_action is not None:
                    request_action("RESET")

            dropdown.on_update(_on_dropdown_change)

    # -- 在 MuJoCo/Viser 场景显示规划落点 ------------------------------ #
    def _debug_vis_impl(self, visualizer) -> None:
        # 红=优化落点, 橙=当前摆动腿锁定的 landing target, 蓝=nominal 点.
        draw_command_footholds(visualizer, self)
