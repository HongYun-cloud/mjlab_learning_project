"""集成 NLP 的待核实假设配置 (Assumptions Config).

本模块把论文/工程之间**尚未由权威来源确认**的约定集中成一个配置对象,
要求:
  1. 每个选项都有明确的当前采用值、来源和可能影响;
  2. :meth:`AssumptionsConfig.describe` 的文本必须进入求解日志/报告,
     不允许把待核实项伪装成已验证事实;
  3. 默认值保持"论文字面"或"工程现状", 不做私自翻转; 需要其他约定时
     由调用方显式配置 (例如 ``inertial_accel_sign="contact_normal_explicit"``).

各待核实项的现状 (与 ``v_a_constraints`` / ``giac_constraints`` 模块文档
中的结论一致, 这里只引用, 不重复推导):

``inertial_accel_sign``
    论文定义 ``a_B = g - p_B_ddot``. 字面约定下静止站立时
    ``a_B = [0,0,-9.81]``: 摩擦锥 (式 5/17a) 要求 ``a_B_z >= 0`` 而被
    违反; GIAC (式 17b) 在**全无序对**枚举下对任意绕行方向都不可行
    (闭合边方向反号 + 对角线对切割支撑多边形内部, 已数值验证).
    ``"contact_normal_explicit"`` 表示调用方显式改用物理上自洽的
    ``a_B' = p_B_ddot - g`` (等价于整体取负), **不是**默认值.
    影响: 全部动力学约束的可行集.

``giac_pairs``
    论文 (17b) 写 "对每一对支撑脚". 数值验证表明全无序对枚举在静止
    站立时不可行 (原因见上); 支撑锥的边界只对应支撑多边形的**有向
    边**, 因此默认改用沿绕行方向的循环边对 (三角形时循环边恰为全部
    三对). 这是对论文字面的解释性偏离, 标记为待核实.
    影响: N >= 3 的 GIAC 不等式集合.

``giac_winding``
    循环边的绕行方向必须与 ``inertial_accel_sign`` 匹配: 字面 a_B 配
    逆时针 (CCW), contact-normal a_B' 配顺时针 (CW) (静态站立数值验证,
    见 :meth:`AssumptionsConfig.resolved_winding`); ``"auto"`` 按该规则
    自动选择. 腿的子集顺序取工程逆时针环 (FL,RL,RR,FR) 的子序列
    (由步态时序决定, 不用符号变量排序).
    影响: (17b)/(17c)/(17d) 的符号.

``include_push``
    推地条件 (V-A-4) 与字面摩擦锥在静止状态互相矛盾 (v_a 模块文档),
    默认 **False**: 未确认符号前不把两者组合成完整优化的"安全保证".
    影响: 是否加入 ``ez.p_ddot >= ez.g``.

``friction_mode``
    ``"enabled"`` 时按 ``inertial_accel_sign`` 加入摩擦锥; ``"disabled"``
    用于在符号确认前把该待核实项整体关闭 (静止基准测试用).
    影响: 式 (5)/(17a).

``convexity_fourth_term``
    式 (30) 第四项论文印刷为 ``(p20 x p24)``, 要求不存在的脚 0.
    ``"p21_engineering"`` 采用 v_a 模块修正的 ``(p21 x p24)`` (平移不
    变, 有几何含义); ``"p20_literal"`` 保留字面版本 (p0 取世界原点),
    仅用于对照测试 (平移后会改变, 见 test_nlp_integration). 默认值
    标记为待核实.
    影响: 足迹凸性不等式第 4 条.

``hip_offsets``
    ``"stance"`` 用工程名义支撑中心 (髋横向 0.0465 + 大腿展宽 0.0955);
    ``"joint"`` 用论文式 (31) 字面的髋旋转中心 (HAA/HFE 轴交点).
    影响: 式 (31)/(33)/(34) 的 r_i^B.

``map_out_of_bounds`` / ``map_fill``
    地图查询的越界与缺失策略, 见 :mod:`map_query` 模块文档. 符号查询
    恒为 clamp (IPOPT 需要有限值), 该策略只作用于数值查询与诊断.
    影响: 越界/空洞处的 h 与梯度取值.

``num_samples`` / ``sample_policy``
    连续约束/目标的采样: 每相位 ``num_samples`` 个点,
    ``linspace(0, T, n+1)[:-1]`` (含起点不含终点, 与 V-A 模块一致);
    采样类目标权重乘 ``T_k = T / num_samples`` (论文 τk/6).
    影响: 约束条数与 sampled 目标权重.

``use_giac_slack``
    论文式 (29): GIAC 族约束 (17b)-(17f) 写成 slacked 不等式
    ``c_dyn <= eps, eps <= 0``, 每相位一个 eps, 目标权重 0.007
    (Table I robustness margin). ``True`` 时松弛变量是决策变量;
    ``False`` 时退化为精确不等式 (giac 模块现状).
    影响: 变量布局与鲁棒裕度.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

InertialAccelSign = Literal["paper_literal", "contact_normal_explicit"]
GiacPairs = Literal["cyclic_edges", "all_unordered"]
GiacWinding = Literal["auto", "ccw", "cw"]
FrictionMode = Literal["enabled", "disabled"]
FrictionConeForm = Literal["paper_literal", "force_balance_derived"]
ConvexityFourth = Literal["p21_engineering", "p20_literal"]
HipOffsets = Literal["stance", "joint"]
MapOutOfBounds = Literal["nan", "clamp", "raise"]
MapFill = Literal["median", "value", "raise"]

SAFETY_NOTE = (
    "SAFETY: 摩擦锥符号 UNRESOLVED (论文印刷式 (5) 与其式 (1)(3)(4)/V-A-4 "
    "在静止点互斥, 见 friction_sign.py 审计表). 默认 paper_literal 下静止 "
    "基准不可行且不作修复; friction_mode='disabled' 与 "
    "inertial_accel_sign='contact_normal_explicit' 为显式诊断配置, "
    "**未验证, 不可用于安全结论**; friction_cone_form="
    "'force_balance_derived' 为论文内部证据派生的修正, 待作者勘误确认, "
    "同样不可用于安全结论."
)


@dataclass(frozen=True)
class AssumptionsConfig:
    """集成 NLP 的全部待核实/可配置约定. 见模块文档逐项说明."""

    inertial_accel_sign: InertialAccelSign = "paper_literal"
    giac_pairs: GiacPairs = "cyclic_edges"
    giac_winding: GiacWinding = "auto"
    include_push: bool = False
    friction_mode: FrictionMode = "enabled"
    friction_form: Literal["norm", "squared"] = "squared"
    friction_cone_form: FrictionConeForm = "paper_literal"
    """摩擦锥公式形式 (只影响摩擦锥及其法向伴生, 不影响 GIAC/推地):

    - ``"paper_literal"`` (默认): 论文印刷式 (5), 以 aB = g - p̈ 代入
      ``||aB_xy||^2 - (μ aB_z)^2 <= 0`` 加 ``-aB_z <= 0``; 静止点不可行
      (法向残差 +9.81), 该事实被基准如实报告, 不作修复;
    - ``"force_balance_derived"``: 受力平衡派生修正 (friction_sign.py
      式 C-three): ``||aB_xy||^2 - (μ(-aB_z))^2 <= 0`` 加 ``aB_z <= 0``;
      依据为论文自身式 (1)(3)(4)+V-A-4 的内部一致性证明 (v1/v2 同印刷,
      arXiv 无勘误, 作者代码为补丁不作权威), 标记为待外部确认.
    """
    mu: float = 0.6
    convexity_fourth_term: ConvexityFourth = "p21_engineering"
    hip_offsets: HipOffsets = "stance"
    map_out_of_bounds: MapOutOfBounds = "clamp"
    map_fill: MapFill = "median"
    num_samples: int = 6
    sample_policy: str = "linspace(0,T,n+1)[:-1]"
    use_giac_slack: bool = True

    def resolved_winding(self) -> Literal["ccw", "cw"]:
        """``giac_winding="auto"`` 时按符号约定解析出的绕行方向.

        数值验证 (静态站立, 基座在支撑多边形内部): 字面 ``a_B = g`` 配
        **CCW** 循环边时 (17b) 全部 <= 0; ``a_B' = p_ddot - g`` 配 **CW**
        循环边时全部 <= 0. 两种组合之外的搭配在静态即违反 GIAC.
        """
        if self.giac_winding != "auto":
            return self.giac_winding
        return "ccw" if self.inertial_accel_sign == "paper_literal" else "cw"

    def describe(self) -> str:
        """供求解日志/报告使用的待核实假设清单 (值 + 来源 + 影响)."""
        rows = [
            (
                "inertial_accel_sign",
                self.inertial_accel_sign,
                "论文 III/式(17) 字面 vs 物理自洽取负 (v_a 模块文档)",
                "全部动力学约束可行集; 字面约定下静止摩擦锥不可行",
            ),
            (
                "giac_pairs",
                self.giac_pairs,
                "论文 (17b) '每对支撑脚' 的字面 vs 支撑多边形有向边解释",
                "N>=3 的 GIAC 不等式集合; 全无序对在静止站立不可行(已数值验证)",
            ),
            (
                "giac_winding",
                f"{self.giac_winding}->{self.resolved_winding()}",
                "giac 模块文档: 不等式方向依赖支撑多边形绕行方向",
                "(17b)/(17c)/(17d) 符号",
            ),
            (
                "include_push",
                self.include_push,
                "v_a 模块: 字面摩擦锥与推地条件静止时互相矛盾",
                "是否加入 ez.p_ddot >= ez.g; 默认不组合为安全保证",
            ),
            (
                "friction_mode",
                self.friction_mode,
                "符号未确认前允许整体关闭该待核实项; "
                "disabled = 未验证, 不可用于安全结论",
                "式 (5)/(17a) 是否加入",
            ),
            (
                "friction_form",
                self.friction_form,
                "v_a/giac: squared 在法向伴生成立前提下与 norm 等价且光滑",
                "摩擦锥残差形式",
            ),
            (
                "friction_cone_form",
                self.friction_cone_form,
                "paper_literal = 论文印刷 (5); force_balance_derived = "
                "论文式 (1)(3)(4)+V-A-4 内部一致性证明派生 (friction_sign.py), "
                "待作者勘误确认, 不可用于安全结论",
                "只改摩擦锥与法向伴生; GIAC/推地不受影响",
            ),
            ("mu", self.mu, "工程经验值, 论文未给 Go2 数值", "摩擦锥开口"),
            (
                "convexity_fourth_term",
                self.convexity_fourth_term,
                "论文式 (30) 印刷 p20 vs v_a 修正 p21 (平移不变性论证)",
                "足迹凸性第 4 条; p20 字面版平移后改变(见测试)",
            ),
            (
                "hip_offsets",
                self.hip_offsets,
                "工程名义支撑中心 vs 论文式 (31) 髋旋转中心",
                "式 (31)/(33)/(34) 的 r_i^B",
            ),
            (
                "map_out_of_bounds",
                self.map_out_of_bounds,
                "map_query 模块文档; 符号查询恒 clamp",
                "越界处 h/梯度取值",
            ),
            (
                "map_fill",
                self.map_fill,
                "map_query 模块文档; 与 map_process.fill_invalid 同策略",
                "缺失格处 h/梯度取值",
            ),
            (
                "num_samples/sample_policy",
                f"{self.num_samples}/{self.sample_policy}",
                "论文 V-C.3 'Tk = τk/6' 与 V-A 模块采样约定",
                "约束条数与 sampled 目标权重 (x T_k)",
            ),
            (
                "use_giac_slack",
                self.use_giac_slack,
                "论文式 (29) + Table I robustness margin 0.007",
                "松弛变量是否为决策变量; 可行集不变(eps 为辅助变量)",
            ),
        ]
        lines = ["AssumptionsConfig (UNVERIFIED items marked *):"]
        for name, value, source, impact in rows:
            lines.append(
                f"  * {name:<24s} = {value!s:<28s} source: {source}\n"
                f"    {'':26s} impact: {impact}"
            )
        lines.append(SAFETY_NOTE)
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "inertial_accel_sign": self.inertial_accel_sign,
            "giac_pairs": self.giac_pairs,
            "giac_winding": self.giac_winding,
            "include_push": self.include_push,
            "friction_mode": self.friction_mode,
            "friction_form": self.friction_form,
            "friction_cone_form": self.friction_cone_form,
            "mu": self.mu,
            "convexity_fourth_term": self.convexity_fourth_term,
            "hip_offsets": self.hip_offsets,
            "map_out_of_bounds": self.map_out_of_bounds,
            "map_fill": self.map_fill,
            "num_samples": self.num_samples,
            "sample_policy": self.sample_policy,
            "use_giac_slack": self.use_giac_slack,
        }


DEFAULT_ASSUMPTIONS = AssumptionsConfig()
"""默认配置: 全部保持论文字面/工程现状, 不做私自翻转."""
