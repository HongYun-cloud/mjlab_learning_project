"""摩擦锥符号审计 + 受力平衡派生修正 + 物理基准用例.

================================================================
A. 受力平衡推导 (世界系 z 向上, g = [0, 0, -9.81])
================================================================
机器人质量 m, 接触力 f_i (地面作用于机器人). 静止时 p̈_B = 0, 线动量
方程 (论文式 (1) 线性部分)::

    Σ_i f_i = m (p̈_B - g)                      ... (1-linear)

静止: Σ f_i = -m g = [0, 0, +9.81 m] -> **接触合力必须向上**.
单接触 Coulomb 锥 (论文式 (3), 水平面 n = ez)::

    F_i = { f | μ ez.f >= ||(I - ez ez^T) f|| }   (隐含 ez.f >= 0, 只能推)

锥是凸锥 => 合力 Σ f_i 也在锥内::

    μ ez.(Σ f) >= ||(Σ f)_xy||,   ez.(Σ f) >= 0
    代入 Σ f = m(p̈ - g):
    μ ez.(p̈ - g) >= ||(p̈ - g)_xy||,   ez.(p̈ - g) >= 0      ... (C-one)

用论文的 GIA 符号 aB := g - p̈ (论文第 4 页 nomenclature) 改写 (C-one)::

    ||aB_xy|| + μ aB_z <= 0,   aB_z <= 0                      ... (C-two)

即平方形式 (在 aB_z <= 0 下等价)::

    ||aB_xy||^2 - (μ (-aB_z))^2 <= 0,   aB_z <= 0             ... (C-three)

================================================================
B. 审计表 (论文原式 / 代码 / 静止数值 / 受力平衡预期 / 差异)
================================================================
+---------+---------------------------+---------------------------+------------------+------------------+-----------------------------+
| 项      | 论文原式                  | 代码表达式 (literal 默认) | 静止点数值       | 受力平衡预期     | 差异与可能原因              |
+=========+===========================+===========================+==================+==================+=============================+
| 动量    | (1) m(p̈-g)=Σf             | 不单独装配; GIAC 用其     | Σf=+9.81 m ez    | 接触合力向上     | 无                          |
| (1)     |                           | 行列式形式 (11)           | (向上)           |                  |                             |
+---------+---------------------------+---------------------------+------------------+------------------+-----------------------------+
| 力锥    | (3) μ n.f >= ||(I-nn)f||  | contact_force_cone_       | -μ*9.81*m <= 0   | 满足             | 无 (本模块新增, 用作物理    |
| (3)     | (n=ez, 对合力即 (C-one))  | residuals (本模块)        | 满足             |                  | 基准的独立判据)             |
+---------+---------------------------+---------------------------+------------------+------------------+-----------------------------+
| 锥隶属  | (4) p̈-g ∈ F               | 不单独装配                | 0-μ*9.81 <= 0    | 满足             | 无                          |
| (4)     |                           |                           | 满足             |                  |                             |
+---------+---------------------------+---------------------------+------------------+------------------+-----------------------------+
| 摩擦锥  | (5) μ ez.(g-p̈) >=         | v_a.friction_squared_     | 锥 -34.64 满足;  | 法向应 aB_z<=0;  | **印刷 (5) 与 (1)(3)(4)     |
| (5)     | ||(I-ez ez')(g-p̈)||       | residual(aB=g-p̈) +        | 法向 +9.81 违反  | 锥应 ||aB_xy||^2 | 除 p̈=g 外不可同真: F 是尖   |
|         |                           | 法向 -aB_z <= 0           |                  | -(μ(-aB_z))^2<=0 | 锥, (4)(5) 同持 => p̈=g.     |
|         |                           |                           |                  |                  | v1/v2 同印刷, arXiv 无勘误; |
|         |                           |                           |                  |                  | 判定为 (5) 符号笔误         |
+---------+---------------------------+---------------------------+------------------+------------------+-----------------------------+
| GIAC    | (11) m det(p_ij,p_B-p_i,  | giac_constraints.giac_    | 全 <= 0 满足     | 满足 (CoM 在支撑 | 无; 需配 CCW 循环边 (已数值   |
| (11)    | g-p̈) <= p_ij.L̇            | residual, aB=g-p̈          | (max|.|=8.367)   | 多边形内)        | 验证); 与摩擦符号问题无关   |
+---------+---------------------------+---------------------------+------------------+------------------+-----------------------------+
| 推地    | V-A-4 ez.p̈ >= ez.g        | v_a.friction_push_records | 0 >= 0 满足      | 满足; 且与修正锥 | 无; 字面 (5) 下与 (5) 互斥,  |
| V-A-4   |                           | (include_push 默认 False) | (等式)           | 法向伴生 aB_z<=0 | 修正锥下二者一致(冗余)      |
|         |                           |                           |                  | 完全等价         |                             |
+---------+---------------------------+---------------------------+------------------+------------------+-----------------------------+
| 作者代码| -                         | constraints.py:76-82      | 静止锥 = 0 (等式,| -              | **不作权威依据**: 锥用裸 p̈  |
| (参考)  |                           | "(mu*a_B[2])^2 - ...      | 零裕度), 且为双锥|                  | ("FIXED" 注释), GIAC 也用 p̈ |
|         |                           | >= 0", a_B=p̈ (helpers:16) | (无法向符号);    |                  | (与论文 (11) 的 g-p̈ 不符);  |
|         |                           |                           | push: p̈_z>=-9.81 |                  | 仅其 push 与 V-A-4/派生一致 |
+---------+---------------------------+---------------------------+------------------+------------------+-----------------------------+

================================================================
C. 证据来源 (含未能确认项)
================================================================
1. 论文 PDF (本地, arXiv v2): /home/hong/下载/TAMOLS.pdf; 提取文本行号:
   式 (1)(3): /tmp/tamols.txt 约 255-300 行; 式 (4): 314 行; 式 (5): 320 行;
   式 (11): 约 310 行; V-A-4 push: 约 528 行; aB 定义: 约 250 行.
2. arXiv 摘要页 https://arxiv.org/abs/2206.14049 : v1 2022-06-28,
   v2 2022-07-05, **无勘误**; ar5iv HTML v1 与 v2 均确认式 (5) 印刷为
   (g - p̈_B) 形式 (https://ar5iv.labs.arxiv.org/html/2206.14049v1 / v2).
3. 作者公开实现 (本机): /home/hong/Tamols/tamols-rl/go2-hrl/fetch/tamols/
   constraints.py:76-82, 183-184; helpers.py:16-17. 结论: 补丁代码, 不作
   权威依据 (见上表最后一行).
4. 作者勘误/回应: **未能确认** (arXiv 无 errata; 未联系作者).

================================================================
D. 修正策略 (C 节纪律)
================================================================
- 默认配置不变: ``friction_cone_form="paper_literal"`` 保留论文印刷 (5),
  静止基准按字面不可行这一事实继续如实报告.
- 新增 ``friction_cone_form="force_balance_derived"``: 只改**摩擦锥及其
  法向伴生** (式 (C-three)), 依据是上面的内部一致性证明; **不改 GIAC**
  (式 (11) 与 aB=g-p̈ + CCW 在物理上自洽), **不改推地** (V-A-4 本就正确,
  修正后与法向伴生等价, 保留为冗余项以便审计).
- 不使用一个全局 sign 开关同时翻转所有公式: GIAC 与摩擦锥的符号需求
  不同 (GIAC 用 aB 本身, 摩擦锥用 -aB 方向), 派生修正只作用于摩擦锥.
- 原始印刷公式保留: v_a_constraints.friction_* 与 VBWeights/记录体系不变,
  本模块 literal 函数与其数值一致 (有测试对照).
- 安全标注: 摩擦符号仍标记 UNRESOLVED 直至作者勘误; 派生修正标记为
  "论文内部证据派生, 待外部确认"; disabled / contact_normal_explicit
  标记 "未验证, 不可用于安全结论".

本模块的物理基准测试能验证什么 / 不能验证什么
----------------------------------------------
能: 合力锥一致性 -- 对给定 p̈, 由 (1-linear) 恢复合力 F = m(p̈-g), 检查
    F 是否在上向锥内 (独立 NumPy), 并核对各摩擦残差形式的符号与之一致;
    边界/锥外/自由落体等解析可判用例的残差符号与量值.
不能: 逐接触力分配 -- NLP 决策变量只有样条系数/落脚点/松弛, 不含接触力
    (contact-force-free 模型, 论文 Assumption 1-2); 因此基准不能验证单脚
    力是否各自在锥内, 只能验证合力必要条件; GIAC/松弛提供的是裕度而非力.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from myrobot_mjlab.task.DTC.tamols.giac_constraints import INEQUALITY
from myrobot_mjlab.task.DTC.tamols.v_a_constraints import VAConstraint

GRAVITY = (0.0, 0.0, -9.81)


# ---------------------------------------------------------------------- #
# 合力与力级锥 (物理判据, 独立于任何论文公式形式)
# ---------------------------------------------------------------------- #
def net_contact_force(
    mass: Any, p_ddot: Sequence[Any], g_vec: Sequence[Any] = GRAVITY
) -> tuple[Any, Any, Any]:
    """``F = m (p̈ - g)``: 式 (1-linear) 恢复的地面接触合力 (作用于机器人)."""
    return (
        mass * (p_ddot[0] - g_vec[0]),
        mass * (p_ddot[1] - g_vec[1]),
        mass * (p_ddot[2] - g_vec[2]),
    )


def contact_force_cone_residuals(
    F: Sequence[Any], mu: Any
) -> tuple[Any, Any]:
    """力级锥判据 (式 (3) 对合力): 返回 ``(cone, normal)``, 均要求 <= 0::

        cone   = ||F_xy|| - μ F_z
        normal = -F_z
    """
    cone = (F[0] ** 2 + F[1] ** 2) ** 0.5 - mu * F[2]
    return cone, -F[2]


# ---------------------------------------------------------------------- #
# 两种 aB 空间的摩擦残差 (literal = 论文印刷 (5); derived = 受力平衡派生)
# ---------------------------------------------------------------------- #
def literal_cone_residuals(
    a_B: Sequence[Any], mu: Any
) -> tuple[Any, Any]:
    """论文印刷式 (5) 的平方形式 + 法向伴生 (与 v_a 模块数值一致)::

        cone   = ||aB_xy||^2 - (μ aB_z)^2      (<= 0)
        normal = -aB_z                         (<= 0, 即要求 aB_z >= 0)
    """
    cone = a_B[0] ** 2 + a_B[1] ** 2 - (mu * a_B[2]) ** 2
    return cone, -a_B[2]


def derived_cone_residuals(
    a_B: Sequence[Any], mu: Any
) -> tuple[Any, Any]:
    """受力平衡派生锥 (式 C-three) + 法向伴生::

        cone   = ||aB_xy||^2 - (μ (-aB_z))^2   (<= 0)
        normal = aB_z                          (<= 0, 即要求 aB_z <= 0)

    在 normal 成立的前提下与 ``||aB_xy|| <= μ(-aB_z)`` 完全等价; 该方向
    与接触合力 F = -m aB 在上向锥内 (式 C-one) 等价.
    """
    cone = a_B[0] ** 2 + a_B[1] ** 2 - (mu * (-a_B[2])) ** 2
    return cone, a_B[2]


def friction_records_derived(
    a_B: Sequence[Any],
    p_ddot: Sequence[Any],
    mu: Any,
    *,
    include_push: bool = False,
    phase: int | None = None,
    g_vec: Sequence[Any] = GRAVITY,
) -> tuple[VAConstraint, ...]:
    """派生摩擦锥的约束记录 (类别仍记 "5", 名称带 ``fb`` 前缀).

    ``include_push=True`` 时额外加入 V-A-4 推地残差 ``g_z - p̈_z``; 注意
    它在代数上与派生法向伴生 ``aB_z = g_z - p̈_z`` **完全相同** (冗余,
    保留仅为与论文 V-A 清单对齐并便于审计).
    """
    cone, normal = derived_cone_residuals(a_B, mu)
    records = [
        VAConstraint(
            name="5fb_cone", category="5", kind=INEQUALITY, residual=cone,
            phase=phase, bound=0.0,
        ),
        VAConstraint(
            name="5fb_normal", category="5", kind=INEQUALITY, residual=normal,
            phase=phase, bound=0.0,
        ),
    ]
    if include_push:
        records.append(
            VAConstraint(
                name="V-A-4_push", category="V-A-4", kind=INEQUALITY,
                residual=g_vec[2] - p_ddot[2], phase=phase, bound=0.0,
            )
        )
    return tuple(records)


# ---------------------------------------------------------------------- #
# 物理基准用例 (解析可判)
# ---------------------------------------------------------------------- #
@dataclass(frozen=True)
class PhysicsCase:
    name: str
    p_ddot: tuple[float, float, float]
    mu: float
    expect_force_cone_ok: bool
    """合力 F=m(p̈-g) 是否在力级锥内 (锥残差 <= 0)."""
    expect_force_normal_ok: bool
    """合力法向 F_z >= 0 (只能推)."""
    note: str = ""


def physics_cases(mu: float = 0.6) -> tuple[PhysicsCase, ...]:
    g = 9.81
    return (
        PhysicsCase("static_standing", (0.0, 0.0, 0.0), mu, True, True,
                    "F = +m g ez 向上, 锥内且有裕度"),
        PhysicsCase("vertical_up_acc", (0.0, 0.0, 2.0), mu, True, True,
                    "F_z = m(g+2) > m g, 纯法向"),
        PhysicsCase("vertical_down_acc", (0.0, 0.0, -2.0), mu, True, True,
                    "F_z = m(g-2) > 0, 仍支撑"),
        PhysicsCase("horizontal_on_cone_boundary", (mu * g, 0.0, 0.0), mu,
                    True, True, "||F_xy|| = μ F_z 恰在锥面 (残差 0)"),
        PhysicsCase("horizontal_outside_cone", (1.5 * mu * g, 0.0, 0.0), mu,
                    False, True, "||F_xy|| = 1.5 μ F_z 锥外"),
        PhysicsCase("freefall_like", (0.0, 0.0, -12.0), mu, False, False,
                    "F_z = m(g-12) < 0: 无法推地; 力级锥残差 = -μ F_z > 0 "
                    "同样违反 (norm 形式在 F_z<0 时自动报警)"),
    )


def force_balance_verdict(
    case: PhysicsCase, mass: float = 15.2064
) -> tuple[float, float]:
    """独立 NumPy: 合力锥残差与法向残差 (<=0 表示满足)."""
    F = net_contact_force(mass, case.p_ddot)
    cone, normal = contact_force_cone_residuals(F, case.mu)
    return float(cone), float(normal)
