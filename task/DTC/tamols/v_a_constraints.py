"""TAMOLS 论文 V-A 节的另外三组约束.

本模块实现论文 V-A 中除 GIAC 之外的约束:

3. 足迹凸性 / Convex GIAC (式 (30));
4. 摩擦锥 (式 (5), 等价于 (17a)) 与推地条件 (V-A-4);
5. 腿部运动学可达范围 (式 (31)).

GIAC (17b)-(17f) 已经在本包的 :mod:`giac_constraints` 中实现并通过测试,
本模块**不修改**该模块, 只在需要时复用其中的残差函数. 初始位姿和相邻
相位连接约束在 :mod:`optimization_problem` 中, 同样不改动.

统一约定
--------
所有不等式都写成 ``g(x) <= 0``. 每条约束通过 :class:`VAConstraint`
记录名称、论文式编号、残差、脚索引、相位和边界, 数值残差可用
:meth:`VAConstraintSet.residual_values` /
:meth:`VAConstraintSet.max_violation_by_category` 查看.

论文原式 / 代码公式 / 变量定义
------------------------------
世界坐标系: x 前, y 左, z 向上 (与工程一致).

变量定义::

    g   = [0, 0, -9.81]^T        重力向量, 代码 GRAVITY
    ez  = [0, 0, 1]^T            水平地面法向, 代码 EZ
    p_B                          机身 (base) 世界系位置
    p_B_ddot                     机身世界系线加速度
    a_B = g - p_B_ddot           论文定义, 代码中作为显式参数 a_B
    L_dot                        机身角动量变化率 (仅 GIAC 使用)
    R_B = Rz(yaw) Ry(pitch) Rx(roll)   机体系 -> 世界系, 代码 R_B
    r_i^B                        第 i 条腿的髋位置 (机体系), 代码 r_i
    p_i                          第 i 只脚的世界系位置, 代码 footholds
    mu                           摩擦系数; l_min/l_max 可达球半径

注意 ``spline_base_acceleration()`` 严格按论文计算 ``a_B = g - p_B_ddot``;
参考实现里同名变量其实是 ``p_B_ddot`` (见 :mod:`giac_constraints` 文档),
移植时必须显式换号, 不能直接同名照搬.

逐项对照::

    式 (30)   (p13 x p12).ez <= 0            -> 30_convex_p13_p12
              (p14 x p13).ez <= 0            -> 30_convex_p14_p13
              (p24 x p23).ez <= 0            -> 30_convex_p24_p23
              (p20 x p24).ez <= 0            -> 30_convex_p21_p24
                                               (p20 采用 p21, 证据见下节)
    式 (5)    mu ez.a_B >= ||(I-ez ez^T)a_B||
              norm:    ||a_B_xy|| - mu a_B_z <= 0     -> 5_friction_cone
              squared: ||a_B_xy||^2 - (mu a_B_z)^2 <= 0
                                                      -> 5_friction_cone
                       -a_B_z <= 0                    -> 5_friction_normal
    推地      ez.p_B_ddot >= ez.g
              g_z - p_B_ddot_z <= 0                   -> V-A-4_push
    式 (31)   l_min^2 <= ||p_B + R_B r_i^B - p_i||^2
              l_min^2 - d^2 <= 0                      -> 31_lower
              d^2 - l_max^2 <= 0                      -> 31_upper

摩擦锥的法向方向: 锥轴是 **+ez (世界系向上)**, 即接触力只能向上推地面
(论文 Assumption 3). 推地条件的字面含义是 ``p_B_ddot_z >= -9.81``,
**不是** ``p_B_ddot_z >= 0``; 两者相差重力项, 不可混淆.

静止状态 (``p_B_ddot = 0``, 因而 ``a_B = g = [0,0,-9.81]``) 是检验符号
约定最直接的例子: 按论文字面定义, 摩擦锥要求 ``a_B_z >= 0``, 静止时
却被违反; 推地条件 ``p_B_ddot_z = 0 >= -9.81`` 则满足. 因此**字面摩擦锥
与字面推地条件在静止时不可同时成立**, 完整优化中暂时只应单独使用其中
一组 (``include_push`` 默认 False). 若改用物理上常见的
``a_B' = p_B_ddot - g`` (等价于整体取负), 静止时 ``a_B' = [0,0,9.81]``,
摩擦锥满足, 推地条件也满足.

**足端编号与绕行方向**(实现前已核对工程):

- 工程里 ``landing_footholds`` / ``desired_footholds`` 的形状是
  ``(4, 3)``, 没有显式腿名. 工程中对应关系见
  ``task/Tamols/fetch/go2_kinematics.py`` 和
  ``task/Tamols/fetch/tamols_foothold.py``: 索引 0=FL, 1=FR, 2=RL,
  3=RR (与 ``config/reward/test_tamols_reward.py`` 中
  ``optimal_footholds[:, 0..3]`` 的用法一致). 本模块用
  :data:`LEG_ORDER` 明确写出这个顺序.
- 该顺序 **不是** 逆时针. 在工程机体坐标系 (x 前, y 左) 下,
  FL(+x,+y), FR(+x,-y), RL(-x,+y), RR(-x,-y) 依次相连是一个
  "蝴蝶结"(自交), 其有向面积为 0. 因此 **不能** 直接把数组下标
  1,2,3,4 当作论文式 (30) 的 p1..p4.
- 论文式 (30) 要求四个脚点 "counterclockwise ordered". 对工程编号,
  逆时针顺序是 FL -> RL -> RR -> FR, 即数组下标
  ``(0, 2, 3, 1)``, 见 :data:`CCW_INDICES`. 本模块所有式 (30) 接口
  都通过显式 ``order`` 参数把公式里的 p1..p4 映射到输入数组下标,
  默认使用这个经过核对的常量, 不做隐式排序.
- 数值输入时会检查给定顺序的有向面积是否为正 (逆时针); 非正
  (顺时针、自交或共线) 会抛出 ``ValueError``, 并提示改用
  :func:`counterclockwise_order` 或显式传入正确 ``order``. 当脚点是
  CasADi 决策变量时无法做数值检查, 此时由调用方保证 ``order`` 正确,
  约束集 ``note`` 会标记 "winding unchecked (symbolic)".

式 (30) 的印刷错误
------------------
论文第 8 页式 (30) 印刷为::

    (p13 x p12) . ez <= 0
    (p14 x p13) . ez <= 0
    (p24 x p23) . ez <= 0
    (p20 x p24) . ez <= 0

其中 ``p_ij = p_j - p_i``. 前三项只用到脚点 1..4, 但第四项的 ``p20``
要求一个不存在的 "脚 0". 本模块的结论是: 第四项应为
``(p21 x p24) . ez <= 0``, 即印刷的 ``p20`` 是 ``p21`` 的下标笔误.
证据:

1. 论文同一段写明四个脚点按逆时针编号为 {1,2,3,4}, 式 (30) 只应
   涉及这四个点;
2. 前三项分别检查脚 2 和脚 4 位于对角线 p1-p3 的两侧; 第四项与第三项
   配对, 检查脚 1 和脚 3 位于对角线 p2-p4 的两侧. 只有
   ``(p21 x p24)`` 具有这个几何含义;
3. ``(p21 x p24)`` 对任意平移不变, 而 "把 p0 当成原点" 的
   ``(p20 x p24)`` 会随世界原点平移而改变符号, 不可能是有意义的
   凸性条件; 把 p0 当成 p4 则 ``p20 = p24``, 该项恒为 0, 完全不起
   约束作用. 两种解释都不成立.
4. 本模块的专门测试
   ``test_v_a_constraints.TestConvexityP20Ambiguity`` 验证了以上性质.

注意: 参考实现
``Tamols/tamols-rl/go2-hrl/fetch/tamols/constraints.py`` 用的是
``(p24 x p21)`` (符号与本模块相反), 并且它的脚顺序是 FL,FR,RL,RR
(自交), 所以不能作为式 (30) 的符号依据.

非完整支撑
----------
论文在 "没有完整支撑相" 时也应用式 (30): 遍历未来的相位事件, 依次
把落脚点补进来, 直到凑齐四个脚点. 工程当前的
``OptimizationStateLayout`` 只有 ``landing_footholds (4,3)``, 没有
步态/相位事件/支撑相数据结构. 因此本模块 **不猜测** 未来落脚点, 而是
提供 :func:`complete_footprint`: 调用方把当前支撑脚和未来相位事件
(:class:`FootholdEvent`, 必须带 ``phase``) 传进来, 函数按相位顺序补齐,
并在 :class:`CompletedFootprint.sources` 中记录每个脚点来自
"stance" 还是哪个未来相位. 四个脚点仍不足时抛出
:class:`IncompleteFootprintError`, 而不是用未定义值填公式.

摩擦锥与推地条件的符号问题
--------------------------
论文定义 ``a_B = g - p_B_ddot``, ``g = [0, 0, -9.81]^T``. 按字面实现:

- 式 (5)/(17a): ``||a_B_xy|| - mu * a_B_z <= 0``. 因为 ``mu > 0``
  且左端范数非负, 该式要求 ``a_B_z >= 0``, 即 ``p_B_ddot_z <= -9.81``
  (机身向下加速至少 1g);
- 推地条件 (V-A-4): ``ez . p_B_ddot >= ez . g``, 即
  ``p_B_ddot_z >= -9.81``.

两者同时成立只在 ``a_B = 0`` (自由落体, 水平和竖直加速度都等于重力)
时可能. 对受支撑的机器人, 这两个**印刷形式**互相矛盾; 物理上一致的
写法应把接触法向力对应的加速度取为 ``p_B_ddot - g`` (等价于给 a_B
取负). 本模块按用户要求实现**字面公式**, 不擅自翻转符号; 若调用方
使用 ``p_B_ddot - g`` 的约定, 需要自己传入该向量并在注释中说明.
参考实现只实现了推地不等式, 没有实现字面摩擦锥, 所以它没有暴露这个
矛盾 (它的摩擦形式以 ``p_B_ddot`` 作为 "a_B", 等价于本模块整体取负).

静止状态 (``p_B_ddot = 0``, ``L_dot = 0``) 的数值 (见
``v_a_verification.scenario_static_state`` 与
``TestFrictionPush.test_static_state_*``)::

    论文 a_B = g = [0,0,-9.81]:
      norm:    ||0|| - 0.6 * (-9.81) = +5.886  -> 不可行
      squared: 0 - (0.6*(-9.81))^2 = -34.64 (锥满足),
               -a_B_z = +9.81                  -> 不可行
      推地:    g_z - 0 = -9.81                -> 满足
    a_B' = p_ddot - g = [0,0,9.81]:
      norm:    -5.886                          -> 可行
      squared: -34.64, -9.81                   -> 可行
      推地:    -9.81                           -> 满足

因此 ``include_push`` 默认 **False**: 在符号约定通过推导或权威来源
确认之前, 摩擦锥和推地条件应分别测试、分别使用, 不要一起放进完整
优化.

数值实现
--------
- 摩擦锥提供 ``form="norm"`` 和 ``form="squared"``. 平方形式在
  ``a_B_z >= 0`` 前提下与范数形式**完全等价** (不是松弛): 两边都
  非负, 平方是单调的. 因此平方形式必须同时加入 ``a_B_z >= 0``, 代码
  自动添加, 且平方形式处处可微, 在 ``a_B_xy = 0`` 处梯度有限.
  范数形式在 ``a_B_xy = 0`` 处梯度未定义 (CasADi 会得到 NaN), 只建议
  做数值检查或非优化评估, 本模块不做任何 epsilon 平滑.
- 式 (31) 的双边范围拆成两条 ``g(x) <= 0``::

      l_min^2 - ||p_B + R_B r_i^B - p_i||^2 <= 0
      ||p_B + R_B r_i^B - p_i||^2 - l_max^2 <= 0

- 这是两球近似, **不是**完整关节运动学, 也不检查膝/腿碰撞.

接触模式适用性
--------------
- 式 (30): 用于构造 / 修正足迹, 只在能凑齐四个脚点时使用; 与
  当前 ``N`` 无关, 但要求支撑面为水平面 (只取 xy 投影).
- 式 (5): 任意 ``N > 0`` 的接触状态都适用 (N=2/1 时同样需要);
  ``N = 0`` 没有接触, 不加入摩擦约束. 推地条件同样只在 ``N > 0``
  时才可能有意义, 但 ``include_push`` 默认 False, 需调用方显式开启;
  在字面符号下它与摩擦锥冲突, 目前应分开测试/使用.
- 式 (31): 只对**着地腿**施加; 摆动腿不要求位于可达球内. 本工程
  没有接触状态数据, 所以 ``stance_legs`` 必须由调用方显式给出.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from myrobot_mjlab.task.DTC.tamols.giac_constraints import (
    EQUALITY,
    INEQUALITY,
    friction_residual,
    friction_squared_residual,
)

# ---------------------------------------------------------------------- #
# 常量
# ---------------------------------------------------------------------- #
GRAVITY = (0.0, 0.0, -9.81)
"""论文约定的重力向量 ``g = [0, 0, -g]^T``, g=9.81 m/s^2."""

EZ = (0.0, 0.0, 1.0)

LEG_ORDER = ("FL", "FR", "RL", "RR")
"""工程中 ``landing_footholds`` 的索引顺序 (0=FL, 1=FR, 2=RL, 3=RR).

来源: ``task/Tamols/fetch/go2_kinematics.py`` 的
``GO2_HIP_OFFSETS_TRUNK`` 注释、``TAMOLS_FOOTHOLD_CFG.hip_offsets``
以及 ``config/reward/test_tamols_reward.py`` 的下标用法.
"""

LEG_INDEX = {name: index for index, name in enumerate(LEG_ORDER)}

CCW_LEG_ORDER = ("FL", "RL", "RR", "FR")
"""工程四足的逆时针顺序 (x 前, y 左, 从 +z 向下看)."""

CCW_INDICES = tuple(LEG_INDEX[name] for name in CCW_LEG_ORDER)
"""数组下标形式的逆时针顺序 ``(0, 2, 3, 1)``."""

GO2_HIP_JOINT_OFFSETS = np.array(
    [
        [0.1934, 0.0465, 0.0],   # FL
        [0.1934, -0.0465, 0.0],  # FR
        [-0.1934, 0.0465, 0.0],  # RL
        [-0.1934, -0.0465, 0.0], # RR
    ],
    dtype=float,
)
"""Go2 髋关节 (外展轴) 在机体系中的位置 [m].

与参考实现 ``tamols-rl/go2-hrl`` 的 ``hip_offsets`` 一致, 是论文
式 (31) 中 "rotation center of the hip" 的直接解释.
"""

GO2_STANCE_HIP_OFFSETS = np.array(
    [
        [0.1934, 0.145, 0.0],   # FL
        [0.1934, -0.145, 0.0],  # FR
        [-0.1934, 0.145, 0.0],  # RL
        [-0.1934, -0.145, 0.0], # RR
    ],
    dtype=float,
)
"""工程 ``TAMOLS_FOOTHOLD_CFG.hip_offsets`` 使用的名义支撑中心 [m].

它在髋关节横向位置 (0.0465) 上叠加了大腿连杆的横向展宽 (0.0955),
得到名义站立宽度约 0.29 m. 现有落脚点规划器用它作为可达球的中心,
本模块默认也使用它, 以便与工程其它部分一致. 如果希望严格按论文
"髋旋转中心" 建模, 传入 :data:`GO2_HIP_JOINT_OFFSETS`.
"""

DEFAULT_HIP_OFFSETS = GO2_STANCE_HIP_OFFSETS
DEFAULT_L_MIN = 0.15
DEFAULT_L_MAX = 0.40
"""工程 ``TAMOLS_FOOTHOLD_CFG`` 中的腿长上下限 [m]."""


# ---------------------------------------------------------------------- #
# 约束记录
# ---------------------------------------------------------------------- #
@dataclass(frozen=True)
class VAConstraint:
    """一条标量约束及其残差.

    Attributes:
      name: 唯一名称, 例如 ``"30_convex_p13_p12"``.
      category: 论文式编号, ``"30"`` / ``"5"`` / ``"V-A-4"`` /
        ``"31_lower"`` / ``"31_upper"``.
      kind: ``EQUALITY`` (残差 == 0) 或 ``INEQUALITY`` (残差 <= 0).
      residual: 符号表达式或数值.
      foot_indices: 参与该约束的脚在输入数组中的下标, 便于调试.
      foot_sources: 每个参与脚的来源说明 (例如 ``"stance"`` /
        ``"phase 3"``), 与 ``foot_indices`` 等长; 未知时为空元组.
      phase: 该约束所属的相位; None 表示调用方未指定.
      bound: 约束对应的边界值; 式 (31) 中为 ``l_min^2`` /
        ``l_max^2``, 其余一般为 0.
      tau: 相位内的采样时刻 [s]; 式 (31) 采样时记录.
    """

    name: str
    category: str
    kind: str
    residual: Any
    foot_indices: tuple[int, ...] = ()
    foot_sources: tuple[str, ...] = ()
    phase: int | None = None
    bound: float | None = None
    tau: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in (EQUALITY, INEQUALITY):
            raise ValueError(
                f"未知的约束类型 {self.kind!r}; "
                f"可选 {EQUALITY!r} 或 {INEQUALITY!r}"
            )
        if self.foot_sources and len(self.foot_sources) != len(
            self.foot_indices
        ):
            raise ValueError(
                "foot_sources 必须与 foot_indices 等长: "
                f"{self.foot_sources} vs {self.foot_indices}"
            )

    def is_satisfied(
        self,
        atol: float = 1e-9,
        evaluate: Callable[[Any], Any] | None = None,
    ) -> bool:
        """数值判定; 符号残差需要传入 ``evaluate``."""
        value = _numeric_value(self.residual, evaluate)
        if self.kind == EQUALITY:
            return abs(value) <= atol
        return value <= atol


@dataclass(frozen=True)
class VAConstraintSet:
    """一组 V-A 约束的记录."""

    records: tuple[VAConstraint, ...]
    phase: int | None = None
    note: str = ""

    def __iter__(self):
        return iter(self.records)

    def __len__(self) -> int:
        return len(self.records)

    @property
    def equalities(self) -> tuple[VAConstraint, ...]:
        return tuple(
            record for record in self.records if record.kind == EQUALITY
        )

    @property
    def inequalities(self) -> tuple[VAConstraint, ...]:
        return tuple(
            record for record in self.records if record.kind == INEQUALITY
        )

    def by_category(self) -> dict[str, tuple[VAConstraint, ...]]:
        grouped: dict[str, list[VAConstraint]] = {}
        for record in self.records:
            grouped.setdefault(record.category, []).append(record)
        return {key: tuple(value) for key, value in grouped.items()}

    def residual_values(
        self, evaluate: Callable[[Any], Any] | None = None
    ) -> dict[str, float]:
        """``{name: 数值残差}``; 符号残差需传 ``evaluate``."""
        return {
            record.name: _numeric_value(record.residual, evaluate)
            for record in self.records
        }

    def max_violation(
        self, evaluate: Callable[[Any], Any] | None = None
    ) -> float:
        """最大违反量: 不等式取 ``max(0, residual)``, 等式取 ``|residual|``."""
        worst = 0.0
        for record in self.records:
            value = _numeric_value(record.residual, evaluate)
            violation = (
                abs(value)
                if record.kind == EQUALITY
                else max(0.0, value)
            )
            worst = max(worst, violation)
        return worst

    def max_violation_by_category(
        self, evaluate: Callable[[Any], Any] | None = None
    ) -> dict[str, float]:
        result: dict[str, float] = {}
        for category, records in self.by_category().items():
            subset = VAConstraintSet(records)
            result[category] = subset.max_violation(evaluate)
        return result

    def is_satisfied(
        self,
        atol: float = 1e-9,
        evaluate: Callable[[Any], Any] | None = None,
    ) -> bool:
        return self.max_violation(evaluate) <= atol


def _numeric_value(
    value: Any, evaluate: Callable[[Any], Any] | None = None
) -> float:
    if evaluate is not None:
        value = evaluate(value)
    if hasattr(value, "full"):  # CasADi DM
        value = value.full()
    return float(np.asarray(value, dtype=float).reshape(-1)[0])


# ---------------------------------------------------------------------- #
# 通用小工具 (同时支持 numpy / CasADi MX / Python 序列)
# ---------------------------------------------------------------------- #
def _is_symbolic(value: Any) -> bool:
    module = type(value).__module__
    return module == "casadi" or module.startswith("casadi.")


def _sin(value: Any) -> Any:
    if _is_symbolic(value):
        import casadi as ca

        return ca.sin(value)
    return math.sin(float(value))


def _cos(value: Any) -> Any:
    if _is_symbolic(value):
        import casadi as ca

        return ca.cos(value)
    return math.cos(float(value))


def _component(vector: Any, index: int) -> Any:
    shape = getattr(vector, "shape", None)
    if shape is not None and len(shape) == 2:
        rows, cols = int(shape[0]), int(shape[1])
        if rows == 1:
            return vector[0, index]
        return vector[index, 0]
    return vector[index]


def point3(value: Any, name: str = "point") -> tuple[Any, Any, Any]:
    """把任意 3 维点/向量规范化为 3 元组 (支持 CasADi MX)."""
    shape = getattr(value, "shape", None)
    if shape is not None and len(shape) == 2:
        if int(np.prod(shape)) != 3:
            raise ValueError(f"{name} 必须包含 3 个分量, 实际 shape={shape}")
        return tuple(_component(value, i) for i in range(3))
    try:
        size = len(value)
    except TypeError:
        raise TypeError(
            f"{name} 必须是长度为 3 的序列或向量, 实际为 "
            f"{type(value).__name__}"
        ) from None
    if size != 3:
        raise ValueError(f"{name} 必须包含 3 个分量, 实际为 {size}")
    return tuple(value[i] for i in range(3))


def _point_at(points: Any, index: int) -> tuple[Any, Any, Any]:
    """取出第 ``index`` 个点; 也接受单个 3 维点 (此时 index 必须为 0).

    - ``(L, 3)`` 的二维数组/变量: 按行索引;
    - ``(3, 1)`` / ``(1, 3)`` 的单个向量: 只有 index 0 有意义;
    - 一维长度 3 的序列: 单个点;
    - 长度为 L 的点序列: 按位置索引.
    """
    shape = getattr(points, "shape", None)
    if shape is not None and len(shape) == 2:
        rows, cols = int(shape[0]), int(shape[1])
        if cols == 3 and rows != 1:
            return point3(points[index, :])
        if index != 0:
            raise IndexError(
                f"{points} 是单个 3 维向量, 不能用索引 {index} 取点"
            )
        return point3(points)
    if shape is not None and len(shape) == 1:
        if index != 0:
            raise IndexError(
                f"一维向量只能作为单个点使用, 不能用索引 {index}"
            )
        return point3(points)

    first = points[0]
    if np.isscalar(first) or _is_symbolic(first):
        if index != 0:
            raise IndexError(
                f"输入是单个 3 维点, 不能用索引 {index} 取点"
            )
        return point3(points)
    return point3(points[index])


def _sub(a: Sequence[Any], b: Sequence[Any]) -> tuple[Any, Any, Any]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross_z(a: Sequence[Any], b: Sequence[Any]) -> Any:
    return a[0] * b[1] - a[1] * b[0]


def _signed_area_xy(
    points: Sequence[Sequence[Any]],
) -> Any:
    """多边形有向面积 (逆时针为正); 只取 x/y."""
    area = 0.0
    count = len(points)
    for i in range(count):
        x1, y1 = points[i][0], points[i][1]
        x2, y2 = points[(i + 1) % count][0], points[(i + 1) % count][1]
        area = area + (x1 * y2 - x2 * y1)
    return 0.5 * area


def _all_numeric(values: Sequence[Any]) -> bool:
    return not any(_is_symbolic(value) for value in values)


# ---------------------------------------------------------------------- #
# 式 (30): 足迹凸性
# ---------------------------------------------------------------------- #
def convexity_residuals(
    p1: Sequence[Any],
    p2: Sequence[Any],
    p3: Sequence[Any],
    p4: Sequence[Any],
) -> tuple[Any, Any, Any, Any]:
    """式 (30) 的四个残差, 全部要求 ``<= 0``.

    ``p_ij = p_j - p_i``. 第四项采用 :func:`convexity_fourth_residual`
    修正后的 ``(p21 x p24)``, 不是论文印刷的 ``p20``.

    Args:
      p1..p4: 按**逆时针**顺序给出的四个脚点 (至少含 x/y 分量).

    Returns:
      ``(r1, r2, r3, r4)``, 依次对应::

          (p13 x p12) . ez
          (p14 x p13) . ez
          (p24 x p23) . ez
          (p21 x p24) . ez
    """
    p1 = point3(p1, "p1")
    p2 = point3(p2, "p2")
    p3 = point3(p3, "p3")
    p4 = point3(p4, "p4")

    p12, p13 = _sub(p2, p1), _sub(p3, p1)
    p14 = _sub(p4, p1)
    p23, p24 = _sub(p3, p2), _sub(p4, p2)
    p21 = _sub(p1, p2)

    return (
        _cross_z(p13, p12),
        _cross_z(p14, p13),
        _cross_z(p24, p23),
        _cross_z(p21, p24),
    )


def convexity_fourth_residual(
    p1: Sequence[Any],
    p2: Sequence[Any],
    p3: Sequence[Any],
    p4: Sequence[Any],
) -> Any:
    """式 (30) 第四项 ``(p21 x p24) . ez`` (论文 ``p20`` 的修正).

    单独暴露这个函数, 是为了让 :class:`TestConvexityP20Ambiguity`
    能明确验证 "为什么不是 p20".
    """
    p2 = point3(p2, "p2")
    p4 = point3(p4, "p4")
    p1 = point3(p1, "p1")
    return _cross_z(_sub(p1, p2), _sub(p4, p2))


def counterclockwise_order(
    points: Sequence[Sequence[Any]],
) -> tuple[int, ...]:
    """返回把点排成逆时针所需的输入下标.

    只对 4 个点使用 (式 (30) 的场景). 以质心为原点按极角排序;
    对凸四边形和不自交的凹四边形, 这给出边界顺序. 起始点不固定,
    但本函数固定从最小极角开始, 保证结果确定. 需要数值输入.

    Raises:
      ValueError: 点数不是 4, 或所有点重合导致质心无定义.
    """
    if len(points) != 4:
        raise ValueError(f"只支持 4 个点, 实际为 {len(points)} 个")
    coords = [point3(point) for point in points]
    if not _all_numeric([c for p in coords for c in p]):
        raise ValueError("counterclockwise_order 需要数值点, 不能用于符号变量")
    xs = [float(_numeric_value(p[0])) for p in coords]
    ys = [float(_numeric_value(p[1])) for p in coords]
    cx = sum(xs) / 4.0
    cy = sum(ys) / 4.0
    angles = [math.atan2(y - cy, x - cx) for x, y in zip(xs, ys)]
    order = tuple(sorted(range(4), key=lambda i: angles[i]))
    return order


def _validate_order(order: Sequence[int]) -> tuple[int, int, int, int]:
    values = tuple(int(i) for i in order)
    if len(values) != 4 or sorted(values) != [0, 1, 2, 3]:
        raise ValueError(
            "order 必须是 0..3 的一个排列, 实际为 "
            f"{values}. 对工程默认编号 (FL,FR,RL,RR), 逆时针顺序是 "
            f"{CCW_INDICES}."
        )
    return values


def convexity_records(
    footholds: Any,
    *,
    order: Sequence[int] = CCW_INDICES,
    require_ccw: bool = True,
    phase: int | None = None,
    foot_sources: Sequence[str] | None = None,
) -> VAConstraintSet:
    """构造式 (30) 的四条不等式记录, 全部 ``g(x) <= 0``.

    Args:
      footholds: 4 个脚点; 可以是 ``(4, 3)`` 的 numpy 数组 / 规则
        ``(4, 3)`` CasADi 变量, 也可以是长度为 4 的点的序列.
      order: 把公式里的 ``p1..p4`` 映射到 ``footholds`` 下标的排列.
        默认 :data:`CCW_INDICES` = ``(0, 2, 3, 1)``, 对应工程编号
        FL, RL, RR, FR.
      require_ccw: 数值输入时是否检查 ``order`` 给出的四点确实逆时针
        (有向面积 > 0). 符号输入无法检查, 会在 ``note`` 中注明.
      phase: 约束所属相位.
      foot_sources: 长度 4 的来源说明, 按输入下标排列; 会随每条约束
        记录参与脚的来源.

    Returns:
      :class:`VAConstraintSet`, 含 4 条 ``category="30"`` 的不等式.

    Raises:
      ValueError: 点数不是 4、``order`` 非法, 或数值点非逆时针.
    """
    order = _validate_order(order)
    shape = getattr(footholds, "shape", None)
    if shape is not None and len(shape) == 2:
        count = int(shape[0])
    else:
        try:
            count = len(footholds)
        except TypeError:
            raise TypeError(
                "footholds 必须是长度为 4 的序列/二维数组"
            ) from None
    if count != 4:
        raise ValueError(f"式 (30) 需要 4 个脚点, 实际为 {count} 个")

    points = tuple(_point_at(footholds, index) for index in range(4))
    ordered = tuple(points[index] for index in order)

    note_parts = []
    numeric = _all_numeric([value for point in ordered for value in point])
    if numeric:
        area = float(_numeric_value(_signed_area_xy(ordered)))
        if require_ccw and area <= 0.0:
            raise ValueError(
                "式 (30) 要求 p1..p4 逆时针排列, 但给定顺序的有向面积为 "
                f"{area:.6g} <= 0 (顺时针/自交/共线). 工程数组顺序 "
                f"{LEG_ORDER} 不是逆时针; 应使用 order={CCW_INDICES} "
                "(FL,RL,RR,FR), 或先用 counterclockwise_order() 求顺序."
            )
        note_parts.append(f"winding=ccw (signed area={area:.6g})")
    else:
        note_parts.append("winding unchecked (symbolic)")

    residuals = convexity_residuals(*ordered)

    # (名称, 论文中的项, 用到的公式脚下标 1..4)
    specs = (
        ("30_convex_p13_p12", "(p13 x p12).ez", (0, 2, 1)),
        ("30_convex_p14_p13", "(p14 x p13).ez", (0, 3, 2)),
        ("30_convex_p24_p23", "(p24 x p23).ez", (1, 3, 2)),
        ("30_convex_p21_p24", "(p21 x p24).ez", (1, 0, 3)),
    )

    records = []
    for residual, (name, _, formula_indices) in zip(residuals, specs):
        input_indices = tuple(order[i] for i in formula_indices)
        sources = ()
        if foot_sources is not None:
            if len(foot_sources) != 4:
                raise ValueError(
                    "foot_sources 必须与 footholds 一样长 (4), "
                    f"实际为 {len(foot_sources)}"
                )
            sources = tuple(str(foot_sources[i]) for i in input_indices)
        records.append(
            VAConstraint(
                name=name,
                category="30",
                kind=INEQUALITY,
                residual=residual,
                foot_indices=input_indices,
                foot_sources=sources,
                phase=phase,
                bound=0.0,
            )
        )

    return VAConstraintSet(
        records=tuple(records), phase=phase, note="; ".join(note_parts)
    )


def add_convexity_constraints(
    backend: Any,
    footholds: Any,
    *,
    order: Sequence[int] = CCW_INDICES,
    require_ccw: bool = True,
    phase: int | None = None,
    foot_sources: Sequence[str] | None = None,
) -> VAConstraintSet:
    """把式 (30) 的四条不等式加入 ``backend`` (要求 ``<= 0``)."""
    result = convexity_records(
        footholds,
        order=order,
        require_ccw=require_ccw,
        phase=phase,
        foot_sources=foot_sources,
    )
    for record in result.records:
        backend.add_inequality(
            record.residual, -float("inf"), 0.0, name=record.name
        )
    return result


# ---------------------------------------------------------------------- #
# 非完整支撑: 用未来相位事件补齐四个脚点
# ---------------------------------------------------------------------- #
@dataclass(frozen=True)
class FootholdEvent:
    """一个未来的落脚事件.

    Attributes:
      leg: 腿名 (``"FL"`` 等) 或工程下标 (0..3).
      position: 世界系落脚点 (3 分量, 数值或符号).
      phase: 该事件所属的未来相位编号, 用于排序和来源标注.
      kind: 事件类型, 仅供调用方标注 (例如 ``"touchdown"``).
    """

    leg: int | str
    position: Any
    phase: int
    kind: str = "touchdown"


class IncompleteFootprintError(ValueError):
    """未来事件用完后仍不足四个脚点."""


@dataclass(frozen=True)
class CompletedFootprint:
    """补齐后的四个脚点, 按 :data:`LEG_ORDER` 排列."""

    points: tuple[tuple[Any, Any, Any], ...]
    sources: tuple[str, ...]
    phases: tuple[int | None, ...]

    def source_of(self, leg: int | str) -> str:
        return self.sources[leg_index(leg)]


def leg_index(leg: int | str) -> int:
    """把腿名或下标规范化为 0..3 的工程下标."""
    if isinstance(leg, str):
        key = leg.upper()
        if key not in LEG_INDEX:
            raise ValueError(
                f"未知腿名 {leg!r}; 可选 {LEG_ORDER}"
            )
        return LEG_INDEX[key]
    index = int(leg)
    if not 0 <= index < len(LEG_ORDER):
        raise ValueError(
            f"腿下标必须在 [0, {len(LEG_ORDER)}) 内, 实际为 {index}"
        )
    return index


def complete_footprint(
    stance: Mapping[int | str, Any],
    future_events: Sequence[FootholdEvent] = (),
    *,
    leg_order: Sequence[str] = LEG_ORDER,
) -> CompletedFootprint:
    """按论文的方式用未来相位事件补齐四个脚点.

    先放入当前支撑脚, 再按 ``phase`` 从小到大遍历未来事件; 某条腿
    还没有脚点时, 采用它最早的那个事件. 不修改已在支撑集中的脚点.
    四个脚点凑齐后停止. 若遍历完仍不足四个, 抛出
    :class:`IncompleteFootprintError`, **不会**用 0 或未定义值填充.

    Args:
      stance: ``{腿名/下标: 位置}`` 当前支撑脚.
      future_events: 未来落脚事件, 按相位补齐.
      leg_order: 返回点的排列顺序, 默认工程顺序.

    Returns:
      :class:`CompletedFootprint`; ``sources[i]`` 为 ``"stance"`` 或
      ``"phase <k>"``, 明确说明补齐点来自哪个未来相位.
    """
    if len(leg_order) != len(LEG_ORDER):
        raise ValueError(
            f"leg_order 必须包含 {len(LEG_ORDER)} 条腿, "
            f"实际为 {len(leg_order)}"
        )

    points: dict[int, tuple[Any, Any, Any]] = {}
    sources: dict[int, str] = {}
    phases: dict[int, int | None] = {}

    for leg, position in stance.items():
        index = leg_index(leg)
        if index in points:
            raise ValueError(f"支撑集中腿 {leg!r} 重复出现")
        points[index] = point3(position, f"stance[{leg!r}]")
        sources[index] = "stance"
        phases[index] = None

    ordered_events = sorted(
        enumerate(future_events), key=lambda item: (item[1].phase, item[0])
    )
    for _, event in ordered_events:
        index = leg_index(event.leg)
        if index not in points:
            points[index] = point3(
                event.position, f"future phase {event.phase}[{event.leg!r}]"
            )
            sources[index] = f"phase {event.phase}"
            phases[index] = int(event.phase)
        if len(points) == len(LEG_ORDER):
            break

    if len(points) < len(LEG_ORDER):
        missing = [
            name
            for name in LEG_ORDER
            if LEG_INDEX[name] not in points
        ]
        raise IncompleteFootprintError(
            "未来相位事件不足以补齐四个脚点, 缺少 "
            f"{missing}. 不会用未定义落脚点填充式 (30); "
            "请提供更远的未来事件, 或等支撑相完整后再加该约束."
        )

    indices = [leg_index(name) for name in leg_order]
    return CompletedFootprint(
        points=tuple(points[index] for index in indices),
        sources=tuple(sources[index] for index in indices),
        phases=tuple(phases[index] for index in indices),
    )


def convexity_records_from_events(
    stance: Mapping[int | str, Any],
    future_events: Sequence[FootholdEvent] = (),
    *,
    order: Sequence[int] = CCW_INDICES,
    require_ccw: bool = True,
    phase: int | None = None,
) -> VAConstraintSet:
    """补齐未来脚点后构造式 (30).

    返回的约束记录中 ``foot_sources`` 会标注每个脚点来自当前支撑
    还是哪个未来相位; ``note`` 会先写补齐来源, 再写绕向检查结果.
    """
    completed = complete_footprint(stance, future_events)
    result = convexity_records(
        completed.points,
        order=order,
        require_ccw=require_ccw,
        phase=phase,
        foot_sources=completed.sources,
    )
    source_text = ", ".join(
        f"{LEG_ORDER[i]}<-{completed.sources[i]}"
        for i in range(len(LEG_ORDER))
    )
    return VAConstraintSet(
        records=result.records,
        phase=phase,
        note=f"sources: {source_text}; {result.note}",
    )


# ---------------------------------------------------------------------- #
# 式 (5): 摩擦锥 + 推地条件
# ---------------------------------------------------------------------- #
def friction_push_records(
    a_B: Sequence[Any],
    p_B_ddot: Sequence[Any],
    mu: Any,
    *,
    form: str = "squared",
    g_vec: Sequence[Any] = GRAVITY,
    include_push: bool = False,
    phase: int | None = None,
) -> VAConstraintSet:
    """式 (5)/(17a) 摩擦锥与 V-A-4 推地条件的记录.

    Args:
      a_B: 论文约定 ``a_B = g - p_B_ddot`` 的 3 维向量.
      p_B_ddot: 机身世界系加速度 (用于推地条件).
      mu: 摩擦系数, 数值时必须为正.
      form: ``"norm"`` 为原始范数形式;
        ``"squared"`` 为等价平方形式并额外加入 ``a_B_z >= 0``.
      g_vec: 重力向量, 默认 ``[0, 0, -9.81]``.
      include_push: 是否加入推地条件. 默认 **False**, 因为按论文字面
        符号它和摩擦锥互相矛盾 (静止状态即为反例, 见模块文档); 在
        符号约定经推导或权威来源确认之前, 两者应分开测试/使用.
        只有在 ``N > 0`` 且有明确依据时才设为 True.
      phase: 约束所属相位.

    Returns:
      ``category="5"`` 的摩擦约束 (norm 1 条 / squared 2 条),
      以及 ``category="V-A-4"`` 的推地不等式 (1 条).
    """
    a_B = point3(a_B, "a_B")
    p_ddot = point3(p_B_ddot, "p_B_ddot")
    g_vec = point3(g_vec, "g_vec")

    _validate_friction(mu, form)

    records: list[VAConstraint] = []
    if form == "norm":
        records.append(
            VAConstraint(
                name="5_friction_cone",
                category="5",
                kind=INEQUALITY,
                residual=friction_residual(a_B, mu),
                phase=phase,
                bound=0.0,
            )
        )
        cone_note = "norm form (gradient undefined at a_B_xy=0)"
    else:
        records.append(
            VAConstraint(
                name="5_friction_cone",
                category="5",
                kind=INEQUALITY,
                residual=friction_squared_residual(a_B, mu),
                phase=phase,
                bound=0.0,
            )
        )
        # 平方形式的前提: a_B_z >= 0. 没有这一条会允许 a_B_z < 0 的伪解.
        records.append(
            VAConstraint(
                name="5_friction_normal",
                category="5",
                kind=INEQUALITY,
                residual=-a_B[2],
                phase=phase,
                bound=0.0,
            )
        )
        cone_note = "squared form + a_B_z >= 0 (exact, smooth)"

    if include_push:
        # ez . p_ddot >= ez . g  <=>  g_z - p_ddot_z <= 0
        records.append(
            VAConstraint(
                name="V-A-4_push",
                category="V-A-4",
                kind=INEQUALITY,
                residual=g_vec[2] - p_ddot[2],
                phase=phase,
                bound=0.0,
            )
        )

    return VAConstraintSet(
        records=tuple(records), phase=phase, note=cone_note
    )


def add_friction_push_constraints(
    backend: Any,
    a_B: Sequence[Any],
    p_B_ddot: Sequence[Any],
    mu: Any,
    *,
    form: str = "squared",
    g_vec: Sequence[Any] = GRAVITY,
    include_push: bool = False,
    phase: int | None = None,
) -> VAConstraintSet:
    """把摩擦锥和推地条件加入 ``backend`` (全部 ``<= 0``).

    ``include_push`` 默认 False; 字面符号下摩擦锥与推地条件不能同时
    成立, 确认符号前不要一起加入完整优化.
    """
    result = friction_push_records(
        a_B,
        p_B_ddot,
        mu,
        form=form,
        g_vec=g_vec,
        include_push=include_push,
        phase=phase,
    )
    for record in result.records:
        backend.add_inequality(
            record.residual, -float("inf"), 0.0, name=record.name
        )
    return result


def spline_base_acceleration(
    coeffs: Any, tau: float, g_vec: Sequence[Any] = GRAVITY
) -> tuple[tuple[Any, Any, Any], tuple[Any, Any, Any]]:
    """由 6D 位姿样条求 ``p_B_ddot`` 和 ``a_B = g - p_B_ddot``.

    ``spline_value(..., order=2)`` 只对位置分量求二阶导; 欧拉角分量
    的二阶导在这里没有用到, 也不应被误当成线加速度.

    Returns:
      ``(p_B_ddot, a_B)``, 均为 3 元组, 支持 CasADi MX.
    """
    g_vec = point3(g_vec, "g_vec")
    pose_ddot = spline_value(coeffs, tau, order=2)
    p_ddot = (pose_ddot[0], pose_ddot[1], pose_ddot[2])
    a_B = (
        g_vec[0] - p_ddot[0],
        g_vec[1] - p_ddot[1],
        g_vec[2] - p_ddot[2],
    )
    return p_ddot, a_B


def add_spline_phase_friction_constraints(
    backend: Any,
    coeffs: Any,
    duration: float,
    mu: Any,
    *,
    g_vec: Sequence[Any] = GRAVITY,
    form: str = "squared",
    include_push: bool = False,
    phase: int | None = None,
    sample_times: Sequence[float] | None = None,
    num_samples: int = 6,
) -> VAConstraintSet:
    """对**一个相位**的样条采样并加入摩擦锥 / 推地约束.

    ``include_push`` 默认 **False**: 按论文的字面符号
    (``a_B = g - p_B_ddot``), 摩擦锥要求 ``p_B_ddot_z <= g_z``,
    推地条件要求 ``p_B_ddot_z >= g_z``; 两者同时加入时可行集只剩
    ``a_B = 0``. 在符号约定确认前应只使用其中一组. 采样时刻默认与
    参考实现一致 (``linspace(0, T, n+1)[:-1]``). 每条约束名称追加
    ``_phase<k>_tau<t>``, 便于按采样点定位.
    """
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError(f"duration 必须为有限正数, 实际为 {duration}")
    times = _resolve_sample_times(duration, sample_times, num_samples)

    records: list[VAConstraint] = []
    for tau in times:
        p_ddot, a_B = spline_base_acceleration(coeffs, tau, g_vec)
        sampled = friction_push_records(
            a_B,
            p_ddot,
            mu,
            form=form,
            g_vec=g_vec,
            include_push=include_push,
            phase=phase,
        )
        suffix = _constraint_suffix(None, phase, float(tau))
        for record in sampled.records:
            record = VAConstraint(
                name=f"{record.name}{suffix}",
                category=record.category,
                kind=record.kind,
                residual=record.residual,
                foot_indices=record.foot_indices,
                foot_sources=record.foot_sources,
                phase=record.phase,
                bound=record.bound,
                tau=float(tau),
            )
            backend.add_inequality(
                record.residual, -float("inf"), 0.0, name=record.name
            )
            records.append(record)

    return VAConstraintSet(
        records=tuple(records),
        phase=phase,
        note=f"sampled at {len(times)} times",
    )


def _validate_friction(mu: Any, form: str) -> None:
    if form not in ("norm", "squared"):
        raise ValueError(
            f"未知的摩擦约束形式 {form!r}; 可选 'norm' 或 'squared'"
        )
    if isinstance(mu, (int, float, np.integer, np.floating)):
        if float(mu) <= 0.0:
            raise ValueError(f"摩擦系数 mu 必须为正, 实际为 {mu}")


# ---------------------------------------------------------------------- #
# 式 (31): 腿部可达范围
# ---------------------------------------------------------------------- #
def rotation_matrix_zyx(
    roll: Any, pitch: Any, yaw: Any
) -> tuple[tuple[Any, Any, Any], ...]:
    """``R_wb = Rz(yaw) @ Ry(pitch) @ Rx(roll)``.

    与 :class:`~myrobot_mjlab.task.DTC.tamols.base_spline.Spline6D`
    的欧拉角约定一致: 输入顺序为 ``[roll, pitch, yaw]``, 旋转顺序
    ZYX, ``R_wb`` 把机体系向量变换到世界系.

    Note:
      参考实现 ``get_R_B(phi_B)`` 期望的输入顺序是
      ``[yaw, pitch, roll]``, 与本工程相反. 移植参考代码时必须重排.
    """
    sr, cr = _sin(roll), _cos(roll)
    sp, cp = _sin(pitch), _cos(pitch)
    sy, cy = _sin(yaw), _cos(yaw)

    return (
        (
            cy * cp,
            cy * sp * sr - sy * cr,
            cy * sp * cr + sy * sr,
        ),
        (
            sy * cp,
            sy * sp * sr + cy * cr,
            sy * sp * cr - cy * sr,
        ),
        (
            -sp,
            cp * sr,
            cp * cr,
        ),
    )


def _matrix3(matrix: Any, name: str = "R_B") -> tuple[tuple[Any, ...], ...]:
    shape = getattr(matrix, "shape", None)
    if shape is not None and len(shape) == 2:
        if tuple(int(s) for s in shape) != (3, 3):
            raise ValueError(f"{name} 必须为 3x3, 实际 shape={shape}")
        return tuple(point3(matrix[i, :]) for i in range(3))
    if len(matrix) != 3:
        raise ValueError(f"{name} 必须为 3x3")
    return tuple(point3(matrix[i]) for i in range(3))


def matvec3(
    matrix: Any, vector: Sequence[Any]
) -> tuple[Any, Any, Any]:
    """3x3 矩阵乘 3 维向量 (支持 numpy / 嵌套序列 / CasADi MX)."""
    rows = _matrix3(matrix)
    vector = point3(vector, "vector")
    return tuple(
        rows[i][0] * vector[0]
        + rows[i][1] * vector[1]
        + rows[i][2] * vector[2]
        for i in range(3)
    )


def spline_value(
    coeffs: Any, tau: float, order: int = 0
) -> tuple[Any, ...]:
    """按工程的系数布局 ``coeffs[dim, k] = a_k[dim]`` 求样条.

    ``pose(tau) = sum_k a_k * tau^k``, ``tau`` 为相位内物理时间 [s].
    返回 6 个分量 ``(x, y, z, roll, pitch, yaw)``.

    Args:
      coeffs: ``(6, K)`` 的 numpy 数组或 CasADi 决策变量.
      tau: 标量时间.
      order: 求导阶数; ``order >= K`` 时返回全 0.
    """
    shape = getattr(coeffs, "shape", None)
    if shape is None or len(shape) != 2:
        raise ValueError(
            "coeffs 必须是二维的 [6, K], "
            f"实际 shape={None if shape is None else tuple(shape)}"
        )
    dims, num_coeffs = int(shape[0]), int(shape[1])
    if dims != 6:
        raise ValueError(f"coeffs 第一维必须为 6, 实际为 {dims}")
    if not isinstance(order, int) or order < 0:
        raise ValueError(f"order 必须是非负整数, 实际为 {order!r}")

    values: list[Any] = []
    for dim in range(6):
        if order >= num_coeffs:
            values.append(0.0)
            continue
        terms = [
            (
                math.factorial(k) / math.factorial(k - order)
            )
            * _coefficient(coeffs, dim, k)
            for k in range(order, num_coeffs)
        ]
        value = terms[-1]
        for term in reversed(terms[:-1]):
            value = value * tau + term
        values.append(value)
    return tuple(values)


def _coefficient(coeffs: Any, dim: int, k: int) -> Any:
    shape = getattr(coeffs, "shape", None)
    if shape is not None and len(shape) == 2:
        return coeffs[dim, k]
    return coeffs[dim][k]


def reachability_residuals(
    p_B: Sequence[Any],
    R_B: Any,
    p_i: Sequence[Any],
    r_i: Sequence[Any],
    l_min: float,
    l_max: float,
) -> tuple[Any, Any]:
    """式 (31) 的两条残差, 都要求 ``<= 0``.

    ``hip = p_B + R_B r_i``, ``d2 = ||hip - p_i||^2``::

        l_min^2 - d2 <= 0
        d2 - l_max^2 <= 0
    """
    _validate_reach_limits(l_min, l_max)
    r_i = point3(r_i, "r_i")
    p_i = point3(p_i, "p_i")
    p_B = point3(p_B, "p_B")

    rotated = matvec3(R_B, r_i)
    distance_sq = (
        (p_B[0] + rotated[0] - p_i[0]) ** 2
        + (p_B[1] + rotated[1] - p_i[1]) ** 2
        + (p_B[2] + rotated[2] - p_i[2]) ** 2
    )
    return (l_min**2 - distance_sq, distance_sq - l_max**2)


def reachability_records(
    p_B: Sequence[Any],
    R_B: Any,
    p_i: Sequence[Any],
    r_i: Sequence[Any],
    l_min: float = DEFAULT_L_MIN,
    l_max: float = DEFAULT_L_MAX,
    *,
    leg_index: int | None = None,
    phase: int | None = None,
    tau: float | None = None,
) -> tuple[VAConstraint, VAConstraint]:
    """构造一条腿在某一时刻的两条式 (31) 约束记录.

    下限和上限分别命名 ``31_lower`` / ``31_upper``; ``bound`` 分别
    记录 ``l_min^2`` 和 ``l_max^2``.
    """
    lower, upper = reachability_residuals(
        p_B, R_B, p_i, r_i, l_min, l_max
    )
    suffix = _constraint_suffix(leg_index, phase, tau)
    foot_indices = () if leg_index is None else (leg_index,)
    record_kwargs = {
        "foot_indices": foot_indices,
        "phase": phase,
        "tau": tau,
    }
    return (
        VAConstraint(
            name=f"31_lower{suffix}",
            category="31_lower",
            kind=INEQUALITY,
            residual=lower,
            bound=l_min**2,
            **record_kwargs,
        ),
        VAConstraint(
            name=f"31_upper{suffix}",
            category="31_upper",
            kind=INEQUALITY,
            residual=upper,
            bound=l_max**2,
            **record_kwargs,
        ),
    )


def add_reachability_constraints(
    backend: Any,
    p_B: Sequence[Any],
    R_B: Any,
    footholds: Any,
    r_hips: Any,
    legs: Sequence[int | str],
    l_min: float = DEFAULT_L_MIN,
    l_max: float = DEFAULT_L_MAX,
    *,
    phase: int | None = None,
    tau: float | None = None,
) -> VAConstraintSet:
    """给若干着地腿在**同一时刻**加入式 (31) 约束.

    Args:
      p_B: 机身位置 (世界系).
      R_B: ``R_wb``, 3x3.
      footholds: ``(L, 3)`` 脚点数组/变量, 按 :data:`LEG_ORDER` 索引.
      r_hips: ``(L, 3)`` 髋位置 (机体系), 按 :data:`LEG_ORDER` 索引.
      legs: 需要约束的着地腿 (名字或下标); 摆动腿不要放进来.
      l_min, l_max: 可达球半径 [m].
      phase, tau: 写入记录, 便于定位.

    Returns:
      ``2 * len(legs)`` 条约束的记录.
    """
    records: list[VAConstraint] = []
    for leg in legs:
        index = leg_index(leg)
        records.extend(
            reachability_records(
                p_B,
                R_B,
                _point_at(footholds, index),
                _point_at(r_hips, index),
                l_min,
                l_max,
                leg_index=index,
                phase=phase,
                tau=tau,
            )
        )
    for record in records:
        backend.add_inequality(
            record.residual, -float("inf"), 0.0, name=record.name
        )
    return VAConstraintSet(records=tuple(records), phase=phase)


def add_spline_phase_reachability_constraints(
    backend: Any,
    coeffs: Any,
    duration: float,
    footholds: Any,
    r_hips: Any,
    stance_legs: Sequence[int | str],
    l_min: float = DEFAULT_L_MIN,
    l_max: float = DEFAULT_L_MAX,
    *,
    phase: int | None = None,
    sample_times: Sequence[float] | None = None,
    num_samples: int = 6,
) -> VAConstraintSet:
    """对**一个相位**的样条按若干时刻采样, 加入式 (31).

    默认采样方式与参考实现一致: ``tau = linspace(0, T, n+1)[:-1]``,
    即包含相位起点、不包含终点, 避免相邻相位的连接点被重复约束.
    采样时刻是相位内物理时间 [s].

    工程没有接触状态数据, 因此 ``stance_legs`` 必须显式给出; 摆动腿
    不应放入 (摆动腿本来就不要求位于可达球内).

    Args:
      coeffs: ``(6, K)`` 样条系数 (数值或 CasADi 变量).
      duration: 相位时长 T [s], 必须为正.
      footholds: ``(L, 3)`` 脚点.
      r_hips: ``(L, 3)`` 髋位置 (机体系).
      stance_legs: 着地腿.
      sample_times: 自定义采样时刻; None 时用默认等距采样.
      num_samples: 默认采样个数 (不含终点).
      phase: 相位编号, 写入记录名称.

    Returns:
      全部记录; 数量为
      ``len(stance_legs) * len(sample_times) * 2`` (受符号无效项影响时
      仍按此规则).
    """
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError(f"duration 必须为有限正数, 实际为 {duration}")
    times = _resolve_sample_times(duration, sample_times, num_samples)

    records: list[VAConstraint] = []
    for tau in times:
        pose = spline_value(coeffs, tau, order=0)
        p_B = pose[0:3]
        R_B = rotation_matrix_zyx(pose[3], pose[4], pose[5])
        for leg in stance_legs:
            index = leg_index(leg)
            records.extend(
                reachability_records(
                    p_B,
                    R_B,
                    _point_at(footholds, index),
                    _point_at(r_hips, index),
                    l_min,
                    l_max,
                    leg_index=index,
                    phase=phase,
                    tau=float(tau),
                )
            )

    for record in records:
        backend.add_inequality(
            record.residual, -float("inf"), 0.0, name=record.name
        )
    return VAConstraintSet(records=tuple(records), phase=phase)


def _resolve_sample_times(
    duration: float,
    sample_times: Sequence[float] | None,
    num_samples: int,
) -> tuple[float, ...]:
    if sample_times is None:
        if not isinstance(num_samples, int) or num_samples < 1:
            raise ValueError(
                f"num_samples 必须为正整数, 实际为 {num_samples!r}"
            )
        times = np.linspace(0.0, duration, num_samples + 1)[:-1]
    else:
        times = np.asarray(sample_times, dtype=float).reshape(-1)
        if times.size == 0:
            raise ValueError("sample_times 不能为空")
    for tau in times:
        value = float(tau)
        if not np.isfinite(value) or value < 0.0 or value > duration:
            raise ValueError(
                f"采样时刻必须在 [0, {duration}] 内, 实际为 {value}"
            )
    return tuple(float(tau) for tau in times)


def _validate_reach_limits(l_min: float, l_max: float) -> None:
    for name, value in (("l_min", l_min), ("l_max", l_max)):
        if not isinstance(value, (int, float, np.integer, np.floating)):
            raise TypeError(f"{name} 必须为数值, 实际为 {type(value).__name__}")
        if not np.isfinite(value) or float(value) <= 0.0:
            raise ValueError(f"{name} 必须为有限正数, 实际为 {value}")
    if float(l_min) > float(l_max):
        raise ValueError(
            f"要求 l_min <= l_max, 实际 l_min={l_min}, l_max={l_max}"
        )


def _constraint_suffix(
    leg_index: int | None, phase: int | None, tau: float | None
) -> str:
    suffix = ""
    if leg_index is not None:
        suffix += f"_leg{int(leg_index)}"
    if phase is not None:
        suffix += f"_phase{int(phase)}"
    if tau is not None:
        suffix += f"_tau{float(tau):.6g}"
    return suffix
