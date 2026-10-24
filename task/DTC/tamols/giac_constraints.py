"""TAMOLS 动态稳定性约束 (GIAC + 摩擦锥), 按接触状态分派.

本模块只实现论文中的稳定性约束, 不包含完整轨迹优化器. 记
``p_ij = p_j - p_i``, ``det(a, b, c) = a . (b x c)``,
``a_B = g - p_B_ddot``, ``L_dot`` 为机身角动量变化率, ``m`` 为质量,
``mu`` 为摩擦系数, ``N`` 为支撑脚数量. 论文正文 (T-RO 版本) 把这些
式子编号为 (2a)-(2f); 参考实现注释称 (17a)-(17d), 本模块按用户约定
统一写作 (17a)-(17f):

(17a) 摩擦锥, N > 0 (ez = [0, 0, 1]^T)::

    mu * (ez . a_B) >= ||(I - ez ez^T) a_B||

(17b) 多足 GIAC, N >= 3, 对每一对支撑脚::

    m * det(p_ij, p_B - p_i, a_B) <= p_ij . L_dot

(17c) 双足 GIAC 等式, N = 2::

    m * det(p_12, p_B - p_1, a_B) = p_12 . L_dot

(17d) 双足力矩不等式, N = 2::

    det(ez, p_12, M_1) >= 0
    M_1 = (p_B - p_1) x a_B - L_dot / m

(17e) 单足 GIAC 向量等式, N = 1::

    m * (p_B - p_i) x a_B - L_dot = 0

(17f) 腾空 (弹道) 条件, N = 0::

    a_B = 0,  L_dot = 0

这些式子是**互斥**的接触状态, 不能把 (17b) 的成对不等式复用于
N = 2/1/0. 本模块由 :func:`contact_constraint_records` 和
:func:`add_contact_constraints` 按 ``len(feet)`` 严格分派. 向量等式
(17e) 和 (17f) 会拆成逐个分量的标量等式.

摩擦约束 (17a) 在有接触 (N > 0) 时都必须加入; N = 0 没有接触点,
因此不加入摩擦约束. 本模块实现的是不带松弛的精确约束, 完整 TAMOLS
中的逐相位松弛变量 epsilon 尚未引入 (后续扩展项).

索引与符号说明
--------------
- ``det(a, b, c) = a . (b x c)``, 与参考实现
  ``tamols-rl/go2-hrl/fetch/tamols/helpers.py`` 中的
  ``determinant(a, b, c) = a.dot(np.cross(b, c))`` 一致.
- (17b)/(17c) 的左端对交换 i, j 变号::

      LHS(i, j) = -LHS(j, i)

  因此 (17b) 只对 **无序对** 添加一次 (默认 ``i < j``); 若把两个
  方向都加进去, 两条不等式合起来迫使 LHS = 0, 那是 N = 2 的 (17c),
  不能拿来当 N >= 3 的约束.
- 符号还依赖支撑脚的排列方向: 同样的无序对 (i, j) 换一个书写方向,
  不等式方向就反了. 所以支撑脚索引应与支撑多边形的环绕方向一致,
  或者显式传入按环绕方向排列的 ``pairs``. 最小示例中支撑脚顺序是固定的.
  这一点在几何上很重要: 论文把 (17b) 解释为支撑锥的边界约束, 若脚点
  索引不是环绕顺序, 直接取所有 ``i < j`` 可能把某条边的方向写反,
  得到的是另一个半空间, 甚至会出现 "ZMP 在支撑多边形内但约束不满足"
  的情况. 参考实现的 ``get_contact_pairs`` 也返回所有 ``i < j``;
  本模块默认与它一致, 但保留 ``pairs`` 参数以便按环绕顺序传入边界对.
- N = 2 时 ``p_12 = feet[1] - feet[0]``. (17c) 的左端对交换两只脚
  严格变号, 等式本身不受影响. (17d) 按论文方向
  ``det(ez, p_12, M_1) >= 0`` 实现, 代码返回 ``-det(...)`` 以统一成
  ``<= 0``. 参考实现的力矩写成
  ``M_i = (p_B - p_i) x (g - a_ref) - L_dot / m``, 其中它的 ``a_ref``
  实际是 ``p_B_ddot``; 代入论文的 ``a_B = g - p_B_ddot`` 后与
  ``M_i = (p_B - p_i) x a_B - L_dot / m`` 逐项相等
  (见测试 ``test_17d_engineering_interpretation_matches_paper``).
  (17d) **不是**简单变号: 它的
  ``M_1`` 以 ``p_1`` 为参考点, 交换后变成以 ``p_2`` 为参考的 ``M_2``,
  数值一般会改变. 所以双脚的先后顺序**有物理含义**, 代码不做任何
  隐式排序, 调用方应按支撑多边形的环绕方向给出 ``feet``.
- N = 1 的 (17e) 和 N = 0 的 (17f) 是向量等式, 本模块拆成 x/y/z
  三个分量等式 (N = 0 共 6 条).
- 论文定义 ``a_B = g - p_B_ddot``. 注意参考代码里的同名变量实际取自
  ``evaluate_spline_acceleration(...)``, 即 ``p_B_ddot``, 与论文的
  约定不同; 本模块按论文/用户给出的 ``g - p_B_ddot`` 实现, 移植参考
  代码时不能直接照搬符号. 在该约定下直立静止时
  ``a_B = [0, 0, -9.81]``, 摩擦约束 (17a) 的左侧 ``mu * a_B_z`` 为负,
  而右侧范数非负, 因而按字面实现该摩擦写法实际要求
  ``a_B_z >= 0``, 即 ``p_B_ddot_z <= -g``. 这是公式的符号约定,
  代码不做隐式翻转; 若你的坐标系/接触力约定与此相反, 需要相应取负.
  论文式 (17a) 与本模块的 ``||a_B_xy|| - mu * a_B_z <= 0`` 在代数上
  完全一致.

模型假设 (只验证约束代码, 不是完整 TAMOLS)
------------------------------------------
论文 Assumption 1:

1. 腿的质量为零 (SRBD 模型);
2. 角动量变化率对接触力的影响可忽略 (但仍作为 ``L_dot`` 出现在约束中);
3. 接触力只能推地面, 不能拉;
4. 摩擦系数为常数;
5. 接触建立在水平面上;
6. 机身位于所有着地脚的上方.

本实现额外假设:

- 刚体 + 点接触模型; 接触力作用在给定的支撑脚位置.
- 每个相位的接触集合由调用方给定且固定不变; 不处理接触切换瞬间.
- 支撑脚位置和机身位置在约束求值时固定; 完整优化中它们由样条/落脚点
  变量给出.
- 摩擦模型为水平地面上的各向同性库仑锥; 不考虑接触力矩、关节力矩、
  腿长/运动学限制、地形高度和脚底打滑.
- 没有引入 TAMOLS 的逐相位松弛变量 epsilon; 本模块的式子是精确约束
  (后续扩展项).
- 该模块的输出是符号表达式, 可以配合 CasADi MX 使用; 数值函数
  (numpy) 也可直接调用, 便于单元测试. 每类约束的残差通过
  :class:`ContactConstraintSet` 记录 (不等式要求残差 ``<= 0``,
  等式要求残差 ``== 0``).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Iterable, Sequence

import numpy as np

GRAVITY = (0.0, 0.0, -9.81)
EZ = (0.0, 0.0, 1.0)

EQUALITY = "equality"
INEQUALITY = "inequality"
"""约束类型: 等式要求残差 == 0, 不等式要求残差 <= 0."""


# ---------------------------------------------------------------------- #
# 基础三元运算 (同时支持 numpy 和 CasADi MX 的标量表达式)
# ---------------------------------------------------------------------- #
def det3(a: Sequence[Any], b: Sequence[Any], c: Sequence[Any]) -> Any:
    """三向量行列式 ``det(a, b, c) = a . (b x c)``.

    用标量分量展开实现, 因此 numpy 数组、Python 序列和 CasADi MX
    向量都可以作为输入.
    """
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        + a[1] * (b[2] * c[0] - b[0] * c[2])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def _dot3(a: Sequence[Any], b: Sequence[Any]) -> Any:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _sub3(a: Sequence[Any], b: Sequence[Any]) -> tuple[Any, Any, Any]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross3(a: Sequence[Any], b: Sequence[Any]) -> tuple[Any, Any, Any]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _numel(value: Any) -> int:
    """返回序列/数组/符号向量的元素个数.

    CasADi 的 MX/DM 没有 ``len()``, 但提供 ``shape``; numpy 和 Python
    序列则用 ``len()`` 或 ``shape`` 都可以.
    """
    shape = getattr(value, "shape", None)
    if shape is None:
        return len(value)
    return int(np.prod(shape))


# ---------------------------------------------------------------------- #
# 重力-惯性加速度
# ---------------------------------------------------------------------- #
def gravity_inertial_acceleration(
    p_B_ddot: Sequence[Any],
    g_vec: Sequence[Any] = GRAVITY,
) -> tuple[Any, Any, Any]:
    """``a_B = g_vec - p_B_ddot``.

    Args:
      p_B_ddot: 机身加速度 (世界系), 3 个分量.
      g_vec: 重力向量, 默认 ``[0, 0, -9.81]``.

    Returns:
      3 元组, 每个分量可以是 float / numpy 标量 / CasADi MX 标量.
    """
    if _numel(p_B_ddot) != 3 or _numel(g_vec) != 3:
        raise ValueError("p_B_ddot 和 g_vec 必须各有 3 个分量")
    return (
        g_vec[0] - p_B_ddot[0],
        g_vec[1] - p_B_ddot[1],
        g_vec[2] - p_B_ddot[2],
    )


# ---------------------------------------------------------------------- #
# GIAC (N >= 3)
# ---------------------------------------------------------------------- #
def giac_residual(
    mass: Any,
    p_B: Sequence[Any],
    a_B: Sequence[Any],
    L_dot: Sequence[Any],
    p_i: Sequence[Any],
    p_j: Sequence[Any],
) -> Any:
    """单对支撑脚的 GIAC 左端 (TAMOLS Eq. 17b).

    返回::

        mass * det(p_j - p_i, p_B - p_i, a_B) - (p_j - p_i) . L_dot

    约束要求该值 ``<= 0``. 注意交换 i, j 会使其变号.
    """
    p_ij = _sub3(p_j, p_i)
    return mass * det3(p_ij, _sub3(p_B, p_i), a_B) - _dot3(p_ij, L_dot)


def contact_pairs(
    num_feet: int,
    pairs: Iterable[Sequence[int]] | None = None,
) -> list[tuple[int, int]]:
    """返回要施加 GIAC 的脚点索引对.

    默认返回所有无序对 ``i < j``. 如果显式传入 ``pairs``, 则保留给定顺序
    (顺序决定不等式的符号, 调用方需自行保证方向一致).
    """
    if pairs is None:
        return list(combinations(range(num_feet), 2))

    result: list[tuple[int, int]] = []
    for pair in pairs:
        if len(pair) != 2:
            raise ValueError(f"pairs 中每个元素必须是 (i, j), 实际为 {pair}")
        i, j = int(pair[0]), int(pair[1])
        if i == j:
            raise ValueError(f"pairs 中不允许 i == j, 实际为 ({i}, {j})")
        if not (0 <= i < num_feet and 0 <= j < num_feet):
            raise ValueError(
                f"pairs 索引越界: ({i}, {j}), num_feet={num_feet}"
            )
        result.append((i, j))
    return result


def giac_residuals(
    mass: Any,
    p_B: Sequence[Any],
    a_B: Sequence[Any],
    L_dot: Sequence[Any],
    feet: Sequence[Sequence[Any]],
    pairs: Iterable[Sequence[int]] | None = None,
) -> list[tuple[tuple[int, int], Any]]:
    """计算所有 GIAC 残差, 返回 ``[((i, j), residual), ...]``."""
    result = []
    for i, j in contact_pairs(len(feet), pairs):
        result.append(
            (
                (i, j),
                giac_residual(mass, p_B, a_B, L_dot, feet[i], feet[j]),
            )
        )
    return result


# ---------------------------------------------------------------------- #
# 其余接触状态: N = 2, N = 1, N = 0
# ---------------------------------------------------------------------- #
def contact_moment(
    mass: Any,
    p_B: Sequence[Any],
    a_B: Sequence[Any],
    L_dot: Sequence[Any],
    p_i: Sequence[Any],
) -> tuple[Any, Any, Any]:
    """接触点 i 关于机身的力矩项 ``M_i`` (论文式 17d 中的 ``M_1``)::

        M_i = (p_B - p_i) x a_B - L_dot / mass

    注意这里用的 ``a_B = g - p_B_ddot``, 与论文一致.
    """
    moment = _cross3(_sub3(p_B, p_i), a_B)
    return (
        moment[0] - L_dot[0] / mass,
        moment[1] - L_dot[1] / mass,
        moment[2] - L_dot[2] / mass,
    )


def double_support_line_residual(
    mass: Any,
    p_B: Sequence[Any],
    a_B: Sequence[Any],
    L_dot: Sequence[Any],
    p_1: Sequence[Any],
    p_2: Sequence[Any],
) -> Any:
    """N = 2 的 GIAC 等式左端 (论文式 17c)::

        m * det(p_12, p_B - p_1, a_B) - p_12 . L_dot = 0

    其中 ``p_12 = p_2 - p_1``. 它与 :func:`giac_residual` 的表达式相同,
    但语义是等式, 不能再当不等式用. 交换两只脚会改变左端符号, 不影响
    等式本身.
    """
    return giac_residual(mass, p_B, a_B, L_dot, p_1, p_2)


def double_support_moment_residual(
    mass: Any,
    p_B: Sequence[Any],
    a_B: Sequence[Any],
    L_dot: Sequence[Any],
    p_1: Sequence[Any],
    p_2: Sequence[Any],
) -> Any:
    """N = 2 的力矩不等式残差 (论文式 17d)::

        det(e_z, p_12, M_1) >= 0,   M_1 = (p_B - p_1) x a_B - L_dot / m

    返回 ``-det(...)``, 因此要求残差 ``<= 0``. 式中的 ``p_12`` 方向
    和参考点 ``p_1`` 都由脚的先后顺序决定; 交换 p_1/p_2 时 ``p_12``
    变号且 ``M_1`` 变成以 ``p_2`` 为参考的 ``M_2``, 所以该式的数值
    一般会改变 (不是简单变号). 调用方应按支撑多边形环绕方向给脚.
    """
    p_12 = _sub3(p_2, p_1)
    M_1 = contact_moment(mass, p_B, a_B, L_dot, p_1)
    return -det3(EZ, p_12, M_1)


def single_support_residuals(
    mass: Any,
    p_B: Sequence[Any],
    a_B: Sequence[Any],
    L_dot: Sequence[Any],
    p_i: Sequence[Any],
) -> tuple[Any, Any, Any]:
    """N = 1 的 GIAC 向量等式 (论文式 17e), 拆成 3 个分量::

        m * (p_B - p_i) x a_B - L_dot = 0

    Returns:
      ``(residual_x, residual_y, residual_z)``, 每个都要求 ``== 0``.
    """
    moment = _cross3(_sub3(p_B, p_i), a_B)
    return (
        mass * moment[0] - L_dot[0],
        mass * moment[1] - L_dot[1],
        mass * moment[2] - L_dot[2],
    )


def ballistic_residuals(
    a_B: Sequence[Any],
    L_dot: Sequence[Any],
) -> tuple[Any, Any, Any, Any, Any, Any]:
    """N = 0 的腾空/弹道条件 (论文式 17f), 拆成 6 个分量::

        a_B = 0,   L_dot = 0

    Returns:
      ``(a_B_x, a_B_y, a_B_z, L_dot_x, L_dot_y, L_dot_z)``, 全部要求
      ``== 0``. ``a_B = 0`` 即 ``p_B_ddot = g``, 自由落体.
    """
    return (
        a_B[0],
        a_B[1],
        a_B[2],
        L_dot[0],
        L_dot[1],
        L_dot[2],
    )


# ---------------------------------------------------------------------- #
# 约束记录: 便于按类别查看/测试每类残差
# ---------------------------------------------------------------------- #
@dataclass(frozen=True)
class ContactConstraint:
    """一条标量约束及其残差.

    Attributes:
      name: 唯一名称, 例如 ``"17b_pair_0_1"``.
      category: 论文式编号, ``"17a"``..``"17f"``.
      kind: ``EQUALITY`` (要求 ``residual == 0``) 或
        ``INEQUALITY`` (要求 ``residual <= 0``).
      residual: 符号表达式或数值.
    """

    name: str
    category: str
    kind: str
    residual: Any

    def __post_init__(self) -> None:
        if self.kind not in (EQUALITY, INEQUALITY):
            raise ValueError(
                f"未知的约束类型 {self.kind!r}; "
                f"可选 {EQUALITY!r} 或 {INEQUALITY!r}"
            )

    def is_satisfied(self, atol: float = 1e-9) -> bool:
        """仅在残差为数值时可用; 符号表达式请先代入数值."""
        value = _scalar_value(self.residual)
        if self.kind == EQUALITY:
            return abs(value) <= atol
        return value <= atol


@dataclass(frozen=True)
class ContactConstraintSet:
    """某一接触状态下所有约束残差的记录.

    不等式要求残差 ``<= 0``; 等式要求残差 ``== 0``.
    """

    num_contacts: int
    records: tuple[ContactConstraint, ...]

    def __iter__(self):
        return iter(self.records)

    def __len__(self) -> int:
        return len(self.records)

    @property
    def equalities(self) -> tuple[ContactConstraint, ...]:
        return tuple(
            record for record in self.records if record.kind == EQUALITY
        )

    @property
    def inequalities(self) -> tuple[ContactConstraint, ...]:
        return tuple(
            record for record in self.records if record.kind == INEQUALITY
        )

    def by_category(self) -> dict[str, tuple[ContactConstraint, ...]]:
        """按论文式编号分组, 例如 ``{"17a": (...), "17c": (...)}``."""
        grouped: dict[str, list[ContactConstraint]] = {}
        for record in self.records:
            grouped.setdefault(record.category, []).append(record)
        return {key: tuple(value) for key, value in grouped.items()}

    def residual_values(self) -> dict[str, float]:
        """把每条约束的数值残差记录成 ``{name: residual}``.

        只适用于数值输入; 若残差是 CasADi 符号表达式, 需要先代入解.
        """
        return {
            record.name: _scalar_value(record.residual)
            for record in self.records
        }


def _scalar_value(value: Any) -> float:
    if hasattr(value, "full"):  # CasADi DM
        value = value.full()
    return float(np.asarray(value, dtype=float).reshape(-1)[0])


def friction_constraint_records(
    a_B: Sequence[Any],
    mu: Any,
    form: str = "squared",
) -> list[ContactConstraint]:
    """构造摩擦锥 (17a) 的约束记录.

    ``form="norm"`` 给出论文的原始形式; ``form="squared"`` 给出在
    ``a_B_z >= 0`` 下等价的平方形式, 并额外记录 ``a_B_z >= 0``
    (残差写成 ``-a_B_z``). 两种形式的可行集相同.
    """
    _validate_friction(mu, form)
    if form == "norm":
        return [
            ContactConstraint(
                name="17a_cone",
                category="17a",
                kind=INEQUALITY,
                residual=friction_residual(a_B, mu),
            )
        ]
    return [
        ContactConstraint(
            name="17a_cone",
            category="17a",
            kind=INEQUALITY,
            residual=friction_squared_residual(a_B, mu),
        ),
        ContactConstraint(
            name="17a_normal",
            category="17a",
            kind=INEQUALITY,
            residual=-a_B[2],
        ),
    ]


def contact_constraint_records(
    mass: Any,
    p_B: Sequence[Any],
    a_B: Sequence[Any],
    L_dot: Sequence[Any],
    feet: Sequence[Sequence[Any]],
    *,
    mu: Any = None,
    friction_form: str = "squared",
    pairs: Iterable[Sequence[int]] | None = None,
) -> ContactConstraintSet:
    """按支撑脚数量 ``N = len(feet)`` 构造对应接触状态的约束记录.

    - ``N >= 3``: (17b), 每对脚一条不等式; ``pairs`` 可选.
    - ``N == 2``: (17c) 一条等式 + (17d) 一条不等式.
    - ``N == 1``: (17e), 拆成 3 条分量等式.
    - ``N == 0``: (17f), 拆成 6 条分量等式 (``a_B`` 与 ``L_dot``).

    Args:
      mu: 摩擦系数. ``N > 0`` 时必须给出 (否则摩擦约束 17a 无法加入);
        ``N == 0`` 时没有接触, 忽略该参数.
      friction_form: ``"norm"`` 或 ``"squared"``, 见
        :func:`friction_constraint_records`.
      pairs: 仅用于 ``N >= 3``, 指定脚点索引对及其方向; 其他接触状态
        的脚点顺序由 ``feet`` 自身决定.

    Raises:
      ValueError: ``mu`` 缺失/非法, ``friction_form`` 非法, 或对
        ``N < 3`` 传入了 ``pairs``.
    """
    num_contacts = len(feet)

    for label, value in (
        ("p_B", p_B),
        ("a_B", a_B),
        ("L_dot", L_dot),
    ):
        if _numel(value) != 3:
            raise ValueError(f"{label} 必须有 3 个分量")
    for index, foot in enumerate(feet):
        if _numel(foot) != 3:
            raise ValueError(f"第 {index} 个支撑脚必须给出 3 维位置")

    if pairs is not None and num_contacts < 3:
        raise ValueError(
            "pairs 只用于 N >= 3 的逐边枚举; N = 2/1/0 的脚点顺序"
            "由 feet 列表本身决定."
        )
    if friction_form not in ("norm", "squared"):
        raise ValueError(
            f"未知的摩擦约束形式 {friction_form!r}; "
            "可选 'norm' 或 'squared'"
        )

    records: list[ContactConstraint] = []

    if num_contacts >= 3:
        for (i, j), residual in giac_residuals(
            mass, p_B, a_B, L_dot, feet, pairs
        ):
            records.append(
                ContactConstraint(
                    name=f"17b_pair_{i}_{j}",
                    category="17b",
                    kind=INEQUALITY,
                    residual=residual,
                )
            )

    elif num_contacts == 2:
        p_1, p_2 = feet[0], feet[1]
        records.append(
            ContactConstraint(
                name="17c_line",
                category="17c",
                kind=EQUALITY,
                residual=double_support_line_residual(
                    mass, p_B, a_B, L_dot, p_1, p_2
                ),
            )
        )
        records.append(
            ContactConstraint(
                name="17d_moment",
                category="17d",
                kind=INEQUALITY,
                residual=double_support_moment_residual(
                    mass, p_B, a_B, L_dot, p_1, p_2
                ),
            )
        )

    elif num_contacts == 1:
        for axis, residual in zip(
            "xyz",
            single_support_residuals(
                mass, p_B, a_B, L_dot, feet[0]
            ),
        ):
            records.append(
                ContactConstraint(
                    name=f"17e_{axis}",
                    category="17e",
                    kind=EQUALITY,
                    residual=residual,
                )
            )

    else:
        residuals = ballistic_residuals(a_B, L_dot)
        for axis, residual in zip("xyz", residuals[:3]):
            records.append(
                ContactConstraint(
                    name=f"17f_a_B_{axis}",
                    category="17f",
                    kind=EQUALITY,
                    residual=residual,
                )
            )
        for axis, residual in zip("xyz", residuals[3:]):
            records.append(
                ContactConstraint(
                    name=f"17f_L_dot_{axis}",
                    category="17f",
                    kind=EQUALITY,
                    residual=residual,
                )
            )

    if num_contacts > 0:
        if mu is None:
            raise ValueError(
                "N > 0 时必须提供摩擦系数 mu, 以便加入论文式 (17a); "
                "只有 N = 0 (腾空) 才不需要摩擦约束."
            )
        records.extend(friction_constraint_records(a_B, mu, friction_form))

    return ContactConstraintSet(num_contacts, tuple(records))


def add_contact_constraints(
    opti: Any,
    mass: Any,
    p_B: Sequence[Any],
    a_B: Sequence[Any],
    L_dot: Sequence[Any],
    feet: Sequence[Sequence[Any]],
    *,
    mu: Any = None,
    friction_form: str = "squared",
    pairs: Iterable[Sequence[int]] | None = None,
) -> ContactConstraintSet:
    """按接触状态加入 GIAC 与摩擦约束, 并返回残差记录.

    这是 :func:`contact_constraint_records` 的 "加入求解器" 版本:
    等式用 ``subject_to(residual == 0)``, 不等式用
    ``subject_to(residual <= 0)``.

    注意 CasADi 的 ``Opti.subject_to`` 不接受常值约束, 因此这里的
    ``a_B``/``L_dot``/``p_B`` 至少应有一部分是决策变量或参数. 只做
    数值残差检查时请用 :func:`contact_constraint_records`.

    Args:
      opti: 具有 ``subject_to(expr)`` 的对象 (CasADi ``Opti`` 或测试替身).
      mass, p_B, a_B, L_dot, feet: 与 :func:`contact_constraint_records`
        相同.
      mu, friction_form, pairs: 与 :func:`contact_constraint_records`
        相同.

    Returns:
      含全部残差记录的 :class:`ContactConstraintSet`.
    """
    result = contact_constraint_records(
        mass,
        p_B,
        a_B,
        L_dot,
        feet,
        mu=mu,
        friction_form=friction_form,
        pairs=pairs,
    )
    for record in result.records:
        if record.kind == EQUALITY:
            opti.subject_to(record.residual == 0)
        else:
            opti.subject_to(record.residual <= 0)
    return result


def add_giac_constraints(
    opti: Any,
    mass: Any,
    p_B: Sequence[Any],
    a_B: Sequence[Any],
    L_dot: Sequence[Any],
    feet: Sequence[Sequence[Any]],
    pairs: Iterable[Sequence[int]] | None = None,
) -> list[Any]:
    """把 N >= 3 的 GIAC 不等式加入 ``opti``, 返回残差表达式列表.

    Args:
      opti: 具有 ``subject_to(expr)`` 的求解器对象 (CasADi Opti 或测试替身).
      feet: 支撑脚位置序列, 至少 3 个; 每个元素为 3 维位置.
      pairs: 可选的索引对顺序; 默认所有无序对 ``i < j``.

    Raises:
      ValueError: 支撑脚少于 3 个 (双足/单足/腾空属于后续扩展, 不能复用
        这一套多足约束).
    """
    if len(feet) < 3:
        raise ValueError(
            "N >= 3 的 GIAC 约束至少需要 3 个支撑脚; "
            f"实际为 {len(feet)} 个. 双足/单足/腾空支撑请分别使用 "
            "add_contact_constraints (按 len(feet) 自动分派), "
            "不能复用本函数."
        )

    residuals = []
    for _, residual in giac_residuals(mass, p_B, a_B, L_dot, feet, pairs):
        opti.subject_to(residual <= 0)
        residuals.append(residual)
    return residuals


# ---------------------------------------------------------------------- #
# 水平地面摩擦锥
# ---------------------------------------------------------------------- #
def horizontal_norm(a_B: Sequence[Any]) -> Any:
    """``||(I - ez ez^T) a_B|| = sqrt(a_x^2 + a_y^2)``.

    注意: 在原点处该函数的导数为 0/0 (CasADi MX 会得到 NaN 梯度).
    约束本身不受影响, 但求解器可能因梯度未定义而失败; 见
    :func:`friction_squared_residual` 的等价平滑写法.
    """
    return (a_B[0] ** 2 + a_B[1] ** 2) ** 0.5


def friction_residual(a_B: Sequence[Any], mu: Any) -> Any:
    """摩擦锥残差 ``||a_B_xy|| - mu * a_B_z``; 要求 ``<= 0``."""
    return horizontal_norm(a_B) - mu * a_B[2]


def friction_squared_residual(a_B: Sequence[Any], mu: Any) -> Any:
    """摩擦锥的等价平方形式 ``||a_B_xy||^2 - (mu * a_B_z)^2``.

    在 ``a_B_z >= 0`` 的前提下, 它与
    ``||a_B_xy|| <= mu * a_B_z`` 完全等价, 且处处光滑 (无需 sqrt),
    因此适合直接作为 IPOPT 的约束. 必须同时加入 ``a_B_z >= 0``,
    否则会允许 ``a_B_z < 0`` 的伪解.
    """
    return a_B[0] ** 2 + a_B[1] ** 2 - (mu * a_B[2]) ** 2


def _validate_friction(mu: Any, form: str) -> None:
    """校验摩擦系数与约束形式 (对符号 mu 只校验形式)."""
    if form not in ("norm", "squared"):
        raise ValueError(
            f"未知的摩擦约束形式 {form!r}; 可选 'norm' 或 'squared'"
        )
    if isinstance(mu, (int, float, np.integer, np.floating)) and float(mu) <= 0.0:
        raise ValueError(f"摩擦系数 mu 必须为正, 实际为 {mu}")


def add_friction_constraint(
    opti: Any,
    a_B: Sequence[Any],
    mu: Any,
    *,
    form: str = "squared",
) -> list[Any]:
    """把水平地面摩擦锥约束加入 ``opti``.

    Args:
      form: ``"norm"`` 使用原始的 ``||a_B_xy|| - mu a_B_z <= 0``;
        ``"squared"`` 使用等价的平方形式并额外加入 ``a_B_z >= 0``.
        平方形式在原点附近梯度良好, 推荐用于 IPOPT.

    Returns:
      添加的约束表达式列表 (squared 形式为 2 条, norm 形式为 1 条).
    """
    _validate_friction(mu, form)

    if form == "norm":
        residual = friction_residual(a_B, mu)
        opti.subject_to(residual <= 0)
        return [residual]

    residual = friction_squared_residual(a_B, mu)
    normal = a_B[2]
    opti.subject_to(residual <= 0)
    opti.subject_to(normal >= 0)
    return [residual, normal]
