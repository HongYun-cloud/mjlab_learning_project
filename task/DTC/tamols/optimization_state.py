"""TAMOLS 优化状态向量 x 的块布局 / 打包 / 解包.

状态向量按固定顺序拼接四块::

    x = [ spline_coeffs | landing_footholds | desired_footholds | slack ]

各块形状 (按 ``num_phases=P``, ``num_legs=L``, ``num_coeffs=K``, ``num_slack=S``)::

    spline_coeffs      (P, 6, K)   基础姿态 6D 样条系数 a0..a_{K-1}
    landing_footholds  (L, 3)      落地的脚点 (世界系, m)
    desired_footholds  (L, 3)      期望的落脚点 (世界系, m)
    slack              (S,)        松弛变量 (默认每个相位一个, S=P)

``spline_coeffs`` 的展平顺序与 :class:`~myrobot_mjlab.task.DTC.tamols.base_spline.Spline6D`
一致: ``coeffs[phase, dim, i] = a_i[dim]``, 即 C 顺序 (i 变化最快), 共
``6*K`` 个标量对应一个相位.

所有 ``pack`` / ``unpack`` 只作用于最后一维, 因此天然支持前导 batch 维:

    layout.pack(...)                -> [*batch, total_size]
    layout.unpack(x)                -> 每个块 [*batch, *block_shape]
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import torch

STATE_BLOCK_NAMES = (
    "spline_coeffs",
    "landing_footholds",
    "desired_footholds",
    "slack",
)
"""状态向量中块的拼接顺序 (同时是 describe() 的显示顺序)."""


@dataclass(frozen=True)
class OptimizationState:
    """解包后的状态向量各块."""

    spline_coeffs: torch.Tensor
    landing_footholds: torch.Tensor
    desired_footholds: torch.Tensor
    slack: torch.Tensor

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "spline_coeffs": self.spline_coeffs,
            "landing_footholds": self.landing_footholds,
            "desired_footholds": self.desired_footholds,
            "slack": self.slack,
        }

    def __getitem__(self, name: str) -> torch.Tensor:
        try:
            return self.as_dict()[name]
        except KeyError:
            raise KeyError(
                f"未知状态块 {name!r}; 可选: {', '.join(STATE_BLOCK_NAMES)}"
            ) from None


@dataclass(frozen=True)
class OptimizationStateLayout:
    """状态向量 x 的结构定义.

    Args:
      num_phases: 相位个数 P.
      num_legs: 腿数 / 脚点个数 L.
      num_coeffs: 每个样条维度的系数个数 K (默认 5, 即 a0..a4).
      base_dims: 基础姿态维度, 固定为 6 (位置 3 + 欧拉角 3).
      num_slack: 松弛变量个数 S; None 表示每个相位一个 (S=P).
    """

    num_phases: int = 1
    num_legs: int = 4
    num_coeffs: int = 5
    base_dims: int = 6
    num_slack: int | None = None

    def __post_init__(self) -> None:
        for name in ("num_phases", "num_legs", "num_coeffs", "base_dims"):
            value = getattr(self, name)
            if not isinstance(value, int):
                raise TypeError(f"{name} 需为 int, 实际为 {type(value).__name__}")
            if value < 1:
                raise ValueError(f"{name} 必须 >= 1, 实际为 {value}")
        if self.base_dims != 6:
            raise ValueError(
                "base_dims 必须为 6 (位置 3 维 + 欧拉角 3 维), "
                f"实际为 {self.base_dims}"
            )

        if self.num_slack is None:
            object.__setattr__(self, "num_slack", self.num_phases)
        else:
            if not isinstance(self.num_slack, int):
                raise TypeError(
                    f"num_slack 需为 int 或 None, 实际为 {type(self.num_slack).__name__}"
                )
            if self.num_slack < 0:
                raise ValueError(f"num_slack 必须 >= 0, 实际为 {self.num_slack}")

    # ------------------------------------------------------------------ #
    # 形状 / 切片
    # ------------------------------------------------------------------ #
    @property
    def block_shapes(self) -> dict[str, tuple[int, ...]]:
        """各块 (无 batch) 的形状, 按拼接顺序."""
        return {
            "spline_coeffs": (
                self.num_phases,
                self.base_dims,
                self.num_coeffs,
            ),
            "landing_footholds": (self.num_legs, 3),
            "desired_footholds": (self.num_legs, 3),
            "slack": (self.num_slack,),
        }

    @property
    def block_sizes(self) -> dict[str, int]:
        return {
            name: math.prod(shape) for name, shape in self.block_shapes.items()
        }

    @property
    def slices(self) -> dict[str, slice]:
        """各块在 x 最后一维上的切片."""
        out: dict[str, slice] = {}
        start = 0
        for name in STATE_BLOCK_NAMES:
            size = self.block_sizes[name]
            out[name] = slice(start, start + size)
            start += size
        return out

    @property
    def total_size(self) -> int:
        """x 的标量元素个数 (单个环境)."""
        return sum(self.block_sizes.values())

    def slice_of(self, name: str) -> slice:
        try:
            return self.slices[name]
        except KeyError:
            raise KeyError(
                f"未知状态块 {name!r}; 可选: {', '.join(STATE_BLOCK_NAMES)}"
            ) from None

    # ------------------------------------------------------------------ #
    # 打包 / 解包
    # ------------------------------------------------------------------ #
    def pack(
        self,
        spline_coeffs: torch.Tensor,
        landing_footholds: torch.Tensor,
        desired_footholds: torch.Tensor,
        slack: torch.Tensor,
    ) -> torch.Tensor:
        """把四块拼成 ``[..., total_size]``.

        每块的最后若干维必须与 :attr:`block_shapes` 完全一致,
        且所有块的前导 batch 维必须相同.

        Raises:
          TypeError: 输入不是浮点张量.
          ValueError: 块形状或 batch 维不匹配.
        """
        tensors: dict[str, torch.Tensor] = {
            "spline_coeffs": spline_coeffs,
            "landing_footholds": landing_footholds,
            "desired_footholds": desired_footholds,
            "slack": slack,
        }

        batch_shape: tuple[int, ...] | None = None
        flat: list[torch.Tensor] = []
        for name in STATE_BLOCK_NAMES:
            t = tensors[name]
            if not isinstance(t, torch.Tensor):
                raise TypeError(
                    f"{name} 需为 torch.Tensor, 实际为 {type(t).__name__}"
                )
            if not t.is_floating_point():
                raise TypeError(f"{name} 需为浮点张量, 实际 dtype={t.dtype}")

            expected = self.block_shapes[name]
            if t.dim() < len(expected) or tuple(t.shape[-len(expected):]) != expected:
                raise ValueError(
                    f"{name} 的形状应为 [*batch, {', '.join(map(str, expected))}], "
                    f"实际 shape={tuple(t.shape)}"
                )

            current_batch = tuple(t.shape[: -len(expected)])
            if batch_shape is None:
                batch_shape = current_batch
            elif current_batch != batch_shape:
                raise ValueError(
                    f"{name} 的 batch 维 {current_batch} 与前面的块 {batch_shape} 不一致"
                )
            flat.append(t.reshape(*current_batch, -1))

        return torch.cat(flat, dim=-1)

    def pack_state(self, state: OptimizationState | Mapping[str, torch.Tensor]) -> torch.Tensor:
        """``pack`` 的字典 / dataclass 版本."""
        values = state.as_dict() if isinstance(state, OptimizationState) else state
        return self.pack(
            spline_coeffs=values["spline_coeffs"],
            landing_footholds=values["landing_footholds"],
            desired_footholds=values["desired_footholds"],
            slack=values["slack"],
        )

    def unpack(self, x: torch.Tensor) -> OptimizationState:
        """把 ``[..., total_size]`` 还原为四块."""
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"x 需为 torch.Tensor, 实际为 {type(x).__name__}")
        if x.dim() < 1 or x.shape[-1] != self.total_size:
            raise ValueError(
                f"x 的最后一维必须为 total_size={self.total_size}, "
                f"实际 shape={tuple(x.shape)}"
            )

        batch_shape = tuple(x.shape[:-1])
        values: dict[str, torch.Tensor] = {}
        for name in STATE_BLOCK_NAMES:
            values[name] = x[..., self.slice_of(name)].reshape(
                *batch_shape, *self.block_shapes[name]
            )
        return OptimizationState(**values)

    def zeros(
        self,
        batch_shape: tuple[int, ...] = (),
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """返回全零状态向量 ``[*batch_shape, total_size]``."""
        return torch.zeros(*batch_shape, self.total_size, dtype=dtype, device=device)

    # ------------------------------------------------------------------ #
    # 可读描述
    # ------------------------------------------------------------------ #
    def describe(self) -> str:
        """返回状态向量的可读布局表."""
        lines = [
            f"OptimizationStateLayout: P={self.num_phases}, L={self.num_legs}, "
            f"K={self.num_coeffs}, S={self.num_slack}, total={self.total_size}",
            "x = [",
        ]
        for name in STATE_BLOCK_NAMES:
            block = self.block_shapes[name]
            start = self.slice_of(name).start or 0
            lines.append(
                f"  {name:<18s} {str(block):<16s} size={self.block_sizes[name]:>4d}  "
                f"[{start}:{start + self.block_sizes[name]})"
            )
        lines.append("]")
        return "\n".join(lines)
