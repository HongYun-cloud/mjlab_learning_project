"""基础姿态的 6D 多项式样条.

状态量 (6 维)::

    [x, y, z, roll, pitch, yaw]

前三维是机身位置 (世界系, 单位 m), 后三维是欧拉角 (单位 rad),
旋转顺序为 ZYX, 即 R_wb = Rz(yaw) @ Ry(pitch) @ Rx(roll).

样条以相位 (phase) 为单位定义. 每个相位有自己的持续时间 T 和一组系数
``a0 .. a_{K-1}``; 相位内使用物理时间 ``tau in [0, T]`` (单位 s)::

    pose(tau) = a0 + a1*tau + a2*tau^2 + ... + a_{K-1}*tau^{K-1}

系数张量布局为 ``[..., 6, K]``, 即 ``coeffs[..., dim, i] == a_i[dim]``,
与参考 TAMOLS 实现中的 ``NewContinuousVariables(base_dims, spline_order)`` 一致.

默认 ``num_coeffs = 5``, 即显式的 a0, a1, a2, a3, a4 五个参数 (四次多项式).
如果后续需要真正的五次多项式 (a0..a5), 构造时传入 ``num_coeffs=6`` 即可,
其余接口不变.

说明:
  - ``tau`` 用物理时间而不是归一化时间, 因此 a0 是相位起始位姿, a1 是起始速度,
    a2 是 1/2 起始加速度, 与 TAMOLS 的初始约束 ``a0[dim,0] == base_pose``、
    ``a0[dim,1] == base_vel`` 直接对应;
  - 所有求值函数都没有对 ``tau`` 做区间裁剪: 相位外求值等价于多项式外推,
    调用方负责只在 ``[0, T]`` 内采样;
  - 支持任意前导 batch 维, 便于多环境 / 多相位同时求值.
"""

from __future__ import annotations

import math

import torch

# 6D 状态的分量划分.
POSITION_DIMS = slice(0, 3)
EULER_DIMS = slice(3, 6)
EULER_NAMES = ("roll", "pitch", "yaw")
EULER_ORDER = "ZYX"


class Spline6D:
    """位置 + 欧拉角的 6D 多项式样条.

    Args:
      num_coeffs: 每个维度的系数个数 K (默认 5, 即 a0..a4).
    """

    BASE_DIMS = 6
    POSITION_DIMS = POSITION_DIMS
    EULER_DIMS = EULER_DIMS

    def __init__(self, num_coeffs: int = 5):
        if not isinstance(num_coeffs, int):
            raise TypeError(f"num_coeffs 需为 int, 实际为 {type(num_coeffs).__name__}")
        if num_coeffs < 1:
            raise ValueError(f"num_coeffs 必须 >= 1, 实际为 {num_coeffs}")
        self.num_coeffs = num_coeffs

    @property
    def degree(self) -> int:
        """多项式次数 K-1."""
        return self.num_coeffs - 1

    def coefficient_shape(self, num_phases: int = 1) -> tuple[int, int, int]:
        """单个 (无 batch) 系数张量的形状 ``[num_phases, 6, K]``."""
        if num_phases < 1:
            raise ValueError(f"num_phases 必须 >= 1, 实际为 {num_phases}")
        return (int(num_phases), self.BASE_DIMS, self.num_coeffs)

    def zeros(
        self,
        num_phases: int = 1,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """返回全零系数张量 ``[num_phases, 6, K]``."""
        return torch.zeros(
            self.coefficient_shape(num_phases), dtype=dtype, device=device
        )

    # ------------------------------------------------------------------ #
    # 求值
    # ------------------------------------------------------------------ #
    def evaluate(
        self,
        coeffs: torch.Tensor,
        tau: float | torch.Tensor,
        order: int = 0,
    ) -> torch.Tensor:
        """在若干时刻求值, 返回对 ``tau`` 的 ``order`` 阶导数.

        Args:
          coeffs: ``[..., 6, K]`` 系数张量.
          tau: 标量或一维时间序列; 内部会展平, 单位与系数时间单位一致 (s).
          order: 求导阶数 (>=0).

        Returns:
          ``[..., N, 6]``, N 为 ``tau`` 的元素个数 (标量视为 N=1).
          ``order >= K`` 时返回全零 (多项式的高阶导数为 0).
        """
        self._check_coeffs(coeffs)
        if not isinstance(order, int):
            raise TypeError(f"order 需为 int, 实际为 {type(order).__name__}")
        if order < 0:
            raise ValueError(f"order 必须 >= 0, 实际为 {order}")

        t = self._as_time_vector(tau, coeffs.dtype, coeffs.device)
        k_out = self.num_coeffs - order
        if k_out <= 0:
            return coeffs.new_zeros(coeffs.shape[:-2] + (t.numel(), self.BASE_DIMS))

        # d^order/dtau^order (a_i * tau^i) = a_i * i!/(i-order)! * tau^(i-order)
        ratio = torch.tensor(
            [
                math.factorial(i) / math.factorial(i - order) if i >= order else 0.0
                for i in range(self.num_coeffs)
            ],
            dtype=coeffs.dtype,
            device=coeffs.device,
        )
        shifted = (coeffs * ratio)[..., order:]  # [..., 6, K-order]
        powers = t.unsqueeze(-1) ** torch.arange(
            k_out, dtype=coeffs.dtype, device=coeffs.device
        )  # [N, K-order]
        return torch.einsum("...dk,nk->...nd", shifted, powers)

    def position(
        self, coeffs: torch.Tensor, tau: float | torch.Tensor, order: int = 0
    ) -> torch.Tensor:
        """位置 (或其对时间的高阶导数) ``[..., N, 3]``."""
        return self.evaluate(coeffs, tau, order)[..., POSITION_DIMS]

    def euler(
        self, coeffs: torch.Tensor, tau: float | torch.Tensor, order: int = 0
    ) -> torch.Tensor:
        """欧拉角 (或其对时间的高阶导数) ``[..., N, 3]``, 顺序 roll, pitch, yaw."""
        return self.evaluate(coeffs, tau, order)[..., EULER_DIMS]

    def velocity(self, coeffs: torch.Tensor, tau: float | torch.Tensor) -> torch.Tensor:
        """位姿速度 ``[..., N, 6]`` (位置线速度 + 欧拉角变化率)."""
        return self.evaluate(coeffs, tau, 1)

    def acceleration(
        self, coeffs: torch.Tensor, tau: float | torch.Tensor
    ) -> torch.Tensor:
        """位姿加速度 ``[..., N, 6]``."""
        return self.evaluate(coeffs, tau, 2)

    def jerk(self, coeffs: torch.Tensor, tau: float | torch.Tensor) -> torch.Tensor:
        """位姿加加速度 ``[..., N, 6]``."""
        return self.evaluate(coeffs, tau, 3)

    # ------------------------------------------------------------------ #
    # 姿态相关
    # ------------------------------------------------------------------ #
    @staticmethod
    def body_omega_from_euler_rates(
        euler: torch.Tensor, euler_rates: torch.Tensor
    ) -> torch.Tensor:
        """把 ZYX 欧拉角变化率 [phidot, thetadot, psidot] 映射为机体系角速度.

        ZYX (yaw-pitch-roll) 下::

            wx = phidot - psidot * sin(theta)
            wy = thetadot * cos(phi) + psidot * sin(phi) * cos(theta)
            wz = -thetadot * sin(phi) + psidot * cos(phi) * cos(theta)

        Args:
          euler: ``[..., 3]``, [roll, pitch, yaw].
          euler_rates: ``[..., 3]``, 欧拉角对时间的一阶导数.

        Returns:
          ``[..., 3]`` 机体系角速度 (rad/s).
        """
        if euler.shape[-1] != 3 or euler_rates.shape[-1] != 3:
            raise ValueError(
                "euler / euler_rates 的最后一维必须为 3, "
                f"实际为 {tuple(euler.shape)} / {tuple(euler_rates.shape)}"
            )
        if euler.shape[:-1] != euler_rates.shape[:-1]:
            raise ValueError(
                "euler / euler_rates 的 batch 形状不一致: "
                f"{tuple(euler.shape[:-1])} vs {tuple(euler_rates.shape[:-1])}"
            )

        phi, theta, _ = euler.unbind(dim=-1)
        phi_dot, theta_dot, psi_dot = euler_rates.unbind(dim=-1)

        sin_phi, cos_phi = torch.sin(phi), torch.cos(phi)
        sin_theta, cos_theta = torch.sin(theta), torch.cos(theta)

        wx = phi_dot - psi_dot * sin_theta
        wy = theta_dot * cos_phi + psi_dot * sin_phi * cos_theta
        wz = -theta_dot * sin_phi + psi_dot * cos_phi * cos_theta
        return torch.stack((wx, wy, wz), dim=-1)

    def body_angular_velocity(
        self, coeffs: torch.Tensor, tau: float | torch.Tensor
    ) -> torch.Tensor:
        """由欧拉角样条计算机体系角速度 ``[..., N, 3]`` (rad/s)."""
        euler = self.euler(coeffs, tau, 0)
        euler_rates = self.euler(coeffs, tau, 1)
        return self.body_omega_from_euler_rates(euler, euler_rates)

    def rotation_matrix(
        self, coeffs: torch.Tensor, tau: float | torch.Tensor
    ) -> torch.Tensor:
        """由欧拉角样条计算 R_wb ``[..., N, 3, 3]`` (ZYX)."""
        phi, theta, psi = self.euler(coeffs, tau, 0).unbind(dim=-1)
        sin_phi, cos_phi = torch.sin(phi), torch.cos(phi)
        sin_theta, cos_theta = torch.sin(theta), torch.cos(theta)
        sin_psi, cos_psi = torch.sin(psi), torch.cos(psi)

        row0 = torch.stack(
            (
                cos_psi * cos_theta,
                cos_psi * sin_theta * sin_phi - sin_psi * cos_phi,
                cos_psi * sin_theta * cos_phi + sin_psi * sin_phi,
            ),
            dim=-1,
        )
        row1 = torch.stack(
            (
                sin_psi * cos_theta,
                sin_psi * sin_theta * sin_phi + cos_psi * cos_phi,
                sin_psi * sin_theta * cos_phi - cos_psi * sin_phi,
            ),
            dim=-1,
        )
        row2 = torch.stack(
            (-sin_theta, cos_theta * sin_phi, cos_theta * cos_phi), dim=-1
        )
        return torch.stack((row0, row1, row2), dim=-2)

    # ------------------------------------------------------------------ #
    # 相位拼接
    # ------------------------------------------------------------------ #
    def junction_residual(
        self,
        coeffs: torch.Tensor,
        durations: float | torch.Tensor | list[float],
        order: int = 0,
    ) -> torch.Tensor:
        """相邻相位的拼接误差.

        对每个相位边界, 计算“上一相位末端”减“下一相位起点”的 ``order`` 阶导数::

            residual[k] = a_k^(order)(T_k) - a_{k+1}^(order)(0)

        Args:
          coeffs: ``[..., P, 6, K]``, P 个相位.
          durations: ``[P]`` 各相位持续时间 (s), 必须为正.
          order: 求导阶数.

        Returns:
          ``[..., P-1, 6]``; P=1 时返回 ``[..., 0, 6]``.
        """
        self._check_coeffs(coeffs)
        if coeffs.dim() < 3:
            raise ValueError(
                "coeffs 需为 [..., num_phases, 6, K], "
                f"实际 shape={tuple(coeffs.shape)}"
            )
        num_phases = coeffs.shape[-3]

        d = torch.as_tensor(
            durations, dtype=coeffs.dtype, device=coeffs.device
        ).reshape(-1)
        if d.numel() != num_phases:
            raise ValueError(
                f"durations 数量 {d.numel()} 与相位个数 {num_phases} 不一致"
            )
        if not bool(torch.isfinite(d).all()) or bool((d <= 0).any()):
            raise ValueError("durations 必须为有限正数")

        if num_phases < 2:
            return coeffs.new_zeros(coeffs.shape[:-3] + (0, self.BASE_DIMS))

        zero = d.new_zeros(())
        ends, starts = [], []
        for phase in range(num_phases - 1):
            ends.append(
                self.evaluate(coeffs[..., phase, :, :], d[phase], order)[..., 0, :]
            )
            starts.append(
                self.evaluate(coeffs[..., phase + 1, :, :], zero, order)[..., 0, :]
            )
        return torch.stack(ends, dim=-2) - torch.stack(starts, dim=-2)

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #
    def _check_coeffs(self, coeffs: torch.Tensor) -> None:
        if not isinstance(coeffs, torch.Tensor):
            raise TypeError(
                f"coeffs 需为 torch.Tensor, 实际为 {type(coeffs).__name__}"
            )
        if not coeffs.is_floating_point():
            raise TypeError(f"coeffs 需为浮点张量, 实际 dtype={coeffs.dtype}")
        if (
            coeffs.dim() < 2
            or coeffs.shape[-2] != self.BASE_DIMS
            or coeffs.shape[-1] != self.num_coeffs
        ):
            raise ValueError(
                f"coeffs 需为 [..., {self.BASE_DIMS}, {self.num_coeffs}], "
                f"实际 shape={tuple(coeffs.shape)}"
            )

    @staticmethod
    def _as_time_vector(
        tau: float | torch.Tensor, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        if isinstance(tau, torch.Tensor):
            t = tau.to(dtype=dtype, device=device)
        else:
            t = torch.as_tensor(tau, dtype=dtype, device=device)
        if t.dim() == 0:
            return t.reshape(1)
        return t.reshape(-1)
