from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import CameraSensor
from mjlab.utils.lab_api.math import wrap_to_pi
from myrobot_mjlab.config.reward.Trackingreward import _get_goal_pos

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


# ==============================================================================
# 地形选项 (Terrain options)
# ==============================================================================

class TerrainType(IntEnum):
    """当前支持的地形类别。

    one-hot 的维度顺序严格等于枚举顺序；新增地形时只需在末尾追加成员
    （并补充一条别名/关键字映射），观测维度 ``NUM_TERRAIN_TYPES`` 会自动更新。
    """

    FLAT = 0
    """平地 (plane / flat)"""
    STAIR_UP = 1
    """上楼梯 (stair up)"""
    STAIR_DOWN = 2
    """下楼梯 (stair down)"""
    TIRE_ARRAY = 3
    """轮胎阵 (tire array)"""
    PLATFORM = 4
    """沟壑 / 两平台跨越 (platform / trench)"""
    SINGLE_BEAM = 5
    """单木桥 (single wooden beam / bridge)"""
    PLUM = 6
    """梅花桩 (plum blossom field)"""
    STONE_2ROWS = 7
    """双排石 (stones-2rows)"""
    LOW_CLEARANCE_BAR = 8
    """低净空横杆 (low clearance bar)"""
    STAIR_COMBINED = 9
    """组合楼梯 (up + flat gap + down)"""
    POLE_SLALOM = 10
    """绕桩 (pole slalom / weave poles)，占位类型"""
    ROUGH_GROUND = 11
    """粗糙地面 (rough ground)，占位类型"""
    SLOPE = 12
    """斜坡 (slope / ramp)，占位类型"""


TERRAIN_TYPE_NAMES: tuple[str, ...] = tuple(t.name.lower() for t in TerrainType)
NUM_TERRAIN_TYPES: int = len(TerrainType)

# 名称/类别别名 -> 地形选项。任务配置里可以直接传这些字符串。
_TERRAIN_TYPE_ALIASES: dict[str, TerrainType] = {
    "flat": TerrainType.FLAT,
    "plane": TerrainType.FLAT,
    "平地": TerrainType.FLAT,
    "stair_up": TerrainType.STAIR_UP,
    "stairs_up": TerrainType.STAIR_UP,
    "up_stairs": TerrainType.STAIR_UP,
    "stairup": TerrainType.STAIR_UP,
    "stair_down": TerrainType.STAIR_DOWN,
    "stairs_down": TerrainType.STAIR_DOWN,
    "down_stairs": TerrainType.STAIR_DOWN,
    "stairdown": TerrainType.STAIR_DOWN,
    # 兼容旧配置：没有方向信息的 "stairs" 是原来的倒金字塔。
    "stairs": TerrainType.STAIR_DOWN,
    "stair": TerrainType.STAIR_DOWN,
    "楼梯": TerrainType.STAIR_DOWN,
    "stair_combined": TerrainType.STAIR_COMBINED,
    "combined_stair": TerrainType.STAIR_COMBINED,
    "combined_stairs": TerrainType.STAIR_COMBINED,
    "tire": TerrainType.TIRE_ARRAY,
    "tires": TerrainType.TIRE_ARRAY,
    "tire_array": TerrainType.TIRE_ARRAY,
    "tyre": TerrainType.TIRE_ARRAY,
    "tyre_array": TerrainType.TIRE_ARRAY,
    "轮胎阵": TerrainType.TIRE_ARRAY,
    "platform": TerrainType.PLATFORM,
    "platforms": TerrainType.PLATFORM,
    "trench": TerrainType.PLATFORM,
    "gap": TerrainType.PLATFORM,
    "沟壑": TerrainType.PLATFORM,
    "single_beam": TerrainType.SINGLE_BEAM,
    "beam": TerrainType.SINGLE_BEAM,
    "single_wooden_beam": TerrainType.SINGLE_BEAM,
    "wooden_beam": TerrainType.SINGLE_BEAM,
    "bridge": TerrainType.SINGLE_BEAM,
    "wooden_bridge": TerrainType.SINGLE_BEAM,
    "single_wooden_bridge": TerrainType.SINGLE_BEAM,
    "单木桥": TerrainType.SINGLE_BEAM,
    "plum": TerrainType.PLUM,
    "plum_blossom": TerrainType.PLUM,
    "plum_blossoms": TerrainType.PLUM,
    "plum_blossom_field": TerrainType.PLUM,
    "梅花桩": TerrainType.PLUM,
    "stone": TerrainType.STONE_2ROWS,
    "stones": TerrainType.STONE_2ROWS,
    "stone_2rows": TerrainType.STONE_2ROWS,
    "stones_2rows": TerrainType.STONE_2ROWS,
    "双排石": TerrainType.STONE_2ROWS,
    "low_clearance_bar": TerrainType.LOW_CLEARANCE_BAR,
    "low_clearance": TerrainType.LOW_CLEARANCE_BAR,
    "clearance_bar": TerrainType.LOW_CLEARANCE_BAR,
    "低净空横杆": TerrainType.LOW_CLEARANCE_BAR,
    # 占位类型：只定义观测类型，暂不生成对应地形。
    "pole_slalom": TerrainType.POLE_SLALOM,
    "pole_slaloms": TerrainType.POLE_SLALOM,
    "slalom": TerrainType.POLE_SLALOM,
    "weave_poles": TerrainType.POLE_SLALOM,
    "pole_weave": TerrainType.POLE_SLALOM,
    "绕桩": TerrainType.POLE_SLALOM,
    "rough": TerrainType.ROUGH_GROUND,
    "rough_ground": TerrainType.ROUGH_GROUND,
    "rough_terrain": TerrainType.ROUGH_GROUND,
    "uneven_ground": TerrainType.ROUGH_GROUND,
    "粗糙地面": TerrainType.ROUGH_GROUND,
    "slope": TerrainType.SLOPE,
    "ramp": TerrainType.SLOPE,
    "incline": TerrainType.SLOPE,
    "inclined_plane": TerrainType.SLOPE,
    "斜坡": TerrainType.SLOPE,
}

# 用于从地形生成器的 sub_terrain 名称/类名中自动推断类型。
_TERRAIN_TYPE_KEYWORDS: tuple[tuple[str, TerrainType], ...] = (
    ("stair_up", TerrainType.STAIR_UP),
    ("stairs_up", TerrainType.STAIR_UP),
    ("up_stair", TerrainType.STAIR_UP),
    ("stair_down", TerrainType.STAIR_DOWN),
    ("stairs_down", TerrainType.STAIR_DOWN),
    ("down_stair", TerrainType.STAIR_DOWN),
    ("stair_combined", TerrainType.STAIR_COMBINED),
    ("combined_stair", TerrainType.STAIR_COMBINED),
    ("combined_stairs", TerrainType.STAIR_COMBINED),
    ("stair_combo", TerrainType.STAIR_COMBINED),
    # 没有方向信息的旧 "stair" 是倒金字塔。
    ("stair", TerrainType.STAIR_DOWN),
    ("plum", TerrainType.PLUM),
    ("stone_2rows", TerrainType.STONE_2ROWS),
    ("stones_2rows", TerrainType.STONE_2ROWS),
    ("stone", TerrainType.STONE_2ROWS),
    ("low_clearance", TerrainType.LOW_CLEARANCE_BAR),
    ("clearance_bar", TerrainType.LOW_CLEARANCE_BAR),
    ("single_beam", TerrainType.SINGLE_BEAM),
    ("wooden_beam", TerrainType.SINGLE_BEAM),
    ("wooden_bridge", TerrainType.SINGLE_BEAM),
    ("single_wood", TerrainType.SINGLE_BEAM),
    ("bridge", TerrainType.SINGLE_BEAM),
    ("beam", TerrainType.SINGLE_BEAM),
    ("tire", TerrainType.TIRE_ARRAY),
    ("tyre", TerrainType.TIRE_ARRAY),
    ("platform", TerrainType.PLATFORM),
    ("trench", TerrainType.PLATFORM),
    ("gap", TerrainType.PLATFORM),
    # 占位类型：只用于从 sub_terrain 名称自动推断观测类型。
    ("pole_slalom", TerrainType.POLE_SLALOM),
    ("weave_pole", TerrainType.POLE_SLALOM),
    ("slalom", TerrainType.POLE_SLALOM),
    ("rough_ground", TerrainType.ROUGH_GROUND),
    ("rough_terrain", TerrainType.ROUGH_GROUND),
    ("rough", TerrainType.ROUGH_GROUND),
    ("slope", TerrainType.SLOPE),
    ("ramp", TerrainType.SLOPE),
    ("incline", TerrainType.SLOPE),
    ("flat", TerrainType.FLAT),
    ("plane", TerrainType.FLAT),
)


def resolve_terrain_type(terrain: str | int | TerrainType) -> TerrainType:
    """把地形名称/编号统一解析为 :class:`TerrainType`。

    支持：
    - ``TerrainType`` 成员；
    - 整数编号（枚举顺序）；
    - 大小写不敏感的别名（如 ``"flat"`` / ``"stairs"`` / ``"single_beam"``）；
    - 包含关键字的 sub_terrain 名称或类名（如 ``"pyramid_stairs"``）。
    """
    if isinstance(terrain, TerrainType):
        return terrain
    if isinstance(terrain, int):
        return TerrainType(terrain)

    key = str(terrain).strip().lower().replace("-", "_").replace(" ", "_")
    if key in _TERRAIN_TYPE_ALIASES:
        return _TERRAIN_TYPE_ALIASES[key]

    for keyword, terrain_type in _TERRAIN_TYPE_KEYWORDS:
        if keyword in key:
            return terrain_type

    valid = ", ".join(TERRAIN_TYPE_NAMES)
    raise ValueError(f"未知地形选项 '{terrain}'，可选值: {valid}")


def _detect_terrain_type_ids(env: ManagerBasedRlEnv) -> torch.Tensor | None:
    """从当前场景自动推断每个环境的地形类别编号。

    返回 ``(num_envs,)`` 的 long 张量；无法推断时返回 ``None``。
    """
    terrain = getattr(env.scene, "terrain", None)
    if terrain is None:
        return None

    terrain_cfg = getattr(terrain, "cfg", None)
    if terrain_cfg is None:
        return None

    if getattr(terrain_cfg, "terrain_type", None) == "plane":
        return torch.full(
            (env.num_envs,), int(TerrainType.FLAT), device=env.device, dtype=torch.long
        )

    generator_cfg = getattr(terrain_cfg, "terrain_generator", None)
    sub_terrains = getattr(generator_cfg, "sub_terrains", None)
    if not sub_terrains:
        return None

    column_types: list[int] = []
    for name, sub_cfg in sub_terrains.items():
        try:
            column_types.append(int(resolve_terrain_type(name)))
        except ValueError:
            try:
                column_types.append(int(resolve_terrain_type(type(sub_cfg).__name__)))
            except ValueError:
                # 出现未知 sub_terrain 时放弃自动推断，由显式参数指定。
                return None

    mapping = torch.tensor(column_types, device=env.device, dtype=torch.long)
    terrain_types = getattr(terrain, "terrain_types", None)
    if terrain_types is None:
        if len(set(column_types)) == 1:
            return torch.full(
                (env.num_envs,), column_types[0], device=env.device, dtype=torch.long
            )
        return None

    if isinstance(terrain_types, torch.Tensor):
        indices = terrain_types.to(device=env.device, dtype=torch.long).reshape(-1)
    else:
        indices = torch.as_tensor(
            terrain_types, device=env.device, dtype=torch.long
        ).reshape(-1)
    if indices.numel() == 0 or mapping.numel() == 0:
        return None
    return mapping[indices.clamp(0, mapping.numel() - 1)]


def terrain_type(
    env: ManagerBasedRlEnv,
    terrain_type: str | int | TerrainType | None = None,
    one_hot: bool = True,
    **kwargs,
) -> torch.Tensor:
    """地形选项观测。

    Args:
        terrain_type:
            显式指定当前任务的地形类别（名称、整数或 :class:`TerrainType`）。
            传 ``None`` 时由场景中的 terrain generator / plane 自动推断。
        one_hot:
            ``True`` 返回 ``(num_envs, NUM_TERRAIN_TYPES)`` 的 one-hot 向量
            （默认，适合直接输入策略网络）；``False`` 返回 ``(num_envs, 1)``
            的地形编号。

    地形顺序固定为：
        0. 平地 (flat)
        1. 上楼梯 (stair up)
        2. 下楼梯 (stair down)
        3. 轮胎阵 (tire array)
        4. 沟壑/平台 (platform)
        5. 单木桥 (single beam)
        6. 梅花桩 (plum)
        7. 双排石 (stones-2rows)
        8. 低净空横杆 (low clearance bar)
        9. 组合楼梯 (up + flat gap + down)
        10. 绕桩 (pole slalom，占位)
        11. 粗糙地面 (rough ground，占位)
        12. 斜坡 (slope，占位)
    """
    if "terrain" in kwargs and kwargs["terrain"] is not None:
        terrain_type = kwargs["terrain"]

    if terrain_type is not None:
        terrain_ids = torch.full(
            (env.num_envs,),
            int(resolve_terrain_type(terrain_type)),
            device=env.device,
            dtype=torch.long,
        )
    else:
        detected = _detect_terrain_type_ids(env)
        if detected is None:
            # 无法自动识别时按平地处理，保证观测维度稳定。
            terrain_ids = torch.full(
                (env.num_envs,), int(TerrainType.FLAT), device=env.device, dtype=torch.long
            )
        else:
            terrain_ids = detected.to(device=env.device, dtype=torch.long).reshape(-1)
            if terrain_ids.numel() != env.num_envs:
                terrain_ids = terrain_ids[:1].expand(env.num_envs)

    if not one_hot:
        return terrain_ids.to(dtype=torch.float32).unsqueeze(-1)

    one_hot_obs = torch.zeros(
        (env.num_envs, NUM_TERRAIN_TYPES), device=env.device, dtype=torch.float32
    )
    valid_ids = terrain_ids.clamp(0, NUM_TERRAIN_TYPES - 1)
    one_hot_obs.scatter_(1, valid_ids.unsqueeze(-1), 1.0)
    return one_hot_obs


# ==============================================================================
# Actor 观测项
# ==============================================================================

def direction_to_goal(
    env: ManagerBasedRlEnv,
    command_name: str = "Goal_direction",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    in_body_frame: bool = True,
    normalize: bool = False,
    **kwargs,
) -> torch.Tensor:
    """当前位置到目标位置的向量 (r_goal - x)。Shape: (num_envs, 2).

    Args:
        in_body_frame: 是否投影到机身局部坐标系 (推荐 True，符合 Actor 自身观测视角).
        normalize: 是否归一化为单位向量 (默认 False，保留距离幅度信息).
    """
    asset: Entity = env.scene[asset_cfg.name]
    robot_pos = asset.data.root_link_pos_w - env.scene.env_origins
    goal_pos = _get_goal_pos(env, command_name=command_name, **kwargs)

    to_goal_w = goal_pos[:, :2] - robot_pos[:, :2]

    if in_body_frame:
        # 旋转到机器人机身局部坐标系 (前向 +X, 左向 +Y)
        yaw = asset.data.heading_w
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        dx_b = cos_yaw * to_goal_w[:, 0] + sin_yaw * to_goal_w[:, 1]
        dy_b = -sin_yaw * to_goal_w[:, 0] + cos_yaw * to_goal_w[:, 1]
        vec = torch.stack([dx_b, dy_b], dim=-1)
    else:
        vec = to_goal_w

    if normalize:
        norm = torch.linalg.norm(vec, dim=-1, keepdim=True)
        vec = torch.where(norm > 1e-6, vec / norm, torch.zeros_like(vec))

    return vec


def heading_to_goal_error(
    env: ManagerBasedRlEnv,
    command_name: str = "Goal_direction",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    **kwargs,
) -> torch.Tensor:
    """当前机身朝向与目标方向的夹角误差 (wrap 到 [-pi, pi]). Shape: (num_envs, 1)."""
    asset: Entity = env.scene[asset_cfg.name]
    robot_pos = asset.data.root_link_pos_w - env.scene.env_origins
    goal_pos = _get_goal_pos(env, command_name=command_name, **kwargs)

    to_goal = goal_pos[:, :2] - robot_pos[:, :2]
    theta_goal = torch.atan2(to_goal[:, 1], to_goal[:, 0])
    heading = asset.data.heading_w
    head_err = wrap_to_pi(theta_goal - heading)
    return head_err.unsqueeze(-1)


def target_distance(
    env: ManagerBasedRlEnv,
    command_name: str = "Goal_direction",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    **kwargs,
) -> torch.Tensor:
    """当前机身到目标点的 XY 距离. Shape: (num_envs, 1)."""
    asset: Entity = env.scene[asset_cfg.name]
    robot_pos = asset.data.root_link_pos_w - env.scene.env_origins
    goal_pos = _get_goal_pos(env, command_name=command_name, **kwargs)
    dist = torch.linalg.norm(goal_pos[:, :2] - robot_pos[:, :2], dim=-1)
    return dist.unsqueeze(-1)


def time_left(
    env: ManagerBasedRlEnv,
    normalize: bool = True,
    **kwargs,
) -> torch.Tensor:
    """剩余 episode 时间. 默认归一化到 [0, 1]. Shape: (num_envs, 1)."""
    remaining = (
        env.max_episode_length - env.episode_length_buf
    ).clamp(min=0).float() * env.step_dt
    if normalize:
        remaining = remaining / max(float(env.max_episode_length_s), 1e-6)
    return remaining.unsqueeze(-1)


def _gaussian_blur_depth(
    depth: torch.Tensor,
    kernel_size: int,
    sigma: float,
) -> torch.Tensor:
    """对 ``(B, 1, H, W)`` 深度图做可分离高斯模糊。"""
    if kernel_size <= 1:
        return depth

    coords = torch.arange(
        kernel_size, device=depth.device, dtype=depth.dtype
    ) - (kernel_size - 1) / 2.0
    g = torch.exp(-0.5 * (coords / max(sigma, 1e-6)) ** 2)
    g = g / g.sum()

    pad = kernel_size // 2
    # 先横向再纵向；depth 只有 1 个通道，直接做 depthwise 卷积。
    depth = F.conv2d(depth, g.view(1, 1, 1, -1), padding=(0, pad))
    depth = F.conv2d(depth, g.view(1, 1, -1, 1), padding=(pad, 0))
    return depth


def camera_depth(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    cutoff_distance: float = 5.0,
    min_depth: float = 0.01,
    target_height: int = 18,
    target_width: int = 32,
    blur_kernel_size: int = 3,
    blur_sigma: float = 0.8,
) -> torch.Tensor:
    """深度相机观测：裁剪 + 压缩 + 高斯模糊，输出 ``(B, 1, H, W)``。

    原始深度图（例如 ``270×480``，即 ``H×W``）先做**中心裁剪**，裁掉
    左右多余视野，使宽高比与目标一致；再用双线性插值**压缩**到目标分辨率
    （默认 ``18×32``，即 ``H=18, W=32``）；最后做**高斯模糊**。

    无效像素（天空 / 超出量程）在 MuJoCo 中为 0，处理前会按最远距离
    ``cutoff_distance`` 填充，避免压缩/模糊后变成近处假障碍。最终深度被
    裁剪到 ``[min_depth, cutoff_distance]`` 并归一化到 ``[0, 1]``。

    Args:
        sensor_name: 场景中 :class:`CameraSensorCfg` 的名称。
        cutoff_distance: 最大有效深度（米），同时用于归一化。
        min_depth: 最小有效深度（米），避免 0 深度导致后续网络输入退化。
        target_height: 输出深度图高度 ``H``。
        target_width: 输出深度图宽度 ``W``。
        blur_kernel_size: 高斯核大小（奇数）；``<=1`` 时不做模糊。
        blur_sigma: 高斯核标准差（以输出图像像素为单位）。
    """
    sensor = env.scene[sensor_name]
    assert isinstance(sensor, CameraSensor)
    depth = sensor.data.depth  # (num_envs, H, W, 1)
    assert depth is not None, f"Camera '{sensor_name}' has no depth data"

    depth = depth.permute(0, 3, 1, 2)  # (num_envs, 1, H, W)

    # 无效像素按最远距离处理，避免模糊/压缩时污染邻近的有效深度。
    depth = torch.where(
        depth <= 0.0,
        torch.full_like(depth, cutoff_distance),
        depth,
    )
    depth = depth.clamp(min=min_depth, max=cutoff_distance)

    # 1) 裁剪：按目标宽高比做中心裁剪（只裁宽度，保留完整高度）。
    raw_height, raw_width = depth.shape[-2:]
    crop_width = min(
        raw_width,
        max(1, round(raw_height * target_width / target_height)),
    )
    left = (raw_width - crop_width) // 2
    depth = depth[..., left : left + crop_width]

    # 2) 压缩：双线性插值到目标分辨率。
    if depth.shape[-2:] != (target_height, target_width):
        depth = F.interpolate(
            depth,
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        )

    # 3) 高斯模糊。
    depth = _gaussian_blur_depth(depth, blur_kernel_size, blur_sigma)

    return depth.clamp(min=min_depth, max=cutoff_distance) / cutoff_distance


# ==============================================================================
# Critic 特权观测项
# ==============================================================================

def base_height(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """机身相对环境原点的真实垂直高度 (z - z_origin). Shape: (num_envs, 1)."""
    asset: Entity = env.scene[asset_cfg.name]
    pos_z = asset.data.root_link_pos_w[:, 2:3] - env.scene.env_origins[:, 2:3]
    return pos_z


def direction_real(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    eps: float = 1e-6,
) -> torch.Tensor:
    """机器人当前真实运动方向 (XY 平面归一化速度向量). Shape: (num_envs, 2)."""
    asset: Entity = env.scene[asset_cfg.name]
    vel_xy = asset.data.root_link_lin_vel_w[:, :2]
    speed = torch.linalg.norm(vel_xy, dim=-1, keepdim=True)
    return torch.where(speed > eps, vel_xy / speed, torch.zeros_like(vel_xy))


def friction_coefficient(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """地面/足端接触摩擦系数 mu. Shape: (num_envs, 1)."""
    asset: Entity = env.scene[asset_cfg.name]
    if hasattr(env.sim, "model") and hasattr(env.sim.model, "geom_friction"):
        fric = env.sim.model.geom_friction
        if fric.ndim == 3:  # (num_envs, n_geoms, 3)
            geom_idx = asset.indexing.geom_ids[0] if len(asset.indexing.geom_ids) > 0 else 0
            return fric[:, geom_idx, 0:1]
        elif fric.ndim == 2:  # (n_geoms, 3)
            geom_idx = asset.indexing.geom_ids[0] if len(asset.indexing.geom_ids) > 0 else 0
            val = fric[geom_idx, 0]
            return torch.full((env.num_envs, 1), float(val), device=env.device)
    return torch.ones((env.num_envs, 1), device=env.device)


__all__ = [
    "TerrainType",
    "TERRAIN_TYPE_NAMES",
    "NUM_TERRAIN_TYPES",
    "resolve_terrain_type",
    "terrain_type",
    "direction_to_goal",
    "heading_to_goal_error",
    "camera_depth",
    "base_height",
    "direction_real",
    "friction_coefficient",
]
