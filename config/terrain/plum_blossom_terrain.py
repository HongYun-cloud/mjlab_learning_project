from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import mujoco
import numpy as np

from mjlab.terrains.terrain_generator import (
    SubTerrainCfg,
    TerrainGeneratorCfg,
    TerrainGeometry,
    TerrainOutput,
)


@dataclass(kw_only=True)
class PlumBlossomPillarsTerrainCfg(SubTerrainCfg):
    """梅花桩地形配置 (Plum Blossom Pillars / Stepping Stones Terrain).

    场景结构组成：
    1. 起始平台 (Start Platform)：
       - 尺寸：80cm × 80cm × 20cm (0.80m × 0.80m × 0.20m)
       - 标高：底面贴合地表 z=0，顶面标高 z=0.20m
    2. 梅花桩组 (Plum Blossom Pillars，共 6 根)：
       - 尺寸：单桩为 20cm × 20cm × 20cm (0.20m × 0.20m × 0.20m)
       - 标高：顶面标高同样为 z=0.20m（与起始平台平齐）
       - 布局：共 6 根桩子沿行进方向交错（左右交替）排列
       - 间距约束：
         * 第 1 根桩子距离起始平台边缘固定为 10cm (0.10m)
         * 第 6 根桩子距离目标平台边缘固定为 10cm (0.10m)
         * 中间各桩子之间沿行进方向均匀交错排列
    3. 目标平台 (End Platform)：
       - 尺寸：80cm × 80cm × 20cm (0.80m × 0.80m × 0.20m)
       - 标高：顶面标高 z=0.20m
    """

    # --- 1. 起始平台参数 ---
    platform_length: float = 0.80
    """起始平台沿行进方向长度 (默认 0.80m = 80cm)"""

    platform_width: float = 0.80
    """起始平台横向宽度 (默认 0.80m = 80cm)"""

    platform_height: float = 0.20
    """起始平台高度 (默认 0.20m = 20cm)"""

    # --- 2. 梅花桩参数 ---
    num_pillars: int = 6
    """梅花桩总数量 (默认 6 个)"""

    pillar_length: float = 0.20
    """梅花桩沿行进方向长度 (默认 0.20m = 20cm)"""

    pillar_width: float = 0.20
    """梅花桩横向宽度 (默认 0.20m = 20cm)"""

    pillar_height: float = 0.20
    """梅花桩高度 (默认 0.20m = 20cm)"""

    pillar_shape: Literal["box", "cylinder"] = "box"
    """桩子几何外形：'box' (方柱) 或 'cylinder' (圆柱)"""

    lateral_offset: float = 0.15
    """桩子横向交错偏离中心线的距离 (单位: 米，默认 0.15m = 15cm，左/右桩中心距 30cm，贴合四足机器人肩宽站姿)"""

    first_pillar_side: Literal["left", "right"] = "left"
    """第一个桩子的交错侧向：'left' 为沿前进方向偏左(+Y)，'right' 偏右(-Y)"""

    # --- 3. 间距与课程难度参数 ---
    platform_to_pillar_gap: float = 0.10
    """最前桩与起始平台、最后桩与目标平台之间的固定间距 (默认 0.10m = 10cm)"""

    pillar_gap: float | None = None
    """相邻梅花桩之间的固定沿行进方向净间距 (单位: 米)。若为 None 则使用 pillar_gap_range 随 difficulty 插值"""

    pillar_gap_range: tuple[float, float] = (0.10, 0.30)
    """相邻梅花桩之间间距的课程变化范围 (默认 10cm ~ 30cm，随 difficulty 从 0 到 1 逐渐拉宽)"""

    vary_gap_with_difficulty: bool = True
    """是否随课程难度调节桩间距"""

    # --- 4. 目标平台参数 ---
    has_second_platform: bool = True
    """是否启用后方的目标平台 (默认 True)"""

    second_platform_length: float = 0.80
    """目标平台沿行进方向长度 (默认 0.80m = 80cm)"""

    second_platform_width: float = 0.80
    """目标平台横向宽度 (默认 0.80m = 80cm)"""

    second_platform_height: float = 0.20
    """目标平台高度 (默认 0.20m = 20cm)"""

    # --- 5. 空间朝向与布局 ---
    direction: Literal["+x", "-x", "+y", "-y"] = "+x"
    """机器人行进方向：
    - '+x': 从 -X 往 +X 方向走，横向为 Y 轴
    - '-x': 从 +X 往 -X 方向走，横向为 Y 轴
    - '+y': 从 -Y 往 +Y 方向走，横向为 X 轴
    - '-y': 从 +Y 往 -Y 方向走，横向为 X 轴
    """

    center_offset: tuple[float, float] = (0.0, 0.0)
    """在当前 sub-terrain patch 内的 (dx, dy) 中心偏移"""

    spawn_on_platform: bool = True
    """机器人出生点是否直接位于起始平台的中央 (默认 True)"""

    spawn_distance_before: float = 1.0
    """当 spawn_on_platform=False 时，出生点位于起始平台前方的水平距离 (米)"""

    # --- 6. 视觉颜色与物理接触 ---
    platform_color: tuple[float, float, float, float] = (0.28, 0.52, 0.72, 1.0)
    """起始平台颜色 (科技深蓝)"""

    second_platform_color: tuple[float, float, float, float] = (0.22, 0.65, 0.55, 1.0)
    """目标平台颜色 (青绿蓝)"""

    pillar_left_color: tuple[float, float, float, float] = (0.85, 0.35, 0.35, 1.0)
    """左侧梅花桩颜色 (赤红/梅红)"""

    pillar_right_color: tuple[float, float, float, float] = (0.90, 0.60, 0.25, 1.0)
    """右侧梅花桩颜色 (琥珀橙)"""

    friction: tuple[float, float, float] = (1.0, 0.005, 0.0001)
    """表面滑动摩擦、扭转摩擦、滚动摩擦"""

    floor_thickness: float = 0.20
    """基准地面平板厚度 (默认 0.20m，顶面标高严格对齐 z = 0.0m)"""

    floor_color: tuple[float, float, float, float] = (0.32, 0.35, 0.38, 1.0)
    """基准地表材质颜色 (深科技灰)"""

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        del rng  # 几何尺寸确定，随机数生成器未使用

        body = spec.body("terrain")
        geometries: list[TerrainGeometry] = []

        # 0. 创建覆盖整块 patch 的基准地面平板 (Ground floor, 顶面位于 z=0.0)
        floor_size_x = self.size[0] / 2.0
        floor_size_y = self.size[1] / 2.0
        floor_geom = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(
                float(np.maximum(1e-4, floor_size_x)),
                float(np.maximum(1e-4, floor_size_y)),
                float(self.floor_thickness / 2.0),
            ),
            pos=(
                float(self.size[0] / 2.0),
                float(self.size[1] / 2.0),
                float(-self.floor_thickness / 2.0),
            ),
        )
        floor_geom.contype = 1
        floor_geom.conaffinity = 1
        for f_idx, f_val in enumerate(self.friction):
            floor_geom.friction[f_idx] = f_val
        geometries.append(TerrainGeometry(geom=floor_geom, color=self.floor_color))

        # 1. 确定梅花桩相邻间距 (effective_pillar_gap)
        if self.pillar_gap is not None:
            effective_pillar_gap = float(self.pillar_gap)
        elif self.vary_gap_with_difficulty:
            diff = float(np.clip(difficulty, 0.0, 1.0))
            effective_pillar_gap = self.pillar_gap_range[0] + diff * (
                self.pillar_gap_range[1] - self.pillar_gap_range[0]
            )
        else:
            effective_pillar_gap = self.pillar_gap_range[0]

        # 2. 计算各部分沿行进方向的长度与全长
        # 布局：起始平台 -> platform_to_pillar_gap -> 6个桩子 (含5个桩间距) -> platform_to_pillar_gap -> 目标平台
        pillars_span_length = (
            self.num_pillars * self.pillar_length
            + max(0, self.num_pillars - 1) * effective_pillar_gap
        )

        total_travel_length = (
            self.platform_length
            + self.platform_to_pillar_gap
            + pillars_span_length
            + (self.platform_to_pillar_gap + self.second_platform_length if self.has_second_platform else 0.0)
        )

        cx = self.size[0] / 2.0 + self.center_offset[0]
        cy = self.size[1] / 2.0 + self.center_offset[1]
        start_travel = -total_travel_length / 2.0

        # 3. 构造所有刚体构件列表: [(name, geom_type, travel_offset, length_t, width_lat, top_z, lateral_y_offset, color)]
        parts = []

        # (a) 起始平台 (80cm × 80cm × 20cm)
        plat1_travel_start = 0.0
        parts.append(
            (
                "platform_start",
                "box",
                plat1_travel_start,
                self.platform_length,
                self.platform_width,
                self.platform_height,
                0.0,  # 平台居中对齐
                self.platform_color,
            )
        )

        # (b) 6 根交错梅花桩 (20cm × 20cm × 20cm)
        pillars_start_offset = self.platform_length + self.platform_to_pillar_gap
        side_multiplier = 1.0 if self.first_pillar_side == "left" else -1.0

        pillar_centers_travel = []
        pillar_centers_lat = []

        for i in range(self.num_pillars):
            pillar_name = f"pillar_{i + 1}"
            pillar_offset = pillars_start_offset + i * (self.pillar_length + effective_pillar_gap)
            # 交替偏置：+lateral_offset 与 -lateral_offset
            lat_sign = side_multiplier * (1.0 if (i % 2 == 0) else -1.0)
            lat_y = lat_sign * self.lateral_offset
            pillar_color = self.pillar_left_color if lat_sign > 0 else self.pillar_right_color

            parts.append(
                (
                    pillar_name,
                    self.pillar_shape,
                    pillar_offset,
                    self.pillar_length,
                    self.pillar_width,
                    self.pillar_height,
                    lat_y,
                    pillar_color,
                )
            )
            pillar_centers_travel.append(pillar_offset + self.pillar_length / 2.0)
            pillar_centers_lat.append(lat_y)

        # (c) 目标平台 (80cm × 80cm × 20cm)
        if self.has_second_platform:
            plat2_travel_start = pillars_start_offset + pillars_span_length + self.platform_to_pillar_gap
            parts.append(
                (
                    "platform_end",
                    "box",
                    plat2_travel_start,
                    self.second_platform_length,
                    self.second_platform_width,
                    self.second_platform_height,
                    0.0,  # 目标平台居中对齐
                    self.second_platform_color,
                )
            )

        # 4. 根据 direction 确定空间位置并在 MuJoCo spec 中生成 Geom
        for name, g_type, travel_offset, length_t, width_lat, top_z, lat_y, color in parts:
            half_z = top_z / 2.0
            pos_z = half_z  # 顶面标高 top_z，底面位于 z=0 地表

            center_travel = start_travel + travel_offset + length_t / 2.0

            if self.direction == "+x":
                pos_x = cx + center_travel
                pos_y = cy + lat_y
                size_x = length_t / 2.0
                size_y = width_lat / 2.0
            elif self.direction == "-x":
                pos_x = cx - center_travel
                pos_y = cy - lat_y
                size_x = length_t / 2.0
                size_y = width_lat / 2.0
            elif self.direction == "+y":
                pos_x = cx - lat_y
                pos_y = cy + center_travel
                size_x = width_lat / 2.0
                size_y = length_t / 2.0
            elif self.direction == "-y":
                pos_x = cx + lat_y
                pos_y = cy - center_travel
                size_x = width_lat / 2.0
                size_y = length_t / 2.0
            else:
                raise ValueError(f"Unsupported direction: {self.direction}")

            if g_type == "cylinder":
                # 圆柱体：size=(radius, half_height)
                radius = min(size_x, size_y)
                geom = body.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                    size=(float(radius), float(half_z)),
                    pos=(float(pos_x), float(pos_y), float(pos_z)),
                )
            else:
                # 矩形方柱体：size=(half_x, half_y, half_z)
                geom = body.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=(
                        float(np.maximum(1e-4, size_x)),
                        float(np.maximum(1e-4, size_y)),
                        float(np.maximum(1e-4, half_z)),
                    ),
                    pos=(float(pos_x), float(pos_y), float(pos_z)),
                )

            geom.contype = 1
            geom.conaffinity = 1
            for f_idx, f_val in enumerate(self.friction):
                geom.friction[f_idx] = f_val

            geometries.append(TerrainGeometry(geom=geom, color=color))

        # 5. 计算关键采样点与标记（出生点、起始平台中心、各梅花桩中心、目标平台中心）
        plat1_center_travel = start_travel + plat1_travel_start + self.platform_length / 2.0
        if self.spawn_on_platform:
            spawn_travel = plat1_center_travel
            spawn_z = float(self.platform_height)
        else:
            spawn_travel = start_travel - self.spawn_distance_before
            spawn_z = 0.0

        if self.has_second_platform:
            plat2_center_travel = (
                start_travel + plat2_travel_start + self.second_platform_length / 2.0
            )
        else:
            plat2_center_travel = plat1_center_travel

        def _coord_from_travel(t_val: float, lat_val: float = 0.0) -> tuple[float, float]:
            if self.direction == "+x":
                return (cx + t_val, cy + lat_val)
            elif self.direction == "-x":
                return (cx - t_val, cy - lat_val)
            elif self.direction == "+y":
                return (cx - lat_val, cy + t_val)
            elif self.direction == "-y":
                return (cx + lat_val, cy - t_val)
            return (cx, cy)

        spawn_x, spawn_y = _coord_from_travel(spawn_travel)
        plat1_x, plat1_y = _coord_from_travel(plat1_center_travel)
        plat2_x, plat2_y = _coord_from_travel(plat2_center_travel)

        origin = np.array([float(spawn_x), float(spawn_y), spawn_z])

        flat_patches = {
            "start": np.array([[float(spawn_x), float(spawn_y), spawn_z]]),
            "platform_start": np.array([[float(plat1_x), float(plat1_y), float(self.platform_height)]]),
        }

        # 登记每一根梅花桩的顶面中心位置
        for idx, (p_t, p_lat) in enumerate(zip(pillar_centers_travel, pillar_centers_lat, strict=True)):
            p_x, p_y = _coord_from_travel(start_travel + p_t, p_lat)
            flat_patches[f"pillar_{idx + 1}"] = np.array(
                [[float(p_x), float(p_y), float(self.pillar_height)]]
            )

        if self.has_second_platform:
            flat_patches["platform_end"] = np.array(
                [[float(plat2_x), float(plat2_y), float(self.second_platform_height)]]
            )
            flat_patches["goal"] = flat_patches["platform_end"]
        else:
            flat_patches["goal"] = flat_patches["platform_start"]

        return TerrainOutput(
            origin=origin,
            geometries=geometries,
            flat_patches=flat_patches,
        )


def plum_blossom_terrain(**overrides: Any) -> PlumBlossomPillarsTerrainCfg:
    """创建梅花桩地形配置构建器。

    场景标准参数（符合用户描述）：
    - 平台：起始平台与目标平台均为 80cm × 80cm × 20cm (0.80m × 0.80m × 0.20m)
    - 梅花桩：共 6 个桩子，尺寸为 20cm × 20cm × 20cm (0.20m × 0.20m × 0.20m)
    - 间距：
      * 最前面的桩子距离起始平台为 10cm (0.10m)
      * 最後面的桩子距离目标平台为 10cm (0.10m)
      * 中间各桩子在左右两侧交错平均放置（横向偏置 15cm，形成交替桩面）
    """
    defaults: dict[str, Any] = dict(
        platform_length=0.80,
        platform_width=0.80,
        platform_height=0.20,
        num_pillars=6,
        pillar_length=0.20,
        pillar_width=0.20,
        pillar_height=0.20,
        pillar_shape="box",
        lateral_offset=0.15,
        first_pillar_side="left",
        platform_to_pillar_gap=0.10,
        pillar_gap=None,
        pillar_gap_range=(0.10, 0.25),
        vary_gap_with_difficulty=True,
        has_second_platform=True,
        second_platform_length=0.80,
        second_platform_width=0.80,
        second_platform_height=0.20,
        direction="+x",
        spawn_on_platform=True,
    )
    defaults.update(overrides)
    return PlumBlossomPillarsTerrainCfg(**defaults)


# 别名兼容
plum_pillars_terrain = plum_blossom_terrain
PlumPillarsTerrainCfg = PlumBlossomPillarsTerrainCfg


# 预置生成器配置模板（仅供配置导入，独立使用，不影响现有训练任务）
PLUM_BLOSSOM_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=5,
    num_cols=1,
    curriculum=True,
    sub_terrains={
        "plum_blossom": plum_blossom_terrain(
            proportion=1.0,
            pillar_gap_range=(0.10, 0.25),
            vary_gap_with_difficulty=True,
        ),
    },
    add_lights=True,
)

