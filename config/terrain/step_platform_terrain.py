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
class StepPlatformTerrainCfg(SubTerrainCfg):
    """两阶台阶 + 第一小平台 + 课程难度间隔 (Gap) + 第二小平台的地形配置。

    结构剖析（沿行进方向）：
    1. 上行台阶（固定 2 阶）：
       - 每阶升高：12cm = 0.12m (step_height = 0.12)
       - 台阶步深：32cm = 0.32m (step_width = 0.32，沿行进方向)
       - 台阶横宽：80cm = 0.80m (step_length = 0.80)
       - 标高：第 1 阶顶面 z = 0.12m，第 2 阶顶面 z = 0.24m
    2. 第一个长方体小平台：
       - 尺寸：80cm × 80cm × 36cm (0.80m × 0.80m × 0.36m)
       - 标高：顶面 z = 0.36m（比第 2 阶再高出 12cm，无缝衔接）
    3. 平台间距间隔（Gap，可随课程难度动态调整）：
       - 长度：可通过 gap_distance 固定，或通过 gap_distance_range 随 difficulty 动态插值
       - 空间：两座高 36cm 的平台之间无几何体，形成中空跨越沟壑
    4. 第二个长方体小平台：
       - 尺寸：80cm × 80cm × 36cm (0.80m × 0.80m × 0.36m)
       - 标高：顶面 z = 0.36m
    5. 出生点：
       - 默认 (spawn_on_platform=True) 位于第一平台顶面中心，origin.z = 0.36m，
         机器人一出生就站在平台上，正对间隙与第二平台；
       - 目标点位于第二平台中心 (相对出生点 +0.8m+gap)。
    """

    # --- 1. 上行台阶参数 ---
    step_height: float = 0.12
    """每阶台阶高度 (默认 0.12m = 12cm)"""

    step_width: float = 0.32
    """台阶步深/沿行进方向宽度 (默认 0.32m = 32cm)"""

    step_length: float = 0.80
    """台阶横向跨度/长度 (默认 0.80m = 80cm)"""

    num_steps: int = 2
    """上行台阶阶数 (默认固定 2 阶)"""

    # --- 2. 第一个长方体平台参数 ---
    platform_length: float = 0.80
    """第一平台沿行进方向长度 (默认 0.80m = 80cm)"""

    platform_width: float = 0.80
    """第一平台横向宽度 (默认 0.80m = 80cm)"""

    platform_height: float = 0.36
    """第一平台高度 (默认 0.36m = 36cm)"""

    # --- 3. 间隔 (Gap) 与课程难度参数 ---
    has_second_platform: bool = True
    """是否启用后方的第二平台与间隔 (默认 True)"""

    gap_distance: float | None = None
    """两平台之间的固定间距 (单位: 米)。若为 None 则使用 gap_distance_range 按 difficulty 自动插值"""

    gap_distance_range: tuple[float, float] = (0.20, 0.60)
    """间距的课程变化范围 (单位: 米，默认 20cm ~ 60cm，随 difficulty 从 0 到 1 逐渐拉宽)"""

    vary_gap_with_difficulty: bool = True
    """是否随课程难度调节间距大小"""

    # --- 4. 第二个长方体平台参数 ---
    second_platform_length: float = 0.80
    """第二平台沿行进方向长度 (默认 0.80m = 80cm)"""

    second_platform_width: float = 0.80
    """第二平台横向宽度 (默认 0.80m = 80cm)"""

    second_platform_height: float = 0.36
    """第二平台高度 (默认 0.36m = 36cm)"""

    # --- 5. 空间朝向与布局 ---
    direction: Literal["+x", "-x", "+y", "-y"] = "+x"
    """机器人行进与台阶上升方向：
    - '+x': 从 -X 往 +X 方向走，横向为 Y 轴
    - '-x': 从 +X 往 -X 方向走，横向为 Y 轴
    - '+y': 从 -Y 往 +Y 方向走，横向为 X 轴
    - '-y': 从 +Y 往 -Y 方向走，横向为 X 轴
    """

    add_down_stairs: bool = False
    """是否在第二平台后方添加对称的下行台阶 (默认 False)"""

    center_offset: tuple[float, float] = (0.0, 0.0)
    """在当前 sub-terrain patch 内的 (dx, dy) 中心偏移"""

    spawn_on_platform: bool = True
    """出生点是否放在第一平台顶面中心。

    True : 出生点 = 第一平台顶面中心，origin.z = platform_height。
           机器人一出生就站在平台上，正对间隙与第二平台。
    False: 出生点 = 第一阶台阶前方 spawn_distance_before 米处的基准地面 (旧行为)。
    """

    spawn_distance_before: float = 1.5
    """spawn_on_platform=False 时，出生点位于第一阶台阶前方的水平距离 (单位: 米)"""

    # --- 6. 视觉颜色与物理接触 ---
    step_color: tuple[float, float, float, float] = (0.35, 0.42, 0.55, 1.0)
    """台阶颜色 (深灰蓝)"""

    platform_color: tuple[float, float, float, float] = (0.28, 0.52, 0.72, 1.0)
    """第一平台颜色 (科技蓝)"""

    second_platform_color: tuple[float, float, float, float] = (0.22, 0.62, 0.65, 1.0)
    """第二平台颜色 (青绿蓝，与第一平台区分)"""

    friction: tuple[float, float, float] = (1.0, 0.005, 0.0001)
    """表面滑动摩擦、扭转摩擦、滚动摩擦"""

    floor_thickness: float = 0.20
    """基准地面平板厚度 (默认 0.20m，顶面标高严格对齐 z = 0.0m)"""

    floor_color: tuple[float, float, float, float] = (0.32, 0.35, 0.38, 1.0)
    """基准地表材质颜色 (深科技灰)"""

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        del rng  # 间距与尺寸确定，随机生成器此处未用

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

        # 1. 确定两平台之间的有效间距 (effective_gap)
        if not self.has_second_platform:
            effective_gap = 0.0
            second_plat_len = 0.0
        elif self.gap_distance is not None:
            effective_gap = float(self.gap_distance)
            second_plat_len = self.second_platform_length
        elif self.vary_gap_with_difficulty:
            # 根据课程难度在 [min_gap, max_gap] 之间线性插值
            diff = float(np.clip(difficulty, 0.0, 1.0))
            effective_gap = self.gap_distance_range[0] + diff * (
                self.gap_distance_range[1] - self.gap_distance_range[0]
            )
            second_plat_len = self.second_platform_length
        else:
            effective_gap = self.gap_distance_range[0]
            second_plat_len = self.second_platform_length

        # 2. 计算各部分沿行进方向的长度与全长
        up_stairs_length = self.num_steps * self.step_width
        down_stairs_length = self.num_steps * self.step_width if self.add_down_stairs else 0.0

        total_travel_length = (
            up_stairs_length
            + self.platform_length
            + (effective_gap + second_plat_len if self.has_second_platform else 0.0)
            + down_stairs_length
        )

        cx = self.size[0] / 2.0 + self.center_offset[0]
        cy = self.size[1] / 2.0 + self.center_offset[1]
        start_travel = -total_travel_length / 2.0

        # 3. 构造所有刚体构件列表: [(name, offset_travel, length_travel, lateral_width, top_z, color)]
        parts = []

        # (a) 上行两阶台阶
        for i in range(self.num_steps):
            step_idx = i + 1
            step_top_z = step_idx * self.step_height
            part_name = f"step_up_{step_idx}"
            travel_start = i * self.step_width
            parts.append(
                (
                    part_name,
                    travel_start,
                    self.step_width,
                    self.step_length,
                    step_top_z,
                    self.step_color,
                )
            )

        # (b) 第一个长方体小平台 (80cm × 80cm × 36cm)
        plat1_travel_start = up_stairs_length
        parts.append(
            (
                "platform_1",
                plat1_travel_start,
                self.platform_length,
                self.platform_width,
                self.platform_height,
                self.platform_color,
            )
        )

        # (c) 第二个长方体小平台 (80cm × 80cm × 36cm)，位于间隔之后
        if self.has_second_platform:
            plat2_travel_start = up_stairs_length + self.platform_length + effective_gap
            parts.append(
                (
                    "platform_2",
                    plat2_travel_start,
                    self.second_platform_length,
                    self.second_platform_width,
                    self.second_platform_height,
                    self.second_platform_color,
                )
            )

        # (d) 可选：下行台阶（位于第二平台之后）
        if self.add_down_stairs:
            down_start = (
                (plat2_travel_start + self.second_platform_length)
                if self.has_second_platform
                else (plat1_travel_start + self.platform_length)
            )
            for i in range(self.num_steps):
                step_idx = self.num_steps - 1 - i
                step_top_z = (step_idx + 1) * self.step_height
                part_name = f"step_down_{i + 1}"
                travel_start = down_start + i * self.step_width
                parts.append(
                    (
                        part_name,
                        travel_start,
                        self.step_width,
                        self.step_length,
                        step_top_z,
                        self.step_color,
                    )
                )

        # 4. 根据 direction 确定空间位置并在 MuJoCo spec 中生成 Geom
        for name, travel_offset, length_t, width_lat, top_z, color in parts:
            half_z = top_z / 2.0
            pos_z = half_z  # 贴合在 z=0 地面

            center_travel = start_travel + travel_offset + length_t / 2.0

            if self.direction == "+x":
                pos_x = cx + center_travel
                pos_y = cy
                size_x = length_t / 2.0
                size_y = width_lat / 2.0
            elif self.direction == "-x":
                pos_x = cx - center_travel
                pos_y = cy
                size_x = length_t / 2.0
                size_y = width_lat / 2.0
            elif self.direction == "+y":
                pos_x = cx
                pos_y = cy + center_travel
                size_x = width_lat / 2.0
                size_y = length_t / 2.0
            elif self.direction == "-y":
                pos_x = cx
                pos_y = cy - center_travel
                size_x = width_lat / 2.0
                size_y = length_t / 2.0
            else:
                raise ValueError(f"Unsupported direction: {self.direction}")

            box_geom = body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=(
                    float(np.maximum(1e-4, size_x)),
                    float(np.maximum(1e-4, size_y)),
                    float(np.maximum(1e-4, half_z)),
                ),
                pos=(float(pos_x), float(pos_y), float(pos_z)),
            )
            box_geom.contype = 1
            box_geom.conaffinity = 1
            for f_idx, f_val in enumerate(self.friction):
                box_geom.friction[f_idx] = f_val

            geometries.append(TerrainGeometry(geom=box_geom, color=color))

        # 5. 计算关键采样点（出生点、第一平台中心、第二平台中心）
        plat1_center_travel = start_travel + plat1_travel_start + self.platform_length / 2.0

        if self.spawn_on_platform:
            # 出生在第一平台顶面中心：一出生就站在平台上，正对间隙与第二平台
            spawn_travel = plat1_center_travel
            spawn_z = float(self.platform_height)
        else:
            # 旧行为：出生在第一阶台阶前方的基准地面上
            spawn_travel = start_travel - self.spawn_distance_before
            spawn_z = 0.0

        if self.has_second_platform:
            plat2_center_travel = (
                start_travel + plat2_travel_start + self.second_platform_length / 2.0
            )
        else:
            plat2_center_travel = plat1_center_travel

        if self.direction == "+x":
            spawn_x = cx + spawn_travel
            spawn_y = cy
            plat1_x = cx + plat1_center_travel
            plat1_y = cy
            plat2_x = cx + plat2_center_travel
            plat2_y = cy
        elif self.direction == "-x":
            spawn_x = cx - spawn_travel
            spawn_y = cy
            plat1_x = cx - plat1_center_travel
            plat1_y = cy
            plat2_x = cx - plat2_center_travel
            plat2_y = cy
        elif self.direction == "+y":
            spawn_x = cx
            spawn_y = cy + spawn_travel
            plat1_x = cx
            plat1_y = cy + plat1_center_travel
            plat2_x = cx
            plat2_y = cy + plat2_center_travel
        elif self.direction == "-y":
            spawn_x = cx
            spawn_y = cy - spawn_travel
            plat1_x = cx
            plat1_y = cy - plat1_center_travel
            plat2_x = cx
            plat2_y = cy - plat2_center_travel

        origin = np.array([float(spawn_x), float(spawn_y), float(spawn_z)])

        flat_patches = {
            "start": np.array([[float(spawn_x), float(spawn_y), float(spawn_z)]]),
            "platform_1": np.array([[float(plat1_x), float(plat1_y), float(self.platform_height)]]),
        }
        if self.has_second_platform:
            flat_patches["platform_2"] = np.array(
                [[float(plat2_x), float(plat2_y), float(self.second_platform_height)]]
            )
            flat_patches["goal"] = flat_patches["platform_2"]
        else:
            flat_patches["goal"] = flat_patches["platform_1"]

        return TerrainOutput(
            origin=origin,
            geometries=geometries,
            flat_patches=flat_patches,
        )


def step_platform_terrain(**overrides: Any) -> StepPlatformTerrainCfg:
    """创建固定两阶台阶 + 第一平台 + 课程间隔 + 第二平台的地形配置构建器。

    默认参数：
    - 台阶：每节高 12cm，步深 32cm，长 80cm，共 2 阶 (顶面 24cm)
    - 第一平台：80cm × 80cm × 36cm (顶面 36cm)
    - 间隔 (Gap)：20cm ~ 60cm 随课程难度自动插值（或指定 gap_distance 固定）
    - 第二平台：80cm × 80cm × 36cm (顶面 36cm)
    - 出生点：第一平台顶面中心 (spawn_on_platform=True)
    """
    defaults: dict[str, Any] = dict(
        step_height=0.12,
        step_width=0.32,
        step_length=0.80,
        num_steps=2,
        platform_length=0.80,
        platform_width=0.80,
        platform_height=0.36,
        has_second_platform=True,
        gap_distance=None,
        gap_distance_range=(0.20, 0.60),
        vary_gap_with_difficulty=True,
        second_platform_length=0.80,
        second_platform_width=0.80,
        second_platform_height=0.36,
        direction="+x",
        add_down_stairs=False,
        spawn_on_platform=True,
        spawn_distance_before=1.5,
    )
    defaults.update(overrides)
    return StepPlatformTerrainCfg(**defaults)


# 预置生成器配置模板（仅供参考与独立调用）
STEP_PLATFORM_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=5,
    num_cols=1,
    curriculum=True,
    sub_terrains={
        "step_platform": step_platform_terrain(
            proportion=1.0,
            gap_distance_range=(0.20, 0.60),
            vary_gap_with_difficulty=True,
        ),
    },
    add_lights=True,
)
