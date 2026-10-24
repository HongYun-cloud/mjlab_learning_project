from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

import mjlab.terrains as terrain_gen
from mjlab.terrains.terrain_generator import (
    SubTerrainCfg,
    TerrainGeometry,
    TerrainGeneratorCfg,
    TerrainOutput,
)
from mjlab.terrains.utils import make_border
from mjlab.utils.color import brand_ramp, darken_rgba

_MUJOCO_DARK = (0.2, 0.2, 0.22)
_TIRE_RUBBER_COLOR = (0.12, 0.12, 0.14, 1.0)


def _quat_z_to_v(v: np.ndarray) -> np.ndarray:
    """Compute quaternion [w, x, y, z] that rotates [0, 0, 1] into vector v."""
    z = np.array([0.0, 0.0, 1.0])
    dot = float(np.dot(z, v))
    if dot >= 0.999999:
        return np.array([1.0, 0.0, 0.0, 0.0])
    elif dot <= -0.999999:
        return np.array([0.0, 1.0, 0.0, 0.0])
    w = np.cross(z, v)
    q = np.array([1.0 + dot, w[0], w[1], w[2]])
    return q / np.linalg.norm(q)


def _generate_torus_mesh(
    r_major: float,
    r_minor: float,
    n_major: int = 16,
    n_minor: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate vertices and triangular faces for a 3D ring/torus mesh.
    
    Args:
        r_major: Distance from center of tube to center of torus.
        r_minor: Radius of the tube (tire thickness / 2).
        n_major: Number of circular segments along the ring.
        n_minor: Number of segments around the tube circumference.
        
    Returns:
        vertices: (N, 3) float array.
        faces: (M, 3) int array of triangular indices (counter-clockwise).
    """
    u = np.linspace(0, 2 * np.pi, n_major, endpoint=False)
    v = np.linspace(0, 2 * np.pi, n_minor, endpoint=False)
    
    # Outer product for grid
    u_grid, v_grid = np.meshgrid(u, v, indexing="ij")
    u_flat = u_grid.flatten()
    v_flat = v_grid.flatten()
    
    # Torus parametric equations (horizontal ring in XY plane, tube extruded along Z)
    x = (r_major + r_minor * np.cos(v_flat)) * np.cos(u_flat)
    y = (r_major + r_minor * np.cos(v_flat)) * np.sin(u_flat)
    z = r_minor * np.sin(v_flat) + r_minor  # 让轮胎底部平贴在 z=0 (地面)
    
    vertices = np.stack([x, y, z], axis=-1)
    
    faces = []
    for i in range(n_major):
        i_next = (i + 1) % n_major
        for j in range(n_minor):
            j_next = (j + 1) % n_minor
            
            p0 = i * n_minor + j
            p1 = i_next * n_minor + j
            p2 = i_next * n_minor + j_next
            p3 = i * n_minor + j_next
            
            # 两个逆时针三角形构成四边形面元 (法向量向外)
            faces.append([p0, p1, p2])
            faces.append([p0, p2, p3])
            
    return vertices, np.array(faces, dtype=np.int32)


@dataclass(kw_only=True)
class TireArrayTerrainCfg(SubTerrainCfg):
    """Configuration for a hollow, round circular tire array (Torus Mesh) terrain.
    
    Each tire is a true circular round ring (Torus) with smooth circular cross-section
    and a hollow inner center, compiled directly into MuJoCo mesh geoms.
    """
    tire_major_radius_range: tuple[float, float] = (0.24, 0.32)
    """Min and max major radius (ring center radius), in meters."""
    
    tire_tube_radius_range: tuple[float, float] = (0.05, 0.09)
    """Min and max tube radius (cross-sectional radius of the tire rubber), in meters."""
    
    n_major_segments: int = 16
    """Number of radial segments around the ring (controls roundness)."""
    
    n_minor_segments: int = 10
    """Number of circular segments along the tube circumference."""
    
    spacing: float = 0.72
    """Nominal center-to-center spacing between adjacent tires in row/col, in meters."""
    
    jitter: float = 0.06
    """Random XY jitter applied to tire center positions, in meters."""
    
    max_tilt_deg: float = 8.0
    """Maximum pitch/roll tilt angle in degrees applied at highest difficulty."""
    
    platform_width: float = 2.0
    """Width of the flat spawn platform along the edge (at Y high), in meters."""
    
    border_width: float = 1.0
    """Flat border frame width around the terrain patch, in meters."""
    
    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        body = spec.body("terrain")
        geometries: list[TerrainGeometry] = []
        
        terrain_height = 1.0
        border_rgba = darken_rgba(brand_ramp(_MUJOCO_DARK, 0.0), 0.85)
        
        # 1. 外部保护平坦边框 (Border Frame)
        if self.border_width > 0.0:
            border_center = (0.5 * self.size[0], 0.5 * self.size[1], -terrain_height / 2)
            border_inner_size = (
                np.maximum(1e-6, self.size[0] - 2 * self.border_width),
                np.maximum(1e-6, self.size[1] - 2 * self.border_width),
            )
            border_boxes = make_border(
                body, self.size, border_inner_size, terrain_height, border_center
            )
            for box in border_boxes:
                geometries.append(TerrainGeometry(geom=box, color=border_rgba))
                
        # 2. 地面底板 (Ground floor - 轮胎内部和轮胎间隙透出的平地)
        floor_geom = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(
                (self.size[0] - 2 * self.border_width) / 2,
                (self.size[1] - 2 * self.border_width) / 2,
                0.05,
            ),
            pos=(self.size[0] / 2, self.size[1] / 2, -0.05),
        )
        geometries.append(TerrainGeometry(geom=floor_geom, color=(0.35, 0.35, 0.35, 1.0)))
        
        # 3. 内部可用有效区域
        x_min = self.border_width + 0.35
        x_max = self.size[0] - self.border_width - 0.35
        y_min = self.border_width + 0.35
        # 预留 Y 方向高端作为出生平坦平台 (Robot Spawn Platform)
        y_max = self.size[1] - self.border_width - self.platform_width
        
        # 4. 根据 difficulty 动态缩放轮胎粗细和倾斜度
        cur_tube_min = self.tire_tube_radius_range[0] + difficulty * 0.01
        cur_tube_max = self.tire_tube_radius_range[0] + difficulty * (
            self.tire_tube_radius_range[1] - self.tire_tube_radius_range[0]
        )
        cur_max_tilt = np.deg2rad(self.max_tilt_deg * difficulty)
        
        # 5. 生成交错蜂窝网格 (Staggered Grid) 的圆环型轮胎 (Torus)
        y_coords = np.arange(y_min, y_max, self.spacing * np.sqrt(3) / 2)
        tire_count = 0
        
        for row_idx, y_center in enumerate(y_coords):
            x_offset = (row_idx % 2) * (self.spacing / 2.0)
            x_coords = np.arange(x_min + x_offset, x_max, self.spacing)
            
            for x_center in x_coords:
                tire_count += 1
                # 随机微小位置扰动
                dx = rng.uniform(-self.jitter, self.jitter)
                dy = rng.uniform(-self.jitter, self.jitter)
                px = np.clip(x_center + dx, x_min, x_max)
                py = np.clip(y_center + dy, y_min, y_max)
                
                r_major = rng.uniform(*self.tire_major_radius_range)
                r_minor = rng.uniform(cur_tube_min, cur_tube_max)
                
                # 生成圆环网格几何 (顶点与三角形面)
                verts, faces = _generate_torus_mesh(
                    r_major=r_major,
                    r_minor=r_minor,
                    n_major=self.n_major_segments,
                    n_minor=self.n_minor_segments,
                )
                
                # 随机微小倾斜 (Pitch / Roll)
                if cur_max_tilt > 1e-4:
                    tilt_pitch = rng.uniform(-cur_max_tilt, cur_max_tilt)
                    tilt_roll = rng.uniform(-cur_max_tilt, cur_max_tilt)
                    
                    R_roll = np.array([
                        [1, 0, 0],
                        [0, np.cos(tilt_roll), -np.sin(tilt_roll)],
                        [0, np.sin(tilt_roll), np.cos(tilt_roll)],
                    ])
                    R_pitch = np.array([
                        [np.cos(tilt_pitch), 0, np.sin(tilt_pitch)],
                        [0, 1, 0],
                        [-np.sin(tilt_pitch), 0, np.cos(tilt_pitch)],
                    ])
                    R_tilt = R_pitch @ R_roll
                    verts = (R_tilt @ verts.T).T
                
                # 将 mesh 注册到 spec
                mesh_name = f"tire_{row_idx}_{tire_count}_{rng.integers(int(1e9))}"
                mesh = spec.add_mesh(
                    name=mesh_name,
                    uservert=verts.flatten().tolist(),
                    userface=faces.flatten().tolist(),
                )
                
                # 颜色随机微调
                color_factor = rng.uniform(0.9, 1.1)
                rgba = (
                    min(1.0, _TIRE_RUBBER_COLOR[0] * color_factor),
                    min(1.0, _TIRE_RUBBER_COLOR[1] * color_factor),
                    min(1.0, _TIRE_RUBBER_COLOR[2] * color_factor),
                    1.0,
                )
                
                # 1. 纯视觉网格：关闭物理碰撞 (contype=0, conaffinity=0)，保留平滑逼真的黑色橡胶圆环外观
                tire_visual_geom = body.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_MESH,
                    meshname=mesh.name,
                    pos=(px, py, 0.0),
                )
                tire_visual_geom.contype = 0
                tire_visual_geom.conaffinity = 0
                tire_visual_geom.group = 2
                geometries.append(TerrainGeometry(geom=tire_visual_geom, color=rgba))
                
                # 2. 物理中空碰撞环：沿圆周均布 12 段胶囊体，无缝首尾相接，中心 100% 物理中空
                n_col_segs = 12
                angles = np.linspace(0, 2 * np.pi, n_col_segs, endpoint=False)
                for s_idx in range(n_col_segs):
                    phi1 = angles[s_idx]
                    phi2 = angles[(s_idx + 1) % n_col_segs]
                    
                    # 局部圆周中心线上的两端点
                    p1_local = np.array([r_major * np.cos(phi1), r_major * np.sin(phi1), r_minor])
                    p2_local = np.array([r_major * np.cos(phi2), r_major * np.sin(phi2), r_minor])
                    
                    # 若存在倾斜，碰撞体与视觉网格同步倾斜
                    if cur_max_tilt > 1e-4:
                        p1_local = R_tilt @ p1_local
                        p2_local = R_tilt @ p2_local
                        
                    p1_world = p1_local + np.array([px, py, 0.0])
                    p2_world = p2_local + np.array([px, py, 0.0])
                    
                    p_mid = 0.5 * (p1_world + p2_world)
                    delta = p2_world - p1_world
                    seg_len = float(np.linalg.norm(delta))
                    half_len = np.maximum(1e-4, seg_len / 2.0)
                    dir_v = delta / seg_len
                    
                    quat = _quat_z_to_v(dir_v)
                    
                    col_geom = body.add_geom(
                        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                        size=(r_minor, half_len),
                        pos=tuple(p_mid),
                    )
                    col_geom.quat = quat
                    col_geom.contype = 1
                    col_geom.conaffinity = 1
                    col_geom.group = 3
                    for f_idx, f_val in enumerate((1.2, 0.005, 0.0001)):
                        col_geom.friction[f_idx] = f_val
                        
                    geometries.append(TerrainGeometry(geom=col_geom, color=rgba))
        
        origin = np.array([self.size[0] / 2.0, self.size[1] / 2.0, 0.0])
        return TerrainOutput(origin=origin, geometries=geometries)


def tire_array_terrain(**overrides: Any) -> TireArrayTerrainCfg:
    defaults: dict[str, Any] = dict(
        tire_major_radius_range=(0.24, 0.32),
        tire_tube_radius_range=(0.06, 0.09),
        n_major_segments=16,
        n_minor_segments=10,
        spacing=0.70,
        jitter=0.06,
        max_tilt_deg=8.0,
        platform_width=2.5,
        border_width=1.0,
    )
    defaults.update(overrides)
    return TireArrayTerrainCfg(**defaults)


TIRE_ARRAY_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=1,
    curriculum=True,
    sub_terrains={
        "tire_array": tire_array_terrain(
            proportion=1.0,
            tire_tube_radius_range=(0.01, 0.2),
        ),
    },
    add_lights=True,
)