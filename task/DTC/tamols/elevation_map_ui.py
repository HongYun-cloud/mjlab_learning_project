"""Standalone 3D Viser view of the live elevation map and TAMOLS footholds."""

from __future__ import annotations

from dataclasses import dataclass
import os
import threading
import time
from typing import TYPE_CHECKING, Any
import webbrowser

import numpy as np
from scipy import ndimage

if TYPE_CHECKING:
    import mujoco
    import viser


LEG_NAMES = ("FL", "FR", "RL", "RR")
FOOTHOLD_COLORS = {
    "optimal": (245, 55, 55),
    "landing": (255, 160, 20),
    "nominal": (55, 150, 255),
}
FOOTHOLD_LABELS = {
    "optimal": "TAMOLS 优化落点",
    "landing": "摆动腿当前目标",
    "nominal": "名义落点",
}


@dataclass(frozen=True)
class ElevationMapSnapshot:
    """3D surface mesh and foothold positions in the robot-yaw frame."""

    vertices: np.ndarray
    faces: np.ndarray
    footholds: dict[str, np.ndarray]
    observed_cells: int
    total_cells: int
    min_height: float | None
    max_height: float | None
    cell_size: float
    x_range: tuple[float, float]
    y_range: tuple[float, float]


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _quat_multiply_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def _downsample_map(
    heights: np.ndarray,
    validity: np.ndarray,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    nx, ny = heights.shape
    out_nx = (nx + stride - 1) // stride
    out_ny = (ny + stride - 1) // stride
    down_h = np.zeros((out_nx, out_ny), dtype=np.float32)
    observed = np.zeros((out_nx, out_ny), dtype=bool)
    for i in range(out_nx):
        for j in range(out_ny):
            i0, i1 = i * stride, min((i + 1) * stride, nx)
            j0, j1 = j * stride, min((j + 1) * stride, ny)
            valid = validity[i0:i1, j0:j1] > 0.5
            if valid.any():
                observed[i, j] = True
                block = heights[i0:i1, j0:j1]
                down_h[i, j] = float(block[valid].mean())
    return down_h, observed


def _make_surface_mesh(
    heights: np.ndarray,
    observed: np.ndarray,
    *,
    x_min: float,
    y_min: float,
    cell_size: float,
    stride: int,
    ground: float,
) -> tuple[np.ndarray, np.ndarray]:
    nx, ny = heights.shape
    if not observed.any():
        return np.zeros((nx * ny, 3), np.float32), np.zeros(0, np.uint32)

    # Fill only one-cell sensor gaps for a connected-looking surface. Regions
    # beyond that remain holes, rather than being shown as invented terrain.
    distance, nearest = ndimage.distance_transform_edt(
        ~observed, return_distances=True, return_indices=True
    )
    filled = heights.copy()
    filled[~observed] = heights[nearest[0][~observed], nearest[1][~observed]]
    surface_mask = observed | (distance <= 1.01)

    x_starts = np.arange(nx, dtype=np.int32) * stride
    y_starts = np.arange(ny, dtype=np.int32) * stride
    x_centers = x_starts + 0.5 * (np.minimum(stride, heights.shape[0] - x_starts) - 1)
    y_centers = y_starts + 0.5 * (np.minimum(stride, heights.shape[1] - y_starts) - 1)
    x = x_min + x_centers.astype(np.float32) * cell_size
    y = y_min + y_centers.astype(np.float32) * cell_size
    xx, yy = np.meshgrid(x, y, indexing="ij")
    vertices = np.stack((xx, yy, filled - ground), axis=-1).reshape(-1, 3)

    faces: list[int] = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            if not surface_mask[i : i + 2, j : j + 2].all():
                continue
            p00 = i * ny + j
            p10 = (i + 1) * ny + j
            p01 = i * ny + j + 1
            p11 = (i + 1) * ny + j + 1
            # Counter-clockwise when viewed from above (+Z).
            faces.extend((p00, p10, p01, p10, p11, p01))
    return vertices.astype(np.float32), np.asarray(faces, dtype=np.uint32)


def _project_footholds(cmd: Any, env_id: int, ground: float) -> dict[str, np.ndarray]:
    if hasattr(cmd, "footholds_initialized") and not bool(
        _numpy(cmd.footholds_initialized[env_id])
    ):
        return {}

    base_xy = _numpy(cmd.robot.data.root_link_pos_w[env_id])[:2]
    yaw = float(_numpy(cmd.robot.data.heading_w[env_id]))
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    projected: dict[str, np.ndarray] = {}
    for category, attr in (
        ("optimal", "optimal_footholds"),
        ("landing", "landing_footholds"),
        ("nominal", "nominal_footholds"),
    ):
        if not hasattr(cmd, attr):
            continue
        world = _numpy(getattr(cmd, attr)[env_id]).astype(np.float32, copy=False)
        rel_x = world[:, 0] - base_xy[0]
        rel_y = world[:, 1] - base_xy[1]
        body_x = cos_yaw * rel_x + sin_yaw * rel_y
        body_y = -sin_yaw * rel_x + cos_yaw * rel_y
        # Raise the marker above the mesh so it remains visible in the scene.
        local_z = world[:, 2] - ground + 0.045
        projected[category] = np.stack((body_x, body_y, local_z), axis=-1).astype(
            np.float32
        )
    return projected


def compile_standalone_robot_model(robot: Any) -> Any:
    """Compile a robot-only MjModel outside the already-attached scene spec.

    ``robot.spec`` is attached by reference to the simulated scene after env
    construction and MuJoCo refuses to compile child specs in that state. The
    UI therefore rebuilds the entity from its original ``spec_fn``.
    """
    spec_fn = getattr(getattr(robot, "cfg", None), "spec_fn", None)
    if spec_fn is not None:
        try:
            return spec_fn().compile()
        except Exception:
            pass
    return robot.spec.copy().compile()


def build_elevation_map_snapshot(
    cmd: Any,
    env_id: int = 0,
    stride: int = 1,
) -> ElevationMapSnapshot:
    """Build a triangulated local height surface and body-frame footholds."""
    stride = max(1, int(stride))
    heights = _numpy(cmd.h_raw[env_id, 0]).astype(np.float32, copy=False)
    validity = _numpy(cmd.h_valid[env_id, 0]).astype(np.float32, copy=False)
    ground = float(_numpy(cmd.ground_default[env_id]))
    cell_size = float(cmd.cell_size)
    x_min, x_max = float(cmd.x_min), float(cmd.x_max)
    y_min, y_max = float(cmd.y_min), float(cmd.y_max)

    down_h, observed = _downsample_map(heights, validity, stride)
    vertices, faces = _make_surface_mesh(
        down_h,
        observed,
        x_min=x_min,
        y_min=y_min,
        cell_size=cell_size,
        stride=stride,
        ground=ground,
    )
    measured = heights[validity > 0.5]
    return ElevationMapSnapshot(
        vertices=vertices,
        faces=faces,
        footholds=_project_footholds(cmd, env_id, ground),
        observed_cells=int(np.count_nonzero(validity > 0.5)),
        total_cells=int(heights.size),
        min_height=float(measured.min()) if measured.size else None,
        max_height=float(measured.max()) if measured.size else None,
        cell_size=cell_size,
        x_range=(x_min, x_max),
        y_range=(y_min, y_max),
    )


class ElevationMapUI:
    """A separate 3D viewer showing the terrain mesh, robot, and footholds."""

    def __init__(
        self,
        *,
        robot: Any,
        port: int = 8081,
        stride: int = 1,
        update_interval: float = 0.1,
        show_nominal: bool = True,
    ):
        import mujoco
        import viser
        from mjviser import ViserMujocoScene

        self.stride = max(1, int(stride))
        self.update_interval = max(0.05, float(update_interval))
        self.show_nominal = show_nominal
        try:
            self.server = viser.ViserServer(
                host="127.0.0.1",
                port=int(port),
                label="TAMOLS 3D Elevation Map",
                verbose=False,
            )
        except OSError:
            self.server = viser.ViserServer(
                host="127.0.0.1",
                port=0,
                label="TAMOLS 3D Elevation Map",
                verbose=False,
            )

        self.server.scene.set_up_direction("+z")
        self.server.initial_camera.position = (0.15, -2.0, 1.15)
        self.server.initial_camera.look_at = (0.15, 0.0, 0.08)
        self.server.initial_camera.up = (0.0, 0.0, 1.0)
        self.server.initial_camera.fov = np.deg2rad(52.0)

        # Build a robot-only MuJoCo scene in this separate viewer; its terrain
        # is supplied by the sensed elevation mesh below, not by the sim world.
        self.robot_model = compile_standalone_robot_model(robot)
        self.robot_data = mujoco.MjData(self.robot_model)
        self.robot_entity = robot
        self.robot_scene = ViserMujocoScene(self.server, self.robot_model, 1)
        self.robot_scene.camera_tracking_enabled = False

        placeholder_vertices = np.zeros((3, 3), dtype=np.float32)
        placeholder_faces = np.array([0, 1, 2], dtype=np.uint32)
        self.surface_mesh = self.server.scene.add_mesh_simple(
            "elevation/surface",
            vertices=placeholder_vertices,
            faces=placeholder_faces,
            color=(170, 180, 228),
            opacity=0.86,
            flat_shading=True,
            side="double",
            visible=False,
        )
        self.wire_mesh = self.server.scene.add_mesh_simple(
            "elevation/wireframe",
            vertices=placeholder_vertices.copy(),
            faces=placeholder_faces.copy(),
            color=(54, 64, 105),
            wireframe=True,
            side="double",
            visible=False,
        )

        self.foothold_markers = {
            (category, leg): self.server.scene.add_icosphere(
                f"footholds/{category}/{leg}",
                radius=0.035 if category != "nominal" else 0.027,
                color=FOOTHOLD_COLORS[category],
                subdivisions=2,
                visible=False,
            )
            for category in ("optimal", "landing", "nominal")
            for leg in LEG_NAMES
        }
        self.status = self.server.gui.add_html(
            "<b>3D 高程图 / TAMOLS 落点</b><br>等待仿真数据…"
        )
        self._last_update = -float("inf")

        actual_port = self.server.get_port()
        url = f"http://127.0.0.1:{actual_port}"
        print(f"[ElevationMapUI] 独立 3D 高程图窗口已启动: {url}")
        if os.getenv("TAMOLS_MAP_UI_OPEN_BROWSER", "1") != "0":
            # Best effort: desktop environments open this in the default browser.
            # Remote/headless sessions still get the printed URL.
            timer = threading.Timer(1.0, self._open_browser, args=(url,))
            timer.daemon = True
            timer.start()

    @staticmethod
    def _open_browser(url: str) -> None:
        try:
            webbrowser.open(url, new=2)
        except Exception:
            pass

    def _update_robot_pose(self, cmd: Any, env_id: int, ground: float) -> None:
        import mujoco

        root_pos = _numpy(cmd.robot.data.root_link_pos_w[env_id])
        root_quat = _numpy(cmd.robot.data.root_link_quat_w[env_id]).astype(
            np.float64
        )
        yaw = float(_numpy(cmd.robot.data.heading_w[env_id]))
        yaw_half = 0.5 * yaw
        inverse_yaw = np.array(
            [np.cos(yaw_half), 0.0, 0.0, -np.sin(yaw_half)], dtype=np.float64
        )
        local_quat = _quat_multiply_wxyz(inverse_yaw, root_quat)

        free_ids = np.flatnonzero(
            self.robot_model.jnt_type == mujoco.mjtJoint.mjJNT_FREE
        )
        if free_ids.size:
            qadr = int(self.robot_model.jnt_qposadr[free_ids[0]])
            self.robot_data.qpos[qadr : qadr + 3] = (0.0, 0.0, root_pos[2] - ground)
            self.robot_data.qpos[qadr + 3 : qadr + 7] = local_quat

        joint_values = _numpy(cmd.robot.data.joint_pos[env_id])
        for name, value in zip(cmd.robot.joint_names, joint_values):
            jid = mujoco.mj_name2id(
                self.robot_model, mujoco.mjtObj.mjOBJ_JOINT, name
            )
            if jid < 0 or self.robot_model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            qadr = int(self.robot_model.jnt_qposadr[jid])
            self.robot_data.qpos[qadr] = value

        mujoco.mj_forward(self.robot_model, self.robot_data)
        self.robot_scene.update_from_mjdata(self.robot_data)

    def update(self, cmd: Any, env_id: int = 0, *, force: bool = False) -> None:
        """Refresh the 3D terrain mesh, robot pose, and landing markers."""
        now = time.monotonic()
        if not force and now - self._last_update < self.update_interval:
            return
        self._last_update = now

        snapshot = build_elevation_map_snapshot(cmd, env_id, self.stride)
        self.surface_mesh.vertices = snapshot.vertices
        self.surface_mesh.faces = snapshot.faces
        self.surface_mesh.visible = bool(len(snapshot.faces))
        # Offset the line mesh slightly to avoid z-fighting with the filled surface.
        wire_vertices = snapshot.vertices.copy()
        wire_vertices[:, 2] += 0.001
        self.wire_mesh.vertices = wire_vertices
        self.wire_mesh.faces = snapshot.faces
        self.wire_mesh.visible = bool(len(snapshot.faces))

        ground = float(_numpy(cmd.ground_default[env_id]))
        self._update_robot_pose(cmd, env_id, ground)

        for (category, leg), marker in self.foothold_markers.items():
            points = snapshot.footholds.get(category)
            if category == "nominal" and not self.show_nominal:
                points = None
            if points is None:
                marker.visible = False
                continue
            point = points[LEG_NAMES.index(leg)]
            marker.position = tuple(float(v) for v in point)
            marker.visible = bool(np.isfinite(point).all())

        height_range = (
            "暂无有效观测"
            if snapshot.min_height is None
            else f"{snapshot.min_height:.3f} ~ {snapshot.max_height:.3f} m"
        )
        coverage = 100.0 * snapshot.observed_cells / max(snapshot.total_cells, 1)
        lines = [
            "<b>3D 高程图 / TAMOLS 落点</b>",
            '<span style="color:#f53737">●</span> 优化落点　'
            '<span style="color:#ffa014">●</span> 当前摆动目标　'
            '<span style="color:#3796ff">●</span> 名义落点',
            f"观测格: {snapshot.observed_cells}/{snapshot.total_cells} "
            f"({coverage:.1f}%)　实测高度: {height_range}",
            f"表面三角面: {len(snapshot.faces) // 3}",
            f"网格分辨率: {snapshot.cell_size:.2f} m　"
            f"X: {snapshot.x_range[0]:.2f}~{snapshot.x_range[1]:.2f} m　"
            f"Y: {snapshot.y_range[0]:.2f}~{snapshot.y_range[1]:.2f} m",
            "坐标系随机器人航向对齐：+X 朝前、+Y 朝左。",
        ]
        for category, title in FOOTHOLD_LABELS.items():
            if category == "nominal" and not self.show_nominal:
                continue
            points = snapshot.footholds.get(category)
            if points is None:
                continue
            color = "#%02x%02x%02x" % FOOTHOLD_COLORS[category]
            coords = "　".join(
                f"{leg} ({pt[0]:.2f}, {pt[1]:.2f}, {pt[2]:.2f})"
                for leg, pt in zip(LEG_NAMES, points)
            )
            lines.append(f'<span style="color:{color}">●</span> {title}: {coords}')
        self.status.content = "<br>".join(lines)
