"""Independent CPU Robot raster extracted from the selected V3 implementation.

Only maintained pipeline geometry modules and explicitly supplied assets are
used. No historical operation imports, session constants or experiment paths.
Origin SHA and function extraction are recorded in the migration receipt.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from numba import njit

from chaoyang.pipeline.robot_renderer_cycles import forward_kinematics
from chaoyang.pipeline.robot_renderer_eevee_fullchain import ARM_JOINT_NAMES

NEAR_M = 0.05
SIDES = ("left", "right")
ROBOT_IVORY = (205, 225, 235)
HAND_WHITE = (235, 242, 245)
FLANGE_IVORY = (195, 218, 232)
M6_ARM_LINKS = frozenset(
    f"{prefix}_{side}"
    for side in ("L", "R")
    for prefix in ("Base", *(f"Link{i}" for i in range(1, 8)))
)
NATURALV2_CENTER_XY_MM = np.asarray((51.62780570983887, 51.30149126052856))
NATURALV2_MATING_Z_MM = 1.612299

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def evidence(path: Path) -> dict:
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    return {"path": str(resolved), "bytes": stat.st_size, "sha256": sha256(resolved)}

def transform(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    return values @ matrix[:3, :3].T + matrix[:3, 3]

def clip_triangle_near(triangle: np.ndarray, near: float) -> list[np.ndarray]:
    """Sutherland-Hodgman clip of one camera-space triangle against z>=near."""
    polygon = [np.asarray(point, dtype=np.float64) for point in triangle]
    output: list[np.ndarray] = []
    for first, second in zip(polygon, polygon[1:] + polygon[:1], strict=True):
        first_inside, second_inside = first[2] >= near, second[2] >= near
        if first_inside:
            output.append(first)
        if first_inside != second_inside:
            fraction = (near - first[2]) / (second[2] - first[2])
            output.append(first + fraction * (second - first))
    if len(output) < 3:
        return []
    return [np.asarray((output[0], output[index], output[index + 1])) for index in range(1, len(output) - 1)]

def base_color(link: str) -> np.ndarray:
    # BGR review colours.  Finger identity remains visible even under shading.
    if "thumb" in link:
        return np.asarray((65, 85, 245), dtype=np.float64)
    if "index" in link:
        return np.asarray((40, 220, 245), dtype=np.float64)
    if "middle" in link:
        return np.asarray((70, 225, 80), dtype=np.float64)
    if "ring" in link:
        return np.asarray((220, 75, 225), dtype=np.float64)
    if "pinky" in link:
        return np.asarray((240, 150, 55), dtype=np.float64)
    if "base_link" in link:
        return np.asarray((190, 190, 190), dtype=np.float64)
    return np.asarray((235, 145, 45), dtype=np.float64)

def camera_triangles(model, fk: dict[str, np.ndarray], root: np.ndarray, allowed,
                     mesh_cache: dict[Path, tuple[np.ndarray, np.ndarray, dict]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    triangles_out: list[np.ndarray] = []
    colors_out: list[np.ndarray] = []
    labels_out: list[np.ndarray] = []
    mesh_rows = {}
    light = np.asarray((-0.35, -0.55, -0.76), dtype=np.float64)
    light /= np.linalg.norm(light)
    for label, visual in enumerate(model.visuals):
        if not allowed(visual.link):
            continue
        if visual.mesh_path not in mesh_cache:
            loaded = trimesh.load_mesh(visual.mesh_path, process=False, force="mesh")
            mesh_cache[visual.mesh_path] = (
                np.asarray(loaded.vertices, dtype=np.float64),
                np.asarray(loaded.faces, dtype=np.int64),
                evidence(visual.mesh_path),
            )
        vertices, faces, mesh_evidence = mesh_cache[visual.mesh_path]
        world = transform(root @ fk[visual.link] @ visual.origin, vertices)
        raw = world[faces]
        wholly_inside = np.all(raw[:, :, 2] >= NEAR_M, axis=1)
        wholly_outside = np.all(raw[:, :, 2] < NEAR_M, axis=1)
        mixed = ~(wholly_inside | wholly_outside)
        pieces = [raw[wholly_inside]] if np.any(wholly_inside) else []
        for triangle in raw[mixed]:
            pieces.extend(clip_triangle_near(triangle, NEAR_M))
        near_clipped = int(np.count_nonzero(~wholly_inside))
        if not pieces:
            mesh_rows[visual.link] = {"source_triangles": int(len(raw)), "raster_triangles": 0, "near_plane_affected": near_clipped}
            continue
        triangles = np.concatenate([np.asarray(piece, dtype=np.float64).reshape((-1, 3, 3)) for piece in pieces], axis=0)
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        lengths = np.linalg.norm(normals, axis=1)
        valid = lengths > 1e-12
        triangles = triangles[valid]
        normals = normals[valid] / lengths[valid, None]
        shade = 0.42 + 0.58 * np.abs(normals @ light)
        colors = np.clip(base_color(visual.link)[None] * shade[:, None], 0, 255).astype(np.uint8)
        triangles_out.append(triangles)
        colors_out.append(colors)
        labels_out.append(np.full(len(triangles), label, dtype=np.int32))
        mesh_rows[visual.link] = {
            "mesh": mesh_evidence,
            "source_triangles": int(len(raw)),
            "raster_triangles": int(len(triangles)),
            "near_plane_affected": int(near_clipped),
        }
    if not triangles_out:
        return np.empty((0, 3, 3)), np.empty((0, 3), np.uint8), np.empty(0, np.int32), mesh_rows
    return np.concatenate(triangles_out), np.concatenate(colors_out), np.concatenate(labels_out), mesh_rows

@njit(cache=True)
def rasterize_zbuffer(triangles: np.ndarray, colors: np.ndarray, labels: np.ndarray,
                      fx: float, fy: float, cx: float, cy: float,
                      width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    depth = np.full((height, width), np.inf, dtype=np.float64)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    link_id = np.full((height, width), -1, dtype=np.int32)
    for index in range(triangles.shape[0]):
        x0 = fx * triangles[index, 0, 0] / triangles[index, 0, 2] + cx
        y0 = fy * triangles[index, 0, 1] / triangles[index, 0, 2] + cy
        x1 = fx * triangles[index, 1, 0] / triangles[index, 1, 2] + cx
        y1 = fy * triangles[index, 1, 1] / triangles[index, 1, 2] + cy
        x2 = fx * triangles[index, 2, 0] / triangles[index, 2, 2] + cx
        y2 = fy * triangles[index, 2, 1] / triangles[index, 2, 2] + cy
        denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denominator) < 1e-12:
            continue
        xmin = max(0, int(np.floor(min(x0, x1, x2))))
        xmax = min(width - 1, int(np.ceil(max(x0, x1, x2))))
        ymin = max(0, int(np.floor(min(y0, y1, y2))))
        ymax = min(height - 1, int(np.ceil(max(y0, y1, y2))))
        if xmin > xmax or ymin > ymax:
            continue
        invz0 = 1.0 / triangles[index, 0, 2]
        invz1 = 1.0 / triangles[index, 1, 2]
        invz2 = 1.0 / triangles[index, 2, 2]
        for y in range(ymin, ymax + 1):
            py = y + 0.5
            for x in range(xmin, xmax + 1):
                px = x + 0.5
                a = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / denominator
                b = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / denominator
                c = 1.0 - a - b
                if a < -1e-9 or b < -1e-9 or c < -1e-9:
                    continue
                inverse_depth = a * invz0 + b * invz1 + c * invz2
                if inverse_depth <= 0.0:
                    continue
                z = 1.0 / inverse_depth
                if z < depth[y, x]:
                    depth[y, x] = z
                    image[y, x, 0] = colors[index, 0]
                    image[y, x, 1] = colors[index, 1]
                    image[y, x, 2] = colors[index, 2]
                    link_id[y, x] = labels[index]
    return depth, image, link_id

def naturalv2_local_triangles(path: Path) -> np.ndarray:
    mesh = trimesh.load_mesh(path, process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64).copy()
    vertices[:, :2] -= NATURALV2_CENTER_XY_MM
    vertices[:, 2] -= NATURALV2_MATING_Z_MM
    vertices *= 0.001
    return vertices[np.asarray(mesh.faces, dtype=np.int64)]

def transform_triangles(triangles: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return triangles @ transform[:3, :3].T + transform[:3, 3]

def render_robot(
    raster: Any,
    assets: Any,
    q_arm: np.ndarray,
    q_hand: np.ndarray,
    mounts: np.ndarray,
    camera: np.ndarray,
    intrinsics: np.ndarray,
    cache: dict[Path, object],
    flange_local: np.ndarray,
    *,
    complete_robot: bool,
    return_buffers: bool = False,
    width: int = 640,
    height: int = 480,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | dict[str, object]]:
    def reference_tint(
        raw_colors: np.ndarray,
        labels: np.ndarray,
        visuals: list[Any],
        target: tuple[int, int, int],
    ) -> np.ndarray:
        """Apply one Robot004 material while preserving per-triangle lighting."""
        result = np.zeros_like(raw_colors)
        for label in np.unique(labels):
            mask = labels == label
            link = visuals[int(label)].link
            source_base = np.asarray(raster.base_color(link), dtype=np.float64)
            shade = np.clip(
                np.mean(raw_colors[mask].astype(np.float64), axis=1)
                / max(float(np.mean(source_base)), 1e-9),
                0.42,
                1.0,
            )
            result[mask] = np.clip(
                np.asarray(target, dtype=np.float64)[None] * shade[:, None], 0, 255
            ).astype(np.uint8)
        return result

    def shaded_untextured_mesh(
        triangles: np.ndarray, target: tuple[int, int, int], label: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        normals = np.cross(
            triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
        )
        lengths = np.linalg.norm(normals, axis=1)
        valid = lengths > 1e-12
        triangles = triangles[valid]
        normals = normals[valid] / lengths[valid, None]
        light = np.asarray((-0.35, -0.55, -0.76), dtype=np.float64)
        light /= np.linalg.norm(light)
        shade = 0.42 + 0.58 * np.abs(normals @ light)
        colors = np.clip(
            np.asarray(target, dtype=np.float64)[None] * shade[:, None], 0, 255
        ).astype(np.uint8)
        return triangles, colors, np.full(len(triangles), label, dtype=np.int32)

    values = {name: 0.0 for row in ARM_JOINT_NAMES for name in row}
    for physical in range(2):
        values.update(dict(zip(ARM_JOINT_NAMES[physical], q_arm[physical], strict=True)))
    arm_fk = forward_kinematics(assets.tianji, values)
    selector = (lambda _link: True) if complete_robot else (lambda link: link in M6_ARM_LINKS)
    robot = raster.camera_triangles(assets.tianji, arm_fk, camera, selector, cache)
    robot_colors = reference_tint(
        robot[1], robot[2], assets.tianji.visuals, ROBOT_IVORY
    )
    geometry = [(robot[0], robot_colors, robot[2])]
    roots = []
    for physical, model in enumerate((assets.left_hand, assets.right_hand)):
        side = SIDES[physical]
        root = camera @ arm_fk[f"{side}_tool"] @ mounts[physical]
        roots.append(root)
        names = tuple(j.name for j in model.joints if j.joint_type != 'fixed')
        hand_fk = forward_kinematics(
            model, dict(zip(names, q_hand[physical], strict=True))
        )
        hand = raster.camera_triangles(model, hand_fk, root, lambda _link: True, cache)
        geometry.append(
            (
                hand[0],
                reference_tint(hand[1], hand[2], model.visuals, HAND_WHITE),
                hand[2] + 2000 + physical * 200,
            )
        )
        flange = camera @ arm_fk[f"flange_{'L' if physical == 0 else 'R'}"]
        geometry.append(
            shaded_untextured_mesh(
                transform_triangles(flange_local, flange),
                FLANGE_IVORY,
                4000 + physical,
            )
        )
    triangles = np.concatenate([item[0] for item in geometry if len(item[0])])
    colors = np.concatenate([item[1] for item in geometry if len(item[0])])
    labels = np.concatenate([item[2] for item in geometry if len(item[0])])
    depth, color, label = raster.rasterize_zbuffer(
        triangles,
        colors,
        labels,
        float(intrinsics[0, 0]),
        float(intrinsics[1, 1]),
        float(intrinsics[0, 2]),
        float(intrinsics[1, 2]),
        width,
        height,
    )
    if return_buffers:
        # A second pass with a unique label per input triangle provides actual
        # triangle provenance.  It uses the identical triangle order and
        # projection, so it cannot silently substitute visual/link labels for
        # triangle IDs.
        _, _, triangle_id = raster.rasterize_zbuffer(
            triangles,
            colors,
            np.arange(len(triangles), dtype=np.int32),
            float(intrinsics[0, 0]),
            float(intrinsics[1, 1]),
            float(intrinsics[0, 2]),
            float(intrinsics[1, 2]),
            width,
            height,
        )

        def label_identity(value: int) -> dict[str, str]:
            if 0 <= value < len(assets.tianji.visuals):
                link_name = str(assets.tianji.visuals[value].link)
                side = "left" if link_name.endswith("_L") else ("right" if link_name.endswith("_R") else "shared")
                return {"instance_id": f"{side}_tianji", "part_id": "arm", "link_id": link_name}
            for physical, model in enumerate((assets.left_hand, assets.right_hand)):
                offset = 2000 + physical * 200
                local = value - offset
                if 0 <= local < len(model.visuals):
                    link_name = str(model.visuals[local].link)
                    part = "palm" if link_name.endswith("base_link") else "finger"
                    return {"instance_id": f"{SIDES[physical]}_kaihand", "part_id": part, "link_id": link_name}
            if value in {4000, 4001}:
                side = SIDES[value - 4000]
                return {"instance_id": f"{side}_flange", "part_id": "flange", "link_id": f"flange_{side}"}
            raise ValueError(f"unmapped rendered label {value}")

        identity = {int(value): label_identity(int(value)) for value in np.unique(label) if int(value) >= 0}
        return color, label, {
            "hand_roots": np.asarray(roots),
            "depth_m": np.asarray(depth, dtype=np.float32),
            "triangle_id": np.asarray(triangle_id, dtype=np.int64),
            "label_identity": identity,
            "one_scene_zbuffer": True,
        }
    return color, label, np.asarray(roots)

