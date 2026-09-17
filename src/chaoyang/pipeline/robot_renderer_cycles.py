#!/usr/bin/env python3
"""Pinned Cycles robot-buffer renderer with fail-closed R2 hand-only smoke.

The reusable core consumes explicit URDF joint states and root transforms.  It
does not solve arm placement and never invents q_arm or a renderer-only offset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


CV_CAMERA_TO_BLENDER = np.diag([1.0, -1.0, -1.0, 1.0])


class RendererError(RuntimeError):
    pass


@dataclass(frozen=True)
class VisualSpec:
    link: str
    mesh_path: Path
    origin: np.ndarray
    rgba: tuple[float, float, float, float]


@dataclass(frozen=True)
class JointSpec:
    name: str
    joint_type: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray
    lower: float | None
    upper: float | None


@dataclass(frozen=True)
class UrdfModel:
    path: Path
    root_link: str
    links: tuple[str, ...]
    joints: tuple[JointSpec, ...]
    visuals: tuple[VisualSpec, ...]


@dataclass(frozen=True)
class HandState:
    side: str
    q_by_joint: Mapping[str, float]
    root_in_cv_camera: np.ndarray
    urdf: UrdfModel


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _vector(text: str | None, length: int, default: Sequence[float]) -> np.ndarray:
    if text is None:
        return np.asarray(default, dtype=np.float64)
    values = np.asarray([float(value) for value in text.split()], dtype=np.float64)
    if values.shape != (length,) or not np.isfinite(values).all():
        raise RendererError(f"invalid vector {text!r}")
    return values


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def origin_matrix(element: ET.Element | None) -> np.ndarray:
    xyz = _vector(element.get("xyz") if element is not None else None, 3, (0, 0, 0))
    rpy = _vector(element.get("rpy") if element is not None else None, 3, (0, 0, 0))
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _rpy_matrix(rpy)
    transform[:3, 3] = xyz
    return transform


def axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    norm = float(np.linalg.norm(axis))
    if norm <= 0 or not np.isfinite(norm):
        raise RendererError("joint axis is degenerate")
    x, y, z = axis / norm
    c, s, one_c = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    rotation = np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float64,
    )
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    return result


def parse_urdf(path: Path) -> UrdfModel:
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as exc:
        raise RendererError(f"cannot parse URDF {path}: {exc}") from exc
    links = tuple(element.get("name", "") for element in root.findall("link"))
    if not links or any(not name for name in links) or len(set(links)) != len(links):
        raise RendererError(f"invalid or duplicate links in {path}")

    visuals: list[VisualSpec] = []
    for link_element in root.findall("link"):
        link_name = link_element.get("name", "")
        for visual in link_element.findall("visual"):
            mesh = visual.find("geometry/mesh")
            if mesh is None or not mesh.get("filename"):
                raise RendererError(f"{path}: {link_name} has unsupported visual geometry")
            filename = str(mesh.get("filename"))
            if filename.startswith("package://"):
                raise RendererError(f"package URI requires an explicit resolver: {filename}")
            mesh_path = (path.parent / filename).resolve()
            if not mesh_path.is_file():
                raise RendererError(f"missing visual mesh: {mesh_path}")
            scale = _vector(mesh.get("scale"), 3, (1, 1, 1))
            visual_origin = origin_matrix(visual.find("origin"))
            visual_origin[:3, :3] = visual_origin[:3, :3] @ np.diag(scale)
            color = visual.find("material/color")
            rgba_values = _vector(
                color.get("rgba") if color is not None else None,
                4,
                (0.7, 0.72, 0.76, 1.0),
            )
            visuals.append(
                VisualSpec(
                    link=link_name,
                    mesh_path=mesh_path,
                    origin=visual_origin,
                    rgba=tuple(float(value) for value in rgba_values),
                )
            )

    joints: list[JointSpec] = []
    children: set[str] = set()
    for element in root.findall("joint"):
        name = element.get("name", "")
        joint_type = element.get("type", "")
        parent = element.find("parent")
        child = element.find("child")
        if not name or parent is None or child is None:
            raise RendererError(f"malformed joint in {path}")
        parent_name, child_name = parent.get("link", ""), child.get("link", "")
        if parent_name not in links or child_name not in links or child_name in children:
            raise RendererError(f"invalid joint topology for {name}")
        children.add(child_name)
        axis = _vector(
            element.find("axis").get("xyz") if element.find("axis") is not None else None,
            3,
            (1, 0, 0),
        )
        limit = element.find("limit")
        lower = float(limit.get("lower")) if limit is not None and limit.get("lower") else None
        upper = float(limit.get("upper")) if limit is not None and limit.get("upper") else None
        if joint_type not in {"fixed", "revolute", "continuous", "prismatic"}:
            raise RendererError(f"unsupported joint type {joint_type!r}: {name}")
        joints.append(
            JointSpec(
                name=name,
                joint_type=joint_type,
                parent=parent_name,
                child=child_name,
                origin=origin_matrix(element.find("origin")),
                axis=axis,
                lower=lower,
                upper=upper,
            )
        )
    root_links = sorted(set(links) - children)
    if len(root_links) != 1:
        raise RendererError(f"URDF must have exactly one root link: {root_links}")
    return UrdfModel(path, root_links[0], links, tuple(joints), tuple(visuals))


def forward_kinematics(model: UrdfModel, q_by_joint: Mapping[str, float]) -> dict[str, np.ndarray]:
    actuated = {joint.name for joint in model.joints if joint.joint_type != "fixed"}
    if set(q_by_joint) != actuated:
        missing = sorted(actuated - set(q_by_joint))
        extra = sorted(set(q_by_joint) - actuated)
        raise RendererError(f"joint state identity mismatch; missing={missing}, extra={extra}")
    transforms = {model.root_link: np.eye(4, dtype=np.float64)}
    pending = list(model.joints)
    while pending:
        progress = False
        for joint in pending[:]:
            if joint.parent not in transforms:
                continue
            motion = np.eye(4, dtype=np.float64)
            if joint.joint_type != "fixed":
                q = float(q_by_joint[joint.name])
                if not np.isfinite(q):
                    raise RendererError(f"non-finite joint state: {joint.name}")
                if joint.lower is not None and q < joint.lower - 1e-5:
                    raise RendererError(f"joint below lower limit: {joint.name}")
                if joint.upper is not None and q > joint.upper + 1e-5:
                    raise RendererError(f"joint above upper limit: {joint.name}")
                if joint.joint_type in {"revolute", "continuous"}:
                    motion = axis_angle_matrix(joint.axis, q)
                elif joint.joint_type == "prismatic":
                    motion[:3, 3] = joint.axis / np.linalg.norm(joint.axis) * q
            transforms[joint.child] = transforms[joint.parent] @ joint.origin @ motion
            pending.remove(joint)
            progress = True
        if not progress:
            raise RendererError("URDF joint graph is disconnected or cyclic")
    if set(transforms) != set(model.links):
        raise RendererError("FK did not resolve every link")
    return transforms


def range_to_metric_z(
    range_buffer: np.ndarray, fx: float, fy: float, cx: float, cy: float
) -> np.ndarray:
    if range_buffer.ndim != 2 or fx <= 0 or fy <= 0:
        raise RendererError("invalid range buffer or intrinsics")
    height, width = range_buffer.shape
    u = np.arange(width, dtype=np.float64)[None, :]
    v = np.arange(height, dtype=np.float64)[:, None]
    ray_norm = np.sqrt(1.0 + ((u - cx) / fx) ** 2 + ((v - cy) / fy) ** 2)
    return np.asarray(range_buffer, dtype=np.float64) / ray_norm


def full_chain_gaps(
    sidecar_keys: Sequence[str], calibration_file_count: int, placement_ref: str | None
) -> list[str]:
    gaps: list[str] = []
    if "q_arm" not in sidecar_keys:
        gaps.append("R2_SIDECAR_HAS_NO_Q_ARM")
    if calibration_file_count == 0:
        gaps.append("TIANJI_CALIBRATION_EMPTY")
    if not placement_ref:
        gaps.append("UNIFORM_BASE_PLACEMENT_CONTRACT_MISSING")
    return gaps


def _verify_used_assets(
    project_root: Path, pin_path: Path, expected_pin_sha256: str, models: Sequence[UrdfModel]
) -> str:
    pin_sha = sha256_file(pin_path)
    if pin_sha != expected_pin_sha256:
        raise RendererError(f"asset pin SHA mismatch: {pin_sha}")
    pin = json.loads(pin_path.read_text(encoding="utf-8"))
    if pin.get("status") != "T0_IDENTITY_PIN_NOT_EXECUTION_AUTHORIZATION":
        raise RendererError("unexpected D1 asset pin status")
    entries = {entry["path"]: entry for entry in pin.get("files", [])}
    used = {model.path.resolve() for model in models}
    used.update(visual.mesh_path.resolve() for model in models for visual in model.visuals)
    for path in sorted(used):
        try:
            relative = path.relative_to(project_root.resolve()).as_posix()
        except ValueError as exc:
            raise RendererError(f"used asset escapes project root: {path}") from exc
        entry = entries.get(relative)
        if entry is None:
            raise RendererError(f"used asset is absent from pin: {relative}")
        if path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
            raise RendererError(f"used asset drifted from D1 pin: {relative}")
    return pin_sha


def _load_r2_hands(
    sidecar_path: Path,
    frame_name: str,
    left_urdf: UrdfModel,
    right_urdf: UrdfModel,
) -> tuple[list[HandState], dict[str, Any]]:
    with np.load(sidecar_path, allow_pickle=False) as data:
        keys = tuple(sorted(data.files))
        if str(data["schema_version"].item()) != "humanego-robot-sidecar-v1":
            raise RendererError("unsupported R2 sidecar schema")
        frames = [str(value) for value in data["frame_names"]]
        if frame_name not in frames:
            raise RendererError(f"frame absent from R2 sidecar: {frame_name}")
        index = frames.index(frame_name)
        states: list[HandState] = []
        for side_index, (side, model) in enumerate(
            (("left", left_urdf), ("right", right_urdf))
        ):
            if not bool(data["valid"][index, side_index]):
                raise RendererError(f"R2 side is invalid for smoke frame: {side}")
            names = [str(value) for value in data["joint_names"][side_index]]
            q_values = np.asarray(data["q"][index, side_index], dtype=np.float64)
            root_transform = np.asarray(
                data["wrist_T_camera"][index, side_index], dtype=np.float64
            )
            if q_values.shape != (len(names),) or root_transform.shape != (4, 4):
                raise RendererError("R2 hand state has unexpected shape")
            if not np.isfinite(q_values).all() or not np.isfinite(root_transform).all():
                raise RendererError("R2 hand state contains NaN/Inf")
            states.append(
                HandState(side, dict(zip(names, map(float, q_values))), root_transform, model)
            )
        audit = {
            "keys": list(keys),
            "q_arm_present": "q_arm" in keys,
            "frame_index": index,
            "frame_name": frame_name,
            "valid": [bool(value) for value in data["valid"][index]],
            "confidence": [float(value) for value in data["confidence"][index]],
        }
    return states, audit


def _configure_cycles(scene: Any, width: int, height: int, samples: int) -> str:
    import bpy

    scene.render.engine = "CYCLES"
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = True
    scene.render.use_file_extension = True
    scene.cycles.samples = samples
    scene.cycles.use_denoising = False
    scene.cycles.use_adaptive_sampling = False
    preferences = bpy.context.preferences.addons["cycles"].preferences
    preferences.get_devices()
    optix = [device for device in preferences.devices if device.type == "OPTIX"]
    if optix:
        try:
            preferences.compute_device_type = "OPTIX"
        except TypeError:
            pass
        for device in preferences.devices:
            device.use = device in optix
        scene.cycles.device = "GPU"
        return "OPTIX:" + ",".join(device.name for device in optix)
    scene.cycles.device = "CPU"
    return "CPU"


def _clear_scene() -> None:
    import bpy

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in (bpy.data.materials, bpy.data.cameras, bpy.data.lights):
        for item in list(collection):
            collection.remove(item)


def _camera_from_k(scene: Any, k: np.ndarray, width: int, height: int) -> Any:
    import bpy

    fx, fy, cx, cy = float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])
    if fx <= 0 or fy <= 0:
        raise RendererError("invalid camera intrinsics")
    camera_data = bpy.data.cameras.new("EgoCamera")
    camera_data.type = "PERSP"
    camera_data.sensor_fit = "HORIZONTAL"
    camera_data.sensor_width = 36.0
    camera_data.lens = fx * camera_data.sensor_width / width
    camera_data.shift_x = -(cx - width / 2.0) / width
    camera_data.shift_y = (cy - height / 2.0) / width
    camera_data.clip_start = 0.01
    camera_data.clip_end = 10.0
    camera = bpy.data.objects.new("EgoCamera", camera_data)
    scene.collection.objects.link(camera)
    camera.matrix_world = np.eye(4).tolist()
    scene.camera = camera
    return camera


def _make_material(name: str, rgba: Sequence[float]) -> Any:
    import bpy

    material = bpy.data.materials.new(name)
    material.diffuse_color = tuple(float(value) for value in rgba)
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    principled.inputs["Base Color"].default_value = tuple(float(value) for value in rgba)
    principled.inputs["Metallic"].default_value = 0.15
    principled.inputs["Roughness"].default_value = 0.32
    return material


def _import_visual(path: Path, object_name: str) -> Any:
    import bpy

    before = set(bpy.data.objects.keys())
    bpy.ops.wm.stl_import(filepath=str(path))
    created_names = sorted(set(bpy.data.objects.keys()) - before)
    if len(created_names) != 1:
        raise RendererError(f"STL import created {len(created_names)} objects: {path}")
    obj = bpy.data.objects[created_names[0]]
    obj.name = object_name
    return obj


def _add_lighting(scene: Any) -> None:
    import bpy

    scene.world.color = (0.025, 0.025, 0.025)
    world_nodes = scene.world.node_tree if scene.world and scene.world.use_nodes else None
    if scene.world:
        scene.world.use_nodes = True
        world_nodes = scene.world.node_tree
        world_nodes.nodes["Background"].inputs["Color"].default_value = (0.06, 0.07, 0.09, 1)
        world_nodes.nodes["Background"].inputs["Strength"].default_value = 0.35
    light_data = bpy.data.lights.new("SyntheticSmokeKey", type="AREA")
    light_data.energy = 550.0
    light_data.shape = "DISK"
    light_data.size = 0.45
    light = bpy.data.objects.new("SyntheticSmokeKey", light_data)
    scene.collection.objects.link(light)
    light.location = (0.0, 0.0, 0.05)
    light.rotation_euler = (0.0, 0.0, 0.0)


def _configure_pass_outputs(scene: Any, output_dir: Path) -> None:
    scene.view_layers[0].use_pass_z = True
    scene.view_layers[0].use_pass_object_index = True
    scene.use_nodes = True
    nodes = scene.node_tree.nodes
    links = scene.node_tree.links
    nodes.clear()
    render_layers = nodes.new("CompositorNodeRLayers")
    composite = nodes.new("CompositorNodeComposite")
    links.new(render_layers.outputs["Image"], composite.inputs["Image"])
    for pass_name, prefix in (("Depth", "Range_"), ("IndexOB", "ObjectIndex_")):
        output = nodes.new("CompositorNodeOutputFile")
        output.base_path = str(output_dir)
        output.format.file_format = "OPEN_EXR"
        output.format.color_mode = "BW"
        output.format.color_depth = "32"
        output.format.exr_codec = "ZIP"
        output.file_slots[0].path = prefix
        links.new(render_layers.outputs[pass_name], output.inputs[0])


def _read_exr_channel(path: Path) -> np.ndarray:
    import bpy

    image = bpy.data.images.load(str(path), check_existing=False)
    width, height = image.size
    pixels = np.asarray(image.pixels[:], dtype=np.float32).reshape(height, width, 4)
    channel = np.flipud(pixels[..., 0].copy())
    bpy.data.images.remove(image)
    return channel


def _save_visualizations(
    output_dir: Path, range_buffer: np.ndarray, object_index: np.ndarray, valid: np.ndarray
) -> None:
    from PIL import Image

    range_vis = np.zeros((*range_buffer.shape, 3), dtype=np.uint8)
    if valid.any():
        values = range_buffer[valid]
        lo, hi = np.percentile(values, [1, 99])
        scale = max(float(hi - lo), 1e-6)
        normalized = np.clip((range_buffer - lo) / scale, 0, 1)
        gray = ((1.0 - normalized) * 255).astype(np.uint8)
        range_vis[valid] = np.repeat(gray[..., None], 3, axis=2)[valid]
    Image.fromarray(range_vis, mode="RGB").save(output_dir / "Range_VIS.png")

    labels = np.rint(object_index).astype(np.int32)
    index_vis = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for label in np.unique(labels[labels > 0]):
        color = np.array(
            [(53 * label) % 256, (97 * label) % 256, (193 * label) % 256],
            dtype=np.uint8,
        )
        index_vis[labels == label] = color
    Image.fromarray(index_vis, mode="RGB").save(output_dir / "ObjectIndex_VIS.png")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def render_r2_kaihand_smoke(args: argparse.Namespace) -> dict[str, Any]:
    import bpy
    from mathutils import Matrix

    if args.output_dir.exists():
        raise RendererError(f"output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    left_model, right_model = parse_urdf(args.left_urdf), parse_urdf(args.right_urdf)
    pin_sha = _verify_used_assets(
        args.project_root,
        args.asset_pin,
        args.expected_asset_pin_sha256,
        (left_model, right_model),
    )
    sidecar_sha = sha256_file(args.sidecar)
    if sidecar_sha != args.expected_sidecar_sha256:
        raise RendererError(f"sidecar SHA mismatch: {sidecar_sha}")
    hands, sidecar_audit = _load_r2_hands(
        args.sidecar, args.frame_name, left_model, right_model
    )
    raw = json.loads(args.raw_training_data.read_text(encoding="utf-8"))
    metadata = raw.get("metadata", {})
    original_k = np.asarray(metadata.get("k"), dtype=np.float64)
    original_width, original_height = int(metadata.get("w", 0)), int(metadata.get("h", 0))
    if original_k.shape != (3, 3) or original_width <= 0 or original_height <= 0:
        raise RendererError("RAW camera metadata is incomplete")
    k = original_k.copy()
    k[0, :] *= args.width / original_width
    k[1, :] *= args.height / original_height

    calibration = json.loads(args.asset_pin.read_text(encoding="utf-8"))["calibration"]
    gaps = full_chain_gaps(
        sidecar_audit["keys"], int(calibration["file_count"]), placement_ref=None
    )
    if not gaps:
        raise RendererError("hand-only smoke expected explicit full-chain gaps")

    setup_start = time.perf_counter()
    _clear_scene()
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    device = _configure_cycles(scene, args.width, args.height, args.samples)
    _camera_from_k(scene, k, args.width, args.height)
    _add_lighting(scene)
    object_index = 1
    rendered_links: list[dict[str, Any]] = []
    for hand in hands:
        link_transforms = forward_kinematics(hand.urdf, hand.q_by_joint)
        root_in_blender = CV_CAMERA_TO_BLENDER @ hand.root_in_cv_camera
        for visual_number, visual in enumerate(hand.urdf.visuals):
            obj = _import_visual(
                visual.mesh_path, f"{hand.side}:{visual.link}:{visual_number}"
            )
            obj.matrix_world = Matrix(
                (root_in_blender @ link_transforms[visual.link] @ visual.origin).tolist()
            )
            obj.pass_index = object_index
            obj.data.materials.clear()
            obj.data.materials.append(
                _make_material(f"{hand.side}:{visual.link}:material", visual.rgba)
            )
            rendered_links.append(
                {"side": hand.side, "link": visual.link, "object_index": object_index}
            )
            object_index += 1
    _configure_pass_outputs(scene, args.output_dir)
    scene.frame_set(1)
    beauty_path = args.output_dir / "beauty.png"
    scene.render.filepath = str(beauty_path)
    bpy.context.view_layer.update()
    setup_seconds = time.perf_counter() - setup_start

    render_start = time.perf_counter()
    bpy.ops.render.render(write_still=True)
    render_seconds = time.perf_counter() - render_start
    range_files = sorted(args.output_dir.glob("Range_*.exr"))
    index_files = sorted(args.output_dir.glob("ObjectIndex_*.exr"))
    if len(range_files) != 1 or len(index_files) != 1 or not beauty_path.is_file():
        raise RendererError("Cycles did not produce all three required buffers")
    range_exr, index_exr = range_files[0], index_files[0]
    range_buffer = _read_exr_channel(range_exr)
    index_buffer = _read_exr_channel(index_exr)
    if range_buffer.shape != (args.height, args.width) or index_buffer.shape != range_buffer.shape:
        raise RendererError("render pass dimensions are incorrect")
    valid = np.isfinite(range_buffer) & (range_buffer > 0) & (range_buffer < 9.999)
    if not valid.any():
        raise RendererError("Range buffer has no robot pixels")
    labels = np.rint(index_buffer[valid]).astype(np.int32)
    if not np.any(labels > 0):
        raise RendererError("ObjectIndex buffer has no positive robot labels")
    metric_z = range_to_metric_z(
        range_buffer, float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])
    )
    np.save(args.output_dir / "Range.npy", range_buffer.astype(np.float32), allow_pickle=False)
    np.save(args.output_dir / "Z_metric.npy", metric_z.astype(np.float32), allow_pickle=False)
    np.save(args.output_dir / "ObjectIndex.npy", index_buffer.astype(np.float32), allow_pickle=False)
    _save_visualizations(args.output_dir, range_buffer, index_buffer, valid)

    outputs: dict[str, dict[str, Any]] = {}
    for path in sorted(args.output_dir.iterdir()):
        if path.is_file() and path.name != "RUN_MANIFEST.json":
            outputs[path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    manifest: dict[str, Any] = {
        "schema_version": "d2-cycles-renderer-smoke-v1",
        "status": "candidate_requires_human_review",
        "next_bucket_blocked": True,
        "advancement_authorized": False,
        "claim_limit": "R2_KAIHAND_ONLY_RENDERER_SMOKE_NOT_TIANJI_FULL_CHAIN_NOT_PRODUCTION",
        "engine": "CYCLES",
        "cycles_samples": args.samples,
        "resolution": [args.width, args.height],
        "device": device,
        "frame": {"session_id": "grap_a_cap_004", "frame_name": args.frame_name},
        "asset_pin": {"path": str(args.asset_pin), "sha256": pin_sha},
        "sidecar": {"path": str(args.sidecar), "sha256": sidecar_sha, **sidecar_audit},
        "raw_camera": {
            "path": str(args.raw_training_data),
            "sha256": sha256_file(args.raw_training_data),
            "original_resolution": [original_width, original_height],
            "scaled_k": k.tolist(),
            "camera_coordinate_conversion": "opencv_xyz_to_blender_x_negY_negZ",
        },
        "placement": {
            "hand_root_source": "frozen_R2_wrist_T_camera",
            "hand_joint_source": "frozen_R2_q_exact_URDF_joint_names",
            "renderer_only_wrist_or_root_offset": False,
            "arm_rendered": False,
            "q_arm_present": False,
            "uniform_base_placement_ref": None,
        },
        "materials": "URDF_RGBA_WITH_FIXED_SMOKE_PBR_PARAMETERS_NOT_SCENE_CALIBRATED",
        "lighting": "SYNTHETIC_SMOKE_AREA_LIGHT_NOT_SCENE_CALIBRATED",
        "rendered_link_visuals": rendered_links,
        "full_chain_gaps": gaps,
        "buffers": {
            "beauty": "beauty.png",
            "range": "Range.npy",
            "metric_z": "Z_metric.npy",
            "object_index": "ObjectIndex.npy",
            "range_semantics": "Cycles Depth pass: Euclidean camera-to-surface range in metres",
            "z_conversion": "range/sqrt(1+((u-cx)/fx)^2+((v-cy)/fy)^2)",
        },
        "metrics": {
            "setup_seconds": setup_seconds,
            "render_seconds": render_seconds,
            "foreground_pixels": int(valid.sum()),
            "range_m_min": float(range_buffer[valid].min()),
            "range_m_max": float(range_buffer[valid].max()),
            "metric_z_m_min": float(metric_z[valid].min()),
            "metric_z_m_max": float(metric_z[valid].max()),
            "positive_object_indices": sorted(int(value) for value in np.unique(labels) if value > 0),
        },
        "outputs": outputs,
    }
    _atomic_json(args.output_dir / "RUN_MANIFEST.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--asset-pin", type=Path, required=True)
    parser.add_argument("--expected-asset-pin-sha256", required=True)
    parser.add_argument("--left-urdf", type=Path, required=True)
    parser.add_argument("--right-urdf", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--expected-sidecar-sha256", required=True)
    parser.add_argument("--raw-training-data", type=Path, required=True)
    parser.add_argument("--frame-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--samples", type=int, default=4)
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0 or args.samples != 4:
        parser.error("positive dimensions and exactly 4 Cycles samples are required")
    try:
        manifest = render_r2_kaihand_smoke(args)
    except (RendererError, OSError, ValueError, KeyError) as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "claim_limit": manifest["claim_limit"],
                "metrics": manifest["metrics"],
                "full_chain_gaps": manifest["full_chain_gaps"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
