"""Candidate-only complete Robot diagnostic consumer of sealed Source/Motion.

This module does not import Robot/FK/Numba at import time. The registered worker
owns authorization; the adopted V1 raster, clipping, colours and assets are
reused only after its gate. Raw+Robot is a diagnostic, not Clean/Composite.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Any, Callable, Mapping

import numpy as np

from .contracts import SCHEMA, file_ref, verify_ref, verify_sealed_stage

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TIMEBASE = "TIMEBASE_UNVERIFIED_HOST_ARRIVAL_QPC_DECLARED_NS"
CODE_CLOSURE = ("src/chaoyang/exact78_baseline/render.py",
                "src/chaoyang/exact78_baseline/robot_model.py",
                "src/chaoyang/pipeline/exact78_robot_raster_v1.py",
                "src/chaoyang/pipeline/robot_renderer_cycles.py",
                "src/chaoyang/pipeline/robot_renderer_eevee_fullchain.py",
                "src/chaoyang/pipeline/robot_wrist_kai_adapter.py",
                "src/chaoyang/pipeline/robot_tool_definition_contract.py",
                "src/chaoyang/pipeline/robot_mount_visual_proxy.py",
                "src/chaoyang/ops/immutable_artifact_io.py",
                "src/chaoyang/inpainting_candidate/contracts.py",
                "src/chaoyang/inpainting_candidate/source_frames.py")


class RenderStageError(ValueError):
    pass


def _read(ref: Mapping) -> Any:
    verify_ref(ref)
    return json.loads(Path(ref["path"]).read_text(encoding="utf-8-sig"))


def _signature(value, name: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise RenderStageError(f"RENDER_{name}_SIGNATURE_REQUIRED")


def _gate(authorization: Callable, identity: Mapping | None = None) -> dict:
    if not callable(authorization):
        raise RenderStageError("RENDER_REGISTERED_AUTHORIZATION_REQUIRED")
    value = authorization()
    if (not isinstance(value, Mapping) or value.get("execution_allowed") is not True
            or value.get("algorithm_execution_allowed") is not True
            or not isinstance(value.get("task_id"), str)
            or not value.get("task_id") or not value.get("fencing_token")):
        raise RenderStageError("RENDER_EXECUTION_NOT_AUTHORIZED")
    _signature(value.get("run_signature"), "RUN")
    current = {key: value[key] for key in ("task_id", "run_signature", "fencing_token")}
    if identity is not None and current != identity:
        raise RenderStageError("RENDER_WRITER_IDENTITY_CHANGED")
    return current


def _sealed(ref: Mapping, signature: str, stage: str, repo: Path) -> tuple[dict, dict]:
    path = Path(ref["path"]).resolve(strict=True)
    if not path.is_relative_to(repo):
        raise RenderStageError("RENDER_UPSTREAM_OUTSIDE_PROJECT")
    receipt = _read(ref)
    if receipt.get("stage") != stage:
        raise RenderStageError(f"RENDER_SEALED_{stage.upper()}_REQUIRED")
    outputs = verify_sealed_stage(receipt, signature, output_root=path.parent)
    # Motion also binds its Native substage. Its FRAME_STATUS.json is a
    # different schema, not a replacement for Motion/FRAME_STATUS.json.
    return receipt, {Path(row["path"]).name: row for row in outputs
                     if Path(row["path"]).resolve(strict=True).parent == path.parent}


class _NearClippedRaster:
    """Carry original labels through V1 near clipping on BOTH z-buffer passes."""
    def __init__(self, raster, near_clip):
        self.raster, self.near_clip, self.calls = raster, near_clip, 0

    def __getattr__(self, key):
        return getattr(self.raster, key)

    def rasterize_zbuffer(self, triangles, colours, labels, *projection):
        self.calls += 1
        # On the triangle pass labels are the original triangle indices. A
        # triangle split by clipping retains that same index on both pieces.
        clipped = self.near_clip(triangles, colours, labels)
        return self.raster.rasterize_zbuffer(*clipped, *projection)


def _load_render_runtime(repo: Path):
    from chaoyang.exact78_baseline.robot_model import Robot, FLANGE_MESH
    from chaoyang.exact78_baseline import render as display
    from chaoyang.pipeline import exact78_robot_raster_v1 as raster
    robot = Robot(repo)
    return SimpleNamespace(robot=robot, raster=raster, display=display,
        flange=raster.naturalv2_local_triangles(repo / FLANGE_MESH), cache={})


def _write_review_video(path: Path, rgb: np.ndarray, mapping: list[dict]) -> dict:
    from .source_frames import write_review_video
    return write_review_video(path, rgb, mapping)


def _json(path: Path, value) -> dict:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return file_ref(path)


def _asset_refs(ref: Mapping | None, repo: Path) -> list[dict]:
    if not isinstance(ref, Mapping):
        raise RenderStageError("RENDER_FROZEN_ASSET_CLOSURE_REQUIRED")
    value = _read(ref)
    assets = value.get("assets")
    if (value.get("status") != "VERIFIED_STATIC_ASSET_CLOSURE"
            or not isinstance(assets, list) or not assets or value.get("issues") != []
            or value.get("asset_count") != len(assets)):
        raise RenderStageError("RENDER_ASSET_CLOSURE_NOT_VERIFIED")
    result = [dict(ref)]
    for row in assets:
        pin = {key: row[key] for key in ("path", "bytes", "sha256")}
        if not Path(pin["path"]).resolve(strict=True).is_relative_to(repo):
            raise RenderStageError("RENDER_ASSET_OUTSIDE_PROJECT")
        verify_ref(pin)
        result.append(pin)
    return result


def _finite_arrays(arrays: Mapping, count: int) -> None:
    shapes = {"q_arm": (count, 2, 7), "q_hand": (count, 2, 22), "K": (3, 3),
              "T_world_base": (4, 4), "c2w": (count, 4, 4), "qpc_ns": (count,)}
    for key, shape in shapes.items():
        value = arrays.get(key)
        if (value is None or value.shape != shape or value.dtype.hasobject
                or not np.isfinite(value).all()):
            raise RenderStageError(f"RENDER_MOTION_ARRAY_INVALID:{key}")
    if arrays["qpc_ns"].dtype.kind not in "iu":
        raise RenderStageError("RENDER_QPC_INTEGER_REQUIRED")
    transforms = np.concatenate((arrays["T_world_base"][None], arrays["c2w"]), axis=0)
    if (not np.allclose(transforms[:, 3], [0, 0, 0, 1])
            or np.any(np.abs(np.linalg.det(transforms[:, :3, :3])) < 1e-12)):
        raise RenderStageError("RENDER_TRANSFORM_NOT_INVERTIBLE_HOMOGENEOUS")


def _buffers(colour, labels, value, height: int, width: int) -> tuple:
    if (not isinstance(value, Mapping) or value.get("one_scene_zbuffer") is not True
            or not isinstance(colour, np.ndarray) or colour.dtype != np.uint8
            or colour.shape != (height, width, 3)
            or not isinstance(labels, np.ndarray) or labels.dtype.kind not in "iu"
            or labels.shape != (height, width)):
        raise RenderStageError("RENDER_REAL_DEPTH_BUFFERS_REQUIRED_NOT_HAND_ROOTS")
    depth, triangle = value.get("depth_m"), value.get("triangle_id")
    if (not isinstance(depth, np.ndarray) or depth.dtype.kind != "f" or depth.shape != labels.shape
            or not isinstance(triangle, np.ndarray) or triangle.dtype.kind not in "iu"
            or triangle.shape != labels.shape):
        raise RenderStageError("RENDER_DEPTH_TRIANGLE_SHAPE_INVALID")
    visible = labels >= 0
    if (np.any(~np.isfinite(depth[visible])) or np.any(depth[visible] <= 0)
            or np.any(triangle[visible] < 0) or np.any(labels[~visible] != -1)
            or np.any(triangle[~visible] != -1) or np.any(~np.isposinf(depth[~visible]))):
        raise RenderStageError("RENDER_DEPTH_LABEL_TRIANGLE_COVERAGE_MISMATCH")
    ids = value.get("label_identity")
    if (not isinstance(ids, Mapping)
            or {int(key) for key in ids} != set(np.unique(labels[visible]).tolist())):
        raise RenderStageError("RENDER_LINK_IDENTITY_MISSING")
    identities = {}
    for label, row in ids.items():
        if (not isinstance(row, Mapping) or set(row) != {"instance_id", "part_id", "link_id"}
                or any(not isinstance(v, str) or not v for v in row.values())):
            raise RenderStageError("RENDER_LINK_IDENTITY_INVALID")
        identities[str(int(label))] = dict(row)
    rgba = np.empty((height, width, 4), np.uint8)
    rgba[..., :3] = colour[..., ::-1]
    rgba[..., 3] = visible.astype(np.uint8) * 255
    return rgba, depth.astype(np.float32), labels.astype(np.int32), triangle.astype(np.int64), identities


def build_render_stage(*, source_receipt: Mapping, source_signature: str,
                       motion_receipt: Mapping, motion_signature: str,
                       output_dir: str | Path, component_signature: str,
                       repo_root: str | Path, authorization: Callable,
                       asset_closure: Mapping | None = None) -> dict:
    """Seal full-link diagnostic buffers/videos; numerical FAIL stays per-frame.

    Camera-z depth comes only from render_robot(return_buffers=True). Neither
    hidden geometry nor scene occlusion is inferred. Rendering raw human pixels
    behind Robot does not repair or qualify Clean.
    """
    identity = _gate(authorization)
    for signature, name in ((source_signature, "SOURCE"), (motion_signature, "MOTION"),
                            (component_signature, "COMPONENT")):
        _signature(signature, name)
    repo, root = Path(repo_root).resolve(strict=True), Path(output_dir).resolve()
    if repo != PROJECT_ROOT.resolve() or root == repo or not root.is_relative_to(repo):
        raise RenderStageError("RENDER_OUTPUT_OR_REPO_OUTSIDE_PROJECT")
    if root.exists():
        raise RenderStageError("RENDER_FRESH_OUTPUT_REQUIRED")
    source, source_outputs = _sealed(source_receipt, source_signature, "Source", repo)
    motion, motion_outputs = _sealed(motion_receipt, motion_signature, "Motion", repo)
    if (motion.get("source_receipt") != dict(source_receipt)
            or motion.get("source_signature") != source_signature
            or motion.get("session_id") != source.get("session_id")
            or motion.get("native_frames") != source.get("native_frames")):
        raise RenderStageError("RENDER_SOURCE_MOTION_BINDING_MISMATCH")
    if (motion.get("kinematic_actions_generated") is not True
            or motion.get("consumer_status", {}).get("Render") != "READY_OFFLINE_VISUAL_CANDIDATE"):
        raise RenderStageError("RENDER_BLOCKED_MOTION_CAMERA_OR_KINEMATIC_INPUT")
    if not {"RAW_RGB.npz", "FRAME_MAP.json", "SOURCE_REPORT.json"} <= source_outputs.keys():
        raise RenderStageError("RENDER_SOURCE_OUTPUT_CONTRACT_INCOMPLETE")
    if not {"MOTION.npz", "FRAME_MAP.json", "FRAME_STATUS.json", "MOTION_REPORT.json"} <= motion_outputs.keys():
        raise RenderStageError("RENDER_MOTION_OUTPUT_CONTRACT_INCOMPLETE")
    mapping = _read(source_outputs["FRAME_MAP.json"])
    states = _read(motion_outputs["FRAME_STATUS.json"])
    motion_report = _read(motion_outputs["MOTION_REPORT.json"])
    count = source.get("native_frames")
    if (type(count) is not int or count < 1 or not isinstance(mapping, list)
            or len(mapping) != count or not isinstance(states, list) or len(states) != count
            or _read(motion_outputs["FRAME_MAP.json"]) != mapping
            or motion_report.get("native_frames") != count
            or motion_report.get("session_id") != source.get("session_id")
            or motion_report.get("native_image_size") != [1280, 960]):
        raise RenderStageError("RENDER_NATIVE_MAPPING_OR_CAMERA_DOMAIN_MISMATCH")
    for index, (row, state) in enumerate(zip(mapping, states, strict=True)):
        if (not isinstance(row, dict) or not isinstance(state, dict)
                or any(type(row.get(key)) is not int or row[key] != index for key in
                       ("native_frame_index", "decoded_display_index", "model_input_frame", "output_frame"))
                or type(row.get("qpc_declared_ns")) is not int
                or state.get("native_frame_index") != index or state.get("source_frame_map") != row):
            raise RenderStageError("RENDER_NATIVE_FRAME_IDENTITY_CHANGED")
    qpc = np.asarray([row["qpc_declared_ns"] for row in mapping], np.int64)
    if np.any(np.diff(qpc) <= 0):
        raise RenderStageError("RENDER_QPC_DUPLICATE_OR_BACKWARD")
    with np.load(source_outputs["RAW_RGB.npz"]["path"], allow_pickle=False) as archive:
        rgb = archive["rgb"]
    if rgb.dtype != np.uint8 or rgb.shape != (count, 480, 640, 3):
        raise RenderStageError("RENDER_SOURCE_NATIVE_RGB_CONTRACT_MISMATCH")
    with np.load(motion_outputs["MOTION.npz"]["path"], allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    _finite_arrays(arrays, count)
    if not np.array_equal(arrays["qpc_ns"], qpc):
        raise RenderStageError("RENDER_MOTION_QPC_CHANGED")
    asset_refs = _asset_refs(asset_closure, repo)
    # Keep the full verified closure, including nested Native outputs, even
    # though named consumer inputs are selected only from the stage root.
    actual_inputs = [dict(source_receipt), dict(motion_receipt), *source["outputs"],
        *motion["outputs"], *asset_refs, file_ref(Path(__file__)),
        *[file_ref(repo / path) for path in CODE_CLOSURE]]
    unique = {row["path"]: row for row in actual_inputs}
    actual_inputs = [unique[key] for key in sorted(unique)]
    _gate(authorization, identity)
    runtime = _load_render_runtime(repo)
    raster = _NearClippedRaster(runtime.raster, runtime.display.near_clip)
    K = runtime.display.scaled_intrinsics(arrays["K"], 640, 480, (1280, 960))
    first_camera = np.linalg.inv(arrays["c2w"][0]) @ arrays["T_world_base"]
    external_camera = runtime.display.external_camera(first_camera)
    external_K = np.asarray(runtime.display.EXTERNAL_K)
    root.mkdir(parents=True, exist_ok=False)
    rgba_all = np.empty((count, 480, 640, 4), np.uint8)
    depth_all = np.empty((count, 480, 640), np.float32)
    label_all = np.empty((count, 480, 640), np.int32)
    triangle_all = np.empty((count, 480, 640), np.int64)
    fpv_video = np.empty((count, 480, 1280, 3), np.uint8)
    third_video = np.empty((count, 480, 640, 3), np.uint8)
    identities, frame_status = {}, []
    render_calls = 0
    for frame in range(count):
        _gate(authorization, identity)
        camera = np.linalg.inv(arrays["c2w"][frame]) @ arrays["T_world_base"]
        views = []
        for view_camera, view_K in ((camera, K), (external_camera, external_K)):
            before = raster.calls
            colour, labels, buffers = runtime.raster.render_robot(raster, runtime.robot.assets,
                arrays["q_arm"][frame], arrays["q_hand"][frame], runtime.robot.tool_mounts,
                view_camera, view_K, runtime.cache, runtime.flange,
                complete_robot=True, return_buffers=True, width=640, height=480)
            render_calls += 1
            if raster.calls - before != 2:
                raise RenderStageError("RENDER_BOTH_ZBUFFER_PASSES_REQUIRED")
            view = _buffers(colour, labels, buffers, 480, 640)
            for key, row in view[4].items():
                if key in identities and identities[key] != row:
                    raise RenderStageError("RENDER_LINK_IDENTITY_DRIFT")
                identities[key] = row
            views.append(view)
        rgba, depth, labels, triangles, _ = views[0]
        rgba_all[frame], depth_all[frame], label_all[frame], triangle_all[frame] = rgba, depth, labels, triangles
        fpv_video[frame, :, :640] = rgb[frame]
        fpv_video[frame, :, 640:] = runtime.display.over(rgb[frame], rgba[..., :3], rgba[..., 3] > 0)
        third_video[frame] = runtime.display.on_grey(views[1][0][..., :3], views[1][0][..., 3] > 0)
        state = deepcopy(states[frame])
        state.update(render_status="COMPLETE_DIAGNOSTIC_RENDER", complete_robot=True,
            fpv_visible_pixels=int((labels >= 0).sum()),
            third_person_visible_pixels=int((views[1][2] >= 0).sum()),
            scene_depth_status="UNKNOWN_NOT_CONSUMED", scene_occlusion_status="NOT_EXECUTED",
            visual_quality="NOT_REVIEWED", human_review="PENDING", training_eligible=False,
            control_ground_truth=False)
        frame_status.append(state)
        _gate(authorization, identity)
    buffer_path = root / "ROBOT_BUFFERS.npz"
    with buffer_path.open("xb") as stream:
        np.savez_compressed(stream, rgba=rgba_all, depth_m=depth_all, link_id=label_all,
            triangle_id=triangle_all, qpc_ns=qpc, K=K, third_person_K=external_K,
            T_world_base=arrays["T_world_base"], c2w=arrays["c2w"],
            T_third_person_base=external_camera)
    fpv = _write_review_video(root / "RAW_ROBOT_FPV_REVIEW.mp4", fpv_video, mapping)
    third = _write_review_video(root / "ROBOT_FULL_ARM_THIRD_PERSON_REVIEW.mp4", third_video, mapping)
    if fpv.get("decoded_frames") != count or third.get("decoded_frames") != count:
        raise RenderStageError("RENDER_REVIEW_FULL_DECODE_COUNT_MISMATCH")
    verify_ref(fpv["video"])
    verify_ref(third["video"])
    report = {"schema": "inpainting-candidate-render-v1", "session_id": source["session_id"],
        "native_frames": count, "complete_robot": True, "render_calls": render_calls,
        "render_views": 2, "zbuffer_passes": raster.calls, "actual_inputs": actual_inputs,
        "source_signature": source_signature, "motion_signature": motion_signature,
        "intrinsics_domain": "MOTION_1280x960_SCALED_ONCE_TO_SOURCE_640x480",
        "rgb_channels": "RGBA_RGB_NOT_BGR", "depth_units": "METRE_CAMERA_Z_ROBOT_ONLY",
        "link_identity": "LABEL_IDENTITY.json",
        "triangle_id_semantics": "PER_FRAME_INPUT_TRIANGLE_INDEX_BEFORE_FINAL_NEAR_CLIP",
        "third_person_camera": "FIXED_BASE_FRAME_CAMERA_FROM_FIRST_EGO_DIRECTION",
        "third_person_depth_used": "SAME_COMPLETE_ROBOT_ZBUFFER_NOT_SAVED_AS_SCENE_DEPTH",
        "near_clip": "FROZEN_V1_NEAR_CLIP_ON_COLOUR_AND_TRIANGLE_ID_PASSES",
        "review": {"fpv": fpv, "third_person": third}, "timebase_status": TIMEBASE,
        "time_mode": "OFFLINE_NONCAUSAL_MOTION_DIAGNOSTIC", "model_calls": 0,
        "scene_geometry": "UNKNOWN_NOT_CONSUMED", "clean_quality": "NOT_EXECUTED",
        "composite_quality": "NOT_EXECUTED", "visual_quality": "NOT_REVIEWED",
        "human_review": "PENDING", "control_ground_truth": False, "training_eligible": False}
    outputs = [file_ref(buffer_path), _json(root / "FRAME_MAP.json", mapping),
        _json(root / "FRAME_STATUS.json", frame_status),
        _json(root / "LABEL_IDENTITY.json", identities),
        _json(root / "RENDER_REPORT.json", report), fpv["video"], third["video"]]
    _gate(authorization, identity)
    for ref in actual_inputs:
        verify_ref(ref)
    _gate(authorization, identity)
    receipt = {"schema": SCHEMA, "stage": "Render", "status": "SEALED_COMPLETE",
        "component_signature": component_signature, "source_signature": source_signature,
        "motion_signature": motion_signature, "source_receipt": dict(source_receipt),
        "motion_receipt": dict(motion_receipt), "session_id": source["session_id"],
        "native_frames": count, "outputs": outputs, "actual_inputs": actual_inputs,
        "complete_robot": True, "render_calls": render_calls, "render_views": 2,
        "zbuffer_passes": raster.calls, "writer_identity": identity, "model_calls": 0,
        "timebase_status": TIMEBASE, "scene_occlusion_status": "NOT_EXECUTED",
        "visual_quality": "NOT_REVIEWED", "human_review": "PENDING",
        "control_ground_truth": False, "training_eligible": False}
    _json(root / "RECEIPT.json", receipt)
    return receipt
