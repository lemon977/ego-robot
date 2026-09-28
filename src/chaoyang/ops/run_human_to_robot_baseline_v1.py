#!/usr/bin/env python3
"""唯一产品入口：同会话 Scene Clean + HandMotion/R0 + 同资产 Robot 渲染。

用法：
  chaoyang run run_human_to_robot_baseline_v1 --input <视频或会话目录> \
    --motion-source hawor --output <新目录> --config <绑定JSON> --dry-run
  去掉 --dry-run 执行；同签名已有完整收据时使用 --resume。

绑定JSON必须含 scene_clean_manifest 与 robot_r0 的 {path,sha256,bytes}。
没有可信 Clean 时明确阻塞，绝不把 Raw overlay 命名为 robot.mp4。
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np

from chaoyang.governance.common import REPO_ROOT
from chaoyang.ops.build_visible_surface_occlusion_canary_v71 import quality_evidence
from chaoyang.pipeline.occlusion_compositor_v1 import (
    Ownership,
    choose_object_pixels,
    exclude_removed_foreground_depth,
    resolve_ownership,
)
from chaoyang.pipeline.v5_product import ProductRobotRenderer, composite


ROUTE = REPO_ROOT / "_run/current/four_stream_visual_delivery_v5/attempts/attempt_0001/ROUTE_MANIFEST.json"
SCHEMA = "chaoyang-human-to-robot-product-v1"


def sha(path: Path) -> dict:
    path = path.resolve(strict=True)
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": h.hexdigest()}


def check_ref(item: dict) -> Path:
    observed = sha(Path(item["path"]))
    if observed != item:
        raise ValueError(f"SHA_BINDING_MISMATCH:{item['path']}")
    return Path(observed["path"])


def pipeline_signature() -> dict:
    paths = (
        Path(__file__),
        REPO_ROOT / "src/chaoyang/pipeline/v5_product.py",
        REPO_ROOT / "src/chaoyang/pipeline/occlusion_compositor_v1.py",
    )
    rows = [sha(path) for path in paths]
    digest = hashlib.sha256(
        "\n".join(str(row["sha256"]) for row in rows).encode("ascii")
    ).hexdigest()
    return {"sha256": digest, "components": rows}


def validate_s2_side_mapping(motion: dict, binding: dict, evidence: dict) -> None:
    expected = np.asarray(binding["expected_human_to_physical"], dtype=np.int64)
    actual = np.asarray(motion["human_to_physical"], dtype=np.int64)
    if expected.shape != (2,) or not np.array_equal(actual, expected):
        raise ValueError("S2_SIDE_MAPPING_BINDING_MISMATCH")
    if evidence.get("human_to_physical") != expected.tolist():
        raise ValueError("S2_SIDE_MAPPING_EVIDENCE_MISMATCH")
    if evidence.get("non_symmetric_test_pass") is not True:
        raise ValueError("S2_SIDE_MAPPING_ASYMMETRIC_TEST_REQUIRED")


def validate_s2_interfaces(binding: dict) -> None:
    """Fail closed when the formal S2 entry is pointed at a legacy RGB path."""
    if binding.get("renderer_interface") != "RGB_ALPHA_OPTICAL_DEPTH_VALID_COMPONENT_ID_V1":
        raise ValueError("S2_RENDERER_INTERFACE_REQUIRED")
    if binding.get("compositor_interface") != "VISIBLE_SURFACE_OWNERSHIP_V1_NO_RGB_FALLBACK":
        raise ValueError("S2_COMPOSITOR_INTERFACE_REQUIRED")


def validate_scene_for_product(scene: dict, binding: dict) -> None:
    """Technical Clean quality precedes product; user adoption follows review."""
    mode = binding.get("product_mode")
    if mode == "STRICT_PRODUCT":
        if scene.get("quality") != "PASS":
            raise ValueError("S2_STRICT_PRODUCT_REQUIRES_QUALITY_PASS_SCENE")
    elif mode == "CANDIDATE_ONLY":
        if not (binding.get("allow_rejected_scene") is True
                and scene.get("quality") == "REJECTED_QUALITY"):
            raise ValueError("S2_CANDIDATE_SCENE_AUTHORIZATION_MISSING")
    else:
        raise ValueError("S2_PRODUCT_MODE_REQUIRED")


def _inspect_s2(binding: dict, binding_path: Path, sid: str, row: dict, domain: dict,
                domain_path: Path, motion: dict, r0_path: Path) -> dict:
    validate_s2_interfaces(binding)
    scene_path = check_ref(binding["scene_clean_manifest"])
    prep_path = check_ref(binding["scene_prep_manifest"])
    mount_path = check_ref(binding["mount_contract"])
    side_path = check_ref(binding["side_mapping_evidence"])
    geometry_mode = binding.get("geometry_mode", "VISIBLE_SURFACE_DEPTH")
    if geometry_mode not in {"VISIBLE_SURFACE_DEPTH", "UNKNOWN_NO_QUALIFIED_DEPTH"}:
        raise ValueError("S2_GEOMETRY_MODE_INVALID")
    mask_manifest_path = check_ref(binding["mask_manifest"])
    mask_index_path = check_ref(binding["mask_frame_index"])
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    prep = json.loads(prep_path.read_text(encoding="utf-8"))
    mount = json.loads(mount_path.read_text(encoding="utf-8"))
    side_evidence = json.loads(side_path.read_text(encoding="utf-8"))
    mask_manifest = json.loads(mask_manifest_path.read_text(encoding="utf-8"))
    mask_index = json.loads(mask_index_path.read_text(encoding="utf-8"))
    depth_result = object_result = None
    depth_result_path = object_path = None
    depth_index = {"rows": []}
    if geometry_mode == "VISIBLE_SURFACE_DEPTH":
        depth_result_path = check_ref(binding["depth_result"])
        depth_index_path = check_ref(binding["depth_frame_index"])
        object_path = check_ref(binding["object6d_result"])
        depth_result = json.loads(depth_result_path.read_text(encoding="utf-8"))
        object_result = json.loads(object_path.read_text(encoding="utf-8"))
        depth_index = json.loads(depth_index_path.read_text(encoding="utf-8"))
    count = int(row["frame_count"])
    if scene.get("session_id") != sid or scene.get("frame_count") != count:
        raise ValueError("S2_SCENE_ID_OR_FRAME_COUNT_MISMATCH")
    if prep.get("image_domain") != domain["image_domain"] or binding.get("image_domain") != domain["image_domain"]:
        raise ValueError("S2_SCENE_IMAGE_DOMAIN_MISMATCH")
    if (len(scene.get("rows", [])) != count
            or (geometry_mode == "VISIBLE_SURFACE_DEPTH" and len(depth_index.get("rows", [])) != count)
            or len(mask_index.get("rows", [])) != count):
        raise ValueError("S2_COMPONENT_FRAME_COUNT_MISMATCH")
    validate_scene_for_product(scene, binding)
    validate_s2_side_mapping(motion, binding, side_evidence)
    contract_mount = np.asarray([
        mount["transforms"]["left_flange_to_hand_root"],
        mount["transforms"]["right_flange_to_hand_root"],
    ], dtype=np.float64)
    if not np.allclose(contract_mount, motion["T_flange_hand"], atol=1e-10, rtol=0):
        raise ValueError("S2_MOUNT_CONTRACT_MISMATCH")
    if geometry_mode == "VISIBLE_SURFACE_DEPTH":
        if depth_result.get("local_stereo_metric_dev") is not True:
            raise ValueError("S2_DEPTH_LOCAL_METRIC_DEV_REQUIRED")
        if object_result.get("coordinate_domain") != "PHYSICAL_LEFT_ENCODED_RESIZE_ONLY_OPTICAL_Z":
            raise ValueError("S2_OBJECT_DEPTH_DOMAIN_MISMATCH")
    if mask_manifest.get("domain") != domain["image_domain"]:
        raise ValueError("S2_MASK_DOMAIN_MISMATCH")
    clean_paths = []
    for index, item in enumerate(scene["rows"]):
        if int(item["frame_id"]) != index:
            raise ValueError(f"S2_CLEAN_FRAME_ID:{index}")
        clean_paths.append(check_ref(item["clean"]))
    depth_paths = ([check_ref(item) for item in depth_index["rows"]]
                   if geometry_mode == "VISIBLE_SURFACE_DEPTH" else None)
    mask_roles = tuple(binding.get("mask_roles", ("human_hand", "human_forearm", "capture_device", "task_object")))
    if "task_object" not in mask_roles or not set(mask_roles).issubset(
            {"human_hand", "human_forearm", "capture_device", "task_object"}):
        raise ValueError("S2_MASK_ROLE_SET_INVALID")
    mask_paths = []
    for frame, item in enumerate(mask_index["rows"]):
        mask_paths.append({role: check_ref(item[role]) for role in mask_roles})
        if any(path.name != f"{frame:06d}.png" for path in mask_paths[-1].values()):
            raise ValueError(f"S2_MASK_FRAME_ID:{frame}")
    return {
        "route": sha(ROUTE), "domain": domain, "domain_ref": sha(domain_path),
        "session": sid, "scene": scene, "scene_ref": sha(scene_path),
        "scene_prep_ref": sha(prep_path), "motion": motion, "robot_ref": sha(r0_path),
        "binding": binding, "binding_ref": sha(binding_path), "execution_round": "S2",
        "clean_paths": clean_paths, "depth_paths": depth_paths, "mask_paths": mask_paths,
        "geometry_mode": geometry_mode, "depth_result": depth_result,
        "depth_result_ref": sha(depth_result_path) if depth_result_path else None,
        "object_result_ref": sha(object_path) if object_path else None,
        "mask_manifest_ref": sha(mask_manifest_path),
        "mount_ref": sha(mount_path), "side_mapping_ref": sha(side_path),
    }


def source_session(path: Path, rows: list[dict]) -> dict:
    if not path.exists():
        raise ValueError(f"INPUT_NOT_FOUND:{path}")
    names = []
    for row in rows:
        sid = row["session_id"]
        if path.name == sid or (path.is_file() and path.parent.name == sid and sid in path.stem):
            names.append(row)
    if len(names) != 1:
        raise ValueError(f"INPUT_SESSION_AMBIGUOUS_OR_UNSUPPORTED:{path}")
    return names[0]


def binding_for(config: Path | None, session: str) -> tuple[Path, dict]:
    path = config or REPO_ROOT / f"tasks/current/four_stream_visual_delivery_v5/BINDING_{session}.json"
    if not path.is_file():
        raise ValueError(f"STAGE_BINDING_MISSING:{path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("route_id") != "HUMAN_TO_ROBOT_BASELINE_V1" or value.get("session_id") != session:
        raise ValueError("STAGE_BINDING_ROUTE_OR_SESSION_MISMATCH")
    return path, value


def inspect(input_path: Path, source: str, config: Path | None) -> dict:
    route = json.loads(ROUTE.read_text(encoding="utf-8"))
    if route.get("route_id") != "HUMAN_TO_ROBOT_BASELINE_V1":
        raise ValueError("ROUTE_ID_MISMATCH")
    row = source_session(input_path, route["sessions"])
    sid = row["session_id"]
    if source != "hawor":
        raise ValueError("CONTROLLER_MANUS_SCENE_FOR_THIS_SESSION_NOT_AVAILABLE")
    domain_path = check_ref(row["domain_manifest"])
    domain = json.loads(domain_path.read_text(encoding="utf-8"))
    binding_path, binding = binding_for(config, sid)
    scene_path = check_ref(binding["scene_clean_manifest"])
    r0_path = check_ref(binding["robot_r0"])
    with np.load(r0_path, allow_pickle=False) as data:
        motion = {key: np.asarray(data[key]) for key in data.files}
    count = row["frame_count"]
    if len(motion["frame_id"]) != count or not np.array_equal(motion["frame_id"], np.arange(count)):
        raise ValueError("ROBOT_SOURCE_FRAME_MISMATCH")
    if binding.get("execution_round") == "S2":
        return _inspect_s2(binding, binding_path, sid, row, domain, domain_path, motion, r0_path)
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    if scene.get("session_id") != sid or scene.get("frame_count") != row["frame_count"]:
        raise ValueError("SCENE_ID_OR_FRAME_COUNT_MISMATCH")
    if scene.get("image_domain") != domain["image_domain"]:
        raise ValueError("SCENE_IMAGE_DOMAIN_MISMATCH")
    if not scene.get("semantic_ready_for_product"):
        raise ValueError("SCENE_SEMANTIC_INPUT_NOT_READY")
    if len(scene.get("frames", [])) != count:
        raise ValueError("CLEAN_FRAME_COUNT_MISMATCH")
    if not np.array_equal(motion["human_to_physical"], np.asarray([1, 0])):
        raise ValueError("ROBOT_SIDE_MAPPING_MISMATCH")
    for i, frame in enumerate(scene["frames"]):
        if frame.get("frame_id") != i or not Path(frame["clean"]).is_file():
            raise ValueError(f"CLEAN_FRAME_INDEX_OR_PATH_MISMATCH:{i}")
    return {"route": sha(ROUTE), "domain": domain, "domain_ref": sha(domain_path),
            "session": sid, "scene": scene, "scene_ref": sha(scene_path),
            "motion": motion, "robot_ref": sha(r0_path), "binding_ref": sha(binding_path)}


def _read_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    # V5 role PNGs are uint16 instance/role maps whose legitimate IDs can be
    # below 256.  IMREAD_GRAYSCALE down-converts by the high byte and silently
    # turns those pixels into zero; preserve the native integer values.
    value = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if value is None or value.shape != shape:
        raise ValueError(f"S2_MASK_DECODE_OR_DOMAIN:{path}")
    return value > 0


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def render_s2(bound: dict, output: Path) -> dict:
    """Render an S2 product through mandatory depth/ID ownership composition."""
    output.mkdir(parents=False, exist_ok=False)
    ownership_root = output / "ownership"
    ownership_root.mkdir()
    count = len(bound["motion"]["frame_id"])
    width, height = int(bound["domain"]["width"]), int(bound["domain"]["height"])
    shape = (height, width)
    path = output / "robot.mp4"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-threads", "2",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
           "-r", "30", "-i", "pipe:0", "-an", "-c:v", "libx264", "-threads", "2",
           "-preset", "fast", "-crf", "19", "-pix_fmt", "yuv420p", str(path)]
    writer = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    renderer = ProductRobotRenderer(REPO_ROOT, bound["motion"], bound["domain"], include_adapter=True)
    counts: Counter[str] = Counter()
    pixel_counts = []
    adapter_pixels = []
    intrinsics_errors = []
    rows = []
    try:
        for frame in range(count):
            clean = cv2.imread(str(bound["clean_paths"][frame]), cv2.IMREAD_COLOR)
            raw = cv2.imread(str(bound["domain"]["frames"][frame]["rgb"]), cv2.IMREAD_COLOR)
            if clean is None or raw is None or clean.shape[:2] != shape or raw.shape[:2] != shape:
                raise ValueError(f"S2_RGB_DECODE_OR_DOMAIN:{frame}")
            layers = renderer.frame_layers(frame)
            mixed = composite(clean, layers.rgb, layers.alpha)
            paths = bound["mask_paths"][frame]
            object_mask = _read_mask(paths["task_object"], shape)
            removed = np.zeros(shape, dtype=bool)
            for role in ("human_hand", "human_forearm", "capture_device"):
                if role in paths:
                    removed |= _read_mask(paths[role], shape)
            object_rgb, object_source = choose_object_pixels(raw_rgb=raw, raw_visible_mask=object_mask)
            decision = ((cv2.dilate(object_mask.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0)
                        & (cv2.dilate(layers.alpha.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0))
            if bound["geometry_mode"] == "VISIBLE_SURFACE_DEPTH":
                depth = _load_npz(bound["depth_paths"][frame])
                if depth["depth_m"].shape != (480, 640):
                    raise ValueError(f"S2_DEPTH_DOMAIN:{frame}")
                removed_low = cv2.resize(removed.astype(np.uint8), (640, 480),
                                         interpolation=cv2.INTER_NEAREST) > 0
                removed_low = cv2.dilate(removed_low.astype(np.uint8),
                                         np.ones((3, 3), np.uint8)) > 0
                scene_low, valid_low = exclude_removed_foreground_depth(
                    scene_depth_m=depth["depth_m"], scene_depth_valid=depth["valid"].astype(bool),
                    human_equipment_mask=removed_low)
                scene_depth = cv2.resize(scene_low.astype(np.float32), (width, height),
                                         interpolation=cv2.INTER_NEAREST)
                scene_valid = cv2.resize(valid_low.astype(np.uint8), (width, height),
                                         interpolation=cv2.INTER_NEAREST) > 0
                disparity = cv2.resize(depth["disparity_physical_left_px"].astype(np.float32),
                                       (width, height), interpolation=cv2.INTER_NEAREST) * 2.0
                k_low = np.asarray(depth["physical_left_intrinsics"], dtype=np.float64)
                k_high = k_low.copy()
                k_high[0, 0] *= 2.0; k_high[1, 1] *= 2.0
                k_high[0, 2] = 2.0 * k_low[0, 2] + 0.5
                k_high[1, 2] = 2.0 * k_low[1, 2] + 0.5
                k_error = float(np.max(np.abs(k_high - renderer.k)))
                intrinsics_errors.append(k_error)
                evidence = quality_evidence(scene_depth, disparity, scene_valid, object_mask, raw)
                ownership = resolve_ownership(
                    human_mask=np.zeros(shape, dtype=bool), object_amodal_mask=object_mask,
                    object_depth_m=scene_depth, object_depth_valid=scene_valid & object_mask,
                    robot_alpha_mask=layers.alpha, robot_depth_m=layers.optical_depth_m,
                    robot_depth_valid=layers.depth_valid, stereo_depth_valid=scene_valid,
                    depth_quality_evidence=evidence, object_rgb=object_rgb,
                    object_pixel_source=object_source, contact_decision_mask=decision).ownership
            else:
                k_error = None
                ownership = np.full(shape, int(Ownership.BACKGROUND), dtype=np.uint8)
                ownership[decision] = int(Ownership.TIE_UNKNOWN)
            scene_front = decision & (ownership == int(Ownership.OBJECT_FRONT))
            robot_front = decision & (ownership == int(Ownership.ROBOT_FRONT))
            unknown = decision & (ownership == int(Ownership.TIE_UNKNOWN))
            mixed[scene_front] = object_rgb[scene_front]
            mixed[robot_front] = layers.rgb[robot_front, ::-1]
            checker = (np.indices(shape).sum(axis=0) // 8) % 2 == 0
            mixed[unknown & checker] = (255, 0, 255)
            mixed[unknown & ~checker] = (25, 25, 25)
            cv2.rectangle(mixed, (0, 0), (width, 34), (20, 20, 20), -1)
            cv2.putText(mixed,
                        f"{bound['session']} {frame:04d} S2 CANDIDATE / VISIBLE-SURFACE OR UNKNOWN",
                        (8, 23), cv2.FONT_HERSHEY_SIMPLEX, .52, (255, 255, 255), 1, cv2.LINE_AA)
            writer.stdin.write(mixed.tobytes())
            ownership_out = np.full(shape, int(Ownership.BACKGROUND), dtype=np.uint8)
            ownership_out[decision] = ownership[decision]
            ownership_path = ownership_root / f"{frame:06d}.png"
            if not cv2.imwrite(str(ownership_path), ownership_out):
                raise RuntimeError(f"S2_OWNERSHIP_WRITE:{frame}")
            counts["SCENE_FRONT"] += int(scene_front.sum())
            counts["ROBOT_FRONT"] += int(robot_front.sum())
            counts["UNKNOWN"] += int(unknown.sum())
            counts["NOT_APPLICABLE"] += int((~decision).sum())
            pixel_counts.append(int(layers.alpha.sum()))
            adapter_pixels.append(int(np.isin(layers.component_id, [2, 3]).sum()))
            rows.append({"frame_id": frame, "scene_front": int(scene_front.sum()),
                         "robot_front": int(robot_front.sum()), "unknown": int(unknown.sum()),
                         "decision_pixels": int(decision.sum()),
                         "robot_pixels": pixel_counts[-1], "adapter_pixels": adapter_pixels[-1],
                         "intrinsics_max_abs_error": k_error, "ownership": sha(ownership_path)})
        writer.stdin.close()
        stderr = writer.stderr.read().decode("utf-8", "replace")
        if writer.wait() != 0:
            raise RuntimeError("FFMPEG_ENCODE_FAILED:" + stderr[-2000:])
    finally:
        renderer.close()
        if writer.poll() is None:
            writer.terminate(); writer.wait()
    capture = cv2.VideoCapture(str(path)); decoded = 0
    while capture.read()[0]: decoded += 1
    capture.release()
    if decoded != count:
        raise ValueError(f"VIDEO_FULL_DECODE_MISMATCH:{decoded}/{count}")
    decision_total = counts["SCENE_FRONT"] + counts["ROBOT_FRONT"] + counts["UNKNOWN"]
    quality = "REJECTED_QUALITY" if bound["scene"].get("quality") != "PASS" else "INCONCLUSIVE"
    result = {
        "schema_version": "chaoyang-human-to-robot-product-s2-v1",
        "route_id": "HUMAN_TO_ROBOT_BASELINE_V1", "execution_round": "S2",
        "status": "EXECUTED_PENDING_VISUAL_REVIEW",
        "session_id": bound["session"], "execution": "EXECUTED", "structure": "PASS",
        "quality": quality, "adoption": "CANDIDATE_ONLY",
        "product_video": sha(path), "decoded_frames": decoded, "expected_frames": count,
        "robot_pixel_p50": float(np.median(pixel_counts)),
        "adapter_visible_frames": int(np.count_nonzero(adapter_pixels)),
        "ownership_counts": dict(counts),
        "known_decision_coverage": ((counts["SCENE_FRONT"] + counts["ROBOT_FRONT"])
                                    / decision_total if decision_total else 0.0),
        "unknown_decision_ratio": counts["UNKNOWN"] / decision_total if decision_total else 1.0,
        "intrinsics_max_abs_error": max(intrinsics_errors) if intrinsics_errors else None,
        "geometry_mode": bound["geometry_mode"], "rows": rows,
        "scene_clean_manifest": bound["scene_ref"], "robot_r0": bound["robot_ref"],
        "domain_manifest": bound["domain_ref"], "binding": bound["binding_ref"],
        "route_manifest": bound["route"], "mount_contract": bound["mount_ref"],
        "side_mapping_evidence": bound["side_mapping_ref"],
        "pipeline_signature": pipeline_signature(),
        "renderer_interface": bound["binding"]["renderer_interface"],
        "compositor_interface": bound["binding"]["compositor_interface"],
        "adapter_collision_scope": bound["binding"]["adapter_collision_scope"],
        "occlusion_scope": ("DIRECT_VISIBLE_OBJECT_SURFACE_ONLY"
                            if bound["geometry_mode"] == "VISIBLE_SURFACE_DEPTH"
                            else "UNKNOWN_NO_QUALIFIED_DEPTH"),
        "occlusion_epsilon_m": 0.003,
        "claim_limit": "Offline rejected-scene candidate; UNKNOWN is not correct and no hidden geometry, Contact, external accuracy, control or deployment authority is claimed.",
        "training_eligible": False, "control_ground_truth": False,
        "physical_deployable": False, "external_metric_authority": False,
    }
    with (output / "PRODUCT_RESULT.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2); stream.write("\n")
    return result


def render(bound: dict, output: Path) -> dict:
    output.mkdir(parents=False, exist_ok=False)
    count = len(bound["motion"]["frame_id"])
    width, height = bound["domain"]["width"], bound["domain"]["height"]
    path = output / "robot.mp4"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-threads", "2",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
           "-r", "30", "-i", "pipe:0", "-an", "-c:v", "libx264", "-threads", "2",
           "-preset", "fast", "-crf", "19", "-pix_fmt", "yuv420p", str(path)]
    writer = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    renderer = ProductRobotRenderer(REPO_ROOT, bound["motion"], bound["domain"])
    pixels = []
    try:
        for i, frame in enumerate(bound["scene"]["frames"]):
            clean = cv2.imread(frame["clean"], cv2.IMREAD_COLOR)
            if clean is None or clean.shape[:2] != (height, width):
                raise ValueError(f"CLEAN_DECODE_OR_DOMAIN:{i}")
            rgb, mask = renderer.frame(i)
            mixed = composite(clean, rgb, mask)
            cv2.rectangle(mixed, (0, 0), (width, 32), (20, 20, 20), -1)
            cv2.putText(mixed, f"{bound['session']}  frame {i:04d}  OFFLINE_VISUAL / NOT_FOR_TRAINING / UNKNOWN_OCCLUSION",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .54, (255, 255, 255), 1, cv2.LINE_AA)
            writer.stdin.write(mixed.tobytes())
            pixels.append(int(mask.sum()))
        writer.stdin.close()
        stderr = writer.stderr.read().decode("utf-8", "replace")
        if writer.wait() != 0:
            raise RuntimeError("FFMPEG_ENCODE_FAILED:" + stderr[-2000:])
    finally:
        renderer.close()
        if writer.poll() is None:
            writer.terminate()
            writer.wait()
    capture = cv2.VideoCapture(str(path))
    decoded = 0
    while capture.read()[0]:
        decoded += 1
    capture.release()
    if decoded != count:
        raise ValueError(f"VIDEO_FULL_DECODE_MISMATCH:{decoded}/{count}")
    result = {
        "schema_version": SCHEMA, "route_id": "HUMAN_TO_ROBOT_BASELINE_V1",
        "session_id": bound["session"], "status": "EXECUTED_PENDING_VISUAL_REVIEW",
        "product_video": sha(path), "decoded_frames": decoded, "expected_frames": count,
        "robot_pixel_p50": float(np.median(pixels)),
        "scene_clean_manifest": bound["scene_ref"], "robot_r0": bound["robot_ref"],
        "domain_manifest": bound["domain_ref"], "binding": bound["binding_ref"],
        "route_manifest": bound["route"],
        "camera_intrinsics_method": renderer.intrinsics_method,
        "scene_visual_quality_adopted": bool(bound["scene"].get("visual_quality_adopted", False)),
        "motion_legacy_candidate": bool(bound["motion"]["legacy_candidate"]),
        "occlusion_policy": "ROBOT_OVER_CLEAN_UNKNOWN_OBJECT_DEPTH_WATERMARKED",
        "training_eligible": False, "control_ground_truth": False,
        "claim_limit": "Offline visual only; unknown occlusion and unverified camera registration cannot establish contact or physical accuracy.",
    }
    with (output / "PRODUCT_RESULT.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--motion-source", choices=("hawor", "controller_manus"), required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        bound = inspect(args.input.resolve(), args.motion_source, args.config)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        print(json.dumps({"status": "BLOCKED_PREREQ", "reason": str(exc)}, ensure_ascii=False))
        return 2
    output = args.output.resolve()
    if args.dry_run:
        print(json.dumps({"status": "READY", "session_id": bound["session"],
                          "frame_count": len(bound["motion"]["frame_id"]),
                          "execution_round": bound.get("execution_round", "LEGACY"),
                          "scene_quality_adopted": bool(bound["scene"].get("visual_quality_adopted", False))},
                         ensure_ascii=False))
        return 0
    if output.exists():
        result_path = output / "PRODUCT_RESULT.json"
        if args.resume and result_path.is_file():
            old = json.loads(result_path.read_text(encoding="utf-8"))
            pipeline_ok = (bound.get("execution_round") != "S2"
                           or old.get("pipeline_signature") == pipeline_signature())
            if (old.get("binding") == bound["binding_ref"]
                    and old.get("product_video") == sha(output / "robot.mp4")
                    and pipeline_ok):
                print(json.dumps({"status": "REUSED", "result": str(result_path)}))
                return 0
        raise ValueError("OUTPUT_EXISTS_OR_INCOMPLETE")
    result = render_s2(bound, output) if bound.get("execution_round") == "S2" else render(bound, output)
    print(json.dumps({"status": result.get("status", result.get("execution", "EXECUTED")),
                      "video": result["product_video"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
