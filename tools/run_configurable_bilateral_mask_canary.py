#!/usr/bin/env python3
"""Configurable fresh SAM3.1 bilateral hand/tracker Mask canary.

This runner preserves the proven assisted bilateral flow-refresh algorithm but
removes every legacy 0901 frame/anchor/object-root binding.  Its first use is a
short contiguous HAND+TRACKER canary.  Object protection is deliberately
pending, so even a hand/tracker PASS remains ineligible for Clean.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from tools import run_assisted_bilateral_flow_refresh as flow  # noqa: E402
from tools import run_chips001_pico_mask_temporal as base  # noqa: E402
from tools import run_newtask_baseline_sam31_mask_probe as baseline  # noqa: E402


SCHEMA_VERSION = "configurable-assisted-bilateral-mask-canary-v1"
CONFIG_SCHEMA_VERSION_V1 = "configurable-assisted-bilateral-mask-canary-config-v1"
CONFIG_SCHEMA_VERSION_V2 = "configurable-assisted-bilateral-mask-canary-config-v2"
FONT = Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc")
ROLE_COLORS_BGR = {
    "left_human": (80, 220, 80),
    "right_human": (40, 150, 255),
    "left_tracker": (255, 230, 20),
    "right_tracker": (220, 40, 240),
}


class CanaryError(RuntimeError):
    pass


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_ref(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        raise CanaryError(f"regular non-symlink file required: {resolved}")
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": sha256(resolved)}


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise CanaryError(f"cannot parse JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise CanaryError("config root must be an object")
    return value


def write_json_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def verify_ref(value: Any, label: str) -> Path:
    if not isinstance(value, dict) or set(value) != {"path", "bytes", "sha256"}:
        raise CanaryError(f"{label}: exact path/bytes/sha256 reference required")
    path = Path(str(value["path"]))
    if not path.is_absolute() or path.resolve(strict=True) != path:
        raise CanaryError(f"{label}: canonical absolute path required")
    observed = file_ref(path)
    if observed != value:
        raise CanaryError(f"{label}: bytes/SHA drift")
    return path


def point(value: Any, width: int, height: int, label: str) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise CanaryError(f"{label}: [x,y] required")
    x, y = (int(value[0]), int(value[1]))
    if not (0 <= x < width and 0 <= y < height):
        raise CanaryError(f"{label}: point outside {width}x{height}")
    return x, y


def validate_config(path: Path, output_root: Path) -> dict[str, Any]:
    config_path = path.resolve(strict=True)
    config = load_json(config_path)
    required = {
        "schema_version", "task_id", "session_id", "raw_all_data",
        "raw_video", "hawor_npz", "frame_indices", "anchor_slot", "fps",
        "human_anchors", "tracker_roles", "tracker_anatomical",
        "tracker_human_internal", "refresh_cadence_frames",
        "object_protection", "review_language",
    }
    schema = config.get("schema_version")
    if schema == CONFIG_SCHEMA_VERSION_V2:
        required.add("tracker_prompt_contract")
    if set(config) != required or schema not in {CONFIG_SCHEMA_VERSION_V1, CONFIG_SCHEMA_VERSION_V2}:
        raise CanaryError(f"config keys/schema mismatch: {sorted(set(config) ^ required)}")
    if schema == CONFIG_SCHEMA_VERSION_V2 and config["tracker_prompt_contract"] != "SAME_ID_POSITIVE_BOOTSTRAP_THEN_FULL_REFINE":
        raise CanaryError("V2 tracker prompt contract must bootstrap and refine the same tracker id")
    if config["task_id"] not in {"chips", "poker"}:
        raise CanaryError("task_id must be chips or poker")
    if config["review_language"] != "zh-CN":
        raise CanaryError("Chinese review language is mandatory")
    if config["object_protection"] != {
        "status": "PENDING_PARALLEL_DEPTH_OBJECT6D_LANE",
        "mask_source": None,
        "clean_authorized": False,
    }:
        raise CanaryError("object protection must remain fail-closed pending")
    if not output_root.is_absolute() or output_root.exists() or output_root.is_symlink():
        raise CanaryError(f"absolute fresh output root required: {output_root}")
    output_root.parent.resolve(strict=True)

    raw_root = Path(str(config["raw_all_data"]))
    if not raw_root.is_absolute() or raw_root.resolve(strict=True) != raw_root or not raw_root.is_dir():
        raise CanaryError("raw_all_data must be an existing canonical directory")
    raw_video = verify_ref(config["raw_video"], "raw_video")
    hawor_path = verify_ref(config["hawor_npz"], "hawor_npz")
    indices = [int(value) for value in config["frame_indices"]]
    if len(indices) != 12 or len(set(indices)) != 12 or indices != list(range(indices[0], indices[0] + 12)):
        raise CanaryError("first canary must contain exactly 12 contiguous frames")
    anchor_slot = int(config["anchor_slot"])
    if not 0 <= anchor_slot < len(indices):
        raise CanaryError("anchor_slot outside canary")
    cadence = int(config["refresh_cadence_frames"])
    if cadence < 2 or cadence > 6:
        raise CanaryError("canary refresh cadence must be 2..6 frames")
    fps = float(config["fps"])
    if not np.isfinite(fps) or fps <= 0:
        raise CanaryError("invalid fps")

    frame_paths = []
    for index in indices:
        frame_path = (raw_root / f"{index:05d}" / "rgb.png").resolve(strict=True)
        if not frame_path.is_file() or frame_path.is_symlink():
            raise CanaryError(f"bad source frame: {frame_path}")
        frame_paths.append(frame_path)
    first = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    if first is None or first.shape[:2] != (960, 1280):
        raise CanaryError("source frames must decode as 1280x960")
    height, width = first.shape[:2]
    for frame_path in frame_paths[1:]:
        image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (height, width):
            raise CanaryError("frame decode/geometry drift")

    capture = cv2.VideoCapture(str(raw_video))
    if not capture.isOpened():
        raise CanaryError("raw video cannot be opened")
    video_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    video_geometry = (
        int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))),
        int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
    )
    capture.release()
    if video_count <= indices[-1] or video_geometry != (width, height):
        raise CanaryError("raw video coverage/geometry mismatch")

    with np.load(hawor_path, allow_pickle=False) as archive:
        needed = {"joints_2d", "original_frame_indices", "anatomical_side_names", "mano_wrist_index"}
        if not needed.issubset(archive.files):
            raise CanaryError("HaWoR NPZ lacks required arrays")
        joints = archive["joints_2d"]
        original = archive["original_frame_indices"].astype(int)
        sides = archive["anatomical_side_names"].astype(str).tolist()
        if joints.ndim != 4 or joints.shape[0] != 2 or joints.shape[2:] != (21, 2):
            raise CanaryError(f"unexpected joints_2d shape: {joints.shape}")
        if sides != ["left", "right"] or int(archive["mano_wrist_index"]) != 0:
            raise CanaryError("HaWoR anatomy/wrist contract mismatch")
        if not set(indices).issubset(set(original.tolist())):
            raise CanaryError("canary frames are absent from HaWoR original_frame_indices")

    human = config["human_anchors"]
    if set(human) != {"left_human", "right_human"}:
        raise CanaryError("both anatomical human anchors required")
    for name, value in human.items():
        point(value, width, height, name)
    tracker_roles = config["tracker_roles"]
    if not isinstance(tracker_roles, dict) or not tracker_roles:
        raise CanaryError("at least one physically expected tracker role required")
    if not set(tracker_roles).issubset({"upper_tracker_wearable", "lower_tracker_wearable"}):
        raise CanaryError("tracker roles must use upper/lower internal names")
    seen_ids = set()
    for name, role in tracker_roles.items():
        if set(role) != {"id", "name", "positive", "negative"} or role["name"] != name:
            raise CanaryError(f"malformed tracker role {name}")
        if int(role["id"]) in seen_ids:
            raise CanaryError("tracker SAM IDs must be distinct")
        seen_ids.add(int(role["id"]))
        if not role["positive"] or not role["negative"]:
            raise CanaryError(f"tracker role {name} needs positive and negative points")
        for slot, value in enumerate(role["positive"]):
            point(value, width, height, f"{name}.positive[{slot}]")
        for slot, value in enumerate(role["negative"]):
            point(value, width, height, f"{name}.negative[{slot}]")
    anatomical = config["tracker_anatomical"]
    if set(anatomical) != set(tracker_roles) or not set(anatomical.values()).issubset({"left_tracker", "right_tracker"}):
        raise CanaryError("tracker_anatomical mapping mismatch")
    internal = config["tracker_human_internal"]
    if set(internal) != {"upper_human_core", "lower_human_core"} or set(internal.values()) != {"left_human", "right_human"}:
        raise CanaryError("tracker_human_internal mapping mismatch")
    return {
        "config": config,
        "config_ref": file_ref(config_path),
        "raw_root": raw_root,
        "raw_video": raw_video,
        "hawor_path": hawor_path,
        "frame_paths": frame_paths,
        "indices": indices,
        "anchor_slot": anchor_slot,
        "fps": fps,
        "width": width,
        "height": height,
        "video_frame_count": video_count,
    }


def track_roles(
    model: Any,
    output: Path,
    frame_paths: list[Path],
    human: dict[str, dict[int, np.ndarray]],
    config: dict[str, Any],
    height: int,
    width: int,
) -> tuple[dict[str, dict[int, np.ndarray]], dict[str, Any]]:
    protocol = base.tracker_protocol()
    protocol["wearables"]["refresh_cadence_frames"] = int(config["refresh_cadence_frames"])
    anchor_slot = int(config["anchor_slot"])
    merged = {name: {} for name in config["tracker_anatomical"].values()}
    metadata: dict[str, Any] = {}
    object_masks = [np.zeros((height, width), bool) for _ in frame_paths]
    for direction, originals in {
        "forward": list(range(anchor_slot, len(frame_paths))),
        "reverse": list(range(anchor_slot, -1, -1)),
    }.items():
        view, paths = base.materialize_view(output / "tracker_views" / direction, originals)
        human_sequence = {
            internal: {slot: human[anatomical][original] for slot, original in enumerate(originals)}
            for internal, anatomical in config["tracker_human_internal"].items()
        }
        cache = flow.FlowCache(paths, 640, 480)
        for internal_name, role in config["tracker_roles"].items():
            stream, meta = flow.collect_flow_refresh_role(
                model,
                role,
                {
                    "skip_discarded_priming": True,
                    "tracker_prompt_contract": config.get("tracker_prompt_contract", "SINGLE_FULL_PROMPT"),
                },
                view,
                paths,
                human_sequence,
                [object_masks[index] for index in originals],
                cache,
                protocol,
                height,
                width,
                output / "tracker_refresh" / direction,
            )
            anatomical = config["tracker_anatomical"][internal_name]
            for slot, mask in stream.items():
                merged[anatomical][originals[slot]] = mask
            metadata[f"{direction}_{anatomical}"] = meta
    return merged, metadata


def mask_containment(mask: np.ndarray, points: np.ndarray) -> tuple[int, int, float]:
    height, width = mask.shape
    finite = np.isfinite(points).all(axis=1)
    rounded = np.rint(points).astype(int)
    inside = finite & (rounded[:, 0] >= 0) & (rounded[:, 0] < width) & (rounded[:, 1] >= 0) & (rounded[:, 1] < height)
    expected = int(inside.sum())
    contained = int(sum(mask[y, x] for x, y in rounded[inside])) if expected else 0
    return contained, expected, float(contained / expected) if expected else 1.0


def evaluate(
    validated: dict[str, Any],
    human: dict[str, dict[int, np.ndarray]],
    human_meta: dict[str, Any],
    human_completion: dict[str, Any],
    trackers: dict[str, dict[int, np.ndarray]],
    tracker_meta: dict[str, Any],
) -> dict[str, Any]:
    indices = validated["indices"]
    with np.load(validated["hawor_path"], allow_pickle=False) as archive:
        original = archive["original_frame_indices"].astype(int)
        joints = archive["joints_2d"]
        slot_by_original = {int(value): slot for slot, value in enumerate(original)}
        side_by_name = {name: slot for slot, name in enumerate(archive["anatomical_side_names"].astype(str).tolist())}
        human_rows: dict[str, list[dict[str, Any]]] = {}
        for anatomical in ("left", "right"):
            role = f"{anatomical}_human"
            rows = []
            for local, source_index in enumerate(indices):
                mask = human[role].get(local, np.zeros((validated["height"], validated["width"]), bool))
                contained, expected, ratio = mask_containment(
                    mask, joints[side_by_name[anatomical], slot_by_original[source_index]]
                )
                rows.append({
                    "source_frame": source_index,
                    "area_pixels": int(mask.sum()),
                    "joint_contained": contained,
                    "joint_visible_denominator": expected,
                    "joint_containment": ratio,
                    "pass": bool(mask.any() and ratio >= 0.50),
                })
            human_rows[role] = rows
    tracker_rows = {
        role: [
            {
                "source_frame": source_index,
                "area_pixels": int(stream.get(local, np.zeros((validated["height"], validated["width"]), bool)).sum()),
                "present": bool(stream.get(local, np.zeros((validated["height"], validated["width"]), bool)).any()),
            }
            for local, source_index in enumerate(indices)
        ]
        for role, stream in trackers.items()
    }
    human_pass = bool(
        human_meta.get("status") == "COMPLETE"
        and all(value.get("all_frames_present") for value in human_completion.values())
        and all(row["pass"] for rows in human_rows.values() for row in rows)
    )
    tracker_pass = bool(
        tracker_rows
        and all(row["present"] for rows in tracker_rows.values() for row in rows)
        and all(meta.get("status") == "PASS" for meta in tracker_meta.values())
    )
    return {
        "human_pass": human_pass,
        "tracker_pass": tracker_pass,
        "hand_tracker_gate_pass": bool(human_pass and tracker_pass),
        "human_fixed_denominator": human_rows,
        "tracker_fixed_denominator": tracker_rows,
        "tracker_temporal_gates": tracker_meta,
    }


def start_video(path: Path, width: int, height: int, fps: float) -> subprocess.Popen:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.mp4")
    command = [
        "ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-r", f"{fps:.8g}", "-i", "-", "-an",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(temporary),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    process._mask_final_path = path  # type: ignore[attr-defined]
    process._mask_temporary = temporary  # type: ignore[attr-defined]
    return process


def finish_video(process: subprocess.Popen) -> Path:
    assert process.stdin is not None
    process.stdin.close()
    code = process.wait()
    temporary = process._mask_temporary  # type: ignore[attr-defined]
    final = process._mask_final_path  # type: ignore[attr-defined]
    if code != 0:
        raise CanaryError(f"ffmpeg encoder failed: {code}")
    os.replace(temporary, final)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-xerror", "-i", str(final), "-f", "null", "-"],
        check=True,
    )
    return final


def render_and_write_masks(
    output: Path,
    validated: dict[str, Any],
    human: dict[str, dict[int, np.ndarray]],
    trackers: dict[str, dict[int, np.ndarray]],
    gate_pass: bool,
) -> dict[str, Any]:
    role_root = output / "raw_role_masks"
    union_root = output / "hand_tracker_union_masks"
    for role in [*human, *trackers]:
        (role_root / role).mkdir(parents=True, exist_ok=True)
    union_root.mkdir(parents=True, exist_ok=True)
    video = output / f"{validated['config']['session_id']}_HAND_TRACKER_MASK_中文逐帧复核.mp4"
    writer = start_video(video, 1280, 600, validated["fps"])
    font = ImageFont.truetype(str(FONT), 24)
    small = ImageFont.truetype(str(FONT), 18)
    try:
        for local, (source_index, frame_path) in enumerate(zip(validated["indices"], validated["frame_paths"])):
            raw = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            overlay = raw.astype(np.float32)
            union = np.zeros(raw.shape[:2], bool)
            for role, stream in {**human, **trackers}.items():
                mask = stream.get(local, np.zeros(raw.shape[:2], bool))
                cv2.imwrite(str(role_root / role / f"{source_index:05d}.png"), mask.astype(np.uint8) * 255)
                union |= mask
                color = np.asarray(ROLE_COLORS_BGR[role], np.float32)
                overlay[mask] = 0.36 * overlay[mask] + 0.64 * color
            cv2.imwrite(str(union_root / f"{source_index:05d}.png"), union.astype(np.uint8) * 255)
            pair = np.hstack((raw, np.clip(overlay, 0, 255).astype(np.uint8)))
            pair = cv2.resize(pair, (1280, 480), interpolation=cv2.INTER_AREA)
            canvas = np.zeros((600, 1280, 3), np.uint8)
            canvas[120:] = pair
            image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(image)
            status = "手与Tracker自动门通过；物体保护待闭包" if gate_pass else "手或Tracker自动门未通过"
            draw.text((18, 12), f"{validated['config']['task_id'].upper()} | {validated['config']['session_id']}", font=font, fill=(255, 255, 255))
            draw.text((18, 48), f"源帧 {source_index} | 左：原图  右：真实像素Mask叠加", font=small, fill=(230, 230, 230))
            draw.text((18, 78), status, font=small, fill=(255, 220, 80) if gate_pass else (255, 100, 100))
            draw.text((790, 78), "绿左手 橙右手 青左Tracker 紫右Tracker", font=small, fill=(230, 230, 230))
            bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
            assert writer.stdin is not None
            writer.stdin.write(bgr.tobytes())
    except Exception:
        if writer.stdin:
            writer.stdin.close()
        writer.kill()
        raise
    final_video = finish_video(writer)
    return {
        "raw_role_masks": str(role_root),
        "hand_tracker_union_masks": str(union_root),
        "review_video": file_ref(final_video),
        "object_protection_masks": None,
    }


def execute(validated: dict[str, Any], output: Path, holder: str) -> int:
    lease_path = PROJECT / "_run/GPU_LEASE.json"
    lease = load_json(lease_path)
    if lease.get("status") != "ACQUIRED" or lease.get("holder") != holder:
        raise CanaryError("central GPU lease is not ACQUIRED by the exact holder")
    if not torch.cuda.is_available():
        raise CanaryError("CUDA_UNAVAILABLE_IN_CURRENT_CONTAINER")

    config = validated["config"]
    base.RAW = validated["raw_root"]
    base.SOURCE_INDICES = validated["indices"]
    base.FRAME_COUNT = len(validated["indices"])
    base.ANCHOR_FRAME = validated["anchor_slot"]
    base.CANARY = list(range(base.FRAME_COUNT))
    base.TASK_ID = config["task_id"]
    base.SESSION_ID = config["session_id"]
    base.HUMAN_ANCHORS = {name: tuple(value) for name, value in config["human_anchors"].items()}
    base.TRACKER_ROLES = config["tracker_roles"]
    base.TRACKER_ANATOMICAL = config["tracker_anatomical"]
    base.TRACKER_HUMAN_INTERNAL = config["tracker_human_internal"]

    output.mkdir(parents=True)
    frame_root, frame_paths = base.materialize_view(output / "input_frames", list(range(base.FRAME_COUNT)))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    adapter_module = baseline.load_adapter_module()
    # The pinned SAM3.1 source is vendored rather than installed as a site
    # package.  Match the already-validated legacy entry points and make that
    # exact code root importable before the compatibility adapter builds it.
    if str(base.CODE_ROOT) not in sys.path:
        sys.path.insert(0, str(base.CODE_ROOT))
    adapter, build_evidence = adapter_module.build_pinned_adapter(
        official_code_root=base.CODE_ROOT, checkpoint_path=base.CHECKPOINT
    )
    try:
        human_raw, human_meta = base.collect_text_roles(
            adapter.model,
            frame_root,
            "a person's hand and forearm",
            base.ANCHOR_FRAME,
            base.HUMAN_ANCHORS,
            validated["height"],
            validated["width"],
            True,
            28.0,
        )
        cache = flow.FlowCache(frame_paths, 640, 480)
        human, human_completion = base.complete_with_flow(
            human_raw, frame_paths, cache, validated["height"], validated["width"]
        )
        trackers, tracker_meta = track_roles(
            adapter.model, output, frame_paths, human, config,
            validated["height"], validated["width"],
        )
    finally:
        adapter.predictor.shutdown()

    gates = evaluate(validated, human, human_meta, human_completion, trackers, tracker_meta)
    artifacts = render_and_write_masks(
        output, validated, human, trackers, gates["hand_tracker_gate_pass"]
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_at": now(),
        "status": "HOLD_OBJECT_PROTECTION_PENDING" if gates["hand_tracker_gate_pass"] else "HOLD_HAND_TRACKER_GATE",
        "automatic_hand_tracker_gate_pass": gates["hand_tracker_gate_pass"],
        "object_protection_status": "PENDING_PARALLEL_DEPTH_OBJECT6D_LANE",
        "consumption_authorized": False,
        "clean_authorized": False,
        "task": config["task_id"],
        "session": config["session_id"],
        "frame_indices": validated["indices"],
        "anchor_slot": validated["anchor_slot"],
        "method": {
            "human": "SAM31_TEXT_TEMPORAL_TRUE_PIXEL_CONTOUR",
            "tracker": "ASSISTED_SAM31_POINT_MASK_WITH_BIDIRECTIONAL_DIS_AND_PERIODIC_INDEPENDENT_CURRENT_FRAME_SAM_REFRESH",
            "refresh_cadence_frames": config["refresh_cadence_frames"],
            "discarded_object_priming": "SKIPPED_OBJECT_PROTECTION_PENDING",
            "tracker_prompt_contract": config.get("tracker_prompt_contract", "SINGLE_FULL_PROMPT"),
            "skeleton_geometry_used_as_output": False,
            "unsupported_flow_policy": "FAIL_AUTOMATIC_GATE",
            "object_mask": "PENDING_NOT_EMITTED",
        },
        "gates": gates,
        "artifacts": artifacts,
        "pins": {
            "config": validated["config_ref"],
            "runner": file_ref(Path(__file__)),
            "checkpoint_sha256": base.CHECKPOINT_SHA256,
            "raw_video": file_ref(validated["raw_video"]),
            "hawor_npz": file_ref(validated["hawor_path"]),
            "gpu_lease": file_ref(lease_path),
            "sam_build_evidence": build_evidence,
        },
        "resource": {
            "wall_seconds": time.perf_counter() - started,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        },
        "claim_limit": "Fresh 12-frame HAND+TRACKER canary only. Object protection, Clean, full-session expansion, Robot and training remain unauthorized.",
    }
    write_json_new(output / "RESULT.json", result)
    return 0 if gates["hand_tracker_gate_pass"] else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--confirm-config-sha", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--lease-holder")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve(strict=True)
    if sha256(config_path) != args.confirm_config_sha:
        raise CanaryError("config SHA confirmation mismatch")
    validated = validate_config(config_path, args.output_root)
    if args.validate_only:
        receipt = {
            "schema_version": "configurable-assisted-bilateral-mask-canary-preflight-v1",
            "created_at": now(),
            "status": "PASS_CPU_STATIC_VALIDATION_GPU_NOT_STARTED",
            "config": validated["config_ref"],
            "output_root": str(args.output_root),
            "frame_indices": validated["indices"],
            "object_protection_status": "PENDING_PARALLEL_DEPTH_OBJECT6D_LANE",
            "clean_authorized": False,
            "gpu_started": False,
            "runner": file_ref(Path(__file__)),
        }
        if args.receipt:
            write_json_new(args.receipt, receipt)
        print(json.dumps(receipt, ensure_ascii=False))
        return 0
    if not args.lease_holder:
        raise CanaryError("--lease-holder required for GPU execution")
    return execute(validated, args.output_root, args.lease_holder)


if __name__ == "__main__":
    raise SystemExit(main())
