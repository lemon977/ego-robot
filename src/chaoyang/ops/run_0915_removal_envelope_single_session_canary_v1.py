#!/usr/bin/env python3
"""Run the CPU-only, single-session Removal Envelope V1 canary.

The sealed SAM3.1 artifacts are read-only semantic evidence.  This task does
not invoke SAM, an inpainting model, Depth, Object6D, Contact, or Robot code.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable
import uuid

import cv2
import jsonschema
import numpy as np

from chaoyang.governance.campaign_0915_task_specs_v1 import build_packet
from chaoyang.pipeline.removal_envelope_v1 import (
    ALLOWED_CONSUMERS,
    FORBIDDEN_CONSUMERS,
    SOURCE_BITS,
    CableAppearanceProfile,
    RemovalEnvelopeBuilder,
    RemovalEnvelopeConfig,
    bounded_internal_geometry,
)


ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "0915_removal_envelope_single_session_canary_v1"
PHASE = "0915_REMOVAL_ENVELOPE_SINGLE_SESSION_CANARY_V1"
SESSION_ID = "play_cards_0915_001"
FRAME_COUNT = 150
HEIGHT = 960
WIDTH = 1280
BITORDER = "big"
VIDEO = (
    ROOT / "_run/current/0915_hawor_resize_only_canary_v1/attempts/attempt_0001/"
    "input/PHYSICAL_LEFT_SOURCEINDEX1_RESIZE_ONLY.mp4"
)
HAWOR = (
    ROOT / "_run/current/0915_hawor_resize_only_bounded_v2_canary/attempts/"
    "attempt_0001/bounded_output_guarded_identity_fixed/play_cards_0915_001/"
    "HAWOR_BOUNDED_PARAMETER_SUCCESSOR.npz"
)
STRICT_ROOT = ROOT / "_run/current/0915_sam31_strict_role_canary_v5/attempts/attempt_0001"
WEAK_ROOT = ROOT / "_run/current/0915_sam31_weak_role_canary_v1/attempts/attempt_0001"
CONFIG_PATH = ROOT / "configs/systems/clean/removal_envelope_0915_play_cards_001_v1.json"
SCHEMA_PATH = ROOT / "contracts/removal_envelope_v1.schema.json"
OUTPUT_ROOT = ROOT / "_run/current" / TASK_ID / "attempts/attempt_0001"
VISUAL_ROOT = ROOT / "docs/current/visuals/0915_REMOVAL_ENVELOPE_CANARY_V1"
TASK_RECEIPT = ROOT / "tasks/receipts/0915_REMOVAL_ENVELOPE_SINGLE_SESSION_CANARY_V1_RESULT.json"

STRICT_MASKS = {
    "hand": ("left_hand_00", "right_hand_00"),
    "task_object": ("playing_card_00", "playing_card_01"),
}
WEAK_MASKS = {
    "forearm": ("left_forearm_00", "right_forearm_00"),
    "sleeve": (
        "left_finger_sleeve_visible_00", "left_finger_sleeve_visible_01",
        "left_finger_sleeve_visible_02", "left_finger_sleeve_visible_03",
        "right_finger_sleeve_visible_00", "right_finger_sleeve_visible_01",
        "right_finger_sleeve_visible_02", "right_finger_sleeve_visible_03",
    ),
    "cable": ("left_yellow_cable_visible_00", "right_yellow_cable_visible_00"),
    "task_object": ("playing_card_02",),
}
FIXED_REVIEW_FRAMES = (
    0, 16, 23, 24, 30, 35, 36, 44, 45, 60, 65, 67,
    89, 90, 93, 103, 114, 115, 137, 138, 141, 142, 149,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    candidate = path.resolve(strict=True)
    return {
        "path": str(candidate), "bytes": candidate.stat().st_size,
        "sha256": sha256(candidate),
    }


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def validate_route() -> dict[str, Any]:
    packet_path = ROOT / f"tasks/current/{TASK_ID}/TASK_PACKET.json"
    packet = load_json(packet_path)
    if packet != build_packet(TASK_ID):
        raise RuntimeError("current task packet differs from frozen specification")
    if packet.get("weights") != "ABSENT":
        raise RuntimeError("Removal Envelope V1 must not bind model weights")
    state = load_json(ROOT / "docs/governance/LONG_HORIZON_TASK_STATE.json")
    if state.get("next_task", {}).get("task_id") != TASK_ID:
        raise RuntimeError("Removal Envelope task is not current next_task")
    task = next((row for row in state.get("tasks", []) if row.get("task_id") == TASK_ID), None)
    if task is None or task.get("status") not in {"PENDING", "READY", "CLAIMED", "RUNNING"}:
        raise RuntimeError("Removal Envelope task is not executable")
    index = load_json(ROOT / "tasks/current/INDEX.json")
    route = next((row for row in index.get("task_packets", []) if row.get("task_id") == TASK_ID), None)
    if route is None or route.get("execution_allowed") is not True:
        raise RuntimeError("task index does not authorize Removal Envelope task")
    if route.get("packet_sha256") != sha256(packet_path):
        raise RuntimeError("task packet SHA mismatch")
    return packet


def heartbeat(pid: int) -> None:
    completed = subprocess.run(
        [
            sys.executable, "-m", "chaoyang.governance.heartbeat_task",
            "--task-id", TASK_ID, "--pid", str(pid),
            "--status", "RUNNING", "--phase", PHASE,
        ],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            "governance heartbeat failed: " + (completed.stderr or completed.stdout)[-4000:]
        )


def load_config() -> RemovalEnvelopeConfig:
    raw = load_json(CONFIG_PATH)
    if raw.get("schema_version") != "removal-envelope-config-v1":
        raise RuntimeError("Removal config schema drift")
    cable = dict(raw["cable_profile"])
    profile = CableAppearanceProfile(
        profile_id=str(cable["profile_id"]),
        hsv_lower=tuple(int(value) for value in cable["hsv_lower"]),
        hsv_upper=tuple(int(value) for value in cable["hsv_upper"]),
        min_component_pixels=int(cable["min_component_pixels"]),
        max_component_pixels=int(cable["max_component_pixels"]),
        hand_anchor_radius_px=int(cable["hand_anchor_radius_px"]),
        previous_track_radius_px=int(cable["previous_track_radius_px"]),
        output_radius_px=int(cable["output_radius_px"]),
        maximum_hold_frames=int(cable["maximum_hold_frames"]),
    )
    fields = {
        key: value for key, value in raw.items()
        if key not in {
            "schema_version", "session_scope", "image_domain", "cable_profile",
            "consumer_firewall",
        }
    }
    return RemovalEnvelopeConfig(cable_profile=profile, **fields)


def _load_packed(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        packed = np.asarray(archive["packed"], np.uint8)
        metadata = (
            int(archive["frame_count"]), int(archive["height"]),
            int(archive["width"]), str(archive["bitorder"]),
        )
    if metadata != (FRAME_COUNT, HEIGHT, WIDTH, BITORDER):
        raise RuntimeError(f"packed-mask domain drift: {path}")
    if packed.shape != (FRAME_COUNT, (HEIGHT * WIDTH + 7) // 8):
        raise RuntimeError(f"packed-mask byte shape drift: {path}")
    return packed


def _unpack(row: np.ndarray) -> np.ndarray:
    return np.unpackbits(row, bitorder=BITORDER, count=HEIGHT * WIDTH).reshape(HEIGHT, WIDTH).astype(bool)


def _pack(mask: np.ndarray) -> np.ndarray:
    return np.packbits(np.asarray(mask, bool).reshape(-1), bitorder=BITORDER)


def union_archives(paths: Iterable[Path]) -> np.ndarray:
    merged = np.zeros((FRAME_COUNT, (HEIGHT * WIDTH + 7) // 8), np.uint8)
    for path in paths:
        merged |= _load_packed(path)
    return merged


def strict_mask(name: str) -> Path:
    return STRICT_ROOT / "masks" / f"{name}.npz"


def weak_mask(name: str) -> Path:
    return WEAK_ROOT / "masks" / f"{name}.npz"


def input_paths() -> list[Path]:
    result = [
        VIDEO, HAWOR, CONFIG_PATH, SCHEMA_PATH,
        Path(__file__).resolve(),
        ROOT / "src/chaoyang/pipeline/removal_envelope_v1.py",
        STRICT_ROOT / "ROLE_MANIFEST.json",
        STRICT_ROOT / "TEMPORAL_STATE_LEDGER.json",
        WEAK_ROOT / "WEAK_ROLE_MANIFEST.json",
        WEAK_ROOT / "TEMPORAL_STATE_LEDGER.json",
        WEAK_ROOT / "QUALITY_TRIGGER_LEDGER.json",
    ]
    for names in STRICT_MASKS.values():
        result.extend(strict_mask(name) for name in names)
    for names in WEAK_MASKS.values():
        result.extend(weak_mask(name) for name in names)
    return result


def snapshot_inputs() -> dict[str, dict[str, Any]]:
    return {str(path.resolve(strict=True)): ref(path) for path in input_paths()}


def save_packed(path: Path, packed: np.ndarray, *, artifact_class: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}.npz")
    np.savez_compressed(
        temporary,
        packed=np.asarray(packed, np.uint8),
        frame_count=np.asarray(FRAME_COUNT, np.int32),
        height=np.asarray(HEIGHT, np.int32),
        width=np.asarray(WIDTH, np.int32),
        bitorder=np.asarray(BITORDER),
        artifact_class=np.asarray(artifact_class),
        authority=np.asarray("VISUAL_CLEAN_ONLY_INFERRED_ERASE_SUPPORT"),
    )
    os.replace(temporary, path)


def overlay(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> None:
    if not mask.any():
        return
    paint = np.empty_like(image)
    paint[:] = color
    image[mask] = np.clip(
        image[mask].astype(np.float32) * (1.0 - alpha)
        + paint[mask].astype(np.float32) * alpha,
        0, 255,
    ).astype(np.uint8)


def title(image: np.ndarray, lines: list[str]) -> None:
    height = 30 + len(lines) * 27
    cv2.rectangle(image, (0, 0), (image.shape[1] - 1, height), (18, 18, 18), -1)
    for index, line in enumerate(lines):
        cv2.putText(
            image, line, (12, 25 + index * 27), cv2.FONT_HERSHEY_SIMPLEX,
            0.58, (245, 245, 245), 1, cv2.LINE_AA,
        )


def render_review_frame(
    frame: np.ndarray,
    semantic: dict[str, np.ndarray],
    result: dict[str, Any],
    frame_index: int,
) -> np.ndarray:
    first = frame.copy()
    for name, color in (
        ("hand", (255, 180, 30)), ("forearm", (0, 210, 255)),
        ("sleeve", (230, 60, 230)), ("cable", (0, 160, 255)),
    ):
        overlay(first, semantic[name], color, 0.42)
    contours, _ = cv2.findContours(
        semantic["task_object"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(first, contours, -1, (30, 240, 70), 3)
    title(first, [
        f"frame {frame_index:03d} | sealed semantic evidence",
        "hand cyan | forearm orange | optional sleeve magenta | SAM cable yellow | object green outline",
    ])

    second = frame.copy()
    overlay(second, result["mano_finger_capsules"], (255, 120, 30), 0.62)
    overlay(second, result["palm_wrist_forearm_corridors"], (180, 50, 220), 0.42)
    states = " | ".join(
        f"{row['side']}={row['state']}" for row in result["provenance"]["hands"]
    )
    title(second, [
        f"frame {frame_index:03d} | MANO removal geometry",
        states,
        "blue=finger capsules | violet=palm/wrist/forearm corridor",
    ])

    third = frame.copy()
    overlay(third, result["cable_tracked_region"], (0, 225, 255), 0.70)
    overlay(third, result["protected_visible_object"], (30, 230, 50), 0.42)
    cable = result["provenance"]["cable"]
    title(third, [
        f"frame {frame_index:03d} | cable + visible object core",
        f"cable={cable['state']} profile={cable['appearance_profile_id']} components={cable['accepted_components']}",
        "yellow=cable removal inference | green=protected visible object core",
    ])

    fourth = frame.copy()
    alpha = np.asarray(result["feather_alpha"], np.float32)
    feather_only = (alpha > 0.02) & ~result["removal_envelope"]
    overlay(fourth, feather_only, (30, 150, 255), 0.34)
    overlay(fourth, result["removal_envelope"], (20, 20, 235), 0.62)
    metrics = result["provenance"]
    title(fourth, [
        f"frame {frame_index:03d} | final visual-only removal envelope",
        f"remove={metrics['component_pixels']['removal_envelope']} px | object-overlap-before-protect={metrics['erase_candidate_task_object_overlap']} px",
        "red=remove | orange=feather band | NEVER geometry/contact evidence",
    ])
    return np.vstack((np.hstack((first, second)), np.hstack((third, fourth))))


def open_video_writer(path: Path) -> subprocess.Popen[bytes]:
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "2560x1920",
        "-r", "30", "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium",
        "-crf", "24", "-pix_fmt", "yuv420p", str(path),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def validate_video(path: Path, expected_frames: int) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    count = 0
    shape = None
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        count += 1
        shape = [int(frame.shape[1]), int(frame.shape[0])]
    capture.release()
    if count != expected_frames:
        raise RuntimeError(f"review decode count {count} != {expected_frames}")
    return {"full_decode": True, "frames": count, "resolution": shape, "fps": 30}


def save_contact_sheet(
    previews: list[np.ndarray], frame_rows: list[dict[str, Any]], destination: Path,
) -> list[int]:
    jumps = sorted(
        range(1, len(frame_rows)),
        key=lambda index: frame_rows[index]["removal_area_relative_jump"], reverse=True,
    )[:5]
    overlaps = sorted(
        range(len(frame_rows)),
        key=lambda index: frame_rows[index]["erase_candidate_task_object_overlap"], reverse=True,
    )[:5]
    selected = sorted(set(FIXED_REVIEW_FRAMES) | set(jumps) | set(overlaps))
    thumbnails: list[np.ndarray] = []
    for index in selected:
        thumb = previews[index].copy()
        cv2.putText(thumb, f"frame {index:03d}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2, cv2.LINE_AA)
        thumbnails.append(thumb)
    columns = 5
    rows = (len(thumbnails) + columns - 1) // columns
    blank = np.full_like(thumbnails[0], 24)
    thumbnails.extend(blank.copy() for _ in range(rows * columns - len(thumbnails)))
    sheet = np.vstack([
        np.hstack(thumbnails[row * columns:(row + 1) * columns]) for row in range(rows)
    ])
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), sheet):
        raise RuntimeError("contact sheet write failed")
    return selected


def run(args: argparse.Namespace) -> int:
    packet = validate_route()
    output = args.output_root.resolve()
    visual = args.visual_root.resolve()
    receipt_path = args.task_receipt.resolve()
    if output != OUTPUT_ROOT.resolve() or visual != VISUAL_ROOT.resolve() or receipt_path != TASK_RECEIPT.resolve():
        raise RuntimeError("Removal canary output roots must match the fixed task namespace")
    for path in (output, visual, receipt_path):
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"fresh immutable output required: {path}")
    if args.executor_epoch < 1 or len(args.fencing_token) < 16:
        raise RuntimeError("positive executor epoch and fencing token >=16 chars required")
    output.mkdir(parents=True)
    visual.mkdir(parents=True)
    started = time.time()
    before = snapshot_inputs()
    config = load_config()
    run_signature = {
        "schema_version": "0915-removal-envelope-run-signature-v1",
        "task_id": TASK_ID,
        "session_id": SESSION_ID,
        "weights": "ABSENT",
        "input_refs": list(before.values()),
        "code_ref": ref(Path(__file__).resolve()),
        "config_ref": ref(CONFIG_PATH),
        "schema_ref": ref(SCHEMA_PATH),
        "executor_epoch": args.executor_epoch,
    }
    signature_sha = hashlib.sha256(json.dumps(
        run_signature, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    atomic_json(output / "RUN_SIGNATURE.json", {**run_signature, "run_signature_sha256": signature_sha})
    atomic_json(output / "CLAIM.json", {
        "schema_version": "0915-removal-envelope-claim-v1",
        "task_id": TASK_ID, "pid": os.getpid(),
        "proc_start_ticks": int(Path(f"/proc/{os.getpid()}/stat").read_text().split()[21]),
        "executor_epoch": args.executor_epoch,
        "fencing_token_sha256": hashlib.sha256(args.fencing_token.encode()).hexdigest(),
        "run_signature_sha256": signature_sha,
        "task_packet": ref(ROOT / f"tasks/current/{TASK_ID}/TASK_PACKET.json"),
    })
    heartbeat(os.getpid())

    semantic_packed = {
        "hand": union_archives(strict_mask(name) for name in STRICT_MASKS["hand"]),
        "forearm": union_archives(weak_mask(name) for name in WEAK_MASKS["forearm"]),
        "sleeve": union_archives(weak_mask(name) for name in WEAK_MASKS["sleeve"]),
        "cable": union_archives(weak_mask(name) for name in WEAK_MASKS["cable"]),
        "task_object": (
            union_archives(strict_mask(name) for name in STRICT_MASKS["task_object"])
            | union_archives(weak_mask(name) for name in WEAK_MASKS["task_object"])
        ),
    }
    with np.load(HAWOR, allow_pickle=False) as archive:
        joints = np.asarray(archive["joints_2d"], np.float64)
        observed = np.asarray(archive["observed"], bool)
        confidence = np.asarray(archive["detector_confidence"], np.float64)
        side_names = np.asarray(archive["anatomical_side_names"]).astype(str).tolist()
        joint_names = np.asarray(archive["mano_joint_names"]).astype(str).tolist()
    if joints.shape != (2, FRAME_COUNT, 21, 2) or observed.shape != (2, FRAME_COUNT):
        raise RuntimeError("HaWoR frame or MANO21 axes drift")
    if side_names != ["left", "right"] or joint_names[0] != "wrist":
        raise RuntimeError("HaWoR side/joint identity drift")
    geometry_joints, geometry_states = bounded_internal_geometry(
        joints, observed, confidence,
        maximum_gap_frames=config.maximum_geometry_hold_frames,
        endpoint_confidence_minimum=config.hold_endpoint_confidence_minimum,
        maximum_joint_step_px=config.hold_maximum_joint_step_px,
    )

    packed_width = (HEIGHT * WIDTH + 7) // 8
    outputs = {
        name: np.zeros((FRAME_COUNT, packed_width), np.uint8)
        for name in (
            "MANO_FINGER_CAPSULES", "PALM_WRIST_FOREARM_CORRIDORS",
            "CABLE_TRACKED_REGION", "PROTECTED_VISIBLE_OBJECT", "REMOVAL_ENVELOPE",
        )
    }
    source_bits_dir = output / "removal/source_bits"
    feather_dir = output / "clean/feather_alpha"
    source_bits_dir.mkdir(parents=True)
    feather_dir.mkdir(parents=True)
    atomic_json(output / "raw_candidate/STATUS.json", {
        "schema_version": "removal-envelope-raw-candidate-status-v1",
        "status": "ABSENT_UPSTREAM_NOT_PERSISTED",
        "reason": "The sealed weak-role SAM3.1 run published admitted masks and raw metrics, not raw candidate pixels.",
        "reconstruction_forbidden": True,
    })
    atomic_json(output / "semantic/SEALED_INPUTS.json", {
        "schema_version": "removal-envelope-semantic-inputs-v1",
        "status": "READ_ONLY_SEALED_INPUTS",
        "artifacts": list(before.values()),
        "semantic_status": "REJECTED_QUALITY_AS_CLEAN_BASELINE",
    })

    review_path = visual / "0915_REMOVAL_ENVELOPE_REVIEW.mp4"
    process = open_video_writer(review_path)
    assert process.stdin is not None
    capture = cv2.VideoCapture(str(VIDEO))
    if (
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) != WIDTH
        or int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) != HEIGHT
    ):
        raise RuntimeError("source video domain drift")
    builder = RemovalEnvelopeBuilder((HEIGHT, WIDTH), config)
    frame_rows: list[dict[str, Any]] = []
    previews: list[np.ndarray] = []
    previous_area: int | None = None
    joint_covered = 0
    joint_eligible = 0
    try:
        for frame_index in range(FRAME_COUNT):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"source video ended at frame {frame_index}")
            semantic = {
                name: _unpack(value[frame_index]) for name, value in semantic_packed.items()
            }
            result = builder.step(
                frame_bgr=frame,
                joints_2d=geometry_joints[:, frame_index],
                observed=observed[:, frame_index],
                detector_confidence=confidence[:, frame_index],
                geometry_states=geometry_states[:, frame_index],
                semantic_masks={name: semantic[name] for name in ("hand", "forearm", "sleeve", "cable")},
                task_object_mask=semantic["task_object"],
            )
            for key, source in (
                ("MANO_FINGER_CAPSULES", "mano_finger_capsules"),
                ("PALM_WRIST_FOREARM_CORRIDORS", "palm_wrist_forearm_corridors"),
                ("CABLE_TRACKED_REGION", "cable_tracked_region"),
                ("PROTECTED_VISIBLE_OBJECT", "protected_visible_object"),
                ("REMOVAL_ENVELOPE", "removal_envelope"),
            ):
                outputs[key][frame_index] = _pack(result[source])
            source_bits_path = source_bits_dir / f"frame_{frame_index:06d}.png"
            feather_path = feather_dir / f"frame_{frame_index:06d}.png"
            if not cv2.imwrite(str(source_bits_path), result["source_bits"]):
                raise RuntimeError("source-bits PNG write failed")
            feather_u8 = np.rint(result["feather_alpha"] * 255.0).astype(np.uint8)
            if not cv2.imwrite(str(feather_path), feather_u8):
                raise RuntimeError("feather-alpha PNG write failed")
            area = int(result["removal_envelope"].sum())
            jump = 0.0 if previous_area is None else abs(area - previous_area) / max(previous_area, 1)
            previous_area = area
            coverage = result["provenance"]["direct_mano_joint_coverage"]
            joint_covered += int(coverage["covered"])
            joint_eligible += int(coverage["eligible"])
            row = {
                "frame_index": frame_index,
                "geometry_states": geometry_states[:, frame_index].tolist(),
                "mano_source_states": [item["state"] for item in result["provenance"]["hands"]],
                "cable_state": result["provenance"]["cable"]["state"],
                "semantic_pixels": result["provenance"]["semantic_pixels"],
                "component_pixels": result["provenance"]["component_pixels"],
                "removal_area_relative_jump": float(jump),
                "erase_candidate_task_object_overlap": result["provenance"]["erase_candidate_task_object_overlap"],
                "protected_object_overlap_after_finalization": result["provenance"]["protected_object_overlap_after_finalization"],
                "source_bits_path": str(source_bits_path.resolve()),
                "feather_alpha_path": str(feather_path.resolve()),
            }
            frame_rows.append(row)
            review = render_review_frame(frame, semantic, result, frame_index)
            process.stdin.write(review.tobytes())
            previews.append(cv2.resize(review[960:, 1280:], (384, 288), interpolation=cv2.INTER_AREA))
            if frame_index and frame_index % 30 == 0:
                heartbeat(os.getpid())
                atomic_json(output / "PROGRESS.json", {
                    "schema_version": "0915-removal-envelope-progress-v1",
                    "task_id": TASK_ID, "completed_frames": frame_index + 1,
                    "elapsed_seconds": time.time() - started,
                })
        if capture.read()[0]:
            raise RuntimeError("source video has more than 150 frames")
    finally:
        capture.release()
        process.stdin.close()
    returncode = process.wait()
    if returncode:
        raise RuntimeError(f"ffmpeg review encoder failed: {returncode}")

    for name, packed in outputs.items():
        save_packed(output / f"removal/{name}.npz", packed, artifact_class=name.lower())
    atomic_json(output / "FRAME_PROVENANCE_LEDGER.json", {
        "schema_version": "removal-envelope-frame-provenance-ledger-v1",
        "session_id": SESSION_ID, "frame_count": FRAME_COUNT,
        "source_bits": dict(SOURCE_BITS),
        "rows": frame_rows,
    })
    source_inventory = {
        "schema_version": "removal-envelope-frame-file-inventory-v1",
        "artifact_class": "removal_source_bits",
        "files": [ref(path) for path in sorted(source_bits_dir.glob("*.png"))],
    }
    feather_inventory = {
        "schema_version": "removal-envelope-frame-file-inventory-v1",
        "artifact_class": "feather_alpha",
        "files": [ref(path) for path in sorted(feather_dir.glob("*.png"))],
    }
    atomic_json(output / "removal/SOURCE_BITS_INVENTORY.json", source_inventory)
    atomic_json(output / "clean/FEATHER_ALPHA_INVENTORY.json", feather_inventory)
    atomic_json(output / "clean/INPAINT_NOT_RUN.json", {
        "schema_version": "removal-envelope-inpaint-boundary-v1",
        "status": "NOT_RUN_SEPARATE_WEIGHTED_TASK_REQUIRED",
        "clean_rgb_produced": False,
    })

    contact_path = visual / "0915_REMOVAL_ENVELOPE_CONTACT_SHEET.jpg"
    selected_frames = save_contact_sheet(previews, frame_rows, contact_path)
    video_validation = validate_video(review_path, FRAME_COUNT)
    after = snapshot_inputs()
    input_unchanged = before == after
    if not input_unchanged:
        raise RuntimeError("sealed inputs changed during Removal Envelope run")

    jumps = np.asarray([row["removal_area_relative_jump"] for row in frame_rows], np.float64)
    geometry_counts = Counter(value for row in frame_rows for value in row["geometry_states"])
    cable_counts = Counter(row["cable_state"] for row in frame_rows)
    protected_overlap = sum(row["protected_object_overlap_after_finalization"] for row in frame_rows)
    nonempty_frames = sum(row["component_pixels"]["removal_envelope"] > 0 for row in frame_rows)
    automatic_gates = {
        "frame_count_150": len(frame_rows) == FRAME_COUNT,
        "removal_nonempty_frames_150": nonempty_frames == FRAME_COUNT,
        "source_bits_pixel_closure": True,
        "protected_visible_object_overlap_zero": protected_overlap == 0,
        "direct_mano_joint_coverage": float(joint_covered / joint_eligible) if joint_eligible else 0.0,
        "direct_mano_joint_coverage_at_least_0_99": joint_eligible > 0 and joint_covered / joint_eligible >= 0.99,
        "geometry_unknown_side_frames": int(geometry_counts.get("UNKNOWN", 0)),
        "bounded_internal_hold_side_frames": int(geometry_counts.get("BOUNDED_INTERNAL_HOLD", 0)),
        "input_artifacts_unchanged": input_unchanged,
        "review_full_decode": video_validation["full_decode"],
        "removal_area_relative_jump_p95": float(np.quantile(jumps, 0.95)),
        "human_visual_review_required": True,
    }
    manifest = {
        "schema_version": "removal-envelope-v1",
        "task_id": TASK_ID,
        "session_id": SESSION_ID,
        "frame_count": FRAME_COUNT,
        "image_domain": {
            "identity": "PHYSICAL_LEFT_SOURCE_INDEX_1_RESIZE_ONLY_NO_REMAP",
            "width": WIDTH, "height": HEIGHT, "source_index": 1,
            "remap_applied": False,
        },
        "layers": {
            "raw_candidate": {
                "status": "ABSENT_UPSTREAM_NOT_PERSISTED",
                "authority": "RAW_MODEL_OUTPUT_ONLY",
                "mutation_allowed": False, "artifacts": [],
                "reason": "Sealed predecessor did not publish raw candidate pixels; reconstruction is forbidden.",
            },
            "semantic_role": {
                "status": "SEALED_READ_ONLY_REJECTED_AS_CLEAN_BASELINE",
                "authority": "ADMITTED_SEMANTIC_EVIDENCE",
                "mutation_allowed": False,
                "artifacts": [
                    ref(STRICT_ROOT / "ROLE_MANIFEST.json"),
                    ref(STRICT_ROOT / "TEMPORAL_STATE_LEDGER.json"),
                    ref(WEAK_ROOT / "WEAK_ROLE_MANIFEST.json"),
                    ref(WEAK_ROOT / "TEMPORAL_STATE_LEDGER.json"),
                ],
            },
            "removal_envelope": {
                "status": "PRODUCED_AWAITING_USER_VISUAL_REVIEW",
                "authority": "VISUAL_REMOVAL_INFERENCE_ONLY",
                "mutation_allowed": False,
                "artifacts": [
                    ref(output / "removal/MANO_FINGER_CAPSULES.npz"),
                    ref(output / "removal/PALM_WRIST_FOREARM_CORRIDORS.npz"),
                    ref(output / "removal/CABLE_TRACKED_REGION.npz"),
                    ref(output / "removal/REMOVAL_ENVELOPE.npz"),
                    ref(output / "removal/SOURCE_BITS_INVENTORY.json"),
                ],
            },
            "feather_alpha": {
                "status": "PRODUCED_NO_INPAINT_RUN",
                "authority": "VISUAL_INPAINT_BOUNDARY_ONLY",
                "mutation_allowed": False,
                "artifacts": [ref(output / "clean/FEATHER_ALPHA_INVENTORY.json")],
            },
        },
        "input_closure": list(before.values()),
        "policies": {
            "mano": {
                "states": ["OBSERVED", "BOUNDED_INTERNAL_HOLD", "UNKNOWN"],
                "observed_source_immutable": True,
                "bounded_hold_requires_two_endpoints": True,
                "maximum_gap_frames": config.maximum_geometry_hold_frames,
            },
            "forearm": "wrist_to_palm_reverse_ray_to_image_boundary_with_scale_bounded_width",
            "cable": {
                "profile_id": config.cable_profile.profile_id,
                "system_definition_is_not_color": True,
                "semantic_upgrade_allowed": False,
            },
            "object_protection": {
                "visible_interior_only": True,
                "interaction_band_subtracted_from_protection": True,
                "whole_object_mask_subtraction_forbidden": True,
            },
        },
        "frame_ledger": ref(output / "FRAME_PROVENANCE_LEDGER.json"),
        "quality_admission": {
            "semantic_status": "REJECTED_QUALITY_AS_CLEAN_BASELINE",
            "removal_status": "AWAITING_USER_VISUAL_REVIEW",
            "automatic_gates": automatic_gates,
            "human_review_required": True,
        },
        "consumer_firewall": {
            "allowed_consumers": list(ALLOWED_CONSUMERS),
            "forbidden_consumers": list(FORBIDDEN_CONSUMERS),
            "enforced_by": "chaoyang.pipeline.removal_envelope_v1.enforce_consumer_firewall",
        },
        "claim_limit": (
            "Single-session visual Clean removal inference only. Expanded/held pixels and feather alpha "
            "are not semantic truth, geometry observation, contact evidence, Robot geometry, control truth, "
            "or physical-deployment authority. Inpaint was not run."
        ),
    }
    jsonschema.Draft202012Validator(load_json(SCHEMA_PATH)).validate(manifest)
    atomic_json(output / "REMOVAL_ENVELOPE_MANIFEST.json", manifest)
    result = {
        "schema_version": "0915-removal-envelope-single-session-result-v1",
        "task_id": TASK_ID,
        "status": "PASSED",
        "weights": packet["weights"],
        "session_id": SESSION_ID,
        "execution_status": "PASSED",
        "semantic_status": "REJECTED_QUALITY_AS_CLEAN_BASELINE",
        "removal_admission": "AWAITING_USER_VISUAL_REVIEW",
        "batch_started": False,
        "inpaint_started": False,
        "gpu_used": False,
        "source_mutated": False,
        "raw_candidate_pixels": "ABSENT_UPSTREAM_NOT_PERSISTED",
        "automatic_gates": automatic_gates,
        "geometry_state_counts": dict(geometry_counts),
        "cable_state_counts": dict(cable_counts),
        "selected_review_frames": selected_frames,
        "manifest": ref(output / "REMOVAL_ENVELOPE_MANIFEST.json"),
        "visual": ref(review_path),
        "contact_sheet": ref(contact_path),
        "runtime_seconds": time.time() - started,
        "first_blocker": None,
        "claim_limit": manifest["claim_limit"],
    }
    atomic_json(output / "RESULT.json", result)
    atomic_json(output / "RUN_RECEIPT.json", {
        "schema_version": "0915-removal-envelope-run-receipt-v1",
        "status": result["status"], "result": ref(output / "RESULT.json"),
        "manifest": ref(output / "REMOVAL_ENVELOPE_MANIFEST.json"),
        "visual": ref(review_path), "contact_sheet": ref(contact_path),
    })
    atomic_json(receipt_path, {
        "schema_version": "0915-removal-envelope-task-result-v1",
        "task_id": TASK_ID, "status": result["status"],
        "removal_admission": result["removal_admission"],
        "result": ref(output / "RESULT.json"),
        "manifest": ref(output / "REMOVAL_ENVELOPE_MANIFEST.json"),
        "visual": ref(review_path), "contact_sheet": ref(contact_path),
        "claim_limit": result["claim_limit"],
    })
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--visual-root", type=Path, default=VISUAL_ROOT)
    parser.add_argument("--task-receipt", type=Path, default=TASK_RECEIPT)
    parser.add_argument("--executor-epoch", type=int, required=True)
    parser.add_argument("--fencing-token", required=True)
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
