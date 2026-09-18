#!/usr/bin/env python3
"""Evaluate Removal Envelope V2 on the fixed real 0915 150-frame canary."""

from __future__ import annotations

import argparse
from dataclasses import fields, replace
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
import numpy as np

from chaoyang.governance.campaign_0915_task_specs_v1 import build_packet
from chaoyang.pipeline.removal_envelope_v2 import (
    SOURCE_BITS,
    CableInstanceProfileV2,
    RemovalEnvelopeV2Builder,
    RemovalEnvelopeV2Config,
)


ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "0915_removal_envelope_v2_real_canary_v1"
PHASE = "0915_REMOVAL_ENVELOPE_V2_REAL_CANARY_V1"
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
CONFIG = ROOT / "configs/systems/clean/removal_envelope_0915_play_cards_001_v2.json"
CORE = ROOT / "src/chaoyang/pipeline/removal_envelope_v2.py"

STRICT_MASKS = {
    "hand": ("left_hand_00", "right_hand_00"),
    "forearm": ("left_forearm_00", "right_forearm_00"),
    "equipment": (
        "left_finger_sleeve_cluster_00", "right_finger_sleeve_cluster_00",
        "left_cable_00", "right_cable_00",
    ),
    "task_object": ("playing_card_00", "playing_card_01", "playing_card_02"),
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
FIXED_REVIEW_FRAMES = (0, 30, 60, 81, 94, 120, 149)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _ref(path: Path) -> dict[str, Any]:
    candidate = path.resolve(strict=True)
    return {
        "path": str(candidate), "bytes": candidate.stat().st_size,
        "sha256": _sha256(candidate),
    }


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True,
                  allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()


def _validate_route(output: Path) -> tuple[dict[str, Any], Path]:
    packet_path = ROOT / f"tasks/current/{TASK_ID}/TASK_PACKET.json"
    packet = _load_json(packet_path)
    if packet != build_packet(TASK_ID):
        raise RuntimeError("current task packet differs from frozen specification")
    if packet.get("weights") != "ABSENT":
        raise RuntimeError("Removal V2 real canary must have weights ABSENT")
    expected = (ROOT / str(packet["write_set"][0])).resolve()
    if output.resolve() != expected:
        raise RuntimeError(f"output root must equal packet writer root: {expected}")
    state = _load_json(ROOT / "docs/governance/LONG_HORIZON_TASK_STATE.json")
    if state.get("next_task", {}).get("task_id") != TASK_ID:
        raise RuntimeError("Removal V2 task is not current next_task")
    index = _load_json(ROOT / "tasks/current/INDEX.json")
    route = next((row for row in index.get("task_packets", [])
                  if row.get("task_id") == TASK_ID), None)
    if route is None or route.get("execution_allowed") is not True:
        raise RuntimeError("task index does not authorize Removal V2")
    if route.get("packet_sha256") != _sha256(packet_path):
        raise RuntimeError("task packet SHA mismatch")
    return packet, (ROOT / str(packet["write_set"][1])).resolve()


def _heartbeat(pid: int) -> None:
    completed = subprocess.run(
        [
            sys.executable, "-m", "chaoyang.governance.heartbeat_task",
            "--task-id", TASK_ID, "--pid", str(pid), "--status", "RUNNING",
            "--phase", PHASE,
        ],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            "governance heartbeat failed: "
            + (completed.stderr or completed.stdout)[-4000:]
        )


def _load_config() -> RemovalEnvelopeV2Config:
    raw = _load_json(CONFIG)
    if raw.get("schema_version") != "removal-envelope-config-v2":
        raise RuntimeError("Removal V2 config schema drift")
    cable_raw = dict(raw["cable_profile"])
    cable_fields = {item.name for item in fields(CableInstanceProfileV2)}
    cable = CableInstanceProfileV2(**{
        key: tuple(value) if key in {"hsv_lower", "hsv_upper"} else value
        for key, value in cable_raw.items() if key in cable_fields
    })
    config_fields = {item.name for item in fields(RemovalEnvelopeV2Config)}
    values = {
        key: value for key, value in raw.items()
        if key in config_fields and key != "cable_profile"
    }
    return RemovalEnvelopeV2Config(cable_profile=cable, **values)


def _load_packed(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        packed = np.asarray(archive["packed"], np.uint8)
        metadata = (
            int(archive["frame_count"]), int(archive["height"]),
            int(archive["width"]), str(archive["bitorder"]),
        )
    expected = (FRAME_COUNT, HEIGHT, WIDTH, BITORDER)
    if metadata != expected:
        raise RuntimeError(f"packed mask domain drift {metadata} != {expected}: {path}")
    if packed.shape != (FRAME_COUNT, (HEIGHT * WIDTH + 7) // 8):
        raise RuntimeError(f"packed mask shape drift: {path}")
    return packed


def unpack_packed_frame(packed: np.ndarray, frame_index: int) -> np.ndarray:
    """Unpack exactly one frame so the runner never materialises all bool masks."""

    row = np.asarray(packed[frame_index], np.uint8)
    return np.unpackbits(
        row, bitorder=BITORDER, count=HEIGHT * WIDTH,
    ).reshape(HEIGHT, WIDTH).astype(bool)


def _pack(mask: np.ndarray) -> np.ndarray:
    return np.packbits(np.asarray(mask, bool).reshape(-1), bitorder=BITORDER)


def _union_archives(paths: Iterable[Path]) -> np.ndarray:
    merged = np.zeros((FRAME_COUNT, (HEIGHT * WIDTH + 7) // 8), np.uint8)
    for path in paths:
        merged |= _load_packed(path)
    return merged


def _strict(name: str) -> Path:
    return STRICT_ROOT / "masks" / f"{name}.npz"


def _weak(name: str) -> Path:
    return WEAK_ROOT / "masks" / f"{name}.npz"


def _input_paths() -> list[Path]:
    paths = [
        VIDEO, HAWOR, CONFIG, CORE, Path(__file__).resolve(),
        STRICT_ROOT / "ROLE_MANIFEST.json",
        STRICT_ROOT / "TEMPORAL_STATE_LEDGER.json",
        WEAK_ROOT / "WEAK_ROLE_MANIFEST.json",
        WEAK_ROOT / "TEMPORAL_STATE_LEDGER.json",
    ]
    for names in STRICT_MASKS.values():
        paths.extend(_strict(name) for name in names)
    for names in WEAK_MASKS.values():
        paths.extend(_weak(name) for name in names)
    return paths


def _input_stats(paths: Iterable[Path]) -> dict[str, tuple[int, int]]:
    return {
        str(path.resolve(strict=True)): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in paths
    }


def _semantic_at(packed: dict[str, np.ndarray], frame: int) -> dict[str, np.ndarray]:
    return {
        name: unpack_packed_frame(value, frame)
        for name, value in packed.items()
    }


def _build_stable_background(
    semantic: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, Any]]:
    capture = cv2.VideoCapture(str(VIDEO))
    if not capture.isOpened():
        raise RuntimeError("cannot open resize-only video for stable-background pass")
    minimum: np.ndarray | None = None
    maximum: np.ndarray | None = None
    evidence_ever = np.zeros((HEIGHT, WIDTH), bool)
    count = 0
    try:
        for frame_index in range(FRAME_COUNT):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"stable-background decode ended at {frame_index}")
            minimum = frame.copy() if minimum is None else np.minimum(minimum, frame)
            maximum = frame.copy() if maximum is None else np.maximum(maximum, frame)
            masks = _semantic_at(semantic, frame_index)
            for mask in masks.values():
                evidence_ever |= mask
            count += 1
        if capture.read()[0]:
            raise RuntimeError("resize-only video has more than 150 frames")
    finally:
        capture.release()
    assert minimum is not None and maximum is not None
    temporal_range = maximum.astype(np.int16) - minimum.astype(np.int16)
    stable_photometric = np.all(temporal_range <= 8, axis=2)
    exclusion = cv2.dilate(
        evidence_ever.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
    ).astype(bool)
    stable = stable_photometric & ~exclusion
    return stable, {
        "decoded_frames": count,
        "maximum_channel_range": 8,
        "evidence_exclusion_radius_px": 6,
        "stable_pixels": int(stable.sum()),
        "stable_fraction": float(stable.mean()),
        "authority": "QA_HOOK_NOT_BACKGROUND_GROUND_TRUTH",
    }


def _read_frame(capture: cv2.VideoCapture, index: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    if not ok or frame.shape != (HEIGHT, WIDTH, 3):
        raise RuntimeError(f"cannot decode frame {index} in resize-only domain")
    return frame


def _reverse_cable_support(
    config: RemovalEnvelopeV2Config,
    semantic: dict[str, np.ndarray],
    joints: np.ndarray,
    observed: np.ndarray,
) -> np.ndarray:
    profile = replace(config.cable_profile, require_reverse_support_after_seed=False)
    reverse_builder = RemovalEnvelopeV2Builder(
        (HEIGHT, WIDTH), replace(config, cable_profile=profile),
    )
    result = np.zeros((FRAME_COUNT, (HEIGHT * WIDTH + 7) // 8), np.uint8)
    capture = cv2.VideoCapture(str(VIDEO))
    if not capture.isOpened():
        raise RuntimeError("cannot open resize-only video for reverse cable pass")
    try:
        for frame_index in reversed(range(FRAME_COUNT)):
            frame = _read_frame(capture, frame_index)
            masks = _semantic_at(semantic, frame_index)
            value = reverse_builder.step(
                frame_bgr=frame,
                semantic_masks={
                    "hand": masks["strict_hand"],
                    "forearm": masks["strict_forearm"],
                    "equipment": masks["strict_equipment"],
                },
                foreground_proposals=masks["weak_forearm"],
                sleeve_candidate_mask=masks["weak_sleeve"],
                cable_candidate_mask=masks["weak_cable"],
                reverse_cable_support_mask=None,
                task_object_mask=masks["task_object"],
                joints_2d=joints[:, frame_index],
                observed=observed[:, frame_index],
                stable_background_mask=None,
            )
            result[frame_index] = _pack(value["cable_instance"])
    finally:
        capture.release()
    return result


def _overlay(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> None:
    if not mask.any():
        return
    color_array = np.asarray(color, np.float32)
    image[mask] = np.clip(
        image[mask].astype(np.float32) * (1.0 - alpha) + color_array * alpha,
        0, 255,
    ).astype(np.uint8)


def _title(image: np.ndarray, lines: list[str]) -> None:
    cv2.rectangle(image, (0, 0), (image.shape[1] - 1, 32 * len(lines) + 8),
                  (15, 15, 15), -1)
    for index, line in enumerate(lines):
        cv2.putText(image, line, (12, 27 + 30 * index),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 245), 1,
                    cv2.LINE_AA)


def _panel(frame: np.ndarray, masks: dict[str, np.ndarray], value: dict[str, Any], index: int) -> np.ndarray:
    semantic = frame.copy()
    _overlay(semantic, masks["strict_hand"], (255, 180, 40), 0.42)
    _overlay(semantic, value["selected_forearm"], (0, 190, 255), 0.42)
    _overlay(semantic, masks["strict_equipment"], (210, 70, 230), 0.42)
    contours, _ = cv2.findContours(masks["task_object"].astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(semantic, contours, -1, (40, 230, 60), 2)
    _title(semantic, [f"frame {index:03d} | sealed SAM base", "hand / selected forearm / equipment / objects"])

    repairs = frame.copy()
    _overlay(repairs, value["mano_gap_repair"], (255, 100, 30), 0.70)
    _overlay(repairs, value["attached_sleeve_repair"], (200, 60, 240), 0.65)
    _overlay(repairs, value["cable_instance"], (0, 230, 255), 0.75)
    _overlay(repairs, value["protected_object_core"], (40, 220, 50), 0.35)
    _title(repairs, ["validated local repairs", "blue=MANO gap | magenta=sleeve | yellow=cable | green=protected"])

    comparison = frame.copy()
    _overlay(comparison, value["semantic_base"], (255, 185, 45), 0.30)
    _overlay(comparison, value["removal_envelope"], (20, 30, 235), 0.45)
    _title(comparison, ["semantic baseline vs V2 removal", "orange=base | red=final (repairs are local additions)"])

    source = np.zeros_like(frame)
    source[value["semantic_base"]] = (80, 170, 230)
    source[value["mano_gap_repair"]] = (255, 100, 30)
    source[value["attached_sleeve_repair"]] = (200, 60, 240)
    source[value["cable_instance"]] = (0, 230, 255)
    metrics = value["quality"]["metrics"]
    _title(source, [
        f"QA: {value['quality']['status']}",
        f"inflate={metrics['area_inflation'] if metrics['area_inflation'] is not None else -1:.3f} "
        f"repair={metrics['repair_contribution_ratio'] if metrics['repair_contribution_ratio'] is not None else -1:.3f} "
        f"spill={metrics['background_spill_ratio'] if metrics['background_spill_ratio'] is not None else -1:.3f}",
    ])
    tiles = [cv2.resize(item, (640, 480), interpolation=cv2.INTER_AREA)
             for item in (semantic, repairs, comparison, source)]
    return np.vstack((np.hstack(tiles[:2]), np.hstack(tiles[2:])))


def _open_writer(path: Path, fps: float) -> subprocess.Popen[bytes]:
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "1280x960",
        "-r", f"{fps:.6f}", "-i", "-", "-an", "-c:v", "libx264",
        "-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p", str(path),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def _validate_video(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    count = 0
    shape: list[int] | None = None
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        count += 1
        shape = [int(frame.shape[1]), int(frame.shape[0])]
    capture.release()
    if count != FRAME_COUNT:
        raise RuntimeError(f"review video decoded {count}, expected {FRAME_COUNT}")
    return {"full_decode": True, "frame_count": count, "resolution": shape}


def aggregate_real_canary_quality(
    rows: list[dict[str, Any]], *, stable_background_fraction: float,
) -> dict[str, Any]:
    """Aggregate the real-video comparison without converting unknown to absence."""

    if len(rows) != FRAME_COUNT:
        raise ValueError("real canary aggregate requires exactly 150 frames")

    def values(name: str) -> np.ndarray:
        return np.asarray([
            row[name] for row in rows if row.get(name) is not None
        ], np.float64)

    area = values("area_inflation")
    spill = values("background_spill_ratio")
    repair = values("repair_contribution_ratio")
    derivative = values("temporal_area_derivative")
    baseline_derivative = values("semantic_temporal_area_derivative")
    object_damage = sum(int(row["protected_object_core_damage_pixels"]) for row in rows)
    repair_majority_frames = sum(
        row.get("repair_contribution_ratio") is not None
        and row["repair_contribution_ratio"] > 0.50
        for row in rows
    )
    p95 = lambda value: float(np.quantile(value, 0.95)) if len(value) else None
    metrics = {
        "stable_background_fraction": stable_background_fraction,
        "area_inflation_p95": p95(area),
        "background_spill_ratio_p95": p95(spill),
        "repair_contribution_ratio_p95": p95(repair),
        "temporal_area_derivative_p95": p95(derivative),
        "semantic_temporal_area_derivative_p95": p95(baseline_derivative),
        "protected_object_core_damage_pixels": object_damage,
        "repair_majority_frames": int(repair_majority_frames),
    }
    gates = {
        "full_150_frames": True,
        "stable_background_hook_nonempty": stable_background_fraction >= 0.01,
        "area_inflation_p95_at_most_1_5": metrics["area_inflation_p95"] is not None
        and metrics["area_inflation_p95"] <= 1.50,
        "background_spill_p95_at_most_0_03": metrics["background_spill_ratio_p95"]
        is not None and metrics["background_spill_ratio_p95"] <= 0.03,
        "repair_p95_at_most_0_25": metrics["repair_contribution_ratio_p95"]
        is not None and metrics["repair_contribution_ratio_p95"] <= 0.25,
        "repair_never_majority": repair_majority_frames == 0,
        "protected_object_core_damage_zero": object_damage == 0,
        "temporal_p95_at_most_0_35": metrics["temporal_area_derivative_p95"]
        is not None and metrics["temporal_area_derivative_p95"] <= 0.35,
        "temporal_flicker_no_worse_than_semantic_plus_0_05": (
            metrics["temporal_area_derivative_p95"] is not None
            and metrics["semantic_temporal_area_derivative_p95"] is not None
            and metrics["temporal_area_derivative_p95"]
            <= metrics["semantic_temporal_area_derivative_p95"] + 0.05
        ),
    }
    return {
        "status": "PASS" if all(gates.values()) else "REJECTED_QUALITY",
        "metrics": metrics,
        "gates": gates,
    }


def _save_layers(path: Path, layers: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}.npz")
    np.savez_compressed(
        temporary,
        **{name: np.asarray(value, np.uint8) for name, value in layers.items()},
        frame_count=np.asarray(FRAME_COUNT, np.int32),
        height=np.asarray(HEIGHT, np.int32), width=np.asarray(WIDTH, np.int32),
        bitorder=np.asarray(BITORDER),
        authority=np.asarray("VISUAL_CLEAN_ONLY_NOT_GEOMETRY_EVIDENCE"),
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"fresh output root required: {output}")
    packet, visual = _validate_route(output)
    if visual.exists() or visual.is_symlink():
        raise RuntimeError(f"fresh visual root required: {visual}")
    output.mkdir(parents=True)
    visual.mkdir(parents=True)
    started = time.time()
    input_paths = _input_paths()
    before_stats = _input_stats(input_paths)

    pid = os.getpid()
    start_ticks = int(Path(f"/proc/{pid}/stat").read_text().split()[21])
    epoch = time.time_ns()
    signature = {
        "schema_version": "0915-removal-envelope-v2-real-run-signature-v1",
        "task_id": TASK_ID, "session_id": SESSION_ID,
        "executor_epoch": epoch, "weights": "ABSENT", "gpu_used": False,
        "inputs": [_ref(path) for path in input_paths],
        "task_packet": _ref(ROOT / f"tasks/current/{TASK_ID}/TASK_PACKET.json"),
        "design": "SAM_BASE_PLUS_VALIDATED_LOCAL_REPAIRS",
        "image_domain": "PHYSICAL_LEFT_SOURCE_INDEX_1_RESIZE_ONLY_NO_REMAP",
    }
    signature["run_signature_sha256"] = _canonical_sha(signature)
    _atomic_json(output / "RUN_SIGNATURE.json", signature)
    _atomic_json(output / "WRITER_CLAIM.json", {
        "schema_version": "0915-removal-envelope-v2-real-writer-claim-v1",
        "task_id": TASK_ID, "attempt_id": output.name, "pid": pid,
        "proc_start_ticks": start_ticks, "executor_epoch": epoch,
        "run_signature_sha256": signature["run_signature_sha256"],
        "fencing_token_sha256": hashlib.sha256(uuid.uuid4().hex.encode()).hexdigest(),
        "unique_write_root": str(output), "status": "CLAIMED",
    })
    _heartbeat(pid)

    semantic_packed = {
        "strict_hand": _union_archives(_strict(name) for name in STRICT_MASKS["hand"]),
        "strict_forearm": _union_archives(_strict(name) for name in STRICT_MASKS["forearm"]),
        "strict_equipment": _union_archives(_strict(name) for name in STRICT_MASKS["equipment"]),
        "weak_forearm": _union_archives(_weak(name) for name in WEAK_MASKS["forearm"]),
        "weak_sleeve": _union_archives(_weak(name) for name in WEAK_MASKS["sleeve"]),
        "weak_cable": _union_archives(_weak(name) for name in WEAK_MASKS["cable"]),
        "task_object": (
            _union_archives(_strict(name) for name in STRICT_MASKS["task_object"])
            | _union_archives(_weak(name) for name in WEAK_MASKS["task_object"])
        ),
    }
    with np.load(HAWOR, allow_pickle=False) as archive:
        joints = np.asarray(archive["joints_2d"], np.float64)
        observed = np.asarray(archive["observed"], bool)
        provenance = np.asarray(archive["provenance"]).astype(str)
        side_names = np.asarray(archive["anatomical_side_names"]).astype(str).tolist()
    if joints.shape != (2, FRAME_COUNT, 21, 2) or observed.shape != (2, FRAME_COUNT):
        raise RuntimeError("HaWoR MANO21 frame axes drift")
    if side_names != ["left", "right"]:
        raise RuntimeError("HaWoR anatomical side order drift")
    direct_observed = observed & (provenance == "BOUNDED_PARAMETER_FIT")
    config = _load_config()
    stable_background, stable_record = _build_stable_background(semantic_packed)
    stable_packed = _pack(stable_background)
    np.savez_compressed(
        output / "STABLE_BACKGROUND.npz", packed=stable_packed,
        height=np.asarray(HEIGHT, np.int32), width=np.asarray(WIDTH, np.int32),
        bitorder=np.asarray(BITORDER),
        authority=np.asarray("QA_HOOK_NOT_BACKGROUND_GROUND_TRUTH"),
    )
    _heartbeat(pid)
    reverse_support = _reverse_cable_support(
        config, semantic_packed, joints, direct_observed,
    )
    _heartbeat(pid)

    packed_width = (HEIGHT * WIDTH + 7) // 8
    layer_names = (
        "semantic_base", "selected_forearm", "mano_gap_repair",
        "attached_sleeve_repair", "cable_instance", "protected_object_core",
        "removal_envelope", "reverse_cable_support",
    )
    layers = {
        name: np.zeros((FRAME_COUNT, packed_width), np.uint8)
        for name in layer_names
    }
    layers["reverse_cable_support"][:] = reverse_support
    rows: list[dict[str, Any]] = []
    previews: dict[int, np.ndarray] = {}
    previous_semantic_pixels: int | None = None
    review_path = visual / "0915_REMOVAL_ENVELOPE_V2_REAL_REVIEW.mp4"
    capture = cv2.VideoCapture(str(VIDEO))
    if not capture.isOpened():
        raise RuntimeError("cannot open resize-only input video")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    writer = _open_writer(review_path, fps)
    assert writer.stdin is not None
    builder = RemovalEnvelopeV2Builder((HEIGHT, WIDTH), config)
    try:
        for frame_index in range(FRAME_COUNT):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"forward decode ended at frame {frame_index}")
            masks = _semantic_at(semantic_packed, frame_index)
            reverse = unpack_packed_frame(reverse_support, frame_index)
            value = builder.step(
                frame_bgr=frame,
                semantic_masks={
                    "hand": masks["strict_hand"],
                    "forearm": masks["strict_forearm"],
                    "equipment": masks["strict_equipment"],
                },
                foreground_proposals=masks["weak_forearm"],
                sleeve_candidate_mask=masks["weak_sleeve"],
                cable_candidate_mask=masks["weak_cable"],
                reverse_cable_support_mask=reverse,
                task_object_mask=masks["task_object"],
                joints_2d=joints[:, frame_index],
                observed=direct_observed[:, frame_index],
                stable_background_mask=stable_background,
            )
            for name in layer_names[:-1]:
                layers[name][frame_index] = _pack(value[name])
            metrics = value["quality"]["metrics"]
            semantic_pixels = int(value["semantic_base"].sum())
            semantic_derivative = (
                None if previous_semantic_pixels is None
                else abs(semantic_pixels - previous_semantic_pixels)
                / max(1, previous_semantic_pixels)
            )
            previous_semantic_pixels = semantic_pixels
            rows.append({
                "frame_index": frame_index,
                **metrics,
                "semantic_temporal_area_derivative": semantic_derivative,
                "quality_status": value["quality"]["status"],
                "quality_gates": value["quality"]["gates"],
                "forearm_state": value["provenance"]["forearm"]["state"],
                "sleeve_state": value["provenance"]["sleeve"]["state"],
                "cable_state": value["provenance"]["cable"]["state"],
                "source_pixels": value["provenance"]["source_pixels"],
                "exclusive_repair_contribution_pixels": value["provenance"]
                ["exclusive_repair_contribution_pixels"],
            })
            panel = _panel(frame, masks, value, frame_index)
            writer.stdin.write(panel.tobytes())
            if frame_index in FIXED_REVIEW_FRAMES:
                previews[frame_index] = cv2.resize(
                    panel, (640, 480), interpolation=cv2.INTER_AREA,
                )
            if frame_index and frame_index % 25 == 0:
                _heartbeat(pid)
        if capture.read()[0]:
            raise RuntimeError("resize-only video has more than 150 frames")
    finally:
        capture.release()
        writer.stdin.close()
    if writer.wait() != 0:
        raise RuntimeError("ffmpeg review writer failed")

    _save_layers(output / "V2_MASK_LAYERS.npz", layers)
    _atomic_json(output / "FRAME_QUALITY_LEDGER.json", {
        "schema_version": "0915-removal-envelope-v2-frame-quality-ledger-v1",
        "task_id": TASK_ID, "session_id": SESSION_ID,
        "frame_count": FRAME_COUNT, "source_bits": dict(SOURCE_BITS),
        "rows": rows,
    })
    quality = aggregate_real_canary_quality(
        rows, stable_background_fraction=stable_record["stable_fraction"],
    )
    review_decode = _validate_video(review_path)
    sheet_path = visual / "0915_REMOVAL_ENVELOPE_V2_REAL_CONTACT_SHEET.jpg"
    sheet = np.vstack([previews[index] for index in FIXED_REVIEW_FRAMES])
    if not cv2.imwrite(str(sheet_path), sheet):
        raise RuntimeError("cannot write V2 contact sheet")
    after_stats = _input_stats(input_paths)
    if before_stats != after_stats:
        raise RuntimeError("sealed Removal V2 input changed during execution")
    summary = {
        "schema_version": "0915-removal-envelope-v2-real-summary-v1",
        "task_id": TASK_ID, "session_id": SESSION_ID,
        "design": "SAM_BASE_PLUS_VALIDATED_LOCAL_REPAIRS",
        "image_domain": {
            "identity": "PHYSICAL_LEFT_SOURCE_INDEX_1_RESIZE_ONLY_NO_REMAP",
            "width": WIDTH, "height": HEIGHT, "source_index": 1,
            "remap_applied": False,
        },
        "frame_count": FRAME_COUNT,
        "stable_background": stable_record,
        "reverse_cable_pass": {
            "performed": True,
            "nonempty_frames": int(sum(
                unpack_packed_frame(reverse_support, index).any()
                for index in range(FRAME_COUNT)
            )),
            "maximum_instances_per_frame": config.cable_profile.maximum_instances,
        },
        "comparison": quality,
        "review_decode": review_decode,
        "inpaint_run": False,
        "sam_rerun": False,
        "gpu_used": False,
        "consumer_firewall": {
            "allowed": ["CLEAN_MASK_QA", "VISUAL_INPAINT"],
            "forbidden": [
                "DEPTH", "OBJECT6D", "CONTACT", "ROBOT_GEOMETRY",
                "CONTROL_GROUND_TRUTH",
            ],
        },
        "human_visual_review_required": True,
    }
    _atomic_json(output / "REMOVAL_ENVELOPE_V2_REAL_SUMMARY.json", summary)
    terminal = "PASSED" if quality["status"] == "PASS" else "REJECTED_QUALITY"
    admission = (
        "AWAITING_USER_VISUAL_REVIEW" if terminal == "PASSED"
        else "REJECTED_QUALITY"
    )
    result = {
        "schema_version": "0915-removal-envelope-v2-real-result-v1",
        "task_id": TASK_ID, "session_id": SESSION_ID, "status": terminal,
        "weights": "ABSENT", "gpu_used": False, "source_mutated": False,
        "sam_rerun": False, "inpaint_started": False, "batch_started": False,
        "removal_admission": admission,
        "human_visual_review_required": terminal == "PASSED",
        "first_blocker": None if terminal == "PASSED" else "V2_REAL_CANARY_QUALITY_GATES_FAILED",
        "summary": _ref(output / "REMOVAL_ENVELOPE_V2_REAL_SUMMARY.json"),
        "frame_ledger": _ref(output / "FRAME_QUALITY_LEDGER.json"),
        "mask_layers": _ref(output / "V2_MASK_LAYERS.npz"),
        "review_video": _ref(review_path), "contact_sheet": _ref(sheet_path),
        "runtime_seconds": time.time() - started,
        "claim_limit": packet["claim_limit"],
    }
    _atomic_json(output / "RESULT.json", result)
    _atomic_json(output / "RUN_RECEIPT.json", {
        "schema_version": "0915-removal-envelope-v2-real-run-receipt-v1",
        "task_id": TASK_ID, "status": terminal,
        "run_signature": _ref(output / "RUN_SIGNATURE.json"),
        "writer_claim": _ref(output / "WRITER_CLAIM.json"),
        "result": _ref(output / "RESULT.json"),
        "review_full_decode": True, "source_mutated": False, "gpu_used": False,
    })
    (visual / "README_ZH.md").write_text(
        "# 0915 Removal Envelope V2 真实视频 Canary\n\n"
        "本目录比较封存 SAM semantic base 与 V2 的局部 repair。V2 没有重跑 "
        "SAM，没有 inpaint，也不允许进入 Depth、Object6D、Contact 或 Robot 几何。\n\n"
        f"- 自动终态：`{terminal}`\n"
        f"- Removal admission：`{admission}`\n"
        f"- 150 帧完整解码：是\n"
        f"- repair-majority frames：`{quality['metrics']['repair_majority_frames']}`\n"
        f"- protected object core damage：`{quality['metrics']['protected_object_core_damage_pixels']}` px\n"
        "- 即使自动门通过，也必须完成人工全片复核后才能成为 Clean authority。\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

