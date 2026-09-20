#!/usr/bin/env python3
"""Normalize the pinned B1+B1R closure into D1_CLEAN_PREP CPU inputs.

This is a local, CPU-only postprocessor.  It never connects to the remote
host and never runs an inpainting model.  Its snapshot input must already be
a read-only copy of the exact upstream artifacts pinned below.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any
import uuid

import cv2
import numpy as np
from PIL import Image


COMPOSITE_SCHEMA = "0915-robot-recovery-v21-d1-clean-prep-b1-b1r-composite-closure-v1"
HANDOFF_SCHEMA = "0915-robot-recovery-v21-b1-d1-clean-prep-handoff-v1"
INPUT_SCHEMA = "0915-robot-recovery-v21-d1-clean-prep-input-v1"
RAW_SCHEMA = "0915-robot-recovery-v21-d1-lossless-raw-frame-manifest-v1"
HAND_SCHEMA = "0915-robot-recovery-v21-b1-side-hand-mask-manifest-v1"
OBJECT_SCHEMA = "0915-robot-recovery-v21-b1-task-object-protection-manifest-v1"
ADMITTED_SIDE = "ADMITTED_B1_PER_SIDE_HAND_MASK"
IMAGE_DOMAIN = "PHYSICAL_LEFT_SOURCEINDEX1_CROP_RESIZE_ONLY"
HEIGHT = 960
WIDTH = 1280
PROJECT_ROOT = Path(__file__).resolve().parents[3]

POKER = "play_cards_0915_044"
CHIPS = "get_potato_chips_0915_097"

# relpath -> (bytes, sha256, immutable remote origin)
PINS: dict[str, tuple[int, str, str]] = {
    "packages/B1R/RESULT.json": (
        1689, "7be40d384d7826dfcb0e57c9b4561f5f71c8e947902f52a6a42bd8c77c999d9a",
        "packages/B1R/RESULT.json",
    ),
    "packages/B1R/BATCH_RESULT.json": (
        3362, "c964b93fff8e434faa12290481369a00ed5d4a5a9b119610ffc311ce284943d2",
        "packages/B1R/BATCH_RESULT.json",
    ),
    "packages/B1R/B1R_HAND_ADMISSION.json": (
        2716, "8dcd30418a777c7b0e120ed09ada61c3cf2f5b742138e346eaf7ae80d260a821",
        "packages/B1R/B1R_HAND_ADMISSION.json",
    ),
    "packages/B1/RESULT.json": (
        1609, "62e08adceff6d21cecf1dea2a171b2e8f381250f06e75fb21fdd3d509481410d",
        "packages/B1/RESULT.json",
    ),
    "packages/B1/BATCH_RESULT.json": (
        10663, "07780f172a0139651aba382dae5363482f4473ee3e17b984a5af98276f38768b",
        "packages/B1/BATCH_RESULT.json",
    ),
    "packages/B1/hand/sessions/potato_chips/get_potato_chips_0915_097/RESULT.json": (
        5562, "94e5f1e4d0ffb095642099c3fde1f6300b6eaa81ff84ae92fa95323d21367d01",
        "packages/B1/hand/sessions/potato_chips/get_potato_chips_0915_097/RESULT.json",
    ),
    "packages/B1/hand/sessions/potato_chips/get_potato_chips_0915_097/ROLE_MANIFEST.json": (
        5009, "12141059165a944c70eae0b76b7a965d44409a38d1f371cbe5f46493413703cc",
        "packages/B1/hand/sessions/potato_chips/get_potato_chips_0915_097/ROLE_MANIFEST.json",
    ),
    "packages/B1/hand/sessions/potato_chips/get_potato_chips_0915_097/semantic/right_hand.npz": (
        554647, "e193feb3d02eba7d0192dbe8c11eb3e91ed9320a03824a53ac5f9d9f8bc2f333",
        "packages/B1/hand/sessions/potato_chips/get_potato_chips_0915_097/semantic/right_hand.npz",
    ),
    "packages/B1/hand/sessions/potato_chips/get_potato_chips_0915_097/temporal/right_hand_STATE_LEDGER.json": (
        100194, "80eb45a39c33d9d537caddc0de1bc633e953a845a1ec97386e30506431ba6f12",
        "packages/B1/hand/sessions/potato_chips/get_potato_chips_0915_097/temporal/right_hand_STATE_LEDGER.json",
    ),
    "packages/B1/object_admission/playing_cards/play_cards_0915_044/B1_OBJECT_ADMISSION.json": (
        642, "38db542417975d868aecab3dc45e3c62782cbe2d1d5e9acd19b4663f66cd026e",
        "packages/B1/object_admission/playing_cards/play_cards_0915_044/B1_OBJECT_ADMISSION.json",
    ),
    "packages/B1/object/sessions/playing_cards/play_cards_0915_044/RESULT.json": (
        6926, "e2ca3f1f8cb3b3daa6e8f161564e4ea30013bdc30fd00c1b1a0ad04775d3e5c1",
        "packages/B1/object/sessions/playing_cards/play_cards_0915_044/RESULT.json",
    ),
    "packages/B1/object/sessions/playing_cards/play_cards_0915_044/OBJECT_INSTANCE_MANIFEST.json": (
        10441, "c67adf1e45f05cf8af7a30f8bd8d4e79bbe9e73d0db95664b412293494239f22",
        "packages/B1/object/sessions/playing_cards/play_cards_0915_044/OBJECT_INSTANCE_MANIFEST.json",
    ),
    "packages/B1/object/sessions/playing_cards/play_cards_0915_044/semantic/playing_card_00.npz": (
        57313, "6a059ea92bb684606fcadc4638afa197727154eb968863ebfed1461dcab86cf0",
        "packages/B1/object/sessions/playing_cards/play_cards_0915_044/semantic/playing_card_00.npz",
    ),
    "packages/B1/object/sessions/playing_cards/play_cards_0915_044/semantic/playing_card_01.npz": (
        59400, "67bbcd82e559cc05b43974563fb1e3da9e5006848739623632fc4d1568865a4c",
        "packages/B1/object/sessions/playing_cards/play_cards_0915_044/semantic/playing_card_01.npz",
    ),
    "packages/B1/object/sessions/playing_cards/play_cards_0915_044/semantic/playing_card_02.npz": (
        48526, "17575bc5b2176a695c61b68afc168c3726bd8755c28182e4989d1b88d481b450",
        "packages/B1/object/sessions/playing_cards/play_cards_0915_044/semantic/playing_card_02.npz",
    ),
    "packages/B1/object/sessions/playing_cards/play_cards_0915_044/temporal/playing_card_00_STATE.json": (
        62536, "9e6f4215d8313ef4bd1fe6b868ef388a92b5dbeda376de57aeaa11eeeb62f374",
        "packages/B1/object/sessions/playing_cards/play_cards_0915_044/temporal/playing_card_00_STATE.json",
    ),
    "packages/B1/object/sessions/playing_cards/play_cards_0915_044/temporal/playing_card_01_STATE.json": (
        61542, "41ceb0d0d13177d3e82e544803e49dac5fd46cb69f6c28a74565afe9207bb41e",
        "packages/B1/object/sessions/playing_cards/play_cards_0915_044/temporal/playing_card_01_STATE.json",
    ),
    "packages/B1/object/sessions/playing_cards/play_cards_0915_044/temporal/playing_card_02_STATE.json": (
        61604, "20c6b5bb91ea8c59151521f64b357c1807d1e034726bc0320df8070f4f966137",
        "packages/B1/object/sessions/playing_cards/play_cards_0915_044/temporal/playing_card_02_STATE.json",
    ),
    "packages/B1/object_admission/potato_chips/get_potato_chips_0915_097/B1_OBJECT_ADMISSION.json": (
        638, "fbf0ffe72a6a469f1f4333f631615423fe8a5963d9d173b7a1fc2a0e5cb4a1de",
        "packages/B1/object_admission/potato_chips/get_potato_chips_0915_097/B1_OBJECT_ADMISSION.json",
    ),
    "packages/B1/object/sessions/potato_chips/get_potato_chips_0915_097/RESULT.json": (
        1666, "25ad3d06196f917b17220e05cb3330b556c32ae573ee2fafb578010c45dc677f",
        "packages/B1/object/sessions/potato_chips/get_potato_chips_0915_097/RESULT.json",
    ),
    "packages/B1/object/sessions/potato_chips/get_potato_chips_0915_097/OBJECT_INSTANCE_MANIFEST.json": (
        2913, "0fa8679b7a08b4534d4ed84497e23028a93aa2fe2ba4af34253252c839d5c72b",
        "packages/B1/object/sessions/potato_chips/get_potato_chips_0915_097/OBJECT_INSTANCE_MANIFEST.json",
    ),
    "packages/A1/prepared/play_cards_0915_044/PHYSICAL_LEFT_SOURCEINDEX1_RESIZE_ONLY.mp4": (
        3297594, "85aac7f6e6a65ecaed18ad0b9d1fa8fa99437e87bdca544afc4ba1c0325e51fa",
        "packages/A1/prepared/play_cards_0915_044/PHYSICAL_LEFT_SOURCEINDEX1_RESIZE_ONLY.mp4",
    ),
    "packages/A1/prepared/get_potato_chips_0915_097/PHYSICAL_LEFT_SOURCEINDEX1_RESIZE_ONLY.mp4": (
        7011557, "a72dba65c7c6ff66711d4357e2336d2247e99fffae9caa981a1deab02fd520e0",
        "packages/A1/prepared/get_potato_chips_0915_097/PHYSICAL_LEFT_SOURCEINDEX1_RESIZE_ONLY.mp4",
    ),
}

REMOTE_PREFIX = (
    str(PROJECT_ROOT / "_run/current/0915_robot_quality_recovery_15h_v2/attempts/attempt_0001")
    + "/"
)


class ClosureError(RuntimeError):
    pass


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ref(path: Path, remote_path: str | None = None) -> dict[str, Any]:
    path = path.resolve(strict=True)
    value: dict[str, Any] = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if remote_path is not None:
        value["remote_path"] = remote_path
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ClosureError(f"JSON object required: {path}")
    return value


def pinned(snapshot: Path, relative: str) -> Path:
    if relative not in PINS:
        raise ClosureError(f"unregistered upstream path: {relative}")
    expected_bytes, expected_sha, remote_relative = PINS[relative]
    path = (snapshot / relative).resolve(strict=True)
    if path.is_symlink() or not path.is_file():
        raise ClosureError(f"regular non-symlink upstream file required: {path}")
    if path.stat().st_size != expected_bytes or sha256_file(path) != expected_sha:
        raise ClosureError(f"upstream pin drift: {relative}")
    return path


def upstream_ref(snapshot: Path, relative: str) -> dict[str, Any]:
    path = pinned(snapshot, relative)
    return ref(path, REMOTE_PREFIX + PINS[relative][2])


def save_rgb(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(value, mode="RGB").save(path, compress_level=6)


def save_mask(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(value.astype(np.uint8) * np.uint8(255), mode="L").save(path, compress_level=9)


def decode_video(video: Path, frame_count: int, output: Path) -> list[dict[str, Any]]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ClosureError(f"cannot decode video: {video}")
    rows: list[dict[str, Any]] = []
    try:
        for frame_id in range(frame_count):
            ok, bgr = capture.read()
            if not ok or bgr is None:
                raise ClosureError(f"short decode at frame {frame_id}: {video}")
            if bgr.shape != (HEIGHT, WIDTH, 3) or bgr.dtype != np.uint8:
                raise ClosureError(f"decoded image-domain drift at frame {frame_id}: {bgr.shape}/{bgr.dtype}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            path = output / f"{frame_id:06d}.png"
            save_rgb(path, rgb)
            rows.append({"frame_id": frame_id, "rgb": ref(path)})
        ok, _ = capture.read()
        if ok:
            raise ClosureError(f"decode has more than {frame_count} frames: {video}")
    finally:
        capture.release()
    return rows


def load_packed_masks(path: Path, frame_count: int) -> tuple[np.ndarray, str]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"packed", "frame_count", "height", "width", "bitorder"}
        if set(archive.files) != required:
            raise ClosureError(f"packed mask keys drift: {path}")
        count = int(archive["frame_count"])
        height = int(archive["height"])
        width = int(archive["width"])
        bitorder = str(archive["bitorder"])
        packed = archive["packed"].copy()
    if (count, height, width) != (frame_count, HEIGHT, WIDTH):
        raise ClosureError(f"packed mask domain drift: {path}")
    if bitorder not in {"big", "little"} or packed.shape != (frame_count, (HEIGHT * WIDTH + 7) // 8):
        raise ClosureError(f"packed mask encoding drift: {path}")
    return packed, bitorder


def unpack_row(packed: np.ndarray, bitorder: str, frame_id: int) -> np.ndarray:
    return np.unpackbits(packed[frame_id], bitorder=bitorder)[: HEIGHT * WIDTH].reshape(HEIGHT, WIDTH).astype(bool)


def ledger_frames(path: Path, frame_count: int) -> list[dict[str, Any]]:
    frames = load_json(path).get("frames")
    if not isinstance(frames, list) or len(frames) != frame_count:
        raise ClosureError(f"state-ledger denominator drift: {path}")
    for frame_id, row in enumerate(frames):
        if not isinstance(row, dict) or row.get("frame_id") != frame_id:
            raise ClosureError(f"state-ledger order drift at {frame_id}: {path}")
    return frames


def make_raw_manifest(
    snapshot: Path, staged: Path, session_id: str, task: str, frame_count: int, video_relative: str,
) -> Path:
    video = pinned(snapshot, video_relative)
    frames = decode_video(video, frame_count, staged / "frames" / session_id)
    path = staged / "manifests" / session_id / "LOSSLESS_RAW_FRAME_MANIFEST.json"
    write_json(path, {
        "schema_version": RAW_SCHEMA,
        "created_at": now(),
        "producer": "D1_CLEAN_PREP_LOCAL_CPU_NORMALIZER",
        "session_id": session_id,
        "task": task,
        "frame_count": frame_count,
        "image_domain": IMAGE_DOMAIN,
        "image_size": [WIDTH, HEIGHT],
        "lossless_frames": True,
        "decode_backend": "OpenCV VideoCapture CPU; BGR decoded then converted to RGB; PNG lossless",
        "upstream_video": upstream_ref(snapshot, video_relative),
        "frames": frames,
    })
    return path


def make_chips_hand_manifest(snapshot: Path, staged: Path) -> Path:
    archive_relative = "packages/B1/hand/sessions/potato_chips/get_potato_chips_0915_097/semantic/right_hand.npz"
    ledger_relative = "packages/B1/hand/sessions/potato_chips/get_potato_chips_0915_097/temporal/right_hand_STATE_LEDGER.json"
    packed, bitorder = load_packed_masks(pinned(snapshot, archive_relative), 394)
    states = ledger_frames(pinned(snapshot, ledger_relative), 394)
    rows = []
    admitted = 0
    unknown = []
    for frame_id, state in enumerate(states):
        mask = unpack_row(packed, bitorder, frame_id)
        observed = state.get("semantic_admitted") is True
        if observed:
            admitted += 1
            status = "ADMITTED_DIRECT_OBSERVED_PROXY"
            if not np.any(mask):
                raise ClosureError(f"admitted Chips right Hand is empty at frame {frame_id}")
        else:
            unknown.append(frame_id)
            status = "UNKNOWN_NOT_ABSENT_TOO_MANY_COMPONENTS" if frame_id == 58 else "UNKNOWN_NOT_ABSENT"
            if np.any(mask):
                raise ClosureError(f"unadmitted Chips right Hand is nonempty at frame {frame_id}")
        path = staged / "masks" / CHIPS / "hand" / "right" / f"{frame_id:06d}.png"
        save_mask(path, mask)
        rows.append({"frame_id": frame_id, "observed": observed, "status": status, "mask": ref(path)})
    if admitted != 393 or unknown != [58]:
        raise ClosureError(f"Chips right Hand admission drift: admitted={admitted}, unknown={unknown}")
    path = staged / "manifests" / CHIPS / "RIGHT_HAND_MASK_MANIFEST.json"
    write_json(path, {
        "schema_version": HAND_SCHEMA,
        "created_at": now(),
        "producer_package": "B1",
        "normalizer_package": "D1_CLEAN_PREP",
        "session_id": CHIPS,
        "side": "right",
        "consumer_status": ADMITTED_SIDE,
        "whole_session_hand_terminal_required": False,
        "frame_count": 394,
        "semantic_admitted_frames": 393,
        "unknown_frames": [58],
        "unknown_semantics": "UNKNOWN_NOT_ABSENT",
        "authority": "DEVELOPMENT_PROXY_NO_MANUAL_GROUND_TRUTH",
        "upstream": {
            "b1_hand_result": upstream_ref(snapshot, "packages/B1/hand/sessions/potato_chips/get_potato_chips_0915_097/RESULT.json"),
            "b1_role_manifest": upstream_ref(snapshot, "packages/B1/hand/sessions/potato_chips/get_potato_chips_0915_097/ROLE_MANIFEST.json"),
            "semantic_archive": upstream_ref(snapshot, archive_relative),
            "state_ledger": upstream_ref(snapshot, ledger_relative),
        },
        "frames": rows,
    })
    return path


def make_poker_object_manifest(snapshot: Path, staged: Path) -> Path:
    instances = ("playing_card_00", "playing_card_01", "playing_card_02")
    packed_by_id: dict[str, tuple[np.ndarray, str]] = {}
    states_by_id: dict[str, list[dict[str, Any]]] = {}
    upstream_instances: dict[str, Any] = {}
    for instance_id in instances:
        archive_relative = f"packages/B1/object/sessions/playing_cards/play_cards_0915_044/semantic/{instance_id}.npz"
        ledger_relative = f"packages/B1/object/sessions/playing_cards/play_cards_0915_044/temporal/{instance_id}_STATE.json"
        packed_by_id[instance_id] = load_packed_masks(pinned(snapshot, archive_relative), 166)
        states_by_id[instance_id] = ledger_frames(pinned(snapshot, ledger_relative), 166)
        upstream_instances[instance_id] = {
            "semantic_archive": upstream_ref(snapshot, archive_relative),
            "state_ledger": upstream_ref(snapshot, ledger_relative),
        }
    rows = []
    nonzero_counts = {instance_id: 0 for instance_id in instances}
    for frame_id in range(166):
        occupied = np.zeros((HEIGHT, WIDTH), dtype=bool)
        items = []
        for instance_id in instances:
            packed, bitorder = packed_by_id[instance_id]
            mask = unpack_row(packed, bitorder, frame_id)
            semantic_admitted = states_by_id[instance_id][frame_id].get("semantic_admitted") is True
            if semantic_admitted != bool(np.any(mask)):
                raise ClosureError(f"Poker {instance_id} state/mask drift at frame {frame_id}")
            if np.any(occupied & mask):
                raise ClosureError(f"Poker candidate protection masks overlap at frame {frame_id}")
            occupied |= mask
            if semantic_admitted:
                nonzero_counts[instance_id] += 1
                status = "UNKNOWN_PHYSICAL_CARD_OR_FACE_IDENTITY_VISIBLE_PROTECTION_ONLY"
            else:
                status = "UNKNOWN_NOT_ABSENT"
            path = staged / "masks" / POKER / "object" / instance_id / f"{frame_id:06d}.png"
            save_mask(path, mask)
            items.append({
                "instance_id": instance_id,
                "observed": semantic_admitted,
                "status": status,
                "mask": ref(path),
            })
        rows.append({"frame_id": frame_id, "instances": items})
    if nonzero_counts != {"playing_card_00": 130, "playing_card_01": 151, "playing_card_02": 80}:
        raise ClosureError(f"Poker visible protection denominator drift: {nonzero_counts}")
    path = staged / "manifests" / POKER / "TASK_OBJECT_PROTECTION_MANIFEST.json"
    write_json(path, {
        "schema_version": OBJECT_SCHEMA,
        "created_at": now(),
        "producer_package": "B1",
        "normalizer_package": "D1_CLEAN_PREP",
        "session_id": POKER,
        "task": "playing_cards",
        "frame_count": 166,
        "depends_on_hand_terminal": False,
        "instance_ids": list(instances),
        "identity_authority": "UNKNOWN_PHYSICAL_CARD_OR_FACE_IDENTITY",
        "protection_authority": "CONSERVATIVE_VISIBLE_RAW_VETO_ONLY",
        "visible_surface_only": True,
        "hidden_shape_inferred": False,
        "hidden_appearance_authorized": False,
        "atlas_authorized": False,
        "union_mask_created": False,
        "upstream": {
            "b1_object_result": upstream_ref(snapshot, "packages/B1/object/sessions/playing_cards/play_cards_0915_044/RESULT.json"),
            "b1_object_instance_manifest": upstream_ref(snapshot, "packages/B1/object/sessions/playing_cards/play_cards_0915_044/OBJECT_INSTANCE_MANIFEST.json"),
            "b1_object_admission": upstream_ref(snapshot, "packages/B1/object_admission/playing_cards/play_cards_0915_044/B1_OBJECT_ADMISSION.json"),
            "instances": upstream_instances,
        },
        "frames": rows,
    })
    return path


def make_chips_object_manifest(snapshot: Path, staged: Path) -> Path:
    instances = ("potato_chip_00", "potato_chip_01", "potato_chip_02")
    rows = []
    empty = np.zeros((HEIGHT, WIDTH), dtype=bool)
    for frame_id in range(394):
        items = []
        for instance_id in instances:
            path = staged / "masks" / CHIPS / "object" / instance_id / f"{frame_id:06d}.png"
            save_mask(path, empty)
            items.append({
                "instance_id": instance_id,
                "observed": False,
                "status": "UNKNOWN_NO_SEPARATE_VISIBLE_INSTANCE_NOT_ABSENT",
                "mask": ref(path),
            })
        rows.append({"frame_id": frame_id, "instances": items})
    path = staged / "manifests" / CHIPS / "TASK_OBJECT_PROTECTION_MANIFEST.json"
    write_json(path, {
        "schema_version": OBJECT_SCHEMA,
        "created_at": now(),
        "producer_package": "B1",
        "normalizer_package": "D1_CLEAN_PREP",
        "session_id": CHIPS,
        "task": "potato_chips",
        "frame_count": 394,
        "depends_on_hand_terminal": False,
        "instance_ids": list(instances),
        "identity_authority": "THREE_SEPARATE_PHYSICAL_INSTANCE_SLOTS",
        "instance_semantics": "ALL_THREE_UNKNOWN_NOT_ABSENT_EMPTY_MASKS",
        "visible_surface_only": True,
        "hidden_shape_inferred": False,
        "hidden_appearance_authorized": False,
        "union_mask_created": False,
        "upstream": {
            "b1_object_result": upstream_ref(snapshot, "packages/B1/object/sessions/potato_chips/get_potato_chips_0915_097/RESULT.json"),
            "b1_object_instance_manifest": upstream_ref(snapshot, "packages/B1/object/sessions/potato_chips/get_potato_chips_0915_097/OBJECT_INSTANCE_MANIFEST.json"),
            "b1_object_admission": upstream_ref(snapshot, "packages/B1/object_admission/potato_chips/get_potato_chips_0915_097/B1_OBJECT_ADMISSION.json"),
        },
        "frames": rows,
    })
    return path


def validate_upstream_semantics(snapshot: Path) -> None:
    for relative in PINS:
        pinned(snapshot, relative)
    b1r_result = load_json(pinned(snapshot, "packages/B1R/RESULT.json"))
    b1r_batch = load_json(pinned(snapshot, "packages/B1R/BATCH_RESULT.json"))
    b1r_admission = load_json(pinned(snapshot, "packages/B1R/B1R_HAND_ADMISSION.json"))
    b1_result = load_json(pinned(snapshot, "packages/B1/RESULT.json"))
    if b1r_result.get("status") != "COMPLETED_WITH_QUALITY_REJECTION":
        raise ClosureError("B1R aggregate terminal drift")
    if b1r_batch.get("capability") != "HAND_ONLY" or b1r_batch.get("session_id") != POKER:
        raise ClosureError("B1R capability scope drift")
    sides = b1r_admission.get("sides")
    expected = [
        ("left_hand", "REJECTED_DIRECT_OBSERVED_ADMISSION", False),
        ("right_hand", "REJECTED_TRACKER_DIRECTION_INCOMPLETE", False),
    ]
    actual = [(row.get("role"), row.get("b1r_status"), row.get("consumer_allowed")) for row in sides or []]
    if actual != expected:
        raise ClosureError(f"B1R Poker Hand admission drift: {actual}")
    if b1_result.get("status") != "FAILED_RUNTIME_FINAL":
        raise ClosureError("original B1 immutable terminal drift")
    chips_hand = load_json(pinned(snapshot, "packages/B1/hand/sessions/potato_chips/get_potato_chips_0915_097/RESULT.json"))
    side_results = chips_hand.get("side_results")
    expected_chips = [
        ("left_hand", "BLOCKED_NO_DIRECT_HAWOR_ANCHOR", False),
        ("right_hand", "PASS_HAND_DIRECT_OBSERVED_PROXY", True),
    ]
    actual_chips = [(row.get("role"), row.get("status"), row.get("consumer_allowed")) for row in side_results or []]
    if actual_chips != expected_chips:
        raise ClosureError(f"original B1 Chips Hand admission drift: {actual_chips}")
    poker_admission = load_json(pinned(snapshot, "packages/B1/object_admission/playing_cards/play_cards_0915_044/B1_OBJECT_ADMISSION.json"))
    expected_poker_status = {f"playing_card_{index:02d}": "UNKNOWN_PHYSICAL_CARD_OR_FACE_IDENTITY" for index in range(3)}
    if poker_admission.get("consumer_allowed_instances") != 0 or poker_admission.get("instance_status") != expected_poker_status:
        raise ClosureError("Poker object identity admission drift")
    chips_admission = load_json(pinned(snapshot, "packages/B1/object_admission/potato_chips/get_potato_chips_0915_097/B1_OBJECT_ADMISSION.json"))
    expected_chips_status = {f"potato_chip_{index:02d}": "UNKNOWN_NO_SEPARATE_VISIBLE_INSTANCE" for index in range(3)}
    if chips_admission.get("consumer_allowed_instances") != 0 or chips_admission.get("instance_status") != expected_chips_status:
        raise ClosureError("Chips object slot admission drift")
    if poker_admission.get("union_mask_created") is not False or chips_admission.get("union_mask_created") is not False:
        raise ClosureError("upstream object admission created a forbidden union")


def materialize(snapshot_root: Path, output_root: Path) -> dict[str, Any]:
    snapshot = snapshot_root.resolve(strict=True)
    output = output_root.resolve()
    if output.exists() or output.is_symlink():
        raise ClosureError(f"fresh output root required: {output}")
    validate_upstream_semantics(snapshot)
    output.parent.mkdir(parents=True, exist_ok=True)
    # The output root is fresh and is removed on any exception.  Writing there
    # directly keeps every emitted absolute ref stable; an atomic directory
    # rename would otherwise invalidate refs created below the staging name.
    staged = output
    staged.mkdir()
    try:
        print("normalizing Raw frames (CPU)", flush=True)
        poker_raw = make_raw_manifest(
            snapshot, staged, POKER, "playing_cards", 166,
            "packages/A1/prepared/play_cards_0915_044/PHYSICAL_LEFT_SOURCEINDEX1_RESIZE_ONLY.mp4",
        )
        chips_raw = make_raw_manifest(
            snapshot, staged, CHIPS, "potato_chips", 394,
            "packages/A1/prepared/get_potato_chips_0915_097/PHYSICAL_LEFT_SOURCEINDEX1_RESIZE_ONLY.mp4",
        )
        print("normalizing admitted/UNKNOWN Hand and Object lanes (CPU)", flush=True)
        chips_hand = make_chips_hand_manifest(snapshot, staged)
        poker_object = make_poker_object_manifest(snapshot, staged)
        chips_object = make_chips_object_manifest(snapshot, staged)

        composite_path = staged / "B1_B1R_D1_CLEAN_PREP_COMPOSITE_CLOSURE.json"
        composite = {
            "schema_version": COMPOSITE_SCHEMA,
            "created_at": now(),
            "status": "COMPOSITE_CLOSED_WITH_QUALITY_REJECTION",
            "terminal": True,
            "fixed_sessions": [POKER, CHIPS],
            "aggregate_gate_relaxed": False,
            "capability_gate_relaxed": False,
            "composition_rule": "B1R replaces only the failed Poker044 Hand capability; all other original B1 subordinate artifacts remain immutable and independently pinned.",
            "b1r_hand_capability_terminal": {
                "scope": {"session_id": POKER, "capability": "HAND_ONLY"},
                "result": upstream_ref(snapshot, "packages/B1R/RESULT.json"),
                "batch_result": upstream_ref(snapshot, "packages/B1R/BATCH_RESULT.json"),
                "hand_admission": upstream_ref(snapshot, "packages/B1R/B1R_HAND_ADMISSION.json"),
                "status": "COMPLETED_WITH_QUALITY_REJECTION",
            },
            "original_b1": {
                "result": upstream_ref(snapshot, "packages/B1/RESULT.json"),
                "batch_result": upstream_ref(snapshot, "packages/B1/BATCH_RESULT.json"),
                "status": "FAILED_RUNTIME_FINAL",
                "original_b1_result_mutated": False,
                "used_as_aggregate_terminal": False,
                "subordinate_artifacts_consumed_only_if_explicitly_consumer_allowed_or_as_conservative_visible_raw_protection": True,
            },
            "pinned_upstream_artifacts": {
                relative: upstream_ref(snapshot, relative) for relative in sorted(PINS)
            },
            "sessions": [
                {
                    "session_id": POKER,
                    "hand": {
                        "left": {"status": "UNKNOWN_B1R_REJECTED_DIRECT_OBSERVED_ADMISSION", "mask_manifest": None},
                        "right": {"status": "UNKNOWN_B1R_REJECTED_TRACKER_DIRECTION_INCOMPLETE", "mask_manifest": None},
                    },
                    "task_object": {
                        "physical_identity": "UNKNOWN_PHYSICAL_CARD_OR_FACE_IDENTITY",
                        "hidden_appearance_authorized": False,
                        "manifest": ref(poker_object),
                    },
                },
                {
                    "session_id": CHIPS,
                    "hand": {
                        "left": {"status": "UNKNOWN_BLOCKED_NO_DIRECT_ANCHOR", "mask_manifest": None},
                        "right": {"status": ADMITTED_SIDE, "mask_manifest": ref(chips_hand)},
                    },
                    "task_object": {
                        "three_slots": ["potato_chip_00", "potato_chip_01", "potato_chip_02"],
                        "slot_status": "UNKNOWN_NO_SEPARATE_VISIBLE_INSTANCE_NOT_ABSENT",
                        "union_mask_created": False,
                        "manifest": ref(chips_object),
                    },
                },
            ],
            "claim_limit": "Composite normalization only. It does not mutate B1/B1R, admit Poker Hand, establish Poker physical identity, union Chips objects, or grant Clean/model/training authority.",
        }
        write_json(composite_path, composite)

        handoff_path = staged / "B1_D1_CLEAN_PREP_HANDOFF.json"
        handoff = {
            "schema_version": HANDOFF_SCHEMA,
            "created_at": now(),
            "terminal": True,
            "status": "COMPLETED_WITH_QUALITY_REJECTION",
            "authority_mode": "B1R_POKER_HAND_CAPABILITY_PLUS_PINNED_ORIGINAL_B1_SUBORDINATE_ARTIFACTS",
            "aggregate_gate_relaxed": False,
            "capability_gate_relaxed": False,
            "terminal_receipt": upstream_ref(snapshot, "packages/B1R/RESULT.json"),
            "composite_closure": ref(composite_path),
            "sessions": [
                {
                    "session_id": POKER,
                    "task": "playing_cards",
                    "frame_count": 166,
                    "hand_sides": {
                        "left": {"status": "UNKNOWN_B1R_REJECTED_DIRECT_OBSERVED_ADMISSION", "mask_manifest": None},
                        "right": {"status": "UNKNOWN_B1R_REJECTED_TRACKER_DIRECTION_INCOMPLETE", "mask_manifest": None},
                    },
                    "task_object": {"depends_on_hand_terminal": False, "manifest": ref(poker_object)},
                },
                {
                    "session_id": CHIPS,
                    "task": "potato_chips",
                    "frame_count": 394,
                    "hand_sides": {
                        "left": {"status": "UNKNOWN_BLOCKED_NO_DIRECT_ANCHOR", "mask_manifest": None},
                        "right": {"status": ADMITTED_SIDE, "mask_manifest": ref(chips_hand)},
                    },
                    "task_object": {"depends_on_hand_terminal": False, "manifest": ref(chips_object)},
                },
            ],
        }
        write_json(handoff_path, handoff)

        input_path = staged / "D1_CLEAN_PREP_INPUT_MANIFEST.json"
        write_json(input_path, {
            "schema_version": INPUT_SCHEMA,
            "created_at": now(),
            "execution_mode": "CPU_PREPARE_ONLY_NO_MODEL_NO_GPU",
            "fixed_sessions": [POKER, CHIPS],
            "b1_handoff": ref(handoff_path),
            "sessions": [
                {"session_id": POKER, "task": "playing_cards", "frame_count": 166, "raw_frame_manifest": ref(poker_raw)},
                {"session_id": CHIPS, "task": "potato_chips", "frame_count": 394, "raw_frame_manifest": ref(chips_raw)},
            ],
        })

        receipt_path = staged / "NORMALIZATION_RECEIPT.json"
        write_json(receipt_path, {
            "schema_version": "0915-robot-recovery-v21-d1-clean-prep-normalization-receipt-v1",
            "created_at": now(),
            "status": "COMPLETED_CPU_NORMALIZED_INPUT_CLOSURE",
            "gpu_used": False,
            "model_execution_performed": False,
            "remote_write_performed": False,
            "upstream_pin_count": len(PINS),
            "original_b1_result_mutated": False,
            "poker_hand": "BOTH_SIDES_UNKNOWN_NO_MASK_CONSUMED",
            "chips_hand": "LEFT_UNKNOWN_RIGHT_393_OF_394_PROXY_FRAME_58_UNKNOWN",
            "poker_object": "THREE_VISIBLE_PROTECTION_CANDIDATES_PHYSICAL_IDENTITY_UNKNOWN",
            "chips_object": "THREE_SEPARATE_UNKNOWN_NOT_ABSENT_EMPTY_SLOTS_NO_UNION",
            "artifacts": {
                "composite_closure": ref(composite_path),
                "handoff": ref(handoff_path),
                "input_manifest": ref(input_path),
                "poker_raw_manifest": ref(poker_raw),
                "chips_raw_manifest": ref(chips_raw),
                "chips_right_hand_manifest": ref(chips_hand),
                "poker_object_manifest": ref(poker_object),
                "chips_object_manifest": ref(chips_object),
            },
            "claim_limit": "CPU normalization and SHA closure only; no inpainting or Clean terminal.",
        })
        return load_json(output / "NORMALIZATION_RECEIPT.json")
    except Exception:
        shutil.rmtree(staged, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    try:
        receipt = materialize(args.snapshot_root, args.output_root)
    except (ClosureError, OSError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({
            "status": "BLOCKED_FAIL_CLOSED",
            "error": f"{type(error).__name__}: {error}",
            "gpu_used": False,
            "model_execution_performed": False,
        }, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
