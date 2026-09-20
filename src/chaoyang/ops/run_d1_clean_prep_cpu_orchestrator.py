#!/usr/bin/env python3
"""Fail-closed CPU preparation for the V2.1 D1_CLEAN_PREP successor.

This runner is deliberately not an inpainter.  It consumes only SHA-bound,
per-side Hand masks admitted by the terminal B1 handoff, validates the Task
Object protection lane independently of Hand terminal state, and materializes
three different domains:

  M_remove: admitted human-hand semantics after the Task Object veto;
  M_write:  frozen pixels an authorized fresh fill may replace;
  M_flow:   larger flow-completion context around M_write.

No fill is executed here.  M_write is the frozen allowed write domain, not a
claim that pixels have already been materialized; every M_write pixel remains
UNKNOWN and the emitted candidate is decoded-pixel byte-identical to Raw
everywhere.  A future model job may consume the registration bundle, but must
publish a fresh candidate and pass the same M_write-domain audit before it can
become a Clean result.
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
from typing import Any, Mapping
import uuid

import cv2
import numpy as np
from PIL import Image


SCHEMA = "0915-robot-recovery-v21-d1-clean-prep-cpu-v1"
INPUT_SCHEMA = "0915-robot-recovery-v21-d1-clean-prep-input-v1"
B1_HANDOFF_SCHEMA = "0915-robot-recovery-v21-b1-d1-clean-prep-handoff-v1"
RAW_SCHEMA = "0915-robot-recovery-v21-d1-lossless-raw-frame-manifest-v1"
HAND_SCHEMA = "0915-robot-recovery-v21-b1-side-hand-mask-manifest-v1"
OBJECT_SCHEMA = "0915-robot-recovery-v21-b1-task-object-protection-manifest-v1"
COMPOSITE_SCHEMA = "0915-robot-recovery-v21-d1-clean-prep-b1-b1r-composite-closure-v1"
COMPOSITE_AUTHORITY_MODE = "B1R_POKER_HAND_CAPABILITY_PLUS_PINNED_ORIGINAL_B1_SUBORDINATE_ARTIFACTS"

FIXED_SESSIONS = (
    "play_cards_0915_044",
    "get_potato_chips_0915_097",
)
TASK_BY_SESSION = {
    "play_cards_0915_044": "playing_cards",
    "get_potato_chips_0915_097": "potato_chips",
}
SIDES = ("left", "right")
ADMITTED_SIDE = "ADMITTED_B1_PER_SIDE_HAND_MASK"
B1_USABLE_TERMINALS = {
    "PASSED_DIAGNOSTIC",
    "COMPLETED_WITH_QUALITY_REJECTION",
}

# Trust anchors for the live read-only B1/B1R closure.  Unit fixtures replace
# this dictionary in-process with their own closed fixture pins; the CLI never
# exposes an override.
PINNED_UPSTREAM_SHA256 = {
    "packages/B1R/RESULT.json": "7be40d384d7826dfcb0e57c9b4561f5f71c8e947902f52a6a42bd8c77c999d9a",
    "packages/B1R/BATCH_RESULT.json": "c964b93fff8e434faa12290481369a00ed5d4a5a9b119610ffc311ce284943d2",
    "packages/B1R/B1R_HAND_ADMISSION.json": "8dcd30418a777c7b0e120ed09ada61c3cf2f5b742138e346eaf7ae80d260a821",
    "packages/B1/RESULT.json": "62e08adceff6d21cecf1dea2a171b2e8f381250f06e75fb21fdd3d509481410d",
    "packages/B1/BATCH_RESULT.json": "07780f172a0139651aba382dae5363482f4473ee3e17b984a5af98276f38768b",
    "packages/B1/hand/sessions/potato_chips/get_potato_chips_0915_097/RESULT.json": "94e5f1e4d0ffb095642099c3fde1f6300b6eaa81ff84ae92fa95323d21367d01",
    "packages/B1/hand/sessions/potato_chips/get_potato_chips_0915_097/ROLE_MANIFEST.json": "12141059165a944c70eae0b76b7a965d44409a38d1f371cbe5f46493413703cc",
    "packages/B1/hand/sessions/potato_chips/get_potato_chips_0915_097/semantic/right_hand.npz": "e193feb3d02eba7d0192dbe8c11eb3e91ed9320a03824a53ac5f9d9f8bc2f333",
    "packages/B1/hand/sessions/potato_chips/get_potato_chips_0915_097/temporal/right_hand_STATE_LEDGER.json": "80eb45a39c33d9d537caddc0de1bc633e953a845a1ec97386e30506431ba6f12",
    "packages/B1/object_admission/playing_cards/play_cards_0915_044/B1_OBJECT_ADMISSION.json": "38db542417975d868aecab3dc45e3c62782cbe2d1d5e9acd19b4663f66cd026e",
    "packages/B1/object/sessions/playing_cards/play_cards_0915_044/RESULT.json": "e2ca3f1f8cb3b3daa6e8f161564e4ea30013bdc30fd00c1b1a0ad04775d3e5c1",
    "packages/B1/object/sessions/playing_cards/play_cards_0915_044/OBJECT_INSTANCE_MANIFEST.json": "c67adf1e45f05cf8af7a30f8bd8d4e79bbe9e73d0db95664b412293494239f22",
    "packages/B1/object/sessions/playing_cards/play_cards_0915_044/semantic/playing_card_00.npz": "6a059ea92bb684606fcadc4638afa197727154eb968863ebfed1461dcab86cf0",
    "packages/B1/object/sessions/playing_cards/play_cards_0915_044/semantic/playing_card_01.npz": "67bbcd82e559cc05b43974563fb1e3da9e5006848739623632fc4d1568865a4c",
    "packages/B1/object/sessions/playing_cards/play_cards_0915_044/semantic/playing_card_02.npz": "17575bc5b2176a695c61b68afc168c3726bd8755c28182e4989d1b88d481b450",
    "packages/B1/object/sessions/playing_cards/play_cards_0915_044/temporal/playing_card_00_STATE.json": "9e6f4215d8313ef4bd1fe6b868ef388a92b5dbeda376de57aeaa11eeeb62f374",
    "packages/B1/object/sessions/playing_cards/play_cards_0915_044/temporal/playing_card_01_STATE.json": "41ceb0d0d13177d3e82e544803e49dac5fd46cb69f6c28a74565afe9207bb41e",
    "packages/B1/object/sessions/playing_cards/play_cards_0915_044/temporal/playing_card_02_STATE.json": "20c6b5bb91ea8c59151521f64b357c1807d1e034726bc0320df8070f4f966137",
    "packages/B1/object_admission/potato_chips/get_potato_chips_0915_097/B1_OBJECT_ADMISSION.json": "fbf0ffe72a6a469f1f4333f631615423fe8a5963d9d173b7a1fc2a0e5cb4a1de",
    "packages/B1/object/sessions/potato_chips/get_potato_chips_0915_097/RESULT.json": "25ad3d06196f917b17220e05cb3330b556c32ae573ee2fafb578010c45dc677f",
    "packages/B1/object/sessions/potato_chips/get_potato_chips_0915_097/OBJECT_INSTANCE_MANIFEST.json": "0fa8679b7a08b4534d4ed84497e23028a93aa2fe2ba4af34253252c839d5c72b",
    "packages/A1/prepared/play_cards_0915_044/PHYSICAL_LEFT_SOURCEINDEX1_RESIZE_ONLY.mp4": "85aac7f6e6a65ecaed18ad0b9d1fa8fa99437e87bdca544afc4ba1c0325e51fa",
    "packages/A1/prepared/get_potato_chips_0915_097/PHYSICAL_LEFT_SOURCEINDEX1_RESIZE_ONLY.mp4": "a72dba65c7c6ff66711d4357e2336d2247e99fffae9caa981a1deab02fd520e0",
}

# Pixel source codes.  There is intentionally no old-Clean, future-donor,
# hidden-truth, or synthetic source code in this CPU preparation stage.
SOURCE_TARGET_RAW = np.uint8(0)
SOURCE_POKER_PROTECTED_IDENTITY_ADMITTED = np.uint8(1)
SOURCE_POKER_PROTECTED_IDENTITY_UNKNOWN = np.uint8(2)
SOURCE_CHIP_00_PROTECTED_RAW = np.uint8(10)
SOURCE_CHIP_01_PROTECTED_RAW = np.uint8(11)
SOURCE_CHIP_02_PROTECTED_RAW = np.uint8(12)
SOURCE_UNKNOWN_UNWRITTEN = np.uint8(250)

FORBIDDEN_KEYS = {
    "legacy_clean",
    "old_clean",
    "old_clean_result",
    "existing_clean",
    "clean_result",
    "donor_manifest",
    "real_donor_manifest",
    "temporal_donor",
    "noncausal_donor",
    "future_donor",
    "future_frame_donor",
    "hidden_ground_truth",
    "hidden_truth",
    "hidden_object_truth",
}


class ContractError(RuntimeError):
    """Fail-closed input, provenance, identity, or pixel-domain error."""


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def decoded_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii") + b"\0")
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot read JSON object: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ContractError(f"JSON object required: {path}")
    return value


def file_ref(path: Path) -> dict[str, Any]:
    target = path.resolve(strict=True)
    if not target.is_file() or target.is_symlink():
        raise ContractError(f"regular non-symlink file required: {target}")
    return {
        "path": str(target),
        "bytes": target.stat().st_size,
        "sha256": sha256_file(target),
    }


def future_ref(actual: Path, published: Path) -> dict[str, Any]:
    actual = actual.resolve(strict=True)
    if not actual.is_file() or actual.is_symlink():
        raise ContractError(f"regular staged file required: {actual}")
    return {
        "path": str(published.resolve()),
        "bytes": actual.stat().st_size,
        "sha256": sha256_file(actual),
    }


def checked_ref(value: Mapping[str, Any], label: str) -> Path:
    if not isinstance(value, Mapping) or not {"path", "bytes", "sha256"}.issubset(value):
        raise ContractError(f"{label}: path/bytes/sha256 ref required")
    path = Path(str(value["path"])).resolve(strict=True)
    actual = file_ref(path)
    for key in ("path", "bytes", "sha256"):
        if actual[key] != value.get(key):
            raise ContractError(f"{label}: {key} drift")
    return path


def checked_pinned_ref(value: Mapping[str, Any], label: str, expected_sha256: str) -> Path:
    path = checked_ref(value, label)
    if value.get("sha256") != expected_sha256:
        raise ContractError(f"{label}: immutable live authority SHA drift")
    return path


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def reject_forbidden_keys(value: Any, label: str) -> None:
    if isinstance(value, Mapping):
        forbidden = FORBIDDEN_KEYS.intersection(str(key) for key in value)
        if forbidden:
            raise ContractError(f"{label}: forbidden authority keys: {sorted(forbidden)}")
        for key, child in value.items():
            reject_forbidden_keys(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_forbidden_keys(child, f"{label}[{index}]")


def read_rgb(reference: Mapping[str, Any], label: str) -> np.ndarray:
    path = checked_ref(reference, label)
    if path.suffix.lower() != ".png":
        raise ContractError(f"{label}: lossless PNG required, not a video or preview")
    with Image.open(path) as image:
        if image.mode != "RGB":
            raise ContractError(f"{label}: exact uint8 RGB PNG required; got mode {image.mode}")
        value = np.asarray(image).copy()
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 3:
        raise ContractError(f"{label}: HxWx3 uint8 RGB required")
    return value


def read_mask(reference: Mapping[str, Any], shape: tuple[int, int], label: str) -> np.ndarray:
    path = checked_ref(reference, label)
    if path.suffix.lower() != ".png":
        raise ContractError(f"{label}: binary lossless PNG required")
    with Image.open(path) as image:
        value = np.asarray(image.convert("L"))
    if value.shape != shape or not set(np.unique(value).tolist()).issubset({0, 255}):
        raise ContractError(f"{label}: binary mask with shape {shape} required")
    return value > 0


def save_mask(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(value.astype(np.uint8) * np.uint8(255), mode="L").save(path)


def save_code_map(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(value.astype(np.uint8), mode="L").save(path)


def save_rgb(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(value, mode="RGB").save(path)


def dilate_mask(value: np.ndarray, radius: int) -> np.ndarray:
    if radius < 0:
        raise ContractError("non-negative dilation radius required")
    if radius == 0:
        return value.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(value.astype(np.uint8), kernel) > 0


def ordered_rows(value: Any, count: int, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != count:
        raise ContractError(f"{label}: exactly {count} frame rows required")
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(value):
        if not isinstance(row, dict) or row.get("frame_id") != index:
            raise ContractError(f"{label}: frame order/identity mismatch at {index}")
        rows.append(row)
    return rows


def validate_b1_handoff(reference: Mapping[str, Any]) -> tuple[dict[str, Any], Path]:
    path = checked_ref(reference, "B1 handoff")
    handoff = load_json(path)
    reject_forbidden_keys(handoff, "B1 handoff")
    if handoff.get("schema_version") != B1_HANDOFF_SCHEMA:
        raise ContractError("B1 handoff schema drift")
    if handoff.get("terminal") is not True or handoff.get("status") not in B1_USABLE_TERMINALS:
        raise ContractError("B1/B1R composite must be a usable terminal before D1_CLEAN_PREP preparation")
    if handoff.get("authority_mode") != COMPOSITE_AUTHORITY_MODE:
        raise ContractError("D1_CLEAN_PREP requires the pinned B1+B1R composite authority mode")
    if handoff.get("aggregate_gate_relaxed") is not False or handoff.get("capability_gate_relaxed") is not False:
        raise ContractError("aggregate/capability terminal gates cannot be relaxed")
    terminal_path = checked_ref(handoff.get("terminal_receipt", {}), "B1 terminal receipt")
    terminal = load_json(terminal_path)
    if terminal.get("status") != handoff.get("status"):
        raise ContractError("B1 handoff/terminal status drift")
    sessions = handoff.get("sessions")
    if not isinstance(sessions, list) or tuple(row.get("session_id") for row in sessions if isinstance(row, dict)) != FIXED_SESSIONS:
        raise ContractError("B1 handoff must contain only Poker044 then Chips097")
    closure_path = checked_ref(handoff.get("composite_closure", {}), "B1+B1R composite closure")
    closure = load_json(closure_path)
    if (
        closure.get("schema_version") != COMPOSITE_SCHEMA
        or closure.get("status") != "COMPOSITE_CLOSED_WITH_QUALITY_REJECTION"
        or closure.get("terminal") is not True
        or tuple(closure.get("fixed_sessions", ())) != FIXED_SESSIONS
        or closure.get("aggregate_gate_relaxed") is not False
        or closure.get("capability_gate_relaxed") is not False
    ):
        raise ContractError("B1+B1R composite terminal contract drift")
    pinned_artifacts = closure.get("pinned_upstream_artifacts")
    if not isinstance(pinned_artifacts, dict) or set(pinned_artifacts) != set(PINNED_UPSTREAM_SHA256):
        raise ContractError("B1+B1R composite upstream pin set drift")
    pinned_paths: dict[str, Path] = {}
    for relative, expected_sha256 in PINNED_UPSTREAM_SHA256.items():
        pinned_paths[relative] = checked_pinned_ref(
            pinned_artifacts[relative], f"composite upstream {relative}", expected_sha256
        )
    b1r_result = load_json(pinned_paths["packages/B1R/RESULT.json"])
    b1r_batch = load_json(pinned_paths["packages/B1R/BATCH_RESULT.json"])
    b1r_admission = load_json(pinned_paths["packages/B1R/B1R_HAND_ADMISSION.json"])
    b1_result = load_json(pinned_paths["packages/B1/RESULT.json"])
    if (
        b1r_result.get("status") != "COMPLETED_WITH_QUALITY_REJECTION"
        or b1r_batch.get("status") != "COMPLETED_WITH_QUALITY_REJECTION"
        or b1r_batch.get("capability") != "HAND_ONLY"
        or b1r_batch.get("session_id") != FIXED_SESSIONS[0]
        or b1r_admission.get("status") != "COMPLETED_WITH_QUALITY_REJECTION"
    ):
        raise ContractError("B1R Poker Hand capability terminal drift")
    b1r_sides = b1r_admission.get("sides") or []
    actual_b1r_sides = [
        (row.get("role"), row.get("b1r_status"), row.get("consumer_allowed"))
        for row in b1r_sides if isinstance(row, dict)
    ]
    if actual_b1r_sides != [
        ("left_hand", "REJECTED_DIRECT_OBSERVED_ADMISSION", False),
        ("right_hand", "REJECTED_TRACKER_DIRECTION_INCOMPLETE", False),
    ]:
        raise ContractError("B1R Poker Hand side admission drift")
    original = closure.get("original_b1")
    if (
        not isinstance(original, dict)
        or b1_result.get("status") != "FAILED_RUNTIME_FINAL"
        or original.get("status") != "FAILED_RUNTIME_FINAL"
        or original.get("original_b1_result_mutated") is not False
        or original.get("used_as_aggregate_terminal") is not False
    ):
        raise ContractError("original B1 FAILED_RUNTIME_FINAL immutability drift")
    if file_ref(terminal_path)["sha256"] != PINNED_UPSTREAM_SHA256["packages/B1R/RESULT.json"]:
        raise ContractError("handoff terminal must be the exact B1R result")
    closure_sessions = closure.get("sessions")
    if not isinstance(closure_sessions, list) or tuple(
        row.get("session_id") for row in closure_sessions if isinstance(row, dict)
    ) != FIXED_SESSIONS:
        raise ContractError("composite closure session scope drift")
    for handoff_row, closure_row in zip(sessions, closure_sessions):
        if handoff_row.get("hand_sides") != closure_row.get("hand"):
            raise ContractError(f"{handoff_row.get('session_id')}: handoff/composite Hand decision drift")
        handoff_object = handoff_row.get("task_object", {}).get("manifest", {})
        closure_object = closure_row.get("task_object", {}).get("manifest", {})
        if handoff_object != closure_object:
            raise ContractError(f"{handoff_row.get('session_id')}: handoff/composite Object manifest drift")
    return handoff, path


def validate_raw_manifest(
    reference: Mapping[str, Any], session_id: str, task: str, frame_count: int,
) -> tuple[dict[str, Any], Path, list[dict[str, Any]]]:
    path = checked_ref(reference, f"{session_id} raw manifest")
    manifest = load_json(path)
    reject_forbidden_keys(manifest, f"{session_id} raw manifest")
    if manifest.get("schema_version") != RAW_SCHEMA:
        raise ContractError(f"{session_id}: raw manifest schema drift")
    if manifest.get("session_id") != session_id or manifest.get("task") != task:
        raise ContractError(f"{session_id}: raw manifest identity drift")
    if manifest.get("image_domain") != "PHYSICAL_LEFT_SOURCEINDEX1_CROP_RESIZE_ONLY":
        raise ContractError(f"{session_id}: B1 image-domain mismatch")
    if manifest.get("image_size") != [1280, 960]:
        raise ContractError(f"{session_id}: expected fixed 1280x960 B1 image domain")
    if manifest.get("lossless_frames") is not True or manifest.get("frame_count") != frame_count:
        raise ContractError(f"{session_id}: raw lossless/frame closure failed")
    rows = ordered_rows(manifest.get("frames"), frame_count, f"{session_id} raw")
    return manifest, path, rows


def validate_hand_manifest(
    reference: Mapping[str, Any], session_id: str, side: str, frame_count: int,
) -> tuple[dict[str, Any], Path, list[dict[str, Any]]]:
    path = checked_ref(reference, f"{session_id}/{side} Hand manifest")
    manifest = load_json(path)
    reject_forbidden_keys(manifest, f"{session_id}/{side} Hand manifest")
    expected = {
        "schema_version": HAND_SCHEMA,
        "producer_package": "B1",
        "session_id": session_id,
        "side": side,
        "consumer_status": ADMITTED_SIDE,
        "frame_count": frame_count,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ContractError(f"{session_id}/{side}: only an admitted B1 per-side Hand manifest is consumable")
    if manifest.get("whole_session_hand_terminal_required") is not False:
        raise ContractError(f"{session_id}/{side}: whole-session Hand terminal cannot be reintroduced")
    rows = ordered_rows(manifest.get("frames"), frame_count, f"{session_id}/{side} Hand")
    return manifest, path, rows


def validate_object_manifest(
    reference: Mapping[str, Any], session_id: str, task: str, frame_count: int,
) -> tuple[dict[str, Any], Path, list[dict[str, Any]], tuple[str, ...]]:
    path = checked_ref(reference, f"{session_id} Task Object manifest")
    manifest = load_json(path)
    reject_forbidden_keys(manifest, f"{session_id} Task Object manifest")
    expected = {
        "schema_version": OBJECT_SCHEMA,
        "producer_package": "B1",
        "session_id": session_id,
        "task": task,
        "frame_count": frame_count,
        "depends_on_hand_terminal": False,
        "union_mask_created": False,
        "visible_surface_only": True,
        "hidden_shape_inferred": False,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ContractError(f"{session_id}: Task Object independence/visibility contract drift")
    instance_ids = tuple(manifest.get("instance_ids", ()))
    if task == "potato_chips":
        if instance_ids != ("potato_chip_00", "potato_chip_01", "potato_chip_02"):
            raise ContractError(f"{session_id}: Chips requires three ordered physical instances")
        if manifest.get("identity_authority") != "THREE_SEPARATE_PHYSICAL_INSTANCE_SLOTS":
            raise ContractError(f"{session_id}: Chips instance authority drift")
    else:
        allowed = {f"playing_card_{index:02d}" for index in range(3)}
        if len(instance_ids) > 3 or len(set(instance_ids)) != len(instance_ids) or any(item not in allowed for item in instance_ids):
            raise ContractError(f"{session_id}: invalid Poker instance namespace")
        identity = str(manifest.get("identity_authority", ""))
        if identity != "PASSED_INDEPENDENT_PHYSICAL_CARD_ID" and not identity.startswith("UNKNOWN_"):
            raise ContractError(f"{session_id}: Poker identity must be independently passed or UNKNOWN")
    rows = ordered_rows(manifest.get("frames"), frame_count, f"{session_id} Task Object")
    for index, row in enumerate(rows):
        instances = row.get("instances")
        if not isinstance(instances, list) or tuple(item.get("instance_id") for item in instances if isinstance(item, dict)) != instance_ids:
            raise ContractError(f"{session_id} frame {index}: separate instance rows required")
        for item in instances:
            status = str(item.get("status", ""))
            if status != "ADMITTED_VISIBLE_PROTECTION" and not status.startswith("UNKNOWN_"):
                raise ContractError(f"{session_id} frame {index}: invalid object instance status")
            if not isinstance(item.get("mask"), Mapping):
                raise ContractError(f"{session_id} frame {index}: explicit per-instance mask required")
    return manifest, path, rows, instance_ids


def validate_contract(manifest_path: Path) -> dict[str, Any]:
    manifest_path = manifest_path.resolve(strict=True)
    manifest = load_json(manifest_path)
    reject_forbidden_keys(manifest, "D1_CLEAN_PREP input")
    if manifest.get("schema_version") != INPUT_SCHEMA:
        raise ContractError("D1_CLEAN_PREP input schema drift")
    if manifest.get("execution_mode") != "CPU_PREPARE_ONLY_NO_MODEL_NO_GPU":
        raise ContractError("D1_CLEAN_PREP execution mode must stay CPU preparation only")
    if tuple(manifest.get("fixed_sessions", ())) != FIXED_SESSIONS:
        raise ContractError("D1_CLEAN_PREP is frozen to Poker044 and Chips097 only")
    handoff, handoff_path = validate_b1_handoff(manifest.get("b1_handoff", {}))
    rows = manifest.get("sessions")
    if not isinstance(rows, list) or tuple(row.get("session_id") for row in rows if isinstance(row, dict)) != FIXED_SESSIONS:
        raise ContractError("D1_CLEAN_PREP session order/scope drift")
    handoff_by_id = {row["session_id"]: row for row in handoff["sessions"]}
    sessions = []
    for row in rows:
        session_id = str(row.get("session_id", ""))
        task = TASK_BY_SESSION[session_id]
        frame_count = row.get("frame_count")
        if row.get("task") != task or not isinstance(frame_count, int) or frame_count <= 0:
            raise ContractError(f"{session_id}: task/frame denominator drift")
        b1_row = handoff_by_id[session_id]
        if b1_row.get("task") != task or b1_row.get("frame_count") != frame_count:
            raise ContractError(f"{session_id}: B1/D1_CLEAN_PREP task or frame-count drift")
        hand_sides = b1_row.get("hand_sides")
        if not isinstance(hand_sides, dict) or tuple(hand_sides) != SIDES:
            raise ContractError(f"{session_id}: exactly left/right B1 side decisions required")
        validated_sides: dict[str, Any] = {}
        for side in SIDES:
            decision = hand_sides[side]
            if not isinstance(decision, dict):
                raise ContractError(f"{session_id}/{side}: B1 side decision object required")
            status = str(decision.get("status", ""))
            mask_ref = decision.get("mask_manifest")
            if status == ADMITTED_SIDE:
                side_manifest, side_path, side_rows = validate_hand_manifest(
                    mask_ref, session_id, side, frame_count
                )
                validated_sides[side] = {
                    "status": status,
                    "manifest": side_manifest,
                    "path": side_path,
                    "frames": side_rows,
                }
            elif status.startswith("UNKNOWN_") and mask_ref is None:
                validated_sides[side] = {"status": status, "manifest": None, "path": None, "frames": None}
            else:
                raise ContractError(f"{session_id}/{side}: unadmitted masks cannot be consumed")
        object_decision = b1_row.get("task_object")
        if not isinstance(object_decision, dict) or object_decision.get("depends_on_hand_terminal") is not False:
            raise ContractError(f"{session_id}: Task Object must remain independent of Hand terminal")
        object_manifest, object_path, object_rows, instance_ids = validate_object_manifest(
            object_decision.get("manifest", {}), session_id, task, frame_count
        )
        raw_manifest, raw_path, raw_rows = validate_raw_manifest(
            row.get("raw_frame_manifest", {}), session_id, task, frame_count
        )
        sessions.append({
            "session_id": session_id,
            "task": task,
            "frame_count": frame_count,
            "raw_manifest": raw_manifest,
            "raw_path": raw_path,
            "raw_frames": raw_rows,
            "hand_sides": validated_sides,
            "object_manifest": object_manifest,
            "object_path": object_path,
            "object_frames": object_rows,
            "instance_ids": instance_ids,
        })
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "handoff": handoff,
        "handoff_path": handoff_path,
        "sessions": sessions,
    }


def audit_domains(
    raw: np.ndarray,
    candidate: np.ndarray,
    m_remove: np.ndarray,
    m_flow: np.ndarray,
    m_write: np.ndarray,
    unknown: np.ndarray,
    protected: np.ndarray,
) -> dict[str, int | bool]:
    if not (raw.shape == candidate.shape and raw.dtype == candidate.dtype == np.uint8):
        raise ContractError("Raw/candidate RGB domain mismatch")
    shape = raw.shape[:2]
    masks = (m_remove, m_flow, m_write, unknown, protected)
    if any(mask.shape != shape or mask.dtype != np.bool_ for mask in masks):
        raise ContractError("mask domain mismatch")
    if np.any(m_remove & ~m_write):
        raise ContractError("M_remove must be a subset of M_write")
    if np.any(m_write & ~m_flow):
        raise ContractError("M_write must be a subset of M_flow")
    if np.any(m_write & protected):
        raise ContractError("Task Object protection must veto M_write")
    if not np.array_equal(unknown, m_write):
        raise ContractError("zero-materialization UNKNOWN must equal M_write")
    changed = np.any(raw != candidate, axis=2)
    changed_outside = changed & ~m_write
    changed_protected = changed & protected
    if np.any(changed_outside):
        raise ContractError("candidate changed Raw pixels outside M_write")
    if np.any(changed_protected):
        raise ContractError("candidate changed protected Task Object Raw pixels")
    return {
        "m_remove_pixels": int(m_remove.sum()),
        "m_flow_pixels": int(m_flow.sum()),
        "m_write_pixels": int(m_write.sum()),
        "unknown_pixels": int(unknown.sum()),
        "protected_pixels": int(protected.sum()),
        "changed_pixels": int(changed.sum()),
        "changed_outside_m_write_pixels": int(changed_outside.sum()),
        "changed_protected_pixels": int(changed_protected.sum()),
        "outside_m_write_raw_pixel_byte_exact": True,
        "protected_object_raw_pixel_byte_exact": True,
    }


def process_session(
    session: dict[str, Any], staged_root: Path, published_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    session_id = session["session_id"]
    task = session["task"]
    frame_count = session["frame_count"]
    staged = staged_root / "sessions" / session_id
    published = published_root / "sessions" / session_id
    frame_records = []
    totals = {
        "m_remove_pixels": 0,
        "m_flow_pixels": 0,
        "m_write_pixels": 0,
        "unknown_pixels": 0,
        "protected_pixels": 0,
        "changed_pixels": 0,
        "changed_outside_m_write_pixels": 0,
        "changed_protected_pixels": 0,
    }
    unknown_sides = [
        side for side in SIDES if session["hand_sides"][side]["status"] != ADMITTED_SIDE
    ]
    poker_identity = session["object_manifest"].get("identity_authority") if task == "playing_cards" else None

    for frame_id in range(frame_count):
        raw = read_rgb(session["raw_frames"][frame_id].get("rgb", {}), f"{session_id} frame {frame_id} Raw")
        shape = raw.shape[:2]
        if shape != (960, 1280):
            raise ContractError(f"{session_id} frame {frame_id}: expected fixed 1280x960 Raw")
        hand_masks: dict[str, np.ndarray] = {}
        for side in SIDES:
            side_data = session["hand_sides"][side]
            if side_data["status"] == ADMITTED_SIDE:
                mask = read_mask(
                    side_data["frames"][frame_id].get("mask", {}), shape,
                    f"{session_id} frame {frame_id} {side} Hand",
                )
                if side_data["frames"][frame_id].get("observed") is False and np.any(mask):
                    raise ContractError(f"{session_id} frame {frame_id} {side}: unobserved Hand mask must be empty")
                hand_masks[side] = mask
            else:
                hand_masks[side] = np.zeros(shape, dtype=bool)
        hand_union = hand_masks["left"] | hand_masks["right"]

        protected = np.zeros(shape, dtype=bool)
        source_map = np.full(shape, SOURCE_TARGET_RAW, dtype=np.uint8)
        instance_pixel_counts: dict[str, int] = {}
        for instance_index, item in enumerate(session["object_frames"][frame_id]["instances"]):
            mask = read_mask(
                item["mask"], shape,
                f"{session_id} frame {frame_id} object {item['instance_id']}",
            )
            if item.get("observed") is False and np.any(mask):
                raise ContractError(f"{session_id} frame {frame_id} {item['instance_id']}: unobserved mask must be empty")
            if np.any(protected & mask):
                raise ContractError(f"{session_id} frame {frame_id}: instance masks overlap/union")
            protected |= mask
            instance_pixel_counts[item["instance_id"]] = int(mask.sum())
            if task == "potato_chips":
                source_map[mask] = (
                    SOURCE_CHIP_00_PROTECTED_RAW,
                    SOURCE_CHIP_01_PROTECTED_RAW,
                    SOURCE_CHIP_02_PROTECTED_RAW,
                )[instance_index]
            elif poker_identity == "PASSED_INDEPENDENT_PHYSICAL_CARD_ID":
                source_map[mask] = SOURCE_POKER_PROTECTED_IDENTITY_ADMITTED
            else:
                source_map[mask] = SOURCE_POKER_PROTECTED_IDENTITY_UNKNOWN

        # Match the live Clean-20 contract: removal is the admitted semantic
        # Hand domain after the object veto; write is the allowed replacement
        # domain; flow is a larger context domain.  With no independently
        # justified adaptive expansion for these B1 masks, fail closed to
        # M_write == M_remove instead of resurrecting old broad dilation.
        m_remove = hand_union & ~protected
        m_write = m_remove.copy()
        m_flow = dilate_mask(m_write, 16)
        unknown = m_write.copy()
        candidate = raw.copy()
        source_map[unknown] = SOURCE_UNKNOWN_UNWRITTEN
        metrics = audit_domains(raw, candidate, m_remove, m_flow, m_write, unknown, protected)
        for key in totals:
            totals[key] += int(metrics[key])

        paths = {
            "candidate_rgb": staged / "raw_candidate_frames" / f"{frame_id:06d}.png",
            "m_remove": staged / "masks" / "M_REMOVE" / f"{frame_id:06d}.png",
            "m_flow": staged / "masks" / "M_FLOW" / f"{frame_id:06d}.png",
            "m_write": staged / "masks" / "M_WRITE" / f"{frame_id:06d}.png",
            "unknown": staged / "masks" / "UNKNOWN" / f"{frame_id:06d}.png",
            "source_map": staged / "source_maps" / f"{frame_id:06d}.png",
        }
        save_rgb(paths["candidate_rgb"], candidate)
        save_mask(paths["m_remove"], m_remove)
        save_mask(paths["m_flow"], m_flow)
        save_mask(paths["m_write"], m_write)
        save_mask(paths["unknown"], unknown)
        save_code_map(paths["source_map"], source_map)
        frame_records.append({
            "frame_id": frame_id,
            "raw_rgb": session["raw_frames"][frame_id]["rgb"],
            "candidate_rgb": future_ref(paths["candidate_rgb"], published / "raw_candidate_frames" / paths["candidate_rgb"].name),
            "M_remove": future_ref(paths["m_remove"], published / "masks" / "M_REMOVE" / paths["m_remove"].name),
            "M_flow": future_ref(paths["m_flow"], published / "masks" / "M_FLOW" / paths["m_flow"].name),
            "M_write": future_ref(paths["m_write"], published / "masks" / "M_WRITE" / paths["m_write"].name),
            "UNKNOWN": future_ref(paths["unknown"], published / "masks" / "UNKNOWN" / paths["unknown"].name),
            "source_map": future_ref(paths["source_map"], published / "source_maps" / paths["source_map"].name),
            "raw_decoded_sha256": decoded_sha256(raw),
            "candidate_decoded_sha256": decoded_sha256(candidate),
            "hand_side_status": {side: session["hand_sides"][side]["status"] for side in SIDES},
            "object_instance_pixels": instance_pixel_counts,
            **metrics,
        })

    frame_manifest_path = staged / "FRAME_MANIFEST.json"
    write_json(frame_manifest_path, {
        "schema_version": f"{SCHEMA}-frame-manifest-v1",
        "created_at": now(),
        "session_id": session_id,
        "task": task,
        "frame_count": frame_count,
        "object_lane_depends_on_hand_terminal": False,
        "unknown_hand_sides": unknown_sides,
        "poker_identity": poker_identity,
        "chips_instance_mode": "THREE_SEPARATE_NO_UNION" if task == "potato_chips" else None,
        "source_kind_codes": {
            "0": "TARGET_RAW_SAME_FRAME_SAME_XY",
            "1": "POKER_PROTECTED_RAW_IDENTITY_ADMITTED",
            "2": "POKER_PROTECTED_RAW_IDENTITY_UNKNOWN",
            "10": "POTATO_CHIP_00_PROTECTED_RAW",
            "11": "POTATO_CHIP_01_PROTECTED_RAW",
            "12": "POTATO_CHIP_02_PROTECTED_RAW",
            "250": "UNKNOWN_UNWRITTEN_NO_PIXEL_AUTHORITY",
        },
        "source_coordinate_contract": {
            "codes_0_1_2_10_11_12": "source_frame=target_frame; source_xy=target_xy",
            "code_250": "source_frame/source_xy=NONE",
        },
        "frames": frame_records,
    })
    has_unknown_prereq = bool(unknown_sides) or totals["unknown_pixels"] > 0 or (
        task == "playing_cards" and poker_identity != "PASSED_INDEPENDENT_PHYSICAL_CARD_ID"
    )
    session_result = {
        "schema_version": f"{SCHEMA}-session-result-v1",
        "created_at": now(),
        "session_id": session_id,
        "task": task,
        "status": "PREPARED_BLOCKED_UNKNOWN_OR_FRESH_INPAINTING" if has_unknown_prereq else "PREPARED_ZERO_WRITE",
        "clean_terminal": False,
        "frame_count": frame_count,
        "M_write_is_frozen_allowed_domain_not_materialized_claim": True,
        "materialized_changed_pixels": totals["changed_pixels"],
        "fresh_inpainting_executed": False,
        "old_clean_consumed": False,
        "future_frame_donor_consumed": False,
        "hidden_ground_truth_consumed": False,
        "object_lane_depends_on_hand_terminal": False,
        "unknown_hand_sides": unknown_sides,
        "poker_identity": poker_identity,
        "chips_instances_unioned": False,
        "totals": totals,
        "hard_gates": {
            "fixed_session": True,
            "b1_per_side_admission_only": True,
            "M_remove_subset_M_write_subset_M_flow": True,
            "M_write_equals_object_vetoed_M_remove_in_fail_closed_mode": True,
            "Task_Object_vetoes_M_write": True,
            "M_write_outside_raw_pixel_byte_exact": totals["changed_outside_m_write_pixels"] == 0,
            "protected_object_raw_pixel_byte_exact": totals["changed_protected_pixels"] == 0,
            "full_frame_source_map": len(frame_records) == frame_count,
            "no_old_clean_or_future_donor_or_hidden_truth": True,
            "chips_three_instances_not_unioned": task != "potato_chips" or len(session["instance_ids"]) == 3,
        },
        "artifacts": {
            "frame_manifest": future_ref(frame_manifest_path, published / "FRAME_MANIFEST.json"),
        },
        "input_pins": {
            "raw_frame_manifest": file_ref(session["raw_path"]),
            "hand_side_manifests": {
                side: file_ref(session["hand_sides"][side]["path"])
                if session["hand_sides"][side]["path"] is not None else None
                for side in SIDES
            },
            "task_object_manifest": file_ref(session["object_path"]),
        },
        "training_eligible": False,
        "claim_limit": "CPU preparation and pixel-domain audit only; no inpainting quality, hidden appearance, training, contact, control, or deployment authority.",
    }
    session_result_path = staged / "RESULT.json"
    write_json(session_result_path, session_result)
    return session_result, {
        "session_id": session_id,
        "task": task,
        "status": session_result["status"],
        "result": future_ref(session_result_path, published / "RESULT.json"),
        "frame_manifest": session_result["artifacts"]["frame_manifest"],
        "unknown_hand_sides": unknown_sides,
        "poker_identity": poker_identity,
        "totals": totals,
    }


def prepare(manifest_path: Path, output_root: Path) -> dict[str, Any]:
    validated = validate_contract(manifest_path)
    output_root = output_root.resolve()
    if output_root.exists() or output_root.is_symlink():
        raise ContractError(f"fresh output root required: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staged_root = output_root.parent / f".{output_root.name}.staging-{os.getpid()}-{uuid.uuid4().hex}"
    staged_root.mkdir()
    try:
        session_rows = []
        for session in validated["sessions"]:
            _, summary = process_session(session, staged_root, output_root)
            session_rows.append(summary)
        registration_path = staged_root / "FRESH_INPAINTING_REGISTRATION.json"
        write_json(registration_path, {
            "schema_version": f"{SCHEMA}-fresh-inpainting-registration-v1",
            "created_at": now(),
            "status": "REGISTERED_INPUT_CLOSURE_NOT_EXECUTED",
            "fixed_sessions": list(FIXED_SESSIONS),
            "gpu_requested": False,
            "model_loaded": False,
            "inference_executed": False,
            "input_contract": "Per-frame lossless Raw + M_flow hole + UNKNOWN + separate Task Object masks, all transitively SHA-bound by each FRAME_MANIFEST.",
            "required_fresh_output_contract": [
                "lossless clean candidate frames",
                "the frozen M_write allowed-write domain per frame",
                "per-pixel source map with no future-frame donor authority",
                "M_remove subset M_write subset M_flow",
                "decoded Raw equality at every pixel outside M_write",
                "protected Task Object decoded Raw equality",
                "UNKNOWN for every M_write pixel lacking an authorized source",
            ],
            "forbidden_authorities": [
                "old Clean output",
                "future-frame donor",
                "hidden ground truth",
                "Poker hidden appearance without independent physical-card identity",
                "Chips union mask",
            ],
            "sessions": [
                {"session_id": row["session_id"], "frame_manifest": row["frame_manifest"]}
                for row in session_rows
            ],
            "claim_limit": "Registration and input closure only. This file is not permission to launch a GPU/model job.",
        })
        blocked = any(
            row["unknown_hand_sides"]
            or row["totals"]["unknown_pixels"] > 0
            or (row["task"] == "playing_cards" and row["poker_identity"] != "PASSED_INDEPENDENT_PHYSICAL_CARD_ID")
            for row in session_rows
        )
        result = {
            "schema_version": f"{SCHEMA}-result-v1",
            "created_at": now(),
            "status": "BLOCKED_PREREQ_FRESH_INPAINTING_OR_UNKNOWN" if blocked else "PREPARED_ZERO_WRITE_NO_CLEAN_CLAIM",
            "stage_terminal": True,
            "clean_terminal": False,
            "execution_mode": "CPU_PREPARE_ONLY_NO_MODEL_NO_GPU",
            "fixed_sessions": list(FIXED_SESSIONS),
            "b1_terminal_status": validated["handoff"]["status"],
            "b1_handoff": file_ref(validated["handoff_path"]),
            "input_manifest": file_ref(validated["manifest_path"]),
            "sessions": session_rows,
            "artifacts": {
                "fresh_inpainting_registration": future_ref(
                    registration_path, output_root / "FRESH_INPAINTING_REGISTRATION.json"
                )
            },
            "gpu_used": False,
            "model_execution_performed": False,
            "old_clean_consumed": False,
            "future_frame_donor_consumed": False,
            "hidden_ground_truth_consumed": False,
            "training_eligible": False,
            "accuracy_reported": False,
            "claim_limit": "D1_CLEAN_PREP CPU orchestration closure only; M_write is an allowed domain, no fill is materialized, and UNKNOWN is retained. No Clean quality, training, contact, control, or deployment authority.",
        }
        result_path = staged_root / "RESULT.json"
        write_json(result_path, result)
        write_json(staged_root / "RUN_RECEIPT.json", {
            "schema_version": f"{SCHEMA}-run-receipt-v1",
            "status": result["status"],
            "result": future_ref(result_path, output_root / "RESULT.json"),
            "gpu_used": False,
            "model_execution_performed": False,
        })
        os.replace(staged_root, output_root)
        return load_json(output_root / "RESULT.json")
    except Exception:
        shutil.rmtree(staged_root, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preflight-receipt", type=Path)
    args = parser.parse_args()
    try:
        if args.preflight_only:
            validated = validate_contract(args.input_manifest)
            payload = {
                "schema_version": f"{SCHEMA}-preflight-receipt-v1",
                "created_at": now(),
                "status": "PASSED_CPU_INPUT_CONTRACT_PREFLIGHT",
                "fixed_sessions": list(FIXED_SESSIONS),
                "b1_status": validated["handoff"]["status"],
                "input_manifest": file_ref(validated["manifest_path"]),
                "b1_handoff": file_ref(validated["handoff_path"]),
                "composite_closure": validated["handoff"]["composite_closure"],
                "poker_hand": "BOTH_SIDES_UNKNOWN_NO_MASK_CONSUMED",
                "chips_hand": "LEFT_UNKNOWN_RIGHT_393_OF_394_PROXY_FRAME_58_UNKNOWN",
                "poker_object": "THREE_VISIBLE_PROTECTION_CANDIDATES_PHYSICAL_IDENTITY_UNKNOWN",
                "chips_object": "THREE_SEPARATE_UNKNOWN_NOT_ABSENT_EMPTY_SLOTS_NO_UNION",
                "gpu_used": False,
                "model_execution_performed": False,
                "clean_terminal": False,
                "claim_limit": "D1_CLEAN_PREP CPU input closure only; no fill, Clean quality, training, contact, control, or deployment authority.",
            }
            if args.preflight_receipt is not None:
                receipt_path = args.preflight_receipt.resolve()
                if receipt_path.exists() or receipt_path.is_symlink():
                    raise ContractError(f"fresh preflight receipt required: {receipt_path}")
                write_json(receipt_path, payload)
        elif args.preflight_receipt is not None:
            raise ContractError("--preflight-receipt requires --preflight-only")
        else:
            payload = prepare(args.input_manifest, args.output_root)
    except (ContractError, OSError, ValueError) as error:
        print(json.dumps({
            "status": "BLOCKED_FAIL_CLOSED",
            "error": f"{type(error).__name__}: {error}",
            "gpu_used": False,
            "model_execution_performed": False,
        }, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
