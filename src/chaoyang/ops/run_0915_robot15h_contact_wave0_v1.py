#!/usr/bin/env python3
"""Build fail-closed W0 Contact diagnostics and tactile-supported hypotheses.

The strict Contact branch deliberately remains closed while FoundationStereo
and visible-patch Object6D have no external metric authority.  A separate
hypothesis ledger may contain only independently checked visual-overlap plus
finger-specific tactile evidence; it is never a Contact label or training data.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence
import uuid

import cv2
import numpy as np

from chaoyang.governance.robot15h_task_specs_v1 import WINDOW_RUN_ID, build_packet


ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "0915_robot15h_contact_dual_evidence_v1"
PHASE = "ROBOT15H_CONTACT_W0_DUAL_EVIDENCE_DIAGNOSTIC"
AUTHORITY = "DEVELOPMENT_RELATIVE_NON_CONTROL_NON_DEPLOYABLE"
TACTILE_SCHEMA = "tactile-acquisition-aligned-frame-v2"
HANDS = ("left", "right")
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
OBJECTS = ("playing_card_00", "playing_card_01", "playing_card_02")
GEOMETRY_NEAR_M = 0.005
TACTILE_MAX_OFFSET_MS = 40.0
MAX_DEPTH_SIGMA_M = 0.005
MAX_PLANE_RESIDUAL_M = 0.005
MAX_LR_RESIDUAL_PX = 1.0
HYPOTHESIS_SESSION = "play_cards_0915_119"
HYPOTHESIS_PAIR = ("right", "index", "playing_card_02")
HYPOTHESIS_FRAMES = frozenset(range(92, 99))
INVENTORY = ROOT / (
    "_run/current/0915_robot15h_window_start_inventory_v1/attempts/"
    "attempt_0001/BATCH_MANIFEST.json"
)
INTERACTION_ROOT = ROOT / (
    "_run/current/0915_robot15h_interaction_occlusion_v1/attempts/attempt_0001"
)
DEPTH_ROOT = ROOT / (
    "_run/current/0915_robot15h_foundationstereo_wave0_recovery_v1/"
    "attempts/attempt_0001"
)
OBJECT_ROOT = ROOT / (
    "_run/current/0915_robot15h_geometry_object6d_wave0_v1/attempts/attempt_0001"
)
OUTPUT = ROOT / f"_run/current/{TASK_ID}/attempts/attempt_0001"
VISUAL = ROOT / "docs/current/visuals/0915_ROBOT15H_W0_CONTACT_V1"
RECEIPT = ROOT / "tasks/receipts/0915_ROBOT15H_CONTACT_DUAL_EVIDENCE_V1_RESULT.json"
CLAIM_LIMIT = (
    "Development diagnostics only. Strict Contact and R1-E remain closed because "
    "external metric authority is absent. The separately named tactile-supported "
    "rows are hypotheses only: no force, probability, ground truth, causal training, "
    "control, deployment, complete geometry or cross-recording-generalization claim."
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
        "path": str(candidate),
        "bytes": candidate.stat().st_size,
        "sha256": sha256(candidate),
    }


def projected_ref(existing: Path, destination: Path) -> dict[str, Any]:
    source = existing.resolve(strict=True)
    return {
        "path": str(destination.resolve()),
        "bytes": source.stat().st_size,
        "sha256": sha256(source),
    }


def verify_ref(value: Mapping[str, Any]) -> Path:
    path = Path(str(value["path"])).resolve(strict=True)
    if path.stat().st_size != int(value["bytes"]) or sha256(path) != value["sha256"]:
        raise RuntimeError(f"artifact reference drift: {path}")
    return path


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def canonical_sha(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def process_start_ticks(pid: int) -> int:
    return int(Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()[21])


def w0_rows(inventory: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = [dict(row) for row in inventory.get("sessions", []) if row.get("wave") == "W0"]
    if inventory.get("cohort_denominator") != 220 or len(rows) != 4:
        raise RuntimeError("frozen 0915 W0 must contain exactly four of 220 sessions")
    expected = {
        "play_cards_0915_031", "play_cards_0915_119",
        "get_potato_chips_0915_007", "get_potato_chips_0915_042",
    }
    if {str(row.get("session_id")) for row in rows} != expected:
        raise RuntimeError("frozen W0 identity set drift")
    return rows


def rows_by_session(batch: Mapping[str, Any], expected: Sequence[str]) -> dict[str, Mapping[str, Any]]:
    key = "sessions" if isinstance(batch.get("sessions"), list) else "rows"
    rows = batch.get(key)
    if not isinstance(rows, list):
        raise RuntimeError("upstream batch lacks session rows")
    result = {str(row.get("session_id")): row for row in rows if isinstance(row, Mapping)}
    if set(result) != set(expected):
        raise RuntimeError("upstream batch session identity differs from frozen W0")
    return result


def validate_route() -> tuple[dict[str, Any], Path]:
    packet_path = ROOT / f"tasks/current/{TASK_ID}/TASK_PACKET.json"
    packet = load_json(packet_path)
    if packet != build_packet(TASK_ID) or packet.get("weights") != "ABSENT":
        raise RuntimeError("task packet differs from frozen weights-ABSENT specification")
    state = load_json(ROOT / "docs/governance/LONG_HORIZON_TASK_STATE.json")
    if state.get("next_task", {}).get("task_id") != TASK_ID:
        raise RuntimeError(f"{TASK_ID} is not current next_task")
    index = load_json(ROOT / "tasks/current/INDEX.json")
    route = next(
        (row for row in index.get("task_packets", []) if row.get("task_id") == TASK_ID), None,
    )
    if (
        route is None or route.get("execution_allowed") is not True
        or route.get("packet_sha256") != sha256(packet_path)
    ):
        raise RuntimeError(f"{TASK_ID} is not SHA-bound routable")
    return packet, packet_path


def validate_writer_claim(
    path: Path, *, signature_sha: str, executor_epoch: int, fencing_sha: str,
) -> dict[str, Any]:
    claim = load_json(path)
    pid, ticks = claim.get("pid"), claim.get("proc_start_ticks")
    if (
        claim.get("schema_version") != "0915-robot15h-contact-writer-claim-v1"
        or claim.get("task_id") != TASK_ID or claim.get("status") != "CLAIMED"
        or claim.get("weights") != "ABSENT" or claim.get("gpu_used") is not False
        or claim.get("unique_write_root") != str(OUTPUT.resolve())
        or claim.get("executor_epoch") != executor_epoch
        or claim.get("run_signature_sha256") != signature_sha
        or claim.get("fencing_token_sha256") != fencing_sha
        or not isinstance(pid, int) or not isinstance(ticks, int)
        or process_start_ticks(pid) != ticks
    ):
        raise RuntimeError("Contact writer claim/fence mismatch")
    return claim


def heartbeat() -> None:
    completed = subprocess.run(
        [
            sys.executable, "-m", "chaoyang.governance.heartbeat_task",
            "--task-id", TASK_ID, "--pid", str(os.getpid()),
            "--status", "RUNNING", "--phase", PHASE,
        ],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            "governance heartbeat failed: " + (completed.stderr or completed.stdout)[-4000:]
        )


def session_root(inventory_row: Mapping[str, Any]) -> Path:
    video = verify_ref(inventory_row["source_stereo"])
    root = video.parent.parent
    expected = str(inventory_row["session_id"])
    if root.name != expected:
        raise RuntimeError(f"source session identity drift: {root}")
    return root


def tactile_paths(inventory_row: Mapping[str, Any]) -> list[Path]:
    root = session_root(inventory_row) / "preprocess/all_data"
    paths = [root / f"{frame:05d}/training_data.json" for frame in range(int(inventory_row["frame_count"]))]
    if not all(path.is_file() for path in paths):
        raise RuntimeError(f"processed tactile frame set incomplete: {root}")
    return paths


def tactile_manifest(inventory_row: Mapping[str, Any]) -> dict[str, Any]:
    references = [ref(path) for path in tactile_paths(inventory_row)]
    return {
        "session_id": inventory_row["session_id"],
        "frame_count": int(inventory_row["frame_count"]),
        "schema": TACTILE_SCHEMA,
        "source": "processed/preprocess/all_data/*/training_data.json:entities.tactile",
        "ordered_reference_manifest_sha256": canonical_sha(references),
        "total_bytes": sum(int(item["bytes"]) for item in references),
    }


def tactile_finger_evidence(
    tactile: Mapping[str, Any], *, hand: str, finger: str,
) -> dict[str, Any]:
    if hand not in HANDS or finger not in FINGERS:
        raise RuntimeError("unknown tactile hand/finger identity")
    side = tactile.get(hand)
    if not isinstance(side, Mapping) or side.get("schema_version") != TACTILE_SCHEMA:
        raise RuntimeError(f"processed tactile-v2 required for {hand}")
    grid = np.asarray(side.get("finger_grid_5x4x8"))
    active = np.asarray(side.get("active_mask_5x4x8"), bool)
    valid = np.asarray(side.get("valid_mask_5x4x8"), bool)
    if grid.shape != (5, 4, 8) or active.shape != grid.shape or valid.shape != grid.shape:
        raise RuntimeError(f"tactile finger-grid contract drift for {hand}")
    raw_offset = side.get("offline_source_offset_ms")
    offset = float(raw_offset) if isinstance(raw_offset, (int, float)) else math.nan
    source_valid = bool(side.get("offline_source_valid") is True)
    time_aligned = bool(source_valid and math.isfinite(offset) and abs(offset) <= TACTILE_MAX_OFFSET_MS)
    index = FINGERS.index(finger)
    accepted = (grid[index] != 0) & active[index] & valid[index]
    return {
        "hand_id": hand,
        "finger_id": finger,
        "grid_index": index,
        "grid_shape": [5, 4, 8],
        "offline_source_valid": source_valid,
        "offline_source_offset_ms": offset if math.isfinite(offset) else None,
        "time_aligned_within_40ms": time_aligned,
        "valid_taxel_count": int(np.count_nonzero(valid[index])),
        "active_valid_nonzero_taxel_count": int(np.count_nonzero(accepted)) if time_aligned else 0,
        "finger_specific_tactile_active": bool(time_aligned and np.any(accepted)),
        "force_claim": False,
    }


def load_tactile_timeline(
    inventory_row: Mapping[str, Any], interaction_rows: Sequence[Mapping[str, Any]] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_frame_time: dict[int, float] = {}
    if interaction_rows is not None:
        for row in interaction_rows:
            frame = int(row["frame_id"])
            timestamp = float(row["timestamp_s"])
            previous = by_frame_time.setdefault(frame, timestamp)
            if abs(previous - timestamp) > 1e-9:
                raise RuntimeError("Interaction row timestamp differs within frame")
    timeline: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    for frame, path in enumerate(tactile_paths(inventory_row)):
        document = load_json(path)
        metadata = document.get("metadata", {})
        if int(metadata.get("idx", -1)) != frame:
            raise RuntimeError(f"processed tactile frame identity drift: {path}")
        timestamp = float(metadata.get("video_time_s"))
        if not math.isfinite(timestamp):
            raise RuntimeError(f"nonfinite processed frame timestamp: {path}")
        if by_frame_time and (
            frame not in by_frame_time or abs(timestamp - by_frame_time[frame]) > 1e-6
        ):
            raise RuntimeError(f"Interaction/tactile timestamp mismatch at frame {frame}")
        tactile = document.get("entities", {}).get("tactile")
        if not isinstance(tactile, Mapping) or set(tactile) != set(HANDS):
            raise RuntimeError(f"processed tactile side contract drift: {path}")
        timeline.append({
            "frame_id": frame,
            "timestamp_s": timestamp,
            "fingers": {
                f"{hand}:{finger}": tactile_finger_evidence(
                    tactile, hand=hand, finger=finger,
                )
                for hand in HANDS for finger in FINGERS
            },
        })
        references.append(ref(path))
    times = np.asarray([row["timestamp_s"] for row in timeline], np.float64)
    if len(times) > 1 and np.any(np.diff(times) <= 0):
        raise RuntimeError("processed tactile video timeline is not strictly increasing")
    return timeline, {
        "frame_count": len(timeline),
        "ordered_reference_manifest_sha256": canonical_sha(references),
        "total_bytes": sum(int(item["bytes"]) for item in references),
    }


def validate_external_authority(
    depth_result: Mapping[str, Any], object_document: Mapping[str, Any],
) -> dict[str, Any]:
    depth_closed = depth_result.get("strict_metric_contact_authorized") is False
    object_closed = object_document.get("contact_authority") == "NONE"
    if not depth_closed or not object_closed:
        raise RuntimeError("unexpected strict metric Contact authority escalation")
    return {
        "depth_strict_metric_contact_authorized": False,
        "object6d_contact_authority": "NONE",
        "external_metric_authority": False,
        "strict_contact_authorized": False,
        "r1_e_authorized": False,
    }


def evaluate_pair(
    interaction: Mapping[str, Any], tactile: Mapping[str, Any],
    *, external_metric_authority: bool,
) -> dict[str, Any]:
    metric = interaction.get("metric")
    sample = interaction.get("finger_associated_visible_surface_point")
    metric = metric if isinstance(metric, Mapping) else None
    sample = sample if isinstance(sample, Mapping) else {}
    distance = metric.get("finite_patch_distance_m") if metric is not None else None
    distance = float(distance) if isinstance(distance, (int, float)) else None
    inside = bool(metric is not None and metric.get("inside_visible_patch") is True)
    geometry_near = bool(
        distance is not None and math.isfinite(distance)
        and distance <= GEOMETRY_NEAR_M and inside
    )
    quality = sample.get("association_quality")
    quality = quality if isinstance(quality, Mapping) else {}
    depth_sigma = quality.get("local_depth_robust_sigma_m")
    lr_residual = quality.get("lr_residual_median_px")
    plane_residual = metric.get("object_plane_residual_p90_m") if metric else None
    numeric_uncertainty = all(
        isinstance(value, (int, float)) and math.isfinite(float(value))
        for value in (depth_sigma, lr_residual, plane_residual)
    )
    component_limits_pass = bool(
        numeric_uncertainty
        and float(depth_sigma) <= MAX_DEPTH_SIGMA_M
        and float(lr_residual) <= MAX_LR_RESIDUAL_PX
        and float(plane_residual) <= MAX_PLANE_RESIDUAL_M
    )
    pixel_registration_bound = False
    uncertainty_admitted = bool(component_limits_pass and pixel_registration_bound)
    tactile_supported = bool(
        tactile.get("time_aligned_within_40ms") is True
        and tactile.get("finger_specific_tactile_active") is True
    )
    strict = bool(
        geometry_near and uncertainty_admitted and tactile_supported
        and external_metric_authority
    )
    blockers: list[str] = []
    if not geometry_near:
        blockers.append("NOT_WITHIN_FIXED_5MM_FINITE_VISIBLE_PATCH")
    if not uncertainty_admitted:
        blockers.append("COMPLETE_CONTACT_DISTANCE_UNCERTAINTY_NOT_BOUND")
    if not tactile_supported:
        blockers.append("NO_ALIGNED_FINGER_SPECIFIC_TACTILE_ACTIVITY")
    if not external_metric_authority:
        blockers.append("EXTERNAL_METRIC_AUTHORITY_UNVERIFIED")
    return {
        "pair_key": interaction.get("pair_key"),
        "frame_id": int(interaction["frame_id"]),
        "timestamp_s": float(interaction["timestamp_s"]),
        "hand_id": interaction["hand_id"],
        "finger_id": interaction["finger_id"],
        "object_id": interaction["object_id"],
        "finite_patch_distance_m": distance,
        "inside_visible_patch": inside,
        "geometric_proximity_within_fixed_5mm": geometry_near,
        "uncertainty": {
            "local_depth_robust_sigma_m": depth_sigma,
            "lr_residual_median_px": lr_residual,
            "object_plane_residual_p90_m": plane_residual,
            "pixel_registration_uncertainty": "UNBOUND",
            "component_limits_pass": component_limits_pass,
            "admitted": uncertainty_admitted,
            "policy": "UNCERTAINTY_NEVER_EXPANDS_THE_FIXED_5MM_DISTANCE_GATE",
        },
        "tactile": dict(tactile),
        "strict_contact_admitted": strict,
        "r1_e_admitted": False,
        "blockers": blockers,
    }


def weak_hypothesis_row(
    interaction: Mapping[str, Any], tactile: Mapping[str, Any],
    *, object_plane_observed: bool,
) -> dict[str, Any] | None:
    identity = (
        str(interaction.get("hand_id")), str(interaction.get("finger_id")),
        str(interaction.get("object_id")),
    )
    frame = int(interaction.get("frame_id", -1))
    adjacency = interaction.get("two_d_adjacency")
    adjacency = adjacency if isinstance(adjacency, Mapping) else {}
    sample = interaction.get("finger_associated_visible_surface_point")
    sample = sample if isinstance(sample, Mapping) else {}
    pixel = np.asarray(sample.get("source_pixel_uv", []), np.float64)
    direct_2d = bool(
        pixel.shape == (2,) and np.isfinite(pixel).all()
        and str(sample.get("association_semantics", "")).startswith("DIRECT_OBSERVED_HAWOR_2D")
        and (
            sample.get("status") == "OBSERVED_VISIBLE_SURFACE"
            or sample.get("reason") == "LOCAL_HAND_ROLE_DEPTH_OR_OBJECT_GATE_FAILED"
        )
    )
    visibility = str(interaction.get("object_visibility", "UNKNOWN"))
    object_observed = visibility.startswith("DIRECT_VISIBLE")
    if (
        identity != HYPOTHESIS_PAIR or frame not in HYPOTHESIS_FRAMES
        or interaction.get("short_gap_inferred") is not False
        or not direct_2d or not object_observed or not object_plane_observed
        or adjacency.get("projected_finger_inside_object_mask") is not True
        or tactile.get("finger_specific_tactile_active") is not True
        or tactile.get("time_aligned_within_40ms") is not True
    ):
        return None
    return {
        "session_id": HYPOTHESIS_SESSION,
        "frame_id": frame,
        "timestamp_s": float(interaction["timestamp_s"]),
        "hand_id": HYPOTHESIS_PAIR[0],
        "finger_id": HYPOTHESIS_PAIR[1],
        "object_id": HYPOTHESIS_PAIR[2],
        "status": "HYPOTHESIS_ONLY",
        "hypothesis_type": "VISUAL_OVERLAP_PLUS_ALIGNED_FINGER_TACTILE",
        "visual_evidence": {
            "direct_observed_hawor_2d_projection": True,
            "finger_visible_surface": (
                "PRESENT_VISIBLE_SURFACE" if sample.get("status") == "OBSERVED_VISIBLE_SURFACE"
                else "ABSENT_OCCLUDED"
            ),
            "direct_visible_object": True,
            "object_plane_observed": True,
            "projected_finger_inside_visible_object_mask": True,
            "object_mask_state": interaction.get("object_mask_state"),
        },
        "tactile_evidence": dict(tactile),
        "metric_contact": False,
        "strict_contact_admitted": False,
        "training_eligible": False,
        "contact_ground_truth": False,
        "force_claim": False,
    }


def no_contact_control_row(
    interaction: Mapping[str, Any], tactile: Mapping[str, Any],
    *, object_plane_observed: bool,
) -> dict[str, Any] | None:
    identity = (
        str(interaction.get("hand_id")), str(interaction.get("finger_id")),
        str(interaction.get("object_id")),
    )
    frame = int(interaction.get("frame_id", -1))
    adjacency = interaction.get("two_d_adjacency")
    adjacency = adjacency if isinstance(adjacency, Mapping) else {}
    sample = interaction.get("finger_associated_visible_surface_point")
    sample = sample if isinstance(sample, Mapping) else {}
    pixel = np.asarray(sample.get("source_pixel_uv", []), np.float64)
    direct_2d = bool(
        pixel.shape == (2,) and np.isfinite(pixel).all()
        and str(sample.get("association_semantics", "")).startswith("DIRECT_OBSERVED_HAWOR_2D")
        and (
            sample.get("status") == "OBSERVED_VISIBLE_SURFACE"
            or sample.get("reason") == "LOCAL_HAND_ROLE_DEPTH_OR_OBJECT_GATE_FAILED"
        )
    )
    if (
        identity != HYPOTHESIS_PAIR or frame in HYPOTHESIS_FRAMES
        or interaction.get("short_gap_inferred") is not False
        or not direct_2d or not object_plane_observed
        or not str(interaction.get("object_visibility", "UNKNOWN")).startswith("DIRECT_VISIBLE")
        or adjacency.get("projected_finger_inside_object_mask") is not False
        or adjacency.get("adjacent_within_20px") is not False
        or tactile.get("time_aligned_within_40ms") is not True
        or tactile.get("finger_specific_tactile_active") is not False
    ):
        return None
    return {
        "session_id": HYPOTHESIS_SESSION,
        "frame_id": frame,
        "timestamp_s": float(interaction["timestamp_s"]),
        "hand_id": HYPOTHESIS_PAIR[0],
        "finger_id": HYPOTHESIS_PAIR[1],
        "object_id": HYPOTHESIS_PAIR[2],
        "status": "NO_CONTACT",
        "control_semantics": (
            "NEGATIVE_DIAGNOSTIC_CONTROL_FROM_NO_2D_ADJACENCY_AND_NO_ALIGNED_"
            "FINGER_TACTILE;NOT_CONTACT_GROUND_TRUTH"
        ),
        "visual_evidence": {
            "direct_observed_hawor_2d_projection": True,
            "finger_visible_surface": (
                "PRESENT_VISIBLE_SURFACE" if sample.get("status") == "OBSERVED_VISIBLE_SURFACE"
                else "ABSENT_OCCLUDED"
            ),
            "direct_visible_object": True,
            "object_plane_observed": True,
            "projected_finger_inside_visible_object_mask": False,
            "adjacent_within_20px": False,
        },
        "tactile_evidence": dict(tactile),
        "metric_contact": False,
        "strict_contact_admitted": False,
        "training_eligible": False,
        "contact_ground_truth": False,
    }


def build_hypotheses(
    session_id: str, interaction_rows: Sequence[Mapping[str, Any]],
    tactile_timeline: Sequence[Mapping[str, Any]],
    object_document: Mapping[str, Any],
) -> dict[str, Any]:
    hypotheses: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    if session_id == HYPOTHESIS_SESSION:
        objects = {
            str(item.get("instance_id")): item
            for item in object_document.get("objects", []) if isinstance(item, Mapping)
        }
        if set(objects) != set(OBJECTS):
            raise RuntimeError("Object6D hypothesis identity set drift")
        for row in interaction_rows:
            frame = int(row["frame_id"])
            object_frames = objects[str(row["object_id"])].get("frames")
            if not isinstance(object_frames, list) or len(object_frames) != len(tactile_timeline):
                raise RuntimeError("Object6D hypothesis frame axis drift")
            object_frame = object_frames[frame]
            plane = object_frame.get("plane_normal", {})
            patch = object_frame.get("finite_visible_patch", {})
            object_plane_observed = bool(
                isinstance(plane, Mapping) and isinstance(patch, Mapping)
                and str(plane.get("observability", "")).startswith("OBSERVABLE")
                and str(patch.get("observability", "")).startswith("OBSERVABLE")
            )
            tactile = tactile_timeline[frame]["fingers"][
                f"{row['hand_id']}:{row['finger_id']}"
            ]
            candidate = weak_hypothesis_row(
                row, tactile, object_plane_observed=object_plane_observed,
            )
            if candidate is not None:
                hypotheses.append(candidate)
            control = no_contact_control_row(
                row, tactile, object_plane_observed=object_plane_observed,
            )
            if control is not None and len(controls) < 7:
                controls.append(control)
    return {
        "hypotheses": hypotheses,
        "no_contact_controls": controls,
        "verified_hypothesis_count": len(hypotheses),
        "no_contact_control_count": len(controls),
        "hypothesis_first_blocker": (
            None if hypotheses else "NO_VERIFIED_119_RIGHT_INDEX_CARD02_OVERLAP_TACTILE_ROW"
        ),
        "control_first_blocker": None if controls else "NO_MATCHED_NO_CONTACT_CONTROL_ROW",
    }


def interaction_rows(document: Mapping[str, Any], expected_frames: int) -> list[dict[str, Any]]:
    rows = document.get("rows")
    expected = expected_frames * len(HANDS) * len(FINGERS) * len(OBJECTS)
    if (
        document.get("frame_count") != expected_frames
        or document.get("pair_denominator") != expected
        or document.get("strict_metric_contact_authorized") is not False
        or not isinstance(rows, list) or len(rows) != expected
    ):
        raise RuntimeError("Interaction fixed-axis contract drift")
    keys: set[tuple[int, str, str, str]] = set()
    clean_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise RuntimeError("Interaction row must be an object")
        key = (
            int(row["frame_id"]), str(row["hand_id"]), str(row["finger_id"]),
            str(row["object_id"]),
        )
        if key in keys or row.get("short_gap_inferred") is not False:
            raise RuntimeError("Interaction fixed-axis duplicate or inferred-gap row")
        keys.add(key)
        clean_rows.append(dict(row))
    expected_keys = {
        (frame, hand, finger, object_id)
        for frame in range(expected_frames) for hand in HANDS
        for finger in FINGERS for object_id in OBJECTS
    }
    if keys != expected_keys:
        raise RuntimeError("Interaction fixed axes are incomplete")
    return clean_rows


def decode_video(path: Path, expected_frames: int) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    count = width = height = 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        height, width = image.shape[:2]
        count += 1
    capture.release()
    return {
        "frame_count": count, "expected_frame_count": expected_frames,
        "full_decode": count == expected_frames, "width": width, "height": height,
    }


def physical_left(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    if width % 2 or width < 2:
        raise RuntimeError(f"invalid SBS geometry: {(width, height)}")
    # W0 inventory binds physical-left sourceIndex=1: the second SBS eye.
    eye = frame[:, width // 2:]
    return cv2.resize(eye, (1280, 960), interpolation=cv2.INTER_AREA)


def render_review(
    source_stereo: Path, destination: Path, frame_count: int,
    *, session_id: str, status: str, evidence_rows: Sequence[Mapping[str, Any]],
    hypothesis_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    evidence_by_frame: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in evidence_rows:
        evidence_by_frame[int(row["frame_id"])].append(row)
    hypothesis_frames = {int(row["frame_id"]) for row in hypothesis_rows}
    capture = cv2.VideoCapture(str(source_stereo))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 960),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("failed to open Contact review writer")
    index = 0
    while True:
        ok, sbs = capture.read()
        if not ok:
            break
        image = physical_left(sbs)
        rows = evidence_by_frame.get(index, [])
        near = sum(row.get("geometric_proximity_within_fixed_5mm") is True for row in rows)
        tactile = sum(
            row.get("tactile", {}).get("finger_specific_tactile_active") is True for row in rows
        )
        weak = index in hypothesis_frames
        cv2.rectangle(image, (8, 8), (1010, 100), (0, 0, 0), -1)
        cv2.putText(
            image, f"{session_id} frame {index:04d} | {status}", (18, 34),
            cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 1, cv2.LINE_AA,
        )
        cv2.putText(
            image, f"fixed-5mm near={near} tactile-pair rows={tactile} hypothesis-only={weak}",
            (18, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
            (0, 230, 255) if weak else (220, 220, 220), 1, cv2.LINE_AA,
        )
        cv2.putText(
            image, "strict Contact=0 | R1-E=0 | external metric authority=FALSE",
            (18, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (80, 120, 255), 1, cv2.LINE_AA,
        )
        writer.write(image)
        index += 1
    capture.release()
    writer.release()
    decoded = decode_video(destination, frame_count)
    if not decoded["full_decode"]:
        raise RuntimeError(f"Contact review failed full decode: {destination}")
    return {**decoded, "fps": fps, "input_operation": "SOURCEINDEX1_CROP_THEN_RESIZE_ONLY"}


def run_session(
    inventory_row: Mapping[str, Any], interaction_row: Mapping[str, Any],
    output: Path, visual: Path, depth_rows: Mapping[str, Mapping[str, Any]],
    object_rows: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    session_id = str(inventory_row["session_id"])
    task = str(inventory_row["task"])
    frame_count = int(inventory_row["frame_count"])
    final = output / "sessions" / task / session_id
    if final.exists() or final.is_symlink():
        raise RuntimeError(f"fresh session output required: {final}")
    stage = output / f".session-staging-{session_id}-{uuid.uuid4().hex}"
    stage.mkdir(parents=True)
    evidence: list[dict[str, Any]] = []
    hypotheses: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    interaction_document: dict[str, Any] | None = None

    if task == "potato_chips":
        status, blocker = "BLOCKED_UPSTREAM_OBJECT6D", "BLOCKED_UPSTREAM_OBJECT6D"
    elif session_id == "play_cards_0915_031":
        status, blocker = "BLOCKED_UPSTREAM_HAND_ROLE", "BLOCKED_UPSTREAM_HAND_ROLE"
    elif session_id != HYPOTHESIS_SESSION:
        raise RuntimeError(f"unexpected Poker W0 identity: {session_id}")
    else:
        if interaction_row.get("interaction_consumer_allowed") is not True:
            raise RuntimeError("119 Interaction evidence is not consumer-admitted")
        interaction_path = verify_ref(interaction_row["interaction_evidence"])
        interaction_document = load_json(interaction_path)
        rows = interaction_rows(interaction_document, frame_count)
        tactile_timeline, tactile_binding = load_tactile_timeline(inventory_row, rows)
        depth_path = verify_ref(depth_rows[session_id]["result"])
        object_path = verify_ref(object_rows[session_id]["object6d"])
        object_document = load_json(object_path)
        authority = validate_external_authority(load_json(depth_path), object_document)
        for row in rows:
            frame = int(row["frame_id"])
            tactile = tactile_timeline[frame]["fingers"][
                f"{row['hand_id']}:{row['finger_id']}"
            ]
            evidence.append(evaluate_pair(
                row, tactile, external_metric_authority=authority["external_metric_authority"],
            ))
        hypothesis = build_hypotheses(
            session_id, rows, tactile_timeline, object_document,
        )
        hypotheses = hypothesis["hypotheses"]
        controls = hypothesis["no_contact_controls"]
        status, blocker = (
            "COMPLETED_DIAGNOSTIC_NO_STRICT_CONTACT",
            "EXTERNAL_METRIC_AUTHORITY_UNVERIFIED",
        )
        interaction_document = {
            "reference": ref(interaction_path),
            "authority": authority,
            "tactile_binding": tactile_binding,
            "hypothesis_first_blocker": hypothesis["hypothesis_first_blocker"],
            "control_first_blocker": hypothesis["control_first_blocker"],
        }

    # Every W0 terminal gets a full-denominator tactile integrity read, including
    # sessions whose geometric branch is upstream-blocked.
    timeline, tactile_binding = load_tactile_timeline(
        inventory_row, None if interaction_document is None else (
            rows if session_id == HYPOTHESIS_SESSION else None
        ),
    )
    active_finger_frames = sum(
        value["finger_specific_tactile_active"] is True
        for frame in timeline for value in frame["fingers"].values()
    )
    evidence_document = {
        "schema_version": "0915-robot15h-contact-evidence-session-v1",
        "task_id": TASK_ID,
        "session_id": session_id,
        "source_group": inventory_row["source_group"],
        "frame_count": frame_count,
        "status": status,
        "first_blocker": blocker,
        "authority": AUTHORITY,
        "interaction": interaction_document,
        "tactile_binding": tactile_binding,
        "rows": evidence,
        "summary": {
            "diagnostic_pair_rows": len(evidence),
            "geometry_near_fixed_5mm_rows": sum(
                row["geometric_proximity_within_fixed_5mm"] for row in evidence
            ),
            "finger_specific_tactile_active_frame_pairs": active_finger_frames,
            "strict_contact_rows": 0,
            "r1_e_windows": 0,
        },
        "fixed_geometry_distance_m": GEOMETRY_NEAR_M,
        "uncertainty_never_expands_distance_gate": True,
        "strict_metric_contact_authorized": False,
        "r1_e_authorized": False,
        "training_eligible": False,
        "control_ground_truth": False,
        "physical_deployment_authorized": False,
        "claim_limit": CLAIM_LIMIT,
    }
    hypotheses_document = {
        "schema_version": "0915-robot15h-contact-hypothesis-session-v1",
        "task_id": TASK_ID,
        "session_id": session_id,
        "status": "HYPOTHESIS_ONLY" if hypotheses else "NO_VERIFIED_HYPOTHESIS",
        "allowed_pair": {
            "hand_id": HYPOTHESIS_PAIR[0], "finger_id": HYPOTHESIS_PAIR[1],
            "object_id": HYPOTHESIS_PAIR[2], "frames_inclusive": [92, 98],
        } if session_id == HYPOTHESIS_SESSION else None,
        "hypotheses": hypotheses,
        "no_contact_controls": controls,
        "training_eligible": False,
        "contact_ground_truth": False,
        "strict_contact_authorized": False,
    }
    evidence_stage = stage / "CONTACT_EVIDENCE.json"
    hypotheses_stage = stage / "CONTACT_HYPOTHESES.json"
    atomic_json(evidence_stage, evidence_document)
    atomic_json(hypotheses_stage, hypotheses_document)
    review_stage = stage / "CONTACT_REVIEW.mp4"
    review = render_review(
        verify_ref(inventory_row["source_stereo"]), review_stage, frame_count,
        session_id=session_id, status=status, evidence_rows=evidence,
        hypothesis_rows=hypotheses,
    )
    result = {
        "schema_version": "0915-robot15h-contact-session-result-v1",
        "task_id": TASK_ID,
        "session_id": session_id,
        "task": task,
        "source_group": inventory_row["source_group"],
        "frame_count": frame_count,
        "status": status,
        "first_blocker": blocker,
        "contact_diagnostic_attempted": session_id == HYPOTHESIS_SESSION,
        "contact_evidence_emitted": True,
        "strict_contact_windows": 0,
        "r1_e_windows": 0,
        "hypothesis_rows": len(hypotheses),
        "no_contact_control_rows": len(controls),
        "strict_metric_contact_authorized": False,
        "training_eligible": False,
        "control_ground_truth": False,
        "physical_deployment_authorized": False,
        "weights": "ABSENT",
        "gpu_used": False,
        "source_mutated": False,
        "contact_evidence": projected_ref(evidence_stage, final / evidence_stage.name),
        "contact_hypotheses": projected_ref(hypotheses_stage, final / hypotheses_stage.name),
        "review": {**review, "video": projected_ref(review_stage, final / review_stage.name)},
    }
    atomic_json(stage / "RESULT.json", result)
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(stage, final)
    published = load_json(final / "RESULT.json")
    verify_ref(published["contact_evidence"])
    verify_ref(published["contact_hypotheses"])
    verify_ref(published["review"]["video"])
    shallow = visual / f"{session_id}_CONTACT_REVIEW.mp4"
    try:
        os.link(final / "CONTACT_REVIEW.mp4", shallow)
    except OSError:
        shutil.copy2(final / "CONTACT_REVIEW.mp4", shallow)
    shallow_reference = ref(shallow)
    if shallow_reference["sha256"] != published["review"]["video"]["sha256"]:
        raise RuntimeError("shallow Contact review differs from atomic session review")
    ledger_row = {
        key: published[key] for key in (
            "session_id", "task", "source_group", "frame_count", "status", "first_blocker",
            "contact_diagnostic_attempted", "contact_evidence_emitted",
            "strict_contact_windows", "r1_e_windows", "hypothesis_rows",
            "no_contact_control_rows",
        )
    }
    ledger_row.update({
        "result": ref(final / "RESULT.json"),
        "contact_evidence": ref(final / "CONTACT_EVIDENCE.json"),
        "contact_hypotheses": ref(final / "CONTACT_HYPOTHESES.json"),
        "review": {**review, "video": shallow_reference},
    })
    return ledger_row, hypotheses, controls


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {
        "total": len(rows), "diagnostic_completed": 0,
        "blocked_upstream_hand_role": 0, "blocked_upstream_object6d": 0,
        "failed_runtime": 0, "strict_contact_windows": 0, "r1_e_windows": 0,
        "hypothesis_rows": 0, "unrun": 0,
    }
    for row in rows:
        status = str(row["status"])
        counts["diagnostic_completed"] += int(status.startswith("COMPLETED_DIAGNOSTIC"))
        counts["blocked_upstream_hand_role"] += int(status == "BLOCKED_UPSTREAM_HAND_ROLE")
        counts["blocked_upstream_object6d"] += int(status == "BLOCKED_UPSTREAM_OBJECT6D")
        counts["failed_runtime"] += int(status == "FAILED_RUNTIME")
        counts["strict_contact_windows"] += int(row.get("strict_contact_windows", 0))
        counts["r1_e_windows"] += int(row.get("r1_e_windows", 0))
        counts["hypothesis_rows"] += int(row.get("hypothesis_rows", 0))
    terminal = (
        counts["diagnostic_completed"] + counts["blocked_upstream_hand_role"]
        + counts["blocked_upstream_object6d"] + counts["failed_runtime"]
    )
    counts["unrun"] = counts["total"] - terminal
    return counts


def validate_fixed_paths(output: Path, visual: Path, receipt: Path) -> None:
    if output.resolve() != OUTPUT.resolve():
        raise RuntimeError(f"output root must equal {OUTPUT}")
    if visual.resolve() != VISUAL.resolve():
        raise RuntimeError(f"visual root must equal {VISUAL}")
    if receipt.resolve() != RECEIPT.resolve():
        raise RuntimeError(f"receipt must equal {RECEIPT}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--visual-root", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--executor-epoch", required=True, type=int)
    parser.add_argument("--fencing-token", required=True)
    args = parser.parse_args()
    output, visual, receipt = (
        args.output_root.resolve(), args.visual_root.resolve(), args.receipt.resolve(),
    )
    validate_fixed_paths(output, visual, receipt)
    if args.executor_epoch < 1 or len(args.fencing_token) < 16:
        raise RuntimeError("positive epoch and fencing token >=16 characters required")
    for path in (output, visual, receipt):
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"fresh output required: {path}")
    packet, packet_path = validate_route()
    required_inputs = [
        INVENTORY,
        INTERACTION_ROOT / "INTERACTION_EVIDENCE_LEDGER.json",
        INTERACTION_ROOT / "BATCH_RESULT.json",
        DEPTH_ROOT / "BATCH_RESULT.json",
        OBJECT_ROOT / "BATCH_RESULT.json",
    ]
    for path in required_inputs:
        if not path.is_file():
            raise RuntimeError(f"required upstream artifact missing: {path}")
    inventory_rows = w0_rows(load_json(INVENTORY))
    tactile_manifests = [tactile_manifest(row) for row in inventory_rows]
    signature_payload = {
        "schema_version": "0915-robot15h-contact-wave0-run-signature-v1",
        "task_id": TASK_ID,
        "window_run_id": WINDOW_RUN_ID,
        "executor_epoch": args.executor_epoch,
        "weights": "ABSENT",
        "gpu_used": False,
        "task_packet": ref(packet_path),
        "inputs": [ref(path) for path in required_inputs],
        "processed_tactile_manifests": tactile_manifests,
        "code": [ref(Path(__file__))],
        "config": {
            "fixed_geometry_distance_m": GEOMETRY_NEAR_M,
            "uncertainty_expands_distance_gate": False,
            "tactile_max_absolute_offset_ms": TACTILE_MAX_OFFSET_MS,
            "tactile_finger_source": "finger_grid_5x4x8[specific_finger]",
            "strict_external_metric_authority_required": True,
            "hypothesis_session": HYPOTHESIS_SESSION,
            "hypothesis_pair": list(HYPOTHESIS_PAIR),
            "hypothesis_frames_inclusive": [92, 98],
        },
    }
    signature = {**signature_payload, "run_signature_sha256": canonical_sha(signature_payload)}
    output.mkdir(parents=True)
    visual.mkdir(parents=True)
    atomic_json(output / "RUN_SIGNATURE.json", signature)
    fencing_sha = hashlib.sha256(args.fencing_token.encode()).hexdigest()
    atomic_json(output / "CLAIM.json", {
        "schema_version": "0915-robot15h-contact-writer-claim-v1",
        "task_id": TASK_ID,
        "window_run_id": WINDOW_RUN_ID,
        "status": "CLAIMED",
        "weights": "ABSENT",
        "gpu_used": False,
        "pid": os.getpid(),
        "proc_start_ticks": process_start_ticks(os.getpid()),
        "executor_epoch": args.executor_epoch,
        "fencing_token_sha256": fencing_sha,
        "run_signature_sha256": signature["run_signature_sha256"],
        "unique_write_root": str(output),
    })
    validate_writer_claim(
        output / "CLAIM.json", signature_sha=signature["run_signature_sha256"],
        executor_epoch=args.executor_epoch, fencing_sha=fencing_sha,
    )
    heartbeat()
    identities = [str(row["session_id"]) for row in inventory_rows]
    interaction_ledger = load_json(INTERACTION_ROOT / "INTERACTION_EVIDENCE_LEDGER.json")
    interaction_batch = load_json(INTERACTION_ROOT / "BATCH_RESULT.json")
    if (
        interaction_ledger.get("strict_metric_contact_authorized") is not False
        or interaction_batch.get("strict_metric_contact_authorized") is not False
    ):
        raise RuntimeError("Interaction strict Contact authority must remain false")
    interaction_by_session = rows_by_session(interaction_batch, identities)
    depth_by_session = rows_by_session(load_json(DEPTH_ROOT / "BATCH_RESULT.json"), identities)
    object_by_session = rows_by_session(load_json(OBJECT_ROOT / "BATCH_RESULT.json"), identities)
    rows: list[dict[str, Any]] = []
    hypotheses: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    for inventory_row in inventory_rows:
        try:
            row, session_hypotheses, session_controls = run_session(
                inventory_row, interaction_by_session[str(inventory_row["session_id"])],
                output, visual, depth_by_session, object_by_session,
            )
            rows.append(row)
            hypotheses.extend(session_hypotheses)
            controls.extend(session_controls)
        except Exception as error:
            rows.append({
                "session_id": inventory_row["session_id"],
                "task": inventory_row["task"],
                "source_group": inventory_row["source_group"],
                "frame_count": int(inventory_row["frame_count"]),
                "status": "FAILED_RUNTIME",
                "first_blocker": f"{type(error).__name__}:{error}",
                "contact_diagnostic_attempted": True,
                "contact_evidence_emitted": False,
                "strict_contact_windows": 0,
                "r1_e_windows": 0,
                "hypothesis_rows": 0,
                "no_contact_control_rows": 0,
            })
        heartbeat()
    counts = summarize(rows)
    ledger = {
        "schema_version": "0915-robot15h-contact-evidence-ledger-v1",
        "task_id": TASK_ID,
        "window_run_id": WINDOW_RUN_ID,
        "status": "COMPLETED_ALL_TERMINAL",
        "rows": rows,
        "counts": counts,
        "fixed_geometry_distance_m": GEOMETRY_NEAR_M,
        "uncertainty_never_expands_distance_gate": True,
        "strict_metric_contact_authorized": False,
        "r1_e_authorized": False,
        "training_eligible": False,
        "control_ground_truth": False,
        "physical_deployment_authorized": False,
        "source_mutated": False,
    }
    evidence_path = output / "CONTACT_EVIDENCE_LEDGER.json"
    atomic_json(evidence_path, ledger)
    hypothesis_ledger = {
        "schema_version": "0915-robot15h-contact-hypothesis-ledger-v1",
        "task_id": TASK_ID,
        "window_run_id": WINDOW_RUN_ID,
        "status": "HYPOTHESIS_ONLY" if hypotheses else "NO_VERIFIED_HYPOTHESIS",
        "scope": {
            "session_id": HYPOTHESIS_SESSION,
            "hand_id": HYPOTHESIS_PAIR[0], "finger_id": HYPOTHESIS_PAIR[1],
            "object_id": HYPOTHESIS_PAIR[2], "frames_inclusive": [92, 98],
        },
        "hypotheses": hypotheses,
        "no_contact_controls": controls,
        "verified_hypothesis_count": len(hypotheses),
        "no_contact_control_count": len(controls),
        "strict_contact_authorized": False,
        "r1_e_authorized": False,
        "training_eligible": False,
        "contact_ground_truth": False,
        "force_claim": False,
        "claim_limit": CLAIM_LIMIT,
    }
    hypothesis_path = output / "CONTACT_HYPOTHESIS_LEDGER.json"
    atomic_json(hypothesis_path, hypothesis_ledger)
    batch = {
        "schema_version": "0915-robot15h-contact-wave0-batch-v1",
        "task_id": TASK_ID,
        "window_run_id": WINDOW_RUN_ID,
        "status": "COMPLETED_ALL_TERMINAL",
        "sessions": rows,
        "counts": counts,
        "strict_metric_contact_authorized": False,
        "r1_e_authorized": False,
    }
    atomic_json(output / "BATCH_RESULT.json", batch)
    atomic_json(output / "METRICS.json", {
        "schema_version": "0915-robot15h-contact-wave0-metrics-v1",
        "task_id": TASK_ID,
        "counts": counts,
        "strict_contact_windows": 0,
        "r1_e_windows": 0,
        "verified_hypothesis_rows": len(hypotheses),
        "no_contact_control_rows": len(controls),
        "r0_affected": False,
        "r2_independent_path_affected": False,
    })
    atomic_json(visual / "INDEX.json", {
        "schema_version": "0915-robot15h-contact-wave0-visual-index-v1",
        "task_id": TASK_ID,
        "status": "COMPLETE" if counts["failed_runtime"] == 0 else "PARTIAL",
        "videos": [row["review"]["video"] for row in rows if "review" in row],
    })
    top_status = "PASSED" if counts["failed_runtime"] == 0 else "FAILED_RUNTIME_FINAL"
    result = {
        "schema_version": "0915-robot15h-contact-wave0-result-v1",
        "task_id": TASK_ID,
        "window_run_id": WINDOW_RUN_ID,
        "status": top_status,
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "counts": counts,
        "weights": "ABSENT",
        "gpu_used": False,
        "authority": AUTHORITY,
        "strict_metric_contact_authorized": False,
        "r1_e_authorized": False,
        "training_eligible": False,
        "control_ground_truth": False,
        "physical_deployment_authorized": False,
        "contact_evidence_ledger": ref(evidence_path),
        "contact_hypothesis_ledger": ref(hypothesis_path),
        "batch_result": ref(output / "BATCH_RESULT.json"),
        "metrics": ref(output / "METRICS.json"),
        "visual_index": ref(visual / "INDEX.json"),
        "run_signature": ref(output / "RUN_SIGNATURE.json"),
        "writer_claim": ref(output / "CLAIM.json"),
        "source_mutated": False,
        "claim_limit": packet.get("claim_limit", CLAIM_LIMIT),
    }
    validate_writer_claim(
        output / "CLAIM.json", signature_sha=signature["run_signature_sha256"],
        executor_epoch=args.executor_epoch, fencing_sha=fencing_sha,
    )
    atomic_json(output / "RESULT.json", result)
    atomic_json(receipt, {**result, "result": ref(output / "RESULT.json")})
    atomic_json(output / "RUN_RECEIPT.json", {
        "schema_version": "0915-robot15h-contact-wave0-run-receipt-v1",
        "task_id": TASK_ID,
        "window_run_id": WINDOW_RUN_ID,
        "status": top_status,
        "result": ref(output / "RESULT.json"),
        "terminal_receipt": ref(receipt),
        "gpu_used": False,
    })
    print(json.dumps({
        "status": top_status, "counts": counts,
        "result": str(output / "RESULT.json"),
    }))
    return 0 if top_status == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
