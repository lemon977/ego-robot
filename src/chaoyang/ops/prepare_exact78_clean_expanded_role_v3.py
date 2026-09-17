#!/usr/bin/env python3
"""Validate the exact78 Clean queue and prepare expanded-role V3 handoffs.

This is a CPU-only admission/packaging producer.  It consumes the immutable
references already present in ``CLEAN_QUEUE.json`` and produces per-session
object-protected deletion masks plus two-stage Clean specifications.  It does
not run a real-donor producer, ProPainter, or change any existing authority.

The four current Poker sessions have two legitimate RGB decode domains:

* role masks are pinned to the lossless ``preprocess/all_data/*/rgb.png``;
* task-object masks are pinned to OpenCV decoding of the selected MP4.

Both domains are independently verified for every frame.  Their array hashes
are not falsely required to be equal, because codec decode/color paths differ;
frame index, resolution, video identity, and each producer's own hash are
closed instead.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import tempfile
from typing import Any, Mapping

import cv2
import numpy as np


PROJECT = Path(__file__).resolve().parents[3]
DEFAULT_QUEUE = (
    PROJECT
    / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260909_exact78_current_baseline_batch_v1"
    / "depth_object6d_stream_v1/CLEAN_QUEUE.json"
)
PROPAINTER_ROOT = PROJECT / "vendor/ProPainter"
PROPAINTER_RUNNER = PROJECT / "src/chaoyang/ops/run_clean_synthetic_propainter_baseline.py"
PROPAINTER_LAUNCHER = PROJECT / "src/chaoyang/ops/launch_clean_synthetic_propainter_once.py"
BASELINE_ROOT = PROJECT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260908_two_task_e2e_baseline_v1"
BASELINE_POKER_HANDOFF = (
    BASELINE_ROOT
    / "clean_synthetic_propainter_v1/poker_expanded_role_handoff_v3/FRAME_MANIFEST.json"
)
BASELINE_POKER_SPEC = (
    BASELINE_ROOT
    / "clean_synthetic_propainter_v1/specs/poker_expanded_successor_v3_fullsession.json"
)
BASELINE_ACCEPTANCE = BASELINE_ROOT / "CLEAN_EXPANDED_V3_USER_ACCEPTANCE_20260909.json"

METHOD = "EXPANDED_ROLE_V3_REAL_DONOR_THEN_PROPAINTER"
EXPANSION = {
    "left_human_radius_px": 24,
    "right_human_radius_px": 18,
    "tracker_radius_px": 60,
    "temporal_radius_frames": 1,
    "temporal_rule": "current spatial dilation UNION intersection of symmetric neighbor dilations",
}
PROPAINTER_PARAMETERS = {
    "process_width": 960,
    "process_height": 720,
    "mask_dilation": 4,
    "ref_stride": 10,
    "neighbor_length": 10,
    "subvideo_length": 80,
    "raft_iter": 20,
}


class PrepareError(RuntimeError):
    """A queue, upstream, frame-identity, or no-clobber gate failed."""


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(value.dtype.str.encode() + b"\0")
    digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    if not path.is_file() or path.is_symlink():
        raise PrepareError(f"regular non-symlink artifact required: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def future_artifact(current: Path, final: Path) -> dict[str, Any]:
    current = current.resolve(strict=True)
    if not current.is_file() or current.is_symlink():
        raise PrepareError(f"regular non-symlink artifact required: {current}")
    return {
        "path": str(final.resolve()),
        "bytes": current.stat().st_size,
        "sha256": sha256(current),
    }


def same_ref(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return all(left.get(key) == right.get(key) for key in ("path", "bytes", "sha256"))


def checked_ref(value: Mapping[str, Any], label: str) -> Path:
    if not isinstance(value, Mapping):
        raise PrepareError(f"{label}: artifact reference required")
    actual = artifact(Path(str(value.get("path", ""))))
    if not same_ref(actual, value):
        raise PrepareError(f"{label}: path/bytes/SHA mismatch")
    return Path(actual["path"])


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PrepareError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise PrepareError(f"JSON object required: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_binary_mask(reference: Mapping[str, Any], label: str, shape: tuple[int, int]) -> np.ndarray:
    path = checked_ref(reference, label)
    value = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if value is None or value.shape != shape:
        raise PrepareError(f"{label}: mask shape mismatch")
    unique = set(np.unique(value).tolist())
    if not unique.issubset({0, 255}):
        raise PrepareError(f"{label}: mask is not binary: {sorted(unique)[:8]}")
    return value > 0


def dilate(value: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return value.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(value.astype(np.uint8), kernel, iterations=1) > 0


def validate_baseline_reuse() -> dict[str, Any]:
    handoff_ref = artifact(BASELINE_POKER_HANDOFF)
    handoff = load_json(BASELINE_POKER_HANDOFF)
    if handoff.get("expansion") != EXPANSION:
        raise PrepareError("accepted Poker042 expanded-role V3 parameters changed")
    spec_ref = artifact(BASELINE_POKER_SPEC)
    spec = load_json(BASELINE_POKER_SPEC)
    for key, expected in PROPAINTER_PARAMETERS.items():
        if spec.get(key) != expected:
            raise PrepareError(f"accepted Poker042 ProPainter parameter changed: {key}")
    acceptance_ref = artifact(BASELINE_ACCEPTANCE)
    acceptance = load_json(BASELINE_ACCEPTANCE)
    poker = acceptance.get("accepted", {}).get("poker", {})
    if (
        poker.get("session") != "play_cards_0902_042"
        or poker.get("scope") != "FULLSESSION_171_FRAMES"
        or "ACCEPTED" not in str(acceptance.get("status"))
    ):
        raise PrepareError("Poker042 expanded-role V3 user acceptance gate failed")
    return {
        "accepted_poker042_handoff": handoff_ref,
        "accepted_poker042_propainter_spec": spec_ref,
        "accepted_poker042_user_acceptance": acceptance_ref,
        "reused_expansion": EXPANSION,
        "reused_propainter_parameters": PROPAINTER_PARAMETERS,
    }


def validate_session(row: dict[str, Any]) -> dict[str, Any]:
    session = str(row.get("session_id", ""))
    task = str(row.get("task", ""))
    if task != "poker" or not session.startswith("play_cards_"):
        raise PrepareError(f"{session}: current producer admits Poker sessions only")
    if row.get("status") != "QUEUED_AFTER_OBJECT6D_GRADE_B" or row.get("method") != METHOD:
        raise PrepareError(f"{session}: queue status/method is not admitted")

    role_result_path = checked_ref(row.get("role_mask_result", {}), f"{session}.role_mask_result")
    task_result_path = checked_ref(row.get("task_object_result", {}), f"{session}.task_object_result")
    object6d_result_path = checked_ref(row.get("object6d_result", {}), f"{session}.object6d_result")
    role_result = load_json(role_result_path)
    task_result = load_json(task_result_path)
    object6d_result = load_json(object6d_result_path)

    if (
        role_result.get("session") != session
        or role_result.get("task") != task
        or role_result.get("grade") not in {"A", "B"}
        or role_result.get("downstream_authorized") is not True
    ):
        raise PrepareError(f"{session}: role-mask A/B authority gate failed")
    if (
        task_result.get("session") != session
        or task_result.get("task") != task
        or task_result.get("grade") not in {"A", "B"}
        or task_result.get("downstream_authorized") is not True
    ):
        raise PrepareError(f"{session}: task-object A/B authority gate failed")
    if (
        object6d_result.get("session_id") != session
        or object6d_result.get("task") != task
        or object6d_result.get("grade") not in {"A", "B"}
        or object6d_result.get("consumption_authorized") is not True
        or "CLEAN_VISUAL_BASELINE_INPUT" not in object6d_result.get("authorized_scopes", [])
        or object6d_result.get("robot_contact_authorized") is not False
    ):
        raise PrepareError(f"{session}: observed-only Object6D Clean-scope gate failed")
    if not same_ref(object6d_result.get("inputs", {}).get("role_mask_result", {}), row["role_mask_result"]):
        raise PrepareError(f"{session}: Object6D does not pin queued role result")
    if not same_ref(object6d_result.get("inputs", {}).get("task_object_result", {}), row["task_object_result"]):
        raise PrepareError(f"{session}: Object6D does not pin queued task-object result")

    role_manifest_path = checked_ref(
        role_result.get("artifacts", {}).get("frame_manifest", {}), f"{session}.role_manifest"
    )
    task_manifest_path = checked_ref(
        task_result.get("artifacts", {}).get("manifest", {}), f"{session}.task_object_manifest"
    )
    object6d_manifest_path = checked_ref(
        object6d_result.get("artifacts", {}).get("frame_manifest", {}), f"{session}.object6d_manifest"
    )
    checked_ref(object6d_result.get("artifacts", {}).get("trajectory", {}), f"{session}.object6d_trajectory")
    role_manifest = load_json(role_manifest_path)
    task_manifest = load_json(task_manifest_path)
    object6d_manifest = load_json(object6d_manifest_path)
    role_frames = role_manifest.get("frames")
    task_frames = task_manifest.get("frames")
    object6d_frames = object6d_manifest.get("frames")
    frame_count = int(role_result.get("frame_count", -1))
    if (
        frame_count <= 0
        or task_result.get("frame_count") != frame_count
        or object6d_result.get("frame_count") != frame_count
        or not all(isinstance(value, list) and len(value) == frame_count for value in (role_frames, task_frames, object6d_frames))
    ):
        raise PrepareError(f"{session}: cross-stage frame-count gate failed")

    config_path = checked_ref(role_result.get("pins", {}).get("config", {}), f"{session}.role_config")
    config = load_json(config_path)
    raw_root = Path(str(config.get("raw_all_data", ""))).resolve(strict=True)
    raw_video_ref = role_result.get("pins", {}).get("raw_video", {})
    raw_video_path = checked_ref(raw_video_ref, f"{session}.role_raw_video")
    if not same_ref(task_manifest.get("input", {}).get("selected_rgb", {}), raw_video_ref):
        raise PrepareError(f"{session}: role and task-object selected MP4 differ")

    capture = cv2.VideoCapture(str(raw_video_path))
    if not capture.isOpened():
        raise PrepareError(f"{session}: cannot decode selected MP4")
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    video_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    if (width, height) != (1280, 960) or video_count != frame_count or abs(fps - 30.0) > 0.05:
        capture.release()
        raise PrepareError(f"{session}: selected MP4 metadata mismatch")

    required_roles = {"left_human", "right_human", "left_tracker", "right_tracker"}
    observed_object_frames = 0
    try:
        for frame_id, (role_frame, task_frame, object6d_frame) in enumerate(
            zip(role_frames, task_frames, object6d_frames, strict=True)
        ):
            if (
                int(role_frame.get("source_frame", -1)) != frame_id
                or int(task_frame.get("source_frame", -1)) != frame_id
                or int(object6d_frame.get("frame_id", -1)) != frame_id
            ):
                raise PrepareError(f"{session}: frame identity mismatch at {frame_id}")
            raw_path = (raw_root / f"{frame_id:05d}" / "rgb.png").resolve(strict=True)
            if not raw_path.is_file() or raw_path.is_symlink():
                raise PrepareError(f"{session}: bad raw PNG at frame {frame_id}")
            raw = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
            if raw is None or raw.shape[:2] != (height, width):
                raise PrepareError(f"{session}: raw PNG decode/shape failed at frame {frame_id}")
            if array_sha256(raw) != role_frame.get("selected_rgb_decoded_sha256"):
                raise PrepareError(f"{session}: role RGB domain mismatch at frame {frame_id}")
            ok, decoded = capture.read()
            if not ok or decoded.shape[:2] != (height, width):
                raise PrepareError(f"{session}: MP4 decode failed at frame {frame_id}")
            task_rgb_sha = str(task_frame.get("selected_rgb_decoded_sha256", ""))
            if array_sha256(decoded) != task_rgb_sha:
                raise PrepareError(f"{session}: task-object RGB domain mismatch at frame {frame_id}")
            if object6d_frame.get("rgb_frame_sha256") != task_rgb_sha:
                raise PrepareError(f"{session}: Object6D/task-object RGB mismatch at frame {frame_id}")

            roles = role_frame.get("role_masks", {})
            if not isinstance(roles, dict) or set(roles) != required_roles:
                raise PrepareError(f"{session}: required independent roles differ at frame {frame_id}")
            for name, reference in roles.items():
                read_binary_mask(reference, f"{session}.{frame_id}.{name}", (height, width))

            instances = task_frame.get("physical_instances", {})
            if not isinstance(instances, dict) or set(instances) != {"0"}:
                raise PrepareError(f"{session}: Poker physical instance contract differs at frame {frame_id}")
            instance = instances["0"]
            value = read_binary_mask(instance.get("mask", {}), f"{session}.{frame_id}.object0", (height, width))
            area = int(value.sum())
            observed = bool(instance.get("observed"))
            if (
                observed != bool(instance.get("valid"))
                or observed != bool(area)
                or area != int(instance.get("area_px", -1))
                or (not observed and int(instance.get("physical_instance_id", -2)) != -1)
            ):
                raise PrepareError(f"{session}: object mask validity/area gate failed at frame {frame_id}")
            observed_object_frames += int(observed)
        ok, _extra = capture.read()
        if ok:
            raise PrepareError(f"{session}: selected MP4 has extra decodable frames")
    finally:
        capture.release()
    if observed_object_frames != int(task_result.get("observed_counts", {}).get("0", -1)):
        raise PrepareError(f"{session}: observed object denominator mismatch")

    return {
        "session": session,
        "task": task,
        "frame_count": frame_count,
        "fps": fps,
        "source_resolution": [width, height],
        "raw_root": raw_root,
        "raw_video": raw_video_ref,
        "role_result": row["role_mask_result"],
        "task_object_result": row["task_object_result"],
        "object6d_result": row["object6d_result"],
        "role_manifest_path": role_manifest_path,
        "task_manifest_path": task_manifest_path,
        "role_manifest": role_manifest,
        "task_manifest": task_manifest,
        "observed_object_frames": observed_object_frames,
    }


def validate_queue(queue_path: Path) -> dict[str, Any]:
    queue_path = queue_path.resolve(strict=True)
    queue = load_json(queue_path)
    if queue.get("schema_version") != "exact78-clean-stream-queue-v1":
        raise PrepareError("unsupported Clean queue schema")
    rows = queue.get("sessions")
    if not isinstance(rows, list) or not rows:
        raise PrepareError("Clean queue is empty or invalid")
    sessions = [str(row.get("session_id", "")) for row in rows]
    if len(set(sessions)) != len(sessions):
        raise PrepareError("duplicate Clean queue session")
    reuse = validate_baseline_reuse()
    validated = [validate_session(row) for row in rows]
    return {
        "schema_version": "exact78-clean-expanded-role-v3-validation-v1",
        "validated_at": now(),
        "status": "PASS_CPU_QUEUE_VALIDATE_ONLY",
        "queue": artifact(queue_path),
        "method": METHOD,
        "session_count": len(validated),
        "frame_count": sum(item["frame_count"] for item in validated),
        "sessions": validated,
        "baseline_reuse": reuse,
        "hard_gates": {
            "queue_artifact_refs_exact": "PASS",
            "role_and_task_object_grade_ab": "PASS",
            "object6d_grade_ab_clean_scope_only": "PASS",
            "object6d_pins_same_role_and_task_results": "PASS",
            "role_png_rgb_domain_all_frames": "PASS",
            "task_object_mp4_rgb_domain_all_frames": "PASS",
            "four_independent_roles_all_frames": "PASS",
            "task_object_validity_and_area_all_frames": "PASS",
            "accepted_poker042_method_parameters_reused": "PASS",
        },
        "claim_limit": "CPU validation only; no Clean pixels, GPU work, stage authority, Robot, or training are claimed.",
    }


def make_handoff(context: dict[str, Any], temporary_root: Path, final_root: Path) -> dict[str, Any]:
    session = context["session"]
    temp_session = temporary_root / "sessions" / session
    final_session = final_root / "sessions" / session
    removal_root = temp_session / "expanded_role_handoff" / "expanded_clean_removal"
    removal_root.mkdir(parents=True)
    shape = (context["source_resolution"][1], context["source_resolution"][0])
    role_frames = context["role_manifest"]["frames"]
    task_frames = context["task_manifest"]["frames"]

    roles_by_frame: list[dict[str, np.ndarray]] = []
    for frame_id, frame in enumerate(role_frames):
        roles_by_frame.append(
            {
                name: read_binary_mask(reference, f"{session}.{frame_id}.{name}", shape)
                for name, reference in frame["role_masks"].items()
            }
        )

    expanded_by_frame: list[dict[str, np.ndarray]] = []
    for frame_id, current in enumerate(roles_by_frame):
        expanded: dict[str, np.ndarray] = {}
        for role, value in current.items():
            radius = EXPANSION["tracker_radius_px"] if "tracker" in role else EXPANSION[f"{role}_radius_px"]
            current_dilated = dilate(value, radius)
            neighbors = []
            if frame_id > 0:
                neighbors.append(dilate(roles_by_frame[frame_id - 1][role], radius))
            if frame_id + 1 < len(roles_by_frame):
                neighbors.append(dilate(roles_by_frame[frame_id + 1][role], radius))
            stable_neighbor = np.logical_and.reduce(neighbors) if len(neighbors) >= 2 else None
            expanded[role] = current_dilated if stable_neighbor is None else current_dilated | stable_neighbor
        expanded_by_frame.append(expanded)

    rows: list[dict[str, Any]] = []
    totals = {"original": 0, "expanded": 0, "protected": 0, "removal": 0}
    for frame_id, (role_frame, task_frame) in enumerate(zip(role_frames, task_frames, strict=True)):
        original_union = np.logical_or.reduce(list(roles_by_frame[frame_id].values()))
        expanded_union = np.logical_or.reduce(list(expanded_by_frame[frame_id].values()))
        protected = np.zeros(shape, dtype=bool)
        physical_objects: dict[str, Any] = {}
        for instance_id, instance in task_frame["physical_instances"].items():
            value = read_binary_mask(instance["mask"], f"{session}.{frame_id}.object{instance_id}", shape)
            if bool(instance["observed"]):
                protected |= value
            physical_objects[f"physical_object_{instance_id}"] = {
                **instance["mask"],
                "global_physical_instance_id": int(instance_id),
                "observed": bool(instance["observed"]),
                "valid": bool(instance["valid"]),
            }
        removal = expanded_union & ~protected
        current_path = removal_root / f"{frame_id:05d}.png"
        if not cv2.imwrite(str(current_path), removal.astype(np.uint8) * 255):
            raise PrepareError(f"{session}: cannot write removal frame {frame_id}")
        final_path = final_session / "expanded_role_handoff/expanded_clean_removal" / current_path.name
        raw_path = context["raw_root"] / f"{frame_id:05d}" / "rgb.png"
        totals["original"] += int(original_union.sum())
        totals["expanded"] += int(expanded_union.sum())
        totals["protected"] += int(protected.sum())
        totals["removal"] += int(removal.sum())
        rows.append(
            {
                "source_frame": frame_id,
                "source_rgb": artifact(raw_path),
                "role_rgb_decoded_sha256": role_frame["selected_rgb_decoded_sha256"],
                "task_object_rgb_decoded_sha256": task_frame["selected_rgb_decoded_sha256"],
                "rgb_decode_domain_contract": "ROLE_PNG_AND_TASK_OBJECT_MP4_INDEPENDENTLY_VERIFIED_SAME_FRAME_AND_RESOLUTION",
                "role_masks": role_frame["role_masks"],
                **physical_objects,
                "clean_removal_object_protected": future_artifact(current_path, final_path),
                "published_removal_object_overlap_pixels": int((removal & protected).sum()),
                "original_role_union_pixels": int(original_union.sum()),
                "expanded_role_union_pixels": int(expanded_union.sum()),
                "added_edge_shadow_pixels": int((expanded_union & ~original_union).sum()),
                "protected_object_pixels": int(protected.sum()),
            }
        )

    manifest = {
        "schema_version": "exact78-expanded-role-clean-frame-manifest-v3",
        "created_at": now(),
        "task": context["task"],
        "session": session,
        "frame_count": context["frame_count"],
        "source_role_manifest": artifact(context["role_manifest_path"]),
        "source_task_object_manifest": artifact(context["task_manifest_path"]),
        "object6d_admission_reference": context["object6d_result"],
        "expansion": EXPANSION,
        "semantic_contract": "expanded independent hands/trackers MINUS observed task-object pixels",
        "frames": rows,
        "claim_limit": "Deletion support only; Object6D is an admission pin and is not used to invent hidden object pixels.",
    }
    temp_manifest = temp_session / "expanded_role_handoff/FRAME_MANIFEST.json"
    final_manifest = final_session / "expanded_role_handoff/FRAME_MANIFEST.json"
    write_json(temp_manifest, manifest)
    result = {
        "schema_version": "exact78-expanded-role-clean-handoff-result-v3",
        "created_at": now(),
        "status": "PASS_CPU_HANDOFF_READY_WAIT_REAL_DONOR",
        "task": context["task"],
        "session": session,
        "frame_count": context["frame_count"],
        "derived_frame_manifest": future_artifact(temp_manifest, final_manifest),
        "upstream": {
            "role_mask_result": context["role_result"],
            "task_object_result": context["task_object_result"],
            "object6d_result": context["object6d_result"],
        },
        "metrics": {
            "original_role_union_pixels": totals["original"],
            "expanded_role_union_pixels": totals["expanded"],
            "added_edge_shadow_pixels": totals["expanded"] - totals["original"],
            "protected_object_pixels": totals["protected"],
            "published_removal_pixels": totals["removal"],
            "observed_object_frames": context["observed_object_frames"],
        },
        "hard_gates": {
            "frame_identity": "PASS",
            "task_object_excluded_from_deletion": "PASS",
            "temporal_rule_fixed": "PASS",
            "fresh_no_clobber": "PASS",
            "gpu_not_started": "PASS",
        },
        "claim_limit": "CPU handoff only; no real-donor result, ProPainter result, Clean grade, authority, Robot, or training claim.",
    }
    temp_result = temp_session / "expanded_role_handoff/RESULT.json"
    final_result = final_session / "expanded_role_handoff/RESULT.json"
    write_json(temp_result, result)
    return {
        "manifest": future_artifact(temp_manifest, final_manifest),
        "result": future_artifact(temp_result, final_result),
        "metrics": result["metrics"],
    }


def make_specs(
    context: dict[str, Any], handoff: dict[str, Any], temporary_root: Path, final_root: Path
) -> dict[str, Any]:
    session = context["session"]
    spec_temp_root = temporary_root / "specs"
    spec_final_root = final_root / "specs"
    expected_donor_manifest = final_root / "real_donor_v1" / session / "SOURCE_MAP_MANIFEST.json"
    donor_output = final_root / "real_donor_v1" / session
    propainter_output = final_root / "propainter_v1" / session
    donor_spec = {
        "schema_version": "exact78-real-donor-clean-stage-input-v1",
        "created_at": now(),
        "status": "INPUTS_PINNED_GENERIC_RUNNER_NOT_FROZEN",
        "task": context["task"],
        "session": session,
        "frame_count": context["frame_count"],
        "source_resolution": context["source_resolution"],
        "fps": context["fps"],
        "inputs": {
            "raw_video": context["raw_video"],
            "expanded_role_mask_manifest": handoff["manifest"],
            "role_mask_result": context["role_result"],
            "task_object_result": context["task_object_result"],
            "object6d_result": context["object6d_result"],
        },
        "method_contract": {
            "donor_scope": "SAME_SESSION_SELECTED_RGB_ONLY",
            "donor_rgb": "EXACT_DECODED_SOURCE_PIXEL_NEAREST_INTEGER_NO_BLEND",
            "object_policy": "BYTE_EXACT_PROTECT_OBSERVED_TASK_OBJECT_MASK",
            "unsupported_policy": "KEEP_RAW_BYTE_EXACT",
            "generative_fill": False,
            "per_pixel_provenance": True,
            "reuse_basis": "USER_ACCEPTED_POKER042_EXPANDED_ROLE_V3_REAL_DONOR_STAGE",
        },
        "output_root": str(donor_output),
        "expected_output": str(expected_donor_manifest),
        "execution_gate": "WAIT_FRESH_GENERIC_REAL_DONOR_RUNNER_AND_AUTHORITY",
        "claim_limit": "Input spec only. No real-donor execution or result is claimed.",
    }
    donor_temp = spec_temp_root / f"{session}_real_donor_input.json"
    donor_final = spec_final_root / donor_temp.name
    write_json(donor_temp, donor_spec)

    propainter_spec = {
        "schema_version": "clean-synthetic-propainter-spec-v1",
        "task": context["task"],
        "session": session,
        "mask_frame_manifest": handoff["manifest"]["path"],
        "real_donor_source_manifest": str(expected_donor_manifest),
        "propainter_root": str(PROPAINTER_ROOT.resolve()),
        "output_root": str(propainter_output),
        "start_frame": 0,
        "stop_frame_exclusive": context["frame_count"],
        "fps": context["fps"],
        **PROPAINTER_PARAMETERS,
        "publication_semantics": "SYNTHETIC_CLEAN_EXPANDED_ROLE_V3_OBJECT_PROTECTED",
        "upstream_admission_pins": {
            "role_mask_result": context["role_result"],
            "task_object_result": context["task_object_result"],
            "object6d_result": context["object6d_result"],
            "real_donor_stage_spec": future_artifact(donor_temp, donor_final),
        },
        "execution_gate": "WAIT_REAL_DONOR_SOURCE_MAP_MANIFEST_AND_INDEPENDENT_AUTHORITY",
        "claim_limit": "Generated pixels must remain SYNTHETIC_PROPAINTER and are not physical background truth.",
    }
    propainter_temp = spec_temp_root / f"{session}_propainter.json"
    propainter_final = spec_final_root / propainter_temp.name
    write_json(propainter_temp, propainter_spec)
    return {
        "real_donor_stage_spec": future_artifact(donor_temp, donor_final),
        "propainter_spec": future_artifact(propainter_temp, propainter_final),
        "expected_real_donor_manifest": str(expected_donor_manifest),
        "real_donor_output_root": str(donor_output),
        "propainter_output_root": str(propainter_output),
    }


def make_launch_plan(session_specs: list[dict[str, Any]], temporary_root: Path, final_root: Path) -> dict[str, Any]:
    jobs = []
    for item in session_specs:
        session = item["session"]
        receipt = final_root / "gpu_receipts" / f"{session}_propainter_launch.json"
        holder = f"exact78-clean-v3-{session}"
        command = " ".join(
            shlex.quote(value)
            for value in (
                "python3",
                str(PROPAINTER_LAUNCHER.resolve()),
                "--spec",
                item["specs"]["propainter_spec"]["path"],
                "--holder",
                holder,
                "--receipt",
                str(receipt),
                "--wall-seconds",
                "3600",
            )
        )
        jobs.append(
            {
                "task": "poker",
                "session": session,
                "status": "NOT_RUNNABLE_WAIT_REAL_DONOR_AND_AUTHORITY",
                "required_real_donor_manifest": item["specs"]["expected_real_donor_manifest"],
                "required_absent_output_root": item["specs"]["propainter_output_root"],
                "gpu_lease_holder": holder,
                "safe_gpu_launch_command_after_all_preconditions_pass": command,
                "preconditions": [
                    "fresh generic real-donor producer is independently frozen and authorized",
                    "same-session SOURCE_MAP_MANIFEST exists and passes provenance/identity validation",
                    "central GPU lease is RELEASED before launcher acquisition",
                    "ProPainter output root and launch receipt are absent",
                    "launch jobs execute serially under the single shared GPU lease",
                ],
            }
        )
    payload = {
        "schema_version": "exact78-clean-expanded-role-v3-gpu-launch-plan-v1",
        "created_at": now(),
        "status": "CPU_PREPARED_GPU_NOT_STARTED",
        "launcher": artifact(PROPAINTER_LAUNCHER),
        "runner": artifact(PROPAINTER_RUNNER),
        "jobs": jobs,
        "claim_limit": "Commands are conditional handoff only. This receipt does not authorize or claim GPU execution.",
    }
    temp_path = temporary_root / "GPU_LAUNCH_PLAN.json"
    final_path = final_root / "GPU_LAUNCH_PLAN.json"
    write_json(temp_path, payload)
    return future_artifact(temp_path, final_path)


def prepare(validation: dict[str, Any], output_root: Path) -> dict[str, Any]:
    output_root = output_root.resolve()
    output_root.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_root.parent / ".exact78_clean_expanded_role_v3_prepare.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if output_root.exists() or output_root.is_symlink():
            raise PrepareError(f"no-clobber output already exists: {output_root}")
        temporary_root = Path(
            tempfile.mkdtemp(prefix=f".{output_root.name}.partial.", dir=output_root.parent)
        )
        try:
            session_rows = []
            for context in validation["sessions"]:
                handoff = make_handoff(context, temporary_root, output_root)
                specs = make_specs(context, handoff, temporary_root, output_root)
                session_rows.append(
                    {
                        "task": context["task"],
                        "session": context["session"],
                        "frame_count": context["frame_count"],
                        "status": "PASS_CPU_HANDOFF_READY_WAIT_REAL_DONOR",
                        "handoff": handoff,
                        "specs": specs,
                    }
                )
            launch_plan = make_launch_plan(session_rows, temporary_root, output_root)
            public_validation = {key: value for key, value in validation.items() if key != "sessions"}
            public_validation["sessions"] = [
                {
                    "task": item["task"],
                    "session": item["session"],
                    "frame_count": item["frame_count"],
                    "observed_object_frames": item["observed_object_frames"],
                }
                for item in validation["sessions"]
            ]
            result = {
                "schema_version": "exact78-clean-expanded-role-v3-prepare-result-v1",
                "created_at": now(),
                "status": "PASS_CPU_HANDOFFS_PREPARED_GPU_NOT_STARTED",
                "output_root": str(output_root),
                "validation": public_validation,
                "sessions": session_rows,
                "gpu_launch_plan": launch_plan,
                "hard_gates": {
                    "fresh_no_clobber_publish": "PASS",
                    "four_upstream_closures": "PASS",
                    "expanded_role_v3_handoffs": "PASS",
                    "task_object_subtracted_all_frames": "PASS",
                    "object6d_admission_pinned": "PASS",
                    "real_donor_not_fabricated": "PASS",
                    "gpu_not_started": "PASS",
                    "old_terminal_and_authority_untouched": "PASS",
                },
                "next_gate": "Freeze/authorize a generic same-session real-donor producer; then validate each SOURCE_MAP_MANIFEST before any ProPainter launch.",
                "claim_limit": "CPU preparation only; no Clean grade/authority, GPU, Robot, training, or deployment claim.",
            }
            write_json(temporary_root / "PREPARE_RESULT.json", result)
            if output_root.exists() or output_root.is_symlink():
                raise PrepareError(f"no-clobber race: output appeared: {output_root}")
            os.rename(temporary_root, output_root)
            return load_json(output_root / "PREPARE_RESULT.json")
        except Exception:
            # Keep an incomplete fresh directory as evidence; never overwrite or
            # silently reuse it on a retry.
            if temporary_root.exists():
                failed = output_root.parent / f"{output_root.name}_FAILED_PARTIAL_{os.getpid()}"
                if not failed.exists():
                    os.rename(temporary_root, failed)
            raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    if args.prepare_only and args.output_root is None:
        parser.error("--prepare-only requires --output-root")
    if args.validate_only and args.output_root is not None:
        parser.error("--validate-only does not accept --output-root")
    validation = validate_queue(args.queue)
    if args.validate_only:
        printable = {key: value for key, value in validation.items() if key != "sessions"}
        printable["sessions"] = [
            {
                "task": item["task"],
                "session": item["session"],
                "frame_count": item["frame_count"],
                "observed_object_frames": item["observed_object_frames"],
            }
            for item in validation["sessions"]
        ]
        print(json.dumps(printable, ensure_ascii=False, indent=2))
        return 0
    result = prepare(validation, args.output_root)
    print(
        json.dumps(
            {
                "status": result["status"],
                "output_root": result["output_root"],
                "sessions": [item["session"] for item in result["sessions"]],
                "gpu_started": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
