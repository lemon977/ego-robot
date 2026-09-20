#!/usr/bin/env python3
"""CPU-only Poker044 Depth→visible-object review canary.

This operation intentionally does *not* publish a consumable Object6D result.
It binds the already-produced FoundationStereo cache to the already-produced
SAM3.1 visible-mask proxy for ``play_cards_0915_044`` and emits only directly
visible finite surface evidence.  Physical-card identity and card-face identity
remain UNKNOWN.  The output is review-only and must not feed Contact, Robot,
Clean, training, control, or deployment.

The operation performs no model inference and never writes its inputs.  The two
W0 Poker regressions (031/119) are checked by immutable SHA pins only.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence
import uuid

import cv2
import numpy as np

from chaoyang.ops.run_0915_robot15h_object6d_wave0_v1 import (
    DEPTH_REFERENCE,
    DEPTH_SHAPE,
    IMAGE_DOMAIN,
    INSTANCE_IDS,
    MASK_SHAPE,
    PIXEL_MAP,
    PackedMask,
    depth_to_mask_map,
    finite_patch_record,
    load_depth_frame,
    registered_support,
    temporally_unify_axes,
)
from chaoyang.pipeline.object6d_planar_observability_v2 import (
    DIRECT_VISIBILITY,
    MASK_REGISTRATION_AUTHORITY,
    PlanarFrameInput,
    estimate_planar_frame,
)


SCHEMA_VERSION = "0915-robot-recovery-v21-b2-poker044-review-canary-v1"
CANDIDATE_ID = "B2_DEPTH_TO_VISIBLE_OBJECT_GEOMETRY_SUCCESSOR_V1"
SESSION_ID = "play_cards_0915_044"
TASK = "playing_cards"
FRAME_COUNT = 166
PHYSICAL_IDENTITY = "UNKNOWN_UNBOUND"
FACE_IDENTITY = "UNKNOWN_UNBOUND"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


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


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    with temporary.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def make_ref(path: Path) -> dict[str, Any]:
    candidate = path.resolve(strict=True)
    return {
        "path": str(candidate),
        "bytes": candidate.stat().st_size,
        "sha256": sha256(candidate),
    }


def projected_ref(existing: Path, destination: Path) -> dict[str, Any]:
    return {
        "path": str(destination.resolve()),
        "bytes": existing.stat().st_size,
        "sha256": sha256(existing),
    }


def verify_ref(value: Mapping[str, Any], *, label: str) -> Path:
    required = {"path", "bytes", "sha256"}
    if set(value) != required:
        raise RuntimeError(f"{label}: exact path/bytes/sha256 reference required")
    path = Path(str(value["path"])).resolve(strict=True)
    if path.stat().st_size != int(value["bytes"]):
        raise RuntimeError(f"{label}: byte-size drift: {path}")
    if sha256(path) != str(value["sha256"]):
        raise RuntimeError(f"{label}: SHA256 drift: {path}")
    return path


def _require_false(config: Mapping[str, Any], names: Sequence[str]) -> None:
    for name in names:
        if config.get(name) is not False:
            raise RuntimeError(f"fail-closed flag must be false: {name}")


def validate_config(config: Mapping[str, Any]) -> None:
    if (
        config.get("schema_version") != SCHEMA_VERSION
        or config.get("candidate_id") != CANDIDATE_ID
        or config.get("session_id") != SESSION_ID
        or config.get("task") != TASK
        or config.get("frame_count") != FRAME_COUNT
        or config.get("review_only") is not True
        or config.get("consumer_allowed") is not False
        or config.get("physical_card_identity") != PHYSICAL_IDENTITY
        or config.get("face_identity") != FACE_IDENTITY
        or config.get("depth_execution") != "REUSE_PINNED_CACHE_NO_RERUN"
        or config.get("mask_scope") != "VISIBLE_CANDIDATE_PROXY_NOT_PHYSICAL_IDENTITY"
    ):
        raise RuntimeError("bounded Poker044 review configuration drift")
    _require_false(
        config,
        (
            "gpu_allowed",
            "foundationstereo_rerun_allowed",
            "hidden_geometry_inferred",
            "external_metric_accuracy_claimed",
            "contact_authority",
            "control_ground_truth",
            "training_eligible",
            "physical_deployment_authorized",
        ),
    )
    if set(config.get("regressions", {})) != {
        "play_cards_0915_031",
        "play_cards_0915_119",
    }:
        raise RuntimeError("exact immutable Poker031/119 regression pins required")


def verify_regressions(config: Mapping[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for session_id in ("play_cards_0915_031", "play_cards_0915_119"):
        pins = config["regressions"][session_id]
        result_path = verify_ref(pins["result"], label=f"{session_id}.result")
        object_path = verify_ref(pins["object6d"], label=f"{session_id}.object6d")
        support_path = verify_ref(pins["support"], label=f"{session_id}.support")
        result = load_json(result_path)
        document = load_json(object_path)
        if (
            result.get("session_id") != session_id
            or result.get("status") != "PASSED_DEVELOPMENT_VISIBLE_SURFACE"
            or result.get("object6d", {}).get("sha256") != sha256(object_path)
            or result.get("visible_patch_support", {}).get("sha256") != sha256(support_path)
            or document.get("session_id") != session_id
            or document.get("hidden_geometry_inferred") is not False
            or document.get("contact_authority") != "NONE"
            or document.get("robot_authority") != "NONE"
        ):
            raise RuntimeError(f"immutable regression semantic drift: {session_id}")
        rows.append(
            {
                "session_id": session_id,
                "status": "SHA_AND_SEMANTIC_PIN_VERIFIED_NOT_MUTATED",
                "result": make_ref(result_path),
                "object6d": make_ref(object_path),
                "support": make_ref(support_path),
            }
        )
    return {
        "status": "PASSED_IMMUTABLE_REGRESSION_PIN_CHECK",
        "regressions_mutated": False,
        "sessions": rows,
    }


def validate_depth(config: Mapping[str, Any]) -> dict[str, Any]:
    refs = config["depth"]
    paths = {
        name: verify_ref(refs[name], label=f"depth.{name}")
        for name in (
            "result",
            "adapter_contract",
            "depth_contract",
            "rgb_alignment_qa",
            "depth_summary",
        )
    }
    result = load_json(paths["result"])
    adapter = load_json(paths["adapter_contract"])
    contract = load_json(paths["depth_contract"])
    alignment = load_json(paths["rgb_alignment_qa"])
    summary = load_json(paths["depth_summary"])
    if (
        result.get("session_id") != SESSION_ID
        or result.get("frame_count") != FRAME_COUNT
        or result.get("status") != "PASSED"
        or result.get("consumption_authorized") is not True
        or result.get("authorized_scopes") != ["VISUAL_OBJECT6D_CANDIDATE_INPUT"]
        or result.get("external_accuracy") != "UNVERIFIED"
        or result.get("strict_metric_contact_authorized") is not False
        or adapter.get("output_domain") != "ORIGINAL_PHYSICAL_LEFT_640x480"
        or adapter.get("physical_source_indices") != {"left": 1, "right": 0}
        or adapter.get("lens_undistortion_applied") is not False
        or adapter.get("lens_remap_applied") is not False
        or adapter.get("camera_swap") is not False
        or adapter.get("output_spatial_unflip") is not True
        or contract.get("depth_reference") != DEPTH_REFERENCE
        or contract.get("frame_geometry") != [640, 480]
        or contract.get("frame_count") != FRAME_COUNT
        or contract.get("consumption_authorized") is not True
        or contract.get("occluded_or_hidden_geometry") != "INVALID_NOT_COMPLETED"
        or alignment.get("depth_grid_domain") != "ORIGINAL_PHYSICAL_LEFT_640x480"
        or alignment.get("mismatched_pixels") != 0
        or alignment.get("maximum_absolute_channel_error") != 0
        or summary.get("frame_count") != FRAME_COUNT
        or summary.get("consumption_authorized") is not True
    ):
        raise RuntimeError("pinned Poker044 Depth contract drift")
    frames = summary.get("frames")
    if (
        not isinstance(frames, list)
        or len(frames) != FRAME_COUNT
        or [row.get("frame") for row in frames] != list(range(FRAME_COUNT))
    ):
        raise RuntimeError("Poker044 Depth frame axis is incomplete")
    return {"frames": frames, "references": {k: make_ref(v) for k, v in paths.items()}}


def validate_masks(config: Mapping[str, Any]) -> dict[str, Any]:
    refs = config["visible_mask_candidate"]
    result_path = verify_ref(refs["result"], label="mask.result")
    manifest_path = verify_ref(refs["instance_manifest"], label="mask.instance_manifest")
    admission_path = verify_ref(refs["identity_admission"], label="mask.identity_admission")
    video_path = verify_ref(refs["input_video"], label="mask.input_video")
    result = load_json(result_path)
    manifest = load_json(manifest_path)
    admission = load_json(admission_path)
    if (
        result.get("session_id") != SESSION_ID
        or result.get("frame_count") != FRAME_COUNT
        or result.get("status") != "PASSED_VISIBLE_OBJECT_MASK_PROXY"
        or result.get("object_mask_consumer_allowed") is not True
        or manifest.get("session_id") != SESSION_ID
        or manifest.get("frame_count") != FRAME_COUNT
        or manifest.get("image_domain") != IMAGE_DOMAIN
        or manifest.get("manual_coordinates_or_boxes") is not False
        or manifest.get("union_mask_created") is not False
        or admission.get("session_id") != SESSION_ID
        or admission.get("consumer_allowed_instances") != 0
        or admission.get("unknown_instances") != 3
        or admission.get("physical_identity_ground_truth") is not False
        or admission.get("mask_accuracy_claimed") is not False
        or set(admission.get("instance_status", {}).values())
        != {"UNKNOWN_PHYSICAL_CARD_OR_FACE_IDENTITY"}
    ):
        raise RuntimeError("Poker044 visible-mask/identity admission contract drift")
    by_id = {str(row.get("instance_id")): row for row in manifest.get("instances", [])}
    if set(by_id) != set(INSTANCE_IDS):
        raise RuntimeError("three frozen candidate track IDs required")
    masks: dict[str, PackedMask] = {}
    states: dict[str, list[dict[str, Any]]] = {}
    references: dict[str, Any] = {
        "result": make_ref(result_path),
        "instance_manifest": make_ref(manifest_path),
        "identity_admission": make_ref(admission_path),
        "input_video": make_ref(video_path),
    }
    for instance_id in INSTANCE_IDS:
        row = by_id[instance_id]
        if (
            row.get("object_mask_consumer_allowed") is not True
            or row.get("visible_surface_only") is not True
            or row.get("hidden_shape_or_extent_inferred") is not False
            or row.get("raw_track_id_scope") != "SESSION_PROMPT_STATE_ONLY"
        ):
            raise RuntimeError(f"visible candidate proxy drift: {instance_id}")
        mask_path = verify_ref(row["semantic_archive"], label=f"{instance_id}.semantic")
        state_path = verify_ref(row["state_ledger"], label=f"{instance_id}.state")
        mask = PackedMask(mask_path)
        ledger = load_json(state_path)
        frame_rows = ledger.get("frames")
        if (
            mask.frame_count != FRAME_COUNT
            or (mask.height, mask.width) != MASK_SHAPE
            or not isinstance(frame_rows, list)
            or len(frame_rows) != FRAME_COUNT
            or [item.get("frame_id") for item in frame_rows] != list(range(FRAME_COUNT))
        ):
            raise RuntimeError(f"visible candidate frame-axis drift: {instance_id}")
        masks[instance_id] = mask
        states[instance_id] = frame_rows
        references[f"semantic_{instance_id}"] = make_ref(mask_path)
        references[f"state_{instance_id}"] = make_ref(state_path)
    return {
        "masks": masks,
        "states": states,
        "references": references,
        "input_video": video_path,
    }


def _project(point: Sequence[float], intrinsics: np.ndarray) -> tuple[int, int] | None:
    x, y, z = (float(value) for value in point)
    if not math.isfinite(z) or z <= 0.0:
        return None
    return (
        int(round(2.0 * (intrinsics[0, 0] * x / z + intrinsics[0, 2]) + 0.5)),
        int(round(2.0 * (intrinsics[1, 1] * y / z + intrinsics[1, 2]) + 0.5)),
    )


def decode_video(path: Path, expected_frames: int) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    count = 0
    width = height = 0
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    while True:
        ok, image = capture.read()
        if not ok:
            break
        height, width = image.shape[:2]
        count += 1
    capture.release()
    return {
        "frame_count": count,
        "expected_frame_count": expected_frames,
        "full_decode": count == expected_frames,
        "width": width,
        "height": height,
        "fps": fps,
    }


def render_review(
    source: Path,
    destination: Path,
    masks: Mapping[str, PackedMask],
    objects: Sequence[Mapping[str, Any]],
    intrinsics_by_frame: Sequence[np.ndarray],
) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(source))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (height, width) != MASK_SHAPE:
        capture.release()
        raise RuntimeError(f"review input geometry drift: {(width, height)}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("failed to open review writer")
    colors = [(70, 220, 70), (60, 170, 255), (230, 100, 230)]
    by_id = {str(row["instance_id"]): row for row in objects}
    frame_index = 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        overlay = image.copy()
        for color, instance_id in zip(colors, INSTANCE_IDS):
            row = by_id[instance_id]["frames"][frame_index]
            if row["mask_state"] != "unknown":
                mask = masks[instance_id].unpack(frame_index)
                overlay[mask] = (
                    0.72 * overlay[mask].astype(np.float32) + 0.28 * np.asarray(color)
                ).astype(np.uint8)
                contours, _ = cv2.findContours(
                    mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(overlay, contours, -1, color, 2)
            center_record = row["center_xyz"]
            if center_record["observability"] != "UNOBSERVABLE":
                center = np.asarray(center_record["estimate"]["xyz_m"], np.float64)
                uv = _project(center, intrinsics_by_frame[frame_index])
                if uv is not None:
                    cv2.circle(overlay, uv, 5, color, -1, cv2.LINE_AA)
                    cv2.putText(
                        overlay,
                        instance_id[-2:] + " ID?",
                        (uv[0] + 7, uv[1] - 7),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.48,
                        color,
                        2,
                        cv2.LINE_AA,
                    )
        cv2.rectangle(overlay, (0, 0), (width, 92), (0, 0, 0), -1)
        cv2.putText(
            overlay,
            f"frame {frame_index:04d} | REVIEW ONLY | CONSUMER FORBIDDEN",
            (18, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            "physical card identity UNKNOWN | face identity UNKNOWN",
            (18, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 200, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            "direct visible optical-Z only | no hidden geometry / Contact / metric truth",
            (18, 84),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(overlay)
        frame_index += 1
    capture.release()
    writer.release()
    decoded = decode_video(destination, FRAME_COUNT)
    if not decoded["full_decode"]:
        raise RuntimeError("review output does not fully decode")
    return decoded


def summarize(objects: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    fields = ("center_xyz", "plane_normal", "inplane_rotation", "finite_visible_patch")
    for obj in objects:
        frames = obj["frames"]
        result[str(obj["instance_id"])] = {
            "geometry_track_id": obj["geometry_track_id"],
            "physical_card_identity": PHYSICAL_IDENTITY,
            "face_identity": FACE_IDENTITY,
            "consumer_allowed": False,
            "frame_count": len(frames),
            "observable": {
                field: sum(row[field]["observability"] != "UNOBSERVABLE" for row in frames)
                for field in fields
            },
            "mask_state_counts": dict(Counter(str(row["mask_state"]) for row in frames)),
        }
    return result


def generate_candidate(
    depth: Mapping[str, Any], masks: Mapping[str, Any], stage: Path, final: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    mapping, mapping_valid = depth_to_mask_map()
    depth_cache: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    intrinsics_by_frame: list[np.ndarray] = []
    for frame_index, frame_row in enumerate(depth["frames"]):
        loaded = load_depth_frame(frame_row, frame_index)
        depth_cache.append(loaded)
        intrinsics_by_frame.append(loaded[2])

    objects: list[dict[str, Any]] = []
    packed_support: list[np.ndarray] = []
    for instance_index, instance_id in enumerate(INSTANCE_IDS):
        frames: list[dict[str, Any]] = []
        supports: list[np.ndarray] = []
        for frame_index, (depth_m, valid, intrinsics) in enumerate(depth_cache):
            state = masks["states"][instance_id][frame_index]
            admitted_visible = (
                state.get("semantic_admitted") is True
                and state.get("visibility_state") == "VISIBLE_CANDIDATE"
                and state.get("tracking_state") in {"seeded", "tracked", "reseeded"}
            )
            mask_state = str(state["tracking_state"]) if admitted_visible else "unknown"
            semantic_mask = masks["masks"][instance_id].unpack(frame_index)
            if not admitted_visible:
                semantic_mask = np.zeros_like(semantic_mask)
            support, _points = registered_support(
                semantic_mask,
                depth_m,
                valid,
                intrinsics,
                mapping,
                mapping_valid,
            )
            planar = estimate_planar_frame(
                PlanarFrameInput(
                    frame_index=frame_index,
                    mask=semantic_mask,
                    mask_state=mask_state,
                    visibility_state=DIRECT_VISIBILITY if admitted_visible else "UNKNOWN",
                    depth_m=depth_m,
                    depth_valid=valid,
                    depth_intrinsics=intrinsics,
                    depth_to_mask_xy=mapping,
                    registration_valid=mapping_valid,
                    depth_reference=DEPTH_REFERENCE,
                    registration_authority=MASK_REGISTRATION_AUTHORITY,
                )
            )
            frames.append(
                {
                    **planar,
                    "finite_visible_patch": finite_patch_record(
                        support,
                        depth_m,
                        intrinsics,
                        archive_row=instance_index * FRAME_COUNT + frame_index,
                    ),
                    "full_extent": {
                        "observability": "UNOBSERVABLE",
                        "estimate": None,
                        "residual": None,
                        "reason": "FULL_OBJECT_BOUNDARY_AND_THICKNESS_UNPROVEN",
                        "semantics": "HIDDEN_EXTENT_NOT_COMPLETED",
                    },
                    "physical_card_identity": PHYSICAL_IDENTITY,
                    "face_identity": FACE_IDENTITY,
                    "consumer_allowed": False,
                }
            )
            supports.append(np.packbits(support.reshape(-1), bitorder="big"))
        temporally_unify_axes(frames)
        packed_support.append(np.stack(supports))
        objects.append(
            {
                "instance_id": instance_id,
                "geometry_track_id": instance_id,
                "geometry_track_id_scope": "SESSION_PROMPT_STATE_ONLY",
                "physical_card_identity": PHYSICAL_IDENTITY,
                "face_identity": FACE_IDENTITY,
                "consumer_allowed": False,
                "allowed_consumers": ["REVIEW_DIAGNOSTIC_ONLY"],
                "entity_role": "VISIBLE_OBJECT_CANDIDATE_NOT_BOUND_PHYSICAL_INSTANCE",
                "geometry_class": "PLAYING_CARD_DIRECT_VISIBLE_FINITE_PATCH",
                "frames": frames,
            }
        )

    support_stage = stage / "VISIBLE_PATCH_SUPPORT_REVIEW_ONLY.npz"
    atomic_npz(
        support_stage,
        packed=np.concatenate(packed_support, axis=0),
        instance_ids=np.asarray(INSTANCE_IDS),
        physical_card_identity=np.asarray(PHYSICAL_IDENTITY),
        face_identity=np.asarray(FACE_IDENTITY),
        consumer_allowed=np.asarray(False),
        review_only=np.asarray(True),
        frame_count=np.asarray(FRAME_COUNT, np.int32),
        height=np.asarray(DEPTH_SHAPE[0], np.int32),
        width=np.asarray(DEPTH_SHAPE[1], np.int32),
        bitorder=np.asarray("big"),
        row_layout=np.asarray("INSTANCE_MAJOR_FLAT_ROW"),
        semantics=np.asarray("EXACT_DIRECT_VISIBLE_DEPTH_SUPPORT_REVIEW_ONLY"),
    )
    final_support = final / support_stage.name
    summary = summarize(objects)
    document = {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": CANDIDATE_ID,
        "session_id": SESSION_ID,
        "task": TASK,
        "frame_count": FRAME_COUNT,
        "status": "COMPLETED_REVIEW_ONLY_VISIBLE_SURFACE",
        "review_only": True,
        "consumer_allowed": False,
        "allowed_consumers": ["REVIEW_DIAGNOSTIC_ONLY"],
        "coordinate_domain": DEPTH_REFERENCE,
        "image_domain": IMAGE_DOMAIN,
        "center_xyz_semantics": "DIRECT_VISIBLE_SURFACE_CENTROID_NOT_OBJECT_FIXED_CENTER",
        "geometry_track_id_semantics": "SESSION_PROMPT_STATE_ONLY_NOT_PHYSICAL_IDENTITY",
        "physical_card_identity": PHYSICAL_IDENTITY,
        "face_identity": FACE_IDENTITY,
        "normal_sign_policy": "TEMPORALLY_CONTINUOUS_SEEDED_Z_NONPOSITIVE",
        "inplane_axis_semantics": "PI_PERIODIC_VISIBLE_SUPPORT_AXIS",
        "visible_patch_support_archive": projected_ref(support_stage, final_support),
        "external_metric_accuracy": "UNVERIFIED",
        "hidden_geometry_inferred": False,
        "object_thickness_inferred": False,
        "full_object_extent_inferred": False,
        "contact_authority": "NONE",
        "robot_authority": "NONE",
        "training_eligible": False,
        "control_ground_truth": False,
        "physical_deployment_authorized": False,
        "inputs": {
            "depth": depth["references"],
            "visible_mask_candidate": masks["references"],
            "depth_to_mask_mapping": PIXEL_MAP,
        },
        "objects": objects,
        "summary": summary,
        "claim_limit": (
            "Review-only direct-visible finite surface evidence. Geometry track IDs are "
            "not physical-card or face identities. No consumer admission, hidden geometry, "
            "external metric accuracy, Contact, Robot, Clean, training, control, or deployment authority."
        ),
    }
    document_stage = stage / "OBJECT6D_VISIBLE_PATCH_REVIEW_ONLY.json"
    atomic_json(document_stage, document)
    review_stage = stage / "OBJECT6D_REVIEW_ONLY.mp4"
    review = render_review(
        masks["input_video"], review_stage, masks["masks"], objects, intrinsics_by_frame
    )
    return document, {
        "document_stage": document_stage,
        "support_stage": support_stage,
        "review_stage": review_stage,
        "review": review,
        "summary": summary,
    }


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config = load_json(config_path)
    validate_config(config)
    output = Path(str(config["output_root"])).resolve()
    visual = Path(str(config["visual_root"])).resolve()
    for path in (output, visual):
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"fresh output required: {path}")

    output.parent.mkdir(parents=True, exist_ok=True)
    visual.parent.mkdir(parents=True, exist_ok=True)
    stage = output.parent / f".{output.name}.staging-{os.getpid()}-{uuid.uuid4().hex}"
    visual_stage = visual.parent / f".{visual.name}.staging-{os.getpid()}-{uuid.uuid4().hex}"
    stage.mkdir()
    visual_stage.mkdir()
    try:
        regression = verify_regressions(config)
        depth = validate_depth(config)
        masks = validate_masks(config)
        input_snapshot_sha = canonical_sha(
            {"depth": depth["references"], "visible_mask_candidate": masks["references"]}
        )
        atomic_json(stage / "REGRESSION_PIN_VERIFICATION.json", regression)
        atomic_json(
            stage / "INPUT_VERIFICATION.json",
            {
                "schema_version": SCHEMA_VERSION,
                "status": "PASSED_PINNED_INPUT_VERIFICATION",
                "input_snapshot_sha256": input_snapshot_sha,
                "depth": depth["references"],
                "visible_mask_candidate": masks["references"],
                "physical_card_identity": PHYSICAL_IDENTITY,
                "face_identity": FACE_IDENTITY,
                "consumer_allowed": False,
                "review_only": True,
            },
        )
        _document, artifacts = generate_candidate(depth, masks, stage, output)

        # Re-read every directly pinned input after computation.  Per-frame Depth
        # archives were verified while loading through load_depth_frame().
        regression_after = verify_regressions(config)
        depth_after = validate_depth(config)
        masks_after = validate_masks(config)
        input_snapshot_after = canonical_sha(
            {
                "depth": depth_after["references"],
                "visible_mask_candidate": masks_after["references"],
            }
        )
        if input_snapshot_after != input_snapshot_sha or regression_after != regression:
            raise RuntimeError("input or immutable regression drift during canary")

        shallow_stage = visual_stage / f"{SESSION_ID}_OBJECT6D_REVIEW_ONLY.mp4"
        shutil.copy2(artifacts["review_stage"], shallow_stage)
        if sha256(shallow_stage) != sha256(artifacts["review_stage"]):
            raise RuntimeError("shallow review copy differs from session review")
        visible_counts = {
            instance_id: artifacts["summary"][instance_id]["observable"]["finite_visible_patch"]
            for instance_id in INSTANCE_IDS
        }
        numeric_candidate_present = any(value > 0 for value in visible_counts.values())
        status = (
            "COMPLETED_REVIEW_ONLY_IDENTITY_UNKNOWN"
            if numeric_candidate_present
            else "REJECTED_NO_DIRECT_VISIBLE_FINITE_PATCH"
        )
        result = {
            "schema_version": SCHEMA_VERSION,
            "candidate_id": CANDIDATE_ID,
            "session_id": SESSION_ID,
            "task": TASK,
            "frame_count": FRAME_COUNT,
            "status": status,
            "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "review_only": True,
            "consumer_allowed": False,
            "allowed_consumers": ["REVIEW_DIAGNOSTIC_ONLY"],
            "physical_card_identity": PHYSICAL_IDENTITY,
            "face_identity": FACE_IDENTITY,
            "geometry_track_id_is_physical_identity": False,
            "visible_patch_counts": visible_counts,
            "coordinate_domain": DEPTH_REFERENCE,
            "image_domain": IMAGE_DOMAIN,
            "weights": "ABSENT",
            "gpu_used": False,
            "foundationstereo_rerun_performed": False,
            "mask_model_rerun_performed": False,
            "external_metric_accuracy_claimed": False,
            "hidden_geometry_inferred": False,
            "contact_authority": False,
            "control_ground_truth": False,
            "training_eligible": False,
            "physical_deployment_authorized": False,
            "source_mutated": False,
            "immutable_regressions_mutated": False,
            "object6d": projected_ref(
                artifacts["document_stage"], output / artifacts["document_stage"].name
            ),
            "visible_patch_support": projected_ref(
                artifacts["support_stage"], output / artifacts["support_stage"].name
            ),
            "review": {
                **artifacts["review"],
                "video": projected_ref(
                    artifacts["review_stage"], output / artifacts["review_stage"].name
                ),
                "shallow_video": projected_ref(
                    shallow_stage, visual / shallow_stage.name
                ),
            },
            "config": make_ref(config_path),
            "input_verification": projected_ref(
                stage / "INPUT_VERIFICATION.json", output / "INPUT_VERIFICATION.json"
            ),
            "regression_verification": projected_ref(
                stage / "REGRESSION_PIN_VERIFICATION.json",
                output / "REGRESSION_PIN_VERIFICATION.json",
            ),
            "claim_limit": (
                "CPU-only Poker044 review canary from pinned cached Depth and visible-mask "
                "candidate data. Identity remains UNKNOWN and every downstream consumer is "
                "forbidden; no Contact, hidden geometry, metric truth, control, training, or deployment claim."
            ),
        }
        atomic_json(stage / "RESULT.json", result)
        atomic_json(
            visual_stage / "INDEX.json",
            {
                "schema_version": SCHEMA_VERSION,
                "status": status,
                "review_only": True,
                "consumer_allowed": False,
                "video": projected_ref(shallow_stage, visual / shallow_stage.name),
                "result": projected_ref(stage / "RESULT.json", output / "RESULT.json"),
            },
        )
        os.replace(stage, output)
        os.replace(visual_stage, visual)
        return load_json(output / "RESULT.json")
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        shutil.rmtree(visual_stage, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.config)
    print(
        json.dumps(
            {
                "status": result["status"],
                "consumer_allowed": result["consumer_allowed"],
                "physical_card_identity": result["physical_card_identity"],
                "face_identity": result["face_identity"],
                "result": result["object6d"]["path"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
