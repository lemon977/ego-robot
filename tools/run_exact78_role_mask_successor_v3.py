#!/usr/bin/env python3
"""Fresh V3 full-session SAM3.1 role-removal Mask runner for exact78.

Outputs only anatomical human/tracker role masks and their union.  Task-object
identity is a separate mandatory artifact and is never inferred here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import cv2
import numpy as np
import torch


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from tools import run_configurable_bilateral_mask_canary as canary
from tools import run_newtask_baseline_sam31_mask_probe as baseline
from tools import run_chips001_pico_mask_temporal as base
from tools import run_assisted_bilateral_flow_refresh as flow
from tools import run_exact78_tracker_reentry_bounded_v2_canary as reentry


class RoleMaskError(RuntimeError):
    pass


def now() -> str:
    return canary.now()


def sha(path: Path) -> str:
    return canary.sha256(path)


def array_sha(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(value.dtype.str.encode() + b"\0")
    digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    return canary.file_ref(path)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RoleMaskError(f"JSON object required: {path}")
    return value


def write(path: Path, value: Any) -> None:
    canary.write_json_new(path, value)


def validate(config_path: Path, output: Path, holder: str) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config = load(config_path)
    required = {
        "schema_version", "task_id", "session_id", "raw_all_data", "raw_video", "hawor_npz",
        "frame_count", "anchor_frame", "fps", "human_anchors", "tracker_roles",
        "tracker_anatomical", "tracker_human_internal", "tracker_prompt_contract",
        "refresh_cadence_frames", "review_language", "admission_canaries",
    }
    if set(config) != required or config["schema_version"] != "exact78-fullsession-role-mask-config-v1":
        raise RoleMaskError("config schema/keys mismatch")
    if output.exists() or output.is_symlink() or not output.is_absolute():
        raise RoleMaskError("fresh absolute output required")
    lease = load(PROJECT / "_run/GPU_LEASE.json")
    if lease.get("status") != "ACQUIRED" or lease.get("holder") != holder:
        raise RoleMaskError("central GPU lease is not held by exact holder")
    if not torch.cuda.is_available():
        raise RoleMaskError("CUDA_UNAVAILABLE_IN_CURRENT_CONTAINER")
    raw_root = Path(config["raw_all_data"]).resolve(strict=True)
    raw_video = canary.verify_ref(config["raw_video"], "raw_video")
    hawor = canary.verify_ref(config["hawor_npz"], "hawor_npz")
    count = int(config["frame_count"])
    anchor = int(config["anchor_frame"])
    if count <= 0 or not 0 <= anchor < count:
        raise RoleMaskError("invalid frame_count/anchor")
    frame_paths = []
    for frame_id in range(count):
        path = (raw_root / f"{frame_id:05d}" / "rgb.png").resolve(strict=True)
        if not path.is_file() or path.is_symlink():
            raise RoleMaskError(f"bad raw frame {path}")
        frame_paths.append(path)
    image = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    if image is None or image.shape[:2] != (960, 1280):
        raise RoleMaskError("expected 1280x960 raw frames")
    capture = cv2.VideoCapture(str(raw_video))
    video_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    video_fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    if video_count <= 0 or not np.isfinite(video_fps) or video_fps <= 0:
        raise RoleMaskError("raw video metadata unreadable")
    # The lossless preprocess/all_data RGB sequence is the actual model input.
    # Historical 0901 MP4s can report a different decoded count or FPS while
    # the complete pinned RGB sequence and HaWoR arrays agree.  Record that
    # diagnostic, but never synthesize/drop RGB frames or abort before SAM.
    video_metadata = {
        "decoded_frame_count": video_count,
        "declared_fps": video_fps,
        "model_input_rgb_frame_count": count,
        "model_input_fps": float(config["fps"]),
        "count_matches_model_input": video_count == count,
        "fps_matches_model_input": abs(video_fps - float(config["fps"])) <= 0.01,
        "authority": "MODEL_INPUT_IS_PINNED_PREPROCESS_RGB_SEQUENCE",
    }
    with np.load(hawor, allow_pickle=False) as archive:
        if archive["joints_2d"].shape != (2, count, 21, 2):
            raise RoleMaskError("HaWoR joints_2d shape mismatch")
        if archive["anatomical_side_names"].astype(str).tolist() != ["left", "right"]:
            raise RoleMaskError("HaWoR anatomical side mismatch")
    if set(config["human_anchors"]) != {"left_human", "right_human"}:
        raise RoleMaskError("both human roles required")
    if not config["tracker_roles"] or set(config["tracker_roles"]) != set(config["tracker_anatomical"]):
        raise RoleMaskError("tracker roles/mapping mismatch")
    for role, value in config["tracker_roles"].items():
        if value["name"] != role or not value["positive"] or not value["negative"]:
            raise RoleMaskError(f"bad tracker prompt {role}")
    return {
        "config": config, "config_path": config_path, "raw_root": raw_root,
        "raw_video": raw_video, "hawor_path": hawor, "frame_paths": frame_paths,
        "indices": list(range(count)), "anchor_slot": anchor, "fps": float(config["fps"]),
        "raw_video_metadata_diagnostic": video_metadata,
        "height": 960, "width": 1280,
    }


def track_roles_bounded_reentry(
    model: Any,
    validated: dict[str, Any],
    output: Path,
    frame_paths: list[Path],
    human: dict[str, dict[int, np.ndarray]],
) -> tuple[dict[str, dict[int, np.ndarray]], dict[str, Any]]:
    with np.load(validated["hawor_path"], allow_pickle=False) as archive:
        joints = archive["joints_2d"].astype(float)
        observed = archive["observed"].astype(bool)
    cache = flow.FlowCache(frame_paths, 640, 480)
    trackers: dict[str, dict[int, np.ndarray]] = {"left_tracker": {}, "right_tracker": {}}
    metadata: dict[str, Any] = {}
    for side, role in enumerate(("left_tracker", "right_tracker")):
        current = np.zeros((validated["height"], validated["width"]), bool)
        previous_visible = False
        attempts = accepted = rejected = warped_accepted = empty_closed = 0
        records = []
        own_human = human[f"{'left' if side == 0 else 'right'}_human"]
        for local, source_frame in enumerate(validated["indices"]):
            geometry = reentry.tracker_geometry(joints, observed, side, source_frame)
            if not geometry["expected_visible"]:
                current = np.zeros_like(current)
                trackers[role][local] = current.copy()
                previous_visible = False
                empty_closed += 1
                records.append({
                    "source_frame": source_frame,
                    "geometry": geometry,
                    "provider": "EMPTY_FAIL_CLOSED_NOT_VISIBLE",
                    "area_pixels": 0,
                })
                continue
            center = np.asarray(geometry["center"], dtype=float)
            if local and previous_visible and current.any():
                warped, flow_record = cache.warp(current, local - 1, local)
                warped = reentry.local_component(warped, center)
                current_warp_gate = reentry.warp_gate(warped, geometry, own_human[local])
            else:
                warped = np.zeros_like(current)
                flow_record = None
                current_warp_gate = {"pass": False, "reason": "NO_VISIBLE_PREDECESSOR"}
            refresh = bool(source_frame % reentry.CADENCE == 0 or not previous_visible or not current.any())
            proposal = None
            proposal_gate = None
            sam_summary = None
            if refresh:
                attempts += 1
                proposal, sam_summary = reentry.sam_current_frame(
                    model,
                    frame_paths[local],
                    geometry,
                    202 if side == 0 else 201,
                    output / "tracker_reentry_inputs" / role / f"{source_frame:05d}",
                )
                proposal_gate = reentry.proposal_gate(proposal, geometry, own_human[local])
            if proposal is not None and proposal_gate and proposal_gate["pass"]:
                current = proposal
                accepted += 1
                provider = "SAM31_CURRENT_BOUNDED_WRIST_PALM_ACCEPTED"
            elif current_warp_gate.get("pass"):
                current = warped
                warped_accepted += 1
                provider = "RAW_DIS_WARP_CURRENT_WRIST_ROI_ACCEPTED"
                if refresh:
                    rejected += 1
            else:
                current = np.zeros_like(current)
                empty_closed += 1
                provider = "EMPTY_FAIL_CLOSED_REFRESH_AND_WARP_REJECTED"
                if refresh:
                    rejected += 1
            trackers[role][local] = current.copy()
            previous_visible = True
            records.append({
                "source_frame": source_frame,
                "geometry": geometry,
                "refresh_attempted": refresh,
                "sam_summary": sam_summary,
                "proposal_gate": proposal_gate,
                "flow_record": flow_record,
                "warp_gate": current_warp_gate,
                "provider": provider,
                "area_pixels": int(current.sum()),
            })
        expected = sum(row["geometry"]["expected_visible"] for row in records)
        present = sum(bool(trackers[role][local].any()) and row["geometry"]["expected_visible"] for local, row in enumerate(records))
        drift = sum(bool(trackers[role][local].any()) and not row["geometry"]["expected_visible"] for local, row in enumerate(records))
        metadata[role] = {
            "full_frame_denominator": len(validated["indices"]),
            "expected_visible_denominator": expected,
            "present_on_expected_visible": present,
            "visible_coverage_fraction": float(present / max(expected, 1)),
            "offscreen_or_unobserved_empty_frames": sum(not row["geometry"]["expected_visible"] for row in records),
            "background_drift_on_not_visible_frames": drift,
            "refresh_attempt_count": attempts,
            "refresh_accept_count": accepted,
            "refresh_reject_count": rejected,
            "warp_accept_count": warped_accepted,
            "empty_fail_closed_count": empty_closed,
            "records": records,
        }
    return trackers, metadata


def adaptive_gates(validated: dict[str, Any], detailed: dict[str, Any], tracker_meta: dict[str, Any]) -> dict[str, Any]:
    human_fractions = {}
    for role, rows in detailed["human_fixed_denominator"].items():
        expected = [row for row in rows if row["joint_visible_denominator"] > 0]
        human_fractions[role] = sum(row["pass"] for row in expected) / max(len(expected), 1)
    tracker_fractions = {
        role: (None if value["expected_visible_denominator"] == 0 else value["visible_coverage_fraction"])
        for role, value in tracker_meta.items()
    }
    tracker_applicability = {
        role: (
            "PASS_NOT_APPLICABLE_ZERO_EXPECTED_VISIBLE_EMPTY_FAIL_CLOSED"
            if value["expected_visible_denominator"] == 0
            and value["present_on_expected_visible"] == 0
            and value["background_drift_on_not_visible_frames"] == 0
            and value["offscreen_or_unobserved_empty_frames"] == value["full_frame_denominator"]
            else "PASS_APPLICABLE_COVERAGE_AT_LEAST_0P80"
            if value["expected_visible_denominator"] > 0 and value["visible_coverage_fraction"] >= 0.80
            else "FAIL"
        )
        for role, value in tracker_meta.items()
    }
    full_count = len(validated["indices"])
    gates = {
        "human_visible_frame_pass_fraction_at_least_0p95": all(value >= 0.95 for value in human_fractions.values()),
        "tracker_expected_visible_coverage_fraction_at_least_0p80": bool(tracker_applicability) and all(value.startswith("PASS_") for value in tracker_applicability.values()),
        "tracker_offscreen_or_unobserved_is_empty": all(value["background_drift_on_not_visible_frames"] == 0 for value in tracker_meta.values()),
        "tracker_current_wrist_roi_bounded": all(
            all(
                (not row["geometry"]["expected_visible"] and row["area_pixels"] == 0)
                or row["provider"] in {
                    "SAM31_CURRENT_BOUNDED_WRIST_PALM_ACCEPTED",
                    "RAW_DIS_WARP_CURRENT_WRIST_ROI_ACCEPTED",
                    "EMPTY_FAIL_CLOSED_REFRESH_AND_WARP_REJECTED",
                }
                for row in value["records"]
            )
            for value in tracker_meta.values()
        ),
        "full_frame_denominator_recorded": all(len(rows) == full_count for rows in detailed["human_fixed_denominator"].values()) and all(value["full_frame_denominator"] == full_count and len(value["records"]) == full_count for value in tracker_meta.values()),
    }
    summary = {role: {key: value for key, value in data.items() if key != "records"} for role, data in tracker_meta.items()}
    return {"hard_gates": gates, "human_pass_fractions": human_fractions, "tracker_expected_visible_coverage_fractions": tracker_fractions, "tracker_coverage_applicability": tracker_applicability, "tracker_reentry_summary": summary, "details": detailed}


def execute(validated: dict[str, Any], output: Path, holder: str) -> int:
    config = validated["config"]
    base.RAW = validated["raw_root"]
    base.SOURCE_INDICES = validated["indices"]
    base.FRAME_COUNT = len(validated["indices"])
    base.ANCHOR_FRAME = validated["anchor_slot"]
    base.CANARY = list(range(base.FRAME_COUNT))
    base.TASK_ID = config["task_id"]
    base.SESSION_ID = config["session_id"]
    base.HUMAN_ANCHORS = {key: tuple(value) for key, value in config["human_anchors"].items()}
    base.TRACKER_ROLES = config["tracker_roles"]
    base.TRACKER_ANATOMICAL = config["tracker_anatomical"]
    base.TRACKER_HUMAN_INTERNAL = config["tracker_human_internal"]
    output.mkdir(parents=True)
    lease_snapshot = output / "GPU_LEASE_AT_EXECUTION.json"
    write(lease_snapshot, load(PROJECT / "_run/GPU_LEASE.json"))
    frame_root, frame_paths = base.materialize_view(output / "input_frames", list(range(base.FRAME_COUNT)))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    adapter_module = baseline.load_adapter_module()
    if str(base.CODE_ROOT) not in sys.path:
        sys.path.insert(0, str(base.CODE_ROOT))
    adapter, build_evidence = adapter_module.build_pinned_adapter(official_code_root=base.CODE_ROOT, checkpoint_path=base.CHECKPOINT)
    try:
        human_raw, human_meta = base.collect_text_roles(
            adapter.model, frame_root, "a person's hand and forearm", base.ANCHOR_FRAME,
            base.HUMAN_ANCHORS, validated["height"], validated["width"], True, 28.0,
        )
        cache = flow.FlowCache(frame_paths, 640, 480)
        human, human_completion = base.complete_with_flow(human_raw, frame_paths, cache, validated["height"], validated["width"])
        trackers, tracker_meta = track_roles_bounded_reentry(adapter.model, validated, output, frame_paths, human)
    finally:
        adapter.predictor.shutdown()
    detailed = canary.evaluate(validated, human, human_meta, human_completion, trackers, tracker_meta)
    metrics = adaptive_gates(validated, detailed, tracker_meta)
    grade = "B" if all(metrics["hard_gates"].values()) else "C"
    artifacts = canary.render_and_write_masks(output, validated, human, trackers, grade == "B")
    tracker_reentry_manifest = output / "TRACKER_REENTRY_MANIFEST.json"
    write(tracker_reentry_manifest, {
        "schema_version": "exact78-fullsession-tracker-reentry-manifest-v3",
        "created_at": now(),
        "task": config["task_id"],
        "session": config["session_id"],
        "method": "SAM31_PERIODIC_CURRENT_FRAME_BOUNDED_V3_WRIST_PALM_REANCHOR_WITH_RAW_DIS_WARP_EMPTY_FAIL_CLOSED_AND_EXPLICIT_APPLICABILITY",
        "cadence_frames": reentry.CADENCE,
        "current_wrist_roi_radius_pixels": reentry.ROI_RADIUS,
        "roles": tracker_meta,
    })
    frame_rows = []
    for local, (source_frame, frame_path) in enumerate(zip(validated["indices"], validated["frame_paths"])):
        image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        role_masks = {}
        for role in [*human, *trackers]:
            mask_path = output / "raw_role_masks" / role / f"{source_frame:05d}.png"
            role_masks[role] = ref(mask_path)
        frame_rows.append({
            "source_frame": source_frame, "selected_rgb_decoded_sha256": array_sha(image),
            "role_masks": role_masks,
            "hand_tracker_union": ref(output / "hand_tracker_union_masks" / f"{source_frame:05d}.png"),
        })
    manifest = output / "FRAME_MANIFEST.json"
    write(manifest, {
        "schema_version": "exact78-fullsession-role-mask-manifest-v1", "created_at": now(),
        "task": config["task_id"], "session": config["session_id"],
        "semantic_type": "ROLE_REMOVAL_MASK_NOT_TASK_OBJECT_IDENTITY_MASK",
        "formula": "left_human UNION right_human UNION visible tracker roles",
        "frames": frame_rows,
    })
    result = output / "RESULT.json"
    write(result, {
        "schema_version": "exact78-fullsession-role-mask-result-v3", "created_at": now(),
        "status": f"TERMINAL_GRADE_{grade}", "grade": grade, "stage": "MASK_ROLE_REMOVAL",
        "task": config["task_id"], "session": config["session_id"], "downstream_authorized": grade == "B",
        "frame_count": len(frame_rows), "semantic_type": "ROLE_REMOVAL_MASK_NOT_TASK_OBJECT_IDENTITY_MASK",
        "hard_gates": metrics["hard_gates"], "metrics": {key: value for key, value in metrics.items() if key != "details"},
        "artifacts": {**artifacts, "frame_manifest": ref(manifest), "tracker_reentry_manifest": ref(tracker_reentry_manifest)},
        "pins": {"config": ref(validated["config_path"]), "runner": ref(Path(__file__)), "raw_video": ref(validated["raw_video"]), "raw_video_metadata_diagnostic": validated["raw_video_metadata_diagnostic"], "model_input_rgb_sequence": {"root": str(validated["raw_root"]), "frame_count": len(frame_rows), "fps": validated["fps"]}, "hawor_npz": ref(validated["hawor_path"]), "gpu_lease_at_execution": ref(lease_snapshot), "checkpoint_sha256": base.CHECKPOINT_SHA256, "sam_build_evidence": build_evidence},
        "resource": {"wall_seconds": time.perf_counter() - started, "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()), "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved())},
        "claim_limit": "Fresh full-session bounded-v3 role-removal Mask only. MP4 metadata is diagnostic; pinned RGB frames are the model inputs. Separate task-object identity A/B is mandatory.",
    })
    review = output / "AGENT_REVIEW.json"
    write(review, {
        "schema_version": "baseline-agent-stage-review-v1", "created_at": now(),
        "stage": "MASK_ROLE_REMOVAL", "task": config["task_id"], "session": config["session_id"],
        "grade": grade, "downstream_authorized": grade == "B",
        "semantic_separation": {"role_removal_mask": "EMITTED", "task_object_identity_mask": "NOT_EMITTED"},
        "hard_gates": {key: "PASS" if value else "FAIL" for key, value in metrics["hard_gates"].items()},
        "result": ref(result),
    })
    with (output / "SHA256SUMS.txt").open("x", encoding="utf-8") as stream:
        for path in sorted(item for item in output.rglob("*") if item.is_file() and item.name != "SHA256SUMS.txt"):
            stream.write(f"{sha(path)}  {path.relative_to(output)}\n")
    return 0 if grade == "B" else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--confirm-config-sha", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--lease-holder", required=True)
    args = parser.parse_args()
    if sha(args.config.resolve(strict=True)) != args.confirm_config_sha:
        raise RoleMaskError("config SHA confirmation mismatch")
    validated = validate(args.config, args.output_root, args.lease_holder)
    return execute(validated, args.output_root, args.lease_holder)


if __name__ == "__main__":
    raise SystemExit(main())
