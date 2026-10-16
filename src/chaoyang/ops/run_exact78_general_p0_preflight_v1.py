#!/usr/bin/env python3
"""CPU-only preflight for the frozen Exact78 156-session visual cohort."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from chaoyang.governance.common import artifact_ref, atomic_json, load_json


EXPECTED_PARENT_ROBOT_SHA = "8338f2246e476e6afef2995eaa18423a4aafe0625ae829bd5315f809120a78e1"
EXPECTED_COHORT_SHA = "b6dcfec1fb41b9ffb0b9b19cc62d029f5cedf57f41552a611272cc9e5c6e43a8"


def probe_video(path: Path, *, decode: bool) -> dict:
    if not path.is_file():
        return {"path": str(path), "exists": False, "decode_ok": False, "error": "MISSING"}
    command = [
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,nb_read_frames",
        "-of", "json", str(path),
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    row = {"path": str(path), "exists": True, "bytes": path.stat().st_size,
           "probe_returncode": result.returncode}
    if result.returncode:
        row.update(decode_ok=False, error=(result.stderr or "FFPROBE_FAILED")[-2000:])
        return row
    stream = json.loads(result.stdout)["streams"][0]
    rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1"
    try:
        fps = float(Fraction(rate))
    except (ValueError, ZeroDivisionError):
        fps = 0.0
    count = stream.get("nb_read_frames") or stream.get("nb_frames")
    row.update(width=int(stream["width"]), height=int(stream["height"]), fps=fps,
               decoded_frames=int(count) if count and str(count).isdigit() else None)
    if decode:
        decoded = subprocess.run(
            ["ffmpeg", "-v", "error", "-threads", "1", "-i", str(path),
             "-map", "0:v:0", "-f", "null", "-"],
            text=True, capture_output=True, check=False,
        )
        row["decode_returncode"] = decoded.returncode
        row["decode_ok"] = decoded.returncode == 0
        if decoded.returncode:
            row["decode_error"] = decoded.stderr[-4000:]
    else:
        row["decode_ok"] = result.returncode == 0
    return row


def valid_matrix(value: object, shape: tuple[int, int]) -> bool:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return array.shape == shape and bool(np.isfinite(array).all())


def inspect_session(spec: dict) -> dict:
    root = Path(spec["raw_path"])
    session = spec["session_id"]
    mono = root / f"CameraRecord_{session}.mp4"
    stereo = root / "source_stereo" / f"CameraRecord_{session}_stereo.mp4"
    clip_path = root / "clip_manifest.json"
    humanego_path = root / "preprocess/pico_humanego_manifest.json"
    camera_path = root / "camera_params.json"
    row = {
        "position": spec["position"], "task": spec["task"], "session_id": session,
        "expected_frames": spec["frame_count"], "expected_input_identity_sha256": spec["immutable_input_identity_sha256"],
        "frozen_calibration_state": spec["stereo_calibration_state"], "source_root": str(root),
    }
    errors: list[str] = []
    for name, path in (("clip_manifest", clip_path), ("humanego_manifest", humanego_path),
                       ("camera_params", camera_path)):
        if not path.is_file():
            errors.append(f"MISSING_{name.upper()}")
    clip = load_json(clip_path) if clip_path.is_file() else {}
    humanego = load_json(humanego_path) if humanego_path.is_file() else {}
    camera = load_json(camera_path) if camera_path.is_file() else {}
    row["manifest_checks"] = {
        "clip_name_matches": clip.get("clip_name") == session,
        "clip_frames": (clip.get("video") or {}).get("frame_count"),
        "humanego_frames": humanego.get("frame_count"),
        "humanego_fps": humanego.get("fps"),
        "K_valid": valid_matrix(humanego.get("intrinsics"), (3, 3)),
        "stereo_calibration_present": valid_matrix((camera.get("extrinsics") or {}).get("left"), (4, 4))
            and valid_matrix((camera.get("extrinsics") or {}).get("right"), (4, 4)),
    }
    frame_dirs = sorted((root / "preprocess/all_data").glob("[0-9][0-9][0-9][0-9][0-9]"))
    ids = [int(path.name) for path in frame_dirs]
    metadata_ok = True
    c2w_ok = True
    k_ok = True
    if len(ids) != spec["frame_count"] or ids != list(range(spec["frame_count"])):
        errors.append("FRAME_DIRECTORY_IDENTITY")
    for frame, directory in enumerate(frame_dirs):
        metadata_path = directory / "training_data.json"
        rgb_path = directory / "rgb.png"
        if not metadata_path.is_file() or not rgb_path.is_file():
            metadata_ok = False
            continue
        try:
            metadata = load_json(metadata_path).get("metadata", {})
        except (OSError, ValueError, json.JSONDecodeError):
            metadata_ok = False
            continue
        metadata_ok &= metadata.get("idx") == frame
        c2w_ok &= valid_matrix(metadata.get("c2w"), (4, 4))
        k_ok &= valid_matrix(metadata.get("k"), (3, 3))
    row["frame_identity"] = {"directories": len(frame_dirs), "contiguous": ids == list(range(spec["frame_count"])),
                             "metadata_idx": bool(metadata_ok), "c2w": bool(c2w_ok), "K": bool(k_ok)}
    row["mono"] = probe_video(mono, decode=True)
    row["stereo"] = probe_video(stereo, decode=True)
    for name in ("mono", "stereo"):
        video = row[name]
        if not video.get("decode_ok"):
            errors.append(f"{name.upper()}_DECODE")
        if video.get("decoded_frames") != spec["frame_count"]:
            errors.append(f"{name.upper()}_FRAME_COUNT")
        if abs(float(video.get("fps") or 0.0) - 30.0) > 0.01:
            errors.append(f"{name.upper()}_FPS")
    checks = row["manifest_checks"]
    if checks.get("clip_frames") != spec["frame_count"] or checks.get("humanego_frames") != spec["frame_count"]:
        errors.append("MANIFEST_FRAME_COUNT")
    if not checks.get("clip_name_matches") or not metadata_ok or not c2w_ok or not k_ok:
        errors.append("FRAME_IDENTITY_OR_CAMERA")
    metric = spec["stereo_calibration_state"] == "CALIBRATED_METRIC_STEREO" and checks.get("stereo_calibration_present")
    row["errors"] = sorted(set(errors))
    row["input_valid"] = not row["errors"]
    row["layer_ceiling"] = "METRIC_OPTIONAL" if row["input_valid"] and metric else ("VISUAL" if row["input_valid"] else "INPUT_INVALID")
    return row


def arrays_equal(left: np.ndarray, right: np.ndarray) -> dict:
    same_shape = left.shape == right.shape
    equal = bool(same_shape and np.array_equal(left, right))
    close = bool(same_shape and np.allclose(left, right, rtol=0.0, atol=1e-12, equal_nan=True))
    maximum = float(np.max(np.abs(left - right))) if same_shape and left.size else None
    return {"same_shape": same_shape, "array_equal": equal, "allclose_atol_1e-12": close, "max_abs_diff": maximum}


def template_audit(packet: dict, v2_arm_path: Path, accepted_robot_path: Path, accepted_hawor_path: Path) -> dict:
    inherited = np.load(accepted_robot_path, allow_pickle=False)
    human = np.load(accepted_hawor_path, allow_pickle=False)
    v2 = np.load(v2_arm_path, allow_pickle=False)
    c2w0 = np.asarray(human["c2w"][0], dtype=np.float64)
    inherited_base = c2w0 @ np.asarray(inherited["T_camera_base"][0], dtype=np.float64)
    inherited_q0 = np.asarray(inherited["q_arm"][0], dtype=np.float64)
    source_human = np.asarray(human["joints_3d_world"][:, 0, 0], dtype=np.float64)
    offsets = np.empty((2, 3), dtype=np.float64)
    for side in range(2):
        target_camera = np.linalg.inv(c2w0) @ inherited["T_target_hand_root_world"][0, side]
        human_camera = np.linalg.inv(c2w0) @ np.r_[source_human[side], 1.0]
        offsets[side] = target_camera[:3, 3] - human_camera[:3]
    v2_q0 = np.asarray(v2["q_arm"][np.asarray(v2["anchor_frame_per_side"], dtype=int), np.arange(2)], dtype=np.float64)
    comparisons = {
        "base_before_fixed_backoff_vs_v2_T_world_base": arrays_equal(inherited_base, np.asarray(v2["T_world_base"])),
        "initial_joint_per_side_vs_v2_anchor_joint": arrays_equal(inherited_q0, v2_q0),
        "derived_wrist_offset_vs_v2_accepted_offset_camera": arrays_equal(offsets, np.asarray(v2["accepted_offset_camera"])),
        "inherited_anchor_frame_assumption_vs_v2_anchor_frame_per_side": arrays_equal(np.zeros(2, dtype=np.int64), np.asarray(v2["anchor_frame_per_side"])),
    }
    source_refs = {
        "accepted_robot_template": artifact_ref(accepted_robot_path),
        "accepted_hawor_template": artifact_ref(accepted_hawor_path),
        "v2_retained_arm": artifact_ref(v2_arm_path),
    }
    retry_refs = {key: packet["inputs"].get(key) for key in ("accepted_robot_template", "accepted_hawor_template")}
    return {
        "source_refs": source_refs, "retry4_refs": retry_refs,
        "retry4_template_bytes_currently_present": {key: Path(value["path"]).is_file() for key, value in retry_refs.items()},
        "derived_values": {"T_world_base_before_backoff": inherited_base.tolist(), "q_arm_initial": inherited_q0.tolist(),
                           "accepted_offset_camera": offsets.tolist(), "anchor_frame_per_side": [0, 0]},
        "v2_values": {key: np.asarray(v2[key]).tolist() for key in ("T_world_base", "accepted_offset_camera", "anchor_frame_per_side")},
        "comparisons": comparisons,
        "conclusion": "CARDS042_FITTED_VALUES_INHERITED" if any(item["allclose_atol_1e-12"] for item in comparisons.values())
                      else "NO_EQUAL_CARDS042_VALUES_FOUND",
        "code_fact": "run_robot_motion_transfer_arm_canary_v3 reads T_camera_base[0] and q_arm[0] from accepted states and derives accepted_offset_camera from the accepted Cards042 pair",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    packet = load_json(args.packet)
    output = args.output_root.resolve()
    if packet["task_id"] not in str(output) or packet["write_set"] != [str(output)]:
        raise RuntimeError("WRITE_SET_MISMATCH")
    cohort_path = Path(packet["inputs"]["frozen_cohort"]["path"])
    if artifact_ref(cohort_path) != packet["inputs"]["frozen_cohort"] or packet["inputs"]["frozen_cohort"]["sha256"] != EXPECTED_COHORT_SHA:
        raise RuntimeError("FROZEN_COHORT_SHA_DRIFT")
    parent = Path(packet["inputs"]["parent_robot_result"]["path"])
    if artifact_ref(parent) != packet["inputs"]["parent_robot_result"] or packet["inputs"]["parent_robot_result"]["sha256"] != EXPECTED_PARENT_ROBOT_SHA:
        raise RuntimeError("PARENT_ROBOT_SHA_DRIFT")
    cohort = load_json(cohort_path)
    sessions = cohort["sessions"]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(inspect_session, spec): spec["position"] for spec in sessions}
        rows = [future.result() for future in as_completed(futures)]
    rows.sort(key=lambda item: item["position"])
    atomic_json(output / "SESSION_PREFLIGHT.json", {"schema_version": "exact78-p0-session-preflight-v1", "sessions": rows})

    retry4_packet = load_json(Path(packet["inputs"]["retry4_packet"]["path"]))
    dependency_rows = []
    for ref in retry4_packet["runtime_code"]:
        path = Path(ref["path"])
        actual = artifact_ref(path) if path.is_file() else None
        dependency_rows.append({"expected": ref, "present": actual is not None, "sha_matches": actual == ref if actual else False,
                                "actual": actual})
    atomic_json(output / "DEPENDENCY_AUDIT.json", {"dependencies": dependency_rows,
                "missing": [row["expected"]["path"] for row in dependency_rows if not row["present"]],
                "sha_drift": [row["expected"]["path"] for row in dependency_rows if row["present"] and not row["sha_matches"]]})

    template = template_audit(retry4_packet, Path(packet["inputs"]["v2_retained_arm"]["path"]),
                              Path(packet["inputs"]["historical_accepted_robot"]["path"]),
                              Path(packet["inputs"]["historical_accepted_hawor"]["path"]))
    atomic_json(output / "TEMPLATE_LEAKAGE_AUDIT.json", template)

    valid = sum(row["input_valid"] for row in rows)
    metric = sum(row["layer_ceiling"] == "METRIC_OPTIONAL" for row in rows)
    total_frames = sum(row["expected_frames"] for row in rows)
    hawor_seconds_per_frame = (55.0 + 79.0) / (171 + 284)
    full_seconds_per_frame = (12 * 60 + 13 * 60) / (171 + 284)
    estimate = {
        "total_frames": total_frames,
        "P2_action_only": {
            "serial_hawor_hours": total_frames * hawor_seconds_per_frame / 3600,
            "planning_nominal_hours": 16.0,
            "planning_range_hours": [5.0, 20.0],
            "basis": "HaWoR measured 55s/171 and 79s/284 gives a 4.63h hard lower bound; CPU bounded/temporal/Arm/Hand was not independently timed, so 16h is the scheduling value until P1 replaces it.",
        },
        "P3_full_chain": {
            "sample_weighted_nominal_hours": total_frames * full_seconds_per_frame / 3600,
            "planning_range_hours": [43.2, 66.3],
            "basis": "Linear extrapolation of 12min/171 and 13min/284; lower/upper endpoints use the two observed per-frame rates.",
        },
    }
    atomic_json(output / "RUNTIME_ESTIMATE.json", estimate)
    clusters: dict[str, list[str]] = {}
    for row in rows:
        for error in row["errors"]:
            clusters.setdefault(error, []).append(row["session_id"])
    result = {
        "schema_version": "exact78-general-p0-result-v1", "status": "PASSED" if valid == 156 else "PASSED_WITH_INPUT_ISSUES",
        "execution_complete": True, "visual_qualified": False, "quality_passed": False,
        "counts": {"sessions": len(rows), "input_valid": valid, "input_invalid": len(rows)-valid,
                   "metric_ceiling": metric, "visual_only_ceiling": valid-metric, "total_frames": total_frames},
        "four_metrics": {
            "motion_completeness": "NOT_RUN_IN_P0", "cross_session_reachability": "NOT_RUN_IN_P0",
            "terminal_coverage": f"{len(rows)}/156 preflight terminal; {valid}/156 input-valid",
            "visual_quality_distribution": "NOT_RUN_IN_P0",
        },
        "dependency_missing_count": sum(not row["present"] for row in dependency_rows),
        "dependency_sha_drift_count": sum(row["present"] and not row["sha_matches"] for row in dependency_rows),
        "template_leakage_conclusion": template["conclusion"], "failure_clusters": clusters,
        "diagnostic_video": None, "diagnostic_video_reason": "P0 is a CPU input/dependency preflight and does not render video",
        "runtime_estimate": estimate, "elapsed_seconds": time.monotonic()-started,
        "training_eligible": False, "deletion_authorized": False,
        "next_action": "REGISTER_P1_ACTION_CANARY",
    }
    atomic_json(output / "RESULT.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
