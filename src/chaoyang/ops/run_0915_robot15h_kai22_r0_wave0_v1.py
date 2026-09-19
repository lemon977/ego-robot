#!/usr/bin/env python3
"""Build bounded HaWoR priors and Kai22 R0 for the four frozen W0 sessions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping
import uuid

import cv2
import numpy as np

from chaoyang.governance.robot15h_task_specs_v1 import WINDOW_RUN_ID, build_packet
from chaoyang.ops.run_0915_hawor_resize_only_canary_v1 import full_decode, open_encoder
from chaoyang.pipeline.interaction_contact_robot_dev_v1 import timestamp_motion_diagnostics
from chaoyang.pipeline.robot_renderer_cycles import forward_kinematics
from chaoyang.pipeline.robot_renderer_eevee_fullchain import load_pinned_robot_assets
from chaoyang.pipeline import robot_wrist_kai_adapter as wrist_adapter
from chaoyang.pipeline.robot_visual_relative_v1 import hand_limits, retarget_kaihand_frame


ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "0915_robot15h_kai22_r0_wave0_v1"
PHASE = "ROBOT15H_KAI22_R0_WAVE0"
INVENTORY = ROOT / "_run/current/0915_robot15h_window_start_inventory_v1/attempts/attempt_0001"
HAWOR_ROOT = ROOT / "_run/current/0915_robot15h_hawor_wave0_recovery_v1/attempts/attempt_0001"
OUTPUT = ROOT / f"_run/current/{TASK_ID}/attempts/attempt_0001"
VISUAL = ROOT / "docs/current/visuals/0915_ROBOT15H_W0_KAI22_R0_V1"
TERMINAL_RECEIPT = ROOT / "tasks/receipts/0915_ROBOT15H_KAI22_R0_WAVE0_V1_RESULT.json"
HANDS = ("left", "right")
MANO_JOINT_NAMES = (
    "wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
)
HUMAN_TO_PHYSICAL = np.asarray((1, 0), dtype=np.int64)
AUTHORITY = "DEVELOPMENT_RELATIVE_NON_CONTROL_NON_DEPLOYABLE"

BOUNDED_THRESHOLDS = {
    "per_side_reprojection_p95_px_max": 12.0,
    "per_side_observed_fraction_drop_max": 0.001,
    "identity_switch_count_max": 0,
    "root_and_pose_rotation_orthogonality_max": 0.0001,
    "root_and_pose_rotation_determinant_min_exclusive": 0.0,
    "bone_length_cv_must_not_regress": True,
    "wrist_step_p95_must_improve_over_raw": True,
    "all_joint_acceleration_p95_must_improve_over_raw": True,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    candidate = path.resolve(strict=True)
    return {"path": str(candidate), "bytes": candidate.stat().st_size, "sha256": sha256(candidate)}


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


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def process_start_ticks(pid: int) -> int:
    return int(Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()[21])


def validate_route() -> tuple[dict[str, Any], Path]:
    packet_path = ROOT / f"tasks/current/{TASK_ID}/TASK_PACKET.json"
    packet = load_json(packet_path)
    if packet != build_packet(TASK_ID) or packet.get("weights") != "ABSENT":
        raise RuntimeError("current Kai22 R0 packet differs from frozen CPU spec")
    state = load_json(ROOT / "docs/governance/LONG_HORIZON_TASK_STATE.json")
    if state.get("next_task", {}).get("task_id") != TASK_ID:
        raise RuntimeError("Kai22 R0 W0 is not current next_task")
    index = load_json(ROOT / "tasks/current/INDEX.json")
    route = next((row for row in index.get("task_packets", []) if row.get("task_id") == TASK_ID), None)
    if route is None or route.get("execution_allowed") is not True:
        raise RuntimeError("Kai22 R0 W0 is not routable")
    if route.get("packet_sha256") != sha256(packet_path):
        raise RuntimeError("Kai22 R0 packet SHA differs from current route")
    return packet, packet_path


def heartbeat(status: str = "RUNNING") -> None:
    completed = subprocess.run(
        [
            sys.executable, "-m", "chaoyang.governance.heartbeat_task",
            "--task-id", TASK_ID, "--pid", str(os.getpid()),
            "--status", status, "--phase", PHASE,
        ],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if completed.returncode:
        raise RuntimeError("governance heartbeat failed: " + (completed.stderr or completed.stdout)[-4000:])


def session_timestamps(source_root: Path, expected_frames: int) -> tuple[np.ndarray, np.ndarray, str]:
    paths = sorted((source_root / "preprocess/all_data").glob("*/training_data.json"))
    if len(paths) != expected_frames:
        raise RuntimeError("processed timestamp axis differs from HaWoR frame axis")
    timestamp_ns: list[int] = []
    relative_s: list[float] = []
    for path in paths:
        metadata = load_json(path).get("metadata", {})
        timestamp_ns.append(int(metadata["ts"]))
        relative_s.append(float(metadata["video_time_s"]))
    ns = np.asarray(timestamp_ns, np.int64)
    seconds = np.asarray(relative_s, np.float64)
    if not np.isfinite(seconds).all() or np.any(np.diff(seconds) <= 0) or np.any(np.diff(ns) <= 0):
        raise RuntimeError("session timestamps are not strictly increasing")
    return ns, seconds, "PROCESSED_TRAINING_DATA_METADATA_TS_AND_VIDEO_TIME_S"


def r0_from_hawor(
    hawor: Mapping[str, np.ndarray], timestamps_s: np.ndarray, assets: Any,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    observed = np.asarray(hawor["observed"], bool)
    joints = np.asarray(hawor["joints_3d_camera"], np.float64)
    if joints.ndim != 4 or joints.shape[0] != 2 or joints.shape[2:] != (21, 3):
        raise RuntimeError("HaWoR MANO21 camera geometry shape mismatch")
    frame_count = joints.shape[1]
    if observed.shape != (2, frame_count) or timestamps_s.shape != (frame_count,):
        raise RuntimeError("HaWoR/timestamp axis mismatch")
    if tuple(str(value) for value in np.asarray(hawor["mano_joint_names"]).tolist()) != MANO_JOINT_NAMES:
        raise RuntimeError("HaWoR MANO21 identity mismatch")
    if tuple(str(value) for value in np.asarray(hawor["anatomical_side_names"]).tolist()) != HANDS:
        raise RuntimeError("HaWoR anatomical side identity mismatch")
    if not np.array_equal(np.asarray(hawor["original_frame_indices"]), np.arange(frame_count)):
        raise RuntimeError("HaWoR original frame axis mismatch")
    limits = hand_limits(assets)
    palms = np.full((2, frame_count, 4, 4), np.nan, np.float64)
    palm_failures = np.zeros((2, frame_count), bool)
    for human_side in range(2):
        for frame in np.flatnonzero(observed[human_side] & np.isfinite(joints[human_side]).all(axis=(1, 2))):
            try:
                rotation = wrist_adapter.final_v3_mano_palm_basis(joints[human_side, frame], handedness=HANDS[human_side])
                palms[human_side, frame] = np.eye(4)
                palms[human_side, frame, :3, :3] = rotation
                palms[human_side, frame, :3, 3] = joints[human_side, frame, 0]
            except (ValueError, np.linalg.LinAlgError):
                palm_failures[human_side, frame] = True
    q22 = np.full((frame_count, 2, 22), np.nan, np.float64)
    relative_wrist = np.full((frame_count, 2, 4, 4), np.nan, np.float64)
    losses = np.full((frame_count, 2), np.nan, np.float64)
    valid = np.zeros((frame_count, 2), bool)
    smoke: dict[str, Any] = {}
    joint_order: dict[str, list[str]] = {}
    limit_rows: dict[str, dict[str, list[float]]] = {}
    source_side_status: dict[str, Any] = {}
    for human_side, human_hand in enumerate(HANDS):
        physical_side = int(HUMAN_TO_PHYSICAL[human_side])
        physical_hand = HANDS[physical_side]
        model = assets.left_hand if physical_side == 0 else assets.right_hand
        moving = tuple(joint for joint in model.joints if joint.joint_type != "fixed")
        joint_order[physical_hand] = [joint.name for joint in moving]
        limit_rows[physical_hand] = {
            "lower_rad": limits[physical_side].lower.tolist(),
            "upper_rad": limits[physical_side].upper.tolist(),
        }
        ids = np.flatnonzero(
            observed[human_side]
            & np.isfinite(joints[human_side]).all(axis=(1, 2))
            & np.isfinite(palms[human_side]).all(axis=(1, 2))
        )
        source_side_status[human_hand] = {
            "physical_kaihand": physical_hand,
            "observed_frames": int(observed[human_side].sum()),
            "retargetable_frames": int(ids.size),
            "palm_basis_failures": int(palm_failures[human_side].sum()),
        }
        if ids.size == 0:
            smoke[physical_hand] = {
                "status": "NO_DIRECT_OBSERVED_FRAME",
                "frame_id": None,
                "finite_link_transforms": None,
                "link_count": len(model.links),
                "moving_joint_names": joint_order[physical_hand],
            }
            continue
        anchor = palms[human_side, int(ids[0])]
        inverse_anchor = np.linalg.inv(anchor)
        for frame in ids:
            q22[frame, physical_side], losses[frame, physical_side] = retarget_kaihand_frame(
                joints[human_side, frame], limits[physical_side]
            )
            relative_wrist[frame, physical_side] = inverse_anchor @ palms[human_side, frame]
            valid[frame, physical_side] = True
        frame = int(ids[len(ids) // 2])
        transforms = forward_kinematics(
            model,
            {joint.name: float(q22[frame, physical_side, index]) for index, joint in enumerate(moving)},
        )
        smoke[physical_hand] = {
            "status": "PASS",
            "frame_id": frame,
            "finite_link_transforms": all(np.isfinite(value).all() for value in transforms.values()),
            "link_count": len(transforms),
            "moving_joint_names": joint_order[physical_hand],
        }
    source_observed = int(observed.sum())
    valid_count = int(valid.sum())
    result = {
        "schema_version": "KAI22_R0_BASELINE_WAVE0_V1",
        "status": "COMPLETED_DEVELOPMENT_BASELINE" if valid_count else "REJECTED_QUALITY_NO_DIRECT_HAND",
        "authority": AUTHORITY,
        "direct_observed_only": True,
        "short_gap_inferred_consumed": False,
        "valid_side_frame_count": valid_count,
        "source_observed_side_frame_count": source_observed,
        "coverage_preserved_exactly": valid_count == source_observed,
        "q22_joint_order": joint_order,
        "q22_limits": limit_rows,
        "q22_units": "radians",
        "side_contract": {
            "source_axis": ["anatomical_left", "anatomical_right"],
            "physical_axis": ["kaihand_left", "kaihand_right"],
            "human_to_physical": {"left": "right", "right": "left"},
            "human_to_physical_index": HUMAN_TO_PHYSICAL.tolist(),
            "image_x_order_inference": False,
        },
        "source_side_status": source_side_status,
        "renderer_fk_smoke_test": smoke,
        "retarget_loss_mean": float(np.nanmean(losses)) if np.isfinite(losses).any() else None,
        "timestamp_motion": {
            hand: timestamp_motion_diagnostics(q22[:, side], timestamps_s, valid[:, side])
            for side, hand in enumerate(HANDS)
        },
        "relative_wrist_semantics": "FIRST_DIRECT_OBSERVED_WRIST_ANCHOR_PER_SIDE_CAMERA_RELATIVE_PRIOR",
        "frame_definition": "HAWOR_PHYSICAL_LEFT_CAMERA_RELATIVE_PER_SIDE_ANCHOR",
        "arm_ik_performed": False,
        "control_ground_truth": False,
        "training_eligible": False,
        "physical_deployment_authorized": False,
        "calibration_authority": "DEVELOPMENT_ONLY",
    }
    arrays = {
        "q22_init": q22,
        "relative_wrist_T": relative_wrist,
        "source_observed_valid_anatomical": observed.T,
        "valid_side_frame": valid,
        "applied_refinement_mask": np.zeros_like(valid),
        "timestamps_s": timestamps_s,
        "frame_id": np.arange(frame_count, dtype=np.int64),
        "human_to_physical": HUMAN_TO_PHYSICAL,
    }
    return result, arrays, {"losses": losses, "joint_order": joint_order}


def _draw_fk_hand(panel: np.ndarray, model: Any, q: np.ndarray, center: tuple[int, int], color: tuple[int, int, int]) -> None:
    moving = tuple(joint for joint in model.joints if joint.joint_type != "fixed")
    transforms = forward_kinematics(model, {joint.name: float(q[index]) for index, joint in enumerate(moving)})
    positions = {name: value[:3, 3] for name, value in transforms.items()}
    origin = positions[model.root_link]
    scale = 1350.0
    projected = {
        name: np.asarray([center[0] + scale * (point[0] - origin[0]), center[1] - scale * (point[1] - origin[1])], np.int32)
        for name, point in positions.items()
    }
    for joint in model.joints:
        cv2.line(panel, tuple(projected[joint.parent]), tuple(projected[joint.child]), color, 3, cv2.LINE_AA)
    for point in projected.values():
        cv2.circle(panel, tuple(point), 3, (235, 235, 235), -1, cv2.LINE_AA)


def render_r0_review(
    video: Path, destination: Path, session_id: str, arrays: Mapping[str, np.ndarray], assets: Any,
) -> dict[str, Any]:
    q22 = np.asarray(arrays["q22_init"])
    valid = np.asarray(arrays["valid_side_frame"], bool)
    frame_count = q22.shape[0]
    capture = cv2.VideoCapture(str(video))
    encoder = open_encoder(destination, 1280, 480, 30.0)
    assert encoder.stdin is not None
    frames = 0
    try:
        while True:
            ok, raw = capture.read()
            if not ok:
                break
            rgb = cv2.resize(raw, (640, 480), interpolation=cv2.INTER_AREA)
            panel = np.full((480, 640, 3), 26, np.uint8)
            cv2.putText(panel, "Kai22 R0 | pinned URDF FK", (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 245, 245), 2, cv2.LINE_AA)
            for side, hand in enumerate(HANDS):
                if valid[frames, side]:
                    model = assets.left_hand if side == 0 else assets.right_hand
                    _draw_fk_hand(panel, model, q22[frames, side], (175 + side * 300, 265), ((255, 150, 50), (50, 120, 255))[side])
                    state = "DIRECT OBSERVED"
                else:
                    state = "INVALID / NO FILL"
                cv2.putText(panel, f"{hand}: {state}", (45 + side * 300, 445), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
            cv2.putText(rgb, f"{session_id} | physical-left resize-only | frame {frames:04d}", (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(panel, "wrist-local view | NON_CONTROL / NON_DEPLOYABLE", (120, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 220, 255), 1, cv2.LINE_AA)
            encoder.stdin.write(np.hstack((rgb, panel)).tobytes())
            frames += 1
    finally:
        capture.release()
        encoder.stdin.close()
    if encoder.wait() != 0 or frames != frame_count:
        raise RuntimeError(f"Kai22 R0 review render failed: {session_id}")
    return {"video": ref(destination), "decode": full_decode(destination, frame_count)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--visual-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--executor-epoch", type=int, required=True)
    parser.add_argument("--fencing-token", required=True)
    args = parser.parse_args()
    packet, packet_path = validate_route()
    output, visual, receipt = args.output_root.resolve(), args.visual_root.resolve(), args.receipt.resolve()
    if output != OUTPUT.resolve() or visual != VISUAL.resolve() or receipt != TERMINAL_RECEIPT.resolve():
        raise RuntimeError("fixed Kai22 R0 W0 output namespace mismatch")
    if args.executor_epoch < 1 or len(args.fencing_token) < 16:
        raise RuntimeError("invalid executor epoch/fencing token")
    for path in (output, visual, receipt):
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"fresh output required: {path}")
    output.mkdir(parents=True)
    heartbeat("CLAIMED")

    batch_path = INVENTORY / "BATCH_MANIFEST.json"
    hawor_batch_path = HAWOR_ROOT / "hawor/BATCH_RESULT.json"
    batch = load_json(batch_path)
    hawor_batch = load_json(hawor_batch_path)
    selected = [row for row in batch["sessions"] if row.get("wave") == "W0"]
    if len(selected) != 4 or hawor_batch.get("session_count") != 4 or hawor_batch.get("failed_runtime") != 0:
        raise RuntimeError("four-session W0 HaWoR terminal is not closed")
    hawor_by_id = {row["session_id"]: row for row in hawor_batch["results"]}
    prepared_by_id = {
        row["session_id"]: row
        for row in load_json(HAWOR_ROOT / "PREPARED_MANIFEST.json")["results"]
    }
    signature_payload = {
        "schema_version": "0915-robot15h-kai22-r0-wave0-signature-v1",
        "task_id": TASK_ID,
        "window_run_id": WINDOW_RUN_ID,
        "executor_epoch": args.executor_epoch,
        "weights": "ABSENT",
        "task_packet": ref(packet_path),
        "inputs": [ref(batch_path), ref(hawor_batch_path), ref(HAWOR_ROOT / "PREPARED_MANIFEST.json")],
        "code": [ref(Path(__file__)), ref(ROOT / "src/chaoyang/ops/run_hawor_bounded_parameter_successor.py"), ref(ROOT / "src/chaoyang/pipeline/robot_visual_relative_v1.py")],
        "model_policy": "FROZEN_HAWOR_OUTPUT_NO_MODEL_RERUN",
        "source_policy": "READ_ONLY",
    }
    signature_sha = canonical_sha(signature_payload)
    atomic_json(output / "RUN_SIGNATURE.json", {**signature_payload, "run_signature_sha256": signature_sha})
    atomic_json(output / "CLAIM.json", {
        "schema_version": "0915-robot15h-kai22-r0-wave0-claim-v1",
        "task_id": TASK_ID,
        "window_run_id": WINDOW_RUN_ID,
        "attempt_id": output.name,
        "status": "CLAIMED",
        "weights": "ABSENT",
        "pid": os.getpid(),
        "proc_start_ticks": process_start_ticks(os.getpid()),
        "executor_epoch": args.executor_epoch,
        "fencing_token_sha256": hashlib.sha256(args.fencing_token.encode()).hexdigest(),
        "run_signature_sha256": signature_sha,
        "unique_write_root": str(output),
        "task_packet": ref(packet_path),
    })

    canaries = []
    for row in selected:
        hawor = hawor_by_id[row["session_id"]]
        prepared = prepared_by_id[row["session_id"]]
        canaries.append({
            "task": row["task"],
            "session_id": row["session_id"],
            "frame_count": row["frame_count"],
            "fps": 30,
            "raw_hawor_npz": hawor["npz"]["path"],
            "raw_hawor_sha256": hawor["npz"]["sha256"],
            "source_video": prepared["prepared_video"]["path"],
            "source_video_sha256": prepared["prepared_video"]["sha256"],
        })
    contract = {
        "schema_version": "hawor-bounded-parameter-wave0-contract-v1",
        "status": "FROZEN_FOUR_SESSION_W0",
        "method": "SINGLE_TRACK_CONFIDENCE_BOUNDED_PARAMETER_FIT_NO_WINDOW_GAUGE",
        "window_run_id": WINDOW_RUN_ID,
        "canaries": canaries,
        "acceptance_thresholds": BOUNDED_THRESHOLDS,
        "policy": {
            "input_domain": "PHYSICAL_LEFT_SOURCEINDEX1_CROP_RESIZE_ONLY",
            "missing": "PRESERVE_NAN_NO_INTERPOLATION",
            "human_gate_required": True,
            "robot_consumption": "NUMERIC_PASS_ONLY_ELSE_RAW_DIRECT_OBSERVED_FALLBACK",
            "control_ground_truth": False,
        },
    }
    atomic_json(output / "BOUNDED_V2_CONTRACT.json", contract)
    bounded_root = output / "bounded_v2"
    bounded_command = [
        str(ROOT / "src/chaoyang/ops/hawor_python.sh"),
        str(ROOT / "src/chaoyang/ops/run_hawor_bounded_parameter_successor.py"),
        "--contract", str(output / "BOUNDED_V2_CONTRACT.json"),
        "--output-root", str(bounded_root),
    ]
    started = time.time()
    with (output / "BOUNDED_V2.log").open("w", encoding="utf-8") as log:
        bounded_process = subprocess.Popen(bounded_command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
        while bounded_process.poll() is None:
            time.sleep(30)
            heartbeat()
    if not (bounded_root / "RESULT.json").is_file():
        raise RuntimeError(f"bounded v2 failed without terminal result: returncode={bounded_process.returncode}")
    bounded_batch = load_json(bounded_root / "RESULT.json")
    bounded_rows = {row["session_id"]: row for row in bounded_batch["canaries"]}

    assets = load_pinned_robot_assets(ROOT)
    visual.mkdir(parents=True)
    results: list[dict[str, Any]] = []
    for row in selected:
        session_id = row["session_id"]
        raw_hawor = Path(str(hawor_by_id[session_id]["npz"]["path"])).resolve(strict=True)
        bounded_summary = bounded_rows.get(session_id, {"status": "HOLD_EXCEPTION"})
        bounded_session_result = bounded_root / session_id / "RESULT.json"
        if bounded_summary.get("status") == "PASS_NUMERIC_NEEDS_HUMAN_REVIEW":
            selected_hawor = bounded_root / session_id / "HAWOR_BOUNDED_PARAMETER_SUCCESSOR.npz"
            human_prior = "HAWOR_BOUNDED_V2_NUMERIC_PASS_HUMAN_REVIEW_PENDING"
        else:
            selected_hawor = raw_hawor
            human_prior = "RAW_DIRECT_OBSERVED_FALLBACK_BOUNDED_NOT_ADMITTED"
        with np.load(selected_hawor, allow_pickle=False) as archive:
            hawor = {key: np.asarray(archive[key]) for key in archive.files}
        source_root = Path(str(row["source_stereo"]["path"])).parent.parent
        timestamp_ns, timestamps_s, timestamp_clock = session_timestamps(source_root, int(row["frame_count"]))
        r0, arrays, _ = r0_from_hawor(hawor, timestamps_s, assets)
        arrays["timestamp_ns"] = timestamp_ns
        arrays["evidence_type"] = np.where(arrays["valid_side_frame"], "OBSERVED", "UNKNOWN").astype("U12")
        target = output / "sessions" / row["task"] / session_id
        staging = target.with_name(f".{target.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
        staging.mkdir(parents=True)
        npz_path = staging / "KAI22_R0_BASELINE_V1.npz"
        atomic_npz(npz_path, **arrays)
        r0.update({
            "session_id": session_id,
            "source_group": row["source_group"],
            "task": row["task"],
            "frame_count": int(row["frame_count"]),
            "timestamp_clock": timestamp_clock,
            "human_prior_selected": human_prior,
            "human_prior_numeric_status": bounded_summary.get("status"),
            "human_full_video_review": "PENDING",
            "r0_exported": True,
            "r0_quality_admitted": (
                hawor_by_id[session_id]["status"] == "PASS_DEVELOPMENT_HAWOR"
                and r0["coverage_preserved_exactly"]
            ),
            "source_hawor": ref(selected_hawor),
            "arrays": {"path": str((target / npz_path.name).resolve()), "bytes": npz_path.stat().st_size, "sha256": sha256(npz_path)},
        })
        if bounded_session_result.is_file():
            r0["bounded_v2_result"] = ref(bounded_session_result)
        atomic_json(staging / "RESULT.json", r0)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, target)
        review = render_r0_review(
            Path(str(prepared_by_id[session_id]["prepared_video"]["path"])),
            visual / f"{session_id}_KAI22_R0_REVIEW.mp4",
            session_id, arrays, assets,
        )
        result = {
            "session_id": session_id,
            "task": row["task"],
            "source_group": row["source_group"],
            "status": r0["status"],
            "r0_exported": True,
            "r0_quality_admitted": r0["r0_quality_admitted"],
            "valid_side_frames": r0["valid_side_frame_count"],
            "human_session_strict_gate": hawor_by_id[session_id]["status"],
            "human_prior_selected": human_prior,
            "result": ref(target / "RESULT.json"),
            "review": review,
        }
        results.append(result)
        heartbeat()

    r0_exported = sum(row["r0_exported"] for row in results)
    r0_success = sum(row["r0_quality_admitted"] for row in results)
    batch_result = {
        "schema_version": "0915-robot15h-kai22-r0-wave0-batch-v1",
        "status": "COMPLETED_ALL_TERMINAL",
        "window_run_id": WINDOW_RUN_ID,
        "session_count": len(results),
        "r0_exported": r0_exported,
        "r0_success": r0_success,
        "r0_rejected": len(results) - r0_success,
        "failed_runtime": 0,
        "valid_side_frames": sum(row["valid_side_frames"] for row in results),
        "bounded_numeric_pass": sum(row["human_prior_selected"].startswith("HAWOR_BOUNDED") for row in results),
        "bounded_raw_fallback": sum(row["human_prior_selected"].startswith("RAW_DIRECT") for row in results),
        "results": results,
        "wall_seconds": time.time() - started,
        "control_ground_truth": False,
        "training_eligible": False,
        "physical_deployment_authorized": False,
    }
    atomic_json(output / "BATCH_RESULT.json", batch_result)
    readme = "# 0915 Robot15h W0 Kai22 R0 审阅\n\n"
    readme += "原 RGB 使用物理左目 `sourceIndex=1 + crop + resize-only`；右栏是 pinned Kai22 URDF 的 wrist-local FK，不是相机叠加、真机控制或物理部署结果。\n\n"
    for row in results:
        name = Path(row["review"]["video"]["path"]).name
        readme += f"- [{row['session_id']}]({name})：已导出 `{row['r0_exported']}`，质量准入 `{row['r0_quality_admitted']}`，有效 side-frame `{row['valid_side_frames']}`，prior `{row['human_prior_selected']}`。\n"
    readme += "\n4 个 HaWoR 会话的严格双手门均拒绝，因此 R0 可审计导出不计作质量成功；缺失手没有补帧。人工视觉验收仍为 PENDING。\n"
    (visual / "README_ZH.md").write_text(readme, encoding="utf-8")
    result = {
        "schema_version": "0915-robot15h-kai22-r0-wave0-result-v1",
        "task_id": TASK_ID,
        "window_run_id": WINDOW_RUN_ID,
        "status": "PASSED",
        "weights": "ABSENT",
        "batch_terminal": "COMPLETED_ALL_TERMINAL",
        "counts": {
            "attempted": len(results), "r0_success": r0_success,
            "r0_exported": r0_exported,
            "r0_rejected": len(results) - r0_success, "failed_runtime": 0,
            "valid_side_frames": batch_result["valid_side_frames"],
        },
        "bounded_v2_returncode": bounded_process.returncode,
        "bounded_v2_result": ref(bounded_root / "RESULT.json"),
        "batch_result": ref(output / "BATCH_RESULT.json"),
        "visual_readme": ref(visual / "README_ZH.md"),
        "source_mutated": False,
        "processed_mutated": False,
        "control_ground_truth": False,
        "training_eligible": False,
        "physical_deployment_authorized": False,
        "user_visual_acceptance": "PENDING",
        "claim_limit": packet["claim_limit"],
    }
    atomic_json(output / "RESULT.json", result)
    atomic_json(receipt, {**result, "result": ref(output / "RESULT.json")})
    atomic_json(output / "RUN_RECEIPT.json", {
        "schema_version": "0915-robot15h-kai22-r0-wave0-run-receipt-v1",
        "task_id": TASK_ID, "window_run_id": WINDOW_RUN_ID, "status": "PASSED",
        "result": ref(output / "RESULT.json"), "terminal_receipt": ref(receipt),
    })
    heartbeat()
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
