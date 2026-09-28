"""Execute the finite shared local-hand/Data lane for the 2026-09-24 task."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from chaoyang.governance.common import REPO_ROOT, artifact_ref, atomic_json, load_json
from chaoyang.ops.run_huro_hand_frame_v2 import ExplicitSelfCollision
from chaoyang.pipeline.huro_hand_only_retarget_v1 import load_hand_model
from chaoyang.pipeline.kai22_full_fk_sidecar_v1 import load_pinned_kaihand_models
from chaoyang.pipeline.robot_renderer_eevee_fullchain import load_pinned_robot_assets
from chaoyang.pipeline.shared_local_hand_target_v1 import (
    HAWOR21_NAMES,
    METHOD_ID,
    anatomical_keypoints_from_q,
    build_local_target,
    enumerate_h50,
    evaluate_local_quality,
    palm_width,
    solve_local_frame,
    solve_local_sequence,
)


TASK = "human_to_robot_shared_hand_delivery_20260924"
ATTEMPT = REPO_ROOT / f"_run/current/{TASK}/attempts/attempt_0001"
OUT = ATTEMPT / "lanes/hand_data"
STATE = OUT / "STATE.json"
PACKET = REPO_ROOT / f"tasks/current/{TASK}/TASK_PACKET.json"
SENSOR_HAND_ROOT = REPO_ROOT / (
    "_run/current/four_stream_visual_delivery_v5/attempts/attempt_0001/lanes/"
    "sensor/run_0001"
)
POKER_HAWOR = REPO_ROOT / (
    "_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/lanes/"
    "ai2/exact_bounded_camera_source_v3/play_cards_0902_042/"
    "BOUNDED_CAMERA_SOURCE_V3.npz"
)
PRIOR_SENSOR = REPO_ROOT / (
    "_run/current/human_to_robot_quality_acceptance_20260924/attempts/"
    "attempt_0001/lanes/sensor"
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as bundle:
        return {key: bundle[key] for key in bundle.files}


def _process_start_ticks(pid: int) -> int:
    return int(Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()[19])


def _summary(values: np.ndarray) -> dict:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return {
        "count": int(finite.size),
        "p50": float(np.median(finite)) if finite.size else None,
        "p95": float(np.percentile(finite, 95)) if finite.size else None,
        "max": float(np.max(finite)) if finite.size else None,
    }


def _claim_state(writer: dict) -> None:
    state = load_json(STATE)
    if state["task_id"] != TASK or state["lane"] != "hand_data":
        raise RuntimeError("HAND_DATA_STATE_IDENTITY_DRIFT")
    repair2 = (
        state.get("execution") == "EXECUTED"
        and state.get("quality") == "REJECTED_QUALITY"
        and not (OUT / "RESULT_REPAIR2.json").exists()
    )
    if state["execution"] not in {"NOT_STARTED", "FAILED_RUNTIME"} and not repair2:
        raise RuntimeError(f"HAND_DATA_STATE_NOT_CLAIMABLE:{state['execution']}")
    old_writer = state.get("writer")
    if state["execution"] == "FAILED_RUNTIME" and isinstance(old_writer, dict):
        old_pid = old_writer.get("pid")
        old_ticks = old_writer.get("proc_start_ticks")
        stat = Path(f"/proc/{old_pid}/stat") if isinstance(old_pid, int) else None
        if stat is not None and stat.exists():
            actual = int(stat.read_text().split(") ", 1)[1].split()[19])
            if actual == old_ticks:
                raise RuntimeError("PREVIOUS_HAND_DATA_WRITER_STILL_ALIVE")
    state.update({
        "status": "RUNNING",
        "execution": "RUNNING",
        "writer": writer,
        "dependencies": [artifact_ref(PACKET)],
        "implementation_repair": 2 if repair2 else state.get("implementation_repair", 0),
        "next_action": "RUN_REPAIR2_SYNTHETIC_AND_SENSOR097_SHARED_LOCAL_TARGET",
        "updated_at": _now(),
    })
    atomic_json(STATE, state)


def _update_state(writer: dict, **changes) -> None:
    state = load_json(STATE)
    if state.get("writer") != writer:
        raise RuntimeError("HAND_DATA_WRITER_FENCE_CHANGED")
    state.update(changes)
    state["updated_at"] = _now()
    atomic_json(STATE, state)


def _synthetic_roundtrip(hands) -> dict:
    rows = []
    for physical, hand in enumerate(hands):
        q = 0.5 * (hand.lower + hand.upper)
        q += 0.04 * (hand.upper - hand.lower) * np.sin(np.arange(22) + physical)
        points = anatomical_keypoints_from_q(hand, q)
        target = build_local_target(
            points,
            points,
            source_reference_width_m=palm_width(points),
            robot_reference_width_m=palm_width(points),
        )
        solved, fk, diagnostic = solve_local_frame(hand, target, previous_q=q)
        rows.append({
            "physical_side": hand.side,
            "solver_success": diagnostic.success,
            "max_direction_angle_deg": diagnostic.max_direction_angle_deg,
            "normalized_pinch_error": diagnostic.normalized_pinch_error,
            "q_max_abs_rad": float(np.max(np.abs(solved - q))),
            "fk_max_abs_m": float(np.max(np.abs(fk - points))),
        })
    passed = all(
        row["solver_success"]
        and row["max_direction_angle_deg"] <= 0.005
        and row["normalized_pinch_error"] <= 5e-5
        and row["fk_max_abs_m"] <= 5e-5
        for row in rows
    )
    return {
        "schema_version": "SHARED_LOCAL_HAND_SYNTHETIC_ROUNDTRIP_V1",
        "task_id": TASK,
        "method_id": METHOD_ID,
        "rows": rows,
        "structure": "PASS" if passed else "FAIL",
        "claim_limit": "same-robot deterministic FK target round trip only",
    }


def _session_input(short: str) -> tuple[Path, dict[str, np.ndarray]]:
    path = SENSOR_HAND_ROOT / f"play_cards_0916_{short}/HAND_MOTION_V1.npz"
    data = _load_npz(path)
    expected = int(short == "097") * 165 + int(short == "098") * 179 + int(short == "101") * 122
    if expected == 0 or data["manus_local_25_m"].shape != (expected, 2, 25, 3):
        raise RuntimeError(f"SENSOR_{short}_SOURCE_SHAPE_DRIFT")
    if tuple(data["anatomical_side_names"].tolist()) != ("left", "right"):
        raise RuntimeError(f"SENSOR_{short}_SIDE_AXIS_DRIFT")
    return path, data


def _session_run(short: str, hands, collision) -> dict:
    source_path, source = _session_input(short)
    session_id = f"play_cards_0916_{short}"
    result_path = OUT / f"{session_id}_SHARED_LOCAL_REPAIR2_RESULT.json"
    states_path = OUT / f"{session_id}_SHARED_LOCAL_REPAIR2_STATES.npz"
    if result_path.exists() or states_path.exists():
        raise FileExistsError(f"HAND_DATA_SESSION_OUTPUT_EXISTS:{session_id}")
    states = solve_local_sequence(
        hands,
        np.asarray(source["manus_local_25_m"]).transpose(1, 0, 2, 3),
        np.asarray(source["manus_hand_valid"]).T,
        np.asarray(source["frame_id"]),
        source_kind="MANUS25",
        joint_names=tuple(source["manus25_joint_names"].tolist()),
        anatomical_side_names=tuple(source["anatomical_side_names"].tolist()),
        units="m",
        max_evaluations=80,
    )
    quality = evaluate_local_quality(hands, states, collision_checker=collision)
    states.update(quality)
    states["timestamp_ns"] = np.asarray(source["timestamp_ns"])
    states["motion_source"] = np.asarray("controller_manus")
    np.savez_compressed(states_path, **states)
    anchors, sides = enumerate_h50(
        quality["numeric_local_gate_pass"],
        source["frame_id"],
        source["timestamp_ns"],
    )
    result = {
        "schema_version": "SHARED_LOCAL_HAND_SENSOR_RESULT_V1",
        "task_id": TASK,
        "session_id": session_id,
        "method_id": METHOD_ID,
        "source": artifact_ref(source_path),
        "states": artifact_ref(states_path),
        "frames": int(len(source["frame_id"])),
        "source_observed_side_frames": int(source["manus_hand_valid"].sum()),
        "solver_success_side_frames": int(states["solver_success"].sum()),
        "complete_target_side_frames": int(quality["complete_target"].sum()),
        "direction_pass_side_frames": int(quality["direction_pass"].sum()),
        "pinch_pass_side_frames": int(quality["pinch_pass"].sum()),
        "limit_pass_side_frames": int(quality["limit_pass"].sum()),
        "collision_known_side_frames": int(quality["collision_known"].sum()),
        "collision_pass_side_frames": int(quality["collision_pass"].sum()),
        "numeric_local_gate_pass_side_frames": int(quality["numeric_local_gate_pass"].sum()),
        "numeric_local_gate_pass_left_right": quality["numeric_local_gate_pass"].sum(axis=0).tolist(),
        "numeric_candidate_h50_windows": int(len(anchors)),
        "numeric_candidate_h50_windows_left_right": [
            int(np.count_nonzero(sides == side)) for side in range(2)
        ],
        "first_h50_anchor_and_physical_side": (
            [int(anchors[0]), int(sides[0])] if len(anchors) else None
        ),
        "max_direction_angle_deg": _summary(states["max_direction_angle_deg"]),
        "normalized_pinch_error": _summary(states["normalized_pinch_error"]),
        "independent_fk_max_abs_m": _summary(quality["independent_fk_max_abs_m"]),
        "collision_scope": "PINNED_KAIHAND_NON_ADJACENT_SELF_COLLISION_ONLY",
        "visual_review": "NOT_EVALUATED_BY_THIS_NUMERIC_PRODUCER",
        "label_quality_contract": {
            "direction_deg_max": 15.0,
            "normalized_pinch_error_max": 0.1,
            "requires_finite_q_fk_limits_and_known_zero_declared_collision": True,
            "human_robot_bone_length_equality_required": False,
        },
        "interface_pass": False,
        "label_quality_pass": False,
        "consumer_qualification": (
            "NUMERIC_LOCAL_Q22_CANDIDATE_PENDING_VISUAL_REVIEW"
            if quality["numeric_local_gate_pass"].any()
            else "NO_NUMERIC_LOCAL_Q22_LABEL_FRAMES"
        ),
        "training_eligible": False,
        "optimizer_steps": 0,
        "control_ground_truth": False,
        "physical_deployable": False,
    }
    atomic_json(result_path, result)
    return result


def _poker_run(hands, collision) -> dict:
    raw = _load_npz(POKER_HAWOR)
    if tuple(raw["mano_joint_names"].tolist()) != HAWOR21_NAMES:
        raise RuntimeError("POKER_HAWOR21_NAME_ORDER_DRIFT")
    states = solve_local_sequence(
        hands,
        raw["joints_3d_camera"],
        raw["observed"],
        raw["original_frame_indices"],
        source_kind="HAWOR21",
        joint_names=tuple(raw["mano_joint_names"].tolist()),
        anatomical_side_names=tuple(raw["anatomical_side_names"].tolist()),
        units="m",
        max_evaluations=80,
    )
    quality = evaluate_local_quality(hands, states, collision_checker=collision)
    states.update(quality)
    states["timestamp_ns"] = raw["timestamp_ns"]
    path = OUT / "play_cards_0902_042_SHARED_LOCAL_HAND_REPAIR2_STATES.npz"
    np.savez_compressed(path, **states)
    receipt = OUT / "play_cards_0902_042_SHARED_LOCAL_HAND_REPAIR2_RESULT.json"
    result = {
        "schema_version": "SHARED_LOCAL_HAND_POKER_RESULT_V1",
        "task_id": TASK,
        "session_id": "play_cards_0902_042",
        "source": artifact_ref(POKER_HAWOR),
        "states": artifact_ref(path),
        "solver_success_side_frames": int(states["solver_success"].sum()),
        "numeric_local_gate_pass_side_frames": int(quality["numeric_local_gate_pass"].sum()),
        "numeric_local_gate_pass_left_right": quality["numeric_local_gate_pass"].sum(axis=0).tolist(),
        "wrist_and_arm_changed": False,
        "visual_review": "NOT_EVALUATED_BY_THIS_NUMERIC_PRODUCER",
        "training_eligible": False,
        "control_ground_truth": False,
    }
    atomic_json(receipt, result)
    return result


def _consumer_forward(sensor_result: dict, hands) -> dict:
    """Exercise the existing partial-field FM consumer on one qualified H50."""
    import torch
    from chaoyang.human_ego.training.FlowMatchingModel import FlowMatchingModel
    from chaoyang.human_ego.training.FlowMatchingTrainer import sanitize_flow_action_targets

    states = _load_npz(Path(sensor_result["states"]["path"]))
    source_path, source = _session_input("097")
    anchors, sides = enumerate_h50(
        states["numeric_local_gate_pass"], source["frame_id"], source["timestamp_ns"]
    )
    if not len(anchors):
        raise RuntimeError("NO_QUALIFIED_097_H50_FOR_CONSUMER")
    start, physical = int(anchors[0]), int(sides[0])
    anatomical = int(np.flatnonzero(states["human_to_physical"] == physical)[0])
    field = slice(18, 40) if anatomical == 0 else slice(40, 62)
    action = np.full((50, 62), np.nan, np.float32)
    action[:, field] = states["q22"][start + 1:start + 51, physical].astype(np.float32)
    valid = np.zeros((50, 62), bool)
    valid[:, field] = True
    packet_path = OUT / "SENSOR_097_QUALIFIED_Q22_H50.npz"
    np.savez_compressed(
        packet_path,
        source_frame_id=source["frame_id"][start:start + 51],
        timestamp_ns=source["timestamp_ns"][start:start + 51],
        physical_side=np.asarray(physical),
        anatomical_side=np.asarray(anatomical),
        q_current=states["q22"][start, physical].astype(np.float32),
        action_future=action,
        action_valid=valid,
        joint_names=states["joint_names"],
        source_session_id=np.asarray("play_cards_0916_097"),
    )
    source_receipt = load_json(SENSOR_HAND_ROOT / "play_cards_0916_097/RESULT.json")
    video_path = Path(source_receipt["source_video"]["path"])
    capture = cv2.VideoCapture(str(video_path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    ok, stereo = capture.read()
    capture.release()
    if not ok or stereo.shape != (1536, 4096, 3):
        raise RuntimeError("SENSOR_097_H50_RGB_DECODE_OR_DOMAIN")
    rgb = cv2.cvtColor(
        cv2.resize(stereo[:, 2048:], (320, 240), interpolation=cv2.INTER_LINEAR),
        cv2.COLOR_BGR2RGB,
    )
    torch.manual_seed(7)
    torch.set_num_threads(2)
    raw_action = torch.from_numpy(action.copy())[None]
    mask = torch.from_numpy(valid.copy())[None]
    weights = torch.ones(1)
    safe = sanitize_flow_action_targets(raw_action, mask, weights)
    alternate = raw_action.clone()
    alternate[~mask] = 123.0
    safe_alternate = sanitize_flow_action_targets(alternate, mask, weights)
    if not torch.equal(safe, safe_alternate):
        raise RuntimeError("INVALID_Q_PAYLOAD_CHANGED_SANITIZED_TARGET")
    noise = torch.randn(safe.shape, generator=torch.Generator().manual_seed(7))
    time = torch.full((1, 1), 0.5)
    flow_state = (1 - time[:, :, None]) * noise + time[:, :, None] * safe
    model = FlowMatchingModel(
        single_hand=False,
        pred_horizon=50,
        img_size=(240, 320),
        vision_embed_dim=128,
        num_decoder_layers=2,
        num_heads=4,
        dropout=0.0,
        use_pcd_features=False,
        use_aux_obj_dynamics=False,
        use_aux_visual_foresight=False,
        use_aux_temporal_contrastive=False,
        use_region_attn=False,
        hand_action_representation="kaihand_joint_state",
    ).eval()
    x_rgb = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float()[None] / 255
    with torch.inference_mode():
        prediction = model(
            x_rgb=x_rgb,
            x_ict=torch.zeros((1, 1, 29)),
            ict_mask=torch.zeros((1, 1), dtype=torch.bool),
            x_t=flow_state,
            t=time,
            x_robot_state=torch.zeros((1, 2, 33)),
            robot_state_mask=torch.zeros((1, 2), dtype=torch.bool),
        )
        loss = model.compute_loss(
            prediction,
            {"v_target": safe - noise, "action_valid_mask": mask,
             "sample_weight": weights},
        )["loss_flow"]
    if not torch.isfinite(prediction["v_pred"]).all() or not bool(torch.isfinite(loss)):
        raise RuntimeError("QUALIFIED_Q22_FORWARD_NONFINITE")
    fk_error = []
    for offset in range(51):
        q = states["q22"][start + offset, physical]
        fk = anatomical_keypoints_from_q(hands[physical], q)
        fk -= fk[0]
        fk_error.append(float(np.max(np.abs(
            fk - states["fk21_root_relative"][start + offset, physical]
        ))))
    result = {
        "schema_version": "SHARED_LOCAL_HAND_H50_CONSUMER_V1",
        "task_id": TASK,
        "packet": artifact_ref(packet_path),
        "source_hand_motion": artifact_ref(source_path),
        "source_video": artifact_ref(video_path),
        "anchor_source_frame": int(source["frame_id"][start]),
        "future_source_frame_range": [
            int(source["frame_id"][start + 1]), int(source["frame_id"][start + 50])
        ],
        "physical_side": physical,
        "anatomical_side": anatomical,
        "effective_supervised_elements": int(mask.sum()),
        "invalid_payload_invariant": True,
        "loss": float(loss),
        "q_to_independent_fk_max_abs_m": max(fk_error),
        "interface_pass": True,
        "label_quality_pass": False,
        "optimizer_steps": 0,
        "checkpoint_loaded": False,
        "training_eligible": False,
        "claim_limit": "single-side numeric-candidate local q22 H50 interface; independent visual review is pending",
        "control_ground_truth": False,
        "physical_deployable": False,
    }
    receipt = OUT / "SENSOR_097_QUALIFIED_Q22_H50_REPAIR2_CONSUMER.json"
    atomic_json(receipt, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the finite shared local-hand/Data lane."
    )
    parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    lock_path = OUT / "WRITER.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        packet = load_json(PACKET)
        if packet["task_id"] != TASK or packet["budgets"]["hand_recipe_max"] != 1:
            raise RuntimeError("HAND_DATA_TASK_PACKET_DRIFT")
        pid = os.getpid()
        writer = {
            "pid": pid,
            "proc_start_ticks": _process_start_ticks(pid),
            "executor_epoch": int(packet["executor_epoch"]),
            "operation": "run_human_to_robot_shared_delivery_hand_data",
        }
        _claim_state(writer)
        collision = None
        try:
            assets = load_pinned_robot_assets(REPO_ROOT)
            hands = (
                load_hand_model(assets.left_hand.path, "left"),
                load_hand_model(assets.right_hand.path, "right"),
            )
            synthetic = _synthetic_roundtrip(hands)
            synthetic_path = OUT / "SYNTHETIC_ROUNDTRIP_REPAIR2.json"
            if synthetic_path.exists():
                raise FileExistsError("SYNTHETIC_REPAIR2_ALREADY_EXISTS")
            atomic_json(synthetic_path, synthetic)
            if synthetic["structure"] != "PASS":
                raise RuntimeError("SHARED_LOCAL_SYNTHETIC_ROUNDTRIP_FAILED")
            pinned, _ = load_pinned_kaihand_models(REPO_ROOT)
            collision = ExplicitSelfCollision(pinned)
            result_097 = _session_run("097", hands, collision)
            evidence = [artifact_ref(synthetic_path), artifact_ref(
                OUT / "play_cards_0916_097_SHARED_LOCAL_REPAIR2_RESULT.json"
            )]
            regressions = []
            poker = None
            consumer = None
            if result_097["numeric_local_gate_pass_side_frames"] > 0:
                for short in ("098", "101"):
                    regressions.append(_session_run(short, hands, collision))
                    evidence.append(artifact_ref(
                        OUT / f"play_cards_0916_{short}_SHARED_LOCAL_REPAIR2_RESULT.json"
                    ))
                poker = _poker_run(hands, collision)
                evidence.append(artifact_ref(
                    OUT / "play_cards_0902_042_SHARED_LOCAL_HAND_REPAIR2_RESULT.json"
                ))
                if result_097["numeric_candidate_h50_windows"] > 0:
                    consumer = _consumer_forward(result_097, hands)
                    evidence.append(artifact_ref(
                        OUT / "SENSOR_097_QUALIFIED_Q22_H50_REPAIR2_CONSUMER.json"
                    ))
            final = {
                "schema_version": "HUMAN_TO_ROBOT_SHARED_HAND_DATA_RESULT_V1",
                "task_id": TASK,
                "execution": "EXECUTED",
                "structure": "PASS",
                "quality": (
                    "PASS_DECLARED_NUMERIC_LOCAL_Q22"
                    if result_097["numeric_local_gate_pass_side_frames"] > 0
                    else "REJECTED_QUALITY"
                ),
                "improvement": "EVALUATED_AGAINST_FROZEN_NEW_CONTRACT",
                "adoption": "CANDIDATE_ONLY",
                "sensor_097": result_097,
                "sensor_regressions_executed": len(regressions),
                "poker_hand_executed": poker is not None,
                "consumer_executed": consumer is not None,
                "interface_pass": bool(consumer and consumer["interface_pass"]),
                "numeric_local_gate_pass": bool(result_097["numeric_local_gate_pass_side_frames"] > 0),
                "label_quality_pass": False,
                "numeric_candidate_h50_windows": int(result_097["numeric_candidate_h50_windows"]),
                "qualified_h50_windows": 0,
                "visual_review": "PENDING_INDEPENDENT_REVIEW",
                "training_eligible": bool(consumer and consumer["training_eligible"]),
                "optimizer_steps": 0,
                "control_ground_truth": False,
                "physical_deployable": False,
                "next_action": (
                    "INDEPENDENT_VISUAL_REVIEW_AND_DELIVERY_BINDING"
                    if result_097["numeric_local_gate_pass_side_frames"] > 0
                    else "HAND_RECIPE_EXHAUSTED_REPORT_NUMERIC_FAILURE; DO_NOT_RUN_098_101_OR_POKER"
                ),
            }
            result_path = OUT / "RESULT_REPAIR2.json"
            atomic_json(result_path, final)
            evidence.append(artifact_ref(result_path))
            _update_state(
                writer,
                status="TERMINAL",
                execution="EXECUTED",
                structure="PASS",
                quality=final["quality"],
                improvement=final["improvement"],
                adoption="NOT_ADOPTED",
                consumer_qualification=(
                    "INTERFACE_AND_NUMERIC_LABEL_PASS_PENDING_VISUAL"
                    if final["interface_pass"] and final["numeric_local_gate_pass"]
                    else "NOT_QUALIFIED"
                ),
                evidence=evidence,
                blocker=(
                    None if final["numeric_local_gate_pass"] else {
                        "code": "SHARED_LOCAL_RECIPE_FAILED_FROZEN_NUMERIC_LABEL_GATE",
                        "consumer": "offline local q22 H50",
                        "owner": "hand_data",
                        "unblock_action": "method-level reassessment after this finite task; no same-recipe retry",
                        "unaffected": ["robot", "clean", "delivery"],
                    }
                ),
                next_action=final["next_action"],
            )
            print(json.dumps({
                "status": "EXECUTED",
                "numeric_local_gate_pass_side_frames": result_097["numeric_local_gate_pass_side_frames"],
                "numeric_candidate_h50_windows": result_097["numeric_candidate_h50_windows"],
                "consumer_executed": consumer is not None,
                "result": str(result_path),
            }, ensure_ascii=False))
            return 0
        except Exception as error:
            _update_state(
                writer,
                status="TERMINAL",
                execution="FAILED_RUNTIME",
                structure="FAIL",
                quality="NOT_EVALUATED",
                improvement="NOT_EVALUATED",
                adoption="NOT_ADOPTED",
                consumer_qualification="NOT_EVALUATED",
                blocker={
                    "code": type(error).__name__,
                    "detail": str(error),
                    "consumer": "hand_data",
                    "owner": "hand_data",
                    "unblock_action": "inspect the preserved partial outputs; do not overwrite or retry under this attempt",
                    "unaffected": ["robot", "clean", "delivery"],
                },
                next_action="TERMINAL_RUNTIME_FAILURE_REQUIRES_PUBLISHER_REVIEW",
            )
            raise
        finally:
            if collision is not None:
                collision.close()


if __name__ == "__main__":
    raise SystemExit(main())
