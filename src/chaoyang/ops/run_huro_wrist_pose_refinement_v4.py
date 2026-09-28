#!/usr/bin/env python3
"""One bounded, full-session HuRo arm-wrist refinement on frozen V3 targets.

HuRo finger q22 stays bit-identical.  Only the 7-DoF arm is re-solved against
the same frozen root pose using the pinned Tianji FK and joint limits.  This is
an offline development ablation, not a HuRo-core reproduction or control action.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
from pathlib import Path
from time import monotonic

import numpy as np
from scipy.optimize import least_squares

from chaoyang.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH,
    artifact_ref, atomic_json, load_json, now_iso, process_identity, publish_bundle,
)
from chaoyang.pipeline import robot_scene_state_cpu as arm
from chaoyang.pipeline.robot_renderer_cycles import forward_kinematics
from chaoyang.pipeline.robot_renderer_eevee_fullchain import ARM_JOINT_NAMES, load_pinned_robot_assets
from chaoyang.pipeline.full_robot_review_v2 import (
    render_review, time_edges, motion_derivatives, tool_mount_from_flange,
)


TASK = "four_stream_full_pipeline_v4_takeover_v2"
LANE = REPO_ROOT / f"_run/current/{TASK}/attempts/attempt_0001/lanes/huro"
V3 = REPO_ROOT / "_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/shared/robot"
DATA = Path("/mnt/data/egodata/datasets/ego/processed/chips_cards_hands_0915/cleaned")
CASES = {
    "get_potato_chips_0915_007": {
        "huro": V3 / "huro_landmark_c3_full_epoch6_v1/get_potato_chips_0915_007/HURO_CAMERA_ROBOT_MOTION_V3.npz",
        "local": V3 / "landmark_c3_full_epoch6_v1/CAMERA_ROBOT_MOTION_V3.npz",
        "video": DATA / "potato_chips/get_potato_chips_0915_007/CameraRecord_get_potato_chips_0915_007.mp4",
        "frames": 378,
    },
    "play_cards_0915_031": {
        "huro": V3 / "huro031_epoch7_c3/play_cards_0915_031/HURO_CAMERA_ROBOT_MOTION_V3.npz",
        "local": V3 / "camera031_epoch7_c3/CAMERA_ROBOT_MOTION_V3.npz",
        "video": DATA / "playing_cards/play_cards_0915_031/CameraRecord_play_cards_0915_031.mp4",
        "frames": 149,
    },
}


def _publish_phase(status: str, phase: str, expected: int | None = None) -> int:
    receipt = load_json(RECEIPT_PATH)
    revision = int(receipt["governance_revision"])
    if expected is not None and revision != expected:
        raise RuntimeError("governance CAS mismatch")
    state = load_json(TASK_STATE_PATH)
    row = next(x for x in state["tasks"] if x.get("task_id") == TASK)
    ident = process_identity(os.getpid())
    time = now_iso()
    if status == "RUNNING":
        if row["status"] not in {"PENDING", "RUNNING"}:
            raise RuntimeError("task not executable")
        if row.get("pid") not in {None, os.getpid()}:
            raise RuntimeError("another primary writer owns task")
        row.update(status="RUNNING", phase=phase, pid=os.getpid(), proc_start_ticks=ident["start_ticks"],
                   heartbeat_at=time, updated_at=time)
    else:
        if row.get("pid") != os.getpid() or row.get("proc_start_ticks") != ident["start_ticks"]:
            raise RuntimeError("writer identity changed")
        row.update(status="PENDING", phase=phase, pid=None, proc_start_ticks=None,
                   heartbeat_at=None, updated_at=time)
    state["recent_events"] = (state.get("recent_events", []) + [{
        "task_id": TASK, "attempt": 1, "status": status, "created_at": time,
        "message": phase, "pid": os.getpid() if status == "RUNNING" else None,
        "proc_start_ticks": ident["start_ticks"] if status == "RUNNING" else None,
    }])[-100:]
    published = publish_bundle(load_json(AUTHORITY_PATH), state,
                               event_type="FOUR_STREAM_V4_TAKEOVER_V2_HURO_" + phase,
                               expected_revision=revision, generator_path=Path(__file__))
    return int(published["governance_revision"])


class Heartbeat:
    def __init__(self, phase: str):
        self.phase = phase
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.error: Exception | None = None

    def _loop(self) -> None:
        while not self.stop.wait(30):
            try:
                _publish_phase("RUNNING", self.phase)
            except Exception as exc:
                self.error = exc
                self.stop.set()

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()
        if self.error:
            raise self.error


def _load_case(session: str) -> tuple[dict, dict, dict]:
    if session not in CASES:
        raise ValueError("unfrozen session")
    spec = CASES[session]
    refs = {name: artifact_ref(path) for name, path in spec.items() if isinstance(path, Path)}
    with np.load(spec["huro"], allow_pickle=False) as z:
        huro = {k: np.asarray(z[k]) for k in z.files}
    with np.load(spec["local"], allow_pickle=False) as z:
        local = {k: np.asarray(z[k]) for k in z.files}
    n = spec["frames"]
    if len(huro["frame_id"]) != n or len(local["frame_id"]) != n:
        raise ValueError("frame count mismatch")
    for name in ("frame_id", "timestamp_ns", "target_valid", "human_to_physical"):
        if not np.array_equal(huro[name], local[name]):
            raise ValueError(f"frozen input mismatch: {name}")
    for name in ("T_target_root_cam", "T_cam_base", "T_flange_hand"):
        if not np.allclose(huro[name], local[name], equal_nan=True, atol=2e-6, rtol=0):
            raise ValueError(f"target/placement mismatch: {name}")
    if not np.array_equal(huro["target_valid"], huro["wrist_valid"]):
        raise ValueError("HuRo validity differs from frozen target")
    if not np.array_equal(huro["human_to_physical"], [0, 1]):
        raise ValueError("side convention mismatch")
    if bool(huro["control_ground_truth"]) or bool(huro["training_eligible"]):
        raise ValueError("unexpected HuRo authority")
    return huro, local, refs


def _summary(x: np.ndarray, valid: np.ndarray) -> dict:
    values = np.asarray(x)[valid]
    values = values[np.isfinite(values)]
    return {
        "count": int(values.size),
        "p50": float(np.percentile(values, 50)) if values.size else None,
        "p95": float(np.percentile(values, 95)) if values.size else None,
        "max": float(np.max(values)) if values.size else None,
    }


def _candidate(session: str, output: Path) -> dict:
    huro, local, refs = _load_case(session)
    assets = load_pinned_robot_assets(REPO_ROOT)
    lower, upper = arm._arm_limits(assets)
    neutral = .5 * (lower + upper)
    fk = forward_kinematics(assets.tianji, {
        name: float(neutral[side, k])
        for side in range(2) for k, name in enumerate(ARM_JOINT_NAMES[side])
    })
    flange_tool = np.stack([np.linalg.inv(fk[f]) @ fk[t] for f, t in
                            (("flange_L", "left_tool"), ("flange_R", "right_tool"))])
    mounts = huro["T_flange_hand"]
    tool_mounts = tool_mount_from_flange(flange_tool, mounts)
    neutral_roots = np.stack([fk["flange_L"], fk["flange_R"]]) @ mounts
    target = huro["T_target_root_cam"]
    valid = np.asarray(huro["target_valid"], bool)
    n = len(valid)
    edge, times = time_edges(huro["timestamp_ns"], huro["frame_id"])
    q_arm = np.full((n, 2, 7), np.nan)
    actual_cam = np.full((n, 2, 4, 4), np.nan)
    actual_base = np.full_like(actual_cam, np.nan)
    pos = np.full((n, 2), np.nan)
    rot = np.full((n, 2), np.nan)
    success = np.zeros((n, 2), bool)
    nfev = np.zeros((n, 2), np.int32)
    t_base_cam = np.linalg.inv(huro["T_cam_base"])
    for side in range(2):
        prior = None
        for frame in range(n):
            if not valid[frame, side]:
                prior = None
                continue
            if not edge[frame] or not valid[frame - 1, side]:
                prior = None
            huro_seed = np.asarray(huro["q_arm"][frame, side], float)
            if not np.isfinite(huro_seed).all():
                raise ValueError("nonfinite HuRo q on valid target")
            seed = np.clip(huro_seed, lower[side] + 1e-6, upper[side] - 1e-6)
            dt = (huro["timestamp_ns"][frame] - huro["timestamp_ns"][frame - 1]) / 1e9 if prior is not None else 1 / 30
            target_tool = t_base_cam @ target[frame, side] @ np.linalg.inv(tool_mounts[side])
            def residual(q):
                pose = arm._pose_residual(arm._tool_fk(assets, side, q), target_tool)
                prior_term = .001 * (q - huro_seed)
                temporal = np.array([]) if prior is None else .001 * (1 / 30) / dt * (q - prior)
                return np.r_[pose, prior_term, temporal]
            fit = least_squares(residual, seed, bounds=(lower[side], upper[side]),
                                max_nfev=150, ftol=1e-9, xtol=1e-9, gtol=1e-9)
            q_arm[frame, side] = fit.x
            root_base = arm._tool_fk(assets, side, fit.x) @ tool_mounts[side]
            root_cam = huro["T_cam_base"] @ root_base
            actual_base[frame, side] = root_base
            actual_cam[frame, side] = root_cam
            delta = np.linalg.inv(target[frame, side]) @ root_cam
            pos[frame, side] = np.linalg.norm(delta[:3, 3]) * 1000
            rot[frame, side] = np.degrees(np.linalg.norm(arm._rotation_vector(delta[:3, :3])))
            success[frame, side] = bool(fit.success)
            nfev[frame, side] = fit.nfev
            prior = fit.x if fit.success else None
    pass_tolerance = valid & success & (pos <= 20) & (rot <= 15)
    render_motion = {
        "q_arm": q_arm, "q22": huro["q22"], "wrist_valid": valid,
        "finger_valid": huro["finger_valid"], "T_target_root": t_base_cam @ target,
        "T_actual_root": actual_base, "T_flange_hand": mounts,
        "neutral_q_arm": neutral, "neutral_roots": neutral_roots,
        "position_residual_mm": pos, "rotation_residual_deg": rot,
        "tolerance_pass": pass_tolerance, "timestamp_ns": huro["timestamp_ns"],
        "frame_id": huro["frame_id"], "human_to_physical": huro["human_to_physical"],
    }
    np.savez_compressed(output / "HURO_WRIST_REFINED_MOTION.npz", **render_motion,
                        arm_solver_success=success, arm_solver_nfev=nfev,
                        source_huro_q22=huro["q22"],
                        **{"arm_" + k: v for k, v in motion_derivatives(q_arm, valid, times, edge).items()})
    video = output / "HURO_WRIST_REFINED_FULL_SESSION.mp4"
    review = render_review(render_motion, assets, CASES[session]["video"], video,
                           session + " | HuRo wrist-pose arm refinement",
                           preview_frames=(0, n // 2, n - 1))
    collision = review["collision_rows"]
    negative_contacts = sum(
        int(v or 0) for row in collision for key in ("hand_self", "arm_hand", "dual_hand", "robot_self")
        for v in (row[key] if isinstance(row[key], list) else [row[key]])
    )
    source_valid = valid & np.isfinite(huro["position_residual_mm"]) & np.isfinite(huro["rotation_residual_deg"])
    prior_pos = _summary(huro["position_residual_mm"], source_valid)
    prior_rot = _summary(huro["rotation_residual_deg"], source_valid)
    candidate_pos = _summary(pos, source_valid)
    candidate_rot = _summary(rot, source_valid)
    baseline_pos = _summary(local["position_residual_mm"], source_valid)
    baseline_rot = _summary(local["rotation_residual_deg"], source_valid)
    result = {
        "schema_version": "chaoyang-huro-wrist-pose-refinement-v4-result-v1",
        "session_id": session, "status": "DEVELOPMENT_CANDIDATE_EVALUATED",
        "input_refs": refs, "code": artifact_ref(Path(__file__)),
        "candidate_method": "HURO_CORE_Q22_BIT_EXACT_PLUS_BOUNDED_ARM_WRIST_POSE_REFINEMENT",
        "target_and_placement_reused_byte_equivalent": True,
        "q22_bit_exact": bool(np.array_equal(huro["q22"], render_motion["q22"], equal_nan=True)),
        "valid_side_frames": valid.sum(axis=0).tolist(),
        "solver_success_side_frames": success.sum(axis=0).tolist(),
        "tolerance_pass_side_frames": pass_tolerance.sum(axis=0).tolist(),
        "huro_original_wrist_position_mm": prior_pos,
        "huro_original_wrist_rotation_deg": prior_rot,
        "local_baseline_wrist_position_mm": baseline_pos,
        "local_baseline_wrist_rotation_deg": baseline_rot,
        "refined_wrist_position_mm": candidate_pos,
        "refined_wrist_rotation_deg": candidate_rot,
        "negative_collision_contacts_count": negative_contacts,
        "collision_scope": review["collision_limitations"],
        "video_decode_frames": review["decoded_frames"],
        "motion": artifact_ref(output / "HURO_WRIST_REFINED_MOTION.npz"),
        "video": artifact_ref(video),
        "quality_adopted": False, "visual_review_status": "PENDING_HUMAN_REVIEW",
        "training_eligible": False, "control_ground_truth": False, "physical_deployable": False,
        "claim_limit": "Offline same-target wrist-pose arm refinement; HuRo finger q unchanged. No object contact, external truth, control or physical validation.",
    }
    atomic_json(output / "RESULT.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", choices=sorted(CASES), required=True)
    parser.add_argument("--expected-revision", type=int, required=True)
    args = parser.parse_args()
    output = LANE / "wrist_pose_candidate_v1" / args.session
    if output.exists():
        raise RuntimeError("immutable candidate output already exists")
    revision = _publish_phase("RUNNING", "HURO_WRIST_POSE_REFINEMENT", args.expected_revision)
    output.mkdir(parents=True)
    started = monotonic()
    with Heartbeat("HURO_WRIST_POSE_REFINEMENT"):
        result = _candidate(args.session, output)
    revision = _publish_phase("PENDING", "V2_NEXT_LANE_AFTER_HURO")
    print(json.dumps({"session": args.session, "status": result["status"],
                      "revision": revision, "wall_seconds": monotonic() - started}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
