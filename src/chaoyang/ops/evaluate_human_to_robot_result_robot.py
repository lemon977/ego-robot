"""Independent FK, real-time motion metrics, and full Poker visual review."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from chaoyang.governance.common import REPO_ROOT, artifact_ref, atomic_json, load_json
from chaoyang.pipeline import robot_scene_state_cpu as arm
from chaoyang.pipeline.full_robot_review_v2 import motion_derivatives, render_review, time_edges
from chaoyang.pipeline.robot_renderer_cycles import forward_kinematics
from chaoyang.pipeline.robot_renderer_eevee_fullchain import ARM_JOINT_NAMES, load_pinned_robot_assets

TASK = "human_to_robot_result_breakthrough_20260924"
ROOT = REPO_ROOT / f"_run/current/{TASK}/attempts/attempt_0001/lanes/robot"
OLD = REPO_ROOT / ("_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/"
                   "lanes/ai2/camera_exact_mount_v2/play_cards_0902_042/CAMERA_ROBOT_MOTION_V3.npz")
NEW = ROOT / "POKER_171_CONSTRAINED_V3.npz"
EVAL_STEM = "POKER_171_INDEPENDENT_FK_V3"
SOURCE_RECEIPT = REPO_ROOT / ("_run/current/human_to_robot_representative_baseline_20260924/attempts/"
                              "attempt_0001/play_cards_0902_042_THIRD_PERSON_RESULT.json")


def _load(path):
    with np.load(path, allow_pickle=False) as bundle:
        return {k: bundle[k] for k in bundle.files}


def _stats(q, valid, edge, times):
    step = np.max(np.abs(np.diff(q, axis=0)), axis=-1)
    eligible = valid[1:] & valid[:-1] & edge[1:, None]
    derivatives = motion_derivatives(q, valid, times, edge)
    jerk = np.max(np.abs(derivatives["jerk"]), axis=-1)
    out = {}
    for side in range(2):
        values = step[:, side][eligible[:, side]]
        js = jerk[:, side][np.isfinite(jerk[:, side])]
        out[str(side)] = {"edge_count": int(len(values)),
                          "step_max_rad": float(np.max(values)),
                          "step_p99_rad": float(np.percentile(values, 99)),
                          "jerk_p95_rad_s3": float(np.percentile(js, 95)) if len(js) else None,
                          "max_step_source_frame": int(np.argmax(np.where(eligible[:, side], step[:, side], -1)) + 1)}
    return out, derivatives


def evaluate():
    if (ROOT / (EVAL_STEM + ".npz")).exists():
        raise FileExistsError("EVALUATION_ALREADY_EXISTS")
    old, new = _load(OLD), _load(NEW)
    if not np.array_equal(old["frame_id"], new["frame_id"]) or len(new["frame_id"]) != 171:
        raise RuntimeError("FRAME_MAP_MISMATCH")
    assets = load_pinned_robot_assets(REPO_ROOT)
    lower, upper = arm._arm_limits(assets)
    q = new["q_arm"]
    actual = np.full((171, 2, 4, 4), np.nan)
    position = np.full((171, 2), np.nan)
    rotation = position.copy()
    independent_parity = 0.0
    for t in range(171):
        joints = {name: float(q[t, side, j]) for side in range(2)
                  for j, name in enumerate(ARM_JOINT_NAMES[side])}
        fk = forward_kinematics(assets.tianji, joints)
        for side, flange in enumerate(("flange_L", "flange_R")):
            actual[t, side] = fk[flange] @ old["T_flange_hand"][side]
            delta = np.linalg.inv(old["T_target_root_base"][t, side]) @ actual[t, side]
            position[t, side] = 1000 * np.linalg.norm(delta[:3, 3])
            rotation[t, side] = np.rad2deg(np.linalg.norm(arm._rotation_vector(delta[:3, :3])))
            independent_parity = max(independent_parity, float(np.max(np.abs(
                arm._tool_fk(assets, side, q[t, side]) @
                np.linalg.inv(arm._tool_fk(assets, side, old["neutral_q_arm"][side])) @
                old["neutral_roots"][side] - actual[t, side]))))
    # Numerical agreement is determined by the independent flange result and
    # the declared mount, never by moving a display hand to the target.
    limit = np.all((q >= lower[None] - 1e-9) & (q <= upper[None] + 1e-9), axis=-1)
    receipt = load_json(ROOT / "POKER_171_CONSTRAINED_V3.json")
    success = np.zeros((171, 2), bool)
    for row in receipt["rows"]:
        if row.get("success"):
            success[row["frame"], row["side"]] = True
    passing = success & limit & (position <= 20.) & (rotation <= 15.)
    edge, times = time_edges(old["timestamp_ns"], old["frame_id"])
    old_stats, _ = _stats(old["q_arm"], old["wrist_valid"], edge, times)
    new_stats, deriv = _stats(q, old["wrist_valid"], edge, times)
    old_finger = old["q22"]
    if not np.array_equal(old_finger, new["q22"], equal_nan=True):
        raise RuntimeError("FINGER_Q_CHANGED_WITHOUT_AUTHORITY")
    finger_104_105 = float(np.max(np.abs(old_finger[105] - old_finger[104])))
    path = ROOT / (EVAL_STEM + ".npz")
    np.savez_compressed(path, q_arm=q, q22=old_finger, T_actual_root=actual,
                        T_target_root_base=old["T_target_root_base"],
                        position_residual_mm=position, rotation_residual_deg=rotation,
                        wrist_valid=old["wrist_valid"], finger_valid=old["finger_valid"],
                        tolerance_pass=passing, frame_id=old["frame_id"],
                        timestamp_ns=old["timestamp_ns"], T_flange_hand=old["T_flange_hand"],
                        neutral_q_arm=old["neutral_q_arm"], neutral_roots=old["neutral_roots"])
    result = {"schema_version": "POKER_171_RESULT_ROBOT_INDEPENDENT_EVALUATION_V1",
              "task_id": TASK, "source": artifact_ref(OLD), "candidate": artifact_ref(NEW),
              "independent_fk": artifact_ref(path), "frames": 171,
              "old_arm": old_stats, "new_arm": new_stats,
              "old_tolerance_pass": int(np.sum(old["tolerance_pass"])),
              "new_tolerance_pass": int(np.sum(passing)),
              "new_position_max_mm": float(np.max(position)),
              "new_rotation_max_deg": float(np.max(rotation)),
              "limit_pass_side_frames": int(np.sum(limit)),
              "independent_fk_max_abs_m": independent_parity,
              "finger_104_105_max_step_rad_unchanged": finger_104_105,
              "finger_quality": "UNCHANGED_OLD_JUMP_UNRESOLVED",
              "collision": "PENDING_DECLARED_SCOPE_RENDER_QUERY",
              "quality": "NOT_PRODUCT_PASS", "adoption": "NOT_ADOPTED"}
    atomic_json(ROOT / (EVAL_STEM + ".json"), result)
    return {"receipt": str(ROOT / (EVAL_STEM + ".json")),
            "new_tolerance_pass": result["new_tolerance_pass"],
            "old_tolerance_pass": result["old_tolerance_pass"],
            "old_step_max": old_stats["1"]["step_max_rad"],
            "new_step_max": new_stats["1"]["step_max_rad"]}


def render():
    result = load_json(ROOT / (EVAL_STEM + ".json"))
    source = Path(load_json(SOURCE_RECEIPT)["source_video"]["path"])
    candidate = _load(ROOT / (EVAL_STEM + ".npz"))
    motion = {**candidate, "T_target_root": candidate["T_target_root_base"]}
    video = ROOT / "POKER_171_FIXED_THIRD_PERSON_CANDIDATE_V3.mp4"
    if video.exists():
        raise FileExistsError(video)
    rendered = render_review(motion, load_pinned_robot_assets(REPO_ROOT), source, video,
                             "play_cards_0902_042", preview_frames=(0, 80, 100, 105, 170))
    collision = rendered.pop("collision_rows")
    result = {"schema_version": "POKER_171_RESULT_ROBOT_REVIEW_V1", "task_id": TASK,
              "evaluation": artifact_ref(ROOT / (EVAL_STEM + ".json")),
              "video": artifact_ref(video), "source_video": artifact_ref(source),
              "decoded_frames": rendered["decoded_frames"],
              "collision_scope": rendered["collision_limitations"],
              "collision_counts": {key: int(sum(value if isinstance(value, int) else 0
                                                for row in collision for value in
                                                (row[key] if isinstance(row[key], list) else [row[key]])))
                                   for key in collision[0]},
              "finger_jump_unresolved": result["finger_104_105_max_step_rad_unchanged"],
              "quality": "CANDIDATE_PENDING_INDEPENDENT_VISUAL_REVIEW_NOT_PRODUCT_PASS",
              "adoption": "NOT_ADOPTED"}
    atomic_json(ROOT / "POKER_171_FIXED_THIRD_PERSON_RESULT_V3.json", result)
    return {"video": str(video), "decoded_frames": rendered["decoded_frames"],
            "receipt": str(ROOT / "POKER_171_FIXED_THIRD_PERSON_RESULT_V3.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("evaluate", "render"))
    args = parser.parse_args()
    print(json.dumps(evaluate() if args.stage == "evaluate" else render(), ensure_ascii=False))


if __name__ == "__main__":
    main()
