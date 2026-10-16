"""Correct the 0916 Robot100 anatomical-to-physical side mapping.

The predecessor stored anatomical left/right sensor streams in physical
right/left slots.  This successor restores identity mapping (human left drives
physical robot left) and reruns arm IK, display bridge and the V6 posture gate.
It never swaps already-solved arm joint arrays between asymmetric robot arms.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from chaoyang.governance.common import REPO_ROOT, artifact_ref, atomic_json, load_json
from chaoyang.ops import run_robot100_batch_baseline_v1 as baseline
from chaoyang.ops import run_robot100_inward_successor_v4 as v4
from chaoyang.ops import run_robot100_inward_successor_v5 as v5
from chaoyang.ops import run_robot100_inward_successor_v6 as v6
from chaoyang.pipeline.full_robot_review_inward_v3 import solve_full_chain
from chaoyang.pipeline.robot_renderer_eevee_fullchain import load_pinned_robot_assets


IDENTITY_MAPPING = np.asarray([0, 1], dtype=np.int64)
LEGACY_SWAPPED_MAPPING = np.asarray([1, 0], dtype=np.int64)
SIDE_INPUT_FIELDS = ("q22", "finger_valid", "wrist_valid", "relative_wrist_T")
ROOT = (
    REPO_ROOT
    / "_run/current/robot100_batch_baseline_20260928/attempts/attempt_0001"
    / "corrections/anatomical_side_identity_rear_view_v8_1"
)
OUTPUT = ROOT / "canaries"


def restore_anatomical_identity(source: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Move anatomical L/R streams back to physical robot L/R slots."""
    if not np.array_equal(source["human_to_physical"], LEGACY_SWAPPED_MAPPING):
        raise RuntimeError("PREDECESSOR_SIDE_MAPPING_DRIFT")
    result = dict(source)
    for name in SIDE_INPUT_FIELDS:
        values = np.asarray(source[name])
        if values.ndim < 2 or values.shape[1] != 2:
            raise RuntimeError(f"SIDE_FIELD_SHAPE:{name}:{values.shape}")
        result[name] = values[:, [1, 0]].copy()
    result["human_to_physical"] = IDENTITY_MAPPING.copy()
    return result


def target_path_mm(transforms: np.ndarray, valid: np.ndarray) -> list[float]:
    answer: list[float] = []
    for side in range(2):
        indices = np.flatnonzero(valid[:, side])
        if len(indices) < 2:
            answer.append(0.0)
            continue
        xyz = transforms[indices, side, :3, 3]
        answer.append(float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum() * 1000.0))
    return answer


def run(row: dict, assets, mounts: np.ndarray, hand_eval: int, arm_eval: int) -> dict:
    folder = OUTPUT / row["session_id"]
    result_path = folder / "RESULT.json"
    if result_path.is_file():
        result = load_json(result_path)
        if result.get("session_id") != row["session_id"]:
            raise RuntimeError("RESULT_IDENTITY_DRIFT")
        return result
    if folder.exists():
        raise RuntimeError(f"PARTIAL_OUTPUT:{folder}")

    predecessor_result_path = v5.OUTPUT / row["session_id"] / "RESULT.json"
    predecessor_result = load_json(predecessor_result_path)
    v6.stable_artifact(predecessor_result["motion"], v5.OUTPUT)
    original, sensor_metrics = baseline._load_motion(row, hand_eval)
    arrays = v4.scaled_motion(restore_anatomical_identity(original))

    started = time.monotonic()
    solved = solve_full_chain(
        arrays,
        assets,
        mounts,
        max_nfev=arm_eval,
        continuity_weight=0.5,
        reference_q_arm=v4.REFERENCE_Q,
        expected_human_to_physical=IDENTITY_MAPPING,
    )
    bridge_windows = v4.apply_display_motion(solved, assets, mounts)
    corrected = v6.apply_posture_projection(solved, assets, mounts)
    raw_source_map = np.asarray(original["source_video_frame_idx"], dtype=np.int64).copy()
    bounded_source_map = np.clip(raw_source_map, 0, int(row["frames"]) - 1)
    corrected["source_video_frame_idx_raw"] = raw_source_map
    corrected["source_video_frame_idx"] = bounded_source_map

    valid = corrected["wrist_valid"]
    if not np.array_equal(corrected["human_to_physical"], IDENTITY_MAPPING):
        raise RuntimeError("CORRECTED_SIDE_MAPPING_DRIFT")
    if not all(
        v6.inward_margin_m(assets, mounts, corrected["neutral_q_arm"], side, corrected["q_arm"][frame, side])
        >= v6.MIN_INWARD_M - 1e-9
        for frame, side in zip(*np.where(valid))
    ):
        raise RuntimeError("CORRECTED_POSTURE_GATE")

    folder.mkdir(parents=True)
    motion = folder / "ROBOT_MOTION_ANATOMICAL_IDENTITY_V8.npz"
    np.savez_compressed(motion, **corrected)
    video = folder / f"{row['session_id']}_解剖左右同侧_同视向_上臂向外_小臂向内_Robot_V8_1_CANARY.mp4"
    review = baseline._render(
        row,
        corrected,
        video,
        camera_yaw_degrees=-140.0,
        view_description="佩戴者同视向后视角（画面左右与解剖左右一致）",
    )

    edge = corrected["time_edge_valid"][1:, None, None] & valid[1:, :, None] & valid[:-1, :, None]
    step = np.abs(np.diff(corrected["q_arm"], axis=0))
    max_step = float(step[edge.repeat(7, axis=2)].max()) if edge.any() else 0.0
    result = {
        "schema_version": "CHAOYANG_ROBOT100_ANATOMICAL_IDENTITY_SESSION_V8_1",
        "session_id": row["session_id"],
        "status": "CANARY_COMPLETE_PENDING_HUMAN_REVIEW",
        "predecessor_v5": artifact_ref(predecessor_result_path),
        "side_mapping": {
            "predecessor_human_to_physical": LEGACY_SWAPPED_MAPPING.tolist(),
            "corrected_human_to_physical": IDENTITY_MAPPING.tolist(),
            "meaning": "anatomical human left/right drives physical robot left/right",
        },
        "review_camera": {
            "yaw_degrees": -140.0,
            "view": "WEARER_ALIGNED_REAR_VIEW",
            "screen_left_right_matches_anatomical_left_right": True,
        },
        "target_path_mm_physical_left_right": target_path_mm(corrected["T_target_root"], valid),
        "bridge_windows": bridge_windows,
        "posture_projected_left_right": corrected["posture_projected"].sum(axis=0).astype(int).tolist(),
        "maximum_display_joint_step_rad": max_step,
        "raw_tracking_pass_left_right": corrected["tolerance_pass_raw"].sum(axis=0).astype(int).tolist(),
        "wrist_valid_left_right": valid.sum(axis=0).astype(int).tolist(),
        "sensor_metrics": sensor_metrics,
        "motion": artifact_ref(motion),
        "video": artifact_ref(video),
        "review": review,
        "elapsed_seconds": time.monotonic() - started,
        "algorithm_quality_adopted": False,
        "claim_limit": "0916 offline side-mapping correction canary; human review required before 100-session expansion.",
    }
    atomic_json(result_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", nargs="+", default=["play_cards_0916_009"])
    parser.add_argument("--max-hand-evaluations", type=int, default=24)
    parser.add_argument("--max-arm-evaluations", type=int, default=60)
    args = parser.parse_args()
    selection = load_json(v5.SELECTION)
    rows = {row["session_id"]: row for row in selection["rows"]}
    if len(set(args.sessions)) != len(args.sessions) or any(name not in rows for name in args.sessions):
        raise RuntimeError("SESSION_SELECTION_INVALID")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    assets = load_pinned_robot_assets(REPO_ROOT)
    mounts, _ = baseline._mounts()
    results = [
        run(rows[name], assets, mounts, args.max_hand_evaluations, args.max_arm_evaluations)
        for name in args.sessions
    ]
    atomic_json(
        ROOT / "RESULT.json",
        {
            "schema_version": "CHAOYANG_ROBOT100_ANATOMICAL_IDENTITY_CANARY_RESULT_V8_1",
            "status": "CANARIES_COMPLETE_PENDING_HUMAN_REVIEW",
            "sessions": args.sessions,
            "results": [artifact_ref(OUTPUT / name / "RESULT.json") for name in args.sessions],
            "algorithm_quality_adopted": False,
        },
    )
    print(json.dumps({"status": "COMPLETE", "count": len(results)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
