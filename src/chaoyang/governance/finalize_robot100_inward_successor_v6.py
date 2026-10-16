"""Strictly finalize the 100-session inward-forearm posture successor."""
from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path

import numpy as np

from chaoyang.governance.common import REPO_ROOT, artifact_ref, atomic_json, load_json
from chaoyang.governance import finalize_robot100_inward_successor_v5 as v5final
from chaoyang.ops import run_robot100_batch_baseline_v1 as baseline
from chaoyang.ops import run_robot100_inward_successor_v5 as v5
from chaoyang.ops import run_robot100_inward_successor_v6 as runner
from chaoyang.pipeline import robot_scene_state_cpu as arm
from chaoyang.pipeline.robot_renderer_eevee_fullchain import load_pinned_robot_assets

BASE = runner.OUTPUT.parent
MAX_Q_STEP_RAD = 0.75
RAW_PRESERVED_ARRAYS = (
    "wrist_valid",
    "T_target_root",
    "q_arm_raw",
    "T_actual_root_raw",
    "position_residual_mm_raw",
    "rotation_residual_deg_raw",
    "tolerance_pass_raw",
)


def verify_ref(ref: dict, allowed_root: Path) -> Path:
    path = Path(ref["path"]).resolve(strict=True)
    if not path.is_relative_to(allowed_root.resolve(strict=True)):
        raise RuntimeError(f"OUTPUT_SCOPE_ESCAPE:{path}")
    size, digest = v5final.stable_sha(path)
    if size != ref["bytes"] or digest != ref["sha256"]:
        raise RuntimeError(f"ARTIFACT_REFERENCE_DRIFT:{path}")
    return path


def verify_raw_predecessor_preserved(current: dict[str, np.ndarray], predecessor: dict[str, np.ndarray]) -> None:
    """Reject any V6 change to the raw V5 target/tracking evidence."""
    for name in RAW_PRESERVED_ARRAYS:
        if name not in current or name not in predecessor:
            raise RuntimeError(f"RAW_PREDECESSOR_ARRAY_MISSING:{name}")
        if current[name].shape != predecessor[name].shape or not np.array_equal(
            current[name], predecessor[name], equal_nan=True
        ):
            raise RuntimeError(f"RAW_PREDECESSOR_DRIFT:{name}")


def audit_session(row: dict, assets, mounts: np.ndarray, lower: np.ndarray, upper: np.ndarray, neutral: np.ndarray) -> tuple[dict, Path]:
    folder = runner.OUTPUT / row["session_id"]
    result_path = folder / "RESULT.json"
    result = load_json(result_path)
    if result.get("schema_version") != "CHAOYANG_ROBOT100_INWARD_POSTURE_SESSION_V6":
        raise RuntimeError(f"RESULT_SCHEMA:{row['session_id']}")
    if result.get("session_id") != row["session_id"] or result.get("status") != "COMPLETE_PENDING_HUMAN_REVIEW":
        raise RuntimeError(f"RESULT_IDENTITY_OR_STATUS:{row['session_id']}")
    predecessor_path = verify_ref(result["predecessor_v5"], v5.OUTPUT)
    predecessor = load_json(predecessor_path)
    if predecessor.get("session_id") != row["session_id"]:
        raise RuntimeError(f"PREDECESSOR_IDENTITY:{row['session_id']}")
    motion_path = verify_ref(result["motion"], folder)
    video_path = verify_ref(result["video"], folder)
    with np.load(motion_path, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    predecessor_motion_path = verify_ref(predecessor["motion"], v5.OUTPUT)
    with np.load(predecessor_motion_path, allow_pickle=False) as source:
        predecessor_arrays = {name: source[name] for name in source.files}
    verify_raw_predecessor_preserved(arrays, predecessor_arrays)
    frames = len(arrays["frame_id"])
    expected = {
        "q_arm": (frames, 2, 7),
        "q_arm_raw": (frames, 2, 7),
        "q_arm_v5_display": (frames, 2, 7),
        "wrist_valid": (frames, 2),
        "posture_projected": (frames, 2),
        "posture_projection_alpha": (frames, 2),
        "tolerance_pass_raw": (frames, 2),
        "source_video_frame_idx": (frames,),
        "source_video_frame_idx_raw": (frames,),
    }
    for name, shape in expected.items():
        if name not in arrays or arrays[name].shape != shape:
            raise RuntimeError(f"ARRAY_SHAPE:{row['session_id']}:{name}")
    valid = arrays["wrist_valid"].astype(bool)
    raw_pass = arrays["tolerance_pass_raw"].astype(bool) & valid
    mapping = arrays["source_video_frame_idx"]
    if np.any(mapping < 0) or np.any(mapping >= int(row["frames"])):
        raise RuntimeError(f"FRAME_MAPPING_BOUNDS:{row['session_id']}")
    q = arrays["q_arm"]
    if np.any(q[valid] < lower[np.where(valid)[1]] - 1e-9) or np.any(q[valid] > upper[np.where(valid)[1]] + 1e-9):
        raise RuntimeError(f"JOINT_LIMIT:{row['session_id']}")
    minimum = [float("inf"), float("inf")]
    for frame, side in zip(*np.where(valid)):
        margin = runner.inward_margin_m(assets, mounts, neutral, int(side), q[frame, side])
        minimum[side] = min(minimum[side], margin)
        if margin < runner.MIN_INWARD_M - 1e-7:
            raise RuntimeError(f"POSTURE_GATE:{row['session_id']}:{frame}:{side}:{margin}")
    edge = arrays["time_edge_valid"][1:, None, None] & valid[1:, :, None] & valid[:-1, :, None]
    delta = np.abs(np.diff(q, axis=0))
    max_step = float(delta[edge.repeat(7, axis=2)].max()) if edge.any() else 0.0
    if max_step > MAX_Q_STEP_RAD:
        raise RuntimeError(f"DISPLAY_STEP_GATE:{row['session_id']}:{max_step}")
    projected = arrays["posture_projected"].sum(axis=0).astype(int).tolist()
    if projected != result["posture_policy"]["projected_frames_left_right"]:
        raise RuntimeError(f"PROJECTED_COUNT_DRIFT:{row['session_id']}")
    return ({
        "session_id": row["session_id"],
        "task": row["task"],
        "frames": int(row["frames"]),
        "valid_left_right": valid.sum(axis=0).astype(int).tolist(),
        "raw_tracking_pass_left_right": raw_pass.sum(axis=0).astype(int).tolist(),
        "raw_tracking_fail_left_right": (valid & ~raw_pass).sum(axis=0).astype(int).tolist(),
        "posture_projected_left_right": projected,
        "posture_minimum_m_left_right": minimum,
        "max_display_joint_step_rad": max_step,
        "maximum_display_position_residual_mm_left_right": np.nanmax(arrays["position_residual_mm"], axis=0).tolist(),
        "maximum_display_rotation_residual_deg_left_right": np.nanmax(arrays["rotation_residual_deg"], axis=0).tolist(),
        "result": artifact_ref(result_path),
        "motion": result["motion"],
        "video": result["video"],
    }, video_path)


def main() -> int:
    selection = load_json(v5.SELECTION)
    if selection.get("count") != 100 or len(selection.get("rows", [])) != 100:
        raise RuntimeError("SELECTION_COUNT")
    first_worker = load_json(BASE / "WORKER_0.json")
    worker_count = int(first_worker.get("worker_count", 0))
    if worker_count not in (2, 4):
        raise RuntimeError(f"WORKER_COUNT:{worker_count}")
    expected_per_worker = 100 // worker_count
    for worker in range(worker_count):
        summary = load_json(BASE / f"WORKER_{worker}.json")
        if summary.get("schema_version") != "CHAOYANG_ROBOT100_INWARD_POSTURE_WORKER_V6":
            raise RuntimeError(f"WORKER_SCHEMA:{worker}")
        if summary.get("worker_count") != worker_count:
            raise RuntimeError(f"WORKER_PARTITION:{worker}")
        if summary.get("failures") or len(summary.get("completed", [])) != expected_per_worker:
            raise RuntimeError(f"WORKER_INCOMPLETE:{worker}")

    assets = load_pinned_robot_assets(REPO_ROOT)
    mounts, mount_ref = baseline._mounts()
    lower, upper = arm._arm_limits(assets)
    neutral = 0.5 * (lower + upper)
    rows, videos = [], []
    for selection_row in selection["rows"]:
        audited, video = audit_session(selection_row, assets, mounts, lower, upper, neutral)
        rows.append(audited)
        videos.append(video)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        decoded = list(pool.map(v5final.decode_video, videos))
    for row, decode in zip(rows, decoded, strict=True):
        if decode["decoded_frames"] != row["frames"]:
            raise RuntimeError(f"VIDEO_FRAME_COUNT:{row['session_id']}")
        row["decode"] = decode

    total_valid = np.sum([row["valid_left_right"] for row in rows], axis=0).astype(int)
    total_projected = np.sum([row["posture_projected_left_right"] for row in rows], axis=0).astype(int)
    audit = {
        "schema_version": "CHAOYANG_ROBOT100_INWARD_POSTURE_QUALITY_AUDIT_V6",
        "status": "PASS",
        "sessions": len(rows),
        "task_counts": {task: sum(row["task"] == task for row in rows) for task in sorted({row["task"] for row in rows})},
        "valid_left_right": total_valid.tolist(),
        "raw_tracking_pass_left_right": np.sum([row["raw_tracking_pass_left_right"] for row in rows], axis=0).astype(int).tolist(),
        "raw_tracking_fail_left_right": np.sum([row["raw_tracking_fail_left_right"] for row in rows], axis=0).astype(int).tolist(),
        "posture_projected_left_right": total_projected.tolist(),
        "posture_projected_fraction_left_right": (total_projected / total_valid).tolist(),
        "minimum_inward_m_left_right": np.min([row["posture_minimum_m_left_right"] for row in rows], axis=0).tolist(),
        "maximum_display_joint_step_rad": max(row["max_display_joint_step_rad"] for row in rows),
        "maximum_display_position_residual_mm_left_right": np.max([row["maximum_display_position_residual_mm_left_right"] for row in rows], axis=0).tolist(),
        "maximum_display_rotation_residual_deg_left_right": np.max([row["maximum_display_rotation_residual_deg_left_right"] for row in rows], axis=0).tolist(),
        "video_full_decode": "PASS_100_OF_100",
        "interpretation": (
            "Raw V5 target/tracking arrays are preserved byte-for-byte at the array level, including any predecessor "
            "tracking failures. V6 enforces the requested visible elbow-outside-wrist posture as an explicitly inferred "
            "display projection; target residual can therefore increase on projected frames."
        ),
    }
    atomic_json(BASE / "QUALITY_AUDIT.json", audit)
    atomic_json(BASE / "VIDEO_INDEX.json", {"schema_version": "CHAOYANG_ROBOT100_INWARD_POSTURE_VIDEO_INDEX_V6", "count": 100, "rows": rows})
    result = {
        "schema_version": "CHAOYANG_ROBOT100_INWARD_POSTURE_RESULT_V6",
        "status": "COMPLETE_MACHINE_PASS_PENDING_HUMAN_REVIEW",
        "count": 100,
        "selection": artifact_ref(v5.SELECTION),
        "mount_contract": mount_ref,
        "quality_audit": artifact_ref(BASE / "QUALITY_AUDIT.json"),
        "video_index": artifact_ref(BASE / "VIDEO_INDEX.json"),
        "algorithm_quality_adopted": False,
        "human_review_complete": False,
        "claim_limit": (
            "100 offline 0916 Robot posture-review videos. Posture-projected frames are inferred display motion, "
            "not exact wrist tracking and not physical/control/training authority."
        ),
    }
    atomic_json(BASE / "RESULT.json", result)
    readme = "# 0916 Robot100 全轨迹内收姿态 V6\n\n"
    readme += "- 状态：100 条机器校验完成，等待人工观看确认。\n"
    readme += "- 修正：不再只检查首帧；所有有效帧均验证肘在腕外侧，最小横向余量 10 mm。\n"
    readme += "- 证据：原始 V5 IK、目标与残差全部保留；发生姿态投影的帧明确标为推断展示。\n"
    readme += f"- 姿态投影帧（左/右）：{total_projected.tolist()} / 有效帧 {total_valid.tolist()}。\n"
    readme += f"- V5 原始腕跟随失败帧（左/右，原样保留）：{audit['raw_tracking_fail_left_right']}。\n"
    readme += f"- 最大连续关节步长：{audit['maximum_display_joint_step_rad']:.6f} rad。\n"
    readme += "- 限制：投影帧可能偏离腕目标；不具备物理标定、控制或训练授权。\n"
    (BASE / "先看这里.md").write_text(readme, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
