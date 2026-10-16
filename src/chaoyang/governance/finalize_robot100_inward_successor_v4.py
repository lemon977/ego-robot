"""Finalize and machine-audit the 100-session stable inward Robot successor."""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

from chaoyang.governance.common import REPO_ROOT, artifact_ref, atomic_json, load_json
from chaoyang.ops import run_robot100_batch_baseline_v1 as predecessor
from chaoyang.ops import run_robot100_inward_successor_v4 as runner
from chaoyang.pipeline.robot_renderer_cycles import forward_kinematics
from chaoyang.pipeline.robot_renderer_eevee_fullchain import ARM_JOINT_NAMES, load_pinned_robot_assets

BASE = runner.OUTPUT.parent
MAX_Q_STEP_RAD = 0.75
MIN_ELBOW_TO_WRIST_INWARD_M = 0.04


def stable_sha(path: Path) -> tuple[int, str]:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    identity = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, name) != getattr(after, name) for name in identity):
        raise RuntimeError(f"UNSTABLE_FILE:{path}")
    return before.st_size, digest.hexdigest()


def verify_ref(ref: dict, allowed_root: Path) -> Path:
    path = Path(ref["path"]).resolve(strict=True)
    if not path.is_relative_to(allowed_root.resolve(strict=True)):
        raise RuntimeError(f"OUTPUT_SCOPE_ESCAPE:{path}")
    size, digest = stable_sha(path)
    if size != ref["bytes"] or digest != ref["sha256"]:
        raise RuntimeError(f"ARTIFACT_REFERENCE_DRIFT:{path}")
    return path


def decode_video(path: Path) -> dict:
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames,width,height,r_frame_rate",
            "-of", "json", str(path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    subprocess.run(
        ["ffmpeg", "-v", "error", "-xerror", "-i", str(path), "-map", "0:v:0", "-f", "null", "-"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    return {
        "decoded_frames": int(stream["nb_read_frames"]),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frame_rate": stream["r_frame_rate"],
        "full_decode": "PASS",
    }


def reference_geometry() -> list[dict]:
    assets = load_pinned_robot_assets(REPO_ROOT)
    mounts, _ = predecessor._mounts()
    joint_map = {
        name: float(runner.REFERENCE_Q[side, index])
        for side in range(2)
        for index, name in enumerate(ARM_JOINT_NAMES[side])
    }
    fk = forward_kinematics(assets.tianji, joint_map)
    rows = []
    for side, suffix in enumerate(("L", "R")):
        sign = 1.0 if side == 0 else -1.0
        elbow = sign * float(fk[f"Link3_{suffix}"][1, 3])
        wrist = sign * float((fk[f"flange_{suffix}"] @ mounts[side])[1, 3])
        inward = elbow - wrist
        rows.append({
            "physical_side": side,
            "signed_elbow_lateral_m": elbow,
            "signed_wrist_lateral_m": wrist,
            "elbow_to_wrist_inward_m": inward,
            "pass": bool(wrist > 0 and inward >= MIN_ELBOW_TO_WRIST_INWARD_M),
        })
    return rows


def audit_session(row: dict) -> tuple[dict, Path]:
    folder = runner.OUTPUT / row["session_id"]
    result_path = folder / "RESULT.json"
    if not result_path.is_file():
        raise RuntimeError(f"MISSING_RESULT:{row['session_id']}")
    result = load_json(result_path)
    if result.get("schema_version") != "CHAOYANG_ROBOT100_INWARD_SUCCESSOR_SESSION_V4":
        raise RuntimeError(f"RESULT_SCHEMA:{row['session_id']}")
    if result.get("session_id") != row["session_id"]:
        raise RuntimeError(f"RESULT_IDENTITY:{row['session_id']}")
    if result.get("status") != "COMPLETE_PENDING_HUMAN_REVIEW":
        raise RuntimeError(f"RESULT_STATUS:{row['session_id']}")
    if result.get("predecessor_selection_row") != row:
        raise RuntimeError(f"SELECTION_BINDING:{row['session_id']}")
    if not np.array_equal(np.asarray(result["reference_q_arm"]), runner.REFERENCE_Q):
        raise RuntimeError(f"REFERENCE_Q_DRIFT:{row['session_id']}")
    motion = verify_ref(result["motion"], folder)
    video = verify_ref(result["video"], folder)
    with np.load(motion, allow_pickle=False) as arrays:
        frames = int(row["frames"])
        expected = {
            "q_arm": (frames, 2, 7),
            "wrist_valid": (frames, 2),
            "tolerance_pass_raw": (frames, 2),
            "display_inferred": (frames, 2),
            "source_video_frame_idx": (frames,),
        }
        for key, shape in expected.items():
            if arrays[key].shape != shape:
                raise RuntimeError(f"ARRAY_SHAPE:{row['session_id']}:{key}")
        if not np.array_equal(arrays["q_arm"][0], runner.REFERENCE_Q):
            raise RuntimeError(f"FRAME0_REFERENCE_DRIFT:{row['session_id']}")
        source_map = arrays["source_video_frame_idx"].astype(np.int64)
        # Sensor timestamps are nearest-neighbour mapped to encoded frames, so
        # adjacent samples may legitimately repeat or skip one frame.  The
        # mapping must nevertheless be ordered, bounded and stay within one
        # frame of the acquisition timeline.
        if not (
            abs(int(source_map[0])) <= 1
            and abs(int(source_map[-1]) - (frames - 1)) <= 1
            and np.all(np.diff(source_map) >= 0)
            and np.all((source_map >= 0) & (source_map < frames))
            and np.max(np.abs(source_map - np.arange(frames))) <= 1
        ):
            raise RuntimeError(f"VIDEO_FRAME_MAPPING:{row['session_id']}")
        valid = arrays["wrist_valid"].astype(bool)
        tolerance = arrays["tolerance_pass_raw"].astype(bool)
        inferred = arrays["display_inferred"].astype(bool)
        edge = arrays["time_edge_valid"].astype(bool)
        q = arrays["q_arm"]
        eligible = edge[1:, None, None] & valid[1:, :, None] & valid[:-1, :, None]
        delta = np.abs(np.diff(q, axis=0))
        max_step = float(delta[eligible.repeat(7, axis=2)].max()) if eligible.any() else 0.0
        valid_by_side = valid.sum(axis=0).astype(int).tolist()
        passed_by_side = tolerance.sum(axis=0).astype(int).tolist()
        nonfinite = int((~np.isfinite(q[valid])).sum())
    decode = decode_video(video)
    if decode["decoded_frames"] != int(row["frames"]):
        raise RuntimeError(f"VIDEO_FRAME_COUNT:{row['session_id']}")
    checks = {
        "finite_valid_q": nonfinite == 0,
        "all_raw_valid_wrist_frames_meet_20mm_15deg": passed_by_side == valid_by_side,
        "max_joint_step_le_0_75_rad": max_step <= MAX_Q_STEP_RAD,
        "full_video_decode": True,
    }
    return ({
        "session_id": row["session_id"],
        "task": row["task"],
        "frames": int(row["frames"]),
        "wrist_valid_physical_left_right": valid_by_side,
        "raw_arm_tolerance_pass_physical_left_right": passed_by_side,
        "max_joint_step_rad": max_step,
        "display_inferred_left_right": inferred.sum(axis=0).astype(int).tolist(),
        "checks": checks,
        "machine_pass": all(checks.values()),
        "video": result["video"],
        "motion": result["motion"],
        "decode": decode,
        "human_review": "PENDING",
    }, result_path)


def main() -> int:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("FFMPEG_TOOLS_REQUIRED")
    selection = load_json(runner.SELECTION)
    rows = selection.get("rows", [])
    if selection.get("count") != 100 or len(rows) != 100:
        raise RuntimeError("SELECTION_NOT_100")
    session_ids = [row["session_id"] for row in rows]
    if len(set(session_ids)) != 100:
        raise RuntimeError("DUPLICATE_SESSION")
    task_counts = {task: sum(row["task"] == task for row in rows) for task in sorted({row["task"] for row in rows})}
    if sorted(task_counts.values()) != [50, 50]:
        raise RuntimeError(f"TASK_BALANCE:{task_counts}")
    worker_results = [load_json(BASE / f"WORKER_{index}.json") for index in range(2)]
    if any(result.get("failures") for result in worker_results):
        raise RuntimeError("WORKER_FAILURES_PRESENT")
    if sum(len(result.get("completed", [])) for result in worker_results) != 100:
        raise RuntimeError("WORKER_COMPLETION_NOT_100")

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        audited_pairs = list(pool.map(audit_session, rows))
    audited = [audit for audit, _ in audited_pairs]
    geometry = reference_geometry()
    geometry_pass = all(row["pass"] for row in geometry)
    machine_pass = geometry_pass and all(row["machine_pass"] for row in audited)

    video_index = {
        "schema_version": "CHAOYANG_ROBOT100_INWARD_VIDEO_INDEX_V4",
        "count": len(audited),
        "task_counts": task_counts,
        "sessions": [{"session_id": row["session_id"], "task": row["task"], "video": row["video"]} for row in audited],
    }
    atomic_json(BASE / "VIDEO_INDEX.json", video_index)
    quality = {
        "schema_version": "CHAOYANG_ROBOT100_INWARD_QUALITY_AUDIT_V4",
        "status": "PASS_PENDING_HUMAN_REVIEW" if machine_pass else "FAIL",
        "reference_policy": "FIXED_75_PERCENT_INWARD_FROM_OLD_MIDPOINT_FOR_ALL_SESSIONS",
        "translation_retarget_scale": runner.TRANSLATION_SCALE,
        "branch_bridge_policy": {
            "jump_threshold_rad": runner.BRANCH_JUMP_RAD,
            "radius_frames": runner.BRIDGE_RADIUS_FRAMES,
            "provenance": "display_inferred is separate from raw IK and sensor validity",
        },
        "reference_geometry": geometry,
        "limits": {
            "position_residual_mm": 20.0,
            "rotation_residual_deg": 15.0,
            "max_joint_step_rad": MAX_Q_STEP_RAD,
            "minimum_elbow_to_wrist_inward_m": MIN_ELBOW_TO_WRIST_INWARD_M,
        },
        "sessions": audited,
        "human_review": "PENDING",
        "claim_limit": "Digital-asset and video-integrity checks only; no physical calibration, control, safety or training authority.",
    }
    atomic_json(BASE / "QUALITY_AUDIT.json", quality)
    readme = BASE / "先看这里.md"
    readme.write_text(
        "# 0916 Robot100 上臂向外、小臂向内 V4\n\n"
        f"- 会话：{len(audited)}（{task_counts}）\n"
        f"- 机器验收：{'通过，待人工审阅' if machine_pass else '未通过'}\n"
        "- 姿态：全体会话使用同一75%固定内收参考和0.75固定平移重定向尺度，不逐会话或逐帧调姿。\n"
        "- 分支桥接：仅对原始IK相邻帧超过0.75 rad的分支切换做固定窗口离线桥接；桥接帧保留推断掩码，不冒充精确传感器跟随。\n"
        "- 检查：逐文件SHA、全片解码、帧映射、原始IK 20 mm/15°腕跟随、显示0.75 rad动态跳变门、肘外腕内几何门。\n"
        "- 旧T形批次、过深V2和仍跳变V3均只作为前驱，不计入本100条。\n"
        "- 权威限制：仅离线虚拟Robot开发可视化；人工审阅前不采用为算法质量基线。\n\n"
        "索引见 `VIDEO_INDEX.json`，逐会话数值见 `QUALITY_AUDIT.json`。\n",
        encoding="utf-8",
    )
    result = {
        "schema_version": "CHAOYANG_ROBOT100_INWARD_SUCCESSOR_RESULT_V4",
        "status": "COMPLETE_MACHINE_PASS_PENDING_HUMAN_REVIEW" if machine_pass else "COMPLETE_MACHINE_FAIL",
        "count": len(audited),
        "task_counts": task_counts,
        "code_commit": subprocess.run(["git", "rev-parse", "HEAD"], check=True, text=True, capture_output=True).stdout.strip(),
        "selection": artifact_ref(runner.SELECTION),
        "video_index": artifact_ref(BASE / "VIDEO_INDEX.json"),
        "quality_audit": artifact_ref(BASE / "QUALITY_AUDIT.json"),
        "readme": artifact_ref(readme),
        "worker_results": [artifact_ref(BASE / f"WORKER_{index}.json") for index in range(2)],
        "predecessor_v2_rejection": artifact_ref(BASE.parent / "inward_forearm_reference_v2" / "REJECTION.json"),
        "predecessor_v3_rejection": artifact_ref(BASE.parent / "inward_forearm_reference_v3" / "REJECTION.json"),
        "algorithm_quality_adopted": False,
        "human_review": "PENDING",
    }
    atomic_json(BASE / "RESULT.json", result)
    print(json.dumps({"status": result["status"], "count": result["count"], "task_counts": task_counts}, ensure_ascii=False))
    return 0 if machine_pass else 3


if __name__ == "__main__":
    raise SystemExit(main())
