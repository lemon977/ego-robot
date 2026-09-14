#!/usr/bin/env python3
"""CPU-only structural/motion audit for exact78 heldout RTC predictions.

This tool never changes inference outputs.  It writes a machine-readable metric
summary and an evenly sampled contact sheet for each completed session.  Visual
PASS/HOLD remains a human review decision and is deliberately not inferred here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar(value: np.ndarray) -> Any:
    item = np.asarray(value).item()
    return item.item() if hasattr(item, "item") else item


def distribution(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(values):
        return {"count": 0, "median": None, "p95": None, "max": None}
    return {
        "count": int(len(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def decoded_positions(action: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    values = np.asarray(action, dtype=np.float64)
    return values[..., :6].reshape(*values.shape[:-1], 2, 3) * std + mean


def to_world(positions: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    rotation = np.asarray(c2w, dtype=np.float64)[..., :3, :3]
    translation = np.asarray(c2w, dtype=np.float64)[..., :3, 3]
    # positions: replans, execution steps, physical hands, xyz
    return np.einsum("nij,nthj->nthi", rotation, positions) + translation[:, None, None, :]


def adjacent_steps(values: np.ndarray, *, flatten_time: bool) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if flatten_time:
        # Keep physical hands independent while joining consecutive RTC prefixes.
        values = values.reshape(-1, values.shape[-2], 3)
        return np.linalg.norm(np.diff(values, axis=0), axis=-1).reshape(-1)
    return np.linalg.norm(np.diff(values, axis=1), axis=-1).reshape(-1)


def q_violation_ratio(action: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    q = np.asarray(action, dtype=np.float64)[..., 18:62]
    lo = np.asarray(lower, dtype=np.float64).reshape(*([1] * (q.ndim - 1)), -1)
    hi = np.asarray(upper, dtype=np.float64).reshape(*([1] * (q.ndim - 1)), -1)
    return float(np.mean((q < lo - 1e-6) | (q > hi + 1e-6)))


def make_contact_sheet(video: Path, output: Path, samples: int) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(video))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if count < 1 or width < 1 or height < 1:
        capture.release()
        raise RuntimeError(f"video is not decodable: {video}")
    indices = np.unique(np.linspace(0, count - 1, min(samples, count), dtype=np.int64))
    frames: list[np.ndarray] = []
    read_indices: list[int] = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if not ok:
            continue
        cv2.rectangle(frame, (0, height - 30), (185, height), (0, 0, 0), -1)
        cv2.putText(frame, f"video frame {int(index):03d}", (8, height - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        frames.append(frame)
        read_indices.append(int(index))
    capture.release()
    if len(frames) < min(12, count):
        raise RuntimeError(f"fewer than 12 review frames decoded: {video}")
    thumb_width = 480
    thumb_height = max(1, round(height * thumb_width / width))
    thumbs = [cv2.resize(frame, (thumb_width, thumb_height), interpolation=cv2.INTER_AREA)
              for frame in frames]
    columns = 4
    rows = (len(thumbs) + columns - 1) // columns
    blank = np.zeros_like(thumbs[0])
    thumbs.extend([blank] * (rows * columns - len(thumbs)))
    sheet = np.vstack([np.hstack(thumbs[row * columns:(row + 1) * columns])
                       for row in range(rows)])
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), sheet):
        raise RuntimeError(f"failed to write contact sheet: {output}")
    return {
        "frame_count": count,
        "fps": fps,
        "width": width,
        "height": height,
        "reviewed_frame_indices": read_indices,
        "reviewed_frame_count": len(read_indices),
        "contact_sheet": str(output),
        "contact_sheet_sha256": sha256(output),
    }


def audit_session(session_dir: Path, stats: dict[str, Any], review_dir: Path,
                  samples: int) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    npz = session_dir / "heldout_checkpoint_predictions.npz"
    video = session_dir / "heldout_checkpoint_prediction.mp4"
    manifest = session_dir / "heldout_prediction_manifest.json"
    if not (npz.is_file() and video.is_file() and manifest.is_file()):
        raise FileNotFoundError(f"incomplete heldout session: {session_dir}")
    mean = np.asarray(stats["pos"]["mean"], dtype=np.float64)
    std = np.asarray(stats["pos"]["std"], dtype=np.float64)
    with np.load(npz, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    full = arrays["checkpoint_full_plan_action"]
    raw_prefix = arrays.get("raw_model_prefix", arrays["causally_executed_prefix"])
    safe_prefix = arrays["causally_executed_prefix"]
    c2w = arrays["observation_c2w"]
    raw_full_position = decoded_positions(full, mean, std)
    raw_position = decoded_positions(raw_prefix, mean, std)
    safe_position = decoded_positions(safe_prefix, mean, std)
    raw_world = to_world(raw_position, c2w)
    safe_world = to_world(safe_position, c2w)
    raw_full_steps = adjacent_steps(raw_full_position, flatten_time=False)
    raw_world_steps = adjacent_steps(raw_world, flatten_time=True)
    safe_world_steps = adjacent_steps(safe_world, flatten_time=True)
    numeric_keys = [key for key, value in arrays.items()
                    if np.issubdtype(value.dtype, np.number)]
    finite = all(np.isfinite(arrays[key]).all() for key in numeric_keys)
    lower = arrays.get("joint_lower")
    upper = arrays.get("joint_upper")
    q = {}
    if lower is not None and upper is not None:
        q = {
            "raw_full_joint_limit_violation_ratio": q_violation_ratio(full, lower, upper),
            "raw_prefix_joint_limit_violation_ratio": q_violation_ratio(raw_prefix, lower, upper),
            "safe_prefix_joint_limit_violation_ratio": q_violation_ratio(safe_prefix, lower, upper),
        }
    video_info = make_contact_sheet(
        video, review_dir / f"{session_dir.name}_review16.png", samples,
    )
    manifest_payload = json.loads(manifest.read_text())
    result = {
        "session": session_dir.name,
        "schema_version": str(scalar(arrays.get("schema_version", np.asarray("missing")))),
        "npz_sha256": sha256(npz),
        "video_sha256": sha256(video),
        "manifest_sha256": sha256(manifest),
        "all_numeric_arrays_finite": finite,
        "shapes": {
            "checkpoint_full_plan_action": list(full.shape),
            "raw_model_prefix": list(raw_prefix.shape),
            "causally_executed_prefix": list(safe_prefix.shape),
        },
        "attestation": {
            "contains_ground_truth": bool(scalar(arrays["contains_ground_truth"])),
            "contains_heldout_robot_q": bool(scalar(arrays["contains_heldout_robot_q"])),
            "execution_safety_applied": bool(scalar(arrays.get("execution_safety_applied", np.asarray(False)))),
            "state_feedback_policy": str(scalar(arrays["state_feedback_policy"])),
            "physical_to_human_side": arrays.get("physical_to_human_side", np.asarray([])).astype(str).tolist(),
        },
        "raw_h50_wrist_step_m": distribution(raw_full_steps),
        "raw_prefix_world_wrist_step_m": distribution(raw_world_steps),
        "safe_prefix_world_wrist_step_m": distribution(safe_world_steps),
        "raw_prefix_world_step_fraction_gt_0p12_m": float(np.mean(raw_world_steps > 0.12)),
        "safe_prefix_world_step_fraction_gt_0p0101_m": float(np.mean(safe_world_steps > 0.0101)),
        "raw_to_safe_position_delta_m": distribution(np.linalg.norm(raw_world - safe_world, axis=-1)),
        "joint_limits": q,
        "video_review": video_info,
        "manifest_execution_safety": manifest_payload.get("execution_safety"),
        "manifest_side_mapping": manifest_payload.get("side_mapping"),
    }
    pools = {
        "raw_full_steps": raw_full_steps,
        "raw_world_steps": raw_world_steps,
        "safe_world_steps": safe_world_steps,
        "raw_safe_delta": np.linalg.norm(raw_world - safe_world, axis=-1).reshape(-1),
    }
    return result, pools


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=16)
    args = parser.parse_args()
    if args.samples < 12:
        parser.error("--samples must be at least 12")
    return args


def main() -> int:
    args = parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"audit output must be new or empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    stats = json.loads((args.run_root / "dataset_stats.json").read_text())
    session_dirs = sorted(path.parent for path in args.prediction_root.glob(
        "*/heldout_checkpoint_predictions.npz"
    ))
    if not session_dirs:
        raise RuntimeError(f"no complete sessions under {args.prediction_root}")
    sessions = []
    pooled: dict[str, list[np.ndarray]] = {
        "raw_full_steps": [], "raw_world_steps": [],
        "safe_world_steps": [], "raw_safe_delta": [],
    }
    for session_dir in session_dirs:
        result, values = audit_session(session_dir, stats, args.output / "review", args.samples)
        sessions.append(result)
        for key in pooled:
            pooled[key].append(values[key])
    aggregate = {key: distribution(np.concatenate(values)) for key, values in pooled.items()}
    aggregate.update({
        "all_sessions_finite": all(item["all_numeric_arrays_finite"] for item in sessions),
        "total_video_frames_reviewed": sum(item["video_review"]["reviewed_frame_count"] for item in sessions),
        "raw_prefix_world_step_fraction_gt_0p12_m": float(np.mean(np.concatenate(pooled["raw_world_steps"]) > 0.12)),
        "safe_prefix_world_step_fraction_gt_0p0101_m": float(np.mean(np.concatenate(pooled["safe_world_steps"]) > 0.0101)),
    })
    q_keys = sorted({key for item in sessions for key in item["joint_limits"]})
    aggregate["joint_limit_ratios_session_mean"] = {
        key: float(np.mean([item["joint_limits"][key] for item in sessions]))
        for key in q_keys
    }
    payload = {
        "schema_version": "exact78-heldout-quality-numeric-audit-v1",
        "created_at": datetime.now().astimezone().replace(microsecond=0).isoformat(),
        "prediction_root": str(args.prediction_root.resolve()),
        "run_root": str(args.run_root.resolve()),
        "status": "NUMERIC_AND_REVIEW_SHEETS_COMPLETE_HUMAN_VISUAL_DECISION_REQUIRED",
        "session_count": len(sessions),
        "aggregate": aggregate,
        "sessions": sessions,
    }
    target = args.output / "NUMERIC_AUDIT.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(target), "sessions": len(sessions), "aggregate": aggregate},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
