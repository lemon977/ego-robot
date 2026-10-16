"""Build deterministic contact sheets and a review queue for Robot100 V6."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from chaoyang.governance.common import artifact_ref, atomic_json
from chaoyang.ops import run_robot100_inward_successor_v6 as runner

BASE = runner.OUTPUT.parent
OUTPUT = BASE / "visual_audit_v1"


def selected_frames(arrays: dict[str, np.ndarray], limit: int = 9) -> list[int]:
    count = len(arrays["frame_id"])
    if count <= 0:
        return []
    projected = arrays["posture_projected"].astype(bool)
    alpha = arrays["posture_projection_alpha"]
    residual = np.nan_to_num(arrays["position_residual_mm"], nan=-1.0)
    valid = arrays["wrist_valid"].astype(bool)
    step = np.zeros(count, dtype=np.float64)
    if count > 1:
        delta = np.max(np.abs(np.diff(arrays["q_arm"], axis=0)), axis=(1, 2))
        edge = arrays["time_edge_valid"][1:] & np.any(valid[1:] & valid[:-1], axis=1)
        step[1:] = np.where(edge, delta, 0.0)
    candidates = [0, count - 1, int(np.argmax(alpha)), int(np.argmax(residual)), int(np.argmax(step))]
    union = np.flatnonzero(np.any(projected, axis=1))
    if union.size:
        candidates.extend([int(union[0]), int(union[len(union) // 2]), int(union[-1])])
    candidates.extend(np.linspace(0, count - 1, limit, dtype=int).tolist())
    chosen: list[int] = []
    for frame in candidates:
        if 0 <= frame < count and frame not in chosen:
            chosen.append(frame)
        if len(chosen) == limit:
            break
    return sorted(chosen)


def decode_selected(video: Path, frames: list[int]) -> dict[int, np.ndarray]:
    wanted = set(frames)
    decoded: dict[int, np.ndarray] = {}
    capture = cv2.VideoCapture(str(video))
    index = 0
    while wanted:
        ok, image = capture.read()
        if not ok:
            break
        if index in wanted:
            decoded[index] = image
            wanted.remove(index)
        index += 1
    capture.release()
    if wanted:
        raise RuntimeError(f"CONTACT_SHEET_DECODE:{video}:{sorted(wanted)}")
    return decoded


def contact_sheet(session: str, arrays: dict[str, np.ndarray], video: Path, frames: list[int]) -> np.ndarray:
    decoded = decode_selected(video, frames)
    tiles: list[np.ndarray] = []
    for frame in frames:
        image = cv2.resize(decoded[frame], (426, 240), interpolation=cv2.INTER_AREA)
        projected = arrays["posture_projected"][frame].astype(bool).tolist()
        alpha = arrays["posture_projection_alpha"][frame].tolist()
        color = (0, 0, 255) if any(projected) else (0, 160, 0)
        cv2.rectangle(image, (0, 0), (425, 239), color, 4)
        label = f"{session} f={frame} proj={list(map(int, projected))} a={alpha[0]:.3f}/{alpha[1]:.3f}"
        cv2.rectangle(image, (0, 207), (425, 239), (0, 0, 0), -1)
        cv2.putText(image, label, (7, 229), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(image)
    blank = np.zeros((240, 426, 3), dtype=np.uint8)
    while len(tiles) < 9:
        tiles.append(blank.copy())
    return np.vstack([np.hstack(tiles[i : i + 3]) for i in range(0, 9, 3)])


def main() -> int:
    index = json.loads((BASE / "VIDEO_INDEX.json").read_text(encoding="utf-8"))
    if index.get("count") != 100 or len(index.get("rows", [])) != 100:
        raise RuntimeError("VIDEO_INDEX_COUNT")
    sheets = OUTPUT / "contact_sheets"
    sheets.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in index["rows"]:
        session = item["session_id"]
        result = json.loads(Path(item["result"]["path"]).read_text(encoding="utf-8"))
        motion = Path(result["motion"]["path"])
        video = Path(result["video"]["path"])
        with np.load(motion, allow_pickle=False) as source:
            arrays = {name: source[name] for name in source.files}
        frames = selected_frames(arrays)
        sheet_path = sheets / f"{session}.jpg"
        if not cv2.imwrite(str(sheet_path), contact_sheet(session, arrays, video, frames), [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise RuntimeError(f"CONTACT_SHEET_WRITE:{sheet_path}")
        valid = arrays["wrist_valid"].astype(bool)
        projected = arrays["posture_projected"].astype(bool)
        raw_fail = valid & ~arrays["tolerance_pass_raw"].astype(bool)
        max_position = float(np.nanmax(arrays["position_residual_mm"]))
        max_rotation = float(np.nanmax(arrays["rotation_residual_deg"]))
        max_alpha = float(np.max(arrays["posture_projection_alpha"]))
        delta = np.max(np.abs(np.diff(arrays["q_arm"], axis=0)), axis=(1, 2)) if len(valid) > 1 else np.zeros(0)
        max_step = float(np.max(delta)) if delta.size else 0.0
        fraction = float(projected.sum() / max(1, valid.sum()))
        score = 2.0 * fraction + max_alpha + max_position / 300.0 + max_rotation / 90.0 + max_step / 0.75 + float(raw_fail.sum())
        rows.append({
            "session_id": session,
            "task": item["task"],
            "frames": int(item["frames"]),
            "selected_frames": frames,
            "posture_projected_left_right": projected.sum(axis=0).astype(int).tolist(),
            "posture_projected_fraction": fraction,
            "raw_tracking_fail_left_right": raw_fail.sum(axis=0).astype(int).tolist(),
            "maximum_projection_alpha": max_alpha,
            "maximum_display_position_residual_mm": max_position,
            "maximum_display_rotation_residual_deg": max_rotation,
            "maximum_display_joint_step_rad": max_step,
            "review_priority_score": score,
            "contact_sheet": artifact_ref(sheet_path),
        })
    rows.sort(key=lambda row: (-row["review_priority_score"], row["session_id"]))
    atomic_json(OUTPUT / "VISUAL_REVIEW_QUEUE.json", {
        "schema_version": "CHAOYANG_ROBOT100_VISUAL_REVIEW_QUEUE_V1",
        "status": "CONTACT_SHEETS_COMPLETE_PENDING_HUMAN_REVIEW",
        "count": len(rows),
        "ranking_policy": "projection fraction, projection alpha, wrist residual, joint step, and raw tracking failure",
        "rows": rows,
    })
    atomic_json(OUTPUT / "RESULT.json", {
        "schema_version": "CHAOYANG_ROBOT100_VISUAL_AUDIT_RESULT_V1",
        "status": "COMPLETE_PENDING_HUMAN_REVIEW",
        "count": len(rows),
        "review_queue": artifact_ref(OUTPUT / "VISUAL_REVIEW_QUEUE.json"),
        "source_result": artifact_ref(BASE / "RESULT.json"),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
