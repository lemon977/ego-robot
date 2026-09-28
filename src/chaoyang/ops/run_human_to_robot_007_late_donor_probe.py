"""Frozen all-session donor search for two previously unsupported 007 frames."""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import time

import cv2
import numpy as np

from chaoyang.governance.common import REPO_ROOT, artifact_ref, load_json
from chaoyang.ops.run_human_to_robot_007_device_donor_probe import (
    HUMAN, RAW, ROLE, fit_heldout_homography, table_mask,
)

TASK = "human_to_robot_007_late_donor_probe_20260923"
ATTEMPT = REPO_ROOT / f"_run/current/{TASK}/attempts/attempt_0001"
DEST = ATTEMPT / "lanes/scene/late_donor_probe_v1"
PRIOR = REPO_ROOT / "_run/current/human_to_robot_007_attachment_clean_canary_20260923/attempts/attempt_0001/lanes/scene/attachment_clean_window_v1"
OBJECT = REPO_ROOT / "_run/current/four_stream_visual_delivery_v5/attempts/attempt_0001/lanes/scene/masks_007_v1/task_object"
TARGETS = (192, 196)
DONORS = tuple(frame for frame in range(378) if not 181 <= frame <= 196)


def image(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    value = cv2.imread(str(path), flags)
    if value is None:
        raise FileNotFoundError(path)
    return value


def source_masks(frame: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = image(RAW / f"{frame:06d}.png")
    human = image(HUMAN / f"{frame:06d}.png", cv2.IMREAD_UNCHANGED)
    role = image(ROLE / f"{frame:06d}.png", cv2.IMREAD_UNCHANGED)
    if raw.shape != (960, 1280, 3) or human.shape != (960, 1280) or role.shape != (960, 1280):
        raise RuntimeError(f"DONOR_DOMAIN:{frame}")
    return raw, human, role


def target_input(frame: int) -> dict:
    local = frame - 181
    raw, _, role = source_masks(frame)
    write = image(PRIOR / "input/write" / f"{local:06d}.png", cv2.IMREAD_UNCHANGED) > 0
    protect = image(PRIOR / "input/protect" / f"{local:06d}.png", cv2.IMREAD_UNCHANGED) > 0
    object_labels = image(OBJECT / f"{frame:06d}.png", cv2.IMREAD_UNCHANGED)
    if any(mask.shape != (960, 1280) for mask in (write, protect, object_labels)):
        raise RuntimeError(f"TARGET_MASK_DOMAIN:{frame}")
    eligible = write & ~protect & (object_labels == 0)
    target_table = table_mask(raw, write | protect | (object_labels > 0) | (role > 0))
    return {"raw": raw, "eligible": eligible, "table": target_table,
            "count": np.zeros((960, 1280), dtype=np.uint16),
            "refs": {"raw": artifact_ref(RAW / f"{frame:06d}.png"),
                     "write": artifact_ref(PRIOR / "input/write" / f"{local:06d}.png"),
                     "protect": artifact_ref(PRIOR / "input/protect" / f"{local:06d}.png"),
                     "object_labels": artifact_ref(OBJECT / f"{frame:06d}.png")}}


def coverage(count: np.ndarray, eligible: np.ndarray) -> dict:
    total = int(np.count_nonzero(eligible))
    return {"eligible_write_pixels": total,
            "zero_donor_pixels": int(np.count_nonzero(eligible & (count == 0))),
            "one_or_more_donor_pixels": int(np.count_nonzero(eligible & (count >= 1))),
            "three_or_more_donor_pixels": int(np.count_nonzero(eligible & (count >= 3)))}


def main() -> int:
    index = load_json(REPO_ROOT / "tasks/current/INDEX.json")
    packets = index.get("task_packets", [])
    if len(packets) != 1 or packets[0].get("task_id") != TASK or not packets[0].get("execution_allowed"):
        raise RuntimeError("TASK_NOT_ROUTABLE")
    if DEST.exists():
        raise FileExistsError(DEST)
    old = load_json(PRIOR / "RESULT.json")
    if old.get("source_frames") != list(range(181, 197)) or old.get("model_returncode") != 0:
        raise RuntimeError("FROZEN_WRITE_INPUT_CHANGED")
    cv2.setNumThreads(2)
    inputs = {frame: target_input(frame) for frame in TARGETS}
    DEST.mkdir(parents=True)
    started = time.monotonic()
    rows = []
    complete = True
    for donor_frame in DONORS:
        if time.monotonic() - started > 3500:
            complete = False
            break
        donor, human, role = source_masks(donor_frame)
        donor_table = table_mask(donor, (human > 0) | (role > 0), donor_frame=donor_frame)
        row = {"donor_frame": donor_frame,
               "source": {"raw": artifact_ref(RAW / f"{donor_frame:06d}.png"),
                          "human": artifact_ref(HUMAN / f"{donor_frame:06d}.png"),
                          "role": artifact_ref(ROLE / f"{donor_frame:06d}.png")},
               "targets": {}}
        for target_frame in TARGETS:
            target = inputs[target_frame]
            matrix, metric = fit_heldout_homography(donor, target["raw"], donor_table, target["table"])
            supported = 0
            if matrix is not None:
                warped_table = cv2.warpPerspective(donor_table, matrix, (1280, 960), flags=cv2.INTER_NEAREST) > 0
                valid = warped_table & target["eligible"]
                supported = int(np.count_nonzero(valid))
                target["count"][valid] += 1
            row["targets"][str(target_frame)] = {**metric, "eligible_table_pixels": supported}
        rows.append(row)
    summaries = []
    for target_frame in TARGETS:
        target = inputs[target_frame]
        output = DEST / f"DONOR_COUNT_{target_frame}.png"
        if not cv2.imwrite(str(output), target["count"]):
            raise RuntimeError(f"COUNT_WRITE_FAILED:{target_frame}")
        summaries.append({"target_frame": target_frame, "inputs": target["refs"],
                          "count_image": artifact_ref(output),
                          **coverage(target["count"], target["eligible"]),
                          "geometry_qualified_donors": sum(bool(row["targets"][str(target_frame)]["geometry_pass"]) for row in rows),
                          "donors_with_eligible_table": sum(row["targets"][str(target_frame)]["eligible_table_pixels"] > 0 for row in rows)})
    result = {"schema_version": "HUMAN_TO_ROBOT_007_LATE_DONOR_PROBE_V1", "task_id": TASK,
              "session_id": "get_potato_chips_0915_007", "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
              "targets": list(TARGETS), "donor_frames": list(DONORS), "donors_evaluated": len(rows),
              "full_cohort_evaluated": complete, "elapsed_seconds": time.monotonic() - started,
              "rows": rows, "summaries": summaries, "execution": "EXECUTED" if complete else "PAUSED_BUDGET",
              "structure": "PASS" if complete else "INCOMPLETE_BUDGET", "quality": "DIAGNOSTIC_ONLY",
              "adoption": "NOT_ADOPTED", "claim_limit": "Same-session planar donor availability only; no pixel synthesis, Clean quality, background truth or product adoption.",
              "training_eligible": False, "control_ground_truth": False, "physical_deployable": False,
              "external_metric_authority": False}
    with (DEST / "RESULT.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"task_id": TASK, "donors_evaluated": len(rows), "summaries": summaries}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
