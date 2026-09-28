"""Fixed 16-frame 007 table-only multi-donor diagnostic, never a Clean adoption."""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import warnings

import cv2
import numpy as np

from chaoyang.governance.common import REPO_ROOT, artifact_ref, load_json
from chaoyang.ops.run_human_to_robot_007_device_donor_probe import (
    HUMAN, RAW, ROLE, fit_heldout_homography, table_mask,
)
from chaoyang.ops.run_human_to_robot_007_attachment_clean_canary import DEST as PRIOR

TASK = "human_to_robot_007_multidonor_table_canary_20260923"
ATTEMPT = REPO_ROOT / f"_run/current/{TASK}/attempts/attempt_0001"
DEST = ATTEMPT / "lanes/scene/multidonor_table_window_v1"
DONORS = (144, 152, 160, 176, 184, 216, 248, 264, 272, 280, 288, 296, 312, 320, 336, 344, 352, 360, 368)
OBJECT = REPO_ROOT / "_run/current/four_stream_visual_delivery_v5/attempts/attempt_0001/lanes/scene/masks_007_v1/task_object"
MIN_DONORS = 3
MAX_PHOTOMETRIC_OFFSET = 25.0


def image(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    value = cv2.imread(str(path), flags)
    if value is None:
        raise FileNotFoundError(path)
    return value


def color_offset(donor: np.ndarray, target: np.ndarray, valid: np.ndarray,
                 target_table: np.ndarray) -> tuple[np.ndarray | None, dict]:
    """Frozen checkerboard split prevents fitting and scoring on the same pixels."""
    yy, xx = np.indices(valid.shape)
    holdout = ((xx // 16 + yy // 16) % 4 == 0)
    train = valid & target_table & ~holdout
    test = valid & target_table & holdout
    if int(train.sum()) < 1000 or int(test.sum()) < 250:
        return None, {"train_pixels": int(train.sum()), "holdout_pixels": int(test.sum()),
                      "reject": "INSUFFICIENT_VISIBLE_TABLE_PHOTOMETRIC_EVIDENCE"}
    delta = np.median(target[train].astype(np.float32) - donor[train].astype(np.float32), axis=0)
    delta = np.clip(delta, -MAX_PHOTOMETRIC_OFFSET, MAX_PHOTOMETRIC_OFFSET)
    corrected = np.clip(donor.astype(np.float32) + delta, 0, 255).astype(np.uint8)
    error = np.max(np.abs(corrected[test].astype(np.int16) - target[test].astype(np.int16)), axis=1)
    return corrected, {"train_pixels": int(train.sum()), "holdout_pixels": int(test.sum()),
                       "offset_bgr": [float(x) for x in delta],
                       "holdout_max_channel_p50": float(np.percentile(error, 50)),
                       "holdout_max_channel_p95": float(np.percentile(error, 95)),
                       "reject": None}


def median_supported(colors: list[np.ndarray], masks: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if not colors or len(colors) != len(masks):
        raise ValueError("EMPTY_OR_UNPAIRED_DONORS")
    count = np.sum(np.stack(masks, axis=0).astype(np.uint8), axis=0)
    pixels = np.stack([np.where(mask[:, :, None], color, np.nan)
                       for color, mask in zip(colors, masks)], axis=0).astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median = np.nanmedian(pixels, axis=0)
    median[~np.isfinite(median)] = 0
    return np.clip(median, 0, 255).astype(np.uint8), count


def main() -> int:
    index = load_json(REPO_ROOT / "tasks/current/INDEX.json")
    packets = index.get("task_packets", [])
    if len(packets) != 1 or packets[0].get("task_id") != TASK or not packets[0].get("execution_allowed"):
        raise RuntimeError("TASK_NOT_ROUTABLE")
    if DEST.exists():
        raise FileExistsError(DEST)
    prior_result = load_json(PRIOR / "RESULT.json")
    if (prior_result.get("task_id") != "human_to_robot_007_attachment_clean_canary_20260923"
            or prior_result.get("source_frames") != list(range(181, 197))
            or prior_result.get("model_returncode") != 0):
        raise RuntimeError("PRIOR_CHANGED_INPUT_CLEAN_NOT_FROZEN")
    donor_manifest = load_json(REPO_ROOT / "_run/current/human_to_robot_007_table_donor_expansion_20260923/attempts/attempt_0001/lanes/scene/table_donor_expansion_v1/RESULT.json")
    qualified = tuple(row["donor_frame"] for row in donor_manifest["rows"] if row["geometry_pass"])
    if qualified != DONORS:
        raise RuntimeError("DONOR_COHORT_CHANGED")
    cv2.setNumThreads(2)
    donor_inputs = []
    for frame in DONORS:
        donor = image(RAW / f"{frame:06d}.png")
        human = image(HUMAN / f"{frame:06d}.png", cv2.IMREAD_UNCHANGED) > 0
        role = image(ROLE / f"{frame:06d}.png", cv2.IMREAD_UNCHANGED) > 0
        if donor.shape != (960, 1280, 3) or human.shape != (960, 1280) or role.shape != human.shape:
            raise RuntimeError(f"DONOR_DOMAIN:{frame}")
        donor_inputs.append((frame, donor, table_mask(donor, human | role, donor_frame=frame)))
    DEST.mkdir(parents=True)
    for sub in ("candidate", "support", "count"):
        (DEST / sub).mkdir()
    review_path = DEST / "007_MULTIDONOR_TABLE_OLD_NEW_16FRAME_REVIEW.mp4"
    writer = cv2.VideoWriter(str(review_path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (2560, 480))
    if not writer.isOpened():
        raise RuntimeError("VIDEO_WRITER_NOT_OPEN")
    rows = []
    try:
        for local, frame in enumerate(range(181, 197)):
            raw_path = RAW / f"{frame:06d}.png"
            raw = image(raw_path)
            previous = image(PRIOR / "clean" / f"{local:06d}.png")
            write = image(PRIOR / "input/write" / f"{local:06d}.png", cv2.IMREAD_UNCHANGED) > 0
            protect = image(PRIOR / "input/protect" / f"{local:06d}.png", cv2.IMREAD_UNCHANGED) > 0
            object_mask = image(OBJECT / f"{frame:06d}.png", cv2.IMREAD_UNCHANGED) > 0
            if raw.shape != (960, 1280, 3) or previous.shape != raw.shape or any(
                    value.shape != raw.shape[:2] for value in (write, protect, object_mask)):
                raise RuntimeError(f"TARGET_DOMAIN:{frame}")
            if np.any(previous[~write] != raw[~write]) or np.any(previous[protect] != raw[protect]):
                raise RuntimeError(f"PRIOR_BYTE_EXACT_INVARIANT:{frame}")
            target_table = table_mask(raw, write | object_mask) > 0
            colors, valid_masks, donor_rows = [], [], []
            for donor_frame, donor, donor_table in donor_inputs:
                if donor_frame == frame:
                    donor_rows.append({"donor_frame": donor_frame, "reject": "SAME_FRAME_NOT_INDEPENDENT"})
                    continue
                h, metric = fit_heldout_homography(donor, raw, donor_table, target_table.astype(np.uint8) * 255)
                if h is None:
                    donor_rows.append({"donor_frame": donor_frame, **metric, "reject": "GEOMETRY_GATE"})
                    continue
                warped = cv2.warpPerspective(donor, h, (1280, 960))
                warped_table = cv2.warpPerspective(donor_table, h, (1280, 960), flags=cv2.INTER_NEAREST) > 0
                corrected, photo = color_offset(warped, raw, warped_table, target_table)
                if corrected is None:
                    donor_rows.append({"donor_frame": donor_frame, **metric, **photo})
                    continue
                colors.append(corrected)
                valid_masks.append(warped_table)
                donor_rows.append({"donor_frame": donor_frame, **metric, **photo,
                                   "table_write_pixels": int(np.count_nonzero(warped_table & write))})
            if len(colors) >= MIN_DONORS:
                median, count = median_supported(colors, valid_masks)
            else:
                median = np.zeros_like(raw)
                count = np.zeros(write.shape, dtype=np.uint8)
            support = (count >= MIN_DONORS) & write & ~protect & ~object_mask
            candidate = previous.copy()
            candidate[support] = median[support]
            if np.any(candidate[~write] != raw[~write]) or np.any(candidate[protect] != raw[protect]):
                raise RuntimeError(f"CANDIDATE_BYTE_EXACT_INVARIANT:{frame}")
            paths = {
                "candidate": DEST / "candidate" / f"{local:06d}.png",
                "support": DEST / "support" / f"{local:06d}.png",
                "count": DEST / "count" / f"{local:06d}.png",
            }
            for name, value in (("candidate", candidate), ("support", support.astype(np.uint8) * 255),
                                ("count", count)):
                if not cv2.imwrite(str(paths[name]), value):
                    raise RuntimeError(f"OUTPUT_WRITE_FAILED:{frame}:{name}")
            overlay = raw.copy()
            overlay[support] = (.55 * overlay[support] + .45 * np.array([0, 255, 0])).astype(np.uint8)
            panels = [cv2.resize(v, (640, 480), interpolation=cv2.INTER_AREA)
                      for v in (raw, previous, overlay, candidate)]
            canvas = np.concatenate(panels, axis=1)
            for x, label in ((8, "RAW"), (648, "PRIOR REJECTED"), (1288, "TABLE SUPPORT"), (1928, "MULTIDONOR CANDIDATE")):
                cv2.putText(canvas, label, (x, 28), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2)
            cv2.putText(canvas, str(frame), (8, 460), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2)
            writer.write(canvas)
            rows.append({"frame_id": frame, "raw": artifact_ref(raw_path),
                         "candidate": artifact_ref(paths["candidate"]),
                         "support": artifact_ref(paths["support"]),
                         "count": artifact_ref(paths["count"]),
                         "donor_geometry_pass": sum(bool(r.get("geometry_pass")) for r in donor_rows),
                         "donor_photo_pass": len(colors), "table_replaced_pixels": int(support.sum()),
                         "write_pixels": int(write.sum()), "protected_pixels": int(protect.sum()),
                         "task_object_pixels": int(object_mask.sum()), "donors": donor_rows})
    finally:
        writer.release()
    capture = cv2.VideoCapture(str(review_path))
    decoded = 0
    while capture.read()[0]:
        decoded += 1
    capture.release()
    if decoded != 16:
        raise RuntimeError(f"REVIEW_DECODE:{decoded}")
    result = {
        "schema_version": "HUMAN_TO_ROBOT_007_MULTIDONOR_TABLE_CANARY_V1",
        "task_id": TASK, "session_id": "get_potato_chips_0915_007",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_frames": list(range(181, 197)), "donor_frames": list(DONORS),
        "execution": "EXECUTED", "structure": "PASS", "quality": "PENDING_INDEPENDENT_VISUAL_REVIEW",
        "adoption": "CANDIDATE_ONLY", "prior_clean": artifact_ref(PRIOR / "RESULT.json"),
        "donor_manifest": artifact_ref(REPO_ROOT / "_run/current/human_to_robot_007_table_donor_expansion_20260923/attempts/attempt_0001/lanes/scene/table_donor_expansion_v1/RESULT.json"),
        "review": {**artifact_ref(review_path), "decoded_frames": decoded}, "rows": rows,
        "claim_limit": "Real 16-frame table-only multi-donor diagnostic; no moving-object, complete Clean or product authority.",
        "training_eligible": False, "control_ground_truth": False,
        "physical_deployable": False, "external_metric_authority": False,
    }
    (DEST / "RESULT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "EXECUTED_CANDIDATE", "result": str(DEST / "RESULT.json"),
                      "review": str(review_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
