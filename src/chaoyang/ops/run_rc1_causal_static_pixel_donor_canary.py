#!/usr/bin/env python3
"""Bounded CPU-only causal static-pixel donor challenger (development only).

No old Clean RGB or bidirectional donor map is read. A pixel is proposed only
after two earlier raw observations agree, no task object has occupied that
coordinate so far, and the current visible ring supports a static scene.
The proposal is not a semantic support-surface label or Clean authority.
"""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from chaoyang.ops.audit_clean_contact_protection_successor_v1 import proposed, unions
from chaoyang.governance.common import artifact_ref, atomic_json, now_iso
from chaoyang.ops.run_clean20_cpu_contract_closure_r22 import CASES


def _disk(mask: np.ndarray, radius: int) -> np.ndarray:
    return cv2.dilate(mask.astype(np.uint8),
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                (2 * radius + 1, 2 * radius + 1))).astype(bool)


def _age(current: int, previous: np.ndarray, limit: int) -> np.ndarray:
    return (previous >= 0) & ((current - previous) <= limit)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=tuple(CASES), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=64)
    args = parser.parse_args()
    out = args.output_root.resolve()
    out.mkdir(parents=True, exist_ok=False)
    manifest_path = CASES[args.case]["manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest["frames"][: min(args.max_frames, len(manifest["frames"]))]
    if not rows:
        raise RuntimeError("empty source manifest")

    shape = (960, 1280)
    last_rgb = np.zeros((*shape, 3), np.uint8)
    second_rgb = np.zeros_like(last_rgb)
    last_index = np.full(shape, -1, np.int16)
    second_index = np.full(shape, -1, np.int16)
    ever_object = np.zeros(shape, bool)
    pixel_grid_y, pixel_grid_x = np.indices(shape)
    holdout_grid = (pixel_grid_x % 32 < 8) & (pixel_grid_y % 32 < 8)
    del pixel_grid_y, pixel_grid_x
    ring_outer = 25
    ring_inner = 4
    max_age = 60
    last_agreement_linf = 10
    ring_p90_limit = 15
    ring_min_pixels = 500
    review_frames = {0, 10, 20, 40, min(len(rows) - 1, 63)}
    review: list[np.ndarray] = []
    frame_rows: list[dict] = []
    total_candidates = 0
    total_write = 0
    holdout_abs: list[np.ndarray] = []

    for frame_id, row in enumerate(rows):
        raw_path = Path(row["source_rgb"]["path"])
        raw = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
        if raw is None or raw.shape != (*shape, 3):
            raise RuntimeError(f"Raw decode/shape failed: {raw_path}")
        human, tracker, obj = unions(row)
        if human.shape != shape or tracker.shape != shape or obj.shape != shape:
            raise RuntimeError(f"mask shape mismatch at frame {frame_id}")
        ever_object |= obj
        possible_object = _disk(ever_object, 5)
        visible_safe = ~(human | tracker | possible_object)
        write = proposed(human, tracker, obj)
        if np.any(write & obj):
            raise RuntimeError("write domain intersects current visible object")

        recent = _age(frame_id, last_index, max_age) & _age(frame_id, second_index, max_age)
        stable = np.max(np.abs(last_rgb.astype(np.int16) - second_rgb.astype(np.int16)),
                        axis=2) <= last_agreement_linf
        prior = recent & stable & ~possible_object
        ring = _disk(write, ring_outer) & ~_disk(write, ring_inner) & visible_safe
        ring &= _age(frame_id, last_index, max_age)
        ring_delta = np.max(np.abs(raw.astype(np.int16) - last_rgb.astype(np.int16)), axis=2)
        ring_count = int(ring.sum())
        ring_p90 = float(np.percentile(ring_delta[ring], 90)) if ring_count else None
        ring_pass = ring_count >= ring_min_pixels and ring_p90 is not None and ring_p90 <= ring_p90_limit
        candidate = write & prior if ring_pass else np.zeros(shape, bool)

        holdout = holdout_grid & visible_safe & prior
        if ring_pass and np.any(holdout):
            holdout_abs.append(np.max(np.abs(raw[holdout].astype(np.int16)
                                              - last_rgb[holdout].astype(np.int16)), axis=1))
        total_candidates += int(candidate.sum())
        total_write += int(write.sum())
        frame_rows.append({
            "frame_id": frame_id,
            "write_pixels": int(write.sum()),
            "causal_static_candidate_pixels": int(candidate.sum()),
            "possible_object_excluded_pixels": int((write & possible_object).sum()),
            "ring_visible_pixels": ring_count,
            "ring_p90_linf_rgb": ring_p90,
            "ring_pass": bool(ring_pass),
            "holdout_visible_pixels": int(holdout.sum()) if ring_pass else 0,
        })

        if frame_id in review_frames:
            donor_preview = raw.copy()
            donor_preview[write] = (127, 127, 127)
            donor_preview[candidate] = last_rgb[candidate]
            overlay = raw.copy()
            overlay[write & ~candidate] = (0.45 * overlay[write & ~candidate]
                                          + 0.55 * np.array([0, 220, 255])).astype(np.uint8)
            overlay[candidate] = (0.45 * overlay[candidate]
                                  + 0.55 * np.array([0, 255, 0])).astype(np.uint8)
            overlay[possible_object] = (0.7 * overlay[possible_object]
                                        + 0.3 * np.array([255, 0, 255])).astype(np.uint8)
            cv2.putText(overlay, f"frame={frame_id} donor={candidate.sum()} unknown={write.sum()-candidate.sum()}",
                        (20, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            review.append(cv2.hconcat([cv2.resize(raw, (640, 480)),
                                       cv2.resize(overlay, (640, 480)),
                                       cv2.resize(donor_preview, (640, 480))]))

        # Only after evaluating t may its Raw pixels become a donor for t+1.
        update = visible_safe
        second_rgb[update] = last_rgb[update]
        second_index[update] = last_index[update]
        last_rgb[update] = raw[update]
        last_index[update] = frame_id

    montage_path = out / f"{args.case}_causal_static_donor_diagnostic.png"
    cv2.imwrite(str(montage_path), cv2.vconcat(review))
    holdout_values = np.concatenate(holdout_abs) if holdout_abs else np.empty(0, np.int16)
    result = {
        "schema_version": "chaoyang-rc1-causal-static-pixel-donor-canary-v1",
        "created_at": now_iso(),
        "case": args.case,
        "session_id": CASES[args.case]["session"],
        "status": "PASSED_DEVELOPMENT_DIAGNOSTIC" if total_candidates else "FAILED_QUALITY_NO_CANDIDATES",
        "frames": len(rows),
        "candidate_pixels": total_candidates,
        "write_pixels": total_write,
        "candidate_fraction_of_write": total_candidates / total_write if total_write else None,
        "holdout_pixels": int(holdout_values.size),
        "holdout_linf_p50": float(np.percentile(holdout_values, 50)) if holdout_values.size else None,
        "holdout_linf_p95": float(np.percentile(holdout_values, 95)) if holdout_values.size else None,
        "frame_rows": frame_rows,
        "algorithm": {
            "two_past_visible_observations": True,
            "same_coordinate_rgb_linf_at_most": last_agreement_linf,
            "past_observation_max_age_frames": max_age,
            "ever_seen_task_object_dilation_px": 5,
            "current_visible_ring_outer_px": ring_outer,
            "current_visible_ring_inner_px": ring_inner,
            "current_visible_ring_min_pixels": ring_min_pixels,
            "current_visible_ring_rgb_linf_p90_at_most": ring_p90_limit,
            "no_future_frames_read": True,
            "old_clean_rgb_read": False,
        },
        "authority_promoted": False,
        "training_eligible": False,
        "claim_limit": "Static same-coordinate donor challenger only. No independent semantic or pixel Gold; zero authority; failed/unknown pixels remain UNKNOWN.",
        "inputs": {"source_manifest": artifact_ref(manifest_path),
                   "raw_frame0": artifact_ref(Path(rows[0]["source_rgb"]["path"])),
                   "raw_last_frame": artifact_ref(Path(rows[-1]["source_rgb"]["path"]))},
        "outputs": {"montage": artifact_ref(montage_path)},
        "code": artifact_ref(Path(__file__)),
    }
    atomic_json(out / "RESULT.json", result)
    print(json.dumps({"case": args.case, "status": result["status"],
                      "frames": len(rows), "candidate_pixels": total_candidates,
                      "write_pixels": total_write, "holdout_p95": result["holdout_linf_p95"],
                      "result_path": str(out / "RESULT.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
