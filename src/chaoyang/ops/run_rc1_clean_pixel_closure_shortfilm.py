#!/usr/bin/env python3
"""Replay a bounded Clean challenger with sparse per-pixel lineage and video.

This is an OFFLINE_VISUAL diagnostic. Existing role masks are not proven causal,
and a static RGB donor is not a semantic background proof. Unknown is retained.
"""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from chaoyang.ops.audit_clean_contact_protection_successor_v1 import mask, proposed, unions
from chaoyang.governance.common import artifact_ref, atomic_json, now_iso
from chaoyang.ops.run_clean20_cpu_contract_closure_r22 import CASES


ROOT = Path(__file__).resolve().parents[3]
BASELINE_RESULT = {
    "Poker245": ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260909_exact78_current_baseline_batch_v1/clean_expanded_role_v3_prepare_v1/propainter_v1/play_cards_0903_245/RESULT.json",
    "Chips039": ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260913_exact78_v3_wave_clean_v1/propainter_v1/get_potato_chips_0902_039/RESULT.json",
}


def dilate(value: np.ndarray, radius: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(value.astype(np.uint8), kernel).astype(bool)


def small(image: np.ndarray) -> np.ndarray:
    return cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA)


def panel(image: np.ndarray, caption: str) -> np.ndarray:
    value = small(image)
    cv2.rectangle(value, (0, 0), (640, 38), (0, 0, 0), -1)
    cv2.putText(value, caption, (8, 27), cv2.FONT_HERSHEY_SIMPLEX, .69, (255, 255, 255), 2)
    return value


def pack(value: np.ndarray) -> np.ndarray:
    return np.packbits(value.reshape(-1).astype(np.uint8), bitorder="little")


def run(case: str, output: Path, max_frames: int) -> None:
    output.mkdir(parents=True, exist_ok=False)
    map_dir = output / "pixel_maps"
    map_dir.mkdir()
    source_manifest = CASES[case]["manifest"]
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    rows = manifest["frames"][:max_frames]
    if not rows:
        raise RuntimeError("empty frozen frame manifest")
    baseline_result_path = BASELINE_RESULT[case]
    baseline_result = json.loads(baseline_result_path.read_text(encoding="utf-8"))
    baseline_ref = baseline_result["artifacts"]["clean_synthetic_master"]
    baseline_path = Path(baseline_ref["path"])
    if artifact_ref(baseline_path) != baseline_ref:
        raise RuntimeError("baseline video SHA collision")
    baseline = cv2.VideoCapture(str(baseline_path))
    if not baseline.isOpened():
        raise RuntimeError("baseline video cannot open")
    shape = (960, 1280)
    last_rgb = np.zeros((*shape, 3), np.uint8)
    second_rgb = np.zeros_like(last_rgb)
    last_index = np.full(shape, -1, np.int16)
    second_index = np.full(shape, -1, np.int16)
    ever_object = np.zeros(shape, bool)
    video_path = output / f"{CASES[case]['session']}_Clean像素来源对照_OFFLINE_短片.mp4"
    video = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 960))
    if not video.isOpened():
        raise RuntimeError("diagnostic video writer failed")
    maps: list[dict] = []
    metrics: list[dict] = []
    for t, row in enumerate(rows):
        raw_path = Path(row["source_rgb"]["path"])
        if artifact_ref(raw_path) != row["source_rgb"]:
            raise RuntimeError(f"raw SHA mismatch at {t}")
        raw = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
        ok, old = baseline.read()
        if raw is None or raw.shape != (*shape, 3) or not ok or old.shape != raw.shape:
            raise RuntimeError(f"decode/shape mismatch at {t}")
        human, wearable, obj = unions(row)
        old_write = mask(row["clean_removal_object_protected"])
        m_remove = human | wearable
        m_flow = np.zeros(shape, bool)  # No optical flow is run in this challenger.
        m_write = proposed(human, wearable, obj)
        ever_object |= obj
        possible_object = dilate(ever_object, 5)
        visible_safe = ~(m_remove | possible_object)
        recent = ((last_index >= 0) & (second_index >= 0)
                  & (t - last_index <= 60) & (t - second_index <= 60))
        stable = np.max(np.abs(last_rgb.astype(np.int16) - second_rgb.astype(np.int16)), axis=2) <= 10
        ring = dilate(m_write, 25) & ~dilate(m_write, 4) & visible_safe & (last_index >= 0)
        ring_diff = np.max(np.abs(raw.astype(np.int16) - last_rgb.astype(np.int16)), axis=2)
        ring_p90 = float(np.percentile(ring_diff[ring], 90)) if ring.any() else None
        ring_pass = int(ring.sum()) >= 500 and ring_p90 is not None and ring_p90 <= 15
        candidate = m_write & recent & stable & ~possible_object if ring_pass else np.zeros(shape, bool)
        reason = np.zeros(shape, np.uint8)
        reason[m_write] = 1  # No two stable prior observations.
        reason[m_write & possible_object] = 2  # Task object possible; never background.
        if not ring_pass:
            reason[m_write & ~possible_object] = 3  # Current visible-ring check failed.
        reason[candidate] = 0
        y, x = np.nonzero(candidate)
        donor_frame = last_index[y, x]
        if donor_frame.size and (donor_frame >= t).any():
            raise RuntimeError("future donor detected")
        view = raw.copy()
        unknown = m_write & ~candidate
        view[unknown] = (127, 127, 127)
        view[y, x] = last_rgb[y, x]
        if np.any(view[~m_write] != raw[~m_write]) or np.any(view[obj] != raw[obj]):
            raise RuntimeError("M_write outside or visible object byte-exact violation")
        # Every changed pixel has a source frame/xy. A candidate remains
        # semantically unverified and therefore is NOT a training Clean pixel.
        source_class = np.zeros(shape, np.uint8)
        source_class[unknown] = 7
        source_class[candidate] = 3
        source_class[obj] = 2
        map_path = map_dir / f"{t:06d}.npz"
        np.savez_compressed(
            map_path, shape=np.asarray(shape), candidate_y=y.astype(np.uint16),
            candidate_x=x.astype(np.uint16), source_frame=donor_frame.astype(np.int16),
            source_y=y.astype(np.uint16), source_x=x.astype(np.uint16),
            transform=np.asarray([1, 0, 0, 0, 1, 0], dtype=np.float32),
            source_class=source_class, reject_reason=reason,
            m_remove=pack(m_remove), m_flow=pack(m_flow), m_write=pack(m_write),
            m_visible_object=pack(obj), m_possible_object=pack(possible_object),
            m_frozen_eval_reference=pack(old_write),
        )
        maps.append({"frame_id": t, "map": artifact_ref(map_path),
                     "raw": row["source_rgb"], "source_frame_min": int(donor_frame.min()) if donor_frame.size else None,
                     "source_frame_max": int(donor_frame.max()) if donor_frame.size else None})
        metrics.append({"frame_id": t, "fixed_eval_reference_pixels": int(old_write.sum()),
                        "m_remove_pixels": int(m_remove.sum()), "m_write_pixels": int(m_write.sum()),
                        "candidate_pixels": int(candidate.sum()), "unknown_pixels": int(unknown.sum()),
                        "current_visible_object_pixels": int(obj.sum()), "changed_visible_object_pixels": 0,
                        "changed_outside_m_write": 0, "ring_p90_linf": ring_p90,
                        "ring_pass": bool(ring_pass)})
        provenance = raw.copy()
        provenance[unknown] = (0, 180, 255)
        provenance[candidate] = (0, 200, 0)
        provenance[obj] = (200, 0, 200)
        frame = cv2.vconcat([
            cv2.hconcat([panel(raw, f"Raw t={t}"), panel(old, "Old Clean MP4 visual baseline")]),
            cv2.hconcat([panel(view, "Static donor / gray UNKNOWN"),
                         panel(provenance, "Green donor / orange UNKNOWN / magenta object")]),
        ])
        video.write(frame)
        update = visible_safe
        second_rgb[update] = last_rgb[update]
        second_index[update] = last_index[update]
        last_rgb[update] = raw[update]
        last_index[update] = t
    video.release()
    baseline.release()
    subprocess.run(["ffmpeg", "-nostdin", "-threads", "2", "-v", "error", "-xerror", "-i",
                    str(video_path), "-f", "null", "-"], check=True, stdout=subprocess.DEVNULL)
    result = {
        "schema_version": "rc1-clean-pixel-closure-shortfilm-v1", "created_at": now_iso(),
        "status": "PASSED_OFFLINE_DIAGNOSTIC", "case": case,
        "session_id": CASES[case]["session"], "frames": len(rows),
        "input_mode": "OFFLINE_VISUAL", "causal_candidate": False,
        "training_eligible": False, "authority_promoted": False,
        "reason_not_causal": "The frozen role/object Mask and old baseline MP4 have not passed producer-level prefix tests.",
        "reason_not_clean_authority": "Static donor has no independent support-surface or hidden-object semantic proof; candidate coverage is not Clean quality improvement.",
        "pixel_source_codes": {"0": "OBSERVED_RAW_UNCHANGED", "2": "OBSERVED_RAW_VISIBLE_OBJECT",
                               "3": "PAST_SAME_XY_SEMANTIC_UNVERIFIED", "7": "UNKNOWN"},
        "rejection_codes": {"0": "NOT_REJECTED", "1": "TWO_PRIOR_OBSERVATIONS_MISSING_OR_UNSTABLE",
                            "2": "POSSIBLE_TASK_OBJECT", "3": "VISIBLE_RING_REJECTED"},
        "metrics": {"candidate_pixels": sum(m["candidate_pixels"] for m in metrics),
                    "unknown_pixels": sum(m["unknown_pixels"] for m in metrics),
                    "changed_outside_m_write": 0, "changed_visible_object_pixels": 0},
        "frame_metrics": metrics, "pixel_maps": maps,
        "inputs": {"manifest": artifact_ref(source_manifest), "old_clean_result": artifact_ref(baseline_result_path),
                   "old_clean_mp4": baseline_ref},
        "outputs": {"comparison_video": artifact_ref(video_path)},
        "code": artifact_ref(Path(__file__)),
        "claim_limit": "64-frame fixed-reference source-lineage experiment, not hidden-pixel accuracy, fresh ProPainter, causal model input, or Clean authority.",
    }
    atomic_json(output / "RESULT.json", result)
    print(json.dumps({"case": case, "session_id": result["session_id"], "frames": len(rows),
                      "candidate_pixels": result["metrics"]["candidate_pixels"],
                      "unknown_pixels": result["metrics"]["unknown_pixels"],
                      "video": str(video_path)}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=tuple(CASES), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=64)
    args = parser.parse_args()
    run(args.case, args.output_root.resolve(), args.max_frames)


if __name__ == "__main__":
    main()
