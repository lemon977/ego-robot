#!/usr/bin/env python3
"""Development-only SAM3.1 native-route regression against frozen Poker B output.

The old B masks select the seed and provide an internal comparison, never Gold.
"""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for path in (ROOT, ROOT / "vendor/SAM3"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from chaoyang.pipeline import sam31_compat_adapter_v1
from chaoyang.ops import run_clean_layered_sam31_t0_visual_v1 as helpers
from chaoyang.governance.common import artifact_ref, atomic_json, now_iso, validate_artifact_ref
from chaoyang.ops.run_sam31_semantic_full_route_diagnostic import CHECKPOINT, collect_selected_object
from chaoyang.ops.run_sam31_state_route_diagnostic import _render


SESSIONS = {
    "play_cards_0901_001": 645,
    "play_cards_0901_005": 520,
}
RAW_ROOT = Path("/mnt/data/egodata/datasets/ego/chips_cards_tracker_0901/playing_cards")
BASELINE_ROOT = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260909_exact78_current_baseline_batch_v1/mask_task_object_identity_v1/poker"


def _read_mask(reference: dict) -> np.ndarray:
    errors = validate_artifact_ref(reference)
    if errors:
        raise RuntimeError("frozen B mask closure failed: " + "; ".join(errors))
    mask = cv2.imread(reference["path"], cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != (960, 1280):
        raise RuntimeError("frozen B mask decode/shape failed")
    return mask > 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-id", choices=tuple(SESSIONS), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--offload-video-to-cpu", action="store_true",
                        help="single-factor frame-storage experiment; does not offload model or object memory")
    args = parser.parse_args()
    session = args.session_id
    count = SESSIONS[session]
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    raw_root = RAW_ROOT / session
    baseline_dir = BASELINE_ROOT / session
    baseline_result_path = baseline_dir / "RESULT.json"
    baseline_result = json.loads(baseline_result_path.read_text(encoding="utf-8"))
    if baseline_result.get("grade") != "B" or baseline_result.get("frame_count") != count:
        raise RuntimeError("frozen B regression identity/count mismatch")
    manifest_ref = baseline_result["artifacts"]["manifest"]
    errors = validate_artifact_ref(manifest_ref)
    if errors:
        raise RuntimeError("frozen B manifest closure failed: " + "; ".join(errors))
    manifest = json.loads(Path(manifest_ref["path"]).read_text(encoding="utf-8"))
    if manifest.get("session") != session or len(manifest["frames"]) != count:
        raise RuntimeError("frozen B manifest frame identity mismatch")
    first_ref = manifest["frames"][0]["physical_instances"]["0"]["mask"]
    first_mask = _read_mask(first_ref)
    ys, xs = np.nonzero(first_mask)
    if len(xs) == 0:
        raise RuntimeError("frozen B initial mask empty")
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    box = [[x0 / 1280, y0 / 960, (x1 - x0 + 1) / 1280, (y1 - y0 + 1) / 960]]
    input_dir = output / "inputs/full_session"
    input_dir.mkdir(parents=True)
    source_paths = []
    for frame in range(count):
        path = (raw_root / "preprocess/all_data" / f"{frame:05d}" / "rgb.png").resolve(strict=True)
        os.symlink(path, input_dir / f"{frame:05d}.png")
        source_paths.append(path)
    adapter, build_evidence = sam31_compat_adapter_v1.build_pinned_adapter(
        official_code_root=ROOT / "vendor/SAM3", checkpoint_path=CHECKPOINT,
    )
    model = adapter.model
    try:
        state = model.init_state(resource_path=str(input_dir), offload_video_to_cpu=args.offload_video_to_cpu,
                                 async_loading_frames=False)
        try:
            _, prompt = model.add_prompt(
                inference_state=state, frame_idx=0, text_str="a purple-backed playing card",
                boxes_xywh=box, box_labels=[1], output_prob_thresh=0.5,
            )
            prompt_masks, prompt_scores, prompt_ids = helpers.normalize(prompt, 960, 1280)
            candidates = []
            for mask, score, object_id in zip(prompt_masks, prompt_scores, prompt_ids, strict=True):
                overlap = np.logical_and(mask, first_mask).sum()
                union = np.logical_or(mask, first_mask).sum()
                candidates.append({"object_id": int(object_id), "score": float(score), "initial_iou_to_frozen_B": float(overlap / union) if union else 0.0})
            candidates.sort(key=lambda item: (-item["initial_iou_to_frozen_B"], item["object_id"]))
            if not candidates or candidates[0]["initial_iou_to_frozen_B"] < 0.3:
                raise RuntimeError("no native detector object matches frozen B seed")
            selected_id = candidates[0]["object_id"]
            route, _ = model.parse_action_history_for_propagation(state)
            if route != "propagation_full":
                raise RuntimeError("native route changed")
            rows, native_masks = collect_selected_object(
                model, state, accepted_id=selected_id, frame_count=count, height=960, width=1280,
            )
        finally:
            state.clear()
    finally:
        close = getattr(adapter, "close", None)
        if callable(close):
            close()
    b_masks: dict[int, np.ndarray] = {}
    comparisons = []
    for frame, baseline_row in enumerate(manifest["frames"]):
        instance = baseline_row["physical_instances"]["0"]
        if instance.get("observed") and instance.get("valid"):
            old = _read_mask(instance["mask"])
            b_masks[frame] = old
            new = native_masks[frame]
            overlap = np.logical_and(old, new).sum()
            union = np.logical_or(old, new).sum()
            comparisons.append({
                "frame_id": frame,
                "frozen_B_area": int(old.sum()),
                "native_area": int(new.sum()),
                "native_present": bool(new.any()),
                "iou_to_frozen_B_internal_only": float(overlap / union) if union else None,
            })
        else:
            b_masks[frame] = np.zeros((960, 1280), dtype=bool)
    packed = output / "PACKED_NATIVE_MASKS.npz"
    np.savez_compressed(packed, frame_ids=np.arange(count, dtype=np.int32), packed=np.stack([np.packbits(native_masks[frame].reshape(-1)) for frame in range(count)]))
    spec = {"frames": count, "source_loader": lambda frame: cv2.imread(str(source_paths[frame]), cv2.IMREAD_COLOR)}
    observed_frames = {item["frame_id"] for item in comparisons}
    video = _render(output, f"{session}_B_vs_native_full", spec, [
        ("冻结旧B内部参考", {"rows": [{"reason": "B_OBSERVED" if frame in observed_frames else "B_INVALID"} for frame in range(count)]}, b_masks),
        ("原生SAM3.1语义Full", {"rows": rows}, native_masks),
    ])
    ious = [item["iou_to_frozen_B_internal_only"] for item in comparisons if item["iou_to_frozen_B_internal_only"] is not None]
    result = {
        "schema_version": "chaoyang-sam31-native-poker-b-regression-v1",
        "task_id": "research_sam31_native_poker_b_regression",
        "session_id": session,
        "created_at": now_iso(),
        "status": "PASSED_INTERNAL_REGRESSION_DIAGNOSTIC",
        "quality_gate_status": "NOT_EVALUATED",
        "resource_mode": {"offload_video_to_cpu": args.offload_video_to_cpu,
                          "offloads_model_or_object_memory": False},
        "authority_promoted": False,
        "training_eligible": False,
        "input_mode": "OFFLINE_FULL_SEQUENCE_DIAGNOSTIC",
        "frames": count,
        "native_present_frames": sum(item["present"] for item in rows),
        "frozen_B_observed_frames": len(comparisons),
        "native_present_on_frozen_B_observed_frames": sum(item["native_present"] for item in comparisons),
        "initial_detector_candidates": candidates,
        "selected_detector_object_id": selected_id,
        "internal_iou_median": float(np.median(ious)) if ious else None,
        "internal_iou_p05": float(np.percentile(ious, 5)) if ious else None,
        "comparisons": comparisons,
        "same_physical_instance_proven": False,
        "pixel_accuracy_proven": False,
        "claim_limit": "Frozen old B is an internal regression reference, not Gold; full-session native output is offline diagnostic and may use future frames.",
        "inputs": {
            "old_B_result": artifact_ref(baseline_result_path),
            "old_B_manifest": manifest_ref,
            "old_B_seed_mask": first_ref,
            "source_rgb_frame0": artifact_ref(source_paths[0]),
            "source_rgb_last_frame": artifact_ref(source_paths[-1]),
            "checkpoint": artifact_ref(CHECKPOINT),
        },
        "outputs": {"review_video": artifact_ref(video), "packed_masks": artifact_ref(packed)},
        "build_evidence": build_evidence,
        "code": artifact_ref(Path(__file__)),
    }
    atomic_json(output / "RESULT.json", result)
    print(json.dumps({"status": result["status"], "session_id": session, "frames": count, "native_present": result["native_present_frames"], "B_observed": len(comparisons), "iou_median_internal": result["internal_iou_median"], "result": artifact_ref(output / "RESULT.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
