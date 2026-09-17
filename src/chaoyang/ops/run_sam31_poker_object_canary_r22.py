#!/usr/bin/env python3
"""Causal, development-only SAM3.1 Poker object canary for Poker015.

The old task-object mask is used only as a frame-0 prompt-selection guide and
as a diagnostic comparison on frames where it published a valid observation.
It is not pixel ground truth.  Propagation is forward-only; no future frame is
consumed.  The task writes one immutable attempt and never updates authority.
"""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
import hashlib
import json
import os
import socket
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chaoyang.pipeline.causal_modal_mask_gpu_adapter_v71 import FileGpuLease
from chaoyang.pipeline import sam31_compat_adapter_v1 as adapter_module
from chaoyang.ops import run_clean_layered_sam31_t0_visual_v1 as sam_helpers

CHECKPOINT = ROOT / "assets/models/checkpoints/sam3.1/sam3.1_multiplex.pt"
CHECKPOINT_SHA = "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6"
CODE_ROOT = ROOT / "vendor/SAM3"
if str(CODE_ROOT) not in sys.path:
    # The pinned compatibility adapter verifies that ``sam3`` resolves inside
    # this exact tree; make the package importable without installing or
    # mutating the third-party checkout.
    sys.path.insert(0, str(CODE_ROOT))
RAW_ROOT = Path("/mnt/data/egodata/datasets/ego/chips_cards_tracker_0901/playing_cards/play_cards_0901_015/preprocess/all_data")
OLD_RESULT = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260909_exact78_current_baseline_batch_v1/mask_task_object_identity_v1/poker/play_cards_0901_015/RESULT.json"
OLD_MANIFEST = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260909_exact78_current_baseline_batch_v1/mask_task_object_identity_v1/poker/play_cards_0901_015/OBJECT_MASK_MANIFEST.json"
EVAL_CONTRACT = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_r22_mask_clean_lane_b_v1/attempts/attempt_0001_cpu_prereq/MASK_CANDIDATE_EVALUATION_REFERENCE.json"
PROMPTS = ("a playing card", "the playing card held by the person", "a purple-backed playing card")


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n"); f.flush(); os.fsync(f.fileno())


def write_text(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8") as f:
        f.write(value); f.flush(); os.fsync(f.fileno())


def read_old_mask(row: dict[str, Any], shape: tuple[int, int]) -> np.ndarray:
    entry = row["physical_instances"]["0"]
    if not entry.get("valid") or not entry.get("mask"):
        return np.zeros(shape, bool)
    value = cv2.imread(entry["mask"]["path"], cv2.IMREAD_GRAYSCALE)
    if value is None or value.shape != shape:
        raise RuntimeError("old object mask decode/shape failure")
    return value > 0


def choose_prompt(adapter: Any, frames: Path, guide: np.ndarray, shape: tuple[int, int]) -> tuple[str, list[dict[str, Any]]]:
    all_rows = []
    for index, prompt in enumerate(PROMPTS):
        session = f"poker015-object-prompt-{index}"
        sam_helpers.start_session(adapter, session, frames)
        try:
            initial = adapter.handle_request({"type": "add_prompt", "session_id": session, "frame_index": 0, "text": prompt, "output_prob_thresh": 0.5})
            masks, scores, ids = sam_helpers.normalize(initial["outputs"], *shape)
            rows = sam_helpers.mask_candidate_rows(masks, scores, ids, guide)
            for row in rows:
                all_rows.append({**row, "prompt": prompt})
        finally:
            adapter.handle_request({"type": "close_session", "session_id": session})
    eligible = [row for row in all_rows if row["guide_intersection_pixels"] > 0]
    if not eligible:
        raise RuntimeError("SAM3.1 prompt sweep found no frame-0 card candidate")
    chosen = max(eligible, key=lambda row: row["selector_score"])
    return str(chosen["prompt"]), all_rows


def write_terminal(output: Path, *, status: str, metrics: dict[str, Any], artifacts: dict[str, Any], error: str | None, lease: dict[str, Any] | None) -> None:
    result = {
        "schema_version": "sam31-poker-object-canary-r22-result-v1", "task_id": "sam31_poker015_object_canary_r22",
        "attempt_id": output.name, "status": status, "artifact_revision": "R7_4_MASK_SAM31_OBJECT_CANARY_1",
        "session": "play_cards_0901_015", "current_baseline": "SAM3.1", "mode": "CAUSAL_FORWARD_ONLY",
        "execution_performed": bool(artifacts), "metrics": metrics, "artifacts": artifacts, "error": error,
        "evaluation_reference": ref(EVAL_CONTRACT), "old_terminal_result": ref(OLD_RESULT), "old_mask_manifest": ref(OLD_MANIFEST),
        "checkpoint": ref(CHECKPOINT), "authority_promoted": False, "training_eligible": False,
        "claim_limit": "Real causal SAM3.1 development canary; old-mask agreement is not Gold accuracy or Mask authority.",
    }
    write_json(output / "RESULT.json", result)
    write_json(output / "METRICS.json", metrics)
    write_json(output / "RUN_RECEIPT.json", {"schema_version": "sam31-poker-object-canary-r22-run-v1", "task_id": result["task_id"], "attempt_id": output.name, "status": status, "created_at": now(), "host": socket.gethostname(), "pid": os.getpid(), "lease": lease, "authority_promoted": False})
    write_json(output / "NEXT_ACTION.json", {"schema_version": "sam31-poker-object-canary-r22-next-v1", "status": status, "next": "Review full-session causal re-entry and run two frozen A/B regressions only if independent review accepts the canary.", "do_not": "Do not call agreement with the old mask accuracy."})
    write_text(output / "DECISION.md", f"# Poker015 SAM3.1 对象 canary\n\n状态：`{status}`。该结果是因果、前向、开发级候选；旧Mask只用于frame-0选择和诊断一致性，不是真值。没有晋升Mask authority。\n")
    pinned = {name: ref(output / name) for name in ("RESULT.json", "METRICS.json", "RUN_RECEIPT.json", "NEXT_ACTION.json", "DECISION.md")}
    pinned.update(artifacts)
    write_json(output / "ARTIFACT_MANIFEST.json", {"schema_version": "sam31-poker-object-canary-r22-artifacts-v1", "task_id": result["task_id"], "status": status, "artifacts": pinned, "authority_promoted": False})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--executor-epoch", type=int, default=1)
    parser.add_argument("--fencing-token", required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if sha(CHECKPOINT) != CHECKPOINT_SHA:
        write_terminal(output, status="UNKNOWN_VERIFICATION_REQUIRED", metrics={}, artifacts={}, error="SAM31_CHECKPOINT_SHA_MISMATCH", lease=None)
        return 3
    old = json.loads(OLD_MANIFEST.read_text(encoding="utf-8"))
    count = len(old["frames"])
    if count != 428:
        raise RuntimeError("frozen Poker015 frame count drift")
    frames = output / "input_frames"
    frames.mkdir()
    for frame_id in range(count):
        source = (RAW_ROOT / f"{frame_id:05d}/rgb.png").resolve(strict=True)
        os.symlink(source, frames / f"{frame_id:05d}.png")
    first = cv2.imread(str(frames / "00000.png"), cv2.IMREAD_COLOR)
    if first is None or first.shape[:2] != (960, 1280):
        raise RuntimeError("Poker015 input image domain mismatch")
    shape = first.shape[:2]
    guide = read_old_mask(old["frames"][0], shape)
    masks = np.zeros((count, *shape), bool)
    prompt_rows: list[dict[str, Any]] = []
    lease_evidence = None
    started = time.perf_counter()
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    guard = FileGpuLease(lease_path=ROOT / "_run/current/GPU_LEASE.json", lock_path=ROOT / "_run/current/GPU_LEASE.lock", task_id="sam31_poker015_object_canary_r22", attempt_id=output.name, gpu_id=0, executor_epoch=args.executor_epoch, fencing_token=args.fencing_token)
    try:
        with guard:
            adapter, build = adapter_module.build_pinned_adapter(official_code_root=CODE_ROOT, checkpoint_path=CHECKPOINT)
            try:
                prompt, prompt_rows = choose_prompt(adapter, frames, guide, shape)
                session = "poker015-object-causal-full"
                sam_helpers.start_session(adapter, session, frames)
                try:
                    initial = adapter.handle_request({"type": "add_prompt", "session_id": session, "frame_index": 0, "text": prompt, "output_prob_thresh": 0.5})
                    initial_masks, scores, ids = sam_helpers.normalize(initial["outputs"], *shape)
                    candidates = sam_helpers.mask_candidate_rows(initial_masks, scores, ids, guide)
                    if not candidates or candidates[0]["guide_intersection_pixels"] <= 0:
                        raise RuntimeError("full-session frame-0 selection failed")
                    object_id = int(candidates[0]["object_id"])
                    masks[0] = initial_masks[int(candidates[0]["instance_index"])]
                    for item in adapter.handle_stream_request({"type": "propagate_in_video", "session_id": session, "propagation_direction": "forward", "start_frame_index": 0, "max_frame_num_to_track": count, "output_prob_thresh": 0.5}):
                        frame_id = int(item["frame_index"])
                        fm, _, fi = sam_helpers.normalize(item["outputs"], *shape)
                        match = np.flatnonzero(fi.astype(np.int64) == object_id)
                        if len(match) == 1:
                            masks[frame_id] = fm[int(match[0])]
                finally:
                    adapter.handle_request({"type": "close_session", "session_id": session})
            finally:
                adapter.predictor.shutdown()
        lease_evidence = {"acquired": {k: v for k, v in (guard.value or {}).items() if k != "fencing_token"}, "released": {k: v for k, v in (guard.released or {}).items() if k != "fencing_token"}, "fencing_token_sha256": hashlib.sha256(args.fencing_token.encode()).hexdigest()}
    except Exception as exc:
        lease_evidence = {
            "released": {
                k: v for k, v in (guard.released or {}).items() if k != "fencing_token"
            },
            "fencing_token_sha256": hashlib.sha256(
                args.fencing_token.encode()
            ).hexdigest(),
        }
        message = str(exc)
        if "GPU lease unavailable" in message:
            terminal_status = "BLOCKED_RESOURCE"
        elif "prompt sweep found no frame-0 card candidate" in message:
            # A successfully executed prompt sweep that finds no eligible task
            # object is a bounded algorithm-quality outcome, not infrastructure
            # failure.  It must not be retried indefinitely.
            terminal_status = "FAILED_QUALITY_C"
        else:
            terminal_status = "FAILED_RUNTIME_FINAL"
        write_terminal(output, status=terminal_status, metrics={}, artifacts={}, error=f"{type(exc).__name__}: {exc}", lease=lease_evidence)
        return 3

    packed_path = output / "play_cards_0901_015_OBJECT_SAM31_CAUSAL_PACKED.npz"
    np.savez_compressed(packed_path, packed=np.packbits(masks.reshape(count, -1), axis=1), frame_count=np.int32(count), height=np.int32(shape[0]), width=np.int32(shape[1]))
    review = output / "play_cards_0901_015_SAM31对象因果canary_全片.mp4"
    writer = cv2.VideoWriter(str(review), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (1280, 480))
    known = overlap = union = post_known = post_present = 0
    for frame_id, row in enumerate(old["frames"]):
        raw = cv2.imread(str(frames / f"{frame_id:05d}.png"), cv2.IMREAD_COLOR)
        old_mask = read_old_mask(row, shape)
        candidate = masks[frame_id]
        if old_mask.any():
            known += 1; overlap += int((old_mask & candidate).sum()); union += int((old_mask | candidate).sum())
            if frame_id >= 3:
                post_known += 1; post_present += int(candidate.any())
        left = cv2.resize(raw, (640, 480), interpolation=cv2.INTER_AREA)
        vis = raw.copy(); vis[old_mask] = (0.45 * vis[old_mask] + 0.55 * np.array([0, 255, 0])).astype(np.uint8); vis[candidate] = (0.45 * vis[candidate] + 0.55 * np.array([255, 0, 255])).astype(np.uint8)
        right = cv2.resize(vis, (640, 480), interpolation=cv2.INTER_AREA)
        canvas = np.hstack((left, right)); cv2.putText(canvas, f"Poker015 frame {frame_id:04d} | green=old diagnostic ref | magenta=SAM3.1 causal", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(canvas)
    writer.release()
    metrics = {
        "schema_version": "sam31-poker-object-canary-r22-metrics-v1", "frame_count": count,
        "candidate_present_frames": int(np.count_nonzero(masks.reshape(count, -1).any(axis=1))),
        "candidate_present_fraction": float(np.mean(masks.reshape(count, -1).any(axis=1))),
        "old_reference_known_frames": known, "old_reference_overlap_iou": float(overlap / union) if union else None,
        "post_loss_old_known_frames": post_known, "candidate_present_on_post_loss_old_known_fraction": float(post_present / post_known) if post_known else None,
        "prompt_sweep": prompt_rows, "wall_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()), "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "claim_limit": "Agreement with predecessor output is diagnostic, not accuracy or identity truth.",
    }
    artifacts = {"packed_masks": ref(packed_path), "full_review": ref(review)}
    write_terminal(output, status="PASSED_DEVELOPMENT", metrics=metrics, artifacts=artifacts, error=None, lease=lease_evidence)
    print(json.dumps({"status": "PASSED_DEVELOPMENT", "result": ref(output / "RESULT.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
