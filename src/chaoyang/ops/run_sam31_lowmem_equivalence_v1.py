#!/usr/bin/env python3
# ruff: noqa: E402
"""One-factor, leased SAM3.1 input-frame CPU-offload equivalence canary.

This is research evidence only.  It does not chunk/reseed tracking, modify
thresholds, or grant Mask authority.  Run only through the central GPU wrapper.
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
import resource
import subprocess
import sys
import threading
import time

import cv2
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SAM_ROOT = ROOT / "vendor/SAM3"
if str(SAM_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM_ROOT))

from chaoyang.pipeline import sam31_compat_adapter_v1
from chaoyang.ops import run_clean_layered_sam31_t0_visual_v1 as sam_helpers
from chaoyang.governance.common import artifact_ref, atomic_json, now_iso
from chaoyang.ops.run_sam31_semantic_batchflag_ablation import collect_variant
from chaoyang.ops.run_sam31_semantic_full_route_diagnostic import ANNOTATIONS, BOOTSTRAP, CHECKPOINT
from chaoyang.ops.run_sam31_state_route_diagnostic import _render

BASE = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_unblock_handoff_v2/sam31_batchflag_ablation/attempts/attempt_0002"
SCENARIOS = {
    "repeated_seed_8": 8,
    "real_prefix_16": 16,
}


def gpu_used_mib() -> int | None:
    sample = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False,
    )
    try:
        return int(sample.stdout.splitlines()[0].strip()) if sample.returncode == 0 else None
    except (IndexError, ValueError):
        return None


def require_central_lease(timeout_s: float = 8.0) -> None:
    lease_path = ROOT / "_run/current/GPU_LEASE.json"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if lease_path.is_file():
            lease = json.loads(lease_path.read_text(encoding="utf-8"))
            if lease.get("status") == "ACQUIRED" and lease.get("gpu_process_pid") == os.getpid():
                return
        time.sleep(0.2)
    raise RuntimeError("central GPU lease is not registered for this process")


def compare_rows(actual: list[dict], baseline: list[dict]) -> dict:
    if len(actual) != len(baseline):
        return {"equivalent": False, "reason": "FRAME_COUNT_MISMATCH"}
    mismatches: list[dict] = []
    for index, (new, old) in enumerate(zip(actual, baseline, strict=True)):
        fields = ["frame_index", "present", "area_pixels", "output_object_ids", "reason", "mask_sha256"]
        differing = [field for field in fields if new.get(field) != old.get(field)]
        new_score, old_score = new.get("score"), old.get("score")
        if (new_score is None) != (old_score is None) or (
            new_score is not None and abs(float(new_score) - float(old_score)) > 1e-6
        ):
            differing.append("score")
        if differing:
            mismatches.append({"frame_index": index, "fields": differing})
    return {
        "equivalent": not mismatches,
        "frames_compared": len(actual),
        "mask_sha_equal_count": sum(a.get("mask_sha256") == b.get("mask_sha256") for a, b in zip(actual, baseline, strict=True)),
        "score_abs_tolerance": 1e-6,
        "mismatches": mismatches,
    }


def run(output: Path, scenario: str) -> int:
    require_central_lease()
    output.mkdir(parents=True, exist_ok=False)
    count = SCENARIOS[scenario]
    input_dir = (BASE / "inputs" / scenario).resolve(strict=True)
    baseline_path = (BASE / f"{scenario}_is_last_batch_true.json").resolve(strict=True)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if int(baseline["frames"]) != count or len(baseline["rows"]) != count:
        raise RuntimeError("frozen baseline frame count differs")
    images = [input_dir / f"{frame:05d}.png" for frame in range(count)]
    if any(not image.is_file() for image in images):
        raise RuntimeError("frozen input sequence incomplete")
    frozen = json.loads(ANNOTATIONS.read_text(encoding="utf-8"))["annotations"]["poker015"]
    bootstrap = json.loads(BOOTSTRAP.read_text(encoding="utf-8"))["tasks"]["poker015"]["object"]
    accepted_id = int(bootstrap["selected_object_id"])
    x0, y0, x1, y1 = frozen["box_xyxy"]
    box = [[x0 / 1280, y0 / 960, (x1 - x0 + 1) / 1280, (y1 - y0 + 1) / 960]]

    samples: list[int] = []
    stop_sampling = threading.Event()

    def sample_memory() -> None:
        while not stop_sampling.is_set():
            value = gpu_used_mib()
            if value is not None:
                samples.append(value)
            stop_sampling.wait(0.5)

    started = time.perf_counter()
    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()
    adapter = None
    try:
        adapter, build_evidence = sam31_compat_adapter_v1.build_pinned_adapter(
            official_code_root=SAM_ROOT, checkpoint_path=CHECKPOINT,
        )
        model = adapter.model
        torch.cuda.reset_peak_memory_stats()
        state = model.init_state(
            resource_path=str(input_dir), offload_video_to_cpu=True,
            async_loading_frames=False,
        )
        try:
            frame_tensor_device = str(state["input_batch"].img_batch.tensors.device)
            if frame_tensor_device != "cpu":
                raise RuntimeError(f"offload flag did not retain input frames on CPU: {frame_tensor_device}")
            _, prompt = model.add_prompt(
                inference_state=state, frame_idx=0,
                text_str="a purple-backed playing card", boxes_xywh=box,
                box_labels=[1], output_prob_thresh=0.5,
            )
            _, _, seed_ids = sam_helpers.normalize(prompt, 960, 1280)
            if accepted_id not in [int(value) for value in seed_ids]:
                raise RuntimeError(f"frozen detector ID missing on seed: {seed_ids}")
            route, action_ids = model.parse_action_history_for_propagation(state)
            rows, masks = collect_variant(
                model, state, accepted_id=accepted_id,
                frame_count=count, is_last_batch=True,
            )
            torch.cuda.synchronize()
            comparison = compare_rows(rows, baseline["rows"])
            specs = {
                "path": input_dir,
                "frames": count,
                "source_loader": lambda index: cv2.imread(str(images[index]), cv2.IMREAD_COLOR),
            }
            video = _render(output, f"{scenario}_CPUFrames", specs, [("CPU帧卸载", {"rows": rows}, masks)])
            payload = {
                "schema_version": "chaoyang-sam31-lowmem-equivalence-v1",
                "task_id": "research_sam31_lowmem_equivalence",
                "status": "PASSED_EQUIVALENCE_DIAGNOSTIC" if comparison["equivalent"] else "NON_EQUIVALENT_CANDIDATE",
                "created_at": now_iso(),
                "scenario": scenario,
                "frame_count": count,
                "changed_factor": "offload_video_to_cpu: false -> true",
                "unchanged": "checkpoint, prompt, resolution, dtype, object ID, semantic-full route, is_last_batch=true",
                "route": route,
                "action_object_ids": action_ids,
                "frame_tensor_device": frame_tensor_device,
                "comparison": comparison,
                "memory": {
                    "torch_peak_allocated_mib_after_build_reset": round(torch.cuda.max_memory_allocated() / 1048576, 3),
                    "torch_peak_reserved_mib_after_build_reset": round(torch.cuda.max_memory_reserved() / 1048576, 3),
                    "physical_used_samples_mib": {"min": min(samples) if samples else None, "max": max(samples) if samples else None, "sample_count": len(samples)},
                    "cpu_maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                },
                "wall_seconds": round(time.perf_counter() - started, 3),
                "inputs": {
                    "baseline": artifact_ref(baseline_path),
                    "annotations": artifact_ref(ANNOTATIONS),
                    "bootstrap": artifact_ref(BOOTSTRAP),
                    "checkpoint": artifact_ref(CHECKPOINT),
                    "frames": [artifact_ref(image.resolve(strict=True)) for image in images],
                },
                "build_evidence": build_evidence,
                "video": artifact_ref(video),
                "authority_promoted": False,
                "training_eligible": False,
                "claim_limit": "Short offline equivalence only; no full-session memory proof, segmentation accuracy, causal input or Mask authority.",
            }
            atomic_json(output / "RESULT.json", payload)
            return 0 if comparison["equivalent"] else 2
        finally:
            state.clear()
    except Exception as error:
        atomic_json(output / "RESULT.json", {
            "schema_version": "chaoyang-sam31-lowmem-equivalence-v1",
            "task_id": "research_sam31_lowmem_equivalence",
            "status": "FAILED_RUNTIME_FINAL",
            "created_at": now_iso(),
            "scenario": scenario,
            "error_type": type(error).__name__,
            "error": str(error),
            "authority_promoted": False,
            "training_eligible": False,
            "claim_limit": "Failed short canary; no equivalence or memory-saving conclusion.",
        })
        raise
    finally:
        stop_sampling.set()
        sampler.join(timeout=3)
        if adapter is not None:
            close = getattr(adapter, "close", None)
            if callable(close):
                close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="repeated_seed_8")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    return run(args.output_root.resolve(), args.scenario)


if __name__ == "__main__":
    raise SystemExit(main())
