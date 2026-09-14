#!/usr/bin/env python3
"""Measure the fastest stable micro-batch on the actual CUDA device."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from preprocess.retarget_labels.schema import EMBODIMENTS  # noqa: E402
from training.FlowMatchingModel import FlowMatchingModel  # noqa: E402
from utils.atomic_io import atomic_write_json  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_model(config: dict) -> FlowMatchingModel:
    return FlowMatchingModel(
        single_hand=False, pred_horizon=50, max_ict=int(config["max_ict"]),
        img_size=tuple(config["image_size"]), patch_size=int(config["patch_size"]),
        vision_embed_dim=int(config["vision_embed_dim"]),
        num_decoder_layers=int(config["num_decoder_layers"]),
        num_heads=int(config["num_heads"]), mlp_ratio=float(config["mlp_ratio"]),
        dropout=float(config["dropout"]), use_pcd_features=False,
        use_aux_obj_dynamics=False, use_aux_visual_foresight=False,
        use_aux_temporal_contrastive=False,
        use_region_attn=bool(config["use_region_attn"]),
        use_pre_norm=bool(config["use_pre_norm"]),
        use_ctx_norm=bool(config["use_ctx_norm"]),
        use_done_in_flow=bool(config["use_done_in_flow"]),
        hand_action_representation=config["hand_action_representation"],
    )


def one_batch(model: FlowMatchingModel, batch: int, device: torch.device) -> dict[str, torch.Tensor]:
    state_dim = model.robot_state_dim
    return {
        "x_rgb": torch.randn(batch, 3, *model.img_size, device=device),
        "x_ict": torch.randn(batch, 8, 29, device=device),
        "ict_mask": torch.ones(batch, 8, dtype=torch.bool, device=device),
        "x_robot_state": torch.randn(batch, 2, state_dim, device=device),
        "robot_state_mask": torch.ones(batch, 2, dtype=torch.bool, device=device),
        "anchor_uv": torch.full((batch, 2), 0.5, device=device),
        "x_t": torch.randn(batch, 50, model.action_dim, device=device),
        "t": torch.rand(batch, 1, device=device),
    }


def benchmark(config: dict, batch: int, steps: int) -> dict[str, object]:
    device = torch.device("cuda:0")
    torch.cuda.empty_cache()
    model = make_model(config).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=True)
    data = one_batch(model, batch, device)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    def train_step() -> None:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=dtype):
            prediction = model(**data)["v_pred"]
            loss = prediction.square().mean()
        loss.backward()
        optimizer.step()
    # Exclude one-off allocator, cuBLAS and kernel startup costs from throughput.
    for _ in range(2):
        train_step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for _ in range(steps):
        train_step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated(device)
    result = {
        "batch_size": batch, "steps": steps, "elapsed_s": elapsed,
        "samples_per_second": batch * steps / elapsed,
        "peak_allocated_bytes": peak,
        "peak_allocated_gib": peak / 2**30,
    }
    del data, optimizer, model
    torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embodiment", required=True, choices=sorted(EMBODIMENTS))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--candidates", type=int, nargs="+", default=[8, 16, 32, 64, 96, 128, 192, 256])
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--max-memory-fraction", type=float, default=0.88)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker-candidate", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_candidate is None and args.output is None:
        parser.error("--output is required for the autotune orchestrator")
    if args.steps < 1:
        parser.error("--steps must be positive")
    if not 0 < args.max_memory_fraction < 1:
        parser.error("--max-memory-fraction must lie in (0, 1)")
    if any(value < 1 for value in args.candidates):
        parser.error("all batch candidates must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this execution container")
    config_path = args.config.resolve()
    if not config_path.is_relative_to((ROOT / "cfg" / "training").resolve()):
        raise ValueError("autotune config must be a reviewed versioned training config")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["hand_action_representation"] != EMBODIMENTS[args.embodiment].representation:
        raise ValueError("config/embodiment mismatch")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    if args.worker_candidate is not None:
        print(json.dumps(benchmark(config, args.worker_candidate, args.steps)))
        return 0
    properties = torch.cuda.get_device_properties(0)
    limit = int(properties.total_memory * args.max_memory_fraction)
    results, failures = [], []
    for candidate in args.candidates:
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()),
             "--embodiment", args.embodiment, "--config", str(config_path),
             "--steps", str(args.steps), "--worker-candidate", str(candidate)],
            cwd=ROOT, text=True, capture_output=True, check=False,
        )
        if completed.returncode:
            failures.append({
                "batch_size": candidate,
                "error": f"isolated worker exit {completed.returncode}",
                "stderr_tail": completed.stderr[-2000:],
            })
            continue
        result = json.loads(completed.stdout.strip().splitlines()[-1])
        if int(result["peak_allocated_bytes"]) > limit:
            failures.append({"batch_size": candidate, "error": "exceeds memory headroom"})
            continue
        results.append(result)
        print(json.dumps(result), flush=True)
    if not results:
        raise RuntimeError("No candidate batch passed")
    # Throughput wins; use the larger batch only when throughput is effectively tied.
    selected = max(results, key=lambda item: (float(item["samples_per_second"]), int(item["batch_size"])))
    report = {
        "schema_version": "humanego-cuda-autotune-v1",
        "embodiment": args.embodiment, "config": str(config_path.resolve()),
        "config_sha256": sha256(config_path), "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda, "device": properties.name,
        "device_total_memory_bytes": properties.total_memory,
        "max_memory_fraction": args.max_memory_fraction,
        "precision": "bf16" if torch.cuda.is_bf16_supported() else "fp16",
        "tf32": True, "results": results, "failures": failures,
        "selected_batch_size": int(selected["batch_size"]),
        "selected_samples_per_second": float(selected["samples_per_second"]),
    }
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
