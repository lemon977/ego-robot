#!/usr/bin/env python3
"""Evaluate one trainer snapshot in bounded fresh CUDA subprocesses."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("DISABLE_ADDMM_CUDA_LT", "1")
os.environ.setdefault("TORCH_BLAS_PREFER_CUBLASLT", "0")

from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.FlowMatchingDataloader import FlowMatchingDataloader, MPSSessions  # noqa: E402
from training.FlowMatchingModel import FlowMatchingModel  # noqa: E402
from training.FlowMatchingTrainer import TrainConfig, eval_ode_inference  # noqa: E402
from utils.atomic_io import atomic_write_json  # noqa: E402
from utils.frozen_contract import (  # noqa: E402
    load_verified_torch_checkpoint,
    read_file_reference,
    read_isolated_evaluation_metrics,
    read_runtime_checkpoint_authority,
    validate_file_reference,
    validate_formal_runtime_contract,
)
from utils.source_contract import (  # noqa: E402
    load_eligible68_frozen_split,
    require_manifest_selector_ready,
    validate_frozen_adapters,
    validate_training_run_role_contract,
)


def load_snapshot(path: Path) -> tuple[dict, TrainConfig, dict, dict]:
    path = path.absolute()
    authority = read_runtime_checkpoint_authority(path)
    loaded_path, payload = load_verified_torch_checkpoint(
        authority["checkpoint_ref"],
        allowed_roots=[path.parent],
        label="isolated evaluation snapshot",
    )
    if loaded_path != path or not isinstance(payload, dict):
        raise ValueError("evaluation snapshot payload/path is invalid")
    if not isinstance(payload.get("run_manifest"), dict):
        raise ValueError("evaluation snapshot has no frozen run manifest")
    run_manifest_reference = authority["run_manifest_ref"]
    run_manifest_path, run_manifest_encoded = read_file_reference(
        run_manifest_reference,
        allowed_roots=[Path(run_manifest_reference["path"]).parent],
        label="snapshot external run manifest",
    )
    try:
        external_run_manifest = json.loads(run_manifest_encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("snapshot external run manifest is not valid JSON") from error
    if payload["run_manifest"] != external_run_manifest:
        raise ValueError("snapshot payload/external run manifest mismatch")
    dataset_stats_reference = authority["dataset_stats_ref"]
    stats_path, stats_encoded = read_file_reference(
        dataset_stats_reference,
        allowed_roots=[Path(dataset_stats_reference["path"]).parent],
        label="snapshot external dataset statistics",
    )
    try:
        stats = json.loads(stats_encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("snapshot external dataset statistics are not valid JSON") from error
    if payload.get("dataset_stats_sha256") != dataset_stats_reference.get("sha256"):
        raise ValueError("evaluation snapshot/dataset statistics hash mismatch")
    selector_binding = require_manifest_selector_ready(
        payload["run_manifest"].get("selector_manifest_ref"),
        payload["run_manifest"].get("paired_kept_manifest_ref"),
        artifact_root=payload["run_manifest"].get("artifact_root"),
    )
    cfg = TrainConfig()
    for key, value in payload["cfg"].items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    manifest = payload["run_manifest"]
    expected_run_directory = Path(str(manifest.get("run_directory")))
    if run_manifest_path != expected_run_directory / "run_manifest.json":
        raise ValueError("snapshot external run-manifest path drift")
    if stats_path != expected_run_directory / "dataset_stats.json":
        raise ValueError("snapshot external dataset-statistics path drift")
    split_path = validate_file_reference(
        manifest.get("split_ref", {}),
        allowed_roots=[ROOT / "data_manifests"],
        label="snapshot frozen split",
    )
    sidecar_root = Path(str(cfg.robot_sidecar_root))
    if not sidecar_root.is_absolute():
        sidecar_root = (ROOT / sidecar_root).resolve()
    split = load_eligible68_frozen_split(
        split_path,
        str(manifest.get("embodiment")),
        sidecar_root=sidecar_root,
        verify_sidecars=True,
    )
    validate_training_run_role_contract(manifest, split)
    validate_formal_runtime_contract(
        asdict(cfg),
        manifest,
        split,
        str(manifest.get("embodiment")),
        config_root=ROOT / "cfg" / "training",
    )
    validate_frozen_adapters(
        split,
        split["production_root"],
        list(manifest.get("validation_sessions", [])),
        selector_binding,
    )
    if cfg.img_name != selector_binding["image_name"]:
        raise ValueError("snapshot image selector differs from selector manifest")
    return payload, cfg, stats, selector_binding


def make_dataset(
    cfg: TrainConfig,
    stats: dict,
    cache_json: bool,
    allowed_window_starts,
    selector_records,
    selector_root,
    sidecar_sha256_by_session,
) -> FlowMatchingDataloader:
    return FlowMatchingDataloader(
        sessions=[MPSSessions(path) for path in cfg.MPS_PATHS_EVAL],
        image_size=cfg.image_size, pred_horizon=cfg.pred_horizon,
        single_hand=cfg.single_hand, single_hand_side=cfg.single_hand_side,
        max_ict=cfg.max_ict, img_name=cfg.img_name,
        centric_mode=cfg.centric_mode, frame_mode=cfg.frame_mode,
        action_mode=cfg.action_mode,
        hand_action_representation=cfg.hand_action_representation,
        robot_sidecar_root=cfg.robot_sidecar_root,
        use_pcd_features=cfg.use_pcd_features,
        use_aux_obj_dynamics=cfg.use_aux_obj_dynamics,
        use_aux_visual_foresight=cfg.use_aux_visual_foresight,
        use_aux_temporal_contrastive=cfg.use_aux_temporal_contrastive,
        enable_augmentation=False,
        hand_tracking_method=cfg.hand_tracking_method,
        use_legacy_image_loading=cfg.use_legacy_image_loading,
        use_legacy_rng=cfg.use_legacy_rng,
        cache_json_in_memory=cache_json,
        # Loading only the current chunk on demand is faster than rebuilding
        # the complete 0.66-GiB eval image cache for every CUDA subprocess.
        cache_image_bytes_in_memory=False,
        seed=cfg.seed, stats=stats,
        allowed_window_starts=allowed_window_starts,
        selector_records=selector_records,
        selector_root=selector_root,
        sidecar_sha256_by_session=sidecar_sha256_by_session,
    )


def make_model(cfg: TrainConfig) -> FlowMatchingModel:
    return FlowMatchingModel(
        single_hand=cfg.single_hand, pred_horizon=cfg.pred_horizon,
        max_ict=cfg.max_ict, img_size=cfg.image_size,
        patch_size=cfg.patch_size, vision_embed_dim=cfg.vision_embed_dim,
        num_decoder_layers=cfg.num_decoder_layers, num_heads=cfg.num_heads,
        mlp_ratio=cfg.mlp_ratio, dropout=cfg.dropout,
        horizon_weighting=cfg.model_horizon_weighting,
        horizon_beta=cfg.model_horizon_beta,
        use_pcd_features=cfg.use_pcd_features,
        use_aux_obj_dynamics=cfg.use_aux_obj_dynamics,
        use_aux_visual_foresight=cfg.use_aux_visual_foresight,
        use_aux_temporal_contrastive=cfg.use_aux_temporal_contrastive,
        use_region_attn=cfg.use_region_attn,
        use_pre_norm=cfg.use_pre_norm, use_ctx_norm=cfg.use_ctx_norm,
        use_done_in_flow=cfg.use_done_in_flow,
        hand_action_representation=cfg.hand_action_representation,
    )


def atomic_json(path: Path, value: dict) -> None:
    atomic_write_json(path, value)


def run_worker(
    snapshot: Path,
    output: Path,
    start_batch: int,
    batch_count: int,
    device: str | None = None,
) -> int:
    payload, cfg, stats, selector_binding = load_snapshot(snapshot)
    if device is not None:
        cfg.device = device
    dataset = make_dataset(
        cfg, stats, cache_json=True,
        allowed_window_starts=selector_binding["window_starts"],
        selector_records=selector_binding["selector_records"],
        selector_root=selector_binding["selector_root"],
        sidecar_sha256_by_session=payload["run_manifest"]["sidecars"],
    )
    batch_size = cfg.eval_batch_size or cfg.batch_size
    start_sample = start_batch * batch_size
    stop_sample = min((start_batch + batch_count) * batch_size, len(dataset))
    if start_sample >= stop_sample:
        raise ValueError("empty validation chunk")
    subset = Subset(dataset, range(start_sample, stop_sample))
    subset.pos_std = dataset.pos_std
    loader = DataLoader(
        subset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )
    model = make_model(cfg).to(cfg.device)
    model.load_state_dict(payload["model"], strict=True)
    metrics = eval_ode_inference(model, loader, cfg=cfg, max_batches=None)
    atomic_json(output, metrics)
    print(json.dumps(metrics, indent=2))
    return 0


def merge_chunks(chunks: list[dict]) -> dict:
    if not chunks:
        raise ValueError("isolated validation produced no chunks")
    batch_total = sum(int(item["batches"]) for item in chunks)
    frame_total = sum(int(item["frames"]) for item in chunks)
    batch_weighted = {
        "pos_err_k1_m", "pos_err_kK_m", "rot_err_k1_deg", "rot_err_kK_deg",
        "pos_err_w_m", "rot_err_w_deg", "grasp_f1_k1", "grasp_f1_kK",
        "grasp_f1_w", "joint_mae_k1_normalized", "joint_mae_kK_normalized",
        "joint_mae_w_normalized", "zero_state_ratio",
    }
    result = {"batches": batch_total, "frames": frame_total}
    result["done_acc"] = sum(
        float(item["done_acc"]) * int(item["frames"]) for item in chunks
    ) / max(1, frame_total)
    for key in batch_weighted:
        result[key] = sum(
            float(item[key]) * int(item["batches"]) for item in chunks
        ) / max(1, batch_total)
    return result


def run_orchestrator(snapshot: Path, output: Path) -> int:
    payload, cfg, stats, selector_binding = load_snapshot(snapshot)
    if cfg.hand_action_representation == "grasp_binary":
        raise ValueError("chunked isolated validation is for direct robot-q checkpoints")
    # This index-only parent never initializes CUDA.
    dataset = make_dataset(
        cfg, stats, cache_json=False,
        allowed_window_starts=selector_binding["window_starts"],
        selector_records=selector_binding["selector_records"],
        selector_root=selector_binding["selector_root"],
        sidecar_sha256_by_session=payload["run_manifest"]["sidecars"],
    )
    batch_size = cfg.eval_batch_size or cfg.batch_size
    total_batches = math.ceil(len(dataset) / batch_size)
    if cfg.max_eval_batches is not None:
        total_batches = min(total_batches, int(cfg.max_eval_batches))
    chunk_batches = int(os.environ.get("HUMANEGO_EVAL_CHUNK_BATCHES", "8"))
    if chunk_batches < 1:
        raise ValueError("HUMANEGO_EVAL_CHUNK_BATCHES must be positive")
    max_retries = int(os.environ.get("HUMANEGO_EVAL_CHUNK_RETRIES", "2"))
    if max_retries < 1:
        raise ValueError("HUMANEGO_EVAL_CHUNK_RETRIES must be positive")
    temporary_chunks: list[Path] = []

    def evaluate_range(start: int, count: int) -> list[dict]:
        chunk_path = output.parent / (
            f".{output.stem}_chunk_{start:04d}_{count:04d}.json"
        )
        temporary_chunks.append(chunk_path)
        command = [
            sys.executable, str(Path(__file__).resolve()),
            "--snapshot", str(snapshot), "--output", str(chunk_path),
            "--start-batch", str(start), "--batch-count", str(count),
        ]
        for attempt in range(1, max_retries + 1):
            chunk_path.unlink(missing_ok=True)
            completed = subprocess.run(command, cwd=ROOT, check=False)
            if completed.returncode == 0:
                return [read_isolated_evaluation_metrics(
                    chunk_path,
                    label=f"isolated evaluation chunk {start}:{start + count}",
                )]
            print(
                f"[isolated-eval] chunk {start}:{start + count} failed "
                f"with {completed.returncode}; retry {attempt}/{max_retries}",
                flush=True,
            )
        if count == 1:
            # Driver 570.133.20 on this H20 deterministically SIGFPEs for the
            # final short validation batch (21 rather than 32 samples).  Keep
            # every validation sample: evaluate only that irreducible batch
            # on CPU, then merge it with the preceding GPU chunks.
            cpu_command = command + ["--device", "cpu"]
            chunk_path.unlink(missing_ok=True)
            cpu_completed = subprocess.run(cpu_command, cwd=ROOT, check=False)
            if cpu_completed.returncode == 0:
                print(
                    f"[isolated-eval] chunk {start}:{start + count} "
                    "completed with CPU fallback",
                    flush=True,
                )
                return [read_isolated_evaluation_metrics(
                    chunk_path,
                    label=f"CPU isolated evaluation chunk {start}:{start + count}",
                )]
            raise RuntimeError(
                f"isolated validation chunk {start}:{start + count} failed "
                f"on CUDA ({completed.returncode}) and CPU "
                f"({cpu_completed.returncode})"
            )
        # A bad cuBLASLt process occasionally exits with SIGFPE even though
        # the same real validation data succeeds in a fresh process.  Split a
        # repeatedly failing range so one transient process never discards a
        # complete epoch's validation.
        left = count // 2
        return evaluate_range(start, left) + evaluate_range(
            start + left, count - left
        )

    chunks = []
    for start in range(0, total_batches, chunk_batches):
        count = min(chunk_batches, total_batches - start)
        chunks.extend(evaluate_range(start, count))
        print(
            f"[isolated-eval] batches {start + count}/{total_batches}",
            flush=True,
        )
    metrics = merge_chunks(chunks)
    atomic_json(output, metrics)
    for chunk_path in temporary_chunks:
        chunk_path.unlink(missing_ok=True)
    print(json.dumps(metrics, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-batch", type=int)
    parser.add_argument("--batch-count", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    if (args.start_batch is None) != (args.batch_count is None):
        parser.error("--start-batch and --batch-count must be specified together")
    if args.start_batch is not None:
        return run_worker(
            args.snapshot.resolve(), args.output.resolve(),
            args.start_batch, args.batch_count, args.device,
        )
    return run_orchestrator(args.snapshot.resolve(), args.output.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
