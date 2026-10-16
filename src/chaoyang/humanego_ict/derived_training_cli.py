"""Separate derived-label training candidate with no implicit model execution."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from datetime import datetime, timedelta, timezone

from .derived_training_gate import (
    authorize_derived_training, bind_cuda_visible_devices, file_ref,
    runtime_cache_paths, storage_ready,
)
from .derived_runtime import write_once


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def validate_config(config: dict, world: int) -> None:
    if (config.get("schema_version") != "pick-place-humanego-ict-derived-training-config-v1"
            or config.get("supervision_class") != "DERIVED_RETARGET"
            or config.get("control_ground_truth") is not False
            or config.get("physical_deployment_authorized") is not False):
        raise ValueError("SEPARATE_DERIVED_TRAINING_CONFIG_REQUIRED")
    model, train = config["model"], config["training"]
    allowed_training = {"global_batch", "micro_batch", "max_updates", "precision", "seed", "learning_rate",
        "warmup_updates", "ema_decay", "checkpoint_every", "grad_clip", "position_weight",
        "rotation_weight", "hand_weight", "wall_seconds_max", "timing_warmup_updates", "log_every"}
    if set(train) - allowed_training:
        raise ValueError("UNSUPPORTED_TRAINING_OVERRIDE")
    seed = train.get("seed", 7)
    if type(seed) is not int or not 0 <= seed <= (2**32 - world):
        raise ValueError("SEED_MUST_FIT_ALL_RANK_NUMPY_GENERATORS")
    if (model.get("pred_horizon") != 50 or model.get("max_ict") != 8
            or model.get("img_size") != [240, 320]):
        raise ValueError("FIXED_H50_ICT8_IMAGE_240_320_REQUIRED")
    allowed = {"pred_horizon", "max_ict", "img_size", "patch_size", "vision_embed_dim",
               "num_decoder_layers", "num_heads", "mlp_ratio", "dropout",
               "horizon_weighting", "horizon_beta", "use_pre_norm", "use_ctx_norm"}
    if set(model) - allowed:
        raise ValueError("UNSUPPORTED_ARCHITECTURE_OVERRIDE")
    for name in ("patch_size", "vision_embed_dim", "num_decoder_layers", "num_heads"):
        value = model.get(name, {"patch_size": 16, "vision_embed_dim": 384,
                                "num_decoder_layers": 6, "num_heads": 8}[name])
        if type(value) is not int or value < 1:
            raise ValueError("POSITIVE_INTEGER_ARCHITECTURE_REQUIRED:" + name)
    patch = model.get("patch_size", 16)
    if (240 % patch or 320 % patch
            or model.get("vision_embed_dim", 384) % model.get("num_heads", 8)):
        raise ValueError("INCOMPATIBLE_PATCH_OR_ATTENTION_DIMENSIONS")
    for name, default in (("mlp_ratio", 4.0), ("dropout", 0.05), ("horizon_beta", 1.0)):
        value = model.get(name, default)
        if (type(value) not in {int, float} or not math.isfinite(value)
                or (not 0 <= value < 1 if name == "dropout" else value <= 0)):
            raise ValueError("FINITE_ARCHITECTURE_VALUE_REQUIRED:" + name)
    for name in ("max_updates", "micro_batch", "global_batch", "checkpoint_every", "log_every"):
        if type(train[name]) is not int or train[name] < 1:
            raise ValueError("FINITE_POSITIVE_INTEGER_TRAINING_BUDGET_REQUIRED:" + name)
    if (type(world) is not int or world not in {1, 8} or train["global_batch"] != 64
            or train["global_batch"] % (world * train["micro_batch"])):
        raise ValueError("FIXED_GLOBAL64_SINGLE_OR_EIGHT_GPU_ACCUMULATION_REQUIRED")
    if train["precision"] not in {"bf16", "float32"}:
        raise ValueError("SUPPORTED_BF16_OR_FLOAT32_REQUIRED")
    for name in ("learning_rate", "wall_seconds_max", "grad_clip", "position_weight", "rotation_weight", "hand_weight"):
        value = train.get(name, 1.0)
        if type(value) not in {int, float} or not math.isfinite(value) or value <= 0:
            raise ValueError("FINITE_POSITIVE_TRAINING_VALUE_REQUIRED:" + name)
    for name in ("warmup_updates", "timing_warmup_updates"):
        value = train.get(name, 0)
        if type(value) is not int or value < 0:
            raise ValueError("NONNEGATIVE_INTEGER_WARMUP_REQUIRED")
    decay = train.get("ema_decay", 0.999)
    if type(decay) not in {int, float} or not math.isfinite(decay) or not 0 <= decay < 1:
        raise ValueError("FINITE_EMA_DECAY_REQUIRED")
    minimum = config.get("minimum_free_output_bytes", 4 * 1024**3)
    if type(minimum) is not int or minimum < 4 * 1024**3:
        raise ValueError("MINIMUM_CHECKPOINT_CAPACITY_NOT_LOWERED")


def _check_caches(output: Path, root: Path) -> None:
    paths = runtime_cache_paths(output, root)
    variables = ["TMPDIR", "TORCH_HOME", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "XDG_CACHE_HOME"]
    for variable, expected in zip(variables, paths):
        value = os.environ.get(variable)
        if value is None or Path(value).resolve() != expected.resolve():
            raise ValueError("TASK_LOCAL_RUNTIME_CACHE_REQUIRED:" + variable)


def _rank_error(error, world, torch):
    if world > 1:
        messages = [None] * world
        torch.distributed.all_gather_object(messages, str(error) if error else None)
        if any(messages):
            raise RuntimeError("; ".join(message for message in messages if message))
    elif error:
        raise error


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--manifest", type=Path, required=True)
    preflight.add_argument("--branch", choices=("raw", "robotized"), required=True)
    train = sub.add_parser("train")
    for name in ("task-packet", "manifest", "config", "pretrained", "output-dir"):
        train.add_argument("--" + name, type=Path, required=True)
    train.add_argument("--branch", choices=("raw", "robotized"), required=True)
    args = parser.parse_args(argv)
    model_loaded = training_started = distributed = False
    try:
        if args.command == "preflight":
            from .derived_data import validate_derived_training_manifest
            result = validate_derived_training_manifest(args.manifest, args.branch)
            print(json.dumps({"status": "QUALIFIED_DERIVED_MANIFEST", "branch": args.branch,
                              "dataset_sha": result["dataset_sha"], "training_started": False,
                              "control_ground_truth": False, "physical_deployment_authorized": False}))
            return 0
        encoded = args.config.read_bytes()
        config = json.loads(encoded)
        world = int(os.environ.get("WORLD_SIZE", "1"))
        validate_config(config, world)
        root, output = repository_root(), args.output_dir.resolve()
        def reauthorize():
            return authorize_derived_training(root, args.task_packet, argv, manifest=args.manifest,
                config=args.config, pretrained=args.pretrained, output=output,
                updates=config["training"]["max_updates"], world_size=world)
        packet = reauthorize()
        from .derived_data import validate_derived_training_manifest
        qualification = validate_derived_training_manifest(args.manifest, args.branch)
        if (qualification["dataset_sha"] != packet["inputs"]["paired_manifest"]["sha256"]
                or hashlib.sha256(encoded).hexdigest() != packet["inputs"]["training_config"]["sha256"]):
            raise ValueError("ADMITTED_BYTES_CHANGED")
        storage_ready(output, config.get("minimum_free_output_bytes", 4 * 1024**3))
        bind_cuda_visible_devices(packet)
        if os.environ.get("ICT_DERIVED_ADMISSION_ONLY") == "1":
            print(json.dumps({"status": "ADMITTED_DERIVED_CANDIDATE_ONLY", "training_started": False}))
            return 0
        _check_caches(output, root)
        for path in runtime_cache_paths(output, root):
            path.mkdir(parents=True, exist_ok=True)
        reauthorize()
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        os.environ.setdefault("DISABLE_ADDMM_CUDA_LT", "1")
        os.environ.setdefault("TORCH_BLAS_PREFER_CUBLASLT", "0")
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() < world:
            raise ValueError("WAIT_GPU_RESOURCE")
        if world > 1:
            remaining = packet["admitted_authorization_expires_at"] - datetime.now(timezone.utc).timestamp()
            if remaining <= 0:
                raise ValueError("AUTHORIZATION_EXPIRED_BEFORE_DISTRIBUTED_INIT")
            torch.distributed.init_process_group("nccl", timeout=timedelta(seconds=min(180, remaining)))
            distributed = True
        rank, local = int(os.environ.get("RANK", "0")), int(os.environ.get("LOCAL_RANK", "0"))
        preparation_error = None
        try:
            if not 0 <= rank < world or not 0 <= local < world:
                raise ValueError("VALID_LAUNCHER_RANKS_REQUIRED")
            torch.cuda.set_device(local)
            from .model import Revo2ICTModel
            from .derived_data import DerivedRevo2ICTDataset
            from .training import TrainingConfig, train_candidate
            from chaoyang.human_ego.training.checkpoint_transfer import load_compatible_pretrained
            torch.manual_seed(config["training"].get("seed", 7))
            reauthorize()
            model = Revo2ICTModel(**config["model"]).to(f"cuda:{local}")
            model_loaded = True
            dataset = DerivedRevo2ICTDataset(args.manifest, args.branch, "train")
        except Exception as error:
            preparation_error = error
        _rank_error(preparation_error, world, torch)
        def initialize(locked_model):
            admitted = reauthorize()
            if (output / "pretrained_load_report.json").exists():
                raise ValueError("FRESH_DERIVED_OUTPUT_REQUIRED")
            load_compatible_pretrained(locked_model, args.pretrained, output / "pretrained_load_report.json",
                                      expected_sha256=admitted["inputs"]["pretrained"]["sha256"])
        signatures = {"run_signature": packet["run_signature"], "data": file_ref(args.manifest),
            "config": file_ref(args.config), "pretrained": file_ref(args.pretrained), "branch": args.branch,
            "code": packet["runtime_code"], "supervision_class": "DERIVED_RETARGET", "control_ground_truth": False}
        training_started = True
        result = train_candidate(model, dataset, TrainingConfig(**config["training"]), output,
            signatures, device=f"cuda:{local}", initializer=initialize,
            authorization_expires_at=packet["admitted_authorization_expires_at"])
        publication_error = None
        try:
            if rank == 0:
                result.update(supervision_class="DERIVED_RETARGET", control_ground_truth=False,
                              physical_deployment_authorized=False, branch=args.branch)
                result_path = output / f"RESULT_{result['optimizer_updates']:06d}.json"
                write_once(result_path, result, root)
                if result["optimizer_updates"] > 0:
                    from .checkpoint_export import export_tensor_candidate
                    manifest = json.loads(args.manifest.read_bytes())
                    normalization_path = output / "inference_normalization.json"
                    write_once(normalization_path, manifest["normalization"], root)
                    frame = {"position_unit": "meters", "tcp_frame_ids": manifest["pose_clock"]["tcp_frame_ids"],
                             "pose_semantics": "DERIVED_RETARGET_ANCHOR", "action_reference_frame": "REFERENCE_CAMERA"}
                    def export_authorizer(provenance):
                        if (provenance.get("task_id") != packet["task_id"]
                                or provenance.get("run_signature") != packet["run_signature"]):
                            raise ValueError("EXPORT_PRODUCER_BINDING_MISMATCH")
                        fresh = reauthorize()
                        return {"status": "PASSED", "export_authorized": True,
                                "task_id": fresh["task_id"], "run_signature": fresh["run_signature"],
                                "execution_mode": "TRAINING_OWNED_LIVE_STATE", "gpu_lease_verified": True,
                                "output_dir": str(output / "inference_tensors")}
                    exported = export_tensor_candidate(model.state_dict(), model_config_ref=file_ref(args.config),
                        normalization_ref=file_ref(normalization_path), provenance={"task_id": packet["task_id"],
                            "run_signature": packet["run_signature"], "branch": args.branch,
                            "supervision_class": "DERIVED_RETARGET", "state_origin": "LIVE_MODEL_STATE",
                            "optimizer_updates": result["optimizer_updates"], "dataset_ref": file_ref(args.manifest),
                            "training_config_ref": file_ref(args.config), "training_result_ref": file_ref(result_path),
                            "code_refs": packet["runtime_code"]}, output_dir=output / "inference_tensors",
                        runtime_authorizer=export_authorizer, target_frame_contract=frame, state_dict_key="model")
                    write_once(output / "INFERENCE_EXPORT.json", exported, root)
                print(json.dumps(result, ensure_ascii=False))
        except Exception as error:
            publication_error = error
        _rank_error(publication_error, world, torch)
        return 0
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
        print(json.dumps({"status": "BLOCKED", "reason": str(error), "model_loaded": model_loaded,
                          "training_started": training_started, "control_ground_truth": False,
                          "physical_deployment_authorized": False}), file=sys.stderr)
        return 2
    finally:
        if distributed:
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
