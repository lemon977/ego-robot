"""Full-source causal derived ICT route: lazy admission, fresh transfer, finite training."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

from .production_training_gate import (TRAINING_SCOPE, SCHEMA, PREPARE, authorize_production_prepare,
    authorize_production_training, file_ref, bind_cuda_visible_devices, runtime_cache_paths, production_storage_ready)
from .derived_runtime import _json, write_once
from .derived_training_cli import validate_config as _numeric_config, _check_caches, _rank_error

CONFIG_SCHEMA = "pick-place-humanego-ict-production-training-config-v1"


def repository_root():
    return Path(__file__).resolve().parents[3]


def make_training_config(wall_seconds, *, manifest_ref, pretrained_ref, pretrained_config_ref, world_size=1):
    if type(world_size) is not int or world_size not in {1, 8}:
        raise ValueError("SINGLE_OR_EIGHT_GPU_CONFIG_REQUIRED")
    return {"schema_version": CONFIG_SCHEMA, "training_scope": TRAINING_SCOPE,
        "supervision_class": "DERIVED_RETARGET", "control_ground_truth": False,
        "physical_deployment_authorized": False, "minimum_free_output_bytes": 4 * 1024**3,
        "minimum_export_margin_seconds": 1200, "initialization": "FRESH_SHARED_PRETRAINED_NEW_IO",
        "input_bindings": {"production_manifest_sha256": manifest_ref["sha256"],
            "pretrained_sha256": pretrained_ref["sha256"], "pretrained_config_sha256": pretrained_config_ref["sha256"]},
        "model": {"pred_horizon": 50, "max_ict": 8, "img_size": [240, 320], "patch_size": 16,
            "vision_embed_dim": 384, "num_decoder_layers": 6, "num_heads": 8, "mlp_ratio": 4.0,
            "dropout": 0.05, "horizon_weighting": "uniform", "horizon_beta": 1.0,
            "use_pre_norm": True, "use_ctx_norm": True},
        "training": {"global_batch": 64, "micro_batch": 64 // world_size, "max_updates": 20000, "precision": "bf16",
            "seed": 7, "learning_rate": 1e-4, "warmup_updates": 200, "ema_decay": 0.999,
            "checkpoint_every": 1000, "grad_clip": 1.0, "position_weight": 5.0, "rotation_weight": 1.0,
            "hand_weight": 5.0, "wall_seconds_max": wall_seconds, "timing_warmup_updates": 20, "log_every": 20}}


def validate_config(config, world):
    if (config.get("schema_version") != CONFIG_SCHEMA or config.get("training_scope") != TRAINING_SCOPE
            or config.get("initialization") != "FRESH_SHARED_PRETRAINED_NEW_IO"):
        raise ValueError("SEPARATE_FRESH_PRODUCTION_RESEARCH_CONFIG_REQUIRED")
    projected = dict(config, schema_version="pick-place-humanego-ict-derived-training-config-v1")
    _numeric_config(projected, world)
    train = config["training"]
    if (train["max_updates"] != 20000 or train["micro_batch"] != 64 // world or train["precision"] != "bf16"
            or train["checkpoint_every"] != 1000 or train["log_every"] != 20):
        raise ValueError("FIXED_20K_MAX_MICRO64_PER_WORLD_BF16_CHECKPOINT1000_LOG20_REQUIRED")
    margin = config.get("minimum_export_margin_seconds")
    if type(margin) is not int or margin < 1200:
        raise ValueError("AT_LEAST_20_MINUTE_EXPORT_MARGIN_REQUIRED")
    binding = config.get("input_bindings", {})
    for name in ("production_manifest_sha256", "pretrained_sha256", "pretrained_config_sha256"):
        value = binding.get(name)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("CONFIG_INPUT_SHA_BINDINGS_REQUIRED:" + name)


def shared_schema(model):
    """Exact shapes from the SHA-bound FlowMatchingModel shared architecture, no construction."""
    width, patch = model.get("vision_embed_dim", 384), model.get("patch_size", 16)
    hidden = int(width * model.get("mlp_ratio", 4.0))
    result = {"rgb_embed.proj.weight": (width, 3, patch, patch), "rgb_embed.proj.bias": (width,),
        "state_pos_emb": (1, model.get("max_ict", 8), width), "pos_embed": (model.get("pred_horizon", 50), width),
        "time_mlp.1.weight": (width * 2, width), "time_mlp.1.bias": (width * 2,),
        "time_mlp.3.weight": (width, width * 2), "time_mlp.3.bias": (width,)}
    if model.get("use_ctx_norm", True):
        result.update({"ctx_norm.weight": (width,), "ctx_norm.bias": (width,)})
    for i in range(model.get("num_decoder_layers", 6)):
        prefix = f"decoder.{i}."
        for kind in ("self_attn", "cross_attn"):
            result.update({prefix + kind + ".in_proj_weight": (width * 3, width),
                prefix + kind + ".in_proj_bias": (width * 3,), prefix + kind + ".out_proj.weight": (width, width),
                prefix + kind + ".out_proj.bias": (width,)})
        result.update({prefix + "mlp.0.weight": (hidden, width), prefix + "mlp.0.bias": (hidden,),
            prefix + "mlp.3.weight": (width, hidden), prefix + "mlp.3.bias": (width,)})
        for norm in ("norm1", "norm2", "norm3"):
            result.update({prefix + norm + ".weight": (width,), prefix + norm + ".bias": (width,)})
    return result


def inspect_pretrained(reference, source_config_ref, model):
    """Safe CPU tensor inspection only; no model, forward, optimizer or unsafe fallback."""
    source_config = _json(source_config_ref)
    for name, default in (("patch_size", 16), ("vision_embed_dim", 384), ("num_decoder_layers", 6),
            ("num_heads", 8), ("mlp_ratio", 4.0), ("pred_horizon", 50), ("max_ict", 8),
            ("use_pre_norm", True), ("use_ctx_norm", True)):
        if source_config.get(name, default) != model.get(name, default):
            raise ValueError("PRETRAINED_SHARED_ARCHITECTURE_MISMATCH:" + name)
    from chaoyang.human_ego.utils.frozen_contract import load_verified_torch_checkpoint
    import torch
    try:
        _, payload = load_verified_torch_checkpoint(reference,
            allowed_roots=[Path(reference["path"]).parent], label="production shared pretrained", map_location="cpu")
    except Exception as error:
        raise ValueError("SAFE_CPU_PRETRAINED_READ_FAILED:" + str(error)) from error
    if not isinstance(payload, dict):
        raise ValueError("PRETRAINED_MODEL_TENSOR_MAPPING_REQUIRED")
    source = payload.get("model", payload)
    if not isinstance(source, dict):
        raise ValueError("PRETRAINED_MODEL_TENSOR_MAPPING_REQUIRED")
    shapes = shared_schema(model)
    selected = {}
    for key, shape in shapes.items():
        tensor = source.get(key)
        if (not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu"
                or tuple(tensor.shape) != shape or not tensor.is_floating_point()
                or not torch.isfinite(tensor).all().item()):
            raise ValueError("MISSING_INCOMPATIBLE_OR_NONFINITE_SHARED_PRETRAINED:" + key)
        selected[key] = tensor
    return selected, {"checkpoint": reference, "pretrained_config": source_config_ref,
        "inspection": "SAFE_CPU_TENSORS_SHARED_SCHEMA_ONLY_NO_MODEL_FORWARD",
        "loaded_keys": sorted(selected), "loaded_key_count": len(selected),
        "loaded_parameter_elements": sum(t.numel() for t in selected.values()),
        "excluded_new_io_prefixes": ["state_proj.", "action_proj.", "head_v.", "head_done."],
        "source_unselected_keys": sorted(set(source) - set(selected)),
        "control_ground_truth": False, "physical_deployment_authorized": False}


def initialize_pretrained(model, packet, config, output, root):
    report_path = output / "pretrained_load_report.json"
    if report_path.exists():
        raise ValueError("FRESH_PRODUCTION_OUTPUT_REQUIRED")
    selected, report = inspect_pretrained(packet["inputs"]["pretrained"], packet["inputs"]["pretrained_config"], config["model"])
    if not selected or not any(k.startswith("rgb_embed.") for k in selected) or not any(k.startswith("decoder.") for k in selected):
        raise ValueError("EMPTY_OR_INCOMPLETE_PRETRAINED_TRANSFER_FORBIDDEN")
    destination = model.state_dict()
    if any(key not in destination or tuple(destination[key].shape) != tuple(value.shape) for key, value in selected.items()):
        raise ValueError("LIVE_MODEL_SHARED_SCHEMA_MISMATCH")
    result = model.load_state_dict(selected, strict=False)
    if result.unexpected_keys:
        raise ValueError("EMPTY_OR_INCOMPLETE_PRETRAINED_TRANSFER_FORBIDDEN")
    report["missing_destination_keys"] = sorted(result.missing_keys)
    report["loaded_parameter_fraction"] = report["loaded_parameter_elements"] / max(1, sum(v.numel() for v in destination.values()))
    write_once(report_path, report, root)


def export_dataset_envelope(qualification, manifest_ref):
    """Numerical export compatibility, never a claim that the legacy reader passed."""
    value = dict(qualification["prepared_pair"])
    value.update(schema="pick-place-humanego-ict-revo2-derived-paired-v1", production_manifest_ref=manifest_ref,
        training_scope=TRAINING_SCOPE, export_compatibility_only=True,
        metadata_qualification="NUMERIC_EXPORT_METADATA_NOT_OLD_READER_QUALIFICATION",
        legacy_reader_qualification="NOT_CLAIMED",
        supervision_class="DERIVED_RETARGET", control_ground_truth=False, physical_deployment_authorized=False)
    return value


def validate_admitted_inputs(config, encoded, packet, qualification):
    """CPU-only final source/config signatures and live export margin, before caches/Torch."""
    for role, name in (("production_manifest", "production_manifest_sha256"), ("pretrained", "pretrained_sha256"),
            ("pretrained_config", "pretrained_config_sha256")):
        if config["input_bindings"][name] != packet["inputs"][role]["sha256"]:
            raise ValueError("CONFIG_DATA_WEIGHT_SIGNATURE_MISMATCH:" + role)
    if (qualification["dataset_sha"] != packet["inputs"]["production_manifest"]["sha256"]
            or qualification["prepared_pair_ref"] != packet["inputs"]["prepared_pair"]
            or hashlib.sha256(encoded).hexdigest() != packet["inputs"]["training_config"]["sha256"]):
        raise ValueError("ADMITTED_BYTES_CHANGED")
    started = datetime.fromisoformat(packet["execution_started_at"])
    if started.tzinfo is None:
        raise ValueError("EXECUTION_STARTED_AT_MUST_BE_AWARE_UTC")
    expiry, margin = packet["admitted_authorization_expires_at"], config["minimum_export_margin_seconds"]
    if started.timestamp() + config["training"]["wall_seconds_max"] + margin > expiry:
        raise ValueError("TRAINING_WALL_BUDGET_MUST_LEAVE_EXPORT_MARGIN")
    if datetime.now(timezone.utc).timestamp() + margin + 60 > expiry:
        raise ValueError("INITIALIZATION_TIME_CONSUMES_TRAINING_AND_EXPORT_BUDGET")


def prepare_main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="Governed CPU source replay production, no training")
    parser.add_argument("--task-packet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    root = repository_root()
    try:
        packet = authorize_production_prepare(root, args.task_packet, argv, output=args.output_dir)
        packet["storage_admission"] = production_storage_ready(args.output_dir, 4 * 1024**3, packet=packet, root=root)
        path = root / "scripts/humanego_ict/production_runner_v1.py"
        spec = importlib.util.spec_from_file_location("frozen_production_runner", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        result = module.prepare(packet, root)
        authorize_production_prepare(root, args.task_packet, argv, output=args.output_dir)
        result["storage_admission"] = packet["storage_admission"]
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except (ValueError, KeyError, TypeError, OSError, RuntimeError, ImportError) as error:
        print(json.dumps({"status": "BLOCKED", "reason": str(error), "training_started": False,
                          "physical_deployment_authorized": False}), file=sys.stderr)
        return 2


def main(argv=None):
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
            from .production_data import validate_production_manifest
            result = validate_production_manifest(args.manifest, args.branch)
            print(json.dumps({"status": "QUALIFIED_NON_DEPLOYABLE_SOURCE", "dataset_sha": result["dataset_sha"],
                "training_started": False, "training_scope": TRAINING_SCOPE, "physical_deployment_authorized": False}))
            return 0
        encoded = args.config.read_bytes()
        config, world = json.loads(encoded), int(os.environ.get("WORLD_SIZE", "1"))
        validate_config(config, world)
        root, output = repository_root(), args.output_dir.resolve()
        def reauthorize():
            return authorize_production_training(root, args.task_packet, argv, manifest=args.manifest,
                config=args.config, pretrained=args.pretrained, output=output, branch=args.branch,
                updates=config["training"]["max_updates"], world_size=world)
        packet = reauthorize()
        from .production_data import validate_production_manifest
        qualification = validate_production_manifest(args.manifest, args.branch)
        validate_admitted_inputs(config, encoded, packet, qualification)
        storage_admission = production_storage_ready(output, config["minimum_free_output_bytes"], packet=packet, root=root)
        bind_cuda_visible_devices(packet)
        if os.environ.get("ICT_PRODUCTION_ADMISSION_ONLY") == "1":
            print(json.dumps({"status": "ADMITTED_PRODUCTION_TRAINING_ONLY", "training_started": False}))
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
        rank, local = int(os.environ.get("RANK", "0")), int(os.environ.get("LOCAL_RANK", "0"))
        if not 0 <= rank < world or not 0 <= local < world:
            raise ValueError("VALID_LAUNCHER_RANKS_REQUIRED")
        torch.cuda.set_device(local)
        if world > 1:
            remaining = packet["admitted_authorization_expires_at"] - datetime.now(timezone.utc).timestamp()
            torch.distributed.init_process_group("nccl", timeout=timedelta(seconds=max(1, min(180, remaining))))
            distributed = True
        error = None
        try:
            from .model import Revo2ICTModel
            from .production_data import ProductionDerivedICTDataset
            from .training import TrainingConfig, train_candidate
            torch.manual_seed(config["training"]["seed"])
            reauthorize()
            model = Revo2ICTModel(**config["model"]).to(f"cuda:{local}")
            model_loaded = True
            dataset = ProductionDerivedICTDataset(args.manifest, args.branch, "train")
        except Exception as caught:
            error = caught
        _rank_error(error, world, torch)
        def initialize(locked_model):
            dataset.verify_all_dependencies()
            initialize_pretrained(locked_model, reauthorize(), config, output, root)
        signatures = {"run_signature": packet["run_signature"], "data": file_ref(args.manifest),
            "config": file_ref(args.config), "pretrained": packet["inputs"]["pretrained"],
            "pretrained_config": packet["inputs"]["pretrained_config"], "branch": args.branch,
            "code": packet["runtime_code"], "supervision_class": "DERIVED_RETARGET", "control_ground_truth": False,
            "prepared_pair": qualification["prepared_pair_ref"], "training_scope": TRAINING_SCOPE,
            "storage_capacity_observation": packet["inputs"].get("storage_capacity_observation")}
        training_started = True
        result = train_candidate(model, dataset, TrainingConfig(**config["training"]), output, signatures,
            device=f"cuda:{local}", initializer=initialize,
            authorization_expires_at=packet["admitted_authorization_expires_at"] - config["minimum_export_margin_seconds"])
        error = None
        try:
            if rank == 0:
                dataset.verify_all_dependencies()
                result.update(supervision_class="DERIVED_RETARGET", control_ground_truth=False,
                    physical_deployment_authorized=False, branch=args.branch, training_scope=TRAINING_SCOPE,
                    task_id=packet["task_id"], run_signature=packet["run_signature"], storage_admission=storage_admission)
                result_path = output / f"RESULT_{result['optimizer_updates']:06d}.json"
                write_once(result_path, result, root)
                if result["optimizer_updates"] > 0:
                    from .checkpoint_export import export_tensor_candidate
                    pair = qualification["prepared_pair"]
                    normal = output / "inference_normalization.json"
                    export_pair = output / "export_dataset.json"
                    write_once(normal, pair["normalization"], root)
                    write_once(export_pair, export_dataset_envelope(qualification, file_ref(args.manifest)), root)
                    provenance = {"task_id": packet["task_id"], "run_signature": packet["run_signature"],
                        "branch": args.branch, "supervision_class": "DERIVED_RETARGET", "state_origin": "LIVE_MODEL_STATE",
                        "optimizer_updates": result["optimizer_updates"], "dataset_ref": file_ref(export_pair),
                        "production_manifest_ref": file_ref(args.manifest), "training_scope": TRAINING_SCOPE,
                        "training_config_ref": file_ref(args.config), "training_result_ref": file_ref(result_path),
                        "code_refs": packet["runtime_code"]}
                    def authorizer(value):
                        if value != provenance:
                            raise ValueError("EXPORT_PRODUCER_BINDING_MISMATCH")
                        fresh = reauthorize()
                        return {"status": "PASSED", "export_authorized": True, "task_id": fresh["task_id"],
                            "run_signature": fresh["run_signature"], "execution_mode": "TRAINING_OWNED_LIVE_STATE",
                            "gpu_lease_verified": True, "output_dir": str(output / "inference_tensors")}
                    frame = {"position_unit": "meters", "tcp_frame_ids": pair["pose_clock"]["tcp_frame_ids"],
                        "pose_semantics": "DERIVED_RETARGET_ANCHOR", "action_reference_frame": "REFERENCE_CAMERA"}
                    exported = export_tensor_candidate(model.state_dict(), model_config_ref=file_ref(args.config),
                        normalization_ref=file_ref(normal), provenance=provenance, output_dir=output / "inference_tensors",
                        runtime_authorizer=authorizer, target_frame_contract=frame, state_dict_key="model")
                    write_once(output / "INFERENCE_EXPORT.json", exported, root)
                print(json.dumps(result, ensure_ascii=False))
        except Exception as caught:
            error = caught
        _rank_error(error, world, torch)
        return 0
    except (ValueError, KeyError, TypeError, OSError, RuntimeError, ImportError) as error:
        print(json.dumps({"status": "BLOCKED", "reason": str(error), "model_loaded": model_loaded,
            "training_started": training_started, "control_ground_truth": False,
            "physical_deployment_authorized": False, "training_scope": TRAINING_SCOPE}), file=sys.stderr)
        return 2
    finally:
        if distributed:
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
