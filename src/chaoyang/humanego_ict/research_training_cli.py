"""Direct paired non-deployable research-derived training with strict lazy admission."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timedelta, timezone

from .research_training_gate import (
    authorize_research_training, bind_cuda_visible_devices, file_ref,
    runtime_cache_paths, storage_ready,
)
from .derived_runtime import write_once
from .research_data import TRAINING_SCOPE
from .derived_training_cli import validate_config as _validate_numeric_config, _check_caches, _rank_error


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


CONFIG_SCHEMA = "pick-place-humanego-ict-research-training-config-v1"


def validate_config(config: dict, world: int) -> None:
    if config.get("schema_version") != CONFIG_SCHEMA or config.get("training_scope") != TRAINING_SCOPE:
        raise ValueError("SEPARATE_NON_DEPLOYABLE_RESEARCH_CONFIG_REQUIRED")
    # A projection reuses the protected numerical recipe; it does not change
    # either the caller's bytes or old derived/REAL_CONTROL admission.
    projected = dict(config)
    projected["schema_version"] = "pick-place-humanego-ict-derived-training-config-v1"
    _validate_numeric_config(projected, world)


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
            from .research_data import validate_research_manifest
            result = validate_research_manifest(args.manifest, args.branch)
            print(json.dumps({"status": "QUALIFIED_RESEARCH_SOURCE_NOT_DEPLOYMENT", "branch": args.branch,
                              "dataset_sha": result["dataset_sha"], "training_started": False,
                              "control_ground_truth": False, "physical_deployment_authorized": False,
                              "training_scope": TRAINING_SCOPE}))
            return 0
        encoded = args.config.read_bytes()
        config = json.loads(encoded)
        world = int(os.environ.get("WORLD_SIZE", "1"))
        validate_config(config, world)
        root, output = repository_root(), args.output_dir.resolve()
        def reauthorize():
            return authorize_research_training(root, args.task_packet, argv, manifest=args.manifest,
                config=args.config, pretrained=args.pretrained, output=output,
                updates=config["training"]["max_updates"], world_size=world)
        packet = reauthorize()
        from .research_data import validate_research_manifest
        qualification = validate_research_manifest(args.manifest, args.branch)
        if (qualification["dataset_sha"] != packet["inputs"]["research_manifest"]["sha256"]
                or hashlib.sha256(encoded).hexdigest() != packet["inputs"]["training_config"]["sha256"]):
            raise ValueError("ADMITTED_BYTES_CHANGED")
        storage_ready(output, config.get("minimum_free_output_bytes", 4 * 1024**3))
        bind_cuda_visible_devices(packet)
        if os.environ.get("ICT_RESEARCH_ADMISSION_ONLY") == "1":
            print(json.dumps({"status": "ADMITTED_RESEARCH_CANDIDATE_ONLY", "training_started": False,
                              "training_scope": TRAINING_SCOPE, "physical_deployment_authorized": False}))
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
            from .research_data import ResearchDerivedICTDataset
            from .training import TrainingConfig, train_candidate
            from chaoyang.human_ego.training.checkpoint_transfer import load_compatible_pretrained
            torch.manual_seed(config["training"].get("seed", 7))
            reauthorize()
            model = Revo2ICTModel(**config["model"]).to(f"cuda:{local}")
            model_loaded = True
            dataset = ResearchDerivedICTDataset(args.manifest, args.branch, "train")
        except Exception as error:
            preparation_error = error
        _rank_error(preparation_error, world, torch)
        def initialize(locked_model):
            admitted = reauthorize()
            if (output / "pretrained_load_report.json").exists():
                raise ValueError("FRESH_RESEARCH_OUTPUT_REQUIRED")
            load_compatible_pretrained(locked_model, args.pretrained, output / "pretrained_load_report.json",
                                      expected_sha256=admitted["inputs"]["pretrained"]["sha256"])
        signatures = {"run_signature": packet["run_signature"], "data": file_ref(args.manifest),
            "config": file_ref(args.config), "pretrained": file_ref(args.pretrained), "branch": args.branch,
            "code": packet["runtime_code"], "supervision_class": "DERIVED_RETARGET", "control_ground_truth": False,
            "prepared_pair": qualification["prepared_pair_ref"], "training_scope": TRAINING_SCOPE}
        training_started = True
        result = train_candidate(model, dataset, TrainingConfig(**config["training"]), output,
            signatures, device=f"cuda:{local}", initializer=initialize,
            authorization_expires_at=packet["admitted_authorization_expires_at"])
        publication_error = None
        try:
            if rank == 0:
                result.update(supervision_class="DERIVED_RETARGET", control_ground_truth=False,
                              physical_deployment_authorized=False, branch=args.branch,
                              training_scope=TRAINING_SCOPE, task_id=packet["task_id"], run_signature=packet["run_signature"])
                result_path = output / f"RESULT_{result['optimizer_updates']:06d}.json"
                write_once(result_path, result, root)
                if result["optimizer_updates"] > 0:
                    from .checkpoint_export import export_tensor_candidate
                    manifest = qualification["prepared_pair"]
                    normalization_path = output / "inference_normalization.json"
                    write_once(normalization_path, manifest["normalization"], root)
                    frame = {"position_unit": "meters", "tcp_frame_ids": manifest["pose_clock"]["tcp_frame_ids"],
                             "pose_semantics": "DERIVED_RETARGET_ANCHOR", "action_reference_frame": "REFERENCE_CAMERA"}
                    def export_authorizer(provenance):
                        if (provenance.get("task_id") != packet["task_id"]
                                or provenance.get("run_signature") != packet["run_signature"]
                                or provenance.get("training_scope") != TRAINING_SCOPE
                                or provenance.get("research_manifest_ref") != packet["inputs"]["research_manifest"]
                                or provenance.get("dataset_ref") != packet["inputs"]["prepared_pair"]
                                or provenance.get("training_config_ref") != packet["inputs"]["training_config"]
                                or provenance.get("code_refs") != packet["runtime_code"]):
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
                            "optimizer_updates": result["optimizer_updates"], "dataset_ref": qualification["prepared_pair_ref"],
                            "research_manifest_ref": file_ref(args.manifest), "training_scope": TRAINING_SCOPE,
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
                          "physical_deployment_authorized": False, "training_scope": TRAINING_SCOPE}), file=sys.stderr)
        return 2
    finally:
        if distributed:
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
