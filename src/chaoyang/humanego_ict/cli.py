"""Read-only preflight and governed candidate training, with lazy model imports."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from .runtime_gate import (
    authorize_training, bind_cuda_visible_devices, file_ref, project_output, storage_ready,
)


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _write_once(path: Path, value: dict) -> None:
    path = project_output(path, repository_root())
    content = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    if path.exists():
        if path.read_bytes() == content:
            return
        raise ValueError("IMMUTABLE_OUTPUT_CONFLICT")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(content)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit", help="read the NAS inputs; never load a model")
    audit.add_argument("--raw-root", type=Path, default=Path("/path/to/private_storage/datasets/ego/pick_place_1008"))
    audit.add_argument("--robotized-root", type=Path, default=Path("/path/to/private_storage/datasets/ego/pick_place_1008_raw_to_robot_v9_original_execution_v1"))
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--expected-sessions", type=int, default=199)
    preflight = sub.add_parser("preflight", help="validate a prepared paired manifest; no torch/model import")
    preflight.add_argument("--manifest", type=Path, required=True)
    train = sub.add_parser("train", help="registered Revo2 candidate training only")
    for name in ("task-packet", "manifest", "config", "output-dir", "pretrained"):
        train.add_argument("--" + name, type=Path, required=True)
    train.add_argument("--branch", choices=("raw", "robotized"), required=True)
    train.add_argument("--resume", type=Path)
    args = parser.parse_args(argv)
    training_started = False
    model_loaded = False
    distributed_group_started = False
    try:
        if args.command == "audit":
            from .readiness import audit_nas_roots
            result = audit_nas_roots(args.raw_root, args.robotized_root, expected_sessions=args.expected_sessions)
            _write_once(args.output, result)
            print(json.dumps({"audit_output": str(args.output.resolve()), "status": result.get("status"),
                              "training_started": False}, ensure_ascii=False))
            return 0  # Completion of an audit is not training eligibility.
        from .data import validate_training_manifest
        qualification = validate_training_manifest(args.manifest)
        if args.command == "preflight":
            summary = {key: qualification[key] for key in ("dataset_sha", "sample_counts", "source_group_splits")}
            print(json.dumps({"status": "PREPARED_MANIFEST_VALID", "manifest": file_ref(args.manifest),
                              "training_started": False, "qualification": summary}, ensure_ascii=False, default=str))
            return 0
        config_bytes = args.config.read_bytes()
        config_dict = json.loads(config_bytes)
        training_dict = config_dict["training"]
        if (training_dict["precision"] not in {"bf16", "float32"}
                or not isinstance(training_dict["max_updates"], int)
                or training_dict["max_updates"] < 1):
            raise ValueError("INVALID_FIXED_UPDATE_OR_PRECISION_CONFIG")
        updates = int(config_dict["training"]["max_updates"])
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        root = repository_root()
        output = project_output(args.output_dir, root)
        packet = authorize_training(root, args.task_packet, argv[1:], manifest=args.manifest,
                                    config=args.config, pretrained=args.pretrained, output=output,
                                    updates=updates, world_size=world_size)
        import hashlib
        if (qualification["dataset_sha"] != packet["inputs"]["paired_manifest"]["sha256"]
                or hashlib.sha256(config_bytes).hexdigest() != packet["inputs"]["training_config"]["sha256"]):
            raise ValueError("READ_BYTES_DIFFER_FROM_ADMITTED_INPUT_SHA")
        storage_ready(output, int(config_dict.get("minimum_free_output_bytes", 4 * 1024**3)))
        if os.environ.get("ICT_ADMISSION_ONLY") == "1":
            print(json.dumps({"status": "ADMITTED_FOR_REGISTERED_CANDIDATE_ONLY", "training_started": False}))
            return 0
        bind_cuda_visible_devices(packet)
        # Everything above this point is CPU-only admission. No implicit GPU or model
        # initialization can happen through --help, preflight, or a rejected packet.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        os.environ.setdefault("DISABLE_ADDMM_CUDA_LT", "1")
        os.environ.setdefault("TORCH_BLAS_PREFER_CUBLASLT", "0")
        import torch
        from .data import Revo2ICTDataset
        from .model import Revo2ICTModel
        from .training import TrainingConfig, train_candidate
        from chaoyang.human_ego.training.checkpoint_transfer import load_compatible_pretrained
        if not torch.cuda.is_available():
            raise ValueError("WAIT_GPU_RESOURCE")
        if torch.cuda.device_count() < world_size:
            raise ValueError("ALLOCATED_GPU_COUNT_UNAVAILABLE")
        if world_size > 1:
            torch.distributed.init_process_group("nccl")
            distributed_group_started = True
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        seed = int(config_dict["training"].get("seed", 7))
        torch.manual_seed(seed)
        architecture = config_dict["model"]
        model = Revo2ICTModel(**architecture).to(f"cuda:{local_rank}")
        model_loaded = True
        dataset = Revo2ICTDataset(args.manifest, args.branch, "train")
        def initialize_weights(locked_model):
            # train_candidate invokes this only on rank0 after acquiring its
            # writer lock, and synchronizes errors before DDP initialization.
            if not args.resume:
                if (output / "pretrained_load_report.json").exists():
                    raise ValueError("FRESH_OUTPUT_REQUIRED_OR_EXPLICIT_RESUME")
                load_compatible_pretrained(locked_model, args.pretrained, output / "pretrained_load_report.json",
                                          expected_sha256=packet["inputs"]["pretrained"]["sha256"])
        signatures = {"run_signature": packet["run_signature"], "data": file_ref(args.manifest),
                      "config": file_ref(args.config), "pretrained": file_ref(args.pretrained),
                      "branch": args.branch, "code": packet["runtime_code"]}
        training_started = True
        result = train_candidate(model, dataset, TrainingConfig(**config_dict["training"]), output,
                                 signatures, resume=args.resume, device=f"cuda:{local_rank}",
                                 initializer=initialize_weights,
                                 authorization_expires_at=packet["admitted_authorization_expires_at"])
        if rank == 0:
            _write_once(output / f"RESULT_{result['optimizer_updates']:06d}.json", result)
            print(json.dumps(result, ensure_ascii=False))
        return 0
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc), "training_started": training_started,
                          "model_loaded": model_loaded}, ensure_ascii=False), file=sys.stderr)
        return 2
    finally:
        if distributed_group_started:
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
