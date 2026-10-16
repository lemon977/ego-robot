"""SHA-bound complete-state successors; no model import before admission."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from .production_resume_gate import (TRAINING_SCOPE, authorize_production_resume,
    parent_refs_from_packet, production_resume_storage_ready, validate_resume_config)
from .production_training_cli import validate_admitted_inputs, export_dataset_envelope
from .derived_training_cli import _check_caches, _rank_error
from .derived_training_gate import bind_cuda_visible_devices
from .runtime_gate import file_ref, runtime_cache_paths
from .derived_runtime import write_once


def repository_root():
    return Path(__file__).resolve().parents[3]


def lineage_signatures(packet, qualification):
    """Preserve data/initial-pretrain lineage while binding the new executor."""
    inputs = packet["inputs"]
    return {"run_signature": packet["run_signature"], "data": inputs["production_manifest"],
        "config": inputs["training_config"], "pretrained": inputs["pretrained"],
        "pretrained_config": inputs["pretrained_config"], "branch": packet["branch"],
        "code": packet["runtime_code"], "supervision_class": "DERIVED_RETARGET",
        "control_ground_truth": False, "prepared_pair": qualification["prepared_pair_ref"],
        "training_scope": TRAINING_SCOPE,
        "storage_capacity_observation": inputs.get("storage_capacity_observation")}


def final_result(result, packet, parent_step, adaptation_ref, storage):
    result = dict(result)
    if type(result.get("optimizer_updates")) is not int or result["optimizer_updates"] <= parent_step:
        raise ValueError("SUCCESSOR_MUST_COMPLETE_NEW_REAL_OPTIMIZER_UPDATES")
    result.update(supervision_class="DERIVED_RETARGET", control_ground_truth=False,
        physical_deployment_authorized=False, branch=packet["branch"], training_scope=TRAINING_SCOPE,
        task_id=packet["task_id"], run_signature=packet["run_signature"], storage_admission=storage,
        initialization="RESUME_COMPLETE_TRAINING_STATE", parent_optimizer_updates=parent_step,
        new_optimizer_updates=result["optimizer_updates"] - parent_step,
        resume_adaptation=adaptation_ref, parent_checkpoint=packet["inputs"]["resume_checkpoint"])
    return result


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    for name in ("task-packet", "manifest", "config", "checkpoint", "output-dir"):
        train.add_argument("--" + name, type=Path, required=True)
    train.add_argument("--branch", choices=("raw", "robotized"), required=True)
    args = parser.parse_args(argv)
    model_loaded = training_started = False
    try:
        encoded = args.config.read_bytes()
        config = json.loads(encoded)
        world = int(os.environ.get("WORLD_SIZE", "1"))
        validate_resume_config(config, world)
        root, output = repository_root(), args.output_dir.resolve()

        def reauthorize():
            return authorize_production_resume(root, args.task_packet, argv, manifest=args.manifest,
                config=args.config, checkpoint=args.checkpoint, output=output, branch=args.branch,
                updates=config["training"]["max_updates"], world_size=world)

        packet = reauthorize()
        # This finite successor is exact on the already-observed one-H20 topology.
        # Changing topology is a different experiment, never an exact resume.
        if world != 1 or int(os.environ.get("RANK", "0")) != 0 or int(os.environ.get("LOCAL_RANK", "0")) != 0:
            raise ValueError("EXACT_SINGLE_H20_PARENT_TOPOLOGY_REQUIRED")
        from .production_data import validate_production_manifest
        qualification = validate_production_manifest(args.manifest, args.branch)
        validate_admitted_inputs(config, encoded, packet, qualification)
        storage = production_resume_storage_ready(output, config["minimum_free_output_bytes"],
            packet=packet, root=root)
        bind_cuda_visible_devices(packet)
        if os.environ.get("ICT_PRODUCTION_RESUME_ADMISSION_ONLY") == "1":
            print(json.dumps({"status": "ADMITTED_COMPLETE_STATE_RESUME_ONLY", "training_started": False}))
            return 0
        _check_caches(output, root)
        for path in runtime_cache_paths(output, root):
            path.mkdir(parents=True, exist_ok=True)
        reauthorize()
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        os.environ.setdefault("DISABLE_ADDMM_CUDA_LT", "1")
        os.environ.setdefault("TORCH_BLAS_PREFER_CUBLASLT", "0")
        import torch
        torch.set_num_threads(2)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("WAIT_GPU_RESOURCE")
        torch.cuda.set_device(0)
        from .model import Revo2ICTModel
        from .production_data import ProductionDerivedICTDataset
        from .training import TrainingConfig, train_candidate
        from .production_resume import prepare_successor_checkpoint
        dataset = ProductionDerivedICTDataset(args.manifest, args.branch, "train")
        dataset.verify_all_dependencies()
        reauthorize()
        signatures = lineage_signatures(packet, qualification)
        training_config = TrainingConfig(**config["training"])
        resume, adaptation = prepare_successor_checkpoint(parent_refs_from_packet(packet), root=root,
            full_config_ref=file_ref(args.config), training_config=training_config, signatures=signatures,
            output_dir=output, dataset_length=len(dataset), world_size=world, device_type="cuda")
        # The adapter verifies and seals full state before any model construction.
        model = Revo2ICTModel(**config["model"])
        model_loaded = True
        reauthorize()
        training_started = True
        result = train_candidate(model, dataset, training_config, output, signatures,
            resume=resume, device="cuda:0", initializer=None,
            authorization_expires_at=packet["admitted_authorization_expires_at"] - config["minimum_export_margin_seconds"])
        dataset.verify_all_dependencies()
        adaptation_ref = file_ref(output / "SUCCESSOR_RESUME_ADAPTATION.json")
        parent = json.loads(Path(packet["inputs"]["parent_inner_result"]["path"]).read_bytes())
        result = final_result(result, packet, parent["optimizer_updates"], adaptation_ref, storage)
        result_path = output / f"RESULT_{result['optimizer_updates']:06d}.json"
        write_once(result_path, result, root)
        from .checkpoint_export import export_tensor_candidate
        pair = qualification["prepared_pair"]
        normal = output / "inference_normalization.json"
        envelope = output / "export_dataset.json"
        write_once(normal, pair["normalization"], root)
        write_once(envelope, export_dataset_envelope(qualification, file_ref(args.manifest)), root)
        provenance = {"task_id": packet["task_id"], "run_signature": packet["run_signature"],
            "branch": args.branch, "supervision_class": "DERIVED_RETARGET", "state_origin": "LIVE_MODEL_STATE",
            "optimizer_updates": result["optimizer_updates"], "dataset_ref": file_ref(envelope),
            "production_manifest_ref": file_ref(args.manifest), "training_scope": TRAINING_SCOPE,
            "training_config_ref": file_ref(args.config), "training_result_ref": file_ref(result_path),
            "code_refs": packet["runtime_code"], "resume_adaptation_ref": adaptation_ref,
            "parent_checkpoint_ref": packet["inputs"]["resume_checkpoint"],
            "parent_optimizer_updates": parent["optimizer_updates"],
            "new_optimizer_updates": result["new_optimizer_updates"]}

        def authorizer(value):
            if value != provenance:
                raise ValueError("EXPORT_SUCCESSOR_PRODUCER_BINDING_MISMATCH")
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
        return 0
    except (ValueError, KeyError, TypeError, OSError, RuntimeError, ImportError) as error:
        print(json.dumps({"status": "BLOCKED", "reason": str(error), "model_loaded": model_loaded,
            "training_started": training_started, "control_ground_truth": False,
            "physical_deployment_authorized": False, "training_scope": TRAINING_SCOPE}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
