#!/usr/bin/env python3
"""Produce checkpoint-bound exact78 visual-A/B inference tensors.

Validation is CPU-only and writes no NPZ when bundle/checkpoint inputs are not
yet real. Actual inference accepts one frozen evaluation session, uses its
selector-bound RGB and shared formal Robot action sidecar, and embeds both the
checkpoint SHA and action-sidecar SHA for the comparison renderer.
"""
from __future__ import annotations

import argparse
import dataclasses
from datetime import datetime, timedelta
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
HUMANEGO = PROJECT / "HumanEgo"
TRAIN_TOOL = HUMANEGO / "tools/train_newtask_robot.py"
LEASE = PROJECT / "_run/GPU_LEASE.json"
LOCK = PROJECT / "_run/GPU_LEASE.lock"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def load_train_module():
    spec = importlib.util.spec_from_file_location("exact78_train_tool", TRAIN_TOOL)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load training tool")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def now() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def lease_acquire(holder: str, max_seconds: int) -> dict[str, Any]:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        current = json.loads(LEASE.read_text(encoding="utf-8")) if LEASE.is_file() else {"status": "RELEASED"}
        if current.get("status") != "RELEASED":
            raise RuntimeError(f"GPU lease held by {current.get('holder')}")
        acquired_at = datetime.now(ZoneInfo("Asia/Shanghai"))
        value = {
            "schema_version": "gpu-lease-v1", "status": "ACQUIRED", "holder": holder,
            "holder_pid": os.getpid(), "worker_pid": os.getpid(), "requester": str(Path(__file__).resolve()),
            "since": acquired_at.isoformat(),
            "scope": {"gpu_indices": [0], "purpose": "exact78 paired checkpoint inference", "max_wall_seconds": max_seconds, "expires_at": (acquired_at + timedelta(seconds=max_seconds)).isoformat()},
        }
        atomic_json(LEASE, value)
        return value


def lease_release(holder: str, reason: str) -> dict[str, Any]:
    with LOCK.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        current = json.loads(LEASE.read_text(encoding="utf-8"))
        if current.get("status") != "ACQUIRED" or current.get("holder") != holder or current.get("holder_pid") != os.getpid():
            raise RuntimeError("refuse to release another GPU lease")
        value = {**current, "status": "RELEASED", "released_at": now(), "release_reason": reason}
        atomic_json(LEASE, value)
        return value


def restore_cfg(raw: dict[str, Any]):
    sys.path.insert(0, str(HUMANEGO))
    from training.FlowMatchingTrainer import TrainConfig
    fields = {item.name for item in dataclasses.fields(TrainConfig)}
    return TrainConfig(**{key: value for key, value in raw.items() if key in fields})


def validate_static(args: argparse.Namespace) -> dict[str, Any]:
    module = load_train_module()
    bundle = args.bundle.resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    if checkpoint.name != "best.pt":
        raise ValueError("inference requires validation-selected best.pt")
    binding, sidecars, split = module.selector_binding(bundle)
    module.exact78_split_binding(split)
    module.bundle_session_set_gate(split, binding, sidecars)
    robot_gate = module.robot_window_gate(split, binding)
    hawor_root, hawor_sidecars = module.hawor_binding(bundle, split)
    object_source = module.object_binding(bundle, split)
    run_manifest_path = checkpoint.parent / "run_manifest.json"
    training_complete_path = checkpoint.parent / "training_complete.json"
    stats_path = checkpoint.parent / "dataset_stats.json"
    for path in (run_manifest_path, training_complete_path, stats_path):
        if not path.is_file():
            raise ValueError(f"checkpoint provenance missing: {path}")
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    complete = json.loads(training_complete_path.read_text(encoding="utf-8"))
    expected_branch = split.get("visual_branch")
    if (
        split.get("task") != args.task or expected_branch != args.branch
        or run_manifest.get("task") != args.task or run_manifest.get("visual_branch") != args.branch
        or run_manifest.get("pairing_id") != split.get("pairing_id")
        or complete.get("status") != "complete"
    ):
        raise ValueError("checkpoint task/branch/pairing/training provenance mismatch")
    import torch
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint_payload, dict) or checkpoint_payload.get("run_manifest") != run_manifest:
        raise ValueError("checkpoint embedded run_manifest mismatch")
    cfg = restore_cfg(checkpoint_payload.get("cfg", {}))
    eval_sessions = module.admitted_sessions(split, "validation")
    session = args.session or (eval_sessions[0] if eval_sessions else "")
    if session not in eval_sessions:
        raise ValueError("inference session must be in frozen admitted validation split")
    if sidecars.get(session) is None:
        raise ValueError("session Robot action sidecar SHA missing")
    return {
        "module": module, "bundle": bundle, "checkpoint": checkpoint,
        "binding": binding, "sidecars": sidecars, "split": split,
        "robot_gate": robot_gate, "hawor_root": hawor_root, "hawor_sidecars": hawor_sidecars,
        "object_source": object_source, "checkpoint_payload": checkpoint_payload,
        "run_manifest": run_manifest, "stats_path": stats_path, "cfg": cfg,
        "session": session,
        "provenance": {"checkpoint": ref(checkpoint), "run_manifest": ref(run_manifest_path), "training_complete": ref(training_complete_path), "dataset_stats": ref(stats_path)},
    }


def infer(args: argparse.Namespace, audit: dict[str, Any]) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader
    sys.path.insert(0, str(HUMANEGO))
    from training.FlowMatchingDataloader import FlowMatchingDataloader, MPSSessions
    from training.FlowMatchingTrainer import FlowMatchingModel

    cfg = audit["cfg"]
    cfg.device = args.device
    binding, split, session = audit["binding"], audit["split"], audit["session"]
    stats = json.loads(audit["stats_path"].read_text(encoding="utf-8"))
    object_source = audit["object_source"]
    production = Path(split["production_root"])
    dataset = FlowMatchingDataloader(
        sessions=[MPSSessions(str(production / session / "09_humanego_adapter"))],
        image_size=cfg.image_size, pred_horizon=cfg.pred_horizon,
        single_hand=cfg.single_hand, single_hand_side=cfg.single_hand_side,
        max_ict=cfg.max_ict, img_name=cfg.img_name, centric_mode=cfg.centric_mode,
        frame_mode=cfg.frame_mode, action_mode=cfg.action_mode,
        hand_action_representation=cfg.hand_action_representation,
        robot_sidecar_root=cfg.robot_sidecar_root,
        hawor_v3_sidecar_root=audit["hawor_root"],
        hawor_v3_sha256_by_session={session: audit["hawor_sidecars"][session]},
        use_pcd_features=cfg.use_pcd_features,
        use_aux_obj_dynamics=cfg.use_aux_obj_dynamics,
        use_aux_visual_foresight=cfg.use_aux_visual_foresight,
        use_aux_temporal_contrastive=cfg.use_aux_temporal_contrastive,
        enable_augmentation=False, hand_tracking_method=cfg.hand_tracking_method,
        use_legacy_image_loading=False, cache_json_in_memory=False,
        cache_image_bytes_in_memory=False, seed=cfg.seed, stats=stats,
        allowed_window_starts={session: binding["window_starts"][session]},
        selector_records={session: binding["selector_records"][session]},
        selector_root=binding["selector_root"],
        sidecar_sha256_by_session={session: audit["sidecars"][session]},
        object_state_sidecar_root=object_source["root"],
        object_state_npz_sha256_by_session={session: object_source["npz_digests"][session]},
        object_state_json_sha256_by_session={session: object_source["json_digests"][session]},
        object_state_consumption_mode=object_source["mode"],
        object_state_confidence_thresholds=object_source["thresholds"],
    )
    if len(dataset) < 1:
        raise RuntimeError("no selector-bound validation samples")
    model = FlowMatchingModel(
        single_hand=cfg.single_hand, pred_horizon=cfg.pred_horizon, max_ict=cfg.max_ict,
        img_size=cfg.image_size, patch_size=cfg.patch_size,
        vision_embed_dim=cfg.vision_embed_dim, num_decoder_layers=cfg.num_decoder_layers,
        num_heads=cfg.num_heads, mlp_ratio=cfg.mlp_ratio, dropout=cfg.dropout,
        horizon_weighting=getattr(cfg, "model_horizon_weighting", getattr(cfg, "model_h_weighting", "uniform")),
        horizon_beta=getattr(cfg, "model_horizon_beta", getattr(cfg, "model_h_beta", 0.0)),
        use_pcd_features=cfg.use_pcd_features, use_aux_obj_dynamics=cfg.use_aux_obj_dynamics,
        use_aux_visual_foresight=cfg.use_aux_visual_foresight,
        use_aux_temporal_contrastive=cfg.use_aux_temporal_contrastive,
        use_region_attn=cfg.use_region_attn, use_pre_norm=cfg.use_pre_norm,
        use_ctx_norm=cfg.use_ctx_norm, use_done_in_flow=cfg.use_done_in_flow,
        hand_action_representation=cfg.hand_action_representation,
    ).to(args.device)
    model.load_state_dict(audit["checkpoint_payload"]["model"], strict=True)
    model.eval()
    indices = list(range(min(len(dataset), args.max_samples)))
    plans=[]; targets=[]; valid=[]; paths=[]; frames=[]; seeds=[]
    with torch.no_grad():
        for index in indices:
            batch = next(iter(DataLoader(torch.utils.data.Subset(dataset, [index]), batch_size=1, shuffle=False, num_workers=0)))
            seed = int(cfg.seed) + index
            generator = torch.Generator(device=args.device); generator.manual_seed(seed)
            x_t = torch.randn((1, cfg.pred_horizon, model.action_dim), generator=generator, device=args.device)
            inputs = {key: batch[key].to(args.device) for key in ("x_rgb", "x_ict", "ict_mask")}
            for key in ("x_pcd", "x_robot_state", "robot_state_mask", "anchor_uv"):
                if key in batch and batch[key] is not None:
                    inputs[key] = batch[key].to(args.device)
            dt = 1.0 / cfg.num_inference_steps
            for step in range(cfg.num_inference_steps):
                t = torch.full((1,1), step*dt, device=args.device)
                out = model(x_t=x_t, t=t, **inputs)
                x_t = x_t + out["v_pred"] * dt
            path = Path(batch["json_path"][0])
            plans.append(x_t[0].float().cpu().numpy()); targets.append(batch["y_action"][0].numpy())
            valid.append(batch["action_valid_mask"][0].numpy().astype(bool)); paths.append(str(path.with_name("rgb.png")))
            frames.append(int(path.parent.name)); seeds.append(seed)
    atomic_npz(
        args.output, session=np.asarray([session]*len(plans)), replan_frame=np.asarray(frames),
        seed=np.asarray(seeds), full_plans=np.stack(plans), targets=np.stack(targets),
        valid=np.stack(valid), rgb_path=np.asarray(paths),
        action_sidecar_sha256=np.asarray([audit["sidecars"][session]]),
        checkpoint_sha256=np.asarray([audit["provenance"]["checkpoint"]["sha256"]]),
    )
    return {"samples": len(plans), "session": session, "prediction": ref(args.output)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=("chips","poker"))
    parser.add_argument("--branch", required=True, choices=("HUMAN_RAW_RGB","ROBOT_VIEW_RGB"))
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--session", default="")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-wall-seconds", type=int, default=3600)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.max_samples < 1:
        parser.error("--max-samples must be positive")
    missing = [str(path) for path in (args.bundle,args.checkpoint) if not path.exists()]
    if args.validate_only and missing:
        payload={"schema_version":"exact78-visual-ab-inference-producer-v1","created_at":now(),"status":"PREPARED_WAIT_REAL_BUNDLE_AND_CHECKPOINT","task":args.task,"branch":args.branch,"missing":missing,"prediction_created":False,"gpu_used":False}
        atomic_json(args.result,payload); print(json.dumps(payload,indent=2)); return 0
    if missing:
        parser.error(f"missing inputs: {missing}")
    audit=validate_static(args)
    if args.validate_only:
        payload={"schema_version":"exact78-visual-ab-inference-producer-v1","created_at":now(),"status":"READY_FOR_LEASE_AWARE_INFERENCE","task":args.task,"branch":args.branch,"session":audit["session"],"provenance":audit["provenance"],"prediction_created":False,"gpu_used":False}
        atomic_json(args.result,payload); print(json.dumps(payload,indent=2)); return 0
    if args.output.exists() or args.result.exists():
        raise FileExistsError("output/result already exists")
    holder=f"humanego_inference:{args.task}:{args.branch.lower()}"
    acquired=lease_acquire(holder,args.max_wall_seconds)
    failure=None
    try:
        produced=infer(args,audit)
    except Exception as error:
        failure=f"{type(error).__name__}:{error}"; raise
    finally:
        released=lease_release(holder,"INFERENCE_EXIT" if failure is None else failure)
    payload={"schema_version":"exact78-visual-ab-inference-producer-v1","created_at":now(),"status":"PASS_CHECKPOINT_BOUND_INFERENCE","task":args.task,"branch":args.branch,"provenance":audit["provenance"],**produced,"lease_acquired":acquired,"lease_released":released,"claim_limit":"Offline validation-split inference only; no deployment or task-success claim."}
    atomic_json(args.result,payload); print(json.dumps(payload,indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
