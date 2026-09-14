#!/usr/bin/env python3
"""Compare CK-poker/CK-chips against frozen CK1/CK2 on the newtask eval split."""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

import torch
from torch.utils.data import DataLoader

HUMANEGO = Path("/mnt/workspace/code/chaoyang/HumanEgo")
sys.path.insert(0, str(HUMANEGO))

from training.FlowMatchingDataloader import FlowMatchingDataloader, MPSSessions  # noqa: E402
from training.FlowMatchingTrainer import (  # noqa: E402
    FlowMatchingModel,
    TrainConfig,
    eval_ode_inference,
)
import importlib.util  # noqa: E402

_HELPER = Path("/mnt/workspace/code/chaoyang/NOW/daemon/tools/train_newtask_robot.py")
_SPEC = importlib.util.spec_from_file_location("train_newtask_robot", _HELPER)
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)
adapter_paths = _MOD.adapter_paths
load_config = _MOD.load_config
selector_binding = _MOD.selector_binding

CK1 = HUMANEGO / "artifacts/frozen_checkpoints/kai22/best.pt"
CK2 = HUMANEGO / "artifacts/retarget_ab_checkpoints/r2/kai22/best.pt"


def inspect_training_ict(checkpoint: Path) -> dict:
    """Fail-closed audit that does not deserialize the large checkpoint."""
    history_path = checkpoint.parent / "train_history.json"
    if not history_path.is_file():
        return {"status": "UNVERIFIED", "reason": str(history_path)}
    try:
        history = json.loads(history_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {"status": "UNVERIFIED", "reason": str(error)}
    ratios = [
        float(row["zero_state_ratio"])
        for row in history
        if isinstance(row, dict)
        and isinstance(row.get("zero_state_ratio"), (int, float))
    ]
    if not ratios or not np.isfinite(ratios).all():
        return {"status": "UNVERIFIED", "reason": "no finite zero_state_ratio"}
    if min(ratios) >= 1.0 - 1e-8:
        return {
            "status": "ZERO_ICT", "epochs_audited": len(ratios),
            "latest_zero_state_ratio": ratios[-1],
        }
    return {
        "status": "VALID", "epochs_audited": len(ratios),
        "latest_zero_state_ratio": ratios[-1],
    }


def build_model(cfg: TrainConfig) -> FlowMatchingModel:
    return FlowMatchingModel(
        single_hand=cfg.single_hand,
        pred_horizon=cfg.pred_horizon,
        max_ict=cfg.max_ict,
        img_size=cfg.image_size,
        patch_size=cfg.patch_size,
        vision_embed_dim=cfg.vision_embed_dim,
        num_decoder_layers=cfg.num_decoder_layers,
        num_heads=cfg.num_heads,
        mlp_ratio=cfg.mlp_ratio,
        dropout=cfg.dropout,
        horizon_weighting=getattr(cfg, "model_horizon_weighting",
                                  getattr(cfg, "model_h_weighting", "uniform")),
        horizon_beta=getattr(cfg, "model_horizon_beta",
                             getattr(cfg, "model_h_beta", 0.0)),
        use_pcd_features=cfg.use_pcd_features,
        use_aux_obj_dynamics=cfg.use_aux_obj_dynamics,
        use_aux_visual_foresight=cfg.use_aux_visual_foresight,
        use_aux_temporal_contrastive=cfg.use_aux_temporal_contrastive,
        use_region_attn=cfg.use_region_attn,
        use_pre_norm=cfg.use_pre_norm,
        use_ctx_norm=cfg.use_ctx_norm,
        use_done_in_flow=cfg.use_done_in_flow,
        hand_action_representation=cfg.hand_action_representation,
    ).to(cfg.device)


def restore_checkpoint_config(payload: dict, overrides: dict) -> TrainConfig:
    """Use checkpoint-owned model/data semantics, not mutable YAML defaults."""
    raw = payload.get("cfg")
    if not isinstance(raw, dict):
        raise ValueError("new checkpoint has no cfg mapping")
    required = {
        "single_hand", "single_hand_side", "pred_horizon", "max_ict",
        "image_size", "img_name", "centric_mode", "frame_mode", "action_mode",
        "hand_tracking_method", "hand_action_representation",
        "use_pcd_features", "use_aux_obj_dynamics",
        "use_aux_visual_foresight", "use_aux_temporal_contrastive",
        "use_region_attn", "use_pre_norm", "use_ctx_norm",
        "use_done_in_flow", "patch_size", "vision_embed_dim",
        "num_decoder_layers", "num_heads", "mlp_ratio", "dropout",
    }
    missing = sorted(required - raw.keys())
    if missing:
        raise ValueError(f"new checkpoint cfg lacks input-contract fields: {missing}")
    fields = {item.name for item in dataclasses.fields(TrainConfig)}
    values = {key: value for key, value in raw.items() if key in fields}
    values.update({key: value for key, value in overrides.items() if key in fields})
    return TrainConfig(**values)


def load_weights(
    model: FlowMatchingModel,
    path: Path,
    device: str,
    payload: dict | None = None,
) -> None:
    if payload is None:
        payload = torch.load(path, map_location="cpu")
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state, strict=False)


def eval_one(
    name: str,
    ckpt: Path,
    cfg: TrainConfig,
    loader: DataLoader,
    payload: dict | None = None,
) -> dict:
    if not ckpt.is_file():
        return {"name": name, "path": str(ckpt), "status": "MISSING"}
    model = build_model(cfg)
    load_weights(model, ckpt, cfg.device, payload)
    model.eval()
    metrics = eval_ode_inference(model, loader, cfg=cfg, max_batches=None)
    return {"name": name, "path": str(ckpt), "status": "OK", "metrics": metrics}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--new-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    binding, sidecars, split = selector_binding(args.bundle.resolve())
    eval_paths = adapter_paths(split, "test") or adapter_paths(split, "validation")
    ict_audit = inspect_training_ict(args.new_checkpoint)
    if args.check_only:
        preflight = {
            "checkpoint": str(args.new_checkpoint.resolve()),
            "eval_sessions": split.get("test") or split["validation"],
            "ict_training_audit": ict_audit,
            "status": (
                "READY" if ict_audit["status"] == "VALID"
                else f"BLOCKED_{ict_audit['status']}"
            ),
        }
        print(json.dumps(preflight, indent=2), flush=True)
        return 0 if preflight["status"] == "READY" else 2
    if ict_audit["status"] != "VALID":
        raise RuntimeError(
            f"evaluation blocked by ICT audit: {ict_audit['status']}"
        )
    new_payload = torch.load(args.new_checkpoint, map_location="cpu")
    cfg = restore_checkpoint_config(new_payload, {
        "task": split["task"],
        "MPS_PATHS_TRAIN": adapter_paths(split, "train"),
        "MPS_PATHS_EVAL": eval_paths,
        "robot_sidecar_root": split["sidecar_root"],
        "batch_size": 64,
        "eval_batch_size": 32,
        "use_legacy_image_loading": False,
        "device": args.device,
        "num_workers": args.num_workers,
        "out_dir": str(args.output.parent),
        "run_root": str(args.output.parent),
    })
    stats = json.loads(
        (args.new_checkpoint.parent / "dataset_stats.json").read_text(encoding="utf-8"))
    ds = FlowMatchingDataloader(
        sessions=[MPSSessions(path) for path in eval_paths],
        image_size=cfg.image_size,
        pred_horizon=cfg.pred_horizon,
        single_hand=cfg.single_hand,
        single_hand_side=cfg.single_hand_side,
        max_ict=cfg.max_ict,
        img_name=cfg.img_name,
        centric_mode=cfg.centric_mode,
        frame_mode=cfg.frame_mode,
        action_mode=cfg.action_mode,
        hand_action_representation=cfg.hand_action_representation,
        robot_sidecar_root=cfg.robot_sidecar_root,
        use_pcd_features=cfg.use_pcd_features,
        use_aux_obj_dynamics=cfg.use_aux_obj_dynamics,
        use_aux_visual_foresight=cfg.use_aux_visual_foresight,
        use_aux_temporal_contrastive=cfg.use_aux_temporal_contrastive,
        enable_augmentation=False,
        hand_tracking_method=cfg.hand_tracking_method,
        use_legacy_image_loading=False,
        seed=cfg.seed,
        stats=stats,
        allowed_window_starts=binding["window_starts"],
        selector_records=binding["selector_records"],
        selector_root=binding["selector_root"],
        sidecar_sha256_by_session=sidecars,
    )
    loader = DataLoader(
        ds, batch_size=cfg.eval_batch_size, shuffle=False,
        num_workers=cfg.num_workers,
    )
    rows = [
        eval_one(args.tag, args.new_checkpoint, cfg, loader, new_payload),
        eval_one("CK1", CK1, cfg, loader),
        eval_one("CK2", CK2, cfg, loader),
    ]
    payload = {
        "task": split["task"],
        "eval_sessions": split.get("test") or split["validation"],
        "n_samples": len(ds),
        "ict_training_audit": ict_audit,
        "checkpoint_input_contract": {
            key: getattr(cfg, key)
            for key in (
                "single_hand", "single_hand_side", "pred_horizon", "max_ict",
                "image_size", "img_name", "centric_mode", "frame_mode",
                "action_mode", "hand_tracking_method",
                "hand_action_representation",
            )
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2)[:4000], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
