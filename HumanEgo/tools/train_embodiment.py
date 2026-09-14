#!/usr/bin/env python3
"""Safe entry point for HumanEgo-Kai22 and HumanEgo-Wuji20."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("DISABLE_ADDMM_CUDA_LT", "1")
os.environ.setdefault("TORCH_BLAS_PREFER_CUBLASLT", "0")

import torch
import yaml

torch.backends.mha.set_fastpath_enabled(False)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from preprocess.retarget_labels.schema import EMBODIMENTS, validate_sidecar  # noqa: E402
from training.FlowMatchingDataloader import FlowMatchingDataloader, MPSSessions  # noqa: E402
from training.FlowMatchingModel import FlowMatchingModel  # noqa: E402
from training.FlowMatchingTrainer import TrainConfig  # noqa: E402
from training.checkpoint_transfer import load_compatible_pretrained  # noqa: E402
from utils.frozen_contract import (  # noqa: E402
    build_formal_run_manifest,
    file_reference,
    read_ordinary_file_bytes,
    sessions_for_role,
    validate_directory_path,
    validate_formal_runtime_contract,
)
from utils.atomic_io import atomic_write_json  # noqa: E402
from utils.source_contract import (  # noqa: E402
    load_eligible68_frozen_split,
    require_manifest_selector_ready as require_reviewed_selector,
    training_evaluation_protocol_fields,
    validate_frame_images,
    validate_training_run_role_contract,
)
from tools.validate_matched_training_configs import (  # noqa: E402
    bind_selector_image_name,
    require_matched_batch_size,
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def session_path(production_root: Path, session_id: str) -> Path:
    return production_root / session_id / "09_humanego_adapter"


def require_manifest_selector_ready(
    *,
    check_only: bool,
    selector_manifest: Path | None = None,
    paired_kept_manifest: Path | None = None,
    artifact_root: Path | None = None,
) -> dict[str, object]:
    """Validate the same manifest pair for preflight and actual training."""
    del check_only  # check-only is not allowed to bypass source validation.
    if selector_manifest is None or paired_kept_manifest is None:
        return require_reviewed_selector()
    return require_reviewed_selector(
        file_reference(selector_manifest),
        file_reference(paired_kept_manifest),
        artifact_root=artifact_root,
    )


def formal_training_status(
    *, cuda_available: bool, selector_ready: bool = False
) -> dict[str, object]:
    """Report the phase gate independently from hardware availability."""
    return {
        "cuda_available": cuda_available,
        "formal_training_ready": bool(selector_ready and cuda_available),
        "formal_training_blocker": (
            None if selector_ready else "HOLD_MANIFEST_SELECTOR_REQUIRED"
        ),
    }


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def git_provenance() -> dict[str, object]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        text=True, capture_output=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT, check=True, text=True, capture_output=True,
    ).stdout
    return {
        "git_head": head,
        "tracked_worktree_dirty": bool(status.strip()),
        "git_status_includes_untracked": True,
    }


def resolve_trainer_config(
    reviewed_config: dict,
    *,
    train_paths: list[Path],
    validation_paths: list[Path],
    run_dir: Path,
    run_root: Path,
    selected_batch: int,
    evaluation_batch: int,
    epochs: int,
    overfit_session: str | None,
    scratch: bool,
) -> dict:
    """Mirror the exact YAML+CLI values passed to ``FlowMatchingTrainer``."""
    cfg = TrainConfig()
    aliases = {
        "enable_aug_jitter": "enable_aug_target_jittering",
        "enable_aug_stride": "enable_aug_temporal_stride",
        "model_h_weighting": "model_horizon_weighting",
        "model_h_beta": "model_horizon_beta",
    }
    for source_key, value in reviewed_config.items():
        key = aliases.get(source_key, source_key)
        if key in {"exp", "job"}:
            continue
        if hasattr(cfg, key):
            if key == "image_size" and isinstance(value, list):
                value = tuple(value)
            setattr(cfg, key, value)
    cfg.MPS_PATHS_TRAIN = [str(path) for path in train_paths]
    cfg.MPS_PATHS_EVAL = [str(path) for path in validation_paths]
    cfg.out_dir = str(run_dir)
    cfg.run_root = str(run_root)
    cfg.device = "cuda"
    cfg.batch_size = selected_batch
    cfg.eval_batch_size = evaluation_batch
    cfg.epochs = epochs
    cfg.skip_visual_eval = True
    if overfit_session:
        cfg.eval_every = 5
        cfg.warmup_steps = 10
        cfg.ema_decay = 0.9
        cfg.dropout = 0.0
        cfg.enable_augmentation = False
        cfg.enable_aug_img = False
        cfg.enable_aug_rrc = False
        cfg.enable_aug_target_jittering = False
        cfg.enable_aug_cutout = False
        cfg.enable_aug_temporal_stride = False
        cfg.enable_aug_interpolation = False
    if scratch:
        cfg.pretrained_checkpoint = None
    return asdict(cfg)


def check_only(
    name: str,
    config: dict,
    sessions: list[Path],
    output: Path,
    selector_binding: dict[str, object],
    sidecar_sha256_by_session: dict[str, str],
) -> dict:
    spec = EMBODIMENTS[name]
    sidecar_root = resolve_repo_path(config["robot_sidecar_root"])
    sidecar_reports = {}
    for session in sessions:
        session_id = session.parent.name
        sidecar = sidecar_root / name / session_id / "sidecar.npz"
        sidecar_reports[session_id] = validate_sidecar(sidecar, spec)
        expected_sidecar_sha256 = sidecar_sha256_by_session.get(session_id)
        if expected_sidecar_sha256 is None:
            raise ValueError(f"frozen split lacks sidecar authority: {session_id}")
        if digest(sidecar) != expected_sidecar_sha256:
            raise ValueError(f"sidecar changed after frozen split validation: {session_id}")
        sidecar_reports[session_id]["sha256"] = expected_sidecar_sha256
        sidecar_reports[session_id]["rgb"] = validate_frame_images(
            session,
            str(config["img_name"]),
            selector_records=selector_binding["selector_records"][session_id],
            selector_root=selector_binding["selector_root"],
        )
    dataset = FlowMatchingDataloader(
        sessions=[MPSSessions(str(path)) for path in sessions],
        image_size=tuple(config["image_size"]), pred_horizon=int(config["pred_horizon"]),
        single_hand=False, max_ict=int(config["max_ict"]), img_name=config["img_name"],
        centric_mode=config["centric_mode"], frame_mode=config["frame_mode"],
        action_mode=config["action_mode"],
        hand_action_representation=config["hand_action_representation"],
        robot_sidecar_root=str(sidecar_root), use_pcd_features=False,
        use_aux_obj_dynamics=False, use_aux_visual_foresight=False,
        use_aux_temporal_contrastive=False, enable_augmentation=False,
        hand_tracking_method="hawor_v3", use_legacy_image_loading=False,
        cache_json_in_memory=True,
        allowed_window_starts=selector_binding["window_starts"],
        selector_records=selector_binding["selector_records"],
        selector_root=selector_binding["selector_root"],
        sidecar_sha256_by_session=sidecar_sha256_by_session,
    )
    if not len(dataset):
        raise RuntimeError("Dataset produced zero valid windows")
    sample = dataset[0]
    expected_action = spec.dual_action_dim
    if tuple(sample["y_action"].shape) != (50, expected_action):
        raise ValueError(f"action shape {tuple(sample['y_action'].shape)}")
    if tuple(sample["action_valid_mask"].shape) != (50, expected_action):
        raise ValueError("action_valid_mask shape mismatch")
    expected_limits = (2, spec.joint_count)
    if tuple(sample["joint_lower"].shape) != expected_limits:
        raise ValueError("joint limit metadata shape mismatch")
    if not torch.all(sample["joint_upper"] > sample["joint_lower"]):
        raise ValueError("joint limits are empty or reversed")
    model = FlowMatchingModel(
        single_hand=False, pred_horizon=50, max_ict=int(config["max_ict"]),
        img_size=tuple(config["image_size"]), patch_size=int(config["patch_size"]),
        vision_embed_dim=int(config["vision_embed_dim"]),
        num_decoder_layers=int(config["num_decoder_layers"]),
        num_heads=int(config["num_heads"]), mlp_ratio=float(config["mlp_ratio"]),
        dropout=float(config["dropout"]), use_pcd_features=False,
        use_aux_obj_dynamics=False, use_aux_visual_foresight=False,
        use_aux_temporal_contrastive=False, use_region_attn=bool(config["use_region_attn"]),
        use_pre_norm=True, use_ctx_norm=True, use_done_in_flow=False,
        hand_action_representation=config["hand_action_representation"],
    )
    transfer_path = output / "pretrained_load_report.json"
    transfer = load_compatible_pretrained(
        model, resolve_repo_path(config["pretrained_checkpoint"]), transfer_path
    )
    model.eval()
    with torch.no_grad():
        result = model(
            x_rgb=sample["x_rgb"].unsqueeze(0),
            x_ict=sample["x_ict"].unsqueeze(0),
            ict_mask=sample["ict_mask"].unsqueeze(0),
            x_robot_state=sample["x_robot_state"].unsqueeze(0),
            robot_state_mask=sample["robot_state_mask"].unsqueeze(0),
            x_t=torch.zeros(1, 50, expected_action), t=torch.zeros(1, 1),
            anchor_uv=sample["anchor_uv"].unsqueeze(0),
        )
    if tuple(result["v_pred"].shape) != (1, 50, expected_action):
        raise ValueError("model output shape mismatch")
    report = {
        "embodiment": name, "check_only": True, "sessions": sidecar_reports,
        "dataset_windows": len(dataset), "action_dim": expected_action,
        "model_output_shape": list(result["v_pred"].shape),
        "robot_state_shape": list(sample["x_robot_state"].shape),
        "joint_limit_shape": list(sample["joint_lower"].shape),
        "pretrained_loaded_fraction": transfer["loaded_parameter_fraction"],
        **formal_training_status(
            cuda_available=torch.cuda.is_available(), selector_ready=True
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "check_only.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embodiment", required=True, choices=sorted(EMBODIMENTS))
    parser.add_argument(
        "--config", type=Path, required=True,
        help="Explicit reviewed training config; implicit legacy configs are forbidden",
    )
    parser.add_argument(
        "--split", type=Path, required=True,
        help="Explicit frozen session split/sidecar manifest",
    )
    parser.add_argument(
        "--run-name",
        help="Isolated run/check-only/performance name; defaults to the config filename",
    )
    parser.add_argument(
        "--production-root", type=Path, required=True,
        help="Explicit reviewed adapter root; implicit legacy roots are forbidden",
    )
    parser.add_argument("--overfit-session")
    parser.add_argument(
        "--scratch", action="store_true",
        help="Run an explicitly named scratch ablation without pretrained weights",
    )
    parser.add_argument(
        "--epochs", type=int,
        help="Runtime epoch override (overfit defaults to 30; formal defaults to YAML)",
    )
    parser.add_argument(
        "--batch-size", type=int,
        help="Freeze a specific training batch size instead of running CUDA autotune",
    )
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--selector-manifest", type=Path, required=True)
    parser.add_argument("--paired-kept-manifest", type=Path, required=True)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        required=True,
        help="Caller-approved root declared by the selected selector manifest",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="Existing ordinary directory that will contain the formal run",
    )
    args = parser.parse_args()
    if not args.run_root.is_absolute():
        parser.error("--run-root must be an absolute canonical path")
    if not args.artifact_root.is_absolute():
        parser.error("--artifact-root must be an absolute canonical path")
    provenance = None
    if not args.check_only:
        provenance = git_provenance()
        if provenance["tracked_worktree_dirty"]:
            raise RuntimeError("formal training requires a clean tracked Git worktree")
    run_root = validate_directory_path(
        args.run_root,
        allowed_roots=[args.run_root.absolute().anchor],
        label="caller-approved formal run root",
    )
    selector_binding = require_manifest_selector_ready(
        check_only=args.check_only,
        selector_manifest=args.selector_manifest,
        paired_kept_manifest=args.paired_kept_manifest,
        artifact_root=args.artifact_root,
    )
    config_candidate = args.config if args.config.is_absolute() else ROOT / args.config
    config_path, config_encoded = read_ordinary_file_bytes(
        config_candidate,
        label="reviewed training config",
    )
    reviewed_config_root = (ROOT / "cfg" / "training").resolve()
    if not config_path.is_relative_to(reviewed_config_root):
        raise ValueError(
            f"training config must be versioned below {reviewed_config_root}: {config_path}"
        )
    try:
        config = yaml.safe_load(config_encoded.decode("utf-8"))
    except UnicodeDecodeError as error:
        raise ValueError("reviewed training config is not UTF-8") from error
    config = bind_selector_image_name(
        config, selector_binding["selector_manifest"], check_only=args.check_only
    )
    base_run_name = args.run_name or config_path.stem
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", base_run_name):
        raise ValueError("run name may contain only letters, digits, dot, underscore and dash")
    artifact_key = (
        args.embodiment
        if args.run_name is None and config_path.stem == f"dual_{args.embodiment}"
        else base_run_name
    )
    expected_representation = EMBODIMENTS[args.embodiment].representation
    if config.get("hand_action_representation") != expected_representation:
        raise ValueError("config/embodiment representation mismatch")
    if config.get("single_hand") is not False or int(config.get("pred_horizon", 0)) != 50:
        raise ValueError("formal contract requires dual hands and H=50")
    split_path = args.split.resolve()
    sidecar_root = resolve_repo_path(config["robot_sidecar_root"])
    split = load_eligible68_frozen_split(
        split_path,
        args.embodiment,
        sidecar_root=sidecar_root,
        verify_sidecars=True,
    )
    production_root = args.production_root.resolve()
    declared_production_root = split.get("production_root")
    if not isinstance(declared_production_root, str) or (
        Path(declared_production_root).resolve() != production_root
    ):
        raise ValueError(
            "explicit production root differs from the frozen split; a new "
            "manifest-backed selector is required for a different image domain"
        )
    if args.overfit_session:
        if args.overfit_session not in sessions_for_role(split, "train"):
            raise ValueError("overfit diagnostics are restricted to frozen train sessions")
        train_ids = [args.overfit_session]
        validation_ids = [args.overfit_session]
    else:
        train_ids = sessions_for_role(split, "train")
        validation_ids = sessions_for_role(split, "validation")
        overlap = sorted(set(train_ids) & set(validation_ids))
        if overlap and not args.check_only:
            raise ValueError(f"train/validation session leakage: {overlap}")
    session_ids = list(dict.fromkeys(train_ids + validation_ids))
    sessions = [session_path(production_root, value) for value in session_ids]
    for session in sessions:
        if not session.is_dir():
            raise FileNotFoundError(session)
        session_id = session.parent.name
        frozen = split["sessions"][session_id]
        if Path(frozen.get("adapter", "")).resolve() != session.resolve():
            raise ValueError(f"adapter path differs from frozen split: {session_id}")
        status_path = production_root / session_id / "status.json"
        expected_status = frozen.get("production_status_sha256")
        if not isinstance(expected_status, str) or digest(status_path) != expected_status:
            raise ValueError(f"production status hash mismatch: {session_id}")
    output = ROOT / "reports" / "check_only" / artifact_key
    # Image content is external to the frozen sidecar SHA.  Reusing an old
    # preflight could therefore hide deleted or changed selector targets.
    sidecar_sha256_by_session = {
        session_id: split["sessions"][session_id]["embodiments"][args.embodiment][
            "sidecar_sha256"
        ]
        for session_id in session_ids
    }
    report = check_only(
        args.embodiment,
        config,
        sessions,
        output,
        selector_binding,
        sidecar_sha256_by_session,
    )
    if args.check_only:
        print(json.dumps(report, indent=2))
    else:
        print(json.dumps({
            "embodiment": args.embodiment,
            "preflight_sessions": len(report["sessions"]),
            "dataset_windows": report["dataset_windows"],
            "action_dim": report["action_dim"],
            "cuda_available": report["cuda_available"],
        }, indent=2))
    if args.check_only:
        return 0
    if args.overfit_session is None:
        require_matched_batch_size(args.batch_size)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA preflight failed; refusing to start formal training")
    performance_path = ROOT / "reports/performance" / f"{artifact_key}_cuda_autotune.json"
    performance = None
    if args.batch_size is not None:
        if args.batch_size < 1:
            raise ValueError("--batch-size must be positive")
        performance = {
            "schema_version": "humanego-cuda-batch-override-v1",
            "embodiment": args.embodiment,
            "config": str(config_path),
            "config_sha256": digest(config_path),
            "device": torch.cuda.get_device_name(0),
            "selected_batch_size": args.batch_size,
            "reason": "explicit command-line override for controlled experiment",
        }
        performance_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(performance_path, performance)
    elif performance_path.is_file():
        candidate = json.loads(performance_path.read_text(encoding="utf-8"))
        if (
            candidate.get("config_sha256") == digest(config_path.resolve())
            and candidate.get("device") == torch.cuda.get_device_name(0)
        ):
            performance = candidate
    if performance is None:
        subprocess.run(
            [sys.executable, str(ROOT / "tools/autotune_training.py"),
             "--embodiment", args.embodiment, "--config", str(config_path),
             "--output", str(performance_path)],
            cwd=ROOT, check=True,
        )
        performance = json.loads(performance_path.read_text(encoding="utf-8"))
    selected_batch = int(performance["selected_batch_size"])
    run_name = base_run_name
    if args.overfit_session:
        run_name += f"_overfit_{args.overfit_session}"
    if args.scratch:
        run_name += "_scratch"
    run_dir = run_root / f"training_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    assert provenance is not None
    evaluation_batch = min(32, selected_batch)
    effective_epochs = args.epochs or (
        30 if args.overfit_session else int(config["epochs"])
    )
    resolved_config = resolve_trainer_config(
        config,
        train_paths=[session_path(production_root, value) for value in train_ids],
        validation_paths=[
            session_path(production_root, value) for value in validation_ids
        ],
        run_dir=run_dir,
        run_root=run_root,
        selected_batch=selected_batch,
        evaluation_batch=evaluation_batch,
        epochs=effective_epochs,
        overfit_session=args.overfit_session,
        scratch=args.scratch,
    )
    manifest_base = {
        **provenance,
        "pretrained_sha256": (
            None if args.scratch
            else digest(resolve_repo_path(config["pretrained_checkpoint"]))
        ),
        "scratch": args.scratch,
        "overfit_session": args.overfit_session,
        "performance_report_sha256": digest(performance_path),
        "selected_batch_size": selected_batch,
        "evaluation_batch_size": evaluation_batch,
        "epochs": effective_epochs,
        "ema_decay": 0.9 if args.overfit_session else float(config["ema_decay"]),
        "selector_manifest_ref": selector_binding["selector_reference"],
        "paired_kept_manifest_ref": selector_binding["paired_kept_reference"],
        "artifact_root": selector_binding["artifact_root"],
        "artifact_roots": selector_binding["artifact_roots"],
        "evaluation_protocol": training_evaluation_protocol_fields(),
    }
    manifest = build_formal_run_manifest(
        split=split,
        split_path=split_path,
        config_path=config_path,
        resolved_config=resolved_config,
        embodiment=args.embodiment,
        sidecar_root=sidecar_root,
        production_root=production_root,
        run_root=run_root,
        run_directory=run_dir,
        base_fields=manifest_base,
    )
    validate_training_run_role_contract(manifest, split, allow_overfit=True)
    validate_formal_runtime_contract(
        resolved_config,
        manifest,
        split,
        args.embodiment,
        config_root=reviewed_config_root,
    )
    manifest_path = run_dir / "run_manifest.json"
    if os.path.lexists(manifest_path):
        _, previous_encoded = read_ordinary_file_bytes(
            manifest_path,
            label="existing formal run manifest",
        )
        try:
            previous = json.loads(previous_encoded)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("existing formal run manifest is not valid JSON") from error
        if previous != manifest:
            raise RuntimeError("existing run manifest differs; refusing unsafe auto-resume")
    else:
        atomic_write_json(manifest_path, manifest)
    lock_path = run_dir / "training.lock"
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Training lock is held: {lock_path}") from error
        command = [
            sys.executable, "-m", "training.FlowMatchingTrainer", "--use_cfg",
            "--task", "grap_a_cap", "--job", config_path.stem,
            "--train_data", *[str(session_path(production_root, value)) for value in train_ids],
            "--eval_data", *[str(session_path(production_root, value)) for value in validation_ids],
            "--device", "cuda", "--batch_size", str(selected_batch),
            "--eval_batch_size", str(min(32, selected_batch)),
            "--out_dir", str(run_dir),
            "--run_root", str(run_root),
            "--skip_visual_eval",
        ]
        epochs = args.epochs or (30 if args.overfit_session else None)
        if epochs is not None:
            command.extend(["--epochs", str(epochs)])
        if args.overfit_session:
            command.extend([
                "--eval_every", "5", "--warmup_steps", "10",
                "--ema_decay", "0.9", "--dropout", "0.0",
                "--disable_augmentation",
            ])
        if args.scratch:
            command.append("--no_pretrained")
        return subprocess.run(command, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
