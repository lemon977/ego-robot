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
from pathlib import Path

os.environ.setdefault("DISABLE_ADDMM_CUDA_LT", "1")
os.environ.setdefault("TORCH_BLAS_PREFER_CUBLASLT", "0")

import numpy as np
import torch
torch.backends.mha.set_fastpath_enabled(False)
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from preprocess.retarget_labels.schema import EMBODIMENTS, validate_sidecar  # noqa: E402
from training.FlowMatchingDataloader import FlowMatchingDataloader, MPSSessions  # noqa: E402
from training.FlowMatchingModel import FlowMatchingModel  # noqa: E402
from training.checkpoint_transfer import load_compatible_pretrained  # noqa: E402


PRODUCTION_DEFAULT = Path(
    "/mnt/workspace/code/chaoyang/hand_benchmark/production_runs/"
    "final_v3_grap_a_cap_0812"
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def session_path(production_root: Path, session_id: str) -> Path:
    return production_root / session_id / "09_humanego_adapter"


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def git_provenance() -> dict[str, object]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        text=True, capture_output=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT, check=True, text=True, capture_output=True,
    ).stdout
    return {"git_head": head, "tracked_worktree_dirty": bool(status.strip())}


def validate_raw_rgb(session: Path) -> dict[str, object]:
    paths = sorted((session / "preprocess" / "all_data").glob("[0-9]*/training_data.json"))
    failures = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        raw = data.get("obs", {}).get("rgb_path")
        if not raw or "clean" in Path(raw).name.lower() or "inpaint" in Path(raw).name.lower():
            failures.append(str(path))
    if failures:
        raise ValueError(f"raw RGB contract failed for {len(failures)} frames")
    return {"frames": len(paths), "model_rgb_path": "obs.rgb_path", "clean_rgb_used": False}


def check_only(name: str, config: dict, sessions: list[Path], output: Path) -> dict:
    spec = EMBODIMENTS[name]
    sidecar_root = resolve_repo_path(config["robot_sidecar_root"])
    sidecar_reports = {}
    for session in sessions:
        session_id = session.parent.name
        sidecar = sidecar_root / name / session_id / "sidecar.npz"
        sidecar_reports[session_id] = validate_sidecar(sidecar, spec)
        sidecar_reports[session_id]["sha256"] = digest(sidecar)
        sidecar_reports[session_id]["rgb"] = validate_raw_rgb(session)
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
        "cuda_available": torch.cuda.is_available(),
        "formal_training_ready": torch.cuda.is_available(),
    }
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "check_only.json.tmp"
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output / "check_only.json")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embodiment", required=True, choices=sorted(EMBODIMENTS))
    parser.add_argument(
        "--config", type=Path, required=True,
        help="Explicit reviewed training config; implicit legacy configs are forbidden",
    )
    parser.add_argument(
        "--split", type=Path, default=ROOT / "data_manifests" / "shared_split.json",
        help="Frozen session split/sidecar manifest (defaults to the production split)",
    )
    parser.add_argument(
        "--run-name",
        help="Isolated run/check-only/performance name; defaults to the config filename",
    )
    parser.add_argument("--production-root", type=Path, default=PRODUCTION_DEFAULT)
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
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
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
    if config.get("img_name") != "rgb.png" or config.get("use_legacy_image_loading") is not False:
        raise ValueError("formal baseline must use normal RGB")
    split_path = args.split.resolve()
    split = {}
    if args.overfit_session:
        train_ids = [args.overfit_session]
        validation_ids = [args.overfit_session]
    else:
        if not split_path.is_file():
            if args.check_only:
                # The documented zero-argument preflight remains useful before
                # a cohort split exists; it is explicitly not a formal split.
                train_ids = ["grap_a_cap_004"]
                validation_ids = ["grap_a_cap_004"]
            else:
                raise FileNotFoundError(
                    "Formal shared split is not frozen. Use --overfit-session for the P0 gate."
                )
        else:
            split = json.loads(split_path.read_text(encoding="utf-8"))
            train_ids = list(dict.fromkeys(split["train"]))
            validation_ids = list(dict.fromkeys(split["validation"]))
        overlap = sorted(set(train_ids) & set(validation_ids))
        if overlap and not args.check_only:
            raise ValueError(f"train/validation session leakage: {overlap}")
    session_ids = list(dict.fromkeys(train_ids + validation_ids))
    evaluation_ids = [] if args.overfit_session else list(dict.fromkeys(split.get("test", [])))
    sessions = [session_path(args.production_root.resolve(), value) for value in session_ids]
    for session in sessions:
        if not session.is_dir():
            raise FileNotFoundError(session)
    output = ROOT / "reports" / "check_only" / artifact_key
    cached_report_path = output / "check_only.json"
    report = None
    if not args.check_only and not args.overfit_session and cached_report_path.is_file():
        candidate = json.loads(cached_report_path.read_text(encoding="utf-8"))
        cached_sessions = candidate.get("sessions", {})
        sidecar_root = resolve_repo_path(config["robot_sidecar_root"])
        sidecars_unchanged = (
            set(cached_sessions) == set(session_ids)
            and all(
                cached_sessions[session_id].get("sha256")
                == digest(sidecar_root / args.embodiment / session_id / "sidecar.npz")
                for session_id in session_ids
            )
        )
        production_unchanged = all(
            split["sessions"][session_id]["production_status_sha256"]
            == digest(args.production_root.resolve() / session_id / "status.json")
            for session_id in session_ids
        )
        if sidecars_unchanged and production_unchanged:
            report = candidate
            print(f"Reusing hash-verified preflight: {cached_report_path}")
    if report is None:
        report = check_only(args.embodiment, config, sessions, output)
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
        temporary = performance_path.with_suffix(performance_path.suffix + ".tmp")
        temporary.write_text(json.dumps(performance, indent=2) + "\n", encoding="utf-8")
        temporary.replace(performance_path)
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
    run_dir = ROOT / "runs" / "grap_a_cap" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    sidecar_root = resolve_repo_path(config["robot_sidecar_root"])
    manifest = {
        **git_provenance(),
        "embodiment": args.embodiment,
        "config_sha256": digest(config_path.resolve()),
        "split_sha256": digest(split_path) if split_path.is_file() and not args.overfit_session else None,
        "pretrained_sha256": (
            None if args.scratch
            else digest(resolve_repo_path(config["pretrained_checkpoint"]))
        ),
        "scratch": args.scratch,
        "train_sessions": train_ids, "validation_sessions": validation_ids,
        "sidecars": {
            session_id: digest(sidecar_root / args.embodiment / session_id / "sidecar.npz")
            for session_id in session_ids
        },
        "evaluation_sidecars": {
            session_id: digest(sidecar_root / args.embodiment / session_id / "sidecar.npz")
            for session_id in evaluation_ids
        },
        "overfit_session": args.overfit_session,
        "performance_report_sha256": digest(performance_path),
        "selected_batch_size": selected_batch,
        "evaluation_batch_size": min(32, selected_batch),
        "epochs": args.epochs or (30 if args.overfit_session else int(config["epochs"])),
        "ema_decay": 0.9 if args.overfit_session else float(config["ema_decay"]),
    }
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        # Runs started before code provenance was added remain resumable under
        # their exact original manifest; new runs freeze these fields.
        for key in (
            "git_head", "tracked_worktree_dirty", "evaluation_sidecars", "scratch",
        ):
            if key not in previous:
                manifest.pop(key, None)
        if previous != manifest:
            raise RuntimeError("existing run manifest differs; refusing unsafe auto-resume")
    else:
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        temporary.replace(manifest_path)
    lock_path = run_dir / "training.lock"
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Training lock is held: {lock_path}") from error
        command = [
            sys.executable, "-m", "training.FlowMatchingTrainer", "--use_cfg",
            "--task", "grap_a_cap", "--job", config_path.stem,
            "--train_data", *[str(session_path(args.production_root.resolve(), value)) for value in train_ids],
            "--eval_data", *[str(session_path(args.production_root.resolve(), value)) for value in validation_ids],
            "--device", "cuda", "--batch_size", str(selected_batch),
            "--eval_batch_size", str(min(32, selected_batch)),
            "--out_dir", str(run_dir),
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
