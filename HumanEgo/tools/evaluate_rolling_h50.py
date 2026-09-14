#!/usr/bin/env python3
"""Causal rolling-horizon evaluation for dual-hand H=50 checkpoints."""
from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("DISABLE_ADDMM_CUDA_LT", "1")
os.environ.setdefault("TORCH_BLAS_PREFER_CUBLASLT", "0")

import numpy as np
import torch

torch.backends.mha.set_fastpath_enabled(False)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inference.embodiment_policy import sample_receding_h50  # noqa: E402
from inference.receding_horizon import (  # noqa: E402
    CausalPoseRateLimiter,
    RecedingH50QController,
)
from preprocess.retarget_labels.schema import EMBODIMENTS, validate_sidecar  # noqa: E402
from tools.evaluate_h50 import (  # noqa: E402
    OBSERVATION_FIELDS,
    atomic_csv,
    atomic_npz,
    make_dataset,
    make_model,
    rotation_error_deg,
    sha256,
)
from utils.utils_math import o6d_to_rotmat, rotmat_to_o6d  # noqa: E402
from utils.atomic_io import atomic_write_json  # noqa: E402
from utils.frozen_contract import (  # noqa: E402
    file_reference,
    load_frozen_checkpoint_payload,
    validate_checkpoint_bundle,
    validate_file_reference,
    validate_formal_runtime_contract,
)
from utils.source_contract import (  # noqa: E402
    FinalTestOutputLease,
    authorize_evaluation_role,
    claim_final_test_output,
    load_eligible68_frozen_split,
    require_manifest_selector_ready,
    validate_frozen_adapters,
    validate_training_run_role_contract,
    verify_eligible68_split_phase2,
)


def audit_contract() -> dict:
    sources = {
        "sampler": inspect.getsource(sample_receding_h50),
        "q_controller": inspect.getsource(RecedingH50QController),
        "pose_controller": inspect.getsource(CausalPoseRateLimiter),
    }
    forbidden = {"retarget", "retargeter", "hawor", "pico", "y_action", "dataset"}
    hits: dict[str, list[str]] = {}
    digests = {}
    for name, source in sources.items():
        tree = ast.parse(source)
        symbols = {
            node.id.lower() for node in ast.walk(tree) if isinstance(node, ast.Name)
        } | {
            node.attr.lower()
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
        }
        hits[name] = sorted(forbidden & symbols)
        digests[name] = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if any(hits.values()):
        raise RuntimeError(f"forbidden rolling-inference symbols: {hits}")
    return {
        "sampler": "inference.embodiment_policy.sample_receding_h50",
        "q_controller": "inference.receding_horizon.RecedingH50QController",
        "pose_controller": "inference.receding_horizon.CausalPoseRateLimiter",
        "source_sha256": digests,
        "allowed_observation_fields": list(OBSERVATION_FIELDS),
        "forbidden_dependencies": ["HaWoR", "PICO", "retargeter", "future GT"],
        "static_forbidden_symbol_hits": hits,
    }


def validate_checkpoint_inputs(
    args: argparse.Namespace,
    selector_binding: dict,
) -> tuple[dict, dict, list[str], dict, dict, dict]:
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.checkpoint.name != "best.pt":
        raise ValueError("rolling H50 evaluation accepts validation-selected best.pt only")
    if not args.split.is_file():
        raise FileNotFoundError(args.split)
    payload, checkpoint_reference, _ = load_frozen_checkpoint_payload(
        args.checkpoint,
        embodiment=args.embodiment,
    )
    manifest = payload.get("run_manifest")
    if not manifest:
        raise ValueError("checkpoint has no frozen run_manifest")
    if manifest.get("embodiment") != args.embodiment:
        raise ValueError("checkpoint embodiment mismatch")
    split_path = validate_file_reference(
        manifest.get("split_ref", {}),
        allowed_roots=[ROOT / "data_manifests"],
        label="frozen rolling-evaluation split",
    )
    if args.split.absolute() != split_path:
        raise ValueError("CLI split path differs from checkpoint authority")
    if manifest.get("split_sha256") != manifest["split_ref"].get("sha256"):
        raise ValueError("checkpoint/split hash mismatch")
    _, stats = validate_checkpoint_bundle(
        args.checkpoint,
        payload,
        args.split,
        args.embodiment,
        return_dataset_stats=True,
    )
    split = load_eligible68_frozen_split(
        args.split,
        args.embodiment,
        sidecar_root=args.sidecar_root,
        verify_sidecars=False,
    )
    validate_training_run_role_contract(manifest, split)
    gate_reference = (
        file_reference(args.final_test_gate) if args.final_test_gate is not None else None
    )
    evaluation_authorization = authorize_evaluation_role(
        split,
        args.evaluation_role,
        final_test_gate_reference=gate_reference,
        split_reference=manifest.get("split_ref"),
        checkpoint_reference=checkpoint_reference,
        product_line=selector_binding["product_line"],
        embodiment=args.embodiment,
        evaluator="H50_ROLLING_CAUSAL_V1",
        output_directory=args.output,
    )
    validate_formal_runtime_contract(
        payload.get("cfg", {}),
        manifest,
        split,
        args.embodiment,
        config_root=ROOT / "cfg" / "training",
    )
    session_ids = list(evaluation_authorization["sessions"])
    cfg = payload["cfg"]
    expected_representation = EMBODIMENTS[args.embodiment].representation
    if cfg.get("hand_action_representation") != expected_representation:
        raise ValueError("checkpoint action representation mismatch")
    return (
        payload,
        split,
        list(session_ids),
        evaluation_authorization,
        checkpoint_reference,
        stats,
    )


def schedule_indices(
    dataset,
    session_ids: set[str],
    execute_steps: int,
    max_per_session: int,
) -> list[int]:
    counts: dict[str, int] = defaultdict(int)
    result = []
    for index, sample_path in enumerate(dataset.samples):
        path = Path(sample_path)
        session = path.parents[4].name
        frame = int(path.parent.name)
        if session not in session_ids or frame % execute_steps:
            continue
        if max_per_session and counts[session] >= max_per_session:
            continue
        result.append(index)
        counts[session] += 1
    return result


def temporal_derivative(
    trajectories: dict[str, list[tuple[int, np.ndarray, np.ndarray]]],
    order: int,
) -> float:
    values = []
    for records in trajectories.values():
        records = sorted(records, key=lambda item: item[0])
        for start in range(0, len(records) - order):
            window = records[start : start + order + 1]
            frames = [item[0] for item in window]
            if any(right != left + 1 for left, right in zip(frames, frames[1:])):
                continue
            q = np.stack([item[1] for item in window])
            valid = np.stack([item[2] for item in window]).all(axis=0)
            if valid.any():
                values.append(float(np.abs(np.diff(q, n=order, axis=0)[0][valid]).mean()))
    return float(np.mean(values)) if values else 0.0


def read_bound_sample_metadata(dataset, sample_path: str | Path) -> dict:
    """Consume the selector-verified metadata payload already owned by dataset."""
    payload = dataset._read_frame(str(sample_path))
    if not isinstance(payload, dict) or not isinstance(payload.get("metadata"), dict):
        raise ValueError(f"selector-bound sample metadata is invalid: {sample_path}")
    return payload["metadata"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embodiment", required=True, choices=sorted(EMBODIMENTS))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument(
        "--evaluation-role",
        required=True,
        choices=("dev", "final_test"),
        help="dev is ordinary evaluation; final_test additionally requires a one-shot gate",
    )
    parser.add_argument(
        "--final-test-gate",
        type=Path,
        help="Immutable gate binding both frozen twins and the one-shot output",
    )
    parser.add_argument("--sidecar-root", type=Path, required=True)
    parser.add_argument("--production-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--execute-steps", type=int, default=5)
    parser.add_argument("--q-ensemble-decay", type=float, default=0.5)
    parser.add_argument("--max-q-step-deg", type=float)
    parser.add_argument("--max-wrist-step-mm", type=float, default=10.0)
    parser.add_argument("--max-wrist-rotation-step-deg", type=float, default=8.0)
    parser.add_argument("--wrist-ema-alpha", type=float, default=0.65)
    parser.add_argument("--max-replans-per-session", type=int, default=0)
    parser.add_argument(
        "--visualization-session",
        default="",
        help="also save rolling predictions for rendering; empty disables",
    )
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--selector-manifest", type=Path, required=True)
    parser.add_argument("--paired-kept-manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.execute_steps <= 50:
        parser.error("--execute-steps must lie in [1, 50]")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.max_replans_per_session < 0:
        parser.error("--max-replans-per-session cannot be negative")
    if not 0 < args.q_ensemble_decay <= 1:
        parser.error("--q-ensemble-decay must lie in (0, 1]")
    if args.max_q_step_deg is not None and args.max_q_step_deg <= 0:
        parser.error("--max-q-step-deg must be positive")
    if args.max_wrist_step_mm <= 0 or args.max_wrist_rotation_step_deg <= 0:
        parser.error("wrist step limits must be positive")
    if not 0 < args.wrist_ema_alpha <= 1:
        parser.error("--wrist-ema-alpha must lie in (0, 1]")
    if args.check_only and args.evaluation_role == "final_test":
        parser.error("--check-only cannot consume the one-shot final-test role")
    selector_binding = require_manifest_selector_ready(
        file_reference(args.selector_manifest),
        file_reference(args.paired_kept_manifest),
        artifact_root=args.artifact_root,
    )

    (
        payload,
        split,
        session_ids,
        evaluation_authorization,
        checkpoint_reference,
        stats,
    ) = (
        validate_checkpoint_inputs(args, selector_binding)
    )
    manifest = payload["run_manifest"]
    if manifest.get("selector_manifest_ref") != selector_binding["selector_reference"]:
        raise ValueError("checkpoint selector manifest differs from evaluation CLI")
    if manifest.get("paired_kept_manifest_ref") != selector_binding["paired_kept_reference"]:
        raise ValueError("checkpoint paired-kept manifest differs from evaluation CLI")
    if manifest.get("artifact_root") != selector_binding["artifact_root"]:
        raise ValueError("checkpoint artifact root differs from evaluation CLI")
    if payload["cfg"].get("img_name") != selector_binding["image_name"]:
        raise ValueError("checkpoint image selector differs from selector manifest")
    declared_production_root = split.get("production_root")
    if (
        not isinstance(declared_production_root, str)
        or Path(declared_production_root).resolve() != args.production_root.resolve()
    ):
        raise ValueError("evaluation production root differs from frozen split")
    evaluation_sessions = set(session_ids)
    visualization_session = args.visualization_session.strip()
    if visualization_session and visualization_session not in evaluation_sessions:
        raise ValueError(
            "visualization session must belong to the explicitly selected split role"
        )
    inference_contract = audit_contract()

    output_lease: FinalTestOutputLease | None = None
    if evaluation_authorization["final_test_authorized"]:
        claimed = claim_final_test_output(evaluation_authorization, hold=True)
        if not isinstance(claimed, FinalTestOutputLease):
            raise RuntimeError("final-test output claim did not return a held lease")
        output_lease = claimed
    split = verify_eligible68_split_phase2(
        args.split,
        split,
        args.embodiment,
        sidecar_root=args.sidecar_root,
        verify_sidecars=True,
        verify_roles=(str(evaluation_authorization["storage_role"]),),
        allow_final_test=bool(evaluation_authorization["final_test_authorized"]),
    )
    source_reports = validate_frozen_adapters(
        split, args.production_root, session_ids, selector_binding
    )
    sidecars = {}
    for session in session_ids:
        path = args.sidecar_root / args.embodiment / session / "sidecar.npz"
        digest = sha256(path)
        lineage_field = (
            "evaluation_sidecars"
            if evaluation_authorization["final_test_authorized"]
            else "sidecars"
        )
        expected = manifest.get(lineage_field, {}).get(session)
        if expected != digest:
            raise ValueError(f"frozen split/sidecar hash mismatch: {session}")
        sidecars[session] = {
            **validate_sidecar(path, EMBODIMENTS[args.embodiment]),
            "sha256": expected,
        }
    visualization_sidecar = None
    dataset_session_ids = list(session_ids)
    if visualization_session:
        visualization_sidecar = sidecars[visualization_session]
    command_dim = EMBODIMENTS[args.embodiment].joint_count
    max_q_step_deg = args.max_q_step_deg
    if max_q_step_deg is None:
        max_q_step_deg = 10.0 if args.embodiment == "kai22" else 12.0
    display_output = Path(
        str(evaluation_authorization.get("output_directory", args.output.absolute()))
    )
    output = (
        output_lease.bound_directory
        if output_lease is not None
        else display_output
    )
    if output_lease is not None:
        expected_entries = {"FINAL_TEST_CLAIM.json"}
        if not output.is_dir() or {path.name for path in output.iterdir()} != expected_entries:
            raise RuntimeError("final-test output claim changed before evaluation")
    else:
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise FileExistsError(f"evaluation output must be a new empty directory: {output}")
        output.mkdir(parents=True, exist_ok=True)
    preflight = {
        "protocol": "H50 receding-horizon control with causal short-prefix execution",
        "prediction_horizon_frames": 50,
        "reobservation_period_frames": args.execute_steps,
        "executed_prefix_frames": args.execute_steps,
        "q_temporal_ensemble": {
            "causal": True,
            "decay_per_replan": args.q_ensemble_decay,
            "coordinate_contract": "q only; wrist pose always comes from newest camera-frame plan",
        },
        "safety_projection": {
            "joint_limits": True,
            "max_q_step_deg": max_q_step_deg,
            "anchor": "current observed robot q at each replan",
        },
        "pose_continuity": {
            "causal": True,
            "future_frames_used": False,
            "common_frame": "tracked world; deployment equivalent is robot base",
            "max_translation_step_m": args.max_wrist_step_mm / 1000.0,
            "max_rotation_step_deg": args.max_wrist_rotation_step_deg,
            "ema_alpha": args.wrist_ema_alpha,
        },
        "future_gt_usage": "offline scoring only after the complete H50 plan is returned",
        "forbidden_at_inference": ["future GT", "HaWoR", "PICO", "retargeting"],
        "embodiment": args.embodiment,
        "checkpoint_sha256": checkpoint_reference["sha256"],
        "split_sha256": manifest["split_ref"]["sha256"],
        "evaluation_role": args.evaluation_role,
        "final_test_authorized": evaluation_authorization["final_test_authorized"],
        "sessions": sidecars,
        "source_reports": source_reports,
        "visualization_session": visualization_session or None,
        "visualization_sidecar": visualization_sidecar,
        "inference_contract": inference_contract,
        "status": "preflight_passed" if args.check_only else "running",
    }
    atomic_write_json(output / "preflight.json", preflight)
    if args.check_only:
        print(json.dumps(preflight, indent=2))
        return 0

    cfg = payload["cfg"]
    sessions = [
        args.production_root / session / "09_humanego_adapter"
        for session in dataset_session_ids
    ]
    dataset = make_dataset(
        cfg,
        sessions,
        args.sidecar_root,
        stats,
        selector_binding["window_starts"],
        selector_binding["selector_records"],
        selector_binding["selector_root"],
        {session_id: row["sha256"] for session_id, row in sidecars.items()},
    )
    indices = schedule_indices(
        dataset,
        set(dataset_session_ids),
        args.execute_steps,
        args.max_replans_per_session,
    )
    if not indices:
        raise RuntimeError("rolling schedule contains no replans")
    model = make_model(cfg).to(args.device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()

    plans = []
    seed_base = int(cfg.get("seed", 7))
    for batch_start in range(0, len(indices), args.batch_size):
        batch_indices = indices[batch_start : batch_start + args.batch_size]
        samples = [dataset[index] for index in batch_indices]
        observation = {
            key: torch.stack([sample[key] for sample in samples]).to(args.device)
            for key in OBSERVATION_FIELDS
        }
        seeds = [seed_base + index for index in batch_indices]
        result = sample_receding_h50(
            model,
            observation,
            seed=seeds,
            execute_steps=args.execute_steps,
            steps=int(cfg.get("num_inference_steps", 20)),
        )
        predicted = result["plan_action"].cpu().numpy()
        for sample, plan, seed in zip(samples, predicted, seeds):
            path = Path(sample["json_path"])
            metadata = read_bound_sample_metadata(dataset, path)
            plans.append({
                "session": path.parents[4].name,
                "frame": int(path.parent.name),
                "role": (
                    "evaluation"
                    if path.parents[4].name in evaluation_sessions
                    else "visualization"
                ),
                "seed": seed,
                "plan": plan.astype(np.float32),
                "target": sample["y_action"].numpy()[: args.execute_steps].astype(np.float32),
                "valid": sample["action_valid_mask"].numpy()[: args.execute_steps].astype(bool),
                "current_q": sample["x_robot_state"][:, 9 : 9 + command_dim].numpy(),
                "current_state": sample["x_robot_state"].numpy(),
                "robot_state_mask": sample["robot_state_mask"].numpy().astype(bool),
                "c2w": np.asarray(metadata["c2w"], dtype=np.float64),
                "joint_lower": sample["joint_lower"].numpy(),
                "joint_upper": sample["joint_upper"].numpy(),
            })

    grouped: dict[str, list[dict]] = defaultdict(list)
    for plan in plans:
        grouped[plan["session"]].append(plan)

    position_sum = np.zeros(args.execute_steps)
    rotation_sum = np.zeros(args.execute_steps)
    raw_q_sum = np.zeros(args.execute_steps)
    executed_q_sum = np.zeros(args.execute_steps)
    wrist_counts = np.zeros(args.execute_steps)
    q_counts = np.zeros(args.execute_steps)
    raw_limit, clip_ratio, ensemble_size = [], [], []
    pose_translation_steps, pose_rotation_steps = [], []
    pose_translation_clips, pose_rotation_clips = [], []
    trajectories: dict[str, list[tuple[int, np.ndarray, np.ndarray]]] = defaultdict(list)
    artifact_records = []
    pos_std = np.asarray(stats["pos"]["std"], dtype=np.float64)

    for session, session_plans in grouped.items():
        controller = RecedingH50QController(
            command_dim,
            execute_steps=args.execute_steps,
            ensemble_decay=args.q_ensemble_decay,
            max_q_step_rad=np.deg2rad(max_q_step_deg),
        )
        pose_controller = CausalPoseRateLimiter(
            max_translation_step_m=args.max_wrist_step_mm / 1000.0,
            max_rotation_step_rad=np.deg2rad(args.max_wrist_rotation_step_deg),
            ema_alpha=args.wrist_ema_alpha,
        )
        pose_initialized = False
        for record in sorted(session_plans, key=lambda item: item["frame"]):
            executed, diagnostic = controller.process(
                replan_frame=record["frame"],
                plan_action=record["plan"],
                current_q=record["current_q"],
                joint_lower=record["joint_lower"],
                joint_upper=record["joint_upper"],
            )
            if not pose_initialized:
                initial_world = []
                for hand in range(2):
                    state = record["current_state"][hand]
                    pose_ref = np.eye(4, dtype=np.float64)
                    pose_ref[:3, 3] = state[:3] * pos_std + np.asarray(
                        stats["pos"]["mean"], dtype=np.float64
                    )
                    pose_ref[:3, :3] = o6d_to_rotmat(state[3:9])
                    initial_world.append(record["c2w"] @ pose_ref)
                pose_controller.process(
                    np.asarray(initial_world), record["robot_state_mask"]
                )
                pose_initialized = True
            for offset in range(args.execute_steps):
                target_world = []
                active_hands = []
                for hand in range(2):
                    pose_ref = np.eye(4, dtype=np.float64)
                    pose_ref[:3, 3] = (
                        executed[offset, hand * 3 : hand * 3 + 3] * pos_std
                        + np.asarray(stats["pos"]["mean"], dtype=np.float64)
                    )
                    pose_ref[:3, :3] = o6d_to_rotmat(
                        executed[offset, 6 + hand * 6 : 12 + hand * 6]
                    )
                    target_world.append(record["c2w"] @ pose_ref)
                    active_hands.append(bool(
                        record["valid"][offset, hand * 3]
                        and record["valid"][offset, 6 + hand * 6]
                    ))
                smooth_world, pose_diagnostic = pose_controller.process(
                    np.asarray(target_world), np.asarray(active_hands)
                )
                world_to_ref = np.linalg.inv(record["c2w"])
                for hand in range(2):
                    smooth_ref = world_to_ref @ smooth_world[hand]
                    executed[offset, hand * 3 : hand * 3 + 3] = (
                        smooth_ref[:3, 3]
                        - np.asarray(stats["pos"]["mean"], dtype=np.float64)
                    ) / pos_std
                    executed[offset, 6 + hand * 6 : 12 + hand * 6] = rotmat_to_o6d(
                        smooth_ref[:3, :3]
                    )
                if record["role"] == "evaluation":
                    pose_translation_steps.append(pose_diagnostic["translation_step_m"])
                    pose_rotation_steps.append(pose_diagnostic["rotation_step_deg"])
                    pose_translation_clips.append(pose_diagnostic["translation_clipped"])
                    pose_rotation_clips.append(pose_diagnostic["rotation_clipped"])
            artifact_records.append({**record, "executed": executed})
            if record["role"] != "evaluation":
                continue
            raw = record["plan"][: args.execute_steps]
            target = record["target"]
            valid = record["valid"]
            lower = record["joint_lower"].reshape(-1)
            upper = record["joint_upper"].reshape(-1)
            q_range = np.maximum(upper - lower, 1e-6)
            for offset in range(args.execute_steps):
                for hand in range(2):
                    p_slice = slice(hand * 3, hand * 3 + 3)
                    r_slice = slice(6 + hand * 6, 6 + hand * 6 + 6)
                    q_slice = slice(
                        18 + hand * command_dim,
                        18 + (hand + 1) * command_dim,
                    )
                    if valid[offset, p_slice].all() and valid[offset, r_slice].all():
                        position_sum[offset] += float(
                            np.linalg.norm(
                                (executed[offset, p_slice] - target[offset, p_slice])
                                * pos_std
                            )
                        )
                        rotation_sum[offset] += float(rotation_error_deg(
                            executed[offset, r_slice][None],
                            target[offset, r_slice][None],
                        )[0])
                        wrist_counts[offset] += 1
                    q_valid = valid[offset, q_slice]
                    if q_valid.any():
                        hand_range = q_range[
                            hand * command_dim : (hand + 1) * command_dim
                        ]
                        raw_q_sum[offset] += float(np.mean(
                            np.abs(raw[offset, q_slice][q_valid] - target[offset, q_slice][q_valid])
                            / hand_range[q_valid]
                        ))
                        executed_q_sum[offset] += float(np.mean(
                            np.abs(executed[offset, q_slice][q_valid] - target[offset, q_slice][q_valid])
                            / hand_range[q_valid]
                        ))
                        q_counts[offset] += 1
                q_slice_all = slice(18, 18 + 2 * command_dim)
                trajectories[session].append((
                    record["frame"] + offset + 1,
                    executed[offset, q_slice_all] / q_range,
                    valid[offset, q_slice_all],
                ))
            raw_limit.append(diagnostic["raw_joint_limit_ratio"])
            clip_ratio.append(diagnostic["q_step_clip_ratio"])
            ensemble_size.append(diagnostic["plans_in_ensemble"])

    per_step = []
    if not any(item["role"] == "evaluation" for item in artifact_records):
        raise RuntimeError("rolling evaluation produced no frozen-role replans")
    for offset in range(args.execute_steps):
        wrist_count = max(1, wrist_counts[offset])
        q_count = max(1, q_counts[offset])
        per_step.append({
            "executed_step": offset + 1,
            "valid_wrist_hands": int(wrist_counts[offset]),
            "valid_q_hands": int(q_counts[offset]),
            "wrist_position_error_mm": position_sum[offset] / wrist_count * 1000.0,
            "wrist_rotation_error_deg": rotation_sum[offset] / wrist_count,
            "raw_q_normalized_mae": raw_q_sum[offset] / q_count,
            "executed_q_normalized_mae": executed_q_sum[offset] / q_count,
        })
    atomic_csv(output / "per_executed_step.csv", list(per_step[0]), per_step)

    artifact = output / "rolling_h50_predictions.npz"
    atomic_npz(
        artifact,
        full_plans=np.stack([item["plan"] for item in artifact_records]),
        raw_prefix=np.stack([
            item["plan"][: args.execute_steps] for item in artifact_records
        ]),
        executed_prefix=np.stack([item["executed"] for item in artifact_records]),
        targets=np.stack([item["target"] for item in artifact_records]),
        valid=np.stack([item["valid"] for item in artifact_records]),
        session=np.asarray([item["session"] for item in artifact_records]),
        replan_frame=np.asarray([item["frame"] for item in artifact_records]),
        role=np.asarray([item["role"] for item in artifact_records]),
        seed=np.asarray([item["seed"] for item in artifact_records]),
        future_gt_usage=np.asarray(["offline scoring only after H50 plan return"]),
        pose_continuity_applied=np.asarray([True]),
    )
    total_wrist_count = max(1, wrist_counts.sum())
    total_q_count = max(1, q_counts.sum())
    metrics = {
        "wrist_position_error_mm": float(position_sum.sum() / total_wrist_count * 1000.0),
        "wrist_rotation_error_deg": float(rotation_sum.sum() / total_wrist_count),
        "raw_q_normalized_mae": float(raw_q_sum.sum() / total_q_count),
        "executed_q_normalized_mae": float(executed_q_sum.sum() / total_q_count),
        "raw_joint_limit_ratio": float(np.mean(raw_limit)),
        "executed_joint_limit_ratio": 0.0,
        "q_step_clip_ratio": float(np.mean(clip_ratio)),
        "mean_plans_in_q_ensemble": float(np.mean(ensemble_size)),
        "normalized_q_velocity": temporal_derivative(trajectories, 1),
        "normalized_q_acceleration": temporal_derivative(trajectories, 2),
        "normalized_q_jerk": temporal_derivative(trajectories, 3),
        "wrist_translation_step_mm_mean": float(np.mean(pose_translation_steps) * 1000.0),
        "wrist_translation_step_mm_max": float(np.max(pose_translation_steps) * 1000.0),
        "wrist_rotation_step_deg_mean": float(np.mean(pose_rotation_steps)),
        "wrist_translation_clip_hands": int(np.sum(pose_translation_clips)),
        "wrist_rotation_clip_hands": int(np.sum(pose_rotation_clips)),
        "replans": int(sum(item["role"] == "evaluation" for item in artifact_records)),
        "executed_frames": int(
            sum(item["role"] == "evaluation" for item in artifact_records)
            * args.execute_steps
        ),
    }
    evaluation = {
        **preflight,
        "status": "rolling_h50_evaluated",
        "metrics": metrics,
        "prediction_artifact": {
            "path": str(display_output / artifact.name),
            "sha256": sha256(artifact),
            "contains_future_gt": True,
            "future_gt_usage": "offline scoring only",
        },
    }
    atomic_write_json(output / "evaluation.json", evaluation)
    atomic_csv(
        output / "evaluation.csv",
        ("metric", "value"),
        [{"metric": key, "value": value} for key, value in metrics.items()],
    )
    if output_lease is not None:
        output_lease.close()
    print(json.dumps(evaluation, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
