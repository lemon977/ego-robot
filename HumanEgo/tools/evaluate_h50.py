#!/usr/bin/env python3
"""Frozen, block-open-loop H=50 evaluator for direct robot-q checkpoints."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import inspect
import sys
from pathlib import Path

os.environ.setdefault("DISABLE_ADDMM_CUDA_LT", "1")
os.environ.setdefault("TORCH_BLAS_PREFER_CUBLASLT", "0")

import numpy as np
import torch
torch.backends.mha.set_fastpath_enabled(False)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from inference.embodiment_policy import sample_h50  # noqa: E402
from preprocess.retarget_labels.schema import EMBODIMENTS, validate_sidecar  # noqa: E402
from training.FlowMatchingDataloader import FlowMatchingDataloader, MPSSessions  # noqa: E402
from training.FlowMatchingModel import FlowMatchingModel  # noqa: E402
from utils.atomic_io import atomic_write, atomic_write_json  # noqa: E402
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


OBSERVATION_FIELDS = (
    "x_rgb", "x_ict", "ict_mask", "x_robot_state", "robot_state_mask", "anchor_uv"
)


def audit_inference_contract() -> dict:
    """Fail closed if deployment sampling gains a forbidden dependency."""
    source = inspect.getsource(sample_h50)
    tree = ast.parse(source)
    names = {node.id.lower() for node in ast.walk(tree) if isinstance(node, ast.Name)}
    attributes = {
        node.attr.lower() for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    forbidden = {"retarget", "retargeter", "hawor", "pico", "y_action", "dataset"}
    hits = sorted(forbidden & (names | attributes))
    if hits:
        raise RuntimeError(f"forbidden symbol in sample_h50 inference path: {hits}")
    return {
        "sampler": "inference.embodiment_policy.sample_h50",
        "sampler_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "allowed_observation_fields": list(OBSERVATION_FIELDS),
        "forbidden_dependencies": ["HaWoR", "PICO", "retargeter", "future GT"],
        "static_forbidden_symbol_hits": hits,
        "future_gt_usage": "scoring only, after the complete H50 prediction is returned",
    }


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def rotation_matrix(o6d: np.ndarray) -> np.ndarray:
    # HumanEgo stores the first two rotation columns by flattening a 3x2
    # matrix: [R00,R01,R10,R11,R20,R21].  Keep this decoder identical to
    # utils.utils_math.rot6d_to_R_batch; splitting the first/last three values
    # would silently evaluate a different 6-D convention.
    columns = np.asarray(o6d, dtype=np.float64).reshape(*o6d.shape[:-1], 3, 2)
    first = columns[..., 0]
    first /= np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1e-8)
    raw_second = columns[..., 1]
    second = raw_second - (first * raw_second).sum(-1, keepdims=True) * first
    second /= np.maximum(np.linalg.norm(second, axis=-1, keepdims=True), 1e-8)
    return np.stack((first, second, np.cross(first, second)), axis=-1)


def rotation_error_deg(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    relative = np.swapaxes(rotation_matrix(predicted), -1, -2) @ rotation_matrix(target)
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(cosine))


def block_fde_mm(fde_metres: list[float]) -> float:
    """Return mean final-displacement error in millimetres.

    Position errors are accumulated in metres throughout evaluation.  Keep the
    unit conversion at the metric boundary so the JSON/CSV ``_mm`` contract
    cannot silently expose metres.
    """
    return float(np.mean(fde_metres) * 1000.0) if fde_metres else 0.0


def make_model(cfg: dict) -> FlowMatchingModel:
    names = {
        "single_hand", "pred_horizon", "max_ict", "patch_size",
        "vision_embed_dim", "num_decoder_layers", "num_heads", "mlp_ratio",
        "dropout", "use_pcd_features", "use_aux_obj_dynamics",
        "use_aux_visual_foresight", "use_aux_temporal_contrastive",
        "use_region_attn", "use_pre_norm", "use_ctx_norm", "use_done_in_flow",
        "hand_action_representation",
    }
    values = {name: cfg[name] for name in names if name in cfg}
    values["img_size"] = tuple(cfg["image_size"])
    values["horizon_weighting"] = cfg.get("model_horizon_weighting", "uniform")
    values["horizon_beta"] = cfg.get("model_horizon_beta", 0.0)
    return FlowMatchingModel(**values)


def make_dataset(
    cfg: dict,
    sessions: list[Path],
    sidecar_root: Path,
    stats: dict,
    allowed_window_starts=None,
    selector_records=None,
    selector_root=None,
    sidecar_sha256_by_session=None,
) -> FlowMatchingDataloader:
    return FlowMatchingDataloader(
        sessions=[MPSSessions(str(path)) for path in sessions],
        image_size=tuple(cfg["image_size"]), pred_horizon=50, single_hand=False,
        max_ict=cfg["max_ict"], img_name=cfg["img_name"],
        centric_mode=cfg["centric_mode"], frame_mode=cfg["frame_mode"],
        action_mode=cfg["action_mode"],
        hand_action_representation=cfg["hand_action_representation"],
        robot_sidecar_root=str(sidecar_root), use_pcd_features=cfg["use_pcd_features"],
        use_aux_obj_dynamics=False, use_aux_visual_foresight=False,
        use_aux_temporal_contrastive=False, enable_augmentation=False,
        hand_tracking_method="hawor_v3", use_legacy_image_loading=False,
        cache_json_in_memory=True, stats=stats, seed=cfg.get("seed", 7),
        allowed_window_starts=allowed_window_starts,
        selector_records=selector_records,
        selector_root=selector_root,
        sidecar_sha256_by_session=sidecar_sha256_by_session,
    )


def block_schedule(dataset: FlowMatchingDataloader) -> list[int]:
    result = []
    for index, path in enumerate(dataset.samples):
        if int(Path(path).parent.name) % 50 == 0:
            result.append(index)
    return result


def atomic_csv(
    path: Path, fieldnames: list[str] | tuple[str, ...], rows: list[dict]
) -> None:
    def write(temporary: Path) -> None:
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    atomic_write(path, write)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    def write(temporary: Path) -> None:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)

    atomic_write(path, write)


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
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--selector-manifest", type=Path, required=True)
    parser.add_argument("--paired-kept-manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument(
        "--visualization-session",
        default="",
        help="also save strict-H50 blocks for final rendering; empty disables",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.check_only and args.evaluation_role == "final_test":
        parser.error("--check-only cannot consume the one-shot final-test role")
    selector_binding = require_manifest_selector_ready(
        file_reference(args.selector_manifest),
        file_reference(args.paired_kept_manifest),
        artifact_root=args.artifact_root,
    )
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"formal best checkpoint is missing: {args.checkpoint}")
    if args.checkpoint.name != "best.pt":
        raise ValueError("formal H=50 evaluation accepts validation-selected best.pt only")
    if not args.split.is_file():
        raise FileNotFoundError(f"frozen shared split is missing: {args.split}")
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
        label="frozen evaluation split",
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
        evaluator="H50_BLOCK_OPEN_LOOP_V1",
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
    if manifest.get("selector_manifest_ref") != selector_binding["selector_reference"]:
        raise ValueError("checkpoint selector manifest differs from evaluation CLI")
    if manifest.get("paired_kept_manifest_ref") != selector_binding["paired_kept_reference"]:
        raise ValueError("checkpoint paired-kept manifest differs from evaluation CLI")
    if manifest.get("artifact_root") != selector_binding["artifact_root"]:
        raise ValueError("checkpoint artifact root differs from evaluation CLI")
    if payload["cfg"].get("img_name") != selector_binding["image_name"]:
        raise ValueError("checkpoint image selector differs from selector manifest")
    expected_representation = EMBODIMENTS[args.embodiment].representation
    if payload["cfg"].get("hand_action_representation") != expected_representation:
        raise ValueError("checkpoint action representation mismatch")
    declared_production_root = split.get("production_root")
    if (
        not isinstance(declared_production_root, str)
        or Path(declared_production_root).resolve() != args.production_root.resolve()
    ):
        raise ValueError("evaluation production root differs from frozen split")
    visualization_session = args.visualization_session.strip()
    if visualization_session and visualization_session not in session_ids:
        raise ValueError(
            "visualization session must belong to the explicitly selected split role"
        )
    inference_contract = audit_inference_contract()

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
        # Older formal manifests froze only train/validation sidecars.  Test
        # sidecars are still immutably covered by the hash-verified shared
        # split, while newer manifests additionally duplicate them under
        # evaluation_sidecars for a self-contained checkpoint audit trail.
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
    if visualization_session:
        visualization_sidecar = sidecars[visualization_session]
    preflight = {
        "protocol": "observe only at frames 0,50,100,...; one H50 action block per observation",
        "reobservation_period_frames": 50,
        "forbidden_inside_block": [
            "RGB re-observation", "future HaWoR/PICO", "future robot q",
            "future action GT", "retargeting",
        ],
        "inference_contract": inference_contract,
        "embodiment": args.embodiment,
        "checkpoint_sha256": checkpoint_reference["sha256"],
        "split_sha256": manifest["split_ref"]["sha256"],
        "evaluation_role": args.evaluation_role,
        "final_test_authorized": evaluation_authorization["final_test_authorized"],
        "sessions": sidecars, "source_reports": source_reports,
        "visualization_session": visualization_session or None,
        "visualization_sidecar": visualization_sidecar,
    }
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
    atomic_write_json(output / "preflight.json", preflight)
    if args.check_only:
        print(json.dumps({**preflight, "status": "preflight_passed"}, indent=2))
        return 0

    cfg = payload["cfg"]
    dataset_session_ids = list(session_ids)
    sessions = [
        args.production_root / value / "09_humanego_adapter"
        for value in dataset_session_ids
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
    indices = block_schedule(dataset)
    model = make_model(cfg).to(args.device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    horizons = {key: np.zeros(50) for key in ("pos_sum", "rot_sum", "q_sum", "count")}
    fde, limit_values, velocity_values, acceleration_values, jerk_values = [], [], [], [], []
    blocks = []
    prediction_blocks: list[np.ndarray] = []
    target_blocks: list[np.ndarray] = []
    valid_blocks: list[np.ndarray] = []
    lower_blocks: list[np.ndarray] = []
    upper_blocks: list[np.ndarray] = []
    command_dim = EMBODIMENTS[args.embodiment].joint_count
    pos_std = np.asarray(stats["pos"]["std"], dtype=np.float64)
    for batch_start in range(0, len(indices), args.batch_size):
        batch_indices = indices[batch_start:batch_start + args.batch_size]
        samples = [dataset[index] for index in batch_indices]
        observation = {
            key: torch.stack([sample[key] for sample in samples]).to(args.device)
            for key in OBSERVATION_FIELDS
        }
        seeds = [cfg.get("seed", 7) + batch_start + offset for offset in range(len(samples))]
        result = sample_h50(
            model, observation, seed=seeds,
            steps=cfg.get("num_inference_steps", 20),
        )
        predictions = result["action"].cpu().numpy()
        for offset, (sample, predicted) in enumerate(zip(samples, predictions)):
            target = sample["y_action"].numpy()
            valid = sample["action_valid_mask"].numpy().astype(bool)
            lower = sample["joint_lower"].numpy().reshape(-1)
            upper = sample["joint_upper"].numpy().reshape(-1)
            # The complete prediction already exists at this point.  Future
            # GT is copied only into an offline scoring/rendering artifact and
            # never re-enters ``sample_h50`` or a block observation.
            prediction_blocks.append(predicted.astype(np.float32, copy=True))
            target_blocks.append(target.astype(np.float32, copy=True))
            valid_blocks.append(valid.astype(bool, copy=True))
            lower_blocks.append(lower.astype(np.float32, copy=True))
            upper_blocks.append(upper.astype(np.float32, copy=True))
            block_session = Path(sample["json_path"]).parents[4].name
            block_role = (
                "evaluation" if block_session in session_ids else "visualization"
            )
            blocks.append({
                "session": block_session,
                "frame": int(Path(sample["json_path"]).parent.name),
                "seed": seeds[offset],
                "role": block_role,
            })
            if block_role != "evaluation":
                continue
            q_range = np.maximum(upper - lower, 1e-6)
            block_pos = []
            for horizon in range(50):
                hand_valid = [valid[horizon, 0], valid[horizon, 3]]
                for hand in range(2):
                    if not hand_valid[hand]:
                        continue
                    p_slice = slice(hand * 3, hand * 3 + 3)
                    r_slice = slice(6 + hand * 6, 6 + hand * 6 + 6)
                    q_slice = slice(18 + hand * command_dim, 18 + (hand + 1) * command_dim)
                    pos_error = float(np.linalg.norm((predicted[horizon, p_slice] - target[horizon, p_slice]) * pos_std))
                    rot_error = float(rotation_error_deg(predicted[horizon, r_slice][None], target[horizon, r_slice][None])[0])
                    q_error = float(np.mean(np.abs(predicted[horizon, q_slice] - target[horizon, q_slice]) /
                                            q_range[hand * command_dim:(hand + 1) * command_dim]))
                    horizons["pos_sum"][horizon] += pos_error
                    horizons["rot_sum"][horizon] += rot_error
                    horizons["q_sum"][horizon] += q_error
                    horizons["count"][horizon] += 1
                    block_pos.append((horizon, pos_error))
            q_pred = predicted[:, 18:18 + 2 * command_dim]
            q_valid = valid[:, 18:18 + 2 * command_dim]
            limit_values.append(float((((q_pred < lower) | (q_pred > upper)) & q_valid).sum() / max(1, q_valid.sum())))
            normalized_q = q_pred / q_range
            for order, bucket in ((1, velocity_values), (2, acceleration_values), (3, jerk_values)):
                bucket.append(float(np.abs(np.diff(normalized_q, n=order, axis=0)).mean()))
            if block_pos:
                last_horizon = max(item[0] for item in block_pos)
                fde.append(float(np.mean([value for horizon, value in block_pos if horizon == last_horizon])))
    rows = []
    active = horizons["count"] > 0
    if not prediction_blocks:
        raise RuntimeError("evaluation produced no H50 prediction blocks")
    if not active.any() or float(horizons["count"].sum()) <= 0:
        raise RuntimeError("evaluation produced no valid frozen-target observations")
    for horizon in range(50):
        count = horizons["count"][horizon]
        rows.append({"horizon": horizon + 1, "valid_hands": int(count),
                     "wrist_position_ade_mm": horizons["pos_sum"][horizon] / max(1, count) * 1000,
                     "wrist_rotation_ade_deg": horizons["rot_sum"][horizon] / max(1, count),
                     "q_normalized_mae": horizons["q_sum"][horizon] / max(1, count)})
    atomic_csv(output / "per_horizon.csv", list(rows[0]), rows)
    prediction_artifact = output / "h50_predictions.npz"
    atomic_npz(
        prediction_artifact,
        predictions=np.stack(prediction_blocks),
        targets=np.stack(target_blocks),
        valid=np.stack(valid_blocks),
        joint_lower=np.stack(lower_blocks),
        joint_upper=np.stack(upper_blocks),
        session=np.asarray([block["session"] for block in blocks]),
        frame=np.asarray([block["frame"] for block in blocks], dtype=np.int64),
        seed=np.asarray([block["seed"] for block in blocks], dtype=np.int64),
        role=np.asarray([block["role"] for block in blocks]),
        action_layout=np.asarray(
            ["left_pos3,right_pos3,left_rot6,right_rot6,left_q,right_q"]
        ),
        future_gt_usage=np.asarray(
            ["offline scoring/rendering only after complete H50 inference"]
        ),
    )
    evaluation = {**preflight, "blocks": blocks, "metrics": {
        "wrist_position_ade_mm": float(horizons["pos_sum"][active].sum() / horizons["count"][active].sum() * 1000),
        "wrist_rotation_ade_deg": float(horizons["rot_sum"][active].sum() / horizons["count"][active].sum()),
        "block_fde_mm": block_fde_mm(fde),
        "q_normalized_mae": float(horizons["q_sum"][active].sum() / horizons["count"][active].sum()),
        "joint_limit_ratio": float(np.mean(limit_values)),
        "normalized_velocity": float(np.mean(velocity_values)),
        "normalized_acceleration": float(np.mean(acceleration_values)),
        "normalized_jerk": float(np.mean(jerk_values)),
    }, "prediction_artifact": {
        "path": str(display_output / prediction_artifact.name),
        "sha256": sha256(prediction_artifact),
        "contains_future_gt": True,
        "future_gt_usage": (
            "offline scoring and final visual comparison only; created after "
            "each complete H50 inference block"
        ),
    }, "limitations": ["FK palm/fingertip metrics require the frozen renderer/FK adapter and are not fabricated here"]}
    atomic_write_json(output / "evaluation.json", evaluation)
    summary_rows = [
        {"metric": key, "value": value}
        for key, value in evaluation["metrics"].items()
    ]
    atomic_csv(output / "evaluation.csv", ("metric", "value"), summary_rows)
    if output_lease is not None:
        output_lease.close()
    print(json.dumps(evaluation, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
