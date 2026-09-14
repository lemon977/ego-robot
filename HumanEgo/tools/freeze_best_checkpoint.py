#!/usr/bin/env python3
"""Validate and immutably snapshot a completed validation-selected best.pt."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from preprocess.retarget_labels.schema import EMBODIMENTS  # noqa: E402
from utils.atomic_io import atomic_directory, atomic_write, atomic_write_json  # noqa: E402
from utils.frozen_contract import (  # noqa: E402
    TRAINING_COMPLETE_SCHEMA,
    load_verified_torch_checkpoint,
    open_verified_file_reference,
    read_file_reference,
    read_ordinary_file_bytes,
    read_runtime_checkpoint_authority,
    runtime_checkpoint_authority_path,
    validate_directory_path,
    validate_file_reference,
    validate_formal_runtime_contract,
)
from utils.source_contract import (  # noqa: E402
    FROZEN_BEST_SCHEMA,
    load_eligible68_frozen_split,
    require_manifest_selector_ready,
    validate_frozen_adapters,
    validate_training_run_role_contract,
)

def validation_score(snapshot: dict) -> float:
    return float(
        snapshot["pos_err_w_m"]
        + snapshot["rot_err_w_deg"] / 100.0
        + snapshot.get("joint_mae_w_normalized", 0.0) * 0.10
    )


def load_completed_checkpoint_authorities(run_dir: Path) -> dict[str, Any]:
    """Resolve best/latest refs only from the trainer's completed-run authority."""
    completion_path = run_dir / "training_complete.json"
    _, completion_encoded = read_ordinary_file_bytes(
        completion_path,
        label="formal training completion authority",
    )
    try:
        completion = json.loads(completion_encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("training completion authority is not valid JSON") from error
    if not isinstance(completion, Mapping):
        raise ValueError("training completion authority must be an object")
    if (
        completion.get("schema_version") != TRAINING_COMPLETE_SCHEMA
        or completion.get("immutable") is not True
        or completion.get("no_fallback") is not True
        or completion.get("status") != "complete"
    ):
        raise ValueError("training is not complete under a frozen authority")

    references: dict[str, Mapping[str, Any]] = {}
    for field in (
        "run_manifest_ref",
        "dataset_stats_ref",
        "best_checkpoint_authority_ref",
        "latest_checkpoint_authority_ref",
    ):
        value = completion.get(field)
        if not isinstance(value, Mapping):
            raise ValueError(f"training completion authority lacks {field}")
        references[field] = value

    run_manifest_path, run_manifest_encoded = read_file_reference(
        references["run_manifest_ref"],
        allowed_roots=[run_dir],
        label="completed-run manifest",
    )
    if run_manifest_path != run_dir / "run_manifest.json":
        raise ValueError("completed-run manifest path drift")
    try:
        run_manifest = json.loads(run_manifest_encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("completed-run manifest is not valid JSON") from error
    if not isinstance(run_manifest, dict) or completion.get("run_manifest") != run_manifest:
        raise ValueError("training completion/run manifest mismatch")

    stats_path = validate_file_reference(
        references["dataset_stats_ref"],
        allowed_roots=[run_dir],
        label="completed-run dataset statistics",
    )
    if stats_path != run_dir / "dataset_stats.json":
        raise ValueError("completed-run dataset-statistics path drift")

    checkpoint_references: dict[str, dict[str, Any]] = {}
    for role in ("best", "latest"):
        checkpoint_path = run_dir / f"{role}.pt"
        authority_reference = references[f"{role}_checkpoint_authority_ref"]
        if authority_reference.get("path") != str(
            runtime_checkpoint_authority_path(checkpoint_path)
        ):
            raise ValueError(f"{role} checkpoint authority path drift")
        authority = read_runtime_checkpoint_authority(
            checkpoint_path,
            authority_reference=authority_reference,
        )
        if authority.get("run_manifest_ref") != references["run_manifest_ref"]:
            raise ValueError(f"{role} checkpoint authority binds another run")
        if authority.get("dataset_stats_ref") != references["dataset_stats_ref"]:
            raise ValueError(f"{role} checkpoint authority binds other dataset statistics")
        checkpoint_reference = authority.get("checkpoint_ref")
        if not isinstance(checkpoint_reference, Mapping):
            raise ValueError(f"{role} checkpoint authority lacks checkpoint_ref")
        validated_path = validate_file_reference(
            checkpoint_reference,
            allowed_roots=[run_dir],
            label=f"completed-run {role} checkpoint",
        )
        if validated_path != checkpoint_path:
            raise ValueError(f"{role} checkpoint authority path drift")
        checkpoint_references[role] = dict(checkpoint_reference)
    if checkpoint_references["best"]["path"] == checkpoint_references["latest"]["path"]:
        raise ValueError("best/latest checkpoint authorities alias one path")
    return {
        "completion": dict(completion),
        "run_manifest": run_manifest,
        "run_manifest_ref": dict(references["run_manifest_ref"]),
        "dataset_stats_ref": dict(references["dataset_stats_ref"]),
        "best_checkpoint_ref": checkpoint_references["best"],
        "latest_checkpoint_ref": checkpoint_references["latest"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embodiment", required=True, choices=sorted(EMBODIMENTS))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--split", type=Path, required=True,
        help="Exact frozen split used by the training run",
    )
    parser.add_argument(
        "--output-root", type=Path, required=True,
        help="Explicit new immutable checkpoint namespace",
    )
    args = parser.parse_args()
    run_dir = validate_directory_path(
        args.run_dir,
        allowed_roots=[args.run_dir.absolute().anchor],
        label="formal training run directory",
    )
    lock_path = run_dir / "training.lock"
    if not lock_path.is_file():
        raise FileNotFoundError(lock_path)
    lock_stream = lock_path.open("r+")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("formal training is still running; refusing to freeze") from error

    best_path = run_dir / "best.pt"
    latest_path = run_dir / "latest.pt"
    split_path = args.split.resolve()
    for required in (
        best_path,
        latest_path,
        split_path,
        run_dir / "dataset_stats.json",
        run_dir / "run_manifest.json",
        run_dir / "config.json",
    ):
        if not required.is_file():
            raise FileNotFoundError(required)
    completed_authority = load_completed_checkpoint_authorities(run_dir)
    best_reference = completed_authority["best_checkpoint_ref"]
    latest_reference = completed_authority["latest_checkpoint_ref"]
    disk_manifest = completed_authority["run_manifest"]
    run_manifest_reference = completed_authority["run_manifest_ref"]
    dataset_stats_reference = completed_authority["dataset_stats_ref"]
    _, best = load_verified_torch_checkpoint(
        best_reference,
        allowed_roots=[run_dir],
        label="validation-selected best checkpoint",
    )
    _, latest = load_verified_torch_checkpoint(
        latest_reference,
        allowed_roots=[run_dir],
        label="latest recovery checkpoint",
    )
    if best.get("model_weights") != "ema":
        raise ValueError("best.pt is not the validation-selected EMA model")
    if best.get("selection_metric") != (
        "wrist_position_m + rotation_deg/100 + normalized_robot_q_term"
    ):
        raise ValueError("unexpected validation selection metric")
    if best.get("epoch") != best.get("best", {}).get("epoch"):
        raise ValueError("best epoch metadata mismatch")
    if int(latest.get("epoch", -1)) < int(best["epoch"]):
        raise ValueError("latest recovery point predates best checkpoint")
    manifest = best.get("run_manifest")
    if not manifest or manifest.get("embodiment") != args.embodiment:
        raise ValueError("checkpoint run manifest/embodiment mismatch")
    frozen_split_path = validate_file_reference(
        manifest.get("split_ref", {}),
        allowed_roots=[ROOT / "data_manifests"],
        label="checkpoint frozen split",
    )
    if frozen_split_path != split_path:
        raise ValueError("CLI split path differs from checkpoint authority")
    if manifest.get("split_sha256") != manifest["split_ref"].get("sha256"):
        raise ValueError("checkpoint/shared split hash mismatch")
    selector_binding = require_manifest_selector_ready(
        manifest.get("selector_manifest_ref"),
        manifest.get("paired_kept_manifest_ref"),
        artifact_root=manifest.get("artifact_root"),
    )
    split = load_eligible68_frozen_split(
        split_path, args.embodiment, verify_sidecars=True
    )
    validate_training_run_role_contract(manifest, split)
    validate_formal_runtime_contract(
        best.get("cfg", {}),
        manifest,
        split,
        args.embodiment,
        config_root=ROOT / "cfg" / "training",
    )
    validate_formal_runtime_contract(
        latest.get("cfg", {}),
        manifest,
        split,
        args.embodiment,
        config_root=ROOT / "cfg" / "training",
    )
    if best.get("cfg", {}).get("img_name") != selector_binding["image_name"]:
        raise ValueError("best checkpoint image selector differs from selector manifest")
    if latest.get("cfg", {}).get("img_name") != selector_binding["image_name"]:
        raise ValueError("latest checkpoint image selector differs from selector manifest")
    validate_frozen_adapters(
        split,
        split["production_root"],
        list(manifest["train_sessions"]) + list(manifest["validation_sessions"]),
        selector_binding,
    )
    if disk_manifest != manifest:
        raise ValueError("embedded and on-disk run manifests differ")
    if latest.get("run_manifest") != manifest:
        raise ValueError("latest checkpoint/run manifest mismatch")
    if best.get("dataset_stats_sha256") != dataset_stats_reference["sha256"]:
        raise ValueError("best checkpoint/dataset statistics hash mismatch")
    if latest.get("dataset_stats_sha256") != dataset_stats_reference["sha256"]:
        raise ValueError("latest checkpoint/dataset statistics hash mismatch")

    snapshot_path = run_dir / "eval_snapshots" / f"eval_ep_{best['epoch']:04d}.json"
    if not snapshot_path.is_file():
        raise FileNotFoundError(snapshot_path)
    _, snapshot_encoded = read_ordinary_file_bytes(
        snapshot_path,
        label="validation snapshot",
    )
    try:
        snapshot = json.loads(snapshot_encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("validation snapshot is not valid JSON") from error
    recomputed_score = validation_score(snapshot)
    selected_score = float(best["best"]["score"])
    if abs(recomputed_score - selected_score) > 1e-9:
        raise ValueError(
            f"validation snapshot score mismatch: {recomputed_score} != {selected_score}"
        )

    _, frozen_run_manifest_encoded = read_file_reference(
        run_manifest_reference,
        allowed_roots=[run_dir],
        label="freeze-source run manifest",
    )
    try:
        frozen_run_manifest = json.loads(frozen_run_manifest_encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("freeze-source run manifest is not valid JSON") from error
    if frozen_run_manifest != disk_manifest:
        raise ValueError("freeze-source run manifest changed after completion")
    _, dataset_stats_encoded = read_file_reference(
        dataset_stats_reference,
        allowed_roots=[run_dir],
        label="freeze-source dataset statistics",
    )
    try:
        dataset_stats = json.loads(dataset_stats_encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("freeze-source dataset statistics are not valid JSON") from error
    if not isinstance(dataset_stats, dict):
        raise ValueError("freeze-source dataset statistics must be an object")
    _, config_encoded = read_ordinary_file_bytes(
        run_dir / "config.json",
        label="freeze-source resolved config",
    )
    try:
        frozen_config = json.loads(config_encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("freeze-source resolved config is not valid JSON") from error
    normalized_best_config = json.loads(json.dumps(best.get("cfg", {}), allow_nan=False))
    normalized_latest_config = json.loads(json.dumps(latest.get("cfg", {}), allow_nan=False))
    if (
        frozen_config != normalized_best_config
        or frozen_config != normalized_latest_config
        or frozen_config != manifest.get("resolved_config")
    ):
        raise ValueError("freeze-source resolved config differs from checkpoint authority")

    frozen_source_bytes = {
        "dataset_stats.json": dataset_stats_encoded,
        "run_manifest.json": frozen_run_manifest_encoded,
        "config.json": config_encoded,
    }

    output_root = validate_directory_path(
        args.output_root,
        allowed_roots=[args.output_root.absolute().anchor],
        label="frozen checkpoint output root",
    )
    output = output_root / args.embodiment
    report: dict = {}

    def build_checkpoint_bundle(staging: Path) -> None:
        frozen_path = staging / "best.pt"

        def copy_checkpoint(temporary: Path) -> None:
            with open_verified_file_reference(
                best_reference,
                allowed_roots=[run_dir],
                label="validation-selected best checkpoint copy",
            ) as (_, source), temporary.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
            _, copied = read_ordinary_file_bytes(
                temporary,
                label="frozen checkpoint staging copy",
            )
            if hashlib.sha256(copied).hexdigest() != best_reference["sha256"]:
                raise RuntimeError("checkpoint copy hash mismatch")

        atomic_write(frozen_path, copy_checkpoint)
        frozen_path.chmod(0o444)
        frozen_sidecars = {}
        frozen_sidecar_refs = {}
        for name, encoded in frozen_source_bytes.items():
            destination = staging / name

            def copy_sidecar(
                temporary: Path, encoded: bytes = encoded
            ) -> None:
                temporary.write_bytes(encoded)

            atomic_write(destination, copy_sidecar)
            destination.chmod(0o444)
            frozen_sidecars[name] = hashlib.sha256(encoded).hexdigest()
            frozen_sidecar_refs[name] = {
                "path": str(output / name),
                "bytes": len(encoded),
                "sha256": frozen_sidecars[name],
            }
        report.update({
            "schema_version": FROZEN_BEST_SCHEMA,
            "immutable": True,
            "no_fallback": True,
            "status": "frozen",
            "embodiment": args.embodiment,
            "product_line": selector_binding["product_line"],
            "source": str(best_path.resolve()),
            "frozen_checkpoint": str(output / "best.pt"),
            "checkpoint_ref": {
                "path": str(output / "best.pt"),
                "bytes": best_reference["bytes"],
                "sha256": best_reference["sha256"],
            },
            "checkpoint_sha256": best_reference["sha256"],
            "epoch": int(best["epoch"]),
            "latest_epoch_at_freeze": int(latest["epoch"]),
            "validation_score": selected_score,
            "validation_snapshot": str(snapshot_path.resolve()),
            "validation_snapshot_sha256": hashlib.sha256(snapshot_encoded).hexdigest(),
            "model_weights": best["model_weights"],
            "selection_metric": best["selection_metric"],
            "selection_role": "dev",
            "selection_sessions": list(manifest["validation_sessions"]),
            "final_test_status": "NOT_EVALUATED",
            "run_schema_version": manifest["schema_version"],
            "run_root": manifest["run_root"],
            "run_directory": manifest["run_directory"],
            "frozen_run_manifest_ref": frozen_sidecar_refs["run_manifest.json"],
            "split_ref": manifest["split_ref"],
            "config_ref": manifest["config_ref"],
            "selector_manifest_ref": manifest["selector_manifest_ref"],
            "paired_kept_manifest_ref": manifest["paired_kept_manifest_ref"],
            "artifact_root": manifest["artifact_root"],
            "artifact_roots": manifest["artifact_roots"],
            "image_name": selector_binding["image_name"],
            "resolved_config_sha256": manifest["resolved_config_sha256"],
            "split_sha256": manifest["split_sha256"],
            "run_manifest_sha256": run_manifest_reference["sha256"],
            "dataset_stats_sha256": dataset_stats_reference["sha256"],
            "frozen_sidecars": frozen_sidecars,
        })
        freeze_manifest_path = staging / "freeze_manifest.json"
        atomic_write_json(freeze_manifest_path, report)
        freeze_manifest_path.chmod(0o444)

    atomic_directory(output, build_checkpoint_bundle)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
