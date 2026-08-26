#!/usr/bin/env python3
"""Validate and immutably snapshot a completed validation-selected best.pt."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from preprocess.retarget_labels.schema import EMBODIMENTS  # noqa: E402


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def validation_score(snapshot: dict) -> float:
    return float(
        snapshot["pos_err_w_m"]
        + snapshot["rot_err_w_deg"] / 100.0
        + snapshot.get("joint_mae_w_normalized", 0.0) * 0.10
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embodiment", required=True, choices=sorted(EMBODIMENTS))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument(
        "--split", type=Path, default=ROOT / "data_manifests/shared_split.json",
        help="Exact frozen split used by the training run",
    )
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "artifacts/frozen_checkpoints"
    )
    args = parser.parse_args()
    run_dir = args.run_dir or ROOT / "runs/grap_a_cap" / f"dual_{args.embodiment}"
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
    for required in (best_path, latest_path, split_path, run_dir / "dataset_stats.json"):
        if not required.is_file():
            raise FileNotFoundError(required)
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    latest = torch.load(latest_path, map_location="cpu", weights_only=False)
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
    if manifest.get("split_sha256") != sha256(split_path):
        raise ValueError("checkpoint/shared split hash mismatch")
    disk_manifest = json.loads((run_dir / "run_manifest.json").read_text())
    if disk_manifest != manifest:
        raise ValueError("embedded and on-disk run manifests differ")

    snapshot_path = run_dir / "eval_snapshots" / f"eval_ep_{best['epoch']:04d}.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    recomputed_score = validation_score(snapshot)
    selected_score = float(best["best"]["score"])
    if abs(recomputed_score - selected_score) > 1e-9:
        raise ValueError(
            f"validation snapshot score mismatch: {recomputed_score} != {selected_score}"
        )

    output = args.output_root / args.embodiment
    output.mkdir(parents=True, exist_ok=True)
    frozen_path = output / "best.pt"
    temporary = output / "best.pt.tmp"
    shutil.copyfile(best_path, temporary)
    if sha256(temporary) != sha256(best_path):
        raise RuntimeError("checkpoint copy hash mismatch")
    os.replace(temporary, frozen_path)
    frozen_path.chmod(0o444)
    frozen_sidecars = {}
    for name in ("dataset_stats.json", "run_manifest.json", "config.json"):
        source = run_dir / name
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = output / name
        sidecar_temporary = output / f"{name}.tmp"
        shutil.copyfile(source, sidecar_temporary)
        if sha256(source) != sha256(sidecar_temporary):
            raise RuntimeError(f"frozen sidecar copy hash mismatch: {name}")
        os.replace(sidecar_temporary, destination)
        destination.chmod(0o444)
        frozen_sidecars[name] = sha256(destination)
    report = {
        "status": "frozen",
        "embodiment": args.embodiment,
        "source": str(best_path.resolve()),
        "frozen_checkpoint": str(frozen_path.resolve()),
        "checkpoint_sha256": sha256(frozen_path),
        "epoch": int(best["epoch"]),
        "latest_epoch_at_freeze": int(latest["epoch"]),
        "validation_score": selected_score,
        "validation_snapshot": str(snapshot_path.resolve()),
        "validation_snapshot_sha256": sha256(snapshot_path),
        "model_weights": best["model_weights"],
        "selection_metric": best["selection_metric"],
        "split_sha256": manifest["split_sha256"],
        "run_manifest_sha256": sha256(run_dir / "run_manifest.json"),
        "dataset_stats_sha256": sha256(run_dir / "dataset_stats.json"),
        "frozen_sidecars": frozen_sidecars,
    }
    report_path = output / "freeze_manifest.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
