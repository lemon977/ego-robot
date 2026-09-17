#!/usr/bin/env python3
"""Run a real-bundle CPU forward/backward smoke test for both RGB branches."""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[3]))


import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import torch
from torch.utils.data import DataLoader

PROJECT = Path(__file__).resolve().parents[4]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from chaoyang.human_ego.tools.train_visual_aux_future2d_v53 import VisualAuxDataset  # noqa: E402
from chaoyang.human_ego.tools.validate_visual_aux_bundle_v52 import validate_bundle  # noqa: E402
from chaoyang.human_ego.training.VisualAuxFuture2DModel import (  # noqa: E402
    VisualAuxFuture2DModel,
    visual_aux_loss,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, object]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = args.manifest.resolve(strict=True)
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"immutable output exists: {output}")
    report = validate_bundle(manifest.parent)
    if int(report["eligible_h50_window_count"]) <= 0:
        raise RuntimeError("real-bundle smoke test requires at least one H50 window")
    torch.manual_seed(7)
    rows: list[dict[str, object]] = []
    for branch in ("HUMAN_RAW_RGB", "ROBOTIZED_RGB"):
        dataset = VisualAuxDataset([manifest], branch, (96, 72))
        batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)))
        model = VisualAuxFuture2DModel(feature_dim=32)
        prediction = model(batch["rgb"])
        losses = visual_aux_loss(prediction, batch["xy"], batch["valid"])
        losses["loss"].backward()
        gradient_finite = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        )
        rows.append(
            {
                "branch": branch,
                "dataset_windows": len(dataset),
                "loss": float(losses["loss"].detach()),
                "coordinate_loss": float(losses["coordinate_loss"].detach()),
                "validity_loss": float(losses["validity_loss"].detach()),
                "gradient_finite": gradient_finite,
            }
        )
    atomic_json(
        output,
        {
            "schema_version": "exact78-visual-aux-real-bundle-smoke-v55-v1",
            "status": "PASS_REAL_BUNDLE_FORWARD_BACKWARD_SMOKE",
            "manifest": ref(manifest),
            "eligible_h50_windows": int(report["eligible_h50_window_count"]),
            "branches": rows,
            "checkpoint_published": False,
            "formal_epoch0": False,
            "control_ground_truth": False,
            "claim_limit": "Real-data CPU smoke test only; heldout data was not moved into training and no checkpoint was published.",
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
