#!/usr/bin/env python3
"""Train one exact78 Visual Aux future-2D checkpoint.

The ledger fixes one task and one RGB branch.  Labels are future image-space
endpoints only; Robot joint commands are neither loaded nor predicted.
"""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[3]))


import argparse
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import time
from typing import Any

import cv2
import jsonschema
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT))

from chaoyang.human_ego.tools.validate_visual_aux_bundle_v52 import validate_bundle  # noqa: E402
from chaoyang.human_ego.tools.build_visual_aux_session_bundle_v54 import (  # noqa: E402
    validate_silver_compositor_receipts,
)
from chaoyang.human_ego.tools.visual_aux_rc1_contract import (  # noqa: E402
    RELEASE_ID as RC1_RELEASE_ID,
    validate_rc1_eligibility,
)
from chaoyang.human_ego.training.VisualAuxFuture2DModel import (  # noqa: E402
    VisualAuxFuture2DModel,
    visual_aux_loss,
    visual_aux_metrics,
)
from chaoyang.human_ego.exact78_v31 import (  # noqa: E402
    FrozenDevelopmentTrainingConfig,
    development_capacity_report,
    future_2d_error_metrics,
    simple_future_2d_predictions,
    training_terminal_status,
)


LEDGER_SCHEMA = PROJECT / "contracts/visual_aux_dataset_ledger_v53.schema.json"
MODEL_SOURCE = PROJECT / "src/chaoyang/human_ego/training/VisualAuxFuture2DModel.py"
BUNDLE_BUILDER_SOURCE = PROJECT / "src/chaoyang/human_ego/tools/build_visual_aux_session_bundle_v54.py"
MIN_FORMAL_VALID_PIXEL_FRACTION = 0.70


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def artifact_ref(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def exact_ref(item: dict[str, Any]) -> Path:
    path = Path(str(item["path"]))
    if (
        not path.is_absolute()
        or not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != item["bytes"]
        or sha256(path) != item["sha256"]
    ):
        raise RuntimeError(f"artifact ref mismatch: {path}")
    return path


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    payload = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_torch_save(path: Path, value: dict[str, Any]) -> None:
    """Publish a checkpoint without exposing a partially written .pt file."""

    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(value, temporary)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def publish_checkpoint_files(
    output_root: Path,
    checkpoint: dict[str, Any],
    *,
    is_best: bool,
) -> dict[str, Path]:
    """Publish V7.1 canonical names and retain the v5.3 best-name alias."""

    last_checkpoint = output_root / "last.pt"
    atomic_torch_save(last_checkpoint, checkpoint)
    paths = {"last": last_checkpoint}
    if is_best:
        best_checkpoint = output_root / "best.pt"
        legacy_best_checkpoint = output_root / "BEST_VISUAL_AUX_CHECKPOINT.pt"
        atomic_torch_save(best_checkpoint, checkpoint)
        atomic_torch_save(legacy_best_checkpoint, checkpoint)
        paths.update(best=best_checkpoint, legacy_best=legacy_best_checkpoint)
    return paths


@lru_cache(maxsize=2)
def load_label_arrays(
    path_text: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path_text, allow_pickle=False) as data:
        future = np.asarray(data["future_2d_xy_normalized"], dtype=np.float32)
        valid = np.asarray(data["future_2d_valid"], dtype=bool)
        rgb_mask = np.asarray(data["rgb_training_valid_mask"], dtype=bool)
        # The previous label row's horizon-0 value is the endpoint at the
        # current frame.  It is available at time t and uses no t+1 suffix.
        endpoint = np.full((len(future), 2, 2), np.nan, dtype=np.float32)
        endpoint_valid = np.zeros((len(future), 2), dtype=bool)
        if len(future) > 1:
            endpoint[1:] = future[:-1, 0]
            endpoint_valid[1:] = valid[:-1, 0]
        return future, valid, rgb_mask, endpoint, endpoint_valid


class VisualAuxDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        manifests: list[Path],
        branch: str,
        image_size: tuple[int, int],
        allowed_starts: dict[str, set[int]] | None = None,
    ) -> None:
        self.branch = branch
        self.image_size = image_size
        self.records: list[dict[str, Any]] = []
        self.session_count = 0
        for manifest_path in manifests:
            root = manifest_path.parent
            report = validate_bundle(root)
            manifest = load_json(manifest_path)
            selector_ref = manifest["branches"][branch]["selector"]
            selector_path = Path(selector_ref["path"])
            if not selector_path.is_absolute():
                selector_path = root / selector_path
            selector = load_json(selector_path)
            label_path = Path(manifest["labels"]["path"])
            if not label_path.is_absolute():
                label_path = root / label_path
            frames = selector["frames"]
            for start in report["eligible_h50_starts"]:
                if allowed_starts is not None and start not in allowed_starts.get(
                    str(manifest["session_id"]), set()
                ):
                    continue
                rgb_path = Path(frames[start]["rgb"]["path"])
                if not rgb_path.is_absolute():
                    rgb_path = root / rgb_path
                self.records.append(
                    {
                        "rgb": rgb_path,
                        "labels": label_path,
                        "frame": int(start),
                        "width": int(manifest["image_width"]),
                        "height": int(manifest["image_height"]),
                    }
                )
            self.session_count += 1
        if not self.records:
            raise RuntimeError("Visual Aux dataset produced zero H50 windows")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.records[index]
        image = cv2.imread(str(row["rgb"]), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (row["height"], row["width"]):
            raise RuntimeError(f"RGB decode/dimension mismatch: {row['rgb']}")
        xy, valid, rgb_masks, endpoints, endpoint_valid = load_label_arrays(
            str(row["labels"])
        )
        frame = row["frame"]
        image[~rgb_masks[frame]] = 0
        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float() / 255.0
        return {
            "rgb": rgb,
            "xy": torch.from_numpy(np.ascontiguousarray(xy[frame])),
            "valid": torch.from_numpy(np.ascontiguousarray(valid[frame])),
            "current_xy": torch.from_numpy(np.ascontiguousarray(endpoints[frame])),
            "current_valid": torch.from_numpy(
                np.ascontiguousarray(endpoint_valid[frame])
            ),
            "previous_xy": torch.from_numpy(
                np.ascontiguousarray(endpoints[max(0, frame - 1)])
            ),
            "previous_valid": torch.from_numpy(
                np.ascontiguousarray(endpoint_valid[max(0, frame - 1)])
            ),
            "width": torch.tensor(row["width"], dtype=torch.float32),
            "height": torch.tensor(row["height"], dtype=torch.float32),
        }


def pair_dataset_signature(ledger: dict[str, Any]) -> str:
    payload = {
        "schema_version": "VISUAL_AUX_PAIR_DATASET_SIGNATURE_V1",
        "task": ledger.get("task"),
        "seed": ledger.get("seed"),
        "input_mode": ledger.get("input_mode"),
        "release_id": ledger.get("release_id"),
        "contract": ledger.get("contract"),
        "rc1_eligibility": ledger.get("rc1_eligibility"),
        "train": ledger.get("train"),
        "validation": ledger.get("validation"),
        "occlusion_hardset": ledger.get("occlusion_hardset"),
    }
    return canonical_sha256(payload)


def expected_bundle_producer_signature(payload: dict[str, Any]) -> dict[str, Any]:
    inputs = payload.get("inputs")
    gate = payload.get("valid_pixel_gate")
    if not isinstance(inputs, dict) or not isinstance(gate, dict):
        raise RuntimeError("bundle signature inputs/gate missing")
    required_inputs = (
        "robot_result",
        "clean_result",
        "occlusion_silver_result",
        "compositor_result",
    )
    if not all(isinstance(inputs.get(key), dict) for key in required_inputs):
        raise RuntimeError("bundle signature input receipts missing")
    signature_payload = {
        "schema_version": "VISUAL_AUX_BUNDLE_PRODUCER_SIGNATURE_V1",
        "builder_code": artifact_ref(BUNDLE_BUILDER_SOURCE),
        "robot_result": inputs["robot_result"],
        "clean_result": inputs["clean_result"],
        "occlusion_silver_result": inputs["occlusion_silver_result"],
        "compositor_result": inputs["compositor_result"],
        "split": payload.get("split"),
        "seed": payload.get("seed"),
        "h50_eligibility_mode": payload.get("h50_eligibility_mode"),
        "minimum_valid_pixel_fraction": gate.get("minimum_fraction"),
    }
    return {
        "payload": signature_payload,
        "sha256": canonical_sha256(signature_payload),
    }


def _validate_manifest_training_gate(
    manifest_path: Path, payload: dict[str, Any], report: dict[str, Any]
) -> None:
    gate = payload.get("valid_pixel_gate")
    if not isinstance(gate, dict):
        raise RuntimeError(f"valid-pixel gate missing: {manifest_path}")
    minimum = gate.get("minimum_fraction")
    if (
        not isinstance(minimum, (int, float))
        or isinstance(minimum, bool)
        or not MIN_FORMAL_VALID_PIXEL_FRACTION <= float(minimum) <= 1.0
        or gate.get("eligible_frames_meet_minimum") is not True
    ):
        raise RuntimeError(f"valid-pixel coverage gate failed: {manifest_path}")
    if float(report["rgb_valid_pixel_fraction"]) < float(minimum):
        raise RuntimeError(f"aggregate valid-pixel coverage failed: {manifest_path}")
    labels = payload.get("labels")
    if not isinstance(labels, dict):
        raise RuntimeError(f"labels receipt missing: {manifest_path}")
    label_path = exact_ref(
        {
            **labels,
            "path": str(
                Path(labels["path"])
                if Path(labels["path"]).is_absolute()
                else manifest_path.parent / labels["path"]
            ),
        }
    )
    with np.load(label_path, allow_pickle=False) as arrays:
        current = np.asarray(arrays["current_frame_valid"], dtype=bool)
        rgb_mask = np.asarray(arrays["rgb_training_valid_mask"], dtype=bool)
    if current.ndim != 1 or rgb_mask.ndim != 3 or len(current) != len(rgb_mask):
        raise RuntimeError(f"valid-pixel arrays have incompatible shapes: {manifest_path}")
    fractions = rgb_mask.reshape(len(rgb_mask), -1).mean(axis=1)
    claimed_fractions = np.asarray(gate.get("per_frame_fraction"), dtype=np.float64)
    if claimed_fractions.shape != fractions.shape or not np.allclose(
        claimed_fractions, fractions, rtol=0.0, atol=1e-12
    ):
        raise RuntimeError(f"valid-pixel per-frame receipt mismatch: {manifest_path}")
    if np.any(current & (fractions < float(minimum))):
        raise RuntimeError(f"current valid frame falls below valid-pixel threshold: {manifest_path}")
    inputs = payload.get("inputs")
    if not isinstance(inputs, dict) or not all(
        isinstance(inputs.get(key), dict)
        for key in (
            "robot_result",
            "clean_result",
            "occlusion_silver_result",
            "compositor_result",
        )
    ):
        raise RuntimeError(f"bundle input receipt binding missing: {manifest_path}")
    for key in (
        "robot_result",
        "clean_result",
        "occlusion_silver_result",
        "compositor_result",
    ):
        exact_ref(inputs[key])
    validate_silver_compositor_receipts(
        silver_path=Path(inputs["occlusion_silver_result"]["path"]),
        compositor_path=Path(inputs["compositor_result"]["path"]),
        task=str(payload.get("task")),
        session=str(payload.get("session_id")),
        frame_count=int(payload.get("frame_count", -1)),
        robot_ref=inputs["robot_result"],
        clean_ref=inputs["clean_result"],
    )
    if payload.get("occlusion_policy", {}).get("status") != "SILVER_BOUND_CAUSAL_COMPOSITOR_PASS":
        raise RuntimeError(f"formal Silver-bound compositor policy required: {manifest_path}")
    signature = payload.get("producer_signature")
    if not isinstance(signature, dict) or signature != expected_bundle_producer_signature(payload):
        raise RuntimeError(f"bundle producer signature mismatch: {manifest_path}")


def _validate_hardset(
    ledger: dict[str, Any],
    validation_reports: dict[str, dict[str, Any]],
) -> tuple[Path, dict[str, set[int]]]:
    hardset_item = ledger.get("occlusion_hardset")
    if not isinstance(hardset_item, dict):
        raise RuntimeError("frozen occlusion hardset receipt is required")
    hardset_path = exact_ref(hardset_item)
    hardset = load_json(hardset_path)
    if (
        hardset.get("schema_version") != "VISUAL_AUX_OCCLUSION_HARDSET_V1"
        or hardset.get("status") != "FROZEN_BEFORE_TRAINING"
        or hardset.get("task") != ledger.get("task")
        or hardset.get("split") != "validation"
    ):
        raise RuntimeError("frozen occlusion hardset identity/status mismatch")
    rows = hardset.get("sessions")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("frozen occlusion hardset must contain sessions")
    selection: dict[str, set[int]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError("invalid occlusion hardset row")
        session = row.get("session_id")
        starts = row.get("eligible_h50_starts")
        if not isinstance(session, str) or session in selection:
            raise RuntimeError("duplicate or invalid hardset session")
        if not isinstance(starts, list) or not starts or any(type(item) is not int for item in starts):
            raise RuntimeError("hardset H50 starts must be a non-empty integer list")
        if len(starts) != len(set(starts)):
            raise RuntimeError("duplicate hardset H50 start")
        report = validation_reports.get(session)
        if report is None or not set(starts).issubset(report["eligible_h50_starts"]):
            raise RuntimeError("hardset selection is outside the frozen validation windows")
        selection[session] = set(starts)
    return hardset_path, selection


def validate_ledger(path: Path) -> tuple[dict[str, Any], dict[str, list[Path]]]:
    ledger = load_json(path)
    schema = load_json(LEDGER_SCHEMA)
    jsonschema.Draft202012Validator(schema).validate(ledger)
    if ledger.get("input_mode") != "CAUSAL_TRAINING_INPUT":
        raise RuntimeError("formal Visual Aux ledger must be causal")
    expected_signature = pair_dataset_signature(ledger)
    if ledger.get("pair_dataset_signature") != expected_signature:
        raise RuntimeError("pair dataset signature mismatch")
    groups: dict[str, list[Path]] = {}
    reports_by_split: dict[str, dict[str, dict[str, Any]]] = {}
    global_sessions: dict[str, str] = {}
    global_source_groups: dict[str, str] = {}
    source_groups_by_split: dict[str, set[str]] = {}
    rc1_mode = ledger.get("release_id") == RC1_RELEASE_ID
    rc1_reports_by_manifest: dict[Path, dict[str, Any]] = {}
    if rc1_mode:
        eligibility_items = ledger.get("rc1_eligibility")
        if not isinstance(eligibility_items, list) or not eligibility_items:
            raise RuntimeError("RC1 ledger requires eligibility receipts")
        for item in eligibility_items:
            if not isinstance(item, dict):
                raise RuntimeError("invalid RC1 eligibility artifact reference")
            eligibility_path = exact_ref(item)
            eligibility_report = validate_rc1_eligibility(eligibility_path)
            manifest_path = Path(eligibility_report["artifacts"]["manifest"]).resolve()
            if manifest_path in rc1_reports_by_manifest:
                raise RuntimeError(f"duplicate RC1 eligibility for manifest: {manifest_path}")
            rc1_reports_by_manifest[manifest_path] = eligibility_report
    for split in ("train", "validation"):
        manifests = [exact_ref(item) for item in ledger[split]]
        seen = set()
        split_reports: dict[str, dict[str, Any]] = {}
        split_source_groups: set[str] = set()
        for manifest_path in manifests:
            payload = load_json(manifest_path)
            identity = payload.get("session_id")
            if payload.get("task") != ledger["task"] or payload.get("split") != split:
                raise RuntimeError(f"ledger task/split mismatch: {manifest_path}")
            causal = payload.get("causal_proof", {})
            if payload.get("input_mode") != "CAUSAL_TRAINING_INPUT" or causal.get("enabled") is not True:
                raise RuntimeError(f"formal Visual Aux training requires causal bundle: {manifest_path}")
            if causal.get("raw_robotized_valid_mask_shared") is not True:
                raise RuntimeError(f"Raw/Robotized causal valid mask is not shared: {manifest_path}")
            if identity in seen:
                raise RuntimeError(f"duplicate session in {split}: {identity}")
            if identity in global_sessions:
                raise RuntimeError(
                    f"session appears in both {global_sessions[identity]} and {split}: {identity}"
                )
            seen.add(identity)
            global_sessions[identity] = split
            report = validate_bundle(manifest_path.parent)
            if rc1_mode:
                rc1_report = rc1_reports_by_manifest.get(manifest_path.resolve())
                if rc1_report is None:
                    raise RuntimeError(f"RC1 eligibility receipt missing: {manifest_path}")
                source_group = str(rc1_report["source_group_id"])
                if source_group == "UNKNOWN_SOURCE_GROUP":
                    raise RuntimeError(f"unknown source group cannot enter RC1: {identity}")
                previous_split = global_source_groups.get(source_group)
                if previous_split is not None and previous_split != split:
                    raise RuntimeError(
                        f"source group crosses splits: {source_group}={previous_split}/{split}"
                    )
                global_source_groups[source_group] = split
                split_source_groups.add(source_group)
            else:
                _validate_manifest_training_gate(manifest_path, payload, report)
                source_group = str(
                    payload.get("source_group_id")
                    or payload.get("source_group")
                    or identity
                )
                previous_split = global_source_groups.get(source_group)
                if previous_split is not None and previous_split != split:
                    raise RuntimeError(
                        f"source group crosses splits: {source_group}={previous_split}/{split}"
                    )
                global_source_groups[source_group] = split
                split_source_groups.add(source_group)
            split_reports[str(identity)] = report
        groups[split] = manifests
        reports_by_split[split] = split_reports
        source_groups_by_split[split] = split_source_groups
    if rc1_mode:
        consumed_manifests = {
            path.resolve() for manifests in groups.values() for path in manifests
        }
        if set(rc1_reports_by_manifest) != consumed_manifests:
            raise RuntimeError("RC1 eligibility set differs from ledger manifests")
    minimum = {"train": (16, 256), "validation": (3, 48)}
    reports = {}
    for split, manifests in groups.items():
        window_counts = [
            int(reports_by_split[split][str(load_json(path).get("session_id"))]["eligible_h50_window_count"])
            for path in manifests
        ]
        zero_window_sessions = [
            load_json(path).get("session_id")
            for path, count in zip(manifests, window_counts, strict=True)
            if count == 0
        ]
        if zero_window_sessions:
            raise RuntimeError(
                f"{split} ledger contains zero-window sessions: {zero_window_sessions}"
            )
        windows = sum(window_counts)
        independent_groups = len(source_groups_by_split[split])
        reports[split] = {
            "sessions": len(manifests),
            "source_groups": independent_groups,
            "windows": windows,
        }
        if rc1_mode and (
            independent_groups < minimum[split][0] or windows < minimum[split][1]
        ):
            raise RuntimeError(f"{split} eligibility minimum failed: {reports[split]}")
    development_capacity = development_capacity_report(reports)
    if not rc1_mode and not development_capacity["training_start_allowed"]:
        raise RuntimeError("DEVELOPMENT ledger has no legal train/validation windows")
    hardset_path, hardset_selection = _validate_hardset(
        ledger, reports_by_split["validation"]
    )
    ledger["validated_counts"] = reports
    ledger["validated_occlusion_hardset"] = {
        "path": str(hardset_path),
        "sessions": len(hardset_selection),
        "windows": sum(len(items) for items in hardset_selection.values()),
        "selection": hardset_selection,
    }
    ledger["development_capacity"] = development_capacity
    ledger["rc1_capacity_gate_enforced"] = rc1_mode
    return ledger, groups


def validate_paired_ledgers(
    first_path: Path, second_path: Path
) -> dict[str, tuple[dict[str, Any], dict[str, list[Path]]]]:
    first = validate_ledger(first_path)
    second = validate_ledger(second_path)
    ledgers = {first[0].get("branch"): first, second[0].get("branch"): second}
    if set(ledgers) != {"HUMAN_RAW_RGB", "ROBOTIZED_RGB"}:
        raise RuntimeError("paired ledgers must contain Raw and Robotized branches exactly once")
    raw, robotized = ledgers["HUMAN_RAW_RGB"][0], ledgers["ROBOTIZED_RGB"][0]
    for key in (
        "task",
        "seed",
        "input_mode",
        "pair_dataset_signature",
        "train",
        "validation",
        "occlusion_hardset",
    ):
        if raw.get(key) != robotized.get(key):
            raise RuntimeError(f"Raw/Robotized paired-ledger mismatch: {key}")
    return ledgers


def training_run_signature(
    *,
    ledger_path: Path,
    paired_ledger_path: Path,
    epoch0: bool,
    epochs: int,
    batch_size: int,
    workers: int,
    learning_rate: float,
    validation_frequency: int,
    patience_validations: int,
    device: str,
    target_updates: int = 10_000,
    checkpoint_every_updates: int = 1_000,
    max_wall_seconds: int = 12 * 60 * 60,
    resume_checkpoint: Path | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": "VISUAL_AUX_TRAINING_RUN_SIGNATURE_V1",
        "trainer_code": artifact_ref(Path(__file__)),
        "model_code": artifact_ref(MODEL_SOURCE),
        "ledger": artifact_ref(ledger_path),
        "paired_ledger": artifact_ref(paired_ledger_path),
        "epoch0": epoch0,
        "epochs": epochs,
        "batch_size": batch_size,
        "workers": workers,
        "learning_rate": learning_rate,
        "validation_frequency": validation_frequency,
        "patience_validations": patience_validations,
        "device": device,
        "target_updates": target_updates,
        "checkpoint_every_updates": checkpoint_every_updates,
        "max_wall_seconds": max_wall_seconds,
        "resume_checkpoint": (
            artifact_ref(resume_checkpoint) if resume_checkpoint is not None else "ABSENT"
        ),
    }
    return {"payload": payload, "sha256": canonical_sha256(payload)}


def validate_training_result(
    result_path: Path,
    *,
    ledger_path: Path,
    paired_ledger_path: Path,
    epoch0: bool,
    epochs: int = 180,
    batch_size: int = 16,
    workers: int = 2,
    learning_rate: float = 3e-4,
    validation_frequency: int = 5,
    patience_validations: int = 12,
    device: str = "cuda",
    target_updates: int = 10_000,
    checkpoint_every_updates: int = 1_000,
    max_wall_seconds: int = 12 * 60 * 60,
    resume_checkpoint: Path | None = None,
) -> dict[str, Any]:
    """Verify an existing result before a retry/adopt path may treat it as passed."""

    result_path = result_path.resolve(strict=True)
    result = load_json(result_path)
    ledger_path = ledger_path.resolve(strict=True)
    paired_ledger_path = paired_ledger_path.resolve(strict=True)
    expected = training_run_signature(
        ledger_path=ledger_path,
        paired_ledger_path=paired_ledger_path,
        epoch0=epoch0,
        epochs=epochs,
        batch_size=batch_size,
        workers=workers,
        learning_rate=learning_rate,
        validation_frequency=validation_frequency,
        patience_validations=patience_validations,
        device=device,
        target_updates=target_updates,
        checkpoint_every_updates=checkpoint_every_updates,
        max_wall_seconds=max_wall_seconds,
        resume_checkpoint=resume_checkpoint,
    )
    required_schema = (
        "exact78-visual-aux-real-epoch0-v53-v1"
        if epoch0
        else "exact78-visual-aux-training-result-v53-v1"
    )
    required_status = "PASS_REAL_DATA_EPOCH0" if epoch0 else "PASSED_VISUAL_AUX_CHECKPOINT"
    if result.get("schema_version") != required_schema or result.get("status") != required_status:
        raise RuntimeError("existing Visual Aux result schema/status mismatch")
    if result.get("run_signature") != expected:
        raise RuntimeError("existing Visual Aux result run signature mismatch")
    ledger = load_json(ledger_path)
    if result.get("task") != ledger.get("task") or result.get("branch") != ledger.get("branch"):
        raise RuntimeError("existing Visual Aux result task/branch mismatch")
    if (
        result.get("control_ground_truth") is not False
        or result.get("physical_deployment_authorized") is not False
        or result.get("policy_checkpoint") is not False
        or result.get("pair_dataset_signature") != ledger.get("pair_dataset_signature")
    ):
        raise RuntimeError("existing Visual Aux result claim/pair boundary mismatch")
    for key, expected_path in (("ledger", ledger_path), ("paired_ledger", paired_ledger_path)):
        item = result.get(key)
        if not isinstance(item, dict) or exact_ref(item) != expected_path:
            raise RuntimeError(f"existing Visual Aux result {key} binding mismatch")
    hardset = result.get("occlusion_hardset")
    if not isinstance(hardset, dict) or hardset != ledger.get("occlusion_hardset"):
        raise RuntimeError("existing Visual Aux result frozen hardset binding mismatch")
    exact_ref(hardset)
    metric_sets = (
        [result.get("validation_metrics"), result.get("occlusion_hardset_metrics")]
        if epoch0
        else [result.get("best_validation_metrics"), result.get("best_occlusion_hardset_metrics")]
    )
    for metrics in metric_sets:
        if not isinstance(metrics, dict):
            raise RuntimeError("existing Visual Aux result metric set missing")
        for key in ("ADE_2D_px", "FDE_2D_px"):
            value = metrics.get(key)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not np.isfinite(value):
                raise RuntimeError(f"existing Visual Aux result metric invalid: {key}")
    if not epoch0:
        for key in ("best_checkpoint", "last_checkpoint", "history", "curve"):
            item = result.get(key)
            if not isinstance(item, dict):
                raise RuntimeError(f"existing Visual Aux output missing: {key}")
            exact_ref(item)
    return result


def make_loader(
    dataset: VisualAuxDataset,
    *,
    batch_size: int,
    shuffle: bool,
    workers: int,
    seed: int,
) -> DataLoader[dict[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def evaluate(
    model: VisualAuxFuture2DModel,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    predictions, targets, validities, widths, heights = [], [], [], [], []
    current_xy, current_valid, previous_xy, previous_valid = [], [], [], []
    losses = []
    with torch.no_grad():
        for batch in loader:
            rgb = batch["rgb"].to(device)
            xy = batch["xy"].to(device)
            valid = batch["valid"].to(device)
            prediction = model(rgb)
            losses.append(float(visual_aux_loss(prediction, xy, valid)["loss"]))
            predictions.append(prediction["xy_normalized"].cpu())
            targets.append(batch["xy"])
            validities.append(batch["valid"])
            widths.append(batch["width"])
            heights.append(batch["height"])
            current_xy.append(batch["current_xy"])
            current_valid.append(batch["current_valid"])
            previous_xy.append(batch["previous_xy"])
            previous_valid.append(batch["previous_valid"])
    metrics = visual_aux_metrics(
        torch.cat(predictions),
        torch.cat(targets),
        torch.cat(validities),
        torch.cat(widths),
        torch.cat(heights),
    )
    metrics["loss"] = float(np.mean(losses))
    targets_np = torch.cat(targets).numpy()
    valid_np = torch.cat(validities).numpy()
    current_np = torch.cat(current_xy).numpy()
    current_valid_np = torch.cat(current_valid).numpy()
    previous_np = torch.cat(previous_xy).numpy()
    previous_valid_np = torch.cat(previous_valid).numpy()
    baselines = simple_future_2d_predictions(current_np, previous_np, horizon=targets_np.shape[-3])
    width = int(round(float(torch.cat(widths)[0])))
    height = int(round(float(torch.cat(heights)[0])))
    metrics["simple_baselines"] = {
        "HOLD_POSITION": future_2d_error_metrics(
            baselines["HOLD_POSITION"], targets_np,
            valid_np & current_valid_np[:, None, :], width=width, height=height,
        ),
        "CONSTANT_VELOCITY": future_2d_error_metrics(
            baselines["CONSTANT_VELOCITY"], targets_np,
            valid_np & current_valid_np[:, None, :] & previous_valid_np[:, None, :],
            width=width, height=height,
        ),
    }
    return metrics


def plot_curves(history: list[dict[str, float]], path: Path) -> None:
    epochs = [item["epoch"] for item in history]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, [item["train_loss"] for item in history], label="train loss")
    axes[0].plot(epochs, [item["validation_loss"] for item in history], label="validation loss")
    axes[0].set_title("Visual Aux 损失")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].plot(epochs, [item["validation_ADE_2D_px"] for item in history], label="ADE px")
    axes[1].plot(epochs, [item["validation_FDE_2D_px"] for item in history], label="FDE px")
    axes[1].set_title("未来二维误差")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--paired-ledger", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epoch0", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--validation-frequency", type=int, default=5)
    parser.add_argument("--patience-validations", type=int, default=12)
    parser.add_argument("--target-updates", type=int, default=10_000)
    parser.add_argument("--checkpoint-every-updates", type=int, default=1_000)
    parser.add_argument("--max-wall-seconds", type=int, default=12 * 60 * 60)
    parser.add_argument("--resume-checkpoint", type=Path)
    args = parser.parse_args()
    ledger_path = args.ledger.resolve(strict=True)
    paired_ledger_path = args.paired_ledger.resolve(strict=True)
    paired = validate_paired_ledgers(ledger_path, paired_ledger_path)
    selected_branch = load_json(ledger_path).get("branch")
    if selected_branch not in paired:
        raise RuntimeError("--ledger is not one of the validated paired ledgers")
    ledger, groups = paired[selected_branch]
    resume_checkpoint = (
        args.resume_checkpoint.resolve(strict=True)
        if args.resume_checkpoint is not None
        else None
    )
    frozen_config = FrozenDevelopmentTrainingConfig(
        seed=int(ledger["seed"]),
        batch_size=args.batch_size,
        optimizer="AdamW",
        learning_rate=args.learning_rate,
        target_optimizer_updates=args.target_updates,
        maximum_epochs=args.epochs,
        checkpoint_every_updates=args.checkpoint_every_updates,
        per_model_gpu_cap_seconds=args.max_wall_seconds,
    )
    frozen_config.validate()
    run_signature = training_run_signature(
        ledger_path=ledger_path,
        paired_ledger_path=paired_ledger_path,
        epoch0=args.epoch0,
        epochs=args.epochs,
        batch_size=args.batch_size,
        workers=args.workers,
        learning_rate=args.learning_rate,
        validation_frequency=args.validation_frequency,
        patience_validations=args.patience_validations,
        device=args.device,
        target_updates=args.target_updates,
        checkpoint_every_updates=args.checkpoint_every_updates,
        max_wall_seconds=args.max_wall_seconds,
        resume_checkpoint=resume_checkpoint,
    )
    if args.output_root.exists():
        raise RuntimeError(f"fresh output root required: {args.output_root}")
    args.output_root.mkdir(parents=True)
    seed = int(ledger["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    image_size = (320, 240)
    train_dataset = VisualAuxDataset(groups["train"], ledger["branch"], image_size)
    validation_dataset = VisualAuxDataset(groups["validation"], ledger["branch"], image_size)
    hardset_dataset = VisualAuxDataset(
        groups["validation"],
        ledger["branch"],
        image_size,
        allowed_starts=ledger["validated_occlusion_hardset"]["selection"],
    )
    validation_loader = make_loader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        workers=args.workers,
        seed=seed,
    )
    hardset_loader = make_loader(
        hardset_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        workers=args.workers,
        seed=seed,
    )
    model = VisualAuxFuture2DModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    updates_per_epoch = int(math.ceil(len(train_dataset) / args.batch_size))
    target_updates = args.target_updates
    global_update = 0
    start_epoch = 1
    resume_batch = 0
    elapsed_before_resume = 0.0
    history: list[dict[str, float]] = []
    best_ade = float("inf")
    best_metrics: dict[str, float] | None = None
    best_hardset_metrics: dict[str, float] | None = None
    best_checkpoint = args.output_root / "best.pt"
    last_checkpoint = args.output_root / "last.pt"
    legacy_best_checkpoint = args.output_root / "BEST_VISUAL_AUX_CHECKPOINT.pt"
    common_checkpoint_paths: list[Path] = []

    if resume_checkpoint is not None:
        resumed = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        if (
            resumed.get("schema_version") != "exact78-visual-aux-checkpoint-v53-v1"
            or resumed.get("task") != ledger["task"]
            or resumed.get("branch") != ledger["branch"]
            or resumed.get("pair_dataset_signature") != ledger["pair_dataset_signature"]
            or resumed.get("ledger_sha256") != sha256(ledger_path)
            or resumed.get("paired_ledger_sha256") != sha256(paired_ledger_path)
        ):
            raise RuntimeError("resume checkpoint identity/ledger binding mismatch")
        model.load_state_dict(resumed["model"])
        optimizer.load_state_dict(resumed["optimizer"])
        global_update = int(resumed["global_update"])
        start_epoch = int(resumed["next_epoch"])
        resume_batch = int(resumed["next_batch_in_epoch"])
        elapsed_before_resume = float(resumed.get("elapsed_seconds", 0.0))
        best_ade = float(resumed.get("best_validation_ADE_2D_px", float("inf")))
        best_metrics = resumed.get("best_validation_metrics")
        best_hardset_metrics = resumed.get("best_occlusion_hardset_metrics")
        if "python_random_state" in resumed:
            random.setstate(resumed["python_random_state"])
        if "numpy_random_state" in resumed:
            np.random.set_state(resumed["numpy_random_state"])
        if "torch_random_state" in resumed:
            torch.set_rng_state(resumed["torch_random_state"])
        if torch.cuda.is_available() and resumed.get("torch_cuda_random_state_all") is not None:
            torch.cuda.set_rng_state_all(resumed["torch_cuda_random_state_all"])
        atomic_json(
            args.output_root / "RESUME_INPUT.json",
            {
                "schema_version": "EXACT78_VISUAL_AUX_RESUME_INPUT_V31",
                "checkpoint": artifact_ref(resume_checkpoint),
                "global_update": global_update,
                "next_epoch": start_epoch,
                "next_batch_in_epoch": resume_batch,
            },
        )
    else:
        # Epoch-0 is a true pre-update evaluation.  Paired branches therefore
        # start from the same seed/model state before either sees branch RGB.
        epoch0_metrics = evaluate(model, validation_loader, device)
        epoch0_hardset_metrics = evaluate(model, hardset_loader, device)
        epoch0_result = {
            "schema_version": "exact78-visual-aux-real-epoch0-v53-v1",
            "status": "PASS_REAL_DATA_EPOCH0",
            "task": ledger["task"],
            "branch": ledger["branch"],
            "dataset_counts": ledger["validated_counts"],
            "development_capacity": ledger["development_capacity"],
            "validation_metrics": epoch0_metrics,
            "occlusion_hardset_metrics": epoch0_hardset_metrics,
            "control_ground_truth": False,
            "physical_deployment_authorized": False,
            "policy_checkpoint": False,
            "ledger": artifact_ref(ledger_path),
            "paired_ledger": artifact_ref(paired_ledger_path),
            "occlusion_hardset": ledger["occlusion_hardset"],
            "pair_dataset_signature": ledger["pair_dataset_signature"],
            "frozen_training_config": frozen_config.as_receipt(),
            "run_signature": run_signature,
        }
        atomic_json(args.output_root / "EPOCH0_RESULT.json", epoch0_result)
        if args.epoch0:
            print(json.dumps(epoch0_result, ensure_ascii=False, indent=2))
            return 0

    wall_started = time.monotonic()
    final_status = "TRAINING_IN_PROGRESS"
    recent_losses: list[float] = []

    def elapsed_seconds() -> float:
        return elapsed_before_resume + (time.monotonic() - wall_started)

    def publish_evaluation(
        *, epoch: int, next_epoch: int, next_batch: int, terminal_status: str
    ) -> None:
        nonlocal best_ade, best_metrics, best_hardset_metrics, recent_losses
        metrics = evaluate(model, validation_loader, device)
        hardset_metrics = evaluate(model, hardset_loader, device)
        row = {
            "epoch": float(epoch),
            "global_update": float(global_update),
            "train_loss": float(np.mean(recent_losses)) if recent_losses else float("nan"),
            "validation_loss": metrics["loss"],
            "validation_ADE_2D_px": metrics["ADE_2D_px"],
            "validation_FDE_2D_px": metrics["FDE_2D_px"],
            "validation_PCK_20px": metrics["PCK_20px"],
            "left_right_identity_error": metrics["left_right_identity_error"],
            "temporal_smoothness_px": metrics["temporal_smoothness_px"],
            "occlusion_hardset_ADE_2D_px": hardset_metrics["ADE_2D_px"],
            "occlusion_hardset_FDE_2D_px": hardset_metrics["FDE_2D_px"],
            "terminal_status": terminal_status,
        }
        history.append(row)
        improved = metrics["ADE_2D_px"] < best_ade
        if improved:
            best_ade = metrics["ADE_2D_px"]
            best_metrics = dict(metrics)
            best_hardset_metrics = dict(hardset_metrics)
        checkpoint_payload = {
            "schema_version": "exact78-visual-aux-checkpoint-v53-v1",
            "task": ledger["task"],
            "branch": ledger["branch"],
            "epoch": epoch,
            "next_epoch": next_epoch,
            "next_batch_in_epoch": next_batch,
            "global_update": global_update,
            "updates_per_epoch": updates_per_epoch,
            "target_updates": target_updates,
            "checkpoint_every_updates": args.checkpoint_every_updates,
            "elapsed_seconds": elapsed_seconds(),
            "training_status": terminal_status,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "validation_metrics": metrics,
            "occlusion_hardset_metrics": hardset_metrics,
            "best_validation_ADE_2D_px": best_ade,
            "best_validation_metrics": best_metrics,
            "best_occlusion_hardset_metrics": best_hardset_metrics,
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
            "torch_cuda_random_state_all": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            "control_ground_truth": False,
            "physical_deployment_authorized": False,
            "policy_checkpoint": False,
            "label_source": "VISUAL_RETARGET_PROJECTION",
            "ledger_sha256": sha256(ledger_path),
            "paired_ledger_sha256": sha256(paired_ledger_path),
            "pair_dataset_signature": ledger["pair_dataset_signature"],
            "frozen_training_config": frozen_config.as_receipt(),
            "run_signature": run_signature,
        }
        publish_checkpoint_files(args.output_root, checkpoint_payload, is_best=improved)
        if global_update % args.checkpoint_every_updates == 0:
            common_path = args.output_root / f"checkpoint_update_{global_update:06d}.pt"
            atomic_torch_save(common_path, checkpoint_payload)
            common_checkpoint_paths.append(common_path)
        recent_losses = []

    stop = False
    for epoch in range(start_epoch, args.epochs + 1):
        train_loader = make_loader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            workers=args.workers,
            seed=seed + epoch,
        )
        model.train()
        for batch_index, batch in enumerate(train_loader):
            if epoch == start_epoch and batch_index < resume_batch:
                continue
            prediction = model(batch["rgb"].to(device, non_blocking=True))
            loss_values = visual_aux_loss(
                prediction,
                batch["xy"].to(device, non_blocking=True),
                batch["valid"].to(device, non_blocking=True),
            )
            optimizer.zero_grad(set_to_none=True)
            loss_values["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            global_update += 1
            recent_losses.append(float(loss_values["loss"].detach()))
            next_epoch = epoch
            next_batch = batch_index + 1
            if next_batch >= updates_per_epoch:
                next_epoch, next_batch = epoch + 1, 0
            final_status = training_terminal_status(
                global_update=global_update,
                completed_epochs=epoch if next_batch == 0 else epoch - 1,
                elapsed_seconds=elapsed_seconds(),
                config=frozen_config,
            )
            checkpoint_due = global_update % args.checkpoint_every_updates == 0
            terminal = final_status != "TRAINING_IN_PROGRESS"
            if checkpoint_due or terminal:
                publish_evaluation(
                    epoch=epoch,
                    next_epoch=next_epoch,
                    next_batch=next_batch,
                    terminal_status=final_status,
                )
                model.train()
            if terminal:
                stop = True
                break
        resume_batch = 0
        if stop:
            break
        if epoch == args.epochs:
            final_status = training_terminal_status(
                global_update=global_update,
                completed_epochs=epoch,
                elapsed_seconds=elapsed_seconds(),
                config=frozen_config,
            )
            if not history or int(history[-1]["global_update"]) != global_update:
                publish_evaluation(
                    epoch=epoch,
                    next_epoch=epoch + 1,
                    next_batch=0,
                    terminal_status=final_status,
                )

    history_path = args.output_root / "TRAINING_HISTORY.json"
    atomic_json(history_path, {"history": history})
    curve = args.output_root / "LOSS_AND_METRICS_CURVE.png"
    plot_curves(history, curve)
    result = {
        "schema_version": "exact78-visual-aux-training-result-v53-v1",
        "status": (
            "PASSED_VISUAL_AUX_CHECKPOINT"
            if final_status.startswith("TRAINING_COMPLETE_")
            else "TRAINING_PAUSED_BUDGET"
        ),
        "training_terminal": final_status,
        "task": ledger["task"],
        "branch": ledger["branch"],
        "checkpoint": artifact_ref(legacy_best_checkpoint),
        "best_checkpoint": artifact_ref(best_checkpoint),
        "last_checkpoint": artifact_ref(last_checkpoint),
        "legacy_best_checkpoint": artifact_ref(legacy_best_checkpoint),
        "history": artifact_ref(history_path),
        "curve": artifact_ref(curve),
        "best_validation_ADE_2D_px": best_ade,
        "best_validation_metrics": best_metrics,
        "best_occlusion_hardset_metrics": best_hardset_metrics,
        "valid_window_coverage": ledger["validated_counts"],
        "development_capacity": ledger["development_capacity"],
        "occlusion_hardset": ledger["occlusion_hardset"],
        "ledger": artifact_ref(ledger_path),
        "paired_ledger": artifact_ref(paired_ledger_path),
        "pair_dataset_signature": ledger["pair_dataset_signature"],
        "run_signature": run_signature,
        "fixed_update_budget": {
            "enabled": True,
            "updates_per_epoch": updates_per_epoch,
            "target_updates": target_updates,
            "actual_updates": global_update,
            "maximum_epochs": args.epochs,
            "checkpoint_every_updates": args.checkpoint_every_updates,
            "common_checkpoints": [artifact_ref(path) for path in common_checkpoint_paths],
            "elapsed_seconds": elapsed_seconds(),
            "per_model_gpu_cap_seconds": args.max_wall_seconds,
            "early_stop_enabled": False,
            "scheduler": "ABSENT",
            "amp": "ABSENT",
        },
        "frozen_training_config": frozen_config.as_receipt(),
        "control_ground_truth": False,
        "physical_deployment_authorized": False,
        "policy_checkpoint": False,
        "claim_limit": "Single-seed Visual Aux engineering checkpoint; not Robot action supervision or a deployable policy.",
    }
    atomic_json(args.output_root / "RESULT.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
