from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chaoyang.human_ego.training.VisualAuxFuture2DModel import (  # noqa: E402
    VisualAuxFuture2DModel,
    visual_aux_loss,
    visual_aux_metrics,
)
from chaoyang.human_ego.tools import train_visual_aux_future2d_v53 as trainer  # noqa: E402


def test_visual_aux_model_has_only_future_2d_outputs() -> None:
    model = VisualAuxFuture2DModel(feature_dim=32)
    output = model(torch.zeros(2, 3, 32, 48))
    assert set(output) == {"xy_normalized", "valid_logits"}
    assert output["xy_normalized"].shape == (2, 50, 2, 2)
    assert output["valid_logits"].shape == (2, 50, 2)
    assert torch.all((output["xy_normalized"] >= 0) & (output["xy_normalized"] <= 1))


def test_visual_aux_loss_ignores_nan_invalid_coordinates() -> None:
    model = VisualAuxFuture2DModel(feature_dim=32)
    output = model(torch.zeros(1, 3, 32, 48))
    target = torch.full((1, 50, 2, 2), float("nan"))
    valid = torch.zeros(1, 50, 2, dtype=torch.bool)
    target[:, :4] = 0.5
    valid[:, :4] = True
    losses = visual_aux_loss(output, target, valid)
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()


def test_visual_aux_metrics_report_pixel_space_and_identity() -> None:
    target = torch.zeros(1, 50, 2, 2)
    target[:, :, 1, 0] = 1.0
    predicted = target.clone()
    valid = torch.ones(1, 50, 2, dtype=torch.bool)
    metrics = visual_aux_metrics(
        predicted,
        target,
        valid,
        torch.tensor([101.0]),
        torch.tensor([51.0]),
    )
    assert metrics["ADE_2D_px"] == 0.0
    assert metrics["FDE_2D_px"] == 0.0
    assert metrics["PCK_20px"] == 1.0
    assert metrics["left_right_identity_error"] == 0.0
    assert metrics["temporal_smoothness_px"] == 0.0


def test_ledger_rejects_a_session_with_zero_h50_windows(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifests = [tmp_path / f"train_{index}.json" for index in range(16)]
    validations = [tmp_path / f"validation_{index}.json" for index in range(3)]
    for path in manifests + validations:
        path.write_text("{}\n", encoding="utf-8")
    ledger_path = tmp_path / "ledger.json"
    reference = lambda path: {  # noqa: E731
        "path": str(path.resolve()),
        "bytes": 1,
        "sha256": "0" * 64,
    }
    ledger = {
        "schema_version": "exact78-visual-aux-dataset-ledger-v53-v1",
        "status": "PASS_VISUAL_AUX_DATASET_LEDGER",
        "task": "chips",
        "branch": "HUMAN_RAW_RGB",
        "seed": 7,
        "control_ground_truth": False,
        "input_mode": "CAUSAL_TRAINING_INPUT",
        "occlusion_hardset": reference(tmp_path / "hardset.json"),
        "train": [reference(path) for path in manifests],
        "validation": [reference(path) for path in validations],
    }
    ledger["pair_dataset_signature"] = trainer.pair_dataset_signature(ledger)
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    real_load = trainer.load_json

    def fake_load(path):
        if path == ledger_path:
            return ledger
        if path == trainer.LEDGER_SCHEMA:
            return real_load(path)
        split = "validation" if "validation" in path.name else "train"
        return {
            "task": "chips",
            "split": split,
            "session_id": path.stem,
            "input_mode": "CAUSAL_TRAINING_INPUT",
            "causal_proof": {"enabled": True, "raw_robotized_valid_mask_shared": True},
        }

    monkeypatch.setattr(trainer, "load_json", fake_load)
    monkeypatch.setattr(trainer, "exact_ref", lambda item: __import__("pathlib").Path(item["path"]))
    monkeypatch.setattr(trainer, "_validate_manifest_training_gate", lambda *args: None)
    calls = iter([0] + [50] * 18)
    monkeypatch.setattr(
        trainer,
        "validate_bundle",
        lambda root: {"eligible_h50_window_count": next(calls)},
    )
    with pytest.raises(RuntimeError, match="zero-window sessions"):
        trainer.validate_ledger(ledger_path)


def test_ledger_rejects_noncausal_bundle(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = tmp_path / "train.json"
    manifest.write_text("{}\n", encoding="utf-8")
    reference = {"path": str(manifest.resolve()), "bytes": 1, "sha256": "0" * 64}
    ledger = {
        "schema_version": "exact78-visual-aux-dataset-ledger-v53-v1",
        "status": "PASS_VISUAL_AUX_DATASET_LEDGER",
        "task": "chips", "branch": "HUMAN_RAW_RGB", "seed": 7,
        "control_ground_truth": False, "input_mode": "CAUSAL_TRAINING_INPUT",
        "occlusion_hardset": reference, "train": [reference], "validation": [reference],
    }
    ledger["pair_dataset_signature"] = trainer.pair_dataset_signature(ledger)
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    real_load = trainer.load_json

    def fake_load(path):
        if path == ledger_path:
            return ledger
        if path == trainer.LEDGER_SCHEMA:
            return real_load(path)
        return {"task": "chips", "split": "train", "session_id": "s"}

    monkeypatch.setattr(trainer, "load_json", fake_load)
    monkeypatch.setattr(trainer, "exact_ref", lambda item: __import__("pathlib").Path(item["path"]))
    with pytest.raises(RuntimeError, match="requires causal bundle"):
        trainer.validate_ledger(ledger_path)


def test_ledger_rejects_session_overlap_between_train_and_validation(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "shared.json"
    manifest.write_text("{}\n", encoding="utf-8")
    item = {"path": str(manifest.resolve()), "bytes": 3, "sha256": "0" * 64}
    ledger = {
        "schema_version": "exact78-visual-aux-dataset-ledger-v53-v1",
        "status": "PASS_VISUAL_AUX_DATASET_LEDGER",
        "task": "chips",
        "branch": "HUMAN_RAW_RGB",
        "seed": 7,
        "control_ground_truth": False,
        "input_mode": "CAUSAL_TRAINING_INPUT",
        "occlusion_hardset": item,
        "train": [item],
        "validation": [item],
    }
    ledger["pair_dataset_signature"] = trainer.pair_dataset_signature(ledger)
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    real_load = trainer.load_json

    def fake_load(path):
        if path == ledger_path:
            return ledger
        if path == trainer.LEDGER_SCHEMA:
            return real_load(path)
        return {
            "task": "chips",
            "split": "train" if not hasattr(fake_load, "seen") else "validation",
            "session_id": "same_session",
            "input_mode": "CAUSAL_TRAINING_INPUT",
            "causal_proof": {"enabled": True, "raw_robotized_valid_mask_shared": True},
        }

    # The manifest is loaded once per split.  Preserve its identity while
    # returning the split expected by that ledger branch.
    calls = iter(("train", "validation"))
    def split_load(path):
        if path == ledger_path:
            return ledger
        if path == trainer.LEDGER_SCHEMA:
            return real_load(path)
        return {
            "task": "chips", "split": next(calls), "session_id": "same_session",
            "input_mode": "CAUSAL_TRAINING_INPUT",
            "causal_proof": {"enabled": True, "raw_robotized_valid_mask_shared": True},
        }

    monkeypatch.setattr(trainer, "load_json", split_load)
    monkeypatch.setattr(trainer, "exact_ref", lambda value: Path(value["path"]))
    monkeypatch.setattr(trainer, "validate_bundle", lambda root: {"eligible_h50_window_count": 50})
    monkeypatch.setattr(trainer, "_validate_manifest_training_gate", lambda *args: None)
    with pytest.raises(RuntimeError, match="both train and validation"):
        trainer.validate_ledger(ledger_path)


def test_paired_ledgers_reject_different_manifest_sha(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = {
        "branch": "HUMAN_RAW_RGB", "task": "chips", "seed": 7,
        "input_mode": "CAUSAL_TRAINING_INPUT", "pair_dataset_signature": "a" * 64,
        "train": [{"path": "/x", "bytes": 1, "sha256": "1" * 64}],
        "validation": [{"path": "/v", "bytes": 1, "sha256": "2" * 64}],
        "occlusion_hardset": {"path": "/h", "bytes": 1, "sha256": "3" * 64},
    }
    robot = {**raw, "branch": "ROBOTIZED_RGB", "train": [{**raw["train"][0], "sha256": "4" * 64}]}
    values = iter(((raw, {}), (robot, {})))
    monkeypatch.setattr(trainer, "validate_ledger", lambda path: next(values))
    with pytest.raises(RuntimeError, match="paired-ledger mismatch: train"):
        trainer.validate_paired_ledgers(Path("raw"), Path("robot"))


def test_valid_pixel_gate_rejects_one_pixel_style_coverage(tmp_path: Path) -> None:
    payload = {
        "valid_pixel_gate": {"minimum_fraction": 0.70, "eligible_frames_meet_minimum": True}
    }
    with pytest.raises(RuntimeError, match="aggregate valid-pixel coverage"):
        trainer._validate_manifest_training_gate(
            tmp_path / "manifest.json", payload, {"rgb_valid_pixel_fraction": 0.01}
        )


def test_manifest_training_gate_recomputes_bundle_signature_and_pixel_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(trainer, "validate_silver_compositor_receipts", lambda **kwargs: {})
    inputs = {}
    for name in ("robot_result", "clean_result", "occlusion_silver_result", "compositor_result"):
        path = tmp_path / f"{name}.json"
        path.write_text("{}\n", encoding="utf-8")
        inputs[name] = trainer.artifact_ref(path)
    labels = tmp_path / "LABELS.npz"
    np.savez_compressed(
        labels,
        current_frame_valid=np.asarray([True, False], dtype=bool),
        rgb_training_valid_mask=np.asarray(
            [
                np.ones((1, 10), dtype=bool),
                np.asarray([[True] * 4 + [False] * 6], dtype=bool),
            ]
        ),
    )
    payload = {
        "split": "train",
        "seed": 7,
        "h50_eligibility_mode": "BOTH_ENDPOINTS_40_OF_50",
        "labels": trainer.artifact_ref(labels),
        "inputs": inputs,
        "valid_pixel_gate": {
            "minimum_fraction": 0.70,
            "per_frame_fraction": [1.0, 0.4],
            "eligible_frames_meet_minimum": True,
        },
        "occlusion_policy": {"status": "SILVER_BOUND_CAUSAL_COMPOSITOR_PASS"},
    }
    payload["producer_signature"] = trainer.expected_bundle_producer_signature(payload)
    trainer._validate_manifest_training_gate(
        tmp_path / "VISUAL_AUX_SESSION_MANIFEST.json",
        payload,
        {"rgb_valid_pixel_fraction": 0.7},
    )

    payload["producer_signature"]["payload"]["seed"] = 8
    with pytest.raises(RuntimeError, match="producer signature mismatch"):
        trainer._validate_manifest_training_gate(
            tmp_path / "VISUAL_AUX_SESSION_MANIFEST.json",
            payload,
            {"rgb_valid_pixel_fraction": 0.7},
        )


def test_manifest_training_gate_rejects_false_per_frame_coverage_receipt(
    tmp_path: Path,
) -> None:
    labels = tmp_path / "LABELS.npz"
    np.savez_compressed(
        labels,
        current_frame_valid=np.asarray([True], dtype=bool),
        rgb_training_valid_mask=np.ones((1, 2, 2), dtype=bool),
    )
    payload = {
        "valid_pixel_gate": {
            "minimum_fraction": 0.70,
            "per_frame_fraction": [0.70],
            "eligible_frames_meet_minimum": True,
        },
        "labels": trainer.artifact_ref(labels),
    }
    with pytest.raises(RuntimeError, match="per-frame receipt mismatch"):
        trainer._validate_manifest_training_gate(
            tmp_path / "VISUAL_AUX_SESSION_MANIFEST.json",
            payload,
            {"rgb_valid_pixel_fraction": 1.0},
        )


def test_existing_training_result_requires_exact_run_signature(tmp_path: Path) -> None:
    ledger = tmp_path / "raw.json"
    paired = tmp_path / "robot.json"
    ledger.write_text(json.dumps({"task": "chips", "branch": "HUMAN_RAW_RGB"}))
    paired.write_text("{}")
    result = tmp_path / "EPOCH0_RESULT.json"
    result.write_text(json.dumps({
        "schema_version": "exact78-visual-aux-real-epoch0-v53-v1",
        "status": "PASS_REAL_DATA_EPOCH0",
        "run_signature": {"payload": {}, "sha256": "0" * 64},
    }))
    with pytest.raises(RuntimeError, match="run signature mismatch"):
        trainer.validate_training_result(
            result,
            ledger_path=ledger,
            paired_ledger_path=paired,
            epoch0=True,
        )


def test_existing_training_result_rejects_tampered_output_sha(tmp_path: Path) -> None:
    hardset = tmp_path / "hardset.json"
    hardset.write_text("{}\n", encoding="utf-8")
    ledger = tmp_path / "raw.json"
    ledger.write_text(
        json.dumps(
            {
                "task": "chips",
                "branch": "HUMAN_RAW_RGB",
                "occlusion_hardset": trainer.artifact_ref(hardset),
                "pair_dataset_signature": "c" * 64,
            }
        ),
        encoding="utf-8",
    )
    paired = tmp_path / "robot.json"
    paired.write_text("{}\n", encoding="utf-8")
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"before")
    artifact_receipt = trainer.artifact_ref(artifact)
    run_signature = trainer.training_run_signature(
        ledger_path=ledger,
        paired_ledger_path=paired,
        epoch0=False,
        epochs=180,
        batch_size=16,
        workers=2,
        learning_rate=3e-4,
        validation_frequency=5,
        patience_validations=12,
        device="cuda",
    )
    result = tmp_path / "RESULT.json"
    result.write_text(
        json.dumps(
            {
                "schema_version": "exact78-visual-aux-training-result-v53-v1",
                "status": "PASSED_VISUAL_AUX_CHECKPOINT",
                "task": "chips",
                "branch": "HUMAN_RAW_RGB",
                "run_signature": run_signature,
                "ledger": trainer.artifact_ref(ledger),
                "paired_ledger": trainer.artifact_ref(paired),
                "occlusion_hardset": trainer.artifact_ref(hardset),
                "pair_dataset_signature": "c" * 64,
                "best_validation_metrics": {"ADE_2D_px": 1.0, "FDE_2D_px": 1.0},
                "best_occlusion_hardset_metrics": {"ADE_2D_px": 1.0, "FDE_2D_px": 1.0},
                "best_checkpoint": artifact_receipt,
                "last_checkpoint": artifact_receipt,
                "history": artifact_receipt,
                "curve": artifact_receipt,
                "control_ground_truth": False,
                "physical_deployment_authorized": False,
                "policy_checkpoint": False,
            }
        ),
        encoding="utf-8",
    )
    artifact.write_bytes(b"after")
    with pytest.raises(RuntimeError, match="artifact ref mismatch"):
        trainer.validate_training_result(
            result,
            ledger_path=ledger,
            paired_ledger_path=paired,
            epoch0=False,
        )


def test_checkpoint_publication_has_v71_best_last_and_legacy_alias(tmp_path: Path) -> None:
    first = {
        "schema_version": "exact78-visual-aux-checkpoint-v53-v1",
        "epoch": 5,
        "model": {"weight": torch.tensor([1.0])},
        "validation_metrics": {"ADE_2D_px": 12.0},
        "control_ground_truth": False,
    }
    paths = trainer.publish_checkpoint_files(tmp_path, first, is_best=True)
    assert paths == {
        "best": tmp_path / "best.pt",
        "last": tmp_path / "last.pt",
        "legacy_best": tmp_path / "BEST_VISUAL_AUX_CHECKPOINT.pt",
    }
    assert torch.load(paths["best"], weights_only=False)["epoch"] == 5
    assert torch.load(paths["best"], weights_only=False)["control_ground_truth"] is False
    assert torch.load(paths["last"], weights_only=False)["epoch"] == 5
    assert torch.load(paths["legacy_best"], weights_only=False)["epoch"] == 5

    later_not_best = {**first, "epoch": 10, "validation_metrics": {"ADE_2D_px": 14.0}}
    trainer.publish_checkpoint_files(tmp_path, later_not_best, is_best=False)
    assert torch.load(paths["best"], weights_only=False)["epoch"] == 5
    assert torch.load(paths["legacy_best"], weights_only=False)["epoch"] == 5
    assert torch.load(paths["last"], weights_only=False)["epoch"] == 10
