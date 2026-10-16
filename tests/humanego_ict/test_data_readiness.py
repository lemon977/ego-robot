"""CPU fixtures model schema validation, never real-world qualification.

Run with --basetemp inside the task attempt directory. Production readers reject
every fixture here unless explicitly given allow_synthetic_fixture=True.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from chaoyang.humanego_ict.data import (
    ACTION_LAYOUT, SCHEMA, TARGET, Revo2ICTDataset, TrainingDataError,
    validate_training_manifest,
)
from chaoyang.humanego_ict.readiness import audit_nas_roots


FIXTURE_POSE_CLOCK = {
    "position_unit": "meters", "pose_frame_id": "SYNTHETIC_WORLD_FRAME",
    "camera_frame_id": "SYNTHETIC_REFERENCE_CAMERA_FRAME",
    "tcp_frame_ids": ["SYNTHETIC_LEFT_TCP", "SYNTHETIC_RIGHT_TCP"],
    "timestamp_unit": "nanoseconds", "clock_domain": "SYNTHETIC_SYNCHRONIZED_CLOCK",
}


def _ref(path: Path):
    content = path.read_bytes()
    return {"path": str(path), "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest()}


def _save_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return _ref(path)


def _save_npz(path, **arrays):
    # Small fixture artifacts only; no torch/model construction.
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    path.write_bytes(buffer.getvalue())
    return _ref(path)


@pytest.fixture
def paired_manifest(tmp_path):
    evidence = _save_json(tmp_path / "fixture_evidence.json", {"fixture_only": True})
    manifest = {
        "schema": SCHEMA, "pair_id": "synthetic_cpu_fixture_only", "target": TARGET,
        "fixture_only": True, "control_ground_truth": True,
        "physical_deployment_authorized": False, "action_layout": ACTION_LAYOUT,
        "ict_schema": "HAND_ONLY_29D_GRASP_UNKNOWN_MINUS_ONE",
        "normalization": {"method": "IDENTITY_NO_FIT", "position_mean": [0, 0, 0], "position_std": [1, 1, 1]},
        "calibration_receipt": _save_json(tmp_path / "camera_receipt.json", {
            "schema": "robot-camera-calibration-v1", "status": "PASSED",
            "geometry_qualified": True, "physical_eye_identity_verified": True,
            **FIXTURE_POSE_CLOCK,
            "evidence_refs": [evidence], "fixture_only": True,
        }), "samples": [],
    }
    for split, session in (("train", "fixture_train"), ("validation", "fixture_validation")):
        qpc = np.arange(51, dtype=np.int64) * 40_000_000 + 1_000_000_000
        wrist = np.broadcast_to(np.eye(4), (51, 2, 4, 4)).copy()
        wrist[:, 0, 0, 3] = np.arange(51) / 100
        wrist[:, 1, 0, 3] = np.arange(51) / 100 + 0.3
        hand = np.arange(51 * 2 * 6, dtype=np.float64).reshape(51, 2, 6) / 1000
        sidecar = _save_npz(tmp_path / f"{session}_real_sidecar.npz",
            wrist_pose_world=wrist, hand_command=hand,
            valid=np.ones((51, 2), dtype=bool), qpc_ns=qpc,
            source_frame_id=np.arange(51, dtype=np.int64),
            segment_id=np.zeros(51, dtype=np.int64))
        action = np.concatenate((wrist[1:, :, :3, 3].reshape(50, 6),
            wrist[1:, :, :3, :2].reshape(50, 12), hand[1:].reshape(50, 12)), axis=1)
        ict = np.zeros((8, 29), dtype=np.float64)
        for index in range(2):
            pose = np.concatenate((wrist[0, index, :3, 3], wrist[0, index, :3, :2].reshape(6)))
            relative = np.linalg.inv(wrist[0, index]) @ wrist[0]
            relations = np.concatenate((relative[:, :3, 3], relative[:, :3, :2].reshape(2, 6)), axis=1).reshape(18)
            ict[index] = np.concatenate(([index + 1], pose, relations, [-1]))
        labels = _save_npz(tmp_path / f"{session}_labels.npz", x_ict=ict,
            ict_mask=np.array([True, True, False, False, False, False, False, False]),
            y_action=action, action_valid_mask=np.ones((50, 30), dtype=bool),
            anchor_qpc_ns=qpc[0], conditioning_source_qpc_ns=np.array([qpc[0], qpc[0], -1, -1, -1, -1, -1, -1]),
            target_qpc_ns=qpc[1:], target_source_frame_ids=np.arange(1, 51, dtype=np.int64),
            target_robot_rows=np.arange(1, 51, dtype=np.int64), current_robot_row=np.array(0, dtype=np.int64),
            reference_camera_to_world=np.eye(4))
        rgb, causal = {}, {}
        for branch, color in (("raw", (10, 20, 30)), ("robotized", (90, 80, 70))):
            image_path = tmp_path / f"{session}_{branch}.png"
            Image.new("RGB", (8, 6), color).save(image_path)
            rgb[branch] = _ref(image_path)
            causal[branch] = _save_json(tmp_path / f"{session}_{branch}_causal.json", {
                "schema": "causal-training-input-v1", "status": "PASSED",
                "input_class": "CAUSAL_TRAINING_INPUT", "sample_id": session,
                "source_frame_id": 0, "branch": branch,
                "image_sha256": rgb[branch]["sha256"], "donor_frame_ids": [0],
                "labels_sha256": labels["sha256"], "anchor_qpc_ns": int(qpc[0]),
                "timestamp_unit": FIXTURE_POSE_CLOCK["timestamp_unit"],
                "clock_domain": FIXTURE_POSE_CLOCK["clock_domain"],
                "evidence_refs": [evidence], "fixture_only": True,
            })
        manifest["samples"].append({
            "sample_id": session, "session_id": session, "source_frame_id": 0,
            "source_group_id": session, "split": split, "labels": labels,
            "rgb": rgb, "causal_receipts": causal,
            "real_robot_action_sidecar": sidecar,
            "real_robot_action_receipt": _save_json(tmp_path / f"{session}_control_receipt.json", {
                "schema": "real-revo2-synchronized-actions-v1", "status": "PASSED",
                "target": TARGET, "control_ground_truth": True, "synchronized": True,
                "real_robot_action": True, "session_id": session,
                "source_kind": "REAL_ROBOT_TELEOP", "sidecar_sha256": sidecar["sha256"],
                **FIXTURE_POSE_CLOCK,
                "hand_command_channel_order": [f"fixture_channel_{number}" for number in range(6)],
                "hand_command_unit": "TEST_SYNTHETIC_ARBITRARY_UNITS",
                "labels_sha256": labels["sha256"], "evidence_refs": [evidence], "fixture_only": True,
            }),
            "source_group_receipt": _save_json(tmp_path / f"{session}_group_receipt.json", {
                "schema": "independent-acquisition-source-group-v1", "status": "PASSED",
                "session_id": session, "source_group_id": session, "independent_acquisition": True,
                "evidence_refs": [evidence], "fixture_only": True,
            }),
        })
    path = tmp_path / "paired_manifest.json"
    _save_json(path, manifest)
    return path, manifest


def _rewrite_manifest(path, manifest):
    _save_json(path, manifest)


def _rewrite_labels(sample, **changes):
    path = Path(sample["labels"]["path"])
    with np.load(path, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    arrays.update(changes)
    sample["labels"] = _save_npz(path, **arrays)
    receipt_path = Path(sample["real_robot_action_receipt"]["path"])
    receipt = json.loads(receipt_path.read_text())
    receipt["labels_sha256"] = sample["labels"]["sha256"]
    sample["real_robot_action_receipt"] = _save_json(receipt_path, receipt)
    for branch in ("raw", "robotized"):
        receipt_path = Path(sample["causal_receipts"][branch]["path"])
        receipt = json.loads(receipt_path.read_text())
        receipt["labels_sha256"] = sample["labels"]["sha256"]
        receipt["anchor_qpc_ns"] = int(arrays["anchor_qpc_ns"])
        sample["causal_receipts"][branch] = _save_json(receipt_path, receipt)


def test_real_manifest_required_and_fixture_is_explicit(paired_manifest):
    path, _ = paired_manifest
    with pytest.raises(TrainingDataError, match="synthetic fixture"):
        validate_training_manifest(path)
    result = validate_training_manifest(path, allow_synthetic_fixture=True)
    assert result["sample_counts"] == {"train": 1, "validation": 1}


def test_paired_dataset_shape_dtype_and_identical_targets(paired_manifest):
    path, _ = paired_manifest
    raw = Revo2ICTDataset(path, "raw", "train", allow_synthetic_fixture=True)
    robotized = Revo2ICTDataset(path, "robotized", "train", allow_synthetic_fixture=True)
    first, second = raw[0], robotized[0]
    assert first["x_rgb"].shape == (3, 240, 320)
    assert first["x_ict"].shape == (8, 29)
    assert first["ict_mask"].shape == (8,)
    assert first["y_action"].shape == (50, 30)
    assert str(first["y_action"].dtype) == "torch.float32"
    assert str(first["action_valid_mask"].dtype) == "torch.bool"
    assert np.array_equal(first["y_action"].numpy(), second["y_action"].numpy())
    assert not np.array_equal(first["x_rgb"].numpy(), second["x_rgb"].numpy())
    assert np.array_equal(first["y_action"].numpy(), raw[0]["y_action"].numpy())
    assert raw.dataset_sha == robotized.dataset_sha


@pytest.mark.parametrize("mutation,match", [
    ("digital", "digital retarget"), ("future", "future sensor"),
    ("nan", "nonfinite admitted"), ("wrong_target", "does not match"),
])
def test_admitted_labels_reject_missing_provenance_and_nonfinite(paired_manifest, mutation, match):
    path, manifest = paired_manifest
    sample = manifest["samples"][0]
    with np.load(sample["labels"]["path"], allow_pickle=False) as source:
        action, source_qpc = source["y_action"].copy(), source["conditioning_source_qpc_ns"].copy()
    if mutation == "digital":
        _rewrite_labels(sample, q_hand=np.zeros((50, 2, 22)))
    elif mutation == "future":
        source_qpc[0] += 1
        _rewrite_labels(sample, conditioning_source_qpc_ns=source_qpc)
    else:
        action[0, 0] = np.nan if mutation == "nan" else 99
        _rewrite_labels(sample, y_action=action)
    _rewrite_manifest(path, manifest)
    with pytest.raises(TrainingDataError, match=match):
        validate_training_manifest(path, allow_synthetic_fixture=True)


def test_unknown_target_is_masked_and_sanitized(paired_manifest):
    path, manifest = paired_manifest
    sample = manifest["samples"][0]
    with np.load(sample["labels"]["path"], allow_pickle=False) as source:
        action, mask = source["y_action"].copy(), source["action_valid_mask"].copy()
    action[0, 18] = np.nan
    mask[0, 18] = False
    _rewrite_labels(sample, y_action=action, action_valid_mask=mask)
    _rewrite_manifest(path, manifest)
    sample = Revo2ICTDataset(path, "raw", "train", allow_synthetic_fixture=True)[0]
    assert sample["y_action"][0, 18].item() == 0
    assert not sample["action_valid_mask"][0, 18].item()


@pytest.mark.parametrize("mutation,match", [
    ("group_overlap", "crosses train/validation"),
    ("no_control", "missing reviewed sample fields"),
    ("future_rgb", "donor"),
    ("offline_rgb", "qualification missing"),
    ("sha_drift", "SHA/size drift"),
])
def test_manifest_evidence_is_fail_closed(paired_manifest, mutation, match):
    path, manifest = paired_manifest
    first = manifest["samples"][0]
    if mutation == "group_overlap":
        manifest["samples"][1]["source_group_id"] = first["source_group_id"]
    elif mutation == "no_control":
        first.pop("real_robot_action_sidecar")
    elif mutation in {"future_rgb", "offline_rgb"}:
        receipt_path = Path(first["causal_receipts"]["robotized"]["path"])
        receipt = json.loads(receipt_path.read_text())
        if mutation == "future_rgb":
            receipt["donor_frame_ids"] = [1]
        else:
            receipt["input_class"] = "OFFLINE_VISUALIZATION_ONLY"
        first["causal_receipts"]["robotized"] = _save_json(receipt_path, receipt)
    else:
        Path(first["rgb"]["raw"]["path"]).write_bytes(b"drift")
    _rewrite_manifest(path, manifest)
    with pytest.raises(TrainingDataError, match=match):
        validate_training_manifest(path, allow_synthetic_fixture=True)


def test_original_nas_is_not_a_training_manifest(tmp_path):
    path = tmp_path / "offline.json"
    _save_json(path, {"schema": "pick-place-original199-relocatable-export-v1", "training_eligible": False})
    with pytest.raises(TrainingDataError, match="reviewed paired real Revo2"):
        validate_training_manifest(path)


@pytest.mark.parametrize("split", ["train", "validation"])
def test_every_output_dimension_needs_effective_supervision(paired_manifest, split):
    path, manifest = paired_manifest
    sample = next(row for row in manifest["samples"] if row["split"] == split)
    with np.load(sample["labels"]["path"], allow_pickle=False) as source:
        mask = source["action_valid_mask"].copy()
    mask[:, 18:] = False
    _rewrite_labels(sample, action_valid_mask=mask)
    _rewrite_manifest(path, manifest)
    with pytest.raises(TrainingDataError, match=f"{split} output dimensions have no effective supervision"):
        validate_training_manifest(path, allow_synthetic_fixture=True)


def test_numeric_cache_reuses_verified_sidecar_and_detects_drift(paired_manifest, monkeypatch):
    path, manifest = paired_manifest
    dataset = Revo2ICTDataset(path, "raw", "train", allow_synthetic_fixture=True)
    sidecar_path = Path(manifest["samples"][0]["real_robot_action_sidecar"]["path"])
    original_read = Path.read_bytes
    sidecar_reads = []

    def read_bytes(current):
        if current == sidecar_path:
            sidecar_reads.append(current)
        return original_read(current)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    first, second = dataset[0], dataset[0]
    assert not sidecar_reads
    assert np.array_equal(first["y_action"].numpy(), second["y_action"].numpy())
    assert dataset.input_cache.current_bytes <= dataset.input_cache.max_bytes
    sidecar_path.write_bytes(b"changed immutable sidecar")
    with pytest.raises(TrainingDataError, match="cached immutable input metadata"):
        dataset[0]
    dataset.input_cache.clear()
    assert dataset.input_cache.current_bytes == 0


def test_hand_channel_order_cannot_be_ambiguous(paired_manifest):
    path, manifest = paired_manifest
    sample = manifest["samples"][0]
    receipt_path = Path(sample["real_robot_action_receipt"]["path"])
    receipt = json.loads(receipt_path.read_text())
    receipt["hand_command_channel_order"] = ["ambiguous"] * 6
    sample["real_robot_action_receipt"] = _save_json(receipt_path, receipt)
    _rewrite_manifest(path, manifest)
    with pytest.raises(TrainingDataError, match="six ordered channels"):
        validate_training_manifest(path, allow_synthetic_fixture=True)


@pytest.mark.parametrize("field,value", [
    ("hand_command_unit", "DIFFERENT_TEST_UNITS"),
    ("hand_command_channel_order", [f"other_channel_{number}" for number in range(6)]),
])
def test_control_semantics_cannot_change_across_sessions(paired_manifest, field, value):
    path, manifest = paired_manifest
    sample = manifest["samples"][1]
    receipt_path = Path(sample["real_robot_action_receipt"]["path"])
    receipt = json.loads(receipt_path.read_text())
    receipt[field] = value
    sample["real_robot_action_receipt"] = _save_json(receipt_path, receipt)
    _rewrite_manifest(path, manifest)
    with pytest.raises(TrainingDataError, match="order/unit differs"):
        validate_training_manifest(path, allow_synthetic_fixture=True)


@pytest.mark.parametrize("location,field,value", [
    ("calibration", "position_unit", "centimeters"),
    ("control", "position_unit", "centimeters"),
    ("calibration", "timestamp_unit", "milliseconds"),
    ("control", "timestamp_unit", "milliseconds"),
])
def test_pose_and_clock_units_require_reviewed_meters_nanoseconds(paired_manifest, location, field, value):
    path, manifest = paired_manifest
    holder = manifest if location == "calibration" else manifest["samples"][0]
    key = "calibration_receipt" if location == "calibration" else "real_robot_action_receipt"
    receipt_path = Path(holder[key]["path"])
    receipt = json.loads(receipt_path.read_text())
    receipt[field] = value
    holder[key] = _save_json(receipt_path, receipt)
    _rewrite_manifest(path, manifest)
    with pytest.raises(TrainingDataError, match=f"{field} must be"):
        validate_training_manifest(path, allow_synthetic_fixture=True)


@pytest.mark.parametrize("location,field,value", [
    ("calibration", "pose_frame_id", ""),
    ("control", "camera_frame_id", " "),
    ("control", "tcp_frame_ids", ["SYNTHETIC_LEFT_TCP"]),
    ("calibration", "clock_domain", ""),
    ("control", "clock_domain", " "),
])
def test_pose_frames_tcp_and_clock_cannot_be_unspecified(paired_manifest, location, field, value):
    path, manifest = paired_manifest
    holder = manifest if location == "calibration" else manifest["samples"][0]
    key = "calibration_receipt" if location == "calibration" else "real_robot_action_receipt"
    receipt_path = Path(holder[key]["path"])
    receipt = json.loads(receipt_path.read_text())
    receipt[field] = value
    holder[key] = _save_json(receipt_path, receipt)
    _rewrite_manifest(path, manifest)
    with pytest.raises(TrainingDataError, match=field):
        validate_training_manifest(path, allow_synthetic_fixture=True)


@pytest.mark.parametrize("field,value", [
    ("pose_frame_id", "DIFFERENT_SYNTHETIC_WORLD"),
    ("camera_frame_id", "DIFFERENT_SYNTHETIC_CAMERA"),
    ("tcp_frame_ids", ["SYNTHETIC_RIGHT_TCP", "SYNTHETIC_LEFT_TCP"]),
    ("clock_domain", "DIFFERENT_SYNTHETIC_CLOCK"),
])
def test_control_and_calibration_frames_clock_must_match(paired_manifest, field, value):
    path, manifest = paired_manifest
    sample = manifest["samples"][0]
    receipt_path = Path(sample["real_robot_action_receipt"]["path"])
    receipt = json.loads(receipt_path.read_text())
    receipt[field] = value
    sample["real_robot_action_receipt"] = _save_json(receipt_path, receipt)
    _rewrite_manifest(path, manifest)
    with pytest.raises(TrainingDataError, match="control/calibration pose frame, TCP, unit or clock_domain differs"):
        validate_training_manifest(path, allow_synthetic_fixture=True)


@pytest.mark.parametrize("field,value", [
    ("labels_sha256", "0" * 64),
    ("anchor_qpc_ns", 2_000_000_000),
    ("clock_domain", "DIFFERENT_SYNTHETIC_CLOCK"),
    ("timestamp_unit", "milliseconds"),
])
def test_rgb_causal_receipt_binds_exact_labels_anchor_and_clock(paired_manifest, field, value):
    path, manifest = paired_manifest
    sample = manifest["samples"][0]
    # The native frame ID and image SHA stay unchanged: they cannot alone prove
    # correspondence when recordings use another timestamp origin/domain.
    receipt_path = Path(sample["causal_receipts"]["robotized"]["path"])
    receipt = json.loads(receipt_path.read_text())
    receipt[field] = value
    sample["causal_receipts"]["robotized"] = _save_json(receipt_path, receipt)
    _rewrite_manifest(path, manifest)
    with pytest.raises(TrainingDataError, match=f"causal-training-input-v1/{field}"):
        validate_training_manifest(path, allow_synthetic_fixture=True)


@pytest.mark.parametrize("normalization", [
    {"method": "FIT_ALL_SESSIONS", "position_mean": [0, 0, 0], "position_std": [1, 1, 1]},
    {"method": "IDENTITY_NO_FIT", "position_mean": [0.1, 0, 0], "position_std": [1, 1, 1]},
])
def test_unproven_normalizer_is_forbidden(paired_manifest, normalization):
    path, manifest = paired_manifest
    manifest["normalization"] = normalization
    _rewrite_manifest(path, manifest)
    with pytest.raises(TrainingDataError, match="IDENTITY_NO_FIT"):
        validate_training_manifest(path, allow_synthetic_fixture=True)


def test_audit_retains_missing_session_and_never_qualifies(tmp_path):
    raw, processed = tmp_path / "raw", tmp_path / "processed"
    raw.mkdir()
    (processed / "sessions").mkdir(parents=True)
    report = audit_nas_roots(raw, processed, expected_sessions=1)
    assert report["status"] == "BLOCKED"
    assert report["training_start_allowed"] is False
    assert report["expected_sessions"] == 1
    assert report["sessions"][0]["audit_status"] == "BLOCKED_AUDIT_ERROR"
    assert report["missing_raw_sessions"] == ["001"]
    assert report["rgb_arrays_loaded"] is False
