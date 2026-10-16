"""Explicit CPU synthetic fixtures: no production qualification or model calls."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from chaoyang.humanego_ict.data import ACTION_LAYOUT, TARGET, TrainingDataError, validate_training_manifest
from chaoyang.humanego_ict.derived_data import (
    CHANNEL_ORDER, SCHEMA, SOURCE_KIND, DerivedRevo2ICTDataset,
    recompute_derived_labels, validate_derived_preparation_manifest, validate_derived_training_manifest,
)
from chaoyang.humanego_ict.retarget import FEATURE_NAMES, REFERENCE_RETARGET_SHA256


def _ref(path):
    encoded = path.read_bytes()
    return {"path": str(path.absolute()), "bytes": len(encoded), "sha256": hashlib.sha256(encoded).hexdigest()}


def _json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return _ref(path)


def _npz(path, arrays):
    encoded = io.BytesIO()
    np.savez(encoded, **arrays)
    path.write_bytes(encoded.getvalue())
    return _ref(path)


def synthetic_hand_contract(source_ref):
    source_sha = source_ref["sha256"]
    opened = {name: 0.0 for name in FEATURE_NAMES}
    closed = {name: float(np.pi) for name in FEATURE_NAMES}
    opened.update(thumb_rotation=-float(np.pi), thumb_full=-1.0)
    closed.update(thumb_rotation=float(np.pi), thumb_full=1.0)
    return {
        "schema": "derived-revo2-hand-v1", "supervision_kind": "DERIVED_RETARGET",
        "synthetic_fixture_only": True, "hand_command_unit": "NORMALIZED_0_1",
        "channel_order": CHANNEL_ORDER, "reference_retarget_sha256": REFERENCE_RETARGET_SHA256,
        "feature_provider": {"kind": "explicit_joint_angles_v1", "source_sha256": source_sha, "source_ref": source_ref},
        "layout": {"joint_count": 25, "coordinate_frame": "wrist_local", "position_unit": "meters",
            "source_sha256": source_sha, "source_ref": source_ref,
            "finger_triplets": {name: [row] for name, row in zip(
                ("thumb", "index", "middle", "ring", "pinky"),
                ([1, 2, 3], [5, 6, 7], [9, 10, 11], [13, 14, 15], [17, 18, 19]))},
            "thumb_palm_indices": {"palm_root": 0, "index_mcp": 5, "middle_mcp": 9,
                "pinky_mcp": 17, "thumb_mcp": 1, "thumb_ip": 2, "thumb_tip": 3},
            "thumb_rotation_side_sign": {"left": 1, "right": -1},
            "palm_depth_side_sign": {"left": -1, "right": 1}},
        "mapping": {"kind": "DERIVED_AFFINE_THUMB_GATED_V1", "source_sha256": source_sha, "source_ref": source_ref,
            "calibration": {side: {"open": opened, "closed": closed} for side in ("left", "right")},
            "gains": {name: 1.0 for name in FEATURE_NAMES}, "thumb_rotation_enabled": True,
            "thumb_opposition_strong": 600, "thumb_full_gate": 400, "thumb_opposition_deadband": 20,
            "soft_margin": [[0, 0]] * 6},
    }


def _joints():
    points = np.zeros((25, 3), dtype=np.float64)
    points[[1, 2, 3]] = [[-.04, .02, 0], [-.02, .035, .005], [-.008, .04, .012]]
    points[[5, 6, 7]] = [[-.025, .06, 0], [-.025, .09, 0], [-.02, .11, .01]]
    points[[9, 10, 11]] = [[0, .065, 0], [0, .10, 0], [.015, .12, .01]]
    points[[13, 14, 15]] = [[.02, .055, 0], [.02, .09, 0], [.035, .11, .01]]
    points[[17, 18, 19]] = [[.035, .04, 0], [.035, .07, 0], [.04, .09, .01]]
    return points


@pytest.fixture
def derived_manifest(tmp_path):
    evidence = _json(tmp_path / "fixture_source_evidence.json", {"fixture_only": True})
    contract = _json(tmp_path / "retarget_contract.json", synthetic_hand_contract(evidence))
    code = _ref(Path(__file__).resolve().parents[2] / "src/chaoyang/humanego_ict/retarget.py")
    pose_clock = {"position_unit": "meters", "pose_frame_id": "SYNTHETIC_PICO_WORLD",
        "camera_frame_id": "SYNTHETIC_REFERENCE_CAMERA", "tcp_frame_ids": ["SYNTHETIC_LEFT", "SYNTHETIC_RIGHT"],
        "timestamp_unit": "nanoseconds", "clock_domain": "SYNTHETIC_CAPTURE_HOST_QPC"}
    manifest = {"schema": SCHEMA, "target": TARGET, "pair_id": "synthetic_paired_only",
        "supervision_class": "DERIVED_RETARGET", "control_ground_truth": False,
        "physical_deployment_authorized": False, "fixture_only": True,
        "action_layout": ACTION_LAYOUT, "ict_schema": "HAND_ONLY_29D_GRASP_UNKNOWN_MINUS_ONE",
        "normalization": {"method": "IDENTITY_NO_FIT", "position_mean": [0, 0, 0], "position_std": [1, 1, 1]},
        "retarget_contract": contract, "retarget_code": code, "pose_clock": pose_clock,
        "calibration_receipt": _json(tmp_path / "calibration.json", {
            "schema": "robot-camera-calibration-v1", "status": "PASSED", "fixture_only": True,
            "geometry_qualified": True, "physical_eye_identity_verified": True,
            "evidence_refs": [evidence], **pose_clock}), "samples": []}
    for split in ("train", "validation"):
        qpc = np.arange(51, dtype=np.int64) * 40_000_000 + 1_000_000_000
        controllers = np.broadcast_to(np.eye(4), (51, 2, 4, 4)).copy()
        controllers[:, 0, 0, 3] = np.arange(51) / 100
        controllers[:, 1, 0, 3] = np.arange(51) / 100 + .3
        prior = np.broadcast_to(np.eye(4), (2, 4, 4)).copy()
        prior[:, 0, 3] = [.02, -.02]
        source = _npz(tmp_path / f"{split}_source.npz", {
            "controller_pose_world": controllers, "controller_to_wrist": prior,
            "controller_valid": np.ones((51, 2), dtype=bool),
            "hand_joints_wrist": np.broadcast_to(_joints(), (51, 2, 25, 3)).copy(),
            "hand_joints_valid": np.ones((51, 2), dtype=bool), "qpc_ns": qpc,
            "source_frame_id": np.arange(51, dtype=np.int64), "segment_id": np.zeros(51, dtype=np.int64),
            "controller_source_qpc_ns": np.repeat(qpc[:, None], 2, axis=1),
            "hand_source_qpc_ns": np.repeat(qpc[:, None], 2, axis=1), "reference_camera_to_world": np.eye(4)})
        prepared = recompute_derived_labels(source, contract, 0, retarget_code_ref=code, allow_synthetic_fixture=True)
        assert not prepared["missing_items"]
        labels = _npz(tmp_path / f"{split}_labels.npz", prepared["labels"])
        sample = {"sample_id": split, "session_id": split, "source_frame_id": 0,
            "source_group_id": split, "split": split, "labels": labels, "human_source_sidecar": source,
            "derivation_receipt": _json(tmp_path / f"{split}_derivation.json", {
                "schema": "humanego-revo2-derived-retarget-v1", "status": "PASSED", "fixture_only": True,
                "source_kind": SOURCE_KIND, "supervision_class": "DERIVED_RETARGET",
                "control_ground_truth": False, "physical_deployment_authorized": False,
                "session_id": split, "source_sha256": source["sha256"], "labels_sha256": labels["sha256"],
                "retarget_contract_sha256": contract["sha256"], "retarget_code_sha256": code["sha256"],
                "hand_command_channel_order": CHANNEL_ORDER, "hand_command_unit": "NORMALIZED_0_1",
                "evidence_refs": [evidence, source, contract, code], **pose_clock}),
            "source_group_receipt": _json(tmp_path / f"{split}_group.json", {
                "schema": "independent-acquisition-source-group-v1", "status": "PASSED", "fixture_only": True,
                "session_id": split, "source_group_id": split, "independent_acquisition": True,
                "evidence_refs": [evidence]}), "rgb": {}, "causal_receipts": {}}
        for branch, color in (("raw", (20, 30, 40)), ("robotized", (70, 80, 90))):
            image = tmp_path / f"{split}_{branch}.png"
            Image.new("RGB", (8, 6), color).save(image)
            sample["rgb"][branch] = _ref(image)
            sample["causal_receipts"][branch] = _json(tmp_path / f"{split}_{branch}_causal.json", {
                "schema": "causal-training-input-v1", "status": "PASSED", "fixture_only": True,
                "input_class": "CAUSAL_TRAINING_INPUT", "sample_id": split, "source_frame_id": 0,
                "branch": branch, "image_sha256": sample["rgb"][branch]["sha256"],
                "labels_sha256": labels["sha256"], "anchor_qpc_ns": int(qpc[0]),
                "timestamp_unit": "nanoseconds", "clock_domain": pose_clock["clock_domain"],
                "donor_frame_ids": [0], "evidence_refs": [evidence]})
        manifest["samples"].append(sample)
    path = tmp_path / "derived_manifest.json"
    _json(path, manifest)
    return path, manifest


def _replace_receipt(holder, key, changes):
    path = Path(holder[key]["path"])
    value = json.loads(path.read_text())
    value.update(changes)
    holder[key] = _json(path, value)


def _update_labels(sample, arrays):
    sample["labels"] = _npz(Path(sample["labels"]["path"]), arrays)
    _replace_receipt(sample, "derivation_receipt", {"labels_sha256": sample["labels"]["sha256"]})
    for branch in ("raw", "robotized"):
        _replace_receipt(sample["causal_receipts"], branch, {"labels_sha256": sample["labels"]["sha256"],
            "anchor_qpc_ns": int(arrays["anchor_qpc_ns"])})


def _update_source(sample, manifest, arrays):
    sample["human_source_sidecar"] = _npz(Path(sample["human_source_sidecar"]["path"]), arrays)
    receipt = json.loads(Path(sample["derivation_receipt"]["path"]).read_text())
    evidence = [sample["human_source_sidecar"] if ref["path"] == sample["human_source_sidecar"]["path"] else ref
                for ref in receipt["evidence_refs"]]
    _replace_receipt(sample, "derivation_receipt", {
        "source_sha256": sample["human_source_sidecar"]["sha256"], "evidence_refs": evidence})
    prepared = recompute_derived_labels(sample["human_source_sidecar"], manifest["retarget_contract"], 0,
        retarget_code_ref=manifest["retarget_code"], allow_synthetic_fixture=True)
    _update_labels(sample, prepared["labels"])
    return prepared


def test_preparation_is_json_compatible_and_never_authorizes_training(derived_manifest):
    path, _ = derived_manifest
    with pytest.raises(TrainingDataError, match="synthetic fixture"):
        validate_derived_preparation_manifest(path)
    result = validate_derived_preparation_manifest(path, allow_synthetic_fixture=True)
    assert result["preparation_status"] == "DERIVED_NUMERIC_LABELS_VERIFIED"
    assert result["training_start_authorized"] is False
    assert result["control_ground_truth"] is False
    assert result["sample_counts"] == {"train": 1, "validation": 1}
    json.dumps(result)


def test_derived_cannot_enter_real_control_reader(derived_manifest):
    path, _ = derived_manifest
    with pytest.raises(TrainingDataError, match="reviewed paired real Revo2"):
        validate_training_manifest(path, allow_synthetic_fixture=True)


def test_pair_shares_labels_and_shapes(derived_manifest):
    path, _ = derived_manifest
    raw = DerivedRevo2ICTDataset(path, "raw", "train", allow_synthetic_fixture=True)
    robotized = DerivedRevo2ICTDataset(path, "robotized", "train", allow_synthetic_fixture=True)
    first, second = raw[0], robotized[0]
    assert first["x_rgb"].shape == (3, 240, 320)
    assert first["x_ict"].shape == (8, 29)
    assert first["y_action"].shape == (50, 30)
    assert str(first["y_action"].dtype) == "torch.float32"
    assert str(first["action_valid_mask"].dtype) == "torch.bool"
    assert first["control_ground_truth"] is False
    assert np.array_equal(first["y_action"].numpy(), second["y_action"].numpy())
    assert not np.array_equal(first["x_rgb"].numpy(), second["x_rgb"].numpy())
    assert np.array_equal(first["y_action"].numpy(), raw[0]["y_action"].numpy())
    assert first["y_action"][0, 0].item() == pytest.approx(.03)
    assert raw.dataset_sha == robotized.dataset_sha


@pytest.mark.parametrize("mutation", ["offline_rgb", "future_rgb", "labels_sha", "clock"])
def test_prepared_labels_do_not_override_causal_rgb_gate(derived_manifest, mutation):
    path, manifest = derived_manifest
    sample = manifest["samples"][0]
    changes = {"offline_rgb": {"input_class": "OFFLINE_VISUALIZATION_ONLY"},
        "future_rgb": {"donor_frame_ids": [1]}, "labels_sha": {"labels_sha256": "0" * 64},
        "clock": {"clock_domain": "ANOTHER_SYNTHETIC_CLOCK"}}[mutation]
    _replace_receipt(sample["causal_receipts"], "robotized", changes)
    _json(path, manifest)
    prepared = validate_derived_preparation_manifest(path, allow_synthetic_fixture=True)
    assert prepared["preparation_status"] == "DERIVED_NUMERIC_LABELS_VERIFIED"
    assert prepared["training_admission"]["robotized"]
    validate_derived_training_manifest(path, "raw", allow_synthetic_fixture=True)
    with pytest.raises(TrainingDataError, match="BLOCKED_RGB_CAUSAL_ADMISSION"):
        validate_derived_training_manifest(path, "robotized", allow_synthetic_fixture=True)


@pytest.mark.parametrize("mutation,match", [
    ("labels", "differs from recomputed"), ("mask", "differs from recomputed"),
    ("group", "crosses train/validation"), ("source_sha", "source_sha256"),
    ("ground_truth", "non-ground-truth"),
])
def test_source_recomputation_and_identity_are_not_receipt_booleans(derived_manifest, mutation, match):
    path, manifest = derived_manifest
    sample = manifest["samples"][0]
    if mutation in {"labels", "mask"}:
        with np.load(sample["labels"]["path"], allow_pickle=False) as source:
            arrays = {key: source[key].copy() for key in source.files}
        if mutation == "labels":
            arrays["y_action"][0, 18] += .1
        else:
            arrays["action_valid_mask"][0, 18] = False
        _update_labels(sample, arrays)
    elif mutation == "group":
        manifest["samples"][1]["source_group_id"] = sample["source_group_id"]
    elif mutation == "source_sha":
        _replace_receipt(sample, "derivation_receipt", {"source_sha256": "0" * 64})
    else:
        manifest["control_ground_truth"] = True
    _json(path, manifest)
    with pytest.raises(TrainingDataError, match=match):
        validate_derived_preparation_manifest(path, allow_synthetic_fixture=True)


@pytest.mark.parametrize("missing", ["calibration", "source_group"])
def test_preparation_does_not_bless_absent_qualification(derived_manifest, missing):
    path, manifest = derived_manifest
    if missing == "calibration":
        manifest.pop("calibration_receipt")
    else:
        manifest["samples"][0].pop("source_group_receipt")
    _json(path, manifest)
    prepared = validate_derived_preparation_manifest(path, allow_synthetic_fixture=True)
    assert prepared["training_start_authorized"] is False
    with pytest.raises(TrainingDataError, match="CALIBRATION|SOURCE_GROUP"):
        validate_derived_training_manifest(path, "raw", allow_synthetic_fixture=True)


def test_missing_source_layout_stays_unknown_and_masks_hand_targets(derived_manifest):
    path, manifest = derived_manifest
    contract_path = Path(manifest["retarget_contract"]["path"])
    contract = json.loads(contract_path.read_text())
    contract.pop("layout")
    manifest["retarget_contract"] = _json(contract_path, contract)
    for sample in manifest["samples"]:
        prepared = recompute_derived_labels(sample["human_source_sidecar"], manifest["retarget_contract"], 0,
            retarget_code_ref=manifest["retarget_code"], allow_synthetic_fixture=True)
        assert not prepared["labels"]["action_valid_mask"][:, 18:].any()
        assert any("BLOCKED_SOURCE_HAND_LAYOUT" in item for item in prepared["missing_items"])
        receipt = json.loads(Path(sample["derivation_receipt"]["path"]).read_text())
        evidence = [manifest["retarget_contract"] if ref["path"] == str(contract_path) else ref
                    for ref in receipt["evidence_refs"]]
        _replace_receipt(sample, "derivation_receipt", {
            "retarget_contract_sha256": manifest["retarget_contract"]["sha256"], "evidence_refs": evidence})
        _update_labels(sample, prepared["labels"])
    _json(path, manifest)
    result = validate_derived_preparation_manifest(path, allow_synthetic_fixture=True)
    assert result["missing_items"]
    with pytest.raises(TrainingDataError, match="BLOCKED_EFFECTIVE_SUPERVISION"):
        validate_derived_training_manifest(path, "raw", allow_synthetic_fixture=True)


def test_source_drift_is_detected_by_numeric_cache(derived_manifest):
    path, manifest = derived_manifest
    dataset = DerivedRevo2ICTDataset(path, "raw", "train", allow_synthetic_fixture=True)
    dataset[0]
    Path(manifest["samples"][0]["human_source_sidecar"]["path"]).write_bytes(b"fixture drift")
    with pytest.raises(TrainingDataError, match="cached immutable input metadata"):
        dataset[0]


def test_future_current_source_is_preserved_as_blocker_not_admitted(derived_manifest):
    path, manifest = derived_manifest
    sample = manifest["samples"][0]
    with np.load(sample["human_source_sidecar"]["path"], allow_pickle=False) as source:
        arrays = {key: source[key].copy() for key in source.files}
    arrays["controller_source_qpc_ns"][0, 0] += 1
    prepared = _update_source(sample, manifest, arrays)
    assert "BLOCKED_FUTURE_CONTROLLER_CONDITIONING" in prepared["missing_items"]
    assert not prepared["labels"]["ict_mask"][0]
    _json(path, manifest)
    with pytest.raises(TrainingDataError, match="BLOCKED_FUTURE_CONTROLLER_CONDITIONING"):
        validate_derived_training_manifest(path, "raw", allow_synthetic_fixture=True)


def test_unknown_hand_labels_can_be_masked_without_inventing_truth(derived_manifest):
    path, manifest = derived_manifest
    sample = manifest["samples"][0]
    with np.load(sample["human_source_sidecar"]["path"], allow_pickle=False) as source:
        arrays = {key: source[key].copy() for key in source.files}
    arrays["hand_joints_valid"][1, 0] = False
    prepared = _update_source(sample, manifest, arrays)
    labels = {key: value.copy() for key, value in prepared["labels"].items()}
    labels["y_action"][0, 18:24] = np.nan
    _update_labels(sample, labels)
    _json(path, manifest)
    item = DerivedRevo2ICTDataset(path, "raw", "train", allow_synthetic_fixture=True)[0]
    assert not item["action_valid_mask"][0, 18:24].any()
    assert np.array_equal(item["y_action"][0, 18:24].numpy(), np.zeros(6))


@pytest.mark.parametrize("changes", [{"geometry_qualified": False}, {"clock_domain": "ANOTHER_SYNTHETIC_CLOCK"}])
def test_derived_does_not_weaken_physical_calibration_clock_gate(derived_manifest, changes):
    path, manifest = derived_manifest
    _replace_receipt(manifest, "calibration_receipt", changes)
    _json(path, manifest)
    assert validate_derived_preparation_manifest(path, allow_synthetic_fixture=True)["training_admission"]["raw"]
    with pytest.raises(TrainingDataError, match="BLOCKED_PHYSICAL_CALIBRATION_OR_CLOCK"):
        validate_derived_training_manifest(path, "raw", allow_synthetic_fixture=True)


def test_preparation_module_import_has_no_torch_or_gpu():
    code = "import sys; import chaoyang.humanego_ict.derived_data; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-B", "-c", code], check=True)


@pytest.mark.parametrize("component", ["feature_provider", "layout", "mapping"])
def test_hex_claim_without_source_ref_is_not_source_evidence(derived_manifest, component):
    path, manifest = derived_manifest
    contract_path = Path(manifest["retarget_contract"]["path"])
    contract = json.loads(contract_path.read_text())
    contract[component].pop("source_ref")
    manifest["retarget_contract"] = _json(contract_path, contract)
    _json(path, manifest)
    with pytest.raises(TrainingDataError, match=f"{component} requires immutable source_ref"):
        validate_derived_preparation_manifest(path, allow_synthetic_fixture=True)


def test_unknown_group_is_not_blessed_by_a_boolean_receipt(derived_manifest):
    path, manifest = derived_manifest
    sample = manifest["samples"][0]
    sample["source_group_id"] = "UNKNOWN_SOURCE_GROUP"
    _replace_receipt(sample, "source_group_receipt", {"source_group_id": "UNKNOWN_SOURCE_GROUP"})
    _json(path, manifest)
    validate_derived_preparation_manifest(path, allow_synthetic_fixture=True)
    with pytest.raises(TrainingDataError, match="UNKNOWN source group"):
        validate_derived_training_manifest(path, "raw", allow_synthetic_fixture=True)
