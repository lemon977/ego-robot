"""Synthetic paired CPU data tests, without executing actual source operations."""
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from chaoyang.humanego_ict.data import ACTION_LAYOUT, TARGET, TrainingDataError
from chaoyang.humanego_ict.production_data import (
    SCHEMA, TRAINING_SCOPE, NORMALIZATION, SELECTION_POLICY,
    ProductionICTDataset, validate_production_manifest,
)
from chaoyang.humanego_ict.production_source import SOURCE_PROFILE, aggregate_group_plan, canonical_sha, fixed_anchor_ids, prepare_anchor
from chaoyang.humanego_ict.production_renderer import DEFAULT_RECIPE
from test_production_source import json_file, ref, replay_fixture, source_fixture


def production_fixture(tmp_path):
    value = source_fixture(tmp_path)
    replays = [replay_fixture(value, i) for i in range(2)]
    plan = aggregate_group_plan([r["metadata"] for r in replays])
    manifest = {"schema": SCHEMA, "pair_id": "f"*64, "fixture_only": True,
        "training_scope": TRAINING_SCOPE, "supervision_class": "DERIVED_RETARGET", "target": TARGET,
        "control_ground_truth": False, "physical_deployment_authorized": False, "action_layout": ACTION_LAYOUT,
        "normalization": NORMALIZATION, "source_profile": SOURCE_PROFILE, "selection_policy": SELECTION_POLICY,
        "source_contract_ref": json_file(tmp_path / "source_contract.json", value["contract"]),
        "mapper_ref": value["mapper_ref"], "calibration_ref": value["calibration_ref"], "group_plan": plan,
        "pose_clock": {"position_unit": "meters", "pose_frame_id": "REP103_X_FORWARD_Y_LEFT_Z_UP", "camera_frame_id": "ANCHOR_PAST_HEAD_FACTORY_CAMERA_PRIOR",
            "tcp_frame_ids": ["LEFT_INSTALLATION_WRIST_PRIOR", "RIGHT_INSTALLATION_WRIST_PRIOR"], "timestamp_unit": "nanoseconds", "clock_domain": value["contract"]["clock_domain"]},
        "samples": [], "candidate_anchors": []}
    manifest["rgb_recipe_ref"] = json_file(tmp_path / "recipe.json", DEFAULT_RECIPE)
    module_dir = Path(__import__("chaoyang.humanego_ict.production_data", fromlist=["__file__"]).__file__).parent
    packet = {"fixture_only": True, "task_id": "SYNTHETIC_PRODUCER_ONLY", "run_signature": manifest["pair_id"],
        "training_scope": TRAINING_SCOPE, "execution_mode": "HUMANEGO_ICT_PRODUCTION_PREPARE_CPU", "deadline_at": "2026-10-11T03:00:00+00:00",
        "writer": {"pid": 1, "proc_start_ticks": 1, "executor_epoch": 1, "fencing_token": "EXPLICIT_FIXTURE_FENCE", "scope": "SYNTHETIC_PRODUCER_ONLY"},
        "inputs": {"source_contract": manifest["source_contract_ref"], "resolved_input_manifest": value["contract"]["input_manifest_ref"],
            "recipe": manifest["rgb_recipe_ref"], "legacy_mapper_refs": [value["mapper_ref"], value["calibration_ref"]]},
        "runtime_code": [ref(module_dir / (name + ".py")) for name in ("production_data", "production_source", "production_renderer")]}
    manifest["producer_packet"] = json_file(tmp_path / "producer_packet.json", packet)
    for replay in replays:
        sid = replay["record"]["session_id"]
        for frame in fixed_anchor_ids(replay):
            result = prepare_anchor(replay, frame)
            manifest["candidate_anchors"].append({"session_id": sid, "source_frame_id": frame, "eligible": result["eligible"], "missing_items": result["missing_items"]})
            root = tmp_path / sid
            labels = root / "labels.npz"; np.savez(labels, **result["labels"])
            trace_ref = json_file(root / "trace.json", result["trace"])
            image = root / "rgb.png"; Image.fromarray(np.full((480, 640, 3), 70, np.uint8)).save(image)
            image_ref = ref(image)
            sample = {"sample_id": sid + "_000000", "session_id": sid, "source_frame_id": frame,
                "source_group_id": plan[sid]["source_group_id"], "split": plan[sid]["split"], "labels": ref(labels),
                "derivation_trace_ref": trace_ref, "rgb": {"raw": image_ref, "robotized": image_ref},
                "causal_receipts": {}, "branch_admission": {"raw": True, "robotized": True}}
            current = result["current"]
            times = {"head": current["head_source_qpc_ns"], "controller_left": int(current["controller_source_qpc_ns"][0]),
                "controller_right": int(current["controller_source_qpc_ns"][1]), "manus_left": int(current["manus_source_qpc_ns"][0]), "manus_right": int(current["manus_source_qpc_ns"][1])}
            render_trace = {"synthetic_fixture": True, "session_id": sid, "source_frame_id": frame,
                "anchor_qpc_ns": current["anchor_qpc_ns"], "clock_domain": current["clock_domain"], "timestamp_unit": "nanoseconds",
                "source_signature": current["source_signature"], "sensor_source_qpc_ns": times, "donor_frame_ids": [frame],
                "motion_inputs": "CURRENT_NATIVE_PLUS_PRIOR_ACCEPTED_ANCHOR_SOLUTION", "mask_inputs": "CURRENT_NATIVE_PROJECTED_GEOMETRY_ONLY",
                "clean_inputs": "CURRENT_RAW_RGB_SPATIAL_ONLY_NO_TEMPORAL_DONORS",
                "base_anchor": {"frame_id": frame, "anchor_qpc_ns": current["anchor_qpc_ns"], "sensor_source_qpc_ns": times, "source_signature": current["source_signature"]},
                "raw_rgb_array_sha256": __import__("hashlib").sha256(np.full((480, 640, 3), 70, np.uint8).tobytes()).hexdigest()}
            render_trace["producer_authorization"] = {key: packet[key] for key in ("task_id", "run_signature", "writer", "deadline_at")}
            render_trace["recipe_sha256"] = canonical_sha(DEFAULT_RECIPE)
            sample["robotized_render_trace_ref"] = json_file(root / "render_trace.json", render_trace)
            sample["robotized_render_diagnostics_ref"] = json_file(root / "render_diagnostics.json", {"synthetic_fixture": True,
                "motion": {"per_side_tracking": [{"numeric_tracking_status": "PASSED"}, {"numeric_tracking_status": "PASSED"}]}})
            for branch in ("raw", "robotized"):
                receipt = {"schema": "causal-training-input-v1", "status": "PASSED", "fixture_only": True,
                    "input_class": "CAUSAL_TRAINING_INPUT", "sample_id": sample["sample_id"], "source_frame_id": frame,
                    "branch": branch, "image_sha256": image_ref["sha256"], "labels_sha256": sample["labels"]["sha256"],
                    "anchor_qpc_ns": int(result["labels"]["anchor_qpc_ns"]), "timestamp_unit": "nanoseconds",
                    "clock_domain": value["contract"]["clock_domain"], "donor_frame_ids": [frame],
                    "source_signature": replay["metadata"]["source_signature"], "source_session_id": sid,
                    "evidence_refs": replay["actual_input_refs"] + ([sample["robotized_render_trace_ref"], sample["robotized_render_diagnostics_ref"]] if branch == "robotized" else [])}
                sample["causal_receipts"][branch] = json_file(root / (branch + "_causal.json"), receipt)
            manifest["samples"].append(sample)
    path = tmp_path / "manifest.json"; json_file(path, manifest)
    return path, manifest


def rewrite(path, manifest):
    json_file(path, manifest)


def test_numpy_preflight_json_summary_and_paired_labels(tmp_path):
    path, manifest = production_fixture(tmp_path)
    raw = validate_production_manifest(path, "raw", allow_synthetic_fixture=True)
    robot = validate_production_manifest(path, "robotized", allow_synthetic_fixture=True)
    json.dumps(raw, allow_nan=False)
    assert raw["sample_counts"] == {"train": 1, "validation": 1}
    assert raw["dataset_sha"] == robot["dataset_sha"] == ref(path)["sha256"]
    assert raw["clock_implementation_verification"] == "UNKNOWN_NOT_CLAIMED"
    assert raw["control_ground_truth"] is False and raw["physical_deployment_authorized"] is False
    assert all(v > 0 for v in raw["valid_action_counts"]["train"])
    assert raw["prepared_pair"]["samples"] == manifest["samples"]


def test_tensor_shapes_exact_repeat_and_mutation_check(tmp_path):
    import torch
    path, manifest = production_fixture(tmp_path)
    dataset = ProductionICTDataset(path, "raw", "train", allow_synthetic_fixture=True)
    first, second = dataset[0], dataset[0]
    for key, shape in {"x_rgb": (3, 240, 320), "x_ict": (8, 29), "ict_mask": (8,), "y_action": (50, 30), "action_valid_mask": (50, 30), "sample_weight": ()}.items():
        assert tuple(first[key].shape) == shape
        assert torch.equal(first[key], second[key])
        assert first[key].dtype == (torch.bool if key.endswith("mask") else torch.float32)
    dataset.verify_all_dependencies()
    source = dataset.session_refs[first["session_id"]][0]["path"]
    with open(source, "ab") as stream:
        stream.write(b" ")
    with pytest.raises(TrainingDataError, match="dependency changed"):
        dataset[0]


def test_fixture_not_admitted_without_flag(tmp_path):
    path, _ = production_fixture(tmp_path)
    with pytest.raises(TrainingDataError, match="synthetic production"):
        validate_production_manifest(path)


@pytest.mark.parametrize("field,value", [("control_ground_truth", True), ("physical_deployment_authorized", True), ("supervision_class", "REAL_CONTROL"), ("source_profile", "Core3_Auto")])
def test_control_or_profile_relabel_rejected(tmp_path, field, value):
    path, manifest = production_fixture(tmp_path); manifest[field] = value; rewrite(path, manifest)
    with pytest.raises(TrainingDataError):
        validate_production_manifest(path, allow_synthetic_fixture=True)


def test_source_recomputed_not_receipt_true_or_hash_only(tmp_path):
    path, manifest = production_fixture(tmp_path)
    sample = manifest["samples"][0]
    labels_path = sample["labels"]["path"]
    with np.load(labels_path) as stream:
        arrays = {k: stream[k] for k in stream.files}
    arrays["y_action"][0, 18] += .1
    np.savez(labels_path, **arrays); sample["labels"] = ref(labels_path)
    rewrite(path, manifest)
    with pytest.raises(TrainingDataError, match="differs from real raw replay"):
        validate_production_manifest(path, allow_synthetic_fixture=True)


def test_future_rgb_donor_or_wrong_anchor_rejected(tmp_path):
    path, manifest = production_fixture(tmp_path)
    sample = manifest["samples"][0]; receipt_ref = sample["causal_receipts"]["raw"]
    receipt = json.loads(open(receipt_ref["path"]).read()); receipt["donor_frame_ids"] = [1]
    sample["causal_receipts"]["raw"] = json_file(receipt_ref["path"], receipt); rewrite(path, manifest)
    with pytest.raises(TrainingDataError, match="donor"):
        validate_production_manifest(path, allow_synthetic_fixture=True)


def test_same_group_cannot_leak_across_split(tmp_path):
    path, manifest = production_fixture(tmp_path)
    manifest["samples"][1]["source_group_id"] = manifest["samples"][0]["source_group_id"]; rewrite(path, manifest)
    with pytest.raises(TrainingDataError, match="group/split"):
        validate_production_manifest(path, allow_synthetic_fixture=True)


def test_fixed_anchors_cannot_silently_omit_eligible_raw(tmp_path):
    path, manifest = production_fixture(tmp_path); manifest["samples"].pop(); rewrite(path, manifest)
    with pytest.raises(TrainingDataError, match="silently omitted"):
        validate_production_manifest(path, allow_synthetic_fixture=True)


def test_branch_failure_retained_does_not_upgrade_offline_rgb(tmp_path):
    path, manifest = production_fixture(tmp_path)
    sample = manifest["samples"][0]; sample["branch_admission"]["robotized"] = False; sample["robotized_missing_items"] = ["ROBOT_RENDER_QUALITY_FAILED"]
    rewrite(path, manifest)
    assert validate_production_manifest(path, "raw", allow_synthetic_fixture=True)["sample_counts"]["validation"] == 1
    with pytest.raises(TrainingDataError, match="both splits"):
        validate_production_manifest(path, "robotized", allow_synthetic_fixture=True)


def test_zero_coverage_hand_labels_cannot_fake30_outputs(tmp_path):
    path, manifest = production_fixture(tmp_path)
    sample = manifest["samples"][0]; labels_path = sample["labels"]["path"]
    with np.load(labels_path) as stream:
        arrays = {k: stream[k] for k in stream.files}
    arrays["action_valid_mask"][:, 18:] = False; np.savez(labels_path, **arrays); sample["labels"] = ref(labels_path); rewrite(path, manifest)
    with pytest.raises(TrainingDataError, match="differs from real raw replay"):
        validate_production_manifest(path, allow_synthetic_fixture=True)


@pytest.mark.parametrize("kind", ["future_sensor", "future_base", "offline_clean", "pixel_mismatch", "numeric_failure"])
def test_actual_robotized_trace_not_just_receipt_flag(tmp_path, kind):
    path, manifest = production_fixture(tmp_path); sample = manifest["samples"][0]
    reference = sample["robotized_render_diagnostics_ref"] if kind == "numeric_failure" else sample["robotized_render_trace_ref"]
    value = json.loads(open(reference["path"]).read())
    if kind == "future_sensor":
        value["sensor_source_qpc_ns"]["head"] = value["anchor_qpc_ns"] + 1
    elif kind == "future_base":
        value["base_anchor"]["frame_id"] = sample["source_frame_id"] + 1
    elif kind == "offline_clean":
        value["clean_inputs"] = "BIDIRECTIONAL_FUTURE_DONOR"
    elif kind == "pixel_mismatch":
        value["raw_rgb_array_sha256"] = "0"*64
    else:
        value["motion"]["per_side_tracking"][0]["numeric_tracking_status"] = "FAILED"
    updated = json_file(reference["path"], value)
    key = "robotized_render_diagnostics_ref" if kind == "numeric_failure" else "robotized_render_trace_ref"
    sample[key] = updated
    causal_ref = sample["causal_receipts"]["robotized"]; receipt = json.loads(open(causal_ref["path"]).read())
    receipt["evidence_refs"] = [updated if r == reference else r for r in receipt["evidence_refs"]]
    sample["causal_receipts"]["robotized"] = json_file(causal_ref["path"], receipt); rewrite(path, manifest)
    with pytest.raises(TrainingDataError, match="robotized"):
        validate_production_manifest(path, "robotized", allow_synthetic_fixture=True)


@pytest.mark.parametrize("kind", ["wrong_producer", "wrong_recipe", "fixture_as_production"])
def test_robotized_historical_producer_and_recipe_bound(tmp_path, kind):
    path, manifest = production_fixture(tmp_path); sample = manifest["samples"][0]; reference = sample["robotized_render_trace_ref"]
    value = json.loads(open(reference["path"]).read())
    if kind == "wrong_producer":
        value["producer_authorization"]["run_signature"] = "a"*64
    elif kind == "wrong_recipe":
        value["recipe_sha256"] = "a"*64
    else:
        value["synthetic_fixture"] = False
    updated = json_file(reference["path"], value); sample["robotized_render_trace_ref"] = updated
    receipt_ref = sample["causal_receipts"]["robotized"]; receipt = json.loads(open(receipt_ref["path"]).read())
    receipt["evidence_refs"] = [updated if r == reference else r for r in receipt["evidence_refs"]]
    sample["causal_receipts"]["robotized"] = json_file(receipt_ref["path"], receipt); rewrite(path, manifest)
    with pytest.raises(TrainingDataError, match="producer/recipe/fixture"):
        validate_production_manifest(path, "robotized", allow_synthetic_fixture=True)
