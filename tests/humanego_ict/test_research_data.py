"""Explicit synthetic research fixtures: no production conversion or models."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pytest
from PIL import Image

from chaoyang.humanego_ict.data import ACTION_LAYOUT, TARGET, TrainingDataError
from chaoyang.humanego_ict.derived_data import SCHEMA as PAIR_SCHEMA, SOURCE_KIND, CHANNEL_ORDER, recompute_derived_labels
from chaoyang.humanego_ict.research_data import (SCHEMA, TRAINING_SCOPE, COORDINATE_ASSUMPTIONS,
    ResearchDerivedICTDataset, validate_research_manifest)
from chaoyang.humanego_ict.source_adapter import SOURCE_ROLES


def helper(name):
    path = Path(__file__).with_name(name + ".py")
    spec = importlib.util.spec_from_file_location("research_fixture_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


native_helpers = helper("test_source_adapter")
label_helpers = helper("test_derived_data")
_ref, _json, _npz = native_helpers._ref, native_helpers._json, native_helpers._npz


def research_fixture(tmp_path, *, target_skew_ns=0, hand_missing=0, original_gate_ms=40, minimum_coverage=.95):
    evidence = _json(tmp_path / "fixture_evidence.json", {"fixture_only": True, "kind": "EXPLICIT_TOY"})
    hand_contract = _json(tmp_path / "hand_contract.json", label_helpers.synthetic_hand_contract(evidence))
    code = _ref(Path(__file__).resolve().parents[2] / "src/chaoyang/humanego_ict/retarget.py")
    pair = {"schema": PAIR_SCHEMA, "target": TARGET, "pair_id": "SYNTHETIC_RESEARCH_PAIR",
        "supervision_class": "DERIVED_RETARGET", "control_ground_truth": False,
        "physical_deployment_authorized": False, "fixture_only": True, "action_layout": ACTION_LAYOUT,
        "ict_schema": "HAND_ONLY_29D_GRASP_UNKNOWN_MINUS_ONE",
        "normalization": {"method": "IDENTITY_NO_FIT", "position_mean": [0,0,0], "position_std": [1,1,1]},
        "retarget_contract": hand_contract, "retarget_code": code, "samples": []}
    clock = {"schema": "capture-host-qpc-nanoseconds-proof-v1", "status": "PASSED", "fixture_only": True,
        "timestamp_unit": "nanoseconds", "clock_domain": "SYNTHETIC_CAPTURE_QPC",
        "clock_value_unit": "nanoseconds", "implementation_kind": "HOST_MONOTONIC_NANOSECONDS",
        "qualification": "VERIFIED_NS_SOURCE_AND_SINGLE_CAPTURE_HOST_DOMAIN", "capture_host_id": "SYNTHETIC_HOST",
        "clock_semantics": "HOST_ARRIVAL_RECEIVE_TIME_NOT_EXPOSURE_SYNCHRONIZATION",
        "units_per_second": 1_000_000_000, "implementation_ref": evidence,
        "session_bindings": [], "evidence_refs": [evidence]}
    envelope = {"schema": SCHEMA, "training_scope": TRAINING_SCOPE, "supervision_class": "DERIVED_RETARGET",
        "control_ground_truth": False, "physical_deployment_authorized": False, "fixture_only": True,
        "coordinate_assumptions": dict(COORDINATE_ASSUMPTIONS), "sample_time_receipts": {}}
    native_records, time_records = {}, {}
    for split in ("train", "validation"):
        directory = tmp_path / split
        directory.mkdir()
        f = native_helpers.source_fixture.__wrapped__(directory)
        f["session"]["session_id"] = split
        f["arrays"][SOURCE_ROLES["hand_joints"]][:] = label_helpers._joints()
        if target_skew_ns:
            qpc = np.arange(55, dtype=np.int64) * 40_000_000 + 1_000_000_000
            f["arrays"][SOURCE_ROLES["qpc"]] = qpc
            for role in ("controller_source_qpc", "left_hand_source_qpc", "right_hand_source_qpc"):
                f["arrays"][SOURCE_ROLES[role]] = qpc - target_skew_ns
        if hand_missing:
            f["arrays"][SOURCE_ROLES["hand_valid"]][4:4+hand_missing] = False
        original = Path(f["contract"]["time_gate_source_ref"]["path"])
        with h5py.File(original, "r+") as hdf:
            hdf.attrs["gate_ms"] = original_gate_ms
            hdf.attrs["max_skew_ms"] = original_gate_ms
            hdf.attrs["min_hand_coverage"] = minimum_coverage
            hdf.attrs["min_complete_coverage"] = minimum_coverage
        f["contract"]["current_max_past_skew_ns"] = original_gate_ms * 1_000_000
        original_ref = _ref(original)
        f["contract"]["time_gate_source_ref"] = original_ref
        f["session"]["source_refs"]["dataset.hdf5"] = original_ref
        native_helpers._publish_changes(f, source=True)
        replay = native_helpers._adapt(f)
        pair["pose_clock"] = f["contract"]["pose_clock"]
        sidecar = _npz(directory / "source_window.npz", replay["source_arrays"])
        prepared = recompute_derived_labels(sidecar, hand_contract, 0, retarget_code_ref=code,
                                           allow_synthetic_fixture=True)
        labels = _npz(directory / "labels.npz", prepared["labels"])
        pose_clock = pair["pose_clock"]
        sample = {"sample_id": split, "session_id": split, "source_frame_id": 3,
            "source_group_id": split, "split": split, "labels": labels, "human_source_sidecar": sidecar,
            "rgb": {}, "causal_receipts": {}, "derivation_receipt": _json(directory / "derived.json", {
                "schema": "humanego-revo2-derived-retarget-v1", "status": "PASSED", "fixture_only": True,
                "source_kind": SOURCE_KIND, "supervision_class": "DERIVED_RETARGET", "control_ground_truth": False,
                "physical_deployment_authorized": False, "session_id": split,
                "source_sha256": sidecar["sha256"], "labels_sha256": labels["sha256"],
                "retarget_contract_sha256": hand_contract["sha256"], "retarget_code_sha256": code["sha256"],
                "hand_command_channel_order": CHANNEL_ORDER, "hand_command_unit": "NORMALIZED_0_1",
                "evidence_refs": [evidence, sidecar, hand_contract, code], **pose_clock}),
            "source_group_receipt": _json(directory / "group.json", {
                "schema": "independent-acquisition-source-group-v1", "status": "PASSED", "fixture_only": True,
                "session_id": split, "source_group_id": split, "independent_acquisition": True,
                "evidence_refs": [evidence]})}
        for branch, color in (("raw", (20,30,40)), ("robotized", (70,80,90))):
            image = directory / (branch + ".png")
            Image.new("RGB", (8,6), color).save(image)
            sample["rgb"][branch] = _ref(image)
            sample["causal_receipts"][branch] = _json(directory / (branch + "_causal.json"), {
                "schema": "causal-training-input-v1", "status": "PASSED", "fixture_only": True,
                "input_class": "CAUSAL_TRAINING_INPUT", "sample_id": split, "source_frame_id": 3,
                "branch": branch, "image_sha256": sample["rgb"][branch]["sha256"], "labels_sha256": labels["sha256"],
                "anchor_qpc_ns": int(prepared["labels"]["anchor_qpc_ns"]), "timestamp_unit": "nanoseconds",
                "clock_domain": pose_clock["clock_domain"], "donor_frame_ids": [3], "evidence_refs": [evidence]})
        pair["samples"].append(sample)
        refs = {"native_source_ref": f["source"], "session_input_ref": f["session_ref"],
            "adapter_contract_ref": f["contract_ref"], "adapter_trace_ref": _json(directory / "trace.json", replay["trace"])}
        time_records[split] = {"schema": "humanego-ict-derived-research-time-v1", "status": "PASSED",
            "fixture_only": True, "sample_id": split, "session_id": split, "labels_sha256": labels["sha256"],
            "source_sha256": sidecar["sha256"], "timestamp_unit": "nanoseconds",
            "clock_domain": pose_clock["clock_domain"], "evidence_refs": list(refs.values()), **refs}
        clock["session_bindings"].append({"session_id": split, "capture_host_id": clock["capture_host_id"],
            "clock_domain": clock["clock_domain"], "timestamp_unit": "nanoseconds",
            "timestamp_roles": ["video_qpc_ns", "controller_source_qpc_ns", "hand_source_qpc_ns"],
            "native_source_ref": f["source"], "session_input_ref": f["session_ref"]})
        clock["evidence_refs"].extend([f["source"], f["session_ref"]])
        native_records[split] = f
    envelope["prepared_pair_ref"] = _json(tmp_path / "prepared_pair.json", pair)
    envelope["clock_proof_ref"] = _json(tmp_path / "clock.json", clock)
    for split, receipt in time_records.items():
        receipt["clock_proof_sha256"] = envelope["clock_proof_ref"]["sha256"]
        envelope["sample_time_receipts"][split] = _json(tmp_path / split / "time.json", receipt)
    path = tmp_path / "research_manifest.json"
    _json(path, envelope)
    return path, envelope, pair, clock, native_records


@pytest.fixture
def researched(tmp_path):
    return research_fixture(tmp_path)


def republish(fixture):
    path, envelope, pair, clock, _ = fixture
    envelope["prepared_pair_ref"] = _json(Path(envelope["prepared_pair_ref"]["path"]), pair)
    envelope["clock_proof_ref"] = _json(Path(envelope["clock_proof_ref"]["path"]), clock)
    for sample_id, ref in envelope["sample_time_receipts"].items():
        receipt = json.loads(Path(ref["path"]).read_text())
        receipt["clock_proof_sha256"] = envelope["clock_proof_ref"]["sha256"]
        envelope["sample_time_receipts"][sample_id] = _json(Path(ref["path"]), receipt)
    _json(path, envelope)


def admit(fixture, branch="raw"):
    return validate_research_manifest(fixture[0], branch, allow_synthetic_fixture=True)


def test_research_not_physical_qualification_and_identical_paired_labels(researched):
    assert "calibration_receipt" not in researched[2]
    result = admit(researched)
    json.dumps(result)
    assert result["sample_counts"] == {"train": 1, "validation": 1}
    assert not result["training_start_authorized"] and not result["physical_deployment_authorized"]
    raw = ResearchDerivedICTDataset(researched[0], "raw", "train", allow_synthetic_fixture=True)
    robotized = ResearchDerivedICTDataset(researched[0], "robotized", "train", allow_synthetic_fixture=True)
    a, b = raw[0], robotized[0]
    assert a["x_rgb"].shape == (3,240,320) and a["x_ict"].shape == (8,29)
    assert a["y_action"].shape == (50,30) and a["action_valid_mask"].all()
    assert str(a["y_action"].dtype) == "torch.float32" and str(a["ict_mask"].dtype) == "torch.bool"
    assert np.array_equal(a["y_action"].numpy(), b["y_action"].numpy())
    assert not np.array_equal(a["x_rgb"].numpy(), b["x_rgb"].numpy())
    assert np.array_equal(raw[0]["y_action"].numpy(), a["y_action"].numpy())
    assert raw.dataset_sha == robotized.dataset_sha
    assert raw.input_cache.current_bytes <= 128 * 1024**2


def test_synthetic_evidence_cannot_be_production(researched):
    with pytest.raises(TrainingDataError, match="synthetic"):
        validate_research_manifest(researched[0], "raw")


@pytest.mark.parametrize("field,value", [("status", "UNKNOWN"), ("units_per_second", 10_000_000),
    ("timestamp_unit", "seconds"), ("capture_host_id", ""), ("clock_semantics", "EXPOSURE_SYNCHRONIZED")])
def test_unknown_or_unproven_clock_never_becomes_true(researched, field, value):
    researched[3][field] = value
    republish(researched)
    with pytest.raises(TrainingDataError): admit(researched)


@pytest.mark.parametrize("mutation", ["other_host", "different_source", "missing_session", "missing_role"])
def test_clock_proof_is_bound_to_actual_capture_sources(researched, mutation):
    binding = researched[3]["session_bindings"][0]
    if mutation == "other_host": binding["capture_host_id"] = "ANOTHER_CAPTURE_HOST"
    elif mutation == "different_source": binding["native_source_ref"] = researched[3]["session_bindings"][1]["native_source_ref"]
    elif mutation == "missing_session": researched[3]["session_bindings"].pop()
    else: binding["timestamp_roles"].pop()
    republish(researched)
    with pytest.raises(TrainingDataError): admit(researched)


@pytest.mark.parametrize("mutation", ["offline", "future_donor", "labels", "clock"])
def test_research_does_not_weaken_rgb_causal_admission(researched, mutation):
    sample = researched[2]["samples"][0]
    changes = {"offline": {"input_class": "OFFLINE_RESEARCH_ONLY"}, "future_donor": {"donor_frame_ids": [4]},
        "labels": {"labels_sha256": "0"*64}, "clock": {"clock_domain": "OTHER_ORIGIN"}}[mutation]
    label_helpers._replace_receipt(sample["causal_receipts"], "robotized", changes)
    republish(researched)
    admit(researched, "raw")
    with pytest.raises(TrainingDataError): admit(researched, "robotized")


def test_independent_acquisition_groups_still_mandatory(researched):
    researched[2]["samples"][1]["source_group_id"] = "train"
    republish(researched)
    with pytest.raises(TrainingDataError, match="crosses train/validation"): admit(researched)


def test_real_native_replay_not_trace_boolean(researched):
    ref = researched[1]["sample_time_receipts"]["train"]
    receipt = json.loads(Path(ref["path"]).read_text())
    trace = json.loads(Path(receipt["adapter_trace_ref"]["path"]).read_text())
    trace["current_selected_native_rows"]["controllers"][0] = 0
    old = receipt["adapter_trace_ref"]
    receipt["adapter_trace_ref"] = _json(Path(old["path"]), trace)
    receipt["evidence_refs"] = [receipt["adapter_trace_ref"] if value == old else value for value in receipt["evidence_refs"]]
    researched[1]["sample_time_receipts"]["train"] = _json(Path(ref["path"]), receipt)
    _json(researched[0], researched[1])
    with pytest.raises(TrainingDataError, match="actual native replay"): admit(researched)


def test_original_p95_30ms_not_relaxed_to_40ms(tmp_path):
    fixture = research_fixture(tmp_path, target_skew_ns=31_000_000)
    with pytest.raises(TrainingDataError, match="p95 timing"): admit(fixture)


def test_original_coverage_gate_still_applies(tmp_path):
    fixture = research_fixture(tmp_path, hand_missing=3)
    with pytest.raises(TrainingDataError, match="coverage"): admit(fixture)


@pytest.mark.parametrize("changes", [{"original_gate_ms": 80}, {"minimum_coverage": .5}])
def test_rehashed_source_cannot_relax_existing_numeric_gates(tmp_path, changes):
    fixture = research_fixture(tmp_path, **changes)
    with pytest.raises(TrainingDataError, match="relaxed"): admit(fixture)


def test_all_hand_outputs_need_actual_supervision(tmp_path):
    fixture = research_fixture(tmp_path, hand_missing=50)
    with pytest.raises(TrainingDataError, match="all30 outputs"): admit(fixture)


def test_cached_authority_drift_is_rejected(researched):
    dataset = ResearchDerivedICTDataset(researched[0], "raw", "train", allow_synthetic_fixture=True)
    path = Path(researched[3]["implementation_ref"]["path"])
    path.write_bytes(path.read_bytes() + b"drift")
    with pytest.raises(TrainingDataError, match="authority changed"): dataset[0]


def test_preflight_import_has_no_torch_or_model():
    script = """
import importlib.abc,sys
class Deny(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path,target=None):
        if fullname == 'torch' or fullname.startswith('torch.') or fullname.endswith('.FlowMatchingModel'):
            raise AssertionError('unexpected model import')
sys.meta_path.insert(0,Deny())
from chaoyang.humanego_ict.research_data import validate_research_manifest
assert 'torch' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
