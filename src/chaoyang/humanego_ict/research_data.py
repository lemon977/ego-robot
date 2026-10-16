"""Independent non-deployable derived-motion training admission.

Physical base/TCP/FK calibration is not research supervision. Source replay,
clock evidence, original numeric timing gates, acquisition groups and causal RGB
remain mandatory. No old REAL_CONTROL or derived admission is modified.
"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any

import numpy as np

from .data import (ACTION_LAYOUT, BRANCHES, TrainingDataError, VerifiedInputCache,
                   _check_rgb_causal, _npz, _object, _ordinary, _receipt, reference_bytes)
from .derived_data import _checked_sample, _read_manifest, _source
from .source_adapter import adapt_native_source, _time_gate

SCHEMA = "pick-place-humanego-ict-derived-research-v1"
TRAINING_SCOPE = "NON_DEPLOYABLE_DERIVED_MOTION_TRAINING"
SUPERVISION_CLASS = "DERIVED_RETARGET"
COORDINATE_ASSUMPTIONS = {
    "authority": "DECLARED_INSTALLATION_PRIOR_AND_ANCHOR_NATIVE_C2W",
    "physical_calibration_status": "UNKNOWN_NOT_REQUIRED_FOR_RESEARCH",
    "metric_accuracy_claimed": False,
}


def _clock(manifest, pair, cache, allow_fixture):
    receipt = _receipt(manifest["clock_proof_ref"], "capture-host-qpc-nanoseconds-proof-v1",
        cache=cache, allow_synthetic_fixture=allow_fixture,
        timestamp_unit="nanoseconds", clock_domain=pair["pose_clock"]["clock_domain"],
        clock_value_unit="nanoseconds", implementation_kind="HOST_MONOTONIC_NANOSECONDS",
        qualification="VERIFIED_NS_SOURCE_AND_SINGLE_CAPTURE_HOST_DOMAIN",
        clock_semantics="HOST_ARRIVAL_RECEIVE_TIME_NOT_EXPOSURE_SYNCHRONIZATION")
    if (not isinstance(receipt.get("capture_host_id"), str) or not receipt["capture_host_id"].strip()
            or type(receipt.get("units_per_second")) is not int or receipt["units_per_second"] != 1_000_000_000
            or receipt.get("implementation_ref") not in receipt["evidence_refs"]):
        raise TrainingDataError("research requires actual clock implementation evidence, not UNKNOWN/nominal FPS")
    reference_bytes(receipt["implementation_ref"], cache=cache)
    bindings = receipt.get("session_bindings")
    if not isinstance(bindings, list) or not bindings:
        raise TrainingDataError("clock proof must bind the actual capture sessions and source bytes")
    seen = set()
    for binding in bindings:
        if (not isinstance(binding, dict) or not isinstance(binding.get("session_id"), str)
                or binding["session_id"] in seen or binding.get("capture_host_id") != receipt["capture_host_id"]
                or binding.get("clock_domain") != receipt["clock_domain"]
                or binding.get("timestamp_unit") != "nanoseconds"
                or binding.get("timestamp_roles") != ["video_qpc_ns", "controller_source_qpc_ns", "hand_source_qpc_ns"]):
            raise TrainingDataError("clock proof capture-host/domain/unit/source-role binding differs")
        seen.add(binding["session_id"])
        for key in ("native_source_ref", "session_input_ref"):
            if binding.get(key) not in receipt["evidence_refs"]:
                raise TrainingDataError("clock proof actual source refs must be immutable evidence")
            reference_bytes(binding[key], cache=cache)
    if seen != {sample["session_id"] for sample in pair["samples"]}:
        raise TrainingDataError("clock proof must cover exactly the prepared acquisition sessions")
    return receipt


def _timing(sample, pair, manifest, cache, allow_fixture):
    proof = _receipt(manifest["sample_time_receipts"][sample["sample_id"]],
        "humanego-ict-derived-research-time-v1", cache=cache, allow_synthetic_fixture=allow_fixture,
        sample_id=sample["sample_id"], session_id=sample["session_id"],
        labels_sha256=sample["labels"]["sha256"], source_sha256=sample["human_source_sidecar"]["sha256"],
        clock_proof_sha256=manifest["clock_proof_ref"]["sha256"],
        timestamp_unit="nanoseconds", clock_domain=pair["pose_clock"]["clock_domain"])
    required = ("native_source_ref", "session_input_ref", "adapter_contract_ref", "adapter_trace_ref")
    for key in required:
        if proof.get(key) not in proof["evidence_refs"]:
            raise TrainingDataError("research source replay requires actual immutable " + key)
        reference_bytes(proof[key], cache=cache)
    clock = _clock(manifest, pair, cache, allow_fixture)
    binding = next(item for item in clock["session_bindings"] if item["session_id"] == sample["session_id"])
    if any(binding[key] != proof[key] for key in ("native_source_ref", "session_input_ref")):
        raise TrainingDataError("clock proof does not cover this actual native/session source")
    adapter = _object(reference_bytes(proof["adapter_contract_ref"], cache=cache))
    session = _object(reference_bytes(proof["session_input_ref"], cache=cache))
    trace = _object(reference_bytes(proof["adapter_trace_ref"], cache=cache))
    source = _source(sample["human_source_sidecar"], cache)
    labels = _npz(sample["labels"], cache=cache)
    if (adapter.get("pose_clock") != pair["pose_clock"]
            or trace.get("pose_clock") != pair["pose_clock"]
            or trace.get("session_id") != sample["session_id"]
            or trace.get("anchor_qpc_ns") != int(labels["anchor_qpc_ns"])):
        raise TrainingDataError("research replay frame/unit/clock/anchor binding differs")
    kind = "research-replay:" + ":".join(proof[key]["sha256"] for key in required)
    replay = cache.get(kind, sample["human_source_sidecar"])
    if replay is None:
        replay = adapt_native_source(proof["native_source_ref"], proof["session_input_ref"],
            proof["adapter_contract_ref"], anchor_row=trace.get("native_anchor_row"),
            allow_synthetic_fixture=allow_fixture)
        cache.put(kind, sample["human_source_sidecar"], replay,
                  sum(value.nbytes for value in replay["source_arrays"].values()))
    if replay["trace"] != trace:
        raise TrainingDataError("research source trace differs from actual native replay")
    for key, expected in replay["source_arrays"].items():
        actual = source[key]
        if (actual.shape != expected.shape or actual.dtype.kind != expected.dtype.kind
                or (not np.allclose(actual, expected, atol=1e-8, rtol=1e-8, equal_nan=True)
                    if expected.dtype.kind == "f" else not np.array_equal(actual, expected))):
            raise TrainingDataError("research sidecar differs from actual native replay: " + key)
    # These preparation UNKNOWNs are resolved only by the independently checked
    # consumer below. No physical qualification is promoted or overwritten.
    allowed_unknowns = {
        "BLOCKED_CAMERA_EYE_IDENTITY_UNQUALIFIED", "BLOCKED_PRODUCER_CALIBRATION_NOT_TRAINING_QUALIFIED",
        "UNKNOWN_SOURCE_TIMEBASE_QUALIFICATION", "SELECTED_CURRENT_AGGREGATE_TIME_QUALITY_NOT_QUALIFIED",
        "INDEPENDENT_SOURCE_GROUP_NOT_ESTABLISHED", "RGB_CAUSAL_ADMISSION_NOT_GRANTED",
    }
    if any(item.startswith("BLOCKED_") and item not in allowed_unknowns for item in replay["missing_items"]):
        raise TrainingDataError("research native source retains a numerical FAIL: " + str(replay["missing_items"]))
    maximum, original = _time_gate(adapter, session)
    # These are the existing capture gates, not a fitted/new research threshold.
    # A rehashed HDF claiming looser values cannot redefine this route's gate.
    if (original["gate_ms"] > 40 or original["max_skew_ms"] > 40
            or original["max_p95_skew_ms"] > 30):
        raise TrainingDataError("original 40ms/30ms timing gates must not be relaxed")
    import h5py
    with h5py.File(io.BytesIO(reference_bytes(adapter["time_gate_source_ref"], cache=cache)), "r") as hdf:
        try:
            minimum_hand = float(hdf.attrs["min_hand_coverage"])
            minimum_complete = float(hdf.attrs["min_complete_coverage"])
        except (KeyError, TypeError, ValueError) as error:
            raise TrainingDataError("original HDF coverage gates absent") from error
    if not (.95 <= minimum_hand <= 1 and .95 <= minimum_complete <= 1):
        raise TrainingDataError("original HDF 0.95 coverage gates invalid or relaxed")
    rows = labels["target_source_rows"]
    current = int(labels["current_source_row"])
    if current != 0 or not np.array_equal(rows, np.arange(1, 51)):
        raise TrainingDataError("research source adapter requires anchor-zero contiguous H50 window")
    if not labels["ict_mask"].any():
        raise TrainingDataError("research current ICT absent")
    if (np.mean(source["controller_valid"][rows].all(axis=1) & source["hand_joints_valid"][rows].all(axis=1))
            < minimum_complete or np.any(source["hand_joints_valid"][rows].mean(axis=0) < minimum_hand)):
        raise TrainingDataError("research H50 fails original source coverage gates")
    p95_limit = original["max_p95_skew_ms"] * 1_000_000
    stats, current_skews = {}, {}
    for kind, valid_key, time_key in (
        ("controller", "controller_valid", "controller_source_qpc_ns"),
        ("hand", "hand_joints_valid", "hand_source_qpc_ns"),
    ):
        valid, times = source[valid_key], source[time_key]
        for hand in range(2):
            selected = rows[valid[rows, hand]]
            skew = np.abs(times[selected, hand] - source["qpc_ns"][selected])
            if not len(skew) or np.max(skew) > maximum or np.percentile(skew, 95) > p95_limit:
                raise TrainingDataError("research target fails original max/p95 timing gates")
            name = f"{kind}_{hand}"
            stats[name] = {"max_abs_ns": int(np.max(skew)), "p95_abs_ns": float(np.percentile(skew, 95))}
            if valid[current, hand]:
                age = int(source["qpc_ns"][current] - times[current, hand])
                if not 0 <= age <= maximum:
                    raise TrainingDataError("research current conditioning is future or outside original time gate")
                current_skews[name] = age
    return {"sample_id": sample["sample_id"], "target_timing": stats, "current_skews_ns": current_skews,
            "max_p95_skew_ns": p95_limit, "original_time_gates": original,
            "physical_qualification": "UNKNOWN_NOT_GRANTED", "native_preparation_missing_items": replay["missing_items"]}


def validate_research_manifest(manifest_path: str | Path, branch: str, *,
        allow_synthetic_fixture: bool = False, _input_cache: VerifiedInputCache | None = None) -> dict[str, Any]:
    """Source-recomputed research admission; task/GPU/deployment authority is separate."""
    if branch not in BRANCHES:
        raise TrainingDataError("unknown research RGB branch")
    cache = _input_cache if _input_cache is not None else VerifiedInputCache()
    path = _ordinary(Path(manifest_path).absolute())
    before = VerifiedInputCache._stat(path)
    encoded = path.read_bytes()
    if before != VerifiedInputCache._stat(path):
        raise TrainingDataError("research manifest changed during validation")
    manifest = _object(encoded)
    if manifest.get("fixture_only") is True and not allow_synthetic_fixture:
        raise TrainingDataError("synthetic research manifest is not production evidence")
    if (manifest.get("schema") != SCHEMA or manifest.get("training_scope") != TRAINING_SCOPE
            or manifest.get("supervision_class") != SUPERVISION_CLASS
            or manifest.get("control_ground_truth") is not False
            or manifest.get("physical_deployment_authorized") is not False
            or manifest.get("coordinate_assumptions") != COORDINATE_ASSUMPTIONS):
        raise TrainingDataError("separate non-deployable research-derived contract required")
    reference_bytes(manifest["prepared_pair_ref"], cache=cache)
    pair, numeric = _read_manifest(manifest["prepared_pair_ref"]["path"], cache, allow_synthetic_fixture)
    if numeric["dataset_sha"] != manifest["prepared_pair_ref"]["sha256"]:
        raise TrainingDataError("prepared pair changed during research validation")
    _clock(manifest, pair, cache, allow_synthetic_fixture)
    samples = pair["samples"]
    if not isinstance(manifest.get("sample_time_receipts"), dict) or set(manifest["sample_time_receipts"]) != {s["sample_id"] for s in samples}:
        raise TrainingDataError("research time evidence must cover exactly the paired samples")
    for split, counts in numeric["valid_action_counts"].items():
        if not numeric["sample_counts"][split] or any(value == 0 for value in counts):
            raise TrainingDataError("research requires effective supervision of all30 outputs in both splits")
    if any(item.startswith("BLOCKED_") for item in numeric["missing_items"]):
        raise TrainingDataError("research labels retain unresolved numeric/retarget source blocker")
    timing = []
    for sample in samples:
        if sample["source_group_id"] == "UNKNOWN_SOURCE_GROUP":
            raise TrainingDataError("research requires independent acquisition source-group proof")
        _receipt(sample["source_group_receipt"], "independent-acquisition-source-group-v1",
            cache=cache, allow_synthetic_fixture=allow_synthetic_fixture,
            session_id=sample["session_id"], source_group_id=sample["source_group_id"], independent_acquisition=True)
        labels = _npz(sample["labels"], cache=cache)
        _check_rgb_causal(sample, BRANCHES[branch], labels, pair["pose_clock"]["clock_domain"],
                          cache=cache, allow_synthetic_fixture=allow_synthetic_fixture)
        timing.append(_timing(sample, pair, manifest, cache, allow_synthetic_fixture))
    for name in ("controller_0", "controller_1", "hand_0", "hand_1"):
        ages = [item["current_skews_ns"][name] for item in timing if name in item["current_skews_ns"]]
        if ages and np.percentile(ages, 95) > min(item["max_p95_skew_ns"] for item in timing):
            raise TrainingDataError("research aggregate current timing exceeds original p95 gate")
    return {"manifest": manifest, "prepared_pair": pair, "prepared_pair_ref": manifest["prepared_pair_ref"],
            "dataset_sha": hashlib.sha256(encoded).hexdigest(), "manifest_metadata": before,
            "pair_metadata": numeric["manifest_metadata"], "sample_counts": numeric["sample_counts"],
            "source_group_splits": numeric["source_group_splits"], "valid_action_counts": numeric["valid_action_counts"],
            "timing": timing, "training_scope": TRAINING_SCOPE, "supervision_class": SUPERVISION_CLASS,
            "admitted_branch": BRANCHES[branch], "control_ground_truth": False,
            "physical_deployment_authorized": False, "training_start_authorized": False,
            "physical_qualification": "NOT_CHECKED_NOT_GRANTED_BY_RESEARCH_ADMISSION"}


class ResearchDerivedICTDataset:
    """Deterministic paired tensor API with bounded immutable evidence cache."""
    def __init__(self, manifest_path, branch, split, *, allow_synthetic_fixture=False):
        if split not in {"train", "validation"}:
            raise TrainingDataError("research split must be train or validation")
        self.input_cache = VerifiedInputCache()
        admitted = validate_research_manifest(manifest_path, branch,
            allow_synthetic_fixture=allow_synthetic_fixture, _input_cache=self.input_cache)
        self.manifest_path = Path(manifest_path).absolute()
        self.manifest_metadata = admitted["manifest_metadata"]
        self.pair_metadata = admitted["pair_metadata"]
        self.manifest, self.pair = admitted["manifest"], admitted["prepared_pair"]
        self.dataset_sha = admitted["dataset_sha"]
        self.branch, self.split = admitted["admitted_branch"], split
        self.samples = [sample for sample in self.pair["samples"] if sample["split"] == split]
        self.allow_fixture = allow_synthetic_fixture

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        if (VerifiedInputCache._stat(self.manifest_path) != self.manifest_metadata
                or VerifiedInputCache._stat(self.manifest["prepared_pair_ref"]["path"]) != self.pair_metadata):
            raise TrainingDataError("immutable research manifest/pair authority changed")
        sample = self.samples[index]
        _clock(self.manifest, self.pair, self.input_cache, self.allow_fixture)
        _timing(sample, self.pair, self.manifest, self.input_cache, self.allow_fixture)
        labels, _ = _checked_sample(sample, self.pair, self.input_cache, self.allow_fixture)
        _receipt(sample["source_group_receipt"], "independent-acquisition-source-group-v1",
            cache=self.input_cache, allow_synthetic_fixture=self.allow_fixture,
            session_id=sample["session_id"], source_group_id=sample["source_group_id"], independent_acquisition=True)
        _check_rgb_causal(sample, self.branch, labels, self.pair["pose_clock"]["clock_domain"],
                          cache=self.input_cache, allow_synthetic_fixture=self.allow_fixture)
        from PIL import Image
        import torch
        try:
            with Image.open(io.BytesIO(reference_bytes(sample["rgb"][self.branch]))) as image:
                rgb = np.asarray(image.convert("RGB").resize((320, 240), Image.Resampling.BILINEAR), dtype=np.float32) / 255
        except (OSError, ValueError) as error:
            raise TrainingDataError("research admitted RGB cannot be decoded") from error
        mask = labels["action_valid_mask"]
        return {"x_rgb": torch.from_numpy(rgb.transpose(2, 0, 1).copy()),
            "x_ict": torch.from_numpy(labels["x_ict"].astype(np.float32)),
            "ict_mask": torch.from_numpy(labels["ict_mask"].copy()),
            "y_action": torch.from_numpy(np.where(mask, labels["y_action"], 0).astype(np.float32)),
            "action_valid_mask": torch.from_numpy(mask.copy()),
            "sample_weight": torch.tensor(sample.get("sample_weight", 1.0), dtype=torch.float32),
            "sample_id": sample["sample_id"], "session_id": sample["session_id"],
            "source_frame_id": sample["source_frame_id"], "source_group_id": sample["source_group_id"],
            "branch": self.branch, "split": self.split, "sample_index": index,
            "training_scope": TRAINING_SCOPE, "supervision_class": SUPERVISION_CLASS,
            "control_ground_truth": False, "physical_deployment_authorized": False}
