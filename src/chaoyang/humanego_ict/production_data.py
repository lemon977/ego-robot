"""Independent paired raw-replay admission, with unchanged causal/numeric gates.

This contract permits DERIVED_RETARGET research motion training, not REAL_CONTROL
or deployment. A mere receipt flag is insufficient: raw source and every H50
label are recomputed using the frozen full legacy pure mapping and its prior.
Preflight is NumPy-only; torch is lazy and RGB never enters the numeric cache.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np

from .data import (ACTION_LAYOUT, BRANCHES, TARGET, TrainingDataError, VerifiedInputCache,
                   _check_rgb_causal, _npz, _object, _ordinary, reference_bytes)
from .production_source import (SCHEMA as SOURCE_SCHEMA, SOURCE_PROFILE, _json,
    aggregate_group_plan, canonical_sha, fixed_anchor_ids, prepare_anchor, replay_session)

SCHEMA = "pick-place-humanego-ict-production-paired-v1"
TRAINING_SCOPE = "NON_DEPLOYABLE_DERIVED_MOTION_TRAINING"
SUPERVISION_CLASS = "DERIVED_RETARGET"
NORMALIZATION = {"method": "IDENTITY_NO_FIT", "position_mean": [0, 0, 0], "position_std": [1, 1, 1]}
SELECTION_POLICY = {"all_source_sessions": True, "ordering": "RAW_SOURCE_PATH_LEXICOGRAPHIC", "anchor_stride": 32,
    "horizon": 50, "split_policy": "ORDERED_CAPTURE_INDEX_MOD10_ZERO_VALIDATION", "quality_based_session_selection": False}


def validate_production_manifest(manifest_path, branch="raw", *, allow_synthetic_fixture=False, _input_cache=None):
    """Return a JSON-compatible summary; no torch, models or file publication.

The recorded-nanoseconds metadata trust boundary is stated, not silently
upgraded to verified QPC frequency/implementation or physical exposure timing.
"""
    if branch not in BRANCHES:
        raise TrainingDataError("unknown paired production branch")
    branch = BRANCHES[branch]
    cache = _input_cache or VerifiedInputCache()
    path = _ordinary(Path(manifest_path).absolute())
    before = VerifiedInputCache._stat(path); encoded = path.read_bytes()
    if before != VerifiedInputCache._stat(path):
        raise TrainingDataError("production manifest changed while reading")
    manifest = _object(encoded)
    ref = {"path": str(path), "bytes": len(encoded), "sha256": hashlib.sha256(encoded).hexdigest()}
    if manifest.get("fixture_only") is True and not allow_synthetic_fixture:
        raise TrainingDataError("synthetic production fixture is not real source evidence")
    if (manifest.get("schema") != SCHEMA or manifest.get("training_scope") != TRAINING_SCOPE
            or manifest.get("supervision_class") != SUPERVISION_CLASS or manifest.get("target") != TARGET
            or manifest.get("control_ground_truth") is not False or manifest.get("physical_deployment_authorized") is not False
            or manifest.get("action_layout") != ACTION_LAYOUT or manifest.get("normalization") != NORMALIZATION
            or manifest.get("source_profile") != SOURCE_PROFILE or manifest.get("selection_policy") != SELECTION_POLICY
            or not isinstance(manifest.get("pair_id"), str) or not manifest["pair_id"]):
        raise TrainingDataError("independent non-deployable production manifest contract differs")
    contract = _json(manifest["source_contract_ref"], cache)
    producer = _producer_identity(manifest, contract, cache, allow_synthetic_fixture)
    original = _json(contract["input_manifest_ref"], cache)
    records = original.get("sessions")
    if (not isinstance(records, list) or not records
            or (manifest.get("fixture_only") is not True and (len(records) != 199 or original.get("expected_sessions") != 199))):
        raise TrainingDataError("production must account for every frozen 199 capture sessions")
    expected_clock = {"position_unit": "meters", "pose_frame_id": "REP103_X_FORWARD_Y_LEFT_Z_UP",
        "camera_frame_id": "ANCHOR_PAST_HEAD_FACTORY_CAMERA_PRIOR", "tcp_frame_ids": ["LEFT_INSTALLATION_WRIST_PRIOR", "RIGHT_INSTALLATION_WRIST_PRIOR"],
        "timestamp_unit": "nanoseconds", "clock_domain": contract.get("clock_domain")}
    if manifest.get("pose_clock") != expected_clock:
        raise TrainingDataError("recorded-clock/frame/prior declaration differs")
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise TrainingDataError("production samples absent")
    samples_by_anchor, seen = {}, set()
    for sample in samples:
        identity = (sample.get("session_id"), sample.get("source_frame_id"))
        if (identity in samples_by_anchor or sample.get("sample_id") in seen
                or not isinstance(sample.get("sample_id"), str) or not sample["sample_id"]
                or type(sample.get("source_frame_id")) is not int):
            raise TrainingDataError("unique sample/native-anchor identities required")
        seen.add(sample["sample_id"]); samples_by_anchor[identity] = sample
    actual_metadata, candidates, timing, counts = [], [], [], {"train": 0, "validation": 0}
    valid_counts = {s: np.zeros(30, np.int64) for s in counts}
    admitted_ids, observed_groups = [], {s: set() for s in counts}
    deps = [manifest["source_contract_ref"], manifest["mapper_ref"], manifest["calibration_ref"], contract["input_manifest_ref"],
        manifest["producer_packet"], manifest["rgb_recipe_ref"], *producer["runtime_code"]]
    for record in sorted(records, key=lambda r: r["source_refs"]["source_manifest"]["path"]):
        replay = replay_session(record, mapper_ref=manifest["mapper_ref"], calibration_ref=manifest["calibration_ref"],
            source_contract=contract, allow_synthetic_fixture=allow_synthetic_fixture, _input_cache=cache)
        actual_metadata.append(replay["metadata"]); deps.extend(replay["actual_input_refs"])
        for frame in fixed_anchor_ids(replay):
            result = prepare_anchor(replay, frame)
            candidates.append({"session_id": record["session_id"], "source_frame_id": frame,
                "eligible": result["eligible"], "missing_items": result["missing_items"]})
            sample = samples_by_anchor.get((record["session_id"], frame))
            if not result["eligible"]:
                if sample is not None:
                    raise TrainingDataError("numerically failed raw H50 candidate cannot be admitted")
                continue
            if sample is None:
                raise TrainingDataError("eligible fixed-stride raw candidate was silently omitted")
            labels = _npz(sample["labels"], cache=cache)
            if set(labels) != set(result["labels"]):
                raise TrainingDataError("exact raw-recomputed label keys required; no digital q")
            for key, expected in result["labels"].items():
                actual = labels[key]
                if actual.shape != expected.shape or actual.dtype.kind != expected.dtype.kind:
                    raise TrainingDataError("production label shape/dtype mismatch: " + key)
                if expected.dtype.kind == "f":
                    matches = np.isfinite(actual).all() and np.allclose(actual, expected, atol=1e-5, rtol=1e-5)
                else:
                    matches = np.array_equal(actual, expected)
                if not matches:
                    raise TrainingDataError("production label differs from real raw replay: " + key)
            trace = _json(sample["derivation_trace_ref"], cache)
            if trace != result["trace"]:
                raise TrainingDataError("source provenance trace does not match actual replay")
            deps.extend([sample["labels"], sample["derivation_trace_ref"]])
            admission = sample.get("branch_admission", {"raw": True, "robotized": True})
            if not isinstance(admission, dict) or set(admission) != {"raw", "robotized"} or any(type(v) is not bool for v in admission.values()) or admission["raw"] is not True:
                raise TrainingDataError("explicit paired branch admission requires all eligible raw anchors")
            if not admission[branch]:
                # Robot render failure may not halt the ready Raw branch. The
                # failed branch stays represented and cannot train its pixels.
                if branch == "robotized" and not sample.get("robotized_missing_items"):
                    raise TrainingDataError("omitted robotized branch needs explicit failure evidence")
                continue
            _check_rgb_causal(sample, branch, labels, contract["clock_domain"], cache=cache, allow_synthetic_fixture=allow_synthetic_fixture)
            receipt = _json(sample["causal_receipts"][branch], cache)
            if (receipt.get("source_signature") != replay["metadata"]["source_signature"]
                    or receipt.get("source_session_id") != record["session_id"]
                    or not set(r["sha256"] for r in replay["actual_input_refs"]).issubset({r["sha256"] for r in receipt["evidence_refs"]})):
                raise TrainingDataError("causal RGB evidence does not bind this actual raw session replay")
            if branch == "robotized":
                _robotized_trace(sample, result, replay, receipt, producer, manifest, cache, allow_synthetic_fixture)
            reference_bytes(sample["rgb"][branch])
            split = sample.get("split")
            if split not in counts:
                raise TrainingDataError("production split must be train/validation")
            weight = sample.get("sample_weight", 1.0)
            if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not np.isfinite(weight) or weight <= 0:
                raise TrainingDataError("finite positive sample weight required")
            counts[split] += 1; valid_counts[split] += labels["action_valid_mask"].sum(axis=0)
            admitted_ids.append(sample["sample_id"]); observed_groups[split].add(sample["source_group_id"])
            timing.append((result["trace"]["current_ages_ns"], replay["quality"]["p95_ns"]))
            deps.append(sample["causal_receipts"][branch]); deps.extend(receipt["evidence_refs"])
    plan = aggregate_group_plan(actual_metadata)
    if manifest.get("group_plan") != plan or manifest.get("candidate_anchors") != candidates:
        raise TrainingDataError("actual independent capture proof/fixed candidate accounting differs")
    if any((s["session_id"], s["source_frame_id"]) not in {(c["session_id"], c["source_frame_id"]) for c in candidates if c["eligible"]} for s in samples):
        raise TrainingDataError("sample is not an eligible fixed source anchor")
    for sample in samples:
        selected = plan[sample["session_id"]]
        if sample["source_group_id"] != selected["source_group_id"] or sample["split"] != selected["split"]:
            raise TrainingDataError("sample group/split does not match original acquisition proof")
    if observed_groups["train"] & observed_groups["validation"]:
        raise TrainingDataError("independent source groups overlap train/validation")
    if any(not counts[s] or np.any(valid_counts[s] == 0) for s in counts):
        raise TrainingDataError("both splits require effective labels for all30 outputs")
    ages = np.asarray([v[0] for v in timing]); p95_limit = min(v[1] for v in timing)
    if np.any(np.percentile(ages, 95, axis=0) > p95_limit):
        raise TrainingDataError("aggregate past-conditioning exceeds original30ms p95 gate")
    unique_deps = {r["path"]: r for r in deps}
    stat_fingerprint = {p: list(VerifiedInputCache._stat(p)) for p in unique_deps}
    return {"dataset_sha": ref["sha256"], "prepared_pair_ref": ref, "prepared_pair": manifest,
        "normalization": NORMALIZATION, "pose_clock": expected_clock, "sample_counts": counts,
        "valid_action_counts": {s: a.tolist() for s, a in valid_counts.items()},
        "source_group_splits": {s: sorted(v) for s, v in observed_groups.items()}, "admitted_sample_ids": admitted_ids,
        "admitted_branch": branch, "manifest_metadata": list(before), "dependency_metadata": stat_fingerprint,
        "source_contract": contract, "source_metadata": actual_metadata, "training_scope": TRAINING_SCOPE,
        "physical_calibration_status": "UNKNOWN_NOT_GRANTED", "clock_trust_boundary": contract["clock_trust_boundary"],
        "clock_implementation_verification": "UNKNOWN_NOT_CLAIMED", "control_ground_truth": False,
        "physical_deployment_authorized": False}


validate_manifest = validate_production_manifest


def _producer_identity(manifest, contract, cache, allow_fixture):
    """Bind static historical producer identity without a live/sealing cycle.

The separate training task gate verifies the parent's terminal machine result.
This reader can also run inside the still-active actual production operation.
"""
    packet = _json(manifest["producer_packet"], cache)
    recipe = _json(manifest["rgb_recipe_ref"], cache)
    fixture = manifest.get("fixture_only") is True
    if packet.get("fixture_only") is True and not (fixture and allow_fixture):
        raise TrainingDataError("synthetic producer packet is not production identity")
    inputs = packet.get("inputs", {})
    if (packet.get("execution_mode") != "HUMANEGO_ICT_PRODUCTION_PREPARE_CPU"
            or packet.get("training_scope") != TRAINING_SCOPE or packet.get("run_signature") != manifest["pair_id"]
            or not isinstance(packet.get("task_id"), str) or not packet["task_id"]
            or inputs.get("source_contract") != manifest["source_contract_ref"]
            or inputs.get("resolved_input_manifest") != contract["input_manifest_ref"]
            or inputs.get("recipe") != manifest["rgb_recipe_ref"]
            or inputs.get("legacy_mapper_refs", [])[:2] != [manifest["mapper_ref"], manifest["calibration_ref"]]):
        raise TrainingDataError("producer packet identity/input/signature binding differs")
    if not fixture:
        repo = Path(__file__).resolve().parents[3]
        if (not Path(manifest["producer_packet"]["path"]).is_relative_to(repo / "tasks/current")
                or packet.get("packet_path") != manifest["producer_packet"]["path"]
                or not Path(manifest["samples"][0]["labels"]["path"]).is_relative_to(Path(packet["output_dir"]))):
            raise TrainingDataError("registered producer packet/output identity required")
    writer = packet.get("writer", {})
    if (writer.get("scope") != packet["task_id"] or any(type(writer.get(k)) is not int or writer[k] <= 0 for k in ("pid", "proc_start_ticks", "executor_epoch"))
            or not isinstance(writer.get("fencing_token"), str) or not writer["fencing_token"]
            or not isinstance(packet.get("deadline_at"), str) or not packet["deadline_at"]):
        raise TrainingDataError("producer historical fenced writer identity missing")
    closure = packet.get("runtime_code")
    if not isinstance(closure, list) or not closure:
        raise TrainingDataError("producer runtime source closure missing")
    for reference in closure:
        reference_bytes(reference, cache=cache)
    required = {str(Path(__file__).absolute()), str(Path(__file__).with_name("production_source.py").absolute()), str(Path(__file__).with_name("production_renderer.py").absolute())}
    if not required.issubset({r["path"] for r in closure}):
        raise TrainingDataError("producer closure does not bind actual source/reader/render modules")
    packet = dict(packet, _recipe_canonical_sha256=canonical_sha(recipe))
    return packet


def _robotized_trace(sample, result, replay, receipt, producer, manifest, cache, allow_fixture):
    """Check the actual current/past render trace, not only its receipt status."""
    trace_ref, diagnostic_ref = sample.get("robotized_render_trace_ref"), sample.get("robotized_render_diagnostics_ref")
    if trace_ref not in receipt["evidence_refs"] or diagnostic_ref not in receipt["evidence_refs"]:
        raise TrainingDataError("robotized RGB requires explicitly bound actual render trace/diagnostics")
    trace, diagnostic = _json(trace_ref, cache), _json(diagnostic_ref, cache)
    # The actual frozen renderer checks execution_mode before rendering; its
    # historical trace carries these four identity fields, not a new mode key.
    expected_authorization = {key: producer[key] for key in ("task_id", "run_signature", "writer", "deadline_at")}
    if (trace.get("producer_authorization") != expected_authorization
            or trace.get("recipe_sha256") != producer["_recipe_canonical_sha256"]
            or trace.get("synthetic_fixture") is not (manifest.get("fixture_only") is True)):
        raise TrainingDataError("robotized actual producer/recipe/fixture identity differs")
    current = result["current"]; anchor = current["anchor_qpc_ns"]
    expected_times = {"head": current["head_source_qpc_ns"],
        "controller_left": int(current["controller_source_qpc_ns"][0]), "controller_right": int(current["controller_source_qpc_ns"][1]),
        "manus_left": int(current["manus_source_qpc_ns"][0]), "manus_right": int(current["manus_source_qpc_ns"][1])}
    if (trace.get("session_id") != sample["session_id"] or trace.get("source_frame_id") != sample["source_frame_id"]
            or trace.get("anchor_qpc_ns") != anchor or trace.get("clock_domain") != current["clock_domain"]
            or trace.get("source_signature") != current["source_signature"] or trace.get("timestamp_unit") != "nanoseconds"
            or trace.get("sensor_source_qpc_ns") != expected_times or any(t > anchor for t in expected_times.values())
            or trace.get("donor_frame_ids") != [sample["source_frame_id"]]
            or trace.get("motion_inputs") != "CURRENT_NATIVE_PLUS_PRIOR_ACCEPTED_ANCHOR_SOLUTION"
            or trace.get("mask_inputs") != "CURRENT_NATIVE_PROJECTED_GEOMETRY_ONLY"
            or trace.get("clean_inputs") != "CURRENT_RAW_RGB_SPATIAL_ONLY_NO_TEMPORAL_DONORS"):
        raise TrainingDataError("robotized actual trace violates current/past-only replay")
    base = trace.get("base_anchor", {})
    if (type(base.get("frame_id")) is not int or type(base.get("anchor_qpc_ns")) is not int
            or not 0 <= base["frame_id"] <= current["frame_id"] or base["anchor_qpc_ns"] > anchor
            or base.get("source_signature") != current["source_signature"]
            or not isinstance(base.get("sensor_source_qpc_ns"), dict)
            or set(base["sensor_source_qpc_ns"]) != set(expected_times)
            or any(type(t) is not int or t < 0 or t > base["anchor_qpc_ns"] for t in base["sensor_source_qpc_ns"].values())):
        raise TrainingDataError("robotized base uses an unknown/future recording anchor")
    from PIL import Image
    with Image.open(io.BytesIO(reference_bytes(sample["rgb"]["raw"]))) as image:
        raw = np.asarray(image.convert("RGB"), np.uint8)
    if hashlib.sha256(raw.tobytes()).hexdigest() != trace.get("raw_rgb_array_sha256"):
        raise TrainingDataError("robotized actual trace does not bind paired raw pixels")
    _check_rgb_causal(sample, "raw", result["labels"], current["clock_domain"], cache=cache, allow_synthetic_fixture=allow_fixture)
    raw_receipt = _json(sample["causal_receipts"]["raw"], cache)
    if (raw_receipt.get("source_signature") != current["source_signature"]
            or raw_receipt.get("source_session_id") != sample["session_id"]
            or not {r["sha256"] for r in replay["actual_input_refs"]}.issubset({r["sha256"] for r in raw_receipt["evidence_refs"]})):
        raise TrainingDataError("robotized paired raw receipt is not the same actual source capture")
    tracking = diagnostic.get("motion", {}).get("per_side_tracking")
    if not isinstance(tracking, list) or len(tracking) != 2 or any(v.get("numeric_tracking_status") != "PASSED" for v in tracking):
        raise TrainingDataError("robotized numeric tracking FAIL cannot be promoted to admitted RGB")


class ProductionDerivedICTDataset:
    """Exact-index tensor reader, bounded numeric cache, no unlimited RGB cache.

All evidence is replayed on admission. Per-item stat checks reject mutation of
any frozen dependency; the same verified numeric labels are reused process-local
up to128MiB (clear via input_cache.clear()). RGB bytes are SHA-checked each read.
"""
    def __init__(self, manifest_path, branch, split, *, allow_synthetic_fixture=False):
        if split not in {"train", "validation"}:
            raise TrainingDataError("unknown production split")
        self.input_cache = VerifiedInputCache()
        admitted = validate_production_manifest(manifest_path, branch, allow_synthetic_fixture=allow_synthetic_fixture, _input_cache=self.input_cache)
        self.manifest_path = Path(manifest_path).absolute(); self.dataset_sha = admitted["dataset_sha"]
        self.manifest, self.branch, self.split = admitted["prepared_pair"], admitted["admitted_branch"], split
        self.manifest_metadata = admitted["manifest_metadata"]; self.dependencies = admitted["dependency_metadata"]
        ids = set(admitted["admitted_sample_ids"])
        self.samples = [s for s in self.manifest["samples"] if s["sample_id"] in ids and s["split"] == split]
        original = _json(admitted["source_contract"]["input_manifest_ref"], self.input_cache)
        self.session_refs = {r["session_id"]: list(r["source_refs"].values()) for r in original["sessions"]}
        self.shared_refs = [self.manifest["source_contract_ref"], self.manifest["mapper_ref"],
            self.manifest["calibration_ref"], admitted["source_contract"]["input_manifest_ref"], self.manifest["producer_packet"], self.manifest["rgb_recipe_ref"]]
        self.sample_evidence_refs = {s["sample_id"]: _json(s["causal_receipts"][self.branch], self.input_cache)["evidence_refs"] for s in self.samples}
        # Evict bulk JSONL from preflight and seal only the small numeric label
        # records; 600 H50 examples are about10MiB, below the128MiB LRU cap.
        self.input_cache.clear()
        for sample in self.samples:
            _npz(sample["labels"], cache=self.input_cache)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        if list(VerifiedInputCache._stat(self.manifest_path)) != self.manifest_metadata:
            raise TrainingDataError("immutable production manifest changed")
        sample = self.samples[index]
        refs = self.shared_refs + self.session_refs[sample["session_id"]] + self.sample_evidence_refs[sample["sample_id"]] + [sample["labels"], sample["derivation_trace_ref"], sample["causal_receipts"][self.branch]]
        for reference in {r["path"]: r for r in refs}.values():
            path = reference["path"]
            if list(VerifiedInputCache._stat(path)) != self.dependencies[path]:
                raise TrainingDataError("immutable production source/evidence dependency changed")
        labels = _npz(sample["labels"], cache=self.input_cache)
        from PIL import Image
        import torch
        try:
            with Image.open(io.BytesIO(reference_bytes(sample["rgb"][self.branch]))) as image:
                rgb = np.asarray(image.convert("RGB").resize((320, 240), Image.Resampling.BILINEAR), np.float32) / 255
        except (OSError, ValueError) as error:
            raise TrainingDataError("admitted production RGB cannot be decoded") from error
        mask = labels["action_valid_mask"]
        return {"x_rgb": torch.from_numpy(rgb.transpose(2, 0, 1).copy()), "x_ict": torch.from_numpy(labels["x_ict"].astype(np.float32)),
            "ict_mask": torch.from_numpy(labels["ict_mask"].copy()), "y_action": torch.from_numpy(np.where(mask, labels["y_action"], 0).astype(np.float32)),
            "action_valid_mask": torch.from_numpy(mask.copy()), "sample_weight": torch.tensor(sample.get("sample_weight", 1), dtype=torch.float32),
            "sample_id": sample["sample_id"], "session_id": sample["session_id"], "source_frame_id": sample["source_frame_id"],
            "source_group_id": sample["source_group_id"], "sample_index": index, "branch": self.branch, "split": self.split,
            "training_scope": TRAINING_SCOPE, "supervision_class": SUPERVISION_CLASS,
            "control_ground_truth": False, "physical_deployment_authorized": False}

    def verify_all_dependencies(self):
        """Optional epoch/batch boundary full-set metadata check, no bulk reads."""
        for path, before in self.dependencies.items():
            if list(VerifiedInputCache._stat(path)) != before:
                raise TrainingDataError("immutable production dependency changed")


ProductionICTDataset = ProductionDerivedICTDataset
