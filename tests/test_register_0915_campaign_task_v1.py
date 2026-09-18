from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from chaoyang.governance import register_0915_campaign_task_v1 as subject


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _depth_closure(tmp_path: Path, monkeypatch) -> Path:
    root = tmp_path / "attempt_0001"
    root.mkdir()
    schemas = {
        "calibration": "0915-foundationstereo-frozen-calibration-v1",
        "registration": "0915-foundationstereo-registration-v1",
        "depth_contract": "0915-foundationstereo-depth-contract-v1",
        "depth_summary": "0915-foundationstereo-depth-summary-v1",
        "worker_result": "0915-foundationstereo-worker-result-v1",
        "terminal_result": "0915-foundationstereo-single-session-canary-result-v1",
    }
    runtime = tmp_path / "RUNTIME.json"
    _write_json(runtime, {
        "schema_version": "foundationstereo-runtime-closure-v1",
        "status": "PASS_BOUNDED_CANARY_RUNTIME_IDENTITY_CLOSED",
        "task_id": subject.FOUNDATION_TASK_ID,
        "execution_performed": False,
        "gpu_used": False,
        "batch_authorized": False,
        "external_accuracy": "UNVERIFIED",
        "consumer_policy": {
            "quality_pass_required": True,
            "authorized_scopes": [subject.FOUNDATION_OBJECT6D_SCOPE],
            "all_other_consumers_authorized": False,
        },
        "output_schema_identities": schemas,
    })
    monkeypatch.setattr(subject, "FOUNDATION_RUNTIME_CLOSURE", runtime)
    packet = tmp_path / "TASK_PACKET.json"
    _write_json(packet, {"task_id": subject.FOUNDATION_TASK_ID})
    selected = "SAME_SESSION_INTRINSICS_HELDOUT_ESTIMATED_RECTIFIED"
    frozen_calibration = {"baseline_m": 0.08, "left": {"fx": 300.0}}
    t0 = tmp_path / "STEREO_PREFLIGHT.json"
    _write_json(t0, {
        "status": "PASS_GPU_DEPTH_ADMISSION",
        "session_id": subject.FOUNDATION_SESSION_ID,
        "decision": {"gpu_depth_allowed": True, "selected_candidate": selected},
        "candidates": {"calibrated_rectified": {
            "gpu_eligible": True,
            "quality": {"passed": True},
            "calibration": frozen_calibration,
        }},
    })
    t0_ref = subject.artifact_ref(t0)
    _write_json(root / "CALIBRATION.json", {
        "schema_version": "0915-foundationstereo-frozen-calibration-v1",
        "source": t0_ref,
        "consumption": "EXACT_T0_ACCEPTED_CALIBRATION_NO_REESTIMATION",
        "selected_domain": selected,
        "calibration": frozen_calibration,
    })
    (root / "REGISTRATION_MAPS.npz").write_bytes(b"npz-map-fixture")
    maps_ref = subject.artifact_ref(root / "REGISTRATION_MAPS.npz")
    _write_json(root / "REGISTRATION.json", {
        "schema_version": "0915-foundationstereo-registration-v1",
        "source_domain": "PHYSICAL_LEFT_RECTIFIED_640x480_PIXEL_CENTERS",
        "depth_reference": "PHYSICAL_LEFT_RECTIFIED_OPTICAL_Z",
        "nonlinear_registration_maps": maps_ref,
        "depth_to_sam_resize_map": {
            "array": "depth_to_sam_resize_xy",
            "source": "PHYSICAL_LEFT_RECTIFIED_640x480_PIXEL_CENTERS",
            "target": "PHYSICAL_LEFT_SOURCEINDEX_1_RESIZE_ONLY_1280x960_PIXEL_CENTERS",
            "identity_assumed": False,
        },
        "mask_join_policy": "IDENTITY_JOIN_FORBIDDEN",
    })
    _write_json(root / "DEPTH_CONTRACT.json", {
        "schema_version": "0915-foundationstereo-depth-contract-v1",
        "task_id": subject.FOUNDATION_TASK_ID,
        "session_id": subject.FOUNDATION_SESSION_ID,
        "frame_count": 150,
        "frame_geometry": [640, 480],
        "frame_arrays": {"depth_reference": "PHYSICAL_LEFT_RECTIFIED_OPTICAL_Z"},
        "native_model_confidence": "ABSENT_NOT_FABRICATED",
        "occluded_or_hidden_geometry": "INVALID_NOT_COMPLETED",
        "external_accuracy": "UNVERIFIED",
        "consumption_authorized": True,
        "authorized_scopes": [subject.FOUNDATION_OBJECT6D_SCOPE],
    })
    _write_json(root / "DEPTH_SUMMARY.json", {
        "schema_version": "0915-foundationstereo-depth-summary-v1",
        "task_id": subject.FOUNDATION_TASK_ID,
        "session_id": subject.FOUNDATION_SESSION_ID,
        "status": "PASS",
        "depth_admission": "PASS",
        "external_accuracy": "UNVERIFIED",
        "consumption_authorized": True,
        "authorized_scopes": [subject.FOUNDATION_OBJECT6D_SCOPE],
        "frame_count": 150,
        "full_sbs_decode": True,
        "model_load_count": 1,
        "model_inference_count": 300,
        "frames": [{"frame": index} for index in range(150)],
        "quality": {
            "passed": True,
            "gates": {
                "full_decode": True,
                "formula_recompute": True,
                "geometric_validity": True,
                "lr_testable_coverage": True,
                "lr_consistency": True,
                "final_validity": True,
                "temporal_distribution_stability": True,
                "rgb_edge_support": True,
            },
        },
        "review_decode": {"full_decode": True, "frame_count": 150},
        "source_mutated": False,
    })
    unsigned = {
        "schema_version": "0915-foundationstereo-run-signature-v1",
        "task_id": subject.FOUNDATION_TASK_ID,
        "session_id": subject.FOUNDATION_SESSION_ID,
        "executor_epoch": 4,
        "weights": [subject.FOUNDATION_WEIGHT],
        "input_manifest": {
            "task_packet": subject.artifact_ref(packet),
            "t0_stereo_preflight": t0_ref,
        },
        "runtime": {"runtime_closure_receipt": subject.artifact_ref(runtime)},
        "calibration_identity": {
            "selected_domain": selected,
            "calibration_sha256": subject._canonical_sha256(frozen_calibration),
            "consumption": "EXACT_T0_ACCEPTED_CALIBRATION_NO_REESTIMATION",
        },
        "schema_identity": schemas,
    }
    signature_digest = subject._canonical_sha256(unsigned)
    _write_json(root / "RUN_SIGNATURE.json", {
        **unsigned, "run_signature_sha256": signature_digest,
    })
    _write_json(root / "CLAIM.json", {
        "schema_version": "0915-foundationstereo-writer-claim-v1",
        "task_id": subject.FOUNDATION_TASK_ID,
        "session_id": subject.FOUNDATION_SESSION_ID,
        "attempt_id": root.name,
        "weights": [subject.FOUNDATION_WEIGHT],
        "status": "CLAIMED",
        "pid": 123,
        "proc_start_ticks": 456,
        "executor_epoch": 4,
        "fencing_token_sha256": hashlib.sha256(b"fixture-fence").hexdigest(),
        "run_signature_sha256": signature_digest,
        "unique_write_root": str(root.resolve()),
        "task_packet": subject.artifact_ref(packet),
    })
    _write_json(root / "GPU_COMMAND_RECEIPT.json", {
        "schema_version": "v71-gpu-command-receipt-v1",
        "status": "PASSED",
        "task_id": subject.FOUNDATION_TASK_ID,
        "attempt_id": root.name,
        "returncode": 0,
        "error": None,
    })
    refs = {
        "calibration": subject.artifact_ref(root / "CALIBRATION.json"),
        "registration": subject.artifact_ref(root / "REGISTRATION.json"),
        "registration_maps": maps_ref,
        "depth_contract": subject.artifact_ref(root / "DEPTH_CONTRACT.json"),
        "depth_summary": subject.artifact_ref(root / "DEPTH_SUMMARY.json"),
        "gpu_command_receipt": subject.artifact_ref(root / "GPU_COMMAND_RECEIPT.json"),
        "run_signature": subject.artifact_ref(root / "RUN_SIGNATURE.json"),
        "writer_claim": subject.artifact_ref(root / "CLAIM.json"),
    }
    _write_json(root / "DEPTH_WORKER_RESULT.json", {
        "schema_version": "0915-foundationstereo-worker-result-v1",
        "status": "COMPLETED",
        "task_id": subject.FOUNDATION_TASK_ID,
        "session_id": subject.FOUNDATION_SESSION_ID,
        "external_accuracy": "UNVERIFIED",
        "consumption_authorized": True,
        "authorized_scopes": [subject.FOUNDATION_OBJECT6D_SCOPE],
        "run_signature": refs["run_signature"],
        "writer_claim": refs["writer_claim"],
        "calibration": refs["calibration"],
        "registration": refs["registration"],
        "registration_maps": refs["registration_maps"],
        "depth_contract": refs["depth_contract"],
        "depth_summary": refs["depth_summary"],
        "access_contract": {"source_mutated": False},
    })
    result_path = root / "RESULT.json"
    _write_json(result_path, {
        "schema_version": "0915-foundationstereo-single-session-canary-result-v1",
        "task_id": subject.FOUNDATION_TASK_ID,
        "session_id": subject.FOUNDATION_SESSION_ID,
        "status": "PASSED",
        "depth_admission": "PASS",
        "external_accuracy": "UNVERIFIED",
        "consumption_authorized": True,
        "authorized_scopes": [subject.FOUNDATION_OBJECT6D_SCOPE],
        "weights": [subject.FOUNDATION_WEIGHT],
        "source_mutated": False,
        **refs,
    })
    return result_path


def _state(predecessor: str, status: str = "PASSED") -> dict:
    return {
        "next_task": None,
        "tasks": [{
            "task_id": predecessor,
            "status": status,
            "result": {"path": "/nonexistent", "bytes": 1, "sha256": "0" * 64},
        }],
    }


def test_later_stage_requires_immediate_predecessor_pass() -> None:
    state = _state("0915_foundationstereo_full_v1")
    predecessor = subject._validate_predecessor(state, "0915_post_geometry_robot_v1")
    assert predecessor["task_id"] == "0915_foundationstereo_full_v1"
    bad = copy.deepcopy(state)
    bad["tasks"][0]["status"] = "FAILED_RUNTIME_FINAL"
    with pytest.raises(RuntimeError, match="did not pass"):
        subject._validate_predecessor(bad, "0915_post_geometry_robot_v1")


def test_registration_rejects_any_live_task() -> None:
    state = _state("0915_hawor_full_v1")
    state["tasks"].append({"task_id": "other", "status": "RUNNING"})
    with pytest.raises(RuntimeError, match="no active"):
        subject._validate_predecessor(state, "0915_sam31_mask_full_v1")


def test_registration_rejects_duplicate_task() -> None:
    state = _state("0915_hawor_full_v1")
    state["tasks"].append({"task_id": "0915_sam31_mask_full_v1", "status": "PASSED"})
    with pytest.raises(RuntimeError, match="already registered"):
        subject._validate_predecessor(state, "0915_sam31_mask_full_v1")


def test_registration_requires_materialized_read_closure_and_weight(tmp_path: Path) -> None:
    (tmp_path / "runner.py").write_text("# ready\n", encoding="utf-8")
    packet = {
        "read_set": ["runner.py", "missing.schema.json"],
        "weights": ["model.pt"],
    }
    errors = subject._validate_materialized_read_closure(packet, repo_root=tmp_path)
    assert errors == [
        "read_set path is not materialized: missing.schema.json",
        "weight is not materialized: model.pt",
    ]
    (tmp_path / "missing.schema.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "model.pt").write_bytes(b"pinned")
    assert subject._validate_materialized_read_closure(
        packet, repo_root=tmp_path,
    ) == []


def test_corrective_v2_accepts_only_exact_pre_execution_cli_failure(
    tmp_path: Path, monkeypatch,
) -> None:
    result_path = tmp_path / "RESULT.json"
    result_path.write_text(json.dumps({
        "returncodes": {"audit": 1},
        "self_containment": None,
        "prepared_manifest": None,
    }))
    (tmp_path / "AUDIT.log").write_text(
        "operation is not maintained by the current algorithm contract: "
        "audit_0915_processed_self_containment_v2\n"
    )
    state = _state("0915_input_prepare_cad_v1", "FAILED_RUNTIME_FINAL")
    state["tasks"][0]["result"]["path"] = str(result_path)
    monkeypatch.setattr(subject, "validate_artifact_ref", lambda _value: [])
    predecessor = subject._validate_predecessor(
        state, "0915_input_prepare_cad_v2",
    )
    assert predecessor["task_id"] == "0915_input_prepare_cad_v1"

    (tmp_path / "AUDIT.log").write_text("different failure\n")
    with pytest.raises(RuntimeError, match="exact pre-execution"):
        subject._validate_predecessor(state, "0915_input_prepare_cad_v2")


def test_vst_research_requires_exact_cancelled_image_domain_hold(
    tmp_path: Path, monkeypatch,
) -> None:
    result_path = tmp_path / "RESULT.json"
    result_path.write_text(json.dumps({
        "task_id": "0915_hawor_full_v1",
        "status": "CANCELLED",
        "first_blocker": "VST_IMAGE_DOMAIN_UNRESOLVED_POSSIBLE_REDUNDANT_UNDISTORTION",
    }))
    state = _state("0915_hawor_full_v1", "CANCELLED")
    state["tasks"][0]["result"]["path"] = str(result_path)
    monkeypatch.setattr(subject, "validate_artifact_ref", lambda _value: [])
    predecessor = subject._validate_predecessor(
        state, "0915_vst_image_domain_ab_v1",
    )
    assert predecessor["task_id"] == "0915_hawor_full_v1"

    bad = copy.deepcopy(state)
    bad["tasks"][0]["status"] = "PASSED"
    with pytest.raises(RuntimeError, match="requires the HaWoR task to be cancelled"):
        subject._validate_predecessor(bad, "0915_vst_image_domain_ab_v1")


def test_sam_full_batch_registration_is_fail_closed_after_bounded_canaries() -> None:
    state = _state("0915_planar_object6d_single_session_canary_v1")
    with pytest.raises(RuntimeError, match="separate batch authorization"):
        subject._validate_predecessor(state, "0915_sam31_mask_full_v1")


def test_wrong_domain_foundationstereo_task_cannot_be_registered_again() -> None:
    state = _state("0915_removal_envelope_single_session_canary_v1")
    with pytest.raises(RuntimeError, match="zero lens-undistortion"):
        subject._validate_predecessor(
            state,
            "0915_foundationstereo_single_session_canary_v1",
        )


def test_object6d_registration_requires_explicit_depth_consumption_scope(
    tmp_path: Path, monkeypatch,
) -> None:
    result_path = _depth_closure(tmp_path, monkeypatch)
    state = _state("0915_foundationstereo_single_session_canary_v1")
    state["tasks"][0]["result"] = subject.artifact_ref(result_path)
    predecessor = subject._validate_predecessor(
        state, "0915_planar_object6d_single_session_canary_v1",
    )
    assert predecessor["task_id"] == "0915_foundationstereo_single_session_canary_v1"

    base = json.loads(result_path.read_text(encoding="utf-8"))
    base["consumption_authorized"] = False
    base["authorized_scopes"] = []
    _write_json(result_path, base)
    state["tasks"][0]["result"] = subject.artifact_ref(result_path)
    with pytest.raises(RuntimeError, match="terminal authority"):
        subject._validate_predecessor(
            state, "0915_planar_object6d_single_session_canary_v1",
        )


def test_object6d_depth_admission_rejects_worker_gpu_or_summary_drift(
    tmp_path: Path, monkeypatch,
) -> None:
    result_path = _depth_closure(tmp_path, monkeypatch)
    admitted = subject.validate_foundationstereo_object6d_admission(result_path)
    assert admitted["summary"]["model_inference_count"] == 300

    gpu_path = result_path.parent / "GPU_COMMAND_RECEIPT.json"
    gpu = json.loads(gpu_path.read_text(encoding="utf-8"))
    gpu["returncode"] = 1
    _write_json(gpu_path, gpu)
    with pytest.raises(RuntimeError, match="gpu_command_receipt artifact drift"):
        subject.validate_foundationstereo_object6d_admission(result_path)


def test_removal_envelope_requires_visual_rejection_and_exact_authorization(
    tmp_path: Path, monkeypatch,
) -> None:
    weak_path = tmp_path / "weak.json"
    weak_path.write_text(json.dumps({
        "session_id": "play_cards_0915_001",
        "session_admission": "AWAITING_USER_VISUAL_REVIEW",
    }))
    state = _state("0915_sam31_weak_role_canary_v1")
    state["tasks"][0]["result"]["path"] = str(weak_path)
    monkeypatch.setattr(subject, "validate_artifact_ref", lambda _value: [])
    real_load = subject.load_json

    def fake_load(path: Path) -> dict:
        if Path(path) == weak_path:
            return json.loads(weak_path.read_text())
        if str(path).endswith("0915_SAM31_WEAK_ROLE_CANARY_V1_USER_VISUAL_REVIEW.json"):
            return {"status": "REJECTED_QUALITY_AS_CLEAN_BASELINE"}
        if str(path).endswith("0915_REMOVAL_ENVELOPE_V1_USER_AUTHORIZATION.json"):
            return {
                "status": "CONFIRMED",
                "authorized_task": "0915_removal_envelope_single_session_canary_v1",
                "authorized_session": "play_cards_0915_001",
                "weights": "ABSENT",
            }
        return real_load(path)

    monkeypatch.setattr(subject, "load_json", fake_load)
    predecessor = subject._validate_predecessor(
        state, "0915_removal_envelope_single_session_canary_v1",
    )
    assert predecessor["task_id"] == "0915_sam31_weak_role_canary_v1"
