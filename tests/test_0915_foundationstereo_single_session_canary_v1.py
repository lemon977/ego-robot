from __future__ import annotations

import inspect
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from chaoyang.ops import run_0915_foundationstereo_single_session_canary_v1 as subject


def test_physical_left_depth_keeps_unflipped_x_plus_disparity_domain() -> None:
    flipped = np.full((2, 8), 2.0, np.float32)
    disparity, depth, valid = subject.physical_left_depth(
        flipped, focal_px=10.0, baseline_m=0.2,
    )
    np.testing.assert_allclose(disparity, 2.0)
    assert valid[:, :6].all()
    assert not valid[:, 6:].any()
    np.testing.assert_allclose(depth[valid], 1.0)


def test_left_right_consistency_uses_x_right_equals_x_left_plus_d() -> None:
    left = np.full((2, 8), 2.0, np.float32)
    right = np.full((2, 8), 2.25, np.float32)
    geometric = np.ones((2, 8), bool)
    residual, testable, consistent = subject.left_right_consistency(
        left, right, geometric, max_residual_px=0.5,
    )
    assert testable[:, :6].all()
    assert not testable[:, 6:].any()
    np.testing.assert_allclose(residual[testable], 0.25)
    assert np.array_equal(consistent, testable)


def test_registration_map_is_nonlinear_map_not_identity_join() -> None:
    xx = np.broadcast_to(np.arange(1280, dtype=np.float32), (960, 1280)).copy()
    yy = np.broadcast_to(np.arange(960, dtype=np.float32)[:, None], (960, 1280)).copy()
    maps = subject.build_registration_maps(
        (xx, yy), source_width=2048, source_height=1536,
    )
    assert maps["depth_to_sam_resize_xy"].shape == (480, 640, 2)
    np.testing.assert_allclose(
        maps["depth_pixel_to_full_rectified_left_xy"][0, 0], [0.5, 0.5],
    )
    # OpenCV pixel-centre resize maps source 0.5 to target 0.125 here;
    # critically, this is not an identity registration.
    np.testing.assert_allclose(
        maps["depth_to_sam_resize_xy"][0, 0], [0.125, 0.125], atol=1e-6,
    )
    assert maps["sam_resize_in_bounds"].dtype == bool


def _rows(count: int = 10) -> list[dict]:
    return [{
        "geometric_valid_fraction": 0.80,
        "lr_testable_fraction": 0.75,
        "lr_consistent_fraction_of_testable": 0.90,
        "lr_residual_p90_px": 0.5,
        "final_valid_fraction": 0.65,
        "depth_p50_m": 0.55 + 0.001 * index,
        "formula_recompute_max_abs_error_m": 0.0,
        "edge": {"rgb_supported_disparity_edge_fraction": 0.50},
    } for index in range(count)]


def test_quality_gate_requires_decode_lr_temporal_and_edges() -> None:
    passed = subject.aggregate_quality(_rows(), 10)
    assert passed["passed"] is True
    assert set(passed["gates"]) == {
        "full_decode", "formula_recompute", "geometric_validity", "lr_testable_coverage",
        "lr_consistency", "final_validity", "temporal_distribution_stability",
        "rgb_edge_support",
    }
    failed = subject.aggregate_quality(_rows(9), 10)
    assert failed["gates"]["full_decode"] is False
    assert failed["passed"] is False


def test_runner_is_single_weight_t0_rectification_and_no_external_accuracy_claim() -> None:
    source = inspect.getsource(subject)
    assert subject.MODEL_WEIGHT.endswith("model_best_bp2.pth")
    assert "EXACT_T0_ACCEPTED_CALIBRATION_NO_REESTIMATION" in source
    assert "SAME_SESSION_INTRINSICS_HELDOUT_ESTIMATED_RECTIFIED" in source
    assert '"external_accuracy": "UNVERIFIED"' in source
    assert "ABSENT_NOT_FABRICATED" in source
    assert "PICO26" not in source
    assert "trackingData_hand" in source and "NOT_CONSUMED" in source


def test_runner_accepts_only_sha_bound_enumerated_processed_root_relocation() -> None:
    relocation = json.loads(subject.PATH_RELOCATION_RECEIPT.read_text(encoding="utf-8"))
    assert relocation["status"] == "PASS_CONTENT_IDENTICAL_PATH_RELOCATION"
    assert relocation["source_data_modified"] is False
    assert subject.LEGACY_SESSION.parts[-4] == "chips_cards_hands__0915"
    assert subject.SESSION.parts[-4] == "chips_cards_hands_0915"
    source = inspect.getsource(subject.validate_preflight)
    assert 'for field in ("bytes", "sha256")' in source
    assert "legacy path identity drift" in source
    assert "content closure drift after path relocation" in source


def test_runner_uses_pinned_environment_and_central_gpu_lease() -> None:
    assert subject.PINNED_PYTHON_LAUNCHER.name == "foundationstereo_gpu_python.sh"
    assert subject.CENTRAL_GPU_LEASE.name == "run_gpu_command_with_v71_lease.py"
    source = Path(subject.__file__).read_text(encoding="utf-8")
    assert '"--priority", "CANARY"' in source
    assert '"--", *worker_command' in source
    assert '"--executor-epoch", str(args.executor_epoch)' in source


def test_orchestrator_polls_worker_before_reading_released_lease() -> None:
    source = Path(subject.__file__).read_text(encoding="utf-8")
    sleep_at = source.index("time.sleep(30)")
    exit_recheck_at = source.index("if process.poll() is not None:", sleep_at)
    lease_read_at = source.index(
        'lease_path = ROOT / "_run/current/GPU_LEASE.json"', sleep_at,
    )
    assert sleep_at < exit_recheck_at < lease_read_at


def test_namespaces_are_task_owned_and_visual_is_fixed() -> None:
    subject.validate_output_namespace(
        subject.OUTPUT_NAMESPACE / "attempts/attempt_0001",
        subject.VISUAL_NAMESPACE,
    )
    with pytest.raises(RuntimeError, match="fresh fixed attempt"):
        subject.validate_output_namespace(
            subject.OUTPUT_NAMESPACE.parent / "another_task/attempt_0001",
            subject.VISUAL_NAMESPACE,
        )
    with pytest.raises(RuntimeError, match="visual root must equal"):
        subject.validate_output_namespace(
            subject.OUTPUT_NAMESPACE / "attempts/attempt_0001",
            subject.VISUAL_NAMESPACE.parent / "other",
        )


def test_edge_support_returns_bounded_fraction() -> None:
    image = np.zeros((960, 1280, 3), np.uint8)
    image[:, 640:] = 255
    disparity = np.ones((480, 640), np.float32)
    disparity[:, 320:] = 10.0
    metrics = subject.edge_support_metrics(image, disparity, np.ones_like(disparity, bool))
    assert metrics["strong_disparity_edge_pixels"] > 0
    assert 0.0 <= metrics["rgb_supported_disparity_edge_fraction"] <= 1.0


def test_exactly_one_sbs_is_required(tmp_path: Path) -> None:
    source = tmp_path / "source_stereo"
    source.mkdir()
    expected = source / f"CameraRecord_{subject.SESSION_ID}_stereo.mp4"
    expected.write_bytes(b"sbs")
    assert subject.resolve_exactly_one_sbs(tmp_path) == expected
    (source / "another_stereo.mp4").write_bytes(b"ambiguous")
    with pytest.raises(RuntimeError, match="exactly one SBS"):
        subject.resolve_exactly_one_sbs(tmp_path)


def test_writer_claim_binds_live_pid_startticks_epoch_token_and_signature(tmp_path: Path) -> None:
    output = tmp_path / "attempt_0001"
    output.mkdir()
    signature_sha = "1" * 64
    claim = {
        "schema_version": "0915-foundationstereo-writer-claim-v1",
        "task_id": subject.TASK_ID,
        "session_id": subject.SESSION_ID,
        "attempt_id": "attempt_0001",
        "weights": [subject.MODEL_WEIGHT],
        "claimed_at": "2026-09-18T16:00:00+08:00",
        "status": "CLAIMED",
        "pid": os.getpid(),
        "proc_start_ticks": subject.process_start_ticks(os.getpid()),
        "executor_epoch": 7,
        "fencing_token_sha256": hashlib.sha256(b"bounded-fence-token").hexdigest(),
        "run_signature_sha256": signature_sha,
        "unique_write_root": str(output.resolve()),
    }
    claim_path = output / "CLAIM.json"
    claim_path.write_text(json.dumps(claim), encoding="utf-8")
    observed = subject.validate_writer_claim(
        claim_path, output=output, signature_sha256=signature_sha,
        executor_epoch=7, require_current_process_descendant=True,
    )
    assert observed["pid"] == os.getpid()
    claim["proc_start_ticks"] += 1
    claim_path.write_text(json.dumps(claim), encoding="utf-8")
    with pytest.raises(RuntimeError, match="claim/fence mismatch"):
        subject.validate_writer_claim(
            claim_path, output=output, signature_sha256=signature_sha,
            executor_epoch=7, require_current_process_descendant=True,
        )


def test_immutable_writer_records_refuse_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "CLAIM.json"
    subject.atomic_json_new(path, {"writer": 1})
    with pytest.raises(FileExistsError):
        subject.atomic_json_new(path, {"writer": 2})
    assert json.loads(path.read_text(encoding="utf-8")) == {"writer": 1}


def test_runtime_closure_receipt_pins_current_bounded_canary_without_claiming_inference() -> None:
    receipt = json.loads(subject.RUNTIME_CLOSURE_RECEIPT.read_text(encoding="utf-8"))
    config = subject.validate_canary_config()
    assert receipt["status"] == "PASS_BOUNDED_CANARY_RUNTIME_IDENTITY_CLOSED"
    assert receipt["execution_performed"] is False
    assert receipt["batch_authorized"] is False
    assert receipt["legacy_environment_authority_role"] == (
        "PINNED_PROVENANCE_NOT_CURRENT_GPU_ADMISSION"
    )
    assert receipt["output_schema_identities"] == config["output_schema_identities"]
    assert receipt["vendor_runtime_tree"] == subject._vendor_manifest_identity()
    assert set(receipt["artifacts"]) == set(subject._runtime_paths())
    # The 3.3 GB checkpoint is closed by its independently materialized pin;
    # this unit test deliberately does not load or rehash model bytes.
    pin = json.loads(subject.ASSET_PIN.read_text(encoding="utf-8"))
    assert receipt["artifacts"]["checkpoint"]["sha256"] == (
        pin["files"][subject.CHECKPOINT.name]["sha256"]
    )


def test_run_signature_covers_every_runtime_and_input_authority() -> None:
    source = inspect.getsource(subject.build_run_signature)
    for token in (
        "task_packet", "t0_stereo_preflight", "processed_root_relocation",
        "camera_params", "source_stereo", "runtime_closure_receipt",
        "vendor_runtime_tree", "calibration_identity", "schema_identity",
        "checkpoint_cfg", "canary_config",
    ):
        assert token in source
    assert set(subject._runtime_paths()) == {
        "runner", "model_worker", "gpu_launcher", "gpu_lease_wrapper",
        "environment_authority", "environment_lock",
        "environment_snapshot_manifest", "vendor_manifest", "checkpoint",
        "checkpoint_cfg", "asset_pin", "canary_config",
    }


def test_worker_cannot_bypass_orchestrator_writer_fence() -> None:
    source = inspect.getsource(subject.run_worker)
    assert "worker claim must be the attempt-owned writer claim" in source
    assert "worker signature must be the attempt-owned run signature" in source
    assert "require_current_process_descendant=True" in source
    main_source = inspect.getsource(subject.main)
    assert "worker requires orchestrator-owned claim and run signature" in main_source


def test_consumption_authority_is_success_only_and_scope_limited() -> None:
    assert subject.consumer_authority(True) == {
        "consumption_authorized": True,
        "authorized_scopes": ["VISUAL_OBJECT6D_CANDIDATE_INPUT"],
    }
    assert subject.consumer_authority(False) == {
        "consumption_authorized": False,
        "authorized_scopes": [],
    }
    source = inspect.getsource(subject.run_worker)
    assert source.count("**consumer_authority(bool(quality[\"passed\"]))") == 3
    assert '"external_accuracy": "UNVERIFIED"' in source


def test_depth_packet_closes_quality_terminal_and_runtime_identity_outputs() -> None:
    packet = subject.build_packet(subject.TASK_ID)
    assert len(packet["read_set"]) <= 8
    assert "REJECTED_QUALITY" in packet["stop_conditions"]
    assert "configs/systems/depth/foundationstereo_0915_canary_v1.json" in packet["read_set"]
    assert "tasks/receipts/FOUNDATIONSTEREO_RUNTIME_CLOSURE_V1.json" in packet["read_set"]
    for name in (
        "CLAIM.json", "RUN_SIGNATURE.json", "DEPTH_CONTRACT.json",
        "DEPTH_WORKER_RESULT.json", "RESULT.json", "RUN_RECEIPT.json",
    ):
        assert name in packet["required_outputs"]
