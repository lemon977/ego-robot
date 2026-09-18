from __future__ import annotations

import inspect
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from chaoyang.ops import run_0915_planar_object6d_observability_canary_v2 as subject


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def passing_result() -> dict:
    return {
        "task_id": subject.DEPTH_TASK_ID,
        "session_id": subject.SESSION_ID,
        "status": "PASSED",
        "depth_admission": "PASS",
        "consumption_authorized": True,
        "authorized_scopes": [subject.AUTHORIZED_SCOPE],
        "source_mutated": False,
        "external_accuracy": "UNVERIFIED",
    }


def passing_contract() -> dict:
    return {
        "schema_version": "0915-foundationstereo-encoded-depth-contract-v1",
        "task_id": subject.DEPTH_TASK_ID,
        "session_id": subject.SESSION_ID,
        "frame_count": 150,
        "frame_geometry": [640, 480],
        "depth_reference": subject.DEPTH_REFERENCE,
        "depth_to_sam_resize_homography": [
            [2.0, 0.0, 0.5],
            [0.0, 2.0, 0.5],
            [0.0, 0.0, 1.0],
        ],
        "native_model_confidence": "ABSENT_NOT_FABRICATED",
        "occluded_or_hidden_geometry": "INVALID_NOT_COMPLETED",
        "consumption_authorized": True,
        "authorized_scopes": [subject.AUTHORIZED_SCOPE],
    }


def passing_summary() -> dict:
    return {
        "schema_version": "0915-foundationstereo-encoded-depth-summary-v1",
        "task_id": subject.DEPTH_TASK_ID,
        "session_id": subject.SESSION_ID,
        "status": "PASS",
        "depth_admission": "PASS",
        "frame_count": 150,
        "model_load_count": 1,
        "model_inference_count": 300,
        "quality": {"passed": True},
        "review_decode": {"full_decode": True, "frame_count": 150},
        "source_mutated": False,
    }


def materialize_depth_headers(root: Path) -> None:
    write_json(root / "RESULT.json", passing_result())
    write_json(root / "ADAPTER_CONTRACT.json", {"encoded": True})
    write_json(root / "RGB_ALIGNMENT_QA.json", {"pixel_exact": True})
    write_json(root / "DEPTH_CONTRACT.json", passing_contract())
    write_json(root / "DEPTH_SUMMARY.json", passing_summary())


def test_runner_is_fixed_to_encoded_depth_and_three_physical_cards() -> None:
    assert subject.TASK_ID == "0915_planar_object6d_observability_canary_v2"
    assert subject.DEPTH_ROOT == (
        subject.ROOT
        / "_run/current/0915_foundationstereo_encoded_domain_canary_v1/"
        "attempts/attempt_0001"
    )
    assert subject.DEPTH_REFERENCE == (
        "PHYSICAL_LEFT_ENCODED_RESIZE_ONLY_OPTICAL_Z"
    )
    assert subject.INSTANCE_IDS == (
        "playing_card_00",
        "playing_card_01",
        "playing_card_02",
    )
    assert subject.SCHEMA.name == "object6d_planar_observability_v2.schema.json"


def test_analytic_pixel_center_mapping_is_exact_2x_plus_half() -> None:
    mapping, valid = subject.depth_to_sam_pixel_center_map()
    assert mapping.shape == (480, 640, 2)
    assert valid.shape == (480, 640)
    assert valid.all()
    np.testing.assert_array_equal(mapping[0, 0], [0.5, 0.5])
    np.testing.assert_array_equal(mapping[17, 23], [46.5, 34.5])
    np.testing.assert_array_equal(mapping[-1, -1], [1278.5, 958.5])
    assert subject.MAPPING_CONTRACT == {
        "type": "ANALYTIC_PIXEL_CENTER_2X",
        "formula_x": "x_sam = 2*x_depth + 0.5",
        "formula_y": "y_sam = 2*y_depth + 0.5",
        "source": "ORIGINAL_PHYSICAL_LEFT_640x480",
        "target": "PHYSICAL_LEFT_SOURCEINDEX_1_RESIZE_ONLY_1280x960",
    }


def test_runner_does_not_reference_wrong_domain_artifacts_or_validator() -> None:
    source = Path(subject.__file__).read_text(encoding="utf-8")
    assert "CALIBRATION.json" not in source
    assert "REGISTRATION.json" not in source
    assert "REGISTRATION_MAPS.npz" not in source
    assert "validate_foundationstereo_object6d_admission(" not in source
    assert "validate_encoded_foundationstereo_object6d_admission" in source
    assert "cv2.remap" not in source


def test_encoded_depth_admission_rechecks_contract_and_summary(
    tmp_path: Path, monkeypatch,
) -> None:
    materialize_depth_headers(tmp_path)
    monkeypatch.setattr(
        subject,
        "validate_encoded_depth_admission",
        lambda _path: {
            "result": passing_result(),
            "contract": passing_contract(),
            "summary": passing_summary(),
        },
    )
    admitted = subject.validate_depth_upstream(tmp_path)
    assert admitted["contract"]["depth_reference"] == subject.DEPTH_REFERENCE
    assert set(admitted["references"]) == {
        "depth_result",
        "depth_adapter_contract",
        "depth_rgb_alignment_qa",
        "depth_contract",
        "depth_summary",
    }

    wrong = passing_contract()
    wrong["depth_to_sam_resize_homography"][0][2] = 0.0
    monkeypatch.setattr(
        subject,
        "validate_encoded_depth_admission",
        lambda _path: {
            "result": passing_result(),
            "contract": wrong,
            "summary": passing_summary(),
        },
    )
    with pytest.raises(RuntimeError, match="CONTRACT_DRIFT"):
        subject.validate_depth_upstream(tmp_path)


def test_depth_frame_requires_encoded_reference_and_physical_intrinsics(
    tmp_path: Path,
) -> None:
    path = tmp_path / "000000.npz"
    intrinsic = np.asarray([
        [300.0, 0.0, 319.5],
        [0.0, 300.0, 239.5],
        [0.0, 0.0, 1.0],
    ])
    np.savez_compressed(
        path,
        frame_id=np.asarray(0, np.int32),
        depth_m=np.ones((480, 640), np.float32),
        valid=np.ones((480, 640), bool),
        physical_left_intrinsics=intrinsic,
        depth_reference=np.asarray(subject.DEPTH_REFERENCE),
    )
    depth, valid, observed_intrinsic = subject.load_depth_frame(path, 0)
    assert depth.shape == (480, 640)
    assert valid.all()
    np.testing.assert_array_equal(observed_intrinsic, intrinsic)

    np.savez_compressed(
        path,
        frame_id=np.asarray(0, np.int32),
        depth_m=np.ones((480, 640), np.float32),
        valid=np.ones((480, 640), bool),
        physical_left_intrinsics=intrinsic,
        depth_reference=np.asarray("PHYSICAL_LEFT_RECTIFIED_OPTICAL_Z"),
    )
    with pytest.raises(RuntimeError, match="coordinate contract drift"):
        subject.load_depth_frame(path, 0)


def test_v2_schema_and_runner_mapping_contract_are_identical() -> None:
    schema = json.loads(subject.SCHEMA.read_text(encoding="utf-8"))
    mapping = schema["properties"]["inputs"]["properties"]["depth_to_sam_mapping"]
    assert set(mapping["required"]) == set(subject.MAPPING_CONTRACT)
    assert {
        key: mapping["properties"][key]["const"]
        for key in mapping["required"]
    } == subject.MAPPING_CONTRACT


def test_runner_is_cpu_only_fenced_and_writes_v2_outputs() -> None:
    source = inspect.getsource(subject.main)
    assert '"weights": "ABSENT"' in source
    assert '"gpu_used": False' in source
    assert 'output / "CLAIM.json"' in source
    assert 'output / "RUN_SIGNATURE.json"' in source
    assert 'output / "OBJECT6D_OBSERVABILITY_V2.json"' in inspect.getsource(
        subject.run_canary
    )
    assert subject.VISUAL_NAMESPACE.is_relative_to(subject.ROOT / "docs/current/visuals")


def test_writer_claim_binds_live_pid_epoch_fence_and_signature(tmp_path: Path) -> None:
    output = tmp_path / "attempt_0001"
    output.mkdir()
    claim = {
        "schema_version": "0915-planar-object6d-observability-writer-claim-v2",
        "task_id": subject.TASK_ID,
        "session_id": subject.SESSION_ID,
        "attempt_id": output.name,
        "status": "CLAIMED",
        "weights": "ABSENT",
        "pid": os.getpid(),
        "proc_start_ticks": subject.process_start_ticks(os.getpid()),
        "executor_epoch": 3,
        "fencing_token_sha256": hashlib.sha256(b"bounded-token").hexdigest(),
        "run_signature_sha256": "a" * 64,
        "unique_write_root": str(output),
    }
    path = output / "CLAIM.json"
    write_json(path, claim)
    observed = subject.validate_writer_claim(
        path, output=output, signature_sha="a" * 64, executor_epoch=3,
    )
    assert observed["pid"] == os.getpid()
    claim["executor_epoch"] = 4
    write_json(path, claim)
    with pytest.raises(RuntimeError, match="claim/fence mismatch"):
        subject.validate_writer_claim(
            path, output=output, signature_sha="a" * 64, executor_epoch=3,
        )
