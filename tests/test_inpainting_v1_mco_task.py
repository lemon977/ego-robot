"""CPU admission/receipt tests; never register, lease, render or run a model."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


EXECUTOR = Path(__file__).resolve().parents[1] / "scripts/maintenance/inpainting_v1_mco_task.py"


@pytest.fixture
def task(monkeypatch, tmp_path):
    definition = importlib.util.spec_from_file_location("_test_original_v1_mco_executor", EXECUTOR)
    module = importlib.util.module_from_spec(definition)
    definition.loader.exec_module(module)
    code = tmp_path / "source.py"
    code.write_text("# synthetic signed code\n")
    ref = {"path": str(code), "bytes": code.stat().st_size, "sha256": "1" * 64}
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(module, "INFRA_PATH", code)
    monkeypatch.setattr(module, "artifact_ref", lambda path: dict(ref, path=str(path)))
    monkeypatch.setattr(module, "verify_ref", lambda reference: None)
    return module, tmp_path, ref


def _spec(task, mode="Clean"):
    _, root, ref = task
    return {"spec_schema": "inpainting-v1-mco-stage-spec-v1", "task_id": "inpainting_v1_mco_clean_a01",
        "mode": mode, "inputs": {"original_v1": ref}, "runtime_code": [ref], "model_weights": [],
        "infrastructure": ref, "calibration": {"scene_depth": "ABSENT"}, "schema": {}, "config": {},
        "read_set": [str(root)], "output_dir": str(root / "_run/fresh_a01"),
        "review_dir": str(root / "review/new_a01"),
        "global_deadline_at": (datetime.now(timezone.utc) + timedelta(hours=8)).isoformat(),
        "sessions": [{"session_key": "050", "native_frames": 128}],
        "budgets": {"wall_seconds_max": 3600, "cpu_threads_per_worker": 2,
                    "gpu_count": 0 if mode == "Boundary" else 1, "model_calls_max": 0 if mode == "Boundary" else 1,
                    "render_calls_max": 128 if mode == "Boundary" else 0,
                    "training_steps_max": 0, "output_bytes_estimate": 128 << 20}}


@pytest.mark.parametrize("mode", ["Mask", "Clean", "Boundary"])
def test_three_finite_modes_validate_without_algorithm_execution(task, mode):
    module, _, _ = task
    module.validate_spec(_spec(task, mode))


@pytest.mark.parametrize("key,value", [("wall_seconds_max", 3601), ("wall_seconds_max", 0),
    ("cpu_threads_per_worker", 3), ("training_steps_max", 1), ("model_calls_max", -1),
    ("render_calls_max", -1), ("output_bytes_estimate", (512 << 20) + 1),
    ("output_bytes_estimate", 0), ("gpu_count", 2), ("gpu_count", True)])
def test_invalid_budgets_rejected(task, key, value):
    module, _, _ = task
    spec = _spec(task)
    spec["budgets"][key] = value
    with pytest.raises(ValueError, match="MCO_"):
        module.validate_spec(spec)


@pytest.mark.parametrize("mode,budget,value", [("Boundary", "gpu_count", 1),
    ("Boundary", "model_calls_max", 1), ("Clean", "render_calls_max", 1), ("Mask", "gpu_count", 0)])
def test_mode_specific_resource_isolation(task, mode, budget, value):
    module, _, _ = task
    spec = _spec(task, mode)
    spec["budgets"][budget] = value
    with pytest.raises(ValueError, match="MCO_MODE_"):
        module.validate_spec(spec)


def test_boundary_frozen_window_is_not_claimed_as_full_sequence(task):
    module, _, _ = task
    spec = _spec(task, "Boundary")
    spec["sessions"][0]["output_frames"] = [0, 64, 96, 127]
    module.validate_spec(spec)


@pytest.mark.parametrize("frames", [[127, 64], [64, 64], [-1], [128], [], [True]])
def test_invalid_boundary_windows_rejected(task, frames):
    module, _, _ = task
    spec = _spec(task, "Boundary")
    spec["sessions"][0]["output_frames"] = frames
    with pytest.raises(ValueError, match="MCO_FROZEN_BOUNDARY_FRAME_SELECTION_REQUIRED"):
        module.validate_spec(spec)


def test_clean_cannot_crop_to_failure_window(task):
    module, _, _ = task
    spec = _spec(task)
    spec["sessions"][0]["output_frames"] = [127]
    with pytest.raises(ValueError, match="MCO_FROZEN_BOUNDARY_FRAME_SELECTION_REQUIRED"):
        module.validate_spec(spec)


@pytest.mark.parametrize("field,value", [("task_id", "old_neutral_review"),
    ("task_id", "inpainting_v1_mco_../unsafe"), ("mode", "Training"), ("spec_schema", "old-schema")])
def test_wrong_identity_or_algorithm_mode_rejected(task, field, value):
    module, _, _ = task
    spec = _spec(task)
    spec[field] = value
    with pytest.raises(ValueError, match="MCO_"):
        module.validate_spec(spec)


def test_duplicate_session_identity_rejected(task):
    module, _, _ = task
    spec = _spec(task)
    spec["sessions"].append(deepcopy(spec["sessions"][0]))
    with pytest.raises(ValueError, match="MCO_FULL_NATIVE_UNIQUE_SESSIONS_REQUIRED"):
        module.validate_spec(spec)


def test_expired_global_deadline_rejected(task):
    module, _, _ = task
    spec = _spec(task)
    spec["global_deadline_at"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    with pytest.raises(ValueError, match="MCO_GLOBAL_DEADLINE"):
        module.validate_spec(spec)


def test_code_sha_drift_rejected_not_repaired(task, monkeypatch):
    module, _, _ = task
    def reject(ref):
        raise ValueError("TEST_SHA_DRIFT")
    monkeypatch.setattr(module, "verify_ref", reject)
    with pytest.raises(ValueError, match="TEST_SHA_DRIFT"):
        module.validate_spec(_spec(task))


def test_wrong_operation_argv_rejected_before_snapshot(task, monkeypatch):
    module, _, _ = task
    monkeypatch.setattr(module, "_infra", lambda: pytest.fail("must not read ledger for invalid argv"))
    with pytest.raises(ValueError, match="MCO_EXACT_PACKET_ARGUMENT_REQUIRED"):
        module.authorize_operation(["--mode", "Clean"])


def test_project_path_rejects_sibling_outside_signed_root(task):
    module, root, _ = task
    with pytest.raises(ValueError, match="MCO_REAL_PROJECT_PATH_REQUIRED"):
        module._project_path(str(root.parent / "outside_scope"))


def _inner(task, spec, *, frames=None):
    _, root, _ = task
    root_out = Path(spec["output_dir"])
    root_out.mkdir(parents=True)
    artifact = root_out / "numeric_evidence.json"
    artifact.write_text("{}\n")
    frames = frames if frames is not None else list(range(128))
    row = {"session_key": "050", "native_frames": 128, "native_output_frames": len(frames),
           "completed_frame_indices": frames, "execution_complete": True}
    inner = {"schema": "inpainting-v1-mco-stage-result-v1", "status": "SEALED_COMPLETE",
        "task_id": spec["task_id"], "run_signature": "2" * 64, "mode": spec["mode"],
        "candidate_adopted": False, "control_ground_truth": False, "model_calls": 0,
        "render_calls": len(frames) if spec["mode"] == "Boundary" else 0,
        "sessions": [row], "artifacts": [{"path": str(artifact), "bytes": artifact.stat().st_size,
            "sha256": "3" * 64}]}
    packet = {**spec, "run_signature": "2" * 64}
    return packet, inner, root_out / "MODEL_RESULT.json"


def test_actual_complete_result_consumable_only_after_verification(task):
    module, _, _ = task
    packet, inner, path = _inner(task, _spec(task))
    path.write_text(json.dumps(inner))
    assert module.validate_model_result(packet)["path"] == str(path)


@pytest.mark.parametrize("change", ["partial", "wrong_signature", "excess_calls", "wrong_identity", "adopted"])
def test_incomplete_or_misbound_result_rejected(task, change):
    module, _, _ = task
    packet, inner, path = _inner(task, _spec(task))
    if change == "partial":
        inner["sessions"][0]["completed_frame_indices"] = [127]
    elif change == "wrong_signature":
        inner["run_signature"] = "4" * 64
    elif change == "excess_calls":
        inner["model_calls"] = 2
    elif change == "wrong_identity":
        inner["sessions"][0]["session_key"] = "001"
    else:
        inner["candidate_adopted"] = True
    path.write_text(json.dumps(inner))
    with pytest.raises(ValueError, match="MCO_"):
        module.validate_model_result(packet)


def test_boundary_window_receipt_preserves_explicit_unseen_range(task):
    module, _, _ = task
    spec = _spec(task, "Boundary")
    spec["sessions"][0]["output_frames"] = [64, 96, 127]
    packet, inner, path = _inner(task, spec, frames=[64, 96, 127])
    path.write_text(json.dumps(inner))
    assert module.validate_model_result(packet)["path"] == str(path)


def test_zero_statvfs_requires_real_recent_write_proof(task, monkeypatch):
    module, root, _ = task
    spec = _spec(task)
    proof_path = root / "CAPACITY_PROOF.json"
    proof = {"schema": "inpainting-v1-mco-capacity-proof-v1",
        "status": "BOUNDED_WRITE_FSYNC_PASS_STATVFS_UNVERIFIED", "actual_written_fsynced_bytes": 512 << 20,
        "own_scratch_truncated_after_probe": True, "files_deleted": 0,
        "completed_at": datetime.now(timezone.utc).isoformat()}
    proof_path.write_text(json.dumps(proof))
    spec["inputs"]["capacity_proof"] = {"path": str(proof_path), "bytes": proof_path.stat().st_size, "sha256": "5" * 64}
    monkeypatch.setattr(module.os, "statvfs", lambda path: SimpleNamespace(f_bavail=0, f_frsize=4096))
    module._output_capacity(spec)
    proof["actual_written_fsynced_bytes"] = 1
    proof_path.write_text(json.dumps(proof))
    with pytest.raises(ValueError, match="MCO_RECENT_BOUNDED_CAPACITY_PROOF_REQUIRED"):
        module._output_capacity(spec)


def test_cpu_worker_environment_hides_cuda_and_stays_project_local(task):
    module, root, _ = task
    output = root / "bounded_cache"
    output.mkdir()
    environment = module._environment(output, None)
    assert environment["CUDA_VISIBLE_DEVICES"] == ""
    assert environment["OMP_NUM_THREADS"] == environment["NUMBA_NUM_THREADS"] == "2"
    assert Path(environment["TMPDIR"]).is_relative_to(output)
    assert Path(environment["TORCH_HOME"]).is_relative_to(output)
