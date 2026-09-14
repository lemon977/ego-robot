from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def daemon_module():
    path = Path(__file__).resolve().parents[2] / "tools/run_exact78_heldout_rtc_daemon.py"
    spec = importlib.util.spec_from_file_location("exact78_heldout_daemon_v3_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v3_contract_schedules_exactly_five_unique_sessions_per_task() -> None:
    module = daemon_module()
    contract = module.read(module.CONTRACT)
    assert contract["schema_version"] == "exact78-heldout-checkpoint-prediction-v3"
    for task in module.TASKS:
        values = module.sessions(task)
        assert len(values) == 5 == len(set(values))


def test_complete_requires_matching_session_and_all_prediction_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = daemon_module()
    root = tmp_path / "session"
    root.mkdir()
    monkeypatch.setattr(module, "output", lambda task, session: root)
    (root / "heldout_prediction_manifest.json").write_text(json.dumps({
        "status": "PREDICTIONS_RENDERED", "task": "chips", "session": "heldout_a",
    }), encoding="utf-8")
    (root / "heldout_checkpoint_predictions.npz").touch()
    (root / "heldout_checkpoint_prediction.mp4").touch()
    artifacts = {
        name: module.file_reference(root / name)
        for name in (
            "heldout_checkpoint_predictions.npz",
            "heldout_checkpoint_prediction.mp4",
            "heldout_prediction_manifest.json",
        )
    }
    (root / "FINALIZATION_BINDING.json").write_text(json.dumps({
        "status": "FINALIZATION_BINDING_PASS",
        "task": "chips", "session": "heldout_a",
        "final_root": str(root.absolute()),
        "stale_path_occurrences": 0,
        "reference_identity_failures": 0,
        "artifacts": artifacts,
    }), encoding="utf-8")
    assert module.complete("chips", "heldout_a")
    assert not module.complete("chips", "heldout_b")


def test_seal_output_rebases_paths_and_closes_reference_sha(
    tmp_path: Path,
) -> None:
    module = daemon_module()
    staging = tmp_path / ".heldout_a.123.inflight"
    final = tmp_path / "heldout_a"
    staging.mkdir()
    prediction = staging / "heldout_checkpoint_predictions.npz"
    video = staging / "heldout_checkpoint_prediction.mp4"
    child = staging / "child.json"
    prediction.write_bytes(b"prediction")
    video.write_bytes(b"video")
    child.write_text(json.dumps({
        "prediction": module.file_reference(prediction),
    }), encoding="utf-8")
    manifest = staging / "heldout_prediction_manifest.json"
    manifest.write_text(json.dumps({
        "status": "PREDICTIONS_RENDERED", "task": "chips", "session": "heldout_a",
        "prediction": module.file_reference(prediction),
        "video": module.file_reference(video),
        "child": module.file_reference(child),
    }), encoding="utf-8")
    binding = module.seal_output_paths(
        "chips", "heldout_a", staging, final, staging,
    )
    assert binding["status"] == "FINALIZATION_BINDING_PASS"
    assert binding["stale_path_occurrences"] == 0
    assert binding["reference_records_checked"] == 4
    assert str(staging) not in manifest.read_text(encoding="utf-8")
    assert str(staging) not in json.dumps(binding)
    assert str(final) in manifest.read_text(encoding="utf-8")
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert manifest_payload["child"]["sha256"] == module.file_reference(child)["sha256"]


def test_recover_complete_dead_inflight_without_gpu_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = daemon_module()
    final = tmp_path / "heldout_a"
    staging = tmp_path / ".heldout_a.999999999.inflight"
    staging.mkdir()
    prediction = staging / "heldout_checkpoint_predictions.npz"
    video = staging / "heldout_checkpoint_prediction.mp4"
    prediction.write_bytes(b"prediction")
    video.write_bytes(b"video")
    (staging / "heldout_prediction_manifest.json").write_text(json.dumps({
        "status": "PREDICTIONS_RENDERED", "task": "chips", "session": "heldout_a",
        "prediction": module.file_reference(prediction),
        "video": module.file_reference(video),
    }), encoding="utf-8")
    monkeypatch.setattr(module, "output", lambda task, session: final)
    recovered, receipt = module.recover_complete_inflight("chips", "heldout_a")
    assert recovered
    assert Path(receipt).is_file()
    assert module.complete("chips", "heldout_a")
    assert not staging.exists()


def test_recover_inflight_refuses_live_or_incomplete_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = daemon_module()
    final = tmp_path / "heldout_a"
    staging = tmp_path / f".heldout_a.{module.os.getpid()}.inflight"
    staging.mkdir()
    monkeypatch.setattr(module, "output", lambda task, session: final)
    recovered, reason = module.recover_complete_inflight("chips", "heldout_a")
    assert not recovered
    assert reason.startswith("HOLD_INFLIGHT_PRODUCER_ALIVE")


def test_failed_attempt_commit_is_atomic_and_no_clobber(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = daemon_module()
    final = tmp_path / "final"
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "partial.bin").write_bytes(b"partial")
    monkeypatch.setattr(module, "output", lambda task, session: final)
    receipt = module.commit_attempt(
        "poker", "heldout_p", staging, "FAILED_TERMINAL_CONTINUE", returncode=7,
    )
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["terminal"] is True
    assert payload["status"] == "FAILED_TERMINAL_CONTINUE"
    assert (final / "partial.bin").read_bytes() == b"partial"
    with pytest.raises(RuntimeError, match="NO_CLOBBER"):
        module.commit_attempt("poker", "heldout_p", tmp_path / "other", "COMPLETE")
