from __future__ import annotations

import copy
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.governance import common
from tools.governance.bootstrap_current_governance import build_authority, build_task_state
from tools.governance.recover_stale_tasks import safe_recovery_candidate
from tools.cleanup_current_only_v6 import active_task_reference_view, collect_artifact_ref_paths


def configure_tmp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = tmp_path / "governance"
    monkeypatch.setattr(common, "GOVERNANCE_ROOT", root)
    monkeypatch.setattr(common, "AUTHORITY_PATH", root / "CURRENT_AUTHORITY_INDEX.json")
    monkeypatch.setattr(common, "TASK_STATE_PATH", root / "LONG_HORIZON_TASK_STATE.json")
    monkeypatch.setattr(common, "STATUS_PATH", root / "CURRENT_PROJECT_STATUS_ZH.md")
    monkeypatch.setattr(common, "RECEIPT_PATH", root / "CURRENT_STATUS_RECEIPT.json")
    monkeypatch.setattr(common, "MIN_STATUS_PATH", root / "CURRENT_PROJECT_STATUS_MIN.json")
    monkeypatch.setattr(common, "TASK_QUEUE_PATH", root / "TASK_QUEUE.json")
    monkeypatch.setattr(common, "CHANGELOG_PATH", root / "STATE_CHANGELOG.jsonl")
    monkeypatch.setattr(common, "LOCK_PATH", root / ".governance.lock")
    monkeypatch.setattr(common, "BASELINE_REGISTRY_PATH", root / "CURRENT_BASELINE_REGISTRY_V2.json")
    monkeypatch.setattr(common, "STAGE_BASELINES_PATH", root / "CURRENT_STAGE_BASELINES_ZH.md")
    monkeypatch.setattr(common, "FILE_LAYOUT_PATH", root / "CURRENT_FILE_LAYOUT.json")
    monkeypatch.setattr(common, "REGRESSION_MANIFEST_PATH", root / "CURRENT_REGRESSION_MANIFEST.json")


def test_publish_bundle_receipt_binds_generated_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    configure_tmp(monkeypatch, tmp_path)
    authority = build_authority()
    state = build_task_state()
    receipt = common.publish_bundle(
        authority,
        state,
        event_type="TEST_BOOTSTRAP",
        expected_revision=0,
        generator_path=Path(__file__),
    )
    assert receipt["governance_revision"] == 1
    assert common.RECEIPT_PATH.is_file()
    assert not [
        error
        for reference in receipt["files"].values()
        for error in common.validate_artifact_ref(reference)
    ]
    assert common.load_json(common.AUTHORITY_PATH)["generation_id"] == receipt["generation_id"]
    assert common.load_json(common.TASK_STATE_PATH)["generation_id"] == receipt["generation_id"]


def test_compare_and_swap_rejects_stale_writer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    configure_tmp(monkeypatch, tmp_path)
    common.publish_bundle(build_authority(), build_task_state(), event_type="ONE", expected_revision=0, generator_path=Path(__file__))
    with pytest.raises(RuntimeError, match="CAS revision mismatch"):
        common.publish_bundle(build_authority(), build_task_state(), event_type="STALE", expected_revision=0, generator_path=Path(__file__))


def test_artifact_tampering_is_detected(tmp_path: Path) -> None:
    artifact = tmp_path / "result.json"
    artifact.write_text("before", encoding="utf-8")
    reference = common.artifact_ref(artifact)
    artifact.write_text("after", encoding="utf-8")
    errors = common.validate_artifact_ref(reference)
    assert any("sha256 mismatch" in error or "bytes mismatch" in error for error in errors)


def test_dead_worker_becomes_suspected() -> None:
    state = build_task_state()
    task = state["tasks"][0]
    task.update(
        status="RUNNING",
        pid=999_999_999,
        proc_start_ticks=1,
        heartbeat_at=(datetime.now().astimezone() - timedelta(seconds=120)).isoformat(),
    )
    assert common.freshness(state)["status"] == "SUSPECTED_DEAD_WORKER"


def test_invalid_task_status_is_rejected() -> None:
    state = build_task_state()
    state["tasks"][0]["status"] = "TOTALLY_FINE"
    errors = common.validate_task_state(state)
    assert errors == ["invalid task status: TOTALLY_FINE"]


def test_markdown_does_not_keep_ghost_task() -> None:
    authority = build_authority()
    state = build_task_state()
    markdown = common.render_status(authority, state)
    assert "当前无活跃任务" in markdown
    assert "RUNNING" not in markdown.split("## D. 当前运行任务", 1)[1].split("## E.", 1)[0]


def test_hypothesis_never_becomes_supported_by_copy() -> None:
    authority = build_authority()
    copied = copy.deepcopy(authority)
    hypothesis = next(item for item in copied["claims"] if "absolute-Z" in item["claim"])
    assert hypothesis["status"] == "HYPOTHESIS_ONLY"


def test_publish_crash_before_receipt_leaves_detectable_conflict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    configure_tmp(monkeypatch, tmp_path)
    first = common.publish_bundle(
        build_authority(),
        build_task_state(),
        event_type="FIRST",
        expected_revision=0,
        generator_path=Path(__file__),
    )
    original_write = common.atomic_write

    def fail_before_markdown(path: Path, data: bytes, mode: int = 0o444) -> None:
        if path == common.STATUS_PATH:
            raise RuntimeError("simulated publication crash")
        original_write(path, data, mode)

    monkeypatch.setattr(common, "atomic_write", fail_before_markdown)
    with pytest.raises(RuntimeError, match="simulated publication crash"):
        common.publish_bundle(
            build_authority(),
            build_task_state(),
            event_type="CRASH",
            expected_revision=1,
            generator_path=Path(__file__),
        )
    stale_receipt = common.load_json(common.RECEIPT_PATH)
    assert stale_receipt["governance_revision"] == first["governance_revision"]
    assert any(
        common.validate_artifact_ref(reference)
        for reference in stale_receipt["files"].values()
    )


def test_manual_markdown_edit_breaks_receipt_binding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    configure_tmp(monkeypatch, tmp_path)
    receipt = common.publish_bundle(
        build_authority(),
        build_task_state(),
        event_type="FIRST",
        expected_revision=0,
        generator_path=Path(__file__),
    )
    common.STATUS_PATH.chmod(0o644)
    common.STATUS_PATH.write_text("hand edited\n", encoding="utf-8")
    assert common.validate_artifact_ref(receipt["files"]["project_status"])


def test_gpu_task_is_not_recovered_while_any_compute_process_remains() -> None:
    task = {
        "pid": 999_999_999,
        "proc_start_ticks": 1,
        "gpu_id": 0,
    }
    assert safe_recovery_candidate(task, {12345}) is False
    assert safe_recovery_candidate(task, set()) is True
    assert safe_recovery_candidate(task, None) is False


def test_cleanup_runtime_view_does_not_promote_terminal_history() -> None:
    state = {
        "tasks": [
            {"task_id": "active", "status": "RUNNING", "result": "/current/read.json"},
            {"task_id": "old", "status": "PASSED", "result": "/historical/old.json"},
        ],
        "recent_events": [
            {"task_id": "active", "message": "/current/event.json"},
            {"task_id": "old", "message": "/historical/event.json"},
        ],
        "next_task": {"task_id": "next", "manifest": "/current/next.json"},
    }
    view = active_task_reference_view(state)
    encoded = str(view)
    assert "/current/read.json" in encoded
    assert "/current/event.json" in encoded
    assert "/current/next.json" in encoded
    assert "/historical/old.json" not in encoded
    assert "/historical/event.json" not in encoded


def test_cleanup_collects_only_explicit_artifact_references() -> None:
    output: set[Path] = set()
    collect_artifact_ref_paths(
        {
            "valid": {
                "path": "/current/result.json",
                "bytes": 123,
                "sha256": "a" * 64,
            },
            "missing_sha": {"path": "/old/readme.md", "bytes": 10},
            "plain_path": "/old/terminal.json",
            "nested": [
                {
                    "path": "/current/video.mp4",
                    "bytes": 456,
                    "sha256": "b" * 64,
                }
            ],
        },
        output,
    )
    assert output == {Path("/current/result.json"), Path("/current/video.mp4")}
