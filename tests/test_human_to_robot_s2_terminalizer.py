from __future__ import annotations

import json
import sys
from datetime import datetime

import pytest

from chaoyang.governance import terminalize_human_to_robot_evidence_unlock_s2 as module


class _Clock(datetime):
    current = datetime.fromisoformat("2026-09-23T23:00:00+08:00")

    @classmethod
    def now(cls, tz=None):
        value = cls.current
        return value if tz is None else value.astimezone(tz)


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def terminal_repo(tmp_path, monkeypatch):
    _Clock.current = datetime.fromisoformat("2026-09-23T23:00:00+08:00")
    root = tmp_path / "attempt_0001"
    receipt = tmp_path / "CURRENT_STATUS_RECEIPT.json"
    task_state = tmp_path / "LONG_HORIZON_TASK_STATE.json"
    authority = tmp_path / "CURRENT_AUTHORITY_INDEX.json"
    index = tmp_path / "tasks/current/INDEX.json"
    receipt_out = tmp_path / "tasks/receipts/S2_RESULT.json"
    plan = tmp_path / "docs/current/PLAN.md"
    work_entry = tmp_path / "docs/current/AI_WORK_ENTRY_ZH.md"
    baseline = tmp_path / "docs/current/HUMAN_TO_ROBOT_BASELINE_V1_ZH.md"
    visual_index = tmp_path / "docs/current/visuals/HUMAN_TO_ROBOT_S2/INDEX_ZH.md"
    _write(receipt, {"governance_revision": 7})
    _write(authority, {})
    _write(
        task_state,
        {
            "next_task": {"task_id": module.TASK},
            "tasks": [
                {
                    "task_id": module.TASK,
                    "status": "PENDING",
                    "t0": "2026-09-23T10:45:07+08:00",
                }
            ],
            "recent_events": [],
        },
    )
    _write(index, {"task_packets": [{"task_id": module.TASK}]})
    _write(
        root / "FINAL_VALIDATION.json",
        {
            "status": "PASS_STRUCTURE_WITH_QUALITY_GAPS",
            "video_count": 11,
            "video_full_decode_pass": True,
            "quality_summary": {"formal_product_quality_pass": 0},
        },
    )
    for checkpoint in ("H3", "H6", "H9"):
        _write(root / f"checkpoints/{checkpoint}_RESULT.json", {"checkpoint": checkpoint})
    _write(root / "formal_entry_resume/attempt_0002/RESULT.json", {"status": "PASS"})
    for lane in module.LANES:
        _write(
            root / f"lanes/{lane}/STATE.json",
            {"status": "RUNNING", "writer": {"pid": None}},
        )
    for path in (plan, work_entry, baseline, visual_index):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("initial\n", encoding="utf-8")

    captured = {}

    def publish(*args, **kwargs):
        captured["state"] = args[1]
        captured["event_type"] = kwargs["event_type"]
        captured["packet"] = kwargs["task_packet_index_value"]
        return {"governance_revision": kwargs["expected_revision"] + 1}

    monkeypatch.setattr(module, "ROOT", root)
    monkeypatch.setattr(module, "INDEX", index)
    monkeypatch.setattr(module, "RECEIPT_OUT", receipt_out)
    monkeypatch.setattr(module, "PLAN", plan)
    monkeypatch.setattr(module, "WORK_ENTRY", work_entry)
    monkeypatch.setattr(module, "BASELINE", baseline)
    monkeypatch.setattr(module, "VISUAL_INDEX", visual_index)
    monkeypatch.setattr(module, "RECEIPT_PATH", receipt)
    monkeypatch.setattr(module, "TASK_STATE_PATH", task_state)
    monkeypatch.setattr(module, "AUTHORITY_PATH", authority)
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(module, "datetime", _Clock)
    monkeypatch.setattr(module, "now_iso", lambda: "2026-09-23T23:00:00+08:00")
    monkeypatch.setattr(module, "publish_bundle", publish)
    return root, receipt_out, captured, plan, work_entry, baseline


def test_terminalizer_preserves_zero_quality_and_clears_current_task(
    terminal_repo, monkeypatch
):
    root, receipt_out, captured, plan, work_entry, baseline = terminal_repo
    monkeypatch.setattr(sys, "argv", ["terminalize", "--expected-revision", "7"])
    assert module.main() == 0
    result = json.loads((root / "RESULT.json").read_text(encoding="utf-8"))
    receipt = json.loads(receipt_out.read_text(encoding="utf-8"))
    assert result["task_terminal_status"] == "REJECTED_QUALITY"
    assert result["execution_summary"]["formal_product_quality_pass"] == 0
    assert result["execution_summary"]["formal_product_adopted"] == 0
    assert result["execution_summary"]["robot_r1_screened"] == 0
    assert "CONTACT_ROBOT_R1_NOT_SCREENED_NO_ADMISSIBLE_WINDOW_ESTABLISHED" in result["reason_codes"]
    assert receipt["products_quality"] == "0/4"
    assert captured["state"]["next_task"] is None
    assert captured["state"]["tasks"][0]["status"] == "REJECTED_QUALITY"
    assert captured["packet"]["task_packets"] == []
    assert "已终态" in plan.read_text(encoding="utf-8")
    assert "当前无活动任务" in work_entry.read_text(encoding="utf-8")
    assert "S2 终态" in baseline.read_text(encoding="utf-8")


def test_h12_time_gate_fails_without_terminal_artifacts(terminal_repo, monkeypatch):
    root, receipt_out, _captured, *_ = terminal_repo
    _Clock.current = datetime.fromisoformat("2026-09-23T21:00:00+08:00")
    monkeypatch.setattr(sys, "argv", ["terminalize", "--expected-revision", "7"])
    with pytest.raises(RuntimeError, match="H12_NOT_REACHED"):
        module.main()
    assert not (root / "RESULT.json").exists()
    assert not receipt_out.exists()


def test_terminalizer_can_close_early_only_with_explicit_flag_and_full_evidence(
    terminal_repo, monkeypatch
):
    root, receipt_out, captured, *_ = terminal_repo
    _Clock.current = datetime.fromisoformat("2026-09-23T21:00:00+08:00")
    monkeypatch.setattr(
        sys,
        "argv",
        ["terminalize", "--expected-revision", "7", "--allow-early-close"],
    )
    assert module.main() == 0
    result = json.loads((root / "RESULT.json").read_text(encoding="utf-8"))
    assert result["early_close_authorized"] is True
    assert result["nominal_time_gate_seconds"] == 12 * 3600
    assert receipt_out.is_file()
    assert captured["state"]["next_task"] is None


def test_live_writer_rejects_before_any_lane_mutation(terminal_repo, monkeypatch):
    root, receipt_out, _captured, *_ = terminal_repo
    path = root / "lanes/sensor/STATE.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["writer"] = {"pid": 123}
    path.write_text(json.dumps(value), encoding="utf-8")
    before = {
        lane: (root / f"lanes/{lane}/STATE.json").read_bytes()
        for lane in module.LANES
    }
    monkeypatch.setattr(module, "_pid_live", lambda pid: pid == 123)
    monkeypatch.setattr(sys, "argv", ["terminalize", "--expected-revision", "7"])
    with pytest.raises(RuntimeError, match="ACTIVE_LANE_WRITER:sensor:123"):
        module.main()
    assert all(
        (root / f"lanes/{lane}/STATE.json").read_bytes() == before[lane]
        for lane in module.LANES
    )
    assert not (root / "RESULT.json").exists()
    assert not receipt_out.exists()


def test_gpu_lease_rejects_before_any_lane_mutation(terminal_repo, monkeypatch):
    root, receipt_out, _captured, *_ = terminal_repo
    lease = module.REPO_ROOT / "_run/current/GPU_LEASE.json"
    _write(lease, {"status": "ACQUIRED", "task_id": module.TASK})
    before = {
        lane: (root / f"lanes/{lane}/STATE.json").read_bytes()
        for lane in module.LANES
    }
    monkeypatch.setattr(sys, "argv", ["terminalize", "--expected-revision", "7"])
    with pytest.raises(RuntimeError, match="S2_GPU_LEASE_ACTIVE"):
        module.main()
    assert all(
        (root / f"lanes/{lane}/STATE.json").read_bytes() == before[lane]
        for lane in module.LANES
    )
    assert not (root / "RESULT.json").exists()
    assert not receipt_out.exists()
