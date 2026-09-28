"""Terminal navigation may not advertise a running Human-to-Robot task."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _read(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def test_terminal_navigation_does_not_claim_running_task() -> None:
    status = _read("docs/current/STATUS.json")
    ledger = _read("docs/governance/LONG_HORIZON_TASK_STATE.json")
    index = _read("tasks/current/INDEX.json")
    if status.get("active_tasks") or ledger.get("next_task") is not None:
        return
    assert index.get("task_packets") == []
    entry = (ROOT / "docs/current/AI_WORK_ENTRY_ZH.md").read_text(encoding="utf-8")
    readme = (ROOT / "docs/current/README_ZH.md").read_text(encoding="utf-8")
    assert "进行中" not in entry.splitlines()[0]
    assert "当前唯一活动任务" not in readme
    assert status["latest_terminal_status"] in {"PASSED", "REJECTED_QUALITY", "FAILED"}


def test_result_counts_and_label_qualification_remain_separate() -> None:
    result = _read(
        "_run/current/human_to_robot_result_breakthrough_20260924/attempts/attempt_0001/RESULT.json"
    )
    assert result["counts"]["products_quality"] == "0/4"
    assert result["counts"]["products_adopted"] == "0/4"
    assert result["data"]["interface_pass"] is True
    assert result["data"]["label_quality_pass"] is False
    assert result["data"]["qualified_h50_windows"] == 0
