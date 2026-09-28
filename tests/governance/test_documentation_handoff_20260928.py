"""Checks for the 2026-09-28 documentation-only handoff."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))

def test_archive_preserves_original_bytes():
    manifest = read("docs/archive/navigation_20260928/MANIFEST.json")
    assert len(manifest["files"]) == 13
    for ref in manifest["files"]:
        b = (ROOT / ref["archive_path"]).read_bytes()
        assert len(b) == ref["bytes"]
        assert hashlib.sha256(b).hexdigest() == ref["sha256"]

def test_history_does_not_authorize_execution():
    history = read("docs/plans/TASK_HISTORY_20260928.json")
    assert history["source_revision"] == 14167
    ids = [r["task_id"] for r in history["tasks"]]
    assert len(ids) == len(set(ids)) == 85
    assert "Historical snapshot" in history["claim_limit"]
    executable = {r["task_id"] for r in read("tasks/current/INDEX.json")["task_packets"] if r.get("execution_allowed")}
    assert not executable.intersection(ids)

def test_historical_quality_is_unchanged():
    status = read("_run/current/human_to_robot_shared_hand_delivery_20260924/attempts/attempt_0001/RESULT.json")
    assert status["task_id"] == "human_to_robot_shared_hand_delivery_20260924"
    assert status["counts"]["products_quality"] == "0/4"
    assert status["counts"]["products_adopted"] == "0/4"
    assert status["counts"]["qualified_kai22_h50_windows"] == 0
    assert not status["training_eligible"]

def test_latest_navigation_and_proposals_registered():
    entries = read("docs/governance/DOC_AUTHORITY_MAP.json")["documents"]
    ids = {e["document_id"] for e in entries if e["status"] == "CURRENT"}
    assert {"shared_hand_terminal_navigation", "next_task_proposals",
            "plan_history_navigation", "git_delivery_boundary"} <= ids

def test_lane_cards_are_short_and_not_running():
    for name in ("V5_SCENE.md", "V5_SENSOR.md", "V5_MOTION.md", "V5_HURO.md"):
        text = (ROOT / "docs/current" / name).read_text(encoding="utf-8")
        assert len(text.splitlines()) <= 35
        assert "无活动任务" in text
        assert "S2 进行中" not in text

def test_readme_uses_latest_result_and_no_stale_interface_pass():
    text = (ROOT / "docs/current/README_ZH.md").read_text(encoding="utf-8")
    assert "SHARED_HAND_DELIVERY_RESULT_ZH.md" in text
    assert "Sensor097接口通过" not in text
    assert "COMPLETION_20260928_ZH.md" in text

def test_proposals_are_not_new_permissions():
    text = (ROOT / "docs/current/NEXT_ACTIONS_ZH.md").read_text(encoding="utf-8")
    assert "未授权、未登记、未执行" in text
    assert "不自动创建任务或占用GPU" in text
