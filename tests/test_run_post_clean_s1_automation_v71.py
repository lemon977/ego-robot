from pathlib import Path

from chaoyang.ops.run_post_clean_s1_automation_v71 import clean_is_closed, payload_status, publish_role_bundle


def test_post_clean_source_contains_bounded_robot_zbuffer_before_s1() -> None:
    source = Path("src/chaoyang/ops/run_post_clean_s1_automation_v71.py").read_text(encoding="utf-8")
    assert '"--frame-count", "420"' in source
    assert "results = [run_robot_zbuffer_fullsession" in source
    assert "role_rows = [run_role_repair" in source
    # Role-regression rows must never replace the frozen S1 input row passed
    # to the object-mask backends.
    assert "row=selected_row" in source
    assert "regression_row = run_role_regression" in source


def test_clean_gate_requires_zero_pending_and_no_clean_active() -> None:
    assert clean_is_closed({"waves": {"wave0_clean_pending": 0}, "active_tasks": []})
    assert not clean_is_closed({"waves": {"wave0_clean_pending": 1}, "active_tasks": []})
    assert not clean_is_closed({
        "waves": {"wave0_clean_pending": 0},
        "active_tasks": [{"task_id": "exact78_wave0_clean_runtime_recovery_v53"}],
    })


def test_payload_status_accepts_wrapped_and_plain_receipts() -> None:
    assert payload_status({"payload": {"status": "PASSED"}}) == "PASSED"
    assert payload_status({"status": "BLOCKED_RESOURCE"}) == "BLOCKED_RESOURCE"


def test_role_bundle_requires_canary_and_two_regressions(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("chaoyang.ops.run_post_clean_s1_automation_v71.ROLE_REPAIR_ROOT", tmp_path)
    rows = []
    for session in ("play_cards_0901_042", "play_cards_0903_189", "play_cards_0903_202"):
        path = tmp_path / f"{session}.json"
        path.write_text('{"session_id":"' + session + '","status":"PASSED"}')
        rows.append({"status": "PASSED", "result": str(path)})
    complete = publish_role_bundle([
        *rows,
    ])
    assert complete["status"] == "PASSED"
