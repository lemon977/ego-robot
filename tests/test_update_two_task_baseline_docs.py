from __future__ import annotations

from tools import update_two_task_baseline_docs as updater


def test_render_contains_all_stage_rows() -> None:
    stages = {name: {"chips": f"C_{name}", "poker": f"P_{name}"} for name in updater.ORDER}
    output = updater.render({"updated_at": "2026-09-08T17:00:00+08:00", "stages": stages})
    assert output.count(updater.BEGIN) == 1
    assert output.count(updater.END) == 1
    for name in updater.ORDER:
        assert f"C_{name}" in output and f"P_{name}" in output
