from pathlib import Path

from chaoyang.ops.run_v71_post_s1_finalize import development_task_status


def test_development_results_do_not_imply_authority_but_close_task() -> None:
    assert development_task_status("PASSED") == "PASSED"
    assert development_task_status("COMPARISON_COMPLETE_REVIEW_REQUIRED") == "PASSED"
    assert development_task_status("NO_GO_BACKEND_EXECUTION") == "FAILED_QUALITY_C"
    assert development_task_status("BLOCKED_RESOURCE") == "BLOCKED_RESOURCE"
    assert development_task_status("unexpected") == "FAILED_RUNTIME_FINAL"


def test_finalizer_uses_governance_module_entrypoint() -> None:
    source = Path("src/chaoyang/ops/run_v71_post_s1_finalize.py").read_text(encoding="utf-8")
    assert '"-m", "chaoyang.governance.update_task_state"' in source
    assert "no Mask/Robot/Contact/Gold/physical authority promotion" in source
