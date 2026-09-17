from chaoyang.ops.run_occlusion_silver_governance_finalizer_v71 import terminal_task_status


def test_unclosed_silver_contract_closes_as_quality_c():
    assert terminal_task_status({"status": "FAILED_QUALITY_C"}) == "FAILED_QUALITY_C"


def test_future_closed_silver_can_pass_without_code_change():
    assert terminal_task_status({"status": "PASSED"}) == "PASSED"
