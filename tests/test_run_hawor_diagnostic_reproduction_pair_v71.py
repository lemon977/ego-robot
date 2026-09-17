from chaoyang.ops.run_hawor_diagnostic_reproduction_pair_v71 import execution_complete, summarize


def test_summary_keeps_numeric_pass_and_hold_separate():
    result = summarize({
        "chips": {"rows": [{"status": "PASS_NUMERIC_NEEDS_HUMAN_REVIEW"}, {"status": "HOLD_NUMERIC_GATES"}]},
        "poker": {"rows": [{"status": "PASS_NUMERIC_NEEDS_HUMAN_REVIEW"}]},
    })
    assert result["contracts"] == 2
    assert result["canaries"] == 3
    assert result["numeric_review_ready"] == 2
    assert result["numeric_hold"] == 1


def test_summary_reads_real_canaries_key_and_exposes_runtime_exception():
    result = summarize({
        "chips": {"canaries": [{"status": "HOLD_EXCEPTION"}]},
        "poker": {"canaries": [{"status": "PASS_NUMERIC_NEEDS_HUMAN_REVIEW"}]},
    })
    assert result["contracts"] == 2
    assert result["canaries"] == 2
    assert result["runtime_exception"] == 1


def test_quality_hold_return_code_is_a_completed_diagnostic():
    results = {
        "chips": {"canaries": [{"status": "HOLD_NUMERIC_GATES"}]},
        "poker": {"canaries": [{"status": "PASS_NUMERIC_NEEDS_HUMAN_REVIEW"}]},
    }
    runtime = {"chips": {"return_code": 2}, "poker": {"return_code": 2}}
    assert execution_complete(results, runtime, expected_canaries=2)


def test_runtime_exception_is_not_a_completed_diagnostic():
    results = {
        "chips": {"canaries": [{"status": "HOLD_EXCEPTION"}]},
        "poker": {"canaries": [{"status": "PASS_NUMERIC_NEEDS_HUMAN_REVIEW"}]},
    }
    runtime = {"chips": {"return_code": 2}, "poker": {"return_code": 0}}
    assert not execution_complete(results, runtime, expected_canaries=2)
