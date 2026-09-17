import pytest

from chaoyang.cli import _maintained_operations, _run_module


def test_contract_exposes_current_operation():
    assert "validate_pipeline_contracts_r3" in _maintained_operations()


def test_unregistered_historical_operation_is_rejected():
    assert "build_rc1_unblock_handoff_v2" not in _maintained_operations()
    with pytest.raises(SystemExit, match="not maintained"):
        _run_module("build_rc1_unblock_handoff_v2", [])
