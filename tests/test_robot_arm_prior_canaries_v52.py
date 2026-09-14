from __future__ import annotations

import pytest

from tools import run_exact78_robot_arm_prior_canaries_v52 as prior


def test_override_must_be_exact_and_registered(monkeypatch: pytest.MonkeyPatch) -> None:
    override = {"path": "/abs/v3.py", "bytes": 3, "sha256": "a" * 64}
    monkeypatch.setattr(prior, "verify_ref", lambda value, label: prior.Path("/abs/v3.py"))
    monkeypatch.setattr(
        prior,
        "ref",
        lambda path: {"path": str(path), "bytes": 3, "sha256": "a" * 64},
    )
    assert prior.select_arm_tool(
        {"arm_forward_tool": override, "programs": {"v3.py": override}}
    ) == prior.Path("/abs/v3.py")
    with pytest.raises(ValueError, match="program closure"):
        prior.select_arm_tool({"arm_forward_tool": override, "programs": {}})


def test_only_exact_empty_bounds_is_deterministic_quality_hold() -> None:
    exact = "tools.render.TemporalReviewError: empty previous-accepted bounds: max_excess=0.01"
    assert prior.deterministic_bounds_failure(1, False, exact)
    assert not prior.deterministic_bounds_failure(0, False, exact)
    assert not prior.deterministic_bounds_failure(1, True, exact)
    assert not prior.deterministic_bounds_failure(1, False, "MemoryError")


def test_command_contract_ignores_only_output_path() -> None:
    first = ["python", "tool.py", "--input", "x", "--output", "old/result.json", "--gain", "1"]
    second = ["python", "tool.py", "--input", "x", "--output", "new/result.json", "--gain", "1"]
    assert prior.command_without_output(first) == prior.command_without_output(second)
    with pytest.raises(ValueError, match="exactly one"):
        prior.command_without_output(["python", "tool.py"])
