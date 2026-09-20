from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from chaoyang.ops import run_exact78_development_pair_audit_v31 as subject


def dump(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def make_lane(tmp_path: Path) -> Path:
    lane = tmp_path / "lanes" / "exact78"
    lane.mkdir(parents=True)
    dump(
        lane / "STATE.json",
        {
            "schema_version": "chaoyang-three-stream-lane-state-v1",
            "parent_task_id": subject.TASK_ID,
            "lane": "exact78",
            "status": "READY_CPU_PREFLIGHT",
            "current_action": "CPU_PREFLIGHT_NOT_STARTED",
            "writer_root": str(lane.resolve()),
        },
    )
    return lane


def published_result(output: Path, value: dict) -> dict:
    dump(output / "RESULT.json", value)
    return value


def test_expected_e0_e1_blockers_are_honest_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lane = make_lane(tmp_path)
    order: list[str] = []

    def e0(**kwargs):
        order.append("E0")
        return published_result(
            kwargs["output_root"],
            {
                "status": "BLOCKED_INPUTS",
                "execution_allowed": False,
                "blockers": ["MISSING_CURRENT_NON_ARCHIVE_156_COHORT_MANIFEST"],
            },
        )

    def e1(session_root, output_root, *, encoded_calibration):
        order.append("E1")
        assert encoded_calibration is None
        return published_result(
            output_root,
            {
                "status": "BLOCKED_EXTERNAL_ASSET",
                "execution_allowed": False,
                "blockers": ["MISSING_ENCODED_DOMAIN_K_P_BASELINE_AUTHORITY"],
            },
        )

    monkeypatch.setattr(subject.e0_builder, "build", e0)
    monkeypatch.setattr(subject.e1_builder, "build", e1)
    result = subject.run(
        lane_root=lane,
        data_root=tmp_path / "data",
        e1_session_root=tmp_path / "chips023",
        cohort_manifest=None,
        candidate_index=None,
        encoded_calibration=None,
    )

    assert order == ["E0", "E1"]
    assert result["status"] == "BLOCKED_INPUTS"
    assert subject.terminal_exit_code(result) == 0
    assert result["foundationstereo_inference_performed"] is False
    assert result["gpu_used"] is False
    assert result["blockers"] == [
        "E0_CURRENT_DEVELOPMENT_PAIRS:MISSING_CURRENT_NON_ARCHIVE_156_COHORT_MANIFEST",
        "E1_STEREO_PREFLIGHT_INPUT:MISSING_ENCODED_DOMAIN_K_P_BASELINE_AUTHORITY",
    ]
    assert (lane / subject.OUTPUT_ROOT.name / "RESULT.json").is_file()


def test_runtime_failure_is_distinct_and_e1_still_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lane = make_lane(tmp_path)
    called = []

    def fail_e0(**kwargs):
        raise ValueError("broken inventory")

    def e1(session_root, output_root, *, encoded_calibration):
        called.append("E1")
        return published_result(
            output_root,
            {
                "status": "BLOCKED_EXTERNAL_ASSET",
                "execution_allowed": False,
                "blockers": ["MISSING_ENCODED_DOMAIN_K_P_BASELINE_AUTHORITY"],
            },
        )

    monkeypatch.setattr(subject.e0_builder, "build", fail_e0)
    monkeypatch.setattr(subject.e1_builder, "build", e1)
    result = subject.run(
        lane_root=lane,
        data_root=tmp_path / "data",
        e1_session_root=tmp_path / "chips023",
        cohort_manifest=None,
        candidate_index=None,
        encoded_calibration=None,
    )

    assert called == ["E1"]
    assert result["status"] == "FAILED_RUNTIME"
    assert subject.terminal_exit_code(result) == 1
    assert result["runtime_errors"][0].startswith(
        "E0_CURRENT_DEVELOPMENT_PAIRS:RUNTIME:ValueError"
    )


def test_lane_state_must_match_parent_writer_and_readiness(tmp_path: Path) -> None:
    lane = make_lane(tmp_path)
    state = json.loads((lane / "STATE.json").read_text())
    state["writer_root"] = str(tmp_path / "other")
    dump(lane / "STATE.json", state)

    with pytest.raises(subject.Exact78LaneAuditError, match="identity/readiness"):
        subject.validate_lane_state(lane)


def test_wrapper_has_no_foundationstereo_runner_dependency() -> None:
    tree = ast.parse(Path(subject.__file__).read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "run_exact78_stereo_preflight_v31" not in imported
    assert all("foundationstereo" not in name.lower() for name in imported)
