from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jsonschema
import pytest

from chaoyang.ops import run_ai2_real_assets_cohort_audit_v31 as op


ROOT = Path(__file__).resolve().parents[1]


def _prepare_lane(root: Path) -> Path:
    lane = (
        root
        / "_run/current/three_stream_stable_baseline_v31/attempts/attempt_0001/lanes/ai2"
    )
    lane.mkdir(parents=True)
    (lane / "STATE.json").write_text(
        json.dumps(
            {
                "lane": "ai2",
                "parent_task_id": "three_stream_stable_baseline_v31",
                "status": "READY_CPU_PREFLIGHT",
            }
        ),
        encoding="utf-8",
    )
    return lane


def _blocked_audit(session_id: str) -> dict:
    return {
        "schema_version": "AI2_REAL_ASSETS_AUDIT_V31",
        "status": "COMPLETED_FAIL_CLOSED_AUDIT",
        "session_id": session_id,
        "blocker_codes": [
            "BLOCKED_INDEPENDENT_PART_OBSERVABILITY",
            "BLOCKED_INDEPENDENT_REPROJECTION",
            "BLOCKED_SUFFIX_PAIR_MATERIALIZATION",
        ],
        "kai22_tier": {
            "sides": {
                "left": {"highest_admitted_level": "KINEMATIC_ONLY"},
                "right": {"highest_admitted_level": "NONE"},
            }
        },
        "model_calls": 0,
        "gpu_calls": 0,
    }


def _schema_validate(value: dict) -> None:
    schema = json.loads(
        (ROOT / "contracts" / "ai2_real_assets_cohort_audit_v31.schema.json").read_text(
            encoding="utf-8"
        )
    )
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(value)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_cohort_runner_publishes_eight_sha_bound_fail_closed_terminals(
    tmp_path: Path,
) -> None:
    lane = _prepare_lane(tmp_path)
    calls: list[dict] = []

    def builder(**kwargs: object) -> dict:
        calls.append(dict(kwargs))
        return _blocked_audit(str(kwargs["session_id"]))

    result = op.run_cohort(root=tmp_path, audit_builder=builder)

    _schema_validate(result)
    assert result["status"] == "BLOCKED_AI2_EVIDENCE"
    assert result["counts"] == {"total": 8, "pass": 0, "blocked": 8, "rejected": 0}
    assert result["all_sessions_terminal"] is True
    assert result["all_existing_inputs_sha_bound"] is True
    assert result["model_calls"] == result["gpu_calls"] == 0
    assert len(calls) == len(op.SESSION_SPECS) == 8
    assert [call["cohort"] for call in calls] == [
        "W0",
        "W0",
        "W0",
        "W0",
        "A1",
        "A1",
        "A2",
        "A2",
    ]
    assert all(call["bounded_npz"] is not None for call in calls[:4])
    assert all(call["sam_result"] is not None for call in calls[:4])
    assert all(call["bounded_npz"] is None for call in calls[4:])
    assert all(call["sam_result"] is None for call in calls[4:])

    output = lane / op.OUTPUT_NAME
    for row in result["sessions"]:
        terminal_path = Path(row["result"]["path"])
        assert terminal_path.is_file()
        assert terminal_path.stat().st_size == row["result"]["bytes"]
        assert _sha(terminal_path) == row["result"]["sha256"]
        terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
        audit_path = Path(terminal["audit"]["path"])
        assert audit_path.is_file()
        assert _sha(audit_path) == terminal["audit"]["sha256"]
        assert terminal["status"] == "BLOCKED_EVIDENCE"
        assert terminal["kai22_highest_admitted_level_by_side"]["left"] == (
            "KINEMATIC_ONLY"
        )
    assert (output / "RESULT.json").is_file()
    assert not list(lane.glob(f".{op.OUTPUT_NAME}.staging-*"))


def test_one_bad_session_is_terminalized_without_hiding_other_sessions(
    tmp_path: Path,
) -> None:
    _prepare_lane(tmp_path)

    def builder(**kwargs: object) -> dict:
        if kwargs["session_id"] == "play_cards_0915_044":
            raise RuntimeError("deliberate bad SHA fixture")
        return _blocked_audit(str(kwargs["session_id"]))

    result = op.run_cohort(root=tmp_path, audit_builder=builder)

    assert result["status"] == "REJECTED_AI2_CURRENT_ASSET_AUDIT"
    assert result["counts"] == {"total": 8, "pass": 0, "blocked": 7, "rejected": 1}
    assert result["all_sessions_terminal"] is True
    assert result["all_existing_inputs_sha_bound"] is False
    rejected = [row for row in result["sessions"] if row["status"].startswith("REJECTED")]
    assert [row["session_id"] for row in rejected] == ["play_cards_0915_044"]
    terminal = json.loads(Path(rejected[0]["result"]["path"]).read_text(encoding="utf-8"))
    assert terminal["status"] == "REJECTED_INPUT_PRECONDITION"
    assert terminal["all_inputs_sha_bound"] is False
    assert terminal["audit"] is None
    _schema_validate(result)


def test_runner_refuses_to_overwrite_lane_output(tmp_path: Path) -> None:
    lane = _prepare_lane(tmp_path)
    (lane / op.OUTPUT_NAME).mkdir()
    with pytest.raises(op.CohortAuditError, match="fresh output root"):
        op.run_cohort(root=tmp_path, audit_builder=lambda **_: {})
