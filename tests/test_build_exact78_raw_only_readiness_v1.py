from __future__ import annotations

import json
from pathlib import Path

import pytest

from chaoyang.ops.build_exact78_raw_only_readiness_v1 import assess


def _write(path: Path, value: dict | str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, str):
        path.write_text(value, encoding="utf-8")
    else:
        path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _inputs(tmp_path: Path) -> dict[str, Path]:
    return {
        "run_signature_path": _write(
            tmp_path / "RUN_SIGNATURE.json",
            {"task_id": "four_stream_pretraining_baseline_v32", "t0": "2026-09-20T19:04:09+08:00"},
        ),
        "rebind_path": _write(
            tmp_path / "REBIND.json",
            {"status": "PASS", "frozen_member_count": 156, "resolved_member_count": 156, "unresolved_members": []},
        ),
        "pair_result_path": _write(tmp_path / "PAIR.json", {"execution_allowed": False}),
        "funnel_result_path": _write(tmp_path / "FUNNEL.json", {"status": "BLOCKED_INPUTS"}),
        "initializer_result_path": _write(
            tmp_path / "INITIALIZER.json",
            {
                "accepted_state_scope": "HAND_Q22_INITIALIZATION_ONLY",
                "arm_admission": {"status": "BLOCKED_UNOBSERVED_INSTALLATION_GEOMETRY"},
            },
        ),
        "trainer_path": _write(tmp_path / "trainer.py", "--paired-ledger validate_paired_ledgers"),
        "bundle_builder_path": _write(tmp_path / "bundle.py", "robotized_rgb"),
    }


def test_readiness_refuses_early_publication(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match=r"before T\+4"):
        assess(**_inputs(tmp_path), observed_at="2026-09-20T22:00:00+08:00")


def test_readiness_blocks_semantic_and_interface_substitution(tmp_path: Path) -> None:
    value = assess(**_inputs(tmp_path), observed_at="2026-09-20T23:04:09+08:00")
    assert value["status"] == "BLOCKED_FAIL_CLOSED"
    assert value["execution_allowed"] is False
    assert value["cohort_closed"] is True
    assert "ABSOLUTE_ROBOT_HAND_ROOT_WORLD_TARGET_ABSENT" in value["blockers"]
    assert "RAW_ONLY_LEDGER_AND_TRAINER_ENTRY_ABSENT" in value["blockers"]
    assert value["authority"]["training_complete"] is False
