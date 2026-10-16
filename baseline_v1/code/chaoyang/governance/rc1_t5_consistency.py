"""Check that the RC1 current projection agrees with its immutable T5 receipt."""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


from typing import Any, Mapping

from chaoyang.governance.common import load_json, validate_artifact_ref
from pathlib import Path


T5_TASK_ID = "rc1_t5_batch_conversion"


def t5_release_flags(task_state: Mapping[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    rows = [row for row in task_state.get("tasks", []) if row.get("task_id") == T5_TASK_ID]
    if not rows:
        return None, []
    if len(rows) != 1:
        return None, ["RC1 T5 task is duplicated"]
    row = rows[0]
    if row.get("status") != "PASSED":
        return None, []
    result_ref = row.get("result")
    if not isinstance(result_ref, dict):
        return None, ["RC1 T5 PASSED task has no result reference"]
    errors = validate_artifact_ref(result_ref)
    if errors:
        return None, errors
    result = load_json(Path(result_ref["path"]))
    if result.get("task_id") != T5_TASK_ID or result.get("status") != "PASSED":
        return None, ["RC1 T5 result contradicts PASSED task"]
    summary_ref = result.get("quality_summary")
    if not isinstance(summary_ref, dict):
        return None, ["RC1 T5 result has no quality summary reference"]
    errors = validate_artifact_ref(summary_ref)
    if errors:
        return None, errors
    summary = load_json(Path(summary_ref["path"]))
    flags = summary.get("release_flags")
    if not isinstance(flags, dict):
        return None, ["RC1 T5 quality summary has no release flags"]
    if flags.get("RC1_RELEASE_STATUS") != result.get("release_status"):
        errors.append("RC1 T5 result/summary release status differs")
    if flags.get("RATE_FINALIZED") != result.get("rate_finalized"):
        errors.append("RC1 T5 result/summary rate_finalized differs")
    for task, key in (("chips", "CHECKPOINT_PAIR_CHIPS"), ("poker", "CHECKPOINT_PAIR_POKER")):
        if flags.get(key) != task_state.get("rc1_pair_status", {}).get(task):
            errors.append(f"RC1 T5 {task} pair status differs from task state")
    return flags, errors


def validate_t5_current_flags(task_state: Mapping[str, Any]) -> list[str]:
    flags, errors = t5_release_flags(task_state)
    if flags is None or errors:
        return errors
    current = task_state.get("rc1_release_flags", {})
    for key, expected in flags.items():
        if current.get(key) != expected:
            errors.append(f"RC1 current flag {key} differs from T5: current={current.get(key)!r} T5={expected!r}")
    return errors
