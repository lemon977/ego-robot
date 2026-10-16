"""Read-only post-seal evaluator; old Robot is never a producer input."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from chaoyang.governance.exact78_cold_start_guard_v1 import PROJECT_ROOT, file_ref


OLD_OUTER = (PROJECT_ROOT / "_run/current/exact78_newraw_robot_play_cards_0902_001_9541579d8de29395_v1"
             "/attempts/attempt_0001/RESULT.json")
OLD_OUTER_SHA = "0db86e80ef251cb10cb3e3c56ee886afdea2f6462278e42f98c530e625aae6ac"
OLD_STAGE_SHA = "cc9b87a947bdc945fbfece7c96efa16d115c2fcc3560913565138d6b3553e2a1"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("TWO_SEED_EVALUATOR_JSON_OBJECT_REQUIRED")
    return value


def _verified(ref: Any) -> Path:
    if not isinstance(ref, dict) or not isinstance(ref.get("path"), str):
        raise RuntimeError("TWO_SEED_EVALUATOR_REF_REQUIRED")
    path = Path(ref["path"])
    if file_ref(path) != ref:
        raise RuntimeError("TWO_SEED_EVALUATOR_REF_SHA_DRIFT")
    return path


def _new_terminal(outer_path: Path) -> dict[str, Any]:
    outer_path = outer_path.resolve(strict=True)
    if not outer_path.is_relative_to(PROJECT_ROOT / "_run/current"):
        raise RuntimeError("TWO_SEED_EVALUATOR_NEW_OUTER_SCOPE")
    outer = _json(outer_path)
    task_id = outer.get("task_id")
    if (not isinstance(task_id, str) or not task_id.startswith("exact78_newraw_two_seed_" )
            or outer.get("status") != "PASSED"
            or outer.get("source_session") != "play_cards_0902_001"):
        raise RuntimeError("TWO_SEED_EVALUATOR_NEW_NOT_TERMINAL_PASSED")
    receipt_path = PROJECT_ROOT / "tasks/receipts" / f"{task_id.upper()}.json"
    receipt = _json(receipt_path)
    if (receipt.get("task_id") != task_id or receipt.get("status") != "PASSED"
            or receipt.get("result") != file_ref(outer_path)):
        raise RuntimeError("TWO_SEED_EVALUATOR_RECEIPT_DRIFT")
    inner_path = _verified((outer.get("stage") or {}).get("result"))
    if not inner_path.is_relative_to(outer_path.parent / "candidate/two_seed_cpu"):
        raise RuntimeError("TWO_SEED_EVALUATOR_INNER_SCOPE")
    inner = _json(inner_path)
    if (inner.get("schema_version") != "exact78-newraw-two-seed-cpu-result-v1"
            or inner.get("status") != "COMPLETE_CPU_NUMERIC_DIAGNOSTIC_NOT_VISUAL_PASS"
            or inner.get("old_robot_result_input") is not False
            or inner.get("frame_count") != 201):
        raise RuntimeError("TWO_SEED_EVALUATOR_INNER_DRIFT")
    return inner


def _same_number(left: Any, right: Any, tolerance: float = 1e-6) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return abs(float(left) - float(right)) <= tolerance


def compare_sealed(outer_path: Path) -> dict[str, Any]:
    """Read old Robot only after independent new outer result and receipt seal."""
    inner = _new_terminal(outer_path)
    outputs = inner["outputs"]
    new_states_path = _verified(outputs["NUMERIC_STATES.npz"])
    shadow_path = _verified(outputs["SHADOW_BASELINE.json"])
    base_path = _verified(outputs["BASE_SEARCH.json"])
    frame_path = _verified(outputs["TWO_SEED_FRAMES.json"])
    if file_ref(OLD_OUTER)["sha256"] != OLD_OUTER_SHA:
        raise RuntimeError("TWO_SEED_EVALUATOR_OLD_OUTER_SHA_DRIFT")
    old_outer = _json(OLD_OUTER)
    old_stage_path = _verified(old_outer["stage"]["stage_result"])
    if file_ref(old_stage_path)["sha256"] != OLD_STAGE_SHA:
        raise RuntimeError("TWO_SEED_EVALUATOR_OLD_STAGE_SHA_DRIFT")
    old_stage = _json(old_stage_path)
    old_states_path = _verified(old_stage["outputs"]["states"])
    old_rows_path = _verified(old_stage["outputs"]["frame_states"])
    old_base_path = _verified(old_stage["outputs"]["base_search"])
    with np.load(new_states_path, allow_pickle=False) as new, np.load(old_states_path, allow_pickle=False) as old:
        q_equal = bool(np.allclose(new["q_arm_shadow"], old["q_arm"], rtol=0, atol=1e-8))
        hand_equal = bool(np.allclose(new["q_hand"], old["q_hand"], rtol=0, atol=1e-8))
        base_equal = bool(np.allclose(new["T_camera_base"], old["T_camera_base"], rtol=0, atol=1e-10))
    base_report_equal = _json(base_path) == _json(old_base_path)
    shadow = _json(shadow_path)["rows"]
    old_rows = _json(old_rows_path)["rows"]
    if len(shadow) != 402 or len(old_rows) != 402:
        raise RuntimeError("TWO_SEED_EVALUATOR_ROW_COUNT_DRIFT")
    row_drift = []
    for new, old in zip(shadow, old_rows, strict=True):
        identity = (new["frame"], new["side"])
        if identity != (old["frame"], old["side"]):
            row_drift.append({"identity": identity, "reason": "FRAME_SIDE"})
            continue
        for component, fields in (("arm", ("status", "pass_numeric", "position_mm",
                                           "rotation_deg", "branch_margin_m_min")),
                                  ("hand", ("status", "pass_numeric", "bone_error_deg_max",
                                            "tip_direction_error_deg_max"))):
            for field in fields:
                a, b = new[component].get(field), old[component].get(field)
                same = (_same_number(a, b) if field not in {"status", "pass_numeric"}
                        else a == b)
                if not same:
                    row_drift.append({"identity": identity, "component": component,
                                      "field": field})
    replay_valid = q_equal and hand_equal and base_equal and base_report_equal and not row_drift
    candidate_rows = _json(frame_path)["rows"]
    if len(candidate_rows) != 402:
        raise RuntimeError("TWO_SEED_EVALUATOR_CANDIDATE_ROW_COUNT_DRIFT")
    original_pass = {side: sum(row["arm"]["pass_numeric"] for row in old_rows
                               if row["side"] == side) for side in ("left", "right")}
    candidate_pass = {side: sum(row["pass_numeric"] for row in candidate_rows
                                if row["side"] == side) for side in ("left", "right")}
    return {"status": "VALID_EXPERIMENT" if replay_valid else "INVALID_DEFAULT_REPLAY_DRIFT",
            "shadow_q_equal": q_equal, "shadow_hand_equal": hand_equal,
            "base_equal": base_equal, "base_report_equal": base_report_equal,
            "row_drift_count": len(row_drift), "row_drift_first_five": row_drift[:5],
            "old_arm_pass": original_pass, "candidate_arm_pass": candidate_pass,
            "worst_side_improved": replay_valid and
                min(candidate_pass.values()) > min(original_pass.values()),
            "visual_qualified": False, "user_review": "PENDING"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sealed-new-outer-result", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(compare_sealed(args.sealed_new_outer_result), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
