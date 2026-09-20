from __future__ import annotations

import hashlib
import json
from pathlib import Path

from chaoyang.governance.build_three_stream_terminal_status_v31 import (
    build_terminal_status,
)


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _artifact(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": str(path),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def test_terminal_status_keeps_runtime_fixes_separate_from_quality(tmp_path: Path) -> None:
    parent_path = tmp_path / "parent.json"
    exact_path = tmp_path / "exact.json"
    ai1_receipt_path = tmp_path / "ai1_receipt.json"
    ai1_result_path = tmp_path / "ai1_result.json"
    ai2_receipt_path = tmp_path / "ai2_receipt.json"
    ai2_result_path = tmp_path / "ai2_result.json"
    rebind_receipt_path = tmp_path / "rebind_receipt.json"
    rebind_result_path = tmp_path / "rebind_result.json"
    for path in (
        parent_path,
        exact_path,
        ai1_receipt_path,
        ai1_result_path,
        ai2_receipt_path,
        ai2_result_path,
        rebind_receipt_path,
        rebind_result_path,
    ):
        _write_json(path, {"placeholder": path.name})

    value = build_terminal_status(
        parent_receipt={
            "task_id": "three_stream_stable_baseline_v31",
            "status": "REJECTED_QUALITY",
            "pipeline_complete": False,
            "training_complete": False,
            "training_eligible": False,
        },
        parent_receipt_path=parent_path,
        exact_result={
            "status": "BLOCKED_INPUTS",
            "blockers": ["MISSING_CURRENT_PAIR_CANDIDATE_INDEX"],
            "stages": {"E0_CURRENT_DEVELOPMENT_PAIRS": {"execution_allowed": False}},
            "gpu_used": False,
            "model_inference_performed": False,
        },
        exact_result_path=exact_path,
        ai1_receipt={"status": "BLOCKED_PREREQ"},
        ai1_receipt_path=ai1_receipt_path,
        ai1_result={
            "status": "BLOCKED_PREREQ",
            "machine_status": "BLOCKED_ADOPTION_OBSERVATIONS",
            "gpu_calls": 0,
            "model_calls": 0,
        },
        ai1_result_path=ai1_result_path,
        ai2_receipt={"status": "BLOCKED_PREREQ"},
        ai2_receipt_path=ai2_receipt_path,
        ai2_result={
            "status": "BLOCKED_PREREQ",
            "machine_status": "BLOCKED_AI2_EVIDENCE",
            "counts": {"total": 8, "blocked": 8, "rejected": 0, "pass": 0},
            "all_existing_inputs_sha_bound": True,
            "all_sessions_terminal": True,
            "gpu_calls": 0,
            "model_calls": 0,
        },
        ai2_result_path=ai2_result_path,
        ai1_rebind_receipt={"status": "PASS_REFERENCE_REBIND"},
        ai1_rebind_receipt_path=rebind_receipt_path,
        ai1_rebind_result={"status": "PASS_REFERENCE_REBIND"},
        ai1_rebind_result_path=rebind_result_path,
        generated_at="2026-09-20T00:00:00+08:00",
    )

    assert value["lanes"]["exact78"]["training_started"] is False
    assert value["lanes"]["ai1"]["status"] == "BLOCKED_ADOPTION_OBSERVATIONS"
    assert len(value["lanes"]["ai1"]["runtime_corrections"]) == 2
    assert value["lanes"]["ai2"]["counts"]["blocked"] == 8
    assert value["final_claims"]["NUMERIC_QUALITY_PASS"] is False
    assert value["final_claims"]["TRAINING_ELIGIBLE"] is False
    assert value["resource_use"]["gpu_used"] is False


def test_terminal_status_without_optional_rebind_is_still_fail_closed(tmp_path: Path) -> None:
    files = {name: tmp_path / f"{name}.json" for name in ("parent", "exact", "ai1r", "ai1", "ai2r", "ai2")}
    for path in files.values():
        _write_json(path, {"placeholder": path.name})
    value = build_terminal_status(
        parent_receipt={"status": "REJECTED_QUALITY"},
        parent_receipt_path=files["parent"],
        exact_result={"status": "BLOCKED_INPUTS", "stages": {}},
        exact_result_path=files["exact"],
        ai1_receipt={"status": "BLOCKED_PREREQ"},
        ai1_receipt_path=files["ai1r"],
        ai1_result={"status": "BLOCKED_PREREQ"},
        ai1_result_path=files["ai1"],
        ai2_receipt={"status": "BLOCKED_PREREQ"},
        ai2_receipt_path=files["ai2r"],
        ai2_result={"status": "BLOCKED_PREREQ", "counts": {}},
        ai2_result_path=files["ai2"],
        ai1_rebind_receipt=None,
        ai1_rebind_receipt_path=None,
        ai1_rebind_result=None,
        ai1_rebind_result_path=None,
        generated_at="2026-09-20T00:00:00+08:00",
    )
    assert value["source"]["runtime_successors"]["ai1_artifact_ref_rebind"] is None
    assert value["final_claims"]["PIPELINE_COMPLETE"] is False
