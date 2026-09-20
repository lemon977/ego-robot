from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from chaoyang.ops import build_0915_robot_recovery_v21_f0 as f0


def dump(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    root = tmp_path / "campaign"
    identity = {"task_id": f0.TASK_ID, "window_run_id": f0.RUN_ID}
    for name in f0.REQUIRED_PARENT_INPUTS:
        dump(root / name, dict(identity))
    dump(root / "W0_FAILURE_MATRIX_V2.json", {
        **identity,
        "strict_session_passed": 0,
        "r0_quality_admitted": 0,
    })
    dump(root / "CANDIDATE_FREEZE_V1.json", {**identity, "candidate_id": "candidate"})
    dump(root / "CONTACT_OBSERVABILITY_AUDIT_V1.json", {
        **identity,
        "pair_rows": 5040,
        "direct_visible_finger_surface_rows": 483,
        "unique_visible_surface_sample_keys": 161,
        "metric_pair_rows": 251,
        "inside_finite_patch_rows": 0,
        "within_fixed_5mm_rows": 0,
        "pixel_registration_bound_rows": 0,
    })
    dump(root / "LOCAL_STEREO_METRIC_DEV_V1.json", {
        **identity,
        "local_stereo_metric_dev": False,
        "r1_e_development_allowed": False,
    })

    package_values = {
        "P0": {},
        "A0": {},
        "A1": {"counts": {"session_strict_pass": 0}},
        "A2": {"counts": {"session_strict_pass": 0}, "r0_quality_upgraded": False},
        "A3_R0/A1": {},
        "A3_R0/A2": {},
        "A4_MOTION_GATE_AUTHORITY_AUDIT": {},
        "A5_KAI22_SATURATION_AUDIT": {},
        "A6_THUMB_BOUNDED_IK_REJECTED": {},
        "A7_STOP_NO_AUTHORITY_AUDIT": {},
        "B0": {},
        "B1": {},
        "B1R": {},
        "B2_DEPTH_TO_OBJECT_POKER044": {},
        "B2_CONTACT_AUTHORITY_SEPARATION_AUDIT": {},
        "B3_FINAL_TERMINAL_AUDIT": {
            "status": "FAILED_RUNTIME_FINAL_TERMINAL",
            "decision": "STOP_NO_THIRD_ATTEMPT",
            "attempts": [{}, {}],
            "authority_promoted": False,
        },
        "C0": {},
        "C0_v2": {},
        "C1": {},
        "C1_SELF_CHECK": {},
        "D1_CLEAN_PREP/PREPARED_V1": {},
        "D2_FRESH_FILL_OFFLINE_V1": {
            "status": "COMPLETED_OFFLINE_VISUAL_CANDIDATE_UNKNOWN_RETAINED",
            "model": {"executed_for_sessions": ["get_potato_chips_0915_097"]},
            "clean_terminal": False,
            "training_eligible": False,
        },
        "D2_HARDENED_INDEPENDENT_AUDIT": {},
    }
    for package, value in package_values.items():
        dump(root / f"packages/{package}/RESULT.json", {**identity, **value})

    dump(root / "COORDINATOR_STATE.json", {
        **identity,
        "deadline_at": "2026-09-20T14:44:12+08:00",
    })
    exit_path = tmp_path / "COORDINATOR_EXIT.json"
    dump(exit_path, {
        **identity,
        "status": "MONITOR_EXITED_NOT_CAMPAIGN_FINALIZATION",
        "deadline_reached": True,
    })
    verification = tmp_path / "VERIFICATION.json"
    dump(verification, {
        "status": "PASS",
        "checks": {
            **{name: "PASS" for name in f0.REQUIRED_VERIFICATION_CHECKS},
            **{name: f0.DEFERRED_POST_CAS_STATUS for name in f0.DEFERRED_POST_CAS_CHECKS},
        },
        "evidence": [],
    })
    receipt = tmp_path / "TERMINAL_RECEIPT.json"
    return root, exit_path, verification, receipt


def run(monkeypatch: pytest.MonkeyPatch, root: Path, exit_path: Path, verification: Path,
        receipt: Path, now: str = "2026-09-20T14:45:00+08:00") -> int:
    monkeypatch.setattr(sys, "argv", [
        "build_0915_robot_recovery_v21_f0",
        "--campaign-root", str(root),
        "--coordinator-exit", str(exit_path),
        "--verification-report", str(verification),
        "--terminal-receipt", str(receipt),
        "--now", now,
    ])
    return f0.main()


def test_builds_rejected_no_recovery_closure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, exit_path, verification, receipt = fixture(tmp_path)
    assert run(monkeypatch, root, exit_path, verification, receipt) == 0
    assert json.loads((root / "ADOPTION_DECISION_V1.json").read_text())["decision"] == "REJECT"
    final_audit = json.loads((root / "FINAL_AUDIT.json").read_text())
    assert final_audit["status"] == "PASS_PRE_CAS"
    assert final_audit["post_cas_governance_audit_required"] is True
    result = json.loads((root / "RESULT.json").read_text())
    assert result["status"] == "REJECTED_QUALITY"
    assert result["campaign_classification"] == "REJECTED_NO_RECOVERY"
    assert receipt.is_file()


def test_refuses_before_deadline_without_partial_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, exit_path, verification, receipt = fixture(tmp_path)
    with pytest.raises(RuntimeError, match="before the deadline without a terminal stop request"):
        run(monkeypatch, root, exit_path, verification, receipt, "2026-09-20T14:00:00+08:00")
    assert not (root / "ADOPTION_DECISION_V1.json").exists()


def test_builds_before_deadline_with_terminal_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, exit_path, verification, receipt = fixture(tmp_path)
    dump(root / "STOP_REQUESTED.json", {
        "task_id": f0.TASK_ID,
        "window_run_id": f0.RUN_ID,
        "requested_at": "2026-09-20T10:57:09+08:00",
        "reason": "ALL_STARTED_PACKAGES_TERMINAL",
    })
    dump(exit_path, {
        "task_id": f0.TASK_ID,
        "window_run_id": f0.RUN_ID,
        "status": "MONITOR_EXITED_NOT_CAMPAIGN_FINALIZATION",
        "at": "2026-09-20T10:57:51+08:00",
        "deadline_reached": False,
    })
    assert run(
        monkeypatch, root, exit_path, verification, receipt,
        "2026-09-20T10:58:00+08:00",
    ) == 0
    final_audit = json.loads((root / "FINAL_AUDIT.json").read_text())
    assert final_audit["stop_condition"] == "ALL_STARTED_PACKAGES_TERMINAL"
    assert final_audit["stop_request"]["path"].endswith("STOP_REQUESTED.json")


def test_refuses_incomplete_verification(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, exit_path, verification, receipt = fixture(tmp_path)
    dump(verification, {"status": "PASS", "checks": {}, "evidence": []})
    with pytest.raises(RuntimeError, match="verification report is incomplete"):
        run(monkeypatch, root, exit_path, verification, receipt)


def test_refuses_to_hide_new_strict_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, exit_path, verification, receipt = fixture(tmp_path)
    path = root / "packages/A2/RESULT.json"
    value = json.loads(path.read_text())
    value["counts"]["session_strict_pass"] = 1
    dump(path, value)
    with pytest.raises(RuntimeError, match="W1-ADOPTION evidence"):
        run(monkeypatch, root, exit_path, verification, receipt)
