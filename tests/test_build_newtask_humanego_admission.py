from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pytest

from tools import build_newtask_humanego_admission as admission


PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def snapshot():
    return admission.build_snapshot(PROJECT)


def test_status_normalization_preserves_four_terminal_classes() -> None:
    assert admission.normalize_status("PASS_NUMERIC") == "PASS"
    assert admission.normalize_status("HOLD_CAPTURE") == "HOLD"
    assert admission.normalize_status("NOT_EVALUATED_MISSING_AUTHORITY") == "NOT_EVALUATED"
    assert admission.normalize_status("NOT_RUN") == "NOT_EVALUATED"
    assert admission.normalize_status("FAIL_ALGORITHM") == "FAIL"
    assert admission.normalize_status("unexpected producer token") == "NOT_EVALUATED"


def test_missing_independent_authority_is_not_algorithm_failure() -> None:
    gates = admission._identity_gates(None, None)
    assert {row["status"] for row in gates} == {"NOT_EVALUATED"}
    assert {row["owner"] for row in gates} == {"INDEPENDENT_AUTHORITY"}
    assert {row["failure_kind"] for row in gates} == {"NOT_EVALUATED"}


def test_v4_missing_labels_remains_authority_hold() -> None:
    result = {
        "gates": {
            "anatomical_identity_continuity": {"status": "HOLD"},
            "independent_contour_mask": {"status": "HOLD"},
            "expected_active_scope": {"status": "HOLD", "metrics": {}},
            "occlusion_out_of_frame": {
                "status": "PASS",
                "metrics": {"visibility_aware_denominator_authorized": False},
            },
        }
    }
    gates = admission._identity_gates(result, "v4/result.json")
    assert {row["status"] for row in gates} == {"HOLD"}
    assert {row["failure_kind"] for row in gates} == {"AUTHORITY_GAP"}


def test_terminal_snapshot_has_expected_profiles_and_no_g2_block(snapshot) -> None:
    matrix, split = snapshot
    assert len(matrix["rows"]) == 59
    assert matrix["summary"]["by_task"]["chips"]["sessions"] == 16
    assert matrix["summary"]["by_task"]["poker"]["sessions"] == 43
    assert all(matrix["summary"]["profiles"][name]["eligible"] == 0 for name in admission.PROFILES)
    assert matrix["profile_contract"]["robot_composite_requires_clean"] is True
    for row in matrix["rows"]:
        assert row["source"]["g2_pico_advisory"]["blocking_profiles"] == []
        for profile in admission.PROFILES:
            gate_ids = {gate["gate_id"] for gate in row["profiles"][profile]["gates"]}
            assert "capture.g2_pico21" not in gate_ids
            assert row["profiles"][profile]["active_split_role"] == "NOT_ADMITTED"
    assert all(set(row["active_roles"].values()) == {"NOT_ADMITTED"} for row in split["rows"])
    assert split["policy"]["EVAL_ONE_SHOT"] == {
        "gradient_updates": False,
        "may_select_best": False,
        "one_shot_after_freeze": True,
    }


def test_only_poker031_is_hawor_numeric_algorithm_failure(snapshot) -> None:
    matrix, _ = snapshot
    numeric = []
    for row in matrix["rows"]:
        raw_gates = {gate["gate_id"]: gate for gate in row["profiles"]["RAW"]["gates"]}
        if raw_gates["hawor.numeric"]["status"] == "FAIL":
            numeric.append((row["task_id"], row["session_id"], raw_gates["hawor.numeric"]))
    assert [(task, session) for task, session, _ in numeric] == [
        ("poker", "play_cards_0901_031")
    ]
    assert numeric[0][2]["owner"] == "HAWOR_ALGORITHM"
    assert numeric[0][2]["failure_kind"] == "ALGORITHM_FAILURE"


def test_v4_coverage_and_profile_status_counts_are_explicit(snapshot) -> None:
    matrix, _ = snapshot
    assert sum(row["evidence"]["identity_v4"] is not None for row in matrix["rows"]) == 4
    assert matrix["summary"]["profiles"]["RAW"]["status_counts"] == {
        "PASS": 0,
        "NOT_EVALUATED": 52,
        "HOLD": 3,
        "FAIL": 4,
    }
    assert matrix["summary"]["profiles"]["ROBOT_COMPOSITE"]["status_counts"] == {
        "PASS": 0,
        "NOT_EVALUATED": 0,
        "HOLD": 45,
        "FAIL": 14,
    }


def test_machine_outputs_have_177_rows_and_valid_sha_ledger(snapshot, tmp_path: Path) -> None:
    matrix, split = snapshot
    admission.write_snapshot(tmp_path, matrix, split)
    with (tmp_path / "ADMISSION_MATRIX.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 59 * 3
    assert {row["profile"] for row in rows} == set(admission.PROFILES)
    ledger = {}
    for line in (tmp_path / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        ledger[name] = digest
    assert set(ledger) == {
        "ADMISSION_MATRIX.json",
        "ADMISSION_MATRIX.csv",
        "SPLIT_CANDIDATES.json",
        "REPORT_ZH.md",
    }
    for name, digest in ledger.items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == digest
