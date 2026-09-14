from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from tools import build_eligible68_future_mask_coverage_report_contract_t0 as subject
from tools import eligible68_no_discovery_guard_t0 as no_discovery_guard


FIXED_TIME = "2026-08-29T12:30:00+08:00"
AUTHORITY_REF = {
    "path": "/future/frozen/A_CLASS_RELEASE.json",
    "bytes": 123,
    "sha256": "a" * 64,
}
QA_CHECKS = {key: True for key in subject.QA_CHECK_KEYS}


def _write_json(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    payload = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    path.write_bytes(payload)
    return {
        "path": str(path),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _unmeasured(reason: str) -> dict[str, Any]:
    return {"status": "UNMEASURED", "reason": reason}


def _side(frame_count: int) -> dict[str, Any]:
    return {
        "total_frames": frame_count,
        "taxonomy": {"ACCEPT": frame_count, "HOLD": 0, "hold_reasons": {}},
        "candidate_audit": {
            "status": "COMPLETE",
            "expected_rows": frame_count,
            "observed_rows": frame_count,
            "missing_rows": 0,
            "duplicate_rows": 0,
        },
        "topology_alert_rate": _unmeasured("NO_BOUND_TOPOLOGY_TERMINAL"),
        "coverage_proxy": _unmeasured("NO_BOUND_COVERAGE_PROXY_TERMINAL"),
        "h_completeness": _unmeasured(
            "STRUCTURAL_ACCEPT_IS_NOT_H_COMPLETENESS"
        ),
    }


def _result(
    row: dict[str, Any], contract_ref: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": "eligible68-mask-session-terminal-result-v1",
        "contract_ref": contract_ref,
        "session_id": row["session_id"],
        "split": row["split"],
        "frame_count": row["frame_count"],
        "terminal_status": "CANDIDATE_ACCEPTABLE",
        "sides": {
            "left": _side(row["frame_count"]),
            "right": _side(row["frame_count"]),
        },
        "governance": {
            "candidate_only": True,
            "baseline_freeze": False,
            "promotion": False,
            "bucket_unlock": False,
        },
        "admission_authority_ref": AUTHORITY_REF,
    }


def _qa(row: dict[str, Any], result_ref: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "eligible68-mask-session-terminal-independent-qa-v1",
        "producer_role": "INDEPENDENT_QA_AGENT",
        "status": "PASS_TERMINAL_RESULT_VERIFIED",
        "subject": result_ref,
        "session_id": row["session_id"],
        "split": row["split"],
        "frame_count": row["frame_count"],
        "checks": QA_CHECKS,
        "admission_authority_ref": AUTHORITY_REF,
    }


@pytest.fixture(scope="module")
def frozen_contract() -> dict[str, Any]:
    return subject.build_contract(created_at=FIXED_TIME)


def _admitted_successor(contract: dict[str, Any]) -> dict[str, Any]:
    successor = deepcopy(contract)
    successor["current_state"]["eligible68_pixel_or_selector_admission_sessions"] = 68
    successor["status"] = "SYNTHETIC_TEST_ONLY_ADMITTED_SUCCESSOR"
    return successor


def _build_complete_fixture(
    tmp_path: Path, contract: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    result_root = tmp_path / "terminals"
    qa_root = tmp_path / "qa"
    result_root.mkdir()
    qa_root.mkdir()
    contract_ref = {
        "path": str(tmp_path / "FUTURE_CONTRACT.json"),
        "bytes": 321,
        "sha256": "b" * 64,
    }
    index_rows: list[dict[str, Any]] = []
    for raw_row in contract["population_contract"]["ordered_sessions"]:
        row = dict(raw_row)
        result_path = result_root / f"{row['eligible68_order']:02d}.result.json"
        result_ref = _write_json(result_path, _result(row, contract_ref))
        qa_path = qa_root / f"{row['eligible68_order']:02d}.qa.json"
        qa_ref = _write_json(qa_path, _qa(row, result_ref))
        index_rows.append(
            {
                "eligible68_order": row["eligible68_order"],
                "session_id": row["session_id"],
                "split": row["split"],
                "frame_count": row["frame_count"],
                "result_ref": result_ref,
                "independent_qa_ref": qa_ref,
            }
        )
    index = {
        "schema_version": "eligible68-mask-session-terminal-index-v1",
        "contract_ref": contract_ref,
        "sessions": index_rows,
    }
    return index, contract_ref, result_root, qa_root


def _aggregate(
    index: dict[str, Any],
    contract: dict[str, Any],
    contract_ref: dict[str, Any],
    result_root: Path,
    qa_root: Path,
) -> dict[str, Any]:
    return subject.aggregate_index_document(
        index,
        contract,
        contract_ref,
        terminal_root=result_root,
        qa_root=qa_root,
        created_at=FIXED_TIME,
    )


def test_frozen_contract_binds_zero_admission_and_emits_no_current_coverage(
    frozen_contract: dict[str, Any],
) -> None:
    assert frozen_contract["status"].endswith("ZERO_OF_68_ADMITTED")
    state = frozen_contract["current_state"]
    assert state["eligible68_pixel_or_selector_admission_sessions"] == 0
    assert state["future_session_terminal_inputs_consumed"] == 0
    assert state["current_coverage_report"] == "NOT_PRODUCED"
    assert state["current_coverage_numbers_emitted"] is False
    assert state["topology_alert_rate"] == {"status": "UNMEASURED"}
    assert state["coverage_proxy"] == {"status": "UNMEASURED"}
    assert state["structural_accept_is_h_completeness"] is False
    assert "sessions" not in state


def test_contract_population_is_exact_train60_validation8_and_forbidden_disjoint(
    frozen_contract: dict[str, Any],
) -> None:
    population = frozen_contract["population_contract"]
    rows = population["ordered_sessions"]
    assert len(rows) == 68
    assert sum(row["split"] == "train" for row in rows) == 60
    assert sum(row["split"] == "validation" for row in rows) == 8
    assert sum(row["frame_count"] for row in rows if row["split"] == "train") == 25_175
    assert sum(row["frame_count"] for row in rows if row["split"] == "validation") == 3_090
    assert {row["session_id"] for row in rows}.isdisjoint(subject.FORBIDDEN_IDENTITIES)
    assert "grap_a_cap_025" in population["forbidden_identities"]
    assert "grap_a_cap_149" in population["forbidden_identities"]
    assert population["forbidden_splits"] == ["test"]


def test_current_zero_admission_rejects_before_any_terminal_read(
    frozen_contract: dict[str, Any],
) -> None:
    calls = 0

    def forbidden_reader(_ref: dict[str, Any], _root: Path) -> Any:
        nonlocal calls
        calls += 1
        raise AssertionError("reader must not run")

    with pytest.raises(subject.CoverageContractError, match="0/68 admission"):
        subject.aggregate_index_document(
            {},
            frozen_contract,
            {"path": "/never", "bytes": 1, "sha256": "0" * 64},
            terminal_root=Path("/never"),
            qa_root=Path("/never"),
            reader=forbidden_reader,
        )
    assert calls == 0


def test_complete_future_fixture_aggregates_per_side_and_split_without_h_claim(
    tmp_path: Path, frozen_contract: dict[str, Any]
) -> None:
    contract = _admitted_successor(frozen_contract)
    index, contract_ref, result_root, qa_root = _build_complete_fixture(tmp_path, contract)
    report = _aggregate(index, contract, contract_ref, result_root, qa_root)
    assert report["status"] == "CANDIDATE_ACCEPTABLE"
    assert len(report["sessions"]) == 68
    assert report["strata"]["train"]["sessions"] == 60
    assert report["strata"]["validation"]["sessions"] == 8
    assert report["strata"]["train"]["frames"] == 25_175
    assert report["strata"]["validation"]["frames"] == 3_090
    assert report["strata"]["train"]["sides"]["left"]["total_frames"] == 25_175
    assert report["semantic_limits"]["structural_accept_is_h_completeness"] is False
    assert report["semantic_limits"]["unmeasured_is_not_zero"] is True
    assert all(
        row[side]["h_completeness"]["status"] == "UNMEASURED"
        for row in report["sessions"]
        for side in subject.SIDES
    )


def test_missing_session_fails_before_any_terminal_read(
    frozen_contract: dict[str, Any],
) -> None:
    contract = _admitted_successor(frozen_contract)
    rows = contract["population_contract"]["ordered_sessions"][:-1]
    index = {
        "schema_version": "eligible68-mask-session-terminal-index-v1",
        "contract_ref": {"path": "/c", "bytes": 1, "sha256": "0" * 64},
        "sessions": [
            {
                "eligible68_order": row["eligible68_order"],
                "session_id": row["session_id"],
                "split": row["split"],
                "frame_count": row["frame_count"],
                "result_ref": {"path": "/r", "bytes": 1, "sha256": "0" * 64},
                "independent_qa_ref": {"path": "/q", "bytes": 1, "sha256": "0" * 64},
            }
            for row in rows
        ],
    }
    calls = 0

    def forbidden_reader(_ref: dict[str, Any], _root: Path) -> Any:
        nonlocal calls
        calls += 1
        raise AssertionError("reader must not run")

    with pytest.raises(subject.CoverageContractError, match="exactly 68"):
        subject.aggregate_index_document(
            index,
            contract,
            index["contract_ref"],
            terminal_root=Path("/r"),
            qa_root=Path("/q"),
            reader=forbidden_reader,
        )
    assert calls == 0


@pytest.mark.parametrize("forbidden", ["grap_a_cap_025", "grap_a_cap_149"])
def test_forbidden_identity_fails_before_any_terminal_read(
    forbidden: str, frozen_contract: dict[str, Any]
) -> None:
    contract = _admitted_successor(frozen_contract)
    contract_ref = {"path": "/c", "bytes": 1, "sha256": "0" * 64}
    rows = []
    for raw_row in contract["population_contract"]["ordered_sessions"]:
        row = {
            "eligible68_order": raw_row["eligible68_order"],
            "session_id": raw_row["session_id"],
            "split": raw_row["split"],
            "frame_count": raw_row["frame_count"],
            "result_ref": {"path": "/r", "bytes": 1, "sha256": "1" * 64},
            "independent_qa_ref": {"path": "/q", "bytes": 1, "sha256": "2" * 64},
        }
        rows.append(row)
    rows[0]["session_id"] = forbidden
    index = {
        "schema_version": "eligible68-mask-session-terminal-index-v1",
        "contract_ref": contract_ref,
        "sessions": rows,
    }
    calls = 0

    def forbidden_reader(_ref: dict[str, Any], _root: Path) -> Any:
        nonlocal calls
        calls += 1
        raise AssertionError("reader must not run")

    with pytest.raises(subject.CoverageContractError, match="forbidden identity"):
        subject.aggregate_index_document(
            index,
            contract,
            contract_ref,
            terminal_root=Path("/r"),
            qa_root=Path("/q"),
            reader=forbidden_reader,
        )
    assert calls == 0


def test_test_split_and_mixed_split_fail_before_any_terminal_read(
    frozen_contract: dict[str, Any],
) -> None:
    contract = _admitted_successor(frozen_contract)
    contract_ref = {"path": "/c", "bytes": 1, "sha256": "0" * 64}
    template = []
    for raw_row in contract["population_contract"]["ordered_sessions"]:
        template.append(
            {
                "eligible68_order": raw_row["eligible68_order"],
                "session_id": raw_row["session_id"],
                "split": raw_row["split"],
                "frame_count": raw_row["frame_count"],
                "result_ref": {"path": "/r", "bytes": 1, "sha256": "1" * 64},
                "independent_qa_ref": {"path": "/q", "bytes": 1, "sha256": "2" * 64},
            }
        )
    for replacement in ("test", "validation"):
        rows = deepcopy(template)
        rows[0]["split"] = replacement
        index = {
            "schema_version": "eligible68-mask-session-terminal-index-v1",
            "contract_ref": contract_ref,
            "sessions": rows,
        }
        with pytest.raises(subject.CoverageContractError):
            subject.aggregate_index_document(
                index,
                contract,
                contract_ref,
                terminal_root=Path("/r"),
                qa_root=Path("/q"),
                reader=lambda _ref, _root: (_ for _ in ()).throw(
                    AssertionError("reader must not run")
                ),
            )


def test_duplicate_hardlink_alias_is_rejected(
    tmp_path: Path, frozen_contract: dict[str, Any]
) -> None:
    contract = _admitted_successor(frozen_contract)
    index, contract_ref, result_root, qa_root = _build_complete_fixture(tmp_path, contract)
    first = Path(index["sessions"][0]["result_ref"]["path"])
    second = Path(index["sessions"][1]["result_ref"]["path"])
    second.unlink()
    os.link(first, second)
    index["sessions"][1]["result_ref"] = dict(index["sessions"][0]["result_ref"])
    index["sessions"][1]["result_ref"]["path"] = str(second)
    with pytest.raises(subject.CoverageContractError, match="duplicate or aliased"):
        _aggregate(index, contract, contract_ref, result_root, qa_root)


def test_fake_qa_is_rejected_even_when_index_hash_matches_it(
    tmp_path: Path, frozen_contract: dict[str, Any]
) -> None:
    contract = _admitted_successor(frozen_contract)
    index, contract_ref, result_root, qa_root = _build_complete_fixture(tmp_path, contract)
    row = index["sessions"][7]
    qa_path = Path(row["independent_qa_ref"]["path"])
    fake = json.loads(qa_path.read_text())
    fake["producer_role"] = "AUTHOR_SELF_QA"
    qa_path.unlink()
    row["independent_qa_ref"] = _write_json(qa_path, fake)
    with pytest.raises(subject.CoverageContractError, match="fake or non-independent"):
        _aggregate(index, contract, contract_ref, result_root, qa_root)


def test_path_toctou_replacement_after_ref_freeze_fails_closed(
    tmp_path: Path, frozen_contract: dict[str, Any]
) -> None:
    contract = _admitted_successor(frozen_contract)
    index, contract_ref, result_root, qa_root = _build_complete_fixture(tmp_path, contract)
    row = index["sessions"][12]
    result_path = Path(row["result_ref"]["path"])
    original_ref = dict(row["result_ref"])
    replacement = _result(row, contract_ref)
    replacement["terminal_status"] = "HOLD"
    replacement["sides"]["left"]["taxonomy"] = {
        "ACCEPT": row["frame_count"] - 1,
        "HOLD": 1,
        "hold_reasons": {"SYNTHETIC": 1},
    }
    result_path.unlink()
    _write_json(result_path, replacement)
    assert row["result_ref"] == original_ref
    with pytest.raises(subject.CoverageContractError, match="same-FD verification"):
        _aggregate(index, contract, contract_ref, result_root, qa_root)


def test_unmeasured_metrics_cannot_smuggle_numeric_zero(
    frozen_contract: dict[str, Any]
) -> None:
    row = dict(frozen_contract["population_contract"]["ordered_sessions"][0])
    contract_ref = {"path": "/c", "bytes": 1, "sha256": "0" * 64}
    result = _result(row, contract_ref)
    result["sides"]["left"]["coverage_proxy"] = {
        "status": "UNMEASURED",
        "reason": "NO_MEASUREMENT",
        "mean": 0.0,
    }
    with pytest.raises(subject.CoverageContractError, match="keys differ"):
        subject._validate_result(result, row, contract_ref)


def test_incomplete_candidate_audit_cannot_be_candidate_acceptable(
    frozen_contract: dict[str, Any]
) -> None:
    row = dict(frozen_contract["population_contract"]["ordered_sessions"][0])
    contract_ref = {"path": "/c", "bytes": 1, "sha256": "0" * 64}
    result = _result(row, contract_ref)
    audit = result["sides"]["right"]["candidate_audit"]
    audit.update(
        {
            "status": "INCOMPLETE",
            "observed_rows": row["frame_count"] - 1,
            "missing_rows": 1,
        }
    )
    with pytest.raises(subject.CoverageContractError, match="must force terminal HOLD"):
        subject._validate_result(result, row, contract_ref)


def test_subject_has_no_discovery_calls() -> None:
    source = Path(subject.__file__).read_text(encoding="utf-8")
    assert no_discovery_guard.find_forbidden_discovery_calls(source) == ()
