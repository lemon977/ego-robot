from __future__ import annotations

import os
from pathlib import Path

from tools import repair_hawor_processed_pointers as pointers


def test_explicit_authorization_cannot_be_inferred_from_numeric_pass() -> None:
    result = {
        "gates": {
            "mask_numeric_candidate": "PASS",
            "formal_mask_seed_authority": "NOT_EVALUATED_MISSING_IDENTITY",
        }
    }
    assert pointers.explicit_consumption_authorized(result) is False
    assert pointers.explicit_consumption_authorized({"consumption_authorized": "true"}) is False
    assert pointers.explicit_consumption_authorized({"consumption_authorized": True}) is True


def test_status_mapping_keeps_fail_hold_and_not_evaluated_distinct() -> None:
    assert pointers.normalized_status("PASS") == "PASS"
    assert pointers.normalized_status("FAIL_ALGORITHM") == "FAIL"
    assert pointers.normalized_status("HOLD_MISSING_AUTHORITY") == "HOLD"
    assert pointers.normalized_status("NOT_EVALUATED") == "NOT_EVALUATED"


def test_atomic_latest_terminal_symlink(tmp_path: Path) -> None:
    version = tmp_path / pointers.VERSION
    version.mkdir()
    pointers._atomic_symlink(tmp_path, "latest_terminal", pointers.VERSION)
    state = pointers.symlink_state(tmp_path / "latest_terminal")
    assert state == {"exists": True, "is_symlink": True, "target": pointers.VERSION}
    assert not any(path.name.startswith(".latest_terminal.tmp") for path in tmp_path.iterdir())


def test_status_contract_separates_current_and_latest_terminal() -> None:
    row = {
        "task_id": "poker",
        "session_id": "play_cards_0901_031",
        "chips001_legacy_current_exception": False,
        "numeric_status": "FAIL",
        "terminal_classification": "HAWOR_NUMERIC_FAIL_INTERNAL_DETECTOR_OR_TRACK_GAP",
        "formal_mask_seed_authority": "NOT_EVALUATED_MISSING_INDEPENDENT_CONTOUR_AND_IDENTITY",
        "formal_robot_seed_authority": "NOT_EVALUATED_MISSING_CONTOUR_IDENTITY_OBJECT6D_AND_CONTACT",
        "independent_gate_v4": {"status": "HOLD_HUMAN_REVIEW_REQUIRED"},
    }
    status = pointers._make_status(
        row,
        "transaction.json",
        {"mask": {"status": "NO_FORMAL_CURRENT"}},
    )
    hawor = status["stages"]["hawor"]
    assert hawor["current_pointer"] is None
    assert hawor["current_status"] == "NO_CONSUMER_CURRENT"
    assert hawor["latest_terminal_pointer"] == "hawor/latest_terminal"
    assert hawor["latest_terminal_is_consumer_authority"] is False
    assert hawor["full59_numeric_status"] == "FAIL"
    assert status["stages"]["mask"] == {"status": "NO_FORMAL_CURRENT"}


def test_symlink_state_does_not_follow_target(tmp_path: Path) -> None:
    broken = tmp_path / "current"
    os.symlink("missing-version", broken)
    assert pointers.symlink_state(broken) == {
        "exists": True,
        "is_symlink": True,
        "target": "missing-version",
    }
