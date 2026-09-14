from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.governance.heartbeat_task import reconcile_clean_authority


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _selection(root: Path) -> tuple[Path, str]:
    rows = []
    for index in range(58):
        rows.append(
            {
                "session_id": f"session_{index:03d}",
                "task": "chips" if index < 29 else "poker",
                "frame_count": 10,
                "existing_clean": {"frozen": True} if index < 4 else None,
            }
        )
    path = root / "EXACT78_WAVE0_SELECTION.json"
    _write(path, {"sessions": rows})
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _authority() -> dict:
    return {
        "waves": {"wave0_clean_passed": 4, "wave0_clean_pending": 54},
        "stages": [
            {
                "stage": "Clean",
                "total": 58,
                "passed": 4,
                "grade_c": 0,
                "running": 0,
                "blocked": 54,
                "authority_scope": "OLD",
                "denominator_semantics": "OLD",
                "evidence": [],
            }
        ],
    }


def test_reconcile_pass_terminal_and_single_running(tmp_path: Path) -> None:
    _, selection_sha = _selection(tmp_path)
    _write(
        tmp_path / "propainter_v1/session_004/RESULT.json",
        {
            "status": "PASS_SYNTHETIC_CLEAN_BASELINE_GRADE_B",
            "grade": "B",
            "downstream_authorized": True,
            "session": "session_004",
            "task": "chips",
            "frame_count": 10,
            "hard_gates": {"decode": "PASS"},
        },
    )
    _write(
        tmp_path / "clean_terminals/session_005/RESULT.json",
        {
            "status": "FAILED_QUALITY_C",
            "terminal": True,
            "downstream_authorized": False,
            "session": "session_005",
            "task": "chips",
        },
    )
    authority = _authority()
    assert reconcile_clean_authority(
        authority,
        active=True,
        clean_root=tmp_path,
        expected_selection_sha256=selection_sha,
    )
    assert authority["waves"]["wave0_clean_passed"] == 5
    assert authority["waves"]["wave0_clean_pending"] == 52
    stage = authority["stages"][0]
    assert (stage["passed"], stage["grade_c"], stage["running"], stage["blocked"]) == (5, 1, 1, 51)
    assert len(stage["evidence"]) == 3


def test_reconcile_fails_closed_on_invalid_pass(tmp_path: Path) -> None:
    _, selection_sha = _selection(tmp_path)
    _write(
        tmp_path / "propainter_v1/session_004/RESULT.json",
        {
            "status": "PASS_SYNTHETIC_CLEAN_BASELINE_GRADE_B",
            "grade": "B",
            "downstream_authorized": False,
            "session": "session_004",
            "task": "chips",
            "frame_count": 10,
            "hard_gates": {"decode": "PASS"},
        },
    )
    authority = _authority()
    before = json.loads(json.dumps(authority))
    assert not reconcile_clean_authority(
        authority,
        active=True,
        clean_root=tmp_path,
        expected_selection_sha256=selection_sha,
    )
    assert authority == before
