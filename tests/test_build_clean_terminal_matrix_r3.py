import hashlib
import json
from pathlib import Path

import pytest

from chaoyang.ops.build_clean_terminal_matrix_r3 import build_rows


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixtures(tmp_path: Path):
    selected = []
    terminal = []
    for index in range(58):
        sid = f"session_{index:03d}"
        result_path = tmp_path / f"{sid}.json"
        result_path.write_text(json.dumps({
            "schema_version": "clean-test-v1",
            "session": sid,
            "status": "PASS_GRADE_B",
            "grade": "B",
            "downstream_authorized": True,
            "hard_gates": {"decode": "PASS"},
        }), encoding="utf-8")
        selected.append({"position": index + 1, "task": "chips" if index < 39 else "poker", "session_id": sid, "frame_count": 10})
        terminal.append({
            "position": index + 1,
            "task": "chips" if index < 39 else "poker",
            "session": sid,
            "status": "PASSED",
            "clean_join_ready": True,
            "terminal_result": {"path": str(result_path), "bytes": result_path.stat().st_size, "sha256": _sha(result_path)},
        })
    return {"sessions": selected}, {"sessions": terminal}


def test_build_rows_closes_exactly_58_and_preserves_join_gate(tmp_path: Path):
    selection, ledger = _fixtures(tmp_path)
    rows = build_rows(selection, ledger)
    assert len(rows) == 58
    assert len({row["session_id"] for row in rows}) == 58
    assert sum(row["clean_join_ready"] for row in rows) == 58
    assert all(row["artifact_revision"] == "R7_0" for row in rows)


def test_build_rows_rejects_terminal_result_sha_mismatch(tmp_path: Path):
    selection, ledger = _fixtures(tmp_path)
    ledger["sessions"][0]["terminal_result"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="bytes/SHA mismatch"):
        build_rows(selection, ledger)
