from __future__ import annotations

import hashlib
import json
from pathlib import Path

from chaoyang.cli import _maintained_operations


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "tasks/receipts/HANDLE_DATA_CLEANING_V3_COMPLETION.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_data_cleaning_completion_receipt_is_terminal_and_bound() -> None:
    value = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert value["status"] == "COMMITTED"
    assert value["totals"] == {
        "session_count": 681,
        "completed": 681,
        "cleaned": 561,
        "rejected": 120,
        "failed": 0,
    }
    assert value["execution"] == {
        "gpu_used": False,
        "raw_data_modified": False,
        "processed_payload_modified_by_provenance_repair": False,
    }
    for reference in [
        value["queue_state"],
        *(dataset[key]
          for dataset in value["datasets"]
          for key in ("dataset_result", "preflight_audit")),
    ]:
        path = Path(reference["path"])
        assert path.is_file(), path
        assert path.stat().st_size == reference["bytes"], path
        assert _sha256(path) == reference["sha256"], path


def test_data_cleaning_operations_are_current_cli_entries() -> None:
    required = {
        "tactile_quality_gate_v1",
        "convert_handle_egodex_v3",
        "batch_clean_handle_content_v3",
        "run_handle_cleaning_v3_queue",
    }
    assert required <= _maintained_operations()


def test_data_cleaning_contract_prevents_implicit_restart() -> None:
    contract = json.loads(
        (ROOT / "docs/governance/ALGORITHM_CONTRACT.json").read_text(
            encoding="utf-8"))
    cleaning = contract["special_status_contracts"]["handle_data_cleaning_v3"]
    assert cleaning["terminal_status"] == "COMMITTED"
    assert cleaning["execution_status"] == "COMPLETE_NO_ACTIVE_TASK"
    assert cleaning["restart_authorized"] is False
    assert cleaning["gpu_required"] is False
    assert cleaning["counts"]["failed"] == 0
