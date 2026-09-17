import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVED_TASKS = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks"


def archived_path(value: str) -> Path:
    prefix = "/mnt/workspace/code/chaoyang/tasks/"
    return ARCHIVED_TASKS / value[len(prefix):] if value.startswith(prefix) else Path(value)


def test_v71_cleanup_receipt_preserves_external_data_and_adopts_v6() -> None:
    value = json.loads((ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/cleanup/CLEANUP_DELETION_RECEIPT.json").read_text())
    assert value["status"] == "PASSED"
    assert value["checks"]["v6_cleanup_passed"]
    assert value["checks"]["processed_archive_absent"]
    assert value["checks"]["processed_canary_absent"]
    assert value["checks"]["protected_egodata_not_targeted"]
    assert value["checks"]["protected_nas_not_targeted"]
    assert archived_path(value["prior_v6_receipt"]["path"]).is_file()
