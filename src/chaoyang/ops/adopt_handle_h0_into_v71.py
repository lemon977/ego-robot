from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


"""Adopt the verified Handle H0 ledger into the canonical V7.1 task path."""

import json
from pathlib import Path

from chaoyang.governance.common import artifact_ref
from chaoyang.governance.v52_contracts import atomic_write_new


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_handle_sensor_pipeline_v71/h0_admission"
TARGET = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/sensor/h0"


def main() -> int:
    source_result = json.loads((SOURCE / "RESULT.json").read_text(encoding="utf-8"))
    if source_result.get("status") != "PASSED" or source_result.get("row_count") != 332:
        raise RuntimeError("H0 source result is not the verified 332-row terminal")
    adopted: dict[str, dict[str, object]] = {}
    for name in ("SENSOR_PIPELINE_ADMISSION_LEDGER.json", "SENSOR_PIPELINE_ADMISSION_LEDGER.csv"):
        source = SOURCE / name
        target = TARGET / name
        atomic_write_new(target, source.read_bytes())
        source_ref = artifact_ref(source)
        target_ref = artifact_ref(target)
        if source_ref["bytes"] != target_ref["bytes"] or source_ref["sha256"] != target_ref["sha256"]:
            raise RuntimeError(f"adoption mismatch: {name}")
        adopted[name] = target_ref
    result = {
        "schema_version": "chaoyang-v71-h0-adoption-v1",
        "status": "PASSED",
        "task_id": "sensor_h0_admission_v1",
        "artifact_revision": "R7_0",
        "validity": "VALID_FOR_PINNED_REVISION",
        "source_result": artifact_ref(SOURCE / "RESULT.json"),
        "row_count": source_result["row_count"],
        "summary": source_result["summary"],
        "files": adopted,
        "claim_limit": source_result["claim_limit"],
    }
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"
    atomic_write_new(TARGET / "RESULT.json", payload)
    print(TARGET / "RESULT.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
