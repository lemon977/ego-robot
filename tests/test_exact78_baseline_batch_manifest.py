from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tools import build_exact78_baseline_batch_manifest as builder


def write(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def sources(tmp_path: Path) -> tuple[Path, Path]:
    cohort_rows = []
    readiness_rows = []
    for index in range(156):
        task = "chips" if index < 78 else "poker"
        session = f"session_{index:03d}"
        raw = f"/raw/{session}"
        cohort_rows.append({
            "task": task, "session_id": session, "date": "0902" if index < 97 else "0901",
            "split": "train", "frame_count": 10, "raw_path": raw,
            "immutable_input_identity_sha256": f"{index:064x}",
        })
        readiness_rows.append({
            "task": task, "session_id": session, "frame_count": 10, "raw_path": raw,
            "stereo_calibration_state": "CALIBRATED_METRIC_STEREO" if index < 97 else "CALIBRATION_MISSING",
        })
    return write(tmp_path / "cohort.json", {"sessions": cohort_rows}), write(tmp_path / "readiness.json", {"sessions": readiness_rows})


def test_exact_routing_counts_and_terminal_semantics(tmp_path: Path) -> None:
    cohort, readiness = sources(tmp_path)
    result = builder.build(cohort, readiness)
    assert result["counts"] == {"sessions": 156, "full_pipeline": 97, "calibration_missing_terminal": 59}
    assert result["sessions"][96]["allowed_stages"] == builder.FULL_STAGES
    assert result["sessions"][97]["terminal_code"] == "C_CALIBRATION_MISSING"


def test_identity_mismatch_fails_closed(tmp_path: Path) -> None:
    cohort, readiness = sources(tmp_path)
    value = json.loads(readiness.read_text())
    value["sessions"][0]["raw_path"] = "/wrong"
    readiness.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match="raw_path mismatch"):
        builder.build(cohort, readiness)


def test_count_drift_fails_closed(tmp_path: Path) -> None:
    cohort, readiness = sources(tmp_path)
    value = json.loads(readiness.read_text())
    value["sessions"][97]["stereo_calibration_state"] = "CALIBRATED_METRIC_STEREO"
    readiness.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match="count mismatch"):
        builder.build(cohort, readiness)


def test_no_clobber(tmp_path: Path) -> None:
    output = tmp_path / "out.json"
    output.write_text("existing")
    with pytest.raises(FileExistsError):
        builder.atomic_json(output, {})
