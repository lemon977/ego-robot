from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jsonschema

from chaoyang.ops import build_handle_sensor_pipeline_admission_v71 as admission


PROJECT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _published_session(root: Path, dataset: str, task: str, name: str, frames: int,
                       content_status: str) -> Path:
    session = root / dataset / "cleaned" / task / name
    (session / "preprocess").mkdir(parents=True)
    (session / "mono.mp4").write_bytes(b"mono")
    (session / "stereo.mp4").write_bytes(b"stereo")
    (session / "controller.jsonl").write_text("{}\n" * frames)
    (session / "camera_params.json").write_text("{}")
    _write_json(session / "clip_manifest.json", {
        "files": {
            "video": "mono.mp4",
            "source_stereo_video": "stereo.mp4",
            "controller_poses": "controller.jsonl",
        },
        "video": {"frame_count": frames},
        "calibration_consistent": True,
        "source_calibration": {"sha256": "a" * 64},
        "timeline_resampling": {
            "hdf5_rows": frames,
            "source_video_frames": frames + 1,
            "unique_source_indices": frames,
            "repeated_transitions": 0,
            "skipped_transitions": 0,
            "max_forward_step": 1,
            "negative_transitions": 0,
            "segment_counts": {"0": frames},
        },
    })
    _write_json(session / "preprocess/pico_humanego_manifest.json", {
        "active_frame_counts": {"left": frames, "right": frames, "both": frames},
        "hand_model": {"name": "MANUS25 wrist-local source extension", "joint_names": list(range(25))},
        "controller6d": {"source": "egodex_v1 HDF5", "wrist_derivation": "same-session calibration"},
        "content_admission": {"status": content_status},
        "intrinsics_authority": "SAME_SESSION_FACTORY_SOURCE_K_SCALED_UNVERIFIED_EXTERNALLY",
        "image_domain_mode": "passthrough_scaled_source_domain",
        "rectified_video_written": False,
    })
    snapshots = {
        name: {"bytes": 1, "sha256": "b" * 64}
        for name in (
            "raw/manus.jsonl", "raw/manus.meta.json",
            "raw/tactile.jsonl", "raw/tactile.meta.json",
        )
    }
    _write_json(session / "CONVERSION_RESULT.json", {
        "source_snapshot": snapshots,
        "validation": {
            "controller_lines": frames,
            "stereo_probe": {
                "nb_read_frames": str(frames), "width": 4096, "height": 1536,
            },
        },
    })
    return session


def _fixture_root(tmp_path: Path) -> Path:
    root = tmp_path / "processed"
    ready = _published_session(
        root, "chips_cards_handle_0909", "potato_chips",
        "get_potato_chips_0909_001", 10, "SESSION_CONTENT_ADMISSIBLE",
    )
    visual = _published_session(
        root, "chips_cards_handle_0910", "playing_cards",
        "play_cards_0910_001", 8, "SESSION_CONTENT_NOT_ADMISSIBLE",
    )
    _write_json(root / "chips_cards_handle_0909/DATASET_RESULT.json", {
        "date_tag": "0909",
        "results": [
            {"task": "potato_chips", "session_id": "001", "classification": "CLEANED",
             "status": "COMMITTED", "frames": 10, "target": str(ready)},
            {"task": "playing_cards", "session_id": "104", "classification": "REJECTED",
             "status": "COMMITTED", "target": str(root / "chips_cards_handle_0909/rejected/playing_cards/104"),
             "error": "required acquisition files missing"},
        ],
    })
    _write_json(root / "chips_cards_handle_0910/DATASET_RESULT.json", {
        "date_tag": "0910",
        "results": [
            {"task": "playing_cards", "session_id": "001", "classification": "CLEANED",
             "status": "COMMITTED", "frames": 8, "target": str(visual)},
        ],
    })
    return root


def test_three_terminal_routes_and_sensor_claim_limits(tmp_path: Path) -> None:
    ledger = admission.build_ledger(_fixture_root(tmp_path))
    assert ledger["row_count"] == 3
    assert ledger["summary"]["by_admission"] == {
        "BLOCKED_SOURCE": 1,
        "SENSOR_GEOMETRY_READY": 1,
        "VISUAL_ONLY": 1,
    }
    rows = {row["session_id"]: row for row in ledger["rows"]}
    ready = rows["get_potato_chips_0909_001"]
    assert ready["eligibility"]["manus25"]["eligible"] is True
    assert ready["eligibility"]["controller_wrist"]["eligible"] is True
    assert ready["eligibility"]["tactile"]["status"] == "SOURCE_SNAPSHOT_PRESENT_NOT_MATERIALIZED"
    assert ready["eligibility"]["tactile"]["materialized_in_processed_frame_contract"] is False
    assert ready["downstream"]["contact"] == "NOT_EVALUATED_REQUIRES_OBJECT6D_AND_MATERIALIZED_TACTILE"
    assert rows["play_cards_0910_001"]["admission"] == "VISUAL_ONLY"
    assert rows["play_cards_0909_104"]["admission"] == "BLOCKED_SOURCE"


def test_schema_and_machine_outputs(tmp_path: Path) -> None:
    ledger = admission.build_ledger(_fixture_root(tmp_path))
    schema = json.loads((PROJECT / "contracts/handle_sensor_pipeline_admission_ledger_v71.schema.json").read_text())
    jsonschema.Draft202012Validator(schema).validate(ledger)
    output = tmp_path / "out"
    admission.write_ledger(output, ledger)
    assert (output / "SENSOR_PIPELINE_ADMISSION_LEDGER.json").is_file()
    assert (output / "SENSOR_PIPELINE_ADMISSION_LEDGER.csv").read_text().count("\n") == 4
    result = json.loads((output / "RESULT.json").read_text())
    result_schema = json.loads((PROJECT / "contracts/handle_h0_admission_result_v71.schema.json").read_text())
    jsonschema.Draft202012Validator(result_schema).validate(result)
    assert result["status"] == "PASSED"
    assert result["row_count"] == 3
    assert all(len(item["sha256"]) == 64 for item in result["files"].values())


def test_duplicate_canonical_identity_is_rejected(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    p = root / "chips_cards_handle_0910/DATASET_RESULT.json"
    value = json.loads(p.read_text())
    duplicate = dict(value["results"][0])
    duplicate["target"] = value["results"][0]["target"]
    value["results"].append(duplicate)
    _write_json(p, value)
    try:
        admission.build_ledger(root)
    except admission.AdmissionError as exc:
        assert "duplicate canonical" in str(exc)
    else:
        raise AssertionError("duplicate session identity was accepted")
