from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jsonschema
import pytest

from chaoyang.ops import build_handle_h3_h4_preflight_v71 as preflight


PROJECT = Path(__file__).resolve().parents[1]


def _fixture(tmp_path: Path, *, stereo_eligible: bool = True) -> tuple[Path, Path]:
    target = tmp_path / "processed/play_cards_0910_001"
    target.mkdir(parents=True)
    mono = target / "mono.mp4"
    stereo = target / "stereo.mp4"
    mono.write_bytes(b"mono")
    stereo.write_bytes(b"stereo")
    left = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    right = [[1, 0, 0, -0.064], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    eye = lambda source: {
        "intrinsics": {"fx": 930.0, "fy": 925.0, "cx": 1020.0, "cy": 770.0},
        "distortion": {"model": "equiDis62", "coeffs": [0.0] * 8},
        "sourceIndex": source,
    }
    (target / "camera_params.json").write_text(json.dumps({
        "width": 2048, "height": 1536, "left": eye(1), "right": eye(0),
        "extrinsics": {"left": left, "right": right}, "selected_calibration_key": "right",
    }))
    (target / "clip_manifest.json").write_text(json.dumps({
        "video": {"width": 1280, "height": 960, "frame_count": 3},
    }))
    rows = [
        {
            "session_id": "play_cards_0910_001", "dataset_id": "chips_cards_handle_0910",
            "task": "playing_cards", "frame_count": 3, "target": str(target),
            "admission": "SENSOR_GEOMETRY_READY",
            "eligibility": {
                "visual_rgb": {"eligible": True, "artifact": {"path": str(mono)}},
                "source_stereo": {"eligible_for_depth_preflight": stereo_eligible, "artifact": {"path": str(stereo)}, "probe": {"width": 4096, "height": 1536, "nb_read_frames": "3", "avg_frame_rate": "30/1"}},
                "camera_contract": {"eligible": True, "artifact": {"path": str(target / "camera_params.json")}, "intrinsics_authority": "SAME_SESSION", "image_domain_mode": "passthrough_scaled_source_domain", "rectified": False},
            },
        },
        {
            "session_id": "play_cards_0909_104", "dataset_id": "chips_cards_handle_0909",
            "task": "playing_cards", "frame_count": None, "target": None,
            "admission": "BLOCKED_SOURCE", "eligibility": {},
        },
    ]
    h0 = tmp_path / "h0.json"
    h0.write_text(json.dumps({"row_count": 2, "rows": rows}))
    status = tmp_path / "status.json"
    status.write_text(json.dumps({
        "generated_at": "2026-09-15T00:00:00+08:00",
        "active_tasks": [{"task_id": "clean", "session": "x", "pid": 123, "gpu_id": 0, "phase": "propainter_gpu", "heartbeat_at": "now"}],
    }))
    return h0, status


def test_h3_same_session_geometry_and_resource_terminal(tmp_path: Path) -> None:
    h0, status = _fixture(tmp_path)
    ledger = preflight.build("h3", h0, status, tmp_path / "h3")
    assert ledger["summary"]["by_status"] == {"BLOCKED_RESOURCE": 1, "BLOCKED_SOURCE": 1}
    row = ledger["rows"][0]
    assert row["cpu_preflight"] == "PASSED"
    assert row["contract"]["camera_contract"]["baseline_m"] == pytest.approx(0.064)
    assert row["contract"]["rectification_contract"]["fixed_exact78_rotation_allowed"] is False
    assert row["contract"]["future_depth_contract"]["controller_wrist_overwrite_allowed"] is False
    assert row["contract"]["future_depth_contract"]["depth_confidence_present"] is False
    schema = json.loads((PROJECT / "contracts/handle_sensor_depth_preflight_ledger_v71.schema.json").read_text())
    jsonschema.Draft202012Validator(schema).validate(ledger)


def test_h4_roles_objects_reentry_and_independent_branch(tmp_path: Path) -> None:
    h0, status = _fixture(tmp_path, stereo_eligible=False)
    ledger = preflight.build("h4", h0, status, tmp_path / "h4")
    assert ledger["rows"][0]["cpu_preflight"] == "PASSED"
    contract = ledger["sensor_role_mask_contract"]
    assert contract["controllers_are_not_gloves"] is True
    assert contract["forearms_are_not_gloves"] is True
    assert contract["chips_union_allowed"] is False
    row_contract = ledger["rows"][0]["contract"]
    assert "OFFSCREEN_REENTRY" in row_contract["regression_cases"]
    assert row_contract["object_contract"]["union_allowed"] is False
    assert row_contract["seed_contract"]["text_prompt_alone_grants_identity"] is False
    schema = json.loads((PROJECT / "contracts/sensor_role_mask_v1.schema.json").read_text())
    jsonschema.Draft202012Validator(schema).validate(ledger)


def test_h3_failure_does_not_block_h4_and_outputs_are_immutable(tmp_path: Path) -> None:
    h0, status = _fixture(tmp_path, stereo_eligible=False)
    h3 = preflight.build("h3", h0, status, tmp_path / "h3")
    assert h3["rows"][0]["status"] == "BLOCKED_PREREQ"
    h4 = preflight.build("h4", h0, status, tmp_path / "h4")
    assert h4["rows"][0]["status"] == "BLOCKED_RESOURCE"
    with pytest.raises(FileExistsError):
        preflight.build("h4", h0, status, tmp_path / "h4")
