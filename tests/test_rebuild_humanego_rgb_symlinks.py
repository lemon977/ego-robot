import json
import os
from pathlib import Path

import pytest

from tools.rebuild_humanego_rgb_symlinks import (
    RebuildError,
    apply_links,
    load_plans,
    preflight_destinations,
    verify_links,
    verify_source_inventories,
)


def _fixture(tmp_path: Path, frames: int = 2) -> tuple[Path, Path]:
    production = tmp_path / "production"
    authority = tmp_path / "raw"
    session = "grap_a_cap_004"
    adapter = production / session / "09_humanego_adapter"
    source_all_data = authority / session / "preprocess" / "all_data"
    destination_all_data = adapter / "preprocess" / "all_data"
    for index in range(frames):
        frame_id = f"{index:05d}"
        source_frame = source_all_data / frame_id
        destination_frame = destination_all_data / frame_id
        source_frame.mkdir(parents=True)
        destination_frame.mkdir(parents=True)
        (source_frame / "rgb.png").write_bytes(b"raw")
        (source_frame / "rgb_WoArm_WArmObjKpts.png").write_bytes(b"untouched")
        (destination_frame / "training_data.json").write_text("{}", encoding="utf-8")
    manifest = {
        "source_session": str(authority / session),
        "source_all_data": str(source_all_data),
        "source_data_modified": False,
        "output_session": str(adapter),
        "all_data": str(destination_all_data),
        "frame_count": frames,
        "image_mode": "symlink",
        "image_files": ["rgb.png", "rgb_WoArm_WArmObjKpts.png"],
    }
    adapter.mkdir(parents=True, exist_ok=True)
    (adapter / "humanego_ict_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return production, authority


def _plans(production: Path, authority: Path, frames: int = 2):
    return load_plans(production, authority, 1, frames)


def test_dry_run_then_apply_creates_only_rgb_links(tmp_path: Path) -> None:
    production, authority = _fixture(tmp_path)
    plans = _plans(production, authority)
    assert preflight_destinations(plans) == {"absent": 2, "correct": 0}
    verify_source_inventories(plans, workers=1)
    assert apply_links(plans) == 2
    assert verify_links(plans)[0]["actual_rgb_links"] == 2
    destination = production / "grap_a_cap_004/09_humanego_adapter/preprocess/all_data"
    assert os.readlink(destination / "00000/rgb.png") == str(
        authority / "grap_a_cap_004/preprocess/all_data/00000/rgb.png"
    )
    assert not (destination / "00000/rgb_WoArm_WArmObjKpts.png").exists()


def test_conflict_fails_before_any_link_is_created(tmp_path: Path) -> None:
    production, authority = _fixture(tmp_path)
    plans = _plans(production, authority)
    destination = production / "grap_a_cap_004/09_humanego_adapter/preprocess/all_data"
    (destination / "00001/rgb.png").write_bytes(b"unknown")
    with pytest.raises(RebuildError, match="destination conflicts"):
        preflight_destinations(plans)
    assert not (destination / "00000/rgb.png").exists()


def test_manifest_cannot_redirect_outside_authority(tmp_path: Path) -> None:
    production, authority = _fixture(tmp_path)
    manifest_path = production / "grap_a_cap_004/09_humanego_adapter/humanego_ict_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_all_data"] = str(tmp_path / "other")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RebuildError, match="source_all_data mismatch"):
        _plans(production, authority)


def test_missing_raw_rgb_fails_inventory(tmp_path: Path) -> None:
    production, authority = _fixture(tmp_path)
    plans = _plans(production, authority)
    (authority / "grap_a_cap_004/preprocess/all_data/00001/rgb.png").unlink()
    with pytest.raises(RebuildError, match="RAW rgb inventory mismatch"):
        verify_source_inventories(plans, workers=1)
