from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "vendor/FoundationStereo/scripts/pico_stereo_depth.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("pico_stereo_depth_r3_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_source_indices_follow_physical_eye_metadata(tmp_path: Path) -> None:
    module = _load_module()
    params = tmp_path / "camera_params.json"
    params.write_text(
        json.dumps({"left": {"sourceIndex": 1}, "right": {"sourceIndex": 0}}),
        encoding="utf-8",
    )
    assert module.load_source_indices(params) == (1, 0)


@pytest.mark.parametrize(
    "payload",
    [
        {"left": {"sourceIndex": 0}, "right": {"sourceIndex": 0}},
        {"left": {"sourceIndex": 2}, "right": {"sourceIndex": 0}},
        {"left": {}, "right": {"sourceIndex": 0}},
        {"left": {"sourceIndex": "1"}, "right": {"sourceIndex": 0}},
    ],
)
def test_source_indices_fail_closed(tmp_path: Path, payload: dict) -> None:
    module = _load_module()
    params = tmp_path / "camera_params.json"
    params.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        module.load_source_indices(params)


def test_remap_pair_routes_sbs_halves_in_physical_eye_order() -> None:
    module = _load_module()
    eye_width = 3
    stereo = np.concatenate(
        (
            np.full((2, eye_width, 3), 17, dtype=np.uint8),
            np.full((2, eye_width, 3), 93, dtype=np.uint8),
        ),
        axis=1,
    )
    map_x, map_y = np.meshgrid(
        np.arange(eye_width, dtype=np.float32),
        np.arange(2, dtype=np.float32),
    )
    physical_left, physical_right = module.remap_pair(
        stereo,
        eye_width,
        [(map_x, map_y), (map_x, map_y)],
        source_indices=(1, 0),
    )
    assert np.all(physical_left == 93)
    assert np.all(physical_right == 17)


def test_remap_pair_rejects_non_permutation_source_indices() -> None:
    module = _load_module()
    stereo = np.zeros((2, 6, 3), dtype=np.uint8)
    map_x, map_y = np.meshgrid(
        np.arange(3, dtype=np.float32), np.arange(2, dtype=np.float32)
    )
    with pytest.raises(ValueError):
        module.remap_pair(
            stereo,
            3,
            [(map_x, map_y), (map_x, map_y)],
            source_indices=(0, 0),
        )


def test_cached_rectification_requires_same_session_and_source_order(
    tmp_path: Path,
) -> None:
    module = _load_module()
    params = tmp_path / "camera_params.json"
    params.write_text("{}", encoding="utf-8")
    valid = {
        "source_camera_params": str(params.resolve()),
        "physical_eye_source_indices": {"left": 1, "right": 0},
    }
    module.validate_cached_calibration_identity(valid, params, (1, 0))

    with pytest.raises(ValueError, match="cross-session calibration is forbidden"):
        module.validate_cached_calibration_identity(
            {**valid, "source_camera_params": str(tmp_path / "other.json")},
            params,
            (1, 0),
        )
    with pytest.raises(ValueError, match="sourceIndex"):
        module.validate_cached_calibration_identity(
            {**valid, "physical_eye_source_indices": {"left": 0, "right": 1}},
            params,
            (1, 0),
        )
