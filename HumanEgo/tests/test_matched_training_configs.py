from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from tools.validate_matched_training_configs import (
    bind_selector_image_name,
    config_key_differences,
    image_name_for_selector_product_line,
    require_matched_batch_size,
    validate_matched_config_pair,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "cfg/training/grap_a_cap"


def _load(name: str) -> dict:
    payload = yaml.safe_load((CONFIG_ROOT / name).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_rebuilt_robot_config_differs_only_at_img_name() -> None:
    raw = _load("kai22_retarget_ab_r2.yaml")
    robot = _load("kai22_robotrgb_tianji_kai_v1.yaml")
    report = validate_matched_config_pair(raw, robot, batch_size=64)
    assert report == {
        "status": "PARTIAL_HOLD_SELECTOR_IMG_NAME_UNDEFINED",
        "batch_size": 64,
        "autotune_allowed": False,
        "differences": {"img_name": {"raw": "rgb.png", "robot": None}},
    }


def test_config_validator_rejects_deliberate_second_difference() -> None:
    raw = _load("kai22_retarget_ab_r2.yaml")
    robot = deepcopy(_load("kai22_robotrgb_tianji_kai_v1.yaml"))
    robot["seed"] = 8
    assert sorted(config_key_differences(raw, robot)) == ["img_name", "seed"]
    with pytest.raises(ValueError, match=r"\['img_name', 'seed'\]"):
        validate_matched_config_pair(raw, robot, batch_size=64)


@pytest.mark.parametrize("batch_size", [None, 1, 32, 63, 65, 128])
def test_matched_batch_guard_forbids_missing_or_different_size(
    batch_size: int | None,
) -> None:
    with pytest.raises(ValueError, match="explicit --batch-size 64"):
        require_matched_batch_size(batch_size)


def test_raw_and_robot_product_lines_resolve_only_explicit_image_names() -> None:
    assert image_name_for_selector_product_line(
        {"product_line": "RAW", "image_name": "rgb.png"}
    ) == "rgb.png"
    assert image_name_for_selector_product_line(
        {"product_line": "ROBOT_RGB", "image_name": "robot.png"}
    ) == "robot.png"
    with pytest.raises(ValueError, match="frame-local basename"):
        image_name_for_selector_product_line(
            {"product_line": "ROBOT_RGB", "image_name": "../robot.png"}
        )


def test_selector_binding_rejects_non_raw_check_only_without_manifest() -> None:
    raw = _load("kai22_retarget_ab_r2.yaml")
    assert bind_selector_image_name(raw, None, check_only=True)["img_name"] == "rgb.png"
    robot = _load("kai22_robotrgb_tianji_kai_v1.yaml")
    with pytest.raises(RuntimeError, match="HOLD_MANIFEST_SELECTOR_UNIMPLEMENTED"):
        bind_selector_image_name(robot, None, check_only=True)
