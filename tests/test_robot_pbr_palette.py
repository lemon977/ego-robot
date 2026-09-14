from copy import deepcopy
from pathlib import Path
import json

import pytest

from pipeline.robot_pbr_palette import (
    CONFIG_ID,
    CONFIG_RELATIVE_PATH,
    RobotPaletteError,
    linear_channel_to_srgb,
    load_shared_robot_palette,
    srgb_channel_to_linear,
    srgb_rgba_to_bgr8,
    validate_robot_palette,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_frozen_shared_pbr_palette_and_ocio_pin_are_valid() -> None:
    config = load_shared_robot_palette(PROJECT_ROOT)
    assert config["config_id"] == CONFIG_ID
    assert config["color_management"]["view_transform"] == "AgX"
    assert config["color_management"]["look"] == "AgX - Medium High Contrast"
    assert config["assignments"]["left_arm_shell"] == config["assignments"]["right_arm_shell"]
    assert config["assignments"]["left_kaihand_shell"] == config["assignments"]["right_kaihand_shell"]


def test_srgb_linear_roundtrip_and_cpu_preview_are_derived() -> None:
    config = load_shared_robot_palette(PROJECT_ROOT)
    for material in config["materials"].values():
        for srgb in material["base_color_srgb"][:3]:
            assert linear_channel_to_srgb(srgb_channel_to_linear(srgb)) == pytest.approx(
                srgb, abs=1e-12
            )
    kaihand = config["materials"]["KAIHAND_SHELL_STEEL_BLUE"]
    assert srgb_rgba_to_bgr8(kaihand["base_color_srgb"]) == (184, 151, 119)


def test_task_material_override_is_rejected() -> None:
    with pytest.raises(RobotPaletteError, match="overrides are forbidden"):
        load_shared_robot_palette(
            PROJECT_ROOT,
            task_overrides={"materials": {"ARM_SHELL_MATTE_WHITE": {"roughness": 0.1}}},
        )


def test_tampered_stored_linear_or_side_assignment_is_rejected() -> None:
    config = load_shared_robot_palette(PROJECT_ROOT)
    bad_linear = deepcopy(config)
    bad_linear["materials"]["ARM_SHELL_MATTE_WHITE"]["base_color_linear"][0] = 0.0
    with pytest.raises(RobotPaletteError, match="stored linear base color"):
        validate_robot_palette(bad_linear)

    asymmetric = deepcopy(config)
    asymmetric["assignments"]["right_arm_shell"] = "JOINT_FASTENER_GRAPHITE"
    with pytest.raises(RobotPaletteError, match="left/right material mismatch"):
        validate_robot_palette(asymmetric)


def test_task_canary_references_shared_config_without_overrides() -> None:
    reference_path = PROJECT_ROOT / (
        "tasks/chips/runs/robot/"
        "20260903_chips001_palette_004ref_canary12_v1/PBR_CONFIG_REFERENCE.json"
    )
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    assert reference["config_id"] == CONFIG_ID
    assert reference["task_override"] is None
    assert reference["path"] == str(CONFIG_RELATIVE_PATH)
