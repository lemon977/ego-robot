"""Shared, immutable robot PBR palette loading and color conversion.

The PBR JSON in ``systems/robot/configs`` is the material authority. OpenCV
BGR colors are derived previews and must never flow back into Cycles inputs.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


CONFIG_ID = "robot.pbr_palette.004ref.v1"
CONFIG_RELATIVE_PATH = Path(
    "systems/robot/configs/robot_pbr_palette_004ref_v1.json"
)


class RobotPaletteError(ValueError):
    """Raised when the shared palette contract is invalid or overridden."""


def srgb_channel_to_linear(value: float) -> float:
    """Convert one IEC 61966-2-1 sRGB channel to scene-linear."""

    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise RobotPaletteError(f"sRGB channel outside [0, 1]: {value}")
    if value <= 0.04045:
        return value / 12.92
    return ((value + 0.055) / 1.055) ** 2.4


def linear_channel_to_srgb(value: float) -> float:
    """Convert one scene-linear Rec.709 channel to IEC sRGB encoding."""

    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise RobotPaletteError(f"linear channel outside [0, 1]: {value}")
    if value <= 0.0031308:
        return 12.92 * value
    return 1.055 * (value ** (1.0 / 2.4)) - 0.055


def srgb_rgba_to_linear(rgba: Sequence[float]) -> tuple[float, float, float, float]:
    if len(rgba) != 4:
        raise RobotPaletteError("base color must be RGBA")
    alpha = float(rgba[3])
    if not 0.0 <= alpha <= 1.0:
        raise RobotPaletteError(f"alpha outside [0, 1]: {alpha}")
    return tuple(srgb_channel_to_linear(value) for value in rgba[:3]) + (alpha,)


def srgb_rgba_to_bgr8(rgba: Sequence[float]) -> tuple[int, int, int]:
    """Derive a non-authoritative OpenCV preview color."""

    if len(rgba) != 4:
        raise RobotPaletteError("base color must be RGBA")
    rgb8 = tuple(int(math.floor(float(value) * 255.0 + 0.5)) for value in rgba[:3])
    if any(value < 0 or value > 255 for value in rgb8):
        raise RobotPaletteError(f"sRGB preview outside uint8: {rgb8}")
    return rgb8[2], rgb8[1], rgb8[0]


def load_shared_robot_palette(
    project_root: Path,
    *,
    task_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load and validate the sole system-owned palette.

    A task is allowed to store the config ID as a reference. Any material or
    color-management override is rejected at this boundary.
    """

    if task_overrides:
        raise RobotPaletteError(
            "task/scene overrides are forbidden; reference the shared config_id only"
        )
    config_path = Path(project_root) / CONFIG_RELATIVE_PATH
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_robot_palette(config)
    return config


def validate_robot_palette(config: Mapping[str, Any]) -> None:
    if config.get("config_id") != CONFIG_ID:
        raise RobotPaletteError(f"unexpected config_id: {config.get('config_id')}")
    if config.get("status") != "FROZEN_SHARED_AUTHORITY":
        raise RobotPaletteError("palette must be a frozen shared authority")

    color_management = config.get("color_management", {})
    required_cm = {
        "scene_linear_role": "Linear Rec.709",
        "base_color_input_space": "sRGB",
        "display_device": "sRGB",
        "view_transform": "AgX",
        "look": "AgX - Medium High Contrast",
        "exposure": 0.0,
        "gamma": 1.0,
        "output_display_space": "sRGB",
        "view_settings_override_forbidden": True,
    }
    for key, expected in required_cm.items():
        if color_management.get(key) != expected:
            raise RobotPaletteError(
                f"color management pin mismatch for {key}: "
                f"{color_management.get(key)!r} != {expected!r}"
            )

    policy = config.get("override_policy", {})
    required_policy_flags = (
        "task_may_reference_config_id_only",
        "task_material_override_forbidden",
        "scene_material_override_forbidden",
        "side_specific_override_forbidden",
        "video_specific_override_forbidden",
        "frame_specific_override_forbidden",
    )
    if any(policy.get(flag) is not True for flag in required_policy_flags):
        raise RobotPaletteError("all material override policy flags must be true")

    materials = config.get("materials")
    if not isinstance(materials, Mapping) or not materials:
        raise RobotPaletteError("materials must be a nonempty object")
    preview = config.get("cpu_preview", {}).get("bgr_uint8", {})
    for name, material in materials.items():
        srgb = material.get("base_color_srgb", [])
        linear = material.get("base_color_linear", [])
        derived_linear = srgb_rgba_to_linear(srgb)
        if len(linear) != 4 or any(
            not math.isclose(float(actual), expected, abs_tol=1e-8)
            for actual, expected in zip(linear, derived_linear)
        ):
            raise RobotPaletteError(f"{name}: stored linear base color is not derived from sRGB")
        if tuple(preview.get(name, ())) != srgb_rgba_to_bgr8(srgb):
            raise RobotPaletteError(f"{name}: CPU BGR preview is not derived from sRGB")
        for scalar in ("roughness", "metallic", "specular_ior_level", "alpha"):
            value = material.get(scalar)
            if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
                raise RobotPaletteError(f"{name}: invalid {scalar}={value!r}")
        if not math.isclose(float(material["alpha"]), float(srgb[3]), abs_tol=1e-12):
            raise RobotPaletteError(f"{name}: alpha disagrees with base color alpha")

    assignments = config.get("assignments", {})
    if set(assignments.values()) - set(materials):
        raise RobotPaletteError("an assignment references an unknown material")
    symmetric_pairs = (
        ("left_arm_shell", "right_arm_shell"),
        ("left_connector_flange", "right_connector_flange"),
        ("left_kaihand_shell", "right_kaihand_shell"),
    )
    for left, right in symmetric_pairs:
        if assignments.get(left) != assignments.get(right):
            raise RobotPaletteError(f"left/right material mismatch: {left}, {right}")
