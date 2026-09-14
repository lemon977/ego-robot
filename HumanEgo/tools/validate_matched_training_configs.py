#!/usr/bin/env python3
"""Fail-closed checks for the RAW/RobotRGB matched training pair."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

MATCHED_BATCH_SIZE = 64
RAW_PRODUCT_LINE = "RAW"
ROBOT_PRODUCT_LINE = "ROBOT_RGB"


def image_name_for_selector_product_line(
    selector_manifest: Mapping[str, Any],
) -> str:
    """Resolve only product lines whose reviewed image-location semantics exist."""
    product_line = selector_manifest.get("product_line")
    if product_line == RAW_PRODUCT_LINE:
        if selector_manifest.get("image_name") != "rgb.png":
            raise ValueError("RAW selector image_name must be rgb.png")
        return "rgb.png"
    if product_line == ROBOT_PRODUCT_LINE:
        image_name = selector_manifest.get("image_name")
        if (
            not isinstance(image_name, str)
            or image_name != Path(image_name).name
            or image_name in {"", ".", ".."}
        ):
            raise ValueError("RobotRGB selector image_name must be a frame-local basename")
        return image_name
    raise ValueError(f"unreviewed selector product_line: {product_line!r}")


def bind_selector_image_name(
    config: Mapping[str, Any],
    selector_manifest: Mapping[str, Any] | None,
    *,
    check_only: bool,
) -> dict[str, Any]:
    """Bind img_name to a reviewed selector; never infer a RobotRGB location."""
    if config.get("use_legacy_image_loading") is not False:
        raise ValueError("legacy image loading is forbidden")
    if selector_manifest is None:
        if not check_only:
            raise RuntimeError("HOLD_MANIFEST_SELECTOR_UNIMPLEMENTED")
        if config.get("img_name") != "rgb.png":
            raise RuntimeError(
                "HOLD_MANIFEST_SELECTOR_UNIMPLEMENTED: non-RAW check-only needs "
                "the reviewed selector"
            )
        return dict(config)
    expected = image_name_for_selector_product_line(selector_manifest)
    configured = config.get("img_name")
    if configured not in {None, expected}:
        raise ValueError(
            f"config img_name {configured!r} conflicts with selector product_line"
        )
    resolved = dict(config)
    resolved["img_name"] = expected
    return resolved


def require_matched_batch_size(batch_size: int | None) -> int:
    """Forbid per-arm autotune in the matched full-cohort comparison."""
    if batch_size != MATCHED_BATCH_SIZE:
        raise ValueError(
            f"matched comparison requires explicit --batch-size {MATCHED_BATCH_SIZE}; "
            "CUDA autotune and other batch sizes are forbidden"
        )
    return MATCHED_BATCH_SIZE


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {prefix: value}
    flattened: dict[str, Any] = {}
    for key in sorted(value, key=str):
        child = f"{prefix}.{key}" if prefix else str(key)
        flattened.update(_flatten(value[key], child))
    return flattened


def config_key_differences(
    raw_config: Mapping[str, Any], robot_config: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    raw = _flatten(raw_config)
    robot = _flatten(robot_config)
    missing = object()
    differences: dict[str, dict[str, Any]] = {}
    for key in sorted(set(raw) | set(robot)):
        raw_value = raw.get(key, missing)
        robot_value = robot.get(key, missing)
        if raw_value != robot_value:
            differences[key] = {
                "raw": "<MISSING>" if raw_value is missing else raw_value,
                "robot": "<MISSING>" if robot_value is missing else robot_value,
            }
    return differences


def validate_matched_config_pair(
    raw_config: Mapping[str, Any],
    robot_config: Mapping[str, Any],
    *,
    batch_size: int | None,
) -> dict[str, Any]:
    selected_batch = require_matched_batch_size(batch_size)
    differences = config_key_differences(raw_config, robot_config)
    if set(differences) != {"img_name"}:
        raise ValueError(
            "matched configs must differ only at img_name; observed differences: "
            f"{sorted(differences)}"
        )
    if raw_config.get("img_name") != "rgb.png":
        raise ValueError("RAW matched config must select rgb.png")
    if robot_config.get("img_name") is not None:
        raise ValueError(
            "RobotRGB img_name must remain null until the reviewed selector "
            "defines its image-location semantics"
        )
    if raw_config.get("persistent_workers") is not False:
        raise ValueError("RAW formal config requires persistent_workers=false")
    if robot_config.get("persistent_workers") is not False:
        raise ValueError("RobotRGB formal config requires persistent_workers=false")
    return {
        "status": "PARTIAL_HOLD_SELECTOR_IMG_NAME_UNDEFINED",
        "batch_size": selected_batch,
        "autotune_allowed": False,
        "differences": differences,
    }


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"training config must be a mapping: {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--robot", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    args = parser.parse_args()
    report = validate_matched_config_pair(
        _load_yaml(args.raw),
        _load_yaml(args.robot),
        batch_size=args.batch_size,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
