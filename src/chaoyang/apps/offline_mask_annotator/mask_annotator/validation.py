"""Fail-closed project and export validation."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops

from .config import AppConfig
from .project import (
    EXPORT_SCHEMA,
    atomic_json,
    load_annotation,
    load_project,
    sha256_file,
)
from .raster import RasterError, binary_mask, rasterize, validate_operations
from .schema import BY_SYMBOL, CLASSES, KNOWN_IDS


def _check_ref(root: Path, ref: dict[str, Any], errors: list[str]) -> Path:
    path = root / ref.get("relpath", "")
    if not path.is_file():
        errors.append(f"missing export file: {path}")
        return path
    if path.stat().st_size != ref.get("bytes"):
        errors.append(f"export byte count drift: {path}")
    if sha256_file(path) != ref.get("sha256"):
        errors.append(f"export checksum drift: {path}")
    return path


def validate_project(config: AppConfig) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    records = []
    class_pixels: Counter[str] = Counter()
    try:
        manifest = load_project(config)
    except Exception as exc:
        result = {"status": "FAIL", "errors": [str(exc)], "warnings": [], "frames": []}
        atomic_json(config.output_dir / "VALIDATION.json", result)
        return result

    for symbol in config.required_nonempty_classes:
        if symbol not in BY_SYMBOL:
            errors.append(f"unknown required_nonempty_class in config: {symbol}")

    for frame in manifest.get("frames", []):
        frame_errors = []
        image_path = config.output_dir / frame["image_relpath"]
        if not image_path.is_file():
            frame_errors.append("SOURCE_FRAME_MISSING")
            errors.append(f"source frame missing: {image_path}")
            continue
        if sha256_file(image_path) != frame["image_sha256"]:
            frame_errors.append("SOURCE_FRAME_CHECKSUM_DRIFT")
            errors.append(f"source frame checksum drift: {image_path}")
        with Image.open(image_path) as image:
            image.load()
            observed_size = image.size
        expected_size = (frame["image_width"], frame["image_height"])
        if observed_size != expected_size:
            frame_errors.append("SOURCE_FRAME_SIZE_DRIFT")
            errors.append(f"source frame size drift: {image_path}")
        try:
            annotation = load_annotation(config, frame)
            operations = validate_operations(annotation.get("operations"), *expected_size)
            class_mask = rasterize(operations, expected_size)
        except (Exception, RasterError) as exc:
            frame_errors.append("ANNOTATION_INVALID")
            errors.append(f"annotation invalid {frame['frame_key']}: {exc}")
            continue
        if config.require_complete and not annotation.get("complete"):
            frame_errors.append("ANNOTATION_NOT_MARKED_COMPLETE")
            errors.append(f"annotation not marked complete: {frame['frame_key']}")
        draft_relpath = annotation.get("draft_class_id_relpath")
        draft_path = config.output_dir / draft_relpath if isinstance(draft_relpath, str) else None
        if draft_path is None or not draft_path.is_file():
            frame_errors.append("DRAFT_CLASS_ID_MISSING")
            errors.append(f"draft class-id PNG missing: {frame['frame_key']}")
        else:
            if draft_path.stat().st_size != annotation.get("draft_class_id_bytes"):
                frame_errors.append("DRAFT_CLASS_ID_BYTE_DRIFT")
                errors.append(f"draft class-id byte count drift: {draft_path}")
            if sha256_file(draft_path) != annotation.get("draft_class_id_sha256"):
                frame_errors.append("DRAFT_CLASS_ID_CHECKSUM_DRIFT")
                errors.append(f"draft class-id checksum drift: {draft_path}")
            with Image.open(draft_path) as draft:
                draft.load()
                if draft.mode != "L" or draft.size != expected_size:
                    frame_errors.append("DRAFT_CLASS_ID_MODE_SIZE_DRIFT")
                    errors.append(f"draft class-id mode/size drift: {draft_path}")
                elif ImageChops.difference(draft, class_mask).getbbox() is not None:
                    frame_errors.append("DRAFT_CLASS_ID_PIXEL_DRIFT")
                    errors.append(f"draft class-id disagrees with annotation JSON: {draft_path}")
        histogram = class_mask.histogram()
        per_class = {item["symbol"]: histogram[item["id"]] for item in CLASSES}
        class_pixels.update(per_class)
        unknown_pixels = sum(value for index, value in enumerate(histogram) if index not in KNOWN_IDS)
        if unknown_pixels:
            frame_errors.append("UNKNOWN_CLASS_PIXELS")
            errors.append(f"unknown class pixels in {frame['frame_key']}: {unknown_pixels}")
        if sum(per_class.values()) != expected_size[0] * expected_size[1]:
            frame_errors.append("NONEXHAUSTIVE_CLASS_MASK")
            errors.append(f"non-exhaustive class mask: {frame['frame_key']}")
        records.append({
            "frame_key": frame["frame_key"],
            "complete": bool(annotation.get("complete")),
            "operation_count": len(operations),
            "class_pixels": per_class,
            "errors": frame_errors,
        })

    empty_classes = [item["symbol"] for item in CLASSES if class_pixels[item["symbol"]] == 0]
    for symbol in empty_classes:
        message = f"class is empty across project: {symbol}"
        if symbol in config.required_nonempty_classes:
            errors.append(message)
        else:
            warnings.append(message)

    export_manifest_path = config.output_dir / "EXPORT_MANIFEST.json"
    export_checked = False
    if export_manifest_path.is_file():
        export_checked = True
        try:
            import json
            export = json.loads(export_manifest_path.read_text(encoding="utf-8"))
            if export.get("schema_version") != EXPORT_SCHEMA:
                errors.append("unsupported EXPORT_MANIFEST schema")
            exported_by_key = {row["frame_key"]: row for row in export.get("frames", [])}
            if len(exported_by_key) != len(manifest["frames"]):
                errors.append("export frame count does not match project")
            for frame in manifest["frames"]:
                row = exported_by_key.get(frame["frame_key"])
                if row is None:
                    errors.append(f"missing export frame record: {frame['frame_key']}")
                    continue
                annotation = load_annotation(config, frame)
                expected = rasterize(annotation["operations"], (frame["image_width"], frame["image_height"]))
                class_path = _check_ref(config.output_dir, row["class_id"], errors)
                overlay_path = _check_ref(config.output_dir, row["overlay"], errors)
                if class_path.is_file():
                    with Image.open(class_path) as observed:
                        observed.load()
                        if observed.mode != "L" or observed.size != expected.size:
                            errors.append(f"class-id mode/size mismatch: {class_path}")
                        elif ImageChops.difference(observed, expected).getbbox() is not None:
                            errors.append(f"class-id pixels disagree with annotation: {class_path}")
                if overlay_path.is_file():
                    with Image.open(overlay_path) as observed:
                        if observed.size != expected.size:
                            errors.append(f"overlay size mismatch: {overlay_path}")
                for item in CLASSES:
                    ref = row.get("binary", {}).get(item["symbol"])
                    if not isinstance(ref, dict):
                        errors.append(f"missing binary ref {frame['frame_key']}:{item['symbol']}")
                        continue
                    binary_path = _check_ref(config.output_dir, ref, errors)
                    if not binary_path.is_file():
                        continue
                    expected_binary = binary_mask(expected, item["id"])
                    with Image.open(binary_path) as observed:
                        observed.load()
                        values = set(observed.getdata())
                        if observed.mode != "L" or observed.size != expected.size:
                            errors.append(f"binary mode/size mismatch: {binary_path}")
                        elif not values <= {0, 255}:
                            errors.append(f"binary mask has non-binary values: {binary_path}")
                        elif ImageChops.difference(observed, expected_binary).getbbox() is not None:
                            errors.append(f"binary mask disagrees with class-id: {binary_path}")
        except Exception as exc:
            errors.append(f"EXPORT_MANIFEST invalid: {exc}")
    else:
        warnings.append("EXPORT_MANIFEST.json not found; run export before final delivery")

    result = {
        "schema_version": "offline-mask-validation-v1",
        "status": "PASS" if not errors else "FAIL",
        "claim_limit": "STRUCTURAL_VALIDATION_ONLY_NOT_SEMANTIC_CORRECTNESS",
        "project_manifest": str(config.output_dir / "FRAME_MANIFEST.json"),
        "frame_count": len(manifest.get("frames", [])),
        "validated_frame_count": len(records),
        "export_checked": export_checked,
        "class_pixels_across_project": dict(class_pixels),
        "empty_classes": empty_classes,
        "errors": errors,
        "warnings": warnings,
        "frames": records,
    }
    atomic_json(config.output_dir / "VALIDATION.json", result)
    return result
