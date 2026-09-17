"""Deterministic vector-operation rasterization and preview export."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from .schema import BY_ID, CLASSES, KNOWN_IDS


class RasterError(RuntimeError):
    pass


def _point(value: Any, width: int, height: int) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise RasterError("point must be [x, y]")
    x, y = value
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        raise RasterError("point coordinates must be numeric")
    if not math.isfinite(x) or not math.isfinite(y):
        raise RasterError("point coordinates must be finite")
    if not 0 <= x < width or not 0 <= y < height:
        raise RasterError(f"point out of bounds: {(x, y)} for {(width, height)}")
    return float(x), float(y)


def validate_operations(operations: Any, width: int, height: int) -> list[dict[str, Any]]:
    if not isinstance(operations, list):
        raise RasterError("operations must be a list")
    if len(operations) > 20000:
        raise RasterError("too many operations")
    validated = []
    seen_ids = set()
    for index, op in enumerate(operations):
        if not isinstance(op, dict):
            raise RasterError(f"operation {index} must be an object")
        op_id = op.get("op_id")
        if not isinstance(op_id, str) or not op_id or len(op_id) > 100 or op_id in seen_ids:
            raise RasterError(f"invalid or duplicate op_id at operation {index}")
        seen_ids.add(op_id)
        class_id = op.get("class_id")
        if class_id not in KNOWN_IDS:
            raise RasterError(f"unknown class_id at operation {index}: {class_id}")
        kind = op.get("kind")
        points = [_point(point, width, height) for point in op.get("points", [])]
        if kind == "polygon":
            if len(points) < 3:
                raise RasterError(f"polygon operation {index} needs at least 3 points")
            clean = {"op_id": op_id, "kind": kind, "class_id": class_id, "points": points}
        elif kind == "brush":
            if not points:
                raise RasterError(f"brush operation {index} needs at least 1 point")
            radius = op.get("radius")
            if not isinstance(radius, (int, float)) or not math.isfinite(radius):
                raise RasterError(f"invalid brush radius at operation {index}")
            if not 0.5 <= radius <= max(width, height):
                raise RasterError(f"brush radius out of range at operation {index}")
            clean = {
                "op_id": op_id,
                "kind": kind,
                "class_id": class_id,
                "radius": float(radius),
                "points": points,
            }
        else:
            raise RasterError(f"unknown operation kind at operation {index}: {kind!r}")
        validated.append(clean)
    return validated


def rasterize(operations: Any, size: tuple[int, int]) -> Image.Image:
    width, height = size
    validated = validate_operations(operations, width, height)
    mask = Image.new("L", size, color=0)
    draw = ImageDraw.Draw(mask)
    for op in validated:
        value = int(op["class_id"])
        points = [tuple(point) for point in op["points"]]
        if op["kind"] == "polygon":
            draw.polygon(points, fill=value)
        else:
            radius = op["radius"]
            width_px = max(1, int(round(2 * radius)))
            if len(points) > 1:
                draw.line(points, fill=value, width=width_px, joint="curve")
            for x, y in (points[0], points[-1]):
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=value)
    return mask


def binary_mask(class_mask: Image.Image, class_id: int) -> Image.Image:
    if class_id not in KNOWN_IDS:
        raise RasterError(f"unknown class id: {class_id}")
    return class_mask.point(lambda value: 255 if value == class_id else 0, mode="L")


def overlay_image(image: Image.Image, class_mask: Image.Image, alpha: int = 128) -> Image.Image:
    image = image.convert("RGB")
    color = Image.new("RGB", image.size)
    palette = []
    for item in CLASSES:
        value = item["color"].lstrip("#")
        palette.extend(int(value[i:i + 2], 16) for i in (0, 2, 4))
    palette.extend([0, 0, 0] * (256 - len(CLASSES)))
    indexed = class_mask.copy()
    indexed.putpalette(palette)
    color = indexed.convert("RGB")
    blended = Image.blend(image, color, alpha / 255.0)
    foreground = class_mask.point(lambda value: 0 if value == BY_ID[0]["id"] else 255, mode="L")
    return Image.composite(blended, image, foreground)
