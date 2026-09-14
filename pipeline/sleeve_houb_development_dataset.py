"""Fail-closed development-only HOUB dataset adapter for sleeve-aware H training.

The frozen benchmark contains development and blind label references in the same
manifests.  This adapter may parse those manifests, but it exposes and opens pixels
for the 15 public development frames only.  Blind frame access fails before any
image or label path is opened.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from pipeline.sam21_route_b_contract import (
    AsymmetricHOUBSupervision,
    DEVELOPMENT_FRAMES,
    fixed_oof_plan,
    make_supervision,
    require_development_split,
)


class SleeveHOUBDatasetError(RuntimeError):
    """Raised when label identity, split or pixel semantics drift."""


PROJECT = Path("/mnt/workspace/code/chaoyang")
FREEZE_ROOT = PROJECT / (
    "_run/g2_mask_benchmark_freeze_v1/"
    "8bdd4cb5e994a9a648adbc20bafb9a59e28a67a1118d1043501159da1888e7ac"
)
PALETTE_SCHEMA = PROJECT / "_run/g2_mask_benchmark_v2/schemas/HOUB_LABEL_SCHEMA.json"
DEVELOPMENT_SESSION = "grap_a_cap_004"
IMAGE_SIZE = (1280, 960)
SYMBOL_TO_VALUE = {"B": 0, "H": 1, "O": 2, "U": 3}
EXPECTED_SHA256 = {
    "BENCHMARK_FREEZE.json": "09f94084a84cd0ac0d3e93a3311da2b41c85788742029e89347b18afb6ed073b",
    "MASK_BENCHMARK.json": "f37b44e5f3fdd495d352705ff0fab471e47d87ba39893efa7ac5d0ab9eabf65c",
    "HOUB_LABELS.json": "0505f60ba0f7b2c2d636db616c4c2e095e30cbf54a59d24ba46520fdc9f161dd",
}
EXPECTED_SCHEMA_SHA256 = "7ff5547b36e218c81587af01dad4da319dac2411a0bb417b30ff87d16ffaa5e6"


@dataclass(frozen=True)
class DevelopmentFrameRef:
    sample_id: str
    frame_index: int
    source_image: dict[str, Any]
    palette: dict[str, Any]
    masks: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class DevelopmentSample:
    ref: DevelopmentFrameRef
    image_rgb: np.ndarray
    palette: np.ndarray
    human: np.ndarray
    object_: np.ndarray
    uncertain: np.ndarray
    background: np.ndarray
    supervision: AsymmetricHOUBSupervision


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SleeveHOUBDatasetError(f"JSON root must be object: {path}")
    return value


def _verify_regular_ref(ref: dict[str, Any], *, expected_parent: Path | None = None) -> Path:
    path = Path(ref.get("path", ""))
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise SleeveHOUBDatasetError(f"reference is not an ordinary absolute file: {path}")
    resolved = path.resolve(strict=True)
    if expected_parent is not None:
        try:
            resolved.relative_to(expected_parent.resolve(strict=True))
        except ValueError as exc:
            raise SleeveHOUBDatasetError(f"reference escaped expected root: {resolved}") from exc
    if path.stat().st_size != int(ref.get("bytes", -1)):
        raise SleeveHOUBDatasetError(f"byte count drift: {path}")
    if sha256_file(path) != ref.get("sha256"):
        raise SleeveHOUBDatasetError(f"SHA256 drift: {path}")
    return path


class SleeveHOUBDevelopmentDataset:
    """Digest-bound reader that cannot open same/cross-session blind pixels."""

    def __init__(self, freeze_root: Path = FREEZE_ROOT) -> None:
        self.freeze_root = Path(freeze_root)
        if self.freeze_root != FREEZE_ROOT:
            raise SleeveHOUBDatasetError("unrecognized freeze root")
        for name, expected in EXPECTED_SHA256.items():
            path = self.freeze_root / name
            if sha256_file(path) != expected:
                raise SleeveHOUBDatasetError(f"frozen manifest drift: {name}")
        if sha256_file(PALETTE_SCHEMA) != EXPECTED_SCHEMA_SHA256:
            raise SleeveHOUBDatasetError("palette schema drift")

        freeze = _read_json(self.freeze_root / "BENCHMARK_FREEZE.json")
        benchmark = _read_json(self.freeze_root / "MASK_BENCHMARK.json")
        labels = _read_json(self.freeze_root / "HOUB_LABELS.json")
        schema = _read_json(PALETTE_SCHEMA)
        if freeze.get("blind_labels_unsealed") is not False:
            raise SleeveHOUBDatasetError("blind label seal is not intact")
        if benchmark.get("artifact_state") != "FROZEN" or labels.get("artifact_state") != "FROZEN":
            raise SleeveHOUBDatasetError("benchmark or labels are not frozen")
        observed_classes = {
            item.get("symbol"): item.get("value") for item in schema.get("classes", [])
        }
        if observed_classes != SYMBOL_TO_VALUE:
            raise SleeveHOUBDatasetError("HOUB palette meaning drift")

        benchmark_rows = {row["sample_id"]: row for row in benchmark.get("frames", [])}
        label_rows = {row["sample_id"]: row for row in labels.get("frames", [])}
        palette_rows = {row["frame"]: row for row in freeze["human_palette"]["refs"]}
        development_rows = [
            row for row in benchmark.get("frames", []) if row.get("split") == "development"
        ]
        observed_frames = tuple(int(row["frame_index"]) for row in development_rows)
        if observed_frames != DEVELOPMENT_FRAMES:
            raise SleeveHOUBDatasetError(f"development identity drift: {observed_frames}")
        fixed_oof_plan(observed_frames)

        refs: dict[int, DevelopmentFrameRef] = {}
        for row in development_rows:
            require_development_split(row["split"])
            frame = int(row["frame_index"])
            sample_id = f"{DEVELOPMENT_SESSION}:{frame:05d}"
            if row.get("sample_id") != sample_id or row.get("session_id") != DEVELOPMENT_SESSION:
                raise SleeveHOUBDatasetError("development sample identity drift")
            label = label_rows.get(sample_id)
            palette = palette_rows.get(sample_id)
            if label is None or palette is None:
                raise SleeveHOUBDatasetError(f"missing development label reference: {sample_id}")
            masks = {
                symbol: label[f"{symbol.lower()}_mask"] for symbol in SYMBOL_TO_VALUE
            }
            expected_filename = f"{frame:05d}.png"
            for symbol, mask_ref in masks.items():
                expected = (
                    self.freeze_root / "labels_0_255" / symbol /
                    DEVELOPMENT_SESSION / expected_filename
                )
                if Path(mask_ref["path"]) != expected:
                    raise SleeveHOUBDatasetError(f"mask path identity drift: {sample_id}:{symbol}")
            refs[frame] = DevelopmentFrameRef(
                sample_id=sample_id,
                frame_index=frame,
                source_image=dict(row["source_image"]),
                palette=dict(palette),
                masks=masks,
            )
        if tuple(refs) != DEVELOPMENT_FRAMES:
            raise SleeveHOUBDatasetError("development reference order drift")
        self._refs = refs
        self.manifest_only_blind_reference_count = len(benchmark_rows) - len(refs)
        self.pixel_access_log: list[str] = []

    def __len__(self) -> int:
        return len(self._refs)

    @property
    def frames(self) -> tuple[int, ...]:
        return tuple(self._refs)

    def reference(self, frame_index: int) -> DevelopmentFrameRef:
        frame = int(frame_index)
        if frame not in self._refs:
            raise SleeveHOUBDatasetError(
                f"supervised frame is sealed or unknown: {frame}"
            )
        return self._refs[frame]

    def load(self, frame_index: int) -> DevelopmentSample:
        ref = self.reference(frame_index)  # split gate happens before any pixel open
        source_path = _verify_regular_ref(ref.source_image)
        palette_path = _verify_regular_ref(ref.palette)
        self.pixel_access_log.extend([ref.sample_id + ":RAW", ref.sample_id + ":PALETTE"])
        with Image.open(source_path) as image:
            if image.size != IMAGE_SIZE:
                raise SleeveHOUBDatasetError("source image size drift")
            image_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        with Image.open(palette_path) as image:
            if image.mode != "P" or image.size != IMAGE_SIZE:
                raise SleeveHOUBDatasetError("palette mode/size drift")
            palette = np.asarray(image, dtype=np.uint8)
        if set(np.unique(palette).tolist()) - set(SYMBOL_TO_VALUE.values()):
            raise SleeveHOUBDatasetError("palette contains forbidden values")

        arrays: dict[str, np.ndarray] = {}
        for symbol, mask_ref in ref.masks.items():
            mask_path = _verify_regular_ref(mask_ref, expected_parent=self.freeze_root)
            self.pixel_access_log.append(ref.sample_id + ":" + symbol)
            with Image.open(mask_path) as image:
                if image.mode != "L" or image.size != IMAGE_SIZE:
                    raise SleeveHOUBDatasetError("binary mask mode/size drift")
                value = np.asarray(image, dtype=np.uint8)
            if set(np.unique(value).tolist()) - {0, 255}:
                raise SleeveHOUBDatasetError("binary mask value drift")
            arrays[symbol] = value == 255
            if not np.array_equal(arrays[symbol], palette == SYMBOL_TO_VALUE[symbol]):
                raise SleeveHOUBDatasetError(f"palette/binary mismatch: {ref.sample_id}:{symbol}")

        supervision = make_supervision(
            arrays["H"], arrays["O"], arrays["U"], arrays["B"]
        )
        return DevelopmentSample(
            ref=ref,
            image_rgb=image_rgb,
            palette=palette,
            human=arrays["H"],
            object_=arrays["O"],
            uncertain=arrays["U"],
            background=arrays["B"],
            supervision=supervision,
        )


def dataset_identity() -> dict[str, Any]:
    return {
        "freeze_root": str(FREEZE_ROOT),
        "manifest_sha256": dict(EXPECTED_SHA256),
        "palette_schema": str(PALETTE_SCHEMA),
        "palette_schema_sha256": EXPECTED_SCHEMA_SHA256,
        "development_session": DEVELOPMENT_SESSION,
        "development_frames": list(DEVELOPMENT_FRAMES),
        "palette": dict(SYMBOL_TO_VALUE),
        "h_semantics": "ego skin + hands + wrist/cuff + forearm + sleeve + human-worn item",
        "subclass_labels_available": False,
        "subclass_note": "hand/forearm/sleeve/tracker are merged into H and cannot be counted separately",
    }
