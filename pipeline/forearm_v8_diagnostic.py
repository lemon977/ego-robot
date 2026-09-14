"""Offline v8 forearm morphology diagnostic from preserved v6/v7 evidence.

The preserved instance mask is an evidence proxy, not a raw SAM forearm mask.
Consequently this diagnostic may reject a proposed morphology route but cannot
authorize a full producer run by itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .mask_producer_sam2 import (
    _anatomy_support_masks,
    _named_point,
    _select_wrist_connected_forearm,
    _valid_hand_points,
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_image_ref(ref: dict[str, Any], flags: int) -> np.ndarray:
    path = Path(ref["path"])
    if _sha256(path) != ref["sha256"]:
        raise RuntimeError(f"evidence SHA mismatch: {path}")
    value = cv2.imread(str(path), flags)
    if value is None:
        raise RuntimeError(f"evidence decode failure: {path}")
    return value


def _mask(ref: dict[str, Any]) -> np.ndarray:
    return _read_image_ref(ref, cv2.IMREAD_GRAYSCALE) > 0


def _overlay(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    result = image.copy()
    color_layer = np.empty_like(result)
    color_layer[:] = color
    result[mask] = cv2.addWeighted(result[mask], 0.48, color_layer[mask], 0.52, 0)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, color, 2)
    return result


def _label(image: np.ndarray, lines: list[str]) -> np.ndarray:
    result = image.copy()
    banner_height = 30 + 22 * (len(lines) - 1)
    cv2.rectangle(result, (0, 0), (result.shape[1], banner_height), (0, 0, 0), -1)
    for index, line in enumerate(lines):
        cv2.putText(
            result,
            line,
            (8, 23 + 22 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return result


def _panel(image: np.ndarray, width: int = 320) -> np.ndarray:
    return cv2.resize(image, (width, int(round(image.shape[0] * width / image.shape[1]))), interpolation=cv2.INTER_AREA)


def build_diagnostic(
    source_manifest_path: Path,
    v6_manifest_path: Path,
    v7_manifest_path: Path,
    frames: list[int],
    output_dir: Path,
    forearm_width_ratio: float,
    wrist_radius_hand_scale: float,
) -> dict[str, Any]:
    source_manifest = _load_json(source_manifest_path)
    v6 = _load_json(v6_manifest_path)
    v7 = _load_json(v7_manifest_path)
    source_session = next(
        item for item in source_manifest["sessions"] if item["session_id"] == v7["session_id"]
    )
    if not (v6["frame_count"] == v7["frame_count"] == len(source_session["frames"])):
        raise RuntimeError("source/v6/v7 frame counts disagree")

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[np.ndarray] = []
    frame_metrics: dict[str, Any] = {}
    for frame_index in frames:
        source = source_session["frames"][frame_index]
        old_frame = v6["frames"][frame_index]
        new_frame = v7["frames"][frame_index]
        if old_frame["frame_index"] != frame_index or new_frame["frame_index"] != frame_index:
            raise RuntimeError(f"manifest order mismatch at {frame_index}")
        raw_path = Path(source["image"]["path"])
        if _sha256(raw_path) != source["image"]["sha256"]:
            raise RuntimeError(f"RAW SHA mismatch at {frame_index}")
        image = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"RAW decode failure at {frame_index}")
        metadata_path = Path(source["metadata"]["path"])
        if _sha256(metadata_path) != source["metadata"]["sha256"]:
            raise RuntimeError(f"metadata SHA mismatch at {frame_index}")
        metadata = _load_json(metadata_path)
        v6_h = _mask(old_frame["h_core"])
        v7_h = _mask(new_frame["h_core"])
        instance = _read_image_ref(old_frame["instance_id"], cv2.IMREAD_GRAYSCALE)
        prompt_path = Path(new_frame["prompt_provenance"]["path"])
        if _sha256(prompt_path) != new_frame["prompt_provenance"]["sha256"]:
            raise RuntimeError(f"prompt SHA mismatch at {frame_index}")
        prompt = _load_json(prompt_path)

        diagnostic = image.copy()
        per_side: dict[str, Any] = {}
        for side, instance_value, color in (
            ("left", 1, (255, 255, 0)),
            ("right", 2, (255, 0, 255)),
        ):
            hand = metadata["entities"]["hands"][side]
            points, _ = _valid_hand_points(hand, image.shape[1], image.shape[0])
            wrist = _named_point(hand, "wrist")
            palm = _named_point(hand, "palm_center")
            wrist_width = float(
                np.linalg.norm(
                    _named_point(hand, "index_proximal")
                    - _named_point(hand, "pinky_proximal")
                )
            )
            _, corridor, support_metrics = _anatomy_support_masks(
                points,
                wrist,
                palm,
                wrist_width,
                forearm_width_ratio=forearm_width_ratio,
                width=image.shape[1],
                height=image.shape[0],
            )
            proxy = instance == instance_value
            hand_scale = float(prompt["sides"][side]["hand_scale_px"])
            wrist_region = np.zeros(proxy.shape, np.uint8)
            cv2.circle(
                wrist_region,
                tuple(np.rint(wrist).astype(int)),
                max(1, int(round(wrist_radius_hand_scale * hand_scale))),
                1,
                -1,
            )
            selected, metrics = _select_wrist_connected_forearm(
                proxy,
                proxy,
                wrist_region.astype(bool),
                corridor,
                wrist,
                palm,
                wrist_width,
            )
            diagnostic = _overlay(diagnostic, selected, color)
            contours, _ = cv2.findContours(
                corridor.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(diagnostic, contours, -1, (0, 255, 255), 2)
            cv2.circle(diagnostic, tuple(np.rint(wrist).astype(int)), 8, color, -1)
            per_side[side] = {
                "evidence_kind": "V6_INSTANCE_SIDE_PROXY_NOT_RAW_FOREARM_SAM",
                "corridor_support": support_metrics,
                "selection": metrics,
            }

        rows.append(
            np.concatenate(
                [
                    _label(_panel(image), [f"RAW {frame_index}"]),
                    _label(_panel(_overlay(image, v6_h, (0, 0, 255))), ["v6 H_core"]),
                    _label(_panel(_overlay(image, v7_h, (0, 0, 255))), ["v7 H_core"]),
                    _label(
                        _panel(diagnostic),
                        [
                            "v8 proxy: L cyan / R magenta",
                            "yellow=old corridor; NOT raw SAM",
                        ],
                    ),
                ],
                axis=1,
            )
        )
        frame_metrics[str(frame_index)] = per_side

    sheet_path = output_dir / "v6_v7_proxy_morphology_232_236_239_240.png"
    if not cv2.imwrite(str(sheet_path), np.concatenate(rows, axis=0)):
        raise RuntimeError("diagnostic contact sheet write failed")
    return {
        "schema_version": "forearm-v8-offline-proxy-diagnostic-v1",
        "session_id": v7["session_id"],
        "frames": frames,
        "evidence_class": "V6_INSTANCE_PROXY_NOT_RAW_FOREARM_SAM",
        "may_reject_algorithm": True,
        "may_authorize_full_h20": False,
        "full_h20_gate": "REQUIRES_TRUE_FOUR_FRAME_SAM_INTERMEDIATE_DIAGNOSTIC",
        "bridge_policy": "DISABLED",
        "frame_metrics": frame_metrics,
        "contact_sheet": {"path": str(sheet_path), "sha256": _sha256(sheet_path)},
        "source_refs": {
            "source_manifest_sha256": _sha256(source_manifest_path),
            "v6_manifest_sha256": _sha256(v6_manifest_path),
            "v7_manifest_sha256": _sha256(v7_manifest_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--v6-manifest", type=Path, required=True)
    parser.add_argument("--v7-manifest", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument("--forearm-width-ratio", type=float, required=True)
    parser.add_argument("--wrist-radius-hand-scale", type=float, required=True)
    args = parser.parse_args()
    result = build_diagnostic(
        args.source_manifest,
        args.v6_manifest,
        args.v7_manifest,
        args.frames,
        args.output_dir,
        args.forearm_width_ratio,
        args.wrist_radius_hand_scale,
    )
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
