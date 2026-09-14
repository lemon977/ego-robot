"""Run an explicitly bounded four-frame SAM forearm diagnostic.

This is not the 460-frame MASK producer and cannot emit a contract MASK
manifest.  It preserves intermediate SAM evidence needed to decide whether a
separately authorized full v8 run is justified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from .mask_producer_sam2 import (
    MaskConfig,
    _InferencePredictor,
    _clamp_px,
    _load_json_nofollow,
    _load_rgb,
    _project_cylinder_mask,
    _read_regular_nofollow,
    _segment_hand_side,
    _sha256_bytes,
    _verify_regular_sha256_nofollow,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_mask(path: Path, mask: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), mask.astype(np.uint8) * 255):
        raise RuntimeError(f"failed to write diagnostic mask: {path}")
    return {"path": str(path), "sha256": _sha256(path), "producer": "v8_sam_diagnostic"}


def _overlay(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    result = image.copy()
    color_layer = np.empty_like(result)
    color_layer[:] = color
    result[mask] = cv2.addWeighted(result[mask], 0.46, color_layer[mask], 0.54, 0)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, color, 2)
    return result


def _panel(image: np.ndarray, text: str, width: int = 320) -> np.ndarray:
    height = int(round(image.shape[0] * width / image.shape[1]))
    value = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    cv2.rectangle(value, (0, 0), (width, 30), (0, 0, 0), -1)
    cv2.putText(value, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    return value


def _validate_diagnostic_output_root(
    output_root: Path,
    *,
    project_root: Path,
    task_id: str,
    session_id: str,
) -> Path:
    expected = (
        project_root
        / "_run"
        / task_id
        / "sessions"
        / session_id
        / "diagnostic"
        / "true_sam"
    ).resolve()
    resolved = output_root.resolve()
    if resolved != expected:
        raise RuntimeError(
            f"diagnostic output must be the task-scoped true_sam root: {expected}"
        )
    return resolved


def run(args: argparse.Namespace) -> None:
    source_path = Path(args.source_manifest).resolve()
    context_path = Path(args.session_context).resolve()
    task_path = Path(args.task_card).resolve()
    output_root = Path(args.output_root).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty diagnostic output: {output_root}")

    source = _load_json_nofollow(source_path, args.source_manifest_sha256)
    context = _load_json_nofollow(context_path, args.session_context_sha256)
    task_payload = _read_regular_nofollow(task_path, args.task_card_sha256)
    task = yaml.safe_load(task_payload.decode("utf-8"))
    if not isinstance(task, dict):
        raise RuntimeError("diagnostic task card must be a YAML object")
    if not (
        task.get("status") == "AUTHORIZED_G2_CALIBRATION"
        and task.get("authorization_scope") == "V8_DIAGNOSTIC_ONLY"
        and task.get("task_id") == "g2_004_mask_clean_candidate_v8"
        and task.get("full_h20_authorized") is False
        and task.get("session_id") == args.session_id
        and task.get("product_line") == "004_CONTACT_GOLD"
    ):
        raise RuntimeError("task card does not authorize v8 diagnostic-only execution")
    output_root = _validate_diagnostic_output_root(
        output_root,
        project_root=Path(__file__).resolve().parents[1],
        task_id=str(task["task_id"]),
        session_id=args.session_id,
    )
    if not (
        context.get("execution_allowed") is True
        and context.get("execution_blockers") == []
        and context.get("task_card_ref", {}).get("sha256") == args.task_card_sha256
        and context.get("source", {}).get("source_manifest_sha256")
        == args.source_manifest_sha256
    ):
        raise RuntimeError("session context does not authorize exact diagnostic inputs")
    frames_requested = list(args.frames)
    if not (1 <= len(frames_requested) <= 24):
        raise RuntimeError("diagnostic frame selector must contain 1-24 frames")
    if frames_requested != sorted(set(frames_requested)):
        raise RuntimeError("diagnostic frames must be unique and ascending")

    sessions = [item for item in source["sessions"] if item["session_id"] == args.session_id]
    if len(sessions) != 1:
        raise RuntimeError("source manifest session identity is not unique")
    session = sessions[0]
    frames = session["frames"]
    if any(index < 0 or index >= len(frames) for index in frames_requested):
        raise RuntimeError("diagnostic frame selector is outside the source manifest")

    overlay = task.get("dimensionless_calibration_overlay")
    if not isinstance(overlay, dict):
        raise RuntimeError("diagnostic task has no dimensionless overlay")
    config = MaskConfig.from_overlay(overlay)
    _verify_regular_sha256_nofollow(Path(args.sam2_checkpoint), args.sam2_checkpoint_sha256)
    _verify_regular_sha256_nofollow(Path(args.sam2_config_file), args.sam2_config_sha256)
    _verify_regular_sha256_nofollow(Path(args.object_pose_npz), args.object_pose_sha256)

    import sys

    os.environ.setdefault("PYTHONPATH", args.sam2_root)
    if args.sam2_root not in sys.path:
        sys.path.insert(0, args.sam2_root)
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    if not torch.cuda.is_available():
        raise RuntimeError("v8 SAM diagnostic requires the task-card-pinned CUDA runtime")
    model = build_sam2(args.sam2_config, args.sam2_checkpoint, device="cuda")
    predictor = _InferencePredictor(SAM2ImagePredictor(model), torch)
    with np.load(args.object_pose_npz, allow_pickle=False) as object_npz:
        transforms = np.asarray(object_npz["T_object_to_camera"], dtype=np.float64)
        radius = float(object_npz["cylinder_radius_m"])
        cylinder_height = float(object_npz["cylinder_height_m"])
    if transforms.shape != (len(frames), 4, 4):
        raise RuntimeError("Object6D frame count/shape mismatch")

    output_root.mkdir(parents=True, exist_ok=False)
    rows: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    for frame_index in frames_requested:
        frame = frames[frame_index]
        if int(frame["frame_index"]) != frame_index:
            raise RuntimeError("source frame order mismatch")
        rgb = _load_rgb(frame)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        height, width = rgb.shape[:2]
        metadata = _load_json_nofollow(
            Path(frame["metadata"]["path"]), frame["metadata"]["sha256"]
        )
        if int(metadata["metadata"]["idx"]) != frame_index:
            raise RuntimeError("metadata frame identity mismatch")
        intrinsics = np.asarray(metadata["metadata"]["k"], dtype=np.float64)
        analytic_object, _ = _project_cylinder_mask(
            transforms[frame_index], intrinsics, radius, cylinder_height, width, height
        )
        object_scale = math.sqrt(float(np.count_nonzero(analytic_object)))
        contact = _clamp_px(
            config.contact_ratio * object_scale,
            min(width, height),
            config.contact_min,
            config.contact_max,
        )
        contact_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * contact + 1, 2 * contact + 1)
        )
        exclusion = cv2.dilate(
            analytic_object.astype(np.uint8), contact_kernel
        ).astype(bool)
        predictor.set_image(rgb)

        side_debug: dict[str, dict[str, np.ndarray]] = {}
        side_metrics: dict[str, Any] = {}
        selected: dict[str, np.ndarray] = {}
        outputs: dict[str, Any] = {}
        for side in ("left", "right"):
            debug: dict[str, np.ndarray] = {}
            mask, _, _, metrics = _segment_hand_side(
                predictor,
                metadata["entities"]["hands"][side],
                width,
                height,
                config,
                exclusion,
                debug_masks=debug,
            )
            side_debug[side] = debug
            side_metrics[side] = metrics
            selected[side] = mask
            side_dir = output_root / "frames" / f"{frame_index:05d}" / side
            outputs[side] = {
                name: _write_mask(side_dir / f"{name}.png", value)
                for name, value in debug.items()
            }

        combined = (selected["left"] | selected["right"]) & ~analytic_object
        object_overlap = int(np.count_nonzero(combined & analytic_object))
        outputs["combined_object_protected"] = _write_mask(
            output_root / "frames" / f"{frame_index:05d}" / "combined_object_protected.png",
            combined,
        )

        candidate_panel = bgr.copy()
        selected_panel = bgr.copy()
        support_panel = bgr.copy()
        for side, color in (("left", (255, 255, 0)), ("right", (255, 0, 255))):
            candidate_panel = _overlay(candidate_panel, side_debug[side]["forearm_candidate"], color)
            selected_panel = _overlay(selected_panel, selected[side], color)
            support_panel = _overlay(support_panel, side_debug[side]["selected_combined"], color)
            contours, _ = cv2.findContours(
                side_debug[side]["prior_corridor"].astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(support_panel, contours, -1, (0, 255, 255), 2)
        rows.append(
            np.concatenate(
                [
                    _panel(bgr, f"RAW {frame_index}"),
                    _panel(candidate_panel, "observed SAM forearm candidates"),
                    _panel(selected_panel, "v8 selected L/R"),
                    _panel(support_panel, "yellow=prior corridor only"),
                ],
                axis=1,
            )
        )
        records.append(
            {
                "frame_index": frame_index,
                "source_image_sha256": frame["image"]["sha256"],
                "object_analytic_overlap_pixels": object_overlap,
                "sides": side_metrics,
                "outputs": outputs,
            }
        )

    sheet = output_root / "TRUE_SAM_4FRAME_DIAGNOSTIC.png"
    if not cv2.imwrite(str(sheet), np.concatenate(rows, axis=0)):
        raise RuntimeError("four-frame diagnostic contact sheet write failed")
    manifest = {
        "schema_version": "forearm-v8-true-sam-diagnostic-v1",
        "artifact_state": "G2_DIAGNOSTIC_ONLY_NOT_MASK_CANDIDATE",
        "session_id": args.session_id,
        "frames": frames_requested,
        "full_h20_authorized": False,
        "clean_allowed": False,
        "sam3_used": False,
        "source_manifest_ref": {
            "path": str(source_path),
            "sha256": args.source_manifest_sha256,
            "producer": "source_resolver",
        },
        "session_context_ref": {
            "path": str(context_path),
            "sha256": args.session_context_sha256,
            "producer": "session_profiler",
        },
        "task_card_ref": {
            "path": str(task_path),
            "sha256": args.task_card_sha256,
            "producer": "explicit_task_card",
        },
        "model": {
            "identity": "SAM2_1_HIERA_LARGE",
            "config_sha256": args.sam2_config_sha256,
            "checkpoint_sha256": args.sam2_checkpoint_sha256,
        },
        "records": records,
        "contact_sheet": {"path": str(sheet), "sha256": _sha256(sheet)},
    }
    encoded = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    manifest_path = output_root / "DIAGNOSTIC_MANIFEST.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_bytes(encoded)
    os.replace(temporary, manifest_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--session-context", required=True)
    parser.add_argument("--session-context-sha256", required=True)
    parser.add_argument("--task-card", required=True)
    parser.add_argument("--task-card-sha256", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--sam2-root", required=True)
    parser.add_argument("--sam2-config", required=True)
    parser.add_argument("--sam2-config-file", required=True)
    parser.add_argument("--sam2-config-sha256", required=True)
    parser.add_argument("--sam2-checkpoint", required=True)
    parser.add_argument("--sam2-checkpoint-sha256", required=True)
    parser.add_argument("--object-pose-npz", required=True)
    parser.add_argument("--object-pose-sha256", required=True)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
