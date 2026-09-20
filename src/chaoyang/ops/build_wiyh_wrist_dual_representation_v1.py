#!/usr/bin/env python3
"""Build the development-only WIYH wrist dual representation V1.

This operation is intentionally not self-authorizing.  It may only be invoked
through ``chaoyang run`` after the coordinator binds it into the current
algorithm contract and publishes an executable task packet.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any

import cv2
import jsonschema
import numpy as np

from chaoyang.research.world_in_your_hands.wrist_dual_representation_v1 import (
    SIDES,
    WristDualRepresentationError,
    admit_visible_wrist_surface,
    bounded_nonaccumulating_fusion,
    compose_camera_wrist,
    evaluate_static_calibration,
    fit_m1_translation,
    fit_m2_se3,
    validate_temporal_authority,
)


SCHEMA_VERSION = "chaoyang-wrist-dual-representation-v1"
CONFIG_VERSION = "chaoyang-wrist-dual-producer-config-v1"
MODELS = (
    "M0_LEGACY",
    "M1_CONTROLLER_LOCAL_TRANSLATION",
    "M2_STATIC_SE3",
)
COLORS = {
    "M0_LEGACY": (160, 160, 160),
    "STATIC_SELECTED": (0, 220, 255),
    "ANATOMICAL_OBSERVED": (0, 145, 255),
    "VISIBLE_SURFACE": (220, 40, 220),
    "FUSED": (255, 255, 40),
}


class WristDualProducerError(RuntimeError):
    """Raised when a producer input or immutable output contract is unsafe."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    value = path.resolve(strict=True)
    return {"path": str(value), "bytes": value.stat().st_size, "sha256": _sha256(value)}


def _atomic_json(path: Path, value: MappingLike) -> None:
    temporary = Path(tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)[1])
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


MappingLike = dict[str, Any]


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {key: np.asarray(source[key]) for key in source.files}


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    repo_root = Path(__file__).resolve().parents[3]
    config_schema = json.loads(
        (repo_root / "contracts/wrist_dual_producer_config_v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    jsonschema.validate(config, config_schema)
    if config.get("schema_version") != CONFIG_VERSION:
        raise WristDualProducerError("unexpected dual-wrist producer config schema")
    return config


def _require(bundle: dict[str, np.ndarray], name: str, shape: tuple[int | None, ...]) -> np.ndarray:
    if name not in bundle:
        raise WristDualProducerError(f"missing input array: {name}")
    value = np.asarray(bundle[name])
    if value.ndim != len(shape) or any(
        expected is not None and actual != expected
        for actual, expected in zip(value.shape, shape, strict=True)
    ):
        raise WristDualProducerError(f"{name} shape drift: {value.shape} != {shape}")
    return value


def preflight(*, input_npz: Path, config_path: Path) -> dict[str, Any]:
    """Read-only CPU validation for a future governed producer attempt."""

    source_path = input_npz.resolve(strict=True)
    config_file = config_path.resolve(strict=True)
    config = _load_config(config_file)
    bundle = _load_npz(source_path)
    frame_id = _require(bundle, "frame_id", (None,))
    frame_count = len(frame_id)
    _require(bundle, "timestamp_s", (frame_count,))
    recording = _require(bundle, "recording_id", (frame_count,)).astype("U128")
    _require(bundle, "T_camera_controller_raw", (frame_count, 2, 4, 4))
    _require(bundle, "T_controller_wrist_M0", (2, 4, 4))
    _require(bundle, "observed_T_camera_wrist", (frame_count, 2, 4, 4))
    _require(bundle, "observed_valid", (frame_count, 2))
    _require(bundle, "orientation_valid", (frame_count, 2))
    _surface_arrays(bundle, frame_count)
    fit_ids = {str(value) for value in config["fit_recording_ids"]}
    forbidden_ids = {str(value) for value in config["forbidden_fit_recording_ids"]}
    if fit_ids & forbidden_ids:
        raise WristDualProducerError("fit and forbidden recording IDs overlap")
    present = {str(value) for value in recording}
    missing_fit = sorted(fit_ids - present)
    if missing_fit:
        raise WristDualProducerError(f"authorized fit recordings are missing: {missing_fit}")
    for value in config["temporal_authority"].values():
        validate_temporal_authority(str(value))
    return {
        "schema_version": "chaoyang-wrist-dual-preflight-v1",
        "status": "PASS_CPU_PREFLIGHT",
        "input": _artifact(source_path),
        "config": _artifact(config_file),
        "frame_count": frame_count,
        "recording_ids": sorted(present),
        "fit_recording_ids": sorted(fit_ids),
        "forbidden_fit_recording_ids": sorted(forbidden_ids),
        "writes_performed": False,
        "gpu_used": False,
    }


def _surface_arrays(bundle: dict[str, np.ndarray], frame_count: int) -> dict[str, np.ndarray]:
    shape = (frame_count, 2)
    source_uv = _require(bundle, "surface_source_pixel_uv", (frame_count, 2, 2))
    patches = _require(bundle, "surface_patch_xyz_camera", (frame_count, 2, None, 3))
    flags = {
        name: _require(bundle, name, shape).astype(bool)
        for name in (
            "rgb_wrist_region_valid",
            "source_pixel_traceable",
            "mask_purity_valid",
            "depth_rgb_registration_valid",
            "local_depth_continuity_valid",
            "contaminant_free",
        )
    }
    roles = _require(bundle, "surface_role", shape).astype("U32")
    point = np.full((frame_count, 2, 3), np.nan, dtype=np.float64)
    admitted_uv = np.full((frame_count, 2, 2), np.nan, dtype=np.float64)
    valid = np.zeros(shape, dtype=bool)
    region_only = np.zeros(shape, dtype=bool)
    blocker = np.full(shape, "", dtype="U48")
    for frame in range(frame_count):
        for side in range(2):
            admission = admit_visible_wrist_surface(
                source_pixel_uv=source_uv[frame, side],
                surface_patch_xyz_camera=patches[frame, side],
                surface_role=str(roles[frame, side]),
                **{name: bool(values[frame, side]) for name, values in flags.items()},
            )
            point[frame, side] = admission.point_camera
            admitted_uv[frame, side] = admission.source_pixel_uv
            valid[frame, side] = admission.surface_point_valid
            region_only[frame, side] = admission.region_registration_only
            blocker[frame, side] = admission.blocker or ""
    return {
        "visible_wrist_surface_point_camera": point,
        "visible_wrist_surface_source_pixel_uv": admitted_uv,
        "visible_wrist_surface_valid": valid,
        "visible_wrist_region_registration_only": region_only,
        "visible_wrist_surface_blocker": blocker,
        "visible_wrist_surface_role": roles,
    }


def _fit_models(
    bundle: dict[str, np.ndarray],
    config: dict[str, Any],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    controller = bundle["T_camera_controller_raw"]
    observed = bundle["observed_T_camera_wrist"]
    observed_valid = bundle["observed_valid"].astype(bool)
    orientation_valid = bundle["orientation_valid"].astype(bool)
    nominal = bundle["T_controller_wrist_M0"]
    recording = bundle["recording_id"].astype("U128")
    fit_ids = {str(value) for value in config["fit_recording_ids"]}
    forbidden_ids = {str(value) for value in config.get("forbidden_fit_recording_ids", [])}
    if fit_ids & forbidden_ids:
        raise WristDualProducerError("fit and forbidden recording IDs overlap")
    fit_frame = np.isin(recording, sorted(fit_ids))
    if not fit_frame.any():
        raise WristDualProducerError("no frame belongs to an authorized fit recording")
    result: dict[str, dict[str, np.ndarray]] = {}
    audit: dict[str, Any] = {
        "fit_recording_ids": sorted(fit_ids),
        "forbidden_fit_recording_ids": sorted(forbidden_ids),
        "holdout_consumed": False,
        "sides": {},
    }
    fit_recordings_present = sorted({str(value) for value in recording[fit_frame]})
    for side_index, side in enumerate(SIDES):
        fit_valid = fit_frame & observed_valid[:, side_index]
        models: dict[str, np.ndarray] = {"M0_LEGACY": nominal[side_index].copy()}
        rows: dict[str, Any] = {
            "M0_LEGACY": {
                "status": "AVAILABLE",
                "evaluation_all": evaluate_static_calibration(
                    controller[:, side_index],
                    nominal[side_index],
                    observed[:, side_index],
                    observed_valid[:, side_index],
                ),
            }
        }
        try:
            m1, evidence = fit_m1_translation(
                controller[:, side_index],
                observed[:, side_index, :3, 3],
                nominal[side_index],
                fit_valid,
            )
            models["M1_CONTROLLER_LOCAL_TRANSLATION"] = m1
            rows["M1_CONTROLLER_LOCAL_TRANSLATION"] = {
                "status": "AVAILABLE",
                "fit": evidence,
                "evaluation_all": evaluate_static_calibration(
                    controller[:, side_index], m1, observed[:, side_index], observed_valid[:, side_index]
                ),
            }
        except WristDualRepresentationError as error:
            rows["M1_CONTROLLER_LOCAL_TRANSLATION"] = {
                "status": "BLOCKED_STATIC_CALIBRATION",
                "reason": str(error),
            }
        try:
            m2, evidence = fit_m2_se3(
                controller[:, side_index],
                observed[:, side_index],
                valid=fit_valid,
                orientation_valid=orientation_valid[:, side_index],
                minimum_orientation_samples=int(config.get("minimum_m2_orientation_samples", 8)),
                minimum_controller_rotation_span_deg=float(
                    config.get("minimum_m2_controller_rotation_span_deg", 10.0)
                ),
            )
            models["M2_STATIC_SE3"] = m2
            rows["M2_STATIC_SE3"] = {
                "status": "AVAILABLE",
                "fit": evidence,
                "evaluation_all": evaluate_static_calibration(
                    controller[:, side_index], m2, observed[:, side_index], observed_valid[:, side_index]
                ),
            }
        except WristDualRepresentationError as error:
            rows["M2_STATIC_SE3"] = {
                "status": "BLOCKED_STATIC_CALIBRATION",
                "reason": str(error),
            }
        selected = str(config["selected_model"][side])
        if selected not in MODELS:
            raise WristDualProducerError(f"unsupported selected model for {side}: {selected}")
        if selected not in models:
            raise WristDualProducerError(f"selected calibration model is unavailable for {side}: {selected}")
        loo: dict[str, Any] = {}
        for held_out in fit_recordings_present:
            train = fit_frame & (recording != held_out) & observed_valid[:, side_index]
            evaluate = (recording == held_out) & observed_valid[:, side_index]
            fold: dict[str, Any] = {
                "fit_recording_ids": [
                    value for value in fit_recordings_present if value != held_out
                ],
                "held_out_recording_id": held_out,
                "models": {},
            }
            fold["models"]["M0_LEGACY"] = {
                "status": "AVAILABLE",
                "evaluation": evaluate_static_calibration(
                    controller[:, side_index],
                    nominal[side_index],
                    observed[:, side_index],
                    evaluate,
                ),
            }
            try:
                fold_m1, _ = fit_m1_translation(
                    controller[:, side_index],
                    observed[:, side_index, :3, 3],
                    nominal[side_index],
                    train,
                )
                fold["models"]["M1_CONTROLLER_LOCAL_TRANSLATION"] = {
                    "status": "AVAILABLE",
                    "evaluation": evaluate_static_calibration(
                        controller[:, side_index],
                        fold_m1,
                        observed[:, side_index],
                        evaluate,
                    ),
                }
            except WristDualRepresentationError as error:
                fold["models"]["M1_CONTROLLER_LOCAL_TRANSLATION"] = {
                    "status": "BLOCKED_STATIC_CALIBRATION",
                    "reason": str(error),
                }
            try:
                fold_m2, _ = fit_m2_se3(
                    controller[:, side_index],
                    observed[:, side_index],
                    valid=train,
                    orientation_valid=orientation_valid[:, side_index],
                    minimum_orientation_samples=int(
                        config.get("minimum_m2_orientation_samples", 8)
                    ),
                    minimum_controller_rotation_span_deg=float(
                        config.get("minimum_m2_controller_rotation_span_deg", 10.0)
                    ),
                )
                fold["models"]["M2_STATIC_SE3"] = {
                    "status": "AVAILABLE",
                    "evaluation": evaluate_static_calibration(
                        controller[:, side_index],
                        fold_m2,
                        observed[:, side_index],
                        evaluate,
                    ),
                }
            except WristDualRepresentationError as error:
                fold["models"]["M2_STATIC_SE3"] = {
                    "status": "BLOCKED_STATIC_CALIBRATION",
                    "reason": str(error),
                }
            loo[held_out] = fold
        result[side] = models
        rows["selected_model"] = selected
        rows["leave_one_recording_out"] = loo
        audit["sides"][side] = rows
    return result, audit


def _stats(value: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    points = np.asarray(value, dtype=np.float64)[valid]
    if not len(points):
        return {"valid_frames": 0, "step_mm_p50": None, "step_mm_p95": None}
    step = np.linalg.norm(np.diff(points, axis=0), axis=1) * 1000.0
    return {
        "valid_frames": int(len(points)),
        "step_mm_p50": float(np.percentile(step, 50)) if len(step) else None,
        "step_mm_p95": float(np.percentile(step, 95)) if len(step) else None,
    }


def _draw_point(image: np.ndarray, uv: np.ndarray, color: tuple[int, int, int], label: str) -> None:
    if not np.isfinite(uv).all():
        return
    x, y = np.rint(uv).astype(int)
    if not (0 <= x < image.shape[1] and 0 <= y < image.shape[0]):
        return
    cv2.circle(image, (int(x), int(y)), 9, color, 3, cv2.LINE_AA)
    cv2.putText(image, label, (int(x) + 10, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def _render_video(
    raw_video: Path,
    output: Path,
    arrays: dict[str, np.ndarray],
    fps: float,
    expected_width: int,
    expected_height: int,
) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(raw_video))
    if not capture.isOpened():
        raise WristDualProducerError(f"cannot open raw video: {raw_video}")
    actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (actual_width, actual_height) != (expected_width, expected_height):
        capture.release()
        raise WristDualProducerError(
            "raw review image domain drift: "
            f"{(actual_width, actual_height)} != {(expected_width, expected_height)}"
        )
    frame_count = len(arrays["frame_id"])
    temporary = output.with_name(f".{output.stem}.partial.mp4")
    writer = cv2.VideoWriter(
        str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1920, 1080)
    )
    if not writer.isOpened():
        capture.release()
        raise WristDualProducerError("cannot open dual-wrist review writer")
    keys = (
        ("static_wrist_uv", "STATIC_SELECTED"),
        ("observed_anatomical_wrist_uv", "ANATOMICAL_OBSERVED"),
        ("visible_wrist_surface_source_pixel_uv", "VISIBLE_SURFACE"),
        ("fused_wrist_uv", "FUSED"),
    )
    try:
        for frame in range(frame_count):
            ok, raw = capture.read()
            if not ok:
                raise WristDualProducerError(f"raw video ended at frame {frame}")
            rgb = cv2.resize(raw, (1280, 960), interpolation=cv2.INTER_AREA)
            source_height = float(raw.shape[0])
            source_width = float(raw.shape[1])
            scale = np.asarray((1280.0 / source_width, 960.0 / source_height))
            for array_name, label in keys:
                if array_name not in arrays:
                    continue
                for side in range(2):
                    _draw_point(
                        rgb,
                        arrays[array_name][frame, side] * scale,
                        COLORS[label],
                        f"{label[:3]}-{SIDES[side][0].upper()}",
                    )
            canvas = np.full((1080, 1920, 3), 24, dtype=np.uint8)
            canvas[90:1050, :1280] = rgb
            cv2.rectangle(canvas, (1280, 90), (1919, 1049), (45, 45, 45), -1)
            cv2.putText(canvas, "WRIST DUAL REPRESENTATION V1", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (245, 245, 245), 2, cv2.LINE_AA)
            cv2.putText(canvas, f"frame {frame:05d}/{frame_count - 1:05d}", (1500, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2, cv2.LINE_AA)
            y = 135
            for side in range(2):
                cv2.putText(canvas, SIDES[side].upper(), (1320, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                y += 38
                for field, label in (
                    ("static_T_camera_wrist", "STATIC_SELECTED"),
                    ("observed_T_camera_wrist", "ANATOMICAL_OBSERVED"),
                    ("fused_T_camera_wrist", "FUSED"),
                ):
                    point = arrays[field][frame, side, :3, 3] * 1000.0
                    value = "INVALID" if not np.isfinite(point).all() else f"{point[0]:+.1f} {point[1]:+.1f} {point[2]:+.1f} mm"
                    cv2.putText(canvas, f"{label}: {value}", (1320, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLORS[label], 1, cv2.LINE_AA)
                    y += 28
                surface = arrays["visible_wrist_surface_point_camera"][frame, side] * 1000.0
                surface_text = "REGION_ONLY/UNKNOWN" if not np.isfinite(surface).all() else f"{surface[0]:+.1f} {surface[1]:+.1f} {surface[2]:+.1f} mm"
                cv2.putText(canvas, f"VISIBLE_SURFACE: {surface_text}", (1320, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLORS["VISIBLE_SURFACE"], 1, cv2.LINE_AA)
                y += 55
            cv2.putText(canvas, "development / non-control / non-deployable", (1320, 1005), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (80, 180, 255), 1, cv2.LINE_AA)
            writer.write(canvas)
    finally:
        writer.release()
        capture.release()
    if not temporary.is_file() or temporary.stat().st_size <= 0:
        raise WristDualProducerError("dual-wrist review is empty")
    temporary.replace(output)
    verify = cv2.VideoCapture(str(output))
    decoded = int(verify.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(verify.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(verify.get(cv2.CAP_PROP_FRAME_HEIGHT))
    verify.release()
    if decoded != frame_count or (width, height) != (1920, 1080):
        raise WristDualProducerError("dual-wrist review decode contract failed")
    return _artifact(output)


def produce(
    *,
    input_npz: Path,
    config_path: Path,
    output_root: Path,
    raw_video: Path | None = None,
) -> dict[str, Any]:
    """Create one immutable numerical bundle and optional review video."""

    source_path = input_npz.resolve(strict=True)
    config_file = config_path.resolve(strict=True)
    output = output_root.resolve()
    if output.exists() or output.is_symlink():
        raise WristDualProducerError(f"fresh output root required: {output}")
    config = _load_config(config_file)
    output.mkdir(parents=True)
    try:
        bundle = _load_npz(source_path)
        frame_id = _require(bundle, "frame_id", (None,))
        frame_count = len(frame_id)
        timestamp = _require(bundle, "timestamp_s", (frame_count,)).astype(np.float64)
        recording = _require(bundle, "recording_id", (frame_count,))
        controller = _require(bundle, "T_camera_controller_raw", (frame_count, 2, 4, 4)).astype(np.float64)
        nominal = _require(bundle, "T_controller_wrist_M0", (2, 4, 4)).astype(np.float64)
        observed = _require(bundle, "observed_T_camera_wrist", (frame_count, 2, 4, 4)).astype(np.float64)
        observed_valid = _require(bundle, "observed_valid", (frame_count, 2)).astype(bool)
        orientation_valid = _require(bundle, "orientation_valid", (frame_count, 2)).astype(bool)
        bundle.update({
            "frame_id": frame_id,
            "timestamp_s": timestamp,
            "recording_id": recording,
            "T_camera_controller_raw": controller,
            "T_controller_wrist_M0": nominal,
            "observed_T_camera_wrist": observed,
            "observed_valid": observed_valid,
            "orientation_valid": orientation_valid,
        })
        models, calibration_audit = _fit_models(bundle, config)
        selected_model = {side: str(config["selected_model"][side]) for side in SIDES}
        selected = np.stack([models[side][selected_model[side]] for side in SIDES])
        static = compose_camera_wrist(controller, selected[None])
        measured_center = observed[:, :, :3, 3].copy()
        fusion = bounded_nonaccumulating_fusion(
            static,
            measured_center,
            observed_valid,
            heuristic_weight=float(config["fusion_heuristic_weight"]),
            correction_bound_mm=float(config.get("development_numeric_correction_bound_mm", 30.0)),
        )
        surface = _surface_arrays(bundle, frame_count)
        temporal = {
            str(name): validate_temporal_authority(str(value))
            for name, value in config["temporal_authority"].items()
        }
        arrays: dict[str, np.ndarray] = {
            "frame_id": frame_id,
            "timestamp_s": timestamp,
            "recording_id": recording,
            "T_camera_controller_raw": controller,
            "T_controller_wrist_M0": nominal,
            "T_controller_wrist_M1": np.stack([
                models[side].get("M1_CONTROLLER_LOCAL_TRANSLATION", np.full((4, 4), np.nan))
                for side in SIDES
            ]),
            "T_controller_wrist_M2": np.stack([
                models[side].get("M2_STATIC_SE3", np.full((4, 4), np.nan))
                for side in SIDES
            ]),
            "selected_T_controller_wrist": selected,
            "static_T_camera_wrist": static,
            "observed_T_camera_wrist": observed,
            "observed_valid": observed_valid,
            **surface,
            **fusion,
        }
        for optional in (
            "static_wrist_uv",
            "observed_anatomical_wrist_uv",
            "fused_wrist_uv",
        ):
            if optional in bundle:
                arrays[optional] = _require(bundle, optional, (frame_count, 2, 2)).astype(np.float64)
        metrics: dict[str, Any] = {
            "schema_version": "chaoyang-wrist-dual-metrics-v1",
            "calibration": calibration_audit,
            "selected_model": selected_model,
            "surface": {
                side: {
                    "valid": int(surface["visible_wrist_surface_valid"][:, index].sum()),
                    "region_only": int(surface["visible_wrist_region_registration_only"][:, index].sum()),
                    "unknown": int((~surface["visible_wrist_surface_valid"][:, index] & ~surface["visible_wrist_region_registration_only"][:, index]).sum()),
                }
                for index, side in enumerate(SIDES)
            },
            "motion": {
                side: {
                    "static": _stats(static[:, index, :3, 3], np.ones(frame_count, dtype=bool)),
                    "observed": _stats(observed[:, index, :3, 3], observed_valid[:, index]),
                    "fused": _stats(fusion["fused_T_camera_wrist"][:, index, :3, 3], fusion["fusion_valid"][:, index]),
                    "correction_touch_bound": int(fusion["correction_clipped"][:, index].sum()),
                }
                for index, side in enumerate(SIDES)
            },
            "temporal_authority": temporal,
            "claim_limit": "Internal development comparison only; no external anatomical, metric, control, or deployment truth.",
        }
        npz_path = output / "WRIST_DUAL_REPRESENTATION_V1.npz"
        np.savez_compressed(npz_path, **arrays)
        metrics_path = output / "METRICS.json"
        _atomic_json(metrics_path, metrics)
        video_ref = None
        if raw_video is not None:
            video_path = output / "WRIST_DUAL_REPRESENTATION_REVIEW.mp4"
            video_ref = _render_video(
                raw_video.resolve(strict=True),
                video_path,
                arrays,
                float(config["fps"]),
                int(config["review_image_domain"]["width"]),
                int(config["review_image_domain"]["height"]),
            )
        result: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": "DEVELOPMENT_RESULT_GENERATED",
            "transform_convention": {
                "notation": "T_A_B_MAPS_B_TO_A",
                "composition": "T_camera_wrist=T_camera_controller@T_controller_wrist",
                "saved_transform_shape": "[T,2,4,4]",
            },
            "representations": {
                "anatomical_wrist_center": "MODEL_DEFINED_KINEMATIC_WRIST_NOT_EXTERNAL_ANATOMICAL_TRUTH",
                "visible_wrist_surface": "VIEW_DEPENDENT_OBSERVATION_NEVER_JOINT_SUBSTITUTE",
            },
            "calibration": {
                "models": list(MODELS),
                "selected_model": selected_model,
                "fit_roles": sorted(str(value) for value in config["fit_recording_ids"]),
                "holdout_consumed": False,
            },
            "fusion": {
                "nonaccumulating": True,
                "static_prior_preserved": True,
                "development_numeric_correction_bound_mm": float(config.get("development_numeric_correction_bound_mm", 30.0)),
                "uncertainty_status": "UNKNOWN",
            },
            "temporal_authority": temporal,
            "inputs": {
                "bundle": _artifact(source_path),
                "config": _artifact(config_file),
                "raw_video": _artifact(raw_video.resolve(strict=True)) if raw_video is not None else None,
            },
            "outputs": {
                "npz": _artifact(npz_path),
                "metrics": _artifact(metrics_path),
                "video": video_ref,
            },
            "authority": {
                "control_ground_truth": False,
                "external_wrist_truth": False,
                "external_metric_authority": False,
                "physical_deployment_authorized": False,
            },
        }
        result_path = output / "RESULT.json"
        _atomic_json(result_path, result)
        return result
    except BaseException:
        # The output attempt is intentionally retained for audit; no source is touched.
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--raw-video", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        print(
            json.dumps(
                preflight(input_npz=args.input_npz, config_path=args.config),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.output_root is None:
        parser.error("--output-root is required unless --preflight-only is set")
    result = produce(
        input_npz=args.input_npz,
        config_path=args.config,
        output_root=args.output_root,
        raw_video=args.raw_video,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
