#!/usr/bin/env python3
"""Build the CPU-only AI1 static-wrist candidate and diagnostic bundle V3.2.

The operation is fixed to 097/098 development and 101 regression.  It never
opens or reads 102/103.  M1 is reproduced as a diagnostic candidate but is not
adopted merely because it is selected or has a lower internal residual.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import jsonschema
import numpy as np

from chaoyang.ops import build_wiyh_wrist_dual_input_v1 as input_builder
from chaoyang.research.world_in_your_hands.ai1_static_wrist_candidate_v32 import (
    DIAGNOSTIC_ORDER,
    SCHEMA_VERSION,
    absolute_depth_diagnostic,
    decide_candidate_authority,
    finite_lag_diagnostics,
    summarize_norm_mm,
    two_dimensional_diagnostic,
    validate_diagnostic_order,
)
from chaoyang.research.world_in_your_hands.wrist_dual_representation_v1 import (
    compose_camera_wrist,
    fit_m1_translation,
    lever_arm_residuals,
)


SIDES = ("left", "right")
FIT_IDS = ("play_cards_0916_097", "play_cards_0916_098")
REGRESSION_ID = "play_cards_0916_101"
ADOPTION_IDS = ("play_cards_0916_102", "play_cards_0916_103")
MANUS25_NAMES = (
    "Hand_Invalid",
    "Thumb_MCP", "Thumb_PIP", "Thumb_DIP", "Thumb_TIP",
    "Index_MCP", "Index_PIP", "Index_IP", "Index_DIP", "Index_TIP",
    "Middle_MCP", "Middle_PIP", "Middle_IP", "Middle_DIP", "Middle_TIP",
    "Ring_MCP", "Ring_PIP", "Ring_IP", "Ring_DIP", "Ring_TIP",
    "Pinky_MCP", "Pinky_PIP", "Pinky_IP", "Pinky_DIP", "Pinky_TIP",
)
# MANO21 wrist is index zero.  MANUS has no anatomical wrist node, so only the
# twenty named finger joints are compared after each representation removes
# its own declared root.  No axis permutation or fitted similarity is allowed.
MANO21_FROM_MANUS25 = (-1, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14, 15, 16, 18, 19, 20, 21, 23, 24)


class Ai1V32RunnerError(RuntimeError):
    """Raised when fixed source evidence cannot satisfy the V3.2 interface."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact(path: Path, *, published_path: Path | None = None) -> dict[str, Any]:
    value = path.resolve(strict=True)
    return {
        "path": str((published_path or value).absolute()),
        "bytes": value.stat().st_size,
        "sha256": _sha256(value),
    }


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Ai1V32RunnerError(f"expected JSON object: {path}")
    return value


def _matrix(value: Any, *, label: str, allow_missing: bool = False) -> np.ndarray:
    if value is None and allow_missing:
        return np.full((4, 4), np.nan, dtype=np.float64)
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise Ai1V32RunnerError(f"{label} must be a finite 4x4 transform")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8):
        raise Ai1V32RunnerError(f"{label} has an invalid homogeneous row")
    return matrix


def _native_observations(
    *, processed_root: Path, experiment_root: Path
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    arrays: dict[str, list[np.ndarray]] = {
        "camera_K": [],
        "manus25_xyz_wrist_local_m": [],
        "manus25_xyz_camera_m": [],
        "manus25_valid": [],
        "pico_controller_T_world": [],
        "pico_controller_world_valid": [],
        "hawor_mano21_xyz_camera_m": [],
        "hawor_mano21_uv": [],
        "hawor_observed": [],
        "image_domain": [],
        "mirror_policy": [],
        "crop_policy": [],
        "camera_eye": [],
        "camera_physical_eye": [],
        "camera_source_index": [],
        "tracking_sync_error_ms": [],
    }
    session_rows: list[dict[str, Any]] = []
    processed = processed_root.resolve(strict=True)
    experiments = experiment_root.resolve(strict=True)
    for session_id in (*FIT_IDS, REGRESSION_ID):
        training_root = processed / session_id / "preprocess" / "all_data"
        records = sorted(training_root.glob("*/training_data.json"))
        if not records:
            raise Ai1V32RunnerError(f"no processed records: {session_id}")
        spec = input_builder.SESSION_SPECS[session_id]
        hawor_path = (
            experiments / spec["experiment"] / "attempts" / "attempt_0001" / spec["hawor"]
        )
        hawor = input_builder._load_hawor(hawor_path, len(records))
        arrays["hawor_mano21_xyz_camera_m"].append(
            np.asarray(hawor["joints_3d_camera"], dtype=np.float64).transpose(1, 0, 2, 3)
        )
        arrays["hawor_mano21_uv"].append(
            np.asarray(hawor["joints_2d"], dtype=np.float64).transpose(1, 0, 2, 3)
        )
        arrays["hawor_observed"].append(np.asarray(hawor["observed"], dtype=bool).T)
        per_session: dict[str, list[Any]] = {name: [] for name in arrays if not name.startswith("hawor_")}
        for frame, path in enumerate(records):
            row = _load_json(path)
            metadata = row.get("metadata", {})
            intrinsics = np.asarray(metadata.get("k"), dtype=np.float64)
            if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
                raise Ai1V32RunnerError(f"missing/invalid camera K at {path}")
            per_session["camera_K"].append(intrinsics)
            per_session["image_domain"].append(
                str(metadata.get("image_domain_mode", metadata.get("camera_model", "UNKNOWN")))
            )
            per_session["mirror_policy"].append(
                str(metadata.get("mirror_policy", "UNDECLARED_NO_INFERENCE"))
            )
            per_session["crop_policy"].append(
                str(metadata.get("crop_policy", "UNDECLARED_NO_INFERENCE"))
            )
            per_session["camera_eye"].append(str(metadata.get("camera_eye", "UNKNOWN")))
            per_session["camera_physical_eye"].append(
                str(metadata.get("camera_physical_calibration_eye", "UNKNOWN"))
            )
            source_index = metadata.get("camera_source_index", -1)
            if not isinstance(source_index, int):
                raise Ai1V32RunnerError(f"camera source index is not integral at {path}")
            per_session["camera_source_index"].append(source_index)
            sync_error = metadata.get("tracking_sync_error_ms", 0.0)
            if not isinstance(sync_error, (int, float)) or not np.isfinite(float(sync_error)):
                raise Ai1V32RunnerError(f"tracking sync error is invalid at {path}")
            per_session["tracking_sync_error_ms"].append(float(sync_error))
            local_sides: list[np.ndarray] = []
            camera_sides: list[np.ndarray] = []
            valid_sides: list[np.ndarray] = []
            world_controller: list[np.ndarray] = []
            world_valid: list[bool] = []
            hands = row.get("entities", {}).get("hands", {})
            for side in SIDES:
                hand = hands.get(side, {})
                manus = hand.get("manus25", {})
                names = tuple(str(value) for value in manus.get("joint_names", []))
                if names != MANUS25_NAMES:
                    raise Ai1V32RunnerError(f"MANUS25 joint contract mismatch at {path}:{side}")
                local = np.asarray(manus.get("keypoints_3d_wrist_local"), dtype=np.float64)
                camera = np.asarray(manus.get("keypoints_3d_camera"), dtype=np.float64)
                valid = np.asarray(manus.get("joint_valid"), dtype=bool)
                if local.shape != (25, 3) or camera.shape != (25, 3) or valid.shape != (25,):
                    raise Ai1V32RunnerError(f"MANUS25 shape mismatch at {path}:{side}")
                if not np.isfinite(local[valid]).all() or not np.isfinite(camera[valid]).all():
                    raise Ai1V32RunnerError(f"non-finite valid MANUS25 point at {path}:{side}")
                local_sides.append(local)
                camera_sides.append(camera)
                valid_sides.append(valid)
                controller = hand.get("controller6d", {})
                if controller.get("units", "metres") not in {"metres", "meters", "m"}:
                    raise Ai1V32RunnerError(f"controller unit is not metres at {path}:{side}")
                world_value = controller.get("T_controller_to_world")
                world_controller.append(
                    _matrix(world_value, label=f"{path}:{side}:T_world_controller", allow_missing=True)
                )
                world_valid.append(world_value is not None)
            per_session["manus25_xyz_wrist_local_m"].append(np.stack(local_sides))
            per_session["manus25_xyz_camera_m"].append(np.stack(camera_sides))
            per_session["manus25_valid"].append(np.stack(valid_sides))
            per_session["pico_controller_T_world"].append(np.stack(world_controller))
            per_session["pico_controller_world_valid"].append(np.asarray(world_valid, dtype=bool))
        for name, values in per_session.items():
            arrays[name].append(np.asarray(values))
        session_rows.append(
            {
                "session_id": session_id,
                "role": "DEVELOPMENT" if session_id in FIT_IDS else "REGRESSION_FORBIDDEN_FIT",
                "frame_count": len(records),
                "manus25_present": True,
                "pico_controller_present": True,
                "hawor_source": _artifact(hawor_path),
            }
        )
    return {name: np.concatenate(values, axis=0) for name, values in arrays.items()}, session_rows


def _relative_shape(
    manus_camera: np.ndarray,
    hawor_camera: np.ndarray,
    candidate_wrist: np.ndarray,
    observed_valid: np.ndarray,
    manus_valid: np.ndarray,
) -> dict[str, Any]:
    source_indices = np.asarray(MANO21_FROM_MANUS25[1:], dtype=np.int64)
    manus_points = manus_camera[:, source_indices]
    manus_relative = manus_points - candidate_wrist[:, None]
    hawor_relative = hawor_camera[:, 1:] - hawor_camera[:, :1]
    valid = observed_valid[:, None] & manus_valid[:, source_indices]
    valid &= np.isfinite(manus_relative).all(axis=2) & np.isfinite(hawor_relative).all(axis=2)
    residual = np.linalg.norm(manus_relative[valid] - hawor_relative[valid], axis=1) * 1000.0
    return {
        "valid_joint_frames": int(len(residual)),
        "residual_mm_p50": float(np.percentile(residual, 50)) if len(residual) else None,
        "residual_mm_p95": float(np.percentile(residual, 95)) if len(residual) else None,
        "alignment_fitted": False,
        "axis_permutation_searched": False,
        "selection_authority": False,
    }


def _side_diagnostics(
    *,
    side_index: int,
    controller: np.ndarray,
    m0: np.ndarray,
    m1: np.ndarray,
    observed: np.ndarray,
    observed_valid: np.ndarray,
    observed_uv: np.ndarray,
    camera_K: np.ndarray,
    manus_camera: np.ndarray,
    manus_valid: np.ndarray,
    hawor_camera: np.ndarray,
    recording_id: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    m0_camera = compose_camera_wrist(controller, m0)
    m1_camera = compose_camera_wrist(controller, m1)
    proxy = observed[:, :3, 3]
    m0_residual = lever_arm_residuals(controller, m0, proxy)
    m1_residual = lever_arm_residuals(controller, m1, proxy)
    rows = {
        "finite_lag": {
            "prescribed_lags_only": True,
            "winner_selected": False,
            "M0": finite_lag_diagnostics(
                m0_camera[:, :3, 3], proxy, observed_valid, group_id=recording_id
            ),
            "M1": finite_lag_diagnostics(
                m1_camera[:, :3, 3], proxy, observed_valid, group_id=recording_id
            ),
            "comparison_authority": "DIAGNOSTIC_PROXY_ONLY_WRIST_SEMANTICS_UNPROVEN",
        },
        "controller_local": {
            "M0_camera": summarize_norm_mm(m0_residual["residual_camera"], observed_valid),
            "M0_controller": summarize_norm_mm(
                m0_residual["residual_controller"], observed_valid
            ),
            "M1_camera": summarize_norm_mm(m1_residual["residual_camera"], observed_valid),
            "M1_controller": summarize_norm_mm(
                m1_residual["residual_controller"], observed_valid
            ),
            "semantic_status": "DIAGNOSTIC_PROXY_ONLY_NOT_ADOPTION_EVIDENCE",
        },
        "hawor_decomposition": {
            "two_dimensional": {
                "M0": two_dimensional_diagnostic(
                    m0_camera[:, :3, 3], observed_uv, camera_K, observed_valid
                ),
                "M1": two_dimensional_diagnostic(
                    m1_camera[:, :3, 3], observed_uv, camera_K, observed_valid
                ),
            },
            "relative_hand_shape": {
                "M0": _relative_shape(
                    manus_camera,
                    hawor_camera,
                    m0_camera[:, :3, 3],
                    observed_valid,
                    manus_valid,
                ),
                "M1": _relative_shape(
                    manus_camera,
                    hawor_camera,
                    m1_camera[:, :3, 3],
                    observed_valid,
                    manus_valid,
                ),
            },
            "absolute_depth": {
                "M0": absolute_depth_diagnostic(
                    m0_camera[:, 2, 3],
                    proxy[:, 2],
                    observed_valid,
                    candidate_wrist_semantics="CONTROLLER_DERIVED_STATIC_WRIST_POSITION_ONLY",
                    observed_wrist_semantics="HAWOR_MODEL_DEFINED_ANATOMICAL_WRIST",
                ),
                "M1": absolute_depth_diagnostic(
                    m1_camera[:, 2, 3],
                    proxy[:, 2],
                    observed_valid,
                    candidate_wrist_semantics="CONTROLLER_DERIVED_STATIC_WRIST_POSITION_ONLY",
                    observed_wrist_semantics="HAWOR_MODEL_DEFINED_ANATOMICAL_WRIST",
                ),
            },
        },
    }
    replay = {
        "M0_T_camera_wrist": m0_camera,
        "M1_T_camera_wrist": m1_camera,
        "M0_residual_camera_m": m0_residual["residual_camera"],
        "M0_residual_controller_m": m0_residual["residual_controller"],
        "M1_residual_camera_m": m1_residual["residual_camera"],
        "M1_residual_controller_m": m1_residual["residual_controller"],
    }
    return rows, replay


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())


def run(*, processed_root: Path, experiment_root: Path, output_root: Path) -> dict[str, Any]:
    """Build one immutable V3.2 candidate attempt without reading adoption sessions."""

    output = output_root.absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise Ai1V32RunnerError(f"fresh output root required: {output}")
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging.", dir=output.parent))
    try:
        (
            base,
            input_sessions,
            m0_provenance,
            legacy_translation_error_m,
            legacy_rotation_error_deg,
        ) = input_builder._assemble(
            processed_root=processed_root,
            experiment_root=experiment_root,
        )
        native, session_rows = _native_observations(
            processed_root=processed_root,
            experiment_root=experiment_root,
        )
        frame_count = len(base["frame_id"])
        if any(len(value) != frame_count for value in native.values()):
            raise Ai1V32RunnerError("native observation alignment drift")
        recording = base["recording_id"].astype(str)
        fit_frame = np.isin(recording, FIT_IDS)
        controller = np.asarray(base["T_camera_controller_raw"], dtype=np.float64)
        observed = np.asarray(base["observed_T_camera_wrist"], dtype=np.float64)
        observed_valid = np.asarray(base["observed_valid"], dtype=bool)
        m0 = np.asarray(base["T_controller_wrist_M0"], dtype=np.float64)
        m1 = np.empty_like(m0)
        m1_fit: dict[str, Any] = {}
        for side_index, side in enumerate(SIDES):
            m1[side_index], m1_fit[side] = fit_m1_translation(
                controller[:, side_index],
                observed[:, side_index, :3, 3],
                m0[side_index],
                fit_frame & observed_valid[:, side_index],
            )
        side_rows: dict[str, Any] = {}
        replay: dict[str, np.ndarray] = {}
        for side_index, side in enumerate(SIDES):
            side_rows[side], side_replay = _side_diagnostics(
                side_index=side_index,
                controller=controller[:, side_index],
                m0=m0[side_index],
                m1=m1[side_index],
                observed=observed[:, side_index],
                observed_valid=observed_valid[:, side_index],
                observed_uv=np.asarray(base["observed_anatomical_wrist_uv"][:, side_index]),
                camera_K=np.asarray(native["camera_K"]),
                manus_camera=np.asarray(native["manus25_xyz_camera_m"][:, side_index]),
                manus_valid=np.asarray(native["manus25_valid"][:, side_index]),
                hawor_camera=np.asarray(native["hawor_mano21_xyz_camera_m"][:, side_index]),
                recording_id=recording,
            )
            replay.update({f"{name}_{side}": value for name, value in side_replay.items()})
        image_tuples = sorted(
            {
                (
                    str(native["image_domain"][index]),
                    str(native["camera_eye"][index]),
                    str(native["camera_physical_eye"][index]),
                    int(native["camera_source_index"][index]),
                )
                for index in range(frame_count)
            }
        )
        mirror_policies = sorted({str(value) for value in native["mirror_policy"]})
        crop_policies = sorted({str(value) for value in native["crop_policy"]})
        intrinsics = np.asarray(native["camera_K"], dtype=np.float64)
        diagnostics = [
            {
                "stage": DIAGNOSTIC_ORDER[0],
                "status": "RECORDED_NOT_NORMALIZED",
                "observed_domain_eye_source_index": [list(value) for value in image_tuples],
                "mirror_policy": mirror_policies,
                "crop_policy": crop_policies,
                "K": {
                    "frame_count": int(len(intrinsics)),
                    "finite": bool(np.isfinite(intrinsics).all()),
                    "fx_range": [float(intrinsics[:, 0, 0].min()), float(intrinsics[:, 0, 0].max())],
                    "fy_range": [float(intrinsics[:, 1, 1].min()), float(intrinsics[:, 1, 1].max())],
                    "cx_range": [float(intrinsics[:, 0, 2].min()), float(intrinsics[:, 0, 2].max())],
                    "cy_range": [float(intrinsics[:, 1, 2].min()), float(intrinsics[:, 1, 2].max())],
                },
                "coordinate_combination_search": False,
            },
            {
                "stage": DIAGNOSTIC_ORDER[1],
                "status": "PASS_EXPLICIT_T_A_B_METRES",
                "notation": "T_A_B_MAPS_B_TO_A",
                "composition": "T_camera_wrist=T_camera_controller@T_controller_wrist",
                "translation_unit": "metres",
                "axis_combination_search": False,
            },
            {
                "stage": DIAGNOSTIC_ORDER[2],
                "status": "PASS_RAW_MANUS25_PRESERVED",
                "joint_count": 25,
                "joint_names": list(MANUS25_NAMES),
                "node0_semantics": "Hand_Invalid_NOT_ANATOMICAL_WRIST",
                "source_axes": "WRIST_LOCAL_SOURCE_AXES_UNCHANGED",
                "duplicate_root_fabricated": False,
            },
            {
                "stage": DIAGNOSTIC_ORDER[3],
                "status": "FINITE_LAGS_REPORTED_NO_WINNER_SELECTION",
                "prescribed_lags": [-2, -1, 0, 1, 2],
                "tracking_sync_error_ms_max_abs": float(
                    np.max(np.abs(native["tracking_sync_error_ms"]))
                ),
                "sides": {side: side_rows[side]["finite_lag"] for side in SIDES},
            },
            {
                "stage": DIAGNOSTIC_ORDER[4],
                "status": "REPORTED_DIAGNOSTIC_PROXY_ONLY",
                "sides": {side: side_rows[side]["controller_local"] for side in SIDES},
            },
            {
                "stage": DIAGNOSTIC_ORDER[5],
                "status": "DECOMPOSED_NO_CROSS_SEMANTIC_DEPTH_ERROR",
                "sides": {side: side_rows[side]["hawor_decomposition"] for side in SIDES},
            },
        ]
        validate_diagnostic_order(diagnostics)
        decision = decide_candidate_authority(
            selected_model="M1_CONTROLLER_LOCAL_TRANSLATION",
            deterministic_error_fixed=False,
            leave_one_recording_out_passed=False,
            regression_101_passed=False,
            wrist_semantics_proven_equivalent=False,
            independent_rotation_evidence=False,
        )
        replay_arrays: dict[str, np.ndarray] = {
            "schema_version": np.asarray(SCHEMA_VERSION),
            "frame_id": np.asarray(base["frame_id"]),
            "timestamp_s": np.asarray(base["timestamp_s"]),
            "recording_id": np.asarray(base["recording_id"]),
            "anatomical_side_names": np.asarray(SIDES),
            "manus25_joint_names": np.asarray(MANUS25_NAMES),
            "T_camera_controller_raw": controller,
            "pico_controller_T_world": np.asarray(native["pico_controller_T_world"]),
            "pico_controller_world_valid": np.asarray(native["pico_controller_world_valid"]),
            "manus25_xyz_wrist_local_m": np.asarray(native["manus25_xyz_wrist_local_m"]),
            "manus25_xyz_camera_m": np.asarray(native["manus25_xyz_camera_m"]),
            "manus25_valid": np.asarray(native["manus25_valid"]),
            "hawor_mano21_xyz_camera_m": np.asarray(native["hawor_mano21_xyz_camera_m"]),
            "hawor_mano21_uv": np.asarray(native["hawor_mano21_uv"]),
            "hawor_observed": np.asarray(native["hawor_observed"]),
            "observed_T_camera_wrist": observed,
            "observed_valid": observed_valid,
            "T_controller_wrist_M0": m0,
            "T_controller_wrist_M1": m1,
            "M0_T_camera_wrist": compose_camera_wrist(controller, m0[None]),
            "M1_T_camera_wrist": compose_camera_wrist(controller, m1[None]),
            "selected_model": np.asarray(decision["selected_model"]),
            "adopted": np.asarray(decision["adopted"]),
            "adopted_model": np.asarray(""),
            **replay,
        }
        replay_path = staging / "AI1_STATIC_WRIST_CANDIDATE_V32.npz"
        _write_npz(replay_path, replay_arrays)
        diagnostics_path = staging / "DIAGNOSTICS.json"
        _write_json(
            diagnostics_path,
            {
                "schema_version": "ai1-static-wrist-diagnostics-v32",
                "fixed_order": list(DIAGNOSTIC_ORDER),
                "diagnostics": diagnostics,
                "m1_reproduction": m1_fit,
                "forbidden_search": {
                    "coordinate_combinations": True,
                    "axis_combinations": True,
                    "unbounded_time_shift": True,
                    "visual_best_candidate_selection": True,
                },
            },
        )
        source_provenance_path = staging / "SOURCE_PROVENANCE.json"
        _write_json(
            source_provenance_path,
            {
                "schema_version": "ai1-static-wrist-source-provenance-v32",
                "fixed_sessions": list((*FIT_IDS, REGRESSION_ID)),
                "adoption_sessions_read": False,
                "v31_input_sessions": input_sessions,
                "m0_legacy_prior": m0_provenance,
                "legacy_relation_audit": {
                    "maximum_translation_error_mm": legacy_translation_error_m * 1000.0,
                    "maximum_rotation_error_deg": legacy_rotation_error_deg,
                },
                "source_mutated": False,
            },
        )
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": decision["terminal_status"],
            "lane": "ai1",
            "sessions": session_rows
            + [
                {
                    "session_id": session_id,
                    "role": "ADOPTION",
                    "status": "NOT_OPENED_GATE_CLOSED",
                }
                for session_id in ADOPTION_IDS
            ],
            "models": {
                "M0_LEGACY": {
                    "available": True,
                    "selected": False,
                    "adopted": False,
                    "role": "DEFAULT_CONSUMPTION_LEGACY_PRIOR",
                },
                "M1_CONTROLLER_LOCAL_TRANSLATION": {
                    "available": True,
                    "selected": True,
                    "adopted": decision["adopted"],
                    "role": "DIAGNOSTIC_CANDIDATE_NOT_DEFAULT_CONSUMPTION",
                },
                "M2_STATIC_SE3": {
                    "available": False,
                    "selected": False,
                    "adopted": False,
                    "status": "BLOCKED_INDEPENDENT_ROTATION_EVIDENCE",
                },
            },
            "candidate_decision": decision,
            "opening_gate_102_103": decision["opening_gate_102_103"],
            "default_consumption_model": "M0_LEGACY",
            "outputs": {
                "native_complete_replay_npz": _artifact(
                    replay_path, published_path=output / replay_path.name
                ),
                "diagnostics": _artifact(
                    diagnostics_path, published_path=output / diagnostics_path.name
                ),
                "source_provenance": _artifact(
                    source_provenance_path,
                    published_path=output / source_provenance_path.name,
                ),
            },
            "claims": {
                "PIPELINE_COMPLETE": True,
                "NUMERIC_QUALITY_PASS": False,
                "VISUAL_REVIEW_STATUS": "NOT_REVIEWED_NUMERIC_REPLAY_ONLY",
                "TRAINING_COMPLETE": False,
                "TRAINING_ELIGIBLE": False,
                "CONTROL_GROUND_TRUTH": False,
                "PHYSICAL_DEPLOYABLE": False,
                "external_metric_authority": False,
            },
            "execution": {
                "cpu_only": True,
                "gpu_used": False,
                "source_mutated": False,
                "adoption_sessions_read": False,
            },
        }
        schema = json.loads(
            (Path(__file__).resolve().parents[3] / "contracts/ai1_static_wrist_candidate_v32.schema.json").read_text(
                encoding="utf-8"
            )
        )
        jsonschema.validate(result, schema)
        _write_json(staging / "RESULT.json", result)
        input_builder._publish_directory_no_clobber(staging, output)
        return result
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                processed_root=args.processed_root,
                experiment_root=args.experiment_root,
                output_root=args.output_root,
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
