#!/usr/bin/env python3
"""Build the fixed, CPU-only WIYH wrist-dual input bundle V1.

The adapter is deliberately narrow.  It reads only the current 097/098/101
processed sessions and their pinned HaWoR diagnostic artifacts.  Sessions 102
and 103 have no pinned anatomical-wrist observation and therefore cannot be
silently filled or admitted by this operation.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import jsonschema
import numpy as np


SCHEMA_VERSION = "chaoyang-wiyh-wrist-dual-input-provenance-v1"
SIDES = ("left", "right")
ALLOWED_SESSION_IDS = (
    "play_cards_0916_097",
    "play_cards_0916_098",
    "play_cards_0916_101",
)
EXCLUDED_MISSING_OBSERVATION_IDS = (
    "play_cards_0916_102",
    "play_cards_0916_103",
)
SESSION_SPECS: dict[str, dict[str, str]] = {
    "play_cards_0916_097": {
        "role": "DEVELOPMENT",
        "experiment": "world_in_your_hands_hawor_scale_adaptive_blind_v44",
        "hawor": "hawor/HAWOR_CROSS_SESSION_BLIND_LANDMARKS.npz",
    },
    "play_cards_0916_098": {
        "role": "DEVELOPMENT",
        "experiment": "world_in_your_hands_conservative_blind_eval_v49",
        "hawor": "HAWOR_CROSS_SESSION_BLIND_LANDMARKS.npz",
    },
    "play_cards_0916_101": {
        "role": "REGRESSION_FORBIDDEN_FIT",
        "experiment": "world_in_your_hands_hawor_combined_observability_blind_v55",
        "hawor": "hawor/HAWOR_CROSS_SESSION_BLIND_LANDMARKS.npz",
    },
}
M0_EXPERIMENT = "world_in_your_hands_exposed_refit_v45"
M0_REFIT = "MANUS_ORIGIN_REFIT_V45.json"
EXPECTED_HAWOR_TERMINAL = "FAIL_CROSS_SESSION_BLIND_MANUS_ORIGIN_ALIGNMENT"
EXPECTED_M0_TERMINAL = "PASS_SESSION_097_EXPOSED_REFIT_PENDING_NEW_BLIND"


class WiyhWristDualInputError(RuntimeError):
    """Raised when a fixed source is absent, ambiguous, or semantically unsafe."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path, *, published_path: Path | None = None) -> dict[str, Any]:
    source = path.resolve(strict=True)
    return {
        "path": str((published_path or source).absolute()),
        "bytes": source.stat().st_size,
        "sha256": _sha256(source),
    }


def _file_set(root: Path, files: list[Path]) -> dict[str, Any]:
    if not files:
        raise WiyhWristDualInputError(f"empty training-data set: {root}")
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha256(path)))
    return {
        "root": str(root.resolve(strict=True)),
        "file_count": len(files),
        "aggregate_sha256": digest.hexdigest(),
    }


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise WiyhWristDualInputError(f"expected JSON object: {path}")
    return value


def _matrix(value: Any, *, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise WiyhWristDualInputError(f"{label} is not a finite 4x4 transform")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8, rtol=0.0):
        raise WiyhWristDualInputError(f"{label} has an invalid homogeneous row")
    return matrix


def _rotation_error_deg(left: np.ndarray, right: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(left.T @ right) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _terminal(path: Path, expected: str) -> tuple[dict[str, Any], dict[str, Any]]:
    value = _load_json(path)
    if value.get("status") != expected:
        raise WiyhWristDualInputError(
            f"terminal status drift at {path}: {value.get('status')!r} != {expected!r}"
        )
    return value, _artifact(path)


def _load_m0(experiment_root: Path) -> tuple[np.ndarray, dict[str, Any]]:
    attempt = experiment_root / M0_EXPERIMENT / "attempts" / "attempt_0001"
    refit_path = attempt / M0_REFIT
    terminal_path = attempt / "TERMINAL_RESULT.json"
    refit = _load_json(refit_path)
    terminal, terminal_ref = _terminal(terminal_path, EXPECTED_M0_TERMINAL)
    if bool(refit.get("authority_promoted")) or bool(terminal.get("authority_promoted")):
        raise WiyhWristDualInputError("V45 M0 source unexpectedly claims promoted authority")
    nominal = np.stack(
        [
            _matrix(
                refit.get("sides", {}).get(side, {}).get("nominal_controller_to_wrist"),
                label=f"V45 {side} nominal_controller_to_wrist",
            )
            for side in SIDES
        ]
    )
    return nominal, {
        "model": "M0_LEGACY",
        "refit_source": _artifact(refit_path),
        "terminal_source": terminal_ref,
        "terminal_status": EXPECTED_M0_TERMINAL,
        "authority_promoted": False,
    }


def _load_hawor(path: Path, frame_count: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "joints_3d_camera",
            "joints_2d",
            "observed",
            "root_orient_camera",
            "mano_joint_names",
            "anatomical_side_names",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise WiyhWristDualInputError(f"HaWoR artifact missing fields {missing}: {path}")
        value = {name: np.asarray(archive[name]) for name in required}
    if value["joints_3d_camera"].shape != (2, frame_count, 21, 3):
        raise WiyhWristDualInputError(f"HaWoR 3D shape drift: {path}")
    if value["joints_2d"].shape != (2, frame_count, 21, 2):
        raise WiyhWristDualInputError(f"HaWoR 2D shape drift: {path}")
    if value["observed"].shape != (2, frame_count):
        raise WiyhWristDualInputError(f"HaWoR observed shape drift: {path}")
    if value["root_orient_camera"].shape != (2, frame_count, 3, 3):
        raise WiyhWristDualInputError(f"HaWoR root orientation shape drift: {path}")
    if tuple(value["anatomical_side_names"].astype(str).tolist()) != SIDES:
        raise WiyhWristDualInputError(f"HaWoR side order drift: {path}")
    names = tuple(value["mano_joint_names"].astype(str).tolist())
    if len(names) != 21 or names[0].lower() != "wrist":
        raise WiyhWristDualInputError(f"HaWoR wrist index is not explicit: {path}")
    return value


def _read_session(
    *,
    processed_root: Path,
    experiment_root: Path,
    session_id: str,
    nominal_m0: np.ndarray,
    output_start: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any], float, float]:
    spec = SESSION_SPECS[session_id]
    session_root = (processed_root / session_id).resolve(strict=True)
    if session_root.parent != processed_root.resolve(strict=True):
        raise WiyhWristDualInputError(f"session escaped fixed processed root: {session_id}")
    training_root = session_root / "preprocess" / "all_data"
    training_files = sorted(training_root.glob("*/training_data.json"))
    frame_count = len(training_files)
    if frame_count == 0:
        raise WiyhWristDualInputError(f"no processed frames for {session_id}")

    attempt = (
        experiment_root
        / spec["experiment"]
        / "attempts"
        / "attempt_0001"
    )
    hawor_path = attempt / spec["hawor"]
    terminal_path = attempt / "TERMINAL_RESULT.json"
    _, terminal_ref = _terminal(terminal_path, EXPECTED_HAWOR_TERMINAL)
    hawor = _load_hawor(hawor_path, frame_count)

    frame_ids = np.empty(frame_count, dtype=np.int64)
    timestamps = np.empty(frame_count, dtype=np.float64)
    controller = np.empty((frame_count, 2, 4, 4), dtype=np.float64)
    maximum_translation_error_m = 0.0
    maximum_rotation_error_deg = 0.0
    previous_time = -np.inf
    for frame, path in enumerate(training_files):
        row = _load_json(path)
        metadata = row.get("metadata", {})
        frame_id = metadata.get("idx")
        timestamp_s = metadata.get("video_time_s")
        if frame_id != frame or not isinstance(timestamp_s, (int, float)):
            raise WiyhWristDualInputError(f"frame/timestamp drift at {path}")
        timestamp = float(timestamp_s)
        if not np.isfinite(timestamp) or timestamp <= previous_time:
            raise WiyhWristDualInputError(f"non-increasing video time at {path}")
        previous_time = timestamp
        frame_ids[frame] = frame_id
        timestamps[frame] = timestamp
        hands = row.get("entities", {}).get("hands", {})
        for side_index, side in enumerate(SIDES):
            hand = hands.get(side, {})
            controller_record = hand.get("controller6d", {})
            if controller_record.get("pose_source") != "egodex_v1_hdf5_controller_pose":
                raise WiyhWristDualInputError(f"controller pose source drift at {path}:{side}")
            if hand.get("pose_source") != "egodex_v1_hdf5_wrist_pose":
                raise WiyhWristDualInputError(f"legacy wrist pose source drift at {path}:{side}")
            camera_controller = _matrix(
                controller_record.get("T_controller_to_camera"),
                label=f"{path}:{side}:T_camera_controller",
            )
            legacy_camera_wrist = _matrix(
                hand.get("T_wrist_to_camera"),
                label=f"{path}:{side}:T_camera_wrist_legacy",
            )
            expected = camera_controller @ nominal_m0[side_index]
            maximum_translation_error_m = max(
                maximum_translation_error_m,
                float(np.linalg.norm(expected[:3, 3] - legacy_camera_wrist[:3, 3])),
            )
            maximum_rotation_error_deg = max(
                maximum_rotation_error_deg,
                _rotation_error_deg(expected[:3, :3], legacy_camera_wrist[:3, :3]),
            )
            controller[frame, side_index] = camera_controller
    if maximum_translation_error_m > 1e-6 or maximum_rotation_error_deg > 1e-4:
        raise WiyhWristDualInputError(
            f"processed legacy relation drift in {session_id}: "
            f"{maximum_translation_error_m * 1000.0:.6g} mm, "
            f"{maximum_rotation_error_deg:.6g} deg"
        )

    observed = np.asarray(hawor["observed"], dtype=bool).T
    wrist_xyz = np.asarray(hawor["joints_3d_camera"], dtype=np.float64)[:, :, 0].transpose(1, 0, 2)
    wrist_uv = np.asarray(hawor["joints_2d"], dtype=np.float64)[:, :, 0].transpose(1, 0, 2)
    root_rotation = np.asarray(hawor["root_orient_camera"], dtype=np.float64).transpose(1, 0, 2, 3)
    finite = np.isfinite(wrist_xyz).all(axis=-1) & np.isfinite(root_rotation).all(axis=(-1, -2))
    observed_valid = observed & finite
    observed_transform = np.full((frame_count, 2, 4, 4), np.nan, dtype=np.float64)
    observed_transform[..., 3, 3] = 1.0
    for frame in range(frame_count):
        for side_index in range(2):
            if not observed_valid[frame, side_index]:
                continue
            observed_transform[frame, side_index, :3, :3] = root_rotation[frame, side_index]
            observed_transform[frame, side_index, :3, 3] = wrist_xyz[frame, side_index]
    wrist_uv[~observed_valid] = np.nan

    arrays = {
        "frame_id": frame_ids,
        "timestamp_s": timestamps,
        "recording_id": np.full(frame_count, session_id, dtype="U32"),
        "T_camera_controller_raw": controller,
        "observed_T_camera_wrist": observed_transform,
        "observed_valid": observed_valid,
        "orientation_valid": np.zeros((frame_count, 2), dtype=bool),
        "observed_anatomical_wrist_uv": wrist_uv,
        "surface_source_pixel_uv": np.full((frame_count, 2, 2), np.nan, dtype=np.float64),
        "surface_patch_xyz_camera": np.full((frame_count, 2, 1, 3), np.nan, dtype=np.float64),
        "rgb_wrist_region_valid": np.zeros((frame_count, 2), dtype=bool),
        "source_pixel_traceable": np.zeros((frame_count, 2), dtype=bool),
        "mask_purity_valid": np.zeros((frame_count, 2), dtype=bool),
        "depth_rgb_registration_valid": np.zeros((frame_count, 2), dtype=bool),
        "local_depth_continuity_valid": np.zeros((frame_count, 2), dtype=bool),
        "contaminant_free": np.zeros((frame_count, 2), dtype=bool),
        "surface_role": np.full((frame_count, 2), "unknown", dtype="U32"),
    }
    provenance = {
        "session_id": session_id,
        "role": spec["role"],
        "frame_count": frame_count,
        "output_slice": {"start": output_start, "stop": output_start + frame_count},
        "processed_root": str(session_root),
        "training_data": _file_set(training_root, training_files),
        "hawor_npz": _artifact(hawor_path),
        "terminal_result": terminal_ref,
        "terminal_status": EXPECTED_HAWOR_TERMINAL,
        "observed_valid_count": {
            side: int(observed_valid[:, index].sum()) for index, side in enumerate(SIDES)
        },
    }
    return arrays, provenance, maximum_translation_error_m, maximum_rotation_error_deg


def _concat_sessions(values: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    names = set(values[0])
    if any(set(value) != names for value in values[1:]):
        raise WiyhWristDualInputError("session field sets differ")
    return {name: np.concatenate([value[name] for value in values], axis=0) for name in names}


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())


def _write_json(path: Path, value: dict[str, Any]) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _publish_directory_no_clobber(staging: Path, output: Path) -> None:
    """Atomically publish a complete directory without replacing a peer."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise WiyhWristDualInputError("atomic RENAME_NOREPLACE is unavailable on this platform")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,  # AT_FDCWD
        os.fsencode(staging),
        -100,
        os.fsencode(output),
        1,  # RENAME_NOREPLACE
    )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in (errno.EEXIST, errno.ENOTEMPTY):
        raise WiyhWristDualInputError(f"refusing to clobber output root: {output}")
    raise OSError(error, os.strerror(error), str(output))


def _assemble(
    *, processed_root: Path, experiment_root: Path
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any], float, float]:
    processed = processed_root.resolve(strict=True)
    experiments = experiment_root.resolve(strict=True)
    nominal_m0, m0_provenance = _load_m0(experiments)
    session_arrays: list[dict[str, np.ndarray]] = []
    session_provenance: list[dict[str, Any]] = []
    total = 0
    max_translation_error = 0.0
    max_rotation_error = 0.0
    for session_id in ALLOWED_SESSION_IDS:
        arrays, provenance, translation_error, rotation_error = _read_session(
            processed_root=processed,
            experiment_root=experiments,
            session_id=session_id,
            nominal_m0=nominal_m0,
            output_start=total,
        )
        session_arrays.append(arrays)
        session_provenance.append(provenance)
        total += len(arrays["frame_id"])
        max_translation_error = max(max_translation_error, translation_error)
        max_rotation_error = max(max_rotation_error, rotation_error)
    bundle = _concat_sessions(session_arrays)
    bundle["T_controller_wrist_M0"] = nominal_m0
    return (
        bundle,
        session_provenance,
        m0_provenance,
        max_translation_error,
        max_rotation_error,
    )


def preflight(*, processed_root: Path, experiment_root: Path) -> dict[str, Any]:
    """Read and validate all fixed sources without writing any output."""

    bundle, sessions, _, translation_error, rotation_error = _assemble(
        processed_root=processed_root,
        experiment_root=experiment_root,
    )
    return {
        "schema_version": "chaoyang-wiyh-wrist-dual-input-preflight-v1",
        "status": "PASS_CPU_SOURCE_PREFLIGHT",
        "allowed_sessions": list(ALLOWED_SESSION_IDS),
        "excluded_missing_observation_sessions": list(EXCLUDED_MISSING_OBSERVATION_IDS),
        "frame_count": int(len(bundle["frame_id"])),
        "sessions": [
            {
                "session_id": row["session_id"],
                "frame_count": row["frame_count"],
                "observed_valid_count": row["observed_valid_count"],
            }
            for row in sessions
        ],
        "maximum_legacy_relation_translation_error_mm": translation_error * 1000.0,
        "maximum_legacy_relation_rotation_error_deg": rotation_error,
        "orientation_valid_count": int(bundle["orientation_valid"].sum()),
        "surface_point_candidate_count": int(
            np.isfinite(bundle["surface_patch_xyz_camera"]).all(axis=-1).sum()
        ),
        "writes_performed": False,
        "source_mutated": False,
        "gpu_used": False,
    }


def build(*, processed_root: Path, experiment_root: Path, output_root: Path) -> dict[str, Any]:
    """Create one immutable bundle from the three pinned current sessions."""

    output = output_root.absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise WiyhWristDualInputError(f"refusing to clobber output root: {output}")
    (
        bundle,
        session_provenance,
        m0_provenance,
        max_translation_error,
        max_rotation_error,
    ) = _assemble(processed_root=processed_root, experiment_root=experiment_root)
    total = len(bundle["frame_id"])

    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging.", dir=output.parent))
    try:
        npz_path = staging / "WRIST_DUAL_INPUT.npz"
        _write_npz(npz_path, bundle)
        m0_provenance["processed_relation_audit"] = {
            "maximum_translation_error_mm": max_translation_error * 1000.0,
            "maximum_rotation_error_deg": max_rotation_error,
            "independent_truth": False,
        }
        provenance = {
            "schema_version": SCHEMA_VERSION,
            "status": "INPUT_BUNDLE_GENERATED",
            "allowed_sessions": list(ALLOWED_SESSION_IDS),
            "source_policy": {
                "current_non_archive_only": True,
                "fixed_source_mapping": True,
                "excluded_missing_observation_sessions": list(EXCLUDED_MISSING_OBSERVATION_IDS),
            },
            "sessions": session_provenance,
            "m0_legacy_prior": m0_provenance,
            "field_policy": {
                "observed_anatomical_wrist": "HAWOR_MODEL_DEFINED_WRIST_NOT_EXTERNAL_TRUTH",
                "orientation_valid_default": False,
                "visible_surface": "FAIL_CLOSED_NAN_AND_FALSE",
                "missing_observation_fill": "FORBIDDEN",
            },
            "temporal_authority": {
                "controller": "UNKNOWN_TEMPORAL_AUTHORITY",
                "hawor_observation": "OFFLINE_NONCAUSAL",
                "visible_surface": "UNKNOWN_TEMPORAL_AUTHORITY",
            },
            "outputs": {
                "npz": _artifact(npz_path, published_path=output / npz_path.name),
                "frame_count": total,
            },
            "authority": {
                "external_wrist_truth": False,
                "external_metric_authority": False,
                "surface_point_authority": False,
                "physical_deployment_authorized": False,
            },
            "execution": {
                "cpu_only": True,
                "gpu_used": False,
                "source_mutated": False,
                "no_clobber": True,
            },
        }
        repo_root = Path(__file__).resolve().parents[3]
        schema = json.loads(
            (repo_root / "contracts/wiyh_wrist_dual_input_provenance_v1.schema.json").read_text(
                encoding="utf-8"
            )
        )
        jsonschema.validate(provenance, schema)
        _write_json(staging / "PROVENANCE.json", provenance)
        # The staging directory and target share a parent/filesystem. Linux
        # RENAME_NOREPLACE publishes both complete files as one namespace update
        # and rejects even an empty target created by a racing writer.
        _publish_directory_no_clobber(staging, output)
        return provenance
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        print(
            json.dumps(
                preflight(
                    processed_root=args.processed_root,
                    experiment_root=args.experiment_root,
                ),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.output_root is None:
        parser.error("--output-root is required unless --preflight-only is set")
    print(
        json.dumps(
            build(
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
