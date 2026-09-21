"""CPU-only HuRo core preflight. This module does not implement or execute IK.

Dependency metadata is not an import/runtime proof. There is intentionally no
fallback to Chaoyang's older HuRo-derived solver and no generated joint angles.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

UPSTREAM_COMMIT = "033197778fcc30edc3631dddf3343a967683da09"
CORE_SOURCE_SHA = "330c0adace131919c2c05a8ffa1e40689a197e3e254fd75cd549cb7cc42939e1"
SOURCE_FILES = {
    "pipeline/retargeting/retargeter.py": (37675, CORE_SOURCE_SHA),
    "common/robot_config.py": (2856, "e0a989a379fe60f6bc8749b6326f2e45a8c44ab4261bd9a8f754b71299c5a2f7"),
    "configs/allex.yaml": (6001, "7c6103b57309fc9e60bcf8b8787b2c4eaf952665c35ff5f7db618b8c3e8d1bb9"),
    "setup/setup.sh": (14167, "9804ae964261b4de59b9d476b775a9dc2821ac46028d7f42cba6686a4b4152e9"),
}
REQUIREMENTS = {
    "jax": "==0.6.2; upstream requests cuda12 extra",
    "jaxlib": "required by JAX; no independent pin in setup.sh",
    "jaxlie": ">=1.0.0",
    "jaxls": "git commit 50a58be88c5ef74532f09e3f55268b4f02c490e3",
    "pyroki": "upstream submodules/pyroki; source checkout required",
    "yourdfpy": "unversioned upstream requirement",
    "jax-dataclasses": ">=1.6.2",
    "jaxtyping": "unversioned upstream requirement",
    "numpy": "==1.26.0 in upstream full-stack final pins",
    "loguru": "==0.7.3 in upstream setup",
}


class AdapterValidationError(ValueError):
    pass


def dependency_metadata(version_fn=None):
    """Inspect installed distribution metadata without importing GPU modules."""
    version_fn = version_fn or importlib.metadata.version
    rows = []
    for name, declared in REQUIREMENTS.items():
        try:
            version = version_fn(name)
        except importlib.metadata.PackageNotFoundError:
            version = None
        rows.append({"distribution": name, "installed_version": version,
                     "upstream_requirement": declared,
                     "presence": "PRESENT_METADATA_ONLY" if version else "MISSING"})
    return {"interpreter": sys.executable, "python": sys.version.split()[0],
            "packages": rows, "missing": [r["distribution"] for r in rows if r["installed_version"] is None],
            "version_compatibility": "NOT_VERIFIED", "runtime_imports": "NOT_EXECUTED"}


def verify_upstream_sources(vendor_root):
    root = Path(vendor_root).resolve(strict=True)
    refs = []
    for relative, (expected_size, expected_sha) in SOURCE_FILES.items():
        path = root / relative
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root):
            raise AdapterValidationError("upstream source path escape")
        before = path.stat()
        raw = path.read_bytes()
        after = path.stat()
        identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        if identity(before) != identity(after):
            raise AdapterValidationError("upstream source changed during read")
        digest = hashlib.sha256(raw).hexdigest()
        if len(raw) != expected_size or digest != expected_sha:
            raise AdapterValidationError(f"upstream source drift: {relative}")
        refs.append({"path": str(path), "bytes": len(raw), "sha256": digest})
    return refs


def validate_frozen_placement(matrix, expected_sha256):
    """Validate a fixed proper SE(3); never estimate or adjust placement."""
    if len(matrix) != 4 or any(len(row) != 4 for row in matrix):
        raise AdapterValidationError("placement must be 4x4")
    if any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v)
           for row in matrix for v in row):
        raise AdapterValidationError("placement must be finite numeric")
    m = [[float(v) for v in row] for row in matrix]
    if any(abs(a-b) > 1e-8 for a, b in zip(m[3], [0, 0, 0, 1])):
        raise AdapterValidationError("invalid homogeneous row")
    for i in range(3):
        for j in range(3):
            if abs(sum(m[k][i]*m[k][j] for k in range(3)) - (i == j)) > 1e-6:
                raise AdapterValidationError("placement rotation not orthonormal")
    determinant = sum(m[0][i] * (m[1][(i+1)%3]*m[2][(i+2)%3] - m[1][(i+2)%3]*m[2][(i+1)%3]) for i in range(3))
    if abs(determinant - 1) > 1e-6:
        raise AdapterValidationError("placement reflection forbidden")
    digest = hashlib.sha256(json.dumps(m, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if digest != expected_sha256:
        raise AdapterValidationError("frozen placement SHA mismatch")
    return {"sha256": digest, "placement_optimized": False}


def validate_samples(frame_ids, timestamps_ns, points, observed, side_names, units):
    """Validate original time axis and independent (T, 2, 21) point masks.

Invalid coordinates may remain NaN. Invalid points are never filled here.
"""
    if list(side_names) != ["left", "right"] or units != "m":
        raise AdapterValidationError("requires anatomical left/right and metres")
    n = len(frame_ids)
    if n == 0 or any(len(v) != n for v in (timestamps_ns, points, observed)):
        raise AdapterValidationError("empty or inconsistent frame axis")
    for axis, label in ((frame_ids, "frame_id"), (timestamps_ns, "timestamp_ns")):
        if any(type(v) is not int for v in axis):
            raise AdapterValidationError(f"{label} must contain integer values")
        if any(b <= a for a, b in zip(axis, axis[1:])):
            raise AdapterValidationError(f"{label} must strictly increase; no duplicate/backward values")
    for t in range(n):
        if len(points[t]) != 2 or len(observed[t]) != 2:
            raise AdapterValidationError("requires two independent hand axes")
        for side in range(2):
            if len(points[t][side]) != 21 or len(observed[t][side]) != 21:
                raise AdapterValidationError("requires 21 point masks per hand")
            for xyz, valid in zip(points[t][side], observed[t][side]):
                if type(valid) is not bool or len(xyz) != 3:
                    raise AdapterValidationError("point mask must be bool; coordinate must have 3 components")
                if valid and any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) for v in xyz):
                    raise AdapterValidationError("observed point has invalid coordinates")
    return {"frames": n, "observed_point_counts": [sum(sum(row[s]) for row in observed) for s in (0, 1)],
            "inference_added": False, "time_unit": "ns", "time_normalized_solver": "UNIMPLEMENTED"}


def continuous_segments(frame_ids, timestamps_ns, observed, *, max_gap_ns, max_frames=224):
    """Describe per-hand contiguous runs, never compress missing source frames.

This is an adapter preflight, not actual solver replay. Chunk boundaries are
reported and cannot be called a full-trajectory optimization.
"""
    if type(max_gap_ns) is not int or max_gap_ns <= 0 or type(max_frames) is not int or max_frames <= 0:
        raise AdapterValidationError("positive integer gap/chunk limits required")
    n = len(frame_ids)
    if n == 0 or len(timestamps_ns) != n or len(observed) != n:
        raise AdapterValidationError("inconsistent time/mask axis")
    if any(type(v) is not int for v in frame_ids + timestamps_ns) or any(b <= a for seq in (frame_ids, timestamps_ns) for a, b in zip(seq, seq[1:])):
        raise AdapterValidationError("time and frame axes must strictly increase")
    if any(len(row) != 2 or any(len(side) != 21 or any(type(v) is not bool for v in side) for side in row) for row in observed):
        raise AdapterValidationError("requires independent per-point bool masks")
    result = []
    for side in (0, 1):
        run = []
        def finish(reason):
            if run:
                result.append({"side": ["left", "right"][side], "source_indices": list(run),
                               "source_frame_ids": [frame_ids[i] for i in run], "boundary": reason,
                               "solver_executed": False})
                run.clear()
        for i in range(n):
            if not any(observed[i][side]):
                finish("INVALID_HAND")
                continue
            if run and (frame_ids[i] != frame_ids[run[-1]]+1 or timestamps_ns[i]-timestamps_ns[run[-1]] > max_gap_ns):
                finish("SOURCE_GAP")
            if len(run) == max_frames:
                finish("CHUNK_LIMIT")
            run.append(i)
        finish("END")
    return result


def validate_robot_mapping(urdf_text, config):
    """Structural mapping proof only; FK parity and physical mounts unverified."""
    root = ET.fromstring(urdf_text)
    links = [x.attrib["name"] for x in root.findall("link")]
    joints = root.findall("joint")
    if len(links) != len(set(links)):
        raise AdapterValidationError("duplicate URDF link names")
    actuated = {x.attrib["name"] for x in joints if x.attrib.get("type") not in ("fixed", "floating")}
    groups = config["joints"]
    active = [j for values in groups.values() for j in values]
    if len(active) != len(set(active)) or not set(active) <= actuated:
        raise AdapterValidationError("active joints duplicate or absent")
    used = []
    for side in ("left", "right"):
        mapping = {int(k): v for k, v in config["keypoint_mapping"][side].items()}
        if not {0, 4, 8, 12, 16, 20} <= set(mapping) or not set(mapping) <= set(range(21)):
            raise AdapterValidationError("requires wrist and five tips, valid MANO21 indices")
        if not set(mapping.values()) <= set(links):
            raise AdapterValidationError("keypoint link absent from URDF")
        if config["eef_link_names"][side] not in links or config["eef_hand_joint_groups"][side] not in groups:
            raise AdapterValidationError("EEF or hand group missing")
        used.append(set(mapping.values()))
    if used[0] & used[1]:
        raise AdapterValidationError("left/right keypoint links must be distinct")
    if config["camera_link"] not in links:
        raise AdapterValidationError("camera reference link missing")
    return {"active_joints": active, "locked_joints": sorted(actuated-set(active)),
            "fk_parity": "NOT_EXECUTED", "mount_accuracy": "UNVERIFIED"}


def core_execution_status(metadata):
    return {"status": "BLOCKED_DEPENDENCY" if metadata["missing"] else "BLOCKED_CORE_ADAPTER_UNIMPLEMENTED",
            "official_core_executed": False, "old_derived_solver_used": False,
            "numeric_quality_pass": False, "training_eligible": False,
            "control_ground_truth": False, "videos_generated": 0,
            "gpu_used": False, "source_data_modified": False,
            "fk_parity": "NOT_EXECUTED", "adapter_contract_tests_only": True}
