#!/usr/bin/env python3
"""Read-only CPU self-check for the frozen V2.1 C1 Poker119 evidence.

This program deliberately has no Contact publisher, registration estimator, model
runner, or threshold-tuning path.  It verifies immutable cached evidence and ends
with Contact UNKNOWN when the overlap window has no independently observed finger
surface or when physical RGB/depth registration remains unbound.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


EXPECTED_DESIGN_SCHEMA = "0915-robot-recovery-v21-c1-minimal-diagnostic-design-v1"
EXPECTED_INDEX_SCHEMA = "0915-robot-recovery-v21-c1-executability-input-index-v1"
EXPECTED_SESSION = "play_cards_0915_119"
EXPECTED_DISTANCE_M = 0.005
EXPECTED_MASK_HW = (960, 1280)
EXPECTED_DEPTH_HW = (480, 640)
EXPECTED_RADIUS_DEPTH_PX = 4
EXPECTED_TARGET = ("right", "index", "playing_card_02")
EXPECTED_TARGET_FRAMES = tuple(range(92, 99))
EXPECTED_CONTROL_FRAMES = (74, 75, 76, 77, 163, 164, 165)
EXPECTED_PAIR_ROWS = 5040
ROUNDTRIP_TOL_PX = 1e-6
DEPTH_TOL_M = 1e-6
METRIC_REPLAY_TOL_M = 2e-6


class SelfCheckError(RuntimeError):
    """A named C1 stop terminal."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def fail(code: str, message: str) -> None:
    raise SelfCheckError(code, message)


def load_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        fail("STOP_INPUT_DRIFT", "cannot read JSON {}: {}".format(path, exc))
    if not isinstance(value, dict):
        fail("STOP_INPUT_DRIFT", "JSON object required: {}".format(path))
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, np.float64), q))


class Inputs:
    def __init__(self, index: Mapping[str, Any], repo_root: Optional[Path]):
        self.index = index
        self.canonical_root = Path(str(index["repo_root"]))
        self.actual_root = repo_root.resolve() if repo_root is not None else None
        self.by_role: Dict[str, Mapping[str, Any]] = {}
        for group in ("data_inputs", "code_inputs"):
            for row in index.get(group, []):
                role = row.get("role")
                if role:
                    self.by_role[str(role)] = row

    def path(self, value: str) -> Path:
        original = Path(value)
        if self.actual_root is None:
            return original
        try:
            relative = original.relative_to(self.canonical_root)
        except ValueError:
            fail("STOP_INPUT_DRIFT", "indexed path escapes repo root: {}".format(original))
        return self.actual_root / relative

    def role_path(self, role: str) -> Path:
        if role not in self.by_role:
            fail("STOP_INPUT_DRIFT", "missing indexed role: {}".format(role))
        return self.path(str(self.by_role[role]["path"]))


def verify_ref(inputs: Inputs, ref: Mapping[str, Any]) -> Tuple[Path, int]:
    path = inputs.path(str(ref.get("path", "")))
    if not path.is_file():
        fail("STOP_INPUT_DRIFT", "missing input: {}".format(path))
    expected_bytes = int(ref.get("bytes", -1))
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        fail(
            "STOP_INPUT_DRIFT",
            "byte drift for {}: {} != {}".format(path, actual_bytes, expected_bytes),
        )
    actual_sha = sha256(path)
    if actual_sha != ref.get("sha256"):
        fail("STOP_INPUT_DRIFT", "SHA-256 drift for {}".format(path))
    return path, actual_bytes


def verify_design(design: Mapping[str, Any]) -> Dict[str, Any]:
    if design.get("schema_version") != EXPECTED_DESIGN_SCHEMA:
        fail("STOP_INPUT_DRIFT", "unexpected C1 design schema")
    frozen = design.get("frozen_constants", {})
    expected = {
        "contact_distance_m": EXPECTED_DISTANCE_M,
        "distance_gate_may_expand_with_uncertainty": False,
        "mask_shape_hw": list(EXPECTED_MASK_HW),
        "depth_shape_hw": list(EXPECTED_DEPTH_HW),
        "depth_to_mask_formula": "mask_uv = 2 * depth_uv + 0.5",
        "sampling_radius_depth_px": EXPECTED_RADIUS_DEPTH_PX,
        "candidate_identity": ":".join(EXPECTED_TARGET),
        "candidate_frames_inclusive": [92, 98],
        "negative_control_frames": list(EXPECTED_CONTROL_FRAMES),
    }
    for key, value in expected.items():
        if frozen.get(key) != value:
            fail("STOP_INPUT_DRIFT", "C1 frozen constant drift: {}".format(key))
    decision = design.get("entry_decision", {})
    if (
        decision.get("internal_coordinate_self_check") != "GO"
        or decision.get("contact_successor") != "NO_GO_WITH_CURRENT_EVIDENCE"
    ):
        fail("STOP_INPUT_DRIFT", "C1 entry decision drift")
    return expected


def verify_package_decisions(
    inputs: Inputs,
    index_path: Path,
    design_path: Path,
    c1_result_path: Path,
) -> Dict[str, Any]:
    invalidated = load_json(inputs.role_path("invalidated_c0_tombstone"))
    c0_v2 = load_json(inputs.role_path("authoritative_corrected_c0_result"))
    c1 = load_json(c1_result_path)
    c0_v2_sha = sha256(inputs.role_path("authoritative_corrected_c0_result"))
    if (
        invalidated.get("package_id") != "C0"
        or invalidated.get("reason_code") != "OBSERVED_SURFACE_STATUS_FILTER_OMITTED"
        or invalidated.get("successor_attempt") != "packages/C0_v2"
        or invalidated.get("successor_result_sha256") != c0_v2_sha
        or invalidated.get("authority_promoted") is not False
    ):
        fail("STOP_INPUT_DRIFT", "C0 invalidation/successor binding drift")
    if (
        c0_v2.get("status") != "PASSED"
        or c0_v2.get("decision") != "CONTACT_STAYS_UNKNOWN_AND_R1_E_CLOSED"
        or c0_v2.get("control_ground_truth") is not False
        or c0_v2.get("training_eligible") is not False
    ):
        fail("STOP_INPUT_DRIFT", "C0_v2 decision drift")
    if (
        c1.get("schema_version") != "0915-robot-recovery-v21-c1-result-v1"
        or c1.get("status") != "PASSED_DIAGNOSTIC_NO_CONTACT_SUCCESSOR"
        or c1.get("decision") != "GO_CPU_INTERNAL_SELF_CHECK_ONLY_NO_GO_CONTACT_SUCCESSOR"
        or c1.get("authoritative_input") != "C0_v2"
        or c1.get("invalidated_input_excluded") != "C0"
        or c1.get("contact_authority") != "NONE"
        or c1.get("r1_e_authorized") is not False
    ):
        fail("STOP_INPUT_DRIFT", "C1 package decision drift")
    expected_paths = {
        "minimal_diagnostic_design": design_path,
        "executability_input_index": index_path,
    }
    artifact_count = 0
    for name, ref in c1.get("artifacts", {}).items():
        if not isinstance(ref, dict):
            fail("STOP_INPUT_DRIFT", "C1 artifact reference drift")
        actual, _size = verify_ref(inputs, ref)
        if name in expected_paths and actual.resolve() != expected_paths[name].resolve():
            fail("STOP_INPUT_DRIFT", "C1 artifact route drift: {}".format(name))
        artifact_count += 1
    if artifact_count != 4:
        fail("STOP_INPUT_DRIFT", "C1 artifact set drift")
    return {
        "status": "PASS",
        "invalidated_c0_excluded": True,
        "authoritative_input": "C0_v2",
        "c0_v2_decision": c0_v2["decision"],
        "c1_decision": c1["decision"],
        "c1_artifact_references_verified": artifact_count,
    }


def verify_all_inputs(
    inputs: Inputs,
) -> Tuple[Dict[str, Any], List[Mapping[str, Any]], Dict[str, Path]]:
    index = inputs.index
    if index.get("schema_version") != EXPECTED_INDEX_SCHEMA:
        fail("STOP_INPUT_DRIFT", "unexpected input-index schema")
    if index.get("status") != "EXECUTABLE_FROM_EXISTING_CACHE_CPU_ONLY":
        fail("STOP_INPUT_DRIFT", "input index is not CPU-only executable")
    files = 0
    total_bytes = 0
    resolved: Dict[str, Path] = {}
    for group in ("data_inputs", "object_mask_inputs", "hand_mask_inputs", "code_inputs"):
        for row in index.get(group, []):
            path, size = verify_ref(inputs, row)
            files += 1
            total_bytes += size
            role = str(row.get("role") or path.name)
            resolved[role] = path

    depth_ref = inputs.by_role.get("depth_frame_manifest")
    if depth_ref is None:
        fail("STOP_INPUT_DRIFT", "depth frame manifest not indexed")
    depth_summary = load_json(inputs.path(str(depth_ref["path"])))
    frame_rows = depth_summary.get("frames")
    if not isinstance(frame_rows, list) or len(frame_rows) != int(depth_summary.get("frame_count", -1)):
        fail("STOP_INPUT_DRIFT", "depth frame axis drift")
    nested_refs: List[Mapping[str, Any]] = []
    nested_bytes = 0
    for expected_frame, row in enumerate(frame_rows):
        if row.get("frame") != expected_frame or not isinstance(row.get("artifact"), dict):
            fail("STOP_INPUT_DRIFT", "depth frame manifest discontinuity at {}".format(expected_frame))
        ref = row["artifact"]
        _path, size = verify_ref(inputs, ref)
        nested_refs.append(ref)
        nested_bytes += size
    if len(nested_refs) != int(depth_ref.get("nested_frame_count", -1)):
        fail("STOP_INPUT_DRIFT", "nested depth count drift")
    if nested_bytes != int(depth_ref.get("nested_total_bytes", -1)):
        fail("STOP_INPUT_DRIFT", "nested depth byte-count drift")
    if canonical_sha(nested_refs) != depth_ref.get("nested_ordered_refs_canonical_sha256"):
        fail("STOP_INPUT_DRIFT", "nested depth ordered-reference SHA drift")
    return (
        {
            "status": "PASS",
            "top_level_files_verified": files,
            "top_level_bytes_verified": total_bytes,
            "nested_depth_files_verified": len(nested_refs),
            "nested_depth_bytes_verified": nested_bytes,
            "nested_ordered_refs_canonical_sha256": canonical_sha(nested_refs),
        },
        frame_rows,
        resolved,
    )


def load_depth(inputs: Inputs, row: Mapping[str, Any], expected_frame: int) -> Dict[str, np.ndarray]:
    path = inputs.path(str(row["artifact"]["path"]))
    try:
        with np.load(str(path), allow_pickle=False) as archive:
            value = {key: np.asarray(archive[key]) for key in archive.files}
    except Exception as exc:
        fail("STOP_INPUT_DRIFT", "cannot read depth frame {}: {}".format(path, exc))
    required = ("frame_id", "depth_m", "valid", "lr_consistent", "physical_left_intrinsics", "depth_reference")
    if any(key not in value for key in required):
        fail("STOP_INPUT_DRIFT", "depth array contract incomplete: {}".format(path))
    if (
        int(value["frame_id"]) != expected_frame
        or value["depth_m"].shape != EXPECTED_DEPTH_HW
        or value["valid"].shape != EXPECTED_DEPTH_HW
        or value["lr_consistent"].shape != EXPECTED_DEPTH_HW
        or value["physical_left_intrinsics"].shape != (3, 3)
        or str(value["depth_reference"]) != "PHYSICAL_LEFT_ENCODED_RESIZE_ONLY_OPTICAL_Z"
    ):
        fail("STOP_INPUT_DRIFT", "depth array contract drift: {}".format(path))
    return value


def project(points: np.ndarray, k: np.ndarray) -> np.ndarray:
    z = points[:, 2]
    if np.any(~np.isfinite(points)) or np.any(z <= 0):
        fail("STOP_INTERNAL_COORDINATE_IMPLEMENTATION_DRIFT", "non-finite/negative 3D point")
    return np.column_stack((
        points[:, 0] * k[0, 0] / z + k[0, 2],
        points[:, 1] * k[1, 1] / z + k[1, 2],
    ))


def point_segment_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    delta = b - a
    denominator = float(np.dot(delta, delta))
    if denominator == 0.0:
        return float(np.linalg.norm(point - a))
    t = min(1.0, max(0.0, float(np.dot(point - a, delta) / denominator)))
    return float(np.linalg.norm(point - (a + t * delta)))


def polygon_inside_and_distance(point: np.ndarray, polygon: np.ndarray) -> Tuple[bool, float]:
    if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2:
        fail("STOP_INTERNAL_COORDINATE_IMPLEMENTATION_DRIFT", "invalid finite patch polygon")
    minimum = float("inf")
    inside = False
    x, y = float(point[0]), float(point[1])
    for i in range(len(polygon)):
        a = polygon[i]
        b = polygon[(i + 1) % len(polygon)]
        minimum = min(minimum, point_segment_distance(point, a, b))
        ay, by = float(a[1]), float(b[1])
        if (ay > y) != (by > y):
            crossing_x = float(a[0]) + (y - ay) * (float(b[0]) - float(a[0])) / (by - ay)
            if crossing_x > x:
                inside = not inside
    if minimum <= 1e-10:
        inside = True
    return inside, minimum


def replay_finite_metric(point_xyz: Sequence[float], frame: Mapping[str, Any]) -> Dict[str, Any]:
    center = np.asarray(frame["center_xyz"]["estimate"]["xyz_m"], np.float64)
    normal = np.asarray(frame["plane_normal"]["estimate"]["unit_xyz"], np.float64)
    normal /= np.linalg.norm(normal)
    axis_u = np.asarray(frame["inplane_rotation"]["estimate"]["axis_unit_xyz"], np.float64)
    axis_u -= normal * float(np.dot(axis_u, normal))
    axis_u /= np.linalg.norm(axis_u)
    axis_v = np.cross(normal, axis_u)
    axis_v /= np.linalg.norm(axis_v)
    boundary = np.asarray(frame["finite_visible_patch"]["estimate"]["boundary_xyz_m"], np.float64)
    polygon = np.column_stack(((boundary - center) @ axis_u, (boundary - center) @ axis_v))
    delta = np.asarray(point_xyz, np.float64) - center
    local = np.asarray([float(delta @ axis_u), float(delta @ axis_v)])
    inside, boundary_distance = polygon_inside_and_distance(local, polygon)
    signed_plane = float(delta @ normal)
    outside = 0.0 if inside else boundary_distance
    return {
        "inside": inside,
        "boundary_distance_m": boundary_distance,
        "signed_plane_m": signed_plane,
        "finite_distance_m": float(math.hypot(signed_plane, outside)),
    }


def _as_object_frame_map(document: Mapping[str, Any]) -> Dict[Tuple[str, int], Mapping[str, Any]]:
    result: Dict[Tuple[str, int], Mapping[str, Any]] = {}
    for obj in document.get("objects", []):
        object_id = str(obj.get("instance_id"))
        for expected, frame in enumerate(obj.get("frames", [])):
            if frame.get("frame_index") != expected:
                fail("STOP_INPUT_DRIFT", "Object6D frame-axis drift for {}".format(object_id))
            result[(object_id, expected)] = frame
    return result


def audit_roundtrip_and_support(
    inputs: Inputs,
    depth_rows: Sequence[Mapping[str, Any]],
    object_document: Mapping[str, Any],
    support_path: Path,
    interaction: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[Tuple[str, int], Mapping[str, Any]]]:
    frame_count = len(depth_rows)
    object_map = _as_object_frame_map(object_document)
    metric_by_frame: Dict[int, List[Mapping[str, Any]]] = {}
    for row in interaction.get("rows", []):
        if isinstance(row.get("metric"), dict):
            metric_by_frame.setdefault(int(row["frame_id"]), []).append(row)
    try:
        with np.load(str(support_path), allow_pickle=False) as archive:
            packed = np.asarray(archive["packed"])
            instance_ids = [str(value) for value in archive["instance_ids"].tolist()]
            height, width = int(archive["height"]), int(archive["width"])
            archived_frames = int(archive["frame_count"])
            bitorder = str(archive["bitorder"])
            row_layout = str(archive["row_layout"])
            semantics = str(archive["semantics"])
    except Exception as exc:
        fail("STOP_INPUT_DRIFT", "cannot read packed finite support: {}".format(exc))
    expected_shape = (len(instance_ids) * frame_count, (height * width + 7) // 8)
    if (
        (height, width) != EXPECTED_DEPTH_HW
        or archived_frames != frame_count
        or bitorder != "big"
        or row_layout != "INSTANCE_MAJOR_FLAT_ROW"
        or semantics != "EXACT_DIRECT_VISIBLE_DEPTH_SUPPORT_ONLY"
        or packed.shape != expected_shape
        or packed.dtype != np.uint8
    ):
        fail("STOP_INPUT_DRIFT", "packed finite-support contract drift")
    document_ids = [str(row.get("instance_id")) for row in object_document.get("objects", [])]
    if instance_ids != document_ids:
        fail("STOP_INPUT_DRIFT", "packed finite-support instance order drift")

    support_counts: List[int] = []
    boundary_residuals: List[float] = []
    boundary_depth_errors: List[float] = []
    sample_residuals: List[float] = []
    source_shifts: List[float] = []
    finite_distances: List[float] = []
    inside_rows = 0
    within_gate_rows = 0
    boundaries_checked = 0
    patches_checked = 0
    unobservable_zero_support_rows = 0
    all_archive_rows: set = set()

    for frame_id, depth_ref in enumerate(depth_rows):
        depth = load_depth(inputs, depth_ref, frame_id)
        depth_m = np.asarray(depth["depth_m"], np.float64)
        valid = np.asarray(depth["valid"], bool)
        k = np.asarray(depth["physical_left_intrinsics"], np.float64)
        if not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
            fail("STOP_INPUT_DRIFT", "invalid depth intrinsics at frame {}".format(frame_id))
        for object_index, object_id in enumerate(instance_ids):
            object_frame = object_map.get((object_id, frame_id))
            if object_frame is None:
                fail("STOP_INPUT_DRIFT", "missing Object6D frame {}:{}".format(object_id, frame_id))
            finite = object_frame.get("finite_visible_patch", {})
            estimate = finite.get("estimate")
            expected_row = object_index * frame_count + frame_id
            if finite.get("observability") != "OBSERVABLE_DIRECT_VISIBLE_FINITE_SUPPORT":
                if estimate is not None or np.any(packed[expected_row]):
                    fail(
                        "STOP_INTERNAL_COORDINATE_IMPLEMENTATION_DRIFT",
                        "unobservable patch carries non-empty packed support {}:{}".format(object_id, frame_id),
                    )
                all_archive_rows.add(expected_row)
                support_counts.append(0)
                unobservable_zero_support_rows += 1
                continue
            if not isinstance(estimate, dict):
                fail("STOP_INTERNAL_COORDINATE_IMPLEMENTATION_DRIFT", "observable patch has no estimate")
            archive_row = int(estimate.get("archive_row", -1))
            if archive_row != expected_row or archive_row in all_archive_rows:
                fail("STOP_INTERNAL_COORDINATE_IMPLEMENTATION_DRIFT", "packed support row-layout drift")
            all_archive_rows.add(archive_row)
            support = np.unpackbits(
                packed[archive_row], bitorder="big", count=height * width
            ).reshape(height, width).astype(bool)
            yy, xx = np.nonzero(support)
            count = int(len(xx))
            support_counts.append(count)
            if count != int(estimate.get("support_pixel_count", -1)):
                fail("STOP_INTERNAL_COORDINATE_IMPLEMENTATION_DRIFT", "packed support count drift")
            if not count or np.any(~valid[yy, xx]) or np.any(~np.isfinite(depth_m[yy, xx])) or np.any(depth_m[yy, xx] <= 0):
                fail("STOP_INTERNAL_COORDINATE_IMPLEMENTATION_DRIFT", "packed support is not finite valid depth")
            bbox = [int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())]
            if bbox != estimate.get("depth_pixel_bbox_xyxy_inclusive"):
                fail("STOP_INTERNAL_COORDINATE_IMPLEMENTATION_DRIFT", "packed support bbox drift")
            boundary_uv = np.asarray(estimate.get("boundary_depth_uv"), np.int64)
            boundary_xyz = np.asarray(estimate.get("boundary_xyz_m"), np.float64)
            if (
                boundary_uv.ndim != 2
                or boundary_uv.shape[1] != 2
                or boundary_xyz.shape != (len(boundary_uv), 3)
                or len(boundary_uv) < 3
            ):
                fail("STOP_INTERNAL_COORDINATE_IMPLEMENTATION_DRIFT", "invalid finite patch boundary")
            bx, by = boundary_uv[:, 0], boundary_uv[:, 1]
            if np.any(bx < 0) or np.any(bx >= width) or np.any(by < 0) or np.any(by >= height):
                fail("STOP_INTERNAL_COORDINATE_IMPLEMENTATION_DRIFT", "finite patch boundary out of bounds")
            if not np.all(support[by, bx]):
                fail("STOP_INTERNAL_COORDINATE_IMPLEMENTATION_DRIFT", "boundary pixel absent from packed support")
            reprojection = project(boundary_xyz, k)
            residual = np.linalg.norm(reprojection - boundary_uv.astype(np.float64), axis=1)
            depth_error = np.abs(boundary_xyz[:, 2] - depth_m[by, bx])
            boundary_residuals.extend(residual.tolist())
            boundary_depth_errors.extend(depth_error.tolist())
            if float(residual.max()) > ROUNDTRIP_TOL_PX or float(depth_error.max()) > DEPTH_TOL_M:
                fail("STOP_INTERNAL_COORDINATE_IMPLEMENTATION_DRIFT", "boundary packed-depth round-trip drift")
            boundaries_checked += len(boundary_uv)
            patches_checked += 1

        for row in metric_by_frame.get(frame_id, []):
            sample = row.get("finger_associated_visible_surface_point", {})
            point = np.asarray(sample.get("surface_point_xyz"), np.float64)
            stored_uv = np.asarray(sample.get("sample_pixel_uv_depth_domain"), np.float64)
            source_uv = np.asarray(sample.get("source_pixel_uv"), np.float64)
            if point.shape != (3,) or stored_uv.shape != (2,) or source_uv.shape != (2,):
                fail("STOP_INTERNAL_COORDINATE_IMPLEMENTATION_DRIFT", "metric sample geometry incomplete")
            projected = project(point.reshape(1, 3), k)[0]
            sample_residual = float(np.linalg.norm(projected - stored_uv))
            analytic_uv = (source_uv - 0.5) / 2.0
            source_shift = float(np.linalg.norm(stored_uv - analytic_uv))
            sample_residuals.append(sample_residual)
            source_shifts.append(source_shift)
            if sample_residual > ROUNDTRIP_TOL_PX or source_shift > 4.75:
                fail("STOP_INTERNAL_COORDINATE_IMPLEMENTATION_DRIFT", "finger surface round-trip drift")
            object_frame = object_map.get((str(row["object_id"]), frame_id))
            replay = replay_finite_metric(point, object_frame)
            metric = row["metric"]
            if replay["inside"] is not (metric.get("inside_visible_patch") is True):
                fail("STOP_INTERNAL_COORDINATE_IMPLEMENTATION_DRIFT", "finite patch inside/outside replay drift")
            comparisons = (
                (replay["finite_distance_m"], metric.get("finite_patch_distance_m")),
                (replay["boundary_distance_m"], metric.get("distance_to_visible_patch_boundary_m")),
                (replay["signed_plane_m"], metric.get("object_plane_signed_distance_m")),
            )
            if any(
                not isinstance(stored, (int, float)) or abs(calculated - float(stored)) > METRIC_REPLAY_TOL_M
                for calculated, stored in comparisons
            ):
                fail("STOP_INTERNAL_COORDINATE_IMPLEMENTATION_DRIFT", "finite patch metric replay drift")
            distance = float(metric["finite_patch_distance_m"])
            finite_distances.append(distance)
            inside_rows += int(metric.get("inside_visible_patch") is True)
            within_gate_rows += int(metric.get("inside_visible_patch") is True and distance <= EXPECTED_DISTANCE_M)

    if len(all_archive_rows) != packed.shape[0]:
        fail("STOP_INTERNAL_COORDINATE_IMPLEMENTATION_DRIFT", "not all packed support rows were consumed")
    return ({
        "status": "PASS",
        "finite_patches_checked": patches_checked,
        "unobservable_zero_support_rows_checked": unobservable_zero_support_rows,
        "packed_support_rows_checked": len(all_archive_rows),
        "support_pixel_count": {
            "min": min(support_counts),
            "p50": percentile(support_counts, 50),
            "max": max(support_counts),
        },
        "boundary_points_checked": boundaries_checked,
        "boundary_membership_fraction": 1.0,
        "support_pixels_finite_valid_fraction": 1.0,
        "boundary_reprojection_max_px": max(boundary_residuals) if boundary_residuals else None,
        "boundary_depth_roundtrip_max_abs_m": max(boundary_depth_errors) if boundary_depth_errors else None,
        "metric_rows_checked": len(finite_distances),
        "surface_reprojection_max_px": max(sample_residuals) if sample_residuals else None,
        "source_to_selected_depth_shift_px": {
            "p50": percentile(source_shifts, 50),
            "p95": percentile(source_shifts, 95),
            "max": max(source_shifts) if source_shifts else None,
        },
        "finite_patch_distance_m": {
            "min": min(finite_distances) if finite_distances else None,
            "p50": percentile(finite_distances, 50),
            "p95": percentile(finite_distances, 95),
            "max": max(finite_distances) if finite_distances else None,
        },
        "inside_finite_patch_rows": inside_rows,
        "within_unchanged_5mm_rows": within_gate_rows,
        "fixed_distance_m": EXPECTED_DISTANCE_M,
    }, object_map)


def load_packed_mask(path: Path, expected_frames: int) -> Dict[str, Any]:
    try:
        with np.load(str(path), allow_pickle=False) as archive:
            packed = np.asarray(archive["packed"])
            height = int(archive["height"])
            width = int(archive["width"])
            frame_count = int(archive["frame_count"])
            bitorder = str(archive["bitorder"])
    except Exception as exc:
        fail("STOP_INPUT_DRIFT", "cannot read packed semantic mask {}: {}".format(path, exc))
    expected_shape = (expected_frames, (height * width + 7) // 8)
    if (
        (height, width) != EXPECTED_MASK_HW
        or frame_count != expected_frames
        or bitorder != "big"
        or packed.shape != expected_shape
        or packed.dtype != np.uint8
    ):
        fail("STOP_INPUT_DRIFT", "packed semantic-mask contract drift: {}".format(path))
    return {"packed": packed, "height": height, "width": width}


def packed_values(mask: Mapping[str, Any], frame: int, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    width = int(mask["width"])
    flat = y.astype(np.int64) * width + x.astype(np.int64)
    byte = mask["packed"][frame, flat // 8]
    shift = 7 - (flat % 8)
    return ((byte >> shift) & 1).astype(bool)


def unique_row(rows: Iterable[Mapping[str, Any]], key: Tuple[int, str, str, str], label: str) -> Mapping[str, Any]:
    frame, hand, finger, object_id = key
    found = [
        row for row in rows
        if int(row.get("frame_id", -1)) == frame
        and row.get("hand_id") == hand
        and row.get("finger_id") == finger
        and row.get("object_id") == object_id
    ]
    if len(found) != 1:
        fail("STOP_SEMANTIC_OR_IDENTITY_DRIFT", "{} row multiplicity drift for {}".format(label, key))
    return found[0]


def replay_local_admission(
    source_uv: np.ndarray,
    depth: Mapping[str, np.ndarray],
    hand_mask: Mapping[str, Any],
    object_masks: Sequence[Mapping[str, Any]],
    frame: int,
) -> Dict[str, Any]:
    depth_uv = (source_uv - 0.5) / 2.0
    x0, y0 = np.rint(depth_uv).astype(int)
    local_y, local_x = np.mgrid[
        max(0, y0 - EXPECTED_RADIUS_DEPTH_PX):min(EXPECTED_DEPTH_HW[0], y0 + EXPECTED_RADIUS_DEPTH_PX + 1),
        max(0, x0 - EXPECTED_RADIUS_DEPTH_PX):min(EXPECTED_DEPTH_HW[1], x0 + EXPECTED_RADIUS_DEPTH_PX + 1),
    ]
    roi = (local_x - x0) ** 2 + (local_y - y0) ** 2 <= EXPECTED_RADIUS_DEPTH_PX ** 2
    ys, xs = local_y[roi], local_x[roi]
    sx = np.clip(np.rint(2.0 * xs + 0.5).astype(int), 0, EXPECTED_MASK_HW[1] - 1)
    sy = np.clip(np.rint(2.0 * ys + 0.5).astype(int), 0, EXPECTED_MASK_HW[0] - 1)
    hand = packed_values(hand_mask, frame, sy, sx)
    objects = np.zeros(len(xs), bool)
    for mask in object_masks:
        objects |= packed_values(mask, frame, sy, sx)
    depth_m = np.asarray(depth["depth_m"])
    admitted = (
        np.asarray(depth["valid"])[ys, xs]
        & np.asarray(depth["lr_consistent"])[ys, xs]
        & np.isfinite(depth_m[ys, xs])
        & (depth_m[ys, xs] > 0)
        & hand
        & ~objects
    )
    return {
        "roi_pixel_count": int(len(xs)),
        "admitted_pixel_count": int(np.count_nonzero(admitted)),
        "object_mask_fraction": float(np.mean(objects)),
        "hand_mask_purity": float(np.mean(hand)),
        "qualifies_independent_surface": bool(
            int(np.count_nonzero(admitted)) >= 6
            and float(np.mean(objects)) <= 0.35
            and float(np.mean(hand)) >= 0.45
        ),
    }


def find_mask_path(index: Mapping[str, Any], inputs: Inputs, basename: str, group: str) -> Path:
    matches = [row for row in index.get(group, []) if Path(str(row.get("path"))).name == basename]
    if len(matches) != 1:
        fail("STOP_INPUT_DRIFT", "mask index multiplicity drift: {}".format(basename))
    return inputs.path(str(matches[0]["path"]))


def audit_target_and_controls(
    inputs: Inputs,
    index: Mapping[str, Any],
    depth_rows: Sequence[Mapping[str, Any]],
    interaction: Mapping[str, Any],
    contact: Mapping[str, Any],
    hypotheses: Mapping[str, Any],
    raw_hawor_path: Path,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    frame_count = len(depth_rows)
    object_masks = [
        load_packed_mask(find_mask_path(index, inputs, "playing_card_{:02d}.npz".format(i), "object_mask_inputs"), frame_count)
        for i in range(3)
    ]
    target_object_mask = object_masks[2]
    right_hand_mask = load_packed_mask(find_mask_path(index, inputs, "right_hand.npz", "hand_mask_inputs"), frame_count)
    try:
        with np.load(str(raw_hawor_path), allow_pickle=False) as archive:
            joints_2d = np.asarray(archive["joints_2d"])
            observed = np.asarray(archive["observed"])
            provenance = np.asarray(archive["provenance"])
            side_names = [str(v) for v in archive["anatomical_side_names"].tolist()]
            joint_names = [str(v) for v in archive["mano_joint_names"].tolist()]
            fps = float(archive["fps"])
    except Exception as exc:
        fail("STOP_INPUT_DRIFT", "cannot read HaWoR observation: {}".format(exc))
    try:
        side_index = side_names.index("right")
        joint_index = joint_names.index("index_tip")
    except ValueError:
        fail("STOP_SEMANTIC_OR_IDENTITY_DRIFT", "HaWoR right/index identity missing")

    interaction_rows = interaction.get("rows", [])
    contact_rows = contact.get("rows", [])
    target_hypotheses = hypotheses.get("hypotheses", [])
    controls = hypotheses.get("no_contact_controls", [])
    if [int(row.get("frame_id", -1)) for row in target_hypotheses] != list(EXPECTED_TARGET_FRAMES):
        fail("STOP_SEMANTIC_OR_IDENTITY_DRIFT", "92-98 hypothesis window drift")
    if [int(row.get("frame_id", -1)) for row in controls] != list(EXPECTED_CONTROL_FRAMES):
        fail("STOP_SEMANTIC_OR_IDENTITY_DRIFT", "negative-control frame axis drift")
    if hypotheses.get("allowed_pair") != {
        "hand_id": "right", "finger_id": "index", "object_id": "playing_card_02", "frames_inclusive": [92, 98]
    }:
        fail("STOP_SEMANTIC_OR_IDENTITY_DRIFT", "frozen candidate identity drift")

    output_rows: List[Dict[str, Any]] = []
    target_visible_surface_count = 0
    target_mask_overlap_count = 0
    target_tactile_count = 0
    for kind, documents, frames in (
        ("target", target_hypotheses, EXPECTED_TARGET_FRAMES),
        ("negative_diagnostic_control", controls, EXPECTED_CONTROL_FRAMES),
    ):
        for doc, frame in zip(documents, frames):
            key = (frame,) + EXPECTED_TARGET
            interaction_row = unique_row(interaction_rows, key, "interaction")
            contact_row = unique_row(contact_rows, key, "contact")
            if any(doc.get(name) != value for name, value in zip(("hand_id", "finger_id", "object_id"), EXPECTED_TARGET)):
                fail("STOP_SEMANTIC_OR_IDENTITY_DRIFT", "target/control identity drift")
            sample = interaction_row.get("finger_associated_visible_surface_point", {})
            source_uv = np.asarray(sample.get("source_pixel_uv"), np.float64)
            raw_uv = np.asarray(joints_2d[side_index, frame, joint_index], np.float64)
            if (
                source_uv.shape != (2,)
                or not np.allclose(source_uv, raw_uv, atol=1e-6, rtol=0)
                or not bool(observed[side_index, frame])
                or str(provenance[side_index, frame]) != "OBSERVED"
            ):
                fail("STOP_SEMANTIC_OR_IDENTITY_DRIFT", "HaWoR direct 2D observation drift at {}".format(frame))
            expected_timestamp = frame / fps
            timestamps = [interaction_row.get("timestamp_s"), contact_row.get("timestamp_s"), doc.get("timestamp_s")]
            if any(not isinstance(v, (int, float)) or abs(float(v) - expected_timestamp) > 1e-12 for v in timestamps):
                fail("STOP_SEMANTIC_OR_IDENTITY_DRIFT", "timestamp drift at frame {}".format(frame))
            rounded = np.rint(source_uv).astype(int)
            if not (0 <= rounded[0] < EXPECTED_MASK_HW[1] and 0 <= rounded[1] < EXPECTED_MASK_HW[0]):
                fail("STOP_SEMANTIC_OR_IDENTITY_DRIFT", "target/control pixel out of bounds")
            exact_object_bit = bool(packed_values(
                target_object_mask, frame,
                np.asarray([rounded[1]]), np.asarray([rounded[0]])
            )[0])
            visual = doc.get("visual_evidence", {})
            published_overlap = bool(visual.get("projected_finger_inside_visible_object_mask"))
            row_overlap = bool(interaction_row.get("two_d_adjacency", {}).get("projected_finger_inside_object_mask"))
            if exact_object_bit != published_overlap or row_overlap != published_overlap:
                fail("STOP_SEMANTIC_OR_IDENTITY_DRIFT", "packed object-overlap replay drift at {}".format(frame))
            depth = load_depth(inputs, depth_rows[frame], frame)
            replay = replay_local_admission(source_uv, depth, right_hand_mask, object_masks, frame)
            quality = sample.get("association_quality", {})
            for field in ("roi_pixel_count", "admitted_pixel_count"):
                if replay[field] != quality.get(field):
                    fail("STOP_SEMANTIC_OR_IDENTITY_DRIFT", "local admission count drift at {}".format(frame))
            for field in ("object_mask_fraction", "hand_mask_purity"):
                if not isinstance(quality.get(field), (int, float)) or abs(replay[field] - float(quality[field])) > 1e-12:
                    fail("STOP_SEMANTIC_OR_IDENTITY_DRIFT", "local admission fraction drift at {}".format(frame))
            visible_surface = sample.get("status") == "OBSERVED_VISIBLE_SURFACE"
            if visible_surface != replay["qualifies_independent_surface"]:
                fail("STOP_SEMANTIC_OR_IDENTITY_DRIFT", "independent surface admission replay drift at {}".format(frame))
            tactile = contact_row.get("tactile", {})
            tactile_active = bool(tactile.get("finger_specific_tactile_active"))
            if tactile != doc.get("tactile_evidence"):
                fail("STOP_SEMANTIC_OR_IDENTITY_DRIFT", "tactile evidence binding drift at {}".format(frame))
            if contact_row.get("strict_contact_admitted") is not False or contact_row.get("r1_e_admitted") is not False:
                fail("STOP_SEMANTIC_OR_IDENTITY_DRIFT", "contact was improperly admitted at {}".format(frame))
            if kind == "target":
                target_mask_overlap_count += int(exact_object_bit)
                target_tactile_count += int(tactile_active)
                target_visible_surface_count += int(visible_surface)
                if (
                    not exact_object_bit
                    or not tactile_active
                    or visible_surface
                    or sample.get("reason") != "LOCAL_HAND_ROLE_DEPTH_OR_OBJECT_GATE_FAILED"
                    or "surface_point_xyz" in sample
                    or interaction_row.get("metric") is not None
                    or contact_row.get("finite_patch_distance_m") is not None
                ):
                    fail("STOP_NO_DIRECT_FINGER_SURFACE_IN_CONTACT_WINDOW", "92-98 fail-closed premise drift at {}".format(frame))
            else:
                if (
                    visual.get("adjacent_within_20px") is not False
                    or exact_object_bit
                    or tactile_active
                    or doc.get("contact_ground_truth") is not False
                ):
                    fail("STOP_SEMANTIC_OR_IDENTITY_DRIFT", "negative diagnostic control predicate drift")
            output_rows.append({
                "kind": kind,
                "frame_id": frame,
                "timestamp_s": expected_timestamp,
                "packed_card02_overlap": exact_object_bit,
                "adjacent_within_20px": bool(visual.get("adjacent_within_20px", exact_object_bit)),
                "tactile_active": tactile_active,
                "independent_visible_finger_surface": visible_surface,
                "sample_status": sample.get("status"),
                "sample_reason": sample.get("reason"),
                "roi_pixel_count": replay["roi_pixel_count"],
                "admitted_pixel_count": replay["admitted_pixel_count"],
                "object_mask_fraction": replay["object_mask_fraction"],
                "hand_mask_purity": replay["hand_mask_purity"],
                "strict_contact_admitted": False,
                "contact_ground_truth": False,
            })
    if target_visible_surface_count != 0:
        fail("STOP_NO_DIRECT_FINGER_SURFACE_IN_CONTACT_WINDOW", "target window unexpectedly has finger surface")
    return ({
        "status": "PASS_FAIL_CLOSED",
        "candidate_identity": ":".join(EXPECTED_TARGET),
        "target_frames": list(EXPECTED_TARGET_FRAMES),
        "target_frame_count": len(EXPECTED_TARGET_FRAMES),
        "target_packed_object_overlap_frames": target_mask_overlap_count,
        "target_aligned_finger_tactile_frames": target_tactile_count,
        "target_independent_visible_finger_surface_frames": target_visible_surface_count,
        "target_metric_rows": 0,
        "negative_control_frames": list(EXPECTED_CONTROL_FRAMES),
        "negative_controls_are_ground_truth": False,
        "timestamp_policy": "UNCHANGED_FRAME_INDEX_DIVIDED_BY_FROZEN_FPS",
        "sampler_policy": {
            "depth_radius_px": EXPECTED_RADIUS_DEPTH_PX,
            "minimum_admitted_pixels": 6,
            "maximum_object_fraction": 0.35,
            "minimum_hand_purity": 0.45,
            "changed": False,
        },
    }, output_rows)


def audit_authority(
    object_document: Mapping[str, Any],
    depth_contract: Mapping[str, Any],
    adapter_contract: Mapping[str, Any],
    rgb_qa: Mapping[str, Any],
    contact: Mapping[str, Any],
    target_audit: Mapping[str, Any],
) -> Dict[str, Any]:
    rows = contact.get("rows", [])
    registration_bound = [
        row for row in rows
        if row.get("uncertainty", {}).get("pixel_registration_uncertainty") != "UNBOUND"
    ]
    predicates = {
        "fixed_5mm_gate_unchanged": contact.get("fixed_geometry_distance_m") == EXPECTED_DISTANCE_M,
        "uncertainty_does_not_expand_gate": contact.get("uncertainty_never_expands_distance_gate") is True,
        "object6d_contact_authority_none": object_document.get("contact_authority") == "NONE",
        "object6d_external_accuracy_unverified": object_document.get("external_accuracy") == "UNVERIFIED",
        "depth_external_accuracy_unverified": depth_contract.get("external_accuracy") == "UNVERIFIED",
        "depth_encoded_projection_matrices_absent": depth_contract.get("encoded_projection_matrices") == "ABSENT",
        "adapter_external_projection_authority_absent": adapter_contract.get("encoded_projection_matrices") == "ABSENT_EXTERNAL_ACCURACY_UNVERIFIED",
        "rgb_qa_is_internal_encoded_roundtrip_only": rgb_qa.get("comparison") == "resize(physical_left)==unflip(resize(flip(physical_left)))",
        "all_contact_rows_registration_unbound": len(registration_bound) == 0,
        "strict_metric_contact_authorized_false": contact.get("strict_metric_contact_authorized") is False,
        "r1_e_authorized_false": contact.get("r1_e_authorized") is False,
        "target_has_no_independent_finger_surface": target_audit.get("target_independent_visible_finger_surface_frames") == 0,
    }
    if not all(predicates.values()):
        fail("STOP_CONTACT_SUCCESSOR_REGISTRATION_UNBOUND", "fail-closed authority contract drift")
    return {
        "status": "EXPECTED_FAIL_CLOSED_TERMINAL",
        "predicates": predicates,
        "independent_physical_registration_bound": False,
        "registration_authority": "UNBOUND",
        "contact": "UNKNOWN",
        "contact_authority": "NONE",
        "r1_e": "CLOSED",
        "training_eligible": False,
        "control_ground_truth": False,
        "fixed_distance_m": EXPECTED_DISTANCE_M,
        "distance_gate_changed": False,
        "external_metric_created": False,
    }


def run(index_path: Path, design_path: Path, repo_root: Optional[Path] = None) -> Dict[str, Any]:
    index = load_json(index_path)
    design = load_json(design_path)
    frozen = verify_design(design)
    inputs = Inputs(index, repo_root)
    verification, depth_rows, resolved = verify_all_inputs(inputs)
    package_decisions = verify_package_decisions(
        inputs, index_path, design_path, index_path.with_name("RESULT.json")
    )
    interaction = load_json(inputs.role_path("interaction_pair_rows"))
    contact = load_json(inputs.role_path("contact_pair_rows"))
    hypotheses = load_json(inputs.role_path("target_and_negative_control_ledger"))
    object_document = load_json(inputs.role_path("object6d_finite_patch_geometry"))
    depth_contract = load_json(inputs.role_path("depth_contract"))
    adapter_contract = load_json(inputs.role_path("encoded_domain_adapter_contract"))
    rgb_qa = load_json(inputs.role_path("rgb_alignment_claim"))
    documents = (interaction, contact, hypotheses, object_document, depth_contract)
    if any(doc.get("session_id") != EXPECTED_SESSION for doc in documents):
        fail("STOP_SEMANTIC_OR_IDENTITY_DRIFT", "session identity drift")
    if len(interaction.get("rows", [])) != EXPECTED_PAIR_ROWS or len(contact.get("rows", [])) != EXPECTED_PAIR_ROWS:
        fail("STOP_SEMANTIC_OR_IDENTITY_DRIFT", "fixed pair denominator drift")
    if interaction.get("pair_denominator") != EXPECTED_PAIR_ROWS:
        fail("STOP_SEMANTIC_OR_IDENTITY_DRIFT", "interaction denominator declaration drift")

    roundtrip, _object_map = audit_roundtrip_and_support(
        inputs,
        depth_rows,
        object_document,
        inputs.role_path("object6d_exact_visible_support"),
        interaction,
    )
    target, detail_rows = audit_target_and_controls(
        inputs,
        index,
        depth_rows,
        interaction,
        contact,
        hypotheses,
        inputs.role_path("hawor_direct_2d_observation"),
    )
    authority = audit_authority(
        object_document, depth_contract, adapter_contract, rgb_qa, contact, target
    )
    return {
        "schema_version": "0915-robot-recovery-v21-c1-cpu-internal-selfcheck-result-v1",
        "package_id": "C1_CPU_INTERNAL_SELFCHECK",
        "status": "PASS_INTERNAL_SELF_CHECK_FAIL_CLOSED_CONTACT_UNKNOWN",
        "decision": "INTERNAL_COORDINATE_SELF_CHECK_PASS_NO_GO_CONTACT_SUCCESSOR",
        "session_id": EXPECTED_SESSION,
        "cpu_only": True,
        "gpu_used": False,
        "model_inference_used": False,
        "source_mutated": False,
        "contact_timestamp_changed": False,
        "threshold_changed": False,
        "external_metric_or_registration_created": False,
        "frozen_constants": frozen,
        "input_verification": verification,
        "package_decision_verification": package_decisions,
        "packed_depth_and_finite_patch_roundtrip": roundtrip,
        "target_and_negative_control_audit": target,
        "target_control_rows": detail_rows,
        "registration_and_contact_authority": authority,
        "stop_terminals_retained": [
            "STOP_CONTACT_SUCCESSOR_REGISTRATION_UNBOUND",
            "STOP_NO_DIRECT_FINGER_SURFACE_IN_CONTACT_WINDOW",
        ],
        "training_eligible": False,
        "control_ground_truth": False,
        "contact_ground_truth": False,
        "claim_limit": (
            "CPU-only immutable-cache implementation self-check. Packed-depth and encoded-domain "
            "round trips are internal consistency checks, not external metric/registration accuracy "
            "or physical Contact authority."
        ),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-index", type=Path, required=True)
    parser.add_argument("--design", type=Path)
    parser.add_argument(
        "--repo-root",
        type=Path,
        help="Optional local replacement for the canonical repo_root (used by fixtures).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write result JSON here. Omit to emit JSON on stdout.",
    )
    args = parser.parse_args(argv)
    design = args.design or args.input_index.with_name("C1_MINIMAL_DIAGNOSTIC_DESIGN_V1.json")
    try:
        result = run(args.input_index, design, args.repo_root)
        encoded = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output is None:
            sys.stdout.write(encoded)
        else:
            args.output.write_text(encoded, encoding="utf-8")
        return 0
    except SelfCheckError as exc:
        sys.stdout.write(json.dumps({
            "schema_version": "0915-robot-recovery-v21-c1-cpu-internal-selfcheck-error-v1",
            "status": "FAIL_CLOSED",
            "stop_code": exc.code,
            "message": str(exc),
            "contact": "UNKNOWN",
            "contact_authority": "NONE",
        }, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
