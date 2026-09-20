#!/usr/bin/env python3
"""CPU-only, fail-closed audit of the frozen 0915 AI2 real assets.

The producer accepts only SHA-bound artifacts from the current W0, A1, or A2
roots.  It does not run HaWoR, SAM, or any GPU code.  Missing independent
part-observability, independent reprojection, or full-versus-truncated current
inputs remain explicit blockers rather than being synthesized from HaWoR-
anchored SAM output.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
from typing import Any
import uuid

import numpy as np

from chaoyang.pipeline.hand_observability_v1 import (
    HAWOR_SENTINEL,
    PARTS,
    audit_provider_independence,
    mano21_part_presence,
)
from chaoyang.pipeline.kai22_r0_tiered_admission_v1 import evaluate_kai22_r0_tiers
from chaoyang.pipeline.robot_renderer_cycles import forward_kinematics
from chaoyang.pipeline.robot_renderer_eevee_fullchain import load_pinned_robot_assets
from chaoyang.pipeline.temporal_authority_v1 import audit_suffix_invariance


ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "AI2_REAL_ASSETS_AUDIT_V31"
COHORT_ROOTS = {
    "W0": {
        "hawor": "_run/current/0915_robot15h_hawor_wave0_recovery_v1/attempts/attempt_0001",
        "kai22": "_run/current/0915_robot15h_kai22_r0_wave0_v1/attempts/attempt_0001",
        "sam": "_run/current/0915_robot15h_sam31_temporal_identity_recovery_v1/attempts/attempt_0001",
    },
    "A1": {
        "hawor": "_run/current/0915_robot_quality_recovery_15h_v2/attempts/attempt_0001/packages/A1",
        "kai22": "_run/current/0915_robot_quality_recovery_15h_v2/attempts/attempt_0001/packages/A3_R0/A1",
    },
    "A2": {
        "hawor": "_run/current/0915_robot_quality_recovery_15h_v2/attempts/attempt_0001/packages/A2",
        "kai22": "_run/current/0915_robot_quality_recovery_15h_v2/attempts/attempt_0001/packages/A3_R0/A2",
    },
}
TEMPORAL_REQUIRED_FIELDS = (
    "human_state",
    "robotized_rgb",
    "crop",
    "confidence",
)


class RealAssetAuditError(RuntimeError):
    """Raised before publication when path or SHA binding is not exact."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_ref(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RealAssetAuditError(f"JSON object required: {path}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _assert_current_scope(path: Path, cohort: str, role: str, *, root: Path) -> Path:
    if cohort not in COHORT_ROOTS or role not in COHORT_ROOTS[cohort]:
        raise RealAssetAuditError(f"unsupported cohort/role: {cohort}/{role}")
    resolved = path.resolve(strict=True)
    allowed = (root / COHORT_ROOTS[cohort][role]).resolve(strict=True)
    if resolved != allowed and allowed not in resolved.parents:
        raise RealAssetAuditError(
            f"{role} asset is outside frozen {cohort} current root: {resolved}"
        )
    if "archive" in resolved.parts:
        raise RealAssetAuditError("archive assets are forbidden")
    return resolved


def _iter_refs(value: Any) -> list[Mapping[str, Any]]:
    refs: list[Mapping[str, Any]] = []
    if isinstance(value, Mapping):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            refs.append(value)
        for child in value.values():
            refs.extend(_iter_refs(child))
    elif isinstance(value, list):
        for child in value:
            refs.extend(_iter_refs(child))
    return refs


def _require_json_binding(document: Mapping[str, Any], artifact: Path, label: str) -> None:
    reference = artifact_ref(artifact)
    matches = [
        row
        for row in _iter_refs(document)
        if Path(str(row["path"])).resolve() == artifact.resolve()
    ]
    if not matches:
        raise RealAssetAuditError(f"{label} JSON does not bind artifact path")
    if not any(row.get("sha256") == reference["sha256"] for row in matches):
        raise RealAssetAuditError(f"{label} JSON artifact SHA mismatch")


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _array_shapes(arrays: Mapping[str, np.ndarray]) -> dict[str, list[int]]:
    return {key: list(value.shape) for key, value in sorted(arrays.items())}


def _hawor_shape_audit(arrays: Mapping[str, np.ndarray], session_id: str) -> tuple[int, dict[str, Any]]:
    required = {
        "joints_3d_camera",
        "joints_2d",
        "observed",
        "original_frame_indices",
        "anatomical_side_names",
        "mano_joint_names",
    }
    missing = sorted(required - set(arrays))
    if missing:
        raise RealAssetAuditError(f"HaWoR NPZ missing arrays: {missing}")
    joints = np.asarray(arrays["joints_3d_camera"])
    if joints.ndim != 4 or joints.shape[0] != 2 or joints.shape[2:] != (21, 3):
        raise RealAssetAuditError("HaWoR joints_3d_camera must have shape [2,T,21,3]")
    frame_count = int(joints.shape[1])
    expected_shapes = {
        "joints_2d": (2, frame_count, 21, 2),
        "observed": (2, frame_count),
        "original_frame_indices": (frame_count,),
    }
    errors = [
        f"{key}:{arrays[key].shape}!={shape}"
        for key, shape in expected_shapes.items()
        if arrays[key].shape != shape
    ]
    if errors:
        raise RealAssetAuditError("; ".join(errors))
    if not np.array_equal(arrays["original_frame_indices"], np.arange(frame_count)):
        raise RealAssetAuditError("HaWoR frame axis is not exact 0..T-1")
    if tuple(str(value) for value in arrays["anatomical_side_names"].tolist()) != (
        "left",
        "right",
    ):
        raise RealAssetAuditError("HaWoR anatomical side axis mismatch")
    return frame_count, {
        "session_id": session_id,
        "frame_count": frame_count,
        "arrays": _array_shapes(arrays),
        "status": "PASS_HAWOR_STRUCTURE",
    }


def _part_presence(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    joints = np.asarray(arrays["joints_3d_camera"], np.float64)
    observed = np.asarray(arrays["observed"], bool)
    joint_valid = observed[..., None] & np.isfinite(joints).all(axis=-1)
    presence = mano21_part_presence(np.transpose(joint_valid, (1, 0, 2)))
    counts = {
        side: {
            part: int(presence[:, side_index, part_index].sum())
            for part_index, part in enumerate(PARTS)
        }
        for side_index, side in enumerate(("left", "right"))
    }
    return {
        "status": "BLOCKED_INDEPENDENT_PART_OBSERVABILITY",
        "structural_hawor_part_presence_counts": counts,
        "independent_observation_rows": 0,
        "pose_quality_rows": 0,
        "denominator_published": False,
        "reason_codes": [
            "INDEPENDENT_FRAME_SIDE_PART_VISIBILITY_ABSENT",
            "INDEPENDENT_PART_POSE_QUALITY_ABSENT",
        ],
        "claim_limit": (
            "HaWoR structural output presence only; it is not an independent "
            "visibility denominator or pose-quality label."
        ),
    }


def _sam_dependency_audit(
    sam_result: Mapping[str, Any] | None, hawor_npz: Path
) -> dict[str, Any]:
    providers: list[dict[str, Any]] = [
        {
            "provider_id": "hawor_raw_output",
            "family": "HAWOR",
            "dependencies": [HAWOR_SENTINEL],
            "independent_from_hawor": False,
        }
    ]
    missing_refs: list[str] = []
    sam_input_bound = False
    if sam_result is not None:
        sam_refs = _iter_refs(sam_result)
        raw_ref = artifact_ref(hawor_npz)
        sam_input_bound = any(
            Path(str(row["path"])).resolve() == hawor_npz.resolve()
            and row.get("sha256") == raw_ref["sha256"]
            for row in sam_refs
        )
        if not sam_input_bound:
            raise RealAssetAuditError("SAM result is not bound to supplied HaWoR NPZ")
        providers.append(
            {
                "provider_id": "sam31_hawor_anchored_hand",
                "family": "SAM31_HAND_TEMPORAL",
                "dependencies": ["hawor_raw_output"],
                "independent_from_hawor": False,
            }
        )
        for row in sam_refs:
            candidate = Path(str(row["path"]))
            if ".staging-" in str(candidate) and not candidate.exists():
                missing_refs.append(str(candidate))
    audit = audit_provider_independence(providers)
    return {
        "status": "PASS_DEPENDENCY_AUDIT_FAIL_CLOSED",
        "providers": audit,
        "sam_input_bound_to_hawor": sam_input_bound,
        "missing_sam_staging_references": sorted(set(missing_refs)),
        "sam_admitted_as_hawor_observability_judge": False,
    }


def _bounded_audit(
    raw: Mapping[str, np.ndarray], candidate: Mapping[str, np.ndarray] | None
) -> dict[str, Any]:
    blockers = [
        "BLOCKED_INDEPENDENT_OBSERVABLE_SET",
        "BLOCKED_INDEPENDENT_REPROJECTION",
    ]
    if candidate is None:
        return {
            "status": "BLOCKED_BOUNDED_CANDIDATE_ABSENT",
            "reason_codes": ["BLOCKED_BOUNDED_CANDIDATE_ABSENT", *blockers],
            "adoption_authorized": False,
        }
    for key in ("joints_3d_camera", "joints_2d", "observed", "original_frame_indices"):
        if key not in candidate or candidate[key].shape != raw[key].shape:
            raise RealAssetAuditError(f"bounded candidate {key} shape mismatch")
    if not np.array_equal(candidate["observed"], raw["observed"]):
        raise RealAssetAuditError("bounded candidate changed observed mask")
    if not np.array_equal(candidate["original_frame_indices"], raw["original_frame_indices"]):
        raise RealAssetAuditError("bounded candidate changed frame axis")
    selected = np.asarray(raw["observed"], bool)
    camera_delta = np.linalg.norm(
        np.asarray(candidate["joints_3d_camera"], np.float64)
        - np.asarray(raw["joints_3d_camera"], np.float64),
        axis=-1,
    ) * 1000.0
    pixel_delta = np.linalg.norm(
        np.asarray(candidate["joints_2d"], np.float64)
        - np.asarray(raw["joints_2d"], np.float64),
        axis=-1,
    )
    per_side: dict[str, Any] = {}
    for side_index, side in enumerate(("left", "right")):
        side_selected = selected[side_index]
        camera = camera_delta[side_index, side_selected]
        pixel = pixel_delta[side_index, side_selected]
        camera = camera[np.isfinite(camera)]
        pixel = pixel[np.isfinite(pixel)]
        per_side[side] = {
            "raw_observed_frames": int(side_selected.sum()),
            "candidate_observed_frames": int(np.asarray(candidate["observed"])[side_index].sum()),
            "camera_correction_p95_mm": float(np.percentile(camera, 95)) if camera.size else None,
            "camera_correction_max_mm": float(camera.max()) if camera.size else None,
            "candidate_to_raw_projection_delta_p95_px": (
                float(np.percentile(pixel, 95)) if pixel.size else None
            ),
        }
    return {
        "status": "BLOCKED_INDEPENDENT_QUALITY_EVIDENCE",
        "reason_codes": blockers,
        "structural_pair_check": "PASS_SAME_FRAME_AND_OBSERVED_AXES",
        "per_side_diagnostic": per_side,
        "candidate_to_raw_projection_is_independent_quality": False,
        "adoption_authorized": False,
    }


def _joint_semantics_pass(result: Mapping[str, Any], assets: Any) -> bool:
    order = result.get("joint_order", result.get("q22_joint_order"))
    if not isinstance(order, Mapping):
        return False
    expected = {}
    for side, model in (("left", assets.left_hand), ("right", assets.right_hand)):
        expected[side] = [joint.name for joint in model.joints if joint.joint_type != "fixed"]
    mapping = result.get("side_contract", {}).get("human_to_physical")
    return (
        order.get("left") == expected["left"]
        and order.get("right") == expected["right"]
        and mapping == {"left": "right", "right": "left"}
        and result.get("q22_units") == "radians"
    )


def _w0_fk_valid(q22: np.ndarray, valid: np.ndarray, assets: Any) -> np.ndarray:
    output = np.zeros(valid.shape, dtype=bool)
    for side, model in enumerate((assets.left_hand, assets.right_hand)):
        moving = tuple(joint for joint in model.joints if joint.joint_type != "fixed")
        if len(moving) != q22.shape[2]:
            raise RealAssetAuditError("Kai22 joint count differs from pinned URDF")
        for frame in np.flatnonzero(valid[:, side]):
            if not np.isfinite(q22[frame, side]).all():
                continue
            transforms = forward_kinematics(
                model,
                {
                    joint.name: float(q22[frame, side, index])
                    for index, joint in enumerate(moving)
                },
            )
            output[frame, side] = all(np.isfinite(value).all() for value in transforms.values())
    return output


def _kai22_tier(
    arrays: Mapping[str, np.ndarray], result: Mapping[str, Any], temporal: Mapping[str, Any], root: Path
) -> dict[str, Any]:
    assets = load_pinned_robot_assets(root)
    if "q22" in arrays:
        q22 = np.asarray(arrays["q22"], np.float64)
        q_valid = np.asarray(arrays["q22_computed"], bool)
        fk_valid = np.asarray(arrays["fk_pass"], bool)
        local = np.asarray(arrays["r0_static_gate_pass"], bool)
    elif "q22_init" in arrays:
        q22 = np.asarray(arrays["q22_init"], np.float64)
        q_valid = np.asarray(arrays["valid_side_frame"], bool)
        fk_valid = _w0_fk_valid(q22, q_valid, assets)
        local = q_valid.copy()
    else:
        raise RealAssetAuditError("Kai22 NPZ lacks q22 or q22_init")
    if q22.ndim != 3 or q22.shape[1:] != (2, 22):
        raise RealAssetAuditError("Kai22 q array must have shape [T,2,22]")
    frame_ids = np.asarray(arrays["frame_id"], np.int64)
    if q_valid.shape != (len(frame_ids), 2) or fk_valid.shape != q_valid.shape:
        raise RealAssetAuditError("Kai22 validity axes mismatch")
    side_ledgers = {}
    for side_index, side in enumerate(("left", "right")):
        side_ledgers[side] = evaluate_kai22_r0_tiers(
            frame_ids=frame_ids,
            q22_valid=q_valid[:, side_index],
            fk_finite=fk_valid[:, side_index],
            joint_semantics_pass=_joint_semantics_pass(result, assets),
            consumer_local_window_mask=local[:, side_index],
            consumer_local_window_quality_pass=False,
            temporal_authority_audit=temporal,
            h50_current_input_fields=TEMPORAL_REQUIRED_FIELDS,
            timestamps_valid=bool(
                "timestamps_s" in arrays
                and np.asarray(arrays["timestamps_s"]).shape == frame_ids.shape
                and np.all(np.diff(np.asarray(arrays["timestamps_s"], np.float64)) > 0)
            ),
        )
    return {
        "status": "COMPLETED_FAIL_CLOSED_TIER_ACCOUNTING",
        "sides": side_ledgers,
        "maximum_possible_level_in_this_audit": "KINEMATIC_ONLY",
        "development_r0_quality_input": "ABSENT_NOT_INFERRED_FROM_STRUCTURAL_WINDOWS",
        "h50_temporal_authority_input": "BLOCKED_SUFFIX_PAIR_MATERIALIZATION",
    }


def build_audit(
    *,
    cohort: str,
    session_id: str,
    hawor_npz: Path,
    hawor_result: Path,
    kai22_npz: Path,
    kai22_result: Path,
    bounded_npz: Path | None = None,
    bounded_result: Path | None = None,
    sam_result: Path | None = None,
    root: Path = ROOT,
) -> dict[str, Any]:
    if cohort not in COHORT_ROOTS:
        raise RealAssetAuditError(f"unsupported cohort: {cohort}")
    if (bounded_npz is None) != (bounded_result is None):
        raise RealAssetAuditError("bounded NPZ and RESULT must be supplied together")
    hawor_npz = _assert_current_scope(hawor_npz, cohort, "hawor", root=root)
    hawor_result = _assert_current_scope(hawor_result, cohort, "hawor", root=root)
    kai22_npz = _assert_current_scope(kai22_npz, cohort, "kai22", root=root)
    kai22_result = _assert_current_scope(kai22_result, cohort, "kai22", root=root)
    if bounded_npz is not None and bounded_result is not None:
        if cohort != "W0":
            raise RealAssetAuditError("only frozen W0 currently has a bound bounded successor")
        bounded_npz = _assert_current_scope(bounded_npz, cohort, "kai22", root=root)
        bounded_result = _assert_current_scope(bounded_result, cohort, "kai22", root=root)
    if sam_result is not None:
        if cohort != "W0":
            raise RealAssetAuditError("only W0 SAM result is accepted by this audit")
        sam_result = _assert_current_scope(sam_result, cohort, "sam", root=root)

    hawor_document = load_json(hawor_result)
    kai22_document = load_json(kai22_result)
    if hawor_document.get("session_id") != session_id or kai22_document.get("session_id") != session_id:
        raise RealAssetAuditError("session_id differs from bound result")
    _require_json_binding(hawor_document, hawor_npz, "HaWoR")
    _require_json_binding(kai22_document, kai22_npz, "Kai22")
    raw = _load_npz(hawor_npz)
    frame_count, shape_audit = _hawor_shape_audit(raw, session_id)
    if int(hawor_document.get("frame_count", -1)) != frame_count:
        raise RealAssetAuditError("HaWoR result frame_count mismatch")

    candidate = None
    bounded_document = None
    if bounded_npz is not None and bounded_result is not None:
        bounded_document = load_json(bounded_result)
        if bounded_document.get("session_id") != session_id:
            raise RealAssetAuditError("bounded session_id differs")
        _require_json_binding(bounded_document, bounded_npz, "bounded")
        _require_json_binding(bounded_document, hawor_npz, "bounded raw input")
        candidate = _load_npz(bounded_npz)

    sam_document = load_json(sam_result) if sam_result is not None else None
    if sam_document is not None and sam_document.get("session_id") != session_id:
        raise RealAssetAuditError("SAM session_id differs")

    kai22_arrays = _load_npz(kai22_npz)
    if "frame_id" not in kai22_arrays:
        raise RealAssetAuditError("Kai22 NPZ lacks frame_id")
    if not np.array_equal(kai22_arrays["frame_id"], raw["original_frame_indices"]):
        raise RealAssetAuditError("Kai22 and HaWoR frame axes differ")

    temporal = audit_suffix_invariance(
        full_current_inputs={},
        truncated_current_inputs={},
        required_fields=TEMPORAL_REQUIRED_FIELDS,
    )
    blockers = [
        "BLOCKED_INDEPENDENT_PART_OBSERVABILITY",
        "BLOCKED_INDEPENDENT_REPROJECTION",
        "BLOCKED_SUFFIX_PAIR_MATERIALIZATION",
        "BLOCKED_R0_LOCAL_QUALITY_POLICY",
    ]
    bounded = _bounded_audit(raw, candidate)
    blockers.extend(bounded["reason_codes"])
    inputs = {
        "hawor_npz": artifact_ref(hawor_npz),
        "hawor_result": artifact_ref(hawor_result),
        "kai22_npz": artifact_ref(kai22_npz),
        "kai22_result": artifact_ref(kai22_result),
    }
    if bounded_npz is not None and bounded_result is not None:
        inputs["bounded_npz"] = artifact_ref(bounded_npz)
        inputs["bounded_result"] = artifact_ref(bounded_result)
    if sam_result is not None:
        inputs["sam_result"] = artifact_ref(sam_result)
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETED_FAIL_CLOSED_AUDIT",
        "cohort": cohort,
        "session_id": session_id,
        "inputs": inputs,
        "shape_audit": {
            "hawor": shape_audit,
            "kai22": {"status": "PASS_ARRAYS_READABLE", "arrays": _array_shapes(kai22_arrays)},
            "bounded": (
                {"status": "PASS_ARRAYS_READABLE", "arrays": _array_shapes(candidate)}
                if candidate is not None
                else {"status": "ABSENT"}
            ),
        },
        "part_presence": _part_presence(raw),
        "provider_dependency": _sam_dependency_audit(sam_document, hawor_npz),
        "temporal_authority": temporal,
        "bounded_comparison": bounded,
        "kai22_tier": _kai22_tier(kai22_arrays, kai22_document, temporal, root),
        "blocker_codes": sorted(set(blockers)),
        "model_calls": 0,
        "gpu_calls": 0,
        "quality_authority_promoted": False,
        "training_eligible": False,
        "control_ground_truth": False,
        "physical_deployment_authorized": False,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=sorted(COHORT_ROOTS), required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--hawor-npz", type=Path, required=True)
    parser.add_argument("--hawor-result", type=Path, required=True)
    parser.add_argument("--kai22-npz", type=Path, required=True)
    parser.add_argument("--kai22-result", type=Path, required=True)
    parser.add_argument("--bounded-npz", type=Path)
    parser.add_argument("--bounded-result", type=Path)
    parser.add_argument("--sam-result", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise RealAssetAuditError(f"fresh output root required: {args.output_root}")
    value = build_audit(
        cohort=args.cohort,
        session_id=args.session_id,
        hawor_npz=args.hawor_npz,
        hawor_result=args.hawor_result,
        kai22_npz=args.kai22_npz,
        kai22_result=args.kai22_result,
        bounded_npz=args.bounded_npz,
        bounded_result=args.bounded_result,
        sam_result=args.sam_result,
    )
    args.output_root.mkdir(parents=True)
    atomic_json(args.output_root / "RESULT.json", value)
    print(json.dumps(value, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
