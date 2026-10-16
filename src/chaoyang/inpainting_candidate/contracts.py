"""Candidate-only input closure and component receipts (no model imports).

These helpers are not a production entry or a replacement for Task Packet gates.
Receipts certify bytes and dependency identity, never visual or training quality.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

SCHEMA = "inpainting-candidate-component-v1"
STAGES = ("Source", "Motion", "Mask", "Clean", "Render", "Composite")
REQUIRED_CATEGORIES = frozenset({"source", "code", "config", "weights", "calibration", "schema"})
FORBIDDEN_ROLES = frozenset({
    "manual_seed", "visible_seed", "session_roi", "historical_mask", "historical_clean",
    "historical_motion", "robot_q", "session_correction", "contact_window", "manual_placement",
    "task_config", "evaluation_annotation",
})
ALLOWED_PROVENANCE = frozenset({"RAW_CAPTURE", "GENERIC_CODE", "FROZEN_WEIGHT", "GENERIC_CONFIG",
                               "CAPTURE_CALIBRATION", "INSTALLATION_CONTRACT", "SCHEMA"})
ALLOWED_ROLES = {
    "source": {"source_manifest", "raw_video", "raw_video_qpc", "raw_video_wall", "raw_controller_head",
               "raw_manus", "raw_manus_meta", "raw_camera", "raw_camera_meta", "raw_installation_container"},
    "code": {"source_code", "motion_code", "mask_code", "clean_code", "render_code", "composite_code", "runtime_code"},
    "weights": {"segmentation_weight", "propainter_weight", "flow_weight", "robot_asset"},
    "config": {"generic_recipe", "installation_contract", "camera_contract"},
    "calibration": {"camera_calibration", "installation_calibration", "stereo_calibration"},
    "schema": {"input_schema", "result_schema", "frame_schema"},
}
ROLE_STAGES = {
    **{role: {"Source"} for role in ("source_manifest", "raw_video", "raw_video_qpc", "raw_video_wall", "raw_camera", "raw_camera_meta")},
    **{role: {"Motion"} for role in ("raw_controller_head", "raw_manus", "raw_manus_meta", "raw_installation_container")},
    **{stage.lower() + "_code": {stage} for stage in STAGES},
    "runtime_code": set(STAGES),
    "segmentation_weight": {"Mask"}, "propainter_weight": {"Clean"},
    "flow_weight": {"Clean"}, "robot_asset": {"Render"},
    "generic_recipe": set(STAGES), "installation_contract": {"Motion"},
    "camera_contract": {"Source", "Render"},
    "camera_calibration": {"Source", "Render"}, "installation_calibration": {"Motion"},
    "stereo_calibration": {"Source", "Render"},
    "input_schema": {"Source"}, "frame_schema": {"Source"}, "result_schema": {"Composite"},
}
REQUIRED_STAGE_ROLES = {
    "Source": {"source_manifest", "raw_video", "raw_video_qpc", "raw_video_wall", "source_code", "input_schema", "frame_schema"},
    "Motion": {"motion_code"},
    "Mask": {"mask_code", "segmentation_weight", "generic_recipe"},
    "Clean": {"clean_code", "propainter_weight", "flow_weight", "generic_recipe"},
    "Render": {"render_code", "robot_asset"},
    "Composite": {"composite_code", "generic_recipe", "result_schema"},
}


def _producer_policy(row: Mapping[str, Any]) -> None:
    category, role = row.get("category"), row.get("role")
    if role in FORBIDDEN_ROLES or role not in ALLOWED_ROLES.get(category, set()):
        raise ContractError(f"PRODUCER_INPUT_FORBIDDEN:{role}")
    provenance = row.get("provenance")
    allowed = {"source": {"RAW_CAPTURE"}, "code": {"GENERIC_CODE"}, "weights": {"FROZEN_WEIGHT"},
               "config": {"GENERIC_CONFIG", "INSTALLATION_CONTRACT"},
               "calibration": {"CAPTURE_CALIBRATION", "INSTALLATION_CONTRACT"}, "schema": {"SCHEMA"}}
    if provenance not in allowed.get(category, set()):
        raise ContractError("INPUT_CATEGORY_PROVENANCE_CONFLICT")
    if category == "source" and "file" in row:
        parts = {part.lower() for part in Path(row["file"]["path"]).parts}
        if parts & {"_run", "review", "processed", "attempts"}:
            raise ContractError("DERIVED_SOURCE_PATH_FORBIDDEN")


class ContractError(ValueError):
    pass


def canonical_sha(value: Any) -> str:
    """Hash an immutable JSON value; NaN and Infinity are never signed."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_ref(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve(strict=True)
    if not path.is_file():
        raise ContractError(f"INPUT_NOT_FILE:{path}")
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ContractError(f"INPUT_DRIFT_DURING_HASH:{path}")
    return {"path": str(path), "bytes": after.st_size, "sha256": digest.hexdigest()}


def verify_ref(reference: Mapping[str, Any]) -> None:
    actual = file_ref(reference["path"])
    if any(actual[k] != reference.get(k) for k in ("path", "bytes", "sha256")):
        raise ContractError(f"INPUT_SHA_DRIFT:{reference.get('path')}")


def input_entry(name: str, *, category: str, provenance: str, role: str,
                path: str | Path | None = None, value: Any = None,
                absent: bool = False) -> dict[str, Any]:
    """Exactly one file/inline/ABSENT representation; absence is calibration-only."""
    if category not in REQUIRED_CATEGORIES or provenance not in ALLOWED_PROVENANCE:
        raise ContractError("INPUT_CATEGORY_OR_PROVENANCE_UNSUPPORTED")
    _producer_policy({"category": category, "provenance": provenance, "role": role})
    if absent:
        if category != "calibration" or path is not None or value is not None:
            raise ContractError("ABSENT_ONLY_CALIBRATION")
        payload = {"status": "ABSENT"}
    elif path is not None:
        if value is not None:
            raise ContractError("INPUT_FILE_AND_INLINE_CONFLICT")
        payload = {"status": "PRESENT", "file": file_ref(path)}
    else:
        if category not in {"config", "schema", "calibration"} or value is None:
            raise ContractError("INPUT_ACTUAL_FILE_REQUIRED")
        payload = {"status": "INLINE", "value": value, "sha256": canonical_sha(value)}
    result = {"name": name, "category": category, "provenance": provenance, "role": role, **payload}
    _producer_policy(result)
    return result


def freeze_input_manifest(entries: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Require all signature domains and independently re-verify actual files."""
    rows = [dict(row) for row in entries]
    names = [row.get("name") for row in rows]
    if not rows or any(not isinstance(name, str) or not name for name in names) or len(names) != len(set(names)):
        raise ContractError("INPUT_NAMES_EMPTY_OR_DUPLICATED")
    if {row.get("category") for row in rows} != REQUIRED_CATEGORIES:
        raise ContractError("INPUT_CLOSURE_CATEGORIES_INCOMPLETE")
    for row in rows:
        _producer_policy(row)
        status = row.get("status")
        if status == "PRESENT":
            verify_ref(row["file"])
        elif status == "INLINE":
            if (row["category"] not in {"config", "schema", "calibration"}
                    or canonical_sha(row.get("value")) != row.get("sha256")):
                raise ContractError("INPUT_INLINE_SIGNATURE_INVALID")
        elif status != "ABSENT" or row["category"] != "calibration":
            raise ContractError("INPUT_STATUS_INVALID")
    rows.sort(key=lambda row: row["name"])
    result = {"schema": SCHEMA, "producer_input_policy": "RAW_AND_GENERIC_ONLY",
              "entries": rows, "control_ground_truth": False}
    return {**result, "sha256": canonical_sha(result)}


def stage_signatures(manifest: Mapping[str, Any], stage_inputs: Mapping[str, Iterable[str]],
                     *, mask_uses_motion: bool = True) -> dict[str, dict[str, Any]]:
    """Sign only actual component dependencies, not the whole package per stage.

    Motion-assisted ROI is a real Mask dependency. When disabled, a Motion edit
    does not invalidate Mask/Clean. Raw source changes still invalidate consumers.
    """
    manifest = freeze_input_manifest(manifest["entries"])
    rows = {entry["name"]: entry for entry in manifest["entries"]}
    if set(stage_inputs) != set(STAGES):
        raise ContractError("STAGE_ASSIGNMENTS_INCOMPLETE")
    used: set[str] = set()
    dependencies = {"Source": (), "Motion": ("Source",),
                    "Mask": ("Source", "Motion") if mask_uses_motion else ("Source",),
                    "Clean": ("Source", "Mask"), "Render": ("Source", "Motion"),
                    "Composite": ("Source", "Mask", "Clean", "Render")}
    result = {}
    for stage in STAGES:
        names = sorted(stage_inputs[stage])
        if not names or len(names) != len(set(names)) or any(name not in rows for name in names):
            raise ContractError(f"STAGE_INPUTS_INVALID:{stage}")
        used.update(names)
        roles = {rows[name]["role"] for name in names}
        if any(stage not in ROLE_STAGES[role] for role in roles):
            raise ContractError(f"STAGE_ROLE_ASSIGNMENT_INVALID:{stage}")
        missing_roles = REQUIRED_STAGE_ROLES[stage] - roles
        if missing_roles:
            raise ContractError(f"STAGE_CLOSURE_INCOMPLETE:{stage}:" + ",".join(sorted(missing_roles)))
        if stage == "Motion" and not roles & {"raw_installation_container", "installation_calibration", "installation_contract"}:
            raise ContractError("STAGE_CLOSURE_INCOMPLETE:Motion:installation_source_or_ABSENT")
        if stage == "Render" and not roles & {"camera_calibration", "stereo_calibration"}:
            raise ContractError("STAGE_CLOSURE_INCOMPLETE:Render:camera_calibration_or_ABSENT")
        payload = {"schema": SCHEMA, "stage": stage,
                   "inputs": [rows[name] for name in names],
                   "upstream": {name: result[name]["sha256"] for name in dependencies[stage]},
                   "mask_uses_motion": mask_uses_motion if stage == "Mask" else None}
        result[stage] = {**payload, "sha256": canonical_sha(payload)}
    if used != set(rows):
        raise ContractError("STAGE_INPUTS_UNASSIGNED:" + ",".join(sorted(set(rows) - used)))
    return result


def verify_sealed_stage(receipt: Mapping[str, Any], expected_signature: str,
                        *, output_root: str | Path) -> list[dict[str, Any]]:
    """Only fully verified same-signature receipts are consumable; not partials."""
    root = Path(output_root).resolve(strict=True)
    if receipt.get("schema") != SCHEMA or receipt.get("status") != "SEALED_COMPLETE":
        raise ContractError("STAGE_PARTIAL_NOT_CONSUMABLE")
    if receipt.get("component_signature") != expected_signature:
        raise ContractError("STAGE_COMPONENT_SIGNATURE_MISMATCH")
    outputs = receipt.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise ContractError("STAGE_OUTPUTS_EMPTY")
    resolved = []
    for reference in outputs:
        path = Path(reference["path"]).resolve(strict=True)
        if not path.is_relative_to(root) or path == root:
            raise ContractError("STAGE_OUTPUT_OUTSIDE_WRITE_SET")
        verify_ref(reference)
        resolved.append(dict(reference))
    if len({row["path"] for row in resolved}) != len(resolved):
        raise ContractError("STAGE_OUTPUTS_DUPLICATED")
    return resolved
