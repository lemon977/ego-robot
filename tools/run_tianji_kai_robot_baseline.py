#!/usr/bin/env python3
"""Unified full-session Tianji + dual KaiHand Robot baseline runner.

The entry point is intentionally split at a hard preflight boundary.  Session
identity, byte hashes, frame identity, camera matrices, Object6D near/far,
Mask/Clean authority and no-clobber are checked before the legacy IK module or
any renderer module is imported or called.  ``--validate-only`` never imports
those modules and never creates the requested Robot output root.

This baseline is a development visualization.  The pinned NaturalV2 mount is
not a calibrated mechanical mount and this runner never grants deployment
authority.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "tianji-kai-robot-baseline-session-spec-v1"
REVIEW_SCHEMA = "baseline-agent-stage-review-v1"
BASELINE_RUN_ID = "20260908_two_task_e2e_baseline_v1"
PROJECT = Path(__file__).resolve().parents[1]
KINEMATIC_RUNNER = PROJECT / "tools/run_newtask_robot_kinematic_canary.py"
POSTPROCESSOR = PROJECT / "pipeline/tianji_kai_robot_baseline_post.py"
CHIPS034_V2_METHOD_CONTRACT = (
    PROJECT
    / "tasks/control/runs/20260908_two_task_e2e_baseline_v1"
    / "robot_formal_method_contracts_v1/chips034_v2.json"
)
EXPECTED_CHIPS034_V2_METHOD_CONTRACT_SHA256 = (
    "bcfc496b9f231b91c5574ebb49125d7a5f59064ce259e97a38c8c4c4fa4f7545"
)
FORMAL_METHOD_CONTRACTS = {
    ("chips", "get_potato_chips_0902_034"): CHIPS034_V2_METHOD_CONTRACT,
}
CANONICAL_ASSETS = {
    "asset_pin": PROJECT / "assets/robot/ROBOT_ASSET_PIN.json",
    "tianji_urdf": PROJECT / "assets/robot/tianji/marvin_description/urdf/marvin_CCS_m6.urdf",
    "kaihand_left_urdf": PROJECT / "assets/robot/kaihand/packages/KaiBot-Dexhand shell-URDF-L-260624(1620)/urdf/KaiBot-Dexhand shell-URDF-L-260624(1620).urdf",
    "kaihand_right_urdf": PROJECT / "assets/robot/kaihand/packages/KaiBot-Dexhand shell-URDF-R-260424(1430)/urdf/KaiBot-Dexhand shell-URDF-R-260424(1430).urdf",
    "naturalv2_stl": PROJECT / "assets/robot/V2相机固定法兰PRO版本.STL",
    "mount_authority": PROJECT / "tasks/chips/runs/robot/20260903_shared_adapter_successor_v1/PINNED_KAIHAND_MOUNT_AUTHORITY_V1.json",
    "thumb_adapter": PROJECT / "tasks/chips/runs/robot/20260903_shared_adapter_successor_v1/calibration_v1/SHARED_THUMB_AXIS_ADAPTER.npz",
    "pbr_palette": PROJECT / "systems/robot/configs/robot_pbr_palette_004ref_v1.json",
}
ALLOWED_TASKS = frozenset({"chips", "poker"})
ALLOWED_GRADES = frozenset({"A", "B"})
MANO_NAMES = (
    "wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
)


class BaselineContractError(RuntimeError):
    """Raised before Robot execution when an input contract is not closed."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise BaselineContractError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BaselineContractError(f"cannot read strict JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BaselineContractError(f"JSON root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _validate_artifact(value: Any, name: str) -> Path:
    if not isinstance(value, dict) or set(value) != {"path", "bytes", "sha256"}:
        raise BaselineContractError(
            f"{name} must contain exactly path/bytes/sha256"
        )
    if not isinstance(value["path"], str) or not Path(value["path"]).is_absolute():
        raise BaselineContractError(f"{name}.path must be absolute")
    path = Path(value["path"])
    if path.is_symlink():
        raise BaselineContractError(f"{name} must not be a symlink: {path}")
    try:
        info = path.stat()
    except OSError as exc:
        raise BaselineContractError(f"{name} is missing: {path}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise BaselineContractError(f"{name} is not an ordinary file: {path}")
    if type(value["bytes"]) is not int or value["bytes"] <= 0:
        raise BaselineContractError(f"{name}.bytes must be a positive integer")
    expected = value["sha256"]
    if (
        not isinstance(expected, str)
        or len(expected) != 64
        or any(character not in "0123456789abcdef" for character in expected)
    ):
        raise BaselineContractError(f"{name}.sha256 is not lowercase SHA256")
    if info.st_size != value["bytes"] or _sha256(path) != expected:
        raise BaselineContractError(f"{name} byte identity drift: {path}")
    return path.resolve(strict=True)


def _validate_formal_method_contract(task: str, session_id: str) -> dict[str, Any] | None:
    """Bind the canary-proven V2 method before any Robot module is imported."""
    path = FORMAL_METHOD_CONTRACTS.get((task, session_id))
    if path is None:
        return None
    if path.is_symlink() or not path.is_file():
        raise BaselineContractError("formal Robot method contract must be an ordinary file")
    resolved = path.resolve(strict=True)
    reference = _artifact(resolved)
    if reference["sha256"] != EXPECTED_CHIPS034_V2_METHOD_CONTRACT_SHA256:
        raise BaselineContractError("formal Robot method contract SHA drift")
    contract = _load_json(resolved)
    scope = contract.get("scope", {})
    hand = contract.get("hand_retarget", {})
    placement = contract.get("frame0_static_placement", {})
    endpoint = contract.get("endpoint_gate_after_placement", {})
    temporal = contract.get("temporal_hard_gates", {})
    if not all(
        (
            contract.get("schema_version") == "tianji-kai-gradeb-cross-chirality-successor-v1",
            contract.get("successor_revision") == "V2_ONE_STEP_STOP_VIABILITY_FORMAL_INTEGRATION",
            contract.get("task_id") == task,
            contract.get("session_id") == session_id,
            scope.get("kind") == "FULL_SESSION_FORMAL_DEVELOPMENT_VISUALIZATION_METHOD",
            scope.get("formal_robot_execution_requires_fresh_preflight") is True,
            scope.get("formal_deployment_authorized") is False,
            scope.get("clean_consumed_at_contract_creation") is False,
            hand.get("method") == "NONLINEAR_PER_FINGER_LEAST_SQUARES",
            hand.get("fast_analytic_allowed") is False,
            hand.get("absolute_bone_error_max_deg") == 60.0,
            hand.get("per_finger_mean_error_max_deg") == 40.0,
            hand.get("distal_bone_error_max_deg") == 25.0,
            hand.get("hundred_degree_relaxation_allowed") is False,
            placement.get("freeze_for_session") is True,
            placement.get("development_visualization_only") is True,
            placement.get("real_robot_measurement_or_calibration") is False,
            endpoint == {"position_max_mm": 40.0, "rotation_max_deg": 20.0},
            temporal.get("later_frame_seed") == "PREVIOUS_ACCEPTED_ONLY",
            temporal.get("arm_step_max_rad") == 0.12,
            temporal.get("hand_step_max_rad") == 0.08,
            temporal.get("arm_second_difference_max_rad") == 0.06,
            temporal.get("hand_second_difference_max_rad") == 0.06,
            temporal.get("arm_solver_effective_step_max_rad") == 0.06,
            temporal.get("hand_solver_effective_step_max_rad") == 0.06,
            temporal.get("one_step_stop_viability_required") is True,
            temporal.get("urdf_limits_required") is True,
            temporal.get("branch_jump_forbidden") is True,
            temporal.get("collision_gate_relaxation_allowed") is False,
            temporal.get("full_fixed_denominator_per_link_collision_required_in_postprocessor") is True,
            temporal.get("five_pad_contact_metrics_required_in_postprocessor") is True,
        )
    ):
        raise BaselineContractError("formal Robot method contract content drift")
    validated = contract.get("validated_canary", {})
    for name in ("readiness_result", "canary_result"):
        _validate_artifact(validated.get(name), f"formal method validated_canary.{name}")
    if validated.get("accepted_frames") != 24 or validated.get("visual_kinematic_candidate") is not True:
        raise BaselineContractError("formal Robot method canary evidence drift")
    return reference


def _raw_frame_tree_digest(root: Path, frame_count: int) -> str:
    value = hashlib.sha256()
    expected = []
    for frame_id in range(frame_count):
        path = root / "preprocess/all_data" / f"{frame_id:05d}" / "rgb.png"
        if not path.is_file() or path.is_symlink():
            raise BaselineContractError(f"missing ordinary RAW identity frame: {path}")
        expected.append(path.resolve(strict=True))
        value.update(
            f"{frame_id:08d}\0{path.stat().st_size}\0{_sha256(path)}\n".encode()
        )
    observed = {
        path.resolve(strict=True)
        for path in (root / "preprocess/all_data").glob("*/rgb.png")
        if path.is_file()
    }
    if observed != set(expected):
        raise BaselineContractError("RAW identity frame set is not exactly 0..N-1")
    return value.hexdigest()


def _require_keys(
    value: Any,
    *,
    required: set[str],
    optional: set[str] = set(),
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise BaselineContractError(f"{name} must be an object")
    missing = required - set(value)
    extra = set(value) - required - optional
    if missing or extra:
        raise BaselineContractError(
            f"{name} key mismatch missing={sorted(missing)} extra={sorted(extra)}"
        )
    return value


def _probe_video(path: Path, name: str, expected_frames: int, fps: float) -> dict[str, Any]:
    command = [
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames,r_frame_rate,width,height",
        "-of", "json", str(path),
    ]
    try:
        completed = subprocess.run(
            command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True,
        )
        stream = json.loads(completed.stdout)["streams"][0]
        frames = int(stream["nb_read_frames"])
        numerator, denominator = map(int, stream["r_frame_rate"].split("/"))
        measured_fps = numerator / denominator
    except (OSError, subprocess.CalledProcessError, KeyError, ValueError, json.JSONDecodeError) as exc:
        raise BaselineContractError(f"{name} ffprobe failed: {exc}") from exc
    if frames != expected_frames:
        raise BaselineContractError(
            f"{name} frame count mismatch: {frames} != {expected_frames}"
        )
    if abs(measured_fps - fps) > 1e-3:
        raise BaselineContractError(
            f"{name} fps mismatch: {measured_fps} != {fps}"
        )
    return {
        "frames": frames,
        "fps": measured_fps,
        "width": int(stream["width"]),
        "height": int(stream["height"]),
    }


def _result_identity(
    value: Mapping[str, Any], *, task: str, session_id: str, frame_count: int, name: str
) -> None:
    observed_session = value.get("session_id", value.get("session"))
    observed_task = value.get("task", value.get("task_id"))
    observed_frames = value.get(
        "frame_count", value.get("full_session_frame_count", value.get("session_frame_count"))
    )
    if observed_frames is None:
        observed_frames = (
            value.get("validation", {}).get("full_video", {}).get("output_frames")
            if isinstance(value.get("validation"), dict)
            else None
        )
    if observed_session != session_id or observed_task != task:
        raise BaselineContractError(f"{name} task/session identity mismatch")
    if type(observed_frames) is not int or observed_frames != frame_count:
        raise BaselineContractError(f"{name} full-session frame count mismatch")


def _validate_review(
    review_path: Path,
    *,
    task: str,
    session_id: str,
    stage: str,
    result_sha256: str,
    required_artifact_sha256s: Mapping[str, str] | None = None,
) -> str:
    if str(PROJECT) not in sys.path:
        sys.path.insert(0, str(PROJECT))
    from tools.validate_baseline_agent_review import validate as validate_agent_review

    try:
        validation = validate_agent_review(review_path)
    except Exception as exc:
        raise BaselineContractError(
            f"{stage} central agent-review validation failed: {exc}"
        ) from exc
    review = _load_json(review_path)
    required = {
        "schema_version", "created_at", "run_id", "stage", "task", "session",
        "grade", "downstream_authorized", "hard_gates", "soft_defects",
        "inputs", "artifacts", "claim_limit",
    }
    _require_keys(review, required=required, optional={"metrics"}, name=f"{stage} review")
    if review["schema_version"] != REVIEW_SCHEMA:
        raise BaselineContractError(f"{stage} review schema mismatch")
    if review["run_id"] != BASELINE_RUN_ID:
        raise BaselineContractError(f"{stage} review run_id mismatch")
    if review["task"] != task or review["session"] != session_id or review["stage"] != stage.upper():
        raise BaselineContractError(f"{stage} review identity mismatch")
    hard_gates = review["hard_gates"]
    if (
        review["grade"] not in ALLOWED_GRADES
        or review["downstream_authorized"] is not True
        or not isinstance(hard_gates, dict)
        or not hard_gates
        or any(value not in {"PASS", "NOT_APPLICABLE"} for value in hard_gates.values())
    ):
        raise BaselineContractError(f"{stage} is not an admitted A/B baseline")
    if (
        validation.get("status") != "PASS_BASELINE_AGENT_REVIEW"
        or validation.get("downstream_authorized") is not True
    ):
        raise BaselineContractError(f"{stage} central agent-review validation failed")
    references = []
    for group_name, group in (("inputs", review["inputs"]), ("artifacts", review["artifacts"])):
        if not isinstance(group, dict):
            raise BaselineContractError(f"{stage} review artifact map is invalid")
        for artifact_name, reference in group.items():
            _validate_artifact(
                reference, f"{stage} review.{group_name}.{artifact_name}"
            )
            references.append(reference)
    if not any(
        isinstance(reference, dict) and reference.get("sha256") == result_sha256
        for reference in references
    ):
        raise BaselineContractError(f"{stage} review/result SHA mismatch")
    observed_sha256s = {
        str(reference["sha256"])
        for reference in references
        if isinstance(reference, dict) and isinstance(reference.get("sha256"), str)
    }
    for artifact_name, expected_sha256 in (required_artifact_sha256s or {}).items():
        if expected_sha256 not in observed_sha256s:
            raise BaselineContractError(
                f"{stage} review does not bind exact {artifact_name} SHA"
            )
    return str(review["grade"])


def _validate_stage(
    entry: Any,
    *,
    task: str,
    session_id: str,
    frame_count: int,
    stage: str,
    payload_key: str | None,
    review_bind_keys: frozenset[str] = frozenset(),
) -> tuple[dict[str, Path], str, Mapping[str, Any]]:
    required = {"result", "agent_review"}
    if payload_key:
        required.add(payload_key)
    stage_value = _require_keys(
        entry,
        required=required,
        optional={"frame_manifest", "source_map_manifest", "video"},
        name=f"inputs.{stage}",
    )
    paths = {
        key: _validate_artifact(ref, f"inputs.{stage}.{key}")
        for key, ref in stage_value.items()
    }
    result = _load_json(paths["result"])
    _result_identity(
        result, task=task, session_id=session_id, frame_count=frame_count,
        name=f"{stage} result",
    )
    status = result.get("status")
    if not isinstance(status, str) or not status:
        raise BaselineContractError(f"{stage} result has no producer status")
    missing_review_bindings = review_bind_keys - set(stage_value)
    if missing_review_bindings:
        raise BaselineContractError(
            f"{stage} review SHA binding inputs are missing: "
            f"{sorted(missing_review_bindings)}"
        )
    grade = _validate_review(
        paths["agent_review"], task=task, session_id=session_id, stage=stage,
        result_sha256=stage_value["result"]["sha256"],
        required_artifact_sha256s={
            key: str(stage_value[key]["sha256"]) for key in review_bind_keys
        },
    )
    # The central agent review is the baseline authority.  A producer may
    # retain an older HUMAN_REVIEW/PENDING label or consumption_authorized=false;
    # those legacy fields do not reinstate a human gate after an A/B review.
    return paths, grade, result


def _validate_frame_manifest(path: Path, session_id: str, frame_count: int, name: str) -> None:
    value = _load_json(path)
    if value.get("session_id", value.get("session")) != session_id:
        raise BaselineContractError(f"{name} session mismatch")
    rows = value.get("frames")
    if not isinstance(rows, list) or len(rows) != frame_count:
        raise BaselineContractError(f"{name} does not cover the full session")
    ids = [row.get("frame_id", row.get("source_frame")) if isinstance(row, dict) else None for row in rows]
    if ids != list(range(frame_count)):
        raise BaselineContractError(f"{name} frame IDs must be exactly 0..N-1")


def _same_artifact_reference(left: Any, right: Any) -> bool:
    return (
        isinstance(left, dict)
        and isinstance(right, dict)
        and set(left) == {"path", "bytes", "sha256"}
        and set(right) == {"path", "bytes", "sha256"}
        and all(left[key] == right[key] for key in ("path", "bytes", "sha256"))
    )


def _require_review_sha_bindings(
    review_path: Path, required: Mapping[str, Mapping[str, Any]], stage: str
) -> None:
    review = _load_json(review_path)
    references: list[Mapping[str, Any]] = []
    for group_name in ("inputs", "artifacts"):
        group = review.get(group_name)
        if not isinstance(group, dict):
            raise BaselineContractError(f"{stage} review.{group_name} is invalid")
        references.extend(
            reference for reference in group.values() if isinstance(reference, dict)
        )
    observed = {str(reference.get("sha256")) for reference in references}
    for name, reference in required.items():
        if reference.get("sha256") not in observed:
            raise BaselineContractError(
                f"{stage} review does not bind exact {name} SHA"
            )


def _validate_clean_source_manifest(
    path: Path,
    *,
    task: str,
    session_id: str,
    frame_count: int,
    width: int,
    height: int,
    raw_video_ref: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    """Validate the dual-eye exact-pixel provenance envelope without Robot imports."""
    import numpy as np

    value = _load_json(path)
    if value.get("task", value.get("task_id")) != task:
        raise BaselineContractError("Clean source-map manifest task mismatch")
    if value.get("session_id", value.get("session")) != session_id:
        raise BaselineContractError("Clean source-map manifest session mismatch")
    if value.get("frame_count") != frame_count:
        raise BaselineContractError("Clean source-map manifest frame_count mismatch")
    lineage = value.get("source_lineage")
    if not isinstance(lineage, dict):
        raise BaselineContractError("Clean source-map manifest source_lineage is required")
    required_lineage = {
        "selected_left_raw_video",
        "synchronized_right_stereo_video",
    }
    if not required_lineage.issubset(lineage):
        raise BaselineContractError("Clean source-map manifest dual-eye lineage is incomplete")
    left_ref = lineage["selected_left_raw_video"]
    right_ref = lineage["synchronized_right_stereo_video"]
    _validate_artifact(left_ref, "Clean selected-left lineage")
    _validate_artifact(right_ref, "Clean synchronized-right stereo lineage")
    if not _same_artifact_reference(left_ref, raw_video_ref):
        raise BaselineContractError(
            "Clean selected-left lineage does not bind the exact Robot RAW video"
        )

    codes = value.get("source_kind_codes")
    expected_codes = {
        "0": "TARGET_RAW",
        "1": "SAME_SESSION_LEFT_TEMPORAL_RAW",
        "2": "PROTECTED_OBJECT_RAW",
        "3": "UNSUPPORTED_RAW",
        "4": "SAME_SESSION_SYNCHRONIZED_RIGHT_RAW",
    }
    if codes != expected_codes:
        raise BaselineContractError("Clean source-map source_kind_codes mismatch")
    rows = value.get("frames")
    if not isinstance(rows, list) or len(rows) != frame_count:
        raise BaselineContractError("Clean source-map manifest does not cover full session")
    y_grid, x_grid = np.indices((height, width), dtype=np.int32)
    for frame_id, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("frame_id") != frame_id:
            raise BaselineContractError(
                "Clean source-map manifest frame IDs must be exactly 0..N-1"
            )
        for key in (
            "selected_left_raw_decoded_sha256",
            "right_raw_decoded_sha256",
        ):
            sha256 = row.get(key)
            if (
                not isinstance(sha256, str)
                or len(sha256) != 64
                or any(character not in "0123456789abcdef" for character in sha256)
            ):
                raise BaselineContractError(
                    f"Clean source-map frame {frame_id} lacks valid {key}"
                )
        source_path = _validate_artifact(
            row.get("pixel_source_map"),
            f"Clean source-map frame {frame_id} pixel_source_map",
        )
        try:
            with np.load(source_path, allow_pickle=False) as archive:
                required_arrays = {
                    "frame_id", "source_kind", "source_eye", "source_frame",
                    "source_x", "source_y",
                }
                missing = required_arrays - set(archive.files)
                if missing:
                    raise BaselineContractError(
                        f"Clean source-map frame {frame_id} lacks arrays: {sorted(missing)}"
                    )
                observed_frame = int(np.asarray(archive["frame_id"]).item())
                source_kind = np.asarray(archive["source_kind"])
                source_eye = np.asarray(archive["source_eye"])
                source_frame = np.asarray(archive["source_frame"])
                source_x = np.asarray(archive["source_x"])
                source_y = np.asarray(archive["source_y"])
        except (OSError, ValueError, KeyError) as exc:
            raise BaselineContractError(
                f"Clean source-map frame {frame_id} NPZ is invalid: {exc}"
            ) from exc
        arrays = (source_kind, source_eye, source_frame, source_x, source_y)
        if observed_frame != frame_id or any(array.shape != (height, width) for array in arrays):
            raise BaselineContractError(
                f"Clean source-map frame {frame_id} identity/shape mismatch"
            )
        if source_kind.dtype != np.uint8 or source_eye.dtype != np.uint8:
            raise BaselineContractError(
                f"Clean source-map frame {frame_id} kind/eye dtype mismatch"
            )
        if any(array.dtype != np.int32 for array in (source_frame, source_x, source_y)):
            raise BaselineContractError(
                f"Clean source-map frame {frame_id} coordinate dtype mismatch"
            )
        if not np.all(np.isin(source_kind, np.arange(5, dtype=np.uint8))):
            raise BaselineContractError(f"Clean source-map frame {frame_id} kind invalid")
        expected_eye = (source_kind == 4).astype(np.uint8)
        if not np.array_equal(source_eye, expected_eye):
            raise BaselineContractError(
                f"Clean source-map frame {frame_id} source_eye/kind mismatch"
            )
        if np.any(source_frame < 0) or np.any(source_frame >= frame_count):
            raise BaselineContractError(
                f"Clean source-map frame {frame_id} source frame out of range"
            )
        right = source_kind == 4
        selected_left = ~right
        if (
            np.any(source_x[selected_left] < 0)
            or np.any(source_x[selected_left] >= width)
            or np.any(source_y[selected_left] < 0)
            or np.any(source_y[selected_left] >= height)
        ):
            raise BaselineContractError(
                f"Clean source-map frame {frame_id} selected-left source coordinate out of range"
            )
        right_width, right_height = 2048, 1536
        if (
            np.any(source_x[right] < 0)
            or np.any(source_x[right] >= right_width)
            or np.any(source_y[right] < 0)
            or np.any(source_y[right] >= right_height)
        ):
            raise BaselineContractError(
                f"Clean source-map frame {frame_id} right-raw source coordinate out of range"
            )
        if np.any(source_frame[right] != frame_id):
            raise BaselineContractError(
                f"Clean source-map frame {frame_id} right donor is not synchronized"
            )
        immutable = np.isin(source_kind, np.asarray((0, 2, 3), dtype=np.uint8))
        if (
            np.any(source_frame[immutable] != frame_id)
            or np.any(source_x[immutable] != x_grid[immutable])
            or np.any(source_y[immutable] != y_grid[immutable])
        ):
            raise BaselineContractError(
                f"Clean source-map frame {frame_id} raw/protected/unsupported provenance is not identity"
            )
        # kind=1 is a selected-left donor from any frame in this same session;
        # its bounded frame/coordinate checks above deliberately allow temporal use.
    return {
        "selected_left_raw_video": left_ref,
        "synchronized_right_stereo_video": right_ref,
    }


def _validate_npz_geometry(
    hawor_path: Path,
    object6d_path: Path,
    *,
    frame_count: int,
    fps: float,
) -> dict[str, Any]:
    # numpy is deliberately imported inside the preflight; no Robot/IK/Blender
    # module has been imported at this point.
    import numpy as np

    with np.load(hawor_path, allow_pickle=False) as archive:
        required = {
            "joints_3d_world", "joints_3d_camera", "joints_2d", "observed",
            "provenance", "c2w", "intrinsics", "original_frame_indices", "fps",
            "mano_joint_names", "mano_wrist_index", "mano_tip_indices",
        }
        missing = required - set(archive.files)
        if missing:
            raise BaselineContractError(f"HaWoR NPZ missing keys: {sorted(missing)}")
        world = np.asarray(archive["joints_3d_world"])
        camera = np.asarray(archive["joints_3d_camera"])
        points2d = np.asarray(archive["joints_2d"])
        c2w = np.asarray(archive["c2w"], dtype=np.float64)
        intrinsics = np.asarray(archive["intrinsics"], dtype=np.float64)
        frames = np.asarray(archive["original_frame_indices"], dtype=np.int64)
        observed = np.asarray(archive["observed"], dtype=bool)
        provenance = np.asarray(archive["provenance"])
        archive_fps = float(np.asarray(archive["fps"]).item())
        names = tuple(str(value) for value in archive["mano_joint_names"].tolist())
        wrist = int(np.asarray(archive["mano_wrist_index"]).item())
        tips = tuple(int(value) for value in archive["mano_tip_indices"].tolist())
    if world.shape != (2, frame_count, 21, 3) or camera.shape != world.shape:
        raise BaselineContractError("HaWoR 3D MANO21 shape mismatch")
    if points2d.shape != (2, frame_count, 21, 2):
        raise BaselineContractError("HaWoR 2D MANO21 shape mismatch")
    if c2w.shape != (frame_count, 4, 4) or intrinsics.shape != (frame_count, 3, 3):
        raise BaselineContractError("HaWoR c2w/K full-session shape mismatch")
    if observed.shape != (2, frame_count) or provenance.shape != (2, frame_count):
        raise BaselineContractError("HaWoR observation identity shape mismatch")
    if not np.isfinite(world).all() or not np.isfinite(camera).all() or not np.isfinite(points2d).all():
        raise BaselineContractError("HaWoR contains non-finite hand geometry")
    if not np.isfinite(c2w).all() or not np.isfinite(intrinsics).all():
        raise BaselineContractError("HaWoR contains non-finite c2w/K")
    if not np.array_equal(frames, np.arange(frame_count)):
        raise BaselineContractError("HaWoR frame IDs are not exactly 0..N-1")
    if abs(archive_fps - fps) > 1e-9:
        raise BaselineContractError("HaWoR fps mismatch")
    if names != MANO_NAMES or wrist != 0 or tips != (4, 8, 12, 16, 20):
        raise BaselineContractError("HaWoR MANO21 semantic identity mismatch")
    allowed_provenance = {"OBSERVED", "BOUNDED_PARAMETER_FIT"}
    observed_labels = set(str(value) for value in np.unique(provenance))
    if not np.all(observed) or not observed_labels or not observed_labels <= allowed_provenance:
        raise BaselineContractError(
            "Robot baseline requires bilateral observed HaWoR with bounded-v2 provenance"
        )
    bottom_error = float(np.max(np.abs(c2w[:, 3] - np.asarray((0.0, 0.0, 0.0, 1.0)))))
    orthogonal_error = float(np.max(np.abs(np.swapaxes(c2w[:, :3, :3], 1, 2) @ c2w[:, :3, :3] - np.eye(3))))
    determinant_error = float(np.max(np.abs(np.linalg.det(c2w[:, :3, :3]) - 1.0)))
    if bottom_error > 1e-8 or orthogonal_error > 1e-4 or determinant_error > 1e-4:
        raise BaselineContractError("HaWoR c2w is not finite right-handed SE(3)")
    if np.any(intrinsics[:, 0, 0] <= 0) or np.any(intrinsics[:, 1, 1] <= 0):
        raise BaselineContractError("HaWoR K has non-positive focal length")

    with np.load(object6d_path, allow_pickle=False) as archive:
        required = {
            "frame_indices", "valid", "observed", "visibility", "physical_instance_id",
            "T_object_to_camera", "T_object_to_world",
            "observed_near_far_optical_z_m", "analytic_near_far_optical_z_m",
        }
        missing = required - set(archive.files)
        if missing:
            raise BaselineContractError(f"Object6D NPZ missing keys: {sorted(missing)}")
        object_frames = np.asarray(archive["frame_indices"], dtype=np.int64)
        valid = np.asarray(archive["valid"], dtype=bool)
        direct_observed = np.asarray(archive["observed"], dtype=bool)
        visibility = np.asarray(archive["visibility"], dtype=np.float64)
        instances = np.asarray(archive["physical_instance_id"], dtype=np.int64)
        object_camera = np.asarray(archive["T_object_to_camera"], dtype=np.float64)
        object_world = np.asarray(archive["T_object_to_world"], dtype=np.float64)
        observed_nf = np.asarray(archive["observed_near_far_optical_z_m"], dtype=np.float64)
        analytic_nf = np.asarray(archive["analytic_near_far_optical_z_m"], dtype=np.float64)
        size_key = "object_size_m" if "object_size_m" in archive.files else "object_dimensions_m"
        if size_key not in archive.files:
            raise BaselineContractError("Object6D NPZ lacks object size")
        size = np.asarray(archive[size_key], dtype=np.float64)
    if not np.array_equal(object_frames, np.arange(frame_count)):
        raise BaselineContractError("Object6D frame IDs are not exactly 0..N-1")
    if valid.shape != (frame_count,) or not np.all(valid):
        raise BaselineContractError("Object6D must provide a valid pose for every frame")
    if direct_observed.shape != (frame_count,):
        raise BaselineContractError("Object6D direct-observation flags have wrong shape")
    observed_fraction = float(direct_observed.mean())
    gap_lengths: list[int] = []
    gap = 0
    for present in direct_observed.tolist() + [True]:
        if not present:
            gap += 1
        elif gap:
            gap_lengths.append(gap)
            gap = 0
    max_gap = max(gap_lengths, default=0)
    if observed_fraction < 0.70 or max_gap > 45:
        raise BaselineContractError(
            "Object6D direct observations do not meet Grade-B occlusion bounds"
        )
    if visibility.shape != (frame_count,) or not np.isfinite(visibility).all() or np.any((visibility < 0) | (visibility > 1)):
        raise BaselineContractError("Object6D visibility is invalid")
    if object_camera.shape != (frame_count, 4, 4) or object_world.shape != (frame_count, 4, 4):
        raise BaselineContractError("Object6D transform shape mismatch")
    if not np.isfinite(object_camera).all() or not np.isfinite(object_world).all():
        raise BaselineContractError("Object6D contains non-finite transforms")
    if observed_nf.shape != (frame_count, 2) or analytic_nf.shape != (frame_count, 2):
        raise BaselineContractError("Object6D near/far shape mismatch")
    if not np.isfinite(observed_nf[direct_observed]).all() or not np.isfinite(analytic_nf).all():
        raise BaselineContractError("Object6D direct near/far or analytic near/far is non-finite")
    if np.isfinite(observed_nf[~direct_observed]).any():
        raise BaselineContractError("Object6D fabricated observed near/far on an occluded frame")
    if np.any(visibility[~direct_observed] != 0.0):
        raise BaselineContractError("Object6D occluded frame must have zero visibility")
    if np.any(observed_nf[direct_observed, 0] <= 0) or np.any(observed_nf[direct_observed, 1] < observed_nf[direct_observed, 0]):
        raise BaselineContractError("Object6D observed near/far is not positive ordered Z")
    if np.any(analytic_nf[:, 0] <= 0) or np.any(analytic_nf[:, 1] < analytic_nf[:, 0]):
        raise BaselineContractError("Object6D analytic near/far is not positive ordered Z")
    active_instances = np.unique(instances[valid])
    if active_instances.shape != (1,) or int(active_instances[0]) < 0:
        raise BaselineContractError("Object6D does not preserve one physical instance")
    if size.shape not in {(3,), (frame_count, 3)} or not np.isfinite(size).all() or np.any(size <= 0):
        raise BaselineContractError("Object6D size must be positive finite metres")
    camera_closure = np.linalg.inv(c2w) @ object_world
    closure_error = float(np.max(np.abs(camera_closure - object_camera)))
    if closure_error > 2e-4:
        raise BaselineContractError(
            f"Object6D world/camera coordinate closure failed: {closure_error}"
        )
    return {
        "c2w_bottom_row_max_error": bottom_error,
        "c2w_rotation_orthogonal_max_error": orthogonal_error,
        "c2w_rotation_determinant_max_error": determinant_error,
        "object6d_camera_world_closure_max_abs": closure_error,
        "physical_instance_id": int(active_instances[0]),
        "object6d_direct_observed_fraction": observed_fraction,
        "object6d_max_propagated_gap_frames": max_gap,
    }


def validate_spec(spec_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    spec_ref = _artifact(spec_path)
    spec = _load_json(spec_path)
    _require_keys(
        spec,
        required={"schema_version", "session", "inputs", "assets", "solver", "review", "output_root"},
        name="spec",
    )
    if spec["schema_version"] != SCHEMA_VERSION:
        raise BaselineContractError("session spec schema_version mismatch")
    session = _require_keys(
        spec["session"], required={"task", "session_id", "frame_count", "fps"}, name="session"
    )
    task, session_id = session["task"], session["session_id"]
    frame_count, fps = session["frame_count"], session["fps"]
    if task not in ALLOWED_TASKS or not isinstance(session_id, str) or not session_id:
        raise BaselineContractError("invalid task/session identity")
    if type(frame_count) is not int or frame_count < 24:
        raise BaselineContractError("frame_count must be an integer >=24")
    if not isinstance(fps, (int, float)) or not (0.0 < float(fps) <= 240.0):
        raise BaselineContractError("fps is invalid")
    formal_method_contract = _validate_formal_method_contract(task, session_id)

    inputs = _require_keys(
        spec["inputs"],
        required={
            "raw_video", "raw_frame_root", "raw_frame_tree_sha256",
            "hawor", "mask", "object6d", "clean",
        },
        name="inputs",
    )
    raw_video = _validate_artifact(inputs["raw_video"], "inputs.raw_video")
    raw_root = Path(inputs["raw_frame_root"])
    if not raw_root.is_absolute() or not raw_root.is_dir() or raw_root.is_symlink():
        raise BaselineContractError("inputs.raw_frame_root must be an existing ordinary directory")
    expected_raw_tree_sha = inputs["raw_frame_tree_sha256"]
    if (
        not isinstance(expected_raw_tree_sha, str)
        or len(expected_raw_tree_sha) != 64
        or any(character not in "0123456789abcdef" for character in expected_raw_tree_sha)
    ):
        raise BaselineContractError("inputs.raw_frame_tree_sha256 is invalid")
    observed_raw_tree_sha = _raw_frame_tree_digest(raw_root, frame_count)
    if observed_raw_tree_sha != expected_raw_tree_sha:
        raise BaselineContractError("RAW frame tree byte identity drift")
    raw_probe = _probe_video(raw_video, "RAW video", frame_count, float(fps))

    hawor_paths, hawor_grade, _ = _validate_stage(
        inputs["hawor"], task=task, session_id=session_id, frame_count=frame_count,
        stage="hawor", payload_key="npz",
    )
    mask_paths, mask_grade, _ = _validate_stage(
        inputs["mask"], task=task, session_id=session_id, frame_count=frame_count,
        stage="mask", payload_key=None,
    )
    object_paths, object_grade, _ = _validate_stage(
        inputs["object6d"], task=task, session_id=session_id, frame_count=frame_count,
        stage="object6d", payload_key="npz",
    )
    clean_paths, clean_grade, _ = _validate_stage(
        inputs["clean"], task=task, session_id=session_id, frame_count=frame_count,
        stage="clean", payload_key=None,
        review_bind_keys=frozenset({"source_map_manifest", "video"}),
    )
    for required_key, stage_paths, stage_name in (
        ("frame_manifest", mask_paths, "Mask frame manifest"),
        ("frame_manifest", object_paths, "Object6D frame manifest"),
    ):
        if required_key not in stage_paths:
            raise BaselineContractError(f"{stage_name} is required")
        _validate_frame_manifest(stage_paths[required_key], session_id, frame_count, stage_name)
    if "video" not in clean_paths:
        raise BaselineContractError("Clean full-session video is required")
    clean_probe = _probe_video(clean_paths["video"], "Clean video", frame_count, float(fps))
    if (
        clean_probe["width"] != raw_probe["width"]
        or clean_probe["height"] != raw_probe["height"]
    ):
        raise BaselineContractError(
            "Clean video must be a full-resolution Clean master matching RAW; "
            "a tiled review video is not a valid Robot background"
        )
    if "source_map_manifest" not in clean_paths:
        raise BaselineContractError("Clean source-map manifest is required")
    clean_lineage = _validate_clean_source_manifest(
        clean_paths["source_map_manifest"],
        task=task,
        session_id=session_id,
        frame_count=frame_count,
        width=raw_probe["width"],
        height=raw_probe["height"],
        raw_video_ref=inputs["raw_video"],
    )
    _require_review_sha_bindings(
        clean_paths["agent_review"],
        {
            "source_map_manifest": inputs["clean"]["source_map_manifest"],
            "clean_master_video": inputs["clean"]["video"],
            **clean_lineage,
        },
        "clean",
    )
    geometry = _validate_npz_geometry(
        hawor_paths["npz"], object_paths["npz"], frame_count=frame_count, fps=float(fps)
    )

    assets = _require_keys(
        spec["assets"],
        required={
            "asset_pin", "tianji_urdf", "kaihand_left_urdf", "kaihand_right_urdf",
            "naturalv2_stl", "mount_authority", "thumb_adapter", "pbr_palette",
        },
        name="assets",
    )
    asset_paths = {
        key: _validate_artifact(ref, f"assets.{key}") for key, ref in assets.items()
    }
    for key, canonical in CANONICAL_ASSETS.items():
        if asset_paths[key] != canonical.resolve(strict=True):
            raise BaselineContractError(
                f"assets.{key} must select the canonical project asset"
            )
    pin = _load_json(asset_paths["asset_pin"])
    if pin.get("status") != "T0_IDENTITY_PIN_NOT_EXECUTION_AUTHORIZATION":
        raise BaselineContractError("robot asset pin status drift")
    mount = _load_json(asset_paths["mount_authority"])
    if mount.get("formal_consumer_allowed") is not False or "DEVELOPMENT_ONLY" not in str(mount.get("status")):
        raise BaselineContractError("mount must remain explicitly development-only")

    solver = _require_keys(
        spec["solver"],
        required={
            "arm_step_limit_rad", "hand_step_limit_rad", "later_frame_seed",
            "first_frame_policy", "human_to_physical", "require_urdf_limits",
            "require_branch_jump_false", "collision_denominator", "pad_names",
        },
        name="solver",
    )
    if solver["arm_step_limit_rad"] != 0.12 or solver["hand_step_limit_rad"] != 0.08:
        raise BaselineContractError("baseline arm/hand step limits must be exactly 0.12/0.08 rad")
    if solver["later_frame_seed"] != "PREVIOUS_ACCEPTED_ONLY":
        raise BaselineContractError("later frames must use only the previous accepted state")
    if solver["first_frame_policy"] != "STATIC_IK_FULL_URDF_LIMITS":
        raise BaselineContractError("first frame policy must be the static IK exception")
    if solver["human_to_physical"] != {"left": "right", "right": "left"}:
        raise BaselineContractError("human/physical side mapping drift")
    if solver["require_urdf_limits"] is not True or solver["require_branch_jump_false"] is not True:
        raise BaselineContractError("URDF-limit and branch-jump hard gates are required")
    if solver["collision_denominator"] != "FRAME_X_SIDE_X_LINK_FIXED":
        raise BaselineContractError("per-link collision denominator is not fixed")
    if solver["pad_names"] != ["thumb", "index", "middle", "ring", "pinky"]:
        raise BaselineContractError("all five KaiHand CAD pad names are required")

    review = _require_keys(
        spec["review"],
        required={"language", "full_session", "panels", "video_name"},
        name="review",
    )
    if review["language"] != "zh-CN" or review["full_session"] is not True:
        raise BaselineContractError("review must be full-session Chinese")
    required_panels = {"raw", "clean", "robot", "contact_collision"}
    if set(review["panels"]) != required_panels:
        raise BaselineContractError("review panels must be RAW/Clean/Robot/contact_collision")
    if not isinstance(review["video_name"], str) or not review["video_name"].endswith(".mp4"):
        raise BaselineContractError("review.video_name must be an MP4 filename")
    if Path(review["video_name"]).name != review["video_name"]:
        raise BaselineContractError("review.video_name must not contain directories")

    output = Path(spec["output_root"])
    if not output.is_absolute():
        raise BaselineContractError("output_root must be absolute")
    if output.exists() or output.is_symlink():
        raise BaselineContractError(f"fresh output_root required: {output}")
    parent = output.parent.resolve(strict=True)
    if not parent.is_dir():
        raise BaselineContractError("output_root parent is not a directory")
    if not os.access(parent, os.W_OK | os.X_OK):
        raise BaselineContractError("output_root parent is not writable")

    implementation = {
        "runner": _artifact(Path(__file__)),
        "kinematic_runner": _artifact(KINEMATIC_RUNNER),
        "postprocessor": _artifact(POSTPROCESSOR),
    }
    if formal_method_contract is not None:
        implementation["formal_method_contract"] = formal_method_contract
    result = {
        "schema_version": "tianji-kai-robot-baseline-preflight-v1",
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "status": "PASS_VALIDATE_ONLY_READY_WAIT_UPSTREAM_EXECUTION" if all(
            grade in ALLOWED_GRADES for grade in (hawor_grade, mask_grade, object_grade, clean_grade)
        ) else "FAIL_CLOSED",
        "task": task,
        "session_id": session_id,
        "frame_count": frame_count,
        "fps": float(fps),
        "no_clobber": True,
        "output_root_absent": True,
        "no_robot_modules_imported_by_preflight": True,
        "input_grades": {
            "hawor": hawor_grade, "mask": mask_grade,
            "object6d": object_grade, "clean": clean_grade,
        },
        "video_probes": {"raw": raw_probe, "clean": clean_probe},
        "raw_frame_tree_sha256": observed_raw_tree_sha,
        "geometry": geometry,
        "solver_contract": dict(solver),
        "kinematic_method": {
            "name": (
                "V2_ONE_STEP_STOP_VIABILITY_FRAME0_STATIC_DEVELOPMENT_PLACEMENT"
                if formal_method_contract is not None
                else "LEGACY_PINNED_METHOD_NO_V2_SESSION_CONTRACT"
            ),
            "formal_method_contract": formal_method_contract,
            "validated_before_robot_import": True,
            "target_is_objective_not_hard_bound": True,
        },
        "mount_authority": {
            "development_only": True,
            "formal_consumer_allowed": False,
            "claim": "GRADE_B_DEVELOPMENT_VISUALIZATION_NOT_CALIBRATED_OR_DEPLOYABLE",
        },
        "spec": spec_ref,
        "implementation": implementation,
        "output_root": str(output),
    }
    return spec, result


def _atomic_new_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise BaselineContractError(f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_grade_c_execution_receipt(
    output: Path,
    spec: Mapping[str, Any],
    preflight: Mapping[str, Any],
    *,
    phase: str,
    reason: str,
) -> dict[str, Any]:
    """Persist a reviewable Grade-C terminal instead of a partial run directory."""
    result_path = output / "RESULT.json"
    if result_path.exists() or result_path.is_symlink():
        return _load_json(result_path)
    kinematic_path = output / "kinematic/RESULT.json"
    rejection_path = output / "kinematic/REJECTED_FRAME_DIAGNOSTIC.json"
    process_path = output / "KINEMATIC_PROCESS.json"
    preflight_path = output / "PREFLIGHT_RESULT.json"
    branch_jump: bool | None = None
    if kinematic_path.is_file() and not kinematic_path.is_symlink():
        try:
            kinematic_result = _load_json(kinematic_path)
            branch_value = kinematic_result.get("previous_accepted_contract", {}).get(
                "branch_jump"
            )
            if isinstance(branch_value, bool):
                branch_jump = branch_value
        except BaselineContractError:
            pass
    if "strict previous-accepted frame gate failed" in reason.lower():
        branch_jump = True
    result: dict[str, Any] = {
        "schema_version": "tianji-kai-robot-baseline-result-v1",
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "status": (
            "FAIL_GRADE_C_BRANCH_JUMP"
            if branch_jump is True
            else "FAIL_GRADE_C_ROBOT_EXECUTION"
        ),
        "grade": "C",
        "consumption_authorized": False,
        "task": spec["session"]["task"],
        "session_id": spec["session"]["session_id"],
        "frame_count": spec["session"]["frame_count"],
        "failure_phase": phase,
        "failure_reason": reason,
        "branch_jump": branch_jump,
        "preflight": _artifact(preflight_path),
        "mount_authority": {
            "development_only": True,
            "formal_consumer_allowed": False,
        },
        "formal_deployment_authorized": False,
        "claim_limit": "Grade-C stopped Robot attempt; no deployment or downstream authority.",
    }
    if process_path.is_file() and not process_path.is_symlink():
        result["kinematic_process"] = _artifact(process_path)
    if kinematic_path.is_file() and not kinematic_path.is_symlink():
        result["kinematic_result"] = _artifact(kinematic_path)
    if rejection_path.is_file() and not rejection_path.is_symlink():
        result["rejected_frame_diagnostic"] = _artifact(rejection_path)
    _atomic_new_json(result_path, result)
    artifact_map = {
        "result": _artifact(result_path),
        "preflight": _artifact(preflight_path),
    }
    if "kinematic_process" in result:
        artifact_map["kinematic_process"] = result["kinematic_process"]
    if "kinematic_result" in result:
        artifact_map["kinematic_result"] = result["kinematic_result"]
    if "rejected_frame_diagnostic" in result:
        artifact_map["rejected_frame_diagnostic"] = result[
            "rejected_frame_diagnostic"
        ]
    review = {
        "schema_version": REVIEW_SCHEMA,
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "run_id": BASELINE_RUN_ID,
        "stage": "ROBOT",
        "task": spec["session"]["task"],
        "session": spec["session"]["session_id"],
        "grade": "C",
        "downstream_authorized": False,
        "hard_gates": {
            "input_identity_sha": "PASS",
            "robot_execution": "FAIL",
            "previous_accepted_branch": (
                "FAIL" if branch_jump is True else "NOT_APPLICABLE"
            ),
            "full_session_video_decode": "NOT_APPLICABLE",
        },
        "soft_defects": [],
        "inputs": {
            "session_spec": preflight["spec"],
            "hawor_result": spec["inputs"]["hawor"]["result"],
            "mask_result": spec["inputs"]["mask"]["result"],
            "object6d_result": spec["inputs"]["object6d"]["result"],
            "clean_result": spec["inputs"]["clean"]["result"],
        },
        "artifacts": artifact_map,
        "metrics": {
            "frame_count": spec["session"]["frame_count"],
            "failure_phase": phase,
            "branch_jump": branch_jump,
            "formal_deployment_authorized": False,
        },
        "claim_limit": "Grade-C stopped Robot attempt; no deployment or downstream authority.",
    }
    review_path = output / "AGENT_REVIEW.json"
    _atomic_new_json(review_path, review)
    from tools.validate_baseline_agent_review import validate as validate_agent_review

    validation = validate_agent_review(review_path)
    _atomic_new_json(output / "AGENT_REVIEW_VALIDATION.json", validation)
    return result


def _build_kinematic_command(
    spec: Mapping[str, Any], preflight: Mapping[str, Any], kinematic: Path
) -> list[str]:
    session = spec["session"]
    inputs = spec["inputs"]
    command = [
        sys.executable, str(KINEMATIC_RUNNER),
        "--task-id", session["task"],
        "--session-id", session["session_id"],
        "--raw-root", inputs["raw_frame_root"],
        "--hawor-npz", inputs["hawor"]["npz"]["path"],
        "--hawor-result", inputs["hawor"]["result"]["path"],
        "--source-admission", inputs["hawor"]["result"]["path"],
        "--output-root", str(kinematic),
        "--window-frames", str(session["frame_count"]),
        "--sparse-review",
        "--fixed-previous-ik-branch", "--first-frame-static-exception",
        "--strict-previous-accepted", "--arm-step-limit", "0.12",
        "--hand-step-limit", "0.08",
        "--thumb-adapter", spec["assets"]["thumb_adapter"]["path"],
        "--accepted-provenance", "BOUNDED_PARAMETER_FIT",
    ]
    method_contract = preflight.get("kinematic_method", {}).get(
        "formal_method_contract"
    )
    if method_contract is not None:
        command.extend(
            [
                "--prefer-earliest-window",
                "--gradeb-successor-contract",
                method_contract["path"],
            ]
        )
    else:
        command.append("--fast-hand-retarget")
    return command


def _run_execution(spec: Mapping[str, Any], preflight: Mapping[str, Any]) -> dict[str, Any]:
    output = Path(spec["output_root"])
    output.mkdir(mode=0o755)
    _atomic_new_json(output / "PREFLIGHT_RESULT.json", preflight)
    kinematic = output / "kinematic"
    session = spec["session"]
    command = _build_kinematic_command(spec, preflight, kinematic)
    method_contract = preflight.get("kinematic_method", {}).get(
        "formal_method_contract"
    )
    phase = "KINEMATIC_SUBPROCESS"
    try:
        completed = subprocess.run(
            command, cwd=PROJECT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        _atomic_new_json(
            output / "KINEMATIC_PROCESS.json",
            {
                "command": command,
                "returncode": completed.returncode,
                "stdout_tail": completed.stdout[-8000:],
                "stderr_tail": completed.stderr[-8000:],
            },
        )
        if completed.returncode != 0:
            detail = completed.stderr[-2000:].strip() or completed.stdout[-2000:].strip()
            raise BaselineContractError(
                f"kinematic subprocess failed with return code {completed.returncode}: {detail}"
            )
        phase = "KINEMATIC_TERMINAL_AUDIT"
        kinematic_result = _load_json(kinematic / "RESULT.json")
        previous = kinematic_result.get("previous_accepted_contract", {})
        if previous.get("branch_jump") is not False:
            raise BaselineContractError("kinematic branch_jump terminal is not false")
        if previous.get("later_frame_seed_policy") != "PREVIOUS_ACCEPTED_ONLY":
            raise BaselineContractError("kinematic runner did not preserve previous-accepted seed")
        previous_frames = previous.get("frames", [])
        if (
            not isinstance(previous_frames, list)
            or len(previous_frames) != session["frame_count"]
            or not all(row.get("accepted") is True for row in previous_frames)
        ):
            raise BaselineContractError("kinematic runner contains a rejected frame")
        if kinematic_result.get("thumb_adapter", {}).get("sha256") != spec["assets"]["thumb_adapter"]["sha256"]:
            raise BaselineContractError("kinematic runner did not consume the pinned thumb adapter")
        if method_contract is not None:
            observed_contract = kinematic_result.get("gradeb_successor_contract", {})
            placement = kinematic_result.get("frame0_static_placement", {})
            temporal = kinematic_result.get("temporal", {})
            if observed_contract.get("sha256") != method_contract["sha256"]:
                raise BaselineContractError("kinematic V2 method contract consumption drift")
            if not (
                placement.get("enabled") is True
                and placement.get("frozen_for_session") is True
                and placement.get("source_frame") == 0
                and placement.get("development_visualization_only") is True
                and placement.get("real_robot_measurement_or_calibration") is False
                and placement.get("pre_and_post_residuals_reported_separately") is True
            ):
                raise BaselineContractError("kinematic V2 frame0 placement audit drift")
            if not (
                temporal.get("one_step_stop_viability_by_velocity_cap") is True
                and temporal.get("arm_solver_effective_velocity_limit_rad_per_frame") == 0.06
                and temporal.get("hand_solver_effective_velocity_limit_rad_per_frame") == 0.06
                and temporal.get("arm_velocity_max_rad_per_frame", float("inf")) <= 0.12
                and temporal.get("hand_velocity_max_rad_per_frame", float("inf")) <= 0.08
                and temporal.get("arm_acceleration_max_rad_per_frame2", float("inf")) <= 0.06
                and temporal.get("hand_acceleration_max_rad_per_frame2", float("inf")) <= 0.06
            ):
                raise BaselineContractError("kinematic V2 temporal hard gate drift")

        # Import the Robot postprocessor only after the complete preflight and IK
        # subprocess have succeeded.  This boundary is intentionally visible.
        phase = "POSTPROCESS_AUDIT_RENDER"
        if str(PROJECT) not in sys.path:
            sys.path.insert(0, str(PROJECT))
        from pipeline.tianji_kai_robot_baseline_post import audit_and_render

        post = audit_and_render(spec, output, kinematic_result)
    except Exception as exc:
        _write_grade_c_execution_receipt(
            output,
            spec,
            preflight,
            phase=phase,
            reason=f"{type(exc).__name__}: {exc}",
        )
        if isinstance(exc, BaselineContractError):
            raise
        raise BaselineContractError(f"{phase} failed: {exc}") from exc
    result = {
        "schema_version": "tianji-kai-robot-baseline-result-v1",
        "status": post["status"],
        "grade": "B" if post["hard_gates_pass"] else "C",
        "consumption_authorized": bool(post["hard_gates_pass"]),
        "task": session["task"],
        "session_id": session["session_id"],
        "frame_count": session["frame_count"],
        "hard_gates_pass": bool(post["hard_gates_pass"]),
        "preflight": _artifact(output / "PREFLIGHT_RESULT.json"),
        "kinematic_result": _artifact(kinematic / "RESULT.json"),
        "trajectory_audit": post["trajectory_audit"],
        "collision_contact": post["collision_contact"],
        "mount_audit": post["mount_audit"],
        "review_video": post["review_video"],
        "mount_authority": {
            "development_only": True,
            "formal_consumer_allowed": False,
        },
        "kinematic_method": preflight.get("kinematic_method"),
        "formal_deployment_authorized": False,
        "claim_limit": "Grade-B development visualization only; nominal URDF and development mount are not calibrated real-robot deployment authority.",
    }
    _atomic_new_json(output / "RESULT.json", result)
    result_ref = _artifact(output / "RESULT.json")
    review = {
        "schema_version": REVIEW_SCHEMA,
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "run_id": BASELINE_RUN_ID,
        "stage": "ROBOT",
        "task": session["task"],
        "session": session["session_id"],
        "grade": result["grade"],
        "downstream_authorized": result["hard_gates_pass"],
        "hard_gates": {
            "input_identity_sha": "PASS",
            "frame_camera_coordinate_closure": "PASS",
            "urdf_joint_limits": "PASS" if result["hard_gates_pass"] else "FAIL",
            "previous_accepted_branch": "PASS" if result["hard_gates_pass"] else "FAIL",
            "per_link_collision_and_pad_contact": "PASS" if result["hard_gates_pass"] else "FAIL",
            "naturalv2_12mm_direction_clearance": "PASS" if result["hard_gates_pass"] else "FAIL",
            "full_session_video_decode": "PASS" if result["hard_gates_pass"] else "FAIL",
        },
        "soft_defects": [
            "NaturalV2 mount is a nominal development-only visual transform, not calibrated deployment authority"
        ],
        "inputs": {
            "session_spec": preflight["spec"],
            "hawor_result": spec["inputs"]["hawor"]["result"],
            "mask_result": spec["inputs"]["mask"]["result"],
            "object6d_result": spec["inputs"]["object6d"]["result"],
            "clean_result": spec["inputs"]["clean"]["result"],
        },
        "artifacts": {
            "result": result_ref,
            "review_video": result["review_video"],
            "trajectory_audit": result["trajectory_audit"],
            "collision_contact": result["collision_contact"],
            "mount_audit": result["mount_audit"],
        },
        "metrics": {
            "frame_count": session["frame_count"],
            "formal_deployment_authorized": False,
        },
        "claim_limit": "Grade-B development visualization only; nominal URDF and development mount are not calibrated real-robot deployment authority.",
    }
    _atomic_new_json(output / "AGENT_REVIEW.json", review)
    from tools.validate_baseline_agent_review import validate as validate_agent_review

    validation = validate_agent_review(output / "AGENT_REVIEW.json")
    _atomic_new_json(output / "AGENT_REVIEW_VALIDATION.json", validation)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-spec", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--validation-output", type=Path)
    args = parser.parse_args(argv)
    if args.validation_output is not None and not args.validate_only:
        parser.error("--validation-output requires --validate-only")
    try:
        spec, preflight = validate_spec(args.session_spec.resolve(strict=True))
        if args.validate_only:
            if args.validation_output is not None:
                _atomic_new_json(args.validation_output.resolve(strict=False), preflight)
            print(json.dumps(preflight, ensure_ascii=False, indent=2))
            return 0
        result = _run_execution(spec, preflight)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["hard_gates_pass"] else 2
    except (BaselineContractError, OSError, ValueError, KeyError) as exc:
        print(
            json.dumps(
                {"status": "FAIL_CLOSED", "error": f"{type(exc).__name__}: {exc}"},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
