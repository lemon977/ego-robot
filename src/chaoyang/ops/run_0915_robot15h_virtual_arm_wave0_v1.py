#!/usr/bin/env python3
"""Build the W0 development-only virtual Tianji + Kai22 R2 review.

The only session-varying input is the sealed R0 bundle: ``q22_init``,
``relative_wrist_T``, validity, frame IDs, and timestamps.  The runner does
not read HaWoR, RGB, SAM, Depth, Object6D, Clean, or archive material.  A
fixed mount *visual proxy* is used exactly as pinned; it is not a measured
installation transform.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Mapping
import uuid
import xml.etree.ElementTree as ET

import cv2
import jsonschema
import numpy as np

from chaoyang.governance.robot15h_task_specs_v1 import WINDOW_RUN_ID, build_packet
from chaoyang.ops.audit_robot_geometry_self_collision_v71 import audit as audit_self_collision
from chaoyang.pipeline.robot_renderer_cycles import forward_kinematics
from chaoyang.pipeline.robot_renderer_eevee_fullchain import (
    ARM_JOINT_NAMES,
    load_pinned_robot_assets,
)
from chaoyang.pipeline import robot_scene_state_cpu as arm_solver


ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "0915_robot15h_robot_virtual_arm_v1"
PHASE = "ROBOT15H_ROBOT_VIRTUAL_ARM_WAVE0"
R0_ROOT = ROOT / "_run/current/0915_robot15h_kai22_r0_wave0_v1/attempts/attempt_0001"
OUTPUT = ROOT / f"_run/current/{TASK_ID}/attempts/attempt_0001"
VISUAL = ROOT / "docs/current/visuals/0915_ROBOT15H_W0_VIRTUAL_R2_V1"
TERMINAL_RECEIPT = ROOT / "tasks/receipts/0915_ROBOT15H_ROBOT_VIRTUAL_ARM_V1_RESULT.json"
MOUNT_CONTRACT = ROOT / "assets/robot/hardware_handoff/kaihand_flange_adapter_v1/excluded_reference_only/MOUNT_VISUAL_PROXY_CONTRACT.json"
ASSET_PIN = ROOT / "assets/robot/ROBOT_ASSET_PIN.json"
VIRTUAL_INSTALLATION_SCHEMA = ROOT / "contracts/robot15h_virtual_installation_r0_motion_v1.schema.json"
VIRTUAL_INSTALLATION_CONTRACT = ROOT / "manifests/hardware/robot15h_virtual_installation_r0_motion_v1.json"
TIANJI_URDF = ROOT / "assets/robot/tianji/marvin_description/urdf/marvin_CCS_m6.urdf"
KAIHAND_URDFS = (
    ROOT / "assets/robot/kaihand/packages/KaiBot-Dexhand shell-URDF-L-260624(1620)/urdf/KaiBot-Dexhand shell-URDF-L-260624(1620).urdf",
    ROOT / "assets/robot/kaihand/packages/KaiBot-Dexhand shell-URDF-R-260424(1430)/urdf/KaiBot-Dexhand shell-URDF-R-260424(1430).urdf",
)
PINNED_SHA256 = {
    "asset_pin": "4bc508a45fd460610a73bc9fb6bf194d97a5cfdef5e863bd1069683921cd1330",
    "tianji_urdf": "3c3bdfa9aa397c55dea2b3bc94d42c3081d4292d041b1ff43573d175d5faf309",
    "kaihand_left_urdf": "dad94b420f5fb675781260562f087e839e746f99e811b54171cd23a33ca8ab74",
    "kaihand_right_urdf": "9444414bb1810d98fc1800654d1e751eb6d3c264a3e630da30f1048f98b98586",
    "mount_visual_proxy_contract": "9f7258480bfa41bbbe8b87e1752c3227d397c609f4c64fb69cd46d4845c2f731",
    "virtual_installation_schema": "a6e6cfb6d6dac03826f64f66edc341e7c5cfaef91f53fe3f50ad4d4b46d9d599",
    "virtual_installation_contract": "d86b18c71a4d45903e372b81ddcc90abbeb1b9289ddb9b5a28cbd38e3d8c7572",
}
SIDES = ("left", "right")
EXPECTED_SESSIONS = {
    "play_cards_0915_031": "playing_cards",
    "play_cards_0915_119": "playing_cards",
    "get_potato_chips_0915_007": "potato_chips",
    "get_potato_chips_0915_042": "potato_chips",
}
AUTHORITY = "DEVELOPMENT_RELATIVE_NON_CONTROL_NON_DEPLOYABLE"
POSITION_GATE_MM = 20.0
ROTATION_GATE_DEG = 15.0
PENETRATION_TOLERANCE_M = 1e-4
DYNAMIC_INPUT_POLICY = (
    "R0_Q22_RELATIVE_WRIST_VALIDITY_FRAME_ID_TIMESTAMPS_ONLY_NO_OTHER_SESSION_INPUT"
)
COLLISION_COVERAGE = "BILATERAL_VALID_FRAMES_ONLY_PARTIAL"
CODE_CLOSURE = (
    ROOT / "src/chaoyang/ops/run_0915_robot15h_virtual_arm_wave0_v1.py",
    ROOT / "src/chaoyang/ops/audit_robot_geometry_self_collision_v71.py",
    ROOT / "src/chaoyang/ops/immutable_artifact_io.py",
    ROOT / "src/chaoyang/pipeline/robot_scene_state_cpu.py",
    ROOT / "src/chaoyang/pipeline/robot_renderer_cycles.py",
    ROOT / "src/chaoyang/pipeline/robot_renderer_eevee_fullchain.py",
    ROOT / "src/chaoyang/pipeline/robot_tool_definition_contract.py",
    ROOT / "src/chaoyang/pipeline/robot_mount_visual_proxy.py",
)


class VirtualR2Error(RuntimeError):
    """One immutable-input or virtual-R2 invariant failed closed."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    candidate = path.resolve(strict=True)
    return {
        "path": str(candidate),
        "bytes": candidate.stat().st_size,
        "sha256": sha256(candidate),
    }


def published_ref(current: Path, published: Path) -> dict[str, Any]:
    """Bind bytes in a staging file to its post-rename published path."""

    candidate = current.resolve(strict=True)
    return {
        "path": str(published.resolve()),
        "bytes": candidate.stat().st_size,
        "sha256": sha256(candidate),
    }


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise VirtualR2Error(f"JSON object required: {path}")
    return value


def declared_path_matches(value: object, expected: Path) -> bool:
    """Accept equivalent repo-relative/absolute spellings of one exact path."""
    declared = Path(str(value))
    candidate = declared if declared.is_absolute() else ROOT / declared
    return candidate.resolve() == expected.resolve()


def load_virtual_installation_contract() -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the immutable schema and the authority that permits virtual motion."""

    for name, path in (
        ("virtual_installation_schema", VIRTUAL_INSTALLATION_SCHEMA),
        ("virtual_installation_contract", VIRTUAL_INSTALLATION_CONTRACT),
    ):
        if sha256(path.resolve(strict=True)) != PINNED_SHA256[name]:
            raise VirtualR2Error(f"pinned virtual installation {name} SHA drift")
    schema = load_json(VIRTUAL_INSTALLATION_SCHEMA)
    contract = load_json(VIRTUAL_INSTALLATION_CONTRACT)
    try:
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.Draft202012Validator(schema).validate(contract)
    except jsonschema.exceptions.JsonSchemaException as exc:
        raise VirtualR2Error(f"virtual installation contract schema failure: {exc.message}") from exc
    declared_schema = contract["contract_schema"]
    if (
        not declared_path_matches(declared_schema["path"], VIRTUAL_INSTALLATION_SCHEMA)
        or declared_schema["bytes"] != VIRTUAL_INSTALLATION_SCHEMA.stat().st_size
        or declared_schema["sha256"] != sha256(VIRTUAL_INSTALLATION_SCHEMA)
    ):
        raise VirtualR2Error("virtual installation schema reference drift")
    if contract["calibration"] != {
        "measured_installation_transform": "ABSENT",
        "robot_tcp": "ABSENT",
        "camera_world_to_base": "ABSENT",
        "physical_accuracy_authority": False,
    }:
        raise VirtualR2Error("virtual installation calibration must remain ABSENT")
    for name, path in (("robot_asset_pin", ASSET_PIN), ("mount_numeric_proxy", MOUNT_CONTRACT)):
        declared = contract["source_contracts"][name]
        if (
            not declared_path_matches(declared["path"], path)
            or declared["bytes"] != path.stat().st_size
            or declared["sha256"] != sha256(path)
        ):
            raise VirtualR2Error(f"virtual installation source contract drift: {name}")
    if contract["collision_authorization"]["coverage"] != COLLISION_COVERAGE:
        raise VirtualR2Error("virtual installation collision coverage drift")
    return contract, schema


def verify_robot_asset_closure() -> list[dict[str, Any]]:
    """Verify and bind every file named by the pinned Robot asset manifest."""

    pin = load_json(ASSET_PIN)
    rows = pin.get("files")
    if not isinstance(rows, list) or not rows:
        raise VirtualR2Error("robot asset pin file closure is empty")
    seen: set[str] = set()
    closure: list[dict[str, Any]] = []
    for row in rows:
        relative = str(row.get("path", ""))
        if not relative or relative in seen:
            raise VirtualR2Error("robot asset pin paths must be non-empty and unique")
        seen.add(relative)
        path = (ROOT / relative).resolve(strict=True)
        try:
            path.relative_to(ROOT)
        except ValueError as exc:
            raise VirtualR2Error(f"robot asset escapes repository: {relative}") from exc
        observed = ref(path)
        if observed["bytes"] != row.get("bytes") or observed["sha256"] != row.get("sha256"):
            raise VirtualR2Error(f"robot asset pin drift: {relative}")
        closure.append(observed)
    return closure


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def process_start_ticks(pid: int) -> int:
    return int(Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()[21])


def heartbeat(status: str = "RUNNING") -> None:
    completed = subprocess.run(
        [
            sys.executable, "-m", "chaoyang.governance.heartbeat_task",
            "--task-id", TASK_ID, "--pid", str(os.getpid()),
            "--status", status, "--phase", PHASE,
        ],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if completed.returncode:
        raise VirtualR2Error(
            "governance heartbeat failed: " + (completed.stderr or completed.stdout)[-4000:]
        )


def validate_route() -> tuple[dict[str, Any], Path]:
    packet_path = ROOT / f"tasks/current/{TASK_ID}/TASK_PACKET.json"
    packet = load_json(packet_path)
    if packet != build_packet(TASK_ID) or packet.get("weights") != "ABSENT":
        raise VirtualR2Error("current virtual R2 packet differs from frozen CPU spec")
    state = load_json(ROOT / "docs/governance/LONG_HORIZON_TASK_STATE.json")
    if state.get("next_task", {}).get("task_id") != TASK_ID:
        raise VirtualR2Error("virtual R2 is not current next_task")
    index = load_json(ROOT / "tasks/current/INDEX.json")
    route = next(
        (row for row in index.get("task_packets", []) if row.get("task_id") == TASK_ID),
        None,
    )
    if route is None or route.get("execution_allowed") is not True:
        raise VirtualR2Error("virtual R2 is not routable")
    if route.get("packet_sha256") != sha256(packet_path):
        raise VirtualR2Error("virtual R2 packet SHA differs from current route")
    return packet, packet_path


def verify_static_assets() -> tuple[dict[str, Any], np.ndarray]:
    paths = {
        "asset_pin": ASSET_PIN,
        "tianji_urdf": TIANJI_URDF,
        "kaihand_left_urdf": KAIHAND_URDFS[0],
        "kaihand_right_urdf": KAIHAND_URDFS[1],
        "mount_visual_proxy_contract": MOUNT_CONTRACT,
    }
    for name, path in paths.items():
        if sha256(path.resolve(strict=True)) != PINNED_SHA256[name]:
            raise VirtualR2Error(f"pinned static asset SHA drift: {name}")
    contract = load_json(MOUNT_CONTRACT)
    if (
        contract.get("status") != "FROZEN_VISUAL_REVIEW_PROXY_NOT_CALIBRATION"
        or contract.get("classification") != "MOUNT_PROXY"
        or contract.get("development_only") is not True
        or contract.get("real_robot_calibration") is not False
        or contract.get("deployment_authorized") is not False
    ):
        raise VirtualR2Error("mount proxy authority contract changed")
    embedded = contract.get("assets", {})
    expected_embedded = {
        "tianji_urdf": PINNED_SHA256["tianji_urdf"],
        "kaihand_left_urdf": PINNED_SHA256["kaihand_left_urdf"],
        "kaihand_right_urdf": PINNED_SHA256["kaihand_right_urdf"],
    }
    if any(embedded.get(name, {}).get("sha256") != digest for name, digest in expected_embedded.items()):
        raise VirtualR2Error("mount proxy embedded URDF identity drift")
    transforms = contract.get("proxy_transforms", {})
    if (
        transforms.get("matrix_direction")
        != "p_flange = T_flange_hand_root_proxy @ p_hand_root"
        or transforms.get("units") != "metres"
        or transforms.get("scope") != "GLOBAL_CONSTANT_BOTH_TASKS_NOT_PER_SESSION_NOT_PER_FRAME"
    ):
        raise VirtualR2Error("mount proxy transform semantics changed")
    mounts = np.asarray(
        [
            transforms["left_flange_L_to_hand_l_base_link"],
            transforms["right_flange_R_to_hand_r_base_link"],
        ],
        dtype=np.float64,
    )
    if mounts.shape != (2, 4, 4):
        raise VirtualR2Error("mount proxy transform shape drift")
    for side, matrix in enumerate(mounts):
        rotation = matrix[:3, :3]
        if (
            not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-12)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-10)
            or np.linalg.det(rotation) < 0.999999
        ):
            raise VirtualR2Error(f"mount proxy side {side} is not a proper transform")
    return {name: ref(path) for name, path in paths.items()}, mounts


def load_r0_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path.resolve(strict=True), allow_pickle=False) as archive:
        required = {
            "q22_init", "relative_wrist_T", "valid_side_frame",
            "timestamps_s", "frame_id",
        }
        if not required.issubset(archive.files):
            raise VirtualR2Error(f"R0 bundle lacks fields: {sorted(required - set(archive.files))}")
        # Deliberately copy only the dynamic fields authorized by this node.
        arrays = {name: np.asarray(archive[name]).copy() for name in required}
    q22 = arrays["q22_init"]
    wrist = arrays["relative_wrist_T"]
    valid = arrays["valid_side_frame"]
    timestamps = arrays["timestamps_s"]
    frame_id = arrays["frame_id"]
    expected_dtypes = {
        "q22_init": np.dtype("float64"),
        "relative_wrist_T": np.dtype("float64"),
        "valid_side_frame": np.dtype("bool"),
        "timestamps_s": np.dtype("float64"),
        "frame_id": np.dtype("int64"),
    }
    for name, dtype in expected_dtypes.items():
        if arrays[name].dtype != dtype:
            raise VirtualR2Error(f"R0 NPZ dtype mismatch for {name}: {arrays[name].dtype}")
    frames = len(timestamps)
    if q22.shape != (frames, 2, 22) or wrist.shape != (frames, 2, 4, 4) or valid.shape != (frames, 2):
        raise VirtualR2Error("R0 q22/wrist/valid shape mismatch")
    if frame_id.shape != (frames,) or not np.array_equal(frame_id, np.arange(frames)):
        raise VirtualR2Error("R0 frame identity is not zero-based complete")
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
        raise VirtualR2Error("R0 timestamps are not finite and strictly increasing")
    if not np.isfinite(q22[valid]).all() or not np.isfinite(wrist[valid]).all():
        raise VirtualR2Error("R0 valid row contains non-finite hand state")
    if not np.isnan(q22[~valid]).all() or not np.isnan(wrist[~valid]).all():
        raise VirtualR2Error("R0 invalid row was filled instead of NaN")
    proper_rows = wrist[valid]
    if proper_rows.size:
        if not np.allclose(proper_rows[:, 3], (0.0, 0.0, 0.0, 1.0), atol=1e-8):
            raise VirtualR2Error("R0 relative wrist homogeneous row invalid")
        rotations = proper_rows[:, :3, :3]
        if not np.allclose(np.swapaxes(rotations, 1, 2) @ rotations, np.eye(3), atol=1e-4):
            raise VirtualR2Error("R0 relative wrist rotation is not orthonormal")
        if np.any(np.linalg.det(rotations) < 0.999):
            raise VirtualR2Error("R0 relative wrist rotation is improper")
    arrays["valid_side_frame"] = valid
    arrays["timestamps_s"] = timestamps
    arrays["frame_id"] = frame_id
    return arrays


def validate_r0_batch(batch: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Validate the sealed batch contract; its row key is exactly ``results``."""

    if (
        batch.get("schema_version") != "0915-robot15h-kai22-r0-wave0-batch-v1"
        or batch.get("status") != "COMPLETED_ALL_TERMINAL"
        or batch.get("window_run_id") != WINDOW_RUN_ID
        or batch.get("session_count") != 4
        or batch.get("r0_exported") != 4
        or batch.get("failed_runtime") != 0
    ):
        raise VirtualR2Error("four-session R0 terminal is not closed")
    if "sessions" in batch or not isinstance(batch.get("results"), list):
        raise VirtualR2Error("R0 batch must use the exact results row key")
    rows = batch["results"]
    if len(rows) != 4:
        raise VirtualR2Error("R0 batch results must contain exactly four rows")
    identities = [str(row.get("session_id")) for row in rows]
    if len(set(identities)) != 4 or set(identities) != set(EXPECTED_SESSIONS):
        raise VirtualR2Error("R0 batch session identity/uniqueness mismatch")
    for row in rows:
        session_id = str(row["session_id"])
        if (
            row.get("task") != EXPECTED_SESSIONS[session_id]
            or row.get("status") != "COMPLETED_DEVELOPMENT_BASELINE"
            or row.get("r0_exported") is not True
            or not isinstance(row.get("r0_quality_admitted"), bool)
            or not isinstance(row.get("result"), dict)
        ):
            raise VirtualR2Error(f"R0 batch row contract mismatch: {session_id}")
    return rows


def validate_r0_session(
    row: Mapping[str, Any],
    r0_result: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    expected_joint_order: Mapping[str, list[str]],
) -> None:
    """Cross-check one batch row, result descriptor, and exact NPZ schema."""

    session_id = str(row["session_id"])
    fixed = {
        "schema_version": "KAI22_R0_BASELINE_WAVE0_V1",
        "session_id": session_id,
        "task": EXPECTED_SESSIONS[session_id],
        "status": "COMPLETED_DEVELOPMENT_BASELINE",
        "authority": AUTHORITY,
        "direct_observed_only": True,
        "short_gap_inferred_consumed": False,
        "arm_ik_performed": False,
        "control_ground_truth": False,
        "training_eligible": False,
        "physical_deployment_authorized": False,
        "calibration_authority": "DEVELOPMENT_ONLY",
        "q22_units": "radians",
    }
    if any(r0_result.get(name) != value for name, value in fixed.items()):
        raise VirtualR2Error(f"R0 result identity/authority mismatch: {session_id}")
    if r0_result.get("source_group") != row.get("source_group"):
        raise VirtualR2Error(f"R0 source group mismatch: {session_id}")
    if r0_result.get("q22_joint_order") != expected_joint_order:
        raise VirtualR2Error(f"R0 q22 joint identity/order mismatch: {session_id}")
    frames = len(arrays["timestamps_s"])
    valid_rows = int(np.asarray(arrays["valid_side_frame"]).sum())
    if (
        r0_result.get("frame_count") != frames
        or r0_result.get("valid_side_frame_count") != valid_rows
        or row.get("valid_side_frames") != valid_rows
        or r0_result.get("r0_quality_admitted") is not row.get("r0_quality_admitted")
    ):
        raise VirtualR2Error(f"R0 result/NPZ counts mismatch: {session_id}")


def relative_hand_root_targets(
    relative_wrist: np.ndarray,
    valid: np.ndarray,
    neutral_hand_roots: np.ndarray,
) -> np.ndarray:
    """Apply R0 motion at exact unit gain in each virtual neutral-root frame."""

    relative = np.asarray(relative_wrist, dtype=np.float64)
    selected = np.asarray(valid, dtype=bool)
    neutral = np.asarray(neutral_hand_roots, dtype=np.float64)
    if relative.ndim != 4 or relative.shape[1:] != (2, 4, 4):
        raise VirtualR2Error("relative wrist must be [T,2,4,4]")
    if selected.shape != relative.shape[:2] or neutral.shape != (2, 4, 4):
        raise VirtualR2Error("relative target shape mismatch")
    targets = np.full_like(relative, np.nan)
    for side in range(2):
        targets[selected[:, side], side] = (
            neutral[side][None, :, :] @ relative[selected[:, side], side]
        )
    return targets


def timestamp_derivatives(
    values: np.ndarray,
    timestamps_s: np.ndarray,
    valid: np.ndarray,
) -> dict[str, Any]:
    """Report physical-time derivatives without bridging invalid frame gaps."""

    q = np.asarray(values, dtype=np.float64)
    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    selected = np.asarray(valid, dtype=bool)
    if q.ndim != 3 or selected.shape != q.shape[:2] or timestamps.shape != (q.shape[0],):
        raise VirtualR2Error("timestamp derivative shape mismatch")
    velocity: list[float] = []
    acceleration: list[float] = []
    for side in range(q.shape[1]):
        ids = np.flatnonzero(selected[:, side])
        for segment in np.split(ids, np.flatnonzero(np.diff(ids) != 1) + 1):
            if len(segment) < 2:
                continue
            dt = np.diff(timestamps[segment])
            if np.any(dt <= 0):
                raise VirtualR2Error("non-positive timestamp delta")
            v = np.diff(q[segment, side], axis=0) / dt[:, None]
            velocity.extend(np.abs(v).ravel().tolist())
            if len(segment) >= 3:
                midpoint_dt = 0.5 * (dt[:-1] + dt[1:])
                a = np.diff(v, axis=0) / midpoint_dt[:, None]
                acceleration.extend(np.abs(a).ravel().tolist())
    return {
        "velocity_rad_s_max": float(max(velocity, default=0.0)),
        "velocity_rad_s_p95": float(np.percentile(velocity, 95)) if velocity else 0.0,
        "acceleration_rad_s2_max": float(max(acceleration, default=0.0)),
        "acceleration_rad_s2_p95": float(np.percentile(acceleration, 95)) if acceleration else 0.0,
        "velocity_sample_count": len(velocity),
        "acceleration_sample_count": len(acceleration),
        "gap_policy": "CONTIGUOUS_FRAME_IDS_ONLY_NO_GAP_BRIDGING_NO_FILL",
    }


def urdf_velocity_limits(path: Path, joint_names: tuple[str, ...]) -> np.ndarray:
    root = ET.parse(path).getroot()
    by_name: dict[str, float] = {}
    for joint in root.findall("joint"):
        limit = joint.find("limit")
        if limit is not None and limit.get("velocity") is not None:
            by_name[str(joint.get("name"))] = float(limit.get("velocity", "nan"))
    values = np.asarray([by_name.get(name, np.nan) for name in joint_names], dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise VirtualR2Error(f"URDF velocity limits incomplete: {path}")
    return values


def solve_virtual_session(
    arrays: Mapping[str, np.ndarray],
    assets: Any,
    mounts: np.ndarray,
    *,
    solve: Callable[..., tuple[np.ndarray, int]] = arm_solver._solve_one_arm,
) -> dict[str, Any]:
    """Solve fixed-base independent arm IK while passing q22 through bit-exactly."""

    q22_source = np.asarray(arrays["q22_init"])
    q22 = q22_source.copy()
    valid = np.asarray(arrays["valid_side_frame"], dtype=bool)
    relative = np.asarray(arrays["relative_wrist_T"], dtype=np.float64)
    timestamps = np.asarray(arrays["timestamps_s"], dtype=np.float64)
    frames = q22.shape[0]
    lower, upper = arm_solver._arm_limits(assets)
    neutral = 0.5 * (lower + upper)
    neutral_tools = np.stack(
        [arm_solver._tool_fk(assets, side, neutral[side]) for side in range(2)]
    )
    neutral_roots = neutral_tools @ mounts
    target_roots = relative_hand_root_targets(relative, valid, neutral_roots)
    target_tools = np.full_like(target_roots, np.nan)
    for side in range(2):
        target_tools[valid[:, side], side] = (
            target_roots[valid[:, side], side] @ np.linalg.inv(mounts[side])
        )
    q_arm = np.full((frames, 2, 7), np.nan, dtype=np.float64)
    actual_roots = np.full((frames, 2, 4, 4), np.nan, dtype=np.float64)
    position_mm = np.full((frames, 2), np.nan, dtype=np.float64)
    rotation_deg = np.full((frames, 2), np.nan, dtype=np.float64)
    evaluations = np.zeros((frames, 2), dtype=np.int64)
    numeric_pass = np.zeros((frames, 2), dtype=bool)
    for side in range(2):
        previous: np.ndarray | None = None
        previous_frame: int | None = None
        for frame in range(frames):
            if not valid[frame, side]:
                previous = None
                previous_frame = None
                continue
            initial = neutral[side] if previous is None or previous_frame != frame - 1 else previous
            solved, count = solve(
                assets,
                side=side,
                base=np.eye(4),
                target_tool=target_tools[frame, side],
                initial_q=initial,
                lower=lower[side],
                upper=upper[side],
            )
            solved = np.asarray(solved, dtype=np.float64)
            if solved.shape != (7,) or not np.isfinite(solved).all():
                raise VirtualR2Error("arm IK returned invalid q")
            q_arm[frame, side] = solved
            evaluations[frame, side] = int(count)
            actual_tool = arm_solver._tool_fk(assets, side, solved)
            actual_roots[frame, side] = actual_tool @ mounts[side]
            delta = np.linalg.inv(target_roots[frame, side]) @ actual_roots[frame, side]
            position_mm[frame, side] = float(np.linalg.norm(delta[:3, 3]) * 1000.0)
            rotation_deg[frame, side] = float(
                np.degrees(np.linalg.norm(arm_solver._rotation_vector(delta[:3, :3])))
            )
            within_limits = bool(
                np.all(solved >= lower[side] - 1e-10)
                and np.all(solved <= upper[side] + 1e-10)
            )
            numeric_pass[frame, side] = bool(
                within_limits
                and position_mm[frame, side] <= POSITION_GATE_MM
                and rotation_deg[frame, side] <= ROTATION_GATE_DEG
            )
            previous, previous_frame = solved, frame
    if not np.array_equal(q22, q22_source, equal_nan=True):
        raise VirtualR2Error("q22 pass-through changed source values")
    if not np.isnan(q_arm[~valid]).all() or not np.isnan(actual_roots[~valid]).all():
        raise VirtualR2Error("invalid R0 rows were filled by virtual R2")
    hand_limits = []
    for model in (assets.left_hand, assets.right_hand):
        moving = tuple(joint for joint in model.joints if joint.joint_type != "fixed")
        lo = np.asarray([joint.lower for joint in moving], dtype=np.float64)
        hi = np.asarray([joint.upper for joint in moving], dtype=np.float64)
        hand_limits.append((lo, hi))
    q22_limit_pass = all(
        bool(
            np.all(q22[:, side][valid[:, side]] >= hand_limits[side][0] - 1e-10)
            and np.all(q22[:, side][valid[:, side]] <= hand_limits[side][1] + 1e-10)
        )
        for side in range(2)
    )
    arm_motion = timestamp_derivatives(q_arm, timestamps, valid)
    hand_motion = timestamp_derivatives(q22, timestamps, valid)
    arm_velocity_limits = np.stack(
        [urdf_velocity_limits(TIANJI_URDF, tuple(ARM_JOINT_NAMES[side])) for side in range(2)]
    )
    arm_velocity_gate = True
    for side in range(2):
        ids = np.flatnonzero(valid[:, side])
        for segment in np.split(ids, np.flatnonzero(np.diff(ids) != 1) + 1):
            if len(segment) > 1:
                dt = np.diff(timestamps[segment])
                velocity = np.abs(np.diff(q_arm[segment, side], axis=0) / dt[:, None])
                arm_velocity_gate &= bool(np.all(velocity <= arm_velocity_limits[side] + 1e-9))
    return {
        "q_arm": q_arm,
        "q22": q22,
        "valid": valid,
        "T_world_base": np.eye(4, dtype=np.float64),
        "T_tool_hand_root_proxy": mounts.copy(),
        "T_target_hand_root_base": target_roots,
        "T_actual_hand_root_base": actual_roots,
        "position_residual_mm": position_mm,
        "rotation_residual_deg": rotation_deg,
        "numeric_pass": numeric_pass,
        "ik_evaluations": evaluations,
        "arm_lower": lower,
        "arm_upper": upper,
        "arm_velocity_limits_rad_s": arm_velocity_limits,
        "arm_motion": arm_motion,
        "hand_motion": hand_motion,
        "q22_joint_limits_pass": q22_limit_pass,
        "arm_velocity_limits_pass": arm_velocity_gate,
        "q22_bit_exact_pass_through": True,
        "motion_scale": 1.0,
        "base_fixed": True,
    }


def _project(point: np.ndarray, *, view: str, center: tuple[int, int], scale: float) -> tuple[int, int]:
    if view == "front":
        first, second = point[0], point[2]
    else:
        first, second = point[1], point[2]
    return int(round(center[0] + scale * first)), int(round(center[1] - scale * second))


def _draw_model(
    canvas: np.ndarray,
    model: Any,
    transforms: Mapping[str, np.ndarray],
    *,
    world: np.ndarray,
    view: str,
    center: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
) -> None:
    points = {
        name: (world @ transform)[:3, 3]
        for name, transform in transforms.items()
    }
    for joint in model.joints:
        if joint.parent not in points or joint.child not in points:
            continue
        cv2.line(
            canvas,
            _project(points[joint.parent], view=view, center=center, scale=scale),
            _project(points[joint.child], view=view, center=center, scale=scale),
            color, 2, cv2.LINE_AA,
        )


def render_virtual_review(
    destination: Path,
    session_id: str,
    solution: Mapping[str, Any],
    assets: Any,
    timestamps_s: np.ndarray,
    visual_labels: list[str],
) -> dict[str, Any]:
    """Render a synthetic, independent robot-space review (never an RGB overlay)."""

    frames = len(solution["q_arm"])
    dt = np.diff(timestamps_s)
    fps = float(1.0 / np.median(dt)) if len(dt) else 30.0
    width, height = 1280, 720
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
        "-r", f"{fps:.12g}", "-i", "-", "-an", "-c:v", "libx264",
        "-preset", "veryfast", "-crf", "19", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(destination),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    q_arm = np.asarray(solution["q_arm"])
    q22 = np.asarray(solution["q22"])
    valid = np.asarray(solution["valid"], dtype=bool)
    mounts = np.asarray(solution["T_tool_hand_root_proxy"])
    target = np.asarray(solution["T_target_hand_root_base"])
    residual = np.asarray(solution["position_residual_mm"])
    arm_names = tuple(tuple(names) for names in ARM_JOINT_NAMES)
    hand_models = (assets.left_hand, assets.right_hand)
    hand_moving = [tuple(j for j in model.joints if j.joint_type != "fixed") for model in hand_models]
    if visual_labels != [
        "VIRTUAL INSTALLATION / R0-RELATIVE MOTION",
        "CALIBRATION ABSENT",
        "NON-CONTROL / NON-DEPLOYABLE",
        "PROXY-SPACE SELF DIAGNOSTIC; OBJECT/ENV UNVERIFIED",
    ]:
        raise VirtualR2Error("virtual installation visual labels drift")
    try:
        for frame in range(frames):
            panel = np.full((height, width, 3), 24, dtype=np.uint8)
            cv2.putText(panel, f"{session_id} | frame {frame:04d}", (24, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (245, 245, 245), 2, cv2.LINE_AA)
            cv2.putText(panel, " | ".join(visual_labels[:3]), (24, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 210, 255), 1, cv2.LINE_AA)
            cv2.putText(panel, visual_labels[3], (24, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 210, 255), 1, cv2.LINE_AA)
            for x, title in ((320, "FRONT X-Z"), (960, "SIDE Y-Z")):
                cv2.putText(panel, title, (x - 65, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (190, 190, 190), 1, cv2.LINE_AA)
                cv2.line(panel, (x - 250, 620), (x + 250, 620), (70, 70, 70), 1)
            values = {name: 0.0 for names in arm_names for name in names}
            for side in range(2):
                if valid[frame, side]:
                    values.update({name: float(value) for name, value in zip(arm_names[side], q_arm[frame, side], strict=True)})
            robot_fk = forward_kinematics(assets.tianji, values)
            for view, center in (("front", (320, 620)), ("side", (960, 620))):
                _draw_model(panel, assets.tianji, robot_fk, world=np.eye(4), view=view, center=center, scale=470.0, color=(170, 170, 170))
                for side, color in enumerate(((255, 140, 45), (50, 100, 255))):
                    if not valid[frame, side]:
                        continue
                    model = hand_models[side]
                    hand_fk = forward_kinematics(
                        model,
                        {joint.name: float(value) for joint, value in zip(hand_moving[side], q22[frame, side], strict=True)},
                    )
                    root = arm_solver._tool_fk(assets, side, q_arm[frame, side]) @ mounts[side]
                    _draw_model(panel, model, hand_fk, world=root, view=view, center=center, scale=470.0, color=color)
                    point = _project(target[frame, side, :3, 3], view=view, center=center, scale=470.0)
                    cv2.drawMarker(panel, point, color, cv2.MARKER_CROSS, 15, 2)
            labels = []
            for side, hand in enumerate(SIDES):
                if valid[frame, side]:
                    labels.append(f"{hand}: IK residual {residual[frame, side]:.1f} mm")
                else:
                    labels.append(f"{hand}: UNKNOWN (NaN preserved)")
            cv2.putText(panel, labels[0], (55, 680), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 140, 45), 1, cv2.LINE_AA)
            cv2.putText(panel, labels[1], (690, 680), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (50, 100, 255), 1, cv2.LINE_AA)
            process.stdin.write(panel.tobytes())
    finally:
        process.stdin.close()
    if process.wait() != 0:
        raise VirtualR2Error(f"virtual review encode failed: {session_id}")
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames", "-of", "default=nw=1:nk=1", str(destination)],
        capture_output=True, text=True, check=False,
    )
    decode = subprocess.run(
        ["ffmpeg", "-v", "error", "-xerror", "-i", str(destination), "-f", "null", "-"],
        capture_output=True, text=True, check=False,
    )
    if probe.returncode or decode.returncode or int(probe.stdout.strip()) != frames:
        raise VirtualR2Error(f"virtual review full-decode/frame-count failed: {session_id}")
    return {"video": ref(destination), "decoded_frames": frames, "full_decode": "PASS"}


def validate_r2_state_npz(path: Path, frame_count: int) -> None:
    expected = {
        "q_arm": ((frame_count, 2, 7), np.dtype("float64")),
        "q22_init": ((frame_count, 2, 22), np.dtype("float64")),
        "q_hand": ((frame_count, 2, 22), np.dtype("float64")),
        "valid_side_frame": ((2, frame_count), np.dtype("bool")),
        "T_world_base": ((4, 4), np.dtype("float64")),
        "T_tool_hand_root": ((2, 4, 4), np.dtype("float64")),
        "T_target_hand_root_world": ((frame_count, 2, 4, 4), np.dtype("float64")),
        "T_actual_hand_root_world": ((frame_count, 2, 4, 4), np.dtype("float64")),
        "position_residual_mm": ((frame_count, 2), np.dtype("float64")),
        "rotation_residual_deg": ((frame_count, 2), np.dtype("float64")),
        "numeric_pass": ((frame_count, 2), np.dtype("bool")),
        "ik_evaluations": ((frame_count, 2), np.dtype("int64")),
        "timestamps_s": ((frame_count,), np.dtype("float64")),
        "source_frames": ((frame_count,), np.dtype("int64")),
    }
    with np.load(path.resolve(strict=True), allow_pickle=False) as archive:
        if set(archive.files) != set(expected):
            raise VirtualR2Error("R2 NPZ field closure mismatch")
        for name, (shape, dtype) in expected.items():
            value = np.asarray(archive[name])
            if value.shape != shape or value.dtype != dtype:
                raise VirtualR2Error(f"R2 NPZ schema mismatch for {name}")
        if not np.array_equal(archive["q22_init"], archive["q_hand"], equal_nan=True):
            raise VirtualR2Error("R2 q22 pass-through aliases differ")
        if not np.array_equal(archive["source_frames"], np.arange(frame_count)):
            raise VirtualR2Error("R2 source frame identity mismatch")


def validate_published_artifacts(results: list[dict[str, Any]], review_manifest: Path) -> None:
    if len(results) != 4 or {row["session_id"] for row in results} != set(EXPECTED_SESSIONS):
        raise VirtualR2Error("published R2 session closure mismatch")
    manifest = load_json(review_manifest)
    if manifest.get("session_count") != 4 or manifest.get("reviews") != [row["review"] for row in results]:
        raise VirtualR2Error("four-video review manifest mismatch")
    for row in results:
        session_id = row["session_id"]
        result_path = Path(row["result"]["path"])
        if ref(result_path) != row["result"]:
            raise VirtualR2Error(f"published R2 result ref drift: {session_id}")
        session = load_json(result_path)
        state_path = Path(session["states"]["path"])
        collision_path = Path(session["collision"]["result"]["path"])
        video_path = Path(row["review"]["video"]["path"])
        if (
            state_path != result_path.parent / "KAI22_R2_VIRTUAL_ARM_V1.npz"
            or collision_path != result_path.parent / "ROBOT_SELF_COLLISION.json"
        ):
            raise VirtualR2Error(f"published R2 artifact path mismatch: {session_id}")
        if ref(state_path) != session["states"] or ref(collision_path) != session["collision"]["result"]:
            raise VirtualR2Error(f"published R2 state/collision ref drift: {session_id}")
        validate_r2_state_npz(state_path, int(session["frame_count"]))
        collision = load_json(collision_path)
        if (
            collision.get("session_id") != session_id
            or collision.get("scope") != "ROBOT_SELF_ONLY"
            or collision.get("coverage") != COLLISION_COVERAGE
            or collision.get("object_collision") != "UNVERIFIED"
            or collision.get("environment_collision") != "UNVERIFIED"
        ):
            raise VirtualR2Error(f"collision scope/identity mismatch: {session_id}")
        expected_video = VISUAL.resolve() / f"{session_id}_KAI22_R2_VIRTUAL_REVIEW.mp4"
        if ref(video_path) != row["review"]["video"] or video_path != expected_video:
            raise VirtualR2Error(f"review video path/ref mismatch: {session_id}")
        if row["review"].get("decoded_frames") != session["frame_count"] or row["review"].get("full_decode") != "PASS":
            raise VirtualR2Error(f"review video validation mismatch: {session_id}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--visual-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--executor-epoch", type=int, required=True)
    parser.add_argument("--fencing-token", required=True)
    args = parser.parse_args()
    packet, packet_path = validate_route()
    output, visual, receipt = args.output_root.resolve(), args.visual_root.resolve(), args.receipt.resolve()
    if output != OUTPUT.resolve() or visual != VISUAL.resolve() or receipt != TERMINAL_RECEIPT.resolve():
        raise VirtualR2Error("fixed virtual R2 output namespace mismatch")
    if args.executor_epoch < 1 or len(args.fencing_token) < 16:
        raise VirtualR2Error("invalid executor epoch/fencing token")
    for path in (output, visual, receipt):
        if path.exists() or path.is_symlink():
            raise VirtualR2Error(f"fresh output required: {path}")

    installation, _ = load_virtual_installation_contract()
    static_refs, mounts = verify_static_assets()
    assets = load_pinned_robot_assets(ROOT)
    asset_closure = verify_robot_asset_closure()
    batch_path = R0_ROOT / "BATCH_RESULT.json"
    batch = load_json(batch_path)
    rows = validate_r0_batch(batch)
    expected_joint_order = {
        side: [joint.name for joint in model.joints if joint.joint_type != "fixed"]
        for side, model in zip(SIDES, (assets.left_hand, assets.right_hand), strict=True)
    }
    prepared: list[tuple[Mapping[str, Any], dict[str, Any], dict[str, np.ndarray]]] = []
    dynamic_inputs = [ref(batch_path)]
    for row in rows:
        session_id = str(row["session_id"])
        r0_result_path = Path(str(row["result"]["path"])).resolve(strict=True)
        if ref(r0_result_path) != row["result"]:
            raise VirtualR2Error(f"R0 result ref drift: {session_id}")
        r0_result = load_json(r0_result_path)
        r0_npz_path = Path(str(r0_result["arrays"]["path"])).resolve(strict=True)
        if ref(r0_npz_path) != r0_result["arrays"]:
            raise VirtualR2Error(f"R0 array ref drift: {session_id}")
        arrays = load_r0_arrays(r0_npz_path)
        validate_r0_session(row, r0_result, arrays, expected_joint_order)
        prepared.append((row, r0_result, arrays))
        dynamic_inputs.extend((ref(r0_result_path), ref(r0_npz_path)))

    output.mkdir(parents=True)
    signature_payload = {
        "schema_version": "0915-robot15h-virtual-arm-wave0-signature-v1",
        "task_id": TASK_ID,
        "window_run_id": WINDOW_RUN_ID,
        "executor_epoch": args.executor_epoch,
        "weights": "ABSENT",
        "task_packet": ref(packet_path),
        "dynamic_inputs": dynamic_inputs,
        "dynamic_input_policy": DYNAMIC_INPUT_POLICY,
        "static_inputs": {
            **static_refs,
            "virtual_installation_schema": ref(VIRTUAL_INSTALLATION_SCHEMA),
            "virtual_installation_contract": ref(VIRTUAL_INSTALLATION_CONTRACT),
        },
        "robot_asset_closure": asset_closure,
        "code": [ref(path) for path in CODE_CLOSURE],
        "calibration": "ABSENT",
        "schemas": {
            "contract": ref(VIRTUAL_INSTALLATION_SCHEMA),
            "r0_npz": installation["npz_schemas"]["r0_input"],
            "r2_npz": installation["npz_schemas"]["r2_output"],
        },
        "motion_scale": 1.0,
        "source_policy": "NO_SOURCE_OR_PROCESSED_DATA_READ_R0_ONLY",
    }
    signature_sha = canonical_sha(signature_payload)
    atomic_json(output / "RUN_SIGNATURE.json", {**signature_payload, "run_signature_sha256": signature_sha})
    atomic_json(output / "CLAIM.json", {
        "schema_version": "0915-robot15h-virtual-arm-wave0-claim-v1",
        "task_id": TASK_ID,
        "window_run_id": WINDOW_RUN_ID,
        "attempt_id": output.name,
        "status": "CLAIMED",
        "weights": "ABSENT",
        "pid": os.getpid(),
        "proc_start_ticks": process_start_ticks(os.getpid()),
        "executor_epoch": args.executor_epoch,
        "fencing_token_sha256": hashlib.sha256(args.fencing_token.encode()).hexdigest(),
        "run_signature_sha256": signature_sha,
        "unique_write_root": str(output),
        "task_packet": ref(packet_path),
    })
    heartbeat("RUNNING")

    visual.mkdir(parents=True)
    results: list[dict[str, Any]] = []
    ordered = sorted(
        prepared,
        key=lambda row: (
            row[0]["session_id"] != "get_potato_chips_0915_042",
            row[0]["session_id"],
        ),
    )
    for row, r0_result, arrays in ordered:
        session_id = str(row["session_id"])
        solution = solve_virtual_session(arrays, assets, mounts)
        target = output / "sessions" / row["task"] / session_id
        staging = target.with_name(f".{target.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
        staging.mkdir(parents=True)
        state_path = staging / "KAI22_R2_VIRTUAL_ARM_V1.npz"
        atomic_npz(
            state_path,
            q_arm=solution["q_arm"],
            q22_init=solution["q22"],
            q_hand=solution["q22"],
            valid_side_frame=solution["valid"].T,
            T_world_base=solution["T_world_base"],
            T_tool_hand_root=solution["T_tool_hand_root_proxy"],
            T_target_hand_root_world=solution["T_target_hand_root_base"],
            T_actual_hand_root_world=solution["T_actual_hand_root_base"],
            position_residual_mm=solution["position_residual_mm"],
            rotation_residual_deg=solution["rotation_residual_deg"],
            numeric_pass=solution["numeric_pass"],
            ik_evaluations=solution["ik_evaluations"],
            timestamps_s=arrays["timestamps_s"],
            source_frames=arrays["frame_id"],
        )
        bilateral_count = int(np.count_nonzero(np.all(solution["valid"], axis=1)))
        if bilateral_count:
            collision = audit_self_collision(
                session_id=session_id,
                arm_path=state_path,
                hand_path=state_path,
                frame_count=bilateral_count,
                penetration_tolerance_m=PENETRATION_TOLERANCE_M,
            )
            collision_status = collision["status"]
            collision_checked_frames = bilateral_count
        else:
            collision = {
                "schema_version": "robot15h-r2-self-collision-unverified-v1",
                "session_id": session_id,
                "status": "UNVERIFIED_NO_BILATERAL_VALID_FRAME",
                "selected_frames": [],
                "illegal_contact_count": None,
                "claim_limit": "No bilateral-valid frame exists; no collision pass is claimed.",
            }
            collision_status = collision["status"]
            collision_checked_frames = 0
        any_valid_frames = int(np.count_nonzero(np.any(solution["valid"], axis=1)))
        collision_complete = collision_checked_frames == any_valid_frames
        collision.update({
            "scope": "ROBOT_SELF_ONLY",
            "coverage": COLLISION_COVERAGE,
            "checked_bilateral_frames": collision_checked_frames,
            "valid_frames_not_collision_checked": any_valid_frames - collision_checked_frames,
            "quality_eligible": collision_complete,
            "physical_collision_authority": False,
            "object_collision": "UNVERIFIED",
            "environment_collision": "UNVERIFIED",
        })
        # The auditor runs while the immutable session tree is staged.  Rewrite
        # only its artifact references to the final atomic-publish location so
        # no consumer is handed a path that disappears after ``os.replace``.
        if isinstance(collision.get("inputs"), dict):
            published_state = published_ref(state_path, target / state_path.name)
            collision["inputs"]["arm_states"] = published_state
            collision["inputs"]["hand_states"] = published_state
        atomic_json(staging / "ROBOT_SELF_COLLISION.json", collision)
        valid_rows = int(solution["valid"].sum())
        numeric_pass_rows = int(solution["numeric_pass"].sum())
        upstream_quality = r0_result.get("r0_quality_admitted") is True
        # The current W0 R0 strict gate rejected every session.  Numeric R2
        # candidates remain reviewable, but none can become an admitted R2.
        r2_quality_admitted = bool(
            upstream_quality
            and valid_rows > 0
            and numeric_pass_rows == valid_rows
            and solution["q22_joint_limits_pass"]
            and solution["arm_velocity_limits_pass"]
            and collision_status.startswith("PASS")
            and collision_complete
        )
        result = {
            "schema_version": "KAI22_R2_VIRTUAL_ARM_V1",
            "session_id": session_id,
            "task": row["task"],
            "status": "PASS_DEVELOPMENT_R2" if r2_quality_admitted else "REJECTED_QUALITY_DEVELOPMENT_CANDIDATE_ONLY",
            "authority": AUTHORITY,
            "r2_exported": True,
            "r2_numeric_candidate": numeric_pass_rows > 0,
            "r2_quality_admitted": r2_quality_admitted,
            "upstream_r0_quality_admitted": upstream_quality,
            "dynamic_input_policy": DYNAMIC_INPUT_POLICY,
            "r0_result": row["result"],
            "r0_arrays": r0_result["arrays"],
            "states": {
                "path": str((target / state_path.name).resolve()),
                "bytes": state_path.stat().st_size,
                "sha256": sha256(state_path),
            },
            "frame_count": len(arrays["timestamps_s"]),
            "valid_side_frames": valid_rows,
            "numeric_pass_side_frames": numeric_pass_rows,
            "numeric_rejected_side_frames": valid_rows - numeric_pass_rows,
            "q22_bit_exact_pass_through": solution["q22_bit_exact_pass_through"],
            "q22_joint_limits_pass": solution["q22_joint_limits_pass"],
            "motion_scale": 1.0,
            "motion_scaling_performed": False,
            "base_fixed": True,
            "base_pose": "IDENTITY_VIRTUAL_BASE_NOT_CAMERA_WORLD_CALIBRATION",
            "mount": {
                "status": "ASSUMED_DEVELOPMENT_PRIOR",
                "measured_installation_transform": "ABSENT",
                "contract": static_refs["mount_visual_proxy_contract"],
                "virtual_motion_authority": ref(VIRTUAL_INSTALLATION_CONTRACT),
            },
            "metrics": {
                "position_residual_p95_mm": float(np.nanpercentile(solution["position_residual_mm"], 95)) if valid_rows else None,
                "rotation_residual_p95_deg": float(np.nanpercentile(solution["rotation_residual_deg"], 95)) if valid_rows else None,
                "arm_motion": solution["arm_motion"],
                "kai22_motion": solution["hand_motion"],
                "arm_velocity_limits_pass": solution["arm_velocity_limits_pass"],
                "ik_function_evaluations": int(solution["ik_evaluations"].sum()),
            },
            "collision": {
                "scope": "ROBOT_SELF_ONLY",
                "coverage": COLLISION_COVERAGE,
                "quality_eligible": collision_complete,
                "robot_self_collision": collision_status,
                "checked_bilateral_frames": collision_checked_frames,
                "valid_frames_not_collision_checked": any_valid_frames - collision_checked_frames,
                "full_object_collision": "UNVERIFIED",
                "environment_collision": "UNVERIFIED",
                "observed_object_patch_collision": "NOT_APPLICABLE_NO_OBJECT_INPUT",
                "result": {
                    "path": str((target / "ROBOT_SELF_COLLISION.json").resolve()),
                    "bytes": (staging / "ROBOT_SELF_COLLISION.json").stat().st_size,
                    "sha256": sha256(staging / "ROBOT_SELF_COLLISION.json"),
                },
            },
            "control_ground_truth": False,
            "training_eligible": False,
            "physical_deployment_authorized": False,
            "calibration_authority": "DEVELOPMENT_ONLY",
            "user_visual_acceptance": "PENDING",
            "claim_limit": (
                "Independent fixed-base virtual arm IK driven only by R0 relative wrist and exact q22. "
                "The mount is an assumed visual proxy; this is not measured installation, camera/base "
                "calibration, object/environment collision, control, training, or deployment authority."
            ),
        }
        atomic_json(staging / "RESULT.json", result)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, target)
        review = render_virtual_review(
            visual / f"{session_id}_KAI22_R2_VIRTUAL_REVIEW.mp4",
            session_id,
            solution,
            assets,
            arrays["timestamps_s"],
            installation["mandatory_visual_labels"],
        )
        results.append({
            "session_id": session_id,
            "task": row["task"],
            "status": result["status"],
            "r2_exported": True,
            "r2_numeric_candidate": result["r2_numeric_candidate"],
            "r2_quality_admitted": result["r2_quality_admitted"],
            "valid_side_frames": valid_rows,
            "numeric_pass_side_frames": numeric_pass_rows,
            "result": ref(target / "RESULT.json"),
            "review": review,
        })
        heartbeat()

    review_manifest_path = output / "REVIEW_MANIFEST.json"
    atomic_json(review_manifest_path, {
        "schema_version": "0915-robot15h-virtual-arm-review-manifest-v1",
        "session_count": len(results),
        "reviews": [row["review"] for row in results],
        "required_video_count": 4,
        "all_full_decode": all(row["review"]["full_decode"] == "PASS" for row in results),
    })
    validate_published_artifacts(results, review_manifest_path)
    batch_result = {
        "schema_version": "0915-robot15h-kai22-r2-virtual-arm-wave0-batch-v1",
        "status": "COMPLETED_ALL_TERMINAL",
        "window_run_id": WINDOW_RUN_ID,
        "session_count": len(results),
        "r2_exported": sum(row["r2_exported"] for row in results),
        "r2_numeric_candidates": sum(row["r2_numeric_candidate"] for row in results),
        "r2_quality_admitted": sum(row["r2_quality_admitted"] for row in results),
        "r2_quality_rejected": len(results) - sum(row["r2_quality_admitted"] for row in results),
        "failed_runtime": 0,
        "results": results,
        "dynamic_input_policy": DYNAMIC_INPUT_POLICY,
        "virtual_installation_contract": ref(VIRTUAL_INSTALLATION_CONTRACT),
        "collision_coverage": COLLISION_COVERAGE,
        "control_ground_truth": False,
        "training_eligible": False,
        "physical_deployment_authorized": False,
    }
    atomic_json(output / "BATCH_RESULT.json", batch_result)
    lines = [
        "# 0915 Robot15h W0 Virtual R2 审阅",
        "",
        "这些视频是固定虚拟底座中的独立 Robot-space 审阅，不是 RGB 叠加。动态输入只有 R0 `q22_init`、相对 wrist、有效位和时间戳。",
        "",
        "固定安装关系仅为 `ASSUMED_DEVELOPMENT_PRIOR` visual proxy；实测安装、TCP、camera/world→base 都不存在。q22 原样传递，无效帧保持 NaN，相对腕部位移没有缩放。",
        "",
    ]
    for row in results:
        name = Path(row["review"]["video"]["path"]).name
        lines.append(
            f"- [{row['session_id']}]({name})：numeric candidate `{row['r2_numeric_candidate']}`，"
            f"质量准入 `{row['r2_quality_admitted']}`，numeric pass side-frame "
            f"`{row['numeric_pass_side_frames']}/{row['valid_side_frames']}`。"
        )
    lines.extend([
        "",
        "由于 W0 的 R0 strict quality 全部拒绝，本轮 `r2_quality_admitted=0`；可视化和 IK 数值候选不能升级为 Robot 成功。碰撞仅为双侧有效帧上的 proxy-space Robot 自碰撞诊断，单侧帧不检查且不可质量准入；完整物体与环境均为 `UNVERIFIED`。",
        "",
    ])
    (visual / "README_ZH.md").write_text("\n".join(lines), encoding="utf-8")
    result = {
        "schema_version": "0915-robot15h-virtual-arm-wave0-result-v1",
        "task_id": TASK_ID,
        "window_run_id": WINDOW_RUN_ID,
        "status": "PASSED",
        "weights": "ABSENT",
        "batch_terminal": "COMPLETED_ALL_TERMINAL",
        "counts": {
            "attempted": len(results),
            "r2_exported": batch_result["r2_exported"],
            "r2_numeric_candidates": batch_result["r2_numeric_candidates"],
            "r2_quality_admitted": batch_result["r2_quality_admitted"],
            "r2_quality_rejected": batch_result["r2_quality_rejected"],
            "failed_runtime": 0,
        },
        "batch_result": ref(output / "BATCH_RESULT.json"),
        "review_manifest": ref(review_manifest_path),
        "virtual_installation_contract": ref(VIRTUAL_INSTALLATION_CONTRACT),
        "visual_readme": ref(visual / "README_ZH.md"),
        "dynamic_input_policy": DYNAMIC_INPUT_POLICY,
        "source_mutated": False,
        "processed_mutated": False,
        "control_ground_truth": False,
        "training_eligible": False,
        "physical_deployment_authorized": False,
        "user_visual_acceptance": "PENDING",
        "claim_limit": packet["claim_limit"],
    }
    atomic_json(output / "RESULT.json", result)
    atomic_json(receipt, {**result, "result": ref(output / "RESULT.json")})
    atomic_json(output / "RUN_RECEIPT.json", {
        "schema_version": "0915-robot15h-virtual-arm-wave0-run-receipt-v1",
        "task_id": TASK_ID,
        "window_run_id": WINDOW_RUN_ID,
        "status": "PASSED",
        "result": ref(output / "RESULT.json"),
        "terminal_receipt": ref(receipt),
    })
    heartbeat()
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
