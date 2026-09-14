"""Candidate-only EEVEE-Next renderer for Tianji arms plus both KaiHands.

The module reconstructs the deleted S7 visual path from the retained URDF and
buffer contract.  It never solves IK, changes a mount, reads collision meshes,
or places a hand from a renderer-only override.  Geometry is driven only by a
validated scene-state bundle:

``T_camera_base @ FK_Tianji(q_arm) @ T_tool_hand @ FK_KaiHand(q_hand)``.

All mount-bearing outputs are visual candidates and must carry
``PROVISIONAL_MOUNT_VISUAL_ONLY``.  The physical contact field is always
``UNMEASURED``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import io
import math
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
from typing import Any, Mapping, Sequence

import numpy as np

from pipeline.robot_mount_visual_proxy import CONTACT_INFEASIBLE, MOUNT_PROVENANCE
from pipeline.robot_renderer_cycles import (
    CV_CAMERA_TO_BLENDER,
    JointSpec,
    RendererError,
    UrdfModel,
    VisualSpec,
    _vector,
    forward_kinematics,
    origin_matrix,
)
from pipeline.robot_tool_definition_contract import (
    load_pinned_tool_definition,
    require_same_tool_definition,
)


ENGINE = "BLENDER_EEVEE_NEXT"
SHARED_IO_RELATIVE = Path("tools/immutable_artifact_io.py")
SHARED_IO_SHA256 = "c0cdcea8b541083de65b2ccd7d1011305e2000bbf49694eef601599d7fbcb5de"
ASSET_PIN_RELATIVE = Path("assets/robot/ROBOT_ASSET_PIN.json")
EGL_VENDOR_RELATIVE = Path("tools/glvnd/10_nvidia.json")
SCENE_SCHEMA = "robot-fullchain-scene-state-v1"
OUTPUT_SCHEMA = "s7-eevee-next-fullchain-candidate-v1"
SCENE_MANIFEST_DEVELOPMENT_STATUS = "DEVELOPMENT_ONLY_ARTIFACT_EXISTS"
SCENE_MANIFEST_VISUAL_CANDIDATE_STATUS = "VISUAL_ONLY_CANDIDATE_ARTIFACT_EXISTS"
SCENE_STATE_MODE_DEVELOPMENT_SOLVER = "DEVELOPMENT_R2_WRIST_IK_SOLVER"
SCENE_STATE_MODE_FIXED_CAMERA_BASE_SOLVER = (
    "DEVELOPMENT_FIXED_CAMERA_BASE_POSITION_THEN_FULL_POSE_R2_WRIST_IK_SOLVER"
)
FIXED_BASE_TWO_STAGE_SOLVER_SCHEMA = (
    "robot-scene-state-cpu-fixed-base-position-then-full-pose-solver-v1"
)
FIXED_BASE_POSITION_STAGE_MAX_EVALUATIONS = 300
FIXED_BASE_FULL_POSE_STAGE_MAX_EVALUATIONS = 300
FIXED_BASE_IK_TRF_FTOL = 1e-10
FIXED_BASE_IK_TRF_XTOL = 1e-10
FIXED_BASE_IK_TRF_GTOL = 1e-10
FIXED_BASE_POSITION_SCALE_M = 0.005
FIXED_BASE_ROTATION_SCALE_RAD = float(np.deg2rad(2.0))
FIXED_BASE_GATE_POSITION_THRESHOLD_MM = 10.0
FIXED_BASE_GATE_ROTATION_THRESHOLD_DEG = 5.0
FIXED_BASE_RESIDUAL_MATCH_ATOL_MM = 1e-6
FIXED_BASE_RESIDUAL_MATCH_ATOL_DEG = 1e-5
SCENE_STATE_MODE_EXTERNAL_AUTHORITY = "EXTERNAL_BASE_Q_AUTHORITY"
FIXED_CAMERA_BASE_CANDIDATE_SCHEMA = "robot-fixed-camera-base-development-candidate-v1"
EXTERNAL_BASE_Q_SCHEMA = "robot-explicit-base-q-authority-v1"
EXTERNAL_BASE_EVIDENCE_SCHEMA = "robot-external-camera-base-evidence-v1"
EXTERNAL_Q_ARM_SIDE_EVIDENCE_SCHEMA = "robot-external-q-arm-side-evidence-v1"
EXTERNAL_AUTHORITY_EVIDENCE_KIND = "DIRECT_EXTERNAL_BASE_AND_ARM_STATE_MEASUREMENT"
EXTERNAL_AUTHORITY_METHOD = "DIRECT_CAMERA_BASE_CALIBRATION_AND_JOINT_ENCODER_STATE"
EXTERNAL_FRAME_TIMESTAMP_MAPPING = (
    "ONE_TIMESTAMP_PER_ZERO_BASED_SOURCE_FRAME_NO_INTERPOLATION"
)
EXTERNAL_BASE_EVIDENCE_KIND = "DIRECT_CAMERA_TO_ROBOT_BASE_CALIBRATION"
EXTERNAL_Q_ARM_EVIDENCE_KIND = "DIRECT_ARM_JOINT_ENCODER_STATE"
EXTERNAL_ARRAY_DIGEST_CANONICALIZATION = (
    "NUMPY_DTYPE_STR_NUL_SHAPE_JSON_NUL_C_ORDER_BYTES_V1"
)
MOUNT_TIME_SCOPE = "SESSION_CONSTANT_FULL_SESSION"
MOUNT_COORDINATE_DEFINITION = (
    "Tianji left/right tool link to corresponding KaiHand root link"
)
MOUNT_MATRIX_DIRECTION = "p_target_link = T_tool_hand @ p_source_link"
MOUNT_SOURCE_LINKS = {
    "left": "hand_l_base_link",
    "right": "hand_r_base_link",
}
MOUNT_TARGET_LINKS = {"left": "left_tool", "right": "right_tool"}
RIGHT_HANDED_AXES = "RIGHT_HANDED_XYZ"
ARM_JOINT_NAMES = tuple(
    tuple(f"Joint{index}_{suffix}" for index in range(1, 8)) for suffix in ("L", "R")
)
SIDES = ("left", "right")


class FullChainRendererError(RendererError):
    pass


@dataclass(frozen=True)
class PinnedRobotAssets:
    project_root: Path
    asset_pin_record: Mapping[str, Any]
    tianji: UrdfModel
    left_hand: UrdfModel
    right_hand: UrdfModel
    mesh_records: Mapping[str, Any]
    tool_definition: Mapping[str, Any]


@dataclass(frozen=True)
class FullChainSceneState:
    session_id: str
    frame_names: tuple[str, ...]
    q_arm: np.ndarray
    q_hand: np.ndarray
    valid: np.ndarray
    wrist_T_camera: np.ndarray
    T_camera_base: np.ndarray
    T_tool_hand: np.ndarray
    camera_intrinsics: np.ndarray
    source_resolution: tuple[int, int]
    arm_joint_names: tuple[tuple[str, ...], tuple[str, ...]]
    hand_joint_names: tuple[tuple[str, ...], tuple[str, ...]]
    scene_state_mode: str
    solver_used: bool
    residual_claimed: bool
    position_residual_mm: np.ndarray | None
    rotation_residual_deg: np.ndarray | None
    ik_function_evaluations: np.ndarray | None
    ik_position_stage_function_evaluations: np.ndarray | None
    ik_full_pose_stage_function_evaluations: np.ndarray | None
    versioned_contract: bool
    mount_descriptor_sha256: str | None
    mount_evidence_mode: str | None
    mount_evidence_sha256_by_side: tuple[str, str] | None
    mount_authority_lineage_sha256: str | None
    timestamp_ns: np.ndarray | None
    external_authority_schema: str | None
    external_authority_descriptor_sha256: str | None
    external_authority_arrays_sha256: str | None
    external_authority_lineage_sha256: str | None
    external_base_evidence_sha256: str | None
    external_q_arm_evidence_sha256_by_side: tuple[str, str] | None
    external_urdf_sha256: tuple[str, str] | None
    external_source_sha256: tuple[str, str] | None
    fixed_camera_base_candidate_schema: str | None
    fixed_camera_base_candidate_descriptor_sha256: str | None
    fixed_camera_base_candidate_lineage_sha256: str | None


@dataclass(frozen=True)
class PlacedVisual:
    component: str
    role: str
    side: str | None
    link: str
    mesh_path: Path
    mesh_relative: str
    transform_blender: np.ndarray
    rgba: tuple[float, float, float, float]
    object_index: int


@dataclass(frozen=True)
class FramePlacement:
    frame_index: int
    frame_name: str
    visuals: tuple[PlacedVisual, ...]
    valid_by_side: tuple[bool, bool]
    hand_root_residual_by_side: Mapping[str, Mapping[str, float] | None]


def _bootstrap_sha256(path: Path) -> str:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def load_strict_io(project_root: Path) -> Any:
    path = project_root.resolve() / SHARED_IO_RELATIVE
    observed = _bootstrap_sha256(path)
    if observed != SHARED_IO_SHA256:
        raise FullChainRendererError(f"shared immutable I/O SHA drift: {observed}")
    name = "s7_immutable_artifact_io"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise FullChainRendererError("cannot load shared immutable I/O")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def resolve_mesh_path(project_root: Path, urdf_path: Path, filename: str) -> Path:
    if filename.startswith("package://marvin_description/"):
        relative = filename.removeprefix("package://marvin_description/")
        candidate = project_root / "assets/robot/tianji/marvin_description" / relative
    elif filename.startswith("package://"):
        raise FullChainRendererError(f"unsupported package URI: {filename}")
    else:
        candidate = urdf_path.parent / filename
    resolved = candidate.resolve(strict=True)
    robot_root = (project_root / "assets/robot").resolve(strict=True)
    if resolved != robot_root and robot_root not in resolved.parents:
        raise FullChainRendererError(f"visual mesh escapes robot assets: {filename}")
    return resolved


def parse_urdf_payload(project_root: Path, path: Path, payload: bytes) -> UrdfModel:
    """Parse only visual geometry from the exact already-verified URDF bytes."""

    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:
        raise FullChainRendererError(f"invalid URDF {path}: {exc}") from exc
    links = tuple(element.get("name", "") for element in root.findall("link"))
    if not links or any(not name for name in links) or len(set(links)) != len(links):
        raise FullChainRendererError(f"invalid or duplicate links in {path}")
    visuals: list[VisualSpec] = []
    for link_element in root.findall("link"):
        link_name = str(link_element.get("name", ""))
        for visual in link_element.findall("visual"):
            mesh = visual.find("geometry/mesh")
            if mesh is None or not mesh.get("filename"):
                raise FullChainRendererError(
                    f"unsupported visual geometry: {path}:{link_name}"
                )
            mesh_path = resolve_mesh_path(project_root, path, str(mesh.get("filename")))
            scale = _vector(mesh.get("scale"), 3, (1, 1, 1))
            visual_origin = origin_matrix(visual.find("origin"))
            visual_origin[:3, :3] = visual_origin[:3, :3] @ np.diag(scale)
            color = visual.find("material/color")
            rgba = _vector(
                color.get("rgba") if color is not None else None,
                4,
                (0.7, 0.72, 0.76, 1.0),
            )
            visuals.append(
                VisualSpec(
                    link=link_name,
                    mesh_path=mesh_path,
                    origin=visual_origin,
                    rgba=tuple(float(value) for value in rgba),
                )
            )
    joints: list[JointSpec] = []
    children: set[str] = set()
    for element in root.findall("joint"):
        name, joint_type = str(element.get("name", "")), str(element.get("type", ""))
        parent, child = element.find("parent"), element.find("child")
        if not name or parent is None or child is None:
            raise FullChainRendererError(f"malformed joint in {path}")
        parent_name, child_name = (
            str(parent.get("link", "")),
            str(child.get("link", "")),
        )
        if (
            parent_name not in links
            or child_name not in links
            or child_name in children
        ):
            raise FullChainRendererError(f"invalid joint topology: {name}")
        children.add(child_name)
        axis_element = element.find("axis")
        axis = _vector(
            axis_element.get("xyz") if axis_element is not None else None,
            3,
            (1, 0, 0),
        )
        limit = element.find("limit")
        lower = (
            float(limit.get("lower"))
            if limit is not None and limit.get("lower")
            else None
        )
        upper = (
            float(limit.get("upper"))
            if limit is not None and limit.get("upper")
            else None
        )
        if joint_type not in {"fixed", "revolute", "continuous", "prismatic"}:
            raise FullChainRendererError(f"unsupported joint type {joint_type}: {name}")
        joints.append(
            JointSpec(
                name=name,
                joint_type=joint_type,
                parent=parent_name,
                child=child_name,
                origin=origin_matrix(element.find("origin")),
                axis=axis,
                lower=lower,
                upper=upper,
            )
        )
    roots = sorted(set(links) - children)
    if len(roots) != 1:
        raise FullChainRendererError(f"URDF must have one root: {roots}")
    return UrdfModel(path, roots[0], links, tuple(joints), tuple(visuals))


def load_pinned_robot_assets(
    project_root: Path, strict_io: Any | None = None
) -> PinnedRobotAssets:
    project = project_root.resolve()
    strict = strict_io or load_strict_io(project)
    pin_path = project / ASSET_PIN_RELATIVE
    pin, pin_record = strict.read_json_nofollow(pin_path, allowed_root=project)
    if pin.get("status") != "T0_IDENTITY_PIN_NOT_EXECUTION_AUTHORIZATION":
        raise FullChainRendererError("unexpected robot asset pin status")
    entries = {str(row["path"]): row for row in pin.get("files", [])}
    urdf_rows = list(pin.get("urdfs", []))
    if len(urdf_rows) != 3:
        raise FullChainRendererError(
            "robot pin must contain exactly three URDF components"
        )
    models: dict[str, UrdfModel] = {}
    for row in urdf_rows:
        relative = str(row["path"])
        expected = entries.get(relative)
        if expected is None:
            raise FullChainRendererError(f"URDF absent from file pin: {relative}")
        path = project / relative
        verified = strict.read_bytes_nofollow(
            path,
            expected_sha256=str(expected["sha256"]),
            expected_bytes=int(expected["bytes"]),
            allowed_root=project,
        )
        model = parse_urdf_payload(project, path, verified.payload)
        if model.root_link == "base_link":
            key = "tianji"
        elif model.root_link == "hand_l_base_link":
            key = "left_hand"
        elif model.root_link == "hand_r_base_link":
            key = "right_hand"
        else:
            raise FullChainRendererError(f"unknown pinned URDF root: {model.root_link}")
        if key in models:
            raise FullChainRendererError(f"duplicate robot component: {key}")
        models[key] = model
    if set(models) != {"tianji", "left_hand", "right_hand"}:
        raise FullChainRendererError("pinned robot component closure is incomplete")

    mesh_records: dict[str, Any] = {}
    for model in models.values():
        for visual in model.visuals:
            relative = visual.mesh_path.relative_to(project).as_posix()
            expected = entries.get(relative)
            if expected is None:
                raise FullChainRendererError(
                    f"visual mesh absent from asset pin: {relative}"
                )
            if relative not in mesh_records:
                mesh_records[relative] = strict.read_bytes_nofollow(
                    visual.mesh_path,
                    expected_sha256=str(expected["sha256"]),
                    expected_bytes=int(expected["bytes"]),
                    allowed_root=project,
                )
    if (
        len(mesh_records) != 63
        or sum(len(model.visuals) for model in models.values()) != 63
    ):
        raise FullChainRendererError(
            "render visual closure must be exactly 63 pinned meshes"
        )
    tool_definition = load_pinned_tool_definition(project).as_manifest()
    return PinnedRobotAssets(
        project_root=project,
        asset_pin_record=pin_record.evidence_ref(),
        tianji=models["tianji"],
        left_hand=models["left_hand"],
        right_hand=models["right_hand"],
        mesh_records=mesh_records,
        tool_definition=tool_definition,
    )


def _scalar_string(arrays: Mapping[str, np.ndarray], name: str) -> str:
    if name not in arrays or np.asarray(arrays[name]).shape != ():
        raise FullChainRendererError(f"scene state scalar missing: {name}")
    return str(np.asarray(arrays[name]).item())


def _scalar_bool(arrays: Mapping[str, np.ndarray], name: str) -> bool:
    if name not in arrays:
        raise FullChainRendererError(f"scene state scalar missing: {name}")
    value = np.asarray(arrays[name])
    if value.shape != () or value.dtype != np.bool_:
        raise FullChainRendererError(f"scene state {name} must be an exact boolean")
    return bool(value.item())


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


EVIDENCE_REF_KEYS = frozenset({"path", "bytes", "sha256", "device", "inode"})


def _manifest_evidence_ref(
    value: Any, *, name: str, allow_extra: bool = False
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise FullChainRendererError(f"{name} evidence ref must be an object")
    if (
        not allow_extra and set(value) != EVIDENCE_REF_KEYS
    ) or not EVIDENCE_REF_KEYS.issubset(value):
        raise FullChainRendererError(f"{name} evidence ref keys are not exact")
    path_value = value.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise FullChainRendererError(f"{name} evidence path is missing")
    path = Path(path_value)
    if (
        not path.is_absolute()
        or str(path) != path_value
        or ".." in path.parts
        or "." in path.parts
    ):
        raise FullChainRendererError(
            f"{name} evidence path must be absolute and canonical"
        )
    if (
        type(value.get("bytes")) is not int
        or int(value["bytes"]) <= 0
        or not _is_sha256(value.get("sha256"))
        or type(value.get("device")) is not int
        or int(value["device"]) < 0
        or type(value.get("inode")) is not int
        or int(value["inode"]) <= 0
    ):
        raise FullChainRendererError(
            f"{name} evidence bytes/SHA/device/inode are invalid"
        )
    return {key: value[key] for key in EVIDENCE_REF_KEYS}


def _require_manifest_direct_flags(value: Mapping[str, Any], *, name: str) -> None:
    fixed = {
        "independent_of_r2_wrist_targets": True,
        "derived_from_r2_wrist_targets": False,
        "derived_from_ik_solver": False,
        "selected_by_ik_residual": False,
        "interpolated_or_filled": False,
        "synthetic_fixture": False,
        "identity_fixture": False,
        "formal_consumer_allowed": False,
    }
    for field, expected in fixed.items():
        if value.get(field) is not expected:
            raise FullChainRendererError(f"{name} {field} must be {expected}")


def _canonical_json_sha256(value: Any) -> str:
    import json

    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_se3(value: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise FullChainRendererError(f"{name} must be one finite 4x4 transform")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-9, rtol=0):
        raise FullChainRendererError(f"{name} homogeneous row is invalid")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0):
        raise FullChainRendererError(f"{name} rotation is not orthonormal")
    if abs(float(np.linalg.det(rotation)) - 1.0) > 1e-6:
        raise FullChainRendererError(f"{name} rotation determinant is not +1")
    return matrix


def decode_scene_state(payload: bytes) -> FullChainSceneState:
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
    except (OSError, ValueError, KeyError) as exc:
        raise FullChainRendererError(f"invalid scene-state NPZ: {exc}") from exc
    if _scalar_string(arrays, "schema_version") != SCENE_SCHEMA:
        raise FullChainRendererError("unsupported scene-state schema")
    session_id = _scalar_string(arrays, "session_id")
    lowered = session_id.lower()
    if "025" in lowered or "blind" in lowered:
        raise FullChainRendererError("blind/025 scene state is forbidden")
    if _scalar_string(arrays, "mount_provenance") != MOUNT_PROVENANCE:
        raise FullChainRendererError(
            "scene state lacks provisional visual-only mount provenance"
        )
    if _scalar_string(arrays, "contact_infeasible") != CONTACT_INFEASIBLE:
        raise FullChainRendererError("contact_infeasible must remain UNMEASURED")
    versioned_contract = "scene_state_mode" in arrays
    if versioned_contract:
        scene_state_mode = _scalar_string(arrays, "scene_state_mode")
        solver_used = _scalar_bool(arrays, "solver_used")
        residual_claimed = _scalar_bool(arrays, "residual_claimed")
    else:
        # Pre-authority scene states are the development IK-solver contract.
        scene_state_mode = SCENE_STATE_MODE_DEVELOPMENT_SOLVER
        solver_used = True
        residual_claimed = True
    if scene_state_mode not in {
        SCENE_STATE_MODE_DEVELOPMENT_SOLVER,
        SCENE_STATE_MODE_FIXED_CAMERA_BASE_SOLVER,
        SCENE_STATE_MODE_EXTERNAL_AUTHORITY,
    }:
        raise FullChainRendererError("unsupported scene-state mode")
    development_solver = scene_state_mode in {
        SCENE_STATE_MODE_DEVELOPMENT_SOLVER,
        SCENE_STATE_MODE_FIXED_CAMERA_BASE_SOLVER,
    }
    fixed_camera_base = scene_state_mode == SCENE_STATE_MODE_FIXED_CAMERA_BASE_SOLVER
    if development_solver:
        if not solver_used or not residual_claimed:
            raise FullChainRendererError(
                "development solver mode must record solver/residual use"
            )
        forbidden_external = {
            "external_authority_schema",
            "external_authority_descriptor_sha256",
            "external_authority_arrays_sha256",
            "timestamp_ns",
        }
        if forbidden_external & set(arrays):
            raise FullChainRendererError(
                "development solver state cannot carry external authority arrays"
            )
    else:
        if solver_used or residual_claimed:
            raise FullChainRendererError(
                "external authority mode must record solver_not_used/residual_not_claimed"
            )
    versioned_common = {
        "schema_version",
        "solver_schema",
        "session_id",
        "mount_provenance",
        "contact_infeasible",
        "scene_state_mode",
        "solver_used",
        "residual_claimed",
        "mount_descriptor_sha256",
        "mount_evidence_mode",
        "mount_evidence_sha256_by_side",
        "mount_authority_lineage_sha256",
        "frame_names",
        "q_arm",
        "q_hand",
        "valid",
        "wrist_T_camera",
        "T_camera_base",
        "T_tool_hand",
        "camera_intrinsics",
        "source_resolution",
        "arm_joint_names",
        "hand_joint_names",
        "position_residual_mm",
        "rotation_residual_deg",
        "ik_function_evaluations",
    }
    versioned_external = {
        "external_authority_schema",
        "external_authority_descriptor_sha256",
        "external_authority_arrays_sha256",
        "external_authority_lineage_sha256",
        "external_base_evidence_sha256",
        "external_q_arm_evidence_sha256_by_side",
        "external_urdf_sha256",
        "external_source_sha256",
        "timestamp_ns",
    }
    versioned_fixed_camera_base = {
        "fixed_camera_base_candidate_schema",
        "fixed_camera_base_candidate_descriptor_sha256",
        "fixed_camera_base_candidate_lineage_sha256",
        "ik_position_stage_function_evaluations",
        "ik_full_pose_stage_function_evaluations",
    }
    if versioned_contract:
        expected_keys = versioned_common.copy()
        if scene_state_mode == SCENE_STATE_MODE_EXTERNAL_AUTHORITY:
            expected_keys |= versioned_external
        elif fixed_camera_base:
            expected_keys |= versioned_fixed_camera_base
        if set(arrays) != expected_keys:
            raise FullChainRendererError("versioned scene-state arrays are not exact")
        expected_solver_schema = (
            FIXED_BASE_TWO_STAGE_SOLVER_SCHEMA
            if fixed_camera_base
            else "robot-scene-state-cpu-solver-v1"
        )
        if _scalar_string(arrays, "solver_schema") != expected_solver_schema:
            raise FullChainRendererError("unsupported scene-state solver schema")
    required = {
        "frame_names",
        "q_arm",
        "q_hand",
        "valid",
        "wrist_T_camera",
        "T_camera_base",
        "T_tool_hand",
        "camera_intrinsics",
        "source_resolution",
        "arm_joint_names",
        "hand_joint_names",
    }
    if missing := sorted(required - set(arrays)):
        raise FullChainRendererError(f"scene-state arrays missing: {missing}")
    frame_names = tuple(str(value) for value in arrays["frame_names"])
    count = len(frame_names)
    if count <= 0 or len(set(frame_names)) != count:
        raise FullChainRendererError("scene frame names must be non-empty and unique")
    mount_descriptor_sha256: str | None = None
    mount_evidence_mode: str | None = None
    mount_evidence_sha256_by_side: tuple[str, str] | None = None
    mount_authority_lineage_sha256: str | None = None
    if versioned_contract:
        mount_descriptor_sha256 = _scalar_string(arrays, "mount_descriptor_sha256")
        mount_evidence_mode = _scalar_string(arrays, "mount_evidence_mode")
        mount_authority_lineage_sha256 = _scalar_string(
            arrays, "mount_authority_lineage_sha256"
        )
        evidence_sha = np.asarray(arrays["mount_evidence_sha256_by_side"])
        if evidence_sha.shape != (2,) or evidence_sha.dtype.kind not in {"U", "S"}:
            raise FullChainRendererError(
                "mount evidence SHA lineage must be an exact two-string array"
            )
        mount_evidence_sha256_by_side = tuple(str(item) for item in evidence_sha)  # type: ignore[assignment]
        if not _is_sha256(mount_descriptor_sha256) or not _is_sha256(
            mount_authority_lineage_sha256
        ):
            raise FullChainRendererError("mount descriptor SHA lineage is invalid")
        if mount_evidence_mode == "DEVELOPMENT_ONLY_SYNTHETIC_FIXTURE":
            if mount_evidence_sha256_by_side != ("", ""):
                raise FullChainRendererError(
                    "synthetic mount state cannot carry measured evidence SHA"
                )
        elif mount_evidence_mode == "INDEPENDENT_BILATERAL_VISUAL_EVIDENCE":
            if not all(_is_sha256(item) for item in mount_evidence_sha256_by_side):
                raise FullChainRendererError(
                    "measured mount state lacks bilateral evidence SHA lineage"
                )
            if mount_evidence_sha256_by_side[0] == mount_evidence_sha256_by_side[1]:
                raise FullChainRendererError(
                    "measured mount side evidence SHA values must be independent"
                )
        else:
            raise FullChainRendererError("unsupported mount evidence mode in state")
    timestamp_ns: np.ndarray | None = None
    external_schema: str | None = None
    external_descriptor_sha256: str | None = None
    external_arrays_sha256: str | None = None
    external_authority_lineage_sha256: str | None = None
    external_base_evidence_sha256: str | None = None
    external_q_arm_evidence_sha256_by_side: tuple[str, str] | None = None
    external_urdf_sha256: tuple[str, str] | None = None
    external_source_sha256: tuple[str, str] | None = None
    fixed_camera_base_candidate_schema: str | None = None
    fixed_camera_base_candidate_descriptor_sha256: str | None = None
    fixed_camera_base_candidate_lineage_sha256: str | None = None
    if scene_state_mode == SCENE_STATE_MODE_EXTERNAL_AUTHORITY:
        external_schema = _scalar_string(arrays, "external_authority_schema")
        external_descriptor_sha256 = _scalar_string(
            arrays, "external_authority_descriptor_sha256"
        )
        external_arrays_sha256 = _scalar_string(
            arrays, "external_authority_arrays_sha256"
        )
        external_authority_lineage_sha256 = _scalar_string(
            arrays, "external_authority_lineage_sha256"
        )
        if external_schema != EXTERNAL_BASE_Q_SCHEMA:
            raise FullChainRendererError("unsupported external base/q authority schema")
        if (
            not _is_sha256(external_descriptor_sha256)
            or not _is_sha256(external_arrays_sha256)
            or not _is_sha256(external_authority_lineage_sha256)
        ):
            raise FullChainRendererError(
                "external authority scene joins require exact lowercase SHA-256"
            )
        external_base_evidence_sha256 = _scalar_string(
            arrays, "external_base_evidence_sha256"
        )
        if not _is_sha256(external_base_evidence_sha256):
            raise FullChainRendererError(
                "external camera/base evidence SHA lineage is invalid"
            )

        def two_sha(name: str) -> tuple[str, str]:
            value = np.asarray(arrays[name])
            if value.shape != (2,) or value.dtype.kind not in {"U", "S"}:
                raise FullChainRendererError(f"{name} must be an exact two-SHA array")
            result = tuple(str(item) for item in value)
            if not all(_is_sha256(item) for item in result) or result[0] == result[1]:
                raise FullChainRendererError(
                    f"{name} must contain two distinct lowercase SHA-256 values"
                )
            return result  # type: ignore[return-value]

        external_q_arm_evidence_sha256_by_side = two_sha(
            "external_q_arm_evidence_sha256_by_side"
        )
        external_urdf_sha256 = two_sha("external_urdf_sha256")
        external_source_sha256 = two_sha("external_source_sha256")
        timestamp_ns = np.asarray(arrays.get("timestamp_ns"))
        if timestamp_ns.dtype != np.int64 or timestamp_ns.shape != (count,):
            raise FullChainRendererError(
                "external authority timestamp_ns must be exact int64[frame_count]"
            )
        if np.any(timestamp_ns < 0) or np.any(np.diff(timestamp_ns) <= 0):
            raise FullChainRendererError(
                "external authority timestamps must be non-negative and increasing"
            )
    elif fixed_camera_base:
        fixed_camera_base_candidate_schema = _scalar_string(
            arrays, "fixed_camera_base_candidate_schema"
        )
        fixed_camera_base_candidate_descriptor_sha256 = _scalar_string(
            arrays, "fixed_camera_base_candidate_descriptor_sha256"
        )
        fixed_camera_base_candidate_lineage_sha256 = _scalar_string(
            arrays, "fixed_camera_base_candidate_lineage_sha256"
        )
        if (
            fixed_camera_base_candidate_schema != FIXED_CAMERA_BASE_CANDIDATE_SCHEMA
            or not _is_sha256(fixed_camera_base_candidate_descriptor_sha256)
            or not _is_sha256(fixed_camera_base_candidate_lineage_sha256)
        ):
            raise FullChainRendererError(
                "fixed camera-base candidate scene lineage is invalid"
            )
    q_arm_raw = np.asarray(arrays["q_arm"])
    q_hand_raw = np.asarray(arrays["q_hand"])
    valid_raw = np.asarray(arrays["valid"])
    wrists_raw = np.asarray(arrays["wrist_T_camera"])
    if versioned_contract and (
        q_arm_raw.dtype != np.float64
        or q_hand_raw.dtype != np.float64
        or valid_raw.dtype != np.bool_
        or wrists_raw.dtype != np.float64
    ):
        raise FullChainRendererError(
            "versioned scene q/valid/wrist arrays must keep exact dtypes"
        )
    q_arm = np.asarray(q_arm_raw, dtype=np.float64)
    q_hand = np.asarray(q_hand_raw, dtype=np.float64)
    valid = np.asarray(valid_raw, dtype=bool)
    wrists = np.asarray(wrists_raw, dtype=np.float64)
    intrinsics = np.asarray(arrays["camera_intrinsics"], dtype=np.float64)
    if q_arm.shape != (count, 2, 7) or q_hand.shape != (count, 2, 22):
        raise FullChainRendererError("scene q_arm/q_hand shape mismatch")
    if valid.shape != (count, 2) or wrists.shape != (count, 2, 4, 4):
        raise FullChainRendererError("scene valid/wrist shape mismatch")
    if intrinsics.shape != (count, 4) or not np.isfinite(intrinsics).all():
        raise FullChainRendererError("scene camera intrinsics shape/value mismatch")
    if np.any(intrinsics[:, :2] <= 0):
        raise FullChainRendererError("camera focal lengths must be positive")
    source_resolution_array = np.asarray(arrays["source_resolution"])
    if source_resolution_array.shape != (2,) or not np.issubdtype(
        source_resolution_array.dtype, np.integer
    ):
        raise FullChainRendererError("source_resolution must be integer [width,height]")
    source_resolution = tuple(int(value) for value in source_resolution_array)
    if min(source_resolution) <= 0:
        raise FullChainRendererError("source_resolution must be positive")
    if not np.isfinite(q_arm[valid]).all() or not np.isfinite(q_hand[valid]).all():
        raise FullChainRendererError("valid side contains non-finite q")
    position_residual_mm: np.ndarray | None = None
    rotation_residual_deg: np.ndarray | None = None
    ik_function_evaluations: np.ndarray | None = None
    ik_position_stage_function_evaluations: np.ndarray | None = None
    ik_full_pose_stage_function_evaluations: np.ndarray | None = None
    if versioned_contract:
        position_raw = np.asarray(arrays["position_residual_mm"])
        rotation_raw = np.asarray(arrays["rotation_residual_deg"])
        evaluation_raw = np.asarray(arrays["ik_function_evaluations"])
        if (
            position_raw.dtype != np.float64
            or rotation_raw.dtype != np.float64
            or evaluation_raw.dtype != np.int32
            or position_raw.shape != (count, 2)
            or rotation_raw.shape != (count, 2)
            or evaluation_raw.shape != (count, 2)
        ):
            raise FullChainRendererError(
                "scene residual/evaluation arrays have wrong dtype or shape"
            )
        position_residual_mm = position_raw
        rotation_residual_deg = rotation_raw
        ik_function_evaluations = evaluation_raw
        if scene_state_mode == SCENE_STATE_MODE_EXTERNAL_AUTHORITY:
            if (
                not np.isnan(position_raw).all()
                or not np.isnan(rotation_raw).all()
                or np.any(evaluation_raw != 0)
            ):
                raise FullChainRendererError(
                    "external authority state cannot carry residual claims or solver work"
                )
            if np.any(~valid) and not np.isnan(q_arm[~valid]).all():
                raise FullChainRendererError(
                    "external invalid q_arm rows must remain all-NaN"
                )
            frame_indices: list[int] = []
            for frame_name in frame_names:
                if len(frame_name) != 5 or not frame_name.isdigit():
                    raise FullChainRendererError(
                        "external frame names must be exact zero-padded indices"
                    )
                frame_indices.append(int(frame_name))
            if frame_indices != sorted(set(frame_indices)):
                raise FullChainRendererError(
                    "external selected frame indices must be sorted and unique"
                )
        else:
            if (
                not np.isfinite(position_raw[valid]).all()
                or not np.isfinite(rotation_raw[valid]).all()
                or np.any(position_raw[valid] < 0)
                or np.any(rotation_raw[valid] < 0)
                or np.any(evaluation_raw[valid] <= 0)
                or not np.isnan(position_raw[~valid]).all()
                or not np.isnan(rotation_raw[~valid]).all()
                or np.any(evaluation_raw[~valid] != 0)
            ):
                raise FullChainRendererError(
                    "development solver residual/evaluation arrays are inconsistent"
                )
            if fixed_camera_base:
                position_stage_raw = np.asarray(
                    arrays["ik_position_stage_function_evaluations"]
                )
                full_pose_stage_raw = np.asarray(
                    arrays["ik_full_pose_stage_function_evaluations"]
                )
                if (
                    position_stage_raw.dtype != np.int32
                    or full_pose_stage_raw.dtype != np.int32
                    or position_stage_raw.shape != (count, 2)
                    or full_pose_stage_raw.shape != (count, 2)
                    or np.any(position_stage_raw[valid] <= 0)
                    or np.any(full_pose_stage_raw[valid] <= 0)
                    or np.any(
                        position_stage_raw[valid]
                        > FIXED_BASE_POSITION_STAGE_MAX_EVALUATIONS
                    )
                    or np.any(
                        full_pose_stage_raw[valid]
                        > FIXED_BASE_FULL_POSE_STAGE_MAX_EVALUATIONS
                    )
                    or np.any(position_stage_raw[~valid] != 0)
                    or np.any(full_pose_stage_raw[~valid] != 0)
                    or not np.array_equal(
                        evaluation_raw,
                        position_stage_raw + full_pose_stage_raw,
                    )
                ):
                    raise FullChainRendererError(
                        "fixed-base two-stage evaluation arrays violate exact contract"
                    )
                ik_position_stage_function_evaluations = position_stage_raw
                ik_full_pose_stage_function_evaluations = full_pose_stage_raw
    for frame, side in zip(*np.nonzero(valid)):
        _validate_se3(wrists[frame, side], f"wrist_T_camera[{frame},{side}]")
    base_raw = np.asarray(arrays["T_camera_base"])
    mounts_raw = np.asarray(arrays["T_tool_hand"])
    if versioned_contract and (
        base_raw.dtype != np.float64 or mounts_raw.dtype != np.float64
    ):
        raise FullChainRendererError(
            "versioned scene base/mount transforms must keep exact float64 dtype"
        )
    base = _validate_se3(base_raw, "T_camera_base")
    mounts = np.asarray(mounts_raw, dtype=np.float64)
    if mounts.shape != (2, 4, 4):
        raise FullChainRendererError(
            "T_tool_hand must contain exactly two session constants"
        )
    for side in range(2):
        _validate_se3(mounts[side], f"T_tool_hand[{side}]")
    arm_names = tuple(
        tuple(str(value) for value in row) for row in arrays["arm_joint_names"]
    )
    hand_names = tuple(
        tuple(str(value) for value in row) for row in arrays["hand_joint_names"]
    )
    if arm_names != ARM_JOINT_NAMES:
        raise FullChainRendererError("arm joint identity/order mismatch")
    if len(hand_names) != 2 or any(
        len(row) != 22 or len(set(row)) != 22 for row in hand_names
    ):
        raise FullChainRendererError("hand joint identity/order mismatch")
    return FullChainSceneState(
        session_id=session_id,
        frame_names=frame_names,
        q_arm=q_arm,
        q_hand=q_hand,
        valid=valid,
        wrist_T_camera=wrists,
        T_camera_base=base,
        T_tool_hand=mounts,
        camera_intrinsics=intrinsics,
        source_resolution=source_resolution,  # type: ignore[arg-type]
        arm_joint_names=arm_names,  # type: ignore[arg-type]
        hand_joint_names=hand_names,  # type: ignore[arg-type]
        scene_state_mode=scene_state_mode,
        solver_used=solver_used,
        residual_claimed=residual_claimed,
        position_residual_mm=position_residual_mm,
        rotation_residual_deg=rotation_residual_deg,
        ik_function_evaluations=ik_function_evaluations,
        ik_position_stage_function_evaluations=(ik_position_stage_function_evaluations),
        ik_full_pose_stage_function_evaluations=(
            ik_full_pose_stage_function_evaluations
        ),
        versioned_contract=versioned_contract,
        mount_descriptor_sha256=mount_descriptor_sha256,
        mount_evidence_mode=mount_evidence_mode,
        mount_evidence_sha256_by_side=mount_evidence_sha256_by_side,
        mount_authority_lineage_sha256=mount_authority_lineage_sha256,
        timestamp_ns=timestamp_ns,
        external_authority_schema=external_schema,
        external_authority_descriptor_sha256=external_descriptor_sha256,
        external_authority_arrays_sha256=external_arrays_sha256,
        external_authority_lineage_sha256=external_authority_lineage_sha256,
        external_base_evidence_sha256=external_base_evidence_sha256,
        external_q_arm_evidence_sha256_by_side=(external_q_arm_evidence_sha256_by_side),
        external_urdf_sha256=external_urdf_sha256,
        external_source_sha256=external_source_sha256,
        fixed_camera_base_candidate_schema=fixed_camera_base_candidate_schema,
        fixed_camera_base_candidate_descriptor_sha256=(
            fixed_camera_base_candidate_descriptor_sha256
        ),
        fixed_camera_base_candidate_lineage_sha256=(
            fixed_camera_base_candidate_lineage_sha256
        ),
    )


def validate_hand_joint_identity(
    state: FullChainSceneState, assets: PinnedRobotAssets
) -> None:
    for index, model in enumerate((assets.left_hand, assets.right_hand)):
        expected = tuple(
            joint.name for joint in model.joints if joint.joint_type != "fixed"
        )
        if state.hand_joint_names[index] != expected:
            raise FullChainRendererError(
                f"{SIDES[index]} KaiHand joint identity differs from pinned URDF"
            )


def _rotation_error_degrees(actual: np.ndarray, target: np.ndarray) -> float:
    relative = target[:3, :3].T @ actual[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    error = math.degrees(math.acos(cosine))
    return 0.0 if error < 1e-5 else error


def _tianji_side_for_link(link: str) -> str | None:
    if link.endswith("_L") or link == "left_tool":
        return "left"
    if link.endswith("_R") or link == "right_tool":
        return "right"
    return None


def _visual_role(component: str, side: str | None) -> str:
    if component in {"left_hand", "right_hand"}:
        return "KAIHAND"
    if side is not None:
        return "TIANJI_ARM"
    return "TIANJI_BASE"


def compose_frame_placement(
    state: FullChainSceneState,
    assets: PinnedRobotAssets,
    frame_index: int,
) -> FramePlacement:
    """Compute exact visual-link poses; invalid sides create no arm/hand pixels."""

    validate_hand_joint_identity(state, assets)
    if frame_index < 0 or frame_index >= len(state.frame_names):
        raise FullChainRendererError("frame index outside scene state")
    valid = tuple(bool(value) for value in state.valid[frame_index])
    arm_values: dict[str, float] = {}
    for side_index, names in enumerate(state.arm_joint_names):
        for offset, name in enumerate(names):
            arm_values[name] = (
                float(state.q_arm[frame_index, side_index, offset])
                if valid[side_index]
                else 0.0
            )
    tianji_fk = forward_kinematics(assets.tianji, arm_values)
    visuals: list[PlacedVisual] = []
    object_index = 1
    for visual in assets.tianji.visuals:
        side = _tianji_side_for_link(visual.link)
        if side is not None and not valid[SIDES.index(side)]:
            object_index += 1
            continue
        transform = state.T_camera_base @ tianji_fk[visual.link] @ visual.origin
        visuals.append(
            PlacedVisual(
                component="tianji",
                role=_visual_role("tianji", side),
                side=side,
                link=visual.link,
                mesh_path=visual.mesh_path,
                mesh_relative=visual.mesh_path.relative_to(
                    assets.project_root
                ).as_posix(),
                transform_blender=CV_CAMERA_TO_BLENDER @ transform,
                rgba=visual.rgba,
                object_index=object_index,
            )
        )
        object_index += 1
    residuals: dict[str, Mapping[str, float] | None] = {}
    for side_index, (side, hand_model) in enumerate(
        zip(SIDES, (assets.left_hand, assets.right_hand), strict=True)
    ):
        if not valid[side_index]:
            residuals[side] = None
            object_index += len(hand_model.visuals)
            continue
        hand_values = {
            name: float(state.q_hand[frame_index, side_index, offset])
            for offset, name in enumerate(state.hand_joint_names[side_index])
        }
        hand_fk = forward_kinematics(hand_model, hand_values)
        tool_link = f"{side}_tool"
        hand_root = (
            state.T_camera_base @ tianji_fk[tool_link] @ state.T_tool_hand[side_index]
        )
        target = state.wrist_T_camera[frame_index, side_index]
        position_mm = float(np.linalg.norm(hand_root[:3, 3] - target[:3, 3]) * 1000.0)
        rotation_deg = _rotation_error_degrees(hand_root, target)
        if state.scene_state_mode == SCENE_STATE_MODE_FIXED_CAMERA_BASE_SOLVER:
            if (
                state.position_residual_mm is None
                or state.rotation_residual_deg is None
            ):
                raise FullChainRendererError(
                    "fixed camera-base state lacks final residual arrays"
                )
            reported_position = float(
                state.position_residual_mm[frame_index, side_index]
            )
            reported_rotation = float(
                state.rotation_residual_deg[frame_index, side_index]
            )
            if not np.isclose(
                position_mm,
                reported_position,
                rtol=1e-9,
                atol=FIXED_BASE_RESIDUAL_MATCH_ATOL_MM,
            ) or not np.isclose(
                rotation_deg,
                reported_rotation,
                rtol=1e-9,
                atol=FIXED_BASE_RESIDUAL_MATCH_ATOL_DEG,
            ):
                raise FullChainRendererError(
                    "fixed camera-base reported residual does not match final q_arm FK"
                )
            if (
                position_mm > FIXED_BASE_GATE_POSITION_THRESHOLD_MM
                or rotation_deg > FIXED_BASE_GATE_ROTATION_THRESHOLD_DEG
            ):
                raise FullChainRendererError(
                    "fixed camera-base final q_arm fails full 6D IK gate"
                )
        residuals[side] = {
            "position_mm": position_mm,
            "rotation_deg": rotation_deg,
        }
        for visual in hand_model.visuals:
            transform = hand_root @ hand_fk[visual.link] @ visual.origin
            visuals.append(
                PlacedVisual(
                    component=f"{side}_hand",
                    role="KAIHAND",
                    side=side,
                    link=visual.link,
                    mesh_path=visual.mesh_path,
                    mesh_relative=visual.mesh_path.relative_to(
                        assets.project_root
                    ).as_posix(),
                    transform_blender=CV_CAMERA_TO_BLENDER @ transform,
                    rgba=visual.rgba,
                    object_index=object_index,
                )
            )
            object_index += 1
    return FramePlacement(
        frame_index=frame_index,
        frame_name=state.frame_names[frame_index],
        visuals=tuple(visuals),
        valid_by_side=valid,  # type: ignore[arg-type]
        hand_root_residual_by_side=residuals,
    )


def validate_scene_manifest(
    manifest: Mapping[str, Any],
    *,
    state_record: Mapping[str, Any],
    state: FullChainSceneState,
    cpu4_tool_definition: Mapping[str, Any],
    renderer_tool_definition: Mapping[str, Any],
) -> None:
    if manifest.get("schema_version") != "robot-fullchain-scene-manifest-v1":
        raise FullChainRendererError("unsupported scene manifest schema")
    if state.versioned_contract:
        expected_manifest_keys = {
            "schema_version",
            "status",
            "completion_mode",
            "session_id",
            "frame_count",
            "mount_provenance",
            "contact_infeasible",
            "visual_only",
            "development_only",
            "candidate_requires_human_review",
            "next_bucket_blocked",
            "advancement_authorized",
            "formal_consumer_allowed",
            "ik_residual_is_independent_accuracy_evidence",
            "q_arm_and_camera_base_authoritative",
            "q_arm_and_camera_base_authority_input_consumed",
            "scene_state_mode",
            "baseline_frozen",
            "gpu_calls",
            "renderer_calls",
            "pixels_produced",
            "scene_state",
            "explicit_mount_input",
            "mount_authority",
            "sources",
            "tool_definition",
            "ik_tool_definition",
            "visual_fit_tool_definition",
            "solver",
            "command_manifest",
            "success_evidence",
            "claim_limits",
        }
        if state.scene_state_mode == SCENE_STATE_MODE_EXTERNAL_AUTHORITY:
            expected_manifest_keys.add("external_base_q_authority")
        elif state.scene_state_mode == SCENE_STATE_MODE_FIXED_CAMERA_BASE_SOLVER:
            expected_manifest_keys.add("fixed_camera_base_candidate")
        if set(manifest) != expected_manifest_keys:
            raise FullChainRendererError("versioned scene manifest keys are not exact")
    if manifest.get("completion_mode") != "ARTIFACT_EXISTS":
        raise FullChainRendererError(
            "scene manifest completion mode must be ARTIFACT_EXISTS"
        )
    development_only = manifest.get("development_only")
    if type(development_only) is not bool:
        raise FullChainRendererError(
            "scene manifest development_only must be an exact boolean"
        )
    expected_status = (
        SCENE_MANIFEST_DEVELOPMENT_STATUS
        if development_only
        else SCENE_MANIFEST_VISUAL_CANDIDATE_STATUS
    )
    if manifest.get("status") != expected_status:
        raise FullChainRendererError(f"scene manifest status must be {expected_status}")
    if manifest.get("mount_provenance") != MOUNT_PROVENANCE:
        raise FullChainRendererError(
            "scene manifest mount provenance is not visual-only"
        )
    if manifest.get("contact_infeasible") != CONTACT_INFEASIBLE:
        raise FullChainRendererError(
            "scene manifest contact_infeasible must be UNMEASURED"
        )
    required_flags = {
        "visual_only": True,
        "candidate_requires_human_review": True,
        "next_bucket_blocked": True,
        "advancement_authorized": False,
        "formal_consumer_allowed": False,
        "baseline_frozen": False,
        "ik_residual_is_independent_accuracy_evidence": False,
        "q_arm_and_camera_base_authoritative": False,
    }
    for field, expected in required_flags.items():
        if manifest.get(field) is not expected:
            raise FullChainRendererError(f"scene manifest {field} must be {expected}")
    if state.versioned_contract and "scene_state_mode" not in manifest:
        raise FullChainRendererError(
            "versioned scene state requires an explicit manifest mode"
        )
    declared_mode = manifest.get(
        "scene_state_mode", SCENE_STATE_MODE_DEVELOPMENT_SOLVER
    )
    if declared_mode != state.scene_state_mode:
        raise FullChainRendererError(
            "scene manifest mode does not match the decoded scene state"
        )
    external = state.scene_state_mode == SCENE_STATE_MODE_EXTERNAL_AUTHORITY
    fixed_camera_base = (
        state.scene_state_mode == SCENE_STATE_MODE_FIXED_CAMERA_BASE_SOLVER
    )
    authority_consumed = manifest.get(
        "q_arm_and_camera_base_authority_input_consumed", False
    )
    if authority_consumed is not external:
        raise FullChainRendererError(
            "scene manifest external base/q consumption flag does not match state mode"
        )
    mount_descriptor_ref: dict[str, Any] | None = None
    if state.versioned_contract:
        for field in ("gpu_calls", "renderer_calls", "pixels_produced"):
            if type(manifest.get(field)) is not int or manifest.get(field) != 0:
                raise FullChainRendererError(
                    f"scene manifest {field} must remain exact integer zero"
                )
        mount_descriptor_ref = _manifest_evidence_ref(
            manifest.get("explicit_mount_input"), name="explicit mount"
        )
        if mount_descriptor_ref["sha256"] != state.mount_descriptor_sha256:
            raise FullChainRendererError(
                "mount descriptor SHA does not join the decoded scene state"
            )
    if manifest.get("session_id") != state.session_id:
        raise FullChainRendererError(
            "scene manifest session does not match decoded state"
        )
    frame_count = manifest.get("frame_count")
    if type(frame_count) is not int or frame_count != len(state.frame_names):
        raise FullChainRendererError(
            "scene manifest frame count does not match decoded state"
        )
    mount_authority = manifest.get("mount_authority")
    if not isinstance(mount_authority, Mapping):
        raise FullChainRendererError("scene manifest lacks mount authority")
    if state.versioned_contract:
        expected_mount_keys = {
            "provider",
            "method",
            "coordinate_definition",
            "matrix_direction",
            "axes_convention",
            "translation_units",
            "rotation_units",
            "source_links",
            "target_links",
            "independent_visual_evidence",
            "synthetic_fixture",
            "evidence_by_side",
            "session_constant",
            "per_frame_mount_forbidden",
            "independent_of_r2_wrist_targets",
            "selected_by_ik_residual",
            "evidence_mode",
        }
        if set(mount_authority) != expected_mount_keys:
            raise FullChainRendererError(
                "versioned scene mount authority keys are not exact"
            )
        if _canonical_json_sha256(dict(mount_authority)) != (
            state.mount_authority_lineage_sha256
        ):
            raise FullChainRendererError(
                "mount authority metadata does not join decoded state lineage"
            )
        mount_fixed = {
            "coordinate_definition": MOUNT_COORDINATE_DEFINITION,
            "matrix_direction": MOUNT_MATRIX_DIRECTION,
            "axes_convention": RIGHT_HANDED_AXES,
            "translation_units": "metres",
            "rotation_units": "radians",
            "source_links": MOUNT_SOURCE_LINKS,
            "target_links": MOUNT_TARGET_LINKS,
        }
        for field, expected in mount_fixed.items():
            if mount_authority.get(field) != expected:
                raise FullChainRendererError(
                    f"scene mount authority {field} must be {expected}"
                )
        for field in ("provider", "method"):
            if (
                not isinstance(mount_authority.get(field), str)
                or not str(mount_authority[field]).strip()
            ):
                raise FullChainRendererError(
                    f"scene mount authority {field} must be non-empty"
                )
    mount_flags = {
        "session_constant": True,
        "per_frame_mount_forbidden": True,
        "independent_of_r2_wrist_targets": True,
        "selected_by_ik_residual": False,
    }
    for field, expected in mount_flags.items():
        if mount_authority.get(field) is not expected:
            raise FullChainRendererError(
                f"scene manifest mount authority {field} must be {expected}"
            )
    if "scene_state_mode" in manifest and "evidence_mode" not in mount_authority:
        raise FullChainRendererError(
            "versioned scene-state mode requires the strengthened mount evidence contract"
        )
    if "evidence_mode" in mount_authority:
        evidence_mode = mount_authority.get("evidence_mode")
        evidence_by_side = mount_authority.get("evidence_by_side")
        independent = mount_authority.get("independent_visual_evidence")
        synthetic = mount_authority.get("synthetic_fixture")
        if evidence_mode == "DEVELOPMENT_ONLY_SYNTHETIC_FIXTURE":
            if (
                development_only is not True
                or independent is not False
                or synthetic is not True
                or evidence_by_side != {}
            ):
                raise FullChainRendererError(
                    "synthetic mount evidence must remain explicit development-only"
                )
            if state.versioned_contract and (
                state.mount_evidence_mode != evidence_mode
                or state.mount_evidence_sha256_by_side != ("", "")
            ):
                raise FullChainRendererError(
                    "synthetic mount manifest does not join state evidence lineage"
                )
        elif evidence_mode == "INDEPENDENT_BILATERAL_VISUAL_EVIDENCE":
            if independent is not True or synthetic is not False:
                raise FullChainRendererError(
                    "real mount evidence mode lacks its independent bilateral flags"
                )
            if mount_authority.get("method") != (
                "DIRECT_BILATERAL_VISUAL_MOUNT_MEASUREMENT"
            ):
                raise FullChainRendererError(
                    "real mount authority method is not direct bilateral measurement"
                )
            if not isinstance(evidence_by_side, Mapping) or set(
                evidence_by_side
            ) != set(SIDES):
                raise FullChainRendererError(
                    "real mount evidence mode requires exact left/right refs"
                )
            identities: set[tuple[int, int]] = set()
            expected_sha = state.mount_evidence_sha256_by_side
            if state.versioned_contract and (
                state.mount_evidence_mode != evidence_mode or expected_sha is None
            ):
                raise FullChainRendererError(
                    "measured mount manifest does not join state evidence mode"
                )
            for side_index, side in enumerate(SIDES):
                record = evidence_by_side[side]
                exact_keys = EVIDENCE_REF_KEYS | {
                    "session_id",
                    "side",
                    "provider",
                    "method",
                    "timestamp_ns",
                    "timestamp_clock_id",
                    "time_scope",
                    "device_id",
                    "calibration_id",
                    "calibration_sha256",
                    "source_link",
                    "target_link",
                    "matrix_direction",
                    "axes_convention",
                    "translation_units",
                    "rotation_units",
                    "independent_of_r2_wrist_targets",
                    "derived_from_r2_wrist_targets",
                    "derived_from_ik_solver",
                    "selected_by_ik_residual",
                    "circular_selection",
                    "synthetic_fixture",
                    "identity_fixture",
                    "formal_consumer_allowed",
                }
                if not isinstance(record, Mapping) or set(record) != exact_keys:
                    raise FullChainRendererError(
                        f"scene mount {side} evidence lineage keys are not exact"
                    )
                ref = _manifest_evidence_ref(
                    record, name=f"scene mount {side}", allow_extra=True
                )
                fixed = {
                    "session_id": state.session_id,
                    "side": side,
                    "method": "DIRECT_VISUAL_MOUNT_MEASUREMENT",
                    "time_scope": MOUNT_TIME_SCOPE,
                    "source_link": MOUNT_SOURCE_LINKS[side],
                    "target_link": MOUNT_TARGET_LINKS[side],
                    "matrix_direction": MOUNT_MATRIX_DIRECTION,
                    "axes_convention": RIGHT_HANDED_AXES,
                    "translation_units": "metres",
                    "rotation_units": "radians",
                    "independent_of_r2_wrist_targets": True,
                    "derived_from_r2_wrist_targets": False,
                    "derived_from_ik_solver": False,
                    "selected_by_ik_residual": False,
                    "circular_selection": False,
                    "synthetic_fixture": False,
                    "identity_fixture": False,
                    "formal_consumer_allowed": False,
                }
                for field, expected in fixed.items():
                    if record.get(field) != expected:
                        raise FullChainRendererError(
                            f"scene mount {side} evidence {field} must be {expected}"
                        )
                if (
                    type(record.get("timestamp_ns")) is not int
                    or int(record["timestamp_ns"]) < 0
                    or not isinstance(record.get("timestamp_clock_id"), str)
                    or not str(record["timestamp_clock_id"]).strip()
                    or not isinstance(record.get("device_id"), str)
                    or not str(record["device_id"]).strip()
                    or not isinstance(record.get("calibration_id"), str)
                    or not str(record["calibration_id"]).strip()
                    or not _is_sha256(record.get("calibration_sha256"))
                    or not isinstance(record.get("provider"), str)
                    or not str(record["provider"]).strip()
                    or (
                        state.versioned_contract
                        and expected_sha is not None
                        and ref["sha256"] != expected_sha[side_index]
                    )
                ):
                    raise FullChainRendererError(
                        f"scene mount {side} evidence time/device/calibration join failed"
                    )
                identity = (int(ref["device"]), int(ref["inode"]))
                if identity in identities:
                    raise FullChainRendererError(
                        "scene mount left/right evidence files must be independent"
                    )
                identities.add(identity)
            if (
                mount_descriptor_ref is not None
                and (
                    int(mount_descriptor_ref["device"]),
                    int(mount_descriptor_ref["inode"]),
                )
                in identities
            ):
                raise FullChainRendererError(
                    "mount descriptor cannot alias either side evidence file"
                )
            if any(np.array_equal(matrix, np.eye(4)) for matrix in state.T_tool_hand):
                raise FullChainRendererError(
                    "identity mount cannot impersonate measured mount authority"
                )
        else:
            raise FullChainRendererError("unsupported scene mount evidence mode")
    solver = manifest.get("solver")
    if not isinstance(solver, Mapping):
        raise FullChainRendererError("scene manifest lacks solver contract")
    if state.versioned_contract:
        solver_common_keys = {
            "schema_version",
            "backend",
            "solver_used",
            "residual_not_claimed",
            "max_position_residual_mm",
            "max_rotation_residual_deg",
            "mean_position_residual_mm",
            "mean_rotation_residual_deg",
            "joint_limit_violations",
            "base_is_one_session_constant",
            "invalid_sides_interpolated",
            "mount_optimized_or_selected",
            "objective_and_reported_residual_use_same_r2_wrist_target",
            "reported_residual_is_independent_validation",
            "independent_camera_base_or_arm_image_authority_consumed",
            "structural_parameter_count",
            "structural_pose_constraint_count",
            "structural_unknown_minus_constraint_count",
            "q_arm_and_camera_base_unique_or_authoritative",
        }
        development_solver_keys = {
            "outer_iterations",
            "ik_max_evaluations",
            "position_scale_m",
            "rotation_scale_rad",
        }
        fixed_camera_base_solver_keys = {
            "fixed_camera_base_optimized_or_refined",
            "per_valid_pair_initial_q",
            "ik_gate_name",
            "ik_gate_position_threshold_mm",
            "ik_gate_rotation_threshold_deg",
            "ik_gate_pair_pass_count",
            "ik_gate_pair_denominator",
            "ik_gate_pair_pass_rate",
            "ik_gate_frame_all_valid_sides_pass_count",
            "ik_gate_frame_denominator",
            "ik_gate_frame_pass_rate",
            "failed_valid_pairs_deleted",
        }
        fixed_camera_base_two_stage_solver_keys = {
            "ik_stage_count",
            "position_stage_backend",
            "position_stage_objective",
            "position_stage_initial_q",
            "position_stage_position_scale_m",
            "position_stage_ftol",
            "position_stage_xtol",
            "position_stage_gtol",
            "position_stage_max_evaluations",
            "position_stage_function_evaluations_total",
            "position_stage_function_evaluations_max",
            "full_pose_stage_backend",
            "full_pose_stage_objective",
            "full_pose_stage_initial_q",
            "full_pose_stage_position_scale_m",
            "full_pose_stage_rotation_scale_rad",
            "full_pose_stage_ftol",
            "full_pose_stage_xtol",
            "full_pose_stage_gtol",
            "full_pose_stage_max_evaluations",
            "full_pose_stage_function_evaluations_total",
            "full_pose_stage_function_evaluations_max",
            "ik_function_evaluations_total",
            "position_stage_result_used_only_by_same_pair",
            "cross_frame_side_or_candidate_warm_start",
        }
        expected_solver_keys = solver_common_keys | (
            set() if external else development_solver_keys
        )
        if fixed_camera_base:
            expected_solver_keys |= (
                fixed_camera_base_solver_keys | fixed_camera_base_two_stage_solver_keys
            )
        if set(solver) != expected_solver_keys:
            raise FullChainRendererError(
                "versioned scene solver manifest keys are not exact"
            )
        expected_solver_schema = (
            FIXED_BASE_TWO_STAGE_SOLVER_SCHEMA
            if fixed_camera_base
            else "robot-scene-state-cpu-solver-v1"
        )
        if solver.get("schema_version") != expected_solver_schema:
            raise FullChainRendererError("unsupported scene solver manifest schema")
        if (
            solver.get("joint_limit_violations") != 0
            or solver.get("base_is_one_session_constant") is not True
            or solver.get("invalid_sides_interpolated") is not False
        ):
            raise FullChainRendererError(
                "scene solver joint/base/interpolation contract is invalid"
            )
        if external:
            if any(
                solver.get(field) is not None
                for field in (
                    "max_position_residual_mm",
                    "max_rotation_residual_deg",
                    "mean_position_residual_mm",
                    "mean_rotation_residual_deg",
                )
            ) or any(
                solver.get(field) != 0
                for field in (
                    "structural_parameter_count",
                    "structural_pose_constraint_count",
                    "structural_unknown_minus_constraint_count",
                )
            ):
                raise FullChainRendererError(
                    "external authority solver manifest cannot carry residual/solver claims"
                )
        else:
            position = state.position_residual_mm
            rotation = state.rotation_residual_deg
            if position is None or rotation is None:
                raise FullChainRendererError(
                    "development solver state lacks residual arrays"
                )
            selected_position = position[state.valid]
            selected_rotation = rotation[state.valid]
            expected_summary = {
                "max_position_residual_mm": float(np.max(selected_position)),
                "max_rotation_residual_deg": float(np.max(selected_rotation)),
                "mean_position_residual_mm": float(np.mean(selected_position)),
                "mean_rotation_residual_deg": float(np.mean(selected_rotation)),
            }
            if any(
                solver.get(field) != expected
                for field, expected in expected_summary.items()
            ):
                raise FullChainRendererError(
                    "development residual summary does not join scene-state arrays"
                )
            if fixed_camera_base:
                position_stage_evaluations = (
                    state.ik_position_stage_function_evaluations
                )
                full_pose_stage_evaluations = (
                    state.ik_full_pose_stage_function_evaluations
                )
                total_evaluations = state.ik_function_evaluations
                if (
                    position_stage_evaluations is None
                    or full_pose_stage_evaluations is None
                    or total_evaluations is None
                ):
                    raise FullChainRendererError(
                        "fixed camera-base state lacks two-stage evaluation arrays"
                    )
                pair_pass = (
                    selected_position <= FIXED_BASE_GATE_POSITION_THRESHOLD_MM
                ) & (selected_rotation <= FIXED_BASE_GATE_ROTATION_THRESHOLD_DEG)
                pair_denominator = int(np.count_nonzero(state.valid))
                pair_pass_count = int(np.count_nonzero(pair_pass))
                pair_pass_matrix = np.zeros_like(state.valid, dtype=np.bool_)
                pair_pass_matrix[state.valid] = pair_pass
                frame_has_valid = np.any(state.valid, axis=1)
                frame_pass = frame_has_valid & np.all(
                    (~state.valid) | pair_pass_matrix, axis=1
                )
                frame_denominator = int(np.count_nonzero(frame_has_valid))
                frame_pass_count = int(np.count_nonzero(frame_pass))
                if pair_denominator == 0 or frame_denominator == 0:
                    raise FullChainRendererError(
                        "fixed camera-base solver requires at least one valid pair"
                    )
                expected_fixed = {
                    "backend": (
                        "SCIPY_CPU_TRF_FIXED_CAMERA_BASE_POSITION_THEN_FULL_POSE_"
                        "PER_SIDE_FRAME"
                    ),
                    "outer_iterations": 0,
                    "ik_max_evaluations": (FIXED_BASE_FULL_POSE_STAGE_MAX_EVALUATIONS),
                    "position_scale_m": FIXED_BASE_POSITION_SCALE_M,
                    "rotation_scale_rad": FIXED_BASE_ROTATION_SCALE_RAD,
                    "fixed_camera_base_optimized_or_refined": False,
                    "per_valid_pair_initial_q": (
                        "POSITION_STAGE_UNIFORM_ZERO_7DOF_THEN_FULL_POSE_FROM_"
                        "SAME_PAIR_POSITION_RESULT_NO_CROSS_PAIR_WARM_START"
                    ),
                    "ik_gate_name": "IK_GATE_PASS_RATE",
                    "ik_gate_position_threshold_mm": (
                        FIXED_BASE_GATE_POSITION_THRESHOLD_MM
                    ),
                    "ik_gate_rotation_threshold_deg": (
                        FIXED_BASE_GATE_ROTATION_THRESHOLD_DEG
                    ),
                    "ik_gate_pair_pass_count": pair_pass_count,
                    "ik_gate_pair_denominator": pair_denominator,
                    "ik_gate_pair_pass_rate": pair_pass_count / pair_denominator,
                    "ik_gate_frame_all_valid_sides_pass_count": frame_pass_count,
                    "ik_gate_frame_denominator": frame_denominator,
                    "ik_gate_frame_pass_rate": frame_pass_count / frame_denominator,
                    "failed_valid_pairs_deleted": False,
                    "structural_parameter_count": 7 * pair_denominator,
                    "structural_pose_constraint_count": 6 * pair_denominator,
                    "structural_unknown_minus_constraint_count": pair_denominator,
                    "ik_stage_count": 2,
                    "position_stage_backend": "SCIPY_CPU_TRF",
                    "position_stage_objective": "TOOL_TRANSLATION_3D_SCALED",
                    "position_stage_initial_q": ("UNIFORM_ZERO_7DOF_PER_VALID_PAIR"),
                    "position_stage_position_scale_m": (FIXED_BASE_POSITION_SCALE_M),
                    "position_stage_ftol": FIXED_BASE_IK_TRF_FTOL,
                    "position_stage_xtol": FIXED_BASE_IK_TRF_XTOL,
                    "position_stage_gtol": FIXED_BASE_IK_TRF_GTOL,
                    "position_stage_max_evaluations": (
                        FIXED_BASE_POSITION_STAGE_MAX_EVALUATIONS
                    ),
                    "position_stage_function_evaluations_total": int(
                        np.sum(position_stage_evaluations[state.valid], dtype=np.int64)
                    ),
                    "position_stage_function_evaluations_max": int(
                        np.max(position_stage_evaluations[state.valid])
                    ),
                    "full_pose_stage_backend": "SCIPY_CPU_TRF",
                    "full_pose_stage_objective": (
                        "TOOL_SE3_TRANSLATION_AND_ROTATION_6D_SCALED"
                    ),
                    "full_pose_stage_initial_q": (
                        "SAME_VALID_PAIR_POSITION_STAGE_RESULT"
                    ),
                    "full_pose_stage_position_scale_m": (FIXED_BASE_POSITION_SCALE_M),
                    "full_pose_stage_rotation_scale_rad": (
                        FIXED_BASE_ROTATION_SCALE_RAD
                    ),
                    "full_pose_stage_ftol": FIXED_BASE_IK_TRF_FTOL,
                    "full_pose_stage_xtol": FIXED_BASE_IK_TRF_XTOL,
                    "full_pose_stage_gtol": FIXED_BASE_IK_TRF_GTOL,
                    "full_pose_stage_max_evaluations": (
                        FIXED_BASE_FULL_POSE_STAGE_MAX_EVALUATIONS
                    ),
                    "full_pose_stage_function_evaluations_total": int(
                        np.sum(full_pose_stage_evaluations[state.valid], dtype=np.int64)
                    ),
                    "full_pose_stage_function_evaluations_max": int(
                        np.max(full_pose_stage_evaluations[state.valid])
                    ),
                    "ik_function_evaluations_total": int(
                        np.sum(total_evaluations[state.valid], dtype=np.int64)
                    ),
                    "position_stage_result_used_only_by_same_pair": True,
                    "cross_frame_side_or_candidate_warm_start": False,
                }
                if any(
                    type(solver.get(field)) is not type(expected)
                    or solver.get(field) != expected
                    for field, expected in expected_fixed.items()
                ):
                    raise FullChainRendererError(
                        "fixed camera-base solver/gate contract does not join state"
                    )
    solver_flags = {
        "mount_optimized_or_selected": False,
        "objective_and_reported_residual_use_same_r2_wrist_target": not external,
        "reported_residual_is_independent_validation": False,
        "independent_camera_base_or_arm_image_authority_consumed": external,
        "q_arm_and_camera_base_unique_or_authoritative": False,
    }
    for field, expected in solver_flags.items():
        if solver.get(field) is not expected:
            raise FullChainRendererError(
                f"scene manifest solver {field} must be {expected}"
            )
    if external and not {"solver_used", "residual_not_claimed"}.issubset(solver):
        raise FullChainRendererError(
            "external authority solver contract lacks explicit no-solver/no-residual flags"
        )
    if "solver_used" in solver and solver.get("solver_used") is not (not external):
        raise FullChainRendererError(
            "scene manifest solver_used does not match scene-state mode"
        )
    if (
        "residual_not_claimed" in solver
        and solver.get("residual_not_claimed") is not external
    ):
        raise FullChainRendererError(
            "scene manifest residual claim does not match scene-state mode"
        )
    if state.solver_used is external or state.residual_claimed is external:
        raise FullChainRendererError(
            "decoded scene solver/residual flags contradict scene-state mode"
        )
    if external:
        authority = manifest.get("external_base_q_authority")
        if not isinstance(authority, Mapping):
            raise FullChainRendererError(
                "external scene-state mode lacks base/q authority manifest"
            )
        expected_authority_keys = {
            "schema_version",
            "descriptor",
            "arrays",
            "direct_evidence",
            "urdf_refs",
            "source_refs",
            "session_id",
            "frame_count",
            "selected_frame_indices",
            "selected_frame_names",
            "selected_timestamp_ns",
            "authority_kind",
            "evidence_kind",
            "provider",
            "method",
            "direct_measurement_evidence",
            "independent_of_r2_wrist_targets",
            "derived_from_r2_wrist_targets",
            "derived_from_ik_solver",
            "selected_by_ik_residual",
            "interpolated_or_filled",
            "synthetic_fixture",
            "identity_fixture",
            "formal_consumer_allowed",
            "timestamp_clock_id",
            "frame_timestamp_mapping",
            "device_id",
            "calibration_id",
            "calibration_sha256",
            "T_camera_base_semantics",
            "q_arm_units",
            "solver_not_used",
            "residual_not_claimed",
        }
        if set(authority) != expected_authority_keys:
            raise FullChainRendererError(
                "external base/q authority manifest keys are not exact"
            )
        if _canonical_json_sha256(dict(authority)) != (
            state.external_authority_lineage_sha256
        ):
            raise FullChainRendererError(
                "external base/q authority metadata does not join decoded state lineage"
            )
        if authority.get("schema_version") != state.external_authority_schema:
            raise FullChainRendererError(
                "external base/q schema differs between manifest and scene state"
            )
        descriptor = _manifest_evidence_ref(
            authority.get("descriptor"), name="external base/q descriptor"
        )
        authority_arrays = _manifest_evidence_ref(
            authority.get("arrays"), name="external base/q arrays"
        )
        if (
            descriptor["sha256"] != state.external_authority_descriptor_sha256
            or authority_arrays["sha256"] != state.external_authority_arrays_sha256
        ):
            raise FullChainRendererError(
                "external base/q descriptor/array SHA join failed"
            )
        fixed_authority = {
            "session_id": state.session_id,
            "authority_kind": "DIRECT_EXTERNAL_BASE_AND_ARM_STATE",
            "evidence_kind": EXTERNAL_AUTHORITY_EVIDENCE_KIND,
            "method": EXTERNAL_AUTHORITY_METHOD,
            "direct_measurement_evidence": True,
            "independent_of_r2_wrist_targets": True,
            "derived_from_r2_wrist_targets": False,
            "derived_from_ik_solver": False,
            "selected_by_ik_residual": False,
            "interpolated_or_filled": False,
            "synthetic_fixture": False,
            "identity_fixture": False,
            "formal_consumer_allowed": False,
            "frame_timestamp_mapping": EXTERNAL_FRAME_TIMESTAMP_MAPPING,
            "T_camera_base_semantics": "p_camera = T_camera_base @ p_base",
            "q_arm_units": "radians",
            "solver_not_used": True,
            "residual_not_claimed": True,
        }
        for field, expected in fixed_authority.items():
            if authority.get(field) != expected:
                raise FullChainRendererError(
                    f"external base/q authority {field} must be {expected}"
                )
        for field in (
            "provider",
            "timestamp_clock_id",
            "device_id",
            "calibration_id",
        ):
            if (
                not isinstance(authority.get(field), str)
                or not str(authority[field]).strip()
            ):
                raise FullChainRendererError(
                    f"external base/q authority {field} must be non-empty"
                )
        if not _is_sha256(authority.get("calibration_sha256")):
            raise FullChainRendererError(
                "external base/q calibration SHA lineage is invalid"
            )
        if (
            type(authority.get("frame_count")) is not int
            or int(authority["frame_count"]) <= 0
        ):
            raise FullChainRendererError(
                "external base/q full frame count must be a positive integer"
            )
        expected_indices = [int(name) for name in state.frame_names]
        expected_timestamps = (
            [int(item) for item in state.timestamp_ns]
            if state.timestamp_ns is not None
            else None
        )
        if (
            authority.get("selected_frame_indices") != expected_indices
            or authority.get("selected_frame_names") != list(state.frame_names)
            or authority.get("selected_timestamp_ns") != expected_timestamps
            or any(index >= int(authority["frame_count"]) for index in expected_indices)
        ):
            raise FullChainRendererError(
                "external frame/timestamp lineage does not join decoded state"
            )

        urdf_refs = authority.get("urdf_refs")
        if not isinstance(urdf_refs, Mapping) or set(urdf_refs) != {
            "robot_asset_pin",
            "tianji_urdf",
        }:
            raise FullChainRendererError("external URDF refs are not exact")
        urdf_records = tuple(
            _manifest_evidence_ref(
                urdf_refs[name], name=f"external {name.replace('_', ' ')}"
            )
            for name in ("robot_asset_pin", "tianji_urdf")
        )
        if state.external_urdf_sha256 != tuple(
            str(record["sha256"]) for record in urdf_records
        ):
            raise FullChainRendererError("external URDF SHA lineage join failed")

        source_refs = authority.get("source_refs")
        if not isinstance(source_refs, Mapping) or set(source_refs) != {
            "r2_sidecar",
            "hawor",
        }:
            raise FullChainRendererError("external source refs are not exact")
        source_records = tuple(
            _manifest_evidence_ref(source_refs[name], name=f"external {name}")
            for name in ("r2_sidecar", "hawor")
        )
        if state.external_source_sha256 != tuple(
            str(record["sha256"]) for record in source_records
        ):
            raise FullChainRendererError("external source SHA lineage join failed")
        manifest_sources = manifest.get("sources")
        if not isinstance(manifest_sources, Mapping) or any(
            manifest_sources.get(name) != source_refs[name]
            for name in ("r2_sidecar", "hawor")
        ):
            raise FullChainRendererError(
                "top-level source records do not join external authority lineage"
            )

        direct = authority.get("direct_evidence")
        if not isinstance(direct, Mapping) or set(direct) != {
            "camera_base",
            "q_arm_by_side",
        }:
            raise FullChainRendererError(
                "external direct evidence closure is not exact"
            )
        camera_base = direct["camera_base"]
        base_keys = EVIDENCE_REF_KEYS | {
            "schema_version",
            "session_id",
            "evidence_kind",
            "provider",
            "method",
            "T_camera_base_semantics",
            "transform_convention",
            "translation_units",
            "timestamp_ns",
            "timestamp_clock_id",
            "device_id",
            "calibration_id",
            "calibration_sha256",
            "independent_of_r2_wrist_targets",
            "derived_from_r2_wrist_targets",
            "derived_from_ik_solver",
            "selected_by_ik_residual",
            "interpolated_or_filled",
            "synthetic_fixture",
            "identity_fixture",
            "formal_consumer_allowed",
            "T_camera_base",
        }
        if not isinstance(camera_base, Mapping) or set(camera_base) != base_keys:
            raise FullChainRendererError(
                "external camera/base evidence keys are not exact"
            )
        base_ref = _manifest_evidence_ref(
            camera_base, name="external camera/base evidence", allow_extra=True
        )
        _require_manifest_direct_flags(
            camera_base, name="external camera/base evidence"
        )
        base_fixed = {
            "schema_version": EXTERNAL_BASE_EVIDENCE_SCHEMA,
            "session_id": state.session_id,
            "evidence_kind": EXTERNAL_BASE_EVIDENCE_KIND,
            "method": "DIRECT_CAMERA_TO_ROBOT_BASE_CALIBRATION",
            "T_camera_base_semantics": "p_camera = T_camera_base @ p_base",
            "transform_convention": RIGHT_HANDED_AXES,
            "translation_units": "metres",
            "timestamp_clock_id": authority["timestamp_clock_id"],
            "device_id": authority["device_id"],
            "calibration_id": authority["calibration_id"],
            "calibration_sha256": authority["calibration_sha256"],
        }
        for field, expected in base_fixed.items():
            if camera_base.get(field) != expected:
                raise FullChainRendererError(
                    f"external camera/base evidence {field} must be {expected}"
                )
        if (
            base_ref["sha256"] != state.external_base_evidence_sha256
            or type(camera_base.get("timestamp_ns")) is not int
            or int(camera_base["timestamp_ns"]) < 0
            or not isinstance(camera_base.get("provider"), str)
            or not str(camera_base["provider"]).strip()
            or not np.array_equal(
                _validate_se3(
                    np.asarray(camera_base.get("T_camera_base"), dtype=np.float64),
                    "external camera/base evidence T_camera_base",
                ),
                state.T_camera_base,
            )
        ):
            raise FullChainRendererError(
                "external camera/base evidence does not join decoded state"
            )

        q_by_side = direct["q_arm_by_side"]
        if not isinstance(q_by_side, Mapping) or set(q_by_side) != set(SIDES):
            raise FullChainRendererError(
                "external q_arm evidence must contain exact left/right records"
            )
        q_expected_sha = state.external_q_arm_evidence_sha256_by_side
        if q_expected_sha is None:
            raise FullChainRendererError("external q_arm state SHA lineage is absent")
        q_records: list[dict[str, Any]] = []
        q_keys = EVIDENCE_REF_KEYS | {
            "schema_version",
            "session_id",
            "side",
            "evidence_kind",
            "provider",
            "method",
            "frame_count",
            "frame_range",
            "frame_timestamp_mapping",
            "timestamp_clock_id",
            "device_id",
            "calibration_id",
            "calibration_sha256",
            "axes_convention",
            "q_arm_units",
            "array_digest_canonicalization",
            "frame_names_sha256",
            "timestamp_ns_sha256",
            "valid_sha256",
            "q_arm_sha256",
            "arm_joint_names_sha256",
            "joint_lower_sha256",
            "joint_upper_sha256",
            "independent_of_r2_wrist_targets",
            "derived_from_r2_wrist_targets",
            "derived_from_ik_solver",
            "selected_by_ik_residual",
            "interpolated_or_filled",
            "synthetic_fixture",
            "identity_fixture",
            "formal_consumer_allowed",
        }
        for side_index, side in enumerate(SIDES):
            q_record = q_by_side[side]
            if not isinstance(q_record, Mapping) or set(q_record) != q_keys:
                raise FullChainRendererError(
                    f"external {side} q_arm evidence keys are not exact"
                )
            ref = _manifest_evidence_ref(
                q_record, name=f"external {side} q_arm evidence", allow_extra=True
            )
            _require_manifest_direct_flags(
                q_record, name=f"external {side} q_arm evidence"
            )
            fixed = {
                "schema_version": EXTERNAL_Q_ARM_SIDE_EVIDENCE_SCHEMA,
                "session_id": state.session_id,
                "side": side,
                "evidence_kind": EXTERNAL_Q_ARM_EVIDENCE_KIND,
                "method": "DIRECT_ARM_JOINT_ENCODER_READOUT",
                "frame_count": authority["frame_count"],
                "frame_range": {
                    "start": 0,
                    "stop_exclusive": authority["frame_count"],
                    "contiguous": True,
                },
                "frame_timestamp_mapping": EXTERNAL_FRAME_TIMESTAMP_MAPPING,
                "timestamp_clock_id": authority["timestamp_clock_id"],
                "axes_convention": RIGHT_HANDED_AXES,
                "q_arm_units": "radians",
                "array_digest_canonicalization": (
                    EXTERNAL_ARRAY_DIGEST_CANONICALIZATION
                ),
            }
            for field, expected in fixed.items():
                if q_record.get(field) != expected:
                    raise FullChainRendererError(
                        f"external {side} q_arm evidence {field} must be {expected}"
                    )
            if (
                ref["sha256"] != q_expected_sha[side_index]
                or not isinstance(q_record.get("provider"), str)
                or not str(q_record["provider"]).strip()
                or not isinstance(q_record.get("device_id"), str)
                or not str(q_record["device_id"]).strip()
                or not isinstance(q_record.get("calibration_id"), str)
                or not str(q_record["calibration_id"]).strip()
                or not _is_sha256(q_record.get("calibration_sha256"))
                or any(
                    not _is_sha256(q_record.get(field))
                    for field in (
                        "frame_names_sha256",
                        "timestamp_ns_sha256",
                        "valid_sha256",
                        "q_arm_sha256",
                        "arm_joint_names_sha256",
                        "joint_lower_sha256",
                        "joint_upper_sha256",
                    )
                )
            ):
                raise FullChainRendererError(
                    f"external {side} q_arm evidence lineage is invalid"
                )
            q_records.append(ref)

        all_records = [
            descriptor,
            authority_arrays,
            base_ref,
            *q_records,
            *urdf_records,
            *source_records,
        ]
        identities = {
            (int(record["device"]), int(record["inode"])) for record in all_records
        }
        if len(identities) != len(all_records):
            raise FullChainRendererError(
                "external authority/evidence/source/URDF records must not alias"
            )
    elif manifest.get("external_base_q_authority") is not None:
        raise FullChainRendererError(
            "development solver mode cannot claim an external base/q authority"
        )
    if fixed_camera_base:
        candidate = manifest.get("fixed_camera_base_candidate")
        if not isinstance(candidate, Mapping):
            raise FullChainRendererError(
                "fixed camera-base scene lacks its candidate input lineage"
            )
        expected_candidate_keys = {
            "schema_version",
            "descriptor",
            "T_camera_base_semantics",
            "axes_convention",
            "translation_units",
            "fixed_across_selected_frames",
            "global_across_sessions",
            "selected_on_session",
            "not_per_session_tuned",
            "per_frame_override_forbidden",
            "per_session_override_forbidden",
            "development_only",
            "candidate_requires_human_review",
            "next_bucket_blocked",
            "advancement_authorized",
            "formal_consumer_allowed",
            "T_camera_base",
        }
        if set(candidate) != expected_candidate_keys:
            raise FullChainRendererError(
                "fixed camera-base candidate manifest keys are not exact"
            )
        candidate_ref = _manifest_evidence_ref(
            candidate.get("descriptor"), name="fixed camera-base candidate descriptor"
        )
        expected_candidate_values = {
            "schema_version": FIXED_CAMERA_BASE_CANDIDATE_SCHEMA,
            "T_camera_base_semantics": "p_camera = T_camera_base @ p_base",
            "axes_convention": "RIGHT_HANDED_XYZ",
            "translation_units": "metres",
            "fixed_across_selected_frames": True,
            "global_across_sessions": True,
            "selected_on_session": "grap_a_cap_004",
            "not_per_session_tuned": True,
            "per_frame_override_forbidden": True,
            "per_session_override_forbidden": True,
            "development_only": True,
            "candidate_requires_human_review": True,
            "next_bucket_blocked": True,
            "advancement_authorized": False,
            "formal_consumer_allowed": False,
        }
        if any(
            type(candidate.get(field)) is not type(expected)
            or candidate.get(field) != expected
            for field, expected in expected_candidate_values.items()
        ):
            raise FullChainRendererError(
                "fixed camera-base candidate development contract is invalid"
            )
        encoded_candidate_base = candidate.get("T_camera_base")
        if (
            not isinstance(encoded_candidate_base, list)
            or len(encoded_candidate_base) != 4
            or any(
                not isinstance(row, list) or len(row) != 4
                for row in encoded_candidate_base
            )
            or any(
                type(item) not in {int, float}
                for row in encoded_candidate_base
                for item in row
            )
        ):
            raise FullChainRendererError(
                "fixed camera-base candidate matrix must be exact numeric JSON[4,4]"
            )
        candidate_base = _validate_se3(
            np.asarray(encoded_candidate_base, dtype=np.float64),
            "fixed camera-base candidate T_camera_base",
        )
        if not np.array_equal(candidate_base, state.T_camera_base):
            raise FullChainRendererError(
                "fixed camera-base candidate transform differs from scene state"
            )
        if (
            state.fixed_camera_base_candidate_schema
            != FIXED_CAMERA_BASE_CANDIDATE_SCHEMA
            or candidate_ref["sha256"]
            != state.fixed_camera_base_candidate_descriptor_sha256
            or _canonical_json_sha256(dict(candidate))
            != state.fixed_camera_base_candidate_lineage_sha256
        ):
            raise FullChainRendererError(
                "fixed camera-base candidate does not join scene-state lineage"
            )
        state_input_ref = _manifest_evidence_ref(
            state_record, name="decoded scene state"
        )
        candidate_identity = (
            int(candidate_ref["device"]),
            int(candidate_ref["inode"]),
        )
        forbidden_identities = {
            (int(state_input_ref["device"]), int(state_input_ref["inode"]))
        }
        if mount_descriptor_ref is not None:
            forbidden_identities.add(
                (
                    int(mount_descriptor_ref["device"]),
                    int(mount_descriptor_ref["inode"]),
                )
            )
        if candidate_identity in forbidden_identities:
            raise FullChainRendererError(
                "fixed camera-base descriptor cannot alias scene or mount input"
            )
    elif manifest.get("fixed_camera_base_candidate") is not None:
        raise FullChainRendererError(
            "non-fixed scene mode cannot claim a fixed camera-base candidate"
        )
    expected_state = manifest.get("scene_state")
    if state.versioned_contract:
        expected_state_ref = _manifest_evidence_ref(
            expected_state, name="manifest scene state"
        )
        actual_state_ref = _manifest_evidence_ref(
            state_record, name="decoded scene state"
        )
        if expected_state_ref != actual_state_ref:
            raise FullChainRendererError(
                "scene-state path/bytes/SHA/device/inode do not match manifest"
            )
        if mount_descriptor_ref is None:
            raise FullChainRendererError("versioned scene lacks mount descriptor ref")
        if (
            int(expected_state_ref["device"]),
            int(expected_state_ref["inode"]),
        ) == (
            int(mount_descriptor_ref["device"]),
            int(mount_descriptor_ref["inode"]),
        ):
            raise FullChainRendererError(
                "scene state and mount descriptor files must not alias"
            )
        success = manifest.get("success_evidence")
        if not isinstance(success, list) or len(success) != 1:
            raise FullChainRendererError(
                "scene manifest must contain one exact scene-state success evidence"
            )
        success_record = success[0]
        if not isinstance(success_record, Mapping) or set(success_record) != (
            EVIDENCE_REF_KEYS | {"minimum_bytes", "verification"}
        ):
            raise FullChainRendererError("scene success evidence keys are not exact")
        success_ref = _manifest_evidence_ref(
            success_record, name="scene success evidence", allow_extra=True
        )
        if (
            success_ref != expected_state_ref
            or success_record.get("minimum_bytes") != 1024
            or success_record.get("verification")
            != "SHA256_MATCH_AND_FULL_DECODE_SCENE_STATE"
        ):
            raise FullChainRendererError(
                "scene success evidence does not join the decoded state"
            )
    elif not isinstance(expected_state, Mapping) or any(
        expected_state.get(key) != state_record.get(key) for key in ("bytes", "sha256")
    ):
        raise FullChainRendererError("scene-state bytes/SHA do not match manifest")
    declared_tool = manifest.get("tool_definition")
    if not isinstance(declared_tool, Mapping):
        raise FullChainRendererError("scene manifest lacks tool definition")
    require_same_tool_definition(
        dict(cpu4_tool_definition), dict(renderer_tool_definition), dict(declared_tool)
    )
    if state.versioned_contract:
        ik_tool = manifest.get("ik_tool_definition")
        visual_tool = manifest.get("visual_fit_tool_definition")
        if not isinstance(ik_tool, Mapping) or not isinstance(visual_tool, Mapping):
            raise FullChainRendererError(
                "versioned scene manifest lacks complete tool-definition lineage"
            )
        require_same_tool_definition(
            dict(declared_tool), dict(ik_tool), dict(visual_tool)
        )


def output_manifest_skeleton(
    *,
    session_id: str,
    frame_start: int,
    frame_count: int,
    resolution: Sequence[int],
    tool_definition: Mapping[str, Any],
    source_records: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": OUTPUT_SCHEMA,
        "status": "CANDIDATE_REQUIRES_HUMAN_REVIEW",
        "engine": ENGINE,
        "session_id": session_id,
        "frame_start": frame_start,
        "frame_count": frame_count,
        "resolution": list(resolution),
        "single_process_multi_frame": True,
        "mount_provenance": MOUNT_PROVENANCE,
        "contact_infeasible": CONTACT_INFEASIBLE,
        "visual_only": True,
        "candidate_requires_human_review": True,
        "next_bucket_blocked": True,
        "advancement_authorized": False,
        "formal_consumer_allowed": False,
        "baseline_frozen": False,
        "tool_definition": dict(tool_definition),
        "geometry_contract": {
            "render_geometry": "PINNED_URDF_VISUAL_ELEMENTS_ONLY",
            "collision_geometry_consumed": False,
            "safety_dilation_consumed": False,
            "renderer_only_offset": False,
        },
        "buffers": {
            "beauty": "RGBA_PNG_TRANSPARENT",
            "range": "OPENEXR_DEPTH_PASS_EUCLIDEAN_CAMERA_RANGE_METRES",
            "object_index": "OPENEXR_QA_ONLY_NOT_COMPOSITOR_DECISION",
        },
        "source_records": dict(source_records),
        "claim_limits": [
            "NO_PHYSICAL_FEASIBILITY",
            "NO_COLLISION_OR_REACHABILITY",
            "NO_TRAINING_GROUND_TRUTH",
            "NO_REAL_ROBOT_EXECUTION",
            "CONTACT_INFEASIBLE_UNMEASURED",
        ],
    }
