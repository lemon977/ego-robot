"""Bounded Robot-only flange-link correction for two sealed development captures.

This is a candidate helper, not a CLI operation. A future governed successor must
pin this file, its transitive closure and its exact output attempt before calling
it. No HaWoR model, detector, historical motion or holdout result is consumed.
Only the hand-root parent link changes from ``side_tool`` to ``flange_L/R``;
all existing base search, IK/hand objective, thresholds and raster code stay pinned.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import threading
from typing import Any, Iterator

import numpy as np

from chaoyang.governance.exact78_cold_start_guard_v1 import file_ref
from chaoyang.pipeline import exact78_fresh_robot_stage_v1 as stage_v1
from chaoyang.pipeline import exact78_fresh_robot_stage_v2 as ffmpeg44
from chaoyang.pipeline import exact78_robot_raster_v1 as raster_v1


ROOT = Path(__file__).resolve().parents[3]
_OVERRIDE_LOCK = threading.Lock()
_CODE_PINS = {
    "src/chaoyang/pipeline/exact78_fresh_robot_stage_v1.py":
        "9018d49be9030ec84c48e470f5b600ac828665bfe7e974b64b5f0736375b196c",
    "src/chaoyang/pipeline/exact78_robot_raster_v1.py":
        "9678646b8b2d02567aa4bf64c68e339921c56d204299660894b7045f82ce6ed6",
    "src/chaoyang/pipeline/exact78_fresh_robot_stage_v2.py":
        "1876c5af135bd97d9e7227b076b1879ce402dc47704c260c15a6b3b0220072c9",
}
_SEALED_DEVELOPMENT = {
    "play_cards_0902_042": {
        "root": "_run/current/exact78_raw_to_robot_t2_0902_cards042_v1_20260930/"
                "attempts/attempt_0001/candidate/hawor",
        "source_kind": "tracker", "frames": 171,
        "seal_sha": "ac419f0abf54dd7a713cce978c97e87b942155ee670bd021887e8ec554719ad0",
        "native_audit_sha": "f061282ca3fa3dfcff6b0507164e9fca815c6ea64d40005c02642849dd240bbd",
        "sha": {
            "RESULT.json": "b2f3917c7bae62afa558569c221268326bb99a048282c2f1120f5968edc2e5f7",
            "HAWOR_CAMERA_PREDICTIONS.npz": "00a417086f9223f0268155f99861d173f7bc229f09ad163b9dae39f2e82fed30",
            "FRAME_PROVENANCE.json": "dfabcb22902000354ec39422b6cdade372582461f6a157aa2d15a7cbf39c6a65",
            "INPUT_AUDIT.json": "dd870a95838bb50f05fd0cf3ee8cd3333de89df7a1fc512e676a0558de9587c5",
        },
    },
    "play_cards_0915_001": {
        "root": "_run/current/exact78_raw_to_robot_t2_0915_cards001_v1_20260930/"
                "attempts/attempt_0001/candidate/hawor",
        "source_kind": "vst0915", "frames": 150,
        "seal_sha": "c06a8e94788e79afd3b343147f0818f90afbe8b001a10cdaa706180f41f89d9c",
        "native_audit_sha": "49d559e4046a13424ff3086bef51176dfd86aa328460b4c0c045846ad0f150eb",
        "sha": {
            "RESULT.json": "36cc62f93430375e5cb9cbf339360a95f2ed67e36c18bd743707d3c9af546567",
            "HAWOR_CAMERA_PREDICTIONS.npz": "3aaba1de7c112ebae21ed3f459ea186d6460e2847168b6f3c4062b4c5e17519c",
            "FRAME_PROVENANCE.json": "96c124586d17aeadc02c6cb51176bdce724d8e79f0f722b813329a76bfe43595",
            "INPUT_AUDIT.json": "a8260d32d11f3a2915178b52aaa8a4414b1cc4fa31b2395150ca50e093498451",
        },
    },
}
_VIDEO_FILENAMES = {
    "ego_robot": "ego_robot_diagnostic.mp4",
    "external_full_robot": "fixed_external_full_robot.mp4",
    "raw_motion_source": "raw_motion_source_review.mp4",
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_code_pins() -> None:
    for relative, expected in _CODE_PINS.items():
        if _sha(ROOT / relative) != expected:
            raise stage_v1.FreshRobotError(f"FLANGE_SUCCESSOR_CODE_DRIFT:{relative}")


def validate_sealed_development(*, hawor_npz: Path, frame_provenance: Path,
                                input_audit: Path, raw_video: Path,
                                source_kind: str, session_id: str) -> dict[str, Any]:
    """Fail closed on anything except the two preselected, immutable HaWoR stages."""
    spec = _SEALED_DEVELOPMENT.get(session_id)
    if spec is None or source_kind != spec["source_kind"]:
        raise stage_v1.FreshRobotError("FLANGE_SUCCESSOR_DEVELOPMENT_ONLY")
    root = ROOT / spec["root"]
    exact = {
        "HAWOR_CAMERA_PREDICTIONS.npz": hawor_npz,
        "FRAME_PROVENANCE.json": frame_provenance,
        "INPUT_AUDIT.json": input_audit,
    }
    if any(path.resolve() != (root / name).resolve() for name, path in exact.items()):
        raise stage_v1.FreshRobotError("FLANGE_SUCCESSOR_SEALED_PATH_DRIFT")
    for name, expected in spec["sha"].items():
        if _sha(root / name) != expected:
            raise stage_v1.FreshRobotError(f"FLANGE_SUCCESSOR_SEALED_SHA_DRIFT:{name}")
    seal_path = root.parent / "HAWOR_STAGE_SEAL.json"
    native_path = root.parent / "NATIVE_OPEN_AUDIT.json"
    if (_sha(seal_path) != spec["seal_sha"]
            or _sha(native_path) != spec["native_audit_sha"]):
        raise stage_v1.FreshRobotError("FLANGE_SUCCESSOR_STAGE_SEAL_SHA_DRIFT")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    native = json.loads(native_path.read_text(encoding="utf-8"))
    if (seal.get("downstream_use") != "SEALED_STAGE_ONLY"
            or seal.get("session_id") != session_id
            or seal.get("result") != file_ref(root / "RESULT.json")
            or seal.get("native_open_audit") != file_ref(native_path)
            or native.get("status") != "PASS_NATIVE_OPEN_SCOPE"
            or native.get("forbidden_paths") != []):
        raise stage_v1.FreshRobotError("FLANGE_SUCCESSOR_STAGE_SEAL_INVALID")
    result = json.loads((root / "RESULT.json").read_text(encoding="utf-8"))
    audit = json.loads((root / "INPUT_AUDIT.json").read_text(encoding="utf-8"))
    if (result.get("status") != "MODEL_PREDICTIONS_CAMERA_ONLY_NO_QUALITY_PASS"
            or result.get("session_id") != session_id
            or result.get("source_frames_decoded") != spec["frames"]
            or result.get("npz") != file_ref(hawor_npz)
            or result.get("frame_provenance") != file_ref(frame_provenance)
            or result.get("input_audit") != file_ref(input_audit)
            or audit.get("infiller_invoked") is not False):
        raise stage_v1.FreshRobotError("FLANGE_SUCCESSOR_SEAL_IDENTITY_DRIFT")
    raw_ref = audit.get("raw_source_ref")
    if (not isinstance(raw_ref, dict)
            or raw_video.resolve() != Path(raw_ref.get("path", "")).resolve()
            or file_ref(raw_video) != raw_ref):
        raise stage_v1.FreshRobotError("FLANGE_SUCCESSOR_RAW_REF_DRIFT")
    return {"sealed_root": root, "source_kind": source_kind,
            "frames": spec["frames"], "raw_ref": raw_ref,
            "stage_seal": file_ref(seal_path), "native_open_audit": file_ref(native_path)}


def flange_fk(assets: Any, side: int, q: np.ndarray) -> np.ndarray:
    """Same Tianji FK as v1's tool target, selecting its flange link only."""
    if side not in (0, 1) or np.asarray(q).shape != (7,):
        raise stage_v1.FreshRobotError("FLANGE_SUCCESSOR_Q_SHAPE")
    values = {name: 0.0 for row in stage_v1.ARM_JOINT_NAMES for name in row}
    values.update(dict(zip(stage_v1.ARM_JOINT_NAMES[side], q, strict=True)))
    fk = stage_v1.forward_kinematics(assets.tianji, values)
    return fk[("flange_L", "flange_R")[side]]


@contextmanager
def _flange_link_override() -> Iterator[None]:
    """Single-process scoped substitution; restore every global on all exits."""
    _check_code_pins()
    with _OVERRIDE_LOCK:
        old_tool_fk = stage_v1._tool_fk
        old_raster_fk = raster_v1.forward_kinematics
        old_encoder = stage_v1._encoder

        def raster_fk_with_flange_parent(model: Any, values: dict[str, float]) -> dict[str, np.ndarray]:
            fk = old_raster_fk(model, values)
            if all(key in fk for key in ("flange_L", "flange_R", "left_tool", "right_tool")):
                fk = dict(fk)
                fk["left_tool"] = fk["flange_L"]
                fk["right_tool"] = fk["flange_R"]
            return fk

        stage_v1._tool_fk = flange_fk
        raster_v1.forward_kinematics = raster_fk_with_flange_parent
        stage_v1._encoder = ffmpeg44.ffmpeg44_encoder
        try:
            yield
        finally:
            stage_v1._tool_fk = old_tool_fk
            raster_v1.forward_kinematics = old_raster_fk
            stage_v1._encoder = old_encoder


def _chain_audit(states_path: Path, expected: int) -> dict[str, Any]:
    """Audit camera-frame hand roots; record motion, do not invent a quality gate."""
    with np.load(states_path, allow_pickle=False) as archive:
        q = archive["q_arm"]
        base = archive["T_camera_base"]
        mounts = archive["T_tool_hand_root"]
    if (q.shape != (expected, 2, 7) or base.shape != (4, 4)
            or mounts.shape != (2, 4, 4)
            or not all(np.isfinite(value).all() for value in (q, base, mounts))):
        raise stage_v1.FreshRobotError("FLANGE_SUCCESSOR_STATES_SHAPE")
    assets = stage_v1.load_pinned_robot_assets(ROOT)
    roots = np.empty((expected, 2, 3), np.float64)
    relative_error = 0.0
    for frame in range(expected):
        for side in range(2):
            flange = flange_fk(assets, side, q[frame, side])
            camera_flange = base @ flange
            camera_hand = camera_flange @ mounts[side]
            roots[frame, side] = camera_hand[:3, 3]
            relative = np.linalg.inv(camera_flange) @ camera_hand
            relative_error = max(relative_error, float(np.max(np.abs(relative - mounts[side]))))
    if not np.isfinite(roots).all() or relative_error > 1e-8:
        raise stage_v1.FreshRobotError("FLANGE_SUCCESSOR_CHAIN_DRIFT")
    steps = np.linalg.norm(np.diff(roots, axis=0), axis=-1)
    return {
        "schema_version": "exact78-flange-hand-root-chain-audit-v1",
        "frame_count": expected,
        "coordinate_frame": "CAMERA_DIAGNOSTIC_NOT_VERIFIED_WORLD",
        "mount_matrix_direction": "p_flange = T_flange_hand_root_proxy @ p_hand_root",
        "flange_to_hand_root_m": [float(np.linalg.norm(mounts[s, :3, 3])) for s in range(2)],
        "max_relative_matrix_error": relative_error,
        "max_consecutive_root_step_m": [float(steps[:, s].max()) for s in range(2)],
        "zero_step_pairs": [int(np.count_nonzero(steps[:, s] <= 1e-10)) for s in range(2)],
        "continuity_quality_pass": False,
        "camera_projection_chain_claim": "NOT_CHECKED: v1 renderer camera/base inversion unchanged",
        "claim_limit": "Finite camera-frame FK and fixed proxy offset only; no physical mount, world or visual qualification.",
    }


def _write_new(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def run_flange_robot_stage(*, hawor_npz: Path, frame_provenance: Path,
                           input_audit: Path, raw_video: Path,
                           source_kind: str, session_id: str,
                           output_root: Path) -> dict[str, Any]:
    """Produce a fresh Robot-only candidate; a governed caller must fence writes."""
    _check_code_pins()
    sealed = validate_sealed_development(
        hawor_npz=hawor_npz, frame_provenance=frame_provenance,
        input_audit=input_audit, raw_video=raw_video,
        source_kind=source_kind, session_id=session_id)
    output = output_root.resolve()
    if not output.is_relative_to(ROOT / "_run") or output.exists():
        raise stage_v1.FreshRobotError("FLANGE_SUCCESSOR_FRESH_PROJECT_OUTPUT_REQUIRED")
    with np.load(hawor_npz, allow_pickle=False) as archive:
        expected = len(archive["original_frame_indices"])
        fps = float(archive["fps_nominal"])
    if expected != sealed["frames"] or not math.isfinite(fps) or fps <= 0:
        raise stage_v1.FreshRobotError("FLANGE_SUCCESSOR_TIMELINE_DRIFT")
    output.mkdir(parents=True)
    stage_root = output / "flange_link_stage_ffmpeg44"
    with _flange_link_override():
        stage = stage_v1.run_robot_stage(
            hawor_npz=hawor_npz, frame_provenance=frame_provenance,
            input_audit=input_audit, raw_video=raw_video, source_kind=source_kind,
            session_id=session_id, output_root=stage_root)
    if (stage.get("status") != "COMPLETE_CAMERA_DIAGNOSTIC_NOT_VISUAL_PASS"
            or stage.get("frame_count") != expected
            or stage.get("session_id") != session_id):
        raise stage_v1.FreshRobotError("FLANGE_SUCCESSOR_STAGE_RESULT_DRIFT")
    stage_result = stage_root / "RESULT.json"
    if json.loads(stage_result.read_text(encoding="utf-8")) != stage:
        raise stage_v1.FreshRobotError("FLANGE_SUCCESSOR_STORED_RESULT_DRIFT")
    videos = stage.get("outputs", {}).get("videos", {})
    if set(videos) != set(_VIDEO_FILENAMES):
        raise stage_v1.FreshRobotError("FLANGE_SUCCESSOR_VIDEO_SET_DRIFT")
    checked_videos = {}
    for name, filename in _VIDEO_FILENAMES.items():
        record = videos[name]
        path = stage_root / "render" / filename
        if file_ref(path) != record.get("video"):
            raise stage_v1.FreshRobotError(f"FLANGE_SUCCESSOR_VIDEO_SHA_DRIFT:{name}")
        check = stage_v1._verify_video(path, expected, (stage_v1.WIDTH, stage_v1.HEIGHT), fps)
        if check != record:
            raise stage_v1.FreshRobotError(f"FLANGE_SUCCESSOR_VIDEO_DECODE_DRIFT:{name}")
        checked_videos[name] = check
    chain_path = output / "MOUNT_CHAIN_AUDIT.json"
    _write_new(chain_path, _chain_audit(stage_root / "ROBOT_CAMERA_DIAGNOSTIC_STATES.npz", expected))
    result = {
        "schema_version": "exact78-flange-link-robot-successor-result-v1",
        "status": "COMPLETE_CAMERA_DIAGNOSTIC_NOT_VISUAL_PASS",
        "session_id": session_id, "frame_count": expected,
        "source_role": "DEVELOPMENT_ONLY", "sealed_hawor_result": file_ref(sealed["sealed_root"] / "RESULT.json"),
        "hawor_stage_seal": sealed["stage_seal"],
        "hawor_native_open_audit": sealed["native_open_audit"],
        "stage_result": file_ref(stage_result), "mount_chain_audit": file_ref(chain_path),
        "verified_videos": checked_videos,
        "single_change": "IK_AND_RASTER_HAND_ROOT_PARENT_FLANGE_INSTEAD_OF_SIDE_TOOL",
        "numeric_gates_changed": False, "base_search_rule_changed": False,
        "hand_shape_rule_changed": False, "renderer_camera_base_rule_changed": False,
        "visual_qualified": False, "product_qualified": False,
        "training_eligible": False, "control_ground_truth": False,
        "claim_limit": "Development-only flange-link camera diagnostic; no physical adapter, world pose, occlusion or visual qualification.",
    }
    _write_new(output / "RESULT.json", result)
    return result
