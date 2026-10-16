"""Unrouted 0901 Chips001 Robot candidate with two structural corrections.

The caller must first publish a separate finite Task Packet and enter through
the governed single-writer operation. This module never reads a prior Robot
state or result. Its only solver/raster differences from the frozen v1 are the
flange hand-root parent and the base-to-camera render direction.
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
from chaoyang.pipeline import exact78_fresh_robot_flange_successor_v1 as flange
from chaoyang.pipeline import exact78_fresh_robot_stage_v1 as legacy
from chaoyang.pipeline import exact78_robot_render_direction_v1 as direction


ROOT = Path(__file__).resolve().parents[3]
SESSION_ID = "get_potato_chips_0901_001"
SOURCE_KIND = "tracker"
TASK_ID = "exact78_newraw_robot_get_potato_chips_0901_001_051c98741044600d_v2"
OPERATION = "run_exact78_newraw_robot_structural_v2_candidate"
FRAMES = 799
FPS = 25.0
_V1_PACKET = ROOT / "tasks/current/exact78_newraw_robot_get_potato_chips_0901_001_051c98741044600d_v1/TASK_PACKET.json"
_V1_PACKET_SHA = "351e2ecc9ec54934d4c70024555f6787679eff9fdf70dd279cea52005501c2dc"
_ASSET_PIN_SHA = "4bc508a45fd460610a73bc9fb6bf194d97a5cfdef5e863bd1069683921cd1330"
_FLANGE_MESH_SHA = "d47925eb8687a9ed9f20020e0f244878a05b37d2f882ec111c4ee277916dccc3"
_MOUNT_PROXY_SHA = "9f7258480bfa41bbbe8b87e1752c3227d397c609f4c64fb69cd46d4845c2f731"
_SOURCE_SHA = {
    "hawor_frame_provenance": "790f18ac12b453e0018a86b4164484863b0d86db9e36bcb13598b1bf4b516dd0",
    "hawor_inner_result": "cbe7ab74866ae31056454996885fbd36227d72acc89a4bb5509c5247c4130100",
    "hawor_input_audit": "8a0bf73b335e7a9b928f5454f9bf7865c9e1f6929965f74d1525ab7b158e41b7",
    "hawor_native_open_audit": "7f574ff406acdb8bd05e030a544a374b2214611cc6ea14c413671ac07545ed7b",
    "hawor_npz": "f4ba83da13ded8eb0913df73959ea4351de68b708ed391fc10b0f9f53c8bff6e",
    "hawor_outer_result": "d78580a73496a3b63e4f007f7051f0340049d5de757b278023a5581aae881d38",
    "hawor_packet": "1266b86539e2e9a91ba8fcfe8238a5afd9090c002748e86e3f215a21a8df8793",
    "hawor_receipt": "1c4d9c00da3f5a8f5e4643d4deacca374e1341ccad88850a5ed5ea314f96e235",
    "hawor_stage_seal": "dcb35ced6a63bf91d9df4a09d9e96e36365c81d4ab92693f7b21ae85eb330d9d",
    "raw_video": "643d0f880049c253ff8a00b56ff32819e97156dc231c8454df528c72d2efc0b2",
}
_CANDIDATE_CODE_SHA = {
    "src/chaoyang/pipeline/exact78_fresh_robot_flange_successor_v1.py":
        "0df9c81e3ee2226212d673e3186cd6169fac2d315b1fc410ec3bfbf85556188c",
    "src/chaoyang/pipeline/exact78_robot_render_direction_v1.py":
        "8158e106ee3f12fe0cdafc950349ee5f92d15eba1312a5af3f731648f1565913",
}
_REQUIRED_ADMISSION = (
    "upstream_admission_descriptor", "upstream_admission_frame_time_map",
    "upstream_admission_seal", "upstream_admission_result",
    "upstream_admission_receipt",
)
_VIDEO_NAMES = {
    "ego_robot": "ego_robot_diagnostic.mp4",
    "external_full_robot": "fixed_external_full_robot.mp4",
    "raw_motion_source": "raw_motion_source_review.mp4",
}
_OVERRIDE_LOCK = threading.Lock()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise legacy.FreshRobotError("STRUCTURAL_V2_JSON_OBJECT_REQUIRED")
    return value


def _check_code() -> None:
    flange._check_code_pins()
    direction.verify_code_pins()
    for relative, expected in _CANDIDATE_CODE_SHA.items():
        if _sha(ROOT / relative) != expected:
            raise legacy.FreshRobotError(f"STRUCTURAL_V2_CODE_DRIFT:{relative}")


def _source_inputs() -> dict[str, dict[str, Any]]:
    """Use the v1 packet only as a pinned source manifest, never as motion."""
    if _sha(_V1_PACKET) != _V1_PACKET_SHA:
        raise legacy.FreshRobotError("STRUCTURAL_V2_SOURCE_MANIFEST_DRIFT")
    packet = _json(_V1_PACKET)
    if (packet.get("task_id") != TASK_ID.removesuffix("_v2") + "_v1"
            or packet.get("config", {}).get("frame_count") != FRAMES
            or packet.get("config", {}).get("source_kind") != SOURCE_KIND):
        raise legacy.FreshRobotError("STRUCTURAL_V2_SOURCE_IDENTITY_DRIFT")
    inputs = packet.get("inputs")
    if not isinstance(inputs, dict):
        raise legacy.FreshRobotError("STRUCTURAL_V2_INPUTS_ABSENT")
    result: dict[str, dict[str, Any]] = {}
    for role, expected_sha in _SOURCE_SHA.items():
        ref = inputs.get(role)
        if (not isinstance(ref, dict) or set(ref) != {"path", "bytes", "sha256"}
                or ref["sha256"] != expected_sha):
            raise legacy.FreshRobotError(f"STRUCTURAL_V2_SOURCE_REF_DRIFT:{role}")
        path = Path(ref["path"])
        if (not path.is_file() or path.is_symlink()
                or file_ref(path) != ref):
            raise legacy.FreshRobotError(f"STRUCTURAL_V2_SOURCE_SHA_DRIFT:{role}")
        result[role] = ref
    for role in _REQUIRED_ADMISSION:
        ref = inputs.get(role)
        if (not isinstance(ref, dict) or set(ref) != {"path", "bytes", "sha256"}
                or file_ref(Path(ref["path"])) != ref):
            raise legacy.FreshRobotError(f"STRUCTURAL_V2_ADMISSION_DRIFT:{role}")
    assets = inputs.get("robot_assets")
    if not isinstance(assets, dict) or not assets:
        raise legacy.FreshRobotError("STRUCTURAL_V2_ASSETS_ABSENT")
    for role, expected_sha in (("asset_pin", _ASSET_PIN_SHA),
                               ("flange_mesh", _FLANGE_MESH_SHA),
                               ("mount_proxy", _MOUNT_PROXY_SHA)):
        if assets.get(role, {}).get("sha256") != expected_sha:
            raise legacy.FreshRobotError(f"STRUCTURAL_V2_ASSET_PIN_DRIFT:{role}")
    for role, ref in assets.items():
        if (not isinstance(ref, dict) or set(ref) != {"path", "bytes", "sha256"}
                or file_ref(Path(ref["path"])) != ref):
            raise legacy.FreshRobotError(f"STRUCTURAL_V2_ASSET_SHA_DRIFT:{role}")
    outer = _json(Path(result["hawor_outer_result"]["path"]))
    inner = _json(Path(result["hawor_inner_result"]["path"]))
    seal = _json(Path(result["hawor_stage_seal"]["path"]))
    receipt = _json(Path(result["hawor_receipt"]["path"]))
    native = _json(Path(result["hawor_native_open_audit"]["path"]))
    if (outer.get("status") != "PASSED"
            or outer.get("stage", {}).get("prediction_counts") != {"left": FRAMES, "right": FRAMES}
            or inner.get("session_id") != SESSION_ID
            or inner.get("source_frames_decoded") != FRAMES
            or seal.get("session_id") != SESSION_ID
            or seal.get("downstream_use") != "SEALED_STAGE_ONLY"
            or receipt.get("status") != "PASSED"
            or native.get("status") != "PASS_NATIVE_OPEN_SCOPE"):
        raise legacy.FreshRobotError("STRUCTURAL_V2_HAWOR_SEAL_INVALID")
    return result


def _verify_active_packet(source_refs: dict[str, dict[str, Any]]) -> None:
    """Future v2 packet must bind the same source/asset manifest and this code."""
    path = ROOT / "tasks/current" / TASK_ID / "TASK_PACKET.json"
    packet = _json(path)
    inputs = packet.get("inputs", {})
    original = _json(_V1_PACKET)["inputs"]
    if (packet.get("task_id") != TASK_ID
            or packet.get("execution_mode") != "EXACT78_CANDIDATE"
            or any(inputs.get(role) != ref for role, ref in source_refs.items())
            or inputs.get("robot_assets") != original.get("robot_assets")
            or any(inputs.get(role) != original.get(role) for role in _REQUIRED_ADMISSION)
            or str(ROOT / "_run/current" / TASK_ID / "attempts/attempt_0001")
            not in {str(ROOT / item) for item in packet.get("write_set", [])}):
        raise legacy.FreshRobotError("STRUCTURAL_V2_ACTIVE_PACKET_BINDING_DRIFT")
    code = packet.get("runtime_code")
    this_path = Path(__file__).resolve()
    if (not isinstance(code, list)
            or not any(ref.get("path") == str(this_path)
                       and file_ref(this_path) == ref for ref in code if isinstance(ref, dict))):
        raise legacy.FreshRobotError("STRUCTURAL_V2_CODE_NOT_PINNED_BY_PACKET")


def check_only() -> dict[str, Any]:
    """CPU-only source/code check; does not solve, render, or create files."""
    _check_code()
    inputs = _source_inputs()
    hawor, _, _ = legacy.load_fresh_inputs(
        Path(inputs["hawor_npz"]["path"]),
        Path(inputs["hawor_frame_provenance"]["path"]),
        Path(inputs["hawor_input_audit"]["path"]),
        Path(inputs["raw_video"]["path"]), SESSION_ID)
    if (len(hawor["original_frame_indices"]) != FRAMES
            or float(hawor["fps_nominal"]) != FPS
            or hawor["prediction_valid"].shape != (2, FRAMES)
            or not bool(hawor["prediction_valid"].all())):
        raise legacy.FreshRobotError("STRUCTURAL_V2_HAWOR_TIMELINE_DRIFT")
    return {"status": "CHECK_ONLY_READY_NOT_ROUTABLE", "session_id": SESSION_ID,
            "frame_count": FRAMES, "fps_nominal": FPS, "source_refs": inputs,
            "model_calls": 0, "render_calls": 0, "old_robot_state_consumed": False}


@contextmanager
def _structural_override() -> Iterator[None]:
    """Both changes share one scoped, single-threaded process-local override."""
    _check_code()
    if threading.current_thread() is not threading.main_thread() or threading.active_count() != 1:
        raise legacy.FreshRobotError("STRUCTURAL_V2_DEDICATED_PROCESS_REQUIRED")
    with _OVERRIDE_LOCK, flange._flange_link_override():
        original_render = legacy.render_camera_diagnostic
        legacy.render_camera_diagnostic = direction.render_camera_diagnostic
        try:
            yield
        finally:
            legacy.render_camera_diagnostic = original_render


def _mount_camera_audit(states_path: Path) -> dict[str, Any]:
    """Recompute the expected raster hand roots from sealed q/base/mount."""
    with np.load(states_path, allow_pickle=False) as archive:
        q = archive["q_arm"]
        base = archive["T_camera_base"]
        mounts = archive["T_tool_hand_root"]
    if (q.shape != (FRAMES, 2, 7) or base.shape != (4, 4)
            or mounts.shape != (2, 4, 4)
            or not all(np.isfinite(item).all() for item in (q, base, mounts))):
        raise legacy.FreshRobotError("STRUCTURAL_V2_STATE_SHAPE_DRIFT")
    assets = legacy.load_pinned_robot_assets(ROOT)
    roots = np.empty((FRAMES, 2, 3), np.float64)
    maximum = 0.0
    for frame in range(FRAMES):
        for side in range(2):
            base_flange = flange.flange_fk(assets, side, q[frame, side])
            camera_flange = base @ base_flange
            camera_hand = camera_flange @ mounts[side]
            roots[frame, side] = camera_hand[:3, 3]
            maximum = max(maximum, float(np.max(np.abs(
                np.linalg.inv(camera_flange) @ camera_hand - mounts[side]))))
    if not np.isfinite(roots).all() or maximum > 1e-8:
        raise legacy.FreshRobotError("STRUCTURAL_V2_MOUNT_CHAIN_DRIFT")
    steps = np.linalg.norm(np.diff(roots, axis=0), axis=-1)
    return {"schema_version": "exact78-newraw-robot-structural-v2-chain-audit-v1",
            "frame_count": FRAMES, "camera_rule": "BASE_TO_CAMERA_NO_INVERSION",
            "hand_root_rule": "CAMERA_AT_FLANGE_LR_AT_PROXY_MOUNT",
            "max_relative_matrix_error": maximum,
            "flange_to_hand_root_m": [float(np.linalg.norm(mounts[s, :3, 3])) for s in range(2)],
            "max_consecutive_root_step_m": [float(steps[:, s].max()) for s in range(2)],
            "zero_step_pairs": [int(np.count_nonzero(steps[:, s] <= 1e-10)) for s in range(2)],
            "continuity_quality_pass": False, "physical_mount_verified": False}


def _write_new(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def run_candidate(*, output_root: Path) -> dict[str, Any]:
    """Future fenced worker call; never use before a v2 packet/route exists."""
    from chaoyang.governance.exact78_operation_gate import authorize_exact78_operation

    authorize_exact78_operation(OPERATION, ["--output-root", str(output_root)])
    checked = check_only()
    output = output_root.resolve()
    expected = ROOT / "_run/current" / TASK_ID / "attempts/attempt_0001/candidate/robot_v2"
    if output != expected or output.exists():
        raise legacy.FreshRobotError("STRUCTURAL_V2_FRESH_OUTPUT_REQUIRED")
    _verify_active_packet(checked["source_refs"])
    if not math.isfinite(FPS):
        raise legacy.FreshRobotError("STRUCTURAL_V2_FPS_INVALID")
    refs = checked["source_refs"]
    output.mkdir(parents=True)
    stage_root = output / "structural_stage_ffmpeg44"
    with _structural_override():
        stage = legacy.run_robot_stage(
            hawor_npz=Path(refs["hawor_npz"]["path"]),
            frame_provenance=Path(refs["hawor_frame_provenance"]["path"]),
            input_audit=Path(refs["hawor_input_audit"]["path"]),
            raw_video=Path(refs["raw_video"]["path"]),
            source_kind=SOURCE_KIND, session_id=SESSION_ID, output_root=stage_root)
    if (stage.get("status") != "COMPLETE_CAMERA_DIAGNOSTIC_NOT_VISUAL_PASS"
            or stage.get("frame_count") != FRAMES
            or stage.get("session_id") != SESSION_ID
            or _json(stage_root / "RESULT.json") != stage):
        raise legacy.FreshRobotError("STRUCTURAL_V2_STAGE_RESULT_DRIFT")
    videos = stage.get("outputs", {}).get("videos", {})
    if set(videos) != set(_VIDEO_NAMES):
        raise legacy.FreshRobotError("STRUCTURAL_V2_VIDEO_SET_DRIFT")
    for role, filename in _VIDEO_NAMES.items():
        path = stage_root / "render" / filename
        if (file_ref(path) != videos[role]["video"]
                or legacy._verify_video(path, FRAMES, (legacy.WIDTH, legacy.HEIGHT), FPS)
                != videos[role]):
            raise legacy.FreshRobotError(f"STRUCTURAL_V2_VIDEO_DRIFT:{role}")
    audit_path = output / "MOUNT_CAMERA_CHAIN_AUDIT.json"
    _write_new(audit_path, _mount_camera_audit(stage_root / "ROBOT_CAMERA_DIAGNOSTIC_STATES.npz"))
    result = {"schema_version": "exact78-newraw-robot-structural-v2-candidate-result-v1",
              "status": "COMPLETE_CAMERA_DIAGNOSTIC_NOT_VISUAL_PASS",
              "session_id": SESSION_ID, "frame_count": FRAMES, "fps_nominal": FPS,
              "source_hawor_result": refs["hawor_outer_result"],
              "stage_result": file_ref(stage_root / "RESULT.json"),
              "mount_camera_chain_audit": file_ref(audit_path),
              "video_refs": {role: videos[role]["video"] for role in _VIDEO_NAMES},
              "base_search_rule_changed": False, "numeric_gates_changed": False,
              "hand_arm_solver_changed_other_than_flange_parent": False,
              "display_acceleration_changed": False,
              "ego_camera_rule": "BASE_TO_CAMERA_NO_INVERSION",
              "hand_root_parent": "FLANGE_LR",
              "old_robot_state_consumed": False,
              "visual_qualified": False, "product_qualified": False,
              "training_eligible": False, "control_ground_truth": False,
              "human_review": "PENDING"}
    _write_new(output / "RESULT.json", result)
    return result


__all__ = ["SESSION_ID", "TASK_ID", "OPERATION", "check_only", "run_candidate"]
