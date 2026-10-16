"""Candidate render-only camera-direction successor; no IK or mount changes.

The render function is the pinned v1 body with exactly one expression changed:
``ego_camera = states["T_camera_base"]`` rather than its inverse.  Its encoder
is the already sealed FFmpeg 4.4 compatibility implementation.  It is not a
routable operation until a new finite Task Packet binds its code and inputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from chaoyang.governance.common import REPO_ROOT, artifact_ref
from chaoyang.pipeline import exact78_fresh_robot_stage_v1 as legacy
from chaoyang.pipeline import exact78_fresh_robot_stage_v2 as ffmpeg44


FreshRobotError = legacy.FreshRobotError
WIDTH, HEIGHT, ROOT, FLANGE = legacy.WIDTH, legacy.HEIGHT, legacy.ROOT, legacy.FLANGE
raster = legacy.raster
load_pinned_robot_assets = legacy.load_pinned_robot_assets
_fixed_external_camera = legacy._fixed_external_camera
_verify_video = legacy._verify_video
_encoder = ffmpeg44.ffmpeg44_encoder

PINNED_CODE_SHA = {
    "src/chaoyang/pipeline/exact78_fresh_robot_stage_v1.py":
        "9018d49be9030ec84c48e470f5b600ac828665bfe7e974b64b5f0736375b196c",
    "src/chaoyang/pipeline/exact78_fresh_robot_stage_v2.py":
        "1876c5af135bd97d9e7227b076b1879ce402dc47704c260c15a6b3b0220072c9",
    "src/chaoyang/pipeline/exact78_robot_raster_v1.py":
        "9678646b8b2d02567aa4bf64c68e339921c56d204299660894b7045f82ce6ed6",
    "src/chaoyang/pipeline/robot_scene_state_cpu.py":
        "8680b6d26dc4703f6e3c099fc8b7a937f92cfdbbe2ba3ed073f693b522839793",
    "src/chaoyang/pipeline/robot_renderer_eevee_fullchain.py":
        "9f4354812a990fce6a34b8da649c7fa81dcbcee4ca88d64dd16a6d7df395740f",
    "src/chaoyang/pipeline/robot_tool_definition_contract.py":
        "c37cdacf21a0bade4f1679941745462a86a0d6a8774deb77e0c7a4b23d09b761",
}
PINNED_ASSET_PIN_SHA = "4bc508a45fd460610a73bc9fb6bf194d97a5cfdef5e863bd1069683921cd1330"
STATE_KEYS = frozenset({
    "q_arm", "q_hand", "T_camera_base", "T_tool_hand_root",
    "display_joints_3d_camera", "prediction_valid", "display_drawable",
    "short_gap", "motion_source", "edge_gap_kind", "original_frame_indices",
    "fps_nominal",
})
INPUT_KEYS = frozenset({"stage_result", "states", "hawor_npz", "frame_provenance",
                        "input_audit", "raw_video"})


def verify_code_pins() -> None:
    for relative, sha in PINNED_CODE_SHA.items():
        if artifact_ref(REPO_ROOT / relative)["sha256"] != sha:
            raise FreshRobotError(f"RENDER_ONLY_CODE_SHA_DRIFT:{relative}")
    if artifact_ref(REPO_ROOT / "assets/robot/ROBOT_ASSET_PIN.json")["sha256"] != PINNED_ASSET_PIN_SHA:
        raise FreshRobotError("RENDER_ONLY_ASSET_PIN_SHA_DRIFT")


def _ref(ref: dict[str, Any], label: str) -> Path:
    if not isinstance(ref, dict) or set(ref) != {"path", "bytes", "sha256"}:
        raise FreshRobotError(f"RENDER_ONLY_INPUT_REF_INVALID:{label}")
    path = Path(ref["path"])
    if artifact_ref(path) != ref:
        raise FreshRobotError(f"RENDER_ONLY_INPUT_SHA_DRIFT:{label}")
    return path


def preflight(spec: dict[str, Any]) -> tuple[dict[str, np.ndarray], np.ndarray, float]:
    """Read only the sealed q/base/mount, original Raw, and same-stage HaWoR."""
    verify_code_pins()
    if set(spec) != {"task_id", "session_id", "source_kind", "inputs", "output_root"}:
        raise FreshRobotError("RENDER_ONLY_SPEC_ROLES")
    if not isinstance(spec["inputs"], dict) or set(spec["inputs"]) != INPUT_KEYS:
        raise FreshRobotError("RENDER_ONLY_INPUT_ROLES")
    paths = {key: _ref(spec["inputs"][key], key) for key in INPUT_KEYS}
    output = Path(spec["output_root"]).resolve()
    expected = REPO_ROOT / "_run/current" / spec["task_id"] / "attempts/attempt_0001/candidate"
    if (not isinstance(spec["task_id"], str) or not spec["task_id"].startswith("exact78_")
            or output != expected or output.exists()):
        raise FreshRobotError("RENDER_ONLY_FRESH_PROJECT_OUTPUT_REQUIRED")
    stage = json.loads(paths["stage_result"].read_text(encoding="utf-8"))
    if (stage.get("status") != "COMPLETE_CAMERA_DIAGNOSTIC_NOT_VISUAL_PASS"
            or stage.get("session_id") != spec["session_id"]
            or stage.get("outputs", {}).get("states") != spec["inputs"]["states"]):
        raise FreshRobotError("RENDER_ONLY_SEALED_STAGE_IDENTITY")
    for key in ("hawor_npz", "frame_provenance", "input_audit", "raw_video"):
        if stage.get("inputs", {}).get(key) != spec["inputs"][key]:
            raise FreshRobotError(f"RENDER_ONLY_STAGE_INPUT_DRIFT:{key}")
    hawor, provenance, audit = legacy.load_fresh_inputs(
        paths["hawor_npz"], paths["frame_provenance"],
        paths["input_audit"], paths["raw_video"], spec["session_id"])
    expected_kind = "tracker" if spec["source_kind"] in {"TRACKER_CLIP_RGB", "tracker"} else "vst0915"
    if (spec["source_kind"] not in {"TRACKER_CLIP_RGB", "tracker", "0915_RAW_VST", "vst0915"}
            or audit.get("source_kind") != expected_kind
            or (expected_kind == "vst0915" and provenance.get("physical_eye") != "left_sourceIndex_1")):
        raise FreshRobotError("RENDER_ONLY_SOURCE_OR_EYE_DRIFT")
    with np.load(paths["states"], allow_pickle=False) as archive:
        if set(archive.files) != STATE_KEYS:
            raise FreshRobotError("RENDER_ONLY_STATE_ROLE_DRIFT")
        states = {key: np.asarray(archive[key]).copy() for key in STATE_KEYS}
    count = len(hawor["original_frame_indices"])
    fps = float(hawor["fps_nominal"])
    if (stage.get("frame_count") != count or states["q_arm"].shape != (count, 2, 7)
            or states["q_hand"].shape != (count, 2, 22)
            or states["T_camera_base"].shape != (4, 4)
            or states["T_tool_hand_root"].shape != (2, 4, 4)
            or not np.isfinite(states["T_camera_base"]).all()
            or not np.isfinite(states["T_tool_hand_root"]).all()
            or not np.array_equal(states["original_frame_indices"], hawor["original_frame_indices"])
            or float(states["fps_nominal"]) != fps):
        raise FreshRobotError("RENDER_ONLY_STATE_DOMAIN_DRIFT")
    return states, hawor["intrinsics"], fps


def dry_run(spec: dict[str, Any]) -> dict[str, Any]:
    """Read-only candidate preview; this does not authorize or invoke rendering."""
    states, _, fps = preflight(spec)
    output = Path(spec["output_root"])
    return {
        "status": "DRY_RUN_READY_NOT_ROUTABLE", "task_id": spec["task_id"],
        "session_id": spec["session_id"], "frame_count": len(states["q_arm"]),
        "nominal_fps": fps, "gpu_count": 0, "model_calls": 0,
        "attempt_write_set_first": str(output.parent.relative_to(REPO_ROOT)),
        "candidate_result": str(output / "RESULT.json"),
        "new_video_names": ["ego_robot_diagnostic.mp4", "fixed_external_full_robot.mp4",
                            "raw_motion_source_review.mp4"],
        "old_render_video_consumed": False,
        "q_base_mount_recomputed": False,
        "next_action": "Register a separate finite packet before any render call",
    }


def render_camera_diagnostic(*, states: dict[str, np.ndarray], intrinsics: np.ndarray,
                             raw_video: Path, source_kind: str, fps: float,
                             output_root: Path) -> dict[str, Any]:
    """Render all N raw/display frames without masks, Clean, or depth claims."""
    if source_kind not in {"TRACKER_CLIP_RGB", "0915_RAW_VST", "tracker", "vst0915"}:
        raise FreshRobotError("RAW_SOURCE_KIND_UNSUPPORTED")
    count = len(states["q_arm"])
    output_root.mkdir(parents=True, exist_ok=False)
    temp_dir = output_root / "tmp"
    temp_dir.mkdir()
    videos = {"ego_robot": output_root / "ego_robot_diagnostic.mp4",
              "external_full_robot": output_root / "fixed_external_full_robot.mp4",
              "raw_motion_source": output_root / "raw_motion_source_review.mp4"}
    encoders = {name: _encoder(path, fps, WIDTH, HEIGHT, temp_dir) for name, path in videos.items()}
    capture = cv2.VideoCapture(str(raw_video))
    if not capture.isOpened():
        raise FreshRobotError("RAW_VIDEO_DECODE_OPEN")
    assets = load_pinned_robot_assets(ROOT)
    mounts = states["T_tool_hand_root"]
    flange = raster.naturalv2_local_triangles(FLANGE)
    cache: dict[Path, object] = {}
    ego_camera = states["T_camera_base"]
    external_camera = _fixed_external_camera(ego_camera)
    external_k = np.asarray(((575.0, 0.0, 319.5), (0.0, 575.0, 239.5), (0.0, 0.0, 1.0)))
    rendered = []
    try:
        for frame in range(count):
            ok, raw = capture.read()
            if not ok:
                raise FreshRobotError(f"RAW_DECODE_EARLY_EOF:{frame}")
            if source_kind in {"0915_RAW_VST", "vst0915"}:
                if raw.shape[1] != 4096 or raw.shape[0] != 1536:
                    raise FreshRobotError(f"RAW_SBS_SHAPE_DRIFT:{frame}")
                raw = raw[:, 2048:4096]  # physical left is sourceIndex 1
            elif raw.shape[1] != 1280 or raw.shape[0] != 960:
                raise FreshRobotError(f"RAW_TRACKER_SHAPE_DRIFT:{frame}")
            raw_small = cv2.resize(raw, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
            k = intrinsics[frame].astype(np.float64).copy()
            k[0] *= WIDTH / 1280.0
            k[1] *= HEIGHT / 960.0
            ego_rgb, ego_labels, ego_buffers = raster.render_robot(
                raster, assets, states["q_arm"][frame], states["q_hand"][frame],
                mounts, ego_camera, k, cache, flange, complete_robot=True,
                return_buffers=True, width=WIDTH, height=HEIGHT)
            external_rgb, external_labels, external_buffers = raster.render_robot(
                raster, assets, states["q_arm"][frame], states["q_hand"][frame],
                mounts, external_camera, external_k, cache, flange, complete_robot=True,
                return_buffers=True, width=WIDTH, height=HEIGHT)
            ego_alpha = ego_labels >= 0
            ext_alpha = external_labels >= 0
            if (not np.isfinite(ego_buffers["depth_m"][ego_alpha]).all()
                    or not np.isfinite(external_buffers["depth_m"][ext_alpha]).all()):
                raise FreshRobotError(f"ROBOT_DEPTH_NONFINITE:{frame}")
            ego = raw_small.copy()
            ego[ego_alpha] = ego_rgb[ego_alpha]
            external = np.full_like(raw_small, 242)
            external[ext_alpha] = external_rgb[ext_alpha]
            for panel, label in ((ego, "CAMERA DIAGNOSTIC - NO OCCLUSION PROOF"),
                                 (external, "FIXED THIRD PERSON - FULL ROBOT"),
                                 (raw_small, "RAW RGB + HAWOR SOURCE AUDIT")):
                cv2.rectangle(panel, (0, 0), (WIDTH, 28), (0, 0, 0), -1)
                cv2.putText(panel, f"{label}  F{frame}", (5, 19), cv2.FONT_HERSHEY_SIMPLEX,
                            .42, (255, 255, 255), 1, cv2.LINE_AA)
            left_source = str(states["motion_source"][0, frame])
            right_source = str(states["motion_source"][1, frame])
            cv2.rectangle(raw_small, (0, HEIGHT - 45), (WIDTH, HEIGHT), (0, 0, 0), -1)
            cv2.putText(raw_small, f"L {left_source}", (5, HEIGHT - 27),
                        cv2.FONT_HERSHEY_SIMPLEX, .40, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(raw_small, f"R {right_source}", (5, HEIGHT - 9),
                        cv2.FONT_HERSHEY_SIMPLEX, .40, (255, 255, 255), 1, cv2.LINE_AA)
            for name, panel in (("ego_robot", ego), ("external_full_robot", external),
                                ("raw_motion_source", raw_small)):
                assert encoders[name].stdin is not None
                encoders[name].stdin.write(panel.tobytes())
            rendered.append({"frame": frame, "raw_decoded_display_index": frame,
                             "ego_robot_pixels": int(ego_alpha.sum()),
                             "external_robot_pixels": int(ext_alpha.sum()),
                             "robot_drawable": True,
                             "final_visible": bool(ego_alpha.any()),
                             "geometry_decision": "UNKNOWN",
                             "display_policy": "DIAGNOSTIC_ROBOT_OVER_RAW_NO_OCCLUSION_PROOF"})
        extra, _ = capture.read()
        if extra:
            raise FreshRobotError("RAW_DECODE_EXTRA_FRAMES")
        for worker in encoders.values():
            assert worker.stdin is not None
            worker.stdin.close()
            if worker.wait(timeout=120):
                raise FreshRobotError("VIDEO_ENCODER_FAILED")
    finally:
        capture.release()
        for worker in encoders.values():
            if worker.poll() is None:
                worker.terminate()
                worker.wait(timeout=10)
    checks = {name: _verify_video(path, count, (WIDTH, HEIGHT), fps) for name, path in videos.items()}
    return {"videos": checks, "frame_rows": rendered,
            "frame_count": count, "nominal_duration_seconds": count / fps,
            "duration_basis": "N_OVER_NOMINAL_FPS_NOT_CAPTURE_SYNCHRONIZATION"}


def run_render_only(spec: dict[str, Any]) -> dict[str, Any]:
    """Future governed worker entry: consume sealed states, render new videos only."""
    states, intrinsics, fps = preflight(spec)
    output = Path(spec["output_root"]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    render = render_camera_diagnostic(
        states=states, intrinsics=intrinsics,
        raw_video=Path(spec["inputs"]["raw_video"]["path"]),
        source_kind=spec["source_kind"], fps=fps, output_root=output / "render")
    if render["frame_count"] != len(states["q_arm"]):
        raise FreshRobotError("RENDER_ONLY_FRAME_COUNT_DRIFT")
    for key, ref in spec["inputs"].items():
        _ref(ref, key)
    result = {
        "schema_version": "exact78-robot-render-direction-only-v1",
        "status": "COMPLETE_RENDER_DIAGNOSTIC_NOT_VISUAL_PASS",
        "session_id": spec["session_id"], "frame_count": render["frame_count"],
        "inputs": spec["inputs"], "videos": render["videos"],
        "frame_rows": render["frame_rows"],
        "ego_camera_rule": "T_CAMERA_BASE_BASE_TO_CAMERA_NO_INVERSION",
        "external_camera_rule": "PINNED_FIXED_EXTERNAL_FROM_BASE_TO_CAMERA",
        "q_base_mount_recomputed": False, "numeric_gates_changed": False,
        "clean_or_depth_consumed": False, "visual_qualified": False,
        "product_qualified": False, "training_eligible": False,
        "control_ground_truth": False, "human_review": "PENDING",
        "claim_limit": "Render-only camera-frame diagnostic; no IK, mount, flange, occlusion, world/timebase or quality repair",
    }
    with (output / "RESULT.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return result


__all__ = ["PINNED_CODE_SHA", "preflight", "dry_run",
           "render_camera_diagnostic", "run_render_only"]
