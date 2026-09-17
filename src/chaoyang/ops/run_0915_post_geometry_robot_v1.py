#!/usr/bin/env python3
"""Run the governed 0915 Object6D/Clean/Contact/Robot post stage.

All joins are by the frozen 220-session identity.  Geometry consumes only
physical-left SAM3.1 masks and FoundationStereo optical-Z.  Clean is a
downstream visual layer and can never feed Object6D or Contact.  Robot Visual
uses the development-only relative-motion solver; strict Contact-aware Robot
remains fail-closed when physical calibration is absent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable
import uuid

import cv2
import jsonschema
import numpy as np

from chaoyang.governance.campaign_0915_task_specs_v1 import build_packet
from chaoyang.pipeline.clean_visual_evidence_v1 import compose_clean_visual
from chaoyang.pipeline.contact_visible_tactile_hypothesis_v1 import (
    ContactHypothesisError,
    tactile_visible_surface_hypotheses,
)
from chaoyang.pipeline.full_funnel_ledger_v1 import (
    initialize as initialize_ledger,
    update_stage,
)
from chaoyang.pipeline.object6d_visible_surface_v1 import estimate_visible_object
from chaoyang.pipeline.robot_renderer_eevee_fullchain import load_pinned_robot_assets
from chaoyang.pipeline.robot_visual_relative_v1 import solve_robot_visual


ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "0915_post_geometry_robot_v1"
PACKET = ROOT / "tasks/current" / TASK_ID / "TASK_PACKET.json"
STATE = ROOT / "docs/governance/LONG_HORIZON_TASK_STATE.json"
INDEX = ROOT / "tasks/current/INDEX.json"
INPUT_ATTEMPT = ROOT / "_run/current/0915_input_prepare_cad_v2/attempts/attempt_0001"
HAWOR_ATTEMPT = ROOT / "_run/current/0915_hawor_full_v1/attempts/attempt_0001/hawor"
MASK_ATTEMPT = ROOT / "_run/current/0915_sam31_mask_full_v1/attempts/attempt_0001/sam31"
DEPTH_ATTEMPT = ROOT / "_run/current/0915_foundationstereo_full_v1/attempts/attempt_0001/depth"
ROBOT_SCHEMA = ROOT / "contracts/robot_visual_sidecar_v1.schema.json"
HAWOR_WEIGHT = "HAWOR_INFERENCE_BUNDLE_V1"
SAM_WEIGHT = "SAM3.1_ONLY_USER_LOCKED"
DEPTH_WEIGHT = "FoundationStereo:model_best_bp2.pth"
CALIBRATION_ID = "SAME_SESSION_EQUIDIS62_PHYSICAL_LEFT_INTERNAL_OPTICAL_Z"
TIP_INDICES = np.asarray((4, 8, 12, 16, 20), dtype=np.int64)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ref(path: Path, *, published_path: Path | None = None) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    public = (published_path or resolved).resolve(strict=False)
    return {"path": str(public), "bytes": resolved.stat().st_size,
            "sha256": sha256(resolved)}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True,
                  allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                    allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def validate_route() -> dict[str, Any]:
    expected = build_packet(TASK_ID)
    packet = load(PACKET)
    if packet != expected or packet.get("weights") != "ABSENT":
        raise RuntimeError("post task packet differs from frozen weights-ABSENT spec")
    state = load(STATE)
    if state.get("next_task", {}).get("task_id") != TASK_ID:
        raise RuntimeError("post task is not current next_task")
    route = next((row for row in load(INDEX).get("task_packets", [])
                  if row.get("task_id") == TASK_ID), None)
    if route is None or route.get("execution_allowed") is not True:
        raise RuntimeError("post task is not authorized by current index")
    if route.get("packet_sha256") != sha256(PACKET):
        raise RuntimeError("post task packet SHA mismatch")
    return packet


def heartbeat() -> None:
    completed = subprocess.run([
        sys.executable, "-m", "chaoyang.governance.heartbeat_task",
        "--task-id", TASK_ID, "--pid", str(os.getpid()), "--status", "RUNNING",
        "--phase", "0915_OBJECT6D_CLEAN_CONTACT_ROBOT_LEDGER",
    ], cwd=ROOT, capture_output=True, text=True, check=False)
    if completed.returncode:
        raise RuntimeError((completed.stderr or completed.stdout)[-4000:])


def terminal(status: str) -> str:
    return {
        "PASS": "PASS",
        "PASS_DEVELOPMENT_HAWOR": "PASS",
        "REJECTED_QUALITY": "REJECTED_QUALITY",
        "FAILED_QUALITY_C": "REJECTED_QUALITY",
        "BLOCKED_UPSTREAM": "BLOCKED_UPSTREAM",
        "FAILED_RUNTIME": "FAILED_RUNTIME",
    }.get(status, "FAILED_RUNTIME")


def signature(*values: str) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _mask(mask_root: Path, instance: dict[str, Any], frame: int,
          shape: tuple[int, int]) -> np.ndarray:
    path = mask_root / instance["mask_directory"] / f"{frame:05d}.png"
    if not path.is_file():
        return np.zeros(shape, dtype=bool)
    value = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if value is None or value.shape != shape:
        raise RuntimeError(f"mask decode/shape mismatch: {path}")
    return np.asarray(value > 0, dtype=bool)


def _task_masks(mask_root: Path, manifest: dict[str, Any], frame: int,
                shape: tuple[int, int]) -> list[tuple[str, np.ndarray]]:
    return [
        (row["instance_id"], _mask(mask_root, row, frame, shape))
        for row in manifest["instances"] if row["role"] == "task_object"
    ]


def _role_masks(mask_root: Path, manifest: dict[str, Any], frame: int,
                shape: tuple[int, int]) -> dict[str, np.ndarray]:
    roles: dict[str, np.ndarray] = {}
    for instance in manifest["instances"]:
        role = instance["role"]
        current = _mask(mask_root, instance, frame, shape)
        roles[role] = roles.get(role, np.zeros(shape, bool)) | current
    return roles


def _representative_slots(frame_count: int) -> set[int]:
    return set(np.unique(np.linspace(0, frame_count - 1, min(6, frame_count), dtype=np.int64)).tolist())


def _review_sheet(panels: list[np.ndarray]) -> np.ndarray:
    if not panels:
        return np.zeros((240, 320, 3), np.uint8)
    resized = [cv2.resize(panel, (320, 240), interpolation=cv2.INTER_AREA)
               for panel in panels]
    while len(resized) < 6:
        resized.append(np.zeros_like(resized[0]))
    return np.concatenate((np.concatenate(resized[:3], axis=1),
                           np.concatenate(resized[3:6], axis=1)), axis=0)


def process_visual_geometry(
    *, task: str, session_id: str, frame_count: int, source_session: Path,
    prepared_video: Path, mask_root: Path, depth_root: Path, hawor_npz: Path,
    output: Path, published_output: Path,
) -> dict[str, Any]:
    manifest = load(mask_root / "ROLE_MANIFEST.json")
    registration = load(depth_root / "REGISTRATION.json")
    homography = np.asarray(
        registration["H_depth_pixel_to_primary_physical_left_pixel"], np.float64,
    )
    primary_to_depth = np.asarray(
        registration["T_rectified_left_camera_to_primary_left_camera"], np.float64,
    )
    depth_to_primary_rotation = primary_to_depth[:3, :3]
    primary_to_depth_rotation = depth_to_primary_rotation.T
    depth_intrinsics = np.asarray(registration["depth_intrinsics"], np.float64)
    with np.load(hawor_npz, allow_pickle=False) as archive:
        joints_camera = np.asarray(archive["joints_3d_camera"], np.float64)
        observed = np.asarray(archive["observed"], bool)
    if joints_camera.shape != (2, frame_count, 21, 3):
        raise RuntimeError("HaWoR frame geometry mismatch")

    capture = cv2.VideoCapture(str(prepared_video))
    if not capture.isOpened():
        raise RuntimeError("prepared physical-left video failed to open")
    object_rows: list[dict[str, Any]] = []
    contact_rows: list[dict[str, Any]] = []
    clean_rows: list[dict[str, Any]] = []
    packed_invalid: list[np.ndarray] = []
    panels: list[np.ndarray] = []
    sample_slots = _representative_slots(frame_count)
    tactile_present = 0
    object_valid_frames = 0
    supported_contact_frames = 0
    try:
        for frame in range(frame_count):
            ok, rgb = capture.read()
            if not ok or rgb.shape[:2] != (960, 1280):
                raise RuntimeError(f"prepared video ended or drifted at frame {frame}")
            roles = _role_masks(mask_root, manifest, frame, (960, 1280))
            clean = compose_clean_visual(rgb, roles)
            packed_invalid.append(np.packbits(
                np.asarray(clean["invalid_mask"], bool).reshape(-1),
            ))
            clean_rows.append({"frame": frame, **clean["evidence"]})
            if frame in sample_slots:
                panel = np.concatenate((rgb, np.asarray(clean["clean_rgb"], np.uint8)), axis=1)
                cv2.putText(panel, f"{session_id} f{frame:05d} RAW | CLEAN(valid-only)",
                            (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (255, 255, 255), 2, cv2.LINE_AA)
                panels.append(panel)

            depth_path = depth_root / "frames" / f"{frame:06d}.npz"
            with np.load(depth_path, allow_pickle=False) as depth_archive:
                depth = np.asarray(depth_archive["depth_m"], np.float64)
                depth_valid = np.asarray(depth_archive["valid"], bool)
            frame_objects: list[dict[str, Any]] = []
            for instance_id, mask in _task_masks(mask_root, manifest, frame, (960, 1280)):
                estimate = estimate_visible_object(
                    task=task, mask_primary=mask, depth_m=depth,
                    depth_valid=depth_valid, depth_intrinsics=depth_intrinsics,
                    h_depth_to_primary=homography,
                )
                frame_objects.append({"instance_id": instance_id, **estimate})
            passing_objects = [row for row in frame_objects
                               if row["status"] == "PASS_VISIBLE_SURFACE"]
            if passing_objects:
                object_valid_frames += 1
            object_rows.append({
                "frame": frame, "instances": frame_objects,
                "hidden_shape_inferred": False,
            })

            training = load(
                source_session / "preprocess/all_data" / f"{frame:05d}" / "training_data.json"
            )
            tactile = training.get("entities", {}).get("tactile")
            if tactile is not None:
                tactile_present += 1
            if tactile is None or not passing_objects:
                contact_rows.append({
                    "frame": frame, "status": (
                        "BLOCKED_MISSING_PROCESSED_TACTILE" if tactile is None
                        else "INVALID_NO_DIRECT_VISIBLE_OBJECT_SURFACE"
                    ), "supported_count": 0,
                })
                continue
            surface = max(passing_objects, key=lambda row: row["visible_point_count"])
            tips_primary = joints_camera[:, frame, TIP_INDICES]
            tips_depth = tips_primary @ primary_to_depth_rotation.T
            try:
                hypotheses = tactile_visible_surface_hypotheses(
                    tactile=tactile, fingertip_xyz_m=tips_depth,
                    fingertip_observed=np.repeat(observed[:, frame, None], 5, axis=1),
                    visible_surface_centroid_m=np.asarray(surface["centroid_m"]),
                    visible_surface_normal=np.asarray(surface["normal_depth_camera"]),
                )
                if hypotheses["supported_count"]:
                    supported_contact_frames += 1
                contact_rows.append({"frame": frame, "status": "PASS", **hypotheses})
            except ContactHypothesisError as exc:
                contact_rows.append({
                    "frame": frame, "status": "REJECTED_QUALITY",
                    "reason": str(exc), "supported_count": 0,
                })
    finally:
        capture.release()

    object_path = output / "OBJECT6D_VISIBLE_SURFACE.jsonl"
    clean_path = output / "CLEAN_EVIDENCE.jsonl"
    contact_path = output / "CONTACT_HYPOTHESES.jsonl"
    write_jsonl(object_path, object_rows)
    write_jsonl(clean_path, clean_rows)
    write_jsonl(contact_path, contact_rows)
    atomic_npz(
        output / "CLEAN_INVALID_MASKS.npz",
        packed_invalid=np.stack(packed_invalid),
        frame_shape=np.asarray((960, 1280), np.int32),
        bitorder=np.asarray("big"),
        hidden_pixels_synthesized=np.asarray(0, np.int32),
    )
    if not cv2.imwrite(str(output / "CLEAN_REVIEW6.png"), _review_sheet(panels)):
        raise RuntimeError("failed to write Clean review sheet")

    minimum_object_frames = max(5, int(0.05 * frame_count))
    object_status = "PASS" if object_valid_frames >= minimum_object_frames else "REJECTED_QUALITY"
    contact_status = (
        "BLOCKED_EXTERNAL" if tactile_present == 0
        else ("PASS" if object_status == "PASS" else "BLOCKED_UPSTREAM")
    )
    atomic_json(output / "OBJECT6D_RESULT.json", {
        "schema_version": "0915-visible-object6d-session-v1",
        "status": object_status, "task": task, "session_id": session_id,
        "frame_count": frame_count, "direct_visible_frames": object_valid_frames,
        "minimum_direct_visible_frames": minimum_object_frames,
        "trajectory": ref(
            object_path, published_path=published_output / object_path.name,
        ), "hidden_shape_inferred": False,
        "claim_limit": "Direct visible-surface optical-Z only; hidden geometry remains invalid.",
    })
    atomic_json(output / "CLEAN_RESULT.json", {
        "schema_version": "0915-clean-visual-session-v1", "status": "PASS",
        "task": task, "session_id": session_id, "frame_count": frame_count,
        "invalid_masks": ref(
            output / "CLEAN_INVALID_MASKS.npz",
            published_path=published_output / "CLEAN_INVALID_MASKS.npz",
        ),
        "evidence": ref(
            clean_path, published_path=published_output / clean_path.name,
        ),
        "review": ref(
            output / "CLEAN_REVIEW6.png",
            published_path=published_output / "CLEAN_REVIEW6.png",
        ),
        "hidden_pixels_synthesized": 0,
        "geometry_consumers_forbidden": ["Depth", "Object6D", "Contact"],
        "claim_limit": "Visual-only invalidation; task-object pixels protected and no hidden RGB invented.",
    })
    atomic_json(output / "CONTACT_RESULT.json", {
        "schema_version": "0915-contact-visible-tactile-session-v1",
        "status": contact_status, "task": task, "session_id": session_id,
        "frame_count": frame_count, "processed_tactile_frames": tactile_present,
        "tactile_supported_frames": supported_contact_frames,
        "hypotheses": ref(
            contact_path, published_path=published_output / contact_path.name,
        ), "authority": "TACTILE_SUPPORTED_HYPOTHESIS",
        "force_claim": False, "contact_ground_truth": False,
        "claim_limit": "Processed tactile activity plus direct visible geometry; not force or contact truth.",
    })

    return {"Object6D": object_status, "Clean": "PASS", "Contact": contact_status}


def process_robot_visual(*, session_id: str, hawor_npz: Path,
                         output: Path, published_output: Path,
                         assets: Any) -> str:
    with np.load(hawor_npz, allow_pickle=False) as archive:
        joints_world = np.asarray(archive["joints_3d_world"], np.float64)
        observed = np.asarray(archive["observed"], bool)
        fps = float(np.asarray(archive["fps"]).item())
    robot = solve_robot_visual(
        project_root=ROOT, joints_world=joints_world, observed=observed,
        fps=fps, assets=assets,
    )
    state_path = output / "ROBOT_VISUAL_STATES.npz"
    atomic_npz(
        state_path, q_arm=robot.pop("q_arm"), q_hand=robot.pop("q_hand"),
        valid_side_frame=robot.pop("valid_side_frame"),
        virtual_tool_targets=robot.pop("virtual_tool_targets"),
        control_ground_truth=np.asarray(False),
        physical_deployment_authorized=np.asarray(False),
    )
    robot_sidecar = {
        "schema_version": "robot-visual-sidecar-v1", "session_id": session_id,
        "lane": "ROBOT_VISUAL", "status": robot["status"],
        "control_ground_truth": False, "physical_deployment_authorized": False,
        "calibration_authority": "DEVELOPMENT_ONLY",
        "calibration_evidence": robot["calibration_evidence"],
        "metrics": robot["metrics"], "claim_limit": robot["claim_limit"],
    }
    jsonschema.Draft202012Validator(load(ROBOT_SCHEMA)).validate(robot_sidecar)
    atomic_json(output / "ROBOT_VISUAL.json", robot_sidecar)
    atomic_json(output / "ROBOT_VISUAL_EVIDENCE.json", {
        "schema_version": "0915-robot-visual-evidence-v1",
        "session_id": session_id,
        "states": ref(
            state_path, published_path=published_output / state_path.name,
        ),
        "workspace_search": robot["workspace_search"],
        "diagnostics": robot["diagnostics"], "gates": robot["gates"],
        "sidecar": {
            "path": str((published_output / "ROBOT_VISUAL.json").resolve()),
            "sha256": sha256(output / "ROBOT_VISUAL.json"),
        },
    })
    return str(robot["status"])


def process_clean_only(*, task: str, session_id: str, frame_count: int,
                       prepared_video: Path, mask_root: Path,
                       output: Path, published_output: Path) -> str:
    """Publish Clean from Raw+Mask when Depth/Object6D are independently blocked."""

    manifest = load(mask_root / "ROLE_MANIFEST.json")
    capture = cv2.VideoCapture(str(prepared_video))
    if not capture.isOpened():
        raise RuntimeError("prepared physical-left video failed to open")
    packed_invalid: list[np.ndarray] = []
    clean_rows: list[dict[str, Any]] = []
    panels: list[np.ndarray] = []
    sample_slots = _representative_slots(frame_count)
    try:
        for frame in range(frame_count):
            ok, rgb = capture.read()
            if not ok or rgb.shape[:2] != (960, 1280):
                raise RuntimeError(f"prepared video ended or drifted at frame {frame}")
            clean = compose_clean_visual(
                rgb, _role_masks(mask_root, manifest, frame, (960, 1280)),
            )
            packed_invalid.append(np.packbits(
                np.asarray(clean["invalid_mask"], bool).reshape(-1),
            ))
            clean_rows.append({"frame": frame, **clean["evidence"]})
            if frame in sample_slots:
                panels.append(np.concatenate(
                    (rgb, np.asarray(clean["clean_rgb"], np.uint8)), axis=1,
                ))
    finally:
        capture.release()
    evidence = output / "CLEAN_EVIDENCE.jsonl"
    write_jsonl(evidence, clean_rows)
    atomic_npz(
        output / "CLEAN_INVALID_MASKS.npz",
        packed_invalid=np.stack(packed_invalid),
        frame_shape=np.asarray((960, 1280), np.int32),
        bitorder=np.asarray("big"),
        hidden_pixels_synthesized=np.asarray(0, np.int32),
    )
    if not cv2.imwrite(str(output / "CLEAN_REVIEW6.png"), _review_sheet(panels)):
        raise RuntimeError("failed to write Clean review sheet")
    atomic_json(output / "CLEAN_RESULT.json", {
        "schema_version": "0915-clean-visual-session-v1", "status": "PASS",
        "task": task, "session_id": session_id, "frame_count": frame_count,
        "invalid_masks": ref(
            output / "CLEAN_INVALID_MASKS.npz",
            published_path=published_output / "CLEAN_INVALID_MASKS.npz",
        ),
        "evidence": ref(
            evidence, published_path=published_output / evidence.name,
        ),
        "review": ref(
            output / "CLEAN_REVIEW6.png",
            published_path=published_output / "CLEAN_REVIEW6.png",
        ),
        "hidden_pixels_synthesized": 0,
        "geometry_consumers_forbidden": ["Depth", "Object6D", "Contact"],
        "claim_limit": "Visual-only invalidation independent of Depth/Object6D/Contact.",
    })
    return "PASS"


def write_contact_aware_sidecar(*, session_id: str, output: Path,
                                object_status: str, contact_status: str,
                                robot_status: str) -> str:
    status = (
        "BLOCKED_UPSTREAM"
        if object_status != "PASS" or contact_status != "PASS" or robot_status != "PASS"
        else "BLOCKED_EXTERNAL"
    )
    strict_sidecar = {
        "schema_version": "robot-visual-sidecar-v1", "session_id": session_id,
        "lane": "ROBOT_CONTACT_AWARE_STRICT",
        "status": status,
        "control_ground_truth": False, "physical_deployment_authorized": False,
        "calibration_authority": "DEVELOPMENT_ONLY",
        "calibration_evidence": {
            "robot_tcp": "ABSENT",
            "tool_to_kaihand_root": "PRESENT_CANDIDATE_GEOMETRY",
            "camera_world_to_base": "ABSENT",
        },
        "metrics": {key: None for key in (
            "arm_ik", "kaihand_retarget", "collision", "joint_limit", "velocity", "acceleration"
        )},
        "claim_limit": (
            "Strict Contact-aware Robot is not promoted without measured TCP, installation "
            "transform and camera/world-to-base calibration."
        ),
    }
    jsonschema.Draft202012Validator(load(ROBOT_SCHEMA)).validate(strict_sidecar)
    atomic_json(output / "ROBOT_CONTACT_AWARE.json", strict_sidecar)
    return status


def _write_terminal_sidecars(output: Path, session_id: str,
                             statuses: dict[str, str], reason: str) -> None:
    for stage, filename in (
        ("Object6D", "OBJECT6D_RESULT.json"), ("Clean", "CLEAN_RESULT.json"),
        ("Contact", "CONTACT_RESULT.json"), ("RobotVisual", "ROBOT_VISUAL.json"),
        ("RobotContactAware", "ROBOT_CONTACT_AWARE.json"),
    ):
        if stage not in statuses:
            continue
        atomic_json(output / filename, {
            "schema_version": "0915-post-terminal-sidecar-v1",
            "session_id": session_id, "stage": stage,
            "status": statuses[stage], "reason": reason,
        })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--heartbeat-sessions", type=int, default=1)
    args = parser.parse_args()
    packet = validate_route()
    output = args.output_root.resolve()
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"fresh immutable attempt required: {output}")
    output.mkdir(parents=True)
    heartbeat()
    started = time.time()

    prepared_path = INPUT_ATTEMPT / "prepared_physical_left/BATCH_RESULT.json"
    prepared = load(prepared_path)
    hawor_batch = load(HAWOR_ATTEMPT / "BATCH_RESULT.json")
    mask_batch = load(MASK_ATTEMPT / "BATCH_RESULT.json")
    depth_batch = load(DEPTH_ATTEMPT / "BATCH_RESULT.json")
    batches = (prepared, hawor_batch, mask_batch, depth_batch)
    if any(row.get("session_count") != 220 for row in batches):
        raise RuntimeError("post join requires four fixed 220-session manifests")
    prepared_by = {(row["task"], row["session_id"]): row for row in prepared["results"]}
    hawor_by = {(row["task"], row["session_id"]): row for row in hawor_batch["results"]}
    mask_by = {(row["task"], row["session_id"]): row for row in mask_batch["results"]}
    depth_by = {(row["task"], row["session_id"]): row for row in depth_batch["results"]}
    if not (set(prepared_by) == set(hawor_by) == set(mask_by) == set(depth_by)):
        raise RuntimeError("post batch session identity sets differ")
    ledger = initialize_ledger([
        {"task": task, "session_id": session_id}
        for task, session_id in prepared_by
    ])
    robot_assets = load_pinned_robot_assets(ROOT)
    results: list[dict[str, Any]] = []
    for index, identity in enumerate(prepared_by, 1):
        task, session_id = identity
        published_session_output = output / "sessions" / task / session_id
        if published_session_output.exists() or published_session_output.is_symlink():
            raise RuntimeError(
                f"session target already exists in fresh attempt: {published_session_output}"
            )
        published_session_output.parent.mkdir(parents=True, exist_ok=True)
        session_output = published_session_output.with_name(
            f".{session_id}.staging-{uuid.uuid4().hex}"
        )
        session_output.mkdir()
        upstream = {
            "Raw": terminal(prepared_by[identity]["status"]),
            "HaWoR": terminal(hawor_by[identity]["status"]),
            "Mask": terminal(mask_by[identity]["status"]),
            "Depth": terminal(depth_by[identity]["status"]),
        }
        for stage in ("Raw", "HaWoR", "Mask", "Depth"):
            batch_path = {
                "Raw": prepared_path, "HaWoR": HAWOR_ATTEMPT / "BATCH_RESULT.json",
                "Mask": MASK_ATTEMPT / "BATCH_RESULT.json",
                "Depth": DEPTH_ATTEMPT / "BATCH_RESULT.json",
            }[stage]
            update_stage(
                ledger, task=task, session_id=session_id, stage=stage,
                status=upstream[stage], reason=(None if upstream[stage] == "PASS" else f"{stage.upper()}_NOT_PASS"),
                input_signature=signature(session_id, stage, sha256(batch_path)),
                weight_identity={"HaWoR": HAWOR_WEIGHT, "Mask": SAM_WEIGHT,
                                 "Depth": DEPTH_WEIGHT}.get(stage),
                calibration_identity=(CALIBRATION_ID if stage in {"Raw", "Depth"} else None),
                artifact=ref(batch_path),
            )
        post_status: dict[str, str] = {}
        if upstream["Mask"] == "PASS" and upstream["Depth"] == "PASS" and upstream["HaWoR"] == "PASS":
            try:
                post_status.update(process_visual_geometry(
                    task=task, session_id=session_id,
                    frame_count=int(prepared_by[identity]["frame_count"]),
                    source_session=Path(prepared_by[identity]["source"]["session"]),
                    prepared_video=(prepared_path.parent / "sessions" / task / session_id
                                    / prepared_by[identity]["output"]["video_relative"]),
                    mask_root=MASK_ATTEMPT / "sessions" / task / session_id,
                    depth_root=DEPTH_ATTEMPT / "sessions" / task / session_id,
                    hawor_npz=HAWOR_ATTEMPT / "sessions" / task / session_id / "HAWOR_RAW_MANO21.npz",
                    output=session_output,
                    published_output=published_session_output,
                ))
            except Exception as exc:  # noqa: BLE001
                post_status.update({stage: "FAILED_RUNTIME" for stage in (
                    "Object6D", "Clean", "Contact"
                )})
                atomic_json(session_output / "FAILED_RUNTIME.json", {
                    "status": "FAILED_RUNTIME", "error": repr(exc),
                })
                _write_terminal_sidecars(session_output, session_id, post_status, repr(exc))
        else:
            geometry_reason = next(
                name for name in ("HaWoR", "Mask", "Depth")
                if upstream[name] != "PASS"
            ) + "_NOT_PASS"
            post_status.update({
                "Object6D": "BLOCKED_UPSTREAM", "Contact": "BLOCKED_UPSTREAM",
            })
            _write_terminal_sidecars(
                session_output, session_id,
                {key: post_status[key] for key in ("Object6D", "Contact")},
                geometry_reason,
            )
            if upstream["Mask"] == "PASS":
                try:
                    post_status["Clean"] = process_clean_only(
                        task=task, session_id=session_id,
                        frame_count=int(prepared_by[identity]["frame_count"]),
                        prepared_video=(prepared_path.parent / "sessions" / task / session_id
                                        / prepared_by[identity]["output"]["video_relative"]),
                        mask_root=MASK_ATTEMPT / "sessions" / task / session_id,
                        output=session_output,
                        published_output=published_session_output,
                    )
                except Exception as exc:  # noqa: BLE001
                    post_status["Clean"] = "FAILED_RUNTIME"
                    _write_terminal_sidecars(
                        session_output, session_id,
                        {"Clean": "FAILED_RUNTIME"}, repr(exc),
                    )
            else:
                post_status["Clean"] = "BLOCKED_UPSTREAM"
                _write_terminal_sidecars(
                    session_output, session_id,
                    {"Clean": "BLOCKED_UPSTREAM"}, "MASK_NOT_PASS",
                )

        if upstream["HaWoR"] == "PASS":
            try:
                post_status["RobotVisual"] = process_robot_visual(
                    session_id=session_id,
                    hawor_npz=HAWOR_ATTEMPT / "sessions" / task / session_id / "HAWOR_RAW_MANO21.npz",
                    output=session_output,
                    published_output=published_session_output,
                    assets=robot_assets,
                )
            except Exception as exc:  # noqa: BLE001
                post_status["RobotVisual"] = "FAILED_RUNTIME"
                _write_terminal_sidecars(
                    session_output, session_id,
                    {"RobotVisual": "FAILED_RUNTIME"}, repr(exc),
                )
        else:
            post_status["RobotVisual"] = "BLOCKED_UPSTREAM"
            _write_terminal_sidecars(
                session_output, session_id,
                {"RobotVisual": "BLOCKED_UPSTREAM"}, "HAWOR_NOT_PASS",
            )
        post_status["RobotContactAware"] = write_contact_aware_sidecar(
            session_id=session_id, output=session_output,
            object_status=post_status["Object6D"],
            contact_status=post_status["Contact"],
            robot_status=post_status["RobotVisual"],
        )
        atomic_json(session_output / "SESSION_RESULT.json", {
            "schema_version": "0915-post-session-result-v1",
            "task": task,
            "session_id": session_id,
            "status": (
                "FAILED_RUNTIME"
                if "FAILED_RUNTIME" in post_status.values() else "TERMINAL"
            ),
            "stages": post_status,
            "atomic_publish": True,
        })
        os.replace(session_output, published_session_output)
        session_output = published_session_output
        stage_files = {
            "Object6D": "OBJECT6D_RESULT.json", "Clean": "CLEAN_RESULT.json",
            "Contact": "CONTACT_RESULT.json", "RobotVisual": "ROBOT_VISUAL.json",
            "RobotContactAware": "ROBOT_CONTACT_AWARE.json",
        }
        for stage, status in post_status.items():
            update_stage(
                ledger, task=task, session_id=session_id, stage=stage,
                status=status, reason=(None if status == "PASS" else load(session_output / stage_files[stage]).get("reason", status)),
                input_signature=signature(
                    session_id, stage, sha256(HAWOR_ATTEMPT / "BATCH_RESULT.json"),
                    sha256(MASK_ATTEMPT / "BATCH_RESULT.json"),
                    sha256(DEPTH_ATTEMPT / "BATCH_RESULT.json"),
                ),
                weight_identity=None,
                calibration_identity=(
                    CALIBRATION_ID if stage in {"Object6D", "Contact"}
                    else "DEVELOPMENT_ONLY_MEASURED_TCP_MOUNT_WORLD_BASE_ABSENT"
                    if stage.startswith("Robot") else None
                ),
                artifact=ref(session_output / stage_files[stage]),
            )
        result = {"task": task, "session_id": session_id, "stages": {**upstream, **post_status}}
        results.append(result)
        atomic_json(output / "FULL_FUNNEL_LEDGER.json", ledger)
        atomic_json(output / "STATE.json", {
            "schema_version": "0915-post-geometry-robot-state-v1",
            "state": "RUNNING", "completed": len(results), "session_count": 220,
            "results": results, "updated_unix": time.time(),
        })
        if index % max(1, args.heartbeat_sessions) == 0:
            heartbeat()
        print(json.dumps({"completed": index, "total": 220,
                          "session": session_id, "stages": post_status},
                         sort_keys=True), flush=True)

    atomic_json(output / "FUNNEL_SUMMARY.json", {
        "schema_version": "0915-full-funnel-summary-v1",
        "fixed_denominator": 220, "summary": ledger["summary"],
        "generated_only_from_ledger_rows": True,
    })
    failed_runtime = sum(
        row["stages"][stage]["status"] == "FAILED_RUNTIME"
        for row in ledger["sessions"] for stage in ledger["stage_order"]
    )
    result = {
        "schema_version": "0915-post-geometry-robot-result-v1",
        "task_id": TASK_ID,
        "status": "PASSED" if failed_runtime == 0 else "FAILED_RUNTIME_FINAL",
        "fixed_denominator": 220, "failed_runtime_stage_cells": failed_runtime,
        "ledger": ref(output / "FULL_FUNNEL_LEDGER.json"),
        "summary": ref(output / "FUNNEL_SUMMARY.json"),
        "weights": "ABSENT", "mask_model_policy": "SAM3.1_ONLY_USER_LOCKED",
        "wall_seconds": time.time() - started,
        "claim_limit": packet["claim_limit"],
    }
    atomic_json(output / "RESULT.json", result)
    atomic_json(output / "RUN_RECEIPT.json", {
        "schema_version": "0915-post-geometry-robot-run-receipt-v1",
        "task_id": TASK_ID, "status": result["status"],
        "result": ref(output / "RESULT.json"),
    })
    return 0 if result["status"] == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
