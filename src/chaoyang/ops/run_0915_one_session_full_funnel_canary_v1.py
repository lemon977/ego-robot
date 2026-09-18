#!/usr/bin/env python3
"""Run the bounded play_cards_0915_001 full-funnel development canary.

GPU stages are deliberately separate so each central-lease invocation binds
exactly one model weight.  Source data are read-only and all publications are
isolated below the caller-provided attempt root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any
import uuid

import cv2
import numpy as np



ROOT = Path(__file__).resolve().parents[3]
TASK = "playing_cards"
SESSION_ID = "play_cards_0915_001"
PREPARED_MANIFEST = (
    ROOT / "_run/current/0915_input_prepare_cad_v2/attempts/attempt_0001"
    / "prepared_physical_left/BATCH_RESULT.json"
)
HAWOR_ROOT = (
    ROOT / "_run/current/0915_hawor_full_v1/attempts/attempt_0001/hawor"
)
VISUAL_ROOT = ROOT / "docs/current/visuals/0915_ONE_SESSION_CANARY_V1"
CHAINS = (
    (0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12),
    (0, 13, 14, 15, 16), (0, 17, 18, 19, 20),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


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


def prepared_row() -> tuple[dict[str, Any], Path]:
    manifest = load(PREPARED_MANIFEST)
    rows = [row for row in manifest["results"]
            if row["task"] == TASK and row["session_id"] == SESSION_ID]
    if len(rows) != 1:
        raise RuntimeError("frozen canary identity is not singular")
    return rows[0], PREPARED_MANIFEST.parent


def paths(output: Path) -> dict[str, Path]:
    row, prepared_root = prepared_row()
    return {
        "video": prepared_root / "sessions" / TASK / SESSION_ID
                 / row["output"]["video_relative"],
        "hawor": HAWOR_ROOT / "sessions" / TASK / SESSION_ID
                 / "HAWOR_RAW_MANO21.npz",
        "sam_session": output / "sam31/sessions" / TASK / SESSION_ID,
        "depth_session": output / "depth/sessions" / TASK / SESSION_ID,
        "post_session": output / "post/sessions" / TASK / SESSION_ID,
    }


def preflight(output: Path) -> dict[str, Any]:
    row, _prepared_root = prepared_row()
    p = paths(output)
    video = p["video"].resolve(strict=True)
    hawor = p["hawor"].resolve(strict=True)
    capture = cv2.VideoCapture(str(video))
    decoded = 0
    width = height = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            height, width = frame.shape[:2]
            decoded += 1
    finally:
        capture.release()
    with np.load(hawor, allow_pickle=False) as archive:
        joints = np.asarray(archive["joints_2d"], np.float64)
        observed = np.asarray(archive["observed"], bool)
        provenance = np.asarray(archive["provenance"]).astype(str)
        original = np.asarray(archive["original_frame_indices"])
    frame_count = int(row["frame_count"])
    if decoded != frame_count or (width, height) != (1280, 960):
        raise RuntimeError("prepared physical-left RGB identity drift")
    if joints.shape != (2, frame_count, 21, 2) or observed.shape != (2, frame_count):
        raise RuntimeError("HaWoR array geometry drift")
    if not np.array_equal(original, np.arange(frame_count)):
        raise RuntimeError("HaWoR original-frame mapping drift")
    if not np.array_equal(observed, provenance == "OBSERVED"):
        raise RuntimeError("HaWoR observed/provenance mismatch")
    observed_counts = {"left": int(observed[0].sum()),
                       "right": int(observed[1].sum()),
                       "bilateral": int(np.all(observed, axis=0).sum())}
    if sum(observed_counts[side] for side in ("left", "right")) == 0:
        raise RuntimeError("HaWoR has no direct observations")
    result = {
        "schema_version": "0915-one-session-canary-preflight-v1",
        "status": "PASS_DEVELOPMENT_CANARY_INPUT",
        "task": TASK, "session_id": SESSION_ID, "frame_count": frame_count,
        "rgb_primary": "PHYSICAL_LEFT_SOURCE_INDEX_1_EQUIDIS62_TO_PINHOLE_1280X960_FOV90",
        "observed_frames": observed_counts,
        "hawor_producer_status": load(p["hawor"].parent / "RESULT.json")["status"],
        "strict_expected_active_gate": {
            "status": "NOT_EVALUATED",
            "reason": "INDEPENDENT_EXPECTED_ACTIVE_VISIBILITY_DENOMINATOR_ABSENT",
        },
        "admission": "OBSERVED_ONLY_DEVELOPMENT_CANARY",
        "pico26": "PRESENT_PRESERVED_NOT_CONSUMED",
        "tracker": "ABSENT_NO_ROLE_CREATED",
        "inputs": {"video": ref(video), "hawor": ref(hawor),
                   "prepared_manifest": ref(PREPARED_MANIFEST)},
        "claim_limit": (
            "One-session development evidence only. Missing hands stay missing; "
            "no independent expected-active recall, identity, physical 3D, or deployment claim."
        ),
    }
    preflight_path = output / "PREFLIGHT.json"
    if preflight_path.is_file():
        if load(preflight_path) != result:
            raise RuntimeError("immutable preflight evidence drift")
    else:
        atomic_json(preflight_path, result)
    return result


def run_sam31(output: Path) -> dict[str, Any]:
    from chaoyang.ops import run_0915_sam31_persistent_masks_v2 as sam_worker

    pre = preflight(output)
    if sha256(sam_worker.CHECKPOINT) != sam_worker.CHECKPOINT_SHA256:
        raise RuntimeError("SAM3.1 checkpoint SHA drift")
    if not sam_worker.torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if str(sam_worker.CODE_ROOT) not in sys.path:
        sys.path.insert(0, str(sam_worker.CODE_ROOT))
    row, prepared_root = prepared_row()
    module = sam_worker.legacy.load_adapter()
    adapter, build_evidence = module.build_pinned_adapter(
        official_code_root=sam_worker.CODE_ROOT,
        checkpoint_path=sam_worker.CHECKPOINT,
    )
    started = time.time()
    try:
        result = sam_worker.process_session(
            model=adapter.model, task=TASK, session_id=SESSION_ID,
            prepared_root=prepared_root, hawor_root=HAWOR_ROOT,
            output_root=output / "sam31", frame_count=int(row["frame_count"]),
        )
    finally:
        adapter.predictor.shutdown()
    stage = {
        "schema_version": "0915-one-session-sam31-stage-v1",
        "status": result["status"], "task": TASK, "session_id": SESSION_ID,
        "model_identity": "SAM3.1_ONLY_USER_LOCKED",
        "weight": ref(sam_worker.CHECKPOINT),
        "preflight": ref(output / "PREFLIGHT.json"),
        "session_result": ref(paths(output)["sam_session"] / "RESULT.json"),
        "build_evidence": build_evidence,
        "wall_seconds": time.time() - started,
        "claim_limit": "One-session SAM3.1 development masks; no alternate model or ground truth.",
    }
    atomic_json(output / "SAM31_STAGE_RESULT.json", stage)
    return stage


def run_depth(output: Path) -> dict[str, Any]:
    from chaoyang.ops import run_0915_foundationstereo_persistent_worker_v2 as depth_worker
    from chaoyang.ops import run_0915_leftmono_foundation_depth_v1 as depth_session
    from chaoyang.ops import run_exact78_foundationstereo_corrected_depth_worker as fs_worker

    preflight(output)
    if sha256(depth_worker.CHECKPOINT) != depth_worker.CHECKPOINT_SHA256:
        raise RuntimeError("FoundationStereo checkpoint SHA drift")
    row, prepared_root = prepared_row()
    pico = depth_session.load_module(depth_session.PICO_PATH, "pico_0915_canary_v1")
    model = fs_worker.Model()
    started = time.time()
    result = depth_worker.process_session(
        row, prepared_root=prepared_root, output_root=output / "depth",
        model=model, pico=pico,
    )
    stage = {
        "schema_version": "0915-one-session-depth-stage-v1",
        "status": result["status"], "task": TASK, "session_id": SESSION_ID,
        "model_identity": "FOUNDATIONSTEREO_PINNED_BASELINE",
        "weight": ref(depth_worker.CHECKPOINT),
        "preflight": ref(output / "PREFLIGHT.json"),
        "session_result": ref(paths(output)["depth_session"] / "RESULT.json"),
        "model_load_count": model.model_load_count,
        "model_inference_count": model.inference_count,
        "wall_seconds": time.time() - started,
        "claim_limit": "Internal optical-Z consistency only; no external millimetre accuracy.",
    }
    atomic_json(output / "DEPTH_STAGE_RESULT.json", stage)
    return stage


def run_post(output: Path) -> dict[str, Any]:
    from chaoyang.ops import run_0915_post_geometry_robot_v1 as post_worker
    from chaoyang.pipeline.robot_renderer_eevee_fullchain import load_pinned_robot_assets

    pre = load(output / "PREFLIGHT.json")
    sam = load(output / "SAM31_STAGE_RESULT.json")
    depth = load(output / "DEPTH_STAGE_RESULT.json")
    p = paths(output)
    target = p["post_session"]
    if target.exists() or target.is_symlink():
        raise RuntimeError(f"fresh post target required: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(f".{SESSION_ID}.staging-{uuid.uuid4().hex}")
    staging.mkdir()
    started = time.time()
    try:
        row, _prepared_root = prepared_row()
        if sam.get("status") == "PASS" and depth.get("status") == "PASS":
            statuses = post_worker.process_visual_geometry(
                task=TASK, session_id=SESSION_ID,
                frame_count=int(pre["frame_count"]),
                source_session=Path(row["source"]["session"]),
                prepared_video=p["video"], mask_root=p["sam_session"],
                depth_root=p["depth_session"], hawor_npz=p["hawor"],
                output=staging, published_output=target,
            )
        else:
            statuses = {
                "Object6D": "BLOCKED_UPSTREAM",
                "Clean": "BLOCKED_UPSTREAM",
                "Contact": "BLOCKED_UPSTREAM",
            }
            reason = (
                "MASK_NOT_PASS" if sam.get("status") != "PASS"
                else "DEPTH_NOT_PASS"
            )
            post_worker._write_terminal_sidecars(
                staging, SESSION_ID, statuses, reason,
            )
        assets = load_pinned_robot_assets(ROOT)
        statuses["RobotVisual"] = post_worker.process_robot_visual(
            session_id=SESSION_ID, hawor_npz=p["hawor"], output=staging,
            published_output=target, assets=assets,
        )
        statuses["RobotContactAware"] = post_worker.write_contact_aware_sidecar(
            session_id=SESSION_ID, output=staging,
            object_status=statuses["Object6D"],
            contact_status=statuses["Contact"],
            robot_status=statuses["RobotVisual"],
        )
        atomic_json(staging / "SESSION_RESULT.json", {
            "schema_version": "0915-one-session-post-result-v1",
            "status": "TERMINAL", "task": TASK, "session_id": SESSION_ID,
            "upstream": {"Raw": pre["status"], "HaWoR": pre["admission"],
                         "Mask": sam["status"], "Depth": depth["status"]},
            "stages": statuses, "weights": "ABSENT",
            "control_ground_truth": False,
            "physical_deployment_authorized": False,
            "wall_seconds": time.time() - started,
        })
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    stage = load(target / "SESSION_RESULT.json")
    atomic_json(output / "POST_STAGE_RESULT.json", {
        **stage, "session_result": ref(target / "SESSION_RESULT.json"),
    })
    return stage


def _draw_hawor(frame: np.ndarray, joints: np.ndarray,
                observed: np.ndarray, frame_index: int) -> np.ndarray:
    panel = frame.copy()
    colors = ((255, 220, 20), (30, 70, 255))
    for side in range(2):
        if not observed[side, frame_index]:
            continue
        uv = joints[side, frame_index]
        for chain in CHAINS:
            for a, b in zip(chain[:-1], chain[1:], strict=True):
                pa, pb = uv[a], uv[b]
                if np.isfinite(pa).all() and np.isfinite(pb).all():
                    cv2.line(panel, tuple(np.rint(pa).astype(int)),
                             tuple(np.rint(pb).astype(int)), colors[side], 3,
                             cv2.LINE_AA)
        for point in uv:
            if np.isfinite(point).all():
                cv2.circle(panel, tuple(np.rint(point).astype(int)), 3,
                           colors[side], -1, cv2.LINE_AA)
    return panel


def _depth_panel(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        depth = np.asarray(archive["depth_m"], np.float32)
        valid = np.asarray(archive["valid"], bool)
    color = np.zeros((*depth.shape, 3), np.uint8)
    if valid.any():
        lo, hi = np.percentile(depth[valid], (5, 95))
        scaled = np.zeros(depth.shape, np.uint8)
        scaled[valid] = np.clip((depth[valid] - lo) / max(float(hi - lo), 1e-6) * 255,
                                0, 255).astype(np.uint8)
        color = cv2.applyColorMap(255 - scaled, cv2.COLORMAP_TURBO)
        color[~valid] = 0
    return cv2.resize(color, (1280, 960), interpolation=cv2.INTER_NEAREST)


def render_sam31_baseline(output: Path, visual_root: Path) -> dict[str, Any]:
    """Render the terminal SAM3.1 baseline while Depth runs independently."""

    from chaoyang.ops import run_0915_post_geometry_robot_v1 as post_worker

    p = paths(output)
    sam = load(output / "SAM31_STAGE_RESULT.json")
    manifest = load(p["sam_session"] / "ROLE_MANIFEST.json")
    visual_root.mkdir(parents=True, exist_ok=True)
    video_path = visual_root / f"{SESSION_ID}_SAM31_BASELINE_REVIEW.mp4"
    sheet_path = visual_root / f"{SESSION_ID}_SAM31_BASELINE_CONTACT_SHEET.jpg"
    result_path = visual_root / "SAM31_BASELINE_RESULT.json"
    for candidate in (video_path, sheet_path, result_path):
        if candidate.exists() or candidate.is_symlink():
            raise RuntimeError(f"immutable visual target exists: {candidate}")
    with np.load(p["hawor"], allow_pickle=False) as archive:
        joints = np.asarray(archive["joints_2d"], np.float64)
        observed = np.asarray(archive["observed"], bool)
    capture = cv2.VideoCapture(str(p["video"]))
    frame_count = int(load(output / "PREFLIGHT.json")["frame_count"])
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             30.0, (1280, 480))
    sample_indices = set(np.linspace(0, frame_count - 1, 12, dtype=int).tolist())
    samples: list[np.ndarray] = []
    if not capture.isOpened() or not writer.isOpened():
        raise RuntimeError("SAM3.1 baseline review video I/O failed")
    palette = {
        "left_human_skin_forearm": (255, 210, 20),
        "right_human_skin_forearm": (20, 70, 255),
        "left_finger_sleeve_attachment": (255, 40, 220),
        "right_finger_sleeve_attachment": (180, 20, 255),
        "left_cable": (20, 180, 255), "right_cable": (20, 180, 255),
        "task_object": (40, 255, 80),
    }
    try:
        for index in range(frame_count):
            ok, rgb = capture.read()
            if not ok:
                raise RuntimeError(f"SAM3.1 baseline decode ended at {index}")
            hawor = _draw_hawor(rgb, joints, observed, index)
            roles = post_worker._role_masks(p["sam_session"], manifest, index,
                                            (960, 1280))
            overlay = rgb.astype(np.float32)
            for role, mask in roles.items():
                color = np.asarray(palette.get(role, (220, 220, 220)), np.float32)
                overlay[mask] = 0.35 * overlay[mask] + 0.65 * color
            left = cv2.resize(hawor, (640, 480), interpolation=cv2.INTER_AREA)
            right = cv2.resize(overlay.astype(np.uint8), (640, 480),
                               interpolation=cv2.INTER_AREA)
            canvas = np.concatenate((left, right), axis=1)
            cv2.rectangle(canvas, (0, 0), (1279, 44), (0, 0, 0), -1)
            cv2.putText(
                canvas,
                f"{SESSION_ID} f{index:03d} | RAW+HaWoR observed | SAM3.1 "
                f"(green=cards, cyan/red=hands) | status={sam['status']}",
                (12, 29), cv2.FONT_HERSHEY_SIMPLEX, .55,
                (255, 255, 255), 2, cv2.LINE_AA,
            )
            writer.write(canvas)
            if index in sample_indices:
                samples.append(cv2.resize(canvas, (480, 180),
                                          interpolation=cv2.INTER_AREA))
    finally:
        capture.release()
        writer.release()
    sheet = np.zeros((3 * 180, 4 * 480, 3), np.uint8)
    for i, sample in enumerate(samples[:12]):
        y, x = divmod(i, 4)
        sheet[y * 180:(y + 1) * 180, x * 480:(x + 1) * 480] = sample
    if not cv2.imwrite(str(sheet_path), sheet):
        raise RuntimeError("failed to write SAM3.1 baseline contact sheet")
    result = {
        "schema_version": "0915-one-session-sam31-visual-evidence-v1",
        "status": "PASS_VISUAL_DECODE_ONLY", "task": TASK,
        "session_id": SESSION_ID, "frame_count": frame_count,
        "sam31_terminal_status": sam["status"],
        "video": ref(video_path), "contact_sheet": ref(sheet_path),
        "claim_limit": "SAM3.1 terminal development review; not mask ground truth.",
    }
    atomic_json(result_path, result)
    return result


def render_visuals(output: Path, visual_root: Path) -> dict[str, Any]:
    from chaoyang.ops import run_0915_post_geometry_robot_v1 as post_worker
    from chaoyang.pipeline.clean_visual_evidence_v1 import compose_clean_visual

    post_path = output / "POST_STAGE_RESULT.json"
    post = load(post_path) if post_path.is_file() else {
        "stages": {
            "Object6D": "NOT_RUN", "Clean": "VISUAL_REVIEW_ONLY",
            "Contact": "NOT_RUN", "RobotVisual": "NOT_RUN",
            "RobotContactAware": "NOT_RUN",
        }
    }
    p = paths(output)
    visual_root.mkdir(parents=True, exist_ok=True)
    video_path = visual_root / f"{SESSION_ID}_FULL_FUNNEL_REVIEW.mp4"
    sheet_path = visual_root / f"{SESSION_ID}_FULL_FUNNEL_CONTACT_SHEET.jpg"
    result_path = visual_root / "RESULT.json"
    for candidate in (video_path, sheet_path, result_path):
        if candidate.exists() or candidate.is_symlink():
            raise RuntimeError(f"immutable visual target exists: {candidate}")
    with np.load(p["hawor"], allow_pickle=False) as archive:
        joints = np.asarray(archive["joints_2d"], np.float64)
        observed = np.asarray(archive["observed"], bool)
    manifest = load(p["sam_session"] / "ROLE_MANIFEST.json")
    robot_path = p["post_session"] / "ROBOT_VISUAL.json"
    robot = load(robot_path) if robot_path.is_file() else {"status": "NOT_RUN"}
    frame_count = int(load(output / "PREFLIGHT.json")["frame_count"])
    sample_indices = set(np.linspace(0, frame_count - 1, 12, dtype=int).tolist())
    samples: list[np.ndarray] = []
    capture = cv2.VideoCapture(str(p["video"]))
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             30.0, (1280, 960))
    if not capture.isOpened() or not writer.isOpened():
        raise RuntimeError("review video I/O failed")
    try:
        for index in range(frame_count):
            ok, rgb = capture.read()
            if not ok:
                raise RuntimeError(f"review decode ended at {index}")
            hawor = _draw_hawor(rgb, joints, observed, index)
            roles = post_worker._role_masks(p["sam_session"], manifest, index,
                                            (960, 1280))
            mask_overlay = rgb.astype(np.float32)
            palette = {
                "left_human_skin_forearm": (255, 210, 20),
                "right_human_skin_forearm": (20, 70, 255),
                "left_finger_sleeve_attachment": (255, 40, 220),
                "right_finger_sleeve_attachment": (180, 20, 255),
                "left_cable": (20, 180, 255), "right_cable": (20, 180, 255),
                "task_object": (40, 255, 80),
            }
            for role, mask in roles.items():
                color = np.asarray(palette.get(role, (220, 220, 220)), np.float32)
                mask_overlay[mask] = 0.35 * mask_overlay[mask] + 0.65 * color
            mask_overlay = mask_overlay.astype(np.uint8)
            depth = _depth_panel(p["depth_session"] / "frames" / f"{index:06d}.npz")
            clean = np.asarray(compose_clean_visual(rgb, roles)["clean_rgb"], np.uint8)
            panels = [hawor, mask_overlay, depth, clean]
            labels = ["RAW + HaWoR observed", "SAM3.1 roles/objects",
                      "FoundationStereo optical-Z", (
                          "Clean visible-only" if post["stages"]["Clean"] == "PASS"
                          else "Clean preview only (Mask not admitted)"
                      )]
            small: list[np.ndarray] = []
            for panel, label in zip(panels, labels, strict=True):
                panel = cv2.resize(panel, (640, 480), interpolation=cv2.INTER_AREA)
                cv2.rectangle(panel, (0, 0), (639, 40), (0, 0, 0), -1)
                cv2.putText(panel, label, (12, 27), cv2.FONT_HERSHEY_SIMPLEX,
                            .65, (255, 255, 255), 2, cv2.LINE_AA)
                small.append(panel)
            canvas = np.concatenate((np.concatenate(small[:2], axis=1),
                                     np.concatenate(small[2:], axis=1)), axis=0)
            cv2.rectangle(canvas, (0, 918), (1279, 959), (0, 0, 0), -1)
            cv2.putText(
                canvas,
                f"{SESSION_ID} f{index:03d} | robot={robot['status']} | "
                f"contact={post['stages']['Contact']} | no PICO/tracker | DEV ONLY",
                (12, 946), cv2.FONT_HERSHEY_SIMPLEX, .58,
                (255, 255, 255), 2, cv2.LINE_AA,
            )
            writer.write(canvas)
            if index in sample_indices:
                samples.append(cv2.resize(canvas, (480, 360), interpolation=cv2.INTER_AREA))
    finally:
        capture.release()
        writer.release()
    sheet = np.zeros((3 * 360, 4 * 480, 3), np.uint8)
    for i, sample in enumerate(samples[:12]):
        y, x = divmod(i, 4)
        sheet[y * 360:(y + 1) * 360, x * 480:(x + 1) * 480] = sample
    if not cv2.imwrite(str(sheet_path), sheet):
        raise RuntimeError("failed to write contact sheet")
    result = {
        "schema_version": "0915-one-session-visual-evidence-v1",
        "status": "PASS_VISUAL_DECODE_ONLY", "task": TASK,
        "session_id": SESSION_ID,
        "frame_count": frame_count, "video": ref(video_path),
        "contact_sheet": ref(sheet_path), "post_stages": post["stages"],
        "control_ground_truth": False, "physical_deployment_authorized": False,
        "claim_limit": "Four-panel development review, not accuracy ground truth.",
    }
    atomic_json(result_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True,
                        choices=("preflight", "sam31", "sam_visualize",
                                 "depth", "post", "visualize"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--visual-root", type=Path, default=VISUAL_ROOT)
    args = parser.parse_args()
    output = args.output_root.resolve()
    attempt_parent = (
        ROOT / "_run/current/0915_one_session_full_funnel_canary_v1/attempts"
    ).resolve()
    if output.parent != attempt_parent or not output.name.startswith("attempt_"):
        raise RuntimeError(
            "output-root must be a dedicated 0915 one-session canary attempt"
        )
    if output == Path("/mnt/data/egodata") or Path("/mnt/data/egodata") in output.parents:
        raise RuntimeError("writing under /mnt/data/egodata is forbidden")
    visual_root = args.visual_root.resolve()
    allowed_visual_parent = (ROOT / "docs/current/visuals").resolve()
    if (visual_root.parent != allowed_visual_parent
            or visual_root.name != "0915_ONE_SESSION_CANARY_V1"):
        raise RuntimeError("visual-root must be the dedicated shallow review directory")
    output.mkdir(parents=True, exist_ok=True)
    runners = {
        "preflight": lambda: preflight(output),
        "sam31": lambda: run_sam31(output),
        "sam_visualize": lambda: render_sam31_baseline(output, visual_root),
        "depth": lambda: run_depth(output),
        "post": lambda: run_post(output),
        "visualize": lambda: render_visuals(output, visual_root),
    }
    result = runners[args.stage]()
    print(json.dumps({"stage": args.stage, "status": result.get("status"),
                      "session_id": SESSION_ID}, ensure_ascii=False, sort_keys=True),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
