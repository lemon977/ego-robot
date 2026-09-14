#!/usr/bin/env python3
"""Generate a review-only 004 KaiHand mesh video from frozen R2.

This tool deliberately does not claim to reconstruct the deleted historical
RobotRGB.  It renders the two pinned KaiHand meshes at the frozen R2 wrist and
joint states, then presents RAW, an additive mesh overlay, and robot-only RGBA
side by side.  Human removal, Tianji q_arm, object-depth occlusion and formal
compositing are outside this diagnostic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from pipeline import robot_renderer_cycles as core


class ReviewVideoError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ReviewVideoError(f"expected regular non-symlink file: {path}")
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        while True:
            chunk = os.read(fd, 8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    finally:
        os.close(fd)
    return digest.hexdigest()


def verify_file(path: Path, expected_sha256: str) -> None:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ReviewVideoError(f"input SHA mismatch for {path}: {actual}")


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def require_new_output_root(project_root: Path, output_root: Path) -> None:
    run_root = (project_root / "_run").resolve()
    resolved = output_root.resolve(strict=False)
    if output_root.is_symlink() or resolved.parent != run_root:
        raise ReviewVideoError("output must be a new direct child of project _run")
    if not resolved.name.startswith("004_r2_kaihand_regenerated_review_"):
        raise ReviewVideoError("output run id has the wrong review namespace")
    if resolved.exists():
        raise ReviewVideoError(f"refusing to overwrite existing output: {resolved}")


def alpha_compose(raw_bgr: np.ndarray, rgba_bgra: np.ndarray) -> np.ndarray:
    if raw_bgr.ndim != 3 or raw_bgr.shape[2] != 3:
        raise ReviewVideoError("RAW frame must be BGR")
    if rgba_bgra.shape[:2] != raw_bgr.shape[:2] or rgba_bgra.shape[2] != 4:
        raise ReviewVideoError("robot RGBA shape mismatch")
    alpha = rgba_bgra[..., 3:4].astype(np.float32) / 255.0
    robot = rgba_bgra[..., :3].astype(np.float32)
    raw = raw_bgr.astype(np.float32)
    return np.clip(robot * alpha + raw * (1.0 - alpha), 0, 255).astype(np.uint8)


def review_panel(
    raw_bgr: np.ndarray,
    rgba_bgra: np.ndarray,
    frame_index: int,
    valid: np.ndarray,
) -> np.ndarray:
    overlay = alpha_compose(raw_bgr, rgba_bgra)
    checker = np.full_like(raw_bgr, 42)
    checker[::20, :] = 58
    checker[:, ::20] = 58
    robot_only = alpha_compose(checker, rgba_bgra)
    panels = [raw_bgr.copy(), overlay, robot_only]
    headings = [
        "RAW",
        "R2 KAIHAND ADDITIVE OVERLAY",
        "ROBOT-ONLY | NO ARM / NO CLEAN",
    ]
    for panel, heading in zip(panels, headings):
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 32), (0, 0, 0), -1)
        cv2.putText(
            panel,
            heading,
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    status = (
        f"004 frame={frame_index:05d} | valid L/R={int(valid[0])}/{int(valid[1])} | "
        "REVIEW ONLY: human not removed, no object-depth occlusion"
    )
    result = np.concatenate(panels, axis=1)
    cv2.rectangle(result, (0, result.shape[0] - 24), (result.shape[1], result.shape[0]), (0, 0, 0), -1)
    cv2.putText(
        result,
        status,
        (8, result.shape[0] - 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (0, 220, 255),
        1,
        cv2.LINE_AA,
    )
    return result


def _load_task_card(path: Path, expected_sha256: str, output_root: Path) -> dict[str, Any]:
    verify_file(path, expected_sha256)
    card = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version": "004-r2-kaihand-regenerated-review-task-v1",
        "auth_tier": "T1_CANDIDATE_REVIEW_ONLY",
        "candidate_requires_human_review": True,
        "next_bucket_blocked": True,
        "advancement_authorized": False,
        "session_id": "grap_a_cap_004",
        "frame_count": 460,
    }
    for key, value in required.items():
        if card.get(key) != value:
            raise ReviewVideoError(f"task card field mismatch: {key}")
    if Path(card.get("output_root", "")).resolve(strict=False) != output_root.resolve(strict=False):
        raise ReviewVideoError("task card output root mismatch")
    return card


def _source_session(source_manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    sessions = [item for item in source_manifest.get("sessions", []) if item.get("session_id") == "grap_a_cap_004"]
    if len(sessions) != 1 or sessions[0].get("frame_count") != 460:
        raise ReviewVideoError("verified source manifest has no unique 460-frame 004")
    return sessions[0]


def _load_sidecar(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        if str(data["schema_version"].item()) != "humanego-robot-sidecar-v1":
            raise ReviewVideoError("unsupported sidecar schema")
        frame_names = np.asarray(data["frame_names"])
        q = np.asarray(data["q"], dtype=np.float64)
        wrist = np.asarray(data["wrist_T_camera"], dtype=np.float64)
        valid = np.asarray(data["valid"], dtype=bool)
        confidence = np.asarray(data["confidence"], dtype=np.float64)
        names = np.asarray(data["joint_names"])
    expected_names = np.asarray([f"{index:05d}" for index in range(460)])
    if not np.array_equal(frame_names, expected_names):
        raise ReviewVideoError("sidecar frame identity/order mismatch")
    if q.shape != (460, 2, 22) or wrist.shape != (460, 2, 4, 4):
        raise ReviewVideoError("sidecar q/wrist shape mismatch")
    if valid.shape != (460, 2) or confidence.shape != (460, 2) or names.shape != (2, 22):
        raise ReviewVideoError("sidecar validity/joint identity shape mismatch")
    if not np.isfinite(q).all() or not np.isfinite(wrist).all():
        raise ReviewVideoError("sidecar contains NaN/Inf")
    return {"q": q, "wrist": wrist, "valid": valid, "confidence": confidence, "joint_names": names}


def _configure_eevee(scene: Any, width: int, height: int) -> None:
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = True
    scene.render.use_file_extension = True
    scene.render.image_settings.color_mode = "RGBA"
    scene.view_settings.look = "AgX - Medium High Contrast"


def _load_robot_objects(models: list[core.UrdfModel]) -> tuple[list[dict[str, Any]], list[dict[str, np.ndarray]]]:
    import bpy

    objects: list[dict[str, Any]] = []
    fk_static: list[dict[str, np.ndarray]] = []
    side_colors = ((0.18, 0.48, 0.92, 1.0), (0.95, 0.38, 0.12, 1.0))
    for side_index, model in enumerate(models):
        fk_static.append({})
        material = core._make_material(f"review_side_{side_index}", side_colors[side_index])
        principled = material.node_tree.nodes.get("Principled BSDF")
        principled.inputs["Metallic"].default_value = 0.45
        principled.inputs["Roughness"].default_value = 0.28
        for visual_index, visual in enumerate(model.visuals):
            obj = core._import_visual(visual.mesh_path, f"side{side_index}:{visual.link}:{visual_index}")
            obj.data.materials.clear()
            obj.data.materials.append(material)
            objects.append(
                {
                    "side": side_index,
                    "link": visual.link,
                    "origin": visual.origin,
                    "object": obj,
                }
            )
    bpy.context.view_layer.update()
    return objects, fk_static


def _encode_start(path: Path, width: int, height: int, fps: float) -> subprocess.Popen[bytes]:
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pixel_format", "bgr24",
        "-video_size", f"{width}x{height}", "-framerate", f"{fps:.8f}",
        "-i", "-", "-an", "-c:v", "libx264", "-preset", "fast",
        "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


def _ffprobe(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
            "-show_entries", "stream=codec_name,width,height,avg_frame_rate,nb_read_frames",
            "-of", "json", str(path),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise ReviewVideoError(f"ffprobe failed: {completed.stderr}")
    return json.loads(completed.stdout)["streams"][0]


def run(args: argparse.Namespace) -> dict[str, Any]:
    import bpy
    from mathutils import Matrix

    project_root = args.project_root.resolve()
    output_root = args.output_root.resolve(strict=False)
    require_new_output_root(project_root, output_root)
    card = _load_task_card(args.task_card, args.expected_task_card_sha256, output_root)

    refs = card["inputs"]
    resolved_refs: dict[str, Path] = {}
    for key, ref in refs.items():
        path = Path(ref["path"])
        verify_file(path, ref["sha256"])
        resolved_refs[key] = path
    verify_file(Path(card["implementation"]["path"]), card["implementation"]["sha256"])

    source_manifest = json.loads(resolved_refs["source_manifest"].read_text(encoding="utf-8"))
    source_session = _source_session(source_manifest)
    sidecar = _load_sidecar(resolved_refs["sidecar"])
    left_model = core.parse_urdf(resolved_refs["left_urdf"])
    right_model = core.parse_urdf(resolved_refs["right_urdf"])
    core._verify_used_assets(
        project_root,
        resolved_refs["asset_pin"],
        refs["asset_pin"]["sha256"],
        (left_model, right_model),
    )

    first_metadata = json.loads(source_session["frames"][0]["metadata"]["path"] and Path(source_session["frames"][0]["metadata"]["path"]).read_text(encoding="utf-8"))
    metadata = first_metadata["metadata"]
    original_k = np.asarray(metadata["k"], dtype=np.float64)
    original_width, original_height = int(metadata["w"]), int(metadata["h"])
    k = original_k.copy()
    k[0, :] *= args.width / original_width
    k[1, :] *= args.height / original_height

    output_root.mkdir(mode=0o755)
    frames_root = output_root / "robot_rgba_frames"
    frames_root.mkdir(mode=0o755)
    partial_video = output_root / "004_R2_KAIHAND_REGENERATED_REVIEW.partial.mp4"
    final_video = output_root / "004_R2_KAIHAND_REGENERATED_REVIEW.mp4"

    core._clear_scene()
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    _configure_eevee(scene, args.width, args.height)
    core._camera_from_k(scene, k, args.width, args.height)
    core._add_lighting(scene)
    objects, _ = _load_robot_objects([left_model, right_model])
    models = [left_model, right_model]

    encoder = _encode_start(partial_video, args.width * 3, args.height, float(source_session["fps"]))
    start = time.perf_counter()
    frame_metrics: list[dict[str, Any]] = []
    try:
        assert encoder.stdin is not None
        for frame_index, source_frame in enumerate(source_session["frames"]):
            image_ref, metadata_ref = source_frame["image"], source_frame["metadata"]
            image_path, metadata_path = Path(image_ref["path"]), Path(metadata_ref["path"])
            if image_path.stat().st_size != image_ref["bytes"] or sha256_file(image_path) != image_ref["sha256"]:
                raise ReviewVideoError(f"RAW image drift at frame {frame_index}")
            if metadata_path.stat().st_size != metadata_ref["bytes"] or sha256_file(metadata_path) != metadata_ref["sha256"]:
                raise ReviewVideoError(f"RAW metadata drift at frame {frame_index}")

            link_fk: list[dict[str, np.ndarray]] = []
            for side in range(2):
                q_by_joint = dict(zip(map(str, sidecar["joint_names"][side]), map(float, sidecar["q"][frame_index, side])))
                link_fk.append(core.forward_kinematics(models[side], q_by_joint))
            for item in objects:
                side = item["side"]
                obj = item["object"]
                is_valid = bool(sidecar["valid"][frame_index, side])
                obj.hide_render = not is_valid
                obj.hide_viewport = not is_valid
                if is_valid:
                    transform = (
                        core.CV_CAMERA_TO_BLENDER
                        @ sidecar["wrist"][frame_index, side]
                        @ link_fk[side][item["link"]]
                        @ item["origin"]
                    )
                    obj.matrix_world = Matrix(transform.tolist())
            rgba_path = frames_root / f"{frame_index:05d}.png"
            scene.render.filepath = str(rgba_path)
            bpy.context.view_layer.update()
            bpy.ops.render.render(write_still=True)
            rgba = cv2.imread(str(rgba_path), cv2.IMREAD_UNCHANGED)
            raw = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if rgba is None or rgba.shape != (args.height, args.width, 4):
                raise ReviewVideoError(f"invalid robot RGBA at frame {frame_index}")
            if raw is None:
                raise ReviewVideoError(f"RAW decode failed at frame {frame_index}")
            raw = cv2.resize(raw, (args.width, args.height), interpolation=cv2.INTER_AREA)
            panel = review_panel(raw, rgba, frame_index, sidecar["valid"][frame_index])
            encoder.stdin.write(panel.tobytes())
            foreground = int(np.count_nonzero(rgba[..., 3]))
            frame_metrics.append(
                {
                    "frame_index": frame_index,
                    "valid": [bool(value) for value in sidecar["valid"][frame_index]],
                    "confidence": [float(value) for value in sidecar["confidence"][frame_index]],
                    "robot_foreground_pixels": foreground,
                    "rgba": {
                        "path": str(rgba_path.relative_to(output_root)),
                        "bytes": rgba_path.stat().st_size,
                        "sha256": sha256_file(rgba_path),
                    },
                }
            )
        encoder.stdin.close()
        stderr = encoder.stderr.read().decode("utf-8", errors="replace") if encoder.stderr else ""
        return_code = encoder.wait()
        if return_code != 0:
            raise ReviewVideoError(f"ffmpeg encoding failed: {stderr}")
    except BaseException:
        encoder.kill()
        encoder.wait()
        raise
    os.replace(partial_video, final_video)
    probe = _ffprobe(final_video)
    if int(probe["nb_read_frames"]) != 460:
        raise ReviewVideoError("encoded video does not contain 460 frames")
    decode = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(final_video), "-f", "null", "-"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    if decode.returncode != 0:
        raise ReviewVideoError("full video decode verification failed")

    manifest = {
        "schema_version": "004-r2-kaihand-regenerated-review-manifest-v1",
        "status": "candidate_requires_human_review",
        "next_bucket_blocked": True,
        "advancement_authorized": False,
        "claim_limit": "NEW_R2_KAIHAND_MESH_REVIEW_NOT_DELETED_HISTORICAL_VIDEO_NOT_ROBOTRGB",
        "session_id": "grap_a_cap_004",
        "frame_count": 460,
        "resolution": [args.width * 3, args.height],
        "fps": float(source_session["fps"]),
        "renderer": "BLENDER_EEVEE_NEXT_REVIEW_ONLY",
        "frozen_r2_q_hand_and_wrist": True,
        "human_removed": False,
        "clean_used": False,
        "tianji_arm_rendered": False,
        "q_arm_available": False,
        "object_depth_occlusion_applied": False,
        "known_limitations": [
            "human hand/arm remains in additive overlay",
            "Tianji arm is absent because frozen q_arm/calibration/base placement do not exist",
            "robot/object depth ordering and final compositor are not applied",
            "fixed diagnostic lighting/material is not scene calibrated",
        ],
        "task_card": {"path": str(args.task_card), "sha256": args.expected_task_card_sha256},
        "inputs": refs,
        "metrics": {
            "wall_seconds": time.perf_counter() - start,
            "valid_left": int(sidecar["valid"][:, 0].sum()),
            "valid_right": int(sidecar["valid"][:, 1].sum()),
            "foreground_nonzero_frames": int(sum(item["robot_foreground_pixels"] > 0 for item in frame_metrics)),
            "foreground_pixels_min": int(min(item["robot_foreground_pixels"] for item in frame_metrics)),
            "foreground_pixels_max": int(max(item["robot_foreground_pixels"] for item in frame_metrics)),
        },
        "video": {
            "path": str(final_video),
            "bytes": final_video.stat().st_size,
            "sha256": sha256_file(final_video),
            "ffprobe": probe,
        },
        "frames": frame_metrics,
    }
    atomic_json(output_root / "RUN_MANIFEST.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--task-card", type=Path, required=True)
    parser.add_argument("--expected-task-card-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    args = parser.parse_args()
    if args.width != 320 or args.height != 240:
        parser.error("review v1 is frozen to 320x240 panels")
    try:
        manifest = run(args)
    except (ReviewVideoError, core.RendererError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps({"status": manifest["status"], "video": manifest["video"], "metrics": manifest["metrics"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
