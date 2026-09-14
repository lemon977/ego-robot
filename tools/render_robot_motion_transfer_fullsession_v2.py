#!/usr/bin/env python3
"""Render an exact-frame Robot motion-transfer review from terminal numeric canaries.

This is intentionally review-only. Missing HaWoR sides remain NaN in state and
cause the entire Robot panel to be blanked for that frame; no visual hold/pad is
used to conceal an UNKNOWN side.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from tools import render_poker_same_side_outward_frame0 as shared  # noqa: E402
from tools import render_poker_symmetric_chirality_flange_successor as old  # noqa: E402
from tools import render_poker_v2c03_p3_naturalv2_successor as fixed  # noqa: E402


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def probe(path: Path) -> dict:
    command = ["ffprobe", "-v", "error", "-select_streams", "v:0",
               "-count_frames", "-show_entries",
               "stream=width,height,avg_frame_rate,nb_read_frames,nb_frames,duration",
               "-of", "json", str(path)]
    return json.loads(subprocess.check_output(command, text=True))["streams"][0]


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def draw_visible_human_skeletons(panel: np.ndarray, human_uv: np.ndarray) -> None:
    """Draw only finite sides; missing HaWoR remains visibly absent."""
    for side, color in enumerate((shared.BLUE, shared.RED)):
        points = human_uv[side]
        if not np.isfinite(points).all():
            continue
        for chain in shared.MANO_CHAINS:
            for first, second in zip(chain[:-1], chain[1:], strict=True):
                cv2.line(panel, tuple(np.rint(points[first]).astype(int)),
                         tuple(np.rint(points[second]).astype(int)), color, 3, cv2.LINE_AA)
        for point in points[[0, 4, 8, 12, 16, 20]]:
            cv2.circle(panel, tuple(np.rint(point).astype(int)), 5, color, -1, cv2.LINE_AA)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=("chips", "poker"), required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--hawor", type=Path, required=True)
    ap.add_argument("--hawor-result", type=Path, required=True)
    ap.add_argument("--arm-states", type=Path, required=True)
    ap.add_argument("--arm-result", type=Path, required=True)
    ap.add_argument("--hand-states", type=Path, required=True)
    ap.add_argument("--hand-result", type=Path, required=True)
    ap.add_argument("--fixed-placement-label", required=True)
    ap.add_argument("--allow-hand-hold-review", action="store_true",
                    help="render a clearly watermarked review when hand numeric gates remain HOLD")
    ap.add_argument("--allow-arm-hold-review", action="store_true",
                    help="render a clearly watermarked review when arm numeric gates remain HOLD")
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    if args.output_dir.exists():
        raise RuntimeError(f"refusing overwrite: {args.output_dir}")
    inputs = (args.hawor, args.hawor_result, args.arm_states, args.arm_result,
              args.hand_states, args.hand_result)
    for path in inputs:
        path.resolve(strict=True)
    if any(args.session not in str(path.resolve()) for path in (args.hawor, args.hawor_result)):
        raise RuntimeError("same-session HaWoR check failed")

    hawor_result = json.load(args.hawor_result.open())
    arm_result = json.load(args.arm_result.open())
    hand_result = json.load(args.hand_result.open())
    arm_numeric_pass = arm_result["status"] == "PASS_NUMERIC_CANARY_NO_AUTHORITY"
    if not arm_numeric_pass and not (args.allow_arm_hold_review and
                                     arm_result["status"] == "HOLD_NUMERIC_CANARY"):
        raise RuntimeError("arm numeric canary is not PASS (use explicit HOLD review flag only for review)")
    hand_numeric_pass = hand_result["status"] == "PASS_NUMERIC_CANARY_NO_AUTHORITY"
    if not hand_numeric_pass and not (args.allow_hand_hold_review and
                                      hand_result["status"] == "HOLD_NUMERIC_CANARY"):
        raise RuntimeError("hand numeric canary is not PASS (use explicit HOLD review flag only for review)")
    if arm_result["session"] != args.session or hand_result["session"] != args.session:
        raise RuntimeError("arm/hand session mismatch")

    hawor = load_npz(args.hawor)
    arm = load_npz(args.arm_states)
    hand = load_npz(args.hand_states)
    n = len(hawor["c2w"])
    frames = np.asarray(hawor["original_frame_indices"], dtype=np.int64)
    if not np.array_equal(frames, np.arange(n)):
        raise RuntimeError("review contract requires exact original frame_ids 0..N-1")
    if any(len(x) != n for x in (arm["q_arm"], hand["q_hand"])):
        raise RuntimeError("state length does not equal HaWoR full-session length")
    valid = np.asarray(arm["valid_side_frame"], dtype=bool)
    hand_valid = np.asarray(hand["valid_side_frame"], dtype=bool)
    if not np.array_equal(valid, hand_valid):
        raise RuntimeError("arm/hand valid masks differ")
    if not np.all(np.isnan(arm["q_arm"][~valid.T])) or not np.all(np.isnan(hand["q_hand"][~valid.T])):
        raise RuntimeError("UNKNOWN side was filled in state")

    source_video = Path(hawor_result["inputs"]["source_video"]["path"])
    source_video.resolve(strict=True)
    source_probe = probe(source_video)
    fps_num, fps_den = map(int, source_probe["avg_frame_rate"].split("/"))
    fps = fps_num / fps_den
    if fps <= 0:
        raise RuntimeError("invalid source fps")

    args.output_dir.mkdir(parents=True)
    tmp_video = args.output_dir / "STAGING_OPENCV.mp4"
    final_video = args.output_dir / f"{args.session}_ROBOT_WORLD_FIRST_GAIN1_FULLSESSION.mp4"
    manifest = args.output_dir / "FRAME_MANIFEST.jsonl"
    writer = cv2.VideoWriter(str(tmp_video), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (fixed.WIDTH * 3, fixed.HEIGHT))
    if not writer.isOpened():
        raise RuntimeError("OpenCV staging writer failed")
    capture = cv2.VideoCapture(str(source_video))
    if not capture.isOpened():
        raise RuntimeError("source capture failed")
    assets = old.load_pinned_robot_assets(PROJECT)
    raster = old.load_module(old.RASTER_SOURCE, f"{args.session}_gain1_full_raster")
    flange = fixed.naturalv2_local_triangles()
    cache = {}
    world_base = arm["T_world_base"]
    if world_base.ndim == 3:
        if not np.allclose(world_base, world_base[0], atol=1e-12):
            raise RuntimeError("world base moved during session")
        world_base = world_base[0]
    mounts = arm["T_tool_hand_root"]
    global_camera = shared.same_direction_global_camera(np.linalg.inv(hawor["c2w"][0]) @ world_base)
    global_k = np.asarray(((575.0, 0, 319.5), (0, 575.0, 239.5), (0, 0, 1.0)))
    manifest_rows = []
    try:
        for frame in range(n):
            ok, native = capture.read()
            if not ok or native is None:
                raise RuntimeError(f"missing source frame {frame}")
            h, w = native.shape[:2]
            raw = cv2.resize(native, (fixed.WIDTH, fixed.HEIGHT), interpolation=cv2.INTER_AREA)
            k = hawor["intrinsics"][frame].copy()
            k[0] *= fixed.WIDTH / w
            k[1] *= fixed.HEIGHT / h
            uv = hawor["joints_2d"][:, frame].copy()
            uv[..., 0] *= fixed.WIDTH / w
            uv[..., 1] *= fixed.HEIGHT / h
            src = raw.copy()
            draw_visible_human_skeletons(src, uv)
            src = fixed.title(src, f"{args.session}｜人手原始帧 {frame}", "蓝=左手｜红=右手", "HaWoR temporal successor")

            statuses = ["SOLVED" if valid[s, frame] else "UNKNOWN_HAWOR_SIDE" for s in range(2)]
            if all(valid[:, frame]):
                camera = np.linalg.inv(hawor["c2w"][frame]) @ world_base
                color, label, _ = shared.render_robot(raster, assets, arm["q_arm"][frame],
                                                       hand["q_hand"][frame], mounts, camera, k,
                                                       cache, flange, complete_robot=False)
                overlay = raw.copy()
                overlay[label >= 0] = color[label >= 0]
                color, label, _ = shared.render_robot(raster, assets, arm["q_arm"][frame],
                                                       hand["q_hand"][frame], mounts, global_camera,
                                                       global_k, cache, flange, complete_robot=True)
                global_view = np.full_like(color, 242)
                global_view[label >= 0] = color[label >= 0]
                missing_note = "both sides SOLVED"
            else:
                overlay = np.full_like(raw, 242)
                global_view = np.full_like(raw, 242)
                cv2.putText(overlay, "UNKNOWN: missing HaWoR side; Robot not rendered",
                            (32, 250), cv2.FONT_HERSHEY_SIMPLEX, .58, (20, 20, 180), 2, cv2.LINE_AA)
                cv2.putText(global_view, "UNKNOWN frame (no hold / no pad)",
                            (70, 250), cv2.FONT_HERSHEY_SIMPLEX, .65, (20, 20, 180), 2, cv2.LINE_AA)
                missing_note = "Robot panels blanked; no substituted state"
            numeric_banner = ("REVIEW ONLY｜authority=false" if arm_numeric_pass and hand_numeric_pass else
                              ("ARM NUMERIC HOLD｜REVIEW ONLY｜NO AUTHORITY" if not arm_numeric_pass else
                               "NUMERIC HAND HOLD｜REVIEW ONLY｜NO AUTHORITY"))
            overlay = fixed.title(overlay, "Robot 同侧 world-first 叠加",
                                  f"motion gain=1.0｜{args.fixed_placement_label}",
                                  numeric_banner)
            global_view = fixed.title(global_view, "固定整机视图", "整段 base 不动｜暖白手/法兰",
                                      "数值门 PASS｜待人工视觉确认")
            writer.write(np.hstack((src, overlay, global_view)))
            manifest_rows.append({"output_frame": frame, "original_frame_id": frame,
                                  "left": statuses[0], "right": statuses[1], "render_policy": missing_note})
    finally:
        capture.release()
        writer.release()

    subprocess.run(["ffmpeg", "-xerror", "-y", "-loglevel", "error", "-i", str(tmp_video),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", f"{fps_num}/{fps_den}",
                    "-frames:v", str(n), str(final_video)], check=True)
    tmp_video.unlink()
    out_probe = probe(final_video)
    decoded = int(out_probe.get("nb_read_frames") or out_probe.get("nb_frames") or 0)
    width, height = int(out_probe["width"]), int(out_probe["height"])
    if decoded != n or (width, height) != (fixed.WIDTH * 3, fixed.HEIGHT):
        raise RuntimeError(f"encoded closure failed: frames={decoded}, size={width}x{height}")
    with manifest.open("w") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    lineage = {}
    for name, path in zip(("hawor_npz", "hawor_result", "arm_states", "arm_result",
                           "hand_states", "hand_result"), inputs):
        lineage[name] = {"path": str(path.resolve()), "sha256": sha(path), "bytes": path.stat().st_size}
    lineage["source_video"] = {"path": str(source_video.resolve()), "sha256": sha(source_video),
                               "bytes": source_video.stat().st_size}
    result = {
        "schema_version": "robot-motion-transfer-fullsession-review-v2",
        "created_at": now(), "status": ("PASS_NUMERIC_RENDER_READY_FOR_HUMAN_REVIEW"
                                          if arm_numeric_pass and hand_numeric_pass else
                                          ("HOLD_ARM_NUMERIC_RENDER_READY_FOR_HUMAN_REVIEW" if not arm_numeric_pass else
                                           "HOLD_HAND_NUMERIC_RENDER_READY_FOR_HUMAN_REVIEW")),
        "task": args.task, "session": args.session, "frame_count": n,
        "original_frame_ids": {"first": 0, "last": n - 1, "exact_contiguous": True},
        "source_fps": f"{fps_num}/{fps_den}", "motion_gain": 1.0,
        "fixed_placement": args.fixed_placement_label,
        "lineage": lineage,
        "contract": {
            "base": "one fixed placement for the entire session; never dynamically moved",
            "arm": "strict 10mm/5deg/60deg/15deg/Link5 and 0.12/0.06 temporal gates unchanged",
            "hand": "thumb q[0:6] independent; four fingers each MCP->PIP->DIP->TIP; no arc-length resampling",
            "missing": "UNKNOWN NaN in state; full Robot panels blanked; no loop/hold/pad/interpolation",
        },
        "numeric": {"arm": arm_result["metrics"], "hand": hand_result["metrics"],
                    "arm_pass": arm_numeric_pass, "hand_pass": hand_numeric_pass},
        "outputs": {
            "video": {"path": str(final_video), "sha256": sha(final_video), "bytes": final_video.stat().st_size,
                      "decoded_frames": decoded, "width": width, "height": height},
            "frame_manifest": {"path": str(manifest), "sha256": sha(manifest), "bytes": manifest.stat().st_size,
                               "rows": len(manifest_rows)},
        },
        "authority": False, "action_sidecar_published": False,
        "claim_limit": "Fresh current-algorithm strict numeric review video only; pending human visual approval and not Robot/action/contact authority.",
    }
    result_path = args.output_dir / "RESULT.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    review_path = args.output_dir / "AGENT_REVIEW.json"
    review = {"schema_version": "robot-review-agent-review-v1", "created_at": now(),
              "result": {"path": str(result_path), "sha256": sha(result_path)},
              "decision": ("READY_FOR_HUMAN_VISUAL_REVIEW_NOT_AUTHORITY" if arm_numeric_pass and hand_numeric_pass else
                           ("HOLD_ARM_NUMERIC_BUT_WATERMARKED_VISUAL_REVIEW_NOT_AUTHORITY" if not arm_numeric_pass else
                            "HOLD_HAND_NUMERIC_BUT_WATERMARKED_VISUAL_REVIEW_NOT_AUTHORITY")),
              "checks": {"numeric_arm_pass": arm_numeric_pass, "numeric_hand_pass": hand_numeric_pass,
                         "exact_frame_closure": True, "no_missing_fill": True,
                         "motion_gain_one": True, "base_fixed": True},
              "authority": False}
    review_path.write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n")
    sha_path = args.output_dir / "SHA256SUMS"
    with sha_path.open("w") as f:
        for path in (final_video, manifest, result_path, review_path):
            f.write(f"{sha(path)}  {path.name}\n")
    print(json.dumps({"result": str(result_path), "result_sha256": sha(result_path),
                      "video": str(final_video), "video_sha256": sha(final_video),
                      "frames": n, "fps": fps}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
