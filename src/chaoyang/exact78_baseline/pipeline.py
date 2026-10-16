"""Raw PICO session -> robot video, three stages per session.

motion (CPU pool): PICO tracks -> world-fixed Tianji base, q_arm, q_hand.
gpu    (one process, models loaded once): SAM3.1 wearer/held-object masks, ProPainter clean plate.
render (CPU pool): ego robot layer + V2 ownership compositor, review and external videos.

Every stage writes into ``<output>/sessions/<session>/``. Reuse requires a
compatible producer-sealed receipt, not merely an existing finished copy.
"""

from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor
import hashlib
import html
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback
from typing import Any

import cv2
import numpy as np

from chaoyang.exact78_baseline import pico_source, stage_reuse
from chaoyang.exact78_baseline.review_labels import motion_review_title


WIDTH, HEIGHT = 640, 480
SCHEMA = "exact78-takeover-baseline-v1"
STAGE_FILES = {"motion": ("motion.npz", "motion.json"),
               "gpu": ("clean.mp4", "masks.npz", "masks.json"),
               "render": ("final.mp4", "review.mp4", "SESSION.json")}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=1, sort_keys=True, default=_json) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def _json(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value))


def session_task(session: Any) -> str:
    return Path(session.raw_dir).parent.name


# ---------------------------------------------------------------- video IO

class Encoder:
    def __init__(self, path: Path, width: int, height: int, fps: float, crf: int = 18, preset: str = "veryfast"):
        self.path = path
        binary = shutil.which("ffmpeg")
        if binary is None:
            raise RuntimeError("FFMPEG_MISSING")
        self.process = subprocess.Popen(
            [binary, "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
             "-r", f"{fps:g}", "-i", "-", "-an", "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
            stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self.shape = (height, width, 3)

    def write(self, frame: np.ndarray) -> None:
        if frame.shape != self.shape or frame.dtype != np.uint8:
            raise RuntimeError(f"ENCODER_FRAME_DOMAIN:{frame.shape}")
        self.process.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        self.process.stdin.close()
        error = self.process.stderr.read().decode(errors="replace")
        if self.process.wait() != 0:
            raise RuntimeError(f"ENCODER_FAILED:{self.path}:{error[-300:]}")


def decode(path: Path, count: int | None = None, size: tuple[int, int] = (WIDTH, HEIGHT)) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    frames = []
    while count is None or len(frames) < count:
        ok, frame = capture.read()
        if not ok:
            break
        if (frame.shape[1], frame.shape[0]) != size:
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        frames.append(frame)
    capture.release()
    if count is not None and len(frames) != count:
        raise RuntimeError(f"VIDEO_FRAME_COUNT:{path}:{len(frames)}!={count}")
    return np.stack(frames)


def pack(masks: np.ndarray) -> np.ndarray:
    return np.packbits(masks.reshape(len(masks), -1), axis=1)


def unpack(packed: np.ndarray, shape: tuple[int, int] = (HEIGHT, WIDTH)) -> np.ndarray:
    return np.unpackbits(packed, axis=1, count=shape[0] * shape[1]).reshape(len(packed), *shape).astype(bool)


# ---------------------------------------------------------------- stage lookup

class Layout:
    def __init__(self, output: Path, reuse: list[Path], reuse_stages: tuple[str, ...] = ("motion",),
                 recompute: tuple[str, ...] = (), *, repo_root: Path | None = None,
                 objects: bool = True, propainter_settings: dict | None = None):
        self.output = output
        self.reuse = reuse
        self.reuse_stages = reuse_stages
        self.recompute = set(recompute)
        self.repo_root = repo_root
        self.objects = objects
        self.propainter_settings = propainter_settings

    def own(self, session_id: str) -> Path:
        return self.output / "sessions" / session_id

    def find(self, session_id: str, stage: str, *, upstream: dict | None = None) -> Path | None:
        reusable = stage in self.reuse_stages and not (stage != "motion" and session_id in self.recompute)
        roots = (self.output, *self.reuse) if reusable else (self.output,)
        for root in roots:
            folder = root / "sessions" / session_id
            # A legacy OK/partial file set is not evidence; recompute normally.
            if not stage_reuse.receipt_path(folder, stage).is_file():
                continue
            if self.repo_root is None:
                raise stage_reuse.StageReuseDenied("STAGE_REUSE_DENIED:REPO_CONTEXT_ABSENT")
            session = pico_source.load_session(session_id)
            expected = stage_reuse.binding(self.repo_root, stage, session, objects=self.objects,
                propainter_settings=self.propainter_settings, upstream=upstream)
            if stage_reuse.validate(folder, stage, expected, self.repo_root):
                return folder
        return None


# ---------------------------------------------------------------- stages

_ROBOT = None


def _robot(repo_root: Path):
    global _ROBOT
    if _ROBOT is None:
        from chaoyang.exact78_baseline.robot_model import Robot
        _ROBOT = Robot(repo_root)
    return _ROBOT


def _worker_init() -> None:
    cv2.setNumThreads(1)


def motion_stage(repo_root: str, session_id: str, folder: str) -> dict[str, Any]:
    from chaoyang.exact78_baseline import retarget

    started = time.monotonic()
    out = Path(folder)
    out.mkdir(parents=True, exist_ok=True)
    session = pico_source.load_session(session_id)
    stage_before = stage_reuse.binding(Path(repo_root), "motion", session)
    robot = _robot(Path(repo_root))
    result = retarget.retarget_session(session, robot)
    np.savez_compressed(out / "motion.npz", q_arm=result.q_arm, q_hand=result.q_hand,
                        T_world_base=result.T_world_base, root_world=result.root_world,
                        arm_position_mm=result.arm_position_mm, arm_rotation_deg=result.arm_rotation_deg,
                        hand_bone_deg=result.hand_bone_deg, c2w=session.c2w,
                        hand_source=session.hand_source.astype("U12"))
    labels = {name: {label: int((session.hand_source[side] == label).sum())
                     for label in np.unique(session.hand_source[side])}
              for side, name in enumerate(("left", "right"))}
    value = {"schema_version": SCHEMA + "-motion", "status": "OK", "session": session_id,
             "task": session_task(session), "frame_count": session.frame_count, "fps": session.fps,
             "calibration": session.calibration, "hand_source_counts": labels,
             "body_observed_frames": int(session.body_observed.sum()),
             "metrics": result.metrics, "base": result.base_report,
             "seconds": round(time.monotonic() - started, 2), "finished_at": now()}
    write_json(out / "motion.json", value)
    stage_reuse.seal(Path(repo_root), out, "motion", session, before=stage_before)
    return {"session": session_id, "stage": "motion", "seconds": value["seconds"]}


def gpu_stage(models: dict[str, Any], session_id: str, folder: Path) -> dict[str, Any]:
    from chaoyang.exact78_baseline import mask as masks_module

    started = time.monotonic()
    folder.mkdir(parents=True, exist_ok=True)
    session = pico_source.load_session(session_id)
    stage_before = stage_reuse.binding(models["repo_root"], "gpu", session,
        objects=models.get("objects", True), propainter_settings=models.get("propainter_settings"))
    if stage_before["dependencies"] != models["model_dependencies"]:
        raise stage_reuse.StageReuseDenied("STAGE_REUSE_DENIED:MODEL_DEPENDENCIES_CHANGED_AFTER_LOAD")
    count, shape = session.frame_count, (HEIGHT, WIDTH)
    frames = decode(session.video, count)
    rgb = np.ascontiguousarray(frames[..., ::-1])
    geometry = masks_module.pico_geometry(session, WIDTH / session.width)
    anchor = masks_module.anchor_frame(geometry, shape)
    timings = {"decode_s": round(time.monotonic() - started, 2)}
    tick = time.monotonic()
    core, removal, info = masks_module.track_wearer(models["sam"], rgb, geometry, count, shape, anchor)
    timings["human_mask_s"] = round(time.monotonic() - tick, 2)
    task = session_task(session)
    prompt = masks_module.OBJECT_PROMPTS.get(task)
    held = np.zeros_like(core)
    kept = np.zeros_like(core)
    object_rows = None
    if prompt and models.get("objects", True):
        tick = time.monotonic()
        object_anchor = anchor
        object_tracks = models["sam"].track_text(list(rgb), prompt, object_anchor)
        object_aborted = list(models["sam"].aborted)
        if not object_tracks and object_aborted and count // 2 != anchor:
            # Nothing at the hands-anchor frame (e.g. chips still in the bag): one retry mid-video.
            object_anchor = count // 2
            object_tracks = models["sam"].track_text(list(rgb), prompt, object_anchor)
            object_aborted = list(models["sam"].aborted)
        timings["sam_object_s"] = round(time.monotonic() - tick, 2)
        held = masks_module.held_objects(object_tracks, core, shape)
        kept = masks_module.object_pixels(object_tracks, core, shape)
        object_rows = {"prompt": prompt, "tracks": len(object_tracks), "anchor_frame": object_anchor,
                       "propagation_aborted": object_aborted,
                       "held_frames": int(held.any(axis=(1, 2)).sum()),
                       "kept_in_removal_fraction_mean": float((kept & removal).mean())}
        del object_tracks
    removal &= ~(held | kept)
    tick = time.monotonic()
    clean = models["propainter"].inpaint(rgb, removal)[..., ::-1]
    timings["propainter_s"] = round(time.monotonic() - tick, 2)
    encoder = Encoder(folder / "clean.mp4", WIDTH, HEIGHT, session.fps, crf=12)
    for frame in clean:
        encoder.write(frame)
    encoder.close()
    provider = info.pop("provider")
    shift = info.pop("pico_shift_px")
    registered = shift[provider == 3]
    info["registered_shift_px_median"] = (float(np.median(np.linalg.norm(registered, axis=-1)))
                                          if len(registered) else None)
    np.savez_compressed(folder / "masks.npz", core=pack(core), removal=pack(removal), held=pack(held),
                        provider=provider, pico_shift_px=shift.astype(np.float32), shape=np.asarray(shape))
    value = {"schema_version": SCHEMA + "-masks", "status": "OK", "session": session_id, "task": task,
             "frame_count": count, "anchor_frame": anchor, "human_prompt": masks_module.HUMAN_PROMPT,
             "objects": object_rows, "human": info,
             "provider_counts": {side: {name: int((provider[:, i] == int(code)).sum())
                                        for code, name in info["provider_enum"].items()}
                                 for i, side in enumerate(("left", "right"))},
             "removal_fraction_mean": float(removal.mean()), "held_fraction_mean": float(held.mean()),
             "propainter": models["propainter"].settings, "timings": timings,
             "seconds": round(time.monotonic() - started, 2), "finished_at": now()}
    write_json(folder / "masks.json", value)
    stage_reuse.seal(models["repo_root"], folder, "gpu", session, objects=models.get("objects", True),
        propainter_settings=models.get("propainter_settings"), before=stage_before)
    return {"session": session_id, "stage": "gpu", "seconds": value["seconds"], "timings": timings}


def render_stage(repo_root: str, session_id: str, folder: str, motion_dir: str, gpu_dir: str) -> dict[str, Any]:
    from chaoyang.exact78_baseline import render as rd
    from chaoyang.pipeline.exact78_visual_compositor_v1 import compose_visual_frame

    started = time.monotonic()
    out = Path(folder)
    out.mkdir(parents=True, exist_ok=True)
    session = pico_source.load_session(session_id)
    stage_upstream = {"motion": Path(motion_dir) / "motion.npz", "masks": Path(gpu_dir) / "masks.npz",
                      "clean": Path(gpu_dir) / "clean.mp4"}
    stage_before = stage_reuse.binding(Path(repo_root), "render", session, upstream=stage_upstream)
    count = session.frame_count
    robot = _robot(Path(repo_root))
    renderer = rd.RobotRenderer(robot)
    motion = np.load(Path(motion_dir) / "motion.npz")
    stored = np.load(Path(gpu_dir) / "masks.npz")
    core, removal, held = (unpack(stored[name]) for name in ("core", "removal", "held"))
    raw = decode(session.video, count)
    clean = decode(Path(gpu_dir) / "clean.mp4", count)
    K = rd.scaled_intrinsics(session.K, WIDTH, HEIGHT, (session.width, session.height))
    T_W_B = motion["T_world_base"]
    external = rd.external_camera(np.linalg.inv(session.c2w[0]) @ T_W_B)
    final_video = Encoder(out / "final.mp4", WIDTH, HEIGHT, session.fps, crf=17, preset="medium")
    review_video = Encoder(out / "review.mp4", WIDTH * 3, HEIGHT * 2, session.fps, crf=23)
    unknown = np.zeros((HEIGHT, WIDTH), bool)
    hand_source = motion["hand_source"]
    rows = {"robot_pixels": [], "object_front_pixels": [], "removal_pixels": [],
            "removal_uncovered_pixels": []}
    try:
        for frame in range(count):
            camera = np.linalg.inv(session.c2w[frame]) @ T_W_B
            bgr, alpha, _ = renderer.render(motion["q_arm"][frame], motion["q_hand"][frame], camera, K,
                                            complete_robot=False)
            composed = compose_visual_frame(clean[frame], bgr, alpha, held[frame], core[frame], unknown,
                                            robot_observed=True, masks_available=True)
            final = rd.over(clean[frame], bgr, composed["draw"])
            final[composed["object_front"]] = clean[frame][composed["object_front"]]
            final_video.write(final)
            ebgr, ealpha, _ = renderer.render(motion["q_arm"][frame], motion["q_hand"][frame], external,
                                              rd.EXTERNAL_K, complete_robot=True)
            skeleton = raw[frame].copy()
            for side, colour in ((0, (255, 128, 0)), (1, (0, 128, 255))):
                if hand_source[side, frame] != "NONE":
                    points = session.project(session.camera_points(session.hands[side, frame], frame))
                    rd.draw_hand(skeleton, points * (WIDTH / session.width), colour,
                                 f"{'LR'[side]}:{hand_source[side, frame]}")
            overlay = raw[frame].copy()
            overlay[removal[frame]] = (0.4 * overlay[removal[frame]] + 0.6 * np.asarray((255, 0, 255))).astype(np.uint8)
            overlay[held[frame]] = (0.4 * overlay[held[frame]] + 0.6 * np.asarray((255, 160, 0))).astype(np.uint8)
            review = np.concatenate((
                np.concatenate((rd.title(skeleton, motion_review_title(session, frame, count)),
                                rd.title(overlay, "REMOVAL (MAGENTA) / HELD OBJECT (BLUE)"),
                                rd.title(clean[frame], "CLEAN (PROPAINTER)")), axis=1),
                np.concatenate((rd.title(rd.on_grey(bgr, alpha), "EGO ROBOT"),
                                rd.title(rd.on_grey(ebgr, ealpha), "EXTERNAL ROBOT"),
                                rd.title(final, "FINAL")), axis=1)), axis=0)
            review_video.write(review)
            rows["robot_pixels"].append(int(alpha.sum()))
            rows["object_front_pixels"].append(int(composed["object_front"].sum()))
            rows["removal_pixels"].append(int(removal[frame].sum()))
            rows["removal_uncovered_pixels"].append(int((removal[frame] & ~composed["draw"]).sum()))
    finally:
        final_video.close()
        review_video.close()
    motion_info = json.loads((Path(motion_dir) / "motion.json").read_text())
    mask_info = json.loads((Path(gpu_dir) / "masks.json").read_text())
    value = {"schema_version": SCHEMA + "-session", "status": "OK", "session": session_id,
             "task": session_task(session), "frame_count": count, "fps": session.fps,
             "raw_video": str(session.video),
             "outputs": {"final": ref(out / "final.mp4"), "review": ref(out / "review.mp4"),
                         "clean": ref(Path(gpu_dir) / "clean.mp4"), "masks": ref(Path(gpu_dir) / "masks.npz"),
                         "motion": ref(Path(motion_dir) / "motion.npz")},
             "motion": {"metrics": motion_info["metrics"], "hand_source_counts": motion_info["hand_source_counts"],
                        "base_selected": motion_info["base"]["selected"], "calibration": motion_info["calibration"]},
             "masks": {key: mask_info[key] for key in ("provider_counts", "removal_fraction_mean",
                                                        "held_fraction_mean", "objects", "timings")},
             "render": {"robot_pixel_mean": float(np.mean(rows["robot_pixels"])),
                        "object_front_frames": int(np.count_nonzero(rows["object_front_pixels"])),
                        "removal_uncovered_fraction": float(np.sum(rows["removal_uncovered_pixels"])
                                                            / max(1, np.sum(rows["removal_pixels"])))},
             "labels": "hand_source per frame lives in motion.npz; OBSERVED / INTERP_SHORT / EASED_LONG / "
                       "EDGE_HOLD / NONE. Mask provider per frame lives in masks.npz.",
             "claim_limit": "Visual baseline candidate; not physical-occlusion, control or training authority.",
             "render_seconds": round(time.monotonic() - started, 2), "finished_at": now()}
    write_json(out / "SESSION.json", value)
    stage_reuse.seal(Path(repo_root), out, "render", session, upstream=stage_upstream, before=stage_before)
    return {"session": session_id, "stage": "render", "seconds": value["render_seconds"]}


# ---------------------------------------------------------------- orchestration

def index_page(output: Path, rows: list[dict[str, Any]]) -> None:
    body = ["<html><head><meta charset='utf-8'><title>Exact78 baseline v1</title>",
            "<style>body{font-family:sans-serif}td,th{padding:4px 8px;border-bottom:1px solid #ddd}"
            "video{width:480px}</style></head><body>",
            f"<h2>Exact78 takeover baseline v1 — {len(rows)} sessions</h2><table>",
            "<tr><th>session</th><th>status</th><th>frames</th><th>final</th><th>review</th>"
            "<th>arm pass (L/R)</th><th>removal uncovered</th></tr>"]
    for row in rows:
        folder = row.get("folder")
        links = ""
        if folder and row["status"] == "OK":
            rel = os.path.relpath(folder, output)
            links = (f"<td><a href='{rel}/final.mp4'>final</a></td><td><a href='{rel}/review.mp4'>review</a></td>")
        else:
            links = "<td></td><td></td>"
        body.append(f"<tr><td>{html.escape(row['session'])}</td><td>{html.escape(row['status'])}</td>"
                    f"<td>{row.get('frames', '')}</td>{links}<td>{row.get('arm_pass', '')}</td>"
                    f"<td>{row.get('uncovered', '')}</td></tr>")
    body.append("</table></body></html>")
    (output / "index.html").write_text("\n".join(body), encoding="utf-8")


def run(repo_root: Path, output: Path, session_ids: list[str], *, workers: int, reuse: list[Path],
        objects: bool, propainter_settings: dict | None,
        reuse_stages: tuple[str, ...] = ("motion",), recompute: tuple[str, ...] = ()) -> dict[str, Any]:
    layout = Layout(output, reuse, reuse_stages, recompute, repo_root=repo_root,
                    objects=objects, propainter_settings=propainter_settings)
    (output / "sessions").mkdir(parents=True, exist_ok=True)
    status: dict[str, dict[str, Any]] = {sid: {"session": sid, "status": "PENDING", "stages": {}}
                                         for sid in session_ids}
    started = time.monotonic()

    def publish() -> None:
        counts: dict[str, int] = {}
        for row in status.values():
            counts[row["status"]] = counts.get(row["status"], 0) + 1
        write_json(output / "PROGRESS.json", {"schema_version": SCHEMA + "-progress", "updated_at": now(),
                                              "elapsed_s": round(time.monotonic() - started, 1),
                                              "counts": counts, "sessions": list(status.values())})

    def fail(sid: str, stage: str, error: BaseException | str) -> None:
        text = error if isinstance(error, str) else "".join(traceback.format_exception(error))[-2000:]
        status[sid]["status"] = f"FAILED_{stage.upper()}"
        status[sid]["error"] = text

    pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("fork"),
                               initializer=_worker_init)
    motion_futures: dict[str, Future] = {}
    render_futures: dict[str, Future] = {}
    motion_dirs: dict[str, Path] = {}
    for sid in session_ids:
        found = layout.find(sid, "motion")
        if found:
            motion_dirs[sid] = found
            status[sid]["stages"]["motion"] = {"reused": str(found)}
        else:
            motion_futures[sid] = pool.submit(motion_stage, str(repo_root), sid, str(layout.own(sid)))
    publish()

    def collect_motion(sid: str, wait: bool) -> bool:
        if sid in motion_dirs:
            return True
        future = motion_futures[sid]
        if not wait and not future.done():
            return False
        try:
            status[sid]["stages"]["motion"] = future.result()
            motion_dirs[sid] = layout.own(sid)
            return True
        except stage_reuse.StageReuseDenied:
            raise
        except Exception as error:  # noqa: BLE001
            fail(sid, "motion", error)
            return False

    def submit_render(sid: str, gpu_dir: Path) -> None:
        if not collect_motion(sid, wait=True):
            return
        found = layout.find(sid, "render", upstream={"motion": motion_dirs[sid] / "motion.npz",
            "masks": gpu_dir / "masks.npz", "clean": gpu_dir / "clean.mp4"})
        if found:
            status[sid]["stages"]["render"] = {"reused": str(found)}
            status[sid]["status"] = "OK"
            status[sid]["folder"] = str(found)
            return
        status[sid]["status"] = "RENDERING"
        render_futures[sid] = pool.submit(render_stage, str(repo_root), sid, str(layout.own(sid)),
                                          str(motion_dirs[sid]), str(gpu_dir))

    def drain(block: bool) -> None:
        for sid, future in list(render_futures.items()):
            if block or future.done():
                try:
                    status[sid]["stages"]["render"] = future.result()
                    status[sid]["status"] = "OK"
                    status[sid]["folder"] = str(layout.own(sid))
                except stage_reuse.StageReuseDenied:
                    raise
                except Exception as error:  # noqa: BLE001
                    fail(sid, "render", error)
                del render_futures[sid]
        for sid in list(motion_futures):
            if sid not in motion_dirs and motion_futures[sid].done() and "error" not in status[sid]:
                collect_motion(sid, wait=False)

    models: dict[str, Any] | None = None
    model_seconds = None
    try:
        for sid in session_ids:
            found = layout.find(sid, "gpu")
            if found is None:
                if models is None:
                    from chaoyang.exact78_baseline.clean import ProPainter
                    from chaoyang.exact78_baseline.mask import Sam3Video
                    tick = time.monotonic()
                    stage_model_dependencies = stage_reuse.dependencies(repo_root, "gpu")
                    models = {"sam": Sam3Video(repo_root), "propainter": ProPainter(repo_root, propainter_settings),
                              "objects": objects, "repo_root": repo_root,
                              "propainter_settings": propainter_settings,
                              "model_dependencies": stage_model_dependencies}
                    model_seconds = round(time.monotonic() - tick, 1)
                status[sid]["status"] = "GPU"
                publish()
                try:
                    status[sid]["stages"]["gpu"] = gpu_stage(models, sid, layout.own(sid))
                    found = layout.own(sid)
                except stage_reuse.StageReuseDenied:
                    raise
                except Exception as error:  # noqa: BLE001
                    fail(sid, "gpu", error)
                    import torch
                    torch.cuda.empty_cache()
            else:
                status[sid]["stages"]["gpu"] = {"reused": str(found)}
            if found is not None:
                submit_render(sid, found)
            drain(block=False)
            publish()
        drain(block=True)
    finally:
        if models is not None:
            models["sam"].close()
        pool.shutdown(wait=True)
    publish()
    rows = []
    for sid in session_ids:
        row = {"session": sid, "status": status[sid]["status"], "folder": status[sid].get("folder")}
        if row["status"] == "OK":
            info = json.loads((Path(row["folder"]) / "SESSION.json").read_text())
            sides = info["motion"]["metrics"]["per_side"]
            row["frames"] = info["frame_count"]
            row["arm_pass"] = "/".join(f"{sides[s]['tracking_pass_rate_observed']:.3f}" for s in ("left", "right"))
            row["uncovered"] = f"{info['render']['removal_uncovered_fraction']:.3f}"
            row["stage_receipts"] = {stage: stage_reuse.ref(stage_reuse.receipt_path(folder, stage))
                for stage, folder in (("motion", motion_dirs[sid]),
                    ("gpu", Path(status[sid]["stages"]["gpu"].get("reused", layout.own(sid)))),
                    ("render", Path(row["folder"])))}
        rows.append(row)
    index_page(output, rows)
    summary = {"schema_version": SCHEMA + "-index", "created_at": now(), "sessions": rows,
               "counts": {s: sum(r["status"] == s for r in rows) for s in {r["status"] for r in rows}},
               "frames_ok": int(sum(r.get("frames", 0) for r in rows)),
               "elapsed_s": round(time.monotonic() - started, 1), "model_load_s": model_seconds,
               "errors": {sid: status[sid].get("error") for sid in session_ids if "error" in status[sid]}}
    write_json(output / "INDEX.json", summary)
    return summary
