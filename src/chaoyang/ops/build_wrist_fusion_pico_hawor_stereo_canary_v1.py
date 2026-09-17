#!/usr/bin/env python3
"""Build a development-only PICO/HaWoR/Stereo causal wrist-fusion canary.

PICO contributes stable 6-DoF motion after a per-side local-frame anatomical
offset is estimated from a short calibration prefix. HaWoR contributes wrist
image/anatomy placement. Stereo contributes optical-Z only after a prefix
surface-to-wrist offset; it is never treated as an anatomical wrist truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


SIDES = ("left", "right")
COLORS = {
    "PICO raw": (42, 208, 70),
    "PICO calibrated": (70, 220, 230),
    "HaWoR": (20, 145, 255),
    "Stereo surface": (220, 50, 220),
    "Fused": (255, 240, 60),
}


def font_path() -> str | None:
    for value in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(value).is_file():
            return value
    return None


def put_text(image: np.ndarray, rows: list[tuple[int, int, str, tuple[int, int, int], int]]) -> np.ndarray:
    canvas = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(canvas)
    cache: dict[int, ImageFont.ImageFont] = {}
    for x, y, value, bgr, size in rows:
        if size not in cache:
            cache[size] = ImageFont.truetype(font_path(), size) if font_path() else ImageFont.load_default()
        draw.text((x, y), value, font=cache[size], fill=(bgr[2], bgr[1], bgr[0]))
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, object]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def finite_vec(value: object) -> np.ndarray | None:
    if not isinstance(value, list) or len(value) != 3:
        return None
    array = np.asarray(value, np.float64)
    return array if np.isfinite(array).all() else None


def load_inputs(session: Path, metrics: Path, hawor_npz: Path):
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((session / "preprocess/all_data").glob("*/training_data.json"))]
    comparison = json.loads(metrics.read_text(encoding="utf-8"))
    records = comparison["per_frame_delta_xyz_mm"]
    hawor_data = np.load(hawor_npz, allow_pickle=True)
    if not (len(rows) == len(records) == hawor_data["joints_3d_camera"].shape[1]):
        raise RuntimeError("frame count mismatch")
    n = len(rows)
    camera = {key: np.full((2, n, 3), np.nan, np.float64) for key in ("PICO raw", "HaWoR", "Stereo surface")}
    rotations = np.full((2, n, 3, 3), np.nan, np.float64)
    for frame, (row, record) in enumerate(zip(rows, records, strict=True)):
        for side_index, side in enumerate(SIDES):
            hand = row["entities"]["hands"][side]
            if hand.get("pose_source") != "pico_wrist_joint":
                continue
            pose = np.asarray(hand["T_wrist_to_camera"], np.float64)
            pico = pose[:3, 3]
            rotations[side_index, frame] = pose[:3, :3]
            camera["PICO raw"][side_index, frame] = pico
            hd = finite_vec(record[side].get("hawor_minus_pico"))
            sd = finite_vec(record[side].get("stereo_minus_pico"))
            if hd is not None:
                camera["HaWoR"][side_index, frame] = pico + hd / 1000.0
            if sd is not None:
                camera["Stereo surface"][side_index, frame] = pico + sd / 1000.0
    return rows, comparison, hawor_data, camera, rotations


def huber_weight(distance_m: float, cutoff_m: float = 0.05) -> float:
    return 1.0 if distance_m <= cutoff_m else cutoff_m / max(distance_m, 1e-9)


def build_fusion(camera: dict[str, np.ndarray], rotations: np.ndarray, prefix_valid: int):
    pico = camera["PICO raw"]
    hawor = camera["HaWoR"]
    stereo = camera["Stereo surface"]
    pico_cal = np.full_like(pico, np.nan)
    stereo_cal = np.full_like(stereo, np.nan)
    fused = np.full_like(pico, np.nan)
    calibration: dict[str, object] = {}
    for side_index, side in enumerate(SIDES):
        both = np.isfinite(pico[side_index]).all(1) & np.isfinite(hawor[side_index]).all(1) & np.isfinite(rotations[side_index]).all((1, 2))
        idx = np.flatnonzero(both)[:prefix_valid]
        if len(idx) < 5:
            raise RuntimeError(f"not enough PICO/HaWoR calibration frames for {side}")
        local_offsets = np.stack([rotations[side_index, i].T @ (hawor[side_index, i] - pico[side_index, i]) for i in idx])
        local_offset = np.median(local_offsets, axis=0)
        for frame in range(pico.shape[1]):
            if np.isfinite(pico[side_index, frame]).all() and np.isfinite(rotations[side_index, frame]).all():
                pico_cal[side_index, frame] = pico[side_index, frame] + rotations[side_index, frame] @ local_offset
        both_stereo = np.isfinite(stereo[side_index]).all(1) & np.isfinite(hawor[side_index]).all(1)
        stereo_idx = np.flatnonzero(both_stereo)[:prefix_valid]
        stereo_z_offset = float(np.median(hawor[side_index, stereo_idx, 2] - stereo[side_index, stereo_idx, 2])) if len(stereo_idx) >= 5 else np.nan
        stereo_cal[side_index] = stereo[side_index]
        if np.isfinite(stereo_z_offset):
            stereo_cal[side_index, :, 2] += stereo_z_offset
        previous = None
        for frame in range(pico.shape[1]):
            candidates: list[tuple[np.ndarray, float]] = []
            if np.isfinite(pico_cal[side_index, frame]).all():
                candidates.append((pico_cal[side_index, frame], 0.50))
            if np.isfinite(hawor[side_index, frame]).all():
                candidates.append((hawor[side_index, frame], 0.35))
            if not candidates:
                continue
            center = np.median(np.stack([p for p, _ in candidates]), axis=0)
            weighted = [(p, w * huber_weight(float(np.linalg.norm(p - center)))) for p, w in candidates]
            measurement = sum(p * w for p, w in weighted) / sum(w for _, w in weighted)
            if np.isfinite(stereo_cal[side_index, frame, 2]):
                z_values = [(measurement[2], 0.85), (stereo_cal[side_index, frame, 2], 0.15)]
                z_center = float(np.median([z for z, _ in z_values]))
                z_weighted = [(z, w * huber_weight(abs(z - z_center))) for z, w in z_values]
                measurement[2] = sum(z * w for z, w in z_weighted) / sum(w for _, w in z_weighted)
            fused[side_index, frame] = measurement if previous is None else 0.68 * measurement + 0.32 * previous
            previous = fused[side_index, frame]
        calibration[side] = {
            "prefix_valid_frames": idx.tolist(),
            "pico_to_hawor_local_offset_mm": (local_offset * 1000.0).tolist(),
            "stereo_surface_to_hawor_z_offset_mm": stereo_z_offset * 1000.0 if np.isfinite(stereo_z_offset) else None,
        }
    camera["PICO calibrated"] = pico_cal
    camera["Stereo surface calibrated"] = stereo_cal
    camera["Fused"] = fused
    return calibration


def series_stats(values: np.ndarray) -> dict[str, object]:
    valid = np.isfinite(values).all(1)
    data = values[valid]
    steps = np.linalg.norm(np.diff(data, axis=0), axis=1) * 1000.0
    zsteps = np.abs(np.diff(data[:, 2])) * 1000.0
    return {
        "valid_frames": int(valid.sum()),
        "step_p50_mm": float(np.percentile(steps, 50)) if len(steps) else None,
        "step_p95_mm": float(np.percentile(steps, 95)) if len(steps) else None,
        "z_step_p95_mm": float(np.percentile(zsteps, 95)) if len(zsteps) else None,
        "z_range_mm": float(np.ptp(data[:, 2]) * 1000.0) if len(data) else None,
    }


def robust_range(values: list[np.ndarray], axes: tuple[int, int]) -> tuple[tuple[float, float], tuple[float, float]]:
    merged = np.concatenate([v.reshape(-1, 3) for v in values])
    merged = merged[np.isfinite(merged).all(1)] * 1000.0
    result = []
    for axis in axes:
        lo, hi = np.percentile(merged[:, axis], [0.5, 99.5])
        span = max(hi - lo, 40.0)
        result.append((lo - .1 * span, hi + .1 * span))
    return result[0], result[1]


def map_point(point: np.ndarray, xr, yr, box):
    x0, y0, x1, y1 = box
    x = int(x0 + (point[0] - xr[0]) / max(xr[1] - xr[0], 1e-6) * (x1 - x0))
    y = int(y1 - (point[1] - yr[0]) / max(yr[1] - yr[0], 1e-6) * (y1 - y0))
    return x, y


def project(point: np.ndarray, intrinsics: np.ndarray):
    if not np.isfinite(point).all() or point[2] <= 1e-6:
        return None
    q = intrinsics @ point
    return int(round(q[0] / q[2])), int(round(q[1] / q[2]))


def render_video(raw_video: Path, output: Path, rows: list[dict], camera: dict[str, np.ndarray], fps: float):
    cap = cv2.VideoCapture(str(raw_video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {raw_video}")
    width, height = 1920, 1080
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    keys = ("PICO raw", "PICO calibrated", "HaWoR", "Stereo surface", "Fused")
    xz_range = robust_range([camera[k] for k in keys], (0, 2))
    z_all = np.concatenate([camera[k][..., 2].reshape(-1) for k in keys]) * 1000.0
    z_all = z_all[np.isfinite(z_all)]
    zlo, zhi = np.percentile(z_all, [.5, 99.5]); zspan = max(zhi-zlo, 40); zr=(zlo-.1*zspan,zhi+.1*zspan)
    n = camera["PICO raw"].shape[1]
    for frame_index in range(n):
        ok, raw = cap.read()
        if not ok:
            raise RuntimeError(f"raw video ended at frame {frame_index}")
        canvas = np.full((height, width, 3), 24, np.uint8)
        rgb = cv2.resize(raw, (900, 675), interpolation=cv2.INTER_AREA)
        K = np.asarray(rows[frame_index]["metadata"]["k"], np.float64).copy()
        K[0] *= rgb.shape[1] / float(rows[frame_index]["metadata"]["w"])
        K[1] *= rgb.shape[0] / float(rows[frame_index]["metadata"]["h"])
        for side_index in range(2):
            for key in keys:
                uv = project(camera[key][side_index, frame_index], K)
                if uv is None: continue
                cv2.circle(rgb, uv, 8 if key != "Fused" else 11, COLORS[key], -1 if key in {"PICO raw", "Fused"} else 3, cv2.LINE_AA)
        canvas[90:765, 30:930] = rgb
        cv2.rectangle(canvas, (965, 90), (1890, 765), (75,75,75), 1)
        box = (1010, 140, 1850, 720)
        cv2.rectangle(canvas, box[:2], box[2:], (75,75,75), 1)
        for side_index in range(2):
            for key in keys:
                points = camera[key][side_index, max(0,frame_index-89):frame_index+1][:, [0,2]] * 1000.0
                points = points[np.isfinite(points).all(1)]
                if len(points):
                    pix=np.asarray([map_point(p, xz_range[0], xz_range[1], box) for p in points],np.int32)
                    if len(pix)>1: cv2.polylines(canvas,[pix],False,COLORS[key],2,cv2.LINE_AA)
                    cv2.circle(canvas,tuple(pix[-1]),7,COLORS[key],-1,cv2.LINE_AA)
        graph=(90,835,1890,1040)
        cv2.rectangle(canvas,graph[:2],graph[2:],(75,75,75),1)
        for side_index in range(2):
            for key in keys:
                z=camera[key][side_index,:,2]*1000.0
                valid=np.isfinite(z[:frame_index+1]); ids=np.flatnonzero(valid)
                if len(ids)>1:
                    xx=graph[0]+ids/(n-1)*(graph[2]-graph[0]); yy=graph[3]-(z[ids]-zr[0])/(zr[1]-zr[0])*(graph[3]-graph[1])
                    cv2.polylines(canvas,[np.c_[xx,yy].astype(np.int32)],False,COLORS[key],1 if side_index else 2,cv2.LINE_AA)
        cx=int(graph[0]+frame_index/max(n-1,1)*(graph[2]-graph[0])); cv2.line(canvas,(cx,graph[1]),(cx,graph[3]),(255,255,255),1)
        rows_text=[
            (30,22,"三路手腕融合 canary：PICO运动 + HaWoR解剖位置 + Stereo表面Z",(255,255,255),34),
            (30,770,"RGB投影（点未重合时才是独立证据）",(220,220,220),22),
            (990,96,"相机侧视 X-Z：实线左手 / 同色轨迹右手",(220,220,220),22),
            (100,842,"绝对 optical-Z 全片曲线；Stereo是表面，不是关节",(220,220,220),20),
            (1530,25,f"帧 {frame_index:04d}/{n-1:04d}  时间 {frame_index/fps:5.2f}s",(255,255,255),25),
        ]
        x=35
        for key in keys:
            rows_text.append((x,58,key,COLORS[key],19)); x += 170 if key != "Stereo surface" else 220
        y=120
        for side_index, side_cn in enumerate(("左","右")):
            vals=[]
            for key in ("PICO raw","HaWoR","Stereo surface","Fused"):
                value=camera[key][side_index,frame_index,2]*1000.0
                vals.append(f"{key.split()[0]}={value:.1f}" if np.isfinite(value) else f"{key.split()[0]}=NA")
            rows_text.append((990,y,f"{side_cn}手 Z/mm  " + "  ".join(vals),(235,235,235),19)); y+=30
        canvas=put_text(canvas,rows_text)
        writer.write(canvas)
    writer.release(); cap.release()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-root", type=Path, required=True)
    parser.add_argument("--metrics-json", type=Path, required=True)
    parser.add_argument("--hawor-npz", type=Path, required=True)
    parser.add_argument("--raw-video", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--calibration-valid-frames", type=int, default=30)
    args = parser.parse_args()
    session=args.session_root.resolve(strict=True); metrics=args.metrics_json.resolve(strict=True); hawor_path=args.hawor_npz.resolve(strict=True); raw=args.raw_video.resolve(strict=True)
    out=args.output_root.resolve(); out.mkdir(parents=True,exist_ok=True)
    rows, comparison, hawor, camera, rotations=load_inputs(session,metrics,hawor_path)
    calibration=build_fusion(camera,rotations,args.calibration_valid_frames)
    joints=np.asarray(hawor["joints_3d_camera"],np.float32)
    fused_joints=joints.copy(); translations=np.full((2,joints.shape[1],3),np.nan,np.float32)
    for side in range(2):
        delta=camera["Fused"][side]-camera["HaWoR"][side]
        translations[side]=delta
        fused_joints[side]+=delta[:,None,:].astype(np.float32)
    npz_path=out/"WRIST_FUSION_DEVELOPMENT_CANARY.npz"
    np.savez_compressed(npz_path,joints_3d_camera_original=joints,joints_3d_camera_fused=fused_joints,wrist_translation_camera=translations,pico_raw=camera["PICO raw"],pico_calibrated=camera["PICO calibrated"],hawor_wrist=camera["HaWoR"],stereo_surface=camera["Stereo surface"],stereo_surface_calibrated=camera["Stereo surface calibrated"],fused_wrist=camera["Fused"],calibration_prefix_valid=np.int32(args.calibration_valid_frames),control_ground_truth=np.bool_(False),external_truth=np.bool_(False))
    video=out/"CHIPS023_PICO_HaWoR_Stereo_手腕融合Canary_全片.mp4"
    fps=float(hawor["fps"]); render_video(raw,video,rows,camera,fps)
    stats={side:{key:series_stats(camera[key][i]) for key in ("PICO raw","PICO calibrated","HaWoR","Stereo surface","Fused")} for i,side in enumerate(SIDES)}
    result={
        "schema_version":"wrist-fusion-pico-hawor-stereo-canary-v1",
        "status":"DEVELOPMENT_CANARY_GENERATED",
        "method":"PICO local-frame anatomical offset + Huber weighted PICO/HaWoR fusion + calibrated Stereo optical-Z surface constraint + causal low-pass",
        "calibration":calibration,
        "metrics":stats,
        "inputs":{"session_root":str(session),"comparison":artifact(metrics),"hawor":artifact(hawor_path),"raw_video":artifact(raw)},
        "outputs":{"npz":artifact(npz_path),"video":artifact(video)},
        "claim_limit":"Internal cross-system fusion diagnostic only. The prefix is anchored to HaWoR, Stereo is a visible surface, and PICO wrist is not externally calibrated anatomical truth. Not Robot control or physical deployment authority."
    }
    result_path=out/"RESULT.json"; result_path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    subprocess.run(["ffmpeg","-v","error","-xerror","-i",str(video),"-f","null","-"],check=True)
    print(json.dumps({"result":str(result_path),"video":str(video),"stats":stats},ensure_ascii=False,indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
