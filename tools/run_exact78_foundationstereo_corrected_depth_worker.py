#!/usr/bin/env python3
"""Generic persistent corrected-depth worker for exact78 calibrated sessions.

The model is loaded exactly once and reused across all sessions in an input
manifest.  GPU execution is fail-closed behind an already-owned central lease;
``--validate-only`` and ``--synthetic-test`` are CPU-only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
PICO_SCRIPT = PROJECT / "third_party/FoundationStereo/scripts/pico_stereo_depth.py"
CHECKPOINT = PROJECT / "assets/models/checkpoints/foundationstereo/23-51-11/model_best_bp2.pth"
CONFIG = CHECKPOINT.parent / "cfg.yaml"
CALIBRATION = PROJECT / "tasks/control/runs/20260907_stereo_object6d_formal_prepared_v1/calibration.json"
LOCAL_PYTHON = PROJECT / "assets/environments/foundationstereo-py311-v1/bin/python"
LEASE = PROJECT / "_run/GPU_LEASE.json"
FONT = Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc")
K_DEPTH = np.asarray([[320.0, 0.0, 319.5], [0.0, 320.0, 239.5], [0.0, 0.0, 1.0]], dtype=np.float64)
WIDTH, HEIGHT, SCALE = 1280, 960, 0.5
MIN_DISPARITY, Z_NEAR, Z_FAR = 0.25, 0.10, 3.0
EXPECTED_CHECKPOINT_SHA = "60e79bde9c6a00acea551625ff814fe06e5a6806e2c0c9829baee248de87c5f1"


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_sha(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    h = hashlib.sha256(value.dtype.str.encode() + b"\0")
    h.update(np.asarray(value.shape, dtype="<i8").tobytes()); h.update(value.tobytes())
    return h.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def published_artifact(current: Path, published: Path) -> dict[str, Any]:
    return {"path": str(published), "bytes": current.stat().st_size, "sha256": sha(current)}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{os.getpid()}.tmp"
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2); stream.write("\n")
        stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{os.getpid()}.tmp"
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, **arrays); stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)


def corrected_depth(disparity: np.ndarray, baseline_m: float) -> tuple[np.ndarray, np.ndarray]:
    disparity = np.asarray(disparity, dtype=np.float32)
    x = np.arange(disparity.shape[1], dtype=np.float32)[None]
    valid = np.isfinite(disparity) & (disparity > MIN_DISPARITY) & ((x - disparity) >= 0)
    depth = np.full(disparity.shape, np.nan, np.float32)
    depth[valid] = np.float32(K_DEPTH[0, 0] * baseline_m) / disparity[valid]
    valid &= (depth >= Z_NEAR) & (depth <= Z_FAR)
    depth[~valid] = np.nan
    return depth, valid


def validate_input_manifest(path: Path, rehash: bool = True) -> dict:
    payload = load_json(path)
    errors = []
    if payload.get("schema_version") != "exact78-corrected-depth-input-manifest-v1":
        errors.append("MANIFEST_SCHEMA")
    if payload.get("errors"):
        errors.append("UPSTREAM_MANIFEST_ERRORS")
    for row in payload.get("sessions", []):
        required_refs = ["raw_stereo", "selected_rgb", "camera_params", "hawor_result", "cohort"]
        # Current streaming manifests bind the two independent Mask lanes into
        # the same immutable input closure.  The dense estimator does not use
        # their pixels, but it must never publish depth under stale or
        # unauthorised session identity.
        if payload.get("join_contract") == "CURRENT_MASK_TWO_LANE_A_OR_B_EXACT_REFS":
            required_refs += [
                "hawor_agent_review",
                "role_mask_result",
                "role_mask_agent_review",
                "task_object_result",
                "task_object_agent_review",
                "task_object_manifest",
            ]
            if row.get("upstream_grades") not in ({"hawor": "A", "role_mask": "A", "task_object": "A"},):
                grades = row.get("upstream_grades", {})
                if any(grades.get(name) not in {"A", "B"} for name in ("hawor", "role_mask", "task_object")):
                    errors.append(f"{row.get('session_id')}:UPSTREAM_NOT_A_OR_B")
        for key in required_refs:
            value = row.get(key)
            p = Path(value["path"]) if value else None
            if p is None or not p.is_file():
                errors.append(f"{row.get('session_id')}:{key}:MISSING")
            elif p.stat().st_size != value["bytes"]:
                errors.append(f"{row.get('session_id')}:{key}:SIZE")
            elif rehash and sha(p) != value["sha256"]:
                errors.append(f"{row.get('session_id')}:{key}:SHA")
        if abs(float(row.get("metric_baseline_m", 0)) - 0.0637716504026918) > 1e-9:
            errors.append(f"{row.get('session_id')}:BASELINE")
    return {"status": "PASS" if not errors else "HOLD", "errors": errors, "sessions": len(payload.get("sessions", [])), "payload": payload}


def load_pico():
    spec = importlib.util.spec_from_file_location("pico_exact78_generic", PICO_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(PICO_SCRIPT)
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
    return module


def registration_arrays(row: dict, calibration: dict) -> dict[str, np.ndarray]:
    clip = Path(row["selected_rgb"]["path"]).parent
    manifest = load_json(clip / "clip_manifest.json")
    geometry = manifest["video_rectification"]
    if not geometry.get("rectified") or geometry.get("eye") != "left" or geometry.get("resolution") != [1280, 960]:
        raise RuntimeError(f"{row['session_id']}: selected RGB geometry is not pinned rectified-left 1280x960")
    selected_k = np.asarray(geometry["intrinsics"], dtype=np.float64)
    r = np.asarray(calibration["rectification_rotation_left"], dtype=np.float64)
    h_full = selected_k @ r.T @ np.linalg.inv(selected_k); h_full /= h_full[2, 2]
    half_to_full = np.asarray([[2.0, 0.0, 0.5], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0]])
    h_depth = h_full @ half_to_full; h_depth /= h_depth[2, 2]
    yy, xx = np.indices((480, 640), dtype=np.float64)
    pts = np.stack((xx, yy, np.ones_like(xx)), axis=-1) @ h_depth.T
    xy = (pts[..., :2] / pts[..., 2:]).astype(np.float32)
    in_bounds = np.isfinite(xy).all(2) & (xy[..., 0] >= 0) & (xy[..., 0] <= 1279) & (xy[..., 1] >= 0) & (xy[..., 1] <= 959)
    t = np.eye(4); t[:3, :3] = r.T
    return {
        "depth_pixel_to_selected_rgb_xy": xy,
        "depth_pixel_maps_in_selected_bounds": in_bounds,
        "H_depth_pixel_to_selected_rgb": h_depth,
        "H_selected_rgb_to_depth_pixel": np.linalg.inv(h_depth),
        "H_full_rectified_to_selected_rgb": h_full,
        "half_depth_to_full_rectified_pixel_centers": half_to_full,
        "T_stereo_rectified_camera_to_selected_camera": t,
        "selected_rgb_intrinsics": selected_k,
        "stereo_rectified_depth_formula_intrinsics": K_DEPTH,
    }


def feature_residual(a: np.ndarray, b: np.ndarray) -> dict:
    detector = cv2.SIFT_create(5000)
    ka, da = detector.detectAndCompute(a, None); kb, db = detector.detectAndCompute(b, None)
    if da is None or db is None:
        return {"matches": 0, "median_px": None, "p90_px": None}
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
    good = [x for x, y in pairs if x.distance < 0.70 * y.distance]
    distance = np.asarray([np.linalg.norm(np.asarray(ka[x.queryIdx].pt) - np.asarray(kb[x.trainIdx].pt)) for x in good])
    return {"matches": len(good), "median_px": float(np.median(distance)) if len(distance) else None, "p90_px": float(np.quantile(distance, .9)) if len(distance) else None}


def validate_registration(row: dict, arrays: dict, calibration: dict, pico) -> list[dict]:
    source = cv2.VideoCapture(row["raw_stereo"]["path"]); selected = cv2.VideoCapture(row["selected_rgb"]["path"])
    eyes, ew, _, _ = pico.load_camera_params(Path(row["camera_params"]["path"]))
    selected_k = arrays["selected_rgb_intrinsics"]
    maps = pico.make_map(eyes[0], np.asarray(calibration["rectification_rotation_left"]), 1280, 960, selected_k)
    frames = np.unique(np.linspace(0, max(0, int(row["frame_count"]) - 1), 3, dtype=int))
    results = []
    try:
        for frame in frames:
            source.set(cv2.CAP_PROP_POS_FRAMES, int(frame)); selected.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
            ok1, stereo = source.read(); ok2, rgb = selected.read()
            if not ok1 or not ok2: raise RuntimeError(f"registration decode frame {frame}")
            rect = cv2.remap(stereo[:, :ew], maps[0], maps[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            predicted = cv2.warpPerspective(rect, arrays["H_full_rectified_to_selected_rgb"], (1280, 960))
            results.append({"frame": int(frame), **feature_residual(rgb, predicted)})
    finally:
        source.release(); selected.release()
    return results


class Model:
    def __init__(self):
        import torch
        from omegaconf import OmegaConf
        sys.path.insert(0, str(PROJECT / "third_party/FoundationStereo"))
        from core.foundation_stereo import FoundationStereo
        from core.utils.utils import InputPadder
        cfg = OmegaConf.load(CONFIG); cfg.setdefault("vit_size", "vitl"); cfg["valid_iters"] = 16
        self.torch, self.InputPadder = torch, InputPadder
        self.model = FoundationStereo(cfg)
        # FoundationStereo's pinned, locally audited checkpoint predates the
        # PyTorch 2.6 ``weights_only=True`` default and contains NumPy scalar
        # metadata.  Loading it with the new default fails before inference.
        # The path is a fixed project asset (not user supplied), so preserve
        # the checkpoint's original loading semantics explicitly.
        checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        self.model.load_state_dict(checkpoint["model"])
        self.model.cuda().eval(); self.model_load_count = 1; self.inference_count = 0

    def infer(self, left: np.ndarray, right: np.ndarray, baseline: float):
        torch = self.torch
        left = cv2.resize(left, (640, 480)); right = cv2.resize(right, (640, 480))
        lt = torch.as_tensor(cv2.cvtColor(left, cv2.COLOR_BGR2RGB)).cuda().float()[None].permute(0, 3, 1, 2)
        rt = torch.as_tensor(cv2.cvtColor(right, cv2.COLOR_BGR2RGB)).cuda().float()[None].permute(0, 3, 1, 2)
        padder = self.InputPadder(lt.shape, divis_by=32, force_square=False); lt, rt = padder.pad(lt, rt)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            disparity = self.model.forward(lt, rt, iters=16, test_mode=True)
        disparity = padder.unpad(disparity.float()).cpu().numpy().reshape(480, 640).astype(np.float32)
        depth, valid = corrected_depth(disparity, baseline); self.inference_count += 1
        return disparity, depth, valid


def validate_frame(path: Path, frame: int, closure: str, baseline: float) -> dict:
    with np.load(path, allow_pickle=False) as b:
        required = {"frame_id", "disparity_px", "depth_m", "valid", "scaled_intrinsics", "input_closure_sha256"}
        if set(b.files) != required: raise RuntimeError("frame keys")
        if int(b["frame_id"]) != frame or str(b["input_closure_sha256"]) != closure: raise RuntimeError("frame identity closure")
        disparity, depth, valid, k = b["disparity_px"], b["depth_m"], b["valid"], b["scaled_intrinsics"]
    if disparity.shape != (480, 640) or depth.shape != (480, 640) or not np.array_equal(k, K_DEPTH): raise RuntimeError("frame geometry")
    expected, expected_valid = corrected_depth(disparity, baseline)
    if not np.array_equal(valid, expected_valid) or not np.allclose(depth[valid], expected[valid], atol=1e-6): raise RuntimeError("corrected formula")
    return {"valid_pixels": int(valid.sum()), "depth_min_m": float(np.nanmin(depth)), "depth_max_m": float(np.nanmax(depth))}


def review_video(row: dict, partial: Path) -> dict:
    from PIL import Image, ImageDraw, ImageFont
    cap = cv2.VideoCapture(row["selected_rgb"]["path"]); fps = float(cap.get(cv2.CAP_PROP_FPS))
    tmp = partial / f".{row['session_id']}.review.tmp.mp4"; final = partial / f"{row['session_id']}_校正公制深度_逐帧.mp4"
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 480))
    font = ImageFont.truetype(str(FONT), 19) if FONT.is_file() else None
    try:
        for frame in range(int(row["frame_count"])):
            ok, rgb = cap.read()
            if not ok: raise RuntimeError("selected RGB ended")
            with np.load(partial / "frames" / f"{frame:06d}.npz", allow_pickle=False) as b: depth, valid = b["depth_m"], b["valid"]
            scalar = np.zeros(depth.shape, np.uint8); scalar[valid] = np.rint(255 * (1 - np.clip((depth[valid] - Z_NEAR) / (Z_FAR - Z_NEAR), 0, 1))).astype(np.uint8)
            colour = cv2.applyColorMap(scalar, cv2.COLORMAP_TURBO); colour[~valid] = 0
            canvas = np.concatenate((cv2.resize(rgb, (640, 480)), colour), axis=1)
            if font:
                image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)); draw = ImageDraw.Draw(image)
                draw.text((12, 8), f"帧 {frame:04d}｜左：RGB　右：双目公制深度", font=font, fill=(255,255,255)); draw.text((652, 36), "Z=320×基线/视差；仅供 Object6D 候选", font=font, fill=(255,255,255)); canvas = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
            writer.write(canvas)
    finally:
        writer.release(); cap.release()
    os.replace(tmp, final)
    check = cv2.VideoCapture(str(final)); frames = int(round(check.get(cv2.CAP_PROP_FRAME_COUNT))); check.release()
    if frames != int(row["frame_count"]): raise RuntimeError("review frame count")
    return {**artifact(final), "frames": frames, "fps": fps}


def process(row: dict, model: Model, pico, calibration: dict) -> dict:
    destination = Path(row["destination"]); partial = Path(row["atomic_temp_destination"])
    if destination.exists():
        result = destination / "RESULT.json"
        if result.is_file() and load_json(result).get("input_closure_sha256") == row["input_closure_sha256"]: return {"session_id": row["session_id"], "status": "RESUME_SKIP_VERIFIED_TERMINAL", "result": artifact(result)}
        raise RuntimeError(f"no-clobber final exists: {destination}")
    run_spec_path = partial / "RUN_SPEC.json"
    if partial.exists():
        if not run_spec_path.is_file() or load_json(run_spec_path).get("input_closure_sha256") != row["input_closure_sha256"]:
            raise RuntimeError(f"partial input closure mismatch; no-clobber hold: {partial}")
    else:
        partial.mkdir(parents=True)
        atomic_json(run_spec_path, {
            "schema_version": "exact78-corrected-depth-run-spec-v1",
            "created_at": now(), "session_id": row["session_id"],
            "input_closure_sha256": row["input_closure_sha256"],
            "depth_formula_intrinsics": K_DEPTH.tolist(),
            "model_load_policy": "one persistent model shared across manifest sessions",
            "publish_policy": "validated per-frame atomic NPZ; final session directory atomic rename",
            "upstream_exact_refs": {
                key: row[key]
                for key in (
                    "hawor_result", "hawor_agent_review", "role_mask_result",
                    "role_mask_agent_review", "task_object_result",
                    "task_object_agent_review", "task_object_manifest",
                )
                if key in row
            },
        })
    (partial / "frames").mkdir(exist_ok=True)
    arrays = registration_arrays(row, calibration); atomic_npz(partial / "REGISTRATION_AUTHORITY.npz", **arrays)
    registration_rows = validate_registration(row, arrays, calibration, pico)
    registration_pass = all(r["matches"] >= 100 and r["median_px"] <= .5 and r["p90_px"] <= 1.5 for r in registration_rows)
    atomic_json(partial / "REGISTRATION_RESULT.json", {"schema_version": "exact78-depth-registration-v1", "status": "PASS" if registration_pass else "HOLD", "input_closure_sha256": row["input_closure_sha256"], "depth_formula_intrinsics": K_DEPTH.tolist(), "feature_rows": registration_rows, "authority": published_artifact(partial / "REGISTRATION_AUTHORITY.npz", destination / "REGISTRATION_AUTHORITY.npz")})
    eyes, ew, eh, baseline = pico.load_camera_params(Path(row["camera_params"]["path"])); k0 = pico.virtual_intrinsics(1280, 960, 90)
    maps = [pico.make_map(eyes[i], np.asarray(calibration[f"rectification_rotation_{'left' if i == 0 else 'right'}"]), 1280, 960, k0) for i in range(2)]
    cap = cv2.VideoCapture(row["raw_stereo"]["path"]); frame_rows = []
    try:
        for frame in range(int(row["frame_count"])):
            ok, stereo = cap.read()
            if not ok: raise RuntimeError(f"stereo ended at {frame}")
            target = partial / "frames" / f"{frame:06d}.npz"
            if target.is_file():
                try: stats = validate_frame(target, frame, row["input_closure_sha256"], baseline); frame_rows.append({"frame_id": frame, "relative_path": f"frames/{frame:06d}.npz", "bytes": target.stat().st_size, "sha256": sha(target), **stats}); continue
                except Exception: os.replace(target, target.with_name(f".{target.name}.invalid-{int(time.time())}"))
            left, right = pico.remap_pair(stereo, ew, maps); disparity, depth, valid = model.infer(left, right, baseline)
            atomic_npz(target, frame_id=np.asarray(frame), disparity_px=disparity, depth_m=depth, valid=valid, scaled_intrinsics=K_DEPTH, input_closure_sha256=np.asarray(row["input_closure_sha256"]))
            stats = validate_frame(target, frame, row["input_closure_sha256"], baseline); frame_rows.append({"frame_id": frame, "relative_path": f"frames/{frame:06d}.npz", "bytes": target.stat().st_size, "sha256": sha(target), **stats, "disparity_array_sha256": array_sha(disparity)})
    finally: cap.release()
    atomic_json(partial / "FRAME_MANIFEST.json", {"schema_version": "exact78-corrected-depth-frame-manifest-v1", "session_id": row["session_id"], "input_closure_sha256": row["input_closure_sha256"], "frames": frame_rows})
    review = review_video(row, partial)
    passed = registration_pass and len(frame_rows) == int(row["frame_count"])
    review["path"] = str(destination / Path(review["path"]).name)
    result = {"schema_version": "exact78-corrected-dense-depth-result-v2", "status": "PASS_CORRECTED_DENSE_METRIC_DEPTH_FULLSESSION" if passed else "HOLD_REGISTRATION", "grade": "B" if passed else "C", "consumption_authorized": passed, "authorized_scopes": ["VISUAL_OBJECT6D_CANDIDATE_INPUT"] if passed else [], "session_id": row["session_id"], "task": row["task"], "input_closure_sha256": row["input_closure_sha256"], "correction_contract": {"formula": "Z=fx*B/d", "fx_px": 320.0, "baseline_m": baseline, "depth_grid": [640,480]}, "inputs": {key: row[key] for key in ("cohort", "raw_stereo", "selected_rgb", "camera_params", "hawor_result", "hawor_agent_review", "role_mask_result", "role_mask_agent_review", "task_object_result", "task_object_agent_review", "task_object_manifest") if key in row}, "frame_manifest": published_artifact(partial / "FRAME_MANIFEST.json", destination / "FRAME_MANIFEST.json"), "registration_authority": published_artifact(partial / "REGISTRATION_AUTHORITY.npz", destination / "REGISTRATION_AUTHORITY.npz"), "registration_result": published_artifact(partial / "REGISTRATION_RESULT.json", destination / "REGISTRATION_RESULT.json"), "review_mp4": review, "robot_contact_authorized": False, "claim_limit": "Corrected stereo metric depth only. Role/task-object Mask refs bind session identity but do not alter depth pixels; no Object6D/contact/Robot authority."}
    atomic_json(partial / "RESULT.json", result)
    review_receipt = {
        "schema_version": "exact78-corrected-depth-agent-review-v1",
        "created_at": now(),
        "session_id": row["session_id"],
        "grade": "B" if passed else "C",
        "downstream_authorized": passed,
        "checks": {
            "registration_gate": registration_pass,
            "frame_count_exact": len(frame_rows) == int(row["frame_count"]),
            "corrected_metric_formula_revalidated_per_frame": len(frame_rows) == int(row["frame_count"]),
            "input_sha_closure_bound": True,
            "review_video_full_decode": review["frames"] == int(row["frame_count"]),
        },
        "review_video": review,
        "claim_limit": "Automatic numeric/registration review; Grade B authorizes observed-only visual Object6D candidate input, not contact or Robot truth.",
    }
    atomic_json(partial / "AGENT_REVIEW.json", review_receipt)
    checksum_lines = []
    for item in sorted(p for p in partial.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt"):
        checksum_lines.append(f"{sha(item)}  {item.relative_to(partial)}")
    checksum_path = partial / "SHA256SUMS.txt"
    checksum_path.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    if not passed: return {"session_id": row["session_id"], "status": result["status"], "partial": str(partial)}
    os.replace(partial, destination)
    return {"session_id": row["session_id"], "status": result["status"], "result": artifact(destination / "RESULT.json")}


def synthetic_test(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True); frame = output / "frame.npz"
    baseline = 0.0637716504026918; disparity = np.full((480,640), 40, np.float32); depth, valid = corrected_depth(disparity, baseline); closure = "a" * 64
    atomic_npz(frame, frame_id=np.asarray(0), disparity_px=disparity, depth_m=depth, valid=valid, scaled_intrinsics=K_DEPTH, input_closure_sha256=np.asarray(closure))
    first_sha = sha(frame); stats = validate_frame(frame, 0, closure, baseline); second_sha = sha(frame)
    expected = 320 * baseline / 40
    result = {"schema_version": "exact78-depth-worker-synthetic-test-v1", "status": "PASS" if abs(float(np.nanmedian(depth))-expected) < 1e-6 and first_sha == second_sha else "HOLD", "gpu_used": False, "expected_depth_m": expected, "observed_depth_m": float(np.nanmedian(depth)), "atomic_frame_sha256": first_sha, "resume_validation_did_not_mutate": first_sha == second_sha, "stats": stats}
    atomic_json(output / "RESULT.json", result); return result


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--manifest", type=Path); ap.add_argument("--validate-only", action="store_true"); ap.add_argument("--execute-under-lease", action="store_true"); ap.add_argument("--lease-holder", default="exact78-object6d-depth-canary-v1"); ap.add_argument("--receipt", type=Path); ap.add_argument("--synthetic-test", type=Path)
    args = ap.parse_args()
    if args.synthetic_test:
        print(json.dumps(synthetic_test(args.synthetic_test), ensure_ascii=False)); return 0
    if not args.manifest: raise SystemExit("--manifest is required")
    check = validate_input_manifest(args.manifest, rehash=True)
    receipt = {"schema_version": "exact78-depth-worker-validate-v1", "created_at": now(), "mode": "CPU_VALIDATE_ONLY" if args.validate_only else "GPU_EXECUTE", "worker": artifact(SCRIPT), "manifest": artifact(args.manifest), "checkpoint": artifact(CHECKPOINT), "checkpoint_pin_pass": sha(CHECKPOINT) == EXPECTED_CHECKPOINT_SHA, "input_validation": {k:v for k,v in check.items() if k != "payload"}, "model_load_count": 0, "gpu_started": False}
    if args.receipt: atomic_json(args.receipt, receipt)
    if args.validate_only:
        print(json.dumps(receipt, ensure_ascii=False)); return 0 if check["status"] == "PASS" else 2
    if not args.execute_under_lease: raise SystemExit("execution requires --execute-under-lease")
    if check["status"] != "PASS": raise SystemExit("input validation HOLD")
    lease = load_json(LEASE)
    if lease.get("status") != "ACQUIRED" or lease.get("holder") != args.lease_holder: raise SystemExit("central lease not owned by requested holder")
    if Path(sys.executable).resolve() != LOCAL_PYTHON.resolve(): raise SystemExit(f"must execute with {LOCAL_PYTHON}")
    pico = load_pico(); calibration = load_json(CALIBRATION); model = Model(); results = [process(row, model, pico, calibration) for row in check["payload"]["sessions"]]
    print(json.dumps({"status": "COMPLETE", "model_load_count": model.model_load_count, "inference_count": model.inference_count, "sessions": results}, ensure_ascii=False)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
