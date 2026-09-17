#!/usr/bin/env python3
"""Run one same-session PICO FoundationStereo development canary.

The canary estimates rectification from the selected session itself.  It never
uses the fixed exact78 rotation and never grants external depth accuracy or a
full H3 batch authority.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
PICO = ROOT / "vendor/FoundationStereo/scripts/pico_stereo_depth.py"
PYTHON = ROOT / "_run/current/environments/foundationstereo-py311-v1/bin/python"
CHECKPOINT = ROOT / "assets/models/checkpoints/foundationstereo/23-51-11/model_best_bp2.pth"
EXPECTED_CHECKPOINT_SHA = "60e79bde9c6a00acea551625ff814fe06e5a6806e2c0c9829baee248de87c5f1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def published_ref(current: Path, published: Path) -> dict[str, Any]:
    """Hash a staging file while recording its post-rename immutable path."""
    return {
        "path": str(published.resolve()),
        "bytes": current.stat().st_size,
        "sha256": sha256(current),
    }


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--cohort-total-frames", type=int, default=106868)
    args = parser.parse_args()
    session = args.session_root.resolve(strict=True)
    output = args.output_root.resolve()
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"fresh immutable output required: {output}")
    if not PYTHON.is_file() or not PICO.is_file():
        raise RuntimeError("pinned FoundationStereo runtime is incomplete")
    if sha256(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA:
        raise RuntimeError("FoundationStereo checkpoint SHA drift")
    camera = session / "camera_params.json"
    stereo_candidates = sorted((session / "source_stereo").glob("CameraRecord_*_stereo.mp4"))
    if not camera.is_file() or len(stereo_candidates) != 1:
        raise RuntimeError("same-session camera/stereo identity is incomplete")

    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    artifacts = temporary / "artifacts"
    artifacts.mkdir(parents=True)
    command = [
        str(PYTHON), str(PICO), "--clip-dir", str(session),
        "--frame", str(args.frame), "--output-dir", str(artifacts),
        "--calibration", str(temporary / "same_session_rectification.json"),
        "--calibration-frames", "8", "--checkpoint", str(CHECKPOINT),
        "--width", "1280", "--height", "960", "--scale", "0.5",
        "--valid-iters", "16", "--z-near", "0.10", "--z-far", "5.0",
    ]
    started = time.monotonic()
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    wall = time.monotonic() - started
    if completed.returncode:
        shutil.rmtree(temporary, ignore_errors=True)
        raise RuntimeError(f"PICO FoundationStereo failed: {completed.stderr[-4000:]}")
    manifest_path = artifacts / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # The underlying tool ran in a staging directory.  Publish only the final
    # path so the manifest does not retain a dangling temporary reference.
    manifest["calibration_file"] = str((output / "same_session_rectification.json").resolve())
    atomic_json(manifest_path, manifest)
    quality = manifest.get("rectification_quality", {})
    inference = manifest.get("inference", {})
    matches = int(quality.get("matches") or 0)
    median = quality.get("median_vertical_error_px")
    p90 = quality.get("p90_vertical_error_px")
    valid_pixels = int(inference.get("valid_depth_pixels") or 0)
    quality_pass = bool(
        manifest.get("model_ran") is True
        and matches >= 20 and median is not None and float(median) <= 2.0
        and p90 is not None and float(p90) <= 5.0
        and valid_pixels > 0
    )
    projected_hours = wall * max(0, args.cohort_total_frames) / 3600.0
    result = {
        "schema_version": "sensor-h3-foundationstereo-canary-v71-v1",
        "status": "PASS_DEVELOPMENT_CANARY" if quality_pass else "FAILED_QUALITY_C",
        "artifact_revision": "R7_1",
        "validity": "VALID_FOR_PINNED_REVISION",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "session_id": session.name,
        "source_frame": args.frame,
        "same_session_rectification": True,
        "fixed_exact78_rotation_used": False,
        "rectification_quality": quality,
        "valid_depth_pixels": valid_pixels,
        "single_frame_wall_seconds_including_calibration_and_model_load": wall,
        "cohort_total_frames": args.cohort_total_frames,
        "conservative_projected_gpu_hours": projected_hours,
        "batch_budget_hours": 12.0,
        "batch_budget_exceeded_by_upper_bound": projected_hours > 12.0,
        "inputs": {
            "camera_params": ref(camera),
            "source_stereo": ref(stereo_candidates[0]),
            "checkpoint": ref(CHECKPOINT),
        },
        "artifacts": {
            "manifest": published_ref(manifest_path, output / "artifacts/manifest.json"),
            "calibration": published_ref(temporary / "same_session_rectification.json", output / "same_session_rectification.json"),
            "rectification_check": published_ref(artifacts / "rectification_check.png", output / "artifacts/rectification_check.png"),
            "depth_visualization": published_ref(artifacts / "depth_vis.png", output / "artifacts/depth_vis.png"),
            "point_cloud": published_ref(artifacts / "cloud.ply", output / "artifacts/cloud.ply"),
        },
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
        "claim_limit": "One-frame same-session development canary. Visible-surface optical-Z only; not full H3 authority, anatomical wrist depth, external metric truth or physical accuracy.",
    }
    atomic_json(temporary / "RESULT.json", result)
    os.rename(temporary, output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    # Quality C is a valid finite research result, not a runtime failure.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
