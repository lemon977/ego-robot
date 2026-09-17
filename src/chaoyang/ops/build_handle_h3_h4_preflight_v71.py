#!/usr/bin/env python3
"""Build CPU-only H3 Stereo and H4 sensor-mask preflight ledgers.

The tool deliberately performs no image decoding, rectification, model load or
GPU work.  It verifies the published same-session contracts and records the
live GPU owner as a resource blocker.  A later immutable successor may consume
READY rows; this preflight never grants Depth or Mask authority.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


STAGES = {"h3", "h4"}
SENSOR_ROLE_IDS = (
    "left_gloved_hand", "right_gloved_hand", "left_forearm", "right_forearm",
    "left_controller", "right_controller",
)
ROLE_PROMPTS = {
    "left_gloved_hand": "left white instrumented glove and all five gloved fingers; exclude forearm, Controller and object",
    "right_gloved_hand": "right white instrumented glove and all five gloved fingers; exclude forearm, Controller and object",
    "left_forearm": "visible left forearm or sleeve proximal to the glove cuff; exclude glove and Controller",
    "right_forearm": "visible right forearm or sleeve proximal to the glove cuff; exclude glove and Controller",
    "left_controller": "left PICO hand Controller rigid body; exclude white glove, forearm and task object",
    "right_controller": "right PICO hand Controller rigid body; exclude white glove, forearm and task object",
}


class PreflightError(RuntimeError):
    """Published input cannot satisfy the CPU contract."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _evidence(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise PreflightError(f"expected JSON object: {path}")
    return value


def _finite_matrix(value: Any, shape: tuple[int, int], name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != shape or not np.isfinite(matrix).all():
        raise PreflightError(f"invalid {name}: shape={matrix.shape}")
    return matrix


def _camera_center(extrinsic: np.ndarray) -> np.ndarray:
    return -(extrinsic[:3, :3].T @ extrinsic[:3, 3])


def stereo_preflight(row: dict[str, Any]) -> dict[str, Any]:
    eligibility = row.get("eligibility", {})
    stereo = eligibility.get("source_stereo", {})
    camera = eligibility.get("camera_contract", {})
    stereo_artifact = stereo.get("artifact", {})
    stereo_path = Path(str(stereo_artifact.get("path", "")))
    camera_path = Path(str(camera.get("artifact", {}).get("path", "")))
    stereo_eligible = stereo.get("eligible_for_depth_preflight", stereo.get("eligible"))
    if not stereo_eligible or not stereo_path.is_file():
        raise PreflightError("same-session SBS artifact is not eligible")
    if not camera.get("eligible") or not camera_path.is_file():
        raise PreflightError("same-session camera contract is not eligible")
    probe = stereo.get("probe", {})
    frame_count = int(row["frame_count"])
    if (int(probe.get("width", -1)), int(probe.get("height", -1))) != (4096, 1536):
        raise PreflightError("SBS must be 4096x1536")
    if int(probe.get("nb_read_frames", -1)) != frame_count:
        raise PreflightError("SBS frame identity mismatch")
    params = _json(camera_path)
    eye_w, eye_h = int(params.get("width", -1)), int(params.get("height", -1))
    if (eye_w, eye_h) != (2048, 1536) or int(probe["width"]) != 2 * eye_w:
        raise PreflightError("camera per-eye geometry does not close with SBS")
    for side in ("left", "right"):
        eye = params.get(side, {})
        intrinsics = eye.get("intrinsics", {})
        values = [intrinsics.get(k) for k in ("fx", "fy", "cx", "cy")]
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
            raise PreflightError(f"non-finite {side} intrinsics")
        fx, fy, cx, cy = map(float, values)
        if fx <= 0 or fy <= 0 or not (0 <= cx < eye_w and 0 <= cy < eye_h):
            raise PreflightError(f"invalid {side} intrinsics domain")
        distortion = eye.get("distortion", {})
        if distortion.get("model") != "equiDis62" or len(distortion.get("coeffs", [])) != 8:
            raise PreflightError(f"unsupported {side} distortion contract")
    extrinsics = params.get("extrinsics", {})
    left = _finite_matrix(extrinsics.get("left"), (4, 4), "left extrinsic")
    right = _finite_matrix(extrinsics.get("right"), (4, 4), "right extrinsic")
    expected_tail = np.array([0.0, 0.0, 0.0, 1.0])
    if not np.allclose(left[3], expected_tail) or not np.allclose(right[3], expected_tail):
        raise PreflightError("extrinsic homogeneous row mismatch")
    baseline_m = float(np.linalg.norm(_camera_center(left) - _camera_center(right)))
    if not (0.02 <= baseline_m <= 0.15):
        raise PreflightError(f"implausible same-session baseline: {baseline_m}")
    selected_key = params.get("selected_calibration_key")
    if selected_key not in ("left", "right"):
        raise PreflightError("selected calibration identity absent")
    return {
        "source_stereo": {
            "path": str(stereo_path.resolve()), "bytes": stereo_path.stat().st_size,
            "width": 4096, "height": 1536, "per_eye_width": eye_w,
            "frame_count": frame_count, "fps": probe.get("avg_frame_rate"),
        },
        "camera_contract": {
            "path": str(camera_path.resolve()), "bytes": camera_path.stat().st_size,
            "selected_calibration_key": selected_key,
            "intrinsics_authority": camera.get("intrinsics_authority"),
            "image_domain_mode": camera.get("image_domain_mode"),
            "source_rectified": camera.get("rectified"), "baseline_m": baseline_m,
        },
        "rectification_contract": {
            "source_distortion": "same-session equiDis62 per eye",
            "rotation_source": "derive per session from camera_params extrinsics",
            "fixed_exact78_rotation_allowed": False,
            "foundationstereo_input": "future common rectified pinhole domain",
        },
        "future_depth_contract": {
            "quantity": "visible-surface optical-Z_m", "formula": "Z=f_rectified*baseline/disparity_px",
            "depth_confidence_present": False,
            "controller_wrist_overwrite_allowed": False,
        },
    }


def mask_preflight(row: dict[str, Any]) -> dict[str, Any]:
    visual = row.get("eligibility", {}).get("visual_rgb", {})
    artifact = visual.get("artifact", {})
    path = Path(str(artifact.get("path", "")))
    if not visual.get("eligible") or not path.is_file():
        raise PreflightError("published 1280x960 visual input is not eligible")
    frame_count = int(row["frame_count"])
    target = Path(str(row["target"]))
    clip = _json(target / "clip_manifest.json")
    video = clip.get("video", {})
    if (video.get("width"), video.get("height"), video.get("frame_count")) != (1280, 960, frame_count):
        raise PreflightError("visual frame geometry/identity mismatch")
    task = row.get("task")
    if task == "potato_chips":
        object_contract = {
            "id_pattern": "chips_instance_<session_seed_id>",
            "prompt": "one specific task-relevant potato-chips package instance identified by the frozen session seed",
            "union_allowed": False,
            "identity_rule": "each physical package has an independent persistent ID through occlusion/re-entry",
        }
    elif task == "playing_cards":
        object_contract = {
            "id_pattern": "playing_cards_instance_<session_seed_id>",
            "prompt": "one specific task-relevant card or deck instance identified by the frozen session seed",
            "union_allowed": False,
            "identity_rule": "do not merge separate cards/decks; preserve the frozen target identity after re-entry",
        }
    else:
        raise PreflightError(f"unsupported task: {task}")
    return {
        "visual_input": {"path": str(path.resolve()), "bytes": path.stat().st_size, "width": 1280, "height": 960, "frame_count": frame_count},
        "role_ids": list(SENSOR_ROLE_IDS),
        "object_contract": object_contract,
        "seed_contract": {
            "source": "per-session first clear visibility plus Controller/MANUS geometry prompts",
            "status": "TO_BE_FROZEN_BY_GPU_CANARY",
            "text_prompt_alone_grants_identity": False,
        },
        "regression_cases": [
            "WHITE_GLOVE_LOW_CONTRAST", "LEFT_RIGHT_CROSSING", "GLOVE_CONTROLLER_CONTACT",
            "FOREARM_CUFF_BOUNDARY", "CONTROLLER_OBJECT_OVERLAP", "OBJECT_HAND_OCCLUSION",
            "OFFSCREEN_REENTRY", "MULTI_OBJECT_INSTANCE_SEPARATION",
        ],
    }


def _gpu_blocker(status_path: Path) -> dict[str, Any]:
    status = _json(status_path)
    active = [task for task in status.get("active_tasks", []) if task.get("gpu_id") is not None]
    return {
        "status": "BLOCKED_RESOURCE" if active else "GPU_NOT_CLAIMED_BY_CPU_PREFLIGHT",
        "observed_at": status.get("generated_at"),
        "active_gpu_tasks": [
            {key: item.get(key) for key in ("task_id", "session", "pid", "gpu_id", "phase", "heartbeat_at")}
            for item in active
        ],
        "claim": "CPU preflight does not acquire or compete for the GPU lease.",
    }


def build(stage: str, h0_path: Path, status_path: Path, output: Path) -> dict[str, Any]:
    if stage not in STAGES:
        raise ValueError(stage)
    if output.exists():
        raise FileExistsError(f"immutable output exists: {output}")
    h0 = _json(h0_path)
    rows = h0.get("rows", [])
    if h0.get("row_count") != len(rows):
        raise PreflightError("H0 row_count mismatch")
    gpu = _gpu_blocker(status_path)
    temp = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True)
    terminals: list[dict[str, Any]] = []
    try:
        for row in rows:
            common = {
                "session_id": row["session_id"], "dataset_id": row["dataset_id"],
                "task": row["task"], "frame_count": row.get("frame_count"),
            }
            if row.get("admission") == "BLOCKED_SOURCE":
                terminals.append({**common, "status": "BLOCKED_SOURCE", "cpu_preflight": "BLOCKED", "primary_blocker": "H0_BLOCKED_SOURCE", "contract": None})
                continue
            try:
                contract = stereo_preflight(row) if stage == "h3" else mask_preflight(row)
                # This attempt intentionally ends at the resource boundary.  It is
                # resumable by a later immutable GPU attempt.
                terminals.append({**common, "status": "BLOCKED_RESOURCE", "cpu_preflight": "PASSED", "primary_blocker": "GPU_LEASE_OWNED_BY_CLEAN" if gpu["status"] == "BLOCKED_RESOURCE" else "GPU_EXECUTION_NOT_REQUESTED", "contract": contract})
            except PreflightError as exc:
                terminals.append({**common, "status": "BLOCKED_PREREQ", "cpu_preflight": "FAILED", "primary_blocker": str(exc), "contract": None})
            except Exception as exc:
                terminals.append({**common, "status": "FAILED_RUNTIME_FINAL", "cpu_preflight": "FAILED", "primary_blocker": f"{type(exc).__name__}: {exc}", "contract": None})
        counts = Counter(row["status"] for row in terminals)
        preflight_counts = Counter(row["cpu_preflight"] for row in terminals)
        ledger_name = "SENSOR_DEPTH_LEDGER.json" if stage == "h3" else "SENSOR_MASK_LEDGER.json"
        schema_version = "handle-sensor-depth-preflight-ledger-v7.1" if stage == "h3" else "handle-sensor-mask-preflight-ledger-v7.1"
        claim_limit = (
            "CPU contract only. No images were rectified and no FoundationStereo depth was generated; future Z is visible-surface optical-Z, not external metric or wrist truth."
            if stage == "h3" else
            "CPU sensor-role mask contract only. No SAM3.1 inference ran and no pixel Mask authority is granted."
        )
        ledger = {
            "schema_version": schema_version, "artifact_revision": "R7_0",
            "validity": "VALID_FOR_PINNED_REVISION", "generated_at": _now(), "stage": stage,
            "source_h0": _evidence(h0_path), "resource_observation": gpu,
            "row_count": len(terminals),
            "summary": {"by_status": dict(sorted(counts.items())), "by_cpu_preflight": dict(sorted(preflight_counts.items())), "total_frames": sum(int(row.get("frame_count") or 0) for row in terminals)},
            "rows": terminals, "claim_limit": claim_limit,
        }
        if stage == "h4":
            ledger["sensor_role_mask_contract"] = {
                "schema": "SENSOR_ROLE_MASK_V1", "role_prompts": ROLE_PROMPTS,
                "visible_pixel_ownership": "one visible semantic/instance owner per pixel",
                "roles_must_remain_separate": True, "controllers_are_not_gloves": True,
                "forearms_are_not_gloves": True, "chips_union_allowed": False,
                "reentry_identity_required": True,
            }
        ledger_path = temp / ledger_name
        ledger_path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + "\n")
        with (temp / ledger_name.replace(".json", ".csv")).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["session_id", "dataset_id", "task", "frame_count", "status", "cpu_preflight", "primary_blocker"])
            writer.writeheader()
            for row in terminals:
                writer.writerow({key: row.get(key) for key in writer.fieldnames})
        result = {
            "schema_version": "handle-h3-h4-preflight-result-v7.1", "stage": stage,
            "status": "BLOCKED_RESOURCE", "artifact_revision": "R7_0",
            "validity": "VALID_FOR_PINNED_REVISION", "generated_at": _now(),
            "row_count": len(terminals), "terminal_count": len(terminals),
            "summary": ledger["summary"], "resource_observation": gpu,
            "resume_condition": "Clean releases the central GPU lease; create a fresh immutable GPU attempt.",
            "claim_limit": claim_limit,
        }
        (temp / "RESULT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        os.rename(temp, output)
        return ledger
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def _args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=sorted(STAGES), required=True)
    parser.add_argument("--h0", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = _args(argv)
    ledger = build(args.stage, args.h0.resolve(), args.status.resolve(), args.output.resolve())
    print(json.dumps({"stage": args.stage, "summary": ledger["summary"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
