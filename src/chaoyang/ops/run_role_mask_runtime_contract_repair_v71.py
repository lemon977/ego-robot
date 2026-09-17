#!/usr/bin/env python3
from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


"""Bounded Poker042 Role Mask rerun with the verified 25 FPS contract."""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chaoyang.pipeline.causal_modal_mask_gpu_adapter_v71 import FileGpuLease
from chaoyang.ops import run_exact78_fullsession_role_mask as legacy
from chaoyang.governance.common import atomic_json, artifact_ref


SOURCE_CONFIG = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260909_exact78_current_baseline_batch_v1/mask_role_bounded_v2_1_configs/play_cards_0901_042.json"
OLD_RESULT = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260909_exact78_current_baseline_batch_v1/mask_role_removal_bounded_v2_1_failures/poker/play_cards_0901_042/RESULT.json"
DEFAULT_OUT = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/successors/role_mask_runtime_contract_repair_poker042/R7_1"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def remap_archived_task_paths(value: Any) -> Any:
    old = str(ROOT / "tasks") + "/"
    new = str(ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks") + "/"
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, list):
        return [remap_archived_task_paths(item) for item in value]
    if isinstance(value, dict):
        return {key: remap_archived_task_paths(item) for key, item in value.items()}
    return value


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def corrected_config(output_root: Path) -> Path:
    source = remap_archived_task_paths(load(SOURCE_CONFIG))
    if float(source["fps"]) != 30.0 or int(source["frame_count"]) != 196:
        raise RuntimeError("unexpected predecessor contract; refusing generic mutation")
    video = Path(source["raw_video"]["path"])
    capture = cv2.VideoCapture(str(video))
    count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    with np.load(source["hawor_npz"]["path"], allow_pickle=False) as archive:
        hawor_count = int(archive["joints_2d"].shape[1])
        hawor_fps = float(archive["fps"])
    if (count, hawor_count) != (196, 196) or abs(fps - 25.0) > 0.01 or abs(hawor_fps - 25.0) > 0.01:
        raise RuntimeError(f"25 FPS closure failed: video={count}@{fps}, hawor={hawor_count}@{hawor_fps}")
    source["fps"] = 25.0
    path = output_root / "input_snapshot" / "play_cards_0901_042_role_mask_config_25fps.json"
    if path.exists():
        if load(path) != source:
            raise RuntimeError("immutable corrected config conflict")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, source)
    return path


def validated(config_path: Path, output: Path) -> dict[str, Any]:
    config = load(config_path)
    if output.exists():
        raise FileExistsError(output)
    count = int(config["frame_count"])
    raw_root = Path(config["raw_all_data"]).resolve(strict=True)
    frame_paths = [(raw_root / f"{frame_id:05d}" / "rgb.png").resolve(strict=True) for frame_id in range(count)]
    image = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    if image is None or image.shape[:2] != (960, 1280):
        raise RuntimeError("raw frame domain mismatch")
    raw_video = legacy.canary.verify_ref(config["raw_video"], "raw_video")
    hawor = legacy.canary.verify_ref(config["hawor_npz"], "hawor_npz")
    capture = cv2.VideoCapture(str(raw_video))
    video_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    video_fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    if video_count <= 0 or not np.isfinite(video_fps) or video_fps <= 0:
        raise RuntimeError("raw video metadata unreadable")
    with np.load(hawor, allow_pickle=False) as archive:
        hawor_fps = float(archive["fps"])
        if archive["joints_2d"].shape != (2, count, 21, 2):
            raise RuntimeError("HaWoR frame identity mismatch")
        if archive["anatomical_side_names"].astype(str).tolist() != ["left", "right"]:
            raise RuntimeError("HaWoR anatomical side mismatch")
    return {
        "config": config, "config_path": config_path.resolve(), "raw_root": raw_root,
        "raw_video": raw_video, "hawor_path": hawor, "frame_paths": frame_paths,
        "indices": list(range(count)), "anchor_slot": int(config["anchor_frame"]),
        # The lossless preprocess/all_data sequence is the model input.  Older
        # 0901 MP4 containers legitimately declare 25 FPS while the frozen
        # model-input contract is 30 FPS.  Record that difference; never drop,
        # duplicate or reject model-input frames because of container metadata.
        "fps": float(config["fps"]), "height": 960, "width": 1280,
        "raw_video_metadata_diagnostic": {
            "decoded_frame_count": video_count,
            "declared_fps": video_fps,
            "model_input_rgb_frame_count": count,
            "model_input_fps": float(config["fps"]),
            "hawor_declared_fps": hawor_fps,
            "count_matches_model_input": video_count == count,
            "fps_matches_model_input": abs(video_fps - float(config["fps"])) <= 0.01,
            "hawor_fps_matches_model_input": abs(hawor_fps - float(config["fps"])) <= 0.01,
            "authority": "MODEL_INPUT_IS_PINNED_PREPROCESS_RGB_SEQUENCE",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--config", type=Path, help="already-consistent regression config")
    parser.add_argument("--confirm-config-sha")
    parser.add_argument("--predecessor-result", type=Path)
    parser.add_argument("--attempt-id", default="attempt_0001")
    parser.add_argument("--artifact-revision", default="R7_1")
    parser.add_argument("--executor-epoch", type=int, required=True)
    parser.add_argument("--fencing-token", required=True)
    parser.add_argument("--lease-path", type=Path, default=ROOT / "_run/current/GPU_LEASE.json")
    parser.add_argument("--lease-lock-path", type=Path, default=ROOT / "_run/current/GPU_LEASE.lock")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    if args.config is None:
        config_path = corrected_config(output_root)
        predecessor_result = OLD_RESULT
        correction = {"field": "fps", "predecessor": 30.0, "verified": 25.0}
    else:
        config_path = args.config.resolve(strict=True)
        if not args.confirm_config_sha or sha(config_path) != args.confirm_config_sha:
            raise RuntimeError("regression config SHA confirmation failed")
        predecessor_result = args.predecessor_result.resolve(strict=True) if args.predecessor_result else None
        correction = {"field": None, "policy": "CONFIG_VIDEO_HAWOR_CONTRACT_ALREADY_CONSISTENT"}
    config = load(config_path)
    session_id = str(config["session_id"])
    attempt = output_root / "sessions" / session_id / "attempts" / args.attempt_id
    role_output = attempt / "role_mask_output"
    preflight = {
        "schema_version": "role-mask-runtime-contract-repair-preflight-v71",
        "status": "READY_DEVELOPMENT_CANARY",
        "session_id": session_id, "verified_fps": float(config["fps"]),
        "source_config": artifact_ref(SOURCE_CONFIG if args.config is None else config_path),
        "execution_config": artifact_ref(config_path),
        "predecessor_result": artifact_ref(predecessor_result) if predecessor_result else None,
        "claim_limit": "Bounded Role Mask canary/regression execution only until aggregator reviews the complete 1+2 bundle.",
    }
    if args.preflight_only:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return 0
    if attempt.exists():
        raise FileExistsError(attempt)
    attempt.mkdir(parents=True)
    status = "FAILED_RUNTIME_FINAL"
    error = None
    return_code = None
    try:
        ready = validated(config_path, role_output)
        with FileGpuLease(
            lease_path=args.lease_path.resolve(), lock_path=args.lease_lock_path.resolve(),
            task_id=f"role_mask_bounded_v71_{session_id}", attempt_id=args.attempt_id,
            gpu_id=0, executor_epoch=args.executor_epoch, fencing_token=args.fencing_token,
        ):
            return_code = legacy.execute(ready, role_output, "V71_FILE_GPU_LEASE")
        payload = load(role_output / "RESULT.json")
        status = "PASSED" if payload.get("grade") == "B" and payload.get("downstream_authorized") is True else "FAILED_QUALITY_C"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        if "GPU lease unavailable" in error:
            status = "BLOCKED_RESOURCE"
    result = {
        "schema_version": "role-mask-runtime-contract-repair-result-v71",
        "artifact_revision": args.artifact_revision, "status": status, "session_id": session_id,
        "attempt_id": args.attempt_id, "execution_return_code": return_code,
        "correction": correction,
        "preflight": preflight, "role_mask_result": artifact_ref(role_output / "RESULT.json") if (role_output / "RESULT.json").is_file() else None,
        "error": error, "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "code": artifact_ref(Path(__file__)),
        "claim_limit": "Bounded Role Mask canary/regression. Grade B remains a successor candidate until the 1+2 bundle and aggregator review; no other stage authority.",
    }
    atomic_json(attempt / "RESULT.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if status in {"PASSED", "FAILED_QUALITY_C"} else 3


if __name__ == "__main__":
    raise SystemExit(main())
