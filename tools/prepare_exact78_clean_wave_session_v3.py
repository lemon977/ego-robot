#!/usr/bin/env python3
"""Prepare one frozen Wave-0 Clean session without touching other rows."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import cv2

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from tools import prepare_exact78_clean_expanded_role_v3 as core  # noqa: E402


EXPANSIONS = {
    "chips": {"left_human_radius_px": 24, "right_human_radius_px": 20,
              "tracker_radius_px": 60, "temporal_radius_frames": 1,
              "temporal_rule": "current spatial dilation UNION intersection of symmetric neighbor dilations"},
    "poker": {"left_human_radius_px": 24, "right_human_radius_px": 18,
              "tracker_radius_px": 60, "temporal_radius_frames": 1,
              "temporal_rule": "current spatial dilation UNION intersection of symmetric neighbor dilations"},
}


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def validate(selection_path: Path, session: str) -> tuple[dict, dict]:
    selection_path = selection_path.resolve(strict=True)
    selection = core.load_json(selection_path)
    if selection.get("schema_version") != "exact78-clean-wave0-selection-v3" or selection.get("status") != "FROZEN_58_EXISTING4_PENDING54":
        raise core.PrepareError("unsupported or non-frozen Wave0 selection")
    matches = [x for x in selection["sessions"] if x.get("session_id") == session]
    if len(matches) != 1 or matches[0].get("status") != "PENDING_FRESH_WAVE0":
        raise core.PrepareError(f"{session}: not exactly one pending Wave0 row")
    row = matches[0]
    task = row["task"]
    frame_count = int(row["frame_count"])
    upstream = row["upstream"]
    for key in ("hawor", "role_mask", "task_object_mask", "depth", "object6d"):
        core.checked_ref(upstream[key], f"{session}.{key}")
    role_result = core.load_json(Path(upstream["role_mask"]["path"]))
    task_result = core.load_json(Path(upstream["task_object_mask"]["path"]))
    object_result = core.load_json(Path(upstream["object6d"]["path"]))
    if (role_result.get("session") != session or role_result.get("task") != task or
            role_result.get("frame_count") != frame_count or role_result.get("grade") not in {"A", "B"} or
            role_result.get("downstream_authorized") is not True):
        raise core.PrepareError(f"{session}: role authority gate failed")
    if (task_result.get("session") != session or task_result.get("task") != task or
            task_result.get("frame_count") != frame_count or task_result.get("grade") not in {"A", "B"} or
            task_result.get("downstream_authorized") is not True):
        raise core.PrepareError(f"{session}: task-object authority gate failed")
    if (object_result.get("session_id") != session or object_result.get("task") != task or
            object_result.get("grade") not in {"A", "B"} or object_result.get("downstream_authorized") is not True or
            "CLEAN_VISUAL_BASELINE_INPUT" not in object_result.get("authorized_scopes", [])):
        raise core.PrepareError(f"{session}: Object6D Clean-scope gate failed")
    role_manifest_path = core.checked_ref(role_result["artifacts"]["frame_manifest"], f"{session}.role_manifest")
    task_manifest_path = core.checked_ref(task_result["artifacts"]["manifest"], f"{session}.task_manifest")
    role_manifest, task_manifest = core.load_json(role_manifest_path), core.load_json(task_manifest_path)
    role_frames, task_frames = role_manifest.get("frames"), task_manifest.get("frames")
    if not isinstance(role_frames, list) or not isinstance(task_frames, list) or len(role_frames) != frame_count or len(task_frames) != frame_count:
        raise core.PrepareError(f"{session}: manifest frame closure failed")
    expected_ids = {"0", "1", "2"} if task == "chips" else {"0"}
    expected_roles = {"left_human", "right_human", "left_tracker", "right_tracker"}
    observed_counts = {key: 0 for key in expected_ids}
    predecessor_path = core.checked_ref(
        role_result.get("pins", {}).get("immutable_predecessor_result", {}),
        f"{session}.role_predecessor_result",
    )
    predecessor = core.load_json(predecessor_path)
    config_path = core.checked_ref(predecessor.get("pins", {}).get("config", {}), f"{session}.role_config")
    config = core.load_json(config_path)
    raw_root = Path(str(config.get("raw_all_data", ""))).resolve(strict=True)
    raw_video_ref = predecessor.get("pins", {}).get("raw_video", {})
    raw_video_path = core.checked_ref(raw_video_ref, f"{session}.role_raw_video")
    video_ref = task_manifest.get("input", {}).get("selected_rgb", {})
    if not core.same_ref(video_ref, raw_video_ref):
        raise core.PrepareError(f"{session}: role and task-object selected MP4 differ")
    capture = cv2.VideoCapture(str(raw_video_path))
    if not capture.isOpened():
        raise core.PrepareError(f"{session}: selected MP4 cannot be decoded")
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    video_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if video_count != frame_count or abs(fps - 30.0) > 0.05:
        capture.release()
        raise core.PrepareError(f"{session}: selected MP4 metadata mismatch")
    shape = (height, width)
    try:
        for frame_id, (rf, tf) in enumerate(zip(role_frames, task_frames, strict=True)):
            if rf.get("source_frame") != frame_id or tf.get("source_frame") != frame_id:
                raise core.PrepareError(f"{session}: frame identity mismatch at {frame_id}")
            raw_path = (raw_root / f"{frame_id:05d}" / "rgb.png").resolve(strict=True)
            if not raw_path.is_file() or raw_path.is_symlink():
                raise core.PrepareError(f"{session}: raw PNG missing or symlink at {frame_id}")
            raw = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
            ok, decoded = capture.read()
            if raw is None or raw.shape[:2] != shape or not ok or decoded.shape[:2] != shape:
                raise core.PrepareError(f"{session}: RGB decode/shape failed at {frame_id}")
            role_sha = str(rf.get("selected_rgb_decoded_sha256", ""))
            task_sha = str(tf.get("selected_rgb_decoded_sha256", ""))
            if core.array_sha256(raw) != role_sha or core.array_sha256(decoded) != task_sha:
                raise core.PrepareError(f"{session}: role/task RGB-domain identity failed at {frame_id}")
            if set(rf.get("role_masks", {})) != expected_roles:
                raise core.PrepareError(f"{session}: role set mismatch at {frame_id}")
            for name, reference in rf["role_masks"].items():
                core.read_binary_mask(reference, f"{session}.{frame_id}.{name}", shape)
            instances = tf.get("physical_instances", {})
            if set(instances) != expected_ids:
                raise core.PrepareError(f"{session}: physical instance set mismatch at {frame_id}")
            for iid, instance in instances.items():
                mask = core.read_binary_mask(instance["mask"], f"{session}.{frame_id}.object{iid}", shape)
                area = int(mask.sum())
                observed = bool(instance.get("observed"))
                valid = bool(instance.get("valid"))
                if observed != valid or observed != bool(area) or area != int(instance.get("area_px", -1)):
                    raise core.PrepareError(f"{session}: instance {iid} validity/area mismatch at {frame_id}")
                observed_counts[iid] += int(observed)
        ok, _extra = capture.read()
        if ok:
            raise core.PrepareError(f"{session}: selected MP4 has extra decodable frames")
    finally:
        capture.release()
    if {str(k): int(v) for k, v in task_result.get("observed_counts", {}).items()} != observed_counts:
        raise core.PrepareError(f"{session}: observed-count closure failed")
    return row, {
        "session": session, "task": task, "frame_count": frame_count, "fps": 30.0,
        "source_resolution": [width, height], "raw_root": raw_root, "raw_video": video_ref,
        "role_result": upstream["role_mask"], "task_object_result": upstream["task_object_mask"],
        "object6d_result": upstream["object6d"], "role_manifest_path": role_manifest_path,
        "task_manifest_path": task_manifest_path, "role_manifest": role_manifest,
        "task_manifest": task_manifest, "observed_object_frames": sum(observed_counts.values()),
    }


def prepare(plan_root: Path, selection: Path, session: str) -> dict:
    plan_root = plan_root.resolve(strict=True)
    row, context = validate(selection, session)
    targets = [plan_root / "sessions" / session, plan_root / "specs" / f"{session}_real_donor_input.json",
               plan_root / "specs" / f"{session}_propainter.json",
               plan_root / "preparation_receipts" / f"{session}.json"]
    if any(x.exists() or x.is_symlink() for x in targets):
        raise core.PrepareError(f"{session}: no-clobber preparation target exists")
    stage = Path(tempfile.mkdtemp(prefix=f".{session}.prepare.", dir=plan_root))
    core.EXPANSION = EXPANSIONS[context["task"]]
    try:
        handoff = core.make_handoff(context, stage, plan_root)
        specs = core.make_specs(context, handoff, stage, plan_root)
        receipt = {
            "schema_version": "exact78-clean-wave-session-prepare-v3", "created_at": now(),
            "status": "PASS_CPU_HANDOFF_AND_SPECS_READY", "task": context["task"], "session": session,
            "frame_count": context["frame_count"], "selection": core.artifact(selection),
            "upstream": row["upstream"], "expansion": EXPANSIONS[context["task"]],
            "handoff": handoff, "specs": specs,
            "hard_gates": {"all_source_refs_exact": "PASS", "all_frames_and_roles_exact": "PASS",
                           "physical_instance_contract": "PASS", "object_pixels_excluded_from_removal": "PASS",
                           "fresh_no_clobber": "PASS"},
            "claim_limit": "CPU handoff/spec preparation only; no donor, synthetic Clean, Robot or training authority.",
        }
        core.write_json(stage / "receipt.json", receipt)
        (plan_root / "sessions").mkdir(exist_ok=True)
        (plan_root / "specs").mkdir(exist_ok=True)
        (plan_root / "preparation_receipts").mkdir(exist_ok=True)
        os.rename(stage / "sessions" / session, targets[0])
        os.rename(stage / "specs" / f"{session}_real_donor_input.json", targets[1])
        os.rename(stage / "specs" / f"{session}_propainter.json", targets[2])
        os.rename(stage / "receipt.json", targets[3])
        shutil.rmtree(stage)
        return core.load_json(targets[3])
    except Exception:
        failed = plan_root / f"FAILED_PREPARE_{session}_{os.getpid()}"
        if stage.exists() and not failed.exists():
            os.rename(stage, failed)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--session", required=True)
    args = parser.parse_args()
    result = prepare(args.plan_root, args.selection, args.session)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
