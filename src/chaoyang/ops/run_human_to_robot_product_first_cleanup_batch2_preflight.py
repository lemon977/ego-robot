"""Freeze exact historical Stereo payloads for a later, separate purge decision."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np

from chaoyang.governance.common import artifact_ref, load_json, validate_artifact_ref
from chaoyang.ops.run_human_to_robot_product_first_cable import ROOT, TASK
from chaoyang.ops.run_human_to_robot_product_first_cleanup_batch1 import _tree

OLD = ROOT / "_run/current/0915_robot15h_foundationstereo_waves_v1/attempts/attempt_0001"
OUT = ROOT / f"_run/current/{TASK}/attempts/attempt_0001/cleanup/batch2"
SESSIONS = ("get_potato_chips_0915_056", "get_potato_chips_0915_068")


def _decode(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    count = 0
    while capture.read()[0]:
        count += 1
    capture.release()
    return count


def _external_refs(session: str) -> list[str]:
    needle = f"{session}/frames/"
    command = ["rg", "-l", "--hidden", "--no-ignore", "--glob", "*.json",
               "--glob", "*.md", "--glob", "*.py", "--fixed-strings", needle,
               "docs/current", "src/chaoyang", "tasks/current", "_run/current"]
    process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    if process.returncode not in (0, 1):
        raise RuntimeError(f"REFERENCE_SCAN_FAILED:{session}:{process.stderr[-500:]}")
    allowed = OLD / f"sessions/potato_chips/{session}/DEPTH_SUMMARY.json"
    return [row for row in process.stdout.splitlines()
            if Path(row).resolve() != allowed.resolve()]


def _busy(target: Path) -> list[str]:
    prefix = str(target.resolve()) + os.sep
    found = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        for link in (proc / "cwd", *(proc / "fd").glob("*")):
            try:
                value = os.readlink(link)
            except (FileNotFoundError, PermissionError, OSError):
                continue
            if value == str(target) or value.startswith(prefix):
                found.append(f"{proc.name}:{link.name}:{value}")
    return found


def main() -> int:
    index = load_json(ROOT / "tasks/current/INDEX.json")
    if len(index.get("task_packets", [])) != 1 or index["task_packets"][0].get("task_id") != TASK:
        raise RuntimeError("TASK_NOT_ROUTABLE")
    if OUT.exists():
        raise FileExistsError(OUT)
    packet = load_json(ROOT / f"tasks/current/{TASK}/TASK_PACKET.json")
    current_sessions = sum(packet["target_sessions"].values(), [])
    for session in SESSIONS:
        if session in current_sessions:
            raise RuntimeError(f"CURRENT_OR_CONDITIONAL_INPUT:{session}")
    retained_environment = ROOT / "_run/current/environments/foundationstereo-py311-v1/bin/python"
    weight = ROOT / "assets/models/checkpoints/foundationstereo/23-51-11/model_best_bp2.pth"
    if not retained_environment.is_file() or not weight.is_file():
        raise RuntimeError("STEREO_REPRODUCER_ENV_OR_WEIGHT_MISSING")
    OUT.mkdir(parents=True)
    samples_root = OUT / "forensic_samples"
    samples_root.mkdir()
    rows = []
    for session in SESSIONS:
        base = OLD / f"sessions/potato_chips/{session}"
        source = base / "frames"
        result = load_json(base / "RESULT.json")
        if result.get("status") != "PASSED" or result.get("strict_metric_contact_authorized") is not False:
            raise RuntimeError(f"HISTORICAL_STATUS_UNEXPECTED:{session}")
        if source.is_symlink() or not source.is_dir():
            raise RuntimeError(f"TARGET_NOT_EXACT_DIRECTORY:{source}")
        refs = _external_refs(session)
        if refs:
            raise RuntimeError(f"EXTERNAL_CURRENT_FRAME_REFERENCE:{session}:{refs[:5]}")
        if _busy(source):
            raise RuntimeError(f"TARGET_BUSY:{session}")
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(source.relative_to(ROOT))],
                          cwd=ROOT, capture_output=True, check=False).returncode == 0:
            raise RuntimeError("TARGET_GIT_TRACKED")
        count = int(result["frame_count"])
        names = (f"{idx:06d}.npz" for idx in (0, count // 2, count - 1))
        selected_samples = {}
        retain = samples_root / session
        retain.mkdir()
        for name in names:
            candidate = source / name
            kept = retain / name
            shutil.copy2(candidate, kept)
            with np.load(kept, allow_pickle=False) as archive:
                if not archive.files:
                    raise RuntimeError(f"SAMPLE_EMPTY:{session}:{name}")
            if artifact_ref(candidate)["sha256"] != artifact_ref(kept)["sha256"]:
                raise RuntimeError(f"SAMPLE_COPY_MISMATCH:{session}:{name}")
            selected_samples[name] = artifact_ref(kept)
        review = result["review"]["video"]
        if validate_artifact_ref(review) or _decode(Path(review["path"])) != count:
            raise RuntimeError(f"RETAINED_REVIEW_INVALID:{session}")
        rows.append({"session_id": session, "source": str(source), "identity": _tree(source),
                     "frame_count": count, "historical_result": artifact_ref(base / "RESULT.json"),
                     "historical_depth_summary": artifact_ref(base / "DEPTH_SUMMARY.json"),
                     "retained_review": review, "retained_samples": selected_samples,
                     "current_or_conditional_reference": False, "active_fd_or_cwd": False,
                     "after_trim_authority": "METADATA_AND_SPARSE_FORENSIC_ONLY_NOT_FULL_DEPTH_RELOADABLE"})
    result = {"schema_version": "PRODUCT_FIRST_CLEANUP_BATCH2_PREFLIGHT_V1",
              "task_id": TASK, "status": "FROZEN_CANDIDATES_NOT_DELETED",
              "targets": rows, "logical_bytes_candidate": sum(row["identity"]["logical_bytes"] for row in rows),
              "allocated_bytes_candidate": sum(row["identity"]["allocated_bytes"] for row in rows),
              "retained_environment": str(retained_environment), "retained_weight": str(weight),
              "protected": ["raw", "processed", "sealed", "archive", "project_external",
                            "current 007/031, 0902 and Sensor inputs", "031 qualified Depth",
                            "HuRo and WIYH unresolved reference", "all current 15-slot videos"],
              "restore_or_reproducer_action": "Old complete frame arrays are intentionally not byte-restorable after trim; regenerate from retained raw source, pinned producer code/config, environment and weight if a future authorized task needs them. Sparse samples and original review remain immediately loadable.",
              "freshness_rule": "Recheck current/conditional refs, active FD/CWD, tree identity, environment and weight before separate purge; any new dependency revokes this selection.",
              "claim_limit": "This preflight does not delete or grant permission to run new Stereo or metric Contact. Historical summaries remain immutable but complete per-frame payload would be trimmed only after separate verification."}
    path = OUT / "SELECTED_TARGETS.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "count": len(rows),
                      "candidate_logical_bytes": result["logical_bytes_candidate"],
                      "selection": str(path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
