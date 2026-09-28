"""Purge one abandoned Stereo staging tree after sparse forensic retention."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np

from chaoyang.governance.common import artifact_ref, load_json
from chaoyang.ops.run_human_to_robot_product_first_cable import ROOT, TASK
from chaoyang.ops.run_human_to_robot_product_first_cleanup_batch1 import (
    _tree, isolate_verified, rollback_verified,
)

OLD = ROOT / "_run/current/0915_robot15h_foundationstereo_wave0_v1/attempts/attempt_0001"
SESSION = OLD / "sessions/potato_chips/get_potato_chips_0915_042"
SOURCE = OLD / "sessions/potato_chips/.get_potato_chips_0915_042.tmp-4181151-ada005615a9f4e7887f9ce4846529d62"
OUT = ROOT / f"_run/current/{TASK}/attempts/attempt_0001/cleanup/batch1b"
SAMPLES = ("000000.npz", "000075.npz", "000150.npz")


def _json(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _event(state: str, **details: object) -> None:
    with (OUT / "DELETE_EVENTS.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"state": state, **details}, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush(); os.fsync(handle.fileno())


def _busy_processes(target: Path) -> list[str]:
    busy: list[str] = []
    prefix = str(target.resolve()) + os.sep
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        for link in (proc / "cwd", *(proc / "fd").glob("*")):
            try:
                value = os.readlink(link)
            except (FileNotFoundError, PermissionError, OSError):
                continue
            if value == str(target) or value.startswith(prefix):
                busy.append(f"{proc.name}:{link.name}:{value}")
    return busy


def _current_ref_scan(target: Path) -> list[str]:
    roots = (ROOT / "docs/current", ROOT / "src/chaoyang", ROOT / f"tasks/current/{TASK}",
             ROOT / f"_run/current/{TASK}/attempts/attempt_0001")
    command = ["rg", "-l", "--hidden", "--no-ignore", "--glob", "*.json",
               "--glob", "*.md", "--glob", "*.py", "--fixed-strings", str(target),
               *(str(root) for root in roots)]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    if result.returncode not in (0, 1):
        raise RuntimeError(f"REFERENCE_SCAN_FAILED:{result.stderr[-500:]}")
    # Our append-only selection and deletion receipts intentionally name the
    # old path; they are not runtime consumers and must not veto post-isolation.
    return [line for line in result.stdout.splitlines()
            if line and not Path(line).is_relative_to(OUT)]


def _decode(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    count = 0
    while capture.read()[0]:
        count += 1
    capture.release()
    return count


def main() -> int:
    index = load_json(ROOT / "tasks/current/INDEX.json")
    if len(index.get("task_packets", [])) != 1 or index["task_packets"][0].get("task_id") != TASK:
        raise RuntimeError("TASK_NOT_ROUTABLE")
    if OUT.exists() or SOURCE.is_symlink() or not SOURCE.is_dir():
        raise RuntimeError("TARGET_NOT_EXACT_ABANDONED_TREE")
    packet = load_json(ROOT / f"tasks/current/{TASK}/TASK_PACKET.json")
    targets = packet["target_sessions"]
    if "get_potato_chips_0915_042" in sum(targets.values(), []):
        raise RuntimeError("TARGET_IN_CURRENT_OR_CONDITIONAL_TASK")
    failed = load_json(SESSION / "RESULT.json")
    parent = load_json(OLD / "RESULT.json")
    if failed.get("status") != "FAILED_RUNTIME" or parent.get("status") != "FAILED_RUNTIME_FINAL":
        raise RuntimeError("STAGING_RESULT_NOT_TERMINAL_FAILURE")
    if _busy_processes(SOURCE):
        raise RuntimeError("TARGET_HAS_ACTIVE_FD_OR_CWD")
    if _current_ref_scan(SOURCE):
        raise RuntimeError("TARGET_HAS_CURRENT_REFERENCE")
    if subprocess.run(["git", "ls-files", "--error-unmatch", str(SOURCE.relative_to(ROOT))],
                      cwd=ROOT, capture_output=True, check=False).returncode == 0:
        raise RuntimeError("TARGET_TRACKED_BY_GIT")
    parent_ref, failed_ref = artifact_ref(OLD / "RESULT.json"), artifact_ref(SESSION / "RESULT.json")
    identity = _tree(SOURCE)
    if identity["files"] < 150 or identity["logical_bytes"] < 500_000_000:
        raise RuntimeError("ABORTED_TREE_IDENTITY_UNEXPECTED")
    OUT.mkdir(parents=True)
    retained = OUT / "forensic_samples"
    retained.mkdir()
    samples = {}
    for name in SAMPLES:
        source = SOURCE / "frames" / name
        target = retained / name
        shutil.copy2(source, target)
        with np.load(target, allow_pickle=False) as archive:
            if not archive.files:
                raise RuntimeError(f"EMPTY_NPZ_SAMPLE:{name}")
        if artifact_ref(source)["sha256"] != artifact_ref(target)["sha256"]:
            raise RuntimeError(f"FORENSIC_COPY_MISMATCH:{name}")
        samples[name] = artifact_ref(target)
    review_source = SOURCE / "get_potato_chips_0915_042_FOUNDATIONSTEREO_REVIEW.mp4"
    review_target = retained / review_source.name
    shutil.copy2(review_source, review_target)
    if artifact_ref(review_source)["sha256"] != artifact_ref(review_target)["sha256"]:
        raise RuntimeError("REVIEW_COPY_MISMATCH")
    old_decoded = _decode(review_source)
    if old_decoded <= 0 or _decode(review_target) != old_decoded:
        raise RuntimeError("RETAINED_REVIEW_DECODE_MISMATCH")
    selected = {"schema_version": "PRODUCT_FIRST_ABORTED_STEREO_DELETE_SELECTION_V1",
                "task_id": TASK, "source": str(SOURCE), "identity": identity,
                "failure_result": failed_ref, "parent_result": parent_ref,
                "retained_samples": samples, "retained_video": artifact_ref(review_target),
                "retained_video_decoded_frames": old_decoded,
                "current_refs": [], "active_fd_or_cwd": [],
                "protected": ["raw", "processed", "sealed", "archive", "project_external",
                              "current 007/031, 0902 and Sensor inputs", "weights and CAD"]}
    _json(OUT / "SELECTED_TARGET.json", selected)
    _event("SELECTED", identity=identity, source=str(SOURCE))
    isolation = OUT / "isolated_aborted_042"
    isolate_verified(SOURCE, isolation, identity)
    _event("ISOLATED", source=str(SOURCE), isolation=str(isolation))
    try:
        if artifact_ref(OLD / "RESULT.json") != parent_ref or artifact_ref(SESSION / "RESULT.json") != failed_ref:
            raise RuntimeError("HISTORICAL_RECEIPT_CHANGED")
        for name in SAMPLES:
            with np.load(retained / name, allow_pickle=False) as archive:
                if not archive.files:
                    raise RuntimeError("FORENSIC_SAMPLE_NOT_LOADABLE")
        if _decode(review_target) != old_decoded:
            raise RuntimeError("RETAINED_REVIEW_NOT_LOADABLE")
        check = subprocess.run(["/usr/local/bin/python", "-B", "-m", "chaoyang.cli", "validate-governance"],
            cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT / "src"),
                           "PYTHONDONTWRITEBYTECODE": "1", "TMPDIR": str(ROOT / ".cache/tmp"),
                           "XDG_CACHE_HOME": str(ROOT / ".cache/xdg")},
            capture_output=True, text=True, check=False)
        if check.returncode:
            raise RuntimeError(f"GOVERNANCE_CHECK_FAILED:{check.stdout[-900:]}")
        if _busy_processes(isolation) or _current_ref_scan(SOURCE):
            raise RuntimeError("TARGET_BECAME_BUSY_OR_REFERENCED")
        _event("CONSUMER_CHECKED", governance="PASS", forensic_samples="PASS")
    except Exception:
        rollback_verified(SOURCE, isolation, identity)
        _event("ROLLED_BACK")
        raise
    if _tree(isolation) != identity or SOURCE.exists():
        raise RuntimeError("PURGE_IDENTITY_OR_NEW_CONTENT")
    shutil.rmtree(isolation)
    _event("PURGED", logical_deleted_bytes=identity["logical_bytes"])
    receipt = {"schema_version": "PRODUCT_FIRST_CLEANUP_ABORTED_STEREO_V1",
               "task_id": TASK, "execution": "ACTUALLY_PURGED", "target": str(SOURCE),
               "logical_deleted_bytes": identity["logical_bytes"],
               "allocated_deleted_bytes_estimate": identity["allocated_bytes"],
               "retained_forensic_logical_bytes": sum(path["bytes"] for path in samples.values())
               + artifact_ref(review_target)["bytes"],
               "selection": artifact_ref(OUT / "SELECTED_TARGET.json"),
               "historical_receipts_modified": False,
               "historical_payload_notice": "Terminal failed Stereo staging payload removed; three NPZ samples and review video retained; the original full staging sequence is no longer reloadable.",
               "physical_reclaimed_bytes": "UNKNOWN_CPFS_SHARED_ALLOCATION",
               "claim_limit": "Only exact abandoned hidden staging tree purged. Current 007/031 and protected source data untouched."}
    _json(OUT / "DELETE_RECEIPT.json", receipt)
    print(json.dumps({"status": "ACTUALLY_PURGED", "logical_deleted_bytes": identity["logical_bytes"],
                      "receipt": str(OUT / "DELETE_RECEIPT.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
