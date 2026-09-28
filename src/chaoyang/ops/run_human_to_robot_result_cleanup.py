"""Exact, recoverable-then-purged duplicates and test caches for this cycle."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np

from chaoyang.governance.common import REPO_ROOT, artifact_ref, atomic_json, load_json
from chaoyang.ops.run_human_to_robot_product_first_cleanup_batch1 import _tree

TASK = "human_to_robot_result_breakthrough_20260924"
OUT = REPO_ROOT / f"_run/current/{TASK}/attempts/attempt_0001/lanes/cleanup/batch1"
R2 = REPO_ROOT / ("_run/current/human_to_robot_root_cause_gated_r2_20260923/attempts/"
                  "attempt_0001/lanes/lane1_scene")
DOC = REPO_ROOT / "docs/current/visuals"
DIRECTORIES = [
    (R2 / "clean_candidate_007_wave5/prep/frames", R2 / "clean_candidate_007_wave4/prep/frames", "EXACT_DUPLICATE"),
    (R2 / "clean_candidate_031_wave5/prep/frames", R2 / "clean_candidate_031_wave4/prep/frames", "EXACT_DUPLICATE"),
    (R2 / "clean_candidate_031_wave4_failed_prepare_attempt_0001/prep/frames",
     R2 / "clean_candidate_031_wave4/prep/frames", "EXACT_PREFIX"),
    *[(REPO_ROOT / "tests" / folder / "__pycache__", None, "REGENERABLE_TEST_CACHE")
      for folder in ("", "unit", "pipeline", "governance", "human_ego", "research", "ops")],
    *[(REPO_ROOT / "_run" / folder, None, "REGENERABLE_PYTEST_TMP")
      for folder in ("pytest-of-root", "pytest_quality_acceptance", "pytest_quality_acceptance_gates")],
]
FILES = [
    (DOC / "0915_ROBOT15H_W0_SAM31_TEMPORAL_IDENTITY_QUALITY_V2/get_potato_chips_0915_007_SAM31_HAND_TEMPORAL_REVIEW.mp4",
     DOC / "0915_ROBOT15H_W0_SAM31_TEMPORAL_IDENTITY_RECOVERY_V1/get_potato_chips_0915_007_SAM31_HAND_TEMPORAL_REVIEW.mp4"),
    (DOC / "0915_ROBOT15H_W0_SAM31_TEMPORAL_IDENTITY_QUALITY_V2/get_potato_chips_0915_042_SAM31_HAND_TEMPORAL_REVIEW.mp4",
     DOC / "0915_ROBOT15H_W0_SAM31_TEMPORAL_IDENTITY_RECOVERY_V1/get_potato_chips_0915_042_SAM31_HAND_TEMPORAL_REVIEW.mp4"),
    (DOC / "0915_ROBOT15H_W0_SAM31_TASK_OBJECT_RECOVERY_V1/play_cards_0915_119_SAM31_TASK_OBJECT_REVIEW.mp4",
     DOC / "0915_ROBOT15H_W0_SAM31_TASK_OBJECT_V1/play_cards_0915_119_SAM31_TASK_OBJECT_REVIEW.mp4"),
    (DOC / "0915_ROBOT15H_W0_SAM31_TASK_OBJECT_RECOVERY_V1/play_cards_0915_031_SAM31_TASK_OBJECT_REVIEW.mp4",
     DOC / "0915_ROBOT15H_W0_SAM31_TASK_OBJECT_V1/play_cards_0915_031_SAM31_TASK_OBJECT_REVIEW.mp4"),
    (DOC / "0915_ROBOT_RECOVERY_15H_V2/D1_CLEAN_PREP_V2/get_potato_chips_0915_097_D1_CLEAN_PREP_OFFLINE_REVIEW.mp4",
     DOC / "0915_ROBOT_RECOVERY_15H_V2/D1_CLEAN_PREP/get_potato_chips_0915_097_D1_CLEAN_PREP_OFFLINE_REVIEW.mp4"),
    (DOC / "0915_ROBOT_RECOVERY_15H_V2/D1_CLEAN_PREP_V2/play_cards_0915_044_D1_CLEAN_PREP_OFFLINE_REVIEW.mp4",
     DOC / "0915_ROBOT_RECOVERY_15H_V2/D1_CLEAN_PREP/play_cards_0915_044_D1_CLEAN_PREP_OFFLINE_REVIEW.mp4"),
]


def _hash(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _file_identity(path: Path) -> dict:
    stat = path.stat()
    return {"files": 1, "logical_bytes": stat.st_size, "allocated_bytes": stat.st_blocks * 512,
            "sha256": _hash(path)}


def _regenerable_tree(path: Path) -> dict:
    """Hash cache symlink *names*, never follow their destinations."""
    digest = hashlib.sha256()
    files = total = allocated = 0
    for child in sorted(path.rglob("*")):
        relative = child.relative_to(path).as_posix()
        stat = child.lstat()
        if child.is_symlink():
            target = os.readlink(child)
            digest.update(f"L\0{relative}\0{target}\n".encode())
            total += len(target.encode())
            files += 1
        elif child.is_file():
            digest.update(f"F\0{relative}\0{stat.st_size}\0{_hash(child)}\n".encode())
            files += 1
            total += stat.st_size
            allocated += stat.st_blocks * 512
    return {"files": files, "logical_bytes": total, "allocated_bytes": allocated,
            "tree_sha256": digest.hexdigest()}


def _identity(path: Path, mode: str) -> dict:
    if path.is_dir():
        return (_regenerable_tree(path) if mode.startswith("REGENERABLE") else _tree(path))
    return _file_identity(path)


def _open_fd_targets(targets: list[Path]) -> list[str]:
    busy = []
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        fd = process / "fd"
        try:
            names = list(fd.iterdir())
        except (OSError, PermissionError):
            continue
        for descriptor in names:
            try:
                value = Path(os.readlink(descriptor))
            except (OSError, PermissionError):
                continue
            if any(value == target or target in value.parents for target in targets):
                busy.append(f"{process.name}:{descriptor.name}:{value}")
    return busy


def _check_retained(source: Path, retained: Path | None, mode: str, identity: dict):
    if mode == "EXACT_DUPLICATE":
        if source.is_dir():
            if not retained.is_dir() or _tree(retained) != identity:
                raise RuntimeError(f"RETAINED_DIR_NOT_IDENTICAL:{source}")
        elif not retained.is_file() or _file_identity(retained)["sha256"] != identity["sha256"]:
            raise RuntimeError(f"RETAINED_FILE_NOT_IDENTICAL:{source}")
    elif mode == "EXACT_PREFIX":
        if not source.is_dir() or not retained.is_dir():
            raise RuntimeError("PREFIX_DIR_MISSING")
        for child in source.iterdir():
            if not child.is_file() or not (retained / child.name).is_file() or _hash(child) != _hash(retained / child.name):
                raise RuntimeError(f"PREFIX_NOT_IDENTICAL:{child}")
    elif mode not in {"REGENERABLE_TEST_CACHE", "REGENERABLE_PYTEST_TMP"}:
        raise RuntimeError("UNKNOWN_CLEANUP_MODE")


def _smoke():
    root = REPO_ROOT / f"_run/current/{TASK}/attempts/attempt_0001"
    numeric = root / "lanes/robot/POKER_171_INDEPENDENT_FK_V3.npz"
    clean = root / "lanes/clean/POKER_076_091_INPUT_REPAIR/clean/000080.png"
    label = root / "lanes/data/SENSOR_097_LOCAL_KAI22_QUALIFICATION.json"
    with np.load(numeric, allow_pickle=False) as arrays:
        if arrays["q_arm"].shape != (171, 2, 7):
            raise RuntimeError("ROBOT_CONSUMER_BROKEN")
    if cv2.imread(str(clean)) is None or not label.is_file():
        raise RuntimeError("CLEAN_OR_DATA_CONSUMER_BROKEN")
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1",
           "TMPDIR": str(REPO_ROOT / "_run"), "XDG_CACHE_HOME": str(REPO_ROOT / ".cache/xdg"),
           "PYTEST_ADDOPTS": "-p no:cacheprovider"}
    test = subprocess.run(["/usr/local/bin/python", "-B", "-m", "pytest", "-q",
                           "tests/unit/test_result_robot_hand_root.py"], cwd=REPO_ROOT,
                          env=env, capture_output=True, text=True)
    if test.returncode:
        raise RuntimeError("TEST_CONSUMER_BROKEN:" + test.stdout[-1000:] + test.stderr[-1000:])
    governance = subprocess.run(["/usr/local/bin/python", "-B", "-m", "chaoyang.cli",
                                 "validate-governance"], cwd=REPO_ROOT,
                                env=env, capture_output=True, text=True)
    if governance.returncode or '"status": "PASS"' not in governance.stdout:
        raise RuntimeError("GOVERNANCE_CONSUMER_BROKEN:" + governance.stdout[-1500:] + governance.stderr[-500:])
    return {"robot_npz": "PASS", "clean_png": "PASS", "data_receipt": "PASS",
            "test": "PASS", "governance": "PASS"}


def main():
    if OUT.exists():
        raise FileExistsError("CLEANUP_BATCH_ALREADY_EXISTS")
    if load_json(REPO_ROOT / f"tasks/current/{TASK}/TASK_PACKET.json")["task_id"] != TASK:
        raise RuntimeError("TASK_NOT_REGISTERED")
    rows = []
    for path, retained, mode in DIRECTORIES + [(a, b, "EXACT_DUPLICATE") for a, b in FILES]:
        if not path.exists() or path.is_symlink() or path == retained:
            raise RuntimeError(f"UNSAFE_TARGET:{path}")
        if not path.resolve().is_relative_to(REPO_ROOT.resolve()):
            raise RuntimeError("OUTSIDE_PROJECT_TARGET")
        identity = _identity(path, mode)
        _check_retained(path, retained, mode, identity)
        rows.append({"path": str(path), "retained": str(retained) if retained else None,
                     "mode": mode, "kind": "dir" if path.is_dir() else "file", "identity": identity})
    busy = _open_fd_targets([Path(row["path"]) for row in rows])
    if busy:
        raise RuntimeError("ACTIVE_FD:" + ";".join(busy[:10]))
    historical = [REPO_ROOT / "_run/current/human_to_robot_root_cause_gated_r2_20260923/attempts/attempt_0001/RESULT.json",
                  REPO_ROOT / "_run/current/human_to_robot_quality_acceptance_20260924/attempts/attempt_0001/RESULT.json"]
    before_receipts = {str(path): _hash(path) for path in historical}
    OUT.mkdir(parents=True)
    isolation = OUT / "isolation"
    isolation.mkdir()
    atomic_json(OUT / "SELECTED.json", {"schema_version": "RESULT_CLEANUP_SELECTED_V1", "targets": rows,
                 "conditional_retention": ["Poker old/new q and target/mount", "Poker old/new Clean and reference masks",
                                           "Sensor097 raw node HDF5 and q/FK", "all model weights and runtime environments",
                                           "original 4 product entities and unresolved complaint videos"]})
    events = OUT / "EVENTS.jsonl"
    def event(row, status):
        with events.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"path": row["path"], "state": status}, ensure_ascii=False) + "\n")
            stream.flush(); os.fsync(stream.fileno())
    isolated = []
    try:
        for index, row in enumerate(rows):
            source = Path(row["path"])
            current = _identity(source, row["mode"])
            if current != row["identity"]:
                raise RuntimeError(f"TARGET_CHANGED:{source}")
            event(row, "SELECTED")
            destination = isolation / f"{index:03d}"
            if destination.exists():
                raise RuntimeError("ISOLATION_COLLISION")
            os.replace(source, destination)
            isolated.append((row, destination))
            event(row, "ISOLATED")
        check = _smoke()
        atomic_json(OUT / "CONSUMER_CHECK.json", check)
        for row, _ in isolated:
            event(row, "CONSUMER_CHECKED")
    except Exception:
        for row, destination in reversed(isolated):
            source = Path(row["path"])
            if source.exists() or not destination.exists():
                event(row, "ROLLBACK_BLOCKED_NEW_CONTENT")
                continue
            current = _identity(destination, row["mode"])
            if current != row["identity"]:
                event(row, "ROLLBACK_BLOCKED_IDENTITY")
                continue
            os.replace(destination, source)
            event(row, "ROLLED_BACK")
        raise
    for row, destination in isolated:
        source = Path(row["path"])
        current = _identity(destination, row["mode"])
        if source.exists() or current != row["identity"]:
            raise RuntimeError("PURGE_ABORT_NEW_PATH_OR_CHANGED_BYTES")
        if row["kind"] == "dir":
            shutil.rmtree(destination)
        else:
            destination.unlink()
        event(row, "PURGED")
    if before_receipts != {str(path): _hash(path) for path in historical}:
        raise RuntimeError("HISTORICAL_RECEIPT_CHANGED")
    notice = {"schema_version": "RESULT_BREAKTHROUGH_HISTORICAL_PAYLOAD_TRIM_V1",
              "task_id": TASK, "historical_receipts_sha256_unchanged": before_receipts,
              "trimmed": [{"old_path": row["path"], "replacement": row["retained"],
                           "mode": row["mode"]} for row in rows if row["mode"] in {"EXACT_DUPLICATE", "EXACT_PREFIX"}],
              "claim_limit": "Only old duplicate/partial prep frames, duplicate docs media and regenerable test caches removed; old receipts still name historical payload paths and some old runs are not byte-replayable at those paths."}
    atomic_json(OUT / "HISTORICAL_PAYLOAD_TRIM_NOTICE.json", notice)
    result = {"schema_version": "RESULT_BREAKTHROUGH_CLEANUP_BATCH1_V1", "task_id": TASK,
              "execution": "ACTUALLY_PURGED", "targets": rows,
              "logical_deleted_bytes": sum(row["identity"]["logical_bytes"] for row in rows),
              "allocated_deleted_bytes_estimate": sum(row["identity"]["allocated_bytes"] for row in rows),
              "isolation_remaining_bytes": 0, "consumer_check": artifact_ref(OUT / "CONSUMER_CHECK.json"),
              "historical_notice": artifact_ref(OUT / "HISTORICAL_PAYLOAD_TRIM_NOTICE.json"),
              "physical_reclaim": "UNKNOWN_SHARED_CPFS", "project_net_change": "NOT_MEASURED_CONCURRENT_OUTPUTS"}
    atomic_json(OUT / "DELETE_RECEIPT.json", result)
    print(json.dumps({"status": result["execution"], "targets": len(rows),
                      "logical_deleted_bytes": result["logical_deleted_bytes"],
                      "receipt": str(OUT / "DELETE_RECEIPT.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
