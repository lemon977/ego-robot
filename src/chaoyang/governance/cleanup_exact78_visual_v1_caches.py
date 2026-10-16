#!/usr/bin/env python3
"""Governed, exact-file cleanup of four rebuildable Exact78 V1 caches.

This module is deliberately not a routed algorithm operation.  It may be run
only after the separate V1 adoption receipt exists.  Importing it does nothing.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
import subprocess
import sys
from typing import Any

from chaoyang.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH,
    artifact_ref, atomic_json, canonical_bytes, load_json, now_iso,
    process_identity, publish_bundle,
)
from chaoyang.governance.register_single_task_packet import _validate_packet


TASK = "exact78_visual_baseline_v1_cache_cleanup_20261008"
ROOT = REPO_ROOT / "_run/current" / TASK / "attempts/attempt_0001"
PACKET = REPO_ROOT / "tasks/current" / TASK / "TASK_PACKET.json"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
RECEIPT = REPO_ROOT / "tasks/receipts/EXACT78_VISUAL_BASELINE_V1_CACHE_CLEANUP_20261008.json"
PROOF = (REPO_ROOT / "docs/plans/EXACT78_VISUAL_BASELINE_V1_CLEANUP_20261008"
         / "lanes/media/PROOF.json")
PROOF_SHA = "9585fe46efbf2b91dc41d6893e630a4b4f02457107c49a416f44564df3303fa2"
V13_RESULT_SHA = "cbb2d4eab493f9fb1132f11f3488c3ef3ce6125cd0fe2b1b1c6194a6be273dd1"
CACHE_ROOTS = (REPO_ROOT / "_run/caches/t1b_import_check",
               REPO_ROOT / "_run/caches/takeover_baseline_v1")
# Literal allowlist: neither a directory walk nor a mutable proof may expand it.
ALLOWLIST = (
    ("_run/caches/t1b_import_check/matplotlib/fontlist-v390.json", 26566,
     "ca608837b5fe5ea49fd787cea991f2d224958f461136502a86c93312b7b7dd9e"),
    ("_run/caches/takeover_baseline_v1/numba/pipeline_454c067ebf03dd582a0924ba189109ef16c6bb4c/"
     "exact78_robot_raster_v1.rasterize_zbuffer-132.py311.1.nbc", 116720,
     "e32dd8d40b0d2acf4dbb098cf53d1dc26a6df8cdcaac69c0e700f6bd1f985bfe"),
    ("_run/caches/takeover_baseline_v1/numba/pipeline_454c067ebf03dd582a0924ba189109ef16c6bb4c/"
     "exact78_robot_raster_v1.rasterize_zbuffer-132.py311.nbi", 1611,
     "bb4fe1df5ad19cda7bf196a0fd0b657381279efb21f51be85e89046e249fc210"),
    ("_run/caches/takeover_baseline_v1/xdg/torch/kernels/"
     "reduction_prod_arch9.0_nvrtc12.8_sass_29437_3eb94150d986fdf056f52a9c214e66903d13e1d6", 68376,
     "db8b0880fa7ceec4e8ed449db3e90cd61f33425d17114cd0281b009395c4d911"),
)
RUNTIME_REF_ROOTS = ("docs/current", "docs/governance", "tasks/current", "src", "scripts",
                     "configs", "manifests", "review")
PROTECTED_POINTERS = (REPO_ROOT / "docs/current/ROBOT100_HARDWARE_BASELINE_V16.json",
                      REPO_ROOT / "docs/current/ROBOT100_BATCH_ZH.md",
                      REPO_ROOT / "src/chaoyang/ops/run_pico_manus_motion_v2.py")


class ReferenceProofError(RuntimeError):
    """Fail closed without inferring deletion authority from a category label."""


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ReferenceProofError(message)


def _index_body() -> dict[str, Any]:
    return {key: copy.deepcopy(value) for key, value in load_json(INDEX).items()
            if key not in {"governance_revision", "generation_id", "generated_at", "supersedes_index"}}


def _sha_fd(fd: int) -> str:
    digest = hashlib.sha256()
    with os.fdopen(os.dup(fd), "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_adoption_receipt(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    _check(resolved.parent == REPO_ROOT / "tasks/receipts", "V1_RECEIPT_OUTSIDE_RECEIPTS")
    receipt = load_json(resolved)
    serialized = json.dumps(receipt, ensure_ascii=False, sort_keys=True)
    status = str(receipt.get("status", ""))
    _check(status.startswith("PASSED") or status in {"PUBLISHED", "ADOPTED"},
           "V1_RECEIPT_NOT_PUBLISHED")
    _check("EXACT78_VISUAL_BASELINE_V1" in serialized and V13_RESULT_SHA in serialized,
           "V1_RECEIPT_DOES_NOT_BIND_V13")
    return artifact_ref(resolved)


def _validate_proof() -> dict[str, Any]:
    ref = artifact_ref(PROOF)
    _check(ref["sha256"] == PROOF_SHA, "MEDIA_PROOF_SHA_DRIFT")
    proof = load_json(PROOF)
    observed = [(row.get("path"), row.get("bytes"), row.get("sha256"))
                for row in proof.get("candidate_files", [])]
    _check(tuple(observed) == ALLOWLIST and sum(row[1] for row in ALLOWLIST) == 213273,
           "MEDIA_PROOF_ALLOWLIST_DRIFT")
    return ref


def _validate_governance() -> None:
    result = subprocess.run(
        [sys.executable, "-B", "-m", "chaoyang.cli", "validate-governance"],
        cwd=REPO_ROOT, env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        capture_output=True, text=True, timeout=120, check=False,
    )
    _check(result.returncode == 0, "GOVERNANCE_VALIDATION_FAILED:" + result.stdout[-400:]
           + result.stderr[-400:])
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise ReferenceProofError("GOVERNANCE_VALIDATION_UNPARSEABLE") from error
    _check(payload.get("status") == "PASS", "GOVERNANCE_NOT_PASS")


def _no_other_active_task(*, self_registered: bool) -> None:
    state, index = load_json(TASK_STATE_PATH), load_json(INDEX)
    permitted = {TASK} if self_registered else set()
    active = [row.get("task_id") for row in state.get("tasks", [])
              if row.get("status") in {"CLAIMED", "RUNNING", "WAIT_GPU_RESOURCE"}
              and row.get("task_id") not in permitted]
    allowed = [row.get("task_id") for row in index.get("task_packets", [])
               if row.get("execution_allowed") and row.get("task_id") not in permitted]
    _check(not active and not allowed and (self_registered or not state.get("next_task")),
           "ACTIVE_TASK_CONFLICT:" + repr((active, allowed)))
    if self_registered:
        own_rows = [row for row in state.get("tasks", []) if row.get("task_id") == TASK]
        own_index = [row for row in index.get("task_packets", []) if row.get("task_id") == TASK]
        _check(len(own_rows) == len(own_index) == 1, "DOUBLE_WRITER_OR_PACKET_ROW")
        own = own_rows[0]
        packet = load_json(PACKET)
        writer = packet.get("writer", {})
        _check(own is not None and own.get("status") == "RUNNING"
               and own.get("pid") == os.getpid()
               and own.get("proc_start_ticks") == process_identity(os.getpid())["start_ticks"],
               "OWN_WRITER_IDENTITY_DRIFT")
        _check(writer.get("pid") == own["pid"]
               and writer.get("proc_start_ticks") == own["proc_start_ticks"]
               and writer.get("executor_epoch") == 1
               and isinstance(writer.get("fencing_token"), str) and len(writer["fencing_token"]) >= 32
               and own.get("run_signature") == packet.get("run_signature")
               and own_index[0].get("execution_allowed") is True
               and own_index[0].get("packet_sha256") == artifact_ref(PACKET)["sha256"],
               "OWN_PACKET_FENCING_DRIFT")


def _protected_snapshot() -> dict[str, dict[str, Any]]:
    # Historical pointer-to-target inconsistencies are reported as existing
    # facts.  The cleanup gate is equality of before/after bytes, not repair
    # or retroactive validation of another branch's historical receipts.
    pointer = load_json(PROTECTED_POINTERS[0])
    paths = list(PROTECTED_POINTERS)
    historical_refs: dict[str, dict[str, Any]] = {}
    for key in ("baseline", "diagnostic_result", "reviewed_video"):
        value = pointer.get(key)
        if isinstance(value, dict) and isinstance(value.get("path"), str):
            target = Path(value["path"])
            if target.is_relative_to(REPO_ROOT):
                paths.append(target)
                historical_refs[key] = value
            else:
                historical_refs[key] = {"reference": value, "outside_project": True}
        else:
            historical_refs[key] = {"reference": value, "incomplete": True}
    for relative in ("docs/current", "tasks/receipts", "src/chaoyang/ops",
                     "src/chaoyang/governance"):
        for path in (REPO_ROOT / relative).iterdir():
            if path.is_file() and any(token in path.name.lower()
                                      for token in ("manus", "0916", "robot100")):
                paths.append(path)
    snapshot = {str(path): artifact_ref(path) if path.is_file() else {"path": str(path), "missing": True}
                for path in sorted(set(paths))}
    for key, expected in historical_refs.items():
        actual = snapshot.get(str(expected.get("path"))) if isinstance(expected, dict) else None
        snapshot["historical_0916_ref:" + key] = {
            "declared": expected, "matches_current_bytes": actual == expected,
        }
    return snapshot


def _exact_file_check(row: tuple[str, int, str]) -> dict[str, Any]:
    relative, expected_bytes, expected_sha = row
    path = REPO_ROOT / relative
    _check(not any(part in {"..", ""} for part in Path(relative).parts), "UNSAFE_RELATIVE_PATH")
    _check(path.parent.is_dir() and not path.is_symlink(), "CACHE_PATH_NOT_REGULAR:" + relative)
    for ancestor in (path, *path.parents):
        if ancestor == REPO_ROOT:
            break
        _check(not ancestor.is_symlink(), "CACHE_SYMLINK_COMPONENT:" + relative)
    resolved = path.resolve(strict=True)
    _check(resolved == path and any(resolved.is_relative_to(root) for root in CACHE_ROOTS),
           "CACHE_REALPATH_OUTSIDE_ALLOWLIST:" + relative)
    metadata = path.lstat()
    _check(stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1
           and metadata.st_size == expected_bytes, "CACHE_STAT_DRIFT:" + relative)
    _check("license" not in path.name.lower(), "LICENSE_TARGET_FORBIDDEN:" + relative)
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        held = os.fstat(fd)
        _check((held.st_dev, held.st_ino, held.st_size) ==
               (metadata.st_dev, metadata.st_ino, metadata.st_size), "CACHE_INODE_RACE:" + relative)
        _check(_sha_fd(fd) == expected_sha, "CACHE_SHA_DRIFT:" + relative)
    finally:
        os.close(fd)
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", "--", relative],
                             cwd=REPO_ROOT, capture_output=True, check=False)
    ignored = subprocess.run(["git", "check-ignore", "-q", "--", relative],
                             cwd=REPO_ROOT, capture_output=True, check=False)
    _check(tracked.returncode == 1 and ignored.returncode == 0,
           "CACHE_GIT_STATE_DRIFT:" + relative)
    return {"path": relative, "bytes": expected_bytes, "sha256": expected_sha,
            "device": metadata.st_dev, "inode": metadata.st_ino}


def _runtime_reference_hits() -> list[str]:
    # Audit proof and this cleanup packet are not runtime consumers.  The
    # selected roots cover current navigation, governance, task routing and code.
    needles = tuple(text.encode() for row in ALLOWLIST for text in
                    (row[0], str(REPO_ROOT / row[0])))
    hits: list[str] = []
    for relative_root in RUNTIME_REF_ROOTS:
        root = REPO_ROOT / relative_root
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.is_symlink() or path in {PACKET, Path(__file__)}:
                continue
            if path.stat().st_size > 4_000_000:
                continue  # Large governance files need the separate exact grep below.
            if any(needle in path.read_bytes() for needle in needles):
                hits.append(str(path.relative_to(REPO_ROOT)))
    # Fixed-string rg also covers large text files under the current roots.
    search = subprocess.run(["rg", "-l", "-F", "-f", "/dev/stdin", *RUNTIME_REF_ROOTS],
                            input="\n".join(row[0] for row in ALLOWLIST), text=True,
                            cwd=REPO_ROOT, capture_output=True, check=False)
    _check(search.returncode in {0, 1}, "RUNTIME_REFERENCE_SCAN_FAILED")
    excluded = {str(PACKET.relative_to(REPO_ROOT)), str(Path(__file__).relative_to(REPO_ROOT))}
    hits.extend(line for line in search.stdout.splitlines() if line not in excluded)
    return sorted(set(hits))


def _process_hits() -> list[dict[str, Any]]:
    targets = {str(REPO_ROOT / row[0]) for row in ALLOWLIST}
    cache_roots = tuple(str(root) for root in CACHE_ROOTS)
    hits: list[dict[str, Any]] = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        identity = process_identity(int(proc.name))
        links = [proc / "cwd"]
        try:
            links.extend((proc / "fd").iterdir())
        except (FileNotFoundError, PermissionError):
            pass
        for link in links:
            try:
                destination = os.readlink(link).removesuffix(" (deleted)")
            except OSError:
                continue
            if destination in targets or (link.name == "cwd" and
                                          any(destination == root or destination.startswith(root + "/")
                                              for root in cache_roots)):
                hits.append({"pid": identity["pid"], "start_ticks": identity["start_ticks"],
                             "link": str(link), "target": destination})
    return hits


def _preflight(*, self_registered: bool) -> list[dict[str, Any]]:
    _no_other_active_task(self_registered=self_registered)
    checked = [_exact_file_check(row) for row in ALLOWLIST]
    _check(not _runtime_reference_hits(), "CURRENT_RUNTIME_REFERENCE_PRESENT")
    _check(not _process_hits(), "ACTIVE_PID_FD_CWD_REFERENCE")
    return checked


def _register(v1_ref: dict[str, Any], proof_ref: dict[str, Any],
              protected: dict[str, dict[str, Any]], checked: list[dict[str, Any]]) -> None:
    _check(not RECEIPT.exists() and not PACKET.exists() and not ROOT.exists(),
           "SINGLE_USE_TASK_ID_ALREADY_USED")
    state, index = copy.deepcopy(load_json(TASK_STATE_PATH)), _index_body()
    _check(not any(row.get("task_id") == TASK for row in state.get("tasks", [])),
           "TASK_ID_ALREADY_IN_LEDGER")
    _check(not any(row.get("task_id") == TASK for row in index.get("task_packets", [])),
           "TASK_ID_ALREADY_IN_INDEX")
    writer = {"pid": os.getpid(), "proc_start_ticks": process_identity(os.getpid())["start_ticks"],
              "executor_epoch": 1, "fencing_token": secrets.token_hex(24), "scope": TASK}
    inputs = {"v1_adoption_receipt": v1_ref, "media_proof": proof_ref,
              "protected_0916_manus": protected, "candidate_files": checked}
    signature = hashlib.sha256(canonical_bytes({"inputs": inputs, "code": artifact_ref(Path(__file__)),
                                                "schema": "exact78-v1-cache-cleanup-v1"})).hexdigest()
    packet = {
        "schema_version": "chaoyang-r22-task-packet-v1", "task_id": TASK,
        "objective": "Delete only four SHA-verified, rebuildable Exact78 cache files (213273 bytes)",
        "phase": "EXACT78_VISUAL_BASELINE_V1_CACHE_CLEANUP",
        "plan_revision": "EXACT78_VISUAL_BASELINE_V1_CLEANUP_20261008",
        "execution_revision": "EXACT78_VISUAL_BASELINE_V1_CACHE_CLEANUP_20261008",
        "session": None,
        "deadline_at": (datetime.now().astimezone() + timedelta(minutes=30)).isoformat(),
        "read_set": ["AGENTS.md", "docs/current/AI_WORK_ENTRY_ZH.md",
                     str(PROOF.relative_to(REPO_ROOT)), "tasks/current/INDEX.json",
                     "docs/governance/DOC_AUTHORITY_MAP.json"],
        "write_set": [*(row[0] for row in ALLOWLIST), str(ROOT.relative_to(REPO_ROOT)),
                      str(PACKET.relative_to(REPO_ROOT)), str(RECEIPT.relative_to(REPO_ROOT)),
                      "tasks/current/INDEX.json", "docs/governance"],
        "prerequisites": ["V1_ADOPTION_RECEIPT_PUBLISHED", "GOVERNANCE_PASS",
                          "NO_OTHER_ACTIVE_TASK", "EXACT_FILE_SHA_AND_REFERENCE_PROOF",
                          "0916_MANUS_PROTECTED_SHA_FROZEN"],
        "required_outputs": ["RESULT.json", "deletion_receipt", "terminal_task_state"],
        "budgets": {"workers_max": 1, "cpu_threads_total": 1, "gpu_count": 0,
                    "model_calls": 0, "render_calls": 0, "training_steps": 0,
                    "wall_seconds_max": 1800, "session_attempts_max": 1},
        "stop_conditions": ["BLOCKED_REFERENCE_PROOF", "SHA_OR_PATH_DRIFT",
                            "DOUBLE_WRITER", "ONE_ATTEMPT_TERMINAL"],
        "claim_limit": "Four generated cache files only; no baseline, model, media, data or branch authority change.",
        "writer": writer, "inputs": inputs, "run_signature": signature,
        "commands": ["PYTHONPATH=src /usr/local/bin/python -B -m "
                     "chaoyang.governance.cleanup_exact78_visual_v1_caches --v1-receipt <published V1 receipt>"],
    }
    _check(not _validate_packet(packet), "INVALID_TASK_PACKET")
    ROOT.mkdir(parents=True, exist_ok=False)
    PACKET.parent.mkdir(parents=True, exist_ok=False)
    atomic_json(ROOT / "WRITER.json", writer)
    atomic_json(PACKET, packet)
    now = now_iso()
    state["tasks"].append({"task_id": TASK, "phase": packet["phase"], "attempt": 1,
                           "status": "RUNNING", "session": "exact78_v1_cache_cleanup",
                           "updated_at": now, "heartbeat_at": now, "pid": writer["pid"],
                           "proc_start_ticks": writer["proc_start_ticks"], "gpu_id": None,
                           "task_packet": artifact_ref(PACKET), "run_signature": signature,
                           "quality": "NOT_APPLICABLE", "adoption": "NOT_APPLICABLE",
                           "review": "NOT_APPLICABLE"})
    state["next_task"] = {"task_id": TASK, "session": "exact78_v1_cache_cleanup",
                          "prerequisites": packet["prerequisites"],
                          "expected_resource": "CPU_ONE_THREAD", "stop_condition": "one terminal cleanup"}
    index["task_packets"].append({"task_id": TASK,
                                   "packet_path": str(PACKET.relative_to(REPO_ROOT)),
                                   "packet_sha256": artifact_ref(PACKET)["sha256"],
                                   "execution_class": "CURRENT_LEDGER_ROUTABLE",
                                   "execution_allowed": True})
    index.update(packet_revision="EXACT78_VISUAL_BASELINE_V1_CACHE_CLEANUP_RUNNING",
                 status="PASS", supersedes_index=artifact_ref(INDEX))
    publish_bundle(copy.deepcopy(load_json(AUTHORITY_PATH)), state,
                   event_type="EXACT78_VISUAL_BASELINE_V1_CACHE_CLEANUP_REGISTERED",
                   expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
                   generator_path=Path(__file__), task_packet_index_path=INDEX,
                   task_packet_index_value=index)


def _unlink_exact(row: tuple[str, int, str]) -> dict[str, Any]:
    # Recheck immediately before unlink.  Only the literal basename is removed;
    # there is no recursive deletion, wildcard or directory removal.
    checked = _exact_file_check(row)
    path = REPO_ROOT / row[0]
    parent_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                        | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        _check((before.st_dev, before.st_ino, before.st_size) ==
               (checked["device"], checked["inode"], checked["bytes"]),
               "CACHE_CHANGED_BEFORE_UNLINK:" + row[0])
        path.unlink()
    finally:
        os.close(parent_fd)
    _check(not path.exists() and not path.is_symlink(), "CACHE_STILL_PRESENT:" + row[0])
    return {"path": row[0], "bytes": row[1], "sha256_before": row[2]}


def _terminal(deleted: list[dict[str, Any]], protected_before: dict[str, Any],
              status: str, reason: str | None) -> str:
    try:
        protected_after = _protected_snapshot()
        if protected_after != protected_before:
            status, reason = "BLOCKED_REFERENCE_PROOF", "0916_MANUS_PROTECTED_SHA_CHANGED"
    except Exception as error:  # Preserve the deletion facts even when protection cannot be re-read.
        protected_after = None
        status, reason = "BLOCKED_REFERENCE_PROOF", "0916_MANUS_PROTECTED_RECHECK_FAILED:" + str(error)
    result = {"schema_version": "exact78-v1-cache-cleanup-result-v1", "task_id": TASK,
              "status": status, "reason": reason, "deleted": deleted,
              "deleted_count": len(deleted), "deleted_bytes": sum(row["bytes"] for row in deleted),
              "files_deleted": bool(deleted), "model_calls": 0, "render_calls": 0,
              "baseline_adoption_changed": False, "training_eligible": False,
              "protected_0916_manus_sha_unchanged": protected_after == protected_before,
              "historical_0916_reference_mismatches": [
                  key for key, value in protected_before.items()
                  if key.startswith("historical_0916_ref:") and not value["matches_current_bytes"]],
              "protected_before": protected_before, "protected_after": protected_after,
              "finished_at": now_iso()}
    atomic_json(ROOT / "RESULT.json", result)
    atomic_json(RECEIPT, {"schema_version": "exact78-v1-cache-cleanup-receipt-v1",
                          "task_id": TASK, "status": status, "result": artifact_ref(ROOT / "RESULT.json"),
                          "additional_deletion_authorized": False, "deleted": deleted,
                          "deleted_bytes": result["deleted_bytes"], "recorded_at": now_iso()})
    state, index = copy.deepcopy(load_json(TASK_STATE_PATH)), _index_body()
    row = next(item for item in state["tasks"] if item["task_id"] == TASK)
    row.update(status=status, pid=None, proc_start_ticks=None, updated_at=now_iso(),
               heartbeat_at=now_iso(), result=artifact_ref(ROOT / "RESULT.json"),
               terminal_receipt=artifact_ref(RECEIPT), quality="NOT_APPLICABLE",
               adoption="NOT_APPLICABLE", review="NOT_APPLICABLE")
    state["next_task"] = None
    for entry in index["task_packets"]:
        if entry.get("task_id") == TASK:
            entry["execution_allowed"] = False
    index.update(packet_revision="EXACT78_VISUAL_BASELINE_V1_CACHE_CLEANUP_TERMINAL",
                 status="PASS_NO_ACTIVE_TASKS", supersedes_index=artifact_ref(INDEX))
    publish_bundle(copy.deepcopy(load_json(AUTHORITY_PATH)), state,
                   event_type="EXACT78_VISUAL_BASELINE_V1_CACHE_CLEANUP_TERMINAL",
                   expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
                   generator_path=Path(__file__), task_packet_index_path=INDEX,
                   task_packet_index_value=index)
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-receipt", type=Path, required=True,
                        help="Published V1 adoption receipt under tasks/receipts")
    args = parser.parse_args()
    v1_ref = _validate_adoption_receipt(args.v1_receipt)
    proof_ref = _validate_proof()
    _validate_governance()
    _no_other_active_task(self_registered=False)
    protected = _protected_snapshot()
    checked = _preflight(self_registered=False)
    _register(v1_ref, proof_ref, protected, checked)
    deleted: list[dict[str, Any]] = []
    status, reason = "PASSED", None
    try:
        _validate_governance()
        _preflight(self_registered=True)
        for item in ALLOWLIST:
            _check(not _process_hits() and not _runtime_reference_hits(),
                   "REFERENCE_CHANGED_DURING_CLEANUP")
            _check(_protected_snapshot() == protected, "0916_MANUS_PROTECTED_SHA_CHANGED")
            deleted.append(_unlink_exact(item))
    except Exception as error:
        status, reason = "BLOCKED_REFERENCE_PROOF", f"{type(error).__name__}:{error}"
    finally:
        status = _terminal(deleted, protected, status, reason)
    return 0 if status == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
