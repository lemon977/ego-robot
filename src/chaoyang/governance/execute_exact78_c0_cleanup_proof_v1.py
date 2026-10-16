"""Registered read-only C0 proof for exact project-local cleanup candidates.

This worker has no deletion API. It only writes its own immutable proof/result,
receipt, and governance heartbeat/terminal projection after packet registration.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from chaoyang.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH,
    artifact_ref, atomic_json, canonical_bytes, load_json, now_iso,
    process_identity, publish_bundle, validate_artifact_ref,
)


TASK = "exact78_raw_to_robot_c0_cleanup_proof_v1_20260930"
SCRIPT = Path(__file__)
PACKET = REPO_ROOT / "tasks/current" / TASK / "TASK_PACKET.json"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
OUT = REPO_ROOT / "_run/current" / TASK / "attempts/attempt_0001"
RECEIPT = REPO_ROOT / "tasks/receipts/EXACT78_RAW_TO_ROBOT_C0_CLEANUP_PROOF_V1.json"
TEXT_SUFFIXES = {".json", ".jsonl", ".md", ".txt", ".toml", ".yaml", ".yml", ".py", ".sh", ".csv"}
REQUIRED_REFERENCE_GLOBS = frozenset({
    "docs/**/*.json", "docs/**/*.md", "tasks/**/*.json", "src/**/*.py",
    "scripts/**/*.py", "_run/current/**/*.json", "_run/current/**/*.md",
    "_run/maintenance/**/*.json", "_run/maintenance/**/*.md",
    "review/**/*.json", "review/**/*.md", ".cache/**/*.json",
})


def reference_scope_gaps(scope: dict[str, Any]) -> list[str]:
    """A scoped scan cannot silently stand in for project-wide reverse proof."""
    return sorted(REQUIRED_REFERENCE_GLOBS.difference(scope.get("file_globs", [])))


def inside_project(path: Path) -> str:
    resolved = path.resolve(strict=True)
    return str(resolved.relative_to(REPO_ROOT.resolve(strict=True)))


def verify_packet(path: Path) -> dict[str, Any]:
    if path.resolve(strict=True) != PACKET.resolve(strict=True):
        raise RuntimeError("WRONG_TASK_PACKET_PATH")
    packet = load_json(path)
    if packet.get("task_id") != TASK or packet.get("phase") != "EXACT78_RAW_TO_ROBOT_C0_CLEANUP_PROOF":
        raise RuntimeError("TASK_ID_OR_PHASE_CONFLICT")
    if packet.get("commands") != [
        "PYTHONPATH=src python -B -m chaoyang.governance.execute_exact78_c0_cleanup_proof_v1 --packet tasks/current/exact78_raw_to_robot_c0_cleanup_proof_v1_20260930/TASK_PACKET.json"]:
        raise RuntimeError("PACKET_COMMAND_CONFLICT")
    if packet.get("runtime_code") != [artifact_ref(SCRIPT)]:
        raise RuntimeError("RUNTIME_CODE_SHA_DRIFT")
    manifest_ref = packet.get("inputs", {}).get("candidate_manifest")
    if not isinstance(manifest_ref, dict) or validate_artifact_ref(manifest_ref):
        raise RuntimeError("CANDIDATE_MANIFEST_SHA_DRIFT")
    manifest_path = Path(manifest_ref["path"])
    if inside_project(manifest_path) != "docs/plans/EXACT78_RAW_TO_ROBOT_V1_20260930/C0_CANDIDATES_V1.json":
        raise RuntimeError("WRONG_CANDIDATE_MANIFEST")
    signature = hashlib.sha256(canonical_bytes({
        "inputs": packet["inputs"], "runtime_code": packet["runtime_code"],
        "reference_scan_scope": packet["reference_scan_scope"],
        "schema": "exact78-c0-proof-v1", "weights": "NOT_CONSUMED",
        "calibration": "NOT_CONSUMED",
    })).hexdigest()
    if signature != packet.get("run_signature"):
        raise RuntimeError("RUN_SIGNATURE_DRIFT")
    writer = packet.get("writer", {})
    identity = process_identity(os.getpid())
    if (writer.get("pid") != os.getpid() or
            writer.get("proc_start_ticks") != identity["start_ticks"] or
            not writer.get("fencing_token") or writer.get("executor_epoch") != 1):
        raise RuntimeError("WRITER_IDENTITY_OR_TOKEN_DRIFT")
    if load_json(OUT / "WRITER.json") != writer:
        raise RuntimeError("WRITER_FILE_DRIFT")
    index = load_json(INDEX)
    rows = [row for row in index["task_packets"] if row["task_id"] == TASK]
    if len(rows) != 1 or not rows[0].get("execution_allowed") or rows[0]["packet_sha256"] != artifact_ref(PACKET)["sha256"]:
        raise RuntimeError("PACKET_NOT_CURRENTLY_ROUTABLE")
    state = load_json(TASK_STATE_PATH)
    active = [row for row in state["tasks"] if row["status"] in ("RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE")]
    if (len(active) != 1 or active[0]["task_id"] != TASK or
            active[0].get("run_signature") != signature or active[0].get("pid") != os.getpid()):
        raise RuntimeError("DOUBLE_WRITER_OR_STATE_CONFLICT")
    if datetime.fromisoformat(packet["deadline_at"]) <= datetime.now().astimezone():
        raise RuntimeError("DEADLINE_EXPIRED")
    return packet


def heartbeat(packet: dict[str, Any]) -> None:
    verify_packet(PACKET)
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    row = next(item for item in state["tasks"] if item["task_id"] == TASK)
    row.update(heartbeat_at=now_iso(), updated_at=now_iso())
    publish_bundle(load_json(AUTHORITY_PATH), state,
        event_type="EXACT78_C0_CLEANUP_PROOF_HEARTBEAT",
        expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
        generator_path=SCRIPT)


def candidate_rows(packet: dict[str, Any]) -> list[dict[str, Any]]:
    manifest = load_json(Path(packet["inputs"]["candidate_manifest"]["path"]))
    if manifest.get("project_root") != str(REPO_ROOT) or manifest.get("schema_version") != "exact78-c0-project-local-candidates-v1":
        raise RuntimeError("CANDIDATE_MANIFEST_SCOPE_CONFLICT")
    rows = []
    seen = set()
    for item in manifest["targets"]:
        relative = item["path"]
        if relative in seen or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise RuntimeError(f"NONEXACT_OR_DUPLICATE_TARGET:{relative}")
        seen.add(relative)
        path = REPO_ROOT / relative
        if path.is_symlink() or not path.is_file() or inside_project(path) != relative:
            raise RuntimeError(f"TARGET_NOT_REGULAR_PROJECT_FILE:{relative}")
        if not (relative.startswith("_run/maintenance/") or relative.startswith(".cache/")):
            raise RuntimeError(f"TARGET_OUTSIDE_C0_ALLOWLIST:{relative}")
        ref = artifact_ref(path)
        rows.append({"path": relative, "kind": item["kind"], "file": ref,
                     "stat_mode": oct(path.stat().st_mode), "symlink": False})
    if not rows or len(rows) > 100:
        raise RuntimeError("CANDIDATE_COUNT_OUT_OF_BOUNDS")
    return rows


def scan_references(packet: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    scope = packet["reference_scan_scope"]
    scope_gaps = reference_scope_gaps(scope)
    max_file = int(scope["max_text_file_bytes"])
    max_total = int(scope["max_total_text_bytes"])
    max_files = int(scope["max_reference_files"])
    skip_files = set(scope.get("skip_files", [])) | {row["path"] for row in candidates}
    tokens = {}
    for row in candidates:
        rel = row["path"]
        tokens[rel] = (rel, str(REPO_ROOT / rel), Path(rel).name, row["file"]["sha256"])
    hits: dict[str, list[dict[str, str]]] = {row["path"]: [] for row in candidates}
    errors, skipped, scanned, total_bytes = [], [], 0, 0
    files: dict[str, tuple[Path, int]] = {}
    for pattern in scope["file_globs"]:
        if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
            raise RuntimeError(f"UNSAFE_REFERENCE_GLOB:{pattern}")
        for path in REPO_ROOT.glob(pattern):
            rel = str(path.relative_to(REPO_ROOT))
            if rel in skip_files or rel.startswith(str(OUT.relative_to(REPO_ROOT)) + "/"):
                continue
            if path.is_symlink() or not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            try:
                size = path.stat().st_size
            except OSError as error:
                errors.append(f"PREFLIGHT_STAT_ERROR:{rel}:{type(error).__name__}")
                continue
            files[rel] = (path, size)
    preflight_bytes = sum(size for _, size in files.values())
    if len(files) > max_files or preflight_bytes > max_total:
        return {"hits": hits, "errors": [*errors, "PREFLIGHT_CAPACITY_EXCEEDED"],
                "scope_gaps": scope_gaps,
                "skipped": [{"reason": "PREFLIGHT_CAPACITY_EXCEEDED", "files": len(files),
                             "bytes": preflight_bytes}],
                "preflight_file_count": len(files), "preflight_bytes": preflight_bytes,
                "scanned_text_files": 0, "scanned_text_bytes": 0,
                "scanned_file_refs": [], "scope_complete": False}
    scanned_refs = []
    last_beat, last_summary, summary_count = time.monotonic(), time.monotonic(), 0
    for rel, (path, preflight_size) in sorted(files.items()):
        if preflight_size > max_file:
            skipped.append({"path": rel, "reason": "TEXT_FILE_TOO_LARGE", "bytes": preflight_size})
            continue
        try:
            raw = path.read_bytes()
            content = raw.decode("utf-8")
        except (OSError, UnicodeError) as error:
            errors.append(f"SCAN_READ_ERROR:{rel}:{type(error).__name__}")
            continue
        if len(raw) != preflight_size:
            errors.append(f"CONCURRENT_SIZE_CHANGE:{rel}")
        scanned += 1
        total_bytes += len(raw)
        scanned_refs.append({"path": str(path), "bytes": len(raw),
                             "sha256": hashlib.sha256(raw).hexdigest()})
        for target, variants in tokens.items():
            match = next((token for token in variants if token in content), None)
            if match:
                hits[target].append({"document": rel, "matched_token": match})
        if time.monotonic() - last_beat >= 55:
            heartbeat(packet)
            last_beat = time.monotonic()
        if time.monotonic() - last_summary >= 600:
            summary_count += 1
            summary = OUT / f"RESUME_SUMMARY_{summary_count:04d}.json"
            if not summary.exists():
                atomic_json(summary, {"schema_version": "exact78-c0-resume-summary-v1",
                    "scanned_text_files": scanned, "scanned_text_bytes": total_bytes,
                    "errors": len(errors), "skipped": len(skipped), "heartbeat_at": now_iso()})
            last_summary = time.monotonic()
    return {"hits": hits, "errors": errors, "skipped": skipped,
            "scope_gaps": scope_gaps,
            "preflight_file_count": len(files), "preflight_bytes": preflight_bytes,
            "scanned_text_files": scanned, "scanned_text_bytes": total_bytes,
            "scanned_file_refs": scanned_refs,
            "scope_complete": not errors and not skipped and not scope_gaps}


def process_holders(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    paths = {row["path"]: REPO_ROOT / row["path"] for row in candidates}
    holders: dict[str, list[dict[str, Any]]] = {key: [] for key in paths}
    errors = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        pid = int(proc.name)
        identity = process_identity(pid)
        if not identity["alive"]:
            continue
        handles = []
        for label, handle in [("cwd", proc / "cwd")]:
            try:
                handles.append((label, os.readlink(handle)))
            except (FileNotFoundError, ProcessLookupError):
                continue
            except OSError as error:
                errors.append(f"PID_{pid}_{label}_{type(error).__name__}")
        try:
            fds = list((proc / "fd").iterdir())
        except (FileNotFoundError, ProcessLookupError):
            continue
        except OSError as error:
            errors.append(f"PID_{pid}_FD_LIST_{type(error).__name__}")
            fds = []
        for fd in fds:
            try:
                handles.append((f"fd/{fd.name}", os.readlink(fd)))
            except (FileNotFoundError, ProcessLookupError):
                continue
            except OSError as error:
                errors.append(f"PID_{pid}_FD_{type(error).__name__}")
        for rel, path in paths.items():
            absolute = str(path)
            parent = str(path.parent) + os.sep
            matched = [{"kind": kind, "path": value} for kind, value in handles
                       if value == absolute or value == absolute + " (deleted)"
                       or (kind == "cwd" and value.startswith(parent))]
            if matched:
                holders[rel].append({"pid": pid, "proc_start_ticks": identity["start_ticks"],
                                     "handles": matched})
    return {"holders": holders, "errors": errors}


def git_state(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    status = subprocess.run(["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
                            cwd=REPO_ROOT, capture_output=True, check=False)
    tracked = subprocess.run(["git", "ls-files", "-z"], cwd=REPO_ROOT, capture_output=True, check=False)
    if status.returncode or tracked.returncode:
        return {"status": "UNKNOWN", "error": "git status or ls-files failed"}
    dirty_text = status.stdout.decode("utf-8", "replace")
    tracked_set = set(tracked.stdout.decode("utf-8", "replace").split("\0"))
    rows = {}
    for row in candidates:
        rel = row["path"]
        ignore = subprocess.run(["git", "check-ignore", "-v", "--", rel], cwd=REPO_ROOT,
                                capture_output=True, text=True, check=False)
        rows[rel] = {"tracked": rel in tracked_set, "dirty_or_untracked": rel in dirty_text,
                     "ignored": ignore.returncode == 0,
                     "ignore_rule": ignore.stdout.strip() if ignore.returncode == 0 else None}
    return {"status": "OK", "rows": rows,
            "limitation": "Git does not enumerate ignored files; exact candidates were enumerated from the frozen manifest."}


def build_proof(packet: dict[str, Any]) -> dict[str, Any]:
    candidates = candidate_rows(packet)
    references = scan_references(packet, candidates)
    processes = process_holders(candidates)
    git = git_state(candidates)
    notices_path = REPO_ROOT / "THIRD_PARTY_NOTICES.md"
    license_sources = {"third_party_notices": artifact_ref(notices_path) if notices_path.is_file() else "ABSENT",
                       "project_license_files": [artifact_ref(path) for path in sorted(REPO_ROOT.glob("LICENSE*")) if path.is_file()]}
    rows = []
    for candidate in candidates:
        rel = candidate["path"]
        refs = references["hits"][rel]
        holders = processes["holders"][rel]
        git_row = git.get("rows", {}).get(rel)
        try:
            target_stable = artifact_ref(REPO_ROOT / rel) == candidate["file"]
        except (OSError, ValueError):
            target_stable = False
        current_refs = [ref for ref in refs if ref["document"].startswith("docs/current/")
                        or ref["document"].startswith("docs/governance/")
                        or ref["document"] == "tasks/current/INDEX.json"]
        historical_refs = [ref for ref in refs if ref not in current_refs]
        cache_provenance = {"regenerable": False, "tool": None, "evidence": []}
        if candidate["kind"] == "generated_cache" and rel in {
                ".cache/mpl/fontlist-v390.json", ".cache/xdg/matplotlib/fontlist-v390.json"}:
            try:
                cache = load_json(REPO_ROOT / rel)
                if (cache.get("_version") == 390 and isinstance(cache.get("afmlist"), list)
                        and isinstance(cache.get("ttflist"), list)
                        and all(isinstance(item.get("fname"), str)
                                for item in cache["afmlist"] + cache["ttflist"])):
                    cache_provenance = {"regenerable": True,
                        "tool": "Matplotlib FontManager JSON cache",
                        "evidence": ["fontlist-v390 filename", "_version=390",
                                     "afmlist/ttflist contain font path metadata, not font binaries"]}
            except (OSError, ValueError, TypeError, AttributeError):
                pass
        license_status = ("NO_EMBEDDED_FONT_BINARY_REGENERABLE_METADATA_CACHE"
                          if cache_provenance["regenerable"] else
                          "UNKNOWN_NO_PER_ARTIFACT_RIGHTS_PROOF")
        blockers = []
        if current_refs:
            blockers.append("CURRENT_REFERENCE")
        if historical_refs:
            blockers.append("HISTORICAL_REFERENCE")
        if holders:
            blockers.append("LIVE_PID_FD_OR_CWD")
        if not target_stable:
            blockers.append("TARGET_SHA_DRIFT_DURING_PROOF")
        if not references["scope_complete"]:
            blockers.append("REFERENCE_SCAN_INCOMPLETE")
        if processes["errors"]:
            blockers.append("PROCESS_SCAN_INCOMPLETE")
        if not git_row or git["status"] != "OK" or git_row["dirty_or_untracked"]:
            blockers.append("GIT_STATE_UNPROVEN_OR_DIRTY")
        if not cache_provenance["regenerable"]:
            blockers.append("LICENSE_OR_PROVENANCE_UNVERIFIED")
        eligible_review = (cache_provenance["regenerable"] and target_stable and not refs and not holders
                           and references["scope_complete"] and not processes["errors"]
                           and git["status"] == "OK" and bool(git_row)
                           and not git_row["dirty_or_untracked"])
        rows.append({**candidate, "current_refs": current_refs, "historical_refs": historical_refs,
            "live_holders": holders, "git": git_row, "license_status": license_status,
            "target_sha_stable_during_proof": target_stable,
            "license_evidence": license_sources,
            "cache_provenance": cache_provenance,
            "proof_status": "MUST_KEEP" if current_refs or holders else
                            "ELIGIBLE_C1_REVIEW" if eligible_review else "BLOCKED_REFERENCE_PROOF",
            "blockers": sorted(set(blockers)), "c1_delete_authorized": False})
    return {"schema_version": "exact78-c0-project-local-proof-v1", "task_id": TASK,
        "status": "AUDIT_ONLY", "candidate_manifest": packet["inputs"]["candidate_manifest"],
        "governance_revision_at_scan": load_json(RECEIPT_PATH)["governance_revision"],
        "scope": packet["reference_scan_scope"],
        "reference_scan": {key: value for key, value in references.items() if key != "hits"},
        "process_scan_errors": processes["errors"], "git": {key: value for key, value in git.items() if key != "rows"},
        "license_sources": license_sources,
        "target_count": len(rows), "total_bytes": sum(row["file"]["bytes"] for row in rows),
        "rows": rows, "deleted_count": 0, "c1_delete_authorized": False,
        "claim_limit": packet["claim_limit"]}


def finish(packet: dict[str, Any], status: str, outputs: dict[str, Path], reason: str | None = None) -> dict:
    verify_packet(PACKET)
    if status not in ("PASSED", "FAILED_RUNTIME_FINAL") or (OUT / "RESULT.json").exists() or RECEIPT.exists():
        raise RuntimeError("INVALID_OR_DUPLICATE_TERMINAL_RESULT")
    result = {"schema_version": "exact78-c0-proof-result-v1", "task_id": TASK,
        "status": status, "execution_complete": status == "PASSED", "reason": reason,
        "outputs": {name: artifact_ref(path) for name, path in outputs.items()},
        "deleted_count": 0, "deletion_authorized": False,
        "next_action": "Review per-target blockers; register separate C1 exact-target packet only after proof is complete",
        "claim_limit": packet["claim_limit"]}
    atomic_json(OUT / "RESULT.json", result)
    atomic_json(RECEIPT, {"schema_version": "exact78-c0-proof-receipt-v1", "task_id": TASK,
        "status": status, "result": artifact_ref(OUT / "RESULT.json"),
        "deletion_authorized": False, "claim_limit": packet["claim_limit"]})
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    row = next(item for item in state["tasks"] if item["task_id"] == TASK)
    row.update(status=status, pid=None, proc_start_ticks=None, updated_at=now_iso(),
               heartbeat_at=now_iso(), result=artifact_ref(OUT / "RESULT.json"),
               terminal_receipt=artifact_ref(RECEIPT), quality="C0_AUDIT_ONLY",
               adoption="NOT_ADOPTED", review="MACHINE_AUDITED")
    state["next_task"] = None
    index = {key: copy.deepcopy(value) for key, value in load_json(INDEX).items()
             if key not in {"governance_revision", "generation_id", "generated_at", "supersedes_index"}}
    for item in index["task_packets"]:
        if item["task_id"] == TASK:
            item["execution_allowed"] = False
    index.update(packet_revision="EXACT78_C0_PROOF_V1_TERMINAL",
        plan_revision=packet["plan_revision"], execution_revision=packet["execution_revision"],
        status="PASS_NO_ACTIVE_TASKS", supersedes_index=artifact_ref(INDEX))
    published = publish_bundle(load_json(AUTHORITY_PATH), state,
        event_type=f"EXACT78_C0_PROOF_{status}",
        expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
        generator_path=SCRIPT, task_packet_index_path=INDEX, task_packet_index_value=index)
    return {"status": status, "governance_revision": published["governance_revision"],
            "result": artifact_ref(OUT / "RESULT.json")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", required=True, type=Path)
    args = parser.parse_args()
    packet = verify_packet(args.packet)
    try:
        proof = build_proof(packet)
        atomic_json(OUT / "C0_PROOF.json", proof)
        print(json.dumps(finish(packet, "PASSED", {"proof": OUT / "C0_PROOF.json"}), ensure_ascii=False))
        return 0
    except Exception as error:
        outputs = {"proof": OUT / "C0_PROOF.json"} if (OUT / "C0_PROOF.json").exists() else {}
        print(json.dumps(finish(packet, "FAILED_RUNTIME_FINAL", outputs,
            f"{type(error).__name__}:{error}"), ensure_ascii=False), file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
