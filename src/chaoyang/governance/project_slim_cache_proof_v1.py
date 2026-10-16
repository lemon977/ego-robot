"""Exact generated-cache inventory, signature checks and pidfd loop retirement."""
from __future__ import annotations
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import time

from chaoyang.governance.common import REPO_ROOT, artifact_ref, atomic_json, load_json, now_iso
from chaoyang.governance import publish_legacy_exact_delete_v1 as util

CACHE_ROOTS = (
    ".cache/s2/pycache", ".cache/pycache", "src/chaoyang/ops/__pycache__", "src/chaoyang/pipeline/__pycache__",
    "src/chaoyang/governance/__pycache__", "tests/__pycache__", "_run/caches/baseline_v1_source_release/pytest_pass1",
    "_run/caches/baseline_v1_source_release/pytest_final", "_run/caches/dual_branch_phase_v1/pytest_pass1",
    "_run/caches/legacy_exact_delete_v1/pytest_0001",
)
MACHINE = ("docs/governance/CURRENT_AUTHORITY_INDEX.json", "docs/governance/DOC_AUTHORITY_MAP.json",
           "docs/governance/ALGORITHM_CONTRACT.json", "docs/governance/CURRENT_BASELINE_REGISTRY_V2.json",
           "docs/governance/CURRENT_REGRESSION_MANIFEST.json", "tasks/current/INDEX.json", "baseline_v1/MANIFEST.json")
LOOPS = ((643811, 352439993, "scripts/takeover/autopilot_watchdog.sh", "120"),
         (919396, 364543469, "scripts/takeover/progress_pulse.sh", "900"))


def context():
    from chaoyang.governance import publish_project_slim_v1 as module
    return module


def original(path, root):
    if root.startswith("_run/caches/"):
        return None  # Whole completed pytest fixture tree, not source/input data.
    if path.suffix == ".pyc":
        stem = re.sub(r"\.cpython-\d+(?:\.opt-\d+)?\.pyc$", ".py", path.name)
        if stem == path.name: return None
        if root.startswith(".cache/"):
            source = Path("/")/path.relative_to(REPO_ROOT/root).parent/stem
        else:
            source = path.parent.parent/stem
    elif path.suffix in {".nbi", ".nbc"}:
        source = path.parent.parent/(path.name.split(".", 1)[0]+".py")
    else:
        return None
    return source if source.is_file() else None


def processes(names):
    result = util.process_scan(names)
    errors = []
    for _ in range(3):
        if not result["errors"]: break
        errors.append(result["errors"]); time.sleep(0.05); result = util.process_scan(names)
    result["transient_error_scans"] = errors
    return result


def root_cwd_holders(records, roots):
    directories = [str(REPO_ROOT/root) for root in roots]
    return [p for p in records if any(p["cwd"] == d or p["cwd"].startswith(d+"/") for d in directories)]


def audit(packet):
    module = context(); module.guard(packet); out, _, _ = module.paths(packet["task_id"])
    authority = load_json(REPO_ROOT/MACHINE[1])
    documents = set(MACHINE)|{str(Path(row["path"]).relative_to(REPO_ROOT)) for row in authority["documents"] if row["status"] == "CURRENT"}
    texts = {name: (REPO_ROOT/name).read_text() for name in sorted(documents)}
    roots, candidates, source_refs, preserved_links = [], [], {}, []
    for root in CACHE_ROOTS:
        directory = REPO_ROOT/root
        if directory.is_symlink() or directory.resolve() != directory or not directory.is_dir(): raise RuntimeError("CACHE_ROOT_DRIFT:"+root)
        size, allocated, count = 0, 0, 0
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                preserved_links.append({"path": str(path.relative_to(REPO_ROOT)), "target": os.readlink(path), "status": "PRESERVE_SYMLINK_NOT_FOLLOWED"})
                continue
            if path.is_dir() and not path.is_symlink(): continue
            name = str(path.relative_to(REPO_ROOT)); blockers = []
            if path.is_symlink() or not path.is_file() or path.resolve() != path:
                raise RuntimeError("NONREGULAR_CACHE_ENTRY:"+name)
            stat = path.stat(); ref = artifact_ref(path)
            if stat.st_nlink != 1: blockers.append("HARDLINK_NOT_UNIQUE_ALLOCATION")
            if name in packet["inputs"]["protected"]: blockers.append("PROTECTED_FILE")
            if root.startswith("_run/caches/"):
                if path.name.upper().startswith(("LICENSE", "NOTICE", "AGENTS")) or ".git" in path.parts:
                    blockers.append("NONFIXTURE_METADATA")
                regen = {"kind": "COMPLETED_PYTEST_GENERATED_FIXTURE", "source_test":
                         "tests/unit/test_"+("baseline_v1_source_release" if "baseline_v1_source_release" in root else
                                             "dual_branch_phase_v1" if "dual_branch_phase_v1" in root else "legacy_exact_delete_v1")+".py"}
            else:
                source = original(path, root)
                if source is None: blockers.append("SOURCE_ABSENT_OR_UNRECOGNIZED_CACHE")
                if source is not None and str(source) not in source_refs: source_refs[str(source)] = artifact_ref(source)
                regen = {"kind": "NUMBA_JIT_CACHE" if path.suffix in {".nbi", ".nbc"} else "PYTHON_BYTECODE_CACHE",
                         "source": source_refs.get(str(source)) if source else None}
            incoming = [doc for doc, text in texts.items() if name in text or str(path) in text]
            if incoming: blockers.append("CURRENT_EXACT_FILE_REFERENCE")
            candidates.append({"path": name, "root": root, "file": ref, "bytes": ref["bytes"], "allocated_bytes": stat.st_blocks*512,
                               "dev": stat.st_dev, "inode": stat.st_ino, "nlink": stat.st_nlink, "regeneration": regen,
                               "current_refs": incoming, "blockers": blockers})
            size += stat.st_size; allocated += stat.st_blocks*512; count += 1
        roots.append({"path": root, "files": count, "bytes": size, "allocated_file_bytes": allocated,
                      "current_root_tokens": [doc for doc, text in texts.items() if root in text or str(directory) in text]})
    names = [r["path"] for r in candidates]; git = util.git_rows(names); process = processes(names)
    for row in candidates:
        row["git"] = git[row["path"]]
        if row["git"]["tracked"]: row["blockers"].append("TRACKED_USER_FILE")
        if process["hits"][row["path"]]: row["blockers"].append("LIVE_FD_OR_CWD_OR_COMMAND")
        if process["errors"]: row["blockers"].append("PROCESS_SCAN_INCOMPLETE")
        directory = str(REPO_ROOT/row["root"])
        if any(p["cwd"] == directory or p["cwd"].startswith(directory+"/") for p in process["processes"]): row["blockers"].append("LIVE_ROOT_CWD")
        row["status"] = "ELIGIBLE_C1" if not row["blockers"] else "BLOCKED_REFERENCE_PROOF"
    proof = {"schema_version": "PROJECT_SLIM_CACHE_PROOF_V1", "task_id": packet["task_id"], "created_at": now_iso(), "roots": roots, "rows": candidates,
             "current_documents_scanned": len(documents), "process_scan": process, "preserved_symlinks": preserved_links,
             "eligible_files": sum(r["status"] == "ELIGIBLE_C1" for r in candidates),
             "eligible_bytes": sum(r["bytes"] for r in candidates if r["status"] == "ELIGIBLE_C1"),
             "eligible_allocated_bytes": sum(r["allocated_bytes"] for r in candidates if r["status"] == "ELIGIBLE_C1"),
             "blocked_files": sum(r["status"] != "ELIGIBLE_C1" for r in candidates),
             "license": {"notices": artifact_ref(REPO_ROOT/"THIRD_PARTY_NOTICES.md"), "policy": "Only derived caches/fixtures; original sources, license/NOTICE/vendor/weights remain. No redistribution or original-byte cache restoration claim."},
             "deletion_authorized": False, "claim_limit": "Exact cache scope only; no claim to complete archive/media/worktree reference proof."}
    atomic_json(out/"PROOF.json", proof)
    print(json.dumps({k: proof[k] for k in ("eligible_files", "eligible_bytes", "eligible_allocated_bytes", "blocked_files")}), flush=True)


def stop_loops(packet):
    module = context(); module.guard(packet); out, _, _ = module.paths(packet["task_id"])
    base = REPO_ROOT/"docs/plans/EXACT78_TAKEOVER_20261004"
    plan, state = load_json(base/"AUTOPILOT_PLAN.json"), load_json(base/"AUTOPILOT_STATE.json")
    if len(plan["steps"]) != 8 or any(state["steps"].get(s["id"], {}).get("status") != "DONE" for s in plan["steps"]): raise RuntimeError("OLD_PLAN_NOT_COMPLETE")
    parent = load_json(REPO_ROOT/"_run/current/exact78_takeover_b2_full_v13/attempts/attempt_0001/RESULT.json")
    if parent["status"] != "PASSED": raise RuntimeError("V13_NOT_SEALED")
    stopped = []
    def save(status):
        atomic_json(out/"STOP_LOOPS.json", {"status": status, "rows": stopped, "other_processes_signaled": 0, "manus_supervisor_preserved": True})
    for pid, ticks, script, seconds in LOOPS:
        before = util.identity(pid)
        if before is None:
            stopped.append({"pid": pid, "expected_startticks": ticks, "status": "ALREADY_ABSENT"}); save("PARTIAL"); continue
        if before["proc_start_ticks"] != ticks or before["argv"] != ["bash", script] or before["cwd"] != str(REPO_ROOT): raise RuntimeError("LOOP_IDENTITY_DRIFT")
        children = [util.identity(int(p.name)) for p in Path("/proc").iterdir() if p.name.isdigit()]
        children = [p for p in children if p and p["ppid"] == pid]
        if len(children) > 1 or any(p["argv"] != ["sleep", seconds] for p in children): raise RuntimeError("UNEXPECTED_LOOP_CHILD")
        handles = [(before, os.pidfd_open(pid))]
        try:
            handles.extend((p, os.pidfd_open(p["pid"])) for p in children)
            for p, _ in handles:
                live = util.identity(p["pid"])
                if not live or any(live[k] != p[k] for k in ("pid", "proc_start_ticks", "argv", "cwd")): raise RuntimeError("PROCESS_CHANGED_BEFORE_SIGNAL")
            entry = {"status": "VERIFIED_BEFORE_SIGNAL", "parent": before, "children": children, "signal": "SIGTERM_PIDFD", "signaled": []}
            stopped.append(entry); save("PARTIAL")
            for p, fd in handles:
                signal.pidfd_send_signal(fd, signal.SIGTERM); entry["signaled"].append({"pid": p["pid"], "proc_start_ticks": p["proc_start_ticks"]}); save("PARTIAL")
            for _ in range(50):
                if all((live := util.identity(p["pid"])) is None or live["state"] == "Z" or live["proc_start_ticks"] != p["proc_start_ticks"] for p, _ in handles): break
                time.sleep(0.1)
            else: raise RuntimeError("LOOP_DID_NOT_EXIT")
            entry["status"] = "STOPPED"; save("PARTIAL")
        finally:
            for _, fd in handles: os.close(fd)
    save("PASS")
    print("OLD_LOOPS_STOPPED", flush=True)


def delete(packet):
    module = context(); module.guard(packet); out, _, _ = module.paths(packet["task_id"])
    proof = load_json(Path(packet["inputs"]["proof"]["path"]))
    rows = [r for r in proof["rows"] if r["status"] == "ELIGIBLE_C1"]
    processes_now = processes([r["path"] for r in rows])
    if processes_now["errors"] or any(processes_now["hits"].values()): raise RuntimeError("FINAL_PROCESS_PROOF_BLOCKED")
    if root_cwd_holders(processes_now["processes"], {r["root"] for r in rows}): raise RuntimeError("FINAL_LIVE_ROOT_CWD")
    git_now = util.git_rows([r["path"] for r in rows])
    if any(r["tracked"] for r in git_now.values()): raise RuntimeError("CACHE_BECAME_GIT_TRACKED")
    authority = load_json(REPO_ROOT/MACHINE[1])
    documents = set(MACHINE)|{str(Path(r["path"]).relative_to(REPO_ROOT)) for r in authority["documents"] if r["status"] == "CURRENT"}
    texts = [(REPO_ROOT/name).read_text() for name in sorted(documents)]
    sources = {}
    for row in rows:
        name = row["path"]; path = REPO_ROOT/name
        if row["root"] not in CACHE_ROOTS or not path.is_relative_to(REPO_ROOT/row["root"]) or path.is_symlink() or path.resolve() != path: raise RuntimeError("TARGET_SCOPE_DRIFT")
        if artifact_ref(path) != row["file"] or path.stat().st_ino != row["inode"] or path.stat().st_nlink != 1: raise RuntimeError("CACHE_SIGNATURE_DRIFT")
        if any(name in text or str(path) in text for text in texts): raise RuntimeError("CURRENT_CACHE_REFERENCE_ADDED")
        ref = row["regeneration"].get("source")
        if ref and ref["path"] not in sources:
            if artifact_ref(ref["path"]) != ref: raise RuntimeError("REGENERATION_SOURCE_DRIFT")
            sources[ref["path"]] = ref
    receipt = {"schema_version": "PROJECT_SLIM_EXACT_DELETE_RECEIPT_V1", "task_id": packet["task_id"], "proof": packet["inputs"]["proof"],
               "files_deleted": 0, "bytes_deleted": 0, "allocated_bytes_deleted": 0, "deleted": [], "regenerable_not_original_byte_restore": True,
               "blocked_files_retained": proof["blocked_files"], "models_or_original_sources_deleted": False}
    journal = out/"DELETED.jsonl"
    with journal.open("x", encoding="utf-8") as log:
        try:
            for row in rows:
                path = REPO_ROOT/row["path"]
                if artifact_ref(path) != row["file"]: raise RuntimeError("TARGET_CHANGED_BEFORE_UNLINK")
                path.unlink()
                entry = {"path": row["path"], "sha256": row["file"]["sha256"], "bytes": row["bytes"], "allocated_bytes": row["allocated_bytes"], "regeneration": row["regeneration"]}
                receipt["deleted"].append(entry); receipt["files_deleted"] += 1; receipt["bytes_deleted"] += row["bytes"]; receipt["allocated_bytes_deleted"] += row["allocated_bytes"]
                log.write(json.dumps(entry)+"\n"); log.flush()
        finally: atomic_json(out/"DELETE_RECEIPT.json", receipt)
    print(json.dumps({k: receipt[k] for k in ("files_deleted", "bytes_deleted", "allocated_bytes_deleted", "blocked_files_retained")}), flush=True)
