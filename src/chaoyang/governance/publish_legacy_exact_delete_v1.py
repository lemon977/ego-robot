"""Two bounded maintenance packets: proof, then exact recoverable deletion."""
from __future__ import annotations

import copy
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import secrets
import select
import signal
import subprocess
import sys
import tarfile
import time

from chaoyang.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH, artifact_ref,
    atomic_json, canonical_bytes, load_json, now_iso, publish_bundle,
)
from chaoyang.governance.publish_project_branch_convergence_v1 import index_body, refs, validate_current
from chaoyang.governance.register_single_task_packet import _validate_packet

PLAN = "docs/plans/LEGACY_EXACT_DELETE_V1_20261008"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
SPEC = REPO_ROOT / PLAN / "TARGETS_V2.json"
REPORT = "docs/current/LEGACY_CLEANUP_V1_ZH.md"
EDITS = ("AGENTS.md", "docs/current/AI_WORK_ENTRY_ZH.md",
         "docs/current/visuals/0915_FOUNDATIONSTEREO_ENCODED_DOMAIN_CANARY_V1/README_ZH.md",
         "docs/governance/RETIRED_OPERATION_ENTRYPOINTS.json")
CURRENT_MACHINE = ("docs/governance/ALGORITHM_CONTRACT.json", "docs/governance/DOC_AUTHORITY_MAP.json",
                   "docs/governance/CURRENT_BASELINE_REGISTRY_V2.json", "docs/governance/CURRENT_REGRESSION_MANIFEST.json")
ACTIVE = {"RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE"}


def identity(pid):
    proc = Path("/proc") / str(pid)
    try:
        fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
        argv = (proc / "cmdline").read_bytes().decode().rstrip("\0").split("\0")
        return {"pid": pid, "proc_start_ticks": int(fields[19]), "ppid": int(fields[1]),
                "state": fields[0], "argv": argv, "cwd": os.readlink(proc / "cwd")}
    except (FileNotFoundError, ProcessLookupError):
        return None


def paths(task):
    return (REPO_ROOT / f"_run/current/{task}/attempts/attempt_0001",
            REPO_ROOT / f"tasks/current/{task}/TASK_PACKET.json",
            REPO_ROOT / f"tasks/receipts/{task.upper()}.json")


def protected():
    parent = REPO_ROOT / "_run/current/baseline_v1_source_release_20261008/attempts/attempt_0001/RESULT.json"
    value = copy.deepcopy(load_json(parent)["protected_after"])
    value.update(refs((str(parent.relative_to(REPO_ROOT)), "baseline_v1/MANIFEST.json")))
    for path in sorted((REPO_ROOT / "baseline_v1").rglob("*")):
        if path.is_file():
            value[str(path.relative_to(REPO_ROOT))] = artifact_ref(path)
    if refs(tuple(value)) != value:
        raise RuntimeError("PROTECTED_SHA_DRIFT")
    return value


def register(task, proof=None):
    validate_current()
    state, index = copy.deepcopy(load_json(TASK_STATE_PATH)), index_body()
    if state.get("next_task") or any(r.get("execution_allowed") for r in index["task_packets"]):
        raise RuntimeError("ACTIVE_TASK_CONFLICT")
    out, packet_path, receipt = paths(task)
    if any(p.exists() for p in (out, packet_path, receipt)):
        raise RuntimeError("FRESH_TASK_REQUIRED")
    keep = protected()
    writer = {**{k: identity(os.getpid())[k] for k in ("pid", "proc_start_ticks")},
              "executor_epoch": 1, "fencing_token": secrets.token_hex(24), "scope": task}
    inputs = {"protected": keep, "spec": artifact_ref(SPEC), "plan": artifact_ref(REPO_ROOT/PLAN/"00_EXECUTION_ZH.md"),
              "publisher": artifact_ref(Path(__file__)), "edit_before": refs(EDITS), "proof": proof}
    signature = hashlib.sha256(canonical_bytes(inputs)).hexdigest()
    targets = [r["path"] for r in load_json(SPEC)["targets"]]
    packet = {"schema_version": "chaoyang-r22-task-packet-v1", "task_id": task,
        "objective": "Prove and retire only exact unselected legacy code/docs; preserve selected V1 runtime",
        "phase": "CPU_EXACT_DELETE" if proof else "CPU_EXACT_DELETE_PROOF", "plan_revision": "LEGACY_EXACT_DELETE_V1_20261008",
        "execution_revision": task.upper(), "session_id": None,
        "deadline_at": (datetime.now().astimezone()+timedelta(hours=1)).isoformat(),
        "read_set": ["AGENTS.md", "docs/current/AI_WORK_ENTRY_ZH.md", PLAN+"/00_EXECUTION_ZH.md", *CURRENT_MACHINE[:3],
                     "baseline_v1/MANIFEST.json", str(SPEC.relative_to(REPO_ROOT))],
        "expanded_read_set": [*keep, *CURRENT_MACHINE, *EDITS, "src", "tests", "scripts", "docs", "tasks", "configs", "contracts", "/proc"],
        "read_expansion_reason": "Exact current/import/command/FD reverse-reference deletion proof; read-only scans.",
        "write_set": [str(out.relative_to(REPO_ROOT)), str(packet_path.relative_to(REPO_ROOT)), str(receipt.relative_to(REPO_ROOT)),
                      "tasks/current/INDEX.json", "docs/governance", PLAN, "_run/caches/legacy_exact_delete_v1",
                      *([*targets, *EDITS, REPORT, "tests/unit/test_legacy_exact_delete_v1.py"] if proof else [])],
        "prerequisites": ["GOVERNANCE_PASS", "NO_ACTIVE_TASK", "V1_PROTECTED_SHA_PINNED"],
        "required_outputs": ["RESULT.json", "PROOF.json"] if not proof else ["RESULT.json", "DELETE_RECEIPT.json", "BACKUP.tar.gz", "REGRESSION.json"],
        "budgets": {"workers_max": 1, "cpu_threads_total": 2, "gpu_count": 0, "model_calls_max": 0,
                    "training_steps": 0, "wall_seconds_max": 3600, "session_attempts_max": 1},
        "stop_conditions": ["source_sha_drift", "live_reference", "double_writer", "backup_verification_failure", "deadline"],
        "claim_limit": "Maintenance only; no algorithm, source release, V1 pointer, numeric gate, model, video or qualification change.",
        "writer": writer, "inputs": inputs, "run_signature": signature,
        "commands": [f"{sys.executable} -B -m chaoyang.governance.publish_legacy_exact_delete_v1" +
                     (" --c1-only" if sys.argv[1:] == ["--c1-only"] else "")]}
    if errors := _validate_packet(packet):
        raise RuntimeError(str(errors))
    out.mkdir(parents=True); packet_path.parent.mkdir(parents=True)
    atomic_json(out/"WRITER.json", writer); atomic_json(packet_path, packet)
    state["tasks"].append({"task_id": task, "phase": packet["phase"], "attempt": 1, "status": "RUNNING", "session": "exact_cleanup",
        "updated_at": now_iso(), "heartbeat_at": now_iso(), "pid": writer["pid"], "proc_start_ticks": writer["proc_start_ticks"],
        "gpu_id": None, "task_packet": artifact_ref(packet_path), "run_signature": signature, "quality": "NOT_APPLICABLE",
        "adoption": "BASELINES_UNCHANGED", "review": "NOT_APPLICABLE"})
    state["next_task"] = {"task_id": task, "session": "exact_cleanup", "prerequisites": packet["prerequisites"],
                          "expected_resource": "CPU_TWO_THREADS", "stop_condition": "finite terminal publication"}
    index["task_packets"].append({"task_id": task, "packet_path": str(packet_path.relative_to(REPO_ROOT)),
        "packet_sha256": artifact_ref(packet_path)["sha256"], "execution_class": "CURRENT_LEDGER_ROUTABLE", "execution_allowed": True})
    index.update(status="PASS", supersedes_index=artifact_ref(INDEX))
    publish_bundle(copy.deepcopy(load_json(AUTHORITY_PATH)), state, event_type="EXACT_CLEANUP_REGISTERED",
        expected_revision=load_json(RECEIPT_PATH)["governance_revision"], generator_path=Path(__file__),
        task_packet_index_path=INDEX, task_packet_index_value=index)
    return packet


def guard(packet):
    task = packet["task_id"]
    out, pp, _ = paths(task)
    ident = identity(os.getpid())
    if (not ident or any(ident[k] != packet["writer"][k] for k in ("pid", "proc_start_ticks"))
            or load_json(out/"WRITER.json") != packet["writer"] or load_json(pp) != packet):
        raise RuntimeError("WRITER_OR_PACKET_DRIFT")
    index = load_json(INDEX)
    allowed = [r for r in index["task_packets"] if r.get("execution_allowed")]
    live = [r for r in load_json(TASK_STATE_PATH)["tasks"] if r["status"] in ACTIVE]
    if (len(allowed) != 1 or allowed[0]["task_id"] != task or allowed[0]["packet_sha256"] != artifact_ref(pp)["sha256"]
            or len(live) != 1 or live[0]["task_id"] != task or live[0].get("run_signature") != packet["run_signature"]):
        raise RuntimeError("DOUBLE_WRITER_OR_ROUTING_CONFLICT")
    if datetime.now().astimezone() >= datetime.fromisoformat(packet["deadline_at"]):
        raise RuntimeError("DEADLINE")
    if refs(tuple(packet["inputs"]["protected"])) != packet["inputs"]["protected"]:
        raise RuntimeError("PROTECTED_SHA_DRIFT")
    for key in ("spec", "publisher", "plan"):
        if artifact_ref(packet["inputs"][key]["path"]) != packet["inputs"][key]:
            raise RuntimeError("INPUT_SIGNATURE_DRIFT:"+key)


def heartbeat(packet):
    guard(packet)
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    next(r for r in state["tasks"] if r["task_id"] == packet["task_id"]).update(updated_at=now_iso(), heartbeat_at=now_iso())
    publish_bundle(copy.deepcopy(load_json(AUTHORITY_PATH)), state, event_type="EXACT_CLEANUP_HEARTBEAT",
        expected_revision=load_json(RECEIPT_PATH)["governance_revision"], generator_path=Path(__file__))


def git_rows(names):
    env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}
    def run(args):
        return subprocess.run(["git", *args], cwd=REPO_ROOT, env=env, capture_output=True, check=True).stdout
    tracked = set(run(["ls-files", "-z"]).decode().split("\0"))
    raw = run(["status", "--porcelain=v1", "-z", "--untracked-files=all"]).decode().split("\0")
    status, i = {}, 0
    while i < len(raw) and raw[i]:
        value = raw[i]; code, name = value[:2], value[3:]; status[name] = code; i += 1
        if "R" in code or "C" in code:
            status[raw[i]] = code; i += 1
    return {name: {"tracked": name in tracked, "porcelain": status.get(name, "  ") if name in tracked else status.get(name, "IGNORED_OR_UNTRACKED")}
            for name in names}


def process_scan(names):
    hits, errors, processes = {n: [] for n in names}, [], []
    for proc in sorted(Path("/proc").iterdir()):
        if not proc.name.isdigit(): continue
        pid = int(proc.name)
        try:
            before = identity(pid)
            if not before: continue
            handles = [("cwd", before["cwd"])]
            for fd in (proc/"fd").iterdir():
                try: handles.append(("fd/"+fd.name, os.readlink(fd)))
                except (FileNotFoundError, ProcessLookupError): pass
            after = identity(pid)
            if not after or before["proc_start_ticks"] != after["proc_start_ticks"]: continue
            processes.append(before)
            for name in names:
                path = REPO_ROOT/name; absolute = str(path)
                module = name.removeprefix("src/").removesuffix(".py").replace("/", ".")
                match = [{"kind": k, "path": v} for k, v in handles if v in (absolute, absolute+" (deleted)")
                         or (k == "cwd" and (v == str(path.parent) or v.startswith(str(path.parent)+"/")))]
                if absolute in before["argv"] or name in before["argv"] or (name.startswith("src/") and module in before["argv"]):
                    match.append({"kind": "cmdline", "path": str(before["argv"])})
                if match: hits[name].append({**before, "matches": match})
        except (FileNotFoundError, ProcessLookupError): pass
        except OSError as exc: errors.append({"pid": pid, "error": repr(exc)})
    return {"hits": hits, "errors": errors, "processes": processes}


def variants(row):
    name = row["path"]
    values = [name, str(REPO_ROOT/name), row["sha256"]]
    if name.startswith("src/"):
        values.append(name.removeprefix("src/").removesuffix(".py").replace("/", "."))
        if "/ops/" in name: values.append(Path(name).stem)
    else: values.append(Path(name).name)
    return values


def references(packet, rows):
    candidates = {r["path"] for r in rows}
    current = set(CURRENT_MACHINE)
    current |= {str(Path(r["path"]).relative_to(REPO_ROOT)) for r in load_json(REPO_ROOT/CURRENT_MACHINE[1])["documents"] if r["status"] == "CURRENT"}
    result = {name: {"current": [], "source": [], "historical": []} for name in candidates}
    errors, seen, count = [], set(), 0
    timer = time.monotonic()
    for scope in ("src", "tests", "scripts", "docs", "tasks", "configs", "contracts"):
        for path in sorted((REPO_ROOT/scope).rglob("*")):
            name = str(path.relative_to(REPO_ROOT))
            if (not path.is_file() or path.is_symlink() or name in candidates or name.startswith(PLAN+"/")
                    or path.suffix not in {".py", ".md", ".json", ".sh", ".toml", ".yaml", ".yml"}
                    or "__pycache__" in path.parts or name in seen): continue
            seen.add(name)
            if name.startswith("tasks/current/legacy_exact_"): continue
            if name in (str(Path(__file__).relative_to(REPO_ROOT)), "tests/unit/test_legacy_exact_delete_v1.py"):
                continue  # Maintenance proof names targets; it is not a producer/consumer.
            try: text = path.read_text()
            except (OSError, UnicodeError) as exc:
                errors.append({"path": name, "error": repr(exc)}); continue
            count += 1
            for row in rows:
                matches = [token for token in variants(row) if token in text]
                # 'from chaoyang import cli_v10' also counts as an external importer.
                if "/cli_v10/" in row["path"] and "cli_v10" in text and name.startswith(("src/", "tests/", "scripts/")):
                    matches.append("cli_v10")
                if matches:
                    kind = "current" if name in current else "source" if name.startswith(("src/", "tests/", "scripts/")) else "historical"
                    result[row["path"]][kind].append({"path": name, "tokens": sorted(set(matches))})
            if time.monotonic()-timer > 30:
                heartbeat(packet); timer = time.monotonic()
    return {"rows": result, "errors": errors, "files_scanned": count,
            "scope": ["receipt-bound CURRENT documents/contracts", "src/tests/scripts", "docs/tasks/configs/contracts history"],
            "not_claimed": "No deletion of historical immutable _run evidence; their frozen refs are preserved and backup-mapped, not republished."}


def prove(packet):
    guard(packet)
    out, _, _ = paths(packet["task_id"])
    rows = load_json(SPEC)["targets"]
    names = [r["path"] for r in rows]
    if not names or len(names) != len(set(names)): raise RuntimeError("DUPLICATE_OR_EMPTY_TARGETS")
    git, processes, scan = git_rows(names), process_scan(names), references(packet, rows)
    transient_error_scans = []
    for _ in range(3):
        if not processes["errors"]: break
        transient_error_scans.append(processes["errors"])
        time.sleep(0.05)
        processes = process_scan(names)
    proofs = []
    closure = {r["relative_path"] for r in load_json(REPO_ROOT/"baseline_v1/MANIFEST.json")["sources"]}
    for row in rows:
        name = row["path"]; path = REPO_ROOT/name
        if Path(name).is_absolute() or ".." in Path(name).parts or not name.startswith(("src/chaoyang/cli_v10/", "src/chaoyang/ops/", "tests/unit/", "docs/current/")):
            raise RuntimeError("TARGET_SCOPE_ESCAPE")
        if path.is_symlink() or not path.is_file() or REPO_ROOT not in path.resolve().parents: raise RuntimeError("TARGET_NOT_REGULAR")
        ref = artifact_ref(path)
        if ref["sha256"] != row["sha256"] or ref["bytes"] != row["bytes"]: raise RuntimeError("TARGET_SHA_DRIFT:"+name)
        hits = scan["rows"][name]
        approved_legacy = set(row.get("nonconsumer_source_referrers", []))
        blockers = []
        if name in packet["inputs"]["protected"] or name in closure: blockers.append("PROTECTED_V1_DEPENDENCY")
        if hits["current"]: blockers.append("CURRENT_REFERENCE")
        external = [r for r in hits["source"] if r["path"] not in approved_legacy]
        if external: blockers.append("SOURCE_REFERENCE")
        if any(name in closure for name in approved_legacy): blockers.append("EXCEPTION_REFERRER_IS_V1_RUNTIME")
        for referrer in approved_legacy:
            if referrer not in {"src/chaoyang/ops/run_robot100_retire_stale_navigation.py", "scripts/publish_robot100_retire_nav.py"} or name != "docs/current/ROBOT100_BATCH_ZH.md.orig":
                blockers.append("UNVERIFIED_SOURCE_EXCEPTION")
            elif referrer.startswith("src/") and "TARGET.unlink()" not in (REPO_ROOT/referrer).read_text():
                blockers.append("DELETE_ONLY_REFERRER_DRIFT")
            elif referrer.startswith("scripts/") and "FRESH_ATTEMPT_REQUIRED" not in (REPO_ROOT/referrer).read_text():
                blockers.append("HISTORICAL_PUBLISHER_GUARD_DRIFT")
        if processes["hits"][name]: blockers.append("LIVE_PID_FD_CWD_OR_COMMAND")
        if processes["errors"] or scan["errors"]: blockers.append("PROOF_SCAN_INCOMPLETE")
        if git[name]["tracked"] and git[name]["porcelain"] != "  ": blockers.append("USER_DIRTY_TRACKED_FILE")
        proofs.append({**row, "file": ref, "git": git[name], "references": hits, "live_holders": processes["hits"][name],
                       "blockers": blockers, "status": "ELIGIBLE_C1" if not blockers else "BLOCKED_REFERENCE_PROOF"})
    proof = {"schema_version": "EXACT_V1_DELETE_PROOF", "task_id": packet["task_id"], "created_at": now_iso(),
             "rows": proofs, "process_scan": processes, "transient_error_scans": transient_error_scans,
             "reference_scan": {k: v for k, v in scan.items() if k != "rows"},
             "license": {"notices": artifact_ref(REPO_ROOT/"THIRD_PARTY_NOTICES.md"),
                         "project_license_files": [artifact_ref(p) for p in REPO_ROOT.glob("LICENSE*") if p.is_file()],
                         "policy": "Project adaptation/maintenance source and docs only. Preserve full original headers in private backup, all license/NOTICE/vendor/asset files unchanged; no redistribution claim."},
             "protected_count": len(packet["inputs"]["protected"]), "deletion_authorized": False}
    atomic_json(out/"PROOF.json", proof)
    return proof


def stop_old_autopilot():
    pid = 35315; expected = 371570676
    before = identity(pid)
    if before is None: return {"status": "ALREADY_ABSENT", "pid": pid, "expected_startticks": expected}
    if (before["proc_start_ticks"] != expected or before["argv"] != ["/usr/local/bin/python", "-B", "scripts/takeover/autopilot.py"]
            or before["cwd"] != str(REPO_ROOT)): raise RuntimeError("OLD_WRITER_IDENTITY_DRIFT")
    proc = Path("/proc")/str(pid)
    if (proc/"task"/str(pid)/"children").read_text().strip(): raise RuntimeError("OLD_WRITER_HAS_CHILDREN")
    if any((value := identity(int(p.name))) and value["ppid"] == pid for p in Path("/proc").iterdir() if p.name.isdigit()):
        raise RuntimeError("OLD_WRITER_HAS_CHILDREN")
    base = REPO_ROOT/"docs/plans/EXACT78_TAKEOVER_20261004"
    plan, state = load_json(base/"AUTOPILOT_PLAN.json"), load_json(base/"AUTOPILOT_STATE.json")
    if len(plan["steps"]) != 8 or any(state["steps"].get(step["id"], {}).get("status") != "DONE" for step in plan["steps"]):
        raise RuntimeError("OLD_PLAN_NOT_DONE")
    fd = os.pidfd_open(pid)
    try:
        latest = identity(pid)
        if not latest or any(latest[k] != before[k] for k in ("pid", "proc_start_ticks", "argv", "cwd")):
            raise RuntimeError("OLD_WRITER_IDENTITY_DRIFT")
        signal.pidfd_send_signal(fd, signal.SIGTERM)
        for _ in range(50):
            after = identity(pid)
            if after is None or after["state"] == "Z":
                return {"status": "STOPPED", "before": before, "signal": "SIGTERM", "plan": artifact_ref(base/"AUTOPILOT_PLAN.json")}
            time.sleep(0.1)
        raise RuntimeError("OLD_WRITER_DID_NOT_EXIT")
    finally: os.close(fd)


def delete(packet):
    guard(packet)
    out, _, _ = paths(packet["task_id"])
    source = packet["inputs"]["proof"]
    if artifact_ref(source["path"]) != source: raise RuntimeError("PROOF_SHA_DRIFT")
    proof = load_json(Path(source["path"]))
    rows = [r for r in proof["rows"] if r["status"] == "ELIGIBLE_C1"]
    if not rows: raise RuntimeError("NO_ELIGIBLE_TARGETS")
    names = [r["path"] for r in rows]
    # Re-run the current/source/process checks before destructive operations.
    latest = prove(packet)
    if any(r["status"] != "ELIGIBLE_C1" for r in latest["rows"] if r["path"] in names):
        raise RuntimeError("C1_FINAL_PROOF_BLOCKED")
    backup = out/"BACKUP.tar.gz"
    with tarfile.open(backup, "x:gz") as archive:
        for row in rows:
            if artifact_ref(REPO_ROOT/row["path"]) != row["file"]: raise RuntimeError("TARGET_SHA_DRIFT")
            archive.add(REPO_ROOT/row["path"], arcname=row["path"], recursive=False)
        archive.add(REPO_ROOT/"THIRD_PARTY_NOTICES.md", arcname="THIRD_PARTY_NOTICES.md", recursive=False)
    with tarfile.open(backup, "r:gz") as archive:
        for row in rows:
            payload = archive.extractfile(row["path"]).read()
            if len(payload) != row["bytes"] or hashlib.sha256(payload).hexdigest() != row["sha256"]:
                raise RuntimeError("BACKUP_SHA_MISMATCH")
    atomic_json(out/"BACKUP_VERIFIED.json", {"status": "PASS", "backup": artifact_ref(backup), "rows": rows,
                "all_original_bytes_verified": True, "restore_requires_new_task": True})
    atomic_json(out/"OLD_WRITER_STOP.json", stop_old_autopilot())
    for row in rows:
        guard(packet)
        path = REPO_ROOT/row["path"]
        if artifact_ref(path) != row["file"]: raise RuntimeError("TARGET_SHA_DRIFT")
        check = process_scan([row["path"]])
        if check["errors"] or check["hits"][row["path"]]: raise RuntimeError("LIVE_HOLDER_BEFORE_DELETE")
    print(json.dumps({"status": "BACKUP_VERIFIED_READY_FOR_PATCH_DELETE", "targets": names}), flush=True)


def finish(packet, failure=None):
    task = packet["task_id"]; out, _, receipt = paths(task)
    guard(packet)
    proof = load_json(out/"PROOF.json") if (out/"PROOF.json").is_file() else None
    result = {"schema_version": "EXACT_V1_CLEANUP_RESULT", "task_id": task, "status": "PASSED" if failure is None else "FAILED_RUNTIME_FINAL",
              "failure_code": failure, "protected_after": refs(tuple(packet["inputs"]["protected"])),
              "algorithm_changed": False, "numeric_gates_changed": False, "models_run": 0, "training_steps": 0,
              "claim_limit": packet["claim_limit"], "outputs": {}}
    if packet["inputs"]["proof"] is not None and failure is None:
        if load_json(out/"REGRESSION.json")["status"] != "PASS": raise RuntimeError("REGRESSION_NOT_PASS")
        saved = load_json(out/"BACKUP_VERIFIED.json")
        if any((REPO_ROOT/r["path"]).exists() for r in saved["rows"]): raise RuntimeError("DELETION_INCOMPLETE")
        deleted = {"schema_version": "EXACT_V1_DELETE_RECEIPT", "task_id": task, "created_at": now_iso(),
                   "backup": saved["backup"], "proof": packet["inputs"]["proof"],
                   "deleted": [{"original_path": r["path"], "sha256": r["sha256"], "bytes": r["bytes"], "tar_member": r["path"],
                                "git_before": r["git"]} for r in saved["rows"]],
                   "files_deleted": len(saved["rows"]), "bytes_deleted": sum(r["bytes"] for r in saved["rows"]),
                   "files_moved": 0, "restorable": True, "restore_requires_new_task": True,
                   "kept_blocked": [r for r in proof["rows"] if r["status"] != "ELIGIBLE_C1"],
                   "claim_limit": "Exact project-local source/docs only. Not a claim that all historical dependencies were removed."}
        atomic_json(out/"DELETE_RECEIPT.json", deleted)
        result.update(files_deleted=deleted["files_deleted"], bytes_deleted=deleted["bytes_deleted"], baseline_pointers_changed=False)
        if refs(EDITS) == packet["inputs"]["edit_before"]: raise RuntimeError("HANDOFF_NOT_UPDATED")
    else:
        deleted = []
        if (out/"BACKUP_VERIFIED.json").is_file():
            saved = load_json(out/"BACKUP_VERIFIED.json")
            deleted = [r for r in saved["rows"] if not (REPO_ROOT/r["path"]).exists()]
            if deleted:
                atomic_json(out/"DELETE_RECEIPT.json", {"schema_version": "EXACT_V1_PARTIAL_DELETE_RECEIPT",
                    "task_id": task, "failure_code": failure, "backup": saved["backup"], "deleted": deleted,
                    "files_deleted": len(deleted), "bytes_deleted": sum(r["bytes"] for r in deleted), "restorable": True})
        result.update(files_deleted=len(deleted), bytes_deleted=sum(r["bytes"] for r in deleted))
    for p in sorted(out.iterdir()):
        if p.is_file() and p.name not in {"RESULT.json", "WRITER.json"}: result["outputs"][p.name] = artifact_ref(p)
    atomic_json(out/"RESULT.json", result)
    atomic_json(receipt, {"schema_version": "EXACT_V1_CLEANUP_TERMINAL", "status": result["status"], "task_id": task, "result": artifact_ref(out/"RESULT.json")})
    state, index = copy.deepcopy(load_json(TASK_STATE_PATH)), index_body()
    next(r for r in state["tasks"] if r["task_id"] == task).update(status=result["status"], pid=None, proc_start_ticks=None,
        updated_at=now_iso(), heartbeat_at=now_iso(), result=artifact_ref(out/"RESULT.json"), terminal_receipt=artifact_ref(receipt), quality="MAINTENANCE_ONLY")
    state["next_task"] = None
    next(r for r in index["task_packets"] if r["task_id"] == task)["execution_allowed"] = False
    index.update(status="PASS_NO_ACTIVE_TASKS", supersedes_index=artifact_ref(INDEX))
    publish_bundle(copy.deepcopy(load_json(AUTHORITY_PATH)), state, event_type="EXACT_CLEANUP_TERMINAL",
        expected_revision=load_json(RECEIPT_PATH)["governance_revision"], generator_path=Path(__file__),
        task_packet_index_path=INDEX, task_packet_index_value=index)
    atomic_json(out/"POST_PUBLICATION_VALIDATION.json", validate_current())
    print(json.dumps({"status": result["status"], "result": artifact_ref(out/"RESULT.json")}), flush=True)


def main():
    c1_only = sys.argv[1:] == ["--c1-only"]
    if c1_only:
        parent, _, _ = paths("legacy_exact_delete_c0_v3_20261008")
        result = load_json(parent/"RESULT.json")
        proof_ref = artifact_ref(parent/"PROOF.json")
        if result["status"] != "PASSED" or result["outputs"]["PROOF.json"] != proof_ref:
            raise RuntimeError("PARENT_PROOF_NOT_SEALED")
        packet = register("legacy_exact_delete_c1_v5_20261008", proof_ref)
    else:
        packet = register("legacy_exact_delete_c0_v3_20261008")
    try:
        if not c1_only:
            proof = prove(packet)
            print(json.dumps({"proof": [{"path": r["path"], "status": r["status"], "blockers": r["blockers"]} for r in proof["rows"]]}), flush=True)
            finish(packet)
            if any(r["status"] != "ELIGIBLE_C1" for r in proof["rows"]):
                raise RuntimeError("C0_BLOCKED_NO_DELETE_PACKET_OPENED")
            out, _, _ = paths(packet["task_id"])
            packet = register("legacy_exact_delete_c1_v3_20261008", artifact_ref(out/"PROOF.json"))
        print("READY_C1", flush=True)
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], 30)
            if not ready: heartbeat(packet); continue
            command = sys.stdin.readline().strip()
            if command == "PREPARE_DELETE": delete(packet)
            elif command == "PUBLISH": finish(packet); return
            else: raise RuntimeError("ABORTED")
    except BaseException as exc:
        out, _, _ = paths(packet["task_id"])
        if not (out/"RESULT.json").exists(): finish(packet, repr(exc))
        raise


if __name__ == "__main__": main()
