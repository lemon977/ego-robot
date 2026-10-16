"""Finite CPU-only, reversible project layout organization; no algorithm runs."""
from __future__ import annotations

import copy
from datetime import datetime, timedelta
import hashlib
import importlib
import json
import os
from pathlib import Path
import secrets
import select
import subprocess
import sys

from chaoyang.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH,
    artifact_ref, atomic_json, canonical_bytes, load_json, now_iso,
    process_identity, publish_bundle,
)
from chaoyang.governance.register_single_task_packet import _validate_packet

TASK = "project_layout_organization_v1_20261008"
ROOT = REPO_ROOT / "_run/current" / TASK / "attempts/attempt_0001"
PACKET = REPO_ROOT / "tasks/current" / TASK / "TASK_PACKET.json"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
RECEIPT = REPO_ROOT / "tasks/receipts/PROJECT_LAYOUT_ORGANIZATION_20261008.json"
PLAN = "docs/plans/PROJECT_LAYOUT_ORGANIZATION_20261008"
PROTECTED = (
    "docs/current/EXACT78_VISUAL_BASELINE_V1.json",
    "tasks/receipts/EXACT78_VISUAL_BASELINE_V1.json",
    "_run/current/exact78_takeover_b2_full_v13/attempts/attempt_0001/RESULT.json",
    "_run/current/exact78_takeover_v13_cards042_full_replay_v1_20261008/attempts/attempt_0001/RESULT.json",
    "review/v13_cards042_replay_20261008/COMPARISON.json",
    "docs/current/ROBOT100_BATCH_ZH.md",
    "docs/current/ROBOT100_HARDWARE_BASELINE_V16.json",
    "src/chaoyang/ops/run_robot100_world_first_v10.py",
    "src/chaoyang/ops/run_robot100_constraint_review_v16_retry1.py",
)
# Fixed exact targets; a directory named src, data, vendor or _run is NEVER moved.
MOVES = (
    (" src", "archive/layout-20261008/root_residue/mistyped_src_empty"),
    ("governance", "archive/layout-20261008/root_residue/governance"),
    (".pytest_cache", "_run/caches/tooling/legacy_20261008/pytest"),
    (".ruff_cache", "_run/caches/tooling/legacy_20261008/ruff"),
)
ALIASES = {
    "final.mp4": "review/v13_cards042_replay_20261008/final.mp4",
    "review.mp4": "review/v13_cards042_replay_20261008/review.mp4",
    "compare.mp4": "review/v13_cards042_replay_20261008/compare.mp4",
    "batch": "_run/current/exact78_takeover_b2_full_v13/attempts/attempt_0001/candidate",
}


def validate() -> dict:
    run = subprocess.run([sys.executable, "-B", "-m", "chaoyang.cli", "validate-governance"],
                         cwd=REPO_ROOT, env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
                         capture_output=True, text=True, timeout=120)
    if run.returncode:
        raise RuntimeError("GOVERNANCE_INVALID:" + run.stdout[-1200:] + run.stderr[-1200:])
    return json.loads(run.stdout)


def index_body() -> dict:
    return {key: copy.deepcopy(value) for key, value in load_json(INDEX).items()
            if key not in {"governance_revision", "generation_id", "generated_at", "supersedes_index"}}


def guards() -> dict:
    refs = {path: artifact_ref(REPO_ROOT / path) for path in PROTECTED}
    adopted = load_json(REPO_ROOT / "tasks/receipts/EXACT78_VISUAL_BASELINE_V1.json")
    for ref in adopted["algorithm_code"]:
        if artifact_ref(Path(ref["path"])) != ref:
            raise RuntimeError("BASELINE_ALGORITHM_DRIFT:" + ref["path"])
        refs[str(Path(ref["path"]).relative_to(REPO_ROOT))] = ref
    return refs


def tree(path: Path) -> dict:
    if path.is_symlink() or not path.resolve().is_relative_to(REPO_ROOT):
        raise RuntimeError("UNSAFE_TREE:" + str(path))
    rows = []
    for item in sorted(path.rglob("*")):
        if item.is_symlink():
            raise RuntimeError("SYMLINK_IN_ARCHIVE_TARGET:" + str(item))
        if item.is_file():
            ref = artifact_ref(item)
            rows.append({"relative": str(item.relative_to(path)), "sha256": ref["sha256"], "bytes": ref["bytes"]})
    return {"files": rows, "file_count": len(rows), "bytes": sum(row["bytes"] for row in rows),
            "content_signature": hashlib.sha256(canonical_bytes(rows)).hexdigest()}


def process_hits(target: Path) -> list[dict]:
    hits = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        identity = process_identity(int(proc.name))
        for link in [proc / "cwd", *list((proc / "fd").glob("*"))]:
            try:
                dest = Path(os.readlink(link).removesuffix(" (deleted)"))
            except OSError:
                continue
            if dest == target or target in dest.parents:
                hits.append({"identity": identity, "link": str(link), "target": str(dest)})
    return hits


def runtime_hits(relative: str) -> list[str]:
    """Exact runtime/file pins, not old inventory prose or our own task manifest."""
    candidates = [REPO_ROOT / "docs/governance/ALGORITHM_CONTRACT.json",
                  REPO_ROOT / "docs/governance/CURRENT_BASELINE_REGISTRY_V2.json",
                  REPO_ROOT / "docs/governance/CURRENT_REGRESSION_MANIFEST.json"]
    candidates += [REPO_ROOT / row["packet_path"] for row in load_json(INDEX)["task_packets"]
                   if row.get("task_id") != TASK]
    absolute = str(REPO_ROOT / relative)
    hits = []
    def visit(value, field=""):
        if isinstance(value, dict):
            for key, child in value.items(): visit(child, key)
        elif isinstance(value, list):
            for child in value: visit(child, field)
        elif isinstance(value, str):
            # A regression group called "governance" is a category, not a
            # dependency on the root governance/ directory. Bare directory
            # names qualify only in explicit path/read/write fields.
            bare_path = value == relative and field in {"path", "read_set", "write_set", "inputs", "code"}
            if (bare_path or value.startswith(relative + "/")
                    or value == absolute or value.startswith(absolute + "/")):
                hits.append(str(file.relative_to(REPO_ROOT)))
    def visit_pinned_inputs(value):
        if isinstance(value, dict):
            if isinstance(value.get("path"), str) and "sha256" in value:
                visit(value["path"], "path")
            else:
                for child in value.values(): visit_pinned_inputs(child)
        elif isinstance(value, list):
            for child in value: visit_pinned_inputs(child)
    active_packets = {row["packet_path"] for row in load_json(INDEX)["task_packets"] if row.get("execution_allowed")}
    for file in candidates:
        payload = load_json(file)
        if file.is_relative_to(REPO_ROOT / "tasks/current"):
            # Completed cleanup plans mentioning the target in read_set,
            # write_set or inputs.moves are audit metadata, not live readers.
            # Preserve actual historical source/input SHA pins, and honor
            # every path declaration of any other active packet.
            visit(payload.get("runtime_code", []), "runtime_code")
            visit_pinned_inputs(payload.get("inputs", {}))
            if str(file.relative_to(REPO_ROOT)) in active_packets: visit(payload)
        else:
            visit(payload)
    # Bare imports/commands and navigation are additional live reference evidence.
    if relative == "governance":
        run = subprocess.run(["rg", "-l", "-F", "audit_exact78_full_baseline_v1",
                              "src", "scripts", "configs", "manifests", "docs/current", "review", "tests"],
                             cwd=REPO_ROOT, capture_output=True, text=True)
        if run.returncode not in (0, 1): raise RuntimeError("REFERENCE_SCAN_FAILED")
        hits += [line for line in run.stdout.splitlines() if line != str(Path(__file__).relative_to(REPO_ROOT))]
    return sorted(set(hits))


def register(before: dict) -> dict:
    state, index = copy.deepcopy(load_json(TASK_STATE_PATH)), index_body()
    if state.get("next_task") or any(row.get("execution_allowed") for row in index["task_packets"]):
        raise RuntimeError("ACTIVE_WRITER_CONFLICT")
    if ROOT.exists() or PACKET.exists() or RECEIPT.exists(): raise RuntimeError("FRESH_TASK_REQUIRED")
    writer = {"pid": os.getpid(), "proc_start_ticks": process_identity(os.getpid())["start_ticks"],
              "executor_epoch": 1, "fencing_token": secrets.token_hex(24), "scope": TASK}
    inputs = {"task_id": TASK, "moves": MOVES, "aliases": ALIASES,
              "protected": before, "plan": artifact_ref(REPO_ROOT / PLAN / "00_EXECUTION_ZH.md"),
              "code": artifact_ref(Path(__file__))}
    signature = hashlib.sha256(canonical_bytes(inputs)).hexdigest()
    packet = {"schema_version": "chaoyang-r22-task-packet-v1", "task_id": TASK,
              "objective": "Reversibly organize root residues and shallow current review navigation",
              "phase": "PROJECT_LAYOUT_ORGANIZATION", "plan_revision": "PROJECT_LAYOUT_20261008",
              "execution_revision": "PROJECT_LAYOUT_V1", "read_set": ["AGENTS.md", "docs/current/AI_WORK_ENTRY_ZH.md",
                  "docs/governance", "tasks/current", "review", "src/chaoyang", "governance", PLAN],
              "write_set": [str(ROOT.relative_to(REPO_ROOT)), str(PACKET.relative_to(REPO_ROOT)),
                  str(RECEIPT.relative_to(REPO_ROOT)), "tasks/current/INDEX.json", "docs/governance", PLAN,
                  "README.md", "docs/README.md", "docs/current/README_ZH.md", "docs/current/PROJECT_LAYOUT_ZH.md",
                  "review/README.md", "review/history", "review/exact78_v1", "pyproject.toml",
                  "src/chaoyang/governance/current_r3_contracts.py", "src/chaoyang/governance/current_baseline_v2.py",
                  *(p for pair in MOVES for p in pair)],
              "prerequisites": ["GOVERNANCE_PASS", "BASELINE_GUARDS_UNCHANGED", "NO_ACTIVE_TASK"],
              "required_outputs": ["RESULT.json", "layout_receipt", "directory_guide", "move_proof", "governance_PASS"],
              "budgets": {"workers_max": 1, "cpu_threads_total": 1, "gpu_count": 0, "model_calls": 0,
                  "render_calls": 0, "training_steps": 0, "wall_seconds_max": 1200, "session_attempts_max": 1},
              "stop_conditions": ["CURRENT_REFERENCE_PRESENT", "ACTIVE_FD_CWD", "GUARD_SHA_DRIFT", "DOUBLE_WRITER", "DEADLINE"],
              "claim_limit": "Directory/navigation only, recoverable moves, no file deletion or algorithm/model/quality change.",
              "deadline_at": (datetime.now().astimezone() + timedelta(minutes=20)).isoformat(),
              "writer": writer, "inputs": inputs, "run_signature": signature,
              "commands": ["python -B -m chaoyang.governance.organize_project_layout_v1" +
                           (" " + sys.argv[1] if sys.argv[1:] else "")]}
    if errors := _validate_packet(packet): raise RuntimeError("INVALID_PACKET:" + str(errors))
    ROOT.mkdir(parents=True); PACKET.parent.mkdir(parents=True)
    atomic_json(ROOT / "WRITER.json", writer); atomic_json(PACKET, packet)
    atomic_json(ROOT / "ROOT_BEFORE.json", {"entries": sorted(p.name for p in REPO_ROOT.iterdir()),
                                         "git_status": subprocess.check_output(["git", "status", "--porcelain=v1"], cwd=REPO_ROOT).decode()})
    state["tasks"].append({"task_id": TASK, "phase": packet["phase"], "attempt": 1, "status": "RUNNING",
        "session": "project_layout", "updated_at": now_iso(), "heartbeat_at": now_iso(),
        "pid": writer["pid"], "proc_start_ticks": writer["proc_start_ticks"], "gpu_id": None,
        "task_packet": artifact_ref(PACKET), "run_signature": signature, "quality": "NOT_APPLICABLE",
        "adoption": "NOT_APPLICABLE", "review": "PENDING"})
    state["next_task"] = {"task_id": TASK, "session": "project_layout", "prerequisites": packet["prerequisites"],
                          "expected_resource": "CPU_ONE_THREAD", "stop_condition": "one terminal layout publication"}
    index["task_packets"].append({"task_id": TASK, "packet_path": str(PACKET.relative_to(REPO_ROOT)),
        "packet_sha256": artifact_ref(PACKET)["sha256"], "execution_class": "CURRENT_LEDGER_ROUTABLE", "execution_allowed": True})
    index.update(packet_revision="PROJECT_LAYOUT_RUNNING", status="PASS", supersedes_index=artifact_ref(INDEX))
    publish_bundle(copy.deepcopy(load_json(AUTHORITY_PATH)), state, event_type="PROJECT_LAYOUT_REGISTERED",
        expected_revision=load_json(RECEIPT_PATH)["governance_revision"], generator_path=Path(__file__),
        task_packet_index_path=INDEX, task_packet_index_value=index)
    return packet


def organize() -> tuple[list, dict]:
    moves = []
    for old, new in MOVES:
        source, dest = REPO_ROOT / old, REPO_ROOT / new
        if not source.exists():
            moves.append({"source": old, "status": "ALREADY_ABSENT"}); continue
        snapshot = tree(source)
        proof = {"source": old, "destination": new, "before": snapshot,
                 "runtime_references": runtime_hits(old), "process_references": process_hits(source),
                 "git_status": subprocess.check_output(["git", "status", "--short", "--", old], cwd=REPO_ROOT).decode(),
                 "license_policy": "Whole tree and notices preserved; no vendor/model/asset/license deletion."}
        if proof["runtime_references"] or proof["process_references"]:
            proof["status"] = "BLOCKED_REFERENCE_PROOF"; moves.append(proof); continue
        if old == " src" and snapshot["file_count"]: raise RuntimeError("MISTYPED_SRC_NOT_EMPTY")
        if dest.exists() or dest.is_symlink(): raise RuntimeError("ARCHIVE_COLLISION:" + new)
        if not dest.resolve().is_relative_to(REPO_ROOT): raise RuntimeError("ARCHIVE_ESCAPES_PROJECT")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if tree(source) != snapshot or process_hits(source): raise RuntimeError("MOVE_PROOF_DRIFT:" + old)
        source.rename(dest)
        if tree(dest) != snapshot: raise RuntimeError("ARCHIVE_CONTENT_DRIFT:" + old)
        proof.update(status="ARCHIVED_RECOVERABLE", after=tree(dest)); moves.append(proof)
        atomic_json(ROOT / "MOVE_RECEIPT.json", {"moves": moves, "files_deleted": False})
    folder = REPO_ROOT / "review/exact78_v1"
    if folder.is_symlink() or not folder.resolve().is_relative_to(REPO_ROOT): raise RuntimeError("UNSAFE_REVIEW_FOLDER")
    folder.mkdir(exist_ok=True)
    aliases = {}
    for name, relative in ALIASES.items():
        target = (REPO_ROOT / relative).resolve(strict=True)
        if not target.is_relative_to(REPO_ROOT): raise RuntimeError("ALIAS_ESCAPES_PROJECT")
        alias = folder / name
        if alias.exists() or alias.is_symlink(): raise RuntimeError("ALIAS_COLLISION:" + name)
        alias.symlink_to(os.path.relpath(target, folder), target_is_directory=target.is_dir())
        aliases[name] = {"target": relative, "file": artifact_ref(alias) if alias.is_file() else None}
    return moves, aliases


def finish(packet: dict, before: dict, moves: list, aliases: dict, status: str, failure: str | None) -> None:
    after = guards()
    if before != after: raise RuntimeError("BASELINE_OR_0916_SHA_DRIFT")
    guide = REPO_ROOT / "docs/current/PROJECT_LAYOUT_ZH.md"
    result = {"schema_version": "chaoyang-project-layout-result-v1", "task_id": TASK, "status": status,
              "failure_code": failure, "moves": moves, "aliases": aliases, "files_deleted": False,
              "protected_sha_unchanged": True, "protected_before": before, "protected_after": after,
              "root_after": sorted(p.name for p in REPO_ROOT.iterdir()), "model_calls": 0,
              "guide": artifact_ref(guide) if guide.is_file() else None, "finished_at": now_iso()}
    atomic_json(ROOT / "RESULT.json", result)
    atomic_json(RECEIPT, {"schema_version": "chaoyang-project-layout-receipt-v1", "task_id": TASK,
        "status": status, "result": artifact_ref(ROOT / "RESULT.json"), "guide": result["guide"],
        "files_deleted": False, "algorithm_changed": False, "baseline_changed": False, "recorded_at": now_iso()})
    from chaoyang.governance import current_baseline_v2, current_r3_contracts
    importlib.reload(current_baseline_v2); importlib.reload(current_r3_contracts)
    state, index = copy.deepcopy(load_json(TASK_STATE_PATH)), index_body()
    row = next(x for x in state["tasks"] if x["task_id"] == TASK)
    row.update(status=status, pid=None, proc_start_ticks=None, updated_at=now_iso(), heartbeat_at=now_iso(),
        result=artifact_ref(ROOT / "RESULT.json"), terminal_receipt=artifact_ref(RECEIPT),
        quality="NOT_APPLICABLE", adoption="DIRECTORY_LAYOUT_ONLY", review="SOURCE_SHA_VERIFIED")
    state["next_task"] = None
    for row in index["task_packets"]:
        if row["task_id"] == TASK: row["execution_allowed"] = False
    index.update(packet_revision="PROJECT_LAYOUT_TERMINAL", status="PASS_NO_ACTIVE_TASKS", supersedes_index=artifact_ref(INDEX))
    publish_bundle(copy.deepcopy(load_json(AUTHORITY_PATH)), state, event_type="PROJECT_LAYOUT_PUBLISHED",
        expected_revision=load_json(RECEIPT_PATH)["governance_revision"], generator_path=Path(__file__),
        task_packet_index_path=INDEX, task_packet_index_value=index)
    validation = validate()
    print(json.dumps({"status": status, "revision": validation["revision"], "moves": len([x for x in moves if x["status"] == "ARCHIVED_RECOVERABLE"]),
                      "result": artifact_ref(ROOT / "RESULT.json"), "receipt": artifact_ref(RECEIPT)}, ensure_ascii=False), flush=True)


def main() -> int:
    global TASK, ROOT, PACKET, RECEIPT, MOVES, ALIASES
    finish_root_audit = sys.argv[1:] in (["--finish-root-audit"], ["--finish-root-audit-v2"])
    if sys.argv[1:] and not finish_root_audit: raise RuntimeError("UNSUPPORTED_LAYOUT_ARGUMENTS")
    if finish_root_audit:
        version = "v2" if sys.argv[1:] == ["--finish-root-audit-v2"] else "v1"
        TASK = f"project_layout_root_audit_archive_{version}_20261008"
        ROOT = REPO_ROOT / "_run/current" / TASK / "attempts/attempt_0001"
        PACKET = REPO_ROOT / "tasks/current" / TASK / "TASK_PACKET.json"
        RECEIPT = REPO_ROOT / "tasks/receipts" / ("PROJECT_LAYOUT_ROOT_AUDIT_ARCHIVE_V2_20261008.json" if version == "v2"
                                                else "PROJECT_LAYOUT_ROOT_AUDIT_ARCHIVE_20261008.json")
        MOVES = (("governance", "archive/layout-20261008/root_residue/governance"),)
        ALIASES = {}
    validate(); before = guards(); packet = register(before)
    print(json.dumps({"status": "READY", "task_id": TASK}, ensure_ascii=False), flush=True)
    moves, aliases, status, failure = [], {}, "PASSED", None
    try:
        if not finish_root_audit:
            ready, _, _ = select.select([sys.stdin], [], [], 900)
            if not ready or sys.stdin.readline().strip() != "PUBLISH": raise RuntimeError("EDIT_WINDOW_EXPIRED")
        state, index = load_json(TASK_STATE_PATH), load_json(INDEX)
        active = [row for row in state["tasks"] if row["status"] in {"RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE"}]
        allowed = [row for row in index["task_packets"] if row.get("execution_allowed")]
        if (len(active) != 1 or active[0]["task_id"] != TASK or active[0].get("run_signature") != packet["run_signature"]
                or active[0].get("pid") != os.getpid() or len(allowed) != 1 or allowed[0]["task_id"] != TASK
                or load_json(ROOT / "WRITER.json") != packet["writer"]
                or artifact_ref(PACKET)["sha256"] != allowed[0]["packet_sha256"]):
            raise RuntimeError("WRITER_FENCE_OR_TASK_DRIFT")
        moves, aliases = organize()
    except Exception as exc:
        status, failure = "FAILED_RUNTIME_FINAL", str(exc)
        if (ROOT / "MOVE_RECEIPT.json").exists(): moves = load_json(ROOT / "MOVE_RECEIPT.json")["moves"]
    finish(packet, before, moves, aliases, status, failure)
    return 0 if status == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
