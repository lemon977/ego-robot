#!/usr/bin/env python3
"""Remove retired Cards042 media/cache artifacts with an explicit governed inventory."""

from __future__ import annotations

import copy
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys

from chaoyang.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH,
    artifact_ref, atomic_json, canonical_bytes, load_json, now_iso,
    process_identity, publish_bundle,
)
from chaoyang.governance.register_single_task_packet import _validate_packet


TASK = "exact78_cards042_v2_retired_cleanup_20260929"
ROOT = REPO_ROOT / "_run/current" / TASK / "attempts/attempt_0001"
PACKET = REPO_ROOT / "tasks/current" / TASK / "TASK_PACKET.json"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
RECEIPT = REPO_ROOT / "tasks/receipts/EXACT78_CARDS042_V2_RETIRED_CLEANUP.json"
BASELINE = REPO_ROOT / "docs/current/EXACT78_CARDS042_VISUAL_BASELINE_V2.json"
BRANCH_LOCK = REPO_ROOT / "docs/current/EXACT78_CARDS042_VISUAL_BRANCH_LOCK_V2.json"

RETIRED_RUNTIME_NAMES = (
    "exact78_cards042_canonical_rebuild_v2_20260929",
    "exact78_cards042_direct_object_rebuild_v1_20260929",
    "exact78_cards042_direct_object_rebuild_v1_retry1_20260929",
    "exact78_cards042_direct_object_rebuild_v1_retry2_20260929",
    "exact78_cards042_direct_object_rebuild_v1_retry3_20260929",
    "exact78_cards042_direct_object_rebuild_v1_retry4_20260929",
    "exact78_cards042_full_clean_robot_v3_20260929",
    "exact78_cards042_full_motion_review_v1_20260929",
    "exact78_cards042_motion_render_continuation_v6_20260929",
    "exact78_cards042_object_prompt_canary_v2_20260929",
    "exact78_cards042_object_prompt_canary_v3_20260929",
    "exact78_cards042_object_prompt_canary_v4_20260929",
    "exact78_cards042_object_prompt_canary_v5_20260929",
    "exact78_cards042_pre_v3_cleanup_v1_20260929",
    "exact78_cards042_propainter_model_replay_v8_20260929",
    "exact78_cards042_raw_motion_replay_probe_v1_20260929",
    "exact78_cards042_raw_motion_replay_probe_v2_20260929",
    "exact78_cards042_raw_motion_replay_probe_v3_20260929",
    "exact78_cards042_raw_motion_replay_probe_v4_20260929",
    "exact78_cards042_raw_motion_replay_probe_v5_20260929",
    "exact78_cards042_unknown_safe_composite_v7_20260929",
    "exact78_cards042_visual_baseline_selection_v1_20260929",
    "exact78_cards042_visual_branch_lock_v1_20260929",
    "exact78_general_baseline_consolidation_v1_20260929",
    "exact78_general_raw_to_robot_v1_20260929",
    "exact78_general_raw_to_robot_v1_retry1_20260929",
    "exact78_general_raw_to_robot_v1_retry2_20260929",
    "exact78_general_raw_to_robot_v1_retry3_20260929",
    "exact78_general_raw_to_robot_v1_retry4_20260929",
    "exact78_inpainting_robot100_20260928",
    "exact78_standalone_visual_replay_v1_20260929",
    "exact78_standalone_visual_replay_v1_retry1_20260929",
)
RECORD_SUFFIXES = {".json", ".jsonl", ".log", ".md", ".txt", ".yaml", ".yml", ".csv"}
KEEP_TASKS = {
    "exact78_cards042_role_mask_rebuild_v1_20260929",
    "exact78_cards042_direct_object_rebuild_v2_20260929",
    "exact78_cards042_fresh_clean_v1_20260929",
    "exact78_cards042_fresh_robot_v1_20260929",
    "exact78_cards042_visual_baseline_v2_publish_20260929",
    TASK,
}
REMOVED_SOURCE_RECORD = [
    "23 retired Cards042 operation scripts",
    "36 retired Exact78/Cards042 governance scripts",
    "2 retired tests",
    "6 superseded docs/current Exact78 files",
    "1 stale current_r3_contracts.py.orig",
]


def index_body() -> dict:
    return {key: copy.deepcopy(value) for key, value in load_json(INDEX).items()
            if key not in {"governance_revision", "generation_id", "generated_at", "supersedes_index"}}


def target_files() -> list[Path]:
    files: set[Path] = set()
    for name in RETIRED_RUNTIME_NAMES:
        root = REPO_ROOT / "_run/current" / name
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.is_file() and not path.is_symlink() and path.suffix.lower() not in RECORD_SUFFIXES:
                files.add(path.resolve())
    old_archive = REPO_ROOT / "archive/exact78-cards042-pre-v3-20260929"
    if old_archive.is_dir():
        for path in old_archive.rglob("*"):
            if path.is_file() and not path.is_symlink() and path.suffix.lower() not in RECORD_SUFFIXES:
                files.add(path.resolve())
    # Bytecode for deleted branch-only modules is never an experiment record.
    retired_tokens = ("cards042_", "exact78_standalone", "exact78_general_baseline",
                      "exact78_adoption_gate", "exact78_full_pipeline_v1")
    for path in (REPO_ROOT / "src/chaoyang").rglob("*.pyc"):
        if any(token in path.name for token in retired_tokens):
            files.add(path.resolve())
    return sorted(files)


def process_hits(targets: list[Path]) -> list[dict]:
    target_set = set(targets)
    roots = [(REPO_ROOT / "_run/current" / name).resolve() for name in RETIRED_RUNTIME_NAMES]
    hits = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        links = [proc / "cwd"]
        try:
            links.extend((proc / "fd").iterdir())
        except (FileNotFoundError, PermissionError):
            pass
        for link in links:
            try:
                value = Path(os.readlink(link)).resolve()
            except OSError:
                continue
            if value in target_set or (link.name == "cwd" and any(value == root or value.is_relative_to(root) for root in roots)):
                hits.append({"pid": int(proc.name), "link": str(link), "path": str(value)})
    return hits


def references(value):
    if isinstance(value, dict):
        if {"path", "bytes", "sha256"} <= value.keys():
            yield Path(value["path"]).resolve()
        for child in value.values():
            yield from references(child)
    elif isinstance(value, list):
        for child in value:
            yield from references(child)


def main() -> int:
    state, index = copy.deepcopy(load_json(TASK_STATE_PATH)), index_body()
    if state.get("next_task") or any(row.get("execution_allowed") for row in index["task_packets"]):
        raise RuntimeError("ACTIVE_TASK_CONFLICT")
    if ROOT.exists() or PACKET.exists() or RECEIPT.exists():
        raise RuntimeError("FRESH_CLEANUP_REQUIRED")

    targets = target_files()
    hits = process_hits(targets)
    if hits:
        raise RuntimeError("TARGET_IN_USE:" + json.dumps(hits[:20]))
    target_set = set(targets)
    for name in ("ALGORITHM_CONTRACT.json", "CURRENT_BASELINE_REGISTRY_V2.json", "CURRENT_REGRESSION_MANIFEST.json"):
        current_refs = set(references(load_json(REPO_ROOT / "docs/governance" / name)))
        overlap = current_refs & target_set
        if overlap:
            raise RuntimeError(f"CURRENT_REFERENCE:{name}:{sorted(map(str, overlap))[:10]}")

    protected = [BASELINE, BRANCH_LOCK,
                 REPO_ROOT / "docs/current/ROBOT100_HARDWARE_BASELINE_V16.json",
                 REPO_ROOT / "_run/current/exact78_cards042_fresh_robot_v1_20260929/attempts/attempt_0001/robot/RESULT.json",
                 REPO_ROOT / "_run/current/exact78_cards042_fresh_clean_v1_20260929/attempts/attempt_0001/clean/RESULT.json",
                 REPO_ROOT / "_run/current/exact78_cards042_direct_object_rebuild_v2_20260929/attempts/attempt_0001/object_identity/RESULT.json",
                 REPO_ROOT / "_run/current/exact78_cards042_role_mask_rebuild_v1_20260929/attempts/attempt_0001/runner/sessions/play_cards_0902_042/attempts/attempt_0001/role_mask_output/RESULT.json"]
    protected_refs = [artifact_ref(path) for path in protected]
    if any(Path(ref["path"]).resolve() in target_set for ref in protected_refs):
        raise RuntimeError("PROTECTED_TARGET_COLLISION")

    ROOT.mkdir(parents=True)
    inventory = []
    digest = hashlib.sha256()
    for path in targets:
        ref = artifact_ref(path)
        relative = str(path.relative_to(REPO_ROOT))
        row = {"relative_path": relative, "bytes": ref["bytes"], "sha256": ref["sha256"]}
        inventory.append(row)
        digest.update(canonical_bytes(row))
    plan = {"schema_version": "exact78-cards042-v2-retired-cleanup-plan",
            "retired_runtime_roots": list(RETIRED_RUNTIME_NAMES),
            "target_count": len(inventory), "target_bytes": sum(row["bytes"] for row in inventory),
            "inventory_sha256": digest.hexdigest(), "targets": inventory,
            "retained_record_suffixes": sorted(RECORD_SUFFIXES),
            "removed_source_record": REMOVED_SOURCE_RECORD,
            "protected": protected_refs, "process_fd_cwd_hits": hits,
            "git_status_note": "Existing shared dirty worktree observed; targets are branch-scoped and 0916 paths are excluded.",
            "license_note": "No vendor/model/third-party licensed assets are deletion targets."}
    atomic_json(ROOT / "DELETION_PLAN.json", plan)

    writer = {"pid": os.getpid(), "proc_start_ticks": process_identity(os.getpid())["start_ticks"],
              "executor_epoch": 1, "fencing_token": secrets.token_hex(24), "scope": TASK}
    inputs = {"baseline": artifact_ref(BASELINE), "branch_lock": artifact_ref(BRANCH_LOCK),
              "deletion_plan": artifact_ref(ROOT / "DELETION_PLAN.json")}
    signature = hashlib.sha256(canonical_bytes({"inputs": inputs, "inventory_sha256": digest.hexdigest()})).hexdigest()
    packet = {"schema_version": "chaoyang-r22-task-packet-v1", "task_id": TASK,
              "objective": "Delete retired Cards042 media/cache and remove stale task routing after V2 publication",
              "phase": "EXACT78_CARDS042_V2_RETIREMENT", "plan_revision": "EXACT78_CARDS042_VISUAL_BASELINE_V2",
              "execution_revision": "EXACT78_CARDS042_V2_RETIRED_CLEANUP", "session_id": "play_cards_0902_042",
              "deadline_at": (datetime.now().astimezone() + timedelta(hours=1)).isoformat(),
              "read_set": ["AGENTS.md", "docs/current/EXACT78_CARDS042_VISUAL_BASELINE_V2.json",
                           str((ROOT / "DELETION_PLAN.json").relative_to(REPO_ROOT))],
              "write_set": [str(ROOT.relative_to(REPO_ROOT)), str(PACKET.relative_to(REPO_ROOT)),
                            str(RECEIPT.relative_to(REPO_ROOT)), "tasks/current/INDEX.json", "docs/governance"],
              "prerequisites": ["V2_PUBLISHED", "NO_ACTIVE_WRITER", "NO_TARGET_FD_OR_CWD",
                                "NO_CURRENT_CONTRACT_TARGET_REFERENCE", "PROTECTED_SHA_FROZEN"],
              "required_outputs": ["exact_deletion_inventory", "retained_record_only_roots",
                                   "stale_packet_routes_removed", "cleanup_receipt", "post_delete_governance_PASS"],
              "budgets": {"workers_max": 1, "cpu_threads_total": 1, "gpu_count": 0,
                          "training_steps": 0, "wall_seconds_max": 3600, "session_attempts_max": 1},
              "stop_conditions": ["target_in_use", "current_reference", "protected_sha_drift",
                                  "other_branch_collision", "one_attempt_terminal"],
              "claim_limit": "Retires only old 0902 Cards042 branch artifacts. 0916 Robot100, shared assets, external data and current V2 artifacts are untouched.",
              "writer": writer, "inputs": inputs, "run_signature": signature,
              "commands": ["python -m chaoyang.governance.cleanup_exact78_cards042_v2_retired_artifacts"]}
    errors = _validate_packet(packet)
    if errors:
        raise RuntimeError("INVALID_PACKET:" + ";".join(errors))
    atomic_json(ROOT / "WRITER.json", writer)
    PACKET.parent.mkdir(parents=True)
    atomic_json(PACKET, packet)
    now = now_iso()
    state["tasks"].append({"task_id": TASK, "phase": packet["phase"], "attempt": 1, "status": "RUNNING",
                           "session": "play_cards_0902_042_cleanup", "updated_at": now, "heartbeat_at": now,
                           "pid": writer["pid"], "proc_start_ticks": writer["proc_start_ticks"], "gpu_id": None,
                           "task_packet": artifact_ref(PACKET), "run_signature": signature,
                           "quality": "PENDING", "adoption": "NOT_APPLICABLE", "review": "MACHINE_REFERENCE_AUDIT"})
    state["next_task"] = {"task_id": TASK, "session": "play_cards_0902_042_cleanup",
                          "prerequisites": packet["prerequisites"], "expected_resource": "CPU_ONE_THREAD",
                          "stop_condition": "explicit inventory deleted or fail closed"}
    # Retired packets remain as compact JSON records on disk but are removed
    # from current routing. Non-Exact78 rows, including 0916 Robot100, remain byte-for-byte entries.
    index["task_packets"] = [row for row in index["task_packets"]
                             if row.get("task_id") not in RETIRED_RUNTIME_NAMES]
    index["task_packets"].append({"task_id": TASK, "packet_path": str(PACKET.relative_to(REPO_ROOT)),
                                  "packet_sha256": artifact_ref(PACKET)["sha256"],
                                  "execution_class": "CURRENT_LEDGER_ROUTABLE", "execution_allowed": True})
    index.update(packet_revision="EXACT78_CARDS042_V2_CLEANUP_RUNNING", status="PASS",
                 supersedes_index=artifact_ref(INDEX))
    authority = copy.deepcopy(load_json(AUTHORITY_PATH))
    authority["claims"] = [claim for claim in authority.get("claims", [])
                           if claim.get("scope") != "exact78/play_cards_0902_042/current_visual_baseline_v2"]
    publish_bundle(authority, state, event_type="EXACT78_CARDS042_V2_CLEANUP_REGISTERED",
                   expected_revision=load_json(RECEIPT_PATH)["governance_revision"], generator_path=Path(__file__),
                   task_packet_index_path=INDEX, task_packet_index_value=index)

    for path in targets:
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"PRE_DELETE_DRIFT:{path}")
        path.unlink()
    for root_name in RETIRED_RUNTIME_NAMES:
        root = REPO_ROOT / "_run/current" / root_name
        if root.is_dir():
            for directory in sorted((p for p in root.rglob("*") if p.is_dir()), reverse=True):
                try:
                    directory.rmdir()
                except OSError:
                    pass
    if any(path.exists() for path in targets):
        raise RuntimeError("POST_DELETE_TARGET_REMAINS")
    for ref in protected_refs:
        if artifact_ref(ref["path"]) != ref:
            raise RuntimeError("PROTECTED_POST_DELETE_DRIFT:" + ref["path"])

    result = {"schema_version": "exact78-cards042-v2-retired-cleanup-result",
              "task_id": TASK, "status": "PASSED_RETIRED_ARTIFACTS_REMOVED",
              "removed_files": len(inventory), "removed_bytes": plan["target_bytes"],
              "inventory_sha256": digest.hexdigest(), "deletion_plan": artifact_ref(ROOT / "DELETION_PLAN.json"),
              "retained": {"current_stage_task_ids": sorted(KEEP_TASKS - {TASK}),
                           "old_runtime_records": "JSON/log/text only",
                           "terminal_receipts": "retained for audit",
                           "current_motion_bundle": load_json(BASELINE)["retained_motion_bundle"]},
              "source_and_doc_cleanup": REMOVED_SOURCE_RECORD,
              "recoverability": "Bulky retired media/cache deleted; compact SHA inventory and terminal receipts retained.",
              "other_ai_branch_unchanged": True, "shared_assets_untouched": True,
              "external_source_data_untouched": True, "training_eligible": False}
    atomic_json(ROOT / "RESULT.json", result)
    atomic_json(RECEIPT, {"task_id": TASK, "status": "PASSED", "result": artifact_ref(ROOT / "RESULT.json"),
                          "baseline": artifact_ref(BASELINE), "branch_lock": artifact_ref(BRANCH_LOCK),
                          "claim_limit": packet["claim_limit"]})
    authority = copy.deepcopy(load_json(AUTHORITY_PATH))
    authority.setdefault("claims", []).extend([
        {"claim": "Exact78 Cards042 V2 是清理后的唯一当前视觉开发基线。",
         "claim_limit": packet["claim_limit"], "scope": "exact78/play_cards_0902_042/current_visual_baseline_v2",
         "status": "DEVELOPMENT_EVIDENCE",
         "evidence": [artifact_ref(BASELINE), artifact_ref(BRANCH_LOCK), artifact_ref(RECEIPT)]},
        {"claim": "旧 Cards042 媒体、缓存和专用脚本已按显式清单退休；紧凑记录保留。",
         "claim_limit": packet["claim_limit"], "scope": "exact78/play_cards_0902_042/v2_retired_cleanup",
         "status": "SUPPORTED_INTERNAL_CONSISTENCY",
         "evidence": [artifact_ref(ROOT / "DELETION_PLAN.json"), artifact_ref(ROOT / "RESULT.json")]},
    ])
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    row = next(item for item in state["tasks"] if item["task_id"] == TASK)
    row.update(status="PASSED", pid=None, proc_start_ticks=None, updated_at=now_iso(), heartbeat_at=now_iso(),
               result=artifact_ref(ROOT / "RESULT.json"), terminal_receipt=artifact_ref(RECEIPT),
               quality="REFERENCE_PROCESS_AND_PROTECTED_SHA_PASS", review="MACHINE_AUDITED")
    state["next_task"] = None
    index = index_body()
    for item in index["task_packets"]:
        if item["task_id"] == TASK:
            item["execution_allowed"] = False
    index.update(packet_revision="EXACT78_CARDS042_V2_CLEANUP_TERMINAL", status="PASS_NO_ACTIVE_TASKS",
                 supersedes_index=artifact_ref(INDEX))
    published = publish_bundle(authority, state, event_type="EXACT78_CARDS042_V2_CLEANUP_COMPLETE",
                               expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
                               generator_path=Path(__file__), task_packet_index_path=INDEX,
                               task_packet_index_value=index)
    validation = subprocess.run([sys.executable, "-B", "-m", "chaoyang.cli", "validate-governance"],
                                cwd=REPO_ROOT, env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
                                capture_output=True, text=True)
    if validation.returncode:
        raise RuntimeError("POST_CLEANUP_INVALID:" + validation.stdout[-4000:] + validation.stderr[-4000:])
    print(json.dumps({"status": "PASSED", "revision": published["governance_revision"],
                      "result": artifact_ref(ROOT / "RESULT.json"), "removed_files": len(inventory),
                      "removed_bytes": plan["target_bytes"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
