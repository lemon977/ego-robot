"""Bounded CPU edit window using the existing governance CAS and task index."""
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
import shutil
import subprocess
import sys
import tarfile
import time

from chaoyang.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH,
    artifact_ref, atomic_json, canonical_bytes, load_json, now_iso,
    process_identity, publish_bundle, validate_artifact_ref,
)
from chaoyang.governance.register_single_task_packet import _validate_packet

TASK = "project_branch_navigation_convergence_v1_20261008"
ATTEMPT = REPO_ROOT / f"_run/current/{TASK}/attempts/attempt_0001"
PACKET = REPO_ROOT / f"tasks/current/{TASK}/TASK_PACKET.json"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
RECEIPT = REPO_ROOT / f"tasks/receipts/{TASK.upper()}.json"
PLAN = "docs/plans/PROJECT_BRANCH_CONVERGENCE_20261008"
WINDOW = 2700
EDIT_FILES = (
    "src/chaoyang/governance/build_human_to_robot_r2_terminal_status.py",
    "src/chaoyang/governance/current_r3_contracts.py",
    "src/chaoyang/governance/exact78_operation_gate.py",
    "src/chaoyang/exact78_baseline/pipeline.py",
    "docs/current/AI_WORK_ENTRY_ZH.md", "docs/current/README_ZH.md",
    "docs/current/HUMAN_TO_ROBOT_BASELINE_V1_ZH.md", "docs/current/PLAN.md",
    "docs/current/GIT_DELIVERY_ZH.md", "docs/current/EXACT78_EXECUTION_ROUTE_ZH.md",
    "docs/current/V5_MOTION.md", "docs/current/V5_HURO.md",
    "README.md", "docs/README.md", "review/two_sessions_20261008/README.md",
)
NEW_FILES = (
    "src/chaoyang/governance/project_branch_navigation_v1.py",
    "src/chaoyang/exact78_baseline/review_labels.py",
    "tests/unit/test_project_branch_navigation_v1.py",
    "docs/current/PROJECT_BRANCH_MAP.json", "docs/current/PROJECT_BRANCH_MAP_ZH.md",
    "review/two_sessions_20261008/SOURCE_LABELS.json",
)
PROTECTED_FILES = (
    "docs/current/EXACT78_VISUAL_BASELINE_V1.json",
    "tasks/receipts/EXACT78_VISUAL_BASELINE_V1.json",
    "docs/current/EXACT78_VISUAL_BASELINE_V1_ZH.md",
    "docs/current/ROBOT100_BATCH_ZH.md",
    "docs/current/ROBOT100_HARDWARE_BASELINE_V16.json",
    "src/chaoyang/ops/run_robot100_world_first_v10.py",
    "src/chaoyang/ops/run_robot100_constraint_review_v16_retry1.py",
    "src/chaoyang/ops/run_robot100_v16_safe_replay.py",
    "_run/current/exact78_takeover_b2_full_v13/attempts/attempt_0001/RESULT.json",
    "_run/current/exact78_takeover_v13_cards042_full_replay_v1_20261008/attempts/attempt_0001/RESULT.json",
    "_run/current/exact78_takeover_v1_two_source_review_v3_20261008/attempts/attempt_0001/RESULT.json",
    "_run/current/robot100_v16_baseline_overlay_20260929/attempts/attempt_0001/BASELINE.json",
    "_run/current/robot100_v16_sync_repair_20261008/attempts/attempt_0002/RESULT.json",
    "docs/current/EXACT78_EXECUTION_ROUTE_V1.json",
    "docs/current/EXACT78_ACTIVE_LINEAGE_V1.json",
    "docs/current/EXACT78_CARDS042_VISUAL_BASELINE_V2.json",
    "docs/current/EXACT78_CARDS042_VISUAL_BRANCH_LOCK_V2.json",
    "docs/current/EXACT78_CARDS042_VISUAL_BASELINE_ZH.md",
    "docs/current/EXACT78_CARDS042_VISUAL_BRANCH_LOCK_ZH.md",
)


def refs(names):
    return {n: artifact_ref(REPO_ROOT / n) for n in names}


def index_body():
    return {k: copy.deepcopy(v) for k, v in load_json(INDEX).items()
            if k not in {"governance_revision", "generation_id", "generated_at", "supersedes_index"}}


def validate_current():
    r = subprocess.run([sys.executable, "-B", "-m", "chaoyang.cli", "validate-governance"],
                       cwd=REPO_ROOT, env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
                       capture_output=True, text=True, timeout=600)
    if r.returncode:
        raise RuntimeError("GOVERNANCE_INVALID:" + r.stdout[-2000:] + r.stderr[-1000:])
    return json.loads(r.stdout)


def register():
    validate_current()
    state, index = copy.deepcopy(load_json(TASK_STATE_PATH)), index_body()
    if state.get("next_task") or any(r.get("execution_allowed") for r in index["task_packets"]):
        raise RuntimeError("ACTIVE_TASK_CONFLICT")
    if any(r.get("task_id") == TASK for r in state["tasks"]) or any(p.exists() for p in (PACKET, ATTEMPT, RECEIPT)):
        raise RuntimeError("FRESH_TASK_REQUIRED")
    freeze = load_json(REPO_ROOT / "docs/plans/EXACT78_V1_TWO_SOURCE_REVIEW_20261008/INPUT_AND_BACKEND_FREEZE.json")
    protected = refs(PROTECTED_FILES)
    for ref in freeze["unchanged_backend"]:
        if Path(ref["path"]).name != "pipeline.py":
            if validate_artifact_ref(ref):
                raise RuntimeError("BACKEND_SHA_DRIFT:" + ref["path"])
            protected[str(Path(ref["path"]).relative_to(REPO_ROOT))] = ref
    # The input adapter and algorithm operation are not part of the old 14 pins.
    protected.update(refs(("src/chaoyang/exact78_baseline/native_input.py",
                           "src/chaoyang/exact78_baseline/pico_source.py",
                           "src/chaoyang/ops/run_exact78_takeover_baseline_v1.py")))
    writer = {"pid": os.getpid(), "proc_start_ticks": process_identity(os.getpid())["start_ticks"],
              "executor_epoch": 1, "fencing_token": secrets.token_hex(24), "scope": TASK}
    inputs = {"protected": protected, "edit_before": refs(EDIT_FILES),
              "publisher": artifact_ref(Path(__file__)),
              "plan": artifact_ref(REPO_ROOT / PLAN / "00_EXECUTION_ZH.md")}
    signature = hashlib.sha256(canonical_bytes(inputs)).hexdigest()
    packet = {
        "schema_version": "chaoyang-r22-task-packet-v1", "task_id": TASK,
        "objective": "Converge branch navigation, live-status projection and review labels without changing algorithms",
        "phase": "CPU_BRANCH_NAVIGATION_CONVERGENCE", "plan_revision": "PROJECT_BRANCH_CONVERGENCE_20261008",
        "execution_revision": TASK.upper(), "session_id": None,
        "deadline_at": (datetime.now().astimezone() + timedelta(seconds=WINDOW + 180)).isoformat(),
        "read_set": ["AGENTS.md", "docs/current/AI_WORK_ENTRY_ZH.md", PLAN + "/00_EXECUTION_ZH.md",
                     "docs/governance/DOC_AUTHORITY_MAP.json", "docs/governance/ALGORITHM_CONTRACT.json",
                     "docs/governance/CURRENT_BASELINE_REGISTRY_V2.json",
                     "docs/current/EXACT78_VISUAL_BASELINE_V1.json", "docs/current/ROBOT100_HARDWARE_BASELINE_V16.json"],
        "write_set": [str(ATTEMPT.relative_to(REPO_ROOT)), str(PACKET.relative_to(REPO_ROOT)),
                      str(RECEIPT.relative_to(REPO_ROOT)), "tasks/current/INDEX.json", "docs/governance",
                      PLAN, "_run/caches/project_branch_convergence_20261008", *EDIT_FILES, *NEW_FILES],
        "prerequisites": ["GOVERNANCE_PASS", "NO_ACTIVE_TASK", "PROTECTED_BASELINES_PINNED", "ZERO_GPU_MODEL_TRAINING"],
        "required_outputs": ["RESULT.json", "CODE_SNAPSHOT.tar.gz", "CODE_SNAPSHOT_MANIFEST.json", "REGRESSION.json"],
        "budgets": {"workers_max": 1, "cpu_threads_total": 2, "gpu_count": 0,
                    "training_steps": 0, "model_calls_max": 0, "wall_seconds_max": WINDOW + 180, "session_attempts_max": 1},
        "stop_conditions": ["protected_sha_drift", "double_writer", "deadline", "one_attempt_terminal"],
        "claim_limit": "Navigation and presentation only; no new visual, metric, control or training qualification. No deletion or MANUS algorithm edits.",
        "writer": writer, "inputs": inputs, "run_signature": signature,
        "commands": [f"{sys.executable} -B -m chaoyang.governance.publish_project_branch_convergence_v1"],
    }
    if errors := _validate_packet(packet):
        raise RuntimeError("INVALID_PACKET:" + ";".join(errors))
    ATTEMPT.mkdir(parents=True)
    PACKET.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(ATTEMPT / "WRITER.json", writer)
    atomic_json(PACKET, packet)
    for name in EDIT_FILES:
        dst = ATTEMPT / "before" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / name, dst)
    state["tasks"].append({"task_id": TASK, "phase": packet["phase"], "attempt": 1, "status": "RUNNING",
        "session": "branch_navigation_only", "updated_at": now_iso(), "heartbeat_at": now_iso(),
        "pid": writer["pid"], "proc_start_ticks": writer["proc_start_ticks"], "gpu_id": None,
        "task_packet": artifact_ref(PACKET), "run_signature": signature,
        "quality": "NOT_ASSESSED", "adoption": "BASELINES_UNCHANGED", "review": "NOT_APPLICABLE"})
    state["next_task"] = {"task_id": TASK, "session": "branch_navigation_only",
        "prerequisites": packet["prerequisites"], "expected_resource": "CPU_TWO_THREADS", "stop_condition": "one terminal publication"}
    index["task_packets"].append({"task_id": TASK, "packet_path": str(PACKET.relative_to(REPO_ROOT)),
        "packet_sha256": artifact_ref(PACKET)["sha256"], "execution_class": "CURRENT_LEDGER_ROUTABLE", "execution_allowed": True})
    index.update(packet_revision=TASK.upper(), status="PASS", supersedes_index=artifact_ref(INDEX))
    publish_bundle(copy.deepcopy(load_json(AUTHORITY_PATH)), state, event_type="BRANCH_NAVIGATION_EDIT_WINDOW_REGISTERED",
                   expected_revision=load_json(RECEIPT_PATH)["governance_revision"], generator_path=Path(__file__),
                   task_packet_index_path=INDEX, task_packet_index_value=index)
    return packet


def publish(state, index, event):
    for module in ("project_branch_navigation_v1", "build_human_to_robot_r2_terminal_status", "current_r3_contracts"):
        importlib.reload(importlib.import_module("chaoyang.governance." + module))
    return publish_bundle(copy.deepcopy(load_json(AUTHORITY_PATH)), state, event_type=event,
        expected_revision=load_json(RECEIPT_PATH)["governance_revision"], generator_path=Path(__file__),
        task_packet_index_path=INDEX, task_packet_index_value=index)


def seal(packet, failure=None):
    protected = refs(tuple(packet["inputs"]["protected"]))
    if protected != packet["inputs"]["protected"]:
        failure = "PROTECTED_SHA_DRIFT"
    if failure is None:
        regression = load_json(ATTEMPT / "REGRESSION.json")
        if regression.get("status") != "PASS":
            raise RuntimeError("REGRESSION_NOT_PASSED")
        from chaoyang.governance.project_branch_navigation_v1 import build_branch_map
        atomic_json(REPO_ROOT / "docs/current/PROJECT_BRANCH_MAP.json", build_branch_map(REPO_ROOT))
        from chaoyang.governance.project_branch_navigation_v1 import snapshot_paths
        rows = [artifact_ref(p) for p in snapshot_paths(REPO_ROOT)]
        atomic_json(ATTEMPT / "CODE_SNAPSHOT_MANIFEST.json", {
            "schema_version": "PROJECT_BRANCH_CODE_SNAPSHOT_V1", "files": rows,
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
            "remote_commit_claimed": False, "contains_models_videos_external_data": False,
            "claim_limit": "Scoped server code/config/contracts snapshot; environment, vendor and model manifests remain external prerequisites."})
        with tarfile.open(ATTEMPT / "CODE_SNAPSHOT.tar.gz", "x:gz") as tar:
            for row in rows:
                p = Path(row["path"])
                if artifact_ref(p) != row:
                    raise RuntimeError("SNAPSHOT_SOURCE_DRIFT")
                tar.add(p, arcname=str(p.relative_to(REPO_ROOT)), recursive=False)
        outputs = {name: artifact_ref(ATTEMPT / name) for name in
                   ("REGRESSION.json", "CODE_SNAPSHOT_MANIFEST.json", "CODE_SNAPSHOT.tar.gz")}
    else:
        outputs = {}
    result = {"schema_version": "PROJECT_BRANCH_CONVERGENCE_RESULT_V1", "task_id": TASK,
              "status": "PASSED" if failure is None else "FAILED_RUNTIME_FINAL", "failure_code": failure,
              "protected_before": packet["inputs"]["protected"], "protected_after": protected,
              "edit_before": packet["inputs"]["edit_before"], "edit_after": refs(EDIT_FILES), "outputs": outputs,
              "files_deleted": 0, "models_run": 0, "training_steps": 0, "numeric_gates_changed": False,
              "existing_media_overwritten": False, "remote_git_commit_claimed": False,
              "claim_limit": packet["claim_limit"]}
    atomic_json(ATTEMPT / "RESULT.json", result)
    atomic_json(RECEIPT, {"schema_version": "PROJECT_BRANCH_CONVERGENCE_RECEIPT_V1",
                        "task_id": TASK, "status": result["status"], "result": artifact_ref(ATTEMPT / "RESULT.json"),
                        "baseline_pointers_changed": False, "claim_limit": packet["claim_limit"]})
    state, index = copy.deepcopy(load_json(TASK_STATE_PATH)), index_body()
    row = next(r for r in state["tasks"] if r.get("task_id") == TASK)
    row.update(status=result["status"], pid=None, proc_start_ticks=None, updated_at=now_iso(), heartbeat_at=now_iso(),
               result=artifact_ref(ATTEMPT / "RESULT.json"), terminal_receipt=artifact_ref(RECEIPT), quality="NAVIGATION_ONLY")
    state["next_task"] = None
    next(r for r in index["task_packets"] if r.get("task_id") == TASK)["execution_allowed"] = False
    index.update(status="PASS_NO_ACTIVE_TASKS", supersedes_index=artifact_ref(INDEX))
    published = publish(state, index, "BRANCH_NAVIGATION_CONVERGENCE_TERMINAL")
    validation = validate_current()
    atomic_json(ATTEMPT / "POST_PUBLICATION_VALIDATION.json", validation)
    print(json.dumps({"status": result["status"], "revision": published["governance_revision"],
                      "result": artifact_ref(ATTEMPT / "RESULT.json")}), flush=True)


def main():
    packet = register()
    print(json.dumps({"status": "READY", "task_id": TASK, "attempt": str(ATTEMPT)}), flush=True)
    deadline = time.monotonic() + WINDOW
    try:
        while time.monotonic() < deadline:
            ready, _, _ = select.select([sys.stdin], [], [], min(30, deadline-time.monotonic()))
            if ready:
                if sys.stdin.readline().strip() != "PUBLISH":
                    raise RuntimeError("EDIT_WINDOW_ABORTED")
                seal(packet)
                return 0
            state = copy.deepcopy(load_json(TASK_STATE_PATH))
            row = next(r for r in state["tasks"] if r.get("task_id") == TASK)
            row.update(heartbeat_at=now_iso(), updated_at=now_iso())
            publish_bundle(copy.deepcopy(load_json(AUTHORITY_PATH)), state, event_type="BRANCH_NAVIGATION_HEARTBEAT",
                           expected_revision=load_json(RECEIPT_PATH)["governance_revision"], generator_path=Path(__file__))
        raise RuntimeError("EDIT_WINDOW_DEADLINE")
    except Exception as exc:
        if not (ATTEMPT / "RESULT.json").exists():
            seal(packet, str(exc))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
