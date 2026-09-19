#!/usr/bin/env python3
"""Publish honest CPU-only W0 eligibility/blocker terminals.

This runner deliberately does not implement Object6D, Interaction, Contact or
Robot refinement.  It closes a node only when its upstream evidence makes the
requested computation ineligible.  It never turns an all-unknown placeholder
into an algorithm artifact.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence
import uuid

from chaoyang.governance.robot15h_task_specs_v1 import (
    DEPENDENCIES,
    WINDOW_RUN_ID,
    build_packet,
)


ROOT = Path(__file__).resolve().parents[3]
INVENTORY = ROOT / "_run/current/0915_robot15h_window_start_inventory_v1/attempts/attempt_0001/BATCH_MANIFEST.json"
TASKS = {
    "0915_robot15h_geometry_object6d_wave0_v1": {
        "phase": "ROBOT15H_OBJECT6D_W0_ELIGIBILITY",
        "ledger": "OBJECT6D_ELIGIBILITY_LEDGER.json",
        "quality_key": "geometry_quality_status",
    },
    "0915_robot15h_interaction_occlusion_v1": {
        "phase": "ROBOT15H_INTERACTION_W0_UPSTREAM_GATE",
        "ledger": "INTERACTION_ELIGIBILITY_LEDGER.json",
        "quality_key": "interaction_quality_status",
    },
    "0915_robot15h_contact_dual_evidence_v1": {
        "phase": "ROBOT15H_CONTACT_W0_UPSTREAM_GATE",
        "ledger": "CONTACT_ELIGIBILITY_LEDGER.json",
        "quality_key": "contact_quality_status",
    },
    "0915_robot15h_robot_relative_refinement_v1": {
        "phase": "ROBOT15H_R1_W0_LOCAL_EVIDENCE_GATE",
        "ledger": "R1_ELIGIBILITY_LEDGER.json",
        "quality_key": "r1_quality_status",
    },
}
AUTHORITY = "DEVELOPMENT_ELIGIBILITY_ONLY_NON_CONTROL_NON_DEPLOYABLE"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    candidate = path.resolve(strict=True)
    return {"path": str(candidate), "bytes": candidate.stat().st_size, "sha256": sha256(candidate)}


def verify_ref(value: Mapping[str, Any]) -> Path:
    path = Path(str(value["path"])).resolve(strict=True)
    if path.stat().st_size != int(value["bytes"]) or sha256(path) != value["sha256"]:
        raise RuntimeError(f"artifact reference drift: {path}")
    return path


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def process_start_ticks(pid: int) -> int:
    return int(Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()[21])


def w0_rows(inventory: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = [dict(row) for row in inventory.get("sessions", []) if row.get("wave") == "W0"]
    if inventory.get("cohort_denominator") != 220 or len(rows) != 4:
        raise RuntimeError("frozen 0915 W0 inventory must contain exactly four of 220 sessions")
    identities = [str(row.get("session_id")) for row in rows]
    if len(set(identities)) != len(identities):
        raise RuntimeError("duplicate W0 session identity")
    return rows


def rows_by_session(batch: Mapping[str, Any], expected: Sequence[str]) -> dict[str, Mapping[str, Any]]:
    rows = batch.get("sessions", batch.get("rows"))
    if not isinstance(rows, list):
        raise RuntimeError("upstream batch lacks per-session rows")
    mapped = {str(row.get("session_id")): row for row in rows if isinstance(row, Mapping)}
    if set(mapped) != set(expected):
        raise RuntimeError("upstream batch session identity differs from frozen W0")
    return mapped


def depth_admitted(row: Mapping[str, Any]) -> bool:
    """Require both the batch terminal and its SHA-bound session result."""
    if row.get("status") != "PASSED" or not isinstance(row.get("result"), Mapping):
        return False
    result = load_json(verify_ref(row["result"]))
    return result.get("status") == "PASSED" and result.get("consumption_authorized") is True


def object_mask_admitted(row: Mapping[str, Any], batch: Mapping[str, Any]) -> bool:
    admitted_count = row.get("object_semantic_admitted")
    if admitted_count is None:
        admitted_count = row.get("instance_counts", {}).get("stable_consumer_allowed", 0)
    return (
        batch.get("object_mask_consumer_allowed") is True
        and row.get("object_mask_consumer_allowed") is True
        and int(admitted_count) > 0
    )


def geometry_eligibility_rows(
    inventory_rows: Sequence[Mapping[str, Any]],
    depth_batch: Mapping[str, Any],
    mask_batch: Mapping[str, Any],
) -> list[dict[str, Any]]:
    identities = [str(row["session_id"]) for row in inventory_rows]
    depths = rows_by_session(depth_batch, identities)
    masks = rows_by_session(mask_batch, identities)
    result: list[dict[str, Any]] = []
    for item in inventory_rows:
        session_id = str(item["session_id"])
        depth_ok = depth_admitted(depths[session_id])
        mask_ok = object_mask_admitted(masks[session_id], mask_batch)
        if not depth_ok:
            status, blocker = "BLOCKED_UPSTREAM_DEPTH", "DEPTH_NOT_CONSUMER_ADMITTED"
        elif not mask_ok:
            status, blocker = "BLOCKED_UPSTREAM_OBJECT_MASK", "NO_INDEPENDENT_CONSUMER_ADMITTED_TASK_OBJECT_MASK"
        else:
            status, blocker = "ELIGIBLE_FOR_SEPARATE_OBJECT6D_EXECUTION", None
        result.append({
            "session_id": session_id,
            "task": item["task"],
            "source_group": item["source_group"],
            "frame_count": int(item["frame_count"]),
            "status": status,
            "first_blocker": blocker,
            "depth_consumer_admitted": depth_ok,
            "object_mask_consumer_admitted": mask_ok,
            "object6d_attempted": False,
            "object6d_artifact_emitted": False,
            "review_video_emitted": False,
        })
    return result


def blocked_rows_from_upstream(
    inventory_rows: Sequence[Mapping[str, Any]],
    upstream: Mapping[str, Any],
    stage: str,
) -> list[dict[str, Any]]:
    identities = [str(row["session_id"]) for row in inventory_rows]
    prior = rows_by_session(upstream, identities)
    result: list[dict[str, Any]] = []
    for item in inventory_rows:
        session_id = str(item["session_id"])
        previous = prior[session_id]
        if stage == "interaction":
            status = "BLOCKED_UPSTREAM_GEOMETRY"
            blocker = "OBJECT6D_NOT_AVAILABLE"
        elif stage == "contact":
            status = "BLOCKED_UPSTREAM_INTERACTION"
            blocker = "INTERACTION_EVIDENCE_NOT_AVAILABLE"
        elif stage == "r1":
            status = "BLOCKED_LOCAL_EVIDENCE"
            blocker = "NO_ADMITTED_LOCAL_CONTACT_WINDOW"
        else:  # pragma: no cover - protected by CLI choices
            raise ValueError(stage)
        prior_status = str(previous.get("status"))
        # This runner is intentionally blocker-only.  An admitted predecessor
        # requires the real algorithm runner rather than a synthetic bundle.
        admitted = prior_status.startswith("PASSED") or prior_status.startswith("ELIGIBLE")
        if admitted:
            raise RuntimeError(f"{stage} needs a real algorithm runner for {session_id}; blocker-only runner refused")
        row = {
            "session_id": session_id,
            "task": item["task"],
            "source_group": item["source_group"],
            "frame_count": int(item["frame_count"]),
            "status": status,
            "first_blocker": blocker,
            "upstream_status": prior_status,
            "algorithm_attempted": False,
            "algorithm_artifact_emitted": False,
            "review_video_emitted": False,
        }
        if stage == "r1":
            row.update({
                "r1_e_status": "BLOCKED_LOCAL_EVIDENCE",
                "r1_h_status": "BLOCKED_LOCAL_EVIDENCE",
                "r1_e_candidate_emitted": False,
                "r1_h_candidate_emitted": False,
                "r0_preserved": True,
                "r2_independent_path_unblocked": True,
            })
        result.append(row)
    return result


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {"total": len(rows), "attempted": 0, "passed": 0}
    for row in rows:
        status = str(row["status"])
        counts[status] = counts.get(status, 0) + 1
        if row.get("object6d_attempted") is True or row.get("algorithm_attempted") is True:
            counts["attempted"] += 1
        if status == "PASSED":
            counts["passed"] += 1
    return counts


def task_paths(task_id: str) -> tuple[Path, Path]:
    output = ROOT / f"_run/current/{task_id}/attempts/attempt_0001"
    receipt_name = task_id.upper() + "_RESULT.json"
    return output, ROOT / "tasks/receipts" / receipt_name


def validate_route(task_id: str) -> tuple[dict[str, Any], Path]:
    packet_path = ROOT / f"tasks/current/{task_id}/TASK_PACKET.json"
    packet = load_json(packet_path)
    if packet != build_packet(task_id) or packet.get("weights") != "ABSENT":
        raise RuntimeError("current task packet differs from frozen weights-ABSENT spec")
    state = load_json(ROOT / "docs/governance/LONG_HORIZON_TASK_STATE.json")
    if state.get("next_task", {}).get("task_id") != task_id:
        raise RuntimeError(f"{task_id} is not current next_task")
    index = load_json(ROOT / "tasks/current/INDEX.json")
    route = next((row for row in index.get("task_packets", []) if row.get("task_id") == task_id), None)
    if route is None or route.get("execution_allowed") is not True or route.get("packet_sha256") != sha256(packet_path):
        raise RuntimeError(f"{task_id} is not SHA-bound routable")
    return packet, packet_path


def heartbeat(task_id: str, phase: str) -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "chaoyang.governance.heartbeat_task", "--task-id", task_id,
         "--pid", str(os.getpid()), "--status", "RUNNING", "--phase", phase],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if completed.returncode:
        raise RuntimeError("governance heartbeat failed: " + (completed.stderr or completed.stdout)[-4000:])


def dependency_artifacts(task_id: str) -> list[Path]:
    paths: list[Path] = []
    for dependency in DEPENDENCIES[task_id]:
        root = ROOT / f"_run/current/{dependency}/attempts/attempt_0001"
        paths.append(root / "RESULT.json")
        batch = root / "BATCH_RESULT.json"
        ledger = root / TASKS.get(dependency, {}).get("ledger", "")
        if batch.is_file():
            paths.append(batch)
        if ledger.name and ledger.is_file():
            paths.append(ledger)
    return paths


def _find_dependency(task_id: str, fragment: str) -> Path:
    candidates = [value for value in DEPENDENCIES[task_id] if fragment in value]
    if not candidates:
        raise RuntimeError(f"expected a {fragment!r} dependency for {task_id}")
    # A bounded recovery/quality successor may retain its failed predecessor as
    # another direct dependency.  DAG order is chronological, so the last
    # matching node is the evidence authority consumed by this gate.
    return ROOT / f"_run/current/{candidates[-1]}/attempts/attempt_0001"


def build_rows(task_id: str, inventory_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if task_id == "0915_robot15h_geometry_object6d_wave0_v1":
        depth_root = _find_dependency(task_id, "foundationstereo")
        sam_root = _find_dependency(task_id, "sam31")
        return geometry_eligibility_rows(
            inventory_rows, load_json(depth_root / "BATCH_RESULT.json"), load_json(sam_root / "BATCH_RESULT.json")
        )
    if task_id == "0915_robot15h_interaction_occlusion_v1":
        prior = _find_dependency(task_id, "geometry_object6d")
        return blocked_rows_from_upstream(
            inventory_rows, load_json(prior / TASKS["0915_robot15h_geometry_object6d_wave0_v1"]["ledger"]), "interaction"
        )
    if task_id == "0915_robot15h_contact_dual_evidence_v1":
        prior = _find_dependency(task_id, "interaction_occlusion")
        return blocked_rows_from_upstream(
            inventory_rows, load_json(prior / TASKS["0915_robot15h_interaction_occlusion_v1"]["ledger"]), "contact"
        )
    if task_id == "0915_robot15h_robot_relative_refinement_v1":
        prior = _find_dependency(task_id, "contact_dual_evidence")
        return blocked_rows_from_upstream(
            inventory_rows, load_json(prior / TASKS["0915_robot15h_contact_dual_evidence_v1"]["ledger"]), "r1"
        )
    raise ValueError(task_id)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True, choices=tuple(TASKS))
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--executor-epoch", required=True, type=int)
    parser.add_argument("--fencing-token", required=True)
    args = parser.parse_args()
    task_id = args.task_id
    spec = TASKS[task_id]
    output, receipt = task_paths(task_id)
    packet, packet_path = validate_route(task_id)
    if args.output_root.resolve() != output.resolve() or args.receipt.resolve() != receipt.resolve():
        raise RuntimeError("fixed task namespace mismatch")
    if args.executor_epoch < 1 or len(args.fencing_token) < 16:
        raise RuntimeError("positive epoch and fencing token >=16 characters required")
    for path in (output, receipt):
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"fresh output required: {path}")

    inputs = [INVENTORY, *dependency_artifacts(task_id)]
    for path in inputs:
        if not path.is_file():
            raise RuntimeError(f"required upstream artifact missing: {path}")
    signature_payload = {
        "schema_version": "0915-robot15h-downstream-terminal-signature-v1",
        "task_id": task_id,
        "window_run_id": WINDOW_RUN_ID,
        "weights": "ABSENT",
        "gpu_used": False,
        "model_run_performed": False,
        "executor_epoch": args.executor_epoch,
        "task_packet": ref(packet_path),
        "inputs": [ref(path) for path in inputs],
        "runner": ref(Path(__file__)),
    }
    signature = {**signature_payload, "run_signature_sha256": canonical_sha(signature_payload)}
    output.mkdir(parents=True)
    atomic_json(output / "RUN_SIGNATURE.json", signature)
    atomic_json(output / "CLAIM.json", {
        "schema_version": "0915-robot15h-downstream-terminal-claim-v1",
        "task_id": task_id,
        "window_run_id": WINDOW_RUN_ID,
        "status": "CLAIMED",
        "weights": "ABSENT",
        "pid": os.getpid(),
        "proc_start_ticks": process_start_ticks(os.getpid()),
        "executor_epoch": args.executor_epoch,
        "fencing_token_sha256": hashlib.sha256(args.fencing_token.encode()).hexdigest(),
        "unique_write_root": str(output.resolve()),
        "run_signature_sha256": signature["run_signature_sha256"],
    })
    heartbeat(task_id, spec["phase"])

    inventory = load_json(INVENTORY)
    rows = build_rows(task_id, w0_rows(inventory))
    counts = summarize(rows)
    statuses = sorted({str(row["status"]) for row in rows})
    quality_status = statuses[0] if len(statuses) == 1 else "MIXED_TERMINAL"
    ledger = {
        "schema_version": "0915-robot15h-downstream-eligibility-ledger-v1",
        "task_id": task_id,
        "window_run_id": WINDOW_RUN_ID,
        "status": "COMPLETED_ALL_TERMINAL",
        spec["quality_key"]: quality_status,
        "weights": "ABSENT",
        "gpu_used": False,
        "rows": rows,
        "counts": counts,
        "r0_independent_and_preserved": True,
        "r2_independent_path_unblocked": True,
        "control_ground_truth": False,
        "training_eligible": False,
        "physical_deployment_authorized": False,
        "source_mutated": False,
    }
    ledger_path = output / spec["ledger"]
    atomic_json(ledger_path, ledger)
    batch = {
        "schema_version": "0915-robot15h-downstream-terminal-batch-v1",
        "task_id": task_id,
        "window_run_id": WINDOW_RUN_ID,
        "status": "COMPLETED_ALL_TERMINAL",
        spec["quality_key"]: quality_status,
        "counts": counts,
        "sessions": rows,
        "algorithm_attempted": False,
        "algorithm_artifact_emitted": False,
        "fake_all_unknown_artifact_emitted": False,
        "review_video_emitted": False,
        "r0_independent_and_preserved": True,
        "r2_independent_path_unblocked": True,
    }
    atomic_json(output / "BATCH_RESULT.json", batch)
    atomic_json(output / "METRICS.json", {
        "schema_version": "0915-robot15h-downstream-terminal-metrics-v1",
        "task_id": task_id,
        "session_count": len(rows),
        "counts": counts,
        "algorithm_attempted_count": counts["attempted"],
        "algorithm_pass_count": counts["passed"],
        "r0_affected": False,
        "r2_independent_path_affected": False,
    })
    result = {
        "schema_version": "0915-robot15h-downstream-terminal-result-v1",
        "task_id": task_id,
        "window_run_id": WINDOW_RUN_ID,
        "status": "PASSED",
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "first_blocker": quality_status if quality_status.startswith("BLOCKED") else None,
        spec["quality_key"]: quality_status,
        "counts": counts,
        "weights": "ABSENT",
        "authority": AUTHORITY,
        "algorithm_attempted": False,
        "algorithm_artifact_emitted": False,
        "fake_all_unknown_artifact_emitted": False,
        "review_video_emitted": False,
        "r0_independent_and_preserved": True,
        "r2_independent_path_unblocked": True,
        "control_ground_truth": False,
        "training_eligible": False,
        "physical_deployment_authorized": False,
        "ledger": ref(ledger_path),
        "batch_result": ref(output / "BATCH_RESULT.json"),
        "metrics": ref(output / "METRICS.json"),
        "source_mutated": False,
        "claim_limit": packet["claim_limit"],
    }
    atomic_json(output / "RESULT.json", result)
    atomic_json(receipt, {**result, "result": ref(output / "RESULT.json")})
    atomic_json(output / "RUN_RECEIPT.json", {
        "schema_version": "0915-robot15h-downstream-terminal-run-receipt-v1",
        "task_id": task_id,
        "window_run_id": WINDOW_RUN_ID,
        "status": "PASSED",
        "result": ref(output / "RESULT.json"),
        "terminal_receipt": ref(receipt),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
