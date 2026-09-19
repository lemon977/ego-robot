#!/usr/bin/env python3
"""Close Robot15h W1 nodes that have no executable evidence scope.

This is deliberately a scope gate, not a model runner.  It consumes the
immutable release-candidate matrix and publishes one honest terminal row for
every frozen W1 session without opening the source videos.  In particular it
must not turn an exported W0 development artifact into W1 quality authority.
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
from typing import Any
import uuid

from chaoyang.governance.robot15h_task_specs_v1 import WINDOW_RUN_ID, build_packet


ROOT = Path(__file__).resolve().parents[3]
INVENTORY = ROOT / "_run/current/0915_robot15h_window_start_inventory_v1/attempts/attempt_0001/BATCH_MANIFEST.json"
MATRIX = ROOT / "_run/current/0915_robot15h_release_candidate_v1/attempts/attempt_0001/CAPABILITY_MATRIX.json"
TASKS: dict[str, dict[str, Any]] = {
    "0915_robot15h_hawor_waves_v1": {
        "phase": "ROBOT15H_HAWOR_W1_RELEASE_SCOPE_GATE",
        "capability": "hawor_direct_observed",
        "status": "BLOCKED_UPSTREAM_RELEASE_SCOPE",
        "blocker": "W0_HAWOR_HAS_ZERO_QUALITY_ADMITTED_SCOPE",
        "reason": (
            "The H9 release matrix grants no W1 HaWoR quality scope.  Running the "
            "pinned weight would consume new RGB without an admitted expansion row."
        ),
    },
    "0915_robot15h_sam31_waves_v1": {
        "phase": "ROBOT15H_SAM31_W1_PROMPT_SOURCE_GATE",
        "capability": "sam31_hand_temporal",
        "status": "BLOCKED_UPSTREAM_PROMPT_SOURCE",
        "blocker": "MISSING_RELEASE_ADMITTED_HAWOR_PROMPT_SOURCE",
        "reason": (
            "The admitted SAM3.1 scope is the direct-anchored hand proxy, but the frozen "
            "initializer requires same-session HaWoR boxes and points.  W1 HaWoR is not "
            "release-admitted, so changing to text-only or fabricated boxes is forbidden."
        ),
    },
    "0915_robot15h_robot_waves_v1": {
        "phase": "ROBOT15H_ROBOT_W1_UPSTREAM_GATE",
        "capability": "kai22_r0",
        "status": "BLOCKED_UPSTREAM_R0",
        "blocker": "NO_RELEASE_ADMITTED_W1_HAWOR_OR_KAI22_R0",
        "reason": (
            "FoundationStereo depth alone cannot create R0, R1, or R2.  No W1 HaWoR/R0 "
            "trajectory is available, so Robot output would be fabricated."
        ),
    },
}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    candidate = path.resolve(strict=True)
    return {
        "path": str(candidate),
        "bytes": candidate.stat().st_size,
        "sha256": sha256(candidate),
    }


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


def validate_route(task_id: str) -> tuple[dict[str, Any], Path]:
    packet_path = ROOT / f"tasks/current/{task_id}/TASK_PACKET.json"
    packet = load_json(packet_path)
    if packet != build_packet(task_id):
        raise RuntimeError("current task packet differs from the frozen W1 scope-gate spec")
    state = load_json(ROOT / "docs/governance/LONG_HORIZON_TASK_STATE.json")
    if state.get("next_task", {}).get("task_id") != task_id:
        raise RuntimeError(f"{task_id} is not current next_task")
    index = load_json(ROOT / "tasks/current/INDEX.json")
    route = next((row for row in index.get("task_packets", []) if row.get("task_id") == task_id), None)
    if (
        route is None
        or route.get("execution_allowed") is not True
        or route.get("packet_sha256") != sha256(packet_path)
    ):
        raise RuntimeError(f"{task_id} is not SHA-bound routable")
    return packet, packet_path


def heartbeat(task_id: str, phase: str) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "chaoyang.governance.heartbeat_task",
            "--task-id",
            task_id,
            "--pid",
            str(os.getpid()),
            "--status",
            "RUNNING",
            "--phase",
            phase,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError("governance heartbeat failed: " + (completed.stderr or completed.stdout)[-4000:])


def w1_rows() -> list[dict[str, Any]]:
    manifest = load_json(INVENTORY)
    rows = [dict(row) for row in manifest.get("sessions", []) if row.get("wave") == "W1"]
    if len(rows) != 8 or sum(int(row["frame_count"]) for row in rows) != 2332:
        raise RuntimeError("frozen W1 denominator drift")
    if len({row["source_group"] for row in rows}) != len(rows):
        raise RuntimeError("W1 source groups are not disjoint")
    return rows


def matrix_rows(capability: str) -> list[dict[str, Any]]:
    matrix = load_json(MATRIX)
    if matrix.get("window_run_id") != WINDOW_RUN_ID:
        raise RuntimeError("release matrix run identity drift")
    return [dict(row) for row in matrix.get("rows", []) if row.get("capability_id") == capability]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True, choices=tuple(TASKS))
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--executor-epoch", required=True, type=int)
    parser.add_argument("--fencing-token", required=True)
    args = parser.parse_args()

    task_id = args.task_id
    config = TASKS[task_id]
    packet, packet_path = validate_route(task_id)
    output = args.output_root.resolve()
    receipt = args.receipt.resolve()
    expected_output = (ROOT / f"_run/current/{task_id}/attempts/attempt_0001").resolve()
    expected_receipt = (ROOT / "tasks/receipts" / f"{task_id.upper()}_RESULT.json").resolve()
    if output != expected_output or receipt != expected_receipt:
        raise RuntimeError("W1 scope-gate namespace drift")
    if args.executor_epoch < 1 or len(args.fencing_token) < 16:
        raise RuntimeError("positive epoch and fencing token >=16 characters required")
    for path in (output, receipt):
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"fresh W1 scope-gate output required: {path}")

    rows = w1_rows()
    release_rows = matrix_rows(str(config["capability"]))
    heartbeat(task_id, str(config["phase"]))
    signature_payload = {
        "schema_version": "0915-robot15h-w1-scope-gate-run-signature-v1",
        "task_id": task_id,
        "window_run_id": WINDOW_RUN_ID,
        "executor_epoch": args.executor_epoch,
        "weights": packet["weights"],
        "inputs": {
            "task_packet": ref(packet_path),
            "batch_manifest": ref(INVENTORY),
            "release_matrix": ref(MATRIX),
        },
        "runtime": {"runner": ref(Path(__file__))},
        "source_video_open_count": 0,
        "gpu_used": False,
    }
    signature = {
        **signature_payload,
        "run_signature_sha256": canonical_sha(signature_payload),
    }
    output.mkdir(parents=True)
    atomic_json(output / "RUN_SIGNATURE.json", signature)
    atomic_json(output / "CLAIM.json", {
        "schema_version": "0915-robot15h-w1-scope-gate-claim-v1",
        "task_id": task_id,
        "window_run_id": WINDOW_RUN_ID,
        "status": "CLAIMED",
        "weights": packet["weights"],
        "claimed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "pid": os.getpid(),
        "proc_start_ticks": process_start_ticks(os.getpid()),
        "executor_epoch": args.executor_epoch,
        "fencing_token_sha256": hashlib.sha256(args.fencing_token.encode()).hexdigest(),
        "run_signature_sha256": signature["run_signature_sha256"],
        "unique_write_root": str(output),
    })

    session_rows = [{
        "session_id": row["session_id"],
        "task": row["task"],
        "split": row["split"],
        "source_group": row["source_group"],
        "frame_count": int(row["frame_count"]),
        "status": config["status"],
        "first_blocker": config["blocker"],
        "algorithm_attempted": False,
        "algorithm_artifact_emitted": False,
        "quality_admitted": False,
        "source_video_opened": False,
    } for row in rows]
    scope_audit = {
        "schema_version": "0915-robot15h-w1-release-scope-audit-v1",
        "task_id": task_id,
        "window_run_id": WINDOW_RUN_ID,
        "capability_id": config["capability"],
        "release_rows": release_rows,
        "release_allowed_rows": sum(bool(row.get("w1_expansion_allowed")) for row in release_rows),
        "decision": config["status"],
        "first_blocker": config["blocker"],
        "reason": config["reason"],
        "source_video_open_count": 0,
        "gpu_used": False,
    }
    atomic_json(output / "SCOPE_AUDIT.json", scope_audit)
    batch = {
        "schema_version": "0915-robot15h-w1-blocked-batch-v1",
        "task_id": task_id,
        "window_run_id": WINDOW_RUN_ID,
        "status": "COMPLETED_ALL_TERMINAL",
        "counts": {
            "total": len(session_rows),
            "attempted": 0,
            "exported": 0,
            "quality_admitted": 0,
            "blocked": len(session_rows),
            "unrun": 0,
            "failed_runtime": 0,
        },
        "sessions": session_rows,
        "source_video_open_count": 0,
        "source_mutated": False,
    }
    atomic_json(output / "BATCH_RESULT.json", batch)
    result = {
        "schema_version": "0915-robot15h-w1-scope-gate-result-v1",
        "task_id": task_id,
        "window_run_id": WINDOW_RUN_ID,
        "status": "REJECTED_QUALITY",
        "first_blocker": config["blocker"],
        "weights": packet["weights"],
        "counts": batch["counts"],
        "scope_audit": ref(output / "SCOPE_AUDIT.json"),
        "batch_result": ref(output / "BATCH_RESULT.json"),
        "source_mutated": False,
        "control_ground_truth": False,
        "training_eligible": False,
        "physical_deployment_authorized": False,
        "claim_limit": packet["claim_limit"],
    }
    atomic_json(output / "RESULT.json", result)
    atomic_json(receipt, {**result, "result": ref(output / "RESULT.json")})
    atomic_json(output / "RUN_RECEIPT.json", {
        "schema_version": "0915-robot15h-w1-scope-gate-run-receipt-v1",
        "task_id": task_id,
        "window_run_id": WINDOW_RUN_ID,
        "status": result["status"],
        "result": ref(output / "RESULT.json"),
        "terminal_receipt": ref(receipt),
    })
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
