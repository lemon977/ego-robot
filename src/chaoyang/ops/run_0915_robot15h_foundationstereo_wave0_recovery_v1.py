#!/usr/bin/env python3
"""Recover only the interrupted fourth FoundationStereo W0 session."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

from chaoyang.governance.robot15h_task_specs_v1 import (
    FOUNDATION_WEIGHT,
    WINDOW_RUN_ID,
    build_packet,
)
from chaoyang.ops import run_0915_robot15h_foundationstereo_wave0_v1 as base
from chaoyang.ops import run_0915_foundationstereo_encoded_domain_canary_v1 as frozen
from chaoyang.ops import run_0915_foundationstereo_single_session_canary_v1 as common


ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "0915_robot15h_foundationstereo_wave0_recovery_v1"
PHASE = "ROBOT15H_FOUNDATIONSTEREO_WAVE0_BOUNDED_RECOVERY"
PREDECESSOR = ROOT / "_run/current/0915_robot15h_foundationstereo_wave0_v1/attempts/attempt_0001"
OUTPUT = ROOT / f"_run/current/{TASK_ID}/attempts/attempt_0001"
VISUAL = ROOT / "docs/current/visuals/0915_ROBOT15H_W0_FOUNDATIONSTEREO_RECOVERY_V1"
TERMINAL_RECEIPT = ROOT / "tasks/receipts/0915_ROBOT15H_FOUNDATIONSTEREO_WAVE0_RECOVERY_V1_RESULT.json"
GPU_LAUNCHER = ROOT / "src/chaoyang/ops/foundationstereo_gpu_python.sh"
GPU_LEASE_WRAPPER = ROOT / "src/chaoyang/ops/run_gpu_command_with_v71_lease.py"
RECOVERY_SESSION = "get_potato_chips_0915_042"
REUSED_SESSIONS = (
    "play_cards_0915_031",
    "play_cards_0915_119",
    "get_potato_chips_0915_007",
)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def ref(path: Path) -> dict[str, Any]:
    return common.file_ref(path)


def sha256(path: Path) -> str:
    return common.sha256(path)


def atomic_json(path: Path, value: Any) -> None:
    common.atomic_json(path, value)


def atomic_json_new(path: Path, value: Any) -> None:
    common.atomic_json_new(path, value)


def canonical_sha(value: Any) -> str:
    return common.canonical_sha256(value)


def validate_route() -> tuple[dict[str, Any], Path]:
    packet_path = ROOT / f"tasks/current/{TASK_ID}/TASK_PACKET.json"
    packet = load_json(packet_path)
    if packet != build_packet(TASK_ID) or packet.get("weights") != [FOUNDATION_WEIGHT]:
        raise RuntimeError("current FoundationStereo recovery packet differs from frozen spec")
    state = load_json(ROOT / "docs/governance/LONG_HORIZON_TASK_STATE.json")
    if state.get("next_task", {}).get("task_id") != TASK_ID:
        raise RuntimeError("FoundationStereo recovery is not current next_task")
    index = load_json(ROOT / "tasks/current/INDEX.json")
    route = next((row for row in index.get("task_packets", []) if row.get("task_id") == TASK_ID), None)
    if route is None or route.get("execution_allowed") is not True or route.get("packet_sha256") != sha256(packet_path):
        raise RuntimeError("FoundationStereo recovery is not SHA-bound routable")
    return packet, packet_path


def heartbeat(status: str, gpu_id: int | None = None) -> None:
    command = [
        sys.executable, "-m", "chaoyang.governance.heartbeat_task", "--task-id", TASK_ID,
        "--pid", str(os.getpid()), "--status", status, "--phase", PHASE,
    ]
    if gpu_id is not None:
        command += ["--gpu-id", str(gpu_id)]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    if completed.returncode:
        raise RuntimeError("governance heartbeat failed: " + (completed.stderr or completed.stdout)[-4000:])


def _session_result_path(row: Mapping[str, Any], root: Path) -> Path:
    return root / "sessions" / str(row["task"]) / str(row["session_id"]) / "RESULT.json"


def recovery_scope() -> dict[str, Any]:
    predecessor = load_json(PREDECESSOR / "RESULT.json")
    interruption = load_json(PREDECESSOR / "INTERRUPTION_EVIDENCE.json")
    preflight = load_json(PREDECESSOR / "PREFLIGHT_BATCH.json")
    if predecessor.get("status") != "FAILED_RUNTIME_FINAL":
        raise RuntimeError("predecessor is not the recorded interrupted terminal")
    if interruption.get("completed_atomic_sessions") != list(REUSED_SESSIONS):
        raise RuntimeError("interruption evidence reused-session set drift")
    rows = base.w0_rows()
    by_id = {str(row["session_id"]): row for row in rows}
    reused = []
    for session_id in REUSED_SESSIONS:
        path = _session_result_path(by_id[session_id], PREDECESSOR)
        result = load_json(path)
        if result.get("status") != "PASSED" or result.get("consumption_authorized") is not True:
            raise RuntimeError(f"predecessor atomic session is not reusable: {session_id}")
        reused.append({"session_id": session_id, "result": ref(path), "review": result["review"]})
    preflight_row = next(row for row in preflight["sessions"] if row["session_id"] == RECOVERY_SESSION)
    if preflight_row.get("gpu_admitted") is not True:
        raise RuntimeError("interrupted session was not preflight-admitted")
    incomplete = next(PREDECESSOR.glob(
        f"sessions/potato_chips/.{RECOVERY_SESSION}.tmp-*"
    ), None)
    if incomplete is None or not incomplete.is_dir():
        raise RuntimeError("preserved interrupted staging evidence is missing")
    return {
        "schema_version": "0915-robot15h-foundationstereo-wave0-recovery-scope-v1",
        "window_run_id": WINDOW_RUN_ID, "status": "BOUNDED_ONE_SESSION_RECOVERY",
        "reused_atomic_sessions": reused, "recovery_session": RECOVERY_SESSION,
        "recovery_frame_count": int(by_id[RECOVERY_SESSION]["frame_count"]),
        "preflight_evidence": preflight_row, "incomplete_staging": {
            "path": str(incomplete.resolve()),
            "frame_artifacts": len(list((incomplete / "frames").glob("*.npz"))),
            "consumed": False, "preserved": True,
        },
        "predecessor_result": ref(PREDECESSOR / "RESULT.json"),
        "interruption_evidence": ref(PREDECESSOR / "INTERRUPTION_EVIDENCE.json"),
        "same_model_and_quality_gates": True, "source_mutated": False,
    }


def build_signature(packet_path: Path, executor_epoch: int, scope: Mapping[str, Any]) -> dict[str, Any]:
    row = next(row for row in base.w0_rows() if row["session_id"] == RECOVERY_SESSION)
    source = Path(str(row["source_stereo"]["path"])).resolve(strict=True)
    payload = {
        "schema_version": "0915-robot15h-foundationstereo-wave0-recovery-signature-v1",
        "task_id": TASK_ID, "window_run_id": WINDOW_RUN_ID, "executor_epoch": executor_epoch,
        "weights": [FOUNDATION_WEIGHT], "recovery_session": {
            "session_id": RECOVERY_SESSION, "source_group": row["source_group"],
            "frame_count": int(row["frame_count"]), "source_stereo": ref(source),
            "camera_params": ref(base.camera_path(row)),
        },
        "reused_atomic_sessions": scope["reused_atomic_sessions"],
        "contracts": {"task_packet": ref(packet_path), "predecessor": ref(PREDECESSOR / "RESULT.json"),
                      "scope_sha256": canonical_sha(scope)},
        "runtime": {"runner": ref(Path(__file__)), "base_runner": ref(Path(base.__file__)),
                    "gpu_launcher": ref(GPU_LAUNCHER), "lease_wrapper": ref(GPU_LEASE_WRAPPER),
                    "checkpoint": ref(ROOT / FOUNDATION_WEIGHT)},
        "pixel_domain": {
            "eyes": {"physical_left": 1, "physical_right": 0},
            "source_transform": "CROP_THEN_RESIZE_ONLY",
            "model_adapter": "SIMULTANEOUS_HORIZONTAL_REFLECTION_NO_CAMERA_SWAP",
            "output_transform": "HORIZONTAL_UNFLIP_TO_PHYSICAL_LEFT",
            "lens_undistortion_applied": False,
        },
    }
    return {**payload, "run_signature_sha256": canonical_sha(payload)}


def validate_claim(path: Path, signature_sha: str, executor_epoch: int, require_descendant: bool) -> None:
    claim = load_json(path)
    pid = claim.get("pid")
    if (
        claim.get("task_id") != TASK_ID or claim.get("weights") != [FOUNDATION_WEIGHT]
        or claim.get("status") != "CLAIMED" or claim.get("executor_epoch") != executor_epoch
        or claim.get("run_signature_sha256") != signature_sha
        or claim.get("unique_write_root") != str(OUTPUT.resolve()) or not isinstance(pid, int)
        or common.process_start_ticks(pid) != claim.get("proc_start_ticks")
    ):
        raise RuntimeError("FoundationStereo recovery writer fence mismatch")
    if require_descendant and not common.process_has_ancestor(os.getpid(), pid):
        raise RuntimeError("FoundationStereo recovery worker is outside writer ancestry")


def run_worker(claim_path: Path, signature_path: Path, executor_epoch: int) -> int:
    validate_route()
    signature = load_json(signature_path)
    stored = signature.pop("run_signature_sha256", None)
    if stored != canonical_sha(signature):
        raise RuntimeError("FoundationStereo recovery signature digest mismatch")
    signature["run_signature_sha256"] = stored
    validate_claim(claim_path, stored, executor_epoch, require_descendant=True)
    load_json(OUTPUT / "RECOVERY_SCOPE.json")
    row = next(row for row in base.w0_rows() if row["session_id"] == RECOVERY_SESSION)
    # Reuse the exact implementation and gates but redirect its module-level
    # per-session publication root into this isolated recovery attempt.
    base.OUTPUT = OUTPUT
    model = frozen.fs_worker.Model()
    recovered = base.process_session(model, row, VISUAL)
    if int(model.model_load_count) != 1 or int(model.inference_count) != 2 * int(row["frame_count"]):
        raise RuntimeError("bounded recovery model-load/inference accounting drift")
    reused = []
    by_id = {str(item["session_id"]): item for item in base.w0_rows()}
    for session_id in REUSED_SESSIONS:
        path = _session_result_path(by_id[session_id], PREDECESSOR)
        reused.append({
            "session_id": session_id, "status": "PASSED",
            "provenance": "REUSED_ATOMIC_PREDECESSOR_OUTPUT_SHA_VERIFIED",
            "result": ref(path),
        })
    all_sessions = [*reused, {**recovered, "provenance": "RECOVERED_ONE_SESSION_FRESH_MODEL_LOAD"}]
    passed = sum(item["status"] == "PASSED" for item in all_sessions)
    rejected = sum(item["status"] == "REJECTED_QUALITY" for item in all_sessions)
    atomic_json(OUTPUT / "BATCH_RESULT.json", {
        "schema_version": "0915-robot15h-foundationstereo-wave0-recovery-batch-v1",
        "window_run_id": WINDOW_RUN_ID, "status": "COMPLETED_ALL_TERMINAL",
        "counts": {"total": 4, "reused_atomic": 3, "freshly_rerun": 1,
                   "passed": passed, "rejected_quality": rejected, "failed_runtime": 0},
        "recovery_model_load_count": int(model.model_load_count),
        "recovery_model_inference_count": int(model.inference_count),
        "expected_recovery_inference_count": 2 * int(row["frame_count"]),
        "sessions": all_sessions, "external_accuracy": "UNVERIFIED",
        "strict_metric_contact_authorized": False,
        "incomplete_predecessor_staging_consumed": False,
        "source_mutated": False,
        "scope": ref(OUTPUT / "RECOVERY_SCOPE.json"),
    })
    return 0


def write_terminal(packet: Mapping[str, Any], status: str, blocker: str | None) -> None:
    batch = load_json(OUTPUT / "BATCH_RESULT.json") if (OUTPUT / "BATCH_RESULT.json").is_file() else None
    result = {
        "schema_version": "0915-robot15h-foundationstereo-wave0-recovery-result-v1",
        "task_id": TASK_ID, "window_run_id": WINDOW_RUN_ID, "status": status,
        "first_blocker": blocker, "weights": packet["weights"],
        "counts": batch.get("counts") if batch else None,
        "recovery_session": RECOVERY_SESSION, "reused_atomic_sessions": list(REUSED_SESSIONS),
        "same_model_and_quality_gates": True,
        "horizontal_reflection_for_disparity_sign": True,
        "output_domain": "ORIGINAL_PHYSICAL_LEFT_640x480",
        "lens_undistortion_applied": False, "external_accuracy": "UNVERIFIED",
        "strict_metric_contact_authorized": False,
        "batch_result": ref(OUTPUT / "BATCH_RESULT.json") if batch else None,
        "recovery_scope": ref(OUTPUT / "RECOVERY_SCOPE.json"),
        "gpu_command_receipt": ref(OUTPUT / "GPU_COMMAND_RECEIPT.json") if (OUTPUT / "GPU_COMMAND_RECEIPT.json").is_file() else None,
        "source_mutated": False, "training_eligible": False,
        "physical_deployment_authorized": False, "claim_limit": packet["claim_limit"],
    }
    atomic_json(OUTPUT / "RESULT.json", result)
    atomic_json(TERMINAL_RECEIPT, {**result, "result": ref(OUTPUT / "RESULT.json")})
    atomic_json(OUTPUT / "RUN_RECEIPT.json", {
        "schema_version": "0915-robot15h-foundationstereo-wave0-recovery-run-receipt-v1",
        "task_id": TASK_ID, "window_run_id": WINDOW_RUN_ID, "status": status,
        "result": ref(OUTPUT / "RESULT.json"), "terminal_receipt": ref(TERMINAL_RECEIPT),
    })


def run_orchestrator(args: argparse.Namespace) -> int:
    packet, packet_path = validate_route()
    if args.output_root.resolve() != OUTPUT.resolve() or args.visual_root.resolve() != VISUAL.resolve() or args.receipt.resolve() != TERMINAL_RECEIPT.resolve():
        raise RuntimeError("FoundationStereo recovery namespace drift")
    if args.executor_epoch < 1 or len(args.fencing_token) < 16:
        raise RuntimeError("positive epoch and fencing token >=16 characters required")
    for path in (OUTPUT, VISUAL, TERMINAL_RECEIPT):
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"fresh recovery output required: {path}")
    scope = recovery_scope()
    signature = build_signature(packet_path, args.executor_epoch, scope)
    OUTPUT.mkdir(parents=True)
    atomic_json_new(OUTPUT / "RECOVERY_SCOPE.json", scope)
    atomic_json_new(OUTPUT / "RUN_SIGNATURE.json", signature)
    claim = {
        "schema_version": "0915-robot15h-foundationstereo-wave0-recovery-claim-v1",
        "task_id": TASK_ID, "window_run_id": WINDOW_RUN_ID, "status": "CLAIMED",
        "weights": packet["weights"], "claimed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "pid": os.getpid(), "proc_start_ticks": common.process_start_ticks(os.getpid()),
        "executor_epoch": args.executor_epoch,
        "fencing_token_sha256": hashlib.sha256(args.fencing_token.encode()).hexdigest(),
        "run_signature_sha256": signature["run_signature_sha256"],
        "unique_write_root": str(OUTPUT.resolve()),
    }
    atomic_json_new(OUTPUT / "CLAIM.json", claim)
    heartbeat("WAIT_GPU_RESOURCE")
    worker_command = [
        str(GPU_LAUNCHER), str(Path(__file__).resolve()), "--worker",
        "--output-root", str(OUTPUT), "--visual-root", str(VISUAL),
        "--executor-epoch", str(args.executor_epoch), "--claim", str(OUTPUT / "CLAIM.json"),
        "--run-signature", str(OUTPUT / "RUN_SIGNATURE.json"),
    ]
    lease_command = [
        sys.executable, str(GPU_LEASE_WRAPPER), "--task-id", TASK_ID,
        "--attempt-id", OUTPUT.name, "--executor-epoch", str(args.executor_epoch),
        "--priority", "CANARY", "--gpu-id", str(args.gpu_id), "--min-free-mib", str(args.min_free_mib),
        "--wait-seconds", str(args.gpu_wait_seconds), "--wall-seconds", str(args.wall_seconds),
        "--receipt", str(OUTPUT / "GPU_COMMAND_RECEIPT.json"), "--claim-limit", packet["claim_limit"],
        "--", *worker_command,
    ]
    atomic_json(OUTPUT / "COMMAND.json", {"worker_command": worker_command, "lease_command": lease_command})
    with (OUTPUT / "GPU_WRAPPER.log").open("x", encoding="utf-8", buffering=1) as log:
        process = subprocess.Popen(lease_command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
        while process.poll() is None:
            time.sleep(30)
            if process.poll() is not None:
                break
            lease_path = ROOT / "_run/current/GPU_LEASE.json"
            lease = load_json(lease_path) if lease_path.is_file() else {}
            acquired = lease.get("status") == "ACQUIRED" and lease.get("task_id") == TASK_ID
            heartbeat("RUNNING" if acquired else "WAIT_GPU_RESOURCE", args.gpu_id if acquired else None)
    gpu = load_json(OUTPUT / "GPU_COMMAND_RECEIPT.json") if (OUTPUT / "GPU_COMMAND_RECEIPT.json").is_file() else {}
    if process.returncode != 0 or gpu.get("status") != "PASSED":
        status = "BLOCKED_RESOURCE" if gpu.get("status") == "BLOCKED_RESOURCE" else "FAILED_RUNTIME_FINAL"
        write_terminal(packet, status, str(gpu.get("reason") or gpu.get("error") or "FOUNDATIONSTEREO_RECOVERY_RUNTIME_FAILED"))
        return 3 if status == "BLOCKED_RESOURCE" else 2
    write_terminal(packet, "PASSED", None)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--visual-root", required=True, type=Path)
    parser.add_argument("--receipt", type=Path, default=TERMINAL_RECEIPT)
    parser.add_argument("--executor-epoch", required=True, type=int)
    parser.add_argument("--fencing-token")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--min-free-mib", type=int, default=61_440)
    parser.add_argument("--gpu-wait-seconds", type=int, default=1800)
    parser.add_argument("--wall-seconds", type=int, default=7200)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--claim", type=Path)
    parser.add_argument("--run-signature", type=Path)
    args = parser.parse_args()
    if args.worker:
        if args.claim is None or args.run_signature is None:
            raise RuntimeError("worker requires claim and run signature")
        return run_worker(args.claim.resolve(strict=True), args.run_signature.resolve(strict=True), args.executor_epoch)
    if args.fencing_token is None:
        raise RuntimeError("orchestrator requires fencing token")
    return run_orchestrator(args)


if __name__ == "__main__":
    raise SystemExit(main())
