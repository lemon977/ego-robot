from __future__ import annotations

"""exact78 V5.2 task-packet, signature and executor-fencing primitives.

The module is intentionally dependency-light so every lane can import the same
rules.  It never scans the repository.  Inputs must be enumerated explicitly.
"""

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


SESSION_STATUSES = {
    "PENDING", "WAIT_GPU_RESOURCE", "RUNNING", "PASSED",
    "FAILED_QUALITY_C", "FAILED_RUNTIME_FINAL", "BLOCKED_PREREQ",
    "BLOCKED_RESOURCE", "BLOCKED_EXTERNAL", "CANCELLED",
}
ATTEMPT_STATUSES = {
    "CLAIMED", "RUNNING", "PASSED", "FAILED_RUNTIME",
    "FAILED_QUALITY", "CANCELLED",
}
SIGNATURE_COMPONENTS = (
    "input_manifest", "code_closure", "config", "weights",
    "calibration_or_absent", "schema",
)


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_ref(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def atomic_write_new(path: Path, payload: bytes) -> None:
    """Create a final exactly once; a different overwrite is always an error."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError:
        if path.read_bytes() == payload:
            return
        raise RuntimeError(f"immutable final already exists with different bytes: {path}")
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def atomic_replace(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def validate_artifact(reference: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    path = Path(str(reference.get("path", "")))
    if not path.is_absolute() or not path.is_file():
        return [f"artifact missing or non-absolute: {path}"]
    if reference.get("bytes") != path.stat().st_size:
        errors.append(f"artifact byte mismatch: {path}")
    if reference.get("sha256") != sha256_file(path):
        errors.append(f"artifact sha mismatch: {path}")
    return errors


def build_run_signature(components: Mapping[str, Any]) -> dict[str, Any]:
    missing = [key for key in SIGNATURE_COMPONENTS if key not in components]
    extra = sorted(set(components) - set(SIGNATURE_COMPONENTS))
    if missing or extra:
        raise ValueError(f"signature component mismatch missing={missing} extra={extra}")
    normalized: dict[str, Any] = {}
    for key in SIGNATURE_COMPONENTS:
        value = components[key]
        if value == "ABSENT":
            normalized[key] = {"status": "ABSENT"}
        elif isinstance(value, dict):
            errors = validate_artifact(value)
            if errors:
                raise ValueError("; ".join(errors))
            normalized[key] = dict(value)
        else:
            raise ValueError(f"{key} must be an artifact ref or ABSENT")
    return {
        "schema_version": "exact78-run-signature-v1",
        "components": normalized,
        "run_signature_sha256": sha256_bytes(canonical_bytes(normalized)),
    }


def validate_task_packet(packet: Mapping[str, Any]) -> list[str]:
    required = {
        "schema_version", "task_id", "objective", "read_set", "write_set",
        "prerequisites", "gates", "budgets", "attempt_max", "stop_condition",
        "output_contract", "claim_limit", "executor_epoch", "fencing",
    }
    errors: list[str] = []
    if packet.get("schema_version") != "exact78-task-packet-v1":
        errors.append("schema_version must be exact78-task-packet-v1")
    missing = sorted(required - set(packet))
    if missing:
        errors.append(f"missing fields: {missing}")
    if len(packet.get("read_set", [])) > 8:
        errors.append("initial read_set exceeds 8 files")
    if int(packet.get("initial_search_result_limit", 20)) > 20:
        errors.append("initial search result limit exceeds 20")
    if int(packet.get("initial_log_line_limit", 80)) > 80:
        errors.append("initial log line limit exceeds 80")
    if not 1 <= int(packet.get("attempt_max", 0)) <= 3:
        errors.append("attempt_max must be in [1,3]")
    budgets = packet.get("budgets", {})
    for key in ("cpu_seconds", "gpu_seconds", "wall_seconds"):
        if key not in budgets or not isinstance(budgets[key], (int, float)) or budgets[key] < 0:
            errors.append(f"invalid budget: {key}")
    if int(packet.get("executor_epoch", 0)) < 1:
        errors.append("executor_epoch must be >=1")
    fencing = packet.get("fencing", {})
    if fencing.get("pid_startticks_required") is not True:
        errors.append("fencing.pid_startticks_required must be true")
    if fencing.get("immutable_final") is not True:
        errors.append("fencing.immutable_final must be true")
    return errors


def process_identity(pid: int) -> dict[str, Any]:
    stat = Path(f"/proc/{pid}/stat")
    if not stat.is_file():
        return {"pid": pid, "alive": False, "start_ticks": None}
    return {"pid": pid, "alive": True, "start_ticks": int(stat.read_text().split()[21])}


def claim_executor(packet_path: Path, executor_id: str, pid: int, run_signature: str) -> dict[str, Any]:
    packet = load_object(packet_path)
    errors = validate_task_packet(packet)
    if errors:
        raise RuntimeError("invalid task packet: " + "; ".join(errors))
    identity = process_identity(pid)
    if not identity["alive"]:
        raise RuntimeError(f"executor pid is not alive: {pid}")
    claim = {
        "schema_version": "exact78-executor-claim-v1",
        "task_id": packet["task_id"],
        "executor_id": executor_id,
        "executor_epoch": packet["executor_epoch"],
        "pid": pid,
        "proc_start_ticks": identity["start_ticks"],
        "run_signature_sha256": run_signature,
        "claimed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "CLAIMED",
    }
    claim["fencing_token"] = sha256_bytes(canonical_bytes(claim))
    claim_path = packet_path.parent / "EXECUTOR_CLAIM.json"
    atomic_write_new(claim_path, json.dumps(claim, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n")
    return claim


def commit_final(candidate_path: Path, final_path: Path, claim_path: Path) -> dict[str, Any]:
    claim = load_object(claim_path)
    identity = process_identity(int(claim["pid"]))
    if not identity["alive"] or identity["start_ticks"] != claim["proc_start_ticks"]:
        raise RuntimeError("executor fencing failed: pid/startticks mismatch")
    candidate = load_object(candidate_path)
    if candidate.get("fencing_token") != claim.get("fencing_token"):
        raise RuntimeError("candidate fencing token does not match claim")
    candidate["commit_state"] = "COMMITTED"
    candidate["committed_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    payload = json.dumps(candidate, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"
    atomic_write_new(final_path, payload)
    return artifact_ref(final_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate-packet")
    validate.add_argument("packet", type=Path)
    signature = sub.add_parser("signature")
    signature.add_argument("manifest", type=Path, help="JSON object with the six signature components")
    signature.add_argument("--output", type=Path)
    claim = sub.add_parser("claim")
    claim.add_argument("packet", type=Path)
    claim.add_argument("--executor-id", required=True)
    claim.add_argument("--pid", type=int, required=True)
    claim.add_argument("--run-signature", required=True)
    final = sub.add_parser("commit-final")
    final.add_argument("candidate", type=Path)
    final.add_argument("final", type=Path)
    final.add_argument("--claim", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "validate-packet":
        errors = validate_task_packet(load_object(args.packet))
        print(json.dumps({"status": "PASS" if not errors else "FAIL", "errors": errors}, ensure_ascii=False, indent=2))
        return 0 if not errors else 2
    if args.command == "signature":
        result = build_run_signature(load_object(args.manifest))
        payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"
        if args.output:
            atomic_replace(args.output, payload)
        print(payload.decode(), end="")
        return 0
    if args.command == "claim":
        print(json.dumps(claim_executor(args.packet, args.executor_id, args.pid, args.run_signature), ensure_ascii=False, indent=2))
        return 0
    print(json.dumps(commit_final(args.candidate, args.final, args.claim), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
