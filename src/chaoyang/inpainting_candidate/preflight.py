"""Read-only G0 audit plus an explicitly bounded, retained local storage probe."""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import subprocess
import time

from chaoyang.governance.common import REPO_ROOT, artifact_ref, atomic_json, load_json, now_iso


def _project(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(REPO_ROOT.resolve()):
        raise ValueError("PROJECT_OUTPUT_REQUIRED")
    return resolved


def _check(ref: dict, *, overrides: dict) -> dict:
    expected = overrides.get(ref["path"], ref)
    started = time.monotonic()
    try:
        actual = artifact_ref(Path(expected["path"]))
        return {"expected": expected, "actual": actual, "status": "PASS" if actual == expected else "SHA_DRIFT", "elapsed_seconds": time.monotonic() - started}
    except OSError as error:
        return {"expected": expected, "status": "MISSING_OR_IO_ERROR", "error": str(error)}


def run_audit(packet: dict, output: Path) -> dict:
    output = _project(output)
    if output.exists() or not output.is_relative_to(Path(packet["output_dir"]).resolve()):
        raise ValueError("FRESH_TASK_OUTPUT_REQUIRED")
    output.mkdir(parents=True)
    manifest_path = REPO_ROOT / "inpainting_v1/MANIFEST.json"
    manifest = load_json(manifest_path)
    supplement = load_json(REPO_ROOT / "inpainting_v1/GOVERNANCE_PUBLICATION_R02.json")
    overrides = {supplement["current_wrapper"]["path"]: supplement["current_wrapper"]}
    groups = {"core": manifest["algorithm_core"], "runtime": manifest["runtime_closure"], "dependencies": manifest["dependencies"], "weights": manifest["models"]["weights"]}
    checks = {}
    for refs in groups.values():
        for ref in refs:
            if ref["path"] not in checks:
                checks[ref["path"]] = _check(ref, overrides=overrides)
    group_counts = {name: {"total": len(refs), "verified": sum(checks[ref["path"]]["status"] == "PASS" for ref in refs)} for name, refs in groups.items()}
    protected = {name: _check(ref, overrides={}) for name, ref in packet["inputs"].items() if isinstance(ref, dict) and {"path", "bytes", "sha256"} <= ref.keys()}
    source = (REPO_ROOT / "src/chaoyang/governance/execute_pick_place_1008_v1.py").read_text()
    defaults = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "GOVERNANCE_CODE" for target in node.targets):
            defaults = list(ast.literal_eval(node.value))
    signature_gaps = [str(path) for path in (Path("src/chaoyang/governance/inpainting_v1_selection.py"), Path("src/chaoyang/governance/operation_retirement.py")) if str(path) not in defaults]
    missing_input_requirement = '"inputs"' not in source.split("SPEC_FIELDS =", 1)[1].split("}", 1)[0]
    vfs = os.statvfs(output)
    storage = {"mount_free_bytes": vfs.f_bfree * vfs.f_frsize, "mount_available_bytes": vfs.f_bavail * vfs.f_frsize, "block_size": vfs.f_frsize, "probe_bytes": 1048576, "probe_retained": True, "large_output_admission": "BLOCKED_STORAGE_CAPACITY_UNPROVEN"}
    probe = output / "STORAGE_PROBE_1M.bin"
    try:
        started = time.monotonic()
        with probe.open("xb") as handle:
            handle.write(bytes(1048576))
            handle.flush()
            os.fsync(handle.fileno())
        storage.update(probe_status="WRITE_AND_FSYNC_PASS", probe=artifact_ref(probe), probe_seconds=time.monotonic() - started)
    except OSError as error:
        storage.update(probe_status="FAILED", error=str(error))
    git = subprocess.run(["git", "status", "--porcelain=v1", "-z"], cwd=REPO_ROOT, capture_output=True, check=True).stdout.split(b"\0")
    dirty = [item.decode(errors="replace") for item in git if item]
    raw_manifest = load_json(Path(manifest["inputs"]["raw_manifest"]["path"]))
    dev = {"001", "003", "008", "100", "114", "050"}
    challenges = {"039", "007", "199"}
    selected = []
    for session in raw_manifest["sessions"]:
        if session["session_id"] in dev | challenges:
            selected.append({"session_id": session["session_id"], "role": "DEVELOPMENT" if session["session_id"] in dev else "KNOWN_CHALLENGE_NOT_BLIND", "raw_frames": session["raw_frame_count"], "decoded_frames": session["decoded_frame_count"], "input_status": session["input_status"], "source_refs": session["source_refs"], "time_stats": session["time_stats"], "camera_snapshot_sha256": session["camera_snapshot_sha256"], "device_serial": session["device_serial"]})
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True)
    errors = [value for value in checks.values() if value["status"] != "PASS"] + [value for value in protected.values() if value["status"] != "PASS"]
    result = {"schema": "inpainting-g0-preflight-v1", "task_id": packet["task_id"], "run_signature": packet["run_signature"], "created_at": now_iso(), "parent": artifact_ref(manifest_path), "parent_unchanged": not errors, "groups": group_counts, "checks": list(checks.values()), "protected": protected, "current_executor_signature_gaps": signature_gaps, "old_spec_inputs_optional": missing_input_requirement, "fix_strategy": "NEW_CANDIDATE_ONLY_STRICT_INPUT_AND_STAGE_SIGNATURES_NOT_MUTATION_OF_FROZEN_V1", "storage": storage, "gpu_query": {"exit_code": gpu.returncode, "stdout": gpu.stdout.strip(), "stderr": gpu.stderr.strip()}, "git": {"dirty_entries": len(dirty), "untracked_entries": sum(row.startswith("?? ") for row in dirty), "commit_status": "NOT_COMMITTED_DIRTY_TREE_PRESERVED", "all_dirty_entries": dirty}, "fixed_samples": selected, "errors": errors, "algorithm_calls": 0, "training_steps": 0, "external_writes": 0, "files_deleted": 0, "visual_quality": "NOT_EVALUATED", "large_run_ready": False}
    atomic_json(output / "PREFLIGHT.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packet = load_json(_project(args.packet))
    if packet.get("task_id") != "inpainting_generalization_g0_20261012" or packet.get("algorithm_execution_allowed") is not False:
        raise ValueError("G0_CPU_PACKET_REQUIRED")
    result = run_audit(packet, args.output)
    print(json.dumps({"task_id": result["task_id"], "groups": result["groups"], "errors": len(result["errors"]), "storage": result["storage"], "large_run_ready": False}))
    return int(bool(result["errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
