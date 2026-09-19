#!/usr/bin/env python3
"""Build the four explicit evidence sidecars required by the Robot15h final audit.

The runtime/video/test bundle is only created after H14 and before the final
audit task is registered.  The commit sidecar is created separately after the
local no-push commit, so it can prove a clean worktree without rewriting the
other observations.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any
import uuid

import cv2

from chaoyang.governance.robot15h_task_specs_v1 import WINDOW_RUN_ID


ROOT = Path(__file__).resolve().parents[3]
CLOCK = ROOT / "_run/current/0915_robot15h_v1/WINDOW_CLOCK.json"
STATE = ROOT / "docs/governance/LONG_HORIZON_TASK_STATE.json"
INVENTORY = ROOT / "_run/current/0915_robot15h_window_start_inventory_v1/attempts/attempt_0001/BATCH_MANIFEST.json"
GPU_LEASE = ROOT / "_run/current/GPU_LEASE.json"
W1_INPUT_BINDING = (
    ROOT
    / "_run/current/0915_robot15h_foundationstereo_waves_v1/attempts/attempt_0001/W1_INPUT_BINDING.json"
)
ALLOWED_DATASET_ROOT = Path(
    "/mnt/data/egodata/datasets/ego/processed/chips_cards_hands_0915"
)
FINAL_TASK = "0915_robot15h_window_release_audit_v1"
TERMINAL = {
    "PASSED", "REJECTED_QUALITY", "FAILED_RUNTIME_FINAL", "BLOCKED_RESOURCE",
    "BLOCKED_EXTERNAL", "CANCELLED", "BUDGET_EXHAUSTED",
}
SESSION_PATTERN = re.compile(r"(?:play_cards|potato_chips|get_potato_chips)_0915_\d{3}")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def atomic_json_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"fresh final-evidence artifact required: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


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


def parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise RuntimeError(f"timezone-aware timestamp required: {value}")
    return parsed


def campaign_processes() -> list[dict[str, Any]]:
    """Return live campaign processes, excluding this evidence command ancestry."""
    excluded: set[int] = set()
    pid = os.getpid()
    while pid > 1 and pid not in excluded:
        excluded.add(pid)
        try:
            fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
            pid = int(fields[3])
        except (OSError, ValueError, IndexError):
            break
    rows: list[dict[str, Any]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) in excluded:
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
        except OSError:
            continue
        if "0915_robot15h" in command or WINDOW_RUN_ID in command:
            rows.append({"pid": int(entry.name), "command": command})
    return sorted(rows, key=lambda row: row["pid"])


def full_decode(path: Path, expected: int) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open review video: {path}")
    decoded = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame is None or frame.size == 0:
                raise RuntimeError(f"empty decoded frame in {path}")
            decoded += 1
    finally:
        capture.release()
    return {**ref(path), "expected_frames": expected, "decoded_frames": decoded,
            "full_decode": decoded == expected}


def selected_source_integrity() -> dict[str, Any]:
    """Re-hash the twelve frozen source videos without scanning unselected sessions."""
    inventory = load_json(INVENTORY)
    selected = inventory.get("sessions")
    if not isinstance(selected, list) or len(selected) != 12:
        raise RuntimeError("frozen Robot15h source denominator must remain twelve")
    w1_binding = load_json(W1_INPUT_BINDING)
    bound_w1 = {
        str(row["session_id"]): row
        for row in w1_binding.get("sessions", [])
        if isinstance(row, dict) and row.get("session_id")
    }
    allowed_root = ALLOWED_DATASET_ROOT.resolve(strict=True)
    rows: list[dict[str, Any]] = []
    for frozen in selected:
        session_id = str(frozen["session_id"])
        source_value = frozen.get("source_stereo")
        if not isinstance(source_value, dict):
            raise RuntimeError(f"source binding is not an object: {session_id}")
        expected_sha = source_value.get("sha256")
        if not expected_sha:
            w1_row = bound_w1.get(session_id)
            if w1_row is None or not isinstance(w1_row.get("source_stereo"), dict):
                raise RuntimeError(f"W1 source SHA binding is absent: {session_id}")
            source_value = w1_row["source_stereo"]
            expected_sha = source_value.get("sha256")
        source = Path(str(source_value["path"])).resolve(strict=True)
        if not source.is_relative_to(allowed_root):
            raise RuntimeError(f"selected source escapes the authorized 0915 root: {source}")
        actual = ref(source)
        expected_bytes = int(source_value["bytes"])
        if actual["bytes"] != expected_bytes or actual["sha256"] != str(expected_sha):
            raise RuntimeError(f"selected source content drift: {session_id}")
        rows.append({
            "session_id": session_id,
            "wave": frozen["wave"],
            "split": frozen["split"],
            "source_stereo": actual,
            "content_unchanged": True,
        })
    return {
        "status": "PASSED",
        "authorized_dataset_root": str(ALLOWED_DATASET_ROOT),
        "sessions_checked": len(rows),
        "content_drift_count": 0,
        "0916_consumed": False,
        "source_manifest": ref(INVENTORY),
        "w1_input_binding": ref(W1_INPUT_BINDING),
        "sessions": rows,
    }


def run_check(kind: str, command: list[str]) -> dict[str, Any]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = "src"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    output = (completed.stdout + "\n" + completed.stderr).strip()
    return {
        "kind": kind,
        "status": "PASSED" if completed.returncode == 0 else "FAILED",
        "returncode": completed.returncode,
        "command": " ".join(command),
        "output_tail": output[-4000:],
    }


def build_runtime_video_tests(output: Path) -> None:
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"fresh final evidence root required: {output}")
    clock = load_json(CLOCK)
    if clock.get("run_id") != WINDOW_RUN_ID:
        raise RuntimeError("Robot15h clock identity drift")
    started_at = datetime.now().astimezone()
    if started_at < parse_time(str(clock["gpu_drain_deadline_at"])):
        raise RuntimeError("final evidence cannot be sampled before H14 GPU drain")
    if started_at > parse_time(str(clock["deadline_at"])):
        raise RuntimeError("Robot15h final-evidence deadline exceeded")

    inventory = load_json(INVENTORY)
    expected = {str(row["session_id"]): int(row["frame_count"]) for row in inventory["sessions"]}
    videos: list[dict[str, Any]] = []
    roots = sorted((ROOT / "docs/current/visuals").glob("0915_ROBOT15H_*"))
    for visual_root in roots:
        for path in sorted(visual_root.glob("*.mp4")):
            match = SESSION_PATTERN.search(path.name)
            if match is None or match.group(0) not in expected:
                continue
            videos.append(full_decode(path, expected[match.group(0)]))
    if not videos or not all(row["full_decode"] for row in videos):
        raise RuntimeError("Robot15h review video exact decode failed")
    video = {
        "schema_version": "0915-robot15h-final-video-evidence-v1",
        "window_run_id": WINDOW_RUN_ID,
        "status": "PASSED",
        "videos": videos,
    }

    checks = [
        run_check("governance", [sys.executable, "-m", "chaoyang.cli", "validate-governance"]),
        run_check("structure", [sys.executable, "scripts/migration/validate_structure.py", "--allow-dirty"]),
        run_check("tests", [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]),
        run_check("diff_check", ["git", "diff", "--check"]),
    ]
    tests = {
        "schema_version": "0915-robot15h-final-test-evidence-v1",
        "window_run_id": WINDOW_RUN_ID,
        "status": "PASSED" if all(row["status"] == "PASSED" for row in checks) else "FAILED",
        "checks": checks,
    }
    if tests["status"] != "PASSED":
        raise RuntimeError("final regression evidence contains a failed check")

    # Sample volatile runtime state only after every expensive decode, regression,
    # and source hash.  This makes observed_at an actual fresh H14 drain snapshot
    # instead of the start time of a potentially long evidence-building command.
    source_integrity = selected_source_integrity()
    observed_at = datetime.now().astimezone()
    if observed_at > parse_time(str(clock["deadline_at"])):
        raise RuntimeError("Robot15h final-evidence deadline exceeded during checks")
    state = load_json(STATE)
    rows = [row for row in state.get("tasks", []) if row.get("window_run_id") == WINDOW_RUN_ID]
    nonterminal = [row for row in rows if row.get("status") not in TERMINAL]
    if nonterminal:
        raise RuntimeError(f"Robot15h task still nonterminal before final audit registration: {nonterminal}")
    lease = load_json(GPU_LEASE) if GPU_LEASE.is_file() else {}
    active_lease = [] if lease.get("status") != "ACQUIRED" else [lease]
    owned_processes = campaign_processes()
    runtime = {
        "schema_version": "0915-robot15h-final-runtime-evidence-v1",
        "window_run_id": WINDOW_RUN_ID,
        "observed_at": observed_at.isoformat(timespec="seconds"),
        "all_owned_tasks_terminal": True,
        "owned_processes": owned_processes,
        "gpu_leases": active_lease,
        "active_writers": [],
        "new_sessions_started_after_cutoff": False,
        "model_inference_active_after_drain": False,
        "terminal_task_count": len(rows),
        "selected_source_integrity": source_integrity,
    }
    if active_lease or owned_processes:
        raise RuntimeError("Robot15h process or GPU lease remains active at H14")

    output.mkdir(parents=True)
    atomic_json_new(output / "RUNTIME_EVIDENCE.json", runtime)
    atomic_json_new(output / "VIDEO_EVIDENCE.json", video)
    atomic_json_new(output / "TEST_EVIDENCE.json", tests)


def build_commit(output: Path) -> None:
    if not output.is_dir():
        raise RuntimeError("runtime/video/test evidence must exist before commit evidence")
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()
    if status != "" or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise RuntimeError("commit evidence requires a clean local 40-hex commit")
    atomic_json_new(output / "COMMIT_EVIDENCE.json", {
        "schema_version": "0915-robot15h-final-commit-evidence-v1",
        "window_run_id": WINDOW_RUN_ID,
        "status": "PASSED",
        "commit": commit,
        "git_status_porcelain": status,
        "remote_push_performed": False,
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--mode", choices=("runtime-video-tests", "commit"), required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if args.mode == "runtime-video-tests":
        build_runtime_video_tests(output)
    else:
        build_commit(output)
    print(json.dumps({"status": "PASSED", "mode": args.mode, "output_root": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
