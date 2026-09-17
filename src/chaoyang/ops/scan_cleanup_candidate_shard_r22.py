#!/usr/bin/env python3
"""Resume a bounded, read-only apparent-byte scan for one cleanup candidate.

This helper never follows symlinks and refuses every path outside the two
repository-local candidate roots.  A completed scan is still not deletion
authorization; it only resolves the recursive-byte evidence debt.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
ALLOWED = (ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs", ROOT / "_run")
FORBIDDEN = (Path("/mnt/data/egodata"), Path("/nas"))


def contained(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def root_identity(path: Path) -> dict[str, int]:
    value = path.lstat()
    return {
        "device": value.st_dev,
        "inode": value.st_ino,
        "mtime_ns": value.st_mtime_ns,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-delete-manifest", type=Path, required=True)
    parser.add_argument("--candidate-index", type=int, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--max-entries", type=int, default=20_000)
    parser.add_argument("--max-wall-seconds", type=float, default=120.0)
    args = parser.parse_args()
    if args.max_entries <= 0 or not 1 <= args.max_wall_seconds <= 600:
        raise SystemExit("bounded limits are invalid")

    manifest = json.loads(args.pre_delete_manifest.read_text(encoding="utf-8"))
    candidates = manifest["candidates"]
    if not 0 <= args.candidate_index < len(candidates):
        raise SystemExit("candidate index out of range")
    candidate = Path(candidates[args.candidate_index]["path"])
    # resolve(strict=True) is intentional: a changed/missing candidate cannot
    # silently inherit a prior scan state.
    root = candidate.resolve(strict=True)
    if any(contained(root, forbidden.resolve()) for forbidden in FORBIDDEN):
        raise SystemExit("external protected path refused")
    if not any(contained(root, allowed.resolve()) and root != allowed.resolve() for allowed in ALLOWED):
        raise SystemExit("candidate is outside a repository-local cleanup root")

    state_path = args.state.resolve()
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["candidate_path"] != str(root) or state["candidate_index"] != args.candidate_index:
            raise SystemExit("resume state belongs to another candidate")
        if state["root_identity_at_start"] != root_identity(root):
            raise SystemExit("candidate root identity changed; start a new scan receipt")
    else:
        first = root.lstat()
        state = {
            "schema_version": "cleanup-candidate-sharded-byte-scan-r22-v1",
            "candidate_index": args.candidate_index,
            "candidate_path": str(root),
            "root_identity_at_start": root_identity(root),
            "pending_directories": [str(root)] if stat.S_ISDIR(first.st_mode) else [],
            "apparent_bytes": first.st_size,
            "files": 0 if stat.S_ISDIR(first.st_mode) else 1,
            "directories": 1 if stat.S_ISDIR(first.st_mode) else 0,
            "symlinks": 1 if stat.S_ISLNK(first.st_mode) else 0,
            "entries_scanned": 1,
            "status": "RUNNING_BOUNDED_SCAN",
            "delete_authorized": False,
        }

    started = time.monotonic()
    this_entries = 0
    while state["pending_directories"]:
        if this_entries >= args.max_entries or time.monotonic() - started >= args.max_wall_seconds:
            break
        directory = Path(state["pending_directories"].pop())
        try:
            entries = list(os.scandir(directory))
        except OSError as error:
            state["status"] = "BLOCKED_REFERENCE_PROOF"
            state["scan_error"] = f"{type(error).__name__}: {error}"
            atomic_json(state_path, state)
            return 3
        for entry in entries:
            value = entry.stat(follow_symlinks=False)
            state["apparent_bytes"] += value.st_size
            state["entries_scanned"] += 1
            this_entries += 1
            if stat.S_ISLNK(value.st_mode):
                state["symlinks"] += 1
                state["files"] += 1
            elif stat.S_ISDIR(value.st_mode):
                state["directories"] += 1
                state["pending_directories"].append(entry.path)
            else:
                state["files"] += 1
            # Finish the currently opened directory so the resumable frontier
            # never loses unvisited siblings.  Budgets are enforced before the
            # next directory; a single exceptionally wide directory may make
            # this shard exceed the soft entry limit, but cannot corrupt state.

    state["last_shard_entries"] = this_entries
    state["last_shard_wall_seconds"] = time.monotonic() - started
    if state["pending_directories"]:
        state["status"] = "RUNNING_BOUNDED_SCAN"
    elif root_identity(root) != state["root_identity_at_start"]:
        state["status"] = "BLOCKED_REFERENCE_PROOF"
        state["scan_error"] = "candidate root metadata changed during scan"
    else:
        state["status"] = "BYTE_SCAN_COMPLETE_NOT_DELETE_AUTHORITY"
    state["delete_authorized"] = False
    atomic_json(state_path, state)
    print(json.dumps({
        "status": state["status"],
        "candidate_index": args.candidate_index,
        "entries_scanned": state["entries_scanned"],
        "pending_directories": len(state["pending_directories"]),
        "apparent_bytes": state["apparent_bytes"],
    }, ensure_ascii=False))
    return 0 if state["status"] == "BYTE_SCAN_COMPLETE_NOT_DELETE_AUTHORITY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
