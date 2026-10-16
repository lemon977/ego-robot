"""Reversible cleanup: move ledger-only task directories of _run/current into this task's archive.

Input is a sealed takeover inventory (INVENTORY.json + sha256). Only rows classified
LEDGER_ONLY_CLEANUP_CANDIDATE that are real task directories (attempts/*/RESULT.json)
with no live writer are moved; each is re-checked against the current packet index
right before the move. Moves are recorded in MOVES.json. If governance validation
fails afterwards, every move is undone. ``--restore`` moves a recorded set back.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from chaoyang.paths import get_paths

CANDIDATE = "LEDGER_ONLY_CLEANUP_CANDIDATE"
PYTHON = "/usr/local/bin/python"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _load_verified(path: Path, sha256: str) -> dict:
    if _sha256(path) != sha256:
        raise SystemExit(f"sha256 mismatch for {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _governance_ok(repo: Path) -> tuple[bool, str]:
    proc = subprocess.run([PYTHON, "-B", "-m", "chaoyang.cli", "validate-governance"], cwd=repo,
                          env={**os.environ, "PYTHONPATH": str(repo / "src")},
                          capture_output=True, text=True, timeout=1800, check=False)
    try:
        status = json.loads(proc.stdout).get("status")
    except ValueError:
        status = None
    return status == "PASS", (proc.stdout + proc.stderr)[-2000:]


def _move_back(moves: list[dict]) -> list[dict]:
    restored = []
    for move in reversed(moves):
        source, target = Path(move["to"]), Path(move["from"])
        if source.is_dir() and not target.exists():
            os.rename(source, target)
            restored.append({**move, "restored_at": _now()})
    return restored


def plan(repo: Path, inventory: dict) -> tuple[list[dict], list[dict]]:
    from chaoyang.exact78_baseline import audit_inventory

    run_root = repo / "_run/current"
    index_text = (repo / "tasks/current/INDEX.json").read_text(encoding="utf-8")
    indexed = set(audit_inventory.RUN_REF.findall(index_text))
    chosen, skipped = [], []
    for row in inventory["run_dirs"]:
        if row["category"] != CANDIDATE:
            continue
        name = row["task_dir"]
        directory = run_root / name
        reason = None
        if not row.get("is_task_dir"):
            reason = "NOT_A_TASK_DIR"
        elif name.startswith("exact78_takeover_") or audit_inventory.OTHER_BRANCH.search(name):
            reason = "PROTECTED_NAME"
        elif not directory.is_dir():
            reason = "MISSING"
        elif audit_inventory.live_writers(directory):
            reason = "LIVE_WRITER"
        elif name in indexed:
            reason = "REFERENCED_BY_PACKET_INDEX"
        if reason:
            skipped.append({"task_dir": name, "reason": reason})
        else:
            chosen.append(row)
    return chosen, skipped


def main(argv: list[str] | None = None) -> int:
    from chaoyang.governance.exact78_operation_gate import authorize_operation_entry
    authorize_operation_entry("run_exact78_takeover_cleanup_v1", argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--inventory-sha256")
    parser.add_argument("--restore", type=Path, help="MOVES.json of an earlier cleanup task")
    parser.add_argument("--restore-sha256")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    repo = get_paths().repo_root.resolve()
    output = args.output_root.resolve()
    if repo not in output.parents or output.exists():
        raise SystemExit("output root must be a fresh path inside the project")
    output.mkdir(parents=True)
    started = time.monotonic()
    if args.restore:
        recorded = _load_verified((repo / args.restore).resolve(), args.restore_sha256)
        restored = _move_back(recorded["moves"])
        result = {"mode": "RESTORE", "restored": restored,
                  "not_restored": len(recorded["moves"]) - len(restored)}
    else:
        inventory = _load_verified((repo / args.inventory).resolve(), args.inventory_sha256)
        chosen, skipped = plan(repo, inventory)
        archive = output / "archive"
        archive.mkdir()
        moves = []
        for row in chosen:
            source = repo / "_run/current" / row["task_dir"]
            target = archive / row["task_dir"]
            os.rename(source, target)
            moves.append({"task_dir": row["task_dir"], "from": str(source), "to": str(target),
                          "size": row.get("size"), "latest_status": row.get("latest_status"),
                          "moved_at": _now()})
        ok, tail = _governance_ok(repo)
        restored = [] if ok else _move_back(moves)
        result = {"mode": "MOVE", "moves": moves, "skipped": skipped, "governance_pass_after_move": ok,
                  "governance_tail": tail if not ok else None, "rolled_back": restored,
                  "bytes_moved": sum((move["size"] or {}).get("bytes", 0) for move in moves) if ok else 0,
                  "inventory": {"path": str(args.inventory), "sha256": args.inventory_sha256}}
        (output / "MOVES.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                                           encoding="utf-8")
        if not ok:
            result["status"] = "ROLLED_BACK_GOVERNANCE_FAILED"
    result.setdefault("status", "COMPLETE")
    result["elapsed_seconds"] = round(time.monotonic() - started, 2)
    result["schema_version"] = "exact78-takeover-cleanup-v1"
    (output / "RESULT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                                        encoding="utf-8")
    summary = {key: (len(value) if isinstance(value, list) else value) for key, value in result.items()
               if key not in ("governance_tail",)}
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if result["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
