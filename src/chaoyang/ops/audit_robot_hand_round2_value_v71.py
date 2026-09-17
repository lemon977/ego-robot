#!/usr/bin/env python3
"""Compare Robot hand round-1 and round-2 summaries without changing a run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


METRICS = ("failed_rows", "tip_direction_error_deg_max", "bone_error_deg_max")


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def elapsed_seconds(row: dict[str, Any]) -> float:
    start = datetime.fromisoformat(str(row["started_at"]))
    finish = datetime.fromisoformat(str(row["finished_at"]))
    return (finish - start).total_seconds()


def compare(first: dict[str, Any], second: dict[str, Any], tolerance: float = 1e-6) -> dict[str, Any]:
    if first["session"] != second["session"]:
        raise RuntimeError("round session mismatch")
    m1, m2 = first["metrics"], second["metrics"]
    delta = {name: float(m1[name]) - float(m2[name]) for name in METRICS}
    summary_improved = (
        delta["failed_rows"] > 0
        or delta["tip_direction_error_deg_max"] > tolerance
        or delta["bone_error_deg_max"] > tolerance
    )
    return {
        "session": first["session"],
        "task": first.get("task"),
        "round1": {name: m1[name] for name in METRICS},
        "round2": {name: m2[name] for name in METRICS},
        "round1_seconds": elapsed_seconds(first),
        "round2_seconds": elapsed_seconds(second),
        "round1_minus_round2": delta,
        "summary_improved": summary_improved,
    }


def audit(robot_root: Path) -> tuple[dict[str, Any], list[Path]]:
    rows = []
    inputs = []
    for batch in sorted(robot_root.resolve(strict=True).glob("batch_*")):
        first_path = batch / "hand_round1" / "RESULT.json"
        second_path = batch / "hand_round2" / "RESULT.json"
        if not first_path.is_file() or not second_path.is_file():
            continue
        inputs.extend([first_path, second_path])
        first = {row["session"]: row for row in load(first_path).get("sessions", [])}
        second = {row["session"]: row for row in load(second_path).get("sessions", [])}
        if first.keys() != second.keys():
            raise RuntimeError(f"round session set mismatch: {batch}")
        rows.extend(compare(first[session], second[session]) for session in sorted(first))
    round2_seconds = sum(row["round2_seconds"] for row in rows)
    improved = sum(row["summary_improved"] for row in rows)
    return {
        "sessions": rows,
        "counts": {
            "sessions": len(rows),
            "summary_improved": improved,
            "summary_unchanged": len(rows) - improved,
        },
        "round2_wall_seconds_sum": round2_seconds,
        "round2_wall_seconds_mean": round2_seconds / len(rows) if rows else None,
    }, inputs


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"immutable output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report, inputs = audit(args.robot_root)
    payload = {
        "schema_version": "robot-hand-round2-value-audit-v71-v1",
        "status": "PASSED_DEVELOPMENT_ROUND_VALUE_AUDIT" if report["counts"]["sessions"] else "INSUFFICIENT_DEVELOPMENT_EVIDENCE",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "audit": report,
        "recommendation": "Do not hot-change R7.3. Round-2 may only be skipped in a future pinned canary after a per-row reachability/saturation predicate and cross-task regression are frozen.",
        "inputs": [ref(path) for path in inputs],
        "code": ref(Path(__file__)),
        "authority": False,
        "claim_limit": "Summary-metric and recorded wall-duration comparison only; not proof that round-2 is globally redundant, not benchmark-isolated speedup, and not Robot authority.",
    }
    atomic_json(args.output.resolve(), payload)
    print(json.dumps({"status": payload["status"], "counts": report["counts"], "round2_wall_seconds_sum": report["round2_wall_seconds_sum"], "output": str(args.output.resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
