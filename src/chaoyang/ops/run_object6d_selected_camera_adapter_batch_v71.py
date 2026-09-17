#!/usr/bin/env python3
"""Publish resumable development coordinate adapters for audited Object6D rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ref(path: Path) -> dict[str, object]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def adapter_command(result_path: Path, output_root: Path) -> tuple[list[str], Path]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    spec_path = Path(result["inputs"]["spec"]["path"])
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    object_name = result_path.parent.name
    if not object_name.startswith("physical_object_"):
        object_name = "physical_object_0"
    destination = output_root / result["task"] / result["session_id"] / object_name
    command = [
        sys.executable,
        "src/chaoyang/ops/adapt_object6d_rectified_to_selected_v71.py",
        "--session-id", result["session_id"],
        "--artifact-revision", "R7_3",
        "--object6d", result["artifacts"]["trajectory"]["path"],
        "--registration-authority", spec["registration_authority"]["path"],
        "--camera-to-world", spec["camera_to_world"]["path"],
        "--camera-to-world-key", spec["camera_to_world_key"],
        "--output-dir", str(destination),
    ]
    return command, destination


def atomic_json(path: Path, value: object) -> None:
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--per-item-timeout-seconds", type=int, default=120)
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows, failures = [], []
    for audited in audit["rows"]:
        source = Path(audited["source_result"])
        command, destination = adapter_command(source, args.output_root)
        result_path = destination / "RESULT.json"
        if result_path.is_file():
            value = json.loads(result_path.read_text(encoding="utf-8"))
            if value.get("status") == "PASS_DEVELOPMENT_COORDINATE_ADAPTER":
                rows.append({"session_id": audited["session_id"], "object": destination.name, "status": "ADOPT_EXACT_EXISTING", "result": ref(result_path)})
                continue
        try:
            completed = subprocess.run(command, text=True, capture_output=True, timeout=args.per_item_timeout_seconds, check=False)
            if completed.returncode != 0:
                raise RuntimeError((completed.stdout + completed.stderr)[-2000:])
            rows.append({"session_id": audited["session_id"], "object": destination.name, "status": "PASSED", "result": ref(result_path)})
        except Exception as error:
            failures.append({"session_id": audited["session_id"], "object": destination.name, "error": f"{type(error).__name__}: {error}"})
    payload = {
        "schema_version": "object6d-selected-camera-adapter-batch-result-v1",
        "artifact_revision": "R7_3",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASSED_DEVELOPMENT_BATCH" if not failures else "PARTIAL_EXPLICIT_FAILURES",
        "counts": {"requested": len(audit["rows"]), "passed_or_adopted": len(rows), "failed": len(failures), "sessions": len({row["session_id"] for row in rows})},
        "source_audit": ref(args.audit),
        "rows": rows,
        "failures": failures,
        "claim_limit": "Development coordinate adapters only; no source Object6D, Contact, Robot or physical authority promotion.",
    }
    atomic_json(args.output_root / "RESULT.json", payload)
    print(json.dumps(payload["counts"], ensure_ascii=False))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
