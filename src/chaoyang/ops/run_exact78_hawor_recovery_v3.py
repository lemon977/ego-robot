#!/usr/bin/env python3
"""HaWoR-environment adapter for the frozen V2 recovery implementation.

The HaWoR environment intentionally does not carry governance-only jsonschema.
This adapter supplies only the four file helpers V2 consumes, then delegates
without changing models, gates, thresholds, or recovery behavior.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import types


REPO_ROOT = Path(__file__).resolve().parents[3]


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def artifact_ref(path: Path) -> dict:
    resolved = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": digest.hexdigest()}


def atomic_json(path: Path, value: dict) -> None:
    destination = Path(path)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    os.replace(temporary, destination)


stub = types.ModuleType("chaoyang.governance.common")
stub.REPO_ROOT = REPO_ROOT
stub.artifact_ref = artifact_ref
stub.atomic_json = atomic_json
stub.load_json = load_json
sys.modules.setdefault("chaoyang.governance.common", stub)

from chaoyang.ops.run_exact78_hawor_recovery_v2 import main


if __name__ == "__main__":
    raise SystemExit(main())
