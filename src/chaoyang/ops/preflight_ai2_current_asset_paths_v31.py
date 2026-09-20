#!/usr/bin/env python3
"""Read-only preflight for the fixed AI2 W0/A1/A2 current asset bindings."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from chaoyang.ops.audit_ai2_real_assets_v31 import ROOT, artifact_ref
from chaoyang.ops.run_ai2_real_assets_cohort_audit_v31 import (
    ASSET_TASK_DIRECTORIES,
    SESSION_SPECS,
    _paths,
)


SCHEMA_VERSION = "AI2_CURRENT_ASSET_PATH_PREFLIGHT_V31"


def _canonical_sha(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_preflight(*, root: Path = ROOT) -> dict[str, Any]:
    """Bind all expected current files without creating or changing any artifact."""

    root = root.resolve(strict=True)
    sessions: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    for spec in SESSION_SPECS:
        logical_task = spec["task"]
        asset_directory = ASSET_TASK_DIRECTORIES[logical_task]
        files: dict[str, Any] = {}
        for role, path in _paths(root, spec).items():
            if path is None:
                files[role] = {"status": "ABSENT_NOT_REQUIRED"}
                continue
            resolved = path.resolve(strict=False)
            if "archive" in resolved.parts or not str(resolved).startswith(
                str((root / "_run/current").resolve()) + "/"
            ):
                missing.append(
                    {
                        "session_id": spec["session_id"],
                        "role": role,
                        "reason": "OUTSIDE_CURRENT_SCOPE",
                    }
                )
                files[role] = {
                    "status": "REJECTED_SCOPE",
                    "path": str(resolved),
                }
            elif not resolved.is_file():
                missing.append(
                    {
                        "session_id": spec["session_id"],
                        "role": role,
                        "reason": "MISSING_FILE",
                    }
                )
                files[role] = {"status": "MISSING", "path": str(resolved)}
            else:
                files[role] = {"status": "SHA_BOUND", **artifact_ref(resolved)}
        sessions.append(
            {
                **spec,
                "asset_directory": asset_directory,
                "logical_task_preserved": True,
                "files": files,
            }
        )

    status = (
        "PASS_8_CURRENT_ASSET_BINDINGS"
        if not missing and len(sessions) == 8
        else "BLOCKED_CURRENT_ASSET_BINDINGS"
    )
    mapping = dict(sorted(ASSET_TASK_DIRECTORIES.items()))
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "task_to_asset_directory": mapping,
        "task_to_asset_directory_sha256": _canonical_sha(mapping),
        "sessions": sessions,
        "counts": {
            "total": len(sessions),
            "path_complete": len(sessions)
            - len({row["session_id"] for row in missing}),
            "missing_or_rejected_files": len(missing),
        },
        "missing": missing,
        "current_only": True,
        "archive_consumed": False,
        "read_only": True,
        "model_calls": 0,
        "gpu_calls": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    result = build_preflight(root=args.root)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "PASS_8_CURRENT_ASSET_BINDINGS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
