"""Read-only, source-neutral Exact78 candidate submission planning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chaoyang.cli_v7.source_neutral_plan_v1 import plan_manifest, plan_one
from chaoyang.governance.exact78_cold_start_guard_v1 import ColdStartInputError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m chaoyang.cli_v7")
    parser.add_argument("command", choices=("submit",))
    parser.add_argument("recipe", choices=("exact78-raw-to-robot",))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--session", type=Path)
    source.add_argument("--manifest", type=Path)
    parser.add_argument("--expected-input-sha256")
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--check-only", action="store_true", required=True)
    args = parser.parse_args(argv)
    try:
        if args.manifest is not None:
            if args.expected_input_sha256 is not None:
                raise ColdStartInputError("MANIFEST_AND_SESSION_SHA_CONFLICT")
            if args.expected_manifest_sha256 is None:
                raise ColdStartInputError("EXPECTED_MANIFEST_SHA_REQUIRED")
            result = plan_manifest(args.manifest, expected_sha=args.expected_manifest_sha256)
        else:
            if args.expected_manifest_sha256 is not None:
                raise ColdStartInputError("SESSION_AND_MANIFEST_SHA_CONFLICT")
            result = plan_one(args.session, expected_sha=args.expected_input_sha256)
    except (ColdStartInputError, OSError, ValueError) as exc:
        print(json.dumps({"status": "REJECTED", "reason": str(exc),
                          "registration_started": False, "model_calls": 0}, sort_keys=True))
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


__all__ = ["main"]
