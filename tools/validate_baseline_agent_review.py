#!/usr/bin/env python3
"""Validate and re-hash a two-task baseline AGENT_REVIEW.json."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import jsonschema


PROJECT = Path(__file__).resolve().parents[1]
SCHEMA = PROJECT / "contracts/baseline_agent_review_v1.schema.json"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return value


def verify_artifacts(review: dict[str, Any]) -> None:
    for group in ("inputs", "artifacts"):
        for name, reference in review[group].items():
            path = Path(reference["path"]).resolve(strict=True)
            if not path.is_file() or path.is_symlink():
                raise RuntimeError(f"{group}.{name}: regular non-symlink file required")
            if path.stat().st_size != reference["bytes"] or digest(path) != reference["sha256"]:
                raise RuntimeError(f"{group}.{name}: bytes/SHA mismatch")


def validate(path: Path) -> dict[str, Any]:
    review = load(path)
    jsonschema.Draft202012Validator(load(SCHEMA)).validate(review)
    verify_artifacts(review)
    hard_failed = any(value == "FAIL" for value in review["hard_gates"].values())
    if hard_failed != (review["grade"] == "C"):
        raise RuntimeError("hard-gate/grade invariant failed")
    return {
        "status": "PASS_BASELINE_AGENT_REVIEW",
        "stage": review["stage"],
        "task": review["task"],
        "session": review["session"],
        "grade": review["grade"],
        "downstream_authorized": review["downstream_authorized"],
        "review_sha256": digest(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("review", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.review), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
