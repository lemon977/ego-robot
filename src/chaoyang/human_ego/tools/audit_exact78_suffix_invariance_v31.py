#!/usr/bin/env python3
"""CPU-only suffix-invariance audit for frozen Exact78 time-t inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from chaoyang.human_ego.exact78_v31 import suffix_invariance_report


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-inputs", type=Path, required=True)
    parser.add_argument("--prefix-inputs", type=Path, required=True)
    parser.add_argument("--authority-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    full_path = args.full_inputs.resolve(strict=True)
    prefix_path = args.prefix_inputs.resolve(strict=True)
    authority_path = args.authority_json.resolve(strict=True)
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    if not isinstance(authority, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in authority.items()
    ):
        raise RuntimeError("authority JSON must be a string-to-string object")
    report = suffix_invariance_report(
        load_npz(full_path), load_npz(prefix_path), authority
    )
    report.update(
        inputs={
            "full": ref(full_path),
            "prefix": ref(prefix_path),
            "declared_authority": ref(authority_path),
        },
        claim_limit=(
            "This tests time-t input invariance to a removed future suffix; it "
            "does not make OFFLINE_NONCAUSAL labels causal."
        ),
    )
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"fresh/no-clobber output required: {output}")
    atomic_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["causal_current_inputs_authorized"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
