"""Registered CPU metadata preflight only; does not execute official HuRo IK."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from chaoyang.pipeline.huro_core_adapter_v2 import (
    dependency_metadata, verify_upstream_sources, core_execution_status, UPSTREAM_COMMIT,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor-root", type=Path, required=True)
    args = parser.parse_args(argv)
    # No files are written: publisher captures stdout only into the own attempt.
    sources = verify_upstream_sources(args.vendor_root)
    metadata = dependency_metadata()
    result = core_execution_status(metadata)
    result.update({"upstream_commit": UPSTREAM_COMMIT, "sources": sources,
                   "dependency_metadata": metadata,
                   "resume_requirements": ["Project-local approved offline packages and source checkouts",
                                           "Frozen Tianji+KaiHand combined URDF/config and FK parity",
                                           "Registered fixed-placement official core adapter and GPU lease",
                                           "Independent per-point masks and explicit timestamp/gap policy"]})
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
