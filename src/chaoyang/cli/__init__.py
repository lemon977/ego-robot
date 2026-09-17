"""Unified Chaoyang command-line entry point."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

from chaoyang.paths import get_paths


def _maintained_operations() -> set[str]:
    contract = get_paths().repo_root / "docs/governance/ALGORITHM_CONTRACT.json"
    value = json.loads(contract.read_text(encoding="utf-8"))
    operations: set[str] = set()

    def visit(item: object) -> None:
        if isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)
        elif isinstance(item, str) and "/src/chaoyang/ops/" in item and item.endswith(".py"):
            operations.add(Path(item).stem)

    visit(value)
    return operations


def _run_module(name: str, args: list[str]) -> int:
    if not name.replace("_", "").isalnum():
        raise SystemExit(f"invalid operation name: {name!r}")
    maintained = _maintained_operations()
    if name not in maintained:
        raise SystemExit(f"operation is not maintained by the current algorithm contract: {name}")
    module = importlib.import_module(f"chaoyang.ops.{name}")
    entry = getattr(module, "main", None)
    if entry is None:
        raise SystemExit(f"operation has no main(): {name}")
    previous = sys.argv
    try:
        sys.argv = [name, *args]
        result = entry()
    finally:
        sys.argv = previous
    return int(result or 0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="chaoyang")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("paths", help="print resolved project paths")
    run = subparsers.add_parser("run", help="run a named maintained operation")
    run.add_argument("operation")
    run.add_argument("args", nargs=argparse.REMAINDER)
    subparsers.add_parser("validate-governance", help="validate current governance state")
    parsed = parser.parse_args(argv)
    if parsed.command == "paths":
        print(json.dumps(get_paths().as_dict(), indent=2, sort_keys=True))
        return 0
    if parsed.command == "validate-governance":
        from chaoyang.governance.validate_governance_state import main as validate
        previous = sys.argv
        try:
            sys.argv = ["validate-governance"]
            return int(validate() or 0)
        finally:
            sys.argv = previous
    return _run_module(parsed.operation, parsed.args)
