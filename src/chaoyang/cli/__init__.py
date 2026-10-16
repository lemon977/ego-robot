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
    from chaoyang.governance.exact78_visual_selection_v1 import filter_selected_operations

    return filter_selected_operations(operations, value)


def _is_exact78_invocation(name: str, args: list[str]) -> bool:
    lowered = name.lower()
    if "exact78" in lowered or "cards042" in lowered or "0902" in lowered:
        return True
    # One historical 0902 task used a shared Role Mask operation.  Gate that
    # invocation by its scoped arguments without changing unrelated branches.
    if name == "run_role_mask_runtime_contract_repair_v71":
        return any("exact78_" in value.lower() or "0902_" in value.lower() for value in args)
    return False


def _run_module(name: str, args: list[str]) -> int:
    if not name.replace("_", "").isalnum():
        raise SystemExit(f"invalid operation name: {name!r}")
    maintained = _maintained_operations()
    from chaoyang.governance.operation_retirement import require_not_retired
    require_not_retired(get_paths().repo_root, name)
    if name not in maintained:
        raise SystemExit(f"operation is not maintained by the current algorithm contract: {name}")
    if name in {"run_inpainting_candidate_cpu_admission_v1", "run_inpainting_candidate_visual_stages_v1"}:
        from chaoyang.governance.inpainting_candidate_registration import authorize_candidate_operation

        authorize_candidate_operation(name, args, repo_root=get_paths().repo_root)
    # The Exact78 authority gate must run before importing any algorithm module.
    if _is_exact78_invocation(name, args):
        from chaoyang.governance.exact78_operation_gate import authorize_exact78_operation

        authorize_exact78_operation(name, args)
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
    submit = subparsers.add_parser("submit", help="submit a governed candidate recipe")
    submit.add_argument("recipe")
    submit.add_argument("args", nargs=argparse.REMAINDER)
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
    if parsed.command == "submit":
        if parsed.recipe == "inpainting-raw-to-robot":
            from chaoyang.cli.inpainting_submit import main as submit_inpainting

            return submit_inpainting(parsed.args)
        if parsed.recipe != "exact78-raw-to-robot":
            raise SystemExit(f"unknown submit recipe: {parsed.recipe}")
        from chaoyang.governance.exact78_visual_selection_v1 import assert_selected_submit_allowed

        contract = json.loads((get_paths().repo_root / "docs/governance/ALGORITHM_CONTRACT.json").read_text(encoding="utf-8"))
        try:
            assert_selected_submit_allowed(contract, parsed.recipe)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        from chaoyang.cli.exact78_submit import main as submit_exact78

        return submit_exact78(parsed.args)
    return _run_module(parsed.operation, parsed.args)
