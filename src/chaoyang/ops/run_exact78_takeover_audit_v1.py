"""Takeover read-only audits (references / inventory); writes only under --output-root."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chaoyang.paths import get_paths


AUDITS = ("references", "inventory")


def main(argv: list[str] | None = None) -> int:
    from chaoyang.governance.exact78_operation_gate import authorize_operation_entry
    authorize_operation_entry("run_exact78_takeover_audit_v1", argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", choices=AUDITS, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    repo = get_paths().repo_root.resolve()
    output = args.output_root.resolve()
    if repo not in output.parents or output.exists():
        raise SystemExit("output root must be a fresh path inside the project")
    output.mkdir(parents=True)
    if args.audit == "references":
        from chaoyang.exact78_baseline import audit_refs
        summary = audit_refs.run(repo, output)
    elif args.audit == "inventory":
        from chaoyang.exact78_baseline import audit_inventory
        summary = audit_inventory.run(repo, output)
    result = {"schema_version": "exact78-takeover-audit-stage-v1", "audit": args.audit,
              "status": "COMPLETE_READ_ONLY_AUDIT", "summary": summary,
              "files_written_outside_output_root": 0}
    (output / "RESULT.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
