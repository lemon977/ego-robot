from __future__ import annotations

import argparse
from pathlib import Path

from tools.governance.common import AUTHORITY_PATH, TASK_STATE_PATH, artifact_ref, load_json, publish_bundle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    parser.add_argument("--authority-scope", required=True)
    parser.add_argument("--denominator-semantics", required=True)
    parser.add_argument("--total", type=int, required=True)
    parser.add_argument("--passed", type=int, required=True)
    parser.add_argument("--grade-c", type=int, default=0)
    parser.add_argument("--running", type=int, default=0)
    parser.add_argument("--blocked", type=int, default=0)
    parser.add_argument("--evidence", action="append", default=[])
    parser.add_argument("--expected-revision", type=int, required=True)
    args = parser.parse_args()
    authority = load_json(AUTHORITY_PATH)
    state = load_json(TASK_STATE_PATH)
    selected = next((item for item in authority["stages"] if item["stage"] == args.stage), None)
    if selected is None:
        raise SystemExit(f"unknown stage: {args.stage}")
    selected.update(
        total=args.total,
        passed=args.passed,
        grade_c=args.grade_c,
        running=args.running,
        blocked=args.blocked,
        authority_scope=args.authority_scope,
        denominator_semantics=args.denominator_semantics,
    )
    existing = {item["path"]: item for item in selected.get("evidence", [])}
    for evidence in args.evidence:
        reference = artifact_ref(evidence)
        existing[reference["path"]] = reference
    selected["evidence"] = list(existing.values())
    receipt = publish_bundle(authority, state, event_type=f"STAGE_AUTHORITY_UPDATED:{args.stage}", expected_revision=args.expected_revision, generator_path=Path(__file__))
    print(receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

