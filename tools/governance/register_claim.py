from __future__ import annotations

import argparse
from pathlib import Path

from tools.governance.common import (
    AUTHORITY_PATH,
    TASK_STATE_PATH,
    artifact_ref,
    load_json,
    publish_bundle,
)


CLAIM_STATUSES = {
    "SUPPORTED_EXTERNAL_TRUTH",
    "SUPPORTED_INTERNAL_CONSISTENCY",
    "DEVELOPMENT_EVIDENCE",
    "HYPOTHESIS_ONLY",
    "UNSUPPORTED",
    "WITHDRAWN",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--claim", required=True)
    parser.add_argument("--status", choices=sorted(CLAIM_STATUSES), required=True)
    parser.add_argument("--scope", required=True)
    parser.add_argument("--claim-limit", required=True)
    parser.add_argument("--evidence", action="append", required=True)
    parser.add_argument("--expected-revision", type=int, required=True)
    args = parser.parse_args()

    authority = load_json(AUTHORITY_PATH)
    state = load_json(TASK_STATE_PATH)
    candidate = {
        "claim": args.claim,
        "status": args.status,
        "scope": args.scope,
        "evidence": [artifact_ref(path) for path in args.evidence],
        "claim_limit": args.claim_limit,
    }

    claims = authority.setdefault("claims", [])
    matching = [index for index, item in enumerate(claims) if item.get("claim") == args.claim and item.get("scope") == args.scope]
    if matching:
        claims[matching[-1]] = candidate
    else:
        claims.append(candidate)

    receipt = publish_bundle(
        authority,
        state,
        event_type="CLAIM_REGISTERED",
        expected_revision=args.expected_revision,
        generator_path=Path(__file__),
    )
    print(receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
