from __future__ import annotations

import argparse
from pathlib import Path

from tools.governance.common import TASK_STATE_PATH, load_json, publish_bundle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--expected-revision", type=int, required=True)
    args = parser.parse_args()
    authority = load_json(args.candidate)
    state = load_json(TASK_STATE_PATH)
    receipt = publish_bundle(authority, state, event_type="AUTHORITY_PUBLISHED", expected_revision=args.expected_revision, generator_path=Path(__file__))
    print(receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

