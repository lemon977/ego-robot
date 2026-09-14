from __future__ import annotations

import argparse
from pathlib import Path

from tools.governance.common import AUTHORITY_PATH, TASK_STATE_PATH, load_json, publish_bundle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-revision", type=int, required=True)
    args = parser.parse_args()
    receipt = publish_bundle(load_json(AUTHORITY_PATH), load_json(TASK_STATE_PATH), event_type="STATUS_REGENERATED", expected_revision=args.expected_revision, generator_path=Path(__file__))
    print(receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
