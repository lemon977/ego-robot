"""Independently fenced candidate Composite operation; no model or rendering."""
from __future__ import annotations
import sys


def main(argv=None):
    from chaoyang.governance.common import REPO_ROOT
    from chaoyang.governance.inpainting_candidate_registration import (
        COMPOSITE_OPERATION, authorize_composite_operation)
    args = list(sys.argv[1:] if argv is None else argv)
    authorize_composite_operation(COMPOSITE_OPERATION, args, repo_root=REPO_ROOT)
    from chaoyang.inpainting_candidate.composite_worker import main as worker_main
    return worker_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
