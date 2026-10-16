"""Fenced Mask or fixed-input Clean candidate operation; no default adoption."""
from __future__ import annotations

import sys


def main(argv=None):
    from chaoyang.governance.common import REPO_ROOT
    from chaoyang.governance.inpainting_candidate_registration import authorize_candidate_operation, VISUAL_OPERATION

    args = list(sys.argv[1:] if argv is None else argv)
    authorize_candidate_operation(VISUAL_OPERATION, args, repo_root=REPO_ROOT)
    from chaoyang.inpainting_candidate.model_worker import main as run
    return run(args)
