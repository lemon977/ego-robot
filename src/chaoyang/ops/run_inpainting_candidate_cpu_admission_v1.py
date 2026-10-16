"""Fenced candidate CPU admission; not a complete Robot producer."""
from __future__ import annotations

import sys


def main(argv=None):
    from chaoyang.governance.common import REPO_ROOT
    from chaoyang.governance.inpainting_candidate_registration import authorize_candidate_operation, OPERATION

    args = list(sys.argv[1:] if argv is None else argv)
    authorize_candidate_operation(OPERATION, args, repo_root=REPO_ROOT)
    from chaoyang.inpainting_candidate.entry import operation_main
    return operation_main(args)
