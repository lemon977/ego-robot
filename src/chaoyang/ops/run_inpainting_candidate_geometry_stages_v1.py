"""Fenced zero-model CPU Motion/Render; never a default production entry."""
from __future__ import annotations

import sys


def main(argv=None):
    from chaoyang.governance.common import REPO_ROOT
    from chaoyang.governance.inpainting_candidate_registration import (
        GEOMETRY_OPERATION, authorize_geometry_operation)
    args = list(sys.argv[1:] if argv is None else argv)
    authorize_geometry_operation(GEOMETRY_OPERATION, args, repo_root=REPO_ROOT)
    from chaoyang.inpainting_candidate.geometry_worker import main as run
    return run(args)
