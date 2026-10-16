"""Finite UNKNOWN-object negative ablation; imports models only after fencing."""
from __future__ import annotations

import importlib.util
import sys


def main(argv=None):
    from chaoyang.governance.common import REPO_ROOT
    args = list(sys.argv[1:] if argv is None else argv)
    path = REPO_ROOT / "scripts/maintenance/inpainting_neutral_object_v1.py"
    spec = importlib.util.spec_from_file_location("_neutral_object_authorization", path)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    packet = gate.authorize_operation(args)
    from chaoyang.inpainting_candidate.neutral_object_worker import execute_neutral_object_experiment
    result = execute_neutral_object_experiment(packet)
    return result if isinstance(result, int) else 0
