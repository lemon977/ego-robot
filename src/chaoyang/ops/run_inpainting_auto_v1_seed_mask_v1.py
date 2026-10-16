"""One fenced automatic-to-V1 seed candidate; never a production default."""
from __future__ import annotations

import importlib.util
import sys


def main(argv=None):
    from chaoyang.governance.common import REPO_ROOT

    args = list(sys.argv[1:] if argv is None else argv)
    path = REPO_ROOT / "scripts/maintenance/inpainting_auto_v1_seed_one_hour_v1.py"
    spec = importlib.util.spec_from_file_location("_auto_v1_seed_authorization", path)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    packet = gate.authorize_operation(args)
    # Model imports are strictly after the packet/writer/deadline/SHA gate.
    from chaoyang.inpainting_candidate.automatic_v1_seeds import execute_mask_experiment

    result = execute_mask_experiment(packet)
    return result if isinstance(result, int) else 0
