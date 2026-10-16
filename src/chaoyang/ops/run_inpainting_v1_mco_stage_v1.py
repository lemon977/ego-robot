"""Finite original-V1 candidate entry: authorize before algorithm imports."""
from __future__ import annotations

import importlib.util
import sys
from chaoyang.governance.common import REPO_ROOT


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    path = REPO_ROOT / "scripts/maintenance/inpainting_v1_mco_task.py"
    spec = importlib.util.spec_from_file_location("_mco_operation_gate", path)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    packet = gate.authorize_operation(args)
    from chaoyang.inpainting_candidate.v1_mco_worker import execute
    return execute(packet)


if __name__ == "__main__":
    raise SystemExit(main())
