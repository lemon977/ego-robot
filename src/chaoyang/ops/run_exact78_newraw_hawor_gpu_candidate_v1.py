"""Fenced, HaWoR-only GPU consumer of one sealed non-T0 CPU admission.

Unregistered candidate: no model or producer import occurs before the
Exact78 CLI operation gate. The executor must audit native opens and seal the
stage after this operation exits; RESULT.json here is provisional until then.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from chaoyang.governance.exact78_operation_gate import authorize_exact78_operation


OPERATION = "run_exact78_newraw_hawor_gpu_candidate_v1"


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    authorize_exact78_operation(OPERATION, arguments)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer-result", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(arguments)

    # Importing the HaWoR stage and its model dependencies before this point
    # would permit a direct Python invocation to evade the operation gate.
    from chaoyang.governance.common import REPO_ROOT, load_json
    from chaoyang.pipeline.exact78_newraw_hawor_stage_candidate_v1 import run_stage

    index = load_json(REPO_ROOT / "tasks/current/INDEX.json")
    active = [item for item in index.get("task_packets", [])
              if item.get("execution_allowed") is True]
    if len(active) != 1:
        raise RuntimeError("SINGLE_ACTIVE_PACKET_REQUIRED")
    packet = load_json(REPO_ROOT / active[0]["packet_path"])
    if args.producer_result.resolve(strict=True) != Path(
            packet["inputs"]["upstream_admission_result"]["path"]).resolve(strict=True):
        raise RuntimeError("PRODUCER_RESULT_COMMAND_BINDING_DRIFT")
    result = run_stage(packet, args.output_root)
    print(json.dumps({"status": result["status"],
                      "result": str(args.output_root / "RESULT.json"),
                      "native_open_audit_pending": True}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
