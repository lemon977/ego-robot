"""Unregistered governed operation for the 0901 Chips001 structural Robot v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from chaoyang.governance.exact78_operation_gate import authorize_exact78_operation


OPERATION = "run_exact78_newraw_robot_structural_v2_candidate"


def main(argv: list[str] | None = None) -> int:
    args_in = sys.argv[1:] if argv is None else argv
    authorize_exact78_operation(OPERATION, args_in)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args(args_in)
    from chaoyang.pipeline.exact78_newraw_robot_structural_v2_candidate import run_candidate

    result = run_candidate(output_root=args.output_root)
    print(json.dumps({"status": result["status"],
                      "result_path": str(args.output_root / "RESULT.json")},
                     ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
