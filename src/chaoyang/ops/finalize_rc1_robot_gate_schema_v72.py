#!/usr/bin/env python3
"""Publish compact receipts for the bounded Robot gate-schema successor."""
from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
import json
from pathlib import Path

from chaoyang.governance.common import artifact_ref, atomic_json, atomic_write, load_json, now_iso


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-result", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result_path = args.audit_result.resolve(strict=True)
    result = load_json(result_path)
    rows = result.get("rows") or []
    if len(rows) != 1 or rows[0].get("session") != "play_cards_0903_189":
        raise RuntimeError("exactly one Poker189 audit row required")
    row = rows[0]
    terminal = "PASSED" if row.get("hard_geometry_pass") is True else "FAILED_QUALITY_C"
    out = args.output_root.resolve()
    atomic_json(out / "RESULT_SUMMARY.json", {
        "task_id": "rc1_robot_gate_schema_v72_poker189",
        "status": terminal,
        "session": row["session"],
        "hard_geometry_pass": row.get("hard_geometry_pass"),
        "strict_pose_match": row.get("strict_pose_match"),
        "hard_gates": row.get("hard_gates"),
        "next_action": "AGGREGATE_ROBOT30_TERMINALS",
    })
    atomic_json(out / "RUN_RECEIPT.json", {
        "schema_version": "chaoyang-rc1-robot-gate-schema-v72-receipt-v1",
        "created_at": now_iso(),
        "status": terminal,
        "result": artifact_ref(result_path),
        "control_ground_truth": False,
        "physical_deployment_authorized": False,
        "training_eligible": False,
    })
    atomic_json(out / "NEXT_ACTION.json", {"next": "AGGREGATE_ROBOT30_TERMINALS"})
    atomic_write(out / "DECISION.md", (
        "# Robot gate schema v72\n\n"
        "Only verified v77 arm gate aliases were resolved. Solver outputs, thresholds, collision evidence and state-limit checks were not changed. "
        "The result remains offline visual development evidence and is not causal training or physical authority.\n"
    ).encode("utf-8"))
    print(json.dumps({"status": terminal, "result": artifact_ref(result_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
