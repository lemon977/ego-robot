#!/usr/bin/env python3
"""Fail-closed RC1 audit for v77 causal Robot input eligibility."""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
import json
from pathlib import Path

from chaoyang.governance.common import REPO_ROOT, artifact_ref, atomic_json, atomic_write, load_json, now_iso


V77 = REPO_ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_chaoyang_r22_4h/lanes/robot_v77_terminal_index_r22/attempts/attempt_0003/RESULT.json"
V78_CAUSAL_DIAGNOSTIC = REPO_ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_r22_robot_v78_causal_metric_extractor_v1/attempts/attempt_0001/RESULT.json"
FORBIDDEN = ("bidirectional", "reverse", "lookahead", "full_sequence", "full-trajectory")


def _walk_strings(value):
    if isinstance(value, dict):
        for item in value.values(): yield from _walk_strings(item)
    elif isinstance(value, list):
        for item in value: yield from _walk_strings(item)
    elif isinstance(value, str):
        yield value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    if args.output_root.exists():
        raise RuntimeError(f"fresh output required: {args.output_root}")
    args.output_root.mkdir(parents=True)
    v77 = load_json(V77)
    strings = list(_walk_strings(v77))
    forbidden_refs = sorted({text for text in strings if any(token in text.lower() for token in FORBIDDEN)})
    # The existing v78 artifact is explicitly a diagnostic extractor, not a
    # prefix solver. It cannot close RC1 T3 merely by renaming its output.
    diagnostic = load_json(V78_CAUSAL_DIAGNOSTIC)
    causal_solver_entry = None
    status = "BLOCKED_PREREQ" if causal_solver_entry is None else "PASSED"
    created = now_iso()
    result = {
        "schema_version": "chaoyang-rc1-t3-causal-preflight-v1",
        "task_id": "rc1_t3_v77_causal_robot", "status": status, "created_at": created,
        "v77_terminal_index": artifact_ref(V77),
        "existing_v78_diagnostic": artifact_ref(V78_CAUSAL_DIAGNOSTIC),
        "forbidden_future_dependent_references": forbidden_refs,
        "causal_prefix_solver_entry": causal_solver_entry,
        "blocked_reason": "No verified prefix-only v77 solver entry exists. Existing bidirectional/full-sequence states and read-only v78 diagnostics cannot be adopted as CAUSAL_TRAINING_INPUT.",
        "control_ground_truth": False, "physical_deployment_authorized": False,
        "claim_limit": "Causality prerequisite audit only; no Robot trajectory or authority produced.",
    }
    result_path = args.output_root / "RESULT.json"; atomic_json(result_path, result)
    atomic_json(args.output_root / "METRICS.json", {"status": status, "forbidden_reference_count": len(forbidden_refs), "causal_solver_entry_present": False})
    atomic_json(args.output_root / "NEXT_ACTION.json", {"task_id": "implement_v77_prefix_recompute_entry", "required": True, "must_not_consume": list(FORBIDDEN)})
    atomic_write(args.output_root / "DECISION.md", b"# RC1 T3\n\nBLOCKED_PREREQ: existing v77/v78 evidence does not provide a verified prefix-only solver entry. Bidirectional trajectories remain offline diagnostics.\n")
    manifest = {"schema_version": "chaoyang-rc1-t3-manifest-v1", "artifacts": [artifact_ref(x) for x in (result_path, args.output_root / "METRICS.json", args.output_root / "NEXT_ACTION.json", args.output_root / "DECISION.md")]}
    manifest_path = args.output_root / "ARTIFACT_MANIFEST.json"; atomic_json(manifest_path, manifest)
    atomic_json(args.output_root / "RUN_RECEIPT.json", {"schema_version": "chaoyang-rc1-t3-run-receipt-v1", "task_id": "rc1_t3_v77_causal_robot", "status": status, "result": artifact_ref(result_path), "manifest": artifact_ref(manifest_path)})
    atomic_json(args.output_root / "RESULT_SUMMARY.json", {"task_id": "rc1_t3_v77_causal_robot", "status": status, "next_action": "implement_v77_prefix_recompute_entry"})
    print(json.dumps({"status": status, "result": str(result_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
