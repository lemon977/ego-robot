#!/usr/bin/env python3
"""生成 V7.1-R3 合同任务包；不修改 current task index 或治理账本。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = PROJECT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_pipeline_contracts_r3"
PLAN = PROJECT / "docs/reference/contracts/FULLCHAIN_PIPELINE_CONTRACTS_R3_ZH.md"


TASKS = (
    ("DEPTH-00", "depth-00", "depth", [], "Audit coordinate, unit, calibration identity and optical-Z source contracts"),
    ("DEPTH-10", "depth-10", "depth", ["DEPTH-00=PASSED"], "Compute FoundationStereo internal quality evidence without native confidence"),
    ("DEPTH-20", "depth-20", "depth", ["DEPTH-10=PASSED", "CONTROLLER_MANUS_INPUT_PRESENT"], "Apply causal bounded Stereo optical-Z correction to Controller wrist"),
    ("MASK-ROLE", "mask-role", "mask", [], "Validate SAM3.1 role identities and formal H4 status semantics"),
    ("MASK-OBJECT", "mask-object", "mask", [], "Validate task-object identity, offscreen and Chips three-instance contracts"),
    ("ATLAS-10", "atlas", "clean", ["MASK-OBJECT=PASSED"], "Validate pose-verified per-instance object atlas inputs"),
    ("DONOR-10K", "donor", "clean", [], "Validate causal same-session donor ordering"),
    ("CLEAN-DOMAINS-R3", "clean", "clean", ["MASK-ROLE=PASSED", "MASK-OBJECT=PASSED"], "Validate M_remove, M_flow, M_write and no-regression Clean gates"),
    ("CONTACT-10", "contact-10", "contact", ["DEPTH-10=PASSED", "MASK-OBJECT=PASSED"], "Validate per-finger hypotheses and acyclic one-way evidence"),
    ("OCCLUSION-SILVER-R3", "occlusion-silver", "occlusion", ["CLEAN-DOMAINS-R3=PASSED", "CONTACT-10=TERMINAL"], "Validate Silver provenance, z-buffer and UNKNOWN training exclusion"),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def materialize(root: Path) -> dict[str, object]:
    created_at = datetime.now(timezone.utc).isoformat()
    packets: list[dict[str, object]] = []
    for task_id, mode, stage, local_prerequisites, objective in TASKS:
        slug = task_id.lower().replace("-", "_")
        packet_dir = root / "task_packets" / slug
        packet_dir.mkdir(parents=True, exist_ok=False)
        artifact_dir = root / "artifacts" / slug
        packet = {
            "schema_version": "exact78-task-packet-v1",
            "task_id": task_id,
            "plan_revision": "chaoyang-v7.1-r3-development",
            "packet_revision": 1,
            "created_at": created_at,
            "stage": stage,
            "objective": objective,
            "workdir": str(PROJECT),
            "command_argv": ["python", str(PROJECT / "src/chaoyang/ops/validate_pipeline_contracts_r3.py"), "--mode", mode, "--dry-run"],
            "read_set": [
                str(PROJECT / "AGENTS.md"),
                str(PROJECT / "docs/governance/CURRENT_PROJECT_STATUS_MIN.json"),
                str(PROJECT / "docs/governance/CURRENT_BASELINE_REGISTRY_V2.json"),
                str(PLAN),
                str(PROJECT / "src/chaoyang/pipeline/depth_mask_clean_contracts_r3.py" if stage in {"depth", "mask", "clean"} else PROJECT / "src/chaoyang/pipeline/contact_occlusion_contracts_r3.py"),
            ],
            "write_set": [str(artifact_dir / "attempts"), str(artifact_dir / "final")],
            "prerequisites": ["G0_CORE_GOVERNANCE=PASS", "governance_freshness=FRESH", *local_prerequisites],
            "gates": ["SCHEMA_VALID", "FAIL_CLOSED", "IMMUTABLE_ATTEMPT", "NO_AUTHORITY_PROMOTION"],
            "quality_gates": ["DEVELOPMENT_FIXTURE_PASS", "CLAIM_LIMIT_PRESENT"],
            "budgets": {"cpu_seconds": 300, "gpu_seconds": 0, "wall_seconds": 600},
            "attempt_max": 1,
            "stop_condition": "PASSED or explicit FAILED/BLOCKED terminal; never remain RUNNING",
            "stop_conditions": ["PASSED", "FAILED_QUALITY_C", "FAILED_RUNTIME_FINAL", "BLOCKED_PREREQ"],
            "output_contract": ["RESULT.json", "ARTIFACT_MANIFEST.json", "METRICS.json", "RUN_RECEIPT.json", "DECISION.md", "NEXT_ACTION.json"],
            "claim_limit": "Development contract/dry-run only; no current authority, external truth, native Stereo confidence, or Gold accuracy.",
            "executor_epoch": 1,
            "fencing": {"pid_startticks_required": True, "immutable_final": True},
            "dispatch_policy": "EXPLICIT_USER_OR_PARENT_AGENT_AUTHORIZATION",
            "authority_promoted": False,
            "artifact_revision_contract": {"forbid_in_place_overwrite": True, "required_revision_status": "VALID_FOR_PINNED_REVISION"},
            "source_execution_plan": {"path": str(PLAN), "bytes": PLAN.stat().st_size, "sha256": sha256(PLAN), "status": "DEVELOPMENT_CONTRACT_NOT_CURRENT_AUTHORITY"},
            "ai_io_limits": {"max_files_initial_read": 8, "max_search_results": 20, "max_log_tail_lines": 80}
        }
        path = packet_dir / "TASK_PACKET.json"
        path.write_text(json.dumps(packet, ensure_ascii=False, indent=2) + "\n")
        packets.append({"task_id": task_id, "path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)})
    index = {
        "schema_version": "PIPELINE_CONTRACT_TASK_PACKET_INDEX_R3",
        "plan_revision": "chaoyang-v7.1-r3-development",
        "created_at": created_at,
        "status": "DEVELOPMENT_PACKETS_MATERIALIZED_NOT_CURRENT",
        "authority_promoted": False,
        "packets": packets,
        "claim_limit": "Routing and contract verification only; current governance index is unchanged."
    }
    index_path = root / "TASK_PACKET_INDEX.json"
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n")
    return {"index": str(index_path), "packets": len(packets)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps({"status": "DRY_RUN", "output_root": str(args.output_root), "task_count": len(TASKS)}, ensure_ascii=False))
        return 0
    if args.output_root.exists():
        raise SystemExit(f"refusing to overwrite immutable root: {args.output_root}")
    args.output_root.mkdir(parents=True)
    print(json.dumps(materialize(args.output_root), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
