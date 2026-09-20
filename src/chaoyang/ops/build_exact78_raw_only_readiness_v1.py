#!/usr/bin/env python3
"""Publish the fail-closed Exact78 Raw-only readiness decision at T+4."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Any

from chaoyang.governance.common import artifact_ref, atomic_json, load_json, now_iso


TASK_ID = "four_stream_pretraining_baseline_v32"
SCHEMA = "EXACT78_RAW_ONLY_READINESS_V1"


def _time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def assess(
    *,
    run_signature_path: Path,
    rebind_path: Path,
    pair_result_path: Path,
    funnel_result_path: Path,
    initializer_result_path: Path,
    trainer_path: Path,
    bundle_builder_path: Path,
    observed_at: str,
) -> dict[str, Any]:
    signature = load_json(run_signature_path)
    if signature.get("task_id") != TASK_ID:
        raise RuntimeError("run signature task mismatch")
    elapsed = (_time(observed_at) - _time(str(signature["t0"]))).total_seconds()
    if elapsed < 4 * 3600:
        raise RuntimeError("Raw-only readiness cannot be published before T+4")

    rebind = load_json(rebind_path)
    pair = load_json(pair_result_path)
    funnel = load_json(funnel_result_path)
    initializer = load_json(initializer_result_path)
    trainer_text = trainer_path.read_text(encoding="utf-8")
    bundle_text = bundle_builder_path.read_text(encoding="utf-8")

    cohort_closed = (
        rebind.get("status") == "PASS"
        and rebind.get("frozen_member_count") == 156
        and rebind.get("resolved_member_count") == 156
        and rebind.get("unresolved_members") == []
    )
    target_semantics_available = (
        initializer.get("accepted_state_scope") != "HAND_Q22_INITIALIZATION_ONLY"
        and initializer.get("arm_admission", {}).get("status")
        != "BLOCKED_UNOBSERVED_INSTALLATION_GEOMETRY"
    )
    pair_available = bool(pair.get("execution_allowed"))
    raw_only_interface_available = not (
        "--paired-ledger" in trainer_text
        and "validate_paired_ledgers" in trainer_text
        and "robotized_rgb" in bundle_text
    )

    blockers: list[str] = []
    if not cohort_closed:
        blockers.append("FROZEN_156_COHORT_NOT_CLOSED")
    if not target_semantics_available:
        blockers.extend(
            [
                "ABSOLUTE_ROBOT_HAND_ROOT_WORLD_TARGET_ABSENT",
                "ROBOT_BASE_WORLD_AND_TOOL_HAND_MOUNT_UNOBSERVED",
                "ROOT_RELATIVE_Q22_FK_CANNOT_SUBSTITUTE_ABSOLUTE_H50_ENDPOINT",
            ]
        )
    if not pair_available:
        blockers.append("CURRENT_RAW_ROBOTIZED_PAIR_ABSENT")
    if not raw_only_interface_available:
        blockers.append("RAW_ONLY_LEDGER_AND_TRAINER_ENTRY_ABSENT")

    executable = cohort_closed and target_semantics_available and raw_only_interface_available
    return {
        "schema_version": SCHEMA,
        "task_id": TASK_ID,
        "observed_at": observed_at,
        "elapsed_seconds": elapsed,
        "status": "READY_RAW_ONLY_DEV" if executable else "BLOCKED_FAIL_CLOSED",
        "execution_allowed": executable,
        "training_started": False,
        "cohort_closed": cohort_closed,
        "target_semantics_available": target_semantics_available,
        "current_pair_available": pair_available,
        "raw_only_interface_available": raw_only_interface_available,
        "blockers": blockers,
        "forbidden_substitutions": [
            "HAWOR_HUMAN_WRIST_AS_ROBOT_HAND_ROOT_TARGET",
            "IDENTITY_OR_ARBITRARY_WORLD_BASE",
            "IDENTITY_OR_ARBITRARY_TOOL_HAND_MOUNT",
            "ROOT_RELATIVE_Q22_FK_AS_ABSOLUTE_WORLD_ENDPOINT",
        ],
        "evidence": {
            "run_signature": artifact_ref(run_signature_path),
            "cohort_rebind": artifact_ref(rebind_path),
            "pair_result": artifact_ref(pair_result_path),
            "pair_funnel": artifact_ref(funnel_result_path),
            "hand_initializer": artifact_ref(initializer_result_path),
            "trainer": artifact_ref(trainer_path),
            "bundle_builder": artifact_ref(bundle_builder_path),
        },
        "funnel_status": funnel.get("status"),
        "authority": {
            "formal_ab_allowed": False,
            "four_model_completion": False,
            "training_complete": False,
            "training_eligible": False,
            "control_ground_truth": False,
            "physical_deployable": False,
        },
        "claim_limit": (
            "A blocked result proves only that the current frozen evidence cannot legally "
            "produce Raw-only H50 training labels; it does not reject the 156 identities "
            "or the root-relative hand-only review."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-signature", type=Path, required=True)
    parser.add_argument("--rebind", type=Path, required=True)
    parser.add_argument("--pair-result", type=Path, required=True)
    parser.add_argument("--funnel-result", type=Path, required=True)
    parser.add_argument("--initializer-result", type=Path, required=True)
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--bundle-builder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--observed-at", default=None)
    args = parser.parse_args()
    result = assess(
        run_signature_path=args.run_signature.resolve(strict=True),
        rebind_path=args.rebind.resolve(strict=True),
        pair_result_path=args.pair_result.resolve(strict=True),
        funnel_result_path=args.funnel_result.resolve(strict=True),
        initializer_result_path=args.initializer_result.resolve(strict=True),
        trainer_path=args.trainer.resolve(strict=True),
        bundle_builder_path=args.bundle_builder.resolve(strict=True),
        observed_at=args.observed_at or now_iso(),
    )
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"immutable Raw-only readiness receipt already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, result)
    print(json.dumps({"status": result["status"], "output": artifact_ref(output)}))
    return 0 if result["execution_allowed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
