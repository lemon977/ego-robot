"""Bounded CPU preparation; reuse the existing publisher, not a new queue."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

from chaoyang.governance.common import REPO_ROOT, artifact_ref, canonical_bytes

TASK = "pick_place_humanego_ict_two_checkpoint_preparation_v1_20261011"
PLAN = "docs/plans/PICK_PLACE_HUMANEGO_ICT_DERIVED_REVO2_V1/checkpoint_delivery_8h_20261011"
HANDOFF = REPO_ROOT / "_run/handoffs/revo2_ict_deploy_completion_v1/codex_20261010T152841Z_2cf21b18"


def main() -> int:
    registrar = REPO_ROOT / "scripts/humanego_ict/register_derived_training_source_task_v1.py"
    spec = importlib.util.spec_from_file_location("_existing_two_checkpoint_publisher", registrar)
    if spec is None or spec.loader is None:
        raise RuntimeError("EXISTING_PUBLISHER_UNAVAILABLE")
    publisher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(publisher)
    publisher.TASK = TASK
    publisher.ROOT = REPO_ROOT / "_run/current" / TASK / "attempts/attempt_0001"
    publisher.PACKET = REPO_ROOT / "tasks/current" / TASK / "TASK_PACKET.json"
    publisher.PLAN = PLAN
    publisher.WALL_SECONDS = 3600
    original_validator = publisher._validate_packet

    def validate_scoped_packet(packet: dict) -> list:
        packet["inputs"]["plan"] = artifact_ref(REPO_ROOT / PLAN / "EXECUTION_ZH.md")
        packet["inputs"]["new_handoff"] = [artifact_ref(HANDOFF / name) for name in
            ("DELIVERY_INDEX.json", "SOURCE_INDEX.json", "REPORT_ZH.md", "MISSING_ITEMS.md")]
        packet["inputs"]["pretrained"] = artifact_ref(
            REPO_ROOT / "assets/models/humanego/pretrained_humanego/latest.pt")
        packet["inputs"]["scope"] = {
            "schema": "humanego-ict-two-checkpoint-preparation-v1",
            "forward_calls": 0, "optimizer_updates": 0, "hardware_calls": 0,
            "safe_cpu_pretrained_inspection_allowed": True,
            "source_datasets_readonly": True, "deadline_utc": "2026-10-11T01:46:32+00:00",
        }
        packet["objective"] = "Prepare direct Raw and causal Robotized ICT training with original source replay and shared derived supervision"
        packet["phase"] = "HUMANEGO_ICT_TWO_CHECKPOINT_CPU_PREPARATION"
        packet["claim_limit"] = "CPU preparation only; no training, inference, physical deployment, deletion or external writes. Numeric source/time/group/window gates unchanged. Production and training require successor packets."
        packet["read_set_expansions"] += [
            "User-provided SHA-indexed complete HAND/GEOM/CLIENT/PROTOCOL handoff; no remote execution",
            "Current source manifest and explicit original NAS source refs, capture/profile/time/group records",
            "Frozen existing Robot raster/asset/helpers for new current/past anchor producer",
            "Existing research training/source contracts and pretrained tensor inspection; no model forward",
        ]
        names = ("production_source.py", "production_data.py", "production_renderer.py",
                 "production_training_cli.py", "production_training_gate.py", "production_contract.py")
        packet["write_set"] = ["src/chaoyang/humanego_ict/" + name for name in names] + [
            "src/chaoyang/ops/prepare_pick_place_humanego_ict_production_v1.py",
            "src/chaoyang/ops/train_pick_place_humanego_ict_production_v1.py",
            "src/chaoyang/governance/current_r3_contracts.py",
            "scripts/humanego_ict/register_two_checkpoint_preparation_v1.py",
            "scripts/humanego_ict/production_runner_v1.py",
            "scripts/humanego_ict/train_production_pick_place.sh",
            "tests/humanego_ict/test_production_source.py",
            "tests/humanego_ict/test_production_data.py",
            "tests/humanego_ict/test_production_renderer.py",
            "tests/humanego_ict/test_production_training.py",
            PLAN, str(publisher.ROOT.relative_to(REPO_ROOT)), str(publisher.PACKET.relative_to(REPO_ROOT)),
            "tasks/receipts/" + TASK.upper() + ".json", "tasks/current/INDEX.json", "docs/governance",
        ]
        packet["required_outputs"] = ["SOURCE_READINESS.json", "CPU_TEST_RESULT.json",
            "PROTECTED_INPUTS_CHECK.json", "RESULT.json", "POST_VALIDATION.json"]
        packet["runtime_code"] = [artifact_ref(Path(__file__)), artifact_ref(registrar)]
        packet["run_signature"] = hashlib.sha256(canonical_bytes({
            "inputs": packet["inputs"], "runtime_code": packet["runtime_code"],
            "schema": packet["schema_version"], "budgets": packet["budgets"],
        })).hexdigest()
        return original_validator(packet)

    publisher._validate_packet = validate_scoped_packet
    # The existing template resolves this name before the scope callback.
    publisher.PLAN = "docs/plans/PICK_PLACE_HUMANEGO_ICT_DERIVED_REVO2_V1/quick_completion_v1"
    return publisher.main()


if __name__ == "__main__":
    raise SystemExit(main())
