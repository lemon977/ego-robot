"""Reuse the existing finite CPU publisher for research/deployment adapters."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

from chaoyang.governance.common import REPO_ROOT, artifact_ref, canonical_bytes, load_json

TASK = "pick_place_humanego_ict_research_deploy_cpu_v1_20261010"
PLAN = "docs/plans/PICK_PLACE_HUMANEGO_ICT_DERIVED_REVO2_V1/quick_completion_v1"


def main() -> int:
    registrar = REPO_ROOT / "scripts/humanego_ict/register_derived_training_source_task_v1.py"
    spec = importlib.util.spec_from_file_location("_existing_ict_cpu_publisher", registrar)
    if spec is None or spec.loader is None:
        raise RuntimeError("EXISTING_PUBLISHER_UNAVAILABLE")
    publisher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(publisher)
    publisher.TASK = TASK
    publisher.ROOT = REPO_ROOT / "_run/current" / TASK / "attempts/attempt_0001"
    publisher.PACKET = REPO_ROOT / "tasks/current" / TASK / "TASK_PACKET.json"
    publisher.PLAN = PLAN
    publisher.WALL_SECONDS = 7200
    original_validator = publisher._validate_packet

    def validate_scoped_packet(packet: dict) -> list:
        contract = load_json(REPO_ROOT / "docs/governance/ALGORITHM_CONTRACT.json")
        packet["inputs"]["protected_training_code"] = contract["special_status_contracts"][
            "pick_place_humanego_ict_derived_training_v1"]["code_closure"]
        packet["inputs"]["scope"] = {
            "schema": "humanego-ict-research-deployment-cpu-v1",
            "model_calls": 0, "hardware_calls": 0, "calibration": "ABSENT_UNTIL_BOUND",
            "numeric_quality_gates_unchanged": True,
        }
        packet["objective"] = "Separate research-derived training from physical deployment and implement measured-state/protocol dry-run adapters"
        packet["phase"] = "HUMANEGO_ICT_RESEARCH_DEPLOY_CPU_ENGINEERING"
        packet["claim_limit"] = (
            "CPU engineering only. Research-derived labels are not measured robot controls; "
            "causal RGB, source/time/group/window gates remain unchanged. No physical deployment, "
            "model loading, training, deletion or writes to source datasets."
        )
        packet["read_set_expansions"].append("Existing tensor inference/bridge, v2 static protocol/FK/feedback evidence; new research and deployment code/tests")
        source = "src/chaoyang/humanego_ict/"
        packet["write_set"] = [
            source + name for name in (
                "research_data.py", "research_training_cli.py", "research_training_gate.py",
                "research_training_contract.py", "deployment_adapter.py",
            )
        ] + [
            "src/chaoyang/ops/train_pick_place_humanego_ict_research_v1.py",
            "scripts/humanego_ict/train_research_pick_place.sh",
            "scripts/humanego_ict/register_research_deploy_task_v1.py",
            "src/chaoyang/governance/current_r3_contracts.py",
            "tests/humanego_ict/test_research_data.py",
            "tests/humanego_ict/test_research_training.py",
            "tests/humanego_ict/test_deployment_adapter.py",
            PLAN, str(publisher.ROOT.relative_to(REPO_ROOT)),
            str(publisher.PACKET.relative_to(REPO_ROOT)),
            "tasks/receipts/" + TASK.upper() + ".json", "tasks/current/INDEX.json", "docs/governance",
        ]
        packet["runtime_code"] = [artifact_ref(Path(__file__)), artifact_ref(registrar)]
        packet["required_outputs"] = [
            "CPU_TEST_RESULT.json", "PROTECTED_INPUTS_CHECK.json", "RESULT.json", "POST_VALIDATION.json",
            "research_training_candidate", "deployment_dry_run_adapter", "REMOTE_RESOURCE_REQUEST_ZH.md",
        ]
        packet["run_signature"] = hashlib.sha256(canonical_bytes({
            "inputs": packet["inputs"], "runtime_code": packet["runtime_code"],
            "schema_version": packet["schema_version"], "budgets": packet["budgets"],
        })).hexdigest()
        return original_validator(packet)

    publisher._validate_packet = validate_scoped_packet
    return publisher.main()


if __name__ == "__main__":
    raise SystemExit(main())
