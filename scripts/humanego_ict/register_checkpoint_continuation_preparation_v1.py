"""Register bounded CPU successor engineering through the existing publisher."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

from chaoyang.governance.common import REPO_ROOT, artifact_ref, canonical_bytes, load_json

TASK = "pick_place_humanego_ict_checkpoint_continuation_cpu_v1_20261011"
PLAN = "docs/plans/PICK_PLACE_HUMANEGO_ICT_DERIVED_REVO2_V1/checkpoint_continue_20k_20261011"


def main():
    registrar = REPO_ROOT / "scripts/humanego_ict/register_derived_training_source_task_v1.py"
    spec = importlib.util.spec_from_file_location("_continuation_existing_publisher", registrar)
    publisher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(publisher)
    publisher.TASK = TASK
    publisher.ROOT = REPO_ROOT / "_run/current" / TASK / "attempts/attempt_0001"
    publisher.PACKET = REPO_ROOT / "tasks/current" / TASK / "TASK_PACKET.json"
    publisher.WALL_SECONDS = 7200
    original = publisher._validate_packet

    def scope(packet):
        packet["objective"] = "Prepare exact sealed-state successor resume for two research ICT checkpoints; no real model calls"
        packet["phase"] = "HUMANEGO_ICT_CHECKPOINT_CONTINUATION_CPU"
        packet["plan_revision"] = "CHECKPOINT_CONTINUE_20K_20261011"
        packet["inputs"]["plan"] = artifact_ref(REPO_ROOT / PLAN / "EXECUTION_ZH.md")
        packet["inputs"]["protected_production_code"] = load_json(
            REPO_ROOT / "docs/governance/ALGORITHM_CONTRACT.json"
        )["special_status_contracts"]["pick_place_humanego_ict_production_training_v1"]["code_closure"]
        parents = {}
        for branch, step in (("raw", 4376), ("robotized", 4768)):
            task = f"pick_place_humanego_ict_production_train_{branch}_a01_20261011"
            directory = REPO_ROOT / "_run/current" / task / "attempts/attempt_0001"
            parents[branch] = {
                "task_packet": artifact_ref(REPO_ROOT / "tasks/current" / task / "TASK_PACKET.json"),
                "outer_result": artifact_ref(directory / "RESULT.json"),
                "inner_result": artifact_ref(directory / "output" / f"RESULT_{step:06d}.json"),
                "training_config": artifact_ref(directory / "TRAINING_CONFIG.json"),
                "checkpoint": artifact_ref(directory / "output" / f"checkpoint_{step:06d}.pt"),
                "checkpoint_sidecar": artifact_ref(directory / "output" / f"checkpoint_{step:06d}.pt.ref.json"),
            }
        packet["inputs"]["parents"] = parents
        packet["inputs"]["scope"] = {"safe_cpu_full_checkpoint_read": True,
            "real_model_forward_calls": 0, "real_optimizer_updates": 0, "original_production_and_training_code_immutable": True}
        packet["read_set"] = ["AGENTS.md", "docs/governance/CURRENT_STATUS_RECEIPT.json",
            "docs/governance/ALGORITHM_CONTRACT.json", "tasks/current/INDEX.json", PLAN + "/EXECUTION_ZH.md",
            "src/chaoyang/humanego_ict/training.py", "src/chaoyang/humanego_ict/production_training_cli.py",
            "src/chaoyang/governance/current_r3_contracts.py"]
        packet["read_set_expansions"] = ["Explicit immutable parent/source/code refs and new successor modules/tests only; original datasets readonly"]
        packet["write_set"] = [
            *["src/chaoyang/humanego_ict/" + name + ".py" for name in (
                "production_resume", "production_resume_gate", "production_resume_cli", "production_resume_contract")],
            "src/chaoyang/ops/train_pick_place_humanego_ict_production_resume_v1.py",
            "src/chaoyang/governance/current_r3_contracts.py",
            "scripts/humanego_ict/register_checkpoint_continuation_preparation_v1.py",
            "scripts/humanego_ict/production_resume_runner_v1.py",
            *["tests/humanego_ict/" + name + ".py" for name in (
                "test_production_resume", "test_production_resume_gate", "test_production_resume_cli")],
            PLAN, str(publisher.ROOT.relative_to(REPO_ROOT)), str(publisher.PACKET.relative_to(REPO_ROOT)),
            "tasks/receipts/" + TASK.upper() + ".json", "tasks/current/INDEX.json", "docs/governance",
        ]
        packet["required_outputs"] = ["CPU_TEST_RESULT.json", "PROTECTED_INPUTS_CHECK.json", "RESULT.json", "POST_VALIDATION.json"]
        packet["runtime_code"] = [artifact_ref(Path(__file__)), artifact_ref(registrar)]
        packet["claim_limit"] = "CPU successor resume engineering only, synthetic tests and safe verified checkpoint reads; no real model/optimizer/GPU/deployment. Source and numeric gates unchanged."
        packet["run_signature"] = hashlib.sha256(canonical_bytes({"inputs": packet["inputs"],
            "runtime_code": packet["runtime_code"], "budgets": packet["budgets"]})).hexdigest()
        return original(packet)

    publisher._validate_packet = scope
    return publisher.main()


if __name__ == "__main__":
    raise SystemExit(main())
