"""Separate finite evaluation operation; no old route/source modification."""
from pathlib import Path
from chaoyang.governance.common import artifact_ref
from chaoyang.humanego_ict.derived_contract import candidate_contract as old_contract, ROUTE as OLD_ROUTE

OPERATION = "infer_pick_place_humanego_ict_revo2_evaluation_v1"
ROUTE = "pick_place_humanego_ict_revo2_evaluation_v1"


def candidate_contract(root: Path) -> dict:
    paths = ["src/chaoyang/humanego_ict/evaluation_runtime_v1.py",
             "src/chaoyang/humanego_ict/evaluation_contract_v1.py",
             "src/chaoyang/ops/" + OPERATION + ".py"]
    previous = old_contract(root).get(OLD_ROUTE)
    if previous is None or any(not (root / p).is_file() for p in paths):
        return {}
    closure = previous["code_closure"] + [artifact_ref(root / p) for p in paths]
    return {ROUTE: {"candidate_operations": [OPERATION], "task_packet_required": True,
        "code_closure": closure, "execution_mode": "HUMANEGO_ICT_REVO2_OFFLINE_EVALUATION",
        "supervision_class": "DERIVED_RETARGET", "source_policy": "FROZEN_ADMITTED_PROJECT_PAYLOAD_SNAPSHOT_RESEARCH_ONLY",
        "control_ground_truth": False, "physical_deployment_authorized": False,
        "current_runnable_operation": None, "optimizer_updates_authorized": 0,
        "samples_max_per_finite_task": 57, "model_calls_max_per_finite_task": 1140,
        "claim_limit": "Real-model offline research diagnostics only. Raw H50 predictions are retained before strict REJECT decoding; numerical/physical qualification is not granted."}}
