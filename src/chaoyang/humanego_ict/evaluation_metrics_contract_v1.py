"""Separately registered finite CPU consumer; no model or physical admission."""
from pathlib import Path
from chaoyang.governance.common import artifact_ref

OPERATION = "evaluate_pick_place_humanego_ict_revo2_offline_v1"
ROUTE = "humanego_ict_offline_metrics_v1"


def candidate_contract(root: Path):
    paths = ["src/chaoyang/humanego_ict/evaluation_metrics_v1.py",
        "src/chaoyang/humanego_ict/evaluation_metrics_contract_v1.py", "src/chaoyang/ops/" + OPERATION + ".py",
        "src/chaoyang/humanego_ict/inference.py", "src/chaoyang/humanego_ict/data.py",
        "src/chaoyang/humanego_ict/runtime_gate.py", "src/chaoyang/governance/common.py", "src/chaoyang/cli/__init__.py"]
    if any(not (root / path).is_file() for path in paths):
        return {}
    return {ROUTE: {"candidate_operation": OPERATION, "task_packet_required": True,
        "execution_mode": "HUMANEGO_ICT_OFFLINE_METRICS_VISUALIZATION_CPU",
        "code_closure": [artifact_ref(root / path) for path in paths], "gpu_count": 0, "model_calls_max": 0,
        "control_ground_truth": False, "physical_deployment_authorized": False, "current_runnable_operation": None,
        "claim_limit": "Descriptive held-out derived research metrics and prediction-labelled offline visuals only; no thresholds, physical qualification, model calls or NAS replay."}}
