"""Candidate authority: research-derived outputs are not deployment permission."""
from pathlib import Path

from chaoyang.governance.common import artifact_ref
from .research_training_gate import ROUTE, OPERATION, EXECUTION_MODE
from .research_data import SCHEMA, TRAINING_SCOPE


def candidate_contract(root: Path) -> dict:
    paths = ["src/chaoyang/ops/" + OPERATION + ".py",
        "scripts/humanego_ict/train_research_pick_place.sh",
        *["src/chaoyang/humanego_ict/" + name + ".py" for name in (
            "research_data", "research_training_cli", "research_training_gate", "research_training_contract",
            "source_adapter", "derived_data", "retarget", "derived_runtime", "derived_contract", "inference", "derived_training_cli",
            "derived_training_gate", "runtime_gate", "checkpoint_export", "data", "model", "training")],
        "src/chaoyang/cli/__init__.py", "src/chaoyang/human_ego/training/FlowMatchingModel.py",
        "src/chaoyang/human_ego/training/checkpoint_transfer.py",
        "src/chaoyang/human_ego/utils/atomic_io.py", "src/chaoyang/human_ego/utils/frozen_contract.py"]
    if any(not (root / path).is_file() for path in paths):
        return {}
    return {ROUTE: {"candidate_operation": OPERATION, "execution_mode": EXECUTION_MODE,
        "execution_status": "SEPARATE_NON_DEPLOYABLE_DERIVED_MOTION_TRAINING_CANDIDATE",
        "code_closure": [artifact_ref(root / path) for path in paths], "task_packet_required": True,
        "manifest_schema": SCHEMA, "training_scope": TRAINING_SCOPE, "supervision_class": "DERIVED_RETARGET",
        "control_ground_truth": False, "physical_deployment_authorized": False,
        "current_runnable_operation": None, "qualification_status": "NOT_PASSED",
        "real_control_route_unchanged": True, "old_derived_route_unchanged": True,
        "causal_time_group_window_gates_unchanged": True,
        "physical_base_tcp_fk_qualification_required_for": "PHYSICAL_DEPLOYMENT_ONLY",
        "claim_limit": "Direct Raw/Robotized derived-motion checkpoints, not measured Revo2 controls or deployment approval. Research source qualification requires ns clock/domain evidence, source replay, original numeric timing/coverage gates, independent acquisition groups and causal RGB. Training separately requires frozen inputs/code, finite packet, live writer, GPU lease and storage."}}
