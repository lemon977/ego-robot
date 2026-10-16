"""Maintained derived training candidate without production authorization."""
from pathlib import Path

from chaoyang.governance.common import artifact_ref
from .derived_training_gate import ROUTE, OPERATION


def candidate_contract(root: Path) -> dict:
    paths = [
        "src/chaoyang/ops/" + OPERATION + ".py",
        "src/chaoyang/humanego_ict/derived_training_cli.py",
        "src/chaoyang/humanego_ict/derived_training_gate.py",
        "src/chaoyang/humanego_ict/derived_training_contract.py",
        "src/chaoyang/humanego_ict/checkpoint_export.py",
        "src/chaoyang/humanego_ict/derived_data.py",
        "src/chaoyang/humanego_ict/retarget.py",
        "src/chaoyang/humanego_ict/runtime_gate.py",
        "src/chaoyang/humanego_ict/data.py",
        "src/chaoyang/humanego_ict/model.py",
        "src/chaoyang/humanego_ict/training.py",
        "src/chaoyang/cli/__init__.py",
        "src/chaoyang/human_ego/training/FlowMatchingModel.py",
        "src/chaoyang/human_ego/training/checkpoint_transfer.py",
        "src/chaoyang/human_ego/utils/atomic_io.py",
        "src/chaoyang/human_ego/utils/frozen_contract.py",
        "scripts/humanego_ict/train_derived_pick_place.sh",
    ]
    if any(not (root / path).is_file() for path in paths):
        return {}
    return {ROUTE: {
        "candidate_operation": OPERATION,
        "execution_status": "SEPARATE_DERIVED_SUPERVISION_TRAINING_CANDIDATE",
        "code_closure": [artifact_ref(root / path) for path in paths],
        "task_packet_required": True, "supervision_class": "DERIVED_RETARGET",
        "control_ground_truth": False, "physical_deployment_authorized": False,
        "current_runnable_operation": None,
        "real_control_route_unchanged": True,
        "qualification_status": "NOT_PASSED",
        "claim_limit": "Derived human-motion training, not measured Revo2 control. Requires qualified causal data, finite training packet, frozen weights/configuration, live writer, GPU lease and storage. CPU engineering is not training or deployment qualification.",
    }}
