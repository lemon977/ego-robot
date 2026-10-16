"""Pure candidate contract builder; registration does not authorize training."""
from __future__ import annotations

from pathlib import Path

from chaoyang.governance.common import artifact_ref
from .runtime_gate import OPERATION, ROUTE


def candidate_contract(repo_root: Path) -> dict:
    operation = repo_root / "src/chaoyang/ops" / (OPERATION + ".py")
    if not operation.is_file():
        return {}
    paths = [
        "src/chaoyang/ops/" + OPERATION + ".py",
        "src/chaoyang/cli/__init__.py",
        "src/chaoyang/humanego_ict/__init__.py", "src/chaoyang/humanego_ict/contracts.py",
        "src/chaoyang/humanego_ict/cli.py", "src/chaoyang/humanego_ict/runtime_gate.py",
        "src/chaoyang/humanego_ict/data.py", "src/chaoyang/humanego_ict/model.py",
        "src/chaoyang/humanego_ict/training.py",
        "src/chaoyang/human_ego/training/FlowMatchingModel.py",
        "src/chaoyang/human_ego/training/checkpoint_transfer.py",
        "src/chaoyang/human_ego/utils/atomic_io.py",
        "src/chaoyang/human_ego/utils/frozen_contract.py",
        "scripts/humanego_ict/train_pick_place.sh",
        "scripts/humanego_ict/preflight_pick_place.sh",
    ]
    return {ROUTE: {
        "execution_status": "CANDIDATE_FENCED_ICT_REVO2_TRAINING_TASK_PACKET_ONLY",
        "candidate_operation": OPERATION, "task_packet_required": True,
        "code_closure": [artifact_ref(repo_root / path) for path in paths],
        "output_schema": "humanego-ict-revo2-resume-v1",
        "target": "REVO2_DUAL_HAND_EEF_POSE_AND_COMMAND6",
        "input_target_qualification_required": True,
        "current_runnable_operation": None, "physical_deployment_authorized": False,
        "training_steps_authorized_without_new_packet": 0,
        "existing_aux_and_baselines_changed": False,
        "claim_limit": "Engineering candidate only. A new finite training packet, qualified causal paired data, real Revo2 supervision and a live GPU lease are mandatory; no checkpoint or deployment qualification.",
    }}
