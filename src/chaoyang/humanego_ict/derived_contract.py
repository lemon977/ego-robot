"""Separate derived-label and offline inference candidates, not baseline adoption."""
from __future__ import annotations

from pathlib import Path

from chaoyang.governance.common import artifact_ref

ROUTE = "pick_place_humanego_ict_derived_revo2_v1"
PREPARE = "prepare_pick_place_humanego_ict_derived_v1"
INFER = "infer_pick_place_humanego_ict_revo2_offline_v1"


def candidate_contract(root: Path) -> dict:
    paths = [
        "src/chaoyang/ops/" + PREPARE + ".py",
        "src/chaoyang/ops/" + INFER + ".py",
        "src/chaoyang/humanego_ict/derived_contract.py",
        "src/chaoyang/humanego_ict/derived_runtime.py",
        "src/chaoyang/humanego_ict/derived_data.py",
        "src/chaoyang/humanego_ict/retarget.py",
        "src/chaoyang/humanego_ict/bridge.py",
        "src/chaoyang/humanego_ict/inference.py",
        "src/chaoyang/humanego_ict/runtime_gate.py",
        "src/chaoyang/humanego_ict/data.py",
        "src/chaoyang/humanego_ict/model.py",
        "src/chaoyang/cli/__init__.py",
        "src/chaoyang/human_ego/training/FlowMatchingModel.py",
    ]
    # An incomplete work-in-progress is not a maintained operation.
    if any(not (root / path).is_file() for path in paths):
        return {}
    return {ROUTE: {
        "execution_status": "SEPARATE_FENCED_DERIVED_LABEL_AND_OFFLINE_INFERENCE_CANDIDATES",
        "candidate_operations": [PREPARE, INFER], "task_packet_required": True,
        "code_closure": [artifact_ref(root / path) for path in paths],
        "supervision_class": "DERIVED_RETARGET", "control_ground_truth": False,
        "physical_deployment_authorized": False, "current_runnable_operation": None,
        "training_steps_authorized_without_new_packet": 0,
        "real_control_route_unchanged": True, "causal_rgb_gate_unchanged": True,
        "claim_limit": "Derived human-motion targets are not measured Revo2 controls. Candidate preparation and offline inference require separate finite packets. Noncausal Clean remains visualization-only; models, calibration, timing and physical deployment need independent qualification.",
    }}
