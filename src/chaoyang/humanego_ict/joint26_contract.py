"""Additive ICT joint-command candidate contract; no model imports or IO on import.

The matching action interface does not make these ICT weights pi0.5 weights.
Derived IK labels remain research supervision, not measured robot controls.
"""
from __future__ import annotations

from pathlib import Path

ROUTE = "humanego_ict_joint26_v1"
PREPARE_ROUTE = "humanego_ict_joint26_prepare_v1"
PREPARE = "prepare_pick_place_humanego_ict_joint26_v1"
TRAIN = OPERATION = "train_pick_place_humanego_ict_joint26_v1"
PREPARE_MODE = "HUMANEGO_ICT_JOINT26_LABEL_PREPARATION"
EXECUTION_MODE = "HUMANEGO_ICT_JOINT26_TRAINING"
PARALLEL_MODE = "HUMANEGO_ICT_JOINT26_PAIRED_TRAINING"
PARALLEL_ROUTE = "humanego_ict_joint26_parallel_v1"
SCHEMA = "humanego-ict-joint26-paired-manifest-v1"
CONFIG_SCHEMA = "humanego-ict-joint26-training-config-v1"
MODEL_SCHEMA = "humanego-ict-tianji-revo2-joint26-model-v1"
ACTION_LAYOUT = "left_arm7,left_hand6,right_arm7,right_hand6"
HORIZON = 30
ACTION_DIM = 26
ICT_DIM = 29
MAX_ICT = 8
IMAGE_SIZE = (240, 320)
ARM_INDICES = tuple(range(7)) + tuple(range(13, 20))
HAND_INDICES = tuple(range(7, 13)) + tuple(range(20, 26))
HAND_CHANNEL_ORDER = (
    "thumb_bend", "thumb_rotation", "index", "middle", "ring", "pinky",
)


def interface_contract() -> dict:
    """Return a new JSON-compatible declaration, not a deployment permission."""
    return {
        "model_schema": MODEL_SCHEMA,
        "action_layout": ACTION_LAYOUT,
        "action_dim": ACTION_DIM,
        "pred_horizon": HORIZON,
        "ict_dim": ICT_DIM,
        "max_ict": MAX_ICT,
        "image_size": list(IMAGE_SIZE),
        "arm_indices": list(ARM_INDICES),
        "hand_indices": list(HAND_INDICES),
        "arm_command_semantics": "ABSOLUTE_TARGET_JOINT_ANGLE",
        "arm_unit": "radians",
        "hand_command_semantics": "NORMALIZED_0_1",
        "hand_channel_order": list(HAND_CHANNEL_ORDER),
        "state_layout": ACTION_LAYOUT,
        "state_dim": ACTION_DIM,
        "state_validity": "PER_DIMENSION_EXPLICIT_BOOLEAN",
        "state_token_encoding": "PER_SIDE_13_VALUES_AND_13_VALIDITY_BITS",
        "state_token_admission": "ALL_SEVEN_ARM_VALUES_VALID",
        "unknown_current_hand": "ZERO_VALUE_WITH_FALSE_VALIDITY_NOT_OBSERVED",
        "supervision_class": "DERIVED_IK_RESEARCH",
        "control_ground_truth": False,
        "physical_deployment_authorized": False,
        "pi05_weight_compatible": False,
        "horizon_is_not_control_hz": True,
        "training_admission_policies": ["STRICT_MODEL_NUMERICS_V1", "EGO_DERIVED_STRUCTURAL_V1"],
        "ego_training_quality_diagnostics_not_hard_gates": [
            "human_to_robot_fk_fit", "assumed_collision", "teleop_step_speed", "inpainting_quality"],
    }


def candidate_contract(root: Path) -> dict:
    """Register only a complete operation closure, never a partial draft."""
    from chaoyang.governance.common import artifact_ref

    root = Path(root).resolve()
    paths = [
        *[f"src/chaoyang/ops/{name}.py" for name in (PREPARE, TRAIN)],
        "scripts/humanego_ict/joint26_runner_v1.py",
        "scripts/humanego_ict/train_joint26_pick_place.sh",
        "scripts/humanego_ict/check_joint26_env.py",
        "scripts/humanego_ict/setup_joint26_env.sh",
        "scripts/humanego_ict/environment/conda.yml",
        "scripts/humanego_ict/environment/runtime-requirements.txt",
        "scripts/humanego_ict/environment/torch-cu128-requirements.txt",
        "scripts/humanego_ict/joint26_parallel_runner_v1.py",
        *[f"src/chaoyang/humanego_ict/{name}.py" for name in (
            "joint26_contract", "joint26_model", "joint26_data", "joint26_prepare",
            "joint26_training", "joint26_runtime", "joint26_inference",
            "project_mount_paths", "governance_mount_compat", "joint26_cli",
            "joint26_parallel_resources", "joint26_parallel_admission",
            "joint26_console", "joint26_limits",
        )],
        "src/chaoyang/human_ego/training/FlowMatchingModel.py",
        "src/chaoyang/humanego_ict/runtime_gate.py",
        "src/chaoyang/humanego_ict/offline_mink_provider_v1.py",
        "src/chaoyang/humanego_ict/offline_mesh_collision_v1.py",
        # Existing geometry helpers are called, but their optional old model
        # loader / renderer / production-training functions are not selected.
        *[f"src/chaoyang/humanego_ict/{name}.py" for name in (
            "assumed_deployment_v1", "bridge", "inference", "retarget", "data",
        )],
        # Only shared storage, lease and fenced publication helpers are reused.
        "scripts/humanego_ict/production_resume_runner_v1.py",
        "scripts/humanego_ict/production_runner_v1.py",
        "src/chaoyang/governance/common.py",
        "src/chaoyang/governance/validate_governance_state.py",
        "src/chaoyang/governance/rc1_t5_consistency.py",
        "src/chaoyang/governance/inpainting_v1_selection.py",
        "src/chaoyang/inpainting_candidate/publication.py",
        "src/chaoyang/humanego_ict/frozen_payload_entry_cli.py",
        "src/chaoyang/humanego_ict/frozen_payload_dataset.py",
        "src/chaoyang/humanego_ict/production_resume.py",
        "src/chaoyang/human_ego/utils/frozen_contract.py",
        "src/chaoyang/governance/register_single_task_packet.py",
        "src/chaoyang/governance/v52_contracts.py",
        "src/chaoyang/governance/v71_contracts.py",
        "src/chaoyang/governance/current_r3_contracts.py",
        "src/chaoyang/governance/dual_branch_phase_v1.py",
        *[f"src/chaoyang/governance/{name}.py" for name in (
            "current_baseline_v2", "build_algorithm_baseline_v2_status",
            "build_full_pipeline_v3_status", "build_full_pipeline_v4_status",
            "build_human_to_robot_r2_terminal_status", "project_branch_navigation_v1",
            "four_stream_completion",
        )],
        "src/chaoyang/governance/gpu_lease_v71.py",
        "src/chaoyang/ops/run_gpu_command_with_v71_lease.py",
        "src/chaoyang/cli/__init__.py",
        "src/chaoyang/paths.py",
        "src/chaoyang/governance/operation_retirement.py",
        "src/chaoyang/governance/exact78_visual_selection_v1.py",
        *[f"docs/governance/schemas/{name}.schema.json" for name in (
            "authority_index", "artifact_ref", "claim", "task_state",
            "algorithm_contract", "doc_authority_map", "current_status_receipt",
            "gpu_lease_v71", "plan_migration_receipt",
        )],
    ]
    if any(not (root / path).is_file() for path in paths):
        return {}
    closure = [artifact_ref(root / path) for path in paths]
    base = {
        "code_closure": closure,
        "task_packet_required": True,
        "manifest_schema": SCHEMA,
        "config_schema": CONFIG_SCHEMA,
        "interface": interface_contract(),
        "supervision_class": "DERIVED_IK_RESEARCH",
        "control_ground_truth": False,
        "physical_deployment_authorized": False,
        "current_runnable_operation": None,
        "qualification_status": "NOT_PASSED",
        "old_derived_route_unchanged": True,
        "real_control_route_unchanged": True,
        "causal_time_group_window_gates_unchanged": True,
        "ego_training_admission_policy": "EGO_DERIVED_STRUCTURAL_V1",
        "ego_engineering_quality_is_not_training_prerequisite": True,
        "dependency_scope": "SELECTED_JOINT26_CONSUMERS_AND_SHARED_PUBLISHER_CORE",
        "unused_old_model_training_renderer_not_selected": True,
        "geometry_backend_sources": "BOUND_BY_PREPARE_GEOMETRY_INDEX_AND_PROFILE",
        "min_train_source_groups": 16,
        "min_validation_source_groups": 3,
        "claim_limit": (
            "HumanEgo ICT H30x26 source-derived joint candidate only; same control "
            "layout is not pi0.5 weight compatibility, measured controls, metric "
            "calibration or physical approval. Fenced finite tasks are mandatory."
        ),
    }
    return {
        PREPARE_ROUTE: dict(base, candidate_operation=PREPARE, execution_mode=PREPARE_MODE),
        ROUTE: dict(base, candidate_operation=TRAIN, execution_mode=EXECUTION_MODE,
            default_termination_policy="OPTIMIZER_UPDATES_ONLY",
            default_wall_seconds_max=None, optional_wall_budget=True,
            rolling_gpu_authorization_required=True),
        PARALLEL_ROUTE: dict(base, candidate_operation=TRAIN, execution_mode=PARALLEL_MODE,
            coordinator_count=1, training_lanes=["raw", "robotized"],
            gpu_groups_disjoint=True, independent_outputs=True,
            legacy_global_gpu_guard_required=True,
            default_termination_policy="OPTIMIZER_UPDATES_ONLY",
            default_wall_seconds_max=None, optional_wall_budget=True,
            rolling_gpu_authorization_required=True),
    }
