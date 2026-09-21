"""Explicit V2 maintenance/algorithm registration; never promotes quality."""
from pathlib import Path
from chaoyang.governance.common import artifact_ref

TASK_ID = 'four_stream_algorithm_baseline_v2'
FILES = (
    'src/chaoyang/governance/algorithm_baseline_v2_contract.py',
    'src/chaoyang/pipeline/clean_object_guard_v2.py',
    'src/chaoyang/ops/run_clean_object_guard_v2.py',
    'tests/unit/test_clean_object_guard_v2.py',
    'src/chaoyang/pipeline/pico_manus_motion_v2.py',
    'src/chaoyang/ops/run_pico_manus_motion_v2.py',
    'tests/unit/test_pico_manus_motion_v2.py',
    'src/chaoyang/ops/run_pico_manus_provenance_correction_v2.py',
    'tests/unit/test_pico_manus_provenance_correction_v2.py',
    'tests/unit/test_producer_provenance_isolated_v2.py',
    'src/chaoyang/pipeline/huro_core_adapter_v2.py',
    'src/chaoyang/ops/run_huro_core_adapter_v2.py',
    'tests/unit/test_huro_core_adapter_v2.py',
    'src/chaoyang/pipeline/full_robot_review_v2.py',
    'src/chaoyang/ops/run_full_robot_review_v2.py',
    'tests/unit/test_full_robot_review_v2.py',
    'tests/unit/test_full_robot_collision_v2.py',
    'src/chaoyang/ops/run_huro_fixed_placement_core_v2.py',
    'src/chaoyang/ops/run_huro_sessions_core_v2.py',
    'src/chaoyang/pipeline/huro_frozen_targets_v2.py',
    'tests/unit/test_huro_frozen_targets_v2.py',
    'src/chaoyang/ops/run_huro_common_review_v2.py',
    'tests/unit/test_huro_common_review_v2.py',
    'src/chaoyang/ops/run_motion_temporal_audit_v2.py',
    'tests/unit/test_motion_temporal_audit_v2.py',
    'src/chaoyang/governance/build_algorithm_baseline_v2_status.py',
    'tests/unit/test_algorithm_baseline_v2_status.py',
)


def build_v2_contract(repo_root: Path) -> dict:
    packet = repo_root / 'tasks/current' / TASK_ID / 'TASK_PACKET.json'
    if not packet.is_file():
        return {}
    present = [path for path in FILES if (repo_root / path).is_file()]
    return {TASK_ID: {
        'execution_scope': 'FOUR_LANE_OFFLINE_ALGORITHM_ENGINEERING_BASELINE',
        'task_packet': artifact_ref(packet),
        'code_closure': [artifact_ref(repo_root / path) for path in present],
        'implementation_files_not_yet_published': [path for path in FILES if path not in present],
        'registered_operation_policy': 'EXPLICIT_PRESENT_CODE_CLOSURE_ONLY_NO_GLOB_OR_EXTERNAL_IMPORTS',
        'weights': 'NO_NEW_MODEL_DOWNLOADS',
        'dependency_fetch_policy': 'USER_AUTHORIZED_HURO_ISOLATED_ENVIRONMENT_ONLY',
        'gpu_required': 'CONDITIONAL_SINGLE_GOVERNED_LEASE',
        'input_policy': 'EXACT_SESSION_SHA_TIME_EYE_SIDE_VALIDITY_REQUIRED',
        'virtual_installation_policy': 'EXPLICIT_NONMEASURED_NO_PHYSICAL_ACCURACY',
        'authority_separation': ['REPRODUCIBLE_REFERENCE', 'ENGINEERING_FIX_ADOPTED',
                                 'ALGORITHM_IMPROVEMENT_ADOPTED', 'VISUAL_REVIEW_STATUS'],
        'training_eligible': False, 'control_ground_truth': False,
        'physical_deployment_authorized': False,
        'claim_limit': 'Operation registration is not evidence of successful execution, improvement, or full project quality.',
    }}
