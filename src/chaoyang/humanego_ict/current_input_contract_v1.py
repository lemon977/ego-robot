"""Finite CPU current-only producer, separate from model or physical adoption."""
from pathlib import Path
from chaoyang.governance.common import artifact_ref
from chaoyang.humanego_ict.derived_contract import candidate_contract as old_contract, ROUTE as OLD_ROUTE

OPERATION = 'prepare_pick_place_humanego_ict_current_inputs_v1'
ROUTE = 'humanego_ict_current_inputs_v1'
BUILDER = '_run/maintenance/humanego_ict_evaluation_engineering_20261011/attempt_0001/prepared_code/evaluation_inputs_v1.py'
BUILDER_SHA = '4413542df6f70aed16e6e1f9b49319d6a695fce22ba9f38ddd0bbedae694332e'


def candidate_contract(root: Path) -> dict:
    paths = ['src/chaoyang/ops/' + OPERATION + '.py',
        'src/chaoyang/humanego_ict/current_input_contract_v1.py',
        'src/chaoyang/humanego_ict/evaluation_current_producer_v1.py', BUILDER]
    previous = old_contract(root).get(OLD_ROUTE)
    if previous is None or any(not (root / p).is_file() for p in paths):
        return {}
    builder = artifact_ref(root / BUILDER)
    if builder['sha256'] != BUILDER_SHA:
        raise RuntimeError('FROZEN_CURRENT_INPUT_BUILDER_SHA_DRIFT')
    return {ROUTE: {'candidate_operation': OPERATION, 'task_packet_required': True,
        'code_closure': previous['code_closure'] + [artifact_ref(root / p) for p in paths],
        'execution_mode': 'HUMANEGO_ICT_CURRENT_INPUT_PRODUCTION_CPU',
        'source_policy': 'FROZEN_ADMITTED_PROJECT_PAYLOAD_SNAPSHOT_RESEARCH_ONLY',
        'control_ground_truth': False, 'physical_deployment_authorized': False,
        'current_runnable_operation': None, 'model_calls_authorized': 0, 'gpu_count': 0,
        'claim_limit': 'Frozen local 57/20 research validation current-input production only; no future targets provided to model, source NAS replay or deployment promotion.'}}
