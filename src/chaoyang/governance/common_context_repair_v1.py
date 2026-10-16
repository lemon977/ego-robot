"""Independent context repair publisher; original common.py stays immutable."""
from __future__ import annotations
import chaoyang.governance.common as _original
for _name,_value in vars(_original).items():
    if _name not in ('__name__','__file__','__loader__','__package__','__spec__','__doc__','__builtins__'):
        globals()[_name]=_value

def publish_bundle(authority: dict[str, Any], task_state: dict[str, Any], *, event_type: str, expected_revision: int | None, generator_path: Path, append_event: bool=True, plan_revision_value: Mapping[str, Any] | None=None, plan_migration_value: Mapping[str, Any] | None=None, task_packet_index_path: Path | None=None, task_packet_index_value: Mapping[str, Any] | None=None) -> dict[str, Any]:
    with governance_lock():
        doc_authority_map_path = AUTHORITY_PATH.parent / 'DOC_AUTHORITY_MAP.json'
        algorithm_contract_path = AUTHORITY_PATH.parent / 'ALGORITHM_CONTRACT.json'
        plan_migration_receipt_path = AUTHORITY_PATH.parent / 'PLAN_MIGRATION_RECEIPT.json'
        task_packet_pointer_path = AUTHORITY_PATH.parent / 'CURRENT_V71_TASK_PACKET_INDEX.json'
        current_revision = 0
        if RECEIPT_PATH.is_file():
            current_revision = int(load_json(RECEIPT_PATH)['governance_revision'])
        if expected_revision is not None and expected_revision != current_revision:
            raise RuntimeError(f'CAS revision mismatch: expected={expected_revision} current={current_revision}')
        revision = current_revision + 1
        generation_id = f'gov-{revision:06d}-{uuid.uuid4().hex[:12]}'
        created = now_iso()
        generator_sha = sha256_file(generator_path)
        repository = git_metadata()
        for value in (authority, task_state):
            value['governance_revision'] = revision
            value['generation_id'] = generation_id
            value['generated_at'] = created
            value['generator_code_sha'] = generator_sha
        authority['repository'] = repository
        task_state['repository'] = repository
        task_state['host'] = socket.gethostname()
        if plan_revision_value is not None:
            plan_revision = dict(plan_revision_value)
            plan_revision.update(governance_revision=revision, generation_id=generation_id, generated_at=created)
            atomic_json(PLAN_REVISION_PATH, plan_revision)
        if plan_migration_value is not None:
            plan_migration = dict(plan_migration_value)
            plan_migration.update(effective_governance_revision=revision, generation_id=generation_id, generated_at=created)
            validate_schema('plan_migration_receipt.schema.json', plan_migration)
            atomic_json(plan_migration_receipt_path, plan_migration)
        if (task_packet_index_path is None) != (task_packet_index_value is None):
            raise RuntimeError('task packet index path and value must be supplied together')
        if task_packet_index_path is not None and task_packet_index_value is not None:
            packet_index = dict(task_packet_index_value)
            packet_index.update(governance_revision=revision, generation_id=generation_id, generated_at=created)
            atomic_json(task_packet_index_path, packet_index)
            index_ref = artifact_ref(task_packet_index_path)
            pointer = {'schema_version': 'chaoyang-v71-task-packet-pointer-v2', 'packet_revision': packet_index['packet_revision'], 'plan_revision': packet_index['plan_revision'], 'execution_revision': packet_index.get('execution_revision', 'R3'), 'governance_revision': revision, 'generation_id': generation_id, 'generated_at': created, 'index_path': str(task_packet_index_path.resolve().relative_to(REPO_ROOT)), 'index_bytes': index_ref['bytes'], 'index_sha256': index_ref['sha256']}
            atomic_json(task_packet_pointer_path, pointer)
        elif task_packet_pointer_path.is_file():
            pointer = load_json(task_packet_pointer_path)
            pointer.update(governance_revision=revision, generation_id=generation_id, generated_at=created)
            atomic_json(task_packet_pointer_path, pointer)
        errors = validate_authority(authority) + validate_task_state(task_state)
        if errors:
            raise RuntimeError('governance validation failed:\n' + '\n'.join(errors))
        from chaoyang.governance.current_baseline_v2 import build_layout, build_registry, build_regression_manifest, render_stage_doc
        baseline_registry = build_registry(authority, task_state)
        file_layout = build_layout(authority, task_state)
        regression_manifest = build_regression_manifest(authority)
        stage_baselines = render_stage_doc(baseline_registry).encode('utf-8')
        markdown = render_status(authority, task_state).encode('utf-8')
        min_status = render_min_status(authority, task_state)
        task_queue = render_task_queue(task_state)
        atomic_json(AUTHORITY_PATH, authority)
        atomic_json(TASK_STATE_PATH, task_state)
        from chaoyang.governance.build_algorithm_baseline_v2_status import publish_navigation_if_v2
        publish_navigation_if_v2(REPO_ROOT, task_state, authority['generated_at'])
        from chaoyang.governance.build_full_pipeline_v3_status import publish_navigation_if_v3
        publish_navigation_if_v3(REPO_ROOT, task_state, authority['generated_at'])
        from chaoyang.governance.build_full_pipeline_v4_status import publish_navigation_if_v4
        publish_navigation_if_v4(REPO_ROOT, task_state, authority['generated_at'])
        from chaoyang.governance.build_human_to_robot_r2_terminal_status import publish_navigation_if_human_to_robot_r2
        publish_navigation_if_human_to_robot_r2(REPO_ROOT, task_state, authority['generated_at'])
        atomic_write(STATUS_PATH, markdown)
        atomic_json(MIN_STATUS_PATH, min_status)
        if RC1_PLAN_PATH.is_file() and RC1_RELEASE_SPEC_PATH.is_file() and RC1_CONTRACT_PATH.is_file():
            atomic_json(RC1_STATUS_MIN_PATH, render_rc1_status_min(authority, task_state))
        atomic_json(TASK_QUEUE_PATH, task_queue)
        atomic_json(BASELINE_REGISTRY_PATH, baseline_registry)
        atomic_write(STAGE_BASELINES_PATH, stage_baselines)
        atomic_json(FILE_LAYOUT_PATH, file_layout)
        atomic_json(REGRESSION_MANIFEST_PATH, regression_manifest)
        from chaoyang.governance.current_context_repair_contracts_v1 import build_algorithm_contract, build_doc_authority_map, validate_doc_authority_map
        algorithm_contract = build_algorithm_contract(baseline_registry, authority)
        validate_schema('algorithm_contract.schema.json', algorithm_contract)
        atomic_json(algorithm_contract_path, algorithm_contract)
        doc_authority_map = build_doc_authority_map(authority, AUTHORITY_PATH.parent, REPO_ROOT)
        validate_schema('doc_authority_map.schema.json', doc_authority_map)
        doc_errors = validate_doc_authority_map(doc_authority_map)
        if doc_errors:
            raise RuntimeError('document authority validation failed:\n' + '\n'.join(doc_errors))
        atomic_json(doc_authority_map_path, doc_authority_map)
        receipt_files = {'authority_index': artifact_ref(AUTHORITY_PATH), 'task_state': artifact_ref(TASK_STATE_PATH), 'project_status': artifact_ref(STATUS_PATH), 'project_status_min': artifact_ref(MIN_STATUS_PATH), 'task_queue': artifact_ref(TASK_QUEUE_PATH), 'baseline_registry_v2': artifact_ref(BASELINE_REGISTRY_PATH), 'stage_baselines': artifact_ref(STAGE_BASELINES_PATH), 'file_layout': artifact_ref(FILE_LAYOUT_PATH), 'regression_manifest': artifact_ref(REGRESSION_MANIFEST_PATH), 'doc_authority_map': artifact_ref(doc_authority_map_path), 'algorithm_contract': artifact_ref(algorithm_contract_path)}
        if PLAN_REVISION_PATH.is_file():
            receipt_files['plan_revision'] = artifact_ref(PLAN_REVISION_PATH)
        if task_packet_pointer_path.is_file():
            receipt_files['task_packet_index_v71'] = artifact_ref(task_packet_pointer_path)
            pointer = load_json(task_packet_pointer_path)
            pointed_index = Path(str(pointer['index_path']))
            if not pointed_index.is_absolute():
                pointed_index = REPO_ROOT / pointed_index
            pointed_ref = artifact_ref(pointed_index)
            if pointer.get('index_bytes', pointed_ref['bytes']) != pointed_ref['bytes']:
                raise RuntimeError('current task packet index pointer bytes mismatch')
            if pointer.get('index_sha256') != pointed_ref['sha256']:
                raise RuntimeError('current task packet index pointer SHA mismatch')
            receipt_files['task_packet_index_current'] = pointed_ref
        if plan_migration_receipt_path.is_file():
            receipt_files['plan_migration_receipt'] = artifact_ref(plan_migration_receipt_path)
        for key, path in (('rc1_plan', RC1_PLAN_PATH), ('rc1_release_spec', RC1_RELEASE_SPEC_PATH), ('visual_aux_rc1_contract', RC1_CONTRACT_PATH), ('rc1_task_dag', RC1_TASK_DAG_PATH), ('rc1_status_min', RC1_STATUS_MIN_PATH)):
            if path.is_file():
                receipt_files[key] = artifact_ref(path)
        receipt = {'schema_version': 'chaoyang-current-status-receipt-v1', 'governance_revision': revision, 'generation_id': generation_id, 'generated_at': created, 'freshness': freshness(task_state), 'files': receipt_files}
        validate_schema('current_status_receipt.schema.json', receipt)
        atomic_json(RECEIPT_PATH, receipt)
        if append_event:
            append_changelog({'schema_version': 'chaoyang-state-change-event-v1', 'created_at': created, 'event_type': event_type, 'governance_revision': revision, 'generation_id': generation_id, 'receipt_sha256': sha256_file(RECEIPT_PATH)})
        return receipt
