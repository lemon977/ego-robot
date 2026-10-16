"""Finite registered CPU validation of sealed ICT predictions; never a model.

No hardware provider is instantiated. Raw prediction files stay immutable and
strict numerical rejection is a diagnostic result, not a runtime success claim
about policy accuracy or robot deployment.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shlex
import stat
import sys

OPERATION = 'audit_pick_place_humanego_ict_offline_envelope_v1'
ROUTE = 'pick_place_humanego_ict_offline_envelope_v1'
MODE = 'HUMANEGO_ICT_OFFLINE_ENVELOPE_FILE_VALIDATION_CPU'
REQUEST_SCHEMA = 'humanego-ict-real20k-offline-sealed-evaluation-bundle-v1'
RESULT_SCHEMA = 'humanego-ict-offline-envelope-file-validation-result-v1'
STATUS = 'COMPLETED_SOFTWARE_FILE_VALIDATION_NOT_DEPLOYMENT'
SOURCE_POLICY = 'FROZEN_ADMITTED_PROJECT_PAYLOAD_SNAPSHOT_RESEARCH_ONLY'
CANDIDATE_SHA = '76b026cc5942e1210b7136b870860e3a7e3b7372bd8f09d17326fa0f61b38354'
CANDIDATE_PATH = '_run/maintenance/ict_inference_deploy_completion_20261012/attempt_0001/lanes/adapter/offline_envelope_candidate.py'


def need(value, code):
    if not value:
        raise ValueError(code)


def project_path(path, root, *, existing=False):
    path = Path(path)
    need(path.is_absolute() and path != root and path.is_relative_to(root)
        and '..' not in path.parts, 'ABSOLUTE_PROJECT_SUBPATH_REQUIRED_BEFORE_IO')
    need(not any(p.is_symlink() for p in (path, *path.parents)), 'SYMLINK_PATH_FORBIDDEN')
    need(path.resolve(strict=existing) == path, 'CANONICAL_PROJECT_PATH_REQUIRED')
    return path


def read_ref(ref, root):
    need(isinstance(ref, dict) and set(ref) == {'path', 'bytes', 'sha256'}
        and type(ref['bytes']) is int and ref['bytes'] >= 0
        and isinstance(ref['sha256'], str) and len(ref['sha256']) == 64
        and all(x in '0123456789abcdef' for x in ref['sha256']), 'EXACT_REFERENCE_REQUIRED')
    p = project_path(ref['path'], root, existing=True)
    before = p.stat()
    need(stat.S_ISREG(before.st_mode), 'REGULAR_FILE_REQUIRED')
    data = p.read_bytes(); after = p.stat()
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    need(identity(before) == identity(after) and len(data) == ref['bytes']
        and hashlib.sha256(data).hexdigest() == ref['sha256'], 'REFERENCE_SHA_BYTES_OR_IDENTITY_DRIFT')
    return data


def unique_json(data):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            need(key not in out, 'DUPLICATE_JSON_KEY')
            out[key] = value
        return out
    def bad(value):
        raise ValueError('NONFINITE_JSON')
    result = json.loads(data, object_pairs_hook=unique, parse_constant=bad)
    need(isinstance(result, dict), 'JSON_OBJECT_REQUIRED')
    return result


def json_ref(ref, root):
    return unique_json(read_ref(ref, root))


def file_ref(path):
    data = path.read_bytes()
    return {'path': str(path), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def write_json(path, value, root):
    path = project_path(path, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    with path.open('xb') as stream:
        stream.write(data)
    ref = file_ref(path)
    need(ref['bytes'] == len(data) and ref['sha256'] == hashlib.sha256(data).hexdigest(),
        'OUTPUT_READBACK_FAILED')
    return ref


def authorize_operation(root, packet_path, arguments, *, request, output):
    """Existing receipt/index/state/fence/ancestry pattern for this exact CPU op."""
    root = Path(root).resolve(strict=True)
    packet_path = project_path(packet_path, root, existing=True)
    request = project_path(request, root, existing=True)
    output = project_path(output, root)
    need(packet_path.is_relative_to(root/'tasks/current'), 'REGISTERED_CPU_PACKET_REQUIRED')
    from chaoyang.humanego_ict.runtime_gate import _is_descendant
    receipt = unique_json((root/'docs/governance/CURRENT_STATUS_RECEIPT.json').read_bytes())
    contract = json_ref(receipt['files']['algorithm_contract'], root)
    route = contract['special_status_contracts'].get(ROUTE, {})
    need(OPERATION in route.get('candidate_operations', []) and route.get('task_packet_required') is True
        and route.get('execution_mode') == MODE and route.get('control_ground_truth') is False
        and route.get('physical_deployment_authorized') is False, 'CPU_ADAPTER_ROUTE_NOT_REGISTERED')
    packet_ref = file_ref(packet_path); packet = json_ref(packet_ref, root)
    expiry = datetime.fromisoformat(packet['deadline_at'])
    need(expiry.tzinfo is not None and expiry > datetime.now(timezone.utc)
        and packet.get('execution_mode') == MODE and packet.get('source_policy') == SOURCE_POLICY
        and packet.get('control_ground_truth') is False
        and packet.get('physical_deployment_authorized') is False, 'FINITE_CPU_TASK_SCOPE_REQUIRED')
    budgets = packet.get('budgets', {})
    need(type(budgets.get('gpu_count')) is int and budgets['gpu_count'] == 0
        and type(budgets.get('model_calls_max')) is int and budgets['model_calls_max'] == 0
        and type(budgets.get('optimizer_updates')) is int and budgets['optimizer_updates'] == 0
        and type(budgets.get('hardware_or_network_calls')) is int and budgets['hardware_or_network_calls'] == 0
        and type(budgets.get('cpu_threads_per_worker')) is int and 1 <= budgets['cpu_threads_per_worker'] <= 2
        and type(budgets.get('wall_seconds_max')) is int and 1 <= budgets['wall_seconds_max'] <= 7200
        and type(budgets.get('samples_max')) is int and budgets['samples_max'] == 114,
        'FINITE_ZERO_MODEL_GPU_CPU_BUDGET_REQUIRED')
    index = json_ref(receipt['files']['task_packet_index_current'], root)
    allowed = [r for r in index['task_packets'] if r.get('execution_allowed') is True]
    need(len(allowed) == 1 and allowed[0]['task_id'] == packet['task_id']
        and allowed[0]['packet_sha256'] == packet_ref['sha256']
        and project_path(root/allowed[0]['packet_path'], root, existing=True) == packet_path,
        'UNIQUE_REGISTERED_CPU_WRITER_REQUIRED')
    state = json_ref(receipt['files']['task_state'], root)
    active = [r for r in state['tasks'] if r.get('status') in {'RUNNING', 'CLAIMED', 'WAIT_GPU_RESOURCE'}]
    writer = packet['writer']
    need(len(active) == 1 and active[0]['task_id'] == packet['task_id']
        and active[0]['status'] == 'RUNNING' and active[0].get('run_signature') == packet['run_signature']
        and active[0].get('pid') == writer['pid'] and active[0].get('proc_start_ticks') == writer['proc_start_ticks']
        and type(writer.get('executor_epoch')) is int and writer['executor_epoch'] > 0
        and isinstance(writer.get('fencing_token'), str) and bool(writer['fencing_token'])
        and writer.get('scope') == packet['task_id'] and _is_descendant(writer), 'LIVE_CPU_WRITER_FENCE_REQUIRED')
    need(unique_json(project_path(packet['writer_file'], root, existing=True).read_bytes()) == writer,
        'CPU_WRITER_IDENTITY_DRIFT')
    need(unique_json((root/'_run/current/GPU_LEASE.json').read_bytes()).get('status') == 'RELEASED',
        'SEALED_MODEL_TASKS_GPU_RELEASE_REQUIRED')
    matched = False
    for command in packet.get('commands', []):
        tokens = shlex.split(command)
        for i in range(len(tokens)-2):
            if tokens[i:i+3] == ['-m', 'chaoyang.cli', 'run']:
                matched |= tokens[i+3:] == [OPERATION, *arguments]
    bundle_ref = packet['inputs'].get('bundle', packet['inputs'].get('request'))
    need(matched and bundle_ref == file_ref(request)
        and project_path(packet['output_dir'], root) == output, 'EXACT_CPU_COMMAND_INPUT_OUTPUT_REQUIRED')
    closure = route.get('code_closure', [])
    need(bool(closure) and all(ref in packet.get('runtime_code', []) for ref in closure),
        'ADOPTED_CPU_CODE_CLOSURE_REQUIRED')
    required_paths = [root/'src/chaoyang/humanego_ict/adapter_file_validation_runtime_v1.py',
        root/'src/chaoyang/humanego_ict/adapter_file_validation_contract_v1.py',
        root/('src/chaoyang/ops/'+OPERATION+'.py'), root/CANDIDATE_PATH]
    need(Path(__file__).resolve() == required_paths[0], 'EXECUTED_ADOPTED_CPU_MODULE_REQUIRED')
    need(all(file_ref(project_path(path,root,existing=True)) in closure for path in required_paths),
        'EXACT_ADOPTED_ADAPTER_FILES_REQUIRED')
    need(file_ref(root/CANDIDATE_PATH)['sha256'] == CANDIDATE_SHA, 'FROZEN_CANDIDATE_SHA_REQUIRED')
    for ref in packet['runtime_code']:
        read_ref(ref, root)
    need(datetime.now(timezone.utc) < expiry, 'CPU_AUTHORIZATION_EXPIRED_DURING_VERIFICATION')
    packet['admitted_context'] = {'root': str(root), 'packet_path': str(packet_path),
        'arguments': list(arguments), 'request': str(request), 'output': str(output), 'bundle_ref':bundle_ref}
    return packet


def load_candidate(reference, root):
    need(reference['sha256'] == CANDIDATE_SHA, 'FROZEN_ENVELOPE_CANDIDATE_SHA_REQUIRED')
    source = read_ref(reference, root)
    spec = importlib.util.spec_from_file_location('_frozen_ict_offline_envelope', reference['path'])
    module = importlib.util.module_from_spec(spec)
    exec(compile(source, reference['path'], 'exec'), module.__dict__)
    read_ref(reference, root)
    return module


def validate_request(request, root, *, fixture_only=False):
    required = {'schema','source_policy','evaluation_targets_ref','branches',
        'control_ground_truth','physical_deployment_authorized'}
    need(required <= set(request) and request['schema'] == REQUEST_SCHEMA
        and request['source_policy'] == SOURCE_POLICY
        and request['control_ground_truth'] is False and request['physical_deployment_authorized'] is False
        and (request.get('fixture_only') is True) == fixture_only, 'EXACT_ADAPTER_FILE_REQUEST_REQUIRED')
    need(set(request['branches']) == {'raw', 'robotized'}, 'TWO_BRANCHES_REQUIRED')
    # Only hash this JSON index; adapter never decodes its future target members.
    read_ref(request['evaluation_targets_ref'], root)
    candidate_ref = file_ref(project_path(root/CANDIDATE_PATH, root, existing=True))
    need(candidate_ref['sha256'] == CANDIDATE_SHA, 'FROZEN_CANDIDATE_SHA_REQUIRED')
    admitted = {}
    for branch, refs in request['branches'].items():
        need(set(refs) == {'task_packet_ref', 'outer_result_ref', 'post_validation_ref',
            'inner_result_ref','current_inputs_outer_result_ref'},
            'EXACT_SEALED_BRANCH_REFS_REQUIRED')
        task, outer, post, result = [json_ref(refs[k], root) for k in
            ('task_packet_ref', 'outer_result_ref', 'post_validation_ref', 'inner_result_ref')]
        need(outer.get('schema') == 'humanego-finite-offline-publisher-result-v1'
            and outer.get('task_terminal_status') == 'PASSED' and outer.get('child_exit_code') == 0
            and outer.get('errors') == [] and outer.get('stage_result_ref') == refs['inner_result_ref']
            and outer.get('task_id') == task['task_id'] and outer.get('run_signature') == task['run_signature'],
            'NORMAL_SEALED_BRANCH_PARENT_REQUIRED')
        need(post.get('exit_code') == 0 and unique_json(post['stdout'].encode()).get('status') == 'PASS',
            'SEALED_BRANCH_POST_PASS_REQUIRED')
        need(result.get('schema') == 'humanego-ict-revo2-offline-evaluation-result-v1'
            and result.get('status') == 'COMPLETED_REAL_MODEL_OFFLINE_DIAGNOSTICS_NOT_DEPLOYMENT'
            and result.get('task_id') == task['task_id'] and result.get('run_signature') == task['run_signature']
            and task.get('branch') == result.get('branch') == branch
            and result.get('source_policy') == task.get('source_policy') == SOURCE_POLICY
            and result.get('control_ground_truth') is False and result.get('physical_deployment_authorized') is False
            and result.get('future_labels_provided_to_model') is False and result.get('clipping_applied') is False
            and (result.get('fixture_only') is True) == fixture_only, 'EXACT_FROZEN_INFERENCE_RESULT_REQUIRED')
        request_ref = result['request']; upstream = json_ref(request_ref, root)
        need(task['inputs']['request'] == request_ref and upstream['branch'] == branch
            and upstream['inference_contract'] == result['inference_contract'], 'INFERENCE_REQUEST_CONTRACT_BINDING_REQUIRED')
        current_outer = json_ref(refs['current_inputs_outer_result_ref'], root)
        need(task['inputs']['current_inputs_sealed_result'] == refs['current_inputs_outer_result_ref']
            and current_outer.get('task_terminal_status') == 'PASSED' and current_outer.get('child_exit_code') == 0
            and current_outer.get('errors') == [], 'NORMAL_SEALED_CURRENT_INPUTS_REQUIRED')
        need(upstream.get('schema') == 'humanego-ict-revo2-offline-evaluation-batch-request-v1'
            and upstream.get('source_policy') == SOURCE_POLICY
            and upstream.get('sampling') == result.get('sampling') == {'method':'euler','steps':20,'seed':7},
            'FROZEN_EULER20_REQUEST_REQUIRED')
        contract = json_ref(result['inference_contract'], root)
        need(contract.get('state_dict_key') == 'model'
            and contract['training_provenance']['branch'] == branch
            and contract['training_provenance']['optimizer_updates'] == 20000, 'ACTUAL20K_SAME_BRANCH_MODEL_REQUIRED')
        samples = result['samples']; original = upstream['samples']
        need(len(samples) == len(original) == result['samples_completed']
            and (fixture_only or (len(samples) == 57 and result['groups_completed'] == 20))
            and result['model_calls'] == 20*len(samples), 'COMPLETE57_REAL_INFERENCE_REQUIRED')
        ids = []
        for sample, source in zip(samples, original):
            need(all(sample[k] == source[k] for k in
                ('sample_id', 'group_id', 'current_input_npz', 'input_receipt'))
                and sample.get('model_calls') == 20, 'EXACT_SAMPLE_ORDER_AND_SOURCE_REQUIRED')
            ids.append((sample['sample_id'], sample['group_id'], sample['source_frame_id'], sample['anchor_qpc_ns']))
            for key in ('current_input_npz', 'input_receipt', 'raw_actions_ref'):
                read_ref(sample[key], root)
        need(len(set(ids)) == len(ids), 'UNIQUE_SAMPLE_ANCHORS_REQUIRED')
        admitted[branch] = {'result': result, 'task': task, 'contract_ref': result['inference_contract'], 'ids': ids}
    need(admitted['raw']['ids'] == admitted['robotized']['ids'], 'EXACT_PAIRED_VALIDATION_ANCHORS_REQUIRED')
    return admitted


def _execute(request, packet, root, output, recheck, *, fixture_only=False):
    admitted = validate_request(request, root, fixture_only=fixture_only)
    recheck()
    write_json(output/'ADAPTER_VALIDATION_RESERVED.json', {'task_id':packet['task_id'],
        'run_signature':packet['run_signature'], 'status':'RESERVED_NOT_COMPLETE', 'fixture_only':fixture_only}, root)
    candidate = load_candidate(file_ref(root/CANDIDATE_PATH), root)
    import numpy as np
    results = []
    for branch in ('raw', 'robotized'):
        entry = admitted[branch]
        for index, sample in enumerate(entry['result']['samples']):
            recheck()
            # Full SHA validation occurs before using current or prediction bytes.
            for key in ('current_input_npz', 'input_receipt'):
                read_ref(sample[key], root)
            raw_bytes = read_ref(sample['raw_actions_ref'], root)
            with np.load(io.BytesIO(raw_bytes), allow_pickle=False) as archive:
                need(archive.files == ['actions_50x30'], 'EXACT_RAW_H50_ARCHIVE_REQUIRED')
                raw = archive['actions_50x30'].copy()
            row = {'branch':branch, 'sample_id':sample['sample_id'], 'group_id':sample['group_id'],
                'source_frame_id':sample['source_frame_id'], 'anchor_qpc_ns':sample['anchor_qpc_ns'],
                'raw_actions_ref':sample['raw_actions_ref'], 'input_ref':sample['current_input_npz'],
                'receipt_ref':sample['input_receipt'], 'request_prepared':False,
                'numeric_accepted':False, 'errors':[], 'ik_status':'NO_PROVIDER_NOT_INVOKED',
                'device_ack_status':'UNKNOWN', 'physical_deployment_authorized':False}
            try:
                prepared = candidate.prepare_offline_request(project_root=root,
                    contract_ref=entry['contract_ref'], current_input_ref=sample['current_input_npz'],
                    current_receipt_ref=sample['input_receipt'], expected_branch=branch,
                    sequence=index+1, policy_id='humanego_ict_'+branch+'_20000_research',
                    allow_synthetic_fixture=fixture_only)
                row['request_prepared'] = True
                q = prepared['request']
                need(q['source_frame_id'] == row['source_frame_id'] and q['anchor_qpc_ns'] == row['anchor_qpc_ns']
                    and q['sample_id'] == row['sample_id'] and q['source_group_id'] == row['group_id'],
                    'ADAPTER_CURRENT_INFERENCE_IDENTITY_DRIFT')
                response = candidate.bind_offline_response(prepared, raw,
                    request_signature=q['request_signature'], checkpoint_sha256=q['checkpoint_ref']['sha256'],
                    sequence=index+1, policy_id=q['policy_id'])
                need(np.array_equal(response['raw_actions'], raw, equal_nan=True), 'RAW_PREDICTION_CHANGED')
                row.update(request_signature=q['request_signature'], request_context=q['request_context'],
                    checkpoint_ref=q['checkpoint_ref'], response_status=response['status'],
                    numeric_accepted=response['numeric_accepted'], errors=response['errors'],
                    raw_actions_sha256=response['raw_actions_sha256'])
                if raw.shape == (50,30) and np.isfinite(raw).all():
                    row.update(hand_min=float(raw[:,18:].min()), hand_max=float(raw[:,18:].max()),
                        hand_out_of_range_values=int(((raw[:,18:]<0)|(raw[:,18:]>1)).sum()))
                need(row['numeric_accepted'] == sample['qualified_for_assumed_bridge'],
                    'FROZEN_STRICT_REJECT_OUTCOME_DRIFT')
            except (ValueError, TypeError, KeyError, OverflowError) as error:
                row['errors'] = [str(error)]
                row['response_status'] = 'FAILED_ADAPTER_SOFTWARE_VALIDATION'
                row['numeric_accepted'] = False
            results.append(row)
            write_json(output/(branch+f'_sample_{index:03d}_VALIDATION.json'), row, root)
    recheck()
    # Final direct prediction/current refs are revalidated; NAS provenance is not traversed.
    validate_request(request, root, fixture_only=fixture_only)
    recheck()
    return {'schema':RESULT_SCHEMA, 'status':STATUS, 'task_id':packet['task_id'],
        'run_signature':packet['run_signature'], 'source_policy':SOURCE_POLICY,
        'bundle_ref':packet['inputs'].get('bundle', packet['inputs'].get('request')), 'branches':request['branches'],
        'samples':results, 'samples_checked':len(results), 'request_prepared':sum(r['request_prepared'] for r in results),
        'strict_decode_passed':sum(r['numeric_accepted'] for r in results),
        'strict_decode_failed':sum(r['request_prepared'] and not r['numeric_accepted'] for r in results),
        'software_validation_failed':sum(r['response_status']=='FAILED_ADAPTER_SOFTWARE_VALIDATION' for r in results),
        'ik_no_provider':len(results), 'initial_q_fabricated':False, 'clipping_applied':False,
        'checkpoint_model_calls':0, 'gpu_calls':0, 'optimizer_updates':0, 'robot_commands_sent':0,
        'hardware_or_network_calls':0, 'source_replay_status':'SOURCE_REPLAY_NOT_PERFORMED',
        'control_ground_truth':False, 'physical_deployment_authorized':False, 'fixture_only':fixture_only}


def execute_request(request, packet):
    context = packet['admitted_context']
    root, output = Path(context['root']), Path(context['output'])
    def recheck():
        current = authorize_operation(root, context['packet_path'], context['arguments'],
            request=context['request'], output=output)
        need(current['run_signature'] == packet['run_signature'] and current['writer'] == packet['writer'],
            'CPU_TASK_FENCE_CHANGED')
    recheck()
    return _execute(request, packet, root, output, recheck)


def execute_synthetic_fixture_request(request, *, root, output):
    """Explicit fixture-only API, no real Task or admission claimed."""
    packet = {'task_id':'synthetic_adapter_fixture', 'run_signature':'fixture', 'inputs':{'request':{}}}
    return _execute(request, packet, Path(root), Path(output), lambda:None, fixture_only=True)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-packet', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    from chaoyang.governance.common import REPO_ROOT
    try:
        packet = authorize_operation(REPO_ROOT, args.task_packet, argv, request=args.bundle, output=args.output_dir)
        result = execute_request(unique_json(read_ref(packet['admitted_context']['bundle_ref'], REPO_ROOT)), packet)
        write_json(args.output_dir/'RESULT.json', result, REPO_ROOT)
        print(json.dumps({'status':result['status'],'samples_checked':result['samples_checked'],
            'strict_decode_failed':result['strict_decode_failed'],'model_calls':0}))
        # Expected numerical REJECT remains a completed diagnostic. Any unexpected
        # software/identity error is evidence-bearing failure, never outer PASSED.
        return 2 if result['software_validation_failed'] else 0
    except (ValueError, TypeError, KeyError, OSError, RuntimeError) as error:
        print(json.dumps({'status':'BLOCKED','failure_code':str(error),'model_calls':0,
            'physical_deployment_authorized':False}), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
