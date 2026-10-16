"""Full-state frozen-payload adapter; no model/optimizer/GPU execution authority.

Normal parents retain the original six-reference schema and implementation.
Periodic parents use an independent five-reference FAILED-infrastructure gate,
never a fabricated normal inner RESULT. Caller still owes current route/source,
fresh owner, current GPU lease and adoption checks before any production use.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
from types import SimpleNamespace

from chaoyang.governance.common import REPO_ROOT as PROJECT_ROOT
MAX_CHECKPOINT_BYTES = 512 * 1024 * 1024
MAX_STATE_BYTES = 384 * 1024 * 1024
MAX_STATE_NODES = 20000
PERIODIC_ROLES = frozenset({'task_packet', 'outer_result', 'training_config', 'checkpoint', 'checkpoint_sidecar'})
PERIODIC_ROLE = 'PERIODIC_NOT_NORMAL_PASSED_PARENT'
RECONCILIATION_SCHEMA = 'frozen-payload-periodic-reconciliation-v1'
AUDIT_SCHEMA = 'frozen-payload-complete-state-audit-v1'
RELEASE_SCHEMA = 'frozen-payload-parent-release-v1'
PERIODIC_CONFIG_SCHEMA = 'pick-place-humanego-ict-frozen-payload-periodic-config-v1'
PERIODIC_INITIALIZATION = 'RESUME_VERIFIED_PERIODIC_COMPLETE_STATE'
PERIODIC_RECEIPT_SCHEMA = 'pick-place-humanego-ict-frozen-payload-periodic-adaptation-v1'
SOURCE_POLICY = 'FROZEN_ADMITTED_PROJECT_PAYLOAD_SNAPSHOT_RESEARCH_ONLY'
FROZEN_EXECUTION_MODE = 'HUMANEGO_ICT_FROZEN_PAYLOAD_RESEARCH_RESUME'
STOP_SCHEMA = 'frozen-payload-infrastructure-stop-v1'
SIGNED_DEPENDENCIES = (
    ('src/chaoyang/humanego_ict/production_resume.py', 23150, '375f0dd4957941860a1d8c02601dbbd7e760645e74afb4a83b868676f169b472'),
    ('src/chaoyang/human_ego/utils/frozen_contract.py', 68491, '12d1ba91557e284c1fb804b783a7cebb62aa6ad8ac26480ffcac43c06f466cc6'),
    ('src/chaoyang/humanego_ict/training.py', 24921, '45834917170cb3073c8269d13e9fb10da97fdfd4da75a76d851e9f6293ff19e2'),
    ('src/chaoyang/governance/common.py', 33846, '34ae41791774496e275943431d2973f78daa939108054b86496cf7f5d1dbf0cf'),
)


class StateRejected(ValueError):
    pass


def _require(condition, code):
    if not condition:
        raise StateRejected(code)


def _legacy():
    for relative, size, sha in SIGNED_DEPENDENCIES:
        path = PROJECT_ROOT / relative
        _require(path.resolve(strict=True) == path and not any(p.is_symlink() for p in (path, *path.parents)),
                 'SIGNED_DEPENDENCY_PATH_DRIFT')
        data = path.read_bytes()
        _require(len(data) == size and hashlib.sha256(data).hexdigest() == sha, 'SIGNED_DEPENDENCY_SHA_DRIFT')
    from chaoyang.humanego_ict import production_resume
    _require(Path(production_resume.__file__).resolve() == PROJECT_ROOT / SIGNED_DEPENDENCIES[0][0],
             'WRONG_LEGACY_IMPORT_PATH')
    return production_resume


def trusted_cpu_state(state):
    """Reject reducers/hidden attrs/cycles before copying or saving any state."""
    import numpy as np
    import torch
    total, nodes, active = 0, 0, set()

    def visit(value):
        nonlocal total, nodes
        nodes += 1
        _require(nodes <= MAX_STATE_NODES, 'STATE_NODE_BOUND')
        kind = type(value)
        if kind is torch.Tensor:
            _require(not vars(value) and value.device.type == 'cpu' and value.layout == torch.strided,
                     'EXACT_CPU_TENSOR_WITHOUT_ATTRS_REQUIRED')
            total += value.numel() * value.element_size()
            _require(not (value.is_floating_point() or value.is_complex())
                     or bool(torch.isfinite(value).all().item()), 'NONFINITE_STATE_TENSOR')
        elif kind is np.ndarray:
            _require(not value.dtype.hasobject and value.dtype.metadata is None and value.dtype.fields is None
                     and value.dtype.subdtype is None and value.dtype.kind in 'buifc', 'UNTRUSTED_NUMPY_DTYPE')
            total += value.nbytes
            _require(bool(np.isfinite(value).all()), 'NONFINITE_STATE_ARRAY')
        elif kind in (dict, OrderedDict, list, tuple):
            _require(id(value) not in active, 'CYCLIC_STATE_REJECTED')
            active.add(id(value))
            if kind is OrderedDict:
                attrs = vars(value)
                _require(all(type(k) is str and k == '_metadata' for k in attrs), 'ORDEREDDICT_EXTRA_ATTRS')
                if '_metadata' in attrs:
                    _require(type(attrs['_metadata']) in (dict, OrderedDict), 'UNTRUSTED_ORDEREDDICT_METADATA')
                    visit(attrs['_metadata'])
            if kind in (dict, OrderedDict):
                _require(all(type(k) in (str, int) for k in value), 'UNTRUSTED_MAPPING_KEYS')
            for child in (value.values() if kind in (dict, OrderedDict) else value):
                visit(child)
            active.remove(id(value))
        elif kind is float:
            _require(math.isfinite(value), 'NONFINITE_STATE_METADATA')
        elif kind in (str, int, bool) or value is None:
            pass
        elif kind is torch.torch_version.TorchVersion:
            _require(not getattr(value, '__dict__', {}), 'TORCH_VERSION_EXTRA_ATTRS')
        else:
            raise StateRejected('UNTRUSTED_STATE_CLASS')
        _require(total <= MAX_STATE_BYTES, 'STATE_TENSOR_BYTE_BOUND')

    visit(state)
    return total


def complete_equal(a, b, legacy=None):
    legacy = legacy or _legacy()
    if not legacy._identical(a, b) or type(a) is not type(b):
        return False
    if type(a) is OrderedDict and not complete_equal(vars(a), vars(b), legacy):
        return False
    if type(a) in (dict, OrderedDict):
        return all(complete_equal(a[k], b[k], legacy) for k in a)
    if type(a) in (tuple, list):
        return all(complete_equal(x, y, legacy) for x, y in zip(a, b))
    return True


def _json_context(value):
    _require(type(value) is dict, 'EXACT_JSON_MAPPING_REQUIRED')
    trusted_cpu_state(value)
    json.dumps(value, allow_nan=False)


def _bounded_load(reference, root, legacy):
    _require(type(reference) is dict and set(reference) == {'path', 'sha256', 'bytes'}
             and type(reference['bytes']) is int and 0 < reference['bytes'] <= MAX_CHECKPOINT_BYTES,
             'CHECKPOINT_FILE_BYTE_BOUND')
    state = legacy._safe_load_checkpoint(reference, root)
    trusted_cpu_state(state)
    return state


def _adam_coverage(state):
    import torch
    optimizer = state['optimizer']
    ids = []
    for group in optimizer['param_groups']:
        _require(type(group) is dict and type(group.get('params')) is list and group['params']
                 and all(type(i) is int and i >= 0 for i in group['params']), 'ADAM_GROUP_PARAMETER_IDS')
        ids.extend(group['params'])
    _require(len(ids) == len(set(ids)) and set(ids) == set(optimizer['state']), 'ADAM_COMPLETE_PARAMETER_COVERAGE')
    for entry in optimizer['state'].values():
        a, b = entry['exp_avg'], entry['exp_avg_sq']
        _require(type(a) is torch.Tensor and type(b) is torch.Tensor and a.shape == b.shape and a.dtype == b.dtype,
                 'ADAM_MATCHED_MOMENT_TENSORS')
        if 'max_exp_avg_sq' in entry:
            c = entry['max_exp_avg_sq']
            _require(type(c) is torch.Tensor and c.shape == a.shape and c.dtype == a.dtype, 'ADAM_MAX_MOMENT_SHAPE')


def _check_checkpoint_writer(state):
    """Trainer worker identity is distinct from the Task publisher identity."""
    writer = state['writer']
    _require(type(writer) is dict and set(writer) == {'pid', 'namespace_pid', 'writer_rank', 'executor_epoch',
             'proc_start_ticks', 'run_signature', 'fencing_token'}
             and all(type(writer[k]) is int and writer[k] > 0 for k in ('pid', 'namespace_pid', 'executor_epoch', 'proc_start_ticks'))
             and type(writer['writer_rank']) is int and writer['writer_rank'] == 0
             and type(writer['fencing_token']) is str and len(writer['fencing_token']) == 32
             and all(c in '0123456789abcdef' for c in writer['fencing_token'])
             and writer['run_signature'] == hashlib.sha256(json.dumps(state['binding'], sort_keys=True).encode()).hexdigest(),
             'EXACT_ORIGINAL_WORKER_BINDING_WRITER_REQUIRED')
    return writer


def _proc_identity(pid):
    try:
        raw = (Path('/proc') / str(pid) / 'stat').read_text()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise StateRejected('OLD_PROCESS_STATE_UNKNOWN') from exc
    fields = raw[raw.rfind(')') + 2:].split()
    _require(len(fields) > 19, 'OLD_PROCESS_STAT_MALFORMED')
    return {'pid': pid, 'state': fields[0], 'pgid': int(fields[2]), 'proc_start_ticks': int(fields[19])}


def _group_members(groups):
    members = []
    try:
        for path in Path('/proc').iterdir():
            if path.name.isdigit():
                identity = _proc_identity(int(path.name))
                if identity and identity['pgid'] in groups:
                    members.append(identity)
    except OSError as exc:
        raise StateRejected('OLD_GROUP_STATE_UNKNOWN') from exc
    return members


def _old_processes_gone(reconciliation, packet):
    processes = reconciliation.get('owned_processes')
    groups = reconciliation.get('owned_process_groups')
    _require(type(processes) is list and len(processes) >= 2 and type(groups) is list and groups
             and all(type(g) is int and g > 0 for g in groups) and len(set(groups)) == len(groups),
             'EXACT_OLD_WRITER_GROUP_EVIDENCE_REQUIRED')
    identities = set()
    for item in processes:
        _require(type(item) is dict and set(item) == {'pid', 'proc_start_ticks', 'pgid'}
                 and all(type(item[k]) is int and item[k] > 0 for k in item)
                 and item['pgid'] in groups, 'EXACT_OLD_PROCESS_IDENTITY_REQUIRED')
        key = (item['pid'], item['proc_start_ticks'])
        _require(key not in identities, 'DUPLICATE_OLD_PROCESS_IDENTITY')
        identities.add(key)
        live = _proc_identity(item['pid'])
        _require(live is None or live['proc_start_ticks'] != item['proc_start_ticks'], 'OLD_PROCESS_STILL_PRESENT_INCLUDING_D_OR_Z')
    _require((packet['writer']['pid'], packet['writer']['proc_start_ticks']) in identities,
             'ORIGINAL_PUBLISHER_IDENTITY_REQUIRED')
    worker = reconciliation.get('checkpoint_writer')
    _require(type(worker) is dict and (worker.get('pid'), worker.get('proc_start_ticks')) in identities,
             'ORIGINAL_CHECKPOINT_WORKER_IDENTITY_REQUIRED')
    _require(not _group_members(set(groups)), 'OLD_OWNED_GROUP_STILL_PRESENT')


def _failure_codes(outer):
    fields = [k for k in ('failure_codes', 'errors') if k in outer]
    _require(fields and all(type(outer[k]) is list and all(type(v) is str for v in outer[k]) for k in fields),
             'ACTUAL_FAILURE_FIELD_REQUIRED_NO_DEFAULT_PASS')
    _require(len(fields) == 1 or outer['failure_codes'] == outer['errors'], 'CONFLICTING_FAILURE_FIELD_ALIASES')
    key = fields[0]
    return key, outer[key]


def _validate_infrastructure_stop(reference, packet, reconciliation, root, legacy):
    stop = legacy._read_json(reference, root, 'sealed owned infrastructure stop')
    _json_context(stop)
    _require(stop.get('schema') == STOP_SCHEMA and stop.get('status') == 'SEALED_OWNED_INFRASTRUCTURE_STOP_REAPED'
             and stop.get('reason') == 'INFRASTRUCTURE_NAS_STALL_STOP'
             and stop.get('task_id') == packet['task_id'] and stop.get('run_signature') == packet['run_signature']
             and stop.get('parent_task_packet_ref') == reconciliation['parent_refs']['task_packet']
             and stop.get('publisher') == packet['writer']
             and stop.get('checkpoint_writer') == reconciliation['checkpoint_writer']
             and stop.get('parent_status_preserved') == reconciliation['parent_status_preserved']
             and stop.get('parent_failure_field') == reconciliation['parent_failure_field']
             and stop.get(reconciliation['parent_failure_field']) == reconciliation[reconciliation['parent_failure_field']]
             and stop.get('child_exit_code_preserved') == reconciliation['child_exit_code']
             and stop.get('issued_signal') == 15 and type(stop['issued_signal']) is int
             and stop.get('signal_target') == {'pid': packet['writer']['pid'], 'proc_start_ticks': packet['writer']['proc_start_ticks']}
             and stop.get('ordinary_user_pause') is False and stop.get('quality_failure') is False
             and stop.get('publisher_reaped') is True and stop.get('worker_reaped') is True and stop.get('unknowns') == [],
             'EXPLICIT_OWNED_NAS_STALL_STOP_RECEIPT_REQUIRED')
    controller = stop.get('controller_handoff')
    _require(type(controller) is dict and controller.get('status_before') == 'WAIT_INITIAL_PUBLISHER'
             and type(controller.get('submitted_successors')) is int and controller['submitted_successors'] == 0
             and controller.get('reaped') is True and type(controller.get('pid')) is int and controller['pid'] > 0
             and type(controller.get('proc_start_ticks')) is int and controller['proc_start_ticks'] > 0,
             'ROOT_CONTROLLER_ZERO_DISPATCH_REAPED_HANDOFF_REQUIRED')
    live = _proc_identity(controller['pid'])
    _require(live is None or live['proc_start_ticks'] != controller['proc_start_ticks'], 'OLD_CONTROLLER_NOT_REAPED')
    stall = legacy._read_json(stop['long_stall_evidence_ref'], root, 'frozen long infrastructure stall evidence')
    _json_context(stall)
    _require(stall.get('schema') == 'frozen-payload-long-io-stall-v1' and stall.get('status') == 'LONG_STALL_CONFIRMED'
             and stall.get('task_id') == packet['task_id'] and stall.get('run_signature') == packet['run_signature']
             and stall.get('checkpoint_writer') == reconciliation['checkpoint_writer']
             and stall.get('blocked_state') == 'D' and stall.get('infrastructure') == 'NFS_METADATA_WAIT_WITH_EXACT_TARGET_UNKNOWN'
             and stall.get('exact_target_file') == 'UNKNOWN'
             and type(stall.get('observations')) is list and len(stall['observations']) >= 2
             and type(stall.get('duration_seconds')) in (int, float) and stall['duration_seconds'] > 0
             and stall.get('logged_optimizer_progress_unchanged') is True
             and stall.get('unlogged_optimizer_updates') == 'UNKNOWN'
             and stall.get('unknowns') == [], 'PINNED_LONG_IO_STALL_NOT_PAUSE_REQUIRED')
    times = []
    for observation in stall['observations']:
        _require(type(observation) is dict and observation.get('worker_pid') == reconciliation['checkpoint_writer']['pid']
                 and observation.get('proc_start_ticks') == reconciliation['checkpoint_writer']['proc_start_ticks']
                 and observation.get('state') == 'D' and type(observation.get('logged_optimizer_updates')) is int,
                 'EXACT_STALL_PROCESS_PROGRESS_OBSERVATIONS_REQUIRED')
        timestamp = datetime.fromisoformat(observation['observed_at'].replace('Z', '+00:00'))
        _require(timestamp.tzinfo is not None, 'STALLED_OBSERVATION_UTC_REQUIRED')
        times.append(timestamp)
    _require(all(b > a for a, b in zip(times, times[1:]))
             and (times[-1] - times[0]).total_seconds() == stall['duration_seconds']
             and len({o['logged_optimizer_updates'] for o in stall['observations']}) == 1, 'REAL_BOUNDED_LONG_STALL_OBSERVATIONS_REQUIRED')
    maintenance = legacy._read_json(stop['maintenance_task_packet_ref'], root, 'explicit finite owner handoff task')
    result = legacy._read_json(stop['maintenance_result_ref'], root, 'sealed owner handoff result')
    for value in (maintenance, result):
        _json_context(value)
    start = datetime.fromisoformat(maintenance['started_at'].replace('Z', '+00:00'))
    deadline = datetime.fromisoformat(maintenance['deadline_at'].replace('Z', '+00:00'))
    targets = [{'pid': controller['pid'], 'proc_start_ticks': controller['proc_start_ticks']},
               {'pid': packet['writer']['pid'], 'proc_start_ticks': packet['writer']['proc_start_ticks']}]
    owner = maintenance.get('writer')
    signals = result.get('issued_signals')
    _require(type(signals) is list and len(signals) == 2, 'EXACT_TWO_OWNED_SIGNAL_RECORDS_REQUIRED')
    signal_times = []
    for signal_record, target in zip(signals, targets):
        _require(type(signal_record) is dict and set(signal_record) == {'pid', 'proc_start_ticks', 'signal', 'issued_at'}
                 and {k: signal_record[k] for k in target} == target
                 and type(signal_record['signal']) is int and signal_record['signal'] == 15,
                 'CONTROLLER_THEN_PUBLISHER_ONLY_SIGNAL_SEQUENCE_REQUIRED')
        at = datetime.fromisoformat(signal_record['issued_at'].replace('Z', '+00:00'))
        _require(at.tzinfo is not None and start <= at <= deadline, 'SIGNAL_OUTSIDE_FINITE_OWNER_AUTHORIZATION')
        signal_times.append(at)
    _require(signal_times[1] > signal_times[0] and stop.get('issued_at') == signals[1]['issued_at'],
             'CONTROLLER_RETIRED_BEFORE_PUBLISHER_STOP_REQUIRED')
    _require(start.tzinfo is not None and deadline.tzinfo is not None and deadline > start
             and maintenance.get('maintenance_action') == 'OWNED_INFRASTRUCTURE_NAS_STALL_STOP'
             and maintenance.get('algorithm_execution_allowed') is False
             and maintenance.get('authorized_owned_signal_targets') == targets
             and maintenance.get('parent_task_packet_ref') == stop['parent_task_packet_ref']
             and maintenance.get('worker_direct_signal_authorized') is False
             and type(owner) is dict and all(type(owner.get(k)) is int and owner[k] > 0
                    for k in ('pid', 'proc_start_ticks', 'executor_epoch'))
             and type(owner.get('fencing_token')) is str and owner['fencing_token']
             and owner.get('scope') == maintenance.get('task_id')
             and stop.get('issuer') == owner and result.get('writer') == owner
             and result.get('task_id') == maintenance.get('task_id')
             and result.get('task_packet_ref') == stop['maintenance_task_packet_ref']
             and result.get('status') == 'PASSED' and _failure_codes(result)[1] == []
             and result.get('all_owned_processes_reaped') is True and result.get('lease_released') is True,
             'FINITE_EXPLICIT_OWNER_HANDOFF_AUTHORIZATION_AND_SEAL_REQUIRED')
    return stop


def validate_periodic_parent(parent_refs, periodic_evidence, *, root):
    """Signed FAILED+PERIODIC proof plus real reaping, before any tensor load."""
    legacy = _legacy()
    root = Path(root).resolve(strict=True)
    for value in (parent_refs, periodic_evidence):
        _json_context(value)
    _require(set(parent_refs) == PERIODIC_ROLES, 'EXACT_FIVE_PERIODIC_PARENT_ROLES_REQUIRED')
    _require(set(periodic_evidence) == {'reconciliation_ref', 'state_audit_ref', 'lease_ref'}, 'EXACT_PERIODIC_EVIDENCE_REFS_REQUIRED')
    packet = legacy._read_json(parent_refs['task_packet'], root, 'periodic original task')
    outer = legacy._read_json(parent_refs['outer_result'], root, 'periodic failed outer result')
    config = legacy._read_json(parent_refs['training_config'], root, 'periodic parent config')
    sidecar = legacy._read_json(parent_refs['checkpoint_sidecar'], root, 'periodic sidecar')
    reconciliation = legacy._read_json(periodic_evidence['reconciliation_ref'], root, 'periodic reconciliation')
    audit = legacy._read_json(periodic_evidence['state_audit_ref'], root, 'periodic complete state audit')
    released = legacy._read_json(periodic_evidence['lease_ref'], root, 'sealed original lease release')
    for item in (packet, outer, config, sidecar, reconciliation, audit, released):
        _json_context(item)
    task, run, branch = packet.get('task_id'), packet.get('run_signature'), packet.get('branch')
    _require(type(task) is str and task and type(run) is str and legacy._SHA.fullmatch(run)
             and branch in {'raw', 'robotized'}, 'PERIODIC_TASK_IDENTITY_REQUIRED')
    for item in (outer, reconciliation, audit, released):
        _require(item.get('task_id') == task and item.get('run_signature') == run, 'PERIODIC_SIGNED_TASK_RUN_DRIFT')
    parent_status = outer.get('task_terminal_status')
    failure_field, failures = _failure_codes(outer)
    _require(parent_status in {'FAILED', 'FAILED_RUNTIME_FINAL'} and failures
             and (outer.get('child_exit_code') is None or type(outer['child_exit_code']) is int), 'EXPLICIT_INFRASTRUCTURE_FAILED_PARENT_REQUIRED')
    _require(not any(any(marker in error.upper() for marker in ('QUALITY', 'NONFINITE', 'BLOCKED_DATA_VOLUME'))
                         for error in failures), 'INFRASTRUCTURE_NOT_QUALITY_FAILURE_REQUIRED')
    _require(reconciliation.get('schema') == RECONCILIATION_SCHEMA
             and reconciliation.get('status') == 'SEALED_INFRASTRUCTURE_FAILED'
             and reconciliation.get('parent_refs') == parent_refs
             and reconciliation.get('parent_status_preserved') == parent_status
             and reconciliation.get('child_exit_code') == outer['child_exit_code']
             and reconciliation.get('parent_failure_field') == failure_field
             and reconciliation.get(failure_field) == failures
             and reconciliation.get('original_writer') == packet.get('writer')
             and reconciliation.get('unknowns') == [], 'SEALED_FAILED_RECONCILIATION_REQUIRED')
    classification = reconciliation.get('failure_classification')
    if classification == 'INFRASTRUCTURE_TIMEOUT':
        _require(any(error in {'INFRASTRUCTURE_TIMEOUT', 'RuntimeError:FINITE_TASK_DEADLINE_EXCEEDED'} for error in failures)
                 and 'RuntimeError:PUBLISHER_SIGNAL_15' not in failures and 'maintenance_stop_ref' not in reconciliation,
                 'ACTUAL_INFRASTRUCTURE_TIMEOUT_EVIDENCE_REQUIRED')
    elif classification == 'INFRASTRUCTURE_NAS_STALL_STOP':
        _require('RuntimeError:PUBLISHER_SIGNAL_15' in failures and 'maintenance_stop_ref' in reconciliation,
                 'EXPLICIT_INFRASTRUCTURE_STOP_EVIDENCE_REQUIRED')
        _validate_infrastructure_stop(reconciliation['maintenance_stop_ref'], packet, reconciliation, root, legacy)
    else:
        raise StateRejected('EXPLICIT_INFRASTRUCTURE_FAILURE_CLASSIFICATION_REQUIRED')
    _require(released.get('schema') == RELEASE_SCHEMA and released.get('status') == 'RELEASED'
             and released.get('writer') == packet.get('writer') and released.get('unknowns') == [],
             'ORIGINAL_TASK_FENCE_EPOCH_LEASE_RELEASE_REQUIRED')
    legacy._nondeployable(outer, 'failed periodic parent')
    legacy._nondeployable(config, 'periodic config', scope=True)
    _require(packet.get('training_scope') == legacy.TRAINING_SCOPE and config.get('supervision_class') == 'DERIVED_RETARGET'
             and (config.get('schema_version'), config.get('initialization')) in {
                 (legacy.INITIAL_CONFIG_SCHEMA, 'FRESH_SHARED_PRETRAINED_NEW_IO'),
                 (legacy.CONFIG_SCHEMA, 'RESUME_COMPLETE_TRAINING_STATE')}, 'KNOWN_PERIODIC_PARENT_RECIPE_REQUIRED')
    _require(config.get('input_bindings') == {
        'production_manifest_sha256': packet['inputs']['production_manifest']['sha256'],
        'pretrained_sha256': packet['inputs']['pretrained']['sha256'],
        'pretrained_config_sha256': packet['inputs']['pretrained_config']['sha256']}, 'CONFIG_SOURCE_INPUT_BINDINGS_DRIFT')
    checkpoint = parent_refs['checkpoint']
    path = legacy._project_path(checkpoint['path'], root, existing=True)
    output = legacy._project_path(packet['output_dir'], root, existing=True)
    _require(path.parent == output and Path(parent_refs['checkpoint_sidecar']['path']) == Path(str(path) + '.ref.json')
             and Path(parent_refs['outer_result']['path']) == output.parent / 'RESULT.json'
             and parent_refs['training_config'] == packet['inputs']['training_config'], 'PERIODIC_ORIGINAL_OUTPUT_CONFIG_IDENTITY')
    _require(sidecar == {'bytes': checkpoint['bytes'], 'sha256': checkpoint['sha256']}, 'PERIODIC_CHECKPOINT_SIDECAR_DRIFT')
    _require(audit.get('schema') == AUDIT_SCHEMA and audit.get('status') == 'SAFE_COMPLETE_STATE_VERIFIED'
             and audit.get('checkpoint_role') == PERIODIC_ROLE and audit.get('branch') == branch
             and audit.get('checkpoint_ref') == checkpoint and audit.get('checkpoint_sidecar_ref') == parent_refs['checkpoint_sidecar']
             and audit.get('training_config_ref') == parent_refs['training_config']
             and type(audit.get('checkpoint_writer')) is dict
             and audit['checkpoint_writer'] == reconciliation.get('checkpoint_writer')
             and audit.get('safe_weights_only') is True and audit.get('unknowns') == []
             and audit.get('state_keys') == 14 and audit.get('binding_keys') == 6
             and set(audit.get('full_state_components', [])) == {'model', 'ema', 'optimizer', 'rank_rng', 'timing', 'progress'},
             'COMPLETE_PINNED_PERIODIC_STATE_AUDIT_REQUIRED')
    step = audit.get('optimizer_updates')
    _require(type(step) is int and 0 < step < 20000 and step % config['training']['checkpoint_every'] == 0
             and path.name == f'checkpoint_{step:06d}.pt' and type(audit.get('cursor')) is int
             and audit['cursor'] == step * config['training']['global_batch'], 'EXACT_PERIODIC_CHECKPOINT_NOT_LOG_PROGRESS')
    _old_processes_gone(reconciliation, packet)
    return {'packet': packet, 'outer_result': outer, 'training_config': config,
            'checkpoint_ref': checkpoint, 'sidecar_ref': parent_refs['checkpoint_sidecar'],
            'branch': branch, 'periodic_audit': audit, 'reconciliation': reconciliation}


def validate_periodic_evidence(parent_refs, periodic_evidence, *, root):
    """Shared gate/runner interface, preserving the labelled nested audit."""
    evidence = validate_periodic_parent(parent_refs, periodic_evidence, root=root)
    return dict(evidence, optimizer_updates=evidence['periodic_audit']['optimizer_updates'],
                cursor=evidence['periodic_audit']['cursor'])


def validate_frozen_normal_parent(parent_refs, *, root):
    """Normal final evidence, including a future frozen periodic run's final.

    Original parents continue through the unmodified validator. A new frozen
    parent is independently validated against its REAL config schema, never a
    rewritten/projected old-schema document. No tensor load or live admission.
    """
    legacy = _legacy()
    root = Path(root).resolve(strict=True)
    _json_context(parent_refs)
    _require(set(parent_refs) == legacy.PARENT_ROLES, 'EXACT_SIX_NORMAL_PARENT_ROLES_REQUIRED')
    packet = legacy._read_json(parent_refs['task_packet'], root, 'normal original packet')
    config = legacy._read_json(parent_refs['training_config'], root, 'normal original config')
    if packet.get('source_policy') is None and config.get('schema_version') != PERIODIC_CONFIG_SCHEMA:
        evidence = legacy.validate_parent_evidence(parent_refs, root=root)
        _require(_failure_codes(evidence['outer_result'])[1] == [], 'NORMAL_PARENT_ACTUAL_FAILURE_CODES_MUST_BE_EMPTY')
        return evidence
    _require(packet.get('source_policy') == SOURCE_POLICY and packet.get('execution_mode') == FROZEN_EXECUTION_MODE,
             'EXPLICIT_FROZEN_NORMAL_PARENT_POLICY_REQUIRED')
    outer = legacy._read_json(parent_refs['outer_result'], root, 'frozen normal sealed result')
    inner = legacy._read_json(parent_refs['inner_result'], root, 'frozen normal final training result')
    sidecar = legacy._read_json(parent_refs['checkpoint_sidecar'], root, 'frozen normal final sidecar')
    for item in (packet, config, outer, inner, sidecar):
        _json_context(item)
    task, run, branch = packet.get('task_id'), packet.get('run_signature'), packet.get('branch')
    _require(type(task) is str and task and type(run) is str and legacy._SHA.fullmatch(run)
             and branch in {'raw', 'robotized'} and inner.get('branch') == branch
             and all(x.get('task_id') == task and x.get('run_signature') == run for x in (outer, inner)),
             'FROZEN_NORMAL_PARENT_TASK_BRANCH_SIGNATURE_DRIFT')
    _, failures = _failure_codes(outer)
    _require(outer.get('task_terminal_status') == 'PASSED' and type(outer.get('child_exit_code')) is int
             and outer['child_exit_code'] == 0 and failures == [], 'PASSED_EXIT_ZERO_ERROR_FREE_FINAL_REQUIRED')
    for item, label, scope in ((outer, 'frozen outer', False), (inner, 'frozen inner', True), (config, 'frozen config', True)):
        legacy._nondeployable(item, label, scope=scope)
    _require(packet.get('training_scope') == legacy.TRAINING_SCOPE
             and inner.get('supervision_class') == config.get('supervision_class') == 'DERIVED_RETARGET'
             and outer.get('source_policy') == inner.get('source_policy') == SOURCE_POLICY,
             'FROZEN_NORMAL_DERIVED_POLICY_REQUIRED')
    _require((config.get('schema_version'), config.get('initialization')) in {
        (legacy.INITIAL_CONFIG_SCHEMA, 'FRESH_SHARED_PRETRAINED_NEW_IO'),
        (legacy.CONFIG_SCHEMA, 'RESUME_COMPLETE_TRAINING_STATE'),
        (PERIODIC_CONFIG_SCHEMA, PERIODIC_INITIALIZATION)}, 'KNOWN_REAL_FROZEN_PARENT_CONFIG_SCHEMA_REQUIRED')
    _require(config.get('input_bindings') == {
        'production_manifest_sha256': packet['inputs']['production_manifest']['sha256'],
        'pretrained_sha256': packet['inputs']['pretrained']['sha256'],
        'pretrained_config_sha256': packet['inputs']['pretrained_config']['sha256']}, 'FROZEN_NORMAL_CONFIG_SOURCE_BINDINGS_DRIFT')
    if config['schema_version'] == PERIODIC_CONFIG_SCHEMA:
        _require(packet.get('resume_mode') == 'periodic' and type(config.get('resume_parent_bindings')) is dict
                 and set(config['resume_parent_bindings']) == PERIODIC_ROLES, 'REAL_PERIODIC_INITIALIZATION_PARENT_BINDING_REQUIRED')
    roles = {'task_packet': 'parent_task_packet', 'outer_result': 'parent_task_result',
             'training_config': 'parent_training_config', 'checkpoint': 'resume_checkpoint',
             'checkpoint_sidecar': 'resume_checkpoint_sidecar'}
    if packet.get('resume_mode') == 'normal':
        roles['inner_result'] = 'parent_inner_result'
    _require(packet.get('resume_mode') in {'normal', 'periodic'} and config.get('resume_parent_bindings') == {
             role: packet['inputs'].get(key) for role, key in roles.items()}, 'REAL_FROZEN_INITIALIZATION_LINEAGE_REQUIRED')
    _require(type(config.get('training')) is dict and type(config['training'].get('max_updates')) is int
             and config['training']['max_updates'] == 20000, 'FROZEN_NORMAL_UNCHANGED_20K_RECIPE_REQUIRED')
    checkpoint = parent_refs['checkpoint']
    path = legacy._project_path(checkpoint['path'], root, existing=True)
    output = legacy._project_path(packet['output_dir'], root, existing=True)
    _require(path.parent == output and Path(parent_refs['checkpoint_sidecar']['path']) == Path(str(path) + '.ref.json')
             and Path(parent_refs['inner_result']['path']).parent == output
             and Path(parent_refs['outer_result']['path']) == output.parent / 'RESULT.json'
             and parent_refs['training_config'] == packet['inputs']['training_config']
             and parent_refs['inner_result'] in outer.get('stage_results', [])
             and checkpoint in outer.get('complete_checkpoints', []), 'FROZEN_NORMAL_FINAL_PUBLISHED_IDENTITY_REQUIRED')
    step = inner.get('optimizer_updates')
    _require(type(step) is int and 0 < step <= 20000 and path.name == f'checkpoint_{step:06d}.pt'
             and inner.get('last_checkpoint') == str(path) and type(inner.get('cursor')) is int
             and inner['cursor'] == step * config['training']['global_batch'], 'FROZEN_NORMAL_FINAL_PROGRESS_REQUIRED')
    start_step = outer.get('start_optimizer_updates')
    _require(type(start_step) is int and 0 <= start_step < step and packet.get('admitted_parent_step') == start_step
             and outer.get('optimizer_updates') == step and outer.get('new_optimizer_updates') == step - start_step
             and outer.get('completed_update_budget') is (step == 20000)
             and inner.get('parent_optimizer_updates') == start_step and inner.get('new_optimizer_updates') == step - start_step,
             'FROZEN_NORMAL_REAL_NONZERO_PROGRESS_REQUIRED')
    _require(sidecar == {'bytes': checkpoint['bytes'], 'sha256': checkpoint['sha256']}, 'FROZEN_NORMAL_FINAL_SIDECAR_DRIFT')
    for ref in outer['complete_checkpoints']:
        name = Path(ref['path']).name
        _require(Path(ref['path']).parent == output and re.fullmatch(r'checkpoint_[0-9]+\.pt', name)
                 and int(name[11:-3]) <= step, 'FROZEN_NORMAL_LATER_CHECKPOINT_FORBIDDEN')
    legacy._check_code_closure(packet['runtime_code'], packet['runtime_code'], root)
    snapshot = packet['inputs']['snapshot_map']
    with legacy.open_verified_file_reference(snapshot, allowed_roots=[root], label='frozen normal snapshot map'):
        pass
    _require(inner.get('snapshot_map_ref') == snapshot, 'FROZEN_NORMAL_SNAPSHOT_BINDING_DRIFT')
    export_ref = outer.get('inference_export')
    _require(type(export_ref) is dict and Path(export_ref['path']) == output / 'INFERENCE_EXPORT.json',
             'FROZEN_NORMAL_SAFE_FINAL_EXPORT_REQUIRED')
    exported = legacy._read_json(export_ref, root, 'frozen normal inference export')
    for name in ('checkpoint_ref', 'export_receipt_ref', 'inference_contract_ref', 'model_config_ref', 'normalization_ref'):
        ref = exported.get(name)
        _require(type(ref) is dict and Path(ref['path']).parent == output / 'inference_tensors', 'FROZEN_NORMAL_EXPORT_REFERENCE_REQUIRED')
        with legacy.open_verified_file_reference(ref, allowed_roots=[root], label='frozen normal export:' + name):
            pass
    receipt = legacy._read_json(exported['export_receipt_ref'], root, 'frozen normal export receipt')
    legacy._nondeployable(receipt, 'frozen normal export')
    provenance = receipt.get('training_provenance', {})
    _require(receipt.get('status') == 'TENSOR_EXPORT_SAFE_ROUNDTRIP_VERIFIED_NOT_MODEL_QUALITY'
             and receipt.get('fixture_only') is False and receipt.get('branch') == branch
             and receipt.get('optimizer_updates') == step and type(provenance) is dict
             and provenance.get('task_id') == task and provenance.get('run_signature') == run
             and provenance.get('branch') == branch and provenance.get('optimizer_updates') == step
             and provenance.get('training_scope') == legacy.TRAINING_SCOPE
             and provenance.get('training_config_ref') == parent_refs['training_config']
             and provenance.get('training_result_ref') == parent_refs['inner_result']
             and provenance.get('production_manifest_ref') == packet['inputs']['production_manifest']
             and provenance.get('code_refs') == packet['runtime_code'] and provenance.get('source_policy') == SOURCE_POLICY
             and provenance.get('snapshot_map_ref') == snapshot and provenance.get('resume_mode') == packet.get('resume_mode'),
             'FROZEN_NORMAL_EXPORT_FULL_PROVENANCE_DRIFT')
    return {'packet': packet, 'outer_result': outer, 'inner_result': inner, 'training_config': config,
            'checkpoint_ref': checkpoint, 'sidecar_ref': parent_refs['checkpoint_sidecar'],
            'branch': branch, 'frozen_payload_parent': True}


def _normal_parent_reaped(state, evidence):
    worker = _check_checkpoint_writer(state)
    publisher = evidence['packet']['writer']
    for owner in (publisher, worker):
        _require(type(owner.get('pid')) is int and owner['pid'] > 0
                 and type(owner.get('proc_start_ticks')) is int and owner['proc_start_ticks'] > 0,
                 'NORMAL_PARENT_REAL_PROCESS_IDENTITY_REQUIRED')
        live = _proc_identity(owner['pid'])
        _require(live is None or live['proc_start_ticks'] != owner['proc_start_ticks'], 'NORMAL_PARENT_PROCESS_NOT_REAPED')
    # Existing publishers launch their rank-zero worker start_new_session=True.
    _require(not _group_members({worker['pid']}), 'NORMAL_PARENT_WORKER_SESSION_STILL_PRESENT')


def _fresh_output(output, *, claimed=False):
    if not output.exists():
        return
    _require(output.is_dir(), 'SUCCESSOR_OUTPUT_DIRECTORY_REQUIRED')
    allowed = {'runtime_cache'} | ({'.successor_adaptation.lock'} if claimed else set())
    _require(all(p.name in allowed and (p.name == '.successor_adaptation.lock' or (p.is_dir() and not p.is_symlink()))
                 for p in output.iterdir()), 'FRESH_OUTPUT_WITH_ONLY_RUNTIME_CACHE_REQUIRED')


@contextmanager
def _claim_output(output):
    """Only the standard local adaptation lock; never a second global writer."""
    output.mkdir(parents=True, exist_ok=True)
    lock = output / '.successor_adaptation.lock'
    owned = None
    try:
        with lock.open('xb') as stream:
            stat = os.fstat(stream.fileno())
            owned = (stat.st_dev, stat.st_ino)
            stream.write(str(os.getpid()).encode())
            stream.flush()
            _fresh_output(output, claimed=True)
            yield
    finally:
        if owned is not None:
            try:
                current = lock.lstat()
            except FileNotFoundError:
                raise StateRejected('OWNED_ADAPTATION_LOCK_DISAPPEARED')
            _require((current.st_dev, current.st_ino) == owned, 'OWNED_ADAPTATION_LOCK_REPLACED')
            lock.unlink()  # Explicitly authorized release of this owned lock only.


def inspect_full_checkpoint(parent_refs, *, root, roundtrip_output=None, roundtrip_ref=None):
    """Engineering-only actual readback, including RUNNING parents; no rebind.

    This does not call either prepare branch or grant parent/execution admission.
    Optional private roundtrip writes the unchanged full state, not a successor.
    """
    legacy = _legacy()
    root = Path(root).resolve(strict=True)
    _json_context(parent_refs)
    _require(set(parent_refs) == {'task_packet', 'training_config', 'checkpoint', 'checkpoint_sidecar'}, 'EXACT_INSPECTION_REFS')
    packet = legacy._read_json(parent_refs['task_packet'], root, 'inspection original task')
    config = legacy._read_json(parent_refs['training_config'], root, 'inspection original config')
    sidecar = legacy._read_json(parent_refs['checkpoint_sidecar'], root, 'inspection sidecar')
    checkpoint = parent_refs['checkpoint']
    _require(sidecar == {'bytes': checkpoint['bytes'], 'sha256': checkpoint['sha256']}
             and parent_refs['training_config'] == packet['inputs']['training_config'], 'INSPECTION_PIN_DRIFT')
    path = legacy._project_path(checkpoint['path'], root, existing=True)
    _require(path.parent == Path(packet['output_dir']) and Path(parent_refs['checkpoint_sidecar']['path']) == Path(str(path) + '.ref.json'), 'INSPECTION_OUTPUT_IDENTITY')
    state = _bounded_load(checkpoint, root, legacy)
    binding = state['binding']
    topology = binding['topology']
    projection = {'optimizer_updates': state['optimizer_updates'], 'cursor': state['cursor'],
                  'world_size': topology['world_size'], 'timing': state['timing']}
    legacy._check_state(state, {'packet': packet, 'training_config': config, 'inner_result': projection},
        SimpleNamespace(**config['training']), binding['dataset_length'], topology['world_size'], topology['device_type'])
    _adam_coverage(state)
    _check_checkpoint_writer(state)
    _require(type(state['authorization_expires_at']) in (int, float) and math.isfinite(state['authorization_expires_at'])
             and state['authorization_expires_at'] > 0, 'INSPECTION_WRITER_AUTHORIZATION_DRIFT')
    result = {'schema': 'frozen-payload-engineering-state-inspection-v1', 'status': 'SAFE_COMPLETE_STATE_INSPECTED_NOT_ADOPTED',
              'parent_admission_granted': False, 'production_authorized': False, 'checkpoint_ref': checkpoint,
              'optimizer_updates': state['optimizer_updates'], 'cursor': state['cursor'], 'state_keys': len(state),
              'binding_keys': len(binding), 'model_tensors': len(state['model']), 'ema_tensors': len(state['ema']),
              'adam_parameter_states': len(state['optimizer']['state']), 'rank_rng_count': len(state['rank_rng']),
              'tensor_array_bytes': trusted_cpu_state(state), 'dataset_length': binding['dataset_length'],
              'topology': topology, 'environment': binding['environment'], 'parent_training_wall_seconds': state['training_wall_seconds'],
              'checkpoint_writer': state['writer'],
              'safe_weights_only': True, 'actual_model_calls': 0, 'actual_optimizer_updates': 0, 'gpu_calls': 0}
    _require(roundtrip_output is None or roundtrip_ref is None, 'ONE_PRIVATE_ROUNDTRIP_MODE_REQUIRED')
    if roundtrip_ref is not None:
        _json_context(roundtrip_ref)
        path_copy = legacy._project_path(roundtrip_ref['path'], root, existing=True)
        _require(path_copy.is_relative_to(Path(__file__).resolve().parent / 'state_runtime'), 'PRIVATE_ENGINEERING_ROUNDTRIP_REF_ONLY')
        restored = _bounded_load(roundtrip_ref, root, legacy)
        _require(complete_equal(state, restored, legacy), 'ACTUAL_FULL_STATE_PRIVATE_ROUNDTRIP_MISMATCH')
        result.update(private_roundtrip_ref=roundtrip_ref, safe_full_state_roundtrip_verified=True, original_state_changed=False)
    elif roundtrip_output is not None:
        output = legacy._project_path(roundtrip_output, root)
        private = Path(__file__).resolve().parent / 'state_runtime'
        _require(output.is_relative_to(private) and not output.exists(), 'PRIVATE_FRESH_INSPECTION_OUTPUT_REQUIRED')
        output.mkdir(parents=True, exist_ok=False)
        from chaoyang.humanego_ict.training import _write_checkpoint
        target = output / path.name
        _write_checkpoint(target, state)
        ref = legacy.file_reference(target)
        restored = _bounded_load(ref, root, legacy)
        _require(complete_equal(state, restored, legacy), 'ACTUAL_FULL_STATE_PRIVATE_ROUNDTRIP_MISMATCH')
        result.update(private_roundtrip_ref=ref, safe_full_state_roundtrip_verified=True, original_state_changed=False)
    return result


def prepare_frozen_payload_checkpoint(parent_refs, *, mode='normal', root, full_config_ref,
        training_config, signatures, output_dir, dataset_length, world_size=1,
        device_type='cuda', periodic_evidence=None):
    """Return (checkpoint_path, receipt), preserving the trainer's 14/6 schema."""
    legacy = _legacy()
    root = Path(root).resolve(strict=True)
    _require(mode in {'normal', 'periodic'}, 'EXPLICIT_PARENT_MODE_REQUIRED')
    for item in (parent_refs, full_config_ref, signatures):
        _json_context(item)
    _require(type(world_size) is int and world_size >= 1 and device_type in {'cpu', 'cuda'}, 'VALID_TOPOLOGY_REQUIRED')
    from chaoyang.humanego_ict.training import TrainingConfig
    _require(type(training_config) is TrainingConfig, 'EXACT_EXISTING_TRAINING_CONFIG_REQUIRED')
    _require(training_config.max_updates == 20000 and type(training_config.max_updates) is int,
             'UNCHANGED_20K_TARGET_REQUIRED')
    _require(type(training_config.wall_seconds_max) in (int, float) and math.isfinite(training_config.wall_seconds_max)
             and 0 < training_config.wall_seconds_max <= 43200, 'FINITE_NEW_TASK_WALL_REQUIRED')
    training_config.validate(world_size)
    if mode == 'normal':
        _require(periodic_evidence is None, 'NORMAL_PARENT_CANNOT_ACCEPT_PERIODIC_EVIDENCE')
        evidence = validate_frozen_normal_parent(parent_refs, root=root)
        state = _bounded_load(evidence['checkpoint_ref'], root, legacy)
        _adam_coverage(state)
        _normal_parent_reaped(state, evidence)
        if evidence['training_config']['schema_version'] != PERIODIC_CONFIG_SCHEMA:
            result = legacy.prepare_successor_checkpoint(parent_refs, root=root, full_config_ref=full_config_ref,
                training_config=training_config, signatures=signatures, output_dir=output_dir,
                dataset_length=dataset_length, world_size=world_size, device_type=device_type)
            restored = _bounded_load(result[1]['checkpoint_ref'], root, legacy)
            for key in set(state) - {'binding', 'training_wall_seconds'}:
                _require(complete_equal(state[key], restored[key], legacy), 'NORMAL_FULL_STATE_RETENTION_MISMATCH')
            return result
    else:
        evidence = validate_periodic_parent(parent_refs, periodic_evidence, root=root)
        state = _bounded_load(evidence['checkpoint_ref'], root, legacy)
    config = legacy._read_json(full_config_ref, root, 'new full recovery config')
    expected_schema = PERIODIC_CONFIG_SCHEMA if mode == 'periodic' else legacy.CONFIG_SCHEMA
    expected_initialization = PERIODIC_INITIALIZATION if mode == 'periodic' else 'RESUME_COMPLETE_TRAINING_STATE'
    _require(config.get('schema_version') == expected_schema and config.get('initialization') == expected_initialization
             and config.get('resume_parent_bindings') == parent_refs, 'EXPLICIT_PARENT_MODE_CONFIG_REQUIRED')
    _require(legacy._config_invariants(config) == legacy._config_invariants(evidence['training_config'])
             and config['training'] == asdict(training_config), 'ONLY_FINITE_WALL_REBIND_CORE_RECIPE_UNCHANGED')
    audit = evidence.get('periodic_audit')
    structural = evidence
    if mode == 'periodic':
        _require(audit.get('world_size') == world_size and audit.get('device_type') == device_type
                 and audit.get('dataset_length') == dataset_length, 'PERIODIC_AUDIT_TOPOLOGY_DATASET_DRIFT')
        # Structural low-level projection only, NEVER a normal published RESULT.
        structural = dict(evidence, inner_result={'optimizer_updates': audit['optimizer_updates'], 'cursor': audit['cursor'],
                          'world_size': world_size, 'timing': audit['timing'], 'evidence_role': PERIODIC_ROLE})
    old_binding = legacy._check_state(state, structural, training_config, dataset_length, world_size, device_type)
    _adam_coverage(state)
    _check_checkpoint_writer(state)
    _require(type(state['authorization_expires_at']) in (int, float) and math.isfinite(state['authorization_expires_at'])
             and state['authorization_expires_at'] > 0 and type(state['training_wall_seconds']) in (int, float),
             'PARENT_FINITE_WALL_AUTHORIZATION_REQUIRED')
    if mode == 'periodic':
        _require(state['training_wall_seconds'] == audit.get('training_wall_seconds')
                 and complete_equal(state['writer'], audit['checkpoint_writer'], legacy), 'PERIODIC_WRITER_WALL_AUTHORIZATION_DRIFT')
    old_signatures = old_binding['signatures']
    _require(set(signatures) == set(old_signatures) and all(signatures[k] == v for k, v in old_signatures.items()
             if k not in legacy._MUTABLE_SIGNATURES), 'ONLY_EXISTING_SIGNATURE_REBINDS_ALLOWED')
    _require(type(signatures['run_signature']) is str and legacy._SHA.fullmatch(signatures['run_signature'])
             and signatures['run_signature'] != old_signatures['run_signature'] and signatures['config'] == full_config_ref,
             'NEW_RUN_CONFIG_SIGNATURE_REQUIRED')
    legacy._check_code_closure(old_signatures['code'], signatures['code'], root)
    output = legacy._project_path(output_dir, root)
    parent_output = Path(evidence['packet']['output_dir'])
    _require(not output.is_relative_to(parent_output) and not parent_output.is_relative_to(output),
             'FRESH_INDEPENDENT_PERIODIC_OUTPUT_REQUIRED')
    _fresh_output(output)
    adapted = dict(state, binding=dict(old_binding, config=asdict(training_config),
                   signatures=json.loads(json.dumps(signatures, sort_keys=True)), output_root=str(output)), training_wall_seconds=0.0)
    trusted_cpu_state(adapted)
    def reaped():
        if mode == 'periodic':
            _old_processes_gone(evidence['reconciliation'], evidence['packet'])
        else:
            _normal_parent_reaped(state, evidence)
    reaped()
    from chaoyang.humanego_ict.training import _write_checkpoint, _atomic
    checkpoint = output / f"checkpoint_{state['optimizer_updates']:06d}.pt"
    with _claim_output(output):
        reaped()
        _write_checkpoint(checkpoint, adapted)
        checkpoint_ref = legacy.file_reference(checkpoint)
        restored = _bounded_load(checkpoint_ref, root, legacy)
        _require(complete_equal(adapted, restored, legacy), 'PERIODIC_FULL_STATE_SAFE_ROUNDTRIP_MISMATCH')
        unchanged = sorted(set(state) - {'binding', 'training_wall_seconds'})
        _require(all(complete_equal(state[k], restored[k], legacy) for k in unchanged), 'PERIODIC_PARENT_COMPONENT_CHANGED')
        receipt = {'schema': PERIODIC_RECEIPT_SCHEMA if mode == 'periodic' else legacy.RECEIPT_SCHEMA,
        'status': 'VERIFIED_COMPLETE_PERIODIC_STATE_NOT_ADOPTED' if mode == 'periodic' else 'VERIFIED_COMPLETE_STATE_SUCCESSOR_NOT_NEW_TRAINING',
        'initialization_mode': mode, 'authorization': 'candidate_not_adopted', 'production_authorized': False,
        'source_policy': SOURCE_POLICY, 'parent_refs': parent_refs,
        'checkpoint_role': PERIODIC_ROLE if mode == 'periodic' else 'NORMAL_PASSED_FINAL_PARENT',
        'parent_task_id': evidence['packet']['task_id'], 'branch': evidence['branch'],
        'parent_terminal_status_preserved': evidence['outer_result']['task_terminal_status'], 'parent_child_exit_code_preserved': evidence['outer_result']['child_exit_code'],
        'parent_failure_field_preserved': _failure_codes(evidence['outer_result'])[0],
        'parent_failure_values_preserved': _failure_codes(evidence['outer_result'])[1], 'parent_run_signature': old_signatures['run_signature'],
        'successor_run_signature': signatures['run_signature'], 'optimizer_updates': state['optimizer_updates'], 'cursor': state['cursor'],
        'checkpoint_ref': checkpoint_ref, 'checkpoint_sidecar_ref': legacy.file_reference(Path(str(checkpoint) + '.ref.json')),
        'unchanged_state_components': unchanged, 'safe_tensor_roundtrip_verified': True, 'unsafe_pickle_fallback': False,
        'parent_training_wall_seconds': state['training_wall_seconds'], 'successor_training_wall_seconds': 0.0,
        'parent_wall_seconds_max': old_binding['config']['wall_seconds_max'], 'successor_wall_seconds_max': training_config.wall_seconds_max,
        'training_scope': legacy.TRAINING_SCOPE, 'control_ground_truth': False, 'physical_deployment_authorized': False}
        if mode == 'periodic':
            receipt['periodic_evidence'] = periodic_evidence
        receipt_name = 'PERIODIC_RECOVERY_ADAPTATION.json' if mode == 'periodic' else 'SUCCESSOR_RESUME_ADAPTATION.json'
        _atomic(output / receipt_name, lambda stream: stream.write(json.dumps(receipt, sort_keys=True, allow_nan=False).encode()))
    return checkpoint, receipt
