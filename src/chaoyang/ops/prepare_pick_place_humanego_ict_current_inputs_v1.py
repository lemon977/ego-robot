"""Registered current-only CPU input producer; no model or hardware imports."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-packet', type=Path, required=True)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--branch', choices=('raw', 'robotized'), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    from chaoyang.governance.common import REPO_ROOT, artifact_ref, load_json

    def authority(request):
        # The producer independently checks this packet, actual writer ancestry,
        # exact command, active registry, deadline, source/code SHA and CPU scope
        # before it imports numerical dependencies or writes any current file.
        return {'status': 'PASSED', 'execution_mode': packet['execution_mode'],
            'production_inputs_write_allowed': True, 'fixture_only': False,
            'task_packet_ref': packet_ref, 'task_id': packet['task_id'],
            'run_signature': packet['run_signature'], 'writer': packet['writer']}

    try:
        # Reject lexical external paths before any SHA/stat/resolve touches.
        for value in (args.task_packet, args.spec, args.manifest, args.output_dir):
            if (not value.is_absolute() or '..' in value.parts or value == REPO_ROOT
                    or not value.is_relative_to(REPO_ROOT)):
                raise ValueError('ABSOLUTE_PROJECT_SUBPATH_REQUIRED_BEFORE_IO')
        if not args.task_packet.is_relative_to(REPO_ROOT / 'tasks/current'):
            raise ValueError('REGISTERED_CURRENT_PACKET_PATH_REQUIRED_BEFORE_IO')
        for value in (args.task_packet, args.spec, args.manifest, args.output_dir):
            if any(item.is_symlink() for item in (value, *value.parents)):
                raise ValueError('SYMLINK_INPUT_OR_OUTPUT_FORBIDDEN_BEFORE_SHA')
        from chaoyang.humanego_ict.evaluation_current_producer_v1 import produce_current_inputs, _write_json
        packet_ref = artifact_ref(args.task_packet)
        packet = load_json(args.task_packet)
        result = produce_current_inputs(artifact_ref(args.spec), artifact_ref(args.manifest),
            branch=args.branch, output_root=args.output_dir, project_root=REPO_ROOT,
            authorization_verifier=authority)
        result.update(schema='humanego-ict-current-input-production-result-v1',
            task_id=packet['task_id'], run_signature=packet['run_signature'],
            evaluation_input_spec_ref=packet['inputs']['evaluation_input_spec'],
            production_manifest_ref=packet['inputs']['production_manifest'],
            source_policy='FROZEN_ADMITTED_PROJECT_PAYLOAD_SNAPSHOT_RESEARCH_ONLY')
        _write_json(args.output_dir / 'RESULT.json', result, REPO_ROOT.resolve(strict=True))
        print(json.dumps({'status': result['status'], 'sample_count': result['sample_count'],
            'source_group_count': result['source_group_count'], 'model_calls': 0}))
        return 0
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as error:
        print(json.dumps({'status': 'BLOCKED', 'failure_code': str(error),
            'model_calls': 0, 'physical_deployment_authorized': False}), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
