"""Pure CPU maintenance contract tests; no models, temp dirs or production."""
import copy
from pathlib import Path
import runpy
import unittest

HELPER = Path(__file__).resolve().parents[2] / 'scripts/maintenance/project_health_fixes_v1.py'
MODULE = runpy.run_path(str(HELPER), run_name='readset_contract_tests')
ROOT = MODULE['ROOT']
normalize = MODULE['normalized_spec']
covered = MODULE['covered']


def sample():
    plan = ROOT / 'docs/plans/PICK_PLACE_1008_RAW_TO_ROBOT_V1/reverse_native_boundary_v9'
    manifest = str(ROOT / '_run/current/source/manifest.json')
    camera = str(ROOT / '_run/current/camera_v9/candidate/camera.json')
    reuse = str(ROOT / '_run/current/fresh_source/candidate')
    review = str(ROOT / 'review/current/INDEX.json')
    mask = str(plan / 'mask_indexes/batch.json')
    return {'task_id': 'fixture', 'operation': 'fixture', 'budgets': {'gpu_count': 1},
        'args': ['--manifest', manifest, '--camera-config', camera, '--mask-seeds', mask,
                 '--reuse-from', reuse],
        'inputs': [manifest, camera, mask, str(plan / 'CODE.json'), reuse + '/INDEX.json', review],
        'read_set': ['AGENTS.md', 'baseline_v1/MANIFEST.json', manifest,
            str(ROOT / '_run/current/shared_b_right_fixed_k_installation_v8_fixture/candidate'),
            mask, str(plan), 'src/chaoyang/pick_place_1008']}


class ReadSetTests(unittest.TestCase):
    def test_all_literal_inputs_covered(self):
        fixed = normalize(sample())
        self.assertTrue(all(covered(p, fixed['read_set']) for p in fixed['inputs']))

    def test_camera_and_reuse_covered(self):
        spec = normalize(sample())
        for flag in ['--camera-config', '--reuse-from']:
            self.assertTrue(covered(MODULE['option'](spec['args'], flag), spec['read_set']))

    def test_initial_eight_file_contract(self):
        self.assertEqual(len(normalize(sample())['read_set']), 8)

    def test_algorithms_arguments_budgets_not_changed(self):
        old = sample(); fixed = normalize(old)
        self.assertEqual({k:v for k,v in old.items() if k != 'read_set'},
                         {k:v for k,v in fixed.items() if k != 'read_set'})

    def test_original_not_mutated(self):
        old = sample(); before = copy.deepcopy(old); normalize(old)
        self.assertEqual(old, before)

    def test_unused_v8_removed(self):
        self.assertFalse(any('installation_v8_' in p for p in normalize(sample())['read_set']))

    def test_required_v8_is_not_discarded(self):
        spec = sample(); spec['inputs'].append(spec['read_set'][3] + '/required.json')
        spec['read_set'].remove('AGENTS.md')
        self.assertTrue(any('installation_v8_' in p for p in normalize(spec)['read_set']))

    def test_redundant_leaf_removed(self):
        old = sample(); mask = MODULE['option'](old['args'], '--mask-seeds')
        self.assertNotIn(mask, normalize(old)['read_set'])

    def test_excess_initial_scope_rejected(self):
        spec = sample(); spec['inputs'].append(str(ROOT / 'unrelated/input.json'))
        with self.assertRaisesRegex(RuntimeError, 'OVER_EIGHT'): normalize(spec)

    def test_project_escape_rejected(self):
        spec = sample(); spec['inputs'].append('/tmp/not-authorized.json')
        with self.assertRaisesRegex(RuntimeError, 'OUTSIDE'): normalize(spec)

    def test_duplicate_option_rejected(self):
        spec = sample(); spec['args'] += ['--camera-config', str(ROOT / 'other.json')]
        with self.assertRaisesRegex(RuntimeError, 'DUPLICATE_OPTION'): normalize(spec)

    def test_idempotent_normalization(self):
        fixed = normalize(sample()); self.assertEqual(normalize(fixed), fixed)


if __name__ == '__main__': unittest.main()
