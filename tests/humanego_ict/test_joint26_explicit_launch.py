"""Launch controls: real config validation, synthetic CLI dispatch, no model execution."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts/humanego_ict'))
import joint26_runner_v1 as single
import joint26_parallel_runner_v1 as paired

def arguments(**values):
    v = dict(steps=10000, config=None, nproc=4, wall_seconds=None,
             global_batch=None, micro_batch=None)
    v.update(values)
    return SimpleNamespace(**v)

class ExplicitLaunch(unittest.TestCase):
    def test_default_preserved(self):
        c, _ = single.resolve_training_config(arguments())
        self.assertEqual((c.global_batch, c.micro_batch, c.max_updates), (64,16,10000))

    def test_four_gpus_micro_eight_accumulates_two(self):
        c, d = single.resolve_training_config(arguments(global_batch=64,micro_batch=8))
        self.assertEqual(c.global_batch // (4*c.micro_batch), 2)
        self.assertEqual(d['training']['micro_batch'], 8)

    def test_single_and_eight_gpu_topologies(self):
        for n, accumulation in [(1,8),(8,1)]:
            c, _ = single.resolve_training_config(arguments(nproc=n,global_batch=64,micro_batch=8))
            self.assertEqual(c.global_batch // (n*c.micro_batch),accumulation)

    def test_invalid_topology_rejected(self):
        with self.assertRaisesRegex(ValueError,'GLOBAL_BATCH_TOPOLOGY_MISMATCH'):
            single.resolve_training_config(arguments(micro_batch=3))

    def test_non_positive_micro_rejected(self):
        for value in [0,-1]:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError,'POSITIVE_INTEGER_MICRO_BATCH'):
                single.resolve_training_config(arguments(micro_batch=value))

    def test_frozen_global_batch_preserved(self):
        with self.assertRaisesRegex(ValueError,'JOINT26_PRODUCTION_GLOBAL64_REQUIRED'):
            single.resolve_training_config(arguments(global_batch=32,micro_batch=8))

    def test_cli_override_frozen_effective_config_without_rewriting_source(self):
        d = single.default_config(4,None,10000)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'config.json';p.write_text(json.dumps(d));before=p.read_bytes()
            c,effective=single.resolve_training_config(arguments(config=p,micro_batch=8,global_batch=64))
            self.assertEqual(c.micro_batch,8);self.assertEqual(effective['training']['micro_batch'],8)
            self.assertEqual(p.read_bytes(),before)

    def test_single_batch_size_alias(self):
        with mock.patch.object(single,'execute',return_value=0) as run:
            self.assertEqual(single.main(['--branch','raw','--manifest','unused.json','--nproc','4',
                 '--steps','10000','--global-batch','64','--batch-size','8']),0)
            args=run.call_args.args[0];self.assertEqual((args.global_batch,args.micro_batch),(64,8))

    def test_parallel_forwards_batch_to_both_configs(self):
        a=paired.parse_args(['--parallel','--manifest','unused.json','--raw-gpu-ids','0,1,2,3',
              '--robotized-gpu-ids','4,5,6,7','--steps','10000','--global-batch','64',
              '--micro-batch','8','--preflight-only'])
        with mock.patch.object(single,'resolve_training_config',wraps=single.resolve_training_config) as cfg, \
             mock.patch('chaoyang.humanego_ict.joint26_training.preflight',return_value=([],[],{})), \
             mock.patch.object(paired,'ConsoleReporter'), \
             mock.patch('chaoyang.humanego_ict.joint26_runtime._ordinary',side_effect=lambda p,**k:p):
            self.assertEqual(paired.execute(a),0)
        self.assertEqual(cfg.call_count,2)
        for call in cfg.call_args_list:
            x=call.args[0];self.assertEqual((x.nproc,x.global_batch,x.micro_batch),(4,64,8))

    def test_parallel_alias(self):
        a=paired.parse_args(['--parallel','--manifest','unused.json','--raw-gpu-ids','0',
            '--robotized-gpu-ids','1','--batch-size','8'])
        self.assertEqual(a.micro_batch,8)

    def test_shell_missing_python_value_is_clear_error(self):
        p=subprocess.run(['bash',str(ROOT/'scripts/humanego_ict/train_joint26_pick_place.sh'),
              '--python'],capture_output=True,text=True,timeout=20)
        self.assertEqual(p.returncode,2);self.assertIn('--python',p.stderr)

    def test_failed_conda_activation_does_not_exit_strict_caller(self):
        # A fresh test shell intentionally inherits strict options; the recipe isolates them.
        conda=Path('/path/to/private_workspace/miniconda3/etc/profile.d/conda.sh')
        if not conda.is_file():self.skipTest('actual Conda unavailable')
        with tempfile.TemporaryDirectory() as tmp:
            command=('set -euo pipefail; (set +e +u +o pipefail; source '+str(conda)+
                ' || exit 2; conda activate '+tmp+'/missing_env || exit 2) || '
                '{ ICT_TEST_STATUS=$?; printf "failure=%s\\n" "$ICT_TEST_STATUS"; }; printf "LOGIN_ALIVE\\n"')
            p=subprocess.run(['bash','-c',command],capture_output=True,text=True,timeout=45)
        self.assertEqual(p.returncode,0);self.assertIn('LOGIN_ALIVE',p.stdout)
        self.assertIn('failure=2',p.stdout)

if __name__ == '__main__':
    unittest.main()
