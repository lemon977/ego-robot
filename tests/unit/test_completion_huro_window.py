import unittest
from unittest.mock import patch
from pathlib import Path
from datetime import datetime, timedelta, timezone
import os
import numpy as np
from chaoyang.ops import run_four_stream_completion_huro as runner

class WindowTests(unittest.TestCase):
    def test_fixed_source_axis_and_independent_masks(self):
        masks = np.ones((378,2),bool); masks[190,0]=False
        hand={'frame_id':np.arange(378), 'timestamp_ns':np.arange(378)*30000000,
              'anatomical_side_names':np.array(['left','right'])}
        r0={'T_flange_hand':np.zeros((2,4,4)), 'T_cam_base':np.eye(4), 'target_valid':masks}
        selected=runner.slice_case(dict(hand=hand,r0=r0,valid=masks,rotation_valid=masks))
        np.testing.assert_array_equal(selected['hand']['frame_id'],np.arange(181,197))
        self.assertFalse(selected['valid'][9,0]); self.assertTrue(selected['valid'][9,1])
        self.assertEqual(selected['r0']['T_flange_hand'].shape,(2,4,4))
        selected['valid'][0,0]=False; self.assertTrue(masks[181,0])

    def lease(self):
        return dict(status='ACQUIRED',task_id=runner.PARENT,pid=44,process_startticks=55,
                    gpu_process_pid=66,gpu_id=0,executor_epoch=31,
                    expires_at=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat())

    def test_lease_owns_actual_child_before_jax(self):
        with patch.object(runner,'read',return_value=self.lease()), patch.object(runner,'ticks',return_value=55), patch.object(os,'getppid',return_value=44), patch.object(os,'getpid',return_value=66), patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'0','JAX_PLATFORMS':'cuda'}):
            self.assertEqual(runner.lease_guard(Path('/irrelevant'),{'executor_epoch':31})['gpu_id'],0)

    def test_wrong_task_owner_epoch_and_expired_lease_rejected(self):
        for key,value in [('task_id','other_task'),('pid',99),('gpu_process_pid',99),('executor_epoch',30),('expires_at','2000-01-01T00:00:00+00:00')]:
            lease=self.lease(); lease[key]=value
            with patch.object(runner,'read',return_value=lease), patch.object(runner,'ticks',return_value=55), patch.object(os,'getppid',return_value=44), patch.object(os,'getpid',return_value=66), patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'0','JAX_PLATFORMS':'cuda'}):
                with self.assertRaises(RuntimeError): runner.lease_guard(Path('/irrelevant'),{'executor_epoch':31})

    def test_bad_attempt_rejected_before_metadata_read(self):
        with self.assertRaises(ValueError): runner.guard(Path.cwd(),'../foreign')

    def test_cpu_backend_rejects_gpu_env_and_excess_affinity(self):
        with patch.dict(os.environ, {'JAX_PLATFORMS':'cpu','CUDA_VISIBLE_DEVICES':''}), patch.object(os,'sched_getaffinity',return_value={2,3}):
            runner.cpu_guard()
        with patch.dict(os.environ, {'JAX_PLATFORMS':'cuda','CUDA_VISIBLE_DEVICES':'0'}), patch.object(os,'sched_getaffinity',return_value={2,3}):
            with self.assertRaises(RuntimeError): runner.cpu_guard()
        with patch.dict(os.environ, {'JAX_PLATFORMS':'cpu','CUDA_VISIBLE_DEVICES':''}), patch.object(os,'sched_getaffinity',return_value={0,1,2}):
            with self.assertRaises(RuntimeError): runner.cpu_guard()

if __name__=='__main__': unittest.main(verbosity=2)
