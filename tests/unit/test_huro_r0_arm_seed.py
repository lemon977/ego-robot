import copy
import unittest
from unittest.mock import patch, MagicMock
from chaoyang.ops import run_four_stream_completion_huro as runner
import numpy as np
from chaoyang.ops.run_four_stream_completion_huro import r0_arm_seed, candidate_suffix

class R0SeedTests(unittest.TestCase):
    def fixture(self):
        names=[f'Joint{i}_{side}' for side in ['L','R'] for i in range(1,8)]+['left_hand','right_hand']
        mapping=dict(names=names,home=[0.]*14+[.25,-.5],joint_mask=[True]*16,
                     lower=[-2.]*16,upper=[2.]*16,full_lower=[-2.]*16,full_upper=[2.]*16,
                     affine_matrix=np.eye(16).tolist(),affine_offset=[0.]*16)
        q=np.arange(16*2*7).reshape(16,2,7)/1000
        pose=np.broadcast_to(np.eye(4),(16,2,4,4)).copy()
        r0=dict(human_to_physical=np.array([0,1]),anatomical_side_names=np.array(['left','right']),
                frame_id=np.arange(181,197),timestamp_ns=np.arange(16)*33000000,
                target_valid=np.ones((16,2),bool),wrist_valid=np.ones((16,2),bool),q_arm=q,
                T_target_root_cam=pose.copy(),T_actual_root_cam=pose.copy())
        case=dict(r0=r0,hand={k:r0[k].copy() for k in ['frame_id','timestamp_ns']},
                  valid=np.ones((16,2),bool),rotation_valid=np.ones((16,2),bool))
        return mapping,case

    def test_name_mapping_padding_and_hand_home_unchanged(self):
        m,c=self.fixture(); original=copy.deepcopy((m,c)); seed=r0_arm_seed(m,c)
        np.testing.assert_array_equal(seed[:16,:7],c['r0']['q_arm'][:,0].astype(np.float32))
        np.testing.assert_array_equal(seed[:16,7:14],c['r0']['q_arm'][:,1].astype(np.float32))
        np.testing.assert_array_equal(seed[16:,:14],np.broadcast_to(seed[15,:14],(16,14)))
        np.testing.assert_array_equal(seed[:,14:],np.broadcast_to([.25,-.5],(32,2)))
        self.assertEqual(m,original[0])
        for k,v in c['r0'].items(): np.testing.assert_array_equal(v,original[1]['r0'][k])
        np.testing.assert_array_equal(c['valid'],original[1]['valid'])
        # Internal ordering need not be arm-first; named binding determines mapping.
        order=np.arange(16)[::-1]; perm=copy.deepcopy(m)
        for k in ['names','home','joint_mask','lower','upper']: perm[k]=[m[k][i] for i in order]
        perm['affine_matrix']=np.eye(16).tolist()
        np.testing.assert_array_equal(r0_arm_seed(perm,c),seed[:,order])

    def test_reject_invalid_or_unknown_rotation_without_changing_masks(self):
        for container,key in [('case','valid'),('case','rotation_valid'),('r0','target_valid'),('r0','wrist_valid')]:
            m,c=self.fixture(); value=(c if container=='case' else c['r0'])[key];value[3,1]=False
            with self.assertRaisesRegex(ValueError,'VALID'):r0_arm_seed(m,c)
            self.assertFalse(value[3,1])

    def test_reject_side_names_frames_time_and_joint_names(self):
        for key,value in [('human_to_physical',[1,0]),('anatomical_side_names',['right','left']),('frame_id',np.arange(16)),('timestamp_ns',np.arange(16))]:
            m,c=self.fixture();c['r0'][key]=np.asarray(value)
            with self.assertRaises(ValueError):r0_arm_seed(m,c)
        for replacement in ['Joint1_L','unknown']:
            m,c=self.fixture();m['names'][1]=replacement
            with self.assertRaises(ValueError):r0_arm_seed(m,c)

    def test_reject_raw_limits_nan_locked_and_mimic_limits_no_clip(self):
        for value in [3.,np.nan]:
            m,c=self.fixture();c['r0']['q_arm'][2,1,3]=value
            with self.assertRaises(ValueError):r0_arm_seed(m,c)
            self.assertTrue(np.isnan(c['r0']['q_arm'][2,1,3]) if np.isnan(value) else c['r0']['q_arm'][2,1,3]==3.)
        m,c=self.fixture();m['joint_mask'][0]=False
        with self.assertRaises(ValueError):r0_arm_seed(m,c)
        m,c=self.fixture();m['full_upper'][2]=.01
        with self.assertRaisesRegex(ValueError,'FULL_LIMITS'):r0_arm_seed(m,c)

    def test_recompute_wrist_gate_not_trust_saved_metric(self):
        m,c=self.fixture();c['r0']['position_residual_mm']=np.zeros((16,2));c['r0']['T_actual_root_cam'][1,1,0,3]=.021
        with self.assertRaisesRegex(ValueError,'WRIST_GATE'):r0_arm_seed(m,c)
        m,c=self.fixture();r=np.array([[0.,-1,0],[1,0,0],[0,0,1]]);c['r0']['T_actual_root_cam'][1,1,:3,:3]=r
        with self.assertRaisesRegex(ValueError,'WRIST_GATE'):r0_arm_seed(m,c)
        m,c=self.fixture();c['r0']['T_actual_root_cam'][1,1,0,0]=2.
        with self.assertRaisesRegex(ValueError,'SO3'):r0_arm_seed(m,c)

    def test_predecessor_rejects_target_gate_loss_home_and_environment_drift(self):
        previous=dict(frames=[181,196],position_gate_mm=20.,rotation_gate_deg=15.,solver_block_size=32,
                      dt_policy='UNCHANGED_UPSTREAM_ADJACENT_Q_DIFFERENCE',environment='env',
                      motion='motion',r0='r0',mapping='map',candidate_source='src',core='core',assets=[],code=[])
        fakepath=MagicMock();fakepath.read_text.return_value='same-core-source'
        with patch.object(runner,'environment',return_value='env'),patch.object(runner,'ref',side_effect=lambda x:x),patch.object(runner,'read',return_value='same-map'),patch.object(runner,'verify',return_value=fakepath):
            runner.verify_c3_predecessor(previous,'same-map','same-core-source','motion','r0')
            for key,value in [('frames',[180,195]),('position_gate_mm',21.),('rotation_gate_deg',16.),('solver_block_size',16),('dt_policy','changed'),('environment','new'),('motion','changed')]:
                bad=copy.deepcopy(previous);bad[key]=value
                with self.assertRaises(RuntimeError):runner.verify_c3_predecessor(bad,'same-map','same-core-source','motion','r0')
            with self.assertRaisesRegex(RuntimeError,'MAPPING_DRIFT'):
                runner.verify_c3_predecessor(previous,'changed-home','same-core-source','motion','r0')
            with self.assertRaisesRegex(RuntimeError,'SOURCE_DRIFT'):
                runner.verify_c3_predecessor(previous,'same-map','changed-loss','motion','r0')

    def test_explicit_non_overlapping_candidate_directories(self):
        self.assertEqual(candidate_suffix('c2','cpu'),'_CPU_V1')
        self.assertEqual(candidate_suffix('c3','cpu'),'_C3_R0_ARM_CPU_V1')
        with self.assertRaises(ValueError):candidate_suffix('c4','cpu')

if __name__=='__main__':unittest.main(verbosity=2)
