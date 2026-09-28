import unittest,copy,json,os,tempfile,contextlib
from pathlib import Path
import numpy as np
from chaoyang.pipeline.motion_consumer_qualification_v2 import qualify,robot_payload,fallback_image,SCHEMA
from chaoyang.pipeline.v5_motion import recover,CONSUMER_SCHEMA,_ref

class ConsumerTests(unittest.TestCase):
    def fixture(self):
        n=3;valid=np.ones((n,2),bool);t=np.broadcast_to(np.eye(4),(n,2,4,4)).copy();t[...,2,3]=1
        hand=dict(original_frame_indices=np.arange(46,49),timestamp_ns=np.arange(n)*33000000,anatomical_side_names=np.array(['left','right']),
                  observed=np.zeros((2,n),bool),inferred=np.ones((2,n),bool),predicted_valid=valid.T.copy(),
                  joints_3d_camera=np.broadcast_to([0.,0.,1.],(2,n,21,3)).copy(),root_orient_camera=np.broadcast_to(np.eye(3),(2,n,3,3)).copy(),
                  hand_pose_rotmat=np.broadcast_to(np.eye(3),(2,n,15,3,3)).copy(),intrinsics=np.broadcast_to(np.eye(3),(n,3,3)).copy())
        robot=dict(frame_id=hand['original_frame_indices'].copy(),timestamp_ns=hand['timestamp_ns'].copy(),human_to_physical=np.array([0,1]),
                   wrist_valid=valid.copy(),finger_valid=valid.copy(),target_valid=valid.copy(),q_arm=np.ones((n,2,7))*.2,q22=np.ones((n,2,22))*.3,
                   position_residual_mm=np.ones((n,2)),rotation_residual_deg=np.ones((n,2)),T_target_root_cam=t.copy(),T_actual_root_cam=t.copy(),
                   actual21_camera=np.ones((n,2,21,3)),T_cam_base=np.eye(4),T_flange_hand=np.broadcast_to(np.eye(4),(2,4,4)).copy(),placement_policy=np.array('FIXED'))
        rows=[];certs=[]
        for frame in range(46,49):
            for side in ['left','right']:
                rows.append(dict(frame_id=frame,side=side,observed=False,inferred=True,predicted_valid=True,roi_valid=True,
                                 joints_inside_image=21,joints_total=21,roi_width_px=100.,roi_height_px=100.,root_depth_m=1.,root_camera_m=[0.,0.,1.]))
                certs.append(dict(frame_id=frame,side=side,estimate_kind='MODEL_INFERENCE',review_status='APPROVED_FOR_CONSUMPTION',independent_of_model_prediction=True,
                                  evidence_refs=[dict(path='/fixture/independent.json',sha256='b'*64)],capabilities=dict(position='PASS',rotation='PASS',fingers='PASS')))
        obs=dict(session_id='fixture',inputs=dict(source=dict(sha256='a'*64)),diagnostic=dict(frame_rows=rows))
        cert=dict(schema_version=SCHEMA,session_id='fixture',source_sha256='a'*64,rows=certs)
        return hand,robot,obs,cert
    def test_qualified_model_inference_not_blanket_rejected(self):
        h,r,o,c=self.fixture();d=qualify(h,r,o,c,'a'*64,'fixture');self.assertTrue(d['masks']['robot_wrist'].all());self.assertTrue(all(x['qualified_model_inference'] and not x['physical_observation'] for x in d['ledger']))
    def test_missing_quality_and_self_evidence_rejected(self):
        h,r,o,c=self.fixture();c['rows']=[];self.assertFalse(qualify(h,r,o,c,'a'*64,'fixture')['masks']['position'].any())
        h,r,o,c=self.fixture();c['rows'][0]['evidence_refs'][0]['sha256']='a'*64
        with self.assertRaises(ValueError):qualify(h,r,o,c,'a'*64,'fixture')
    def test_interpolation_hold_copy_not_observed_or_consumed(self):
        for kind in ['INTERPOLATION','HOLD','COPY_OTHER_SIDE']:
            h,r,o,c=self.fixture();c['rows'][0]['estimate_kind']=kind;d=qualify(h,r,o,c,'a'*64,'fixture');self.assertFalse(d['masks']['position'][0,0]);self.assertTrue(d['masks']['position'][0,1])
    def test_031_zero_projection_no_neutral_hold_or_copy(self):
        h,r,o,c=self.fixture();o['diagnostic']['frame_rows'][3]['joints_inside_image']=0;d=qualify(h,r,o,c,'a'*64,'fixture');p=robot_payload(r,d)
        self.assertTrue(np.isnan(p['q_arm'][1,1]).all());self.assertTrue(np.isfinite(p['q_arm'][0,1]).all());self.assertTrue(np.isfinite(p['q_arm'][1,0]).all());np.testing.assert_array_equal(r['q_arm'],.2*np.ones((3,2,7)))
        raw=np.arange(36,dtype=np.uint8).reshape(3,4,3);np.testing.assert_array_equal(fallback_image(raw,[d['ledger'][3]]),raw)
    def test_007_missing_left_stays_missing_right_independent(self):
        h,r,o,c=self.fixture();h['predicted_valid'][0]=False;h['inferred'][0]=False
        for row in o['diagnostic']['frame_rows'][::2]:row.update(predicted_valid=False,inferred=False,joints_inside_image=0,roi_valid=False,root_camera_m=None,root_depth_m=None)
        d=qualify(h,r,o,c,'a'*64,'fixture');self.assertFalse(d['masks']['robot_wrist'][:,0].any());self.assertTrue(d['masks']['robot_wrist'][:,1].all())
    def test_poker_93_95_existing_hard_gates_not_relaxed(self):
        h,r,o,c=self.fixture();r['position_residual_mm'][0,1]=20.01;r['rotation_residual_deg'][1,1]=15.01
        d=qualify(h,r,o,c,'a'*64,'fixture');self.assertFalse(d['masks']['robot_wrist'][:2,1].any());self.assertTrue(d['masks']['robot_wrist'][2,1])
    def test_real_recover_entry_invokes_qualified_export(self):
        h,r,o,c=self.fixture()
        with contextlib.nullcontext(tempfile.mkdtemp(prefix='m01_consumer_fixture_',dir=os.environ['TMPDIR'])) as tmp:
            p=Path(tmp);np.savez(p/'hand.npz',**h);np.savez(p/'robot.npz',**r);np.savez(p/'roi.npz',roi_valid=np.ones((3,2),bool),anatomical_side_names=h['anatomical_side_names'])
            (p/'model.json').write_text(json.dumps(dict(chunks=[dict(side=s,frames=[46,47,48]) for s in [0,1]])))
            source=_ref(p/'hand.npz');o['inputs']['source']=source;c['source_sha256']=source['sha256']
            (p/'independent.json').write_text('{"fixture_only":true}')
            for row in c['rows']:row['evidence_refs']=[_ref(p/'independent.json')]
            o['diagnostic']['frame_rows'][3]['joints_inside_image']=0
            (p/'obs.json').write_text(json.dumps(o));(p/'cert.json').write_text(json.dumps(c))
            config=dict(schema_version=CONSUMER_SCHEMA,session_id='fixture',frame_count=3,output=str(p/'out'),hawor_source=source,robot_r0=_ref(p/'robot.npz'),roi=_ref(p/'roi.npz'),hawor_result=_ref(p/'model.json'),observability=_ref(p/'obs.json'),quality_qualification=_ref(p/'cert.json'))
            (p/'config.json').write_text(json.dumps(config));receipt=recover(p/'config.json');a=np.load(p/'out/ROBOT_R0_V1.npz');b=np.load(p/'out/HAND_MOTION_V1.npz')
            self.assertFalse(a['wrist_valid'][1,1]);self.assertTrue(np.isnan(a['q_arm'][1,1]).all());self.assertFalse(b['position_valid'][1,1]);self.assertFalse(b['observed_physical'].any());self.assertTrue(b['source_inferred'].all());self.assertIn('consumption_ledger',receipt['outputs'])
            self.assertEqual(recover(p/'config.json')['cache'],'REUSED')

if __name__=='__main__':unittest.main(verbosity=2)
