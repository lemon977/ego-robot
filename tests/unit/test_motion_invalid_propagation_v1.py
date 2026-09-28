import unittest,copy
from chaoyang.pipeline.motion_invalid_propagation_v1 import Policy,decide,propagate,consumer_payload

class PolicyTests(unittest.TestCase):
    def row(self,frame=1,side='right'):
        return dict(frame_id=frame,side=side,observed=True,inferred=False,predicted_valid=True,roi_valid=True,
                    joints_inside_image=21,joints_total=21,roi_width_px=100.,roi_height_px=100.,
                    root_depth_m=1.,root_camera_m=[0.,0.,1.],touches_bottom=False)
    def policy(self):return Policy(32.,.5,'SYNTHETIC_REVIEW_ONLY_NOT_PRODUCTION_GATE')
    def test_observed_and_inferred_remain_distinct(self):
        r=self.row();self.assertEqual(decide(r)['status'],'OBSERVED_SUPPORTED');r.update(observed=False,inferred=True)
        d=decide(r);self.assertEqual(d['status'],'INFERRED_SUPPORTED');self.assertFalse(d['effective_observed']);self.assertFalse(d['consumer_masks']['robot_position_target'])
        self.assertTrue(d['consumer_masks']['diagnostic_overlay']);self.assertFalse(d['consumer_masks']['wrist_rotation'])
    def test_small_roi_is_unknown_not_physical_invalid(self):
        r=self.row();r['roi_height_px']=12.;d=decide(r,self.policy());self.assertEqual(d['status'],'UNKNOWN')
        self.assertIn('ROI_BELOW_REVIEW_ONLY_MIN_SIDE',d['reasons'])
    def test_border_contact_alone_not_invalid(self):
        r=self.row();r['touches_bottom']=True;d=decide(r,self.policy());self.assertEqual(d['status'],'OBSERVED_SUPPORTED')
        self.assertIn('ROI_TOUCHES_BORDER_NOT_AUTOMATIC_INVALID',d['flags'])
    def test_zero_projected_joints_unknown_partial_not_all_joint_valid(self):
        r=self.row();r['joints_inside_image']=0;self.assertEqual(decide(r)['status'],'UNKNOWN')
        r['joints_inside_image']=7;d=decide(r);self.assertFalse(d['consumer_masks']['fingers']);self.assertEqual(d['status'],'OBSERVED_SUPPORTED')
    def test_nonpositive_nan_and_depth_jump(self):
        r=self.row();r['root_depth_m']=0.;r['root_camera_m'][2]=0.;self.assertEqual(decide(r)['status'],'INVALID')
        r['root_depth_m']=float('nan');self.assertEqual(decide(r)['status'],'INVALID')
        old=self.row();new=self.row(2);new['root_depth_m']=.3;new['root_camera_m'][2]=.3
        d=decide(new,self.policy(),old);self.assertEqual(d['status'],'UNKNOWN');self.assertIn('DEPTH_DISCONTINUITY_REQUIRES_REVIEW_NOT_PROVEN_COLLAPSE',d['reasons'])
    def test_gap_reset_not_depth_collapse(self):
        old=self.row();new=self.row(3);new['root_depth_m']=.3;new['root_camera_m'][2]=.3
        self.assertEqual(decide(new,self.policy(),old)['status'],'OBSERVED_SUPPORTED')
        new['frame_id']=2;new['tracking_reset']=True;self.assertEqual(decide(new,self.policy(),old)['status'],'OBSERVED_SUPPORTED')
    def test_hands_independent_and_frame_count_unchanged(self):
        right=self.row();left=self.row(side='left');left.update(predicted_valid=False,observed=False,roi_valid=False,root_camera_m=None,root_depth_m=None,joints_inside_image=0)
        rows=[left,right];before=copy.deepcopy(rows);r=propagate(rows,self.policy());self.assertEqual(rows,before)
        self.assertEqual(r['input_row_count'],r['output_row_count']);self.assertEqual(r['rows'][0]['status'],'UNKNOWN');self.assertEqual(r['rows'][1]['status'],'OBSERVED_SUPPORTED')
        with self.assertRaises(ValueError):decide(right,self.policy(),left)
    def test_missing_does_not_reset_or_hold(self):
        rows=[self.row(i) for i in range(1,20)]
        for r in rows[1:]:r['joints_inside_image']=0
        decisions=propagate(rows)['rows'];values=[consumer_payload([1.,2.,3.],d,'diagnostic_overlay') for d in decisions]
        self.assertEqual(len(values),19);self.assertEqual(values[0]['value'],[1.,2.,3.]);self.assertTrue(all(v['value'] is None for v in values[1:]))
    def test_bad_provenance_schema_and_unknown_consumer(self):
        r=self.row();r['inferred']=True;self.assertEqual(decide(r)['status'],'INVALID')
        r=self.row();r['observed']='true'
        with self.assertRaises(ValueError):decide(r)
        with self.assertRaises(ValueError):Policy(32.)
        with self.assertRaises(ValueError):consumer_payload([1.],decide(self.row()),'made_up')
    def test_no_side_copy_no_mutable_alias(self):
        r=self.row();v=[1.,2.,3.];p=consumer_payload(v,decide(r),'diagnostic_overlay');p['value'][0]=9.;self.assertEqual(v,[1.,2.,3.])
        r['root_depth_m']=2.;self.assertEqual(decide(r)['status'],'INVALID')

if __name__=='__main__':unittest.main(verbosity=2)
