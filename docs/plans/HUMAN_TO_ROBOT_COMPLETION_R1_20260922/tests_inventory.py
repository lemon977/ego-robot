"""Tests ONLY the bundled inventory checker, not the chaoyang pipeline."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from check_delivery_inventory import digest, verify


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.contract = self.root / 'contract.json'
        self.delivery = self.root / 'delivery.json'
        self.source = self.root / 'source.json'
        self.receipt = self.root / 'receipt.json'
        self.clean = self.root / 'clean.json'
        self.video = self.root / 'video.mp4'
        # Bytes are deliberately not a video; metadata-only tests must NOT claim decode.
        self.video.write_bytes(b'fixture-not-a-real-video')
        self.spec = {'artifact_id':'product:get_potato_chips_0915_007',
                     'session_id':'get_potato_chips_0915_007','kind':'robot_product','frames':2}
        self.c = {'schema_version':1,'route':'HUMAN_TO_ROBOT_BASELINE_V1',
                  'expected_artifacts':[self.spec]}
        self.s = {'session_id':self.spec['session_id'],'frame_count':2,
                  'source_frame_ids':[10,11],'timestamps_ns':[1000000000,1033333333]}
        self.cl = {'session_id':self.spec['session_id'],'frame_count':2,
                   'execution_state':'COMPLETED','model_invocations':1}
        self.r = {**self.s,'artifact_id':self.spec['artifact_id'],'kind':self.spec['kind'],
                  'output_frame_ids':[0,1]}
        self.d = {'schema_version':1,'route':self.c['route'],'run_id':'TEST_ONLY',
                  'scope':'OFFLINE_VISUAL','training_eligible':False,
                  'control_ground_truth':False,'artifacts':[]}
        self.write(self.contract,self.c)
        self.pin=digest(self.contract)
        self.sync()

    def write(self,path,obj):
        path.write_text(json.dumps(obj),encoding='utf-8')

    def ref(self,path):
        return {'path':path.name,'sha256':digest(path)}

    def sync(self):
        self.write(self.source,self.s)
        self.write(self.clean,self.cl)
        self.r.update(source_ref=self.ref(self.source),media_sha256=digest(self.video),
                      background={'kind':'CLEAN_SYNTHETIC','result_ref':self.ref(self.clean)})
        self.write(self.receipt,self.r)
        self.d['artifacts']=[{k:self.spec[k] for k in ('artifact_id','session_id','kind')}]
        self.d['artifacts'][0].update(file=self.ref(self.video),receipt=self.ref(self.receipt))
        self.write(self.delivery,self.d)

    def runcheck(self,media=False):
        return verify(self.contract,self.delivery,self.pin,media=media,timeout=30)

    def bad(self):
        self.assertEqual(self.runcheck()['inventory_status'],'FAIL')

    def test_good_inventory_does_not_certify_quality_or_media(self):
        r=self.runcheck()
        self.assertEqual(r['inventory_status'],'PASS')
        self.assertEqual(r['quality_status'],'NOT_ASSESSED')
        self.assertEqual(r['checked'][0]['media_status'],'NOT_EVALUATED')

    def test_missing(self):
        self.d['artifacts']=[]; self.write(self.delivery,self.d); self.bad()

    def test_wrong_session_label(self):
        self.d['artifacts'][0]['session_id']='play_cards_0902_042'
        self.write(self.delivery,self.d); self.bad()

    def test_wrong_source_session(self):
        self.s['session_id']='play_cards_0902_042'; self.sync(); self.bad()

    def test_video_sha_tamper(self):
        self.video.write_bytes(b'different'); self.bad()

    def test_receipt_not_bound_to_video(self):
        self.r['media_sha256']='0'*64; self.write(self.receipt,self.r)
        self.d['artifacts'][0]['receipt']=self.ref(self.receipt)
        self.write(self.delivery,self.d); self.bad()

    def test_receipt_missing_frame(self):
        self.r['source_frame_ids']=[10]; self.sync(); self.bad()

    def test_duplicate_source_frame(self):
        self.s['source_frame_ids']=[10,10]; self.r['source_frame_ids']=[10,10]
        self.sync(); self.bad()

    def test_nonmonotonic_timestamps(self):
        self.s['timestamps_ns']=[10,9]; self.r['timestamps_ns']=[10,9]
        self.sync(); self.bad()

    def test_different_source_timeline(self):
        self.r['timestamps_ns']=[1000000000,1040000000]; self.sync(); self.bad()

    def test_invalid_output_indices(self):
        self.r['output_frame_ids']=[1,0]; self.sync(); self.bad()

    def test_symlink_disallowed(self):
        target=self.root/'real.mp4'; self.video.rename(target); self.video.symlink_to(target)
        self.sync(); self.bad()

    def test_duplicate_artifact(self):
        self.d['artifacts'].append(copy.deepcopy(self.d['artifacts'][0]))
        self.write(self.delivery,self.d); self.bad()

    def test_unsafe_qualification(self):
        self.d['control_ground_truth']=True; self.write(self.delivery,self.d); self.bad()

    def test_changed_frozen_contract(self):
        self.c['expected_artifacts'][0]['frames']=3; self.write(self.contract,self.c); self.bad()

    def test_clean_never_executed(self):
        self.cl['model_invocations']=0; self.sync(); self.bad()

    def test_raw_background_not_product(self):
        self.r['background']['kind']='RAW'; self.write(self.receipt,self.r)
        self.d['artifacts'][0]['receipt']=self.ref(self.receipt)
        self.write(self.delivery,self.d); self.bad()

    def test_missing_run(self):
        self.d['run_id']=None; self.write(self.delivery,self.d); self.bad()

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),'media tools unavailable')
    def test_actual_media_decode_and_wrong_count(self):
        subprocess.run(['ffmpeg','-nostdin','-v','error','-f','lavfi','-i',
                        'color=size=32x32:rate=2:duration=1','-frames:v','2',
                        '-c:v','mpeg4','-threads','1','-y',str(self.video)],check=True)
        self.sync()
        out=self.runcheck(media=True)
        self.assertEqual(out['inventory_status'],'PASS',out['errors'])
        self.assertEqual(out['checked'][0]['decoded_frames'],2)
        # Rebind all metadata to 3 frames; actual two-frame MP4 must still fail.
        self.spec['frames']=3; self.write(self.contract,self.c); self.pin=digest(self.contract)
        for x in (self.s,self.r):
            x['frame_count']=3; x['source_frame_ids']=[10,11,12]
            x['timestamps_ns']=[1000000000,1033333333,1066666666]
        self.r['output_frame_ids']=[0,1,2]; self.cl['frame_count']=3; self.sync()
        self.assertEqual(self.runcheck(media=True)['inventory_status'],'FAIL')

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),'media tools unavailable')
    def test_fake_media_fails_decode(self):
        self.assertEqual(self.runcheck(media=True)['inventory_status'],'FAIL')


if __name__=='__main__':
    unittest.main(verbosity=2)
