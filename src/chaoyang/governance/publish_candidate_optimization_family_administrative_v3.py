"""Explicit Root-approved administrative family publisher; no algorithm/task execute."""
from pathlib import Path
import argparse,copy,json
from chaoyang.governance import common_candidate_optimization_v3 as common
from chaoyang.governance.candidate_optimization_family_v2 import descriptor,ref,published_current
SCRIPT=Path(__file__)
def main(argv=None):
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--descriptor',type=Path,required=True)
 a=parser.parse_args(argv);pin=ref(a.descriptor);d=descriptor(pin)
 approval=common.load_json(Path(d['root_preapproval']['path']))
 common.verify_administrative_publication_runtime(approval,SCRIPT)
 current=published_current()
 if d['family_key'] in current['special_status_contracts']:
  if current['special_status_contracts'][d['family_key']].get('root_registered_descriptor')==pin:
   print(json.dumps({'status':'ALREADY_EXACT_ROOT_DESCRIPTOR_PUBLISHED_NO_WRITE','family_key':d['family_key'],'algorithm_called':False}));return 0
  raise ValueError('ADMIN_EXISTING_FAMILY_DIFFERENT_DESCRIPTOR_REQUIRES_NEW_VERSION')
 receipt=common.publish_bundle(common.load_json(common.AUTHORITY_PATH),copy.deepcopy(common.load_json(common.TASK_STATE_PATH)),event_type='ROOT_EXPLICIT_NEW_ONLY_ADMINISTRATIVE_PATH_INTERFACE_V2_FAMILY_PUBLISHED',expected_revision=common.load_json(common.RECEIPT_PATH)['governance_revision'],generator_path=SCRIPT,planned_family_descriptor_ref=pin)
 import subprocess,os
 cp=subprocess.run(['/usr/local/bin/python','-B','-m','chaoyang.cli','validate-governance'],cwd=common.REPO_ROOT,env={**os.environ,'PYTHONPATH':str(common.REPO_ROOT/'src'),'PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},capture_output=True,text=True,timeout=120)
 output=json.loads(cp.stdout)
 if output.get('status')!='PASS' or cp.returncode!=0:raise ValueError('ADMIN_PUBLICATION_POST_GOVERNANCE_FAILED:'+str(output.get('errors'))[:600])
 generator_sha=common.sha256_file(SCRIPT)
 for p in [common.AUTHORITY_PATH,common.TASK_STATE_PATH]:
  if common.load_json(p).get('generator_code_sha')!=generator_sha:raise ValueError('ADMIN_STANDARD_GENERATOR_CODE_SHA_NOT_BOUND')
 if common.load_json(common.RECEIPT_PATH)['files']['algorithm_contract']!=common.artifact_ref(common.AUTHORITY_PATH.parent/'ALGORITHM_CONTRACT.json'):raise ValueError('ADMIN_STANDARD_RECEIPT_ALGORITHM_COMMIT_NOT_BOUND')
 print(json.dumps({'status':'PASS','family_key':d['family_key'],'governance_revision':receipt['governance_revision'],'descriptor':pin,'administrative_publication_runtime_code':approval['administrative_publication_runtime_code'],'standard_generator_code_sha':generator_sha,'algorithm_called':False,'task_registered':False}));return 0
if __name__=='__main__':raise SystemExit(main())
