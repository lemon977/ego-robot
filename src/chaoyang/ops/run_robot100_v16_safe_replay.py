"""Explicit fresh-output replay; never calls the legacy fixed-output main()."""
import argparse
import io
import json
import sys
from pathlib import Path
from chaoyang.pipeline.robot100_v16_replay_safety import claim_output, read_verified, require_task, validate_output, require_manifest, require_environment


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--task-id',required=True)
    parser.add_argument('--mode',choices=('verify','solve','render'),default='verify')
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[3]
    # All guards precede imports which can initialize scientific libraries or write outputs.
    validate_output(args.output_root,repo/'_run/current')
    command_args=tuple(sys.argv[1:])
    packet=require_task(repo,args.task_id,args.output_root,'run_robot100_v16_safe_replay',command_args)
    manifest=require_manifest(packet,args.manifest)
    if manifest.get('schema_version')!='ROBOT100_V16_REPRODUCTION_REVISION_2': raise ValueError('MANIFEST_SCHEMA')
    for category in ('runtime_code','frozen_inputs','assets','snapshot_refs'):
        for ref in manifest[category]: read_verified(ref)
    environment=json.loads(read_verified(manifest['environment']))
    if args.mode!='verify':
        require_environment(environment)
    # Recheck authorization after hashing, before reserving any output directory.
    if require_task(repo,args.task_id,args.output_root,'run_robot100_v16_safe_replay',command_args)!=packet:
        raise ValueError('AUTHORIZATION_CHANGED')
    output=claim_output(args.output_root,repo/'_run/current')
    verified={'status':'PASS','mode':args.mode,'manifest':str(args.manifest),'legacy_main_called':False,'historical_output_modified':False}
    if args.mode!='verify':
        import numpy as np
        from chaoyang.ops import run_robot100_constraint_review_v16_retry1 as legacy
        with np.load(io.BytesIO(read_verified(manifest['v15_motion'])),allow_pickle=False) as z:
            source={k:z[k] for k in z.files}
        assets=legacy.old.load_pinned_robot_assets(repo)
        mounts,_=legacy.old.baseline._mounts()
        solved,changes=legacy.refine(source,assets,mounts)
        with np.load(io.BytesIO(read_verified(manifest['selected_motion'])),allow_pickle=False) as z:
            if set(solved)!=set(z.files): raise ValueError('MOTION_FIELDS_CHANGED')
            for key in ('q_arm','T_actual_root','q22','T_target_root'):
                np.testing.assert_allclose(solved[key],z[key],rtol=1e-7,atol=1e-8,equal_nan=True)
            for key in set(z.files)-{'q_arm','T_actual_root','q22','T_target_root'}:
                if np.asarray(z[key]).dtype.kind in 'fc':
                    np.testing.assert_allclose(solved[key],z[key],rtol=1e-7,atol=1e-8,equal_nan=True,err_msg=key)
                else:
                    np.testing.assert_equal(solved[key],z[key],err_msg=key)
        np.savez_compressed(output/'ROBOT_MOTION_REPLAY.npz',**solved)
        verified['semantic_replay']='PASS'
        verified['tracking_pass_left_right']=solved['tolerance_pass'].sum(axis=0).tolist()
        if args.mode=='render':
            # render() uses ROOT only for its own screenshots; new subprocess-local binding.
            legacy.ROOT=output
            verified['decode']=legacy.render(manifest['source_row'],solved,output/'Robot_V16_replay.mp4')
    with (output/'RESULT.json').open('x',encoding='utf8') as f: json.dump(verified,f,indent=2)
    print(json.dumps(verified))
    return 0


if __name__=='__main__': raise SystemExit(main())
