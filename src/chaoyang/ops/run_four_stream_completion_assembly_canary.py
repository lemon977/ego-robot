"""Registered three-frame assembly canary; legacy motion is diagnostic only."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import numpy as np
from chaoyang.ops.run_four_stream_completion_sensor import checked_path, read_pinned, PARENT, TASK
from chaoyang.pipeline.assembly_visibility_canary_v1 import FRAMES, frame_report, PANEL_DESIGN

SESSION = 'get_potato_chips_0915_007'
MOTION = '_run/current/four_stream_visual_delivery_v5/attempts/attempt_0001/lanes/motion/recovered_007_v1/ROBOT_R0_V1.npz'
DOMAIN = '_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/lanes/exact78/prepare_full_v1/get_potato_chips_0915_007/DOMAIN_MANIFEST.json'
MOUNT = '_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/shared/cad/interfaces_mount_v2_corrected/MOUNT_CONTRACT.json'


def authority(config):
    root = Path(config['repo_root']).resolve(strict=True)
    packet = json.loads(read_pinned(config['packet'], root))
    parent = json.loads(read_pinned(config['parent_packet'], root))
    if packet['task_id'] != TASK or parent['task_id'] != PARENT or packet['writer'] != parent['writer']:
        raise ValueError('delegated publisher mismatch')
    index = json.loads((root/'tasks/current/INDEX.json').read_text())
    entries = {x['task_id']:x for x in index['task_packets']}
    if not entries[PARENT]['execution_allowed']:
        raise ValueError('parent not routable')
    for task,key in ((PARENT,'parent_packet'),(TASK,'packet')):
        if entries[task]['packet_sha256'] != config[key]['sha256'] or checked_path(entries[task]['packet_path'],root) != checked_path(config[key]['path'],root):
            raise ValueError('packet registration drift')
    writer = packet['writer']
    ticks = Path(f"/proc/{writer['pid']}/stat").read_text().split(') ',1)[1].split()[19]
    if ticks != str(writer['proc_start_ticks']):
        raise ValueError('publisher identity changed')
    lane = checked_path(packet['output_root'],root)
    expected = root/'_run/current'/PARENT/'attempts/attempt_0001/lanes/sensor'
    if lane != expected or lane not in [checked_path(p,root) for p in packet['write_set']]:
        raise ValueError('wrong delegated output lane')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('GPU prohibited')
    return root,lane,index['task_packets'],ticks


def validate_motion(motion, domain, predecessor):
    if domain['session_id'] != SESSION or predecessor['session_id'] != SESSION:
        raise ValueError('wrong session')
    if predecessor['status'] != 'LEGACY_MOTION_RECOVERED_FOR_DIAGNOSTIC' or not bool(motion['legacy_candidate']):
        raise ValueError('legacy diagnostic identity required')
    if bool(motion['control_ground_truth']) or bool(motion['training_eligible']):
        raise ValueError('legacy authority unexpectedly elevated')
    for key,shape in [('q_arm',(378,2,7)),('q_hand22',(378,2,22)),('wrist_valid',(378,2)),('finger_valid',(378,2))]:
        if motion[key].shape != shape:
            raise ValueError('motion shape mismatch: '+key)
    for key in ('wrist_valid','finger_valid'):
        if motion[key].dtype != np.bool_:
            raise ValueError('validity dtype must be boolean')
    np.testing.assert_array_equal(motion['frame_id'],np.arange(378))
    np.testing.assert_array_equal(motion['anatomical_side_names'],['left','right'])
    np.testing.assert_array_equal(motion['human_to_physical'],[0,1])
    if len(domain['frames']) != 378 or [x['frame_id'] for x in domain['frames']] != list(range(378)):
        raise ValueError('domain timeline mismatch')
    times = np.array([x['capture_time']['original_time_fields']['ts'] for x in domain['frames']],dtype=np.int64)
    np.testing.assert_array_equal(motion['timestamp_ns'],times)
    if (domain['width'],domain['height'],domain['source_index']) != (1280,960,1):
        raise ValueError('frozen encoded domain changed')


def run(config):
    root,lane,index,ticks = authority(config)
    if config['frame_indices'] != list(FRAMES) or config['input_status'] != 'LEGACY_DIAGNOSTIC_ONLY':
        raise ValueError('fixed diagnostic scope required')
    refs = config['references']
    expected = {'motion':MOTION,'domain':DOMAIN,'mount':MOUNT,
                'predecessor':str(Path(MOTION).parent/'RESULT.json'),
                'asset_pin':'assets/robot/ROBOT_ASSET_PIN.json',
                'route':'_run/current/four_stream_visual_delivery_v5/attempts/attempt_0001/ROUTE_MANIFEST.json'}
    for key,path in expected.items():
        if checked_path(refs[key]['path'],root) != root/path:
            raise ValueError('input route mismatch: '+key)
    data = {key:read_pinned(ref,root) for key,ref in refs.items()}
    domain, mount, predecessor = [json.loads(data[k]) for k in ('domain','mount','predecessor')]
    if refs['motion'] != predecessor['outputs']['robot_r0']:
        raise ValueError('predecessor output mismatch')
    routes = [x for x in json.loads(data['route'])['sessions'] if x['session_id'] == SESSION]
    if len(routes) != 1 or routes[0]['domain_manifest'] != refs['domain']:
        raise ValueError('registered legacy route mismatch')
    if routes[0]['camera_source'] != domain['source_camera']:
        raise ValueError('registered camera route mismatch')
    for key,field in [('mesh','decoded_review_mesh'),('step','source_step')]:
        if any(refs[key][f] != mount[field][f] for f in ('path','bytes','sha256')):
            raise ValueError('CAD reference mismatch')
    camera = domain['source_camera']
    if camera != config['camera']:
        raise ValueError('camera pin mismatch')
    read_pinned(camera,Path('/mnt/data/egodata/datasets/ego'))
    with np.load(io.BytesIO(data['motion']),allow_pickle=False) as source:
        motion = {k:source[k] for k in source.files}
    validate_motion(motion,domain,predecessor)
    for ref in config['code_closure']:
        read_pinned(ref,root)
    # Runtime asset loader verifies the existing pin; explicitly pin consumed URDF/mesh bytes too.
    pin = json.loads(data['asset_pin'])
    for ref in pin['files']:
        if Path(ref['path']).suffix.lower() in ('.urdf','.stl','.obj','.dae'):
            read_pinned(ref,root)
    out = lane/'ASSEMBLY_CANARY_V1'
    required_code = {str(Path(__file__).resolve()),
        str(root/'src/chaoyang/pipeline/assembly_visibility_canary_v1.py'),
        str(root/'src/chaoyang/pipeline/v5_product.py'),
        str(root/'src/chaoyang/ops/run_four_stream_completion_sensor.py'),
        str(root/'src/chaoyang/pipeline/robot_renderer_eevee_fullchain.py'),
        str(root/'src/chaoyang/ops/render_tianji_kai_mount_proxy_audit.py')}
    if not required_code.issubset({str(checked_path(ref['path'],root)) for ref in config['code_closure']}):
        raise ValueError('missing required code closure')
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    import trimesh
    from chaoyang.pipeline.v5_product import ProductRobotRenderer
    mesh = trimesh.load(io.BytesIO(data['mesh']),file_type='stl',process=False)
    renderer = ProductRobotRenderer(root,motion,domain,include_adapter=True)
    reports=[]
    try:
        if renderer.mount_contract_path != root/MOUNT:
            raise ValueError('renderer mount drift')
        out.mkdir()
        for frame in FRAMES:
            report,layers = frame_report(renderer,frame,checked_path(refs['mesh']['path'],root),
                            mount['decoded_review_mesh']['mesh_scale_to_metre'],mesh.vertices)
            reports.append(report)
            with (out/f'frame_{frame:06d}_layers.npz').open('xb') as f:
                np.savez_compressed(f,rgb=layers.rgb,alpha=layers.alpha,component_id=layers.component_id,
                                    optical_depth_m=layers.optical_depth_m,depth_valid=layers.depth_valid)
    finally:
        renderer.close()
    _,_,index_after,ticks_after = authority(config)
    if index != index_after or ticks != ticks_after:
        raise ValueError('publisher changed during render')
    for ref in list(refs.values()) + config['code_closure']:
        read_pinned(ref,root)
    read_pinned(camera,Path('/mnt/data/egodata/datasets/ego'))
    for ref in pin['files']:
        if Path(ref['path']).suffix.lower() in ('.urdf','.stl','.obj','.dae'):
            read_pinned(ref,root)
    artifacts=[]
    for path in sorted(out.glob('*.npz')):
        raw=path.read_bytes()
        artifacts.append(dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
    result=dict(schema_version='ASSEMBLY_VISIBILITY_CANARY_V1',task_id=TASK,execution='EXECUTED',
                quality='NOT_EVALUATED',adoption='NOT_ADOPTED',input_status='LEGACY_DIAGNOSTIC_ONLY',
                physical_mount_verified=False,environment_occlusion_evaluated=False,gpu_used=False,
                inputs=config,frames=reports,artifacts=artifacts,panel_design=PANEL_DESIGN)
    with (out/'RESULT.json').open('x',encoding='utf8') as f:
        json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    return result

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    args=parser.parse_args(argv)
    result=run(json.loads(args.config.read_text(encoding='utf8')))
    print(json.dumps({'execution':result['execution'],'quality':result['quality']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
