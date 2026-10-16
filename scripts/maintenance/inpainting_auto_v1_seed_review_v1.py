"""CPU-only evaluation/publication of sealed Mask; never a producer/model entry."""
from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from chaoyang.governance.common import artifact_ref
from chaoyang.inpainting_candidate.source_frames import write_review_video, atomic_json
from chaoyang.storage_relocation.pick_place_original199_v1 import Original199Reader

ROOT = Path(__file__).resolve().parents[2]
TASK = 'inpainting_auto_v1_seed_review_20261013'
OUT = ROOT / '_run' / TASK / 'attempt_0001'
MODEL = ROOT / '_run/inpainting_auto_v1_seed_mask_20261013/attempt_0001'
SOURCE = ROOT / '_run/inpainting_generalization_g0_publication_20261013/attempt_0001/sessions'
OLD = ROOT / '_run/inpainting_post_g1_glove_concept_mask_a01_20261013/attempt_0001/sessions'
FIXED = {'001':[0,50,119,132,202,237], '050':[0,28,64,91,96,127]}
HIST = {
 '001':('exact78_pick_place_development_full_v9_reverse_boundary_revision2_20261009',2301201,'cdd5f79198cf3d037154dab37577625068804893121de915b5210ddf44e0681c'),
 '050':('exact78_pick_place_production_batch05_full_v9_reverse_boundary_revision2_20261009',1427374,'27065c7e0040d85f4f88e5e517a7010e879be88682a7808ff5f02826bdb3b2f8')}

def main():
    packet = json.loads((ROOT / 'tasks/current' / TASK / 'TASK_PACKET.json').read_text())
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location('_existing_review_infra', ROOT/'scripts/humanego_ict/production_resume_runner_v1.py')
    infra = module_from_spec(spec); spec.loader.exec_module(infra)
    _, state, index = infra._snapshot(); infra._own_row(packet,state,index)
    binding = artifact_ref(ROOT/'_run/maintenance/pick_place_1008_nas_export_20261010_v1/R0_READONLY_VALIDATION_152328_9dfee33b/STORAGE_BINDING_V1.json')
    reader = Original199Reader(binding)
    font = ImageFont.truetype('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',25)
    rows=[]
    for key in FIXED:
        native = 238 if key=='001' else 128
        source = SOURCE/key/'Source'; current=MODEL/'sessions'/key/'Mask'
        with np.load(source/'RAW_RGB.npz',allow_pickle=False) as a:
            rgb=a['rgb'].copy()
        mapping=json.loads((source/'FRAME_MAP.json').read_text())
        with np.load(current/'MASK_ARRAYS.npz',allow_pickle=False) as a:
            new={k:a[k].copy() for k in ('effective_removal','protected_visible','unknown')}
        with np.load(OLD/key/'Mask/MASK_ARRAYS.npz',allow_pickle=False) as a:
            old={k:a[k].copy() for k in ('effective_removal','protected_visible','unknown')}
        t,b,s=HIST[key]
        original_ref={'path':str(ROOT/'_run/current'/t/'attempts/attempt_0001/candidate/sessions'/key/'masks.npz'),'bytes':b,'sha256':s}
        packed=reader.arrays(original_ref)
        original={k:np.unpackbits(packed[k],axis=1,count=480*640).reshape(native,480,640).astype(bool) for k in new}
        assert len(rgb)==len(mapping)==native and all(a.shape==rgb.shape[:3] for a in new.values())
        def overlay(arrays,i):
            pixels=rgb[i].copy()
            for name,color in [('effective_removal',(255,60,60)),('protected_visible',(40,235,90)),('unknown',(255,215,0))]:
                m=arrays[name][i]; pixels[m]=(pixels[m]*.55+np.array(color)*.45).astype(np.uint8)
            return pixels
        # Full-native comparison, unchanged original source timing, no output-r FPS.
        frames=np.zeros((native,548,1920,3),np.uint8)
        labels=['原始图像','原199 V1去除区','自动实例提示候选（非基线）']
        for i in range(native):
            frames[i,68:,:640]=rgb[i]
            frames[i,68:,640:1280]=overlay(original,i)
            frames[i,68:,1280:]=overlay(new,i)
            canvas=Image.fromarray(frames[i]); draw=ImageDraw.Draw(canvas)
            for col,label in enumerate(labels): draw.text((col*640+12,4),label,font=font,fill='white')
            draw.text((12,35),f'{key} 帧 {i}/{native-1}   红：删除  绿：保护  黄：未知   仅Mask对照，无新Clean／Robot',font=font,fill='white')
            frames[i]=np.asarray(canvas)
            if i in FIXED[key]: canvas.save(OUT/f'{key}_comparison_{i:04d}.png')
        video=write_review_video(ROOT/'review/inpainting_optimization'/f'AUTO_V1_SEED_{key}_Mask_Comparison.mp4',frames,mapping)
        probes = ([(i,275,80,'background_real_robot') for i in FIXED[key]] if key=='001' else
                  [(i,85,355,'left_white_palm') for i in FIXED[key]]+
                  [(127,434,310,'right_gray_finger'),(0,535,320,'right_mount'),(64,20,330,'left_mount')])
        points=[]
        for i,x,y,kind in probes:
            points.append({'frame':i,'xy':[x,y],'kind':kind,'not_ground_truth':True,
                'original_v1':{k:bool(a[i,y,x]) for k,a in original.items()},
                'previous_automatic':{k:bool(a[i,y,x]) for k,a in old.items()},
                'automatic_v1_seed':{k:bool(a[i,y,x]) for k,a in new.items()}})
        issues=[]
        for i in FIXED[key]:
            if key=='001':
                descriptions=['HIGH：左右控制器／白安装件及腕盒仍有可见漏删；不是全设备去除通过。']
                reviewer='/root/mask_review_one_hour'
            else:
                descriptions=['HIGH：左侧画面边缘白安装件仍有漏删。']
                if i in (0,28,64,127): descriptions+=['HIGH：右腕白盒／连接板局部未进删除域。']
                if i==127: descriptions+=['HIGH：右手灰指套有可见漏删。']
                reviewer='/root'
            issues.append({'frame':i,'evidence':artifact_ref(current/f'anchor_{i:04d}.png'),
                'comparison':artifact_ref(OUT/f'{key}_comparison_{i:04d}.png'),
                'actual_image_review':True,'reviewer':reviewer,'severity':'HIGH',
                'visible_defects':descriptions,'hidden_regions':'UNKNOWN','continuity':'NOT_FULLY_REVIEWED'})
        rows.append({'session_key':key,'native_frames':native,'source':artifact_ref(source/'RECEIPT.json'),
             'new_mask':artifact_ref(current/'RECEIPT.json'),'actual_arrays':artifact_ref(current/'MASK_ARRAYS.npz'),
             'prior_automatic_arrays':artifact_ref(OLD/key/'Mask/MASK_ARRAYS.npz'),
             'original_v1_evaluator_only':original_ref,'video':video,'fixed_frames':FIXED[key],
             'seen_frames':6,'unseen_frames':native-6,'issues':issues,'point_probes':points,
             'original_vs_candidate_effective_removal_different_pixels':int(np.count_nonzero(original['effective_removal']^new['effective_removal'])),
             'previous_automatic_vs_candidate_effective_removal_different_pixels':int(np.count_nonzero(old['effective_removal']^new['effective_removal'])),
             'difference_count_is_not_accuracy':True,'visual_quality':'FAILED','clean_executions':0,
             'human_review':'PENDING','hidden_regions':'UNKNOWN'})
        print(json.dumps({'session':key,'video':video,'probes':points},ensure_ascii=False),flush=True)
        del frames,rgb,packed,original,new,old
    result={'schema':'inpainting-auto-v1-seed-actual-review-v1','task_id':TASK,'sessions':rows,
       'execution_completed_sessions':2,'native_frames':366,'ai_actually_seen_frames':12,'ai_unseen_frames':354,
       'visual_qualified_sessions':0,'visual_quality':'FAILED','quality_pass':False,
       'clean_executions':0,'candidate_adopted':False,'producer_historical_mask_reads':0,
       'evaluator_historical_mask_reads':True,'files_deleted':0,'training_steps':0,
       'human_review':'PENDING','hidden_regions':'UNKNOWN','production_default':'INPAINTING_RAW_TO_ROBOT_V1',
       'model_result':artifact_ref(MODEL/'MODEL_RESULT.json'),'readonly_binding':binding}
    atomic_json(OUT/'ACTUAL_REVIEW.json',result)

if __name__=='__main__': main()
