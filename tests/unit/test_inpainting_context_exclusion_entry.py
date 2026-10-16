import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from chaoyang.inpainting_candidate import automation as auto
from chaoyang.inpainting_candidate.contracts import STAGES,file_ref
from tests.test_inpainting_v1_automation_flow import authority,project,save,synthetic_builders
from tests.unit.test_inpainting_repair_rounds_entry import specs

POLICY='EXCLUDE_UNCERTAIN_EQUIPMENT_RGB'

def recipe(project):
    for name in ['v1_context_builder.py','v1_context_exclusion.py','v1_equipment_rgb.py','source_frames.py']:
        save(project.root/'src/chaoyang/inpainting_candidate'/name,{'synthetic':name})
    save(project.root/'src/chaoyang/exact78_baseline/mask.py',{'synthetic':'mask'})
    cache=save(project.root/'inpainting_optimization/clean_context_8h/STAGE_CACHE.json',
        {'schema':'inpainting-context-explicit-stage-cache-v1','sessions':[]})
    save(project.root/'inpainting_optimization/clean_context_8h/EXECUTION_RECIPE.json',{
        'schema':'inpainting-v1-clean-reference-execution-recipe-v1','clean_reference_policy':POLICY,
        'allowed_repair_rounds':[2,3],'plan_revision':'SYNTHETIC_CONTEXT','objective':'Synthetic optional candidate',
        'output_root':'_run/new_context','review_root':'review/new_context','wall_seconds_max':3600,'stage_cache':cache})

def test_opt_in_changes_only_clean_composite_not_source_motion_mask_render(project,specs):
    recipe(project);r=auto.build_v1_request(project_root=project.root,session=project.raw/'050',repair_rounds=2)
    ref=save(project.root/'REQUEST.json',r)
    old=specs.compatible_spec(project.root,r,ref,execution_context=specs.temporal_execution_context(project.root,wall_seconds=120))
    context=specs.clean_reference_execution_context(project.root,wall_seconds=120)
    # Passing normalized context twice must not fail relative/absolute roots.
    normalized=specs._execution_context(project.root,context)
    new=specs.compatible_spec(project.root,r,ref,execution_context=normalized,clean_reference_policy=POLICY)
    assert {s for s in STAGES if old['component_signatures']['050'][s]!=new['component_signatures']['050'][s]}=={'Clean','Composite'}
    assert old['task_id']!=new['task_id'] and new['component_config']['Clean']['reference_policy']==POLICY

def test_candidate_cannot_use_unbound_recipe_or_default_context(project,specs):
    recipe(project);r=auto.build_v1_request(project_root=project.root,session=project.raw/'050',repair_rounds=2)
    ref=save(project.root/'REQUEST.json',r)
    with pytest.raises(ValueError,match='REFERENCE'):
        specs.compatible_spec(project.root,r,ref,execution_context=specs.temporal_execution_context(project.root),clean_reference_policy=POLICY)

def test_cli_policy_is_explicit_and_not_raw_auto(project,monkeypatch,capsys):
    from chaoyang.cli import inpainting_submit
    from chaoyang import paths
    from chaoyang.governance import inpainting_candidate_registration as registration
    monkeypatch.setattr(paths,'get_paths',lambda:SimpleNamespace(repo_root=project.root));seen=[]
    monkeypatch.setattr(registration,'submit_v1_compatible_request',lambda req,**kw:seen.append(kw)or{'status':'SYNTHETIC'})
    inpainting_submit.main(['--session',str(project.raw/'050'),'--mode','v1-compatible','--repair-rounds','3',
        '--clean-reference-policy','exclude-uncertain-equipment','--preview'])
    assert seen[0]['clean_reference_policy']==POLICY
    with pytest.raises(SystemExit):inpainting_submit.main(['--session','/not/read','--clean-reference-policy','exclude-uncertain-equipment'])

def test_clean_builder_writes_legacy_consumable_arrays_and_fixed_domain(monkeypatch,tmp_path):
    from PIL import Image
    from chaoyang.inpainting_candidate import v1_context_builder as module,v1_equipment_rgb,source_frames
    from chaoyang.pick_place_1008 import visual
    raw=np.full((2,16,24,3),80,np.uint8);Image.fromarray(raw[0]).save(tmp_path/'seed.png')
    seed=save(tmp_path/'seed.json',{'session_id':'S','raw_frame_index':0,'image':file_ref(tmp_path/'seed.png')})
    masks={k:np.zeros(raw.shape[:3],bool)for k in ['core','sam_core','equipment','removal','effective_removal','protected_visible','cup_observed']}
    masks['effective_removal'][:,3:8,3:8]=True
    mask_report=save(tmp_path/'MASK_REPORT.json',{'seed':seed,'metadata':{}})
    np.savez_compressed(tmp_path/'MASK_ARRAYS.npz',**masks)
    mask_receipt=save(tmp_path/'MASK_RECEIPT.json',{'outputs':[mask_report,file_ref(tmp_path/'MASK_ARRAYS.npz')]})
    packet={'project_root':str(tmp_path),'component_config':{'Clean':{'reference_policy':POLICY}},'budgets':{'propainter_calls_max':1}}
    builder=module.ContextExclusionBuilders(packet,lambda:None)
    monkeypatch.setattr(builder,'_rgb',lambda upstream:raw)
    monkeypatch.setattr(builder,'_mapping',lambda upstream:[{}]*2)
    monkeypatch.setattr(builder,'_seal',lambda *args,**kw:{'extra':args[-1]})
    class Painter:
        settings={};last_validation={'status':'PASS'}
        def __init__(self,parent):pass
        def inpaint(self,*args):return np.full_like(raw,190)
    monkeypatch.setattr(visual,'ProtectedProPainter',Painter)
    monkeypatch.setattr(builder,'_models',lambda stage:{'propainter':None})
    monkeypatch.setattr(v1_equipment_rgb,'extend',lambda raw,masks,seed,**kw:(masks,np.zeros(raw.shape[:3],bool),{}))
    monkeypatch.setattr(source_frames,'write_review_video',lambda path,*args:Path(path).write_bytes(b'SYNTHETIC'))
    upstream={'Mask':mask_receipt,'Source':seed}
    builder.clean(row={'session_id':'S','seed':seed},output_dir=tmp_path/'Clean',signature='a'*64,upstream=upstream,request={})
    with np.load(tmp_path/'Clean/CLEAN_RGB.npz')as z:
        assert set(z.files)=={'rgb','effective_removal'}
        assert np.array_equal(z['rgb'][~masks['effective_removal']],raw[~masks['effective_removal']])
    assert json.loads((tmp_path/'Clean/CLEAN_REPORT.json').read_text())['primary_clean_variant']=='CONTEXT_EXCLUSION'
    assert builder.clean_calls==1


@pytest.fixture
def boundary_cache_case(project,specs):
    """Real planner/receipt validation, fake CPU producers, no model inference."""
    from chaoyang.inpainting_candidate.automation_worker import _arrays,_outputs,_save_arrays
    recipe(project)
    for name in ['v1_boundary_builder.py','v1_context_seam.py']:
        save(project.root/'src/chaoyang/inpainting_candidate'/name,{'synthetic':name})
    request=auto.build_v1_request(project_root=project.root,session=project.raw/'050',repair_rounds=2)
    request_ref=save(project.root/'REQUEST.json',request)
    context=specs.clean_reference_execution_context(project.root,wall_seconds=120)
    base=specs.compatible_spec(project.root,request,request_ref,execution_context=context,clean_reference_policy=POLICY)

    def parent(name,*,prefix=None,reuse=None,clean_config=None):
        packet=deepcopy(base)
        output=project.root/name;output.mkdir()
        packet.update(task_id='synthetic_'+name,project_root=str(project.root),output_dir=str(output),
            request_path=request_ref['path'],run_signature=('a'if name=='A'else'b')*64,
            writer={'fencing_token':'synthetic-only'},reuse_stages={'050':reuse}if reuse else{})
        if clean_config is not None:
            packet['component_config']['Clean'].update(clean_config)
            packet['component_signatures']=auto.component_signatures(request,packet['component_code'],
                packet['component_weights'],packet['component_config'])
        builders=synthetic_builders([])
        original_mask,original_clean=builders['Mask'],builders['Clean']

        def seal_extra(kwargs,extra=None):
            root=kwargs['output_dir'];path=root/'RECEIPT.json'
            receipt=json.loads(path.read_text())
            receipt.update(extra or{},outputs=[file_ref(p)for p in sorted(root.iterdir())if p.name!='RECEIPT.json'])
            save(path,receipt)

        def mask(**kwargs):
            original_mask(**kwargs)
            raw=_arrays(_outputs(kwargs['upstream']['Source'])['RAW_RGB.npz'])['rgb']
            masks={name:np.zeros(raw.shape[:3],bool)for name in ['core','sam_core','equipment','removal',
                'effective_removal','protected_visible','cup_observed','unknown']}
            masks['effective_removal'][:,4:12,5:18]=True
            masks['protected_visible'][:,7:9,10:12]=True
            masks['effective_removal'][masks['protected_visible']]=False
            masks['removal'][:]=masks['effective_removal'];masks['core'][:]=masks['effective_removal']
            _save_arrays(kwargs['output_dir']/'MASK_ARRAYS.npz',**masks)
            save(kwargs['output_dir']/'MASK_REPORT.json',{'seed':kwargs['row']['seed'],'metadata':{'synthetic':True}})
            seal_extra(kwargs)

        def clean(**kwargs):
            original_clean(**kwargs)
            root=kwargs['output_dir'];upstream=kwargs['upstream']
            raw=_arrays(_outputs(upstream['Source'])['RAW_RGB.npz'])['rgb']
            masks=_arrays(_outputs(upstream['Mask'])['MASK_ARRAYS.npz'])
            working=np.zeros(raw.shape[:3],bool);working[:,2:14,2:22]=True
            working[masks['protected_visible']]=False
            uncertain=np.zeros_like(working);uncertain[:,2:5,2:7]=True
            work=raw.copy();work[working]=100;work[:,5:7,6:9]=150
            legacy=raw.copy();legacy[masks['effective_removal']]=work[masks['effective_removal']]
            _save_arrays(root/'WORKING_CLEAN_RGB.npz',rgb=work)
            _save_arrays(root/'CLEAN_RGB.npz',rgb=legacy,effective_removal=masks['effective_removal'])
            _save_arrays(root/'CONTEXT_WORKING_MASKS.npz',removal=working,effective_removal=working,
                write_effective_removal=masks['effective_removal'],rgb_uncertain_support=uncertain)
            save(root/'CONTEXT_REFERENCE_AUDIT.json',{'source_receipt':upstream['Source'],
                'mask_receipt':upstream['Mask'],'seed':kwargs['row']['seed'],'reference_policy':POLICY,'synthetic':True})
            save(root/'CLEAN_REPORT.json',{'primary_clean_variant':'CONTEXT_EXCLUSION',
                'reference_policy':POLICY,'propainter_calls':0,'synthetic':True})
            seal_extra(kwargs,{'primary_clean_variant':'CONTEXT_EXCLUSION','reference_policy':POLICY})

        builders.update(Mask=mask,Clean=clean)
        model=auto.execute_automation(packet,authorization=authority(packet),builders=builders)
        references=model['sessions'][0]['stage_receipts']
        packet_ref=save(project.root/'tasks'/name/'TASK_PACKET.json',packet)
        terminal={'task_id':packet['task_id'],'run_signature':packet['run_signature'],
                  'status':'FAILED_RUNTIME_FINAL'if prefix else'PASSED'}
        if not prefix:
            terminal.update(model_result=save(output/'MODEL_RESULT.json',model),model_result_consumable=True)
        candidate={'packet':packet_ref,'result':save(output/'RESULT.json',terminal)}
        if prefix:candidate['stage_receipts']={'050':{stage:references[stage]for stage in prefix}}
        return SimpleNamespace(candidate=candidate,refs=references)

    def boundary(parents,*,as_list=False):
        candidates=[parent.candidate for parent in parents]
        manifest=save(project.root/'RESUME.json',candidates if as_list else{
            'schema':'inpainting-v1-resume-candidates-v1','parents':candidates})
        result=specs.compatible_spec(project.root,request,request_ref,execution_context=context,
            clean_reference_policy=POLICY,clean_write_policy='POISSON_RGB_UNCERTAIN_BOUNDARY_VETO',
            resume_manifest=manifest)
        assert result['component_config']['Clean']=={**base['component_config']['Clean'],
            'write_boundary_policy':'POISSON_RGB_UNCERTAIN_BOUNDARY_VETO'}
        return result

    return SimpleNamespace(parent=parent,boundary=boundary,base=base,request=request,root=project.root)


@pytest.mark.parametrize('matching_clean',[True,False])
@pytest.mark.parametrize('as_list',[True,False])
def test_work_cache_is_pinned_to_final_source_mask_not_fuller_parent(boundary_cache_case,matching_clean,as_list):
    case=boundary_cache_case
    prefix=['Source','Motion','Mask','Render']+(['Clean']if matching_clean else[])
    a=case.parent('A',prefix=prefix);b=case.parent('B')
    # Same semantic signatures are insufficient: the two legitimate executions
    # have different immutable upstream paths/receipt bytes.
    assert a.refs['Source']!=b.refs['Source']and a.refs['Mask']!=b.refs['Mask']
    spec=case.boundary([a,b],as_list=as_list);selected=spec['reuse_stages']['050']
    assert set(selected)=={'Source','Motion','Mask','Render'}
    assert selected=={stage:a.refs[stage]for stage in selected}
    if matching_clean:
        assert spec['clean_work_reuse']=={'050':{'receipt':a.refs['Clean'],
            'expected_signature':case.base['component_signatures']['050']['Clean']}}
    else:
        assert spec['clean_work_reuse']=={}


def test_work_cache_only_needs_source_mask_match_not_motion_render(boundary_cache_case):
    case=boundary_cache_case
    a=case.parent('A',prefix=['Source','Motion','Mask','Render'])
    b=case.parent('B',reuse={stage:a.refs[stage]for stage in ['Source','Mask']})
    assert b.refs['Source']==a.refs['Source']and b.refs['Mask']==a.refs['Mask']
    assert b.refs['Motion']!=a.refs['Motion']and b.refs['Render']!=a.refs['Render']
    spec=case.boundary([a,b]);selected=spec['reuse_stages']['050']
    assert selected=={stage:a.refs[stage]for stage in selected}
    assert spec['clean_work_reuse']['050']['receipt']==b.refs['Clean']


def test_work_cache_accepts_failed_explicit_clean_only_catalog(boundary_cache_case):
    case=boundary_cache_case
    a=case.parent('A',prefix=['Source','Motion','Mask','Render'])
    b=case.parent('B',prefix=['Clean'],reuse={stage:a.refs[stage]for stage in ['Source','Mask']})
    spec=case.boundary([a,b]);selected=spec['reuse_stages']['050']
    assert selected=={stage:a.refs[stage]for stage in selected}
    assert spec['clean_work_reuse']['050']['receipt']==b.refs['Clean']


@pytest.mark.parametrize('second_matches',[True,False])
def test_work_cache_skips_different_base_signature_even_with_matching_upstream(boundary_cache_case,second_matches):
    case=boundary_cache_case
    a=case.parent('A',prefix=['Source','Motion','Mask','Clean','Render'],
        clean_config={'synthetic_algorithm_change':True})
    b=case.parent('B',reuse={stage:a.refs[stage]for stage in ['Source','Mask']}if second_matches else None)
    spec=case.boundary([a,b]);selected=spec['reuse_stages']['050']
    assert selected=={stage:a.refs[stage]for stage in selected}
    if second_matches:assert spec['clean_work_reuse']['050']['receipt']==b.refs['Clean']
    else:assert spec['clean_work_reuse']=={}


def test_work_cache_first_matching_clean_wins_without_changing_final_plan(boundary_cache_case):
    case=boundary_cache_case
    a=case.parent('A',prefix=['Source','Motion','Mask','Clean','Render'])
    b=case.parent('B',reuse={stage:a.refs[stage]for stage in ['Source','Mask']})
    spec=case.boundary([a,b]);selected=spec['reuse_stages']['050']
    assert selected=={stage:a.refs[stage]for stage in selected}
    assert spec['clean_work_reuse']['050']['receipt']==a.refs['Clean']


def test_matching_work_selection_reaches_real_boundary_builder_with_zero_pp(boundary_cache_case,monkeypatch):
    from chaoyang.inpainting_candidate.v1_boundary_builder import BoundaryContextBuilders
    from chaoyang.inpainting_candidate import source_frames
    case=boundary_cache_case
    a=case.parent('A',prefix=['Source','Motion','Mask','Clean','Render']);b=case.parent('B')
    spec=case.boundary([a,b]);spec['project_root']=str(case.root)
    spec['budgets']['propainter_calls_max']=0
    builder=BoundaryContextBuilders(spec,lambda:None)
    monkeypatch.setattr(builder,'_models',lambda stage:pytest.fail('WORK_CACHE_MUST_NOT_LOAD_A_MODEL'))
    monkeypatch.setattr(source_frames,'write_review_video',lambda path,*args:Path(path).write_bytes(b'SYNTHETIC_VIDEO'))
    receipt=builder.clean(row=case.request['sessions'][0],output_dir=case.root/'BoundaryClean',
        signature=spec['component_signatures']['050']['Clean'],upstream=spec['reuse_stages']['050'],request=case.request)
    assert builder.clean_calls==receipt['propainter_calls']==0
    assert receipt['work_cache']['consumed']is True
    assert receipt['actual_inputs']['base_clean_receipt']==a.refs['Clean']
    assert not(case.root/'BoundaryClean/Work').exists()
