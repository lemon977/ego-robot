"""Wrapper-level CPU regressions: no model or renderer calls."""
import json
from pathlib import Path
import numpy as np
import pytest
from chaoyang.inpainting_candidate import v1_mco_worker as worker,automation
from chaoyang.inpainting_candidate.contracts import file_ref
from tests.test_inpainting_v1_automation_flow import project,save
from tests.unit.test_inpainting_repair_rounds_entry import specs
from tests.unit.test_inpainting_context_exclusion_entry import recipe


def synthetic_result(packet):
    return {'schema':'inpainting-v1-compatible-automation-result-v1','status':'SEALED_COMPLETE',
        'task_id':packet['task_id'],'run_signature':packet['run_signature'],
        'model_calls':0,'render_calls':0,'artifacts':[],'candidate_adopted':False,'control_ground_truth':False,
        'sessions':[{'session_id':'S','native_frames':3,'execution':'SEALED_COMPLETE_ALL_SIX_STAGES',
            'reused_stages':['Source','Motion','Mask','Clean','Render','Composite']}],
        'robot_video_produced':True}


def test_all_six_reuse_wrapper_seals_own_graph_without_copying_pixels(monkeypatch,tmp_path):
    packet={'mode':'Automation','component_config':{},'output_dir':str(tmp_path),'task_id':'synthetic','run_signature':'a'*64}
    def execute(*args,**kwargs):
        result=synthetic_result(packet)
        (tmp_path/'AUTOMATION_STAGE.json').write_text(json.dumps(result));return result
    monkeypatch.setattr(worker,'_authority',lambda p:lambda:None)
    monkeypatch.setattr(automation,'execute_automation',execute)
    assert worker.execute_v1_mco_stage(packet)==0
    result=json.loads((tmp_path/'MODEL_RESULT.json').read_text())
    assert result['model_calls']==result['render_calls']==0
    assert result['artifacts']==[file_ref(tmp_path/'AUTOMATION_STAGE.json')]
    assert result['sessions'][0]['completed_frame_indices']==[0,1,2]
    assert result['reused_pixels_are_new_outputs']is False


def test_missing_or_changed_sealed_graph_is_rejected(monkeypatch,tmp_path):
    packet={'mode':'Automation','component_config':{},'output_dir':str(tmp_path),'task_id':'synthetic','run_signature':'a'*64}
    monkeypatch.setattr(worker,'_authority',lambda p:lambda:None)
    monkeypatch.setattr(automation,'execute_automation',lambda *args,**kwargs:synthetic_result(packet))
    with pytest.raises(ValueError,match='SEALED_GRAPH_REQUIRED'):worker.execute_v1_mco_stage(packet)
    (tmp_path/'AUTOMATION_STAGE.json').write_text('{}')
    with pytest.raises(ValueError,match='SEALED_GRAPH_REQUIRED'):worker.execute_v1_mco_stage(packet)


def test_cpu_seam_dispatch_does_not_fall_into_neural_path(monkeypatch):
    from chaoyang.inpainting_candidate import v1_context_seam
    called=[]
    monkeypatch.setattr(worker,'_authority',lambda p:lambda:None)
    monkeypatch.setattr(v1_context_seam,'execute_candidate',lambda p,**kw:called.append(p)or 0)
    packet={'config':{'cpu_context_seam':True}}
    assert worker.execute_v1_mco_stage(packet)==0 and called==[packet]


def test_publisher_requires_exact_zero_model_cpu_seam_spec(monkeypatch,tmp_path):
    import importlib.util
    from datetime import datetime,timedelta,timezone
    source=Path(__file__).resolve().parents[2]/'scripts/maintenance/inpainting_v1_mco_task.py'
    definition=importlib.util.spec_from_file_location('_seam_publisher_test',source)
    module=importlib.util.module_from_spec(definition);definition.loader.exec_module(module)
    code=tmp_path/'code.py';code.write_text('# synthetic\n')
    monkeypatch.setattr(module,'REPO_ROOT',tmp_path);monkeypatch.setattr(module,'INFRA_PATH',code)
    spec={'spec_schema':module.SPEC_SCHEMA,'task_id':'inpainting_v1_mco_synthetic_seam','mode':'Clean',
        'config':{'cpu_context_seam':True,'algorithm':'POISSON_WORKING_GRADIENT_RAW_FIXED_BOUNDARY'},
        'budgets':{'wall_seconds_max':60,'cpu_threads_per_worker':2,'gpu_count':0,'model_calls_max':0,
            'render_calls_max':0,'training_steps_max':0,'output_bytes_estimate':1024},
        'model_weights':[],'global_deadline_at':(datetime.now(timezone.utc)+timedelta(seconds=90)).isoformat(),
        'output_dir':str(tmp_path/'out'),'review_dir':str(tmp_path/'review'),
        'sessions':[{'session_key':'S','native_frames':3}],'inputs':{'synthetic':file_ref(code)},
        'runtime_code':[file_ref(code)],'calibration':{'status':'ABSENT'},'schema':{},'read_set':[],
        'infrastructure':file_ref(code)}
    module.validate_spec(spec)
    spec['budgets']['gpu_count']=1
    with pytest.raises(ValueError,match='CARD_BUDGET'):module.validate_spec(spec)
    spec['budgets']['gpu_count']=0;spec['config']['cpu_raw_donor']=True
    with pytest.raises(ValueError,match='EXCLUSIVITY'):module.validate_spec(spec)


def test_opt_in_boundary_changes_only_clean_consumers_and_binds_base_work(project,specs):
    recipe(project)
    for name in ['v1_boundary_builder.py','v1_context_seam.py']:
        save(project.root/'src/chaoyang/inpainting_candidate'/name,{'synthetic':name})
    req=automation.build_v1_request(project_root=project.root,session=project.raw/'050',repair_rounds=2)
    ref=save(project.root/'REQUEST.json',req);context=specs.clean_reference_execution_context(project.root,wall_seconds=120)
    base=specs.compatible_spec(project.root,req,ref,execution_context=context,
        clean_reference_policy='EXCLUDE_UNCERTAIN_EQUIPMENT_RGB')
    new=specs.compatible_spec(project.root,req,ref,execution_context=context,
        clean_reference_policy='EXCLUDE_UNCERTAIN_EQUIPMENT_RGB',clean_write_policy='POISSON_RGB_UNCERTAIN_BOUNDARY_VETO')
    changes={s for s in base['component_signatures']['050']if base['component_signatures']['050'][s]!=new['component_signatures']['050'][s]}
    assert changes=={'Clean','Composite'}
    assert new['base_clean_signatures']['050']==base['component_signatures']['050']['Clean']
    assert new['clean_work_reuse']=={}


def test_cli_boundary_option_requires_explicit_context_candidate(project,monkeypatch):
    from types import SimpleNamespace
    from chaoyang.cli import inpainting_submit
    from chaoyang import paths
    from chaoyang.governance import inpainting_candidate_registration as registration
    monkeypatch.setattr(paths,'get_paths',lambda:SimpleNamespace(repo_root=project.root));seen=[]
    monkeypatch.setattr(registration,'submit_v1_compatible_request',lambda req,**kw:seen.append(kw)or{'status':'SYNTHETIC'})
    args=['--session',str(project.raw/'050'),'--mode','v1-compatible','--repair-rounds','2']
    inpainting_submit.main([*args,'--clean-reference-policy','exclude-uncertain-equipment',
        '--clean-write-policy','uncertain-boundary-veto','--preview'])
    assert seen[0]['clean_write_policy']=='POISSON_RGB_UNCERTAIN_BOUNDARY_VETO'
    with pytest.raises(SystemExit):
        inpainting_submit.main([*args,'--clean-write-policy','gradient-boundary'])
