import copy
import pytest
from chaoyang.pipeline.motion_four_certificate_ledger_v1 import generate,KINDS

def fixture():
    ref={'path':'review.json','sha256':'b'*64,'bytes_verified':True,'stable_read':True}
    base={'frame_id':47,'session_id':'031','source_sha256':'a'*64,'status':'PASS','reviewed_independent':True,'evidence_refs':[ref],'capabilities':dict.fromkeys(('position','rotation','fingers'),'PASS'),'replacement_required':True}
    return dict(session_id='031',source_sha256='a'*64,frame_ids=[47],source_rows=[{'frame_id':47,'side':s,'valid':True,'kind':'MODEL_INFERENCE'} for s in ('left','right')],certificates={k:[dict(base,side=s) for s in ('left','right')] for k in KINDS},verified_refs=[ref])

def test_qualified_model_and_not_product():
    d=generate(**fixture()); assert d['disposition_counts']['ROBOT']==1; assert not d['product_completed']

def test_missing_all_never_complete():
    a=fixture(); a['certificates']={}; d=generate(**a)
    assert d['disposition_counts']['ORIGINAL_FRAME_FALLBACK']==1 and len(d['missing_or_failed_evidence'])==8
    assert not d['quality_adopted']

def test_missing_visibility_not_absent_or_partial():
    a=fixture(); a['certificates']['visibility']=a['certificates']['visibility'][1:]
    d=generate(**a); assert d['disposition_counts']['ORIGINAL_FRAME_FALLBACK']==1

def test_source_bad_and_partial_independence():
    a=fixture(); a['source_rows'][0]['valid']=False
    d=generate(**a); assert d['disposition_counts']['PARTIAL_PRESERVE']==1
    assert d['rows'][0]['disposition']['robot_replaced_sides']==('right',)

@pytest.mark.parametrize('kind',['TIME_INTERPOLATION','HOLD','COPIED_OTHER_SIDE','UNKNOWN'])
def test_unqualified_source_kinds(kind):
    a=fixture(); a['source_rows'][0]['kind']=kind
    assert generate(**a)['disposition_counts']['PARTIAL_PRESERVE']==1

@pytest.mark.parametrize('field,value',[('session_id','007'),('source_sha256','c'*64)])
def test_binding_rejected(field,value):
    a=fixture(); a['certificates']['motion'][0][field]=value
    with pytest.raises(ValueError,match='BINDING'): generate(**a)

def test_self_evidence_unstable_and_missing_rotation_fail():
    for change in ('self','unstable','rotation'):
        a=fixture()
        if change=='self': a['certificates']['motion'][0]['evidence_refs']=[{'path':'source','sha256':'a'*64}]
        elif change=='unstable': a['verified_refs'][0]['stable_read']=False
        else: a['certificates']['motion'][0]['capabilities']['rotation']='UNKNOWN'
        assert generate(**a)['disposition_counts']['ROBOT']==0

def test_duplicate_timeline_and_missing_side_rejected():
    a=fixture(); a['frame_ids']=[47,47]
    with pytest.raises(ValueError): generate(**a)
    a=fixture(); a['source_rows'].pop()
    with pytest.raises(ValueError): generate(**a)

def test_missing_render_not_implicitly_valid():
    a=fixture(); a['certificates'].pop('render')
    assert generate(**a)['disposition_counts']['ORIGINAL_FRAME_FALLBACK']==1
