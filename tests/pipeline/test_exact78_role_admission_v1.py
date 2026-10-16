import pytest
from chaoyang.pipeline.exact78_role_admission_v1 import check_required_roles


def sample(equipment):
    return {'rows':[dict(frame_id=i,instance_counts=dict(human=2,equipment=e)) for i,e in enumerate(equipment)]}


def test_all_frame_equipment_loss_rejected():
    r=check_required_roles(sample([0]*171),['human','equipment'],171)
    assert r['status']=='HOLD_REQUIRED_ROLE_ABSENT'
    assert r['missing_required_roles']==['equipment']


def test_some_presence_does_not_claim_quality():
    r=check_required_roles(sample([0,1,0]),['human','equipment'],3)
    assert not r['missing_required_roles']
    assert r['visual_quality_pass'] is False


def test_explicit_contract_required():
    with pytest.raises(ValueError,match='EXPLICIT_REQUIRED'):
        check_required_roles(sample([0]),None,1)


def test_frame_identity_required():
    with pytest.raises(ValueError,match='FRAME_SET'):
        check_required_roles(sample([1]),['human'],2)


def test_missing_counter_rejected():
    with pytest.raises(ValueError,match='COUNTS_MISSING'):
        check_required_roles(sample([None]),['equipment'],1)
