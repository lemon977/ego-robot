"""Finite MCO context wiring with synthetic CPU providers and sealed inputs."""
import json
import pytest
from tests.unit.test_inpainting_v1_mask_causal_wiring import causal_fixture
from chaoyang.inpainting_candidate import v1_mco_worker

def configure(f):
 p=f.packet;p['config'].update(mask_causal_pair=False,context_clean_pair=True,context_experiment='FULL_SESSION_EQUAL_COUNT')
 p['inputs']['frozen_repaired_masks']=p['inputs']['frozen_base_masks']
 return p

def test_context_route_seals_three_calls_and_fixed_mask(causal_fixture):
 f=causal_fixture;configure(f)
 assert v1_mco_worker.execute_v1_mco_stage(f.packet)==0
 assert len(f.loads)==1 and f.parent.model.pixel_calls==3
 result=json.loads((f.output/'MODEL_RESULT.json').read_text())
 assert result['context_clean_pair'] is True and result['model_calls']==3 and result['render_calls']==0
 receipt=json.loads((f.output/'sessions/synthetic_session/Clean/CLEAN_CONTEXT_PAIR_RECEIPT.json').read_text())
 assert receipt['experiment']=='FULL_SESSION_EQUAL_COUNT' and receipt['A_repeat_reproducibility']['exact_equal']
 assert receipt['frozen_base_masks']==receipt['frozen_repaired_masks']
 assert len(result['sessions'][0]['completed_frame_indices'])==4

def test_mask_change_rejected_before_model_load(causal_fixture):
 f=causal_fixture;f.packet['config'].update(mask_causal_pair=False,context_clean_pair=True,context_experiment='FULL_SESSION_EQUAL_COUNT')
 with pytest.raises(ValueError,match='MASK_CHANGE_FORBIDDEN'):
  v1_mco_worker.execute_v1_mco_stage(f.packet)
 assert not f.loads

@pytest.mark.parametrize('experiment',['UNREGISTERED_POLICY',None])
def test_unknown_experiment_rejected_before_load(causal_fixture,experiment):
 f=causal_fixture;configure(f);f.packet['config']['context_experiment']=experiment
 with pytest.raises(ValueError,match='CONTEXT_PAIR'):
  v1_mco_worker.execute_v1_mco_stage(f.packet)
 assert not f.loads
