"""Third round is bounded; first two masks stay independent and identical."""
import numpy as np
import pytest
from tests.unit.test_inpainting_temporal_repair_rounds import repair_case

def test_third_round_keeps_two_round_prefix_and_one_rgb_collection(repair_case):
 c=repair_case
 two=c.invoke(repair_rounds=2,return_round_masks=True,tracker=c.tracker)
 snapshots={}
 three=c.invoke(repair_rounds=3,return_round_masks=True,tracker=c.tracker,round_snapshots=snapshots)
 assert [v['prompts'][0]['frame_idx'] for v in three[1]['rounds']]==[3,0,1]
 assert three[1]['point_tracking_calls']==3 and set(snapshots)=={2}
 assert c.calls=={'collect':2,'associate':2,'qualify':2}
 for k,v in two[0].items():
  if isinstance(v,np.ndarray):
   assert np.array_equal(v,snapshots[2][k])
   assert not np.shares_memory(snapshots[2][k],three[0][k])
 assert np.any(three[0]['core'] & ~two[0]['core'])
 assert np.array_equal(three[0]['protected_visible'],two[0]['protected_visible'])

def test_snapshot_mapping_must_start_empty(repair_case):
 with pytest.raises(ValueError,match='EMPTY_ROUND_SNAPSHOT'):
  repair_case.invoke(repair_rounds=3,tracker=repair_case.tracker,round_snapshots={2:{}})
