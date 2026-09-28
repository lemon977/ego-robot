from copy import deepcopy
from unittest.mock import patch
import pytest

from chaoyang.governance.finalize_completion import TASK, close_state, closed_index
from chaoyang.governance.four_stream_completion import build_status


def fixture():
    ids = [TASK] + [TASK + '_' + x for x in ('scene', 'sensor', 'motion')]
    state = dict(governance_revision=14210, tasks=[dict(task_id=x, status='PENDING', pid=99,
                 proc_start_ticks=123, parent_task_id=TASK if x != TASK else None) for x in ids],
                 next_task=dict(task_id=TASK))
    state['tasks'] += [dict(task_id=TASK + '_huro', status='REJECTED_QUALITY', parent_task_id=TASK),
                      dict(task_id='other_ai', status='RUNNING', pid=55)]
    return state, {x: dict(path=x + '/RESULT.json') for x in ids}


def test_only_owned_rows_change():
    state, refs = fixture()
    original = deepcopy(state)
    result = close_state(state, refs, 'now')
    assert state == original
    assert result['tasks'][4:] == state['tasks'][4:]
    assert result['next_task'] is None
    for row in result['tasks'][:4]:
        assert row['status'] in {'CANCELLED', 'REJECTED_QUALITY'}
        assert row['adoption'] == 'NOT_ADOPTED'
        assert row['review'] == 'NOT_REVIEWED'
        assert row['pid'] is None and row['proc_start_ticks'] is None


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'terminal'])
def test_bad_task_set_fails(mutation):
    state, refs = fixture()
    if mutation == 'missing': state['tasks'].pop(0)
    if mutation == 'duplicate': state['tasks'].append(deepcopy(state['tasks'][0]))
    if mutation == 'terminal': state['tasks'][0]['status'] = 'CANCELLED'
    with pytest.raises(RuntimeError): close_state(state, refs, 'now')


def test_other_owner_routing_kept():
    state, refs = fixture()
    state['next_task'] = dict(task_id='other_ai')
    result = close_state(state, refs, 'now')
    assert result['next_task'] == state['next_task']
    index = dict(task_packets=[dict(task_id=r['task_id']) for r in state['tasks']])
    assert closed_index(index)['task_packets'] == [dict(task_id='other_ai')]
    assert len(index['task_packets']) == 6


def test_generated_status_does_not_claim_active_or_quality_pass():
    state, refs = fixture()
    state = close_state(state, refs, 'now')
    with patch('chaoyang.governance.four_stream_completion.artifact_ref', return_value={}), \
         patch('chaoyang.governance.four_stream_completion.load_json', return_value={'counts': {}}):
        result = build_status(state, 'now')
    assert result['active_tasks'] == []
    assert result['campaign_closed'] is True
    assert result['final_result'] == refs[TASK]
    assert result['adoption'] == 'NOT_ADOPTED'
    assert result['new_quality'] == 'UNMET_OR_UNVERIFIED'
    assert 'Active repair' not in result['claim_limit']
    assert result['training_eligible'] is False
