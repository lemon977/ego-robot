"""No assets/models: exercise the production provenance path with fake FK/solve."""
from types import SimpleNamespace

import numpy as np
import pytest

from chaoyang.pipeline import shared_local_hand_target_v1 as m


def points():
    p = np.zeros((21, 3))
    for f in range(5):
        for j in range(4):
            p[1 + 4*f + j] = ((2-f)*.02, .04+j*.02, .001*j*j)
    return p


def setup_solver(monkeypatch):
    hand = SimpleNamespace(lower=np.zeros(22), upper=np.ones(22),
                           joint_names=tuple(map(str, range(22))))
    monkeypatch.setattr(m, 'anatomical_keypoints_from_q', lambda *a: points())
    def fake(*args, **kwargs):
        diagnostic = m.LocalSolveDiagnostics(True, 1, 0., 0., 0., 0.)
        return np.full(22, .5), points(), diagnostic
    monkeypatch.setattr(m, 'solve_local_frame', fake)
    return (hand, hand)


def sequence(monkeypatch, valid, direct=None):
    return m.solve_local_sequence(
        setup_solver(monkeypatch), np.broadcast_to(points(), (2, 3, 21, 3)),
        valid, np.arange(3), source_kind='HAWOR21', joint_names=m.HAWOR21_NAMES,
        source_observed=direct)


def test_legacy_valid_true_does_not_create_observations(monkeypatch):
    valid = np.ones((2, 3), bool)
    result = sequence(monkeypatch, valid)
    assert result['solver_success'].sum() == 6
    assert result['source_valid_physical'].sum() == 6
    assert result['source_observed_physical'].sum() == 0
    assert result['source_valid_not_directly_observed_physical'].sum() == 6
    assert str(result['source_observation_policy']) == 'EXPLICIT_PROVENANCE_ONLY_V1'


def test_explicit_direct_observation_preserves_independent_side_masks(monkeypatch):
    valid = np.ones((2, 3), bool)
    valid[0, 1] = False
    direct = np.zeros_like(valid)
    direct[1, 2] = True
    result = sequence(monkeypatch, valid, direct)
    np.testing.assert_array_equal(result['source_valid_physical'], valid.T)
    np.testing.assert_array_equal(result['source_observed_physical'], direct.T)
    assert result['solver_success'].sum() == 5
    assert not result['solver_success'][1, 0]
    assert result['solver_success'][1, 1]


@pytest.mark.parametrize('valid,direct,message', [
    (np.ones((2, 3), int), None, 'boolean'),
    (np.ones((3, 2), bool), None, 'anatomical'),
    (np.ones((2, 3), bool), np.ones((3, 2), bool), 'match'),
    (np.ones((2, 3), bool), np.ones((2, 3), int), 'boolean'),
    (np.zeros((2, 3), bool), np.ones((2, 3), bool), 'invalid'),
])
def test_reject_invalid_masks(valid, direct, message):
    with pytest.raises(m.HuroHandOnlyError, match=message):
        m.source_provenance_masks(valid, direct)


def test_masks_do_not_alias_inputs():
    valid = np.ones((2, 3), bool)
    observed = np.ones_like(valid)
    a, b = m.source_provenance_masks(valid, observed)
    a[:] = False
    b[:] = False
    assert valid.all() and observed.all()


def test_provenance_change_does_not_change_candidate_q(monkeypatch):
    valid = np.ones((2, 3), bool)
    unknown = sequence(monkeypatch, valid)
    direct = sequence(monkeypatch, valid, valid)
    np.testing.assert_array_equal(unknown['q22'], direct['q22'])
    np.testing.assert_array_equal(unknown['solver_success'], direct['solver_success'])
