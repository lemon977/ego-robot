import pytest
from chaoyang.pipeline.s01_point_semantics_v1 import metadata_errors, point_comparison


def test_actual_conflict_rejected():
    assert metadata_errors({'installation_fitted':False},{'sides':[{'success':True}]}) == ['INSTALLATION_FIT_METADATA_CONFLICT']


def test_nominal_and_valid_fit_not_confused():
    assert metadata_errors({'installation_fitted':False},{'sides':[]}) == []
    assert metadata_errors({'installation_fitted':True},{'sides':[{'success':True}]}) == []
    assert 'INSTALLATION_FIT_SUCCESS_EVIDENCE_MISSING' in metadata_errors({'installation_fitted':True},{'sides':[]})


def test_unknown_is_not_false_or_success():
    assert 'INSTALLATION_FITTED_FLAG_UNKNOWN' in metadata_errors({}, {'sides':[]})
    assert 'INSTALLATION_FITTED_FLAG_UNKNOWN' in metadata_errors({'installation_fitted':0}, {'sides':[]})
    assert 'INVALID_INSTALLATION_SUCCESS_TYPE' in metadata_errors({'installation_fitted':False},{'sides':[{'success':'true'}]})


@pytest.mark.parametrize('other',['controller_housing_center','anatomical_wrist_center','manus_virtual_root'])
def test_different_points_need_independent_mapping(other):
    assert point_comparison('controller_tracked_origin',other) == 'NOT_COMPARABLE_AS_SAME_POINT'
    assert point_comparison('controller_tracked_origin',other,{'independently_verified':False}) == 'NOT_COMPARABLE_AS_SAME_POINT'


def test_same_definition_not_accuracy():
    assert point_comparison('manus_virtual_root','manus_virtual_root') == 'SAME_POINT_DEFINITION_NOT_ACCURACY_PROOF'
    with pytest.raises(ValueError):point_comparison('unknown','manus_virtual_root')
