from chaoyang.ops.run_rc1_poker245_donor_observability import evaluate_observability


def test_four_independent_gates_required():
    base = dict(source_observed=True, source_precedes_target=True,
                identity_proof=True, face_proof=True, geometry_pass=True)
    assert evaluate_observability(**base)['accept_donor_pixels']
    for key in base:
        trial = {**base, key: False}
        assert not evaluate_observability(**trial)['accept_donor_pixels']


def test_same_back_and_homography_cannot_prove_identity_or_face():
    result = evaluate_observability(source_observed=True, source_precedes_target=True,
                                    identity_proof=False, face_proof=False,
                                    geometry_pass=True)
    assert not result['accept_donor_pixels']
    assert result['missing'] == [
        'B_SAME_PHYSICAL_CARD_INDEPENDENTLY_PROVEN',
        'C_SAME_FACE_INDEPENDENTLY_PROVEN',
    ]
