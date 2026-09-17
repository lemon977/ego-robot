from chaoyang.ops.run_poker245_atlas_min_canary_r22 import face_label


def test_face_labels_are_derived_from_frozen_provider_semantics():
    assert face_label("CURRENT_RGB_RIGHTMOST_BACK_ACTION_ANCHOR") == "CARD_BACK_VISIBLE"
    assert face_label("CURRENT_RGB_REVEALED_FACE_SAME_ACTION_ID") == "CARD_FACE_VISIBLE"
    assert face_label("UNKNOWN") == "UNKNOWN"
