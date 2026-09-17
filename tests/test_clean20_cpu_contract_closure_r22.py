import numpy as np

from chaoyang.ops.run_clean20_cpu_contract_closure_r22 import (
    classify_proposals_fail_closed,
    pack_mask,
    unpack_mask,
)


def test_unknown_semantic_layer_is_rejected_not_accepted():
    shape = (2, 3)
    write = np.ones(shape, bool)
    kind = np.ones(shape, np.uint8)
    source_frame = np.array([[0, 2, -1], [0, 0, 0]], np.int32)
    source_x = np.array([[0, 0, 0], [1, 2, 8]], np.int32)
    source_y = np.array([[0, 0, 0], [1, 1, 1]], np.int32)
    objects = [np.array([[False, False, False], [False, True, False]]), np.zeros(shape, bool)]
    reason, counts = classify_proposals_fail_closed(
        1, write, kind, source_frame, source_x, source_y, objects
    )
    assert reason.tolist() == [[5, 2, 3], [4, 5, 3]]
    assert counts["accepted_temporal_pixels"] == 0
    assert counts["support_or_semantic_unknown_rejected"] == 2
    assert counts["future_rejected"] == 1
    assert counts["source_task_object_rejected"] == 1


def test_pack_mask_round_trip():
    mask = np.array([[True, False, True], [False, True, False]], bool)
    assert np.array_equal(unpack_mask(pack_mask(mask), mask.shape), mask)
