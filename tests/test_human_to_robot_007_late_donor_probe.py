import numpy as np

from chaoyang.ops.run_human_to_robot_007_late_donor_probe import DONORS, TARGETS, coverage


def test_frozen_cohort_excludes_target_window():
    assert TARGETS == (192, 196)
    assert DONORS == tuple(frame for frame in range(378) if not 181 <= frame <= 196)
    assert len(DONORS) == 362


def test_coverage_keeps_zero_and_single_donor_separate():
    count = np.array([[0, 1, 3], [4, 0, 2]], dtype=np.uint16)
    eligible = np.array([[1, 1, 1], [0, 1, 1]], dtype=bool)
    assert coverage(count, eligible) == {
        "eligible_write_pixels": 5,
        "zero_donor_pixels": 2,
        "one_or_more_donor_pixels": 3,
        "three_or_more_donor_pixels": 1,
    }


def test_u16_task_object_labels_are_not_grayscale_proxy():
    labels = np.array([0, 1, 256, 512], dtype=np.uint16)
    write = np.ones(4, dtype=bool)
    eligible = write & (labels == 0)
    assert eligible.tolist() == [True, False, False, False]
