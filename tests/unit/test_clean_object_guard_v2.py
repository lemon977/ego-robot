import numpy as np
import pytest
from chaoyang.pipeline.clean_object_guard_v2 import guard_frame


def fixture():
    raw = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
    old = np.full_like(raw, 240)
    kind = np.full((3, 4), 3, np.uint8)
    role = np.ones((3, 4), bool)
    protected = np.zeros((3, 4), bool)
    protected[0, 0] = True
    identity = [dict(observed=True, valid=True, physical_instance_id=0,
                     expected_instance_id=0, mask_area_px=1)]
    return [raw, old, kind, role, protected, identity]


def test_known_protected_and_outside_role_are_raw():
    a = fixture(); a[3][1, 1] = False
    out, source, write, stats = guard_frame(*a)
    assert np.array_equal(out[~write], a[0][~write])
    assert np.array_equal(out[write], a[1][write])
    assert not write[0, 0] and not write[1, 1]
    assert source[0, 0] == 1 and source[2, 2] == 3
    assert stats['outside_write_changed_pixels'] == 0


@pytest.mark.parametrize('key,value', [('observed', False), ('valid', False),
    ('physical_instance_id', -1), ('physical_instance_id', 7),
    ('observed', 'false'), ('valid', 1), ('mask_area_px', 0)])
def test_unknown_cannot_become_empty_protection(key, value):
    a = fixture(); a[5][0][key] = value
    out, source, write, stats = guard_frame(*a)
    assert np.array_equal(out, a[0]) and not write.any()
    assert stats['status'] == 'ABSTAIN_OBJECT_IDENTITY_UNKNOWN'
    assert np.all(source[~a[4]] == 4)


def test_one_unknown_of_three_objects_abstains():
    a = fixture(); a[5] *= 3; a[5][2] = dict(a[5][2], valid=False)
    assert not guard_frame(*a)[2].any()


def test_legacy_target_pixels_never_new_write():
    a = fixture(); a[2][2, 1] = 0; a[2][2, 2] = 2; a[2][2, 3] = 1
    out, source, write, _ = guard_frame(*a)
    assert not write[2, 1] and not write[2, 2] and source[2, 3] == 2


@pytest.mark.parametrize('index,value', [(2, np.full((3, 4), 7, np.uint8)),
    (3, np.ones((3, 4), np.uint8)), (4, np.zeros((2, 4), bool)), (5, [])])
def test_bad_schema_rejected(index, value):
    a = fixture(); a[index] = value
    with pytest.raises(ValueError): guard_frame(*a)


def test_real_negative_control_old_composite_erases_unknown_object():
    a = fixture(); a[5][0]['observed'] = False; a[4][:] = False
    # Reproduce the legacy rule: unknown -> empty protection -> all deletion allowed.
    legacy_result = a[0].copy(); legacy_result[a[2] == 3] = a[1][a[2] == 3]
    assert not np.array_equal(legacy_result, a[0])
    assert np.array_equal(guard_frame(*a)[0], a[0])
