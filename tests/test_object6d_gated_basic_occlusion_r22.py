from pathlib import Path

import pytest

from chaoyang.ops.run_object6d_gated_basic_occlusion_r22 import physical_instance_id


def test_legacy_object6d_identity_is_bound_to_immutable_parent() -> None:
    path = Path("/authority/physical_object_2/RESULT.json")
    assert physical_instance_id(path, {}) == 2


def test_explicit_object6d_identity_has_priority() -> None:
    path = Path("/authority/not_a_legacy_identity/RESULT.json")
    assert physical_instance_id(path, {"physical_instance_id": 1}) == 1


def test_missing_object6d_identity_is_fail_closed() -> None:
    with pytest.raises(ValueError, match="no physical_instance_id"):
        physical_instance_id(Path("/authority/object/RESULT.json"), {})
