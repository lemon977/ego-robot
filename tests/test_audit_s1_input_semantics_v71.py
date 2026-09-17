from chaoyang.ops.audit_s1_input_semantics_v71 import build


def test_s1_inputs_are_object_masks_not_role_masks() -> None:
    value = build()
    assert value["status"] == "PASS_WITH_SCOPE_CORRECTION"
    assert value["counts"] == {"rows": 7, "verified": 6, "blocked": 1}
    for row in value["rows"]:
        if row["status"] == "VERIFIED":
            assert row["prompt_semantics"] == "TASK_OBJECT_IDENTITY_MASK_NOT_ROLE_REMOVAL_MASK"
            assert "ROLE_MASK_SUCCESSOR" in row["forbidden_scopes"]
