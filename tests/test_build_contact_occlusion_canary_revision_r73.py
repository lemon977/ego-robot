from chaoyang.ops.build_contact_occlusion_canary_revision_r73 import build


def test_r73_limits_s1_to_task_object_masks() -> None:
    value = build()
    assert value["artifact_revision"] == "R7_3"
    s1 = next(item for item in value["canaries"] if item["canary_id"] == "S1")
    assert "Task-object modal masks only" in s1["scope"]
    assert "ROLE_MASK_SUCCESSOR" in s1["authority_limit"]["forbidden_claims"]
    assert "withdraws any Role Mask implication" in value["claim_limit"]
