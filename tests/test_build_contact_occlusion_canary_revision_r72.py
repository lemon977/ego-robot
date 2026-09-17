import json
from pathlib import Path

from chaoyang.ops.build_contact_occlusion_canary_revision_r72 import build


def test_r72_closes_s1_inputs_without_claiming_result() -> None:
    value = build()
    assert value["artifact_revision"] == "R7_2"
    s1 = next(item for item in value["canaries"] if item["canary_id"] == "S1")
    assert s1["resource_status"] == "READY_FOR_BOUNDED_RUN"
    assert s1["blockers"] == []
    statuses = {item["dependency_id"]: item["status"] for item in s1["algorithm_prerequisites"]}
    assert statuses["CUTIE_CODE_AND_WEIGHTS"] == "VERIFIED_LOCAL"
    assert statuses["FROZEN_REENTRY_AUDIT_FRAMES"] == "VERIFIED_INPUT_ONLY"
    assert value["authority"] is False
    assert "No inference result" in value["claim_limit"]
