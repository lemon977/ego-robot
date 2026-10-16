import copy
from pathlib import Path

import pytest

from chaoyang.inpainting_candidate import contracts as c


@pytest.fixture
def manifest_fixture(tmp_path, monkeypatch):
    """Raw is a read-only reference fixture, not a derived production path."""
    actual_ref = c.file_ref
    raw_refs = {role: {"path": "/readonly-original/session/" + relative, "bytes": 7, "sha256": "a" * 64}
                for role, relative in (("source_manifest", "manifest.json"), ("raw_video", "raw/vst.h264"),
                                       ("raw_video_qpc", "raw/vst.qpc.ts.jsonl"), ("raw_video_wall", "raw/vst.ts.jsonl"))}
    reference_by_path = {row["path"]: row for row in raw_refs.values()}
    monkeypatch.setattr(c, "file_ref", lambda path: reference_by_path[str(path)] if str(path) in reference_by_path else actual_ref(path))
    code = tmp_path / "component.py"
    weight = tmp_path / "weight.bin"
    code.write_text("frozen code")
    weight.write_bytes(b"frozen weight")
    rows = [{"name": role, "category": "source", "provenance": "RAW_CAPTURE", "role": role,
             "status": "PRESENT", "file": reference} for role, reference in raw_refs.items()]
    for stage in c.STAGES:
        rows.append(c.input_entry(stage + "_code", category="code", provenance="GENERIC_CODE",
                                  role=stage.lower() + "_code", path=code))
    rows += [c.input_entry(role, category="weights", provenance="FROZEN_WEIGHT", role=role, path=weight)
             for role in ("segmentation_weight", "propainter_weight", "flow_weight", "robot_asset")]
    rows += [c.input_entry(stage + "_recipe", category="config", provenance="GENERIC_CONFIG", role="generic_recipe", value={"size": 8})
             for stage in ("Mask", "Clean", "Composite")]
    rows += [c.input_entry("installation", category="calibration", provenance="INSTALLATION_CONTRACT", role="installation_calibration", absent=True),
             c.input_entry("calibration", category="calibration", provenance="CAPTURE_CALIBRATION", role="camera_calibration", absent=True),
             c.input_entry("runtime", category="code", provenance="GENERIC_CODE", role="runtime_code", path=code)]
    rows += [c.input_entry(role, category="schema", provenance="SCHEMA", role=role, value={"version": 1})
             for role in ("input_schema", "frame_schema", "result_schema")]
    assignments = {stage: [stage + "_code"] for stage in c.STAGES}
    assignments["Source"] += list(raw_refs) + ["input_schema", "frame_schema", "runtime"]
    assignments["Motion"] += ["installation"]
    assignments["Mask"] += ["segmentation_weight", "Mask_recipe"]
    assignments["Clean"] += ["propainter_weight", "flow_weight", "Clean_recipe"]
    assignments["Render"] += ["robot_asset", "calibration"]
    assignments["Composite"] += ["Composite_recipe", "result_schema"]
    return c.freeze_input_manifest(rows), assignments, weight


def test_manifest_is_order_independent_and_actual_files_bound(manifest_fixture):
    manifest, _, _ = manifest_fixture
    assert manifest["sha256"] == c.freeze_input_manifest(reversed(manifest["entries"]))["sha256"]
    assert manifest["control_ground_truth"] is False


def test_input_drift_is_rejected(manifest_fixture):
    manifest, assignments, weight = manifest_fixture
    weight.write_bytes(b"different weight")
    with pytest.raises(c.ContractError, match="INPUT_SHA_DRIFT"):
        c.stage_signatures(manifest, assignments)


@pytest.mark.parametrize("role", ["manual_seed", "historical_mask", "historical_clean", "robot_q", "task_config", "evaluation_annotation"])
def test_forbidden_producer_roles_rejected(role):
    with pytest.raises(c.ContractError, match="PRODUCER_INPUT_FORBIDDEN"):
        c.input_entry("bad", category="config", provenance="GENERIC_CONFIG", role=role, value={"secret": 1})


def test_missing_input_category_fails(manifest_fixture):
    manifest, _, _ = manifest_fixture
    with pytest.raises(c.ContractError, match="CATEGORIES_INCOMPLETE"):
        c.freeze_input_manifest([row for row in manifest["entries"] if row["category"] != "weights"])


def test_unknown_roles_and_derived_source_path_fail(tmp_path):
    path = tmp_path / "raw.bin"
    path.write_bytes(b"raw")
    with pytest.raises(c.ContractError, match="PRODUCER_INPUT_FORBIDDEN"):
        c.input_entry("raw", category="source", provenance="RAW_CAPTURE", role="anything", path=path)
    with pytest.raises(c.ContractError, match="DERIVED_SOURCE_PATH"):
        c.input_entry("raw", category="source", provenance="RAW_CAPTURE", role="raw_video", path=path)


def test_absent_is_calibration_only():
    with pytest.raises(c.ContractError, match="ABSENT_ONLY_CALIBRATION"):
        c.input_entry("w", category="weights", provenance="FROZEN_WEIGHT", role="flow_weight", absent=True)


def test_nan_configuration_not_signed():
    with pytest.raises(ValueError):
        c.input_entry("r", category="config", provenance="GENERIC_CONFIG", role="generic_recipe", value={"x": float("nan")})


def test_mask_edit_invalidates_only_mask_clean_composite(manifest_fixture):
    manifest, assignments, _ = manifest_fixture
    before = c.stage_signatures(manifest, assignments)
    changed = copy.deepcopy(manifest)
    recipe = next(row for row in changed["entries"] if row["name"] == "Mask_recipe")
    recipe["value"]["size"] = 9
    recipe["sha256"] = c.canonical_sha(recipe["value"])
    after = c.stage_signatures(changed, assignments)
    assert {stage for stage in c.STAGES if before[stage]["sha256"] != after[stage]["sha256"]} == {"Mask", "Clean", "Composite"}


def test_clean_edit_does_not_invalidate_render(manifest_fixture):
    manifest, assignments, _ = manifest_fixture
    before = c.stage_signatures(manifest, assignments)
    changed_assignments = copy.deepcopy(assignments)
    changed_assignments["Clean"].append("Mask_recipe")
    after = c.stage_signatures(manifest, changed_assignments)
    assert {stage for stage in c.STAGES if before[stage]["sha256"] != after[stage]["sha256"]} == {"Clean", "Composite"}


def test_unassigned_dependency_rejected(manifest_fixture):
    manifest, assignments, _ = manifest_fixture
    assignments["Source"].remove("runtime")
    with pytest.raises(c.ContractError, match="UNASSIGNED"):
        c.stage_signatures(manifest, assignments)


def test_verified_complete_stage_resumable_and_partial_rejected(tmp_path):
    output = tmp_path / "stage.json"
    output.write_text("sealed output")
    receipt = {"schema": c.SCHEMA, "status": "SEALED_COMPLETE", "component_signature": "s",
               "outputs": [c.file_ref(output)]}
    assert c.verify_sealed_stage(receipt, "s", output_root=tmp_path)[0]["path"] == str(output)
    receipt["status"] = "PARTIAL"
    with pytest.raises(c.ContractError, match="PARTIAL_NOT_CONSUMABLE"):
        c.verify_sealed_stage(receipt, "s", output_root=tmp_path)


def test_receipt_wrong_signature_and_outside_path_rejected(tmp_path):
    output = tmp_path / "output.json"
    other = tmp_path / "other"
    other.mkdir()
    output.write_text("sealed")
    receipt = {"schema": c.SCHEMA, "status": "SEALED_COMPLETE", "component_signature": "s", "outputs": [c.file_ref(output)]}
    with pytest.raises(c.ContractError, match="SIGNATURE_MISMATCH"):
        c.verify_sealed_stage(receipt, "wrong", output_root=tmp_path)
    with pytest.raises(c.ContractError, match="OUTSIDE_WRITE_SET"):
        c.verify_sealed_stage(receipt, "s", output_root=other)


@pytest.mark.parametrize("name,target", [("Mask_code", "Render"), ("segmentation_weight", "Render"),
                                          ("propainter_weight", "Mask"), ("result_schema", "Source")])
def test_semantic_dependency_cannot_be_signed_on_wrong_stage(manifest_fixture, name, target):
    manifest, assignments, _ = manifest_fixture
    for names in assignments.values():
        if name in names:
            names.remove(name)
    assignments[target].append(name)
    with pytest.raises(c.ContractError, match="STAGE_(ROLE_ASSIGNMENT_INVALID|CLOSURE_INCOMPLETE)"):
        c.stage_signatures(manifest, assignments)


@pytest.mark.parametrize("stage,name", [("Source", "input_schema"), ("Source", "frame_schema"),
                                        ("Motion", "installation"), ("Mask", "segmentation_weight"),
                                        ("Clean", "propainter_weight"), ("Clean", "flow_weight"),
                                        ("Render", "robot_asset"), ("Render", "calibration"),
                                        ("Composite", "result_schema"), ("Composite", "Composite_recipe")])
def test_each_stage_requires_actual_semantic_closure(manifest_fixture, stage, name):
    manifest, assignments, _ = manifest_fixture
    assignments[stage].remove(name)
    with pytest.raises(c.ContractError, match="STAGE_CLOSURE_INCOMPLETE"):
        c.stage_signatures(manifest, assignments)
