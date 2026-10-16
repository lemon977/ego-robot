"""Portable, synthetic contract checks; no server evidence, src import or model."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


CHECK = module("synthetic_handoff_checker", ROOT / "scripts/maintenance/project_handoff_check_v1.py")
FIXTURE = module("synthetic_handoff_fixtures", ROOT / "tests/helpers/handoff_fixtures_v1.py")


@pytest.fixture
def tree(tmp_path):
    return FIXTURE.build(tmp_path / "synthetic_project")


def fingerprint(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


def test_valid_view_is_readonly_and_preserves_qualification(tree):
    before = fingerprint(tree)
    view = CHECK.inspect_project(tree)
    assert view["status"] == "PASSED_READONLY_BINDING"
    assert view["frozen_release_files_sha_verified"] == 2
    assert {row["phase_alias"] for row in view["selected_baselines"]} == CHECK.ALIASES
    assert all(not row["training_eligible"] and not row["control_ground_truth"] for row in view["selected_baselines"])
    assert not view["execution_authorization"] and view["files_written"] == 0
    assert fingerprint(tree) == before


def test_explicit_alias_selection_not_version_sort(tree):
    view = CHECK.inspect_project(tree, "MANUS_PHASE_V1")
    assert [row["phase_alias"] for row in view["selected_baselines"]] == ["MANUS_PHASE_V1"]


@pytest.mark.parametrize("alias", ["latest", "takeover_v12", "Cards042_V2", "UNKNOWN"])
def test_historical_or_latest_alias_rejected(tree, alias):
    with pytest.raises(CHECK.CheckError, match="ALIAS_NOT_SELECTED"):
        CHECK.inspect_project(tree, alias)


def test_tampered_frozen_file_is_not_excused_by_live_signature(tree):
    manifest = FIXTURE.load(tree / "baseline_v1/MANIFEST.json")
    (tree / "baseline_v1" / manifest["sources"][0]["bundle_path"]).write_text("tampered\n")
    FIXTURE.set_runtime(tree, [FIXTURE.ALGORITHM], active=True)
    with pytest.raises(CHECK.CheckError, match="SOURCE_RELEASE_SHA_DRIFT"):
        CHECK.inspect_project(tree)


def test_nonwrapper_live_algorithm_drift_is_rejected_even_if_signed(tree):
    (tree / FIXTURE.ALGORITHM).write_text("changed algorithm\n")
    FIXTURE.set_runtime(tree, [FIXTURE.ALGORITHM], active=True)
    with pytest.raises(CHECK.CheckError, match="LIVE_ALGORITHM_DRIFT"):
        CHECK.inspect_project(tree)


@pytest.mark.parametrize("active", [False, True])
def test_exact_signed_runtime_wrapper_drift_is_separate_from_frozen_release(tree, active):
    (tree / FIXTURE.WRAPPER).write_text("legitimate wrapper migration\n")
    FIXTURE.set_runtime(tree, [FIXTURE.WRAPPER], active=active)
    view = CHECK.inspect_project(tree)
    assert view["frozen_release_files_sha_verified"] == 2
    assert view["signed_runtime_wrapper_differences"] == [FIXTURE.WRAPPER]
    if active:
        assert view["current_tasks"][0]["baseline_adoption"] == "NOT_INFERRED_FROM_TASK_OR_VIDEO"
        assert len(view["selected_baselines"]) == 2


def set_special_closure(root, key, paths):
    path = root / "docs/governance/ALGORITHM_CONTRACT.json"
    algorithm = FIXTURE.load(path)
    algorithm["special_status_contracts"][key] = {
        "code_closure": [FIXTURE.ref(root / relative) for relative in paths]}
    FIXTURE.write_json(path, algorithm)
    FIXTURE.refresh(root)


@pytest.mark.parametrize("key", ["humanego_ict_joint26_prepare_v1", "humanego_ict_joint26_v1"])
def test_joint26_signed_wrapper_preserves_baselines_without_active_task(tree, key):
    selected_before = CHECK.inspect_project(tree)["selected_baselines"]
    (tree / FIXTURE.WRAPPER).write_text("joint26 wrapper migration\n")
    set_special_closure(tree, key, [FIXTURE.WRAPPER])
    before = fingerprint(tree)
    view = CHECK.inspect_project(tree)
    assert view["status"] == "PASSED_READONLY_BINDING"
    assert view["signed_runtime_wrapper_differences"] == [FIXTURE.WRAPPER]
    assert view["frozen_release_files_sha_verified"] == 2
    assert view["selected_baselines"] == selected_before
    assert {row["phase_alias"] for row in view["selected_baselines"]} == CHECK.ALIASES
    assert all(not row["training_eligible"] and not row["control_ground_truth"]
               for row in view["selected_baselines"])
    assert view["current_tasks"] == []
    assert not view["execution_authorization"] and view["files_written"] == 0
    assert fingerprint(tree) == before


def test_historical_special_cannot_authorize_current_wrapper(tree):
    (tree / FIXTURE.WRAPPER).write_text("historically signed wrapper\n")
    set_special_closure(tree, "humanego_ict_joint26_v0_historical", [FIXTURE.WRAPPER])
    with pytest.raises(CHECK.CheckError, match="LIVE_WRAPPER_UNBOUND"):
        CHECK.inspect_project(tree)


@pytest.mark.parametrize("key", ["humanego_ict_joint26_prepare_v1", "humanego_ict_joint26_v1"])
def test_joint26_wrapper_changed_after_closure_freeze_is_rejected(tree, key):
    set_special_closure(tree, key, [FIXTURE.WRAPPER])
    (tree / FIXTURE.WRAPPER).write_text("changed after joint26 closure freeze\n")
    with pytest.raises(CHECK.CheckError, match="CURRENT_RUNTIME_SHA_DRIFT"):
        CHECK.inspect_project(tree)


@pytest.mark.parametrize("key", ["humanego_ict_joint26_prepare_v1", "humanego_ict_joint26_v1"])
def test_joint26_signature_cannot_excuse_nonwrapper_algorithm_drift(tree, key):
    (tree / FIXTURE.ALGORITHM).write_text("changed algorithm signed by joint26\n")
    set_special_closure(tree, key, [FIXTURE.ALGORITHM])
    with pytest.raises(CHECK.CheckError, match="LIVE_ALGORITHM_DRIFT"):
        CHECK.inspect_project(tree)


def test_unsigned_wrapper_drift_rejected(tree):
    (tree / FIXTURE.WRAPPER).write_text("unbound wrapper\n")
    with pytest.raises(CHECK.CheckError, match="LIVE_WRAPPER_UNBOUND"):
        CHECK.inspect_project(tree)


def test_current_runtime_sha_mismatch_rejected(tree):
    FIXTURE.set_runtime(tree, [FIXTURE.WRAPPER], active=True)
    (tree / FIXTURE.WRAPPER).write_text("changed after packet freeze\n")
    with pytest.raises(CHECK.CheckError, match="CURRENT_RUNTIME_SHA_DRIFT"):
        CHECK.inspect_project(tree)


def test_active_packet_sha_mismatch_rejected(tree):
    FIXTURE.set_runtime(tree, [FIXTURE.WRAPPER], active=True)
    packet = tree / "tasks/synthetic/TASK_PACKET.json"
    packet.write_text(packet.read_text() + " ")
    with pytest.raises(CHECK.CheckError, match="CURRENT_TASK_PACKET_SHA_DRIFT"):
        CHECK.inspect_project(tree)


def test_inactive_state_cannot_authorize_packet(tree):
    FIXTURE.set_runtime(tree, [FIXTURE.WRAPPER], active=True, status="DONE")
    with pytest.raises(CHECK.CheckError, match="CURRENT_TASK_STATE_CONFLICT"):
        CHECK.inspect_project(tree)


def test_receipt_input_sha_failure_does_not_read_another_latest(tree):
    path = tree / "docs/governance/ALGORITHM_CONTRACT.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(CHECK.CheckError, match="INPUT_SHA_MISMATCH"):
        CHECK.inspect_project(tree)


def test_receipt_revision_conflict(tree):
    path = tree / "docs/governance/CURRENT_BASELINE_REGISTRY_V2.json"
    FIXTURE.write_json(path, {"governance_revision": 2})
    FIXTURE.refresh(tree)
    with pytest.raises(CHECK.CheckError, match="SNAPSHOT_REVISION_CONFLICT"):
        CHECK.inspect_project(tree)


@pytest.mark.parametrize("key,code", [("task_state", "SNAPSHOT_GENERATION_CONFLICT"),
                                     ("task_packet_index_current", "INDEX_GENERATION_REVISION_CONFLICT")])
def test_task_docs_generation_conflicts_are_checked(tree, key, code):
    path = tree / "docs/governance" / FIXTURE.DOC_NAMES[key]
    value = FIXTURE.load(path)
    value["generation_id"] = "gov-999999-wrong"
    FIXTURE.write_json(path, value)
    FIXTURE.refresh(tree)
    with pytest.raises(CHECK.CheckError, match=code):
        CHECK.inspect_project(tree)


@pytest.mark.parametrize("key", ["task_packet_index_current", "task_state"])
def test_duplicate_or_multiple_active_task_rejected(tree, key):
    FIXTURE.set_runtime(tree, [FIXTURE.WRAPPER], active=True)
    path = tree / "docs/governance" / FIXTURE.DOC_NAMES[key]
    data = FIXTURE.load(path)
    collection = "task_packets" if key == "task_packet_index_current" else "tasks"
    data[collection].append(dict(data[collection][0]))
    FIXTURE.write_json(path, data)
    FIXTURE.refresh(tree)
    with pytest.raises(CHECK.CheckError, match="CURRENT_TASK_NOT_UNIQUE"):
        CHECK.inspect_project(tree)


def test_verified_task_view_exposes_declarations_not_inferred_next_action(tree):
    FIXTURE.set_runtime(tree, [FIXTURE.WRAPPER], active=True)
    row = CHECK.inspect_project(tree)["current_tasks"][0]
    assert row["verified_packet"] == FIXTURE.ref(tree / "tasks/synthetic/TASK_PACKET.json")
    assert row["next_action"] == row["writer_declared"] == row["commands"] == row["read_set"] == "ABSENT"
    assert row["writer_liveness"] == "NOT_CHECKED_BY_READONLY_VIEW"


def test_authority_reference_conflict(tree):
    path = tree / "docs/governance/DOC_AUTHORITY_MAP.json"
    data = FIXTURE.load(path)
    data["documents"][0]["sha256"] = "0" * 64
    FIXTURE.write_json(path, data)
    receipt = tree / "docs/governance/CURRENT_STATUS_RECEIPT.json"
    value = FIXTURE.load(receipt)
    value["files"]["doc_authority_map"] = FIXTURE.ref(path)
    FIXTURE.write_json(receipt, value)
    with pytest.raises(CHECK.CheckError, match="PHASE_AUTHORITY_REF_CONFLICT"):
        CHECK.inspect_project(tree)


@pytest.mark.parametrize("change,code", [
    ("duplicate", "SOURCE_RELEASE_DUPLICATE_PATH"), ("count", "SOURCE_RELEASE_COUNT_DRIFT"),
    ("escape", "SOURCE_RELEASE_PATH_ESCAPE"), ("role", "SOURCE_RELEASE_ROLE_DRIFT"),
    ("operation", "PHASE_OPERATION_DRIFT"), ("phase", "FROZEN_PHASE_BINDING_DRIFT")])
def test_refrozen_manifest_does_not_relax_structural_contract(tree, change, code):
    path = tree / "baseline_v1/MANIFEST.json"
    data = FIXTURE.load(path)
    if change == "duplicate":
        data["sources"].append(data["sources"][0])
    elif change == "count":
        data["counts"]["source_files"] = 42
    elif change == "escape":
        data["sources"][0]["bundle_path"] = "../outside"
    elif change == "role":
        data["production_loader_changed"] = True
    elif change == "operation":
        data["selected_operations"]["EXACT78_PHASE_V1"] = "run_old"
    else:
        data["phase_contract"] = {"path": "not-the-selected-phase"}
    FIXTURE.write_json(path, data)
    FIXTURE.refresh(tree, manifest_changed=True)
    with pytest.raises(CHECK.CheckError, match=code):
        CHECK.inspect_project(tree)


def test_selected_phase_set_not_expanded_by_new_pointer(tree):
    path = tree / "docs/current/DUAL_BRANCH_BASELINE_V1.json"
    phase = FIXTURE.load(path)
    phase["aliases"]["THIRD_CANDIDATE"] = dict(phase["aliases"]["EXACT78_PHASE_V1"])
    FIXTURE.write_json(path, phase)
    FIXTURE.refresh(tree, phase_changed=True)
    with pytest.raises(CHECK.CheckError, match="SELECTED_ALIAS_SET_DRIFT"):
        CHECK.inspect_project(tree)


def test_pointer_sha_is_not_inferred_from_playable_result(tree):
    (tree / "docs/current/EXACT78_PHASE_V1.json").write_text("{}")
    with pytest.raises(CHECK.CheckError, match="BASELINE_POINTER_DRIFT"):
        CHECK.inspect_project(tree)


def test_symlink_inside_project_not_accepted_as_actual_input(tree):
    pointer = tree / "docs/current/EXACT78_PHASE_V1.json"
    alternate = pointer.with_name("alternate.json")
    alternate.write_bytes(pointer.read_bytes())
    pointer.unlink()
    pointer.symlink_to(alternate)
    with pytest.raises(CHECK.CheckError, match="PATH_SCOPE_OR_SYMLINK"):
        CHECK.inspect_project(tree)


def test_missing_server_tree_fails_cleanly_without_writes(tmp_path, capsys):
    before = fingerprint(tmp_path)
    assert CHECK.main(["--project-root", str(tmp_path)]) == 2
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "FAILED_READONLY_CHECK"
    assert not output["execution_authorization"]
    assert fingerprint(tmp_path) == before

# Synthetic coverage for the receipt-selected Inpainting publication chain.
def _sync_inpainting_snapshot(root):
    governance = root / "docs/governance"
    receipt = FIXTURE.load(governance / "CURRENT_STATUS_RECEIPT.json")
    receipt["files"] = {key: FIXTURE.ref(governance / name) for key, name in FIXTURE.DOC_NAMES.items()}
    FIXTURE.write_json(governance / "CURRENT_STATUS_RECEIPT.json", receipt)


def _inpainting_branch(root):
    return FIXTURE.load(root / "docs/governance/ALGORITHM_CONTRACT.json")["special_status_contracts"]["inpainting_raw_to_robot_v1"]


def _save_inpainting_branch(root, branch):
    path = root / "docs/governance/ALGORITHM_CONTRACT.json"
    value = FIXTURE.load(path)
    value["special_status_contracts"]["inpainting_raw_to_robot_v1"] = branch
    FIXTURE.write_json(path, value)
    _sync_inpainting_snapshot(root)


def _build_inpainting(root, publication="r03"):
    wrappers = [CHECK.INPAINTING_WRAPPER, "src/chaoyang/cli/__init__.py", "src/chaoyang/governance/current_r3_contracts.py"]
    core_paths = [f"src/chaoyang/synthetic_inpainting/core_{i:02}.py" for i in range(19)]
    paths = core_paths + wrappers + [f"src/chaoyang/synthetic_inpainting/dependency_{i:02}.py" for i in range(72)]
    for name in paths:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# synthetic frozen " + name + "\n")
    frozen = [FIXTURE.ref(root / name) for name in paths]
    manifest = {"baseline_id": CHECK.INPAINTING_ID, "selected_operation": CHECK.INPAINTING_OPERATION,
                "selected_baseline_count": 1, "actual_motion_source": "RAW_CONTROLLER_HEAD_PLUS_RAW_MANUS",
                "scope": {"completed_sessions": 199, "native_frames": 29322},
                "quality": {"all199_visual_accepted": False, "control_ground_truth": False},
                "adoption_sideeffects": {key: 0 for key in ("algorithm_calls", "training_steps", "files_deleted", "external_writes")},
                "algorithm_core": frozen[:19], "runtime_closure": frozen, "dependencies": frozen[:23]}
    pointer = FIXTURE.write_json(root / "inpainting_v1/MANIFEST.json", manifest)
    adoption = {"baseline_id": CHECK.INPAINTING_ID, "status": "PASSED", "baseline_pointer": pointer,
                "adoption_only": True, "control_ground_truth": False, "all199_visual_accepted": False,
                **{key: 0 for key in manifest["adoption_sideeffects"]}}
    adoption_ref = FIXTURE.write_json(root / "tasks/receipts/INPAINTING_RAW_TO_ROBOT_V1.json", adoption)
    branch = {"baseline_id": CHECK.INPAINTING_ID, "baseline_pointer": pointer, "publication_receipt": adoption_ref,
              "selected_operation": CHECK.INPAINTING_OPERATION, "selected_baseline_count": 1,
              "actual_motion_source": manifest["actual_motion_source"], "validated_scope": manifest["scope"],
              "quality": manifest["quality"], "control_ground_truth": False, "training_eligible": False,
              "task_packet_required": True, "current_runnable_operation": None,
              "code_closure": manifest["algorithm_core"], "runtime_closure": frozen,
              "dependencies": manifest["dependencies"]}
    original = {ref["path"]: ref for ref in frozen}
    replacement = {}
    if publication in {"r02", "r03"}:
        selection = root / CHECK.INPAINTING_WRAPPER
        selection.write_text("# synthetic R02 selection wrapper\n")
        parent_selection = FIXTURE.ref(selection)
        parent_packet = FIXTURE.write_json(root / "tasks/synthetic/R02_TASK_PACKET.json", {"synthetic_only": True})
        parent = {"schema": "inpainting-v1-governance-publication-supplement", "baseline_id": CHECK.INPAINTING_ID,
                  "baseline_pointer": pointer, "original_wrapper": original[str(selection)],
                  "current_wrapper": parent_selection, "task_packet": parent_packet,
                  "algorithm_changed": False, "files_deleted": 0, "algorithm_calls": 0}
        parent_ref = FIXTURE.write_json(root / "inpainting_v1/GOVERNANCE_PUBLICATION_R02.json", parent)
        branch["governance_publication_supplement"] = parent_ref
        replacement[str(selection)] = parent_selection
    if publication == "r03":
        rows = []
        for name in wrappers:
            path = root / name
            previous = parent_selection if name == CHECK.INPAINTING_WRAPPER else original[str(path)]
            path.write_text("# synthetic R03 " + name + "\n")
            current = FIXTURE.ref(path)
            replacement[str(path)] = current
            rows.append({"relative_path": name, "original": original[str(path)], "parent": previous,
                         "current": current, "role": "GOVERNANCE_RUNTIME_ONLY"})
        helper = root / "src/chaoyang/inpainting_candidate/publication.py"
        helper.parent.mkdir(parents=True, exist_ok=True)
        helper.write_text("# synthetic signed helper; never imported\n")
        task = {"task_id": "inpainting_generalization_g0_publication_synthetic",
                "deletion_allowed": False, "algorithm_execution_allowed": False,
                "write_set": wrappers + ["inpainting_v1/GOVERNANCE_PUBLICATION_R03.json"]}
        task_ref = FIXTURE.write_json(root / "tasks/synthetic/R03_TASK_PACKET.json", task)
        r03 = {"schema": "inpainting-v1-runtime-publication-r03", "baseline_id": CHECK.INPAINTING_ID,
               "baseline_pointer": pointer, "parent_supplement": parent_ref, "wrapper_overrides": rows,
               "additional_runtime": [FIXTURE.ref(helper)], "task_packet": task_ref, "algorithm_changed": False,
               **{key: 0 for key in ("files_deleted", "model_calls", "training_steps", "external_writes")}}
        branch["governance_publication_supplement"] = FIXTURE.write_json(root / "inpainting_v1/GOVERNANCE_PUBLICATION_R03.json", r03)
    branch["runtime_closure"] = [replacement.get(ref["path"], ref) for ref in frozen]
    branch["dependencies"] = [replacement.get(ref["path"], ref) for ref in manifest["dependencies"]]
    if publication == "r03":
        branch["runtime_closure"] += [FIXTURE.ref(helper)]
    authority_path = root / "docs/governance/DOC_AUTHORITY_MAP.json"
    authority = FIXTURE.load(authority_path)
    authority["documents"].append({"document_id": "inpainting_raw_to_robot_v1", "status": "CURRENT",
                                   **pointer, "evidence_receipts": [adoption_ref]})
    FIXTURE.write_json(authority_path, authority)
    _save_inpainting_branch(root, branch)
    return root


@pytest.mark.parametrize("publication,count,changed", [(None, 94, 0), ("r02", 94, 1), ("r03", 95, 3)])
def test_receipt_selected_inpainting_publications_preserve_core_and_readonly(tree, publication, count, changed):
    _build_inpainting(tree, publication)
    before = fingerprint(tree)
    view = CHECK.inspect_project(tree)
    assert view["status"] == "PASSED_READONLY_BINDING"
    assert view["algorithm_core_files_sha_verified"] == 19
    assert view["runtime_sources_sha_verified"] == count
    assert len(view["signed_runtime_wrapper_differences"]) == changed
    assert len(view["selected_baselines"]) == 1
    assert view["selected_baselines"][0]["baseline_id"] == CHECK.INPAINTING_ID
    assert not view["selected_baselines"][0]["control_ground_truth"]
    assert not view["selected_baselines"][0]["training_eligible"]
    assert not view["execution_authorization"] and view["models_run"] == view["files_written"] == 0
    assert fingerprint(tree) == before


@pytest.mark.parametrize("change", ["sideeffect", "boolean_zero", "role", "wrapper_set", "helper_extra", "parent_binding", "task_permission", "task_scope"])
def test_r03_rejects_rehashed_but_unauthorized_publication(tree, change):
    _build_inpainting(tree)
    branch = _inpainting_branch(tree)
    path = tree / "inpainting_v1/GOVERNANCE_PUBLICATION_R03.json"
    value = FIXTURE.load(path)
    if change == "sideeffect": value["model_calls"] = 1
    elif change == "boolean_zero": value["model_calls"] = False
    elif change == "role": value["wrapper_overrides"][0]["role"] = "MODEL_ALGORITHM"
    elif change == "wrapper_set": value["wrapper_overrides"][0]["relative_path"] = "src/chaoyang/synthetic_inpainting/core_00.py"
    elif change == "helper_extra": value["additional_runtime"] += [value["additional_runtime"][0]]
    elif change == "parent_binding":
        parent_path = Path(value["parent_supplement"]["path"])
        parent = FIXTURE.load(parent_path)
        parent["baseline_id"] = "UNAUTHORIZED_BASELINE"
        value["parent_supplement"] = FIXTURE.write_json(parent_path, parent)
    else:
        task_path = Path(value["task_packet"]["path"])
        task = FIXTURE.load(task_path)
        if change == "task_permission": task["algorithm_execution_allowed"] = True
        else: task["write_set"] = []
        value["task_packet"] = FIXTURE.write_json(task_path, task)
    branch["governance_publication_supplement"] = FIXTURE.write_json(path, value)
    _save_inpainting_branch(tree, branch)
    with pytest.raises(CHECK.CheckError, match="INPAINTING_R03_"):
        CHECK.inspect_project(tree)


@pytest.mark.parametrize("name", ["src/chaoyang/synthetic_inpainting/core_00.py", CHECK.INPAINTING_WRAPPER,
                                  "src/chaoyang/inpainting_candidate/publication.py"])
def test_r03_never_excuses_tampered_core_wrapper_or_helper(tree, name):
    _build_inpainting(tree)
    (tree / name).write_text("tampered after the signed publication\n")
    with pytest.raises(CHECK.CheckError, match="INPAINTING_.*SHA_DRIFT"):
        CHECK.inspect_project(tree)


def test_r03_rejects_rebinding_an_unapproved_runtime_source(tree):
    _build_inpainting(tree)
    branch = _inpainting_branch(tree)
    path = tree / "src/chaoyang/synthetic_inpainting/dependency_10.py"
    path.write_text("changed nonwrapper; even a new runtime SHA must be rejected\n")
    branch["runtime_closure"] = [FIXTURE.ref(path) if row["path"] == str(path) else row for row in branch["runtime_closure"]]
    _save_inpainting_branch(tree, branch)
    with pytest.raises(CHECK.CheckError, match="INPAINTING_R03_RUNTIME_BINDING_DRIFT"):
        CHECK.inspect_project(tree)


def test_95_sources_without_receipt_selected_r03_are_rejected(tree):
    _build_inpainting(tree)
    branch = _inpainting_branch(tree)
    branch.pop("governance_publication_supplement")
    _save_inpainting_branch(tree, branch)
    with pytest.raises(CHECK.CheckError, match="INPAINTING_RUNTIME_CLOSURE_DRIFT"):
        CHECK.inspect_project(tree)
