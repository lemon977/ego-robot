"""CPU-only adoption and negative-entry tests; never import an algorithm.

Run with a project-local ``--basetemp`` and disable pytest's cache provider.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from chaoyang.governance.common import artifact_ref
from chaoyang.governance import inpainting_v1_selection as selection


PROJECT = Path(__file__).resolve().parents[2]
TAKEOVER = (
    "run_exact78_takeover_audit_v1",
    "run_exact78_takeover_baseline_v1",
    "run_exact78_takeover_cleanup_v1",
)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def manifest_value() -> dict:
    return {
        "baseline_id": selection.BASELINE_ID,
        "selected_operation": selection.OPERATION,
        "selected_baseline_count": 1,
        "actual_motion_source": "RAW_CONTROLLER_HEAD_PLUS_RAW_MANUS",
        "scope": {"completed_sessions": 199, "native_frames": 29322},
        "quality": {"control_ground_truth": False, "all199_visual_accepted": False},
        "adoption_sideeffects": {
            "algorithm_calls": 0, "training_steps": 0,
            "files_deleted": 0, "external_writes": 0,
        },
        "algorithm_core": [], "runtime_closure": [],
        "dependencies": [], "inputs": {},
        "claim_limit": "Adoption only; no all-frame quality or training qualification.",
    }


def write_receipt(root: Path, **overrides) -> None:
    write_json(root / selection.RECEIPT, {
        "status": "PASSED", "baseline_id": selection.BASELINE_ID,
        "baseline_pointer": artifact_ref(root / selection.MANIFEST),
        **overrides,
    })


@pytest.fixture
def project_root(tmp_path):
    # The repository protocol prohibits even test scratch output outside it.
    assert tmp_path.resolve().is_relative_to(PROJECT), "Use a project-local --basetemp"
    root = tmp_path / "fixture_project"
    root.mkdir()
    return root


@pytest.fixture
def adopted(project_root):
    root = project_root
    write_json(root / selection.MANIFEST, manifest_value())
    write_receipt(root)
    write_json(root / "archive/inpainting_baselines/INDEX.json", {"history": []})
    write_json(root / "docs/governance/RETIRED_OPERATION_ENTRYPOINTS.json", {
        "schema_version": "CHAOYANG_RETIRED_ENTRYPOINTS_V1", "operations": {},
    })
    write_json(root / "docs/current/ROBOT100_HARDWARE_BASELINE_V16.json", {
        "baseline_id": "0916_HARDWARE_WORLD_WRIST_V16", "baseline_selected": True,
    })
    return root


def test_draft_without_receipt_is_not_selected(project_root):
    assert selection.selected(project_root) is None
    write_json(project_root / selection.MANIFEST, manifest_value())
    assert selection.selected(project_root) is None
    assert selection.contract(project_root) == {}
    registry, layout = {"entries": ["retained"]}, {"canonical_documents": {}}
    selection.overlay_registry(registry, layout, project_root)
    assert registry == {"entries": ["retained"]}
    assert layout == {"canonical_documents": {}}


def test_receipt_without_manifest_is_rejected(project_root):
    write_json(project_root / selection.RECEIPT, {"status": "PASSED"})
    with pytest.raises(RuntimeError, match="INPAINTING_V1_MANIFEST_ABSENT"):
        selection.selected(project_root)


def test_complete_adoption_has_one_baseline_and_exact_original_denominator(adopted):
    value = selection.selected(adopted)
    assert value["selected_baseline_count"] == 1
    assert value["scope"] == {"completed_sessions": 199, "native_frames": 29322}
    contract = selection.contract(adopted)["inpainting_raw_to_robot_v1"]
    assert contract["baseline_pointer"] == artifact_ref(adopted / selection.MANIFEST)
    assert contract["publication_receipt"] == artifact_ref(adopted / selection.RECEIPT)
    assert contract["selected_operation"] == selection.OPERATION
    assert contract["current_runnable_operation"] is None
    assert contract["task_packet_required"] is True
    assert contract["algorithm_changed"] is False
    assert contract["training_steps_authorized_without_new_packet"] == 0
    assert contract["control_ground_truth"] is contract["training_eligible"] is False


@pytest.mark.parametrize("path,bad", [
    (("baseline_id",), "HISTORICAL_V13"),
    (("selected_operation",), TAKEOVER[1]),
    (("selected_baseline_count",), 2),
    (("actual_motion_source",), "PURE_RGB_HAWOR"),
    (("scope", "completed_sessions"), 198),
    (("scope", "native_frames"), 29321),
    (("quality", "control_ground_truth"), True),
    (("quality", "all199_visual_accepted"), True),
    (("quality", "control_ground_truth"), 0),
    (("adoption_sideeffects", "algorithm_calls"), 1),
    (("adoption_sideeffects", "training_steps"), 1),
    (("adoption_sideeffects", "files_deleted"), 1),
    (("adoption_sideeffects", "external_writes"), 1),
])
def test_signed_but_invalid_selection_is_rejected(adopted, path, bad):
    value = manifest_value()
    target = value
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = bad
    write_json(adopted / selection.MANIFEST, value)
    # Rebind the receipt so this tests semantics rather than a generic SHA drift.
    write_receipt(adopted)
    with pytest.raises(RuntimeError, match="INPAINTING_V1_ADOPTION_DRIFT"):
        selection.selected(adopted)


@pytest.mark.parametrize("overrides", [
    {"status": "RUNNING"},
    {"baseline_id": "OLD_BASELINE"},
    {"baseline_pointer": {"path": "/not/current", "sha256": "0" * 64, "bytes": 0}},
])
def test_wrong_receipt_is_rejected(adopted, overrides):
    write_receipt(adopted, **overrides)
    with pytest.raises(RuntimeError, match="INPAINTING_V1_ADOPTION_DRIFT"):
        selection.selected(adopted)


def test_manifest_byte_drift_is_rejected(adopted):
    path = adopted / selection.MANIFEST
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="INPAINTING_V1_ADOPTION_DRIFT"):
        selection.selected(adopted)


@pytest.mark.parametrize("bad_field", [None, "baseline_id", "baseline_pointer", "original_wrapper", "current_wrapper", "algorithm_changed"])
def test_governance_supplement_is_explicit_and_does_not_rewrite_algorithm_manifest(adopted, bad_field):
    wrapper = adopted / "src/chaoyang/governance/inpainting_v1_selection.py"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text("# original selection wrapper\n", encoding="utf-8")
    original = artifact_ref(wrapper)
    value = manifest_value()
    value["runtime_closure"] = [original]
    write_json(adopted / selection.MANIFEST, value)
    write_receipt(adopted)
    frozen = artifact_ref(adopted / selection.MANIFEST)
    wrapper.write_text("# navigation compatibility correction only\n", encoding="utf-8")
    current = artifact_ref(wrapper)
    supplement = {
        "baseline_id": selection.BASELINE_ID, "baseline_pointer": frozen,
        "original_wrapper": original, "current_wrapper": current,
        "algorithm_changed": False,
    }
    if bad_field is not None:
        supplement[bad_field] = True if bad_field == "algorithm_changed" else "DRIFT"
    write_json(adopted / "inpainting_v1/GOVERNANCE_PUBLICATION_R02.json", supplement)
    if bad_field is None:
        branch = selection.contract(adopted)["inpainting_raw_to_robot_v1"]
        assert branch["runtime_closure"] == [current]
        assert branch["algorithm_changed"] is False
        assert branch["governance_publication_supplement"] == artifact_ref(adopted / "inpainting_v1/GOVERNANCE_PUBLICATION_R02.json")
    else:
        with pytest.raises(RuntimeError, match="INPAINTING_V1_PUBLICATION_SUPPLEMENT_DRIFT"):
            selection.contract(adopted)
    assert artifact_ref(adopted / selection.MANIFEST) == frozen


def test_registry_overlay_preserves_stage_counts_and_other_branch(adopted):
    registry = {
        "entries": [{"stage": "Clean", "current_counts": {"total": 156, "passed": 58}}],
        "parallel_pipelines": [{"pipeline": "manus_0916", "counts": {"total": 100}}],
    }
    before = deepcopy(registry)
    layout = {"canonical_documents": {"legacy_pointer": "KEEP"}}
    selection.overlay_registry(registry, layout, adopted)
    assert registry["entries"] == before["entries"]
    assert registry["parallel_pipelines"] == before["parallel_pipelines"]
    assert registry["selected_visual_baseline"]["selected_baseline_count"] == 1
    assert registry["selected_visual_baseline"]["other_manus_branch"] == "OUT_OF_SCOPE_UNCHANGED"
    assert layout["canonical_documents"]["legacy_pointer"] == "KEEP"
    assert layout["inpainting_default_entry"] == str(adopted / "inpainting_v1/README.md")


def test_legacy_annotations_do_not_rewrite_candidate_gate_or_original_pointers(adopted):
    original = adopted / "docs/current/EXACT78_VISUAL_BASELINE_V1.json"
    write_json(original, {"baseline_id": "EXACT78_VISUAL_BASELINE_V1"})
    old_ref = artifact_ref(original)
    candidate = {
        "execution_status": "CANDIDATE_FENCED_PICK_PLACE_TASK_PACKET_ONLY",
        "scope": "pick_place_1008", "plan_revision": "PICK_PLACE_1008_RAW_TO_ROBOT_V1",
        "task_packet_required": True, "candidate_operations": [selection.OPERATION, "audit"],
        "training_steps_authorized": 0, "current_runnable_operation": None,
        "code_closure": [{"path": "unchanged", "sha256": "0" * 64, "bytes": 1}],
    }
    special = {
        "baseline_v1_source_release": {"manifest": old_ref},
        "dual_branch_phase_baselines_v1": {"selected_alias_count": 2},
        "exact78_visual_baseline_v1": {"current_visual_operation": TAKEOVER[1], "baseline_pointer": old_ref},
        "pick_place_1008_candidate_v1": deepcopy(candidate),
        "manus_0916": {"operation": "run_robot100_v16_safe_replay"},
    }
    selection.annotate_legacy_contracts(special, adopted)
    for key, value in candidate.items():
        assert special["pick_place_1008_candidate_v1"][key] == value
    assert special["exact78_visual_baseline_v1"]["current_visual_operation"] is None
    assert special["exact78_visual_baseline_v1"]["baseline_pointer"] == old_ref
    assert special["manus_0916"] == {"operation": "run_robot100_v16_safe_replay"}
    assert artifact_ref(original) == old_ref


@pytest.mark.parametrize("live", [False, True])
def test_live_status_has_one_inpainting_and_other_readonly_branch(adopted, live):
    state = {
        "governance_revision": 123,
        "tasks": [{"task_id": "working", "status": "RUNNING" if live else "PASSED"},
                  {"task_id": "not_running", "status": "PENDING"}],
        "next_task": {"task_id": "working"} if live else None,
    }
    hardware = adopted / "docs/current/ROBOT100_HARDWARE_BASELINE_V16.json"
    before = artifact_ref(hardware)
    selection.publish_status(adopted, state, "2026-10-12T00:00:00Z")
    status = json.loads((adopted / "docs/current/STATUS.json").read_text(encoding="utf-8"))
    assert status["counts"] == {"selected_inpainting_baselines": 1,
                                "original_completed_sessions": 199, "original_native_frames": 29322}
    assert set(status["branches"]) == {"inpainting_visual", "manus_0916"}
    assert status["branches"]["manus_0916"]["ownership"] == "OTHER_BRANCH_READ_ONLY"
    assert status["active_tasks"] == (["working"] if live else [])
    assert status["control_ground_truth"] is status["training_eligible"] is False
    assert artifact_ref(hardware) == before


def cli_fixture(root):
    names = [selection.OPERATION, "run_exact78_pick_place_audit_candidate_v1"]
    write_json(root / "docs/governance/ALGORITHM_CONTRACT.json", {
        "special_status_contracts": {"pick_place_1008_candidate_v1": {
            "execution_status": "CANDIDATE_FENCED_PICK_PLACE_TASK_PACKET_ONLY",
            "scope": "pick_place_1008", "plan_revision": "PICK_PLACE_1008_RAW_TO_ROBOT_V1",
            "candidate_operations": names, "task_packet_required": True,
            "training_steps_authorized": 0, "current_runnable_operation": None,
            "code_closure": [{"path": str(root / "src/chaoyang/ops" / (name + ".py")),
                              "sha256": "0" * 64, "bytes": 0} for name in names],
        }},
    })
    write_json(root / "docs/governance/RETIRED_OPERATION_ENTRYPOINTS.json", {
        "schema_version": "CHAOYANG_RETIRED_ENTRYPOINTS_V1",
        "operations": {name: {"replacement": selection.OPERATION,
                              "reason": "Historical inpainting route"} for name in TAKEOVER},
    })
    write_json(root / "tasks/current/INDEX.json", {"task_packets": []})


@pytest.mark.parametrize("operation", TAKEOVER)
def test_real_cli_rejects_retired_takeover_before_algorithm_import(project_root, monkeypatch, operation):
    import chaoyang.cli as cli
    cli_fixture(project_root)
    monkeypatch.setattr(cli, "get_paths", lambda: SimpleNamespace(repo_root=project_root))
    imported = []
    monkeypatch.setattr(cli.importlib, "import_module", lambda name: imported.append(name))
    with pytest.raises(SystemExit, match="RETIRED_OPERATION"):
        cli.main(["run", operation])
    assert imported == []


def test_real_cli_selected_operation_without_packet_is_rejected_before_import(project_root, monkeypatch):
    import chaoyang.cli as cli
    from chaoyang.governance import exact78_operation_gate as gate
    cli_fixture(project_root)
    paths = lambda: SimpleNamespace(repo_root=project_root)
    monkeypatch.setattr(cli, "get_paths", paths)
    monkeypatch.setattr(gate, "get_paths", paths)
    imported = []
    monkeypatch.setattr(cli.importlib, "import_module", lambda name: imported.append(name))
    with pytest.raises(gate.Exact78GateDenied, match="exactly one execution_allowed"):
        cli.main(["run", selection.OPERATION])
    assert imported == []


def test_real_published_contract_matches_adoption_if_already_sealed():
    path = PROJECT / "docs/governance/ALGORITHM_CONTRACT.json"
    current = json.loads(path.read_text(encoding="utf-8"))
    branch = current.get("special_status_contracts", {}).get("inpainting_raw_to_robot_v1")
    if branch is None:
        pytest.skip("The adoption is still being prepared; no current contract claims it yet")
    assert branch == selection.contract(PROJECT)["inpainting_raw_to_robot_v1"]
    assert branch["selected_baseline_count"] == 1
    assert branch["validated_scope"]["completed_sessions"] == 199
    assert branch["validated_scope"]["native_frames"] == 29322
