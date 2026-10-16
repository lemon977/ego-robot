"""Synthetic single-V1 handoff checks; project-local scratch, no model imports."""
from __future__ import annotations

import builtins
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("inpainting_readonly_handoff_checker",
                                            ROOT / "scripts/maintenance/project_handoff_check_v1.py")
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)
DOC_NAMES = {
    "doc_authority_map": "DOC_AUTHORITY_MAP.json",
    "algorithm_contract": "ALGORITHM_CONTRACT.json",
    "baseline_registry_v2": "CURRENT_BASELINE_REGISTRY_V2.json",
    "task_packet_index_current": "INDEX.json",
    "task_state": "LONG_HORIZON_TASK_STATE.json",
}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return CHECK.file_ref(path)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def refresh(root):
    gov = root / "docs/governance"
    write(gov / "CURRENT_STATUS_RECEIPT.json", {
        "governance_revision": 1, "generation_id": "gov-000001-synthetic",
        "files": {key: CHECK.file_ref(gov / name) for key, name in DOC_NAMES.items()},
    })


@pytest.fixture
def tree(tmp_path):
    assert tmp_path.resolve().is_relative_to(ROOT), "Use a project-local --basetemp"
    root = tmp_path / "fixture_project"
    runtime = []
    for index in range(93):
        path = root / "src/chaoyang/pick_place_1008" / f"source_{index:03}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# Source {index}; never imported\n", encoding="utf-8")
        runtime.append(CHECK.file_ref(path))
    wrapper = root / CHECK.INPAINTING_WRAPPER
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.write_text("# original selection wrapper\n", encoding="utf-8")
    runtime.append(CHECK.file_ref(wrapper))
    manifest = {
        "baseline_id": CHECK.INPAINTING_ID, "selected_baseline_count": 1,
        "selected_operation": CHECK.INPAINTING_OPERATION,
        "parent_implementation": "V9_REVERSE_NATIVE_BOUNDARY_ORIGINAL_ALL199",
        "actual_motion_source": "RAW_CONTROLLER_HEAD_PLUS_RAW_MANUS",
        "scope": {"completed_sessions": 199, "native_frames": 29322},
        "quality": {"control_ground_truth": False, "all199_visual_accepted": False},
        "adoption_sideeffects": {"algorithm_calls": 0, "training_steps": 0,
                                 "files_deleted": 0, "external_writes": 0},
        "algorithm_core": runtime[:19], "runtime_closure": runtime,
    }
    pointer = write(root / "inpainting_v1/MANIFEST.json", manifest)
    adoption_ref = write(root / "tasks/receipts/INPAINTING_RAW_TO_ROBOT_V1.json", {
        "baseline_id": CHECK.INPAINTING_ID, "status": "PASSED", "baseline_pointer": pointer,
        "adoption_only": True, "control_ground_truth": False, "all199_visual_accepted": False,
        **manifest["adoption_sideeffects"],
    })
    wrapper.write_text("# explicitly signed publication wrapper\n", encoding="utf-8")
    live = runtime[:-1] + [CHECK.file_ref(wrapper)]
    task_ref = write(root / "tasks/receipts/WRAPPER_PUBLICATION_PACKET.json", {"synthetic": True})
    supplement_ref = write(root / "inpainting_v1/GOVERNANCE_PUBLICATION_R02.json", {
        "schema": "inpainting-v1-governance-publication-supplement", "baseline_id": CHECK.INPAINTING_ID,
        "baseline_pointer": pointer, "original_wrapper": runtime[-1], "current_wrapper": live[-1],
        "task_packet": task_ref, "algorithm_changed": False, "files_deleted": 0, "algorithm_calls": 0,
    })
    header = {"governance_revision": 1, "generation_id": "gov-000001-synthetic"}
    gov = root / "docs/governance"
    write(gov / DOC_NAMES["algorithm_contract"], {**header, "special_status_contracts": {
        "inpainting_raw_to_robot_v1": {
            "baseline_id": CHECK.INPAINTING_ID, "selected_baseline_count": 1,
            "baseline_pointer": pointer, "publication_receipt": adoption_ref,
            "selected_operation": CHECK.INPAINTING_OPERATION,
            "actual_motion_source": manifest["actual_motion_source"],
            "validated_scope": manifest["scope"], "quality": manifest["quality"],
            "code_closure": manifest["algorithm_core"], "runtime_closure": live,
            "governance_publication_supplement": supplement_ref,
            "control_ground_truth": False, "training_eligible": False,
            "task_packet_required": True, "current_runnable_operation": None,
        },
        # An old contract cannot take priority simply by remaining present.
        "dual_branch_phase_baselines_v1": {"selected_alias_count": 2},
    }})
    write(gov / DOC_NAMES["doc_authority_map"], {**header, "documents": [{
        "document_id": "inpainting_raw_to_robot_v1", "status": "CURRENT", **pointer,
        "evidence_receipts": [adoption_ref],
    }]})
    write(gov / DOC_NAMES["baseline_registry_v2"], {**header, "entries": []})
    write(gov / DOC_NAMES["task_packet_index_current"], {**header, "task_packets": []})
    write(gov / DOC_NAMES["task_state"], {**header, "tasks": []})
    refresh(root)
    return root


def algorithm(root):
    path = root / "docs/governance/ALGORITHM_CONTRACT.json"
    return path, load(path)


def fingerprint(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


def test_default_is_single_adopted_v1_with_19_core_and_94_runtime(tree):
    before = fingerprint(tree)
    result = CHECK.inspect_project(tree)
    assert [row["phase_alias"] for row in result["selected_baselines"]] == [CHECK.INPAINTING_ID]
    assert result["selected_baselines"][0]["validated_scope"] == {"completed_sessions": 199, "native_frames": 29322}
    assert result["algorithm_core_files_sha_verified"] == 19
    assert result["runtime_sources_sha_verified"] == 94
    assert result["signed_runtime_wrapper_differences"] == [CHECK.INPAINTING_WRAPPER]
    assert result["other_branches"]["manus_0916"].startswith("OUT_OF_SCOPE")
    assert not result["execution_authorization"] and result["models_run"] == result["files_written"] == 0
    assert fingerprint(tree) == before
    assert CHECK.inspect_project(tree, CHECK.INPAINTING_ID)["selected_baselines"] == result["selected_baselines"]


@pytest.mark.parametrize("alias", ["EXACT78_PHASE_V1", "MANUS_PHASE_V1", "Cards042_V2", "latest"])
def test_old_alias_not_selected_under_current_v1(tree, alias):
    with pytest.raises(CHECK.CheckError, match="ALIAS_NOT_SELECTED"):
        CHECK.inspect_project(tree, alias)


@pytest.mark.parametrize("relative,code", [
    ("inpainting_v1/MANIFEST.json", "INPAINTING_MANIFEST_SHA_DRIFT"),
    ("tasks/receipts/INPAINTING_RAW_TO_ROBOT_V1.json", "INPAINTING_ADOPTION_SHA_DRIFT"),
    ("src/chaoyang/pick_place_1008/source_000.py", "INPAINTING_CORE_SHA_DRIFT"),
    ("src/chaoyang/pick_place_1008/source_050.py", "INPAINTING_RUNTIME_SHA_DRIFT"),
    (CHECK.INPAINTING_WRAPPER, "INPAINTING_RUNTIME_SHA_DRIFT"),
    ("inpainting_v1/GOVERNANCE_PUBLICATION_R02.json", "INPAINTING_WRAPPER_SUPPLEMENT_SHA_DRIFT"),
    ("tasks/receipts/WRAPPER_PUBLICATION_PACKET.json", "INPAINTING_WRAPPER_SUPPLEMENT_PACKET_DRIFT"),
])
def test_tampered_required_evidence_rejected(tree, relative, code):
    path = tree / relative
    path.write_bytes(path.read_bytes() + b"\n# tampered")
    with pytest.raises(CHECK.CheckError, match=code):
        CHECK.inspect_project(tree)


def test_new_signed_runtime_cannot_rebind_nonwrapper_algorithm(tree):
    path, value = algorithm(tree)
    branch = value["special_status_contracts"]["inpainting_raw_to_robot_v1"]
    source = Path(branch["runtime_closure"][50]["path"])
    source.write_text("# signed but unauthorized algorithm change\n", encoding="utf-8")
    branch["runtime_closure"][50] = CHECK.file_ref(source)
    write(path, value)
    refresh(tree)
    with pytest.raises(CHECK.CheckError, match="INPAINTING_RUNTIME_UNAUTHORIZED_REBIND"):
        CHECK.inspect_project(tree)


def test_missing_wrapper_supplement_is_not_excused_by_current_signature(tree):
    path, value = algorithm(tree)
    del value["special_status_contracts"]["inpainting_raw_to_robot_v1"]["governance_publication_supplement"]
    write(path, value)
    refresh(tree)
    with pytest.raises(CHECK.CheckError, match="INPAINTING_WRAPPER_SUPPLEMENT_ABSENT"):
        CHECK.inspect_project(tree)


def test_rebound_supplement_must_keep_original_manifest_and_wrapper(tree):
    path, value = algorithm(tree)
    branch = value["special_status_contracts"]["inpainting_raw_to_robot_v1"]
    supplement_path = Path(branch["governance_publication_supplement"]["path"])
    supplement = load(supplement_path)
    supplement["original_wrapper"]["sha256"] = "0" * 64
    branch["governance_publication_supplement"] = write(supplement_path, supplement)
    write(path, value)
    refresh(tree)
    with pytest.raises(CHECK.CheckError, match="INPAINTING_WRAPPER_SUPPLEMENT_BINDING_DRIFT"):
        CHECK.inspect_project(tree)


def test_adoption_semantics_checked_even_after_receipt_is_rebound(tree):
    path, value = algorithm(tree)
    branch = value["special_status_contracts"]["inpainting_raw_to_robot_v1"]
    adoption_path = Path(branch["publication_receipt"]["path"])
    adoption = load(adoption_path)
    adoption["baseline_id"] = "OLD_V13"
    branch["publication_receipt"] = write(adoption_path, adoption)
    authority_path = tree / "docs/governance/DOC_AUTHORITY_MAP.json"
    authority = load(authority_path)
    authority["documents"][0]["evidence_receipts"] = [branch["publication_receipt"]]
    write(authority_path, authority)
    write(path, value)
    refresh(tree)
    with pytest.raises(CHECK.CheckError, match="INPAINTING_ADOPTION_BINDING_DRIFT"):
        CHECK.inspect_project(tree)


def test_checker_does_not_import_chaoyang_or_models(tree, monkeypatch):
    original = builtins.__import__
    seen = []

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"chaoyang", "torch", "cv2", "transformers"}:
            seen.append(name)
            raise AssertionError("Algorithm or model import during read-only handoff")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    CHECK.inspect_project(tree)
    assert seen == []
