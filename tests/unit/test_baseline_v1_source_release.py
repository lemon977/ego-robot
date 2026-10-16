"""Source release integrity only; no models or video re-render."""
import ast
import copy
import json
from pathlib import Path

import pytest

from chaoyang.governance import baseline_v1_source_release as release
from chaoyang.governance.common import artifact_ref, load_json
from test_frozen_bundle_runtime_separation_v1 import (
    verify_frozen_imports, verify_frozen_release, verify_pinned_runtime,
)

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "baseline_v1"


def test_actual_frozen_bundle_sha_and_release_contract_are_intact():
    result = verify_frozen_release(ROOT, BUNDLE)
    assert result["counts"]["source_files"] == 111
    assert result["counts"]["metadata_files"] == 52
    assert result["copied_files_sha_verified"] == 163
    assert result["algorithm_changed"] is False
    assert result["production_loader_changed"] is False


def test_shared_code_has_one_copy_and_no_links():
    manifest = load_json(BUNDLE / "MANIFEST.json")
    rows = manifest["sources"]
    assert len({row["bundle_path"] for row in rows}) == len(rows)
    shared = [row for row in rows if len(row["owners"]) == 2]
    assert len(shared) == 75
    for row in shared:
        source, destination = Path(row["source"]["path"]), BUNDLE / row["bundle_path"]
        assert source.stat().st_ino != destination.stat().st_ino
        assert not destination.is_symlink()


def test_manifest_names_only_two_selected_production_operations():
    manifest = load_json(BUNDLE / "MANIFEST.json")
    assert manifest["selected_operations"] == release.SELECTED_OPERATIONS
    selected = {Path(row["bundle_path"]).stem for row in manifest["sources"] if row["standalone_operation_selected"]}
    assert selected == set(release.SELECTED_OPERATIONS.values())
    for row in manifest["sources"]:
        if "huro" in row["bundle_path"] or "newraw" in row["bundle_path"]:
            assert row["standalone_operation_selected"] is False


def test_dynamic_sam_adapter_is_explicitly_included():
    assert (BUNDLE / "code/chaoyang/pipeline/sam31_compat_adapter_v1.py").is_file()
    assert any(row["relative_path"].endswith("sam31_compat_adapter_v1.py")
               for row in load_json(BUNDLE / "MANIFEST.json")["sources"])


def test_static_propainter_parameters_preserved():
    params = load_json(BUNDLE / "PARAMETERS.json")
    clean = next(row for row in params["modules"] if row["source"]["path"].endswith("exact78_baseline/clean.py"))
    assert clean["constants"]["SETTINGS"] == {"mask_dilation":4, "raft_iter":20, "neighbor_length":10,
        "ref_stride":10, "subvideo_length":80, "fp16":True}


def test_no_unlisted_files_models_media_or_bytecode_in_bundle():
    manifest = load_json(BUNDLE / "MANIFEST.json")
    expected = {row["bundle_path"] for row in manifest["sources"]+manifest["metadata"]}
    expected.update(release.RELEASE_FILES)
    expected.update({"README.md", "AGENTS.md", "CODE_MAP_ZH.md"})
    actual = {str(path.relative_to(BUNDLE)) for path in BUNDLE.rglob("*") if path.is_file()}
    assert actual == expected
    assert not any(Path(name).suffix in {".mp4", ".hdf5", ".npz", ".pt", ".pth", ".pyc"} for name in actual)


def test_live_sources_allow_only_the_explicit_signed_runtime_wrappers():
    result = verify_pinned_runtime(ROOT, load_json(BUNDLE / "MANIFEST.json"))
    assert result["historical_runtime_ref_count"] == 93
    assert result["historical_nonwrapper_refs_sha_verified"] == 91
    assert len(result["historical_wrapper_records"]) == 2
    assert result["effective_runtime_refs_verified"] == 93
    assert set(result["signed_wrapper_differences"]) <= {
        "src/chaoyang/governance/current_r3_contracts.py",
        "src/chaoyang/governance/exact78_visual_selection_v1.py",
    }


@pytest.fixture
def tiny_plan(tmp_path):
    source = tmp_path / "fixture.py"
    source.write_text("def answer():\n    return 42\n")
    return {"sources":[{"source":artifact_ref(source),"bundle_path":"code/fixture.py"}],"metadata":[]}


def test_copy_is_mechanical_and_fresh(tmp_path, tiny_plan):
    destination = tmp_path / "bundle"
    release.copy_plan(tiny_plan, destination)
    assert artifact_ref(destination/"code/fixture.py")["sha256"] == tiny_plan["sources"][0]["source"]["sha256"]
    with pytest.raises(RuntimeError, match="FRESH_BUNDLE_REQUIRED"):
        release.copy_plan(tiny_plan, destination)


@pytest.mark.parametrize("bad_path", ["../escape.py", "/absolute.py"])
def test_escape_path_rejected_before_creation(tmp_path, tiny_plan, bad_path):
    tiny_plan["sources"][0]["bundle_path"] = bad_path
    destination = tmp_path / "bundle"
    with pytest.raises(RuntimeError, match="SCOPE_ESCAPE"):
        release.copy_plan(tiny_plan, destination)
    assert not destination.exists()


def test_duplicate_paths_rejected_before_creation(tmp_path, tiny_plan):
    tiny_plan["sources"].append(copy.deepcopy(tiny_plan["sources"][0]))
    destination = tmp_path / "bundle"
    with pytest.raises(RuntimeError, match="DUPLICATE_BUNDLE_PATH"):
        release.copy_plan(tiny_plan, destination)
    assert not destination.exists()


def test_source_drift_rejected_before_creation(tmp_path, tiny_plan):
    tiny_plan["sources"][0]["source"]["sha256"] = "0"*64
    destination = tmp_path / "bundle"
    with pytest.raises(RuntimeError, match="SOURCE_DRIFT"):
        release.copy_plan(tiny_plan, destination)
    assert not destination.exists()


def test_import_closure_files_are_all_covered_by_bundle():
    # The release contract concerns sealed ASTs, not later runtime imports.
    manifest = load_json(BUNDLE / "MANIFEST.json")
    assert len(verify_frozen_imports(ROOT, BUNDLE, manifest)) == 111
    original_modules = {row["relative_path"] for row in manifest["sources"]}
    assert set(release.VISUAL_SEEDS + release.MANUS_SEEDS) <= original_modules
