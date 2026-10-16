"""CPU-only usage snapshot consistency; no model or video inference."""
from pathlib import Path
import re

import pytest

from chaoyang.governance.common import artifact_ref, load_json
from chaoyang.governance.baseline_v1_source_release import verify_bundle
from chaoyang.governance.publish_project_branch_convergence_v1 import refs

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/"_run/current/project_usage_snapshot_v1_20261008/attempts/attempt_0001"
DOC = ROOT/"docs/current/PROJECT_USAGE_V1_ZH.md"


def snapshot():
    return load_json(ROOT/"docs/current/PROJECT_USAGE_V1_SNAPSHOT.json")


def test_as_of_snapshot_is_not_live_or_authorization():
    s = snapshot()
    assert s["status"] == "IMMUTABLE_AS_OF_USAGE_NOT_LIVE_ROUTING"
    assert s["execution_authorization"] is False
    assert s["before_documentation_task_live_state"]["active_tasks"] == []
    assert s["authority_revision_at_start"] == 19197
    assert (OUT/"USAGE_SNAPSHOT.json").read_bytes() == (ROOT/"docs/current/PROJECT_USAGE_V1_SNAPSHOT.json").read_bytes()


def test_selected_aliases_equal_frozen_contract():
    s = snapshot(); d = load_json(ROOT/"docs/current/DUAL_BRANCH_BASELINE_V1.json")
    assert s["selected_alias_count"] == 2
    assert s["selected_aliases"] == d["aliases"]


def test_visual_scope_and_actual_source_are_not_hawor():
    v = snapshot()["selected_aliases"]["EXACT78_PHASE_V1"]
    assert v["actual_motion_source"] == "PICO_OPENXR_SAME_SESSION"
    assert v["validated_scope"]["completed"] == 234
    assert v["validated_scope"]["total_frames"] == 78989
    assert v["pure_rgb"] is False
    assert v["validated_scope"]["source_0915"] == "NOT_COVERED"
    assert v["training_eligible"] is False and v["control_ground_truth"] is False


def test_manus_scope_does_not_expand_to_batch():
    m = snapshot()["selected_aliases"]["MANUS_PHASE_V1"]
    assert m["validated_scope"]["frames"] == 176
    assert m["validated_scope"]["batch_100_validated"] is False
    assert m["image_alignment_status"] == "VISIBLE_MISMATCH_NOT_ACCEPTED"
    assert m["implementation_owner"] == "OTHER_BRANCH_UNCHANGED"


def test_adapters_and_pure_rgb_not_promoted():
    s = snapshot(); e = s["input_adapter_experiments"]
    assert e["frames"] == 439 and e["new_baseline"] is False
    assert e["ai_sample_quality"] == "NOT_PASSED"
    assert e["user_full_film_review"] == "PENDING"
    assert s["future_pure_rgb_hawor"]["current_runnable_operation"] is None


def test_scene_geometries_not_consumed():
    v = snapshot()["visual_pipeline"]
    assert v["rendering_engine"] == "CPU_ZBUFFER_NOT_BLENDER"
    assert not any(v[k] for k in ("scene_depth_consumed", "object6d_consumed", "contact_truth_consumed"))


def test_protected_sha_and_frozen_source_copies():
    p = load_json(OUT/"REFERENCE_BINDINGS.json")["protected"]
    assert len(p) == 370 and refs(tuple(p)) == p
    assert verify_bundle(ROOT, ROOT/"baseline_v1")["copied_files_sha_verified"] == 163


def test_reference_documents_have_exact_original_byte_copies():
    r = load_json(OUT/"REFERENCE_BINDINGS.json")
    for row in r["archived_documents"].values():
        copied = row["immutable_copy"]
        assert artifact_ref(copied["path"]) == copied
        assert all(copied[k] == row["original"][k] for k in ("bytes", "sha256"))
    assert "docs/governance/ALGORITHM_CONTRACT.json" in r["archived_documents"]


@pytest.mark.parametrize("name", ["PROJECT_USAGE_V1_ZH.md", "AI_WORK_ENTRY_ZH.md", "README_ZH.md", "PROJECT_BRANCH_MAP_ZH.md"])
def test_local_navigation_links_exist(name):
    p = ROOT/"docs/current"/name
    for target in re.findall(r"\]\(([^)]+)\)", p.read_text()):
        assert (p.parent/target.split("#", 1)[0]).resolve().exists(), target


def test_new_usage_guide_reachable_from_all_current_entries():
    for name in ("AI_WORK_ENTRY_ZH.md", "README_ZH.md", "PROJECT_BRANCH_MAP_ZH.md"):
        assert "PROJECT_USAGE_V1_ZH.md" in (ROOT/"docs/current"/name).read_text()


def test_cleanup_counts_are_previous_maintenance_not_new_work():
    c = snapshot()["cleanup_as_of"]
    assert c["old_files_deleted"] == 18
    assert c["cache_files_deleted"] == 5261
    assert c["cache_allocated_file_bytes"] == 105481728
    assert c["large_history_purge_claimed"] is False
    assert "本次只记录，不重复删除" in DOC.read_text()


def test_new_changes_preserve_immutable_baseline_and_owner():
    f = snapshot()["future_changes"]
    assert f["new_finite_packet_required"] is True
    assert f["overwrite_immutable_results_allowed"] is False
    assert f["overwrite_frozen_source_package_allowed"] is False
    assert f["preserve_other_branch_ownership"] is True
