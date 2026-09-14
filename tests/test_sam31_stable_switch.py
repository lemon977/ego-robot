from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools.validate_sam31_stable_switch import validate


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "docs/organization/2026-09-03/SAM31_STABLE_SWITCH_PREPARED.json"


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_prepared_or_post_switch_state_validates_without_reading_weight_bytes():
    result = validate(state="auto", deep_weight_hash=False)
    assert not result["errors"], result["errors"]
    assert result["status"].startswith("PASS_")
    assert result["read_only"] is True
    assert result["deep_weight_hash"] is False
    assert result["checks"]["checkpoint_hardlink"]["same_inode"] is True
    assert result["checks"]["checkpoint_hardlink"]["fresh_sha256"] is None
    assert result["checks"]["pin_closure"]["tasks"].keys() == {
        "cap004",
        "chips",
        "poker",
    }


def test_transaction_scope_is_exact_and_excludes_active_or_frozen_paths():
    plan = _load(PLAN)
    transaction = plan["transaction"]
    existing = {item["path"] for item in transaction["existing_file_preimages"]}
    new = {item["path"] for item in transaction["new_paths"]}
    expected_tools = {
        "tools/launch_task32_autonomous_gpu1.py",
        "tools/profile_mask_wearable_sam31_resource.py",
        "tools/run_004_wearable_union_review_candidate_t0.py",
        "tools/run_assisted_bilateral_sam31_point_canary.py",
        "tools/run_chips001_mask_frame_canary.py",
        "tools/run_chips001_pico_mask_temporal.py",
        "tools/run_d1_d4_live_producer.py",
        "tools/run_mask_wearable_semantic_rescue.py",
        "tools/run_newtask_baseline_sam31_mask_probe.py",
        "tools/run_pico_geometry_raw_point_mask_canary.py",
        "tools/run_shared_mask_successor_canary.py",
    }
    assert {path for path in existing if path.startswith("tools/")} == expected_tools
    assert {
        "systems/mask/system.manifest.json",
        "systems/registry.json",
        "tasks/cap004/task.manifest.json",
        "tasks/chips/task.manifest.json",
        "tasks/poker/task.manifest.json",
        "tasks/registry.json",
        "THIRD_PARTY_NOTICES.md",
    }.issubset(existing)
    assert new == {
        "third_party/SAM3",
        "systems/mask/configs/sam31_runtime_source_v1.json",
        "docs/organization/2026-09-03/SAM31_STABLE_SWITCH_EXECUTION.json",
    }
    frozen = {item["path"] for item in plan["frozen_contracts"]}
    assert frozen.isdisjoint(existing | new)
    assert not any(path == "_run" or path.startswith("_run/") for path in existing | new)
    assert not any(path.startswith("third_party/HaWoR/") for path in existing | new)
    assert not any("/runs/" in path for path in existing | new)
    assert transaction["authorized"] is False
    assert transaction["permanent_deletes"] == 0
    assert transaction["checkpoint_copy_bytes"] == 0
    assert transaction["historical_artifact_rewrites"] == 0


def test_frozen_v2_chain_and_legacy_source_retention_are_byte_exact():
    plan = _load(PLAN)
    frozen = {item["path"]: item["sha256"] for item in plan["frozen_contracts"]}
    for relative, expected in frozen.items():
        assert _sha256(ROOT / relative) == expected

    v1 = _load(ROOT / "systems/mask/configs/object_authority_state_layout_ab_v1.json")
    v2 = _load(ROOT / "systems/mask/configs/object_authority_state_layout_ab_v2.json")
    assert v2["base_contract"]["sha256"] == frozen[
        "systems/mask/configs/object_authority_state_layout_ab_v1.json"
    ]
    evidence = v1["static_source_audit"]["evidence"]
    assert len(evidence) == 3
    assert all(item["path"].startswith(plan["legacy_source_retention"]["root"] + "/") for item in evidence)
    assert plan["legacy_source_retention"]["policy"] == "HARD_KEEP"


def test_reference_rewrite_is_exact_not_a_threshold():
    plan = _load(PLAN)
    inventory = plan["reference_inventory"]
    prepared = inventory["prepared_old_token_counts"]
    post = inventory["post_switch_old_token_counts"]
    assert len(prepared) == inventory["prepared_old_token_file_count"] == 13
    assert sum(prepared.values()) == inventory["prepared_old_token_match_count"] == 26
    assert post == {
        "systems/mask/configs/object_authority_state_layout_ab_v1.json": 3
    }
    assert set(post).issubset(prepared)
