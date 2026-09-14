from __future__ import annotations

import copy
import json
from pathlib import Path

from tools.validate_task_registry import (
    ValidationResult,
    validate_repository,
    validate_system_manifest,
    validate_task_manifest,
)


ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def _loaded_state():
    system_schema = _read("systems/schema/system_manifest.schema.json")
    task_schema = _read("tasks/schema/task_manifest.schema.json")
    system_registry = _read("systems/registry.json")
    systems = {}
    algorithms = {}
    weights = {}
    result = ValidationResult()
    for system_id in ("mask", "clean", "robot"):
        manifest = _read(f"systems/{system_id}/system.manifest.json")
        systems[system_id] = manifest
        algorithms[system_id], weights[system_id] = validate_system_manifest(
            ROOT,
            system_id,
            manifest,
            system_schema,
            result,
            deep_weight_hash=False,
        )
    assert result.ok, result.errors
    return task_schema, system_registry, systems, algorithms, weights


def test_repository_registry_passes_without_large_weight_read():
    result = validate_repository(ROOT, deep_weight_hash=False, check_source_hashes=True)
    expected_task_ids = {"cap004", "poker", "chips"}
    registered_task_ids = set(_read("tasks/registry.json")["tasks"])
    assert result.ok, result.errors
    assert result.systems_checked == 3
    assert registered_task_ids == expected_task_ids
    assert result.tasks_checked == len(expected_task_ids)
    assert result.weights_deep_hashed == 0


def test_poker_and_chips_share_mask_weight_but_no_clean_weight():
    poker = _read("tasks/poker/task.manifest.json")
    chips = _read("tasks/chips/task.manifest.json")
    assert poker["system_refs"]["mask"] == chips["system_refs"]["mask"]
    assert poker["system_refs"]["mask"]["weight_ids"] == [
        "sam31.multiplex.0567debe"
    ]
    assert poker["system_refs"]["clean"]["weight_ids"] == []
    assert chips["system_refs"]["clean"]["weight_ids"] == []


def test_task_local_checkpoint_is_rejected():
    task_schema, registry, systems, algorithms, weights = _loaded_state()
    manifest = _read("tasks/poker/task.manifest.json")
    manifest["task_config"]["mask"]["checkpoint_path"] = "/tmp/secret.pt"
    result = ValidationResult()
    validate_task_manifest(
        ROOT,
        "poker",
        manifest,
        task_schema,
        registry,
        systems,
        algorithms,
        weights,
        result,
        check_source_hashes=False,
    )
    assert any("task-local weight/checkpoint key" in item for item in result.errors)


def test_unregistered_weight_id_is_rejected():
    task_schema, registry, systems, algorithms, weights = _loaded_state()
    manifest = _read("tasks/chips/task.manifest.json")
    manifest["system_refs"]["mask"]["weight_ids"] = ["chips.private.weight"]
    result = ValidationResult()
    validate_task_manifest(
        ROOT,
        "chips",
        manifest,
        task_schema,
        registry,
        systems,
        algorithms,
        weights,
        result,
        check_source_hashes=False,
    )
    assert any("must exactly equal algorithm requirements" in item for item in result.errors)


def test_project_finetune_without_training_and_eval_lineage_is_rejected():
    schema = _read("systems/schema/system_manifest.schema.json")
    manifest = copy.deepcopy(_read("systems/mask/system.manifest.json"))
    weight = manifest["learned_weights"][0]
    weight["origin"] = "project_finetuned"
    weight["base_weight_id"] = "sam31.multiplex.0567debe"
    weight.pop("training", None)
    result = ValidationResult()
    validate_system_manifest(
        ROOT,
        "mask",
        manifest,
        schema,
        result,
        deep_weight_hash=False,
    )
    assert any("training" in item for item in result.errors)


def test_input_classes_have_exact_task_specific_safety_policy():
    expected_phone_ids = {
        "phone_action_reference",
        "phone_object_reference",
        "shared_phone_empty_static",
        "shared_phone_empty_sweep",
    }
    for task_id in ("poker", "chips"):
        manifest = _read(f"tasks/{task_id}/task.manifest.json")
        phone_sources = [
            source
            for source in manifest["input_views"]
            if source["source_class"] == "PHONE_DEV_REFERENCE"
        ]
        formal_sources = [
            source
            for source in manifest["input_views"]
            if source["source_class"] == "FORMAL_CAPTURE"
        ]
        assert {source["source_class"] for source in manifest["input_views"]} == {
            "PHONE_DEV_REFERENCE",
            "FORMAL_CAPTURE",
        }
        assert {source["input_id"] for source in phone_sources} == expected_phone_ids
        for source in phone_sources:
            assert any("formal" in item.lower() for item in source["forbidden_uses"])
        assert [source["input_id"] for source in formal_sources] == [
            "formal_pico_batch_0901_canonical"
        ]
        assert formal_sources[0]["path"] == f"tasks/{task_id}/inputs/raw_source_manifest.json"
        assert "bypass_per_session_source_gate" in formal_sources[0]["forbidden_uses"]
