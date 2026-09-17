import json

from chaoyang.ops.run_role_mask_runtime_contract_repair_v71 import corrected_config, validated


def test_corrected_config_changes_only_verified_fps(tmp_path) -> None:
    path = corrected_config(tmp_path)
    value = json.loads(path.read_text())
    assert value["session_id"] == "play_cards_0901_042"
    assert value["frame_count"] == 196
    assert value["fps"] == 25.0


def test_validator_is_not_hardcoded_to_25fps() -> None:
    import inspect

    source = inspect.getsource(validated)
    assert 'float(config["fps"])' in source
