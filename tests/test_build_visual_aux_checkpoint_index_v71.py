import json
from pathlib import Path

import jsonschema


ROOT = Path(__file__).resolve().parents[1]


def test_published_index_closes_four_branches_without_fake_checkpoints() -> None:
    value = json.loads((ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/visual_aux/VISUAL_AUX_CHECKPOINT_INDEX.json").read_text())
    schema = json.loads((ROOT / "contracts/visual_aux_checkpoint_index_v71.schema.json").read_text())
    jsonschema.validate(value, schema)
    assert len(value["rows"]) == 4
    assert {(r["task"], r["visual_input"]) for r in value["rows"]} == {
        ("chips", "HUMAN_RAW_RGB"), ("chips", "ROBOTIZED_RGB"),
        ("poker", "HUMAN_RAW_RGB"), ("poker", "ROBOTIZED_RGB"),
    }
    assert all(r["status"] == "BLOCKED_DATA_VOLUME" for r in value["rows"])
    assert all(r["checkpoint"] is None and r["loss_curve"] is None for r in value["rows"])
    assert value["counts"]["published_checkpoints"] == 0
    assert value["split_movement_allowed"] is False
