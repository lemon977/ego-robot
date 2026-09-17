import json
from pathlib import Path

from chaoyang.ops.run_mask_sam31_temporal_identity_real_canary_r22 import (
    DEFAULT_ANNOTATIONS,
    DEFAULT_BOOTSTRAP,
    DEFAULT_SELECTION,
    bind_frozen_targets,
    evaluate_target,
    seal_terminal,
)


def test_cli_binding_gates_and_terminal_six_pack(tmp_path: Path):
    targets = bind_frozen_targets(DEFAULT_SELECTION, DEFAULT_ANNOTATIONS, DEFAULT_BOOTSTRAP)
    assert [target["session_id"] for target in targets] == [
        "get_potato_chips_0901_010", "get_potato_chips_0901_010", "play_cards_0901_015"
    ]
    target = {"frame_count": 3, "seed_frame": 0}
    rows = [
        {"source_frame": frame, "present": True, "reason": "ACCEPTED_SEED_ID_PRESENT", "unexpected_object_ids": [], "connected_components": 1, "area_pixels": 10}
        for frame in range(3)
    ]
    assert evaluate_target(target, rows, seed_area=10)["pass"]
    output = tmp_path / "attempt_0001"
    output.mkdir()
    seal_terminal(output, "PASSED_DEVELOPMENT", targets, {"mock": {"gates": {"pass": True}}}, {}, None, 0.0)
    required = {"RESULT.json", "ARTIFACT_MANIFEST.json", "METRICS.json", "RUN_RECEIPT.json", "DECISION.md", "NEXT_ACTION.json"}
    assert required == {path.name for path in output.iterdir()}
    assert json.loads((output / "NEXT_ACTION.json").read_text())["regression_authorized"] is True
