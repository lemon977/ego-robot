from __future__ import annotations

import inspect

from chaoyang.ops import run_0915_input_prepare_cad_v1 as subject


def test_weightless_successor_runs_v2_audit_before_prepare_and_cad() -> None:
    source = inspect.getsource(subject.main)
    assert source.index("audit_0915_processed_self_containment_v2") < source.index(
        "prepare_0915_physical_left_batch_v1"
    ) < source.index("audit_kaihand_adapter_step_v1")
    assert '"weights": "ABSENT"' in source


def test_successor_has_no_model_or_pico_hand_execution() -> None:
    source = inspect.getsource(subject)
    assert "torch" not in source
    assert "SAM3" not in source
    assert "HaWoR" not in source
    assert '"pico26": "PRESENT_PRESERVED_NOT_CONSUMED"' in source
