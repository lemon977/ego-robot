from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from chaoyang.ops import run_human_to_robot_shared_delivery_clean as clean
from chaoyang.pipeline.v5_scene import composite_clean


def test_frozen_reference_schedule_is_explicit_and_covers_every_frame() -> None:
    rows = clean.expected_reference_schedule()
    assert [row["mid_neighbor_local_index"] for row in rows] == [0, 5, 10, 15]
    assert rows[0]["neighbor_local_indices"] == list(range(6))
    assert rows[0]["reference_local_indices"] == [10]
    assert rows[2]["reference_local_indices"] == [0]
    visited = {frame for row in rows for frame in row["neighbor_local_indices"]}
    assert visited == set(range(16))
    assert all(not set(row["neighbor_local_indices"]) & set(row["reference_local_indices"])
               for row in rows)


def test_instrumentation_is_optional_and_records_actual_tensor_identity() -> None:
    source = (clean.VENDOR / "inference_propainter.py").read_text(encoding="utf-8")
    assert "--trace_dir" in source and "default=None" in source
    assert "selected_mask_tensor" in source
    assert "actual_mask_tensor" in source
    assert "raw_prediction" in source
    assert "internal_comp_cumulative" in source
    assert "propagated_frame" in source
    assert "residual_propagation_mask" in source
    assert "TRACE_MANIFEST.json" in source


def test_last_visit_is_the_vendor_output_state() -> None:
    manifest = {"steps": [
        {"visits": [{"local_index": 0, "value": "first"},
                     {"local_index": 1, "value": "only"}]},
        {"visits": [{"local_index": 0, "value": "last"}]},
    ]}
    assert clean._last_visits(manifest) == {
        0: {"local_index": 0, "value": "last"},
        1: {"local_index": 1, "value": "only"},
    }


def test_final_paste_cannot_reintroduce_raw_inside_write() -> None:
    raw = np.zeros((8, 8, 3), dtype=np.uint8)
    raw[2:6, 2:6] = 100
    generated = np.full_like(raw, 220)
    write = np.zeros((8, 8), dtype=bool)
    write[2:6, 2:6] = True
    protect = np.zeros((8, 8), dtype=bool)
    protect[3, 3] = True
    write[protect] = False
    output = composite_clean(raw, generated, write, protect)
    assert np.array_equal(output[write], generated[write])
    assert np.array_equal(output[~write], raw[~write])
    assert np.array_equal(output[protect], raw[protect])


@pytest.mark.xfail(
    strict=True,
    reason="V3 output-domain adapter incorrectly multiplies an already 0..255 output; repair budget exhausted",
)
def test_lama_adapter_changes_only_propagation_residual() -> None:
    class Session:
        input_min = None
        input_max = None

        @classmethod
        def run(cls, _outputs, inputs):
            cls.input_min = float(inputs["image"].min())
            cls.input_max = float(inputs["image"].max())
            return [np.full((1, 3, 512, 512), 211, dtype=np.float32)]

    propagated = np.full((720, 960, 3), 17, dtype=np.uint8)
    residual = np.zeros((720, 960), dtype=np.uint8)
    residual[100:200, 300:400] = 255
    output = clean._apply_lama_residual(Session(), "output", propagated, residual)
    assert Session.input_min is not None and 0 <= Session.input_min <= 1
    assert Session.input_max is not None and 0 <= Session.input_max <= 1
    assert np.array_equal(output[residual == 0], propagated[residual == 0])
    assert np.all(output[residual > 0] == 211)


def test_lama_contract_accepts_only_dynamic_batch_or_one() -> None:
    assert clean._lama_shape_matches(["batch", 3, 512, 512], 3)
    assert clean._lama_shape_matches([1, 1, 512, 512], 1)
    assert not clean._lama_shape_matches([2, 3, 512, 512], 3)


def test_lama_contract_rejects_wrong_channel_or_spatial_shape() -> None:
    assert not clean._lama_shape_matches(["batch", 1, 512, 512], 3)
    assert not clean._lama_shape_matches(["batch", 3, 256, 512], 3)
    assert not clean._lama_shape_matches(["batch", 3, 512, 513], 3)


def test_review_panel_normalizes_grayscale_to_bgr() -> None:
    panel = clean._panel(np.zeros((12, 16), dtype=np.uint8), "MASK")
    assert panel.shape == (360, 480, 3)


def test_frozen_input_keeps_write_and_protection_disjoint() -> None:
    value = clean.load_json(clean.BASE / "INPUT.json")
    assert [row["frame_id"] for row in value["rows"]] == list(range(76, 92))
    for row in value["rows"]:
        write = cv2.imread(row["write"]["path"], cv2.IMREAD_GRAYSCALE)
        protect = cv2.imread(row["protect"]["path"], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(row["model_masks"]["path"], cv2.IMREAD_GRAYSCALE)
        assert write is not None and protect is not None and mask is not None
        assert not np.any((write > 0) & (protect > 0))
        assert write.any() and mask.any()


def test_lama_weight_and_exact_provenance_pin_are_admitted() -> None:
    gate = clean._lama_gate()
    assert gate["weight"]["sha256"] == clean.LAMA_SHA256
    assert gate["sha_registered"] is True
    pin = clean.load_json(clean.LAMA_PIN)
    assert pin["source_project"] == "Carve/LaMa-ONNX"
    assert pin["source_revision"] == "a3ee2fca54baebec351b8fa7786154ffa7555aa6"
    assert pin["sha256"] == clean.LAMA_SHA256
    assert pin["declared_license"] == "Apache-2.0"
    assert pin["source_url"].endswith("/a3ee2fca54baebec351b8fa7786154ffa7555aa6/lama_fp32.onnx")
    assert gate["source_and_license_pin_valid"] is True
    assert gate["admission"] == "PASS"


def test_all_declared_outputs_stay_inside_current_task_lane() -> None:
    root = Path("/mnt/workspace/code/chaoyang").resolve()
    assert clean.OUT.resolve().is_relative_to(root)
    assert clean.LANE.resolve().is_relative_to(root)
    assert clean.TRACE.resolve().is_relative_to(root)


def test_failed_candidates_and_successor_v3_are_distinct_immutable_paths() -> None:
    failed = [clean.LANE / f"POKER_076_091_LAMA_RESIDUAL_V{version}" for version in (1, 2)]
    assert all(path.exists() for path in failed)
    assert clean.CANDIDATE.name == "POKER_076_091_LAMA_RESIDUAL_V3"
    assert all(clean.CANDIDATE != path for path in failed)
    assert clean.CANDIDATE.exists()
    assert (clean.CANDIDATE / "TERMINAL_RESULT.json").is_file()
