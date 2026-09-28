from __future__ import annotations
import numpy as np
import pytest
from chaoyang.ops.run_human_to_robot_shared_delivery_clean import _apply_lama_residual

def inputs():
    image = np.zeros((720, 960, 3), dtype=np.uint8)
    image[:] = [17, 83, 211]
    mask = np.zeros((720, 960), dtype=np.uint8)
    mask[100:200, 300:400] = 255
    return image, mask

def test_rgb_output_pixel_domain_and_bgr_input():
    class Session:
        def run(self, outputs, data):
            assert np.allclose(data["image"][0,:,0,0], np.array([211,83,17])/255)
            assert data["mask"].shape == (1,1,512,512)
            assert set(np.unique(data["mask"])) == {0,1}
            result = np.zeros((1,3,512,512), dtype=np.float32)
            result[:,0] = 31
            result[:,1] = 107
            result[:,2] = 211
            return [result]
    image, mask = inputs()
    output = _apply_lama_residual(Session(), "output", image, mask)
    assert np.array_equal(output[mask == 0], image[mask == 0])
    assert np.all(output[mask != 0] == [211,107,31])
    assert output.dtype == np.uint8
    assert np.all(image == [17,83,211])

def test_empty_mask_returns_copy_without_call():
    class Session:
        def run(self, *args):
            raise AssertionError("empty mask must not call model")
    image, mask = inputs()
    mask[:] = 0
    output = _apply_lama_residual(Session(), "output", image, mask)
    assert np.array_equal(output, image)
    assert not np.shares_memory(output, image)

@pytest.mark.parametrize("kind", ["nan", "shape"])
def test_invalid_prediction_is_rejected(kind):
    class Session:
        def run(self, *args):
            return [np.full((1,3,512,512) if kind == "nan" else (1,1,512,512),
                            np.nan if kind == "nan" else 10, dtype=np.float32)]
    image, mask = inputs()
    with pytest.raises(RuntimeError, match="LAMA_OUTPUT_INVALID"):
        _apply_lama_residual(Session(), "output", image, mask)

def test_wrong_input_domain_rejected():
    image, mask = inputs()
    with pytest.raises(ValueError, match="LAMA_RESIDUAL_DOMAIN"):
        _apply_lama_residual(None, "output", image[:100], mask)

def test_finite_rgb_values_not_multiplied_or_normalized():
    class Session:
        def run(self, *args):
            return [np.full((1,3,512,512), 211, dtype=np.float32)]
    image, mask = inputs()
    result = _apply_lama_residual(Session(), "output", image, mask)
    assert np.all(result[mask != 0] == 211)

def test_frame_probe_does_not_leak_previous_inference_into_empty_mask():
    from chaoyang.ops.run_four_stream_completion_clean import AuditedSession, apply_with_probe
    class Session:
        def run(self, *args):
            return [np.full((1,3,512,512), 211, dtype=np.float32)]
    image, mask = inputs()
    session = AuditedSession(Session())
    _, first = apply_with_probe(session, "output", image, mask)
    assert first["model_called"] is True
    assert first["probe"]["output_p50"] == 211
    mask[:] = 0
    _, second = apply_with_probe(session, "output", image, mask)
    assert second == {"model_called": False, "probe": None}
    assert len(session.rows) == 1
