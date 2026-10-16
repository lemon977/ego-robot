
from pathlib import Path
import numpy as np
import pytest
from chaoyang.pick_place_1008.mask_probe import _save_point_diagnostics

def fixture_masks():
    bitmap = np.zeros((3, 5), bool)
    bitmap[0, 0] = bitmap[2, 4] = True
    return {"point_object_tracks": {5: {1: np.packbits(bitmap.ravel())}},
            "metadata": {"point_tracking": {"objects": {
                "5": {"role": "controller", "side": "right"},
                "9": {"role": "glove", "side": "right"}}}}}

def test_raw_masks_have_explicit_id_and_native_frame_binding(tmp_path):
    masks = fixture_masks()
    result = _save_point_diagnostics(tmp_path, masks, (3, 3, 5))
    masks["point_object_tracks"][5][1][:] = 0
    with np.load(result["arrays"]["path"], allow_pickle=False) as arrays:
        assert arrays["object_ids"].tolist() == [5, 9]
        assert arrays["model_nonempty"].tolist() == [[False, True, False], [False, False, False]]
        decoded = np.unpackbits(arrays["packed_masks"][0, 1], count=15).reshape(3, 5)
        assert int(decoded.sum()) == 2 and decoded[0, 0] and decoded[2, 4]
        assert not arrays["packed_masks"][1].any()
    import json
    info = json.loads(Path(result["metadata"]["path"]).read_text())
    assert info["packed_arrays"] == result["arrays"]
    assert info["deletion_authority_added_by_diagnostics"] is False
    assert info["physical_ground_truth"] is False

@pytest.mark.parametrize("invalid", ["unknown_id", "frame", "dtype", "length", "padding"])
def test_corrupt_point_diagnostics_do_not_create_artifacts(tmp_path, invalid):
    masks = fixture_masks()
    if invalid == "unknown_id":
        masks["point_object_tracks"][14] = {}
    elif invalid == "frame":
        masks["point_object_tracks"][5][3] = masks["point_object_tracks"][5].pop(1)
    elif invalid == "dtype":
        masks["point_object_tracks"][5][1] = np.zeros(2, np.float64)
    elif invalid == "length":
        masks["point_object_tracks"][5][1] = np.zeros(3, np.uint8)
    else:
        masks["point_object_tracks"][5][1][-1] |= 1
    with pytest.raises(ValueError):
        _save_point_diagnostics(tmp_path, masks, (3, 3, 5))
    assert list(tmp_path.iterdir()) == []

def test_diagnostics_are_immutable(tmp_path):
    masks = fixture_masks()
    _save_point_diagnostics(tmp_path, masks, (3, 3, 5))
    with pytest.raises(ValueError, match="FRESH_OUTPUT"):
        _save_point_diagnostics(tmp_path, masks, (3, 3, 5))
