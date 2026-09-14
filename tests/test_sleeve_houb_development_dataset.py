from pathlib import Path

import numpy as np
import pytest

from pipeline.sam21_route_b_contract import DEVELOPMENT_FRAMES
from pipeline.sleeve_houb_development_dataset import (
    FREEZE_ROOT,
    SleeveHOUBDatasetError,
    SleeveHOUBDevelopmentDataset,
    dataset_identity,
)


def test_exact_development_index_without_pixel_reads():
    dataset = SleeveHOUBDevelopmentDataset()
    assert dataset.frames == DEVELOPMENT_FRAMES
    assert len(dataset) == 15
    assert dataset.manifest_only_blind_reference_count == 30
    assert dataset.pixel_access_log == []


@pytest.mark.parametrize("frame", [70, 144, 442, 212, 247, 83, 390])
def test_blind_or_unknown_frame_fails_before_pixel_open(frame):
    dataset = SleeveHOUBDevelopmentDataset()
    with pytest.raises(SleeveHOUBDatasetError, match="sealed or unknown"):
        dataset.load(frame)
    assert dataset.pixel_access_log == []


def test_one_real_development_sample_round_trips_palette_and_binary_masks():
    dataset = SleeveHOUBDevelopmentDataset()
    sample = dataset.load(228)
    assert sample.image_rgb.shape == (960, 1280, 3)
    assert sample.palette.shape == (960, 1280)
    assert set(np.unique(sample.palette).tolist()) <= {0, 1, 2, 3}
    membership = sum(
        value.astype(np.uint8)
        for value in (sample.human, sample.object_, sample.uncertain, sample.background)
    )
    assert np.all(membership == 1)
    assert np.array_equal(sample.supervision.human_positive, sample.human)
    assert np.array_equal(
        sample.supervision.object_or_background_negative,
        sample.object_ | sample.background,
    )
    assert np.array_equal(
        sample.supervision.uncertain_false_negative_only, sample.uncertain
    )
    assert all(item.startswith("grap_a_cap_004:00228:") for item in dataset.pixel_access_log)


def test_identity_states_merged_h_subclasses_and_digest_binding():
    identity = dataset_identity()
    assert identity["subclass_labels_available"] is False
    assert "sleeve" in identity["h_semantics"]
    assert "human-worn item" in identity["h_semantics"]
    assert Path(identity["freeze_root"]) == FREEZE_ROOT
    assert len(identity["manifest_sha256"]) == 3
