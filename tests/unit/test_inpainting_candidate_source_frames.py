from pathlib import Path

import numpy as np
import pytest

from chaoyang.inpainting_candidate.source_frames import bind_decoded_frames, write_review_video


def report(n=3):
    return {"camera": {"physical_eye": "right", "sbs_index": 0},
            "native_sidecar_frames": n, "timebase_status": "TIMEBASE_UNVERIFIED",
            "frame_mapping": [{"native_frame_index": i, "sbs_index": 0,
                               "qpc_declared_ns": 1_000_000_000 + i * 40_000_000}
                              for i in range(n)]}


def frames(n=3):
    return iter([(np.full((480, 640, 3), i, dtype=np.uint8), {"decoder_pts": None}) for i in range(n)])


def test_native_frame_binding_never_qualifies_unknown_capture_id():
    rgb, mapping = bind_decoded_frames(report(), frames())
    assert len(rgb) == len(mapping) == 3
    assert [r["model_input_frame"] for r in mapping] == [0, 1, 2]
    assert all("UNKNOWN" in r["mapping_qualification"] for r in mapping)


@pytest.mark.parametrize("decoded", [2, 4])
def test_no_silent_duplication_or_dropping(decoded):
    with pytest.raises(ValueError, match="COUNT|MORE_DECODED"):
        bind_decoded_frames(report(), frames(decoded))


def test_physical_eye_not_guessed():
    source = report()
    source["camera"]["physical_eye"] = None
    with pytest.raises(ValueError, match="PHYSICAL_EYE"):
        bind_decoded_frames(source, frames())


def test_mapping_identity_not_just_count():
    source = report()
    source["frame_mapping"][1]["native_frame_index"] = 0
    with pytest.raises(ValueError, match="IDENTITY"):
        bind_decoded_frames(source, frames())


def test_review_preserves_native_25fps_intervals(tmp_path):
    from chaoyang.inpainting_candidate.resources import PROJECT_ROOT
    assert tmp_path.is_relative_to(PROJECT_ROOT)
    rgb, mapping = bind_decoded_frames(report(), frames())
    receipt = write_review_video(tmp_path / "review.mp4", rgb, mapping)
    assert receipt["decoded_frames"] == 3
    assert receipt["pts_roundtrip_max_error_seconds"] <= 1e-6
    assert receipt["visual_quality"] == "NOT_REVIEWED"
    assert receipt["causal_training_input"] is False


def test_video_write_outside_project_rejected():
    rgb, mapping = bind_decoded_frames(report(), frames())
    with pytest.raises(ValueError, match="PROJECT_OUTPUT"):
        write_review_video(Path("/tmp/do-not-create-inpainting.mp4"), rgb, mapping)
