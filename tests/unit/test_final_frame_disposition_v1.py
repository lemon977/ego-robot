import pytest

from chaoyang.pipeline.final_frame_disposition_v1 import (
    FrameDisplay,
    MotionSource,
    SideDisplay,
    SideEvidence,
    decide_frame,
    decide_timeline,
)


SHA = "a" * 64


def row(side, **overrides):
    value = dict(
        frame_id=47,
        side=side,
        replacement_required=True,
        motion_source=MotionSource.MODEL_INFERENCE,
        source_valid=True,
        quality_pass=True,
        renderer_valid=True,
        source_artifact_sha256=SHA,
    )
    value.update(overrides)
    return SideEvidence(**value)


def test_qualified_model_inference_is_admitted_for_offline_robot():
    result = decide_frame([row("left"), row("right")], compositor_inputs_valid=True)
    assert result.display is FrameDisplay.ROBOT
    assert result.robot_replaced_sides == ("left", "right")
    assert result.training_authorized is False


@pytest.mark.parametrize(
    "source",
    [MotionSource.TIME_INTERPOLATION, MotionSource.HOLD, MotionSource.COPIED_OTHER_SIDE],
)
def test_interpolation_hold_and_side_copy_preserve_original(source):
    result = decide_frame(
        [row("left", motion_source=source), row("right", replacement_required=False)],
        compositor_inputs_valid=True,
    )
    assert result.display is FrameDisplay.ORIGINAL_FRAME_FALLBACK
    assert result.sides[0].display is SideDisplay.ORIGINAL_PIXELS


def test_bad_inference_partially_preserves_raw_pixels():
    result = decide_frame(
        [row("left"), row("right", quality_pass=False, failure_reason="ROOT_JUMP_1P285650M")],
        compositor_inputs_valid=True,
    )
    assert result.display is FrameDisplay.PARTIAL_PRESERVE
    assert result.robot_replaced_sides == ("left",)
    assert result.original_preserved_sides == ("right",)


def test_invalid_compositor_forces_full_original_frame_fallback():
    result = decide_frame([row("left"), row("right")], compositor_inputs_valid=False)
    assert result.display is FrameDisplay.ORIGINAL_FRAME_FALLBACK
    assert result.original_preserved_sides == ("left", "right")


def test_missing_sha_fails_closed_without_dropping_frame():
    result = decide_frame(
        [row("left", source_artifact_sha256=None), row("right", replacement_required=False)],
        compositor_inputs_valid=True,
    )
    assert result.display is FrameDisplay.ORIGINAL_FRAME_FALLBACK
    assert result.sides[0].reason == "SOURCE_ARTIFACT_SHA256_REQUIRED"


def test_source_sha_must_be_lowercase_hex():
    result = decide_frame(
        [row("left", source_artifact_sha256="G" * 64), row("right", replacement_required=False)],
        compositor_inputs_valid=True,
    )
    assert result.sides[0].display is SideDisplay.ORIGINAL_PIXELS
    assert result.sides[0].reason == "SOURCE_ARTIFACT_SHA256_REQUIRED"


def test_timeline_keeps_every_frame_and_rejects_reordering():
    frames = []
    for frame_id in (46, 47, 48):
        frames.append([row("left", frame_id=frame_id), row("right", frame_id=frame_id)])
    assert [item.frame_id for item in decide_timeline(frames, compositor_inputs_valid=[True] * 3)] == [46, 47, 48]
    with pytest.raises(ValueError, match="STRICTLY_INCREASING"):
        decide_timeline(frames[::-1], compositor_inputs_valid=[True] * 3)


def test_timeline_rejects_empty_input():
    with pytest.raises(ValueError, match="EMPTY_TIMELINE"):
        decide_timeline([], compositor_inputs_valid=[])


def test_side_schema_is_exact():
    with pytest.raises(ValueError, match="EXACTLY_ONE"):
        decide_frame([row("left"), row("left")], compositor_inputs_valid=True)
