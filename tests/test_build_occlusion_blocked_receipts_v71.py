from pathlib import Path


def test_blocked_receipt_builder_never_reports_accuracy() -> None:
    text = (Path(__file__).resolve().parents[1] / "src/chaoyang/ops/build_occlusion_blocked_receipts_v71.py").read_text()
    assert '"accuracy_reported": False' in text
    assert "INDEPENDENT_240_FRAME_DOUBLE_REVIEWED_GOLDSET_ABSENT" in text
    assert "NO_CAUSAL_ROBOTIZED_CANDIDATE_WITH_PIXEL_PROVENANCE" in text
    assert "Silver evidence cannot substitute" in text
