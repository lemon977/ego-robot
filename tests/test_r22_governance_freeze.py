from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_r22_registration_copies_mutable_inputs_into_content_addressed_bundle() -> None:
    source = (ROOT / "src/chaoyang/governance/register_r22_integration.py").read_text(encoding="utf-8")
    assert "def _freeze_file" in source
    assert 'ATTEMPT_ROOT / "input_snapshot"' in source
    assert '"source": reference, "frozen": frozen' in source


def test_r22_repair_never_silently_substitutes_latest_for_missing_old_bytes() -> None:
    source = (ROOT / "src/chaoyang/governance/repair_r22_start_snapshot.py").read_text(encoding="utf-8")
    assert "G0_MUTABLE_REFERENCE_BYTES_NOT_FROZEN" in source
    assert "no worker may silently substitute" in source
