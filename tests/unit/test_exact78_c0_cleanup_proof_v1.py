"""C0 proof scope must fail closed before any C1 consideration."""

from chaoyang.governance.execute_exact78_c0_cleanup_proof_v1 import (
    REQUIRED_REFERENCE_GLOBS,
    reference_scope_gaps,
)


def test_draft_scope_cannot_claim_complete_project_reverse_refs() -> None:
    narrow = {"file_globs": ["docs/current/**/*.md", "src/chaoyang/**/*exact78*.py"]}
    gaps = reference_scope_gaps(narrow)
    assert "scripts/**/*.py" in gaps
    assert "_run/current/**/*.json" in gaps


def test_declared_full_text_scope_has_no_static_gap() -> None:
    assert reference_scope_gaps({"file_globs": sorted(REQUIRED_REFERENCE_GLOBS)}) == []
