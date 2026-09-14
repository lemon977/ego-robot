from __future__ import annotations

from pathlib import Path
import sys

import pytest


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from tools.audit_naturalv2_kaihand_mount_lineage_v1 import AuditError, run  # noqa: E402


def test_mount_audit_separates_visual_candidate_from_physical(tmp_path: Path) -> None:
    result = run(tmp_path / "audit")
    assert result["conclusion"]["visual_training_only_assembly_candidate_closed"] is True
    assert result["conclusion"]["physical_assembly_closed"] is False
    assert result["conclusion"]["physical_eligible_candidates"] == []
    by_id = {row["candidate_id"]: row for row in result["candidates"]}
    assert by_id["NATURALV2_STL_IDENTIFIED"]["usable_for_physical"] == "NO_AS_KAIHAND_ADAPTER"
    assert by_id["FLANGE_RING_145MM_PROXY"]["true_cad_or_proxy"] == "PROCEDURAL_PROXY_NOT_REAL_CAD"
    assert result["central_authority_modified"] is False
    with pytest.raises(AuditError, match="refusing to overwrite"):
        run(tmp_path / "audit")
