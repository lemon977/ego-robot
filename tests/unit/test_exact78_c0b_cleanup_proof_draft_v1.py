"""CPU-only C0b inventory and fail-closed reference-scope tests."""

from pathlib import Path

from chaoyang.governance import exact78_c0b_cleanup_proof_draft_v1 as c0b


def test_exact_inventory_excludes_protected_parent_and_current_candidates() -> None:
    rows = c0b.inventory()
    assert len(rows) == 29
    paths = {row["path"] for row in rows}
    assert len(paths) == 29
    assert not any("0916" in path or "robot100" in path for path in paths)
    assert not any(path.startswith("review/") for path in paths)
    assert not any("run_exact78_hawor_raw_v3" in path for path in paths)
    assert {row["kind"] for row in rows} == {
        "old_diagnostic_media", "generated_cache", "legacy_algorithm_code_retention_probe"}


def test_reference_universe_reports_omitted_roots(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "pointer.md").write_text("candidate.png", encoding="utf-8")
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "opaque.bin").write_bytes(b"candidate.png")
    monkeypatch.setattr(c0b, "ROOT", tmp_path)
    monkeypatch.setattr(c0b, "CONTENT_ROOTS", ("docs",))
    monkeypatch.setattr(c0b, "INTENTIONALLY_OMITTED", ("assets",))
    universe = c0b.reference_universe(max_entries=20, max_seconds=2)
    assert universe["complete"] is False
    assert "INTENTIONALLY_OMITTED:assets" in universe["omissions"]
    assert tmp_path / "docs" / "pointer.md" in universe["files"]


def test_reference_hit_uses_relative_name_and_sha(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(c0b, "ROOT", tmp_path)
    monkeypatch.setattr(c0b, "MANIFEST", tmp_path / "C0B_CANDIDATES_V1.json")
    target = tmp_path / "_run/maintenance/old.png"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"image")
    pointer = tmp_path / "docs/current/INDEX.md"
    pointer.parent.mkdir(parents=True)
    pointer.write_text("_run/maintenance/old.png", encoding="utf-8")
    candidate = {"path": "_run/maintenance/old.png", "kind": "old_diagnostic_media",
                 "file": c0b.ref(target)}
    universe = {"files": [target, pointer], "complete": True}
    result = c0b.scan_references([candidate], universe)
    assert result["hits"][candidate["path"]] == ["docs/current/INDEX.md"]
    assert result["complete"] is True


def test_text_budget_gap_blocks_complete_proof(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(c0b, "ROOT", tmp_path)
    monkeypatch.setattr(c0b, "MANIFEST", tmp_path / "C0B_CANDIDATES_V1.json")
    monkeypatch.setattr(c0b, "MAX_SINGLE_TEXT_BYTES", 4)
    pointer = tmp_path / "docs/current/INDEX.md"
    pointer.parent.mkdir(parents=True)
    pointer.write_text("too long", encoding="utf-8")
    result = c0b.scan_references([], {"files": [pointer], "complete": True})
    assert result["complete"] is False
    assert result["errors"][0].startswith("TEXT_SCAN_BUDGET_EXCEEDED:")
