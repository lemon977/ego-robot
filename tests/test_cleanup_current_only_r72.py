from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_r72_cleanup_is_dry_run_only() -> None:
    source = (ROOT / "src/chaoyang/ops/cleanup_current_only_r72.py").read_text(encoding="utf-8")
    assert "DRY_RUN_ONLY_NO_OLD_ASSET_DELETE_AUTHORIZED" in source
    assert "delete_executed\": False" in source
    assert "shutil.rmtree" not in source
    assert ".unlink(" not in source


def test_r72_external_roots_are_explicitly_protected() -> None:
    source = (ROOT / "src/chaoyang/ops/cleanup_current_only_r72.py").read_text(encoding="utf-8")
    assert "/mnt/data/egodata" in source
    assert "/nas/chenxianchi" in source
