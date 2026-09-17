from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
FINALIZE = ROOT / "scripts/migration/finalize_archive.py"
PURGE = ROOT / "scripts/migration/purge_archive_prefixes.py"
VERIFY = ROOT / "scripts/migration/verify_archive.py"


def run(*args: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *(str(arg) for arg in args)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_receipted_purge_keeps_inventory_verifiable(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    purged = archive / "content/regenerable/cache/item.bin"
    kept = archive / "content/keep.txt"
    purged.parent.mkdir(parents=True)
    kept.parent.mkdir(parents=True, exist_ok=True)
    purged.write_bytes(b"regenerable")
    kept.write_text("keep\n", encoding="utf-8")
    (archive / "MOVES.tsv").write_text("", encoding="utf-8")
    (archive / "RESTORE.md").write_text("fixture\n", encoding="utf-8")

    run(FINALIZE, "--archive", archive)
    summary = json.loads((archive / "MERKLE_ROOT.json").read_text(encoding="utf-8"))
    run(
        PURGE,
        "--archive", archive,
        "--prefix", "content/regenerable/cache/",
        "--reason", "test fixture",
        "--expected-merkle", summary["merkle_root_sha256"],
        "--execute",
    )
    assert not purged.exists()

    result = json.loads(run(VERIFY, "--archive", archive).stdout)
    assert result["status"] == "PASS"
    assert result["all_inventory_entries_accounted_for"] is True
    assert result["all_paths_and_sizes_verified"] is False
    assert result["receipted_purge"]["entries"] == 1
    assert result["receipted_purge"]["bytes"] == len(b"regenerable")
