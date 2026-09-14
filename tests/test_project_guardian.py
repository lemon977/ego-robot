from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "project_guardian", ROOT / "tools" / "project_guardian.py"
)
assert SPEC is not None and SPEC.loader is not None
GUARDIAN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GUARDIAN)


def test_numbered_candidates_are_sorted_and_ignore_symlink(tmp_path, monkeypatch):
    monkeypatch.setattr(GUARDIAN, "PROJECT_ROOT", tmp_path)
    run_root = tmp_path / "_run"
    for version in (10, 2, 1):
        candidate = run_root / f"g2_004_mask_clean_candidate_v{version}"
        candidate.mkdir(parents=True)
        (candidate / "run_manifest.json").write_text(
            '{"status":"CANDIDATE"}\n', encoding="utf-8"
        )
    (run_root / "g2_004_mask_clean_candidate_v0").mkdir()
    (run_root / "g2_004_mask_clean_candidate_latest").symlink_to(
        run_root / "g2_004_mask_clean_candidate_v10", target_is_directory=True
    )

    candidates = GUARDIAN.numbered_g2_candidates()

    assert [item[0] for item in candidates] == [1, 2, 10]
    assert [item[1].name for item in candidates] == [
        "g2_004_mask_clean_candidate_v1",
        "g2_004_mask_clean_candidate_v2",
        "g2_004_mask_clean_candidate_v10",
    ]


def test_snapshot_reports_raw_verdict_and_latest_candidate(tmp_path, monkeypatch):
    monkeypatch.setattr(GUARDIAN, "PROJECT_ROOT", tmp_path)
    raw = tmp_path / "archive/legacy_runs/unclassified/g0_20260826_parallel_v1/raw"
    raw.mkdir(parents=True)
    (raw / "RAW_VALIDATION.json").write_text(
        '{"verdict":"PASS_G0_SOURCE_READABLE"}\n', encoding="utf-8"
    )
    qa = tmp_path / "archive/legacy_runs/unclassified/g0_20260826_parallel_v1/qa"
    qa.mkdir()
    (qa / "G1_CONTRACT_VALIDATION.json").write_text(
        '{"overall_verdict":"PASS_G1"}\n', encoding="utf-8"
    )
    for version in (1, 3):
        candidate = tmp_path / f"_run/g2_004_mask_clean_candidate_v{version}"
        candidate.mkdir()
        (candidate / "run_manifest.json").write_text(
            f'{{"status":"STATUS_V{version}"}}\n', encoding="utf-8"
        )
    monkeypatch.setattr(GUARDIAN, "command", lambda _args: (0, ""))

    state = GUARDIAN.snapshot()

    assert state["gates"] == {
        "g0": "PASS_G0_SOURCE_READABLE",
        "g1": "PASS_G1",
        "g2": "STATUS_V3",
        "g2_candidate_count": 2,
        "g2_latest_run_id": "g2_004_mask_clean_candidate_v3",
        "mask_clean_review": None,
    }
