from pathlib import Path

from tools.project_progress_watchdog import build_snapshot, event_fingerprint


ROOT = Path(__file__).resolve().parents[1]


def test_snapshot_is_observation_only_and_stable() -> None:
    snapshot = build_snapshot(ROOT, 60)
    assert snapshot["claim_limit"] == "OBSERVATION_ONLY_NOT_AUTHORIZATION_OR_PRODUCER_STATUS"
    assert snapshot["interval_seconds"] == 60
    assert len(snapshot["authority_docs"]) == 5
    assert len(event_fingerprint(snapshot)) == 64


def test_authority_documents_are_observed() -> None:
    snapshot = build_snapshot(ROOT, 60)
    records = {item["path"]: item for item in snapshot["authority_docs"]}
    assert set(records) == {"README.md", "docs/governance/CURRENT_TASK.md", "docs/governance/TECHNICAL.md", "docs/data/DATA_SPEC.md", "docs/data/DATA_CATALOG.json"}
    assert all(item["available"] for item in records.values())
