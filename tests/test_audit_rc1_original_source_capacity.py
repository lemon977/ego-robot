from __future__ import annotations

import json

from chaoyang.ops.audit_rc1_original_source_capacity import connected_lineage_clusters, lineage_from_jsonl


def test_header_source_list_is_not_clipped_lineage(tmp_path):
    source = tmp_path / "tracking.jsonl"
    source.write_text(
        "\n".join(json.dumps(row) for row in [
            {"_merge": {"sourceSessions": ["A", "B", "C"]}},
            {"_merge": {"sourceSession": "B", "sourceTimeStampNs": 100, "sourceRecordIndex": 4}},
            {"_merge": {"sourceSession": "B", "sourceTimeStampNs": 200, "sourceRecordIndex": 7}},
        ]) + "\n",
        encoding="utf-8",
    )
    result = lineage_from_jsonl(source, "_merge")
    assert result["source_sessions"] == ["B"]
    assert result["rows_without_source"] == 1
    assert result["source_timestamp_ns_range"]["B"] == [100, 200]
    assert result["source_record_index_range"]["B"] == [4, 7]


def test_shared_source_and_cross_source_clip_cannot_cross_split():
    rows = [
        {"session_id": "a", "task": "chips", "claimed_source_sessions": ["A"], "rc1_split": "train"},
        {"session_id": "b", "task": "chips", "claimed_source_sessions": ["A", "B"], "rc1_split": "validation"},
        {"session_id": "c", "task": "chips", "claimed_source_sessions": ["B"], "rc1_split": "not_candidate"},
        {"session_id": "d", "task": "chips", "claimed_source_sessions": ["C"], "rc1_split": "validation"},
    ]
    clusters = connected_lineage_clusters(rows)
    assert len(clusters) == 2
    assert clusters[0]["sessions"] == ["a", "b", "c"]
    assert clusters[0]["split_conflict"] is True
    assert clusters[1]["sessions"] == ["d"]
    assert clusters[1]["split_conflict"] is False
