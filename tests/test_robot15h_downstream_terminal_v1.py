from __future__ import annotations

import json
from pathlib import Path

import pytest

from chaoyang.ops.run_0915_robot15h_downstream_terminal_v1 import (
    TASKS,
    blocked_rows_from_upstream,
    geometry_eligibility_rows,
    summarize,
)


def _inventory() -> list[dict[str, object]]:
    return [
        {"session_id": f"session_{index}", "task": "playing_cards", "source_group": f"source_{index}", "frame_count": 100 + index}
        for index in range(4)
    ]


def _session_result(tmp_path: Path, session_id: str, admitted: bool) -> dict[str, object]:
    path = tmp_path / f"{session_id}.json"
    path.write_text(json.dumps({"status": "PASSED" if admitted else "REJECTED_QUALITY", "consumption_authorized": admitted}))
    import hashlib
    data = path.read_bytes()
    return {"path": str(path.resolve()), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _depth(tmp_path: Path, bad: str | None = None) -> dict[str, object]:
    return {
        "sessions": [
            {
                "session_id": f"session_{index}",
                "status": "REJECTED_QUALITY" if bad == f"session_{index}" else "PASSED",
                "result": _session_result(tmp_path, f"session_{index}", bad != f"session_{index}"),
            }
            for index in range(4)
        ]
    }


def _masks(admitted: bool = False) -> dict[str, object]:
    return {
        "object_mask_consumer_allowed": admitted,
        "sessions": [
            {
                "session_id": f"session_{index}",
                "object_mask_consumer_allowed": admitted,
                "object_semantic_admitted": 1 if admitted else 0,
            }
            for index in range(4)
        ],
    }


def test_geometry_blocks_all_four_on_missing_object_masks_without_fake_artifacts(tmp_path: Path) -> None:
    rows = geometry_eligibility_rows(_inventory(), _depth(tmp_path), _masks(False))
    assert [row["status"] for row in rows] == ["BLOCKED_UPSTREAM_OBJECT_MASK"] * 4
    assert all(row["object6d_attempted"] is False for row in rows)
    assert all(row["object6d_artifact_emitted"] is False for row in rows)
    assert all(row["review_video_emitted"] is False for row in rows)
    assert summarize(rows) == {
        "total": 4, "attempted": 0, "passed": 0, "BLOCKED_UPSTREAM_OBJECT_MASK": 4,
    }


def test_depth_blocker_has_precedence_over_missing_object_mask(tmp_path: Path) -> None:
    rows = geometry_eligibility_rows(_inventory(), _depth(tmp_path, "session_2"), _masks(False))
    assert rows[2]["status"] == "BLOCKED_UPSTREAM_DEPTH"
    assert rows[2]["first_blocker"] == "DEPTH_NOT_CONSUMER_ADMITTED"
    assert rows[0]["status"] == "BLOCKED_UPSTREAM_OBJECT_MASK"


def test_geometry_only_reports_eligibility_and_still_does_not_fake_object6d(tmp_path: Path) -> None:
    rows = geometry_eligibility_rows(_inventory(), _depth(tmp_path), _masks(True))
    assert all(row["status"] == "ELIGIBLE_FOR_SEPARATE_OBJECT6D_EXECUTION" for row in rows)
    assert all(row["object6d_attempted"] is False and row["object6d_artifact_emitted"] is False for row in rows)


def test_geometry_accepts_task_object_instance_count_schema(tmp_path: Path) -> None:
    masks = {
        "object_mask_consumer_allowed": True,
        "sessions": [
            {
                "session_id": f"session_{index}",
                "object_mask_consumer_allowed": True,
                "instance_counts": {"stable_consumer_allowed": 3},
            }
            for index in range(4)
        ],
    }
    rows = geometry_eligibility_rows(_inventory(), _depth(tmp_path), masks)
    assert all(row["status"] == "ELIGIBLE_FOR_SEPARATE_OBJECT6D_EXECUTION" for row in rows)


@pytest.mark.parametrize(
    ("stage", "prior", "expected"),
    [
        ("interaction", "BLOCKED_UPSTREAM_OBJECT_MASK", "BLOCKED_UPSTREAM_GEOMETRY"),
        ("contact", "BLOCKED_UPSTREAM_GEOMETRY", "BLOCKED_UPSTREAM_INTERACTION"),
        ("r1", "BLOCKED_UPSTREAM_INTERACTION", "BLOCKED_LOCAL_EVIDENCE"),
    ],
)
def test_blockers_propagate_per_session_without_algorithm_artifacts(stage: str, prior: str, expected: str) -> None:
    upstream = {"sessions": [{"session_id": f"session_{index}", "status": prior} for index in range(4)]}
    rows = blocked_rows_from_upstream(_inventory(), upstream, stage)
    assert len(rows) == 4
    assert all(row["status"] == expected for row in rows)
    assert all(row["algorithm_attempted"] is False for row in rows)
    assert all(row["algorithm_artifact_emitted"] is False for row in rows)
    if stage == "r1":
        assert all(row["r1_e_status"] == "BLOCKED_LOCAL_EVIDENCE" for row in rows)
        assert all(row["r1_h_status"] == "BLOCKED_LOCAL_EVIDENCE" for row in rows)
        assert all(row["r0_preserved"] is True for row in rows)
        assert all(row["r2_independent_path_unblocked"] is True for row in rows)


def test_blocker_only_runner_refuses_to_replace_real_algorithm_after_admission() -> None:
    upstream = {"sessions": [{"session_id": f"session_{index}", "status": "PASSED"} for index in range(4)]}
    with pytest.raises(RuntimeError, match="real algorithm runner"):
        blocked_rows_from_upstream(_inventory(), upstream, "interaction")


def test_downstream_accepts_prior_eligibility_ledger_rows_key() -> None:
    upstream = {"rows": [
        {"session_id": f"session_{index}", "status": "BLOCKED_UPSTREAM_OBJECT_MASK"}
        for index in range(4)
    ]}
    rows = blocked_rows_from_upstream(_inventory(), upstream, "interaction")
    assert [row["status"] for row in rows] == ["BLOCKED_UPSTREAM_GEOMETRY"] * 4


def test_all_supported_tasks_are_weights_absent_nodes() -> None:
    assert set(TASKS) == {
        "0915_robot15h_geometry_object6d_wave0_v1",
        "0915_robot15h_interaction_occlusion_v1",
        "0915_robot15h_contact_dual_evidence_v1",
        "0915_robot15h_robot_relative_refinement_v1",
    }
