from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from chaoyang.governance import finalize_ai1_artifact_ref_rebind_v31 as finalize
from chaoyang.governance import register_ai1_artifact_ref_rebind_v31 as register
from chaoyang.governance.common import artifact_ref, atomic_json, load_json
from chaoyang.ops import run_ai1_artifact_ref_rebind_v31 as run


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, value)


def _stale(reference: dict[str, Any], *, root: Path, relative: Path) -> dict[str, Any]:
    value = copy.deepcopy(reference)
    value["path"] = str(
        root / "_run/current/ai1_cpfs_publish_fix_v31/attempts/"
        ".attempt_0001.staging-deadbeef/AI1_CURRENT_ONLY_LANE" / relative
    )
    return value


def _setup_source_attempt(root: Path) -> tuple[Path, dict[str, bytes]]:
    attempt = root / run.SOURCE_ATTEMPT_RELATIVE
    lane = attempt / run.LANE_RELATIVE
    input_npz = lane / "input_bundle/WRIST_DUAL_INPUT.npz"
    config = lane / "PRODUCER_CONFIG.json"
    metrics = lane / "dual_representation/METRICS.json"
    representation = lane / "dual_representation/WRIST_DUAL_REPRESENTATION_V1.npz"
    m0 = lane / "M0_BASELINE.npz"
    for path, payload in (
        (input_npz, b"input-npz"),
        (config, b'{"config":true}\n'),
        (metrics, b'{"metric":true}\n'),
        (representation, b"representation-npz"),
        (m0, b"m0-npz"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    provenance = lane / "input_bundle/PROVENANCE.json"
    _write(
        provenance,
        {
            "schema_version": "fixture-provenance",
            "outputs": {
                "npz": _stale(
                    artifact_ref(input_npz), root=root, relative=input_npz.relative_to(lane)
                )
            },
        },
    )
    producer = lane / "dual_representation/RESULT.json"
    _write(
        producer,
        {
            "schema_version": "fixture-producer",
            "inputs": {
                "bundle": _stale(
                    artifact_ref(input_npz), root=root, relative=input_npz.relative_to(lane)
                ),
                "config": _stale(
                    artifact_ref(config), root=root, relative=config.relative_to(lane)
                ),
            },
            "outputs": {
                "metrics": _stale(
                    artifact_ref(metrics), root=root, relative=metrics.relative_to(lane)
                ),
                "npz": _stale(
                    artifact_ref(representation),
                    root=root,
                    relative=representation.relative_to(lane),
                ),
            },
        },
    )
    ledger = lane / "LANE_LEDGER.json"
    ledger_refs = {
        "input_provenance": _stale(
            artifact_ref(provenance), root=root, relative=provenance.relative_to(lane)
        ),
        "m0_baseline": _stale(artifact_ref(m0), root=root, relative=m0.relative_to(lane)),
        "producer_result": _stale(
            artifact_ref(producer), root=root, relative=producer.relative_to(lane)
        ),
    }
    _write(
        ledger,
        {
            "schema_version": "chaoyang-wiyh-ai1-lane-ledger-v31",
            "status": "BLOCKED_ADOPTION_OBSERVATIONS",
            "lane": "ai1",
            "outputs": ledger_refs,
        },
    )
    lane_result = lane / "RESULT.json"
    latest = [
        *ledger_refs.values(),
        _stale(artifact_ref(ledger), root=root, relative=ledger.relative_to(lane)),
    ]
    _write(
        lane_result,
        {
            "schema_version": "chaoyang-wiyh-ai1-lane-result-v31",
            "status": "BLOCKED_ADOPTION_OBSERVATIONS",
            "lane": "ai1",
            "latest_artifacts": latest,
            "ledger": _stale(artifact_ref(ledger), root=root, relative=ledger.relative_to(lane)),
        },
    )
    _write(
        attempt / "RESULT.json",
        {
            "schema_version": "AI1_CPFS_PUBLISH_FIX_V31_RESULT",
            "task_id": "ai1_cpfs_publish_fix_v31",
            "status": "BLOCKED_PREREQ",
            "lane_result": artifact_ref(lane_result),
        },
    )
    _write(
        attempt / "RUN_RECEIPT.json",
        {
            "schema_version": "ai1-cpfs-publish-fix-v31-run-receipt-v1",
            "task_id": "ai1_cpfs_publish_fix_v31",
            "status": "BLOCKED_PREREQ",
            "finalization_status": "PASSED_CAS_PUBLISHED",
        },
    )
    snapshot = {
        str(path.relative_to(attempt)): path.read_bytes()
        for path in attempt.rglob("*")
        if path.is_file()
    }
    return attempt, snapshot


def _setup_governance(root: Path) -> None:
    for relative in register.CODE_PATHS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture {relative}\n", encoding="utf-8")
    paths = register._paths(root)
    _write(paths["authority"], {"schema_version": "test-authority"})
    _write(paths["receipt"], {"governance_revision": 30, "files": {}})
    _write(paths["task_state"], {"tasks": [], "next_task": None, "recent_events": []})
    _write(
        paths["index"],
        {
            "schema_version": "chaoyang-v71-task-packet-index-v3",
            "status": "PASS_NO_ACTIVE_TASKS",
            "task_packets": [],
        },
    )
    index_ref = artifact_ref(paths["index"])
    _write(
        paths["pointer"],
        {
            "index_path": str(paths["index"].relative_to(root)),
            "index_sha256": index_ref["sha256"],
        },
    )


def _publisher(root: Path):
    def publish(
        _authority: dict[str, Any],
        task_state: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        revision = int(kwargs["expected_revision"]) + 1
        generation = f"gov-test-{revision}"
        index = dict(kwargs["task_packet_index_value"])
        index.update(governance_revision=revision, generation_id=generation)
        atomic_json(kwargs["task_packet_index_path"], index)
        atomic_json(root / "docs/governance/LONG_HORIZON_TASK_STATE.json", task_state)
        atomic_json(
            root / "docs/governance/CURRENT_STATUS_RECEIPT.json",
            {"governance_revision": revision, "files": {}},
        )
        return {"governance_revision": revision, "generation_id": generation}

    return publish


def _assert_snapshot(attempt: Path, snapshot: dict[str, bytes]) -> None:
    assert {
        str(path.relative_to(attempt)): path.read_bytes()
        for path in attempt.rglob("*")
        if path.is_file()
    } == snapshot


def test_rebind_registers_runs_and_finalizes_without_changing_source_tree(tmp_path: Path) -> None:
    source_attempt, snapshot = _setup_source_attempt(tmp_path)
    _setup_governance(tmp_path)
    publisher = _publisher(tmp_path)
    registration = register.register(
        expected_revision=30,
        root=tmp_path,
        publisher=publisher,
        created_at="2026-09-20T17:00:00+08:00",
    )
    assert registration["status"] == "PASSED"

    result = run.run_successor(root=tmp_path)
    assert result["status"] == "PASSED"
    assert result["source_lane_status"] == "BLOCKED_ADOPTION_OBSERVATIONS"
    assert result["stale_reference_occurrences"] == 13
    assert result["unique_stale_paths"] == 8
    assert result["all_mappings_unique"] is True
    assert result["all_bytes_sha_exact"] is True
    assert result["model_calls"] == 0
    assert result["gpu_calls"] == 0
    _assert_snapshot(source_attempt, snapshot)

    corrected_result = load_json(Path(result["corrected_lane_result"]["path"]))
    corrected_ledger = load_json(Path(result["corrected_lane_ledger"]["path"]))
    assert all(
        run.STALE_COMPONENT_PREFIX not in str(reference["path"])
        for _, reference in run._walk_artifact_refs(corrected_result)
    )
    assert all(
        run.STALE_COMPONENT_PREFIX not in str(reference["path"])
        for _, reference in run._walk_artifact_refs(corrected_ledger)
    )
    assert "CORRECTED_EVIDENCE" in corrected_result["ledger"]["path"]
    assert "CORRECTED_EVIDENCE" in corrected_ledger["outputs"]["input_provenance"]["path"]
    with pytest.raises(run.RebindError, match="fresh successor attempt"):
        run.run_successor(root=tmp_path)

    receipt = finalize.finalize(
        expected_revision=31,
        root=tmp_path,
        publisher=publisher,
        created_at="2026-09-20T17:05:00+08:00",
    )
    assert receipt["status"] == "PASSED"
    assert receipt["current_index_status"] == "PASS_NO_ACTIVE_TASKS"
    terminal_index = load_json(tmp_path / "tasks/current/INDEX.json")
    assert terminal_index["status"] == "PASS_NO_ACTIVE_TASKS"
    assert terminal_index["task_packets"] == []
    _assert_snapshot(source_attempt, snapshot)


def test_rebind_fails_closed_on_sha_mismatch_without_publishing(tmp_path: Path) -> None:
    source_attempt, snapshot = _setup_source_attempt(tmp_path)
    lane_result_path = source_attempt / "AI1_CURRENT_ONLY_LANE/RESULT.json"
    lane_result = load_json(lane_result_path)
    lane_result["ledger"]["sha256"] = "0" * 64
    atomic_json(lane_result_path, lane_result)
    snapshot[str(lane_result_path.relative_to(source_attempt))] = lane_result_path.read_bytes()
    with pytest.raises(run.RebindError, match="SHA mismatch"):
        run.run_successor(root=tmp_path, require_route=False)
    assert not (tmp_path / run.OUTPUT_ROOT_RELATIVE).exists()
    _assert_snapshot(source_attempt, snapshot)


def test_rebind_fails_closed_on_ambiguous_stale_suffix(tmp_path: Path) -> None:
    source_attempt, snapshot = _setup_source_attempt(tmp_path)
    lane_result_path = source_attempt / "AI1_CURRENT_ONLY_LANE/RESULT.json"
    lane_result = load_json(lane_result_path)
    lane_result["ledger"]["path"] = lane_result["ledger"]["path"].replace(
        "/.attempt_0001.staging-deadbeef/",
        "/AI1_CURRENT_ONLY_LANE/.attempt_0001.staging-deadbeef/",
    )
    atomic_json(lane_result_path, lane_result)
    snapshot[str(lane_result_path.relative_to(source_attempt))] = lane_result_path.read_bytes()
    with pytest.raises(run.RebindError, match="ambiguous"):
        run.run_successor(root=tmp_path, require_route=False)
    assert not (tmp_path / run.OUTPUT_ROOT_RELATIVE).exists()
    _assert_snapshot(source_attempt, snapshot)
