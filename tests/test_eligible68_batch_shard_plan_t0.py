from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import author_qa_eligible68_batch_shard_plan_t0 as author_qa
from tools import build_eligible68_batch_shard_plan_t0 as subject
from tools import eligible68_no_discovery_guard_t0 as no_discovery_guard
from tools import run_eligible68_batch_shard_plan_verification_t0 as verification


@pytest.fixture(scope="session")
def upstream() -> dict[str, object]:
    manifest, _ = subject._read_exact_json(
        subject.MANIFEST, subject.MANIFEST_BYTES, subject.MANIFEST_SHA256
    )
    freeze, _ = subject._read_exact_json(
        subject.MANIFEST_FREEZE,
        subject.MANIFEST_FREEZE_BYTES,
        subject.MANIFEST_FREEZE_SHA256,
    )
    manifest_qa, _ = subject._read_exact_json(
        subject.MANIFEST_INDEPENDENT_QA,
        subject.MANIFEST_INDEPENDENT_QA_BYTES,
        subject.MANIFEST_INDEPENDENT_QA_SHA256,
    )
    scope, _ = subject._read_exact_json(
        subject.SCOPE, subject.SCOPE_BYTES, subject.SCOPE_SHA256
    )
    scope_qa, _ = subject._read_exact_json(
        subject.SCOPE_INDEPENDENT_QA,
        subject.SCOPE_INDEPENDENT_QA_BYTES,
        subject.SCOPE_INDEPENDENT_QA_SHA256,
    )
    decision_qa, _ = subject._read_exact_json(
        subject.A_DECISION_QA,
        subject.A_DECISION_QA_BYTES,
        subject.A_DECISION_QA_SHA256,
    )
    throughput, _ = subject._read_exact_json(
        subject.THROUGHPUT_SOURCE,
        subject.THROUGHPUT_SOURCE_BYTES,
        subject.THROUGHPUT_SOURCE_SHA256,
    )
    storage, _ = subject._read_exact_json(
        subject.STORAGE_SOURCE,
        subject.STORAGE_SOURCE_BYTES,
        subject.STORAGE_SOURCE_SHA256,
    )
    return {
        "manifest": manifest,
        "freeze": freeze,
        "manifest_qa": manifest_qa,
        "scope": scope,
        "scope_qa": scope_qa,
        "decision_qa": decision_qa,
        "throughput": throughput,
        "storage": storage,
    }


@pytest.fixture(scope="session")
def plan() -> dict[str, object]:
    return subject.build_plan()


def validate(values: dict[str, object]) -> list[dict[str, object]]:
    return subject._validate_upstream(
        values["manifest"],
        values["freeze"],
        values["manifest_qa"],
        values["scope"],
        values["scope_qa"],
        values["decision_qa"],
        values["throughput"],
        values["storage"],
    )


def test_literal_population_order_and_forbidden_union_are_disjoint() -> None:
    assert len(subject.TRAIN60) == 60
    assert len(subject.VALIDATION8) == 8
    assert len(subject.ELIGIBLE68) == len(set(subject.ELIGIBLE68)) == 68
    assert subject.ELIGIBLE68 == subject.TRAIN60 + subject.VALIDATION8
    assert len(subject.FORBIDDEN10) == 10
    assert set(subject.ELIGIBLE68).isdisjoint(subject.FORBIDDEN10)
    assert subject.TRAIN60[0] == "grap_a_cap_004"
    assert subject.VALIDATION8[0] == "grap_a_cap_002"


def test_exact_upstream_validates_to_full_order(upstream: dict[str, object]) -> None:
    rows = validate(upstream)
    assert [row["session_id"] for row in rows] == list(subject.ELIGIBLE68)
    assert sum(row["frame_count"] for row in rows[:60]) == 25175
    assert sum(row["frame_count"] for row in rows[60:]) == 3090


def test_balanced_contiguous_partition_is_deterministic_and_indivisible(
    upstream: dict[str, object],
) -> None:
    rows = validate(upstream)
    train = subject.balanced_contiguous_partitions(rows[:60])
    validation = subject.balanced_contiguous_partitions(rows[60:])
    assert len(train) == 13
    assert len(validation) == 2
    assert [sum(row["frame_count"] for row in shard) for shard in train] == [
        2075, 1922, 2028, 1664, 2062, 2088, 1774, 2106, 1729, 1805, 2226, 1655, 2041
    ]
    assert [sum(row["frame_count"] for row in shard) for shard in validation] == [1479, 1611]
    flattened = [row["session_id"] for shard in train + validation for row in shard]
    assert flattened == list(subject.ELIGIBLE68)
    assert len(flattened) == len(set(flattened)) == 68


def test_plan_has_all_session_checkpoint_boundaries_and_zero_admission(
    plan: dict[str, object],
) -> None:
    assert plan["status"] == "DESIGN_ONLY_WAITING_A1_RELEASE"
    assert plan["population"]["eligible68_frames"] == 28265
    assert plan["sharding_contract"]["total_shards"] == 15
    shards = plan["shards"]
    assert sum(shard["session_count"] for shard in shards) == 68
    assert sum(shard["frame_count"] for shard in shards) == 28265
    assert sum(len(shard["checkpoint_boundaries"]) for shard in shards) == 68
    assert all(
        shard["restart_contract_design"]["session_semantics_split_across_shards"] is False
        for shard in shards
    )
    assert plan["admission"] == {
        "a1_released": False,
        "pixel_or_selector_execution_admission": False,
        "gpu_queueable_sessions": 0,
        "commands_generated": 0,
        "gpu_queue_records_written": 0,
        "supervisor_cpu_or_gpu_queue_mutations": 0,
        "coverage_results_created": False,
        "candidate_acceptance_or_baseline_freeze": False,
        "promotion_or_bucket_unlock": False,
        "next_required_transition": "OWNER_A1_RELEASE_PLUS_EXACT_BYTE_SUCCESSOR_IMPLEMENTATION_AND_INDEPENDENT_CPU_QA;_THIS_PLAN_ALONE_CAN_NEVER_QUEUE_GPU",
    }


def test_estimates_are_integer_recomputable_and_not_admission(
    plan: dict[str, object],
) -> None:
    pure = plan["pure_estimates"]
    assert pure["throughput"]["aggregate_gpu_seconds_ceiling"] == 17256
    assert pure["throughput"]["estimate_only_not_sla_or_admission"] is True
    assert pure["storage"]["eligible68_proportional_hard_cap_estimate_bytes"] == 60137343439
    assert pure["storage"]["prediction_is_not_storage_admission"] is True
    for shard in plan["shards"]:
        frames = shard["frame_count"]
        assert shard["estimated_gpu_seconds_ceiling_at_1_638_fps"] == subject._ceil_ratio(
            frames * 1000, 1638
        )
        cap = subject._ceil_ratio(frames * 1610612736, 757)
        assert shard["proportional_hard_cap_estimate_bytes"] == cap
        assert shard["minimum_free_before_shard_estimate_bytes"] == 8589934592 + cap + 8388608


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda value: value["manifest"]["sessions"][0].__setitem__(
                "session_id", "grap_a_cap_025"
            ),
            "literal allowlist/order drift",
        ),
        (
            lambda value: value["manifest"]["sessions"][0].__setitem__("frame_count", 459),
            "train60 frame aggregate drift",
        ),
        (
            lambda value: value["scope"]["execution_admission"].__setitem__(
                "current_pixel_or_selector_admission", True
            ),
            "unexpectedly grants execution admission",
        ),
        (
            lambda value: value["decision_qa"]["admission"].__setitem__("execution", "RUN"),
            "unexpectedly admits execution",
        ),
        (
            lambda value: value["storage"]["budget"].__setitem__(
                "minimum_free_before_admission_bytes", 8858365952
            ),
            "does not reserve the full hard cap",
        ),
        (
            lambda value: value["throughput"]["throughput"].__setitem__(
                "processed_fps", 1.7
            ),
            "does not recompute",
        ),
    ],
)
def test_fail_closed_on_population_admission_storage_and_rate_drift(
    upstream: dict[str, object], mutator: object, message: str
) -> None:
    changed = deepcopy(upstream)
    mutator(changed)
    with pytest.raises(subject.ShardPlanError, match=message):
        validate(changed)


def test_producer_and_qa_sources_have_no_discovery_or_queue_mutation_calls() -> None:
    paths = (
        Path(subject.__file__),
        Path(author_qa.__file__),
        Path("/mnt/workspace/code/chaoyang/tools/freeze_eligible68_batch_shard_plan_t0.py"),
    )
    for path in paths:
        source_text = path.read_text(encoding="utf-8")
        assert no_discovery_guard.find_forbidden_discovery_calls(source_text) == ()
        assert "append_jsonl" not in source_text
        assert "QUEUE_GPU.jsonl" not in source_text
        assert "QUEUE_CPU.jsonl" not in source_text


def test_author_qa_independently_recomputes_in_memory_plan(
    plan: dict[str, object],
) -> None:
    qa = author_qa.run_qa(
        plan,
        SimpleNamespace(
            path="/synthetic/ELIGIBLE68_BATCH_SHARD_PLAN_T0_V1.json",
            bytes=1,
            sha256="0" * 64,
            device=0,
            inode=0,
        ),
    )
    assert qa["status"] == (
        "PASS_B_CLASS_SUCCESSOR_GUARD_BOUND_DESIGN_ONLY_WAITING_A1_RELEASE_"
        "ZERO_EXECUTION_ADMISSION"
    )
    assert qa["recomputed"]["total_shards"] == 15
    assert qa["recomputed"]["session_terminal_checkpoint_boundaries"] == 68
    assert qa["verdict"]["pixel_or_selector_execution_admission"] is False
    assert qa["verdict"]["gpu_queueable_sessions"] == 0
    assert qa["access_boundary"]["explicit_named_metadata_or_code_files_opened"] == 13
    assert qa["access_boundary"]["explicit_named_open_formula"] == (
        "subject 1 + independent_sources 5 + implementation 7"
    )
    assert qa["b_class_successor"]["v1_author_qa_independently_recomputed_explicit_opens"] == 12
    assert qa["b_class_successor"]["v2_explicit_opens"] == 13
    assert qa["b_class_successor"]["static_guard_findings"] == 0
    guard = qa["implementation"]["no_discovery_guard"]
    assert guard["bytes"] == 5572
    assert guard["sha256"] == "a032954aaa0a322a42e7c841a12184c2dc5f4616aeded224473408cc829aae97"
    assert "shared_immutable_io" in qa["implementation"]


def test_durable_successor_binds_guard_in_dependency_and_exact_argv() -> None:
    assert "tools/eligible68_no_discovery_guard_t0.py" in verification.FILES
    assert verification.NO_DISCOVERY_GUARD_BYTES == 5572
    assert verification.NO_DISCOVERY_GUARD_SHA256 == (
        "a032954aaa0a322a42e7c841a12184c2dc5f4616aeded224473408cc829aae97"
    )
