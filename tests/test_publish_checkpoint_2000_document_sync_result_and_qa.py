from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest


TOOLS = Path("/mnt/workspace/code/chaoyang/tools")
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import immutable_artifact_io as strict_io  # noqa: E402
import publish_checkpoint_2000_document_sync_result_and_qa as subject  # noqa: E402


class Clock:
    def __init__(self, *values: int) -> None:
        self._values = iter(values)

    def __call__(self) -> int:
        return next(self._values)


def _paths(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    checkpoint = tmp_path / "checkpoints"
    audit = checkpoint / "audits"
    audit.mkdir(parents=True)
    return (
        checkpoint,
        audit,
        checkpoint / "CHECKPOINT_2000_DOCUMENT_SYNC_RESULT_T0_V1.json",
        audit / "INDEPENDENT_QA_CHECKPOINT_2000_DOCUMENT_SYNC_RESULT_T0_V1.json",
    )


def _base_payloads() -> tuple[dict[str, object], dict[str, object]]:
    result = {
        "schema_version": "checkpoint-2000-document-sync-result-t0-v1",
        "cpu11_single_wall_timer": {"clock": "CLOCK_MONOTONIC", "budget_seconds": 1200},
        "execution_gpu_pixel_queue_admission": 0,
        "lineage_contract": {"result_contains_qa_hash": False, "hash_cycle": False},
    }
    qa = {
        "schema_version": "independent-qa-checkpoint-2000-document-sync-result-t0-v1",
        "overall_verdict": {"verdict": subject.PASS_VERDICT},
        "subject": {"runtime_explicit_reference_count": 13},
        "cpu11_single_wall_timer": {"clock": "CLOCK_MONOTONIC", "budget_seconds": 1200},
        "execution_gpu_pixel_queue_admission": 0,
    }
    return result, qa


def test_result_is_published_before_qa_and_qa_binds_exact_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, audit, result_path, qa_path = _paths(tmp_path)
    result, qa = _base_payloads()
    calls: list[str] = []
    original = subject.strict_io.write_new_json

    def traced(path: Path, value: dict[str, object], **kwargs: object) -> dict[str, object]:
        calls.append(str(path))
        return original(path, value, **kwargs)

    monkeypatch.setattr(subject.strict_io, "write_new_json", traced)
    command = subject._publish_pair(
        result,
        qa,
        result_path=result_path,
        qa_path=qa_path,
        result_root=checkpoint,
        qa_root=audit,
        timer_start_ns=1_000,
        monotonic_ns=Clock(2_000, 3_000),
    )

    assert calls == [str(result_path), str(qa_path)]
    result_bytes = result_path.read_bytes()
    qa_payload = json.loads(qa_path.read_bytes())
    bound = qa_payload["subject"]["document_sync_result"]
    assert bound["path"] == str(result_path)
    assert bound["bytes"] == len(result_bytes)
    assert bound["sha256"] == hashlib.sha256(result_bytes).hexdigest()
    assert str(qa_path).encode() not in result_bytes
    assert command["binding_direction"] == "QA_TO_EARLIER_RESULT_NO_HASH_CYCLE"
    assert command["execution_gpu_pixel_queue_admission"] == 0


def test_preexisting_qa_rejects_before_result_creation(tmp_path: Path) -> None:
    checkpoint, audit, result_path, qa_path = _paths(tmp_path)
    strict_io.write_new_json(qa_path, {"historical": True}, mode=0o440, allowed_root=audit)
    result, qa = _base_payloads()
    with pytest.raises(subject.SuccessorError, match="already exists"):
        subject._publish_pair(
            result,
            qa,
            result_path=result_path,
            qa_path=qa_path,
            result_root=checkpoint,
            qa_root=audit,
            timer_start_ns=0,
            monotonic_ns=Clock(1, 2),
        )
    assert not result_path.exists()


def test_second_timer_gate_crossing_rejects_both_outputs(tmp_path: Path) -> None:
    checkpoint, audit, result_path, qa_path = _paths(tmp_path)
    result, qa = _base_payloads()
    with pytest.raises(subject.SuccessorError, match="QA final prepublication gate exceeds"):
        subject._publish_pair(
            result,
            qa,
            result_path=result_path,
            qa_path=qa_path,
            result_root=checkpoint,
            qa_root=audit,
            timer_start_ns=0,
            monotonic_ns=Clock(1_199_900_000_000, 1_200_000_000_001),
        )
    assert not result_path.exists()
    assert not qa_path.exists()


def test_existing_result_is_never_overwritten(tmp_path: Path) -> None:
    checkpoint, audit, result_path, qa_path = _paths(tmp_path)
    original = {"immutable": "historical"}
    strict_io.write_new_json(result_path, original, mode=0o440, allowed_root=checkpoint)
    before = result_path.read_bytes()
    result, qa = _base_payloads()
    with pytest.raises(subject.SuccessorError, match="already exists"):
        subject._publish_pair(
            result,
            qa,
            result_path=result_path,
            qa_path=qa_path,
            result_root=checkpoint,
            qa_root=audit,
            timer_start_ns=0,
            monotonic_ns=Clock(1, 2),
        )
    assert result_path.read_bytes() == before
    assert not qa_path.exists()


def test_cli_and_frozen_contract_are_exact() -> None:
    parser = subject.parser()
    assert tuple(subject.ROLE_ORDER) == (
        "pre2000_pointer",
        "owner_snapshot",
        "task05_snapshot",
        "task06_snapshot",
        "task15_snapshot",
        "owner_postlive",
        "task05_postlive",
        "task06_postlive",
        "task15_postlive",
        "final_metrics",
        "final_redline",
        "final_pack",
        "final_artifacts_qa",
    )
    assert parser.prog
    assert subject.PASS_VERDICT.startswith("PASS_CHECKPOINT_2000_DOCUMENT_SYNC_")
    assert subject.PREDECESSOR_BYTES == 54673
    assert subject.PREDECESSOR_SHA256 == "93a4a2992bb09323a4057cfd7e4513f06a366f23203f937a51258bc99ef94fdf"
