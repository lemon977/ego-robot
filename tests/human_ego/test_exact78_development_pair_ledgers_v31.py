from __future__ import annotations

import json
from pathlib import Path

import pytest

from chaoyang.human_ego.tools import (
    build_exact78_development_pair_ledgers_v31 as subject,
)
from chaoyang.human_ego.tools import train_visual_aux_future2d_v53 as trainer


def dump(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def make_inventory(data_root: Path) -> dict:
    for task, date, relative, count in subject.ROOT_SPECS:
        prefix = "get_potato_chips_" if task == "chips" else "play_cards_"
        for index in range(1, count + 1):
            session_id = f"{prefix}{date}_{index:03d}"
            session = data_root / relative / session_id
            session.mkdir(parents=True)
            (session / f"CameraRecord_{session_id}.mp4").write_bytes(b"rgb")
            dump(
                session / "clip_manifest.json",
                {
                    "clip_name": session_id,
                    "run_id": f"run-{session_id}",
                    "pieces": [{"session": f"source-{session_id}"}],
                },
            )
    inventory, blockers = subject.discover_raw_inventory(data_root)
    assert blockers == []
    return inventory


def make_cohort(inventory: dict) -> dict:
    rows = []
    for task in ("chips", "poker"):
        task_rows = sorted(
            (row for row in inventory["rows"] if row["task"] == task),
            key=lambda row: row["session_id"],
        )
        boundaries = (60, 68, 73, 78)
        for index, row in enumerate(task_rows):
            split = (
                "train"
                if index < boundaries[0]
                else "validation"
                if index < boundaries[1]
                else "test"
                if index < boundaries[2]
                else "heldout"
            )
            rows.append(
                {
                    "session_id": row["session_id"],
                    "task": task,
                    "split": split,
                    "source_group_id": f"source-group:{row['session_id']}",
                    "raw_path": row["raw_path"],
                }
            )
    return {"schema_version": subject.COHORT_SCHEMA, "sessions": rows}


def suffix_payload(*, bad_field: str | None = None) -> dict:
    fields = {
        name: {
            "suffix_invariant": name != bad_field,
            "effective_temporal_authority": (
                "OFFLINE_NONCAUSAL" if name == bad_field else "CAUSAL_CURRENT"
            ),
        }
        for name in trainer.V31_REQUIRED_SUFFIX_FIELDS
    }
    return {
        "schema_version": "EXACT78_SUFFIX_INVARIANCE_V31",
        "status": "PASS" if bad_field is None else "FAIL_CAUSAL_CURRENT_SUFFIX_DEPENDENCE",
        "causal_current_inputs_authorized": bad_field is None,
        "fields": fields,
    }


def test_current_raw_inventory_has_exact78_candidate_capacity(tmp_path) -> None:
    inventory = make_inventory(tmp_path / "datasets" / "ego")
    assert inventory["status"] == "PASS_RAW_CANDIDATE_POOL"
    assert inventory["counts"] == {"sessions": 156, "chips": 78, "poker": 78}
    assert all(row["split"] == "UNKNOWN" for row in inventory["rows"])
    assert all(row["independence"] == "UNKNOWN" for row in inventory["rows"])


def test_current_raw_inventory_allows_extra_candidates_without_changing_denominator(
    tmp_path,
) -> None:
    data_root = tmp_path / "datasets" / "ego"
    make_inventory(data_root)
    session_id = "get_potato_chips_0902_999"
    session = data_root / "chips_cards_tracker_0902/potato_chips" / session_id
    session.mkdir(parents=True)
    (session / f"CameraRecord_{session_id}.mp4").write_bytes(b"rgb")
    dump(
        session / "clip_manifest.json",
        {
            "clip_name": session_id,
            "run_id": f"run-{session_id}",
            "pieces": [{"session": f"source-{session_id}"}],
        },
    )

    inventory, blockers = subject.discover_raw_inventory(data_root)

    assert blockers == []
    assert inventory["status"] == "PASS_RAW_CANDIDATE_POOL"
    assert inventory["counts"] == {"sessions": 157, "chips": 79, "poker": 78}
    assert "not the frozen Exact78 denominator" in inventory["claim_limit"]


def test_current_cohort_closes_split_and_rejects_cross_split_source_group(
    tmp_path,
) -> None:
    inventory = make_inventory(tmp_path / "datasets" / "ego")
    cohort = make_cohort(inventory)
    path = tmp_path / "CURRENT_EXACT78_COHORT.json"
    dump(path, cohort)
    _, indexed = subject.validate_current_cohort(path, inventory)
    assert len(indexed) == 156

    validation = next(
        row for row in cohort["sessions"] if row["task"] == "chips" and row["split"] == "validation"
    )
    train = next(
        row for row in cohort["sessions"] if row["task"] == "chips" and row["split"] == "train"
    )
    validation["source_group_id"] = train["source_group_id"]
    dump(path, cohort)
    with pytest.raises(subject.AdmissionError, match="source group crosses splits"):
        subject.validate_current_cohort(path, inventory)


def test_suffix_audit_requires_all_current_inputs_to_be_causal(tmp_path) -> None:
    good = tmp_path / "good.json"
    dump(good, suffix_payload())
    assert subject.validate_suffix_audit(good)["status"] == "PASS"
    bad = tmp_path / "bad.json"
    dump(bad, suffix_payload(bad_field="robotized_rgb"))
    with pytest.raises(subject.AdmissionError, match="did not pass"):
        subject.validate_suffix_audit(bad)


def test_task_ledgers_bind_same_manifests_and_suffix_receipts(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cohort = tmp_path / "CURRENT_EXACT78_COHORT.json"
    cohort.write_text("{}\n", encoding="utf-8")
    admitted = []
    for split in ("train", "validation"):
        session = f"chips_{split}"
        manifest = tmp_path / session / "VISUAL_AUX_SESSION_MANIFEST.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text("{}\n", encoding="utf-8")
        audit = tmp_path / session / "SUFFIX.json"
        dump(audit, suffix_payload())
        admitted.append(
            {
                "session_id": session,
                "task": "chips",
                "split": split,
                "source_group_id": f"group-{split}",
                "manifest": subject.artifact(manifest),
                "suffix_audit": subject.artifact(audit),
                "eligible_h50_starts": [0],
                "eligible_h50_window_count": 1,
                "suffix_status": "PASS",
            }
        )
    monkeypatch.setattr(trainer, "validate_paired_ledgers", lambda *args: {})
    result, blockers = subject._task_outputs(
        "chips", admitted, tmp_path / "out", subject.artifact(cohort)
    )
    assert blockers == []
    assert result["train_pairs"] == 1
    assert result["validation_pairs"] == 1
    raw = json.loads(
        Path(result["ledgers"]["HUMAN_RAW_RGB"]["path"]).read_text()
    )
    robot = json.loads(
        Path(result["ledgers"]["ROBOTIZED_RGB"]["path"]).read_text()
    )
    assert raw["pair_dataset_signature"] == robot["pair_dataset_signature"]
    assert raw["train"] == robot["train"]
    assert raw["validation"] == robot["validation"]
    assert raw["suffix_invariance_receipts"] == robot["suffix_invariance_receipts"]


def test_trainer_rechecks_v31_suffix_receipt_binding(tmp_path) -> None:
    cohort = tmp_path / "cohort.json"
    cohort.write_text("{}\n", encoding="utf-8")
    manifest = tmp_path / "VISUAL_AUX_SESSION_MANIFEST.json"
    manifest.write_text("{}\n", encoding="utf-8")
    audit = tmp_path / "suffix.json"
    dump(audit, suffix_payload())
    ledger = {
        "contract": trainer.V31_DEVELOPMENT_CONTRACT,
        "cohort_authority": trainer.artifact_ref(cohort),
        "suffix_invariance_receipts": [
            {
                "session_id": "session_a",
                "bundle_manifest_sha256": trainer.sha256(manifest),
                "audit": trainer.artifact_ref(audit),
            }
        ],
    }
    trainer._validate_v31_suffix_receipts(ledger, {"session_a": manifest})
    ledger["suffix_invariance_receipts"][0]["bundle_manifest_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="manifest SHA mismatch"):
        trainer._validate_v31_suffix_receipts(ledger, {"session_a": manifest})


def test_trainer_rechecks_current_cohort_source_groups(tmp_path) -> None:
    inventory = make_inventory(tmp_path / "datasets" / "ego")
    cohort = tmp_path / "CURRENT_EXACT78_COHORT.json"
    dump(cohort, make_cohort(inventory))
    ledger = {
        "contract": trainer.V31_DEVELOPMENT_CONTRACT,
        "cohort_authority": trainer.artifact_ref(cohort),
    }

    indexed = trainer._validate_v31_cohort_authority(ledger)

    assert len(indexed) == 156
    assert {row["task"] for row in indexed.values()} == {"chips", "poker"}
    assert all(row["source_group_id"] for row in indexed.values())


def test_trainer_rejects_archive_cohort(tmp_path) -> None:
    archive = tmp_path / "archive" / "history"
    cohort = archive / "cohort.json"
    cohort.parent.mkdir(parents=True)
    cohort.write_text("{}\n", encoding="utf-8")
    ledger = {
        "contract": trainer.V31_DEVELOPMENT_CONTRACT,
        "cohort_authority": trainer.artifact_ref(cohort),
    }
    with pytest.raises(RuntimeError, match="cohort authority must be current"):
        trainer._validate_v31_cohort_authority(ledger)


def test_current_bundle_rejects_indirect_archive_artifact(tmp_path) -> None:
    bundle = tmp_path / "current" / "VISUAL_AUX_SESSION_MANIFEST.json"
    archived = tmp_path / "archive" / "robotized.png"
    archived.parent.mkdir(parents=True)
    archived.write_bytes(b"pixels")
    dump(
        bundle,
        {
            "input_mode": "CAUSAL_TRAINING_INPUT",
            "causal_proof": {
                "enabled": True,
                "raw_robotized_valid_mask_shared": True,
            },
            "inputs": {"robotized": subject.artifact(archived)},
        },
    )

    with pytest.raises(subject.AdmissionError, match="archive artifact reference"):
        subject.validate_current_bundle(bundle)


def test_builder_without_current_cohort_or_candidates_fails_closed(tmp_path) -> None:
    data_root = tmp_path / "datasets" / "ego"
    make_inventory(data_root)
    result = subject.build(
        data_root=data_root,
        cohort_manifest=None,
        candidate_index=None,
        output_root=tmp_path / "out",
    )
    assert result["status"] == "BLOCKED_INPUTS"
    assert result["execution_allowed"] is False
    assert result["archive_consumed"] is False
    assert "MISSING_CURRENT_NON_ARCHIVE_156_COHORT_MANIFEST" in result["blockers"]
    assert "CHIPS_NO_CURRENT_LEGAL_RAW_ROBOTIZED_PAIR" in result["blockers"]
    assert "POKER_NO_CURRENT_LEGAL_RAW_ROBOTIZED_PAIR" in result["blockers"]
