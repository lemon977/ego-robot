from __future__ import annotations

import inspect
import hashlib
import json
from pathlib import Path

import pytest
import torch

from tools import (
    evaluate_h50,
    evaluate_rolling_h50,
    evaluate_training_snapshot,
    freeze_best_checkpoint,
    train_embodiment,
)
from tools.train_embodiment import formal_training_status, require_manifest_selector_ready
from training import FlowMatchingTrainer
from training.FlowMatchingTrainer import require_formal_selector_for_trainer
from utils.atomic_io import atomic_write_json
from utils.frozen_contract import (
    TRAINING_COMPLETE_SCHEMA,
    file_reference,
    runtime_checkpoint_authority_path,
    write_runtime_checkpoint_authority,
)
from utils import source_contract


def test_check_only_and_training_both_require_manifest_selector() -> None:
    with pytest.raises(RuntimeError, match="HOLD_MANIFEST_SELECTOR_REQUIRED"):
        require_manifest_selector_ready(check_only=True)
    with pytest.raises(RuntimeError, match="HOLD_MANIFEST_SELECTOR_REQUIRED"):
        require_manifest_selector_ready(check_only=False)


def test_cuda_does_not_override_formal_selector_hold() -> None:
    status = formal_training_status(cuda_available=True)
    assert status == {
        "cuda_available": True,
        "formal_training_ready": False,
        "formal_training_blocker": "HOLD_MANIFEST_SELECTOR_REQUIRED",
    }


def test_direct_trainer_cannot_bypass_selector_hold() -> None:
    with pytest.raises(RuntimeError, match="HOLD_MANIFEST_SELECTOR_REQUIRED"):
        require_formal_selector_for_trainer({})


def test_dirty_launcher_rejects_config_hidden_untracked_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "HumanEgo"
    repo_root.mkdir()
    git = train_embodiment.subprocess.run
    git(["git", "init", "-q"], cwd=repo_root, check=True)
    git(
        [
            "git",
            "-c",
            "user.name=review",
            "-c",
            "user.email=review@example.invalid",
            "commit",
            "--allow-empty",
            "-q",
            "-m",
            "initial",
        ],
        cwd=repo_root,
        check=True,
    )
    (repo_root / "untracked_sentinel").write_text("untracked\n", encoding="utf-8")
    git(
        ["git", "config", "status.showUntrackedFiles", "no"],
        cwd=repo_root,
        check=True,
    )
    hidden = git(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )
    assert hidden.stdout == ""

    run_root = tmp_path / "runs"
    artifact_root = tmp_path / "artifacts"
    run_root.mkdir()
    artifact_root.mkdir()

    def forbidden_after_dirty_gate(*_args, **_kwargs):
        raise AssertionError("formal launcher passed the dirty gate")

    monkeypatch.setattr(train_embodiment, "ROOT", repo_root)
    monkeypatch.setattr(
        train_embodiment,
        "validate_directory_path",
        forbidden_after_dirty_gate,
    )
    monkeypatch.setattr(
        train_embodiment,
        "require_manifest_selector_ready",
        forbidden_after_dirty_gate,
    )
    monkeypatch.setattr(train_embodiment, "atomic_write_json", forbidden_after_dirty_gate)
    monkeypatch.setattr(
        train_embodiment.sys,
        "argv",
        [
            "train_embodiment.py",
            "--embodiment",
            "kai22",
            "--config",
            str(repo_root / "cfg" / "training" / "missing.yaml"),
            "--split",
            str(tmp_path / "missing_split.json"),
            "--production-root",
            str(tmp_path / "production"),
            "--run-name",
            "hidden-untracked-zero-output",
            "--batch-size",
            "64",
            "--selector-manifest",
            str(tmp_path / "missing_selector.json"),
            "--paired-kept-manifest",
            str(tmp_path / "missing_paired.json"),
            "--artifact-root",
            str(artifact_root),
            "--run-root",
            str(run_root),
        ],
    )

    with pytest.raises(
        RuntimeError,
        match="formal training requires a clean tracked Git worktree",
    ):
        train_embodiment.main()

    assert list(run_root.iterdir()) == []
    assert list(artifact_root.iterdir()) == []


def test_dirty_launcher_fails_before_any_output_or_autotune(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "HumanEgo"
    repo_root.mkdir()
    run_root = tmp_path / "runs"
    artifact_root = tmp_path / "artifacts"
    run_root.mkdir()
    artifact_root.mkdir()
    writes_or_commands: list[str] = []

    monkeypatch.setattr(train_embodiment, "ROOT", repo_root)
    monkeypatch.setattr(
        train_embodiment,
        "git_provenance",
        lambda: {
            "git_head": "dirty-test-head",
            "tracked_worktree_dirty": True,
            "git_status_includes_untracked": True,
        },
    )
    monkeypatch.setattr(
        train_embodiment,
        "atomic_write_json",
        lambda *_args, **_kwargs: writes_or_commands.append("atomic_write_json"),
    )
    monkeypatch.setattr(
        train_embodiment.subprocess,
        "run",
        lambda *_args, **_kwargs: writes_or_commands.append("subprocess.run"),
    )
    monkeypatch.setattr(
        train_embodiment.sys,
        "argv",
        [
            "train_embodiment.py",
            "--embodiment",
            "kai22",
            "--config",
            str(repo_root / "cfg" / "training" / "missing.yaml"),
            "--split",
            str(tmp_path / "missing_split.json"),
            "--production-root",
            str(tmp_path / "production"),
            "--run-name",
            "dirty-zero-output",
            "--batch-size",
            "64",
            "--selector-manifest",
            str(tmp_path / "missing_selector.json"),
            "--paired-kept-manifest",
            str(tmp_path / "missing_paired.json"),
            "--artifact-root",
            str(artifact_root),
            "--run-root",
            str(run_root),
        ],
    )

    with pytest.raises(
        RuntimeError,
        match="formal training requires a clean tracked Git worktree",
    ):
        train_embodiment.main()

    assert writes_or_commands == []
    assert list(repo_root.iterdir()) == []
    assert list(run_root.iterdir()) == []
    assert list(artifact_root.iterdir()) == []


def test_all_formal_entrypoints_validate_selector_before_split_phase2() -> None:
    direct = (
        train_embodiment.main,
        evaluate_h50.main,
        evaluate_training_snapshot.load_snapshot,
        freeze_best_checkpoint.main,
        FlowMatchingTrainer.main,
    )
    for entrypoint in direct:
        source = inspect.getsource(entrypoint)
        selector_index = min(
            index
            for marker in (
                "require_manifest_selector_ready(",
                "require_formal_selector_for_trainer(",
            )
            if (index := source.find(marker)) >= 0
        )
        assert selector_index < source.index("load_eligible68_frozen_split(")

    rolling_main = inspect.getsource(evaluate_rolling_h50.main)
    assert rolling_main.index("require_manifest_selector_ready(") < rolling_main.index(
        "validate_checkpoint_inputs(args, selector_binding)"
    )
    assert "load_eligible68_frozen_split(" in inspect.getsource(
        evaluate_rolling_h50.validate_checkpoint_inputs
    )


def test_external_artifact_and_run_roots_reach_every_formal_consumer() -> None:
    selector_consumers = (
        train_embodiment.main,
        evaluate_h50.main,
        evaluate_rolling_h50.main,
        evaluate_training_snapshot.load_snapshot,
        freeze_best_checkpoint.main,
        FlowMatchingTrainer.require_formal_selector_for_trainer,
    )
    for entrypoint in selector_consumers:
        source = inspect.getsource(entrypoint)
        assert "artifact_root=" in source

    dataset_builders = (
        train_embodiment.check_only,
        evaluate_h50.make_dataset,
        evaluate_rolling_h50.make_dataset,
        evaluate_training_snapshot.make_dataset,
        FlowMatchingTrainer.main,
    )
    for builder in dataset_builders:
        source = inspect.getsource(builder)
        assert "selector_records=" in source
        assert "selector_root=" in source

    launcher_source = inspect.getsource(train_embodiment.main)
    trainer_source = inspect.getsource(FlowMatchingTrainer.main)
    assert '"--run_root", str(run_root)' in launcher_source
    assert "validate_run_root_binding(" in trainer_source


def eligible_split_roles() -> dict:
    return {
        **source_contract.eligible68_split_protocol_fields(),
        "train": list(source_contract.TRAIN52),
        "validation": list(source_contract.DEV8),
        "test": list(source_contract.FINAL_TEST8),
    }


def eligible_run_manifest() -> dict:
    adapter_paths = {
        session_id: f"/approved/{session_id}/09_humanego_adapter"
        for session_id in (*source_contract.TRAIN52, *source_contract.DEV8)
    }
    return {
        "train_sessions": list(source_contract.TRAIN52),
        "validation_sessions": list(source_contract.DEV8),
        "evaluation_protocol": source_contract.training_evaluation_protocol_fields(),
        "adapter_paths": adapter_paths,
        "resolved_config": {
            "MPS_PATHS_TRAIN": [
                adapter_paths[session_id] for session_id in source_contract.TRAIN52
            ],
            "MPS_PATHS_EVAL": [
                adapter_paths[session_id] for session_id in source_contract.DEV8
            ],
        },
        "overfit_session": None,
    }


def completed_training_authority(
    tmp_path: Path,
    *,
    status: str = "complete",
    drift: str | None = None,
) -> Path:
    run_directory = tmp_path / "completed_run"
    run_directory.mkdir()
    run_manifest_path = run_directory / "run_manifest.json"
    run_manifest = {
        "run_directory": str(run_directory),
        "embodiment": "kai22",
    }
    run_manifest_path.write_text(json.dumps(run_manifest), encoding="utf-8")
    stats_path = run_directory / "dataset_stats.json"
    stats_path.write_text('{"mean": 1}', encoding="utf-8")
    run_manifest_reference = file_reference(run_manifest_path)
    stats_reference = file_reference(stats_path)
    best = run_directory / "best.pt"
    latest = run_directory / "latest.pt"
    torch.save({"model": {"weight": torch.tensor([1.0])}}, best)
    torch.save({"model": {"weight": torch.tensor([2.0])}}, latest)
    write_runtime_checkpoint_authority(
        best,
        run_manifest_reference=run_manifest_reference,
        dataset_stats_reference=stats_reference,
    )
    latest_run_reference = run_manifest_reference
    latest_stats_reference = stats_reference
    if drift == "run":
        alternate = run_directory / "other_run_manifest.json"
        alternate.write_text('{"run": "other"}', encoding="utf-8")
        latest_run_reference = file_reference(alternate)
    elif drift == "stats":
        alternate = run_directory / "other_dataset_stats.json"
        alternate.write_text('{"mean": 2}', encoding="utf-8")
        latest_stats_reference = file_reference(alternate)
    write_runtime_checkpoint_authority(
        latest,
        run_manifest_reference=latest_run_reference,
        dataset_stats_reference=latest_stats_reference,
    )
    completion = {
        "schema_version": TRAINING_COMPLETE_SCHEMA,
        "immutable": True,
        "no_fallback": True,
        "status": status,
        "run_manifest": run_manifest,
        "run_manifest_ref": run_manifest_reference,
        "dataset_stats_ref": stats_reference,
        "best_checkpoint_authority_ref": file_reference(
            runtime_checkpoint_authority_path(best)
        ),
        "latest_checkpoint_authority_ref": file_reference(
            runtime_checkpoint_authority_path(latest)
        ),
    }
    (run_directory / "training_complete.json").write_text(
        json.dumps(completion),
        encoding="utf-8",
    )
    return run_directory


def test_freeze_consumes_completed_external_checkpoint_authorities(
    tmp_path: Path,
) -> None:
    run_directory = completed_training_authority(tmp_path)
    authority = freeze_best_checkpoint.load_completed_checkpoint_authorities(
        run_directory
    )
    assert authority["best_checkpoint_ref"]["path"] == str(
        run_directory / "best.pt"
    )
    assert authority["latest_checkpoint_ref"]["path"] == str(
        run_directory / "latest.pt"
    )


def test_freeze_rejects_unfinished_or_missing_completion_authority(
    tmp_path: Path,
) -> None:
    unfinished = completed_training_authority(tmp_path, status="running")
    with pytest.raises(ValueError, match="training is not complete"):
        freeze_best_checkpoint.load_completed_checkpoint_authorities(unfinished)
    missing = tmp_path / "missing_completion"
    missing.mkdir()
    with pytest.raises(FileNotFoundError):
        freeze_best_checkpoint.load_completed_checkpoint_authorities(missing)


def test_freeze_rejects_missing_checkpoint_authority(tmp_path: Path) -> None:
    run_directory = completed_training_authority(tmp_path)
    runtime_checkpoint_authority_path(run_directory / "latest.pt").unlink()
    with pytest.raises(FileNotFoundError):
        freeze_best_checkpoint.load_completed_checkpoint_authorities(run_directory)


@pytest.mark.parametrize("drift", ("run", "stats"))
def test_freeze_rejects_checkpoint_authority_lineage_drift(
    tmp_path: Path,
    drift: str,
) -> None:
    run_directory = completed_training_authority(tmp_path, drift=drift)
    with pytest.raises(ValueError, match="another run|other dataset statistics"):
        freeze_best_checkpoint.load_completed_checkpoint_authorities(run_directory)


def test_freeze_rejects_aliased_best_latest_checkpoints(tmp_path: Path) -> None:
    run_directory = completed_training_authority(tmp_path)
    best = run_directory / "best.pt"
    latest = run_directory / "latest.pt"
    latest.unlink()
    latest.hardlink_to(best)
    with pytest.raises(ValueError, match="hard-linked"):
        freeze_best_checkpoint.load_completed_checkpoint_authorities(run_directory)


def test_resolved_training_paths_are_exactly_bound_to_train52_and_dev8() -> None:
    split = eligible_split_roles()
    manifest = eligible_run_manifest()
    source_contract.validate_training_run_role_contract(manifest, split)

    mutations = (
        (
            "MPS_PATHS_TRAIN",
            ["/forbidden/grap_a_cap_002/09_humanego_adapter"],
        ),
        (
            "MPS_PATHS_EVAL",
            ["/forbidden/grap_a_cap_019/09_humanego_adapter"],
        ),
        ("MPS_PATHS_TRAIN", manifest["resolved_config"]["MPS_PATHS_EVAL"]),
        ("MPS_PATHS_EVAL", manifest["resolved_config"]["MPS_PATHS_TRAIN"]),
        (
            "MPS_PATHS_EVAL",
            manifest["resolved_config"]["MPS_PATHS_EVAL"]
            + [manifest["resolved_config"]["MPS_PATHS_EVAL"][0]],
        ),
    )
    for field, replacement in mutations:
        candidate = json.loads(json.dumps(manifest))
        candidate["resolved_config"][field] = replacement
        with pytest.raises(ValueError, match="exact ordered role paths"):
            source_contract.validate_training_run_role_contract(candidate, split)

    extra_adapter = json.loads(json.dumps(manifest))
    extra_adapter["adapter_paths"][source_contract.FINAL_TEST8[0]] = (
        "/forbidden/grap_a_cap_002/09_humanego_adapter"
    )
    with pytest.raises(ValueError, match="adapter path roles"):
        source_contract.validate_training_run_role_contract(extra_adapter, split)


def test_overfit_is_train52_only_and_never_enters_formal_consumers() -> None:
    split = eligible_split_roles()
    manifest = eligible_run_manifest()
    overfit_session = source_contract.TRAIN52[0]
    overfit_path = manifest["adapter_paths"][overfit_session]
    manifest["overfit_session"] = overfit_session
    manifest["resolved_config"]["MPS_PATHS_TRAIN"] = [overfit_path]
    manifest["resolved_config"]["MPS_PATHS_EVAL"] = [overfit_path]
    source_contract.validate_training_run_role_contract(
        manifest,
        split,
        allow_overfit=True,
    )
    with pytest.raises(ValueError, match="cannot enter snapshot/freeze/formal"):
        source_contract.validate_training_run_role_contract(manifest, split)

    forbidden = json.loads(json.dumps(manifest))
    forbidden["overfit_session"] = source_contract.FINAL_TEST8[0]
    with pytest.raises(ValueError, match="exactly one train52"):
        source_contract.validate_training_run_role_contract(
            forbidden,
            split,
            allow_overfit=True,
        )


def snapshot_authority_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    payload_manifest_drift: bool = False,
    payload_stats_drift: bool = False,
) -> Path:
    run_directory = tmp_path / "training_run"
    run_directory.mkdir()
    checkpoint = run_directory / "eval_snapshots" / ".pending_ep_0001.pt"
    checkpoint.parent.mkdir()
    run_manifest = {
        "run_directory": str(run_directory),
        "embodiment": "kai22",
    }
    run_manifest_path = run_directory / "run_manifest.json"
    run_manifest_path.write_text(json.dumps(run_manifest), encoding="utf-8")
    stats_path = run_directory / "dataset_stats.json"
    stats_path.write_text(json.dumps({"mean": 1}), encoding="utf-8")
    payload_manifest = (
        {**run_manifest, "embodiment": "humanego30"}
        if payload_manifest_drift
        else run_manifest
    )
    payload_stats_sha256 = (
        "0" * 64 if payload_stats_drift else file_reference(stats_path)["sha256"]
    )
    torch.save(
        {
            "model": {"weight": torch.tensor([1.0])},
            "cfg": {
                "out_dir": str(run_directory),
                "img_name": "rgb.png",
                "robot_sidecar_root": str(tmp_path / "sidecars"),
            },
            "run_manifest": payload_manifest,
            "dataset_stats_sha256": payload_stats_sha256,
        },
        checkpoint,
    )
    write_runtime_checkpoint_authority(
        checkpoint,
        run_manifest_reference=file_reference(run_manifest_path),
        dataset_stats_reference=file_reference(stats_path),
    )
    monkeypatch.setattr(
        evaluate_training_snapshot,
        "require_manifest_selector_ready",
        lambda *args, **kwargs: {"image_name": "rgb.png"},
    )
    monkeypatch.setattr(
        evaluate_training_snapshot,
        "validate_file_reference",
        lambda *args, **kwargs: tmp_path / "split.json",
    )
    monkeypatch.setattr(
        evaluate_training_snapshot,
        "load_eligible68_frozen_split",
        lambda *args, **kwargs: {"production_root": str(tmp_path)},
    )
    monkeypatch.setattr(
        evaluate_training_snapshot,
        "validate_training_run_role_contract",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        evaluate_training_snapshot,
        "validate_formal_runtime_contract",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        evaluate_training_snapshot,
        "validate_frozen_adapters",
        lambda *args, **kwargs: None,
    )
    return checkpoint


def test_pending_snapshot_requires_and_matches_external_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = snapshot_authority_fixture(tmp_path, monkeypatch)
    payload, cfg, stats, binding = evaluate_training_snapshot.load_snapshot(checkpoint)
    assert payload["run_manifest"]["embodiment"] == "kai22"
    assert cfg.img_name == binding["image_name"] == "rgb.png"
    assert stats == {"mean": 1}


@pytest.mark.parametrize("drift", ("manifest", "stats"))
def test_pending_snapshot_rejects_payload_authority_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    checkpoint = snapshot_authority_fixture(
        tmp_path,
        monkeypatch,
        payload_manifest_drift=drift == "manifest",
        payload_stats_drift=drift == "stats",
    )
    expected = "run manifest mismatch" if drift == "manifest" else "statistics hash mismatch"
    with pytest.raises(ValueError, match=expected):
        evaluate_training_snapshot.load_snapshot(checkpoint)


def test_pending_snapshot_missing_authority_prevents_deserialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "pending.pt"
    torch.save({"run_manifest": {}}, checkpoint)
    calls = 0

    def forbidden_deserialize(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("torch.load must not run without authority")

    monkeypatch.setattr(torch, "load", forbidden_deserialize)
    with pytest.raises(FileNotFoundError):
        evaluate_training_snapshot.load_snapshot(checkpoint)
    assert calls == 0


def test_dev_is_the_only_ordinary_evaluation_role() -> None:
    split = eligible_split_roles()
    authorization = source_contract.authorize_evaluation_role(split, "dev")
    assert authorization["storage_role"] == "validation"
    assert authorization["sessions"] == list(source_contract.DEV8)
    assert authorization["final_test_authorized"] is False
    with pytest.raises(ValueError, match="may not be supplied for dev"):
        source_contract.authorize_evaluation_role(
            split,
            "dev",
            final_test_gate_reference={"path": "/never-opened"},
        )
    with pytest.raises(RuntimeError, match="HOLD_FINAL_TEST_GATE_REQUIRED"):
        source_contract.authorize_evaluation_role(split, "final_test")


def final_test_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate=None,
) -> tuple[dict, dict, Path]:
    split = eligible_split_roles()
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split), encoding="utf-8")
    split_ref = file_reference(split_path)
    output_parent = tmp_path / "results"
    output_parent.mkdir()
    run_root = tmp_path / "runs"
    run_root.mkdir()
    artifact_roots = {}
    selector_refs = {}
    for product_line in sorted(source_contract.PRODUCT_LINES):
        artifact_root = tmp_path / "artifacts" / product_line.lower()
        artifact_root.mkdir(parents=True)
        artifact_roots[product_line] = str(artifact_root)
        selector_path = tmp_path / f"{product_line}_SELECTOR.json"
        selector_path.write_text(
            json.dumps(
                {
                    "product_line": product_line,
                    "image_name": (
                        "rgb.png" if product_line == "RAW" else "robot_rgb.png"
                    ),
                }
            ),
            encoding="utf-8",
        )
        selector_refs[product_line] = file_reference(selector_path)
    paired_path = tmp_path / "PAIRED.json"
    paired_path.write_text(json.dumps({"selector_refs": selector_refs}), encoding="utf-8")
    paired_ref = file_reference(paired_path)

    def selector_binding(selector_reference, paired_reference, *, artifact_root):
        product_line = next(
            product
            for product, reference in selector_refs.items()
            if reference == selector_reference
        )
        return {
            "product_line": product_line,
            "image_name": "rgb.png" if product_line == "RAW" else "robot_rgb.png",
            "artifact_roots": artifact_roots,
        }

    monkeypatch.setattr(
        source_contract,
        "require_manifest_selector_ready",
        selector_binding,
    )
    twins = {}
    for product_line in sorted(source_contract.PRODUCT_LINES):
        bundle = tmp_path / product_line
        bundle.mkdir()
        run_directory = run_root / product_line.lower()
        run_directory.mkdir()
        checkpoint = bundle / "best.pt"
        checkpoint.write_bytes(product_line.encode("utf-8"))
        checkpoint_ref = file_reference(checkpoint)
        run_manifest = {
            "embodiment": "kai22",
            "split_ref": split_ref,
            "selector_manifest_ref": selector_refs[product_line],
            "paired_kept_manifest_ref": paired_ref,
            "artifact_root": artifact_roots[product_line],
            "run_root": str(run_root),
            "run_directory": str(run_directory),
        }
        run_manifest_path = bundle / "run_manifest.json"
        run_manifest_path.write_text(json.dumps(run_manifest), encoding="utf-8")
        freeze = {
            "schema_version": source_contract.FROZEN_BEST_SCHEMA,
            "immutable": True,
            "no_fallback": True,
            "status": "frozen",
            "embodiment": "kai22",
            "product_line": product_line,
            "selection_role": "dev",
            "selection_sessions": list(source_contract.DEV8),
            "final_test_status": "NOT_EVALUATED",
            "split_ref": split_ref,
            "checkpoint_ref": checkpoint_ref,
            "frozen_run_manifest_ref": file_reference(run_manifest_path),
            "selector_manifest_ref": selector_refs[product_line],
            "paired_kept_manifest_ref": paired_ref,
            "artifact_root": artifact_roots[product_line],
            "artifact_roots": artifact_roots,
            "image_name": (
                "rgb.png" if product_line == "RAW" else "robot_rgb.png"
            ),
            "run_root": str(run_root),
            "run_directory": str(run_directory),
        }
        freeze_path = bundle / "freeze_manifest.json"
        freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
        twins[product_line] = {
            "freeze_manifest_ref": file_reference(freeze_path),
            "checkpoint_ref": checkpoint_ref,
            "output_directory": str(output_parent / product_line.lower()),
        }
    gate = {
        "schema_version": source_contract.FINAL_TEST_GATE_SCHEMA,
        "immutable": True,
        "no_fallback": True,
        "status": "READY_FOR_ONE_SHOT_FINAL_TEST",
        "cohort": "eligible68",
        "split_protocol": source_contract.ELIGIBLE68_SPLIT_PROTOCOL,
        "final_test_sessions": list(source_contract.FINAL_TEST8),
        "one_shot": True,
        "selection_role": "dev",
        "embodiment": "kai22",
        "evaluator": "H50_BLOCK_OPEN_LOOP_V1",
        "evaluation_id": "eligible68-final-v1",
        "split_ref": split_ref,
        "frozen_twins": twins,
    }
    if mutate is not None:
        mutate(gate, twins)
    gate_path = tmp_path / "FINAL_TEST_GATE.json"
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    authorization = source_contract.authorize_evaluation_role(
        split,
        "final_test",
        final_test_gate_reference=file_reference(gate_path),
        split_reference=split_ref,
        checkpoint_reference=twins["RAW"]["checkpoint_ref"],
        product_line="RAW",
        embodiment="kai22",
        evaluator="H50_BLOCK_OPEN_LOOP_V1",
        output_directory=twins["RAW"]["output_directory"],
    )
    return authorization, twins, gate_path


def test_final_test_gate_binds_both_frozen_twins_and_claims_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorization, _, _ = final_test_authorization(tmp_path, monkeypatch)
    claim = source_contract.claim_final_test_output(authorization)
    assert claim.is_file()
    with pytest.raises(FileExistsError, match="already claimed"):
        source_contract.claim_final_test_output(authorization)


def test_held_final_test_output_cannot_be_redirected_after_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorization, _, _ = final_test_authorization(tmp_path, monkeypatch)
    lease = source_contract.claim_final_test_output(authorization, hold=True)
    assert isinstance(lease, source_contract.FinalTestOutputLease)
    output = Path(authorization["output_directory"])
    moved = output.with_name(f"{output.name}.moved")
    output.rename(moved)
    external = tmp_path / "post_claim_external"
    external.mkdir()
    output.symlink_to(external, target_is_directory=True)

    atomic_write_json(
        lease.bound_directory / "evaluation.json",
        {"status": "bound-to-claimed-inode"},
    )
    assert (moved / "evaluation.json").is_file()
    assert not (external / "evaluation.json").exists()
    with pytest.raises(ValueError, match="non-symlink directory|changed"):
        lease.close()


def test_dataset_stats_parse_and_digest_share_one_byte_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "dataset_stats.json"
    first = b'{"pos":{"mean":[1],"std":[2]}}\n'
    second = b'{"pos":{"mean":[9],"std":[9]}}\n'
    path.write_bytes(first)
    real_reader = FlowMatchingTrainer.read_ordinary_file_bytes

    def read_then_replace(candidate, *, label):
        result = real_reader(candidate, label=label)
        path.write_bytes(second)
        return result

    monkeypatch.setattr(
        FlowMatchingTrainer,
        "read_ordinary_file_bytes",
        read_then_replace,
    )
    stats, digest = FlowMatchingTrainer.load_dataset_stats_binding(path)
    assert stats == {"pos": {"mean": [1], "std": [2]}}
    assert digest == hashlib.sha256(first).hexdigest()
    assert path.read_bytes() == second


def test_final_test_gate_rejects_same_checkpoint_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def alias_checkpoint(gate, twins):
        robot_freeze_path = Path(twins["ROBOT_RGB"]["freeze_manifest_ref"]["path"])
        robot_freeze = json.loads(robot_freeze_path.read_text(encoding="utf-8"))
        robot_freeze["checkpoint_ref"] = twins["RAW"]["checkpoint_ref"]
        robot_freeze_path.write_text(json.dumps(robot_freeze), encoding="utf-8")
        twins["ROBOT_RGB"]["freeze_manifest_ref"] = file_reference(robot_freeze_path)
        twins["ROBOT_RGB"]["checkpoint_ref"] = twins["RAW"]["checkpoint_ref"]

    with pytest.raises(ValueError, match="escapes approved roots|checkpoint path drift"):
        final_test_authorization(tmp_path, monkeypatch, mutate=alias_checkpoint)


def test_final_test_gate_rejects_hardlinked_twin_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def hardlink_checkpoint(gate, twins):
        raw_checkpoint = Path(twins["RAW"]["checkpoint_ref"]["path"])
        robot_checkpoint = Path(twins["ROBOT_RGB"]["checkpoint_ref"]["path"])
        robot_checkpoint.unlink()
        robot_checkpoint.hardlink_to(raw_checkpoint)
        aliased_reference = {
            **twins["RAW"]["checkpoint_ref"],
            "path": str(robot_checkpoint),
        }
        robot_freeze_path = Path(twins["ROBOT_RGB"]["freeze_manifest_ref"]["path"])
        robot_freeze = json.loads(robot_freeze_path.read_text(encoding="utf-8"))
        robot_freeze["checkpoint_ref"] = aliased_reference
        robot_freeze_path.write_text(json.dumps(robot_freeze), encoding="utf-8")
        twins["ROBOT_RGB"]["freeze_manifest_ref"] = file_reference(robot_freeze_path)
        twins["ROBOT_RGB"]["checkpoint_ref"] = aliased_reference

    with pytest.raises(ValueError, match="hard-linked"):
        final_test_authorization(tmp_path, monkeypatch, mutate=hardlink_checkpoint)


def test_final_test_gate_rejects_same_run_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def alias_run(gate, twins):
        raw_freeze_path = Path(twins["RAW"]["freeze_manifest_ref"]["path"])
        raw_freeze = json.loads(raw_freeze_path.read_text(encoding="utf-8"))
        robot_freeze_path = Path(twins["ROBOT_RGB"]["freeze_manifest_ref"]["path"])
        robot_freeze = json.loads(robot_freeze_path.read_text(encoding="utf-8"))
        robot_run_path = Path(robot_freeze["frozen_run_manifest_ref"]["path"])
        robot_run = json.loads(robot_run_path.read_text(encoding="utf-8"))
        robot_run["run_directory"] = raw_freeze["run_directory"]
        robot_run_path.write_text(json.dumps(robot_run), encoding="utf-8")
        robot_freeze["run_directory"] = raw_freeze["run_directory"]
        robot_freeze["frozen_run_manifest_ref"] = file_reference(robot_run_path)
        robot_freeze_path.write_text(json.dumps(robot_freeze), encoding="utf-8")
        twins["ROBOT_RGB"]["freeze_manifest_ref"] = file_reference(robot_freeze_path)

    with pytest.raises(ValueError, match="alias run_directory_binding"):
        final_test_authorization(tmp_path, monkeypatch, mutate=alias_run)


def test_final_test_gate_rejects_selector_alias_and_paired_conflict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def alias_selector(gate, twins):
        raw_freeze_path = Path(twins["RAW"]["freeze_manifest_ref"]["path"])
        raw_freeze = json.loads(raw_freeze_path.read_text(encoding="utf-8"))
        robot_freeze_path = Path(twins["ROBOT_RGB"]["freeze_manifest_ref"]["path"])
        robot_freeze = json.loads(robot_freeze_path.read_text(encoding="utf-8"))
        robot_run_path = Path(robot_freeze["frozen_run_manifest_ref"]["path"])
        robot_run = json.loads(robot_run_path.read_text(encoding="utf-8"))
        robot_run["selector_manifest_ref"] = raw_freeze["selector_manifest_ref"]
        robot_run_path.write_text(json.dumps(robot_run), encoding="utf-8")
        robot_freeze["selector_manifest_ref"] = raw_freeze["selector_manifest_ref"]
        robot_freeze["frozen_run_manifest_ref"] = file_reference(robot_run_path)
        robot_freeze_path.write_text(json.dumps(robot_freeze), encoding="utf-8")
        twins["ROBOT_RGB"]["freeze_manifest_ref"] = file_reference(robot_freeze_path)

    with pytest.raises(ValueError, match="selector domain mismatch"):
        final_test_authorization(tmp_path, monkeypatch, mutate=alias_selector)

    second_root = tmp_path / "paired_conflict"
    second_root.mkdir()

    def conflict_paired(gate, twins):
        alternate = second_root / "PAIRED.json"
        alternate.write_text("{}", encoding="utf-8")
        alternate_ref = file_reference(alternate)
        robot_freeze_path = Path(twins["ROBOT_RGB"]["freeze_manifest_ref"]["path"])
        robot_freeze = json.loads(robot_freeze_path.read_text(encoding="utf-8"))
        robot_run_path = Path(robot_freeze["frozen_run_manifest_ref"]["path"])
        robot_run = json.loads(robot_run_path.read_text(encoding="utf-8"))
        robot_run["paired_kept_manifest_ref"] = alternate_ref
        robot_run_path.write_text(json.dumps(robot_run), encoding="utf-8")
        robot_freeze["paired_kept_manifest_ref"] = alternate_ref
        robot_freeze["frozen_run_manifest_ref"] = file_reference(robot_run_path)
        robot_freeze_path.write_text(json.dumps(robot_freeze), encoding="utf-8")
        twins["ROBOT_RGB"]["freeze_manifest_ref"] = file_reference(robot_freeze_path)

    with pytest.raises(ValueError, match="do not share one paired ledger"):
        final_test_authorization(second_root, monkeypatch, mutate=conflict_paired)


def test_final_test_claim_rejects_authorized_parent_replacement_without_external_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorization, _, _ = final_test_authorization(tmp_path, monkeypatch)
    parent = Path(authorization["output_directory"]).parent
    checked_parent = tmp_path / "checked_parent"
    parent.rename(checked_parent)
    external = tmp_path / "external"
    external.mkdir()
    parent.symlink_to(external, target_is_directory=True)

    with pytest.raises(ValueError, match="non-symlink directory"):
        source_contract.claim_final_test_output(authorization)
    assert not (external / "raw").exists()


def test_final_test_claim_rejects_post_mkdir_symlink_swap_without_external_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorization, _, _ = final_test_authorization(tmp_path, monkeypatch)
    external = tmp_path / "external"
    external.mkdir()
    real_mkdir = source_contract.os.mkdir

    def mkdir_then_swap(path, mode=0o777, *, dir_fd=None):
        real_mkdir(path, mode=mode, dir_fd=dir_fd)
        source_contract.os.rename(
            path,
            f"{path}.moved",
            src_dir_fd=dir_fd,
            dst_dir_fd=dir_fd,
        )
        source_contract.os.symlink(
            external,
            path,
            target_is_directory=True,
            dir_fd=dir_fd,
        )

    monkeypatch.setattr(source_contract.os, "mkdir", mkdir_then_swap)
    with pytest.raises(ValueError, match="changed after its atomic creation"):
        source_contract.claim_final_test_output(authorization)
    assert not (external / "FINAL_TEST_CLAIM.json").exists()


@pytest.mark.parametrize("preexisting", ("empty", "nonempty", "symlink"))
def test_final_test_claim_rejects_every_preexisting_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    preexisting: str,
) -> None:
    authorization, _, _ = final_test_authorization(tmp_path, monkeypatch)
    output = Path(authorization["output_directory"])
    if preexisting == "symlink":
        target = tmp_path / "external"
        target.mkdir()
        output.symlink_to(target, target_is_directory=True)
    else:
        output.mkdir()
        if preexisting == "nonempty":
            (output / "decoy").write_text("not a claim", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already claimed"):
        source_contract.claim_final_test_output(authorization)


def test_evaluation_entrypoints_expose_only_dev_or_gated_final_test() -> None:
    for entrypoint in (evaluate_h50.main, evaluate_rolling_h50.main):
        source = inspect.getsource(entrypoint)
        assert '"--evaluation-role"' in source
        assert '("dev", "final_test")' in source
        assert '"--final-test-gate"' in source
        assert '"--split-role"' not in source
    h50_source = inspect.getsource(evaluate_h50.main)
    assert h50_source.index("authorize_evaluation_role(") < h50_source.index(
        "claim_final_test_output("
    ) < h50_source.index("verify_eligible68_split_phase2(")
    rolling_validate = inspect.getsource(evaluate_rolling_h50.validate_checkpoint_inputs)
    rolling_main = inspect.getsource(evaluate_rolling_h50.main)
    assert "authorize_evaluation_role(" in rolling_validate
    assert rolling_main.index("claim_final_test_output(") < rolling_main.index(
        "verify_eligible68_split_phase2("
    )
    for source in (h50_source, rolling_main):
        assert source.index(
            'args.check_only and args.evaluation_role == "final_test"'
        ) < source.index("claim_final_test_output(")

    freeze_source = inspect.getsource(freeze_best_checkpoint.main)
    assert "frozen_source_bytes" in freeze_source
    assert "shutil.copyfile(source, temporary)" not in freeze_source

    for entrypoint in (
        train_embodiment.main,
        evaluate_training_snapshot.load_snapshot,
        freeze_best_checkpoint.main,
        FlowMatchingTrainer.main,
    ):
        assert "validate_training_run_role_contract(" in inspect.getsource(entrypoint)


def test_rolling_metadata_consumes_the_dataset_bound_payload(tmp_path: Path) -> None:
    metadata_path = tmp_path / "training_data.json"
    metadata_path.write_text(
        json.dumps({"metadata": {"c2w": "replacement"}}),
        encoding="utf-8",
    )

    class BoundDataset:
        def _read_frame(self, path: str) -> dict:
            assert path == str(metadata_path)
            return {"metadata": {"c2w": "accepted"}}

    assert evaluate_rolling_h50.read_bound_sample_metadata(
        BoundDataset(), metadata_path
    ) == {"c2w": "accepted"}
