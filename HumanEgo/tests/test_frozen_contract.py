from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path

import pytest
import torch
import utils.frozen_contract as frozen_contract

from utils.frozen_contract import (
    build_formal_run_manifest,
    canonical_json_sha256,
    ensure_runtime_checkpoint_authority,
    file_reference,
    load_frozen_checkpoint_payload,
    load_frozen_split,
    load_verified_torch_checkpoint,
    read_runtime_checkpoint_authority,
    sessions_for_role,
    validate_checkpoint_bundle,
    validate_file_reference,
    validate_formal_runtime_contract,
    validate_resolved_config,
    validate_run_root_binding,
    validate_run_manifest,
    validate_session_paths,
    write_runtime_checkpoint_authority,
)


def build_split(tmp_path: Path) -> tuple[Path, Path, dict]:
    root = tmp_path / "sidecars"
    roles = {
        "train": ["grap_a_cap_001"],
        "validation": ["grap_a_cap_002"],
        "test": ["grap_a_cap_003"],
    }
    sessions = {}
    for session_id in sum(roles.values(), []):
        path = root / "kai22" / session_id / "sidecar.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(session_id.encode())
        sessions[session_id] = {
            "embodiments": {
                "kai22": {
                    "sidecar_sha256": hashlib.sha256(path.read_bytes()).hexdigest()
                }
            }
        }
    split = {**roles, "sessions": sessions, "sidecar_root": str(root)}
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split), encoding="utf-8")
    return split_path, root, split


def build_formal_fixture(
    tmp_path: Path,
) -> tuple[dict, dict, Path, Path, Path]:
    split_path, sidecar_root, split = build_split(tmp_path)
    production_root = tmp_path / "production"
    for session_id in split["train"] + split["validation"] + split["test"]:
        adapter = production_root / session_id / "09_humanego_adapter"
        adapter.mkdir(parents=True)
        split["sessions"][session_id]["adapter"] = str(adapter.resolve())
    split["production_root"] = str(production_root.resolve())
    split["sidecar_root"] = str(sidecar_root.resolve())
    split_path.write_text(json.dumps(split), encoding="utf-8")
    config_root = tmp_path / "cfg"
    config_root.mkdir()
    config_path = config_root / "reviewed.yaml"
    config_path.write_text("seed: 7\nepochs: 10\n", encoding="utf-8")
    run_root = tmp_path / "runs"
    run_directory = run_root / "training_fixture"
    run_directory.mkdir(parents=True)
    resolved_config = {
        "seed": 7,
        "epochs": 10,
        "img_name": "rgb.png",
        "run_root": str(run_root.resolve()),
        "out_dir": str(run_directory.resolve()),
    }
    loaded = load_frozen_split(split_path, "kai22", sidecar_root=sidecar_root)
    manifest = build_formal_run_manifest(
        split=loaded,
        split_path=split_path,
        config_path=config_path,
        resolved_config=resolved_config,
        embodiment="kai22",
        sidecar_root=sidecar_root,
        production_root=production_root,
        run_root=run_root,
        run_directory=run_directory,
        base_fields={"scratch": False},
    )
    return manifest, loaded, split_path, config_root, sidecar_root


def test_frozen_split_verifies_complete_partition_and_hashes(tmp_path: Path) -> None:
    split_path, root, _ = build_split(tmp_path)
    loaded = load_frozen_split(
        split_path, "kai22", sidecar_root=root, verify_sidecars=True
    )
    assert sessions_for_role(loaded, "test") == ["grap_a_cap_003"]


def test_empty_test_requires_explicit_no_repurpose_policy(tmp_path: Path) -> None:
    split_path, root, split = build_split(tmp_path)
    split["test"] = []
    split["sessions"].pop("grap_a_cap_003")
    split_path.write_text(json.dumps(split), encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty"):
        load_frozen_split(split_path, "kai22", sidecar_root=root)

    split["independent_test_policy"] = "UNAVAILABLE_NOT_REPURPOSED"
    split_path.write_text(json.dumps(split), encoding="utf-8")
    loaded = load_frozen_split(
        split_path, "kai22", sidecar_root=root, verify_sidecars=True
    )
    assert sessions_for_role(loaded, "test") == []


def test_frozen_split_rejects_leakage_and_wrong_root(tmp_path: Path) -> None:
    split_path, root, split = build_split(tmp_path)
    split["validation"] = list(split["train"])
    split_path.write_text(json.dumps(split), encoding="utf-8")
    with pytest.raises(ValueError, match="leakage"):
        load_frozen_split(split_path, "kai22", sidecar_root=root)

    _, _, split = build_split(tmp_path)
    split_path.write_text(json.dumps(split), encoding="utf-8")
    with pytest.raises(ValueError, match="sidecar root differs"):
        load_frozen_split(split_path, "kai22", sidecar_root=tmp_path / "wrong")


def test_frozen_split_rejects_hash_drift(tmp_path: Path) -> None:
    split_path, root, _ = build_split(tmp_path)
    (root / "kai22/grap_a_cap_002/sidecar.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_frozen_split(
            split_path, "kai22", sidecar_root=root, verify_sidecars=True
        )


def test_frozen_split_rejects_symlinked_sidecar_root(tmp_path: Path) -> None:
    split_path, root, split = build_split(tmp_path)
    linked_root = tmp_path / "linked_sidecars"
    linked_root.symlink_to(root, target_is_directory=True)
    split["sidecar_root"] = str(linked_root)
    split_path.write_text(json.dumps(split), encoding="utf-8")

    with pytest.raises(ValueError, match="symlink component"):
        load_frozen_split(
            split_path,
            "kai22",
            sidecar_root=linked_root,
            verify_sidecars=True,
        )


def test_run_manifest_requires_all_three_frozen_roles(tmp_path: Path) -> None:
    split_path, root, split = build_split(tmp_path)
    loaded = load_frozen_split(split_path, "kai22", sidecar_root=root)
    hashes = {
        session: metadata["embodiments"]["kai22"]["sidecar_sha256"]
        for session, metadata in split["sessions"].items()
    }
    manifest = {
        "embodiment": "kai22",
        "train_sessions": split["train"],
        "validation_sessions": split["validation"],
        "sidecars": {key: hashes[key] for key in split["train"] + split["validation"]},
        "evaluation_sidecars": {key: hashes[key] for key in split["test"]},
    }
    validate_run_manifest(manifest, loaded, "kai22")
    manifest["evaluation_sidecars"] = {}
    with pytest.raises(ValueError, match="test sidecar lineage"):
        validate_run_manifest(manifest, loaded, "kai22")


def test_session_paths_cannot_change_manifest_order_or_location(tmp_path: Path) -> None:
    paths = []
    for session_id in ("grap_a_cap_001", "grap_a_cap_002"):
        path = tmp_path / session_id / "09_humanego_adapter"
        path.mkdir(parents=True)
        paths.append(path)
    validate_session_paths(paths, ["grap_a_cap_001", "grap_a_cap_002"])
    with pytest.raises(ValueError, match="differ from run manifest"):
        validate_session_paths(paths[::-1], ["grap_a_cap_001", "grap_a_cap_002"])

    other = tmp_path / "unreviewed/grap_a_cap_001/09_humanego_adapter"
    other.mkdir(parents=True)
    with pytest.raises(ValueError, match="frozen manifest paths"):
        validate_session_paths(
            [other],
            ["grap_a_cap_001"],
            expected_paths={"grap_a_cap_001": paths[0]},
        )


def test_file_reference_rejects_hash_root_and_symlink_escape(tmp_path: Path) -> None:
    approved = tmp_path / "approved"
    approved.mkdir()
    path = approved / "config.yaml"
    path.write_text("seed: 7\n", encoding="utf-8")
    reference = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    assert validate_file_reference(
        reference, allowed_roots=[approved], label="config"
    ) == path
    changed = dict(reference, sha256="0" * 64)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        validate_file_reference(changed, allowed_roots=[approved], label="config")
    other_root = tmp_path / "other"
    other_root.mkdir()
    with pytest.raises(ValueError, match="escapes approved roots"):
        validate_file_reference(reference, allowed_roots=[other_root], label="config")

    link = approved / "link.yaml"
    link.symlink_to(path)
    linked_reference = dict(reference, path=str(link.absolute()))
    with pytest.raises(ValueError, match="symlink component"):
        validate_file_reference(
            linked_reference, allowed_roots=[approved], label="config"
        )


def test_unapproved_reference_is_rejected_before_its_bytes_are_opened(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    approved = tmp_path / "approved"
    outside = tmp_path / "outside"
    approved.mkdir()
    outside.mkdir()
    target = outside / "manifest.json"
    target.write_text("{}", encoding="utf-8")
    reference = {
        "path": str(target.resolve()),
        "bytes": target.stat().st_size,
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }
    opened = 0
    original_open = frozen_contract.os.open

    def spy_open(path, *args, **kwargs):
        nonlocal opened
        if Path(path) == target:
            opened += 1
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(frozen_contract.os, "open", spy_open)
    with pytest.raises(ValueError, match="escapes approved roots"):
        validate_file_reference(reference, allowed_roots=[approved], label="manifest")
    assert opened == 0


def test_resolved_config_is_bound_after_cli_overrides(tmp_path: Path) -> None:
    config_root = tmp_path / "cfg"
    config_root.mkdir()
    config_path = config_root / "reviewed.yaml"
    config_path.write_text("seed: 7\nepochs: 10\n", encoding="utf-8")
    resolved = {"seed": 7, "epochs": 10, "augmentation": True}
    manifest = {
        "config_ref": {
            "path": str(config_path.resolve()),
            "bytes": config_path.stat().st_size,
            "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        },
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "resolved_config": resolved,
        "resolved_config_sha256": canonical_json_sha256(resolved),
    }
    assert validate_resolved_config(
        resolved, manifest, config_root=config_root
    ) == config_path
    with pytest.raises(ValueError, match="effective Trainer config differs"):
        validate_resolved_config(
            {**resolved, "seed": 8}, manifest, config_root=config_root
        )


def test_launcher_v2_manifest_passes_trainer_runtime_contract(tmp_path: Path) -> None:
    manifest, loaded, _, config_root, _ = build_formal_fixture(tmp_path)
    assert {
        "schema_version",
        "split_ref",
        "config_ref",
        "resolved_config",
        "resolved_config_sha256",
        "sidecar_refs",
        "adapter_paths",
        "run_root",
        "run_directory",
    }.issubset(manifest)
    assert manifest["schema_version"] == "humanego-formal-run-v2"
    validate_run_manifest(
        manifest, loaded, "kai22", require_runtime_fields=True
    )
    validate_formal_runtime_contract(
        manifest["resolved_config"],
        manifest,
        loaded,
        "kai22",
        config_root=config_root,
    )
    manifest["sidecars"] = {}
    with pytest.raises(ValueError, match="training sidecar lineage"):
        validate_run_manifest(
            manifest, loaded, "kai22", require_runtime_fields=True
        )


def test_runtime_manifest_rejects_sidecar_reference_path_or_bytes_drift(
    tmp_path: Path,
) -> None:
    manifest, loaded, _, _, _ = build_formal_fixture(tmp_path)
    manifest["sidecar_refs"]["grap_a_cap_001"]["bytes"] += 1
    with pytest.raises(ValueError, match="byte-size mismatch"):
        validate_run_manifest(
            manifest, loaded, "kai22", require_runtime_fields=True
        )


def test_formal_run_root_is_explicit_exact_and_not_symlinked(tmp_path: Path) -> None:
    manifest, _, _, _, _ = build_formal_fixture(tmp_path)
    run_root = Path(manifest["run_root"])
    run_directory = Path(manifest["run_directory"])
    assert validate_run_root_binding(
        run_directory, manifest, expected_run_root=run_root
    ) == (run_root, run_directory)

    other_root = tmp_path / "other_runs"
    other_root.mkdir()
    with pytest.raises(ValueError, match="caller-approved root"):
        validate_run_root_binding(
            run_directory, manifest, expected_run_root=other_root
        )

    other_directory = run_root / "training_other"
    other_directory.mkdir()
    with pytest.raises(ValueError, match="differs from the run manifest"):
        validate_run_root_binding(
            other_directory,
            manifest,
            expected_run_root=run_root,
        )

    nested_directory = run_directory / "nested"
    nested_directory.mkdir()
    with pytest.raises(ValueError, match="immediate child"):
        validate_run_root_binding(
            nested_directory,
            {**manifest, "run_directory": str(nested_directory)},
            expected_run_root=run_root,
        )

    alias = tmp_path / "run_root_alias"
    alias.symlink_to(run_root, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink component"):
        validate_run_root_binding(
            run_directory,
            {**manifest, "run_root": str(alias)},
            expected_run_root=alias,
        )


def test_checkpoint_bundle_binds_checkpoint_split_stats_and_manifest(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best.pt"
    split = tmp_path / "split.json"
    stats = tmp_path / "dataset_stats.json"
    run_manifest = tmp_path / "run_manifest.json"
    checkpoint.write_bytes(b"checkpoint")
    split.write_text("{}", encoding="utf-8")
    stats.write_text("{}", encoding="utf-8")
    manifest_value = {"embodiment": "kai22"}
    run_manifest.write_text(json.dumps(manifest_value), encoding="utf-8")
    stats_digest = hashlib.sha256(stats.read_bytes()).hexdigest()
    payload = {
        "run_manifest": manifest_value,
        "dataset_stats_sha256": stats_digest,
    }
    freeze = {
        "status": "frozen",
        "embodiment": "kai22",
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "split_sha256": hashlib.sha256(split.read_bytes()).hexdigest(),
        "dataset_stats_sha256": stats_digest,
        "run_manifest_sha256": hashlib.sha256(run_manifest.read_bytes()).hexdigest(),
    }
    (tmp_path / "freeze_manifest.json").write_text(json.dumps(freeze), encoding="utf-8")
    assert validate_checkpoint_bundle(
        checkpoint, payload, split, "kai22"
    ) == stats
    stats_path, stats_payload = validate_checkpoint_bundle(
        checkpoint,
        payload,
        split,
        "kai22",
        return_dataset_stats=True,
    )
    assert stats_path == stats
    assert stats_payload == {}
    with pytest.raises(ValueError, match="payload/dataset statistics"):
        validate_checkpoint_bundle(
            checkpoint,
            {**payload, "dataset_stats_sha256": "0" * 64},
            split,
            "kai22",
        )
    stats.write_text('{"changed": true}', encoding="utf-8")
    with pytest.raises(ValueError, match="dataset_stats_sha256 mismatch"):
        validate_checkpoint_bundle(
            checkpoint, payload, split, "kai22"
        )


def test_checkpoint_payload_rejects_stats_bundle_replaced_after_model_load(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "best.pt"
    split = tmp_path / "split.json"
    stats = tmp_path / "dataset_stats.json"
    run_manifest_path = tmp_path / "run_manifest.json"
    split.write_text("{}", encoding="utf-8")
    stats.write_text('{"mean": 1}', encoding="utf-8")
    run_manifest = {"embodiment": "kai22"}
    run_manifest_path.write_text(json.dumps(run_manifest), encoding="utf-8")
    accepted_stats_digest = hashlib.sha256(stats.read_bytes()).hexdigest()
    torch.save(
        {
            "model": {"weight": torch.tensor([1.0])},
            "run_manifest": run_manifest,
            "dataset_stats_sha256": accepted_stats_digest,
        },
        checkpoint,
    )
    freeze = {
        "schema_version": "humanego-frozen-best-v2",
        "immutable": True,
        "no_fallback": True,
        "status": "frozen",
        "embodiment": "kai22",
        "checkpoint_ref": file_reference(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "split_sha256": hashlib.sha256(split.read_bytes()).hexdigest(),
        "run_manifest_sha256": hashlib.sha256(
            run_manifest_path.read_bytes()
        ).hexdigest(),
        "dataset_stats_sha256": accepted_stats_digest,
    }
    freeze_path = tmp_path / "freeze_manifest.json"
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")

    loaded, _, _ = load_frozen_checkpoint_payload(checkpoint, embodiment="kai22")
    stats.write_text('{"mean": 2}', encoding="utf-8")
    replacement_digest = hashlib.sha256(stats.read_bytes()).hexdigest()
    freeze["dataset_stats_sha256"] = replacement_digest
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")

    with pytest.raises(ValueError, match="payload/dataset statistics"):
        validate_checkpoint_bundle(checkpoint, loaded, split, "kai22")


def test_verified_torch_loader_uses_frozen_bytes_and_weights_only(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"value": torch.tensor([1, 2, 3])}, checkpoint)
    reference = file_reference(checkpoint)
    path, payload = load_verified_torch_checkpoint(
        reference,
        allowed_roots=[tmp_path],
        label="test checkpoint",
    )
    assert path == checkpoint
    assert torch.equal(payload["value"], torch.tensor([1, 2, 3]))


@pytest.mark.parametrize("replacement", ("symlink", "hardlink"))
def test_verified_torch_loader_rejects_links_before_deserialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement: str,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"value": torch.tensor([1])}, checkpoint)
    reference = file_reference(checkpoint)
    alias = tmp_path / "alias.pt"
    if replacement == "symlink":
        alias.symlink_to(checkpoint)
        expected = "symlink component"
    else:
        alias.hardlink_to(checkpoint)
        expected = "singly-linked"
    alias_reference = {**reference, "path": str(alias)}
    calls = 0

    def forbidden_deserialize(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("torch.load must not run")

    monkeypatch.setattr(torch, "load", forbidden_deserialize)
    with pytest.raises(ValueError, match=expected):
        load_verified_torch_checkpoint(
            alias_reference,
            allowed_roots=[tmp_path],
            label="linked checkpoint",
        )
    assert calls == 0


@pytest.mark.parametrize("replacement", ("path", "bytes"))
def test_verified_torch_loader_detects_post_verify_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement: str,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"value": torch.tensor([1])}, checkpoint)
    reference = file_reference(checkpoint)
    real_torch_load = torch.load

    def deserialize_then_replace(stream, *args, **kwargs):
        payload = real_torch_load(stream, *args, **kwargs)
        if replacement == "path":
            checkpoint.rename(tmp_path / "moved.pt")
            torch.save({"value": torch.tensor([2])}, checkpoint)
        else:
            with checkpoint.open("r+b") as output:
                output.seek(0)
                output.write(b"X")
                output.flush()
                os.fsync(output.fileno())
        return payload

    monkeypatch.setattr(torch, "load", deserialize_then_replace)
    with pytest.raises(ValueError, match="changed during consumption"):
        load_verified_torch_checkpoint(
            reference,
            allowed_roots=[tmp_path],
            label="replaced checkpoint",
        )


def test_verified_torch_loader_detects_parent_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    approved = tmp_path / "approved"
    approved.mkdir()
    checkpoint = approved / "checkpoint.pt"
    torch.save({"value": torch.tensor([1])}, checkpoint)
    reference = file_reference(checkpoint)
    moved = tmp_path / "approved.moved"
    external = tmp_path / "external"
    external.mkdir()
    torch.save({"value": torch.tensor([2])}, external / checkpoint.name)
    real_torch_load = torch.load

    def deserialize_then_replace_parent(stream, *args, **kwargs):
        payload = real_torch_load(stream, *args, **kwargs)
        assert torch.equal(payload["value"], torch.tensor([1]))
        approved.rename(moved)
        approved.symlink_to(external, target_is_directory=True)
        return payload

    monkeypatch.setattr(torch, "load", deserialize_then_replace_parent)
    with pytest.raises(ValueError, match="symlink component|changed during consumption"):
        load_verified_torch_checkpoint(
            reference,
            allowed_roots=[approved],
            label="parent-replaced checkpoint",
        )


def test_runtime_checkpoint_authority_is_exclusive_and_complete(tmp_path: Path) -> None:
    checkpoint = tmp_path / "pending.pt"
    run_manifest = tmp_path / "run_manifest.json"
    stats = tmp_path / "dataset_stats.json"
    torch.save({"value": torch.tensor([1])}, checkpoint)
    run_manifest.write_text('{"run": 1}', encoding="utf-8")
    stats.write_text('{"stats": 1}', encoding="utf-8")
    authority_path = write_runtime_checkpoint_authority(
        checkpoint,
        run_manifest_reference=file_reference(run_manifest),
        dataset_stats_reference=file_reference(stats),
    )
    authority = read_runtime_checkpoint_authority(checkpoint)
    assert authority_path.is_file()
    assert authority["checkpoint_ref"] == file_reference(checkpoint)
    with pytest.raises(FileExistsError):
        write_runtime_checkpoint_authority(
            checkpoint,
            run_manifest_reference=file_reference(run_manifest),
            dataset_stats_reference=file_reference(stats),
        )


def test_runtime_checkpoint_authority_exactly_recovers_after_partial_completion(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "best.pt"
    run_manifest = tmp_path / "run_manifest.json"
    stats = tmp_path / "dataset_stats.json"
    torch.save({"model": {"weight": torch.tensor([1.0])}}, checkpoint)
    run_manifest.write_text('{"run":1}\n', encoding="utf-8")
    stats.write_text('{"stats":1}\n', encoding="utf-8")
    run_reference = file_reference(run_manifest)
    stats_reference = file_reference(stats)

    first = ensure_runtime_checkpoint_authority(
        checkpoint,
        run_manifest_reference=run_reference,
        dataset_stats_reference=stats_reference,
    )
    before = first.read_bytes()
    second = ensure_runtime_checkpoint_authority(
        checkpoint,
        run_manifest_reference=run_reference,
        dataset_stats_reference=stats_reference,
    )
    assert second == first
    assert second.read_bytes() == before

    checkpoint.unlink()
    torch.save({"model": {"weight": torch.tensor([2.0])}}, checkpoint)
    with pytest.raises(ValueError, match="stale checkpoint bytes"):
        ensure_runtime_checkpoint_authority(
            checkpoint,
            run_manifest_reference=run_reference,
            dataset_stats_reference=stats_reference,
        )


@pytest.mark.parametrize("crash_state", ("staging_only", "linked_before_cleanup"))
def test_runtime_checkpoint_authority_recovers_exact_transaction_states(
    tmp_path: Path,
    crash_state: str,
) -> None:
    checkpoint = tmp_path / "best.pt"
    run_manifest = tmp_path / "run_manifest.json"
    stats = tmp_path / "dataset_stats.json"
    torch.save({"model": {"weight": torch.tensor([1.0])}}, checkpoint)
    run_manifest.write_text('{"run":1}\n', encoding="utf-8")
    stats.write_text('{"stats":1}\n', encoding="utf-8")
    run_reference = file_reference(run_manifest)
    stats_reference = file_reference(stats)
    authority = write_runtime_checkpoint_authority(
        checkpoint,
        run_manifest_reference=run_reference,
        dataset_stats_reference=stats_reference,
    )
    accepted = authority.read_bytes()
    staging = authority.with_name(f".{authority.name}.staging")
    if crash_state == "staging_only":
        authority.rename(staging)
    else:
        staging.hardlink_to(authority)
        assert authority.stat().st_nlink == 2

    recovered = ensure_runtime_checkpoint_authority(
        checkpoint,
        run_manifest_reference=run_reference,
        dataset_stats_reference=stats_reference,
    )

    assert recovered == authority
    assert recovered.read_bytes() == accepted
    assert recovered.stat().st_nlink == 1
    assert not staging.exists()


def test_isolated_metrics_parse_the_same_fd_snapshot_and_reject_schema_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics_path = tmp_path / "metrics.json"
    accepted = {
        key: (3 if key in {"batches", "frames"} else 0.25)
        for key in frozen_contract.ISOLATED_EVALUATION_METRIC_KEYS
    }
    replacement = {**accepted, "frames": 99}
    metrics_path.write_text(json.dumps(accepted), encoding="utf-8")
    original_reader = frozen_contract._read_ordinary_file_bytes

    def replace_after_verified_read(path, *, label):
        resolved, encoded = original_reader(path, label=label)
        metrics_path.write_text(json.dumps(replacement), encoding="utf-8")
        return resolved, encoded

    monkeypatch.setattr(
        frozen_contract,
        "_read_ordinary_file_bytes",
        replace_after_verified_read,
    )
    observed = frozen_contract.read_isolated_evaluation_metrics(
        metrics_path,
        label="synthetic isolated metrics",
    )
    assert observed == accepted

    monkeypatch.setattr(
        frozen_contract,
        "_read_ordinary_file_bytes",
        original_reader,
    )
    metrics_path.write_text(
        json.dumps({key: value for key, value in accepted.items() if key != "done_acc"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="metric schema drift"):
        frozen_contract.read_isolated_evaluation_metrics(
            metrics_path,
            label="synthetic isolated metrics",
        )
    metrics_path.write_text(
        json.dumps({**accepted, "done_acc": float("nan")}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must be finite"):
        frozen_contract.read_isolated_evaluation_metrics(
            metrics_path,
            label="synthetic isolated metrics",
        )


def test_runtime_checkpoint_authority_rejects_parent_swap_without_external_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    approved = tmp_path / "approved"
    approved.mkdir()
    checkpoint = approved / "pending.pt"
    run_manifest = approved / "run_manifest.json"
    stats = approved / "dataset_stats.json"
    torch.save({"value": torch.tensor([1])}, checkpoint)
    run_manifest.write_text('{"run": 1}', encoding="utf-8")
    stats.write_text('{"stats": 1}', encoding="utf-8")
    moved = tmp_path / "approved.moved"
    external = tmp_path / "external"
    external.mkdir()
    real_open_verified = frozen_contract.open_verified_file_reference

    @contextmanager
    def swap_parent_after_checkpoint_verification(*args, **kwargs):
        with real_open_verified(*args, **kwargs) as opened:
            approved.rename(moved)
            approved.symlink_to(external, target_is_directory=True)
            yield opened

    monkeypatch.setattr(
        frozen_contract,
        "open_verified_file_reference",
        swap_parent_after_checkpoint_verification,
    )
    authority_name = f"{checkpoint.name}.authority.json"
    with pytest.raises(ValueError, match="symlink component|path changed"):
        write_runtime_checkpoint_authority(
            checkpoint,
            run_manifest_reference=file_reference(run_manifest),
            dataset_stats_reference=file_reference(stats),
        )
    assert not (external / authority_name).exists()
    assert not (moved / authority_name).exists()


def test_frozen_checkpoint_authority_precedes_deserialization(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best.pt"
    torch.save({"value": torch.tensor([1])}, checkpoint)
    reference = file_reference(checkpoint)
    freeze = {
        "schema_version": "humanego-frozen-best-v2",
        "immutable": True,
        "no_fallback": True,
        "status": "frozen",
        "embodiment": "kai22",
        "checkpoint_ref": reference,
    }
    (tmp_path / "freeze_manifest.json").write_text(
        json.dumps(freeze),
        encoding="utf-8",
    )
    payload, observed_reference, observed_freeze = load_frozen_checkpoint_payload(
        checkpoint,
        embodiment="kai22",
    )
    assert torch.equal(payload["value"], torch.tensor([1]))
    assert observed_reference == reference
    assert observed_freeze == freeze
