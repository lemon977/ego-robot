from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest

from tools import publish_matched_selector_manifests as publisher
from utils import source_contract


SESSION_ID = "grap_a_cap_004"
EMBODIMENT = "kai22"


def file_ref(path: Path) -> dict[str, object]:
    encoded = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


@dataclass
class Fixture:
    project_root: Path
    split_ref: dict[str, object]
    raw_ref: dict[str, object]
    robot_ref: dict[str, object]
    raw_payload: dict
    robot_payload: dict
    output_root: Path
    raw_image_paths: list[Path]
    metadata_paths: list[Path]


def _small_contract(monkeypatch: pytest.MonkeyPatch, frame_count: int) -> None:
    monkeypatch.setattr(source_contract, "ELIGIBLE68", frozenset({SESSION_ID}))
    monkeypatch.setattr(source_contract, "ELIGIBLE68_ORDER", (SESSION_ID,))
    monkeypatch.setattr(
        source_contract, "ELIGIBLE68_FRAME_COUNTS", {SESSION_ID: frame_count}
    )
    monkeypatch.setattr(source_contract, "TRAIN52", (SESSION_ID,))
    monkeypatch.setattr(source_contract, "DEV8", ())
    monkeypatch.setattr(source_contract, "FINAL_TEST8", ())


def _write_robot_authority(
    project_root: Path,
    selector_frames: dict[str, dict[str, object]],
) -> dict[str, object]:
    lineage_root = project_root / "formal_robot_lineage"
    evidence_root = lineage_root / "evidence"
    evidence_root.mkdir(parents=True)
    stage_refs: dict[str, dict[str, object]] = {}
    for stage in source_contract.ROBOT_RGB_STAGE_REFERENCE_FIELDS:
        payload: dict[str, object] = {
            "schema_version": source_contract.ROBOT_RGB_STAGE_SCHEMA,
            "lineage_stage": stage,
            **source_contract._FORMAL_ROBOT_RGB_STATE,
            "session_id": SESSION_ID,
            "frame_count": len(selector_frames),
        }
        payload.update(
            {
                source_contract.ROBOT_RGB_STAGE_REFERENCE_FIELDS[upstream]: dict(
                    stage_refs[upstream]
                )
                for upstream in source_contract.ROBOT_RGB_STAGE_UPSTREAM_STAGES[stage]
            }
        )
        if stage == "S8":
            outputs = {
                frame_key: dict(record["image"])
                for frame_key, record in selector_frames.items()
            }
            payload.update(
                {
                    "s8_authority": "FORMAL_S8_ROBOT_RGB",
                    "robot_rgb_outputs": outputs,
                    "success_evidence": list(outputs.values()),
                }
            )
        else:
            evidence = evidence_root / f"{stage.lower()}.bin"
            evidence.write_bytes(f"formal-{stage}".encode())
            payload["success_evidence"] = [file_ref(evidence)]
            if stage == "SCENE_STATE":
                payload.update(
                    {
                        "scene_state_mode": "EXTERNAL_FORMAL_AUTHORITY",
                        "q_arm_and_camera_base_authoritative": True,
                    }
                )
            elif stage == "MOUNT_AUTHORITY":
                payload.update(
                    {
                        "mount_provenance": "INDEPENDENT_BILATERAL_FORMAL_MOUNT",
                        "independent_measurement": True,
                        "selected_by_ik_residual": False,
                    }
                )
            elif stage == "EEVEE_RENDER":
                payload["engine"] = "BLENDER_EEVEE_NEXT"
            elif stage == "CLEAN":
                payload["clean_authority"] = "FORMAL_CLEAN"
        stage_path = lineage_root / f"{stage.lower()}.json"
        stage_path.write_text(json.dumps(payload), encoding="utf-8")
        stage_refs[stage] = file_ref(stage_path)

    session_row = {
        "frame_count": len(selector_frames),
        **{
            field: stage_refs[stage]
            for stage, field in source_contract.ROBOT_RGB_STAGE_REFERENCE_FIELDS.items()
        },
    }
    authority = {
        "schema_version": source_contract.ROBOT_RGB_LINEAGE_SCHEMA,
        **source_contract._FORMAL_ROBOT_RGB_STATE,
        "cohort": "eligible68",
        "session_count": 1,
        "frame_count": len(selector_frames),
        "success_evidence": [
            session_row[field]
            for field in source_contract.ROBOT_RGB_STAGE_REFERENCE_FIELDS.values()
        ],
        "sessions": {SESSION_ID: session_row},
    }
    authority_path = lineage_root / "ROBOT_RGB_LINEAGE_AUTHORITY.json"
    authority_path.write_text(json.dumps(authority), encoding="utf-8")
    return file_ref(authority_path)


def make_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    frame_count: int = 52,
    reviewed: bool = True,
) -> Fixture:
    _small_contract(monkeypatch, frame_count)
    project_root = tmp_path / "project"
    split_root = project_root / "HumanEgo/data_manifests"
    production_root = project_root / "production"
    sidecar_root = project_root / "sidecars"
    raw_root = project_root / "raw_artifacts"
    robot_root = project_root / "robot_artifacts"
    incoming = project_root / "incoming"
    published = project_root / "published"
    for path in (
        split_root,
        production_root,
        sidecar_root,
        raw_root,
        robot_root,
        incoming,
        published,
    ):
        path.mkdir(parents=True, exist_ok=True)

    split = {
        "schema_version": source_contract.ELIGIBLE68_SPLIT_SCHEMA,
        "immutable": True,
        "no_fallback": True,
        **source_contract.eligible68_split_protocol_fields(),
        "train": [SESSION_ID],
        "validation": [],
        "test": [],
        "sessions": {
            SESSION_ID: {
                "adapter": str(production_root / SESSION_ID / "09_humanego_adapter"),
                "production_status_sha256": "0" * 64,
                "embodiments": {
                    EMBODIMENT: {
                        "frames": frame_count,
                        "sidecar_sha256": "1" * 64,
                    }
                },
            }
        },
        "production_root": str(production_root),
        "sidecar_root": str(sidecar_root),
    }
    split_path = split_root / "eligible68_synthetic.json"
    split_path.write_text(json.dumps(split), encoding="utf-8")

    raw_frames: dict[str, dict[str, object]] = {}
    robot_frames: dict[str, dict[str, object]] = {}
    raw_images: list[Path] = []
    metadata_paths: list[Path] = []
    for frame_index in range(frame_count):
        frame_key = f"{frame_index:05d}"
        metadata = (
            production_root
            / SESSION_ID
            / "09_humanego_adapter/preprocess/all_data"
            / frame_key
            / "training_data.json"
        )
        metadata.parent.mkdir(parents=True)
        metadata.write_text(
            json.dumps({"synthetic_frame": frame_index}), encoding="utf-8"
        )
        raw_image = raw_root / SESSION_ID / frame_key / "rgb.png"
        raw_image.parent.mkdir(parents=True)
        raw_image.write_bytes(f"raw-domain-{frame_key}".encode())
        robot_image = robot_root / SESSION_ID / frame_key / "robot.png"
        robot_image.parent.mkdir(parents=True)
        robot_image.write_bytes(f"robot-domain-{frame_key}".encode())
        raw_record = {
            "metadata": file_ref(metadata),
            "image": file_ref(raw_image),
            "unresolved": frame_index % 3 == 0,
        }
        robot_record = {
            "metadata": file_ref(metadata),
            "image": file_ref(robot_image),
            # Formal S8 requires RobotRGB outputs to be resolved.  RAW remains
            # deliberately asymmetric so the paired ledger proves unresolved
            # is retained rather than used as a one-sided deletion signal.
            "unresolved": False,
        }
        raw_frames[frame_key] = raw_record
        robot_frames[frame_key] = robot_record
        raw_images.append(raw_image)
        metadata_paths.append(metadata)

    authority_ref = _write_robot_authority(project_root, robot_frames)
    if reviewed:
        monkeypatch.setattr(
            source_contract,
            "REVIEWED_ROBOT_RGB_LINEAGE_AUTHORITY_REF",
            dict(authority_ref),
        )
    else:
        monkeypatch.setattr(
            source_contract, "REVIEWED_ROBOT_RGB_LINEAGE_AUTHORITY_REF", None
        )

    raw_payload = {
        "schema_version": source_contract.SELECTOR_SCHEMA,
        "immutable": True,
        "no_fallback": True,
        "product_line": "RAW",
        "image_name": "rgb.png",
        "artifact_root": str(raw_root),
        "selector_root": str(raw_root),
        "sessions": {SESSION_ID: {"frames": raw_frames}},
    }
    robot_payload = {
        "schema_version": source_contract.SELECTOR_SCHEMA,
        "immutable": True,
        "no_fallback": True,
        "product_line": "ROBOT_RGB",
        "image_name": "robot.png",
        "artifact_root": str(robot_root),
        "selector_root": str(robot_root),
        "robot_rgb_lineage_authority_ref": dict(authority_ref),
        "sessions": {SESSION_ID: {"frames": robot_frames}},
    }
    raw_path = incoming / "RAW_SELECTOR_RECORDS.json"
    robot_path = incoming / "ROBOT_SELECTOR_RECORDS.json"
    raw_path.write_text(json.dumps(raw_payload), encoding="utf-8")
    robot_path.write_text(json.dumps(robot_payload), encoding="utf-8")
    return Fixture(
        project_root=project_root,
        split_ref=file_ref(split_path),
        raw_ref=file_ref(raw_path),
        robot_ref=file_ref(robot_path),
        raw_payload=raw_payload,
        robot_payload=robot_payload,
        output_root=published / "matched_twins_v1",
        raw_image_paths=raw_images,
        metadata_paths=metadata_paths,
    )


def run_publish(fixture: Fixture) -> dict:
    return publisher.publish_matched_selector_manifests(
        split_reference=fixture.split_ref,
        raw_selector_records_reference=fixture.raw_ref,
        robot_selector_records_reference=fixture.robot_ref,
        output_root=fixture.output_root,
        embodiment=EMBODIMENT,
        project_root=fixture.project_root,
    )


def test_publisher_keeps_unresolved_twins_and_symmetric_h50(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch)
    report = run_publish(fixture)

    assert report["status"] == "ARTIFACT_EXISTS"
    assert report["schema_scope"] == "TWIN_ONLY_V1_RAW_AND_ROBOT_RGB"
    assert report["third_arm_defined"] is False
    assert report["product_lines"] == ["RAW", "ROBOT_RGB"]
    assert report["frame_count"] == 52
    assert stat.S_IMODE(fixture.output_root.stat().st_mode) == 0o555
    assert {path.name for path in fixture.output_root.iterdir()} == {
        publisher.RAW_NAME,
        publisher.ROBOT_NAME,
        publisher.PAIRED_NAME,
        publisher.PAIRED_PRIVATE_NAME,
    }

    raw = json.loads((fixture.output_root / publisher.RAW_NAME).read_text())
    robot = json.loads((fixture.output_root / publisher.ROBOT_NAME).read_text())
    paired = json.loads((fixture.output_root / publisher.PAIRED_NAME).read_text())
    alias = fixture.output_root / publisher.PAIRED_PRIVATE_NAME
    terminal = fixture.output_root / publisher.PAIRED_NAME
    assert raw == fixture.raw_payload
    assert robot == fixture.robot_payload
    assert raw["sessions"][SESSION_ID]["frames"]["00000"]["unresolved"] is True
    assert robot["sessions"][SESSION_ID]["frames"]["00003"]["unresolved"] is False
    assert paired["sessions"][SESSION_ID]["frames"] == list(range(52))
    assert paired["sessions"][SESSION_ID]["window_starts"] == [0, 1]
    assert set(paired["selector_refs"]) == {"RAW", "ROBOT_RGB"}
    assert paired["terminal_publication"] == {
        "profile": source_contract.PAIRED_TERMINAL_PUBLICATION_PROFILE,
        "terminal_name": publisher.PAIRED_NAME,
        "permanent_alias_name": publisher.PAIRED_PRIVATE_NAME,
        "required_nlink": 2,
    }
    alias_state = alias.stat()
    terminal_state = terminal.stat()
    assert (alias_state.st_dev, alias_state.st_ino) == (
        terminal_state.st_dev,
        terminal_state.st_ino,
    )
    assert alias_state.st_nlink == terminal_state.st_nlink == 2
    assert stat.S_IMODE(alias_state.st_mode) == 0o444
    assert alias.read_bytes() == terminal.read_bytes()
    assert report["permanent_terminal_alias_ref"]["path"] == str(alias)
    assert report["postlink_source_contract_revalidation"] is True
    assert report["terminal_manifest_source_retained"] is True
    assert report["terminal_hardlink_nlink"] == 2

    source_contract.require_manifest_selector_ready(
        report["selector_references"]["RAW"],
        report["paired_kept_reference"],
        artifact_root=fixture.raw_payload["artifact_root"],
        project_root=fixture.project_root,
    )
    with pytest.raises(ValueError, match="alias is not a terminal input"):
        source_contract.require_manifest_selector_ready(
            report["selector_references"]["RAW"],
            report["permanent_terminal_alias_ref"],
            artifact_root=fixture.raw_payload["artifact_root"],
            project_root=fixture.project_root,
        )


def test_missing_owner_pin_holds_before_any_read_or_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch, reviewed=False)
    read_calls = 0

    def forbidden_read(*args, **kwargs):
        nonlocal read_calls
        read_calls += 1
        raise AssertionError("no input may be opened before the owner pin gate")

    monkeypatch.setattr(publisher, "_read_snapshot", forbidden_read)
    with pytest.raises(RuntimeError, match=source_contract.ROBOT_RGB_LINEAGE_HOLD):
        run_publish(fixture)
    assert read_calls == 0
    assert not os.path.lexists(fixture.output_root)


def test_asymmetric_frame_deletion_fails_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch)
    robot_path = Path(str(fixture.robot_ref["path"]))
    robot = json.loads(robot_path.read_text())
    robot["sessions"][SESSION_ID]["frames"].pop("00051")
    robot_path.write_text(json.dumps(robot), encoding="utf-8")
    fixture.robot_ref = file_ref(robot_path)
    with pytest.raises(ValueError, match="frame ledger differs"):
        run_publish(fixture)
    assert not os.path.lexists(fixture.output_root)


def test_nested_hardlink_alias_fails_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch)
    source = fixture.raw_image_paths[0]
    target = fixture.raw_image_paths[1]
    target.unlink()
    target.hardlink_to(source)
    raw_path = Path(str(fixture.raw_ref["path"]))
    raw = json.loads(raw_path.read_text())
    raw["sessions"][SESSION_ID]["frames"]["00001"]["image"] = file_ref(target)
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    fixture.raw_ref = file_ref(raw_path)
    with pytest.raises(ValueError, match="singly-linked"):
        run_publish(fixture)
    assert not os.path.lexists(fixture.output_root)


def test_owner_pin_hardlink_alias_fails_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch)
    robot_path = Path(str(fixture.robot_ref["path"]))
    robot = json.loads(robot_path.read_text())
    authority_path = Path(str(robot["robot_rgb_lineage_authority_ref"]["path"]))
    alias = authority_path.with_name("ROBOT_RGB_LINEAGE_AUTHORITY_ALIAS.json")
    alias.hardlink_to(authority_path)
    alias_ref = file_ref(alias)
    robot["robot_rgb_lineage_authority_ref"] = alias_ref
    robot_path.write_text(json.dumps(robot), encoding="utf-8")
    fixture.robot_ref = file_ref(robot_path)
    monkeypatch.setattr(
        source_contract,
        "REVIEWED_ROBOT_RGB_LINEAGE_AUTHORITY_REF",
        dict(alias_ref),
    )
    with pytest.raises(ValueError, match="singly-linked"):
        run_publish(fixture)
    assert not os.path.lexists(fixture.output_root)


def test_same_path_replacement_is_detected_before_terminal_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch)
    original_reverify = publisher._reverify_snapshots
    calls = 0

    def replace_after_pre_output_reverify(snapshots):
        nonlocal calls
        calls += 1
        original_reverify(snapshots)
        if calls == 2:
            target = fixture.metadata_paths[0]
            encoded = target.read_bytes()
            target.rename(target.with_name("training_data.original.json"))
            target.write_bytes(encoded)

    monkeypatch.setattr(
        publisher, "_reverify_snapshots", replace_after_pre_output_reverify
    )
    with pytest.raises(ValueError, match="named identity changed"):
        run_publish(fixture)
    assert fixture.output_root.is_dir()
    assert stat.S_IMODE(fixture.output_root.stat().st_mode) == 0o700
    assert not (fixture.output_root / publisher.PAIRED_NAME).exists()


def test_owner_pin_change_during_publication_cannot_commit_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch)
    original_reverify = publisher._reverify_snapshots
    calls = 0

    def change_pin_after_pre_output_reverify(snapshots):
        nonlocal calls
        calls += 1
        original_reverify(snapshots)
        if calls == 2:
            changed = dict(
                source_contract.REVIEWED_ROBOT_RGB_LINEAGE_AUTHORITY_REF or {}
            )
            changed["sha256"] = "f" * 64
            monkeypatch.setattr(
                source_contract,
                "REVIEWED_ROBOT_RGB_LINEAGE_AUTHORITY_REF",
                changed,
            )

    monkeypatch.setattr(
        publisher, "_reverify_snapshots", change_pin_after_pre_output_reverify
    )
    with pytest.raises(RuntimeError, match=source_contract.ROBOT_RGB_LINEAGE_HOLD):
        run_publish(fixture)
    assert fixture.output_root.is_dir()
    assert stat.S_IMODE(fixture.output_root.stat().st_mode) == 0o700
    assert not (fixture.output_root / publisher.PAIRED_NAME).exists()


def test_existing_output_root_is_never_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch)
    fixture.output_root.mkdir()
    marker = fixture.output_root / "foreign.marker"
    marker.write_bytes(b"foreign")
    marker_state = marker.stat()
    with pytest.raises(FileExistsError, match="already exists"):
        run_publish(fixture)
    assert marker.read_bytes() == b"foreign"
    assert (marker.stat().st_dev, marker.stat().st_ino) == (
        marker_state.st_dev,
        marker_state.st_ino,
    )


def test_foreign_terminal_race_is_preserved_and_publication_stays_private(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch)
    original_link = publisher.os.link

    def inject_foreign(source: str, target: str, *args, **kwargs) -> None:
        root_descriptor = kwargs["dst_dir_fd"]
        descriptor = os.open(
            target,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o444,
            dir_fd=root_descriptor,
        )
        try:
            os.write(descriptor, b"foreign-terminal")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        original_link(source, target, *args, **kwargs)

    monkeypatch.setattr(publisher.os, "link", inject_foreign)
    with pytest.raises(FileExistsError, match="already exists"):
        run_publish(fixture)
    terminal = fixture.output_root / publisher.PAIRED_NAME
    assert terminal.read_bytes() == b"foreign-terminal"
    assert (fixture.output_root / publisher.PAIRED_PRIVATE_NAME).exists()
    assert stat.S_IMODE(fixture.output_root.stat().st_mode) == 0o700


def test_terminal_name_is_absent_until_held_fd_hardlink_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch)
    original_link = publisher.os.link
    observed = False

    def inspect_then_commit(source: str, target: str, *args, **kwargs) -> None:
        nonlocal observed
        root_descriptor = kwargs["dst_dir_fd"]
        names = publisher._fresh_directory_inventory(
            root_descriptor, label="terminal-last test output root"
        )
        assert target not in names
        assert names == {
            publisher.RAW_NAME,
            publisher.ROBOT_NAME,
            publisher.PAIRED_PRIVATE_NAME,
        }
        observed = True
        original_link(source, target, *args, **kwargs)

    monkeypatch.setattr(publisher.os, "link", inspect_then_commit)
    run_publish(fixture)
    assert observed is True
    assert (fixture.output_root / publisher.PAIRED_NAME).is_file()
    assert (fixture.output_root / publisher.PAIRED_PRIVATE_NAME).is_file()


def test_cpfs_rename_einval_and_all_destructive_cleanup_are_irrelevant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch)
    forbidden_calls: list[str] = []

    def forbidden(name: str):
        def reject(*args, **kwargs):
            forbidden_calls.append(name)
            raise OSError(22, f"{name} unavailable on CPFS")

        return reject

    monkeypatch.setattr(publisher.os, "rename", forbidden("rename"))
    monkeypatch.setattr(publisher.os, "replace", forbidden("replace"))
    monkeypatch.setattr(publisher.os, "unlink", forbidden("unlink"))
    monkeypatch.setattr(publisher.os, "rmdir", forbidden("rmdir"))
    report = run_publish(fixture)
    assert report["status"] == "ARTIFACT_EXISTS"
    assert forbidden_calls == []


def test_alias_same_path_replacement_is_rejected_before_terminal_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch)
    original_linker = publisher._link_terminal_from_held

    def replace_alias(root_descriptor, root_path, held_descriptor, private_record):
        alias = fixture.output_root / publisher.PAIRED_PRIVATE_NAME
        encoded = alias.read_bytes()
        alias.rename(fixture.output_root / ".foreign-complete-payload")
        alias.write_bytes(encoded)
        return original_linker(
            root_descriptor,
            root_path,
            held_descriptor,
            private_record,
        )

    monkeypatch.setattr(publisher, "_link_terminal_from_held", replace_alias)
    with pytest.raises(ValueError, match="alias changed"):
        run_publish(fixture)
    assert not (fixture.output_root / publisher.PAIRED_NAME).exists()
    assert stat.S_IMODE(fixture.output_root.stat().st_mode) == 0o700


def test_postpublish_alias_replacement_cannot_downgrade_to_legacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch)
    report = run_publish(fixture)
    alias = fixture.output_root / publisher.PAIRED_PRIVATE_NAME
    encoded = alias.read_bytes()
    os.chmod(fixture.output_root, 0o700)
    alias.rename(fixture.output_root / ".replaced-original-alias")
    alias.write_bytes(encoded)
    alias.chmod(0o444)
    os.chmod(fixture.output_root, 0o555)
    with pytest.raises(ValueError, match="hardlink output inventory drift"):
        source_contract.require_manifest_selector_ready(
            report["selector_references"]["RAW"],
            report["paired_kept_reference"],
            artifact_root=fixture.raw_payload["artifact_root"],
            project_root=fixture.project_root,
        )


def test_fresh_inventory_ignores_stale_integer_fd_offset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "inventory"
    directory.mkdir()
    (directory / "first").write_bytes(b"1")
    (directory / "second").write_bytes(b"2")
    held = os.open(
        directory,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    real_listdir = os.listdir

    def stale_held_listdir(value):
        if value == held:
            return []
        return real_listdir(value)

    monkeypatch.setattr(publisher.os, "listdir", stale_held_listdir)
    try:
        assert publisher.os.listdir(held) == []
        assert publisher._fresh_directory_inventory(
            held, label="publisher stale-offset fixture"
        ) == {"first", "second"}
        assert source_contract._fresh_directory_inventory(
            held, label="source-contract stale-offset fixture"
        ) == {"first", "second"}
    finally:
        os.close(held)


def test_project_cpfs_synthetic_full_publish_canary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = ".humanego-selector-cpfs-canary-"
    canary_parent = Path(tempfile.mkdtemp(prefix=prefix, dir=publisher.PROJECT_ROOT))
    assert canary_parent.parent == publisher.PROJECT_ROOT
    assert canary_parent.name.startswith(prefix)
    try:
        fixture = make_fixture(
            canary_parent,
            monkeypatch,
            frame_count=51,
        )
        report = run_publish(fixture)
        alias = fixture.output_root / publisher.PAIRED_PRIVATE_NAME
        terminal = fixture.output_root / publisher.PAIRED_NAME
        alias_state = alias.stat()
        terminal_state = terminal.stat()
        assert report["postlink_source_contract_revalidation"] is True
        assert alias_state.st_nlink == terminal_state.st_nlink == 2
        assert (alias_state.st_dev, alias_state.st_ino) == (
            terminal_state.st_dev,
            terminal_state.st_ino,
        )
        assert (
            source_contract.require_manifest_selector_ready(
                report["selector_references"]["RAW"],
                report["paired_kept_reference"],
                artifact_root=fixture.raw_payload["artifact_root"],
                project_root=fixture.project_root,
            )["product_line"]
            == "RAW"
        )
    finally:
        for current, directories, _ in os.walk(canary_parent):
            os.chmod(current, 0o700)
            for name in directories:
                os.chmod(Path(current) / name, 0o700)
        shutil.rmtree(canary_parent)
