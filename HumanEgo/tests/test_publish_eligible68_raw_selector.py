from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytest

from tools import publish_eligible68_raw_selector as publisher
from utils import source_contract


SESSION_ID = "grap_a_cap_004"


@dataclass
class Fixture:
    project_root: Path
    production_root: Path
    raw_root: Path
    output_root: Path
    metadata_paths: list[Path]
    adapter_links: list[Path]
    raw_paths: list[Path]


def _file_ref(path: Path) -> dict[str, object]:
    encoded = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _small_contract(monkeypatch: pytest.MonkeyPatch, frame_count: int = 2) -> None:
    monkeypatch.setattr(source_contract, "ELIGIBLE68", frozenset({SESSION_ID}))
    monkeypatch.setattr(source_contract, "ELIGIBLE68_ORDER", (SESSION_ID,))
    monkeypatch.setattr(
        source_contract,
        "ELIGIBLE68_FRAME_COUNTS",
        {SESSION_ID: frame_count},
    )


def _png_bytes(frame_index: int, *, shape: tuple[int, ...] = (960, 1280, 3)) -> bytes:
    pixels = np.zeros(shape, dtype=np.uint8)
    pixels[...] = frame_index + 17
    ok, encoded = cv2.imencode(".png", pixels)
    assert ok
    return encoded.tobytes()


def _write_metadata(path: Path, rgb_path: Path, **extra: object) -> None:
    payload: dict[str, object] = {"obs": {"rgb_path": str(rgb_path)}}
    payload.update(extra)
    path.write_text(json.dumps(payload), encoding="utf-8")


def make_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    frame_count: int = 2,
) -> Fixture:
    _small_contract(monkeypatch, frame_count)
    project_root = tmp_path / "project"
    production_root = project_root / "formal_final_v3"
    raw_root = tmp_path / "original_raw"
    output_root = project_root / "HumanEgo/outputs/eligible68_raw_selector_20990101_v1"
    (project_root / "HumanEgo/outputs").mkdir(parents=True)
    metadata_paths: list[Path] = []
    adapter_links: list[Path] = []
    raw_paths: list[Path] = []
    for frame_index in range(frame_count):
        frame_key = f"{frame_index:05d}"
        raw_path = raw_root / SESSION_ID / "preprocess/all_data" / frame_key / "rgb.png"
        raw_path.parent.mkdir(parents=True)
        raw_path.write_bytes(_png_bytes(frame_index))
        metadata_path = (
            production_root
            / SESSION_ID
            / "09_humanego_adapter/preprocess/all_data"
            / frame_key
            / "training_data.json"
        )
        metadata_path.parent.mkdir(parents=True)
        adapter_link = metadata_path.parent / "rgb.png"
        adapter_link.symlink_to(raw_path)
        _write_metadata(
            metadata_path,
            adapter_link,
            hand_side_status="MISSING" if frame_index == 0 else "OUTSIDE_IMAGE",
            action_status="MISSING",
        )
        metadata_paths.append(metadata_path)
        adapter_links.append(adapter_link)
        raw_paths.append(raw_path)
    return Fixture(
        project_root=project_root,
        production_root=production_root,
        raw_root=raw_root,
        output_root=output_root,
        metadata_paths=metadata_paths,
        adapter_links=adapter_links,
        raw_paths=raw_paths,
    )


def _publish(fixture: Fixture, *, check_only: bool = False) -> dict[str, object]:
    return publisher.publish_eligible68_raw_selector(
        project_root=fixture.project_root,
        production_root=fixture.production_root,
        raw_artifact_root=fixture.raw_root,
        output_root=fixture.output_root,
        check_only=check_only,
    )


def _publish_argv(fixture: Fixture) -> list[str]:
    return [
        "--project-root",
        str(fixture.project_root),
        "--production-root",
        str(fixture.production_root),
        "--raw-artifact-root",
        str(fixture.raw_root),
        "--output-root",
        str(fixture.output_root),
    ]


def test_publish_exact_raw_selector_and_ignore_nonraw_statuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch)
    report = _publish(fixture)
    manifest_path = fixture.output_root / publisher.OUTPUT_NAME
    alias_path = fixture.output_root / publisher.PRIVATE_NAME
    payload = json.loads(manifest_path.read_bytes())
    assert report["status"] == "ARTIFACT_EXISTS"
    assert report["frame_count"] == 2
    assert report["terminal_publication_profile"] == (
        source_contract.RAW_SELECTOR_STANDALONE_PUBLICATION_PROFILE
    )
    assert set(fixture.output_root.iterdir()) == {manifest_path, alias_path}
    assert stat.S_IMODE(fixture.output_root.stat().st_mode) == 0o555
    assert stat.S_IMODE(manifest_path.stat().st_mode) == 0o444
    assert stat.S_IMODE(alias_path.stat().st_mode) == 0o444
    assert manifest_path.stat().st_nlink == 2
    assert alias_path.stat().st_nlink == 2
    assert (manifest_path.stat().st_dev, manifest_path.stat().st_ino) == (
        alias_path.stat().st_dev,
        alias_path.stat().st_ino,
    )
    assert report["permanent_terminal_alias_ref"]["path"] == str(alias_path)
    frames = payload["sessions"][SESSION_ID]["frames"]
    assert list(frames) == ["00000", "00001"]
    for frame_key, record in frames.items():
        assert set(record) == {"metadata", "image", "unresolved"}
        assert record["unresolved"] is False
        assert record["image"]["path"] == str(
            fixture.raw_paths[int(frame_key)].resolve()
        )
        assert "status" not in record
    source_contract.validate_selector_manifest_reference(
        report["selector_manifest_ref"],
        artifact_root=fixture.raw_root,
        project_root=fixture.project_root,
    )


def test_publisher_uses_shared_semantic_core_before_terminal_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch)
    original_semantic = source_contract._validate_standalone_raw_selector_payload
    original_public = source_contract.validate_selector_manifest_reference
    observed_states: list[tuple[int, set[str]] | None] = []
    public_states: list[tuple[int, set[str]]] = []

    def guarded_public(*args: object, **kwargs: object) -> dict[str, object]:
        public_states.append(
            (
                stat.S_IMODE(fixture.output_root.stat().st_mode),
                {path.name for path in fixture.output_root.iterdir()},
            )
        )
        return original_public(*args, **kwargs)

    def guarded_semantic(*args: object, **kwargs: object) -> dict[str, object]:
        if fixture.output_root.exists():
            observed_states.append(
                (
                    stat.S_IMODE(fixture.output_root.stat().st_mode),
                    {path.name for path in fixture.output_root.iterdir()},
                )
            )
        else:
            observed_states.append(None)
        return original_semantic(*args, **kwargs)

    monkeypatch.setattr(
        source_contract,
        "validate_selector_manifest_reference",
        guarded_public,
    )
    monkeypatch.setattr(
        source_contract,
        "_validate_standalone_raw_selector_payload",
        guarded_semantic,
    )
    report = _publish(fixture)
    assert report["status"] == "ARTIFACT_EXISTS"
    assert observed_states == [
        None,
        (0o555, {publisher.PRIVATE_NAME}),
    ]
    assert public_states == [(0o555, {publisher.PRIVATE_NAME, publisher.OUTPUT_NAME})]


def test_check_only_is_full_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch)
    report = _publish(fixture, check_only=True)
    assert report["status"] == "CHECK_ONLY_PASS_NO_OUTPUT_WRITTEN"
    assert report["raw_png_count"] == 2
    assert report["adapter_rgb_symlink_count"] == 2
    assert not os.path.lexists(fixture.output_root)


def test_light_check_scans_all_metadata_and_bounded_raw_sample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch, frame_count=3)
    report = publisher.light_check_eligible68_raw_selector_inputs(
        project_root=fixture.project_root,
        production_root=fixture.production_root,
        raw_artifact_root=fixture.raw_root,
        estimate_session=SESSION_ID,
        output_root=fixture.output_root,
    )
    assert report["status"] == "LIGHT_CHECK_ONLY_PASS_NO_SELECTOR_CLAIM"
    assert report["metadata_count"] == 3
    assert report["sampled_raw_png_count"] == 3
    assert report["estimate_session_raw_png_count"] == 3
    assert report["full_selector_validated"] is False
    assert report["output_written"] is False
    assert not os.path.lexists(fixture.output_root)


def test_light_check_requires_literal_estimate_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="literal eligible68"):
        publisher.light_check_eligible68_raw_selector_inputs(
            project_root=fixture.project_root,
            production_root=fixture.production_root,
            raw_artifact_root=fixture.raw_root,
            estimate_session="grap_a_cap_999",
            output_root=fixture.output_root,
        )
    assert not os.path.lexists(fixture.output_root)


def test_direct_original_raw_obs_path_is_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch)
    fixture.adapter_links[0].unlink()
    _write_metadata(fixture.metadata_paths[0], fixture.raw_paths[0])
    report = _publish(fixture, check_only=True)
    assert report["adapter_rgb_symlink_count"] == 1


@pytest.mark.parametrize("attack", ["wrong_target", "relative_escape", "clean"])
def test_obs_rgb_path_attacks_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attack: str,
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch)
    link = fixture.adapter_links[0]
    if attack == "wrong_target":
        link.unlink()
        link.symlink_to(fixture.raw_paths[1])
    elif attack == "relative_escape":
        link.unlink()
        relative = os.path.relpath(fixture.raw_paths[0], link.parent)
        assert ".." in Path(relative).parts
        link.symlink_to(relative)
    else:
        clean_path = link.parent / "clean_rgb.png"
        clean_path.write_bytes(_png_bytes(0))
        _write_metadata(fixture.metadata_paths[0], clean_path)
    with pytest.raises(ValueError):
        _publish(fixture, check_only=True)
    assert not os.path.lexists(fixture.output_root)


@pytest.mark.parametrize("attack", ["corrupt", "wrong_shape", "hardlink_alias"])
def test_raw_pixel_and_alias_attacks_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attack: str,
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch)
    if attack == "corrupt":
        fixture.raw_paths[0].write_bytes(b"not-a-png")
    elif attack == "wrong_shape":
        fixture.raw_paths[0].write_bytes(_png_bytes(0, shape=(480, 640, 3)))
    else:
        fixture.raw_paths[1].unlink()
        os.link(fixture.raw_paths[0], fixture.raw_paths[1])
    with pytest.raises(ValueError):
        _publish(fixture, check_only=True)
    assert not os.path.lexists(fixture.output_root)


def test_existing_foreign_output_is_preserved_before_source_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch)
    fixture.output_root.mkdir()
    marker = fixture.output_root / "FOREIGN"
    marker.write_text("keep", encoding="utf-8")

    def forbidden_build(**_: object) -> None:
        raise AssertionError("source scan must not begin for an occupied output")

    monkeypatch.setattr(publisher, "_build_plan", forbidden_build)
    with pytest.raises(FileExistsError):
        _publish(fixture)
    assert marker.read_text(encoding="utf-8") == "keep"


def test_input_same_path_replacement_is_detected_and_evidence_is_retained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch)
    original_reverify = publisher._reverify_sources
    calls = 0

    def replace_after_prepublication_reverify(plan: publisher.SelectorPlan) -> None:
        nonlocal calls
        original_reverify(plan)
        calls += 1
        if calls == 1:
            replacement = fixture.raw_paths[0].with_suffix(".replacement")
            replacement.write_bytes(fixture.raw_paths[0].read_bytes())
            os.replace(replacement, fixture.raw_paths[0])

    monkeypatch.setattr(
        publisher,
        "_reverify_sources",
        replace_after_prepublication_reverify,
    )
    with pytest.raises(ValueError, match="identity changed"):
        _publish(fixture)
    assert fixture.output_root.is_dir()
    alias = fixture.output_root / publisher.PRIVATE_NAME
    terminal = fixture.output_root / publisher.OUTPUT_NAME
    assert alias.is_file()
    assert not terminal.exists()
    assert stat.S_IMODE(fixture.output_root.stat().st_mode) == 0o555
    reference = _file_ref(alias)
    reference["path"] = str(terminal)
    with pytest.raises((FileNotFoundError, ValueError)):
        source_contract.validate_selector_manifest_reference(
            reference,
            artifact_root=fixture.raw_root,
            project_root=fixture.project_root,
        )


def test_public_validator_rejects_foreign_leaf_in_standalone_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch)
    report = _publish(fixture)
    os.chmod(fixture.output_root, 0o755)
    marker = fixture.output_root / "FOREIGN"
    marker.write_text("keep", encoding="utf-8")
    os.chmod(fixture.output_root, 0o555)
    with pytest.raises(ValueError, match="root inventory drift"):
        source_contract.validate_selector_manifest_reference(
            report["selector_manifest_ref"],
            artifact_root=fixture.raw_root,
            project_root=fixture.project_root,
        )
    assert marker.read_text(encoding="utf-8") == "keep"


def test_public_validator_rejects_standalone_root_mode_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch)
    report = _publish(fixture)
    os.chmod(fixture.output_root, 0o755)
    with pytest.raises(ValueError, match="root identity/mode drift"):
        source_contract.validate_selector_manifest_reference(
            report["selector_manifest_ref"],
            artifact_root=fixture.raw_root,
            project_root=fixture.project_root,
        )


def test_public_validator_rejects_standalone_terminal_hardlink_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch)
    report = _publish(fixture)
    alias_root = fixture.project_root / "aliases"
    alias_root.mkdir()
    alias = alias_root / "raw-selector-alias.json"
    os.link(fixture.output_root / publisher.OUTPUT_NAME, alias)
    with pytest.raises(ValueError, match="terminal/alias identity drift"):
        source_contract.validate_selector_manifest_reference(
            report["selector_manifest_ref"],
            artifact_root=fixture.raw_root,
            project_root=fixture.project_root,
        )
    assert alias.is_file()


def test_public_validator_rejects_permanent_alias_as_consumer_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch)
    report = _publish(fixture)
    with pytest.raises(ValueError, match="fixed terminal basename"):
        source_contract.validate_selector_manifest_reference(
            report["permanent_terminal_alias_ref"],
            artifact_root=fixture.raw_root,
            project_root=fixture.project_root,
        )


def test_public_validator_rejects_terminal_same_path_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch)
    report = _publish(fixture)
    manifest = fixture.output_root / publisher.OUTPUT_NAME
    original_validate_common = source_contract._validate_common
    attacked = False

    def replace_after_initial_terminal_load(
        payload: object, schema: str, label: str
    ) -> None:
        nonlocal attacked
        if not attacked:
            encoded = manifest.read_bytes()
            replacement = fixture.project_root / "replacement-selector.json"
            replacement.write_bytes(encoded)
            os.chmod(replacement, 0o444)
            os.chmod(fixture.output_root, 0o755)
            os.replace(replacement, manifest)
            os.chmod(fixture.output_root, 0o555)
            attacked = True
        original_validate_common(payload, schema, label)

    monkeypatch.setattr(
        source_contract,
        "_validate_common",
        replace_after_initial_terminal_load,
    )
    with pytest.raises(ValueError, match="terminal/alias identity drift"):
        source_contract.validate_selector_manifest_reference(
            report["selector_manifest_ref"],
            artifact_root=fixture.raw_root,
            project_root=fixture.project_root,
        )
    assert attacked is True


def test_public_validator_rejects_alias_same_path_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch)
    report = _publish(fixture)
    alias = fixture.output_root / publisher.PRIVATE_NAME
    original_validate_common = source_contract._validate_common
    attacked = False

    def replace_after_initial_pair_load(
        payload: object, schema: str, label: str
    ) -> None:
        nonlocal attacked
        if not attacked:
            replacement = fixture.project_root / "replacement-alias.json"
            replacement.write_bytes(alias.read_bytes())
            os.chmod(replacement, 0o444)
            os.chmod(fixture.output_root, 0o755)
            os.replace(replacement, alias)
            os.chmod(fixture.output_root, 0o555)
            attacked = True
        original_validate_common(payload, schema, label)

    monkeypatch.setattr(
        source_contract,
        "_validate_common",
        replace_after_initial_pair_load,
    )
    with pytest.raises(ValueError, match="terminal/alias identity drift"):
        source_contract.validate_selector_manifest_reference(
            report["selector_manifest_ref"],
            artifact_root=fixture.raw_root,
            project_root=fixture.project_root,
        )
    assert attacked is True


def test_standalone_wrapper_cannot_bypass_gate_with_dotdot_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch)
    report = _publish(fixture)
    reference = dict(report["selector_manifest_ref"])
    reference["path"] = str(
        fixture.output_root / ".." / fixture.output_root.name / publisher.OUTPUT_NAME
    )
    with pytest.raises(ValueError, match="not canonical|path/profile drift"):
        source_contract.validate_selector_manifest_reference(
            reference,
            artifact_root=fixture.raw_root,
            project_root=fixture.project_root,
        )


def test_generic_legacy_selector_reference_does_not_inherit_wrapper_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch)
    _publish(fixture)
    generic_root = fixture.project_root / "generic_selector_records"
    generic_root.mkdir(mode=0o700)
    generic_manifest = generic_root / publisher.OUTPUT_NAME
    generic_manifest.write_bytes(
        (fixture.output_root / publisher.OUTPUT_NAME).read_bytes()
    )
    (generic_root / "OTHER_GENERIC_INPUT").write_text("allowed", encoding="utf-8")
    validated = source_contract.validate_selector_manifest_reference(
        _file_ref(generic_manifest),
        artifact_root=fixture.raw_root,
        project_root=fixture.project_root,
    )
    assert validated["product_line"] == "RAW"


def test_partial_manifest_write_leaves_nonconsumable_private_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch)
    intended: bytes | None = None

    def partial_write(descriptor: int, encoded: bytes) -> None:
        nonlocal intended
        intended = encoded
        os.write(descriptor, encoded[:17])
        raise OSError("synthetic partial write")

    monkeypatch.setattr(publisher, "_write_all", partial_write)
    with pytest.raises(OSError, match="synthetic partial"):
        _publish(fixture)
    assert intended is not None
    manifest = fixture.output_root / publisher.PRIVATE_NAME
    assert fixture.output_root.is_dir()
    assert stat.S_IMODE(fixture.output_root.stat().st_mode) == 0o700
    assert manifest.read_bytes() == intended[:17]
    intended_ref = {
        "path": str(fixture.output_root / publisher.OUTPUT_NAME),
        "bytes": len(intended),
        "sha256": hashlib.sha256(intended).hexdigest(),
    }
    with pytest.raises(ValueError):
        source_contract.validate_selector_manifest_reference(
            intended_ref,
            artifact_root=fixture.raw_root,
            project_root=fixture.project_root,
        )


def test_output_ancestor_symlink_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch)
    humanego = fixture.project_root / "HumanEgo"
    moved = fixture.project_root / "HumanEgo.real"
    humanego.rename(moved)
    humanego.symlink_to(moved, target_is_directory=True)
    with pytest.raises(OSError):
        _publish(fixture, check_only=True)
    assert not os.path.lexists(moved / "outputs" / fixture.output_root.name)


def test_held_outputs_parent_rejects_rename_then_symlink_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch)
    outputs = fixture.project_root / "HumanEgo/outputs"
    moved = fixture.project_root / "HumanEgo/outputs.held"
    original_reverify = publisher._reverify_sources
    attacked = False

    def attack(plan: publisher.SelectorPlan) -> None:
        nonlocal attacked
        original_reverify(plan)
        if not attacked:
            outputs.rename(moved)
            outputs.symlink_to(moved, target_is_directory=True)
            attacked = True

    monkeypatch.setattr(publisher, "_reverify_sources", attack)
    with pytest.raises(ValueError, match="outputs directory was replaced"):
        _publish(fixture)
    assert not os.path.lexists(moved / fixture.output_root.name)


def test_full_check_only_terminal_outputs_swap_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch)
    outputs = fixture.project_root / "HumanEgo/outputs"
    moved = fixture.project_root / "HumanEgo/outputs.held"
    original_reverify = publisher._reverify_sources

    def attack(plan: publisher.SelectorPlan) -> None:
        original_reverify(plan)
        outputs.rename(moved)
        outputs.symlink_to(moved, target_is_directory=True)

    monkeypatch.setattr(publisher, "_reverify_sources", attack)
    with pytest.raises(ValueError, match="outputs directory was replaced"):
        _publish(fixture, check_only=True)
    assert outputs.is_symlink()
    assert not os.path.lexists(moved / fixture.output_root.name)


def test_light_check_terminal_outputs_swap_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch)
    outputs = fixture.project_root / "HumanEgo/outputs"
    moved = fixture.project_root / "HumanEgo/outputs.held"
    original_reverify = publisher._reverify_obs_symlink
    calls = 0

    def attack(snapshot: publisher.ObsSymlinkSnapshot) -> None:
        nonlocal calls
        original_reverify(snapshot)
        calls += 1
        if calls == len(fixture.adapter_links):
            outputs.rename(moved)
            outputs.symlink_to(moved, target_is_directory=True)

    monkeypatch.setattr(publisher, "_reverify_obs_symlink", attack)
    with pytest.raises(ValueError, match="outputs directory was replaced"):
        publisher.light_check_eligible68_raw_selector_inputs(
            project_root=fixture.project_root,
            production_root=fixture.production_root,
            raw_artifact_root=fixture.raw_root,
            estimate_session=SESSION_ID,
            output_root=fixture.output_root,
        )
    assert calls == len(fixture.adapter_links)
    assert outputs.is_symlink()
    assert not os.path.lexists(moved / fixture.output_root.name)


@pytest.mark.parametrize("mode", ["full", "light"])
def test_check_only_terminal_foreign_child_is_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch)
    marker = fixture.output_root / "FOREIGN"
    if mode == "full":
        original_reverify = publisher._reverify_sources

        def attack_full(plan: publisher.SelectorPlan) -> None:
            original_reverify(plan)
            fixture.output_root.mkdir()
            marker.write_text("keep", encoding="utf-8")

        monkeypatch.setattr(publisher, "_reverify_sources", attack_full)

        def action() -> object:
            return _publish(fixture, check_only=True)

    else:
        original_reverify_symlink = publisher._reverify_obs_symlink
        calls = 0

        def attack_light(snapshot: publisher.ObsSymlinkSnapshot) -> None:
            nonlocal calls
            original_reverify_symlink(snapshot)
            calls += 1
            if calls == len(fixture.adapter_links):
                fixture.output_root.mkdir()
                marker.write_text("keep", encoding="utf-8")

        monkeypatch.setattr(publisher, "_reverify_obs_symlink", attack_light)

        def action() -> object:
            return publisher.light_check_eligible68_raw_selector_inputs(
                project_root=fixture.project_root,
                production_root=fixture.production_root,
                raw_artifact_root=fixture.raw_root,
                estimate_session=SESSION_ID,
                output_root=fixture.output_root,
            )

    with pytest.raises(FileExistsError):
        action()
    assert marker.read_text(encoding="utf-8") == "keep"


def test_no_output_argument_allows_absent_outputs_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch)
    outputs = fixture.project_root / "HumanEgo/outputs"
    outputs.rmdir()
    full = publisher.publish_eligible68_raw_selector(
        project_root=fixture.project_root,
        production_root=fixture.production_root,
        raw_artifact_root=fixture.raw_root,
        output_root=None,
        check_only=True,
    )
    light = publisher.light_check_eligible68_raw_selector_inputs(
        project_root=fixture.project_root,
        production_root=fixture.production_root,
        raw_artifact_root=fixture.raw_root,
        estimate_session=SESSION_ID,
        output_root=None,
    )
    assert full["status"] == "CHECK_ONLY_PASS_NO_OUTPUT_WRITTEN"
    assert light["status"] == "LIGHT_CHECK_ONLY_PASS_NO_SELECTOR_CLAIM"
    assert not os.path.lexists(outputs)


def test_no_output_argument_still_terminally_pins_humanego(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch)
    humanego = fixture.project_root / "HumanEgo"
    moved = fixture.project_root / "HumanEgo.held"
    original_reverify = publisher._reverify_sources

    def attack(plan: publisher.SelectorPlan) -> None:
        original_reverify(plan)
        humanego.rename(moved)
        humanego.symlink_to(moved, target_is_directory=True)

    monkeypatch.setattr(publisher, "_reverify_sources", attack)
    with pytest.raises(ValueError, match="HumanEgo directory was replaced"):
        publisher.publish_eligible68_raw_selector(
            project_root=fixture.project_root,
            production_root=fixture.production_root,
            raw_artifact_root=fixture.raw_root,
            output_root=None,
            check_only=True,
        )


def test_publisher_never_calls_delete_or_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch)

    def forbidden(*_: object, **__: object) -> None:
        raise AssertionError("destructive namespace operation called")

    monkeypatch.setattr(publisher.os, "unlink", forbidden)
    monkeypatch.setattr(publisher.os, "rmdir", forbidden)
    monkeypatch.setattr(publisher.os, "rename", forbidden)
    monkeypatch.setattr(publisher.os, "replace", forbidden)
    report = _publish(fixture)
    assert report["no_delete_or_rename"] is True


def test_output_must_be_directly_under_held_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch)
    nested = fixture.project_root / "HumanEgo/outputs/nested/raw_selector"
    nested.parent.mkdir()
    with pytest.raises(ValueError, match="direct child"):
        publisher.publish_eligible68_raw_selector(
            project_root=fixture.project_root,
            production_root=fixture.production_root,
            raw_artifact_root=fixture.raw_root,
            output_root=nested,
            check_only=True,
        )


def test_standalone_output_requires_reserved_wrapper_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch)
    unprofiled = fixture.project_root / "HumanEgo/outputs/raw_selector_unprofiled"
    with pytest.raises(ValueError, match="standalone RAW wrapper name"):
        publisher.publish_eligible68_raw_selector(
            project_root=fixture.project_root,
            production_root=fixture.production_root,
            raw_artifact_root=fixture.raw_root,
            output_root=unprofiled,
            check_only=True,
        )
    assert not os.path.lexists(unprofiled)


def _assert_alias_only_is_publicly_rejected(fixture: Fixture) -> None:
    alias = fixture.output_root / publisher.PRIVATE_NAME
    terminal = fixture.output_root / publisher.OUTPUT_NAME
    assert alias.is_file()
    assert not os.path.lexists(terminal)
    assert {path.name for path in fixture.output_root.iterdir()} == {
        publisher.PRIVATE_NAME
    }
    reference = _file_ref(alias)
    reference["path"] = str(terminal)
    with pytest.raises((FileNotFoundError, ValueError)):
        source_contract.validate_selector_manifest_reference(
            reference,
            artifact_root=fixture.raw_root,
            project_root=fixture.project_root,
        )


@pytest.mark.parametrize(
    "window",
    [
        "root_fsync",
        "parent_fsync",
        "named_verify",
        "semantic",
        "source_reverify",
        "final_verify",
    ],
)
def test_every_precommit_fault_window_leaves_nonconsumable_alias_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    window: str,
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch)

    if window in {"root_fsync", "parent_fsync"}:
        original_fsync = publisher.os.fsync
        injected = False

        def fail_selected_fsync(descriptor: int) -> None:
            nonlocal injected
            state = os.fstat(descriptor)
            output_exists = fixture.output_root.exists()
            output_mode = (
                stat.S_IMODE(fixture.output_root.stat().st_mode)
                if output_exists
                else None
            )
            target_identity = (
                (fixture.output_root.stat().st_dev, fixture.output_root.stat().st_ino)
                if window == "root_fsync" and output_exists
                else (
                    fixture.output_root.parent.stat().st_dev,
                    fixture.output_root.parent.stat().st_ino,
                )
            )
            if (
                not injected
                and output_mode == 0o555
                and (state.st_dev, state.st_ino) == target_identity
            ):
                injected = True
                raise OSError(f"synthetic {window}")
            original_fsync(descriptor)

        monkeypatch.setattr(publisher.os, "fsync", fail_selected_fsync)
    elif window == "named_verify":
        original_verify = publisher._verify_manifest

        def fail_named(*args: object, **kwargs: object) -> None:
            if kwargs.get("expected_root_mode") == 0o555:
                raise ValueError("synthetic named verify")
            original_verify(*args, **kwargs)

        monkeypatch.setattr(publisher, "_verify_manifest", fail_named)
    elif window == "semantic":
        original_semantic = source_contract._validate_standalone_raw_selector_payload
        calls = 0

        def fail_terminal_semantic(
            *args: object, **kwargs: object
        ) -> dict[str, object]:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise ValueError("synthetic terminal semantic validation")
            return original_semantic(*args, **kwargs)

        monkeypatch.setattr(
            source_contract,
            "_validate_standalone_raw_selector_payload",
            fail_terminal_semantic,
        )
    elif window == "source_reverify":
        original_reverify = publisher._reverify_sources
        calls = 0

        def fail_terminal_reverify(plan: publisher.SelectorPlan) -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise ValueError("synthetic terminal source reverify")
            original_reverify(plan)

        monkeypatch.setattr(publisher, "_reverify_sources", fail_terminal_reverify)
    else:
        original_verify = publisher._verify_manifest
        terminal_mode_calls = 0

        def fail_final(*args: object, **kwargs: object) -> None:
            nonlocal terminal_mode_calls
            if kwargs.get("expected_root_mode") == 0o555:
                terminal_mode_calls += 1
                if terminal_mode_calls == 2:
                    raise ValueError("synthetic final verify")
            original_verify(*args, **kwargs)

        monkeypatch.setattr(publisher, "_verify_manifest", fail_final)

    return_code = publisher.main(
        [
            "--project-root",
            str(fixture.project_root),
            "--production-root",
            str(fixture.production_root),
            "--raw-artifact-root",
            str(fixture.raw_root),
            "--output-root",
            str(fixture.output_root),
        ]
    )
    assert return_code == 2
    _assert_alias_only_is_publicly_rejected(fixture)


def test_terminal_hardlink_failure_leaves_alias_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch)

    def fail_link(*_: object, **__: object) -> None:
        raise OSError("synthetic link failure")

    monkeypatch.setattr(publisher.os, "link", fail_link)
    with pytest.raises(RuntimeError, match="hardlink publication is unsupported"):
        _publish(fixture)
    _assert_alias_only_is_publicly_rejected(fixture)


def test_foreign_terminal_race_is_preserved_and_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch)
    original_link = publisher.os.link
    foreign = b"FOREIGN TERMINAL"

    def inject_foreign_then_link(*args: object, **kwargs: object) -> None:
        terminal = fixture.output_root / publisher.OUTPUT_NAME
        terminal.write_bytes(foreign)
        original_link(*args, **kwargs)

    monkeypatch.setattr(publisher.os, "link", inject_foreign_then_link)
    with pytest.raises(FileExistsError, match="terminal already exists"):
        _publish(fixture)
    assert (fixture.output_root / publisher.OUTPUT_NAME).read_bytes() == foreign
    assert (fixture.output_root / publisher.PRIVATE_NAME).is_file()


def test_postcommit_close_error_cannot_downgrade_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch)
    original_commit = publisher._commit_terminal_from_held
    original_close = publisher.os.close
    committed = False
    injected = False

    def observed_commit(root_descriptor: int, manifest_descriptor: int) -> None:
        nonlocal committed
        original_commit(root_descriptor, manifest_descriptor)
        committed = True

    def close_then_fail_once(descriptor: int) -> None:
        nonlocal injected
        original_close(descriptor)
        if committed and not injected:
            injected = True
            raise OSError("synthetic postcommit close failure")

    monkeypatch.setattr(publisher, "_commit_terminal_from_held", observed_commit)
    monkeypatch.setattr(publisher.os, "close", close_then_fail_once)
    report = _publish(fixture)
    assert injected is True
    assert report["status"] == "ARTIFACT_EXISTS_POSTCOMMIT_RECOVERED"
    assert report["postcommit_recovered"] is True
    source_contract.validate_selector_manifest_reference(
        report["selector_manifest_ref"],
        artifact_root=fixture.raw_root,
        project_root=fixture.project_root,
    )


def test_postcommit_validation_runs_after_terminal_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch)
    original_commit = publisher._commit_terminal_from_held
    original_postcommit = publisher._postcommit_validate
    committed = False
    postcommit_calls = 0

    def observed_commit(root_descriptor: int, manifest_descriptor: int) -> None:
        nonlocal committed
        original_commit(root_descriptor, manifest_descriptor)
        committed = True

    def guarded_postcommit(**kwargs: object) -> None:
        nonlocal postcommit_calls
        assert committed is True
        postcommit_calls += 1
        original_postcommit(**kwargs)

    monkeypatch.setattr(publisher, "_commit_terminal_from_held", observed_commit)
    monkeypatch.setattr(publisher, "_postcommit_validate", guarded_postcommit)
    report = _publish(fixture)
    assert committed is True
    assert postcommit_calls == 1
    assert report["status"] == "ARTIFACT_EXISTS"
    assert report["terminal_commit_then_canonical_classification"] is True
    assert report["rc2_implies_canonical_consumer_reject"] is True


@pytest.mark.parametrize(
    "window",
    ["root_fsync", "parent_fsync", "named_verify", "public", "final_verify"],
)
def test_postcommit_fault_recovers_when_canonical_consumer_accepts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    window: str,
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch)
    original_commit = publisher._commit_terminal_from_held
    committed = False

    def observed_commit(root_descriptor: int, manifest_descriptor: int) -> None:
        nonlocal committed
        original_commit(root_descriptor, manifest_descriptor)
        committed = True

    monkeypatch.setattr(publisher, "_commit_terminal_from_held", observed_commit)

    if window in {"root_fsync", "parent_fsync"}:
        original_fsync = publisher.os.fsync
        injected = False

        def fail_postcommit_fsync(descriptor: int) -> None:
            nonlocal injected
            state = os.fstat(descriptor)
            target = (
                fixture.output_root
                if window == "root_fsync"
                else fixture.output_root.parent
            ).stat()
            if (
                committed
                and not injected
                and (state.st_dev, state.st_ino) == (target.st_dev, target.st_ino)
            ):
                injected = True
                raise OSError(f"synthetic postcommit {window}")
            original_fsync(descriptor)

        monkeypatch.setattr(publisher.os, "fsync", fail_postcommit_fsync)
    elif window in {"named_verify", "final_verify"}:
        original_pair_verify = publisher._verify_committed_terminal_pair
        calls = 0
        failure_call = 1 if window == "named_verify" else 2

        def fail_selected_pair_verify(*args: object, **kwargs: object) -> None:
            nonlocal calls
            calls += 1
            if calls == failure_call:
                raise ValueError(f"synthetic postcommit {window}")
            original_pair_verify(*args, **kwargs)

        monkeypatch.setattr(
            publisher,
            "_verify_committed_terminal_pair",
            fail_selected_pair_verify,
        )
    else:
        original_public = source_contract.validate_selector_manifest_reference
        calls = 0

        def fail_public_once(*args: object, **kwargs: object) -> dict[str, object]:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ValueError("synthetic postcommit public validator")
            return original_public(*args, **kwargs)

        monkeypatch.setattr(
            source_contract,
            "validate_selector_manifest_reference",
            fail_public_once,
        )

    report = _publish(fixture)
    assert committed is True
    assert report["status"] == "ARTIFACT_EXISTS_POSTCOMMIT_RECOVERED"
    assert report["postcommit_recovered"] is True
    source_contract.validate_selector_manifest_reference(
        report["selector_manifest_ref"],
        artifact_root=fixture.raw_root,
        project_root=fixture.project_root,
    )


def test_named_root_swap_before_link_returns_rc2_and_consumer_rejects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch)
    original_commit = publisher._commit_terminal_from_held
    moved = fixture.output_root.with_name(f"{fixture.output_root.name}.held")

    def swap_root_then_commit(root_descriptor: int, manifest_descriptor: int) -> None:
        fixture.output_root.rename(moved)
        fixture.output_root.mkdir(mode=0o555)
        original_commit(root_descriptor, manifest_descriptor)

    monkeypatch.setattr(
        publisher,
        "_commit_terminal_from_held",
        swap_root_then_commit,
    )
    assert publisher.main(_publish_argv(fixture)) == 2
    assert {path.name for path in moved.iterdir()} == {
        publisher.PRIVATE_NAME,
        publisher.OUTPUT_NAME,
    }
    assert list(fixture.output_root.iterdir()) == []
    rejected_reference = _file_ref(moved / publisher.OUTPUT_NAME)
    rejected_reference["path"] = str(fixture.output_root / publisher.OUTPUT_NAME)
    with pytest.raises((FileNotFoundError, ValueError)):
        source_contract.validate_selector_manifest_reference(
            rejected_reference,
            artifact_root=fixture.raw_root,
            project_root=fixture.project_root,
        )


def test_alias_external_link_swap_returns_rc2_and_preserves_foreign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch)
    original_commit = publisher._commit_terminal_from_held
    external_payload = fixture.project_root / "foreign-selector-payload.json"
    external_alias = fixture.project_root / "foreign-selector-alias.json"

    def swap_alias_after_link(root_descriptor: int, manifest_descriptor: int) -> None:
        original_commit(root_descriptor, manifest_descriptor)
        alias = fixture.output_root / publisher.PRIVATE_NAME
        external_payload.write_bytes(alias.read_bytes())
        os.chmod(external_payload, 0o444)
        os.link(external_payload, external_alias)
        os.chmod(fixture.output_root, 0o755)
        os.replace(external_payload, alias)
        os.chmod(fixture.output_root, 0o555)

    monkeypatch.setattr(
        publisher,
        "_commit_terminal_from_held",
        swap_alias_after_link,
    )
    assert publisher.main(_publish_argv(fixture)) == 2
    assert external_alias.is_file()
    alias = fixture.output_root / publisher.PRIVATE_NAME
    terminal = fixture.output_root / publisher.OUTPUT_NAME
    assert (alias.stat().st_dev, alias.stat().st_ino) == (
        external_alias.stat().st_dev,
        external_alias.stat().st_ino,
    )
    assert (alias.stat().st_dev, alias.stat().st_ino) != (
        terminal.stat().st_dev,
        terminal.stat().st_ino,
    )
    with pytest.raises(ValueError, match="terminal/alias identity drift"):
        source_contract.validate_selector_manifest_reference(
            _file_ref(terminal),
            artifact_root=fixture.raw_root,
            project_root=fixture.project_root,
        )


def test_external_hardlink_added_before_terminal_commit_returns_rc2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch)
    original_commit = publisher._commit_terminal_from_held
    external = fixture.project_root / "foreign-held-payload-link.json"

    def add_external_link_then_commit(
        root_descriptor: int, manifest_descriptor: int
    ) -> None:
        os.link(
            fixture.output_root / publisher.PRIVATE_NAME,
            external,
        )
        original_commit(root_descriptor, manifest_descriptor)

    monkeypatch.setattr(
        publisher,
        "_commit_terminal_from_held",
        add_external_link_then_commit,
    )
    assert publisher.main(_publish_argv(fixture)) == 2
    assert external.is_file()
    terminal = fixture.output_root / publisher.OUTPUT_NAME
    assert terminal.stat().st_nlink == 3
    with pytest.raises(ValueError, match="terminal/alias identity drift"):
        source_contract.validate_selector_manifest_reference(
            _file_ref(terminal),
            artifact_root=fixture.raw_root,
            project_root=fixture.project_root,
        )


def test_one_shot_postcommit_exception_returns_cli_rc0_recovered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch)

    def fail_postcommit_once(**_: object) -> None:
        raise OSError("synthetic one-shot postcommit failure")

    monkeypatch.setattr(publisher, "_postcommit_validate", fail_postcommit_once)
    assert publisher.main(_publish_argv(fixture)) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "ARTIFACT_EXISTS_POSTCOMMIT_RECOVERED"
    assert report["postcommit_recovered"] is True
    source_contract.validate_selector_manifest_reference(
        report["selector_manifest_ref"],
        artifact_root=fixture.raw_root,
        project_root=fixture.project_root,
    )


def test_valid_competing_terminal_commit_recovers_instead_of_rc2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch)
    original_commit = publisher._commit_terminal_from_held

    def competing_commit(root_descriptor: int, manifest_descriptor: int) -> None:
        os.link(
            f"/proc/self/fd/{manifest_descriptor}",
            publisher.OUTPUT_NAME,
            dst_dir_fd=root_descriptor,
            follow_symlinks=True,
        )
        original_commit(root_descriptor, manifest_descriptor)

    monkeypatch.setattr(
        publisher,
        "_commit_terminal_from_held",
        competing_commit,
    )
    report = _publish(fixture)
    assert report["status"] == "ARTIFACT_EXISTS_POSTCOMMIT_RECOVERED"
    source_contract.validate_selector_manifest_reference(
        report["selector_manifest_ref"],
        artifact_root=fixture.raw_root,
        project_root=fixture.project_root,
    )


def test_valid_terminal_race_during_final_precommit_verify_recovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch)
    original_verify = publisher._verify_manifest
    terminal_mode_calls = 0

    def competing_terminal_then_fail(*args: object, **kwargs: object) -> None:
        nonlocal terminal_mode_calls
        if kwargs.get("expected_root_mode") == 0o555:
            terminal_mode_calls += 1
            if terminal_mode_calls == 2:
                publisher._commit_terminal_from_held(
                    int(args[1]),
                    int(args[4]),
                )
                raise ValueError("synthetic final precommit verifier failure")
        original_verify(*args, **kwargs)

    monkeypatch.setattr(publisher, "_verify_manifest", competing_terminal_then_fail)
    report = _publish(fixture)
    assert terminal_mode_calls == 2
    assert report["status"] == "ARTIFACT_EXISTS_POSTCOMMIT_RECOVERED"
    assert report["postcommit_recovered"] is True
    source_contract.validate_selector_manifest_reference(
        report["selector_manifest_ref"],
        artifact_root=fixture.raw_root,
        project_root=fixture.project_root,
    )
