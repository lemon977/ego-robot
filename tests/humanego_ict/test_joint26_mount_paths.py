"""Synthetic mount/reference admission; no model, GPU or real dataset calls.

The engineering runner supplies an attempt-local pytest ``--basetemp`` and
cache directory, so every file made by these fixtures stays in that attempt.
"""
from copy import deepcopy
import shlex
from types import SimpleNamespace

import pytest

from chaoyang.humanego_ict import joint26_data as data
from chaoyang.humanego_ict import joint26_runtime as runtime
from chaoyang.humanego_ict import project_mount_paths as mounts
from chaoyang.humanego_ict.joint26_contract import PREPARE, TRAIN


@pytest.fixture
def mount_project(tmp_path, monkeypatch):
    """A real current mount plus an absent, explicitly named historical mount."""
    root, old_root = tmp_path / "current_project", tmp_path / "old_project"
    root.mkdir()
    monkeypatch.setattr(mounts, "PROJECT_ALIASES", (root, old_root))
    monkeypatch.setattr(runtime, "ROOT", root)
    payload = root / "inputs" / "payload.bin"
    payload.parent.mkdir()
    payload.write_bytes(b"synthetic immutable signed input\x00\x01")
    current_ref = data.file_ref(payload)
    old_ref = dict(current_ref, path=str(old_root / "inputs" / "payload.bin"))
    return SimpleNamespace(root=root, old_root=old_root, payload=payload,
                           current_ref=current_ref, old_ref=old_ref)


def test_old_signed_reference_reads_same_bytes_without_mutation(mount_project):
    project = mount_project
    signed_before = deepcopy(project.old_ref)
    bytes_before = project.payload.read_bytes()
    normalized = mounts.normalize_ref(project.old_ref, project.root)
    assert normalized == project.current_ref
    assert normalized is not project.old_ref
    assert mounts.resolve_input_path(project.old_ref["path"], project.root) == project.payload
    assert data.reference_bytes(project.old_ref, project.root) == bytes_before
    assert project.old_ref == signed_before
    assert project.payload.read_bytes() == bytes_before


@pytest.mark.parametrize("field,value", [("bytes", 0), ("sha256", "0" * 64)])
def test_mapped_reference_still_checks_signed_size_and_sha(mount_project, field, value):
    project = mount_project
    bad = dict(project.old_ref, **{field: value})
    signed_before, bytes_before = deepcopy(bad), project.payload.read_bytes()
    with pytest.raises(data.Joint26DataError, match="SHA_OR_IDENTITY_DRIFT"):
        data.reference_bytes(bad, project.root)
    assert bad == signed_before
    assert project.payload.read_bytes() == bytes_before


def test_reference_equality_compares_aliases_without_mutation(mount_project):
    project = mount_project
    before = deepcopy((project.old_ref, project.current_ref))
    assert mounts.reference_equal(project.old_ref, project.current_ref, project.root)
    assert (project.old_ref, project.current_ref) == before


@pytest.mark.parametrize("field,value", [("bytes", 0), ("sha256", "0" * 64)])
def test_reference_equality_rejects_signature_difference(mount_project, field, value):
    project = mount_project
    other = dict(project.current_ref, **{field: value})
    assert not mounts.reference_equal(project.old_ref, other, project.root)


@pytest.mark.parametrize("case", ["neighbor_project", "parent_traversal", "missing_file"])
def test_input_rejects_noncanonical_or_missing_source(mount_project, case):
    project = mount_project
    if case == "neighbor_project":
        path = project.root.parent / "neighbor_project" / "inputs" / "payload.bin"
        path.parent.mkdir(parents=True)
        path.write_bytes(project.payload.read_bytes())
    elif case == "parent_traversal":
        path = project.root / "inputs" / ".." / "inputs" / "payload.bin"
    else:
        path = project.old_root / "inputs" / "missing.bin"
    with pytest.raises(ValueError):
        mounts.resolve_input_path(path, project.root)


@pytest.mark.parametrize("case", ["file", "parent"])
def test_input_rejects_symlink_at_file_or_parent(mount_project, case):
    project = mount_project
    if case == "file":
        path = project.root / "linked.bin"
        path.symlink_to(project.payload)
    else:
        parent = project.root / "linked_inputs"
        parent.symlink_to(project.payload.parent, target_is_directory=True)
        path = parent / project.payload.name
    with pytest.raises(ValueError, match="SYMLINK"):
        mounts.resolve_input_path(path, project.root)


def test_relative_input_is_resolved_only_under_current_root(mount_project):
    project = mount_project
    assert mounts.resolve_input_path("inputs/payload.bin", project.root) == project.payload


def test_matching_suffix_cannot_guess_a_project_alias(mount_project):
    project = mount_project
    unregistered = project.old_root.parent / "other_mount" / "inputs" / "payload.bin"
    assert project.payload.is_file()
    with pytest.raises(ValueError):
        mounts.resolve_input_path(unregistered, project.root)


def test_output_accepts_new_path_only_under_current_root(mount_project):
    project = mount_project
    output = project.root / "results" / "future.json"
    assert not output.exists()
    assert mounts.resolve_output_path(output, project.root) == output


def test_output_cannot_be_redirected_from_historical_alias(mount_project):
    project = mount_project
    with pytest.raises(ValueError):
        mounts.resolve_output_path(project.old_root / "results" / "future.json", project.root)


def test_existing_alias_roots_with_distinct_identity_are_rejected(mount_project):
    project = mount_project
    before = project.payload.read_bytes()
    project.old_root.mkdir()
    assert (project.root.stat().st_dev, project.root.stat().st_ino) != (
        project.old_root.stat().st_dev, project.old_root.stat().st_ino)
    with pytest.raises(ValueError, match="ALIAS_IDENTITY_CONFLICT"):
        mounts.resolve_input_path(project.old_ref["path"], project.root)
    assert project.payload.read_bytes() == before


def _consumer(project):
    # Exercise reference admission directly without constructing label arrays.
    dataset = object.__new__(data.Joint26Dataset)
    dataset.root, dataset._refs = project.root, {}
    return dataset


def test_dataset_consumer_deduplicates_current_and_historical_aliases(mount_project):
    project = mount_project
    dataset = _consumer(project)
    before = deepcopy((project.old_ref, project.current_ref))
    assert dataset._consume(project.old_ref) == project.payload.read_bytes()
    assert dataset._consume(project.current_ref) == project.payload.read_bytes()
    assert set(dataset._refs) == {str(project.payload)}
    dataset.verify_all_dependencies()
    assert (project.old_ref, project.current_ref) == before


def test_dataset_consumer_rejects_conflicting_alias_signature(mount_project):
    project = mount_project
    dataset = _consumer(project)
    dataset._consume(project.old_ref)
    accepted = deepcopy(dataset._refs)
    conflict = dict(project.current_ref, sha256="0" * 64)
    with pytest.raises(data.Joint26DataError, match="CONFLICTING_PATH_SIGNATURE"):
        dataset._consume(conflict)
    assert dataset._refs == accepted


def _arguments(project, operation):
    if operation == PREPARE:
        return ["--task-packet", str(project.payload), "--source-map", str(project.payload),
                "--output", str(project.root / "results")]
    return ["train", "--task-packet", str(project.payload),
            "--manifest", str(project.payload), "--config", str(project.payload),
            "--output", str(project.root / "results"), "--branch", "raw"]


def _command(module, operation, arguments):
    return shlex.join(["python", "-m", module, "run", operation, *arguments])


@pytest.mark.parametrize("module", ["chaoyang.cli", "chaoyang.humanego_ict.joint26_cli"])
@pytest.mark.parametrize("operation", [PREPARE, TRAIN])
def test_supported_command_modules_pin_exact_joint26_arguments(mount_project, module, operation):
    args = _arguments(mount_project, operation)
    assert runtime._command_is_pinned([_command(module, operation, args)], operation, args)


@pytest.mark.parametrize("case", ["changed_value", "extra_argument", "missing_argument", "reordered"])
def test_command_pin_rejects_argument_deviation(mount_project, case):
    args = _arguments(mount_project, TRAIN)
    changed = args.copy()
    if case == "changed_value":
        changed[-1] = "robotized"
    elif case == "extra_argument":
        changed += ["--unexpected", "value"]
    elif case == "missing_argument":
        changed = changed[:-2]
    else:
        changed[1:5] = changed[3:5] + changed[1:3]
    command = _command("chaoyang.humanego_ict.joint26_cli", TRAIN, changed)
    assert not runtime._command_is_pinned([command], TRAIN, args)


@pytest.mark.parametrize("module", ["chaoyang.cli", "chaoyang.humanego_ict.joint26_cli"])
def test_command_pin_does_not_admit_unmaintained_operation(mount_project, module):
    operation = "unmaintained_synthetic_operation"
    args = _arguments(mount_project, TRAIN)
    assert not runtime._command_is_pinned([_command(module, operation, args)], operation, args)


def test_command_pin_rejects_another_module(mount_project):
    args = _arguments(mount_project, TRAIN)
    command = _command("other_project.joint26_cli", TRAIN, args)
    assert not runtime._command_is_pinned([command], TRAIN, args)


def test_command_pin_does_not_reauthorize_changed_input_alias_arguments(mount_project):
    project = mount_project
    args = _arguments(project, PREPARE)
    signed_args = args.copy()
    signed_args[signed_args.index("--source-map") + 1] = project.old_ref["path"]
    signed_before = signed_args.copy()
    command = _command("chaoyang.humanego_ict.joint26_cli", PREPARE, signed_args)
    assert not runtime._command_is_pinned([command], PREPARE, args)
    assert signed_args == signed_before
