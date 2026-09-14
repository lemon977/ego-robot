from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import author_g4_robot_cpu_fk_wireframe_fallback as subject  # noqa: E402


REGISTRY = (
    PROJECT_ROOT / "_run/g4_robot_scene_seed573_10session_REGISTRY_20260831_v1.json"
)
REGISTRY_SHA = "4845c672d47fbc1e57f13e32748941552704343f163ba2f33841bec37fe253e2"


def _five_ref(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    info = path.stat()
    return {
        "path": str(path.resolve()),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "device": info.st_dev,
        "inode": info.st_ino,
    }


def _clean_plan(tmp_path: Path, frame_count: int = 3) -> dict[str, object]:
    clean_root = tmp_path / "artifact" / "CLEAN"
    clean_root.mkdir(parents=True)
    path = clean_root / "00000.png"
    assert cv2.imwrite(str(path), np.full((48, 64, 3), 20, np.uint8))
    return {
        "session_id": "grap_a_cap_fixture",
        "denominator": frame_count,
        "selected": {0, 1},
        "passed": (0,),
        "internal_by_frame": {0: 0},
        "scene": object(),
        "clean_authority": {
            "profile": "UNIFIED_V4",
            "refs": {
                0: {
                    "clean_image": _five_ref(path),
                    "shard_manifest": {
                        "path": str((tmp_path / "shard.json").resolve()),
                        "bytes": 1,
                        "sha256": "1" * 64,
                        "device": path.stat().st_dev,
                        "inode": path.stat().st_ino,
                    },
                    "source_raw_sha256": "2" * 64,
                    "consumed_v4_combined_mask_sha256": "3" * 64,
                    "mask_lineage_profile": "UNIFIED_V4",
                }
            },
        },
        "source_resolution": [64, 48],
    }


def _new_clean_authority_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, object], Path]:
    monkeypatch.setattr(subject, "PROJECT_ROOT", tmp_path)
    artifact = tmp_path / "_run" / "fixture_clean" / "grap_a_cap_139" / "artifact"
    clean_dir = artifact / "CLEAN"
    clean_dir.mkdir(parents=True)
    clean_path = clean_dir / "00000.png"
    assert cv2.imwrite(str(clean_path), np.full((48, 64, 3), 80, np.uint8))
    clean_ref = _five_ref(clean_path)
    raw_sha = "a" * 64
    mask_sha = "b" * 64
    shard_path = artifact / "CLEAN_MANIFEST.json"
    shard = {
        "schema_version": "deterministic-table-clean-development-run-v2",
        "status": "ARTIFACT_EXISTS",
        "terminal": True,
        "execution_mode": "DEVELOPMENT_ONLY",
        "formal_consumer_allowed": False,
        "is_formal_clean": False,
        "session_id": "grap_a_cap_139",
        "processed_frame_count": 1,
        "target_frames": [0],
        "frames": [
            {
                "frame_index": 0,
                "raw": {"sha256": raw_sha},
                "clean": {
                    "image": clean_ref,
                    "source_raw_sha256": raw_sha,
                    "consumed_mask_sha256": mask_sha,
                },
            }
        ],
    }
    shard_path.write_text(json.dumps(shard))
    shard_ref = _five_ref(shard_path)
    session_path = tmp_path / "_run" / "fixture_session.json"
    session_manifest = {
        "schema_version": subject.CLEAN_SESSION_SCHEMA,
        "status": "ARTIFACT_EXISTS",
        "terminal": True,
        "formal": False,
        "candidate_requires_human_review": True,
        "claim_limit": "VISUALIZATION_ONLY_NOT_FORMAL_CLEAN_AUTHORITY",
        "partial_session_status": "DEVELOPMENT_ONLY",
        "session_id": "grap_a_cap_139",
        "original_frame_count": 2,
        "ready_frame_count": 1,
        "pending_frame_count": 1,
        "ready_frame_lineage": [
            {
                "frame_index": 0,
                "clean_image": clean_ref,
                "shard_manifest": shard_ref,
                "source_raw_sha256": raw_sha,
                "consumed_v4_combined_mask_sha256": mask_sha,
            }
        ],
    }
    session_path.write_text(json.dumps(session_manifest))
    item = {
        "session_id": "grap_a_cap_139",
        "original_frame_count": 2,
        "clean_visualization_manifest": _five_ref(session_path),
    }
    return item, clean_path


def test_expand_ranges_is_exact_and_rejects_overlap() -> None:
    assert subject._expand_ranges([[0, 2], [4, 7]], denominator=8, label="fixture") == (
        0,
        1,
        4,
        5,
        6,
    )
    with pytest.raises(subject.FallbackAuthorError, match="overlaps"):
        subject._expand_ranges([[0, 3], [2, 4]], denominator=8, label="fixture")


def test_registry_byte_change_is_rejected_by_cli_sha_pin(tmp_path: Path) -> None:
    changed = tmp_path / "changed_registry.json"
    payload = REGISTRY.read_bytes() + b"\n"
    changed.write_bytes(payload)
    assert hashlib.sha256(payload).hexdigest() != REGISTRY_SHA
    with pytest.raises(subject.FallbackAuthorError, match="CLI pin"):
        subject._exact_registry(changed, REGISTRY_SHA)


def test_nine_session_clean_registry_frozen_claims_are_hard_gates(
    tmp_path: Path,
) -> None:
    robot_registry, _ = subject._exact_registry(REGISTRY, REGISTRY_SHA)
    order = robot_registry["session_order"][1:]
    value = {
        "schema_version": subject.CLEAN_REGISTRY_SCHEMA,
        "status": "ARTIFACT_EXISTS",
        "terminal": True,
        "formal": False,
        "candidate_requires_human_review": True,
        "claim_limit": "VISUALIZATION_ONLY_NOT_FORMAL_CLEAN_AUTHORITY",
        "session_count": 9,
        "session_order": order,
        "sessions": [{"session_id": session_id} for session_id in order],
    }
    path = tmp_path / "CLEAN_VISUALIZATION_REGISTRY.json"
    path.write_text(json.dumps(value))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    registry, _ = subject._exact_clean_registry(path, sha, robot_registry)
    assert registry["formal"] is False
    value["formal"] = True
    path.write_text(json.dumps(value))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(subject.FallbackAuthorError, match="contract mismatch"):
        subject._exact_clean_registry(path, sha, robot_registry)


def test_preloaded_or_monkeypatched_public_robot_modules_are_not_executable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, _ = subject._exact_registry(REGISTRY, REGISTRY_SHA)
    from pipeline import robot_renderer_eevee_fullchain as public_core

    called = False

    def attacker(*_: object, **__: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("attacker callable executed")

    attacker.__module__ = public_core.__name__
    monkeypatch.setattr(public_core, "compose_frame_placement", attacker)
    monkeypatch.setattr(public_core, "validate_hand_joint_identity", attacker)
    monkeypatch.setitem(
        sys.modules,
        "pipeline.robot_renderer_cycles",
        SimpleNamespace(forward_kinematics=attacker),
    )
    runtime = subject._runtime_dependencies(registry)
    try:
        assert runtime.core is not public_core
        assert runtime.core.compose_frame_placement is not attacker
        assert runtime.core.compose_frame_placement.__globals__ is runtime.core.__dict__
        assert not called
        subject._reverify_runtime_dependencies(runtime, registry)
    finally:
        subject._close_trusted_robot_runtime(runtime)


def test_author_self_wrong_sha_and_import_after_named_replacement_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(subject.FallbackAuthorError, match="CLI pin"):
        subject._hold_author_implementation("0" * 64)
    imported_payload = b"# implementation A\n"
    path = tmp_path / "author_g4_robot_cpu_fk_wireframe_fallback.py"
    path.write_bytes(imported_payload)
    expected_sha = hashlib.sha256(imported_payload).hexdigest()
    monkeypatch.setattr(subject, "AUTHOR_PATH", path)
    monkeypatch.setattr(subject, "__file__", str(path))
    replacement = tmp_path / "replacement.py"
    replacement.write_bytes(b"# implementation B\n")
    replacement.replace(path)
    with pytest.raises(subject.FallbackAuthorError, match="CLI pin"):
        subject._hold_author_implementation(expected_sha)


def test_copied_author_path_is_rejected_even_with_matching_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    copied = tmp_path / subject.AUTHOR_PATH.name
    copied.write_bytes(subject.AUTHOR_PATH.read_bytes())
    monkeypatch.setattr(subject, "__file__", str(copied))
    with pytest.raises(subject.FallbackAuthorError, match="canonical exact tool path"):
        subject._hold_author_implementation(hashlib.sha256(copied.read_bytes()).hexdigest())


def test_held_author_detects_same_path_replacement_after_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "author_g4_robot_cpu_fk_wireframe_fallback.py"
    path.write_bytes(b"# implementation A\n")
    expected_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(subject, "AUTHOR_PATH", path)
    monkeypatch.setattr(subject, "__file__", str(path))
    descriptor, evidence = subject._hold_author_implementation(expected_sha)
    try:
        replacement = tmp_path / "replacement.py"
        replacement.write_bytes(b"# implementation B\n")
        replacement.replace(path)
        with pytest.raises(subject.FallbackAuthorError, match="identity changed"):
            subject._reverify_held_author_implementation(descriptor, evidence)
    finally:
        subject.os.close(descriptor)


def test_output_parent_rejects_dangling_target(tmp_path: Path) -> None:
    output = tmp_path / "delivery"
    output.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError):
        subject._hold_output_parent(output)


def test_output_parent_same_path_replacement_is_rejected(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    output = parent / "delivery"
    descriptor, evidence = subject._hold_output_parent(output)
    moved = tmp_path / "moved_parent"
    try:
        parent.rename(moved)
        parent.mkdir()
        with pytest.raises(subject.FallbackAuthorError, match="parent identity"):
            subject._verify_output_parent_and_target(
                descriptor,
                evidence,
                target_name=output.name,
                expected_target=None,
            )
    finally:
        subject.os.close(descriptor)


def test_output_parent_preserves_foreign_target(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    output = parent / "delivery"
    descriptor, evidence = subject._hold_output_parent(output)
    try:
        output.mkdir()
        marker = output / "foreign.txt"
        marker.write_text("foreign")
        with pytest.raises(FileExistsError):
            subject._verify_output_parent_and_target(
                descriptor,
                evidence,
                target_name=output.name,
                expected_target=None,
            )
        assert marker.read_text() == "foreign"
    finally:
        subject.os.close(descriptor)


def test_production_refuses_more_than_two_affinity_cpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(subject.os, "sched_getaffinity", lambda _pid: {0, 1, 2})
    with pytest.raises(subject.FallbackAuthorError, match="at most two CPUs"):
        subject.run(
            SimpleNamespace(
                expected_self_sha256=hashlib.sha256(
                    subject.AUTHOR_PATH.read_bytes()
                ).hexdigest(),
                registry=REGISTRY,
                registry_sha256=REGISTRY_SHA,
                clean_registry=tmp_path / "absent_clean_registry.json",
                clean_registry_sha256="0" * 64,
                output_root=tmp_path / "absent",
                fps=30.0,
                output_width=640,
                output_height=480,
                ffmpeg_threads=1,
                preflight_only=False,
            )
        )


def test_original_n_mapping_reads_clean_only_for_pass_and_pending_is_card(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_projection(background: np.ndarray, **_: object) -> np.ndarray:
        result = background.copy()
        result[20:25, 20:25] = (0, 255, 255)
        return result

    monkeypatch.setattr(subject, "_project_visual_origins", fake_projection)
    reads: list[str] = []
    real_read = subject._read_exact_ref

    def read_spy(value: object, *, label: str) -> subject.LoadedRef:
        reads.append(label)
        return real_read(value, label=label)

    monkeypatch.setattr(subject, "_read_exact_ref", read_spy)
    records: list[dict[str, object]] = []
    counts = {
        "diagnostic_wireframe_frames": 0,
        "pending_selected_not_strict_pass": 0,
        "pending_outside_selection": 0,
    }
    frames = list(
            subject._frames(
                _clean_plan(tmp_path),
                core=object(),
                assets=object(),  # type: ignore[arg-type]
            output_width=64,
            output_height=48,
            records=records,  # type: ignore[arg-type]
            counts=counts,
        )
    )
    assert len(frames) == len(records) == 3
    assert [record["status"] for record in records] == [
        "DIAGNOSTIC_CPU_FK_WIREFRAME",
        "PENDING",
        "PENDING",
    ]
    assert len(reads) == 1 and "CLEAN background 0" in reads[0]
    assert all(
        record.get("background_leaf_reads") == 0
        for record in records
        if record["status"] == "PENDING"
    )
    assert counts == {
        "diagnostic_wireframe_frames": 1,
        "pending_selected_not_strict_pass": 1,
        "pending_outside_selection": 1,
    }


def test_pending_only_timeline_never_opens_background(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _clean_plan(tmp_path, frame_count=2)
    plan["passed"] = ()
    plan["internal_by_frame"] = {}

    def forbidden_read(*_: object, **__: object) -> subject.LoadedRef:
        raise AssertionError("PENDING opened a background leaf")

    monkeypatch.setattr(subject, "_read_exact_ref", forbidden_read)
    records: list[dict[str, object]] = []
    counts = {
        "diagnostic_wireframe_frames": 0,
        "pending_selected_not_strict_pass": 0,
        "pending_outside_selection": 0,
    }
    assert (
        len(
            list(
                    subject._frames(
                        plan,
                        core=object(),
                        assets=object(),  # type: ignore[arg-type]
                    output_width=64,
                    output_height=48,
                    records=records,  # type: ignore[arg-type]
                    counts=counts,
                )
            )
        )
        == 2
    )


def test_new_clean_manifest_join_and_same_path_replacement_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    item, clean_path = _new_clean_authority_fixture(tmp_path, monkeypatch)
    authority = subject._load_new_clean_session_authority(
        session_id="grap_a_cap_139", denominator=2, item=item
    )
    ref = authority["refs"][0]["clean_image"]
    subject._read_exact_ref(ref, label="fixture CLEAN")
    replacement = clean_path.with_name("replacement.png")
    assert cv2.imwrite(str(replacement), np.full((48, 64, 3), 80, np.uint8))
    replacement.replace(clean_path)
    with pytest.raises(subject.FallbackAuthorError, match="exact evidence mismatch"):
        subject._read_exact_ref(ref, label="fixture CLEAN")


def test_new_clean_shard_manifest_same_path_replacement_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    item, clean_path = _new_clean_authority_fixture(tmp_path, monkeypatch)
    shard_path = clean_path.parent.parent / "CLEAN_MANIFEST.json"
    shard_path.write_bytes(shard_path.read_bytes() + b"\n")
    with pytest.raises(subject.FallbackAuthorError, match="exact evidence mismatch"):
        subject._load_new_clean_session_authority(
            session_id="grap_a_cap_139", denominator=2, item=item
        )


def test_new_clean_wrong_original_frame_path_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    item, clean_path = _new_clean_authority_fixture(tmp_path, monkeypatch)
    wrong_path = clean_path.with_name("00001.png")
    wrong_path.write_bytes(clean_path.read_bytes())
    session_path = Path(item["clean_visualization_manifest"]["path"])  # type: ignore[index]
    session = json.loads(session_path.read_bytes())
    session["ready_frame_lineage"][0]["clean_image"] = _five_ref(wrong_path)
    session_path.write_text(json.dumps(session))
    item["clean_visualization_manifest"] = _five_ref(session_path)
    with pytest.raises(subject.FallbackAuthorError, match="shard join"):
        subject._load_new_clean_session_authority(
            session_id="grap_a_cap_139", denominator=2, item=item
        )


def test_real_legacy_004_authority_joins_pass_without_reading_pending() -> None:
    authority = subject._load_legacy_004_clean_authority(
        denominator=460, passed=(2, 40)
    )
    assert authority["profile"] == "LEGACY_004_V3_NOT_UNIFIED_FINAL_H"
    assert set(authority["refs"]) == {2, 40}
    assert len(authority["shard_manifests"]) == 8
    assert all(
        record["consumed_v4_combined_mask_sha256"] is None
        and len(record["consumed_legacy_combined_mask_sha256"]) == 64
        for record in authority["refs"].values()
    )


def test_legacy_004_extra_shard_inventory_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_listdir = subject.os.listdir

    def fake_listdir(path: object) -> list[str]:
        result = list(real_listdir(path))
        if Path(path) == Path(subject.LEGACY_004_AGGREGATE_REF["path"]).parents[1]:
            result.append("clean_shard_8")
        return result

    monkeypatch.setattr(subject.os, "listdir", fake_listdir)
    with pytest.raises(subject.FallbackAuthorError, match="root inventory"):
        subject._load_legacy_004_clean_authority(denominator=460, passed=(2,))


def test_atomic_noreplace_unsupported_fails_closed_and_preserves_foreign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    marker = target / "foreign.txt"
    marker.write_text("foreign")
    parent_fd = subject.os.open(tmp_path, subject.os.O_RDONLY | subject.os.O_DIRECTORY)

    def unsupported(*_: object) -> None:
        raise OSError(subject.errno.ENOSYS, "unsupported")

    monkeypatch.setattr(subject, "_renameat2", unsupported)
    try:
        with pytest.raises(subject.FallbackAuthorError, match="unsupported"):
            subject._rename_noreplace(parent_fd, source.name, target.name)
        assert source.is_dir()
        assert marker.read_text() == "foreign"
    finally:
        subject.os.close(parent_fd)


def test_terminal_manifest_is_atomically_linked_only_after_complete_payload(
    tmp_path: Path,
) -> None:
    root_fd = subject.os.open(tmp_path, subject.os.O_RDONLY | subject.os.O_DIRECTORY)
    payload = b'{"complete":true}\n'
    try:
        ref = subject._publish_terminal_manifest(
            root_fd,
            payload,
            final_path=tmp_path / subject.AGGREGATE_MANIFEST_NAME,
        )
        final = tmp_path / subject.AGGREGATE_MANIFEST_NAME
        source = tmp_path / subject.TERMINAL_MANIFEST_SOURCE_NAME
        assert final.read_bytes() == payload
        assert source.read_bytes() == payload
        assert final.stat().st_ino == source.stat().st_ino
        assert final.stat().st_nlink == source.stat().st_nlink == 2
        assert ref == _five_ref(final)
        assert set(path.name for path in tmp_path.iterdir()) == {
            subject.AGGREGATE_MANIFEST_NAME,
            subject.TERMINAL_MANIFEST_SOURCE_NAME,
        }
    finally:
        subject.os.close(root_fd)


def test_terminal_manifest_half_write_never_exposes_terminal_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root_fd = subject.os.open(tmp_path, subject.os.O_RDONLY | subject.os.O_DIRECTORY)
    real_unlink = subject.os.unlink

    def fail_mid_write(descriptor: int, payload: bytes) -> None:
        subject.os.write(descriptor, payload[:3])
        assert not (tmp_path / subject.AGGREGATE_MANIFEST_NAME).exists()
        raise OSError("injected partial write")

    monkeypatch.setattr(subject, "_write_all", fail_mid_write)
    monkeypatch.setattr(
        subject.os,
        "unlink",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("failure path attempted deletion")
        ),
    )
    try:
        with pytest.raises(OSError, match="injected partial write"):
            subject._publish_terminal_manifest(
                root_fd,
                b"complete payload",
                final_path=tmp_path / subject.AGGREGATE_MANIFEST_NAME,
            )
        assert not (tmp_path / subject.AGGREGATE_MANIFEST_NAME).exists()
        retained = tmp_path / subject.TERMINAL_MANIFEST_SOURCE_NAME
        assert retained.read_bytes() == b"com"
        real_unlink(retained)
    finally:
        subject.os.close(root_fd)


def test_terminal_manifest_foreign_target_is_preserved(tmp_path: Path) -> None:
    final = tmp_path / subject.AGGREGATE_MANIFEST_NAME
    final.write_bytes(b"foreign")
    root_fd = subject.os.open(tmp_path, subject.os.O_RDONLY | subject.os.O_DIRECTORY)
    try:
        with pytest.raises(FileExistsError):
            subject._publish_terminal_manifest(
                root_fd,
                b"ours",
                final_path=final,
            )
        assert final.read_bytes() == b"foreign"
        assert set(path.name for path in tmp_path.iterdir()) == {
            subject.AGGREGATE_MANIFEST_NAME
        }
    finally:
        subject.os.close(root_fd)


def test_nonconsumable_reserved_root_is_preserved_without_automatic_deletion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "delivery"
    root.mkdir(mode=0o700)
    marker_path = root / subject.IN_PROGRESS_MARKER_NAME
    marker_path.write_bytes(
        subject._json_bytes(
            {
                "schema_version": "g4-nonconsumable-in-progress-marker-v1",
                "status": (
                    "NONCONSUMABLE_IN_PROGRESS_UNLESS_DELIVERY_MANIFEST_"
                    "PRESENT_AND_VALID"
                ),
                "output_root": str(root),
                "terminal_manifest_required": subject.AGGREGATE_MANIFEST_NAME,
                "whole_tree_atomic_rename": False,
                "no_overwrite": True,
                "terminal_manifest_last": True,
                "filesystem_note": "CPFS_RENAME_NOREPLACE_UNSUPPORTED",
            }
        )
    )
    marker_ref = _five_ref(marker_path)
    root_fd = subject.os.open(root, subject.os.O_RDONLY | subject.os.O_DIRECTORY)
    parent_fd = subject.os.open(tmp_path, subject.os.O_RDONLY | subject.os.O_DIRECTORY)
    try:
        observed = subject._verify_output_tree(
            root_fd=root_fd,
            expected_root=subject.os.fstat(root_fd),
            output_root=root,
            sessions=[],
            in_progress_marker_ref=marker_ref,
            implementation_snapshot_ref=None,
            aggregate_ref=None,
            full_video_decode=False,
            expected_width=1,
            expected_height=1,
        )
        assert observed["verified_session_count"] == 0
        assert not (root / subject.AGGREGATE_MANIFEST_NAME).exists()
        monkeypatch.setattr(
            subject.os,
            "unlink",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("reserved-root failure attempted unlink")
            ),
        )
        monkeypatch.setattr(
            subject.os,
            "rmdir",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("reserved-root failure attempted rmdir")
            ),
        )
        with pytest.raises(subject.FallbackAuthorError, match="deletion is forbidden"):
            subject._cleanup_known_stage(
                parent_fd,
                root_fd,
                root.name,
                output_root=root,
                sessions=[],
                in_progress_marker_ref=marker_ref,
                implementation_snapshot_ref=None,
                aggregate_ref=None,
            )
        assert root.exists()
        assert marker_path.exists()
    finally:
        subject.os.close(root_fd)
        subject.os.close(parent_fd)


def test_scene_state_same_path_replacement_is_rejected(tmp_path: Path) -> None:
    manifest_path = tmp_path / "SCENE_MANIFEST.json"
    state_path = tmp_path / "scene_state.npz"
    state_path.write_bytes(b"state A")
    state_ref = _five_ref(state_path)
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "robot-fullchain-scene-manifest-v1",
                "status": "DEVELOPMENT_ONLY_ARTIFACT_EXISTS",
                "session_id": "grap_a_cap_fixture",
                "frame_count": 1,
                "scene_state": state_ref,
            }
        )
    )
    plan = {
        "session_id": "grap_a_cap_fixture",
        "passed": (0,),
        "internal_by_frame": {0: 0},
        "source_resolution": [64, 48],
        "scene_manifest": _five_ref(manifest_path),
        "scene": subject.WireframeSource(
            state=object(), scene_ref=state_ref
        ),
    }
    core = SimpleNamespace(
        decode_scene_state=lambda _: SimpleNamespace(
            session_id="grap_a_cap_fixture",
            frame_names=("00000",),
            source_resolution=(64, 48),
        )
    )
    assert subject._reverify_all_scene_inputs([plan], core=core) == 2
    replacement = tmp_path / "replacement.npz"
    replacement.write_bytes(b"state A")
    replacement.replace(state_path)
    with pytest.raises(subject.FallbackAuthorError, match="exact evidence mismatch"):
        subject._reverify_all_scene_inputs([plan], core=core)


def test_cleanup_refuses_replaced_session_leaf_and_preserves_foreign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "stage"
    session_id = "grap_a_cap_fixture"
    session_dir = root / session_id
    session_dir.mkdir(parents=True)
    video = session_dir / subject.VIDEO_NAME
    video.write_bytes(b"authored video")
    directory_stat = session_dir.stat()
    directory_ref = {
        "path": str(root / session_id),
        "device": directory_stat.st_dev,
        "inode": directory_stat.st_ino,
    }
    video_ref = _five_ref(video)
    counts = {
        "diagnostic_wireframe_frames": 1,
        "pending_selected_not_strict_pass": 0,
        "pending_outside_selection": 0,
    }
    manifest = {
        "schema_version": subject.SESSION_SCHEMA,
        "session_id": session_id,
        "output_directory": directory_ref,
        "original_frame_count": 1,
        "video": video_ref,
        "counts": counts,
    }
    manifest_path = session_dir / subject.SESSION_MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest))
    session_record = {
        "session_id": session_id,
        "output_directory": directory_ref,
        "original_frame_count": 1,
        "counts": counts,
        "video": video_ref,
        "session_manifest": _five_ref(manifest_path),
    }
    replacement = session_dir / "replacement.mp4"
    replacement.write_bytes(b"foreign")
    replacement.replace(video)
    root_fd = subject.os.open(root, subject.os.O_RDONLY | subject.os.O_DIRECTORY)
    parent_fd = subject.os.open(tmp_path, subject.os.O_RDONLY | subject.os.O_DIRECTORY)
    try:
        monkeypatch.setattr(
            subject.os,
            "unlink",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("foreign failure attempted unlink")
            ),
        )
        monkeypatch.setattr(
            subject.os,
            "rmdir",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("foreign failure attempted rmdir")
            ),
        )
        with pytest.raises(subject.FallbackAuthorError, match="deletion is forbidden"):
            subject._cleanup_known_stage(
                parent_fd,
                root_fd,
                root.name,
                output_root=root,
                sessions=[session_record],
                in_progress_marker_ref=None,
                implementation_snapshot_ref=None,
                aggregate_ref=None,
            )
        assert video.read_bytes() == b"foreign"
        assert manifest_path.exists()
        assert root.exists()
    finally:
        subject.os.close(root_fd)
        subject.os.close(parent_fd)
