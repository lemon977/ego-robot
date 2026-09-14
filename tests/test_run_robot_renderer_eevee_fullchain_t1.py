import argparse
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

import tools.render_robot_eevee_fullchain_worker as worker
import tools.run_robot_renderer_eevee_fullchain_t1 as runner
from tools.run_robot_renderer_eevee_fullchain_t1 import build_cpu_freeze, parser


PROJECT = Path(__file__).resolve().parents[1]


def _lease_payload() -> dict[str, object]:
    now = datetime.now(timezone.utc)
    return {
        "holder": "gpt",
        "since": (now - timedelta(minutes=1)).isoformat(),
        "task_id": "robot-renderer-004-frame0",
        "requester": {
            "who": "gpt",
            "purpose": "004 robot EEVEE candidate frame0 only",
            "max_minutes": 20,
            "at": (now - timedelta(minutes=2)).isoformat(),
        },
        "scope": {
            "session_id": "grap_a_cap_004",
            "frame_range": {"start": 0, "count": 1},
            "generation_token": "renderer-generation-token-004-frame0-v1",
        },
    }


def _lease_expected_fields() -> dict[str, object]:
    return {
        "expected_holder": "gpt",
        "expected_requester": "gpt",
        "expected_task_id": "robot-renderer-004-frame0",
        "expected_purpose": "004 robot EEVEE candidate frame0 only",
        "expected_session_id": "grap_a_cap_004",
        "expected_frame_start": 0,
        "expected_frame_count": 1,
        "expected_generation_token": "renderer-generation-token-004-frame0-v1",
    }


def _write_lease(path: Path, payload: object) -> tuple[int, str]:
    encoded = (json.dumps(payload, sort_keys=True) + "\n").encode()
    path.write_bytes(encoded)
    return len(encoded), hashlib.sha256(encoded).hexdigest()


def _acquire(path: Path, payload: object | None = None) -> runner.HeldGpuLease:
    byte_count, digest = _write_lease(
        path, _lease_payload() if payload is None else payload
    )
    return runner.HeldGpuLease.acquire(
        path=path,
        strict=runner.load_strict_io(PROJECT),
        expected_bytes=byte_count,
        expected_sha256=digest,
        expected_fields=_lease_expected_fields(),
    )


def test_cpu_freeze_binds_authorization_cpu4_shared_io_and_three_way_tool_source() -> (
    None
):
    freeze = build_cpu_freeze(PROJECT)
    assert freeze["status"] == "CPU_READY_WAITING_IMMUTABLE_SCENE_STATE_NO_GPU_EXECUTED"
    assert freeze["semantic_source"]["sha256"] == (
        "be41fff506302f489f43e55f41955a514b252308774415658f3fa77c198f23be"
    )
    assert freeze["shared_immutable_io"]["sha256"] == (
        "e9af8d38146dee0813d415330fb626a295482740b160435fac544a045d74e57e"
    )
    consistency = freeze["three_way_tool_consistency"]
    assert consistency["equal"] is True
    assert consistency["observed_z_m"] == [0.145, 0.145]
    assert consistency["mismatch_action"] == "A_CLASS_P0_STOP_RENDER_LINE"


def test_cpu_freeze_is_candidate_only_visual_geometry_and_no_execution() -> None:
    freeze = build_cpu_freeze(PROJECT)
    candidate = freeze["candidate_contract"]
    assert candidate["mount_provenance"] == "PROVISIONAL_MOUNT_VISUAL_ONLY"
    assert candidate["contact_infeasible"] == "UNMEASURED"
    assert candidate["formal_consumer_allowed"] is False
    assert candidate["baseline_frozen"] is False
    assert freeze["geometry_contract"]["collision_geometry"] == "NOT_CONSUMED"
    assert freeze["asset_closure"]["visual_meshes"] == 63
    assert set(freeze["execution_counters"].values()) == {0}
    assert set(freeze["implementation"]) == {
        "core",
        "worker",
        "runner",
        "core_tests",
        "runner_tests",
    }


def test_historical_video_is_reference_only_and_not_pixel_input() -> None:
    reference = build_cpu_freeze(PROJECT)["historical_visual_reference_only"]
    assert reference["sha256"] == (
        "efc98e79a71b2379793152688346e369c48af1260018e5e3d3ab5a1205ec1888"
    )
    assert reference["decoded_contract"]["frames"] == 460
    assert reference["not_an_input_to_pixel_generation"] is True


def test_execute_cli_requires_explicit_mode_and_inputs() -> None:
    parsed = parser().parse_args(
        ["--project-root", str(PROJECT), "--prepare-freeze", "--output", "out.json"]
    )
    assert parsed.prepare_freeze is True
    assert parsed.execute is False


def test_worker_consumes_stl_from_verified_memfd_not_asset_path() -> None:
    source = (PROJECT / "tools/render_robot_eevee_fullchain_worker.py").read_text()
    assert "os.memfd_create" in source
    assert 'filepath=f"/proc/self/fd/{descriptor}"' in source
    assert "verified.payload" in source


class _StringLookupBrokenOutputs(list[SimpleNamespace]):
    def __getitem__(self, key):
        if isinstance(key, str):
            raise KeyError(key)
        return super().__getitem__(key)


def _socket(
    name: str,
    *,
    identifier: str | None = None,
    socket_type: str = "VALUE",
    is_output: bool = True,
    enabled: bool = True,
    is_unavailable: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        identifier=name if identifier is None else identifier,
        type=socket_type,
        is_output=is_output,
        enabled=enabled,
        is_unavailable=is_unavailable,
    )


def test_worker_render_pass_socket_ignores_broken_string_lookup() -> None:
    expected = _socket("IndexOB")
    outputs = _StringLookupBrokenOutputs(
        [
            _socket("Deprecated"),
            _socket("Deprecated", identifier="Deprecated.001"),
            expected,
        ]
    )
    with pytest.raises(KeyError, match="IndexOB"):
        outputs["IndexOB"]
    assert worker._render_pass_output_socket(outputs, "IndexOB") is expected


def test_worker_render_pass_socket_uses_identifier_to_resolve_same_name() -> None:
    deprecated = _socket("IndexOB", identifier="Deprecated.IndexOB")
    expected = _socket("IndexOB")
    assert (
        worker._render_pass_output_socket([deprecated, expected], "IndexOB") is expected
    )


def test_worker_render_pass_socket_rejects_exact_duplicate_ambiguity() -> None:
    with pytest.raises(
        runner.FullChainRendererError, match="IndexOB identity is not unique"
    ):
        worker._render_pass_output_socket(
            [_socket("IndexOB"), _socket("IndexOB")], "IndexOB"
        )


def test_worker_render_pass_socket_depth_regression() -> None:
    expected = _socket("Depth")
    assert (
        worker._render_pass_output_socket([_socket("Deprecated"), expected], "Depth")
        is expected
    )


@pytest.mark.parametrize(
    "candidate",
    [
        _socket("IndexOB", socket_type="RGBA"),
        _socket("IndexOB", is_output=False),
        _socket("IndexOB", enabled=False),
        _socket("IndexOB", is_unavailable=True),
    ],
)
def test_worker_render_pass_socket_rejects_wrong_type_or_availability(
    candidate: SimpleNamespace,
) -> None:
    with pytest.raises(
        runner.FullChainRendererError, match="IndexOB identity is not unique"
    ):
        worker._render_pass_output_socket([candidate], "IndexOB")


def test_gpu_release_requires_memory_observation_not_only_child_exit(
    monkeypatch,
) -> None:
    samples = iter(((10.0, 240.0), (0.0, 112.0)))
    monkeypatch.setattr(runner, "_gpu_sample", lambda: next(samples, None))
    observed_samples, observed = runner._observe_gpu_release(
        (0.0, 100.0), timeout_seconds=0.1, poll_seconds=0.0
    )
    assert observed is True
    assert observed_samples == [(10.0, 240.0), (0.0, 112.0)]


def test_gpu_release_fails_closed_without_baseline_or_return_to_baseline(
    monkeypatch,
) -> None:
    assert runner._observe_gpu_release(None) == ([], False)
    monkeypatch.setattr(runner, "_gpu_sample", lambda: (0.0, 240.0))
    observed_samples, observed = runner._observe_gpu_release(
        (0.0, 100.0), timeout_seconds=0.0, poll_seconds=0.0
    )
    assert observed is False
    assert observed_samples == [(0.0, 240.0)]


def test_gpu_lease_exact_scope_is_held_and_second_executor_is_rejected(
    tmp_path: Path,
) -> None:
    lease_path = tmp_path / "GPU_LEASE.json"
    lease = _acquire(lease_path)
    try:
        evidence = lease.evidence_ref()
        assert evidence["nlink"] == 1
        assert evidence["lifecycle_lock"] == "FLOCK_EXCLUSIVE_HELD"
        assert evidence["scope"]["session_id"] == "grap_a_cap_004"
        with pytest.raises(runner.FullChainRendererError, match="cannot be held"):
            byte_count = lease_path.stat().st_size
            digest = hashlib.sha256(lease_path.read_bytes()).hexdigest()
            runner.HeldGpuLease.acquire(
                path=lease_path,
                strict=runner.load_strict_io(PROJECT),
                expected_bytes=byte_count,
                expected_sha256=digest,
                expected_fields=_lease_expected_fields(),
            )
    finally:
        lease.close()


@pytest.mark.parametrize(
    ("field", "mutate"),
    [
        ("holder", lambda value: value.__setitem__("holder", "claude")),
        (
            "requester.who",
            lambda value: value["requester"].__setitem__("who", "claude"),
        ),
        ("task_id", lambda value: value.__setitem__("task_id", "other-task")),
        (
            "requester.purpose",
            lambda value: value["requester"].__setitem__("purpose", "batch"),
        ),
        (
            "scope.session_id",
            lambda value: value["scope"].__setitem__("session_id", "other-session"),
        ),
        (
            "scope.frame_range.start",
            lambda value: value["scope"]["frame_range"].__setitem__("start", 1),
        ),
        (
            "scope.frame_range.count",
            lambda value: value["scope"]["frame_range"].__setitem__("count", 2),
        ),
        (
            "scope.generation_token",
            lambda value: value["scope"].__setitem__(
                "generation_token", "different-generation-token"
            ),
        ),
    ],
)
def test_gpu_lease_rejects_each_scope_mismatch(
    tmp_path: Path, field: str, mutate
) -> None:
    payload = deepcopy(_lease_payload())
    mutate(payload)
    lease_path = tmp_path / "GPU_LEASE.json"
    with pytest.raises(runner.FullChainRendererError, match=field):
        _acquire(lease_path, payload)


def test_gpu_lease_rejects_extra_keys_expiry_and_malformed_json(tmp_path: Path) -> None:
    extra = _lease_payload()
    extra["unscoped_fallback"] = True
    with pytest.raises(runner.FullChainRendererError, match="root keys mismatch"):
        _acquire(tmp_path / "extra.json", extra)

    expired = _lease_payload()
    expired["since"] = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    expired["requester"]["at"] = (
        datetime.now(timezone.utc) - timedelta(hours=3)
    ).isoformat()
    with pytest.raises(runner.FullChainRendererError, match="expired"):
        _acquire(tmp_path / "expired.json", expired)

    malformed = b"{not-json\n"
    malformed_path = tmp_path / "malformed.json"
    malformed_path.write_bytes(malformed)
    with pytest.raises(runner.FullChainRendererError, match="malformed JSON"):
        runner.HeldGpuLease.acquire(
            path=malformed_path,
            strict=runner.load_strict_io(PROJECT),
            expected_bytes=len(malformed),
            expected_sha256=hashlib.sha256(malformed).hexdigest(),
            expected_fields=_lease_expected_fields(),
        )

    duplicate = (
        json.dumps(_lease_payload(), separators=(",", ":"))[:-1] + ',"holder":"gpt"}'
    ).encode()
    duplicate_path = tmp_path / "duplicate.json"
    duplicate_path.write_bytes(duplicate)
    with pytest.raises(runner.FullChainRendererError, match="duplicate JSON key"):
        runner.HeldGpuLease.acquire(
            path=duplicate_path,
            strict=runner.load_strict_io(PROJECT),
            expected_bytes=len(duplicate),
            expected_sha256=hashlib.sha256(duplicate).hexdigest(),
            expected_fields=_lease_expected_fields(),
        )


def test_gpu_lease_rejects_symlink_and_hardlink(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    byte_count, digest = _write_lease(target, _lease_payload())
    symlink = tmp_path / "symlink.json"
    symlink.symlink_to(target.name)
    with pytest.raises(runner.FullChainRendererError, match="cannot be held"):
        runner.HeldGpuLease.acquire(
            path=symlink,
            strict=runner.load_strict_io(PROJECT),
            expected_bytes=byte_count,
            expected_sha256=digest,
            expected_fields=_lease_expected_fields(),
        )

    hardlink = tmp_path / "hardlink.json"
    os.link(target, hardlink)
    with pytest.raises(runner.FullChainRendererError, match="nlink=1"):
        runner.HeldGpuLease.acquire(
            path=hardlink,
            strict=runner.load_strict_io(PROJECT),
            expected_bytes=byte_count,
            expected_sha256=digest,
            expected_fields=_lease_expected_fields(),
        )


def test_gpu_lease_rejects_atomic_replacement_even_with_identical_bytes(
    tmp_path: Path,
) -> None:
    lease_path = tmp_path / "GPU_LEASE.json"
    lease = _acquire(lease_path)
    try:
        replacement = tmp_path / "replacement.json"
        replacement.write_bytes(lease.payload_bytes)
        os.replace(replacement, lease_path)
        with pytest.raises(
            runner.FullChainRendererError, match="path identity changed"
        ):
            lease.revalidate("race-test")
    finally:
        lease.close()


def test_gpu_lease_rejects_parent_directory_replacement_even_with_identical_bytes(
    tmp_path: Path,
) -> None:
    lease_directory = tmp_path / "lease-directory"
    lease_directory.mkdir()
    lease_path = lease_directory / "GPU_LEASE.json"
    lease = _acquire(lease_path)
    try:
        displaced = tmp_path / "displaced-lease-directory"
        os.replace(lease_directory, displaced)
        lease_directory.mkdir()
        (lease_directory / "GPU_LEASE.json").write_bytes(lease.payload_bytes)
        with pytest.raises(
            runner.FullChainRendererError, match="parent path identity changed"
        ):
            lease.revalidate("ancestor-race-test")
    finally:
        lease.close()


class _FakeRecord:
    payload = b"fake-scene-state"

    def evidence_ref(self) -> dict[str, object]:
        return {"path": "/fake", "bytes": len(self.payload), "sha256": "0" * 64}


class _FakeStrict:
    WRITE_NEW_FLAGS = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )

    def read_bytes_nofollow(self, *_args, **_kwargs) -> _FakeRecord:
        return _FakeRecord()

    def read_json_nofollow(self, path: Path, **_kwargs):
        if Path(path).name == runner.CPU4_RESULT.name:
            return {"consumed_tool_definition": {}}, _FakeRecord()
        return {}, _FakeRecord()

    def open_dir_nofollow(self, path: Path) -> int:
        return os.open(
            path,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )

    def ensure_direct_child_dir(self, parent: Path, name: str) -> Path:
        result = parent / name
        result.mkdir()
        return result

    def write_new_json(self, path: Path, value, **_kwargs):
        payload = (json.dumps(value, sort_keys=True) + "\n").encode()
        path.write_bytes(payload)
        return {
            "path": str(path),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }


def _execute_args(tmp_path: Path) -> argparse.Namespace:
    run_root = tmp_path / "_run"
    run_root.mkdir()
    return argparse.Namespace(
        project_root=tmp_path,
        cpu_freeze=run_root / "CPU_FREEZE.json",
        cpu_freeze_sha256="0" * 64,
        cpu_freeze_bytes=1,
        scene_state=run_root / "SCENE_STATE.npz",
        scene_state_sha256="0" * 64,
        scene_state_bytes=1,
        scene_manifest=run_root / "SCENE_MANIFEST.json",
        scene_manifest_sha256="0" * 64,
        scene_manifest_bytes=1,
        output_dir=run_root / "renderer-output",
        frame_start=0,
        frame_count=1,
        width=320,
        height=240,
        timeout_seconds=1.0,
        gpu_lease_bytes=1,
        gpu_lease_sha256="0" * 64,
        gpu_lease_holder="gpt",
        gpu_lease_requester="gpt",
        gpu_lease_task_id="robot-renderer-004-frame0",
        gpu_lease_purpose="004 robot EEVEE candidate frame0 only",
        gpu_lease_generation_token="renderer-generation-token-004-frame0-v1",
    )


def _patch_execute_cpu_preflight(monkeypatch) -> None:
    monkeypatch.setattr(runner, "load_strict_io", lambda _project: _FakeStrict())
    monkeypatch.setattr(
        runner,
        "decode_scene_state",
        lambda _payload: SimpleNamespace(
            session_id="grap_a_cap_004",
            frame_names=("00000",),
            source_resolution=(160, 120),
        ),
    )
    monkeypatch.setattr(
        runner,
        "load_pinned_robot_assets",
        lambda *_args: SimpleNamespace(tool_definition={}),
    )
    monkeypatch.setattr(
        runner, "validate_scene_manifest", lambda *_args, **_kwargs: None
    )

    class FakeHeldWorkerSource:
        execution_descriptor = 123456
        execution_path = "/proc/self/fd/123456"

        @classmethod
        def acquire(cls, **_kwargs):
            return cls()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def revalidate(self, _stage: str):
            return None

        def evidence_ref(self):
            return {
                "path": "/held/worker.py",
                "bytes": 1,
                "sha256": "0" * 64,
                "execution_transport": "SEALED_MEMFD_EXACT_BYTES",
            }

    monkeypatch.setattr(runner, "HeldWorkerSource", FakeHeldWorkerSource)


def test_missing_lease_is_zero_output_zero_nvidia_smi_zero_popen(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_execute_cpu_preflight(monkeypatch)
    args = _execute_args(tmp_path)
    monkeypatch.setattr(
        runner, "_gpu_sample", lambda: pytest.fail("nvidia-smi was reached")
    )
    monkeypatch.setattr(
        runner.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("Popen reached"),
    )
    with pytest.raises(runner.FullChainRendererError, match="cannot be held"):
        runner.execute(args)
    assert not args.output_dir.exists()


def test_missing_pre_child_gpu_observation_is_zero_output_zero_popen(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_execute_cpu_preflight(monkeypatch)
    args = _execute_args(tmp_path)

    class Lease:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def revalidate(self, _stage: str):
            return {}

    class FakeHeldGpuLease:
        @classmethod
        def acquire(cls, **_kwargs):
            return Lease()

    monkeypatch.setattr(runner, "HeldGpuLease", FakeHeldGpuLease)
    monkeypatch.setattr(runner, "_gpu_sample", lambda: None)
    monkeypatch.setattr(
        runner.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("Popen reached"),
    )
    with pytest.raises(
        runner.FullChainRendererError, match="observation is unavailable"
    ):
        runner.execute(args)
    assert not args.output_dir.exists()


def test_pre_popen_lease_race_cleans_empty_output_and_never_spawns(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_execute_cpu_preflight(monkeypatch)
    args = _execute_args(tmp_path)

    class RacingLease:
        calls = 0

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def revalidate(self, stage: str):
            self.calls += 1
            if self.calls == 3:
                raise runner.FullChainRendererError(f"lease replaced at {stage}")
            return {}

    racing = RacingLease()

    class FakeHeldGpuLease:
        @classmethod
        def acquire(cls, **_kwargs):
            return racing

    monkeypatch.setattr(runner, "HeldGpuLease", FakeHeldGpuLease)
    monkeypatch.setattr(runner, "_gpu_sample", lambda: (0.0, 0.0))
    monkeypatch.setattr(
        runner.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("Popen reached"),
    )
    with pytest.raises(runner.FullChainRendererError, match="lease replaced"):
        runner.execute(args)
    assert racing.calls == 3
    assert not args.output_dir.exists()


@pytest.mark.parametrize(
    "revocation_stage",
    (
        "immediately_after_worker_popen",
        "worker_lifecycle_poll",
        "after_worker_gpu_sample",
    ),
)
def test_lease_revocation_at_or_after_popen_terminates_process_group_and_holds(
    tmp_path: Path, monkeypatch, revocation_stage: str
) -> None:
    _patch_execute_cpu_preflight(monkeypatch)
    args = _execute_args(tmp_path)
    stages: list[str] = []

    class RevokedLease:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def revalidate(self, stage: str):
            stages.append(stage)
            if stage == revocation_stage or (
                revocation_stage == "immediately_after_worker_popen"
                and "immediately_after_worker_popen" in stages
            ):
                raise runner.FullChainRendererError(f"lease revoked at {stage}")
            return {}

        def evidence_ref(self):
            return {"scope": {}, "lifecycle_lock": "FLOCK_EXCLUSIVE_HELD"}

    revoked = RevokedLease()

    class FakeHeldGpuLease:
        @classmethod
        def acquire(cls, **_kwargs):
            return revoked

    class FakeProcess:
        pid = 987654
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self, *_args, **_kwargs):
            return self.returncode

    process = FakeProcess()
    terminated: list[int] = []

    def terminate(candidate) -> None:
        terminated.append(candidate.pid)
        candidate.returncode = -15

    monkeypatch.setattr(runner, "HeldGpuLease", FakeHeldGpuLease)
    monkeypatch.setattr(runner, "_gpu_sample", lambda: (0.0, 0.0))
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(runner, "_terminate_process_group", terminate)

    terminal = runner.execute(args)
    assert terminated == [process.pid]
    assert terminal["status"] == "HOLD_WORKER_OR_GPU_PROOF_FAILED"
    assert terminal["gpu_lease_valid_throughout"] is False
    assert terminal["gpu_lease_valid_at_terminal"] is False
    assert terminal["gpu_lease_valid_at_launch"] is (
        revocation_stage != "immediately_after_worker_popen"
    )
    assert revocation_stage in terminal["gpu_lease_failure"]
    assert (args.output_dir / "ORCHESTRATOR_TERMINAL.json").is_file()


def _worker_source_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, dict[str, object], object]:
    project = tmp_path / "project"
    worker_path = project / runner.WORKER
    worker_path.parent.mkdir(parents=True)
    worker_path.write_bytes(b"#!/usr/bin/env python3\nprint('held worker')\n")
    strict = runner.load_strict_io(PROJECT)
    worker_record = strict.read_bytes_nofollow(worker_path, allowed_root=project)
    worker_ref = worker_record.evidence_ref()
    freeze: dict[str, object] = {
        "schema_version": "s7-eevee-next-fullchain-cpu-freeze-v1",
        "status": "CPU_READY_WAITING_IMMUTABLE_SCENE_STATE_NO_GPU_EXECUTED",
        "auth_tier": "T1_CANDIDATE_RENDERER_CPU_PREP",
        "implementation": {
            "core": worker_ref,
            "worker": worker_ref,
            "runner": worker_ref,
            "core_tests": worker_ref,
            "runner_tests": worker_ref,
        },
    }
    return project.resolve(), worker_path, freeze, strict


def test_worker_source_is_exact_cpu_freeze_bytes_on_sealed_memfd(
    tmp_path: Path,
) -> None:
    project, _worker_path, freeze, strict = _worker_source_fixture(tmp_path)
    with runner.HeldWorkerSource.acquire(
        project=project, strict=strict, cpu_freeze=freeze
    ) as held:
        held.revalidate("test")
        assert held.execution_path.startswith("/proc/self/fd/")
        assert held.execution_path != str(project / runner.WORKER)
        assert held.evidence_ref()["execution_transport"] == (
            "SEALED_MEMFD_EXACT_BYTES"
        )
        assert (
            runner.fcntl.fcntl(held.execution_descriptor, runner.fcntl.F_GET_SEALS)
            & runner.WORKER_SOURCE_REQUIRED_SEALS
        ) == runner.WORKER_SOURCE_REQUIRED_SEALS
        with pytest.raises(OSError):
            os.write(held.execution_descriptor, b"mutation")
        result = subprocess.run(
            [sys.executable, held.execution_path],
            pass_fds=(held.execution_descriptor,),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout == "held worker\n"


def test_worker_source_rejects_path_swap_after_same_fd_freeze_join(
    tmp_path: Path,
) -> None:
    project, worker_path, freeze, strict = _worker_source_fixture(tmp_path)
    with runner.HeldWorkerSource.acquire(
        project=project, strict=strict, cpu_freeze=freeze
    ) as held:
        replacement = worker_path.with_name("replacement.py")
        replacement.write_bytes(held.payload_bytes)
        os.replace(replacement, worker_path)
        with pytest.raises(
            runner.FullChainRendererError, match="path identity changed"
        ):
            held.revalidate("source-swap")


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("bytes", True, "types are invalid"),
        ("device", "1", "types are invalid"),
        ("sha256", "A" * 64, "types are invalid"),
        ("path", "/tmp/other-worker.py", "canonical worker"),
    ],
)
def test_worker_source_rejects_wrong_ref_types_or_source(
    tmp_path: Path, field: str, replacement: object, message: str
) -> None:
    project, _worker_path, freeze, strict = _worker_source_fixture(tmp_path)
    worker_ref = freeze["implementation"]["worker"]
    worker_ref[field] = replacement
    with pytest.raises(runner.FullChainRendererError, match=message):
        runner.HeldWorkerSource.acquire(
            project=project, strict=strict, cpu_freeze=freeze
        )


@pytest.mark.parametrize(
    ("width", "height"),
    [
        (319, 240),
        (320, 241),
        (True, 240),
        (320, "240"),
    ],
)
def test_wrong_or_noninteger_resolution_fails_before_gpu_output_and_popen(
    tmp_path: Path, monkeypatch, width: object, height: object
) -> None:
    _patch_execute_cpu_preflight(monkeypatch)
    args = _execute_args(tmp_path)
    args.width = width
    args.height = height
    monkeypatch.setattr(
        runner, "_gpu_sample", lambda: pytest.fail("nvidia-smi was reached")
    )
    monkeypatch.setattr(
        runner.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("Popen reached"),
    )
    with pytest.raises(runner.FullChainRendererError, match="caller (width|height)"):
        runner.execute(args)
    assert not args.output_dir.exists()


def test_omitted_dimensions_derive_exact_2x_and_launch_only_held_memfd(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_execute_cpu_preflight(monkeypatch)
    args = _execute_args(tmp_path)
    args.width = None
    args.height = None

    class Lease:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def revalidate(self, _stage: str):
            return {}

        def evidence_ref(self):
            return {"scope": {}, "lifecycle_lock": "FLOCK_EXCLUSIVE_HELD"}

    class FakeHeldGpuLease:
        @classmethod
        def acquire(cls, **_kwargs):
            return Lease()

    class FinishedProcess:
        pid = 987655
        returncode = 0

        def poll(self):
            return self.returncode

        def wait(self, *_args, **_kwargs):
            return self.returncode

    launched: dict[str, object] = {}

    def popen(command, **kwargs):
        launched["command"] = command
        launched["kwargs"] = kwargs
        return FinishedProcess()

    monkeypatch.setattr(runner, "HeldGpuLease", FakeHeldGpuLease)
    monkeypatch.setattr(runner, "_gpu_sample", lambda: (0.0, 0.0))
    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    terminal = runner.execute(args)
    command = launched["command"]
    assert command[1] == "/proc/self/fd/123456"
    assert str(tmp_path / runner.WORKER) not in command
    assert command[command.index("--width") + 1] == "320"
    assert command[command.index("--height") + 1] == "240"
    assert launched["kwargs"]["pass_fds"] == (123456,)
    assert terminal["source_resolution"] == [160, 120]
    assert terminal["render_resolution"] == [320, 240]


def test_worker_source_race_before_popen_cleans_output_and_never_spawns(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_execute_cpu_preflight(monkeypatch)
    args = _execute_args(tmp_path)

    class RacingWorkerSource:
        execution_descriptor = 123456
        execution_path = "/proc/self/fd/123456"

        @classmethod
        def acquire(cls, **_kwargs):
            return cls()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def revalidate(self, stage: str):
            if stage == "immediately_before_worker_popen":
                raise runner.FullChainRendererError("worker source swapped")

        def evidence_ref(self):
            return {}

    class Lease:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def revalidate(self, _stage: str):
            return {}

    class FakeHeldGpuLease:
        @classmethod
        def acquire(cls, **_kwargs):
            return Lease()

    monkeypatch.setattr(runner, "HeldWorkerSource", RacingWorkerSource)
    monkeypatch.setattr(runner, "HeldGpuLease", FakeHeldGpuLease)
    monkeypatch.setattr(runner, "_gpu_sample", lambda: (0.0, 0.0))
    monkeypatch.setattr(
        runner.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("Popen reached"),
    )
    with pytest.raises(runner.FullChainRendererError, match="source swapped"):
        runner.execute(args)
    assert not args.output_dir.exists()


def test_help_bootstraps_project_import_from_external_cwd_and_empty_pythonpath(
    tmp_path: Path,
) -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = ""
    result = subprocess.run(
        [sys.executable, str(PROJECT / runner.RUNNER), "--help"],
        cwd=tmp_path,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--gpu-lease-generation-token" in result.stdout
