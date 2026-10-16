"""Bounded CPU regressions for the two-lane joint26 process supervisor."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import importlib.util
import json
import os
from pathlib import Path
import select
import shlex
import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ATTEMPT_ROOT = PROJECT_ROOT / "_run/maintenance/joint26_parallel_20261014/attempt_0001"

_CHILD = r"""
import json
import os
import sys
import time

print(json.dumps({"event": "start", "time": time.monotonic(),
                  "pid": os.getpid(), "pgid": os.getpgrp()}), flush=True)
time.sleep(float(sys.argv[1]))
print(json.dumps({"event": "end", "time": time.monotonic()}), flush=True)
raise SystemExit(int(sys.argv[2]))
"""


@pytest.fixture(scope="module")
def runner():
    path = PROJECT_ROOT / "scripts/humanego_ict/joint26_parallel_runner_v1.py"
    spec = importlib.util.spec_from_file_location("_joint26_parallel_runner_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stop_group(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=2)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


@pytest.fixture
def child_processes(runner, monkeypatch):
    """Keep assertion failures from leaking any real CPU fixture process."""
    created = []
    original_popen = subprocess.Popen

    def record_popen(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        created.append(process)
        return process

    monkeypatch.setattr(runner.subprocess, "Popen", record_popen)
    yield created
    for process in created:
        _stop_group(process)


def _launches(tmp_path, *, raw_delay=0.4, robotized_delay=0.4, raw_exit=0):
    assert tmp_path.resolve().is_relative_to(ATTEMPT_ROOT.resolve()), (
        "ENGINEERING_ATTEMPT_BASETEMP_REQUIRED"
    )
    environment = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        TMPDIR=str(tmp_path),
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        CUDA_VISIBLE_DEVICES="",
    )
    return {
        branch: {
            "command": [sys.executable, "-B", "-c", _CHILD, str(delay), str(exitcode)],
            "env": environment,
            "cwd": tmp_path,
            "log_path": tmp_path / f"{branch}.log",
        }
        for branch, delay, exitcode in (
            ("raw", raw_delay, raw_exit),
            ("robotized", robotized_delay, 0),
        )
    }


def _events(launch):
    return [json.loads(line) for line in launch["log_path"].read_text().splitlines()]


def _supervise(runner, launches, *, deadline=None, signals=None,
               on_heartbeat=None, on_finish=None):
    finishes = []
    stopped = []
    heartbeats = []

    def heartbeat(active):
        assert all(type(pid) is int and pid > 0 for pid in active.values())
        heartbeats.append(dict(active))
        if on_heartbeat is not None:
            on_heartbeat(active)

    def stop(process):
        if process.poll() is None:
            stopped.append(process.pid)
        _stop_group(process)

    def finish(branch, returncode):
        finishes.append((branch, returncode))
        if on_finish is not None:
            on_finish(branch, returncode)

    result = runner.supervise_lanes(
        launches,
        deadline=deadline or datetime.now(timezone.utc) + timedelta(seconds=5),
        on_heartbeat=heartbeat,
        on_finish=finish,
        stop_child=stop,
        signal_state=[] if signals is None else signals,
        heartbeat_interval=0.01,
    )
    return result, finishes, stopped, heartbeats


def test_real_children_start_before_either_lane_finishes(runner, child_processes, tmp_path):
    launches = _launches(tmp_path)
    result, finishes, stopped, heartbeats = _supervise(runner, launches)

    observations = {branch: _events(launch) for branch, launch in launches.items()}
    starts = [events[0] for events in observations.values()]
    ends = [events[-1] for events in observations.values()]
    assert all(event["event"] == "start" for event in starts)
    assert all(event["event"] == "end" for event in ends)
    assert max(event["time"] for event in starts) < min(event["time"] for event in ends)
    assert all(event["pid"] == event["pgid"] for event in starts)
    assert {branch: item["returncode"] for branch, item in result.items()} == {
        "raw": 0, "robotized": 0
    }
    assert sorted(finishes) == [("raw", 0), ("robotized", 0)]
    assert not stopped
    assert any(set(active) == {"raw", "robotized"} for active in heartbeats)
    assert len(child_processes) == 2
    assert all(process.poll() == 0 for process in child_processes)


def test_one_failed_lane_does_not_stop_its_running_peer(runner, child_processes, tmp_path):
    launches = _launches(tmp_path, raw_delay=0.15, robotized_delay=0.4, raw_exit=7)
    result, finishes, stopped, heartbeats = _supervise(runner, launches)

    raw_events = _events(launches["raw"])
    robotized_events = _events(launches["robotized"])
    assert robotized_events[-1]["event"] == "end"
    assert raw_events[-1]["time"] < robotized_events[-1]["time"]
    assert result["raw"]["returncode"] == 7
    assert result["robotized"]["returncode"] == 0
    assert finishes == [("raw", 7), ("robotized", 0)]
    assert not stopped
    assert any(set(active) == {"robotized"} for active in heartbeats)
    assert all(process.poll() is not None for process in child_processes)


def test_second_spawn_failure_cleans_up_started_lane(runner, child_processes, tmp_path):
    launches = _launches(tmp_path, raw_delay=3)
    launches["robotized"]["command"] = [str(tmp_path / "missing-executable")]
    result, finishes, stopped, _ = _supervise(runner, launches)

    assert set(result) == {"raw", "robotized"}
    assert result["robotized"]["returncode"] != 0
    assert result["robotized"]["errors"]
    assert len(child_processes) == 1
    assert stopped == [child_processes[0].pid]
    assert child_processes[0].poll() is not None
    assert {branch for branch, _ in finishes} == {"raw", "robotized"}


@pytest.mark.parametrize("reason", ["deadline", "signal"])
def test_global_stop_terminates_both_process_groups(runner, child_processes, tmp_path, reason):
    launches = _launches(tmp_path, raw_delay=3, robotized_delay=3)
    signals = []

    def heartbeat(active):
        if reason == "signal" and set(active) == {"raw", "robotized"}:
            signals.append(signal.SIGTERM)

    deadline = datetime.now(timezone.utc) + timedelta(
        seconds=0.15 if reason == "deadline" else 5
    )
    result, finishes, stopped, _ = _supervise(
        runner, launches, deadline=deadline, signals=signals, on_heartbeat=heartbeat
    )

    assert len(child_processes) == 2
    assert set(stopped) == {process.pid for process in child_processes}
    assert all(process.poll() is not None for process in child_processes)
    assert all(item["returncode"] != 0 for item in result.values())
    assert {branch for branch, _ in finishes} == {"raw", "robotized"}
    assert len(finishes) == 2


@pytest.mark.parametrize("callback", ["heartbeat", "finish"])
def test_callback_failure_stops_remaining_children(runner, child_processes, tmp_path, callback):
    launches = _launches(tmp_path, raw_delay=0.15, robotized_delay=3)

    def heartbeat(_active):
        if callback == "heartbeat":
            raise RuntimeError("synthetic heartbeat failure")

    def finish(branch, _returncode):
        if callback == "finish" and branch == "raw":
            raise RuntimeError("synthetic finish failure")

    result, finishes, stopped, _ = _supervise(
        runner, launches, on_heartbeat=heartbeat, on_finish=finish
    )

    assert len(child_processes) == 2
    assert all(process.poll() is not None for process in child_processes)
    assert child_processes[1].pid in stopped
    if callback == "heartbeat":
        assert set(stopped) == {process.pid for process in child_processes}
    else:
        assert result["raw"]["returncode"] == 0
    assert any(callback.upper() in str(error)
               for row in result.values() for error in row["errors"])
    assert {branch for branch, _ in finishes} == {"raw", "robotized"}


def test_owned_group_cleanup_stops_descendant_after_parent_exit(runner, child_processes, tmp_path):
    environment = _launches(tmp_path)["raw"]["env"]
    command = (
        "import subprocess, sys; "
        "child = subprocess.Popen([sys.executable, '-B', '-c', "
        "'import time; time.sleep(3)']); "
        "print(child.pid, flush=True)"
    )
    process = subprocess.Popen(
        [sys.executable, "-B", "-c", command], cwd=tmp_path, env=environment,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True
    )
    assert int(process.stdout.readline().strip()) > 0
    assert process.wait(timeout=2) == 0
    # The descendant keeps the pipe open after its parent exits. This observes
    # the real child without assuming /proc has the same PID namespace.
    assert not select.select([process.stdout], [], [], 0)[0]

    runner.stop_owned_group(process)

    stdout, stderr = process.communicate(timeout=2)
    assert stdout == "" and stderr == ""
    assert len(child_processes) == 1


@pytest.fixture
def coordinator(runner, monkeypatch, tmp_path):
    """Execute real coordinator publication logic against only a private ledger."""
    from chaoyang.humanego_ict import joint26_parallel_resources as resources
    from chaoyang.humanego_ict import joint26_runtime as runtime
    from chaoyang.humanego_ict import joint26_training as training

    assert tmp_path.resolve().is_relative_to(ATTEMPT_ROOT.resolve())
    root = tmp_path / "synthetic_project"
    root.mkdir()
    monkeypatch.setattr(runner, "ROOT", root)
    publications, preflights, configs, supervisors, pools, postchecks = [], [], [], [], [], []
    cache_keys = (
        "TMPDIR", "TMP", "TEMP", "TORCH_HOME", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR",
        "XDG_CACHE_HOME", "NUMBA_CACHE_DIR", "CUDA_CACHE_PATH", "TORCH_EXTENSIONS_DIR",
        "HF_HOME", "HUGGINGFACE_HUB_CACHE", "MPLCONFIGDIR",
    )
    for key in cache_keys:
        # execute updates these keys itself; register their original values for
        # restoration before entering the real coordinator.
        monkeypatch.setenv(key, str(root / "cache"))

    def dump(path, value):
        path = Path(path)
        assert path.resolve().is_relative_to(root.resolve())
        path.parent.mkdir(parents=True, exist_ok=True)
        data = runner.canonical_bytes(value)
        if path.exists():
            assert path.read_bytes() == data, "SYNTHETIC_IMMUTABLE_FILE_CONFLICT"
        else:
            path.write_bytes(data)
        return runner.artifact_ref(path)

    closure = [dump(root / "synthetic_runtime.json", {"synthetic_cpu_only": True})]
    contract_ref = dump(root / "contract.json", {"special_status_contracts": {
        runner.PARALLEL_ROUTE: {"code_closure": closure}
    }})
    manifest = dump(root / "manifest.json", {"synthetic_cpu_only": True})
    handoff_path = root / "docs/plans/PICK_PLACE_HUMANEGO_ICT_JOINT26_V1/STATUS.json"
    old_raw = dump(root / "previous_raw_checkpoint.json", {"synthetic_previous": "raw"})
    old_robotized = dump(root / "previous_robotized_checkpoint.json", {"synthetic_previous": "robotized"})
    dump(handoff_path, {"raw_joint26_checkpoint": old_raw,
                        "robotized_joint26_checkpoint": old_robotized,
                        "optimizer_updates": {"raw": 1, "robotized": 1}})
    (root / "_run/current").mkdir(parents=True)

    class PrivatePublisher:
        def __init__(self):
            self.state = {"tasks": [], "next_task": None}
            self.index = {"task_packets": []}
            self.revision = 10

        def _snapshot(self):
            receipt = {"governance_revision": self.revision,
                       "files": {"algorithm_contract": contract_ref}}
            return receipt, deepcopy(self.state), deepcopy(self.index)

        def _no_active_writer(self, state, index):
            assert not any(row["status"] == "RUNNING" for row in state["tasks"])

        def _writer_identity(self, task_id, attempt):
            assert attempt == 1
            return {"pid": os.getpid(), "proc_start_ticks": 123, "executor_epoch": 1,
                    "fencing_token": "synthetic_fencing_token", "scope": task_id}

        def _existing_helpers(self):
            return (lambda directory, task_id: {"synthetic_cpu_capacity": True}, None)

        def _environment(self, lane_dir):
            cache = lane_dir / "cache"
            cache.mkdir()
            return {**{key: str(cache) for key in cache_keys},
                    "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}

        def _json_once(self, path, document):
            return dump(path, document)

        def _verify(self, ref):
            assert runner.artifact_ref(Path(ref["path"])) == ref

        def _own_row(self, packet, state, index):
            rows = [row for row in state["tasks"] if row["task_id"] == packet["task_id"]]
            pointers = [row for row in index["task_packets"] if row["task_id"] == packet["task_id"]]
            assert len(rows) == len(pointers) == 1
            return rows[0], pointers[0]

        def _publish(self, state, index, event, *, expected_revision):
            assert expected_revision == self.revision
            self.state, self.index = deepcopy(state), deepcopy(index)
            publications.append((event, deepcopy(state), deepcopy(index)))
            self.revision += 1

    publisher = PrivatePublisher()

    def resolve_config(args):
        configs.append(args)
        cfg = SimpleNamespace(max_updates=args.steps, global_batch=64,
                              micro_batch=64 // args.nproc, validation_every=2)
        return cfg, {"schema": "synthetic-training-config", "training": vars(cfg)}

    bootstrap = SimpleNamespace(
        resolve_training_config=resolve_config,
        _preflight=lambda command: preflights.append(tuple(command)),
        _infra=lambda: publisher,
    )
    monkeypatch.setitem(sys.modules, "joint26_runner_v1", bootstrap)
    monkeypatch.setattr(runner, "candidate_contract", lambda project: {
        runner.PARALLEL_ROUTE: {"code_closure": closure}
    })
    monkeypatch.setattr(runner, "process_identity", lambda pid: {"alive": True, "start_ticks": 123})

    def ordinary(path, existing=True):
        path = Path(path).absolute()
        assert path.resolve().is_relative_to(root.resolve())
        if existing:
            assert path.exists()
        return path

    monkeypatch.setattr(runtime, "_ordinary", ordinary)

    def preflight(path, branch, *, world_size):
        preflights.append((branch, world_size))
        assert path == Path(manifest["path"])
        return ["synthetic_train"] * 3, ["synthetic_validation"] * 2, {"synthetic_cpu_only": True}

    monkeypatch.setattr(training, "preflight", preflight)

    class FakeLeases:
        def __init__(self, project, writer, task_id, groups, *, guard_expires_at=None):
            assert project == root
            if guard_expires_at is not None:
                current = datetime.now(timezone.utc)
                assert isinstance(guard_expires_at, datetime)
                assert guard_expires_at.utcoffset() is not None
                assert current < guard_expires_at <= current + timedelta(seconds=43200 + 120)
            self.guard_expires_at = guard_expires_at
            self.groups = deepcopy(groups)
            self.lease_paths = {
                gpu_id: root / f"_run/current/ICT_GPU_LEASES/gpu_{gpu_id}.json"
                for gpu_ids in groups.values() for gpu_id in gpu_ids
            }
            self.central_path = root / "_run/current/GPU_LEASE.json"
            self.calls = []
            pools.append(self)

        def acquire(self):
            self.calls.append("acquire")

        def beat(self, worker_pids):
            assert set(worker_pids) == {"raw", "robotized"}
            self.calls.append("beat")

        def release_all(self, reason):
            assert reason == "JOINT26_PARALLEL_FINITE_EXIT"
            self.calls.append("release")
            if context.release_error:
                raise RuntimeError("synthetic release failure")

    monkeypatch.setattr(resources, "ParallelGpuLeases", FakeLeases)

    def postcheck(command, **kwargs):
        assert kwargs["cwd"] == root
        postchecks.append(command)
        return subprocess.CompletedProcess(command, 0, "synthetic governance validation", "")

    monkeypatch.setattr(runner.subprocess, "run", postcheck)
    args = runner.parse_args([
        "--parallel", "--manifest", manifest["path"], "--raw-gpu-ids", "0,1",
        "--robotized-gpu-ids", "4", "--raw-steps", "3", "--robotized-steps", "5",
        "--wall-seconds", "900", "--output", str(root / "output"),
    ])
    context = SimpleNamespace(
        root=root, args=args, publisher=publisher, publications=publications,
        configs=configs, preflights=preflights, supervisors=supervisors, pools=pools,
        postchecks=postchecks, handoff_path=handoff_path, old_raw=old_raw,
        old_robotized=old_robotized, raw_exit=0, release_error=False,
    )

    def supervise(launches, **callbacks):
        supervisors.append(deepcopy(launches))
        assert set(launches) == {"raw", "robotized"}
        active = publisher.state["tasks"]
        assert len(active) == 1 and active[0]["status"] == "RUNNING"
        assert len(publisher.index["task_packets"]) == 1
        assert publisher.index["task_packets"][0]["execution_allowed"] is True
        callbacks["on_heartbeat"]({"raw": 501, "robotized": 502})
        rows = {}
        for branch, updates, rc in (("raw", 3, context.raw_exit), ("robotized", 5, 0)):
            output = root / "output/lanes" / branch / "output"
            if rc == 0:
                checkpoint = dump(output / f"checkpoint_{updates:06d}.pt", {
                    "branch": branch, "synthetic_cpu_only": True
                })
                tensors = dump(output / "inference_tensors/synthetic_tensors.json", {
                    "synthetic_cpu_only": True
                })
                dump(output / "TRAINING_RESULT.json", {
                    "branch": branch, "optimizer_updates": updates,
                    "requested_optimizer_updates": updates, "completed_update_budget": True,
                    "last_checkpoint": checkpoint["path"],
                })
                dump(output / "inference_tensors/EXPORT_MANIFEST.json", {
                    "optimizer_updates": updates, "requested_optimizer_updates": updates,
                    "files": {"synthetic_tensors": tensors},
                })
            callbacks["on_finish"](branch, rc)
            rows[branch] = {"returncode": rc, "errors": []}
        return rows

    monkeypatch.setattr(runner, "supervise_lanes", supervise)
    return context


@pytest.mark.parametrize("scenario", ["passed", "raw_exit", "lease_release"])
def test_coordinator_seals_one_parent_and_preserves_independent_lane_results(runner, coordinator, scenario):
    from chaoyang.humanego_ict.joint26_parallel_admission import lane_signature

    coordinator.raw_exit = 7 if scenario == "raw_exit" else 0
    coordinator.release_error = scenario == "lease_release"
    returncode = runner.execute(coordinator.args)
    output = coordinator.root / "output"
    packet_paths = list((coordinator.root / "tasks/current").glob("*/TASK_PACKET.json"))
    assert len(packet_paths) == 1
    packet = runner.load_json(packet_paths[0])
    assert packet["route_name"] == runner.PARALLEL_ROUTE
    assert packet["execution_mode"] == runner.PARALLEL_MODE
    assert packet["allocated_gpu_ids"] == [0, 1, 4]
    assert packet["budgets"]["gpu_count"] == 3
    assert packet["budgets"]["optimizer_updates_max"] == 8
    assert packet["control_ground_truth"] is False
    assert packet["physical_deployment_authorized"] is False
    assert packet["run_signature"] == runner.hashlib.sha256(runner.canonical_bytes({
        key: value for key, value in packet.items() if key != "run_signature"
    })).hexdigest()
    assert len(coordinator.supervisors) == 1
    launches = coordinator.supervisors[0]
    commands = [shlex.split(command) for command in packet["commands"]]
    assert len(commands) == 2
    assert len({command[command.index("--output") + 1] for command in commands}) == 2
    for branch, gpu_ids, nproc in (("raw", [0, 1], 2), ("robotized", [4], 1)):
        lane = packet["lanes"][branch]
        assert lane["gpu_ids"] == gpu_ids
        assert lane["lane_signature"] == lane_signature(lane)
        assert launches[branch]["env"]["CUDA_VISIBLE_DEVICES"] == ",".join(map(str, gpu_ids))
        assert "--nproc_per_node=" + str(nproc) in launches[branch]["command"]
        assert launches[branch]["command"][-1] == branch
        pin = shlex.split(lane["commands"][0])
        for command in (pin, launches[branch]["command"]):
            assert command[command.index("--task-packet") + 1] == str(packet_paths[0])
            assert command[command.index("--manifest") + 1] == str(coordinator.args.manifest)
            assert command[command.index("--config") + 1] == lane["inputs"]["training_config"]["path"]
            assert command[command.index("--output") + 1] == lane["output_dir"]
        assert lane["output_dir"] == str(output / "lanes" / branch / "output")
        assert lane["budgets"]["optimizer_updates_max"] == (3 if branch == "raw" else 5)
    assert [config.nproc for config in coordinator.configs] == [2, 1]
    assert coordinator.pools[0].guard_expires_at == (
        datetime.fromisoformat(packet["deadline_at"]) + timedelta(seconds=120)
    )
    assert coordinator.pools[0].calls == ["acquire", "beat", "release"]
    assert len(coordinator.postchecks) == 1
    result = runner.load_json(output / "RESULT.json")
    expected_status = "PASSED" if scenario == "passed" else "FAILED_RUNTIME_FINAL"
    assert result["status"] == expected_status
    assert returncode == (0 if scenario == "passed" else 2)
    assert set(result["lanes"]) == {"raw", "robotized"}
    raw = runner.load_json(Path(result["lanes"]["raw"]["path"]))
    robotized = runner.load_json(Path(result["lanes"]["robotized"]["path"]))
    assert raw["status"] == ("FAILED_RUNTIME_FINAL" if scenario == "raw_exit" else "PASSED")
    assert raw["returncode"] == coordinator.raw_exit
    assert robotized["status"] == "PASSED" and robotized["artifacts"]
    if scenario == "lease_release":
        assert any(error.startswith("LEASE_RELEASE:") for error in result["errors"])
    handoff = runner.load_json(coordinator.handoff_path)
    assert handoff["last_task_status"] == expected_status
    assert handoff["last_parallel_lane_results"] == result["lanes"]
    assert handoff["robotized_joint26_checkpoint"]["path"].startswith(str(output / "lanes/robotized/output"))
    assert handoff["optimizer_updates"]["robotized"] == 5
    if scenario == "raw_exit":
        assert handoff["raw_joint26_checkpoint"] == coordinator.old_raw
        assert handoff["optimizer_updates"]["raw"] == 1
    else:
        assert handoff["raw_joint26_checkpoint"]["path"].startswith(str(output / "lanes/raw/output"))
    assert len(coordinator.publisher.state["tasks"]) == 1
    assert coordinator.publisher.state["tasks"][0]["status"] == expected_status
    pointer = coordinator.publisher.index["task_packets"][0]
    assert pointer["execution_allowed"] is False
    assert pointer["execution_class"] == "TERMINAL_LEDGER_EVIDENCE"
    assert [event for event, _, _ in coordinator.publications] == [
        "ICT_JOINT26_PARALLEL_REGISTERED", "ICT_JOINT26_PARALLEL_HEARTBEAT",
        "ICT_JOINT26_PARALLEL_SEALED",
    ]


def test_coordinator_rejects_overlapping_cli_gpu_groups_before_publication(runner, coordinator):
    coordinator.args.robotized_gpu_ids = "1"
    with pytest.raises(ValueError, match="DISJOINT_PHYSICAL_GPU_GROUPS_REQUIRED"):
        runner.execute(coordinator.args)
    assert not coordinator.supervisors
    assert not coordinator.publications
    assert not coordinator.preflights
    assert not (coordinator.root / "output").exists()
