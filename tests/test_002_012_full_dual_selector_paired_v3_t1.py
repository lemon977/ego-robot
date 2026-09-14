import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "task29_paired_runner_v3_test",
    PROJECT / "tools/run_002_012_full_dual_selector_paired_v3_t1.py",
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


class FakeCuda:
    @staticmethod
    def synchronize():
        return None

    @staticmethod
    def empty_cache():
        return None


class FakeTorch:
    cuda = FakeCuda()


def make_adapter(frames, events, *, prompt_output="PROMPT_A"):
    class Adapter:
        prompt_calls = 0
        stream_calls = 0
        close_calls = 0

        def handle_request(self, request):
            events.append(request["type"])
            if request["type"] == "start_session":
                return {"parameter_mapping": {"text": "an arm"}}
            if request["type"] == "add_prompt":
                self.prompt_calls += 1
                return {"outputs": prompt_output}
            if request["type"] == "close_session":
                self.close_calls += 1
                return {}
            raise AssertionError(request)

        def handle_stream_request(self, request):
            self.stream_calls += 1
            events.append("propagation_started")
            for frame, output in frames:
                events.append(f"propagation_item:{frame}")
                yield {"frame_index": frame, "outputs": output}

    return Adapter()


def make_old(events):
    class Conversion:
        @staticmethod
        def outputs_to_instances(outputs, height, width):
            events.append(f"converted:{outputs}")
            return SimpleNamespace(source=outputs, masks=SimpleNamespace(shape=(1, height, width)))

    return SimpleNamespace(base=Conversion())


def run_stream(frames, frame_count):
    events = []
    adapter = make_adapter(frames, events)
    old = make_old(events)
    previous = runner.base.torch
    runner.base.torch = FakeTorch()
    timing = {}
    try:
        rows = list(
            runner.propagation_only_stream(
                adapter,
                old,
                Path("/not-read"),
                session="grap_a_cap_002",
                frame_count=frame_count,
                timing=timing,
            )
        )
    finally:
        runner.base.torch = previous
    return adapter, rows, timing, events


def test_prompt_output_is_never_formal_and_propagation_frame0_is_canonical():
    adapter, rows, timing, events = run_stream(
        [(0, "PROP_B0"), (1, "PROP_B1"), (2, "PROP_B2")], 3
    )
    assert [frame for frame, _ in rows] == [0, 1, 2]
    assert [value.source for _, value in rows] == ["PROP_B0", "PROP_B1", "PROP_B2"]
    assert "converted:PROMPT_A" not in events
    assert events.index("propagation_started") < events.index("converted:PROP_B0")
    assert not any(item.startswith("converted:") for item in events[: events.index("propagation_started")])
    assert adapter.prompt_calls == 1
    assert adapter.stream_calls == 1
    assert adapter.close_calls == 1
    assert timing["formal_frames_before_propagation"] == 0
    assert timing["add_prompt_output_consumed_as_formal_inventory"] is False
    assert timing["formal_inventory_source"] == "COMPLETE_PROPAGATION_STREAM_ONLY"
    assert timing["propagation_frame0_consumed"] is True


def test_add_prompt_and_propagation_frame0_raw_drift_is_out_of_inventory_not_error():
    _, rows, _, events = run_stream([(0, "REFINED_FRAME0")], 1)
    assert rows[0][1].source == "REFINED_FRAME0"
    assert events.count("converted:REFINED_FRAME0") == 1
    assert "converted:PROMPT_A" not in events


@pytest.mark.parametrize(
    "frames,count,message",
    [
        ([(0, "A"), (0, "B")], 2, "noncontiguous propagation frame"),
        ([(0, "A"), (2, "B")], 2, "noncontiguous propagation frame"),
        ([(0, "A")], 2, "propagation frame count drift"),
    ],
)
def test_duplicate_gap_and_short_propagation_fail_closed(frames, count, message):
    with pytest.raises(runner.LaunchV3Error, match=message):
        run_stream(frames, count)


def test_exact_sequence_has_no_n_plus_one_conversion_and_preserves_object_identity():
    outputs = [object(), object(), object(), object()]
    _, rows, timing, events = run_stream(list(enumerate(outputs)), 4)
    assert len(rows) == 4
    assert sum(item.startswith("converted:") for item in events) == 4
    assert all(row[1].source is outputs[index] for index, row in enumerate(rows))
    assert timing["frame_count"] == 4
    assert timing["instance_counts"] == [1, 1, 1, 1]


def exact_qa(pins):
    return {
        "schema_version": "task29-independent-cpu-qa-v3",
        "status": "PASS_TASK29_V3_CPU_QA_P0_ZERO_GPU_ADMISSION_ALLOWED",
        "producer": "INDEPENDENT_QA",
        "AUTH_TIER": "T1_INDEPENDENT_CPU_QA",
        "WHY_NOT_BLOCKED": "READ_ONLY_QA_OF_FROZEN_TASK29_V3_LAUNCH_WRAPPER",
        "p0_findings": 0,
        "p1_findings": 0,
        "access_counters": {key: 0 for key in runner.base.QA_ACCESS_KEYS},
        "frozen_refs": pins,
        "checks": {},
        "findings": [],
        "claim_limit": "CPU_QA_ONLY_GPU_ADMISSION_ALLOWED_CANDIDATE_STILL_UNREVIEWED",
        "created_at": "2026-08-28T20:00:00+08:00",
    }


def test_v3_governance_accepts_only_exact_qa(monkeypatch):
    pins = {"runner": {"path": "/r", "bytes": 1, "sha256": "a" * 64}}
    qa = exact_qa(pins)
    monkeypatch.setattr(runner.base, "context_helper", lambda _project: object())
    monkeypatch.setattr(runner.base, "verified_json", lambda *_args: (qa, pins["runner"]))
    assert runner.require_governance_v3(PROJECT, pins)["qa"] is qa

    attacks = []
    for key, value in (
        ("schema_version", "task29-independent-cpu-qa-v2"),
        ("status", "PASS_ATTACKER_PREFIX"),
        ("p0_findings", False),
    ):
        attacked = exact_qa(pins)
        attacked[key] = value
        attacks.append(attacked)
    attacked = exact_qa(pins)
    attacked["unknown"] = 1
    attacks.append(attacked)
    for attacked in attacks:
        monkeypatch.setattr(
            runner.base,
            "verified_json",
            lambda *_args, value=attacked: (value, pins["runner"]),
        )
        with pytest.raises(runner.LaunchV3Error):
            runner.require_governance_v3(PROJECT, pins)


class ReleaseHelper:
    @staticmethod
    def identity(path):
        data = Path(path).read_bytes()
        import hashlib

        return {
            "path": str(Path(path).resolve()),
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }

    def exclusive_json(self, path, payload):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o640)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)


def terminal_tree(root, run_id, *, success):
    run_root = root / "_run" / run_id
    component = run_root / runner.COMPONENT_NAME
    component.mkdir(parents=True)
    (run_root / "GPU_ADMISSION.json").write_text("{}", encoding="utf-8")
    name = "RUN_MANIFEST.json" if success else "FAILURE_MANIFEST.json"
    status = (
        "COMPLETED_UNREVIEWED_AWAITING_INDEPENDENT_QA"
        if success
        else "HOLD_EXECUTION_FAILED_PARTIAL_PRESERVED"
    )
    (component / name).write_text(json.dumps({"status": status}), encoding="utf-8")
    return run_root


def patch_release_environment(monkeypatch, root, processes):
    helper = ReleaseHelper()
    monkeypatch.setattr(runner, "PROJECT", root)
    monkeypatch.setattr(runner.base, "context_helper", lambda _project: helper)
    monkeypatch.setattr(
        runner.base,
        "verified_json",
        lambda helper_arg, path: (json.loads(path.read_text()), helper.identity(path)),
    )
    monkeypatch.setattr(runner, "gpu_process_snapshot", lambda: (processes, "nvidia-smi frozen"))


def test_release_cannot_unlock_while_any_gpu_process_remains(tmp_path, monkeypatch):
    run_id = "v3-self-present"
    terminal_tree(tmp_path, run_id, success=True)
    patch_release_environment(
        monkeypatch,
        tmp_path,
        [{"pid": os.getpid(), "used_memory_mib": 1}],
    )
    release = runner.finalize_gpu_release(run_id, 0)
    assert release["schema_version"] == "task29-gpu-release-v3"
    assert release["gpu_child_has_exited"] is True
    assert release["gpu_processes_remaining"] == 1
    assert release["unlocking_p2"] is False
    assert release["status"] == "GPU_RELEASE_PENDING_PROCESSES"


def test_post_child_exit_zero_process_success_can_unlock(tmp_path, monkeypatch):
    run_id = "v3-success"
    run_root = terminal_tree(tmp_path, run_id, success=True)
    patch_release_environment(monkeypatch, tmp_path, [])
    release = runner.finalize_gpu_release(run_id, 0)
    assert release["release_observation_phase"] == "OUTER_LAUNCHER_AFTER_GPU_CHILD_EXIT"
    assert release["successful_p1_terminal"] is True
    assert release["gpu_processes_remaining"] == 0
    assert release["unlocking_p2"] is True
    assert (run_root / "GPU_RELEASE.json").is_file()


def test_failure_release_never_unlocks_even_after_zero_process(tmp_path, monkeypatch):
    run_id = "v3-failure"
    terminal_tree(tmp_path, run_id, success=False)
    patch_release_environment(monkeypatch, tmp_path, [])
    release = runner.finalize_gpu_release(run_id, 2)
    assert release["successful_p1_terminal"] is False
    assert release["gpu_processes_remaining"] == 0
    assert release["unlocking_p2"] is False
    assert release["status"] == "GPU_RELEASED_AFTER_FAILURE"


def test_frozen_v2_semantics_and_versioned_component_are_exact():
    assert runner.file_sha(PROJECT / runner.BASE_REL) == (runner.BASE_BYTES, runner.BASE_SHA)
    constants = runner.base.run.__code__.co_consts
    assert runner.COMPONENT_NAME in constants
    assert "dual_selector_paired_v2" not in constants
    assert runner.base.SELECTORS == ("old_v3", "task26")
    assert runner.base.MAX_RUN_BYTES == int(1.5 * 1024**3)
    assert runner.base.OLD_MODULE_SHA == "694a9bc3b24bae3e36eda8648befc22db690ac4c515c8251985c2d57e81f2e5f"
    assert runner.base.NEW_MODULE_SHA == "bca51f866de925a2e795c5257540abc9c06fb8d298f26dd049196fda14fb9ff3"
