from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib
import builtins
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest

from pipeline import sam31_compat_adapter_v1 as compat

from pipeline.sam31_compat_adapter_v1 import (
    Sam31CompatAdapterV1,
    Sam31CompatContractError,
    Sam31PinnedIdentity,
    validate_pinned_signatures,
)
from tools.run_sam31_compat_adapter_smoke import prepare_owned_run


def _git(*args: str, cwd: Path) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def _canonical_identity_fixture(tmp_path: Path) -> dict[str, object]:
    project = tmp_path / "project"
    project.mkdir()
    _git("init", "-q", cwd=project)
    _git("config", "user.email", "test@example.invalid", cwd=project)
    _git("config", "user.name", "Test", cwd=project)
    (project / "parent.txt").write_text("enclosing-project\n", encoding="utf-8")
    _git("add", "parent.txt", cwd=project)
    _git("commit", "-qm", "parent", cwd=project)
    parent_head = _git("rev-parse", "HEAD", cwd=project)

    source = project / "_run/official"
    source.mkdir(parents=True)
    _git("init", "-q", cwd=source)
    _git("config", "user.email", "test@example.invalid", cwd=source)
    _git("config", "user.name", "Test", cwd=source)
    tracked = source / "sam3/model_builder.py"
    tracked.parent.mkdir(parents=True)
    tracked.write_text("VALUE = 'official'\n", encoding="utf-8")
    _git("add", "sam3/model_builder.py", cwd=source)
    _git("commit", "-qm", "official", cwd=source)
    official_head = _git("rev-parse", "HEAD", cwd=source)
    official_tree = _git("rev-parse", "HEAD^{tree}", cwd=source)

    canonical = project / "third_party/SAM3"
    canonical_file = canonical / "sam3/model_builder.py"
    canonical_file.parent.mkdir(parents=True)
    os.link(tracked, canonical_file)
    assert _git("rev-parse", "HEAD", cwd=canonical) == parent_head
    assert parent_head != official_head

    runtime_path = project / compat.CANONICAL_RUNTIME_CONTRACT_RELATIVE
    runtime_path.parent.mkdir(parents=True)
    runtime = {
        "status": "CANONICAL_SOURCE_CONTRACT",
        "official_code": {
            "canonical_root": "third_party/SAM3",
            "source_root": "_run/official",
            "commit": official_head,
            "git_tree": official_tree,
            "tracked_regular_files": 1,
            "tracked_regular_bytes": tracked.stat().st_size,
            "nested_git_directory_included": False,
            "materialization": "SAME_FILESYSTEM_HARDLINK_EACH_GIT_TRACKED_REGULAR_FILE",
        },
    }
    runtime_path.write_text(json.dumps(runtime, sort_keys=True), encoding="utf-8")
    runtime_sha = hashlib.sha256(runtime_path.read_bytes()).hexdigest()
    receipt_path = project / compat.CANONICAL_SWITCH_RECEIPT_RELATIVE
    receipt_path.parent.mkdir(parents=True)
    code_record = {
        "canonical_root": "third_party/SAM3",
        "source_root": "_run/official",
        "commit": official_head,
        "git_tree": official_tree,
        "tracked_regular_files": 1,
        "tracked_regular_bytes": tracked.stat().st_size,
        "git_status_clean": True,
        "canonical_hardlink_mismatches": [],
    }
    receipt = {
        "status": "COMPLETE_QA_PASS",
        "runtime_candidate_sha256": runtime_sha,
        "operations": {
            "canonical_code": {
                "target": code_record["canonical_root"],
                "source": code_record["source_root"],
                "commit": official_head,
                "git_tree": official_tree,
                "files": 1,
                "bytes": tracked.stat().st_size,
                "materialization": "SAME_FILESYSTEM_HARDLINK_EACH_GIT_TRACKED_REGULAR_FILE",
                "source_moved_or_modified": False,
            }
        },
        "switch_qa": {"official_code": code_record},
    }
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    receipt_sha = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    return {
        "project": project,
        "canonical": canonical,
        "tracked": tracked,
        "runtime": runtime_path,
        "runtime_sha": runtime_sha,
        "receipt": receipt_path,
        "receipt_sha": receipt_sha,
        "official_head": official_head,
        "official_tree": official_tree,
        "tracked_bytes": tracked.stat().st_size,
        "parent_head": parent_head,
    }


def _verify_canonical_fixture(fixture: dict[str, object]) -> dict[str, object]:
    return compat.verify_canonical_source_identity(
        fixture["canonical"],
        fixture["project"],
        expected_runtime_contract_sha256=fixture["runtime_sha"],
        expected_switch_receipt_sha256=fixture["receipt_sha"],
        expected_commit=fixture["official_head"],
        expected_tree=fixture["official_tree"],
        expected_tracked_files=1,
        expected_tracked_bytes=fixture["tracked_bytes"],
    )


def test_canonical_identity_ignores_enclosing_project_head(tmp_path: Path) -> None:
    fixture = _canonical_identity_fixture(tmp_path)
    assert fixture["parent_head"] != fixture["official_head"]
    evidence = _verify_canonical_fixture(fixture)
    assert evidence["official_code_commit"] == fixture["official_head"]
    assert evidence["enclosing_project_git_head_used"] is False
    assert evidence["all_tracked_git_blobs_verified"] is True
    assert evidence["all_canonical_files_same_source_inode"] is True


@pytest.mark.parametrize("target", ["runtime", "receipt"])
def test_canonical_identity_rejects_contract_or_receipt_sha_drift(
    tmp_path: Path, target: str
) -> None:
    fixture = _canonical_identity_fixture(tmp_path)
    path = fixture[target]
    assert isinstance(path, Path)
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(Sam31CompatContractError, match="provenance SHA mismatch"):
        _verify_canonical_fixture(fixture)


def test_canonical_identity_rejects_tracked_file_drift(tmp_path: Path) -> None:
    fixture = _canonical_identity_fixture(tmp_path)
    tracked = fixture["tracked"]
    assert isinstance(tracked, Path)
    tracked.write_text("VALUE = 'drift'\n", encoding="utf-8")
    with pytest.raises(Sam31CompatContractError, match="Git blob mismatch"):
        _verify_canonical_fixture(fixture)


class Sam3MultiplexTrackingWithInteractivity:
    def __init__(self) -> None:
        self.init_calls = []
        self.prompt_calls = []
        self.propagate_calls = []
        self.fail_init = False

    def init_state(
        self,
        resource_path,
        offload_video_to_cpu=False,
        async_loading_frames=False,
        use_torchcodec=False,
        use_cv2=False,
        input_is_mp4=False,
    ):
        call = {
            "resource_path": resource_path,
            "offload_video_to_cpu": offload_video_to_cpu,
            "async_loading_frames": async_loading_frames,
        }
        self.init_calls.append(call)
        if self.fail_init:
            raise RuntimeError("upstream-init-failure")
        return {"resource_path": resource_path}

    def add_prompt(
        self,
        inference_state,
        frame_idx,
        text_str=None,
        clear_old_points=True,
        points=None,
        point_labels=None,
        boxes_xywh=None,
        box_labels=None,
        clear_old_boxes=True,
        output_prob_thresh=0.5,
        obj_id=None,
        rel_coordinates=True,
    ):
        call = {
            "inference_state": inference_state,
            "frame_idx": frame_idx,
            "text_str": text_str,
            "output_prob_thresh": output_prob_thresh,
        }
        self.prompt_calls.append(call)
        return frame_idx, {"masks": [True]}

    def propagate_in_video(
        self,
        inference_state,
        start_frame_idx=None,
        max_frame_num_to_track=None,
        reverse=False,
        output_prob_thresh=0.5,
        compute_stability_score=False,
        is_instance_processing=False,
        is_last_batch=False,
    ):
        call = {
            "inference_state": inference_state,
            "start_frame_idx": start_frame_idx,
            "max_frame_num_to_track": max_frame_num_to_track,
            "reverse": reverse,
            "output_prob_thresh": output_prob_thresh,
        }
        self.propagate_calls.append(call)
        yield 1, {"masks": [True]}


Sam3MultiplexTrackingWithInteractivity.__module__ = (
    "sam3.model.sam3_multiplex_tracking"
)


class Sam3MultiplexVideoPredictor:
    def __init__(self, model=None) -> None:
        self.model = model or Sam3MultiplexTrackingWithInteractivity()


Sam3MultiplexVideoPredictor.__module__ = (
    "sam3.model.sam3_multiplex_video_predictor"
)


@pytest.fixture
def predictor():
    return Sam3MultiplexVideoPredictor()


def _start(adapter, **overrides):
    request = {
        "type": "start_session",
        "resource_path": "/verified/raw/frames",
        "session_id": "s1",
        "offload_video_to_cpu": False,
        "offload_state_to_cpu": False,
        "async_loading_frames": False,
    }
    request.update(overrides)
    return adapter.handle_request(request)


def test_exact_signatures_and_false_state_offload_mapping(predictor):
    adapter = Sam31CompatAdapterV1(predictor)
    response = _start(adapter)

    assert response["parameter_mapping"]["offload_state_to_cpu"] == (
        "validated_false_no_model_argument"
    )
    assert predictor.model.init_calls == [
        {
            "resource_path": "/verified/raw/frames",
            "offload_video_to_cpu": False,
            "async_loading_frames": False,
        }
    ]
    assert adapter.signature_evidence["signatures_match"] is True


def test_true_state_offload_fails_closed_before_upstream_call(predictor):
    adapter = Sam31CompatAdapterV1(predictor)
    with pytest.raises(Sam31CompatContractError, match="no pinned multiplex"):
        _start(adapter, offload_state_to_cpu=True)
    assert predictor.model.init_calls == []


@pytest.mark.parametrize(
    "change,pattern",
    [
        ({"unexpected": 1}, "unknown keys"),
        ({"offload_state_to_cpu": "false"}, "must be bool"),
        ({"resource_path": ""}, "non-empty str"),
    ],
)
def test_invalid_start_request_rejected(predictor, change, pattern):
    adapter = Sam31CompatAdapterV1(predictor)
    with pytest.raises(Sam31CompatContractError, match=pattern):
        _start(adapter, **change)


def test_upstream_failure_is_not_hidden_or_retried(predictor):
    predictor.model.fail_init = True
    adapter = Sam31CompatAdapterV1(predictor)
    with pytest.raises(RuntimeError, match="upstream-init-failure"):
        _start(adapter)
    assert len(predictor.model.init_calls) == 1


def test_prompt_and_stream_mapping_are_finite_and_explicit(predictor):
    adapter = Sam31CompatAdapterV1(predictor)
    _start(adapter)
    prompt = adapter.handle_request(
        {
            "type": "add_prompt",
            "session_id": "s1",
            "frame_index": 0,
            "text": "a hand",
            "output_prob_thresh": 0.6,
        }
    )
    streamed = list(
        adapter.handle_stream_request(
            {
                "type": "propagate_in_video",
                "session_id": "s1",
                "propagation_direction": "backward",
                "start_frame_index": 3,
                "max_frame_num_to_track": 4,
                "output_prob_thresh": 0.7,
            }
        )
    )
    assert prompt["frame_index"] == 0
    assert streamed[0]["frame_index"] == 1
    assert predictor.model.prompt_calls[0]["text_str"] == "a hand"
    assert predictor.model.propagate_calls[0] == {
        "inference_state": {"resource_path": "/verified/raw/frames"},
        "start_frame_idx": 3,
        "max_frame_num_to_track": 4,
        "reverse": True,
        "output_prob_thresh": 0.7,
    }


def test_duplicate_unknown_session_and_unsupported_routes_rejected(predictor):
    adapter = Sam31CompatAdapterV1(predictor)
    _start(adapter)
    with pytest.raises(Sam31CompatContractError, match="already exists"):
        _start(adapter)
    with pytest.raises(Sam31CompatContractError, match="unknown session_id"):
        adapter.handle_request(
            {
                "type": "add_prompt",
                "session_id": "missing",
                "frame_index": 0,
                "text": "a hand",
            }
        )
    with pytest.raises(Sam31CompatContractError, match="unsupported request"):
        adapter.handle_request({"type": "mystery"})
    with pytest.raises(Sam31CompatContractError, match="forward or backward"):
        list(
            adapter.handle_stream_request(
                {
                    "type": "propagate_in_video",
                    "session_id": "s1",
                    "propagation_direction": "sideways",
                }
            )
        )


def test_close_is_explicit_and_clears_state(predictor):
    adapter = Sam31CompatAdapterV1(predictor)
    _start(adapter)
    state = adapter._sessions["s1"]
    assert adapter.handle_request(
        {"type": "close_session", "session_id": "s1"}
    ) == {"is_success": True}
    assert state == {}
    with pytest.raises(Sam31CompatContractError, match="unknown session_id"):
        adapter.handle_request({"type": "close_session", "session_id": "s1"})


def test_identity_pin_drift_rejected(predictor):
    identity = replace(Sam31PinnedIdentity(), use_rope_real=True)
    with pytest.raises(Sam31CompatContractError, match="pinned identity drift"):
        Sam31CompatAdapterV1(predictor, identity=identity)


def test_class_identity_drift_rejected():
    class WrongPredictor:
        model = Sam3MultiplexTrackingWithInteractivity()

    with pytest.raises(Sam31CompatContractError, match="predictor identity drift"):
        validate_pinned_signatures(WrongPredictor())


def test_signature_drift_rejected(predictor):
    def drifted_init_state(resource_path, unexpected=False):
        return {}

    predictor.model.init_state = drifted_init_state
    with pytest.raises(Sam31CompatContractError, match="init_state signature drift"):
        Sam31CompatAdapterV1(predictor)


def test_run_root_and_owner_marker_are_exclusive(tmp_path):
    (tmp_path / "_run").mkdir()
    token = "a" * 64
    adapter_root, marker = prepare_owned_run(tmp_path, "candidate_v1", token)
    assert adapter_root.is_dir()
    assert marker.is_file()
    with pytest.raises(FileExistsError):
        prepare_owned_run(tmp_path, "candidate_v1", token)


@pytest.mark.parametrize(
    "run_id,token",
    [("../escape", "a" * 64), ("valid_id", "not-a-valid-token")],
)
def test_run_owner_inputs_fail_closed(tmp_path, run_id, token):
    (tmp_path / "_run").mkdir()
    with pytest.raises(ValueError):
        prepare_owned_run(tmp_path, run_id, token)


def test_checkpoint_hash_builder_and_strict_reload_consume_one_held_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload_a = b"checkpoint-A"
    payload_b = b"checkpoint-B"
    checkpoint = tmp_path / "checkpoint.pt"
    replacement = tmp_path / "replacement.pt"
    checkpoint.write_bytes(payload_a)
    replacement.write_bytes(payload_b)
    descriptor = os.open(checkpoint, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    os.replace(replacement, checkpoint)

    predictor = Sam3MultiplexVideoPredictor()

    class FakeTensor:
        @staticmethod
        def numel() -> int:
            return 1

    predictor.model.load_state_dict = lambda _state, strict: types.SimpleNamespace(
        missing_keys=[], unexpected_keys=[]
    )
    predictor.model.parameters = lambda: [FakeTensor()]
    consumed: list[bytes] = []
    official = tmp_path / "official"
    official.mkdir()
    builder = types.ModuleType("sam3.model_builder")
    builder.__file__ = str(official / "sam3/model_builder.py")
    Path(builder.__file__).parent.mkdir(parents=True)
    Path(builder.__file__).write_text("# fixture\n", encoding="utf-8")

    def build(**kwargs):
        consumed.append(Path(kwargs["checkpoint_path"]).read_bytes())
        return predictor

    builder.build_sam3_multiplex_video_predictor = build
    fake_torch = types.SimpleNamespace(
        Tensor=FakeTensor,
        cuda=types.SimpleNamespace(synchronize=lambda: None),
        load=lambda path, **_kwargs: (
            consumed.append(Path(path).read_bytes()) or {"model": {"w": FakeTensor()}}
        ),
    )
    monkeypatch.setattr(compat, "CHECKPOINT_BYTES", len(payload_a))
    monkeypatch.setattr(
        compat, "CHECKPOINT_SHA256", hashlib.sha256(payload_a).hexdigest()
    )
    monkeypatch.setattr(
        compat.subprocess,
        "check_output",
        lambda *_args, **_kwargs: compat.OFFICIAL_CODE_COMMIT + "\n",
    )
    monkeypatch.setattr(
        compat.importlib,
        "import_module",
        lambda name: fake_torch if name == "torch" else builder,
    )
    try:
        _, evidence = compat.build_pinned_adapter(
            official_code_root=official,
            checkpoint_path=checkpoint,
            checkpoint_fd=descriptor,
        )
    finally:
        os.close(descriptor)
    assert consumed == [payload_a, payload_a]
    assert checkpoint.read_bytes() == payload_b
    assert evidence["checkpoint_same_held_fd_consumed"] is True
    assert evidence["checkpoint_sha256"] == hashlib.sha256(payload_a).hexdigest()


def test_official_module_code_object_comes_from_held_snapshot_payload_not_path(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "snapshot"
    package = snapshot / "sam3"
    package.mkdir(parents=True)
    payloads = {
        "sam3/__init__.py": b"# held package\n",
        "sam3/probe.py": b"VALUE = 'HELD_GOOD'\n",
    }
    (package / "__init__.py").write_bytes(b"# mutable package path\n")
    (package / "probe.py").write_bytes(b"VALUE = 'PATH_MALICIOUS'\n")
    digest = compat.official_source_bundle_sha256(payloads)
    finder = compat.HeldOfficialSourceFinder(payloads, snapshot, digest)
    saved = {
        name: module
        for name, module in list(sys.modules.items())
        if name == "sam3" or name.startswith("sam3.")
    }
    for name in saved:
        sys.modules.pop(name, None)
    sys.meta_path.insert(0, finder)
    try:
        imported = importlib.import_module("sam3.probe")
        assert imported.VALUE == "HELD_GOOD"
        assert imported.__held_source_sha256__ == hashlib.sha256(
            payloads["sam3/probe.py"]
        ).hexdigest()
        assert imported.__held_official_bundle_sha256__ == digest
    finally:
        sys.meta_path.remove(finder)
        for name in list(sys.modules):
            if name == "sam3" or name.startswith("sam3."):
                sys.modules.pop(name, None)
        sys.modules.update(saved)


def test_build_pinned_adapter_imports_the_verified_held_builder_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_payload = b"held-checkpoint"
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(checkpoint_payload)
    checkpoint_fd = os.open(checkpoint, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    official = tmp_path / "official"
    official.mkdir()
    snapshot = tmp_path / "snapshot"
    (snapshot / "sam3/assets").mkdir(parents=True)
    builder_payload = (
        b"import builtins\n"
        b"def build_sam3_multiplex_video_predictor(**kwargs):\n"
        b"    builtins._held_builder_kwargs = kwargs\n"
        b"    return builtins._held_predictor\n"
    )
    bpe_payload = b"held-bpe"
    payloads = {
        "sam3/__init__.py": b"# held package\n",
        "sam3/model_builder.py": builder_payload,
        "sam3/assets/bpe_simple_vocab_16e6.txt.gz": bpe_payload,
    }
    (snapshot / "sam3/__init__.py").write_bytes(b"# path package\n")
    (snapshot / "sam3/model_builder.py").write_bytes(
        b"raise RuntimeError('MUTABLE PATH BUILDER EXECUTED')\n"
    )
    (snapshot / "sam3/assets/bpe_simple_vocab_16e6.txt.gz").write_bytes(bpe_payload)

    predictor = Sam3MultiplexVideoPredictor()

    class FakeTensor:
        @staticmethod
        def numel() -> int:
            return 1

    predictor.model.load_state_dict = lambda _state, strict: types.SimpleNamespace(
        missing_keys=[], unexpected_keys=[]
    )
    predictor.model.parameters = lambda: [FakeTensor()]
    fake_torch = types.SimpleNamespace(
        Tensor=FakeTensor,
        cuda=types.SimpleNamespace(synchronize=lambda: None),
        load=lambda *_args, **_kwargs: {"model": {"w": FakeTensor()}},
    )
    real_import_module = importlib.import_module
    monkeypatch.setattr(compat, "CHECKPOINT_BYTES", len(checkpoint_payload))
    monkeypatch.setattr(
        compat, "CHECKPOINT_SHA256", hashlib.sha256(checkpoint_payload).hexdigest()
    )
    monkeypatch.setattr(
        compat.subprocess,
        "check_output",
        lambda *_args, **_kwargs: compat.OFFICIAL_CODE_COMMIT + "\n",
    )
    monkeypatch.setattr(
        compat.importlib,
        "import_module",
        lambda name: fake_torch if name == "torch" else real_import_module(name),
    )
    saved_modules = {
        name: module
        for name, module in list(sys.modules.items())
        if name == "sam3" or name.startswith("sam3.")
    }
    for name in saved_modules:
        sys.modules.pop(name, None)
    builtins._held_predictor = predictor
    try:
        _, evidence = compat.build_pinned_adapter(
            official_code_root=official,
            checkpoint_path=checkpoint,
            checkpoint_fd=checkpoint_fd,
            official_source_payloads=payloads,
            official_snapshot_root=snapshot,
            expected_official_source_bundle_sha256=(
                compat.official_source_bundle_sha256(payloads)
            ),
        )
        assert evidence["official_source_execution"]["method"] == (
            "HELD_PAYLOAD_META_PATH_LOADER"
        )
        assert builtins._held_builder_kwargs["checkpoint_path"] == (
            f"/proc/self/fd/{checkpoint_fd}"
        )
    finally:
        os.close(checkpoint_fd)
        for finder in list(sys.meta_path):
            if isinstance(finder, compat.HeldOfficialSourceFinder):
                sys.meta_path.remove(finder)
        for name in list(sys.modules):
            if name == "sam3" or name.startswith("sam3."):
                sys.modules.pop(name, None)
        sys.modules.update(saved_modules)
        del builtins._held_predictor
        if hasattr(builtins, "_held_builder_kwargs"):
            del builtins._held_builder_kwargs
