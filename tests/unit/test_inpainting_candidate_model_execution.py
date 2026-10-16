"""CPU-only execution/fence/call-budget tests; no GPU or real model calls."""
from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from chaoyang.inpainting_candidate import model_worker as worker
from chaoyang.inpainting_candidate.contracts import file_ref

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/maintenance/inpainting_candidate_model_task_v1.py"
spec = importlib.util.spec_from_file_location("_candidate_model_task_cpu_tests", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))
    return file_ref(path)


class FakeSam:
    def __init__(self):
        self.calls = []
        self.model = FakePoint(self.calls)

    def track_text(self, *args, **kwargs):
        self.calls.append("text")
        return {"synthetic": True}


class FakePoint:
    def __init__(self, calls):
        self.calls = calls

    def init_state(self):
        self.calls.append("init")
        return {}

    def add_prompt(self):
        self.calls.append("add")
        return "prompt"

    def propagate_in_video(self):
        self.calls.append("propagate")
        yield 0
        yield 1


def test_exact_public_api_budget_not_gate_or_yields():
    budget = worker.ModelCallBudget(3, worker.MODEL_CALLS_UNIT)
    original = FakeSam()
    sam = worker.CountedSam(original, budget)
    sam.model.init_state()
    assert budget.total == 0
    sam.track_text()
    sam.model.add_prompt()
    assert list(sam.model.propagate_in_video()) == [0, 1]
    assert budget.receipt()["by_api"] == {"track_text": 1, "point_add_prompt": 1, "point_propagate_in_video": 1}
    assert budget.total == 3
    with pytest.raises(ValueError, match="BUDGET_EXHAUSTED"):
        sam.track_text()
    assert original.calls == ["init", "text", "add", "propagate"]


def test_failed_api_attempt_still_counted():
    class FailedSam(FakeSam):
        def track_text(self):
            raise RuntimeError("actual provider attempt failed")
    budget = worker.ModelCallBudget(1, worker.MODEL_CALLS_UNIT)
    sam = worker.CountedSam(FailedSam(), budget)
    with pytest.raises(RuntimeError):
        sam.track_text()
    assert budget.total == 1
    with pytest.raises(ValueError, match="BUDGET_EXHAUSTED"):
        sam.track_text()


@pytest.mark.parametrize("maximum,unit", [(0, worker.MODEL_CALLS_UNIT), (True, worker.MODEL_CALLS_UNIT),
                                        (5, "NEURAL_FORWARD_PASSES"), (-1, worker.MODEL_CALLS_UNIT)])
def test_call_budget_contract_explicit(maximum, unit):
    with pytest.raises(ValueError, match="UNIT_OR_MAX"):
        worker.ModelCallBudget(maximum, unit)


def test_sam_weight_is_selected_by_fixed_path_not_list_position(tmp_path):
    sam = tmp_path / worker.SAM_CHECKPOINT
    sam.parent.mkdir(parents=True)
    sam.write_bytes(b"CPU fake SAM checkpoint, not loaded")
    other = tmp_path / "raft.pth"
    other.write_bytes(b"CPU fake unrelated weight")
    selected = worker.sam_weight_reference([file_ref(other), file_ref(sam)], repo_root=tmp_path)
    assert selected == file_ref(sam)
    with pytest.raises(ValueError, match="UNIQUE_FROZEN_SAM"):
        worker.sam_weight_reference([file_ref(other)], repo_root=tmp_path)


def test_direct_run_refused_before_model_module_import(monkeypatch):
    from chaoyang.governance import inpainting_candidate_registration as registration
    calls = []
    def denied(*args, **kwargs):
        calls.append("gate")
        raise RuntimeError("unregistered direct invocation")
    monkeypatch.setattr(registration, "authorize_candidate_operation", denied)
    monkeypatch.setattr(worker, "model_authority", lambda packet: pytest.fail("model authority reached"))
    with pytest.raises(RuntimeError, match="unregistered"):
        worker.run({"packet_path": "/invalid/packet.json"})
    assert calls == ["gate"]


def test_memory_packet_mismatch_refused_before_model_loading(monkeypatch):
    from chaoyang.governance import inpainting_candidate_registration as registration
    monkeypatch.setattr(registration, "authorize_candidate_operation", lambda *a, **kw: {"different": True})
    monkeypatch.setattr(worker, "model_authority", lambda packet: pytest.fail("model authority reached"))
    with pytest.raises(ValueError, match="NOT_FROZEN"):
        worker.run({"packet_path": "/invalid/packet.json"})


@pytest.fixture
def fenced(tmp_path, monkeypatch):
    writer = {"pid": 1234, "proc_start_ticks": 5678, "executor_epoch": 1,
              "fencing_token": "cpu-fixture-fence", "scope": "inpainting_generalization_g1_fixture"}
    packet_path = tmp_path / "TASK_PACKET.json"
    writer_path = tmp_path / "WRITER.json"
    packet = {"task_id": writer["scope"], "run_signature": "a" * 64,
              "packet_path": str(packet_path), "writer_file": str(writer_path), "writer": writer}
    put(writer_path, writer)
    reference = put(packet_path, packet)
    row = {"task_id": packet["task_id"], "status": "WAIT_GPU_RESOURCE", "attempt": 0,
           "pid": writer["pid"], "proc_start_ticks": writer["proc_start_ticks"],
           "run_signature": packet["run_signature"], "task_packet": reference}
    state = {"tasks": [row], "next_task": {"task_id": packet["task_id"]}}
    pointer = {"task_id": packet["task_id"], "execution_allowed": True, "packet_sha256": reference["sha256"]}
    monkeypatch.setattr(runner, "proc_start_ticks", lambda pid: 5678)
    return packet, state, {"task_packets": [pointer]}


@pytest.mark.parametrize("status", ["WAIT_GPU_RESOURCE", "RUNNING"])
def test_own_candidate_fence_accepts_wait_and_run(fenced, status):
    packet, state, index = fenced
    state["tasks"][0]["status"] = status
    row, pointer = runner.own_row(packet, state, index)
    assert row is state["tasks"][0] and pointer is index["task_packets"][0]
    if status == "WAIT_GPU_RESOURCE":
        assert row["attempt"] == 0


@pytest.mark.parametrize("kind", ["status", "pid", "ticks", "signature", "next", "writer", "double"])
def test_own_candidate_fence_rejects_drift(fenced, kind):
    packet, state, index = fenced
    row = state["tasks"][0]
    if kind == "status": row["status"] = "PASSED"
    elif kind == "pid": row["pid"] += 1
    elif kind == "ticks": row["proc_start_ticks"] += 1
    elif kind == "signature": row["run_signature"] = "b" * 64
    elif kind == "next": state["next_task"] = None
    elif kind == "writer": put(Path(packet["writer_file"]), {"different": True})
    elif kind == "double": index["task_packets"].append({"execution_allowed": True, "task_id": "other"})
    with pytest.raises(RuntimeError, match="FENCE_DRIFT"):
        runner.own_row(packet, state, index)


def test_child_stop_confirmed_before_lease_release(monkeypatch):
    events = []
    lease = SimpleNamespace(_held=True, release=lambda: events.append("release"))
    child = object()
    monkeypatch.setattr(runner, "stop_owned_child", lambda actual: events.append("stop") if actual is child else pytest.fail("other child"))
    with pytest.raises(RuntimeError, match="simulated failure"):
        with runner.lease_until_child_stops(lease, [child]):
            events.append("body")
            raise RuntimeError("simulated failure")
    assert events == ["body", "stop", "release"]


def test_unconfirmed_child_keeps_explicit_lease(monkeypatch):
    events = []
    lease = SimpleNamespace(_held=True, release=lambda: events.append("release"))
    def failed_stop(child):
        events.append("stop")
        raise RuntimeError("OWN_CHILD_STILL_RUNNING_STOP_REQUIRED")
    monkeypatch.setattr(runner, "stop_owned_child", failed_stop)
    with pytest.raises(RuntimeError, match="STILL_RUNNING"):
        with runner.lease_until_child_stops(lease, [object()]):
            pass
    assert events == ["stop"]


@pytest.fixture
def batch(tmp_path):
    root = tmp_path / "candidate"
    root.mkdir()
    planned, rows = [], []
    for key in runner.EXPECTED_SESSION_KEYS:
        source = tmp_path / "source" / key
        report_ref = put(source / "SOURCE_REPORT.json", {"session_id": "raw_" + key})
        source_ref = put(source / "RECEIPT.json", {"schema": "inpainting-candidate-component-v1", "stage": "Source",
            "status": "SEALED_COMPLETE", "component_signature": "c" * 64, "native_frames": 2, "outputs": [report_ref]})
        mask = root / "sessions" / key / "Mask"
        mask_array = mask / "MASK_ARRAYS.npz"
        mask_array.parent.mkdir(parents=True)
        mask_array.write_bytes(b"CPU fixture, not actual NumPy arrays")
        mask_ref = put(mask / "RECEIPT.json", {"schema": "inpainting-candidate-component-v1", "stage": "Mask",
            "status": "SEALED_COMPLETE", "component_signature": "d" * 64, "outputs": [file_ref(mask_array)],
            "native_input_frames": 2, "native_output_frames": 2, "manual_points_submitted": 0, "producer_manual_seed_reads": 0})
        video = mask.parent / "raw_mask_review.mp4"
        video.write_bytes(b"CPU fixture, decoder is mocked; no video quality claim")
        planned.append({"session_key": key, "source_receipt": source_ref, "mask_signature": "d" * 64})
        rows.append({"session_id": "raw_" + key, "session_key": key, "execution": "COMPLETE_MASK_ONLY",
            "source": source_ref, "mask": mask_ref, "review": {"video": file_ref(video), "decoded_frames": 2},
            "native_frames": 2, "manual_seed_reads": 0, "control_ground_truth": False,
            "training_eligible": False, "visual_quality": "NOT_REVIEWED", "model_calls_unit": runner.MODEL_CALLS_UNIT,
            "model_api_calls_attempted": 1})
    packet = {"task_id": "inpainting_generalization_g1_fixture", "run_signature": "a" * 64,
              "output_dir": str(root), "sessions": planned,
              "budgets": {"model_calls_unit": runner.MODEL_CALLS_UNIT, "model_calls_max": 10}}
    result = {"schema": "inpainting-g1-mask-batch-result-v1", "task_id": packet["task_id"], "run_signature": packet["run_signature"],
        "sessions": rows, "planned_sessions": 6, "completed_mask_sessions": 6, "stage_scope": "G1_AUTOMATIC_MASK_ONLY",
        "candidate_adopted": False, "visual_quality": "NOT_REVIEWED", "model_call_accounting": {
            "unit": runner.MODEL_CALLS_UNIT, "maximum": 10, "attempted_api_calls": 6,
            "by_api": {"track_text": 6, "point_add_prompt": 0, "point_propagate_in_video": 0}}}
    put(root / "MODEL_RESULT.json", result)
    return packet, result


def save_batch(packet, result):
    return put(Path(packet["output_dir"]) / "MODEL_RESULT.json", result)


def test_six_identity_mask_sha_and_actual_decode_verified(batch):
    packet, _ = batch
    decoded = []
    assert runner.validate_model_result(packet, decoder=lambda path: decoded.append(path) or 2)["path"].endswith("MODEL_RESULT.json")
    assert len(decoded) == 6


@pytest.mark.parametrize("kind", ["task", "signature", "partial", "order", "identity", "frames", "manual", "accounting"])
def test_batch_contract_drift_cannot_pass(batch, kind):
    packet, result = batch
    if kind == "task": result["task_id"] = "other"
    elif kind == "signature": result["run_signature"] = "b" * 64
    elif kind == "partial": result["sessions"] = result["sessions"][:-1]
    elif kind == "order": result["sessions"] = list(reversed(result["sessions"]))
    elif kind == "identity": result["sessions"][0]["session_id"] = "another_raw_session"
    elif kind == "frames": result["sessions"][0]["native_frames"] = 1
    elif kind == "manual": result["sessions"][0]["manual_seed_reads"] = 1
    elif kind == "accounting": result["model_call_accounting"]["attempted_api_calls"] = 20
    save_batch(packet, result)
    with pytest.raises(ValueError):
        runner.validate_model_result(packet, decoder=lambda path: 2)


def test_review_actual_decode_count_not_self_report(batch):
    packet, _ = batch
    with pytest.raises(ValueError, match="DECODE_COUNT"):
        runner.validate_model_result(packet, decoder=lambda path: 1)


@pytest.mark.parametrize("artifact", ["mask_array", "video"])
def test_corrupt_output_sha_cannot_pass(batch, artifact):
    packet, result = batch
    row = result["sessions"][0]
    if artifact == "video": path = Path(row["review"]["video"]["path"])
    else: path = Path(row["mask"]["path"]).parent / "MASK_ARRAYS.npz"
    path.write_bytes(b"changed after sealing")
    with pytest.raises(ValueError, match="SHA_DRIFT"):
        runner.validate_model_result(packet, decoder=lambda path: 2)


def test_mask_component_signature_drift_rejected(batch):
    packet, result = batch
    row = result["sessions"][0]
    mask_path = Path(row["mask"]["path"])
    mask = json.loads(mask_path.read_text())
    mask["component_signature"] = "f" * 64
    row["mask"] = put(mask_path, mask)
    save_batch(packet, result)
    with pytest.raises(ValueError, match="COMPONENT_SIGNATURE_MISMATCH"):
        runner.validate_model_result(packet, decoder=lambda path: 2)


@pytest.fixture
def source_worker_batch(tmp_path, monkeypatch):
    """Real Source serializer -> real Mask bridge; only provider/device/codec fake.

    Do not stub load_json, np.load, stage sealing or run_automatic_mask_stage:
    those are the boundaries that the first registered GPU attempt crossed.
    An empty RGB proposal is legitimate synthetic model output, not a claim of
    mask quality. Existing point-provider tests cover nonempty refinement.
    """
    from chaoyang.governance import inpainting_candidate_registration as registration
    from chaoyang.inpainting_candidate import model_stages, source_frames

    monkeypatch.setattr(source_frames, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(model_stages, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(worker, "REPO_ROOT", tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-cpu-fixture")
    planned, expected = [], {}
    for number, key in enumerate(runner.EXPECTED_SESSION_KEYS):
        count = 2 + number % 2
        rgb = np.stack([np.full((480, 640, 3), frame + number, dtype=np.uint8)
                        for frame in range(count)])
        raw_path = tmp_path / "raw" / key / "vst.h264"
        raw_path.parent.mkdir(parents=True)
        raw_path.write_bytes(b"CPU source fixture: decoder substituted, not neural output")
        report = {"status": "SOURCE_PREFLIGHT_CANDIDATE", "session_id": "raw_" + key,
                  "camera": {"physical_eye": "right", "sbs_index": 0, "status": "SOURCE_INDEX_DECLARATION"},
                  "actual_inputs": {"video": file_ref(raw_path)}, "native_sidecar_frames": count,
                  "timebase_status": "TIMEBASE_UNVERIFIED_HOST_ARRIVAL_QPC_DECLARED_NS",
                  "frame_mapping": [{"native_frame_index": i, "sbs_index": 0,
                                     "qpc_declared_ns": 1_000_000_000 + i * 40_000_000}
                                    for i in range(count)]}
        monkeypatch.setattr(source_frames, "decode_physical_eye", lambda *a, frames=rgb, **k:
                            iter((pixels, {"decoder_pts": None}) for pixels in frames))
        folder = tmp_path / "source" / key
        source_frames.prepare_rgb_source(report, folder, component_signature="c" * 64)
        mapping = json.loads((folder / "FRAME_MAP.json").read_text())
        assert isinstance(mapping, list)  # Exactly the real producer schema.
        planned.append({"session_key": key, "source_receipt": file_ref(folder / "RECEIPT.json"),
                        "mask_signature": "d" * 64})
        expected[key] = (rgb, mapping)
    weight = tmp_path / worker.SAM_CHECKPOINT
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"CPU checkpoint fixture; never torch.load or model construction")
    packet = {"task_id": "inpainting_generalization_g1_source_worker_fixture", "run_signature": "a" * 64,
              "packet_path": str(tmp_path / "TASK_PACKET.json"), "output_dir": str(tmp_path / "candidate"),
              "sessions": planned, "model_weights": [file_ref(weight)], "runtime_code": [],
              "budgets": {"model_calls_max": 100, "model_calls_unit": worker.MODEL_CALLS_UNIT}}
    put(Path(packet["packet_path"]), packet)
    owner = {"execution_allowed": True, "algorithm_execution_allowed": True,
             "task_id": packet["task_id"], "run_signature": packet["run_signature"],
             "fencing_token": "cpu-test-authority-not-real-model-permission",
             "gpu_uuid": "GPU-cpu-fixture", "worker_logical_index": 0}
    monkeypatch.setattr(registration, "authorize_candidate_operation", lambda *a, **k: packet)
    monkeypatch.setattr(worker, "model_authority", lambda p: dict(owner))
    select_weight = worker.sam_weight_reference
    monkeypatch.setattr(worker, "sam_weight_reference", lambda weights: select_weight(weights, repo_root=tmp_path))

    class EmptyProposalSam(FakeSam):
        torch = None  # Never inherit an unrelated test's real CUDA module.
        def track_text(self, frames, prompt, anchor):
            self.calls.append((prompt, anchor, len(frames)))
            return {}
    sam = EmptyProposalSam()
    monkeypatch.setattr(model_stages, "load_frozen_models", lambda *a, **k: {"sam": sam})
    reviews = []
    def fake_video(path, frames, mapping):
        key = path.parent.name
        raw, original_mapping = expected[key]
        assert isinstance(mapping, list) and mapping == original_mapping
        assert frames.shape == (len(raw), 480, 1280, 3)
        assert np.array_equal(frames[:, :, :640], raw)
        path.write_bytes(b"CPU codec fixture; do not report visual quality")
        reviews.append(key)
        return {"video": file_ref(path), "decoded_frames": len(mapping)}
    monkeypatch.setattr(source_frames, "write_review_video", fake_video)
    return SimpleNamespace(packet=packet, expected=expected, sam=sam, reviews=reviews)


def test_worker_real_source_list_schema_to_sealed_mask_batch(source_worker_batch):
    fixture = source_worker_batch
    result = worker.run(fixture.packet)
    assert result["completed_mask_sessions"] == result["planned_sessions"] == 6
    assert fixture.reviews == runner.EXPECTED_SESSION_KEYS
    assert len(fixture.sam.calls) == result["model_call_accounting"]["attempted_api_calls"] == 36
    for planned, row in zip(fixture.packet["sessions"], result["sessions"]):
        raw, mapping = fixture.expected[row["session_key"]]
        assert row["source"] == planned["source_receipt"]
        assert row["session_id"] == "raw_" + row["session_key"]
        assert row["native_frames"] == row["review"]["decoded_frames"] == len(raw)
        mask_folder = Path(row["mask"]["path"]).parent
        assert json.loads((mask_folder / "FRAME_MAP.json").read_text()) == mapping
        with np.load(mask_folder / "RAW_RGB.npz", allow_pickle=False) as arrays:
            assert np.array_equal(arrays["raw_rgb"], raw)
    decoder = lambda path: len(fixture.expected[path.parent.name][0])
    assert runner.validate_model_result(fixture.packet, decoder=decoder)["path"].endswith("MODEL_RESULT.json")


@pytest.mark.parametrize("bad", ["object", "row_nonmapping", "duplicate", "short"])
def test_worker_malformed_source_map_never_reaches_sam(source_worker_batch, bad):
    fixture = source_worker_batch
    row = fixture.packet["sessions"][0]
    folder = Path(row["source_receipt"]["path"]).parent
    mapping = json.loads((folder / "FRAME_MAP.json").read_text())
    if bad == "object": mapping = {"frames": mapping}
    elif bad == "row_nonmapping": mapping[0] = 42
    elif bad == "duplicate": mapping[1]["native_frame_index"] = 0
    elif bad == "short": mapping.pop()
    receipt = json.loads((folder / "RECEIPT.json").read_text())
    malformed = folder.parent / "malformed_source"
    malformed.mkdir()
    with np.load(folder / "RAW_RGB.npz", allow_pickle=False) as arrays:
        np.savez_compressed(malformed / "RAW_RGB.npz", rgb=arrays["rgb"])
    put(malformed / "FRAME_MAP.json", mapping)
    put(malformed / "SOURCE_REPORT.json", json.loads((folder / "SOURCE_REPORT.json").read_text()))
    receipt["outputs"] = [file_ref(malformed / name) for name in ("RAW_RGB.npz", "FRAME_MAP.json", "SOURCE_REPORT.json")]
    row["source_receipt"] = put(malformed / "RECEIPT.json", receipt)
    put(Path(fixture.packet["packet_path"]), fixture.packet)
    result = worker.run(fixture.packet)
    assert result["completed_mask_sessions"] == 5
    failed = result["sessions"][0]
    assert failed["execution"] == "FAILED_RUNTIME" and failed["partial_consumable"] is False
    assert failed["model_api_calls_attempted"] == 0
    assert "001" not in fixture.reviews and len(fixture.sam.calls) == 30
    with pytest.raises(ValueError, match="BATCH_BINDING_OR_COMPLETENESS"):
        runner.validate_model_result(fixture.packet, decoder=lambda path: 2)


@pytest.fixture
def resumed_batch(batch, tmp_path):
    """Synthetic sealed C02 evidence; new five Mask outputs have a fresh root.

    The retained 001 is not supplied as a new producer's input. Its fixed 238
    native-frame count follows the actual interrupted C02 contract. The fake
    media bytes are never decoded or claimed to have visual quality here.
    """
    parent, old_result = batch
    parent["task_id"] = "inpainting_generalization_g1_mask_c02_20261013"
    parent["run_signature"] = "e" * 64
    source_path = Path(parent["sessions"][0]["source_receipt"]["path"])
    source = json.loads(source_path.read_text())
    source["native_frames"] = 238
    parent["sessions"][0]["source_receipt"] = put(source_path, source)
    retained = copy.deepcopy(old_result["sessions"][0])
    retained["source"] = parent["sessions"][0]["source_receipt"]
    retained["native_frames"] = retained["review"]["decoded_frames"] = 238
    mask_path = Path(retained["mask"]["path"])
    mask = json.loads(mask_path.read_text())
    mask["native_input_frames"] = mask["native_output_frames"] = 238
    retained["mask"] = put(mask_path, mask)
    retained_ref = put(mask_path.parent.parent / "SESSION_RESULT.json", retained)
    parent_ref = put(tmp_path / "parent_packet.json", parent)
    failed = {"schema": "inpainting-g1-mask-executor-result-v1", "task_id": parent["task_id"],
              "run_signature": parent["run_signature"], "status": "FAILED_RUNTIME_FINAL",
              "planned_sessions": 6, "completed_mask_sessions": 1,
              "retained_sealed_001": retained["mask"], "prior_model_api_calls": 30,
              "partial_consumable": False, "model_result_consumable": False}
    failed_ref = put(tmp_path / "failed_parent_result.json", failed)
    root = tmp_path / "successor"
    root.mkdir()
    successor = copy.deepcopy(parent)
    successor.update(task_id="inpainting_generalization_g1_remaining_fixture", run_signature="f" * 64,
                     output_dir=str(root), sessions=copy.deepcopy(parent["sessions"][1:]),
                     resume_context={"failed_parent_result": failed_ref, "parent_packet": parent_ref,
                                     "retained_session": retained_ref})
    rows = []
    for original in old_result["sessions"][1:]:
        row = copy.deepcopy(original)
        folder = root / "sessions" / row["session_key"] / "Mask"
        folder.mkdir(parents=True)
        # New synthetic outputs, not copied old masks or files consumed by a model.
        (folder / "MASK_ARRAYS.npz").write_bytes(b"fresh CPU successor output fixture")
        new_mask = json.loads(Path(original["mask"]["path"]).read_text())
        new_mask["outputs"] = [file_ref(folder / "MASK_ARRAYS.npz")]
        row["mask"] = put(folder / "RECEIPT.json", new_mask)
        video = folder.parent / "raw_mask_review.mp4"
        video.write_bytes(b"fresh CPU successor codec fixture; no visual claim")
        row["review"]["video"] = file_ref(video)
        rows.append(row)
    result = copy.deepcopy(old_result)
    result.update(task_id=successor["task_id"], run_signature=successor["run_signature"],
                  sessions=rows, planned_sessions=5, completed_mask_sessions=5)
    result["model_call_accounting"]["attempted_api_calls"] = 5
    result["model_call_accounting"]["by_api"]["track_text"] = 5
    save_batch(successor, result)
    return SimpleNamespace(packet=successor, result=result, parent=parent, failed=failed, retained=retained)


def replace_resume_document(fixture, name, value):
    context = fixture.packet["resume_context"]
    context[name] = put(Path(context[name]["path"]), value)


def test_default_plan_is_fixed_original_six(batch):
    packet, _ = batch
    assert runner.planned_session_keys(packet) == ["001", "003", "008", "100", "114", "050"]


def post_controller_batch_fixture(batch):
    packet, result = batch
    root = Path(packet["output_dir"])
    recipe = {"stage": "G1_AUTOMATIC_MASK_ONLY", "image_size": [640, 480],
              "mask_rule_variant": "RGB_GLOVE_WEARER_SUPPORT_B", "manual_seed_inputs": 0}
    indexed = []
    for planned, row, count in zip(packet["sessions"], result["sessions"], (238, 183, 200, 139, 145, 128)):
        source_path = Path(planned["source_receipt"]["path"])
        source = json.loads(source_path.read_text())
        source["native_frames"] = count
        planned["source_receipt"] = put(source_path, source)
        row.update(source=planned["source_receipt"], native_frames=count,
                   experiment_scope=runner.POST_CONTROLLER_SCOPE)
        row["review"]["decoded_frames"] = count
        mask_path = Path(row["mask"]["path"])
        mask = json.loads(mask_path.read_text())
        calls_ref = put(mask_path.parent / "SAM_TEXT.json", {"calls": [{"role": "controller", "prompt": "VR controller"}]})
        mask.update(native_input_frames=count, native_output_frames=count,
                    actual_inputs={"controller_text_prompt": "VR controller"})
        mask["outputs"].append(calls_ref)
        row["mask"] = put(mask_path, mask)
        indexed.append({"session_key": planned["session_key"], "source_receipt": planned["source_receipt"], "native_frames": count})
    parent_ref = put(root / "B_PACKET.json", {"task_id": "inpainting_generalization_g1_mask_wearer_b_20261013",
        "model_stages": ["Mask"], "sessions": copy.deepcopy(packet["sessions"]), "config": recipe})
    index_ref = put(root / "SOURCE_INDEX.json", {"schema": "inpainting-six-source-stage-index-v1",
                                               "native_frames": 1033, "sessions": indexed})
    packet.update(task_id="inpainting_post_g1_controller_rgb_model_a01_20261013",
        experiment_scope=runner.POST_CONTROLLER_SCOPE, model_stages=["Mask"],
        config={**recipe, "controller_text_prompt": "VR controller"},
        inputs={"source_index": index_ref, "parent_recipe_packet": parent_ref})
    packet["budgets"].update(candidates_max=1, gpu_count=1, wall_seconds_max=7200)
    result.update(task_id=packet["task_id"], experiment_scope=runner.POST_CONTROLLER_SCOPE)
    save_batch(packet, result)
    return packet, result


def test_post_controller_result_tracks_real_prompt_source_and_scope(batch, monkeypatch):
    packet, result = post_controller_batch_fixture(batch)
    monkeypatch.setattr(runner, "REPO_ROOT", Path(packet["output_dir"]).parent)
    counts = dict(zip(runner.EXPECTED_SESSION_KEYS, (238, 183, 200, 139, 145, 128)))
    assert runner.planned_session_keys(packet) == runner.EXPECTED_SESSION_KEYS
    runner.validate_model_result(packet, decoder=lambda path: counts[path.parent.name])


@pytest.mark.parametrize("kind", ["result_scope", "session_scope", "receipt_prompt", "actual_text_prompt"])
def test_post_controller_actual_receipt_prompt_and_scope_cannot_be_self_reported(batch, monkeypatch, kind):
    packet, result = post_controller_batch_fixture(batch)
    monkeypatch.setattr(runner, "REPO_ROOT", Path(packet["output_dir"]).parent)
    row = result["sessions"][0]
    if kind == "result_scope":
        result["experiment_scope"] = "G1_AUTOMATIC_MASK_ONLY"
    elif kind == "session_scope":
        row["experiment_scope"] = "G1_AUTOMATIC_MASK_ONLY"
    else:
        mask_path = Path(row["mask"]["path"])
        mask = json.loads(mask_path.read_text())
        if kind == "receipt_prompt":
            mask["actual_inputs"]["controller_text_prompt"] = "old long prompt"
        else:
            calls_path = mask_path.parent / "SAM_TEXT.json"
            changed = put(calls_path, {"calls": [{"role": "controller", "prompt": "old long prompt"}]})
            mask["outputs"] = [changed if Path(ref["path"]).name == "SAM_TEXT.json" else ref for ref in mask["outputs"]]
        row["mask"] = put(mask_path, mask)
    save_batch(packet, result)
    with pytest.raises(ValueError, match="SCOPE_DRIFT|PROMPT_OR_SCOPE"):
        runner.validate_model_result(packet, decoder=lambda path: 238)


@pytest.mark.parametrize("field", ["resume_context", "original_g1_deadline_at"])
def test_post_controller_not_a_resumed_g1_deadline(field):
    spec = {"task_id": "inpainting_post_g1_controller_rgb_model_a01_20261013",
            "budgets": {"wall_seconds_max": 7200}, field: None}
    with pytest.raises(ValueError, match="NOT_G1_SUCCESSOR"):
        runner.effective_deadline(spec, datetime.now(timezone.utc))


def test_post_controller_independent_deadline_does_not_reset_original_g1():
    started = datetime.now(timezone.utc)
    spec = {"task_id": "inpainting_post_g1_controller_rgb_model_a01_20261013", "budgets": {"wall_seconds_max": 7200}}
    assert runner.effective_deadline(spec, started) == started + timedelta(seconds=7200)


@pytest.mark.parametrize("keys", [[], ["003", "008", "100", "114", "050"],
                                  ["001", "003", "008", "100", "114"],
                                  ["050", "114", "100", "008", "003", "001"]])
def test_unproven_skip_or_reorder_rejected(batch, keys):
    packet, _ = batch
    packet["sessions"] = [{"session_key": key} for key in keys]
    with pytest.raises(ValueError, match="FIXED_SIX"):
        runner.planned_session_keys(packet)


def test_proven_retained_001_allows_only_five_new_outputs(resumed_batch):
    fixture = resumed_batch
    assert runner.planned_session_keys(fixture.packet) == ["003", "008", "100", "114", "050"]
    assert all("mask" not in planned for planned in fixture.packet["sessions"])
    assert all(planned["source_receipt"] == original["source_receipt"]
               for planned, original in zip(fixture.packet["sessions"], fixture.parent["sessions"][1:]))
    decoded = []
    reference = runner.validate_model_result(fixture.packet, decoder=lambda path: decoded.append(path) or 2)
    assert reference["path"].endswith("successor/MODEL_RESULT.json")
    assert len(decoded) == 5 and all(path.parent.name != "001" for path in decoded)
    assert all(row["mask"]["path"].startswith(fixture.packet["output_dir"])
               for row in fixture.result["sessions"])


@pytest.mark.parametrize("keys", [["001", "003", "008", "100", "114", "050"],
                                  ["008", "100", "114", "050"],
                                  ["003", "008", "100", "050", "114"]])
def test_resume_cannot_change_fixed_remaining_set(resumed_batch, keys):
    fixture = resumed_batch
    fixture.packet["sessions"] = [{"session_key": key} for key in keys]
    with pytest.raises(ValueError, match="EXACT_PROVEN_REMAINING_FIVE"):
        runner.planned_session_keys(fixture.packet)


@pytest.mark.parametrize("name", ["failed_parent_result", "parent_packet", "retained_session"])
def test_resume_evidence_sha_must_be_current(resumed_batch, name):
    fixture = resumed_batch
    Path(fixture.packet["resume_context"][name]["path"]).write_bytes(b"modified after signing")
    with pytest.raises(ValueError, match="SHA_DRIFT"):
        runner.planned_session_keys(fixture.packet)


@pytest.mark.parametrize("field,value", [("status", "RUNNING"), ("status", "PASSED"),
                                        ("task_id", "inpainting_generalization_g1_mask_c01_20261013"),
                                        ("run_signature", "0" * 64),
                                        ("completed_mask_sessions", 0), ("completed_mask_sessions", 2),
                                        ("planned_sessions", 5)])
def test_resume_failed_parent_identity_and_unique_completion(resumed_batch, field, value):
    fixture = resumed_batch
    fixture.failed[field] = value
    replace_resume_document(fixture, "failed_parent_result", fixture.failed)
    with pytest.raises(ValueError):
        runner.planned_session_keys(fixture.packet)


@pytest.mark.parametrize("field,value", [("session_key", "003"), ("execution", "FAILED_RUNTIME"),
                                        ("execution", "PARTIAL"), ("native_frames", 237),
                                        ("source", {"path": "wrong", "sha256": "0" * 64, "bytes": 0})])
def test_resume_retained_row_is_complete_native_001(resumed_batch, field, value):
    fixture = resumed_batch
    fixture.retained[field] = value
    replace_resume_document(fixture, "retained_session", fixture.retained)
    with pytest.raises(ValueError):
        runner.planned_session_keys(fixture.packet)


@pytest.mark.parametrize("kind", ["source_receipt", "source_output", "mask_output", "video"])
def test_resume_retained_artifacts_cannot_drift(resumed_batch, kind):
    fixture = resumed_batch
    if kind == "source_receipt":
        path = Path(fixture.retained["source"]["path"])
    elif kind == "source_output":
        source = json.loads(Path(fixture.retained["source"]["path"]).read_text())
        path = Path(source["outputs"][0]["path"])
    elif kind == "mask_output":
        mask = json.loads(Path(fixture.retained["mask"]["path"]).read_text())
        path = Path(mask["outputs"][0]["path"])
    else:
        path = Path(fixture.retained["review"]["video"]["path"])
    path.write_bytes(b"post-sealing artifact drift")
    with pytest.raises(ValueError, match="SHA_DRIFT"):
        runner.planned_session_keys(fixture.packet)


@pytest.mark.parametrize("kind", ["mask_partial", "mask_signature", "mask_frames", "review_frames",
                                 "source_partial", "unbound_mask"])
def test_resume_partial_or_unbound_retained_stage_rejected(resumed_batch, kind):
    fixture = resumed_batch
    if kind == "review_frames":
        fixture.retained["review"]["decoded_frames"] = 237
    elif kind == "unbound_mask":
        fixture.failed["retained_sealed_001"] = {"path": "another_receipt", "bytes": 1, "sha256": "0" * 64}
        replace_resume_document(fixture, "failed_parent_result", fixture.failed)
    elif kind == "source_partial":
        path = Path(fixture.retained["source"]["path"])
        source = json.loads(path.read_text())
        source["status"] = "PARTIAL"
        source_ref = put(path, source)
        fixture.parent["sessions"][0]["source_receipt"] = source_ref
        fixture.retained["source"] = source_ref
        replace_resume_document(fixture, "parent_packet", fixture.parent)
    else:
        path = Path(fixture.retained["mask"]["path"])
        mask = json.loads(path.read_text())
        if kind == "mask_partial": mask["status"] = "PARTIAL"
        elif kind == "mask_signature": mask["component_signature"] = "0" * 64
        elif kind == "mask_frames": mask["native_output_frames"] = 237
        fixture.retained["mask"] = put(path, mask)
        fixture.failed["retained_sealed_001"] = fixture.retained["mask"]
        replace_resume_document(fixture, "failed_parent_result", fixture.failed)
    replace_resume_document(fixture, "retained_session", fixture.retained)
    with pytest.raises(ValueError):
        runner.planned_session_keys(fixture.packet)


def test_resume_remaining_source_refs_are_not_replaceable(resumed_batch):
    fixture = resumed_batch
    fixture.packet["sessions"][0]["source_receipt"] = fixture.parent["sessions"][2]["source_receipt"]
    with pytest.raises(ValueError, match="REMAINING_NATIVE_SOURCE_DRIFT"):
        runner.planned_session_keys(fixture.packet)


def test_resume_partial_remaining_result_cannot_be_success(resumed_batch):
    fixture = resumed_batch
    fixture.result["sessions"][0]["execution"] = "FAILED_RUNTIME"
    fixture.result["completed_mask_sessions"] = 4
    save_batch(fixture.packet, fixture.result)
    with pytest.raises(ValueError, match="BATCH_BINDING_OR_COMPLETENESS"):
        runner.validate_model_result(fixture.packet, decoder=lambda path: 2)


def test_deadline_defaults_to_finite_wall_budget():
    started = datetime(2026, 10, 12, 18, 0, tzinfo=timezone.utc)
    spec = {"budgets": {"wall_seconds_max": 1800}}
    assert runner.effective_deadline(spec, started) == started + timedelta(seconds=1800)


@pytest.mark.parametrize("remaining", [60, 1800, 3600])
def test_deadline_is_minimum_of_original_cutoff_and_new_wall_budget(remaining):
    started = datetime(2026, 10, 12, 18, 0, tzinfo=timezone.utc)
    original = started + timedelta(seconds=remaining)
    spec = {"budgets": {"wall_seconds_max": 1800}, "original_g1_deadline_at": original.isoformat()}
    assert runner.effective_deadline(spec, started) == min(original, started + timedelta(seconds=1800))


@pytest.mark.parametrize("remaining", [0, -1, -1800])
def test_exhausted_original_deadline_refuses_start(remaining):
    started = datetime(2026, 10, 12, 18, 0, tzinfo=timezone.utc)
    spec = {"budgets": {"wall_seconds_max": 1800},
            "original_g1_deadline_at": (started + timedelta(seconds=remaining)).isoformat()}
    with pytest.raises(ValueError, match="ORIGINAL_TIMEBOX_EXHAUSTED"):
        runner.effective_deadline(spec, started)


def test_spec_freeze_to_start_gap_never_extends_original_g1_cutoff():
    started = datetime(2026, 10, 13, 1, 36, tzinfo=timezone.utc)
    cutoff = datetime(2026, 10, 13, 1, 37, 7, 507360, tzinfo=timezone.utc)
    spec = {"budgets": {"wall_seconds_max": 28112}, "original_g1_deadline_at": cutoff.isoformat()}
    assert runner.effective_deadline(spec, started) == cutoff


def clean_packet_fixture(tmp_path):
    return {"task_id": "inpainting_generalization_g2_clean_cpu_fixture",
            "spec_schema": runner.CLEAN_SPEC_SCHEMA, "actual_stage_scope": runner.CLEAN_STAGE_SCOPE,
            "model_stages": ["Clean"], "sessions": [{"session_key": "001", "native_frames": 238}],
            "packet_path": str(tmp_path / "TASK_PACKET.json"), "output_dir": str(tmp_path),
            "runtime_code": [], "clean_config": {}, "run_signature": "a" * 64,
            "budgets": {"gpu_count": 1, "model_calls_unit": runner.CLEAN_MODEL_CALLS_UNIT,
                        "model_calls_max": 2, "wall_seconds_max": 28800}}


def test_g2_fixed_plan_is_only_001_full_native_238(tmp_path):
    assert runner.planned_session_keys(clean_packet_fixture(tmp_path)) == ["001"]


@pytest.mark.parametrize("field,value", [("spec_schema", "inpainting-finite-g1-mask-spec-v1"),
    ("task_id", "inpainting_generalization_g1_wrong_stage"),
    ("actual_stage_scope", "FULL_RAW_TO_ROBOT"),
    ("sessions", [{"session_key": "003", "native_frames": 238}]),
    ("sessions", [{"session_key": "001", "native_frames": 237}]),
    ("sessions", [{"session_key": "001", "native_frames": True}]),
    ("sessions", [{"session_key": "001", "native_frames": 238}, {"session_key": "003", "native_frames": 238}]),
    ("resume_context", {"skip": "unproven"})])
def test_g2_plan_cannot_shorten_repurpose_or_reuse_g1_resume(tmp_path, field, value):
    packet = clean_packet_fixture(tmp_path)
    packet[field] = value
    with pytest.raises(ValueError, match="G2_FIXED_001_FULL_238"):
        runner.planned_session_keys(packet)


def test_clean_worker_dispatch_occurs_after_registered_gate_and_authority_without_sam(tmp_path, monkeypatch):
    from chaoyang.governance import inpainting_candidate_registration as registration
    from chaoyang.inpainting_candidate import model_stages
    packet = clean_packet_fixture(tmp_path)
    events, answer = [], {"completed_clean_sessions": 1, "planned_sessions": 1}

    def gate(operation, args, **kwargs):
        events.append("operation_gate")
        assert operation == registration.VISUAL_OPERATION
        assert args == ["--packet", packet["packet_path"]]
        return packet

    def authority(value):
        events.append("lease_authority")
        assert value is packet
        return {"synthetic_cpu_authority": True}

    def run_clean(value, *, authority):
        events.append("clean_dispatch")
        assert value is packet
        authority()
        return answer

    monkeypatch.setattr(registration, "authorize_candidate_operation", gate)
    monkeypatch.setattr(worker, "model_authority", authority)
    monkeypatch.setattr(model_stages, "load_frozen_models", lambda *a, **k: pytest.fail("Clean dispatch entered Mask/SAM loader"))
    monkeypatch.setitem(sys.modules, "chaoyang.inpainting_candidate.clean_worker", SimpleNamespace(run_clean_batch=run_clean))
    assert worker.run(packet) is answer
    assert events == ["operation_gate", "lease_authority", "clean_dispatch", "lease_authority"]


def test_clean_model_result_uses_stage_validator_with_actual_decoder_not_g1_accounting(tmp_path, monkeypatch):
    packet = clean_packet_fixture(tmp_path)
    value = {"schema": "inpainting-g2-clean-batch-result-v1", "task_id": packet["task_id"],
             "run_signature": packet["run_signature"], "completed_model_candidates": 1,
             "B_execution": "NOT_EXECUTED_NO_REFERENCE_CHANGE"}
    put(tmp_path / "MODEL_RESULT.json", value)
    calls = []
    decoder = lambda path: 238

    def verify(received_packet, received_result, *, decoder):
        calls.append((received_packet, received_result, decoder))

    monkeypatch.setitem(sys.modules, "chaoyang.inpainting_candidate.clean_worker", SimpleNamespace(validate_clean_result=verify))
    reference = runner.validate_model_result(packet, decoder=decoder)
    assert reference == file_ref(tmp_path / "MODEL_RESULT.json")
    assert calls == [(packet, value, decoder)]


def test_clean_validator_failure_cannot_be_sealed(tmp_path, monkeypatch):
    packet = clean_packet_fixture(tmp_path)
    put(tmp_path / "MODEL_RESULT.json", {"schema": "partial"})
    error = ValueError("CLEAN_REAL_DECODE_NOT_238")

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setitem(sys.modules, "chaoyang.inpainting_candidate.clean_worker", SimpleNamespace(validate_clean_result=fail))
    with pytest.raises(ValueError) as caught:
        runner.validate_model_result(packet, decoder=lambda path: 237)
    assert caught.value is error


def test_clean_packet_mismatch_refused_before_clean_import(tmp_path, monkeypatch):
    from chaoyang.governance import inpainting_candidate_registration as registration
    packet = clean_packet_fixture(tmp_path)
    monkeypatch.setattr(registration, "authorize_candidate_operation", lambda *a, **kw: {"different": True})
    monkeypatch.setattr(worker, "model_authority", lambda *a: pytest.fail("authority called before packet match"))
    monkeypatch.setitem(sys.modules, "chaoyang.inpainting_candidate.clean_worker",
                        SimpleNamespace(run_clean_batch=lambda *a, **k: pytest.fail("unregistered clean model executed")))
    with pytest.raises(ValueError, match="MODEL_PACKET_NOT_FROZEN_CURRENT_PACKET"):
        worker.run(packet)


@pytest.fixture
def readonly_external_raw_preflight(tmp_path, monkeypatch):
    """Project-external/NAS role without any writes outside the real task tree."""
    infrastructure_path = SCRIPT.parents[1] / "humanego_ict/production_resume_runner_v1.py"
    module_spec = importlib.util.spec_from_file_location("_cpu_test_existing_project_verifier", infrastructure_path)
    infra = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(infra)
    project = tmp_path / "simulated_project"
    project.mkdir()
    monkeypatch.setattr(infra, "REPO_ROOT", project)
    external = tmp_path / "simulated_readonly_nas/vst.h264"
    external.parent.mkdir()
    external.write_bytes(b"CPU read-only NAS-role source; never a model or runtime path")
    raw_ref = file_ref(external)
    folder = project / "Source"
    folder.mkdir()
    report_ref = put(folder / "SOURCE_REPORT.json", {"actual_inputs": {"video": raw_ref}})
    map_ref = put(folder / "FRAME_MAP.json", [{"native_frame_index": 0}])
    source_ref = put(folder / "RECEIPT.json", {"schema": "inpainting-candidate-component-v1",
        "stage": "Source", "status": "SEALED_COMPLETE", "component_signature": "c" * 64,
        "outputs": [report_ref, map_ref]})
    mask_ref = put(project / "Mask/RECEIPT.json", {"stage": "Mask", "fixture": "not consumed here"})
    row = {"source_receipt": source_ref, "mask_receipt": mask_ref,
           "frame_map_ref": map_ref, "raw_source_ref": raw_ref}
    return SimpleNamespace(infra=infra, row=row, external=external, project=project, folder=folder)


def test_readonly_nas_role_raw_uses_generic_sha_not_project_runtime_gate(readonly_external_raw_preflight):
    f = readonly_external_raw_preflight
    with pytest.raises(ValueError, match="PROJECT_ONLY_RUNTIME_PATH"):
        f.infra._verify(f.row["raw_source_ref"])
    runner.verify_session_inputs(f.infra, f.row, clean=True)


def test_readonly_external_raw_sha_drift_is_rejected(readonly_external_raw_preflight):
    f = readonly_external_raw_preflight
    f.external.write_bytes(b"changed raw source after freeze")
    with pytest.raises(ValueError, match="SHA_DRIFT"):
        runner.verify_session_inputs(f.infra, f.row, clean=True)


def test_external_runtime_code_remains_forbidden(readonly_external_raw_preflight):
    f = readonly_external_raw_preflight
    with pytest.raises(ValueError, match="PROJECT_ONLY_RUNTIME_PATH"):
        f.infra._verify(file_ref(f.external))


def test_valid_external_raw_not_bound_to_frozen_source_is_rejected(readonly_external_raw_preflight):
    f = readonly_external_raw_preflight
    other = f.external.parent / "another_vst.h264"
    other.write_bytes(b"different legal file, not this sealed Source")
    f.row["raw_source_ref"] = file_ref(other)
    with pytest.raises(ValueError, match="NOT_CURRENT_SEALED_SOURCE"):
        runner.verify_session_inputs(f.infra, f.row, clean=True)


@pytest.mark.parametrize("field", ["source_receipt", "mask_receipt", "frame_map_ref"])
def test_external_derived_stage_still_project_only(readonly_external_raw_preflight, field):
    f = readonly_external_raw_preflight
    f.row[field] = file_ref(f.external)
    with pytest.raises(ValueError, match="PROJECT_ONLY_RUNTIME_PATH"):
        runner.verify_session_inputs(f.infra, f.row, clean=True)
