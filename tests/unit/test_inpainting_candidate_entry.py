"""Synthetic CPU entry tests: no models, external writes or real registration."""
from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pytest

from chaoyang.inpainting_candidate import entry
from chaoyang.inpainting_candidate.contracts import canonical_sha, file_ref
from chaoyang.governance import inpainting_candidate_registration as reg


@pytest.fixture
def raw(tmp_path, monkeypatch):
    capture = tmp_path / "original_captures"
    session = capture / "unseen_task" / "001"
    (session / "raw").mkdir(parents=True)
    (session / "manifest.json").write_text('{"session":"unseen_001"}')
    (session / "raw/vst.h264").write_bytes(b"synthetic CPU fixture, not a decodable video")
    (session / "raw/vst.qpc.ts.jsonl").write_text("100\n200\n")
    (session / "raw/vst.ts.jsonl").write_text("1000\n2000\n")
    monkeypatch.setattr(entry, "ALLOWED_RAW_ROOTS", (capture,))
    return session


def inspector(path, *, decode_probe=False):
    assert decode_probe is False
    refs = {role: file_ref(path / relative) for role, relative in entry.RAW_FILES.items()
            if (path / relative).is_file()}
    identity = json.loads((path / "manifest.json").read_text())["session"]
    return {"session_dir": str(path), "session_id": identity, "actual_inputs": refs,
            "status": "SOURCE_PREFLIGHT_CANDIDATE", "aligned_rows_read": False,
            "processed_video_read": False, "hdf5_datasets_read": False, "control_ground_truth": False,
            "camera": {"status": "ABSENT"}}


def request(raw, tmp_path):
    return entry.build_request(project_root=tmp_path, session=raw, inspector=inspector)


def rewrite_signature(value):
    payload = {key: value[key] for key in ("schema", "recipe", "manifest", "sources", "motion_source",
                "producer_policy", "manual_session_edits")}
    value["submission_input_sha256"] = canonical_sha(payload)


def test_single_request_no_task_name_config_or_seeds(raw, tmp_path):
    value = request(raw, tmp_path)
    entry.verify_request(value)
    assert value["motion_source"] == "RAW_CONTROLLER_HEAD_PLUS_RAW_MANUS"
    assert value["manual_session_edits"] == 0
    assert value["status"] == "CPU_ADMISSION_REQUEST_ONLY"
    assert value["robot_video_produced"] is False


def test_actual_inspector_missing_motion_does_not_block_rgb(raw, tmp_path):
    value = entry.build_request(project_root=tmp_path, session=raw)
    entry.verify_request(value)
    assert value["sources"][0]["capabilities"]["native_motion"] == "BLOCKED_NATIVE_MOTION_INPUT_ABSENT"
    assert value["sources"][0]["timebase_status"].startswith("TIMEBASE_UNVERIFIED")


@pytest.mark.parametrize("both", [False, True])
def test_exactly_one_selection_required(raw, tmp_path, both):
    with pytest.raises(entry.EntryError, match="EXACTLY_ONE"):
        entry.build_request(project_root=tmp_path, session=raw if both else None,
                            manifest=tmp_path / "m.json" if both else None)


@pytest.mark.parametrize("part", ["processed", "_run", "review", "attempts", "archive"])
def test_derived_session_denied(raw, part):
    bad = raw.parents[1] / part / "001"
    bad.mkdir(parents=True)
    with pytest.raises(entry.EntryError, match="DERIVED_SESSION"):
        entry.raw_session_path(bad)


def test_path_alias_denied(raw):
    alias = raw.parent / "alias"
    alias.symlink_to(raw, target_is_directory=True)
    with pytest.raises(entry.EntryError, match="ALIAS"):
        entry.raw_session_path(alias)


def test_outside_capture_root_denied(raw, tmp_path):
    with pytest.raises(entry.EntryError, match="OUTSIDE_ALLOWED"):
        entry.raw_session_path(tmp_path)


def test_frozen_manifest_preserves_order_and_identity(raw, tmp_path):
    manifest = tmp_path / "frozen.json"
    manifest.write_text(json.dumps({"schema": entry.MANIFEST_SCHEMA, "sessions": [str(raw)]}))
    value = entry.build_request(project_root=tmp_path, manifest=manifest, inspector=inspector)
    assert value["manifest"] == file_ref(manifest)
    assert value["sources"][0]["session_dir"] == str(raw)
    entry.verify_request(value)


@pytest.mark.parametrize("extra", ["task", "manual_seed", "session_config", "visible_seed"])
def test_manifest_cannot_pass_task_or_manual_parameters(raw, tmp_path, extra):
    manifest = tmp_path / "frozen.json"
    manifest.write_text(json.dumps({"schema": entry.MANIFEST_SCHEMA, "sessions": [str(raw)], extra: "fake"}))
    with pytest.raises(entry.EntryError, match="TASK_CONFIG_PRESENT"):
        entry.build_request(project_root=tmp_path, manifest=manifest, inspector=inspector)


def test_duplicate_manifest_session_denied(raw, tmp_path):
    manifest = tmp_path / "frozen.json"
    manifest.write_text(json.dumps({"schema": entry.MANIFEST_SCHEMA, "sessions": [str(raw), str(raw)]}))
    with pytest.raises(entry.EntryError, match="DUPLICATE"):
        entry.build_request(project_root=tmp_path, manifest=manifest, inspector=inspector)


def test_source_byte_drift_rejected(raw, tmp_path):
    value = request(raw, tmp_path)
    (raw / "raw/vst.h264").write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA_DRIFT"):
        entry.verify_request(value)


def test_unknown_role_rejected_even_if_resigned(raw, tmp_path):
    value = request(raw, tmp_path)
    value["sources"][0]["actual_inputs"]["historical_clean"] = file_ref(raw / "raw/vst.h264")
    rewrite_signature(value)
    with pytest.raises(entry.EntryError, match="NON_RAW_ROLE"):
        entry.verify_request(value)


@pytest.mark.parametrize("field", ["aligned_rows_read", "processed_video_read", "hdf5_datasets_read", "control_ground_truth"])
def test_derived_or_truth_claim_rejected_even_if_resigned(raw, tmp_path, field):
    value = request(raw, tmp_path)
    value["sources"][0][field] = True
    rewrite_signature(value)
    with pytest.raises(entry.EntryError, match="DERIVED_SOURCE"):
        entry.verify_request(value)


def test_preview_no_route_or_registration_required(raw, tmp_path, monkeypatch):
    value = request(raw, tmp_path)
    monkeypatch.setattr(reg, "register", lambda *a, **kw: pytest.fail("preview registered"))
    monkeypatch.setattr(reg, "_snapshot", lambda *a: pytest.fail("preview read live governance"))
    result = reg.submit_request(value, repo_root=tmp_path, preview=True)
    assert result["status"] == "READ_ONLY_CPU_REQUEST_PREVIEW"
    assert result["execution_started_here"] is False


def test_full_route_absent_without_operation_wrapper(tmp_path):
    assert reg.route_contract(tmp_path) == {}


def snapshot_fixture(root, monkeypatch):
    route = {"code_closure": [], "candidate_operation": reg.OPERATION}
    state = {"tasks": [], "next_task": None}
    index = {"task_packets": []}
    monkeypatch.setattr(reg, "_snapshot", lambda unused: ({}, state, index, {}))
    monkeypatch.setattr(reg, "_route", lambda unused, contract: route)
    return state, index


def test_ready_is_not_registered_or_robot_produced(raw, tmp_path, monkeypatch):
    snapshot_fixture(tmp_path, monkeypatch)
    result = reg.plan_request(request(raw, tmp_path), repo_root=tmp_path)
    assert result["status"] == "CPU_ADMISSION_READY_NOT_REGISTERED"
    assert result["robot_video_produced"] is False
    assert not (tmp_path / "_run/current").exists()


def test_active_same_signature_attaches_without_new_writer(raw, tmp_path, monkeypatch):
    state, index = snapshot_fixture(tmp_path, monkeypatch)
    value = request(raw, tmp_path)
    signature = reg.cpu_signature(value, [])
    index["task_packets"] = [{"execution_allowed": True}]
    monkeypatch.setattr(reg, "_active_packet", lambda *a: {"phase": "INPAINTING_CANDIDATE_CPU_ADMISSION",
        "run_signature": signature, "inputs": {"request_sha256": canonical_sha(value)}})
    monkeypatch.setattr(reg, "register", lambda *a, **kw: pytest.fail("duplicate writer registered"))
    result = reg.submit_request(value, repo_root=tmp_path)
    assert result["status"] == "ATTACHED_TO_ACTIVE_CPU_ADMISSION"


def test_active_other_signature_rejected(raw, tmp_path, monkeypatch):
    _, index = snapshot_fixture(tmp_path, monkeypatch)
    index["task_packets"] = [{"execution_allowed": True}]
    monkeypatch.setattr(reg, "_active_packet", lambda *a: {"run_signature": "other"})
    with pytest.raises(reg.RegistrationError, match="DIFFERENT_SIGNATURE"):
        reg.plan_request(request(raw, tmp_path), repo_root=tmp_path)


def test_unrouted_active_state_rejected(raw, tmp_path, monkeypatch):
    state, _ = snapshot_fixture(tmp_path, monkeypatch)
    state["next_task"] = {"task_id": "other"}
    with pytest.raises(reg.RegistrationError, match="ACTIVE_STATE_WITHOUT_PACKET"):
        reg.plan_request(request(raw, tmp_path), repo_root=tmp_path)


def test_unsealed_attempt_cannot_be_reused(raw, tmp_path, monkeypatch):
    snapshot_fixture(tmp_path, monkeypatch)
    value = request(raw, tmp_path)
    signature = reg.cpu_signature(value, [])
    task_id = "inpainting_cpu_admission_" + signature[:24]
    reg._paths(tmp_path, task_id)[0].mkdir(parents=True)
    with pytest.raises(reg.RegistrationError, match="UNSEALED_ATTEMPT"):
        reg.plan_request(value, repo_root=tmp_path)


def test_route_closure_requires_exact_frozen_bytes(tmp_path, monkeypatch):
    expected = {"code_closure": [{"sha256": "a"}], "candidate_operation": reg.OPERATION}
    monkeypatch.setattr(reg, "route_contract", lambda unused: {reg.ROUTE_KEY: expected})
    with pytest.raises(reg.RegistrationError, match="ROUTE_ABSENT_OR_SHA_DRIFT"):
        reg._route(tmp_path, {"special_status_contracts": {reg.ROUTE_KEY: {"code_closure": []}}})


def test_cpu_signature_binds_code_bytes(raw, tmp_path):
    value = request(raw, tmp_path)
    assert reg.cpu_signature(value, [{"sha256": "a"}]) != reg.cpu_signature(value, [{"sha256": "b"}])


def test_gate_empty_packet_rejects_before_any_operation_write(tmp_path, monkeypatch):
    monkeypatch.setattr(reg, "_snapshot", lambda root: ({}, {"tasks": []}, {"task_packets": []}, {}))
    monkeypatch.setattr(reg, "_route", lambda *a: {"code_closure": []})
    with pytest.raises(reg.RegistrationError, match="MULTIPLE_ACTIVE_PACKETS"):
        reg.authorize_cpu_operation([], repo_root=tmp_path)


def test_register_refuses_non_project_publisher(raw, tmp_path):
    with pytest.raises(reg.RegistrationError, match="REAL_PROJECT_REQUIRED"):
        reg.register(request(raw, tmp_path), repo_root=tmp_path)


def test_write_once_cannot_overwrite_prior_output(tmp_path):
    path = tmp_path / "result.json"
    reg.write_once_json(path, {"first": True})
    with pytest.raises(reg.RegistrationError, match="ALREADY_EXISTS"):
        reg.write_once_json(path, {"first": False})
    assert json.loads(path.read_text()) == {"first": True}


def test_default_submit_calls_existing_registration_then_bounded_child(raw, tmp_path, monkeypatch):
    value = request(raw, tmp_path)
    monkeypatch.setattr(reg, "plan_request", lambda *a, **kw: {"status": "CPU_ADMISSION_READY_NOT_REGISTERED"})
    packet = {"task_id": "synthetic_cpu"}
    monkeypatch.setattr(reg, "register", lambda *a, **kw: packet)
    monkeypatch.setattr(reg, "run_registered", lambda actual: {"status": "PASSED", "scope": "CPU_ADMISSION_ONLY", "robot_video_produced": False} if actual is packet else pytest.fail("packet drift"))
    result = reg.submit_request(value, repo_root=tmp_path)
    assert result["scope"] == "CPU_ADMISSION_ONLY"
    assert result["robot_video_produced"] is False


def visual_gate_fixture(tmp_path, monkeypatch):
    from chaoyang.governance import exact78_operation_gate
    runtime_path = tmp_path / "src/chaoyang/ops" / (reg.VISUAL_OPERATION + ".py")
    runtime_path.parent.mkdir(parents=True)
    runtime_path.write_text("# synthetic operation wrapper; no algorithm\n")
    runtime = [file_ref(runtime_path)]
    packet = {"task_id": "inpainting_generalization_g1_synthetic", "operation": reg.VISUAL_OPERATION,
        "algorithm_execution_allowed": True, "runtime_code": runtime,
        "budgets": {"gpu_count": 1, "wall_seconds_max": 28800, "model_calls_max": 1, "candidates_max": 2},
        "commands": ["python -B -m chaoyang.cli run " + reg.VISUAL_OPERATION + " --fixture synthetic"],
        "writer": {"pid": 123, "proc_start_ticks": 456},
        "deadline_at": (datetime.now(timezone.utc) + timedelta(minutes=1)).isoformat(),
        "inputs": {}, "weights": {}, "config": {}, "calibration": {"status": "ABSENT"}, "schema": {}}
    contract = {"special_status_contracts": {reg.VISUAL_ROUTE_KEY: {
        "candidate_operation": reg.VISUAL_OPERATION, "task_packet_required": True,
        "current_runnable_operation": None, "code_closure": runtime}}}
    monkeypatch.setattr(reg, "_snapshot", lambda root: ({}, {}, {"task_packets": [{"execution_allowed": True}]}, contract))
    monkeypatch.setattr(reg, "visual_route_contract", lambda root: {reg.VISUAL_ROUTE_KEY: copy.deepcopy(contract["special_status_contracts"][reg.VISUAL_ROUTE_KEY])})
    monkeypatch.setattr(reg, "_active_packet", lambda *a: packet)
    monkeypatch.setattr(exact78_operation_gate, "_caller_is_writer_descendant", lambda *a: True)
    return packet, contract


def test_visual_gate_valid_finite_candidate(tmp_path, monkeypatch):
    packet, _ = visual_gate_fixture(tmp_path, monkeypatch)
    assert reg.authorize_candidate_operation(reg.VISUAL_OPERATION, ["--fixture", "synthetic"], repo_root=tmp_path) is packet


@pytest.mark.parametrize("key,value", [("gpu_count", 0), ("gpu_count", 3), ("gpu_count", True),
    ("wall_seconds_max", 28801), ("wall_seconds_max", 0), ("model_calls_max", 0),
    ("model_calls_max", True), ("candidates_max", 3), ("candidates_max", 0)])
def test_visual_gate_rejects_changed_bounds(tmp_path, monkeypatch, key, value):
    packet, _ = visual_gate_fixture(tmp_path, monkeypatch)
    packet["budgets"][key] = value
    with pytest.raises(reg.RegistrationError, match="FINITE_G1_BOUNDS"):
        reg.authorize_candidate_operation(reg.VISUAL_OPERATION, ["--fixture", "synthetic"], repo_root=tmp_path)


@pytest.mark.parametrize("key,value", [("algorithm_execution_allowed", False), ("task_id", "historical_0918"),
                                       ("operation", "other_model")])
def test_visual_gate_cannot_reuse_other_or_historical_packet(tmp_path, monkeypatch, key, value):
    packet, _ = visual_gate_fixture(tmp_path, monkeypatch)
    packet[key] = value
    with pytest.raises(reg.RegistrationError, match="FINITE_G1_BOUNDS"):
        reg.authorize_candidate_operation(reg.VISUAL_OPERATION, ["--fixture", "synthetic"], repo_root=tmp_path)


def test_visual_gate_exact_command_required(tmp_path, monkeypatch):
    visual_gate_fixture(tmp_path, monkeypatch)
    with pytest.raises(reg.RegistrationError, match="COMMAND_DRIFT"):
        reg.authorize_candidate_operation(reg.VISUAL_OPERATION, ["--fixture", "other"], repo_root=tmp_path)


def test_visual_gate_deadline(tmp_path, monkeypatch):
    packet, _ = visual_gate_fixture(tmp_path, monkeypatch)
    packet["deadline_at"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    with pytest.raises(reg.RegistrationError, match="FINITE_DEADLINE"):
        reg.authorize_candidate_operation(reg.VISUAL_OPERATION, ["--fixture", "synthetic"], repo_root=tmp_path)


@pytest.mark.parametrize("domain", ["inputs", "weights", "config", "calibration", "schema"])
def test_visual_gate_all_signature_domains_required(tmp_path, monkeypatch, domain):
    packet, _ = visual_gate_fixture(tmp_path, monkeypatch)
    del packet[domain]
    with pytest.raises(reg.RegistrationError, match="SIGNATURE_DOMAIN_ABSENT"):
        reg.authorize_candidate_operation(reg.VISUAL_OPERATION, ["--fixture", "synthetic"], repo_root=tmp_path)


def test_visual_gate_runtime_actual_bytes_not_self_reported(tmp_path, monkeypatch):
    packet, _ = visual_gate_fixture(tmp_path, monkeypatch)
    Path(packet["runtime_code"][0]["path"]).write_text("# changed\n")
    with pytest.raises(ValueError, match="SHA_DRIFT"):
        reg.authorize_candidate_operation(reg.VISUAL_OPERATION, ["--fixture", "synthetic"], repo_root=tmp_path)


def test_visual_gate_actual_weight_bytes(tmp_path, monkeypatch):
    packet, _ = visual_gate_fixture(tmp_path, monkeypatch)
    path = tmp_path / "frozen-weight.dat"
    path.write_bytes(b"frozen")
    packet["weights"] = {"nested": {"sam": file_ref(path)}}
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA_DRIFT"):
        reg.authorize_candidate_operation(reg.VISUAL_OPERATION, ["--fixture", "synthetic"], repo_root=tmp_path)


def test_visual_gate_non_descendant_cannot_import_models(tmp_path, monkeypatch):
    from chaoyang.governance import exact78_operation_gate
    visual_gate_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(exact78_operation_gate, "_caller_is_writer_descendant", lambda *a: False)
    with pytest.raises(reg.RegistrationError, match="NOT_FENCED_WRITER_DESCENDANT"):
        reg.authorize_candidate_operation(reg.VISUAL_OPERATION, ["--fixture", "synthetic"], repo_root=tmp_path)


def test_unknown_operation_rejected(tmp_path):
    with pytest.raises(reg.RegistrationError, match="OPERATION_UNSUPPORTED"):
        reg.authorize_candidate_operation("arbitrary_model", [], repo_root=tmp_path)


def snapshot_files(tmp_path):
    lock = tmp_path / "_run/current/locks/governance.lock"
    lock.parent.mkdir(parents=True)
    lock.write_bytes(b"")
    data = {"governance_revision": 7, "generation_id": "cpu-fixture-generation"}
    refs = {}
    for name, relative in (("task_state", "docs/governance/LONG_HORIZON_TASK_STATE.json"),
                           ("task_packet_index_current", "tasks/current/INDEX.json"),
                           ("algorithm_contract", "docs/governance/ALGORITHM_CONTRACT.json")):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
        refs[name] = file_ref(path)
    receipt = tmp_path / "docs/governance/CURRENT_STATUS_RECEIPT.json"
    receipt.write_text(json.dumps({**data, "files": refs}))
    return lock, refs


def test_snapshot_reads_all_bound_files_under_existing_shared_lock(tmp_path, monkeypatch):
    import fcntl
    lock, _ = snapshot_files(tmp_path)
    events = []
    original_load = reg._load
    original_flock = fcntl.flock
    def tracked_flock(fd, mode):
        events.append("lock" if mode == fcntl.LOCK_SH else "unlock")
        return original_flock(fd, mode)
    def tracked_load(path):
        assert events and events[0] == "lock" and events[-1] != "unlock"
        events.append("read")
        return original_load(path)
    monkeypatch.setattr(reg.fcntl, "flock", tracked_flock)
    monkeypatch.setattr(reg, "_load", tracked_load)
    receipt, state, index, contract = reg._snapshot(tmp_path)
    assert receipt["governance_revision"] == state["governance_revision"] == index["governance_revision"] == contract["governance_revision"] == 7
    assert events == ["lock", "read", "read", "read", "read", "unlock"]
    assert lock.read_bytes() == b""


def test_snapshot_failure_unlocks_and_does_not_create_missing_lock(tmp_path, monkeypatch):
    import fcntl
    lock, refs = snapshot_files(tmp_path)
    Path(refs["task_state"]["path"]).write_text("changed outside valid publisher")
    events = []
    original = fcntl.flock
    def track(fd, mode):
        events.append(mode)
        return original(fd, mode)
    monkeypatch.setattr(reg.fcntl, "flock", track)
    with pytest.raises(ValueError, match="SHA_DRIFT"):
        reg._snapshot(tmp_path)
    assert events == [fcntl.LOCK_SH, fcntl.LOCK_UN]
    missing = tmp_path / "another-project"
    missing.mkdir()
    with pytest.raises(FileNotFoundError):
        reg._snapshot(missing)
    assert not (missing / "_run").exists()


def visual_runtime_files(tmp_path):
    required = set(reg.RUNTIME_PATHS) | {
        "src/chaoyang/ops/" + reg.VISUAL_OPERATION + ".py",
        "scripts/maintenance/inpainting_candidate_model_task_v1.py",
        "scripts/humanego_ict/production_resume_runner_v1.py",
        "src/chaoyang/exact78_baseline/__init__.py", "src/chaoyang/exact78_baseline/mask.py",
        "src/chaoyang/exact78_baseline/clean.py", "src/chaoyang/pick_place_1008/__init__.py",
        "src/chaoyang/pick_place_1008/sam_point_adapter.py", "src/chaoyang/pipeline/__init__.py",
        "src/chaoyang/pipeline/sam31_compat_adapter_v1.py", "src/chaoyang/inpainting_candidate/model_worker.py",
        "src/chaoyang/inpainting_candidate/model_stages.py", "src/chaoyang/inpainting_candidate/resources.py",
        "src/chaoyang/inpainting_candidate/automatic_masks.py", "src/chaoyang/inpainting_candidate/clean_references.py",
        "src/chaoyang/inpainting_candidate/source_frames.py", "vendor/SAM3/sam3/__init__.py",
        "src/chaoyang/inpainting_candidate/clean_worker.py", "src/chaoyang/inpainting_candidate/clean_flow.py",
        "src/chaoyang/pick_place_1008/visual.py", "src/chaoyang/pipeline/exact78_visual_compositor_v1.py",
        "vendor/ProPainter/model/__init__.py", "vendor/ProPainter/model/propainter.py",
        "vendor/ProPainter/model/recurrent_flow_completion.py", "vendor/ProPainter/model/modules/flow_comp_raft.py",
        "vendor/ProPainter/model/modules/flow_loss_utils.py", "vendor/ProPainter/RAFT/__init__.py",
        "vendor/ProPainter/RAFT/raft.py", "vendor/ProPainter/utils/synthetic_inference_dependency.py",
        "vendor/SAM3/sam3/model_builder.py", "vendor/SAM3/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        "configs/systems/mask/configs/sam31_runtime_source_v1.json", "manifests/vendor.json",
        "vendor/SAM3/sam3/model/synthetic_runtime.py",
    }
    for relative in required:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"CPU synthetic pinned closure file, not executable model code")
    return required


def test_visual_route_draft_without_ops_is_not_runnable(tmp_path):
    assert reg.visual_route_contract(tmp_path) == {}


def test_visual_route_freezes_complete_generic_runtime_not_models(tmp_path):
    required = visual_runtime_files(tmp_path)
    route = reg.visual_route_contract(tmp_path)[reg.VISUAL_ROUTE_KEY]
    paths = {str(Path(reference["path"]).relative_to(tmp_path)) for reference in route["code_closure"]}
    assert paths == required
    assert route["model_stages"] == ["Mask", "Clean"]
    assert route["stage_model_calls_units"]["Mask"] == "SAM_ADAPTER_INFERENCE_API_INVOCATION"
    assert route["stage_model_calls_units"]["Clean"] == reg.CLEAN_MODEL_CALLS_UNIT
    assert route["current_runnable_operation"] is None
    assert route["render_calls_authorized"] == route["training_steps_authorized"] == 0
    assert not any(path.endswith(".pt") or "manual_seed" in path for path in paths)
    assert "src/chaoyang/governance/exact78_operation_gate.py" in paths
    assert reg.route_contract(tmp_path)[reg.ROUTE_KEY]["model_calls_authorized"] == 0


def test_visual_route_changes_when_actual_runtime_changes(tmp_path):
    visual_runtime_files(tmp_path)
    first = reg.visual_route_contract(tmp_path)
    (tmp_path / "vendor/SAM3/sam3/model/synthetic_runtime.py").write_bytes(b"new actual code bytes")
    assert first != reg.visual_route_contract(tmp_path)


def test_visual_route_ignores_unrelated_candidate_helper(tmp_path):
    visual_runtime_files(tmp_path)
    first = reg.visual_route_contract(tmp_path)
    (tmp_path / "src/chaoyang/inpainting_candidate/future_g2_helper.py").write_bytes(b"unrelated later component")
    assert first == reg.visual_route_contract(tmp_path)


def test_visual_route_no_source_alias_outside_project(tmp_path):
    target = tmp_path / "unbound-outside-root.py"
    target.write_bytes(b"external source")
    alias = tmp_path / "src/chaoyang/inpainting_candidate/resources.py"
    alias.parent.mkdir(parents=True)
    alias.symlink_to(target)
    visual_runtime_files(tmp_path)
    with pytest.raises(reg.RegistrationError, match="ALIAS_OR_PROJECT_ESCAPE"):
        reg.visual_route_contract(tmp_path)


def test_visual_gate_rejects_self_declared_route_fields(tmp_path, monkeypatch):
    packet, contract = visual_gate_fixture(tmp_path, monkeypatch)
    expected = copy.deepcopy(contract["special_status_contracts"][reg.VISUAL_ROUTE_KEY])
    monkeypatch.setattr(reg, "visual_route_contract", lambda root: {reg.VISUAL_ROUTE_KEY: expected})
    contract["special_status_contracts"][reg.VISUAL_ROUTE_KEY]["unregistered_permission"] = True
    with pytest.raises(reg.RegistrationError, match="VISUAL_CANDIDATE_ROUTE_ABSENT"):
        reg.authorize_candidate_operation(reg.VISUAL_OPERATION, ["--fixture", "synthetic"], repo_root=tmp_path)


def clean_gate_fixture(tmp_path, monkeypatch):
    packet, contract = visual_gate_fixture(tmp_path, monkeypatch)
    packet.update(task_id="inpainting_generalization_g2_clean_synthetic", spec_schema=reg.CLEAN_SPEC_SCHEMA,
                  actual_stage_scope=reg.CLEAN_STAGE_SCOPE, model_stages=["Clean"], clean_config={},
                  sessions=[{"session_key": "001", "native_frames": 238}])
    packet["budgets"].update(model_calls_unit=reg.CLEAN_MODEL_CALLS_UNIT, model_calls_max=2)
    return packet, contract


def test_clean_gate_reuses_exact_operation_and_finite_fence(tmp_path, monkeypatch):
    packet, _ = clean_gate_fixture(tmp_path, monkeypatch)
    assert reg.authorize_candidate_operation(reg.VISUAL_OPERATION, ["--fixture", "synthetic"], repo_root=tmp_path) is packet


@pytest.mark.parametrize("field,value", [("gpu_count", 0), ("gpu_count", 2), ("gpu_count", True),
    ("model_calls_max", 0), ("model_calls_max", 3), ("model_calls_max", True),
    ("model_calls_unit", "SAM_ADAPTER_INFERENCE_API_INVOCATION"),
    ("wall_seconds_max", 28801), ("wall_seconds_max", 0)])
def test_clean_gate_cannot_borrow_g1_or_expand_budget(tmp_path, monkeypatch, field, value):
    packet, _ = clean_gate_fixture(tmp_path, monkeypatch)
    packet["budgets"][field] = value
    with pytest.raises(reg.RegistrationError, match="G2_CLEAN_BOUNDS"):
        reg.authorize_candidate_operation(reg.VISUAL_OPERATION, ["--fixture", "synthetic"], repo_root=tmp_path)


@pytest.mark.parametrize("field,value", [("task_id", "inpainting_generalization_g1_wrong_stage"),
    ("spec_schema", "inpainting-finite-g1-mask-spec-v1"), ("model_stages", ["Mask", "Clean"]),
    ("actual_stage_scope", "FULL_RAW_TO_ROBOT"), ("sessions", [{"session_key": "003", "native_frames": 238}]),
    ("sessions", [{"session_key": "001", "native_frames": 237}]),
    ("sessions", [{"session_key": "001", "native_frames": True}]),
    ("resume_context", {"skip": "unproven"})])
def test_clean_gate_fixed_native_scope_not_session_patch(tmp_path, monkeypatch, field, value):
    packet, _ = clean_gate_fixture(tmp_path, monkeypatch)
    packet[field] = value
    with pytest.raises(reg.RegistrationError, match="G2_CLEAN_BOUNDS"):
        reg.authorize_candidate_operation(reg.VISUAL_OPERATION, ["--fixture", "synthetic"], repo_root=tmp_path)


def test_clean_gate_nested_capture_config_sha_is_actual(tmp_path, monkeypatch):
    packet, _ = clean_gate_fixture(tmp_path, monkeypatch)
    path = tmp_path / "capture_config.json"
    path.write_text("{}")
    packet["clean_config"] = {"capture_config_ref": file_ref(path)}
    path.write_text("drift")
    with pytest.raises(ValueError, match="SHA_DRIFT"):
        reg.authorize_candidate_operation(reg.VISUAL_OPERATION, ["--fixture", "synthetic"], repo_root=tmp_path)


def test_clean_gate_missing_config_not_runnable(tmp_path, monkeypatch):
    packet, _ = clean_gate_fixture(tmp_path, monkeypatch)
    del packet["clean_config"]
    with pytest.raises(reg.RegistrationError, match="CLEAN_CONFIG_REQUIRED"):
        reg.authorize_candidate_operation(reg.VISUAL_OPERATION, ["--fixture", "synthetic"], repo_root=tmp_path)


def controller_gate_fixture(tmp_path, monkeypatch):
    packet, contract = visual_gate_fixture(tmp_path, monkeypatch)
    base_config = {"candidate_id": "B", "algorithm_parameter_change": "RGB_GLOVE_INDEPENDENT_WEARER_BOOTSTRAP_ONLY",
        "stage": "G1_AUTOMATIC_MASK_ONLY", "mask_rule_variant": "RGB_GLOVE_WEARER_SUPPORT_B",
        "image_size": [640, 480], "manual_seed_inputs": 0, "recipe_session_branches": 0,
        "projection_hints": None, "diagnostic_proposal_archive": "TEXT_POINT_PACKBITS_NOT_ALGORITHM_GAIN"}
    indexed, planned = [], []
    for key, count in zip(reg.POST_CONTROLLER_SESSION_KEYS, (238, 183, 200, 139, 145, 128)):
        folder = tmp_path / "source" / key
        folder.mkdir(parents=True)
        report = folder / "SOURCE_REPORT.json"
        report.write_text(json.dumps({"session_id": "cpu_fake_" + key}))
        receipt_path = folder / "RECEIPT.json"
        receipt_path.write_text(json.dumps({"schema": "inpainting-candidate-component-v1", "stage": "Source",
            "status": "SEALED_COMPLETE", "component_signature": "c" * 64,
            "native_frames": count, "outputs": [file_ref(report)]}))
        reference = file_ref(receipt_path)
        indexed.append({"session_key": key, "native_frames": count, "source_receipt": reference})
        planned.append({"session_key": key, "source_receipt": reference, "mask_signature": "d" * 64})
    index_path = tmp_path / "SOURCE_INDEX.json"
    index_path.write_text(json.dumps({"schema": "inpainting-six-source-stage-index-v1",
        "native_frames": 1033, "sessions": indexed}))
    parent_path = tmp_path / "B_PACKET.json"
    parent_path.write_text(json.dumps({"task_id": "inpainting_generalization_g1_mask_wearer_b_20261013",
        "model_stages": ["Mask"], "config": base_config, "sessions": planned}))
    packet.update(task_id="inpainting_post_g1_controller_rgb_model_a01_20261013",
        experiment_scope=reg.POST_CONTROLLER_SCOPE, phase=reg.POST_CONTROLLER_PHASE,
        execution_mode="CANDIDATE_VISUAL_MODEL", model_stages=["Mask"], sessions=planned,
        config={**base_config, "candidate_id": "POST_G1_CONTROLLER_A", "algorithm_parameter_change": "CONTROLLER_TEXT_ONLY",
                "controller_text_prompt": "VR controller"},
        inputs={"source_index": file_ref(index_path), "parent_recipe_packet": file_ref(parent_path)})
    packet["budgets"].update(candidates_max=1, model_calls_unit="SAM_ADAPTER_INFERENCE_API_INVOCATION")
    return packet, contract


def test_post_controller_exact_scope_one_fresh_candidate_same_sources(tmp_path, monkeypatch):
    packet, _ = controller_gate_fixture(tmp_path, monkeypatch)
    assert reg.authorize_candidate_operation(reg.VISUAL_OPERATION, ["--fixture", "synthetic"], repo_root=tmp_path) is packet


@pytest.mark.parametrize("prompt", [None, "", "controller", "VR controllers", "paper cup", {"001": "VR controller"}])
def test_post_controller_only_one_global_prompt(tmp_path, monkeypatch, prompt):
    packet, _ = controller_gate_fixture(tmp_path, monkeypatch)
    packet["config"]["controller_text_prompt"] = prompt
    with pytest.raises(reg.RegistrationError, match="UNIQUE_GLOBAL_PROMPT"):
        reg.authorize_candidate_operation(reg.VISUAL_OPERATION, ["--fixture", "synthetic"], repo_root=tmp_path)


@pytest.mark.parametrize("field,value", [("experiment_scope", "G1_AUTOMATIC_MASK_ONLY"),
    ("phase", "INPAINTING_G1_AUTOMATIC_MASK"), ("model_stages", ["Clean"]),
    ("resume_context", {}), ("original_g1_deadline_at", None)])
def test_post_controller_not_resumed_or_mislabelled_g1(tmp_path, monkeypatch, field, value):
    packet, _ = controller_gate_fixture(tmp_path, monkeypatch)
    packet[field] = value
    with pytest.raises(reg.RegistrationError):
        reg.authorize_candidate_operation(reg.VISUAL_OPERATION, ["--fixture", "synthetic"], repo_root=tmp_path)


@pytest.mark.parametrize("field,value", [("candidates_max", 2), ("candidates_max", True),
    ("gpu_count", 2), ("wall_seconds_max", 28801), ("model_calls_max", 0),
    ("model_calls_unit", reg.CLEAN_MODEL_CALLS_UNIT)])
def test_post_controller_new_budget_stays_independently_finite(tmp_path, monkeypatch, field, value):
    packet, _ = controller_gate_fixture(tmp_path, monkeypatch)
    packet["budgets"][field] = value
    with pytest.raises(reg.RegistrationError, match="FINITE_BOUNDS"):
        reg.authorize_candidate_operation(reg.VISUAL_OPERATION, ["--fixture", "synthetic"], repo_root=tmp_path)


@pytest.mark.parametrize("field,value", [("image_size", [320, 240]), ("manual_seed_inputs", 1),
    ("mask_rule_variant", "NEW_RULE"), ("task_name", "pick_place"), ("projection_hints", {"fake": True})])
def test_post_controller_other_recipe_settings_cannot_change(tmp_path, monkeypatch, field, value):
    packet, _ = controller_gate_fixture(tmp_path, monkeypatch)
    packet["config"][field] = value
    with pytest.raises(reg.RegistrationError, match="OTHER_RECIPE_CONFIG_DRIFT"):
        reg.authorize_candidate_operation(reg.VISUAL_OPERATION, ["--fixture", "synthetic"], repo_root=tmp_path)


def test_post_controller_swapped_source_not_accepted_by_same_count(tmp_path, monkeypatch):
    packet, _ = controller_gate_fixture(tmp_path, monkeypatch)
    packet["sessions"][0]["source_receipt"] = packet["sessions"][1]["source_receipt"]
    with pytest.raises(reg.RegistrationError, match="SOURCE_SWAPPED"):
        reg.authorize_candidate_operation(reg.VISUAL_OPERATION, ["--fixture", "synthetic"], repo_root=tmp_path)


def test_post_controller_resigned_short_source_is_not_full1033(tmp_path, monkeypatch):
    packet, _ = controller_gate_fixture(tmp_path, monkeypatch)
    row = packet["sessions"][0]
    path = Path(row["source_receipt"]["path"])
    shortened = json.loads(path.read_text())
    shortened["native_frames"] = 237
    path.write_text(json.dumps(shortened))
    changed_ref = file_ref(path)
    row["source_receipt"] = changed_ref
    # Re-sign index and parent metadata too: full-count assertion still rejects.
    for name in ("source_index", "parent_recipe_packet"):
        path = Path(packet["inputs"][name]["path"])
        metadata = json.loads(path.read_text())
        metadata["sessions"][0]["source_receipt"] = changed_ref
        if name == "source_index":
            metadata["sessions"][0]["native_frames"] = 237
        path.write_text(json.dumps(metadata))
        packet["inputs"][name] = file_ref(path)
    with pytest.raises(reg.RegistrationError, match="FULL_1033"):
        reg.authorize_candidate_operation(reg.VISUAL_OPERATION, ["--fixture", "synthetic"], repo_root=tmp_path)


@pytest.mark.parametrize("clean", [False, True])
def test_explicit_controller_parameter_cannot_enter_default_g1_or_clean(tmp_path, monkeypatch, clean):
    packet, _ = (clean_gate_fixture if clean else visual_gate_fixture)(tmp_path, monkeypatch)
    packet["config"]["controller_text_prompt"] = "VR controller"
    with pytest.raises(reg.RegistrationError, match="INDEPENDENT_POST_G1"):
        reg.authorize_candidate_operation(reg.VISUAL_OPERATION, ["--fixture", "synthetic"], repo_root=tmp_path)


def test_post_controller_no_active_packet_refuses_before_algorithm_import(tmp_path, monkeypatch):
    controller_gate_fixture(tmp_path, monkeypatch)
    def deny(*args):
        raise reg.RegistrationError("NO_ACTIVE_PACKET")
    monkeypatch.setattr(reg, "_active_packet", deny)
    with pytest.raises(reg.RegistrationError, match="NO_ACTIVE_PACKET"):
        reg.authorize_candidate_operation(reg.VISUAL_OPERATION, ["--fixture", "synthetic"], repo_root=tmp_path)


def test_visual_route_tracks_actual_clean_dependency_change(tmp_path):
    visual_runtime_files(tmp_path)
    initial = reg.visual_route_contract(tmp_path)
    (tmp_path / "vendor/ProPainter/utils/synthetic_inference_dependency.py").write_bytes(b"new PP inference dependency")
    assert initial != reg.visual_route_contract(tmp_path)
