"""Synthetic CPU producer fixtures; never publish production inputs or run a model."""
import hashlib
import importlib.util
import io
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from chaoyang.humanego_ict import evaluation_current_producer_v1 as producer
from chaoyang.humanego_ict.inference import build_hand_ict


PROJECT = Path(__file__).resolve().parents[2]
BUILDER = PROJECT / "_run/maintenance/humanego_ict_evaluation_engineering_20261011/attempt_0001/prepared_code/evaluation_inputs_v1.py"
IDENTITY = {"method": "IDENTITY_NO_FIT", "position_mean": [0, 0, 0], "position_std": [1, 1, 1]}


def frozen_dependencies():
    source = BUILDER.read_bytes()
    assert hashlib.sha256(source).hexdigest() == producer.BUILDER_SHA256
    spec = importlib.util.spec_from_file_location("synthetic_verified_current_builder", BUILDER)
    module = importlib.util.module_from_spec(spec)
    exec(compile(source, str(BUILDER), "exec"), module.__dict__)
    return SimpleNamespace(np=np, Image=Image, read_current_fields=module.read_current_fields_npz,
        build_current_arrays=module.build_current_input_arrays)


def write_bytes(path, blob):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)
    return {"path": str(path), "bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()}


def write_json(path, value):
    return write_bytes(path, json.dumps(value, sort_keys=True).encode())


class Fixture:
    def __init__(self, root, *, rows=1):
        self.root = root
        self.root.mkdir(exist_ok=True)
        self.output = root / "output"
        self.fields, self.samples, self.rows = {}, [], []
        self.clock = {"clock_domain": "SYNTHETIC_NATIVE_NS", "timestamp_unit": "nanoseconds",
            "pose_frame_id": "FIXTURE_WORLD", "camera_frame_id": "FIXTURE_REFERENCE_CAMERA",
            "position_unit": "meters", "tcp_frame_ids": ["FIXTURE_LEFT_PRIOR", "FIXTURE_RIGHT_PRIOR"]}
        for i in range(rows):
            sid = f"fixture_{i:03d}_f000003"
            session, group = f"{i:03d}", f"synthetic_capture:{i:03d}"
            wrists = np.repeat(np.eye(4)[None], 2, axis=0)
            wrists[0, :3, 3] = [.1, .2, .3]
            wrists[1, :3, 3] = [-.2, .1, .4]
            current = build_hand_ict(wrists, np.ones(2, dtype=bool), np.eye(4),
                normalization=IDENTITY, source_qpc_ns=np.array([90, 91]), anchor_qpc_ns=100)
            fields = {k: current[k] for k in ("x_ict", "ict_mask", "anchor_qpc_ns",
                "conditioning_source_qpc_ns", "reference_camera_to_world")}
            self.fields[sid] = fields
            buf = io.BytesIO()
            # If accidentally decoded, this future member fails allow_pickle=False.
            np.savez(buf, **fields, y_action=np.array([{"must": "never_decode_future"}], dtype=object),
                target_qpc_ns=np.array([999]), action_valid_mask=np.array([False]))
            labels = write_bytes(root / "sources" / session / "labels.npz", buf.getvalue())
            row = {"sample_id": sid, "session_id": session, "source_frame_id": 3, "source_group_id": group}
            self.rows.append(row)
            rgbs, receipts = {}, {}
            for branch in ("raw", "robotized"):
                image = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
                if branch == "robotized":
                    image = 255 - image
                buf = io.BytesIO(); Image.fromarray(image).save(buf, format="PNG")
                rgb = write_bytes(root / "sources" / session / (branch + ".png"), buf.getvalue())
                rgbs[branch] = rgb
                causal = {"schema": "causal-training-input-v1", "status": "PASSED", "branch": branch,
                    "sample_id": sid, "source_session_id": session, "source_frame_id": 3,
                    "anchor_qpc_ns": 100, "image_sha256": rgb["sha256"], "labels_sha256": labels["sha256"],
                    "input_class": "CAUSAL_TRAINING_INPUT", "timestamp_unit": "nanoseconds",
                    "clock_domain": self.clock["clock_domain"], "donor_frame_ids": [3],
                    "physical_eye_identity_verified": False, "geometry_qualification": "UNKNOWN",
                    "time_trust_boundary": "SOURCE_RECORD_DECLARATION_NOT_COUNTER_IMPLEMENTATION_PROOF",
                    "evidence_refs": [labels, rgb], "control_ground_truth": False,
                    "physical_deployment_authorized": False, "fixture_only": True}
                receipts[branch] = write_json(root / "sources" / session / (branch + "_causal.json"), causal)
            self.samples.append({**row, "split": "validation", "branch_admission": {"raw": True, "robotized": True},
                "labels": labels, "rgb": rgbs, "causal_receipts": receipts})
        self.refresh()

    def refresh(self):
        self.manifest_ref = write_json(self.root / "PRODUCTION_MANIFEST.json", {
            "schema": producer.MANIFEST_SCHEMA, "fixture_only": True, "normalization": IDENTITY,
            "supervision_class": "DERIVED_RETARGET", "training_scope": "NON_DEPLOYABLE_DERIVED_MOTION_TRAINING",
            "control_ground_truth": False, "physical_deployment_authorized": False,
            "pose_clock": self.clock, "samples": self.samples})
        self.spec_ref = write_json(self.root / "EVALUATION_INPUT_SPEC.json", {
            "schema": producer.SPEC_SCHEMA, "fixture_only": True, "normalization": IDENTITY,
            "manifest_ref": self.manifest_ref, "common_validation_rows": self.rows})

    def replace_causal(self, change, *, branch="raw"):
        ref = self.samples[0]["causal_receipts"][branch]
        obj = json.loads(Path(ref["path"]).read_bytes()); change(obj)
        self.samples[0]["causal_receipts"][branch] = write_json(Path(ref["path"]), obj)
        self.refresh()

    def run(self, *, branch="raw", output=None, deps=None):
        return producer.produce_synthetic_fixture_inputs(self.spec_ref, self.manifest_ref,
            branch=branch, output_root=self.output if output is None else output,
            project_root=self.root, dependencies=frozen_dependencies() if deps is None else deps)

    def metadata_authority(self, *, packet_change=None, authority_change=None, state_change=None, index_change=None):
        """Only synthetic JSON authority predicate fixtures, not a run context."""
        stat = Path("/proc/self/stat").read_text()
        rest = stat[stat.rfind(")") + 2:].split()
        task_id = "synthetic_cpu_metadata_only"
        writer = {"pid": int(stat.split(" ", 1)[0]), "proc_start_ticks": int(rest[19]),
            "executor_epoch": 1, "fencing_token": "fixture_fence", "scope": task_id}
        builder_ref = write_bytes(self.root / "frozen/evaluation_inputs_v1.py", BUILDER.read_bytes())
        writer_file = self.root / "WRITER.json"; write_json(writer_file, writer)
        packet_path = self.root / "tasks/current" / task_id / "TASK_PACKET.json"
        command = [producer.OPERATION, "--task-packet", str(packet_path), "--spec", self.spec_ref["path"],
            "--manifest", self.manifest_ref["path"], "--branch", "raw", "--output-dir", str(self.output)]
        packet = {"task_id": task_id, "execution_mode": producer.EXECUTION_MODE, "branch": "raw",
            "physical_deployment_authorized": False, "control_ground_truth": False,
            "inputs": {"evaluation_input_spec": self.spec_ref, "production_manifest": self.manifest_ref},
            "output_dir": str(self.output), "deadline_at": (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat(),
            "budgets": {"gpu_count": 0, "model_calls_max": 0, "wall_seconds_max": 600, "cpu_threads_per_worker": 2},
            "writer": writer, "writer_file": str(writer_file), "run_signature": "a" * 64,
            "runtime_code": [builder_ref], "commands": ["/usr/local/bin/python -m chaoyang.cli run " + " ".join(command)]}
        if packet_change: packet_change(packet)
        packet_ref = write_json(packet_path, packet)
        index_row = {"task_id": task_id, "execution_allowed": True, "packet_sha256": packet_ref["sha256"],
            "packet_path": str(packet_path.relative_to(self.root))}
        index = {"task_packets": [index_row]}
        if index_change: index_change(index)
        state = {"tasks": [{"task_id": task_id, "status": "RUNNING", "run_signature": packet["run_signature"],
            "pid": writer["pid"], "proc_start_ticks": writer["proc_start_ticks"]}]}
        if state_change: state_change(state)
        files = {"task_packet_index_current": write_json(self.root / "INDEX.json", index),
            "task_state": write_json(self.root / "STATE.json", state),
            "algorithm_contract": write_json(self.root / "CONTRACT.json", {"special_status_contracts": {
                producer.ROUTE: {"candidate_operation": producer.OPERATION, "task_packet_required": True,
                    "physical_deployment_authorized": False, "code_closure": [builder_ref]}}})}
        write_json(self.root / "docs/governance/CURRENT_STATUS_RECEIPT.json", {"files": files})
        authority = {"status": "PASSED", "execution_mode": producer.EXECUTION_MODE,
            "production_inputs_write_allowed": True, "task_packet_ref": packet_ref,
            "task_id": task_id, "run_signature": packet["run_signature"], "writer": writer}
        if authority_change: authority_change(authority)
        request = {"evaluation_input_spec_ref": self.spec_ref, "production_manifest_ref": self.manifest_ref,
            "branch": "raw", "output_root": str(self.output)}
        return authority, request, packet


@pytest.fixture
def data(tmp_path):
    return Fixture(tmp_path / "project")


@pytest.mark.parametrize("branch", ["raw", "robotized"])
def test_actual_image_decode_same_preprocessing_exact_seven_arrays_no_future_copy(data, branch):
    result = data.run(branch=branch)
    manifest = json.loads(Path(result["manifest_ref"]["path"]).read_bytes())
    row = manifest["samples"][0]
    with np.load(row["current_input_npz"]["path"], allow_pickle=False) as archive:
        assert len(archive.files) == 7 and set(archive.files) == producer.CURRENT_KEYS
        with Image.open(data.samples[0]["rgb"][branch]["path"]) as image:
            rgb = np.asarray(image.convert("RGB").resize((320, 240), Image.Resampling.BILINEAR),
                dtype=np.float32).transpose(2, 0, 1)[None] / np.float32(255.)
        expected = frozen_dependencies().build_current_arrays(data.fields[row["sample_id"]],
            x_rgb=rgb, source_frame_id=3, normalization=IDENTITY)
        for key in producer.CURRENT_KEYS:
            np.testing.assert_array_equal(archive[key], expected[key])
    receipt = json.loads(Path(row["input_receipt"]["path"]).read_bytes())
    assert receipt["source_rgb_decoded"] is True and receipt["source_rgb_size"] == [7, 5]
    assert receipt["future_target_members_decoded"] == [] and not receipt["source_label_archive_copied"]
    assert receipt["physical_eye_identity_verified"] is False and receipt["geometry_qualification"] == "UNKNOWN"
    assert receipt["time_trust_boundary"] == "SOURCE_RECORD_DECLARATION_NOT_COUNTER_IMPLEMENTATION_PROOF"
    assert receipt["source_labels_ref"] == data.samples[0]["labels"]
    assert not list(data.output.glob("*labels*"))
    assert result["sample_count"] == 1 and result["gpu_calls"] == result["model_forward_calls"] == 0
    assert result["fixture_only"] is True and manifest["fixture_only"] is True
    for ref in (row["current_input_npz"], row["input_receipt"], result["manifest_ref"]):
        blob = Path(ref["path"]).read_bytes()
        assert len(blob) == ref["bytes"] and hashlib.sha256(blob).hexdigest() == ref["sha256"]


def test_two_rows_keep_original_order_and_independent_group_count(tmp_path):
    data = Fixture(tmp_path / "project", rows=2)
    result = data.run(); manifest = json.loads(Path(result["manifest_ref"]["path"]).read_bytes())
    assert [row["sample_id"] for row in manifest["samples"]] == [row["sample_id"] for row in data.rows]
    assert result["source_group_count"] == 2


@pytest.mark.parametrize("field,value", [("session_id", "other"), ("source_frame_id", 4),
    ("source_group_id", "other"), ("split", "train")])
def test_frozen_identity_and_validation_admission_not_reselected(data, field, value):
    data.samples[0][field] = value; data.refresh()
    with pytest.raises(producer.CurrentProducerError, match="IDENTITY"):
        data.run()
    assert not data.output.exists()


@pytest.mark.parametrize("branch", ["raw", "robotized"])
def test_common_anchor_requires_both_original_branch_admissions(data, branch):
    data.samples[0]["branch_admission"][branch] = False; data.refresh()
    with pytest.raises(producer.CurrentProducerError, match="IDENTITY"):
        data.run()


@pytest.mark.parametrize("role", ["labels", "rgb", "causal"])
def test_source_byte_drift_rejected_and_partial_has_no_final_manifest(data, role):
    sample = data.samples[0]
    ref = sample["labels"] if role == "labels" else sample["rgb"]["raw"] if role == "rgb" else sample["causal_receipts"]["raw"]
    Path(ref["path"]).write_bytes(b"drift")
    with pytest.raises(producer.CurrentProducerError, match="SOURCE_SHA_OR_SIZE_DRIFT"):
        data.run()
    assert not (data.output / "CURRENT_INPUTS_MANIFEST.json").exists()


@pytest.mark.parametrize("field,value", [("branch", "robotized"), ("sample_id", "wrong"),
    ("source_session_id", "wrong"), ("source_frame_id", 1), ("image_sha256", "0" * 64),
    ("labels_sha256", "0" * 64), ("control_ground_truth", True), ("physical_deployment_authorized", True)])
def test_source_causal_identity_and_qualification_never_upgraded(data, field, value):
    data.replace_causal(lambda obj: obj.update({field: value}))
    with pytest.raises(producer.CurrentProducerError, match="CAUSAL_IDENTITY"):
        data.run()


@pytest.mark.parametrize("donors", [[], [4], [-1], [True]])
def test_future_donor_or_unstated_rgb_source_rejected(data, donors):
    data.replace_causal(lambda obj: obj.update(donor_frame_ids=donors))
    with pytest.raises(producer.CurrentProducerError, match="RGB_DONOR"):
        data.run()


def test_anchor_mismatch_not_overwritten(data):
    data.replace_causal(lambda obj: obj.update(anchor_qpc_ns=101))
    with pytest.raises(producer.CurrentProducerError, match="ANCHOR_TIME"):
        data.run()


@pytest.mark.parametrize("key,value", [("clock_domain", "other"), ("position_unit", "mm"), ("timestamp_unit", "seconds")])
def test_source_clock_frame_unit_contract_preserved(data, key, value):
    data.clock[key] = value; data.refresh()
    with pytest.raises(producer.CurrentProducerError, match="CLOCK_CONTRACT"):
        data.run()


def test_actual_rgb_decode_not_just_file_hash(data):
    sample = data.samples[0]
    sample["rgb"]["raw"] = write_bytes(Path(sample["rgb"]["raw"]["path"]), b"not an image")
    data.replace_causal(lambda obj: obj.update(image_sha256=sample["rgb"]["raw"]["sha256"]))
    with pytest.raises((OSError, ValueError)):
        data.run()
    assert not (data.output / "CURRENT_INPUTS_MANIFEST.json").exists()


def test_different_size_valid_rgb_is_not_resized_with_another_rule(data):
    test_actual_image_decode_same_preprocessing_exact_seven_arrays_no_future_copy(data, "raw")


def test_output_repeat_cannot_overwrite_or_reset_reservation(data):
    first = data.run(); old = Path(first["manifest_ref"]["path"]).read_bytes()
    with pytest.raises(FileExistsError):
        data.run()
    assert Path(first["manifest_ref"]["path"]).read_bytes() == old


@pytest.mark.parametrize("outside", [True, False])
def test_output_cannot_be_project_root_or_external(data, outside):
    output = data.root.parent / "external" if outside else data.root
    with pytest.raises(producer.CurrentProducerError, match="PROJECT_SUBPATH"):
        data.run(output=output)


def test_output_symlink_rejected(data):
    outside = data.root.parent / "external"; outside.mkdir()
    link = data.root / "link"; link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(producer.CurrentProducerError, match="SYMLINK"):
        data.run(output=link / "output")
    assert not (outside / "output").exists()


def test_source_symlink_rejected_even_same_bytes(data):
    sample = data.samples[0]; old = Path(sample["labels"]["path"])
    link = old.parent / "alias.npz"; link.symlink_to(old)
    sample["labels"] = {**sample["labels"], "path": str(link)}; data.refresh()
    with pytest.raises(producer.CurrentProducerError, match="SYMLINK"):
        data.run()


def test_duplicate_manifest_samples_and_frozen_ids_rejected(data):
    data.samples.append(dict(data.samples[0])); data.refresh()
    with pytest.raises(producer.CurrentProducerError, match="DUPLICATE_PRODUCTION_SAMPLE"):
        data.run()
    data.samples.pop(); data.rows.append(dict(data.rows[0])); data.refresh()
    with pytest.raises(producer.CurrentProducerError, match="UNIQUE_FROZEN_ANCHORS"):
        data.run()


@pytest.mark.parametrize("sid", ["../escape", "/absolute", "with space", "nested/name"])
def test_sample_id_cannot_escape_output(data, sid):
    data.samples[0]["sample_id"] = sid; data.rows[0]["sample_id"] = sid; data.refresh()
    with pytest.raises(producer.CurrentProducerError, match="UNSAFE_SAMPLE_ID"):
        data.run()


def test_missing_or_extra_builder_fields_cannot_leak_future(data):
    deps = frozen_dependencies(); original = deps.build_current_arrays
    deps.build_current_arrays = lambda *args, **kwargs: {**original(*args, **kwargs), "y_action": np.zeros(1)}
    with pytest.raises(producer.CurrentProducerError, match="SEVEN_CURRENT_KEYS"):
        data.run(deps=deps)


def test_source_reader_current_allowlist_is_used_not_npz_copy(data):
    deps = frozen_dependencies(); original = deps.read_current_fields; seen = []
    def read(ref, **kwargs):
        seen.append(ref)
        current = original(ref, **kwargs)
        assert set(current) == {"x_ict", "ict_mask", "anchor_qpc_ns", "conditioning_source_qpc_ns", "reference_camera_to_world"}
        return current
    deps.read_current_fields = read
    data.run(deps=deps)
    assert seen == [data.samples[0]["labels"]]


def test_staging_or_pass_only_does_not_authorize_real_production_before_imports(data, monkeypatch):
    def no_import(*args):
        raise AssertionError("numerical import before real admission")
    monkeypatch.setattr(producer, "_real_dependencies", no_import)
    for auth in [None, lambda request: {"status": "PASSED"}, lambda request: {
        "status": "PASSED", "execution_mode": producer.EXECUTION_MODE,
        "production_inputs_write_allowed": True, "fixture_only": True}]:
        with pytest.raises(producer.CurrentProducerError):
            producer.produce_current_inputs(data.spec_ref, data.manifest_ref, branch="raw",
                output_root=data.output, project_root=data.root, authorization_verifier=auth)
    assert not data.output.exists()


def test_synthetic_source_cannot_enter_actual_metadata_path(data):
    # Test the real metadata rule without admitting a writer or importing a model.
    request = {"evaluation_input_spec_ref": data.spec_ref, "production_manifest_ref": data.manifest_ref,
        "branch": "raw", "output_root": str(data.output)}
    with pytest.raises(producer.CurrentProducerError, match="FIXTURE_SCOPE"):
        producer._produce(request, data.root, data.output, frozen_dependencies(), {}, lambda: None, fixture_only=False)
    assert not data.output.exists()


def test_final_source_drift_does_not_publish_a_complete_manifest(data, monkeypatch):
    original = producer._write_json
    def mutate(path, value, root):
        ref = original(path, value, root)
        if path.name.endswith(".input_receipt.json"):
            Path(data.samples[0]["labels"]["path"]).write_bytes(b"drift after row")
        return ref
    monkeypatch.setattr(producer, "_write_json", mutate)
    with pytest.raises(producer.CurrentProducerError, match="SOURCE_SHA_OR_SIZE_DRIFT"):
        data.run()
    assert not (data.output / "CURRENT_INPUTS_MANIFEST.json").exists()


def test_signed_registered_metadata_uses_real_field_types_only_without_production(data):
    authority, request, packet = data.metadata_authority()
    admitted = producer._real_authorization(lambda request: authority, request, data.root)
    assert admitted["task_id"] == packet["task_id"] and admitted["current_input_builder_ref"]["sha256"] == producer.BUILDER_SHA256
    assert not data.output.exists()


@pytest.mark.parametrize("change,code", [
    (lambda p: p.update(execution_mode="PREPARING_NOT_PRODUCTION_AUTHORITY"), "TASK_MODE"),
    (lambda p: p.update(branch="robotized"), "TASK_MODE"),
    (lambda p: p.update(deadline_at="2020-01-01T00:00:00Z"), "EXPIRED"),
    (lambda p: p.update(deadline_at="2099-01-01T00:00:00"), "EXPIRED"),
    (lambda p: p["budgets"].update(gpu_count=True), "CPU_ONLY_BUDGET"),
    (lambda p: p["budgets"].update(model_calls_max=1), "CPU_ONLY_BUDGET"),
    (lambda p: p["budgets"].update(wall_seconds_max=7201), "CPU_ONLY_BUDGET"),
    (lambda p: p["budgets"].update(cpu_threads_per_worker=3), "CPU_ONLY_BUDGET"),
    (lambda p: p["writer"].update(executor_epoch="1"), "WRITER_IDENTITY"),
    (lambda p: p["inputs"].update(production_manifest={}), "SOURCE_OUTPUT_BINDING"),
    (lambda p: p.update(commands=["python -m chaoyang.cli run wrong_operation"]), "COMMAND"),
])
def test_cpu_authority_rejects_unregistered_wrong_expired_or_mutated_packet(data, change, code):
    authority, request, packet = data.metadata_authority(packet_change=change)
    with pytest.raises(producer.CurrentProducerError, match=code):
        producer._real_authorization(lambda request: authority, request, data.root)
    assert not data.output.exists()


def test_authority_flag_does_not_override_actual_live_state_or_duplicate_writer(data):
    authority, request, packet = data.metadata_authority(state_change=lambda s: s["tasks"].append(dict(s["tasks"][0])))
    with pytest.raises(producer.CurrentProducerError, match="LIVE_UNIQUE"):
        producer._real_authorization(lambda request: authority, request, data.root)


def test_task_index_must_enable_exact_unique_sha_bound_packet(data):
    authority, request, packet = data.metadata_authority(index_change=lambda s: s["task_packets"][0].update(packet_sha256="0" * 64))
    with pytest.raises(producer.CurrentProducerError, match="REGISTERED_CPU_PACKET"):
        producer._real_authorization(lambda request: authority, request, data.root)


def test_fencing_file_drift_stops_real_authority(data):
    authority, request, packet = data.metadata_authority()
    write_json(Path(packet["writer_file"]), {**packet["writer"], "fencing_token": "changed"})
    with pytest.raises(producer.CurrentProducerError, match="FENCE_DRIFT"):
        producer._real_authorization(lambda request: authority, request, data.root)


def test_producer_frame_row_cannot_carry_future_labels_even_in_json(data):
    data.rows[0]["y_action"] = [123]; data.refresh()
    with pytest.raises(producer.CurrentProducerError, match="NO_FUTURE_MEMBERS"):
        data.run()


def test_duplicate_json_keys_are_not_silently_overwritten(data):
    blob = b'{"schema":"wrong","schema":"also-wrong"}'
    data.spec_ref = write_bytes(data.root / "EVALUATION_INPUT_SPEC.json", blob)
    with pytest.raises(producer.CurrentProducerError, match="DUPLICATE_JSON_KEY"):
        data.run()


def test_numeric_source_current_context_must_still_pass_frozen_adapter(data):
    original = data.fields[data.rows[0]["sample_id"]]
    original["conditioning_source_qpc_ns"][0] = 101
    buf = io.BytesIO(); np.savez(buf, **original)
    data.samples[0]["labels"] = write_bytes(Path(data.samples[0]["labels"]["path"]), buf.getvalue())
    data.replace_causal(lambda obj: obj.update(labels_sha256=data.samples[0]["labels"]["sha256"]))
    with pytest.raises(ValueError, match="CONDITIONING_TIMESTAMP"):
        data.run()


@pytest.mark.parametrize("mode", ["L", "RGBA"])
def test_source_color_conversion_matches_training_instead_of_new_mode_gate(data, mode):
    buf = io.BytesIO(); Image.new(mode, (9, 8)).save(buf, format="PNG")
    sample = data.samples[0]
    sample["rgb"]["raw"] = write_bytes(Path(sample["rgb"]["raw"]["path"]), buf.getvalue())
    data.replace_causal(lambda obj: obj.update(image_sha256=sample["rgb"]["raw"]["sha256"]))
    result = data.run(); manifest = json.loads(Path(result["manifest_ref"]["path"]).read_bytes())
    receipt = json.loads(Path(manifest["samples"][0]["input_receipt"]["path"]).read_bytes())
    assert receipt["source_rgb_original_mode"] == mode and receipt["source_rgb_size"] == [9, 8]
    with np.load(manifest["samples"][0]["current_input_npz"]["path"], allow_pickle=False) as archive:
        with Image.open(sample["rgb"]["raw"]["path"]) as image:
            expected = np.asarray(image.convert("RGB").resize((320, 240), Image.Resampling.BILINEAR),
                dtype=np.float32).transpose(2, 0, 1)[None] / np.float32(255.)
        np.testing.assert_array_equal(archive["x_rgb"], expected)


@pytest.mark.parametrize("field,value", [("source_frame_id", True), ("source_frame_id", 2**63),
    ("source_group_id", []), ("sample_id", None)])
def test_bad_native_row_types_rejected_before_hash_sets_or_output(data, field, value):
    data.rows[0][field] = value; data.refresh()
    with pytest.raises(producer.CurrentProducerError, match="NATIVE_ANCHOR_IDENTITY"):
        data.run()
    assert not data.output.exists()
