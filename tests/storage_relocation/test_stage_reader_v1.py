"""Tiny project-local fixtures; no NAS, models, production data, or writes API."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from chaoyang.pick_place_1008.audit import validate_action_arrays
from chaoyang.pick_place_1008.stage_reuse import SCHEMA, _canonical_hash, _validate
from chaoyang.storage_relocation.stage_reader_v1 import (
    validate_sealed_stage,
    validate_session_actions,
)


def _ref(path):
    path = Path(path)
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


class FixtureReader:
    """Minimal strict protocol implementation, not a NAS binding substitute."""

    def __init__(self, repo, mapping=None):
        self.repo = repo
        self.mapping = mapping or {}

    def reference_for(self, path):
        path = Path(path)
        if str(path) in self.mapping:
            return copy.deepcopy(self.mapping[str(path)][0])
        return _ref(path)

    def verify_ref(self, pin):
        self.physical_path(pin)
        return copy.deepcopy(pin)

    def physical_path(self, pin):
        if not isinstance(pin, dict) or set(pin) != {"path", "bytes", "sha256"}:
            raise RuntimeError("FIXTURE_EXACT_REF_REQUIRED")
        logical = Path(pin["path"])
        if not logical.is_absolute() or ".." in logical.parts:
            raise RuntimeError("FIXTURE_LOGICAL_PATH_REJECTED")
        if str(logical) in self.mapping:
            original, physical = self.mapping[str(logical)]
            if pin != original:
                raise RuntimeError("FIXTURE_FROZEN_REF_CHANGED")
        else:
            physical = logical
            if not physical.is_relative_to(self.repo):
                raise RuntimeError("FIXTURE_UNMAPPED_EXTERNAL_PATH")
        for component in (physical, *physical.parents):
            if component == self.repo:
                break
            if component.is_symlink():
                raise RuntimeError("FIXTURE_SYMLINK")
        actual = _ref(physical)
        if actual["bytes"] != pin["bytes"] or actual["sha256"] != pin["sha256"]:
            raise RuntimeError("FIXTURE_BYTES_OR_SHA_DRIFT")
        return physical

    def read_json(self, pin):
        return json.loads(self.physical_path(pin).read_text(encoding="utf-8-sig"))


class SealedFixture:
    def __init__(self, repo, stage="motion"):
        self.repo, self.sid, self.stage = repo, "001", stage
        self.tid = "exact78_pick_place_fixture_v9_20261010"
        self.base = repo / "_run/current" / self.tid / "attempts/attempt_0001"
        self.root = self.base / "candidate"
        self.folder = self.root / "sessions" / self.sid
        self.folder.mkdir(parents=True)
        self.actions = self.folder / "actions.npz"
        np.savez(self.actions, q_arm=np.zeros((2, 2, 7)),
                 q_hand=np.zeros((2, 2, 22)), T_world_base=np.eye(4),
                 c2w=np.repeat(np.eye(4)[None], 2, axis=0))
        self.packet_path = repo / "tasks/current" / self.tid / "TASK_PACKET.json"
        self.packet = {"task_id": self.tid, "run_signature": "fixed-fixture",
                       "execution_mode": "EXACT78_CANDIDATE"}
        _json(self.packet_path, self.packet)
        code = repo / "src/frozen_code.py"
        code.parent.mkdir(parents=True)
        code.write_bytes(b"# tiny frozen fixture\n")
        raw = repo / "raw/physical_eye.mp4"
        raw.parent.mkdir(parents=True)
        raw.write_bytes(b"tiny raw provenance, not decoded")
        camera = repo / "camera.json"
        _json(camera, {"camera": "fixture-not-calibration"})
        binding = {"stage": self.stage, "session_id": self.sid,
                   "raw_frame_count": 2, "code_and_dependencies": [_ref(code)],
                   "raw_source": {"video": _ref(raw)},
                   "camera_config": _ref(camera), "upstream": {}}
        self.receipt = {"schema": SCHEMA, "status": "COMPLETE",
                        "binding": binding, "binding_sha256": _canonical_hash(binding),
                        "files": {"actions.npz": _ref(self.actions)},
                        "producer": {"task_id": self.tid,
                                     "task_packet": _ref(self.packet_path),
                                     "run_signature": "fixed-fixture",
                                     "result_path": str(self.base / "RESULT.json")}}
        self.stage_path = self.folder / ("STAGE_" + stage + ".json")
        self.index = {"sessions": [{"session_id": self.sid,
                                    "stage_receipts": {}, "stages": {}}]}
        self.terminal = {"status": "PASSED", "task_id": self.tid,
                         "run_signature": "fixed-fixture", "required_outputs": {},
                         "task_packet": _ref(self.packet_path)}
        self.ledger_row = {"task_id": self.tid, "status": "PASSED",
                           "pid": None, "proc_start_ticks": None}
        self.required = ("actions.npz",)
        self.rebind()

    def rebind(self, *, corrupt_digest=False, index_edit=None, terminal_edit=None):
        _json(self.packet_path, self.packet)
        packet_ref = _ref(self.packet_path)
        self.receipt["producer"]["task_packet"] = packet_ref if isinstance(
            self.receipt.get("producer"), dict) else None
        binding = self.receipt.get("binding")
        self.expected = copy.deepcopy(binding)
        self.receipt["binding_sha256"] = "wrong" if corrupt_digest else _canonical_hash(binding)
        _json(self.stage_path, self.receipt)
        stage_ref = _ref(self.stage_path)
        self.index["sessions"][0]["stage_receipts"] = {self.stage: stage_ref}
        self.index["sessions"][0]["stages"] = {self.stage: {"receipt": stage_ref}}
        if index_edit:
            index_edit(self.index)
        index_path = self.root / "INDEX.json"
        _json(index_path, self.index)
        self.terminal["task_packet"] = packet_ref
        self.terminal["required_outputs"] = {"candidate/INDEX.json": _ref(index_path)}
        if terminal_edit:
            terminal_edit(self.terminal)
        _json(self.base / "RESULT.json", self.terminal)
        self.ledger_row["result"] = _ref(self.base / "RESULT.json")
        _json(self.repo / "docs/governance/LONG_HORIZON_TASK_STATE.json",
              {"tasks": [self.ledger_row]})

    def original(self, expected=None):
        return _validate(self.repo, self.root, self.sid, self.stage,
                         self.expected if expected is None else expected, self.required)

    def adapted(self, reader=None, expected=None):
        return validate_sealed_stage(reader or FixtureReader(self.repo), self.repo,
                                     self.root, self.sid, self.stage,
                                     self.expected if expected is None else expected,
                                     self.required)


@pytest.fixture
def sealed(tmp_path):
    return SealedFixture(tmp_path / "project")


def test_complete_proof_matches_original(sealed):
    assert sealed.adapted() == sealed.original()
    assert sealed.adapted()[1] == _ref(sealed.stage_path)


@pytest.mark.parametrize("stage", ["source", "motion", "gpu", "render"])
def test_stage_identity_gate_parity(tmp_path, stage):
    fixture = SealedFixture(tmp_path / "project", stage)
    assert fixture.adapted() == fixture.original()


def test_nested_only_stage_reference_keeps_original_contract(sealed):
    sealed.rebind(index_edit=lambda value: value["sessions"][0].update(stage_receipts={}))
    assert sealed.adapted() == sealed.original()


def test_failed_runtime_terminal_retains_original_stage_semantics(sealed):
    # Session/product adoption is a separate gate; _validate's accepted producer
    # terminal set must not silently be changed by this storage-only adapter.
    sealed.ledger_row["status"] = sealed.terminal["status"] = "FAILED_RUNTIME_FINAL"
    sealed.rebind()
    assert sealed.adapted() == sealed.original()


def test_relocated_payload_keeps_original_receipt_and_folder(sealed):
    before = sealed.original()
    pin = copy.deepcopy(sealed.receipt["files"]["actions.npz"])
    physical = sealed.repo / "fixture_export/actions.npz"
    physical.parent.mkdir()
    physical.write_bytes(sealed.actions.read_bytes())
    reader = FixtureReader(sealed.repo, {str(sealed.actions): (pin, physical)})
    # Only this synthetic fixture file is removed, never any project data.
    sealed.actions.unlink()
    assert sealed.adapted(reader) == before
    assert sealed.adapted(reader)[0] == sealed.folder
    assert sealed.adapted(reader)[1]["path"] == str(sealed.stage_path)
    assert validate_session_actions(reader, pin, 2) == []


def test_missing_stage_is_same_cache_miss(sealed):
    sealed.stage_path.unlink()
    assert sealed.original() == sealed.adapted() == (None, None)


def test_different_intact_binding_is_same_cache_miss(sealed):
    expected = copy.deepcopy(sealed.expected)
    expected["raw_frame_count"] = 3
    assert sealed.original(expected) == sealed.adapted(expected=expected) == (None, None)


@pytest.mark.parametrize("case,reason", [
    ("schema", "IDENTITY_SCHEMA_OR_INCOMPLETE_STAGE"),
    ("partial", "IDENTITY_SCHEMA_OR_INCOMPLETE_STAGE"),
    ("binding_missing", "BINDING_MISSING"),
    ("digest", "BINDING_DIGEST_TAMPERED"),
    ("stage_identity", "STAGE_SESSION_OR_FRAME_IDENTITY_MISMATCH"),
    ("session_identity", "STAGE_SESSION_OR_FRAME_IDENTITY_MISMATCH"),
    ("frame_zero", "STAGE_SESSION_OR_FRAME_IDENTITY_MISMATCH"),
    ("frame_bool", "STAGE_SESSION_OR_FRAME_IDENTITY_MISMATCH"),
    ("output_set", "MANDATORY_OUTPUT_SET_MISMATCH"),
    ("output_escape", "MANDATORY_OUTPUT_NAME_ESCAPE"),
    ("output_wrong_path", "ARTIFACT_PATH_MISMATCH"),
    ("code_empty", "CODE_DEPENDENCY_CLOSURE_MISSING"),
    ("raw_empty", "RAW_SOURCE_CLOSURE_MISSING"),
    ("upstream_missing", "UPSTREAM_CLOSURE_MISSING"),
    ("producer_task", "PRODUCER_TASK_IDENTITY"),
    ("packet_mode", "PRODUCER_PACKET_IDENTITY_OR_SIGNATURE"),
    ("packet_signature", "PRODUCER_PACKET_IDENTITY_OR_SIGNATURE"),
    ("result_path", "PRODUCER_TERMINAL_PATH_MISMATCH"),
    ("ledger_unsealed", "PRODUCER_NOT_SEALED_TERMINAL"),
    ("writer_pid", "PRODUCER_TERMINAL_WRITER_PRESENT"),
    ("writer_ticks", "PRODUCER_TERMINAL_WRITER_PRESENT"),
    ("terminal_identity", "PRODUCER_TERMINAL_IDENTITY_OR_SIGNATURE"),
    ("index_missing", "TERMINAL_INDEX_REFERENCE_MISSING"),
    ("index_session", "PRODUCER_INDEX_SESSION_IDENTITY"),
    ("index_conflict", "PRODUCER_INDEX_STAGE_REFERENCE_CONFLICT"),
    ("index_unbound", "PRODUCER_INDEX_DOES_NOT_BIND_STAGE"),
])
def test_original_gate_rejection_parity(sealed, case, reason):
    index_edit = terminal_edit = None
    if case == "schema": sealed.receipt["schema"] = "wrong"
    elif case == "partial": sealed.receipt["status"] = "PARTIAL"
    elif case == "binding_missing": sealed.receipt["binding"] = None
    elif case == "stage_identity": sealed.receipt["binding"]["stage"] = "gpu"
    elif case == "session_identity": sealed.receipt["binding"]["session_id"] = "002"
    elif case == "frame_zero": sealed.receipt["binding"]["raw_frame_count"] = 0
    elif case == "frame_bool": sealed.receipt["binding"]["raw_frame_count"] = True
    elif case == "output_set": sealed.receipt["files"] = {}
    elif case == "output_escape":
        sealed.receipt["files"] = {"../actions.npz": sealed.receipt["files"]["actions.npz"]}
        sealed.required = ("../actions.npz",)
    elif case == "output_wrong_path": sealed.receipt["files"]["actions.npz"]["path"] += ".wrong"
    elif case == "code_empty": sealed.receipt["binding"]["code_and_dependencies"] = []
    elif case == "raw_empty": sealed.receipt["binding"]["raw_source"] = {}
    elif case == "upstream_missing": sealed.receipt["binding"]["upstream"] = None
    elif case == "producer_task": sealed.receipt["producer"]["task_id"] = "../wrong"
    elif case == "packet_mode": sealed.packet["execution_mode"] = "OTHER"
    elif case == "packet_signature": sealed.packet["run_signature"] = "changed"
    elif case == "result_path": sealed.receipt["producer"]["result_path"] += ".wrong"
    elif case == "ledger_unsealed": sealed.ledger_row["status"] = "RUNNING"
    elif case == "writer_pid": sealed.ledger_row["pid"] = 123
    elif case == "writer_ticks": sealed.ledger_row["proc_start_ticks"] = 123
    elif case == "terminal_identity": sealed.terminal["run_signature"] = "changed"
    elif case == "index_missing": terminal_edit = lambda v: v.update(required_outputs={})
    elif case == "index_session": index_edit = lambda v: v["sessions"][0].update(session_id="002")
    elif case == "index_conflict":
        index_edit = lambda v: v["sessions"][0]["stages"]["motion"].update(receipt={"wrong": True})
    elif case == "index_unbound":
        def index_edit(value):
            value["sessions"][0].update(stage_receipts={}, stages={})
    sealed.rebind(corrupt_digest=case == "digest", index_edit=index_edit,
                  terminal_edit=terminal_edit)
    with pytest.raises(RuntimeError, match=reason):
        sealed.original()
    with pytest.raises(RuntimeError, match=reason):
        sealed.adapted()


def test_changed_payload_bytes_rejected(sealed):
    sealed.actions.write_bytes(b"changed tiny fixture")
    with pytest.raises(RuntimeError):
        sealed.original()
    with pytest.raises(RuntimeError, match="BYTES_OR_SHA_DRIFT"):
        sealed.adapted()


def test_changed_nas_fixture_sha_rejected_even_if_local_copy_is_valid(sealed):
    pin = sealed.receipt["files"]["actions.npz"]
    physical = sealed.repo / "fixture_export/actions.npz"
    physical.parent.mkdir()
    data = bytearray(sealed.actions.read_bytes())
    data[-1] ^= 1
    physical.write_bytes(data)
    reader = FixtureReader(sealed.repo, {str(sealed.actions): (pin, physical)})
    assert sealed.original()[0] == sealed.folder
    with pytest.raises(RuntimeError, match="BYTES_OR_SHA_DRIFT"):
        sealed.adapted(reader)


def test_stage_metadata_symlink_rejected(sealed):
    target = sealed.folder / "fixture_receipt_copy.json"
    target.write_bytes(sealed.stage_path.read_bytes())
    sealed.stage_path.unlink()
    sealed.stage_path.symlink_to(target)
    with pytest.raises(RuntimeError, match="SYMLINK_PATH"):
        sealed.original()
    with pytest.raises(RuntimeError, match="SYMLINK_PATH"):
        sealed.adapted()


def test_reader_cannot_replace_logical_ref_identity(sealed):
    class BadIdentityReader(FixtureReader):
        def verify_ref(self, pin):
            result = super().verify_ref(pin)
            result["path"] += ".rewritten"
            return result
    with pytest.raises(RuntimeError, match="ARTIFACT_REFERENCE_IDENTITY_CHANGED"):
        sealed.adapted(BadIdentityReader(sealed.repo))


@pytest.mark.parametrize("frame_count", [1, 2, 3])
def test_action_numerical_gate_is_unchanged(sealed, frame_count):
    pin = sealed.receipt["files"]["actions.npz"]
    assert validate_session_actions(FixtureReader(sealed.repo), pin, frame_count) == \
        validate_action_arrays(sealed.actions, frame_count)


@pytest.mark.parametrize("invalid", ["nan", "base_last_row", "missing_hand"])
def test_action_invalid_values_are_not_relaxed(sealed, invalid):
    with np.load(sealed.actions, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    if invalid == "nan": arrays["q_arm"][0, 0, 0] = np.nan
    elif invalid == "base_last_row": arrays["T_world_base"][3, 0] = 1
    else: del arrays["q_hand"]
    np.savez(sealed.actions, **arrays)
    pin = _ref(sealed.actions)
    original = validate_action_arrays(sealed.actions, 2)
    assert original
    assert validate_session_actions(FixtureReader(sealed.repo), pin, 2) == original
