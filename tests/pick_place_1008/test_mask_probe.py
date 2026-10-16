"""Mask-probe producer closure tests: synthetic files, no decoding or models."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from chaoyang.pick_place_1008 import audit, mask_probe, source
from chaoyang.pick_place_1008.pipeline import STAGE_FILES
from chaoyang.pick_place_1008.stage_reuse import StageReuseDenied


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def _read(path):
    return json.loads(path.read_text())


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _refresh_terminal(f):
    terminal = _read(f["terminal"])
    terminal["required_outputs"]["candidate/INDEX.json"] = source.ref(f["index"])
    _write(f["terminal"], terminal)
    ledger = _read(f["ledger"])
    ledger["tasks"][0]["result"] = source.ref(f["terminal"])
    _write(f["ledger"], ledger)


def _refresh_receipt_chain(f):
    index = _read(f["index"])
    index["sessions"][0]["stage_receipts"]["source"] = source.ref(f["receipt"])
    _write(f["index"], index)
    _refresh_terminal(f)


@pytest.fixture
def sealed_source(tmp_path, monkeypatch):
    # Only relocate the audit's project boundary; all SHA/producer checks are real.
    repo = tmp_path / "repo"
    monkeypatch.setattr(audit, "PROJECT_ROOT", repo)
    task_id = "exact78_pick_place_mask_probe_source_fixture_v1"
    signature = "a" * 64
    root = repo / "_run/current" / task_id / "attempts/attempt_0001/candidate"
    folder = root / "sessions/001"
    folder.mkdir(parents=True)
    raw, code, config = repo / "raw.bin", repo / "source.py", repo / "camera.json"
    for path in (raw, code, config):
        path.write_bytes((path.name + ": fixed fixture").encode())
    packet = repo / "tasks/current" / task_id / "TASK_PACKET.json"
    _write(packet, {"task_id": task_id, "run_signature": signature,
                    "execution_mode": "EXACT78_CANDIDATE"})
    terminal = root.parent / "RESULT.json"
    binding = {"stage": "source", "session_id": "001", "raw_frame_count": 3,
               "raw_source": {"raw_vst": source.ref(raw)},
               "camera_config": source.ref(config),
               "code_and_dependencies": [source.ref(code)], "upstream": {}}
    files = {}
    for name in STAGE_FILES["source"]:
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((name + ": immutable source fixture").encode())
        files[name] = source.ref(path)
    receipt = folder / "STAGE_source.json"
    _write(receipt, {"schema": "pick_place_1008_sealed_stage_v1", "status": "COMPLETE",
                     "binding": binding, "binding_sha256": _digest(binding), "files": files,
                     "producer": {"task_id": task_id, "task_packet": source.ref(packet),
                                  "run_signature": signature, "result_path": str(terminal)}})
    index = root / "INDEX.json"
    _write(index, {"schema": "pick_place_1008_pipeline_v1", "status": "COMPLETE",
                   "sessions": [{"session_id": "001", "status": "COMPLETE",
                                 "stage_receipts": {"source": source.ref(receipt)}}]})
    _write(terminal, {"task_id": task_id, "run_signature": signature, "status": "PASSED",
                      "validation_errors": [], "task_packet": source.ref(packet),
                      "required_outputs": {"candidate/INDEX.json": source.ref(index)}})
    ledger = repo / "docs/governance/LONG_HORIZON_TASK_STATE.json"
    _write(ledger, {"tasks": [{"task_id": task_id, "status": "PASSED",
                              "run_signature": signature, "pid": None,
                              "proc_start_ticks": None, "result": source.ref(terminal)}]})
    return {"repo": repo, "root": root, "folder": folder, "index": index,
            "receipt": receipt, "terminal": terminal, "ledger": ledger,
            "packet": packet, "raw": raw, "code": code, "config": config}


def _consume(f, sid="001", receipt_pin=None):
    value, pin = mask_probe._sealed_index(f["repo"], f["index"])
    selected = receipt_pin or value["sessions"][0]["stage_receipts"]["source"]
    folder = mask_probe._sealed_source(f["repo"], f["root"], sid, selected)
    return value, pin, folder


def test_passed_source_has_terminal_ledger_and_receipt_closure(sealed_source):
    f = sealed_source
    value, pin, folder = _consume(f)
    assert value["status"] == "COMPLETE"
    assert pin == source.ref(f["index"])
    assert folder == f["folder"] / "source"


@pytest.mark.parametrize("mutation", ["absent", "duplicate", "running", "failed", "pid", "start_ticks"])
def test_index_rejects_missing_unsealed_or_live_producer(sealed_source, mutation):
    f = sealed_source
    ledger = _read(f["ledger"])
    row = ledger["tasks"][0]
    if mutation == "absent": ledger["tasks"] = []
    elif mutation == "duplicate": ledger["tasks"].append(copy.deepcopy(row))
    elif mutation == "running": row["status"] = "RUNNING"
    elif mutation == "failed": row["status"] = "FAILED_RUNTIME_FINAL"
    elif mutation == "pid": row["pid"] = 123
    else: row["proc_start_ticks"] = 456
    _write(f["ledger"], ledger)
    with pytest.raises(ValueError, match="MASK_PROBE_PRODUCER_NOT_SEALED"):
        _consume(f)


def test_outer_json_cannot_be_forged_without_ledger_terminal_pin(sealed_source):
    f = sealed_source
    terminal = _read(f["terminal"])
    terminal["forged_after_seal"] = True
    _write(f["terminal"], terminal)
    with pytest.raises(audit.AuditError, match="EVIDENCE_REFERENCE_DRIFT"):
        _consume(f)


def test_ledger_pin_cannot_redirect_to_another_terminal(sealed_source):
    f = sealed_source
    decoy = f["repo"] / "copied_RESULT.json"
    decoy.write_bytes(f["terminal"].read_bytes())
    ledger = _read(f["ledger"])
    ledger["tasks"][0]["result"] = source.ref(decoy)
    _write(f["ledger"], ledger)
    with pytest.raises(ValueError, match="MASK_PROBE_TERMINAL_LEDGER_PIN"):
        _consume(f)


@pytest.mark.parametrize("component", ["status", "validation_errors", "signature", "index_pin"])
def test_fresh_terminal_pin_does_not_authorize_wrong_index_identity(sealed_source, component):
    f = sealed_source
    terminal = _read(f["terminal"])
    if component == "status": terminal["status"] = "FAILED_RUNTIME_FINAL"
    elif component == "validation_errors": terminal["validation_errors"] = ["FAILED_CHECK"]
    elif component == "signature": terminal["run_signature"] = "b" * 64
    else: terminal["required_outputs"]["candidate/INDEX.json"] = source.ref(f["raw"])
    _write(f["terminal"], terminal)
    ledger = _read(f["ledger"])
    ledger["tasks"][0]["result"] = source.ref(f["terminal"])
    _write(f["ledger"], ledger)
    with pytest.raises(ValueError, match="MASK_PROBE_INDEX_NOT_SEALED"):
        _consume(f)


def test_copied_index_cannot_claim_original_producer(sealed_source):
    f = sealed_source
    alternate = f["repo"] / "_run/current/exact78_pick_place_decoy_v1/attempts/attempt_0001/candidate"
    alternate.mkdir(parents=True)
    (alternate / "INDEX.json").write_bytes(f["index"].read_bytes())
    (alternate.parent / "RESULT.json").write_bytes(f["terminal"].read_bytes())
    with pytest.raises(ValueError, match="MASK_PROBE_INDEX_OWNERSHIP"):
        mask_probe._sealed_index(f["repo"], alternate / "INDEX.json")


def test_partial_source_index_is_not_consumable_even_with_updated_pins(sealed_source):
    f = sealed_source
    index = _read(f["index"])
    index["status"] = "PARTIAL"
    _write(f["index"], index)
    _refresh_terminal(f)
    with pytest.raises(ValueError, match="MASK_PROBE_SOURCE_INDEX_INCOMPLETE"):
        _consume(f)


def test_wrong_session_cannot_consume_valid_source_receipt(sealed_source):
    f = sealed_source
    with pytest.raises(ValueError, match="MASK_PROBE_CANONICAL_SOURCE_NOT_SEALED"):
        _consume(f, sid="008")
    # Even a copied receipt placed at the requested session fails its identity.
    other = f["root"] / "sessions/008/STAGE_source.json"
    other.parent.mkdir(parents=True)
    other.write_bytes(f["receipt"].read_bytes())
    with pytest.raises(StageReuseDenied, match="STAGE_SESSION_OR_FRAME_IDENTITY_MISMATCH"):
        mask_probe._sealed_source(f["repo"], f["root"], "008", source.ref(other))


def test_source_binding_tamper_rejected_after_repinning_outer_chain(sealed_source):
    f = sealed_source
    receipt = _read(f["receipt"])
    receipt["binding"]["raw_frame_count"] = 4
    _write(f["receipt"], receipt)
    _refresh_receipt_chain(f)
    with pytest.raises(StageReuseDenied, match="BINDING_DIGEST_TAMPERED"):
        _consume(f)


@pytest.mark.parametrize("target", ["raw", "code", "config", "packet", "source/source.npz"])
def test_source_receipt_verifies_live_raw_code_config_packet_and_outputs(sealed_source, target):
    f = sealed_source
    path = f[target] if target in f else f["folder"] / target
    path.write_bytes(b"drift after terminal sealing")
    with pytest.raises((audit.AuditError, RuntimeError), match="DRIFT|MISSING|INVALID"):
        _consume(f)


def test_source_index_must_pin_exact_current_stage_receipt(sealed_source):
    f = sealed_source
    index = _read(f["index"])
    original_pin = index["sessions"][0]["stage_receipts"]["source"]
    index["sessions"][0]["stage_receipts"]["source"] = source.ref(f["raw"])
    _write(f["index"], index)
    _refresh_terminal(f)
    with pytest.raises(StageReuseDenied, match="PRODUCER_INDEX_DOES_NOT_BIND_STAGE"):
        _consume(f, receipt_pin=original_pin)


def test_conflicting_direct_and_nested_source_refs_are_rejected(sealed_source):
    f = sealed_source
    index = _read(f["index"])
    index["sessions"][0]["stages"] = {"source": {"receipt": source.ref(f["raw"])}}
    _write(f["index"], index)
    _refresh_terminal(f)
    with pytest.raises(StageReuseDenied, match="PRODUCER_INDEX_STAGE_REFERENCE_CONFLICT"):
        _consume(f)


def test_exact_ref_from_foreign_producer_folder_is_rejected(sealed_source):
    f = sealed_source
    decoy_root = f["repo"] / "_run/current/exact78_pick_place_decoy_v1/attempts/attempt_0001/candidate"
    decoy = decoy_root / "sessions/001/STAGE_source.json"
    decoy.parent.mkdir(parents=True)
    decoy.write_bytes(f["receipt"].read_bytes())
    # File pins still bind original canonical outputs; folder substitution fails.
    with pytest.raises(StageReuseDenied, match="ARTIFACT_PATH_MISMATCH|STAGE_NOT_OWNED_BY_PRODUCER"):
        mask_probe._sealed_source(f["repo"], decoy_root, "001", source.ref(decoy))


def test_arbitrary_packet_path_cannot_pass_combined_source_proof(sealed_source):
    f = sealed_source
    packet_copy = f["repo"] / "copied_TASK_PACKET.json"
    packet_copy.write_bytes(f["packet"].read_bytes())
    receipt = _read(f["receipt"])
    receipt["producer"]["task_packet"] = source.ref(packet_copy)
    _write(f["receipt"], receipt)
    terminal = _read(f["terminal"])
    terminal["task_packet"] = source.ref(packet_copy)
    _write(f["terminal"], terminal)
    _refresh_receipt_chain(f)
    with pytest.raises(ValueError, match="MASK_PROBE_PACKET_OWNERSHIP"):
        _consume(f)
