"""Known-answer proof-chain tests; no model, robot or real dataset access."""
import copy
import hashlib
import json
from pathlib import Path
import pytest
from chaoyang.pick_place_1008.stage_reuse import find_reuse, StageReuseDenied
from chaoyang.exact78_baseline.stage_reuse import ref

REQUIRED = ("actions.npz", "motion.json", "frame_mapping.jsonl")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def refresh_terminal(f):
    terminal = json.loads(f["result"].read_text())
    terminal["required_outputs"]["candidate/INDEX.json"] = ref(f["index"])
    write(f["result"], terminal)
    ledger = json.loads(f["ledger"].read_text())
    ledger["tasks"][0]["result"] = ref(f["result"])
    write(f["ledger"], ledger)


def fixture(tmp_path, terminal_status="PASSED", nested=False):
    repo = tmp_path / "repo"
    task_id = "exact78_pick_place_fixture_v1"
    root = repo / "_run/current" / task_id / "attempts/attempt_0001/candidate"
    folder = root / "sessions/001"
    folder.mkdir(parents=True)
    source, code, config = repo/"source.bin", repo/"code.py", repo/"config.json"
    for path in [source, code, config]: path.write_text(path.name)
    binding = {"stage": "motion", "session_id": "001", "raw_frame_count": 3,
               "raw_source": {"raw": ref(source)}, "camera_config": ref(config),
               "code_and_dependencies": [ref(code)], "upstream": {}}
    packet = repo/"tasks/current"/task_id/"TASK_PACKET.json"
    write(packet, {"task_id": task_id, "run_signature": "a"*64, "execution_mode": "EXACT78_CANDIDATE"})
    result = root.parent/"RESULT.json"
    producer = {"task_id": task_id, "task_packet": ref(packet), "run_signature": "a"*64,
                "result_path": str(result)}
    files = {}
    for name in REQUIRED:
        path = folder/name
        path.write_bytes((name+":complete fixture").encode())
        files[name] = ref(path)
    receipt = folder/"STAGE_motion.json"
    write(receipt, {"schema": "pick_place_1008_sealed_stage_v1", "status": "COMPLETE",
                    "binding": binding, "binding_sha256": digest(binding), "producer": producer, "files": files})
    row = {"session_id": "001", "status": "COMPLETE" if terminal_status == "PASSED" else "FAILED"}
    if nested: row["stages"] = {"motion": {"receipt": ref(receipt)}}
    else: row["stage_receipts"] = {"motion": ref(receipt)}
    index = root/"INDEX.json"
    write(index, {"schema": "pick_place_1008_pipeline_v1", "sessions": [row]})
    write(result, {"task_id": task_id, "run_signature": "a"*64, "status": terminal_status,
                   "task_packet": ref(packet), "required_outputs": {"candidate/INDEX.json": ref(index)}})
    ledger = repo/"docs/governance/LONG_HORIZON_TASK_STATE.json"
    write(ledger, {"tasks": [{"task_id": task_id, "status": terminal_status,
                              "pid": None, "proc_start_ticks": None, "result": ref(result)}]})
    return {"repo": repo, "root": root, "folder": folder, "binding": binding,
            "packet": packet, "result": result, "index": index, "ledger": ledger,
            "receipt": receipt, "source": source, "code": code, "config": config}


def lookup(f, binding=None):
    return find_reuse(f["repo"], [f["root"]], "001", "motion",
                      f["binding"] if binding is None else binding, REQUIRED)


@pytest.mark.parametrize("status,nested", [("PASSED", False), ("PASSED", True), ("FAILED_RUNTIME_FINAL", True)])
def test_complete_stage_has_exact_terminal_index_chain(tmp_path, status, nested):
    f = fixture(tmp_path, status, nested)
    folder, pin = lookup(f)
    assert folder == f["folder"] and pin == ref(f["receipt"])


def test_intact_different_binding_is_only_cache_miss(tmp_path):
    f = fixture(tmp_path)
    other = copy.deepcopy(f["binding"])
    other["raw_frame_count"] = 4
    assert lookup(f, other) == (None, None)


def test_absent_stage_is_normal_miss(tmp_path):
    f = fixture(tmp_path)
    f["receipt"].unlink()
    assert lookup(f) == (None, None)


def test_binding_tamper_and_partial_receipt_are_rejected(tmp_path):
    f = fixture(tmp_path)
    r = json.loads(f["receipt"].read_text())
    r["binding"]["raw_frame_count"] = 4
    write(f["receipt"], r)
    with pytest.raises(StageReuseDenied, match="BINDING_DIGEST_TAMPERED"): lookup(f)
    r["binding"] = f["binding"]
    r["status"] = "PARTIAL"
    write(f["receipt"], r)
    with pytest.raises(StageReuseDenied, match="INCOMPLETE_STAGE"): lookup(f)


@pytest.mark.parametrize("extra", [False, True])
def test_mandatory_output_set_is_exact(tmp_path, extra):
    f = fixture(tmp_path)
    r = json.loads(f["receipt"].read_text())
    if extra: r["files"]["extra.npz"] = r["files"]["actions.npz"]
    else: del r["files"]["motion.json"]
    write(f["receipt"], r)
    with pytest.raises(StageReuseDenied, match="MANDATORY_OUTPUT_SET"): lookup(f)


def test_output_sha_drift_and_path_escape_are_rejected(tmp_path):
    f = fixture(tmp_path)
    (f["folder"]/"actions.npz").write_bytes(b"changed after seal")
    with pytest.raises(StageReuseDenied, match="SHA_OR_SIZE_DRIFT"): lookup(f)
    r = json.loads(f["receipt"].read_text())
    r["files"]["actions.npz"] = ref(f["source"])
    write(f["receipt"], r)
    with pytest.raises(StageReuseDenied, match="PATH_MISMATCH"): lookup(f)


def test_output_symlink_is_rejected_even_with_matching_target_sha(tmp_path):
    f = fixture(tmp_path)
    path = f["folder"]/"actions.npz"
    data = path.read_bytes()
    path.unlink()
    target = f["folder"]/"real_actions.npz"
    target.write_bytes(data)
    path.symlink_to(target)
    with pytest.raises(StageReuseDenied, match="SYMLINK_PATH|PATH_MISMATCH"): lookup(f)


@pytest.mark.parametrize("status", ["RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE", "BLOCKED_RESOURCE"])
def test_unsealed_or_resource_wait_producer_cannot_be_consumed(tmp_path, status):
    f = fixture(tmp_path)
    ledger = json.loads(f["ledger"].read_text())
    ledger["tasks"][0]["status"] = status
    write(f["ledger"], ledger)
    with pytest.raises(StageReuseDenied, match="NOT_SEALED_TERMINAL"): lookup(f)


def test_terminal_writer_or_missing_terminal_is_rejected(tmp_path):
    f = fixture(tmp_path)
    ledger = json.loads(f["ledger"].read_text())
    ledger["tasks"][0]["pid"] = 1234
    write(f["ledger"], ledger)
    with pytest.raises(StageReuseDenied, match="WRITER_PRESENT"): lookup(f)
    ledger["tasks"][0]["pid"] = None
    write(f["ledger"], ledger)
    f["result"].unlink()
    with pytest.raises(StageReuseDenied, match="UNSEALED_RECEIPT|MISSING_OR_INVALID"): lookup(f)


@pytest.mark.parametrize("kind", ["missing_index_ref", "index_sha_drift", "unbound_stage", "changed_stage_pin", "wrong_session"])
def test_terminal_must_pin_index_and_exact_session_stage(tmp_path, kind):
    f = fixture(tmp_path)
    terminal = json.loads(f["result"].read_text())
    index = json.loads(f["index"].read_text())
    if kind == "missing_index_ref":
        del terminal["required_outputs"]["candidate/INDEX.json"]
        write(f["result"], terminal)
        ledger = json.loads(f["ledger"].read_text())
        ledger["tasks"][0]["result"] = ref(f["result"])
        write(f["ledger"], ledger)
    elif kind == "index_sha_drift":
        index["not_originally_pinned"] = True
        write(f["index"], index)
    else:
        if kind == "unbound_stage": index["sessions"][0]["stage_receipts"] = {}
        if kind == "changed_stage_pin": index["sessions"][0]["stage_receipts"]["motion"] = ref(f["source"])
        if kind == "wrong_session": index["sessions"][0]["session_id"] = "002"
        write(f["index"], index)
        refresh_terminal(f)
    with pytest.raises(StageReuseDenied): lookup(f)


@pytest.mark.parametrize("component", ["source", "code", "config", "packet", "result"])
def test_all_signed_inputs_and_producer_files_are_verified(tmp_path, component):
    f = fixture(tmp_path)
    f[component].write_bytes(b"sha drift")
    with pytest.raises(StageReuseDenied): lookup(f)


def test_producer_terminal_signature_mismatch_rejected_after_fresh_pin(tmp_path):
    f = fixture(tmp_path)
    result = json.loads(f["result"].read_text())
    result["run_signature"] = "b"*64
    write(f["result"], result)
    ledger = json.loads(f["ledger"].read_text())
    ledger["tasks"][0]["result"] = ref(f["result"])
    write(f["ledger"], ledger)
    with pytest.raises(StageReuseDenied, match="TERMINAL_IDENTITY_OR_SIGNATURE"): lookup(f)


def test_conflicting_index_receipt_routes_are_rejected(tmp_path):
    f = fixture(tmp_path)
    index = json.loads(f["index"].read_text())
    index["sessions"][0]["stages"] = {"motion": {"receipt": ref(f["source"])}}
    write(f["index"], index)
    refresh_terminal(f)
    with pytest.raises(StageReuseDenied, match="REFERENCE_CONFLICT"): lookup(f)


def test_single_bad_receipt_can_be_isolated_by_session_caller(tmp_path):
    bad = fixture(tmp_path/"bad")
    good = fixture(tmp_path/"good")
    (bad["folder"]/"actions.npz").write_bytes(b"broken")
    rows = []
    for candidate in (bad, good):
        try:
            folder, _ = lookup(candidate)
            rows.append("COMPLETE" if folder else "MISS")
        except StageReuseDenied:
            rows.append("REUSE_REJECTED")
    assert rows == ["REUSE_REJECTED", "COMPLETE"]
