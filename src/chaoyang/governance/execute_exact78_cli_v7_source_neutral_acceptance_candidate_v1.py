"""Finite, CPU-only code acceptance for the v7 check-only source planner.

No model or stage executor is imported. Registration is available only after
the exact source/code probes pass in an empty governance window.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
from typing import Any

from chaoyang.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH, artifact_ref,
    atomic_json, canonical_bytes, load_json, now_iso, process_identity,
    publish_bundle, validate_artifact_ref, validate_task_state,
)
from chaoyang.governance.register_single_task_packet import _validate_packet


TASK_ID = "exact78_generic_submit_cli_v7_code_acceptance_v1"
PHASE = "EXACT78_GENERIC_SUBMIT_CLI_V7_CODE_ACCEPTANCE"
RESULT_SCHEMA = "exact78-generic-submit-cli-v7-acceptance-result-v1"
RECEIPT_SCHEMA = "exact78-generic-submit-cli-v7-acceptance-receipt-v1"
SCRIPT = Path(__file__).resolve()
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
ATTEMPT = REPO_ROOT / "_run/current" / TASK_ID / "attempts/attempt_0001"
PACKET = REPO_ROOT / "tasks/current" / TASK_ID / "TASK_PACKET.json"
RECEIPT = REPO_ROOT / "tasks/receipts" / f"{TASK_ID.upper()}.json"
ACTIVE = frozenset({"CLAIMED", "RUNNING", "WAIT_GPU_RESOURCE"})
CODE = (
    "src/chaoyang/cli_v7/__init__.py",
    "src/chaoyang/cli_v7/__main__.py",
    "src/chaoyang/cli_v7/source_neutral_plan_v1.py",
    "src/chaoyang/governance/exact78_cold_start_guard_v1.py",
    "src/chaoyang/cli/exact78_first_submit_bridge_candidate.py",
    "src/chaoyang/governance/execute_exact78_first_submit_cpu_candidate_v1.py",
)
CODE_SHA = {
    CODE[0]: "1ce512eb2f078f9b7da633516c3e0e2cf5d9d1c564aa1c3c85b6afbf8e057f51",
    CODE[1]: "fc18bfbe5a487b2d969698ce45f351e86dd77666877f1018a23b53df3a15e2d3",
    CODE[2]: "28027c07b20593130ff63446f4ed688571d7642b7e7b1e3ea650403ae741c548",
    CODE[3]: "4a7dd114c6da336e71ab29ad925140ae80cdc57c601c57c3821ae2fe5b7786bf",
    CODE[4]: "96cd8666c790acb48e6eb92cf7b796039533d82ad7451f729a27361d9e7daa43",
    CODE[5]: "8d60090a7473c5c0c68df84becc8bcc17f9b6aafeb044ab1e9bdf660834fb455",
}
TESTS = (
    "tests/unit/test_exact78_cli_v7_source_neutral_plan_v1.py",
    "tests/unit/test_exact78_cli_v7_source_neutral_acceptance_candidate_v1.py",
)
EXAMPLES = {
    "tracker": {
        "root": "/path/to/private_data/egodata/datasets/ego/chips_cards_tracker_0902/playing_cards/play_cards_0902_002",
        "source_kind": "TRACKER_CLIP_RGB",
        "signature": "a0ecd906ecec339a723eb974730a779774bea5f020029af8d59d13f1e30e80b8",
        "roles": {
            "video": "2c42380141cceafb4a6f4105b01fb0c52ad6e123e7c4175c85d846b7b869e132",
            "source_stereo_video": "8bcbad123f20cdce5106df64a9141e3b5a65d00d1c7910827a75dbe323603c90",
            "camera_params": "f44e237577f640e6e0c1262dcf384f524c25d38e91e2b52c4a17239d9b40f2bf",
            "slam": "3f6d0ad6cca753761b8f604e5a1fcf1ee2f35dd3b5ffbccc93c9d4e5a1f34b97",
            "clip_manifest": "d8de7ab150cff75cab6302a7b92c30943da1c7b487026c434c69e1c9c1710b20",
        },
        "physical_eye": "tracker_session_left_RGB",
        "time_mode": "TRACKER_CLIP_METADATA_ONLY",
    },
    "vst": {
        "root": "/path/to/private_data/egodata/datasets/ego/chips_cards_hands_0915/cards_120_0915/004",
        "source_kind": "0915_RAW_VST",
        "signature": "57d10591aa3a2ee9b71091f1db0b19e3eb04795497f88f0e33469684edd62ef5",
        "roles": {
            "raw/vst.h264": "efe3c5b9a9656331c37c05015d39df4df842a41d2a91506f5c851af8380a8181",
            "raw/vst.qpc.ts.jsonl": "fe95ec716fd6636fda6dc054d18c1d5e49c9437300b79268d36be7e4622d2d6c",
            "raw/vst.ts.jsonl": "b5e983a6ff69b05cbd38a3fb0abb18a6e9dfa5cd42039e59530af1d0823bb33b",
            "raw/camera_params.json": "19c8e5dbe471c91e84214a047882c1abf3087b865c863d8a398c497236be3352",
            "raw/camera_params.meta.json": "9c1a6c0469ae0ebe43031bd5bb7d477fe1a7d4b4f2a83c719d401091b31d613a",
        },
        "physical_eye": "left_sourceIndex_1",
        "time_mode": "TIMEBASE_UNVERIFIED",
    },
}


def _snapshot() -> tuple[int, dict[str, Any], dict[str, Any], bool]:
    receipt, state, index = load_json(RECEIPT_PATH), load_json(TASK_STATE_PATH), load_json(INDEX)
    for key, path in (("task_state", TASK_STATE_PATH), ("task_packet_index_current", INDEX)):
        ref = receipt.get("files", {}).get(key)
        if not isinstance(ref, dict) or ref.get("path") != str(path) or validate_artifact_ref(ref):
            raise RuntimeError("V7_ACCEPTANCE_GOVERNANCE_REF_DRIFT:" + key)
    if (any(row.get("task_id") == TASK_ID for row in state.get("tasks", []))
            or any(row.get("task_id") == TASK_ID for row in index.get("task_packets", []))
            or any(path.exists() for path in (ATTEMPT, PACKET.parent, PACKET, RECEIPT))):
        raise RuntimeError("V7_ACCEPTANCE_TASK_ALREADY_USED")
    active_tasks = [row for row in state.get("tasks", []) if row.get("status") in ACTIVE]
    active_packets = [row for row in index.get("task_packets", [])
                      if row.get("execution_allowed") is True]
    if len(active_tasks) > 1 or len(active_packets) > 1:
        raise RuntimeError("V7_ACCEPTANCE_DOUBLE_WRITER")
    if (bool(active_tasks) != bool(active_packets)
            or bool(state.get("next_task")) != bool(active_tasks)
            or (active_tasks and active_tasks[0].get("task_id") != active_packets[0].get("task_id"))):
        raise RuntimeError("V7_ACCEPTANCE_ACTIVE_LEDGER_INDEX_DRIFT")
    active = bool(active_tasks)
    return int(receipt["governance_revision"]), state, index, active


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(PYTHONPATH=str(REPO_ROOT / "src"), PYTHONDONTWRITEBYTECODE="1",
               PYTHONNOUSERSITE="1", TMPDIR=str(REPO_ROOT / "_run"),
               XDG_CACHE_HOME=str(REPO_ROOT / "_run/caches"))
    return env


def _probe_tests() -> dict[str, Any]:
    command = ["/usr/local/bin/python", "-B", "-m", "unittest", "-q",
               "tests.unit.test_exact78_cli_v7_source_neutral_plan_v1",
               "tests.unit.test_exact78_cli_v7_source_neutral_acceptance_candidate_v1"]
    result = subprocess.run(command, cwd=REPO_ROOT, env=_env(),
                            capture_output=True, timeout=120, check=False)
    output = result.stdout + result.stderr
    count = re.search(rb"Ran (\d+) tests?", output)
    if result.returncode or count is None or int(count.group(1)) < 10 or b"OK" not in output:
        raise RuntimeError("V7_ACCEPTANCE_NEGATIVE_TESTS_NOT_PASS:" +
                           output.decode("utf-8", "replace")[-400:])
    normalized = re.sub(rb"(Ran \d+ tests? in )\d+(?:\.\d+)?s",
                        rb"\g<1><elapsed>s", output)
    return {"status": "PASSED", "test_count": int(count.group(1)),
            "output_sha256": hashlib.sha256(normalized).hexdigest()}


def _probe_example(name: str) -> dict[str, Any]:
    expected = EXAMPLES[name]
    command = ["/usr/local/bin/python", "-B", "-m", "chaoyang.cli_v7", "submit",
               "exact78-raw-to-robot", "--session", expected["root"],
               "--expected-input-sha256", expected["signature"], "--check-only"]
    result = subprocess.run(command, cwd=REPO_ROOT, env=_env(),
                            capture_output=True, timeout=180, check=False)
    try:
        value = json.loads(result.stdout)
    except (UnicodeDecodeError, ValueError) as exc:
        raise RuntimeError("V7_ACCEPTANCE_CLI_NOT_JSON:" + name) from exc
    source = value.get("source") if isinstance(value, dict) else None
    if (result.returncode != 0 or not isinstance(source, dict)
            or source.get("source_kind") != expected["source_kind"]
            or source.get("session_root") != expected["root"]
            or value.get("submission_input_sha256") != expected["signature"]
            or source.get("physical_eye") != expected["physical_eye"]
            or source.get("time_mode") != expected["time_mode"]
            or set(source.get("source_files", {})) != set(expected["roles"])
            or any(source["source_files"][role].get("sha256") != digest
                   for role, digest in expected["roles"].items())
            or value.get("registration_started") is not False
            or value.get("model_calls") != 0
            or value.get("decode_verified") is not False
            or value.get("control_ground_truth") is not False):
        raise RuntimeError("V7_ACCEPTANCE_SOURCE_OR_CLI_DRIFT:" + name)
    return {"source_kind": expected["source_kind"],
            "source_root": expected["root"], "source_files": source["source_files"],
            "submission_input_sha256": expected["signature"],
            "physical_eye": expected["physical_eye"], "time_mode": expected["time_mode"],
            "status": value["status"],
            "stdout_sha256": hashlib.sha256(result.stdout).hexdigest()}


def inspect() -> tuple[int, dict[str, Any], dict[str, Any], dict[str, Any], bool]:
    revision, state, index, active = _snapshot()
    code = {name: artifact_ref(REPO_ROOT / name) for name in CODE}
    if any(code[name]["sha256"] != CODE_SHA[name] for name in CODE):
        raise RuntimeError("V7_ACCEPTANCE_PINNED_CODE_SHA_DRIFT")
    tests = {name: artifact_ref(REPO_ROOT / name) for name in TESTS}
    examples = {name: _probe_example(name) for name in EXAMPLES}
    if any(row["status"] != ("WAIT_ACTIVE_WRITER" if active else
                             "READY_FOR_FUTURE_CODE_ACCEPTANCE_ONLY")
           for row in examples.values()):
        raise RuntimeError("V7_ACCEPTANCE_WRITER_STATUS_DRIFT")
    negatives = _probe_tests()
    if int(load_json(RECEIPT_PATH)["governance_revision"]) != revision:
        raise RuntimeError("V7_ACCEPTANCE_REVISION_CHANGED_DURING_PROBE")
    inputs = {"code": code, "tests": tests, "examples": examples,
              "negative_tests": negatives,
              "source_scope": "0901_0902_TRACKER_PLUS_0915_VST_CHECK_ONLY"}
    for ref in [*code.values(), *tests.values(),
                *(source for row in examples.values()
                  for source in row["source_files"].values())]:
        if validate_artifact_ref(ref):
            raise RuntimeError("V7_ACCEPTANCE_CODE_OR_SOURCE_SHA_DRIFT")
    return revision, state, index, inputs, active


def packet_value(inputs: dict[str, Any]) -> dict[str, Any]:
    now = datetime.now().astimezone()
    writer = {"pid": os.getpid(), "proc_start_ticks": process_identity(os.getpid())["start_ticks"],
              "executor_epoch": 1, "fencing_token": secrets.token_hex(24), "scope": TASK_ID}
    runtime = [artifact_ref(path) for path in
               (SCRIPT, REPO_ROOT / "src/chaoyang/governance/common.py",
                REPO_ROOT / "src/chaoyang/governance/register_single_task_packet.py")]
    command = ("PYTHONPATH=src /usr/local/bin/python -B -m "
               "chaoyang.governance.execute_exact78_cli_v7_source_neutral_acceptance_candidate_v1 "
               "--register-and-run")
    config = {"cpu_threads_total": 2, "gpu_count": 0, "model_calls": 0,
              "render_calls": 0, "training_steps": 0,
              "session_attempts_max": 1, "wall_seconds_max": 600,
              "calibration_or_absent": "NOT_CONSUMED", "weights": "NOT_CONSUMED"}
    signature = hashlib.sha256(canonical_bytes({"inputs": inputs, "runtime_code": runtime,
        "config": config, "command": command, "schema": RESULT_SCHEMA})).hexdigest()
    packet = {
        "schema_version": "chaoyang-r22-task-packet-v1", "task_id": TASK_ID,
        "objective": "Accept v7 source-neutral check-only planning code for two raw domains",
        "phase": PHASE, "plan_revision": "EXACT78_RAW_TO_ROBOT_V1_20260930",
        "execution_revision": "EXACT78_GENERIC_SUBMIT_CLI_V7_ACCEPTANCE_V1",
        "execution_mode": "EXACT78_GOVERNANCE_ONLY", "registered_at": now.isoformat(),
        "deadline_at": (now + timedelta(minutes=10)).isoformat(),
        "read_set": ["AGENTS.md", "docs/current/EXACT78_EXECUTION_ROUTE_V1.json",
                     "docs/plans/EXACT78_RAW_TO_ROBOT_V1_20260930/63_GENERIC_SUBMIT_CHECK_ONLY_CODE_PREP_ZH.md",
                     CODE[0], CODE[2], CODE[3], *TESTS],
        "write_set": [str(ATTEMPT.relative_to(REPO_ROOT)),
                      str(PACKET.relative_to(REPO_ROOT)),
                      str(RECEIPT.relative_to(REPO_ROOT)),
                      str(INDEX.relative_to(REPO_ROOT)), "docs/governance"],
        "prerequisites": ["GOVERNANCE_PASS", "NO_ACTIVE_TASK", "SOURCE_SHA_PINNED",
                          "CLI_V7_CHECK_ONLY_READY", "CPU_ONLY", "0916_READ_ONLY",
                          "NO_MODEL_OR_ROBOT"],
        "required_outputs": ["RESULT.json", "terminal_receipt"],
        "budgets": {"workers_max": 1, **config},
        "stop_conditions": ["CODE_OR_SOURCE_SHA_DRIFT", "DOUBLE_WRITER", "DEADLINE",
                            "CHECK_ONLY_NOT_READY", "ONE_ATTEMPT_TERMINAL"],
        "claim_limit": "v7 planner code acceptance only; no dynamic stage packets, model, Robot, visual, product, control or training qualification",
        "writer": writer, "inputs": copy.deepcopy(inputs), "runtime_code": runtime,
        "config": config, "run_signature": signature, "commands": [command],
    }
    if errors := _validate_packet(packet):
        raise RuntimeError("V7_ACCEPTANCE_PACKET_INVALID:" + ";".join(errors))
    return packet


def _virtual_ref(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    raw = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"
    return {"path": str(path), "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest()}


def _index_base(index: dict[str, Any]) -> dict[str, Any]:
    return {key: copy.deepcopy(value) for key, value in index.items()
            if key not in {"governance_revision", "generation_id", "generated_at",
                           "supersedes_index"}}


def register(packet: dict[str, Any], revision: int) -> None:
    current, _state, _index, active = _snapshot()
    if active or current != revision:
        raise RuntimeError("V7_ACCEPTANCE_ACTIVE_WRITER_OR_REVISION_CONFLICT")
    if any(row["status"] != "READY_FOR_FUTURE_CODE_ACCEPTANCE_ONLY"
           for row in packet["inputs"]["examples"].values()):
        raise RuntimeError("V7_ACCEPTANCE_CHECK_ONLY_NOT_READY")
    if (packet["writer"]["pid"] != os.getpid()
            or packet["writer"]["proc_start_ticks"] != process_identity(os.getpid())["start_ticks"]):
        raise RuntimeError("V7_ACCEPTANCE_WRITER_IDENTITY_DRIFT")
    refs = [*packet["runtime_code"], *packet["inputs"]["code"].values(),
            *packet["inputs"]["tests"].values(),
            *(ref for row in packet["inputs"]["examples"].values()
              for ref in row["source_files"].values())]
    if any(validate_artifact_ref(ref) for ref in refs):
        raise RuntimeError("V7_ACCEPTANCE_INPUT_SHA_DRIFT")
    packet_ref = _virtual_ref(PACKET, packet)
    state, index = copy.deepcopy(load_json(TASK_STATE_PATH)), _index_base(load_json(INDEX))
    stamp = now_iso()
    state["tasks"].append({"task_id": TASK_ID, "phase": PHASE, "attempt": 1,
        "status": "RUNNING", "session": "tracker_0902_and_vst_0915_check_only",
        "updated_at": stamp, "heartbeat_at": stamp,
        "pid": packet["writer"]["pid"],
        "proc_start_ticks": packet["writer"]["proc_start_ticks"],
        "gpu_id": None, "task_packet": packet_ref,
        "run_signature": packet["run_signature"], "quality": "PENDING",
        "adoption": "NOT_ADOPTED", "review": "MACHINE_AUDIT"})
    state["next_task"] = {"task_id": TASK_ID,
        "session": "tracker_0902_and_vst_0915_check_only",
        "prerequisites": packet["prerequisites"],
        "expected_resource": "CPU_TWO_THREADS_NO_GPU",
        "stop_condition": "one immutable code acceptance or ten-minute deadline"}
    if errors := validate_task_state(state):
        raise RuntimeError("V7_ACCEPTANCE_STATE_INVALID:" + ";".join(errors))
    index["task_packets"].append({"task_id": TASK_ID,
        "packet_path": str(PACKET.relative_to(REPO_ROOT)),
        "packet_sha256": packet_ref["sha256"],
        "execution_class": "CURRENT_LEDGER_ROUTABLE", "execution_allowed": True})
    index.update(packet_revision="EXACT78_GENERIC_SUBMIT_CLI_V7_ACCEPTANCE_V1_RUNNING",
                 plan_revision=packet["plan_revision"],
                 execution_revision=packet["execution_revision"],
                 status="PASS", supersedes_index=artifact_ref(INDEX))
    ATTEMPT.mkdir(parents=True, exist_ok=False)
    atomic_json(ATTEMPT / "WRITER.json", packet["writer"])
    PACKET.parent.mkdir(parents=True, exist_ok=False)
    atomic_json(PACKET, packet)
    if artifact_ref(PACKET) != packet_ref:
        raise RuntimeError("V7_ACCEPTANCE_PACKET_VIRTUAL_SHA_DRIFT")
    publish_bundle(load_json(AUTHORITY_PATH), state,
                   event_type="EXACT78_GENERIC_SUBMIT_CLI_V7_ACCEPTANCE_REGISTERED",
                   expected_revision=revision, generator_path=SCRIPT,
                   task_packet_index_path=INDEX, task_packet_index_value=index)


def _fence(packet: dict[str, Any]) -> None:
    writer = packet["writer"]
    if (writer["pid"] != os.getpid()
            or writer["proc_start_ticks"] != process_identity(os.getpid())["start_ticks"]
            or load_json(ATTEMPT / "WRITER.json") != writer
            or load_json(PACKET) != packet):
        raise RuntimeError("V7_ACCEPTANCE_WRITER_FENCE_DRIFT")
    state, index = load_json(TASK_STATE_PATH), load_json(INDEX)
    rows = [row for row in state["tasks"] if row.get("task_id") == TASK_ID]
    live = [row for row in index["task_packets"] if row.get("execution_allowed") is True]
    if (len(rows) != 1 or rows[0].get("status") != "RUNNING"
            or rows[0].get("run_signature") != packet["run_signature"]
            or (state.get("next_task") or {}).get("task_id") != TASK_ID
            or len(live) != 1 or live[0].get("task_id") != TASK_ID
            or live[0].get("packet_sha256") != artifact_ref(PACKET)["sha256"]):
        raise RuntimeError("V7_ACCEPTANCE_DOUBLE_WRITER_OR_PACKET_DRIFT")


def finish(packet: dict[str, Any]) -> dict[str, Any]:
    _fence(packet)
    error = None
    try:
        if datetime.now().astimezone() >= datetime.fromisoformat(packet["deadline_at"]):
            error = "V7_ACCEPTANCE_DEADLINE"
        else:
            actual = {name: artifact_ref(REPO_ROOT / name) for name in CODE}
            runtime = [artifact_ref(Path(ref["path"])) for ref in packet["runtime_code"]]
            tests = {name: artifact_ref(REPO_ROOT / name) for name in TESTS}
            if (runtime != packet["runtime_code"] or tests != packet["inputs"]["tests"]
                    or actual != packet["inputs"]["code"]
                    or any(actual[name]["sha256"] != CODE_SHA[name] for name in CODE)):
                error = "V7_ACCEPTANCE_CODE_SHA_DRIFT"
            elif _probe_tests() != packet["inputs"]["negative_tests"]:
                error = "V7_ACCEPTANCE_TEST_DRIFT"
            else:
                for name in EXAMPLES:
                    row = _probe_example(name)
                    old = packet["inputs"]["examples"][name]
                    if row["source_files"] != old["source_files"] or row["submission_input_sha256"] != old["submission_input_sha256"]:
                        error = "V7_ACCEPTANCE_SOURCE_DRIFT:" + name
                        break
    except Exception as exc:
        error = f"V7_ACCEPTANCE_VERIFICATION_ERROR:{type(exc).__name__}:{exc}"
    status = "PASSED" if error is None else "FAILED_RUNTIME_FINAL"
    packet_ref = artifact_ref(PACKET)
    result = {"schema_version": RESULT_SCHEMA, "task_id": TASK_ID,
              "status": status, "run_signature": packet["run_signature"],
              "task_packet": packet_ref, "code": packet["inputs"]["code"],
              "tests": packet["inputs"]["tests"],
              "examples": packet["inputs"]["examples"],
              "negative_tests": packet["inputs"]["negative_tests"],
              "error": error, "model_calls": 0, "render_calls": 0,
              "visual_qualified": False, "training_eligible": False,
              "control_ground_truth": False}
    atomic_json(ATTEMPT / "RESULT.json", result)
    result_ref = artifact_ref(ATTEMPT / "RESULT.json")
    atomic_json(RECEIPT, {"schema_version": RECEIPT_SCHEMA, "task_id": TASK_ID,
            "status": status, "run_signature": packet["run_signature"],
            "task_packet": packet_ref, "result": result_ref})
    receipt_ref = artifact_ref(RECEIPT)
    state, index = copy.deepcopy(load_json(TASK_STATE_PATH)), _index_base(load_json(INDEX))
    row = next(row for row in state["tasks"] if row["task_id"] == TASK_ID)
    row.update(status=status, result=result_ref, terminal_receipt=receipt_ref,
               updated_at=now_iso(), heartbeat_at=now_iso(),
               quality="MACHINE_CODE_ACCEPTANCE_ONLY" if status == "PASSED" else "FAILED")
    state["next_task"] = None
    for item in index["task_packets"]:
        if item.get("task_id") == TASK_ID:
            item["execution_allowed"] = False
    index.update(packet_revision="EXACT78_GENERIC_SUBMIT_CLI_V7_ACCEPTANCE_V1_TERMINAL",
                 plan_revision=packet["plan_revision"],
                 execution_revision=packet["execution_revision"],
                 status="PASS", supersedes_index=artifact_ref(INDEX))
    publish_bundle(load_json(AUTHORITY_PATH), state,
                   event_type="EXACT78_GENERIC_SUBMIT_CLI_V7_ACCEPTANCE_TERMINAL",
                   expected_revision=int(load_json(RECEIPT_PATH)["governance_revision"]),
                   generator_path=SCRIPT, task_packet_index_path=INDEX,
                   task_packet_index_value=index)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--register-and-run", action="store_true")
    args = parser.parse_args(argv)
    revision, _state, _index, inputs, active = inspect()
    if args.check_only:
        if active:
            print(json.dumps({"status": "WAIT_ACTIVE_WRITER", "task_id": TASK_ID,
                              "governance_revision": revision, "examples": inputs["examples"],
                              "negative_tests": inputs["negative_tests"],
                              "registration_started": False, "model_calls": 0}, sort_keys=True))
            return 0
        packet = packet_value(inputs)
        print(json.dumps({"status": "READY_NOT_REGISTERED", "task_id": TASK_ID,
                          "governance_revision": revision,
                          "run_signature_preview": packet["run_signature"],
                          "examples": inputs["examples"],
                          "negative_tests": inputs["negative_tests"],
                          "registration_started": False, "model_calls": 0}, sort_keys=True))
        return 0
    if active:
        raise RuntimeError("V7_ACCEPTANCE_ACTIVE_WRITER_CONFLICT")
    packet = packet_value(inputs)
    register(packet, revision)
    result = finish(packet)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
