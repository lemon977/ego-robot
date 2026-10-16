"""Finite v2 successor for v9 CPU-bridge code acceptance after sealed v1 failure."""

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


TASK_ID = "exact78_cli_v9_cpu_bridge_code_acceptance_v2"
PHASE = "EXACT78_CLI_V9_CPU_BRIDGE_CODE_ACCEPTANCE"
RESULT_SCHEMA = "exact78-cli-v9-cpu-bridge-acceptance-result-v2"
RECEIPT_SCHEMA = "exact78-cli-v9-cpu-bridge-acceptance-receipt-v2"
SCRIPT = Path(__file__).resolve()
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
ATTEMPT = REPO_ROOT / "_run/current" / TASK_ID / "attempts/attempt_0001"
PACKET = REPO_ROOT / "tasks/current" / TASK_ID / "TASK_PACKET.json"
RECEIPT = REPO_ROOT / "tasks/receipts" / f"{TASK_ID.upper()}.json"
ACTIVE = frozenset({"CLAIMED", "RUNNING", "WAIT_GPU_RESOURCE"})
V7_RECEIPT = REPO_ROOT / "tasks/receipts/EXACT78_GENERIC_SUBMIT_CLI_V7_CODE_ACCEPTANCE_V1.json"
V7_RECEIPT_SHA = "3a164d42a9712801580d8f70d786b16b27c6fb4acc4906817ee543b1fbc8dd40"
V1_FAILURE_RECEIPT = REPO_ROOT / "tasks/receipts/EXACT78_CLI_V9_CPU_BRIDGE_CODE_ACCEPTANCE_V1.json"
V1_FAILURE_RECEIPT_SHA = "0e4ad1f575b500f0534e1efbfc26008391d415303bf0b0a4732b68d2ca74bffa"
V1_FAILURE_RESULT_SHA = "d6985959f382886c3b86bd653d8aba3a752c58ffe657923f084cd5667fcb4a01"
CODE_SHA = {
    "src/chaoyang/cli_v9/__init__.py": "969331acc9fb9af355f851a0a411e535b5e74e77f56f8c24717af7cf796b826c",
    "src/chaoyang/cli_v9/cpu_admission_packet_v1.py": "a326b60513dac10da261d0ac392c2c67d61a18763eefb76632d27817c21b2d24",
    "src/chaoyang/governance/execute_exact78_cli_v9_cpu_admission_candidate_v1.py": "cdc9ae4d932c119e89ef35f2d67ae58130474c7ce3d570a3d5952b8ee54db520",
    "tests/unit/test_exact78_cli_v9_cpu_admission_candidate_v1.py": "e51fdaa4de7134ff07804959b1c724562053be8dfd9ee6a6e167188631330ded",
    "tests/unit/test_exact78_cli_v9_cpu_bridge_acceptance_candidate_v1.py": "fb39291f87c577448f339db7dfbc04b839969b346f15a8909d918ce6602a318c",
    "tests/unit/test_exact78_cli_v9_cpu_bridge_acceptance_candidate_v2.py": "e630ad59b27e396676d740a2bd3afa5bdb2dd89a0f1a04e09016578c3cf4a6dc",
}
DEPENDENCY_SHA = {
    "src/chaoyang/cli_v7/source_neutral_plan_v1.py": "28027c07b20593130ff63446f4ed688571d7642b7e7b1e3ea650403ae741c548",
    "src/chaoyang/cli_v8/source_neutral_stage_packet_plan_v1.py": "2b9731c1fabd0cfff79011fc8f3b682311cb5d6ec30a4b36e48590edfc27c749",
    "src/chaoyang/governance/execute_exact78_first_submit_cpu_candidate_v1.py": "8d60090a7473c5c0c68df84becc8bcc17f9b6aafeb044ab1e9bdf660834fb455",
    "src/chaoyang/governance/execute_exact78_cli_v9_cpu_bridge_acceptance_candidate_v1.py": "18a1a366190dfdca774eb12d61d178577e949b2299e54693d8d8964bce69a796",
}
EXAMPLES = {
    "tracker": {
        "root": "/path/to/private_data/egodata/datasets/ego/chips_cards_tracker_0902/playing_cards/play_cards_0902_005",
        "source_kind": "TRACKER_CLIP_RGB", "decoded_frames": 183,
        "physical_eye": "tracker_session_left_RGB", "time_mode": "TRACKER_CLIP_METADATA_ONLY",
        "v7_source_only_sha256": "45ea9f87c6f6c397a72aaaa725ed53c1cd2b17ec31645cd958e95e9e7cf477d9",
        "descriptor_sha256": "dc8f177afb7f53e43117d42c2e5d2a6df9dfc6870d34462ed300aabf8a5eb080",
        "task_id_preview": "exact78_newraw_play_cards_0902_005_dc8f177afb7f53e4_v1",
        "run_signature_preview": "5062b2f9b3bed2d99b695913127c96d3418650af95377f2e2087ae6208c7b98d",
        "roles": {"video": "e26461a363e558a8fabed8376a4786d6def68da02be7bdfc7bb17f617edcdf0f",
                  "source_stereo_video": "3f658e5538b48ca71075cc27080245e42f50f45b8c352b7c3d56e6b0768d6797",
                  "camera_params": "f44e237577f640e6e0c1262dcf384f524c25d38e91e2b52c4a17239d9b40f2bf",
                  "slam": "3d99a002b11554ebd0937cd4a69db779e01880ad3fcaf6b2e917931205ee10eb",
                  "clip_manifest": "12b3c51744d5a10c7eeefb1a4aa50bb4b1aaea4cfc706fad95925df65b1f2487"},
    },
    "vst": {
        "root": "/path/to/private_data/egodata/datasets/ego/chips_cards_hands_0915/cards_120_0915/005",
        "source_kind": "0915_RAW_VST", "decoded_frames": 149,
        "physical_eye": "left_sourceIndex_1", "time_mode": "TIMEBASE_UNVERIFIED",
        "v7_source_only_sha256": "e1382d626d24b52ff2f40b704945ae95031921359b8d26ee3f4d53e4363ac235",
        "descriptor_sha256": "388cb761cdcfb0301418c99ad3e3a16227e8f0459f4b3066234ecf1abfd91b48",
        "task_id_preview": "exact78_newraw_play_cards_0915_005_388cb761cdcfb030_v1",
        "run_signature_preview": "a68a8dec4c64c8e84afbdd0c32df1dfb0bcfe3601d0c0a988d4583459a9ca5b7",
        "roles": {"raw/vst.h264": "0c10de4183a8c64e9a92b452995fac236db968367bf5c57f6691424edd1f6cac",
                  "raw/vst.qpc.ts.jsonl": "4b34506cf6dfdb7c385f965944bffa723e71bf41e43d4bc3549a473edf60cf8f",
                  "raw/vst.ts.jsonl": "44ced372766a0ec832eb9a8465f604d60bd9d59a235a33e4e4551ba44743be80",
                  "raw/camera_params.json": "19c8e5dbe471c91e84214a047882c1abf3087b865c863d8a398c497236be3352",
                  "raw/camera_params.meta.json": "32450510c43ea153c6689e858fdb3515fbe6aa2f65709f0b9cc588f59a2eac87"},
    },
}


def _snapshot() -> tuple[int, dict[str, Any], dict[str, Any], bool]:
    receipt, state, index = load_json(RECEIPT_PATH), load_json(TASK_STATE_PATH), load_json(INDEX)
    for key, path in (("task_state", TASK_STATE_PATH), ("task_packet_index_current", INDEX)):
        ref = receipt.get("files", {}).get(key)
        if not isinstance(ref, dict) or ref.get("path") != str(path) or validate_artifact_ref(ref):
            raise RuntimeError("V9_ACCEPTANCE_V2_GOVERNANCE_REF_DRIFT:" + key)
    if (any(row.get("task_id") == TASK_ID for row in state.get("tasks", []))
            or any(row.get("task_id") == TASK_ID for row in index.get("task_packets", []))
            or any(path.exists() for path in (ATTEMPT, PACKET.parent, PACKET, RECEIPT))):
        raise RuntimeError("V9_ACCEPTANCE_V2_TASK_ALREADY_USED")
    active_tasks = [row for row in state.get("tasks", []) if row.get("status") in ACTIVE]
    active_packets = [row for row in index.get("task_packets", []) if row.get("execution_allowed") is True]
    if len(active_tasks) > 1 or len(active_packets) > 1:
        raise RuntimeError("V9_ACCEPTANCE_V2_DOUBLE_WRITER")
    if (bool(active_tasks) != bool(active_packets)
            or bool(state.get("next_task")) != bool(active_tasks)
            or (active_tasks and active_tasks[0].get("task_id") != active_packets[0].get("task_id"))):
        raise RuntimeError("V9_ACCEPTANCE_V2_ACTIVE_LEDGER_INDEX_DRIFT")
    return int(receipt["governance_revision"]), state, index, bool(active_tasks)


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(PYTHONPATH=str(REPO_ROOT / "src"), PYTHONDONTWRITEBYTECODE="1",
               PYTHONNOUSERSITE="1", TMPDIR=str(REPO_ROOT / "_run"),
               XDG_CACHE_HOME=str(REPO_ROOT / "_run/caches"))
    return env


def _probe_tests() -> dict[str, Any]:
    command = ["/usr/local/bin/python", "-B", "-m", "unittest", "-q",
               "tests.unit.test_exact78_cli_v9_cpu_bridge_acceptance_candidate_v2"]
    result = subprocess.run(command, cwd=REPO_ROOT, env=_env(),
                            capture_output=True, timeout=120, check=False)
    output = result.stdout + result.stderr
    count = re.search(rb"Ran (\d+) tests?", output)
    if result.returncode or count is None or int(count.group(1)) != 21 or b"OK" not in output:
        raise RuntimeError("V9_ACCEPTANCE_V2_TESTS_NOT_PASS:" + output.decode("utf-8", "replace")[-400:])
    normalized = re.sub(rb"(Ran \d+ tests? in )\d+(?:\.\d+)?s", rb"\g<1><elapsed>s", output)
    return {"status": "PASSED", "test_count": int(count.group(1)),
            "output_sha256": hashlib.sha256(normalized).hexdigest()}


def _probe_example(name: str) -> dict[str, Any]:
    expected = EXAMPLES[name]
    command = ["/usr/local/bin/python", "-B", "-m",
               "chaoyang.governance.execute_exact78_cli_v9_cpu_admission_candidate_v1",
               "--session", expected["root"], "--check-only"]
    result = subprocess.run(command, cwd=REPO_ROOT, env=_env(),
                            capture_output=True, timeout=120, check=False)
    try:
        value = json.loads(result.stdout)
    except (UnicodeDecodeError, ValueError) as exc:
        raise RuntimeError("V9_ACCEPTANCE_V2_CHECK_ONLY_NOT_JSON:" + name) from exc
    source_files = value.get("source_files") if isinstance(value, dict) else None
    required = ("source_kind", "decoded_frames", "physical_eye", "time_mode",
                "v7_source_only_sha256", "descriptor_sha256", "task_id_preview",
                "run_signature_preview")
    if (result.returncode or not isinstance(source_files, dict)
            or any(value.get(key) != expected[key] for key in required)
            or value.get("status") not in {"READY_NOT_REGISTERED", "WAIT_ACTIVE_WRITER"}
            or value.get("route") != "PUBLISHED_CPU_CANDIDATE_ROUTE"
            or set(source_files) != set(expected["roles"])
            or any(source_files[role].get("sha256") != digest
                   for role, digest in expected["roles"].items())
            or value.get("registration_started") is not False
            or value.get("model_calls") != 0 or value.get("gpu_count") != 0
            or value.get("training_eligible") is not False
            or value.get("visual_qualified") is not False):
        raise RuntimeError("V9_ACCEPTANCE_V2_SOURCE_OR_CHECK_ONLY_DRIFT:" + name)
    return {key: value[key] for key in required} | {
        "status": value["status"], "route": value["route"],
        "source_root": expected["root"], "source_files": source_files}


def inspect() -> tuple[int, dict[str, Any], dict[str, Any], dict[str, Any], bool]:
    revision, state, index, active = _snapshot()
    code = {name: artifact_ref(REPO_ROOT / name) for name in CODE_SHA}
    dependencies = {name: artifact_ref(REPO_ROOT / name) for name in DEPENDENCY_SHA}
    if (any(code[name]["sha256"] != sha for name, sha in CODE_SHA.items())
            or any(dependencies[name]["sha256"] != sha for name, sha in DEPENDENCY_SHA.items())):
        raise RuntimeError("V9_ACCEPTANCE_V2_PINNED_CODE_SHA_DRIFT")
    receipt_ref = artifact_ref(V7_RECEIPT)
    v7_receipt = load_json(V7_RECEIPT)
    if (receipt_ref["sha256"] != V7_RECEIPT_SHA or v7_receipt.get("status") != "PASSED"
            or validate_artifact_ref(v7_receipt["result"])
            or load_json(Path(v7_receipt["result"]["path"])).get("status") != "PASSED"):
        raise RuntimeError("V9_ACCEPTANCE_V2_V7_EVIDENCE_DRIFT")
    failed_ref = artifact_ref(V1_FAILURE_RECEIPT)
    failed = load_json(V1_FAILURE_RECEIPT)
    if (failed_ref["sha256"] != V1_FAILURE_RECEIPT_SHA
            or failed.get("status") != "FAILED_RUNTIME_FINAL"
            or failed.get("result", {}).get("sha256") != V1_FAILURE_RESULT_SHA
            or validate_artifact_ref(failed["result"])):
        raise RuntimeError("V9_ACCEPTANCE_V2_V1_FAILURE_EVIDENCE_DRIFT")
    examples = {name: _probe_example(name) for name in EXAMPLES}
    expected_status = "WAIT_ACTIVE_WRITER" if active else "READY_NOT_REGISTERED"
    if any(row["status"] != expected_status for row in examples.values()):
        raise RuntimeError("V9_ACCEPTANCE_V2_WRITER_STATUS_DRIFT")
    negatives = _probe_tests()
    if int(load_json(RECEIPT_PATH)["governance_revision"]) != revision:
        raise RuntimeError("V9_ACCEPTANCE_V2_REVISION_CHANGED_DURING_PROBE")
    inputs = {"code": code, "dependencies": dependencies, "v7_receipt": receipt_ref,
              "v7_result": v7_receipt["result"], "v1_failure_receipt": failed_ref,
              "v1_failure_result": failed["result"], "examples": examples,
              "negative_tests": negatives,
              "source_scope": "0902_TRACKER_CARDS005_PLUS_0915_VST_CARDS005_CHECK_ONLY"}
    refs = [*code.values(), *dependencies.values(), receipt_ref, v7_receipt["result"],
            failed_ref, failed["result"],
            *(ref for row in examples.values() for ref in row["source_files"].values())]
    if any(validate_artifact_ref(ref) for ref in refs):
        raise RuntimeError("V9_ACCEPTANCE_V2_SOURCE_OR_CODE_SHA_DRIFT")
    return revision, state, index, inputs, active


def packet_value(inputs: dict[str, Any]) -> dict[str, Any]:
    now = datetime.now().astimezone()
    writer = {"pid": os.getpid(), "proc_start_ticks": process_identity(os.getpid())["start_ticks"],
              "executor_epoch": 1, "fencing_token": secrets.token_hex(24), "scope": TASK_ID}
    runtime = [artifact_ref(path) for path in
               (SCRIPT, REPO_ROOT / "src/chaoyang/governance/common.py",
                REPO_ROOT / "src/chaoyang/governance/register_single_task_packet.py")]
    command = ("PYTHONPATH=src /usr/local/bin/python -B -m "
               "chaoyang.governance.execute_exact78_cli_v9_cpu_bridge_acceptance_candidate_v2 "
               "--register-and-run")
    config = {"cpu_threads_total": 2, "gpu_count": 0, "model_calls": 0,
              "render_calls": 0, "training_steps": 0, "session_attempts_max": 1,
              "wall_seconds_max": 600, "calibration_or_absent": "NOT_CONSUMED",
              "weights": "NOT_CONSUMED"}
    signature = hashlib.sha256(canonical_bytes({"inputs": inputs, "runtime_code": runtime,
        "config": config, "command": command, "schema": RESULT_SCHEMA})).hexdigest()
    packet = {
        "schema_version": "chaoyang-r22-task-packet-v1", "task_id": TASK_ID,
        "objective": "Accept v9 CPU identity bridge in v2 successor with isolated idle and live-writer tests",
        "phase": PHASE, "plan_revision": "EXACT78_RAW_TO_ROBOT_V1_20260930",
        "execution_revision": "EXACT78_CLI_V9_CPU_BRIDGE_ACCEPTANCE_V2",
        "execution_mode": "EXACT78_GOVERNANCE_ONLY", "registered_at": now.isoformat(),
        "deadline_at": (now + timedelta(minutes=10)).isoformat(),
        "read_set": ["AGENTS.md", "docs/current/EXACT78_EXECUTION_ROUTE_V1.json",
                     "docs/plans/EXACT78_RAW_TO_ROBOT_V1_20260930/66_V9_CPU_BRIDGE_ACCEPTANCE_V2_SUCCESSOR_PREP_ZH.md",
                     "tasks/receipts/EXACT78_GENERIC_SUBMIT_CLI_V7_CODE_ACCEPTANCE_V1.json",
                     "tasks/receipts/EXACT78_CLI_V9_CPU_BRIDGE_CODE_ACCEPTANCE_V1.json",
                     "src/chaoyang/cli_v9/cpu_admission_packet_v1.py",
                     "src/chaoyang/governance/execute_exact78_cli_v9_cpu_admission_candidate_v1.py",
                     "tests/unit/test_exact78_cli_v9_cpu_bridge_acceptance_candidate_v2.py"],
        "write_set": [str(ATTEMPT.relative_to(REPO_ROOT)), str(PACKET.relative_to(REPO_ROOT)),
                      str(RECEIPT.relative_to(REPO_ROOT)), str(INDEX.relative_to(REPO_ROOT)),
                      "docs/governance"],
        "prerequisites": ["GOVERNANCE_PASS", "NO_ACTIVE_TASK", "V7_RECEIPT_PASSED",
                          "V9_V1_FAILURE_SEALED",
                          "SOURCE_SHA_PINNED", "V9_TWO_DOMAIN_CHECK_ONLY_READY",
                          "CPU_ONLY", "0916_READ_ONLY", "NO_MODEL_OR_ROBOT"],
        "required_outputs": ["RESULT.json", "terminal_receipt"],
        "budgets": {"workers_max": 1, **config},
        "stop_conditions": ["CODE_OR_SOURCE_SHA_DRIFT", "DOUBLE_WRITER", "DEADLINE",
                            "CHECK_ONLY_NOT_READY", "ONE_ATTEMPT_TERMINAL"],
        "claim_limit": "v9 CPU identity bridge v2 code acceptance only; v1 remains failed; no CPU admission run, HaWoR, Robot, Scene, control, visual or training qualification",
        "writer": writer, "inputs": copy.deepcopy(inputs), "runtime_code": runtime,
        "config": config, "run_signature": signature, "commands": [command],
    }
    if errors := _validate_packet(packet):
        raise RuntimeError("V9_ACCEPTANCE_V2_PACKET_INVALID:" + ";".join(errors))
    return packet


def _virtual_ref(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    raw = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _index_base(index: dict[str, Any]) -> dict[str, Any]:
    return {key: copy.deepcopy(value) for key, value in index.items()
            if key not in {"governance_revision", "generation_id", "generated_at", "supersedes_index"}}


def register(packet: dict[str, Any], revision: int) -> None:
    current, _state, _index, active = _snapshot()
    if active or current != revision:
        raise RuntimeError("V9_ACCEPTANCE_V2_ACTIVE_WRITER_OR_REVISION_CONFLICT")
    if any(row["status"] != "READY_NOT_REGISTERED" for row in packet["inputs"]["examples"].values()):
        raise RuntimeError("V9_ACCEPTANCE_V2_CHECK_ONLY_NOT_READY")
    if (packet["writer"]["pid"] != os.getpid()
            or packet["writer"]["proc_start_ticks"] != process_identity(os.getpid())["start_ticks"]):
        raise RuntimeError("V9_ACCEPTANCE_V2_WRITER_IDENTITY_DRIFT")
    refs = [*packet["runtime_code"], *packet["inputs"]["code"].values(),
            *packet["inputs"]["dependencies"].values(), packet["inputs"]["v7_receipt"],
            packet["inputs"]["v7_result"], packet["inputs"]["v1_failure_receipt"],
            packet["inputs"]["v1_failure_result"],
            *(ref for row in packet["inputs"]["examples"].values()
              for ref in row["source_files"].values())]
    if any(validate_artifact_ref(ref) for ref in refs):
        raise RuntimeError("V9_ACCEPTANCE_V2_INPUT_SHA_DRIFT")
    packet_ref = _virtual_ref(PACKET, packet)
    state, index = copy.deepcopy(load_json(TASK_STATE_PATH)), _index_base(load_json(INDEX))
    stamp = now_iso()
    state["tasks"].append({"task_id": TASK_ID, "phase": PHASE, "attempt": 1,
        "status": "RUNNING", "session": "tracker_0902_cards005_and_vst_0915_cards005_check_only",
        "updated_at": stamp, "heartbeat_at": stamp,
        "pid": packet["writer"]["pid"], "proc_start_ticks": packet["writer"]["proc_start_ticks"],
        "gpu_id": None, "task_packet": packet_ref, "run_signature": packet["run_signature"],
        "quality": "PENDING", "adoption": "NOT_ADOPTED", "review": "MACHINE_AUDIT"})
    state["next_task"] = {"task_id": TASK_ID,
        "session": "tracker_0902_cards005_and_vst_0915_cards005_check_only",
        "prerequisites": packet["prerequisites"], "expected_resource": "CPU_TWO_THREADS_NO_GPU",
        "stop_condition": "one immutable code acceptance or ten-minute deadline"}
    if errors := validate_task_state(state):
        raise RuntimeError("V9_ACCEPTANCE_V2_STATE_INVALID:" + ";".join(errors))
    index["task_packets"].append({"task_id": TASK_ID,
        "packet_path": str(PACKET.relative_to(REPO_ROOT)), "packet_sha256": packet_ref["sha256"],
        "execution_class": "CURRENT_LEDGER_ROUTABLE", "execution_allowed": True})
    index.update(packet_revision="EXACT78_CLI_V9_CPU_BRIDGE_ACCEPTANCE_V2_RUNNING",
                 plan_revision=packet["plan_revision"], execution_revision=packet["execution_revision"],
                 status="PASS", supersedes_index=artifact_ref(INDEX))
    ATTEMPT.mkdir(parents=True, exist_ok=False)
    atomic_json(ATTEMPT / "WRITER.json", packet["writer"])
    PACKET.parent.mkdir(parents=True, exist_ok=False)
    atomic_json(PACKET, packet)
    if artifact_ref(PACKET) != packet_ref:
        raise RuntimeError("V9_ACCEPTANCE_V2_PACKET_VIRTUAL_SHA_DRIFT")
    publish_bundle(load_json(AUTHORITY_PATH), state,
                   event_type="EXACT78_CLI_V9_CPU_BRIDGE_ACCEPTANCE_REGISTERED",
                   expected_revision=revision, generator_path=SCRIPT,
                   task_packet_index_path=INDEX, task_packet_index_value=index)


def _fence(packet: dict[str, Any]) -> None:
    writer = packet["writer"]
    if (writer["pid"] != os.getpid()
            or writer["proc_start_ticks"] != process_identity(os.getpid())["start_ticks"]
            or load_json(ATTEMPT / "WRITER.json") != writer
            or load_json(PACKET) != packet):
        raise RuntimeError("V9_ACCEPTANCE_V2_WRITER_FENCE_DRIFT")
    state, index = load_json(TASK_STATE_PATH), load_json(INDEX)
    rows = [row for row in state["tasks"] if row.get("task_id") == TASK_ID]
    live = [row for row in index["task_packets"] if row.get("execution_allowed") is True]
    if (len(rows) != 1 or rows[0].get("status") != "RUNNING"
            or rows[0].get("run_signature") != packet["run_signature"]
            or (state.get("next_task") or {}).get("task_id") != TASK_ID
            or len(live) != 1 or live[0].get("task_id") != TASK_ID
            or live[0].get("packet_sha256") != artifact_ref(PACKET)["sha256"]):
        raise RuntimeError("V9_ACCEPTANCE_V2_DOUBLE_WRITER_OR_PACKET_DRIFT")


def finish(packet: dict[str, Any]) -> dict[str, Any]:
    _fence(packet)
    error = None
    try:
        if datetime.now().astimezone() >= datetime.fromisoformat(packet["deadline_at"]):
            error = "V9_ACCEPTANCE_V2_DEADLINE"
        else:
            code = {name: artifact_ref(REPO_ROOT / name) for name in CODE_SHA}
            deps = {name: artifact_ref(REPO_ROOT / name) for name in DEPENDENCY_SHA}
            runtime = [artifact_ref(Path(ref["path"])) for ref in packet["runtime_code"]]
            if (code != packet["inputs"]["code"] or deps != packet["inputs"]["dependencies"]
                    or runtime != packet["runtime_code"]
                    or artifact_ref(V7_RECEIPT) != packet["inputs"]["v7_receipt"]
                    or validate_artifact_ref(packet["inputs"]["v7_result"])
                    or artifact_ref(V1_FAILURE_RECEIPT) != packet["inputs"]["v1_failure_receipt"]
                    or validate_artifact_ref(packet["inputs"]["v1_failure_result"])
                    or any(code[name]["sha256"] != sha for name, sha in CODE_SHA.items())
                    or any(deps[name]["sha256"] != sha for name, sha in DEPENDENCY_SHA.items())):
                error = "V9_ACCEPTANCE_V2_CODE_OR_V7_SHA_DRIFT"
            elif _probe_tests() != packet["inputs"]["negative_tests"]:
                error = "V9_ACCEPTANCE_V2_TEST_DRIFT"
            else:
                for name in EXAMPLES:
                    row = _probe_example(name)
                    if (row["source_files"] != packet["inputs"]["examples"][name]["source_files"]
                            or row["v7_source_only_sha256"] != packet["inputs"]["examples"][name]["v7_source_only_sha256"]
                            or row["descriptor_sha256"] != packet["inputs"]["examples"][name]["descriptor_sha256"]):
                        error = "V9_ACCEPTANCE_V2_SOURCE_DRIFT:" + name
                        break
    except Exception as exc:
        error = f"V9_ACCEPTANCE_V2_VERIFICATION_ERROR:{type(exc).__name__}:{exc}"
    status = "PASSED" if error is None else "FAILED_RUNTIME_FINAL"
    packet_ref = artifact_ref(PACKET)
    result = {"schema_version": RESULT_SCHEMA, "task_id": TASK_ID, "status": status,
              "run_signature": packet["run_signature"], "task_packet": packet_ref,
              "code": packet["inputs"]["code"], "dependencies": packet["inputs"]["dependencies"],
              "v7_receipt": packet["inputs"]["v7_receipt"],
              "v1_failure_receipt": packet["inputs"]["v1_failure_receipt"],
              "examples": packet["inputs"]["examples"],
              "negative_tests": packet["inputs"]["negative_tests"], "error": error,
              "model_calls": 0, "render_calls": 0, "visual_qualified": False,
              "training_eligible": False, "control_ground_truth": False}
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
    index.update(packet_revision="EXACT78_CLI_V9_CPU_BRIDGE_ACCEPTANCE_V2_TERMINAL",
                 plan_revision=packet["plan_revision"], execution_revision=packet["execution_revision"],
                 status="PASS", supersedes_index=artifact_ref(INDEX))
    publish_bundle(load_json(AUTHORITY_PATH), state,
                   event_type="EXACT78_CLI_V9_CPU_BRIDGE_ACCEPTANCE_TERMINAL",
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
        raise RuntimeError("V9_ACCEPTANCE_V2_ACTIVE_WRITER_CONFLICT")
    packet = packet_value(inputs)
    register(packet, revision)
    result = finish(packet)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
