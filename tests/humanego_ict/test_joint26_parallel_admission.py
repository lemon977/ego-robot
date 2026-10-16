"""Signed synthetic metadata only; no hardware, data or model calls."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import sys
import tempfile
from types import SimpleNamespace

import pytest

from chaoyang.governance.common import canonical_bytes
from chaoyang.humanego_ict import joint26_parallel_admission as admission
from chaoyang.humanego_ict import joint26_runtime as runtime
from chaoyang.humanego_ict.joint26_contract import PARALLEL_MODE, PARALLEL_ROUTE, TRAIN


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value))
    return runtime.file_ref(path)


@pytest.fixture
def paired(monkeypatch):
    project = Path(__file__).resolve().parents[2]
    cache = project / "_run/maintenance/joint26_parallel_20261014/attempt_0001/test_cache/admission"
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="signed_", dir=cache) as directory:
        root = Path(directory).resolve()
        monkeypatch.setattr(runtime, "ROOT", root)
        monkeypatch.setattr(runtime, "_is_descendant", lambda writer: True)
        lock_path = root / "_run/current/locks/governance.lock"
        lock_path.parent.mkdir(parents=True)
        lock_path.touch()
        packet_path = root / "tasks/current/paired/TASK_PACKET.json"
        writer = {"pid": os.getpid(), "proc_start_ticks": 123,
                  "executor_epoch": 10, "fencing_token": "synthetic_fence"}
        writer_file = root / "_run/coordinator/writer.json"
        dump(writer_file, writer)
        code = dump(root / "runtime_code.json", {"synthetic": True})
        manifest = dump(root / "paired_manifest.json", {"synthetic": True})
        expiry = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()
        leases = {}
        lanes = {}
        for branch, gpu_ids in (("raw", [0, 1]), ("robotized", [2, 3])):
            config = dump(root / f"{branch}_config.json", {
                "schema": "humanego-ict-joint26-training-config-v1",
                "training": {"max_updates": 1234}})
            output = root / f"output/lanes/{branch}/output"
            args = ["train", "--task-packet", str(packet_path), "--manifest", manifest["path"],
                    "--config", config["path"], "--output", str(output), "--branch", branch]
            paths = []
            for gpu_id in gpu_ids:
                path = root / f"_run/current/ICT_GPU_LEASES/gpu_{gpu_id}.json"
                lease = {"status": "ACQUIRED", "task_id": "paired", "hostname": socket.gethostname(),
                         "pid": writer["pid"], "process_startticks": writer["proc_start_ticks"],
                         "executor_epoch": writer["executor_epoch"], "fencing_token": writer["fencing_token"],
                         "gpu_id": gpu_id, "expires_at": expiry}
                dump(path, lease)
                leases[gpu_id] = lease
                paths.append(str(path))
            lane = {"branch": branch, "inputs": {"paired_manifest": manifest, "training_config": config},
                    "output_dir": str(output), "gpu_ids": gpu_ids, "gpu_lease_paths": paths,
                    "budgets": {"gpu_count": 2, "optimizer_updates_max": 1234},
                    "commands": [shlex.join(["python", "-m", "chaoyang.humanego_ict.joint26_cli", "run", TRAIN, *args])]}
            lane["lane_signature"] = admission.lane_signature(lane)
            lanes[branch] = lane
        guard_path = root / "_run/current/GPU_LEASE.json"
        guard = deepcopy(leases[0])
        dump(guard_path, guard)
        packet = {"task_id": "paired", "execution_mode": PARALLEL_MODE, "route_name": PARALLEL_ROUTE,
                  "deadline_at": expiry, "writer": writer, "writer_file": str(writer_file),
                  "runtime_code": [code], "gpu_lease_path": str(guard_path), "lanes": lanes,
                  "commands": [command for lane in lanes.values() for command in lane["commands"]],
                  "control_ground_truth": False, "physical_deployment_authorized": False}
        state = {"tasks": [{"task_id": "paired", "status": "RUNNING", "pid": writer["pid"],
                            "proc_start_ticks": writer["proc_start_ticks"]}]}
        contract = {"special_status_contracts": {PARALLEL_ROUTE: {
            "candidate_operation": TRAIN, "task_packet_required": True,
            "physical_deployment_authorized": False, "code_closure": [code]}}}
        index = {"task_packets": []}

        def publish(*, resign=True, resign_lanes=True, reset_index=True):
            if resign_lanes:
                for lane in lanes.values():
                    lane["lane_signature"] = admission.lane_signature(lane)
            if resign:
                packet["run_signature"] = hashlib.sha256(canonical_bytes({
                    key: value for key, value in packet.items() if key != "run_signature"})).hexdigest()
            state["tasks"][0]["run_signature"] = packet["run_signature"]
            ref = dump(packet_path, packet)
            if reset_index:
                index["task_packets"] = [{"task_id": "paired", "execution_allowed": True,
                                          "packet_path": str(packet_path.relative_to(root)),
                                          "packet_sha256": ref["sha256"]}]
            files = {"algorithm_contract": dump(root / "contract.json", contract),
                     "task_state": dump(root / "state.json", state),
                     "task_packet_index_current": dump(root / "index.json", index)}
            dump(root / "docs/governance/CURRENT_STATUS_RECEIPT.json", {"files": files})
        publish()
        yield SimpleNamespace(root=root, packet=packet, path=packet_path, lanes=lanes, state=state,
                              contract=contract, index=index, leases=leases, guard=guard, publish=publish)


def admit(paired, branch="raw", *, world_size=2, rank=0, dispatch=False):
    lane = paired.lanes[branch]
    gate = runtime.authorize_joint26_training if dispatch else admission.authorize_parallel_training
    return gate(paired.path, lane["inputs"]["paired_manifest"]["path"],
                lane["inputs"]["training_config"]["path"], lane["output_dir"],
                branch, world_size, rank=rank)


@pytest.mark.parametrize("branch", ["raw", "robotized"])
def test_exact_lane_admission_and_runtime_dispatch(paired, branch):
    before = set(sys.modules)
    packet = admit(paired, branch, rank=1, dispatch=True)
    assert packet["task_id"] == paired.packet["task_id"]
    assert packet["run_signature"] == paired.packet["run_signature"]
    assert packet["writer"] == paired.packet["writer"]
    assert packet["allocated_gpu_ids"] == paired.lanes[branch]["gpu_ids"]
    assert packet["gpu_lease_path"] == paired.lanes[branch]["gpu_lease_paths"][0]
    assert packet["lane_signature"] == paired.lanes[branch]["lane_signature"]
    assert not any(name == "torch" or name.startswith("torch.") for name in set(sys.modules) - before)


def test_lane_signature_uses_governance_canonical_bytes(paired):
    lane = paired.lanes["raw"]
    payload = {key: value for key, value in lane.items() if key != "lane_signature"}
    assert lane["lane_signature"] == hashlib.sha256(canonical_bytes(payload)).hexdigest()


@pytest.mark.parametrize("gpu_id", [0, 1, 2, 3])
@pytest.mark.parametrize("field,value", [
    ("status", "RELEASED"), ("pid", -1), ("process_startticks", -1),
    ("hostname", "wrong-host"), ("task_id", "other"), ("fencing_token", "wrong-fence"),
    ("executor_epoch", -1), ("gpu_id", -1),
    ("expires_at", "2020-01-01T00:00:00+00:00"),
])
def test_every_card_lease_is_required_for_either_lane(paired, gpu_id, field, value):
    paired.leases[gpu_id][field] = value
    dump(paired.root / f"_run/current/ICT_GPU_LEASES/gpu_{gpu_id}.json", paired.leases[gpu_id])
    with pytest.raises(ValueError, match="LIVE_GPU_LEASE|AUTHORIZATION_EXPIRED"):
        admit(paired)


def test_central_guard_must_remain_owned(paired):
    paired.guard["fencing_token"] = "other"
    dump(Path(paired.packet["gpu_lease_path"]), paired.guard)
    with pytest.raises(ValueError, match="LIVE_GPU_LEASE"):
        admit(paired, "robotized")


def test_finished_lane_keeps_peer_admitted_while_parent_keeps_all_leases(paired):
    paired.leases[0]["gpu_process_pid"] = None
    dump(paired.root / "_run/current/ICT_GPU_LEASES/gpu_0.json", paired.leases[0])
    assert admit(paired, "robotized")["branch"] == "robotized"


def test_lease_renewal_does_not_change_immutable_signatures(paired):
    signature = paired.packet["run_signature"]
    lease = paired.leases[3]
    lease["expires_at"] = (datetime.now(timezone.utc) + timedelta(minutes=2)).isoformat()
    dump(paired.root / "_run/current/ICT_GPU_LEASES/gpu_3.json", lease)
    renewed = admit(paired)
    assert renewed["run_signature"] == signature
    assert renewed["admitted_authorization_expires_at"] == datetime.fromisoformat(lease["expires_at"]).timestamp()


def test_overlapping_gpu_groups_are_rejected(paired):
    paired.lanes["robotized"]["gpu_ids"] = [1, 3]
    paired.lanes["robotized"]["gpu_lease_paths"][0] = paired.lanes["raw"]["gpu_lease_paths"][1]
    paired.publish()
    with pytest.raises(ValueError, match="GPU_GROUP_OVERLAP"):
        admit(paired)


def test_outputs_cannot_overlap(paired):
    raw, robotized = paired.lanes["raw"], paired.lanes["robotized"]
    robotized["commands"][0] = robotized["commands"][0].replace(robotized["output_dir"], raw["output_dir"])
    robotized["output_dir"] = raw["output_dir"]
    paired.packet["commands"] = [lane["commands"][0] for lane in paired.lanes.values()]
    paired.publish()
    with pytest.raises(ValueError, match="OUTPUT_OVERLAP"):
        admit(paired)


@pytest.mark.parametrize("mutation,code", [
    (lambda p: p.lanes["raw"].update(branch="robotized"), "LANE_BRANCH_DRIFT"),
    (lambda p: p.lanes["raw"].update(gpu_ids=[0, 0]), "EXACT_GPU_GROUP"),
    (lambda p: p.lanes["raw"].update(gpu_ids=[True, 1]), "EXACT_GPU_GROUP"),
    (lambda p: p.lanes["raw"].update(gpu_lease_paths=[]), "EXACT_GPU_GROUP"),
    (lambda p: p.lanes["raw"]["budgets"].update(optimizer_updates_max=20000), "UPDATE_OR_DDP"),
    (lambda p: p.lanes["raw"]["budgets"].update(gpu_count=True), "UPDATE_OR_DDP"),
    (lambda p: p.lanes["raw"].update(commands=[]), "COMMAND_NOT_PINNED"),
    (lambda p: p.packet.update(commands=[]), "PARENT_COMMANDS_DRIFT"),
    (lambda p: p.packet.update(execution_mode="CPU_ENGINEERING"), "TRAINING_TASK_REQUIRED"),
    (lambda p: p.packet.update(physical_deployment_authorized=True), "RESEARCH_FLAGS"),
    (lambda p: p.packet.update(deadline_at="2020-01-01T00:00:00+00:00"), "AUTHORIZATION_EXPIRED"),
    (lambda p: p.contract["special_status_contracts"][PARALLEL_ROUTE].update(candidate_operation="other"), "ROUTE_NOT_REGISTERED"),
    (lambda p: p.contract["special_status_contracts"][PARALLEL_ROUTE].update(code_closure=[]), "FROZEN_CLOSURE"),
])
def test_signed_metadata_still_must_satisfy_parallel_contract(paired, mutation, code):
    mutation(paired)
    paired.publish()
    with pytest.raises(ValueError, match=code):
        admit(paired)


def test_lane_signature_mismatch_is_rejected_even_with_signed_parent(paired):
    paired.lanes["raw"]["lane_signature"] = "changed"
    paired.publish(resign_lanes=False)
    with pytest.raises(ValueError, match="LANE_SIGNATURE_DRIFT"):
        admit(paired)


def test_parent_signature_mismatch_is_rejected(paired):
    paired.packet["run_signature"] = "changed"
    paired.publish(resign=False)
    with pytest.raises(ValueError, match="COORDINATOR_SIGNATURE_DRIFT"):
        admit(paired)


def test_input_mutation_in_peer_lane_is_rejected(paired):
    Path(paired.lanes["robotized"]["inputs"]["training_config"]["path"]).write_text("{}")
    with pytest.raises(ValueError, match="ARTIFACT_SHA_DRIFT"):
        admit(paired)


def test_closure_mutation_is_rejected(paired):
    Path(paired.packet["runtime_code"][0]["path"]).write_text("{}")
    with pytest.raises(ValueError, match="ARTIFACT_SHA_DRIFT"):
        admit(paired)


def test_writer_file_and_ancestry_are_required(paired, monkeypatch):
    monkeypatch.setattr(runtime, "_is_descendant", lambda writer: False)
    with pytest.raises(ValueError, match="WRITER_ANCESTRY"):
        admit(paired)
    monkeypatch.setattr(runtime, "_is_descendant", lambda writer: True)
    dump(Path(paired.packet["writer_file"]), dict(paired.packet["writer"], fencing_token="changed"))
    with pytest.raises(ValueError, match="WRITER_FENCING_TOKEN"):
        admit(paired)


@pytest.mark.parametrize("status", ["CLAIMED", "WAIT_GPU_RESOURCE", "FAILED"])
def test_only_running_coordinator_is_admitted(paired, status):
    paired.state["tasks"][0]["status"] = status
    paired.publish()
    with pytest.raises(ValueError, match="LIVE_COORDINATOR_STATE"):
        admit(paired)


def test_extra_active_task_is_rejected(paired):
    paired.state["tasks"].append({"task_id": "other", "status": "RUNNING"})
    paired.publish()
    with pytest.raises(ValueError, match="LIVE_COORDINATOR_STATE"):
        admit(paired)


def test_extra_allowed_packet_is_rejected(paired):
    paired.index["task_packets"].append({"task_id": "other", "execution_allowed": True})
    paired.publish(reset_index=False)
    with pytest.raises(ValueError, match="UNIQUE_COORDINATOR_PACKET"):
        admit(paired)


@pytest.mark.parametrize("world_size,rank", [(0, 0), (True, 0), (2, -1), (2, 2), (2, True)])
def test_invalid_ddp_topology_rejected(paired, world_size, rank):
    with pytest.raises(ValueError, match="EXPLICIT_DDP_TOPOLOGY"):
        admit(paired, world_size=world_size, rank=rank)


def test_wrong_lane_world_size_is_rejected(paired):
    with pytest.raises(ValueError, match="UPDATE_OR_DDP"):
        admit(paired, world_size=1)
