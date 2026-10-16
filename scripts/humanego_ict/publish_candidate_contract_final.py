"""Final metadata publication for the already-sealed bounded CPU route task.

Run after its held publisher exits, so the new pure builder is imported from
the frozen source, not the engineering process's earlier import cache. Never
registers or invokes an algorithm, GPU, training or physical deployment.
"""
from __future__ import annotations

from pathlib import Path

from chaoyang.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH, artifact_ref,
    atomic_json, load_json, publish_bundle,
)

TASK = "pick_place_humanego_ict_revo2_route_publication_v1_20261010"


def main() -> int:
    root = REPO_ROOT / "_run/current" / TASK / "attempts/attempt_0001"
    state = load_json(TASK_STATE_PATH)
    index = load_json(REPO_ROOT / "tasks/current/INDEX.json")
    if (state.get("next_task") or any(row["status"] in {"RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE"} for row in state["tasks"])
            or any(row.get("execution_allowed") for row in index["task_packets"])):
        raise RuntimeError("ACTIVE_PUBLISHER_CONFLICT")
    task = next(row for row in state["tasks"] if row["task_id"] == TASK)
    if task["status"] != "PASSED" or task["result"] != artifact_ref(root / "RESULT.json"):
        raise RuntimeError("SEALED_CPU_ROUTE_RESULT_REQUIRED")
    packet = load_json(REPO_ROOT / "tasks/current" / TASK / "TASK_PACKET.json")
    baseline = packet["inputs"]["baseline_before"]
    if artifact_ref(Path(baseline["path"])) != baseline:
        raise RuntimeError("BASELINE_POINTER_DRIFT")
    result = load_json(root / "RESULT.json")
    for ref in result["output_refs"]:
        if artifact_ref(Path(ref["path"])) != ref:
            raise RuntimeError("ENGINEERING_OUTPUT_SHA_DRIFT:" + ref["path"])
    publish_bundle(load_json(AUTHORITY_PATH), state,
                   event_type="HUMANEGO_ICT_REVO2_CANDIDATE_FINAL_FRESH_PROCESS_PUBLICATION",
                   expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
                   generator_path=Path(__file__), task_packet_index_path=REPO_ROOT / "tasks/current/INDEX.json",
                   task_packet_index_value=index)
    from chaoyang.humanego_ict.runtime_gate import ROUTE, OPERATION
    contract = load_json(REPO_ROOT / "docs/governance/ALGORITHM_CONTRACT.json")
    route = contract["special_status_contracts"][ROUTE]
    if route["candidate_operation"] != OPERATION or route["current_runnable_operation"] is not None:
        raise RuntimeError("CANDIDATE_ONLY_PUBLICATION_REQUIRED")
    atomic_json(root / "CONTRACT_PUBLICATION.json", {
        "status": "PASSED", "task_id": TASK, "candidate_operation": OPERATION,
        "algorithm_contract": artifact_ref(REPO_ROOT / "docs/governance/ALGORITHM_CONTRACT.json"),
        "receipt": artifact_ref(RECEIPT_PATH), "formal_training_started": False,
        "execution_allowed": False, "physical_deployment_authorized": False,
    })
    print("Candidate contract published; no training task authorized.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
