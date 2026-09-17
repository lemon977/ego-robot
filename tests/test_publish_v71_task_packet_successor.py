from chaoyang.governance.publish_v71_task_packet_successor import OLD_LEDGER, UPSTREAM_STATE, successor_packet


def test_successor_replaces_missing_future_ledger_with_existing_upstream_state():
    packet = {
        "schema_version": "exact78-task-packet-v1",
        "plan_revision": "chaoyang-v7.1",
        "task_id": "visual_aux_chips_pair_v1",
        "objective": "test",
        "read_set": ["docs/governance/CURRENT_STATUS_RECEIPT.json", OLD_LEDGER],
        "write_set": ["out"],
        "prerequisites": [],
        "gates": [],
        "budgets": {"cpu_seconds": 1, "gpu_seconds": 1, "wall_seconds": 1},
        "attempt_max": 1,
        "stop_condition": "terminal",
        "output_contract": ["RESULT.json"],
        "claim_limit": "test",
        "executor_epoch": 1,
        "fencing": {"pid_startticks_required": True, "immutable_final": True, "partial_attempt_is_never_successor_input": True},
        "artifact_revision_contract": {"required_revision_status": "VALID_FOR_PINNED_REVISION", "forbid_in_place_overwrite": True},
        "ai_io_limits": {"max_files_initial_read": 8, "max_search_results": 20, "max_log_tail_lines": 80, "max_directory_depth": 3, "full_log_read_requires_failure": True},
        "initial_search_result_limit": 20,
        "initial_log_line_limit": 80,
        "created_at": "old",
    }
    result = successor_packet(packet, {"path": "old", "bytes": 1, "sha256": "0" * 64})
    assert OLD_LEDGER not in result["read_set"]
    assert UPSTREAM_STATE in result["read_set"]
    assert result["packet_revision"] == "R7_2"
    assert result["executor_epoch"] == 2
