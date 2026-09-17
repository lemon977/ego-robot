import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chaoyang.ops.run_chips001_pico_mask_temporal import validate_gpu_lease_owner
from chaoyang.ops.run_sensor_h3_foundationstereo_canary_v71 import published_ref
from chaoyang.ops.run_v71_post_visual_aux_sensor_canaries import terminal_payload


def test_h4_accepts_complete_v71_lease_and_rejects_wrong_owner() -> None:
    lease = {
        "schema_version": "chaoyang-gpu-lease-v71",
        "status": "ACQUIRED",
        "task_id": "sensor_h4_mask_v1",
        "fencing_token": "token",
        "process_startticks": "1",
        "heartbeat_at": "2026-09-15T00:00:00+08:00",
        "expires_at": "2026-09-15T00:02:00+08:00",
        "executor_epoch": 1,
    }
    validate_gpu_lease_owner(lease, "sensor_h4_mask_v1")
    with pytest.raises(RuntimeError, match="owner mismatch"):
        validate_gpu_lease_owner(lease, "another_task")


def test_h4_rejects_incomplete_v71_lease() -> None:
    with pytest.raises(RuntimeError, match="incomplete"):
        validate_gpu_lease_owner(
            {"schema_version": "chaoyang-gpu-lease-v71", "status": "ACQUIRED", "task_id": "sensor_h4_mask_v1"},
            "sensor_h4_mask_v1",
        )


def test_published_ref_hashes_staging_but_records_final_path(tmp_path: Path) -> None:
    staging = tmp_path / "staging.bin"
    staging.write_bytes(b"abc")
    published = tmp_path / "final" / "artifact.bin"
    value = published_ref(staging, published)
    assert value == {
        "path": str(published.resolve()),
        "bytes": 3,
        "sha256": hashlib.sha256(b"abc").hexdigest(),
    }


def test_h3_passed_canary_closes_full_batch_as_resource_budget(tmp_path: Path) -> None:
    canary = tmp_path / "canary.json"
    gpu = tmp_path / "gpu.json"
    canary.write_text(json.dumps({
        "status": "PASS_DEVELOPMENT_CANARY",
        "single_frame_wall_seconds_including_calibration_and_model_load": 60,
    }))
    gpu.write_text(json.dumps({"status": "PASSED"}))
    status, result = terminal_payload("h3", canary, gpu)
    assert status == "BLOCKED_RESOURCE"
    assert result["reason"] == "FULL_BATCH_EXCEEDS_12_GPU_HOUR_BUDGET"
    assert result["canary_automatic_gate_pass"] is True


def test_h4_failed_automatic_gate_is_quality_terminal(tmp_path: Path) -> None:
    canary = tmp_path / "canary.json"
    gpu = tmp_path / "gpu.json"
    canary.write_text(json.dumps({
        "status": "HOLD_AUTOMATIC_GATE",
        "automatic_gate_pass": False,
        "frame_count": 12,
        "resource": {"wall_seconds": 120},
    }))
    gpu.write_text(json.dumps({"status": "FAILED_RUNTIME_FINAL"}))
    status, result = terminal_payload("h4", canary, gpu)
    assert status == "FAILED_QUALITY_C"
    assert result["reason"] == "DEVELOPMENT_CANARY_FAILED_QUALITY_GATE"
