from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from chaoyang.ops import run_0915_stereo_interaction_cpu_canary_v1 as subject


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def test_lane_signatures_have_disjoint_inputs_and_no_weight() -> None:
    stereo = subject.lane_signature("stereo_preflight")
    interaction = subject.lane_signature("interaction_v0a")
    assert stereo["lane_id"] != interaction["lane_id"]
    assert stereo["weights"] == interaction["weights"] == "ABSENT"
    assert "source_stereo" not in str(interaction)
    assert "sam_root" not in str(stereo)
    with pytest.raises(ValueError, match="unsupported lane"):
        subject.lane_signature("sibling_escape")


def test_validate_lane_terminal_checks_pid_token_signature_and_write_root(
    tmp_path: Path,
) -> None:
    lane = "interaction_v0a"
    root = tmp_path / "lanes" / lane
    root.mkdir(parents=True)
    signature = subject.lane_signature(lane)
    signature_sha = subject._canonical_sha(signature)
    parent_token = "parent-fence-token-for-test"
    token = hashlib.sha256(
        f"{parent_token}:{lane}:{signature_sha}".encode()
    ).hexdigest()
    token_sha = hashlib.sha256(token.encode()).hexdigest()
    result = root / "RESULT.json"
    _write_json(result, {"status": "COMPLETED_DEVELOPMENT_EVIDENCE"})
    _write_json(root / "RUN_SIGNATURE.json", {
        **signature, "run_signature_sha256": signature_sha,
    })
    claim = {
        "parent_task_id": subject.TASK_ID,
        "lane_id": lane,
        "pid": 123,
        "proc_start_ticks": 456,
        "executor_epoch": 7,
        "run_signature_sha256": signature_sha,
        "fencing_token_sha256": token_sha,
        "unique_write_root": str(root.resolve()),
    }
    _write_json(root / "LANE_CLAIM.json", claim)
    _write_json(root / "LANE_TERMINAL.json", {
        "parent_task_id": subject.TASK_ID,
        "lane_id": lane,
        "run_signature_sha256": signature_sha,
        "fencing_token_sha256": token_sha,
        "result": subject.ref(result),
    })
    value = subject.validate_lane_terminal(
        lane, root, parent_token=parent_token, executor_epoch=7,
        expected_pid=123,
    )
    assert value["result"]["status"] == "COMPLETED_DEVELOPMENT_EVIDENCE"

    claim["unique_write_root"] = str((tmp_path / "lanes/stereo_preflight").resolve())
    _write_json(root / "LANE_CLAIM.json", claim)
    with pytest.raises(RuntimeError, match="claim_write_root"):
        subject.validate_lane_terminal(
            lane, root, parent_token=parent_token, executor_epoch=7,
            expected_pid=123,
        )


def test_packed_mask_loader_preserves_big_endian_bits(tmp_path: Path) -> None:
    masks = np.zeros((2, 3, 5), bool)
    masks[0, 1, 2] = True
    masks[1, 2, 4] = True
    packed = np.packbits(masks.reshape(2, -1), axis=1, bitorder="big")
    path = tmp_path / "mask.npz"
    np.savez_compressed(
        path, packed=packed, frame_count=np.int32(2), height=np.int32(3),
        width=np.int32(5), bitorder=np.asarray("big"),
    )
    assert np.array_equal(subject._load_packed_masks(path), masks)


def test_parent_fails_closed_before_join_if_any_lane_fails(
    tmp_path: Path,
) -> None:
    # This is a structural assertion: only zero return codes reach terminal
    # validation and the joined PASSED result.
    source = Path(subject.__file__).read_text(encoding="utf-8")
    assert "if any(value != 0 for value in returncodes.values())" in source
    assert '"first_blocker": "CPU_LANE_RUNTIME_FAILURE"' in source
    assert "validate_lane_terminal" in source
