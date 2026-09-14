from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import sys

import pytest


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
import immutable_artifact_io as strict_io  # noqa: E402
import qa_checkpoint_2000_final_control_plane_artifacts as qa  # noqa: E402


def record(path: Path, payload: bytes) -> strict_io.VerifiedBytes:
    return strict_io.VerifiedBytes(
        path=str(path), payload=payload, bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(), device=7, inode=abs(hash(str(path))) + 1,
    )


def evidence(item: strict_io.VerifiedBytes) -> dict[str, object]:
    return item.evidence_ref()


def terminal_fixture() -> tuple[dict[str, object], dict[str, strict_io.VerifiedBytes], dict[str, object]]:
    records = {
        name: record(Path(f"/synthetic/{name}.json"), name.encode())
        for name in ("metrics", "redline", "pack", "supervisor_hb", "guardian_hb")
    }
    pack = {"exact_refs": {
        "supervisor_cutoff_heartbeat": evidence(records["supervisor_hb"]),
        "guardian_cutoff_heartbeat": evidence(records["guardian_hb"]),
    }}
    terminal = {
        "status": qa.TERMINAL_SUCCESS,
        "execution_gpu_pixel_queue_admission": 0,
        "result": {
            "final_metrics": evidence(records["metrics"]),
            "final_redline": evidence(records["redline"]),
            "final_pack": evidence(records["pack"]),
        },
    }
    return terminal, records, pack


def test_terminal_success_and_transitive_five_output_binding() -> None:
    terminal, records, pack = terminal_fixture()
    assert qa._terminal(terminal, records, pack) == "THREE_DIRECT_TWO_TRANSITIVE_VIA_EXACT_PACK"


@pytest.mark.parametrize("bad", ["HOLD_WAITING_OWNER", "FAILED_FINALIZER"])
def test_terminal_hold_and_failed_are_rejected(bad: str) -> None:
    terminal, records, pack = terminal_fixture()
    terminal["status"] = bad
    with pytest.raises(qa.QAError, match="exact success"):
        qa._terminal(terminal, records, pack)


def test_terminal_conflicting_overall_failed_is_rejected() -> None:
    terminal, records, pack = terminal_fixture()
    terminal["overall_verdict"] = {"verdict": "FAILED_BUT_TOP_LEVEL_SUCCESS"}
    with pytest.raises(qa.QAError, match="conflicting"):
        qa._terminal(terminal, records, pack)


def test_terminal_nested_result_failed_is_rejected() -> None:
    terminal, records, pack = terminal_fixture()
    terminal["result"]["status"] = "FAILED_NESTED"
    with pytest.raises(qa.QAError, match="conflicting"):
        qa._terminal(terminal, records, pack)


def test_conflicting_reference_fields_are_rejected() -> None:
    value = {
        "path": "/synthetic/a.json", "bytes": 1, "sha256": "0" * 64,
        "observed_bytes": 2, "observed_sha256": "1" * 64,
    }
    with pytest.raises(qa.QAError, match="conflicting"):
        qa._mapping_ref(value)


def test_hardlink_inode_alias_is_rejected(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_bytes(b"{}\n")
    os.link(first, second)
    left = strict_io.read_bytes_nofollow(first)
    right = strict_io.read_bytes_nofollow(second)
    with pytest.raises(qa.QAError, match="hardlink"):
        qa._unique({"left": left, "right": right})


def test_single_supplied_hardlink_with_external_alias_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    first = tmp_path / "first.json"
    alias = tmp_path / "external_alias.json"
    payload = b"{}\n"
    first.write_bytes(payload)
    os.link(first, alias)
    monkeypatch.setattr(qa, "PROJECT_ROOT", tmp_path)
    with pytest.raises(qa.QAError, match="hardlink count"):
        qa._read(qa.Expected(first, len(payload), hashlib.sha256(payload).hexdigest()))


def test_symlink_is_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    link = tmp_path / "link.json"
    target.write_bytes(b"{}\n")
    link.symlink_to(target)
    monkeypatch.setattr(qa, "PROJECT_ROOT", tmp_path)
    expected = qa.Expected(link, 3, hashlib.sha256(b"{}\n").hexdigest())
    with pytest.raises(qa.QAError, match="nofollow"):
        qa._read(expected)


def test_expected_sha_drift_is_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "drift.json"
    path.write_bytes(b"{}\n")
    monkeypatch.setattr(qa, "PROJECT_ROOT", tmp_path)
    with pytest.raises(qa.QAError, match="exact nofollow"):
        qa._read(qa.Expected(path, 3, "0" * 64))


def test_nonzero_admission_is_rejected() -> None:
    with pytest.raises(qa.QAError, match="nonzero admission"):
        qa._assert_zero_admission({"nested": {"execution_gpu_pixel_queue_admission": 1}})


def test_metrics_nonzero_admission_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    value = metrics_fixture(monkeypatch, tmp_path)
    value["execution_gpu_pixel_queue_admission"] = 1
    with pytest.raises(qa.QAError, match="nonzero admission"):
        qa._metrics(value)


@pytest.mark.parametrize(
    ("window", "target"),
    [
        ({"start": "2026-08-29T15:59:00+08:00", "cutoff": "2026-08-29T19:40:00+08:00"}, qa.TARGET_TEXT),
        ({"start": qa.START_TEXT, "cutoff": "2026-08-29T20:01:00+08:00"}, qa.TARGET_TEXT),
        ({"start": qa.START_TEXT, "cutoff": "2026-08-29T19:40:00+08:00"}, "2026-08-29T20:01:00+08:00"),
    ],
)
def test_cross_window_is_rejected(window: dict[str, str], target: str) -> None:
    with pytest.raises(qa.QAError, match="cross-window"):
        qa._window(window, target)


def test_cutoff_minute_must_match_output_names() -> None:
    refs = {
        "metrics": qa.Expected(qa.CHECKPOINT_ROOT / "CHECKPOINT_2000_UTILIZATION_CUTOFF_0000_T0_V1.json", 1, "0" * 64),
        "redline": qa.Expected(qa.CHECKPOINT_ROOT / "RED_LINE_INCREMENT_1600_0000_T0_V1.json", 1, "0" * 64),
    }
    with pytest.raises(qa.QAError, match="actual cutoff minute"):
        qa._cutoff_names(refs, qa.datetime.fromisoformat("2026-08-29T19:40:00+08:00"))


def metrics_fixture(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, object]:
    monkeypatch.setattr(qa, "RUN_ROOT", tmp_path)
    rows = [
        {"timestamp": "2026-08-29T16:00:00+08:00", "gpu_utilization_percent": 0, "gpu_memory_used_mib": 0, "gpu_channel_busy": False, "cpu_channel_busy": False, "load_average_1m": 1},
        {"timestamp": "2026-08-29T16:01:00+08:00", "gpu_utilization_percent": 50, "gpu_memory_used_mib": 100, "gpu_channel_busy": True, "cpu_channel_busy": False, "load_average_1m": 3},
    ]
    payload = b"".join((json.dumps(row, sort_keys=True) + "\n").encode() for row in rows)
    source = tmp_path / "UTILIZATION_SAMPLES.jsonl"
    source.write_bytes(payload)
    info = source.stat()
    return {
        "schema_version": "autonomous-20h-checkpoint-utilization-t0-v1", "status": "CHECKPOINT_SNAPSHOT_PREFIX",
        "checkpoint_target": qa.TARGET_TEXT, "requested_window": {"start": qa.START_TEXT, "cutoff": "2026-08-29T16:02:00+08:00"}, "checkpoint_window_complete": False,
        "input_prefixes": {"utilization": {"path": str(source), "observed_prefix_bytes": len(payload), "observed_prefix_sha256": hashlib.sha256(payload).hexdigest(), "device": info.st_dev, "inode": info.st_ino, "total_jsonl_rows": 2}},
        "telemetry_coverage": {"selected_sample_count": 2, "first_sample": "2026-08-29T16:00:00+08:00", "last_sample": "2026-08-29T16:01:00+08:00", "seconds_start_to_first_sample": 0.0, "seconds_last_sample_to_cutoff": 60.0, "median_inter_sample_gap_seconds": 60.0, "maximum_inter_sample_gap_seconds": 60.0, "gap_break_rule_seconds": 150.0},
        "gpu": {"sampled_mean_utilization_percent": 25.0, "sampled_max_utilization_percent": 50.0, "nonzero_utilization_sample_count": 1, "nonzero_utilization_sample_fraction_percent": 50.0, "gpu_channel_busy_sample_count": 1, "gpu_channel_busy_sample_fraction_percent": 50.0, "memory_nonzero_sample_count": 1, "memory_max_mib": 100.0, "sampled_compute_idle_segments": [{"first_sample": "2026-08-29T16:00:00+08:00", "last_sample": "2026-08-29T16:00:00+08:00", "sample_count": 1}]},
        "cpu": {"true_busy_fraction": "UNMEASURED", "online_logical_cpus_from_affinity": 4, "mean_load_average_1m": 2.0, "median_load_average_1m": 2.0, "max_load_average_1m": 3.0, "normalized_mean_load_proxy_percent": 50.0, "supervisor_cpu_channel_busy_sample_count": 0, "supervisor_cpu_channel_busy_sample_fraction_percent": 0.0, "sampled_supervisor_channel_idle_segments": [{"first_sample": "2026-08-29T16:00:00+08:00", "last_sample": "2026-08-29T16:01:00+08:00", "sample_count": 2}]},
    }


def test_metrics_samples_gpu_and_cpu_proxy_recompute(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    value = metrics_fixture(monkeypatch, tmp_path)
    cutoff, recomputed = qa._metrics(value)
    assert cutoff.isoformat() == "2026-08-29T16:02:00+08:00"
    assert recomputed["selected_sample_count"] == 2
    assert recomputed["cpu_true_busy_fraction"] == "UNMEASURED"


def test_metrics_gpu_tamper_is_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    value = metrics_fixture(monkeypatch, tmp_path)
    value["gpu"]["sampled_mean_utilization_percent"] = 24.0
    with pytest.raises(qa.QAError, match="GPU metrics"):
        qa._metrics(value)


def test_exclusive_publication_is_mode_0440_and_no_replace(tmp_path: Path) -> None:
    output = tmp_path / "qa.json"
    strict_io.write_new_json(output, {"overall_verdict": {"verdict": qa.PASS_VERDICT}}, mode=0o440, allowed_root=tmp_path)
    assert stat.S_IMODE(output.stat().st_mode) == 0o440
    with pytest.raises(FileExistsError):
        strict_io.write_new_json(output, {"x": 1}, mode=0o440, allowed_root=tmp_path)


def test_parser_accepts_only_six_explicit_refs() -> None:
    action_dests = {action.dest for action in qa.parser()._actions if action.dest != "help"}
    assert action_dests == {"terminal_ref", "metrics_ref", "redline_ref", "pack_ref", "supervisor_heartbeat_ref", "guardian_heartbeat_ref"}
