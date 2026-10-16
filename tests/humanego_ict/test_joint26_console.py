import io
import json

import pytest

from chaoyang.humanego_ict import joint26_console as console
from chaoyang.humanego_ict.joint26_console import ConsoleReporter


def reporter(**kwargs):
    stream = io.StringIO()
    return ConsoleReporter(stream=stream, **kwargs), stream


def updates(output, *rows, suffix=b""):
    output.mkdir(parents=True, exist_ok=True)
    content = b"".join(json.dumps(row).encode() + b"\n" for row in rows) + suffix
    (output / "updates.jsonl").write_bytes(content)


@pytest.mark.parametrize("branch, label", [
    ("raw", "Raw（原始）"), ("Raw", "Raw（原始）"), ("原始", "Raw（原始）"),
    ("robotized", "Robotized（机器人化）"), ("Robotized", "Robotized（机器人化）"),
    ("机器人化", "Robotized（机器人化）"),
])
def test_human_branch_labels(branch, label):
    report, stream = reporter()
    report.stage(branch, "校验配置与数据")
    assert stream.getvalue() == f"[{label}] 校验配置与数据\n"


def test_preflight_is_compact_and_claims_only_step_completion():
    report, stream = reporter()
    report.preflight("raw", {"schema": "internal-schema", "status": "PASS_RESEARCH_DATA_ONLY",
        "training_samples": 160, "validation_samples": 30, "training_groups": 16,
        "validation_groups": 3, "manifest_ref": {"sha256": "secret-signature"}})
    text = stream.getvalue()
    assert "预检步骤完成" in text
    assert "训练 160 样本／16 组，验证 30 样本／3 组" in text
    assert "schema" not in text and "SHA" not in text and "secret-signature" not in text
    assert "效果通过" not in text and "预检通过" not in text


def test_registration_shows_cards_target_and_full_log_path(tmp_path):
    report, stream = reporter()
    report.registered("robotized", [2, 3], 40000, tmp_path / "output",
                      {"schema": "internal", "run_signature": "sha"})
    text = stream.getvalue()
    assert "卡 2,3，目标 40000 步" in text
    assert str(tmp_path / "EXECUTION.log") in text
    assert "schema" not in text and "sha" not in text


def test_json_preserves_exact_original_objects_and_only_three_events(tmp_path):
    report, stream = reporter(json_mode=True)
    qualification = {"schema": "qualification", "branch": "raw", "ref": {"sha256": "abc"}}
    metadata = {"status": "REGISTERED", "output": str(tmp_path), "task_id": "a"}
    result = {"schema": "result", "status": "PASSED", "artifacts": [{"sha256": "xyz"}]}
    report.stage("raw", "加载")
    report.preflight("raw", qualification)
    report.registered("raw", [0], 40000, tmp_path, metadata)
    report.progress("raw", tmp_path, 40000)
    report.finished("raw", "PASSED", result, tmp_path, 40000)
    assert [json.loads(line) for line in stream.getvalue().splitlines()] == [qualification, metadata, result]


def test_progress_reads_last_complete_row_ignoring_partial_append(tmp_path):
    report, stream = reporter()
    updates(tmp_path, {"optimizer_update": 20, "flow_loss": .4, "validation_flow_loss": .3,
        "step_seconds": 2, "remaining_hours": 5}, suffix=b'{"optimizer_update":40,"flow_loss":')
    report.progress("raw", tmp_path, 20000)
    text = stream.getvalue()
    assert "20/20000 步" in text and "训练损失=0.4" in text and "验证损失=0.3" in text
    assert "flow_loss" not in text and "val_loss" not in text
    assert "步时 2.000秒，预计剩余 5.00小时" in text
    assert "40/20000" not in text


def test_complete_json_without_newline_is_still_an_unfinished_record(tmp_path):
    report, stream = reporter()
    updates(tmp_path, suffix=b'{"optimizer_update":20}')
    report.progress("raw", tmp_path, 20000)
    assert "等待加载" in stream.getvalue()
    assert "20/20000" not in stream.getvalue()


def test_progress_deduplicates_per_branch_and_never_regresses(tmp_path):
    report, stream = reporter()
    updates(tmp_path, {"optimizer_update": 40})
    report.progress("raw", tmp_path, 20000)
    report.progress("raw", tmp_path, 20000)
    report.progress("raw", tmp_path, 20000, final=True)
    report.progress("robotized", tmp_path, 20000)
    updates(tmp_path, {"optimizer_update": 20})
    report.progress("raw", tmp_path, 20000)
    assert stream.getvalue().count("40/20000") == 2
    assert "20/20000" not in stream.getvalue()


@pytest.mark.parametrize("suffix", [b"broken\n", b"\xff\xfe\n", b"[]\n", b"null\n"])
def test_damaged_trailing_records_do_not_throw(tmp_path, suffix):
    report, stream = reporter()
    updates(tmp_path, {"optimizer_update": 20}, suffix=suffix)
    report.progress("raw", tmp_path, 20000)
    assert "20/20000" in stream.getvalue()


def test_missing_or_nonfinite_metrics_are_dashes_without_fake_zero(tmp_path):
    report, stream = reporter()
    updates(tmp_path, {"optimizer_update": 20, "flow_loss": float("nan"),
        "validation_flow_loss": float("inf"), "step_seconds": True, "remaining_hours": None})
    report.progress("raw", tmp_path, 20000)
    text = stream.getvalue()
    assert "训练损失=—，验证损失=—，步时 —，预计剩余 —" in text
    assert "损失=0" not in text and "nan" not in text and "inf" not in text


def test_eta_uses_logged_remaining_hours_and_steady_duration(tmp_path):
    report, stream = reporter()
    updates(tmp_path, {"optimizer_update": 20, "step_seconds": 100,
        "steady_step_seconds": 2.5, "remaining_hours": .75})
    report.progress("raw", tmp_path, 100)
    assert "步时 2.500秒，预计剩余 0.75小时" in stream.getvalue()


def test_eta_is_missing_when_only_step_time_exists(tmp_path):
    report, stream = reporter()
    updates(tmp_path, {"optimizer_update": 20, "step_seconds": 2})
    report.progress("raw", tmp_path, 20000)
    assert "预计剩余 —" in stream.getvalue()


def test_unrepresentable_integer_metric_is_missing(tmp_path):
    report, stream = reporter()
    updates(tmp_path, {"optimizer_update": 20, "flow_loss": 10 ** 1000})
    report.progress("raw", tmp_path, 20000)
    assert "训练损失=—" in stream.getvalue()


def test_excessively_nested_json_is_skipped_without_stopping_worker(tmp_path):
    report, stream = reporter()
    updates(tmp_path, {"optimizer_update": 20}, suffix=b"[" * 2000 + b"0" + b"]" * 2000 + b"\n")
    report.progress("raw", tmp_path, 20000)
    assert "20/20000" in stream.getvalue()


@pytest.mark.parametrize("step", [-1, True, "20", 20.0, 20001])
def test_invalid_update_is_not_counted(tmp_path, step):
    report, stream = reporter()
    updates(tmp_path, {"optimizer_update": step, "flow_loss": .3})
    report.progress("raw", tmp_path, 20000)
    assert "等待加载" in stream.getvalue()
    assert "/20000" not in stream.getvalue() and "训练损失" not in stream.getvalue()


def test_positive_target_above_20000_remains_compatible(tmp_path):
    report, stream = reporter()
    updates(tmp_path, {"optimizer_update": 30000})
    report.progress("raw", tmp_path, 40000)
    assert "30000/40000 步（75.0%）" in stream.getvalue()


def test_tail_read_is_bounded_and_handles_large_files(tmp_path, monkeypatch):
    report, stream = reporter()
    updates(tmp_path, suffix=b"x" * (console._TAIL_BYTES * 2) + b'\n{"optimizer_update":60}\n')
    real_open = console.Path.open
    read_sizes = []

    class ObservedStream:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def seek(self, *args):
            return self.stream.seek(*args)

        def read(self, count):
            read_sizes.append(count)
            return self.stream.read(count)

    monkeypatch.setattr(console.Path, "open", lambda path, *args, **kwargs:
                        ObservedStream(real_open(path, *args, **kwargs)))
    report.progress("raw", tmp_path, 20000)
    assert "60/20000" in stream.getvalue()
    assert read_sizes == [64 * 1024]


def test_loading_wait_message_is_rate_limited(tmp_path, monkeypatch):
    report, stream = reporter(wait_interval=15)
    ticks = iter([0, 14.99, 15, 15.1, 30])
    monkeypatch.setattr(console.time, "monotonic", lambda: next(ticks))
    for _ in range(5):
        report.progress("raw", tmp_path, 20000)
    assert stream.getvalue().count("等待加载") == 3


def test_file_read_error_does_not_affect_reporting(tmp_path, monkeypatch):
    report, stream = reporter()

    def unreadable(*args, **kwargs):
        raise OSError("damaged file")

    monkeypatch.setattr(console.Path, "open", unreadable)
    report.progress("raw", tmp_path, 20000)
    report.finished("raw", "FAILED_RUNTIME_FINAL", {}, tmp_path, 20000)
    assert "等待加载" in stream.getvalue() and "实际完成 —/20000" in stream.getvalue()


def test_finished_reports_actual_step_and_loss_paths(tmp_path):
    report, stream = reporter()
    updates(tmp_path, {"optimizer_update": 120, "flow_loss": .25, "validation_flow_loss": .5})
    (tmp_path / "loss.csv").write_text("optimizer_update,flow_loss\n120,.25\n")
    stage = {"optimizer_updates": 120, "last_checkpoint": str(tmp_path / "checkpoint_000120.pt"),
             "validation_flow_loss": .4}
    report.finished("raw", "FAILED_RUNTIME_FINAL", {"schema": "internal"}, tmp_path, 40000, stage=stage)
    text = stream.getvalue()
    lines = text.splitlines()
    assert "执行失败：实际完成 120/40000 步，训练损失=0.25，验证损失=0.4" in lines[0]
    assert lines[1] == f"权重：{tmp_path / 'checkpoint_000120.pt'}"
    assert lines[2] == f"损失记录：{tmp_path / 'loss.csv'}"
    assert "schema" not in text and "效果通过" not in text


def test_finished_reads_existing_result_when_stage_is_omitted(tmp_path):
    report, stream = reporter()
    stage = {"optimizer_updates": 40000, "last_checkpoint": "checkpoint.pt", "validation_flow_loss": .2}
    (tmp_path / "TRAINING_RESULT.json").write_text(json.dumps(stage))
    report.finished("robotized", "PASSED", {}, tmp_path, 40000)
    assert "执行完成：实际完成 40000/40000 步，训练损失=—，验证损失=0.2" in stream.getvalue()
    assert "损失记录：—" in stream.getvalue()


@pytest.mark.parametrize("step", [-1, True, 20001])
def test_invalid_final_step_is_missing_rather_than_replaced_by_older_log(tmp_path, step):
    report, stream = reporter()
    updates(tmp_path, {"optimizer_update": 20})
    report.finished("raw", "FAILED_RUNTIME_FINAL", {}, tmp_path, 20000,
                    stage={"optimizer_updates": step})
    assert "实际完成 —/20000" in stream.getvalue()


def test_governance_failure_is_not_execution_success(tmp_path):
    report, stream = reporter()
    report.finished("raw", "PASSED", {}, tmp_path, 20000, governance_exit=2)
    assert "执行失败" in stream.getvalue() and "实际完成 —/20000" in stream.getvalue()


def test_failed_result_shows_up_to_three_short_actual_reasons(tmp_path):
    report, stream = reporter()
    errors = ["CHILD_EXIT_2：数据加载失败\n请查看日志", "RESULT_READBACK:" + "坏" * 500,
              "LEASE_RELEASE：释放租约失败", "第四条不应刷屏"]
    report.finished("raw", "FAILED_RUNTIME_FINAL", {"errors": errors}, tmp_path, 20000)
    reason_lines = [line for line in stream.getvalue().splitlines() if line.startswith("失败原因 ")]
    assert len(reason_lines) == 3 and all(len(line) <= 240 for line in reason_lines)
    assert "CHILD_EXIT_2：数据加载失败 请查看日志" in reason_lines[0]
    assert "RESULT_READBACK:" in reason_lines[1] and reason_lines[1].endswith("…")
    assert "LEASE_RELEASE：释放租约失败" in reason_lines[2]
    assert "第四条" not in stream.getvalue()


def test_json_failure_reasons_are_not_truncated(tmp_path):
    report, stream = reporter(json_mode=True)
    result = {"status": "FAILED_RUNTIME_FINAL", "errors": ["错误" * 300] * 4}
    report.finished("raw", result["status"], result, tmp_path, 20000)
    assert json.loads(stream.getvalue()) == result


def test_parallel_json_does_not_expand_lane_objects(tmp_path):
    report, stream = reporter(json_mode=True)
    metadata = {"status": "REGISTERED_PARALLEL", "lanes": {
        "raw": {"gpu_ids": [0], "output": str(tmp_path / "raw" / "output")},
        "robotized": {"gpu_ids": [1], "output": str(tmp_path / "robotized" / "output")}}}
    report.registered("parallel", {"raw": [0], "robotized": [1]}, {"raw": 100, "robotized": 200},
                      tmp_path, metadata)
    assert [json.loads(line) for line in stream.getvalue().splitlines()] == [metadata]


def test_broken_console_stream_does_not_stop_worker(tmp_path):
    stream = io.StringIO()
    stream.close()
    report = ConsoleReporter(stream=stream)
    report.stage("raw", "加载")
    report.preflight("raw", {})
    report.registered("raw", [0], 100, tmp_path, {})
    report.progress("raw", tmp_path, 100)
    report.finished("raw", "FAILED_RUNTIME_FINAL", {}, tmp_path, 100)
