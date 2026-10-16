"""Read-only, compact terminal reporting for the finite joint26 launchers.

The training worker owns its JSONL, CSV and signed results.  This helper only
observes a bounded tail of those files; a console failure never stops a worker.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time


_TAIL_BYTES = 64 * 1024
_MISSING = "—"


def _label(branch):
    value = "" if branch is None else str(branch)
    return {
        "raw": "Raw（原始）", "原始": "Raw（原始）", "原始分支": "Raw（原始）",
        "robotized": "Robotized（机器人化）", "机器人化": "Robotized（机器人化）",
        "机器人化分支": "Robotized（机器人化）",
        "parallel": "并行协调", "并行": "并行协调", "prepare": "数据准备",
    }.get(value.casefold(), value or "任务")


def _finite(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _metric(value):
    return f"{value:.6g}" if _finite(value) else _MISSING


def _count(value):
    return str(value) if type(value) is int and value >= 0 else _MISSING


def _step(value, target):
    return (value if type(target) is int and target > 0 and type(value) is int
            and 0 <= value <= target else None)


def _last_update(output_dir):
    """Find the last complete JSON object using at most 64 KiB of file data."""
    try:
        with (Path(output_dir) / "updates.jsonl").open("rb") as stream:
            size = stream.seek(0, 2)
            offset = max(0, size - _TAIL_BYTES)
            stream.seek(offset)
            tail = stream.read(_TAIL_BYTES)
        if offset:
            # The first fragment may begin in the middle of a JSON record.
            first_newline = tail.find(b"\n")
            if first_newline < 0:
                return None
            tail = tail[first_newline + 1:]
        complete = tail.rsplit(b"\n", 1)[0] if b"\n" in tail else b""
        for line in reversed(complete.split(b"\n")):
            try:
                row = json.loads(line)
            except (ValueError, UnicodeError, RecursionError):
                continue
            if isinstance(row, dict):
                return row
    except (OSError, ValueError, TypeError):
        pass
    return None


def _training_result(output_dir):
    try:
        with (Path(output_dir) / "TRAINING_RESULT.json").open("rb") as stream:
            content = stream.read(_TAIL_BYTES + 1)
        if len(content) <= _TAIL_BYTES:
            value = json.loads(content)
            if isinstance(value, dict):
                return value
    except (OSError, ValueError, TypeError, UnicodeError, RecursionError):
        pass
    return {}


class ConsoleReporter:
    def __init__(self, json_mode=False, stream=None, wait_interval=15.0):
        self.json_mode = json_mode
        self.stream = sys.stdout if stream is None else stream
        self.wait_interval = wait_interval
        self._updates = {}
        self._waits = {}

    def _write(self, text):
        try:
            print(text, file=self.stream, flush=True)
        except (OSError, ValueError):
            pass

    def _json(self, document):
        try:
            self._write(json.dumps(document, ensure_ascii=False))
        except (TypeError, ValueError):
            pass

    def stage(self, branch, text):
        if not self.json_mode:
            self._write(f"[{_label(branch)}] {text}")

    def preflight(self, branch, qualification):
        if self.json_mode:
            self._json(qualification)
            return
        row = qualification if isinstance(qualification, dict) else {}
        self._write(
            f"[{_label(branch)}] 预检步骤完成："
            f"训练 {_count(row.get('training_samples'))} 样本／{_count(row.get('training_groups'))} 组，"
            f"验证 {_count(row.get('validation_samples'))} 样本／{_count(row.get('validation_groups'))} 组。")

    def registered(self, branch, gpu_ids, target, output, metadata):
        if self.json_mode:
            self._json(metadata)
            return
        if isinstance(gpu_ids, dict):
            lanes = metadata.get("lanes", {}) if isinstance(metadata, dict) else {}
            for lane, ids in gpu_ids.items():
                budget = target.get(lane) if isinstance(target, dict) else target
                lane_output = lanes.get(lane, {}).get("output", output)
                self.registered(lane, ids, budget, lane_output, {})
            return
        cards = ",".join(str(card) for card in gpu_ids) if gpu_ids else "CPU"
        try:
            log = str(Path(output).parent / "EXECUTION.log")
        except (TypeError, ValueError):
            log = _MISSING
        self._write(f"[{_label(branch)}] 已登记：卡 {cards}，目标 {_count(target)} 步；日志 {log}")

    def progress(self, branch, output_dir, target, final=False):
        if self.json_mode:
            return
        row = _last_update(output_dir)
        update = _step(row.get("optimizer_update"), target) if row is not None else None
        if update is None:
            now = time.monotonic()
            previous = self._waits.get(branch)
            if not final and (previous is None or now - previous >= self.wait_interval):
                self._write(f"[{_label(branch)}] 等待加载／首条训练日志，尚无可报告的训练步数。")
                self._waits[branch] = now
            return
        if update <= self._updates.get(branch, -1):
            return
        self._updates[branch] = update
        duration = row.get("steady_step_seconds")
        if not _finite(duration) or duration < 0:
            duration = row.get("step_seconds")
        seconds = f"{duration:.3f}秒" if _finite(duration) and duration >= 0 else _MISSING
        remaining = row.get("remaining_hours")
        eta = f"{remaining:.2f}小时" if _finite(remaining) and remaining >= 0 else _MISSING
        self._write(
            f"[{_label(branch)}] {update}/{target} 步（{update / target:.1%}），"
            f"训练损失={_metric(row.get('flow_loss'))}，"
            f"验证损失={_metric(row.get('validation_flow_loss'))}，步时 {seconds}，预计剩余 {eta}")

    def finished(self, branch, status, result, output_dir, target, stage=None, governance_exit=None):
        if self.json_mode:
            self._json(result)
            return
        stage = stage if isinstance(stage, dict) else _training_result(output_dir)
        row = _last_update(output_dir) or {}
        completed = (_step(stage.get("optimizer_updates"), target) if "optimizer_updates" in stage
                     else _step(row.get("optimizer_update"), target))
        successful = status in {"PASSED", "PASS", "COMPLETED", "SUCCESS", "SUCCEEDED"}
        if governance_exit is not None and governance_exit != 0:
            successful = False
        outcome = "执行完成" if successful else "执行失败"
        text = f"[{_label(branch)}] {outcome}"
        checkpoint = loss_path = None
        if type(target) is int and target > 0:
            text += f"：实际完成 {_count(completed)}/{target} 步"
            val_loss = stage.get("validation_flow_loss", row.get("validation_flow_loss"))
            text += f"，训练损失={_metric(row.get('flow_loss'))}，验证损失={_metric(val_loss)}"
            checkpoint = stage.get("last_checkpoint")
            checkpoint = checkpoint if isinstance(checkpoint, str) and checkpoint else _MISSING
            try:
                csv = Path(output_dir) / "loss.csv"
                loss_path = str(csv) if csv.is_file() else _MISSING
            except (OSError, TypeError, ValueError):
                loss_path = _MISSING
        self._write(text + "。")
        if checkpoint is not None:
            self._write(f"权重：{checkpoint}")
            self._write(f"损失记录：{loss_path}")
        if not successful:
            errors = result.get("errors", []) if isinstance(result, dict) else []
            if isinstance(errors, str):
                errors = [errors]
            if isinstance(errors, (list, tuple)):
                for index, error in enumerate(errors[:3], start=1):
                    prefix = f"失败原因 {index}："
                    reason = " ".join(str(error).split())
                    limit = 240 - len(prefix)
                    if len(reason) > limit:
                        reason = reason[:limit - 1] + "…"
                    self._write(prefix + reason)
