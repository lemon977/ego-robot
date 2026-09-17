#!/usr/bin/env python3
"""Fail-closed content and integrity audit for one tactile capture session."""

from __future__ import annotations

import argparse
from array import array
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any
import zlib


SIDES = ("left", "right")
VALUE_COUNT = 369
FRAME_SCHEMA = "pico_tactile_wire_frame_v1"
META_SCHEMA = "pico_tactile_raw_meta_v1"
POLICY = {
    "schema_version": "tactile-content-quality-policy-v1",
    "session_signal_scope": "AT_LEAST_ONE_SIDE",
    "min_signal_frames": 24,
    "min_nonzero_channels": 2,
    "min_peak_abs_value": 2,
    "single_inactive_side": "WARNING_NOT_REJECTION",
    "transient_capture_warning": "WARNING_WHEN_CAPTURE_VALID",
    "saturated_int16": "REJECT",
    "missing_or_malformed_stream": "REJECT",
    "claim_limit": "Raw integer activity/integrity only; not force or contact truth.",
}


class TactileQualityError(RuntimeError):
    pass


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True,
                      allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _failure(failures: list[dict[str, Any]], code: str, detail: Any) -> None:
    failures.append({"code": code, "detail": detail})


def analyze_session(source: Path) -> dict[str, Any]:
    source = Path(source).resolve(strict=True)
    tactile_path = source / "raw" / "tactile.jsonl"
    meta_path = source / "raw" / "tactile.meta.json"
    failures: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    if not tactile_path.is_file():
        _failure(failures, "TACTILE_FILE_MISSING", str(tactile_path))
    if not meta_path.is_file():
        _failure(failures, "TACTILE_META_MISSING", str(meta_path))
    if failures:
        return {
            "schema_version": "tactile-session-quality-v1",
            "source": str(source), "status": "REJECTED_TACTILE_QUALITY",
            "policy": POLICY, "failures": failures, "warnings": warnings,
        }

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        _failure(failures, "TACTILE_META_UNREADABLE", repr(exc))
        meta = {}

    summary = meta.get("summary") if isinstance(meta.get("summary"), dict) else {}
    if meta.get("schema") != META_SCHEMA:
        _failure(failures, "TACTILE_META_SCHEMA_INVALID", meta.get("schema"))
    for key, expected in (("state", "complete"), ("complete", True),
                          ("capture_valid", True), ("frame_schema", FRAME_SCHEMA),
                          ("values_semantics", "wire_active_order_unmasked")):
        if meta.get(key) != expected:
            _failure(failures, f"TACTILE_META_{key.upper()}_INVALID",
                     {"actual": meta.get(key), "expected": expected})
    if summary.get("complete") is not True:
        _failure(failures, "TACTILE_SUMMARY_INCOMPLETE", summary.get("complete"))
    for key in ("writer_errors", "subscription_gap_frames"):
        if summary.get(key) != 0:
            _failure(failures, f"TACTILE_{key.upper()}", summary.get(key))
    quality_status = summary.get("quality_status")
    if quality_status not in ("ok", "warning"):
        _failure(failures, "TACTILE_META_QUALITY_INVALID", quality_status)
    elif quality_status == "warning":
        warnings.append({
            "code": "TACTILE_CAPTURE_TRANSIENT_WARNING",
            "incident_count": int(((summary.get("transient_health_warnings") or {})
                                    .get("incident_count")) or 0),
        })

    streams = meta.get("streams") if isinstance(meta.get("streams"), dict) else {}
    expected_stream_ids: dict[str, str] = {}
    for side in SIDES:
        stream = streams.get(side) if isinstance(streams.get(side), dict) else {}
        layout = stream.get("wire_layout") if isinstance(stream.get("wire_layout"), dict) else {}
        stream_id = stream.get("stream_id")
        if not isinstance(stream_id, str) or not stream_id:
            _failure(failures, f"TACTILE_{side.upper()}_STREAM_ID_MISSING", stream_id)
        else:
            expected_stream_ids[side] = stream_id
        expected_layout = {
            "rows": 24, "cols": 16, "active_count": VALUE_COUNT,
            "wire_payload_encoding": "signed_int16_le",
        }
        mismatch = {key: {"actual": layout.get(key), "expected": value}
                    for key, value in expected_layout.items() if layout.get(key) != value}
        if mismatch:
            _failure(failures, f"TACTILE_{side.upper()}_WIRE_LAYOUT_INVALID", mismatch)

    metrics: dict[str, dict[str, Any]] = {}
    state: dict[str, dict[str, Any]] = {}
    for side in SIDES:
        state[side] = {
            "frames": 0, "zero_frames": 0, "changed_transitions": 0,
            "unique_frames": set(), "nonzero_channels": set(),
            "minimum": 32768, "maximum": -32769, "peak_abs": 0,
            "saturated_values": 0, "previous_crc": None,
            "previous_stream_seq": None, "previous_wall_ns": None,
            "previous_qpc_ns": None,
        }
    previous_record_seq = None
    records_total = 0
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with tactile_path.open("rb") as stream:
            for line_number, raw_line in enumerate(stream, 1):
                digest.update(raw_line)
                byte_count += len(raw_line)
                if not raw_line.strip():
                    continue
                try:
                    row = json.loads(raw_line.decode("utf-8"))
                except Exception as exc:  # noqa: BLE001
                    _failure(failures, "TACTILE_JSON_INVALID",
                             {"line": line_number, "error": repr(exc)})
                    continue
                side = row.get("side")
                if row.get("schema") != FRAME_SCHEMA or row.get("type") != "tactile_frame":
                    _failure(failures, "TACTILE_FRAME_SCHEMA_INVALID", line_number)
                    continue
                if side not in SIDES:
                    _failure(failures, "TACTILE_SIDE_INVALID",
                             {"line": line_number, "side": side})
                    continue
                current = state[side]
                current["frames"] += 1
                records_total += 1
                if row.get("stream_id") != expected_stream_ids.get(side):
                    _failure(failures, "TACTILE_STREAM_ID_MISMATCH",
                             {"line": line_number, "side": side})
                record_seq = row.get("record_seq")
                stream_seq = row.get("stream_seq")
                wall_ns = row.get("recv_wall_ns")
                qpc_ns = row.get("recv_qpc_ns")
                integer_fields = {"record_seq": record_seq, "stream_seq": stream_seq,
                                  "recv_wall_ns": wall_ns, "recv_qpc_ns": qpc_ns}
                if any(isinstance(value, bool) or not isinstance(value, int)
                       for value in integer_fields.values()):
                    _failure(failures, "TACTILE_FRAME_INTEGER_FIELD_INVALID",
                             {"line": line_number, "values": integer_fields})
                    continue
                if previous_record_seq is not None and record_seq <= previous_record_seq:
                    _failure(failures, "TACTILE_RECORD_SEQ_NOT_INCREASING", line_number)
                for field, value in (("stream_seq", stream_seq),
                                     ("wall_ns", wall_ns), ("qpc_ns", qpc_ns)):
                    previous = current[f"previous_{field}"]
                    if previous is not None and value <= previous:
                        _failure(failures, f"TACTILE_{side.upper()}_{field.upper()}_NOT_INCREASING",
                                 line_number)
                    current[f"previous_{field}"] = value
                previous_record_seq = record_seq
                values = row.get("wire_values")
                if (not isinstance(values, list) or len(values) != VALUE_COUNT or
                        any(isinstance(value, bool) or not isinstance(value, int) or
                            value < -32768 or value > 32767 for value in values)):
                    _failure(failures, "TACTILE_VALUES_INVALID", line_number)
                    continue
                minimum, maximum = min(values), max(values)
                current["minimum"] = min(current["minimum"], minimum)
                current["maximum"] = max(current["maximum"], maximum)
                current["peak_abs"] = max(current["peak_abs"], abs(minimum), abs(maximum))
                nonzero = [index for index, value in enumerate(values) if value]
                if not nonzero:
                    current["zero_frames"] += 1
                current["nonzero_channels"].update(nonzero)
                current["saturated_values"] += sum(value in (-32768, 32767)
                                                   for value in values)
                checksum = zlib.crc32(array("h", values).tobytes())
                current["unique_frames"].add(checksum)
                if (current["previous_crc"] is not None and
                        checksum != current["previous_crc"]):
                    current["changed_transitions"] += 1
                current["previous_crc"] = checksum
    except Exception as exc:  # noqa: BLE001
        _failure(failures, "TACTILE_FILE_UNREADABLE", repr(exc))

    for side in SIDES:
        current = state[side]
        frames = current["frames"]
        metrics[side] = {
            "frames": frames,
            "zero_frames": current["zero_frames"],
            "signal_frames": frames - current["zero_frames"],
            "zero_frame_fraction": current["zero_frames"] / frames if frames else 1.0,
            "unique_frame_count": len(current["unique_frames"]),
            "changed_transition_count": current["changed_transitions"],
            "nonzero_channel_count": len(current["nonzero_channels"]),
            "minimum": current["minimum"] if frames else None,
            "maximum": current["maximum"] if frames else None,
            "peak_abs_value": current["peak_abs"],
            "saturated_value_count": current["saturated_values"],
        }
        if not frames:
            _failure(failures, f"TACTILE_{side.upper()}_STREAM_EMPTY", None)
        if current["saturated_values"]:
            _failure(failures, f"TACTILE_{side.upper()}_SATURATED_INT16",
                     current["saturated_values"])
        if frames and current["zero_frames"] == frames:
            warnings.append({"code": f"TACTILE_{side.upper()}_ALL_ZERO"})
        elif frames and len(current["unique_frames"]) <= 1:
            _failure(failures, f"TACTILE_{side.upper()}_FROZEN", None)

    expected_frames = summary.get("frames_by_side") or {}
    for side in SIDES:
        if expected_frames.get(side) != metrics[side]["frames"]:
            _failure(failures, f"TACTILE_{side.upper()}_FRAME_COUNT_MISMATCH",
                     {"actual": metrics[side]["frames"],
                      "expected": expected_frames.get(side)})
    for key in ("records_total", "durable_records"):
        if summary.get(key) != records_total:
            _failure(failures, f"TACTILE_{key.upper()}_MISMATCH",
                     {"actual": records_total, "expected": summary.get(key)})
    if summary.get("data_bytes") != byte_count:
        _failure(failures, "TACTILE_DATA_BYTES_MISMATCH",
                 {"actual": byte_count, "expected": summary.get("data_bytes")})
    if summary.get("data_sha256") != digest.hexdigest():
        _failure(failures, "TACTILE_DATA_SHA256_MISMATCH",
                 {"actual": digest.hexdigest(), "expected": summary.get("data_sha256")})

    session_signal_frames = sum(metrics[side]["signal_frames"] for side in SIDES)
    session_nonzero_channels = sum(metrics[side]["nonzero_channel_count"] for side in SIDES)
    session_peak = max(metrics[side]["peak_abs_value"] for side in SIDES)
    signal_metrics = {
        "signal_frames": session_signal_frames,
        "nonzero_channel_count_side_distinct": session_nonzero_channels,
        "peak_abs_value": session_peak,
    }
    if session_signal_frames < POLICY["min_signal_frames"]:
        _failure(failures, "TACTILE_SIGNAL_FRAMES_BELOW_MINIMUM", signal_metrics)
    if session_nonzero_channels < POLICY["min_nonzero_channels"]:
        _failure(failures, "TACTILE_NONZERO_CHANNELS_BELOW_MINIMUM", signal_metrics)
    if session_peak < POLICY["min_peak_abs_value"]:
        _failure(failures, "TACTILE_PEAK_BELOW_MINIMUM", signal_metrics)

    status = "PASSED_TACTILE_QUALITY" if not failures else "REJECTED_TACTILE_QUALITY"
    return {
        "schema_version": "tactile-session-quality-v1",
        "source": str(source), "status": status, "policy": POLICY,
        "metadata": {
            "state": meta.get("state"), "complete": meta.get("complete"),
            "capture_valid": meta.get("capture_valid"),
            "quality_status": quality_status,
        },
        "metrics": {"by_side": metrics, "session_signal": signal_metrics,
                    "records_total": records_total, "bytes": byte_count,
                    "sha256": digest.hexdigest()},
        "warnings": warnings, "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = analyze_session(args.source)
    if args.report:
        _atomic_json(args.report, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASSED_TACTILE_QUALITY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
