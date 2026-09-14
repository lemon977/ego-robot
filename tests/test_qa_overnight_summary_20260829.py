from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


SOURCE = Path("/mnt/workspace/code/chaoyang/tools/qa_overnight_summary_20260829.py")
sys.path.insert(0, str(SOURCE.parent))
SPEC = importlib.util.spec_from_file_location("qa_overnight_summary_20260829", SOURCE)
assert SPEC is not None and SPEC.loader is not None
qa = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = qa
SPEC.loader.exec_module(qa)


def record(role: str, payload: bytes = b"{}"):
    path = f"/mnt/workspace/code/chaoyang/archive/legacy_runs/unclassified/AUTONOMOUS_20H_20260829/checkpoints/{role}.json"
    return qa.strict_io.VerifiedBytes(path, payload, len(payload), hashlib.sha256(payload).hexdigest(), 101, abs(hash(role)) + 1)


def ref_node(item):
    return {"path": item.path, "bytes": item.bytes, "sha256": item.sha256}


def summary_text() -> bytes:
    lines = ["# OVERNIGHT_SUMMARY_20260829"]
    for session in ("grap_a_cap_004", "grap_a_cap_002"):
        for slot in ("raw", "mask", "clean", "arm+kaihand", "four-column video"):
            lines.append(f"| {session} | {slot} | NOT_READY | MISSING | MISSING | MISSING |")
    lines.extend(
        [
            "current admitted coverage: `0/68`",
            "Coverage report: `NOT_READY`; path=`MISSING`, bytes=`MISSING`, SHA-256=`MISSING`",
            "Future aggregator remains `HOLD_FORBIDDEN` with retained A/P0=`2` and P1=`2`",
            "PROVEN=`3`, CONTRADICTED=`44`, INCOMPLETE=`84`, MISSING=`0`",
            "`goal_complete=false`",
            "BQ05 (`P1-STARTED-APPEND-ACK-AMBIGUITY-R3-FINAL`)",
            "Future coverage contract remains 2 A/P0 + 2 P1 and is not admitted.",
        ]
    )
    return ("\n".join(lines) + "\n").encode()


def final_chain():
    records = {role: record(role) for role in ("final_metrics", "final_redline", "final_pack", "finalizer_terminal")}
    refs = {role: ref_node(records[role]) for role in ("final_metrics", "final_redline", "final_pack")}
    scopes = {
        "named_ledger_scope": {"new_a_class_event": False},
        "canonical_evidence_scope": {
            "retained_a_at_window_start": {
                "owner_decision_group_ids": ["A1", "A2", "A3"],
                "bq05_state": "HOLD_ROUND3_EXHAUSTED_BQ05_DEGRADED_TO_A",
                "coverage_a_p0_count": 2,
                "coverage_p1_count": 2,
            },
            "delivery_matrix": {"readiness": {
                "future_coverage_retained_a_class_p0_count": 2,
                "future_coverage_retained_p1_count": 2,
                "grap_a_cap_004_all_columns": "NOT_READY",
                "grap_a_cap_002_all_columns": "NOT_READY",
            }},
        },
    }
    payloads = {
        "finalizer_terminal": {"status": qa.TERMINAL_STATUS, "execution_gpu_pixel_queue_admission": 0, "result": refs},
        "final_artifacts_qa": {"subject": ref_node(records["finalizer_terminal"]), "execution_gpu_pixel_queue_admission": 0},
        "final_pack": {"completion": qa.EXPECTED_COMPLETION.copy(), "scope_separation": scopes, "execution_gpu_pixel_queue_admission": 0},
        "final_redline": {"scope_separation": scopes, "execution_gpu_pixel_queue_admission": 0},
        "final_metrics": {"execution_gpu_pixel_queue_admission": 0},
    }
    return payloads, records


def pointer_fixture(start: int):
    pointer_record = record("pre2000_pointer")
    pointer = {
        "schema_version": "OWNER_AND_TASK_DOC_PRE_2000_PAYLOAD_PRESERVATION_T0_V1",
        "created_at": "2026-08-29T19:40:00+08:00",
        "cpu11_single_wall_timer": {
            "clock": "CLOCK_MONOTONIC",
            "budget_seconds": 1200,
            "start_before_step": "PRE_2000_FOUR_DOCUMENT_SNAPSHOT_CAPTURE",
            "start_monotonic_ns": start,
            "timer_restart_forbidden": True,
        },
    }
    read = qa.ReadResult(b"{}", pointer_record, 0o444, 1)
    records = {role: record(role) for role in qa.LIVE_DOC_PATHS}
    refs = [ref_node(pointer_record), *(ref_node(item) for item in records.values())]
    payloads = {
        "document_sync_result": {"refs": refs, "execution_gpu_pixel_queue_admission": 0},
        "document_sync_result_qa": {"refs": refs, "execution_gpu_pixel_queue_admission": 0},
    }
    return pointer, read, payloads, records


class OvernightSummaryQATests(unittest.TestCase):
    def test_exact_frozen_builder_loads_with_dataclasses(self):
        source = qa._read_same_fd(qa.FIXED["builder_source"])
        namespace = qa._load_builder(source)
        self.assertIn("_render_summary", namespace)

    def test_exact_ten_negative_rows_pass(self):
        payload = summary_text()
        qa._validate_summary(payload, payload, {})

    def test_ready_row_rejected(self):
        payload = summary_text().replace(b"| NOT_READY | MISSING", b"| READY | /fake/path", 1)
        with self.assertRaises(qa.QAError):
            qa._validate_summary(payload, payload, {})

    def test_fake_delivery_path_rejected(self):
        payload = summary_text().replace(b"| MISSING | MISSING | MISSING |", b"| /tmp/fake.mp4 | 1 | deadbeef |", 1)
        with self.assertRaises(qa.QAError):
            qa._validate_summary(payload, payload, {})

    def test_summary_byte_drift_rejected(self):
        with self.assertRaises(qa.QAError):
            qa._validate_summary(summary_text() + b"drift", summary_text(), {})

    def test_goal_true_rejected(self):
        payloads, records = final_chain()
        payloads["final_pack"]["completion"] = {**qa.EXPECTED_COMPLETION, "goal_complete": True}
        with self.assertRaises(qa.QAError):
            qa._validate_final_chain(payloads, records)

    def test_131_count_drift_rejected(self):
        payloads, records = final_chain()
        payloads["final_pack"]["completion"] = {**qa.EXPECTED_COMPLETION, "requirement_count": 130}
        with self.assertRaises(qa.QAError):
            qa._validate_final_chain(payloads, records)

    def test_nonzero_admission_rejected(self):
        with self.assertRaises(qa.QAError):
            qa._require_zero_admission({"execution_gpu_pixel_queue_admission": 1}, "synthetic")

    def test_nested_nonzero_admission_rejected(self):
        with self.assertRaises(qa.QAError):
            qa._require_zero_admission({"admission": {"gpu_queue": 1}}, "synthetic")

    def test_terminal_ref_drift_rejected(self):
        payloads, records = final_chain()
        payloads["finalizer_terminal"]["result"]["final_pack"]["sha256"] = "0" * 64
        with self.assertRaises(qa.QAError):
            qa._validate_final_chain(payloads, records)

    def test_terminal_result_decoy_refs_rejected(self):
        payloads, records = final_chain()
        payloads["finalizer_terminal"]["decoy_refs"] = payloads["finalizer_terminal"]["result"]
        payloads["finalizer_terminal"]["result"] = {}
        with self.assertRaises(qa.QAError):
            qa._validate_final_chain(payloads, records)

    def test_cross_scope_drift_rejected(self):
        payloads, records = final_chain()
        payloads["final_redline"]["scope_separation"] = {"named_ledger_scope": {}, "canonical_evidence_scope": {}}
        with self.assertRaises(qa.QAError):
            qa._validate_final_chain(payloads, records)

    def test_timer_exact_budget_passes(self):
        start = 100
        pointer, read, payloads, records = pointer_fixture(start)
        elapsed = qa._validate_pointer(pointer, read, start, start + qa.TIMER_BUDGET_NS, payloads, records)
        self.assertEqual(elapsed, qa.TIMER_BUDGET_NS)

    def test_timer_crossing_rejected(self):
        start = 100
        pointer, read, payloads, records = pointer_fixture(start)
        with self.assertRaises(qa.QAError):
            qa._validate_pointer(pointer, read, start, start + qa.TIMER_BUDGET_NS + 1, payloads, records)

    def test_symlink_input_rejected(self):
        with tempfile.TemporaryDirectory(dir="/mnt/workspace/code/chaoyang") as directory:
            root = Path(directory)
            target = root / "target.json"
            target.write_bytes(b"{}")
            alias = root / "alias.json"
            alias.symlink_to(target)
            expected = qa.Expected("synthetic", alias, 2, hashlib.sha256(b"{}").hexdigest())
            with self.assertRaises(OSError):
                qa._read_same_fd(expected)

    def test_hardlink_input_rejected(self):
        with tempfile.TemporaryDirectory(dir="/mnt/workspace/code/chaoyang") as directory:
            root = Path(directory)
            source = root / "source.json"
            source.write_bytes(b"{}")
            alias = root / "alias.json"
            alias.hardlink_to(source)
            expected = qa.Expected("synthetic", source, 2, hashlib.sha256(b"{}").hexdigest())
            with self.assertRaises(qa.QAError):
                qa._read_same_fd(expected)

    def test_existing_output_rejected_without_mutation(self):
        with tempfile.TemporaryDirectory(dir="/mnt/workspace/code/chaoyang") as directory:
            path = Path(directory) / "qa.json"
            path.write_bytes(b"sentinel")
            with self.assertRaises(FileExistsError):
                qa._output_absent(path)
            self.assertEqual(path.read_bytes(), b"sentinel")

    def test_noncanonical_summary_path_rejected(self):
        digest = "0" * 64
        refs = {role: qa.Expected(role, qa.PROJECT_ROOT / f"{role}.json", 0, digest) for role in qa.RUNTIME_ROLES}
        refs["summary"] = qa.Expected("summary", qa.PROJECT_ROOT / "archive/audits/fake.md", 0, digest)
        with self.assertRaises(qa.QAError):
            qa._validate_paths(refs)

    def test_conflicting_pass_qa_rejected(self):
        with self.assertRaises(qa.QAError):
            qa._require_exact_pass(
                {"overall_verdict": {"verdict": "PASS_EXPECTED"}, "status": "HOLD"},
                "PASS_EXPECTED",
                "synthetic",
            )


if __name__ == "__main__":
    unittest.main()
