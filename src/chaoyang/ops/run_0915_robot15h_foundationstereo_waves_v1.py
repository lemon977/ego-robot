#!/usr/bin/env python3
"""Run the frozen encoded-domain FoundationStereo adapter on the eight W1 sessions.

The implementation intentionally reuses the already-tested W0 algorithm body
without modifying it.  This wrapper changes only the governed task namespace,
the frozen W1 cohort, accounting, and input binding.  Final-holdout source files
cannot be opened before their immutable H9 release time.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from chaoyang.ops import run_0915_robot15h_foundationstereo_wave0_v1 as base


ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "0915_robot15h_foundationstereo_waves_v1"
PHASE = "ROBOT15H_FOUNDATIONSTEREO_W1_ENCODED_DOMAIN"
INVENTORY = ROOT / "_run/current/0915_robot15h_window_start_inventory_v1/attempts/attempt_0001/BATCH_MANIFEST.json"
MATRIX = ROOT / "_run/current/0915_robot15h_release_candidate_v1/attempts/attempt_0001/CAPABILITY_MATRIX.json"
WINDOW_CLOCK = ROOT / "_run/current/0915_robot15h_v1/WINDOW_CLOCK.json"
OUTPUT = ROOT / f"_run/current/{TASK_ID}/attempts/attempt_0001"
VISUAL = ROOT / "docs/current/visuals/0915_ROBOT15H_W1_FOUNDATIONSTEREO_V1"
TERMINAL_RECEIPT = ROOT / "tasks/receipts/0915_ROBOT15H_FOUNDATIONSTEREO_WAVES_V1_RESULT.json"
BASE_RUNNER = Path(base.__file__).resolve()


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise RuntimeError("timezone-aware W1 release time required")
    return parsed


def w1_rows() -> list[dict[str, Any]]:
    """Return SHA-bound W1 copies without mutating the frozen manifest."""
    manifest = base.load_json(INVENTORY)
    rows = [dict(row) for row in manifest.get("sessions", []) if row.get("wave") == "W1"]
    if len(rows) != 8 or sum(int(row["frame_count"]) for row in rows) != 2332:
        raise RuntimeError("frozen FoundationStereo W1 denominator drift")
    if len({row["source_group"] for row in rows}) != len(rows):
        raise RuntimeError("FoundationStereo W1 source groups are not disjoint")
    now = datetime.now().astimezone()
    clock = base.load_json(WINDOW_CLOCK)
    if now > _parse_time(str(clock["start_new_sessions_deadline_at"])):
        raise RuntimeError("H13.5 new-session cutoff has been reached")
    prepared: list[dict[str, Any]] = []
    for row in rows:
        if now < _parse_time(str(row["earliest_consumption_at"])):
            raise RuntimeError(
                f"W1 source is still sealed until {row['earliest_consumption_at']}: {row['session_id']}"
            )
        source = Path(str(row["source_stereo"]["path"])).resolve(strict=True)
        expected_bytes = int(row["source_stereo"]["bytes"])
        if source.stat().st_size != expected_bytes:
            raise RuntimeError(f"W1 source byte drift: {row['session_id']}")
        bound = dict(row)
        bound["source_stereo"] = {
            "path": str(source),
            "bytes": expected_bytes,
            "sha256": base.sha256(source),
            "sha_policy": "BOUND_AT_W1_SCHEDULING_WITHOUT_MUTATING_FROZEN_MANIFEST",
        }
        prepared.append(bound)
    return prepared


def _validate_release_scope() -> list[dict[str, Any]]:
    matrix = base.load_json(MATRIX)
    allowed = [
        dict(row)
        for row in matrix.get("rows", [])
        if row.get("capability_id") == "foundationstereo"
        and row.get("scope") == "DEVELOPMENT_DEPTH_VISUAL_OBJECT6D_INPUT_ONLY"
        and row.get("w1_expansion_allowed") is True
    ]
    tasks = {row.get("task") for row in allowed}
    if tasks != {"playing_cards", "potato_chips"}:
        raise RuntimeError("release candidate does not admit both frozen FoundationStereo W1 task scopes")
    return allowed


_original_build_signature = base.build_signature


def build_signature(
    packet_path: Path, executor_epoch: int, preflight: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_release_scope()
    signature = _original_build_signature(packet_path, executor_epoch, preflight)
    signature.pop("run_signature_sha256", None)
    frozen_rows = {str(row["session_id"]): row for row in w1_rows()}
    for session in signature["sessions"]:
        frozen = frozen_rows[str(session["session_id"])]
        session["split"] = frozen["split"]
        session["earliest_consumption_at"] = frozen["earliest_consumption_at"]
    signature["schema_version"] = "0915-robot15h-foundationstereo-w1-run-signature-v1"
    signature["runtime"]["base_w0_algorithm_runner"] = base.ref(BASE_RUNNER)
    signature["release_candidate_matrix"] = base.ref(MATRIX)
    signature["input_binding_policy"] = (
        "COPY_FROZEN_W1_ROWS_AND_BIND_SOURCE_SHA_AT_SCHEDULING; "
        "DO_NOT_MUTATE_BATCH_MANIFEST"
    )
    signature["scheduled_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    signature["run_signature_sha256"] = base.canonical_sha(signature)
    return signature


def _correct_batch() -> dict[str, Any] | None:
    path = OUTPUT / "BATCH_RESULT.json"
    if not path.is_file():
        return None
    batch = base.load_json(path)
    rows = batch.get("sessions", [])
    counts = batch.setdefault("counts", {})
    counts.update({
        "attempted": len(rows),
        "passed": sum(row.get("status") == "PASSED" for row in rows),
        "rejected_quality": sum(row.get("status") == "REJECTED_QUALITY" for row in rows),
        "failed_runtime": sum(row.get("status") == "FAILED_RUNTIME" for row in rows),
        "unrun": 0,
    })
    batch["schema_version"] = "0915-robot15h-foundationstereo-w1-batch-v1"
    batch["wave"] = "W1"
    batch["source_sha_bound_at_scheduling"] = True
    batch["final_holdout_opened_before_h9"] = False
    batch["release_candidate_matrix"] = base.ref(MATRIX)
    base.atomic_json(path, batch)
    return batch


def write_terminal(packet: Mapping[str, Any], status: str, blocker: str | None) -> None:
    batch = _correct_batch()
    signature = base.load_json(OUTPUT / "RUN_SIGNATURE.json")
    binding_path = OUTPUT / "W1_INPUT_BINDING.json"
    base.atomic_json(binding_path, {
        "schema_version": "0915-robot15h-foundationstereo-w1-input-binding-v1",
        "task_id": TASK_ID,
        "window_run_id": base.WINDOW_RUN_ID,
        "binding_policy": signature["input_binding_policy"],
        "source_manifest": base.ref(INVENTORY),
        "release_candidate_matrix": base.ref(MATRIX),
        "sessions": signature["sessions"],
        "counts": {"sessions": len(signature["sessions"]), "frames": 2332},
        "all_source_sha_bound": all(
            row.get("source_stereo", {}).get("sha256") for row in signature["sessions"]
        ),
        "frozen_manifest_mutated": False,
    })
    counts = (
        batch.get("counts")
        if batch is not None
        else {"attempted": 0, "passed": 0, "rejected_quality": 0, "failed_runtime": 8, "unrun": 0}
    )
    result = {
        "schema_version": "0915-robot15h-foundationstereo-w1-result-v1",
        "task_id": TASK_ID,
        "window_run_id": base.WINDOW_RUN_ID,
        "status": status,
        "first_blocker": blocker,
        "weights": packet["weights"],
        "counts": counts,
        "wave": "W1",
        "w1_session_count": 8,
        "w1_frame_count": 2332,
        "horizontal_reflection_for_disparity_sign": True,
        "output_domain": "ORIGINAL_PHYSICAL_LEFT_640x480",
        "lens_undistortion_applied": False,
        "external_accuracy": "UNVERIFIED",
        "strict_metric_contact_authorized": False,
        "authorized_scope": "DEVELOPMENT_DEPTH_VISUAL_OBJECT6D_INPUT_ONLY",
        "source_mutated": False,
        "training_eligible": False,
        "physical_deployment_authorized": False,
        "release_candidate_matrix": base.ref(MATRIX),
        "input_binding": base.ref(binding_path),
        "batch_result": base.ref(OUTPUT / "BATCH_RESULT.json") if batch else None,
        "gpu_command_receipt": (
            base.ref(OUTPUT / "GPU_COMMAND_RECEIPT.json")
            if (OUTPUT / "GPU_COMMAND_RECEIPT.json").is_file()
            else None
        ),
        "claim_limit": packet["claim_limit"],
    }
    base.atomic_json(OUTPUT / "RESULT.json", result)
    base.atomic_json(TERMINAL_RECEIPT, {**result, "result": base.ref(OUTPUT / "RESULT.json")})
    base.atomic_json(OUTPUT / "RUN_RECEIPT.json", {
        "schema_version": "0915-robot15h-foundationstereo-w1-run-receipt-v1",
        "task_id": TASK_ID,
        "window_run_id": base.WINDOW_RUN_ID,
        "status": status,
        "result": base.ref(OUTPUT / "RESULT.json"),
        "terminal_receipt": base.ref(TERMINAL_RECEIPT),
    })


def configure() -> None:
    # Store the immutable base path before changing ``base.__file__``.  The W0
    # body uses its module globals dynamically when it creates the worker
    # command, validates the route, and writes outputs.
    base.TASK_ID = TASK_ID
    base.PHASE = PHASE
    base.OUTPUT = OUTPUT
    base.VISUAL = VISUAL
    base.TERMINAL_RECEIPT = TERMINAL_RECEIPT
    base.w0_rows = w1_rows
    base.build_signature = build_signature
    base.write_terminal = write_terminal
    base.__file__ = str(Path(__file__).resolve())


def main() -> int:
    configure()
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
