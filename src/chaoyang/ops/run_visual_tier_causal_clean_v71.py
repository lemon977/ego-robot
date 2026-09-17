#!/usr/bin/env python3
"""Build causal real-donor Clean inputs for frozen calibration-missing Visual Tier rows.

No Depth/Object6D or ProPainter is used.  The result is Grade-B only for the
causal visual/pose-only route and cannot support metric geometry or Contact.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import signal
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PLAN = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/visual_aux/expansion_plan_R7_2/VISUAL_AUX_ROBOT_EXPANSION_PLAN.json"
DEFAULT_MATRIX = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260913_exact78_v52/matrices/CURRENT_DRAFT_EXACT78_CROSS_STAGE_MATRIX.json"
DEFAULT_OUTPUT = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/visual_aux/visual_tier_clean_R7_2"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def atomic_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def run(command: list[str], log: Path, timeout: int = 7200) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
            text=True, start_new_session=True,
        )
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=30)
            handle.write(f"\nPHASE_TIMEOUT_SECONDS={timeout}\n")
            return 124


def ensure_selection(plan_path: Path, matrix_path: Path, output: Path) -> tuple[Path, list[str]]:
    selection_path = output / "VISUAL_TIER_CAUSAL_CLEAN_SELECTION.json"
    if selection_path.is_file():
        payload = load(selection_path)
        return selection_path, [row["session_id"] for row in payload["sessions"]]
    plan, matrix = load(plan_path), load(matrix_path)
    matrix_rows = {row["session_id"]: row for row in matrix["rows"]}
    selected = [row for row in plan["selection"] if row["tier"] == "TIER_V_VISUAL"]
    rows = []
    for chosen in selected:
        session = chosen["session_id"]
        row = matrix_rows[session]
        if row.get("three_upstream_ab") is not True or row.get("metric_ready_wave0") is not False:
            raise RuntimeError(f"{session}: not a calibration-missing triple-A/B Visual Tier row")
        absence_path = output / "object6d_absent" / session / "RESULT.json"
        absence = {
            "schema_version": "visual-tier-object6d-absence-v71-v1",
            "status": "ABSENT_VISUAL_TIER_CALIBRATION_MISSING", "session_id": session,
            "task": row["task"], "metric_geometry_authorized": False,
            "claim_limit": "Explicit absence receipt; not Object6D evidence.",
        }
        if absence_path.is_file():
            if load(absence_path) != absence:
                raise RuntimeError(f"{session}: conflicting Object6D-absence receipt")
        else:
            atomic_new(absence_path, absence)
        rows.append({
            "session_id": session, "task": row["task"], "frame_count": row["frame_count"],
            "split": row["split"], "status": "PENDING_VISUAL_TIER", "existing_clean": None,
            "upstream": {
                "hawor": row["hawor"]["result"], "role_mask": row["role_mask"]["result"],
                "task_object_mask": row["object_mask"]["result"], "object6d": ref(absence_path),
            },
        })
    atomic_new(selection_path, {
        "schema_version": "exact78-visual-clean-wave1-selection-v71-v1",
        "status": "FROZEN_VISUAL_TIER_CAUSAL_DONOR_ONLY", "artifact_revision": "R7_2",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_plan": ref(plan_path), "source_matrix": ref(matrix_path), "sessions": rows,
        "claim_limit": "Visual-only Clean selection; calibration missing, no Depth/Object6D/Contact authority.",
    })
    return selection_path, [row["session_id"] for row in rows]


def publish_clean(output: Path, session: str, selection: dict[str, Any], artifact_revision: str = "R7_2") -> Path:
    final = output / "clean_results" / session / "RESULT.json"
    if final.is_file():
        return final
    row = next(item for item in selection["sessions"] if item["session_id"] == session)
    donor_result = output / "real_donor_v1" / session / "RESULT.json"
    donor = load(donor_result)
    if donor.get("status") != "TERMINAL_GRADE_B" or donor.get("consumption_authorized") is not True:
        raise RuntimeError(f"{session}: causal real donor is not Grade-B")
    manifest = output / "sessions" / session / "expanded_role_handoff" / "FRAME_MANIFEST.json"
    source_map = output / "real_donor_v1" / session / "SOURCE_MAP_MANIFEST.json"
    atomic_new(final, {
        "schema_version": "visual-tier-causal-clean-result-v71-v1",
        "artifact_revision": artifact_revision, "status": "PASS_CAUSAL_REAL_DONOR_VISUAL_TIER_GRADE_B",
        "grade": "B", "downstream_authorized": True,
        "authorized_scopes": ["POSE_ONLY_VISUAL_ROBOT_INPUT", "CAUSAL_VISUAL_AUX_INPUT"],
        "clean_tier": "TIER_V_VISUAL", "task": row["task"], "session": session,
        "frame_count": row["frame_count"], "metric_geometry": False,
        "inputs": {"mask_frame_manifest": ref(manifest), "real_donor_source_manifest": ref(source_map)},
        "artifacts": {"real_donor_result": ref(donor_result), "source_map_manifest": ref(source_map)},
        "object6d": row["upstream"]["object6d"], "control_ground_truth": False,
        "claim_limit": "Causal visual/pose-only Clean only; unsupported pixels stay Raw and later become invalid. No metric/contact/background truth.",
    })
    return final


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--session", action="append", dest="only_sessions")
    parser.add_argument("--artifact-revision", default="R7_2")
    parser.add_argument("--result-path", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    output = args.output_root.resolve(); output.mkdir(parents=True, exist_ok=True)
    selection_path, sessions = ensure_selection(args.plan.resolve(strict=True), args.matrix.resolve(strict=True), output)
    if args.only_sessions:
        requested = list(dict.fromkeys(args.only_sessions))
        unknown = sorted(set(requested) - set(sessions))
        if unknown:
            raise RuntimeError(f"requested sessions are outside frozen selection: {unknown}")
        sessions = requested
    if args.dry_run:
        print(json.dumps({"status": "DRY_RUN_READY", "sessions": sessions, "selection": ref(selection_path)}, ensure_ascii=False))
        return 0
    selection = load(selection_path)
    rows = []
    for session in sessions:
        clean_result = output / "clean_results" / session / "RESULT.json"
        if not clean_result.is_file():
            prepare_receipt = output / "preparation_receipts" / f"{session}.json"
            if not prepare_receipt.is_file():
                rc = run([sys.executable, "src/chaoyang/ops/prepare_exact78_clean_wave_session_v52.py", "--plan-root", str(output), "--selection", str(selection_path), "--session", session], output / "logs" / f"prepare_{session}.log")
                if rc:
                    rows.append({"session": session, "status": "FAILED_RUNTIME_FINAL", "phase": "PREPARE", "return_code": rc}); continue
            donor_result = output / "real_donor_v1" / session / "RESULT.json"
            if not donor_result.is_file():
                rc = run([sys.executable, "src/chaoyang/ops/run_generic_same_session_real_donor_v1.py", "--spec", str(output / "specs" / f"{session}_real_donor_input.json")], output / "logs" / f"donor_{session}.log")
                if rc:
                    rows.append({"session": session, "status": "FAILED_RUNTIME_FINAL", "phase": "DONOR", "return_code": rc}); continue
            authority = output / "real_donor_v1" / session / "INDEPENDENT_AUTHORITY.json"
            if not authority.is_file():
                rc = run([sys.executable, "src/chaoyang/ops/validate_generic_same_session_real_donor_v1.py", "--result", str(donor_result), "--authority", str(authority)], output / "logs" / f"validate_{session}.log")
                if rc:
                    rows.append({"session": session, "status": "FAILED_RUNTIME_FINAL", "phase": "VALIDATE", "return_code": rc}); continue
            clean_result = publish_clean(output, session, selection, args.artifact_revision)
        rows.append({"session": session, "status": "PASSED", "result": ref(clean_result)})
    result_path = args.result_path.resolve() if args.result_path else output / "RESULT.json"
    if not result_path.exists():
        atomic_new(result_path, {
            "schema_version": "visual-tier-causal-clean-batch-result-v71-v1",
            "status": "PASSED" if all(row["status"] == "PASSED" for row in rows) else "FAILED_RUNTIME_FINAL",
            "selection": ref(selection_path), "rows": rows,
            "artifact_revision": args.artifact_revision,
            "selected_sessions": sessions,
            "claim_limit": "Visual-only causal donor Clean; no metric geometry, Contact or background truth.",
        })
    print(json.dumps(load(result_path), ensure_ascii=False, indent=2))
    return 0 if load(result_path)["status"] == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
