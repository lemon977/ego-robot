#!/usr/bin/env python3
"""Close the R1 diagnostic attempt without changing any predecessor artifact."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TASK = "human_to_robot_completion_r1_20260922"
ATTEMPT = ROOT / "_run/current" / TASK / "attempts/attempt_0001"


def sha(path: Path) -> dict:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": h.hexdigest()}


def main() -> int:
    lanes = {}
    for state_path in sorted((ATTEMPT / "lanes").glob("*/STATE.json")):
        state = json.loads(state_path.read_text(encoding="utf-8"))
        lanes[state["lane"]] = {
            "status": state.get("status"),
            "execution": state.get("execution"),
            "integrity": state.get("integrity"),
            "quality": state.get("quality"),
            "improvement": state.get("improvement"),
            "blocker": state.get("blocker"),
            "state_ref": sha(state_path),
        }
    progress = ATTEMPT / "PROGRESS_2H.json"
    result = {
        "schema_version": "human-to-robot-completion-r1-result-v1",
        "task_id": TASK,
        "status": "TERMINAL_WITH_GAPS",
        "terminal": True,
        "terminal_reason": "R1_DIAGNOSTIC_EVIDENCE_COMPLETE_SUCCESSOR_REQUIRED",
        "execution": "PARTIAL_DIAGNOSTIC_AUDIT",
        "quality": "NOT_ADOPTED",
        "lanes": lanes,
        "progress_ref": sha(progress) if progress.is_file() else None,
        "training_eligible": False,
        "control_ground_truth": False,
        "physical_deployable": False,
        "external_metric_authority": False,
        "predecessor_v5_immutable": True,
        "successor": "human_to_robot_root_cause_gated_r2_20260923",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "claim_limit": "Diagnostic evidence only; no product adoption or physical control.",
    }
    out = ATTEMPT / "RESULT.json"
    if out.exists():
        existing = json.loads(out.read_text(encoding="utf-8"))
        if existing != result:
            raise RuntimeError(f"immutable R1 result conflict: {out}")
    else:
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "result": sha(out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
