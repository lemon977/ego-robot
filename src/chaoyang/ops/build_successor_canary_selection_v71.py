#!/usr/bin/env python3
"""Freeze deterministic failure canaries and A/B regressions for V7.1."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/conversion/CONVERSION_CAUSE_LEDGER_V2.json"
OUT = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/successors/SUCCESSOR_CANARY_SELECTION_V71.json"
STAGES = {
    "hawor": "HAWOR_C",
    "role_mask": "ROLE_AFTER_HAWOR",
    "object_identity": "OBJECT_AFTER_HAWOR_ROLE",
}


def main() -> None:
    source_bytes = SOURCE.read_bytes()
    rows = json.loads(source_bytes)["rows"]
    groups = []
    for stage, blocker in STAGES.items():
        tasks = sorted({row["task"] for row in rows if row["first_blocker"] == blocker})
        for task in tasks:
            failures = sorted((row for row in rows if row["first_blocker"] == blocker and row["task"] == task), key=lambda row: row["position"])
            canary = failures[0]
            same_contract = [
                row for row in rows
                if row["first_blocker"] == "METRIC_READY" and row["task"] == task
                and row["acquisition_contract"] == canary["acquisition_contract"]
            ]
            pool = same_contract or [row for row in rows if row["first_blocker"] == "METRIC_READY" and row["task"] == task]
            regressions = sorted(pool, key=lambda row: row["position"])[:2]
            if len(regressions) != 2:
                raise RuntimeError(f"two regressions unavailable for {stage}/{task}")
            groups.append({
                "stage": stage,
                "task": task,
                "failure_cluster_size": len(failures),
                "canary_session": canary["session_id"],
                "canary_split": canary["split"],
                "acquisition_contract": canary["acquisition_contract"],
                "regression_sessions": [row["session_id"] for row in regressions],
                "round_budget": {"max_rounds": 2, "wall_seconds_per_round": 14400, "gpu_seconds_per_round": 7200},
                "expansion_gate": "CANARY_PASS_AND_BOTH_REGRESSIONS_NO_DEGRADATION",
            })
    value = {
        "schema_version": "SUCCESSOR_CANARY_SELECTION_V71",
        "plan_revision": "chaoyang-v7.1",
        "artifact_revision": "R7_1",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "source": {"path": str(SOURCE), "bytes": len(source_bytes), "sha256": hashlib.sha256(source_bytes).hexdigest()},
        "groups": groups,
        "counts": {"groups": len(groups), "failure_canaries": len(groups), "regression_slots": 2 * len(groups)},
        "immutable": True,
        "claim_limit": "Selection and bounded budget only; no successor has passed until immutable run receipts and regressions exist.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(value["counts"]))


if __name__ == "__main__":
    main()
