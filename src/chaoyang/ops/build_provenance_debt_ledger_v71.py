from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


"""Publish V7.1 provenance debts as local, finite terminals."""

import json
from pathlib import Path

from chaoyang.governance.common import artifact_ref
from chaoyang.governance.v52_contracts import atomic_write_new


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/provenance_debt"
REGISTRY = ROOT / "docs/governance/CURRENT_BASELINE_REGISTRY_V2.json"


def main() -> int:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    hawor = next(item for item in registry["entries"] if item["stage"] == "HaWoR")
    policy = next(item for item in registry["entries"] if item["stage"] == "HumanEgo Policy")
    debts = [
        {
            "debt_id": "HAWOR_WEIGHT_SHA",
            "status": "UNKNOWN_VERIFICATION_REQUIRED",
            "scope": ["HaWoR reproducibility", "HaWoR-derived Robot candidates"],
            "does_not_block": ["Clean R7_0", "Handle H0/H2/H3", "visual-only processing"],
            "evidence": hawor["weights"],
            "resolution": "Locate the exact current weight file and verify bytes/SHA against an immutable producer receipt.",
        },
        {
            "debt_id": "KAIHAND_ADAPTER_TCP_INSTALL_CALIBRATION",
            "status": "BLOCKED_EXTERNAL",
            "scope": ["PHYSICAL_DEPLOYMENT_AUTHORITY"],
            "does_not_block": ["Robot Geometry", "pose-only visualization", "Visual Aux"],
            "evidence": "ABSENT",
            "resolution": "Provide measured adapter CAD, TCP/install and world/camera-to-base calibration.",
        },
        {
            "debt_id": "REAL_ROBOT_ACTION",
            "status": "BLOCKED_EXTERNAL",
            "scope": ["HumanEgo Policy"],
            "does_not_block": ["future-2D Visual Aux checkpoints"],
            "evidence": policy["weights"],
            "resolution": "Provide synchronized real_robot_action_sidecar data under the policy contract.",
        },
        {
            "debt_id": "OCCLUSION_GOLDSET",
            "status": "BLOCKED_PREREQ",
            "scope": ["Gold occlusion accuracy"],
            "does_not_block": ["Occlusion Silver", "Robot Geometry", "offline diagnostic visualization"],
            "evidence": "NO_FROZEN_INDEPENDENT_LABELS",
            "resolution": "Freeze and independently review 120 frames per task before scoring accuracy.",
        },
    ]
    ledger = {
        "schema_version": "chaoyang-provenance-debt-ledger-v71",
        "status": "CLOSED_WITH_LOCAL_DEBTS",
        "task_id": "g0_provenance_debt_v71",
        "registry": artifact_ref(REGISTRY),
        "debts": debts,
        "global_authority_blocked": False,
        "claim_limit": "Inventory closure only. Each debt restricts only its declared scope and remains unresolved.",
    }
    payload = json.dumps(ledger, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"
    atomic_write_new(OUTPUT / "PROVENANCE_DEBT_LEDGER.json", payload)
    result = {
        "schema_version": "chaoyang-provenance-debt-result-v71",
        "status": "PASSED",
        "task_id": "g0_provenance_debt_v71",
        "local_debt_count": len(debts),
        "ledger": artifact_ref(OUTPUT / "PROVENANCE_DEBT_LEDGER.json"),
        "claim_limit": ledger["claim_limit"],
    }
    atomic_write_new(
        OUTPUT / "RESULT.json",
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n",
    )
    print(OUTPUT / "RESULT.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
