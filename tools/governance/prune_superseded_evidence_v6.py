from __future__ import annotations

"""Remove superseded canary/attempt references from current stage evidence.

This changes navigation only.  The removed immutable artifacts are recorded in
an evidence capsule with their original bytes/SHA before the CAS publication.
"""

import argparse
import json
import os
from pathlib import Path

from tools.governance.common import (
    AUTHORITY_PATH,
    RECEIPT_PATH,
    TASK_STATE_PATH,
    load_json,
    now_iso,
    publish_bundle,
)


ROOT = Path("/mnt/workspace/code/chaoyang")
RUN = ROOT / "tasks/control/runs/20260914_chaoyang_cleanup_v6"
DROP_PREFIXES = {
    "Contact": (
        str(ROOT / "tasks/control/runs/20260913_contact_robot_v1_canary"),
        str(ROOT / "tasks/control/runs/20260913_contact_robot_v1_canary_v2"),
    ),
    "Robot Visual": (
        str(ROOT / "tasks/control/runs/20260913_poker042_contact_retarget_fourframe_visual_v2"),
    ),
}


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(temporary, 0o444)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", action="store_true")
    args = parser.parse_args()
    authority = load_json(AUTHORITY_PATH)
    removed = []
    for stage in authority.get("stages", []):
        prefixes = DROP_PREFIXES.get(stage.get("stage"), ())
        if not prefixes:
            continue
        kept = []
        for reference in stage.get("evidence", []):
            path = str(reference.get("path", ""))
            if any(path == prefix or path.startswith(prefix + "/") for prefix in prefixes):
                removed.append({"stage": stage["stage"], **reference})
            else:
                kept.append(reference)
        stage["evidence"] = kept
    capsule = {
        "schema_version": "chaoyang-superseded-evidence-navigation-capsule-v1",
        "created_at": now_iso(),
        "status": "DRY_RUN" if not args.commit else "PENDING_CAS",
        "removed_from_current_navigation": removed,
        "claim_limit": "Navigation pruning only; original immutable results remain recoverable until their separate cleanup batch is committed.",
    }
    atomic_json(RUN / "LEGACY_EVIDENCE_CAPSULE_CURRENT_PRUNE.json", capsule)
    if not args.commit:
        print(json.dumps({"status": "DRY_RUN", "references": len(removed)}))
        return 0
    published = None
    for _ in range(30):
        receipt = load_json(RECEIPT_PATH)
        latest = load_json(AUTHORITY_PATH)
        for target in latest.get("stages", []):
            prefixes = DROP_PREFIXES.get(target.get("stage"), ())
            if prefixes:
                target["evidence"] = [
                    ref for ref in target.get("evidence", [])
                    if not any(str(ref.get("path", "")) == prefix or str(ref.get("path", "")).startswith(prefix + "/") for prefix in prefixes)
                ]
        try:
            published = publish_bundle(
                latest,
                load_json(TASK_STATE_PATH),
                event_type="SUPERSEDED_CURRENT_EVIDENCE_PRUNED_V6",
                expected_revision=int(receipt["governance_revision"]),
                generator_path=Path(__file__),
            )
            break
        except RuntimeError as error:
            if "CAS revision mismatch" not in str(error):
                raise
    if published is None:
        raise SystemExit("governance CAS remained busy")
    capsule["status"] = "PASS_CURRENT_NAVIGATION_PRUNED"
    capsule["published_governance_revision"] = published["governance_revision"]
    atomic_json(RUN / "LEGACY_EVIDENCE_CAPSULE_CURRENT_PRUNE.json", capsule)
    print(json.dumps({"status": capsule["status"], "references": len(removed), "revision": published["governance_revision"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
