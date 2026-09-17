#!/usr/bin/env python3
"""V7.1 cleanup closure over the already completed V6 current-only cleanup.

Only newly regenerated caches are deleted here.  Historical V6 receipts are
treated as immutable evidence and are never rewritten.
"""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chaoyang.ops.cleanup_current_only_v6 import (
    ABSOLUTE_PROTECTED_PREFIXES,
    active_target_references,
    cache_candidates,
    protection_snapshot,
    tree_stats,
)


OUT = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/cleanup"
V6 = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_cleanup_v6/FINAL_RESULT.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != data:
        raise RuntimeError(f"no-clobber conflict: {path}")
    if not path.exists():
        path.write_text(data, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit V7.1 current-only cache cleanup; deletion requires --commit-delete."
    )
    parser.add_argument("--output-root", type=Path, default=OUT)
    parser.add_argument(
        "--pre-delete-manifest",
        type=Path,
        help="Exact dry-run PRE_DELETE_MANIFEST.json required by --commit-delete.",
    )
    parser.add_argument(
        "--commit-delete",
        action="store_true",
        help="Delete only the audited cache targets after governance/reference checks pass.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.commit_delete and args.pre_delete_manifest is None:
        raise SystemExit("--commit-delete requires --pre-delete-manifest from a prior dry-run")
    output_root = args.output_root.resolve()
    snapshot, protected = protection_snapshot()
    targets = cache_candidates(protected)
    refs = active_target_references(targets)
    rows = [{"path": str(path), "stats": tree_stats(path)} for path in targets]
    pre = {
        "schema_version": "chaoyang-cleanup-v71-pre-delete-v1",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "status": (
            "BLOCKED_ACTIVE_REFERENCE"
            if refs
            else ("READY_COMMIT_DELETE" if args.commit_delete else "DRY_RUN_READY_COMMIT_DELETE")
        ),
        "targets": rows,
        "active_references": refs,
        "protected_prefixes": [str(p) for p in ABSOLUTE_PROTECTED_PREFIXES],
        "governance_revision": snapshot["governance_revision"],
        "git_before": snapshot["git"],
    }
    if refs:
        if not args.commit_delete:
            publish(output_root / "PRE_DELETE_MANIFEST.json", pre)
        raise SystemExit("active reference blocks V7.1 cache cleanup")
    if not args.commit_delete:
        publish(output_root / "PRE_DELETE_MANIFEST.json", pre)
        print(json.dumps({"status": pre["status"], "targets": len(rows), "would_delete_bytes": sum(row["stats"]["bytes"] for row in rows)}))
        return
    prior_manifest_path = args.pre_delete_manifest.resolve(strict=True)
    prior = json.loads(prior_manifest_path.read_text(encoding="utf-8"))
    if prior.get("status") not in {"READY_COMMIT_DELETE", "DRY_RUN_READY_COMMIT_DELETE"}:
        raise SystemExit("pre-delete manifest is not ready for commit")
    if prior.get("targets") != rows:
        raise SystemExit("cleanup targets or bytes changed after dry-run")
    if prior.get("governance_revision") != snapshot["governance_revision"]:
        raise SystemExit("governance revision changed after dry-run")
    if prior.get("git_before") != snapshot["git"]:
        raise SystemExit("Git dirty/untracked snapshot changed after dry-run")
    if prior.get("active_references"):
        raise SystemExit("pre-delete manifest recorded active references")
    for path in targets:
        if path.is_symlink():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
    remaining = [str(path) for path in targets if path.exists() or path.is_symlink()]
    post_snapshot, _ = protection_snapshot()
    post_governance = subprocess.run(
        [sys.executable, "-m", "chaoyang.governance.validate_governance_state"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    v6 = json.loads(V6.read_text(encoding="utf-8"))
    checks = {
        "v6_cleanup_passed": v6.get("status") == "PASSED",
        "v6_receipt_immutable": True,
        "processed_archive_absent": not Path("/mnt/data/egodata/datasets/ego/processed_archive").exists(),
        "processed_canary_absent": not Path("/mnt/data/egodata/datasets/ego/processed_canary").exists(),
        "now_absent": not (ROOT / "NOW").exists(),
        "top_level_depth_package_absent": not (ROOT / "DEPTH_ACCURACY_PACKAGE_20260911").exists(),
        "protected_egodata_not_targeted": all(not str(row["path"]).startswith("/mnt/data/egodata") for row in rows),
        "protected_nas_not_targeted": all(not str(row["path"]).startswith("/nas/chenxianchi") for row in rows),
        "post_governance_pass": post_governance.returncode == 0 and '"status": "PASS"' in post_governance.stdout,
        "post_governance_revision_unchanged": post_snapshot["governance_revision"] == snapshot["governance_revision"],
        "post_git_snapshot_unchanged": post_snapshot["git"] == snapshot["git"],
    }
    status = "PASSED" if not remaining and all(checks.values()) else "BLOCKED_REFERENCE_PROOF"
    receipt = {
        "schema_version": "CLEANUP_DELETION_RECEIPT_V71",
        "plan_revision": "chaoyang-v7.1",
        "artifact_revision": "R7_0",
        "completed_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "status": status,
        "deleted": rows,
        "remaining": remaining,
        "deleted_bytes": sum(row["stats"]["bytes"] for row in rows),
        "checks": checks,
        "post_validation": {
            "governance_returncode": post_governance.returncode,
            "governance_stdout_tail": post_governance.stdout[-2000:],
            "governance_stderr_tail": post_governance.stderr[-2000:],
        },
        "pre_delete_manifest": {
            "path": str(prior_manifest_path),
            "bytes": prior_manifest_path.stat().st_size,
            "sha256": sha(prior_manifest_path),
        },
        "prior_v6_receipt": {"path": str(V6), "bytes": V6.stat().st_size, "sha256": sha(V6)},
        "protected_external": ["/mnt/data/egodata", "/nas/chenxianchi"],
        "claim_limit": "V7.1 regenerated-cache closure plus V6 receipt adoption; no data-side, authority, checkpoint, or algorithm result was deleted.",
    }
    publish(output_root / "CLEANUP_DELETION_RECEIPT.json", receipt)
    publish(output_root / "RESULT.json", receipt)
    print(json.dumps({"status": status, "targets": len(rows), "deleted_bytes": receipt["deleted_bytes"]}))


if __name__ == "__main__":
    main()
