"""Record remaining large objects without granting deletion authority."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

from chaoyang.governance.common import load_json
from chaoyang.ops.run_human_to_robot_product_first_cable import ROOT, TASK

BASE = ROOT / "_run/current"
OUT = BASE / TASK / "attempts/attempt_0001/cleanup/LARGE_OBJECT_INVENTORY.json"


def main() -> int:
    if OUT.exists():
        raise FileExistsError(OUT)
    index = load_json(ROOT / "tasks/current/INDEX.json")
    if len(index.get("task_packets", [])) != 1 or index["task_packets"][0].get("task_id") != TASK:
        raise RuntimeError("TASK_NOT_ROUTABLE")
    use = subprocess.run(["du", "-x", "-B1", "-d1", str(BASE)], cwd=ROOT,
                         capture_output=True, text=True, timeout=600, check=False)
    if use.returncode:
        raise RuntimeError(f"DU_FAILED:{use.stderr[-1000:]}")
    rows = []
    root_allocated = None
    for line in use.stdout.splitlines():
        value, path = line.split("\t", 1)
        if Path(path).resolve() == BASE.resolve():
            root_allocated = int(value)
            continue
        rows.append({"path": path, "allocated_bytes_at_inventory": int(value)})
    rows.sort(key=lambda row: row["allocated_bytes_at_inventory"], reverse=True)
    if root_allocated is None:
        raise RuntimeError("ROOT_DU_MISSING")
    for row in rows[:20]:
        name = Path(row["path"]).name
        if name == "environments":
            reason = "RETAIN_PINNED_ACTIVE_MODELS_NO_VALIDATED_REPLACEMENT"
        elif name == "worktrees":
            reason = "RETAIN_UNTIL_TRACKED_UNTRACKED_IGNORED_UNIQUE_COMMIT_AND_RESTORE_PROOF"
        elif name == TASK:
            reason = "RETAIN_CURRENT_TASK_AND_FORENSIC_EVIDENCE"
        else:
            reason = "NOT_SELECTED_NO_EXACT_CURRENT_CONDITIONAL_AND_REPRODUCTION_DEPENDENCY_PROOF"
        row["deletion_decision"] = reason
    large = subprocess.run(["find", str(BASE), "-type", "f", "-size", "+536870911c",
                            "-printf", "%s\t%p\n"], cwd=ROOT,
                           capture_output=True, text=True, timeout=600, check=False)
    if large.returncode:
        raise RuntimeError(f"LARGE_FILE_SCAN_FAILED:{large.stderr[-1000:]}")
    files = []
    for line in large.stdout.splitlines():
        size, path = line.split("\t", 1)
        files.append({"path": path, "logical_bytes": int(size),
                      "deletion_decision": "RETAIN_OR_PROVE_EXACT_DEPENDENCY_BEFORE_ANY_DELETE"})
    files.sort(key=lambda row: row["logical_bytes"], reverse=True)
    result = {"schema_version": "PRODUCT_FIRST_LARGE_OBJECT_INVENTORY_V1",
              "task_id": TASK, "root": str(BASE),
              "root_allocated_bytes_at_inventory": root_allocated,
              "top20_subdirectories": rows[:20], "files_ge_512_mib": files,
              "subdirectories_ge_1_gib": [row for row in rows if row["allocated_bytes_at_inventory"] >= 1024**3],
              "deletion_authority": "NONE_FROM_INVENTORY_ALONE",
              "claim_limit": "Allocated du numbers are a point-in-time filesystem accounting view, not physical reclaimed bytes. Unselected objects are deferred per exact dependency proof; they are not declared permanently necessary."}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "INVENTORIED_NOT_DELETED", "root_allocated_bytes": root_allocated,
                      "top20": len(rows[:20]), "large_files": len(files), "result": str(OUT)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
