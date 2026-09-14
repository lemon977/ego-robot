from __future__ import annotations

import argparse
import json
from tools.governance.common import AUTHORITY_PATH, CHANGELOG_PATH, MIN_STATUS_PATH, RECEIPT_PATH, STATUS_PATH, TASK_QUEUE_PATH, TASK_STATE_PATH, freshness, load_json, sha256_bytes, validate_artifact_ref, validate_authority, validate_task_state


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-stale", action="store_true")
    args = parser.parse_args()
    errors: list[str] = []
    receipt = load_json(RECEIPT_PATH)
    authority = load_json(AUTHORITY_PATH)
    task_state = load_json(TASK_STATE_PATH)
    for item in receipt["files"].values():
        errors.extend(validate_artifact_ref(item))
    for value, name in ((authority, "authority"), (task_state, "task_state")):
        if value["governance_revision"] != receipt["governance_revision"]:
            errors.append(f"{name} revision does not match receipt")
        if value["generation_id"] != receipt["generation_id"]:
            errors.append(f"{name} generation_id does not match receipt")
    errors.extend(validate_authority(authority))
    errors.extend(validate_task_state(task_state))
    if not STATUS_PATH.read_text(encoding="utf-8").startswith("# 当前项目实时事实页"):
        errors.append("project status is not generated content")
    for projection_path, label in (
        (MIN_STATUS_PATH, "project_status_min"),
        (TASK_QUEUE_PATH, "task_queue"),
    ):
        projection = load_json(projection_path)
        if projection.get("governance_revision") != receipt["governance_revision"]:
            errors.append(f"{label} revision does not match receipt")
        if projection.get("generation_id") != receipt["generation_id"]:
            errors.append(f"{label} generation_id does not match receipt")
    previous = None
    if CHANGELOG_PATH.is_file():
        for index, line in enumerate(CHANGELOG_PATH.read_text(encoding="utf-8").splitlines(), 1):
            event = json.loads(line)
            claimed = event.pop("event_sha256")
            if event.get("previous_event_sha256") != previous:
                errors.append(f"changelog chain mismatch at line {index}")
            actual = sha256_bytes((json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode())
            if actual != claimed:
                errors.append(f"changelog event sha mismatch at line {index}")
            previous = claimed
    fresh = freshness(task_state)
    if fresh["status"] in {"STALE", "SUSPECTED_DEAD_WORKER"} and not args.allow_stale:
        errors.append(f"freshness gate failed: {fresh}")
    output = {"status": "PASS" if not errors else "STATUS_CONFLICT", "errors": errors, "freshness": fresh, "revision": receipt["governance_revision"]}
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
