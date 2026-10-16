"""Small governed runtime helpers shared by the retained Exact78 V2 stages."""

from __future__ import annotations

from pathlib import Path

from chaoyang.governance.common import REPO_ROOT, artifact_ref, load_json, process_identity


def authorize(task_id: str) -> tuple[dict, Path]:
    routes = [row for row in load_json(REPO_ROOT / "tasks/current/INDEX.json")["task_packets"]
              if row["task_id"] == task_id and row.get("execution_allowed")]
    if len(routes) != 1:
        raise RuntimeError("PIPELINE_NOT_ROUTABLE")
    path = REPO_ROOT / routes[0]["packet_path"]
    if artifact_ref(path)["sha256"] != routes[0]["packet_sha256"]:
        raise RuntimeError("PACKET_SHA_DRIFT")
    packet = load_json(path)
    identity = process_identity(packet["writer"]["pid"])
    if not identity["alive"] or identity["start_ticks"] != packet["writer"]["proc_start_ticks"]:
        raise RuntimeError("WRITER_FENCE_DRIFT")
    root = Path(packet["write_set"][0]).resolve()
    if load_json(root / "WRITER.json") != packet["writer"]:
        raise RuntimeError("WRITER_TOKEN_DRIFT")
    return packet, root


def checked(reference: dict) -> Path:
    path = Path(reference["path"])
    if artifact_ref(path) != reference:
        raise RuntimeError(f"INPUT_SHA_DRIFT:{path}")
    return path


def verify_refs(value) -> None:
    if isinstance(value, dict):
        if {"path", "bytes", "sha256"} <= value.keys():
            checked(value)
        else:
            for child in value.values():
                verify_refs(child)
    elif isinstance(value, list):
        for child in value:
            verify_refs(child)
