#!/usr/bin/env python3
"""Run the three handle-cleaning V3 datasets serially and persistently."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
import uuid


PROCESSED = Path("/mnt/data/egodata/datasets/ego/processed")
PYTHON = Path("/cpfs_infra/user/chenxianchi/miniconda3/envs/egoforce/bin/python")
REPO = Path(__file__).resolve().parents[3]
BATCH = REPO / "src/chaoyang/ops/batch_clean_handle_content_v3.py"
DATASETS = (
    {
        "name": "chips_cards_handle_0911",
        "date": "0911",
        "dataset_id": "chips_cards_handle_0911_v3",
        "cards": "/mnt/data/egodata/datasets/ego/chips_cards_handle_0911/cards_130_0911",
        "chips": "/mnt/data/egodata/datasets/ego/chips_cards_handle_0911/chips_111_0911",
    },
    {
        "name": "chips_cards_handle_highview_0914",
        "date": "0914",
        "dataset_id": "chips_cards_handle_highview_0914_v3",
        "cards": "/mnt/data/egodata/datasets/ego/chips_cards_handle_highview_0914/cards_120_0914",
        "chips": "/mnt/data/egodata/datasets/ego/chips_cards_handle_highview_0914/chips_100_0914",
    },
    {
        "name": "chips_cards_hands__0915",
        "date": "0915",
        "dataset_id": "chips_cards_hands__0915_v3",
        "cards": "/mnt/data/egodata/datasets/ego/chips_cards_hands__0915/cards_120_0915",
        "chips": "/mnt/data/egodata/datasets/ego/chips_cards_hands__0915/chips_100_0915",
    },
)


class QueueError(RuntimeError):
    pass


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                   allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def terminal(root: Path) -> dict[str, Any] | None:
    path = root / "DATASET_RESULT.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def process_matches(pid: int, root: Path) -> bool:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ")
    except OSError:
        return False
    return (b"batch_clean_handle_content_v3.py" in raw and
            str(root).encode() in raw)


def wait_for_first(pid: int, root: Path, state_path: Path,
                   queue_state: dict[str, Any]) -> dict[str, Any]:
    while True:
        result = terminal(root)
        if result is not None:
            return result
        if not process_matches(pid, root):
            raise QueueError(
                f"first batch pid {pid} ended without DATASET_RESULT.json"
            )
        queue_state.update({"state": "WAITING_FOR_0911",
                            "current_dataset": root.name,
                            "updated_unix": time.time()})
        atomic_json(state_path, queue_state)
        time.sleep(30)


def run_dataset(config: dict[str, str], state_path: Path,
                queue_state: dict[str, Any]) -> dict[str, Any]:
    root = PROCESSED / config["name"]
    prior = terminal(root)
    if prior is not None:
        return prior
    command = [
        str(PYTHON), str(BATCH), "--mode", "convert",
        "--task-source", f"playing_cards={config['cards']}",
        "--task-source", f"potato_chips={config['chips']}",
        "--date-tag", config["date"], "--dataset-id", config["dataset_id"],
        "--target-root", str(root),
    ]
    queue_state.update({"state": "RUNNING", "current_dataset": config["name"],
                        "updated_unix": time.time()})
    atomic_json(state_path, queue_state)
    log_path = root / "RUN.log"
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write(json.dumps({"queue_launch_unix": time.time(),
                              "command": command}, ensure_ascii=False) + "\n")
        completed = subprocess.run(command, cwd=REPO, stdout=log,
                                   stderr=subprocess.STDOUT, text=True)
    result = terminal(root)
    if completed.returncode or result is None:
        raise QueueError(
            f"{config['name']} batch failed rc={completed.returncode}; see {log_path}"
        )
    return result


def require_committed(name: str, result: dict[str, Any]) -> None:
    if result.get("state") != "COMMITTED" or int(result.get("failed", -1)) != 0:
        raise QueueError(
            f"{name} terminal state is not clean: state={result.get('state')} "
            f"failed={result.get('failed')}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-pid", type=int, required=True)
    args = parser.parse_args()
    state_path = PROCESSED / DATASETS[0]["name"] / "QUEUE_STATE.json"
    queue_state: dict[str, Any] = {
        "schema_version": "handle-cleaning-v3-queue-state-v1",
        "state": "STARTING", "pid": os.getpid(),
        "started_unix": time.time(), "datasets": [item["name"] for item in DATASETS],
        "completed_datasets": [],
    }
    atomic_json(state_path, queue_state)
    try:
        first_root = PROCESSED / DATASETS[0]["name"]
        first = wait_for_first(args.first_pid, first_root, state_path, queue_state)
        require_committed(DATASETS[0]["name"], first)
        queue_state["completed_datasets"].append(DATASETS[0]["name"])
        for config in DATASETS[1:]:
            result = run_dataset(config, state_path, queue_state)
            require_committed(config["name"], result)
            queue_state["completed_datasets"].append(config["name"])
        queue_state.update({"state": "COMMITTED", "current_dataset": None,
                            "finished_unix": time.time(),
                            "updated_unix": time.time()})
        atomic_json(state_path, queue_state)
        return 0
    except BaseException as exc:
        queue_state.update({"state": "FAILED", "error": repr(exc),
                            "finished_unix": time.time(),
                            "updated_unix": time.time()})
        atomic_json(state_path, queue_state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
