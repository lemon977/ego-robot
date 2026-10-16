#!/usr/bin/env python3
"""P1 arm validation that can reuse SHA-pinned same-session HaWoR outputs.

Only HaWoR artifacts produced by the already-terminal retry6 task may be
reused, and only when every file reference in the new manifest still matches
its recorded SHA.  Arm/base/hand/diagnostic stages always run fresh with the
v4 per-session solver.  Sessions without a reuse bundle run fresh HaWoR v2.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

from chaoyang.governance.common import artifact_ref, atomic_json
from chaoyang.ops import run_exact78_general_p1_action_canary_v9 as previous


REUSE_FILES = {
    "hawor_l0": ("HAWOR_RAW_MANO21.npz", "RESULT.json"),
    "hawor_recovery": ("HAWOR_RECOVERED_VISUAL_MANO21.npz", "RESULT.json"),
}


def _manifest_path() -> Path:
    return Path(sys.argv[sys.argv.index("--manifest") + 1]).resolve(strict=True)


def main() -> int:
    manifest = json.loads(_manifest_path().read_text(encoding="utf-8"))
    reuse = {
        row["session_id"]: row["reuse_hawor"]
        for row in manifest["sessions"] if row.get("reuse_hawor")
    }
    runner = previous.previous.base
    original_gpu = runner.gpu

    def pinned_gpu(command: list[str], session: Path, task_id: str, stage: str) -> None:
        del task_id
        session_id = session.name
        bundle = reuse.get(session_id)
        if bundle is None or stage not in REUSE_FILES:
            return original_gpu(command, session, manifest["task_id"], stage)
        destination = session / stage
        destination.mkdir()
        copied = {}
        for filename in REUSE_FILES[stage]:
            reference = bundle[stage][filename]
            source = runner.checked(reference)
            target = destination / filename
            shutil.copy2(source, target)
            copied[filename] = {
                "source": reference,
                "copy": artifact_ref(target),
                "sha_equal": artifact_ref(target)["sha256"] == reference["sha256"],
            }
            if not copied[filename]["sha_equal"]:
                raise RuntimeError(f"REUSED_HAWOR_COPY_SHA_DRIFT:{session_id}:{stage}:{filename}")
        atomic_json(destination / "REUSE_RECEIPT.json", {
            "session_id": session_id,
            "stage": stage,
            "source_task": bundle["source_task"],
            "copied": copied,
            "claim_limit": "SHA-identical same-session HaWoR reuse only; all downstream stages run fresh.",
        })

    runner.gpu = pinned_gpu
    try:
        return previous.main()
    finally:
        runner.gpu = original_gpu


if __name__ == "__main__":
    raise SystemExit(main())
