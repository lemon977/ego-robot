#!/usr/bin/env python3
"""Robot-ready batch runner using per-side first-observed wrist anchoring.

This is an immutable orchestration successor to v5.3.  It first creates the
ordinary frozen v5.2 input preflight, then publishes the narrow anchor-v3 code
closure.  All downstream arm phases consume that successor preflight, so a
session is no longer rejected merely because frame 0 misses one physical hand.
Every other placement, solver, hand, rendering, and terminal rule is delegated
unchanged to v5.3.
"""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[3]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from chaoyang.ops import run_exact78_robot_ready_batches_v53 as legacy

LEGACY_BATCH_COMMANDS = legacy.batch_commands


def batch_commands_v54(batch_root: Path, sessions: list[str], matrix_snapshot: Path) -> None:
    """Publish the anchor-v3 preflight before delegating the remaining DAG."""
    label = "__".join(sessions)
    contract = batch_root / "ROBOT_READY_INPUT.json"
    base_preflight = batch_root / "preflight_base_v52" / "RESULT.json"
    successor_preflight = batch_root / "preflight" / "RESULT.json"

    legacy.run_phase(
        batch_root,
        "build_contract",
        [
            sys.executable,
            "src/chaoyang/ops/build_exact78_robot_ready_contract_v52.py",
            "--matrix",
            str(matrix_snapshot),
            "--sessions",
            *sessions,
            "--output",
            str(contract),
        ],
        contract,
        label,
    )
    frozen_matrix = Path(legacy.load_json(contract)["matrix_snapshot"]["path"])
    legacy.run_phase(
        batch_root,
        "preflight_base_v52",
        [
            sys.executable,
            "src/chaoyang/ops/preflight_exact78_robot_ready_batch_v52.py",
            "--contract",
            str(contract),
            "--matrix",
            str(frozen_matrix),
            "--output",
            str(base_preflight),
        ],
        base_preflight,
        label,
    )
    legacy.run_phase(
        batch_root,
        "preflight_anchor_v3",
        [
            sys.executable,
            "src/chaoyang/ops/preflight_exact78_robot_ready_anchor_v3.py",
            "--base-preflight",
            str(base_preflight),
            "--output",
            str(successor_preflight),
        ],
        successor_preflight,
        label,
    )
    # The anchor-v3 preflight intentionally replaces the legacy preflight.
    # Continue after input construction rather than relying on file existence.
    LEGACY_BATCH_COMMANDS(batch_root, sessions, matrix_snapshot, prepared_inputs=True)


def main() -> int:
    legacy.batch_commands = batch_commands_v54
    return legacy.main()


if __name__ == "__main__":
    raise SystemExit(main())
