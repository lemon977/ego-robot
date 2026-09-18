#!/usr/bin/env python3
"""Corrective executor for the provenance-enum-only V1 early stop.

All algorithm code remains in V1.  This successor changes only the frozen
HaWoR provenance validator: observed rows must be ``BOUNDED_PARAMETER_FIT``
and absent rows must be ``MISSING``.  The boolean ``observed`` axis remains the
authority, so no short-gap inference is admitted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from chaoyang.ops import run_0915_interaction_contact_robot_dev_v1 as implementation


TASK_ID = "0915_interaction_contact_robot_dev_v2"
ROOT = Path(__file__).resolve().parents[3]


def load_frozen_inputs_v2() -> dict[str, Any]:
    depth_result = implementation.load_json(implementation.DEPTH_ROOT / "RESULT.json")
    object_result = implementation.load_json(implementation.OBJECT_ROOT / "RESULT.json")
    object_doc = implementation.load_json(
        implementation.OBJECT_ROOT / "OBJECT6D_OBSERVABILITY_V2.json"
    )
    if depth_result.get("status") != "PASSED" or object_result.get("status") != "PASSED":
        raise RuntimeError("frozen Depth/Object6D inputs are not terminal PASSED")
    if (
        object_doc.get("session_id") != implementation.SESSION_ID
        or object_doc.get("frame_count") != implementation.FRAME_COUNT
    ):
        raise RuntimeError("Object6D session/frame denominator drift")
    with np.load(implementation.HAWOR, allow_pickle=False) as archive:
        hawor = {key: np.asarray(archive[key]) for key in archive.files}
    observed = np.asarray(hawor["observed"], bool)
    provenance = np.asarray(hawor["provenance"])
    if (
        hawor["joints_3d_camera"].shape != (2, implementation.FRAME_COUNT, 21, 3)
        or hawor["joints_2d"].shape != (2, implementation.FRAME_COUNT, 21, 2)
        or observed.shape != (2, implementation.FRAME_COUNT)
        or provenance.shape != observed.shape
        or not np.array_equal(
            hawor["original_frame_indices"], np.arange(implementation.FRAME_COUNT)
        )
    ):
        raise RuntimeError("frozen direct-observed HaWoR axis drift")
    if not np.all(provenance[observed] == "BOUNDED_PARAMETER_FIT"):
        raise RuntimeError("observed HaWoR rows lack bounded direct-fit provenance")
    if not np.all(provenance[~observed] == "MISSING"):
        raise RuntimeError("missing HaWoR rows contain inferred provenance")
    return {
        "depth_result": depth_result,
        "object_result": object_result,
        "object_doc": object_doc,
        "hawor": hawor,
    }


def main() -> int:
    implementation.TASK_ID = TASK_ID
    implementation.PHASE = "0915_INTERACTION_CONTACT_KAI22_DEVELOPMENT_V2"
    implementation.OUTPUT = ROOT / f"_run/current/{TASK_ID}/attempts/attempt_0001"
    implementation.TERMINAL_RECEIPT = (
        ROOT / "tasks/receipts/0915_INTERACTION_CONTACT_ROBOT_DEV_V2_RESULT.json"
    )
    implementation.load_frozen_inputs = load_frozen_inputs_v2
    implementation.__file__ = __file__
    return implementation.main()


if __name__ == "__main__":
    raise SystemExit(main())
