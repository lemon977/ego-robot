#!/usr/bin/env python3
"""Non-recursive adapter for the v5 mount-frame correction."""
from __future__ import annotations

import numpy as np

from chaoyang.governance.common import REPO_ROOT, artifact_ref
from chaoyang.ops import run_exact78_session_base_arm_v1 as base
from chaoyang.ops import run_exact78_session_base_arm_v5 as previous
from chaoyang.pipeline.robot_renderer_cycles import forward_kinematics
from chaoyang.pipeline.robot_renderer_eevee_fullchain import ARM_JOINT_NAMES


ORIGINAL_MOUNTS = base.mounts


def corrected_mounts() -> tuple[np.ndarray, dict]:
    flange_hand, source = ORIGINAL_MOUNTS()
    assets = base.arm.old.load_pinned_robot_assets(REPO_ROOT)
    values = {name: 0.0 for row in ARM_JOINT_NAMES for name in row}
    fk = forward_kinematics(assets.tianji, values)
    result = []
    flange_tool = []
    for physical, side in enumerate(("left", "right")):
        flange = fk[f"flange_{'L' if physical == 0 else 'R'}"]
        tool = fk[f"{side}_tool"]
        fixed = np.linalg.inv(flange) @ tool
        converted = np.linalg.inv(fixed) @ flange_hand[physical]
        if not np.allclose(fixed @ converted, flange_hand[physical], atol=1e-10):
            raise RuntimeError(f"MOUNT_FRAME_CONVERSION_FAILED:{side}")
        flange_tool.append(fixed)
        result.append(converted)
    reference = dict(source)
    reference.update({
        "source": artifact_ref(base.MOUNT_CONTRACT),
        "source_matrix_direction": "p_flange = T_flange_hand_root_proxy @ p_hand_root",
        "consumer_matrix_direction": "p_tool = T_tool_hand_root_proxy @ p_hand_root",
        "conversion": "inv(T_flange_tool_from_pinned_urdf) @ T_flange_hand_root_proxy",
        "T_flange_tool_from_pinned_urdf": np.asarray(flange_tool).tolist(),
        "cards042_fitted_values_read": False,
    })
    return np.asarray(result), reference


def main() -> int:
    old = previous.corrected_mounts
    previous.corrected_mounts = corrected_mounts
    try:
        return previous.main()
    finally:
        previous.corrected_mounts = old


if __name__ == "__main__":
    raise SystemExit(main())
