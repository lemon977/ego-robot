from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import numpy as np

from chaoyang.ops import run_0915_post_geometry_robot_v1 as post


def test_robot_visual_sidecar_stays_schema_exact_and_evidence_is_separate(
    tmp_path: Path, monkeypatch,
) -> None:
    hawor = tmp_path / "hawor.npz"
    np.savez_compressed(
        hawor,
        joints_3d_world=np.zeros((2, 3, 21, 3), dtype=np.float32),
        observed=np.ones((2, 3), dtype=bool),
        fps=np.asarray(30.0),
    )
    arrays = {
        "q_arm": np.zeros((2, 3, 6), dtype=np.float32),
        "q_hand": np.zeros((2, 3, 10), dtype=np.float32),
        "valid_side_frame": np.ones((2, 3), dtype=bool),
        "virtual_tool_targets": np.zeros((2, 3, 3), dtype=np.float32),
    }
    metrics = {
        key: 0.0 for key in (
            "arm_ik", "kaihand_retarget", "collision", "joint_limit",
            "velocity", "acceleration",
        )
    }

    def fake_solve_robot_visual(**_kwargs):
        return {
            "status": "PASS",
            **arrays,
            "calibration_evidence": {
                "robot_tcp": "ABSENT",
                "tool_to_kaihand_root": "PRESENT_CANDIDATE_GEOMETRY",
                "camera_world_to_base": "ABSENT",
            },
            "metrics": metrics,
            "claim_limit": "Development-only test result.",
            "workspace_search": {"candidate_count": 1},
            "diagnostics": {"sample_frames": [0]},
            "gates": {"ik": True},
        }

    monkeypatch.setattr(post, "solve_robot_visual", fake_solve_robot_visual)
    staging = tmp_path / ".001.staging"
    published = tmp_path / "001"
    staging.mkdir()
    assert post.process_robot_visual(
        session_id="001", hawor_npz=hawor, output=staging,
        published_output=published, assets=object(),
    ) == "PASS"

    sidecar = json.loads((staging / "ROBOT_VISUAL.json").read_text())
    schema = json.loads(post.ROBOT_SCHEMA.read_text())
    jsonschema.Draft202012Validator(schema).validate(sidecar)
    assert set(sidecar) == set(schema["required"])

    evidence = json.loads((staging / "ROBOT_VISUAL_EVIDENCE.json").read_text())
    assert evidence["states"]["path"] == str(
        (published / "ROBOT_VISUAL_STATES.npz").resolve()
    )
    assert evidence["sidecar"]["path"] == str(
        (published / "ROBOT_VISUAL.json").resolve()
    )
