#!/usr/bin/env python3
"""Build a provenance-closed, same-session OFFLINE_VISUAL Robot initializer.

The output mimics the *input schema* of v3's historical accepted template, but
does not read an accepted run or any other session.  Identity tool-to-hand-root
is an explicit development-only proxy, not flange or deployment calibration.
"""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from chaoyang.ops import run_robot_hand_fullsession_v2 as hand
from chaoyang.ops import run_robot_motion_transfer_arm_canary_v2 as arm


def file_ref(path: Path) -> dict:
    path = path.resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def verify_pin(pin: dict, label: str) -> Path:
    path = Path(pin["path"])
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise RuntimeError(f"{label}: regular absolute file required")
    actual = file_ref(path)
    if (actual["bytes"], actual["sha256"]) != (pin["bytes"], pin["sha256"]):
        raise RuntimeError(f"{label}: bytes/SHA mismatch")
    return path


def first_observed(joints: np.ndarray, limit: int) -> tuple[np.ndarray, np.ndarray]:
    if joints.ndim != 4 or joints.shape[0] != 2 or joints.shape[2:] != (21, 3):
        raise RuntimeError("expected physical-left/right MANO [2,T,21,3]")
    valid = np.isfinite(joints).all(axis=(2, 3))
    anchors = []
    for side in range(2):
        choices = np.flatnonzero(valid[side, :limit])
        if len(choices) == 0:
            raise RuntimeError(f"side {side}: no valid observation inside frozen prefix")
        anchors.append(int(choices[0]))
    return np.asarray(anchors, dtype=np.int64), valid


def independent_base_translation(wrists_world: np.ndarray, neutral_roots_base: np.ndarray) -> np.ndarray:
    """One fixed world base, least-squares translation; no foreign placement."""
    if wrists_world.shape != (2, 3) or neutral_roots_base.shape != (2, 3):
        raise RuntimeError("bilateral world wrist and neutral FK positions required")
    if not np.isfinite(wrists_world).all() or not np.isfinite(neutral_roots_base).all():
        raise RuntimeError("non-finite placement input")
    return np.mean(wrists_world - neutral_roots_base, axis=0)


def prefix_arrays(full: dict[str, np.ndarray], count: int) -> dict[str, np.ndarray]:
    total = len(full["c2w"])
    if not 1 <= count <= total:
        raise RuntimeError("invalid prefix frame count")
    result = {}
    for name, value in full.items():
        if value.ndim >= 2 and value.shape[:2] == (2, total):
            result[name] = value[:, :count]
        elif value.ndim >= 1 and value.shape[0] == total:
            result[name] = value[:count]
        else:
            result[name] = value
    if not np.array_equal(result["original_frame_indices"], np.arange(count)):
        raise RuntimeError("prefix source-frame identity not 0..N-1")
    return result


def build(packet_path: Path, output_dir: Path, prefix_frames: int) -> dict:
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    session = packet["session_id"]
    if not session.startswith(("get_potato_chips_", "play_cards_")):
        raise RuntimeError("full session ID required")
    source = verify_pin(packet["hawor_npz"], "same-session HaWoR")
    result = verify_pin(packet["hawor_result"], "same-session HaWoR RESULT")
    asset_pin = verify_pin(packet["robot_asset_pin"], "robot asset closure")
    preset_path = verify_pin(packet["preset"], "frozen offline preset")
    preset = json.loads(preset_path.read_text(encoding="utf-8"))
    if packet.get("source_session_id") != session or session not in str(source) or session not in str(result):
        raise RuntimeError("foreign or ambiguous source session")
    if preset != {
        "schema_version": "same-session-neutral-placement-preset-v1",
        "base_rotation_world": "IDENTITY_WORLD_ASSUMPTION",
        "arm_initial_q": "URDF_ZERO_CLIPPED",
        "hand_initial_q": "URDF_NEUTRAL",
        "tool_hand_root": "IDENTITY_VISUAL_PROXY",
        "base_translation": "MEAN_TWO_FIRST_OBSERVED_WRISTS_MINUS_NEUTRAL_FK_ROOTS",
        "scope": "OFFLINE_VISUAL_DIAGNOSTIC_ONLY",
    }:
        raise RuntimeError("unreviewed task preset or physical-calibration claim")
    source_doc = json.loads(result.read_text(encoding="utf-8"))
    documented = source_doc.get("outputs", {}).get("npz", source_doc.get("npz", {}))
    if documented.get("sha256") != packet["hawor_npz"]["sha256"]:
        raise RuntimeError("HaWoR RESULT/NPZ closure mismatch")
    with np.load(source, allow_pickle=False) as data:
        full = {key: np.asarray(data[key]) for key in data.files}
    count = min(prefix_frames, len(full["c2w"]))
    prefix = prefix_arrays(full, count)
    joints = np.asarray(full["joints_3d_world"], dtype=np.float64)
    anchors, _ = first_observed(joints, count)
    c2w0 = np.asarray(full["c2w"][0], dtype=np.float64)
    if not np.isfinite(c2w0).all() or not np.allclose(c2w0[3], [0, 0, 0, 1], atol=1e-8):
        raise RuntimeError("non-finite or non-homogeneous same-session c2w")
    if not np.allclose(c2w0[:3, :3].T @ c2w0[:3, :3], np.eye(3), atol=1e-4):
        raise RuntimeError("same-session c2w rotation not orthonormal")
    assets = arm.old.load_pinned_robot_assets(Path(__file__).resolve().parents[3])
    lower, upper = arm.taskfit.arm_limits(assets)
    q_arm0 = np.clip(np.zeros((2, 7), dtype=np.float64), lower, upper)
    contracts = hand.handfit.model_contract(hand.old.official, hand.shared.wrist_adapter, assets)
    q_hand0 = np.stack([np.asarray(contract["neutral"], dtype=np.float64) for contract in contracts])
    mounts = np.repeat(np.eye(4, dtype=np.float64)[None], 2, axis=0)
    neutral_fk = np.stack([arm.old.official._tool_fk(assets, side, q_arm0[side]) @ mounts[side]
                           for side in range(2)])
    wrists = np.stack([joints[side, anchors[side], 0] for side in range(2)])
    world_base = np.eye(4, dtype=np.float64)
    world_base[:3, 3] = independent_base_translation(wrists, neutral_fk[:, :3, 3])
    camera_base = np.linalg.inv(c2w0) @ world_base
    target0 = np.stack([world_base @ neutral_fk[side] for side in range(2)])
    target0[:, :3, 3] = wrists
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    prefix_path = output_dir / f"{session}_HAWOR_PREFIX_{count}.npz"
    states_path = output_dir / f"{session}_SAME_SESSION_NEUTRAL_INITIALIZER.npz"
    anchor_path = output_dir / f"{session}_SAME_SESSION_ANCHOR_HAWOR.npz"
    np.savez_compressed(prefix_path, **prefix)
    np.savez_compressed(states_path, q_arm=q_arm0[None], q_hand=q_hand0[None],
                        T_tool_hand_root=mounts, T_camera_base=camera_base[None],
                        T_target_hand_root_world=target0[None])
    anchor_joints = np.stack([joints[side, anchors[side]][None] for side in range(2)])
    np.savez_compressed(anchor_path, c2w=c2w0[None], joints_3d_world=anchor_joints)
    manifest = {
        "schema_version": "same-session-neutral-robot-initializer-v1",
        "session_id": session, "source_session_id": session, "prefix_frames": count,
        "anchor_frames": anchors.tolist(), "source_packet": file_ref(packet_path),
        "source_hawor": file_ref(source), "source_result": file_ref(result),
        "robot_asset_pin": file_ref(asset_pin), "preset": file_ref(preset_path),
        "code": file_ref(Path(__file__)),
        "outputs": {"prefix_hawor": file_ref(prefix_path), "initializer_states": file_ref(states_path),
                    "anchor_hawor": file_ref(anchor_path)},
        "world_base_translation_m": world_base[:3, 3].tolist(),
        "neutral_robot_root_separation_m": float(np.linalg.norm(neutral_fk[0, :3, 3] - neutral_fk[1, :3, 3])),
        "observed_human_wrist_separation_m": float(np.linalg.norm(wrists[0] - wrists[1])),
        "input_mode": "OFFLINE_VISUAL", "training_eligible": False,
        "control_ground_truth": False, "physical_deployment_authorized": False,
        "claim_limit": "Independent same-session initialization hypothesis only. Identity mount and world-axis base are explicit visual proxies, not calibrated hardware geometry.",
    }
    manifest_path = output_dir / "INITIALIZER_RESULT.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix-frames", type=int, default=4)
    args = parser.parse_args()
    result = build(args.packet, args.output_dir, args.prefix_frames)
    print(json.dumps({"session_id": result["session_id"], "prefix_frames": result["prefix_frames"],
                      "status": "BUILT_OFFLINE_DIAGNOSTIC_ONLY"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
