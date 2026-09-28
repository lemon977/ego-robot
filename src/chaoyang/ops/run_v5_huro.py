"""Bounded HuRo core wrist-pose candidate on the V5 common HumanMotion/R0 input.

The pinned vendor source is read unchanged. A SHA-checked in-memory copy of its
trajectory function adds one masked left/right wrist SE(3) cost inside the same
JAX least-squares problem. Targets, placement, mount and evaluation masks are
frozen from the corresponding local R0 artifact. CPU only, visual use only.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path

import numpy as np


REPO = Path("/mnt/workspace/code/chaoyang")
LANE = REPO / "_run/current/four_stream_visual_delivery_v5/attempts/attempt_0001/lanes/huro"
HURO_ENV = REPO / "_run/current/four_stream_algorithm_baseline_v2/attempts/attempt_0001/lanes/ai4_huro/env/bin/python"
BLOCK_SIZE = 32
MAX_GAP_NS = 250_000_000
WRIST_POSITION_SCALE_M = 0.005
WRIST_ROTATION_SCALE_RAD = np.deg2rad(2.0)
SESSION_FRAMES = {"get_potato_chips_0915_007": 378, "play_cards_0915_031": 149}


def reference(path: Path) -> dict:
    value = path.resolve(strict=True)
    before = value.stat()
    digest = hashlib.sha256()
    with value.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    after = value.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError(f"unstable artifact: {value}")
    return {"path": str(value), "bytes": after.st_size, "sha256": digest.hexdigest()}


def write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
        stream.write("\n")


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.array(archive[key], copy=True) for key in archive.files}


def load_case(session_id: str, motion_path: Path, r0_path: Path) -> dict:
    if session_id not in SESSION_FRAMES:
        raise ValueError(f"unsupported session: {session_id}")
    motion_ref, r0_ref = reference(motion_path), reference(r0_path)
    hand, r0 = load_npz(motion_path), load_npz(r0_path)
    n = SESSION_FRAMES[session_id]
    if hand["frame_id"].shape != (n,) or r0["frame_id"].shape != (n,):
        raise ValueError(f"{session_id}: frame count drift")
    for key in ("frame_id", "timestamp_ns", "anatomical_side_names"):
        if not np.array_equal(hand[key], r0[key]):
            raise ValueError(f"{session_id}: HandMotion/R0 {key} mismatch")
    if tuple(hand["anatomical_side_names"].tolist()) != ("left", "right"):
        raise ValueError(f"{session_id}: anatomical side drift")
    if not np.array_equal(r0["human_to_physical"], [0, 1]):
        raise ValueError(f"{session_id}: unsupported physical mapping")
    if np.any(np.diff(hand["timestamp_ns"]) <= 0) or not np.array_equal(hand["frame_id"], np.arange(n)):
        raise ValueError(f"{session_id}: timeline not intact")
    if hand["joints21_camera"].shape != (n, 2, 21, 3) or hand["joint_valid"].shape != (n, 2, 21):
        raise ValueError(f"{session_id}: HandMotion schema drift")
    if r0["T_target_root_cam"].shape != (n, 2, 4, 4) or r0["target_valid"].shape != (n, 2):
        raise ValueError(f"{session_id}: R0 schema drift")
    if r0["T_cam_base"].shape != (4, 4) or r0["T_flange_hand"].shape != (2, 4, 4):
        raise ValueError(f"{session_id}: placement/mount drift")
    if bool(hand["training_eligible"]) or bool(r0["training_eligible"]) or bool(hand["control_ground_truth"]) or bool(r0["control_ground_truth"]):
        raise ValueError(f"{session_id}: unexpected authority")
    valid = r0["target_valid"] & hand["position_valid"] & hand["joint_valid"].all(axis=-1)
    rotation_valid = valid & hand["rotation_valid"]
    if np.any(valid & ~np.isfinite(hand["joints21_camera"]).all(axis=(-1, -2))):
        raise ValueError(f"{session_id}: nonfinite valid points")
    if np.any(valid & ~np.isfinite(r0["T_target_root_cam"]).all(axis=(-1, -2))):
        raise ValueError(f"{session_id}: nonfinite valid wrist target")
    if not np.array_equal(valid, r0["target_valid"]):
        raise ValueError(f"{session_id}: common validity would drop R0 target frames")
    return {"session_id": session_id, "hand": hand, "r0": r0, "valid": valid,
            "rotation_valid": rotation_valid, "motion_ref": motion_ref, "r0_ref": r0_ref}


def adapted_core(core_path: Path):
    """Load the pinned upstream core and inject one auditable cost into its solve."""
    import jax.numpy as jnp
    from chaoyang.ops.run_huro_fixed_placement_core_v2 import load_core
    from chaoyang.pipeline.huro_core_adapter_v2 import CORE_SOURCE_SHA

    raw = core_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != CORE_SOURCE_SHA:
        raise ValueError("pinned HuRo core source drift")
    core = load_core(core_path)
    source = raw.decode("utf-8")
    parsed = ast.parse(source)
    original = next(node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name == "solve_retargeting")
    lines = source.splitlines(keepends=True)
    start = min((item.lineno for item in original.decorator_list), default=original.lineno) - 1
    function = "".join(lines[start:original.end_lineno])

    def replace_once(old: str, new: str) -> None:
        nonlocal function
        if function.count(old) != 1:
            raise ValueError(f"HuRo injection anchor drift: {old[:55]!r}")
        function = function.replace(old, new, 1)

    replace_once("def solve_retargeting(", "def solve_retargeting_with_wrist(")
    replace_once(
        "    hand_joint_mask: jnp.ndarray,\n) -> Tuple[jnp.ndarray, jnp.ndarray]:",
        "    hand_joint_mask: jnp.ndarray,\n    target_wrist_se3: jaxlie.SE3,\n    wrist_pos_mask: jnp.ndarray,\n    wrist_rot_mask: jnp.ndarray,\n    wrist_link_indices: jnp.ndarray,\n) -> Tuple[jnp.ndarray, jnp.ndarray]:",
    )
    wrist_cost = '''    @jaxls.Cost.factory
    def wrist_pose_cost(
        var_values: jaxls.VarValues,
        var_robot_cfg: jaxls.Var[jnp.ndarray],
        target_se3: jaxlie.SE3,
        pos_mask: jnp.ndarray,
        rot_mask: jnp.ndarray,
    ) -> jax.Array:
        cfg_raw = var_values[var_robot_cfg]
        cfg_eff = initial_cfg + joint_mask * (cfg_raw - initial_cfg)
        all_fk = jaxlie.SE3(robot.forward_kinematics(cfg=cfg_eff))
        actual = jaxlie.SE3(all_fk.wxyz_xyz[wrist_link_indices])
        return _v5_wrist_residual(actual, target_se3, pos_mask, rot_mask)

'''
    replace_once("    @jaxls.Cost.factory\n    def joint_limit_cost(", wrist_cost + "    @jaxls.Cost.factory\n    def joint_limit_cost(")
    replace_once(
        "        camera_alignment_cost(var_joints, target_cam_se3, cam_mask),\n",
        "        camera_alignment_cost(var_joints, target_cam_se3, cam_mask),\n"
        "        wrist_pose_cost(var_joints, target_wrist_se3, wrist_pos_mask, wrist_rot_mask),\n",
    )

    def wrist_residual(actual, target, pos_mask, rot_mask):
        error = (target.inverse() @ actual).log()
        translation = error[..., :3] * (pos_mask[..., None] / WRIST_POSITION_SCALE_M)
        rotation = error[..., 3:] * (rot_mask[..., None] / WRIST_ROTATION_SCALE_RAD)
        return jnp.concatenate((translation, rotation), axis=-1).reshape(-1)

    core.__dict__["_v5_wrist_residual"] = wrist_residual
    exec(compile(function, str(core_path) + ":v5_wrist", "exec"), core.__dict__)
    return core, core.__dict__["solve_retargeting_with_wrist"], wrist_residual, function


def directional_tests(wrist_residual) -> dict:
    import jax
    import jax.numpy as jnp
    import jaxlie

    identity = jaxlie.SE3.from_matrix(jnp.broadcast_to(jnp.eye(4), (2, 4, 4)))
    def objective(vector, pos_mask, rot_mask):
        left = jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3.exp(jnp.array([0.0, 0.0, vector[1]])),
            jnp.array([vector[0], 0.0, 0.0]),
        )
        actual = jaxlie.SE3(jnp.stack((left.wxyz_xyz, identity.wxyz_xyz[1])))
        residual = wrist_residual(actual, identity, pos_mask, rot_mask)
        return jnp.sum(residual * residual)

    point = jnp.array([0.01, np.deg2rad(4.0)])
    active_pos = jnp.array([1.0, 0.0])
    active_rot = jnp.array([1.0, 0.0])
    gradient = np.asarray(jax.grad(objective)(point, active_pos, active_rot))
    step = 1e-4
    finite = np.array([(float(objective(point.at[i].add(step), active_pos, active_rot)) -
                        float(objective(point.at[i].add(-step), active_pos, active_rot))) / (2 * step)
                       for i in range(2)])
    relative_error = np.abs(gradient - finite) / np.maximum(1.0, np.abs(finite))
    if not np.all(relative_error < 2e-2) or not np.all(gradient > 0):
        raise ValueError(f"wrist active finite-difference test failed: {gradient}, {finite}")
    invalid_grad = np.asarray(jax.grad(objective)(point, jnp.zeros(2), jnp.zeros(2)))
    if not np.allclose(invalid_grad, 0, atol=1e-9):
        raise ValueError(f"invalid wrist target creates gradient: {invalid_grad}")
    only_pos = np.asarray(jax.grad(objective)(jnp.array([0.01, 0.0]), active_pos, jnp.zeros(2)))
    only_rot = np.asarray(jax.grad(objective)(jnp.array([0.0, np.deg2rad(4.0)]), jnp.zeros(2), active_rot))
    if abs(only_pos[1]) > 1e-5 or abs(only_rot[0]) > 1e-5:
        raise ValueError("position/rotation validity masks are coupled")
    return {"status": "PASS", "active_gradient": gradient.tolist(), "finite_difference": finite.tolist(),
            "relative_error": relative_error.tolist(), "invalid_gradient": invalid_grad.tolist(),
            "position_only_gradient": only_pos.tolist(), "rotation_only_gradient": only_rot.tolist(),
            "position_residual_scale_m": WRIST_POSITION_SCALE_M,
            "rotation_residual_scale_deg": 2.0}


def summary(values: np.ndarray, valid: np.ndarray) -> dict:
    selected = np.asarray(values)[valid]
    selected = selected[np.isfinite(selected)]
    return {"count": int(selected.size), "median": float(np.median(selected)) if selected.size else None,
            "p95": float(np.percentile(selected, 95)) if selected.size else None,
            "maximum": float(selected.max()) if selected.size else None}


def solve_case(core, solver, robot, assets, case: dict, output: Path) -> dict:
    import jax.numpy as jnp
    import jaxlie
    from chaoyang.pipeline.huro_frozen_targets_v2 import segmented_chunks
    from chaoyang.ops.run_huro_common_review_v2 import forward_points
    from chaoyang.pipeline.robot_renderer_eevee_fullchain import ARM_JOINT_NAMES

    hand, r0, valid = case["hand"], case["r0"], case["valid"]
    n = len(valid)
    inverse_placement = np.linalg.inv(r0["T_cam_base"])
    target_points = np.full((n, 2, 21, 3), np.nan)
    target_wrist = np.broadcast_to(np.eye(4), (n, 2, 4, 4)).copy()
    source_points = hand["joints21_camera"]
    target_points[valid] = source_points[valid] @ inverse_placement[:3, :3].T + inverse_placement[:3, 3]
    target_wrist[valid] = inverse_placement @ r0["T_target_root_cam"][valid]
    chunks = segmented_chunks(hand["frame_id"], hand["timestamp_ns"], valid,
                              max_frames=BLOCK_SIZE, max_gap_ns=MAX_GAP_NS)
    total_joints = len(robot.get_neutral_config())
    solved = np.full((n, total_joints), np.nan)
    local_links = np.concatenate((robot.left_local_link_indices, robot.right_local_link_indices))
    global_indices = np.array([0, 4, 8, 12, 16, 20, 21, 25, 29, 33, 37, 41])
    wrist_links = np.array([robot.left_eef_link_index, robot.right_eef_link_index])
    weights = dict(core.DEFAULT_WEIGHTS)
    weights["ego_view_rot"] = weights["ego_view_pos"] = 0.0
    rest = np.where(robot.hand_joint_mask > .5,
                    weights["rest_weight_default"] * weights["hand_rest_scale"],
                    weights["rest_weight_default"])
    blocks = []
    for block_id, indices in enumerate(chunks):
        count = len(indices)
        points = target_points[indices].reshape(count, 42, 3)
        mask = np.repeat(valid[indices], 21, axis=1)
        points = np.where(mask[..., None], points, 0.0)
        points = np.pad(points, ((0, BLOCK_SIZE - count), (0, 0), (0, 0)))
        mask = np.pad(mask, ((0, BLOCK_SIZE - count), (0, 0)))
        wrist = np.pad(target_wrist[indices], ((0, BLOCK_SIZE - count), (0, 0), (0, 0), (0, 0)))
        wrist[count:] = np.eye(4)
        position_mask = np.pad(valid[indices], ((0, BLOCK_SIZE - count), (0, 0))).astype(np.float32)
        rotation_mask = np.pad(case["rotation_valid"][indices], ((0, BLOCK_SIZE - count), (0, 0))).astype(np.float32)
        q, cost = solver(
            robot=robot.robot,
            local_keypoints=jnp.asarray(points, dtype=jnp.float32),
            local_link_indices=jnp.asarray(local_links), local_conn_mask=jnp.asarray(robot.conn_mask),
            local_kpt_mask=jnp.asarray(mask, dtype=jnp.float32),
            global_keypoints=jnp.asarray(points[:, global_indices], dtype=jnp.float32),
            global_link_indices=jnp.asarray(local_links[global_indices]),
            global_kpt_mask=jnp.asarray(mask[:, global_indices], dtype=jnp.float32),
            joint_mask=jnp.asarray(robot.joint_mask),
            initial_cfg=jnp.asarray(robot.get_neutral_config(), dtype=jnp.float32),
            weights=weights, rest_weight_per_joint=jnp.asarray(rest, dtype=jnp.float32),
            padding_mask=jnp.asarray(np.arange(BLOCK_SIZE) < count, dtype=jnp.float32),
            camera_link_index=robot.camera_link_index,
            target_cam_se3=jaxlie.SE3.from_matrix(jnp.broadcast_to(jnp.eye(4), (BLOCK_SIZE, 4, 4))),
            cam_mask=jnp.zeros(BLOCK_SIZE), hand_joint_mask=jnp.asarray(robot.hand_joint_mask),
            target_wrist_se3=jaxlie.SE3.from_matrix(jnp.asarray(wrist, dtype=jnp.float32)),
            wrist_pos_mask=jnp.asarray(position_mask), wrist_rot_mask=jnp.asarray(rotation_mask),
            wrist_link_indices=jnp.asarray(wrist_links),
        )
        q = np.asarray(q)[:count]
        if q.shape != (count, total_joints) or not np.isfinite(q).all() or not np.isfinite(float(cost)):
            raise ValueError(f"nonfinite HuRo core block {block_id}")
        solved[indices] = q
        blocks.append({"block": block_id, "frames": count, "source_indices": indices,
                       "solver_cost": float(cost), "convergence": "NOT_EXPOSED_BY_UPSTREAM"})
        print(json.dumps({"session": case["session_id"], "block": block_id + 1,
                          "blocks": len(chunks), "frames": count}), flush=True)
    names = list(robot.robot.joints.actuated_names)
    q_arm = np.full((n, 2, 7), np.nan)
    q_hand = np.full((n, 2, 22), np.nan)
    for side, model in enumerate((assets.left_hand, assets.right_hand)):
        arm_indices = [names.index(name) for name in ARM_JOINT_NAMES[side]]
        hand_indices = [names.index(j.name) for j in model.joints if j.joint_type != "fixed"]
        if len(hand_indices) != 22:
            raise ValueError("HuRo hand joint count drift")
        q_arm[valid[:, side], side] = solved[valid[:, side]][:, arm_indices]
        q_hand[valid[:, side], side] = solved[valid[:, side]][:, hand_indices]
    roots_base, points_base = forward_points(assets, q_arm, q_hand, valid, r0["T_flange_hand"])
    roots_cam = r0["T_cam_base"] @ roots_base
    points_cam = points_base @ r0["T_cam_base"][:3, :3].T + r0["T_cam_base"][:3, 3]
    position = np.full((n, 2), np.nan)
    rotation = np.full((n, 2), np.nan)
    for frame, side in np.argwhere(valid):
        delta = np.linalg.inv(r0["T_target_root_cam"][frame, side]) @ roots_cam[frame, side]
        position[frame, side] = np.linalg.norm(delta[:3, 3]) * 1000
        cosine = np.clip((np.trace(delta[:3, :3]) - 1) / 2, -1, 1)
        rotation[frame, side] = np.degrees(np.arccos(cosine))
    candidate = {
        "frame_id": hand["frame_id"], "timestamp_ns": hand["timestamp_ns"],
        "anatomical_side_names": hand["anatomical_side_names"], "target_valid": valid,
        "wrist_valid": valid, "finger_valid": valid, "q_arm": q_arm, "q_hand22": q_hand,
        "T_target_root_cam": r0["T_target_root_cam"], "T_actual_root_cam": roots_cam,
        "actual21_camera_m": points_cam, "T_cam_base": r0["T_cam_base"],
        "T_flange_hand": r0["T_flange_hand"], "human_to_physical": r0["human_to_physical"],
        "position_residual_mm": position, "rotation_residual_deg": rotation,
        "placement_policy": r0["placement_policy"], "legacy_candidate": np.asarray(False),
        "control_ground_truth": np.asarray(False), "training_eligible": np.asarray(False),
    }
    with (output / "HURO_CORE_V1.npz").open("xb") as stream:
        np.savez_compressed(stream, **candidate)
    target_error_r0 = np.linalg.norm(r0["actual21_camera_m"] - source_points, axis=-1) * 1000
    target_error_huro = np.linalg.norm(points_cam - source_points, axis=-1) * 1000
    common = valid & r0["wrist_valid"] & r0["finger_valid"]
    comparison = {
        "scope": "SAME_HAND_MOTION_ASSET_MOUNT_PLACEMENT_VALID_FRAMES_VISUAL_PROXY_ONLY",
        "session_id": case["session_id"], "common_valid_side_frames": common.sum(axis=0).tolist(),
        "R0_position_mm": summary(r0["position_residual_mm"], common),
        "HuRo_position_mm": summary(position, common),
        "R0_rotation_deg": summary(r0["rotation_residual_deg"], common),
        "HuRo_rotation_deg": summary(rotation, common),
        "R0_joint21_error_mm": summary(target_error_r0, np.broadcast_to(common[..., None], target_error_r0.shape)),
        "HuRo_joint21_error_mm": summary(target_error_huro, np.broadcast_to(common[..., None], target_error_huro.shape)),
        "solver_convergence": "UNKNOWN_NOT_EXPOSED_BY_UPSTREAM",
        "collision": "NOT_EVALUATED_BY_CORE_RUNNER",
        "quality_pass": False,
    }
    write_json(output / "COMPARISON.json", comparison)
    result = {"session_id": case["session_id"], "status": "CORE_EXECUTED_NUMERIC_UNREVIEWED",
              "frames": n, "blocks": blocks, "common_valid_side_frames": comparison["common_valid_side_frames"],
              "hand_motion": case["motion_ref"], "R0": case["r0_ref"],
              "core_motion": reference(output / "HURO_CORE_V1.npz"),
              "comparison": reference(output / "COMPARISON.json"),
              "same_asset_mount_placement": True, "training_eligible": False,
              "control_ground_truth": False, "quality_pass": False,
              "render_status": "PENDING_SHARED_SCENE_COMPOSITOR"}
    write_json(output / "RESULT.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for tag in ("007", "031"):
        parser.add_argument(f"--motion-{tag}", required=True, type=Path)
        parser.add_argument(f"--r0-{tag}", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--mode", choices=("canary", "full"), required=True)
    args = parser.parse_args()
    if os.environ.get("JAX_PLATFORMS") != "cpu" or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU_ONLY_ENV_REQUIRED")
    if Path(os.sys.executable).resolve() != HURO_ENV.resolve(strict=True):
        raise RuntimeError("PINNED_HURO_ENV_REQUIRED")
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    root = args.output_root.absolute()
    if root.parent.resolve(strict=True) != LANE.resolve(strict=True) or root.exists():
        raise ValueError("fresh direct child of V5 huro lane required")
    cases = [load_case("get_potato_chips_0915_007", args.motion_007, args.r0_007),
             load_case("play_cards_0915_031", args.motion_031, args.r0_031)]
    if not np.allclose(cases[0]["r0"]["T_flange_hand"], cases[1]["r0"]["T_flange_hand"], atol=1e-8):
        raise ValueError("two sessions use different mount")
    from chaoyang.ops.run_huro_fixed_placement_core_v2 import compose_with_mounts

    root.mkdir()
    core_path = REPO / "vendor/HuRo/pipeline/retargeting/retargeter.py"
    core, solver, wrist_residual, patched_source = adapted_core(core_path)
    write_json(root / "DIRECTIONAL_TEST.json", directional_tests(wrist_residual))
    with (root / "PATCHED_CORE_TRAJECTORY.py").open("x", encoding="utf-8") as stream:
        stream.write(patched_source)
    audit = {"core_source": reference(core_path), "patched_trajectory": reference(root / "PATCHED_CORE_TRAJECTORY.py"),
             "directional_test": reference(root / "DIRECTIONAL_TEST.json"),
             "wrist_position_scale_m": WRIST_POSITION_SCALE_M, "wrist_rotation_scale_deg": 2.0,
             "mode": args.mode, "gpu_used": False}
    write_json(root / "CORE_AUDIT.json", audit)
    if args.mode == "canary":
        write_json(root / "RESULT.json", {"status": "CANARY_DIRECTIONAL_TEST_PASS", "core_audit": reference(root / "CORE_AUDIT.json"),
                                           "real_session_executed": False, "control_ground_truth": False})
        print(json.dumps({"status": "CANARY_DIRECTIONAL_TEST_PASS", "output": str(root)}))
        return 0
    config, assets, _neutral = compose_with_mounts(REPO, root, cases[0]["r0"]["T_flange_hand"])
    robot = core.Retargeter(config)
    results = []
    for case in cases:
        output = root / case["session_id"]
        output.mkdir()
        results.append(solve_case(core, solver, robot, assets, case, output))
    final = {"status": "CORE_TWO_SESSION_NUMERIC_EXECUTED_PENDING_SHARED_RENDER",
             "sessions": [reference(root / result["session_id"] / "RESULT.json") for result in results],
             "core_audit": reference(root / "CORE_AUDIT.json"),
             "quality_pass": False, "training_eligible": False, "control_ground_truth": False, "gpu_used": False}
    write_json(root / "RESULT.json", final)
    print(json.dumps(final, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
