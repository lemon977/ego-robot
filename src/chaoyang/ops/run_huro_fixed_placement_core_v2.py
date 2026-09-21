"""Invoke pinned, unmodified HuRo trajectory core for a virtual Tianji/Kai rig.

This is a robot/target adapter, not the official full HuRo visual pipeline.
CPU backend only. No placement search, interpolation, model downloads or q clip.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import types
import xml.etree.ElementTree as ET


def ref(path):
    p = Path(path)
    return {"path": str(p.resolve()), "bytes": p.stat().st_size,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}


def verified_npz(reference):
    import io
    import numpy as np
    p = Path(reference["path"])
    before = p.stat()
    raw = p.read_bytes()
    after = p.stat()
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if identity(before) != identity(after) or len(raw) != reference["bytes"] or hashlib.sha256(raw).hexdigest() != reference["sha256"]:
        raise ValueError("input reference changed: " + str(p))
    with np.load(io.BytesIO(raw), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def dump(path, value):
    with Path(path).open("x", encoding="utf8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def compose_with_mounts(repo, output, flange_mounts):
    import numpy as np
    from scipy.spatial.transform import Rotation
    from chaoyang.pipeline.robot_renderer_eevee_fullchain import load_pinned_robot_assets, ARM_JOINT_NAMES
    from chaoyang.pipeline.robot_renderer_cycles import forward_kinematics
    from chaoyang.pipeline.robot_scene_state_cpu import _arm_limits
    from chaoyang.pipeline.huro_hand_only_retarget_v1 import _keypoint_links
    assets = load_pinned_robot_assets(repo)
    models = [assets.tianji, assets.left_hand, assets.right_hand]
    roots = [ET.parse(model.path).getroot() for model in models]
    combined = copy.deepcopy(roots[0])
    combined.set("name", "tianji_kai_virtual_huro_v2")
    for hand in roots[1:]:
        for child in hand:
            if child.tag in ("joint", "link"):
                combined.append(copy.deepcopy(child))
    # IK model uses identical kinematics, no visual/collision loading. The common
    # renderer and collision evaluator retain the original mesh-bearing URDFs.
    for link in combined.findall("link"):
        for child in list(link):
            if child.tag in ("visual", "collision"):
                link.remove(child)
    mounts = np.asarray(flange_mounts, dtype=np.float64)
    if mounts.shape != (2, 4, 4) or not np.isfinite(mounts).all():
        raise ValueError("two finite fixed flange transforms required")
    for side, suffix in enumerate(("L", "R")):
        mount = mounts[side]
        if not np.allclose(mount[3], [0, 0, 0, 1]) or not np.allclose(mount[:3,:3].T@mount[:3,:3], np.eye(3), atol=1e-7) or np.linalg.det(mount[:3,:3]) < .999999:
            raise ValueError("mount must be proper SE3")
        joint = ET.SubElement(combined, "joint", name=f"virtual_flange_mount_{suffix}", type="fixed")
        ET.SubElement(joint, "parent", link=f"flange_{suffix}")
        ET.SubElement(joint, "child", link=f"hand_{suffix.lower()}_base_link")
        ET.SubElement(joint, "origin", xyz=" ".join(map(str,mount[:3,3])),
                      rpy=" ".join(map(str,Rotation.from_matrix(mount[:3,:3]).as_euler("xyz"))))
    urdf = output / "TIANJI_KAI_KINEMATIC_COMBINED.urdf"
    with urdf.open("xb") as f:
        f.write(ET.tostring(combined, encoding="utf-8", xml_declaration=True))
    lo, hi = _arm_limits(assets)
    neutral_arm = (lo+hi)/2
    groups = {"left_arm": list(ARM_JOINT_NAMES[0]), "right_arm": list(ARM_JOINT_NAMES[1])}
    home = {name: float(v) for side in range(2) for name,v in zip(ARM_JOINT_NAMES[side], neutral_arm[side])}
    for side, model in zip(("left", "right"), models[1:]):
        moving = [j for j in model.joints if j.joint_type != "fixed"]
        groups[side+"_hand"] = [j.name for j in moving]
        home.update({j.name: float((j.lower+j.upper)/2) for j in moving})
    child_links = {j.find("child").attrib["link"] for j in combined.findall("joint")}
    base_links = {x.attrib["name"] for x in combined.findall("link")} - child_links
    if len(base_links) != 1:
        raise ValueError("combined robot must have exactly one root")
    cfg = {"joints": groups, "hand_groups": ["left_hand", "right_hand"], "stiff_groups": [],
           "keypoint_mapping": {s: dict(enumerate(_keypoint_links(s))) for s in ("left","right")},
           "camera_link": next(iter(base_links)),
           "eef_link_names": {"left": "hand_l_base_link", "right": "hand_r_base_link"},
           "eef_hand_joint_groups": {"left": "left_hand", "right": "right_hand"}, "home_config": home}
    dump(output / "ROBOT_CONFIG.json", cfg)
    arm_fk = forward_kinematics(assets.tianji, {j.name: home.get(j.name,0.) for j in assets.tianji.joints if j.joint_type != "fixed"})
    neutral_roots = np.stack([arm_fk[f"flange_{s}"]@mounts[i] for i,s in enumerate(("L","R"))])
    dump(output / "COMBINED_ASSET_MANIFEST.json", {"sources": [ref(m.path) for m in models],
         "combined": ref(urdf), "mount_definition": "T_flange_hand, NOT T_tool_hand",
         "flange_mounts": mounts.tolist(), "neutral_roots": neutral_roots.tolist(),
         "visual_collision_elements_removed_for_ik_only": True, "physical_mount_accuracy": "UNVERIFIED"})
    return types.SimpleNamespace(config=cfg, urdf_path=str(urdf)), assets, neutral_roots


def load_core(source):
    from chaoyang.pipeline.huro_core_adapter_v2 import CORE_SOURCE_SHA
    if hashlib.sha256(source.read_bytes()).hexdigest() != CORE_SOURCE_SHA:
        raise ValueError("official core source SHA drift")
    spec = importlib.util.spec_from_file_location("chaoyang_pinned_official_huro_core", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def core_call(core, retargeter, points, mask, initial=None):
    """Original solve_retargeting, fixed base targets and wrist+tip global map."""
    import numpy as np
    import jax.numpy as jnp
    import jaxlie
    t = len(points)
    cfg = retargeter.get_neutral_config() if initial is None else initial
    local_links = np.concatenate([retargeter.left_local_link_indices,retargeter.right_local_link_indices])
    global_points = np.array([0,4,8,12,16,20,21,25,29,33,37,41])
    global_links = local_links[global_points]
    weights = dict(core.DEFAULT_WEIGHTS)
    weights["ego_view_rot"] = weights["ego_view_pos"] = 0.0
    rest = np.where(retargeter.hand_joint_mask > .5,
                    weights["rest_weight_default"]*weights["hand_rest_scale"],weights["rest_weight_default"])
    # Unknown coords are finite computation placeholders with mask zero only;
    # outputs for unknown sides remain invalid/NaN in the exported artifact.
    finite_points = np.where(mask[...,None], points, 0.)
    if not np.isfinite(finite_points).all():
        raise ValueError("valid target contains nonfinite coordinates")
    solved, cost = core.solve_retargeting(robot=retargeter.robot,
        local_keypoints=jnp.asarray(finite_points,dtype=jnp.float32),
        local_link_indices=jnp.asarray(local_links),local_conn_mask=jnp.asarray(retargeter.conn_mask),
        local_kpt_mask=jnp.asarray(mask,dtype=jnp.float32),
        global_keypoints=jnp.asarray(finite_points[:,global_points],dtype=jnp.float32),
        global_link_indices=jnp.asarray(global_links),global_kpt_mask=jnp.asarray(mask[:,global_points],dtype=jnp.float32),
        joint_mask=jnp.asarray(retargeter.joint_mask),initial_cfg=jnp.asarray(cfg,dtype=jnp.float32),
        weights=weights,rest_weight_per_joint=jnp.asarray(rest,dtype=jnp.float32),
        padding_mask=jnp.ones(t),camera_link_index=retargeter.camera_link_index,
        target_cam_se3=jaxlie.SE3.from_matrix(jnp.tile(jnp.eye(4),(t,1,1))),cam_mask=jnp.zeros(t),
        hand_joint_mask=jnp.asarray(retargeter.hand_joint_mask))
    return np.asarray(solved), float(cost)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    if os.environ.get("JAX_PLATFORMS") != "cpu" or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU_ONLY_ENV_REQUIRED")
    allowed = sorted(os.sched_getaffinity(0))[:2]
    os.sched_setaffinity(0, allowed)
    output = Path(config["output"])
    lane = Path(config["lane_root"]).resolve()
    if not output.resolve().is_relative_to(lane) or output.exists():
        raise ValueError("output must be a new child of own lane")
    output.mkdir(parents=True)
    import numpy as np
    import jax
    import jax.numpy as jnp
    import jaxlie
    from chaoyang.pipeline.robot_renderer_cycles import forward_kinematics
    repo = Path(config["repo"])
    robot_cfg, assets, neutral_roots = compose_with_mounts(repo, output, config["T_flange_hand"])
    core = load_core(repo / "vendor/HuRo/pipeline/retargeting/retargeter.py")
    robot = core.Retargeter(robot_cfg)
    q0 = robot.get_neutral_config()
    jf = np.asarray(jaxlie.SE3(robot.robot.forward_kinematics(jnp.asarray(q0))).as_matrix())
    names = list(robot.robot.links.names)
    home = robot_cfg.config["home_config"]
    residuals = []
    for side, model in enumerate((assets.left_hand, assets.right_hand)):
        handfk = forward_kinematics(model,{j.name: home[j.name] for j in model.joints if j.joint_type != "fixed"})
        for name, transform in handfk.items():
            expected = neutral_roots[side] @ transform
            residuals.append(float(np.max(np.abs(jf[names.index(name)]-expected))))
    max_error = max(residuals)
    if max_error > 2e-5:
        raise RuntimeError(f"combined FK parity failed: {max_error}")
    links = np.concatenate([robot.left_local_link_indices, robot.right_local_link_indices])
    # Known-answer reachable synthetic target using the real 58-DOF robot.
    target = jf[links,:3,3][None]
    started = time.monotonic()
    q, cost = core_call(core,robot,target,np.ones((1,42),bool))
    elapsed = time.monotonic()-started
    if q.shape != (1,len(q0)) or not np.isfinite(q).all() or not np.isfinite(cost):
        raise RuntimeError("official core returned invalid output")
    np.savez_compressed(output / "SMOKE_OUTPUT.npz", q=q, target=target, initial_q=q0)
    result = {"status": "OFFICIAL_CORE_CPU_SMOKE_EXECUTED", "official_core_executed": True,
              "upstream_function": "solve_retargeting", "upstream_source": ref(repo/"vendor/HuRo/pipeline/retargeting/retargeter.py"),
              "fk_parity_max_abs": max_error, "elapsed_seconds": elapsed, "solver_cost": cost,
              "solved_shape": list(q.shape), "cpu_affinity": allowed,
              "jax_backend": jax.default_backend(), "gpu_used": False, "placement_optimized": False,
              "global_target_mapping": "wrist plus five tips per hand (adapter configuration)",
              "real_session_executed": False, "numeric_quality_pass": False, "training_eligible": False,
              "control_ground_truth": False, "video_count": 0,
              "temporal_policy": "SMOKE_SINGLE_FRAME; actual-session gap/dt adapter not yet run"}
    dump(output / "RESULT.json", result)
    print(json.dumps(result,indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
