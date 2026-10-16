"""Frozen Tianji model mathematics, not measured geometry or device control.

The R1 backend is reused verbatim, lazily, in an isolated CPU environment. Its
degrees/mm/model-world API is converted explicitly to radians/metres/per-arm
base. One solve is exactly one original differential-IK step, never an automatic
retry or gate relaxation. All modeled non-adjacent collision geometry is checked
by MuJoCo's actual collision pass; the original two QP pairs additionally retain
their original minimum distance. Missing Revo2/scene geometry remains UNKNOWN.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import types
from typing import Mapping
import xml.etree.ElementTree as ET

import numpy as np

from .bridge import IKSolution, JOINT_NAMES, _array, _se3

BACKEND_SHA256 = "05a8d6e147f2ce3d585287a46ef7548bbbca382442e26a177098ce59a9c1b6fc"
CLIENT_SHA256 = "6ea14f756bfbe428914e78b4d0fbb65e586e3a0d22d729a99559554dc9f98c91"
DEPENDENCIES = {"numpy": "1.26.4", "mujoco": "3.9.0", "mink": "1.2.0",
                "qpsolvers": "4.12.0", "daqp": "0.8.2", "PyYAML": "6.0.3"}
FRAME = "PER_ARM_BASE_FROM_MODEL_TOOL"
ENGINEERING_LIMIT_SOURCE = "NEW_OFFLINE_ENGINEERING_ASSUMPTION_NOT_EXISTING_OR_PHYSICAL_GATE"
ORIGINAL_COLLISION_MODE = "ORIGINAL_CAPSULE_V1"
MESH_COLLISION_MODE = "FROZEN_STL_MESH_AUDIT_V1"
ORIGINAL_INITIALIZATION_MODE = "STATIC_NUMERICAL_CURRENT_TARGET_ONLY"
FRAME_TASK_INITIALIZATION_MODE = "STATIC_FRAME_TASK_CURRENT_TARGET_ONLY"
STRICT_TRAINING_ADMISSION = "STRICT_MODEL_NUMERICS_V1"
EGO_TRAINING_ADMISSION = "EGO_DERIVED_STRUCTURAL_V1"
EGO_PROJECTION_POLICY = "FINITE_IK_SOFTWARE_BOUNDS_BOX_PROJECTION_WITH_RECOMPUTED_DIAGNOSTICS"
STATIC_FRAME_TASK_PARAMETERS = {"position_cost": 100.0,
    "orientation_cost": 11.459155902616466, "solve_damping": 1e-6,
    "regularizer": "NO_POSTURE_DAMPING_OR_KINETIC_TASK",
    "limits": "ORIGINAL_CONFIGURATION_VELOCITY_AND_QP_COLLISION_LIMITS_UNCHANGED"}


class OfflineMinkError(ValueError):
    pass


class GeometryCallBudget:
    """Shared finite *attempted* backend/native geometry calls, including failures.

    Composite backend construction/solve and MuJoCo collision passes count once;
    their internal native work is not a count of learned-policy forwards.
    The default cap stays 1000; larger, explicitly signed budgets may select a
    read-only cap up to 5000 without changing any geometry or quality gate.
    """
    __slots__ = ("_limit", "_hard_limit", "used")

    def __init__(self, limit=1000, *, hard_limit=1000):
        if (type(hard_limit) is not int or not 0 < hard_limit <= 5000
                or type(limit) is not int or not 0 < limit <= hard_limit):
            raise OfflineMinkError("FINITE_GEOMETRY_BUDGET_REQUIRED")
        self._limit, self._hard_limit, self.used = limit, hard_limit, 0

    @property
    def limit(self):
        return self._limit

    @property
    def hard_limit(self):
        return self._hard_limit

    def claim(self, count=1):
        if type(count) is not int or count <= 0 or self.used + count > self.limit:
            raise OfflineMinkError("GEOMETRY_CALL_BUDGET_EXHAUSTED")
        self.used += count


def _project_path(root, value):
    path = Path(value)
    path = path if path.is_absolute() else root / path
    if path.absolute() != path.resolve(strict=True):
        raise OfflineMinkError("SOURCE_SYMLINK_OR_NONCANONICAL_PATH")
    path.resolve().relative_to(root)
    if not path.is_file():
        raise OfflineMinkError("SOURCE_FILE_REQUIRED")
    return path


def _sha(value):
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise OfflineMinkError("EXACT_SOURCE_SHA_REQUIRED")
    return value


def _read_verified(path, expected):
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if identity(before) != identity(after) or hashlib.sha256(raw).hexdigest() != _sha(expected):
        raise OfflineMinkError(f"SOURCE_SHA_DRIFT:{path.name}")
    return raw, identity(after)


def _client_step(source):
    values = []
    for node in ast.walk(ast.parse(source)):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument" and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "--tianji-arm-step-rad"):
            try:
                values.extend(ast.literal_eval(k.value) for k in node.keywords if k.arg == "default")
            except (ValueError, TypeError) as error:
                raise OfflineMinkError("FROZEN_CLIENT_STEP_DEFAULT_NOT_LITERAL") from error
    if len(values) != 1 or not np.isfinite(values[0]) or values[0] <= 0:
        raise OfflineMinkError("FROZEN_CLIENT_STEP_DEFAULT_ABSENT")
    return float(values[0])


def _model_to_arm(poses_mm, world_base):
    poses = _se3(poses_mm, (2, 4, 4), "model_world_from_tool_mm")
    poses[:, :3, 3] /= 1000.0
    return np.linalg.inv(world_base) @ poses


def _arm_to_model(poses_m, world_base):
    poses = world_base @ _se3(poses_m, (2, 4, 4), "per_arm_base_from_tool_m")
    poses[:, :3, 3] *= 1000.0
    return poses


def _collision_pairs(model):
    """Enumerate every enabled pair, excluding welded/adjacent topology only."""
    geoms = [g for g in range(model.ngeom)
             if int(model.geom_contype[g]) or int(model.geom_conaffinity[g])]
    pairs, excluded = [], []
    signatures = {int(x) for x in model.exclude_signature}
    for index, a in enumerate(geoms):
        for b in geoms[index + 1:]:
            if not ((int(model.geom_contype[a]) & int(model.geom_conaffinity[b]))
                    or (int(model.geom_contype[b]) & int(model.geom_conaffinity[a]))):
                continue
            ba, bb = int(model.geom_bodyid[a]), int(model.geom_bodyid[b])
            wa, wb = int(model.body_weldid[ba]), int(model.body_weldid[bb])
            pa, pb = int(model.body_parentid[wa]), int(model.body_parentid[wb])
            reason = None
            if wa == wb:
                reason = "SAME_WELDED_BODY"
            elif ba != 0 and bb != 0 and (int(model.body_weldid[pa]) == wb
                                         or int(model.body_weldid[pb]) == wa):
                reason = "ADJACENT_JOINT"
            elif (min(ba, bb) << 16) + max(ba, bb) in signatures:
                reason = "MODEL_EXPLICIT_EXCLUDE"
            if reason:
                excluded.append((a, b, reason))
            else:
                pairs.append((a, b))
    return geoms, pairs, excluded


class OfflineMinkProvider:
    offline_only = True
    ik_frame_id = FRAME
    tool_frame_ids = ("left_tool_site", "right_tool_site")

    def __init__(self, *, project_root, source_index_path, expected_index_sha256,
                 client_source_path, budget=None, collision_validation=ORIGINAL_COLLISION_MODE,
                 current_initialization_mode=ORIGINAL_INITIALIZATION_MODE,
                 training_admission_policy=STRICT_TRAINING_ADMISSION):
        if (type(collision_validation) is not str
                or collision_validation not in (ORIGINAL_COLLISION_MODE, MESH_COLLISION_MODE)):
            raise OfflineMinkError("EXPLICIT_SUPPORTED_COLLISION_VALIDATION_REQUIRED")
        if (type(current_initialization_mode) is not str or current_initialization_mode not in (
                ORIGINAL_INITIALIZATION_MODE, FRAME_TASK_INITIALIZATION_MODE)):
            raise OfflineMinkError("EXPLICIT_SUPPORTED_CURRENT_INITIALIZATION_REQUIRED")
        if (type(training_admission_policy) is not str or training_admission_policy not in (
                STRICT_TRAINING_ADMISSION, EGO_TRAINING_ADMISSION)):
            raise OfflineMinkError("EXPLICIT_SUPPORTED_TRAINING_ADMISSION_REQUIRED")
        if (training_admission_policy == EGO_TRAINING_ADMISSION
                and collision_validation != ORIGINAL_COLLISION_MODE):
            raise OfflineMinkError("EGO_TRAINING_USES_ORIGINAL_CAPSULE_DIAGNOSTICS_ONLY")
        self.collision_validation = collision_validation
        self.current_initialization_mode = current_initialization_mode
        self.training_admission_policy = training_admission_policy
        self.root = Path(project_root).resolve(strict=True)
        self.budget = budget if budget is not None else GeometryCallBudget()
        if not isinstance(self.budget, GeometryCallBudget):
            raise OfflineMinkError("SHARED_GEOMETRY_BUDGET_REQUIRED")
        self._sources, self._identities = {}, {}
        index_path = _project_path(self.root, source_index_path)
        index_raw, identity = _read_verified(index_path, expected_index_sha256)
        self._identities[index_path] = identity
        index = json.loads(index_raw)
        if index.get("schema") != "revo2-static-geometry-full-source-index-v1":
            raise OfflineMinkError("FROZEN_R1_SOURCE_INDEX_REQUIRED")
        entries = index.get("sources", [])
        lookup = {}
        for entry in entries:
            if entry.get("role") in {"MUJOCO_SCENE_OR_MODEL_XML", "REQUIRED_MUJOCO_MESH"} or (
                    entry.get("copy_path", "").endswith(("/mink_ik_backend.py", "/config/teleop.yaml"))):
                path = _project_path(self.root, entry["copy_path"])
                if path in lookup:
                    raise OfflineMinkError("DUPLICATE_FROZEN_SOURCE")
                raw, stat = _read_verified(path, entry["copy_sha256"])
                if len(raw) != entry["copy_bytes"]:
                    raise OfflineMinkError("SOURCE_BYTE_COUNT_DRIFT")
                self._sources[path], self._identities[path] = raw, stat
                lookup[path] = entry
        pick = lambda suffix: [p for p in lookup if str(p).endswith(suffix)]
        for attr, suffix in (("backend_path", "/mink_ik_backend.py"),
                             ("scene_path", "/teleop_scene_pro.xml"),
                             ("config_path", "/config/teleop.yaml")):
            found = pick(suffix)
            if len(found) != 1:
                raise OfflineMinkError(f"UNIQUE_FROZEN_RESOURCE_REQUIRED:{suffix}")
            setattr(self, attr, found[0])
        if hashlib.sha256(self._sources[self.backend_path]).hexdigest() != BACKEND_SHA256:
            raise OfflineMinkError("ONLY_FROZEN_CPU_BACKEND_ALLOWED")
        xmls = [p for p in lookup if lookup[p]["role"] == "MUJOCO_SCENE_OR_MODEL_XML"]
        referenced_meshes = set()
        for path in xmls:
            for node in ET.fromstring(self._sources[path]).iter():
                if "file" in node.attrib:
                    ref = (path.parent / node.attrib["file"]).resolve(strict=True)
                    if node.tag not in {"include", "mesh"} or ref not in lookup:
                        raise OfflineMinkError("UNBOUND_XML_FILE_DEPENDENCY")
                    if node.tag == "mesh":
                        referenced_meshes.add(ref)
        if len(referenced_meshes) != 17:
            raise OfflineMinkError("FROZEN_SEVENTEEN_MESH_CLOSURE_REQUIRED")
        self.client_path = _project_path(self.root, client_source_path)
        raw, stat = _read_verified(self.client_path, CLIENT_SHA256)
        self._sources[self.client_path], self._identities[self.client_path] = raw, stat
        self.client_step_rad = _client_step(raw.decode("utf-8"))
        source_receipts = [{"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
                            "bytes": len(raw)} for path, raw in sorted(self._sources.items())]
        self.source_receipts = source_receipts
        self.signature = hashlib.sha256(json.dumps({"sources": source_receipts,
            "index_sha256": expected_index_sha256, "dependencies": DEPENDENCIES,
            "frame": FRAME, "collision_validation": self.collision_validation,
            "current_initialization_mode": self.current_initialization_mode,
            "training_admission_policy": self.training_admission_policy,
            "ego_projection_policy": EGO_PROJECTION_POLICY,
            "static_frame_task_parameters": STATIC_FRAME_TASK_PARAMETERS,
            "mesh_implementation_sha256": (hashlib.sha256(Path(__file__).with_name(
                "offline_mesh_collision_v1.py").read_bytes()).hexdigest()
                if self.collision_validation == MESH_COLLISION_MODE else None),
            "implementation_sha256": hashlib.sha256(
                Path(__file__).read_bytes()).hexdigest()}, sort_keys=True).encode()).hexdigest()
        self._backend, self._mujoco, self._base = None, None, None
        self._mesh_audit, self._static_hand_tasks = None, None
        self._mesh_initialization_error = None

    def _assert_frozen(self):
        for path, expected in self._identities.items():
            s = path.stat()
            actual = (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
            if actual != expected:
                raise OfflineMinkError(f"SOURCE_IDENTITY_DRIFT:{path.name}")

    def _ensure_backend(self):
        self._assert_frozen()
        if self._backend is not None:
            self._ensure_selected_collision_layer()
            return
        if os.environ.get("MUJOCO_GL") != "disable":
            raise OfflineMinkError("EXPLICIT_CPU_MUJOCO_GL_DISABLE_REQUIRED")
        for name, version in DEPENDENCIES.items():
            try:
                actual = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError as error:
                raise OfflineMinkError(f"ISOLATED_DEPENDENCY_ABSENT:{name}") from error
            if actual != version:
                raise OfflineMinkError(f"FROZEN_DEPENDENCY_VERSION_MISMATCH:{name}")
        import mujoco
        import yaml
        cfg = yaml.safe_load(self._sources[self.config_path])["arm_teleop"]
        names = {"speed_limit_scale": "mink_speed_scale", "orientation_cost": "mink_orientation_cost",
            "posture_cost": "mink_posture_cost", "j3_posture_cost": "mink_j3_posture_cost",
            "j3_target_deg": "mink_j3_target_deg", "collision_min_dist": "mink_collision_min_dist",
            "collision_detect_dist": "mink_collision_detect_dist", "j2_upper_deg": "mink_j2_upper_deg",
            "j4_upper_deg": "mink_j4_upper_deg", "joint_margin_deg": "mink_joint_margin_deg"}
        self.backend_parameters = {key: cfg[value] for key, value in names.items()}
        # These are original verified backend defaults, not new tuned values.
        self.backend_parameters.update(position_cost=1.0, solver="daqp", solve_damping=0.1)
        module = types.ModuleType("verified_tianji_cpu_geometry_backend")
        module.__file__ = str(self.backend_path)
        exec(compile(self._sources[self.backend_path], str(self.backend_path), "exec"), module.__dict__)
        self.budget.claim()
        self._backend = module.MinkIKBackend(scene_path=str(self.scene_path), **self.backend_parameters)
        self._backend_module, self._mujoco = module, mujoco
        if self._backend.model.nq != 14 or self._backend.model.nv != 14:
            raise OfflineMinkError("FROZEN_FOURTEEN_JOINT_MODEL_REQUIRED")
        base = np.repeat(np.eye(4)[None], 2, axis=0)
        for side, name in enumerate(("Base_L", "Base_R")):
            bid = mujoco.mj_name2id(self._backend.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                raise OfflineMinkError("PER_ARM_BASE_BODY_ABSENT")
            base[side, :3, :3] = self._backend.data.xmat[bid].reshape(3, 3)
            base[side, :3, 3] = self._backend.data.xpos[bid]
        self._base = _se3(base, (2, 4, 4), "nominal_model_world_from_arm_base")
        self._collision_data = mujoco.MjData(self._backend.model)
        self._geom_ids, self._pairs, self._excluded = _collision_pairs(self._backend.model)
        self._pair_set = set(self._pairs)
        self._qp_pairs = [(mujoco.mj_name2id(self._backend.model, mujoco.mjtObj.mjOBJ_GEOM, a),
                           mujoco.mj_name2id(self._backend.model, mujoco.mjtObj.mjOBJ_GEOM, "base_link_collision"))
                          for a in ("Link5_L_collision", "Link5_R_collision")]
        if any(min(pair) < 0 for pair in self._qp_pairs):
            raise OfflineMinkError("ORIGINAL_QP_COLLISION_GEOMETRY_ABSENT")
        covered = {mujoco.mj_id2name(self._backend.model, mujoco.mjtObj.mjOBJ_BODY,
                                  int(self._backend.model.geom_bodyid[g])) for g in self._geom_ids}
        required = {f"Link{i}_{side}" for side in ("L", "R") for i in range(1, 8)} | {"base_link", "Base_L", "Base_R"}
        self._collision_coverage_complete = required <= covered and bool(self._pairs)
        self._ensure_selected_collision_layer()

    def _ensure_selected_collision_layer(self):
        if self.collision_validation != MESH_COLLISION_MODE or self._mesh_audit is not None:
            return
        if self._mesh_initialization_error is not None:
            raise self._mesh_initialization_error
        from .offline_mesh_collision_v1 import OfflineMeshCollisionAudit
        try:
            self._mesh_audit = OfflineMeshCollisionAudit(self)
        except Exception as error:
            self._mesh_initialization_error = error
            raise

    @property
    def model_world_from_arm_base(self):
        self._ensure_backend()
        return self._base.copy()

    @property
    def collision_inventory(self):
        self._ensure_backend()
        model, mj = self._backend.model, self._mujoco
        body = lambda g: mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g]))
        def category(a, b):
            names = (body(a), body(b))
            if "world" in names:
                return "MODELED_FLOOR_OR_WORLD"
            if any(n.endswith("_L") for n in names) and any(n.endswith("_R") for n in names):
                return "CROSS_ARM"
            if any(n in {"base_link", "Base_L", "Base_R"} for n in names):
                return "ARM_BODY"
            return "SAME_ARM_NONADJACENT"
        return {"scope": "ALL_ENABLED_MODELED_NONADJACENT_GEOMETRY_ONLY",
            "geometry_representation": "FROZEN_XML_COLLISION_PRIMITIVES_NOT_VISUAL_MESH_SURFACES",
            "geometries": [{"id": g, "name": mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, g),
                "body": body(g), "mujoco_type": int(model.geom_type[g]),
                "contype": int(model.geom_contype[g]), "conaffinity": int(model.geom_conaffinity[g])}
                for g in self._geom_ids],
            "pairs": [{"geom_ids": [a, b], "body_names": [body(a), body(b)],
                "category": category(a, b)} for a, b in self._pairs],
            "excluded_pairs": [{"geom_ids": [a, b], "body_names": [body(a), body(b)],
                "reason": reason} for a, b, reason in self._excluded],
            "exclusion_policy": "SAME_WELDED_BODY_OR_ADJACENT_JOINT_OR_MODEL_EXPLICIT_EXCLUDE; WORLD_IS_NOT_A_JOINT_NEIGHBOR",
            "coverage_complete_for_existing_collision_model": bool(self._collision_coverage_complete),
            "physical_geometry_qualified": False,
            "unmodeled_geometry": ["Revo2 hands/fingers", "real surrounding scene"]}

    @property
    def synthetic_home_q_rad(self):
        self._ensure_backend()
        # This is the frozen XML keyframe, never current hardware feedback.
        key = self._mujoco.mj_name2id(self._backend.model, self._mujoco.mjtObj.mjOBJ_KEY, "home")
        return self._backend.model.key_qpos[key].copy()

    def fk(self, q_rad):
        q = _array(q_rad, (14,), "synthetic_q_rad")
        self._ensure_backend()
        poses = []
        for side in ("A", "B"):
            self.budget.claim()
            poses.append(self._backend.fk_mm(side)(np.degrees(q)))
        return _model_to_arm(np.stack(poses), self._base)

    def _check_collisions(self, q):
        checked, free, legacy = self._check_legacy_collisions(q)
        if self.collision_validation == ORIGINAL_COLLISION_MODE:
            return checked, free, legacy
        self._ensure_selected_collision_layer()
        mesh_checked, mesh_free, mesh = self._mesh_audit.check(q, self._collision_data)
        selected_checked = bool(checked and mesh_checked)
        distances = legacy["original_qp_distances_m_capped_at_detection_horizon"]
        qp_passed = bool(np.isfinite(distances).all() and
                         min(distances) >= legacy["original_qp_minimum_distance_m"])
        selected_free = bool(selected_checked and mesh_free and qp_passed)
        return selected_checked, selected_free, {**legacy, **mesh,
            "collision_checked": selected_checked, "collision_free": selected_free,
            "legacy_capsule_diagnostics": {**legacy, "collision_checked": bool(checked),
                                            "collision_free": bool(free)},
            "mesh_diagnostics": mesh, "original_qp_margin_passed": qp_passed,
            "selected_rule": "FULL_MESH_NONPENETRATION_AND_ORIGINAL_QP_MARGIN",
            "selected_nonpenetration_passed": bool(mesh_free)}

    def _check_legacy_collisions(self, q):
        model, data, mj = self._backend.model, self._collision_data, self._mujoco
        data.qpos[:] = q
        self.budget.claim()
        mj.mj_forward(model, data)  # actual enabled-pair broad/narrow collision pass
        contacts = []
        contact_numerics_finite = True
        for contact in data.contact:
            contact_numerics_finite &= bool(np.isfinite(float(contact.dist)))
            pair = tuple(sorted((int(contact.geom1), int(contact.geom2))))
            if pair in self._pair_set and float(contact.dist) < 0:
                contacts.append({"geom_ids": list(pair), "body_names": [mj.mj_id2name(
                    model, mj.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) for g in pair],
                    "signed_distance_m": float(contact.dist)})
        distances = []
        margin = float(self.backend_parameters["collision_min_dist"])
        horizon = float(self.backend_parameters["collision_detect_dist"])
        for a, b in self._qp_pairs:
            self.budget.claim()
            distance = float(mj.mj_geomDistance(model, data, a, b, horizon, None))
            distances.append(distance)
        native_collision_enabled = not bool(int(model.opt.disableflags)
            & int(mj.mjtDisableBit.mjDSBL_CONTACT))
        checked = (self._collision_coverage_complete and native_collision_enabled
                   and contact_numerics_finite and np.isfinite(distances).all())
        free = bool(checked and not contacts and min(distances) >= margin)
        inventory = self.collision_inventory
        categories = {name: sum(p["category"] == name for p in inventory["pairs"])
                      for name in sorted({p["category"] for p in inventory["pairs"]})}
        return bool(checked), free, {"scope": "ALL_ENABLED_MODELED_NONADJACENT_GEOMETRY_ONLY",
            "enabled_geom_count": len(self._geom_ids), "eligible_pair_count": len(self._pairs),
            "excluded_topology_pair_count": len(self._excluded), "original_qp_pair_count": 2,
            "original_qp_distances_m_capped_at_detection_horizon": distances,
            "original_qp_minimum_distance_m": margin, "detection_horizon_m": horizon,
            "actual_nonadjacent_penetration_contacts": contacts,
            "native_contact_calculation_enabled": native_collision_enabled,
            "contact_numerics_finite": contact_numerics_finite,
            "full_pair_test": "MUJOCO_ENABLED_COLLISION_PASS_NONPENETRATION",
            "pair_category_counts": categories,
            "collision_inventory_sha256": hashlib.sha256(json.dumps(
                inventory, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "full_pair_penetration_threshold_m": 0.0,
            "physical_scene_collision_status": "UNKNOWN",
            "unmodeled_geometry": ["Revo2 hands/fingers", "real surrounding scene"],
            "collision_is_physical_safety": False}

    def solve(self, target_poses, seed_q_rad, dt_s):
        targets = _se3(target_poses, (2, 4, 4), "per_arm_base_from_model_tool_m")
        seed = _array(seed_q_rad, (14,), "synthetic_seed_q_rad")
        if not np.isfinite(dt_s) or dt_s <= 0:
            raise OfflineMinkError("POSITIVE_FINITE_NATIVE_DT_REQUIRED")
        self._ensure_backend()
        model_target_mm = _arm_to_model(targets, self._base)
        self.budget.claim()
        self._backend.sync_config(np.degrees(seed), reset_posture_target=True)
        for side, target in zip(("A", "B"), model_target_mm):
            self.budget.claim()
            self._backend.set_target(side, target)
        self.budget.claim()
        q = np.radians(_array(self._backend.solve(float(dt_s)), (14,), "backend_q_deg"))
        checked, free, collision = self._check_collisions(q)
        recovered = self.fk(q)
        relative = np.swapaxes(targets[:, :3, :3], -1, -2) @ recovered[:, :3, :3]
        rotation_error = np.arccos(np.clip((np.trace(relative, axis1=-2, axis2=-1)-1)/2, -1, 1))
        return IKSolution(q, checked, free, {"mode": "ASSUMPTION_DRIVEN_OFFLINE",
            "provider_sha256": self.signature, "frame": FRAME, "position_unit": "meters",
            "joint_unit": "radians", "original_backend_position_unit": "millimeters",
            "original_backend_joint_unit": "degrees", "differential_steps": 1,
            "position_error_m": np.linalg.norm(recovered[:, :3, 3]-targets[:, :3, 3], axis=1).tolist(),
            "rotation_error_rad": rotation_error.tolist(), "collision": collision,
            "geometry_calls_used": self.budget.used, "measured": False,
            "geometry_qualified": False, "physical_deployment_authorized": False,
            "control_ground_truth": False, "robot_commands_sent": 0})

    def solve_static(self, target_poses, seed_q_rad, dt_s):
        """One current-only numerical iteration; never the future teleop solver.

        Task scales normalize the unchanged 1cm/5deg absolute FK gates. Original
        joint/velocity/QP-collision limits remain in force; no posture or kinetic
        objective competes with this static pose fit. The caller bounds iteration
        count and only admits the final pose after the original independent gates.
        """
        if self.current_initialization_mode != FRAME_TASK_INITIALIZATION_MODE:
            raise OfflineMinkError("STATIC_FRAME_TASK_CURRENT_MODE_REQUIRED")
        targets = _se3(target_poses, (2, 4, 4), "per_arm_base_from_model_tool_m")
        seed = _array(seed_q_rad, (14,), "synthetic_seed_q_rad")
        if not np.isfinite(dt_s) or dt_s <= 0:
            raise OfflineMinkError("POSITIVE_FINITE_NATIVE_DT_REQUIRED")
        self._ensure_backend()
        import mink
        if self._static_hand_tasks is None:
            self._static_hand_tasks = {side: mink.FrameTask(frame_name=name, frame_type="site",
                position_cost=STATIC_FRAME_TASK_PARAMETERS["position_cost"],
                orientation_cost=STATIC_FRAME_TASK_PARAMETERS["orientation_cost"])
                for side, name in zip(("A", "B"), self.tool_frame_ids)}
        model_target_mm = _arm_to_model(targets, self._base)
        self.budget.claim()
        self._backend.sync_config(np.degrees(seed), reset_posture_target=False)
        for side, target in zip(("A", "B"), model_target_mm):
            self.budget.claim()
            self._static_hand_tasks[side].set_target(mink.SE3.from_rotation_and_translation(
                mink.SO3.from_matrix(target[:3, :3]), target[:3, 3] / 1000.0))
        self.budget.claim()
        velocity = mink.solve_ik(self._backend.configuration,
            [self._static_hand_tasks["A"], self._static_hand_tasks["B"]], float(dt_s),
            self._backend.solver, damping=STATIC_FRAME_TASK_PARAMETERS["solve_damping"],
            limits=self._backend.limits)
        self._backend.configuration.integrate_inplace(velocity, float(dt_s))
        q = _array(self._backend.configuration.q, (14,), "static_backend_q_rad")
        checked, free, collision = self._check_collisions(q)
        recovered = self.fk(q)
        relative = np.swapaxes(targets[:, :3, :3], -1, -2) @ recovered[:, :3, :3]
        rotation_error = np.arccos(np.clip((np.trace(relative, axis1=-2, axis2=-1)-1)/2, -1, 1))
        return IKSolution(q, checked, free, {"mode": "ASSUMPTION_DRIVEN_OFFLINE",
            "provider_sha256": self.signature, "frame": FRAME, "position_unit": "meters",
            "joint_unit": "radians", "current_initialization_mode": self.current_initialization_mode,
            "static_frame_task_parameters": dict(STATIC_FRAME_TASK_PARAMETERS),
            "original_future_solver_unchanged": True, "differential_steps": 1,
            "position_error_m": np.linalg.norm(recovered[:, :3, 3]-targets[:, :3, 3], axis=1).tolist(),
            "rotation_error_rad": rotation_error.tolist(), "collision": collision,
            "geometry_calls_used": self.budget.used, "measured": False,
            "geometry_qualified": False, "physical_deployment_authorized": False,
            "control_ground_truth": False, "robot_commands_sent": 0})

    def solve_retarget_current(self, target_poses, seed_q_rad, dt_s):
        """Ego-derived current labels: static IK, finite box projection, diagnostics."""
        return self._solve_retarget(target_poses, seed_q_rad, dt_s, current=True)

    def solve_retarget_future(self, target_poses, seed_q_rad, dt_s):
        """Ego-derived future labels retain the original solver and actual dt."""
        return self._solve_retarget(target_poses, seed_q_rad, dt_s, current=False)

    def _solve_retarget(self, target_poses, seed_q_rad, dt_s, *, current):
        if getattr(self, "training_admission_policy", STRICT_TRAINING_ADMISSION) != EGO_TRAINING_ADMISSION:
            raise OfflineMinkError("EXPLICIT_EGO_DERIVED_TRAINING_ADMISSION_REQUIRED")
        if self.collision_validation != ORIGINAL_COLLISION_MODE:
            raise OfflineMinkError("EGO_TRAINING_USES_ORIGINAL_CAPSULE_DIAGNOSTICS_ONLY")
        targets = _se3(target_poses, (2, 4, 4), "ego_retarget_per_arm_base_from_model_tool_m")
        seed = _array(seed_q_rad, (14,), "ego_retarget_seed_q_rad")
        if not np.isfinite(dt_s) or dt_s <= 0:
            raise OfflineMinkError("POSITIVE_FINITE_NATIVE_DT_REQUIRED")
        self._ensure_backend()
        bounds = self._backend._joint_bounds_deg
        if [(item[0], item[1]) for item in bounds] != [
                (side, joint) for side in ("A", "B") for joint in range(1, 8)]:
            raise OfflineMinkError("FROZEN_MODEL_JOINT_BOUND_ORDER_REQUIRED")
        lower = _array(np.radians([item[2] for item in bounds]), (14,), "ego_joint_lower_rad")
        upper = _array(np.radians([item[3] for item in bounds]), (14,), "ego_joint_upper_rad")
        if np.any(lower >= upper) or np.any(seed < lower) or np.any(seed > upper):
            raise OfflineMinkError("VALID_SOFTWARE_BOUNDS_AND_SEED_REQUIRED")
        # No q is invented if the original finite solver fails before returning.
        raw = (self.solve_static if current else self.solve)(targets.copy(), seed.copy(), float(dt_s))
        if not isinstance(raw, IKSolution):
            raise OfflineMinkError("EGO_RETARGET_IK_SOLUTION_REQUIRED")
        raw_q = _array(raw.q_rad, (14,), "ego_retarget_raw_q_rad")
        projected = np.clip(raw_q, lower, upper)
        correction = projected - raw_q
        checked, free, collision = self._check_legacy_collisions(projected)
        recovered = self.fk(projected)
        relative = np.swapaxes(targets[:, :3, :3], -1, -2) @ recovered[:, :3, :3]
        position_error = np.linalg.norm(recovered[:, :3, 3] - targets[:, :3, 3], axis=1).tolist()
        rotation_error = np.arccos(np.clip((np.trace(relative, axis1=-2, axis2=-1)-1)/2, -1, 1)).tolist()
        collision = {**collision, "collision_validation": ORIGINAL_COLLISION_MODE,
                     "collision_checked": bool(checked), "collision_free": bool(free)}
        post_fit = {"position_error_m": position_error, "rotation_error_rad": rotation_error,
                    "collision": deepcopy(collision)}
        return IKSolution(projected.copy(), bool(checked), bool(free), {
            **deepcopy(raw.diagnostics), "provider_sha256": self.signature,
            "training_admission_policy": EGO_TRAINING_ADMISSION,
            "label_semantics": "EGO_DERIVED_MODEL_JOINTS_NOT_MEASURED_CONTROLS",
            "projection_policy": EGO_PROJECTION_POLICY, "frame": FRAME,
            "position_unit": "meters", "joint_unit": "radians",
            "position_error_m": position_error, "rotation_error_rad": rotation_error,
            "collision": collision, "geometry_calls_used": self.budget.used,
            "retargeting": {"raw_q_rad": raw_q.tolist(), "projected_q_rad": projected.tolist(),
                "projection_correction_rad": correction.tolist(), "joint_lower_rad": lower.tolist(),
                "joint_upper_rad": upper.tolist(), "projection_applied": bool(np.any(correction != 0)),
                "max_abs_projection_correction_rad": float(np.max(np.abs(correction))),
                "raw_fit_diagnostics": deepcopy(raw.diagnostics), "post_projection_fit_diagnostics": post_fit,
                "source_solver": "solve_static" if current else "solve",
                "dt_s": float(dt_s), "diagnostics_are_not_training_precision_or_collision_gates": True},
            "measured": False, "geometry_qualified": False,
            "physical_deployment_authorized": False, "control_ground_truth": False,
            "robot_commands_sent": 0})

    def assumed_profile(self, explicit_profile):
        """Derive source software limits; only absolute FK tolerances are NEW assumptions."""
        from .assumed_deployment_v1 import _profile
        _profile(explicit_profile)  # do not inherit a forged measured/qualified flag
        self._ensure_backend()
        profile = deepcopy(explicit_profile)
        profile.setdefault("transforms", {})["T_world_from_arm_base"] = self._base.tolist()
        profile["numerical_limits"] = {
            "joint_lower_rad": np.radians([b[2] for b in self._backend._joint_bounds_deg]).tolist(),
            "joint_upper_rad": np.radians([b[3] for b in self._backend._joint_bounds_deg]).tolist(),
            "joint_speed_max_rad_s": [float(self._backend_module._MAX_VELOCITY[j]) * float(
                self.backend_parameters["speed_limit_scale"]) for j in JOINT_NAMES],
            "joint_step_max_rad": [self.client_step_rad] * 14,
            "position_tolerance_m": 0.01, "rotation_tolerance_rad": 0.08726646259971647}
        profile["numeric_limit_sources"] = {
            "joint_bounds": "FROZEN_XML_PLUS_EXPLICIT_FROZEN_YAML_J2_J4_OVERRIDES",
            "joint_speed": "FROZEN_BACKEND_MAX_VELOCITY_TIMES_YAML_0.55",
            "joint_step": "FROZEN_CLIENT_SOFTWARE_DEFAULT_NOT_MEASURED_CONTROL_LIMIT",
            "absolute_fk_tolerances": ENGINEERING_LIMIT_SOURCE,
            "collision": "FROZEN_YAML_QP_MARGIN_PLUS_ACTUAL_ALL_MODELED_PAIR_NONPENETRATION"}
        if self.collision_validation == MESH_COLLISION_MODE:
            profile["numeric_limit_sources"]["collision"] = (
                "FROZEN_YAML_ORIGINAL_QP_MARGIN_PLUS_INDEPENDENT_FROZEN_STL_CONVEX_HULL_AUDIT")
        profile["mathematical_provider"] = {"signature": self.signature, "frame": FRAME,
            "source_receipts": deepcopy(self.source_receipts), "dependencies": dict(DEPENDENCIES),
            "backend_parameters": deepcopy(self.backend_parameters),
            "synthetic_initial_state_source": "FROZEN_XML_HOME_NOT_MEASURED_FEEDBACK",
            "missing_physical_geometry": ["Revo2 hands/fingers", "real scene"],
            "physical_deployment_authorized": False}
        profile["mathematical_provider"].update(collision_validation=self.collision_validation,
            current_initialization_mode=self.current_initialization_mode,
            training_admission_policy=self.training_admission_policy,
            ego_projection_policy=EGO_PROJECTION_POLICY,
            static_frame_task_parameters=dict(STATIC_FRAME_TASK_PARAMETERS))
        if self._mesh_audit is not None:
            profile["mathematical_provider"]["mesh_collision_inventory"] = deepcopy(self._mesh_audit.inventory)
        return profile
