"""Additive frozen-STL convex-hull audit, never measured geometry or control.

Only an in-memory collision clone is changed. The original backend, its two
QP capsule pairs, assets, joint bounds and exclusions remain authoritative for
their original consumers. MuJoCo mesh collision uses convex hulls: this layer
does not claim exact CAD surfaces, Revo2 coverage or physical scene safety.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

MODE = "FROZEN_STL_MESH_AUDIT_V1"
REPRESENTATION = "MUJOCO_CONVEX_HULL_OF_FROZEN_STL_NOT_MEASURED"
ROBOT_BODIES = frozenset({"base_link", "Base_L", "Base_R"}
    | {f"Link{i}_{side}" for side in ("L", "R") for i in range(1, 8)})
MESH_NAMES = frozenset((ROBOT_BODIES - {"base_link"}) | {"ZJ_Robot_link"})
POSE_ATTRIBUTES = ("pos", "quat", "euler", "axisangle", "xyaxes", "zaxis")


def _error(code):
    from .offline_mink_provider_v1 import OfflineMinkError
    return OfflineMinkError(code)


def build_mesh_vfs(sources, scene_path):
    """Return verified-input-only XML/assets; no filesystem reads or writes."""
    sources = {Path(path): raw for path, raw in sources.items()}
    scene_path = Path(scene_path)

    def bound(path, file):
        dependency = (path.parent / file).resolve()
        if dependency not in sources:
            raise _error("UNBOUND_MESH_CLONE_DEPENDENCY")
        return dependency

    if scene_path not in sources:
        raise _error("FROZEN_MESH_SCENE_REQUIRED")
    scene = ET.fromstring(sources[scene_path])
    includes = list(scene.iter("include"))
    if len(includes) != 1:
        raise _error("EXACT_SINGLE_FROZEN_MODEL_INCLUDE_REQUIRED")
    model_path = bound(scene_path, includes[0].attrib["file"])
    model = ET.fromstring(sources[model_path])
    if list(model.iter("include")):
        raise _error("NESTED_MESH_CLONE_INCLUDE_NOT_ALLOWED")
    assets, mesh_names = {}, set()
    for node in model.iter("mesh"):
        name = node.attrib.get("name")
        if name not in MESH_NAMES or name in mesh_names or "file" not in node.attrib:
            raise _error("EXACT_SEVENTEEN_FROZEN_STL_ASSETS_REQUIRED")
        mesh_names.add(name)
        source_path = bound(model_path, node.attrib["file"])
        filename = f"frozen_mesh_{name}.stl"
        assets[filename] = sources[source_path]
        node.set("file", filename)
    if mesh_names != MESH_NAMES:
        raise _error("EXACT_SEVENTEEN_FROZEN_STL_ASSETS_REQUIRED")
    converted = []
    for body in model.iter("body"):
        name = body.attrib.get("name")
        geoms = [node for node in body.findall("geom")
                 if node.attrib.get("class") == "collision"]
        if not geoms:
            continue
        if name not in ROBOT_BODIES or len(geoms) != 1:
            raise _error("EXACT_ONE_COLLISION_GEOM_PER_FROZEN_ROBOT_BODY_REQUIRED")
        mesh = "ZJ_Robot_link" if name == "base_link" else name
        visuals = [node for node in body.findall("geom")
                   if node.attrib.get("class") == "visual" and node.attrib.get("mesh") == mesh]
        if len(visuals) != 1:
            raise _error("FROZEN_MESH_VISUAL_POSE_REQUIRED")
        geom = geoms[0]
        for attribute in ("fromto", "size", *POSE_ATTRIBUTES):
            geom.attrib.pop(attribute, None)
        geom.set("type", "mesh")
        geom.set("mesh", mesh)
        # Mesh coordinates are the frozen visual coordinates, not the center
        # and orientation that were inferred from a capsule's fromto segment.
        for attribute in POSE_ATTRIBUTES:
            if attribute in visuals[0].attrib:
                geom.set(attribute, visuals[0].attrib[attribute])
        converted.append({"body": name, "mesh": mesh,
                          "geom_name": geom.attrib.get("name")})
    if len(converted) != 17 or {item["body"] for item in converted} != ROBOT_BODIES:
        raise _error("COMPLETE_SEVENTEEN_BODY_MESH_CLONE_REQUIRED")
    includes[0].set("file", "frozen_collision_model.xml")
    assets["frozen_collision_model.xml"] = ET.tostring(model, encoding="utf-8")
    xml = ET.tostring(scene, encoding="unicode")
    receipt = {"collision_validation": MODE, "geometry_representation": REPRESENTATION,
        "source_xml_sha256": {str(path): hashlib.sha256(sources[path]).hexdigest()
                              for path in (scene_path, model_path)},
        "virtual_scene_sha256": hashlib.sha256(xml.encode()).hexdigest(),
        "virtual_asset_sha256": {name: hashlib.sha256(raw).hexdigest()
                                  for name, raw in sorted(assets.items())},
        "converted_collision_bodies": sorted(converted, key=lambda item: item["body"]),
        "floor_and_original_qp_capsules_preserved_in_original_model": True,
        "physical_geometry_qualified": False, "physical_deployment_authorized": False}
    return xml, assets, receipt


def assert_same_kinematics(original, clone):
    """Reject any unrequested joint/body/site/base/model-order change."""
    for field in ("nq", "nv", "njnt", "nbody", "nsite", "ngeom", "nmesh", "nkey", "names"):
        if getattr(original, field) != getattr(clone, field):
            raise _error(f"MESH_CLONE_KINEMATICS_DRIFT:{field}")
    for field in ("qpos0", "key_qpos", "jnt_type", "jnt_qposadr", "jnt_dofadr",
            "jnt_bodyid", "jnt_pos", "jnt_axis", "jnt_range", "jnt_limited",
            "body_parentid", "body_weldid", "body_pos", "body_quat", "body_mass",
            "body_inertia", "site_bodyid", "site_pos", "site_quat", "geom_bodyid",
            "geom_contype", "geom_conaffinity", "exclude_signature"):
        if not np.array_equal(getattr(original, field), getattr(clone, field)):
            raise _error(f"MESH_CLONE_KINEMATICS_DRIFT:{field}")


class OfflineMeshCollisionAudit:
    """Read q only, with an independently compiled, shared-budget clone."""

    def __init__(self, provider):
        from .offline_mink_provider_v1 import _collision_pairs
        provider._assert_frozen()
        self.provider, self.mj, self.budget = provider, provider._mujoco, provider.budget
        xml, assets, self.receipt = build_mesh_vfs(provider._sources, provider.scene_path)
        self.budget.claim()  # attempted native clone compilation, including failure
        self.model = self.mj.MjModel.from_xml_string(xml, assets=assets)
        original = provider._backend.model
        # Preserve the verified backend's explicit runtime J2/J4 overrides.
        # This copies limits into the clone only, never modifies the original.
        self.model.jnt_range[:] = original.jnt_range
        assert_same_kinematics(original, self.model)
        self.data = self.mj.MjData(self.model)
        self.geom_ids, self.pairs, self.excluded = _collision_pairs(self.model)
        if (self.geom_ids, self.pairs, self.excluded) != _collision_pairs(original):
            raise _error("MESH_CLONE_COLLISION_PAIR_POLICY_DRIFT")
        self.pair_set = set(self.pairs)
        covered = set()
        for geom in self.geom_ids:
            body = self._body(geom)
            if body in ROBOT_BODIES:
                covered.add(body)
                if int(self.model.geom_type[geom]) != int(self.mj.mjtGeom.mjGEOM_MESH):
                    raise _error("MESH_CLONE_ENABLED_ROBOT_GEOM_NOT_MESH")
        self.coverage_complete = ROBOT_BODIES <= covered and bool(self.pairs)
        self.inventory = {**self.receipt, "scope": "ALL_ENABLED_MODELED_NONADJACENT_GEOMETRY_ONLY",
            "enabled_geom_count": len(self.geom_ids), "eligible_pair_count": len(self.pairs),
            "excluded_topology_pair_count": len(self.excluded),
            "pairs": [{"geom_ids": [a, b], "body_names": [self._body(a), self._body(b)]}
                      for a, b in self.pairs],
            "excluded_pairs": [{"geom_ids": [a, b], "reason": reason}
                               for a, b, reason in self.excluded],
            "same_pair_policy_as_original": True,
            "unmodeled_geometry": ["Revo2 hands/fingers", "real surrounding scene"]}
        self.signature = hashlib.sha256(json.dumps(self.inventory, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest()

    def _body(self, geom):
        return self.mj.mj_id2name(self.model, self.mj.mjtObj.mjOBJ_BODY,
                                int(self.model.geom_bodyid[geom]))

    def check(self, q_rad, original_data):
        from .bridge import _array
        self.provider._assert_frozen()
        q = _array(q_rad, (14,), "mesh_audit_q_rad")
        if not np.array_equal(np.asarray(original_data.qpos), q):
            raise _error("MESH_AUDIT_ORIGINAL_Q_NOT_MATCHED")
        self.data.qpos[:] = q
        self.budget = self.provider.budget  # each restored window has a new shared budget
        self.budget.claim()
        self.mj.mj_forward(self.model, self.data)
        parity = True
        parity_errors = {}
        for object_type, names, fields in ((self.mj.mjtObj.mjOBJ_SITE,
                ("left_tool_site", "right_tool_site"), ("site_xpos", "site_xmat")),
                (self.mj.mjtObj.mjOBJ_BODY, ("Base_L", "Base_R"), ("xpos", "xmat"))):
            for name in names:
                original_id = self.mj.mj_name2id(self.provider._backend.model, object_type, name)
                clone_id = self.mj.mj_name2id(self.model, object_type, name)
                if min(original_id, clone_id) < 0:
                    raise _error("MESH_AUDIT_REQUIRED_FK_FRAME_ABSENT")
                for field in fields:
                    a, b = getattr(original_data, field)[original_id], getattr(self.data, field)[clone_id]
                    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
                    error = float(np.max(np.abs(a - b))) if finite else None
                    parity_errors[f"{name}:{field}"] = error
                    parity &= finite and np.allclose(a, b, atol=1e-12, rtol=0)
        if not parity:
            raise _error("MESH_CLONE_SAME_Q_FK_PARITY_FAILED")
        contacts, contact_finite = [], True
        for contact in self.data.contact:
            distance = float(contact.dist)
            contact_finite &= bool(np.isfinite(distance))
            pair = tuple(sorted((int(contact.geom1), int(contact.geom2))))
            if pair in self.pair_set and distance < 0:
                contacts.append({"geom_ids": list(pair),
                    "body_names": [self._body(geom) for geom in pair],
                    "signed_distance_m": distance})
        native_enabled = not bool(int(self.model.opt.disableflags)
            & int(self.mj.mjtDisableBit.mjDSBL_CONTACT))
        checked = bool(self.coverage_complete and native_enabled and contact_finite and parity)
        free = bool(checked and not contacts)
        return checked, free, {"collision_validation": MODE,
            "geometry_representation": REPRESENTATION,
            "collision_checked": checked, "collision_free": free,
            "scope": "ALL_ENABLED_MODELED_NONADJACENT_GEOMETRY_ONLY",
            "enabled_geom_count": len(self.geom_ids), "eligible_pair_count": len(self.pairs),
            "excluded_topology_pair_count": len(self.excluded),
            "actual_nonadjacent_penetration_contacts": contacts,
            "native_contact_calculation_enabled": native_enabled,
            "contact_numerics_finite": contact_finite,
            "full_pair_test": "MUJOCO_ENABLED_COLLISION_PASS_NONPENETRATION",
            "full_pair_penetration_threshold_m": 0.0,
            "same_q_fk_parity_passed": bool(parity), "same_q_fk_max_absolute_errors": parity_errors,
            "collision_inventory_sha256": self.signature,
            "physical_geometry_qualified": False, "physical_scene_collision_status": "UNKNOWN",
            "collision_is_physical_safety": False, "physical_deployment_authorized": False,
            "unmodeled_geometry": ["Revo2 hands/fingers", "real surrounding scene"]}
