"""Exact per-session visible SAM seed evidence. Seeds are image labels only."""
from pathlib import Path
import copy
import cv2
import numpy as np
from . import audit, source

ROLES = {"glove", "controller", "mount", "attachment", "forearm", "cable", "human"}
UNSEEN_VISIBILITY = "NOT_VISIBLE_IN_INSPECTED_NATIVE_FRAMES"
UNSEEN_REVIEW = "AI_CONFIRMED_NOT_VISIBLE"
UNSEEN_VALIDATION = "NATIVE_QPC_AND_ORIGINAL_RGB_EXACT_VERIFIED"
CORRECTION_VALIDATION = "NATIVE_QPC_ORIGINAL_RGB_AND_SOURCE_IMAGE_ROW_EXACT_VERIFIED"
CORRECTION_SEMANTICS = "VISIBLE_FOREGROUND_PIXEL_NOT_JOINT_OR_CAD_POINT"
MAX_CORRECTIONS_PER_OBJECT = 16


def _verify_native_image(frame, qpc, reference, session, rgb):
    if (type(frame) is not int or not 0 <= frame < len(rgb)
            or type(qpc) is not int or qpc != int(session.qpc_ns[frame])):
        raise ValueError("MASK_SEED_NATIVE_FRAME_QPC_MISMATCH")
    if not isinstance(reference, dict):
        raise ValueError("MASK_SEED_NATIVE_IMAGE_REFERENCE_REQUIRED")
    audit._verify_exact_ref(reference, project_only=True)
    original = cv2.imread(reference["path"], cv2.IMREAD_COLOR)
    if (original is None or original.shape != rgb[frame].shape
            or not np.array_equal(original[:, :, ::-1], rgb[frame])):
        raise ValueError("MASK_SEED_ORIGINAL_RGB_PIXEL_MISMATCH")
    return frame


def _object_frame(seed, default):
    values = [seed[key] for key in ("frame_idx", "frame_index", "raw_frame_index") if key in seed]
    if not values:
        return default
    if any(type(value) is not int for value in values) or len(set(values)) != 1:
        raise ValueError("MASK_SEED_OBJECT_FRAME_ALIASES")
    return values[0]


def _unseeded_roles(record, session, rgb, seeded_roles):
    declarations = record.get("unseeded_roles", [])
    if not isinstance(declarations, list):
        raise ValueError("MASK_SEED_UNSEEDED_DECLARATIONS")
    if declarations and record["schema"] not in {"pick_place_visible_mask_seeds_v2", "pick_place_visible_mask_seeds_v3"}:
        raise ValueError("MASK_SEED_UNSEEDED_V2_REQUIRED")
    accepted, seen = [], set()
    for declaration in declarations:
        if not isinstance(declaration, dict):
            raise ValueError("MASK_SEED_UNSEEDED_VISIBILITY_PROOF")
        key = (declaration.get("side"), declaration.get("role"))
        if (key[0] not in source.SIDES or key[1] not in ROLES or key in seen
                or key in seeded_roles
                or declaration.get("visibility") != UNSEEN_VISIBILITY
                or declaration.get("review") != UNSEEN_REVIEW
                or declaration.get("physical_ground_truth", False) is not False
                or declaration.get("training_eligible", False) is not False
                or declaration.get("unreviewed_visibility_status", "UNKNOWN") != "UNKNOWN"
                or declaration.get("later_visibility_review_needed", True) is not True
                or declaration.get("all_frames_invisible", False) is not False
                or declaration.get("global_all_frames_invisible", False) is not False):
            raise ValueError("MASK_SEED_UNSEEDED_VISIBILITY_PROOF")
        frames = declaration.get("inspected_frames")
        if not isinstance(frames, list) or len(frames) < 3:
            raise ValueError("MASK_SEED_UNSEEDED_ZERO_MID_LAST_REQUIRED")
        inspected = []
        for evidence in frames:
            if not isinstance(evidence, dict):
                raise ValueError("MASK_SEED_UNSEEDED_VISIBILITY_PROOF")
            if evidence.get("device_serial", record["device_serial"]) != record["device_serial"]:
                raise ValueError("MASK_SEED_UNSEEDED_DEVICE_MISMATCH")
            inspected.append(_verify_native_image(evidence.get("raw_frame_index"),
                evidence.get("qpc_ns"), evidence.get("image"), session, rgb))
        if (len(set(inspected)) != len(inspected) or len(rgb) < 3
                or 0 not in inspected or len(rgb) - 1 not in inspected
                or not ({(len(rgb) - 1) // 2, len(rgb) // 2} & set(inspected))):
            raise ValueError("MASK_SEED_UNSEEDED_ZERO_MID_LAST_REQUIRED")
        value = copy.deepcopy(declaration)
        value.update(unreviewed_visibility_status="UNKNOWN", later_visibility_review_needed=True,
                     global_all_frames_invisible=False, validation=UNSEEN_VALIDATION)
        accepted.append(value)
        seen.add(key)
    return accepted


def _correction_source_rows(record):
    reference = record.get("source_images")
    if not isinstance(reference, dict):
        raise ValueError("MASK_SEED_CORRECTION_SOURCE_IMAGES_REQUIRED")
    audit._verify_exact_ref(reference, project_only=True)
    data = source.read_json(Path(reference["path"]))
    if (data.get("schema") not in {"pick_place_calibration_source_images_v1",
                                    "pick_place_visible_seed_source_images_v1"}
            or data.get("status") != "PASSED" or data.get("no_overlays") is not True
            or data.get("qpc_modified") is not False or data.get("lens_remap") is not False):
        raise ValueError("MASK_SEED_CORRECTION_SOURCE_IMAGES_DOMAIN")
    if data["schema"] == "pick_place_visible_seed_source_images_v1":
        return data.get("images", [])
    rows = []
    for session in data.get("sessions", []):
        for image in session.get("images", []):
            if (image.get("session_id") != session.get("session_id")
                    or image.get("device_serial") != session.get("device_serial")):
                raise ValueError("MASK_SEED_CORRECTION_SOURCE_SESSION_BINDING")
            rows.append(image)
    return rows


def _validated_corrections(seed, record, session, rgb, own_frame):
    corrections = seed.get("correction_points", [])
    if not isinstance(corrections, list) or len(corrections) > MAX_CORRECTIONS_PER_OBJECT:
        raise ValueError("MASK_SEED_CORRECTION_COUNT")
    if not corrections:
        return []
    if record["schema"] != "pick_place_visible_mask_seeds_v3":
        raise ValueError("MASK_SEED_CORRECTIONS_V3_REQUIRED")
    rows = _correction_source_rows(record)
    frames, result = {own_frame}, []
    height, width = rgb.shape[1:3]
    for correction in corrections:
        if (not isinstance(correction, dict)
                or correction.get("session_id") != record["session_id"]
                or correction.get("device_serial") != record["device_serial"]
                or any(correction.get(key, seed[key]) != seed[key]
                       for key in ("obj_id", "side", "role"))
                or correction.get("visible_confirmed") is not True
                or correction.get("seed_semantics") != CORRECTION_SEMANTICS
                or correction.get("physical_ground_truth", False) is not False
                or correction.get("training_eligible", False) is not False):
            raise ValueError("MASK_SEED_CORRECTION_IDENTITY_OR_VISIBILITY")
        frame = _object_frame(correction, None)
        if frame in frames:
            raise ValueError("MASK_SEED_CORRECTION_DISTINCT_NATIVE_FRAME_REQUIRED")
        qpc, image = correction.get("qpc_ns"), correction.get("image")
        _verify_native_image(frame, qpc, image, session, rgb)
        matches = [row for row in rows if isinstance(row, dict)
                   and row.get("session_id") == record["session_id"]
                   and row.get("device_serial") == record["device_serial"]
                   and row.get("raw_frame_index") == frame and row.get("qpc_ns") == qpc
                   and row.get("image") == image and row.get("pixels_overlaid") is False]
        if len(matches) != 1:
            raise ValueError("MASK_SEED_CORRECTION_SOURCE_IMAGE_ROW_MEMBERSHIP")
        try:
            points = np.asarray(correction.get("points_px"), dtype=float)
            labels = np.asarray(correction.get("point_labels"))
        except (TypeError, ValueError) as error:
            raise ValueError("MASK_SEED_CORRECTION_POINT_DOMAIN") from error
        if (points.ndim != 2 or points.shape[1] != 2 or not 1 <= len(points) <= 32
                or labels.shape != (len(points),) or labels.dtype.kind not in "iu"
                or not np.isfinite(points).all() or not np.isin(labels, [0, 1]).all()
                or not (labels == 1).any() or np.any(points[:, 0] < 0)
                or np.any(points[:, 0] >= width) or np.any(points[:, 1] < 0)
                or np.any(points[:, 1] >= height)):
            raise ValueError("MASK_SEED_CORRECTION_POINT_DOMAIN")
        value = copy.deepcopy(correction)
        value.update(frame_idx=frame, raw_frame_index=frame,
                     native_evidence_validation=CORRECTION_VALIDATION)
        result.append(value)
        frames.add(frame)
    return sorted(result, key=lambda value: value["frame_idx"])


def load_visible_seed(index_path, session_id, session, rgb):
    index_path = Path(index_path)
    index = source.read_json(index_path)
    if index.get("schema") != "pick_place_mask_seed_index_v1":
        raise ValueError("MASK_SEED_INDEX_SCHEMA")
    rows = [row for row in index.get("records", []) if row.get("session_id") == session_id]
    if len(rows) != 1:
        raise ValueError("MASK_SEED_SESSION_UNIQUE_REQUIRED")
    reference = rows[0]["seed"]
    audit._verify_exact_ref(reference, project_only=True)
    record = source.read_json(Path(reference["path"]))
    if (record.get("schema") not in {"pick_place_visible_mask_seeds_v1", "pick_place_visible_mask_seeds_v2",
                                        "pick_place_visible_mask_seeds_v3"}
            or record.get("session_id") != session_id
            or record.get("device_serial") != session.calibration["device_serial"]
            or record.get("visibility_review_status") != "AI_CONFIRMED_VISIBLE_PIXELS"
            or record.get("training_eligible") is not False
            or record.get("physical_ground_truth") is not False):
        raise ValueError("MASK_SEED_IDENTITY_OR_VISIBILITY_REVIEW")
    if (np.asarray(rgb).ndim != 4 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8
            or len(rgb) != len(session.qpc_ns)):
        raise ValueError("MASK_SEED_NATIVE_RGB_DOMAIN")
    anchor = _verify_native_image(record.get("raw_frame_index"), record.get("qpc_ns"),
                                  record.get("image"), session, rgb)
    objects = record.get("objects")
    if not isinstance(objects, list) or not 1 <= len(objects) <= 16:
        raise ValueError("MASK_SEED_OBJECT_COUNT")
    ids, roles, own_frames = set(), set(), []
    selected = copy.deepcopy(objects)
    height, width = rgb.shape[1:3]
    for seed in selected:
        object_id, role, side = seed.get("obj_id"), seed.get("role"), seed.get("side")
        if (type(object_id) is not int or not 1 <= object_id <= 16 or object_id in ids
                or role not in ROLES or side not in source.SIDES or seed.get("visible_confirmed") is not True):
            raise ValueError("MASK_SEED_OBJECT_ROLE_SIDE_OR_VISIBILITY")
        ids.add(object_id)
        roles.add((side, role))
        points, labels = np.asarray(seed.get("points_px"), float), np.asarray(seed.get("point_labels"))
        if (points.ndim != 2 or points.shape[1] != 2 or len(points) == 0 or len(points) > 32
                or labels.shape != (len(points),) or labels.dtype.kind not in "iu"
                or not np.isfinite(points).all() or not np.isin(labels, [0, 1]).all()
                or not (labels == 1).any() or np.any(points[:, 0] < 0) or np.any(points[:, 0] >= width)
                or np.any(points[:, 1] < 0) or np.any(points[:, 1] >= height)):
            raise ValueError("MASK_SEED_POINT_DOMAIN")
        own_frame = _object_frame(seed, anchor)
        if record["schema"] == "pick_place_visible_mask_seeds_v1":
            if own_frame != anchor:
                raise ValueError("MASK_SEED_SHARED_ANCHOR_REQUIRED")
        else:
            if seed.get("device_serial", record["device_serial"]) != record["device_serial"]:
                raise ValueError("MASK_SEED_OBJECT_DEVICE_MISMATCH")
            qpc = seed.get("qpc_ns", record["qpc_ns"] if own_frame == anchor else None)
            image = seed.get("image", record["image"] if own_frame == anchor else None)
            _verify_native_image(own_frame, qpc, image, session, rgb)
            seed.update(frame_idx=own_frame, raw_frame_index=own_frame, qpc_ns=qpc, image=copy.deepcopy(image))
        own_frames.append(own_frame)
        corrections = _validated_corrections(seed, record, session, rgb, own_frame)
        if corrections:
            seed["correction_points"] = corrections
            own_frames.extend(value["frame_idx"] for value in corrections)
    declarations = _unseeded_roles(record, session, rgb, roles)
    exempt = {(item["side"], item["role"]) for item in declarations}
    for side in source.SIDES:
        for role in ("glove", "controller"):
            if (side, role) not in roles and (side, role) not in exempt:
                raise ValueError("MASK_SEED_REQUIRED_GLOVE_CONTROLLER")
    return {"anchor_frame": min(own_frames), "objects": selected, "seed_reference": reference,
            "seed_index_reference": source.ref(index_path), "image": record["image"],
            "source_images": record.get("source_images"), "seed_schema": record["schema"],
            "unseeded_roles": declarations, "unseeded_roles_validation": UNSEEN_VALIDATION,
            "unreviewed_visibility_status": "UNKNOWN",
            "later_visibility_review_needed": bool(declarations),
            "source_authority": "AI_CONFIRMED_VISIBLE_IMAGE_LABELS_NOT_JOINT_CAD_OR_CONTROL_TRUTH"}


def seed_reference(index_path, session_id):
    data = source.read_json(Path(index_path))
    if data.get("schema") != "pick_place_mask_seed_index_v1":
        raise ValueError("MASK_SEED_INDEX_SCHEMA")
    found = [row["seed"] for row in data.get("records", []) if row.get("session_id") == session_id]
    if len(found) != 1:
        raise ValueError("MASK_SEED_SESSION_UNIQUE_REQUIRED")
    audit._verify_exact_ref(found[0], project_only=True)
    return found[0]
