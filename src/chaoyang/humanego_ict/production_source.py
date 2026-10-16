"""Read-only raw replay for non-deployable H50 derived-motion supervision.

Current head/controller/MANUS are selected from the COMPLETE raw streams, past
only. Nearest samples are used solely for future supervision. Recorded capture
metadata is the explicit clock trust boundary, not an invented Windows counter
implementation or exposure synchronization certificate. Nothing is published.
"""
from __future__ import annotations

import ast
import hashlib
import io
import json
import sys
import types
from typing import Any

import numpy as np

from .data import TrainingDataError, VerifiedInputCache, _pose9, _proper_se3, reference_bytes

SCHEMA = "pick-place-raw-causal-motion-source-v1"
SOURCE_PROFILE = "RAW_WRIST_LOCAL_HANDMOTION_NONE"
CLOCK_QUALIFICATION = "RECORDED_CAPTURE_METADATA_NS_SINGLE_HOST_DOMAIN"
CLOCK_TRUST_BOUNDARY = "SOURCE_RECORD_DECLARATION_NOT_COUNTER_IMPLEMENTATION_PROOF"
MAX_NS, P95_NS, MIN_COVERAGE = 40_000_000, 30_000_000, .95
CHANNEL_ORDER = ["thumb_bend", "thumb_rotation", "index", "middle", "ring", "pinky"]
LEGACY_SHA = "769bc18d4f35a0c6155b51e47a9dc9c2910fb4950ffe2a8c9de68fca24daaa25"
CALIBRATION_SHA = "37468f45d44e16daed48932803351e2254df76c4ffe61a94fada9666f6a50ff6"
MAP25 = np.array([0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24])
PARENTS = (0, 0, 1, 2, 3, 0, 5, 6, 7, 8, 0, 10, 11, 12, 13, 0, 15, 16, 17, 18, 0, 20, 21, 22, 23)
KINDS = ("Invalid", "Metacarpal", "Proximal", "Distal", "Tip") + ("Metacarpal", "Proximal", "Intermediate", "Distal", "Tip") * 4
ALIASES = ("Invalid", "MCP", "PIP", "DIP", "TIP") + ("MCP", "PIP", "IP", "DIP", "TIP") * 4
CHAINS = ("Hand",) + ("Thumb",) * 4 + tuple(f for f in ("Index", "Middle", "Ring", "Pinky") for _ in range(5))
_MAPPER_MODULES: dict[str, Any] = {}


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _json(reference, cache=None):
    try:
        return json.loads(reference_bytes(reference, cache=cache).decode("utf-8-sig"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise TrainingDataError("invalid source JSON") from exc


def _rows(reference, cache=None):
    result = []
    for number, text in enumerate(reference_bytes(reference, cache=cache).decode("utf-8-sig").splitlines(), 1):
        if text.strip():
            try:
                result.append((number, json.loads(text)))
            except ValueError as exc:
                raise TrainingDataError("invalid source JSONL") from exc
    if not result:
        raise TrainingDataError("empty source stream")
    return result


def _timeline(values):
    if not values or any(type(v) is not int or v < 0 or v > np.iinfo(np.int64).max for v in values):
        raise TrainingDataError("signed integer recorded nanoseconds required")
    result = np.asarray(values, np.int64)
    if np.any(np.diff(result) <= 0):
        raise TrainingDataError("source timestamps duplicate/backwards")
    return result


def _select(times, targets, past):
    if past:
        indices = np.searchsorted(times, targets, side="right") - 1
        available = indices >= 0
        return np.maximum(indices, 0), available
    after = np.minimum(np.searchsorted(times, targets), len(times) - 1)
    before = np.maximum(after - 1, 0)
    return np.where(abs(times[before] - targets) <= abs(times[after] - targets), before, after), np.ones(len(targets), bool)


def _pose(value, *, convert=True):
    try:
        if isinstance(value.get("pos"), list) and isinstance(value.get("quat"), list):
            a = np.asarray(value["pos"] + value["quat"], float)
        else:
            a = np.asarray([float(v) for v in value.get("pose", "").split(",")], float)
        if a.shape != (7,) or not np.isfinite(a).all() or abs(np.linalg.norm(a[3:]) - 1) > 1e-3:
            return np.full((4, 4), np.nan), False
        x, y, z, w = a[3:] / np.linalg.norm(a[3:])
        rotation = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                             [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                             [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
        t = np.eye(4)
        t[:3, :3], t[:3, 3] = rotation, a[:3]
        if convert:
            b = np.array([[0., 0., 1.], [-1., 0., 0.], [0., 1., 0.]])
            reflection = np.diag([1., 1., -1.])
            t[:3, :3] = b @ reflection @ rotation @ reflection @ b.T
            t[:3, 3] = b @ reflection @ a[:3]
        return t, True
    except (ValueError, TypeError, AttributeError):
        return np.full((4, 4), np.nan), False


def _topology(row):
    chains = tuple(v.removeprefix("ChainType_").removeprefix("Finger") for v in row.get("chain_types", []) if isinstance(v, str))
    kinds = tuple(row.get("joint_types", []))
    if all(isinstance(v, str) for v in kinds):
        kinds = tuple(v.removeprefix("FingerJointType_") for v in kinds)
    elif all(type(v) is int and 0 <= v < 6 for v in kinds):
        enums = ("Invalid", "Metacarpal", "Proximal", "Intermediate", "Distal", "Tip")
        kinds = tuple(enums[v] for v in kinds)
    if (type(row.get("node_count")) is not int or row.get("node_count") != 25 or row.get("node_ids") != list(range(25))
            or tuple(row.get("parent_ids", [])) != PARENTS or chains != CHAINS
            or kinds not in (KINDS, ALIASES)):
        raise TrainingDataError("UNKNOWN_MANUS_RAW25_TOPOLOGY")


class LegacyHandMapper:
    """Execute the exact SHA-frozen *pure* helper; never import ROS/SDK modules.

The calibration file is a software mapping prior, not captured Revo2 commands.
No live smoother/pick-up gate is applied. Bad/degenerate joints remain unknown.
"""
    def __init__(self, mapper_ref, calibration_ref, *, allow_synthetic_fixture=False, cache=None):
        encoded = reference_bytes(mapper_ref, cache=cache)
        if mapper_ref["sha256"] != LEGACY_SHA:
            raise TrainingDataError("executed legacy pure mapper SHA differs")
        if calibration_ref["sha256"] != CALIBRATION_SHA and not allow_synthetic_fixture:
            raise TrainingDataError("frozen software calibration prior SHA differs")
        module = _MAPPER_MODULES.get(LEGACY_SHA)
        if module is None:
            tree = ast.parse(encoded)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                    if any(name.split(".")[0] not in {"__future__", "json", "dataclasses", "pathlib", "typing", "numpy"} for name in names):
                        raise TrainingDataError("legacy helper contains non-pure imports")
            name = "_chaoyang_frozen_revo2_" + LEGACY_SHA
            module = types.ModuleType(name)
            sys.modules[name] = module
            exec(compile(tree, mapper_ref["path"], "exec"), module.__dict__)
            _MAPPER_MODULES[LEGACY_SHA] = module
        raw = _json(calibration_ref, cache)
        features = ("thumb_bend", "thumb_rot", "thumb_full", "index", "middle", "ring", "pinky")
        if raw.get("mapping_version") != "manus_revo2_mano21_v1":
            raise TrainingDataError("legacy calibration mapping version differs")
        self.calibration = {}
        for side in ("left", "right"):
            points = []
            for state in ("open", "closed"):
                value = raw.get(side + "_" + state)
                if not isinstance(value, dict) or set(value) != set(features) or not np.isfinite(list(value.values())).all():
                    raise TrainingDataError("complete finite legacy software calibration required")
                points.append(module.CalibPoint(**value))
            self.calibration[side] = points
        self.module = module

    def __call__(self, joints25, side):
        joints = np.asarray(joints25, float)
        if side not in self.calibration or joints.shape != (25, 3) or not np.isfinite(joints).all():
            return np.zeros(6), np.zeros(6, bool)
        twenty = self.module.mano21_to_mhand20(joints[MAP25])
        # Prevent helper zero-vector fallbacks from becoming observed commands.
        for a, b, c in self.module._FINGER_JOINTS.values():
            if min(np.linalg.norm(twenty[b]-twenty[a]), np.linalg.norm(twenty[c]-twenty[b])) <= 1e-6:
                return np.zeros(6), np.zeros(6, bool)
        if (np.linalg.norm(np.cross(twenty[4]-twenty[0], twenty[8]-twenty[0])) <= 1e-6
                or np.linalg.norm(np.cross(twenty[4]-twenty[0], twenty[16]-twenty[0])) <= 1e-6
                or np.linalg.norm(twenty[2]-twenty[1]) <= 1e-6):
            return np.zeros(6), np.zeros(6, bool)
        values = self.module.normalize_motor(self.module.retarget_mano21(joints[MAP25], *self.calibration[side], side))
        return values, np.isfinite(values) & (values >= 0) & (values <= 1)


def _contract(value):
    if (value.get("schema") != SCHEMA or value.get("source_profile") != SOURCE_PROFILE
            or value.get("clock_qualification") != CLOCK_QUALIFICATION
            or value.get("clock_trust_boundary") != CLOCK_TRUST_BOUNDARY
            or value.get("timestamp_unit") != "nanoseconds"
            or value.get("timestamp_semantics") != "HOST_ARRIVAL_RECEIVE_TIME_NOT_EXPOSURE_SYNCHRONIZATION"
            or value.get("implementation_verification") != "UNKNOWN_NOT_CLAIMED"
            or not isinstance(value.get("clock_domain"), str) or not value["clock_domain"].strip()
            or value.get("control_ground_truth") is not False or value.get("physical_deployment_authorized") is not False
            or value.get("camera") != {"eye": "right", "sbs_index": 1, "output_size": [640, 480],
                "upright_axes": "capture_sensor_from_upright", "authority": "FACTORY_METADATA_PRIOR_NOT_PHYSICAL_QUALIFICATION"}):
        raise TrainingDataError("explicit recorded-clock/wrist-local/camera-prior source contract required")


def _factory_camera(parameters):
    """Read the two captured factory formats verbatim; never fit/calibrate."""
    try:
        right = parameters["right"]
        if "intrinsics" in right:
            intrinsics = right["intrinsics"]
            extrinsic = np.asarray(parameters["extrinsics"]["right"], float)
            if extrinsic.shape != (4, 4):
                raise ValueError("nested factory extrinsic must be 4x4")
            resolution = parameters["reference_resolution"]
            size = [resolution["w"], resolution["h"]]
            schema = "nested_4x4"
        else:
            intrinsics = right
            flat = np.asarray(right["extrinsic"], float)
            if flat.shape != (12,):
                raise ValueError("flat factory extrinsic must contain12 values")
            extrinsic = np.eye(4)
            extrinsic[:3] = flat.reshape(3, 4)
            size = [parameters["width"], parameters["height"]]
            schema = "flat_3x4"
        k = np.array([[intrinsics["fx"], 0., intrinsics["cx"]],
                      [0., intrinsics["fy"], intrinsics["cy"]], [0., 0., 1.]], float)
        if (not np.isfinite(k).all() or not np.isfinite(extrinsic).all()
                or any(type(v) is not int or v <= 0 for v in size)):
            raise ValueError("nonfinite camera or invalid reference resolution")
        return k, extrinsic, size, schema
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise TrainingDataError("raw flat-3x4/nested-4x4 factory camera parameters missing or invalid") from exc


def replay_session(record, *, mapper_ref, calibration_ref, source_contract,
                   allow_synthetic_fixture=False, _input_cache=None):
    """Pure replay of one manifest record. No RGB decode, digital-q or writes.

Returns full-session arrays for separate past current and nearest-target routes.
The immutable input manifest and original capture JSON are verified, not merely
an arbitrary receipt=true. Only HDF attributes (never datasets) are accessed.
"""
    cache = _input_cache or VerifiedInputCache()
    _contract(source_contract)
    original = _json(source_contract["input_manifest_ref"], cache)
    if original.get("fixture_only") is True and not allow_synthetic_fixture:
        raise TrainingDataError("synthetic raw evidence is not production")
    matches = [r for r in original.get("sessions", []) if r.get("session_id") == record.get("session_id")]
    if len(matches) != 1 or matches[0] != record:
        raise TrainingDataError("session does not match the actual frozen input manifest")
    policy = original.get("time_policy", {})
    if (policy.get("source_clock_precision") != "recorded_nanoseconds"
            or policy.get("timestamp_semantics") != "software_stream_receipt_not_exposure_sync"):
        raise TrainingDataError("recorded source nanosecond/arrival declaration absent")
    refs = record["source_refs"]
    required = {"source_manifest", "selection", "workflow", "manus_meta", "pico", "manus", "video_qpc", "video_wall", "camera_params", "camera_meta", "hdf5_prior_only", "raw_vst"}
    if not required.issubset(refs):
        raise TrainingDataError("raw capture roles absent")
    manifest, selection, workflow, meta = (_json(refs[k], cache) for k in ("source_manifest", "selection", "workflow", "manus_meta"))
    identity = manifest.get("session")
    if (identity != "pick_place_1008_" + record["session_id"] or selection.get("session") != identity
            or workflow.get("session") != identity or meta.get("session") != identity
            or selection.get("schema") != "capture_selection_v1" or workflow.get("schema") != "capture_workflow_v1"
            or not selection.get("captured_at") or workflow.get("outcome") != "exported"):
        raise TrainingDataError("independent original capture identity/selection/workflow absent")
    clocks, coords = manifest.get("clocks", {}), manifest.get("coordinates", {})
    if (clocks.get("align_clock") != "single capture-host monotonic/QPC domain; never mixed"
            or "recv_qpc_ns" not in clocks.get("align_key", "")
            or "HandMotion_None" not in coords.get("manus", "")
            or "wrist-local" not in coords.get("manus", "")
            or coords.get("training_frame") != "right-handed X-forward Y-left Z-up meters"):
        raise TrainingDataError("original capture profile/clock domain declaration missing")
    video = _timeline([row for _, row in _rows(refs["video_qpc"], cache)])
    wall = _timeline([row for _, row in _rows(refs["video_wall"], cache)])
    if len(video) != record["raw_frame_count"] or len(video) != len(wall) or len(video) < 51 or np.median(np.diff(video)) < 1_000_000:
        raise TrainingDataError("native picture count or recorded nanosecond cadence differs")
    pico = _rows(refs["pico"], cache)
    pt = _timeline([row["recv_qpc_ns"] for _, row in pico])
    serial = record["device_serial"]
    if any(row.get("dev") != serial for _, row in pico):
        raise TrainingDataError("capture PICO identity differs")
    ph, pc = np.full((len(pico), 4, 4), np.nan), np.full((len(pico), 2, 4, 4), np.nan)
    hv, cv = np.zeros(len(pico), bool), np.zeros((len(pico), 2), bool)
    for i, (_, row) in enumerate(pico):
        data = row.get("data", {})
        ph[i], good = _pose(data.get("Head", {}))
        try:
            hv[i] = good and (int(data.get("Head", {}).get("status", 0)) & 3) == 3
        except (ValueError, TypeError):
            pass
        for hand, side in enumerate(("left", "right")):
            pc[i, hand], cv[i, hand] = _pose(data.get("Controller", {}).get(side, {}))
    import h5py
    with h5py.File(io.BytesIO(reference_bytes(refs["hdf5_prior_only"], cache=cache)), "r") as stream:
        raw = stream.attrs["controller_to_wrist_calibration"]
        prior = json.loads(raw.decode("utf-8-sig") if isinstance(raw, bytes) else raw)
        max_ns = min(MAX_NS, int(float(stream.attrs["max_skew_ms"]) * 1e6), int(float(stream.attrs["gate_ms"]) * 1e6))
        p95_ns = min(P95_NS, int(float(stream.attrs["max_p95_skew_ms"]) * 1e6))
        minimum = max(MIN_COVERAGE, float(stream.attrs["min_hand_coverage"]), float(stream.attrs["min_complete_coverage"]))
    if not 0 < max_ns <= MAX_NS or not 0 < p95_ns <= P95_NS or not MIN_COVERAGE <= minimum <= 1:
        raise TrainingDataError("original time/coverage gates invalid")
    if (prior.get("schema") != "controller_to_wrist_v1" or prior.get("convention") != "wrist_pose = compose_pose(controller_pose, transform)"
            or prior.get("controller_pose_frame") != "pico_world_rh_x_forward_y_left_z_up"):
        raise TrainingDataError("explicit controller installation prior required")
    install = np.array([_pose(prior[s], convert=False)[0] for s in ("left", "right")])
    _proper_se3(install, "installation prior")
    camera = record["camera"]["right"]
    # Re-extract actual factory values, not an independently editable K claim.
    parameters = _json(refs["camera_params"], cache)
    camera_meta = _json(refs["camera_meta"], cache)
    raw_k, raw_extrinsic, camera_size, camera_schema = _factory_camera(parameters)
    if (not np.allclose(raw_k, camera["K"], atol=1e-10, rtol=0)
            or not np.allclose(raw_extrinsic, camera["extrinsic_4x4"], atol=1e-10, rtol=0)
            or camera_size != record["reference_eye_size"]
            or record.get("camera_schema", camera_schema) != camera_schema
            or camera_meta.get("device_serial") != serial):
        raise TrainingDataError("frozen camera snapshot differs from actual source metadata")
    camera_metadata_conflicts = []
    if camera_meta.get("extrinsics_included") is False:
        camera_metadata_conflicts.append("SNAPSHOT_META_EXTRINSICS_FALSE_BUT_FACTORY_JSON_CONTAINS_MATRIX")
    k = raw_k.copy(); k[:2] *= 640 / record["reference_eye_size"][0]
    extrinsic = raw_extrinsic
    a, u = np.eye(4), np.eye(4)
    a[:3, :3] = [[0, -1, 0], [0, 0, 1], [-1, 0, 0]]
    u[:3, :3] = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]
    head_from_camera = np.linalg.inv(extrinsic @ a) @ u
    _proper_se3(head_from_camera, "factory camera prior")
    if (k.shape != (3, 3) or not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0
            or record["reference_eye_size"][0] / record["reference_eye_size"][1] != 640 / 480):
        raise TrainingDataError("camera pixel geometry invalid")
    mapper = LegacyHandMapper(mapper_ref, calibration_ref, allow_synthetic_fixture=allow_synthetic_fixture, cache=cache)
    arrays = {"qpc_ns": video, "frame_id": np.arange(len(video), dtype=np.int64), "controller_to_wrist": install, "K": k, "head_from_camera": head_from_camera}
    for mode, past in (("past", True), ("target", False)):
        indices, available = _select(pt, video, past)
        skew = pt[indices] - video
        good = available & (abs(skew) <= max_ns)
        arrays[mode + "_head"] = ph[indices]
        arrays[mode + "_head_valid"] = hv[indices] & good
        arrays[mode + "_controller"] = pc[indices]
        arrays[mode + "_controller_valid"] = cv[indices] & good[:, None]
        arrays[mode + "_pico_qpc_ns"] = pt[indices]
        arrays[mode + "_pico_line"] = np.array([pico[i][0] for i in indices], np.int64)
        arrays[mode + "_hands"] = np.full((len(video), 2, 25, 3), np.nan)
        arrays[mode + "_hand_valid"] = np.zeros((len(video), 2), bool)
        arrays[mode + "_hand_qpc_ns"] = np.zeros((len(video), 2), np.int64)
        arrays[mode + "_hand_line"] = np.zeros((len(video), 2), np.int64)
    manus = _rows(refs["manus"], cache)
    all_times = [video, pt]
    for hand, side in enumerate(("left", "right")):
        rows = [(line, row) for line, row in manus if row.get("side") == side]
        times = _timeline([row["recv_qpc_ns"] for _, row in rows]); all_times.append(times)
        joints = np.full((len(rows), 25, 3), np.nan); valid = np.zeros(len(rows), bool)
        expected = meta.get("calibration", {}).get(side, {}).get("glove_id")
        if expected is None or meta["calibration"][side].get("sdk_applied") is not True:
            raise TrainingDataError("original MANUS glove calibration evidence absent")
        for i, (_, row) in enumerate(rows):
            _topology(row)
            if row.get("glove_id") != expected or row.get("type") != "manus_frame":
                raise TrainingDataError("original MANUS profile/glove identity differs")
            try:
                nodes = np.asarray(row.get("nodes", []), float)
            except (ValueError, TypeError):
                continue
            valid[i] = (row.get("calibration_applied") is True and nodes.shape == (25, 7)
                        and np.isfinite(nodes).all() and np.allclose(nodes[0, :3], 0, atol=1e-6)
                        and np.max(np.linalg.norm(nodes[:, :3], axis=-1)) < 1)
            if valid[i]:
                joints[i] = nodes[:, :3]  # already wrist-local None; NO quaternion canonicalization
        for mode, past in (("past", True), ("target", False)):
            indices, available = _select(times, video, past)
            arrays[mode + "_hands"][:, hand] = joints[indices]
            arrays[mode + "_hand_valid"][:, hand] = valid[indices] & available & (abs(times[indices] - video) <= max_ns)
            arrays[mode + "_hand_qpc_ns"][:, hand] = times[indices]
            arrays[mode + "_hand_line"][:, hand] = [rows[i][0] for i in indices]
    arrays["target_commands"] = np.zeros((len(video), 2, 6)); arrays["target_command_valid"] = np.zeros((len(video), 2, 6), bool)
    for frame in range(len(video)):
        for hand, side in enumerate(("left", "right")):
            if arrays["target_hand_valid"][frame, hand]:
                arrays["target_commands"][frame, hand], arrays["target_command_valid"][frame, hand] = mapper(arrays["target_hands"][frame, hand], side)
    signature = canonical_sha({"refs": refs, "mapper_ref": mapper_ref, "calibration_ref": calibration_ref, "source_contract": source_contract})
    metadata = {"session_id": record["session_id"], "capture_id": identity, "captured_at": selection["captured_at"],
        "source_group_id": "capture:" + refs["source_manifest"]["sha256"],
        "capture_interval_ns": [int(wall[0]), int(wall[-1])], "capture_interval_clock": "RECORDED_HOST_WALL_NS_RECORDING_BOUNDARY",
        "capture_qpc_interval_ns": [min(int(t[0]) for t in all_times), max(int(t[-1]) for t in all_times)],
        "recording_sha256": {role: refs[role]["sha256"] for role in ("raw_vst", "pico", "manus", "video_qpc", "video_wall")},
        "source_signature": signature, "source_profile": SOURCE_PROFILE, "clock_qualification": CLOCK_QUALIFICATION,
        "clock_trust_boundary": CLOCK_TRUST_BOUNDARY, "implementation_verification": "UNKNOWN_NOT_CLAIMED",
        "physical_calibration_status": "UNKNOWN_NOT_GRANTED", "source_path": refs["source_manifest"]["path"],
        "camera_schema": camera_schema, "camera_metadata_conflicts": camera_metadata_conflicts}
    metadata["mapping_authority"] = "SHA_FROZEN_UNSMOOTHED_SOFTWARE_MAPPING_AND_CALIBRATION_PRIOR_NOT_CAPTURED_COMMANDS"
    for value in arrays.values():
        value.flags.writeable = False
    return {"record": record, "contract": source_contract, "arrays": arrays, "metadata": metadata,
            "quality": {"max_ns": max_ns, "p95_ns": p95_ns, "min_coverage": minimum},
            "actual_input_refs": [*refs.values(), mapper_ref, calibration_ref, source_contract["input_manifest_ref"]]}


def fixed_anchor_ids(replay):
    return list(range(0, len(replay["arrays"]["qpc_ns"]) - 50, 32))


def aggregate_group_plan(metadata):
    """Prove distinct recording identities/bytes + disjoint original intervals.

The split is fixed BEFORE quality evaluation. Shared device/install priors are
not groups, and no segmenting/merged-recording identity is silently admitted.
"""
    ordered = sorted(metadata, key=lambda v: v["source_path"])
    if len({v["capture_id"] for v in ordered}) != len(ordered) or len({v["source_group_id"] for v in ordered}) != len(ordered):
        raise TrainingDataError("duplicate/merged capture identity cannot count twice")
    for role in ("raw_vst", "pico", "manus", "video_qpc", "video_wall"):
        if len({v["recording_sha256"][role] for v in ordered}) != len(ordered):
            raise TrainingDataError("shared original recording cannot count as independent captures")
    for i, first in enumerate(ordered):
        for second in ordered[i+1:]:
            a, b = first["capture_interval_ns"], second["capture_interval_ns"]
            if max(a[0], b[0]) <= min(a[1], b[1]):
                raise TrainingDataError("capture intervals overlap; independent recording proof unresolved")
    return {v["session_id"]: {"source_group_id": v["source_group_id"], "split": "validation" if i % 10 == 0 else "train",
        "capture_id": v["capture_id"], "capture_interval_ns": v["capture_interval_ns"], "source_signature": v["source_signature"]} for i, v in enumerate(ordered)}


def prepare_anchor(replay, anchor_frame_id):
    """Pure H50 + causal renderer snapshot. Ineligible candidates stay explicit."""
    a, gates = replay["arrays"], replay["quality"]
    if type(anchor_frame_id) is not int or anchor_frame_id not in fixed_anchor_ids(replay):
        raise TrainingDataError("anchor must be fixed native stride32 with contiguous H50")
    anchor = anchor_frame_id; rows = np.arange(anchor+1, anchor+51)
    ticks = int(a["qpc_ns"][anchor]); missing = []
    pico_age = ticks - int(a["past_pico_qpc_ns"][anchor]); hand_age = ticks - a["past_hand_qpc_ns"][anchor]
    current_valid = a["past_controller_valid"][anchor]
    if (not a["past_head_valid"][anchor] or not current_valid.all() or not a["past_hand_valid"][anchor].all()
            or not 0 <= pico_age <= gates["max_ns"] or np.any((hand_age < 0) | (hand_age > gates["max_ns"]))):
        missing.append("BLOCKED_CURRENT_PAST_SENSOR_OBSERVATION")
    timing = {}
    for name, times in (("pico", a["target_pico_qpc_ns"][rows]), ("manus_left", a["target_hand_qpc_ns"][rows, 0]), ("manus_right", a["target_hand_qpc_ns"][rows, 1])):
        offsets = abs(times - a["qpc_ns"][rows]); timing[name] = {"max_abs_ns": int(offsets.max()), "p95_abs_ns": float(np.percentile(offsets, 95))}
        if offsets.max() > gates["max_ns"] or np.percentile(offsets, 95) > gates["p95_ns"]:
            missing.append("BLOCKED_TARGET_ORIGINAL_TIME_GATE:" + name)
    pose_valid = a["target_controller_valid"][rows]
    command_valid = a["target_command_valid"][rows]
    complete = pose_valid.all(axis=1) & a["target_hand_valid"][rows].all(axis=1) & command_valid.all(axis=(1, 2))
    if complete.mean() < gates["min_coverage"]:
        missing.append("BLOCKED_H50_ORIGINAL_COVERAGE_GATE")
    c2w = a["past_head"][anchor] @ a["head_from_camera"]
    # Unknown current camera never becomes identity with a true observation mask.
    if not a["past_head_valid"][anchor]:
        c2w = np.full((4, 4), np.nan)
    camera_from_world = np.linalg.inv(c2w) if np.isfinite(c2w).all() else np.eye(4)
    controllers = a["target_controller"][rows].copy(); controllers[~pose_valid] = np.eye(4)
    wrists = controllers @ a["controller_to_wrist"]
    poses = _pose9(camera_from_world @ wrists, np.zeros(3), np.ones(3))
    mask = np.concatenate((np.repeat(pose_valid, 3, axis=1), np.repeat(pose_valid, 6, axis=1), command_valid.reshape(50, 12)), axis=1)
    actions = np.concatenate((poses[..., :3].reshape(50, 6), poses[..., 3:].reshape(50, 12), a["target_commands"][rows].reshape(50, 12)), axis=1)
    current_ctrl = a["past_controller"][anchor].copy(); current_ctrl[~current_valid] = np.eye(4)
    current_wrist = current_ctrl @ a["controller_to_wrist"]
    ict, ict_mask, conditioning = np.zeros((8, 29)), np.zeros(8, bool), np.full(8, -1, np.int64)
    for hand in range(2):
        if current_valid[hand] and a["past_head_valid"][anchor] and 0 <= pico_age <= gates["max_ns"]:
            relations = _pose9(np.linalg.inv(current_wrist[hand]) @ current_wrist, np.zeros(3), np.ones(3)); relations[~current_valid] = 0
            ict[hand] = np.concatenate(([hand+1], _pose9(camera_from_world @ current_wrist[hand], np.zeros(3), np.ones(3)), relations.reshape(18), [-1]))
            ict_mask[hand], conditioning[hand] = True, a["past_pico_qpc_ns"][anchor]
    labels = {"x_ict": ict, "ict_mask": ict_mask, "y_action": np.where(mask, actions, 0), "action_valid_mask": mask,
        "anchor_qpc_ns": np.array(ticks, np.int64), "conditioning_source_qpc_ns": conditioning,
        "target_qpc_ns": a["qpc_ns"][rows].copy(), "target_source_frame_ids": rows.astype(np.int64),
        "reference_camera_to_world": c2w, "current_source_row": np.array(anchor, np.int64)}
    hand_valid = a["past_hand_valid"][anchor] & current_valid
    joints = a["past_hands"][anchor][:, MAP25]
    world = np.einsum("sij,skj->ski", current_wrist[:, :3, :3], joints) + current_wrist[:, None, :3, 3]
    current = {"session_id": replay["record"]["session_id"], "frame_id": anchor, "anchor_qpc_ns": ticks,
        "clock_domain": replay["contract"]["clock_domain"], "timestamp_unit": "nanoseconds",
        "head_source_qpc_ns": int(a["past_pico_qpc_ns"][anchor]), "controller_source_qpc_ns": np.repeat(a["past_pico_qpc_ns"][anchor], 2),
        "manus_source_qpc_ns": a["past_hand_qpc_ns"][anchor].copy(), "head": a["past_head"][anchor],
        "controller": current_ctrl, "wrist": current_wrist, "c2w": c2w, "K": a["K"], "hands_world": world,
        "hand_valid": hand_valid, "controller_valid": current_valid, "head_valid": bool(a["past_head_valid"][anchor]),
        "gravity": np.array([0., 0., -1.]), "actual_input_refs": replay["actual_input_refs"],
        "source_signature": replay["metadata"]["source_signature"], "camera_pose_direction": "p_world = c2w @ p_camera", "hand_coordinate_frame": "WORLD_METRES"}
    trace = {"schema": "pick-place-raw-h50-anchor-replay-v1", "session_id": current["session_id"], "source_signature": current["source_signature"],
        "source_frame_id": anchor, "anchor_qpc_ns": ticks, "past_pico_source_line": int(a["past_pico_line"][anchor]),
        "past_manus_source_lines": a["past_hand_line"][anchor].tolist(), "target_pico_source_lines": a["target_pico_line"][rows].tolist(),
        "target_manus_source_lines": a["target_hand_line"][rows].tolist(), "target_timing": timing,
        "current_ages_ns": [pico_age, *hand_age.tolist()], "complete_coverage": float(complete.mean()), "original_time_gates": gates,
        "camera_schema": replay["metadata"]["camera_schema"], "camera_metadata_conflicts": replay["metadata"]["camera_metadata_conflicts"],
        "supervision_class": "DERIVED_RETARGET", "control_ground_truth": False, "physical_deployment_authorized": False}
    return {"labels": labels, "current": current, "trace": trace, "eligible": not missing, "missing_items": sorted(set(missing))}
