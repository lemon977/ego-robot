"""Raw VST QPC + raw PICO/Controller + raw MANUS input producer.

Only controller_to_wrist_calibration is read from the HDF5 attribute namespace.
No HDF5 datasets, aligned rows, preview pixels or historical robot actions are
consumed. Native host-arrival clocks are retained; they are not exposure truth.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from typing import Any
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from chaoyang.exact78_baseline.pico_source import PicoSession
from chaoyang.pipeline.pico_manus_motion_v2 import (
    MAP_25_TO_21 as LEGACY_PARENT_MAP_25_TO_21,
    MANUS_NAMES as LEGACY_PARENT_MANUS_NAMES, installation_matrices, pose_matrices,
)
from .camera import CameraModel, normalize_camera, sha256

SIDES = ("left", "right")
# SDK raw25 has one nonthumb metacarpal before MCP/PIP/DIP/TIP. The
# collector labels MCP/PIP/IP are aliases that conflict with SDK categories.
# Keep the selected parent map immutable; only this raw-source adapter exports
# canonical MANO21 semantics to the unchanged retarget consumer.
RAW_SDK_TO_CANONICAL21 = np.array([0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14,
                                  16, 17, 18, 19, 21, 22, 23, 24], dtype=np.int64)
RAW_SDK_TO_CANONICAL21.setflags(write=False)
MAP_25_TO_21 = RAW_SDK_TO_CANONICAL21
RAW_SDK_PARENT_IDS = (0, 0, 1, 2, 3, 0, 5, 6, 7, 8, 0, 10, 11, 12, 13,
                      0, 15, 16, 17, 18, 0, 20, 21, 22, 23)
RAW_SDK_CHAIN_TYPES = ("Hand",) + ("Thumb",) * 4 + tuple(
    finger for finger in ("Index", "Middle", "Ring", "Pinky") for _ in range(5))
RAW_SDK_JOINT_TYPES = ("Invalid", "Metacarpal", "Proximal", "Distal", "Tip") + (
    "Metacarpal", "Proximal", "Intermediate", "Distal", "Tip") * 4
RAW_CAPTURE_JOINT_ALIASES = ("Invalid", "MCP", "PIP", "DIP", "TIP") + (
    "MCP", "PIP", "IP", "DIP", "TIP") * 4
RAW_SDK_NODE_NAMES = ("Hand_Invalid",) + tuple(
    finger + "_" + kind for finger, types in (
        ("Thumb", ("Metacarpal", "Proximal", "Distal", "Tip")),
        *((f, ("Metacarpal", "Proximal", "Intermediate", "Distal", "Tip"))
          for f in ("Index", "Middle", "Ring", "Pinky"))) for kind in types)
MANUS_NAMES = RAW_SDK_NODE_NAMES
CANONICAL_JOINT_NAMES_21 = ("virtual_wrist_root", "Thumb_CMC", "Thumb_MCP", "Thumb_IP", "Thumb_TIP") + tuple(
    finger + "_" + kind for finger in ("Index", "Middle", "Ring", "Pinky")
    for kind in ("MCP", "PIP", "DIP", "TIP"))
JOINT_MAPPING_SCHEMA = "pick_place_raw_sdk25_canonical21_v1"
RAW_INPUT_SCHEMA = "pick-place-raw-input-v2-canonical21"
SDK_TOPOLOGY_REFERENCES = (
    "https://docs.manus-meta.com/3.1.0/Software/Skeletons/",
    "https://docs.manus-meta.com/2.5.0/SDK/_manus_s_d_k_types_8h/",
    "https://docs.manus-meta.com/2.3.0/Plugins/SDK/Windows/SDK%20Minimal%20Client/",
)
GATE_NS = 40_000_000
P95_NS = 30_000_000
SHORT_GAP_NS = 500_000_000
# This is the exact two-step transformation of the existing raw converter
# convert_handle_egodex_v3._pico_pose_to_rep103, not its dataset producer.
PICO_RH_TO_REP103 = np.array([[0., 0., 1.], [-1., 0., 0.], [0., 1., 0.]])
LH_TO_RH = np.diag([1., 1., -1.])
RAW_LH_TO_REP103 = PICO_RH_TO_REP103 @ LH_TO_RH

class SourceContractError(ValueError):
    pass

@dataclass
class RawSession(PicoSession):
    qpc_ns: np.ndarray = field(default_factory=lambda: np.empty(0, np.int64))
    mapping: dict[str, np.ndarray] = field(default_factory=dict)
    motion_valid: dict[str, np.ndarray] = field(default_factory=dict)
    native_motion: dict[str, np.ndarray] = field(default_factory=dict)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def json_rows(path: Path) -> list[tuple[int, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line, text in enumerate(stream, 1):
            if text.strip():
                try:
                    rows.append((line, json.loads(text)))
                except ValueError as exc:
                    raise SourceContractError(f"RAW_JSON_INVALID:{path.name}:{line}") from exc
    if not rows:
        raise SourceContractError(f"RAW_STREAM_EMPTY:{path.name}")
    return rows


def timeline(values, name: str) -> np.ndarray:
    original = np.asarray(values)
    if (original.ndim != 1 or not len(original)
            or not np.issubdtype(original.dtype, np.integer)):
        raise SourceContractError(f"RAW_QPC_INTEGER_VECTOR_REQUIRED:{name}")
    ts = original.astype(np.int64)
    if np.any(np.diff(ts) <= 0):
        raise SourceContractError(f"RAW_QPC_DUPLICATE_OR_BACKWARD:{name}")
    return ts


def nearest(source, targets) -> tuple[np.ndarray, np.ndarray]:
    source, targets = timeline(source, "source"), timeline(targets, "video")
    after = np.clip(np.searchsorted(source, targets), 0, len(source) - 1)
    before = np.maximum(after - 1, 0)
    selected = np.where(np.abs(source[before] - targets) <= np.abs(source[after] - targets), before, after)
    return selected, source[selected] - targets


def raw_pose_to_rep103(value: dict) -> tuple[np.ndarray, bool]:
    try:
        if not isinstance(value, dict):
            return np.full((4, 4), np.nan), False
        if isinstance(value.get("pos"), list) and isinstance(value.get("quat"), list):
            pose = np.array(value["pos"] + value["quat"], dtype=float)
        else:
            pose = np.array([float(v) for v in str(value.get("pose", "")).split(",")])
        matrix, good = pose_matrices(pose)
        if not bool(good):
            return np.full((4, 4), np.nan), False
        # PICO LH -> RH z reflection, followed by the acquisition REP103 axes.
        rh_rotation = LH_TO_RH @ matrix[:3, :3] @ LH_TO_RH
        result = np.eye(4)
        result[:3, :3] = PICO_RH_TO_REP103 @ rh_rotation @ PICO_RH_TO_REP103.T
        result[:3, 3] = PICO_RH_TO_REP103 @ LH_TO_RH @ matrix[:3, 3]
        return result, True
    except (ValueError, TypeError, KeyError):
        return np.full((4, 4), np.nan), False


def read_installation(source_dir: Path) -> tuple[dict, dict]:
    import h5py
    hdf5 = source_dir / "dataset.hdf5"
    with h5py.File(hdf5, "r") as stream:
        # Intentionally no dataset indexing or iteration in this producer.
        raw = stream.attrs["controller_to_wrist_calibration"]
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8-sig")
        prior = json.loads(raw)
    installation_matrices(prior)
    encoded = json.dumps(prior, sort_keys=True, separators=(",", ":")).encode()
    return prior, {"source_hdf5": ref(hdf5), "attribute": "controller_to_wrist_calibration",
                   "prior_sha256": hashlib.sha256(encoded).hexdigest(),
                   "prior_authority": "INSTALLATION_PRIOR_NOT_MEASURED_ANATOMICAL_WRIST"}


def ref(path: Path) -> dict:
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def route_quality(skew: np.ndarray, computable: np.ndarray) -> dict:
    a = np.abs(np.asarray(skew, np.int64))
    valid = np.asarray(computable, bool) & (a <= GATE_NS)
    coverage = float(valid.mean())
    maximum, p95 = int(a.max()), float(np.percentile(a, 95))
    return {"max_abs_ms": maximum / 1e6, "p95_abs_ms": p95 / 1e6,
            "complete_coverage": coverage, "valid_count": int(valid.sum()),
            "frames": len(a), "passed": maximum <= GATE_NS and p95 <= P95_NS and coverage >= .95}


def _diagnostic_hand() -> np.ndarray:
    """Finite display-only hand when no source anchor exists; never valid."""
    hand = np.zeros((21, 3))
    for finger, y in enumerate([-.045, -.025, 0., .025, .045]):
        for joint in range(4):
            hand[1 + finger * 4 + joint] = [.03 + .028 * joint, y, 0.]
    return hand


def fill_points(values, observed, qpc_ns) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, float).copy()
    observed, ts = np.asarray(observed, bool), timeline(qpc_ns, "fill")
    label = np.where(observed, "RAW_COMPOSED", "NONE").astype("U32")
    display_valid = observed.copy()
    anchors = np.flatnonzero(observed)
    if not len(anchors):
        values[:] = _diagnostic_hand() if values.shape[1:] == (21, 3) else 0.
        label[:] = "DIAGNOSTIC_NO_ANCHOR"
        return values, label, display_valid
    for frame in np.flatnonzero(~observed):
        pos = np.searchsorted(anchors, frame)
        left = int(anchors[max(pos - 1, 0)])
        right = int(anchors[min(pos, len(anchors) - 1)])
        if left != right and ts[right] - ts[left] <= SHORT_GAP_NS:
            phase = (ts[frame] - ts[left]) / float(ts[right] - ts[left])
            ease = .5 - .5 * np.cos(np.pi * phase)
            values[frame] = (1. - ease) * values[left] + ease * values[right]
            label[frame], display_valid[frame] = "INTERP_SHORT", True
        else:
            chosen = left if abs(ts[frame] - ts[left]) <= abs(ts[right] - ts[frame]) else right
            values[frame] = values[chosen]
            if abs(ts[frame] - ts[chosen]) <= SHORT_GAP_NS and (frame < anchors[0] or frame > anchors[-1]):
                label[frame], display_valid[frame] = "EDGE_HOLD_SHORT", True
            else:
                label[frame] = "DIAGNOSTIC_LONG_HOLD"
    return values, label, display_valid


def fill_transforms(values, observed, qpc_ns) -> tuple[np.ndarray, np.ndarray]:
    out = np.asarray(values, float).copy()
    observed, ts = np.asarray(observed, bool), timeline(qpc_ns, "head")
    label = np.where(observed, "RAW_POSE", "NONE").astype("U32")
    anchors = np.flatnonzero(observed)
    if not len(anchors):
        out[:] = np.eye(4)
        label[:] = "DIAGNOSTIC_NO_ANCHOR"
        return out, label
    for frame in np.flatnonzero(~observed):
        pos = np.searchsorted(anchors, frame)
        left = int(anchors[max(pos - 1, 0)])
        right = int(anchors[min(pos, len(anchors) - 1)])
        if left != right and ts[right] - ts[left] <= SHORT_GAP_NS:
            phase = (ts[frame] - ts[left]) / float(ts[right] - ts[left])
            out[frame] = np.eye(4)
            out[frame, :3, :3] = Slerp([0., 1.], Rotation.from_matrix(out[[left, right], :3, :3]))([phase]).as_matrix()[0]
            out[frame, :3, 3] = (1. - phase) * out[left, :3, 3] + phase * out[right, :3, 3]
            label[frame] = "INTERP_SHORT"
        else:
            chosen = left if abs(ts[frame] - ts[left]) <= abs(ts[right] - ts[frame]) else right
            out[frame] = out[chosen]
            label[frame] = "EDGE_HOLD_SHORT" if abs(ts[frame] - ts[chosen]) <= SHORT_GAP_NS else "DIAGNOSTIC_LONG_HOLD"
    return out, label


def validate_manus_topology(row: dict) -> str:
    """Match the evidenced raw25 schema; never infer anatomy from MCP aliases.

    Raw SDK IDs are version-dependent. Unknown IDs, parent edges, categories or
    mixed category conventions are rejected rather than silently using this map.
    Numeric SDK enum values and canonical SDK names are explicit supported forms.
    """
    if not isinstance(row, dict) or type(row.get("node_count")) is not int or row["node_count"] != 25:
        raise SourceContractError("RAW_MANUS_TOPOLOGY_NODE_COUNT")
    ids, parents = row.get("node_ids"), row.get("parent_ids")
    if not isinstance(ids, list) or any(type(v) is not int for v in ids) or ids != list(range(25)):
        raise SourceContractError("RAW_MANUS_TOPOLOGY_NODE_IDS")
    if not isinstance(parents, list) or any(type(v) is not int for v in parents) or tuple(parents) != RAW_SDK_PARENT_IDS:
        raise SourceContractError("RAW_MANUS_TOPOLOGY_PARENTS")
    chains = row.get("chain_types")
    chain_aliases = {"Hand": "Hand", "ChainType_Hand": "Hand"}
    for finger in ("Thumb", "Index", "Middle", "Ring", "Pinky"):
        for name in (finger, "Finger" + finger, "ChainType_Finger" + finger):
            chain_aliases[name] = finger
    if not isinstance(chains, list) or len(chains) != 25 or any(not isinstance(v, str) for v in chains):
        raise SourceContractError("RAW_MANUS_TOPOLOGY_CHAIN_TYPES")
    if tuple(chain_aliases.get(v) for v in chains) != RAW_SDK_CHAIN_TYPES:
        raise SourceContractError("RAW_MANUS_TOPOLOGY_CHAIN_TYPES")
    kinds = row.get("joint_types")
    if not isinstance(kinds, list) or len(kinds) != 25:
        raise SourceContractError("RAW_MANUS_TOPOLOGY_JOINT_TYPES")
    if tuple(kinds) == RAW_CAPTURE_JOINT_ALIASES:
        return "CAPTURE_MCP_PIP_IP_ALIASES_DISAMBIGUATED_BY_VERIFIED_SDK_RAW25"
    if all(isinstance(v, str) for v in kinds):
        normalized = tuple(v.removeprefix("FingerJointType_") for v in kinds)
        if normalized == RAW_SDK_JOINT_TYPES:
            return "SDK_RAW25_CANONICAL_CATEGORY_NAMES"
    enum_names = ("Invalid", "Metacarpal", "Proximal", "Intermediate", "Distal", "Tip")
    if all(type(v) is int and 0 <= v < len(enum_names) for v in kinds):
        if tuple(enum_names[v] for v in kinds) == RAW_SDK_JOINT_TYPES:
            return "SDK_RAW25_FINGERJOINTTYPE_ENUM_VALUES"
    raise SourceContractError("RAW_MANUS_TOPOLOGY_JOINT_TYPES_ALIAS_CONFLICT_OR_UNSUPPORTED")


def joint_mapping_metadata(native: dict) -> dict:
    qualification = str(native["manus_topology_qualification"].item())
    return {"schema": JOINT_MAPPING_SCHEMA,
            "authority": "VERIFIED_SDK_RAW25_TO_CANONICAL_MANO21",
            "source_topology_qualification": qualification,
            "raw_node_ids_scope": "ONLY_VERIFIED_25_NODE_TOPOLOGY_NO_UNKNOWN_VERSION_INFERENCE",
            "sdk_binary_version_recorded": False,
            "raw_joint_types": native["raw_manus_joint_types"].tolist(),
            "sdk_joint_types_25": list(RAW_SDK_JOINT_TYPES),
            "raw_parent_ids": list(RAW_SDK_PARENT_IDS),
            "raw_chain_types": list(RAW_SDK_CHAIN_TYPES),
            "raw_alias_conflict": qualification.startswith("CAPTURE_MCP_PIP_IP"),
            "raw_alias_resolution": "nonthumb MCP/PIP/IP aliases represent SDK Metacarpal/Proximal/Intermediate; anatomical MCP/PIP are Proximal/Intermediate",
            "raw25_to_canonical21": MAP_25_TO_21.tolist(),
            "canonical_joint_names_21": list(CANONICAL_JOINT_NAMES_21),
            "legacy_parent_mapping_for_audit_only": LEGACY_PARENT_MAP_25_TO_21.tolist(),
            "tip_raw_ids_unchanged": [4, 9, 14, 19, 24],
            "raw25_positions_preserved": True,
            "official_references": list(SDK_TOPOLOGY_REFERENCES),
            "proof_review": {"path": str(Path(__file__).resolve().parents[3] / "review/pick_place_1008/mapping_sdk25_to_canonical21_audit_v1.json"),
                             "bytes": 81671, "sha256": "68171cfd524bf82aa25d7accca1f369c5799db34b225af43fa5513c82909b210"},
            "control_ground_truth": False, "physical_calibration_verified": False}


def load_raw_motion(source_dir: Path, qpc_ns: np.ndarray, *, expected_serial: str,
                    prior: dict) -> tuple[dict, dict, dict]:
    qpc_ns = timeline(qpc_ns, "video")
    pico = json_rows(source_dir / "raw/pico.jsonl")
    if any(row.get("dev") != expected_serial for _, row in pico):
        raise SourceContractError("RAW_PICO_DEVICE_IDENTITY_MISMATCH")
    ticks = timeline([row["recv_qpc_ns"] for _, row in pico], "PICO")
    chosen, skew = nearest(ticks, qpc_ns)
    n = len(qpc_ns)
    head = np.full((n, 4, 4), np.nan)
    controller = np.full((n, 2, 4, 4), np.nan)
    head_valid, ctrl_valid = np.zeros(n, bool), np.zeros((n, 2), bool)
    for frame, index in enumerate(chosen):
        data = pico[int(index)][1].get("data", {})
        head[frame], good = raw_pose_to_rep103(data.get("Head", {}))
        try:
            status = int(data.get("Head", {}).get("status", 0))
        except (ValueError, TypeError, AttributeError):
            status = 0
        head_valid[frame] = good and (status & 3) == 3
        for side, name in enumerate(SIDES):
            controller[frame, side], ctrl_valid[frame, side] = raw_pose_to_rep103(data.get("Controller", {}).get(name, {}))
    head_valid &= abs(skew) <= GATE_NS
    ctrl_valid &= (abs(skew) <= GATE_NS)[:, None]
    mapping = {"video_qpc_ns": qpc_ns, "frame_id": np.arange(n, dtype=np.int64),
               "pico_source_row": chosen,
               "pico_source_line": np.array([pico[int(i)][0] for i in chosen], np.int64),
               "pico_source_qpc_ns": ticks[chosen], "pico_skew_ns": skew,
               "pico_source_sensor_ns": np.array([pico[int(i)][1].get("data", {}).get("timeStampNs", -1) for i in chosen], np.int64),
               "pico_source_wall_ns": np.array([pico[int(i)][1].get("recv_wall_ns", -1) for i in chosen], np.int64)}
    manus = json_rows(source_dir / "raw/manus.jsonl")
    qualifications = {validate_manus_topology(row) for _, row in manus}
    if len(qualifications) != 1:
        raise SourceContractError("RAW_MANUS_TOPOLOGY_CATEGORY_SCHEMA_DRIFT")
    topology_qualification = next(iter(qualifications))
    manus_meta = read_json(source_dir / "raw/manus.meta.json")
    local = np.full((n, 2, 25, 3), np.nan)
    manus_valid = np.zeros((n, 2), bool)
    quality = {"head": route_quality(skew, head_valid),
               "controller_left": route_quality(skew, ctrl_valid[:, 0]),
               "controller_right": route_quality(skew, ctrl_valid[:, 1])}
    for side, name in enumerate(SIDES):
        rows = [(line, row) for line, row in manus if row.get("side") == name]
        if not rows:
            raise SourceContractError(f"RAW_MANUS_SIDE_EMPTY:{name}")
        times = timeline([row["recv_qpc_ns"] for _, row in rows], f"MANUS_{name}")
        index, offset = nearest(times, qpc_ns)
        expected_glove = manus_meta.get("calibration", {}).get(name, {}).get("glove_id")
        if expected_glove is None:
            raise SourceContractError(f"RAW_MANUS_CALIBRATION_ID_MISSING:{name}")
        for frame, idx in enumerate(index):
            row = rows[int(idx)][1]
            if row.get("glove_id") != expected_glove:
                raise SourceContractError(f"RAW_MANUS_GLOVE_ID_MISMATCH:{name}")
            try:
                nodes = np.asarray(row.get("nodes", []), dtype=float)
            except (ValueError, TypeError):
                nodes = np.empty((0, 7), dtype=float)
            good = (row.get("calibration_applied") is True and row.get("node_count") == 25
                    and row.get("node_ids") == list(range(25)) and nodes.shape == (25, 7)
                    and np.isfinite(nodes).all() and np.allclose(nodes[0, :3], 0., atol=1e-6)
                    and np.max(np.linalg.norm(nodes[:, :3], axis=-1)) < 1.)
            if good:
                # Raw MANUS nodes are native RH wrist-local; do not apply the
                # PICO world basis to these wrist-local coordinates a second time.
                local[frame, side] = nodes[:, :3]
                manus_valid[frame, side] = abs(offset[frame]) <= GATE_NS
        mapping[f"manus_{name}_source_row"] = index
        mapping[f"manus_{name}_source_line"] = np.array([rows[int(i)][0] for i in index], np.int64)
        mapping[f"manus_{name}_source_qpc_ns"] = times[index]
        mapping[f"manus_{name}_skew_ns"] = offset
        mapping[f"manus_{name}_source_wall_ns"] = np.array([rows[int(i)][1].get("recv_wall_ns", -1) for i in index], np.int64)
        mapping[f"manus_{name}_collector_qpc_ns"] = np.array([rows[int(i)][1].get("collector_recv_qpc_ns", -1) for i in index], np.int64)
        quality[f"manus_{name}"] = route_quality(offset, manus_valid[:, side])
    install = installation_matrices(prior)
    wrists = controller @ install[None]
    world = np.einsum("nsij,nskj->nski", wrists[:, :, :3, :3], local) + wrists[:, :, None, :3, 3]
    original_valid = ctrl_valid & manus_valid & np.isfinite(world).all(axis=(2, 3))
    hands = np.empty((2, n, 21, 3))
    labels = np.empty((2, n), "U32")
    display_valid = np.empty((2, n), bool)
    for side in range(2):
        hands[side], labels[side], display_valid[side] = fill_points(world[:, side, MAP_25_TO_21], original_valid[:, side], qpc_ns)
    head_display, head_labels = fill_transforms(head, head_valid, qpc_ns)
    complete = head_valid & original_valid.all(axis=1)
    quality["complete_coverage"] = float(complete.mean())
    quality["passed"] = all(q["passed"] for q in quality.values() if isinstance(q, dict)) and quality["complete_coverage"] >= .95
    native = {"head": head, "head_display": head_display, "head_source": head_labels,
              "controller": controller, "wrist": wrists, "manus_local_25": local,
              "hands": hands, "hand_source": labels,
              "head_valid": head_valid, "controller_valid": ctrl_valid,
              "manus_valid": manus_valid, "hand_observed": original_valid.T,
              "hand_display_valid": display_valid, "complete_valid": complete,
              "joint_observed_local": np.broadcast_to(manus_valid[:, :, None], (n, 2, 21)).copy()}
    native["joint_observed_local"][:, :, 0] = False
    native["manus25_to21"] = MAP_25_TO_21.copy()
    native["joint_names_21"] = np.asarray(CANONICAL_JOINT_NAMES_21)
    native["manus_topology_qualification"] = np.asarray(topology_qualification)
    native["raw_manus_joint_types"] = np.asarray(manus[0][1]["joint_types"])
    return native, mapping, quality


def pts_from_qpc(qpc_ns) -> np.ndarray:
    qpc = timeline(qpc_ns, "video")
    pts = ((qpc - qpc[0] + 500) // 1000).astype(np.int64)
    if np.any(np.diff(pts) <= 0):
        raise SourceContractError("RAW_QPC_PTS_QUANTIZATION_COLLISION")
    return pts


def encode_qpc_frames(path: Path, frames, qpc_ns, *, width: int, height: int) -> dict:
    import av
    pts = pts_from_qpc(qpc_ns)
    tick = Fraction(1, 1_000_000)
    if path.exists():
        raise SourceContractError("RAW_INPUT_VIDEO_EXISTS")
    count = 0
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=Fraction(30, 1))
        stream.width, stream.height = width, height
        stream.pix_fmt, stream.time_base = "yuv420p", tick
        stream.codec_context.time_base = tick
        stream.codec_context.thread_count = 2
        stream.options = {"crf": "12", "preset": "fast", "bf": "0"}
        for count, pixels in enumerate(frames, 1):
            if count > len(pts) or pixels.shape != (height, width, 3):
                raise SourceContractError("RAW_VIDEO_FRAME_DOMAIN_OR_COUNT")
            frame = av.VideoFrame.from_ndarray(pixels, format="bgr24")
            frame.pts, frame.time_base = int(pts[count - 1]), tick
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    if count != len(pts):
        raise SourceContractError(f"RAW_VIDEO_QPC_FRAME_COUNT:{count}:{len(pts)}")
    recovered = []
    with av.open(str(path)) as container:
        container.streams.video[0].codec_context.thread_count = 2
        for frame in container.decode(video=0):
            if (frame.width, frame.height) != (width, height) or frame.pts is None:
                raise SourceContractError("RAW_ENCODED_FRAME_DOMAIN_OR_PTS")
            recovered.append(float(frame.pts * frame.time_base))
        recovered_tick = float(container.streams.video[0].time_base)
    if len(recovered) != len(pts):
        raise SourceContractError("RAW_ENCODED_FRAME_COUNT_MISMATCH")
    error = np.abs(np.asarray(recovered) - pts / 1e6)
    if error.max(initial=0.) > max(float(tick), recovered_tick) + 1e-9:
        raise SourceContractError("RAW_ENCODED_PTS_DRIFT")
    return {"time_base": "1/1000000", "quantization_max_ns": 500,
            "recovered_time_base_seconds": recovered_tick, "pts_roundtrip_max_error_seconds": float(error.max(initial=0.)),
            "decoded_frames": len(recovered), "frame_order": "RAW_VIDEO_PICTURE_ORDER_WITH_QPC_PTS"}


def crop_frames(source_dir: Path, camera: CameraModel):
    import av
    import cv2
    cv2.setNumThreads(1)
    with av.open(str(source_dir / "raw/vst.h264")) as container:
        container.streams.video[0].codec_context.thread_count = 2
        for frame in container.decode(video=0):
            if (frame.width, frame.height) != (2 * camera.native_width, camera.native_height):
                raise SourceContractError("RAW_SBS_FRAME_DOMAIN_MISMATCH")
            pixels = frame.to_ndarray(format="bgr24")
            x, y, w, h = camera.crop
            yield cv2.resize(pixels[y:y+h, x:x+w], (camera.width, camera.height), interpolation=cv2.INTER_AREA)


def _project_output(path: Path) -> Path:
    repo = Path(__file__).resolve().parents[3]
    path = path.resolve()
    if not path.is_relative_to(repo) or path == repo:
        raise SourceContractError("RAW_OUTPUT_OUTSIDE_PROJECT")
    return path


def validated_installation_override(selected_config: dict, original_reference: dict,
                                    device_serial: str) -> dict:
    from . import audit
    expected_sha = original_reference["prior_sha256"]
    expected_group = device_serial + ":" + expected_sha
    if (selected_config.get("installation_original_prior_sha256") != expected_sha
            or selected_config.get("installation_group_key") != expected_group
            or selected_config.get("fit_quality_status") != "PASSED"):
        raise SourceContractError("RAW_INSTALLATION_OVERRIDE_GROUP_OR_QUALITY_MISMATCH")
    evidence = selected_config.get("installation_correction_evidence")
    if not isinstance(evidence, dict):
        raise SourceContractError("RAW_INSTALLATION_CORRECTION_EVIDENCE_MISSING")
    audit._verify_exact_ref(evidence, project_only=True)
    report = read_json(Path(evidence["path"]))
    if report.get("schema") == "pick_place_fixed_encoded_k_right_installation_visual_fit_v1":
        from .installation_fixed_k import validate_override_evidence, FixedKInstallationError
        try:
            return validate_override_evidence(selected_config, original_reference, device_serial)
        except FixedKInstallationError as exc:
            raise SourceContractError(str(exc)) from exc
    group = report.get("groups", {}).get(expected_group, {})
    prior = selected_config["installation_prior_override"]
    if (report.get("schema") != "pick_place_installation_group_visual_fit_v1"
            or group.get("status") != "PASSED"
            or group.get("device_serial") != device_serial
            or group.get("original_prior_sha256") != expected_sha
            or group.get("prior_used") != prior):
        raise SourceContractError("RAW_INSTALLATION_OVERRIDE_EVIDENCE_GROUP_MISMATCH")
    installation_matrices(prior)
    return prior


def prepare_session(source_dir: Path, output_dir: Path, config: dict,
                    expected_frames: int | None = None) -> RawSession:
    source_dir = Path(source_dir).resolve(strict=True)
    output_dir = _project_output(Path(output_dir))
    session_manifest = read_json(source_dir / "manifest.json")
    expected_id = "pick_place_1008_" + source_dir.name
    if session_manifest.get("session") != expected_id:
        raise SourceContractError("RAW_SESSION_IDENTITY_MISMATCH")
    camera_params = read_json(source_dir / "raw/camera_params.json")
    camera_meta = read_json(source_dir / "raw/camera_params.meta.json")
    serial = camera_meta.get("device_serial")
    selected_config = config.get("devices", {}).get(serial, config)
    required = ["manifest.json", "raw/camera_params.json", "raw/camera_params.meta.json",
                "raw/pico.jsonl", "raw/manus.jsonl", "raw/manus.meta.json", "raw/vst.h264",
                "raw/vst.qpc.ts.jsonl", "raw/vst.ts.jsonl", "dataset.hdf5"]
    source_refs = {name: ref(source_dir / name) for name in required}
    qpc_rows, wall_rows = json_rows(source_dir / "raw/vst.qpc.ts.jsonl"), json_rows(source_dir / "raw/vst.ts.jsonl")
    qpc = timeline([v for _, v in qpc_rows], "video")
    wall = np.asarray([v for _, v in wall_rows], dtype=np.int64)
    if len(wall) != len(qpc) or (expected_frames is not None and expected_frames != len(qpc)):
        raise SourceContractError("RAW_VIDEO_QPC_EXPECTED_COUNT_MISMATCH")
    if len(qpc) < 2 or np.median(np.diff(qpc)) < 1_000_000:
        raise SourceContractError("RAW_VIDEO_QPC_UNIT_IMPLAUSIBLE")
    camera = normalize_camera(camera_params, camera_meta, device_serial=serial,
                              camera_sha256=source_refs["raw/camera_params.json"]["sha256"],
                              config=selected_config, output_size=tuple(selected_config.get("output_size", [1280, 960])))
    prior, installation_ref = read_installation(source_dir)
    if "installation_prior_override" in selected_config:
        prior = validated_installation_override(selected_config, installation_ref, serial)
        installation_ref["override_evidence"] = selected_config["installation_correction_evidence"]
    used_prior_bytes = json.dumps(prior, sort_keys=True, separators=(",", ":")).encode()
    installation_ref["used_prior_sha256"] = hashlib.sha256(used_prior_bytes).hexdigest()
    native, mapping, quality = load_raw_motion(source_dir, qpc, expected_serial=serial, prior=prior)
    mapping["video_wall_ns"] = wall
    mapping["video_qpc_source_line"] = np.array([line for line, _ in qpc_rows], np.int64)
    mapping["video_pts_us"] = pts_from_qpc(qpc)
    output_dir.mkdir(parents=True, exist_ok=False)
    video = output_dir / "physical_eye.mp4"
    video_result = encode_qpc_frames(video, crop_frames(source_dir, camera), qpc,
                                     width=camera.width, height=camera.height)
    count = len(qpc)
    fps = (count - 1) * 1e9 / int(qpc[-1] - qpc[0])
    c2w = native["head_display"] @ camera.T_head_camera
    c2w_factory = native["head_display"] @ camera.T_head_camera_factory
    calibration = dict(camera.calibration)
    calibration.update({"actual_motion_source": "RAW_PICO_CONTROLLER_PLUS_RAW_MANUS_INSTALLATION_PRIOR",
                        "world_frame": "REP103_X_FORWARD_Y_LEFT_Z_UP_METRES",
                        "raw_pose_conversion": "LH_z_reflection_then_acquisition_PICO_RH_to_REP103; xyz=[-raw_z,-raw_x,raw_y]",
                        "raw_pose_conversion_reference": "src/chaoyang/ops/convert_handle_egodex_v3.py:_pico_pose_to_rep103",
                        "installation": installation_ref, "installation_prior_used": prior,
                        "calibration_group_key": serial + ":" + camera.camera_sha256 + ":" + installation_ref["used_prior_sha256"],
                        "time_quality": quality, "timebase_status": "TIMEBASE_UNVERIFIED_HOST_ARRIVAL_QPC_DECLARED_NS",
                        "time_alignment_mode": "PER_RAW_VIDEO_QPC_NEAREST_NATIVE_PICO_AND_INDEPENDENT_MANUS_SIDES",
                        "video_encoding": video_result,
                        "body_source": "ABSENT_EXISTING_V13_SHOULDER_ELBOW_PRIOR",
                        "wrist_observation_authority": "DERIVED_INSTALLATION_PRIOR_NOT_ANATOMICAL_OBSERVATION",
                        "joint_names_25": list(MANUS_NAMES), "manus25_to21": MAP_25_TO_21.tolist(),
                        "joint_names_21": list(CANONICAL_JOINT_NAMES_21),
                        "hand_joint_schema": JOINT_MAPPING_SCHEMA,
                        "manus_joint_mapping": joint_mapping_metadata(native),
                        "raw_collector_joint_names_25_for_audit": list(LEGACY_PARENT_MANUS_NAMES),
                        "joint_root_observed": False, "preview_read": False, "hdf5_datasets_read": False,
                        "training_eligible": False, "control_ground_truth": False})
    valid = {"head_valid": native["head_valid"], "controller_valid": native["controller_valid"],
             "manus_valid": native["manus_valid"], "hand_display_valid": native["hand_display_valid"],
             "complete_valid": native["complete_valid"], "joint_observed_local": native["joint_observed_local"]}
    session = RawSession(expected_id, output_dir, video, fps, count, camera.width, camera.height,
                         camera.K, c2w, c2w_factory, native["hands"], native["hand_observed"],
                         native["hand_source"], np.zeros((count, 24, 3)), np.zeros(count, bool),
                         np.array([0., 0., -1.]), calibration, qpc, mapping, valid, native)
    arrays = {"K": session.K, "c2w": session.c2w, "c2w_editor": session.c2w_editor,
              "hands": session.hands, "hand_observed": session.hand_observed,
              "hand_source": session.hand_source, "body": session.body,
              "body_observed": session.body_observed, "gravity": session.gravity, "qpc_ns": qpc}
    arrays.update({"mapping__" + k: v for k, v in mapping.items()})
    arrays.update({"valid__" + k: v for k, v in valid.items()})
    arrays.update({"native__" + k: v for k, v in native.items()})
    np.savez_compressed(output_dir / "source.npz", **arrays)
    after_refs = {name: ref(source_dir / name) for name in required}
    if after_refs != source_refs:
        raise SourceContractError("RAW_SOURCE_DRIFT")
    meta = {"schema_version": RAW_INPUT_SCHEMA, "status": "COMPLETE_RAW_INPUT",
            "session_id": expected_id, "source_dir": str(source_dir), "source_refs": source_refs,
            "frame_count": count, "fps": fps, "width": camera.width, "height": camera.height,
            "video": ref(video), "arrays": ref(output_dir / "source.npz"), "calibration": calibration}
    with (output_dir / "SESSION_INPUT.json").open("x", encoding="utf-8") as stream:
        json.dump(meta, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return session


def load_prepared(output_dir: Path) -> RawSession:
    output_dir = _project_output(Path(output_dir))
    meta = read_json(output_dir / "SESSION_INPUT.json")
    if meta.get("status") != "COMPLETE_RAW_INPUT":
        raise SourceContractError("RAW_PREPARED_NOT_COMPLETE")
    if (meta.get("schema_version") != RAW_INPUT_SCHEMA
            or meta.get("calibration", {}).get("hand_joint_schema") != JOINT_MAPPING_SCHEMA
            or meta["calibration"].get("manus25_to21") != MAP_25_TO_21.tolist()):
        raise SourceContractError("RAW_PREPARED_CANONICAL21_SCHEMA_REQUIRED")
    for name in ("video", "arrays"):
        if not Path(meta[name]["path"]).resolve(strict=True).is_relative_to(output_dir):
            raise SourceContractError("RAW_PREPARED_COMPONENT_OUTSIDE_SESSION")
        if ref(Path(meta[name]["path"])) != meta[name]:
            raise SourceContractError("RAW_PREPARED_SHA_DRIFT")
    with np.load(meta["arrays"]["path"], allow_pickle=False) as stream:
        arrays = {key: stream[key] for key in stream.files}
    group = lambda prefix: {key[len(prefix):]: value for key, value in arrays.items() if key.startswith(prefix)}
    return RawSession(meta["session_id"], output_dir, Path(meta["video"]["path"]), meta["fps"],
                      meta["frame_count"], meta["width"], meta["height"], arrays["K"], arrays["c2w"],
                      arrays["c2w_editor"], arrays["hands"], arrays["hand_observed"], arrays["hand_source"],
                      arrays["body"], arrays["body_observed"], arrays["gravity"], meta["calibration"],
                      arrays["qpc_ns"], group("mapping__"), group("valid__"), group("native__"))


def load_rgb_frames(session: RawSession, width: int = 640, height: int = 480) -> np.ndarray:
    """Decode original pinned SBS pixels; intermediate MP4 is review-only.

    Returns uint8 RGB in native picture order. No lens warp, preview decode or
    compressed prepared-eye roundtrip enters the Clean producer.
    """
    import av
    import cv2
    cv2.setNumThreads(1)
    meta = read_json(session.raw_dir / "SESSION_INPUT.json")
    if meta.get("session_id") != session.session_id or meta.get("status") != "COMPLETE_RAW_INPUT":
        raise SourceContractError("RAW_RGB_SESSION_IDENTITY_OR_STATUS")
    pin = meta["source_refs"]["raw/vst.h264"]
    source = Path(pin["path"]).resolve(strict=True)
    if ref(source) != pin:
        raise SourceContractError("RAW_RGB_SOURCE_SHA_DRIFT")
    calibration = meta["calibration"]
    native_w, native_h = calibration["native_resolution"]
    index = calibration["sbs_index"]
    if type(index) is not int or index not in (0, 1) or calibration.get("lens_remap") is not False:
        raise SourceContractError("RAW_RGB_ROUTE_OR_LENS_POLICY")
    if not np.isclose(width/native_w, height/native_h, atol=1e-12):
        raise SourceContractError("RAW_RGB_NONUNIFORM_RESIZE")
    frames = []
    with av.open(str(source)) as container:
        container.streams.video[0].codec_context.thread_count = 2
        for frame in container.decode(video=0):
            if (frame.width, frame.height) != (2*native_w, native_h):
                raise SourceContractError("RAW_RGB_SBS_DOMAIN")
            rgb = frame.to_ndarray(format="rgb24")
            frames.append(cv2.resize(rgb[:, index*native_w:(index+1)*native_w],
                                     (width, height), interpolation=cv2.INTER_AREA))
    if len(frames) != session.frame_count or len(frames) != len(session.qpc_ns):
        raise SourceContractError("RAW_RGB_QPC_FRAME_COUNT")
    if ref(source) != pin:
        raise SourceContractError("RAW_RGB_SOURCE_DRIFT_WHILE_DECODING")
    return np.stack(frames)
