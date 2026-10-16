"""Explicit, same-session input adaptation for two authorized V1 reviews.

No model, robot solver, mask or compositor lives here. Encoded VST pixels are
already undistorted: physical-eye crop and resize only. Factory K and mount
priors are visual assumptions, not an encoded-domain calibration certificate.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

MANIFEST_ENV = "CHAOYANG_EXACT78_PREPARED_INPUTS"
WIDTH, HEIGHT = 1280, 960
REP_TO_XR = np.array([[0., -1., 0.], [0., 0., 1.], [-1., 0., 0.]])
MANUS21 = np.array([0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14, 15, 16, 18, 19, 20, 21, 23, 24])


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(ref):
    p = Path(ref["path"]).resolve(strict=True)
    if p.stat().st_size != ref["bytes"] or sha(p) != ref["sha256"]:
        raise ValueError(f"NATIVE_INPUT_SHA_DRIFT:{p}")
    return p


def ref(path):
    p = Path(path).resolve(strict=True)
    return {"path": str(p), "bytes": p.stat().st_size, "sha256": sha(p)}


def nearest(source, targets):
    source = np.asarray(source, dtype=np.int64)
    targets = np.asarray(targets, dtype=np.int64)
    if len(source) < 2 or np.any(np.diff(source) <= 0) or np.any(np.diff(targets) <= 0):
        raise ValueError("NATIVE_TIMELINE_DUPLICATE_OR_BACKWARD")
    right = np.clip(np.searchsorted(source, targets), 0, len(source) - 1)
    left = np.maximum(right - 1, 0)
    return np.where(abs(source[left] - targets) <= abs(source[right] - targets), left, right)


def matrices(poses):
    poses = np.asarray(poses, dtype=float)
    if poses.shape[-1] != 7 or not np.isfinite(poses).all():
        raise ValueError("NATIVE_POSE_LAYOUT")
    flat = poses.reshape(-1, 7)
    if np.max(abs(np.linalg.norm(flat[:, 3:], axis=1) - 1)) > 1e-3:
        raise ValueError("NATIVE_QUATERNION_INVALID")
    result = np.broadcast_to(np.eye(4), (len(flat), 4, 4)).copy()
    result[:, :3, :3] = Rotation.from_quat(flat[:, 3:]).as_matrix()
    result[:, :3, 3] = flat[:, :3]
    return result.reshape(*poses.shape[:-1], 4, 4)


def head_to_optical(camera):
    from chaoyang.exact78_baseline.pico_source import HEAD_TO_CAMERA_TRANSLATION
    if camera.get("extrinsicConvention") != "head_to_camera_3x4_row_major":
        raise ValueError("NATIVE_CAMERA_CONVENTION_ABSENT")
    eye = camera["left"]
    if eye.get("sourceIndex") != 1 or not eye.get("isValidCalibration"):
        raise ValueError("NATIVE_PHYSICAL_LEFT_NOT_CONFIRMED")
    e = np.eye(4)
    e[:3] = np.asarray(eye["extrinsic"]).reshape(3, 4)
    if not np.allclose(e[:3, :3].T @ e[:3, :3], np.eye(3), atol=1e-5):
        raise ValueError("NATIVE_CAMERA_ROTATION_INVALID")
    # Capturelib sensor optical axes -> the upright encoded optical axes.
    # This is the same fixed axis conversion as the V1 editor camera, not
    # another pixel warp or a hand-fitted session-specific transform.
    upright = np.eye(4)
    upright[:3, :3] = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]
    x = np.linalg.inv(e) @ upright
    x[:3, 3] = HEAD_TO_CAMERA_TRANSLATION
    return x


def _pico(source, qpc):
    from chaoyang.exact78_baseline import pico_source as ps
    rows = [json.loads(line) for line in (source / "raw/pico.jsonl").read_text().splitlines() if line.strip()]
    ticks = np.array([r["recv_qpc_ns"] for r in rows], np.int64)
    chosen = nearest(ticks, qpc)
    selected = [rows[i]["data"] for i in chosen]
    head = matrices([[float(x) for x in d["Head"]["pose"].split(",")] for d in selected])
    hands = np.zeros((2, len(qpc), 21, 3))
    observed = np.zeros((2, len(qpc)), bool)
    labels = np.empty((2, len(qpc)), "U16")
    for side, name in enumerate(("left", "right")):
        for i, d in enumerate(selected):
            hand = d.get("Hand", {}).get(name + "Hand", {})
            js = hand.get("HandJointLocations", [])
            if hand.get("isActive") and len(js) == 26 and all(int(j.get("s", 0)) & ps.POSITION_VALID for j in js):
                xyz = np.array([[float(v) for v in j["p"].split(",")[:3]] for j in js])
                hands[side, i] = xyz[list(ps.OPENXR_TO_MANO)]
                observed[side, i] = np.isfinite(hands[side, i]).all()
        if not observed[side].any():
            raise ValueError(f"NATIVE_PICO_NO_HAND_ANCHOR:{name}")
        hands[side], labels[side] = ps.fill_track(ps.smooth_observed(hands[side], observed[side]), observed[side], 30.)
    return head, hands, observed, labels, chosen, ticks[chosen] - qpc, "PICO_OPENXR26_SAME_SESSION"


def _manus(source, qpc):
    import h5py
    with h5py.File(source / "dataset.hdf5", "r") as h:
        if h.attrs.get("alignment_clock") != "host_qpc" or "REP-103" not in str(h.attrs.get("coord")):
            raise ValueError("NATIVE_MANUS_CLOCK_OR_COORDINATE_CONTRACT")
        prior = json.loads(h.attrs["controller_to_wrist_calibration"])
        if (prior.get("schema") != "controller_to_wrist_v1"
                or prior.get("controller_pose_frame") != "pico_world_rh_x_forward_y_left_z_up"
                or prior.get("convention") != "wrist_pose = compose_pose(controller_pose, transform)"):
            raise ValueError("NATIVE_CONTROLLER_WRIST_PRIOR_CONTRACT")
        ticks = h["recv_qpc_ns"][:]
        chosen = nearest(ticks, qpc)
        head_rep = matrices(h["head_pose"][:][chosen])
        head = head_rep.copy()
        head[:, :3, :3] = REP_TO_XR @ head_rep[:, :3, :3] @ REP_TO_XR.T
        head[:, :3, 3] = head_rep[:, :3, 3] @ REP_TO_XR.T
        hands = np.zeros((2, len(qpc), 21, 3))
        observed = np.zeros((2, len(qpc)), bool)
        for side, name in enumerate(("left", "right")):
            controller = matrices(h[name + "_controller_pose"][:][chosen])
            install = matrices(np.r_[prior[name]["pos"], prior[name]["quat"]])
            wrist = controller @ install
            local = h[name + "_hand_joints"][:][chosen][:, MANUS21]
            if not np.allclose(local[:, 0], 0, atol=1e-6) or not np.isfinite(local).all():
                raise ValueError("NATIVE_MANUS_LOCAL_ORIGIN")
            world_rep = np.einsum("nij,nkj->nki", wrist[:, :3, :3], local) + wrist[:, None, :3, 3]
            hands[side] = world_rep @ REP_TO_XR.T
            observed[side] = h[name + "_hand_valid"][:][chosen]
            if not observed[side].all():
                raise ValueError("NATIVE_MANUS_INVALID_HAND_FRAME")
    return head, hands, observed, np.full(observed.shape, "MANUS_PRIOR", "U16"), chosen, ticks[chosen] - qpc, "PICO_CONTROLLER_PLUS_MANUS_INSTALLATION_PRIOR"


def prepare(manifest_path, output):
    from chaoyang.paths import get_paths
    from chaoyang.exact78_baseline.pipeline import Encoder, write_json
    cv2.setNumThreads(1)
    repo = get_paths().repo_root.resolve()
    manifest_path = Path(manifest_path).resolve(strict=True)
    if repo not in manifest_path.parents or repo not in output.resolve().parents:
        raise ValueError("NATIVE_MANIFEST_OR_OUTPUT_OUTSIDE_PROJECT")
    manifest = json.loads(manifest_path.read_text())
    mapping = {}
    for row in manifest["sessions"]:
        source = Path(row["source_dir"]).resolve(strict=True)
        for pin in row["inputs"]:
            verify(pin)
        folder = output / "inputs" / row["task"] / row["session_id"]
        folder.mkdir(parents=True, exist_ok=False)
        qpc = np.array([json.loads(x) for x in (source / "raw/vst.qpc.ts.jsonl").read_text().splitlines() if x.strip()], np.int64)
        wall = np.array([json.loads(x) for x in (source / "raw/vst.ts.jsonl").read_text().splitlines() if x.strip()], np.int64)
        if len(qpc) != len(wall) or np.any(np.diff(qpc) <= 0) or np.any(np.diff(wall) <= 0):
            raise ValueError("NATIVE_VIDEO_TIMESTAMP_INVALID")
        camera = json.loads((source / "raw/camera_params.json").read_text())
        x = head_to_optical(camera)
        if row["motion_source"] == "PICO_OPENXR26":
            head, hands, observed, labels, chosen, skew, kind = _pico(source, qpc)
        elif row["motion_source"] == "CONTROLLER_MANUS":
            head, hands, observed, labels, chosen, skew, kind = _manus(source, qpc)
        else:
            raise ValueError("NATIVE_UNAUTHORIZED_MOTION_SOURCE")
        if np.max(abs(skew)) > 500_000_000:
            raise ValueError("NATIVE_MOTION_TIME_GAP_EXCEEDS_V1_SHORT_GAP")
        fps = (len(qpc) - 1) * 1e9 / int(qpc[-1] - qpc[0])
        if not 20 <= fps <= 60:
            raise ValueError("NATIVE_QPC_RATE_IMPLAUSIBLE")
        c2w = head @ x
        eye = camera["left"]
        sx, sy = WIDTH / camera["width"], HEIGHT / camera["height"]
        if not np.isclose(sx, sy):
            raise ValueError("NATIVE_NONUNIFORM_RESIZE")
        k = np.array([[eye["fx"] * sx, 0, eye["cx"] * sx], [0, eye["fy"] * sy, eye["cy"] * sy], [0, 0, 1.]])
        video = folder / "physical_left.mp4"
        cap = cv2.VideoCapture(str(source / "raw/vst.h264"))
        enc = Encoder(video, WIDTH, HEIGHT, fps, crf=12)
        count = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame.shape[:2] != (int(camera["outputHeight"]), int(camera["outputWidth"])):
                    raise ValueError("NATIVE_SBS_FRAME_DOMAIN")
                mono = frame[:, camera["width"]:2 * camera["width"]]
                enc.write(cv2.resize(mono, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA))
                count += 1
        finally:
            cap.release()
            enc.close()
        if count != len(qpc) or count != row["expected_frames"]:
            raise ValueError(f"NATIVE_VIDEO_TIMESTAMP_FRAME_COUNT:{count}:{len(qpc)}")
        data = folder / "source.npz"
        np.savez_compressed(data, K=k, c2w=c2w, hands=hands, hand_observed=observed, hand_source=labels,
                            body=np.zeros((count, 24, 3)), body_observed=np.zeros(count, bool),
                            gravity=np.array([0., -1., 0.]), frame_id=np.arange(count),
                            video_qpc=qpc, video_wall=wall, motion_row=chosen, motion_skew_ns=skew)
        calibration = {"actual_motion_source": kind, "physical_eye": "left_sourceIndex_1", "lens_remap": False,
                       "camera_K_authority": "FACTORY_K_SCALED_UNVERIFIED_FOR_ENCODED_DOMAIN",
                       "head_to_camera_rotation": x[:3, :3].tolist(), "head_to_camera_translation_used": x[:3, 3].tolist(),
                       "head_to_camera_rotation_rule": "inverse_factory_extrinsic_times_capture_upright_axes",
                       "timebase_status": "TIMEBASE_UNVERIFIED_CAPTURE_IMPLEMENTATION_ABSENT",
                       "time_alignment_mode": "same_session_host_qpc_nearest_assuming_declared_ns",
                       "motion_sync_max_abs_ms": float(abs(skew).max() / 1e6),
                       "body_source": "ABSENT_V1_EXISTING_SHOULDER_AND_ELBOW_PRIOR",
                       "training_eligible": False, "control_ground_truth": False}
        meta = {"status": "VERIFIED_COMPLETE_INPUT", "session_id": row["session_id"], "frame_count": count,
                "fps": fps, "width": WIDTH, "height": HEIGHT, "video": ref(video), "arrays": ref(data),
                "source_dir": str(source), "source_inputs": row["inputs"], "calibration": calibration}
        write_json(folder / "SESSION_INPUT.json", meta)
        mapping[row["session_id"]] = ref(folder / "SESSION_INPUT.json")
    path = output / "PREPARED_INPUTS.json"
    write_json(path, {"schema_version": "exact78-explicit-native-inputs-v1", "sessions": mapping})
    return path


def find_prepared(session_id):
    from chaoyang.paths import get_paths
    path = Path(os.environ[MANIFEST_ENV]).resolve(strict=True)
    repo = get_paths().repo_root.resolve()
    if repo not in path.parents:
        raise ValueError("NATIVE_PREPARED_MANIFEST_OUTSIDE_PROJECT")
    pin = json.loads(path.read_text())["sessions"][session_id]
    meta = verify(pin)
    if repo not in meta.parents:
        raise ValueError("NATIVE_PREPARED_INPUT_OUTSIDE_PROJECT")
    return meta.parent


def load_prepared(session_id, folder):
    from chaoyang.exact78_baseline.pico_source import PicoSession
    meta = json.loads((folder / "SESSION_INPUT.json").read_text())
    if meta["status"] != "VERIFIED_COMPLETE_INPUT" or meta["session_id"] != session_id:
        raise ValueError("NATIVE_PREPARED_IDENTITY_OR_STATUS")
    video, data = verify(meta["video"]), verify(meta["arrays"])
    with np.load(data, allow_pickle=False) as a:
        return PicoSession(session_id=session_id, raw_dir=folder, video=video, fps=meta["fps"],
                           frame_count=meta["frame_count"], width=meta["width"], height=meta["height"],
                           K=a["K"], c2w=a["c2w"], c2w_editor=a["c2w"], hands=a["hands"],
                           hand_observed=a["hand_observed"], hand_source=a["hand_source"],
                           body=a["body"], body_observed=a["body_observed"], gravity=a["gravity"],
                           calibration=meta["calibration"])


def main():
    import argparse
    from chaoyang.paths import get_paths
    from chaoyang.governance.exact78_operation_gate import authorize_exact78_operation, _command_tail
    parser = argparse.ArgumentParser(description="Fenced input-only helper of the selected V1 operation")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = get_paths().repo_root.resolve()
    index = json.loads((repo / "tasks/current/INDEX.json").read_text())
    active = [r for r in index["task_packets"] if r.get("execution_allowed")]
    if len(active) != 1:
        raise ValueError("NATIVE_PREPARE_REQUIRES_LIVE_PACKET")
    packet = json.loads(Path(active[0]["packet_path"]).read_text())
    command = _command_tail(packet["commands"][0])
    if (command[0] != "run_exact78_takeover_baseline_v1"
            or command[command.index("--raw-input-manifest") + 1] != str(args.manifest.resolve())
            or command[command.index("--output-root") + 1] != str(args.output.resolve())):
        raise ValueError("NATIVE_PREPARE_PACKET_ARGUMENT_MISMATCH")
    authorize_exact78_operation(command[0], command[1:])
    prepare(args.manifest, args.output)


if __name__ == "__main__":
    main()
