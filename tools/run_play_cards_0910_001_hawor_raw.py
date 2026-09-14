#!/usr/bin/env python3
"""Materialize raw HaWoR MANO21 for the 0910 controller canary.

Run only through tools/hawor_python.sh.  This is a development baseline: direct
detector/model frames remain OBSERVED and missing frames remain missing.  The
same-session acquisition c2w is used; HaWoR SLAM is deliberately not rerun.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import cv2
import joblib
import numpy as np
import torch


PROJECT = Path("/mnt/workspace/code/chaoyang")
HAWOR = PROJECT / "third_party/HaWoR"
MANO_NAMES = (
    "wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
)
CHAINS = ((0,1,2,3,4),(0,5,6,7,8),(0,9,10,11,12),(0,13,14,15,16),(0,17,18,19,20))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--reuse-upstream", type=Path)
    args = ap.parse_args()
    session = args.session_root.resolve(strict=True)
    out = args.output_root.resolve()
    if out.exists():
        raise RuntimeError(f"fresh output required: {out}")
    out.mkdir(parents=True)
    work = out / "upstream_work"
    if args.reuse_upstream is not None:
        shutil.copytree(args.reuse_upstream.resolve(strict=True), work)
    else:
        work.mkdir()
    source_video = session / "CameraRecord_play_cards_0910_001.mp4"
    staged_video = work / source_video.name
    if not staged_video.is_file():
        shutil.copy2(source_video, staged_video)

    frame_jsons = sorted((session / "preprocess/all_data").glob("*/training_data.json"))
    if not frame_jsons:
        raise RuntimeError("no training_data.json frames")
    rows = [json.loads(path.read_text()) for path in frame_jsons]
    frame_count = len(rows)
    c2w = np.asarray([row["metadata"]["c2w"] for row in rows], dtype=np.float64)
    intrinsics = np.asarray([row["metadata"]["k"] for row in rows], dtype=np.float64)
    fps = float(rows[0]["metadata"]["fps"])

    sys.path.insert(0, str(HAWOR))
    from scripts.scripts_test_video.detect_track_video import detect_track_video
    from scripts.scripts_test_video.hawor_video import hawor_motion_estimation
    from hawor.utils.process import run_mano, run_mano_left
    from hawor.utils.rotation import rotation_matrix_to_angle_axis

    class A: pass
    a = A()
    a.video_path = str(staged_video)
    a.input_type = "file"
    a.img_focal = float(np.mean(intrinsics[:, [0,1], [0,1]]))
    a.checkpoint = str(HAWOR / "weights/hawor/checkpoints/hawor.ckpt")
    a.infiller_weight = str(HAWOR / "weights/hawor/checkpoints/infiller.pt")
    prior = Path.cwd()
    started = time.time()
    try:
        os.chdir(HAWOR)
        start, end, seq_folder_s, image_files = detect_track_video(a)
        chunks, _ = hawor_motion_estimation(a, start, end, seq_folder_s)
    finally:
        os.chdir(prior)
    seq_folder = Path(seq_folder_s)
    if len(image_files) != frame_count:
        raise RuntimeError(f"HaWoR decoded {len(image_files)} frames, expected {frame_count}")

    joints_cam = np.full((2, frame_count, 21, 3), np.nan, np.float32)
    joints_world = np.full_like(joints_cam, np.nan)
    joints_2d = np.full((2, frame_count, 21, 2), np.nan, np.float32)
    betas = np.full((2, frame_count, 10), np.nan, np.float32)
    root_rot = np.full((2, frame_count, 3, 3), np.nan, np.float32)
    hand_rot = np.full((2, frame_count, 15, 3, 3), np.nan, np.float32)
    observed = np.zeros((2, frame_count), bool)
    provenance = np.full((2, frame_count), "MISSING", dtype="<U24")
    confidence = np.zeros((2, frame_count), np.float32)
    boxes = np.full((2, frame_count, 4), np.nan, np.float32)

    tracks = np.load(seq_folder / f"tracks_{start}_{end}/model_tracks.npy", allow_pickle=True).item()
    handed_rows = {0: [], 1: []}
    for track in tracks.values():
        det_rows = [row for row in track if bool(row.get("det", False))]
        if not det_rows:
            continue
        handed = np.asarray([float(np.asarray(row.get("det_handedness", [0])).reshape(-1)[0]) for row in det_rows])
        side = 1 if float(handed.mean()) >= 0.5 else 0
        handed_rows[side].extend(track)
    for side in (0,1):
        for row in handed_rows[side]:
            frame = int(row.get("frame", -1))
            if not (0 <= frame < frame_count) or not bool(row.get("det", False)):
                continue
            box_raw = np.asarray(row.get("det_box", np.full(4, np.nan)), dtype=np.float32).reshape(-1)
            if box_raw.size >= 4:
                boxes[side, frame] = box_raw[:4]
            score = row.get("det_score", row.get("score", row.get("confidence", 0.5)))
            confidence[side, frame] = float(box_raw[4] if box_raw.size >= 5 else np.asarray(score).reshape(-1)[0])

    materialize_prior = Path.cwd()
    try:
        os.chdir(HAWOR)
        for side, mano in ((0, run_mano_left), (1, run_mano)):
            for frame_chunk in chunks.get(side, []):
                frames = np.asarray(frame_chunk, dtype=np.int64)
                p = seq_folder / "cam_space" / str(side) / f"{frames[0]}_{frames[-1]}.json"
                payload = json.loads(p.read_text())
                rr = np.asarray(payload["init_root_orient"], np.float32)[0]
                hr = np.asarray(payload["init_hand_pose"], np.float32)[0]
                tr = np.asarray(payload["init_trans"], np.float32)[0]
                be = np.asarray(payload["init_betas"], np.float32)[0]
                with torch.inference_mode():
                    result = mano(
                        torch.from_numpy(tr[None]),
                        rotation_matrix_to_angle_axis(torch.from_numpy(rr[None])),
                        rotation_matrix_to_angle_axis(torch.from_numpy(hr[None])),
                        betas=torch.from_numpy(be[None]),
                        use_cuda=False,
                    )["joints"][0].detach().cpu().numpy().astype(np.float32)
                joints_cam[side, frames] = result
                root_rot[side, frames] = rr
                hand_rot[side, frames] = hr
                betas[side, frames] = be
                observed[side, frames] = True
                provenance[side, frames] = "OBSERVED"
                confidence[side, frames] = np.maximum(confidence[side, frames], 0.5)
    finally:
        os.chdir(materialize_prior)

    for side in (0,1):
        for frame in np.flatnonzero(observed[side]):
            xyz = joints_cam[side, frame].astype(np.float64)
            joints_world[side, frame] = (xyz @ c2w[frame,:3,:3].T + c2w[frame,:3,3]).astype(np.float32)
            q = xyz @ intrinsics[frame].T
            joints_2d[side, frame] = (q[:,:2] / q[:,2:3]).astype(np.float32)

    npz = out / "HAWOR_RAW_MANO21.npz"
    np.savez_compressed(
        npz, joints_3d_camera=joints_cam, joints_3d_world=joints_world,
        joints_2d=joints_2d, betas=betas, root_orient_camera=root_rot,
        hand_pose_rotmat=hand_rot, observed=observed, provenance=provenance,
        detector_confidence=confidence, detector_boxes_xyxy=boxes, c2w=c2w,
        intrinsics=intrinsics, original_frame_indices=np.arange(frame_count,dtype=np.int32),
        fps=np.asarray(fps), mano_joint_names=np.asarray(MANO_NAMES),
        anatomical_side_names=np.asarray(("left","right")), mano_wrist_index=np.asarray(0,np.int32),
        mano_tip_indices=np.asarray((4,8,12,16,20),np.int32),
        mano_mcp_indices=np.asarray((1,5,9,13,17),np.int32),
    )

    review = out / "HAWOR_RAW_REVIEW.mp4"
    cap = cv2.VideoCapture(str(source_video))
    writer = cv2.VideoWriter(str(review), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280,960))
    colors = ((255,120,20),(20,30,255))
    try:
        for frame in range(frame_count):
            ok, image = cap.read()
            if not ok: raise RuntimeError(f"video ended at {frame}")
            for side in (0,1):
                if not observed[side,frame]: continue
                uv = np.rint(joints_2d[side,frame]).astype(np.int32)
                for chain in CHAINS: cv2.polylines(image,[uv[np.asarray(chain)]],False,colors[side],3,cv2.LINE_AA)
                cv2.circle(image,tuple(uv[0]),7,colors[side],-1,cv2.LINE_AA)
            cv2.rectangle(image,(0,0),(1279,60),(0,0,0),-1)
            cv2.putText(image,f"play_cards_0910_001 | HaWoR raw | frame {frame:03d}",(12,27),cv2.FONT_HERSHEY_SIMPLEX,.68,(255,255,255),2,cv2.LINE_AA)
            cv2.putText(image,f"left={'OBS' if observed[0,frame] else 'MISS'} right={'OBS' if observed[1,frame] else 'MISS'} | BLUE=L RED=R",(12,52),cv2.FONT_HERSHEY_SIMPLEX,.58,(255,255,255),2,cv2.LINE_AA)
            writer.write(image)
    finally:
        cap.release(); writer.release()
    result = {
        "schema_version":"play-cards-0910-hawor-raw-baseline-v1",
        "status":"COMPLETE_BASELINE_NO_QUALITY_GATE",
        "session_id":"play_cards_0910_001","frame_count":frame_count,"fps":fps,
        "observed_frames":{"left":int(observed[0].sum()),"right":int(observed[1].sum()),"bilateral":int(np.all(observed,axis=0).sum())},
        "npz":{"path":str(npz),"bytes":npz.stat().st_size,"sha256":sha256(npz)},
        "review":{"path":str(review),"bytes":review.stat().st_size,"sha256":sha256(review)},
        "camera_motion_source":"SAME_SESSION_ACQUISITION_C2W_NOT_HAWOR_SLAM",
        "control_ground_truth":False,"physical_deployment":False,
        "claim_limit":"Raw HaWoR development prediction only; missing frames are not fabricated.",
        "wall_seconds":time.time()-started,
    }
    (out/"RESULT.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(result,ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
