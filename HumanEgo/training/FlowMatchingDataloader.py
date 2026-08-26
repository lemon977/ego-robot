# -*- coding: utf-8 -*-
# @FileName: FlowMatchingDataloader.py
"""
The Ultimate FlowMatchingDataloader for Multi-Object Flow Matching & Paradigm Ablations.

Features & Ablations Supported:
- Variable Object Topology: Parses N objects and hands into a sequence of Interaction-Centric Tokens (ICTs).
- Visual Input Ablation: Supports raw RGB, inpainted RGB, or no visual input (State-only).
- Coordinate System Ablation: Toggles between Object-Centric and Ego-Centric reference frames.
- Action Representation Ablation: Supports 'absolute' or 'delta' action spaces.
- Kinematic Latching: Freezes T_obj_in_hand upon grasp to fix visual occlusion jitter via FK.
- Safe Temporal Stride: Dynamically caps stride near sequence end to prevent frozen-time bugs.
- 3D Geometry Injection: Explicitly extracts point clouds (PCD) for spatial boundary awareness.
- Dual-Hand Ready: Automatically expands Action and Heatmap dimensions for dual-arm tasks.
"""

from __future__ import annotations

import os
import json
import random
import time
import concurrent.futures
from dataclasses import dataclass
from typing import List, Optional, Tuple, Any, Dict

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as ScipyR
from scipy.spatial.transform import Slerp

# DataLoader and preload already provide process/thread parallelism. Prevent
# OpenCV from nesting another native thread pool inside every worker.
cv2.setNumThreads(1)

# --- Custom Unified Utils ---
from utils.utils_io import read_json, safe_imread_gray, safe_imread_rgb
from utils.utils_math import (
    clip01,
    rotmat_to_o6d,
    o6d_to_rotmat,
    normalize_o6d,
    normalize_pos,
    unnormalize_pos,
    interpolate_pose,
    get_rrc_params,
    apply_photometric_aug,
    apply_random_erasing
)

# =========================================================
# ---- Hyperparameters & Constants ----
# =========================================================

JSON_NAME = "training_data.json"
IMG_NAME = "rgb_WoArm_WArmObjKpts.png"

# Mapping from hand_tracking_method → entity key in training_data.json
HAND_METHOD_ENTITY_KEY = {
    "aria_mps":   "hands",
    "mediapipe":  "hands_mediapipe",
    "wilor":      "hands_wilor",
    "hamer":      "hands_hamer",
    "hawor_v3":   "hands_hawor_v3",
}
DEFAULT_STATS = {
    'pos': {
        'mean':[0.0, 0.0, 0.0],
        'std':[1.0, 1.0, 1.0]
    }
}

# Token Type IDs (For Transformer Positional / Semantic Identification)
TYPE_PAD = 0.0
TYPE_HAND_L = 1.0
TYPE_HAND_R = 2.0
TYPE_OBJ_ANCHOR = 3.0
TYPE_OBJ_OTHER = 4.0

# Number of points sampled per entity for explicit Geometry Injection
MAX_PTS_PER_ENTITY = 64

# --- Augmentation Switches ---
AUG_ENABLE_DEFAULT = True

# Photometric Augmentation
AUG_IMG = True
AUG_IMG_PROB = 0.8
AUG_BRIGHTNESS_DELTA = 0.20
AUG_CONTRAST_DELTA = 0.20
AUG_GAMMA_DELTA = 0.15
AUG_NOISE_STD = 0.02
AUG_BLUR_PROB = 0.15
AUG_BLUR_KSIZE = 3
AUG_GRAY_PROB = 0.10
AUG_HUE_DELTA = 10
AUG_SAT_RANGE = (0.6, 1.4)

# Random Resized Crop (Viewpoint variability)
AUG_RRC = True
AUG_RRC_PROB = 0.5
AUG_RRC_SCALE = (0.7, 1.0)
AUG_RRC_RATIO = (0.9, 1.1)

# Target Action Jittering
AUG_TARGET_JITTERING = True
AUG_TARGET_POS_STD = 0.001
AUG_TARGET_ROT_STD = 0.5

# Random Erasing (Cutout)
AUG_CUTOUT = True
AUG_CUTOUT_PROB = 0.5
AUG_CUTOUT_N_HOLES = (3, 8)
AUG_CUTOUT_SIZE = (0.05, 0.2)

# Sub-step Interpolation (Temporal Stride)
AUG_TEMPORAL_STRIDE = False
AUG_STRIDE_RANGE = (1, 3)

# Sub-step Interpolation (Legacy Feature: interpolate between adjacent frames)
AUG_INTERP_PROB = 0.5

# =========================================================
# -------- Dataset Class --------
# =========================================================

@dataclass
class MPSSessions:
    mps_path: str


class FlowMatchingDataloader(Dataset):
    """
    The Ultimate ICT Dataloader for Paradigm Ablations.
    """

    def __init__(
        self,
        sessions: List[MPSSessions],
        image_size: Tuple[int, int] = (240, 320),
        pred_horizon: int = 50,
        single_hand: bool = True,
        single_hand_side: str = "right",
        max_ict: int = 8,

        # --- Global Paradigm Ablation Switches ---
        img_name: Optional[str] = IMG_NAME,
        centric_mode: str = 'object_centric',       # 'object_centric' | 'ego_centric'
        frame_mode: str = 'anchor_frame',            # 'anchor_frame' | 'camera_frame'
        action_mode: str = 'absolute',              # 'absolute' or 'delta'
        hand_action_representation: str = 'grasp_binary',
        robot_sidecar_root: Optional[str] = None,
        use_pcd_features: bool = True,              # Explicit 3D Point Cloud Injection

        # --- Aux Switches ---
        use_aux_obj_dynamics: bool = True,
        use_aux_visual_foresight: bool = True,
        use_aux_temporal_contrastive: bool = True,

        enable_augmentation: bool = AUG_ENABLE_DEFAULT,
        enable_aug_img: bool = AUG_IMG,
        enable_aug_rrc: bool = AUG_RRC,
        enable_aug_target_jittering: bool = AUG_TARGET_JITTERING,
        enable_aug_cutout: bool = AUG_CUTOUT,
        enable_aug_temporal_stride: bool = AUG_TEMPORAL_STRIDE,
        enable_aug_interpolation: bool = False,
        seed: int = 7,
        stats: Optional[Dict] = None,
        disable_kinematic_latching: bool = False,

        # --- Hand Tracking Method ---
        hand_tracking_method: str = "aria_mps",   # 也支持独立 HumanEgoAdapter 的 "hawor_v3"

        # --- Legacy Compatibility Switches ---
        use_legacy_image_loading: bool = False,   # True: read from JSON abs path at original res
        use_legacy_rng: bool = False,             # True: deterministic RNG per sample (seed + idx*N)
        cache_json_in_memory: bool = False,
        cache_image_bytes_in_memory: bool = False,
    ):
        super().__init__()
        assert pred_horizon >= 1, "pred_horizon must be >= 1"

        self.sessions = sessions
        self.H, self.W = image_size
        self.pred_horizon = pred_horizon
        self.single_hand = single_hand
        self.single_hand_side = single_hand_side
        self.max_ict = max_ict

        self.img_name = img_name
        self.centric_mode = centric_mode
        self.frame_mode = frame_mode
        self.action_mode = action_mode
        if hand_action_representation not in {
            "grasp_binary", "kaihand_joint_state", "wuji_joint_state"
        }:
            raise ValueError(
                "hand_action_representation must be grasp_binary, "
                "kaihand_joint_state or wuji_joint_state"
            )
        self.hand_action_representation = hand_action_representation
        self.hand_command_dim = {
            "grasp_binary": 1,
            "kaihand_joint_state": 22,
            "wuji_joint_state": 20,
        }[hand_action_representation]
        self.embodiment = {
            "kaihand_joint_state": "kai22",
            "wuji_joint_state": "wuji20",
        }.get(hand_action_representation)
        self.robot_sidecar_root = (
            os.path.abspath(robot_sidecar_root) if robot_sidecar_root else None
        )
        self.use_pcd_features = use_pcd_features
        self.use_object_tokens = (centric_mode != 'ego_centric')  # derived from centric_mode

        self.use_aux_obj_dynamics = use_aux_obj_dynamics
        self.use_aux_visual_foresight = use_aux_visual_foresight
        self.use_aux_temporal_contrastive = use_aux_temporal_contrastive

        self.enable_augmentation = enable_augmentation
        self.enable_aug_img = enable_aug_img
        self.enable_aug_rrc = enable_aug_rrc
        self.enable_aug_target_jittering = enable_aug_target_jittering
        self.enable_aug_cutout = enable_aug_cutout
        self.enable_aug_temporal_stride = enable_aug_temporal_stride
        self.enable_aug_interpolation = enable_aug_interpolation
        self.seed = seed

        self.disable_kinematic_latching = disable_kinematic_latching
        self.hand_tracking_method = hand_tracking_method
        self.hand_entity_key = HAND_METHOD_ENTITY_KEY.get(hand_tracking_method, "hands")
        self.use_legacy_image_loading = use_legacy_image_loading
        self.use_legacy_rng = use_legacy_rng
        self.cache_json_in_memory = cache_json_in_memory
        self.cache_image_bytes_in_memory = cache_image_bytes_in_memory
        # The stock loader opens K future JSON files for every sample.  On a
        # network filesystem this means roughly 1.4M small-file reads per epoch
        # for the grap_a_cap cohort.  Preloading keeps the exact parsed objects
        # and target construction semantics, while DataLoader fork workers share
        # these read-only pages through copy-on-write.
        self._json_cache: Dict[str, dict] = {}
        self._image_bytes_cache: Dict[str, bytes] = {}
        self._image_array_cache: Dict[str, np.ndarray] = {}
        self.stats = stats if stats is not None else DEFAULT_STATS
        self.pos_mean = np.array(self.stats['pos']['mean'], dtype=np.float32)
        self.pos_std  = np.array(self.stats['pos']['std'], dtype=np.float32)

        # Token Dimension:[TypeID(1) + Pose_in_Ref(9) + HandL_in_This(9) + (Optional HandR_in_This(9)) + Flag(1)]
        self.ict_dim = 20 if self.single_hand else 29

        # KaiHand supervision is intentionally kept as a session-level sidecar.
        # This preserves the stock HumanEgo JSON schema and avoids rewriting every
        # frame.  Values are the URDF joint positions in radians, in the exact
        # joint_names order stored by PicoKaiHandVisualReplacement.py.
        self._robot_sources: Dict[str, Dict[str, Any]] = {}
        if self.embodiment is not None:
            self._load_robot_sources()

        self.samples: List[str] = []
        self._build_index()
        if self.cache_image_bytes_in_memory and self.img_name:
            self._preload_image_bytes()

        print(f"[FlowMatchingDataloader] Built {len(self.samples)} samples.")
        print(f"  - Hand Method  : {self.hand_tracking_method} (key='{self.hand_entity_key}')")
        print(f"  - Image Input  : {self.img_name if self.img_name else 'DISABLED (State-only)'}")
        print(f"  - Frame Mode   : {self.frame_mode.upper()}")
        print(f"  - Action Mode  : {self.action_mode.upper()}")
        print(
            f"  - Hand Command : {self.hand_action_representation} "
            f"({self.hand_command_dim}D/hand)"
        )
        print(f"  - Aux  : ObjDynamics={self.use_aux_obj_dynamics} | VisForesight={self.use_aux_visual_foresight} | TempContrastive={self.use_aux_temporal_contrastive}")
        print(f"  - 3D PCD Feats : {self.use_pcd_features}")
        if self.cache_json_in_memory:
            print(f"  - JSON Cache   : {len(self._json_cache)} frames in memory")
        if self.cache_image_bytes_in_memory:
            cached_bytes = sum(map(len, self._image_bytes_cache.values()))
            cached_bytes += sum(value.nbytes for value in self._image_array_cache.values())
            cached_gib = cached_bytes / (1024 ** 3)
            print(
                f"  - Image Cache  : "
                f"{len(self._image_bytes_cache) + len(self._image_array_cache)} files / "
                f"{cached_gib:.2f} GiB cached image data in memory"
            )

    def __len__(self):
        return len(self.samples)

    # ---------------------
    # Index Builder
    # ---------------------
    @staticmethod
    def _session_root_from_json_path(json_path: str) -> str:
        all_data_dir = os.path.dirname(os.path.dirname(json_path))
        return os.path.dirname(os.path.dirname(all_data_dir))

    def _load_robot_sources(self) -> None:
        expected_names: Dict[str, Tuple[str, ...]] = {}
        missing = []
        for spec in self.sessions:
            root = os.path.abspath(spec.mps_path)
            session_id = os.path.basename(os.path.dirname(root))
            path = (
                os.path.join(
                    self.robot_sidecar_root, self.embodiment, session_id,
                    "sidecar.npz",
                )
                if self.robot_sidecar_root else ""
            )
            if not os.path.isfile(path):
                missing.append(path)
                continue
            with np.load(path, allow_pickle=False) as archive:
                schema = str(archive["schema_version"].item())
                if schema != "humanego-robot-sidecar-v1":
                    raise ValueError(f"{path}: unsupported schema {schema}")
                if str(archive["embodiment"].item()) != self.embodiment:
                    raise ValueError(f"{path}: embodiment mismatch")
                frame_names = [str(v) for v in archive["frame_names"]]
                source: Dict[str, Any] = {
                    "path": path,
                    "frame_to_index": {
                        name: index for index, name in enumerate(frame_names)
                    },
                }
                q_all = np.asarray(archive["q"], dtype=np.float32)
                active_all = np.asarray(archive["valid"], dtype=bool)
                confidence_all = np.asarray(archive["confidence"], dtype=np.float32)
                wrist_all = np.asarray(archive["wrist_T_camera"], dtype=np.float32)
                joint_names = np.asarray(archive["joint_names"]).astype(str)
                joint_lower = np.asarray(archive["joint_lower"], dtype=np.float32)
                joint_upper = np.asarray(archive["joint_upper"], dtype=np.float32)
                for side_index, side in enumerate(("left", "right")):
                    q = q_all[:, side_index]
                    active = active_all[:, side_index]
                    names = tuple(joint_names[side_index])
                    if q.shape != (len(frame_names), self.hand_command_dim):
                        raise ValueError(
                            f"{path}: {side}_q must have shape "
                            f"({len(frame_names)}, {self.hand_command_dim}), got {q.shape}"
                        )
                    if side not in expected_names:
                        expected_names[side] = names
                    elif names != expected_names[side]:
                        raise ValueError(
                            f"{path}: KaiHand joint order differs between sessions"
                        )
                    source[f"{side}_q"] = q
                    source[f"{side}_active"] = active
                    source[f"{side}_confidence"] = confidence_all[:, side_index]
                    source[f"{side}_wrist_T_camera"] = wrist_all[:, side_index]
                    source[f"{side}_joint_names"] = names
                    source[f"{side}_joint_lower"] = joint_lower[side_index]
                    source[f"{side}_joint_upper"] = joint_upper[side_index]
                self._robot_sources[root] = source
        if missing:
            raise FileNotFoundError(
                f"{self.embodiment} supervision requested, but frozen sidecar "
                "is missing:\n  " + "\n  ".join(missing)
            )
        self.robot_joint_names = expected_names

    def _get_robot_state(
        self, json_path: str, side: str
    ) -> Optional[Dict[str, Any]]:
        root = self._session_root_from_json_path(json_path)
        source = self._robot_sources.get(root)
        if source is None:
            return None
        frame_name = os.path.basename(os.path.dirname(json_path))
        frame_index = source["frame_to_index"].get(frame_name)
        if frame_index is None or not source[f"{side}_active"][frame_index]:
            return None
        q = source[f"{side}_q"][frame_index]
        wrist = source[f"{side}_wrist_T_camera"][frame_index]
        if not np.isfinite(q).all() or not np.isfinite(wrist).all():
            return None
        return {
            "q": q.copy(),
            "wrist_T_camera": wrist.copy(),
            "confidence": float(source[f"{side}_confidence"][frame_index]),
        }

    def _build_index(self):
        self.samples.clear()
        cache_started = time.time()
        for spec in self.sessions:
            all_data_dir = os.path.join(spec.mps_path, "preprocess", "all_data")
            if not os.path.isdir(all_data_dir):
                continue

            frame_names = sorted([d for d in os.listdir(all_data_dir) if d.isdigit()])
            for fn in frame_names:
                jpath = os.path.join(all_data_dir, fn, JSON_NAME)
                if not os.path.exists(jpath):
                    continue

                d0 = read_json(jpath)
                if self.cache_json_in_memory and d0 is not None:
                    self._json_cache[jpath] = d0
                if d0 is None or not self._frame_has_required_fields(d0):
                    continue
                if self.embodiment is not None:
                    required_sides = (
                        [self.single_hand_side]
                        if self.single_hand else ["left", "right"]
                    )
                    # At least one hand is enough for the dual-hand network;
                    # each future hand has its own mask.
                    if not any(
                        self._get_robot_state(jpath, side) is not None
                        for side in required_sides
                    ):
                        continue
                    # The final frame has no future action to supervise.  The
                    # Robot supervision never treats a missing future frame as
                    # a held ground-truth action.
                    if not os.path.isfile(self._get_future_json_path(jpath, 1)):
                        continue

                self.samples.append(jpath)
        if self.cache_json_in_memory:
            print(
                f"[FlowMatchingDataloader] Preloaded {len(self._json_cache)} "
                f"frame JSONs in {time.time() - cache_started:.1f}s"
            )

    def _read_frame(self, json_path: str) -> Optional[dict]:
        """Read one frame, preferring the immutable preload cache."""
        if self.cache_json_in_memory:
            cached = self._json_cache.get(json_path)
            if cached is not None:
                return cached
        return read_json(json_path)

    def _frame_exists(self, json_path: str) -> bool:
        if self.cache_json_in_memory and json_path in self._json_cache:
            return True
        return bool(json_path) and os.path.exists(json_path)

    def _image_path_from_frame(self, frame: dict) -> str:
        if self.use_legacy_image_loading:
            return frame.get("obs", {}).get(
                "rgb_WoArm_WArmObjKpts_path", ""
            )
        return self._resolve_image_path(frame)

    def _preload_image_bytes(self) -> None:
        if not self.cache_json_in_memory:
            raise ValueError(
                "cache_image_bytes_in_memory requires cache_json_in_memory"
            )
        started = time.time()
        paths = sorted({
            image_path
            for frame in self._json_cache.values()
            if (image_path := self._image_path_from_frame(frame))
            and image_path not in self._image_bytes_cache
            and image_path not in self._image_array_cache
        })

        def read_cached(image_path: str) -> tuple[str, bytes | np.ndarray]:
            try:
                with open(image_path, "rb") as stream:
                    encoded = stream.read()
            except OSError:
                return image_path, b""
            if self.use_legacy_image_loading:
                return image_path, encoded
            bgr = cv2.imdecode(
                np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if bgr is None:
                bgr = np.zeros((self.H, self.W, 3), dtype=np.uint8)
            elif bgr.shape[:2] != (self.H, self.W):
                bgr = cv2.resize(
                    bgr, (self.W, self.H), interpolation=cv2.INTER_LINEAR
                )
            return image_path, bgr

        total_bytes = 0
        workers = max(1, min(16, (os.cpu_count() or 2) - 4))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            for image_path, cached in executor.map(read_cached, paths):
                if isinstance(cached, bytes) and not cached:
                    continue
                if isinstance(cached, bytes):
                    self._image_bytes_cache[image_path] = cached
                    total_bytes += len(cached)
                else:
                    self._image_array_cache[image_path] = cached
                    total_bytes += cached.nbytes
        print(
            f"[FlowMatchingDataloader] Preloaded "
            f"{len(self._image_bytes_cache) + len(self._image_array_cache)} images "
            f"({total_bytes / (1024 ** 3):.2f} GiB, {workers} workers) in "
            f"{time.time() - started:.1f}s"
        )

    def _frame_has_required_fields(self, d: dict) -> bool:
        if not isinstance(d, dict): return False

        # Legacy mode: require image path exists in JSON
        if self.use_legacy_image_loading:
            obs = d.get("obs", {})
            if not isinstance(obs, dict): return False
            if obs.get("rgb_WoArm_WArmObjKpts_path") is None: return False

        hands = d.get("entities", {}).get(self.hand_entity_key, {})
        if hands is None:
            hands = {}  # method not generated → treat as empty
        if self.single_hand and self.single_hand_side not in hands:
            return False
        return True

    def _get_future_json_path(self, json_path: str, step: int) -> str:
        frame_dir = os.path.dirname(json_path)
        all_data_dir = os.path.dirname(frame_dir)
        try:
            t = int(os.path.basename(frame_dir))
        except Exception:
            return ""
        fut_dir = os.path.join(all_data_dir, f"{t + step:05d}")
        return os.path.join(fut_dir, JSON_NAME)

    # ---------------------
    # Geometry Architecture
    # ---------------------
    def _get_T_w2ref(self, d: dict) -> np.ndarray:
        w_trans = d["metadata"]["world_transforms"]
        if self.frame_mode == 'anchor_frame':
            T_ref_w = np.array(w_trans["virtual_static_anchor"], dtype=np.float32)
        elif self.frame_mode == 'camera_frame':
            T_ref_w = np.array(w_trans["cam0"], dtype=np.float32)
        else:
            raise ValueError(f"Unknown frame_mode: {self.frame_mode}")
        return np.linalg.inv(T_ref_w)

    def _encode_geometry(self, T_matrix: np.ndarray) -> np.ndarray:
        pos = normalize_pos(T_matrix[:3, 3], self.pos_mean, self.pos_std)
        o6d = normalize_o6d(rotmat_to_o6d(T_matrix[:3, :3]))
        return np.concatenate([pos, o6d])

    def _pad_point_cloud(self, pts: np.ndarray) -> np.ndarray:
        out = np.zeros((MAX_PTS_PER_ENTITY, 3), dtype=np.float32)
        if pts is None or len(pts) == 0:
            return out
        n = len(pts)
        if n >= MAX_PTS_PER_ENTITY:
            indices = np.linspace(0, n - 1, MAX_PTS_PER_ENTITY, dtype=int)
            out[:] = pts[indices]
        else:
            out[:n] = pts
            pad_indices = np.random.choice(n, MAX_PTS_PER_ENTITY - n, replace=True)
            out[n:] = pts[pad_indices]
        return out

    # ---------------------
    # ICT Building
    # ---------------------
    def _build_ict(self, d: dict, T_w2ref: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        tokens = []
        pcds =[]

        ents = d.get("entities", {})
        hands = ents.get(self.hand_entity_key, {})
        if hands is None: hands = {}  # null means method not generated
        objs = ents.get("objects", {})
        anchor_key = d["metadata"].get("anchor_key", "obj1")
        obs_kpts = d.get("obs", {}).get("objects_kpts", {})

        T_hL_w = np.array(hands["left"]["T_hand_to_world"]) if "left" in hands else None
        T_hR_w = np.array(hands["right"]["T_hand_to_world"]) if "right" in hands else None

        def calc_hand_relations(T_ent_w: np.ndarray) -> np.ndarray:
            rels =[]
            T_w2ent = np.linalg.inv(T_ent_w)
            if self.single_hand:
                T_h = T_hR_w if self.single_hand_side == "right" else T_hL_w
                rels.append(self._encode_geometry(T_w2ent @ T_h) if T_h is not None else np.zeros(9, dtype=np.float32))
            else:
                rels.append(self._encode_geometry(T_w2ent @ T_hL_w) if T_hL_w is not None else np.zeros(9, dtype=np.float32))
                rels.append(self._encode_geometry(T_w2ent @ T_hR_w) if T_hR_w is not None else np.zeros(9, dtype=np.float32))
            return np.concatenate(rels)

        # 1. Add Hand Tokens
        n_hand_rel = 9 if self.single_hand else 18   # size of hand_in_hand vector
        hand_sides = [self.single_hand_side] if self.single_hand else ["left", "right"]
        for side in hand_sides:
            if side in hands:
                T_h_w = np.array(hands[side]["T_hand_to_world"])
                grasp = float(hands[side]["grasp"])
                # Binarize grasp for consistency (aria=binary, teleop may be continuous)
                grasp = 1.0 if grasp > 0.5 else 0.0
                type_id = TYPE_HAND_L if side == "left" else TYPE_HAND_R

                pose_in_ref = self._encode_geometry(T_w2ref @ T_h_w)
                if self.use_object_tokens:
                    hand_in_hand = calc_hand_relations(T_h_w)
                else:
                    # Pure ego: zero out hand-to-entity relations
                    hand_in_hand = np.zeros(n_hand_rel, dtype=np.float32)
                tok = np.concatenate([[type_id], pose_in_ref, hand_in_hand, [grasp]])
                tokens.append(tok.astype(np.float32))

                p_hand_ref = (T_w2ref @ T_h_w)[:3, 3]
                pcds.append(self._pad_point_cloud(np.array([p_hand_ref])))
            else:
                tokens.append(np.zeros(self.ict_dim, dtype=np.float32))
                pcds.append(np.zeros((MAX_PTS_PER_ENTITY, 3), dtype=np.float32))

        if self.use_object_tokens:
            # 2. Add Anchor Object Token
            if anchor_key in objs:
                T_anc_w = np.array(objs[anchor_key]["T_obj_to_world"])
                pose_in_ref = self._encode_geometry(T_w2ref @ T_anc_w)
                hand_in_anc = calc_hand_relations(T_anc_w)

                tok = np.concatenate([[TYPE_OBJ_ANCHOR], pose_in_ref, hand_in_anc, [-1.0]])
                tokens.append(tok.astype(np.float32))

                pts_w = np.array(obs_kpts.get(anchor_key, {}).get("world", []))
                pts_ref = (T_w2ref[:3, :3] @ pts_w.T).T + T_w2ref[:3, 3] if len(pts_w) > 0 else np.array([(T_w2ref @ T_anc_w)[:3, 3]])
                pcds.append(self._pad_point_cloud(pts_ref))

            # 3. Add Other Object Tokens
            for k, v in objs.items():
                if k == anchor_key: continue
                T_obj_w = np.array(v["T_obj_to_world"])
                pose_in_ref = self._encode_geometry(T_w2ref @ T_obj_w)
                hand_in_obj = calc_hand_relations(T_obj_w)

                tok = np.concatenate([[TYPE_OBJ_OTHER], pose_in_ref, hand_in_obj, [-1.0]])
                tokens.append(tok.astype(np.float32))

                pts_w = np.array(obs_kpts.get(k, {}).get("world", []))
                pts_ref = (T_w2ref[:3, :3] @ pts_w.T).T + T_w2ref[:3, 3] if len(pts_w) > 0 else np.array([(T_w2ref @ T_obj_w)[:3, 3]])
                pcds.append(self._pad_point_cloud(pts_ref))

        # 4. Final Padding & Masking
        state = np.zeros((self.max_ict, self.ict_dim), dtype=np.float32)
        pcd_state = np.zeros((self.max_ict, MAX_PTS_PER_ENTITY, 3), dtype=np.float32)
        mask = np.zeros(self.max_ict, dtype=bool)
        n_tok = min(len(tokens), self.max_ict)

        for i in range(n_tok):
            if tokens[i][0] == TYPE_PAD: continue
            state[i] = tokens[i]
            pcd_state[i] = pcds[i]
            if tokens[i][0] != TYPE_PAD:
                mask[i] = True

        return state, pcd_state, mask

    # ---------------------
    # Legacy Sub-step Interpolation Helpers
    # ---------------------
    def _interpolate_T_w2ref(self, d0: dict, d1: dict, alpha: float) -> np.ndarray:
        """Interpolates the snapshot reference frame for sub-step training (legacy feature)."""
        T_w2ref_0 = self._get_T_w2ref(d0)
        T_w2ref_1 = self._get_T_w2ref(d1)

        T_ref0_w = np.linalg.inv(T_w2ref_0)
        T_ref1_w = np.linalg.inv(T_w2ref_1)

        p0, R0 = T_ref0_w[:3, 3], T_ref0_w[:3, :3]
        p1, R1 = T_ref1_w[:3, 3], T_ref1_w[:3, :3]

        p_a, R_a = interpolate_pose(p0, R0, p1, R1, alpha)
        T_ref_a_w = np.eye(4, dtype=np.float32)
        T_ref_a_w[:3, :3] = R_a
        T_ref_a_w[:3, 3] = p_a

        return np.linalg.inv(T_ref_a_w)

    def _build_ict_interpolated(self, d0: dict, d1: dict, T_w2ref_a: np.ndarray, alpha: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Linearly interpolates tokens between two frames for sub-step augmentation (legacy feature)."""
        state0, pcd0, mask0 = self._build_ict(d0, T_w2ref_a)
        state1, pcd1, mask1 = self._build_ict(d1, T_w2ref_a)

        state_interp = np.zeros_like(state0)
        pcd_interp = np.zeros_like(pcd0)
        mask_interp = mask0 & mask1  # Token must be valid in both frames

        for i in range(self.max_ict):
            if mask_interp[i]:
                state_interp[i] = state0[i] + (state1[i] - state0[i]) * alpha
                state_interp[i, 0] = state0[i, 0]  # Keep TypeID exact
                pcd_interp[i] = pcd0[i] + (pcd1[i] - pcd0[i]) * alpha

        return state_interp, pcd_interp, mask_interp

    # ---------------------
    # Aux Targets & Trajectory Building
    # ---------------------
    def _build_targets(self, json_path0: str, T_w2ref_base: np.ndarray, idx: int, stride: int, d0: dict) -> Dict[str, torch.Tensor]:
        K = self.pred_horizon

        # Dual-Hand dynamic dimension mapping
        hand_sides =[self.single_hand_side] if self.single_hand else ["left", "right"]
        num_hands = len(hand_sides)

        y_pos = np.zeros((K, num_hands, 3), dtype=np.float32)
        y_o6d = np.zeros((K, num_hands, 6), dtype=np.float32)
        y_hand_command = np.zeros(
            (K, num_hands, self.hand_command_dim), dtype=np.float32
        )
        y_hand_valid = np.zeros((K, num_hands), dtype=bool)

        y_obj_pos = np.zeros((K, 3), dtype=np.float32)
        y_obj_o6d = np.zeros((K, 6), dtype=np.float32)

        # --- 2D TRACE TARGET (Replaces Heatmap) ---
        num_trace_targets = num_hands + (1 if self.use_aux_obj_dynamics else 0)
        y_2d_trace = np.zeros((K, num_trace_targets, 2), dtype=np.float32)

        # Target Array for the 'Done' flag
        y_done = np.zeros((K, 1), dtype=np.float32)

        anchor_key = d0["metadata"].get("anchor_key", "obj1")
        c2w = np.array(d0["metadata"]["c2w"])
        K_cam = np.array(d0["metadata"]["k"]).reshape(3, 3)
        w2c = np.linalg.inv(c2w)

        # get resolution of orig image
        orig_w = int(d0["metadata"].get("w", 640))
        orig_h = int(d0["metadata"].get("h", 480))

        # Base poses for Delta Action
        T_h_ref_base = {side: None for side in hand_sides}
        T_obj_ref_base = None

        # Kinematic Latching Tracking
        latched_T_obj_in_hand = None

        for k in range(1, K + 1):
            fut_idx = k * stride
            dk = self._read_frame(self._get_future_json_path(json_path0, fut_idx))

            # Missing future frames remain invalid.  Never turn a held value
            # into apparent robot ground truth.
            if not dk:
                continue

            # Parse Done Flag from Metadata
            y_done[k-1, 0] = float(dk.get("metadata", {}).get("is_finished", 0.0))

            hands = dk.get("entities", {}).get(self.hand_entity_key, {})
            if hands is None: hands = {}  # null means method not generated
            objs = dk.get("entities", {}).get("objects", {})

            # ---------------------------------------------
            # Extract Hand Trajectories
            # ---------------------------------------------
            for h_idx, side in enumerate(hand_sides):
                future_path = self._get_future_json_path(json_path0, fut_idx)
                robot_state = (
                    self._get_robot_state(future_path, side)
                    if self.embodiment is not None else None
                )
                if self.embodiment is not None and robot_state is not None:
                    # Frozen sidecars store the embodiment wrist/base in the
                    # current camera frame.  Convert through that frame's c2w.
                    T_hk_w = (
                        np.asarray(dk["metadata"]["c2w"], dtype=np.float32)
                        @ robot_state["wrist_T_camera"]
                    )
                    y_hand_command[k-1, h_idx] = robot_state["q"]
                    y_hand_valid[k-1, h_idx] = True
                elif self.embodiment is None and side in hands:
                    T_hk_w = np.array(hands[side]["T_hand_to_world"], dtype=np.float32)
                    raw_g = float(hands[side]["grasp"])
                    y_hand_command[k-1, h_idx, 0] = (
                        1.0 if raw_g > 0.5 else 0.0
                    )
                    y_hand_valid[k-1, h_idx] = True
                else:
                    continue
                if y_hand_valid[k-1, h_idx]:
                    T_hk_ref = T_w2ref_base @ T_hk_w

                    if T_h_ref_base[side] is None:
                        T_h_ref_base[side] = T_hk_ref.copy()

                    if self.action_mode == 'delta':
                        T_delta = np.linalg.inv(T_h_ref_base[side]) @ T_hk_ref
                        y_pos[k-1, h_idx] = normalize_pos(T_delta[:3, 3], self.pos_mean, self.pos_std)
                        y_o6d[k-1, h_idx] = normalize_o6d(rotmat_to_o6d(T_delta[:3, :3]))
                    else:
                        y_pos[k-1, h_idx] = normalize_pos(T_hk_ref[:3, 3], self.pos_mean, self.pos_std)
                        y_o6d[k-1, h_idx] = normalize_o6d(rotmat_to_o6d(T_hk_ref[:3, :3]))

                    # Visual Foresight Projection (Hands)
                    if self.use_aux_visual_foresight:
                        y_2d_trace[k-1, h_idx] = self._project_to_2d_trace(T_hk_w[:3, 3], w2c, K_cam, orig_w, orig_h)
            # ---------------------------------------------
            # Extract Object Dynamics (with Kinematic Latching)
            # ---------------------------------------------
            if self.use_aux_obj_dynamics and anchor_key in objs:
                T_ok_w_raw = np.array(objs[anchor_key]["T_obj_to_world"], dtype=np.float32)

                # --- KINEMATIC LATCHING LOGIC ---
                # Check if dominant hand is grasping the object
                dominant_hand = self.single_hand_side if self.single_hand else "right"
                is_grasped = (dominant_hand in hands) and (float(hands[dominant_hand]["grasp"]) > 0.5)

                if is_grasped and not self.disable_kinematic_latching:
                    T_hk_w = np.array(hands[dominant_hand]["T_hand_to_world"])
                    if latched_T_obj_in_hand is None:
                        # Compute relative transform at the exact moment of grasp
                        latched_T_obj_in_hand = np.linalg.inv(T_hk_w) @ T_ok_w_raw
                    # Override visual tracking with Forward Kinematics
                    T_ok_w = T_hk_w @ latched_T_obj_in_hand
                else:
                    # Unlatch
                    latched_T_obj_in_hand = None
                    T_ok_w = T_ok_w_raw

                T_ok_ref = T_w2ref_base @ T_ok_w

                if T_obj_ref_base is None:
                    T_obj_ref_base = T_ok_ref.copy()

                if self.action_mode == 'delta':
                    T_obj_delta = np.linalg.inv(T_obj_ref_base) @ T_ok_ref
                    y_obj_pos[k-1] = normalize_pos(T_obj_delta[:3, 3], self.pos_mean, self.pos_std)
                    y_obj_o6d[k-1] = normalize_o6d(rotmat_to_o6d(T_obj_delta[:3, :3]))
                else:
                    y_obj_pos[k-1] = normalize_pos(T_ok_ref[:3, 3], self.pos_mean, self.pos_std)
                    y_obj_o6d[k-1] = normalize_o6d(rotmat_to_o6d(T_ok_ref[:3, :3]))

                # Visual Foresight Projection (Object is the last channel)
                if self.use_aux_visual_foresight:
                    y_2d_trace[k-1, -1] = self._project_to_2d_trace(T_ok_w[:3, 3], w2c, K_cam, orig_w, orig_h)
        # ---------------------------------------------
        # Jittering & Flattening
        # ---------------------------------------------
        if self.enable_augmentation and self.enable_aug_target_jittering:
            rng = np.random.RandomState(self.seed + idx * 123) if self.use_legacy_rng else np.random.RandomState()
            y_pos += ((rng.randn(*y_pos.shape).astype(np.float32) * AUG_TARGET_POS_STD) / self.pos_std) * y_hand_valid[..., None]
            y_o6d += (rng.randn(*y_o6d.shape).astype(np.float32) * (AUG_TARGET_ROT_STD / 180.0)) * y_hand_valid[..., None]

        # Modality-major layout:
        # [all wrist positions | all wrist rotations | all hand commands].
        # Commands are binary grasp, Kai q22, or Wuji q20.
        y_action_flat = np.concatenate([
            y_pos.reshape(K, -1),
            y_o6d.reshape(K, -1),
            y_hand_command.reshape(K, -1),
        ], axis=-1)
        y_action_tensor = torch.from_numpy(y_action_flat).float()
        action_valid = np.concatenate(
            (
                np.repeat(y_hand_valid, 3, axis=1),
                np.repeat(y_hand_valid, 6, axis=1),
                np.repeat(y_hand_valid, self.hand_command_dim, axis=1),
            ),
            axis=1,
        )

        # Object Trajectory (B, K, 9)
        y_obj_action_tensor = torch.cat([torch.from_numpy(y_obj_pos), torch.from_numpy(y_obj_o6d)], dim=-1).float()

        return {
            "y_action": y_action_tensor,
            "y_hand_command": torch.from_numpy(y_hand_command).float(),
            "y_obj_action": y_obj_action_tensor,
            "y_2d_trace": torch.from_numpy(y_2d_trace).float(),
            "y_done": torch.from_numpy(y_done).float()
            ,"y_hand_valid": torch.from_numpy(y_hand_valid).bool()
            ,"action_valid_mask": torch.from_numpy(action_valid).bool()
        }


    def _project_to_2d_trace(self, pt_w: np.ndarray, T_w2c: np.ndarray, K_cam: np.ndarray, orig_W: float, orig_H: float) -> Tuple[float, float]:
        """ PROJECTS 3D POINT TO 2D NORMALIZED UV[0, 1] """
        pt_c = (T_w2c[:3, :3] @ pt_w.reshape(3) + T_w2c[:3, 3])
        if pt_c[2] < 1e-4: return 0.5, 0.5
        u = K_cam[0, 0] * pt_c[0] / pt_c[2] + K_cam[0, 2]
        v = K_cam[1, 1] * pt_c[1] / pt_c[2] + K_cam[1, 2]
        return float(np.clip(u / orig_W, 0.0, 1.0)), float(np.clip(v / orig_H, 0.0, 1.0))


    # ---------------------
    # Images Loader
    # ---------------------
    def _resolve_image_path(self, d0: dict) -> str:
        frame_dir = os.path.dirname(d0["obs"]["rgb_path"])
        if isinstance(self.img_name, str) and self.img_name.startswith("@"):
            raise ValueError(
                "Legacy virtual image selectors were removed during repository cleanup. "
                "A formal manifest-backed image selector must be implemented under "
                "pipeline_contract_v1 before RobotRGB training is allowed."
            )
        return os.path.join(frame_dir, self.img_name)

    def _load_image_tensor(self, d0: dict, idx: int) -> torch.Tensor:
        if not self.img_name:
            return torch.zeros((3, self.H, self.W), dtype=torch.float32)

        # --- Path Resolution ---
        if self.use_legacy_image_loading:
            # Legacy: read absolute path from JSON, load at original resolution
            img_path = d0.get("obs", {}).get("rgb_WoArm_WArmObjKpts_path", "")
            md = d0.get("metadata", {})
            h0 = int(md.get("h", 640) or 640)
            w0 = int(md.get("w", 640) or 640)
            encoded = self._image_bytes_cache.get(img_path)
            if encoded is not None:
                bgr = cv2.imdecode(
                    np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR
                )
                if bgr is None:
                    bgr = np.zeros((h0, w0, 3), dtype=np.uint8)
                elif bgr.shape[:2] != (h0, w0):
                    bgr = cv2.resize(
                        bgr, (w0, h0), interpolation=cv2.INTER_LINEAR
                    )
            else:
                bgr = safe_imread_rgb(img_path, h0, w0)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        else:
            # New: construct path from frame_dir + img_name, load at target resolution
            img_path = self._resolve_image_path(d0)
            bgr = self._image_array_cache.get(img_path)
            encoded = self._image_bytes_cache.get(img_path)
            if bgr is not None:
                pass
            elif encoded is not None:
                bgr = cv2.imdecode(
                    np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR
                )
                if bgr is None:
                    bgr = np.zeros((self.H, self.W, 3), dtype=np.uint8)
                elif bgr.shape[:2] != (self.H, self.W):
                    bgr = cv2.resize(
                        bgr, (self.W, self.H), interpolation=cv2.INTER_LINEAR
                    )
            else:
                bgr = safe_imread_rgb(img_path, self.H, self.W)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # --- RNG Strategy ---
        def _make_rng(salt: int) -> np.random.RandomState:
            if self.use_legacy_rng:
                return np.random.RandomState(self.seed + idx * salt)
            return np.random.RandomState()

        # --- Random Resized Crop (at current resolution, before final resize) ---
        if self.enable_augmentation and self.enable_aug_rrc:
            rng_rrc = _make_rng(7)
            if rng_rrc.rand() < AUG_RRC_PROB:
                y, x, h, w = get_rrc_params(rgb.shape[0], rgb.shape[1], AUG_RRC_SCALE, AUG_RRC_RATIO, rng_rrc)
                rgb = rgb[y:y+h, x:x+w]

        # --- Final Resize ---
        rgb = cv2.resize(rgb, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        rgb_f = rgb.astype(np.float32) / 255.0

        # --- Cutout ---
        if self.enable_augmentation and self.enable_aug_cutout:
            rng_cutout = _make_rng(888)
            if rng_cutout.rand() < AUG_CUTOUT_PROB:
                rgb_f = apply_random_erasing(rgb_f, rng_cutout, AUG_CUTOUT_PROB, AUG_CUTOUT_N_HOLES, AUG_CUTOUT_SIZE)

        # --- Photometric ---
        if self.enable_augmentation and self.enable_aug_img:
            rng_photo = _make_rng(9973)
            rgb_f = apply_photometric_aug(rgb_f, rng_photo, AUG_IMG_PROB, AUG_BRIGHTNESS_DELTA, AUG_CONTRAST_DELTA, AUG_GAMMA_DELTA, AUG_NOISE_STD, AUG_BLUR_PROB, AUG_BLUR_KSIZE, AUG_GRAY_PROB, AUG_HUE_DELTA, AUG_SAT_RANGE)

        return torch.from_numpy(np.transpose(rgb_f, (2, 0, 1))).float()

    # ---------------------
    # __getitem__
    # ---------------------
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        json_path0 = self.samples[idx]
        d0 = self._read_frame(json_path0)

        # 1. Temporal Speed Perturbation: Safe Dynamic Stride Sample
        stride = 1
        if self.enable_augmentation and self.enable_aug_temporal_stride:
            rng = np.random.RandomState()
            max_s = AUG_STRIDE_RANGE[1]

            while max_s > 1:
                test_path = self._get_future_json_path(json_path0, self.pred_horizon * max_s)
                if self._frame_exists(test_path):
                    break
                max_s -= 1

            if max_s >= AUG_STRIDE_RANGE[0]:
                stride = rng.randint(AUG_STRIDE_RANGE[0], max_s + 1)

        # 2. Sub-step Interpolation (Legacy Feature)
        alpha = 0.0
        json_path1 = self._get_future_json_path(json_path0, 1)
        exists_next = self._frame_exists(json_path1)

        if self.enable_augmentation and self.enable_aug_interpolation and exists_next:
            if np.random.rand() < AUG_INTERP_PROB:
                alpha = np.random.rand()

        d1 = self._read_frame(json_path1) if alpha > 0 else d0
        d_vis = d1 if alpha > 0.5 else d0  # Use visually closer frame for image

        # 3. Load Visual Representation
        x_rgb = self._load_image_tensor(d_vis, idx)

        # 4. Build Base Reference Frame & Tokens
        if alpha > 0 and d1 is not None:
            T_w2ref_base = self._interpolate_T_w2ref(d0, d1, alpha)
            x_ict_np, x_pcd_np, ict_mask_np = self._build_ict_interpolated(d0, d1, T_w2ref_base, alpha)
        else:
            T_w2ref_base = self._get_T_w2ref(d0)
            x_ict_np, x_pcd_np, ict_mask_np = self._build_ict(d0, T_w2ref_base)

        anchor_uv = np.array([0.5, 0.5], dtype=np.float32)
        anchor_key = d0["metadata"].get("anchor_key", "obj1")
        if anchor_key in d0.get("entities", {}).get("objects", {}):
            T_anc_w = np.array(d0["entities"]["objects"][anchor_key]["T_obj_to_world"], dtype=np.float32)
            c2w = np.array(d0["metadata"]["c2w"])
            w2c = np.linalg.inv(c2w)
            K_cam = np.array(d0["metadata"]["k"]).reshape(3, 3)
            orig_w, orig_h = K_cam[0, 2] * 2.0, K_cam[1, 2] * 2.0
            u_norm, v_norm = self._project_to_2d_trace(T_anc_w[:3, 3], w2c, K_cam, orig_w, orig_h)
            anchor_uv = np.array([u_norm, v_norm], dtype=np.float32)

        # 5. Build Action Trajectories
        y_dict = self._build_targets(json_path0, T_w2ref_base, idx, stride, d0)

        x_robot_state = None
        robot_state_mask = None
        if self.embodiment is not None:
            state_sides = (
                [self.single_hand_side] if self.single_hand else ["left", "right"]
            )
            state_dim = 9 + self.hand_command_dim + 2
            x_robot_state = np.zeros((len(state_sides), state_dim), dtype=np.float32)
            robot_state_mask = np.zeros((len(state_sides),), dtype=bool)
            c2w0 = np.asarray(d0["metadata"]["c2w"], dtype=np.float32)
            for side_index, side in enumerate(state_sides):
                state = self._get_robot_state(json_path0, side)
                if state is None:
                    continue
                T_robot_ref = (
                    T_w2ref_base @ c2w0 @ state["wrist_T_camera"]
                )
                wrist9 = np.concatenate(
                    (
                        normalize_pos(T_robot_ref[:3, 3], self.pos_mean, self.pos_std),
                        normalize_o6d(rotmat_to_o6d(T_robot_ref[:3, :3])),
                    )
                )
                x_robot_state[side_index] = np.concatenate(
                    (wrist9, state["q"], [state["confidence"], 1.0])
                )
                robot_state_mask[side_index] = True

        out = {
            "x_rgb": x_rgb,
            "x_ict": torch.from_numpy(x_ict_np).float(),
            "x_pcd": torch.from_numpy(x_pcd_np).float() if self.use_pcd_features else torch.zeros(1),
            "ict_mask": torch.from_numpy(ict_mask_np).bool(),

            "meta_t": torch.tensor(int(d0.get("metadata", {}).get("idx", 0)), dtype=torch.int64),
            "temporal_stride": torch.tensor(stride, dtype=torch.int64),
            "json_path": json_path0,

            "anchor_uv": torch.from_numpy(anchor_uv).float(),
        }
        if x_robot_state is not None:
            out["x_robot_state"] = torch.from_numpy(x_robot_state).float()
            out["robot_state_mask"] = torch.from_numpy(robot_state_mask).bool()
            source = self._robot_sources[
                self._session_root_from_json_path(json_path0)
            ]
            out["joint_lower"] = torch.from_numpy(np.stack([
                source[f"{side}_joint_lower"] for side in state_sides
            ])).float()
            out["joint_upper"] = torch.from_numpy(np.stack([
                source[f"{side}_joint_upper"] for side in state_sides
            ])).float()

        # 6. Temporal Contrastive Target Extraction (t + K)
        if self.use_aux_temporal_contrastive:
            future_idx_offset = self.pred_horizon * stride
            future_path = self._get_future_json_path(json_path0, future_idx_offset)
            d_fut = self._read_frame(future_path)

            if not d_fut:
                d_fut = d0

            x_ict_fut_np, _, ict_mask_fut_np = self._build_ict(d_fut, T_w2ref_base)
            out["x_ict_future"] = torch.from_numpy(x_ict_fut_np).float()
            out["ict_mask_future"] = torch.from_numpy(ict_mask_fut_np).bool()

        out.update(y_dict)
        return out


# =========================================================
# --------- Main for Testing ----------
# =========================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mps_path", type=str, required=True)
    parser.add_argument("--dual_hand", action="store_true")
    args = parser.parse_args()

    ds = FlowMatchingDataloader(
        sessions=[MPSSessions(mps_path=args.mps_path)],
        single_hand=(not args.dual_hand)
    )

    if len(ds) > 0:
        sample = ds[0]
        print(f"\n[FlowMatchingDataloader Check]")
        print(f"Total samples: {len(ds)}")
        print(f"x_rgb shape: {sample['x_rgb'].shape}")
        print(f"x_ict shape: {sample['x_ict'].shape}  Valid Tokens: {sample['ict_mask'].sum().item()}")
        print(f"x_pcd shape: {sample['x_pcd'].shape}")

        if "x_ict_future" in sample:
            print(f"x_ict_future shape: {sample['x_ict_future'].shape}")

        print(f"y_action shape: {sample['y_action'].shape} (10D per hand * num_hands)")
        print(f"y_obj_action shape: {sample['y_obj_action'].shape}")
        print(f"y_2d_trace shape: {sample['y_2d_trace'].shape}")
        print(f"Temporal Stride Sampled: {sample['temporal_stride'].item()}")

        print("\n--- First Valid Token (Features) ---")
        idx = torch.where(sample['ict_mask'])[0][0]
        tok = sample['x_ict'][idx]
        print(f"Type={tok[0]:.0f} | Pose_in_Ref={tok[1:4].numpy().round(3)} | Flag={tok[-1]:.1f}")

# python -m training.FlowMatchingDataloader --mps_path  "./data/serve_bread/serve_bread_0/mps_serve_bread_0_029_vrs/"
