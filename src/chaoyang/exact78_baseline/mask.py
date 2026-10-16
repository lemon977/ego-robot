"""Removal masks (wearer hands + forearms + PICO wrist trackers) and held-object masks.

SAM3.1 multiplex video tracking with a text prompt proposes instances on every
frame; PICO keypoints decide which tracks belong to the wearer. Where SAM misses
a tracked hand, a PICO geometric hull is used instead and the frame is labelled;
if a 2D shift puts the PICO hand onto SAM there, the tracker band and wrist
closure are also placed at the shifted position. The wrist tracker (strap + dorsal
puck) is always covered by the hull of a PICO-projected band volume, unioned over
adjacent frames to absorb tracking lag.
The removal mask additionally closes the SAM wrist region, grows into
tracker-coloured pixels next to the band and closes notches; tracked task
objects outside the wearer core are never removed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np


HUMAN_PROMPT = "a person's hand and forearm"
OBJECT_PROMPTS = {"playing_cards": "playing card", "potato_chips": "potato chip"}
ADAPTER = Path("src/chaoyang/pipeline/sam31_compat_adapter_v1.py")
SAM_CODE_ROOT = Path("vendor/SAM3")
SAM_CHECKPOINT = Path("assets/models/checkpoints/sam3.1/sam3.1_multiplex.pt")
PROPAGATION_ABORTED = "No points are provided"
TRACK_HIT_RATE_MIN = 0.25
HIT_RADIUS_PX = 8
FALLBACK_COVERAGE_MIN = 0.5
# The PICO hand can sit 5-20 cm (up to ~220 px) off the image hand for whole segments (seen at the lower-left
# periphery); the band hull then misses the puck, and a puck exposed in even 3 frames is smeared by ProPainter over
# the whole static wrist segment. In fallback frames the band and closure are repeated at a near-best keypoint
# coverage shift (>= REGISTER_COVERAGE_MIN) that follows the side's recent shift (the offset grows over a segment;
# a free search over 256 px snapped onto the forearm); fallback frames that do not register take the nearest
# registered shift of that side within REGISTER_CARRY_FRAMES.
REGISTER_COVERAGE_MIN = 0.8
REGISTER_RADIUS_PX, REGISTER_STEP_PX = 256, 4
REGISTER_CARRY_FRAMES = 15
REGISTER_TIE = 0.05
REGISTER_LOCAL_PX = 24
REGISTER_JUMP_MAX_PX = 64
REGISTER_DEPTH_CAP_PX = 12
# PICO wrist tracker: strap ring around the wrist plus a puck on the dorsal side.
BAND_BEHIND_WRIST_M, BAND_HALF_WIDTH_M, BAND_RADIUS_M = 0.030, 0.030, 0.050
BAND_PUCK_M, BAND_PUCK_RADIUS_M, BAND_RING_POINTS = 0.070, 0.035, 12
BAND_PAD_PX = 10
BAND_TEMPORAL_FRAMES = 4
# PICO wrist positions can drift ~6 cm for several frames while the hand is still; the SAM wearer mask near
# the wrist is closed under its convex hull so the strap gap between the hand and forearm parts is removed too.
WRIST_ROI_M, WRIST_ROI_MIN_PX = 0.11, 200
# A few tracker pixels left on the mask edge are enough for ProPainter's flow propagation to smear them across
# the whole hole, so the edge near the band is grown into tracker-coloured pixels (any neutral grey: white shell
# and dark strap; green rim) and notches between the convex pieces are closed. Neutral table / wall pixels next to
# the wrist are taken too, which ProPainter fills without visible cost. Neutral is judged by RGB chroma, not HSV
# saturation, which is inflated on the dark, slightly blue strap (RGB ~(61, 66, 74) has S = 45).
BAND_COLOUR_REACH_PX, BAND_COLOUR_STEP_PX, BAND_COLOUR_STEPS = 36, 12, 2
BAND_COLOUR_MARGIN_PX = 3
BAND_NEUTRAL_CHROMA = 24
REMOVAL_CLOSE_PX = 10
FOREARM_RADIUS_M = 0.040
MIN_DEPTH_M = 0.05
HAND_HULL_PAD_PX = 10
HUMAN_DILATE_PX = 6
CONTACT_PX = 12
REMOVAL_BRACKET_FRAMES = 8
OBJECT_AREA_MAX = 0.25
OBJECT_TOP2_FRACTION_MIN = 0.6
# The text prompt is placed on the frame with the most forearm in view (up to the wrist and the first
# forearm sample per side), then the most hand keypoints: SAM3.1 tracks the instance it found there, so
# an anchor showing only the hands (chips sessions start with both hands at the bottom edge) keeps the
# forearm out of the mask for the whole video (0901_008 / 0902_120 anchored at frames 13 / 9).
ANCHOR_FOREARM_SAMPLES = 2
FOREARM_PROBE_SAMPLES = (2, 3)
# The shared anchor can still show a side's forearm only at the frame edge (0902_028 chips anchored on frame 0),
# and SAM then keeps that side hand-only. When a side's forearm probes mostly miss SAM, the prompt runs once
# more on the frame showing most of that side's forearm; the new tracks pass the same PICO filter and are
# added to the first pass, so the wearer mask only grows. Providers and registration still come from the
# first pass only: a retry track can drift onto the spot where an offset PICO hand sits (0902_028 cards frames
# 126-143, PICO ~200 px off), which turned those fallback frames into SAM frames and dropped the shifted band
# and closure, exposing the strap.
FOREARM_RETRY_RATE = 0.6
FOREARM_RETRY_PROBES_MIN = 8


def _disk(radius: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    return cv2.dilate(mask.astype(np.uint8), _disk(radius)) > 0 if radius > 0 else mask.copy()


class Sam3Video:
    def __init__(self, repo_root: Path):
        import torch  # noqa: F401  (CUDA must be importable before the adapter)

        spec = importlib.util.spec_from_file_location("takeover_sam31_adapter", repo_root / ADAPTER)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        code_root = repo_root / SAM_CODE_ROOT
        if str(code_root) not in sys.path:
            sys.path.insert(0, str(code_root))
        self.adapter, self.evidence = module.build_pinned_adapter(
            official_code_root=code_root, checkpoint_path=repo_root / SAM_CHECKPOINT)
        self.model = self.adapter.model

    def close(self) -> None:
        self.adapter.predictor.shutdown()

    @staticmethod
    def _normalize(outputs: dict[str, Any], height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
        import torch

        masks, ids = outputs.get("out_binary_masks"), outputs.get("out_obj_ids")
        if isinstance(masks, torch.Tensor):
            masks = masks.detach().cpu().numpy()
        elif isinstance(masks, (list, tuple)):
            masks = np.stack([m.detach().cpu().numpy() if isinstance(m, torch.Tensor) else np.asarray(m)
                              for m in masks]) if masks else None
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().numpy()
        masks = np.zeros((0, height, width), bool) if masks is None else np.asarray(masks)
        while masks.ndim > 3 and masks.shape[1] == 1:
            masks = masks[:, 0]
        if masks.ndim == 2:
            masks = masks[None]
        if len(masks) and masks.shape[1:] != (height, width):
            raise RuntimeError(f"SAM_MASK_GEOMETRY:{masks.shape}")
        return masks.astype(bool), np.asarray(ids if ids is not None else [], np.int64).reshape(-1)

    def track_text(self, frames_rgb: list[np.ndarray], prompt: str, anchor: int) -> dict[int, dict[int, np.ndarray]]:
        """obj_id -> frame -> packed bool mask (np.packbits of HxW).

        `self.aborted` lists the directions SAM3.1 could not propagate: its tracker raises
        "No points are provided" when the text prompt found no instance at the anchor frame.
        """
        from PIL import Image

        height, width = frames_rgb[0].shape[:2]
        images = [Image.fromarray(frame) for frame in frames_rgb]
        state = self.model.init_state(resource_path=images, offload_video_to_cpu=True,
                                      async_loading_frames=False)
        tracks: dict[int, dict[int, np.ndarray]] = {}
        self.aborted: list[str] = []
        try:
            _, initial = self.model.add_prompt(inference_state=state, frame_idx=anchor, text_str=prompt,
                                               output_prob_thresh=0.5)
            self._store(tracks, anchor, initial, height, width)
            count = len(images)
            directions = [(False, count - anchor)] + ([(True, anchor + 1)] if anchor else [])
            for reverse, maximum in directions:
                try:
                    for frame, outputs in self.model.propagate_in_video(
                            inference_state=state, start_frame_idx=anchor, max_frame_num_to_track=maximum,
                            reverse=reverse, output_prob_thresh=0.5):
                        self._store(tracks, int(frame), outputs, height, width)
                except RuntimeError as error:
                    if PROPAGATION_ABORTED not in str(error):
                        raise
                    self.aborted.append("reverse" if reverse else "forward")
        finally:
            state.clear()
        return tracks

    def _store(self, tracks, frame, outputs, height, width) -> None:
        masks, ids = self._normalize(outputs, height, width)
        for mask, obj in zip(masks, ids):
            if mask.any():
                tracks.setdefault(int(obj), {})[frame] = np.packbits(mask)


def unpack(packed: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return np.unpackbits(packed, count=shape[0] * shape[1]).reshape(shape).astype(bool)


# ---------------------------------------------------------------- PICO geometry (working resolution)

def tracker_band_points(hand: np.ndarray, elbow: np.ndarray, side: int) -> np.ndarray:
    """World points bounding the wrist tracker: two strap rings and the dorsal puck."""
    wrist = hand[0]
    axis = elbow - wrist
    axis = axis / max(np.linalg.norm(axis), 1e-6)
    dorsal = np.cross(hand[5] - wrist, hand[17] - wrist) * (-1.0 if side == 1 else 1.0)
    dorsal = dorsal - dorsal.dot(axis) * axis
    dorsal = dorsal / max(np.linalg.norm(dorsal), 1e-6)
    lateral = np.cross(axis, dorsal)
    centre = wrist + BAND_BEHIND_WRIST_M * axis
    angles = np.linspace(0.0, 2 * np.pi, BAND_RING_POINTS, endpoint=False)
    ring = BAND_RADIUS_M * (np.cos(angles)[:, None] * dorsal + np.sin(angles)[:, None] * lateral)
    rings = [centre + offset * axis + ring for offset in (-BAND_HALF_WIDTH_M, BAND_HALF_WIDTH_M)]
    puck_centre = centre + BAND_PUCK_M * dorsal
    puck = puck_centre + BAND_PUCK_RADIUS_M * (np.cos(angles)[:, None] * axis + np.sin(angles)[:, None] * lateral)
    return np.concatenate(rings + [puck])


def pico_geometry(session: Any, scale: float) -> dict[str, np.ndarray]:
    """Projected hand keypoints, forearm samples and wrist-tracker band points at working resolution."""
    count = session.frame_count
    width, height = session.width * scale, session.height * scale
    hands = np.full((2, count, 21, 2), np.nan)
    forearm = np.full((2, count, 6, 2), np.nan)
    forearm_radius = np.full((2, count), np.nan)
    tracker = np.full((2, count, 3 * BAND_RING_POINTS, 2), np.nan)   # projected band volume
    wrist_px_per_m = np.full((2, count), np.nan)
    elbow_index = (18, 19)
    fx = session.K[0, 0] * scale
    for frame in range(count):
        for side in range(2):
            camera = session.camera_points(session.hands[side, frame], frame)
            if session.hand_source[side, frame] != "NONE" and (camera[:, 2] > MIN_DEPTH_M).all():
                hands[side, frame] = session.project(camera) * scale
                wrist_px_per_m[side, frame] = fx / camera[0, 2]
            wrist = session.hands[side, frame, 0]
            middle = session.hands[side, frame, 9]
            if session.body_observed[frame]:
                elbow = session.body[frame, elbow_index[side]]
            else:
                elbow = wrist + 0.25 * (wrist - middle) / max(np.linalg.norm(wrist - middle), 1e-6)
            if session.hand_source[side, frame] == "NONE":
                continue
            # The elbow is often beside or behind the camera; keep every sample that is in front of it.
            samples = wrist[None] + np.linspace(0.0, 1.0, 6)[:, None] * (elbow - wrist)[None]
            camera = session.camera_points(samples, frame)
            front = camera[:, 2] > MIN_DEPTH_M
            if front.any():
                forearm[side, frame, front] = session.project(camera[front]) * scale
                forearm_radius[side, frame] = fx * FOREARM_RADIUS_M / float(np.median(camera[front, 2]))
            band_camera = session.camera_points(tracker_band_points(session.hands[side, frame], elbow, side), frame)
            front = band_camera[:, 2] > MIN_DEPTH_M
            if front.sum() >= 3:
                tracker[side, frame, front] = session.project(band_camera[front]) * scale
    with np.errstate(invalid="ignore"):
        rounded = np.round(hands)
        inside = ((rounded[..., 0] >= 0) & (rounded[..., 0] <= round(width) - 1)
                  & (rounded[..., 1] >= 0) & (rounded[..., 1] <= round(height) - 1))
    return {"hands": hands, "forearm": forearm, "forearm_radius": forearm_radius, "tracker": tracker,
            "wrist_px_per_m": wrist_px_per_m, "hands_inside": inside & np.isfinite(hands).all(-1)}


def _inside(points: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    rounded = np.round(points)
    with np.errstate(invalid="ignore"):
        return (np.isfinite(points).all(-1) & (rounded[..., 0] >= 0) & (rounded[..., 0] < shape[1])
                & (rounded[..., 1] >= 0) & (rounded[..., 1] < shape[0]))


def anchor_frame(geometry: dict[str, np.ndarray], shape: tuple[int, int]) -> int:
    forearm = np.minimum(_inside(geometry["forearm"], shape).sum(-1), ANCHOR_FOREARM_SAMPLES).sum(0)
    hands = geometry["hands_inside"].sum(axis=(0, 2))
    return int(np.argmax(forearm * (hands.max() + 1) + hands))


def side_forearm_anchor(geometry: dict[str, np.ndarray], shape: tuple[int, int], side: int) -> int:
    forearm = _inside(geometry["forearm"][side], shape).sum(-1)
    hands = geometry["hands_inside"][side].sum(-1)
    return int(np.argmax(forearm * (hands.max() + 1) + hands))


def _hull_mask(points: np.ndarray, shape: tuple[int, int], pad: int) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    valid = points[np.isfinite(points).all(-1)]
    if len(valid) >= 3:
        valid = np.clip(valid, -4 * max(shape), 4 * max(shape))
        cv2.fillConvexPoly(mask, cv2.convexHull(np.round(valid).astype(np.int32)), 1)
    return dilate(mask > 0, pad)


def _capsule_mask(points: np.ndarray, radius: float, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    if not np.isfinite(radius):
        return mask > 0
    limit = 4 * max(shape)
    thickness = max(2, int(round(2 * radius)))
    for a, b in zip(points[:-1], points[1:]):
        if np.isfinite(a).all() and np.isfinite(b).all():
            a, b = (tuple(np.round(np.clip(p, -limit, limit)).astype(int)) for p in (a, b))
            cv2.line(mask, a, b, 1, thickness, cv2.LINE_8)
    return mask > 0


def _wrist_closure(sam: np.ndarray, wrist: np.ndarray, px_per_m: float, shape: tuple[int, int]) -> np.ndarray:
    """Convex hull of the SAM wearer mask inside a disk around the PICO wrist."""
    if not (np.isfinite(wrist).all() and np.isfinite(px_per_m)):
        return np.zeros(shape, bool)
    roi = np.zeros(shape, np.uint8)
    centre = tuple(np.round(np.clip(wrist, -4 * max(shape), 4 * max(shape))).astype(int))
    cv2.circle(roi, centre, int(round(WRIST_ROI_M * px_per_m)), 1, -1)
    ys, xs = np.nonzero(sam & (roi > 0))
    if len(xs) < WRIST_ROI_MIN_PX:
        return np.zeros(shape, bool)
    return _hull_mask(np.stack([xs, ys], -1).astype(float), shape, 0)


def tracker_coloured(frame_rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    chroma = frame_rgb.max(axis=2).astype(np.int16) - frame_rgb.min(axis=2)
    return (chroma < BAND_NEUTRAL_CHROMA) | ((hue >= 25) & (hue <= 55) & (sat > 90) & (val > 90))


def _grow_into_tracker(removal: np.ndarray, band: np.ndarray, frame_rgb: np.ndarray) -> np.ndarray:
    reach = dilate(band, BAND_COLOUR_REACH_PX)
    if not reach.any():
        return removal
    colour = tracker_coloured(frame_rgb) & reach
    for _ in range(BAND_COLOUR_STEPS):
        added = dilate(removal, BAND_COLOUR_STEP_PX) & ~removal & colour
        if not added.any():
            break
        removal = removal | dilate(added, BAND_COLOUR_MARGIN_PX)
    return removal


def _coverage(points: np.ndarray, near: np.ndarray) -> np.ndarray:
    """Fraction of points on `near`; points is (..., P, 2) pixel coordinates."""
    ij = np.round(points).astype(int)
    height, width = near.shape
    ok = (ij[..., 0] >= 0) & (ij[..., 0] < width) & (ij[..., 1] >= 0) & (ij[..., 1] < height)
    hit = near[np.clip(ij[..., 1], 0, height - 1), np.clip(ij[..., 0], 0, width - 1)] & ok
    return hit.mean(-1)


def register_to_sam(points: np.ndarray, near: np.ndarray, depth: np.ndarray,
                    prior: np.ndarray | None = None) -> np.ndarray | None:
    """2D shift that puts projected hand keypoints on the SAM wearer mask, or None.

    Among near-best shifts, the ones close to `prior` (the side's recent shift; zero if none) are kept and the
    deepest inside the mask is taken: the thick forearm scores as well as the hand but lies further away.
    """
    offsets = np.arange(-REGISTER_RADIUS_PX, REGISTER_RADIUS_PX + 1, REGISTER_STEP_PX)
    grid = np.stack(np.meshgrid(offsets, offsets), -1).reshape(-1, 2).astype(float)
    moved = points[None] + grid[:, None]
    coverage = _coverage(moved, near)
    best = coverage.max()
    if best < REGISTER_COVERAGE_MIN:
        return None
    candidates = np.flatnonzero(coverage >= best - REGISTER_TIE)
    anchor = np.zeros(2) if prior is None else prior
    distance = np.linalg.norm(grid[candidates] - anchor, axis=1)
    candidates = candidates[distance <= distance.min() + REGISTER_LOCAL_PX]
    ij = np.round(moved[candidates]).astype(int)
    height, width = depth.shape
    inside = np.minimum(depth[np.clip(ij[..., 1], 0, height - 1), np.clip(ij[..., 0], 0, width - 1)],
                        REGISTER_DEPTH_CAP_PX).mean(-1)
    choice = grid[candidates[np.argmax(inside)]]
    if prior is not None and np.linalg.norm(choice - prior) > REGISTER_JUMP_MAX_PX:
        return None
    return choice


def _wearer_sam(tracks: dict[int, dict[int, np.ndarray]], kept: list[int], frame: int,
                shape: tuple[int, int]) -> np.ndarray:
    sam = np.zeros(shape, bool)
    for obj in kept:
        if frame in tracks[obj]:
            sam |= unpack(tracks[obj][frame], shape)
    return sam


def _providers(tracks: dict[int, dict[int, np.ndarray]], kept: list[int], geometry: dict[str, np.ndarray],
               count: int, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame, per-side mask provider code and PICO->SAM shift (px); see provider_enum."""
    provider = np.zeros((count, 2), np.uint8)
    shifts = np.zeros((count, 2, 2))
    last = [None, None]     # (frame, shift) of the side's latest registration
    for frame in range(count):
        sam = _wearer_sam(tracks, kept, frame, shape)
        near = dilate(sam, HIT_RADIUS_PX)
        depth = None
        for side in range(2):
            inside = geometry["hands_inside"][side, frame]
            if inside.sum() < 5:
                provider[frame, side] = 2
                continue
            points = geometry["hands"][side, frame][inside]
            if float(_coverage(points, near)) >= FALLBACK_COVERAGE_MIN:
                continue
            provider[frame, side] = 1
            # Additive only (the PICO hull stays): a true SAM miss can register onto the forearm or the other hand.
            if sam.any():
                if depth is None:
                    depth = cv2.distanceTransform(sam.astype(np.uint8), cv2.DIST_L2, 3)
                prior = last[side][1] if last[side] and frame - last[side][0] <= REGISTER_CARRY_FRAMES else None
                registered = register_to_sam(points, near, depth, prior)
                if registered is not None and registered.any():
                    provider[frame, side], shifts[frame, side] = 3, registered
                    last[side] = (frame, registered)
    for side in range(2):
        registered = np.flatnonzero(provider[:, side] == 3)
        if not len(registered):
            continue
        for frame in np.flatnonzero(provider[:, side] == 1):
            nearest = registered[np.argmin(np.abs(registered - frame))]
            if abs(int(nearest) - int(frame)) <= REGISTER_CARRY_FRAMES:
                provider[frame, side], shifts[frame, side] = 4, shifts[nearest, side]
    return provider, shifts


def bracket_fill(removal: np.ndarray, frames: int) -> np.ndarray:
    """Also remove a pixel that is removed within `frames` frames both before and after.

    ProPainter copies a tracker pixel exposed for even a few frames over a whole static segment.
    """
    count = len(removal)
    far = np.int32(1 << 20)
    before = np.zeros(removal.shape, bool)
    last = np.full(removal.shape[1:], -far, np.int32)
    for frame in range(count):
        before[frame] = frame - last <= frames
        last[removal[frame]] = frame
    filled = removal.copy()
    following = np.full(removal.shape[1:], far, np.int32)
    for frame in range(count - 1, -1, -1):
        filled[frame] |= before[frame] & (following - frame <= frames)
        following[removal[frame]] = frame
    return filled


def build_masks(tracks: dict[int, dict[int, np.ndarray]], geometry: dict[str, np.ndarray],
                count: int, shape: tuple[int, int], frames_rgb: np.ndarray | None = None,
                provider_tracks: set[int] | None = None) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Wearer core mask, removal mask (both (count,H,W)) and per-frame provenance.

    Only kept tracks in `provider_tracks` (default: all) decide providers and registration shifts.
    """
    keypoints = np.concatenate((geometry["hands"], geometry["forearm"][:, :, :3]), axis=2)  # (2,N,24,2)
    kept, track_rows = [], []
    for obj, frames in tracks.items():
        hits = 0
        for frame, packed in frames.items():
            near = dilate(unpack(packed, shape), HIT_RADIUS_PX)
            points = keypoints[:, frame].reshape(-1, 2)
            points = points[np.isfinite(points).all(-1)]
            ij = np.round(points).astype(int)
            ok = (ij[:, 0] >= 0) & (ij[:, 0] < shape[1]) & (ij[:, 1] >= 0) & (ij[:, 1] < shape[0])
            hits += bool(near[ij[ok, 1], ij[ok, 0]].any())
        rate = hits / max(1, len(frames))
        track_rows.append({"obj_id": obj, "frames": len(frames), "pico_hit_rate": round(rate, 3),
                           "kept": rate >= TRACK_HIT_RATE_MIN})
        if rate >= TRACK_HIT_RATE_MIN:
            kept.append(obj)
    deciding = kept if provider_tracks is None else [obj for obj in kept if obj in provider_tracks]
    provider, shifts = _providers(tracks, deciding, geometry, count, shape)
    core = np.zeros((count, *shape), bool)
    removal = np.zeros((count, *shape), bool)
    probe = geometry["forearm"][:, :, list(FOREARM_PROBE_SAMPLES)]
    probe_inside = _inside(probe, shape)
    probe_hits = np.zeros(2, int)
    probe_total = np.zeros(2, int)
    for frame in range(count):
        sam = _wearer_sam(tracks, kept, frame, shape)
        near = dilate(sam, HIT_RADIUS_PX)
        for side in range(2):
            if provider[frame, side] == 0:
                ij = np.round(probe[side, frame][probe_inside[side, frame]]).astype(int)
                probe_hits[side] += int(near[ij[:, 1], ij[:, 0]].sum())
                probe_total[side] += len(ij)
        wearer = sam.copy()
        bands = np.zeros(shape, bool)
        closure = np.zeros(shape, bool)
        for side in range(2):
            offsets = [np.zeros(2)]
            if provider[frame, side] in (1, 3, 4):
                wearer |= _hull_mask(geometry["hands"][side, frame], shape, HAND_HULL_PAD_PX)
                wearer |= _capsule_mask(geometry["forearm"][side, frame],
                                        geometry["forearm_radius"][side, frame], shape)
            if provider[frame, side] in (3, 4):
                offsets.append(shifts[frame, side])
            band = np.zeros(shape, bool)
            for offset in offsets:
                for near_frame in range(max(0, frame - BAND_TEMPORAL_FRAMES),
                                        min(count, frame + BAND_TEMPORAL_FRAMES + 1)):
                    band |= _hull_mask(geometry["tracker"][side, near_frame] + offset, shape, 0)
                closure |= _wrist_closure(sam, geometry["hands"][side, frame, 0] + offset,
                                          geometry["wrist_px_per_m"][side, frame], shape)
            wearer |= dilate(band, BAND_PAD_PX)
            bands |= band
        # The closure and colour growth go into removal only: held objects are core-relative and are
        # subtracted from removal afterwards, so a hull spanning both hands cannot erase a held card.
        core[frame] = wearer
        grown = dilate(wearer | closure, HUMAN_DILATE_PX)
        if frames_rgb is not None:
            grown = _grow_into_tracker(grown, bands, frames_rgb[frame])
        removal[frame] = cv2.morphologyEx(grown.astype(np.uint8), cv2.MORPH_CLOSE, _disk(REMOVAL_CLOSE_PX)) > 0
    expanded = bracket_fill(removal, REMOVAL_BRACKET_FRAMES)
    # Share of PICO forearm samples (40% / 60% wrist->elbow) in view that SAM covers on SAM-provider frames;
    # a low rate flags a SAM instance that stops at the wrist.
    forearm_rate = {side: (round(float(probe_hits[i] / probe_total[i]), 3) if probe_total[i] else None)
                    for i, side in enumerate(("left", "right"))}
    return core, expanded, {"tracks": track_rows, "kept_obj_ids": kept, "provider_obj_ids": deciding,
                      "removal_bracket_frames": REMOVAL_BRACKET_FRAMES,
                      "sam_forearm_hit_rate": forearm_rate,
                      "sam_forearm_probe_points": {side: int(probe_total[i])
                                                   for i, side in enumerate(("left", "right"))},
                      "provider_enum": {"0": "SAM31_TEXT_TRACK_PICO_FILTERED", "1": "PICO_HULL_FALLBACK",
                                        "2": "HAND_NOT_IN_VIEW", "3": "PICO_HULL_FALLBACK_SAM_REGISTERED",
                                        "4": "PICO_HULL_FALLBACK_SHIFT_CARRIED"},
                      "provider": provider, "pico_shift_px": shifts}


def track_wearer(sam: Any, frames_rgb: np.ndarray, geometry: dict[str, np.ndarray], count: int,
                 shape: tuple[int, int], anchor: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Wearer prompt at `anchor`, re-run once per side whose forearm probes mostly miss SAM; see build_masks."""
    tracks = sam.track_text(list(frames_rgb), HUMAN_PROMPT, anchor)
    aborted = list(sam.aborted)
    first_pass = set(tracks)
    core, removal, info = build_masks(tracks, geometry, count, shape, frames_rgb=frames_rgb)
    retries: list[dict[str, Any]] = []
    for index, side in enumerate(("left", "right")):
        rate, points = info["sam_forearm_hit_rate"][side], info["sam_forearm_probe_points"][side]
        if rate is None or points < FOREARM_RETRY_PROBES_MIN or rate >= FOREARM_RETRY_RATE:
            continue
        retry_anchor = side_forearm_anchor(geometry, shape, index)
        if retry_anchor == anchor or any(row["anchor_frame"] == retry_anchor for row in retries):
            continue
        extra = sam.track_text(list(frames_rgb), HUMAN_PROMPT, retry_anchor)
        offset = max(tracks, default=-1) + 1
        tracks.update({offset + obj: frames for obj, frames in extra.items()})
        retries.append({"side": side, "anchor_frame": retry_anchor, "hit_rate_before": rate,
                        "tracks": len(extra), "propagation_aborted": list(sam.aborted)})
    if retries:
        core, removal, info = build_masks(tracks, geometry, count, shape, frames_rgb=frames_rgb,
                                          provider_tracks=first_pass)
    info["propagation_aborted"] = aborted
    info["forearm_retry"] = retries
    return core, removal, info


def _object_like(mask: np.ndarray) -> bool:
    """Small and compact: a SAM object track that degenerates into table speckle (0902_120 frames 261-266:
    455-609 components, the largest two holding 10-54% of the area) is ignored in that frame."""
    area = int(mask.sum())
    if not area or area >= OBJECT_AREA_MAX * mask.size:
        return False
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    largest = np.sort(stats[1:, cv2.CC_STAT_AREA])[::-1][:2].sum()
    return largest >= OBJECT_TOP2_FRACTION_MIN * area


def held_objects(tracks: dict[int, dict[int, np.ndarray]], removal_core: np.ndarray,
                 shape: tuple[int, int]) -> np.ndarray:
    """Object pixels belonging to instances touching the wearer's hands in that frame."""
    count = removal_core.shape[0]
    held = np.zeros((count, *shape), bool)
    for frame in range(count):
        contact = dilate(removal_core[frame], CONTACT_PX)
        for frames in tracks.values():
            if frame in frames:
                mask = unpack(frames[frame], shape)
                if (mask & contact).any() and _object_like(mask):
                    held[frame] |= mask
    return held & ~removal_core


def object_pixels(tracks: dict[int, dict[int, np.ndarray]], removal_core: np.ndarray,
                  shape: tuple[int, int]) -> np.ndarray:
    """Pixels of every tracked task object outside the wearer core; removal must not erase them."""
    count = removal_core.shape[0]
    kept = np.zeros((count, *shape), bool)
    for frame in range(count):
        for frames in tracks.values():
            if frame in frames:
                mask = unpack(frames[frame], shape)
                if _object_like(mask):
                    kept[frame] |= mask
    return kept & ~removal_core
