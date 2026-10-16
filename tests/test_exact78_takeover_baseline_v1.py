"""Exact78 takeover baseline v1: geometry, mask and IK invariants (no raw data, no GPU)."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest

from chaoyang.exact78_baseline import mask, pipeline, render, retarget

REPO = Path(__file__).resolve().parents[1]
SHAPE = (480, 640)


def _palm_down_hand(side: int) -> np.ndarray:
    """Right-handed world (x right, y up, z toward viewer); palm faces -y, fingers point -z."""
    mirror = 1.0 if side == 1 else -1.0
    hand = np.zeros((21, 3))
    hand[5] = (-0.02 * mirror, 0.0, -0.08)
    hand[9] = (0.005 * mirror, 0.0, -0.085)
    hand[17] = (0.03 * mirror, 0.0, -0.07)
    return hand


@pytest.mark.parametrize("side", [0, 1])
def test_tracker_puck_sits_on_back_of_hand_behind_wrist(side):
    points = mask.tracker_band_points(_palm_down_hand(side), np.array([0.0, 0.0, 0.25]), side)
    rings, puck = points[:2 * mask.BAND_RING_POINTS], points[2 * mask.BAND_RING_POINTS:]
    assert puck[:, 1].min() > 0.04
    assert rings[:, 2].min() > -1e-9
    assert np.allclose(np.linalg.norm(rings[:, :2], axis=1), mask.BAND_RADIUS_M)


class _CameraWorldSession:
    """World == camera optical frame (x right, y down, z forward); one frame, left hand only."""

    def __init__(self, elbow_z: float):
        self.frame_count, self.width, self.height = 1, 1280, 960
        self.K = np.array([[640.0, 0, 640], [0, 640.0, 480], [0, 0, 1]])
        self.hands = np.zeros((2, 1, 21, 3))
        self.hands[0, 0] = np.array([-0.05, 0.10, 0.30]) + np.random.default_rng(2).uniform(-0.03, 0.03, (21, 3))
        self.hands[1, 0] = (0.0, 0.0, -1.0)
        self.hand_source = np.array([["OBSERVED"], ["NONE"]])
        self.body_observed = np.array([True])
        self.body = np.zeros((1, 24, 3))
        self.body[0, 18] = (-0.15, 0.35, elbow_z)

    def camera_points(self, points: np.ndarray, frame: int) -> np.ndarray:
        return np.asarray(points, float)

    def project(self, points: np.ndarray) -> np.ndarray:
        q = points @ self.K.T
        return q[..., :2] / q[..., 2:3]


def test_wrist_band_is_kept_when_the_elbow_is_behind_the_camera():
    geometry = mask.pico_geometry(_CameraWorldSession(elbow_z=-0.05), 0.5)
    band = geometry["tracker"][0, 0]
    assert np.isfinite(band).all(-1).sum() >= 3
    assert mask._hull_mask(band, SHAPE, 0).sum() > 500
    forearm = geometry["forearm"][0, 0]
    assert np.isfinite(forearm[0]).all() and not np.isfinite(forearm[-1]).all()
    assert mask._capsule_mask(forearm, geometry["forearm_radius"][0, 0], SHAPE).any()
    assert not np.isfinite(geometry["tracker"][1, 0]).any()


def _geometry(count: int, band_frames: tuple[int, ...]) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    hands = np.full((2, count, 21, 2), np.nan)
    hands[1] = rng.uniform((300, 200), (360, 260), size=(count, 21, 2))
    forearm = np.full((2, count, 6, 2), np.nan)
    forearm[1] = np.stack([np.full(6, 330.0), np.linspace(260, 400, 6)], -1)
    radius = np.full((2, count), np.nan)
    radius[1] = 20.0
    tracker = np.full((2, count, 3 * mask.BAND_RING_POINTS, 2), np.nan)
    angles = np.linspace(0, 2 * np.pi, 3 * mask.BAND_RING_POINTS, endpoint=False)
    for frame in band_frames:
        tracker[1, frame] = np.stack([500 + 15 * np.cos(angles), 100 + 15 * np.sin(angles)], -1)
    inside = np.isfinite(hands).all(-1)
    px_per_m = np.full((2, count), np.nan)
    px_per_m[1] = 1000.0
    return {"hands": hands, "forearm": forearm, "forearm_radius": radius, "tracker": tracker,
            "wrist_px_per_m": px_per_m, "hands_inside": inside}


def test_build_masks_without_sam_uses_pico_fallback_and_covers_tracker_in_nearby_frames():
    count = 2 * mask.BAND_TEMPORAL_FRAMES + 3
    middle = count // 2
    core, removal, info = mask.build_masks({}, _geometry(count, band_frames=(middle,)), count, SHAPE)
    assert (info["provider"][:, 0] == 2).all()
    assert (info["provider"][:, 1] == 1).all()
    assert core[:, 230, 330].all() and core[:, 330, 330].all()
    covered = [bool(core[f, 100, 500]) for f in range(count)]
    assert covered == [abs(f - middle) <= mask.BAND_TEMPORAL_FRAMES for f in range(count)]
    assert (removal | ~core).all()
    assert removal.sum() > core.sum()


def test_wrist_closure_bridges_the_strap_gap_between_hand_and_forearm_masks():
    count = 1
    geometry = _geometry(count, band_frames=())
    geometry["hands"][1, 0, 0] = (330, 230)
    sam = np.zeros(SHAPE, bool)
    sam[150:215, 300:360] = True     # hand part
    sam[245:330, 300:360] = True     # forearm part; strap gap rows 215-245
    tracks = {0: {0: np.packbits(sam)}}
    core, removal, info = mask.build_masks(tracks, geometry, count, SHAPE)
    assert info["kept_obj_ids"] == [0] and info["provider"][0, 1] == 0
    # forearm probes at rows 316 / 344: the forearm part ends at row 330
    assert info["sam_forearm_hit_rate"] == {"left": None, "right": 0.5}
    assert info["removal_bracket_frames"] == mask.REMOVAL_BRACKET_FRAMES
    assert removal[0, 220:240, 305:355].all() and not removal[0, 230, 450]
    assert not core[0, 230, 330]
    card = np.zeros(SHAPE, bool)
    card[218:242, 320:340] = True    # held between the hand and forearm parts
    held = mask.held_objects({5: {0: np.packbits(card)}}, core, SHAPE)[0]
    assert held[230, 330] and not (removal[0] & ~held)[230, 330]


def test_tracker_colours_are_neutral_greys_and_the_green_rim():
    swatches = np.asarray([[[235, 235, 235], [110, 112, 115], [61, 66, 74], [150, 200, 40],   # shell, straps, rim
                            [200, 150, 120], [200, 30, 30], [40, 60, 200]]], np.uint8)  # skin, red, blue card back
    assert mask.tracker_coloured(swatches)[0].tolist() == [True, True, True, True, False, False, False]


def test_removal_grows_into_tracker_coloured_pixels_next_to_the_band_only():
    count = 1
    geometry = _geometry(count, band_frames=(0,))
    rgb = np.full((1, *SHAPE, 3), 60, np.uint8)            # dark table
    rgb[0, 52:66, 480:520] = 220                            # light shell sticking out above the band hull
    rgb[0, 400:410, 100:140] = 220                          # light object far from any band
    _, plain, _ = mask.build_masks({}, geometry, count, SHAPE)
    _, grown, _ = mask.build_masks({}, geometry, count, SHAPE, frames_rgb=rgb)
    assert not plain[0, 55:62, 500].any() and grown[0, 55:62, 500].all()
    assert not grown[0, 405, 120]
    core, _, _ = mask.build_masks({}, geometry, count, SHAPE, frames_rgb=rgb)
    card = np.zeros(SHAPE, bool)
    card[52:66, 470:530] = True      # a white card lying next to the band, not touching the hand
    card[100, 500] = True            # inside the band hull: still removed
    kept = mask.object_pixels({3: {0: np.packbits(card)}}, core, SHAPE)[0]
    assert kept[58, 500] and not kept[100, 500]


def test_fallback_frames_also_place_band_and_closure_where_the_pico_hand_registers_onto_sam():
    count = 1
    geometry = _geometry(count, band_frames=(0,))
    sam = np.zeros(SHAPE, bool)
    sam[140:220, 240:320] = True      # image hand 50 px up-left of the PICO keypoint box (300-360, 200-260)
    sam[250:470, 20:200] = True       # thick forearm further away: covers the keypoints as well, deeper inside
    sam[10:40, 600:630] = True        # a far piece of the same track that must not attract the keypoints
    tracks = {0: {0: np.packbits(sam)}}
    core, _, info = mask.build_masks(tracks, geometry, count, SHAPE)
    assert info["provider"][0, 1] == 3
    assert np.abs(info["pico_shift_px"][0, 1] - (-50, -50)).max() <= 8
    assert core[0, 230, 330] and core[0, 100, 500]          # PICO hull and unshifted band are kept
    assert core[0, 50, 450] and not core[0, 160, 560]       # band repeated at the registered shift only


def test_fallback_frames_that_do_not_register_carry_the_nearest_registered_shift():
    count = 3
    geometry = _geometry(count, band_frames=(0, 1, 2))
    sam = np.zeros(SHAPE, bool)
    sam[140:220, 240:320] = True
    tracks = {0: {0: np.packbits(sam), 2: np.packbits(sam)}}     # SAM misses the hand in frame 1
    core, _, info = mask.build_masks(tracks, geometry, count, SHAPE)
    assert info["provider"][:, 1].tolist() == [3, 4, 3]
    assert (info["pico_shift_px"][1, 1] == info["pico_shift_px"][0, 1]).all()
    assert core[1, 230, 330] and core[1, 50, 450]


def test_registration_needs_sam_under_most_keypoints():
    points = np.random.default_rng(1).uniform((300, 200), (360, 260), size=(21, 2))
    near = np.zeros(SHAPE, bool)
    near[400:470, 20:40] = True
    depth = near.astype(np.float32)
    assert mask.register_to_sam(points, near, depth) is None


def test_triangles_crossing_the_camera_plane_are_near_clipped_before_rasterising():
    triangles = np.asarray([[[0, 0, 0.5], [0.1, 0, 0.5], [0, 0.1, 0.5]],                  # in front
                            [[0.14, -0.14, -0.004], [0.1, -0.1, 0.09], [0.16, -0.1, 0.09]],  # flange crossing z = 0
                            [[0, 0, -0.2], [0.1, 0, -0.2], [0, 0.1, -0.1]]])              # behind
    colors = np.asarray([[1, 1, 1], [2, 2, 2], [3, 3, 3]], np.uint8)
    labels = np.asarray([10, 4001, 20], np.int32)
    clipped, piece_colors, piece_labels = render.near_clip(triangles, colors, labels)
    assert clipped[:, :, 2].min() >= render.raster.NEAR_M - 1e-12
    assert set(piece_labels.tolist()) == {10, 4001} and (piece_colors[piece_labels == 4001] == 2).all()
    camera = (500.0, 500.0, 320.0, 240.0, 640, 480)
    _, _, unclipped = render.raster.rasterize_zbuffer(triangles, colors, labels, *camera)
    _, _, link = render._NearClippedRaster().rasterize_zbuffer(triangles, colors, labels, *camera)
    assert (unclipped == 4001).mean() > 0.3 and (link == 4001).mean() < 0.01


@pytest.mark.parametrize("message, aborted", [("No points are provided; please add points first", True),
                                              ("CUDA out of memory", False)])
def test_sam_propagation_without_instances_is_recorded_and_other_errors_raise(message, aborted, monkeypatch):
    monkeypatch.setitem(sys.modules, "PIL", types.SimpleNamespace(Image=types.SimpleNamespace(fromarray=lambda a: a)))

    class Model:
        def init_state(self, **_):
            return types.SimpleNamespace(clear=lambda: None)

        def add_prompt(self, **_):
            return None, {}

        def propagate_in_video(self, **_):
            raise RuntimeError(message)
            yield

    video = object.__new__(mask.Sam3Video)
    video.model = Model()
    video._store = lambda *args: None
    frames = [np.zeros((4, 4, 3), np.uint8)] * 3
    if aborted:
        assert video.track_text(frames, "potato chip", 1) == {}
        assert video.aborted == ["forward", "reverse"]
    else:
        with pytest.raises(RuntimeError, match="out of memory"):
            video.track_text(frames, "potato chip", 1)


def test_sam_track_far_from_pico_keypoints_is_rejected():
    count = 3
    far = np.zeros(SHAPE, bool)
    far[10:40, 10:40] = True
    tracks = {7: {frame: np.packbits(far) for frame in range(count)}}
    core, _, info = mask.build_masks(tracks, _geometry(count, band_frames=()), count, SHAPE)
    assert info["kept_obj_ids"] == []
    assert not core[:, 25, 25].any()


def test_held_objects_keep_only_small_instances_touching_the_wearer():
    core = np.zeros((1, *SHAPE), bool)
    core[0, 200:260, 300:360] = True
    touching, distant, huge = (np.zeros(SHAPE, bool) for _ in range(3))
    touching[240:280, 355:400] = True
    distant[20:40, 20:40] = True
    huge[:300, :] = True
    tracks = {obj: {0: np.packbits(m)} for obj, m in enumerate((touching, distant, huge))}
    held = mask.held_objects(tracks, core, SHAPE)[0]
    assert held[270, 380] and not held[30, 30] and not held[100, 600]
    assert not (held & core[0]).any()


def test_object_tracks_that_degenerate_into_speckle_are_neither_held_nor_kept():
    core = np.zeros((1, *SHAPE), bool)
    core[0, 200:260, 300:360] = True
    speckle = np.zeros(SHAPE, bool)
    speckle[250:470:6, 280:620:6] = True       # isolated dots over the table, touching the hand
    chip = np.zeros(SHAPE, bool)
    chip[255:275, 355:380] = True
    tracks = {0: {0: np.packbits(speckle)}, 1: {0: np.packbits(chip)}}
    held = mask.held_objects(tracks, core, SHAPE)[0]
    kept = mask.object_pixels(tracks, core, SHAPE)[0]
    assert held[265, 370] and kept[265, 370]
    assert not (held & speckle & ~chip).any() and not (kept & speckle & ~chip).any()


def test_bracket_fill_covers_short_gaps_between_removed_frames_only():
    removal = np.zeros((14, 4, 4), bool)
    removal[[0, 1, 7, 8], 0, 0] = True      # exposed in frames 2-6
    removal[0:3, 3, 3] = True               # never removed again
    filled = mask.bracket_fill(removal, 8)
    assert filled[2:7, 0, 0].all() and not filled[9:, 0, 0].any()
    assert not filled[3:, 3, 3].any()
    assert mask.bracket_fill(removal, 3)[2:7, 0, 0].tolist() == [False, False, True, False, False]


def test_recomputed_sessions_take_only_motion_from_reuse_roots(tmp_path):
    old = tmp_path / "old"
    folder = old / "sessions" / "s1"
    folder.mkdir(parents=True)
    for names in pipeline.STAGE_FILES.values():
        for name in names:
            (folder / name).write_text(json.dumps({"status": "OK"}))
    stages = ("motion", "gpu", "render")
    layout = pipeline.Layout(tmp_path / "new", [old], stages, recompute=("s1",))
    assert layout.find("s1", "motion") == folder
    assert layout.find("s1", "gpu") is None and layout.find("s1", "render") is None
    assert pipeline.Layout(tmp_path / "new", [old], stages).find("s1", "gpu") == folder


def test_prompt_anchor_prefers_a_frame_with_the_forearm_in_view():
    geometry = _geometry(3, band_frames=())
    geometry["forearm"][1, 0] = np.nan             # hand only
    geometry["forearm"][1, 1, 1:] = np.nan         # wrist only
    geometry["hands_inside"][1, 2, :3] = False     # wrist + forearm, three keypoints cut off
    assert mask.anchor_frame(geometry, SHAPE) == 2
    geometry["forearm"][1, 2] = np.nan
    assert mask.anchor_frame(geometry, SHAPE) == 1
    edge = np.array([[SHAPE[1] - 0.4, 10.0], [SHAPE[1] - 0.6, 10.0], [-0.4, 10.0], [-0.6, 10.0]])
    assert mask._inside(edge, SHAPE).tolist() == [False, True, True, False]


class _AnchoredSam:
    """Finds hand and forearm only when prompted on `arm_frame`, otherwise the hand alone."""

    def __init__(self, arm_frame: int | None):
        self.arm_frame, self.anchors = arm_frame, []
        self.hand = np.zeros(SHAPE, bool)
        self.hand[195:265, 295:365] = True
        self.arm = self.hand.copy()
        self.arm[265:410, 310:350] = True

    def track_text(self, frames, prompt, anchor):
        self.anchors.append(anchor)
        self.aborted = []
        found = self.arm if self.arm_frame in (None, anchor) else self.hand
        return {0: {frame: np.packbits(found) for frame in range(len(frames))}}


def test_wearer_prompt_reruns_on_the_frame_showing_the_forearm_when_sam_misses_it():
    count = 12
    geometry = _geometry(count, band_frames=())
    geometry["forearm"][1, :-1, 4:] = np.nan        # elbow end in view only in the last frame
    frames = np.zeros((count, *SHAPE, 3), np.uint8)
    sam = _AnchoredSam(arm_frame=count - 1)
    core, removal, info = mask.track_wearer(sam, frames, geometry, count, SHAPE, anchor=0)
    assert sam.anchors == [0, count - 1]
    assert info["forearm_retry"] == [{"side": "right", "anchor_frame": count - 1, "hit_rate_before": 0.0,
                                      "tracks": 1, "propagation_aborted": []}]
    assert info["sam_forearm_hit_rate"]["right"] == 1.0 and info["kept_obj_ids"] == [0, 1]
    assert core[:, 380, 330].all() and removal[:, 380, 330].all()
    sam = _AnchoredSam(arm_frame=None)
    _, _, info = mask.track_wearer(sam, frames, geometry, count, SHAPE, anchor=0)
    assert sam.anchors == [0] and info["forearm_retry"] == [] and info["propagation_aborted"] == []


def test_retry_tracks_add_to_the_mask_but_do_not_turn_fallback_frames_into_sam_frames():
    count = 4
    geometry = _geometry(count, band_frames=())
    on_pico = np.zeros(SHAPE, bool)
    on_pico[195:265, 295:365] = True
    off_pico = np.roll(on_pico, -200, axis=1)        # the image hand, 200 px left of the PICO hand
    first = {frame: np.packbits(on_pico) for frame in range(3)} | {3: np.packbits(off_pico)}
    tracks = {0: first, 1: {3: np.packbits(on_pico)}}  # retry track drifted onto the PICO hand spot
    _, removal, info = mask.build_masks(tracks, geometry, count, SHAPE, provider_tracks={0})
    assert info["kept_obj_ids"] == [0, 1] and info["provider_obj_ids"] == [0]
    assert info["provider"][:3, 1].tolist() == [0, 0, 0] and info["provider"][3, 1] == 3
    assert abs(info["pico_shift_px"][3, 1, 0] + 200) <= 8
    assert removal[3, 230, 130] and removal[3, 230, 330]
    _, _, info = mask.build_masks(tracks, geometry, count, SHAPE)
    assert info["provider"][3, 1] == 0


def test_smooth_joints_respects_limits_and_short_tracks():
    lower, upper = np.full(3, -1.0), np.full(3, 1.0)
    noisy = np.clip(np.random.default_rng(1).normal(0, 2, (60, 3)), -1, 1)
    smooth = retarget.smooth_joints(noisy, 30.0, lower, upper)
    assert smooth.shape == noisy.shape and (smooth >= lower).all() and (smooth <= upper).all()
    assert np.abs(np.diff(smooth, axis=0)).mean() < np.abs(np.diff(noisy, axis=0)).mean()
    short = noisy[:3]
    assert np.array_equal(retarget.smooth_joints(short, 30.0, lower, upper), short)


@pytest.mark.skipif(not (REPO / "assets/robot").exists(), reason="robot assets not present")
@pytest.mark.parametrize("side", [0, 1])
def test_arm_ik_recovers_reachable_hand_root_pose(side):
    from chaoyang.exact78_baseline.robot_model import Robot

    robot = Robot(REPO)
    span = robot.arm_upper[side] - robot.arm_lower[side]
    q_true = robot.arm_lower[side] + span * np.random.default_rng(side).uniform(0.35, 0.65, span.shape)
    target = robot.arm_frames(side, q_true)["root"]
    seed = np.clip(q_true + 0.1, robot.arm_lower[side], robot.arm_upper[side])
    q = retarget.solve_arm_frame(robot, side, target, None, seed, None)
    position_mm, rotation_deg = retarget.arm_errors(robot, side, q, target)
    assert position_mm < 1.0 and rotation_deg < 0.5
