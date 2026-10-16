"""Robot layer rendering (z-buffer raster shared with the Cards042 V2 render)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from chaoyang.exact78_baseline.robot_model import FLANGE_MESH, Robot
from chaoyang.pipeline import exact78_robot_raster_v1 as raster


WIDTH, HEIGHT = 640, 480
EXTERNAL_K = np.asarray(((575.0, 0, 319.5), (0, 575.0, 239.5), (0, 0, 1.0)))
REVIEW_BACKGROUND = 242


def scaled_intrinsics(K: np.ndarray, width: int, height: int, native: tuple[int, int]) -> np.ndarray:
    value = np.asarray(K, dtype=np.float64).copy()
    value[0] *= width / float(native[0])
    value[1] *= height / float(native[1])
    return value


def external_camera(ego_camera_frame0: np.ndarray, target_base=(0.30, 0.0, 1.0),
                    distance: float = 2.2) -> np.ndarray:
    """Same viewing direction as the first ego frame, pulled back to show the whole robot."""
    transform = np.eye(4)
    transform[:3, :3] = ego_camera_frame0[:3, :3]
    transform[:3, 3] = np.asarray((0.0, 0.0, distance)) - transform[:3, :3] @ np.asarray(target_base)
    return transform


def near_clip(triangles: np.ndarray, colors: np.ndarray, labels: np.ndarray,
              near: float = raster.NEAR_M) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Camera-space triangles clipped against z >= near, colours and labels carried per piece."""
    in_front = triangles[:, :, 2] >= near
    whole = in_front.all(axis=1)
    mixed = in_front.any(axis=1) & ~whole
    pieces, piece_colors, piece_labels = [triangles[whole]], [colors[whole]], [labels[whole]]
    for index in np.flatnonzero(mixed):
        for piece in raster.clip_triangle_near(triangles[index], near):
            pieces.append(piece[None])
            piece_colors.append(colors[index][None])
            piece_labels.append(labels[index][None])
    return (np.ascontiguousarray(np.concatenate(pieces), dtype=np.float64),
            np.ascontiguousarray(np.concatenate(piece_colors)),
            np.ascontiguousarray(np.concatenate(piece_labels)))


class _NearClippedRaster:
    """The shared raster module, except that every triangle is near-clipped before z-buffering.

    render_robot clips arm and hand meshes but not the camera-mount flange; a flange crossing the camera
    plane (robot wrist next to the head camera) otherwise projects to a screen-filling slab.
    """

    def __getattr__(self, name: str):
        return getattr(raster, name)

    @staticmethod
    def rasterize_zbuffer(triangles, colors, labels, *intrinsics_and_size):
        return raster.rasterize_zbuffer(*near_clip(triangles, colors, labels), *intrinsics_and_size)


class RobotRenderer:
    def __init__(self, robot: Robot):
        self.robot = robot
        self.flange = raster.naturalv2_local_triangles(robot.repo_root / FLANGE_MESH)
        self.cache: dict = {}

    def render(self, q_arm: np.ndarray, q_hand: np.ndarray, T_camera_base: np.ndarray,
               K: np.ndarray, *, complete_robot: bool, width: int = WIDTH,
               height: int = HEIGHT) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (bgr, alpha, hand_roots_in_camera)."""
        bgr, labels, roots = raster.render_robot(
            _NearClippedRaster(), self.robot.assets, q_arm, q_hand, self.robot.tool_mounts, T_camera_base, K,
            self.cache, self.flange, complete_robot=complete_robot, return_buffers=False,
            width=width, height=height)
        return bgr, labels >= 0, roots


def soft_alpha(alpha: np.ndarray) -> np.ndarray:
    """Anti-aliased coverage from a binary raster mask (1 px feather)."""
    value = cv2.GaussianBlur(alpha.astype(np.float32), (3, 3), 0.7)
    value[alpha] = np.maximum(value[alpha], 0.5)
    return value


def over(background: np.ndarray, layer: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    weight = soft_alpha(alpha)[..., None]
    return np.clip(background.astype(np.float32) * (1 - weight) + layer.astype(np.float32) * weight,
                   0, 255).astype(np.uint8)


def on_grey(layer: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    return over(np.full_like(layer, REVIEW_BACKGROUND), layer, alpha)


def title(image: np.ndarray, text: str) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(output, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return output


HAND_EDGES = ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10),
              (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18),
              (18, 19), (19, 20), (5, 9), (9, 13), (13, 17))


def draw_hand(image: np.ndarray, points: np.ndarray, colour: tuple[int, int, int],
              label: str | None = None) -> None:
    if not np.isfinite(points).all():
        return
    pts = np.round(points).astype(int)
    for a, b in HAND_EDGES:
        cv2.line(image, tuple(pts[a]), tuple(pts[b]), colour, 2, cv2.LINE_AA)
    for point in pts:
        cv2.circle(image, tuple(point), 3, colour, -1, cv2.LINE_AA)
    if label:
        cv2.putText(image, label, tuple(pts[0] + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
