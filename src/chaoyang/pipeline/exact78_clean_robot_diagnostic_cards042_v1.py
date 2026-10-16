"""Unrouted Cards042 Clean + Robot diagnostic compositor candidate.

This module has no CLI or writer. A future finite Task Packet must bind the
three sealed outer RESULT refs and call it through a governed executor. Clean
is a visual background only; no scene depth or object-front decision is made.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np

from chaoyang.governance.common import REPO_ROOT, artifact_ref
from chaoyang.pipeline import exact78_robot_render_direction_v1 as direction


SESSION = "play_cards_0902_042"
FRAME_COUNT = 171
FPS = 30
OUTER_SHA = {
    "scene_result": "d065012d7d774dd5c7f8b4d42be0555dc7f5f3d9c28e29a950be65de7e027684",
    "action_result": "ad79f92d7cc822617e733d70975b29c5f7890ecba708c351a91aaec8ae6e31d6",
    "render_result": "27608fe7778accaf6f151f81ce6fb21a332ab98ee0afa8c0bc0e19f620953c0f",
}
RENDER_CODE_SHA = "8158e106ee3f12fe0cdafc950349ee5f92d15eba1312a5af3f731648f1565913"
DISPLAY_POLICY = "DIAGNOSTIC_ROBOT_OVER_SYNTHETIC_CLEAN_NO_OCCLUSION_PROOF"


class CompositeBindingError(RuntimeError):
    """A finite candidate cannot be bound to its sealed inputs."""


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise CompositeBindingError(code)


def _array_sha(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    return hashlib.sha256(value.tobytes()).hexdigest()


def _read_ref(ref: dict[str, Any], role: str) -> tuple[Path, Any]:
    _require(isinstance(ref, dict) and set(ref) == {"path", "bytes", "sha256"},
             f"BAD_REF:{role}")
    path = Path(ref["path"])
    _require(path.is_absolute() and path.is_file() and not path.is_symlink()
             and path.resolve().is_relative_to(REPO_ROOT), f"OUTSIDE_SEALED_PROJECT:{role}")
    _require(artifact_ref(path) == ref, f"SHA_DRIFT:{role}")
    return path, json.loads(path.read_text(encoding="utf-8"))


def _file_ref(ref: dict[str, Any], role: str) -> Path:
    _require(isinstance(ref, dict) and set(ref) == {"path", "bytes", "sha256"},
             f"BAD_REF:{role}")
    path = Path(ref["path"])
    _require(path.is_absolute() and path.is_file() and not path.is_symlink()
             and path.resolve().is_relative_to(REPO_ROOT), f"OUTSIDE_SEALED_PROJECT:{role}")
    _require(artifact_ref(path) == ref, f"SHA_DRIFT:{role}")
    return path


def validate_frame_rows(raw_rows: list[dict[str, Any]], mask_rows: list[dict[str, Any]],
                        clean_rows: list[dict[str, Any]], provenance_rows: list[dict[str, Any]],
                        jpeg_rows: list[dict[str, Any]], count: int) -> None:
    """Check clip-index identity, never infer capture/QPC synchronization."""
    _require(all(len(rows) == count for rows in
                 (raw_rows, mask_rows, clean_rows, provenance_rows, jpeg_rows)),
             "FRAME_COUNT_MISMATCH")
    for i, (raw, mask, clean, provenance, jpeg) in enumerate(zip(
            raw_rows, mask_rows, clean_rows, provenance_rows, jpeg_rows, strict=True)):
        _require(raw.get("decode_display_index") == i and raw.get("model_input_index") == i
                 and mask.get("decode_display_index") == i
                 and clean.get("decode_display_index") == i
                 and clean.get("model_output_index") == i
                 and provenance.get("decoded_display_index") == i
                 and provenance.get("model_input_index") == i
                 and provenance.get("output_index") == i
                 and jpeg.get("decoded_display_index") == i
                 and jpeg.get("model_input_index") == i, f"FRAME_INDEX_MISMATCH:{i}")
        _require(raw.get("model_input") == mask.get("model_input")
                 == clean.get("model_input") and clean.get("masks") == mask.get("masks"),
                 f"SCENE_FRAME_REF_MISMATCH:{i}")
        _require(raw.get("time_mode") == "TIMEBASE_UNVERIFIED"
                 and provenance.get("time_mode") == "CLIP_FRAME_INDEX_ONLY"
                 and raw.get("capture_frame_id") is None
                 and provenance.get("capture_frame_id") is None,
                 f"TIMEBASE_CLAIM_DRIFT:{i}")


def prepare_binding(spec: dict[str, Any]) -> dict[str, Any]:
    """CPU-only read of sealed *fresh* results; no output or model call."""
    _require(isinstance(spec, dict) and set(spec) == set(OUTER_SHA), "SPEC_ROLES")
    _require(artifact_ref(REPO_ROOT / "src/chaoyang/pipeline/exact78_robot_render_direction_v1.py")
             ["sha256"] == RENDER_CODE_SHA, "RENDER_CODE_SHA_DRIFT")
    direction.verify_code_pins()
    documents = {}
    for role, sha in OUTER_SHA.items():
        _require(spec[role].get("sha256") == sha, f"UNFROZEN_OUTER_RESULT:{role}")
        path, documents[role] = _read_ref(spec[role], role)
        expected_task = {
            "scene_result": "exact78_raw_to_robot_t2_cards042_frontend_v1_20260930",
            "action_result": "exact78_raw_to_robot_t2_0902_cards042_v1_20260930",
            "render_result": "exact78_robot_render_direction_cards042_v1_20260930",
        }[role]
        _require(documents[role].get("task_id") == expected_task
                 and documents[role].get("source_session") == SESSION
                 and documents[role].get("source_frames_expected") == FRAME_COUNT
                 and documents[role].get("status") == "PASSED"
                 and documents[role].get("execution_complete") is True,
                 f"OUTER_NOT_SEALED:{role}")
        _require(path.parent.name == "attempt_0001", f"UNEXPECTED_ATTEMPT:{role}")

    scene = documents["scene_result"]
    action = documents["action_result"]
    rendered = documents["render_result"]
    seals = {}
    scene_stage = {}
    for name in ("MAP", "RAW", "MASK", "PREPARE", "CLEAN"):
        _, seal = _read_ref(scene["stage_seals"][name], f"seal_{name}")
        expected_count = 0 if name == "MAP" else FRAME_COUNT
        _require(seal.get("stage") == name and seal.get("session_id") == SESSION
                 and seal.get("frame_count") == expected_count, f"SEAL_ROLE_DRIFT:{name}")
        _, scene_stage[name] = _read_ref(seal["result"], f"stage_{name}")
        seals[name] = seal
    for previous, current in zip(("MAP", "RAW", "MASK", "PREPARE"),
                                 ("RAW", "MASK", "PREPARE", "CLEAN"), strict=True):
        _require(seals[current].get("upstream_seal") == scene["stage_seals"][previous],
                 f"SEAL_CHAIN_BROKEN:{current}")
    raw, mask, clean, camera_map = (scene_stage[k] for k in ("RAW", "MASK", "CLEAN", "MAP"))
    _require(all(stage.get("session_id") == SESSION for stage in (raw, mask, clean, camera_map))
             and raw.get("source_fps") == FPS and clean.get("source_fps") == FPS
             and raw.get("time_mode") == "TIMEBASE_UNVERIFIED"
             and clean.get("time_mode") == "TIMEBASE_UNVERIFIED"
             and clean.get("quality") == "NOT_YET_ADOPTED"
             and clean.get("use_scope") == "OFFLINE_BIDIRECTIONAL_VISUAL_ONLY"
             and clean.get("depth_object6d_contact_truth") is False,
             "SCENE_DOMAIN_OR_CLAIM_DRIFT")
    _require(mask.get("visible_object_identity") == "UNVERIFIED_TEXT_UNION",
             "MASK_EVIDENCE_CLAIM_DRIFT")

    _, child = _read_ref(action["child_result"], "action_child")
    _require(child.get("session_id") == SESSION and child.get("execution_complete") is True,
             "ACTION_CHILD_DRIFT")
    _, hawor_seal = _read_ref(child["stages"]["hawor_seal"], "hawor_seal")
    _, robot_seal = _read_ref(child["stages"]["robot_seal"], "robot_seal")
    _require(robot_seal.get("input_hawor_seal") == child["stages"]["hawor_seal"],
             "ROBOT_HAWOR_CHAIN_BROKEN")
    _, hawor_result = _read_ref(hawor_seal["result"], "hawor_result")
    _, robot_result = _read_ref(robot_seal["result"], "robot_result")
    _, render_candidate = _read_ref(rendered["candidate_result"], "render_candidate")
    _, render_spec = _read_ref(rendered["render_spec"], "render_spec")
    _require(render_candidate.get("status") == "COMPLETE_RENDER_DIAGNOSTIC_NOT_VISUAL_PASS"
             and render_candidate.get("session_id") == SESSION
             and render_candidate.get("frame_count") == FRAME_COUNT
             and render_candidate.get("ego_camera_rule") == "T_CAMERA_BASE_BASE_TO_CAMERA_NO_INVERSION"
             and render_candidate.get("visual_qualified") is False
             and render_candidate.get("clean_or_depth_consumed") is False
             and render_spec.get("inputs") == render_candidate.get("inputs")
             and render_candidate["inputs"]["stage_result"] == robot_result["stage_result"]
             and render_candidate["inputs"]["hawor_npz"] == hawor_result["npz"]
             and render_candidate["inputs"]["frame_provenance"] == hawor_result["frame_provenance"]
             and render_candidate["inputs"]["input_audit"] == hawor_result["input_audit"],
             "RENDER_ACTION_CHAIN_BROKEN")
    _file_ref(render_candidate["inputs"]["states"], "robot_states")
    _, provenance = _read_ref(hawor_result["frame_provenance"], "frame_provenance")
    _, audit = _read_ref(hawor_result["input_audit"], "hawor_input_audit")
    _require(provenance.get("session_id") == SESSION
             and provenance.get("frame_count") == FRAME_COUNT
             and provenance.get("fps_nominal") == FPS
             and provenance.get("time_mode") == "CLIP_FRAME_INDEX_ONLY"
             and audit.get("session_id") == SESSION,
             "ACTION_FRAME_DOMAIN_DRIFT")
    validate_frame_rows(raw["rows"], mask["rows"], clean["rows"],
                        provenance["frame_rows"], audit["model_input_jpeg_refs"], FRAME_COUNT)

    with np.load(_file_ref(hawor_result["npz"], "hawor_npz"), allow_pickle=False) as archive:
        intrinsics = np.asarray(archive["intrinsics"]).copy()
        _require(intrinsics.shape == (FRAME_COUNT, 3, 3)
                 and np.array_equal(archive["original_frame_indices"], np.arange(FRAME_COUNT))
                 and float(archive["fps_nominal"]) == FPS, "HAWOR_ARRAY_DOMAIN_DRIFT")
    with np.load(_file_ref(render_candidate["inputs"]["states"], "robot_states"),
                 allow_pickle=False) as archive:
        _require(set(archive.files) == direction.STATE_KEYS, "ROBOT_STATE_KEYS_DRIFT")
        states = {key: np.asarray(archive[key]).copy() for key in direction.STATE_KEYS}
    _require(states["q_arm"].shape == (FRAME_COUNT, 2, 7)
             and states["q_hand"].shape == (FRAME_COUNT, 2, 22)
             and states["T_camera_base"].shape == (4, 4)
             and states["T_tool_hand_root"].shape == (2, 4, 4)
             and np.array_equal(states["original_frame_indices"], np.arange(FRAME_COUNT))
             and float(states["fps_nominal"]) == FPS,
             "ROBOT_STATE_DOMAIN_DRIFT")
    expected_k = np.asarray(camera_map["output_intrinsics"], dtype=np.float64)
    _require(expected_k.shape == (3, 3)
             and np.isfinite(intrinsics).all()
             and np.allclose(intrinsics, expected_k[None], rtol=0, atol=1e-8),
             "CAMERA_INTRINSICS_DOMAIN_DRIFT")
    for i, row in enumerate(clean["rows"]):
        _file_ref(row["clean"], f"clean_{i}")
        _file_ref(raw["rows"][i]["model_input"], f"raw_{i}")
        _file_ref(audit["model_input_jpeg_refs"][i]["input_jpeg"], f"hawor_input_{i}")
    return {"scene": scene, "action": action, "render": rendered,
            "robot_states_ref": render_candidate["inputs"]["states"],
            "raw_rows": raw["rows"], "mask_rows": mask["rows"],
            "clean_rows": clean["rows"],
            "provenance_rows": provenance["frame_rows"],
            "intrinsics": intrinsics, "states": states,
            "frame_count": FRAME_COUNT, "fps_nominal": FPS,
            "time_mode": "CLIP_INDEX_ONLY_TIMEBASE_UNVERIFIED",
            "image_domain_claim": "INDEX_AND_K_MATCH_COMPRESSED_PIXEL_EQUIVALENCE_NOT_PROVEN",
            "geometry_decision": "UNKNOWN", "display_policy": DISPLAY_POLICY}


def compose_frame(clean_bgr: np.ndarray, robot_rgb: np.ndarray | None,
                  robot_labels: np.ndarray | None, robot_buffers: dict[str, Any] | None,
                  *, frame_index: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Overlay only Robot-visible pixels on Clean; never infer scene order."""
    if robot_rgb is None or robot_labels is None or robot_buffers is None:
        raise CompositeBindingError("BLOCKED_RENDER_LAYER_ABSENT")
    if "depth_m" not in robot_buffers:
        raise CompositeBindingError("BLOCKED_RENDER_LAYER_ABSENT")
    _require(clean_bgr.shape == (960, 1280, 3) and clean_bgr.dtype == np.uint8,
             f"CLEAN_IMAGE_DOMAIN_DRIFT:{frame_index}")
    _require(robot_rgb.shape == (direction.HEIGHT, direction.WIDTH, 3)
             and robot_rgb.dtype == np.uint8
             and robot_labels.shape == (direction.HEIGHT, direction.WIDTH)
             and np.issubdtype(robot_labels.dtype, np.integer)
             and np.asarray(robot_buffers["depth_m"]).shape == (direction.HEIGHT, direction.WIDTH),
             f"ROBOT_LAYER_DOMAIN_DRIFT:{frame_index}")
    alpha = robot_labels >= 0
    depth = np.asarray(robot_buffers["depth_m"])
    _require(np.isfinite(depth[alpha]).all() and np.all(depth[alpha] > 0),
             f"ROBOT_LAYER_DEPTH_INVALID:{frame_index}")
    composite = cv2.resize(clean_bgr, (direction.WIDTH, direction.HEIGHT),
                           interpolation=cv2.INTER_AREA)
    composite[alpha] = robot_rgb[alpha]
    pixel_source = np.zeros(alpha.shape, dtype=np.uint8)
    pixel_source[alpha] = 1
    return composite, {
        "output_index": frame_index,
        "geometry_decision": "UNKNOWN",
        "display_policy": DISPLAY_POLICY,
        "robot_visible_pixels": int(alpha.sum()),
        "robot_link_labels_present": sorted(int(v) for v in np.unique(robot_labels[alpha])),
        "robot_rgb_array_sha256": _array_sha(robot_rgb),
        "robot_labels_array_sha256": _array_sha(robot_labels),
        "robot_depth_array_sha256": _array_sha(depth),
        "pixel_source_array_sha256": _array_sha(pixel_source),
        "composite_array_sha256": _array_sha(composite),
        "pixel_source_enum": {"0": "SYNTHETIC_CLEAN_BACKGROUND", "1": "RENDERED_ROBOT"},
        "clean_role": "OFFLINE_SYNTHETIC_VISUAL_BACKGROUND_ONLY",
        "robot_depth_role": "ROBOT_SELF_OCCLUSION_AND_LAYER_VALIDATION_ONLY",
        "scene_depth_consumed": False,
        "object_mask_consumed_for_occlusion": False,
        "visual_qualified": False,
        "product_qualified": False,
        "training_eligible": False,
        "control_ground_truth": False,
    }


def render_frame(binding: dict[str, Any], frame_index: int, *, assets: Any,
                 cache: dict[Path, object], flange: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """One pure CPU frame; future governed writer owns encoding and final seal."""
    _require(0 <= frame_index < binding["frame_count"], "FRAME_OUT_OF_RANGE")
    clean_ref = binding["clean_rows"][frame_index]["clean"]
    clean = cv2.imread(str(_file_ref(clean_ref, f"clean_{frame_index}")), cv2.IMREAD_COLOR)
    _require(clean is not None, f"CLEAN_DECODE_FAILED:{frame_index}")
    k = binding["intrinsics"][frame_index].astype(np.float64).copy()
    k[0] *= direction.WIDTH / 1280.0
    k[1] *= direction.HEIGHT / 960.0
    states = binding["states"]
    rgb, labels, buffers = direction.raster.render_robot(
        direction.raster, assets, states["q_arm"][frame_index],
        states["q_hand"][frame_index], states["T_tool_hand_root"],
        states["T_camera_base"], k, cache, flange,
        complete_robot=True, return_buffers=True,
        width=direction.WIDTH, height=direction.HEIGHT)
    panel, row = compose_frame(clean, rgb, labels, buffers, frame_index=frame_index)
    row.update({
        "raw_model_input": binding["raw_rows"][frame_index]["model_input"],
        "display_mask_refs": {
            role: binding["mask_rows"][frame_index]["masks"][role]
            for role in ("removal", "protected")},
        "clean": clean_ref,
        "q_state_ref": binding["robot_states_ref"],
        "intrinsics_frame_array_sha256": _array_sha(binding["intrinsics"][frame_index]),
        "q_arm_frame_array_sha256": _array_sha(states["q_arm"][frame_index]),
        "q_hand_frame_array_sha256": _array_sha(states["q_hand"][frame_index]),
        "scene_decode_display_index": binding["raw_rows"][frame_index]["decode_display_index"],
        "hawor_output_index": binding["provenance_rows"][frame_index]["output_index"],
        "motion_source": binding["provenance_rows"][frame_index]["hands"],
    })
    return panel, row


def iter_frames(binding: dict[str, Any]) -> Iterator[tuple[np.ndarray, dict[str, Any]]]:
    """Generate all 171 diagnostic panels; a future governed writer encodes/seals."""
    _require(binding.get("frame_count") == FRAME_COUNT
             and binding.get("fps_nominal") == FPS
             and binding.get("display_policy") == DISPLAY_POLICY,
             "UNBOUND_FRAME_ITERATOR")
    assets = direction.load_pinned_robot_assets(direction.ROOT)
    flange = direction.raster.naturalv2_local_triangles(direction.FLANGE)
    cache: dict[Path, object] = {}
    for frame_index in range(FRAME_COUNT):
        yield render_frame(binding, frame_index, assets=assets, cache=cache, flange=flange)


__all__ = ["CompositeBindingError", "prepare_binding", "validate_frame_rows",
           "compose_frame", "render_frame", "iter_frames", "DISPLAY_POLICY"]
