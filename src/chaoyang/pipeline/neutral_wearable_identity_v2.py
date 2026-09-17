"""Pixel-conservative identity-layer union for archive/legacy/task_cards/32 V2.

This CPU module neither invokes a model nor edits a mask.  A wearable pixel may
enter the result only through one accepted raw model instance.  HaWoR wrist
points are used only to route a raw instance to one physical side.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Literal, Sequence

import numpy as np


Side = Literal["left", "right"]
SIDES: tuple[Side, Side] = ("left", "right")
MIN_MODEL_SCORE = 0.50
WRIST_SUPPORT_RADIUS_OVER_HAND_SCALE = 0.12
PROMPT_CONTRACT_SHA256 = (
    "f85b34cb56041129bafa3020da81c84dab2ed00eb8c21e193ee15506a29a8596"
)
SESSION_ID_PATTERN = re.compile(r"^grap_a_cap_[0-9]{3}$")
FROZEN_PROMPTS = {
    "W1": "a wrist-worn object",
    "W2": "a wearable object around a wrist",
    "W3": "a wrist accessory",
    "W4": "an object worn on a wrist",
}


class WearableIdentityError(RuntimeError):
    pass


@dataclass(frozen=True)
class EvidenceRef:
    path: str
    bytes: int
    sha256: str
    source_kind: str

    def read_verified(self, root: Path) -> bytes:
        base = root.resolve(strict=True)
        target = Path(self.path)
        if not target.is_absolute():
            target = base / target
        target = Path(os.path.abspath(target))
        try:
            relative = target.relative_to(base)
        except ValueError as exc:
            raise WearableIdentityError("evidence escapes root") from exc
        directory_fd = os.open(
            base,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            for component in relative.parts[:-1]:
                next_fd = os.open(
                    component,
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=directory_fd,
                )
                os.close(directory_fd)
                directory_fd = next_fd
            fd = os.open(
                relative.parts[-1],
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_fd,
            )
            try:
                if not stat.S_ISREG(os.fstat(fd).st_mode):
                    raise WearableIdentityError("evidence is not regular")
                chunks = []
                while chunk := os.read(fd, 1024 * 1024):
                    chunks.append(chunk)
            finally:
                os.close(fd)
        except OSError as exc:
            raise WearableIdentityError("nofollow evidence read failed") from exc
        finally:
            os.close(directory_fd)
        payload = b"".join(chunks)
        if (
            len(payload) != self.bytes
            or hashlib.sha256(payload).hexdigest() != self.sha256
        ):
            raise WearableIdentityError("evidence bytes/SHA mismatch")
        return payload


@dataclass(frozen=True)
class VerifiedPromptContract:
    source_ref: EvidenceRef
    prompts: dict[str, str]


def verify_prompt_contract(
    source_ref: EvidenceRef, root: Path
) -> VerifiedPromptContract:
    if source_ref.source_kind != "FROZEN_NEUTRAL_WEARABLE_PROMPT_CONTRACT":
        raise WearableIdentityError("prompt contract source kind mismatch")
    payload = source_ref.read_verified(root)
    if source_ref.sha256 != PROMPT_CONTRACT_SHA256:
        raise WearableIdentityError("prompt contract SHA is not frozen")
    value = json.loads(payload)
    prompts = {row["prompt_id"]: row["text"] for row in value.get("prompts", [])}
    if (
        prompts != FROZEN_PROMPTS
        or value.get("execution_rule")
        != "EXECUTE_AND_PERSIST_ALL_PROMPTS_NO_RESULT_DRIVEN_CHOICE"
    ):
        raise WearableIdentityError("prompt contract semantics drift")
    return VerifiedPromptContract(source_ref, prompts)


@dataclass(frozen=True)
class WristAuthority:
    side: Side
    frame_index: int
    xy: tuple[float, float] | None
    hand_scale_px: float | None
    lineage_id: str
    source_slot: int
    source_track_index: int
    optimized_source_ref: EvidenceRef
    raw_hawor_source_ref: EvidenceRef

    def valid(self, shape: tuple[int, int]) -> bool:
        if (
            self.side not in SIDES
            or not self.lineage_id
            or self.optimized_source_ref.source_kind != "HAWOR_OPTIMIZED_JOINTS_NPZ"
            or self.raw_hawor_source_ref.source_kind != "HAWOR_RAW_PROJECTION_NPZ"
        ):
            return False
        if self.xy is None or self.hand_scale_px is None or self.hand_scale_px <= 0:
            return False
        x, y = self.xy
        height, width = shape
        return bool(
            np.isfinite((x, y, self.hand_scale_px)).all()
            and 0 <= x < width
            and 0 <= y < height
        )


@dataclass(frozen=True)
class RawWearableInstance:
    prompt_id: str
    prompt_text: str
    prompt_contract_ref: EvidenceRef
    frame_index: int
    raw_instance_offset: int
    instance_id: int
    score: float
    mask: np.ndarray
    raw_mask_sha256: str

    def validate(self, shape: tuple[int, int]) -> None:
        if self.mask.dtype != np.bool_ or self.mask.shape != shape:
            raise WearableIdentityError("wearable mask must be unmodified bool HxW")
        if not self.prompt_id or not self.prompt_text:
            raise WearableIdentityError("prompt provenance is incomplete")
        if self.raw_mask_sha256 != mask_sha256(self.mask):
            raise WearableIdentityError("raw wearable mask SHA mismatch")
        if not np.isfinite(self.score):
            raise WearableIdentityError("non-finite model score")


@dataclass(frozen=True)
class WearableAudit:
    side: Side
    prompt_id: str
    raw_instance_offset: int
    instance_id: int
    score: float
    own_wrist_support: float
    other_wrist_support: float
    side_margin: float
    accepted: bool
    rejection_reasons: tuple[str, ...]
    raw_mask_sha256: str


@dataclass(frozen=True)
class SideWearableDecision:
    side: Side
    status: Literal["ACCEPT", "HOLD"]
    selected_offset: int | None
    selected_instance_id: int | None
    selected_raw_mask_sha256: str | None
    audits: tuple[WearableAudit, ...]


@dataclass(frozen=True)
class AcceptedArmRawIdentity:
    """Typed join from a task27 ACCEPT decision to its raw SAM evidence."""

    side: Side
    session_id: str
    frame_id: str
    frame_index: int
    status: str
    selected_offset: int
    selected_instance_id: int
    selected_raw_evidence_ref: EvidenceRef
    selected_raw_evidence_bytes: bytes
    selected_candidate_eligible: bool
    selected_candidate_rejection_reasons: tuple[str, ...]
    raw_mask_sha256: str

    def verify(self, mask: np.ndarray) -> dict[str, object]:
        if self.side not in SIDES or self.status != "ACCEPT":
            raise WearableIdentityError("arm identity is not accepted")
        if (
            SESSION_ID_PATTERN.fullmatch(self.session_id) is None
            or self.session_id == "grap_a_cap_025"
        ):
            raise WearableIdentityError("arm session identity is invalid")
        if (
            type(self.frame_index) is not int
            or type(self.selected_offset) is not int
            or type(self.selected_instance_id) is not int
            or type(self.selected_candidate_eligible) is not bool
            or not self.selected_candidate_eligible
            or self.selected_candidate_rejection_reasons
        ):
            raise WearableIdentityError("arm accepted-decision fields are invalid")
        if self.frame_id != f"{self.session_id}:{self.frame_index}":
            raise WearableIdentityError("arm frame identity mismatch")
        expected_name = (
            f"frame_{self.frame_index:05d}_offset_{self.selected_offset:03d}.json"
        )
        if Path(self.selected_raw_evidence_ref.path).name != expected_name:
            raise WearableIdentityError("arm evidence path/offset mismatch")
        if (
            self.selected_raw_evidence_ref.source_kind
            != "TASK27_SELECTED_ARM_RAW_EVIDENCE_JSON"
        ):
            raise WearableIdentityError("arm evidence source kind mismatch")
        payload = self.selected_raw_evidence_bytes
        if (
            type(payload) is not bytes
            or len(payload) != self.selected_raw_evidence_ref.bytes
            or hashlib.sha256(payload).hexdigest()
            != self.selected_raw_evidence_ref.sha256
        ):
            raise WearableIdentityError("arm evidence verified-byte identity mismatch")
        try:
            evidence = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise WearableIdentityError("arm evidence JSON decode failed") from exc
        if not isinstance(evidence, dict) or set(evidence) != {
            "frame_id",
            "instance_id",
            "mask_sha256",
            "source_kind",
        }:
            raise WearableIdentityError("arm evidence schema mismatch")
        if (
            evidence["source_kind"] != "SAM_RAW_INSTANCE"
            or evidence["frame_id"] != self.frame_id
            or type(evidence["instance_id"]) is not int
            or evidence["instance_id"] != self.selected_instance_id
            or evidence["mask_sha256"] != self.raw_mask_sha256
        ):
            raise WearableIdentityError("arm evidence payload does not join decision")
        if (
            mask.dtype != np.bool_
            or task27_raw_mask_sha256(mask) != self.raw_mask_sha256
        ):
            raise WearableIdentityError("arm raw mask does not join evidence")
        return evidence


def mask_sha256(mask: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(mask, dtype=np.bool_))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def task27_raw_mask_sha256(mask: np.ndarray) -> str:
    """Exact pinned task27 digest: shape/dtype header plus raw bool bytes."""

    array = np.ascontiguousarray(np.asarray(mask, dtype=np.bool_))
    header = f"{array.shape[0]}x{array.shape[1]}:bool:".encode("ascii")
    return hashlib.sha256(header + array.tobytes(order="C")).hexdigest()


def disk_support(
    mask: np.ndarray, xy: tuple[float, float] | None, radius: float | None
) -> float:
    """Return the fraction of an in-image wrist disk covered by a raw mask."""

    if xy is None or radius is None or radius <= 0:
        return 0.0
    x, y = xy
    height, width = mask.shape
    if not np.isfinite((x, y, radius)).all() or not (
        0 <= x < width and 0 <= y < height
    ):
        return 0.0
    x0, x1 = max(0, int(np.floor(x - radius))), min(width, int(np.ceil(x + radius + 1)))
    y0, y1 = (
        max(0, int(np.floor(y - radius))),
        min(height, int(np.ceil(y + radius + 1))),
    )
    yy, xx = np.ogrid[y0:y1, x0:x1]
    disk = (xx - x) ** 2 + (yy - y) ** 2 <= radius**2
    return float(mask[y0:y1, x0:x1][disk].mean()) if disk.any() else 0.0


def select_wearable_per_side(
    instances: Sequence[RawWearableInstance],
    authorities: dict[Side, WristAuthority],
    shape: tuple[int, int],
    *,
    prompt_contract: VerifiedPromptContract,
) -> dict[Side, SideWearableDecision]:
    if set(authorities) != set(SIDES):
        raise WearableIdentityError("both independent side authorities are required")
    for side in SIDES:
        if authorities[side].side != side:
            raise WearableIdentityError("side authority alias")
    left, right = authorities["left"], authorities["right"]
    if left.frame_index != right.frame_index:
        raise WearableIdentityError("left/right authority frame mismatch")
    if (
        left.lineage_id == right.lineage_id
        or left.source_slot == right.source_slot
        or (left.source_track_index, left.source_slot)
        == (right.source_track_index, right.source_slot)
    ):
        raise WearableIdentityError("left/right authority provenance alias")
    if (
        left.optimized_source_ref != right.optimized_source_ref
        or left.raw_hawor_source_ref != right.raw_hawor_source_ref
    ):
        raise WearableIdentityError(
            "left/right authorities must join the same frozen HaWoR sources"
        )
    if prompt_contract.prompts != FROZEN_PROMPTS:
        raise WearableIdentityError("unverified prompt contract")
    prompt_ids = {instance.prompt_id for instance in instances}
    if len(prompt_ids) > 1:
        raise WearableIdentityError("cross-prompt candidate-pool mixing forbidden")
    for instance in instances:
        instance.validate(shape)
        if (
            instance.prompt_contract_ref != prompt_contract.source_ref
            or prompt_contract.prompts.get(instance.prompt_id) != instance.prompt_text
        ):
            raise WearableIdentityError(
                "raw instance does not join frozen prompt contract"
            )

    support: dict[tuple[int, Side], float] = {}
    for index, instance in enumerate(instances):
        for side in SIDES:
            authority = authorities[side]
            radius = None
            if authority.valid(shape):
                radius = WRIST_SUPPORT_RADIUS_OVER_HAND_SCALE * float(
                    authority.hand_scale_px
                )
            support[(index, side)] = disk_support(instance.mask, authority.xy, radius)

    result: dict[Side, SideWearableDecision] = {}
    for side in SIDES:
        other: Side = "right" if side == "left" else "left"
        audits: list[WearableAudit] = []
        eligible: list[tuple[float, int, int]] = []
        for index, instance in enumerate(instances):
            own = support[(index, side)]
            opposite = support[(index, other)]
            margin = own - opposite
            reasons: list[str] = []
            if not authorities[side].valid(shape):
                reasons.append("SIDE_AUTHORITY_UNAVAILABLE")
            if instance.score < MIN_MODEL_SCORE:
                reasons.append("MODEL_SCORE_BELOW_FROZEN_THRESHOLD")
            if own <= 0.0:
                reasons.append("OWN_WRIST_NOT_SUPPORTED")
            if margin <= 0.0:
                reasons.append("NOT_UNIQUELY_ROUTED_TO_SIDE")
            accepted = not reasons
            if accepted:
                eligible.append((-instance.score, -margin, index))
            audits.append(
                WearableAudit(
                    side,
                    instance.prompt_id,
                    instance.raw_instance_offset,
                    instance.instance_id,
                    float(instance.score),
                    own,
                    opposite,
                    margin,
                    accepted,
                    tuple(reasons),
                    instance.raw_mask_sha256,
                )
            )
        if not eligible:
            result[side] = SideWearableDecision(
                side, "HOLD", None, None, None, tuple(audits)
            )
        else:
            selected = instances[min(eligible)[2]]
            result[side] = SideWearableDecision(
                side,
                "ACCEPT",
                selected.raw_instance_offset,
                selected.instance_id,
                selected.raw_mask_sha256,
                tuple(audits),
            )
    return result


def identity_union(
    arm_raw_mask: np.ndarray,
    wearable_raw_mask: np.ndarray,
    *,
    side: Side,
    arm_identity: AcceptedArmRawIdentity,
    wearable_decision: SideWearableDecision,
    wearable_instance: RawWearableInstance,
) -> np.ndarray:
    """Union two accepted raw identities without manufacturing pixels."""

    if side not in SIDES or arm_identity.side != side:
        raise WearableIdentityError("arm identity is not accepted for this side")
    arm_identity.verify(arm_raw_mask)
    if wearable_decision.side != side or wearable_decision.status != "ACCEPT":
        raise WearableIdentityError("wearable decision is not accepted for this side")
    if (
        wearable_decision.selected_offset != wearable_instance.raw_instance_offset
        or wearable_decision.selected_instance_id != wearable_instance.instance_id
        or wearable_decision.selected_raw_mask_sha256
        != wearable_instance.raw_mask_sha256
    ):
        raise WearableIdentityError("wearable instance does not join accepted decision")
    if arm_raw_mask.dtype != np.bool_ or wearable_raw_mask.dtype != np.bool_:
        raise WearableIdentityError("identity masks must be raw bool arrays")
    if arm_raw_mask.shape != wearable_raw_mask.shape:
        raise WearableIdentityError("identity mask shape mismatch")
    if mask_sha256(wearable_raw_mask) != wearable_instance.raw_mask_sha256:
        raise WearableIdentityError("wearable provenance SHA mismatch")
    union = np.logical_or(arm_raw_mask, wearable_raw_mask)
    if np.logical_and(
        union, np.logical_not(np.logical_or(arm_raw_mask, wearable_raw_mask))
    ).any():
        raise WearableIdentityError("identity union manufactured pixels")
    return union
