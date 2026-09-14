"""Fail-closed CPU contract helpers for the SAM2.1 Route-B adapter.

This module does not open labels, build a model, run a producer, or create MASK
pixels.  It fixes the development-only folds, H/O/U/B loss semantics, decoder-
only parameter boundary, and candidate governance before GPU work begins.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


class RouteBContractError(RuntimeError):
    """Raised when a Route-B operation would cross a frozen boundary."""


MODEL_ID = "SAM2_1_HIERA_LARGE"
DECODER_PREFIX = "sam_mask_decoder."
EXPECTED_DECODER_PARAMETERS = 4_215_109
DEVELOPMENT_FRAMES = tuple(range(228, 243))
HELD_OUT_FOLDS = {
    "F1": (228, 231, 234, 237, 240),
    "F2": (229, 232, 235, 238, 241),
    "F3": (230, 233, 236, 239, 242),
}
FORBIDDEN_OVERRIDE_KEYS = frozenset(
    {
        "session_overrides",
        "frame_overrides",
        "session_id_thresholds",
        "per_frame_thresholds",
        "legacy_fallback",
    }
)


@dataclass(frozen=True)
class FoldSpec:
    fold_id: str
    train_frames: tuple[int, ...]
    held_out_frames: tuple[int, ...]


@dataclass(frozen=True)
class AsymmetricHOUBSupervision:
    """Disjoint evidence for the asymmetric U-band loss.

    U is deliberately neither a zero-weight pixel nor a hard H target.  Its
    only training term is a false-negative hinge at the binary decision
    boundary, implemented by :func:`asymmetric_houb_loss`.
    """

    human_positive: np.ndarray
    object_or_background_negative: np.ndarray
    uncertain_false_negative_only: np.ndarray


U_SEMANTICS = "ASYMMETRIC_FALSE_NEGATIVE_ONLY"
FORBIDDEN_U_SEMANTICS = frozenset({"ZERO_WEIGHT", "MERGE_U_INTO_H_HARD_POSITIVE"})


def fixed_oof_plan(frames: Sequence[int] = DEVELOPMENT_FRAMES) -> tuple[FoldSpec, ...]:
    """Return the immutable time-spread train-10/held-out-5 plan."""
    observed = tuple(int(x) for x in frames)
    if observed != DEVELOPMENT_FRAMES:
        raise RouteBContractError(
            f"development frame identity drift: {observed!r}"
        )
    universe = set(DEVELOPMENT_FRAMES)
    plan = []
    held_union: set[int] = set()
    for fold_id, held in HELD_OUT_FOLDS.items():
        if len(held) != 5 or not set(held) <= universe:
            raise RouteBContractError(f"invalid frozen fold {fold_id}")
        held_union.update(held)
        train = tuple(frame for frame in DEVELOPMENT_FRAMES if frame not in held)
        if len(train) != 10 or set(train) & set(held):
            raise RouteBContractError(f"train/held-out leakage in {fold_id}")
        plan.append(FoldSpec(fold_id, train, held))
    if held_union != universe or sum(len(x) for x in HELD_OUT_FOLDS.values()) != 15:
        raise RouteBContractError("held-out folds do not partition development")
    return tuple(plan)


def require_development_split(split: str) -> None:
    """Reject every supervised read except the public development split."""
    if split != "development":
        raise RouteBContractError(f"supervised split is sealed: {split!r}")


def make_supervision(
    human: np.ndarray,
    object_: np.ndarray,
    uncertain: np.ndarray,
    background: np.ndarray,
) -> AsymmetricHOUBSupervision:
    """Map one-hot H/O/U/B evidence without erasing or hard-labeling U.

    H is a standard positive, O+B are standard negatives, and U is retained
    as its own directional evidence.  A static per-pixel ``loss_weight`` is
    intentionally not returned because ``weight=0`` would hide contact-side
    false negatives while merging U into H would make it a hard positive.
    """
    arrays = [np.asarray(x, dtype=bool) for x in (human, object_, uncertain, background)]
    if not arrays or any(x.shape != arrays[0].shape for x in arrays):
        raise RouteBContractError("H/O/U/B shape drift")
    membership = sum(x.astype(np.uint8) for x in arrays)
    if np.any(membership != 1):
        raise RouteBContractError("H/O/U/B must be exhaustive and mutually exclusive")
    return AsymmetricHOUBSupervision(
        arrays[0].copy(),
        np.logical_or(arrays[1], arrays[3]),
        arrays[2].copy(),
    )


def validate_u_semantics(policy: str) -> None:
    """Require the project-owner-authorized directional U policy."""

    if policy in FORBIDDEN_U_SEMANTICS:
        raise RouteBContractError(f"forbidden U-band supervision: {policy}")
    if policy != U_SEMANTICS:
        raise RouteBContractError(f"unknown U-band supervision: {policy}")


def asymmetric_houb_loss(logits: Any, supervision: AsymmetricHOUBSupervision) -> Any:
    """Return mean binary loss with a one-sided U false-negative term.

    For U pixels, ``relu(-logit)`` is exactly zero once the prediction is on
    the H side of the decision boundary and has non-zero corrective gradient
    while it remains on the background side.  This is neither zero-weight U
    nor the ordinary positive BCE used for hard H labels.
    """

    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:  # pragma: no cover - Route B requires torch
        raise RouteBContractError("torch is required for Route-B loss") from exc

    if not isinstance(logits, torch.Tensor):
        raise RouteBContractError("logits must be a torch.Tensor")
    shapes = {
        tuple(supervision.human_positive.shape),
        tuple(supervision.object_or_background_negative.shape),
        tuple(supervision.uncertain_false_negative_only.shape),
    }
    if len(shapes) != 1 or tuple(logits.shape) not in shapes:
        raise RouteBContractError("logit/supervision shape drift")
    device = logits.device
    dtype = logits.dtype
    human = torch.as_tensor(supervision.human_positive, device=device, dtype=dtype)
    negative = torch.as_tensor(
        supervision.object_or_background_negative, device=device, dtype=dtype
    )
    uncertain = torch.as_tensor(
        supervision.uncertain_false_negative_only, device=device, dtype=dtype
    )
    per_pixel = (
        human * functional.softplus(-logits)
        + negative * functional.softplus(logits)
        + uncertain * functional.relu(-logits)
    )
    return per_pixel.mean()


def freeze_except_decoder(
    named_parameters: Iterable[tuple[str, Any]],
) -> tuple[str, ...]:
    """Freeze every parameter except the exact SAM2.1 mask decoder."""
    items = list(named_parameters)
    decoder_names = []
    decoder_numel = 0
    for name, parameter in items:
        allowed = name.startswith(DECODER_PREFIX)
        parameter.requires_grad_(allowed)
        if allowed:
            decoder_names.append(name)
            decoder_numel += int(parameter.numel())
    if decoder_numel != EXPECTED_DECODER_PARAMETERS:
        raise RouteBContractError(
            f"decoder parameter count drift: {decoder_numel} != "
            f"{EXPECTED_DECODER_PARAMETERS}"
        )
    if not decoder_names:
        raise RouteBContractError("decoder parameter prefix matched nothing")
    return tuple(decoder_names)


def verify_gradient_boundary(named_parameters: Iterable[tuple[str, Any]]) -> None:
    """Prove after backward that no frozen parameter received a gradient."""
    items = list(named_parameters)
    trainable_numel = sum(int(p.numel()) for n, p in items if p.requires_grad)
    if trainable_numel != EXPECTED_DECODER_PARAMETERS:
        raise RouteBContractError(f"trainable parameter drift: {trainable_numel}")
    offenders = [
        name
        for name, parameter in items
        if not name.startswith(DECODER_PREFIX)
        and getattr(parameter, "grad", None) is not None
    ]
    if offenders:
        raise RouteBContractError(f"gradient escaped decoder: {offenders[:5]}")
    decoder_grad = [
        name
        for name, parameter in items
        if name.startswith(DECODER_PREFIX)
        and getattr(parameter, "grad", None) is not None
    ]
    if not decoder_grad:
        raise RouteBContractError("no decoder parameter received a gradient")


def tensor_sha256(tensor: Any) -> str:
    """Hash one dense parameter tensor in a device-independent representation."""
    value = tensor.detach().cpu().contiguous().numpy()
    header = f"{value.dtype.str}|{value.shape}|".encode("ascii")
    return hashlib.sha256(header + value.tobytes(order="C")).hexdigest()


def snapshot_parameter_hashes(
    named_parameters: Iterable[tuple[str, Any]],
) -> dict[str, str]:
    items = list(named_parameters)
    if len({name for name, _ in items}) != len(items):
        raise RouteBContractError("duplicate parameter name")
    return {name: tensor_sha256(parameter) for name, parameter in items}


def verify_weight_delta_boundary(
    before: Mapping[str, str],
    after: Mapping[str, str],
) -> tuple[str, ...]:
    """Require at least one decoder change and byte-identical frozen weights."""
    if set(before) != set(after):
        raise RouteBContractError("parameter key set changed")
    changed = tuple(sorted(name for name in before if before[name] != after[name]))
    if not changed:
        raise RouteBContractError("training changed no parameter")
    offenders = [name for name in changed if not name.startswith(DECODER_PREFIX)]
    if offenders:
        raise RouteBContractError(f"frozen weight changed: {offenders[:5]}")
    return changed


def validate_candidate_governance(record: Mapping[str, Any]) -> None:
    """Validate the non-advancing T1 candidate state before writing review media."""
    expected = {
        "AUTH_TIER": "T1",
        "candidate_requires_human_review": True,
        "next_bucket_blocked": True,
        "advancement_authorized": False,
        "formal_consumer_allowed": False,
        "labels_read_splits": ["development"],
        "u_semantics": U_SEMANTICS,
        "u_metric_variants_required": ["u_as_h", "u_excluded"],
        "u_gap_review_flag_required": True,
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise RouteBContractError(f"candidate governance drift: {key}")
    forbidden = sorted(FORBIDDEN_OVERRIDE_KEYS & set(record))
    if forbidden:
        raise RouteBContractError(f"forbidden override keys: {forbidden}")


fixed_oof_plan()
