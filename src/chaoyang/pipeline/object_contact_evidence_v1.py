"""Fail-closed V7.1 Object/Contact evidence DAG and geometry fixtures.

The graph records provenance direction.  It intentionally does not estimate
poses or contact: callers must provide immutable evidence records, and this
module only rejects cycles, illegal ancestry, capability escalation, and
task-specific geometry contract violations.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


class EvidenceGraphError(ValueError):
    """Raised when evidence provenance or a fixture fails closed."""


class EvidenceType(str, Enum):
    DIRECT_OBJECT6D = "DIRECT_OBJECT6D"
    BIDIRECTIONAL_TRACKED = "BIDIRECTIONAL_TRACKED"
    CONTACT_SEED = "CONTACT_SEED"
    HAND_OBJECT_ATTACHMENT = "HAND_OBJECT_ATTACHMENT"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class EvidenceNode:
    evidence_id: str
    evidence_type: EvidenceType
    parent_evidence_ids: tuple[str, ...]
    evidence_depth: int
    may_support_contact_authority: bool
    may_support_object6d_authority: bool
    may_support_gold_contact_accuracy: bool
    may_upgrade_tactile_supported_contact: bool


@dataclass(frozen=True)
class EvidenceGraph:
    nodes_by_id: Mapping[str, EvidenceNode]
    topological_order: tuple[str, ...]


# Exact, fail-closed capability policy for the evidence kinds in this version.
# Gold accuracy needs independent frozen human labels, and tactile-supported
# contact needs direct tactile evidence; neither exists among these node kinds.
_CAPABILITIES: Mapping[EvidenceType, tuple[bool, bool, bool, bool]] = {
    EvidenceType.DIRECT_OBJECT6D: (True, True, False, False),
    EvidenceType.BIDIRECTIONAL_TRACKED: (True, True, False, False),
    EvidenceType.CONTACT_SEED: (True, False, False, False),
    EvidenceType.HAND_OBJECT_ATTACHMENT: (False, False, False, False),
    EvidenceType.UNKNOWN: (False, False, False, False),
}

_PARENT_TYPES: Mapping[EvidenceType, frozenset[EvidenceType]] = {
    EvidenceType.DIRECT_OBJECT6D: frozenset(),
    # A tracked pose may be represented as a root when the tracker owns its
    # own immutable source receipt, or may retain the two bounding direct
    # observations as parents.  Both forms remain upstream of CONTACT_SEED.
    EvidenceType.BIDIRECTIONAL_TRACKED: frozenset({EvidenceType.DIRECT_OBJECT6D}),
    EvidenceType.CONTACT_SEED: frozenset(
        {EvidenceType.DIRECT_OBJECT6D, EvidenceType.BIDIRECTIONAL_TRACKED}
    ),
    EvidenceType.HAND_OBJECT_ATTACHMENT: frozenset({EvidenceType.CONTACT_SEED}),
    EvidenceType.UNKNOWN: frozenset(),
}


def _require_bool(record: Mapping[str, Any], key: str) -> bool:
    value = record.get(key)
    if type(value) is not bool:
        raise EvidenceGraphError(f"{key} must be boolean")
    return value


def _parse_node(record: Mapping[str, Any]) -> EvidenceNode:
    evidence_id = record.get("evidence_id")
    if not isinstance(evidence_id, str) or not evidence_id:
        raise EvidenceGraphError("evidence_id must be a non-empty string")
    try:
        evidence_type = EvidenceType(record.get("evidence_type"))
    except (TypeError, ValueError) as error:
        raise EvidenceGraphError(f"{evidence_id}: unknown evidence_type") from error
    raw_parents = record.get("parent_evidence_ids")
    if not isinstance(raw_parents, list) or not all(
        isinstance(parent, str) and parent for parent in raw_parents
    ):
        raise EvidenceGraphError(f"{evidence_id}: parent_evidence_ids must be string array")
    parents = tuple(raw_parents)
    if len(parents) != len(set(parents)):
        raise EvidenceGraphError(f"{evidence_id}: duplicate parent evidence")
    depth = record.get("evidence_depth")
    if type(depth) is not int or depth < 0:
        raise EvidenceGraphError(f"{evidence_id}: evidence_depth must be non-negative integer")
    return EvidenceNode(
        evidence_id=evidence_id,
        evidence_type=evidence_type,
        parent_evidence_ids=parents,
        evidence_depth=depth,
        may_support_contact_authority=_require_bool(record, "may_support_contact_authority"),
        may_support_object6d_authority=_require_bool(record, "may_support_object6d_authority"),
        may_support_gold_contact_accuracy=_require_bool(
            record, "may_support_gold_contact_accuracy"
        ),
        may_upgrade_tactile_supported_contact=_require_bool(
            record, "may_upgrade_tactile_supported_contact"
        ),
    )


def validate_evidence_dag(records: Iterable[Mapping[str, Any]]) -> EvidenceGraph:
    """Validate an evidence DAG and return a deterministic topological order.

    ``HAND_OBJECT_ATTACHMENT`` can only descend from a ``CONTACT_SEED`` and is
    never authoritative.  Consequently an attachment cannot become an
    ancestor of a seed and cannot self-prove contact, Object6D, gold accuracy,
    or tactile support.
    """

    nodes: dict[str, EvidenceNode] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise EvidenceGraphError("each evidence node must be an object")
        node = _parse_node(record)
        if node.evidence_id in nodes:
            raise EvidenceGraphError(f"duplicate evidence_id: {node.evidence_id}")
        nodes[node.evidence_id] = node
    if not nodes:
        raise EvidenceGraphError("evidence graph must not be empty")

    for node in nodes.values():
        for parent in node.parent_evidence_ids:
            if parent not in nodes:
                raise EvidenceGraphError(f"{node.evidence_id}: missing parent {parent}")
            if parent == node.evidence_id:
                raise EvidenceGraphError(f"cycle detected at {node.evidence_id}")

    visiting: set[str] = set()
    visited: set[str] = set()
    order: list[str] = []

    def visit(evidence_id: str) -> None:
        if evidence_id in visiting:
            raise EvidenceGraphError(f"cycle detected at {evidence_id}")
        if evidence_id in visited:
            return
        visiting.add(evidence_id)
        for parent in sorted(nodes[evidence_id].parent_evidence_ids):
            visit(parent)
        visiting.remove(evidence_id)
        visited.add(evidence_id)
        order.append(evidence_id)

    for evidence_id in sorted(nodes):
        visit(evidence_id)

    for evidence_id in order:
        node = nodes[evidence_id]
        allowed = _PARENT_TYPES[node.evidence_type]
        parent_types = {nodes[parent].evidence_type for parent in node.parent_evidence_ids}
        if node.evidence_type in {
            EvidenceType.CONTACT_SEED,
            EvidenceType.HAND_OBJECT_ATTACHMENT,
        } and not node.parent_evidence_ids:
            raise EvidenceGraphError(f"{evidence_id}: {node.evidence_type.value} needs a parent")
        if not parent_types.issubset(allowed):
            found = ",".join(sorted(kind.value for kind in parent_types))
            raise EvidenceGraphError(
                f"{evidence_id}: illegal parent type(s) {found} for {node.evidence_type.value}"
            )
        expected_depth = (
            0
            if not node.parent_evidence_ids
            else 1 + max(nodes[parent].evidence_depth for parent in node.parent_evidence_ids)
        )
        if node.evidence_depth != expected_depth:
            raise EvidenceGraphError(
                f"{evidence_id}: evidence_depth {node.evidence_depth} != {expected_depth}"
            )
        observed_capabilities = (
            node.may_support_contact_authority,
            node.may_support_object6d_authority,
            node.may_support_gold_contact_accuracy,
            node.may_upgrade_tactile_supported_contact,
        )
        if observed_capabilities != _CAPABILITIES[node.evidence_type]:
            raise EvidenceGraphError(
                f"{evidence_id}: capability escalation for {node.evidence_type.value}"
            )

    return EvidenceGraph(nodes_by_id=nodes, topological_order=tuple(order))


class CardRegion(str, Enum):
    FRONT = "FRONT"
    BACK = "BACK"
    EDGE = "EDGE"
    PENETRATION = "PENETRATION"
    OUTSIDE = "OUTSIDE"


def classify_poker_card_local_point(
    local_point_m: Sequence[float], dimensions_m: Sequence[float], *, tolerance_m: float = 1e-9
) -> CardRegion:
    """Classify a local point against a thin, two-sided rigid card box."""

    point = np.asarray(local_point_m, dtype=np.float64)
    dimensions = np.asarray(dimensions_m, dtype=np.float64)
    tolerance = float(tolerance_m)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise EvidenceGraphError("card point must be finite (3,)")
    if dimensions.shape != (3,) or not np.isfinite(dimensions).all() or np.any(dimensions <= 0):
        raise EvidenceGraphError("card dimensions must be positive finite (3,)")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise EvidenceGraphError("tolerance_m must be finite and non-negative")
    width, height, thickness = dimensions.tolist()
    if thickness >= min(width, height) * 0.05:
        raise EvidenceGraphError("Poker card must satisfy the thin-card contract")
    half = dimensions * 0.5
    absolute = np.abs(point)
    if np.any(absolute > half + tolerance):
        return CardRegion.OUTSIDE
    if np.all(absolute < half - tolerance):
        return CardRegion.PENETRATION
    if abs(absolute[2] - half[2]) <= tolerance:
        return CardRegion.FRONT if point[2] >= 0 else CardRegion.BACK
    if abs(absolute[0] - half[0]) <= tolerance or abs(absolute[1] - half[1]) <= tolerance:
        return CardRegion.EDGE
    return CardRegion.PENETRATION


def validate_poker_thin_card_fixture(fixture: Mapping[str, Any]) -> tuple[CardRegion, ...]:
    """Validate deterministic front/back/edge/penetration Poker probes."""

    if fixture.get("task") != "poker" or fixture.get("rigid") is not True:
        raise EvidenceGraphError("Poker fixture must describe one rigid card")
    if fixture.get("two_sided") is not True:
        raise EvidenceGraphError("Poker card must be two-sided")
    if set(fixture.get("surface_ids", [])) != {"FRONT", "BACK", "EDGE"}:
        raise EvidenceGraphError("Poker surfaces must distinguish FRONT, BACK and EDGE")
    dimensions = fixture.get("dimensions_m")
    probes = fixture.get("probes")
    if not isinstance(probes, list) or not probes:
        raise EvidenceGraphError("Poker fixture probes are required")
    observed: list[CardRegion] = []
    for probe in probes:
        try:
            expected = CardRegion(probe["expected_region"])
            point = probe["local_point_m"]
        except (KeyError, TypeError, ValueError) as error:
            raise EvidenceGraphError("malformed Poker probe") from error
        actual = classify_poker_card_local_point(point, dimensions)
        if actual is not expected:
            raise EvidenceGraphError(
                f"Poker probe {probe.get('probe_id', '<unknown>')} expected {expected.value}, got {actual.value}"
            )
        observed.append(actual)
    required = {CardRegion.FRONT, CardRegion.BACK, CardRegion.EDGE, CardRegion.PENETRATION}
    if not required.issubset(observed):
        raise EvidenceGraphError("Poker fixture must cover front, back, edge and penetration")
    return tuple(observed)


def validate_chips_three_instance_fixture(fixture: Mapping[str, Any]) -> tuple[str, ...]:
    """Validate independent Chips instances and deformation-to-UNKNOWN routing."""

    if fixture.get("task") != "chips":
        raise EvidenceGraphError("Chips fixture task mismatch")
    instance_ids = fixture.get("instance_ids")
    if (
        not isinstance(instance_ids, list)
        or len(instance_ids) != 3
        or len(set(instance_ids)) != 3
        or not all(isinstance(item, str) and item for item in instance_ids)
    ):
        raise EvidenceGraphError("Chips requires exactly three independent instance ids")
    frames = fixture.get("frames")
    if not isinstance(frames, list) or not frames:
        raise EvidenceGraphError("Chips fixture frames are required")
    saw_deformation = False
    expected_ids = set(instance_ids)
    for frame in frames:
        instances = frame.get("instances") if isinstance(frame, Mapping) else None
        if not isinstance(instances, list):
            raise EvidenceGraphError("Chips frame instances must be an array")
        row_ids = [item.get("instance_id") for item in instances if isinstance(item, Mapping)]
        if len(row_ids) != 3 or set(row_ids) != expected_ids:
            raise EvidenceGraphError("each Chips frame must preserve all three independent ids")
        for item in instances:
            try:
                evidence_type = EvidenceType(item["evidence_type"])
                deformed = item["visibly_deformed"]
            except (KeyError, TypeError, ValueError) as error:
                raise EvidenceGraphError("malformed Chips instance row") from error
            if type(deformed) is not bool:
                raise EvidenceGraphError("visibly_deformed must be boolean")
            if deformed:
                saw_deformation = True
                if evidence_type is not EvidenceType.UNKNOWN:
                    raise EvidenceGraphError("visibly deformed Chips instance must degrade to UNKNOWN")
            elif evidence_type not in {
                EvidenceType.DIRECT_OBJECT6D,
                EvidenceType.BIDIRECTIONAL_TRACKED,
                EvidenceType.UNKNOWN,
            }:
                raise EvidenceGraphError("Chips pose row uses a non-pose evidence type")
    if not saw_deformation:
        raise EvidenceGraphError("Chips fixture must exercise deformation degradation")
    return tuple(instance_ids)
