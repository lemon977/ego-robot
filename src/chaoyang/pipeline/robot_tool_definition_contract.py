"""One fail-closed Tianji tool-definition source for visual robot candidates.

This module does not decide whether the Tianji URDF or the external teleoperation
MJCF is physically more accurate.  It binds every in-repository visual consumer
to the same pinned bytes.  A future full-chain renderer, IK solver, and visual
mount fitter must all persist :meth:`PinnedToolDefinition.as_manifest` and the
records must be byte-identical before their outputs may be combined.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


TOOL_URDF_RELATIVE = Path(
    "assets/robot/tianji/marvin_description/urdf/marvin_CCS_m6.urdf"
)
ASSET_PIN_RELATIVE = Path("assets/robot/ROBOT_ASSET_PIN.json")


class ToolDefinitionError(RuntimeError):
    """The tool-definition bytes, pin, or two-sided joint contract disagree."""


def read_regular_bytes(path: Path) -> bytes:
    """Read and hash the exact bytes from one no-follow file descriptor."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ToolDefinitionError(f"cannot securely open {path}: {exc}") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ToolDefinitionError(f"not a regular file: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class ToolJointDefinition:
    side: str
    name: str
    joint_type: str
    parent: str
    child: str
    xyz_m: tuple[float, float, float]
    rpy_rad: tuple[float, float, float]


@dataclass(frozen=True)
class PinnedToolDefinition:
    asset_pin_path: str
    asset_pin_bytes: int
    asset_pin_sha256: str
    urdf_path: str
    urdf_bytes: int
    urdf_sha256: str
    left: ToolJointDefinition
    right: ToolJointDefinition

    def as_manifest(self) -> dict[str, Any]:
        """Return the exact record every renderer/IK/fitter manifest must copy."""

        return {
            "schema_version": "tianji-tool-definition-source-v1",
            "source_policy": "ONE_PINNED_URDF_FOR_RENDERER_IK_AND_VISUAL_FIT",
            "asset_pin": {
                "path": self.asset_pin_path,
                "bytes": self.asset_pin_bytes,
                "sha256": self.asset_pin_sha256,
            },
            "urdf": {
                "path": self.urdf_path,
                "bytes": self.urdf_bytes,
                "sha256": self.urdf_sha256,
            },
            "tool_joints": {
                "left": asdict(self.left),
                "right": asdict(self.right),
            },
            "external_mjcf_consumed": False,
            "world_translation_interpretation_forbidden": True,
            "axis_note": (
                "Each xyz is expressed in its parent flange frame; the 95 mm "
                "URDF/MJCF difference follows the rotating tool/flange chain and "
                "must not be treated as a constant world-frame translation."
            ),
        }


def _three_floats(value: str | None, *, field: str) -> tuple[float, float, float]:
    try:
        parsed = tuple(float(item) for item in str(value or "").split())
    except ValueError as exc:
        raise ToolDefinitionError(f"invalid {field}") from exc
    if len(parsed) != 3:
        raise ToolDefinitionError(f"{field} must contain three floats")
    return parsed  # type: ignore[return-value]


def _tool_joint(root: ET.Element, side: str) -> ToolJointDefinition:
    name = f"{side}_tool_joint"
    matches = [joint for joint in root.findall("joint") if joint.get("name") == name]
    if len(matches) != 1:
        raise ToolDefinitionError(f"expected exactly one {name}")
    joint = matches[0]
    parent, child, origin = joint.find("parent"), joint.find("child"), joint.find("origin")
    if parent is None or child is None or origin is None:
        raise ToolDefinitionError(f"incomplete {name}")
    definition = ToolJointDefinition(
        side=side,
        name=name,
        joint_type=str(joint.get("type", "")),
        parent=str(parent.get("link", "")),
        child=str(child.get("link", "")),
        xyz_m=_three_floats(origin.get("xyz"), field=f"{name}.xyz"),
        rpy_rad=_three_floats(origin.get("rpy"), field=f"{name}.rpy"),
    )
    if definition.joint_type != "fixed":
        raise ToolDefinitionError(f"{name} is not fixed")
    if definition.parent != ("flange_L" if side == "left" else "flange_R"):
        raise ToolDefinitionError(f"unexpected parent for {name}: {definition.parent}")
    if definition.child != f"{side}_tool":
        raise ToolDefinitionError(f"unexpected child for {name}: {definition.child}")
    return definition


def load_pinned_tool_definition(project_root: Path) -> PinnedToolDefinition:
    """Load the pin and consumed URDF from immutable bytes and cross-check them."""

    root = project_root.resolve()
    pin_path = root / ASSET_PIN_RELATIVE
    urdf_path = root / TOOL_URDF_RELATIVE
    pin_bytes = read_regular_bytes(pin_path)
    urdf_bytes = read_regular_bytes(urdf_path)
    pin_sha = hashlib.sha256(pin_bytes).hexdigest()
    urdf_sha = hashlib.sha256(urdf_bytes).hexdigest()
    try:
        pin = json.loads(pin_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ToolDefinitionError("invalid robot asset pin JSON") from exc
    if pin.get("status") != "T0_IDENTITY_PIN_NOT_EXECUTION_AUTHORIZATION":
        raise ToolDefinitionError("unexpected robot asset pin status")
    matches = [row for row in pin.get("files", []) if row.get("path") == TOOL_URDF_RELATIVE.as_posix()]
    if len(matches) != 1:
        raise ToolDefinitionError("Tianji URDF does not have one asset-pin entry")
    pinned = matches[0]
    if pinned.get("bytes") != len(urdf_bytes) or pinned.get("sha256") != urdf_sha:
        raise ToolDefinitionError("Tianji URDF bytes do not match the robot asset pin")
    try:
        xml_root = ET.fromstring(urdf_bytes)
    except ET.ParseError as exc:
        raise ToolDefinitionError("invalid Tianji URDF XML") from exc
    left, right = _tool_joint(xml_root, "left"), _tool_joint(xml_root, "right")
    if left.xyz_m != right.xyz_m:
        raise ToolDefinitionError("left/right tool translations disagree in the consumed URDF")
    return PinnedToolDefinition(
        asset_pin_path=ASSET_PIN_RELATIVE.as_posix(),
        asset_pin_bytes=len(pin_bytes),
        asset_pin_sha256=pin_sha,
        urdf_path=TOOL_URDF_RELATIVE.as_posix(),
        urdf_bytes=len(urdf_bytes),
        urdf_sha256=urdf_sha,
        left=left,
        right=right,
    )


def require_same_tool_definition(*records: dict[str, Any]) -> None:
    """Fail closed if full-chain renderer, IK, and fit did not bind equal bytes."""

    if len(records) != 3:
        raise ToolDefinitionError("renderer, IK, and visual-fit records are all required")
    canonical = json.dumps(records[0], sort_keys=True, separators=(",", ":"))
    if any(json.dumps(record, sort_keys=True, separators=(",", ":")) != canonical for record in records[1:]):
        raise ToolDefinitionError("renderer/IK/visual-fit tool-definition records disagree")
