"""Candidate frontend v3: sealed v2 recipe with exact MP4 acceptance only.

Run this as a separate ``chaoyang run`` operation after a new Task Packet pins
this file, the v2 implementation, and the independent MP4 verifier.  No model
is imported or executed by importing this module.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Any

from chaoyang.governance.common import artifact_ref
from chaoyang.ops import run_exact78_fresh_frontend_v2 as frontend_v2
from chaoyang.pipeline import propainter_mp4_acceptance_v2 as mp4_acceptance


PINNED_V2_SHA256 = "879adb05fbf8702cf9b997ef7ef93c8ff1e231cee161e9a8b4c794bcc0718081"
PINNED_ACCEPTANCE_SHA256 = "381dea1bb14d043a68ff3c4e315a75154d83916e7a9304f7adc38b656d7ea8c4"


def verify_pinned_closure() -> None:
    for module, expected, label in (
        (frontend_v2, PINNED_V2_SHA256, "FRONTEND_V2"),
        (mp4_acceptance, PINNED_ACCEPTANCE_SHA256, "MP4_ACCEPTANCE"),
    ):
        path = Path(module.__file__).resolve(strict=True)
        if artifact_ref(path)["sha256"] != expected:
            raise RuntimeError(f"V3_PINNED_CODE_SHA_DRIFT:{label}")


def verify_internal_mp4_v3(path: Path, frame_count: int, fps: int) -> dict[str, Any]:
    """Adapt stricter MP4 evidence to v2 CLEAN's exact return-field contract."""
    evidence = mp4_acceptance.verify_propainter_mp4(path, frame_count, fps)
    return {
        "artifact": evidence["artifact"],
        "decoded_frames": evidence["decoded_frames"],
        "fps": fps,
        "duration_seconds": float(Fraction(frame_count, fps)),
        "dimensions_wh": [960, 720],
        "pts_mode": "CFR_INDEX_ALIGNED",
    }


def main() -> int:
    """Patch only this process's v2 CLEAN verifier; restore even on failure."""
    verify_pinned_closure()
    original = frontend_v2.verify_internal_mp4
    if (original.__module__ != frontend_v2.__name__
            or original.__name__ != "verify_internal_mp4"):
        raise RuntimeError("V3_V2_VERIFIER_ALREADY_PATCHED")
    try:
        frontend_v2.verify_internal_mp4 = verify_internal_mp4_v3
        return frontend_v2.main()
    finally:
        frontend_v2.verify_internal_mp4 = original


if __name__ == "__main__":
    raise SystemExit(main())
