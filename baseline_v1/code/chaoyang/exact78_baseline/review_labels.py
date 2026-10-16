"""Review-only source labels; no motion, image or solver imports."""
from typing import Any


def motion_review_title(session: Any, frame: int, count: int) -> str:
    source = (getattr(session, "calibration", None) or {}).get("actual_motion_source")
    if source is None or source == "PICO_OPENXR26_SAME_SESSION":
        name = "PICO"  # Exact legacy tracker title is preserved.
    elif source == "PICO_CONTROLLER_PLUS_MANUS_INSTALLATION_PRIOR":
        name = "CONTROLLER + MANUS PRIOR"
    elif source in {"HAWOR_RGB_PREDICTION", "HAWOR"}:
        name = "HAWOR PREDICTION"
    else:
        name = "UNKNOWN MOTION SOURCE"
    return f"RAW + {name} {frame}/{count}"
