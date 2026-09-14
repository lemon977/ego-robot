from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audit_poker042_local_stereo_geometry import output_decision


def test_geometry_validate_pass_requires_every_unchanged_gate():
    status, passed, hard = output_decision(0.8, 0.8, 0.8)
    assert passed and status.startswith("PASS_")
    assert set(hard.values()) == {"PASS"}


def test_geometry_validate_fails_if_not_distinct_from_chips():
    status, passed, hard = output_decision(0.79, 1.0, 1.0)
    assert not passed and "GRADE_C" in status
    assert hard["geometry_distinct_from_chips_zero_acceptance"] == "FAIL"


def test_geometry_validate_fails_photometric_gate():
    _, passed, hard = output_decision(1.0, 1.0, 0.79)
    assert not passed
    assert hard["known_surface_photometric_consistency_at_least_0p80"] == "FAIL"
