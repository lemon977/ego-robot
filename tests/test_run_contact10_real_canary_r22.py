from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from chaoyang.ops.run_contact10_real_canary_r22 import state_for


def test_contact_state_thresholds_are_fail_closed() -> None:
    assert state_for(3.0, -1.0, 2.0, False) == "TOUCH_CANDIDATE"
    assert state_for(3.0, 0.0, 30.0, True) == "SLIDE_CANDIDATE"
    assert state_for(20.0, -6.0, 1.0, False) == "APPROACH"
    assert state_for(20.0, 6.0, 1.0, True) == "RELEASE"
    assert state_for(100.0, -100.0, 0.0, False) == "UNKNOWN"
