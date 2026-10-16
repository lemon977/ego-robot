"""Path-portable dispatcher restricted to the existing Joint26 ICT operations."""
from pathlib import Path
import sys


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    from .joint26_contract import PREPARE, TRAIN
    if arguments != ["validate-governance"] and not (
            len(arguments) >= 2 and arguments[0] == "run"
            and arguments[1] in {PREPARE, TRAIN}):
        raise ValueError("JOINT26_PORTABLE_DISPATCH_SCOPE_REQUIRED")
    from .governance_mount_compat import install_governance_mount_compat
    install_governance_mount_compat(Path(__file__).resolve().parents[3])
    from chaoyang.cli import main as dispatch
    return dispatch(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
