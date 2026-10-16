"""Authorize finite observer replay before importing algorithm modules."""
import importlib.util
import sys
from pathlib import Path


def main(argv=None):
    root = Path(__file__).resolve().parents[3]
    definition = importlib.util.spec_from_file_location(
        "_fenced_controller_observer", root / "scripts/maintenance/inpainting_controller_observer_v1.py")
    runner = importlib.util.module_from_spec(definition)
    definition.loader.exec_module(runner)
    packet = runner.authorize_operation(list(sys.argv[1:] if argv is None else argv))
    from chaoyang.inpainting_candidate.controller_replay import execute_controller_replay
    return execute_controller_replay(packet)
