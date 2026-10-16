"""Authorize the finite local-crop diagnostic before importing model code."""
import importlib.util
import sys
from pathlib import Path


def main(argv=None):
    root = Path(__file__).resolve().parents[3]
    definition = importlib.util.spec_from_file_location(
        "_fenced_local_crop", root / "scripts/maintenance/inpainting_local_crop_v1.py")
    runner = importlib.util.module_from_spec(definition)
    definition.loader.exec_module(runner)
    packet = runner.authorize_operation(list(sys.argv[1:] if argv is None else argv))
    from chaoyang.inpainting_candidate.local_crop_worker import execute_local_crop
    return execute_local_crop(packet)

