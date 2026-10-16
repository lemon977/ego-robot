"""Fenced experimental late-entry Mask; not the default V1."""
import importlib.util
import sys

def main(argv=None):
    from chaoyang.governance.common import REPO_ROOT
    path=REPO_ROOT/'scripts/maintenance/inpainting_late_entry_one_hour_v1.py'
    spec=importlib.util.spec_from_file_location('_late_entry_fence',path)
    gate=importlib.util.module_from_spec(spec); spec.loader.exec_module(gate)
    packet=gate.authorize_operation(list(sys.argv[1:] if argv is None else argv))
    from chaoyang.inpainting_candidate.late_entry_seeds import execute_mask_experiment
    return execute_mask_experiment(packet)
