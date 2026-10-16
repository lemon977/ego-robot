"""Explicit explicit nine-family builder; legacy builders and existing code remain immutable."""
from chaoyang.governance.current_r3_contracts import *
from chaoyang.governance import current_r3_contracts as _original
def build_algorithm_contract(baseline_registry,authority,*,planned_family_descriptor_ref=None):
 value=_original.build_algorithm_contract(baseline_registry,authority)
 from chaoyang.governance.candidate_optimization_family_v10 import build_candidate_contract
 return build_candidate_contract(value,planned_family_descriptor_ref=planned_family_descriptor_ref)
