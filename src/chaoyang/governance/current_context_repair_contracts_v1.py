"""Explicit added finite contextpad4 family; preserves legacy contract fields."""
from chaoyang.governance.current_r3_contracts import *
from chaoyang.governance import current_r3_contracts as _original
def build_algorithm_contract(baseline_registry, authority):
    value = _original.build_algorithm_contract(baseline_registry, authority)
    from chaoyang.governance.execute_pick_place_context_pad4_v1 import ROUTE_KEY, route_contract
    from chaoyang.governance.common import REPO_ROOT
    routes = value['special_status_contracts']
    if ROUTE_KEY in routes:
        raise RuntimeError('CONTEXT_REPAIR_ROUTE_ALREADY_EXISTS')
    routes[ROUTE_KEY] = route_contract(REPO_ROOT)
    return value
