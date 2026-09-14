import numpy as np

from tools.diagnose_002_012_task26_topology_coverage_t0 import raw_bool_sha, summary, topology


def test_topology_empty_is_retained():
    out = topology(np.zeros((20, 20), bool))
    assert out["topology_state"] == "EMPTY"
    assert out["significant_component_count"] == 0


def test_topology_significant_definition():
    mask = np.zeros((100, 100), bool)
    mask[2:12, 2:12] = True
    mask[50:60, 50:60] = True
    out = topology(mask)
    assert out["significant_component_threshold_px"] == 32
    assert out["significant_component_count"] == 2
    assert out["topology_state"] == "MULTI_SIGNIFICANT"


def test_topology_does_not_change_pixels():
    mask = np.zeros((30, 30), bool)
    mask[4:20, 6:25] = True
    before = raw_bool_sha(mask)
    topology(mask)
    assert raw_bool_sha(mask) == before


def test_summary_never_promotes_proxy_to_semantic_coverage():
    rows = [
        {
            "selector_status": "ACCEPT",
            "topology_state": "ONE_SIGNIFICANT",
            "component_count_all": 1,
            "mask_area_px": 100,
            "boundary_robust_band_state": "MEASURED_ON_VISIBLE_IMAGE_INTERSECTION",
            "projected_wrist_location": "IN_IMAGE",
            "boundary_robust_wrist_band_coverage_proxy": 0.5,
        },
        {
            "selector_status": "REJECT",
            "topology_state": "EMPTY",
            "component_count_all": 0,
            "mask_area_px": 0,
            "boundary_robust_band_state": "GEOMETRY_UNAVAILABLE",
            "projected_wrist_location": "UNAVAILABLE",
            "boundary_robust_wrist_band_coverage_proxy": None,
        },
    ]
    out = summary(rows)
    assert out["semantic_wearable_coverage_status"] == "WEARABLE_COVERAGE_UNMEASURED"
    assert out["topology_counts"]["EMPTY"] == 1
    assert out["boundary_robust_wrist_band_coverage_proxy"]["count"] == 1

