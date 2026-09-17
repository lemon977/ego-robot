import cv2
import numpy as np

from chaoyang.ops.run_poker_causal_plane_donor_runtime_gate_v2 import evaluate_pair_v2


def _fixture():
    rng = np.random.default_rng(7)
    image = rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8)
    mask = np.zeros((240, 320), np.uint8)
    mask[30:210, 35:285] = 255
    matrix = np.float32([[1, 0, 4], [0, 1, 3]])
    target = cv2.warpAffine(image, matrix, (320, 240), borderMode=cv2.BORDER_REFLECT_101)
    target_mask = cv2.warpAffine(mask, matrix, (320, 240), flags=cv2.INTER_NEAREST)
    hole = np.zeros_like(mask, dtype=bool)
    hole[90:145, 120:205] = True
    return image, target, mask, target_mask, hole


def test_hidden_truth_cannot_change_runtime_decision():
    source, target, source_mask, target_mask, hole = _fixture()
    first, _ = evaluate_pair_v2(source, target, source_mask, target_mask, hole)
    changed = target.copy()
    changed[hole] = (255, 0, 255)
    second, _ = evaluate_pair_v2(source, changed, source_mask, target_mask, hole)

    assert first["runtime_decision"] == second["runtime_decision"]
    assert first["runtime_checks"] == second["runtime_checks"]
    assert first["runtime_evidence"] == second["runtime_evidence"]
    assert first["oracle_evaluation"] != second["oracle_evaluation"]
