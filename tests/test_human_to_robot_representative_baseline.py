"""Failure cases for the existing Flow Matching consumer's missing-field path."""
import json
from types import SimpleNamespace

import pytest
import torch
import numpy as np

from chaoyang.human_ego.training.FlowMatchingModel import FlowMatchingModel
from chaoyang.human_ego.training.FlowMatchingTrainer import sanitize_flow_action_targets
from chaoyang.ops.run_human_to_robot_representative_baseline import anchor_camera_wrist
from chaoyang.governance.common import artifact_ref
from chaoyang.governance import build_human_to_robot_r2_terminal_status as status_builder


def _loss(target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    stub = SimpleNamespace(
        horizon_weights=torch.ones(target.shape[1]),
        num_hands=2, hand_command_dim=22, base_action_dim=62,
        use_aux_obj_dynamics=False, use_done_in_flow=False,
        use_robot_state_conditioning=False,
        use_aux_visual_foresight=False,
        use_aux_temporal_contrastive=False,
    )
    preds = {"v_pred": torch.zeros_like(target)}
    data = {"v_target": target, "action_valid_mask": valid,
            "sample_weight": torch.ones(target.shape[0])}
    return FlowMatchingModel.compute_loss(stub, preds, data)["loss"]


def test_invalid_payload_does_not_reach_flow_path_or_valid_loss():
    mask = torch.zeros((1, 2, 62), dtype=torch.bool)
    mask[..., 0] = True
    a = torch.zeros((1, 2, 62))
    a[..., 0] = 0.5
    b = a.clone()
    a[..., 1] = float("nan")
    b[..., 1] = 999.0
    weight = torch.ones(1)
    safe_a = sanitize_flow_action_targets(a, mask, weight)
    safe_b = sanitize_flow_action_targets(b, mask, weight)
    assert torch.equal(safe_a, safe_b)
    torch.manual_seed(7)
    noise = torch.where(mask, torch.randn_like(safe_a), torch.zeros_like(safe_a))
    t = torch.tensor([[[0.25]]])
    assert torch.equal((1 - t) * noise + t * safe_a,
                       (1 - t) * noise + t * safe_b)
    assert torch.allclose(_loss(a, mask), _loss(b, mask))


def test_valid_supervision_changes_loss_and_zero_supervision_rejects():
    mask = torch.zeros((1, 2, 62), dtype=torch.bool)
    mask[..., 0] = True
    zero = torch.zeros((1, 2, 62))
    changed = zero.clone()
    changed[..., 0] = 1.0
    assert _loss(changed, mask) > _loss(zero, mask)
    changed[..., 1] = float("nan")
    assert torch.allclose(_loss(changed, mask), _loss(torch.where(mask, changed, zero), mask))
    with pytest.raises(ValueError, match="No effective action supervision"):
        _loss(zero, torch.zeros_like(mask))
    with pytest.raises(RuntimeError, match="No effective action supervision"):
        sanitize_flow_action_targets(zero, torch.zeros_like(mask), torch.ones(1))


def test_masked_current_robot_token_is_safe_before_actual_forward():
    torch.manual_seed(7)
    model = FlowMatchingModel(
        single_hand=False, pred_horizon=2, img_size=(32, 32),
        patch_size=16, vision_embed_dim=64, num_decoder_layers=1,
        num_heads=4, dropout=0.0, use_pcd_features=False,
        use_aux_obj_dynamics=False, use_aux_visual_foresight=False,
        use_aux_temporal_contrastive=False, use_region_attn=False,
        hand_action_representation="kaihand_joint_state",
    ).eval()
    rgb = torch.zeros((1, 3, 32, 32))
    ict = torch.zeros((1, 8, 29))
    ict_mask = torch.zeros((1, 8), dtype=torch.bool)
    xt = torch.zeros((1, 2, 62))
    time = torch.full((1, 1), 0.5)
    token_mask = torch.tensor([[True, False]])
    state_a = torch.zeros((1, 2, 33))
    state_b = state_a.clone()
    state_a[:, 1] = float("nan")
    state_b[:, 1] = 999.0
    with torch.inference_mode():
        first = model(x_rgb=rgb, x_ict=ict, ict_mask=ict_mask,
                      x_t=xt, t=time, x_robot_state=state_a,
                      robot_state_mask=token_mask)["v_pred"]
        second = model(x_rgb=rgb, x_ict=ict, ict_mask=ict_mask,
                       x_t=xt, t=time, x_robot_state=state_b,
                       robot_state_mask=token_mask)["v_pred"]
    assert torch.isfinite(first).all()
    assert torch.allclose(first, second, atol=0, rtol=0)


def test_future_wrist_uses_anchor_camera_not_its_own_moving_camera():
    anchor_c2w = np.eye(4)
    future_c2w = np.eye(4)
    future_c2w[0, 3] = .25
    future_camera_wrist = np.eye(4)
    future_camera_wrist[:3, 3] = [.10, -.02, .40]
    anchored = anchor_camera_wrist(anchor_c2w, future_c2w, future_camera_wrist)
    assert np.allclose(anchored[:3, 3], [.35, -.02, .40])
    assert not np.allclose(anchored[:3, 3], future_camera_wrist[:3, 3])
    with pytest.raises(ValueError, match="CAMERA_OR_WRIST_TRANSFORM_INVALID"):
        anchor_camera_wrist(anchor_c2w, np.full((4, 4), np.nan), future_camera_wrist)


def test_current_status_does_not_revert_to_historical_four_lane_row(tmp_path, monkeypatch):
    result_path = tmp_path / "representative_result.json"
    result_path.write_text(json.dumps({
        "task_id": status_builder.REPRESENTATIVE_BASELINE_TASK,
        "status": "REJECTED_QUALITY_LIMITED_DELIVERY",
        "product_counts": {"products_structure": "4/4", "products_quality": "0/4",
                           "products_adopted": "0/4"},
        "validation": {"path": "unchanged-fixture"},
    }), encoding="utf-8")
    monkeypatch.setattr(status_builder, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(status_builder, "OUTPUT", tmp_path / "STATUS.json")
    state = {
        "governance_revision": 1,
        "next_task": None,
        "tasks": [
            {"task_id": status_builder.FOUR_LANE_INCREMENT_TASK,
             "status": "REJECTED_QUALITY"},
            {"task_id": status_builder.REPRESENTATIVE_BASELINE_TASK,
             "status": "REJECTED_QUALITY", "result": artifact_ref(result_path)},
        ],
        "recent_events": [{"task_id": status_builder.REPRESENTATIVE_BASELINE_TASK}],
    }
    assert status_builder.publish_navigation_if_human_to_robot_r2(tmp_path, state, "now")
    status = json.loads((tmp_path / "STATUS.json").read_text(encoding="utf-8"))
    assert status["latest_task"] == status_builder.REPRESENTATIVE_BASELINE_TASK
    assert status["counts"]["products_quality"] == "0/4"
