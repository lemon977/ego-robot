"""Independent Revo2 ICT candidate; importing this module never loads weights."""
from __future__ import annotations

from torch import nn

from chaoyang.human_ego.training.FlowMatchingModel import FlowMatchingModel


class Revo2ICTModel(FlowMatchingModel):
    """RGB + bimanual ICT to wrist poses and six real hand commands per side.

    Action order: L_xyz, R_xyz, L_rot6, R_rot6, L_hand6, R_hand6.
    The frozen legacy done head is retained only for state-dict compatibility;
    its output never enters this candidate's loss or action space.
    """

    control_ground_truth = False
    physical_deployment_authorized = False
    action_layout = "modality_major_revo2_30"

    def __init__(self, *, pred_horizon: int = 50, **architecture):
        forbidden = {"single_hand", "hand_action_representation", "use_done_in_flow",
                     "use_aux_obj_dynamics", "use_aux_visual_foresight",
                     "use_aux_temporal_contrastive", "use_pcd_features", "use_region_attn"}
        if forbidden.intersection(architecture):
            raise ValueError("Revo2 candidate branches are fixed, not architecture overrides")
        super().__init__(single_hand=False, pred_horizon=pred_horizon,
                         hand_action_representation="grasp_binary", use_done_in_flow=False,
                         use_aux_obj_dynamics=False, use_aux_visual_foresight=False,
                         use_aux_temporal_contrastive=False, use_pcd_features=False,
                         use_region_attn=False, **architecture)
        width = self.action_proj.out_features
        self.hand_action_representation = "revo2_hand6_candidate"
        self.hand_command_dim = 6
        self.base_action_dim = self.action_dim = 30
        self.action_proj = nn.Linear(30, width)
        self.head_v = nn.Linear(width, 30)
        self.head_done.requires_grad_(False)

    def forward(self, *, x_rgb, x_ict, ict_mask, x_t, t):
        if x_t.shape[-1] != 30 or x_ict.shape[-1] != 29:
            raise ValueError("Revo2 requires 30D actions and 29D bimanual ICT")
        prediction = super().forward(x_rgb=x_rgb, x_ict=x_ict, ict_mask=ict_mask,
                                     x_t=x_t, t=t)
        return {"v_pred": prediction["v_pred"]}
