# -*- coding: utf-8 -*-
# @FileName: FlowMatchingTrainer.py
"""
FlowMatchingTrainer (Flow Matching, multi-step) - The Ultimate Paradigm Shift Version

Pipeline Alignment:
- Dataset: training.FlowMatchingDataloader
    Inputs: x_rgb, x_ict (Tokens), ict_mask, x_pcd (Explicit Geometry)
    Targets: y_action, y_obj_action, y_trace, x_ict_future
- Model: training.FlowMatchingModel
    Forward outputs: v_pred, trace_pred, ict_fut_pred

Features:
- Generative Training via Flow Matching.
- OT-CFM (Optimal Transport Conditional Flow Matching): Hungarian matching for straighter flows.
- ICT: Handles variable number of objects natively.
- Full Ablation Suite: Image modes, Frame modes, Action modes, Dual-hand modes.
- Aux: Visual Foresight, Object Dynamics, Temporal Contrastive.
"""

from __future__ import annotations

import os
import glob
import json
import sys
import subprocess
import yaml
import time
import math
from pathlib import Path

# H20 + CUDA 12.8 can raise SIGFPE inside libcublasLt for some validation
# GEMM shapes.  Force the stable cuBLAS path before importing torch.
os.environ.setdefault("DISABLE_ADDMM_CUDA_LT", "1")
os.environ.setdefault("TORCH_BLAS_PREFER_CUBLASLT", "0")

from tqdm import tqdm
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional
import numpy as np
import scipy.spatial.distance
from scipy.optimize import linear_sum_assignment
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
torch.backends.mha.set_fastpath_enabled(False)
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn as nn

from utils.utils_math import (
    rot6d_to_R_batch,
    geodesic_deg_from_R,
    set_seed
)

from training.FlowMatchingDataloader import (
    MPSSessions,
    FlowMatchingDataloader,
    DEFAULT_STATS,
    IMG_NAME
)
from training.FlowMatchingModel import FlowMatchingModel
from training.checkpoint_transfer import load_compatible_pretrained
from training.FlowMatchingEvaluator import run_teacher_forced_vis

console = Console()

# =========================================================
# -------- Config ---------
# =========================================================

def discover_session_paths(data_root: str, task: str, source_type: str) -> list:
    """
    Auto-discover session paths for a given source type.
    Looks in: data_root/task/source_type/  (e.g. data/serve_bread/aria/)
    Returns sorted list of session paths.
    """
    type_dir = os.path.join(data_root, task, source_type)
    if not os.path.isdir(type_dir):
        return []
    sessions = []
    for name in sorted(os.listdir(type_dir)):
        if os.path.isdir(os.path.join(type_dir, name)):
            sessions.append(os.path.join(type_dir, name))
    return sessions


def resolve_data_sources(data_sources: dict, data_root: str, task: str, eval_source: str = "aria") -> tuple:
    """
    Resolve data_sources config into train/eval path lists.

    Cross-session split: session 0 of the eval_source type is reserved for evaluation,
    all remaining sessions are used for training. This ensures evaluation measures
    generalization to unseen demonstrations (not within-session interpolation).

    data_sources: e.g. {"aria": 60, "teleop": 10}
      - int value = take first N sessions of that type (clamped to available count)

    eval_source: which source type to draw eval from (default: "aria")
      - eval always uses session index 0 (e.g. mps_*_000_*)

    Returns: (train_paths, eval_paths)
    """
    # Check if eval_source type exists in data_sources
    eval_source_in_data = eval_source in data_sources and int(data_sources[eval_source]) > 0

    # If eval_source not in data_sources, pick the first active source as eval_source
    if not eval_source_in_data:
        for source_type, count in data_sources.items():
            if int(count) > 0:
                eval_source = source_type
                break

    train_paths = []
    eval_paths = []

    for source_type, count in data_sources.items():
        count = int(count)
        if count <= 0:
            continue
        all_sessions = discover_session_paths(data_root, task, source_type)

        if source_type == eval_source and len(all_sessions) > 0:
            eval_paths.append(all_sessions[0])
            available_train = all_sessions[1:]
        else:
            available_train = all_sessions

        # clamp to available count
        train_paths.extend(available_train[:count])

    # Print cross-session split summary
    print(f"[Data Split] Cross-session evaluation:")
    print(f"  Eval sessions  ({len(eval_paths)}): {[os.path.basename(p) for p in eval_paths]}")
    print(f"  Train sessions ({len(train_paths)}): {len(train_paths)} sessions "
          f"({os.path.basename(train_paths[0])} .. {os.path.basename(train_paths[-1])})" if train_paths else
          f"  Train sessions (0): NONE — need more data!")

    return train_paths, eval_paths


@dataclass
class TrainConfig:
    # --- Data & Paths ---
    out_dir: str = "./runs/YOUR_TASK/YOUR_JOB"
    data_root: str = "./data"
    MPS_PATHS_TRAIN: list = field(default_factory=lambda:[f"./data/serve_bread/aria/mps_serve_bread_{i:03d}_vrs" for i in range(1, 43)])
    MPS_PATHS_EVAL: list = field(default_factory=lambda:["./data/serve_bread/aria/mps_serve_bread_000_vrs"])
    data_sources: dict = None   # e.g. {"aria": 20, "teleop": 10, "teaching": 0}
    eval_source: str = "aria"   # which source type to use as eval (session 0)
    data_num: int = None
    task: str = "serve_bread"

    # --- Paradigm Ablation Switches ---
    img_name: Optional[str] = IMG_NAME
    centric_mode: str = 'object_centric'     # 'object_centric' | 'ego_centric'
    frame_mode: str = 'anchor_frame'          # 'anchor_frame' | 'camera_frame'
    action_mode: str = 'absolute'
    hand_action_representation: str = 'grasp_binary'
    robot_sidecar_root: Optional[str] = None
    pretrained_checkpoint: Optional[str] = None
    use_pcd_features: bool = False

    # --- Aux Switches ---
    use_aux_obj_dynamics: bool = False
    use_aux_visual_foresight: bool = False
    use_aux_temporal_contrastive: bool = False

    use_region_attn: bool = False
    use_ot_cfm: bool = False

    # --- Fine-grained Augmentation Switches ---
    enable_augmentation: bool = True
    enable_aug_img: bool = True
    enable_aug_rrc: bool = True
    enable_aug_target_jittering: bool = True
    enable_aug_cutout: bool = True
    enable_aug_temporal_stride: bool = False
    enable_aug_interpolation: bool = True


    # --- AMP & LR Schedule ---
    use_amp: bool = False
    use_lr_schedule: bool = False
    warmup_steps: int = 200
    min_lr_ratio: float = 0.05

    # --- Mode & Sizing ---
    image_size: Tuple[int, int] = (240, 320)
    pred_horizon: int = 50
    single_hand: bool = False
    single_hand_side: str = "right"
    max_ict: int = 8

    # --- Training Hyperparams ---
    batch_size: int = 32
    eval_batch_size: Optional[int] = None
    epochs: int = 400
    lr: float = 1e-4
    weight_decay: float = 0.01
    num_workers: int = 8
    cache_json_in_memory: bool = False
    cache_image_bytes_in_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: int = 2
    max_train_batches: Optional[int] = None
    max_eval_batches: Optional[int] = None
    early_stop_patience: Optional[int] = None
    skip_visual_eval: bool = False
    grad_clip: float = 1.0

    # --- EMA ---
    use_ema: bool = True
    ema_decay: float = 0.999

    # --- Loss Weights ---
    w_flow: float = 3.0
    w_pos: float = 2.0
    w_rot: float = 1.0
    w_g: float = 10.0
    w_hand_joint: float = 5.0
    w_joint_limit: float = 0.1
    w_velocity: float = 0.1
    w_done: float = 5.0
    w_foresight: float = 1.0
    w_contrastive: float = 1.0

    # --- Legacy Compatibility Switches ---
    use_pre_norm: bool = False            # False = Post-Norm (legacy)
    use_ctx_norm: bool = False            # False = no LayerNorm on context (legacy)
    use_done_in_flow: bool = False       # True = Done flag as last dim in flow matching (legacy)
    use_legacy_image_loading: bool = True  # True = load from JSON abs path at original res (legacy)
    use_legacy_rng: bool = False         # True = deterministic RNG seeds per augmentation (legacy)
    enable_aug_interpolation: bool = False  # True = sub-step interpolation augmentation (legacy)

    # --- State Noise ---
    use_state_noise: bool = False
    state_pos_noise_std: float = 0.001
    state_rot_noise_deg: float = 1.0
    state_grasp_noise_std: float = 0.00

    # --- Hand Tracking Method ---
    hand_tracking_method: str = "aria_mps"   # 也支持 HumanEgoAdapter 生成的 "hawor_v3"

    # --- Flow Matching Inference ---
    num_inference_steps: int = 10
    model_horizon_weighting: str = "uniform"
    model_horizon_beta: float = 0.0

    # --- Model Architecture ---
    patch_size: int = 16
    vision_embed_dim: int = 384
    num_decoder_layers: int = 6
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.05

    # --- Eval Scheduling ---
    eval_every: int = 1
    vis_eval_every: int = 50

    make_video: bool = True

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 7


# =========================================================
# -------- Utilities -------
# =========================================================

def _panel_cfg(cfg: TrainConfig) -> Panel:
    txt = "\n".join([
        f"[bold]Device[/bold]: {cfg.device}  AMP={cfg.use_amp}",
        f"[bold]Train/Eval[/bold]: {len(cfg.MPS_PATHS_TRAIN)} sessions / {len(cfg.MPS_PATHS_EVAL)} sessions",
        f"[bold]Paradigm[/bold]: Centric=[cyan]{cfg.centric_mode}[/] | Frame=[cyan]{cfg.frame_mode}[/] | Action=[cyan]{cfg.action_mode}[/] | HandCommand=[cyan]{cfg.hand_action_representation}[/] | Image=[cyan]{cfg.img_name}[/]",
        f"[bold]Aux[/bold]: ObjDynamics={cfg.use_aux_obj_dynamics} | VisForesight={cfg.use_aux_visual_foresight} | TempContrastive={cfg.use_aux_temporal_contrastive}",
        f"[bold]Architecture[/bold]: RegionAttn={cfg.use_region_attn} | PCD={cfg.use_pcd_features} | OT-CFM={cfg.use_ot_cfm} | Hands={'Single' if cfg.single_hand else 'Dual'}",
        f"[bold]Legacy[/bold]: PreNorm={cfg.use_pre_norm} | CtxNorm={cfg.use_ctx_norm} | DoneInFlow={cfg.use_done_in_flow} | LegacyImg={cfg.use_legacy_image_loading} | LegacyRNG={cfg.use_legacy_rng} | AugInterp={cfg.enable_aug_interpolation}",
        f"[bold]Model[/bold]: patch={cfg.patch_size} D={cfg.vision_embed_dim} L={cfg.num_decoder_layers} heads={cfg.num_heads}",
        f"[bold]Loss Weights[/bold]: w_p={cfg.w_pos} w_r={cfg.w_rot} w_g={cfg.w_g} w_joint={cfg.w_hand_joint} lam_f={cfg.w_foresight} lam_c={cfg.w_contrastive}",
        f"[bold]Training[/bold]: Batch={cfg.batch_size} Epochs={cfg.epochs} LR={cfg.lr} EMA={cfg.use_ema}({cfg.ema_decay})",
        f"[bold]Input Pipeline[/bold]: JsonCache={cfg.cache_json_in_memory} ImageBytesCache={cfg.cache_image_bytes_in_memory} PersistentWorkers={cfg.persistent_workers} Prefetch={cfg.prefetch_factor}",
        f"[bold]Out dir[/bold]: {cfg.out_dir}",
    ])
    return Panel(txt, title="GRASP POLICY TRAIN CONFIG (ULTIMATE FM)", expand=False)


def is_zero_state(x_ict: torch.Tensor, ict_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # Token layout index 1:4 is Position. Token 0 is Hand.
    return (x_ict[:, 0, 1:4].abs().sum(dim=1) <= eps)

def get_dataset_stats(mps_paths: list, cfg: TrainConfig) -> dict:
    console.print(f"[bold yellow]Computing dataset statistics for {len(mps_paths)} sessions...[/]")
    tmp_ds = FlowMatchingDataloader(
        sessions=[MPSSessions(p) for p in mps_paths],
        single_hand=cfg.single_hand, single_hand_side=cfg.single_hand_side,
        centric_mode=cfg.centric_mode, frame_mode=cfg.frame_mode,
        action_mode=cfg.action_mode, img_name=cfg.img_name,
        hand_action_representation=cfg.hand_action_representation,
        robot_sidecar_root=cfg.robot_sidecar_root,
        hand_tracking_method=cfg.hand_tracking_method,
        use_legacy_image_loading=cfg.use_legacy_image_loading,
        max_ict=cfg.max_ict,
        enable_augmentation=False,
        stats=None,
        cache_json_in_memory=cfg.cache_json_in_memory,
    )
    all_pos =[]
    num_samples = len(tmp_ds)
    step = max(1, num_samples // 10000)

    for i in tqdm(range(0, num_samples, step), desc="Collecting Positions"):
        # Statistics only need the current geometry.  Calling __getitem__ also
        # decodes an image and reads K future JSON files, which made PICO NAS
        # sessions unnecessarily slow (minutes instead of seconds).
        json_path = tmp_ds.samples[i]
        frame = tmp_ds._read_frame(json_path)
        T_w2ref = tmp_ds._get_T_w2ref(frame)
        state_np, _, mask_np = tmp_ds._build_ict(frame, T_w2ref)
        state = torch.from_numpy(state_np)
        mask = torch.from_numpy(mask_np)

        # Robustly find Hand tokens (TypeID == 1.0 or 2.0)
        for j in range(state.shape[0]):
            if mask[j] and (state[j, 0] == 1.0 or state[j, 0] == 2.0):
                all_pos.append(state[j, 1:4].numpy()) # 1:4 is Position X, Y, Z

    if len(all_pos) == 0:
        return DEFAULT_STATS

    all_pos = np.array(all_pos)
    mean = all_pos.mean(axis=0).tolist()
    std  = all_pos.std(axis=0).tolist()
    std = [float(max(s, 1e-4)) for s in std]
    stats = {"pos": {"mean": [float(x) for x in mean], "std":[float(x) for x in std]}}
    console.print(f"[green]Stats Computed![/]\n Mean: {stats['pos']['mean']}\n Std: {stats['pos']['std']}")
    return stats

def get_lr_cosine_with_warmup(base_lr: float, step: int, warmup_steps: int, total_steps: int, min_lr_ratio: float) -> float:
    if total_steps <= 0: return base_lr
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)
    t = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    t = min(max(t, 0.0), 1.0)
    min_lr = base_lr * float(min_lr_ratio)
    return float(min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * t)))

def apply_ot_matching(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """
    Optimal Transport Conditional Flow Matching (OT-CFM).
    Uses Hungarian Algorithm to align noise x0 with target manifold x1.
    Since our manifold is pre-normalized via dataset stats, Euclidean cost is balanced.
    """
    B = x0.size(0)
    if B <= 1: return x0

    x0_np = x0.view(B, -1).detach().cpu().numpy()
    x1_np = x1.view(B, -1).detach().cpu().numpy()

    cost_matrix = scipy.spatial.distance.cdist(x0_np, x1_np, metric='sqeuclidean')
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    x0_matched = torch.empty_like(x0)
    for r, c in zip(row_ind, col_ind):
        x0_matched[c] = x0[r]

    return x0_matched

def add_rot6d_noise(o6d: torch.Tensor, max_deg: float):
    # Dummy placeholder for SO(3) noise, simplified to Gaussian for speed
    if max_deg <= 0: return o6d
    return o6d + torch.randn_like(o6d) * (max_deg / 180.0)

# ------------------------------------------------------------
# EMA Helper
# ------------------------------------------------------------
class EMAModel:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])

    def state_dict(self):
        return {name: value.clone() for name, value in self.shadow.items()}

    def load_state_dict(self, state):
        if set(state) != set(self.shadow):
            raise ValueError("EMA checkpoint keys do not match the model")
        self.shadow = {
            name: value.to(self.shadow[name].device).clone()
            for name, value in state.items()
        }


def save_checkpoint_atomic(payload: dict, path: str) -> None:
    """Never expose a partially written checkpoint to resume/watchdog code."""
    temporary = path + ".tmp"
    torch.save(payload, temporary)
    os.replace(temporary, path)


# =========================================================
# ---- Train Epoch --------
# =========================================================
def train_one_epoch(
    model: FlowMatchingModel,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    cfg: TrainConfig,
    epoch: int,
    global_step: int,
    total_steps: int,
    ema: Optional[EMAModel] = None,
) -> Tuple[Dict[str, float], int]:

    model.train()

    agg = {"loss": 0.0, "l_flow": 0.0, "l_pos": 0.0, "l_rot": 0.0, "l_g": 0.0, "l_done": 0.0, "l_foresight": 0.0, "l_contrastive": 0.0, "zero_ratio":[]}
    n = 0

    def _nan_guard(x: torch.Tensor, tag: str):
        if not torch.isfinite(x).all():
            raise RuntimeError(f"[NaN/Inf] {tag} at epoch={epoch} global_step={global_step}")

    def _maybe_apply_state_noise(st: torch.Tensor) -> torch.Tensor:
        if not cfg.use_state_noise: return st
        st = st.clone()
        B, T, D = st.shape

        # Only inject noise into Hand Tokens (Type 1.0 or 2.0)
        hand_mask = (st[:, :, 0] == 1.0) | (st[:, :, 0] == 2.0)

        if cfg.state_pos_noise_std > 0:
            pos_std_tensor = torch.from_numpy(loader.dataset.pos_std).to(st.device)
            noise_pos = (torch.randn_like(st[:, :, 1:4]) * cfg.state_pos_noise_std) / pos_std_tensor
            st[:, :, 1:4] = torch.where(hand_mask.unsqueeze(-1), st[:, :, 1:4] + noise_pos, st[:, :, 1:4])

        if cfg.state_rot_noise_deg > 0:
            noisy_rot = add_rot6d_noise(st[:, :, 4:10].reshape(B*T, 6), max_deg=cfg.state_rot_noise_deg).reshape(B, T, 6)
            st[:, :, 4:10] = torch.where(hand_mask.unsqueeze(-1), noisy_rot, st[:, :, 4:10])

        if cfg.state_grasp_noise_std > 0:
            noise_g = torch.randn_like(st[:, :, -1:]) * cfg.state_grasp_noise_std
            st[:, :, -1:] = torch.where(hand_mask.unsqueeze(-1), torch.clamp(st[:, :, -1:] + noise_g, 0.0, 1.0), st[:, :, -1:])

        return st

    for batch in loader:
        x_rgb = batch["x_rgb"].to(cfg.device, non_blocking=True)
        x_ict = batch["x_ict"].to(cfg.device, non_blocking=True)
        ict_mask = batch["ict_mask"].to(cfg.device, non_blocking=True)
        x_robot_state = batch.get("x_robot_state")
        robot_state_mask = batch.get("robot_state_mask")
        if x_robot_state is not None:
            x_robot_state = x_robot_state.to(cfg.device, non_blocking=True)
            robot_state_mask = robot_state_mask.to(cfg.device, non_blocking=True)

        # Explicit Geometry injection
        x_pcd = batch.get("x_pcd", None)
        if x_pcd is not None: x_pcd = x_pcd.to(cfg.device, non_blocking=True)

        # --- Joint Manifold Target Assembly ---
        y_action = batch["y_action"].to(cfg.device, non_blocking=True)
        y_done = batch["y_done"].to(cfg.device, non_blocking=True)
        if cfg.use_aux_obj_dynamics:
            y_obj_action = batch["y_obj_action"].to(cfg.device, non_blocking=True)
            x_1 = torch.cat([y_action, y_obj_action], dim=-1)  # 19D or 29D
        else:
            x_1 = y_action  # 10D or 20D
        action_valid_mask = batch["action_valid_mask"].to(cfg.device)
        if cfg.use_aux_obj_dynamics:
            action_valid_mask = torch.cat(
                [action_valid_mask, torch.ones_like(y_obj_action, dtype=torch.bool)],
                dim=-1,
            )

        # Done-in-flow: append done flag as last dimension of flow target
        if cfg.use_done_in_flow:
            x_1 = torch.cat([x_1, y_done], dim=-1)  # +1D
            action_valid_mask = torch.cat(
                [action_valid_mask, torch.ones_like(y_done, dtype=torch.bool)],
                dim=-1,
            )

        B = x_1.shape[0]

        if not torch.isfinite(x_1).all() or not torch.isfinite(x_ict).all():
            raise RuntimeError("NaN found in Dataloader outputs!")

        agg["zero_ratio"].append(is_zero_state(x_ict, ict_mask).float().mean().detach().cpu().item())

        if cfg.use_lr_schedule:
            lr = get_lr_cosine_with_warmup(cfg.lr, global_step, cfg.warmup_steps, total_steps, cfg.min_lr_ratio)
            for pg in opt.param_groups: pg["lr"] = lr

        opt.zero_grad(set_to_none=True)

        # --- FLOW MATCHING & OT-CFM ---
        x_0 = torch.randn_like(x_1)
        if cfg.use_ot_cfm:
            x_0 = apply_ot_matching(x_0, x_1)

        t = torch.rand((B, 1), device=cfg.device)
        t_expand = t.unsqueeze(-1)

        x_t = (1 - t_expand) * x_0 + t_expand * x_1
        v_target = x_1 - x_0

        st_in = _maybe_apply_state_noise(x_ict)
        anchor_uv = batch["anchor_uv"].to(cfg.device, non_blocking=True) if cfg.use_region_attn else None

        # --- Aux Targets Assembly ---
        loss_targets = {
            "v_target": v_target,
            "action_valid_mask": action_valid_mask,
            "y_action": y_action,
        }
        if "joint_lower" in batch:
            loss_targets["joint_lower"] = batch["joint_lower"].to(cfg.device)
            loss_targets["joint_upper"] = batch["joint_upper"].to(cfg.device)
        # Done target: use last step of horizon as trajectory-level label
        loss_targets["y_done"] = y_done[:, -1, :]  # (B, 1)
        if cfg.use_aux_visual_foresight:
            loss_targets["y_2d_trace"] = batch["y_2d_trace"].to(cfg.device, non_blocking=True)
        if cfg.use_aux_temporal_contrastive:
            loss_targets["x_ict_future"] = batch["x_ict_future"].to(cfg.device, non_blocking=True)
            loss_targets["ict_mask_future"] = batch["ict_mask_future"].to(cfg.device, non_blocking=True)

        def _forward_and_loss():
            preds = model(
                x_rgb=x_rgb, x_ict=st_in, ict_mask=ict_mask,
                x_t=x_t, t=t, x_pcd=x_pcd,
                x_robot_state=x_robot_state,
                robot_state_mask=robot_state_mask,
                anchor_uv=anchor_uv
            )
            preds["action_pred"] = (
                x_t + (1.0 - t.unsqueeze(1)) * preds["v_pred"]
            )
            loss_weights = {
                "w_pos": cfg.w_pos,
                "w_rot": cfg.w_rot,
                "w_g": cfg.w_g,
                "w_hand_joint": cfg.w_hand_joint,
                "w_done": cfg.w_done,
                "w_flow": cfg.w_flow,
            }
            loss_lambdas = {
                "lambda_foresight": cfg.w_foresight,
                "lambda_contrastive": cfg.w_contrastive,
                "lambda_joint_limit": cfg.w_joint_limit,
                "lambda_velocity": cfg.w_velocity,
            }
            loss_dict = model.compute_loss(preds, loss_targets, weights=loss_weights, loss_lambdas=loss_lambdas)
            return loss_dict["loss"], loss_dict

        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        if cfg.use_amp and scaler is not None:
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                loss, loss_dict = _forward_and_loss()
            _nan_guard(loss, "loss")
            scaler.scale(loss).backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
        else:
            loss, loss_dict = _forward_and_loss()
            _nan_guard(loss, "loss")
            loss.backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

        if ema is not None: ema.update()

        # Update Aggregators
        agg["loss"] += float(loss.item())
        agg["l_flow"] += float(loss_dict.get("loss_flow", 0.0))
        agg["l_pos"] += float(loss_dict.get("loss_pos", 0.0))
        agg["l_rot"] += float(loss_dict.get("loss_rot", 0.0))
        agg["l_g"] += float(loss_dict.get("loss_g", 0.0))
        agg["l_done"] += float(loss_dict.get("loss_done", 0.0))
        agg["l_foresight"] += float(loss_dict.get("loss_foresight", 0.0))
        agg["l_contrastive"] += float(loss_dict.get("loss_contrastive", 0.0))

        n += 1
        global_step += 1
        if cfg.max_train_batches is not None and n >= cfg.max_train_batches:
            break

    for k in["loss", "l_flow", "l_pos", "l_rot", "l_g", "l_done", "l_foresight", "l_contrastive"]:
        agg[k] /= max(1, n)

    agg["zero_state_ratio"] = float(np.mean(agg["zero_ratio"])) if agg["zero_ratio"] else 0.0
    agg["lr"] = float(opt.param_groups[0]["lr"])

    return agg, global_step


# =========================
# ---- Eval Inference -----
# =========================
@torch.no_grad()
def eval_ode_inference(
    model: FlowMatchingModel,
    loader: DataLoader,
    cfg: TrainConfig,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:

    model.eval()
    K = cfg.pred_horizon

    pos_err_k = np.zeros((K,), dtype=np.float64)
    rot_err_k = np.zeros((K,), dtype=np.float64)
    grasp_acc_k = np.zeros((K,), dtype=np.float64)
    joint_mae_k = np.zeros((K,), dtype=np.float64)
    done_correct, done_total = 0, 0

    w = model.horizon_weights.detach().float()
    if (not torch.isfinite(w).all()) or (w.abs().sum() < 1e-12): w = torch.ones_like(w)
    w = w / (w.sum() + 1e-8)
    w_cpu = w.cpu().numpy().astype(np.float64)

    n_batches, n_frames = 0, 0
    zero_ratio = []
    g_scores_cache, g_labels_cache = [[] for _ in range(K)], [[] for _ in range(K)]
    command_dim = {
        "grasp_binary": 1,
        "kaihand_joint_state": 22,
        "wuji_joint_state": 20,
    }[cfg.hand_action_representation]

    for batch in loader:
        x_rgb = batch["x_rgb"].to(cfg.device)
        x_ict = batch["x_ict"].to(cfg.device)
        ict_mask = batch["ict_mask"].to(cfg.device)
        x_pcd = batch.get("x_pcd", None)
        if x_pcd is not None: x_pcd = x_pcd.to(cfg.device)
        x_robot_state = batch.get("x_robot_state")
        robot_state_mask = batch.get("robot_state_mask")
        if x_robot_state is not None:
            x_robot_state = x_robot_state.to(cfg.device)
            robot_state_mask = robot_state_mask.to(cfg.device)

        y_action_full = batch["y_action"].to(cfg.device)
        action_valid = batch["action_valid_mask"].to(cfg.device).bool()
        y_done = batch["y_done"].to(cfg.device)

        # Re-split for evaluation metrics
        if cfg.single_hand:
            y_pos = y_action_full[:, :, 0:3]
            y_o6d = y_action_full[:, :, 3:9]
            y_g = y_action_full[:, :, 9:9 + command_dim]
        else:
            y_pos = y_action_full[:, :, 0:6]
            y_o6d = y_action_full[:, :, 6:18]
            y_g = y_action_full[:, :, 18:18 + 2 * command_dim]

        B = y_action_full.shape[0]

        # Build x_t with correct dimension based on Aux flags
        x_t = torch.randn(B, K, model.action_dim, device=cfg.device)

        dt = 1.0 / cfg.num_inference_steps
        anchor_uv = batch["anchor_uv"].to(cfg.device) if cfg.use_region_attn else None

        for i in range(cfg.num_inference_steps):
            t_tensor = torch.full((B, 1), i * dt, device=cfg.device)
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16,
                enabled=(cfg.use_amp and cfg.device.startswith("cuda")),
            ):
                out = model(
                    x_rgb=x_rgb, x_ict=x_ict, ict_mask=ict_mask, x_t=x_t,
                    t=t_tensor, x_pcd=x_pcd, x_robot_state=x_robot_state,
                    robot_state_mask=robot_state_mask, anchor_uv=anchor_uv,
                )
            x_t = x_t + out["v_pred"] * dt

        if cfg.single_hand:
            p_pos, p_o6d, p_glogit = x_t[:, :, 0:3], x_t[:, :, 3:9], x_t[:, :, 9:9 + command_dim]
        else:
            p_pos, p_o6d, p_glogit = x_t[:, :, 0:6], x_t[:, :, 6:18], x_t[:, :, 18:18 + 2 * command_dim]

        p_gprob = (
            p_glogit
            if cfg.hand_action_representation != "grasp_binary"
            else torch.sigmoid(p_glogit)
        )
        n_frames += int(B)

        zero_ratio.append(is_zero_state(x_ict, ict_mask).float().mean().detach().cpu().item())

        for k in range(K):
            # Per-hand masks prevent absent hands from becoming zero-valued GT.
            hands = 1 if cfg.single_hand else 2
            pos_pred_h = p_pos[:, k].reshape(B, hands, 3)
            pos_target_h = y_pos[:, k].reshape(B, hands, 3)
            pos_valid_h = action_valid[:, k, : hands * 3].reshape(B, hands, 3).all(-1)
            pos_error_h = torch.linalg.norm(
                (pos_pred_h - pos_target_h)
                * torch.from_numpy(loader.dataset.pos_std).to(cfg.device).view(1, 1, 3),
                dim=-1,
            )
            pos_err_k[k] += float(
                (pos_error_h * pos_valid_h).sum().cpu()
                / pos_valid_h.sum().clamp(min=1).cpu()
            )

            # Rot Error
            pd = p_o6d[:, k].reshape(B * hands, 6)
            yd = y_o6d[:, k].reshape(B * hands, 6)
            rot_valid_h = action_valid[
                :, k, hands * 3 : hands * 9
            ].reshape(B * hands, 6).all(-1)
            angle = geodesic_deg_from_R(
                rot6d_to_R_batch(pd), rot6d_to_R_batch(yd)
            )
            rot_err_k[k] += float(
                (angle * rot_valid_h).sum().cpu()
                / rot_valid_h.sum().clamp(min=1).cpu()
            )

            if cfg.hand_action_representation != "grasp_binary":
                lower = batch["joint_lower"].to(cfg.device).reshape(B, -1)
                upper = batch["joint_upper"].to(cfg.device).reshape(B, -1)
                ranges = (upper - lower).clamp(min=1e-4)
                q_valid = action_valid[:, k, hands * 9 : hands * (9 + command_dim)]
                joint_mae_k[k] += float(
                    ((torch.abs(p_gprob[:, k] - y_g[:, k]) / ranges) * q_valid).sum().cpu()
                    / q_valid.sum().clamp(min=1).cpu()
                )
            else:
                g_scores_cache[k].append(p_gprob[:, k].cpu().numpy().reshape(-1))
                g_labels_cache[k].append(y_g[:, k].cpu().numpy().reshape(-1))
                grasp_acc_k[k] += float(((p_gprob[:, k] > 0.5).float() == (y_g[:, k] > 0.5).float()).float().mean().cpu())

        # Done accuracy
        if cfg.use_done_in_flow:
            # Done is last dim of x_t after ODE solve — use per-step done
            for k in range(K):
                p_done_k = x_t[:, k, -1:]  # (B, 1)
                y_done_k = y_done[:, k]     # (B, 1)
                done_correct += int(((p_done_k > 0.5).float() == (y_done_k > 0.5).float()).sum().cpu())
                done_total += int(B)
        else:
            # Independent BCE head (trajectory-level, not per-step)
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16,
                enabled=(cfg.use_amp and cfg.device.startswith("cuda")),
            ):
                done_out = model(
                    x_rgb=x_rgb, x_ict=x_ict, ict_mask=ict_mask,
                    x_t=x_t, t=torch.ones(B, 1, device=cfg.device),
                    x_pcd=x_pcd, x_robot_state=x_robot_state,
                    robot_state_mask=robot_state_mask, anchor_uv=anchor_uv,
                )
            p_done_prob = torch.sigmoid(done_out["done_logit"])  # (B, 1)
            y_done_label = y_done[:, -1, :]  # (B, 1) — last step as GT
            done_correct += int(((p_done_prob > 0.5).float() == (y_done_label > 0.5).float()).sum().cpu())
            done_total += int(B)

        n_batches += 1
        if max_batches is not None and n_batches >= max_batches: break

    if n_batches == 0: return {"frames": 0, "pos_err_w_m": 0.0, "rot_err_w_deg": 0.0, "grasp_f1_w": 0.0}

    pos_err_k /= float(n_batches)
    rot_err_k /= float(n_batches)
    joint_mae_k /= float(n_batches)

    g_f1_k = np.zeros((K,), dtype=np.float64)
    for k in range(K):
        if cfg.hand_action_representation != "grasp_binary":
            continue
        sc, lb = np.concatenate(g_scores_cache[k]), np.concatenate(g_labels_cache[k])
        pred = (sc >= 0.5).astype(np.int32)
        lab = (lb >= 0.5).astype(np.int32)
        tp = int(((pred == 1) & (lab == 1)).sum())
        fp = int(((pred == 1) & (lab == 0)).sum())
        fn = int(((pred == 0) & (lab == 1)).sum())
        prec = tp / max(1.0, tp + fp)
        rec = tp / max(1.0, tp + fn)
        g_f1_k[k] = (2.0 * prec * rec) / max(1e-8, prec + rec)

    out = {
        "batches": int(n_batches),
        "frames": int(n_frames),
        "done_acc": float(done_correct / max(1, done_total)),
        "pos_err_k1_m": float(pos_err_k[0]),
        "pos_err_kK_m": float(pos_err_k[-1]),
        "rot_err_k1_deg": float(rot_err_k[0]),
        "rot_err_kK_deg": float(rot_err_k[-1]),
        "pos_err_w_m": float((pos_err_k * w_cpu).sum()),
        "rot_err_w_deg": float((rot_err_k * w_cpu).sum()),
        "grasp_f1_k1": float(g_f1_k[0]),
        "grasp_f1_kK": float(g_f1_k[-1]),
        "grasp_f1_w": float((g_f1_k * w_cpu).sum()),
        "joint_mae_k1_normalized": float(joint_mae_k[0]),
        "joint_mae_kK_normalized": float(joint_mae_k[-1]),
        "joint_mae_w_normalized": float((joint_mae_k * w_cpu).sum()),
        "zero_state_ratio": float(np.mean(zero_ratio)) if zero_ratio else 0.0,
    }
    return out


def eval_ode_inference_isolated(
    model: FlowMatchingModel,
    cfg: TrainConfig,
    epoch: int,
) -> Dict[str, float]:
    """Run CUDA validation in a fresh process.

    The H20 driver stack used by this project can SIGFPE inside libcublasLt
    when validation follows thousands of training GEMMs in the same CUDA
    context.  A fresh process is deterministic and avoids losing the trainer.
    """
    run_dir = Path(cfg.out_dir).resolve()
    pending = run_dir / "eval_snapshots" / f".pending_ep_{epoch:04d}.pt"
    result = run_dir / "eval_snapshots" / f".isolated_ep_{epoch:04d}.json"
    pending.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint_atomic(
        {
            "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "cfg": asdict(cfg),
        },
        str(pending),
    )
    command = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "tools/evaluate_training_snapshot.py"),
        "--snapshot", str(pending),
        "--output", str(result),
    ]
    try:
        subprocess.run(command, cwd=Path(__file__).resolve().parents[1], check=True)
        return json.loads(result.read_text(encoding="utf-8"))
    finally:
        pending.unlink(missing_ok=True)
        result.unlink(missing_ok=True)


# =========================
# ---- Plot helpers --------
# =========================
def plot_train_curve(history: List[Dict[str, float]], out_dir: str):
    fig = plt.figure()
    plt.plot([h["loss"] for h in history], label="Total Loss")
    plt.plot([h.get("l_flow", 0) for h in history], label="Flow Loss")
    plt.plot([h.get("l_foresight", 0) for h in history], label="Foresight Loss")
    plt.plot([h.get("l_contrastive", 0) for h in history], label="Contrastive Loss")
    plt.xlabel("epoch")
    plt.legend()
    plt.title("Train Curve (Multi-Task Flow Matching)")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "train_curve.png"), dpi=180)
    plt.close(fig)

def plot_eval_curve(history: List[Dict[str, float]], out_dir: str):
    evaluated = [
        (epoch, row)
        for epoch, row in enumerate(history, start=1)
        if row.get("frames", 0) > 0
    ]
    if not evaluated:
        return
    epochs = [epoch for epoch, _ in evaluated]
    rows = [row for _, row in evaluated]
    fig = plt.figure()
    plt.plot(epochs, [h["pos_err_k1_m"] * 1000 for h in rows], label="eval_pos@k1(mm)")
    plt.plot(epochs, [h["pos_err_kK_m"] * 1000 for h in rows], label="eval_pos@kK(mm)")
    plt.plot(epochs, [h["rot_err_k1_deg"] for h in rows], label="eval_rot@k1(deg)")
    plt.plot(epochs, [h["rot_err_kK_deg"] for h in rows], label="eval_rot@kK(deg)")
    if any("joint_mae_w_normalized" in h for h in rows):
        plt.plot(
            epochs,
            [h["joint_mae_w_normalized"] for h in rows],
            label="eval_robot_joint_normalized_mae",
        )
    else:
        plt.plot(
            epochs,
            [h["grasp_f1_kK"] * 100.0 for h in rows],
            label="eval_f1@kK(%)",
        )
    plt.xlabel("epoch")
    plt.legend()
    plt.title("Eval Curve (ODE Euler Inference)")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "eval_curve.png"), dpi=180)
    plt.close(fig)


# =========================
# -------- Main ------------
# =========================
def main(cfg: TrainConfig):
    set_seed(cfg.seed)
    if cfg.device.startswith("cuda"):
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    os.makedirs(cfg.out_dir, exist_ok=True)
    run_manifest_path = os.path.join(cfg.out_dir, "run_manifest.json")
    run_manifest = None
    if os.path.isfile(run_manifest_path):
        with open(run_manifest_path, "r", encoding="utf-8") as stream:
            run_manifest = json.load(stream)

    # --- Data Search Logic ---
    if cfg.data_sources is not None:
        # New data_sources mode: e.g. {"aria": 20, "teleop": 10, "teaching": 0}
        train_paths, eval_paths = resolve_data_sources(
            cfg.data_sources, cfg.data_root, cfg.task, cfg.eval_source
        )
        cfg.MPS_PATHS_TRAIN = train_paths
        if eval_paths:
            cfg.MPS_PATHS_EVAL = eval_paths
        source_summary = ", ".join(f"{k}: {v}" for k, v in cfg.data_sources.items())
        console.print(f"[bold green]Data Sources:[/] {source_summary}")
        console.print(f"  Train: {len(cfg.MPS_PATHS_TRAIN)} sessions, Eval: {len(cfg.MPS_PATHS_EVAL)} sessions")
        for p in cfg.MPS_PATHS_TRAIN:
            console.print(f"    [train] {p}")
        for p in cfg.MPS_PATHS_EVAL:
            console.print(f"    [eval]  {p}")
    elif (
        cfg.MPS_PATHS_TRAIN
        and cfg.MPS_PATHS_EVAL
        and all(
            os.path.isdir(path)
            for path in cfg.MPS_PATHS_TRAIN + cfg.MPS_PATHS_EVAL
        )
    ):
        console.print(
            f"[bold green]Explicit Sessions:[/] "
            f"{len(cfg.MPS_PATHS_TRAIN)} train / "
            f"{len(cfg.MPS_PATHS_EVAL)} eval"
        )
        for path in cfg.MPS_PATHS_TRAIN:
            console.print(f"    [train] {path}")
        for path in cfg.MPS_PATHS_EVAL:
            console.print(f"    [eval]  {path}")
    elif getattr(cfg, 'task', None) is not None:
        # Legacy: auto-discover all mps_* sessions
        search_pattern = os.path.join(cfg.data_root, cfg.task, f"mps_{cfg.task}_*_vrs")
        all_sessions = sorted(glob.glob(search_pattern))

        if not all_sessions:
            raise ValueError(f"No sessions found for task '{cfg.task}' in path: {search_pattern}")

        cfg.MPS_PATHS_EVAL = [all_sessions[0]]
        cfg.MPS_PATHS_TRAIN = all_sessions[1:]
        console.print(f"[bold green]Task Mode:[/][white] Found {len(all_sessions)} sessions for {cfg.task}[/]")

    # Convert paths to SessionSpec objects
    training_sessions = [MPSSessions(p) for p in cfg.MPS_PATHS_TRAIN]
    eval_sessions = [MPSSessions(p) for p in cfg.MPS_PATHS_EVAL]

    # --- Handle data_num limit ---
    if cfg.data_num is not None:
        training_sessions = training_sessions[:cfg.data_num]
        console.print(f"[bold yellow]Data Limit:[/] Using {len(training_sessions)} training sessions")

    stats_path = os.path.join(cfg.out_dir, "dataset_stats.json")
    if os.path.exists(stats_path):
        with open(stats_path, "r") as f:
            stats = json.load(f)
        console.print(f"[bold green]Loaded existing stats from:[/] {stats_path}")
    else:
        stats = get_dataset_stats(cfg.MPS_PATHS_TRAIN, cfg)
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=4)
        console.print(f"[bold green]Saved computed stats to:[/] {stats_path}")

    config_save_path = os.path.join(cfg.out_dir, "config.json")
    with open(config_save_path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=4, ensure_ascii=False)

    console.print(_panel_cfg(cfg))

    ds_train = FlowMatchingDataloader(
        sessions=training_sessions,
        image_size=cfg.image_size,
        pred_horizon=cfg.pred_horizon,
        single_hand=cfg.single_hand, single_hand_side=cfg.single_hand_side,
        max_ict=cfg.max_ict,
        img_name=cfg.img_name, centric_mode=cfg.centric_mode,
        frame_mode=cfg.frame_mode, action_mode=cfg.action_mode,
        hand_action_representation=cfg.hand_action_representation,
        robot_sidecar_root=cfg.robot_sidecar_root,
        use_pcd_features=cfg.use_pcd_features,
        use_aux_obj_dynamics=cfg.use_aux_obj_dynamics,
        use_aux_visual_foresight=cfg.use_aux_visual_foresight,
        use_aux_temporal_contrastive=cfg.use_aux_temporal_contrastive,
        enable_augmentation=cfg.enable_augmentation,
        enable_aug_img=cfg.enable_aug_img,
        enable_aug_rrc=cfg.enable_aug_rrc,
        enable_aug_target_jittering=cfg.enable_aug_target_jittering,
        enable_aug_cutout=cfg.enable_aug_cutout,
        enable_aug_temporal_stride=cfg.enable_aug_temporal_stride,
        enable_aug_interpolation=cfg.enable_aug_interpolation,
        hand_tracking_method=cfg.hand_tracking_method,
        use_legacy_image_loading=cfg.use_legacy_image_loading,
        use_legacy_rng=cfg.use_legacy_rng,
        cache_json_in_memory=cfg.cache_json_in_memory,
        cache_image_bytes_in_memory=cfg.cache_image_bytes_in_memory,
        seed=cfg.seed, stats=stats,
    )

    ds_eval = FlowMatchingDataloader(
        sessions=eval_sessions,
        image_size=cfg.image_size,
        pred_horizon=cfg.pred_horizon,
        single_hand=cfg.single_hand, single_hand_side=cfg.single_hand_side,
        max_ict=cfg.max_ict,
        img_name=cfg.img_name, centric_mode=cfg.centric_mode,
        frame_mode=cfg.frame_mode, action_mode=cfg.action_mode,
        hand_action_representation=cfg.hand_action_representation,
        robot_sidecar_root=cfg.robot_sidecar_root,
        use_pcd_features=cfg.use_pcd_features,
        use_aux_obj_dynamics=cfg.use_aux_obj_dynamics,
        use_aux_visual_foresight=cfg.use_aux_visual_foresight,
        use_aux_temporal_contrastive=cfg.use_aux_temporal_contrastive,
        enable_augmentation=False,
        hand_tracking_method=cfg.hand_tracking_method,
        use_legacy_image_loading=cfg.use_legacy_image_loading,
        use_legacy_rng=cfg.use_legacy_rng,
        cache_json_in_memory=cfg.cache_json_in_memory,
        cache_image_bytes_in_memory=cfg.cache_image_bytes_in_memory,
        seed=cfg.seed, stats=stats,
    )

    console.print(Panel(f"train samples={len(ds_train)}  eval samples={len(ds_eval)}", title="DATASET", expand=False))

    loader_kwargs = {
        "num_workers": cfg.num_workers,
        "pin_memory": True,
    }
    if cfg.num_workers > 0:
        loader_kwargs.update({
            "persistent_workers": cfg.persistent_workers,
            "prefetch_factor": cfg.prefetch_factor,
        })
    dl_train = DataLoader(
        ds_train, batch_size=cfg.batch_size, shuffle=True, **loader_kwargs
    )
    dl_eval = DataLoader(
        ds_eval, batch_size=cfg.eval_batch_size or cfg.batch_size,
        shuffle=False, **loader_kwargs
    )

    model = FlowMatchingModel(
        single_hand=cfg.single_hand,
        pred_horizon=cfg.pred_horizon,
        max_ict=cfg.max_ict,
        img_size=cfg.image_size,
        patch_size=cfg.patch_size,
        vision_embed_dim=cfg.vision_embed_dim,
        num_decoder_layers=cfg.num_decoder_layers,
        num_heads=cfg.num_heads,
        mlp_ratio=cfg.mlp_ratio,
        dropout=cfg.dropout,
        horizon_weighting=cfg.model_horizon_weighting,
        horizon_beta=cfg.model_horizon_beta,
        use_pcd_features=cfg.use_pcd_features,
        use_aux_obj_dynamics=cfg.use_aux_obj_dynamics,
        use_aux_visual_foresight=cfg.use_aux_visual_foresight,
        use_aux_temporal_contrastive=cfg.use_aux_temporal_contrastive,
        use_region_attn=cfg.use_region_attn,
        use_pre_norm=cfg.use_pre_norm,
        use_ctx_norm=cfg.use_ctx_norm,
        use_done_in_flow=cfg.use_done_in_flow,
        hand_action_representation=cfg.hand_action_representation,
    ).to(cfg.device)

    if cfg.pretrained_checkpoint and not os.path.exists(
        os.path.join(cfg.out_dir, "latest.pt")
    ):
        transfer = load_compatible_pretrained(
            model,
            cfg.pretrained_checkpoint,
            os.path.join(cfg.out_dir, "pretrained_load_report.json"),
        )
        console.print(
            f"[bold green]Pretrained shared weights:[/] "
            f"{transfer['loaded_key_count']} keys / "
            f"{transfer['loaded_parameter_fraction']:.1%} parameters"
        )

    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    ema = EMAModel(model, decay=cfg.ema_decay) if cfg.use_ema else None
    scaler = torch.amp.GradScaler('cuda', enabled=(cfg.use_amp and cfg.device.startswith("cuda")))

    table = Table(title="Training Progress (Ultimate Multi-Task)", show_lines=False)
    table.add_column("Epoch", justify="right")
    table.add_column("Loss(Fl|Vs|Tc)", justify="center")
    table.add_column("eval_pos_w", justify="right")
    table.add_column("eval_rot_w", justify="right")
    table.add_column("eval_hand_cmd", justify="right")

    history =[]
    best = {"score": 1e9, "epoch": -1}
    evaluations_without_improvement = 0
    t0 = time.time()
    total_steps = int(cfg.epochs * max(1, len(dl_train)))

    start_epoch = 1
    global_step = 0
    completed_epoch = 0
    termination_reason = "max_epochs"

    # =================================================================
    # AUTO-RESUME LOGIC
    # =================================================================
    resume_ckpt_path = os.path.join(cfg.out_dir, "latest.pt")
    if os.path.exists(resume_ckpt_path):
        console.print(f"\n[bold yellow]🔄 Found checkpoint at {resume_ckpt_path}. Resuming training...[/]")
        ckpt = torch.load(resume_ckpt_path, map_location=cfg.device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=True)
        opt.load_state_dict(ckpt["opt"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt.get("global_step", 0)
        best = ckpt.get("best", {"score": 1e9, "epoch": -1})
        evaluations_without_improvement = ckpt.get(
            "evaluations_without_improvement", 0
        )
        if ema is not None: ema = EMAModel(model, decay=cfg.ema_decay)
        if ema is not None and ckpt.get("ema_shadow") is not None:
            ema.load_state_dict(ckpt["ema_shadow"])
        history_path = os.path.join(cfg.out_dir, "train_history.json")
        if os.path.exists(history_path):
            with open(history_path, "r") as f: history = json.load(f)
        console.print(f"[bold green]✅ Successfully resumed from Epoch {start_epoch - 1}. Next: Epoch {start_epoch}[/]\n")

    completed_epoch = start_epoch - 1
    for ep in range(start_epoch, cfg.epochs + 1):
        tr, global_step = train_one_epoch(model, dl_train, opt, scaler, cfg, ep, global_step, total_steps, ema)

        # Persist the just-completed training epoch before potentially expensive
        # multi-step validation. A CUDA inference-kernel failure can then resume
        # without replaying successful optimizer steps.
        if ep % 5 == 0 or ep == cfg.epochs:
            latest_payload = {
                "epoch": ep, "model": model.state_dict(), "opt": opt.state_dict(),
                "cfg": cfg.__dict__, "best": best, "global_step": global_step,
                "run_manifest": run_manifest, "model_weights": "train",
                "ema_shadow": ema.state_dict() if ema is not None else None,
            }
            save_checkpoint_atomic(
                latest_payload, os.path.join(cfg.out_dir, "latest.pt")
            )

        if ep % cfg.eval_every == 0:
            if ema is not None:
                ema.apply_shadow()
            try:
                if os.environ.get("HUMANEGO_ISOLATED_EVAL") == "1":
                    ev = eval_ode_inference_isolated(model, cfg, ep)
                else:
                    ev = eval_ode_inference(
                        model, dl_eval, cfg=cfg,
                        max_batches=cfg.max_eval_batches,
                    )
            finally:
                if ema is not None:
                    ema.restore()

            os.makedirs(os.path.join(cfg.out_dir, "eval_snapshots"), exist_ok=True)
            with open(os.path.join(cfg.out_dir, "eval_snapshots", f"eval_ep_{ep:04d}.json"), "w") as f:
                json.dump({"epoch": ep, **ev}, f, indent=2)
        else:
            ev = {}

        row = {**tr, **ev}
        history.append(row)

        if ep % cfg.eval_every == 0:
            if cfg.hand_action_representation != "grasp_binary":
                score = (
                    ev["pos_err_w_m"]
                    + ev["rot_err_w_deg"] / 100.0
                    + ev.get("joint_mae_w_normalized", 0.0) * 0.10
                )
            else:
                score = (
                    ev["pos_err_w_m"]
                    + ev["rot_err_w_deg"] / 100.0
                    - ev["grasp_f1_w"] * 0.10
                )
            if score < best["score"]:
                best = {"score": float(score), "epoch": int(ep)}
                evaluations_without_improvement = 0
                if ema is not None:
                    ema.apply_shadow()
                best_payload = {
                    "epoch": ep,
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "cfg": cfg.__dict__,
                    "best": best,
                    "global_step": global_step,
                    "selection_metric": "wrist_position_m + rotation_deg/100 + normalized_robot_q_term",
                    "run_manifest": run_manifest,
                    "model_weights": "ema" if ema is not None else "train",
                }
                save_checkpoint_atomic(
                    best_payload, os.path.join(cfg.out_dir, "best.pt")
                )
                if ema is not None:
                    ema.restore()
            else:
                evaluations_without_improvement += 1

            # Refresh the pre-validation recovery point with the selected best
            # and early-stop counter while retaining raw train weights.
            latest_payload = {
                "epoch": ep, "model": model.state_dict(), "opt": opt.state_dict(),
                "cfg": cfg.__dict__, "best": best, "global_step": global_step,
                "run_manifest": run_manifest, "model_weights": "train",
                "evaluations_without_improvement": evaluations_without_improvement,
                "ema_shadow": ema.state_dict() if ema is not None else None,
            }
            save_checkpoint_atomic(
                latest_payload, os.path.join(cfg.out_dir, "latest.pt")
            )

        if ep == 1 or ep % 5 == 0 or ep == cfg.epochs:
            l_str = f"{tr['l_flow']:.3f}|{tr['l_foresight']:.3f}|{tr['l_contrastive']:.3f}"
            eval_pos = f"{ev['pos_err_w_m']:.4f}" if ev else "-"
            eval_rot = f"{ev['rot_err_w_deg']:.1f}" if ev else "-"
            if not ev:
                eval_command = "-"
            elif cfg.hand_action_representation != "grasp_binary":
                eval_command = (
                    f"{ev.get('joint_mae_w_normalized', 0.0):.3f} range"
                )
            else:
                eval_command = f"{ev['grasp_f1_w']*100.0:.1f}%"
            table.add_row(
                f"{ep}",
                l_str,
                eval_pos,
                eval_rot,
                eval_command,
            )

        K = cfg.pred_horizon

        # Calculate weighted values
        val_flow = tr['l_flow'] * cfg.w_flow
        val_vis = tr['l_foresight'] * cfg.w_foresight
        val_cont = tr['l_contrastive'] * cfg.w_contrastive

        val_pos = tr['l_pos'] * cfg.w_pos
        val_rot = tr['l_rot'] * cfg.w_rot
        val_g = tr['l_g'] * cfg.w_g
        val_done = tr['l_done'] * cfg.w_done

        loss_str = (
            f"Flow:{tr['l_flow']:.3f}*{cfg.w_flow}={val_flow:.3f}, "
            f"Vis:{tr['l_foresight']:.4f}*{cfg.w_foresight}={val_vis:.3f}, "
            f"Cont:{tr['l_contrastive']:.3f}*{cfg.w_contrastive}={val_cont:.3f}"
        )
        command_name = "RobotQ" if cfg.hand_action_representation != "grasp_binary" else "Grasp"
        command_weight = cfg.w_hand_joint if cfg.hand_action_representation != "grasp_binary" else cfg.w_g
        flow_detail_str = (
            f"Pos:{tr['l_pos']:.4f}*{cfg.w_pos}={val_pos:.3f}, "
            f"Rot:{tr['l_rot']:.4f}*{cfg.w_rot}={val_rot:.3f}, "
            f"{command_name}:{tr['l_g']:.4f}*{command_weight}={tr['l_g'] * command_weight:.3f}, "
            f"Done:{tr['l_done']:.4f}*{cfg.w_done}={val_done:.3f}"
        )

        console.print(f"[bold cyan]EP {ep:03d}/{cfg.epochs}[/] | [bold white]Total Loss: {tr['loss']:.4f}[/] [dim]({loss_str})[/] | [dim]lr: {tr['lr']:.1e}[/]")
        console.print(f"    [dim]↳ Flow Details: {flow_detail_str}[/]")
        if not ev:
            console.print(f"[dim]Eval skipped (every {cfg.eval_every} epochs)[/]")
        else:
            command_k1 = (
                f"RobotQ nMAE={ev.get('joint_mae_k1_normalized', 0.0):.3f}"
                if cfg.hand_action_representation != "grasp_binary"
                else f"F1={ev['grasp_f1_k1']*100:.1f}%"
            )
            command_kK = (
                f"RobotQ nMAE={ev.get('joint_mae_kK_normalized', 0.0):.3f}"
                if cfg.hand_action_representation != "grasp_binary"
                else f"F1={ev['grasp_f1_kK']*100:.1f}%"
            )
            console.print(f"[dim]Eval k=1   : [/][green]{ev['pos_err_k1_m']*1000:>6.2f}mm[/]  [yellow]{ev['rot_err_k1_deg']:>5.1f}°[/]  [blue]{command_k1}[/]")
            console.print(f"[dim]Eval k={K:<2d}  : [/][green]{ev['pos_err_kK_m']*1000:>6.2f}mm[/]  [yellow]{ev['rot_err_kK_deg']:>5.1f}°[/]  [blue]{command_kK}[/]")
            command_eval = (
                f"RobotQ nMAE={ev.get('joint_mae_w_normalized', 0.0):.4f}"
                if cfg.hand_action_representation != "grasp_binary"
                else f"F1={ev['grasp_f1_w']*100:.1f}%"
            )
            console.print(f"[dim]Eval W 1-{K:<2d}: [/][green]{ev['pos_err_w_m']*1000:>6.2f}mm[/]  [yellow]{ev['rot_err_w_deg']:>5.1f}°[/] [blue]{command_eval}[/][dim]| Score:[/] [bold magenta]{score:.4f}[/]")
        console.print("[dim]" + "-" * 75 + "[/]")

        history_path = os.path.join(cfg.out_dir, "train_history.json")
        history_tmp = history_path + ".tmp"
        with open(history_tmp, "w", encoding="utf-8") as stream:
            json.dump(history, stream, indent=2)
        os.replace(history_tmp, history_path)
        completed_epoch = ep

        if ep % 5 == 0 or ep == cfg.epochs:
            plot_train_curve(history, cfg.out_dir)
            plot_eval_curve(history, cfg.out_dir)

        if (
            cfg.early_stop_patience is not None
            and ep % cfg.eval_every == 0
            and evaluations_without_improvement >= cfg.early_stop_patience
        ):
            termination_reason = "early_stop"
            console.print(
                f"[bold yellow]Early stop:[/] no validation improvement for "
                f"{evaluations_without_improvement} evaluations; best epoch="
                f"{best['epoch']} score={best['score']:.6f}"
            )
            break

        # =====================================================================
        # Professional Visualization Eval (Placeholder implementation trigger)
        # =====================================================================
        if (
            not cfg.skip_visual_eval
            and ((ep % cfg.vis_eval_every == 0) or (ep == cfg.epochs))
        ):
            if ema is not None: ema.apply_shadow()
            console.print(f"\n[bold magenta]>>> Starting Professional Visualization Eval at Epoch {ep} <<<[/]")

            current_ckpt = os.path.join(cfg.out_dir, "latest.pt")
            ckpt_vis = {"epoch": ep, "model": model.state_dict(), "opt": opt.state_dict(), "cfg": cfg.__dict__, "best": best, "global_step": global_step}
            torch.save(ckpt_vis, current_ckpt)

            epoch_render_dir = os.path.join(cfg.out_dir, "eval_render", f"epoch_{ep:04d}")

            if run_teacher_forced_vis is not None:
                for mps_path in cfg.MPS_PATHS_EVAL:
                    session_name = os.path.basename(mps_path.rstrip("/"))
                    render_base_dir = os.path.join(epoch_render_dir, session_name)

                    console.print(f"  -> Rendering session: [cyan]{session_name}[/]")
                    try:
                        run_teacher_forced_vis(
                            model=model,
                            ckpt_path=current_ckpt,
                            mps_path=mps_path,
                            out_dir=render_base_dir,
                            single_hand_side=cfg.single_hand_side,
                            pred_horizon=cfg.pred_horizon,
                            image_size=cfg.image_size,
                            device=cfg.device,
                            centric_mode=cfg.centric_mode,
                            frame_mode=cfg.frame_mode,
                            action_mode=cfg.action_mode,
                            max_ict=cfg.max_ict,
                            stats=stats,
                            num_inference_steps=cfg.num_inference_steps,
                            make_video=cfg.make_video,
                            use_done_in_flow=cfg.use_done_in_flow,
                            hand_action_representation=cfg.hand_action_representation,
                            hand_tracking_method=cfg.hand_tracking_method,
                        )

                    except Exception as e:
                        import traceback
                        console.print(f"[bold red]Evaluator failed on {session_name}:[/] {e}")
                        console.print(traceback.format_exc())
            else:
                console.print("[dim]Evaluator not available yet, skipping rendering...[/]")

            if ema is not None: ema.restore()

    completion = {
        "status": "complete",
        "termination_reason": termination_reason,
        "completed_epoch": int(completed_epoch),
        "configured_epochs": int(cfg.epochs),
        "best": best,
        "global_step": int(global_step),
        "elapsed_s": float(time.time() - t0),
        "run_manifest": run_manifest,
    }
    completion_path = os.path.join(cfg.out_dir, "training_complete.json")
    completion_tmp = completion_path + ".tmp"
    with open(completion_tmp, "w", encoding="utf-8") as stream:
        json.dump(completion, stream, indent=2)
        stream.write("\n")
    os.replace(completion_tmp, completion_path)
    console.print(table)
    console.print(Panel(f"Elapsed: {time.time() - t0:.1f}s", title="DONE", expand=False))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    # ---- Optimization & Schedule ----
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--grad_clip", type=float, default=None)

    parser.add_argument("--use_lr_schedule", action="store_true")
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--min_lr_ratio", type=float, default=None)
    parser.add_argument("--use_amp", action="store_true")

    # ---- EMA ----
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=None)

    # ---- Data & Task ----
    parser.add_argument("--image_size", type=int, nargs=2, default=None, help="--image_size 240 320")
    parser.add_argument("--pred_horizon", type=int, default=None)
    parser.add_argument("--single_hand", action="store_true")
    parser.add_argument("--single_hand_side", type=str, default=None, choices=["left", "right"])

    # ---- Paradigm Ablations ----
    parser.add_argument("--img_name", type=str, default=None, help="Name of image file to load.")
    parser.add_argument("--centric_mode", type=str, default=None, choices=["object_centric", "ego_centric"])
    parser.add_argument("--frame_mode", type=str, default=None, choices=["anchor_frame", "camera_frame"])
    parser.add_argument("--action_mode", type=str, default=None, choices=["absolute", "delta"])
    parser.add_argument(
        "--hand_action_representation", type=str, default=None,
        choices=["grasp_binary", "kaihand_joint_state", "wuji_joint_state"]
    )
    parser.add_argument("--max_ict", type=int, default=None)
    parser.add_argument("--hand_tracking_method", type=str, default=None,
                        choices=["aria_mps", "mediapipe", "wilor", "hamer", "hawor_v3"],
                        help="Hand tracking method to use for training data")

    parser.add_argument("--use_pcd_features", action="store_true", dest="use_pcd_features")
    parser.add_argument("--use_aux_obj_dynamics", action="store_true")
    parser.add_argument("--use_aux_visual_foresight", action="store_true")
    parser.add_argument("--use_aux_temporal_contrastive", action="store_true")
    parser.add_argument("--use_region_attn", action="store_true", dest="use_region_attn")
    parser.add_argument("--use_ot_cfm", action="store_true", help="Enable Minibatch Optimal Transport Matching")

    # ---- Fine-grained Augmentations ----
    parser.add_argument("--enable_augmentation", action="store_true", dest="enable_augmentation")
    parser.add_argument(
        "--disable_augmentation", action="store_true",
        help="Disable all training augmentation for deterministic overfit gates",
    )
    parser.add_argument("--enable_aug_img", action="store_true", dest="enable_aug_img")
    parser.add_argument("--enable_aug_rrc", action="store_true", dest="enable_aug_rrc")
    parser.add_argument("--enable_aug_target_jittering", action="store_true", dest="enable_aug_target_jittering")
    parser.add_argument("--enable_aug_cutout", action="store_true", dest="enable_aug_cutout")
    parser.add_argument("--enable_aug_temporal_stride", action="store_true", dest="enable_aug_temporal_stride")

    # ---- Legacy Compatibility Switches ----
    parser.add_argument("--use_pre_norm", action="store_true", dest="use_pre_norm", default=None)
    parser.add_argument("--use_ctx_norm", action="store_true", dest="use_ctx_norm", default=None)
    parser.add_argument("--use_done_in_flow", action="store_true", dest="use_done_in_flow", default=None)
    parser.add_argument("--use_legacy_image_loading", action="store_true", dest="use_legacy_image_loading", default=None)
    parser.add_argument("--use_legacy_rng", action="store_true", dest="use_legacy_rng", default=None)
    parser.add_argument("--enable_aug_interpolation", action="store_true", dest="enable_aug_interpolation", default=None)

    # ---- Model Hyperparams ----
    parser.add_argument("--patch_size", type=int, default=None)
    parser.add_argument("--vision_embed_dim", type=int, default=None)
    parser.add_argument("--num_decoder_layers", type=int, default=None)
    parser.add_argument("--num_heads", type=int, default=None)
    parser.add_argument("--mlp_ratio", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)

    # ---- Loss Weights ----
    parser.add_argument("--w_flow", type=float, default=None)
    parser.add_argument("--w_pos", type=float, default=None)
    parser.add_argument("--w_rot", type=float, default=None)
    parser.add_argument("--w_g", type=float, default=None)
    parser.add_argument("--w_hand_joint", type=float, default=None)
    parser.add_argument("--w_done", type=float, default=None)
    parser.add_argument("--w_foresight", type=float, default=None)
    parser.add_argument("--w_contrastive", type=float, default=None)

    # ---- State Noise ----
    parser.add_argument("--use_state_noise", action="store_true")
    parser.add_argument("--state_pos_noise_std", type=float, default=None)
    parser.add_argument("--state_rot_noise_deg", type=float, default=None)
    parser.add_argument("--state_grasp_noise_std", type=float, default=None)

    # ---- Flow Matching ----
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--model_h_weighting", type=str, default=None, choices=["uniform", "linear", "exp"])
    parser.add_argument("--model_h_beta", type=float, default=None)

    # ---- Eval Scheduling ----
    parser.add_argument("--eval_every", type=int, default=None)
    parser.add_argument("--vis_eval_every", type=int, default=None)

    # ---- Paths & Runtime ----
    parser.add_argument("--task", type=str, default=None, help="Task name to auto-search data (e.g., 'downstack_cups')")
    parser.add_argument("--exp", type=str, default=None, help="Experiment group name (e.g., 'AuxTraining'). Enables cfg/training/{task}/{exp}/{job}.yaml and runs/{task}/{exp}/{job}/")
    parser.add_argument("--job", type=str, default=None, required=True)
    parser.add_argument("--train_data", type=str, nargs="*", default=None, help="Explicit list of training session paths")
    parser.add_argument("--eval_data", type=str, nargs="*", default=None, help="Explicit list of evaluation session paths")
    parser.add_argument("--data_num", type=int, default=None, help="Limit number of training sessions")
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_eval_batches", type=int, default=None)
    parser.add_argument("--early_stop_patience", type=int, default=None)
    parser.add_argument("--skip_visual_eval", action="store_true")
    parser.add_argument(
        "--out_dir", type=str, default=None,
        help="Explicit run directory; used to isolate overfit gates from formal runs",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--no_pretrained", action="store_true",
        help="Initialize all weights from scratch for an explicit ablation run",
    )

    # --- For YAML control ---
    parser.add_argument("--use_cfg", action="store_true", help="Automatically load yaml from cfg/training/{task}/{exp}/{job}.yaml or cfg/training/{task}/{job}.yaml")

    args = parser.parse_args()
    cfg = TrainConfig()

    # YAML meta keys that come from CLI args — skip them when loading YAML
    _yaml_meta_keys = {"task", "exp", "job"}

    # Automatically find and load YAML if --use_cfg is passed, or if --exp is provided
    _load_cfg = args.use_cfg or (args.exp is not None)
    if _load_cfg and args.task and args.job:
        if args.exp:
            yaml_path = os.path.join("cfg", "training", args.task, args.exp, f"{args.job}.yaml")
        else:
            yaml_path = os.path.join("cfg", "training", args.task, f"{args.job}.yaml")
        if os.path.exists(yaml_path):
            console.print(f"[bold green]Loading config from:[/] {yaml_path}")
            # YAML key aliases: map short/legacy YAML keys to actual config field names
            _yaml_aliases = {
                "enable_aug_jitter": "enable_aug_target_jittering",
                "enable_aug_stride": "enable_aug_temporal_stride",
                "model_h_weighting": "model_horizon_weighting",
                "model_h_beta": "model_horizon_beta",
            }
            with open(yaml_path, "r") as f:
                yaml_cfg = yaml.safe_load(f)
                if yaml_cfg:
                    for k, v in yaml_cfg.items():
                        k = _yaml_aliases.get(k, k)  # resolve alias
                        if k in _yaml_meta_keys:
                            continue  # derived from CLI args, skip
                        if hasattr(cfg, k):
                            if k == "image_size" and isinstance(v, list):
                                v = tuple(v)
                            setattr(cfg, k, v)
        else:
            console.print(f"[bold red]Warning:[/] Config not found at {yaml_path}. Using defaults/args.")
    provided_args = [a.lstrip('-').replace('-', '_') for a in sys.argv if a.startswith('-')]
    arg_dict = vars(args)
    for k, v in arg_dict.items():
        if k in provided_args and v is not None:
            if k == "image_size" and isinstance(v, list):
                v = tuple(v)
            setattr(cfg, k, v)
    if args.out_dir:
        out_dir = args.out_dir
    elif args.exp:
        out_dir = os.path.join("./runs", cfg.task, args.exp, args.job)
    else:
        out_dir = os.path.join("./runs", cfg.task, args.job)
    cfg.out_dir = out_dir
    os.makedirs(cfg.out_dir, exist_ok=True)

    if args.epochs is not None: cfg.epochs = args.epochs
    if args.batch_size is not None: cfg.batch_size = args.batch_size
    if args.eval_batch_size is not None: cfg.eval_batch_size = args.eval_batch_size
    if args.lr is not None: cfg.lr = args.lr
    if args.weight_decay is not None: cfg.weight_decay = args.weight_decay
    if args.grad_clip is not None: cfg.grad_clip = args.grad_clip
    if args.use_lr_schedule: cfg.use_lr_schedule = True
    if args.warmup_steps is not None: cfg.warmup_steps = args.warmup_steps
    if args.min_lr_ratio is not None: cfg.min_lr_ratio = args.min_lr_ratio
    if args.use_amp: cfg.use_amp = True
    if args.use_ema: cfg.use_ema = True
    if args.ema_decay is not None: cfg.ema_decay = args.ema_decay

    if args.image_size is not None: cfg.image_size = tuple(args.image_size)
    if args.pred_horizon is not None: cfg.pred_horizon = args.pred_horizon
    if args.single_hand: cfg.single_hand = True
    if args.single_hand_side is not None: cfg.single_hand_side = args.single_hand_side
    if args.max_ict is not None: cfg.max_ict = args.max_ict

    # Paradigm Ablations
    if args.img_name is not None: cfg.img_name = args.img_name if args.img_name != 'None' else None
    if args.centric_mode is not None: cfg.centric_mode = args.centric_mode
    if args.frame_mode is not None: cfg.frame_mode = args.frame_mode
    if args.action_mode is not None: cfg.action_mode = args.action_mode
    if args.hand_action_representation is not None:
        cfg.hand_action_representation = args.hand_action_representation

    if args.hand_tracking_method is not None: cfg.hand_tracking_method = args.hand_tracking_method
    if args.use_pcd_features: cfg.use_pcd_features = True
    if args.use_aux_obj_dynamics: cfg.use_aux_obj_dynamics = True
    if args.use_aux_visual_foresight: cfg.use_aux_visual_foresight = True
    if args.use_aux_temporal_contrastive: cfg.use_aux_temporal_contrastive = True
    if args.use_region_attn: cfg.use_region_attn = True
    if args.use_ot_cfm: cfg.use_ot_cfm = True

    # Fine-grained augmentations
    if args.enable_augmentation: cfg.enable_augmentation = True
    if args.enable_aug_img: cfg.enable_aug_img = True
    if args.enable_aug_rrc: cfg.enable_aug_rrc = True
    if args.enable_aug_target_jittering: cfg.enable_aug_target_jittering = True
    if args.enable_aug_cutout: cfg.enable_aug_cutout = True
    if args.enable_aug_temporal_stride: cfg.enable_aug_temporal_stride = True
    if args.disable_augmentation:
        cfg.enable_augmentation = False
        cfg.enable_aug_img = False
        cfg.enable_aug_rrc = False
        cfg.enable_aug_target_jittering = False
        cfg.enable_aug_cutout = False
        cfg.enable_aug_temporal_stride = False
        cfg.enable_aug_interpolation = False

    # Legacy switches (only override if explicitly set via CLI)
    if args.use_pre_norm: cfg.use_pre_norm = True
    if args.use_ctx_norm: cfg.use_ctx_norm = True
    if args.use_done_in_flow: cfg.use_done_in_flow = True
    if args.use_legacy_image_loading: cfg.use_legacy_image_loading = True
    if args.use_legacy_rng: cfg.use_legacy_rng = True
    if args.enable_aug_interpolation: cfg.enable_aug_interpolation = True

    if args.w_flow is not None: cfg.w_flow = args.w_flow
    if args.w_pos is not None: cfg.w_pos = args.w_pos
    if args.w_rot is not None: cfg.w_rot = args.w_rot
    if args.w_g is not None: cfg.w_g = args.w_g
    if args.w_hand_joint is not None: cfg.w_hand_joint = args.w_hand_joint
    if args.w_done is not None: cfg.w_done = args.w_done
    if args.w_foresight is not None: cfg.w_foresight = args.w_foresight
    if args.w_contrastive is not None: cfg.w_contrastive = args.w_contrastive

    if args.patch_size is not None: cfg.patch_size = args.patch_size
    if args.vision_embed_dim is not None: cfg.vision_embed_dim = args.vision_embed_dim
    if args.num_decoder_layers is not None: cfg.num_decoder_layers = args.num_decoder_layers
    if args.num_heads is not None: cfg.num_heads = args.num_heads
    if args.mlp_ratio is not None: cfg.mlp_ratio = args.mlp_ratio
    if args.dropout is not None: cfg.dropout = args.dropout

    if args.use_state_noise: cfg.use_state_noise = True
    if args.state_pos_noise_std is not None: cfg.state_pos_noise_std = args.state_pos_noise_std
    if args.state_rot_noise_deg is not None: cfg.state_rot_noise_deg = args.state_rot_noise_deg
    if args.state_grasp_noise_std is not None: cfg.state_grasp_noise_std = args.state_grasp_noise_std

    if args.num_inference_steps is not None: cfg.num_inference_steps = args.num_inference_steps
    if args.model_h_weighting is not None: cfg.model_horizon_weighting = args.model_h_weighting
    if args.model_h_beta is not None: cfg.model_horizon_beta = args.model_h_beta

    if args.eval_every is not None: cfg.eval_every = args.eval_every
    if args.vis_eval_every is not None: cfg.vis_eval_every = args.vis_eval_every

    if args.task is not None: cfg.task = args.task
    if args.train_data is not None and len(args.train_data) > 0: cfg.MPS_PATHS_TRAIN = list(args.train_data)
    if args.eval_data is not None and len(args.eval_data) > 0: cfg.MPS_PATHS_EVAL = list(args.eval_data)
    if args.data_num is not None: cfg.data_num = args.data_num

    if args.num_workers is not None: cfg.num_workers = args.num_workers
    if args.max_train_batches is not None:
        cfg.max_train_batches = args.max_train_batches
    if args.max_eval_batches is not None:
        cfg.max_eval_batches = args.max_eval_batches
    if args.early_stop_patience is not None:
        cfg.early_stop_patience = args.early_stop_patience
    if args.skip_visual_eval:
        cfg.skip_visual_eval = True
    if args.device is not None: cfg.device = args.device
    if args.seed is not None: cfg.seed = args.seed
    if args.no_pretrained:
        cfg.pretrained_checkpoint = None
    main(cfg)

# python -m training.FlowMatchingTrainer --task "serve_bread" --use_cfg --job "00_Baseline"
# python -m training.FlowMatchingTrainer --task "serve_bread" --use_cfg --job "00_Baseline_Egocentric"
# python -m training.FlowMatchingTrainer --task "serve_bread" --use_cfg --job "00_Baseline_WPreoNorm_WCtxNorm_Wotcfm"
# python -m training.FlowMatchingTrainer --task "serve_bread" --use_cfg --job "00_Baseline_WPreoNorm_WCtxNorm_WRegionAttn"
# python -m training.FlowMatchingTrainer --task "serve_bread" --use_cfg --job "07_AuxObjVisCont"
# python -m training.FlowMatchingTrainer --task "serve_bread" --use_cfg --job "08_Egocentric"
# python -m training.FlowMatchingTrainer --task "serve_bread" --use_cfg --job "Exp1_ScalingLaw_01_DataNum1"
# python -m training.FlowMatchingTrainer --task "serve_bread" --use_cfg --job "Exp1_ScalingLaw_02_DataNum3"
# python -m training.FlowMatchingTrainer --task "serve_bread" --use_cfg --job "Exp1_ScalingLaw_03_DataNum5"
# python -m training.FlowMatchingTrainer --task "serve_bread" --use_cfg --job "Exp1_ScalingLaw_04_DataNum10"
# python -m training.FlowMatchingTrainer --task "serve_bread" --use_cfg --job "Exp1_ScalingLaw_05_DataNum20"
# python -m training.FlowMatchingTrainer --task "serve_bread" --use_cfg --job "Exp1_ScalingLaw_07_DataNum40"

# python -m training.FlowMatchingTrainer --use_cfg --task "serve_bread" --exp "HandTracking" --job "01_MediaPipe"
# python -m training.FlowMatchingTrainer --use_cfg --task "serve_bread" --exp "HandTracking" --job "02_WiLoR"
# python -m training.FlowMatchingTrainer --use_cfg --task "serve_bread" --exp "HandTracking" --job "03_HaMeR"

# python -m training.FlowMatchingTrainer --task "adjust_table" --use_cfg --job "00_Baseline"
# python -m training.FlowMatchingTrainer --task "adjust_table" --use_cfg --job "07_AuxObjVisCont"
# python -m training.FlowMatchingTrainer --task "adjust_table" --use_cfg --job "00_Baseline_ObjectCentric_PCA2"

# python -m training.FlowMatchingTrainer --task "downstack_cups" --use_cfg --job "00_Baseline"
# python -m training.FlowMatchingTrainer --task "downstack_cups" --use_cfg --job "00_Baseline_Egocentric"
# python -m training.FlowMatchingTrainer --task "downstack_cups" --use_cfg --job "07_AuxObjVisCont"
# python -m training.FlowMatchingTrainer --task "downstack_cups" --use_cfg --job "08_Egocentric"
# python -m training.FlowMatchingTrainer --task "downstack_cups" --use_cfg --job "00_Baseline_WRegionAttn_Wotcfm"
# python -m training.FlowMatchingTrainer --task "downstack_cups" --use_cfg --job "00_Baseline_WPreNorm_WCtxNorm"
# python -m training.FlowMatchingTrainer --task "downstack_cups" --use_cfg --job "00_Baseline_WRegionAttn_Wotcfm_WPreNorm_WCtxNorm"


# python -m training.FlowMatchingTrainer --task "water_flowers" --use_cfg --job "00_Baseline_v3_WPos5Rot1G10"
# python -m training.FlowMatchingTrainer --task "water_flowers" --use_cfg --job "06_AuxVisCont_v3_WPos5Rot1G10"
# python -m training.FlowMatchingTrainer --task "water_flowers" --use_cfg --job "00_Baseline"
# python -m training.FlowMatchingTrainer --task "water_flowers" --use_cfg --job "07_AuxObjVisCont"



# python -m training.FlowMatchingTrainer \
#     --epochs 400 --batch_size 32 \
#     --lr 1e-4 --weight_decay 1e-2 \
#     --grad_clip 1.0 \
#     --use_lr_schedule --warmup_steps 200 --min_lr_ratio 0.05 \
#     --use_amp \
#     --use_ema --ema_decay 0.999 \
#     \
#     --num_inference_steps 20 \
#     --model_h_weighting "uniform" --model_h_beta 0.0 \
#     --patch_size 16 --vision_embed_dim 384 --num_decoder_layers 6 --num_heads 8 \
#     --mlp_ratio 4.0 --dropout 0.05 \
#     --eval_every 1  --vis_eval_every 50 \
#     \
#     --task "water_flowers" --job "00_Baseline" \
#     --data_num 50  --num_workers 8 \
#     --enable_augmentation --enable_aug_img --enable_aug_rrc --enable_aug_jitter --enable_aug_cutout \
#     \
#     --single_hand --single_hand_side "right" \
#     --pred_horizon 50 \
#     --img_name "rgb_WoArm_WArmObjKpts.png" --image_size 240 320 \
#     --centric_mode "object_centric" --frame_mode "anchor_frame" \
#     --action_mode "absolute" \
#     --max_ict 8 \
#     --use_region_attn \
#     --use_ot_cfm \
#     --w_flow 3.0 --w_pos 5.0 --w_rot 1.0 --w_g 10.0 --w_done 5.0
