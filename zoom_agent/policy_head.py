"""Lightweight Active Zoom Policy Network (~1.90M trainable params, Module 3).

Aligns strictly with ``SmolVLM.md`` Module 3 and Phase 0 Red-Team Audit Section 5:
  1. Owns ALL trainable projections (``visual_proj`` 1x1 Conv C_vis->256, ``style_proj``
     2*C_vis->256) so both SFT & GRPO receive gradients and ``pi_ref`` snapshots are
     100% self-contained.
  2. Spatial state compression: ``Conv3x3(261->128)+GN+SiLU -> Conv3x3(128->128)+GN+SiLU``
     with toggleable style-conditioned FiLM modulation ``Z = (1 + gamma) * Z + beta``
     (zero-initialized final linear layer for exact identity startup; ``use_film=False``
     for clean ``w/o FiLM`` ablation).
  3. Candidate window aggregation via ``roi_align`` on grid ``H_g x W_g = 12 x 20``
     (strictly scaling x by ``W_g`` and y by ``H_g``, ``output_size=(3, 3)``,
     ``spatial_scale=1.0``, ``aligned=True``) + learnable window positional & group
     embeddings + global average-pooled ``No-Crop`` token (index 18).
  4. 2-layer lightweight TransformerEncoder (4 heads, PyTorch C++ fastpath enabled in
     eval mode) for inter-window competition + 19-action logit head (``0..17`` crop
     windows, ``18`` No-Crop/Stop).
  5. Unified action distribution protocol for Stage-1 SFT (raw 19-dim logits) & Stage-2
     GRPO (exact normalized two-step without-replacement masked distribution for
     ``K_max <= 2``, sharing evaluation overlap threshold ``nms_window_iou=0.35``),
     including ``act_greedy``, ``sample_group``, ``log_prob``, exact zero-variance
     analytical entropy ``H(pi_theta)``, and exact analytical ``D_KL(pi_theta || pi_ref)``.
"""

from __future__ import annotations

import copy
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

sys.dont_write_bytecode = True

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import roi_align

from .candidate_grid import CandidateGrid
from .state_encoder import AgentBatchState

ActionTrajectory = Tuple[int, ...]


@dataclass
class PolicyActionOutput:
    """Output of deterministic greedy decoding on a batch of states."""

    actions: List[ActionTrajectory]  # Length B, each tuple of 0..K_max distinct window IDs
    log_probs: Tensor  # [B] exact log pi_theta(a | s)
    action_probs: Tensor  # [B] exp(log_probs)
    step1_probs: Tensor  # [B, N+1] Step-1 categorical probabilities (index N = No-Crop)
    logits: Tensor  # [B, N+1] raw logits


@dataclass
class PolicyGroupSampleOutput:
    """Output of GRPO group sampling (G trajectories per state in batch)."""

    actions: List[List[ActionTrajectory]]  # [B][G] sampled legal trajectories
    log_probs: Tensor  # [B, G] exact log pi_theta(a_{b,g} | s_b)
    step1_probs: Tensor  # [B, N+1] Step-1 categorical probabilities
    logits: Tensor  # [B, N+1] raw logits


def _resolve_num_groups(num_channels: int, preferred: int) -> int:
    g = min(preferred, num_channels)
    while g > 1 and (num_channels % g != 0):
        g -= 1
    return max(1, g)


class ActiveZoomPolicyHead(nn.Module):
    """Only trainable network (~1.90M params at RN101 C_vis=1024) in AZ-Agent."""

    def __init__(
        self,
        grid: Optional[CandidateGrid] = None,
        visual_channels: int = 1024,
        cue_channels: int = 5,
        visual_proj_dim: int = 256,
        style_proj_dim: int = 256,
        conv_hidden_dim: int = 128,
        token_dim: int = 128,
        default_grid_size: Tuple[int, int] = (12, 20),
        roi_output_size: Tuple[int, int] = (3, 3),
        roi_sampling_ratio: int = 2,
        num_transformer_layers: int = 2,
        num_heads: int = 4,
        ffn_dim: int = 512,
        dropout: float = 0.0,
        use_film: bool = True,
        k_max: int = 2,
        nms_window_iou: float = 0.35,
    ) -> None:
        super().__init__()
        if visual_channels < 1:
            raise ValueError(f"visual_channels must be >= 1, got {visual_channels}")
        if cue_channels != 5:
            raise ValueError(f"cue_channels must be 5 (2 RPN + 3 Det), got {cue_channels}")
        if k_max < 0 or k_max > 2:
            raise ValueError(f"ActiveZoomPolicyHead supports k_max in {{0, 1, 2}}, got {k_max}")
        if not (0.0 <= nms_window_iou <= 1.0):
            raise ValueError(f"nms_window_iou must be in [0, 1], got {nms_window_iou}")
        if token_dim % num_heads != 0:
            raise ValueError(f"token_dim ({token_dim}) must be divisible by num_heads ({num_heads})")

        self.grid = grid if grid is not None else CandidateGrid()
        self.num_windows: int = self.grid.num_windows
        self.no_crop_action: int = self.grid.no_crop_action  # 18
        self.num_actions: int = self.num_windows + 1  # 19

        self.visual_channels = int(visual_channels)
        self.cue_channels = int(cue_channels)
        self.visual_proj_dim = int(visual_proj_dim)
        self.style_proj_dim = int(style_proj_dim)
        self.conv_hidden_dim = int(conv_hidden_dim)
        self.token_dim = int(token_dim)
        self.default_grid_size = (int(default_grid_size[0]), int(default_grid_size[1]))
        self.roi_output_size = (int(roi_output_size[0]), int(roi_output_size[1]))
        self.roi_sampling_ratio = int(roi_sampling_ratio)
        self.num_transformer_layers = int(num_transformer_layers)
        self.num_heads = int(num_heads)
        self.ffn_dim = int(ffn_dim)
        self.dropout = float(dropout)
        self.use_film = bool(use_film)
        self.k_max = int(k_max)
        self.nms_window_iou = float(nms_window_iou)

        # Register deterministic candidate window geometry & Step-2 overlap suppression mask
        with torch.inference_mode(False):
            norm_boxes = self.grid.get_normalized_boxes().to(dtype=torch.float32)  # [N, 4] (x1, y1, x2, y2)
            group_ids = torch.tensor(
                [0 if w.group == "grid_3x4" else 1 for w in self.grid.windows],
                dtype=torch.long,
            )  # [N]
            win_iou = self.grid.window_iou_matrix().to(dtype=torch.float32)  # [N, N]

            # Precompute default B=1 RoIAlign boxes on (H_g=12, W_g=20): strictly x * W_g, y * H_g
            dh, dw = self.default_grid_size
            default_scale = torch.tensor([float(dw), float(dh), float(dw), float(dh)], dtype=torch.float32)
            default_rois_b1 = torch.cat(
                [torch.zeros((self.num_windows, 1), dtype=torch.float32), norm_boxes * default_scale],
                dim=1,
            )  # [N, 5]

            step2_valid_base = torch.ones((self.num_windows, self.num_actions), dtype=torch.bool)
            for w1 in range(self.num_windows):
                suppressed = win_iou[w1] > self.nms_window_iou
                suppressed[w1] = True
                step2_valid_base[w1, : self.num_windows] = ~suppressed
                step2_valid_base[w1, self.no_crop_action] = True  # Stop is always legal at Step 2

        self.register_buffer("window_norm_boxes", norm_boxes)
        self.register_buffer("window_group_ids", group_ids)
        self.register_buffer("window_iou_matrix", win_iou)
        self.register_buffer("step2_valid_base", step2_valid_base)
        self.register_buffer("_default_rois_b1", default_rois_b1, persistent=False)
        self._roi_cache: Dict[Tuple[int, int, int, torch.device, torch.dtype], Tensor] = {}
        self._eval_cuda_graph_enabled: bool = False
        self._eval_graph_bundle: Optional[Dict] = None

        # 1. Trainable visual projection: Conv1x1(C_vis -> 256) + GroupNorm + SiLU
        vis_gn_groups = _resolve_num_groups(self.visual_proj_dim, 16)
        self.visual_proj = nn.Sequential(
            nn.Conv2d(self.visual_channels, self.visual_proj_dim, kernel_size=1, bias=False),
            nn.GroupNorm(vis_gn_groups, self.visual_proj_dim),
            nn.SiLU(inplace=True),
        )

        # 2. Spatial state encoder: Conv3x3(261->128)+GN+SiLU -> Conv3x3(128->128)+GN+SiLU
        s_map_ch = self.visual_proj_dim + self.cue_channels  # 261
        stem_gn_groups = _resolve_num_groups(self.conv_hidden_dim, 8)
        self.conv_stem = nn.Sequential(
            nn.Conv2d(s_map_ch, self.conv_hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(stem_gn_groups, self.conv_hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.conv_hidden_dim, self.conv_hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(stem_gn_groups, self.conv_hidden_dim),
            nn.SiLU(inplace=True),
        )

        # 3. Optional scene-difficulty style projection & zero-initialized FiLM modulation
        if self.use_film:
            style_in_dim = 2 * self.visual_channels
            self.style_proj: Optional[nn.Module] = nn.Sequential(
                nn.LayerNorm(style_in_dim),
                nn.Linear(style_in_dim, self.style_proj_dim, bias=True),
                nn.SiLU(inplace=True),
            )
            # Final linear layer zero-initialized => gamma=0, beta=0 => exact identity at init
            self.film_head: Optional[nn.Linear] = nn.Linear(self.style_proj_dim, 2 * self.conv_hidden_dim, bias=True)
            nn.init.zeros_(self.film_head.weight)
            nn.init.zeros_(self.film_head.bias)
            global_in_dim = self.conv_hidden_dim + self.style_proj_dim
        else:
            self.style_proj = None
            self.film_head = None
            global_in_dim = self.conv_hidden_dim

        # 4. Candidate window RoIAlign token projection + global No-Crop token projection
        roi_flat_dim = self.conv_hidden_dim * self.roi_output_size[0] * self.roi_output_size[1]
        self.roi_proj = nn.Sequential(
            nn.Linear(roi_flat_dim, self.token_dim, bias=False),
            nn.LayerNorm(self.token_dim),
        )

        self.global_proj = nn.Sequential(
            nn.Linear(global_in_dim, self.token_dim, bias=False),
            nn.LayerNorm(self.token_dim),
        )

        self.win_pos_embed = nn.Parameter(torch.zeros(1, self.num_windows, self.token_dim))
        self.win_group_embed = nn.Embedding(2, self.token_dim)
        self.stop_token_embed = nn.Parameter(torch.zeros(1, 1, self.token_dim))
        nn.init.trunc_normal_(self.win_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.win_group_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.stop_token_embed, std=0.02)

        # 5. Inter-region competition Transformer (2 layers, 4 heads, C++ fastpath in eval)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.token_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ffn_dim,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=self.num_transformer_layers,
            enable_nested_tensor=False,
        )

        # 6. Action logit head: per-token linear projection + per-action prior bias [19]
        self.action_head = nn.Linear(self.token_dim, 1, bias=False)
        self.action_bias = nn.Parameter(torch.zeros(self.num_actions))

    def config_dict(self) -> Dict:
        return {
            "visual_channels": self.visual_channels,
            "cue_channels": self.cue_channels,
            "visual_proj_dim": self.visual_proj_dim,
            "style_proj_dim": self.style_proj_dim,
            "conv_hidden_dim": self.conv_hidden_dim,
            "token_dim": self.token_dim,
            "default_grid_size": list(self.default_grid_size),
            "roi_output_size": list(self.roi_output_size),
            "roi_sampling_ratio": self.roi_sampling_ratio,
            "roi_spatial_scale": 1.0,
            "roi_aligned": True,
            "num_transformer_layers": self.num_transformer_layers,
            "num_heads": self.num_heads,
            "ffn_dim": self.ffn_dim,
            "dropout": self.dropout,
            "use_film": self.use_film,
            "k_max": self.k_max,
            "nms_window_iou": self.nms_window_iou,
            "num_windows": self.num_windows,
            "num_actions": self.num_actions,
            "no_crop_action": self.no_crop_action,
            "trainable_parameters": self.count_parameters(trainable_only=True),
        }

    def count_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def parameter_breakdown(self) -> "OrderedDict[str, int]":
        """Return exact parameter count per logical submodule plus verified total."""
        def _cnt(mod: Optional[nn.Module]) -> int:
            return sum(p.numel() for p in mod.parameters()) if mod is not None else 0

        breakdown: "OrderedDict[str, int]" = OrderedDict()
        breakdown["visual_proj"] = _cnt(self.visual_proj)
        breakdown["style_proj"] = _cnt(self.style_proj)
        breakdown["conv_stem"] = _cnt(self.conv_stem)
        breakdown["film_head"] = _cnt(self.film_head)
        breakdown["roi_proj"] = _cnt(self.roi_proj)
        breakdown["global_proj"] = _cnt(self.global_proj)
        breakdown["token_embeddings"] = (
            self.win_pos_embed.numel()
            + _cnt(self.win_group_embed)
            + self.stop_token_embed.numel()
        )
        breakdown["transformer"] = _cnt(self.transformer)
        breakdown["action_head"] = _cnt(self.action_head) + self.action_bias.numel()
        part_sum = sum(breakdown.values())
        actual_total = self.count_parameters(trainable_only=False)
        assert part_sum == actual_total, f"Breakdown sum {part_sum} != total {actual_total}"
        breakdown["total"] = actual_total
        return breakdown

    def clone_as_reference(self) -> "ActiveZoomPolicyHead":
        """Create a deeply-copied frozen reference policy pi_ref for GRPO KL regularization."""
        saved_bundle = self._eval_graph_bundle
        self._eval_graph_bundle = None
        try:
            ref = copy.deepcopy(self)
        finally:
            self._eval_graph_bundle = saved_bundle
        ref._roi_cache.clear()
        ref._eval_cuda_graph_enabled = False
        ref._eval_graph_bundle = None
        ref.eval()
        for p in ref.parameters():
            p.requires_grad = False
        return ref

    def release_eval_cuda_graph(self) -> None:
        """Release any captured eval-mode CUDA Graph buffer."""
        self._eval_graph_bundle = None

    def enable_eval_cuda_graph(self, enabled: bool = True) -> "ActiveZoomPolicyHead":
        """Enable or disable exact-FP32 CUDA Graph acceleration for B=1 eval/no_grad inference.

        Eliminates Windows WDDM host kernel-launch overhead (~0.3 ms vs ~1.8 ms eager) while
        running the exact same FP32 layers. Automatically bypassed in train() or grad-enabled
        contexts and automatically invalidated on train(), device/dtype changes, or state loads.
        """
        self._eval_cuda_graph_enabled = bool(enabled)
        if not self._eval_cuda_graph_enabled:
            self.release_eval_cuda_graph()
        return self

    def train(self, mode: bool = True) -> "ActiveZoomPolicyHead":
        if mode:
            self.release_eval_cuda_graph()
            self._roi_cache.clear()
        return super().train(mode)

    def _apply(self, fn):
        self.release_eval_cuda_graph()
        self._roi_cache.clear()
        return super()._apply(fn)

    def _rebuild_default_rois_b1(self) -> None:
        """Synchronize non-persistent _default_rois_b1 buffer from window_norm_boxes."""
        dh, dw = self.default_grid_size
        with torch.inference_mode(False), torch.no_grad():
            norm_boxes = (
                self.window_norm_boxes.clone()
                if self.window_norm_boxes.is_inference()
                else self.window_norm_boxes
            )
            scale = norm_boxes.new_tensor([float(dw), float(dh), float(dw), float(dh)], dtype=norm_boxes.dtype)
            zeros = torch.zeros((self.num_windows, 1), device=norm_boxes.device, dtype=norm_boxes.dtype)
            rebuilt = torch.cat([zeros, norm_boxes * scale], dim=1)
        self.register_buffer("_default_rois_b1", rebuilt, persistent=False)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Refuse silent geometry / overlap-rule mismatch across different CandidateGrid or nms_window_iou configs
        boxes_key = prefix + "window_norm_boxes"
        if boxes_key in state_dict:
            inc_boxes = state_dict[boxes_key].detach().to(
                device=self.window_norm_boxes.device, dtype=self.window_norm_boxes.dtype
            )
            if inc_boxes.shape != self.window_norm_boxes.shape or float(
                (inc_boxes - self.window_norm_boxes).abs().max().item()
            ) > 1e-5:
                raise RuntimeError(
                    "Refusing to load state_dict with mismatched CandidateGrid geometry (window_norm_boxes). "
                    "Construct a matching CandidateGrid or use ActiveZoomPolicyHead.from_checkpoint_dict(ckpt)."
                )
        group_key = prefix + "window_group_ids"
        if group_key in state_dict:
            inc_grp = state_dict[group_key].detach().to(
                device=self.window_group_ids.device, dtype=self.window_group_ids.dtype
            )
            if inc_grp.shape != self.window_group_ids.shape or not torch.equal(inc_grp, self.window_group_ids):
                raise RuntimeError(
                    "Refusing to load state_dict with mismatched CandidateGrid group IDs (window_group_ids). "
                    "Use ActiveZoomPolicyHead.from_checkpoint_dict(ckpt)."
                )
        mask_key = prefix + "step2_valid_base"
        if mask_key in state_dict:
            inc_mask = state_dict[mask_key].detach().to(
                device=self.step2_valid_base.device, dtype=self.step2_valid_base.dtype
            )
            if inc_mask.shape != self.step2_valid_base.shape or not torch.equal(inc_mask, self.step2_valid_base):
                raise RuntimeError(
                    "Refusing to load state_dict with mismatched Step-2 overlap mask (step2_valid_base / "
                    "nms_window_iou). Use ActiveZoomPolicyHead.from_checkpoint_dict(ckpt)."
                )

        self.release_eval_cuda_graph()
        self._roi_cache.clear()
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
        self._rebuild_default_rois_b1()

    def to_checkpoint_dict(self, extra: Optional[Dict] = None) -> Dict:
        """Export self-contained checkpoint dict with full policy & CandidateGrid configuration."""
        return {
            "schema_version": 1,
            "policy_config": self.config_dict(),
            "grid_config": self.grid.config_dict(),
            "state_dict": {k: v.detach().cpu().clone() for k, v in self.state_dict().items()},
            "extra": dict(extra) if extra is not None else {},
        }

    @classmethod
    def from_checkpoint_dict(
        cls,
        ckpt: Dict,
        map_location: torch.device | str = "cpu",
        strict: bool = True,
    ) -> "ActiveZoomPolicyHead":
        """Reconstruct CandidateGrid + ActiveZoomPolicyHead cleanly from ``to_checkpoint_dict``."""
        if "policy_config" not in ckpt or "grid_config" not in ckpt or "state_dict" not in ckpt:
            raise KeyError("Checkpoint dict must contain 'policy_config', 'grid_config', and 'state_dict'.")
        g_cfg = ckpt["grid_config"]
        grid = CandidateGrid(
            grid_rows=int(g_cfg["grid_rows"]),
            grid_cols=int(g_cfg["grid_cols"]),
            grid_win_w=float(g_cfg["grid_win_w"]),
            grid_win_h=float(g_cfg["grid_win_h"]),
        )
        if "windows" in g_cfg:
            expected_xyxy = [tuple(float(v) for v in w["norm_xyxy"]) for w in g_cfg["windows"]]
            actual_xyxy = [tuple(float(v) for v in w.norm_xyxy) for w in grid.windows]
            if len(expected_xyxy) != len(actual_xyxy) or any(
                max(abs(a - b) for a, b in zip(ew, aw)) > 1e-4 for ew, aw in zip(expected_xyxy, actual_xyxy)
            ):
                raise RuntimeError("Reconstructed CandidateGrid windows do not match checkpoint grid_config.")

        p_cfg = ckpt["policy_config"]
        policy = cls(
            grid=grid,
            visual_channels=int(p_cfg["visual_channels"]),
            cue_channels=int(p_cfg["cue_channels"]),
            visual_proj_dim=int(p_cfg["visual_proj_dim"]),
            style_proj_dim=int(p_cfg["style_proj_dim"]),
            conv_hidden_dim=int(p_cfg["conv_hidden_dim"]),
            token_dim=int(p_cfg["token_dim"]),
            default_grid_size=(int(p_cfg["default_grid_size"][0]), int(p_cfg["default_grid_size"][1])),
            roi_output_size=(int(p_cfg["roi_output_size"][0]), int(p_cfg["roi_output_size"][1])),
            roi_sampling_ratio=int(p_cfg["roi_sampling_ratio"]),
            num_transformer_layers=int(p_cfg["num_transformer_layers"]),
            num_heads=int(p_cfg["num_heads"]),
            ffn_dim=int(p_cfg["ffn_dim"]),
            dropout=float(p_cfg["dropout"]),
            use_film=bool(p_cfg["use_film"]),
            k_max=int(p_cfg["k_max"]),
            nms_window_iou=float(p_cfg["nms_window_iou"]),
        )
        policy.load_state_dict(ckpt["state_dict"], strict=strict)
        return policy.to(torch.device(map_location))

    def save_checkpoint(self, path: str | Path, extra: Optional[Dict] = None) -> Path:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.to_checkpoint_dict(extra=extra), str(out_path))
        return out_path

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        map_location: torch.device | str = "cpu",
        strict: bool = True,
    ) -> "ActiveZoomPolicyHead":
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        return cls.from_checkpoint_dict(ckpt, map_location=map_location, strict=strict)

    def _forward_eval_cuda_graph(self, state: AgentBatchState) -> Tensor:
        key = (state.batch_size, state.grid_size, state.device)
        with torch.inference_mode(False), torch.no_grad():
            bundle = self._eval_graph_bundle
            if bundle is None or bundle["key"] != key:
                static_state = state.clone()
                # Warmup on side stream before CUDA Graph capture (PyTorch requirement)
                s = torch.cuda.Stream(device=state.device)
                s.wait_stream(torch.cuda.current_stream(state.device))
                with torch.cuda.stream(s):
                    for _ in range(3):
                        _, _tok = self.forward_features(static_state)
                        _ = self.action_head(_tok).squeeze(-1) + self.action_bias
                torch.cuda.current_stream(state.device).wait_stream(s)

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    _, tok_g = self.forward_features(static_state)
                    static_logits = self.action_head(tok_g).squeeze(-1) + self.action_bias

                bundle = {
                    "key": key,
                    "graph": graph,
                    "static_state": static_state,
                    "static_logits": static_logits,
                }
                self._eval_graph_bundle = bundle

            st_buf: AgentBatchState = bundle["static_state"]
            st_buf.c4_pooled.copy_(state.c4_pooled)
            st_buf.cue_maps.copy_(state.cue_maps)
            st_buf.style_stats.copy_(state.style_stats)
            bundle["graph"].replay()
            return bundle["static_logits"].clone()

    def build_roi_boxes_for_grid(self, batch_size: int, grid_h: int, grid_w: int, dtype: torch.dtype) -> Tensor:
        """Build (or fetch cached) [B * N, 5] RoIAlign boxes (batch_idx, x1*W_g, y1*H_g, x2*W_g, y2*H_g).

        Always constructs regular (non-inference-mode) detached tensors so eval passes under
        ``torch.inference_mode()`` can never poison subsequent ``train()`` backward passes.
        """
        if (
            batch_size == 1
            and grid_h == self.default_grid_size[0]
            and grid_w == self.default_grid_size[1]
            and dtype == self._default_rois_b1.dtype
            and not self._default_rois_b1.is_inference()
        ):
            return self._default_rois_b1

        dev = self.window_norm_boxes.device
        cache_key = (batch_size, grid_h, grid_w, dev, dtype)
        cached = self._roi_cache.get(cache_key)
        if cached is not None and not cached.is_inference():
            return cached

        with torch.inference_mode(False), torch.no_grad():
            norm_boxes = (
                self.window_norm_boxes.clone()
                if self.window_norm_boxes.is_inference()
                else self.window_norm_boxes
            )
            scale = norm_boxes.new_tensor(
                [float(grid_w), float(grid_h), float(grid_w), float(grid_h)],
                dtype=dtype,
            )  # strictly x * W_g, y * H_g
            scaled_boxes = norm_boxes.to(dtype=dtype) * scale  # [N, 4]
            batch_idx = (
                torch.arange(batch_size, device=dev, dtype=dtype)
                .unsqueeze(1)
                .expand(batch_size, self.num_windows)
                .reshape(-1, 1)
            )
            boxes_rep = scaled_boxes.unsqueeze(0).expand(batch_size, self.num_windows, 4).reshape(-1, 4)
            rois = torch.cat([batch_idx, boxes_rep], dim=1)
        if len(self._roi_cache) < 16:
            self._roi_cache[cache_key] = rois
        return rois

    def forward_features(self, state: AgentBatchState) -> Tuple[Tensor, Tensor]:
        """Compute modulated spatial feature Z [B, C_h, H_g, W_g] and contextual tokens [B, 19, D]."""
        if state.visual_channels != self.visual_channels:
            raise ValueError(
                f"Expected visual_channels={self.visual_channels}, got {state.visual_channels}"
            )

        # 1. Project visual features and concatenate with parameter-free 5-ch cues -> [B, 261, H_g, W_g]
        f_vis = self.visual_proj(state.c4_pooled)
        s_map = torch.cat([f_vis, state.cue_maps], dim=1)

        # 2. Spatial conv stem + optional style FiLM modulation
        z = self.conv_stem(s_map)  # [B, C_h, H_g, W_g]
        bsz, c_h, hg, wg = z.shape

        e_style: Optional[Tensor] = None
        if self.use_film:
            assert self.style_proj is not None and self.film_head is not None
            e_style = self.style_proj(state.style_stats)  # [B, style_proj_dim]
            gamma, beta = self.film_head(e_style).view(bsz, 2, c_h, 1, 1).unbind(dim=1)
            z = (1.0 + gamma) * z + beta

        # 3. RoIAlign over N=18 normalized candidate windows on H_g x W_g (x * W_g, y * H_g)
        rois = self.build_roi_boxes_for_grid(bsz, hg, wg, dtype=z.dtype)  # [B*N, 5]
        roi_feats = roi_align(
            z,
            rois,
            output_size=self.roi_output_size,
            spatial_scale=1.0,
            sampling_ratio=self.roi_sampling_ratio,
            aligned=True,
        )  # [B*N, C_h, 3, 3]
        roi_flat = roi_feats.reshape(bsz, self.num_windows, -1)
        win_bias = self.win_pos_embed + self.win_group_embed(self.window_group_ids).unsqueeze(0)
        t_win = self.roi_proj(roi_flat) + win_bias  # [B, N, D]

        # 4. Global average-pooled No-Crop token (index 18)
        z_gap = z.mean(dim=(2, 3))  # [B, C_h]
        g_in = torch.cat([z_gap, e_style], dim=-1) if (self.use_film and e_style is not None) else z_gap
        t_global = self.global_proj(g_in).unsqueeze(1) + self.stop_token_embed  # [B, 1, D]

        # 5. Cross-region competition Transformer over [B, 19, D]
        tokens_in = torch.cat([t_win, t_global], dim=1)
        tokens_out = self.transformer(tokens_in)
        return z, tokens_out

    def forward(self, state: AgentBatchState) -> Tensor:
        """Return raw 19-dim action logits [B, 19] (0..17 candidate windows, 18 No-Crop)."""
        if (
            self._eval_cuda_graph_enabled
            and not self.training
            and not torch.is_grad_enabled()
            and state.batch_size == 1
            and state.device.type == "cuda"
        ):
            return self._forward_eval_cuda_graph(state)
        _, tokens = self.forward_features(state)
        return self.action_head(tokens).squeeze(-1) + self.action_bias

    # =========================================================================
    # Action-Space & Exact Normalized Two-Step Masked Distribution Protocol
    # =========================================================================

    def _resolve_k_max(self, k_max: Optional[int]) -> int:
        k = self.k_max if k_max is None else int(k_max)
        if k < 0 or k > 2:
            raise ValueError(f"k_max must be in {{0, 1, 2}}, got {k}")
        return k

    def _adjust_logits(
        self,
        logits: Tensor,
        temperature: float = 1.0,
        stop_logit_bias: float = 0.0,
    ) -> Tensor:
        if logits.ndim != 2 or logits.shape[1] != self.num_actions:
            raise ValueError(f"Expected logits [B, {self.num_actions}], got {tuple(logits.shape)}")
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}")

        adj = logits if temperature == 1.0 else (logits / float(temperature))
        if stop_logit_bias != 0.0:
            bias_vec = torch.zeros(self.num_actions, device=adj.device, dtype=adj.dtype)
            bias_vec[self.no_crop_action] = float(stop_logit_bias) / float(temperature)
            adj = adj + bias_vec.unsqueeze(0)
        return adj

    def _prepare_logits_and_mask1(
        self,
        logits: Tensor,
        valid_action_mask: Optional[Tensor] = None,
        temperature: float = 1.0,
        stop_logit_bias: float = 0.0,
    ) -> Tuple[Tensor, Tensor]:
        adj = self._adjust_logits(logits, temperature, stop_logit_bias)
        bsz = adj.shape[0]
        if valid_action_mask is None:
            mask1 = torch.ones((bsz, self.num_actions), dtype=torch.bool, device=adj.device)
        else:
            if valid_action_mask.shape != (bsz, self.num_actions):
                raise ValueError(
                    f"valid_action_mask must have shape {(bsz, self.num_actions)}, "
                    f"got {tuple(valid_action_mask.shape)}"
                )
            mask1 = valid_action_mask.to(device=adj.device, dtype=torch.bool).clone()
            mask1[:, self.no_crop_action] = True  # Stop / No-Crop is always legal
        return adj, mask1

    @staticmethod
    def _masked_log_softmax(logits: Tensor, mask: Tensor) -> Tensor:
        """Compute exact log_softmax over ``mask == True`` entries with gradient-safe masking."""
        masked = logits.masked_fill(~mask, float("-inf"))
        return F.log_softmax(masked, dim=-1)

    def step1_log_probs(
        self,
        logits: Tensor,
        valid_action_mask: Optional[Tensor] = None,
        temperature: float = 1.0,
        stop_logit_bias: float = 0.0,
    ) -> Tuple[Tensor, Tensor]:
        """Return (log_p1 [B, 19], mask1 [B, 19]) for Step 1."""
        if valid_action_mask is None:
            adj = self._adjust_logits(logits, temperature, stop_logit_bias)
            mask1 = torch.ones((adj.shape[0], self.num_actions), dtype=torch.bool, device=adj.device)
            return F.log_softmax(adj, dim=-1), mask1
        adj, mask1 = self._prepare_logits_and_mask1(logits, valid_action_mask, temperature, stop_logit_bias)
        return self._masked_log_softmax(adj, mask1), mask1

    def step2_log_probs(
        self,
        logits: Tensor,
        valid_action_mask: Optional[Tensor] = None,
        temperature: float = 1.0,
        stop_logit_bias: float = 0.0,
    ) -> Tuple[Tensor, Tensor]:
        """Return (log_p2 [B, 18, 19], mask2 [B, 18, 19]) conditioned on Step-1 window w1."""
        adj, mask1 = self._prepare_logits_and_mask1(logits, valid_action_mask, temperature, stop_logit_bias)
        step2_base = self.step2_valid_base.to(device=adj.device)  # [18, 19]
        mask2 = step2_base.unsqueeze(0) & mask1.unsqueeze(1)  # [B, 18, 19]
        mask2[:, :, self.no_crop_action] = True
        logits_exp = adj.unsqueeze(1).expand(-1, self.num_windows, -1)  # [B, 18, 19]
        return self._masked_log_softmax(logits_exp, mask2), mask2

    def enumerate_valid_actions(
        self,
        k_max: Optional[int] = None,
        valid_action_mask: Optional[Tensor] = None,
    ) -> List[ActionTrajectory]:
        """Enumerate every legal ordered trajectory a in A_{<=k_max}."""
        k = self._resolve_k_max(k_max)
        if valid_action_mask is not None:
            m = valid_action_mask.detach().cpu().bool().reshape(-1)
            if m.numel() != self.num_actions:
                raise ValueError(f"Expected 1D mask of length {self.num_actions}, got {tuple(valid_action_mask.shape)}")
            allowed_w = [w for w in range(self.num_windows) if bool(m[w].item())]
        else:
            allowed_w = list(range(self.num_windows))

        actions: List[ActionTrajectory] = [()]  # No-Crop
        if k == 0:
            return actions

        for w1 in allowed_w:
            actions.append((w1,))
        if k == 1:
            return actions

        base_cpu = self.step2_valid_base.detach().cpu()
        allowed_set = set(allowed_w)
        for w1 in allowed_w:
            for w2 in range(self.num_windows):
                if w2 in allowed_set and bool(base_cpu[w1, w2].item()):
                    actions.append((w1, w2))
        return actions

    def _validate_single_trajectory(
        self,
        action: Sequence[int],
        k_max: int,
        mask1_row: Tensor,
        mask2_mat: Optional[Tensor],
    ) -> ActionTrajectory:
        traj = tuple(int(w) for w in action)
        if len(traj) > k_max:
            raise ValueError(f"Action trajectory {traj} exceeds k_max={k_max}")
        if len(traj) == 0:
            return traj

        w1 = traj[0]
        if w1 < 0 or w1 >= self.num_windows:
            raise ValueError(f"Invalid Step-1 window ID {w1}; expected [0, {self.num_windows - 1}]")
        if not bool(mask1_row[w1].item()):
            raise ValueError(f"Step-1 window ID {w1} is masked out as illegal.")

        if len(traj) == 2:
            w2 = traj[1]
            if w2 < 0 or w2 >= self.num_windows:
                raise ValueError(f"Invalid Step-2 window ID {w2}; expected [0, {self.num_windows - 1}]")
            assert mask2_mat is not None
            if not bool(mask2_mat[w1, w2].item()):
                raise ValueError(
                    f"Illegal Step-2 transition ({w1} -> {w2}): identical or IoU > {self.nms_window_iou} or masked."
                )
        return traj

    def log_prob_from_logits(
        self,
        logits: Tensor,
        actions: Union[Sequence[Sequence[int]], Sequence[Sequence[Sequence[int]]]],
        valid_action_mask: Optional[Tensor] = None,
        k_max: Optional[int] = None,
        temperature: float = 1.0,
        stop_logit_bias: float = 0.0,
    ) -> Tensor:
        """Compute exact log pi_theta(a | s) from raw logits.

        Supports:
          - 1D list of trajectories [B] -> returns Tensor [B]
          - 1D list of M trajectories with B=1 -> returns Tensor [M]
          - 2D grouped trajectories [B][G] -> returns Tensor [B, G]
        """
        k = self._resolve_k_max(k_max)
        bsz = int(logits.shape[0])
        if len(actions) == 0:
            raise ValueError("actions must be non-empty.")

        # Detect 2D [B][G] vs 1D [B] (or [M] when B=1)
        is_2d = False
        for item in actions:
            if isinstance(item, (list, tuple)) and len(item) > 0 and isinstance(item[0], (list, tuple)):
                is_2d = True
                break

        log_p1, mask1 = self.step1_log_probs(logits, valid_action_mask, temperature, stop_logit_bias)
        log_p2: Optional[Tensor] = None
        mask2: Optional[Tensor] = None
        if k == 2:
            log_p2, mask2 = self.step2_log_probs(logits, valid_action_mask, temperature, stop_logit_bias)

        def _eval_one(b_idx: int, act: Sequence[int]) -> Tensor:
            traj = self._validate_single_trajectory(
                act,
                k_max=k,
                mask1_row=mask1[b_idx],
                mask2_mat=mask2[b_idx] if mask2 is not None else None,
            )
            if k == 0:
                return logits[b_idx].sum() * 0.0
            if len(traj) == 0:
                return log_p1[b_idx, self.no_crop_action]
            if k == 1:
                return log_p1[b_idx, traj[0]]
            # k == 2
            assert log_p2 is not None
            if len(traj) == 1:
                return log_p1[b_idx, traj[0]] + log_p2[b_idx, traj[0], self.no_crop_action]
            return log_p1[b_idx, traj[0]] + log_p2[b_idx, traj[0], traj[1]]

        if is_2d:
            if len(actions) != bsz:
                raise ValueError(f"2D actions outer length {len(actions)} != batch size {bsz}")
            g_size = len(actions[0])
            if g_size < 1:
                raise ValueError("Group size G must be >= 1.")
            rows: List[Tensor] = []
            for b in range(bsz):
                grp = actions[b]
                if len(grp) != g_size:
                    raise ValueError(f"Inconsistent group size at batch {b}: {len(grp)} != {g_size}")
                rows.append(torch.stack([_eval_one(b, a) for a in grp], dim=0))
            return torch.stack(rows, dim=0)  # [B, G]

        # 1D sequence of trajectories
        if len(actions) == bsz:
            return torch.stack([_eval_one(b, actions[b]) for b in range(bsz)], dim=0)  # [B]
        if bsz == 1:
            return torch.stack([_eval_one(0, a) for a in actions], dim=0)  # [M]
        raise ValueError(f"1D actions length {len(actions)} != batch size {bsz} (and B != 1).")

    def log_prob(
        self,
        state: AgentBatchState,
        actions: Union[Sequence[Sequence[int]], Sequence[Sequence[Sequence[int]]]],
        k_max: Optional[int] = None,
        temperature: float = 1.0,
        stop_logit_bias: float = 0.0,
    ) -> Tensor:
        """Compute exact log pi_theta(a | state) preserving autograd gradients."""
        logits = self.forward(state)
        return self.log_prob_from_logits(
            logits=logits,
            actions=actions,
            valid_action_mask=state.valid_action_mask,
            k_max=k_max,
            temperature=temperature,
            stop_logit_bias=stop_logit_bias,
        )

    def act_greedy_from_logits(
        self,
        logits: Tensor,
        valid_action_mask: Optional[Tensor] = None,
        k_max: Optional[int] = None,
        stop_logit_bias: float = 0.0,
    ) -> PolicyActionOutput:
        """Step-wise greedy decoding with exact trajectory log-probability (minimal GPU syncs)."""
        k = self._resolve_k_max(k_max)
        bsz = int(logits.shape[0])
        adj = self._adjust_logits(logits, temperature=1.0, stop_logit_bias=stop_logit_bias)

        if valid_action_mask is None:
            log_p1 = F.log_softmax(adj, dim=-1)
            step1_probs = log_p1.exp()
            mask1: Optional[Tensor] = None
        else:
            adj, mask1 = self._prepare_logits_and_mask1(logits, valid_action_mask, 1.0, stop_logit_bias)
            log_p1 = self._masked_log_softmax(adj, mask1)
            step1_probs = torch.where(mask1, log_p1.exp(), torch.zeros_like(log_p1))

        if k == 0:
            actions = [() for _ in range(bsz)]
            lp = logits.sum(dim=-1) * 0.0
            return PolicyActionOutput(
                actions=actions,
                log_probs=lp,
                action_probs=lp.exp(),
                step1_probs=step1_probs,
                logits=logits,
            )

        c1 = log_p1.argmax(dim=-1)  # [B]
        if k == 1:
            c1_list = c1.tolist()
            actions_out = [() if w1 == self.no_crop_action else (w1,) for w1 in c1_list]
            lp_tensor = log_p1.gather(1, c1.unsqueeze(1)).squeeze(1)
            return PolicyActionOutput(
                actions=actions_out,
                log_probs=lp_tensor,
                action_probs=lp_tensor.exp(),
                step1_probs=step1_probs,
                logits=logits,
            )

        # k == 2: compute Step-2 distribution ONLY for the chosen w1 (single batch tensor op)
        c1_clamped = c1.clamp(max=self.num_windows - 1)  # [B]
        row_mask2 = self.step2_valid_base[c1_clamped]  # [B, 19]
        if mask1 is not None:
            row_mask2 = row_mask2 & mask1
            row_mask2[:, self.no_crop_action] = True
        log_p2_sel = self._masked_log_softmax(adj, row_mask2)  # [B, 19]
        c2 = log_p2_sel.argmax(dim=-1)  # [B]

        lp1_sel = log_p1.gather(1, c1.unsqueeze(1)).squeeze(1)
        lp2_sel = log_p2_sel.gather(1, c2.unsqueeze(1)).squeeze(1)
        is_stop1 = c1 == self.no_crop_action
        lp_tensor = torch.where(is_stop1, lp1_sel, lp1_sel + lp2_sel)

        # Single combined CPU transfer for [c1, c2]
        pairs = torch.stack([c1, c2], dim=1).tolist()
        actions_out = []
        for w1, w2 in pairs:
            if w1 == self.no_crop_action:
                actions_out.append(())
            elif w2 == self.no_crop_action:
                actions_out.append((w1,))
            else:
                actions_out.append((w1, w2))

        return PolicyActionOutput(
            actions=actions_out,
            log_probs=lp_tensor,
            action_probs=lp_tensor.exp(),
            step1_probs=step1_probs,
            logits=logits,
        )

    def act_greedy(
        self,
        state: AgentBatchState,
        k_max: Optional[int] = None,
        stop_logit_bias: float = 0.0,
    ) -> PolicyActionOutput:
        """Run forward pass and return greedy actions with probabilities."""
        logits = self.forward(state)
        return self.act_greedy_from_logits(
            logits=logits,
            valid_action_mask=state.valid_action_mask,
            k_max=k_max,
            stop_logit_bias=stop_logit_bias,
        )

    def sample_group_from_logits(
        self,
        logits: Tensor,
        G: int = 4,
        valid_action_mask: Optional[Tensor] = None,
        k_max: Optional[int] = None,
        temperature: float = 1.0,
        stop_logit_bias: float = 0.0,
        generator: Optional[torch.Generator] = None,
    ) -> PolicyGroupSampleOutput:
        """Sample G independent legal action trajectories per batch element for GRPO."""
        if G < 1:
            raise ValueError(f"Group size G must be >= 1, got {G}")
        k = self._resolve_k_max(k_max)
        bsz = int(logits.shape[0])

        log_p1, mask1 = self.step1_log_probs(logits, valid_action_mask, temperature, stop_logit_bias)
        step1_probs = torch.where(mask1, log_p1.exp(), torch.zeros_like(log_p1))

        if k == 0:
            actions = [[() for _ in range(G)] for _ in range(bsz)]
            lp = (logits.sum(dim=-1) * 0.0).unsqueeze(-1).expand(-1, G)
            return PolicyGroupSampleOutput(actions=actions, log_probs=lp, step1_probs=step1_probs, logits=logits)

        # Sample Step 1: [B, G]
        probs1_safe = step1_probs.clamp(min=0.0)
        c1 = torch.multinomial(probs1_safe, num_samples=G, replacement=True, generator=generator)  # [B, G]

        log_p2: Optional[Tensor] = None
        probs2_safe: Optional[Tensor] = None
        if k == 2:
            log_p2, mask2 = self.step2_log_probs(logits, valid_action_mask, temperature, stop_logit_bias)
            probs2_safe = torch.where(mask2, log_p2.exp(), torch.zeros_like(log_p2)).clamp(min=0.0)  # [B, 18, 19]

        actions_out: List[List[ActionTrajectory]] = []
        lp_rows: List[Tensor] = []

        for b in range(bsz):
            b_acts: List[ActionTrajectory] = []
            b_lps: List[Tensor] = []
            for g in range(G):
                w1 = int(c1[b, g].item())
                if w1 == self.no_crop_action:
                    b_acts.append(())
                    b_lps.append(log_p1[b, self.no_crop_action])
                elif k == 1:
                    b_acts.append((w1,))
                    b_lps.append(log_p1[b, w1])
                else:
                    assert log_p2 is not None and probs2_safe is not None
                    w2 = int(torch.multinomial(probs2_safe[b, w1], num_samples=1, generator=generator).item())
                    if w2 == self.no_crop_action:
                        b_acts.append((w1,))
                        b_lps.append(log_p1[b, w1] + log_p2[b, w1, self.no_crop_action])
                    else:
                        b_acts.append((w1, w2))
                        b_lps.append(log_p1[b, w1] + log_p2[b, w1, w2])
            actions_out.append(b_acts)
            lp_rows.append(torch.stack(b_lps, dim=0))

        return PolicyGroupSampleOutput(
            actions=actions_out,
            log_probs=torch.stack(lp_rows, dim=0),
            step1_probs=step1_probs,
            logits=logits,
        )

    def sample_group(
        self,
        state: AgentBatchState,
        G: int = 4,
        k_max: Optional[int] = None,
        temperature: float = 1.0,
        stop_logit_bias: float = 0.0,
        generator: Optional[torch.Generator] = None,
    ) -> PolicyGroupSampleOutput:
        """Run forward pass once and sample G trajectories per batch element."""
        logits = self.forward(state)
        return self.sample_group_from_logits(
            logits=logits,
            G=G,
            valid_action_mask=state.valid_action_mask,
            k_max=k_max,
            temperature=temperature,
            stop_logit_bias=stop_logit_bias,
            generator=generator,
        )

    def entropy_from_logits(
        self,
        logits: Tensor,
        valid_action_mask: Optional[Tensor] = None,
        k_max: Optional[int] = None,
        temperature: float = 1.0,
        stop_logit_bias: float = 0.0,
    ) -> Tensor:
        """Compute exact analytical policy entropy H(pi_theta(.|s)) [B] (zero MC variance).

        Uses exact chain rule over all legal trajectories A_{<=k_max}:
          H(pi) = H(p_1) + sum_{w1=0}^{N-1} p_1(w1) * H(p_2(. | w1))
        """
        k = self._resolve_k_max(k_max)
        if k == 0:
            return logits.sum(dim=-1) * 0.0

        log_p1, mask1 = self.step1_log_probs(logits, valid_action_mask, temperature, stop_logit_bias)
        p1 = torch.where(mask1, log_p1.exp(), torch.zeros_like(log_p1))
        safe_log_p1 = torch.where(mask1, log_p1, torch.zeros_like(log_p1))
        ent1 = -(p1 * safe_log_p1).sum(dim=-1)  # [B]
        if k == 1:
            return ent1

        log_p2, mask2 = self.step2_log_probs(logits, valid_action_mask, temperature, stop_logit_bias)
        p2 = torch.where(mask2, log_p2.exp(), torch.zeros_like(log_p2))
        safe_log_p2 = torch.where(mask2, log_p2, torch.zeros_like(log_p2))
        ent2_given_w1 = -(p2 * safe_log_p2).sum(dim=-1)  # [B, N]
        return ent1 + (p1[:, : self.num_windows] * ent2_given_w1).sum(dim=-1)

    def entropy(
        self,
        state: AgentBatchState,
        k_max: Optional[int] = None,
        temperature: float = 1.0,
        stop_logit_bias: float = 0.0,
    ) -> Tensor:
        """Compute exact analytical policy entropy H(pi_theta(.|state)) [B]."""
        logits = self.forward(state)
        return self.entropy_from_logits(
            logits=logits,
            valid_action_mask=state.valid_action_mask,
            k_max=k_max,
            temperature=temperature,
            stop_logit_bias=stop_logit_bias,
        )

    def kl_divergence_from_logits(
        self,
        logits: Tensor,
        ref_logits: Tensor,
        valid_action_mask: Optional[Tensor] = None,
        k_max: Optional[int] = None,
        temperature: float = 1.0,
        stop_logit_bias: float = 0.0,
    ) -> Tensor:
        """Compute exact analytical D_KL(pi_theta || pi_ref) [B] over all legal trajectories.

        Uses exact chain rule over A_{<=k_max}:
          KL(pi || pi_ref) = KL(p_1 || q_1) + sum_{w1=0}^{N-1} p_1(w1) * KL(p_2(.|w1) || q_2(.|w1))
        """
        k = self._resolve_k_max(k_max)
        if logits.shape != ref_logits.shape:
            raise ValueError(f"logits shape {tuple(logits.shape)} != ref_logits shape {tuple(ref_logits.shape)}")
        if k == 0:
            return logits.sum(dim=-1) * 0.0

        log_p1, mask1 = self.step1_log_probs(logits, valid_action_mask, temperature, stop_logit_bias)
        log_q1, _ = self.step1_log_probs(ref_logits, valid_action_mask, temperature, stop_logit_bias)
        p1 = torch.where(mask1, log_p1.exp(), torch.zeros_like(log_p1))
        safe_log_p1 = torch.where(mask1, log_p1, torch.zeros_like(log_p1))
        safe_log_q1 = torch.where(mask1, log_q1, torch.zeros_like(log_q1))
        kl1 = (p1 * torch.where(mask1, safe_log_p1 - safe_log_q1, torch.zeros_like(safe_log_p1))).sum(dim=-1)
        if k == 1:
            return kl1

        log_p2, mask2 = self.step2_log_probs(logits, valid_action_mask, temperature, stop_logit_bias)
        log_q2, _ = self.step2_log_probs(ref_logits, valid_action_mask, temperature, stop_logit_bias)
        p2 = torch.where(mask2, log_p2.exp(), torch.zeros_like(log_p2))
        safe_log_p2 = torch.where(mask2, log_p2, torch.zeros_like(log_p2))
        safe_log_q2 = torch.where(mask2, log_q2, torch.zeros_like(log_q2))
        kl2_given_w1 = (
            p2 * torch.where(mask2, safe_log_p2 - safe_log_q2, torch.zeros_like(safe_log_p2))
        ).sum(dim=-1)  # [B, N]
        return kl1 + (p1[:, : self.num_windows] * kl2_given_w1).sum(dim=-1)

    def kl_divergence(
        self,
        state: AgentBatchState,
        ref_policy_or_logits: Union["ActiveZoomPolicyHead", Tensor],
        k_max: Optional[int] = None,
        temperature: float = 1.0,
        stop_logit_bias: float = 0.0,
    ) -> Tensor:
        """Compute exact analytical D_KL(pi_theta || pi_ref) [B] given ref policy module or ref logits."""
        logits = self.forward(state)
        if isinstance(ref_policy_or_logits, nn.Module):
            with torch.no_grad():
                ref_logits = ref_policy_or_logits(state).detach()
        else:
            ref_logits = ref_policy_or_logits.detach()
        return self.kl_divergence_from_logits(
            logits=logits,
            ref_logits=ref_logits,
            valid_action_mask=state.valid_action_mask,
            k_max=k_max,
            temperature=temperature,
            stop_logit_bias=stop_logit_bias,
        )
