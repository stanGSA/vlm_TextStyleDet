"""Parameter-free deterministic multi-source cue encoder & batch state container (Module 1).

Constructs spatial cue maps aligned to a fixed grid (H_g, W_g):
  - M_rpn [2, H_g, W_g]:
      Ch 0: Probability-weighted RPN grey-zone proposal center density (0.10 < p_obj < 0.45),
            divided by 3.0 and clipped to [0, 1].
      Ch 1: Top-3 anchor mean foreground binary entropy H(p)/ln(2), adaptive-avg-pooled to (H_g, W_g).
  - M_det [3, H_g, W_g]:
      Ch 0: Weak/small-object classes (person, bike, rider, motor) low-mid score (0.10 < s < 0.55) max coverage.
      Ch 1: Small predicted box (< 1.5% image area, score > 0.08) max-scaled coverage min(1.0, 1.5*s).
      Ch 2: High-confidence all-class box (score >= 0.65) binary mask.

Phase 1 adds ``AgentBatchState`` and ``StateEncoder.encode_state`` / ``encode_batch_state``:
  - Assembles single or multiple ``GlobalPerceptionOutput`` items into GPU-ready regular
    detached FP32 tensors (strictly ``torch.no_grad()`` + ``.detach()``, NEVER passing
    ``torch.inference_mode()`` tensors into trainable layers):
      1. ``c4_pooled``:   [B, C_vis, H_g, W_g]  (parameter-free AdaptiveAvgPool2d of valid C4)
      2. ``cue_maps``:    [B, 5, H_g, W_g]      (Concat(M_rpn, M_det))
      3. ``style_stats``: [B, 2 * C_vis]        (Concat(channel_mu, channel_log_sigma))

Note: Trainable 1x1 visual feature projection (C_vis -> 256) and style statistics projection
(2*C_vis -> 256) belong strictly to ``ActiveZoomPolicyHead`` so SFT & GRPO gradients flow cleanly.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

sys.dont_write_bytecode = True

import torch
import torch.nn.functional as F
from torch import Tensor

from .detector_wrapper import GlobalPerceptionOutput

# Sky-7 1-indexed IDs for vulnerable/small classes: person(2), bike(4), rider(6), motor(7)
DEFAULT_WEAK_CLASS_IDS = (2, 4, 6, 7)


def ensure_grad_safe_fp32(
    tensor: Tensor,
    device: Optional[torch.device | str] = None,
) -> Tensor:
    """Return a regular (non-inference-mode), detached FP32 tensor on ``device``.

    Regular autograd-compatible detached tensors can safely serve as leaf inputs to
    trainable networks (SFT/GRPO), whereas ``torch.inference_mode()`` tensors cannot
    be saved for backward by trainable layers. Safe even when called inside an outer
    ``torch.inference_mode()`` context (including FP16 -> FP32 conversions).
    """
    target_device = torch.device(device) if device is not None else tensor.device
    with torch.inference_mode(False):
        base = tensor.clone() if tensor.is_inference() else tensor
        out = base.detach().to(device=target_device, dtype=torch.float32)
        if out.is_inference():
            out = out.clone()
    return out


def _ensure_grad_safe_bool(
    tensor: Tensor,
    device: Optional[torch.device | str] = None,
    non_blocking: bool = False,
) -> Tensor:
    """Return a regular (non-inference-mode), detached bool tensor on ``device``."""
    target_device = torch.device(device) if device is not None else tensor.device
    with torch.inference_mode(False):
        base = tensor.clone() if tensor.is_inference() else tensor
        out = base.detach().to(device=target_device, dtype=torch.bool, non_blocking=non_blocking)
        if out.is_inference():
            out = out.clone()
    return out


@dataclass
class AgentBatchState:
    """Unified GPU-ready batched state container bridging frozen perception and trainable policy.

    All tensor fields are strictly validated to be regular (non-inference-mode) detached FP32
    tensors residing on the same ``device``.
    """

    c4_pooled: Tensor  # [B, C_vis, H_g, W_g], regular detached FP32
    cue_maps: Tensor  # [B, 5, H_g, W_g], regular detached FP32
    style_stats: Tensor  # [B, 2 * C_vis], regular detached FP32
    orig_sizes: Tuple[Tuple[int, int], ...] = ()  # ((orig_h, orig_w), ...)
    resized_sizes: Tuple[Tuple[int, int], ...] = ()  # ((img_h, img_w), ...)
    canvas_sizes: Tuple[Tuple[int, int], ...] = ()  # ((pad_h, pad_w), ...)
    image_ids: Tuple[str, ...] = ()
    valid_action_mask: Optional[Tensor] = None  # optional [B, num_actions] bool on same device
    coordinate_system: str = "orig_0based_xyxy"

    def __post_init__(self) -> None:
        if self.c4_pooled.ndim != 4:
            raise ValueError(f"c4_pooled must be 4D [B, C, H_g, W_g], got {tuple(self.c4_pooled.shape)}")
        if self.cue_maps.ndim != 4:
            raise ValueError(f"cue_maps must be 4D [B, 5, H_g, W_g], got {tuple(self.cue_maps.shape)}")
        if self.style_stats.ndim != 2:
            raise ValueError(f"style_stats must be 2D [B, 2*C], got {tuple(self.style_stats.shape)}")

        bsz, c_vis, hg, wg = self.c4_pooled.shape
        if bsz < 1:
            raise ValueError(f"Batch size must be >= 1, got {bsz}")
        if self.cue_maps.shape[0] != bsz or self.style_stats.shape[0] != bsz:
            raise ValueError(
                f"Batch size mismatch: c4={bsz}, cues={self.cue_maps.shape[0]}, style={self.style_stats.shape[0]}"
            )
        if self.cue_maps.shape[1] != 5:
            raise ValueError(f"cue_maps must have 5 channels (2 RPN + 3 Det), got {self.cue_maps.shape[1]}")
        if tuple(self.cue_maps.shape[-2:]) != (hg, wg):
            raise ValueError(
                f"Grid size mismatch: c4={(hg, wg)} vs cue_maps={tuple(self.cue_maps.shape[-2:])}"
            )
        if self.style_stats.shape[1] != 2 * c_vis:
            raise ValueError(
                f"style_stats dim must equal 2 * C_vis ({2 * c_vis}), got {self.style_stats.shape[1]}"
            )

        for name, t in (
            ("c4_pooled", self.c4_pooled),
            ("cue_maps", self.cue_maps),
            ("style_stats", self.style_stats),
        ):
            if t.is_inference():
                raise RuntimeError(
                    f"AgentBatchState.{name} is a torch.inference_mode() tensor, which is forbidden as "
                    "input to trainable layers. Produce states under torch.no_grad() + .detach() or "
                    "convert via ensure_grad_safe_fp32()."
                )
            if t.requires_grad:
                raise RuntimeError(
                    f"AgentBatchState.{name} has requires_grad=True, violating the detached state contract. "
                    "Pass detached tensors via .detach() or ensure_grad_safe_fp32()."
                )
            if t.dtype != torch.float32:
                raise TypeError(f"AgentBatchState.{name} must be torch.float32, got {t.dtype}")

        dev = self.c4_pooled.device
        if self.cue_maps.device != dev or self.style_stats.device != dev:
            raise ValueError(
                f"Device mismatch in AgentBatchState: c4={dev}, cues={self.cue_maps.device}, "
                f"style={self.style_stats.device}"
            )

        if self.valid_action_mask is not None:
            if self.valid_action_mask.is_inference():
                raise RuntimeError("AgentBatchState.valid_action_mask must not be an inference_mode tensor.")
            if self.valid_action_mask.ndim != 2 or self.valid_action_mask.shape[0] != bsz:
                raise ValueError(
                    f"valid_action_mask must be [B, num_actions] with B={bsz}, "
                    f"got {tuple(self.valid_action_mask.shape)}"
                )
            if self.valid_action_mask.dtype != torch.bool:
                raise TypeError(f"valid_action_mask must be torch.bool, got {self.valid_action_mask.dtype}")
            if self.valid_action_mask.device != dev:
                raise ValueError(
                    f"valid_action_mask device {self.valid_action_mask.device} != state device {dev}"
                )

        if self.orig_sizes and len(self.orig_sizes) != bsz:
            raise ValueError(f"orig_sizes length {len(self.orig_sizes)} != batch size {bsz}")
        if self.resized_sizes and len(self.resized_sizes) != bsz:
            raise ValueError(f"resized_sizes length {len(self.resized_sizes)} != batch size {bsz}")
        if self.canvas_sizes and len(self.canvas_sizes) != bsz:
            raise ValueError(f"canvas_sizes length {len(self.canvas_sizes)} != batch size {bsz}")
        if self.image_ids and len(self.image_ids) != bsz:
            raise ValueError(f"image_ids length {len(self.image_ids)} != batch size {bsz}")

    @property
    def batch_size(self) -> int:
        return int(self.c4_pooled.shape[0])

    @property
    def visual_channels(self) -> int:
        return int(self.c4_pooled.shape[1])

    @property
    def style_dim(self) -> int:
        return int(self.style_stats.shape[1])

    @property
    def grid_size(self) -> Tuple[int, int]:
        return (int(self.c4_pooled.shape[2]), int(self.c4_pooled.shape[3]))

    @property
    def device(self) -> torch.device:
        return self.c4_pooled.device

    def to(self, device: torch.device | str, non_blocking: bool = False) -> "AgentBatchState":
        target = torch.device(device)
        with torch.inference_mode(False):
            c4 = self.c4_pooled.detach().to(device=target, dtype=torch.float32, non_blocking=non_blocking)
            cues = self.cue_maps.detach().to(device=target, dtype=torch.float32, non_blocking=non_blocking)
            style = self.style_stats.detach().to(device=target, dtype=torch.float32, non_blocking=non_blocking)
            mask = (
                _ensure_grad_safe_bool(self.valid_action_mask, device=target, non_blocking=non_blocking)
                if self.valid_action_mask is not None
                else None
            )
        return AgentBatchState(
            c4_pooled=c4,
            cue_maps=cues,
            style_stats=style,
            orig_sizes=self.orig_sizes,
            resized_sizes=self.resized_sizes,
            canvas_sizes=self.canvas_sizes,
            image_ids=self.image_ids,
            valid_action_mask=mask,
            coordinate_system=self.coordinate_system,
        )

    def detach(self) -> "AgentBatchState":
        return AgentBatchState(
            c4_pooled=ensure_grad_safe_fp32(self.c4_pooled),
            cue_maps=ensure_grad_safe_fp32(self.cue_maps),
            style_stats=ensure_grad_safe_fp32(self.style_stats),
            orig_sizes=self.orig_sizes,
            resized_sizes=self.resized_sizes,
            canvas_sizes=self.canvas_sizes,
            image_ids=self.image_ids,
            valid_action_mask=(
                _ensure_grad_safe_bool(self.valid_action_mask) if self.valid_action_mask is not None else None
            ),
            coordinate_system=self.coordinate_system,
        )

    def clone(self) -> "AgentBatchState":
        with torch.inference_mode(False):
            c4 = self.c4_pooled.detach().clone()
            cues = self.cue_maps.detach().clone()
            style = self.style_stats.detach().clone()
            mask = self.valid_action_mask.detach().clone() if self.valid_action_mask is not None else None
        return AgentBatchState(
            c4_pooled=c4,
            cue_maps=cues,
            style_stats=style,
            orig_sizes=self.orig_sizes,
            resized_sizes=self.resized_sizes,
            canvas_sizes=self.canvas_sizes,
            image_ids=self.image_ids,
            valid_action_mask=mask,
            coordinate_system=self.coordinate_system,
        )

    def to_cache_dict(self, storage_dtype: torch.dtype = torch.float16) -> Dict:
        """Serialize detached state tensors to compact CPU dict for Phase 2/3 offline caching."""
        return {
            "c4_pooled": self.c4_pooled.detach().cpu().to(dtype=storage_dtype),
            "cue_maps": self.cue_maps.detach().cpu().to(dtype=storage_dtype),
            "style_stats": self.style_stats.detach().cpu().to(dtype=torch.float32),
            "orig_sizes": list(self.orig_sizes),
            "resized_sizes": list(self.resized_sizes),
            "canvas_sizes": list(self.canvas_sizes),
            "image_ids": list(self.image_ids),
            "valid_action_mask": (
                self.valid_action_mask.detach().cpu() if self.valid_action_mask is not None else None
            ),
            "coordinate_system": self.coordinate_system,
        }

    @classmethod
    def from_cache_dict(
        cls,
        payload: Dict,
        device: torch.device | str = "cpu",
    ) -> "AgentBatchState":
        """Restore a cached state dict into regular detached FP32 tensors on ``device``."""
        target = torch.device(device)
        mask = payload.get("valid_action_mask")
        if mask is not None:
            mask = _ensure_grad_safe_bool(mask, device=target)
        return cls(
            c4_pooled=ensure_grad_safe_fp32(payload["c4_pooled"], device=target),
            cue_maps=ensure_grad_safe_fp32(payload["cue_maps"], device=target),
            style_stats=ensure_grad_safe_fp32(payload["style_stats"], device=target),
            orig_sizes=tuple(tuple(int(v) for v in s) for s in payload.get("orig_sizes", ())),
            resized_sizes=tuple(tuple(int(v) for v in s) for s in payload.get("resized_sizes", ())),
            canvas_sizes=tuple(tuple(int(v) for v in s) for s in payload.get("canvas_sizes", ())),
            image_ids=tuple(str(x) for x in payload.get("image_ids", ())),
            valid_action_mask=mask,
            coordinate_system=str(payload.get("coordinate_system", "orig_0based_xyxy")),
        )

    @classmethod
    def collate(
        cls,
        states: Sequence["AgentBatchState"],
        device: Optional[torch.device | str] = None,
    ) -> "AgentBatchState":
        """Concatenate a sequence of ``AgentBatchState`` instances along the batch dimension."""
        if not states:
            raise ValueError("Cannot collate an empty sequence of AgentBatchState.")
        coord_systems = {s.coordinate_system for s in states}
        if len(coord_systems) > 1:
            raise ValueError(
                f"Cannot collate AgentBatchState items with mismatched coordinate_system: {sorted(coord_systems)}"
            )
        target = torch.device(device) if device is not None else states[0].device

        with torch.inference_mode(False):
            c4 = torch.cat([ensure_grad_safe_fp32(s.c4_pooled, device=target) for s in states], dim=0)
            cues = torch.cat([ensure_grad_safe_fp32(s.cue_maps, device=target) for s in states], dim=0)
            styles = torch.cat([ensure_grad_safe_fp32(s.style_stats, device=target) for s in states], dim=0)

            has_orig = all(len(s.orig_sizes) == s.batch_size for s in states)
            has_res = all(len(s.resized_sizes) == s.batch_size for s in states)
            has_canv = all(len(s.canvas_sizes) == s.batch_size for s in states)
            has_ids = all(len(s.image_ids) == s.batch_size for s in states)

            any_mask = any(s.valid_action_mask is not None for s in states)
            valid_mask: Optional[Tensor] = None
            if any_mask:
                first_mask = next(s.valid_action_mask for s in states if s.valid_action_mask is not None)
                num_actions = int(first_mask.shape[1])
                mask_parts: List[Tensor] = []
                for s in states:
                    if s.valid_action_mask is None:
                        mask_parts.append(torch.ones((s.batch_size, num_actions), dtype=torch.bool, device=target))
                    else:
                        mask_parts.append(_ensure_grad_safe_bool(s.valid_action_mask, device=target))
                valid_mask = torch.cat(mask_parts, dim=0)

        return cls(
            c4_pooled=c4,
            cue_maps=cues,
            style_stats=styles,
            orig_sizes=tuple(sz for s in states for sz in s.orig_sizes) if has_orig else (),
            resized_sizes=tuple(sz for s in states for sz in s.resized_sizes) if has_res else (),
            canvas_sizes=tuple(sz for s in states for sz in s.canvas_sizes) if has_canv else (),
            image_ids=tuple(iid for s in states for iid in s.image_ids) if has_ids else (),
            valid_action_mask=valid_mask,
            coordinate_system=states[0].coordinate_system,
        )


class StateEncoder:
    """Deterministic spatial uncertainty & cue constructor on grid (H_g, W_g)."""

    def __init__(
        self,
        grid_size: Tuple[int, int] = (12, 20),
        weak_class_ids: Sequence[int] = DEFAULT_WEAK_CLASS_IDS,
        rpn_grey_range: Tuple[float, float] = (0.10, 0.45),
        weak_score_range: Tuple[float, float] = (0.10, 0.55),
        small_area_ratio: float = 0.015,
        small_min_score: float = 0.08,
        high_conf_thresh: float = 0.65,
    ) -> None:
        if grid_size[0] < 2 or grid_size[1] < 2:
            raise ValueError(f"Invalid grid_size: {grid_size}")
        self.grid_h, self.grid_w = int(grid_size[0]), int(grid_size[1])
        self.weak_class_ids = tuple(int(c) for c in weak_class_ids)
        self.rpn_grey_lo, self.rpn_grey_hi = float(rpn_grey_range[0]), float(rpn_grey_range[1])
        self.weak_score_lo, self.weak_score_hi = float(weak_score_range[0]), float(weak_score_range[1])
        self.small_area_ratio = float(small_area_ratio)
        self.small_min_score = float(small_min_score)
        self.high_conf_thresh = float(high_conf_thresh)

    def config_dict(self) -> Dict:
        return {
            "grid_size": [self.grid_h, self.grid_w],
            "weak_class_ids": list(self.weak_class_ids),
            "rpn_grey_range": [self.rpn_grey_lo, self.rpn_grey_hi],
            "weak_score_range": [self.weak_score_lo, self.weak_score_hi],
            "small_area_ratio": self.small_area_ratio,
            "small_min_score": self.small_min_score,
            "high_conf_thresh": self.high_conf_thresh,
        }

    @torch.no_grad()
    def encode_cue_maps(
        self,
        perc: GlobalPerceptionOutput,
        device: torch.device | str = "cpu",
    ) -> Dict[str, Tensor]:
        """Return M_rpn [2, H_g, W_g] and M_det [3, H_g, W_g] on ``device``."""
        orig_h, orig_w = perc.orig_size
        hg, wg = self.grid_h, self.grid_w
        out_dev = torch.device(device)

        m_rpn = torch.zeros((2, hg, wg), dtype=torch.float32)

        props = perc.proposals
        p_probs = perc.proposal_probs
        if props.numel() > 0:
            grey_mask = (p_probs > self.rpn_grey_lo) & (p_probs < self.rpn_grey_hi)
            if grey_mask.any():
                g_boxes = props[grey_mask]
                g_weights = p_probs[grey_mask]
                cx = ((g_boxes[:, 0] + g_boxes[:, 2]) * 0.5 / max(orig_w, 1) * wg).long().clamp(0, wg - 1)
                cy = ((g_boxes[:, 1] + g_boxes[:, 3]) * 0.5 / max(orig_h, 1) * hg).long().clamp(0, hg - 1)
                flat_idx = cy * wg + cx
                m_rpn[0].view(-1).scatter_add_(0, flat_idx, g_weights)
                m_rpn[0] = (m_rpn[0] / 3.0).clamp(0.0, 1.0)

        p_map = perc.rpn_prob_map.clamp(1e-6, 1.0 - 1e-6)  # [1, A, H_f, W_f]
        entropy = -(p_map * torch.log(p_map) + (1.0 - p_map) * torch.log(1.0 - p_map)) / math.log(2.0)
        topk_k = min(3, entropy.shape[1])
        ent_spatial = entropy.topk(topk_k, dim=1).values.mean(dim=1, keepdim=True)
        m_rpn[1] = F.adaptive_avg_pool2d(ent_spatial, output_size=(hg, wg))[0, 0].cpu()

        m_det = torch.zeros((3, hg, wg), dtype=torch.float32)
        boxes = perc.detections["boxes"]
        scores = perc.detections["scores"]
        labels = perc.detections["labels"]
        img_area = float(max(orig_h * orig_w, 1))

        if boxes.numel() > 0:
            weak_set = torch.tensor(self.weak_class_ids, dtype=labels.dtype)
            is_weak = torch.isin(labels, weak_set)
            box_areas = ((boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)) / img_area

            gx1 = (boxes[:, 0] / max(orig_w, 1) * wg).floor().long().clamp(0, wg - 1)
            gy1 = (boxes[:, 1] / max(orig_h, 1) * hg).floor().long().clamp(0, hg - 1)
            gx2 = (boxes[:, 2] / max(orig_w, 1) * wg).ceil().long().clamp(1, wg)
            gy2 = (boxes[:, 3] / max(orig_h, 1) * hg).ceil().long().clamp(1, hg)

            for i in range(boxes.shape[0]):
                s = float(scores[i].item())
                x_s, x_e = int(gx1[i].item()), max(int(gx1[i].item()) + 1, int(gx2[i].item()))
                y_s, y_e = int(gy1[i].item()), max(int(gy1[i].item()) + 1, int(gy2[i].item()))

                if bool(is_weak[i].item()) and (self.weak_score_lo < s < self.weak_score_hi):
                    m_det[0, y_s:y_e, x_s:x_e] = torch.maximum(
                        m_det[0, y_s:y_e, x_s:x_e], torch.tensor(s, dtype=torch.float32)
                    )
                if float(box_areas[i].item()) < self.small_area_ratio and s > self.small_min_score:
                    m_det[1, y_s:y_e, x_s:x_e] = torch.maximum(
                        m_det[1, y_s:y_e, x_s:x_e], torch.tensor(min(1.0, s * 1.5), dtype=torch.float32)
                    )
                if s >= self.high_conf_thresh:
                    m_det[2, y_s:y_e, x_s:x_e] = 1.0

        return {
            "m_rpn": ensure_grad_safe_fp32(m_rpn, device=out_dev),
            "m_det": ensure_grad_safe_fp32(m_det, device=out_dev),
        }

    @torch.no_grad()
    def encode_state(
        self,
        perc: GlobalPerceptionOutput,
        device: Optional[torch.device | str] = None,
        image_id: Optional[str] = None,
        valid_action_mask: Optional[Tensor] = None,
    ) -> AgentBatchState:
        """Assemble a single ``GlobalPerceptionOutput`` into a B=1 ``AgentBatchState``."""
        return self.encode_batch_state(
            perceptions=[perc],
            device=device,
            image_ids=[image_id] if image_id is not None else None,
            valid_action_mask=valid_action_mask,
        )

    @torch.no_grad()
    def encode_batch_state(
        self,
        perceptions: Union[GlobalPerceptionOutput, Sequence[GlobalPerceptionOutput]],
        device: Optional[torch.device | str] = None,
        image_ids: Optional[Sequence[str]] = None,
        valid_action_mask: Optional[Tensor] = None,
    ) -> AgentBatchState:
        """Assemble 1 or more ``GlobalPerceptionOutput`` objects into a GPU-ready ``AgentBatchState``.

        Strictly runs under ``torch.no_grad()`` and emits regular detached FP32 tensors so
        trainable layers in ``ActiveZoomPolicyHead`` can backpropagate cleanly.
        """
        if isinstance(perceptions, GlobalPerceptionOutput):
            perc_list: List[GlobalPerceptionOutput] = [perceptions]
        else:
            perc_list = list(perceptions)

        if not perc_list:
            raise ValueError("perceptions must be a GlobalPerceptionOutput or non-empty sequence.")

        target_dev = torch.device(device) if device is not None else perc_list[0].c4_feature.device

        c4_parts: List[Tensor] = []
        cue_parts: List[Tensor] = []
        style_parts: List[Tensor] = []
        orig_sizes: List[Tuple[int, int]] = []
        resized_sizes: List[Tuple[int, int]] = []
        canvas_sizes: List[Tuple[int, int]] = []

        for perc in perc_list:
            c4 = perc.c4_feature
            if c4.ndim == 3:
                c4 = c4.unsqueeze(0)
            if c4.ndim != 4 or c4.shape[0] != 1:
                raise ValueError(f"Expected perc.c4_feature [1, C, H, W], got {tuple(perc.c4_feature.shape)}")

            c4_safe = ensure_grad_safe_fp32(c4, device=target_dev)
            c4_pool = F.adaptive_avg_pool2d(c4_safe, output_size=(self.grid_h, self.grid_w))
            c4_parts.append(ensure_grad_safe_fp32(c4_pool, device=target_dev))

            cues = self.encode_cue_maps(perc, device=target_dev)
            cue_5ch = torch.cat([cues["m_rpn"], cues["m_det"]], dim=0).unsqueeze(0)  # [1, 5, H_g, W_g]
            cue_parts.append(ensure_grad_safe_fp32(cue_5ch, device=target_dev))

            mu = perc.channel_mu
            log_sigma = perc.channel_log_sigma
            if mu.ndim == 1:
                mu = mu.unsqueeze(0)
            if log_sigma.ndim == 1:
                log_sigma = log_sigma.unsqueeze(0)
            if mu.shape != log_sigma.shape or mu.shape[0] != 1:
                raise ValueError(
                    f"Expected channel_mu & channel_log_sigma shape [1, C], got {tuple(mu.shape)} & {tuple(log_sigma.shape)}"
                )
            style_vec = torch.cat(
                [ensure_grad_safe_fp32(mu, device=target_dev), ensure_grad_safe_fp32(log_sigma, device=target_dev)],
                dim=-1,
            )  # [1, 2 * C_vis]
            style_parts.append(ensure_grad_safe_fp32(style_vec, device=target_dev))

            orig_sizes.append((int(perc.orig_size[0]), int(perc.orig_size[1])))
            resized_sizes.append((int(perc.resized_size[0]), int(perc.resized_size[1])))
            canvas_sizes.append((int(perc.canvas_size[0]), int(perc.canvas_size[1])))

        with torch.inference_mode(False):
            c4_batch = torch.cat(c4_parts, dim=0)
            cue_batch = torch.cat(cue_parts, dim=0)
            style_batch = torch.cat(style_parts, dim=0)

        resolved_ids: Tuple[str, ...] = ()
        if image_ids is not None:
            resolved_ids = tuple(str(x) for x in image_ids)
            if len(resolved_ids) != len(perc_list):
                raise ValueError(f"image_ids length {len(resolved_ids)} != len(perceptions) {len(perc_list)}")

        resolved_mask: Optional[Tensor] = None
        if valid_action_mask is not None:
            mask = valid_action_mask
            if mask.ndim == 1:
                mask = mask.unsqueeze(0)
            resolved_mask = _ensure_grad_safe_bool(mask, device=target_dev)

        return AgentBatchState(
            c4_pooled=c4_batch,
            cue_maps=cue_batch,
            style_stats=style_batch,
            orig_sizes=tuple(orig_sizes),
            resized_sizes=tuple(resized_sizes),
            canvas_sizes=tuple(canvas_sizes),
            image_ids=resolved_ids,
            valid_action_mask=resolved_mask,
            coordinate_system="orig_0based_xyxy",
        )
