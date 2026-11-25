"""Parameter-free deterministic multi-source cue encoder (Module 1, parameterless part).

Constructs spatial cue maps aligned to a fixed grid (H_g, W_g):
  - M_rpn [2, H_g, W_g]:
      Ch 0: Probability-weighted RPN grey-zone proposal center density (0.10 < p_obj < 0.45),
            divided by 3.0 and clipped to [0, 1].
      Ch 1: Top-3 anchor mean foreground binary entropy H(p)/ln(2), adaptive-avg-pooled to (H_g, W_g).
  - M_det [3, H_g, W_g]:
      Ch 0: Weak/small-object classes (person, bike, rider, motor) low-mid score (0.10 < s < 0.55) max coverage.
      Ch 1: Small predicted box (< 1.5% image area, score > 0.08) max-scaled coverage min(1.0, 1.5*s).
      Ch 2: High-confidence all-class box (score >= 0.65) binary mask.

Note: Trainable 1x1 visual feature projection (1024 -> 256) and style statistics projection
(2048 -> 256) belong strictly to ``ActiveZoomPolicyHead`` (Phase 1) so gradients flow cleanly.
"""

from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from .detector_wrapper import GlobalPerceptionOutput

# Sky-7 1-indexed IDs for vulnerable/small classes: person(2), bike(4), rider(6), motor(7)
DEFAULT_WEAK_CLASS_IDS = (2, 4, 6, 7)


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

        return {"m_rpn": m_rpn.to(out_dev), "m_det": m_det.to(out_dev)}
