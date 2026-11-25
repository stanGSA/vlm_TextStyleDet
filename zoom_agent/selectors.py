"""Phase-0 Crop Selectors: RPN/Weak-Cue Heuristic & Static GT-Error-Coverage Oracle.

1. ``RPNHeuristicSelector`` (zero-GT, inference-valid baseline):
   Scores the N=18 candidate windows using RPN grey-zone density + foreground
   entropy + weak/small-class low-confidence cue maps, suppresses overlapping
   windows, and chooses No-Crop when scene uncertainty is below threshold.

2. ``OracleSelector`` (Static Error-Coverage Oracle — uses GT ONLY for Phase-0
   diagnostic reference & Stage-1 source-only SFT pseudo-label mining):
   Finds missed GT objects (FN) and coarsely localized GT objects (IoU < 0.75)
   from global predictions, requires candidate windows to cleanly contain >=85%
   of the target GT box, and greedily selects up to K_max windows.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
from torch import Tensor
from torchvision.ops import box_iou

from .candidate_grid import CandidateGrid
from .detector_wrapper import GlobalPerceptionOutput
from .state_encoder import DEFAULT_WEAK_CLASS_IDS, StateEncoder


class RPNHeuristicSelector:
    """Non-learned RPN grey-zone + weak-class cue heuristic crop selector."""

    def __init__(
        self,
        grid: CandidateGrid,
        state_encoder: StateEncoder,
        k_max: int = 2,
        min_window_score: float = 0.52,
        second_window_ratio: float = 0.78,
        nms_window_iou: float = 0.35,
    ) -> None:
        if k_max < 0 or k_max > grid.num_windows:
            raise ValueError(f"Invalid k_max={k_max}")
        self.grid = grid
        self.state_encoder = state_encoder
        self.k_max = int(k_max)
        self.min_window_score = float(min_window_score)
        self.second_window_ratio = float(second_window_ratio)
        self.nms_window_iou = float(nms_window_iou)
        self._win_iou = grid.window_iou_matrix()

    def config_dict(self) -> Dict:
        return {
            "k_max": self.k_max,
            "min_window_score": self.min_window_score,
            "second_window_ratio": self.second_window_ratio,
            "nms_window_iou": self.nms_window_iou,
            "priority_weights": [1.20, 0.60, 1.60, 0.90, -0.25],
            "hotspot_top25_blend": [0.65, 0.35],
            "horizon_bonus_mult": 1.08,
        }

    @torch.no_grad()
    def select(self, perc: GlobalPerceptionOutput) -> Tuple[List[int], Tensor]:
        """Return (selected_window_ids, window_scores [N]). Zero GT access."""
        cues = self.state_encoder.encode_cue_maps(perc, device="cpu")
        m_rpn = cues["m_rpn"]  # [2, H_g, W_g]
        m_det = cues["m_det"]  # [3, H_g, W_g]
        hg, wg = m_rpn.shape[1], m_rpn.shape[2]

        priority_map = (
            1.20 * m_rpn[0]
            + 0.60 * m_rpn[1]
            + 1.60 * m_det[0]
            + 0.90 * m_det[1]
            - 0.25 * m_det[2]
        ).clamp(min=0.0)

        scores = torch.zeros(self.grid.num_windows, dtype=torch.float32)
        for w in self.grid.windows:
            nx1, ny1, nx2, ny2 = w.norm_xyxy
            gx1 = max(0, min(int(nx1 * wg), wg - 1))
            gy1 = max(0, min(int(ny1 * hg), hg - 1))
            gx2 = max(gx1 + 1, min(int(round(nx2 * wg)), wg))
            gy2 = max(gy1 + 1, min(int(round(ny2 * hg)), hg))

            patch = priority_map[gy1:gy2, gx1:gx2].reshape(-1)
            top_n = max(1, patch.numel() // 4)
            s_val = 0.65 * float(patch.topk(top_n).values.mean().item()) + 0.35 * float(patch.mean().item())
            if w.group == "horizon_band":
                s_val *= 1.08
            scores[w.window_id] = s_val

        if self.k_max == 0:
            return [], scores

        selected: List[int] = []
        suppressed = torch.zeros(self.grid.num_windows, dtype=torch.bool)
        best_first = 0.0

        for step in range(self.k_max):
            masked_scores = scores.clone()
            masked_scores[suppressed] = -1.0
            best_val, best_idx = torch.max(masked_scores, dim=0)
            val = float(best_val.item())
            wid = int(best_idx.item())

            if step == 0:
                if val < self.min_window_score:
                    break
                best_first = val
            else:
                if val < max(self.min_window_score, best_first * self.second_window_ratio):
                    break

            selected.append(wid)
            suppressed |= self._win_iou[wid] > self.nms_window_iou
            suppressed[wid] = True

        return selected, scores


class OracleSelector:
    """Static GT-error-coverage selector (Stage-1 SFT mining & Phase-0 diagnostic reference)."""

    def __init__(
        self,
        grid: CandidateGrid,
        k_max: int = 2,
        weak_class_ids: Sequence[int] = DEFAULT_WEAK_CLASS_IDS,
        fn_score_floor: float = 0.25,
        refine_iou_ceil: float = 0.75,
        min_containment_ioa: float = 0.85,
        max_gt_to_window_area_ratio: float = 0.70,
        nms_window_iou: float = 0.45,
    ) -> None:
        if k_max < 0 or k_max > grid.num_windows:
            raise ValueError(f"Invalid k_max={k_max}")
        self.grid = grid
        self.k_max = int(k_max)
        self.weak_class_ids = set(int(c) for c in weak_class_ids)
        self.fn_score_floor = float(fn_score_floor)
        self.refine_iou_ceil = float(refine_iou_ceil)
        self.min_containment_ioa = float(min_containment_ioa)
        self.max_gt_to_window_area_ratio = float(max_gt_to_window_area_ratio)
        self.nms_window_iou = float(nms_window_iou)
        self._win_iou = grid.window_iou_matrix()

    def config_dict(self) -> Dict:
        return {
            "k_max": self.k_max,
            "weak_class_ids": sorted(self.weak_class_ids),
            "fn_score_floor": self.fn_score_floor,
            "refine_iou_ceil": self.refine_iou_ceil,
            "min_containment_ioa": self.min_containment_ioa,
            "max_gt_to_window_area_ratio": self.max_gt_to_window_area_ratio,
            "nms_window_iou": self.nms_window_iou,
        }

    @torch.no_grad()
    def select(
        self,
        perc: GlobalPerceptionOutput,
        gt_boxes_0based: Tensor,
        gt_labels: Tensor,
        gt_difficult: Tensor | None = None,
    ) -> Tuple[List[int], Tensor]:
        """Select up to k_max windows maximizing static coverage of FN & coarse-IoU (<0.75) GTs."""
        num_w = self.grid.num_windows
        zero_scores = torch.zeros(num_w, dtype=torch.float32)

        if self.k_max == 0 or gt_boxes_0based.numel() == 0:
            return [], zero_scores

        if gt_difficult is not None:
            valid_gt = ~gt_difficult.bool()
            gt_boxes = gt_boxes_0based[valid_gt]
            gt_cls = gt_labels[valid_gt]
        else:
            gt_boxes = gt_boxes_0based
            gt_cls = gt_labels

        if gt_boxes.numel() == 0:
            return [], zero_scores

        orig_h, orig_w = perc.orig_size
        img_area = float(max(orig_h * orig_w, 1))

        p_boxes = perc.detections["boxes"]
        p_scores = perc.detections["scores"]
        p_labels = perc.detections["labels"]

        num_gt = gt_boxes.shape[0]
        gt_weights = torch.zeros(num_gt, dtype=torch.float32)

        for g_idx in range(num_gt):
            cls_id = int(gt_cls[g_idx].item())
            g_box = gt_boxes[g_idx : g_idx + 1]
            g_area_ratio = float(((g_box[0, 2] - g_box[0, 0]) * (g_box[0, 3] - g_box[0, 1])).item()) / img_area

            if g_area_ratio > 0.20:
                continue

            is_weak_or_small = (cls_id in self.weak_class_ids) or (g_area_ratio < 0.015)
            cls_mult = 1.6 if is_weak_or_small else 1.0

            same_cls = (p_labels == cls_id) & (p_scores >= self.fn_score_floor)
            if not same_cls.any():
                gt_weights[g_idx] = 3.0 * cls_mult
            else:
                ious = box_iou(g_box, p_boxes[same_cls])[0]
                best_iou = float(ious.max().item())
                if best_iou < 0.50:
                    gt_weights[g_idx] = 3.0 * cls_mult
                elif best_iou < self.refine_iou_ceil:
                    gt_weights[g_idx] = 2.0 * (self.refine_iou_ceil - best_iou) * cls_mult

        if float(gt_weights.sum().item()) <= 1e-5:
            return [], zero_scores

        win_boxes = self.grid.get_pixel_boxes(orig_w, orig_h)
        win_areas = (win_boxes[:, 2] - win_boxes[:, 0]) * (win_boxes[:, 3] - win_boxes[:, 1])
        gt_areas = ((gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])).clamp(min=1e-6)

        ix1 = torch.max(win_boxes[:, None, 0], gt_boxes[None, :, 0])
        iy1 = torch.max(win_boxes[:, None, 1], gt_boxes[None, :, 1])
        ix2 = torch.min(win_boxes[:, None, 2], gt_boxes[None, :, 2])
        iy2 = torch.min(win_boxes[:, None, 3], gt_boxes[None, :, 3])
        inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)

        ioa_gt = inter / gt_areas[None, :]
        size_ok = (gt_areas[None, :] / win_areas[:, None].clamp(min=1e-6)) <= self.max_gt_to_window_area_ratio
        valid_cover = (ioa_gt >= self.min_containment_ioa) & size_ok

        zoom_bonus = (float(orig_w * orig_h) / win_areas.clamp(min=1.0)).sqrt().clamp(1.0, 3.5) * 0.05

        initial_scores = valid_cover.float() @ gt_weights
        remaining_weights = gt_weights.clone()
        suppressed = torch.zeros(num_w, dtype=torch.bool)
        selected: List[int] = []

        for _ in range(self.k_max):
            step_gain = valid_cover.float() @ remaining_weights
            step_scores = torch.where(step_gain > 1e-5, step_gain + zoom_bonus, torch.zeros_like(step_gain))
            step_scores[suppressed] = -1.0

            best_val, best_idx = torch.max(step_scores, dim=0)
            if float(best_val.item()) <= 1e-5:
                break

            wid = int(best_idx.item())
            selected.append(wid)
            covered_gts = valid_cover[wid]
            remaining_weights[covered_gts] = 0.0
            suppressed |= self._win_iou[wid] > self.nms_window_iou
            suppressed[wid] = True

            if float(remaining_weights.sum().item()) <= 1e-5:
                break

        return selected, initial_scores
