"""Edge Truncation Filtering & Coverage-Aware Dual-View WBF / Symmetric NMS Fusion (Module 4).

Audit-hardened guarantees:
  - **P1-01 Fixed**: Same-view exclusivity during greedy clustering — a box from view ``vid``
    only considers clusters that do NOT already contain ``vid``; if the closest cluster
    already has a ``vid`` member, it tries the next eligible cluster (`IoU >= iou_thr`) or
    starts a new cluster, NEVER silently swallowing a distinct object from the same view.
  - **P1-03 Fixed**: ``fusion_mode="nms"`` uses the EXACT same clustering threshold (`iou_thr`),
    candidate score thresholds, truncation filter, local-only rules, and ``max_detections=100``
    as ``fusion_mode="wbf"`` — differing ONLY in box coordinate selection (highest-weighted-score
    member box for NMS vs weighted coordinate average for WBF) and single-winner vs multi-view
    score fusion, enabling a strictly controlled fusion-operator ablation.
  - Default ``max_detections=100`` matches ``C4ClipROIHead.detections_per_img=100`` so output
    capacity is identical between ``No-Crop`` and Dual-View paths.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
from torch import Tensor
from torchvision.ops import box_iou


def filter_edge_truncated_boxes(
    crop_det: Dict[str, Tensor],
    trunc_margin_px: float = 5.0,
    min_truncated_area_px: float = 144.0,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Filter local crop boxes touching internal crop cut-lines within ``trunc_margin_px`` (orig crop pixels).

    Only filters boxes with pixel area >= ``min_truncated_area_px`` (12x12) touching an internal
    cut-line (outer image boundaries are exempt).
    """
    boxes_crop = crop_det["boxes_in_crop"]
    boxes_orig = crop_det["boxes_in_orig"]
    scores = crop_det["scores"]
    labels = crop_det["labels"]

    x1, y1, x2, y2 = crop_det["crop_xyxy"]
    orig_h, orig_w = crop_det["orig_size"]
    crop_w = max(float(x2 - x1), 1.0)
    crop_h = max(float(y2 - y1), 1.0)
    crop_area = crop_w * crop_h

    if boxes_crop.numel() == 0:
        return boxes_orig, scores, labels, torch.zeros((0,), dtype=torch.float32)

    bw = (boxes_crop[:, 2] - boxes_crop[:, 0]).clamp(min=0.0)
    bh = (boxes_crop[:, 3] - boxes_crop[:, 1]).clamp(min=0.0)
    box_area = bw * bh
    area_ratio = box_area / crop_area

    if trunc_margin_px <= 0.0:
        return boxes_orig, scores, labels, area_ratio

    is_non_tiny = box_area >= float(min_truncated_area_px)

    touch_internal = torch.zeros(boxes_crop.shape[0], dtype=torch.bool)
    if x1 > 0:
        touch_internal |= boxes_crop[:, 0] < trunc_margin_px
    if y1 > 0:
        touch_internal |= boxes_crop[:, 1] < trunc_margin_px
    if x2 < orig_w:
        touch_internal |= (crop_w - boxes_crop[:, 2]) < trunc_margin_px
    if y2 < orig_h:
        touch_internal |= (crop_h - boxes_crop[:, 3]) < trunc_margin_px

    keep = ~(touch_internal & is_non_tiny)
    return boxes_orig[keep], scores[keep], labels[keep], area_ratio[keep]


class DualViewWBFFusion:
    """Dual-view fusion between global detections and K<=2 zoomed crop detections."""

    def __init__(
        self,
        global_weight: float = 1.0,
        local_weight: float = 0.85,
        iou_thr: float = 0.55,
        trunc_margin_px: float = 5.0,
        min_truncated_area_px: float = 144.0,
        local_min_score: float = 0.05,
        local_only_min_score: float = 0.14,
        local_only_single_scale: float = 0.30,
        local_only_multi_scale: float = 0.50,
        max_local_only_crop_area_ratio: float = 0.35,
        global_refine_min_ioa: float = 0.72,
        max_detections: int = 100,
        fusion_mode: str = "wbf",  # "wbf" or "nms"
    ) -> None:
        if fusion_mode not in {"wbf", "nms"}:
            raise ValueError(f"fusion_mode must be 'wbf' or 'nms', got {fusion_mode}")
        self.global_weight = float(global_weight)
        self.local_weight = float(local_weight)
        self.iou_thr = float(iou_thr)
        self.trunc_margin_px = float(trunc_margin_px)
        self.min_truncated_area_px = float(min_truncated_area_px)
        self.local_min_score = float(local_min_score)
        self.local_only_min_score = float(local_only_min_score)
        self.local_only_single_scale = float(local_only_single_scale)
        self.local_only_multi_scale = float(local_only_multi_scale)
        self.max_local_only_crop_area_ratio = float(max_local_only_crop_area_ratio)
        self.global_refine_min_ioa = float(global_refine_min_ioa)
        self.max_detections = int(max_detections)
        self.fusion_mode = fusion_mode

    def config_dict(self) -> Dict:
        return {
            "global_weight": self.global_weight,
            "local_weight": self.local_weight,
            "iou_thr": self.iou_thr,
            "trunc_margin_px": self.trunc_margin_px,
            "min_truncated_area_px": self.min_truncated_area_px,
            "local_min_score": self.local_min_score,
            "local_only_min_score": self.local_only_min_score,
            "local_only_single_scale": self.local_only_single_scale,
            "local_only_multi_scale": self.local_only_multi_scale,
            "max_local_only_crop_area_ratio": self.max_local_only_crop_area_ratio,
            "global_refine_min_ioa": self.global_refine_min_ioa,
            "max_detections": self.max_detections,
            "fusion_mode": self.fusion_mode,
        }

    @staticmethod
    def _box_ioa_in_rect(box: Tensor, rect: Tuple[int, int, int, int]) -> float:
        bx1, by1, bx2, by2 = [float(v) for v in box.tolist()]
        b_area = max((bx2 - bx1) * (by2 - by1), 1e-6)
        cx1, cy1, cx2, cy2 = rect
        ix1, iy1 = max(bx1, float(cx1)), max(by1, float(cy1))
        ix2, iy2 = min(bx2, float(cx2)), min(by2, float(cy2))
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        return inter / b_area

    def fuse(
        self,
        global_det: Dict[str, Tensor],
        crop_dets: Sequence[Dict[str, Tensor]],
        orig_size: Tuple[int, int],
    ) -> Dict[str, Tensor]:
        """Fuse global_det with 0..K crop_dets in original 0-based image coordinates."""
        if not crop_dets:
            boxes = global_det["boxes"].clone()
            scores = global_det["scores"].clone()
            labels = global_det["labels"].clone()
            if boxes.shape[0] > self.max_detections:
                topk = scores.topk(self.max_detections).indices
                boxes, scores, labels = boxes[topk], scores[topk], labels[topk]
            return {"boxes": boxes, "scores": scores, "labels": labels}

        orig_h, orig_w = orig_size
        g_boxes = global_det["boxes"].float()
        g_scores = global_det["scores"].float()
        g_labels = global_det["labels"].long()

        views_boxes: List[Tensor] = [g_boxes]
        views_scores: List[Tensor] = [g_scores]
        views_labels: List[Tensor] = [g_labels]
        views_area_ratios: List[Tensor] = [torch.zeros((g_boxes.shape[0],), dtype=torch.float32)]
        views_weights: List[float] = [self.global_weight]
        crop_rects: List[Tuple[int, int, int, int]] = []

        for c_det in crop_dets:
            crop_rects.append(c_det["crop_xyxy"])
            l_boxes, l_scores, l_labels, l_ratios = filter_edge_truncated_boxes(
                c_det,
                trunc_margin_px=self.trunc_margin_px,
                min_truncated_area_px=self.min_truncated_area_px,
            )
            if l_boxes.numel() > 0 and self.local_min_score > 0.0:
                keep = l_scores >= self.local_min_score
                l_boxes = l_boxes[keep]
                l_scores = l_scores[keep]
                l_labels = l_labels[keep]
                l_ratios = l_ratios[keep]
            views_boxes.append(l_boxes)
            views_scores.append(l_scores)
            views_labels.append(l_labels)
            views_area_ratios.append(l_ratios)
            views_weights.append(self.local_weight)

        return self._fuse_clusters(
            views_boxes, views_scores, views_labels, views_area_ratios, views_weights, crop_rects, orig_h, orig_w
        )

    def _fuse_clusters(
        self,
        views_boxes: List[Tensor],
        views_scores: List[Tensor],
        views_labels: List[Tensor],
        views_area_ratios: List[Tensor],
        views_weights: List[float],
        crop_rects: List[Tuple[int, int, int, int]],
        orig_h: int,
        orig_w: int,
    ) -> Dict[str, Tensor]:
        out_boxes: List[Tensor] = []
        out_scores: List[float] = []
        out_labels: List[int] = []

        all_classes = sorted(
            set(int(c) for l in views_labels if l.numel() > 0 for c in l.unique().tolist())
        )

        for cls_id in all_classes:
            cand_boxes: List[Tensor] = []
            cand_scores: List[float] = []
            cand_weights: List[float] = []
            cand_ratios: List[float] = []
            cand_views: List[int] = []

            for vid, (b, s, l, r, w) in enumerate(
                zip(views_boxes, views_scores, views_labels, views_area_ratios, views_weights)
            ):
                if b.numel() == 0:
                    continue
                mask = l == cls_id
                if not mask.any():
                    continue
                b_c, s_c, r_c = b[mask], s[mask], r[mask]
                for i in range(b_c.shape[0]):
                    cand_boxes.append(b_c[i])
                    cand_scores.append(float(s_c[i].item()))
                    cand_weights.append(w)
                    cand_ratios.append(float(r_c[i].item()))
                    cand_views.append(vid)

            if not cand_boxes:
                continue

            # Deterministic sort: descending weighted score, tie-break by view_id ascending (global first)
            order = sorted(
                range(len(cand_boxes)),
                key=lambda idx: (-cand_weights[idx] * cand_scores[idx], cand_views[idx], idx),
            )
            clusters: List[Dict] = []

            for idx in order:
                box = cand_boxes[idx]
                score = cand_scores[idx]
                weight = cand_weights[idx]
                ratio = cand_ratios[idx]
                vid = cand_views[idx]

                # P1-01 Fix: Only clusters that do NOT already have a box from `vid` can accept `box`
                eligible_c_indices = [ci for ci, c in enumerate(clusters) if vid not in c["members"]]

                if not eligible_c_indices:
                    clusters.append({"members": {vid: (box, score, weight, ratio)}, "fused_box": box.clone()})
                    continue

                elig_boxes = torch.stack([clusters[ci]["fused_box"] for ci in eligible_c_indices], dim=0)
                ious = box_iou(box.unsqueeze(0), elig_boxes)[0]
                best_iou, best_rel_idx = torch.max(ious, dim=0)

                if float(best_iou.item()) >= self.iou_thr:
                    c_idx = eligible_c_indices[int(best_rel_idx.item())]
                    clusters[c_idx]["members"][vid] = (box, score, weight, ratio)
                    if self.fusion_mode == "wbf":
                        clusters[c_idx]["fused_box"] = self._recompute_cluster_box_wbf(
                            clusters[c_idx]["members"], crop_rects
                        )
                    else:
                        clusters[c_idx]["fused_box"] = self._recompute_cluster_box_nms(
                            clusters[c_idx]["members"], crop_rects
                        )
                else:
                    clusters.append({"members": {vid: (box, score, weight, ratio)}, "fused_box": box.clone()})

            for cluster in clusters:
                members = cluster["members"]
                has_global = 0 in members
                local_items = [(vid, item) for vid, item in members.items() if vid != 0]

                if not has_global:
                    max_s = max(item[1] for _, item in local_items)
                    max_r = max(item[3] for _, item in local_items)
                    if max_s < self.local_only_min_score or max_r > self.max_local_only_crop_area_ratio:
                        continue

                f_box = cluster["fused_box"].clone()
                f_box[0::2] = f_box[0::2].clamp(0.0, float(orig_w))
                f_box[1::2] = f_box[1::2].clamp(0.0, float(orig_h))

                if has_global and not local_items:
                    f_score = members[0][1]
                elif has_global and local_items:
                    g_box, g_score, g_w, _ = members[0]
                    if self.fusion_mode == "nms":
                        # Symmetric NMS: keep winner's calibrated score without multi-view WBF averaging
                        f_score = g_score
                    else:
                        # Only let local members that cleanly contain global box contribute to confidence boost
                        valid_locals = [
                            it for vid, it in local_items
                            if self._box_ioa_in_rect(g_box, crop_rects[vid - 1]) >= self.global_refine_min_ioa
                        ]
                        if valid_locals:
                            w_sum = g_w * g_score + sum(it[1] * it[2] for it in valid_locals)
                            w_den = g_w + sum(it[2] for it in valid_locals)
                            f_score = max(g_score, w_sum / max(w_den, 1e-6))
                        else:
                            f_score = g_score
                else:
                    if self.fusion_mode == "nms":
                        best_s = max(it[1] for _, it in local_items)
                        f_score = best_s * self.local_only_single_scale
                    else:
                        mean_s = sum(it[1] for _, it in local_items) / len(local_items)
                        scale = (
                            self.local_only_multi_scale if len(local_items) >= 2 else self.local_only_single_scale
                        )
                        f_score = mean_s * scale

                out_boxes.append(f_box)
                out_scores.append(float(f_score))
                out_labels.append(cls_id)

        if not out_boxes:
            return {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "scores": torch.zeros((0,), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.long),
            }

        boxes_t = torch.stack(out_boxes, dim=0)
        scores_t = torch.tensor(out_scores, dtype=torch.float32)
        labels_t = torch.tensor(out_labels, dtype=torch.long)

        if boxes_t.shape[0] > self.max_detections:
            topk = scores_t.topk(self.max_detections, largest=True, sorted=True).indices
            boxes_t = boxes_t[topk]
            scores_t = scores_t[topk]
            labels_t = labels_t[topk]

        return {"boxes": boxes_t, "scores": scores_t, "labels": labels_t}

    def _recompute_cluster_box_wbf(
        self,
        members: Dict[int, Tuple[Tensor, float, float, float]],
        crop_rects: List[Tuple[int, int, int, int]],
    ) -> Tensor:
        """Compute WBF weighted box coordinates while guarding global boxes that straddle crop cut-lines."""
        if 0 in members:
            g_box, g_score, g_w, _ = members[0]
            num = (g_w * g_score) * g_box.clone()
            den = g_w * g_score
            for vid, (l_box, l_score, l_w, _) in members.items():
                if vid == 0:
                    continue
                crop_rect = crop_rects[vid - 1]
                if self._box_ioa_in_rect(g_box, crop_rect) >= self.global_refine_min_ioa:
                    ws = l_w * l_score
                    num = num + ws * l_box
                    den += ws
            return num / max(den, 1e-6)

        num = torch.zeros(4, dtype=torch.float32)
        den = 0.0
        for l_box, l_score, l_w, _ in members.values():
            ws = l_w * l_score
            num = num + ws * l_box
            den += ws
        return num / max(den, 1e-6)

    def _recompute_cluster_box_nms(
        self,
        members: Dict[int, Tuple[Tensor, float, float, float]],
        crop_rects: List[Tuple[int, int, int, int]],
    ) -> Tensor:
        """Symmetric NMS box selection: pick single highest-weighted-score valid box (no coordinate blending)."""
        if 0 in members:
            g_box, g_score, g_w, _ = members[0]
            best_box = g_box.clone()
            best_ws = g_w * g_score
            for vid, (l_box, l_score, l_w, _) in members.items():
                if vid == 0:
                    continue
                crop_rect = crop_rects[vid - 1]
                if self._box_ioa_in_rect(g_box, crop_rect) >= self.global_refine_min_ioa:
                    ws = l_w * l_score
                    if ws > best_ws:
                        best_ws = ws
                        best_box = l_box.clone()
            return best_box

        best_vid = max(members.keys(), key=lambda v: members[v][2] * members[v][1])
        return members[best_vid][0].clone()
