"""Audit-Hardened Phase-0 Error Diagnosis & Dual-View Benchmark.

Resolves Red-Team Audit items P0-01, P1-01, P1-03, P1-04, P2-01, P2-02:
  1. Strict temporal & computational zero-GT isolation for ``baseline`` and
     ``rpn_heuristic_wbf``: both execute completely BEFORE any GT annotation is
     converted or inspected on each image, and every crop window runs at
     ``batch_size = 1`` via ``forward_single_crop`` (zero cross-window canvas coupling).
  2. True end-to-end deployment wall-clock latency (p50, p95, mean ms, measured FPS)
     recorded directly on the isolated ``baseline`` and ``rpn_heuristic_wbf`` paths
     after GPU warmup; offline GT-Oracle search overhead is timed separately.
  3. Unbiased fusion-operator & truncation-filter ablations under unified
     ``Static Error-Coverage Oracle (K<=2)`` actions (``static_oracle_nms``,
     ``static_oracle_wbf_notrunc``, ``static_oracle_wbf``), PLUS independent
     per-fuser ``Restricted Reward-Proxy Action Oracle (K<=2)`` (``reward_oracle_nms``,
     ``reward_oracle_wbf``).
  4. Raw integer hit/outcome counters persisted alongside largest-remainder 100.00%
     display percentages, explicit dual denominators (% of all GT vs % of detected GT),
     and 1-to-1 greedy score-sorted Recall alongside geometric GT coverage.
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

sys.dont_write_bytecode = True

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torchvision.ops import box_iou
from tqdm import tqdm

from .candidate_grid import CandidateGrid
from .detector_wrapper import FrozenTextStyleDetWrapper
from .eval_utils import (
     append_voc_prediction_lines,
    evaluate_voc_predictions,
    largest_remainder_round_pct,
    load_domain_records,
    record_to_0based_gt,
)
from .selectors import OracleSelector, RPNHeuristicSelector
from .state_encoder import DEFAULT_WEAK_CLASS_IDS, StateEncoder
from .wbf_fusion import DualViewWBFFusion

COMPARED_METHODS: Tuple[str, ...] = (
    "baseline",
    "rpn_heuristic_wbf",
    "static_oracle_nms",
    "static_oracle_wbf_notrunc",
    "static_oracle_wbf",
    "reward_oracle_nms",
    "reward_oracle_wbf",
)


def _match_1to1_greedy_ious(
    pred_boxes: Tensor,
    pred_scores: Tensor,
    pred_labels: Tensor,
    gt_boxes: Tensor,
    gt_labels: Tensor,
    match_iou_floor: float = 0.50,
) -> Tensor:
    """Perform VOC-style 1-to-1 confidence-sorted greedy matching per class.

    Returns ``matched_iou_per_gt`` [N_gt], with 0.0 for unmatched GTs.
    """
    n_gt = gt_boxes.shape[0]
    matched_ious = torch.zeros(n_gt, dtype=torch.float32)
    if n_gt == 0 or pred_boxes.numel() == 0:
        return matched_ious

    for cls_id in gt_labels.unique().tolist():
        gt_idx = torch.where(gt_labels == cls_id)[0]
        p_idx = torch.where(pred_labels == cls_id)[0]
        if p_idx.numel() == 0:
            continue

        c_gt = gt_boxes[gt_idx]
        c_pb = pred_boxes[p_idx]
        c_ps = pred_scores[p_idx]
        order = c_ps.argsort(descending=True)
        c_pb = c_pb[order]

        ious = box_iou(c_pb, c_gt)  # [N_p, N_g_cls]
        taken_gt = set()
        for pi in range(c_pb.shape[0]):
            best_v = 0.0
            best_gi = -1
            for gi in range(c_gt.shape[0]):
                if gi in taken_gt:
                    continue
                v = float(ious[pi, gi].item())
                if v > best_v:
                    best_v = v
                    best_gi = gi
            if best_v >= match_iou_floor and best_gi >= 0:
                taken_gt.add(best_gi)
                orig_gt_i = int(gt_idx[best_gi].item())
                matched_ious[orig_gt_i] = best_v

    return matched_ious


class DomainErrorAccumulator:
    """Accumulates exact integer RPN recall, 1-to-1 detection recall, and GT outcomes."""

    def __init__(self, weak_class_ids: Sequence[int] = DEFAULT_WEAK_CLASS_IDS) -> None:
        self.weak_set = set(int(c) for c in weak_class_ids)
        self.total_images = 0
        self.total_gt = 0
        self.total_weak_gt = 0
        self.size_gt = {"small": 0, "medium": 0, "large": 0}

        # Exact integer RPN geometric proposal coverage counters
        self.rpn_counts = {
            "hit_100_50": 0,
            "hit_300_50": 0,
            "hit_1000_50": 0,
            "hit_1000_75": 0,
            "weak_hit_1000_50": 0,
            "small_hit_1000_50": 0,
        }

        # Mutually exclusive 4-way 1-to-1 GT outcome counts for Global Baseline (score >= 0.10)
        self.global_outcome_counts = {
            "fn_rpn_unrecovered": 0,  # Unmatched at 1-to-1 det IoU>=0.50 AND RPN@1000 IoU < 0.50
            "fn_roi_or_nms_dropped": 0,  # Unmatched at 1-to-1 det IoU>=0.50 DESPITE RPN@1000 IoU >= 0.50
            "tp_coarse_50_75": 0,  # Matched 1-to-1 with 0.50 <= IoU < 0.75
            "tp_tight_75": 0,  # Matched 1-to-1 with IoU >= 0.75
        }

        self.matched_ious_1to1: Dict[str, List[float]] = {
            "baseline": [],
            "rpn_heuristic_wbf": [],
            "static_oracle_wbf": [],
            "reward_oracle_wbf": [],
        }

        self.tracked_recall_methods = (
            "baseline",
            "rpn_heuristic_wbf",
            "static_oracle_wbf",
            "reward_oracle_wbf",
        )
        # 1-to-1 greedy matched TP counts by size (score >= 0.10)
        self.det_1to1_size_hits = {
            m: {"small_50": 0, "small_75": 0, "medium_50": 0, "medium_75": 0, "large_50": 0, "large_75": 0}
            for m in self.tracked_recall_methods
        }
        # Geometric coverage counts (any same-class box IoU >= thr, score >= 0.10)
        self.det_cov_size_hits = {
            m: {"small_50": 0, "small_75": 0, "medium_50": 0, "medium_75": 0, "large_50": 0, "large_75": 0}
            for m in self.tracked_recall_methods
        }

    @staticmethod
    def _size_bucket(area_px: float) -> str:
        if area_px < 32.0 * 32.0:
            return "small"
        if area_px < 96.0 * 96.0:
            return "medium"
        return "large"

    def update(
        self,
        gt_boxes: Tensor,
        gt_labels: Tensor,
        gt_difficult: Tensor,
        proposals: Tensor,
        dets_by_method: Dict[str, Dict[str, Tensor]],
        det_recall_score_thr: float = 0.10,
    ) -> None:
        self.total_images += 1
        valid = ~gt_difficult.bool()
        g_boxes = gt_boxes[valid]
        g_labels = gt_labels[valid]
        n_gt = int(g_boxes.shape[0])
        if n_gt == 0:
            return

        self.total_gt += n_gt
        areas = (g_boxes[:, 2] - g_boxes[:, 0]).clamp(min=0) * (g_boxes[:, 3] - g_boxes[:, 1]).clamp(min=0)
        buckets = [self._size_bucket(float(a.item())) for a in areas]
        is_weak = [int(c.item()) in self.weak_set for c in g_labels]

        for b_name, w_flag in zip(buckets, is_weak):
            self.size_gt[b_name] += 1
            if w_flag:
                self.total_weak_gt += 1

        # 1) RPN geometric proposal recall
        rpn_1000_hit = torch.zeros(n_gt, dtype=torch.bool)
        if proposals.numel() > 0:
            ious_all = box_iou(g_boxes, proposals)
            for k, key in ((100, "hit_100_50"), (300, "hit_300_50"), (1000, "hit_1000_50")):
                sub = ious_all[:, : min(k, proposals.shape[0])]
                hits = sub.max(dim=1).values >= 0.50
                self.rpn_counts[key] += int(hits.sum().item())
                if k == 1000:
                    rpn_1000_hit = hits
                    hits_75 = sub.max(dim=1).values >= 0.75
                    self.rpn_counts["hit_1000_75"] += int(hits_75.sum().item())
                    for gi in range(n_gt):
                        if bool(hits[gi].item()):
                            if is_weak[gi]:
                                self.rpn_counts["weak_hit_1000_50"] += 1
                            if buckets[gi] == "small":
                                self.rpn_counts["small_hit_1000_50"] += 1

        # 2) 1-to-1 greedy detection recall + geometric coverage
        for m_name in self.tracked_recall_methods:
            det = dets_by_method[m_name]
            keep = det["scores"] >= det_recall_score_thr
            p_boxes = det["boxes"][keep]
            p_scores = det["scores"][keep]
            p_labels = det["labels"][keep]

            matched_1to1 = _match_1to1_greedy_ious(
                p_boxes, p_scores, p_labels, g_boxes, g_labels, match_iou_floor=0.50
            )

            for gi in range(n_gt):
                b_name = buckets[gi]
                m_iou = float(matched_1to1[gi].item())
                if m_iou >= 0.50:
                    self.det_1to1_size_hits[m_name][f"{b_name}_50"] += 1
                    self.matched_ious_1to1[m_name].append(m_iou)
                if m_iou >= 0.75:
                    self.det_1to1_size_hits[m_name][f"{b_name}_75"] += 1

                # Geometric coverage (any same-class box)
                same_c = p_labels == g_labels[gi]
                cov_iou = float(box_iou(g_boxes[gi : gi + 1], p_boxes[same_c])[0].max().item()) if same_c.any() else 0.0
                if cov_iou >= 0.50:
                    self.det_cov_size_hits[m_name][f"{b_name}_50"] += 1
                if cov_iou >= 0.75:
                    self.det_cov_size_hits[m_name][f"{b_name}_75"] += 1

                if m_name == "baseline":
                    if m_iou >= 0.75:
                        self.global_outcome_counts["tp_tight_75"] += 1
                    elif m_iou >= 0.50:
                        self.global_outcome_counts["tp_coarse_50_75"] += 1
                    else:
                        if bool(rpn_1000_hit[gi].item()):
                            self.global_outcome_counts["fn_roi_or_nms_dropped"] += 1
                        else:
                            self.global_outcome_counts["fn_rpn_unrecovered"] += 1

    def summarize(self) -> Dict:
        def pct(num: int, den: int) -> float:
            return round(100.0 * num / max(den, 1), 2)

        def build_size_table(raw_dict: Dict[str, Dict[str, int]]) -> Dict:
            out = {}
            for m_name, d in raw_dict.items():
                out[m_name] = {
                    "counts": dict(d),
                    "small_R50": pct(d["small_50"], self.size_gt["small"]),
                    "small_R75": pct(d["small_75"], self.size_gt["small"]),
                    "medium_R50": pct(d["medium_50"], self.size_gt["medium"]),
                    "medium_R75": pct(d["medium_75"], self.size_gt["medium"]),
                    "large_R50": pct(d["large_50"], self.size_gt["large"]),
                    "large_R75": pct(d["large_75"], self.size_gt["large"]),
                }
            return out

        oc = self.global_outcome_counts
        ordered_counts = [
            oc["fn_rpn_unrecovered"],
            oc["fn_roi_or_nms_dropped"],
            oc["tp_coarse_50_75"],
            oc["tp_tight_75"],
        ]
        assert sum(ordered_counts) == self.total_gt, "Outcome partition must equal total_gt"
        disp_pcts = largest_remainder_round_pct(ordered_counts, self.total_gt, decimals=2)
        total_tp50 = oc["tp_coarse_50_75"] + oc["tp_tight_75"]

        return {
            "num_images": self.total_images,
            "num_gt": self.total_gt,
            "num_weak_gt": self.total_weak_gt,
            "gt_by_size": dict(self.size_gt),
            "rpn_proposal_coverage": {
                "counts": dict(self.rpn_counts),
                "R100_50": pct(self.rpn_counts["hit_100_50"], self.total_gt),
                "R300_50": pct(self.rpn_counts["hit_300_50"], self.total_gt),
                "R1000_50": pct(self.rpn_counts["hit_1000_50"], self.total_gt),
                "R1000_75": pct(self.rpn_counts["hit_1000_75"], self.total_gt),
                "weak_R1000_50": pct(self.rpn_counts["weak_hit_1000_50"], self.total_weak_gt),
                "small_R1000_50": pct(self.rpn_counts["small_hit_1000_50"], self.size_gt["small"]),
            },
            "global_gt_1to1_partition_at_score010": {
                "counts": dict(oc),
                "fn_rpn_unrecovered_pct_of_all_gt": disp_pcts[0],
                "fn_roi_or_nms_dropped_pct_of_all_gt": disp_pcts[1],
                "tp_coarse_50_75_pct_of_all_gt": disp_pcts[2],
                "tp_tight_75_pct_of_all_gt": disp_pcts[3],
                "coarse_share_within_detected_tp50_pct": pct(oc["tp_coarse_50_75"], total_tp50),
                "mean_1to1_matched_iou": {
                    k: round(float(np.mean(v)) if v else 0.0, 4)
                    for k, v in self.matched_ious_1to1.items()
                },
            },
            "size_stratified_1to1_recall_at_score010": build_size_table(self.det_1to1_size_hits),
            "size_stratified_gt_coverage_at_score010": build_size_table(self.det_cov_size_hits),
        }


def compute_image_det_utility(
    det: Dict[str, Tensor],
    gt_boxes: Tensor,
    gt_labels: Tensor,
    gt_difficult: Tensor,
    score_thr: float = 0.12,
) -> float:
    """Per-image detection reward proxy for Phase-0 Restricted Reward-Action Oracle.

    Explicit proxy formula: 2.0*TP50 + 2.5*TP75 + 1.5*sum(IoU) - 0.8*score_weighted_FP.
    Note: This is a Phase-0 per-image proxy over candidate subsets, NOT direct dataset-level AP optimization.
    """
    valid = ~gt_difficult.bool()
    g_boxes = gt_boxes[valid]
    g_labels = gt_labels[valid]

    p_boxes = det["boxes"]
    p_scores = det["scores"]
    p_labels = det["labels"]
    keep = p_scores >= score_thr
    p_boxes, p_scores, p_labels = p_boxes[keep], p_scores[keep], p_labels[keep]

    if g_boxes.numel() == 0:
        return -0.25 * float(p_boxes.shape[0])
    if p_boxes.numel() == 0:
        return 0.0

    tp50 = 0.0
    tp75 = 0.0
    iou_sum = 0.0
    fp_pen = 0.0

    for cls_id in g_labels.unique().tolist():
        c_gt = g_boxes[g_labels == cls_id]
        c_mask = p_labels == cls_id
        if not c_mask.any():
            continue
        c_pb = p_boxes[c_mask]
        c_ps = p_scores[c_mask]
        order = c_ps.argsort(descending=True)
        c_pb, c_ps = c_pb[order], c_ps[order]

        ious = box_iou(c_pb, c_gt)
        matched_gt = set()
        for pi in range(c_pb.shape[0]):
            best_iou = 0.0
            best_gi = -1
            for gi in range(c_gt.shape[0]):
                if gi in matched_gt:
                    continue
                v = float(ious[pi, gi].item())
                if v > best_iou:
                    best_iou = v
                    best_gi = gi
            if best_iou >= 0.50 and best_gi >= 0:
                matched_gt.add(best_gi)
                tp50 += 1.0
                iou_sum += best_iou
                if best_iou >= 0.75:
                    tp75 += 1.0
            else:
                fp_pen += float(c_ps[pi].item())

    present_cls = set(g_labels.unique().tolist())
    for pi in range(p_boxes.shape[0]):
        if int(p_labels[pi].item()) not in present_cls:
            fp_pen += 1.2 * float(p_scores[pi].item())

    return 2.0 * tp50 + 2.5 * tp75 + 1.5 * iou_sum - 0.8 * fp_pen


def _select_best_reward_subset(
    global_det: Dict[str, Tensor],
    cand_wids: Sequence[int],
    crop_det_cache: Dict[int, Dict[str, Tensor]],
    fuser: DualViewWBFFusion,
    orig_size: Tuple[int, int],
    gt_boxes: Tensor,
    gt_labels: Tensor,
    gt_diff: Tensor,
    k_max: int,
    step_cost: float = 0.02,
) -> List[int]:
    if not cand_wids or k_max <= 0:
        return []
    best_u = compute_image_det_utility(global_det, gt_boxes, gt_labels, gt_diff)
    best_sub: List[int] = []
    for r_len in range(1, min(k_max, len(cand_wids)) + 1):
        for sub in combinations(cand_wids, r_len):
            fused = fuser.fuse(global_det, [crop_det_cache[w] for w in sub], orig_size)
            u_val = compute_image_det_utility(fused, gt_boxes, gt_labels, gt_diff) - step_cost * r_len
            if u_val > best_u + 1e-4:
                best_u = u_val
                best_sub = list(sub)
    return best_sub


def _latency_summary(ms_list: List[float]) -> Dict[str, float]:
    if not ms_list:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "measured_e2e_fps": 0.0}
    arr = np.asarray(ms_list, dtype=np.float64)
    mean_ms = float(arr.mean())
    return {
        "mean_ms": round(mean_ms, 2),
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "measured_e2e_fps": round(1000.0 / max(mean_ms, 1e-3), 2),
    }


def build_phase0_components(k_max: int = 2, trunc_margin_px: float = 5.0, max_detections: int = 100):
    if k_max < 1 or k_max > 2:
        raise ValueError(f"Phase 0 requires k_max in {{1, 2}}, got {k_max}")
    grid = CandidateGrid()
    state_encoder = StateEncoder(grid_size=(12, 20))
    heur_selector = RPNHeuristicSelector(grid=grid, state_encoder=state_encoder, k_max=k_max)
    static_oracle_selector = OracleSelector(grid=grid, k_max=k_max)
    reward_pool_selector = OracleSelector(grid=grid, k_max=min(grid.num_windows, k_max + 1))

    fuser_wbf = DualViewWBFFusion(
        global_weight=1.0,
        local_weight=0.85,
        iou_thr=0.55,
        trunc_margin_px=trunc_margin_px,
        max_detections=max_detections,
        fusion_mode="wbf",
    )
    fuser_wbf_notrunc = DualViewWBFFusion(
        global_weight=1.0,
        local_weight=0.85,
        iou_thr=0.55,
        trunc_margin_px=0.0,
        max_detections=max_detections,
        fusion_mode="wbf",
    )
    fuser_nms = DualViewWBFFusion(
        global_weight=1.0,
        local_weight=0.85,
        iou_thr=0.55,
        trunc_margin_px=trunc_margin_px,
        max_detections=max_detections,
        fusion_mode="nms",
    )
    return {
        "grid": grid,
        "state_encoder": state_encoder,
        "heur_selector": heur_selector,
        "static_oracle_selector": static_oracle_selector,
        "reward_pool_selector": reward_pool_selector,
        "fuser_wbf": fuser_wbf,
        "fuser_wbf_notrunc": fuser_wbf_notrunc,
        "fuser_nms": fuser_nms,
    }


def evaluate_domain_phase0(
    wrapper: FrozenTextStyleDetWrapper,
    data_dir: str | Path,
    domain: str,
    split: str | None = None,
    k_max: int = 2,
    trunc_margin_px: float = 5.0,
    max_detections: int = 100,
    max_images: int = 0,
    warmup_images: int = 2,
    show_progress: bool = True,
) -> Dict:
    """Run audit-hardened Phase-0 diagnosis & isolated Dual-View benchmark on one domain."""
    voc_root, resolved_split, image_ids, records, split_sha256 = load_domain_records(
        data_dir=data_dir,
        domain=domain,
        split=split,
        max_images=max_images,
        rn_root=wrapper.rn_root,
    )
    img_dir = voc_root / "JPEGImages"
    comps = build_phase0_components(k_max=k_max, trunc_margin_px=trunc_margin_px, max_detections=max_detections)
    grid: CandidateGrid = comps["grid"]
    heur_selector: RPNHeuristicSelector = comps["heur_selector"]
    static_oracle_selector: OracleSelector = comps["static_oracle_selector"]
    reward_pool_selector: OracleSelector = comps["reward_pool_selector"]
    fuser_wbf: DualViewWBFFusion = comps["fuser_wbf"]
    fuser_wbf_notrunc: DualViewWBFFusion = comps["fuser_wbf_notrunc"]
    fuser_nms: DualViewWBFFusion = comps["fuser_nms"]

    # GPU warmup before recording end-to-end latencies
    if wrapper.device.type == "cuda" and image_ids and warmup_images > 0:
        for wid_warm in image_ids[: min(warmup_images, len(image_ids))]:
            w_img = Image.open(img_dir / f"{wid_warm}.jpg").convert("RGB")
            _ = wrapper.forward_global(w_img)
        torch.cuda.synchronize(wrapper.device)
        torch.cuda.reset_peak_memory_stats(wrapper.device)

    pred_lines = {m: defaultdict(list) for m in COMPARED_METHODS}
    diag = DomainErrorAccumulator()

    baseline_e2e_ms: List[float] = []
    heuristic_e2e_ms: List[float] = []
    single_crop_ms_list: List[float] = []
    offline_oracle_search_ms: List[float] = []

    crop_counts: Dict[str, List[int]] = {
        "rpn_heuristic": [],
        "static_oracle": [],
        "reward_oracle_nms": [],
        "reward_oracle_wbf": [],
    }
    window_hists: Dict[str, np.ndarray] = {
        k: np.zeros(grid.num_windows, dtype=np.int64) for k in crop_counts
    }

    iterator = tqdm(image_ids, desc=f"Phase0 [{domain}/{resolved_split}]", leave=True) if show_progress else image_ids

    for img_id in iterator:
        # =====================================================================
        # PART A: STRICT ZERO-GT DEPLOYMENT PATHS (Baseline & RPN-Heuristic WBF)
        # Executed BEFORE reading/converting any GT annotation for `img_id`.
        # =====================================================================
        if wrapper.device.type == "cuda":
            torch.cuda.synchronize(wrapper.device)
        t_img_start = time.perf_counter()

        pil_img = Image.open(img_dir / f"{img_id}.jpg").convert("RGB")
        perc = wrapper.forward_global(pil_img)

        if wrapper.device.type == "cuda":
            torch.cuda.synchronize(wrapper.device)
        t_after_global = time.perf_counter()

        orig_h, orig_w = perc.orig_size
        base_det = fuser_wbf.fuse(perc.detections, [], perc.orig_size)
        base_ms = (t_after_global - t_img_start) * 1000.0
        baseline_e2e_ms.append(base_ms)

        # Heuristic selection + isolated single-crop forwards + WBF
        t_heur_start = time.perf_counter()
        heur_wids, _ = heur_selector.select(perc)

        crop_det_cache: Dict[int, Dict[str, Tensor]] = {}
        for wid in heur_wids:
            rect = grid.windows[wid].to_pixel_xyxy(orig_w, orig_h)
            if wrapper.device.type == "cuda":
                torch.cuda.synchronize(wrapper.device)
            tc0 = time.perf_counter()
            c_det = wrapper.forward_single_crop(pil_img, rect)
            if wrapper.device.type == "cuda":
                torch.cuda.synchronize(wrapper.device)
            single_crop_ms_list.append((time.perf_counter() - tc0) * 1000.0)
            crop_det_cache[wid] = c_det

        heur_crop_dets = [crop_det_cache[w] for w in heur_wids]
        heur_det = fuser_wbf.fuse(perc.detections, heur_crop_dets, perc.orig_size)
        if wrapper.device.type == "cuda":
            torch.cuda.synchronize(wrapper.device)
        heur_total_ms = base_ms + (time.perf_counter() - t_heur_start) * 1000.0
        heuristic_e2e_ms.append(heur_total_ms)

        crop_counts["rpn_heuristic"].append(len(heur_wids))
        for wid in heur_wids:
            window_hists["rpn_heuristic"][wid] += 1

        # =====================================================================
        # PART B: OFFLINE GT-DIAGNOSTIC PATHS (Static SFT Oracle & Reward Oracle)
        # Cannot retroactively alter `base_det` or `heur_det` (already finalized).
        # =====================================================================
        t_orac_start = time.perf_counter()
        gt_boxes, gt_labels, gt_diff = record_to_0based_gt(records[img_id], wrapper.class_names)

        static_wids, _ = static_oracle_selector.select(perc, gt_boxes, gt_labels, gt_diff)
        reward_pool_wids, _ = reward_pool_selector.select(perc, gt_boxes, gt_labels, gt_diff)

        needed_orac_wids = sorted((set(static_wids) | set(reward_pool_wids)) - set(crop_det_cache.keys()))
        for wid in needed_orac_wids:
            rect = grid.windows[wid].to_pixel_xyxy(orig_w, orig_h)
            if wrapper.device.type == "cuda":
                torch.cuda.synchronize(wrapper.device)
            tc0 = time.perf_counter()
            c_det = wrapper.forward_single_crop(pil_img, rect)
            if wrapper.device.type == "cuda":
                torch.cuda.synchronize(wrapper.device)
            single_crop_ms_list.append((time.perf_counter() - tc0) * 1000.0)
            crop_det_cache[wid] = c_det

        # 1) Static Error-Coverage Oracle (unbiased shared actions across NMS / NoTrunc-WBF / Full-WBF)
        static_crop_dets = [crop_det_cache[w] for w in static_wids]
        static_nms_det = fuser_nms.fuse(perc.detections, static_crop_dets, perc.orig_size)
        static_notrunc_det = fuser_wbf_notrunc.fuse(perc.detections, static_crop_dets, perc.orig_size)
        static_wbf_det = fuser_wbf.fuse(perc.detections, static_crop_dets, perc.orig_size)

        # 2) Restricted Reward-Proxy Action Oracle (independent search per fuser)
        reward_nms_wids = _select_best_reward_subset(
            base_det, reward_pool_wids, crop_det_cache, fuser_nms, perc.orig_size, gt_boxes, gt_labels, gt_diff, k_max
        )
        reward_wbf_wids = _select_best_reward_subset(
            base_det, reward_pool_wids, crop_det_cache, fuser_wbf, perc.orig_size, gt_boxes, gt_labels, gt_diff, k_max
        )
        reward_nms_det = fuser_nms.fuse(
            perc.detections, [crop_det_cache[w] for w in reward_nms_wids], perc.orig_size
        )
        reward_wbf_det = fuser_wbf.fuse(
            perc.detections, [crop_det_cache[w] for w in reward_wbf_wids], perc.orig_size
        )

        if wrapper.device.type == "cuda":
            torch.cuda.synchronize(wrapper.device)
        offline_oracle_search_ms.append((time.perf_counter() - t_orac_start) * 1000.0)

        crop_counts["static_oracle"].append(len(static_wids))
        crop_counts["reward_oracle_nms"].append(len(reward_nms_wids))
        crop_counts["reward_oracle_wbf"].append(len(reward_wbf_wids))
        for wid in static_wids:
            window_hists["static_oracle"][wid] += 1
        for wid in reward_nms_wids:
            window_hists["reward_oracle_nms"][wid] += 1
        for wid in reward_wbf_wids:
            window_hists["reward_oracle_wbf"][wid] += 1

        dets_by_method = {
            "baseline": base_det,
            "rpn_heuristic_wbf": heur_det,
            "static_oracle_nms": static_nms_det,
            "static_oracle_wbf_notrunc": static_notrunc_det,
            "static_oracle_wbf": static_wbf_det,
            "reward_oracle_nms": reward_nms_det,
            "reward_oracle_wbf": reward_wbf_det,
        }

        for m_name, m_det in dets_by_method.items():
            append_voc_prediction_lines(
                pred_lines[m_name],
                image_id=img_id,
                det_0based=m_det,
                orig_size=perc.orig_size,
                num_classes=wrapper.num_classes,
                voc_1_based=True,
            )

        diag.update(
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            gt_difficult=gt_diff,
            proposals=perc.proposals,
            dets_by_method=dets_by_method,
        )

    peak_alloc_gib = (
        round(torch.cuda.max_memory_allocated(wrapper.device) / (1024**3), 3)
        if wrapper.device.type == "cuda"
        else 0.0
    )
    peak_reserved_gib = (
        round(torch.cuda.max_memory_reserved(wrapper.device) / (1024**3), 3)
        if wrapper.device.type == "cuda"
        else 0.0
    )

    n_imgs = max(len(image_ids), 1)
    horizon_ids = [w.window_id for w in grid.windows if w.group == "horizon_band"]

    def _summarize_selector_stats(key: str) -> Dict:
        cnts = crop_counts[key]
        hist = window_hists[key]
        total_c = int(hist.sum())
        mean_k = float(np.mean(cnts)) if cnts else 0.0
        return {
            "total_crops": total_c,
            "avg_crops": round(mean_k, 4),
            "no_crop_count": int(sum(c == 0 for c in cnts)),
            "no_crop_pct": round(100.0 * sum(c == 0 for c in cnts) / n_imgs, 2),
            "horizon_window_frequency_pct": round(
                100.0 * float(hist[horizon_ids].sum()) / max(total_c, 1), 2
            ),
            "window_histogram": [int(x) for x in hist.tolist()],
        }

    base_lat = _latency_summary(baseline_e2e_ms)
    heur_lat = _latency_summary(heuristic_e2e_ms)
    mean_single_crop_ms = float(np.mean(single_crop_ms_list)) if single_crop_ms_list else 0.0

    sel_stats = {k: _summarize_selector_stats(k) for k in crop_counts}
    for k, st in sel_stats.items():
        if k == "rpn_heuristic":
            st["measured_e2e_fps"] = heur_lat["measured_e2e_fps"]
        else:
            # Explicitly labeled projected policy runtime if a lightweight ~2ms head chose the same K crops
            proj_ms = base_lat["mean_ms"] + 2.0 + st["avg_crops"] * mean_single_crop_ms + 1.5
            st["projected_policy_fps_at_same_k"] = round(1000.0 / max(proj_ms, 1e-3), 2)

    voc_metrics = {}
    for m_name in COMPARED_METHODS:
        voc_metrics[m_name] = evaluate_voc_predictions(
            pred_lines_by_class=pred_lines[m_name],
            class_names=wrapper.class_names,
            annotation_records=records,
            use_07_metric=True,
            rn_root=wrapper.rn_root,
        )

    return {
        "domain": domain,
        "split": resolved_split,
        "split_ids_sha256": split_sha256,
        "num_images": len(image_ids),
        "gpu_memory_gib": {
            "peak_allocated_gib": peak_alloc_gib,
            "peak_reserved_gib": peak_reserved_gib,
        },
        "timing_and_budget": {
            "baseline_e2e_latency": base_lat,
            "rpn_heuristic_e2e_latency": heur_lat,
            "mean_isolated_single_crop_ms": round(mean_single_crop_ms, 2),
            "offline_oracle_eval_mean_ms": round(
                float(np.mean(offline_oracle_search_ms)) if offline_oracle_search_ms else 0.0, 2
            ),
            "selectors": sel_stats,
        },
        "diagnosis": diag.summarize(),
        "metrics": voc_metrics,
    }
