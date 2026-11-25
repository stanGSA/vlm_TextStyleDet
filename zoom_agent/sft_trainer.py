"""Stage-1 Annotation-Free SFT Trainer, Reference Policy Exporter & Source-Dev E2E Evaluator.

Design & Audit Guarantees:
  - Offline training loop NEVER loads the frozen StyleDetV6 detector; it streams
    verified sharded FP16 cache records via ``AgentBatchState.from_cache_dict`` and
    batches via ``AgentBatchState.collate``, optimizing ONLY ``ActiveZoomPolicyHead``
    (~1.902M params w/ FiLM, ~1.298M w/o FiLM).
  - Composite SFT objective:
      L_SFT = BCEWithLogitsLoss(logits, sft_multi_hot, pos_weight)
            + traj_nll_weight * (-policy.log_prob_from_logits(logits, oracle_actions).mean())
    where ``pos_weight`` is derived strictly from ``source_train`` statistics (clamped to
    ``[1.0, max_pos_weight]``) to prevent all-zero collapse while preserving No-Crop balance,
    and the trajectory NLL directly calibrates the exact two-step without-replacement Softmax
    distribution used by ``act_greedy`` and Stage-2 GRPO.
  - Strictly locks best checkpoint on ``source_dev`` and exports bitwise-consistent
    ``policy_sft.pth`` and ``policy_sft_ref.pth`` (verified via reload + ``D_KL == 0``).
  - Real end-to-end detection verification runs strictly on ``source_dev`` (clean +
    deterministic synthetic degraded views — ZERO target-domain access) with strict
    zero-GT temporal & computational isolation (P0-01 compliant).
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import copy
import math
import os
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

from .agent_pipeline import AZAgentDetector
from .candidate_grid import CandidateGrid
from .detector_wrapper import FrozenTextStyleDetWrapper
from .eval_utils import (
    append_voc_prediction_lines,
    atomic_write_text,
    evaluate_voc_predictions,
    record_to_0based_gt,
    sha256_file,
    sha256_json,
)
from .policy_head import ActiveZoomPolicyHead
from .selectors import OracleSelector, RPNHeuristicSelector
from .sft_cache import ShardedSFTCacheReader, atomic_torch_save
from .source_degradation import (
    ALLOWED_SOURCE_DOMAIN,
    ALLOWED_SOURCE_SPLIT,
    DEGRADATION_TAGS,
    SourceOnlyDatasetLoader,
    apply_physical_degradation,
    select_adverse_tag_for_view,
    validate_source_only_access,
)
from .state_encoder import StateEncoder
from .wbf_fusion import DualViewWBFFusion


@dataclass
class SFTTrainConfig:
    """Hyperparameters for offline Stage-1 SFT training of ``ActiveZoomPolicyHead``."""

    batch_size: int = 32
    epochs: int = 12
    lr: float = 3e-4
    min_lr: float = 1e-5
    warmup_epochs: float = 1.0
    weight_decay: float = 1e-2
    max_grad_norm: float = 1.0
    traj_nll_weight: float = 0.25
    max_pos_weight: float = 4.0
    no_crop_pos_weight_scale: float = 1.0
    dropout: float = 0.1
    use_film: bool = True
    seed: int = 20260421
    shard_group_size: int = 4
    device: str = "cuda"

    def to_dict(self) -> Dict:
        return asdict(self)


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_source_train_pos_weight(
    train_reader: ShardedSFTCacheReader,
    max_pos_weight: float = 4.0,
    no_crop_pos_weight_scale: float = 1.0,
) -> Tuple[Tensor, Dict]:
    """Compute balanced 19-dim BCE ``pos_weight`` strictly from ``source_train`` statistics."""
    pos_counts, total = train_reader.compute_label_positive_counts()
    neg_counts = (float(total) - pos_counts).clamp(min=0.0)
    safe_pos = pos_counts.clamp(min=1.0)
    raw_ratio = neg_counts / safe_pos
    # Square-root damping avoids extreme weights on rare sky windows while preventing all-zero collapse
    pw = torch.sqrt(raw_ratio).clamp(min=1.0, max=float(max_pos_weight))
    pw[18] = (pw[18] * float(no_crop_pos_weight_scale)).clamp(min=1.0, max=float(max_pos_weight))

    stats = {
        "total_train_samples": int(total),
        "positive_counts": [int(round(x)) for x in pos_counts.tolist()],
        "positive_rates_pct": [round(100.0 * float(x) / max(total, 1), 2) for x in pos_counts.tolist()],
        "pos_weight": [round(float(x), 4) for x in pw.tolist()],
    }
    return pw, stats


def build_policy_from_cache_manifest(
    manifest: Dict,
    use_film: bool = True,
    dropout: float = 0.1,
) -> ActiveZoomPolicyHead:
    """Instantiate ``ActiveZoomPolicyHead`` strictly matching the cache's grid & action contract."""
    g_cfg = manifest["candidate_grid"]
    grid = CandidateGrid(
        grid_rows=int(g_cfg["grid_rows"]),
        grid_cols=int(g_cfg["grid_cols"]),
        grid_win_w=float(g_cfg["grid_win_w"]),
        grid_win_h=float(g_cfg["grid_win_h"]),
    )
    se_cfg = manifest["state_encoder"]
    grid_size = (int(se_cfg["grid_size"][0]), int(se_cfg["grid_size"][1]))
    p_contract = manifest["policy_action_contract"]
    return ActiveZoomPolicyHead(
        grid=grid,
        visual_channels=1024,
        cue_channels=5,
        default_grid_size=grid_size,
        dropout=float(dropout),
        use_film=bool(use_film),
        k_max=int(p_contract["k_max"]),
        nms_window_iou=float(p_contract["nms_window_iou"]),
    )


def _compute_binary_prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    prec = tp / max(tp + fp, 1) if (tp + fp) > 0 else 0.0
    rec = tp / max(tp + fn, 1) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0.0 else 0.0
    return round(100.0 * prec, 2), round(100.0 * rec, 2), round(100.0 * f1, 2)


@torch.no_grad()
def evaluate_policy_on_cache(
    policy: ActiveZoomPolicyHead,
    reader: ShardedSFTCacheReader,
    pos_weight: Tensor,
    traj_nll_weight: float,
    batch_size: int,
    device: torch.device,
) -> Dict:
    """Evaluate SFT loss, Top-1/Top-2 window hit rates, No-Crop P/R/F1, entropy & avg_K."""
    was_training = policy.training
    policy.eval()
    pw_dev = pos_weight.to(device=device, dtype=torch.float32)
    horizon_set = {w.window_id for w in policy.grid.windows if w.group == "horizon_band"}

    total_samples = 0
    loss_sum = 0.0
    bce_sum = 0.0
    nll_sum = 0.0
    ent_sum = 0.0

    crop_gt_samples = 0
    top1_hits = 0
    top2_any_hits = 0
    top2_target_hits = 0
    top2_target_total = 0
    greedy_any_hits = 0
    exact_set_matches = 0

    nc_tp = 0
    nc_fp = 0
    nc_fn = 0
    nc_tn = 0

    pred_k_counts = {0: 0, 1: 0, 2: 0}
    oracle_k_counts = {0: 0, 1: 0, 2: 0}
    pred_total_crops = 0
    pred_horizon_crops = 0

    # Breakdown by clean vs degraded views
    sub_acc = {
        "clean": {
            "n": 0,
            "pred_k_sum": 0,
            "pred_nc": 0,
            "orac_k_sum": 0,
            "orac_nc": 0,
            "ent_sum": 0.0,
            "crop_gt": 0,
            "top1_hits": 0,
            "nc_tp": 0,
            "nc_fp": 0,
            "nc_fn": 0,
        },
        "degraded": {
            "n": 0,
            "pred_k_sum": 0,
            "pred_nc": 0,
            "orac_k_sum": 0,
            "orac_nc": 0,
            "ent_sum": 0.0,
            "crop_gt": 0,
            "top1_hits": 0,
            "nc_tp": 0,
            "nc_fp": 0,
            "nc_fn": 0,
        },
    }

    try:
        for batch in reader.iter_batches(batch_size=batch_size, shuffle=False, seed=0, device=device):
            state = batch["state"]
            targets = batch["sft_multi_hot"]
            oracle_actions = batch["oracle_actions"]
            deg_tags = batch["degradation_tags"]
            bsz = state.batch_size

            logits = policy(state)  # [B, 19]
            bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw_dev)
            nll = -policy.log_prob_from_logits(logits, oracle_actions).mean()
            loss = bce + float(traj_nll_weight) * nll
            ent = policy.entropy_from_logits(logits)  # [B]
            greedy = policy.act_greedy_from_logits(logits)

            total_samples += bsz
            loss_sum += float(loss.item()) * bsz
            bce_sum += float(bce.item()) * bsz
            nll_sum += float(nll.item()) * bsz
            ent_sum += float(ent.sum().item())

            win_logits_cpu = logits[:, : policy.num_windows].detach().cpu()
            top2_idx = torch.topk(win_logits_cpu, k=min(2, policy.num_windows), dim=-1).indices.tolist()
            ent_cpu = ent.detach().cpu().tolist()

            for i in range(bsz):
                orac_act = tuple(oracle_actions[i])
                pred_act = tuple(greedy.actions[i])
                orac_set = set(orac_act)
                pred_set = set(pred_act)
                bucket = "clean" if deg_tags[i] == "clean" else "degraded"
                sb = sub_acc[bucket]

                pred_k_counts[len(pred_act)] = pred_k_counts.get(len(pred_act), 0) + 1
                oracle_k_counts[len(orac_act)] = oracle_k_counts.get(len(orac_act), 0) + 1
                pred_total_crops += len(pred_act)
                pred_horizon_crops += sum(1 for w in pred_act if w in horizon_set)

                sb["n"] += 1
                sb["pred_k_sum"] += len(pred_act)
                sb["orac_k_sum"] += len(orac_act)
                sb["ent_sum"] += float(ent_cpu[i])

                if orac_set == pred_set:
                    exact_set_matches += 1

                gt_is_nc = len(orac_act) == 0
                pred_is_nc = len(pred_act) == 0
                if pred_is_nc:
                    sb["pred_nc"] += 1
                if gt_is_nc:
                    sb["orac_nc"] += 1

                if gt_is_nc and pred_is_nc:
                    nc_tp += 1
                    sb["nc_tp"] += 1
                elif (not gt_is_nc) and pred_is_nc:
                    nc_fp += 1
                    sb["nc_fp"] += 1
                elif gt_is_nc and (not pred_is_nc):
                    nc_fn += 1
                    sb["nc_fn"] += 1
                else:
                    nc_tn += 1

                if not gt_is_nc:
                    crop_gt_samples += 1
                    sb["crop_gt"] += 1
                    t1 = int(top2_idx[i][0])
                    t2_set = set(int(x) for x in top2_idx[i])
                    if t1 in orac_set:
                        top1_hits += 1
                        sb["top1_hits"] += 1
                    if t2_set & orac_set:
                        top2_any_hits += 1
                    top2_target_hits += len(t2_set & orac_set)
                    top2_target_total += len(orac_set)
                    if pred_set & orac_set:
                        greedy_any_hits += 1
    finally:
        if was_training:
            policy.train(True)

    denom = max(total_samples, 1)
    nc_prec, nc_rec, nc_f1 = _compute_binary_prf(nc_tp, nc_fp, nc_fn)
    pred_avg_k = (pred_k_counts.get(1, 0) + 2 * pred_k_counts.get(2, 0)) / denom
    orac_avg_k = (oracle_k_counts.get(1, 0) + 2 * oracle_k_counts.get(2, 0)) / denom

    view_breakdown: Dict[str, Dict] = {}
    for b_name, sb in sub_acc.items():
        bn = max(sb["n"], 1)
        bp, br, bf1 = _compute_binary_prf(sb["nc_tp"], sb["nc_fp"], sb["nc_fn"])
        view_breakdown[b_name] = {
            "num_samples": sb["n"],
            "pred_avg_k": round(sb["pred_k_sum"] / bn, 4),
            "pred_no_crop_pct": round(100.0 * sb["pred_nc"] / bn, 2),
            "oracle_avg_k": round(sb["orac_k_sum"] / bn, 4),
            "oracle_no_crop_pct": round(100.0 * sb["orac_nc"] / bn, 2),
            "mean_entropy": round(sb["ent_sum"] / bn, 4),
            "top1_window_hit_pct": round(100.0 * sb["top1_hits"] / max(sb["crop_gt"], 1), 2),
            "no_crop_precision_pct": bp,
            "no_crop_recall_pct": br,
            "no_crop_f1_pct": bf1,
        }

    return {
        "num_samples": total_samples,
        "total_loss": round(loss_sum / denom, 5),
        "bce_loss": round(bce_sum / denom, 5),
        "traj_nll_loss": round(nll_sum / denom, 5),
        "top1_window_hit_pct": round(100.0 * top1_hits / max(crop_gt_samples, 1), 2),
        "top2_window_hit_pct": round(100.0 * top2_any_hits / max(crop_gt_samples, 1), 2),
        "top2_window_recall_pct": round(100.0 * top2_target_hits / max(top2_target_total, 1), 2),
        "greedy_window_hit_pct": round(100.0 * greedy_any_hits / max(crop_gt_samples, 1), 2),
        "exact_action_set_match_pct": round(100.0 * exact_set_matches / denom, 2),
        "no_crop_precision_pct": nc_prec,
        "no_crop_recall_pct": nc_rec,
        "no_crop_f1_pct": nc_f1,
        "no_crop_confusion": {"tp": nc_tp, "fp": nc_fp, "fn": nc_fn, "tn": nc_tn},
        "mean_entropy": round(ent_sum / denom, 4),
        "pred_avg_k": round(pred_avg_k, 4),
        "pred_no_crop_pct": round(100.0 * pred_k_counts.get(0, 0) / denom, 2),
        "pred_k_pct": {
            "0": round(100.0 * pred_k_counts.get(0, 0) / denom, 2),
            "1": round(100.0 * pred_k_counts.get(1, 0) / denom, 2),
            "2": round(100.0 * pred_k_counts.get(2, 0) / denom, 2),
        },
        "oracle_avg_k": round(orac_avg_k, 4),
        "oracle_no_crop_pct": round(100.0 * oracle_k_counts.get(0, 0) / denom, 2),
        "pred_horizon_share_pct": round(100.0 * pred_horizon_crops / max(pred_total_crops, 1), 2),
        "by_view_type": view_breakdown,
    }


def export_sft_and_reference_checkpoints(
    policy: ActiveZoomPolicyHead,
    sft_path: str | Path,
    ref_path: str | Path,
    extra: Dict,
    verify_batch_state=None,
) -> Dict:
    """Save ``policy_sft.pth`` and bitwise-identical frozen reference ``policy_sft_ref.pth``.

    Verifies:
      1. ``clone_as_reference()`` state_dict is bitwise identical to ``policy``.
      2. Both files share identical SHA256 digest and reconstruct identically via
         ``ActiveZoomPolicyHead.from_checkpoint``.
      3. Forward logits on ``verify_batch_state`` are bitwise equal with ``D_KL == 0.0``.
    """
    sft_out = Path(sft_path).resolve()
    ref_out = Path(ref_path).resolve()

    policy_eval = copy.deepcopy(policy).cpu().eval()
    ref_policy = policy_eval.clone_as_reference()

    # 1) Verify clone_as_reference bitwise equality
    sd_p = policy_eval.state_dict()
    sd_r = ref_policy.state_dict()
    assert sd_p.keys() == sd_r.keys(), "state_dict keys diverged in clone_as_reference"
    for k in sd_p:
        assert torch.equal(sd_p[k], sd_r[k]), f"Tensor mismatch in clone_as_reference: {k}"

    # 2) Export policy_sft.pth atomically via ActiveZoomPolicyHead checkpoint schema
    ckpt_dict = policy_eval.to_checkpoint_dict(extra=extra)
    atomic_torch_save(ckpt_dict, sft_out)

    # 3) Write bitwise-identical policy_sft_ref.pth atomically from exact bytes
    raw_bytes = sft_out.read_bytes()
    ref_out.parent.mkdir(parents=True, exist_ok=True)
    tmp_ref = ref_out.with_name(f"{ref_out.name}.tmp.{os.getpid()}")
    try:
        tmp_ref.write_bytes(raw_bytes)
        os.replace(tmp_ref, ref_out)
    finally:
        if tmp_ref.exists():
            tmp_ref.unlink()

    sha_sft = sha256_file(sft_out)
    sha_ref = sha256_file(ref_out)
    if sha_sft != sha_ref:
        raise RuntimeError(f"Exported checkpoint SHA256 mismatch: sft={sha_sft} != ref={sha_ref}")

    # 4) Round-trip reload verification
    re_sft = ActiveZoomPolicyHead.from_checkpoint(sft_out, map_location="cpu").eval()
    re_ref = ActiveZoomPolicyHead.from_checkpoint(ref_out, map_location="cpu").eval()
    for p in re_ref.parameters():
        p.requires_grad = False

    for k in re_sft.state_dict():
        assert torch.equal(re_sft.state_dict()[k], re_ref.state_dict()[k]), (
            f"Reloaded state_dict mismatch for {k}"
        )

    max_logit_diff = 0.0
    max_kl = 0.0
    if verify_batch_state is not None:
        st_cpu = verify_batch_state.to("cpu")
        with torch.no_grad():
            l_sft = re_sft(st_cpu)
            l_ref = re_ref(st_cpu)
            assert torch.equal(l_sft, l_ref), "Reloaded policy_sft and policy_sft_ref logits are not bitwise equal!"
            max_logit_diff = float((l_sft - l_ref).abs().max().item())
            kl = re_sft.kl_divergence_from_logits(l_sft, l_ref)
            max_kl = float(kl.abs().max().item())
            assert max_kl == 0.0, f"Expected D_KL(pi_sft || pi_ref) == 0.0, got {max_kl}"

    return {
        "policy_sft_path": str(sft_out),
        "policy_sft_ref_path": str(ref_out),
        "policy_sft_sha256": sha_sft,
        "policy_sft_ref_sha256": sha_ref,
        "bitwise_identical_files": sha_sft == sha_ref,
        "reload_max_logit_diff": max_logit_diff,
        "reload_max_kl_divergence": max_kl,
    }


def train_sft_policy_offline(
    cache_dir: str | Path,
    output_dir: str | Path,
    config: SFTTrainConfig,
    sft_ckpt_name: str = "policy_sft.pth",
    ref_ckpt_name: str = "policy_sft_ref.pth",
    show_progress: bool = True,
) -> Dict:
    """Run pure offline Stage-1 SFT training of ``ActiveZoomPolicyHead`` without loading detector."""
    set_deterministic_seed(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Open & cryptographically verify source-only sharded cache
    train_reader = ShardedSFTCacheReader(
        cache_dir=cache_dir,
        split_role="source_train",
        verify_shard_hashes=True,
        require_completed=True,
        max_cached_shards_in_ram=max(2, config.shard_group_size),
    )
    dev_reader = ShardedSFTCacheReader(
        cache_dir=cache_dir,
        split_role="source_dev",
        verify_shard_hashes=False,  # already verified in train_reader constructor
        require_completed=True,
        max_cached_shards_in_ram=2,
    )

    manifest = train_reader.manifest
    manifest_digest = train_reader.manifest_digest
    validate_source_only_access(manifest["source_domain"], manifest["source_split"])

    # Compute source_train-only pos_weight
    pos_weight, pos_weight_stats = compute_source_train_pos_weight(
        train_reader=train_reader,
        max_pos_weight=config.max_pos_weight,
        no_crop_pos_weight_scale=config.no_crop_pos_weight_scale,
    )
    pw_dev = pos_weight.to(device=device, dtype=torch.float32)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    policy = build_policy_from_cache_manifest(
        manifest=manifest,
        use_film=config.use_film,
        dropout=config.dropout,
    ).to(device)

    param_breakdown = dict(policy.parameter_breakdown())
    trainable_params = policy.count_parameters(trainable_only=True)

    steps_per_epoch = max(1, math.ceil(len(train_reader) / config.batch_size))
    total_steps = max(1, steps_per_epoch * config.epochs)
    warmup_steps = int(round(config.warmup_epochs * steps_per_epoch))

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    def _lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_ratio = config.min_lr / max(config.lr, 1e-12)
        return min_ratio + (1.0 - min_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)

    print(
        f"[Phase 2 SFT] Training ActiveZoomPolicyHead offline (use_film={config.use_film}, "
        f"params={trainable_params:,}, batch_size={config.batch_size}, epochs={config.epochs}, "
        f"lr={config.lr}, traj_nll_weight={config.traj_nll_weight}, device={device})",
        flush=True,
    )
    print(
        f"[Phase 2 SFT] Verified source-only cache {manifest_digest[:16]}... | "
        f"train={len(train_reader)} samples, dev={len(dev_reader)} samples",
        flush=True,
    )

    # Evaluate untrained Epoch 0 baseline on source_dev for clear convergence reference
    dev_init = evaluate_policy_on_cache(
        policy=policy,
        reader=dev_reader,
        pos_weight=pos_weight,
        traj_nll_weight=config.traj_nll_weight,
        batch_size=config.batch_size,
        device=device,
    )
    print(
        f"  [Epoch 00 (init)] dev_loss={dev_init['total_loss']:.4f} "
        f"(bce={dev_init['bce_loss']:.4f}, nll={dev_init['traj_nll_loss']:.4f}) | "
        f"Top1={dev_init['top1_window_hit_pct']:.1f}% Top2={dev_init['top2_window_hit_pct']:.1f}% | "
        f"NoCrop F1={dev_init['no_crop_f1_pct']:.1f}% | H={dev_init['mean_entropy']:.3f} | "
        f"avg_K={dev_init['pred_avg_k']:.2f}",
        flush=True,
    )

    history: List[Dict] = []
    best_dev_loss = float("inf")
    best_epoch = 0
    best_dev_metrics: Optional[Dict] = None
    best_state_dict: Optional[Dict[str, Tensor]] = None

    t_train_start = time.perf_counter()
    global_step = 0

    for epoch in range(1, config.epochs + 1):
        t_ep0 = time.perf_counter()
        policy.train()

        tr_samples = 0
        tr_loss_sum = 0.0
        tr_bce_sum = 0.0
        tr_nll_sum = 0.0
        tr_grad_norm_sum = 0.0
        tr_steps = 0

        batch_iter = train_reader.iter_batches(
            batch_size=config.batch_size,
            shuffle=True,
            seed=config.seed + epoch * 9973,
            device=device,
            drop_last=False,
            shard_group_size=config.shard_group_size,
        )
        if show_progress:
            batch_iter = tqdm(batch_iter, total=steps_per_epoch, desc=f"SFT Epoch {epoch}/{config.epochs}", leave=False)

        for batch in batch_iter:
            state = batch["state"]
            targets = batch["sft_multi_hot"]
            oracle_actions = batch["oracle_actions"]
            bsz = state.batch_size

            optimizer.zero_grad(set_to_none=True)
            logits = policy(state)
            bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw_dev)
            nll = -policy.log_prob_from_logits(logits, oracle_actions).mean()
            loss = bce + float(config.traj_nll_weight) * nll

            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError(f"Non-finite SFT loss detected at epoch {epoch}, step {global_step}: {loss}")

            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm).item())
            optimizer.step()
            scheduler.step()
            global_step += 1

            tr_samples += bsz
            tr_loss_sum += float(loss.item()) * bsz
            tr_bce_sum += float(bce.item()) * bsz
            tr_nll_sum += float(nll.item()) * bsz
            tr_grad_norm_sum += grad_norm
            tr_steps += 1

        dev_metrics = evaluate_policy_on_cache(
            policy=policy,
            reader=dev_reader,
            pos_weight=pos_weight,
            traj_nll_weight=config.traj_nll_weight,
            batch_size=config.batch_size,
            device=device,
        )
        ep_sec = time.perf_counter() - t_ep0
        cur_lr = float(optimizer.param_groups[0]["lr"])

        train_metrics = {
            "num_samples": tr_samples,
            "total_loss": round(tr_loss_sum / max(tr_samples, 1), 5),
            "bce_loss": round(tr_bce_sum / max(tr_samples, 1), 5),
            "traj_nll_loss": round(tr_nll_sum / max(tr_samples, 1), 5),
            "mean_grad_norm": round(tr_grad_norm_sum / max(tr_steps, 1), 4),
            "lr_end": cur_lr,
        }

        is_best = dev_metrics["total_loss"] < best_dev_loss
        if is_best:
            best_dev_loss = dev_metrics["total_loss"]
            best_epoch = epoch
            best_dev_metrics = copy.deepcopy(dev_metrics)
            best_state_dict = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}

        ep_record = {
            "epoch": epoch,
            "elapsed_sec": round(ep_sec, 2),
            "is_best_dev": is_best,
            "source_train": train_metrics,
            "source_dev": dev_metrics,
        }
        history.append(ep_record)

        clean_dv = dev_metrics["by_view_type"]["clean"]
        deg_dv = dev_metrics["by_view_type"]["degraded"]
        star = " *BEST*" if is_best else ""
        print(
            f"  [Epoch {epoch:02d}/{config.epochs:02d}] "
            f"tr_loss={train_metrics['total_loss']:.4f} | "
            f"dev_loss={dev_metrics['total_loss']:.4f} "
            f"(bce={dev_metrics['bce_loss']:.4f}, nll={dev_metrics['traj_nll_loss']:.4f}) | "
            f"Top1={dev_metrics['top1_window_hit_pct']:.1f}% Top2={dev_metrics['top2_window_hit_pct']:.1f}% | "
            f"NC_F1={dev_metrics['no_crop_f1_pct']:.1f}% (P={dev_metrics['no_crop_precision_pct']:.1f}%/R={dev_metrics['no_crop_recall_pct']:.1f}%) | "
            f"H={dev_metrics['mean_entropy']:.3f} | "
            f"K={dev_metrics['pred_avg_k']:.2f} (clean_NC={clean_dv['pred_no_crop_pct']:.1f}%, deg_NC={deg_dv['pred_no_crop_pct']:.1f}%) | "
            f"{ep_sec:.1f}s{star}",
            flush=True,
        )

    assert best_state_dict is not None and best_dev_metrics is not None
    total_train_sec = time.perf_counter() - t_train_start

    peak_alloc_gib = (
        round(torch.cuda.max_memory_allocated(device) / (1024**3), 4) if device.type == "cuda" else 0.0
    )
    peak_res_gib = (
        round(torch.cuda.max_memory_reserved(device) / (1024**3), 4) if device.type == "cuda" else 0.0
    )

    # Restore best source_dev weights & export self-contained policy_sft.pth + policy_sft_ref.pth
    policy.load_state_dict(best_state_dict, strict=True)
    policy.eval()

    # Grab a real source_dev mini-batch (B<=8) to verify exact reload & D_KL(pi_sft || pi_ref) == 0
    verify_batch = next(
        dev_reader.iter_batches(batch_size=min(8, len(dev_reader)), shuffle=False, seed=0, device="cpu")
    )

    training_manifest = {
        "stage": "phase2_stage1_sft",
        "source_only_compliance": True,
        "source_domain": manifest["source_domain"],
        "source_split": manifest["source_split"],
        "cache_manifest_digest": manifest_digest,
        "train_config": config.to_dict(),
        "pos_weight_stats": pos_weight_stats,
        "model_selection_criterion": "lowest_source_dev_total_loss",
        "best_epoch": best_epoch,
        "best_source_dev_metrics": best_dev_metrics,
    }
    training_manifest_digest = sha256_json(training_manifest)

    ckpt_extra = {
        "training_manifest_digest": training_manifest_digest,
        "training_manifest": training_manifest,
        "cache_manifest_digest": manifest_digest,
        "best_epoch": best_epoch,
        "source_dev_metrics": best_dev_metrics,
    }
    export_audit = export_sft_and_reference_checkpoints(
        policy=policy,
        sft_path=out_dir / sft_ckpt_name,
        ref_path=out_dir / ref_ckpt_name,
        extra=ckpt_extra,
        verify_batch_state=verify_batch["state"],
    )

    return {
        "training_manifest_digest": training_manifest_digest,
        "training_manifest": training_manifest,
        "cache_manifest_digest": manifest_digest,
        "cache_stats": train_reader.manifest_doc["stats"],
        "cache_timing": train_reader.manifest_doc["timing"],
        "cache_split_config": manifest["split_config"],
        "policy_config": policy.config_dict(),
        "parameter_breakdown": param_breakdown,
        "epoch0_init_source_dev": dev_init,
        "best_epoch": best_epoch,
        "best_source_dev": best_dev_metrics,
        "history": history,
        "training_elapsed_sec": round(total_train_sec, 2),
        "training_gpu_memory_gib": {
            "peak_allocated_gib": peak_alloc_gib,
            "peak_reserved_gib": peak_res_gib,
        },
        "checkpoint_export": export_audit,
    }


# =============================================================================
# Real End-to-End Detection Verification on source_dev (Clean & Synthetic Degraded)
# =============================================================================


def _latency_stats(ms_list: Sequence[float]) -> Dict:
    if not ms_list:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "measured_e2e_fps": 0.0}
    arr = np.asarray(ms_list, dtype=np.float64)
    mean_ms = float(arr.mean())
    return {
        "mean_ms": round(mean_ms, 2),
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "measured_e2e_fps": round(1000.0 / max(mean_ms, 1e-6), 2),
    }


def _snapshot_bn_buffers(model: torch.nn.Module) -> Dict[str, Tensor]:
    snap: Dict[str, Tensor] = {}
    for name, buf in model.named_buffers():
        if any(k in name for k in ("running_mean", "running_var", "num_batches_tracked")):
            snap[name] = buf.detach().cpu().clone()
    return snap


def evaluate_sft_e2e_on_source_dev(
    policy_sft_path: str | Path,
    wrapper: FrozenTextStyleDetWrapper,
    loader: SourceOnlyDatasetLoader,
    dev_image_ids: Sequence[str],
    degradation_seed: int = 20260420,
    max_eval_images: int = 250,
    trunc_margin_px: float = 5.0,
    max_detections: int = 100,
    warmup_images: int = 2,
    show_progress: bool = True,
) -> Dict:
    """Evaluate Baseline, RPN Heuristic, AZ-Agent (SFT greedy), and Oracle.for_policy on source_dev.

    Strictly source-only (``daytime_clear/train`` dev partition; ZERO target-domain access).
    Evaluates both ``source_dev_clean`` and ``source_dev_degraded`` with strict P0-01 zero-GT
    isolation and genuine ``AZAgentDetector.predict`` end-to-end latency measurement.
    """
    validate_source_only_access(loader.domain, loader.split)
    eval_ids = list(dev_image_ids[: max_eval_images if max_eval_images > 0 else len(dev_image_ids)])
    if not eval_ids:
        raise ValueError("eval_ids is empty.")

    policy_sft = ActiveZoomPolicyHead.from_checkpoint(policy_sft_path, map_location=wrapper.device).eval()
    fuser = DualViewWBFFusion(
        global_weight=1.0,
        local_weight=0.85,
        iou_thr=0.55,
        trunc_margin_px=trunc_margin_px,
        max_detections=max_detections,
        fusion_mode="wbf",
    )
    state_encoder = StateEncoder(grid_size=policy_sft.default_grid_size)
    agent = AZAgentDetector(
        detector_wrapper=wrapper,
        policy_head=policy_sft,
        state_encoder=state_encoder,
        fuser=fuser,
        move_policy_to_detector_device=True,
    ).eval()
    heur_selector = RPNHeuristicSelector(agent.grid, agent.state_encoder, k_max=policy_sft.k_max)
    oracle_selector = OracleSelector.for_policy(policy_sft)
    assert oracle_selector.nms_window_iou == policy_sft.nms_window_iou

    bn_before = _snapshot_bn_buffers(wrapper.model)

    # Parse ONLY the selected source_dev XML annotations in memory (zero .pkl writes)
    records = loader.parse_records_on_demand(eval_ids)
    horizon_set = {w.window_id for w in agent.grid.windows if w.group == "horizon_band"}
    dummy_logits = torch.zeros((1, policy_sft.num_actions), dtype=torch.float32, device=wrapper.device)

    # GPU warmup before timing
    if wrapper.device.type == "cuda" and warmup_images > 0:
        for wid_warm in eval_ids[: min(warmup_images, len(eval_ids))]:
            w_img = loader.load_image(wid_warm)
            _ = agent.predict(w_img, mode="greedy", sync_cuda_for_timing=True)
        torch.cuda.synchronize(wrapper.device)
        torch.cuda.reset_peak_memory_stats(wrapper.device)

    methods = ("baseline", "rpn_heuristic_wbf", "az_agent_sft_greedy", "static_oracle_for_policy_wbf")
    views_results: Dict[str, Dict] = {}

    for view_mode in ("source_dev_clean", "source_dev_degraded"):
        pred_lines = {m: defaultdict(list) for m in methods}
        lat_ms = {
            "baseline": [],
            "rpn_heuristic_wbf": [],
            "az_agent_sft_greedy": [],
            "single_crop": [],
            "sft_non_crop_overhead": [],
        }
        crop_counts = {
            "rpn_heuristic_wbf": [],
            "az_agent_sft_greedy": [],
            "static_oracle_for_policy_wbf": [],
        }
        win_hists = {
            k: np.zeros(agent.grid.num_windows, dtype=np.int64) for k in crop_counts
        }
        per_tag_sft: Dict[str, Dict[str, int]] = defaultdict(lambda: {"n": 0, "k_sum": 0, "no_crop": 0})
        per_tag_oracle: Dict[str, Dict[str, int]] = defaultdict(lambda: {"n": 0, "k_sum": 0, "no_crop": 0})
        tag_counts: Dict[str, int] = defaultdict(int)

        iterator = tqdm(eval_ids, desc=f"E2E Eval [{view_mode}]", leave=True) if show_progress else eval_ids

        for img_id in iterator:
            base_pil = loader.load_image(img_id)
            if view_mode == "source_dev_clean":
                deg_tag = "clean"
            else:
                deg_tag = select_adverse_tag_for_view(img_id, view_index=0, base_seed=degradation_seed)
            tag_counts[deg_tag] += 1

            view_pil = apply_physical_degradation(
                base_pil,
                image_id=img_id,
                view_tag=deg_tag,
                seed=degradation_seed,
            )

            # -----------------------------------------------------------------
            # PART A-1: Standalone Zero-GT AZAgentDetector.predict (SFT greedy)
            # Also supplies exact global_detections = Baseline (No-Crop)
            # -----------------------------------------------------------------
            out_sft = agent.predict(view_pil, mode="greedy", sync_cuda_for_timing=True)
            base_det = out_sft.global_detections
            sft_det = out_sft.detections
            sft_act = out_sft.action

            lat_ms["baseline"].append(float(out_sft.timings_ms["global_perception_ms"]))
            lat_ms["az_agent_sft_greedy"].append(float(out_sft.timings_ms["total_ms"]))
            lat_ms["sft_non_crop_overhead"].append(
                float(out_sft.timings_ms["state_encode_ms"])
                + float(out_sft.timings_ms["policy_ms"])
                + float(out_sft.timings_ms["fusion_ms"])
            )
            crop_counts["az_agent_sft_greedy"].append(len(sft_act))
            for wid in sft_act:
                win_hists["az_agent_sft_greedy"][wid] += 1

            per_tag_sft[deg_tag]["n"] += 1
            per_tag_sft[deg_tag]["k_sum"] += len(sft_act)
            if len(sft_act) == 0:
                per_tag_sft[deg_tag]["no_crop"] += 1

            # -----------------------------------------------------------------
            # PART A-2: Standalone Zero-GT RPNHeuristicSelector + WBF
            # Executed strictly BEFORE touching GT for Oracle
            # -----------------------------------------------------------------
            if wrapper.device.type == "cuda":
                torch.cuda.synchronize(wrapper.device)
            t_h0 = time.perf_counter()
            perc = wrapper.forward_global(view_pil)
            orig_h, orig_w = perc.orig_size
            heur_wids, _ = heur_selector.select(perc)

            crop_cache: Dict[int, Dict[str, Tensor]] = {}
            for wid in heur_wids:
                rect = agent.grid.windows[wid].to_pixel_xyxy(orig_w, orig_h)
                if wrapper.device.type == "cuda":
                    torch.cuda.synchronize(wrapper.device)
                tc0 = time.perf_counter()
                c_det = wrapper.forward_single_crop(view_pil, rect)
                if wrapper.device.type == "cuda":
                    torch.cuda.synchronize(wrapper.device)
                lat_ms["single_crop"].append((time.perf_counter() - tc0) * 1000.0)
                crop_cache[wid] = c_det

            heur_det = fuser.fuse(perc.detections, [crop_cache[w] for w in heur_wids], perc.orig_size)
            if wrapper.device.type == "cuda":
                torch.cuda.synchronize(wrapper.device)
            lat_ms["rpn_heuristic_wbf"].append((time.perf_counter() - t_h0) * 1000.0)
            crop_counts["rpn_heuristic_wbf"].append(len(heur_wids))
            for wid in heur_wids:
                win_hists["rpn_heuristic_wbf"][wid] += 1

            # -----------------------------------------------------------------
            # PART B: Offline Static Oracle (for_policy, nms=0.35) + WBF
            # Cannot retroactively alter Baseline, SFT, or Heuristic outputs.
            # -----------------------------------------------------------------
            gt_boxes, gt_labels, gt_diff = record_to_0based_gt(records[img_id], wrapper.class_names)
            orac_wids, _ = oracle_selector.select(perc, gt_boxes, gt_labels, gt_diff)
            lp_ok = policy_sft.log_prob_from_logits(dummy_logits, [tuple(orac_wids)])
            assert bool(torch.isfinite(lp_ok).all().item()), f"Illegal Oracle trajectory: {orac_wids}"

            for wid in orac_wids:
                if wid not in crop_cache:
                    rect = agent.grid.windows[wid].to_pixel_xyxy(orig_w, orig_h)
                    if wrapper.device.type == "cuda":
                        torch.cuda.synchronize(wrapper.device)
                    tc0 = time.perf_counter()
                    c_det = wrapper.forward_single_crop(view_pil, rect)
                    if wrapper.device.type == "cuda":
                        torch.cuda.synchronize(wrapper.device)
                    lat_ms["single_crop"].append((time.perf_counter() - tc0) * 1000.0)
                    crop_cache[wid] = c_det

            orac_det = fuser.fuse(perc.detections, [crop_cache[w] for w in orac_wids], perc.orig_size)
            crop_counts["static_oracle_for_policy_wbf"].append(len(orac_wids))
            for wid in orac_wids:
                win_hists["static_oracle_for_policy_wbf"][wid] += 1

            per_tag_oracle[deg_tag]["n"] += 1
            per_tag_oracle[deg_tag]["k_sum"] += len(orac_wids)
            if len(orac_wids) == 0:
                per_tag_oracle[deg_tag]["no_crop"] += 1

            for m_name, m_det in (
                ("baseline", base_det),
                ("rpn_heuristic_wbf", heur_det),
                ("az_agent_sft_greedy", sft_det),
                ("static_oracle_for_policy_wbf", orac_det),
            ):
                append_voc_prediction_lines(
                    pred_lines[m_name],
                    image_id=img_id,
                    det_0based=m_det,
                    orig_size=perc.orig_size,
                    num_classes=wrapper.num_classes,
                    voc_1_based=True,
                )

        n_imgs = len(eval_ids)
        base_lat = _latency_stats(lat_ms["baseline"])
        heur_lat = _latency_stats(lat_ms["rpn_heuristic_wbf"])
        sft_lat = _latency_stats(lat_ms["az_agent_sft_greedy"])
        mean_crop_ms = float(np.mean(lat_ms["single_crop"])) if lat_ms["single_crop"] else 0.0
        mean_non_crop_overhead_ms = (
            float(np.mean(lat_ms["sft_non_crop_overhead"])) if lat_ms["sft_non_crop_overhead"] else 0.0
        )

        def _budget_summary(key: str) -> Dict:
            cnts = crop_counts[key]
            hist = win_hists[key]
            tot_c = int(hist.sum())
            avg_k = float(np.mean(cnts)) if cnts else 0.0
            nc = sum(c == 0 for c in cnts)
            hz = sum(int(hist[w]) for w in horizon_set)
            out = {
                "avg_k": round(avg_k, 4),
                "no_crop_count": int(nc),
                "no_crop_pct": round(100.0 * nc / max(n_imgs, 1), 2),
                "k_pct": {
                    "0": round(100.0 * sum(c == 0 for c in cnts) / max(n_imgs, 1), 2),
                    "1": round(100.0 * sum(c == 1 for c in cnts) / max(n_imgs, 1), 2),
                    "2": round(100.0 * sum(c == 2 for c in cnts) / max(n_imgs, 1), 2),
                },
                "total_crops": tot_c,
                "horizon_window_share_pct": round(100.0 * hz / max(tot_c, 1), 2),
            }
            if key == "rpn_heuristic_wbf":
                out["measured_e2e_fps"] = heur_lat["measured_e2e_fps"]
                out["latency_ms"] = heur_lat
            elif key == "az_agent_sft_greedy":
                out["measured_e2e_fps"] = sft_lat["measured_e2e_fps"]
                out["latency_ms"] = sft_lat
            else:
                # P2-03: Use measured non-crop overhead (state+policy+WBF) rather than hardcoded 3.5ms,
                # and mark GT-dependent Oracle clearly as offline diagnostic reference in reports.
                proj_ms = base_lat["mean_ms"] + mean_non_crop_overhead_ms + avg_k * mean_crop_ms
                out["projected_policy_fps"] = round(1000.0 / max(proj_ms, 1e-3), 2)
                out["offline_reference_only"] = True
            return out

        metrics_out: Dict[str, Dict] = {}
        for m_name in methods:
            metrics_out[m_name] = evaluate_voc_predictions(
                pred_lines_by_class=pred_lines[m_name],
                class_names=wrapper.class_names,
                annotation_records=records,
                use_07_metric=True,
                rn_root=wrapper.rn_root,
            )

        by_tag_summary = {}
        for tag in DEGRADATION_TAGS:
            if tag_counts.get(tag, 0) > 0:
                s_st = per_tag_sft[tag]
                o_st = per_tag_oracle[tag]
                by_tag_summary[tag] = {
                    "num_images": int(tag_counts[tag]),
                    "sft_avg_k": round(s_st["k_sum"] / max(s_st["n"], 1), 4),
                    "sft_no_crop_pct": round(100.0 * s_st["no_crop"] / max(s_st["n"], 1), 2),
                    "oracle_avg_k": round(o_st["k_sum"] / max(o_st["n"], 1), 4),
                    "oracle_no_crop_pct": round(100.0 * o_st["no_crop"] / max(o_st["n"], 1), 2),
                }

        views_results[view_mode] = {
            "num_images": n_imgs,
            "degradation_counts": dict(tag_counts),
            "metrics": metrics_out,
            "budget": {
                "baseline": {
                    "avg_k": 0.0,
                    "no_crop_pct": 100.0,
                    "measured_e2e_fps": base_lat["measured_e2e_fps"],
                    "latency_ms": base_lat,
                },
                "rpn_heuristic_wbf": _budget_summary("rpn_heuristic_wbf"),
                "az_agent_sft_greedy": _budget_summary("az_agent_sft_greedy"),
                "static_oracle_for_policy_wbf": _budget_summary("static_oracle_for_policy_wbf"),
                "mean_isolated_single_crop_ms": round(mean_crop_ms, 2),
            },
            "adaptive_by_degradation": by_tag_summary,
        }

    # Assert zero frozen BN buffer mutation during end-to-end eval
    bn_after = _snapshot_bn_buffers(wrapper.model)
    for k in bn_before:
        assert torch.equal(bn_before[k], bn_after[k]), f"Frozen detector BN mutated during E2E eval: {k}"

    peak_alloc_gib = (
        round(torch.cuda.max_memory_allocated(wrapper.device) / (1024**3), 3)
        if wrapper.device.type == "cuda"
        else 0.0
    )
    peak_res_gib = (
        round(torch.cuda.max_memory_reserved(wrapper.device) / (1024**3), 3)
        if wrapper.device.type == "cuda"
        else 0.0
    )

    # 2-view macro average (clean + degraded)
    macro_2view: Dict[str, Dict] = {}
    for m_name in methods:
        c_m = views_results["source_dev_clean"]["metrics"][m_name]
        d_m = views_results["source_dev_degraded"]["metrics"][m_name]
        c_b = views_results["source_dev_clean"]["budget"][m_name]
        d_b = views_results["source_dev_degraded"]["budget"][m_name]
        fps_key = "measured_e2e_fps" if "measured_e2e_fps" in c_b else "projected_policy_fps"
        macro_2view[m_name] = {
            "AP": round(0.5 * (c_m["AP"] + d_m["AP"]), 4),
            "AP50": round(0.5 * (c_m["AP50"] + d_m["AP50"]), 4),
            "AP75": round(0.5 * (c_m["AP75"] + d_m["AP75"]), 4),
            "weak_AP": round(0.5 * (c_m["weak_AP"] + d_m["weak_AP"]), 4),
            "weak_AP50": round(0.5 * (c_m["weak_AP50"] + d_m["weak_AP50"]), 4),
            "weak_AP75": round(0.5 * (c_m["weak_AP75"] + d_m["weak_AP75"]), 4),
            "avg_k": round(0.5 * (c_b["avg_k"] + d_b["avg_k"]), 4),
            "no_crop_pct": round(0.5 * (c_b["no_crop_pct"] + d_b["no_crop_pct"]), 2),
            fps_key: round(0.5 * (c_b[fps_key] + d_b[fps_key]), 2),
        }

    return {
        "source_domain": ALLOWED_SOURCE_DOMAIN,
        "source_split": ALLOWED_SOURCE_SPLIT,
        "split_role": "source_dev",
        "num_eval_images_per_view": len(eval_ids),
        "eval_ids_sha256": sha256_json(eval_ids),
        "degradation_seed": int(degradation_seed),
        "frozen_bn_verified_bitwise_identical": True,
        "e2e_gpu_memory_gib": {
            "peak_allocated_gib": peak_alloc_gib,
            "peak_reserved_gib": peak_res_gib,
        },
        "views": views_results,
        "macro_2view": macro_2view,
    }


# =============================================================================
# Deterministic Markdown Report Generator (100% Derived from Results JSON)
# =============================================================================


def generate_phase2_markdown_report(results: Dict, out_md: Path) -> None:
    """Deterministically generate ``phase2_sft_report.md`` purely from ``results`` JSON."""
    tr_res = results["sft_training"]
    c_stats = tr_res["cache_stats"]
    c_tr = c_stats["source_train"]
    c_dv = c_stats["source_dev"]
    c_time = tr_res["cache_timing"]
    t_cfg = tr_res["training_manifest"]["train_config"]
    ckpt_exp = tr_res["checkpoint_export"]
    best_dv = tr_res["best_source_dev"]
    e2e = results.get("source_dev_e2e_eval")

    lines: List[str] = []
    lines.append("# Phase 2 实验报告：源域合成退化缓存与 Stage-1 SFT 冷启动训练\n")
    lines.append(f"- **Single-DGOD 合规声明**：全程严格限定于源域 `daytime_clear/train`（确定性 `source_train / source_dev` 零重叠划分）与纯物理合成退化，**零触碰任何目标域图像、标注或 `daytime_clear/test`**。")
    lines.append(f"- **离线缓存 Manifest SHA256**：`{tr_res['cache_manifest_digest']}`")
    lines.append(f"- **SFT 训练 Manifest SHA256**：`{tr_res['training_manifest_digest']}`")
    lines.append(
        f"- **策略网络配置**：`ActiveZoomPolicyHead`（`use_film={t_cfg['use_film']}`，可训练参数 `{tr_res['policy_config']['trainable_parameters']:,}`，"
        f"`k_max={tr_res['policy_config']['k_max']}`，`nms_window_iou={tr_res['policy_config']['nms_window_iou']}`）"
    )
    lines.append(
        f"- **导出权重与 $\\pi_{{\\text{{ref}}}}$ 一致性**：`policy_sft.pth` 与 `policy_sft_ref.pth` SHA256 逐位一致（`{ckpt_exp['policy_sft_sha256'][:16]}...`），"
        f"重载验证 `max_logit_diff = {ckpt_exp['reload_max_logit_diff']:.1e}`，`D_KL(pi_sft || pi_ref) = {ckpt_exp['reload_max_kl_divergence']:.1e}`。"
    )
    lines.append(
        f"- **SFT 离线训练峰值显存与耗时**：分配峰值 `{tr_res['training_gpu_memory_gib']['peak_allocated_gib']:.4f} GiB`（保留 `{tr_res['training_gpu_memory_gib']['peak_reserved_gib']:.4f} GiB`），"
        f"总耗时 `{tr_res['training_elapsed_sec']:.1f} s`，最优轮次 **Epoch {tr_res['best_epoch']}**（严格按 `source_dev` 总损失锁定）。\n"
    )

    lines.append("## 一、源域离线状态缓存与 `OracleSelector.for_policy` 伪标签统计\n")
    lines.append(
        f"- **总规模与存储**：总计 `{c_stats['total_samples']}` 条视图样本（`source_train={c_tr['num_samples']}`、`source_dev={c_dv['num_samples']}`），"
        f"总磁盘体积 `{c_stats['total_size_gib']:.3f} GiB`（`{c_stats['total_size_mib']:.1f} MiB`），"
        f"构建耗时 `{c_time['total_elapsed_sec']:.1f} s`（均值 `{c_time['mean_sec_per_view']:.3f} s/view`）。\n"
    )
    lines.append(
        "| 划分 | 样本数 | 分片数 | 体积 (MiB) | `K=0` (No-Crop) | `K=1` | `K=2` | 平均 $\\bar{K}$ | 清晰图 No-Crop% | 退化图 No-Crop% | 远景窗占比 |"
    )
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for role_lbl, st in (("`source_train`", c_tr), ("`source_dev`", c_dv)):
        lines.append(
            f"| {role_lbl} | {st['num_samples']} | {st['num_shards']} | {st['size_mib']:.1f} | "
            f"{st['k_pct']['0']:.2f}% | {st['k_pct']['1']:.2f}% | {st['k_pct']['2']:.2f}% | "
            f"{st['avg_k']:.3f} | {st['clean_no_crop_pct']:.2f}% | {st['degraded_no_crop_pct']:.2f}% | "
            f"{st['horizon_window_share_pct']:.2f}% |"
        )
    lines.append("")
    lines.append("**退化视图分布**：")
    lines.append(f"- `source_train`: `{c_tr['degradation_counts']}`")
    lines.append(f"- `source_dev`: `{c_dv['degradation_counts']}`\n")

    lines.append("## 二、Stage-1 SFT 离线训练收敛曲线（`source_train` / `source_dev`）\n")
    lines.append(
        "| Epoch | Train Loss | Dev Loss | Dev BCE | Dev NLL | Dev Top-1 命中 | Dev Top-2 命中 | Dev No-Crop P/R/F1 | Dev 熵 $\\mathcal{H}$ | Dev $\\bar{K}$ | Clean→Deg No-Crop% |"
    )
    lines.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    init_dv = tr_res["epoch0_init_source_dev"]
    lines.append(
        f"| 0 (init) | — | {init_dv['total_loss']:.4f} | {init_dv['bce_loss']:.4f} | {init_dv['traj_nll_loss']:.4f} | "
        f"{init_dv['top1_window_hit_pct']:.1f}% | {init_dv['top2_window_hit_pct']:.1f}% | "
        f"{init_dv['no_crop_precision_pct']:.1f}/{init_dv['no_crop_recall_pct']:.1f}/{init_dv['no_crop_f1_pct']:.1f}% | "
        f"{init_dv['mean_entropy']:.3f} | {init_dv['pred_avg_k']:.2f} | "
        f"{init_dv['by_view_type']['clean']['pred_no_crop_pct']:.1f}% → {init_dv['by_view_type']['degraded']['pred_no_crop_pct']:.1f}% |"
    )
    for ep in tr_res["history"]:
        tr_e = ep["source_train"]
        dv_e = ep["source_dev"]
        tag = f"**{ep['epoch']}***" if ep["epoch"] == tr_res["best_epoch"] else str(ep["epoch"])
        lines.append(
            f"| {tag} | {tr_e['total_loss']:.4f} | {dv_e['total_loss']:.4f} | {dv_e['bce_loss']:.4f} | {dv_e['traj_nll_loss']:.4f} | "
            f"{dv_e['top1_window_hit_pct']:.1f}% | {dv_e['top2_window_hit_pct']:.1f}% | "
            f"{dv_e['no_crop_precision_pct']:.1f}/{dv_e['no_crop_recall_pct']:.1f}/{dv_e['no_crop_f1_pct']:.1f}% | "
            f"{dv_e['mean_entropy']:.3f} | {dv_e['pred_avg_k']:.2f} | "
            f"{dv_e['by_view_type']['clean']['pred_no_crop_pct']:.1f}% → {dv_e['by_view_type']['degraded']['pred_no_crop_pct']:.1f}% |"
        )
    lines.append("")

    if e2e is not None:
        lines.append(
            f"## 三、`source_dev` 真实端到端检测验收（清晰图 vs 合成退化图，各 `{e2e['num_eval_images_per_view']}` 张，零目标域泄漏）\n"
        )
        lines.append(
            f"- **端到端推理显存（冻结 StyleDetV6 + AZAgentDetector FP32）**："
            f"峰值已分配 `{e2e['e2e_gpu_memory_gib']['peak_allocated_gib']:.3f} GiB`，"
            f"峰值保留 `{e2e['e2e_gpu_memory_gib']['peak_reserved_gib']:.3f} GiB`；冻结 BN 缓冲区逐位校验一致。\n"
        )

        method_rows = [
            ("baseline", "Baseline (No-Crop)"),
            ("rpn_heuristic_wbf", "RPNHeuristicSelector + WBF"),
            ("az_agent_sft_greedy", "**AZ-Agent (SFT, greedy)**"),
            ("static_oracle_for_policy_wbf", "Static Oracle (for_policy) + WBF"),
        ]

        for view_key, view_title in (
            ("source_dev_clean", "3.1 `source_dev` 清晰图（Clean）"),
            ("source_dev_degraded", "3.2 `source_dev` 物理合成退化图（Synthetic Adverse）"),
        ):
            v_data = e2e["views"][view_key]
            base_m = v_data["metrics"]["baseline"]
            lines.append(f"### {view_title}（`{v_data['num_images']}` 张，退化分布 `{v_data['degradation_counts']}`）\n")
            lines.append("| 方案 | AP50 (Δ) | AP75 (Δ) | AP (50:95) | weak_AP50 | avg_K | No-Crop% | FPS |")
            lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
            for m_key, m_label in method_rows:
                m = v_data["metrics"][m_key]
                b = v_data["budget"][m_key]
                d50 = m["AP50"] - base_m["AP50"]
                d75 = m["AP75"] - base_m["AP75"]
                s50 = f"{m['AP50']:.2f}" if m_key == "baseline" else f"{m['AP50']:.2f} ({d50:+.2f})"
                s75 = f"{m['AP75']:.2f}" if m_key == "baseline" else f"{m['AP75']:.2f} ({d75:+.2f})"
                fps_s = (
                    f"{b['measured_e2e_fps']:.2f} (实测)"
                    if "measured_e2e_fps" in b
                    else "— (离线参考)"
                )
                lines.append(
                    f"| {m_label} | {s50} | {s75} | {m['AP']:.2f} | {m['weak_AP50']:.2f} | "
                    f"{b['avg_k']:.2f} | {b['no_crop_pct']:.1f}% | {fps_s} |"
                )
            lines.append("")

        # 2-view macro average
        m2 = e2e["macro_2view"]
        b2 = m2["baseline"]
        lines.append("### 3.3 `source_dev` 双视图宏平均（Clean + Synthetic Degraded）\n")
        lines.append("| 方案 | AP50 (Δ) | AP75 (Δ) | AP (50:95) | weak_AP50 | avg_K | No-Crop% | FPS |")
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for m_key, m_label in method_rows:
            m = m2[m_key]
            d50 = m["AP50"] - b2["AP50"]
            d75 = m["AP75"] - b2["AP75"]
            s50 = f"{m['AP50']:.2f}" if m_key == "baseline" else f"{m['AP50']:.2f} ({d50:+.2f})"
            s75 = f"{m['AP75']:.2f}" if m_key == "baseline" else f"{m['AP75']:.2f} ({d75:+.2f})"
            fps_s = (
                f"{m['measured_e2e_fps']:.2f} (实测)"
                if "measured_e2e_fps" in m
                else "— (离线参考)"
            )
            lines.append(
                f"| {m_label} | {s50} | {s75} | {m['AP']:.2f} | {m['weak_AP50']:.2f} | "
                f"{m['avg_k']:.2f} | {m['no_crop_pct']:.1f}% | {fps_s} |"
            )
        lines.append("")

        # Adaptive behavior breakdown across weather tags
        lines.append("### 3.4 清晰 vs 各退化类型自适应裁剪行为（`AZ-Agent SFT` vs `Oracle`）\n")
        lines.append("| 视图 / 退化类型 | 图像数 | SFT `avg_K` | SFT `No-Crop%` | Oracle `avg_K` | Oracle `No-Crop%` |")
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
        c_adapt = e2e["views"]["source_dev_clean"]["adaptive_by_degradation"].get("clean")
        if c_adapt:
            lines.append(
                f"| `clean` | {c_adapt['num_images']} | {c_adapt['sft_avg_k']:.2f} | {c_adapt['sft_no_crop_pct']:.1f}% | "
                f"{c_adapt['oracle_avg_k']:.2f} | {c_adapt['oracle_no_crop_pct']:.1f}% |"
            )
        for tag in ("low_light", "fog", "rain", "compound_adverse"):
            d_adapt = e2e["views"]["source_dev_degraded"]["adaptive_by_degradation"].get(tag)
            if d_adapt:
                lines.append(
                    f"| `{tag}` | {d_adapt['num_images']} | {d_adapt['sft_avg_k']:.2f} | {d_adapt['sft_no_crop_pct']:.1f}% | "
                    f"{d_adapt['oracle_avg_k']:.2f} | {d_adapt['oracle_no_crop_pct']:.1f}% |"
                )
        lines.append("")

        # Objective auto-computed summary
        c_base = e2e["views"]["source_dev_clean"]["metrics"]["baseline"]
        c_sft = e2e["views"]["source_dev_clean"]["metrics"]["az_agent_sft_greedy"]
        c_sft_b = e2e["views"]["source_dev_clean"]["budget"]["az_agent_sft_greedy"]
        d_base = e2e["views"]["source_dev_degraded"]["metrics"]["baseline"]
        d_sft = e2e["views"]["source_dev_degraded"]["metrics"]["az_agent_sft_greedy"]
        d_sft_b = e2e["views"]["source_dev_degraded"]["budget"]["az_agent_sft_greedy"]
        lines.append("## 四、客观数据结论（脚本自动计算）\n")
        lines.append(
            f"1. **离线收敛与策略分布**：Epoch `{tr_res['best_epoch']}` 在 `source_dev` 达到最低总损失 `{best_dv['total_loss']:.4f}`"
            f"（Top-1 窗口命中 `{best_dv['top1_window_hit_pct']:.1f}%`，Top-2 命中 `{best_dv['top2_window_hit_pct']:.1f}%`，"
            f"`No-Crop F1 = {best_dv['no_crop_f1_pct']:.1f}%`，策略熵 $\\mathcal{{H}}={best_dv['mean_entropy']:.3f}$）。"
        )
        lines.append(
            f"2. **`source_dev` 端到端实测增益**：清晰图上 `AZ-Agent (SFT)` 相对基线 `AP50 {c_sft['AP50'] - c_base['AP50']:+.2f}` / "
            f"`AP75 {c_sft['AP75'] - c_base['AP75']:+.2f}` / `weak_AP50 {c_sft['weak_AP50'] - c_base['weak_AP50']:+.2f}`"
            f"（$\\bar{{K}}={c_sft_b['avg_k']:.2f}$，`No-Crop={c_sft_b['no_crop_pct']:.1f}%`）；"
            f"合成退化图上相对基线 `AP50 {d_sft['AP50'] - d_base['AP50']:+.2f}` / "
            f"`AP75 {d_sft['AP75'] - d_base['AP75']:+.2f}` / `weak_AP50 {d_sft['weak_AP50'] - d_base['weak_AP50']:+.2f}`"
            f"（$\\bar{{K}}={d_sft_b['avg_k']:.2f}$，`No-Crop={d_sft_b['no_crop_pct']:.1f}%`）。"
        )
        lines.append(
            f"3. **参考策略锁定就绪**：`policy_sft_ref.pth` 已完成与 `policy_sft.pth` 的逐位校验及重载 `D_KL=0` 验收，"
            f"可直接作为 Phase 3 GRPO 强化学习的冻结参考策略 $\\pi_{{\\text{{ref}}}}$。"
        )

    atomic_write_text(out_md, "\n".join(lines) + "\n")
