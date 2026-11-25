#!/usr/bin/env python
"""Phase 1 Comprehensive Unit & Real-Image Integration Test Suite.

Verifies all Phase 1 deliverables and audit constraints:
  1. Parameter budget (~2M target) & RTX 5060 single-image policy forward latency (< 2.0 ms).
  2. Ablation & batch-size checks (use_film=True/False, B=1/8), RoIAlign H/W axis alignment,
     grad-safe inference_mode guard, and offline cache round-trip.
  3. Exact probability axioms over all enumerated legal trajectories A_{<=2}:
     sum_a exp(log pi_theta(a|s)) == 1.0 (err < 1e-5), exact analytical entropy vs enumeration,
     D_KL(pi_theta || pi_theta) == 0, D_KL(pi_theta || pi_ref) > 0 & matches enumeration,
     illegal-action rejection, and zero-NaN gradients through entropy & KL.
  4. Gradient flow & frozen StyleDetV6 protection: all trainable params in ActiveZoomPolicyHead
     receive non-zero finite gradients (under both composite SFT+GRPO and pure GRPO loss),
     while frozen StyleDetV6 stays in eval(), grad is None, and BN stats are bitwise unchanged.
  5. Real-image end-to-end integration on daytime_clear & daytime_foggy via AZAgentDetector
     (fast path No-Crop & slow path K=1/2 crops), VRAM peak < 2.0 GiB, and read-only boundary
     integrity on E:\\textstyleDet\\textstyleDet_rn and E:\\textstyleDet\\data\\sky.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.ops import roi_align

from run_phase0 import DEFAULT_CKPT, DEFAULT_CLIP_RN101, DEFAULT_SKY_DIR, snapshot_readonly_tree
from zoom_agent import (
    AZAgentDetector,
    ActiveZoomPolicyHead,
    AgentBatchState,
    CandidateGrid,
    DualViewWBFFusion,
    FrozenTextStyleDetWrapper,
    GlobalPerceptionOutput,
    OracleSelector,
    StateEncoder,
    ensure_grad_safe_fp32,
)
from zoom_agent.eval_utils import format_voc_prediction_lines, load_domain_records, record_to_0based_gt


def _make_synthetic_state(
    batch_size: int,
    device: torch.device,
    visual_channels: int = 1024,
    grid_size: Tuple[int, int] = (12, 20),
    seed: int = 42,
) -> AgentBatchState:
    g = torch.Generator(device="cpu").manual_seed(seed)
    hg, wg = grid_size
    c4 = torch.randn((batch_size, visual_channels, hg, wg), generator=g, dtype=torch.float32)
    cues = torch.rand((batch_size, 5, hg, wg), generator=g, dtype=torch.float32)
    mu = torch.randn((batch_size, visual_channels), generator=g, dtype=torch.float32)
    log_sigma = torch.randn((batch_size, visual_channels), generator=g, dtype=torch.float32) * 0.5 - 1.0
    style_stats = torch.cat([mu, log_sigma], dim=-1)
    return AgentBatchState(
        c4_pooled=c4.to(device),
        cue_maps=cues.to(device),
        style_stats=style_stats.to(device),
        orig_sizes=tuple((720, 1280) for _ in range(batch_size)),
        resized_sizes=tuple((619, 1100) for _ in range(batch_size)),
        canvas_sizes=tuple((640, 1120) for _ in range(batch_size)),
        image_ids=tuple(f"synthetic_{i}" for i in range(batch_size)),
    )


def test_1_params_and_latency(device: torch.device) -> Dict:
    print("\n" + "=" * 78)
    print("[TEST 1] Parameter Count Breakdown & Single-Image Policy Latency")
    print("=" * 78)

    policy_film = ActiveZoomPolicyHead(use_film=True).to(device).eval()
    policy_nofilm = ActiveZoomPolicyHead(use_film=False).to(device).eval()
    policy_swin = ActiveZoomPolicyHead(visual_channels=256, use_film=True).to(device).eval()

    bd_film = policy_film.parameter_breakdown()
    bd_nofilm = policy_nofilm.parameter_breakdown()
    bd_swin = policy_swin.parameter_breakdown()

    print(f"{'Submodule':26s} | {'w/ FiLM (RN101)':>16s} | {'w/o FiLM (RN101)':>16s} | {'w/ FiLM (Swin256)':>17s}")
    print("-" * 82)
    for k in bd_film:
        sep = "-" * 82 if k == "total" else ""
        if sep:
            print(sep)
        print(f"{k:26s} | {bd_film[k]:16,d} | {bd_nofilm[k]:16,d} | {bd_swin[k]:17,d}")

    total_film = bd_film["total"]
    assert 1_500_000 <= total_film <= 2_500_000, f"Total params {total_film:,} outside [1.5M, 2.5M]"
    assert policy_film.film_head is not None
    assert float(policy_film.film_head.weight.abs().max().item()) == 0.0, "FiLM final weight must be zero-init"
    assert float(policy_film.film_head.bias.abs().max().item()) == 0.0, "FiLM final bias must be zero-init"

    # Benchmark single-image policy forward (B=1) with proper GPU clock warmup & sync
    state_b1 = _make_synthetic_state(batch_size=1, device=device, seed=101)
    state_b1_alt = _make_synthetic_state(batch_size=1, device=device, seed=202)
    warmup_iters = 100
    bench_iters = 500

    with torch.no_grad():
        if device.type == "cuda":
            # Wake laptop GPU out of idle P8 power state (as in real pipeline after StyleDetV6)
            wake = torch.randn((1024, 1024), device=device)
            for _ in range(25):
                wake = wake @ wake
                wake = wake / wake.norm().clamp_min(1e-6)
            del wake

        for _ in range(warmup_iters):
            _ = policy_film(state_b1)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        # 1) Pure Eager FP32 (amortized + per-iter synchronized wall-clock)
        t_am0 = time.perf_counter()
        for _ in range(bench_iters):
            _ = policy_film(state_b1)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        eager_amort_ms = (time.perf_counter() - t_am0) * 1000.0 / bench_iters

        eager_out_1 = policy_film(state_b1).clone()
        eager_out_2 = policy_film(state_b1_alt).clone()

        eager_ms: List[float] = []
        for _ in range(bench_iters):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            _ = policy_film(state_b1)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            eager_ms.append((time.perf_counter() - t0) * 1000.0)

        # 2) Eval CUDA-Graph FP32 (dynamic copy_ + replay + clone, zero WDDM launch jitter)
        graph_arr = None
        greedy_arr = None
        if device.type == "cuda":
            policy_film.enable_eval_cuda_graph(True)
            g_out_1 = policy_film(state_b1)
            g_out_2 = policy_film(state_b1_alt)
            max_diff_1 = float((eager_out_1 - g_out_1).abs().max().item())
            max_diff_2 = float((eager_out_2 - g_out_2).abs().max().item())
            assert max_diff_1 < 1e-4 and max_diff_2 < 1e-4, (
                f"CUDA-Graph eval mismatch vs eager: diff1={max_diff_1}, diff2={max_diff_2}"
            )
            assert not torch.allclose(g_out_1, g_out_2), "CUDA-Graph must reflect dynamic input changes"

            for _ in range(warmup_iters):
                _ = policy_film(state_b1)
            torch.cuda.synchronize(device)

            g_ms: List[float] = []
            for _ in range(bench_iters):
                torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                _ = policy_film(state_b1)
                torch.cuda.synchronize(device)
                g_ms.append((time.perf_counter() - t0) * 1000.0)
            graph_arr = np.asarray(g_ms, dtype=np.float64)

            grd_ms: List[float] = []
            for _ in range(bench_iters):
                torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                _ = policy_film.act_greedy(state_b1)
                torch.cuda.synchronize(device)
                grd_ms.append((time.perf_counter() - t0) * 1000.0)
            greedy_arr = np.asarray(grd_ms, dtype=np.float64)
            policy_film.enable_eval_cuda_graph(False)

    eager_arr = np.asarray(eager_ms, dtype=np.float64)
    dev_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(
        f"\n[Latency on {dev_name}, B=1, {bench_iters} runs after {warmup_iters} warmup]\n"
        f"  Eager FP32 [default] (amortized)  : mean={eager_amort_ms:.3f} ms\n"
        f"  Eager FP32 [default] (wall-sync)  : mean={eager_arr.mean():.3f} ms | p50={np.percentile(eager_arr, 50):.3f} ms | p95={np.percentile(eager_arr, 95):.3f} ms"
    )
    if graph_arr is not None and greedy_arr is not None:
        print(
            f"  Eval CUDA-Graph FP32 [opt-in]     : mean={graph_arr.mean():.3f} ms | p50={np.percentile(graph_arr, 50):.3f} ms | p95={np.percentile(graph_arr, 95):.3f} ms (max_diff={max(max_diff_1, max_diff_2):.2e})\n"
            f"  Eval Graph [opt-in] + act_greedy  : mean={greedy_arr.mean():.3f} ms | p50={np.percentile(greedy_arr, 50):.3f} ms | p95={np.percentile(greedy_arr, 95):.3f} ms"
        )
        assert graph_arr.mean() < 2.0, f"CUDA-Graph forward mean {graph_arr.mean():.3f} ms >= 2.0 ms"
        assert float(np.percentile(graph_arr, 50)) < 2.0, "CUDA-Graph forward p50 >= 2.0 ms"
        assert float(np.percentile(graph_arr, 95)) < 2.0, "CUDA-Graph forward p95 >= 2.0 ms"
        assert greedy_arr.mean() < 2.0, f"CUDA-Graph act_greedy mean {greedy_arr.mean():.3f} ms >= 2.0 ms"
    else:
        assert eager_arr.mean() < 2.0, f"Policy forward mean latency {eager_arr.mean():.3f} ms >= 2.0 ms"

    print(
        f"[PASS] TEST 1: Total params = {total_film:,} ({total_film / 1e6:.3f}M); "
        f"Eval CUDA-Graph FP32 < 2.0 ms (Eager mean={eager_arr.mean():.3f} ms, p95={np.percentile(eager_arr, 95):.3f} ms)."
    )

    return {
        "breakdown_film": dict(bd_film),
        "breakdown_nofilm": dict(bd_nofilm),
        "eager_amort_ms": float(eager_amort_ms),
        "fwd_mean_ms": float(graph_arr.mean()) if graph_arr is not None else float(eager_arr.mean()),
        "fwd_p50_ms": float(np.percentile(graph_arr, 50)) if graph_arr is not None else float(np.percentile(eager_arr, 50)),
        "fwd_p95_ms": float(np.percentile(graph_arr, 95)) if graph_arr is not None else float(np.percentile(eager_arr, 95)),
        "eager_mean_ms": float(eager_arr.mean()),
        "eager_p50_ms": float(np.percentile(eager_arr, 50)),
        "eager_p95_ms": float(np.percentile(eager_arr, 95)),
    }


def test_2_ablations_batch_roi_and_grad_bridge(device: torch.device) -> None:
    print("\n" + "=" * 78)
    print("[TEST 2] Ablation (use_film=True/False), Batch (B=1/8), RoIAlign Axes & Grad Bridge")
    print("=" * 78)

    # 2a. use_film=True/False x B=1/8
    for use_film in (True, False):
        policy = ActiveZoomPolicyHead(use_film=use_film).to(device).eval()
        for bsz in (1, 8):
            st = _make_synthetic_state(batch_size=bsz, device=device, seed=200 + bsz)
            logits = policy(st)
            assert logits.shape == (bsz, 19), f"Unexpected logits shape {tuple(logits.shape)}"
            assert torch.isfinite(logits).all(), f"Non-finite logits for use_film={use_film}, B={bsz}"

            g_out = policy.act_greedy(st)
            assert len(g_out.actions) == bsz
            assert g_out.log_probs.shape == (bsz,)
            assert g_out.action_probs.shape == (bsz,)
            assert g_out.step1_probs.shape == (bsz, 19)
            assert torch.allclose(g_out.step1_probs.sum(dim=-1), torch.ones(bsz, device=device), atol=1e-5)

            s_out = policy.sample_group(st, G=4)
            assert len(s_out.actions) == bsz and all(len(row) == 4 for row in s_out.actions)
            assert s_out.log_probs.shape == (bsz, 4)
            lp_recheck = policy.log_prob(st, s_out.actions)
            assert torch.allclose(s_out.log_probs, lp_recheck, atol=1e-5)
    print("[PASS] 2a: use_film in {True, False} x B in {1, 8} shapes, finiteness & sampling verified.")

    # 2b. Verify RoIAlign x*W_g (20) and y*H_g (12) — catches any H/W swap bug
    policy = ActiveZoomPolicyHead(use_film=True).to(device).eval()
    hg, wg = 12, 20
    # Place bright asymmetric impulse in top-right corner: y in [0, 3) (0..0.25), x in [15, 20) (0.75..1.0)
    # Window 3 (grid_r0_c3: x in [0.65, 1.0], y in [0.0, 0.42]) MUST capture it.
    # Window 8 (grid_r2_c0: x in [0.0, 0.35], y in [0.58, 1.0]) and Window 0 (top-left) MUST get 0.
    # If x/y were swapped, x=[0.65..1.0]*12=[7.8..12] and y=[0..0.42]*20=[0..8.4] would miss x in [15, 20)!
    z_probe = torch.zeros((1, 1, hg, wg), device=device, dtype=torch.float32)
    z_probe[0, 0, 0:3, 15:20] = 10.0
    rois = policy.build_roi_boxes_for_grid(batch_size=1, grid_h=hg, grid_w=wg, dtype=torch.float32)
    pooled = roi_align(z_probe, rois, output_size=(3, 3), spatial_scale=1.0, sampling_ratio=2, aligned=True)
    win_means = pooled.mean(dim=(1, 2, 3)).cpu()
    assert float(win_means[3].item()) > 2.5, f"Top-right window 3 missed impulse: {win_means[3].item():.4f}"
    assert float(win_means[0].item()) == 0.0, f"Top-left window 0 leaked impulse: {win_means[0].item():.4f}"
    assert float(win_means[8].item()) == 0.0, f"Bottom-left window 8 leaked impulse: {win_means[8].item():.4f}"
    assert float(win_means[11].item()) == 0.0, f"Bottom-right window 11 leaked impulse: {win_means[11].item():.4f}"
    print(
        f"[PASS] 2b: RoIAlign (H_g=12, W_g=20, x*W_g, y*H_g, aligned=True) axis check passed "
        f"(win3={win_means[3].item():.2f}, win0/8/11=0.00)."
    )

    # 2c. Inference-mode rejection + grad-safe conversion + cache round-trip + collate
    with torch.inference_mode():
        inf_c4 = torch.randn((1, 1024, 12, 20), device=device)
        inf_cues = torch.rand((1, 5, 12, 20), device=device)
        inf_style = torch.randn((1, 2048), device=device)
    assert inf_c4.is_inference()
    rejected = False
    try:
        _ = AgentBatchState(c4_pooled=inf_c4, cue_maps=inf_cues, style_stats=inf_style)
    except RuntimeError:
        rejected = True
    assert rejected, "AgentBatchState must reject torch.inference_mode() tensors!"

    safe_state = AgentBatchState(
        c4_pooled=ensure_grad_safe_fp32(inf_c4, device=device),
        cue_maps=ensure_grad_safe_fp32(inf_cues, device=device),
        style_stats=ensure_grad_safe_fp32(inf_style, device=device),
    )
    assert not safe_state.c4_pooled.is_inference()
    # Verify backward succeeds using converted tensor
    policy_train = ActiveZoomPolicyHead(use_film=True).to(device).train()
    loss_safe = policy_train(safe_state).sum()
    loss_safe.backward()

    # Cache round-trip & collate
    st_a = _make_synthetic_state(1, device, seed=11)
    st_b = _make_synthetic_state(1, device, seed=22)
    restored_a = AgentBatchState.from_cache_dict(st_a.to_cache_dict(torch.float16), device=device)
    restored_b = AgentBatchState.from_cache_dict(st_b.to_cache_dict(torch.float16), device=device)
    collated = AgentBatchState.collate([restored_a, restored_b], device=device)
    assert collated.batch_size == 2 and collated.device == device and collated.c4_pooled.dtype == torch.float32
    assert not collated.c4_pooled.is_inference()
    print("[PASS] 2c: Inference-mode guard, grad-safe FP32 conversion, cache round-trip & collate verified.")

    # 2d. Targeted Red-Team Audit Counterexample Regressions (P1-01, P1-02, P1-03, P2-2, P2-3, P2-4)
    # [P1-01 + P2-4] Outer torch.inference_mode() -> RoI cache (B=2 & custom grid), CUDA Graph, FP16->FP32 -> train() backward
    with torch.inference_mode():
        fp16_in = torch.randn((1, 1024, 12, 20), device=device, dtype=torch.float16)
        fp32_out = ensure_grad_safe_fp32(fp16_in, device=device)
        assert not fp32_out.is_inference() and fp32_out.dtype == torch.float32 and not fp32_out.requires_grad

    for test_dev in (torch.device("cpu"), device):
        pol_sw = ActiveZoomPolicyHead(use_film=True).to(test_dev).eval()
        st_b2 = _make_synthetic_state(2, test_dev, grid_size=(12, 20), seed=501)
        st_custom = _make_synthetic_state(2, test_dev, grid_size=(8, 12), seed=502)
        st_b1 = _make_synthetic_state(1, test_dev, grid_size=(12, 20), seed=503)
        if test_dev.type == "cuda":
            pol_sw.enable_eval_cuda_graph(True)
        with torch.inference_mode():
            _ = pol_sw(st_b2)
            _ = pol_sw(st_custom)
            _ = pol_sw(st_b1)
        pol_sw.train()
        pol_sw.zero_grad(set_to_none=True)
        (pol_sw(st_b2).sum() + pol_sw(st_custom).sum()).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in pol_sw.parameters())

    # [P2-3] AgentBatchState rejects requires_grad=True & collate rejects mismatched coordinate_system
    req_c4 = torch.randn((1, 1024, 12, 20), device=device, requires_grad=True)
    req_rejected = False
    try:
        _ = AgentBatchState(
            c4_pooled=req_c4,
            cue_maps=torch.rand((1, 5, 12, 20), device=device),
            style_stats=torch.randn((1, 2048), device=device),
        )
    except RuntimeError:
        req_rejected = True
    assert req_rejected, "AgentBatchState must reject requires_grad=True tensors!"

    st_c1 = _make_synthetic_state(1, device, seed=510)
    st_c2 = _make_synthetic_state(1, device, seed=511)
    st_c2.coordinate_system = "voc_1based_xyxy"
    coord_rejected = False
    try:
        _ = AgentBatchState.collate([st_c1, st_c2], device=device)
    except ValueError:
        coord_rejected = True
    assert coord_rejected, "AgentBatchState.collate must reject mismatched coordinate_system!"

    # [P1-02] Geometry/NMS mismatch rejected on raw load_state_dict; checkpoint dict round-trip exact
    custom_grid = CandidateGrid(grid_win_w=0.30, grid_win_h=0.40)
    pol_custom = ActiveZoomPolicyHead(grid=custom_grid, use_film=True, nms_window_iou=0.40).to(device).eval()
    pol_default = ActiveZoomPolicyHead(use_film=True).to(device).eval()

    geom_rejected = False
    try:
        pol_default.load_state_dict(pol_custom.state_dict(), strict=True)
    except RuntimeError:
        geom_rejected = True
    assert geom_rejected, "Raw load_state_dict must reject mismatched CandidateGrid geometry!"

    pol_nms45 = ActiveZoomPolicyHead(use_film=True, nms_window_iou=0.45).to(device).eval()
    nms_rejected = False
    try:
        pol_default.load_state_dict(pol_nms45.state_dict(), strict=True)
    except RuntimeError:
        nms_rejected = True
    assert nms_rejected, "Raw load_state_dict must reject mismatched nms_window_iou / step2_valid_base!"

    ckpt_dict = pol_custom.to_checkpoint_dict(extra={"stage": "phase1_test"})
    pol_rebuilt = ActiveZoomPolicyHead.from_checkpoint_dict(ckpt_dict, map_location=device).eval()
    st_ck = _make_synthetic_state(1, device, seed=520)
    assert pol_rebuilt.grid.config_dict() == pol_custom.grid.config_dict()
    assert torch.equal(pol_rebuilt._default_rois_b1, pol_custom._default_rois_b1)
    assert torch.allclose(pol_rebuilt(st_ck), pol_custom(st_ck), atol=1e-6)

    # [P2-2] k_max=0 preserves zero-grad autograd graph for log_prob / entropy / KL
    pol_k0 = ActiveZoomPolicyHead(use_film=True, k_max=0).to(device).train()
    ref_k0 = pol_k0.clone_as_reference().to(device)
    st_k0 = _make_synthetic_state(2, device, seed=530)
    lp0 = pol_k0.log_prob(st_k0, [(), ()], k_max=0)
    ent0 = pol_k0.entropy(st_k0, k_max=0)
    kl0 = pol_k0.kl_divergence(st_k0, ref_k0, k_max=0)
    assert float(lp0.abs().max().item()) == 0.0 and float(ent0.abs().max().item()) == 0.0 and float(kl0.abs().max().item()) == 0.0
    (lp0.sum() + ent0.sum() + kl0.sum()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in pol_k0.parameters())

    # [P1-03] OracleSelector.for_policy enforces policy nms_window_iou=0.35 (blocks (17, 15) IoU=0.404 conflict)
    pol_align = ActiveZoomPolicyHead(use_film=True, k_max=2, nms_window_iou=0.35).to(device).eval()
    orac_p0 = OracleSelector(grid=pol_align.grid, k_max=2)  # Phase-0 historical default 0.45
    orac_sft = OracleSelector.for_policy(pol_align)  # Phase-2 SFT aligned 0.35
    assert orac_p0.nms_window_iou == 0.45 and orac_sft.nms_window_iou == 0.35
    # Place two missed small weak-class GTs: one inside window 17 (0.46..0.92, 0.35..0.73) and one inside window 15 (0.68..1.0, 0.36..0.70)
    # where window 17 & 15 have IoU=0.404 (allowed by 0.45, forbidden by 0.35)
    dummy_perc = GlobalPerceptionOutput(
        orig_size=(1000, 1000),
        resized_size=(700, 700),
        canvas_size=(704, 704),
        detections={
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "scores": torch.zeros((0,), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.long),
        },
        proposals=torch.zeros((0, 4), dtype=torch.float32),
        proposal_probs=torch.zeros((0,), dtype=torch.float32),
        c4_feature=torch.zeros((1, 1024, 12, 20), dtype=torch.float32),
        rpn_prob_map=torch.full((1, 3, 12, 20), 0.1, dtype=torch.float32),
        channel_mu=torch.zeros((1, 1024), dtype=torch.float32),
        channel_log_sigma=torch.zeros((1, 1024), dtype=torch.float32),
    )
    gt_boxes_syn = torch.tensor(
        [
            [500.0, 450.0, 560.0, 510.0],  # inside win 17 (and win 14)
            [860.0, 460.0, 910.0, 510.0],  # inside win 17 & win 15
            [930.0, 460.0, 980.0, 510.0],  # inside win 15 only
        ],
        dtype=torch.float32,
    )
    gt_labels_syn = torch.tensor([2, 2, 2], dtype=torch.long)
    sel_sft, _ = orac_sft.select(dummy_perc, gt_boxes_syn, gt_labels_syn)
    lp_sft = pol_align.log_prob(_make_synthetic_state(1, device, seed=540), [tuple(sel_sft)])
    assert torch.isfinite(lp_sft).all(), f"OracleSelector.for_policy emitted illegal trajectory {sel_sft}"
    print(
        f"[PASS] 2d: Audit regressions verified (P1-01/P2-4 inference_mode->train on CPU & CUDA, "
        f"P1-02 checkpoint guard & rebuild, P1-03 Oracle.for_policy={sel_sft}, P2-2 k_max=0 grad, P2-3 state guards)."
    )


def test_3_probability_axioms(device: torch.device) -> Dict:
    print("\n" + "=" * 78)
    print("[TEST 3] Probability Axioms, Exact Enumeration, Analytical Entropy & KL")
    print("=" * 78)

    torch.manual_seed(1234)
    policy = ActiveZoomPolicyHead(use_film=True, k_max=2, nms_window_iou=0.35).to(device).eval()
    ref_policy = policy.clone_as_reference().to(device)
    # Perturb ref_policy weights deterministically
    with torch.no_grad():
        for p in ref_policy.parameters():
            p.add_(torch.randn_like(p) * 0.05)

    all_actions_k2 = policy.enumerate_valid_actions(k_max=2)
    all_actions_k1 = policy.enumerate_valid_actions(k_max=1)
    print(f"  Enumerated legal action space: |A_<=1| = {len(all_actions_k1)}, |A_<=2| = {len(all_actions_k2)}")
    assert len(all_actions_k1) == 19
    assert len(all_actions_k2) > 19

    max_prob_sum_err = 0.0
    max_ent_diff = 0.0
    max_kl_self = 0.0
    max_kl_ref_diff = 0.0
    min_kl_perturbed = float("inf")

    for seed in (1, 2, 3):
        st = _make_synthetic_state(batch_size=1, device=device, seed=300 + seed)
        # Scale logits across different temperatures to test both peaked and flat regimes
        for temp in (0.5, 1.0, 2.0):
            lp_all = policy.log_prob(st, all_actions_k2, k_max=2, temperature=temp)  # [M]
            probs_all = lp_all.exp()
            prob_sum = float(probs_all.sum().item())
            err = abs(prob_sum - 1.0)
            max_prob_sum_err = max(max_prob_sum_err, err)
            assert err < 1e-5, f"Sum of probs {prob_sum} != 1.0 (err={err})"

            # Analytical entropy vs brute-force enumeration
            ent_enum = float(-(probs_all * lp_all).sum().item())
            ent_analytic = float(policy.entropy(st, k_max=2, temperature=temp)[0].item())
            ent_diff = abs(ent_enum - ent_analytic)
            max_ent_diff = max(max_ent_diff, ent_diff)
            assert ent_diff < 1e-5, f"Entropy mismatch: enum={ent_enum}, analytic={ent_analytic}"

            # Self-KL == 0
            kl_self = float(policy.kl_divergence(st, policy, k_max=2, temperature=temp)[0].item())
            max_kl_self = max(max_kl_self, abs(kl_self))
            assert abs(kl_self) < 1e-6, f"Self KL should be ~0, got {kl_self}"

            # Perturbed ref KL > 0 and matches exact enumeration
            ref_lp_all = ref_policy.log_prob(st, all_actions_k2, k_max=2, temperature=temp)
            kl_enum = float((probs_all * (lp_all - ref_lp_all)).sum().item())
            kl_analytic = float(policy.kl_divergence(st, ref_policy, k_max=2, temperature=temp)[0].item())
            kl_diff = abs(kl_enum - kl_analytic)
            max_kl_ref_diff = max(max_kl_ref_diff, kl_diff)
            min_kl_perturbed = min(min_kl_perturbed, kl_analytic)
            assert kl_diff < 1e-5, f"KL mismatch: enum={kl_enum}, analytic={kl_analytic}"
            assert kl_analytic > 0.0, f"Perturbed KL must be > 0, got {kl_analytic}"

    # Verify custom valid_action_mask normalization & analytical consistency
    custom_mask = torch.ones((1, 19), dtype=torch.bool, device=device)
    custom_mask[0, [0, 1, 2, 12, 17]] = False
    st_masked = _make_synthetic_state(1, device, seed=399)
    st_masked.valid_action_mask = custom_mask
    masked_actions = policy.enumerate_valid_actions(k_max=2, valid_action_mask=custom_mask)
    lp_m = policy.log_prob(st_masked, masked_actions, k_max=2)
    err_m = abs(float(lp_m.exp().sum().item()) - 1.0)
    assert err_m < 1e-5, f"Masked action prob sum error={err_m}"
    ent_m_enum = float(-(lp_m.exp() * lp_m).sum().item())
    ent_m_ana = float(policy.entropy(st_masked, k_max=2)[0].item())
    assert abs(ent_m_enum - ent_m_ana) < 1e-5, f"Masked entropy diff {abs(ent_m_enum - ent_m_ana)}"

    # Verify illegal action rejection
    win_iou = policy.grid.window_iou_matrix()
    overlapping_pair = None
    for i in range(policy.num_windows):
        for j in range(policy.num_windows):
            if i != j and float(win_iou[i, j].item()) > policy.nms_window_iou:
                overlapping_pair = (i, j)
                break
        if overlapping_pair is not None:
            break
    assert overlapping_pair is not None

    st_1 = _make_synthetic_state(1, device, seed=350)
    for bad_act in [(0, 0), overlapping_pair, (0, 5, 11), (18,), (-1,)]:
        raised = False
        try:
            _ = policy.log_prob(st_1, [bad_act], k_max=2)
        except ValueError:
            raised = True
        assert raised, f"Illegal action {bad_act} should have raised ValueError!"

    # Verify zero-NaN gradients through entropy & KL with masked Step-2 log_softmax
    policy.train()
    st_b4 = _make_synthetic_state(4, device, seed=360)
    policy.zero_grad(set_to_none=True)
    loss_dist = policy.kl_divergence(st_b4, ref_policy).mean() - 0.1 * policy.entropy(st_b4).mean()
    loss_dist.backward()
    for name, p in policy.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), f"NaN/None grad in {name} through KL/entropy"
    policy.zero_grad(set_to_none=True)

    print(
        f"  max |sum pi(a) - 1.0|             = {max_prob_sum_err:.3e} (< 1e-5)\n"
        f"  max |H_analytic - H_enum|         = {max_ent_diff:.3e} (< 1e-5)\n"
        f"  max |D_KL(pi || pi)|              = {max_kl_self:.3e} (< 1e-6)\n"
        f"  max |KL_analytic - KL_enum|       = {max_kl_ref_diff:.3e} (< 1e-5)\n"
        f"  min D_KL(pi || pi_ref_perturbed)  = {min_kl_perturbed:.6f} (> 0)\n"
        f"  masked-action prob sum err        = {err_m:.3e} (< 1e-5)"
    )
    print("[PASS] TEST 3: All probability, entropy, KL, masking & gradient-safety axioms verified.")

    return {
        "num_actions_k2": len(all_actions_k2),
        "max_prob_sum_err": max_prob_sum_err,
        "max_ent_diff": max_ent_diff,
        "max_kl_self": max_kl_self,
        "max_kl_ref_diff": max_kl_ref_diff,
        "min_kl_perturbed": min_kl_perturbed,
    }


def _snapshot_bn_buffers(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    snap: Dict[str, torch.Tensor] = {}
    for name, buf in model.named_buffers():
        if any(k in name for k in ("running_mean", "running_var", "num_batches_tracked")):
            snap[name] = buf.detach().cpu().clone()
    return snap


def test_4_and_5_real_detector_grad_isolation_and_e2e(device: torch.device) -> Dict:
    print("\n" + "=" * 78)
    print("[TEST 4 & 5] Real Frozen StyleDetV6 Gradient Isolation, Dual-Path E2E & VRAM")
    print("=" * 78)

    rn_root = Path(r"E:\textstyleDet\textstyleDet_rn")
    sky_root = Path(DEFAULT_SKY_DIR)
    rn_snap_before = snapshot_readonly_tree(rn_root, (".py", ".pyc", ".pth", ".json"))
    sky_pkl_before = snapshot_readonly_tree(sky_root, (".pkl",))

    # Load one real daytime_clear image and one real daytime_foggy image
    clear_root, _, clear_ids, clear_recs, _ = load_domain_records(DEFAULT_SKY_DIR, "daytime_clear", split="test", max_images=1)
    foggy_root, _, foggy_ids, foggy_recs, _ = load_domain_records(DEFAULT_SKY_DIR, "daytime_foggy", split="test", max_images=1)
    img_clear_path = clear_root / "JPEGImages" / f"{clear_ids[0]}.jpg"
    img_foggy_path = foggy_root / "JPEGImages" / f"{foggy_ids[0]}.jpg"
    pil_clear = Image.open(img_clear_path).convert("RGB")
    pil_foggy = Image.open(img_foggy_path).convert("RGB")
    print(f"  Loaded real daytime_clear: {clear_ids[0]} {pil_clear.size}")
    print(f"  Loaded real daytime_foggy: {foggy_ids[0]} {pil_foggy.size}")

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    wrapper = FrozenTextStyleDetWrapper(
        checkpoint_path=DEFAULT_CKPT,
        clip_roi_weights_path=DEFAULT_CLIP_RN101,
        device=device,
        use_amp_bf16=False,
        roi_chunk_size=256,
        crop_max_proposals=300,
    )
    policy = ActiveZoomPolicyHead(use_film=True, k_max=2, nms_window_iou=0.35)
    agent = AZAgentDetector(detector_wrapper=wrapper, policy_head=policy)

    # Snapshot BN buffers BEFORE any train() call or forward passes
    bn_before = _snapshot_bn_buffers(agent.detector_wrapper.model)
    assert len(bn_before) > 0, "Expected BatchNorm buffers in StyleDetV6 backbone/ROI head"

    # --- TEST 4: Training mode isolation & gradient flow ---
    agent.train(True)
    agent.requires_grad_(True)

    assert agent.policy_head.training is True, "policy_head must enter train() mode"
    assert agent.detector_wrapper.model.training is False, "Frozen detector must NEVER enter train() mode"
    for mod_name, mod in agent.detector_wrapper.model.named_modules():
        assert mod.training is False, f"Frozen detector submodule {mod_name} entered train() mode!"
    for p_name, p in agent.detector_wrapper.model.named_parameters():
        assert p.requires_grad is False, f"Frozen detector param {p_name} has requires_grad=True!"
    assert len(list(agent.parameters())) == len(list(agent.policy_head.parameters())), (
        "agent.parameters() must contain ONLY policy_head parameters"
    )

    # Extract real batched state (B=2) from frozen detector
    percs, batch_state = agent.extract_batch_state(
        [pil_clear, pil_foggy],
        image_ids=[clear_ids[0], foggy_ids[0]],
    )
    assert batch_state.batch_size == 2
    assert batch_state.c4_pooled.shape == (2, 1024, 12, 20)
    assert batch_state.cue_maps.shape == (2, 5, 12, 20)
    assert batch_state.style_stats.shape == (2, 2048)
    assert not batch_state.c4_pooled.is_inference()
    assert not batch_state.cue_maps.is_inference()
    assert not batch_state.style_stats.is_inference()

    # Build a perturbed reference policy for simulated GRPO KL term
    ref_policy = agent.policy_head.clone_as_reference().to(device)
    with torch.no_grad():
        for p in ref_policy.parameters():
            p.add_(torch.randn_like(p) * 0.02)

    # 4a. Check PURE GRPO-style loss backward (no BCE)
    agent.zero_grad(set_to_none=True)
    logits = agent(batch_state)  # Dispatches to policy_head with gradients
    sampled = agent.policy_head.sample_group_from_logits(logits, G=4)
    adv = torch.tensor([[1.0, -0.5, 0.2, -0.7], [-0.8, 0.6, 0.5, -0.3]], device=device)
    kl_val = agent.policy_head.kl_divergence(batch_state, ref_policy)
    ent_val = agent.policy_head.entropy_from_logits(logits)
    grpo_loss = -(sampled.log_probs * adv).mean() + 0.1 * kl_val.mean() - 0.01 * ent_val.mean()
    grpo_loss.backward()

    zero_grad_params: List[str] = []
    for name, p in agent.policy_head.named_parameters():
        assert p.grad is not None, f"[Pure GRPO] Missing grad for {name}"
        assert torch.isfinite(p.grad).all(), f"[Pure GRPO] Non-finite grad for {name}"
        if float(p.grad.abs().sum().item()) == 0.0:
            zero_grad_params.append(name)
    assert not zero_grad_params, f"[Pure GRPO] Zero gradients on params: {zero_grad_params}"

    # 4b. Check SFT BCE loss backward
    agent.zero_grad(set_to_none=True)
    logits_sft = agent(batch_state)
    targets = torch.zeros_like(logits_sft)
    targets[0, 18] = 1.0  # clear -> No-Crop
    targets[1, 13] = 1.0  # foggy -> horizon window
    targets[1, 15] = 1.0
    sft_loss = F.binary_cross_entropy_with_logits(logits_sft, targets)
    sft_loss.backward()

    checked_groups = {
        "visual_proj": 0,
        "style_proj": 0,
        "conv_stem": 0,
        "film_head": 0,
        "roi_proj": 0,
        "global_proj": 0,
        "transformer": 0,
        "action_head": 0,
    }
    for name, p in agent.policy_head.named_parameters():
        assert p.grad is not None, f"[SFT] Missing grad for {name}"
        assert torch.isfinite(p.grad).all(), f"[SFT] Non-finite grad for {name}"
        assert float(p.grad.abs().sum().item()) > 0.0, f"[SFT] Zero grad for {name}"
        for g_key in checked_groups:
            if name.startswith(g_key):
                checked_groups[g_key] += 1
    assert all(v > 0 for v in checked_groups.values()), f"Missing expected parameter groups: {checked_groups}"

    # Verify frozen detector received ZERO gradients
    for p_name, p in agent.detector_wrapper.model.named_parameters():
        assert p.grad is None, f"Frozen StyleDetV6 param {p_name} unexpectedly received a gradient!"

    # Also verify w/o FiLM ablation backward pass
    policy_nofilm = ActiveZoomPolicyHead(use_film=False).to(device).train()
    loss_nf = F.binary_cross_entropy_with_logits(policy_nofilm(batch_state), targets)
    loss_nf.backward()
    for name, p in policy_nofilm.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and float(p.grad.abs().sum().item()) > 0.0, (
            f"[w/o FiLM] Bad grad for {name}"
        )

    del ref_policy, policy_nofilm, logits, logits_sft, grpo_loss, sft_loss, loss_nf
    agent.zero_grad(set_to_none=True)
    agent.eval()
    print(
        f"[PASS] TEST 4: All {len(list(agent.policy_head.parameters()))} trainable policy tensors received "
        "non-zero finite gradients (both pure GRPO & SFT BCE); frozen StyleDetV6 grad is None."
    )

    # --- TEST 5: Real-image Fast Path & Slow Path Integration + VRAM < 2.0 GiB ---
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    # Pick a guaranteed legal 2-crop action from enumerate_valid_actions
    legal_pairs = [a for a in agent.policy_head.enumerate_valid_actions(k_max=2) if len(a) == 2]
    test_pair = next((a for a in legal_pairs if a[0] >= 12 and a[1] >= 12), legal_pairs[0])

    # 5a. Fast Path (No-Crop) on daytime_clear
    out_fast = agent.predict(pil_clear, mode="greedy", forced_action=(), sync_cuda_for_timing=True)
    assert out_fast.used_slow_path is False
    assert out_fast.action == ()
    assert out_fast.selected_crop_boxes == []
    assert out_fast.timings_ms["local_crop_det_ms"] == 0.0
    assert out_fast.timings_ms["fusion_ms"] == 0.0
    assert torch.equal(out_fast.detections["boxes"], out_fast.global_detections["boxes"])
    assert torch.equal(out_fast.detections["scores"], out_fast.global_detections["scores"])
    assert torch.equal(out_fast.detections["labels"], out_fast.global_detections["labels"])
    assert 0.0 < out_fast.action_prob <= 1.0

    # 5b. Slow Path (K=1 and K=2 crops) on daytime_foggy
    out_slow1 = agent.predict(pil_foggy, mode="greedy", forced_action=(test_pair[0],), sync_cuda_for_timing=True)
    assert out_slow1.used_slow_path is True
    assert out_slow1.action == (test_pair[0],)
    assert len(out_slow1.selected_crop_boxes) == 1
    assert out_slow1.timings_ms["local_crop_det_ms"] > 0.0
    assert out_slow1.detections["boxes"].shape[0] <= agent.fuser.max_detections

    out_slow2 = agent.predict(pil_foggy, mode="greedy", forced_action=test_pair, sync_cuda_for_timing=True)
    assert out_slow2.used_slow_path is True
    assert out_slow2.action == test_pair
    assert len(out_slow2.selected_crop_boxes) == 2
    assert out_slow2.timings_ms["local_crop_det_ms"] > 0.0
    assert out_slow2.detections["boxes"].shape[0] <= agent.fuser.max_detections

    # Verify slow-path boxes are valid within image bounds AND bitwise equal to manual standalone Phase-0 path (P2-5)
    orig_h, orig_w = out_slow2.orig_size
    if out_slow2.detections["boxes"].numel() > 0:
        bxs = out_slow2.detections["boxes"]
        assert float(bxs[:, 0::2].min().item()) >= 0.0 and float(bxs[:, 0::2].max().item()) <= float(orig_w) + 1e-4
        assert float(bxs[:, 1::2].min().item()) >= 0.0 and float(bxs[:, 1::2].max().item()) <= float(orig_h) + 1e-4

    manual_perc = agent.detector_wrapper.forward_global(pil_foggy)
    manual_crops = [
        agent.detector_wrapper.forward_single_crop(pil_foggy, agent.grid.windows[w].to_pixel_xyxy(orig_w, orig_h))
        for w in test_pair
    ]
    manual_fused = agent.fuser.fuse(manual_perc.detections, manual_crops, manual_perc.orig_size)
    assert torch.equal(out_slow2.detections["boxes"], manual_fused["boxes"]), "Slow-path boxes != manual Phase-0 path"
    assert torch.equal(out_slow2.detections["scores"], manual_fused["scores"]), "Slow-path scores != manual Phase-0 path"
    assert torch.equal(out_slow2.detections["labels"], manual_fused["labels"]), "Slow-path labels != manual Phase-0 path"
    voc_agent = format_voc_prediction_lines(foggy_ids[0], out_slow2.detections, out_slow2.orig_size, wrapper.num_classes)
    voc_manual = format_voc_prediction_lines(foggy_ids[0], manual_fused, manual_perc.orig_size, wrapper.num_classes)
    assert voc_agent == voc_manual, "Slow-path VOC lines != standalone manual Phase-0 path!"

    # Also verify OracleSelector.for_policy(policy) on real images always yields legal policy trajectories (P1-03)
    orac_real = OracleSelector.for_policy(agent.policy_head)
    for img_id_r, rec_r, perc_r in (
        (clear_ids[0], clear_recs[clear_ids[0]], percs[0]),
        (foggy_ids[0], foggy_recs[foggy_ids[0]], percs[1]),
    ):
        gt_b, gt_l, gt_d = record_to_0based_gt(rec_r, wrapper.class_names)
        wids_r, _ = orac_real.select(perc_r, gt_b, gt_l, gt_d)
        st_r = agent.state_encoder.encode_state(perc_r, device=device)
        lp_r = agent.policy_head.log_prob(st_r, [tuple(wids_r)])
        assert torch.isfinite(lp_r).all(), f"Real-image Oracle.for_policy trajectory {wids_r} rejected on {img_id_r}"

    # 5c. Natural greedy & stochastic sample modes (no forced action)
    out_auto_clear = agent.predict(pil_clear, mode="greedy", sync_cuda_for_timing=True)
    out_samp_foggy = agent.predict(pil_foggy, mode="sample", sync_cuda_for_timing=True)
    assert len(out_auto_clear.action) <= 2 and len(out_samp_foggy.action) <= 2

    # Check BN stats bitwise identical after ALL inference + training backward runs
    bn_after = _snapshot_bn_buffers(agent.detector_wrapper.model)
    assert bn_before.keys() == bn_after.keys()
    for b_name in bn_before:
        assert torch.equal(bn_before[b_name], bn_after[b_name]), f"Frozen BN buffer mutated: {b_name}"

    peak_alloc_gib = (
        torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == "cuda" else 0.0
    )
    peak_res_gib = (
        torch.cuda.max_memory_reserved(device) / (1024**3) if device.type == "cuda" else 0.0
    )

    print(
        f"\n  [Fast Path  - daytime_clear] action={out_fast.action}, boxes={out_fast.detections['boxes'].shape[0]}, "
        f"prob={out_fast.action_prob:.4f}, timings={out_fast.timings_ms}"
    )
    print(
        f"  [Slow Path1 - daytime_foggy] action={out_slow1.action}, boxes={out_slow1.detections['boxes'].shape[0]}, "
        f"prob={out_slow1.action_prob:.4f}, timings={out_slow1.timings_ms}"
    )
    print(
        f"  [Slow Path2 - daytime_foggy] action={out_slow2.action}, boxes={out_slow2.detections['boxes'].shape[0]}, "
        f"prob={out_slow2.action_prob:.4f}, timings={out_slow2.timings_ms}"
    )
    print(
        f"  [Natural Greedy / Sample]    clear_act={out_auto_clear.action}, foggy_sample={out_samp_foggy.action}"
    )
    print(
        f"  [GPU Peak Memory (E2E)]      allocated={peak_alloc_gib:.3f} GiB (< 2.0 GiB), "
        f"reserved={peak_res_gib:.3f} GiB"
    )
    assert peak_alloc_gib < 2.0, f"Peak allocated VRAM {peak_alloc_gib:.3f} GiB >= 2.0 GiB"

    # Verify read-only directories 100% untouched
    rn_snap_after = snapshot_readonly_tree(rn_root, (".py", ".pyc", ".pth", ".json"))
    sky_pkl_after = snapshot_readonly_tree(sky_root, (".pkl",))
    rn_diff = [k for k in set(rn_snap_before) | set(rn_snap_after) if rn_snap_before.get(k) != rn_snap_after.get(k)]
    sky_diff = [k for k in set(sky_pkl_before) | set(sky_pkl_after) if sky_pkl_before.get(k) != sky_pkl_after.get(k)]
    assert not rn_diff, f"Read-only upstream textstyleDet_rn mutated: {rn_diff}"
    assert not sky_diff, f"Read-only dataset sky mutated: {sky_diff}"

    print(
        f"[PASS] TEST 5: Dual-path real-image E2E verified, {len(bn_before)} BN buffers bitwise unchanged, "
        f"peak VRAM {peak_alloc_gib:.3f} GiB < 2.0 GiB, zero read-only mutations."
    )
    return {
        "bn_buffers_checked": len(bn_before),
        "peak_allocated_gib": round(peak_alloc_gib, 3),
        "peak_reserved_gib": round(peak_res_gib, 3),
        "fast_timings_ms": out_fast.timings_ms,
        "slow1_timings_ms": out_slow1.timings_ms,
        "slow2_timings_ms": out_slow2.timings_ms,
    }


def main() -> None:
    assert sys.dont_write_bytecode is True, "sys.dont_write_bytecode must be True"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Phase 1 Suite] Running on device={device} | torch={torch.__version__} | cuda={torch.version.cuda}")

    t1 = test_1_params_and_latency(device)
    test_2_ablations_batch_roi_and_grad_bridge(device)
    t3 = test_3_probability_axioms(device)
    t45 = test_4_and_5_real_detector_grad_isolation_and_e2e(device)

    print("\n" + "=" * 78)
    print("ALL PHASE 1 UNIT & REAL-IMAGE INTEGRATION TESTS PASSED!")
    print("=" * 78)
    print(
        f"  Total Policy Params (w/ FiLM) : {t1['breakdown_film']['total']:,} ({t1['breakdown_film']['total'] / 1e6:.3f}M)\n"
        f"  Eager FP32 [default] (B=1)    : mean={t1['eager_mean_ms']:.3f} ms, p50={t1['eager_p50_ms']:.3f} ms, p95={t1['eager_p95_ms']:.3f} ms\n"
        f"  CUDA-Graph FP32 [opt-in] (B=1): mean={t1['fwd_mean_ms']:.3f} ms, p50={t1['fwd_p50_ms']:.3f} ms, p95={t1['fwd_p95_ms']:.3f} ms\n"
        f"  Prob Sum Max Error (|A|={t3['num_actions_k2']}) : {t3['max_prob_sum_err']:.3e}\n"
        f"  Self-KL Max / Perturbed Min   : {t3['max_kl_self']:.3e} / {t3['min_kl_perturbed']:.6f}\n"
        f"  E2E Peak VRAM Allocated       : {t45['peak_allocated_gib']:.3f} GiB (reserved {t45['peak_reserved_gib']:.3f} GiB)\n"
        f"  Frozen BN Buffers Unchanged   : {t45['bn_buffers_checked']} / {t45['bn_buffers_checked']}"
    )


if __name__ == "__main__":
    main()
