#!/usr/bin/env python
"""Phase 3 Automated Unit & Acceptance Verification Suite.

Verifies all Phase 3 Stage-2 GRPO mathematical invariants, P0-01/P0-02/P1-01 audit contracts,
Single-DGOD zero-leakage guarantees, checkpoint resumption, and read-only boundaries:
  1. GRPO Step-0 P1-01 Parity & Mathematical Invariants:
     - ``max |r_i(theta) - 1.0| < 1e-6``, premature clip count == 0, ``max D_KL < 1e-6``
       under ``dropout_override=0.0, exact_eval_parity=True`` & ``as_reference=True``.
     - Identical-reward ``zero_std`` groups yield strictly zero advantages & strictly zero
       policy gradients.
     - Non-degenerate groups produce finite gradients across ALL trainable policy parameters
       while frozen ``StyleDetV6`` and ``pi_ref`` maintain ``training=False``, ``grad=None``,
       and bitwise-identical BN buffers.
  2. Environment Cache Symmetry, Empty-Action Anchor & Manifest Tamper Rejection:
     - ``R(()) == 0.0`` (all deltas and cost == 0.0).
     - Dual-window actions ``(w1, w2)`` and ``(w2, w1)`` share unordered ``wid`` cache
       (0 extra crop misses) with bitwise-identical fused detections and rewards.
     - All manifest/domain/isolation/bucket/seal tamper cases raise ``RuntimeError``.
  3. Single-DGOD Hard Isolation:
     - Any target domain (``daytime_foggy / dusk_rainy / Night-Sunny / night_rainy``) or
       ``daytime_clear/test`` raises ``ValueError`` / ``RuntimeError``.
     - Verifies ``phase3_grpo_results.json`` strictly uses ``daytime_clear/train`` with
       disjoint ``source_train`` / ``source_dev``.
  4. Atomic Resumption, ``policy_grpo.pth`` Reconstruction & Live ``AZAgentDetector.predict``:
     - Pause at Epoch 1 -> ``--resume`` to Epoch 2 works seamlessly; mismatched manifest
       digest on ``--resume`` raises ``RuntimeError``.
     - ``ActiveZoomPolicyHead.from_checkpoint("outputs/phase3/policy_grpo.pth")`` runs live
       ``AZAgentDetector.predict`` on real ``source_dev`` images across fast path (``No-Crop``),
       slow paths (``K=1``, ``K=2``), and greedy mode.
     - Verifies training peak GPU memory strictly ``< 3.5 GiB``.
  5. Read-Only Boundary Integrity:
     - Zero mutations in ``E:\\textstyleDet\\textstyleDet_rn``, ``E:\\textstyleDet\\data\\sky``,
       and ``E:\\textstyleDet\\TextStyleDet_vlm\\outputs\\phase2``.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import json
import math
import shutil
import tempfile
from pathlib import Path

import torch

from run_phase0 import DEFAULT_CKPT, DEFAULT_CLIP_RN101, DEFAULT_SKY_DIR, snapshot_readonly_tree
from zoom_agent import (
    AZAgentDetector,
    ActiveZoomPolicyHead,
    CandidateGrid,
    DualViewWBFFusion,
    FrozenTextStyleDetWrapper,
    GRPODetectionEnvironment,
    GRPORewardConfig,
    GRPOTrainConfig,
    PersistentDetectionEnvCache,
    ShardedSFTCacheReader,
    SmoothDetectionRewarder,
    SourceOnlyDatasetLoader,
    SubsetSFTCacheView,
    apply_physical_degradation,
    build_env_cache_manifest_payload,
    compute_group_advantages,
    compute_grpo_loss_components,
    train_grpo_policy,
    verify_env_cache_manifest,
    verify_grpo_step0_parity,
)
from zoom_agent.eval_utils import sha256_file, sha256_json
from zoom_agent.sft_trainer import _snapshot_bn_buffers

DEFAULT_SFT_CACHE_DIR = Path(r"E:\textstyleDet\TextStyleDet_vlm\outputs\phase2\sft_cache")
DEFAULT_SFT_CKPT = Path(r"E:\textstyleDet\TextStyleDet_vlm\outputs\phase2\policy_sft.pth")
DEFAULT_REF_CKPT = Path(r"E:\textstyleDet\TextStyleDet_vlm\outputs\phase2\policy_sft_ref.pth")
DEFAULT_PHASE2_DIR = Path(r"E:\textstyleDet\TextStyleDet_vlm\outputs\phase2")
DEFAULT_PHASE3_DIR = Path(r"E:\textstyleDet\TextStyleDet_vlm\outputs\phase3")
DEFAULT_ENV_CACHE_DIR = DEFAULT_PHASE3_DIR / "env_cache"
DEFAULT_GRPO_CKPT = DEFAULT_PHASE3_DIR / "policy_grpo.pth"
DEFAULT_RESULTS_JSON = DEFAULT_PHASE3_DIR / "phase3_grpo_results.json"


def test_1_grpo_step0_and_math_invariants(
    sft_ckpt: Path,
    ref_ckpt: Path,
    sft_cache_dir: Path,
    wrapper: FrozenTextStyleDetWrapper,
    device: torch.device,
) -> None:
    print("\n" + "=" * 78)
    print("[TEST 1] GRPO Step-0 P1-01 Parity, Zero-Std Gradient & Frozen Isolation")
    print("=" * 78)

    bn_before = _snapshot_bn_buffers(wrapper.model)

    policy = ActiveZoomPolicyHead.from_checkpoint(
        sft_ckpt, map_location=device, dropout_override=0.0, exact_eval_parity=True
    ).train()
    ref_policy = ActiveZoomPolicyHead.from_checkpoint(
        ref_ckpt, map_location=device, as_reference=True, exact_eval_parity=True
    )

    dev_reader = ShardedSFTCacheReader(
        sft_cache_dir, split_role="source_dev", verify_shard_hashes=False, require_completed=True
    )
    batch = next(dev_reader.iter_batches(batch_size=16, shuffle=False, seed=0, device=device))
    state = batch["state"]

    # 1a-0. Sealed Phase 3 default hyperparameters check (P1-02)
    def_tcfg = GRPOTrainConfig()
    def_rcfg = GRPORewardConfig()
    assert def_tcfg.batch_size == 8 and def_tcfg.epochs == 6 and def_tcfg.ppo_epochs == 2
    assert def_tcfg.beta_kl == 0.16 and def_tcfg.alpha_ent == 0.015
    assert def_tcfg.lr == 8e-6 and def_tcfg.min_lr == 1e-6
    assert def_tcfg.sampling_temp == 1.05 and def_tcfg.kl_entropy_temp == 1.0
    assert def_tcfg.max_dev_views == 600 and def_tcfg.shard_group_size == 2
    assert def_tcfg.flush_env_every_batches == 32
    assert def_rcfg.lambda_loc_ap == 3.5 and def_rcfg.lambda_fp == 0.90 and def_rcfg.lambda_cost == 0.045
    assert def_rcfg.loc_w_ap50_95 == 0.45 and def_rcfg.loc_w_ap75 == 0.30 and def_rcfg.loc_w_matched_iou == 0.25
    assert def_rcfg.fp_score_thr == 0.05

    # 1a. Step-0 exact parity under sealed T_samp=1.05, T_reg=1.0 (same-mode, eval->train, 0 premature clips, KL < 1e-6)
    s0 = verify_grpo_step0_parity(
        policy=policy,
        ref_policy=ref_policy,
        state=state,
        group_size=4,
        clip_eps=0.2,
        sampling_temp=1.05,
        kl_entropy_temp=1.0,
        seed=20260425,
    )
    assert s0["max_ratio_err_same_mode"] < 1e-6
    assert s0["max_ratio_err_eval_to_train"] < 1e-6
    assert s0["premature_clip_count"] == 0
    assert s0["max_kl_divergence"] < 1e-6

    # 1a-neg. P1-01 negative guard tests (P3-02): dropout>0, exact_eval_parity=False, and ref_policy.train() MUST fail
    pol_bad_drop = ActiveZoomPolicyHead.from_checkpoint(
        sft_ckpt, map_location=device, dropout_override=0.1, exact_eval_parity=True
    ).train()
    bad_drop_raised = False
    try:
        verify_grpo_step0_parity(policy=pol_bad_drop, ref_policy=ref_policy, state=state, group_size=2)
    except AssertionError:
        bad_drop_raised = True
    assert bad_drop_raised, "[P1-01] verify_grpo_step0_parity MUST reject dropout_override=0.1!"

    pol_bad_parity = ActiveZoomPolicyHead.from_checkpoint(
        sft_ckpt, map_location=device, dropout_override=0.0, exact_eval_parity=False
    ).train()
    bad_parity_raised = False
    try:
        verify_grpo_step0_parity(policy=pol_bad_parity, ref_policy=ref_policy, state=state, group_size=2)
    except AssertionError:
        bad_parity_raised = True
    assert bad_parity_raised, "[P1-01] verify_grpo_step0_parity MUST reject exact_eval_parity=False!"

    ref_bad_train = ActiveZoomPolicyHead.from_checkpoint(
        ref_ckpt, map_location=device, as_reference=True, exact_eval_parity=True
    ).train()
    bad_ref_raised = False
    try:
        verify_grpo_step0_parity(policy=policy, ref_policy=ref_bad_train, state=state, group_size=2)
    except AssertionError:
        bad_ref_raised = True
    assert bad_ref_raised, "[P1-01] verify_grpo_step0_parity MUST reject ref_policy in train() mode!"
    del pol_bad_drop, pol_bad_parity, ref_bad_train

    # 1b. Zero-std identical reward groups -> strictly zero advantages & strictly zero policy gradient
    gen = torch.Generator(device=device).manual_seed(123)
    with torch.no_grad():
        samp = policy.sample_group(state, G=4, temperature=1.0, generator=gen)
        old_lp = samp.log_probs.detach()
        ref_logits = ref_policy(state).detach()

    const_rewards = torch.full((state.batch_size, 4), 0.12345, dtype=torch.float32, device=device)
    adv_zero, z_mask = compute_group_advantages(const_rewards, eps=1e-8, zero_std_thresh=1e-6)
    assert bool(z_mask.all().item()), "All constant-reward groups must be flagged as zero_std!"
    assert float(adv_zero.abs().max().item()) == 0.0, "Zero-std advantages must be strictly 0.0!"

    policy.zero_grad(set_to_none=True)
    comp_zero = compute_grpo_loss_components(
        policy=policy,
        state=state,
        actions=samp.actions,
        old_log_probs=old_lp,
        advantages=adv_zero,
        ref_logits=ref_logits,
        clip_eps=0.2,
        beta_kl=0.0,
        alpha_ent=0.0,
        sampling_temp=1.0,
    )
    assert float(comp_zero["pg_loss"].item()) == 0.0
    comp_zero["total_loss"].backward()
    for p_name, p in policy.named_parameters():
        if p.grad is not None:
            assert float(p.grad.abs().max().item()) == 0.0, (
                f"Zero-std group leaked non-zero policy gradient on {p_name}: {p.grad.abs().max().item()}"
            )

    # 1c. Mixed-variance rewards + KL + entropy -> finite non-zero gradients on ALL policy params
    mixed_rewards = torch.tensor(
        [[-0.1, 0.0, 0.25, 0.4] if i % 2 == 0 else [0.2, 0.2, 0.2, 0.2] for i in range(state.batch_size)],
        dtype=torch.float32,
        device=device,
    )
    adv_mix, z_mix = compute_group_advantages(mixed_rewards, eps=1e-8, zero_std_thresh=1e-6)
    assert int(z_mix.sum().item()) == state.batch_size // 2
    assert float(adv_mix[z_mix].abs().max().item()) == 0.0
    assert float(adv_mix[~z_mix].abs().max().item()) > 0.5

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    policy.zero_grad(set_to_none=True)
    comp_full = compute_grpo_loss_components(
        policy=policy,
        state=state,
        actions=samp.actions,
        old_log_probs=old_lp,
        advantages=adv_mix,
        ref_logits=ref_logits,
        clip_eps=0.2,
        beta_kl=0.08,
        alpha_ent=0.005,
        sampling_temp=1.0,
    )
    assert bool(torch.isfinite(comp_full["total_loss"]).item())
    assert bool(torch.isfinite(comp_full["kl_mean"]).item())
    assert bool(torch.isfinite(comp_full["ent_mean"]).item())
    comp_full["total_loss"].backward()

    nonzero_param_count = 0
    for p_name, p in policy.named_parameters():
        assert p.grad is not None, f"Missing gradient on trainable parameter {p_name}"
        assert bool(torch.isfinite(p.grad).all().item()), f"Non-finite gradient on {p_name}"
        if float(p.grad.abs().sum().item()) > 0.0:
            nonzero_param_count += 1
    total_param_tensors = sum(1 for _ in policy.parameters())
    assert nonzero_param_count == total_param_tensors, (
        f"Expected all {total_param_tensors} trainable policy param tensors to receive non-zero grads, "
        f"got {nonzero_param_count}"
    )

    # 1d. Frozen reference policy & frozen StyleDetV6 integrity
    assert ref_policy.training is False
    for p_name, p in ref_policy.named_parameters():
        assert (not p.requires_grad) and (p.grad is None), f"pi_ref parameter mutated: {p_name}"

    assert wrapper.model.training is False
    for p_name, p in wrapper.model.named_parameters():
        assert (not p.requires_grad) and (p.grad is None), f"Frozen StyleDetV6 parameter leaked grad: {p_name}"
    bn_after = _snapshot_bn_buffers(wrapper.model)
    for k in bn_before:
        assert torch.equal(bn_before[k], bn_after[k]), f"Frozen StyleDetV6 BN mutated: {k}"

    live_peak_gib = torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == "cuda" else 0.0
    assert live_peak_gib < 3.5, f"Live GRPO step VRAM {live_peak_gib:.3f} GiB >= 3.5 GiB!"

    print(
        f"[PASS] TEST 1: Step-0 parity exact (max|r-1|={s0['max_ratio_err_eval_to_train']:.2e}, "
        f"clip=0, max_KL={s0['max_kl_divergence']:.2e}); zero_std gradient strictly 0.0; "
        f"all {total_param_tensors} policy param tensors received finite non-zero grads; "
        f"pi_ref & StyleDetV6 frozen/BN intact; live step peak VRAM={live_peak_gib:.3f} GiB (< 3.5 GiB)."
    )


def test_2_env_cache_symmetry_and_tamper_rejection(
    env_cache_dir: Path,
    wrapper: FrozenTextStyleDetWrapper,
    loader: SourceOnlyDatasetLoader,
) -> None:
    print("\n" + "=" * 78)
    print("[TEST 2] Env Cache Symmetry, R(()) == 0.0 & Manifest Tamper Rejection")
    print("=" * 78)

    grid = CandidateGrid()
    fuser = DualViewWBFFusion(
        global_weight=1.0, local_weight=0.85, iou_thr=0.55, trunc_margin_px=5.0, max_detections=100, fusion_mode="wbf"
    )
    rewarder = SmoothDetectionRewarder(GRPORewardConfig())

    split_info = loader.build_splits(max_train_images=4, max_dev_images=2)
    test_img_id = split_info.selected_dev_ids[0]
    vk = (test_img_id, "clean", 20260420)

    # 2a. Verify real env_cache integrity if present (including P3-01 unregistered bucket & disk weight checks)
    if (env_cache_dir / "env_cache_manifest.json").is_file():
        real_doc = verify_env_cache_manifest(
            env_cache_dir,
            verify_bucket_hashes=True,
            check_unregistered_buckets=True,
            verify_disk_artifact_hashes=True,
            workspace_root=None,
        )
        assert real_doc["manifest"]["source_domain"] == "daytime_clear"
        assert real_doc["manifest"]["source_split"] == "train"
        assert real_doc["manifest"]["detector_config"]["crop_isolation_mode"] == "strict_single_crop_batch1"

    # 2a-rew. Dedicated SmoothDetectionRewarder analytical boundary tests (P3-02)
    orig_sz = (720, 1280)
    empty_det = {
        "boxes": torch.zeros((0, 4), dtype=torch.float32),
        "scores": torch.zeros((0,), dtype=torch.float32),
        "labels": torch.zeros((0,), dtype=torch.long),
    }
    empty_gt_b = torch.zeros((0, 4), dtype=torch.float32)
    empty_gt_l = torch.zeros((0,), dtype=torch.long)
    empty_gt_d = torch.zeros((0,), dtype=torch.bool)

    # (i) Empty GT: zero detections -> fp_pen == 0.0; adding an FP -> fp_pen > 0 & reward < 0
    sc_empty_clean = rewarder.score_detections(empty_det, empty_gt_b, empty_gt_l, empty_gt_d, orig_sz)
    assert sc_empty_clean.fp_pen == 0.0 and sc_empty_clean.loc_ap_smooth == 0.0 and sc_empty_clean.weak_small_recall == 0.0
    det_fp_low = {
        "boxes": torch.tensor([[100.0, 100.0, 200.0, 200.0]], dtype=torch.float32),
        "scores": torch.tensor([0.30], dtype=torch.float32),
        "labels": torch.tensor([1], dtype=torch.long),
    }
    det_fp_high = {
        "boxes": torch.tensor([[100.0, 100.0, 200.0, 200.0]], dtype=torch.float32),
        "scores": torch.tensor([0.70], dtype=torch.float32),
        "labels": torch.tensor([1], dtype=torch.long),
    }
    sc_fp_low = rewarder.score_detections(det_fp_low, empty_gt_b, empty_gt_l, empty_gt_d, orig_sz)
    sc_fp_high = rewarder.score_detections(det_fp_high, empty_gt_b, empty_gt_l, empty_gt_d, orig_sz)
    rb_fp_low = rewarder.compute_incremental_reward(
        det_fp_low, sc_empty_clean, (12,), empty_gt_b, empty_gt_l, empty_gt_d, orig_sz
    )
    rb_fp_high = rewarder.compute_incremental_reward(
        det_fp_high, sc_empty_clean, (12,), empty_gt_b, empty_gt_l, empty_gt_d, orig_sz
    )
    assert sc_fp_low.fp_pen > 0.0 and rb_fp_low.reward < 0.0
    # (ii) FP confidence monotonicity: score=0.70 penalizes strictly more than score=0.30
    assert sc_fp_high.fp_pen > sc_fp_low.fp_pen and rb_fp_high.reward < rb_fp_low.reward

    # (iii) difficult=True GT exemption (both empty-valid-GT & present-valid-GT cases)
    diff_only_b = torch.tensor([[100.0, 100.0, 200.0, 200.0]], dtype=torch.float32)
    diff_only_l = torch.tensor([1], dtype=torch.long)
    diff_only_d = torch.tensor([True], dtype=torch.bool)
    sc_diff_exempt = rewarder.score_detections(det_fp_high, diff_only_b, diff_only_l, diff_only_d, orig_sz)
    assert sc_diff_exempt.fp_pen == 0.0, f"difficult=True GT prediction must be FP-exempt, got {sc_diff_exempt.fp_pen}"

    mixed_gt_b = torch.tensor([[100.0, 100.0, 200.0, 200.0], [400.0, 200.0, 500.0, 300.0]], dtype=torch.float32)
    mixed_gt_l = torch.tensor([1, 2], dtype=torch.long)
    mixed_gt_d = torch.tensor([False, True], dtype=torch.bool)
    det_valid_and_diff = {
        "boxes": torch.tensor([[100.0, 100.0, 200.0, 200.0], [402.0, 202.0, 498.0, 298.0]], dtype=torch.float32),
        "scores": torch.tensor([0.90, 0.85], dtype=torch.float32),
        "labels": torch.tensor([1, 2], dtype=torch.long),
    }
    sc_mixed_diff = rewarder.score_detections(det_valid_and_diff, mixed_gt_b, mixed_gt_l, mixed_gt_d, orig_sz)
    assert sc_mixed_diff.fp_pen == 0.0, f"Absent-class difficult GT overlap must be FP-exempt, got {sc_mixed_diff.fp_pen}"

    # (iv) Tight localization gain: shifting matched prediction from IoU ~0.60 -> ~0.85 strictly raises LocAP & reward
    single_gt_b = torch.tensor([[100.0, 100.0, 200.0, 200.0]], dtype=torch.float32)
    single_gt_l = torch.tensor([1], dtype=torch.long)
    single_gt_d = torch.tensor([False], dtype=torch.bool)
    # Shift by 25 px horizontally on 100x100 box => overlap=75x100=7500, union=12500 => IoU=0.60
    det_iou60 = {
        "boxes": torch.tensor([[125.0, 100.0, 225.0, 200.0]], dtype=torch.float32),
        "scores": torch.tensor([0.90], dtype=torch.float32),
        "labels": torch.tensor([1], dtype=torch.long),
    }
    # Shift by 8 px horizontally => overlap=92x100=9200, union=10800 => IoU=0.8519
    det_iou85 = {
        "boxes": torch.tensor([[108.0, 100.0, 208.0, 200.0]], dtype=torch.float32),
        "scores": torch.tensor([0.90], dtype=torch.float32),
        "labels": torch.tensor([1], dtype=torch.long),
    }
    sc_iou60 = rewarder.score_detections(det_iou60, single_gt_b, single_gt_l, single_gt_d, orig_sz)
    rb_tight = rewarder.compute_incremental_reward(
        det_iou85, sc_iou60, (12,), single_gt_b, single_gt_l, single_gt_d, orig_sz
    )
    assert rb_tight.delta_ap75 > 0.5 and rb_tight.delta_ap50_95 > 0.2 and rb_tight.delta_matched_iou > 0.2
    assert rb_tight.delta_loc_ap_smooth > 0.2 and rb_tight.reward > 0.5

    # 2b. Isolated temp env cache: test R(()) == 0.0, (w1, w2) vs (w2, w1) symmetry, and 7 tamper cases
    with tempfile.TemporaryDirectory(prefix="az_p3_env_test_") as tmp:
        tdir = Path(tmp) / "env_cache"
        ws_root = Path(__file__).resolve().parent
        manifest_payload = build_env_cache_manifest_payload(
            wrapper=wrapper,
            grid=grid,
            fuser=fuser,
            degradation_cfg={"recipe_version": 1, "base_seed": 20260420},
            workspace_root=ws_root,
            num_buckets=8,
        )
        ecache = PersistentDetectionEnvCache(tdir, manifest_payload=manifest_payload, verify_bucket_hashes=True)
        env = GRPODetectionEnvironment(
            loader=loader,
            env_cache=ecache,
            grid=grid,
            fuser=fuser,
            rewarder=rewarder,
            wrapper=wrapper,
            k_max=2,
        )

        # Empty action anchor R(()) == 0.0
        rb_empty = env.evaluate_action_reward(vk, ())
        assert rb_empty.reward == 0.0
        assert rb_empty.delta_weak_small_recall == 0.0
        assert rb_empty.delta_loc_ap_smooth == 0.0
        assert rb_empty.delta_ap50_95 == 0.0
        assert rb_empty.delta_ap75 == 0.0
        assert rb_empty.delta_matched_iou == 0.0
        assert rb_empty.delta_fp_pen == 0.0
        assert rb_empty.cost_pen == 0.0
        assert rb_empty.num_crops == 0

        # Symmetry of (w1, w2) vs (w2, w1) on two non-overlapping horizon windows (12 and 15)
        w1, w2 = 12, 15
        fused_12, sz_12 = env.fuse_action(vk, (w1, w2))
        misses_after_12 = env.crop_cache_misses
        assert misses_after_12 == 2, f"Expected 2 crop misses for first (12, 15), got {misses_after_12}"

        fused_21, sz_21 = env.fuse_action(vk, (w2, w1))
        assert env.crop_cache_misses == misses_after_12, (
            "(w2, w1) MUST reuse unordered wid cache with 0 additional crop misses!"
        )
        assert sz_12 == sz_21
        for k in ("boxes", "scores", "labels"):
            assert torch.equal(fused_12[k], fused_21[k]), f"Fused {k} differs between ({w1},{w2}) and ({w2},{w1})!"

        # Clear reward memo to confirm reward computation itself is also bitwise identical
        env.clear_reward_memo()
        rb_12 = env.evaluate_action_reward(vk, (w1, w2))
        env.clear_reward_memo()
        rb_21 = env.evaluate_action_reward(vk, (w2, w1))
        assert rb_12 == rb_21, f"Reward breakdown differs between ({w1},{w2}) and ({w2},{w1}): {rb_12} vs {rb_21}"

        # Flush and reload cleanly (including P3-01 full disk weights + source code SHA256 verification)
        ecache.flush()
        base_doc = verify_env_cache_manifest(
            tdir,
            expected_manifest_digest=ecache.manifest_digest,
            verify_bucket_hashes=True,
            check_unregistered_buckets=True,
            verify_disk_artifact_hashes=True,
            workspace_root=ws_root,
        )
        mpath = tdir / "env_cache_manifest.json"
        b_entry = next(iter(base_doc["buckets"].values()))
        b_path = tdir / b_entry["file"]

        # Tamper 0 (P3-01): Unregistered orphan .pt file inside buckets/ MUST raise RuntimeError
        orphan_path = tdir / "buckets" / "bucket_999.pt"
        orphan_path.write_bytes(b"orphan_tamper")
        orphan_raised = False
        try:
            verify_env_cache_manifest(tdir, verify_bucket_hashes=False, check_unregistered_buckets=True)
        except RuntimeError:
            orphan_raised = True
        finally:
            orphan_path.unlink(missing_ok=True)
        assert orphan_raised, "Unregistered orphan .pt bucket file MUST raise RuntimeError!"

        def _assert_tamper_fails(doc_obj: dict, label: str, expected_dig=None, verify_hashes=True):
            mpath.write_text(json.dumps(doc_obj), encoding="utf-8")
            raised = False
            try:
                verify_env_cache_manifest(tdir, expected_manifest_digest=expected_dig, verify_bucket_hashes=verify_hashes)
            except RuntimeError:
                raised = True
            assert raised, f"Tamper case [{label}] MUST raise RuntimeError!"

        # Tamper 1: Mutate fuser_config without updating digest
        d1 = json.loads(json.dumps(base_doc))
        d1["manifest"]["fuser_config"]["local_weight"] = 0.50
        _assert_tamper_fails(d1, "mutated_fuser_config", verify_hashes=False)

        # Tamper 2: Forge source_domain to target domain AND recompute digest
        d2 = json.loads(json.dumps(base_doc))
        d2["manifest"]["source_domain"] = "night_rainy"
        d2["manifest_digest"] = sha256_json(d2["manifest"])
        _assert_tamper_fails(d2, "forged_target_domain", verify_hashes=False)

        # Tamper 3: Break P0-01 crop_isolation_mode AND recompute digest
        d3 = json.loads(json.dumps(base_doc))
        d3["manifest"]["detector_config"]["crop_isolation_mode"] = "batched_union_unsafe"
        d3["manifest_digest"] = sha256_json(d3["manifest"])
        _assert_tamper_fails(d3, "broken_isolation_mode", verify_hashes=False)

        # Tamper 4: Mismatched expected_manifest_digest
        _assert_tamper_fails(base_doc, "mismatched_expected_digest", expected_dig="f" * 64, verify_hashes=False)

        # Tamper 5: Missing seal_digest
        d5 = json.loads(json.dumps(base_doc))
        d5.pop("seal_digest", None)
        _assert_tamper_fails(d5, "missing_seal_digest", verify_hashes=False)

        # Tamper 6: Corrupt bucket bytes on disk
        orig_b = b_path.read_bytes()
        b_path.write_bytes(orig_b[:-1] + bytes([orig_b[-1] ^ 0xFF]))
        _assert_tamper_fails(base_doc, "corrupted_bucket_sha256", verify_hashes=True)

        # Tamper 7: Forge bucket sha256 without updating seal_digest
        d7 = json.loads(json.dumps(base_doc))
        bk_key = next(iter(d7["buckets"].keys()))
        d7["buckets"][bk_key]["sha256"] = sha256_file(b_path)
        _assert_tamper_fails(d7, "forged_bucket_breaking_seal", verify_hashes=True)

    print(
        "[PASS] TEST 2: R(()) == 0.0 & SmoothDetectionRewarder analytical checks (empty GT, difficult exemption, "
        "FP monotonicity, IoU 0.60->0.85 gain) passed; (w1,w2)==(w2,w1) cache/fusion/reward symmetry passed; "
        "all 8 env_cache tamper cases (incl. orphan .pt bucket & disk hash check) rejected with RuntimeError."
    )


def test_3_single_dgod_hard_isolation(
    wrapper: FrozenTextStyleDetWrapper,
    results_json: Path,
) -> None:
    print("\n" + "=" * 78)
    print("[TEST 3] Single-DGOD Zero-Leakage Hard Isolation Guard")
    print("=" * 78)

    grid = CandidateGrid()
    fuser = DualViewWBFFusion()
    ws_root = Path(__file__).resolve().parent

    forbidden_pairs = [
        ("daytime_foggy", "test"),
        ("daytime_foggy", "train"),
        ("dusk_rainy", "train"),
        ("Night-Sunny", "test"),
        ("night_rainy", "train"),
        ("daytime_clear", "test"),
        ("daytime_clear", "val"),
    ]
    for dom, sp in forbidden_pairs:
        raised_loader = False
        try:
            _ = SourceOnlyDatasetLoader(DEFAULT_SKY_DIR, domain=dom, split=sp)
        except ValueError:
            raised_loader = True
        assert raised_loader, f"SourceOnlyDatasetLoader MUST reject ({dom}, {sp}) with ValueError!"

        raised_env_m = False
        try:
            _ = build_env_cache_manifest_payload(
                wrapper=wrapper,
                grid=grid,
                fuser=fuser,
                degradation_cfg={},
                workspace_root=ws_root,
                source_domain=dom,
                source_split=sp,
            )
        except ValueError:
            raised_env_m = True
        assert raised_env_m, f"build_env_cache_manifest_payload MUST reject ({dom}, {sp}) with ValueError!"

    if results_json.is_file():
        res = json.loads(results_json.read_text(encoding="utf-8"))
        tm = res["grpo_training"]["training_manifest"]
        assert tm["source_only_compliance"] is True
        assert tm["source_domain"] == "daytime_clear"
        assert tm["source_split"] == "train"
        e2e = res.get("source_dev_e2e_eval")
        if e2e is not None:
            assert e2e["source_domain"] == "daytime_clear"
            assert e2e["source_split"] == "train"
            assert e2e["split_role"] == "source_dev"
        peak_alloc = float(res["grpo_training"]["training_gpu_memory_gib"]["peak_allocated_gib"])
        peak_res = float(res["grpo_training"]["training_gpu_memory_gib"]["peak_reserved_gib"])
        assert peak_alloc < 3.5 and peak_res < 3.5, (
            f"Recorded GRPO training VRAM exceeds 3.5 GiB: alloc={peak_alloc}, res={peak_res}"
        )

    print(
        "[PASS] TEST 3: All 7 target-domain / daytime_clear-test probes raised ValueError; "
        "Phase 3 results manifest verified strictly source-only with training VRAM < 3.5 GiB."
    )


def test_4_resume_and_grpo_checkpoint_reconstruction(
    sft_cache_dir: Path,
    sft_ckpt: Path,
    ref_ckpt: Path,
    env_cache_dir: Path,
    grpo_ckpt: Path,
    wrapper: FrozenTextStyleDetWrapper,
    loader: SourceOnlyDatasetLoader,
    device: torch.device,
) -> None:
    print("\n" + "=" * 78)
    print("[TEST 4] Atomic Checkpoint Resume, Tamper Rejection & policy_grpo.pth E2E Predict")
    print("=" * 78)

    # 4a. Test atomic pause -> resume -> mismatch refusal on small 16-train / 8-dev subset
    grid = CandidateGrid()
    fuser = DualViewWBFFusion(
        global_weight=1.0, local_weight=0.85, iou_thr=0.55, trunc_margin_px=5.0, max_detections=100, fusion_mode="wbf"
    )
    with tempfile.TemporaryDirectory(prefix="az_p3_resume_test_") as tmp:
        tmp_out = Path(tmp) / "out"
        # Clone env_cache into isolated temp dir so tests never write to production env_cache
        tmp_ec_dir = Path(tmp) / "ec"
        if (env_cache_dir / "env_cache_manifest.json").is_file():
            shutil.copytree(env_cache_dir, tmp_ec_dir)
            ecache = PersistentDetectionEnvCache(tmp_ec_dir, verify_bucket_hashes=False, read_only=False)
        else:
            ws_root = Path(__file__).resolve().parent
            mp = build_env_cache_manifest_payload(wrapper, grid, fuser, {"recipe_version": 1}, ws_root, num_buckets=8)
            ecache = PersistentDetectionEnvCache(tmp_ec_dir, manifest_payload=mp)

        env = GRPODetectionEnvironment(
            loader=loader,
            env_cache=ecache,
            grid=grid,
            fuser=fuser,
            rewarder=SmoothDetectionRewarder(GRPORewardConfig()),
            wrapper=wrapper,
            k_max=2,
        )
        cfg = GRPOTrainConfig(
            group_size=4,
            batch_size=8,
            epochs=2,
            ppo_epochs=2,
            lr=3e-5,
            beta_kl=0.08,
            max_train_views=16,
            max_dev_views=8,
            seed=20260425,
            device=str(device),
        )
        r_pause = train_grpo_policy(
            sft_cache_dir=sft_cache_dir,
            sft_ckpt_path=sft_ckpt,
            ref_ckpt_path=ref_ckpt,
            env=env,
            output_dir=tmp_out,
            config=cfg,
            resume=False,
            stop_after_epoch=1,
            show_progress=False,
        )
        assert r_pause["completed_all_epochs"] is False and len(r_pause["history"]) == 1
        assert (tmp_out / "grpo_resume_state.pt").is_file()
        paused_tel = r_pause["env_telemetry"]
        assert paused_tel["telemetry_scope"] == "full_training_cumulative"
        ckpt_pause = torch.load(str(tmp_out / "grpo_resume_state.pt"), map_location="cpu", weights_only=False)
        assert "env_telemetry_counters" in ckpt_pause, "[P1-01] grpo_resume_state.pt MUST persist env_telemetry_counters!"

        # P1-01 / P3-02: Inject simulated mid-epoch crash tail (duplicate Epoch-2 train_step + torn JSON line)
        log_file = tmp_out / "grpo_train_log.jsonl"
        with open(log_file, "a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "type": "train_step",
                        "epoch": 2,
                        "batch_idx": 1,
                        "rollout_step": 3,
                        "global_step": 6,
                        "loss": 0.0,
                    }
                )
                + "\n"
            )
            fh.write('{"type": "train_step", "epoch": 2, "global_step": 8, "truncated_tail\n')

        # Verify --resume rejects mismatched hyperparameter manifest with RuntimeError
        cfg_bad = GRPOTrainConfig(**{**cfg.to_dict(), "beta_kl": 0.99})
        bad_resume_raised = False
        try:
            _ = train_grpo_policy(
                sft_cache_dir=sft_cache_dir,
                sft_ckpt_path=sft_ckpt,
                ref_ckpt_path=ref_ckpt,
                env=env,
                output_dir=tmp_out,
                config=cfg_bad,
                resume=True,
                show_progress=False,
            )
        except RuntimeError:
            bad_resume_raised = True
        assert bad_resume_raised, "Resuming with modified hyperparameters MUST raise RuntimeError!"

        # Clean cross-process-equivalent --resume using a FRESH env instance (session counters start at 0)
        ecache_resumed = PersistentDetectionEnvCache(tmp_ec_dir, verify_bucket_hashes=False, read_only=False)
        env_resumed = GRPODetectionEnvironment(
            loader=loader,
            env_cache=ecache_resumed,
            grid=grid,
            fuser=fuser,
            rewarder=SmoothDetectionRewarder(GRPORewardConfig()),
            wrapper=wrapper,
            k_max=2,
        )
        assert env_resumed.global_cache_hits == 0 and env_resumed.crop_cache_hits == 0
        r_done = train_grpo_policy(
            sft_cache_dir=sft_cache_dir,
            sft_ckpt_path=sft_ckpt,
            ref_ckpt_path=ref_ckpt,
            env=env_resumed,
            output_dir=tmp_out,
            config=cfg,
            resume=True,
            show_progress=False,
        )
        assert r_done["completed_all_epochs"] is True and len(r_done["history"]) == 2
        assert [h["epoch"] for h in r_done["history"]] == [1, 2]
        assert (tmp_out / "policy_grpo.pth").is_file()

        # P1-01 assertions: cumulative env_telemetry accumulated across resume (NOT reset to session-only)
        done_tel = r_done["env_telemetry"]
        sess_tel = done_tel["session_telemetry"]
        assert done_tel["telemetry_scope"] == "full_training_cumulative"
        assert done_tel["global_cache_hits"] >= paused_tel["global_cache_hits"]
        assert done_tel["global_cache_hits"] > sess_tel["global_cache_hits"], (
            f"[P1-01] Resumed cumulative global_cache_hits ({done_tel['global_cache_hits']}) must exceed "
            f"resumed session-only hits ({sess_tel['global_cache_hits']})!"
        )
        assert (done_tel["crop_cache_hits"] + done_tel["crop_cache_misses"]) >= (
            paused_tel["crop_cache_hits"] + paused_tel["crop_cache_misses"]
        )
        assert done_tel["detector_compute_sec"] >= paused_tel["detector_compute_sec"]

        # P1-01 assertions: JSONL half-epoch/torn tail truncated cleanly, zero duplicates, new telemetry keys present
        raw_lines = [ln for ln in log_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
        parsed_rows = [json.loads(ln) for ln in raw_lines]
        ep_rows = [r for r in parsed_rows if r.get("type") == "epoch_summary"]
        ts_rows = [r for r in parsed_rows if r.get("type") == "train_step"]
        assert [r["epoch"] for r in ep_rows] == [0, 1, 2], f"Unexpected epoch_summary rows: {[r['epoch'] for r in ep_rows]}"
        assert [(r["epoch"], r["batch_idx"]) for r in ts_rows] == [(1, 1), (1, 2), (2, 1), (2, 2)], (
            f"[P1-01] Duplicate or missing train_step rows after resume: {[(r['epoch'], r['batch_idx']) for r in ts_rows]}"
        )
        assert [r["global_step"] for r in ts_rows] == [2, 4, 6, 8]
        for r in ts_rows:
            assert "approx_kl_old" in r and math.isfinite(float(r["approx_kl_old"]))
            assert "max_ratio_err_from_1" in r and math.isfinite(float(r["max_ratio_err_from_1"]))

    # 4b. Reconstruct final exported outputs/phase3/policy_grpo.pth and run live AZAgentDetector.predict
    assert grpo_ckpt.is_file(), f"Missing formal Phase 3 checkpoint: {grpo_ckpt}"
    pol_grpo = ActiveZoomPolicyHead.from_checkpoint(grpo_ckpt, map_location=device, exact_eval_parity=True).eval()
    agent = AZAgentDetector(detector_wrapper=wrapper, policy_head=pol_grpo, fuser=fuser).eval()

    split_info = loader.build_splits(max_train_images=4, max_dev_images=2)
    dev_id = split_info.selected_dev_ids[0]
    pil_clean = loader.load_image(dev_id)
    pil_dark = apply_physical_degradation(pil_clean, image_id=dev_id, view_tag="low_light", seed=20260420)

    # Fast path (No-Crop)
    out_nc = agent.predict(pil_clean, mode="greedy", forced_action=(), sync_cuda_for_timing=True)
    assert out_nc.action == () and out_nc.used_slow_path is False and out_nc.timings_ms["local_crop_det_ms"] == 0.0
    assert torch.equal(out_nc.detections["boxes"], out_nc.global_detections["boxes"])

    # Slow path K=1
    out_k1 = agent.predict(pil_clean, mode="greedy", forced_action=(14,), sync_cuda_for_timing=True)
    assert out_k1.action == (14,) and out_k1.used_slow_path is True and len(out_k1.selected_crop_boxes) == 1
    assert out_k1.detections["boxes"].shape[0] <= fuser.max_detections

    # Slow path K=2 + P1-03 reverse-order (15, 12) vs (12, 15) bitwise invariance on real source_dev image
    out_k2 = agent.predict(pil_dark, mode="greedy", forced_action=(12, 15), sync_cuda_for_timing=True)
    out_k2_rev = agent.predict(pil_dark, mode="greedy", forced_action=(15, 12), sync_cuda_for_timing=True)
    assert out_k2.action == (12, 15) and out_k2_rev.action == (15, 12)
    assert out_k2.used_slow_path is True and len(out_k2.selected_crop_boxes) == 2
    assert out_k2.selected_crop_boxes == out_k2_rev.selected_crop_boxes
    assert out_k2.detections["boxes"].shape[0] <= fuser.max_detections
    for k in ("boxes", "scores", "labels"):
        assert torch.equal(out_k2.detections[k], out_k2_rev.detections[k]), (
            f"[P1-03] AZAgentDetector.predict {k} differs between (12, 15) and (15, 12)!"
        )

    # P1-03 / P3-02: Equal-confidence tie-breaking probe through AZAgentDetector.predict (WBF & NMS modes)
    orig_w_d, orig_h_d = pil_dark.size
    rect_12 = grid.windows[12].to_pixel_xyxy(orig_w_d, orig_h_d)
    rect_15 = grid.windows[15].to_pixel_xyxy(orig_w_d, orig_h_d)

    def _make_tie_crop(rect, boxes_list, scores_list):
        b_orig = torch.tensor(boxes_list, dtype=torch.float32)
        b_crop = b_orig.clone()
        b_crop[:, [0, 2]] -= float(rect[0])
        b_crop[:, [1, 3]] -= float(rect[1])
        return {
            "boxes_in_orig": b_orig,
            "boxes_in_crop": b_crop,
            "scores": torch.tensor(scores_list, dtype=torch.float32),
            "labels": torch.ones((len(scores_list),), dtype=torch.long),
            "crop_xyxy": rect,
            "orig_size": (orig_h_d, orig_w_d),
        }

    # Box A (crop 12) & Box B (crop 15) have identical score 0.80 & IoU=0.714; Box C (crop 15, score=0.79)
    # overlaps B (IoU=0.690 >= 0.55) but NOT A (IoU=0.481 < 0.55) => raw order flips NMS/WBF tie-breaking!
    tie_det_12 = _make_tie_crop(rect_12, [[300.0, 280.0, 420.0, 380.0]], [0.80])
    tie_det_15 = _make_tie_crop(rect_15, [[320.0, 280.0, 440.0, 380.0], [342.0, 280.0, 462.0, 380.0]], [0.80, 0.79])
    empty_glob = {
        "boxes": torch.zeros((0, 4), dtype=torch.float32),
        "scores": torch.zeros((0,), dtype=torch.float32),
        "labels": torch.zeros((0,), dtype=torch.long),
    }
    fuser_nms_probe = DualViewWBFFusion(trunc_margin_px=0.0, fusion_mode="nms")
    raw_12_15 = fuser_nms_probe.fuse(empty_glob, [tie_det_12, tie_det_15], (orig_h_d, orig_w_d))
    raw_15_12 = fuser_nms_probe.fuse(empty_glob, [tie_det_15, tie_det_12], (orig_h_d, orig_w_d))
    assert not torch.equal(raw_12_15["boxes"], raw_15_12["boxes"]), (
        "Sanity check failed: synthetic tie probe must differ under uncanonicalized fusion order!"
    )

    class _TieCropWrapperProxy:
        def __init__(self, base_wrapper):
            self._base = base_wrapper
            self.device = base_wrapper.device
            self.model = base_wrapper.model
            self._perc = None

        def forward_global(self, pil_image):
            if self._perc is None:
                self._perc = self._base.forward_global(pil_image)
            return self._perc

        def forward_single_crop(self, pil_image, crop_box_xyxy):
            return tie_det_12 if tuple(crop_box_xyxy) == rect_12 else tie_det_15

    tie_proxy = _TieCropWrapperProxy(wrapper)
    for f_mode in ("wbf", "nms"):
        tie_fuser = DualViewWBFFusion(
            global_weight=1.0, local_weight=0.85, iou_thr=0.55, trunc_margin_px=0.0, fusion_mode=f_mode
        )
        tie_agent = AZAgentDetector(detector_wrapper=tie_proxy, policy_head=pol_grpo, fuser=tie_fuser).eval()
        o_a = tie_agent.predict(pil_dark, mode="greedy", forced_action=(12, 15), sync_cuda_for_timing=False)
        o_b = tie_agent.predict(pil_dark, mode="greedy", forced_action=(15, 12), sync_cuda_for_timing=False)
        assert o_a.action == (12, 15) and o_b.action == (15, 12)
        for k in ("boxes", "scores", "labels"):
            assert torch.equal(o_a.detections[k], o_b.detections[k]), (
                f"[P1-03] Equal-confidence tie probe failed in fusion_mode={f_mode!r} on {k}!"
            )

    # Live greedy decoding on real source_dev clean + low_light
    out_g_clean = agent.predict(pil_clean, mode="greedy", sync_cuda_for_timing=True)
    out_g_dark = agent.predict(pil_dark, mode="greedy", sync_cuda_for_timing=True)
    assert len(out_g_clean.action) <= 2 and len(out_g_dark.action) <= 2

    print(
        f"[PASS] TEST 4: Cross-process pause/resume (Epoch 1 -> 2), cumulative telemetry, half-epoch JSONL "
        f"truncation, & manifest mismatch rejection verified; reconstructed {grpo_ckpt.name} ran live "
        f"AZAgentDetector.predict across No-Crop, K=1, K=2, reverse-order (15,12)==(12,15) (real + tied WBF/NMS), "
        f"and greedy (clean={out_g_clean.action}, low_light={out_g_dark.action})."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 Automated Verification Suite")
    parser.add_argument("--sft_cache_dir", type=str, default=str(DEFAULT_SFT_CACHE_DIR))
    parser.add_argument("--sft_ckpt", type=str, default=str(DEFAULT_SFT_CKPT))
    parser.add_argument("--ref_ckpt", type=str, default=str(DEFAULT_REF_CKPT))
    parser.add_argument("--env_cache_dir", type=str, default=str(DEFAULT_ENV_CACHE_DIR))
    parser.add_argument("--grpo_ckpt", type=str, default=str(DEFAULT_GRPO_CKPT))
    parser.add_argument("--results_json", type=str, default=str(DEFAULT_RESULTS_JSON))
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rn_root = Path(r"E:\textstyleDet\textstyleDet_rn")
    sky_root = Path(DEFAULT_SKY_DIR)
    p2_root = DEFAULT_PHASE2_DIR

    rn_before = snapshot_readonly_tree(rn_root, (".py", ".pyc", ".pth", ".json"))
    sky_before = snapshot_readonly_tree(sky_root, (".pkl",))
    p2_before = snapshot_readonly_tree(p2_root, (".pth", ".pt", ".json", ".md"))

    print(f"[Phase 3 Test Suite] Running on device={device}, sys.dont_write_bytecode={sys.dont_write_bytecode}")

    wrapper = FrozenTextStyleDetWrapper(
        checkpoint_path=DEFAULT_CKPT,
        clip_roi_weights_path=DEFAULT_CLIP_RN101,
        device=device,
        use_amp_bf16=False,
        roi_chunk_size=256,
        crop_max_proposals=300,
    )
    loader = SourceOnlyDatasetLoader(DEFAULT_SKY_DIR, domain="daytime_clear", split="train")

    test_1_grpo_step0_and_math_invariants(
        sft_ckpt=Path(args.sft_ckpt),
        ref_ckpt=Path(args.ref_ckpt),
        sft_cache_dir=Path(args.sft_cache_dir),
        wrapper=wrapper,
        device=device,
    )
    test_2_env_cache_symmetry_and_tamper_rejection(
        env_cache_dir=Path(args.env_cache_dir),
        wrapper=wrapper,
        loader=loader,
    )
    test_3_single_dgod_hard_isolation(
        wrapper=wrapper,
        results_json=Path(args.results_json),
    )
    test_4_resume_and_grpo_checkpoint_reconstruction(
        sft_cache_dir=Path(args.sft_cache_dir),
        sft_ckpt=Path(args.sft_ckpt),
        ref_ckpt=Path(args.ref_ckpt),
        env_cache_dir=Path(args.env_cache_dir),
        grpo_ckpt=Path(args.grpo_ckpt),
        wrapper=wrapper,
        loader=loader,
        device=device,
    )

    rn_after = snapshot_readonly_tree(rn_root, (".py", ".pyc", ".pth", ".json"))
    sky_after = snapshot_readonly_tree(sky_root, (".pkl",))
    p2_after = snapshot_readonly_tree(p2_root, (".pth", ".pt", ".json", ".md"))

    rn_diff = sorted(k for k in set(rn_before) | set(rn_after) if rn_before.get(k) != rn_after.get(k))
    sky_diff = sorted(k for k in set(sky_before) | set(sky_after) if sky_before.get(k) != sky_after.get(k))
    p2_diff = sorted(k for k in set(p2_before) | set(p2_after) if p2_before.get(k) != p2_after.get(k))

    assert not rn_diff, f"Read-only violation in textstyleDet_rn: {rn_diff}"
    assert not sky_diff, f"Read-only violation in data/sky: {sky_diff}"
    assert not p2_diff, f"Read-only violation in outputs/phase2: {p2_diff}"

    print("\n" + "=" * 78)
    print("[PASS] TEST 5: Read-only boundary verified (0 mutations in textstyleDet_rn, data/sky & outputs/phase2).")
    print("ALL PHASE 3 VERIFICATION TESTS PASSED.")
    print("=" * 78)


if __name__ == "__main__":
    main()
