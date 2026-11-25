#!/usr/bin/env python
"""Phase 3 CLI: Stage-2 GRPO Reinforcement Learning, Resumption & 5-Way Source-Dev E2E Eval.

Workflow:
  1. Verifies Phase 2 sealed ``sft_cache`` manifest and Phase 2 ``policy_sft.pth`` / ``policy_sft_ref.pth``
     (read-only — never overwrites or mutates ``outputs/phase2/``).
  2. Builds or resumes P0-01 isolated, symmetry-aware persistent detection environment cache
     (``outputs/phase3/env_cache/``) guarded by cryptographic ``env_cache_manifest.json``.
  3. Loads online ``pi_theta`` with ``dropout_override=0.0, exact_eval_parity=True`` and frozen
     ``pi_ref`` with ``as_reference=True, exact_eval_parity=True``, enforcing Phase 2 P1-01
     Step-0 alignment assertions (``max|r-1|<1e-6``, ``premature_clip==0``, ``max D_KL<1e-6``).
  4. Runs Stage-2 GRPO optimization strictly on source-only ``daytime_clear/train`` views with
     atomic per-epoch resumable checkpointing (``--resume``) and ``source_dev`` best-epoch locking,
     exporting ``outputs/phase3/policy_grpo.pth``.
  5. Evaluates all 5 schemes end-to-end on the exact deterministic Phase-2 ``source_dev``
     250 clean + 250 synthetic degraded views (zero target-domain access, zero GT coupling).
  6. Generates 100% JSON-derived ``phase3_grpo_results.json`` & ``phase3_grpo_report.md`` and
     verifies zero mutations in ``textstyleDet_rn``, ``data/sky``, and ``outputs/phase2``.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import json
from pathlib import Path

from run_phase0 import DEFAULT_CKPT, DEFAULT_CLIP_RN101, DEFAULT_SKY_DIR, snapshot_readonly_tree
from zoom_agent.candidate_grid import CandidateGrid
from zoom_agent.detector_wrapper import FrozenTextStyleDetWrapper
from zoom_agent.eval_utils import atomic_write_text, sha256_file
from zoom_agent.grpo_env import (
    DEFAULT_NUM_BUCKETS,
    GRPODetectionEnvironment,
    GRPORewardConfig,
    PersistentDetectionEnvCache,
    SmoothDetectionRewarder,
    build_env_cache_manifest_payload,
)
from zoom_agent.grpo_trainer import (
    GRPOTrainConfig,
    SubsetSFTCacheView,
    evaluate_phase3_e2e_on_source_dev,
    generate_phase3_markdown_report,
    train_grpo_policy,
)
from zoom_agent.sft_cache import ShardedSFTCacheReader, verify_cache_manifest
from zoom_agent.source_degradation import (
    ALLOWED_SOURCE_DOMAIN,
    ALLOWED_SOURCE_SPLIT,
    SourceOnlyDatasetLoader,
)
from zoom_agent.wbf_fusion import DualViewWBFFusion

DEFAULT_PHASE2_DIR = r"E:\textstyleDet\TextStyleDet_vlm\outputs\phase2"
DEFAULT_SFT_CACHE_DIR = r"E:\textstyleDet\TextStyleDet_vlm\outputs\phase2\sft_cache"
DEFAULT_SFT_CKPT = r"E:\textstyleDet\TextStyleDet_vlm\outputs\phase2\policy_sft.pth"
DEFAULT_REF_CKPT = r"E:\textstyleDet\TextStyleDet_vlm\outputs\phase2\policy_sft_ref.pth"
DEFAULT_PHASE3_OUT_DIR = r"E:\textstyleDet\TextStyleDet_vlm\outputs\phase3"
DEFAULT_ENV_CACHE_DIR = r"E:\textstyleDet\TextStyleDet_vlm\outputs\phase3\env_cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 3 Stage-2 GRPO Trainer, Isolated Env Cache & 5-Way Source-Dev E2E Evaluator"
    )
    # Input / output paths
    parser.add_argument("--sft_cache_dir", type=str, default=DEFAULT_SFT_CACHE_DIR)
    parser.add_argument("--sft_ckpt", type=str, default=DEFAULT_SFT_CKPT)
    parser.add_argument("--ref_ckpt", type=str, default=DEFAULT_REF_CKPT)
    parser.add_argument("--env_cache_dir", type=str, default=DEFAULT_ENV_CACHE_DIR)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_PHASE3_OUT_DIR)
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CKPT)
    parser.add_argument("--clip_weights", type=str, default=DEFAULT_CLIP_RN101)
    parser.add_argument("--data_dir", type=str, default=DEFAULT_SKY_DIR)

    # Dataset view scale (strictly within sealed source_train / source_dev; synced to Phase 3 sealed defaults)
    parser.add_argument("--max_train_views", type=int, default=2000)
    parser.add_argument("--max_dev_views", type=int, default=600)
    parser.add_argument("--shard_group_size", type=int, default=2)
    parser.add_argument("--num_env_buckets", type=int, default=DEFAULT_NUM_BUCKETS)
    parser.add_argument(
        "--precache_only",
        action="store_true",
        help="Only pre-cache global detections into env_cache and exit without GRPO updates.",
    )
    parser.add_argument("--precache_globals_first", action="store_true")

    # Smooth reward hyperparameters (Phase 3 sealed defaults; P1-02 synced)
    parser.add_argument("--lambda_recall", type=float, default=2.0)
    parser.add_argument("--lambda_loc_ap", type=float, default=3.5)
    parser.add_argument("--lambda_fp", type=float, default=0.90)
    parser.add_argument("--lambda_cost", type=float, default=0.045)
    parser.add_argument("--loc_w_ap50_95", type=float, default=0.45)
    parser.add_argument("--loc_w_ap75", type=float, default=0.30)
    parser.add_argument("--loc_w_matched_iou", type=float, default=0.25)
    parser.add_argument(
        "--fp_score_thr",
        type=float,
        default=0.05,
        help="FP penalty score floor (sealed Phase 3 default=0.05; set 0.04 for future P2-03 aligned ablations).",
    )

    # GRPO optimization hyperparameters (Phase 3 sealed defaults; P1-02 synced)
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--ppo_epochs", type=int, default=2)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--beta_kl", type=float, default=0.16)
    parser.add_argument("--alpha_ent", type=float, default=0.015)
    parser.add_argument("--lr", type=float, default=8e-6)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--warmup_epochs", type=float, default=0.5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--sampling_temp", type=float, default=1.05)
    parser.add_argument(
        "--kl_entropy_temp",
        type=float,
        default=1.0,
        help="Temperature T_reg for analytical KL(pi||pi_ref) and entropy regularization (P2-01, default=1.0).",
    )
    parser.add_argument("--selection_metric", type=str, default="composite", choices=["composite", "reward"])
    parser.add_argument("--flush_env_every_batches", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260425)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume", action="store_true", help="Atomically resume GRPO training from grpo_resume_state.pt")
    parser.add_argument(
        "--stop_after_epoch",
        type=int,
        default=0,
        help="Optional graceful pause after N completed epochs (for testing/staged --resume; 0 = run all epochs).",
    )

    # Detector & Fusion contract (must match Phase 2 FP32 baseline)
    parser.add_argument("--trunc_margin_px", type=float, default=5.0)
    parser.add_argument("--max_detections", type=int, default=100)
    parser.add_argument("--roi_chunk_size", type=int, default=256)
    parser.add_argument("--crop_max_proposals", type=int, default=300)
    parser.add_argument("--use_bf16", action="store_true", help="Default False (standard FP32)")

    # Output naming & E2E verification
    parser.add_argument("--grpo_ckpt_name", type=str, default="policy_grpo.pth")
    parser.add_argument("--resume_ckpt_name", type=str, default="grpo_resume_state.pt")
    parser.add_argument("--train_log_name", type=str, default="grpo_train_log.jsonl")
    parser.add_argument("--report_prefix", type=str, default="phase3_grpo")
    parser.add_argument("--skip_e2e_eval", action="store_true")
    parser.add_argument(
        "--e2e_only",
        action="store_true",
        help="Load verified existing grpo_training JSON + policy_grpo.pth and run 5-way source_dev E2E evaluation.",
    )
    parser.add_argument("--eval_dev_images", type=int, default=250)
    parser.add_argument(
        "--batch_cooldown_ms",
        type=float,
        default=0.0,
        help="Optional thermal cooldown pause (ms) after each GRPO rollout batch to reduce laptop GPU load.",
    )
    parser.add_argument(
        "--e2e_cooldown_ms",
        type=float,
        default=0.0,
        help="Optional thermal cooldown pause (ms) outside timed E2E blocks to reduce laptop GPU heat.",
    )
    parser.add_argument("--no_progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace_root = Path(__file__).resolve().parent
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rn_root = Path(r"E:\textstyleDet\textstyleDet_rn")
    sky_root = Path(args.data_dir)
    phase2_root = Path(DEFAULT_PHASE2_DIR)

    # Read-only boundary snapshots BEFORE execution
    rn_snap_before = snapshot_readonly_tree(rn_root, (".py", ".pyc", ".pth", ".json"))
    sky_pkl_before = snapshot_readonly_tree(sky_root, (".pkl",))
    phase2_snap_before = (
        snapshot_readonly_tree(phase2_root, (".pth", ".pt", ".json", ".md"))
        if phase2_root.is_dir()
        else {}
    )

    # 1) Verify Phase 2 sealed SFT cache manifest
    sft_doc = verify_cache_manifest(
        args.sft_cache_dir,
        verify_shard_hashes=False,
        require_completed=True,
        require_readonly_audit=True,
        require_completed_seal=True,
    )
    sft_manifest = sft_doc["manifest"]
    split_cfg = sft_manifest["split_config"]
    deg_cfg = sft_manifest["degradation_recipe"]
    deg_seed = int(deg_cfg["base_seed"])

    # 2) Instantiate strict source-only loader & verify split SHA256 matches sealed SFT cache
    loader = SourceOnlyDatasetLoader(
        data_dir=args.data_dir,
        domain=ALLOWED_SOURCE_DOMAIN,
        split=ALLOWED_SOURCE_SPLIT,
        dev_ratio=float(split_cfg["dev_ratio"]),
        split_seed=int(split_cfg["split_seed"]),
        rn_root=rn_root,
    )
    split_info = loader.build_splits(
        max_train_images=int(split_cfg["selected_train_count"]),
        max_dev_images=int(split_cfg["selected_dev_count"]),
    )
    assert split_info.selected_train_sha256 == split_cfg["selected_train_sha256"], (
        "source_train split SHA256 mismatch vs Phase 2 sft_cache!"
    )
    assert split_info.selected_dev_sha256 == split_cfg["selected_dev_sha256"], (
        "source_dev split SHA256 mismatch vs Phase 2 sft_cache!"
    )

    # 3) Load permanently frozen StyleDetV6 (FP32) + Grid + WBF Fuser
    print(
        f"[Phase 3] Loading permanently frozen StyleDetV6 (use_bf16={args.use_bf16}, "
        f"roi_chunk_size={args.roi_chunk_size}, crop_max_proposals={args.crop_max_proposals})...",
        flush=True,
    )
    wrapper = FrozenTextStyleDetWrapper(
        checkpoint_path=args.checkpoint,
        clip_roi_weights_path=args.clip_weights,
        rn_root=rn_root,
        device=args.device,
        use_amp_bf16=args.use_bf16,
        roi_chunk_size=args.roi_chunk_size,
        crop_max_proposals=args.crop_max_proposals,
    )

    g_cfg = sft_manifest["candidate_grid"]
    grid = CandidateGrid(
        grid_rows=int(g_cfg["grid_rows"]),
        grid_cols=int(g_cfg["grid_cols"]),
        grid_win_w=float(g_cfg["grid_win_w"]),
        grid_win_h=float(g_cfg["grid_win_h"]),
    )
    fuser = DualViewWBFFusion(
        global_weight=1.0,
        local_weight=0.85,
        iou_thr=0.55,
        trunc_margin_px=args.trunc_margin_px,
        max_detections=args.max_detections,
        fusion_mode="wbf",
    )

    # 4) Initialize or verify persistent P0-01 isolated detection environment cache
    env_manifest_payload = build_env_cache_manifest_payload(
        wrapper=wrapper,
        grid=grid,
        fuser=fuser,
        degradation_cfg=deg_cfg,
        workspace_root=workspace_root,
        source_domain=ALLOWED_SOURCE_DOMAIN,
        source_split=ALLOWED_SOURCE_SPLIT,
        num_buckets=args.num_env_buckets,
        sft_cache_manifest_digest=sft_doc["manifest_digest"],
    )
    env_cache = PersistentDetectionEnvCache(
        cache_dir=args.env_cache_dir,
        manifest_payload=env_manifest_payload,
        verify_bucket_hashes=True,
        read_only=False,
    )
    reward_cfg = GRPORewardConfig(
        lambda_recall=args.lambda_recall,
        lambda_loc_ap=args.lambda_loc_ap,
        lambda_fp=args.lambda_fp,
        lambda_cost=args.lambda_cost,
        loc_w_ap50_95=args.loc_w_ap50_95,
        loc_w_ap75=args.loc_w_ap75,
        loc_w_matched_iou=args.loc_w_matched_iou,
        fp_score_thr=args.fp_score_thr,
    )
    rewarder = SmoothDetectionRewarder(config=reward_cfg)
    env = GRPODetectionEnvironment(
        loader=loader,
        env_cache=env_cache,
        grid=grid,
        fuser=fuser,
        rewarder=rewarder,
        wrapper=wrapper,
        k_max=int(sft_manifest["policy_action_contract"]["k_max"]),
    )

    if args.precache_only or args.precache_globals_first:
        tr_r = ShardedSFTCacheReader(args.sft_cache_dir, "source_train", verify_shard_hashes=False, require_completed=True)
        dv_r = ShardedSFTCacheReader(args.sft_cache_dir, "source_dev", verify_shard_hashes=False, require_completed=True)
        tr_vks, _, _ = SubsetSFTCacheView(tr_r, args.max_train_views).collect_metadata()
        dv_vks, _, _ = SubsetSFTCacheView(dv_r, args.max_dev_views).collect_metadata()
        pre_res = env.precache_global_views(tr_vks + dv_vks, show_progress=not args.no_progress)
        print(f"[Phase 3] Global env pre-cache complete: {pre_res}", flush=True)
        if args.precache_only:
            return

    # 5) Stage-2 GRPO Reinforcement Learning (or load verified existing training record under --e2e_only)
    json_path = out_dir / f"{args.report_prefix}_results.json"
    if args.e2e_only:
        if not json_path.is_file():
            raise FileNotFoundError(f"--e2e_only requires existing training results JSON: {json_path}")
        prev_doc = json.loads(json_path.read_text(encoding="utf-8"))
        grpo_res = prev_doc["grpo_training"]
        grpo_ckpt_path = out_dir / args.grpo_ckpt_name
        if not grpo_ckpt_path.is_file():
            raise FileNotFoundError(f"--e2e_only requires policy checkpoint: {grpo_ckpt_path}")
        actual_sha = sha256_file(grpo_ckpt_path)
        expected_sha = str(grpo_res["checkpoint_export"]["policy_grpo_sha256"])
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"policy_grpo.pth SHA256 mismatch under --e2e_only: actual={actual_sha} != recorded={expected_sha}"
            )
        grpo_res["checkpoint_export"]["policy_grpo_path"] = str(grpo_ckpt_path.resolve())
        grpo_res["checkpoint_export"]["resume_state_path"] = str((out_dir / args.resume_ckpt_name).resolve())
        grpo_res["checkpoint_export"]["train_log_path"] = str((out_dir / args.train_log_name).resolve())
        print(
            f"[Phase 3] --e2e_only verified existing {grpo_ckpt_path.name} (SHA256={actual_sha[:16]}..., "
            f"Best Epoch={grpo_res['best_epoch']}).",
            flush=True,
        )
    else:
        train_cfg = GRPOTrainConfig(
            group_size=args.group_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            ppo_epochs=args.ppo_epochs,
            clip_eps=args.clip_eps,
            beta_kl=args.beta_kl,
            alpha_ent=args.alpha_ent,
            lr=args.lr,
            min_lr=args.min_lr,
            warmup_epochs=args.warmup_epochs,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            sampling_temp=args.sampling_temp,
            kl_entropy_temp=args.kl_entropy_temp,
            selection_metric=args.selection_metric,
            max_train_views=args.max_train_views,
            max_dev_views=args.max_dev_views,
            shard_group_size=args.shard_group_size,
            flush_env_every_batches=args.flush_env_every_batches,
            seed=args.seed,
            device=args.device,
        )

        grpo_res = train_grpo_policy(
            sft_cache_dir=args.sft_cache_dir,
            sft_ckpt_path=args.sft_ckpt,
            ref_ckpt_path=args.ref_ckpt,
            env=env,
            output_dir=out_dir,
            config=train_cfg,
            grpo_ckpt_name=args.grpo_ckpt_name,
            resume_ckpt_name=args.resume_ckpt_name,
            train_log_name=args.train_log_name,
            resume=args.resume,
            stop_after_epoch=args.stop_after_epoch,
            batch_cooldown_ms=args.batch_cooldown_ms,
            show_progress=not args.no_progress,
        )

        if not grpo_res.get("completed_all_epochs", True):
            print(
                f"[Phase 3] Paused cleanly after epoch {len(grpo_res['history'])}/{args.epochs}. "
                "Re-run with --resume to finish remaining epochs and E2E evaluation.",
                flush=True,
            )
            return

    # 6) 5-Way Real End-to-End Detection Verification strictly on source_dev (250 clean + 250 degraded)
    e2e_res = None
    if not args.skip_e2e_eval:
        print(
            f"\n[Phase 3 E2E Eval] Evaluating 5 schemes on deterministic source_dev "
            f"({min(args.eval_dev_images, len(split_info.selected_dev_ids))} images x 2 views)...",
            flush=True,
        )
        e2e_res = evaluate_phase3_e2e_on_source_dev(
            policy_sft_path=args.sft_ckpt,
            policy_grpo_path=out_dir / args.grpo_ckpt_name,
            wrapper=wrapper,
            loader=loader,
            dev_image_ids=split_info.selected_dev_ids,
            degradation_seed=deg_seed,
            max_eval_images=args.eval_dev_images,
            trunc_margin_px=args.trunc_margin_px,
            max_detections=args.max_detections,
            e2e_cooldown_ms=args.e2e_cooldown_ms,
            e2e_resume_path=out_dir / "e2e_progress.pt",
            show_progress=not args.no_progress,
        )

    # 7) Read-only boundary verification across ALL 3 protected directories
    rn_snap_after = snapshot_readonly_tree(rn_root, (".py", ".pyc", ".pth", ".json"))
    sky_pkl_after = snapshot_readonly_tree(sky_root, (".pkl",))
    phase2_snap_after = (
        snapshot_readonly_tree(phase2_root, (".pth", ".pt", ".json", ".md"))
        if phase2_root.is_dir()
        else {}
    )
    readonly_audit = {
        "textstyle_rn_mutated_or_added": sorted(
            k for k in set(rn_snap_before) | set(rn_snap_after) if rn_snap_before.get(k) != rn_snap_after.get(k)
        ),
        "sky_pkl_mutated_or_added": sorted(
            k for k in set(sky_pkl_before) | set(sky_pkl_after) if sky_pkl_before.get(k) != sky_pkl_after.get(k)
        ),
        "outputs_phase2_mutated_or_added": sorted(
            k for k in set(phase2_snap_before) | set(phase2_snap_after)
            if phase2_snap_before.get(k) != phase2_snap_after.get(k)
        ),
    }
    if any(readonly_audit.values()):
        raise RuntimeError(f"Read-only boundary violation detected: {readonly_audit}")

    results = {
        "schema_version": 1,
        "readonly_boundary_audit": readonly_audit,
        "grpo_training": grpo_res,
        "source_dev_e2e_eval": e2e_res,
    }

    md_path = out_dir / f"{args.report_prefix}_report.md"
    atomic_write_text(json_path, json.dumps(results, indent=2, ensure_ascii=False))
    generate_phase3_markdown_report(results, md_path)

    print("\n================ PHASE 3 GRPO COMPLETE ================")
    print(
        f"  Best Epoch        : {grpo_res['best_epoch']} "
        f"(dev_R={grpo_res['best_source_dev']['mean_reward']:+.4f} vs SFT {grpo_res['epoch0_sft_source_dev']['mean_reward']:+.4f}, "
        f"sel={grpo_res['best_selection_score']:.4f})"
    )
    print(f"  Policy GRPO Ckpt  : {grpo_res['checkpoint_export']['policy_grpo_path']}")
    print(
        f"  Train Peak VRAM   : alloc={grpo_res['training_gpu_memory_gib']['peak_allocated_gib']:.4f} GiB | "
        f"res={grpo_res['training_gpu_memory_gib']['peak_reserved_gib']:.4f} GiB"
    )
    if e2e_res is not None:
        for vk in ("source_dev_clean", "source_dev_degraded"):
            vm = e2e_res["views"][vk]["metrics"]
            vb = e2e_res["views"][vk]["budget"]
            print(f"\n  [{vk}] ({e2e_res['views'][vk]['num_images']} imgs)")
            for mk in (
                "baseline",
                "rpn_heuristic_wbf",
                "az_agent_sft_greedy",
                "az_agent_grpo_greedy",
                "static_oracle_for_policy_wbf",
            ):
                fps_v = vb[mk].get("measured_e2e_fps")
                fps_s = f"{fps_v:.2f}" if fps_v is not None else "— (offline)"
                print(
                    f"    {mk:30s}: AP50={vm[mk]['AP50']:.2f} | AP75={vm[mk]['AP75']:.2f} | "
                    f"AP={vm[mk]['AP']:.2f} | weak_AP50={vm[mk]['weak_AP50']:.2f} | "
                    f"K={vb[mk]['avg_k']:.2f} | NC={vb[mk]['no_crop_pct']:.1f}% | FPS={fps_s}"
                )
    print(f"\n  Saved JSON        : {json_path}")
    print(f"  Saved Report      : {md_path}")
    print("  Read-only audit   : PASS (0 mutations in textstyleDet_rn, data/sky, and outputs/phase2)")


if __name__ == "__main__":
    main()
