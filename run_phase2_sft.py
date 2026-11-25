#!/usr/bin/env python
"""Phase 2 CLI: Stage-1 Offline SFT Training, pi_ref Export & Source-Dev E2E Verification.

Workflow:
  1. Verifies source-only offline cache manifest (zero detector loaded during SFT training).
  2. Trains ONLY ``ActiveZoomPolicyHead`` (~1.902M params w/ FiLM; ``--no_film`` ablation supported)
     using composite Multi-Label BCE + Two-Step Trajectory NLL loss with AdamW + Cosine LR.
  3. Selects best epoch strictly on ``source_dev`` and exports bitwise-identical
     ``policy_sft.pth`` and ``policy_sft_ref.pth`` via ``ActiveZoomPolicyHead.save_checkpoint``.
  4. Loads frozen ``StyleDetV6`` + ``AZAgentDetector`` with ``policy_sft.pth`` and evaluates
     real end-to-end detection on deterministic ``source_dev`` clean & synthetic degraded views
     (strictly ZERO target-domain access).
  5. Writes deterministic ``phase2_sft_results.json`` & ``phase2_sft_report.md`` and asserts
     read-only boundary integrity on ``textstyleDet_rn`` and ``data/sky``.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import json
from pathlib import Path

from run_phase0 import DEFAULT_CKPT, DEFAULT_CLIP_RN101, DEFAULT_SKY_DIR, snapshot_readonly_tree
from zoom_agent.detector_wrapper import FrozenTextStyleDetWrapper
from zoom_agent.eval_utils import atomic_write_text
from zoom_agent.sft_trainer import (
    SFTTrainConfig,
    evaluate_sft_e2e_on_source_dev,
    generate_phase2_markdown_report,
    train_sft_policy_offline,
)
from zoom_agent.source_degradation import (
    ALLOWED_SOURCE_DOMAIN,
    ALLOWED_SOURCE_SPLIT,
    SourceOnlyDatasetLoader,
)

DEFAULT_PHASE2_OUT_DIR = r"E:\textstyleDet\TextStyleDet_vlm\outputs\phase2"
DEFAULT_PHASE2_CACHE_DIR = r"E:\textstyleDet\TextStyleDet_vlm\outputs\phase2\sft_cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 2 Stage-1 Offline SFT Trainer, pi_ref Exporter & Source-Dev E2E Evaluator"
    )
    parser.add_argument("--cache_dir", type=str, default=DEFAULT_PHASE2_CACHE_DIR)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_PHASE2_OUT_DIR)
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CKPT)
    parser.add_argument("--clip_weights", type=str, default=DEFAULT_CLIP_RN101)
    parser.add_argument("--data_dir", type=str, default=DEFAULT_SKY_DIR)

    # SFT training hyperparameters
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--warmup_epochs", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--traj_nll_weight", type=float, default=0.25)
    parser.add_argument("--max_pos_weight", type=float, default=4.0)
    parser.add_argument("--no_crop_pos_weight_scale", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--no_film", action="store_true", help="Disable FiLM style modulation (w/o FiLM ablation)")
    parser.add_argument("--seed", type=int, default=20260421)
    parser.add_argument("--shard_group_size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")

    # Output naming
    parser.add_argument("--sft_ckpt_name", type=str, default="policy_sft.pth")
    parser.add_argument("--ref_ckpt_name", type=str, default="policy_sft_ref.pth")
    parser.add_argument("--report_prefix", type=str, default="phase2_sft")

    # Source-dev real E2E detection verification
    parser.add_argument("--skip_e2e_eval", action="store_true")
    parser.add_argument(
        "--eval_dev_images",
        type=int,
        default=250,
        help="Deterministic source_dev subset size for clean + degraded E2E detection eval (200~300 recommended)",
    )
    parser.add_argument("--trunc_margin_px", type=float, default=5.0)
    parser.add_argument("--max_detections", type=int, default=100)
    parser.add_argument("--roi_chunk_size", type=int, default=256)
    parser.add_argument("--crop_max_proposals", type=int, default=300)
    parser.add_argument("--use_bf16", action="store_true", help="Default False (standard FP32)")
    parser.add_argument("--no_progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rn_root = Path(r"E:\textstyleDet\textstyleDet_rn")
    sky_root = Path(args.data_dir)

    rn_snap_before = snapshot_readonly_tree(rn_root, (".py", ".pyc", ".pth", ".json"))
    sky_pkl_before = snapshot_readonly_tree(sky_root, (".pkl",))

    # Stage A: Pure offline SFT training (frozen detector is NOT loaded here)
    train_cfg = SFTTrainConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        min_lr=args.min_lr,
        warmup_epochs=args.warmup_epochs,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        traj_nll_weight=args.traj_nll_weight,
        max_pos_weight=args.max_pos_weight,
        no_crop_pos_weight_scale=args.no_crop_pos_weight_scale,
        dropout=args.dropout,
        use_film=not args.no_film,
        seed=args.seed,
        shard_group_size=args.shard_group_size,
        device=args.device,
    )

    sft_res = train_sft_policy_offline(
        cache_dir=args.cache_dir,
        output_dir=out_dir,
        config=train_cfg,
        sft_ckpt_name=args.sft_ckpt_name,
        ref_ckpt_name=args.ref_ckpt_name,
        show_progress=not args.no_progress,
    )

    # Stage B: Real end-to-end detection verification strictly on source_dev (clean + degraded)
    e2e_res = None
    if not args.skip_e2e_eval:
        split_cfg = sft_res["cache_split_config"]
        cache_manifest_path = Path(args.cache_dir) / "cache_manifest.json"
        cache_doc = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
        deg_seed = int(cache_doc["manifest"]["degradation_recipe"]["base_seed"])

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
        # Verify exact split alignment with offline cache
        assert split_info.selected_dev_sha256 == split_cfg["selected_dev_sha256"], (
            "source_dev split SHA256 mismatch between live loader and offline cache manifest!"
        )

        print(
            f"\n[Phase 2 E2E Eval] Loading frozen StyleDetV6 + AZAgentDetector ({args.sft_ckpt_name}) "
            f"on source_dev subset ({min(args.eval_dev_images, len(split_info.selected_dev_ids))} images x 2 views)...",
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
        e2e_res = evaluate_sft_e2e_on_source_dev(
            policy_sft_path=out_dir / args.sft_ckpt_name,
            wrapper=wrapper,
            loader=loader,
            dev_image_ids=split_info.selected_dev_ids,
            degradation_seed=deg_seed,
            max_eval_images=args.eval_dev_images,
            trunc_margin_px=args.trunc_margin_px,
            max_detections=args.max_detections,
            show_progress=not args.no_progress,
        )

    # Read-only boundary verification
    rn_snap_after = snapshot_readonly_tree(rn_root, (".py", ".pyc", ".pth", ".json"))
    sky_pkl_after = snapshot_readonly_tree(sky_root, (".pkl",))
    readonly_audit = {
        "textstyle_rn_mutated_or_added": sorted(
            k for k in set(rn_snap_before) | set(rn_snap_after) if rn_snap_before.get(k) != rn_snap_after.get(k)
        ),
        "sky_pkl_mutated_or_added": sorted(
            k for k in set(sky_pkl_before) | set(sky_pkl_after) if sky_pkl_before.get(k) != sky_pkl_after.get(k)
        ),
    }
    if readonly_audit["textstyle_rn_mutated_or_added"] or readonly_audit["sky_pkl_mutated_or_added"]:
        raise RuntimeError(f"Read-only boundary violation detected: {readonly_audit}")

    results = {
        "schema_version": 1,
        "readonly_boundary_audit": readonly_audit,
        "sft_training": sft_res,
        "source_dev_e2e_eval": e2e_res,
    }

    json_path = out_dir / f"{args.report_prefix}_results.json"
    md_path = out_dir / f"{args.report_prefix}_report.md"
    atomic_write_text(json_path, json.dumps(results, indent=2, ensure_ascii=False))
    generate_phase2_markdown_report(results, md_path)

    print("\n================ PHASE 2 SFT COMPLETE ================")
    print(f"  Best Epoch        : {sft_res['best_epoch']} (dev_loss={sft_res['best_source_dev']['total_loss']:.4f})")
    print(f"  Policy SFT Ckpt   : {sft_res['checkpoint_export']['policy_sft_path']}")
    print(f"  Policy Ref Ckpt   : {sft_res['checkpoint_export']['policy_sft_ref_path']}")
    print(
        f"  Ref Consistency   : bitwise={sft_res['checkpoint_export']['bitwise_identical_files']} | "
        f"max_logit_diff={sft_res['checkpoint_export']['reload_max_logit_diff']:.1e} | "
        f"D_KL={sft_res['checkpoint_export']['reload_max_kl_divergence']:.1e}"
    )
    if e2e_res is not None:
        for vk in ("source_dev_clean", "source_dev_degraded"):
            vm = e2e_res["views"][vk]["metrics"]
            vb = e2e_res["views"][vk]["budget"]
            print(f"\n  [{vk}] ({e2e_res['views'][vk]['num_images']} imgs)")
            for mk in ("baseline", "rpn_heuristic_wbf", "az_agent_sft_greedy", "static_oracle_for_policy_wbf"):
                fps_v = vb[mk].get("measured_e2e_fps", vb[mk].get("projected_policy_fps"))
                print(
                    f"    {mk:30s}: AP50={vm[mk]['AP50']:.2f} | AP75={vm[mk]['AP75']:.2f} | "
                    f"AP={vm[mk]['AP']:.2f} | weak_AP50={vm[mk]['weak_AP50']:.2f} | "
                    f"K={vb[mk]['avg_k']:.2f} | NC={vb[mk]['no_crop_pct']:.1f}% | FPS={fps_v:.2f}"
                )
    print(f"\n  Saved JSON        : {json_path}")
    print(f"  Saved Report      : {md_path}")
    print("  Read-only audit   : PASS (0 mutations in textstyleDet_rn and data/sky)")


if __name__ == "__main__":
    main()
