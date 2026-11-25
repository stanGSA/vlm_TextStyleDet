#!/usr/bin/env python
"""Phase 2 CLI: Build Sharded Source-Only Offline State & SFT Pseudo-Label Cache.

Strictly enforces Single-DGOD compliance:
  - Uses ONLY source domain ``daytime_clear/train`` + deterministic physical synthetic
    degradations (zero target-domain image or annotation access).
  - Uses ``OracleSelector.for_policy(policy)`` (``nms_window_iou=0.35``) and asserts every
    mined trajectory passes ``ActiveZoomPolicyHead.log_prob`` before persistence.
  - Verifies read-only boundary integrity on ``E:\\textstyleDet\\textstyleDet_rn`` and
    ``E:\\textstyleDet\\data\\sky`` before and after execution.
  - Supports safe atomic resume via immutable ``cache_manifest.json`` SHA256 verification.
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
from zoom_agent.sft_cache import (
    DEFAULT_SHARD_SIZE,
    build_sharded_sft_cache,
    compute_completed_seal_digest,
    verify_cache_manifest,
)
from zoom_agent.source_degradation import (
    ALLOWED_SOURCE_DOMAIN,
    ALLOWED_SOURCE_SPLIT,
    SourceOnlyDatasetLoader,
)

DEFAULT_PHASE2_CACHE_DIR = r"E:\textstyleDet\TextStyleDet_vlm\outputs\phase2\sft_cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Phase 2 Source-Only Sharded Offline SFT Cache (FP16 States + Oracle.for_policy Labels)"
    )
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CKPT)
    parser.add_argument("--clip_weights", type=str, default=DEFAULT_CLIP_RN101)
    parser.add_argument("--data_dir", type=str, default=DEFAULT_SKY_DIR)
    parser.add_argument("--cache_dir", type=str, default=DEFAULT_PHASE2_CACHE_DIR)
    parser.add_argument("--domain", type=str, default=ALLOWED_SOURCE_DOMAIN, help="Must be daytime_clear")
    parser.add_argument("--split", type=str, default=ALLOWED_SOURCE_SPLIT, help="Must be train")
    parser.add_argument("--dev_ratio", type=float, default=0.10)
    parser.add_argument("--split_seed", type=int, default=202604)
    parser.add_argument(
        "--max_train_images",
        type=int,
        default=4000,
        help="Number of source_train images (0 = full ~17,455 split)",
    )
    parser.add_argument(
        "--max_dev_images",
        type=int,
        default=500,
        help="Number of source_dev images (0 = full ~1,940 split)",
    )
    parser.add_argument(
        "--degraded_views_per_image",
        type=int,
        default=1,
        help="Number of deterministic physical degraded views per source image (in addition to clean)",
    )
    parser.add_argument("--no_clean", action="store_true", help="Disable clean pass-through view")
    parser.add_argument("--degradation_seed", type=int, default=20260420)
    parser.add_argument("--shard_size", type=int, default=DEFAULT_SHARD_SIZE)
    parser.add_argument("--k_max", type=int, default=2)
    parser.add_argument("--nms_window_iou", type=float, default=0.35)
    parser.add_argument("--roi_chunk_size", type=int, default=256)
    parser.add_argument("--crop_max_proposals", type=int, default=300)
    parser.add_argument("--use_bf16", action="store_true", help="Default False (standard audited FP32)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace_root = Path(__file__).resolve().parent
    rn_root = Path(r"E:\textstyleDet\textstyleDet_rn")
    sky_root = Path(args.data_dir)

    # Hard Single-DGOD source-only validation before loading model or touching dataset
    loader = SourceOnlyDatasetLoader(
        data_dir=args.data_dir,
        domain=args.domain,
        split=args.split,
        dev_ratio=args.dev_ratio,
        split_seed=args.split_seed,
        rn_root=rn_root,
    )
    split_info = loader.build_splits(
        max_train_images=args.max_train_images,
        max_dev_images=args.max_dev_images,
    )

    rn_snap_before = snapshot_readonly_tree(rn_root, (".py", ".pyc", ".pth", ".json"))
    sky_pkl_before = snapshot_readonly_tree(sky_root, (".pkl",))

    views_per_img = (0 if args.no_clean else 1) + args.degraded_views_per_image
    print(
        f"[Phase 2 Cache] Source-only split locked: {split_info.domain}/{split_info.split} "
        f"(full={split_info.total_valid_images}: train={len(split_info.full_source_train_ids)}, "
        f"dev={len(split_info.full_source_dev_ids)})",
        flush=True,
    )
    print(
        f"[Phase 2 Cache] Selected images: source_train={len(split_info.selected_train_ids)} "
        f"({len(split_info.selected_train_ids) * views_per_img} views) | "
        f"source_dev={len(split_info.selected_dev_ids)} "
        f"({len(split_info.selected_dev_ids) * views_per_img} views) | "
        f"shard_size={args.shard_size}",
        flush=True,
    )
    print(
        f"[Phase 2 Cache] Split SHA256: train={split_info.selected_train_sha256[:16]}... "
        f"| dev={split_info.selected_dev_sha256[:16]}...",
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

    manifest_doc = build_sharded_sft_cache(
        wrapper=wrapper,
        loader=loader,
        split_info=split_info,
        cache_dir=args.cache_dir,
        workspace_root=workspace_root,
        shard_size=args.shard_size,
        base_seed=args.degradation_seed,
        include_clean=not args.no_clean,
        degraded_views_per_image=args.degraded_views_per_image,
        k_max=args.k_max,
        nms_window_iou=args.nms_window_iou,
        show_progress=not args.no_progress,
    )

    # Post-run read-only integrity check
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

    # Persist readonly_audit & seal completed_seal_digest atomically into cache_manifest.json
    manifest_doc["readonly_boundary_audit"] = readonly_audit
    manifest_doc["completed_seal_digest"] = compute_completed_seal_digest(
        manifest_digest=manifest_doc["manifest_digest"],
        shards=manifest_doc["shards"],
        readonly_boundary_audit=readonly_audit,
    )
    atomic_write_text(
        Path(args.cache_dir) / "cache_manifest.json",
        json.dumps(manifest_doc, indent=2, ensure_ascii=False),
    )
    verify_cache_manifest(
        cache_dir=args.cache_dir,
        expected_manifest_digest=manifest_doc["manifest_digest"],
        verify_shard_hashes=True,
        require_completed=True,
        require_readonly_audit=True,
        require_completed_seal=True,
    )

    st_tr = manifest_doc["stats"]["source_train"]
    st_dv = manifest_doc["stats"]["source_dev"]
    print("\n================ PHASE 2 OFFLINE SFT CACHE SUMMARY ================")
    print(f"  Cache dir       : {Path(args.cache_dir).resolve()}")
    print(f"  Manifest SHA256 : {manifest_doc['manifest_digest']}")
    print(
        f"  Total samples   : {manifest_doc['stats']['total_samples']} "
        f"(train={st_tr['num_samples']}, dev={st_dv['num_samples']}) | "
        f"Disk={manifest_doc['stats']['total_size_gib']:.3f} GiB ({manifest_doc['stats']['total_size_mib']:.1f} MiB)"
    )
    print(
        f"  Timing / VRAM   : elapsed={manifest_doc['timing']['total_elapsed_sec']:.1f}s "
        f"({manifest_doc['timing']['mean_sec_per_view']:.3f}s/view) | "
        f"Alloc={manifest_doc['gpu_memory_gib']['peak_allocated_gib']:.3f} GiB | "
        f"Reserved={manifest_doc['gpu_memory_gib']['peak_reserved_gib']:.3f} GiB"
    )
    print(
        f"  Train Oracle K  : K=0:{st_tr['k_pct']['0']:.1f}% / K=1:{st_tr['k_pct']['1']:.1f}% / "
        f"K=2:{st_tr['k_pct']['2']:.1f}% (avg_K={st_tr['avg_k']:.3f}, "
        f"clean_NoCrop={st_tr['clean_no_crop_pct']:.1f}%, deg_NoCrop={st_tr['degraded_no_crop_pct']:.1f}%, "
        f"horizon={st_tr['horizon_window_share_pct']:.1f}%)"
    )
    print(
        f"  Dev Oracle K    : K=0:{st_dv['k_pct']['0']:.1f}% / K=1:{st_dv['k_pct']['1']:.1f}% / "
        f"K=2:{st_dv['k_pct']['2']:.1f}% (avg_K={st_dv['avg_k']:.3f}, "
        f"clean_NoCrop={st_dv['clean_no_crop_pct']:.1f}%, deg_NoCrop={st_dv['degraded_no_crop_pct']:.1f}%, "
        f"horizon={st_dv['horizon_window_share_pct']:.1f}%)"
    )
    print(f"  Train views     : {st_tr['degradation_counts']}")
    print(f"  Dev views       : {st_dv['degradation_counts']}")
    print("  Read-only audit : PASS (0 mutations in textstyleDet_rn and data/sky)")


if __name__ == "__main__":
    main()
