#!/usr/bin/env python
"""Phase 2 Automated Unit & Acceptance Verification Suite.

Verifies all Phase 2 requirements and Single-DGOD audit invariants:
  1. Single-DGOD Hard Isolation: passing ANY target domain or non-train split
     (including ``daytime_clear/test``) to ``SourceOnlyDatasetLoader`` MUST raise
     ``ValueError``; ``source_train`` and ``source_dev`` are strictly disjoint and
     reproducible by SHA256.
  2. Physical Degradation Determinism & Geometric Alignment: identical
     ``(image_id, view_tag, seed)`` produces bitwise-identical RGB pixels across
     all 5 operators, exact ``(orig_w, orig_h)`` preservation, valid ``[0, 255]``
     uint8 range, and distinct outputs across seeds/tags.
  3. Cache Manifest Tamper Rejection & ``OracleSelector.for_policy`` Legality:
     tampering with any manifest field, digest, domain, NMS rule, or shard file
     MUST raise ``RuntimeError``; sampled cached records from both ``source_train``
     and ``source_dev`` restore to non-inference, non-requires_grad FP32
     ``AgentBatchState`` tensors and 100% pass ``policy.log_prob``.
  4. SFT Checkpoint & ``pi_ref`` Consistency: ``ActiveZoomPolicyHead.from_checkpoint``
     on ``policy_sft.pth`` and ``policy_sft_ref.pth`` rebuild identically with
     bitwise-equal forward logits and ``D_KL(pi_sft || pi_ref) == 0.0`` on real
     source_dev inputs; ``use_film=False`` ablation backward verified.
  5. Read-Only Boundary Integrity: zero ``.py/.pyc/.pth/.json`` mutations in
     ``E:\\textstyleDet\\textstyleDet_rn`` and zero ``.pkl`` mutations in
     ``E:\\textstyleDet\\data\\sky``.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from run_phase0 import DEFAULT_CKPT, DEFAULT_CLIP_RN101, DEFAULT_SKY_DIR, snapshot_readonly_tree
from zoom_agent import (
    ADVERSE_DEGRADATION_TAGS,
    AZAgentDetector,
    ActiveZoomPolicyHead,
    AgentBatchState,
    DEGRADATION_TAGS,
    FrozenTextStyleDetWrapper,
    ShardedSFTCacheReader,
    SourceOnlyDatasetLoader,
    apply_physical_degradation,
    validate_source_only_access,
    verify_cache_manifest,
)
from zoom_agent.eval_utils import sha256_file, sha256_json
from zoom_agent.sft_cache import collate_sft_samples, compute_completed_seal_digest
from zoom_agent.sft_trainer import build_policy_from_cache_manifest

DEFAULT_CACHE_DIR = Path(r"E:\textstyleDet\TextStyleDet_vlm\outputs\phase2\sft_cache")
DEFAULT_SFT_CKPT = Path(r"E:\textstyleDet\TextStyleDet_vlm\outputs\phase2\policy_sft.pth")
DEFAULT_REF_CKPT = Path(r"E:\textstyleDet\TextStyleDet_vlm\outputs\phase2\policy_sft_ref.pth")


def test_1_single_dgod_hard_isolation() -> SourceOnlyDatasetLoader:
    print("\n" + "=" * 78)
    print("[TEST 1] Single-DGOD Source-Only Hard Isolation & Disjoint Split Fingerprints")
    print("=" * 78)

    forbidden_domains = [
        "daytime_foggy",
        "dusk_rainy",
        "Night-Sunny",
        "night_rainy",
        "Daytime_Clear",
        "../daytime_foggy",
        "",
    ]
    for bad_dom in forbidden_domains:
        raised = False
        try:
            _ = SourceOnlyDatasetLoader(DEFAULT_SKY_DIR, domain=bad_dom, split="train")
        except ValueError:
            raised = True
        assert raised, f"SourceOnlyDatasetLoader MUST reject forbidden domain={bad_dom!r} with ValueError!"

        raised_v = False
        try:
            validate_source_only_access(bad_dom, "train")
        except ValueError:
            raised_v = True
        assert raised_v, f"validate_source_only_access MUST reject domain={bad_dom!r}!"

    forbidden_splits = ["test", "val", "trainval", "TRAIN", ""]
    for bad_split in forbidden_splits:
        raised = False
        try:
            _ = SourceOnlyDatasetLoader(DEFAULT_SKY_DIR, domain="daytime_clear", split=bad_split)
        except ValueError:
            raised = True
        assert raised, f"SourceOnlyDatasetLoader MUST reject daytime_clear/{bad_split!r} with ValueError!"

    loader_a = SourceOnlyDatasetLoader(DEFAULT_SKY_DIR, domain="daytime_clear", split="train", dev_ratio=0.10, split_seed=202604)
    loader_b = SourceOnlyDatasetLoader(DEFAULT_SKY_DIR, domain="daytime_clear", split="train", dev_ratio=0.10, split_seed=202604)
    split_a = loader_a.build_splits(max_train_images=0, max_dev_images=0)
    split_b = loader_b.build_splits(max_train_images=0, max_dev_images=0)

    assert split_a.total_valid_images == 19395, f"Expected 19395 source train images, got {split_a.total_valid_images}"
    assert len(split_a.full_source_train_ids) + len(split_a.full_source_dev_ids) == 19395
    assert set(split_a.full_source_train_ids).isdisjoint(set(split_a.full_source_dev_ids)), (
        "source_train and source_dev overlap!"
    )
    assert split_a.full_source_train_sha256 == split_b.full_source_train_sha256
    assert split_a.full_source_dev_sha256 == split_b.full_source_dev_sha256

    # On-demand XML parsing check (5 images only; zero .pkl)
    sample_ids = list(split_a.full_source_dev_ids[:5])
    recs = loader_a.parse_records_on_demand(sample_ids)
    assert len(recs) == 5 and all(isinstance(v, list) for v in recs.values())

    # P3-01 Path traversal guard on image_id
    for bad_id in ("../daytime_foggy/foo", "..\\evil", "sub/img", "sub\\img", "..", ""):
        img_rej = False
        try:
            _ = loader_a.image_path(bad_id)
        except ValueError:
            img_rej = True
        assert img_rej, f"image_path MUST reject unsafe image_id={bad_id!r}!"
        xml_rej = False
        try:
            _ = loader_a.xml_path(bad_id)
        except ValueError:
            xml_rej = True
        assert xml_rej, f"xml_path MUST reject unsafe image_id={bad_id!r}!"

    print(
        f"[PASS] TEST 1: All target domains, non-train splits & traversal IDs rejected with ValueError; "
        f"full train={len(split_a.full_source_train_ids)} / dev={len(split_a.full_source_dev_ids)} "
        f"strictly disjoint (train_sha={split_a.full_source_train_sha256[:12]}..., "
        f"dev_sha={split_a.full_source_dev_sha256[:12]}...)."
    )
    return loader_a


def test_2_degradation_determinism_and_geometry(loader: SourceOnlyDatasetLoader) -> None:
    print("\n" + "=" * 78)
    print("[TEST 2] Physical Synthetic Degradation Determinism & Exact Geometry Preservation")
    print("=" * 78)

    split_info = loader.build_splits(max_train_images=4, max_dev_images=2)
    img_id = split_info.selected_dev_ids[0]
    pil_orig = loader.load_image(img_id)
    orig_w, orig_h = pil_orig.size
    arr_orig = np.asarray(pil_orig, dtype=np.uint8)

    for tag in DEGRADATION_TAGS:
        out_1, meta_1 = apply_physical_degradation(pil_orig, image_id=img_id, view_tag=tag, seed=20260420, return_params=True)
        out_2, meta_2 = apply_physical_degradation(pil_orig, image_id=img_id, view_tag=tag, seed=20260420, return_params=True)
        assert out_1.size == (orig_w, orig_h), f"[{tag}] Size changed: {out_1.size} != {(orig_w, orig_h)}"
        assert out_1.mode == "RGB", f"[{tag}] Mode changed: {out_1.mode}"

        arr_1 = np.asarray(out_1, dtype=np.uint8)
        arr_2 = np.asarray(out_2, dtype=np.uint8)
        assert arr_1.dtype == np.uint8 and arr_1.shape == (orig_h, orig_w, 3)
        assert int(arr_1.min()) >= 0 and int(arr_1.max()) <= 255
        assert np.array_equal(arr_1, arr_2), f"[{tag}] Non-deterministic output for identical (image_id, tag, seed)!"
        assert meta_1 == meta_2, f"[{tag}] Non-deterministic parameter metadata!"

        if tag == "clean":
            assert np.array_equal(arr_1, arr_orig), "clean view must be bitwise identical to source image!"
        else:
            assert not np.array_equal(arr_1, arr_orig), f"[{tag}] Adverse degradation did not modify image!"
            out_diff_seed = apply_physical_degradation(pil_orig, image_id=img_id, view_tag=tag, seed=999999)
            assert not np.array_equal(arr_1, np.asarray(out_diff_seed, dtype=np.uint8)), (
                f"[{tag}] Different seed produced identical pixels!"
            )

    # Verify low_light darkens average brightness & compound_adverse covers both physical sub-modes
    ll_arr = np.asarray(apply_physical_degradation(pil_orig, image_id=img_id, view_tag="low_light", seed=20260420), dtype=np.float32)
    assert float(ll_arr.mean()) < float(arr_orig.astype(np.float32).mean()), "low_light should reduce mean luminance!"

    seen_submodes = set()
    for s in range(20):
        _, meta_c = apply_physical_degradation(pil_orig, image_id=f"probe_{s}", view_tag="compound_adverse", seed=20260420, return_params=True)
        seen_submodes.add(meta_c["params"]["sub_mode"])
    assert seen_submodes == {"low_light_rain", "fog_low_contrast"}, f"Missing compound submodes: {seen_submodes}"

    # Invalid tag rejected
    bad_tag_raised = False
    try:
        _ = apply_physical_degradation(pil_orig, image_id=img_id, view_tag="snow_invalid", seed=1)
    except ValueError:
        bad_tag_raised = True
    assert bad_tag_raised, "Unknown degradation tag must raise ValueError!"

    print(
        f"[PASS] TEST 2: All 5 degradation tags bitwise deterministic, size={(orig_w, orig_h)} preserved, "
        f"uint8 [0, 255] valid, compound sub-modes={sorted(seen_submodes)} verified."
    )


def test_3_cache_manifest_tamper_and_oracle_legality(cache_dir: Path, device: torch.device) -> None:
    print("\n" + "=" * 78)
    print("[TEST 3] Cache Manifest Tamper Rejection, State Contract & Oracle.for_policy Legality")
    print("=" * 78)

    assert (cache_dir / "cache_manifest.json").is_file(), f"Cache manifest missing: {cache_dir / 'cache_manifest.json'}"

    # 3a. Verify real cache integrity (manifest digest + completeness + all shard SHA256 + readonly audit + seal)
    doc = verify_cache_manifest(
        cache_dir,
        verify_shard_hashes=True,
        require_completed=True,
        require_readonly_audit=True,
        require_completed_seal=True,
    )
    manifest = doc["manifest"]
    assert manifest["source_domain"] == "daytime_clear" and manifest["source_split"] == "train"
    assert float(manifest["oracle_selector"]["nms_window_iou"]) == 0.35
    assert float(manifest["policy_action_contract"]["nms_window_iou"]) == 0.35

    # 3b. Test manifest / seal / completeness / readonly-audit tamper rejection in isolated temp directory
    with tempfile.TemporaryDirectory(prefix="az_p2_tamper_") as tmp:
        tdir = Path(tmp)
        first_dev_entry = doc["shards"]["source_dev"][0]
        first_dev_rel = first_dev_entry["file"]
        src_shard = cache_dir / first_dev_rel
        dst_shard = tdir / first_dev_rel
        dst_shard.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_shard, dst_shard)

        # Case 0 (P1-02): Deleted shards while keeping completed=True & manifest_digest unchanged -> RuntimeError
        d_deleted = json.loads((cache_dir / "cache_manifest.json").read_text(encoding="utf-8"))
        d_deleted["shards"] = {"source_train": [], "source_dev": [first_dev_entry]}
        d_deleted.pop("completed_seal_digest", None)
        (tdir / "cache_manifest.json").write_text(json.dumps(d_deleted), encoding="utf-8")
        raised = False
        try:
            verify_cache_manifest(tdir, verify_shard_hashes=False, require_completed=True)
        except RuntimeError:
            raised = True
        assert raised, "[P1-02] Deleting shards under completed=True MUST raise RuntimeError!"

        # Build a valid self-consistent sealed 1-shard synthetic manifest matching first_dev_entry
        deg_cfg = d_deleted["manifest"]["degradation_recipe"]
        vpi = (1 if bool(deg_cfg.get("include_clean", True)) else 0) + int(deg_cfg.get("degraded_views_per_image", 0))
        base_doc = json.loads(json.dumps(d_deleted))
        base_doc["manifest"]["split_config"]["selected_train_count"] = 0
        base_doc["manifest"]["split_config"]["selected_dev_count"] = int(first_dev_entry["num_samples"]) // max(vpi, 1)
        base_doc["manifest_digest"] = sha256_json(base_doc["manifest"])
        base_doc["readonly_boundary_audit"] = {
            "textstyle_rn_mutated_or_added": [],
            "sky_pkl_mutated_or_added": [],
        }
        base_doc["completed_seal_digest"] = compute_completed_seal_digest(
            manifest_digest=base_doc["manifest_digest"],
            shards=base_doc["shards"],
            readonly_boundary_audit=base_doc["readonly_boundary_audit"],
        )
        base_digest = base_doc["manifest_digest"]
        (tdir / "cache_manifest.json").write_text(json.dumps(base_doc), encoding="utf-8")
        # Baseline clean sealed copy passes all strict checks
        verify_cache_manifest(
            tdir,
            expected_manifest_digest=base_digest,
            verify_shard_hashes=True,
            require_completed=True,
            require_readonly_audit=True,
            require_completed_seal=True,
        )

        # Case 1: Tamper manifest field without updating digest -> RuntimeError
        d1 = json.loads(json.dumps(base_doc))
        d1["manifest"]["oracle_selector"]["nms_window_iou"] = 0.45
        (tdir / "cache_manifest.json").write_text(json.dumps(d1), encoding="utf-8")
        raised = False
        try:
            verify_cache_manifest(tdir, verify_shard_hashes=False)
        except RuntimeError:
            raised = True
        assert raised, "Tampered manifest field (nms_window_iou) MUST raise RuntimeError!"

        # Case 2: Tamper source_domain AND forge digest -> MUST still raise RuntimeError!
        d2 = json.loads(json.dumps(base_doc))
        d2["manifest"]["source_domain"] = "daytime_foggy"
        d2["manifest_digest"] = sha256_json(d2["manifest"])
        (tdir / "cache_manifest.json").write_text(json.dumps(d2), encoding="utf-8")
        raised = False
        try:
            verify_cache_manifest(tdir, verify_shard_hashes=False)
        except RuntimeError:
            raised = True
        assert raised, "Forged target-domain manifest MUST raise RuntimeError!"

        # Case 3: Resume with mismatched expected_manifest_digest -> RuntimeError
        (tdir / "cache_manifest.json").write_text(json.dumps(base_doc), encoding="utf-8")
        raised = False
        try:
            verify_cache_manifest(tdir, expected_manifest_digest="0" * 64, verify_shard_hashes=False)
        except RuntimeError:
            raised = True
        assert raised, "Mismatched expected_manifest_digest MUST raise RuntimeError!"

        # Case 4 (P1-02): Non-empty readonly_boundary_audit -> RuntimeError
        d4 = json.loads(json.dumps(base_doc))
        d4["readonly_boundary_audit"]["textstyle_rn_mutated_or_added"] = ["eval/__pycache__/fake.pyc"]
        (tdir / "cache_manifest.json").write_text(json.dumps(d4), encoding="utf-8")
        raised = False
        try:
            verify_cache_manifest(tdir, verify_shard_hashes=False)
        except RuntimeError:
            raised = True
        assert raised, "[P1-02] Non-empty readonly_boundary_audit MUST raise RuntimeError!"

        # Case 5 (P1-02): Missing completed_seal_digest when required -> RuntimeError
        d5 = json.loads(json.dumps(base_doc))
        d5.pop("completed_seal_digest", None)
        (tdir / "cache_manifest.json").write_text(json.dumps(d5), encoding="utf-8")
        raised = False
        try:
            verify_cache_manifest(tdir, verify_shard_hashes=False, require_completed_seal=True)
        except RuntimeError:
            raised = True
        assert raised, "[P1-02] Missing completed_seal_digest MUST raise RuntimeError when required!"

        # Case 6: Corrupt shard bytes -> RuntimeError
        (tdir / "cache_manifest.json").write_text(json.dumps(base_doc), encoding="utf-8")
        orig_bytes = dst_shard.read_bytes()
        dst_shard.write_bytes(orig_bytes[:-1] + bytes([orig_bytes[-1] ^ 0xFF]))
        raised = False
        try:
            verify_cache_manifest(tdir, verify_shard_hashes=True)
        except RuntimeError:
            raised = True
        assert raised, "Corrupted shard SHA256 MUST raise RuntimeError!"

        # Case 7 (P1-02): Forged shard sha256 in manifest without updating completed_seal_digest -> RuntimeError
        d7 = json.loads(json.dumps(base_doc))
        d7["shards"]["source_dev"][0]["sha256"] = sha256_file(dst_shard)
        (tdir / "cache_manifest.json").write_text(json.dumps(d7), encoding="utf-8")
        raised = False
        try:
            verify_cache_manifest(tdir, verify_shard_hashes=True, require_completed=True, require_completed_seal=True)
        except RuntimeError:
            raised = True
        assert raised, "[P1-02] Forged shard sha256 breaking completed_seal_digest MUST raise RuntimeError!"

    # 3c. Sample real cached records from source_train & source_dev, verify tensor contract & policy.log_prob
    policy = build_policy_from_cache_manifest(manifest, use_film=True, dropout=0.0).to(device).eval()
    checked_total = 0
    for role in ("source_train", "source_dev"):
        reader = ShardedSFTCacheReader(cache_dir, split_role=role, verify_shard_hashes=False, require_completed=True)
        n = len(reader)
        # Deterministic evenly spaced indices up to 64 per split
        step = max(1, n // 64)
        sample_indices = list(range(0, n, step))[:64]
        samples = []
        for idx in sample_indices:
            rec = reader.get_sample(idx, device="cpu")
            st: AgentBatchState = rec["state"]
            assert st.c4_pooled.dtype == torch.float32 and not st.c4_pooled.is_inference() and not st.c4_pooled.requires_grad
            assert st.cue_maps.dtype == torch.float32 and not st.cue_maps.is_inference() and not st.cue_maps.requires_grad
            assert st.style_stats.dtype == torch.float32 and not st.style_stats.is_inference() and not st.style_stats.requires_grad
            assert rec["split_role"] == role
            assert rec["degradation_tag"] in DEGRADATION_TAGS
            assert rec["oracle_window_scores"].shape == (18,)
            assert rec["sft_multi_hot"].shape == (19,)

            act = rec["oracle_action"]
            mhot = rec["sft_multi_hot"]
            if len(act) == 0:
                assert float(mhot[18].item()) == 1.0 and float(mhot[:18].sum().item()) == 0.0
            else:
                assert float(mhot[18].item()) == 0.0
                active_ids = set(torch.where(mhot > 0.5)[0].tolist())
                assert active_ids == set(act), f"sft_multi_hot mismatch: {active_ids} != {set(act)}"
            samples.append(rec)

        batch = collate_sft_samples(samples, device=device)
        b_state: AgentBatchState = batch["state"]
        assert b_state.batch_size == len(samples)
        assert not b_state.c4_pooled.is_inference() and not b_state.c4_pooled.requires_grad
        with torch.no_grad():
            lp = policy.log_prob(b_state, batch["oracle_actions"])
        assert lp.shape == (len(samples),) and bool(torch.isfinite(lp).all().item()), (
            f"Cached oracle_actions failed policy.log_prob on {role}!"
        )
        checked_total += len(samples)

    # Also verify w/o FiLM ablation backward works cleanly on real cached batch
    pol_nofilm = build_policy_from_cache_manifest(manifest, use_film=False, dropout=0.0).to(device).train()
    pol_nofilm.zero_grad(set_to_none=True)
    logits_nf = pol_nofilm(batch["state"])
    loss_nf = F.binary_cross_entropy_with_logits(logits_nf, batch["sft_multi_hot"]) - 0.25 * pol_nofilm.log_prob_from_logits(
        logits_nf, batch["oracle_actions"]
    ).mean()
    loss_nf.backward()
    for p_name, p in pol_nofilm.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and float(p.grad.abs().sum().item()) > 0.0, (
            f"[w/o FiLM] Invalid grad on cached batch for {p_name}"
        )

    print(
        f"[PASS] TEST 3: All 8 manifest/seal/completeness/readonly/shard tamper cases rejected with RuntimeError; "
        f"{checked_total} sampled cached records passed AgentBatchState FP32 grad-safe contract "
        f"& policy.log_prob legality; w/o FiLM backward verified."
    )


def test_4_sft_and_pi_ref_consistency(
    sft_ckpt: Path,
    ref_ckpt: Path,
    cache_dir: Path,
    loader: SourceOnlyDatasetLoader,
    device: torch.device,
) -> None:
    print("\n" + "=" * 78)
    print("[TEST 4] policy_sft.pth & policy_sft_ref.pth Consistency + Real-Input D_KL == 0")
    print("=" * 78)

    assert sft_ckpt.is_file(), f"Missing SFT checkpoint: {sft_ckpt}"
    assert ref_ckpt.is_file(), f"Missing reference checkpoint: {ref_ckpt}"
    sha_sft = sha256_file(sft_ckpt)
    sha_ref = sha256_file(ref_ckpt)
    assert sha_sft == sha_ref, f"Checkpoint SHA256 mismatch: sft={sha_sft} != ref={sha_ref}"

    pol_sft = ActiveZoomPolicyHead.from_checkpoint(sft_ckpt, map_location=device).eval()
    pol_ref = ActiveZoomPolicyHead.from_checkpoint(ref_ckpt, map_location=device).eval()

    assert pol_sft.config_dict() == pol_ref.config_dict(), "policy_config mismatch!"
    assert pol_sft.grid.config_dict() == pol_ref.grid.config_dict(), "grid_config mismatch!"

    for p in pol_ref.parameters():
        p.requires_grad = False
    assert pol_ref.count_parameters(trainable_only=True) == 0
    assert pol_ref.count_parameters(trainable_only=False) == pol_sft.count_parameters(trainable_only=False)
    sd_sft, sd_ref = pol_sft.state_dict(), pol_ref.state_dict()
    assert sd_sft.keys() == sd_ref.keys()
    for k in sd_sft:
        assert torch.equal(sd_sft[k], sd_ref[k]), f"Weight tensor mismatch for {k}"

    # 4a. Check on cached source_dev batch
    dev_reader = ShardedSFTCacheReader(cache_dir, split_role="source_dev", verify_shard_hashes=False, require_completed=True)
    dev_batch = next(dev_reader.iter_batches(batch_size=min(16, len(dev_reader)), shuffle=False, seed=0, device=device))
    with torch.no_grad():
        logits_sft = pol_sft(dev_batch["state"])
        logits_ref = pol_ref(dev_batch["state"])
        assert torch.equal(logits_sft, logits_ref), "Forward logits differ on cached source_dev batch!"
        kl_cached = pol_sft.kl_divergence(dev_batch["state"], pol_ref)
        assert float(kl_cached.abs().max().item()) == 0.0, f"Non-zero D_KL on cached batch: {kl_cached}"
        act_sft = pol_sft.act_greedy_from_logits(logits_sft).actions
        act_ref = pol_ref.act_greedy_from_logits(logits_ref).actions
        assert act_sft == act_ref

    # 4b. P1-01 Phase-3 GRPO Step-0 Exactness Regression
    #     (dropout_override=0.0, as_reference=True, exact_eval_parity=True, KL eval guard)
    pol_grpo = ActiveZoomPolicyHead.from_checkpoint(
        sft_ckpt, map_location=device, dropout_override=0.0, exact_eval_parity=True
    ).train()
    pol_ref_safe = ActiveZoomPolicyHead.from_checkpoint(
        ref_ckpt, map_location=device, as_reference=True, exact_eval_parity=True
    )
    assert pol_grpo.dropout == 0.0 and pol_grpo.training is True
    assert pol_ref_safe.training is False and pol_ref_safe.count_parameters(trainable_only=True) == 0

    # Ref policy left in train() mode MUST be rejected by kl_divergence()
    pol_ref_bad = ActiveZoomPolicyHead.from_checkpoint(ref_ckpt, map_location=device).train()
    kl_guard_raised = False
    try:
        _ = pol_grpo.kl_divergence(dev_batch["state"], pol_ref_bad)
    except RuntimeError:
        kl_guard_raised = True
    assert kl_guard_raised, "[P1-01] kl_divergence MUST reject training=True reference policy!"

    # Verify Step-0 r_i(theta) == 1.0 & D_KL(pi_theta || pi_ref) == 0.0 on real source_dev batch (G=4)
    grpo_batch = next(dev_reader.iter_batches(batch_size=min(32, len(dev_reader)), shuffle=False, seed=0, device=device))
    g_gen = torch.Generator(device=device).manual_seed(2026)
    with torch.no_grad():
        samp_train = pol_grpo.sample_group(grpo_batch["state"], G=4, generator=g_gen)
        old_lp_train = samp_train.log_probs.detach()
        pol_grpo.eval()
        old_lp_eval = pol_grpo.log_prob(grpo_batch["state"], samp_train.actions).detach()
        pol_grpo.train()

    new_lp = pol_grpo.log_prob(grpo_batch["state"], samp_train.actions)
    ratio_same = torch.exp(new_lp - old_lp_train).detach()
    ratio_cross = torch.exp(new_lp - old_lp_eval).detach()
    kl_step0 = pol_grpo.kl_divergence(grpo_batch["state"], pol_ref_safe).detach()
    max_r_same_err = float((ratio_same - 1.0).abs().max().item())
    max_r_cross_err = float((ratio_cross - 1.0).abs().max().item())
    max_kl_step0 = float(kl_step0.abs().max().item())
    clipped_cnt = int(((ratio_cross < 0.8) | (ratio_cross > 1.2)).sum().item())
    assert max_r_same_err < 1e-6, f"[P1-01] Step-0 same-path ratio error {max_r_same_err} >= 1e-6!"
    assert max_r_cross_err < 1e-6, f"[P1-01] Step-0 eval->train ratio error {max_r_cross_err} >= 1e-6!"
    assert clipped_cnt == 0, f"[P1-01] Step-0 premature PPO clipping count {clipped_cnt} != 0!"
    assert max_kl_step0 < 1e-6, f"[P1-01] Step-0 train-vs-ref D_KL {max_kl_step0} >= 1e-6!"

    # 4c. Check on REAL frozen StyleDetV6 perception inputs (source_dev clean + low_light)
    split_info = loader.build_splits(max_train_images=4, max_dev_images=2)
    dev_img_id = split_info.selected_dev_ids[0]
    pil_clean = loader.load_image(dev_img_id)
    pil_dark = apply_physical_degradation(pil_clean, image_id=dev_img_id, view_tag="low_light", seed=20260420)

    wrapper = FrozenTextStyleDetWrapper(
        checkpoint_path=DEFAULT_CKPT,
        clip_roi_weights_path=DEFAULT_CLIP_RN101,
        device=device,
        use_amp_bf16=False,
        roi_chunk_size=256,
        crop_max_proposals=300,
    )
    agent = AZAgentDetector(detector_wrapper=wrapper, policy_head=pol_sft)
    _, real_state = agent.extract_batch_state([pil_clean, pil_dark], image_ids=[f"{dev_img_id}_clean", f"{dev_img_id}_dark"])

    with torch.no_grad():
        rl_sft = pol_sft(real_state)
        rl_ref = pol_ref(real_state)
        assert torch.equal(rl_sft, rl_ref), "Forward logits differ on live StyleDetV6 real source_dev inputs!"
        kl_real = pol_sft.kl_divergence(real_state, pol_ref)
        assert float(kl_real.abs().max().item()) == 0.0, f"Non-zero D_KL on real inputs: {kl_real}"

    # Verify live AZAgentDetector.predict runs end-to-end with loaded policy_sft.pth
    pred_clean = agent.predict(pil_clean, mode="greedy", sync_cuda_for_timing=True)
    pred_dark = agent.predict(pil_dark, mode="greedy", sync_cuda_for_timing=True)
    assert len(pred_clean.action) <= 2 and len(pred_dark.action) <= 2
    assert pred_clean.detections["boxes"].shape[0] <= agent.fuser.max_detections

    print(
        f"[PASS] TEST 4: policy_sft.pth == policy_sft_ref.pth (SHA256={sha_sft[:16]}...), "
        f"rebuilt & forward bitwise equal; D_KL == 0.0 on cached & live StyleDetV6 inputs; "
        f"P1-01 GRPO Step-0 verified (max|r_same-1|={max_r_same_err:.2e}, max|r_eval2train-1|={max_r_cross_err:.2e}, "
        f"clipped=0, max_KL={max_kl_step0:.2e}; live actions: clean={pred_clean.action}, low_light={pred_dark.action})."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 Automated Verification Suite")
    parser.add_argument("--cache_dir", type=str, default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--sft_ckpt", type=str, default=str(DEFAULT_SFT_CKPT))
    parser.add_argument("--ref_ckpt", type=str, default=str(DEFAULT_REF_CKPT))
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rn_root = Path(r"E:\textstyleDet\textstyleDet_rn")
    sky_root = Path(DEFAULT_SKY_DIR)

    rn_before = snapshot_readonly_tree(rn_root, (".py", ".pyc", ".pth", ".json"))
    sky_before = snapshot_readonly_tree(sky_root, (".pkl",))

    print(f"[Phase 2 Test Suite] Running on device={device}, sys.dont_write_bytecode={sys.dont_write_bytecode}")
    loader = test_1_single_dgod_hard_isolation()
    test_2_degradation_determinism_and_geometry(loader)
    test_3_cache_manifest_tamper_and_oracle_legality(Path(args.cache_dir), device=device)
    test_4_sft_and_pi_ref_consistency(
        sft_ckpt=Path(args.sft_ckpt),
        ref_ckpt=Path(args.ref_ckpt),
        cache_dir=Path(args.cache_dir),
        loader=loader,
        device=device,
    )

    rn_after = snapshot_readonly_tree(rn_root, (".py", ".pyc", ".pth", ".json"))
    sky_after = snapshot_readonly_tree(sky_root, (".pkl",))
    rn_diff = sorted(k for k in set(rn_before) | set(rn_after) if rn_before.get(k) != rn_after.get(k))
    sky_diff = sorted(k for k in set(sky_before) | set(sky_after) if sky_before.get(k) != sky_after.get(k))
    assert not rn_diff, f"Read-only violation in textstyleDet_rn: {rn_diff}"
    assert not sky_diff, f"Read-only violation in data/sky: {sky_diff}"

    print("\n" + "=" * 78)
    print("[PASS] TEST 5: Read-only boundary verified (0 mutations in textstyleDet_rn & data/sky).")
    print("ALL PHASE 2 VERIFICATION TESTS PASSED.")
    print("=" * 78)


if __name__ == "__main__":
    main()
