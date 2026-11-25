"""Sharded Offline State & SFT Pseudo-Label Cache (Module 5 Stage-1, Audit P0-02 Hardened).

Design guarantees:
  - Strictly decoupled from training: frozen perception runs once offline on source-only
    ``daytime_clear/train`` (+ deterministic physical synthetic degradations) and writes
    compact FP16 state tensors + FP32 style stats + ``OracleSelector.for_policy`` pseudo-labels.
  - Strict action-space alignment: uses ``OracleSelector.for_policy(policy)``
    (``nms_window_iou=0.35``, ``k_max=2``) and validates EVERY mined trajectory against
    ``policy.log_prob_from_logits`` prior to writing.
  - Atomic shard persistence (``tmp -> os.replace``) with cryptographic ``cache_manifest.json``
    incorporating detector/CLIP weight SHA256, CandidateGrid/StateEncoder/OracleSelector/
    degradation configs, split SHA256 fingerprints, shard SHA256 digests, and code hashes.
  - Refuses to resume or read if ``cache_manifest.json`` is tampered or mismatched
    (raises ``RuntimeError``).
  - Reader restores every sample via ``AgentBatchState.from_cache_dict`` (guaranteed
    non-inference-mode, non-requires_grad FP32 tensors) and batches via
    ``AgentBatchState.collate``.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json
import os
import platform
import shutil
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torchvision
from torch import Tensor
from tqdm import tqdm

from .candidate_grid import CandidateGrid
from .detector_wrapper import FrozenTextStyleDetWrapper
from .eval_utils import atomic_write_text, record_to_0based_gt, sha256_file, sha256_json
from .policy_head import ActionTrajectory, ActiveZoomPolicyHead
from .selectors import OracleSelector
from .source_degradation import (
    ADVERSE_DEGRADATION_TAGS,
    ALLOWED_SOURCE_DOMAIN,
    ALLOWED_SOURCE_SPLIT,
    DEGRADATION_TAGS,
    SourceOnlyDatasetLoader,
    SourceSplitInfo,
    apply_physical_degradation,
    degradation_recipe_config,
    select_adverse_tag_for_view,
    validate_source_only_access,
)
from .state_encoder import AgentBatchState, StateEncoder

CACHE_SCHEMA_VERSION: int = 1
DEFAULT_SHARD_SIZE: int = 512
ESTIMATED_BYTES_PER_SAMPLE: int = 530_000  # ~518 KiB safety margin (actual ~493-505 KiB)
MIN_FREE_DISK_HEADROOM_BYTES: int = 2 * (1024**3)  # 2 GiB safety margin


@dataclass(frozen=True)
class PlannedViewItem:
    """One deterministic (image_id, split_role, degradation_tag, degradation_seed) cache item."""

    image_id: str
    split_role: str  # "source_train" or "source_dev"
    degradation_tag: str
    degradation_seed: int


def atomic_torch_save(payload: Dict, target_path: Path) -> None:
    """Write a PyTorch file atomically via temporary file + ``os.replace``."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_name(f"{target_path.name}.tmp.{os.getpid()}")
    try:
        torch.save(payload, str(tmp_path))
        os.replace(tmp_path, target_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def build_view_plan(
    image_ids: Sequence[str],
    split_role: str,
    base_seed: int = 20260420,
    include_clean: bool = True,
    degraded_views_per_image: int = 1,
) -> List[PlannedViewItem]:
    """Build deterministic ordered list of (clean + degraded) views for a source split."""
    if split_role not in {"source_train", "source_dev"}:
        raise ValueError(f"split_role must be 'source_train' or 'source_dev', got {split_role!r}")
    if degraded_views_per_image < 0:
        raise ValueError(f"degraded_views_per_image must be >= 0, got {degraded_views_per_image}")
    if not include_clean and degraded_views_per_image == 0:
        raise ValueError("At least one of include_clean or degraded_views_per_image > 0 must be enabled.")

    plan: List[PlannedViewItem] = []
    n_adverse = len(ADVERSE_DEGRADATION_TAGS)
    for img_id in image_ids:
        if include_clean:
            plan.append(
                PlannedViewItem(
                    image_id=img_id,
                    split_role=split_role,
                    degradation_tag="clean",
                    degradation_seed=int(base_seed),
                )
            )
        for v_idx in range(degraded_views_per_image):
            tag = select_adverse_tag_for_view(
                image_id=img_id,
                view_index=v_idx,
                base_seed=base_seed,
            )
            # For v_idx < 4 each adverse tag on the same image is unique so base_seed suffices;
            # if >4 degraded views are ever requested, offset seed by cycle index for uniqueness.
            v_seed = int(base_seed) + (v_idx // n_adverse)
            plan.append(
                PlannedViewItem(
                    image_id=img_id,
                    split_role=split_role,
                    degradation_tag=tag,
                    degradation_seed=v_seed,
                )
            )
    return plan


def check_disk_space_for_cache(cache_dir: Path, total_views: int) -> Dict:
    """Verify sufficient free disk space on target drive before building offline cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(str(cache_dir))
    estimated_bytes = int(total_views) * ESTIMATED_BYTES_PER_SAMPLE
    required_bytes = estimated_bytes + MIN_FREE_DISK_HEADROOM_BYTES
    info = {
        "target_path": str(cache_dir.resolve()),
        "free_bytes": int(usage.free),
        "free_gib": round(usage.free / (1024**3), 2),
        "estimated_cache_bytes": estimated_bytes,
        "estimated_cache_gib": round(estimated_bytes / (1024**3), 3),
        "required_with_headroom_gib": round(required_bytes / (1024**3), 3),
    }
    if usage.free < required_bytes:
        raise RuntimeError(
            f"Insufficient disk space at {cache_dir}: free={info['free_gib']:.2f} GiB < "
            f"required={info['required_with_headroom_gib']:.2f} GiB "
            f"(estimated cache={info['estimated_cache_gib']:.2f} GiB + 2.0 GiB headroom)."
        )
    return info


def build_sft_cache_manifest_payload(
    wrapper: FrozenTextStyleDetWrapper,
    grid: CandidateGrid,
    state_encoder: StateEncoder,
    policy_contract: ActiveZoomPolicyHead,
    oracle_selector: OracleSelector,
    split_info: SourceSplitInfo,
    degradation_cfg: Dict,
    shard_size: int,
    workspace_root: Path,
) -> Dict:
    """Construct canonical immutable manifest dictionary hashed to produce ``manifest_digest``."""
    validate_source_only_access(split_info.domain, split_info.split)
    if oracle_selector.nms_window_iou != policy_contract.nms_window_iou:
        raise RuntimeError(
            f"Action-space mismatch: oracle.nms_window_iou={oracle_selector.nms_window_iou} "
            f"!= policy.nms_window_iou={policy_contract.nms_window_iou}. "
            "Must construct via OracleSelector.for_policy(policy)."
        )
    if oracle_selector.k_max != policy_contract.k_max:
        raise RuntimeError(
            f"Budget mismatch: oracle.k_max={oracle_selector.k_max} != policy.k_max={policy_contract.k_max}."
        )

    code_paths = [
        workspace_root / "build_phase2_cache.py",
        workspace_root / "zoom_agent" / "candidate_grid.py",
        workspace_root / "zoom_agent" / "detector_wrapper.py",
        workspace_root / "zoom_agent" / "eval_utils.py",
        workspace_root / "zoom_agent" / "policy_head.py",
        workspace_root / "zoom_agent" / "selectors.py",
        workspace_root / "zoom_agent" / "source_degradation.py",
        workspace_root / "zoom_agent" / "state_encoder.py",
        workspace_root / "zoom_agent" / "sft_cache.py",
    ]
    code_hashes = {
        str(p.relative_to(workspace_root)).replace("\\", "/"): sha256_file(p)
        for p in code_paths
        if p.is_file()
    }

    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "source_domain": split_info.domain,
        "source_split": split_info.split,
        "checkpoint_path": str(wrapper.checkpoint_path),
        "checkpoint_sha256": sha256_file(wrapper.checkpoint_path),
        "clip_roi_weights_path": str(wrapper.clip_roi_weights_path),
        "clip_roi_weights_sha256": sha256_file(wrapper.clip_roi_weights_path),
        "detector_config": wrapper.config_dict(),
        "candidate_grid": grid.config_dict(),
        "state_encoder": state_encoder.config_dict(),
        "policy_action_contract": {
            "num_windows": policy_contract.num_windows,
            "no_crop_action": policy_contract.no_crop_action,
            "num_actions": policy_contract.num_actions,
            "k_max": policy_contract.k_max,
            "nms_window_iou": policy_contract.nms_window_iou,
        },
        "oracle_selector": oracle_selector.config_dict(),
        "degradation_recipe": degradation_cfg,
        "split_config": split_info.config_dict(),
        "shard_size": int(shard_size),
        "storage_dtype": "float16",
        "style_storage_dtype": "float32",
        "coordinate_system": "orig_0based_xyxy",
        "code_sha256": code_hashes,
    }


def compute_completed_seal_digest(
    manifest_digest: str,
    shards: Dict[str, Sequence[Dict]],
    readonly_boundary_audit: Optional[Dict] = None,
) -> str:
    """Compute top-level tamper-evident seal over ``manifest_digest``, all shards, and readonly audit."""
    canonical_shards: Dict[str, List[Dict]] = {}
    for role in ("source_train", "source_dev"):
        canonical_shards[role] = [
            {
                "file": str(e["file"]),
                "shard_index": int(e["shard_index"]),
                "num_samples": int(e["num_samples"]),
                "size_bytes": int(e["size_bytes"]),
                "sha256": str(e["sha256"]),
            }
            for e in shards.get(role, [])
        ]
    payload: Dict = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "manifest_digest": str(manifest_digest),
        "shards": canonical_shards,
    }
    if readonly_boundary_audit is not None:
        payload["readonly_boundary_audit"] = {
            "textstyle_rn_mutated_or_added": list(readonly_boundary_audit.get("textstyle_rn_mutated_or_added", [])),
            "sky_pkl_mutated_or_added": list(readonly_boundary_audit.get("sky_pkl_mutated_or_added", [])),
        }
    return sha256_json(payload)


def seal_completed_cache_manifest(cache_dir: str | Path) -> Dict:
    """Verify an existing completed cache and atomically attach ``completed_seal_digest``."""
    cdir = Path(cache_dir).resolve()
    doc = verify_cache_manifest(
        cache_dir=cdir,
        expected_manifest_digest=None,
        verify_shard_hashes=True,
        require_completed=True,
        require_readonly_audit=True,
        require_completed_seal=False,
    )
    doc["completed_seal_digest"] = compute_completed_seal_digest(
        manifest_digest=doc["manifest_digest"],
        shards=doc["shards"],
        readonly_boundary_audit=doc.get("readonly_boundary_audit"),
    )
    atomic_write_text(cdir / "cache_manifest.json", json.dumps(doc, indent=2, ensure_ascii=False))
    return verify_cache_manifest(
        cache_dir=cdir,
        expected_manifest_digest=doc["manifest_digest"],
        verify_shard_hashes=True,
        require_completed=True,
        require_readonly_audit=True,
        require_completed_seal=True,
    )


def verify_cache_manifest(
    cache_dir: str | Path,
    expected_manifest_digest: Optional[str] = None,
    verify_shard_hashes: bool = True,
    require_completed: bool = False,
    require_readonly_audit: bool = False,
    require_completed_seal: bool = False,
    verify_disk_artifact_hashes: bool = False,
    workspace_root: Optional[Path] = None,
) -> Dict:
    """Strictly verify ``cache_manifest.json`` integrity, source-only lock, completeness, and seal.

    Raises ``RuntimeError`` on ANY manifest tampering, digest mismatch, action-space
    mismatch, source-domain violation, missing/deleted shards when completed,
    non-empty readonly boundary mutations, broken ``completed_seal_digest``, or shard corruption.
    """
    cdir = Path(cache_dir)
    manifest_path = cdir / "cache_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Cache manifest not found: {manifest_path}")

    try:
        doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to parse cache_manifest.json at {manifest_path}: {exc}") from exc

    if not isinstance(doc, dict) or "manifest" not in doc or "manifest_digest" not in doc:
        raise RuntimeError(f"Malformed cache_manifest.json at {manifest_path}: missing required keys.")

    manifest = doc["manifest"]
    recorded_digest = doc["manifest_digest"]
    recomputed_digest = sha256_json(manifest)
    if recorded_digest != recomputed_digest:
        raise RuntimeError(
            f"Tampered cache manifest detected at {manifest_path}: "
            f"recorded manifest_digest={recorded_digest} != recomputed={recomputed_digest}."
        )

    if expected_manifest_digest is not None and recorded_digest != expected_manifest_digest:
        raise RuntimeError(
            f"Cache manifest mismatch at {manifest_path}: "
            f"existing manifest_digest={recorded_digest} != expected={expected_manifest_digest}. "
            "Refusing to reuse incompatible cache; use a new --cache_dir or rebuild cleanly."
        )

    # Single-DGOD source-only invariant check
    try:
        validate_source_only_access(manifest.get("source_domain", ""), manifest.get("source_split", ""))
    except ValueError as exc:
        raise RuntimeError(f"Cache manifest failed source-only validation: {exc}") from exc

    # Policy / Oracle action-space alignment invariant check
    p_contract = manifest.get("policy_action_contract") or {}
    o_cfg = manifest.get("oracle_selector") or {}
    if float(o_cfg.get("nms_window_iou", -1.0)) != float(p_contract.get("nms_window_iou", -2.0)):
        raise RuntimeError(
            f"Cache manifest action-space mismatch: oracle nms_window_iou={o_cfg.get('nms_window_iou')} "
            f"!= policy nms_window_iou={p_contract.get('nms_window_iou')}."
        )
    if int(o_cfg.get("k_max", -1)) != int(p_contract.get("k_max", -2)):
        raise RuntimeError(
            f"Cache manifest k_max mismatch: oracle={o_cfg.get('k_max')} != policy={p_contract.get('k_max')}."
        )

    # Read-only boundary audit check (P1-02): unconditionally reject any recorded mutation
    if "readonly_boundary_audit" in doc:
        ro_audit = doc["readonly_boundary_audit"]
        if not isinstance(ro_audit, dict):
            raise RuntimeError(f"Malformed readonly_boundary_audit in {manifest_path}: {ro_audit!r}")
        for ro_key in ("textstyle_rn_mutated_or_added", "sky_pkl_mutated_or_added"):
            vals = ro_audit.get(ro_key)
            if not isinstance(vals, list) or len(vals) > 0:
                raise RuntimeError(
                    f"Cache manifest recorded read-only boundary violation ({ro_key}={vals!r}) at {manifest_path}."
                )
    elif require_readonly_audit:
        raise RuntimeError(f"Cache manifest at {manifest_path} is missing required 'readonly_boundary_audit'.")

    if require_completed and not bool(doc.get("completed", False)):
        raise RuntimeError(
            f"Offline cache at {cdir} is incomplete (completed=False). "
            "Resume build_phase2_cache.py before training."
        )

    shards = doc.get("shards")
    if not isinstance(shards, dict):
        raise RuntimeError(f"Malformed cache manifest shards section at {manifest_path}.")

    # Expected sample/shard completeness validation when completed (P1-02)
    expected_counts: Optional[Dict[str, int]] = None
    shard_size_cfg = int(manifest.get("shard_size", 0))
    if require_completed:
        split_cfg = manifest.get("split_config") or {}
        deg_cfg = manifest.get("degradation_recipe") or {}
        views_per_img = (1 if bool(deg_cfg.get("include_clean", True)) else 0) + int(
            deg_cfg.get("degraded_views_per_image", 0)
        )
        if shard_size_cfg < 1 or views_per_img < 1:
            raise RuntimeError(
                f"Invalid shard_size ({shard_size_cfg}) or views_per_img ({views_per_img}) in {manifest_path}."
            )
        expected_counts = {
            "source_train": int(split_cfg.get("selected_train_count", -1)) * views_per_img,
            "source_dev": int(split_cfg.get("selected_dev_count", -1)) * views_per_img,
        }

    for role in ("source_train", "source_dev"):
        entries = shards.get(role, [])
        if not isinstance(entries, list):
            raise RuntimeError(f"Malformed shards[{role!r}] in {manifest_path}.")

        seen_indices: List[int] = []
        role_samples = 0
        for entry in entries:
            rel_file = entry.get("file")
            if not rel_file:
                raise RuntimeError(f"Missing 'file' in shard entry: {entry}")
            seen_indices.append(int(entry.get("shard_index", -1)))
            role_samples += int(entry.get("num_samples", 0))
            shard_path = cdir / rel_file
            if not shard_path.is_file():
                raise RuntimeError(f"Missing cached shard file listed in manifest: {shard_path}")
            actual_size = shard_path.stat().st_size
            expected_size = int(entry.get("size_bytes", -1))
            if actual_size != expected_size:
                raise RuntimeError(
                    f"Shard size mismatch for {shard_path}: actual={actual_size} != manifest={expected_size}."
                )
            if verify_shard_hashes:
                actual_sha = sha256_file(shard_path)
                expected_sha = entry.get("sha256")
                if actual_sha != expected_sha:
                    raise RuntimeError(
                        f"Shard SHA256 tamper/corruption detected for {shard_path}: "
                        f"actual={actual_sha} != manifest={expected_sha}."
                    )

        if expected_counts is not None:
            exp_n = expected_counts[role]
            exp_shards = (exp_n + shard_size_cfg - 1) // shard_size_cfg if exp_n > 0 else 0
            if len(entries) != exp_shards:
                raise RuntimeError(
                    f"Incomplete shard list for {role} in {manifest_path}: "
                    f"found {len(entries)} shards, expected {exp_shards} for {exp_n} samples."
                )
            if seen_indices != list(range(len(entries))):
                raise RuntimeError(
                    f"Non-contiguous shard_index sequence for {role} in {manifest_path}: {seen_indices}"
                )
            if role_samples != exp_n:
                raise RuntimeError(
                    f"Sample count mismatch for {role} in {manifest_path}: "
                    f"shards sum to {role_samples}, expected {exp_n}."
                )

    # Top-level completed seal digest check (P1-02)
    recorded_seal = doc.get("completed_seal_digest")
    if recorded_seal is not None:
        recomputed_seal = compute_completed_seal_digest(
            manifest_digest=recorded_digest,
            shards=shards,
            readonly_boundary_audit=doc.get("readonly_boundary_audit"),
        )
        if str(recorded_seal) != recomputed_seal:
            raise RuntimeError(
                f"Broken completed_seal_digest at {manifest_path}: "
                f"recorded={recorded_seal} != recomputed={recomputed_seal}. "
                "Shard list, shard hashes, or readonly_boundary_audit was modified after sealing."
            )
    elif require_completed_seal:
        raise RuntimeError(f"Cache manifest at {manifest_path} is missing required 'completed_seal_digest'.")

    # Optional on-disk checkpoint / CLIP / source code hash verification
    if verify_disk_artifact_hashes:
        for path_key, sha_key in (
            ("checkpoint_path", "checkpoint_sha256"),
            ("clip_roi_weights_path", "clip_roi_weights_sha256"),
        ):
            artifact_path = Path(str(manifest.get(path_key, "")))
            if not artifact_path.is_file():
                raise RuntimeError(f"Recorded artifact missing on disk ({path_key}): {artifact_path}")
            actual_a_sha = sha256_file(artifact_path)
            if actual_a_sha != manifest.get(sha_key):
                raise RuntimeError(
                    f"On-disk artifact hash mismatch for {artifact_path}: "
                    f"actual={actual_a_sha} != manifest={manifest.get(sha_key)}."
                )
        if workspace_root is not None:
            ws = Path(workspace_root).resolve()
            for rel_code, exp_code_sha in (manifest.get("code_sha256") or {}).items():
                code_p = ws / rel_code
                if not code_p.is_file():
                    raise RuntimeError(f"Recorded source file missing on disk: {code_p}")
                if sha256_file(code_p) != exp_code_sha:
                    raise RuntimeError(f"On-disk source file hash mismatch for {code_p}.")

    return doc


def _summarize_shard_stats(shard_entries: Sequence[Dict], horizon_window_ids: Sequence[int]) -> Dict:
    """Aggregate summary statistics across completed shards for one split_role."""
    horizon_set = set(int(w) for w in horizon_window_ids)
    total_samples = 0
    total_bytes = 0
    deg_counts: Counter = Counter()
    k_counts: Counter = Counter()
    win_hist = np.zeros(18, dtype=np.int64)
    total_crops = 0
    horizon_crops = 0

    for entry in shard_entries:
        st = entry.get("stats") or {}
        total_samples += int(entry.get("num_samples", 0))
        total_bytes += int(entry.get("size_bytes", 0))
        for tag, cnt in (st.get("degradation_counts") or {}).items():
            deg_counts[tag] += int(cnt)
        for k_str, cnt in (st.get("k_counts") or {}).items():
            k_counts[str(k_str)] += int(cnt)
        hist = st.get("window_histogram")
        if hist and len(hist) == 18:
            win_hist += np.asarray(hist, dtype=np.int64)
        total_crops += int(st.get("total_crops", 0))
        horizon_crops += int(st.get("horizon_crops", 0))

    denom = max(total_samples, 1)
    k0 = int(k_counts.get("0", 0))
    k1 = int(k_counts.get("1", 0))
    k2 = int(k_counts.get("2", 0))
    avg_k = (k1 + 2 * k2) / denom

    # Also compute clean vs degraded breakdown if available
    clean_k_counts: Counter = Counter()
    deg_k_counts: Counter = Counter()
    for entry in shard_entries:
        st = entry.get("stats") or {}
        for k_str, cnt in (st.get("clean_k_counts") or {}).items():
            clean_k_counts[str(k_str)] += int(cnt)
        for k_str, cnt in (st.get("degraded_k_counts") or {}).items():
            deg_k_counts[str(k_str)] += int(cnt)

    clean_n = sum(clean_k_counts.values())
    deg_n = sum(deg_k_counts.values())
    clean_avg_k = (clean_k_counts.get("1", 0) + 2 * clean_k_counts.get("2", 0)) / max(clean_n, 1)
    deg_avg_k = (deg_k_counts.get("1", 0) + 2 * deg_k_counts.get("2", 0)) / max(deg_n, 1)

    return {
        "num_shards": len(shard_entries),
        "num_samples": total_samples,
        "size_bytes": total_bytes,
        "size_mib": round(total_bytes / (1024**2), 2),
        "degradation_counts": {tag: int(deg_counts.get(tag, 0)) for tag in DEGRADATION_TAGS},
        "k_counts": {"0": k0, "1": k1, "2": k2},
        "k_pct": {
            "0": round(100.0 * k0 / denom, 2),
            "1": round(100.0 * k1 / denom, 2),
            "2": round(100.0 * k2 / denom, 2),
        },
        "avg_k": round(avg_k, 4),
        "clean_no_crop_pct": round(100.0 * clean_k_counts.get("0", 0) / max(clean_n, 1), 2),
        "clean_avg_k": round(clean_avg_k, 4),
        "degraded_no_crop_pct": round(100.0 * deg_k_counts.get("0", 0) / max(deg_n, 1), 2),
        "degraded_avg_k": round(deg_avg_k, 4),
        "total_crops": total_crops,
        "horizon_crops": horizon_crops,
        "horizon_window_share_pct": round(100.0 * horizon_crops / max(total_crops, 1), 2),
        "window_histogram": [int(x) for x in win_hist.tolist()],
        "horizon_window_ids": sorted(horizon_set),
    }


def _compute_single_shard_stats(
    degradation_tags: Sequence[str],
    oracle_actions: Sequence[ActionTrajectory],
    horizon_window_ids: Sequence[int],
) -> Dict:
    horizon_set = set(int(w) for w in horizon_window_ids)
    deg_counts: Counter = Counter()
    k_counts: Counter = Counter()
    clean_k_counts: Counter = Counter()
    deg_k_counts: Counter = Counter()
    win_hist = np.zeros(18, dtype=np.int64)
    total_crops = 0
    horizon_crops = 0

    for tag, act in zip(degradation_tags, oracle_actions):
        deg_counts[tag] += 1
        k_str = str(len(act))
        k_counts[k_str] += 1
        if tag == "clean":
            clean_k_counts[k_str] += 1
        else:
            deg_k_counts[k_str] += 1
        for wid in act:
            win_hist[wid] += 1
            total_crops += 1
            if wid in horizon_set:
                horizon_crops += 1

    return {
        "degradation_counts": dict(deg_counts),
        "k_counts": {k: int(k_counts.get(k, 0)) for k in ("0", "1", "2")},
        "clean_k_counts": {k: int(clean_k_counts.get(k, 0)) for k in ("0", "1", "2")},
        "degraded_k_counts": {k: int(deg_k_counts.get(k, 0)) for k in ("0", "1", "2")},
        "total_crops": total_crops,
        "horizon_crops": horizon_crops,
        "window_histogram": [int(x) for x in win_hist.tolist()],
    }


def build_sharded_sft_cache(
    wrapper: FrozenTextStyleDetWrapper,
    loader: SourceOnlyDatasetLoader,
    split_info: SourceSplitInfo,
    cache_dir: str | Path,
    workspace_root: Path,
    shard_size: int = DEFAULT_SHARD_SIZE,
    base_seed: int = 20260420,
    include_clean: bool = True,
    degraded_views_per_image: int = 1,
    k_max: int = 2,
    nms_window_iou: float = 0.35,
    show_progress: bool = True,
) -> Dict:
    """Build (or resume after exact manifest match) sharded offline SFT state & pseudo-label cache."""
    validate_source_only_access(split_info.domain, split_info.split)
    if shard_size < 1:
        raise ValueError(f"shard_size must be >= 1, got {shard_size}")

    cdir = Path(cache_dir).resolve()
    shards_dir = cdir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cdir / "cache_manifest.json"

    # Canonical policy contract + strictly aligned OracleSelector.for_policy
    grid = CandidateGrid()
    state_encoder = StateEncoder(grid_size=(12, 20))
    policy_contract = ActiveZoomPolicyHead(
        grid=grid,
        visual_channels=1024,
        k_max=k_max,
        nms_window_iou=nms_window_iou,
    ).eval()
    oracle = OracleSelector.for_policy(policy_contract)
    assert oracle.nms_window_iou == policy_contract.nms_window_iou == float(nms_window_iou)
    assert oracle.k_max == policy_contract.k_max == int(k_max)

    horizon_window_ids = [w.window_id for w in grid.windows if w.group == "horizon_band"]
    deg_cfg = degradation_recipe_config(
        base_seed=base_seed,
        include_clean=include_clean,
        degraded_views_per_image=degraded_views_per_image,
    )
    manifest_payload = build_sft_cache_manifest_payload(
        wrapper=wrapper,
        grid=grid,
        state_encoder=state_encoder,
        policy_contract=policy_contract,
        oracle_selector=oracle,
        split_info=split_info,
        degradation_cfg=deg_cfg,
        shard_size=shard_size,
        workspace_root=workspace_root,
    )
    manifest_digest = sha256_json(manifest_payload)

    plans_by_role: Dict[str, List[PlannedViewItem]] = {
        "source_train": build_view_plan(
            split_info.selected_train_ids,
            split_role="source_train",
            base_seed=base_seed,
            include_clean=include_clean,
            degraded_views_per_image=degraded_views_per_image,
        ),
        "source_dev": build_view_plan(
            split_info.selected_dev_ids,
            split_role="source_dev",
            base_seed=base_seed,
            include_clean=include_clean,
            degraded_views_per_image=degraded_views_per_image,
        ),
    }

    # Check existing manifest for safe resume vs tamper/mismatch refusal
    existing_shards: Dict[str, Dict[int, Dict]] = {"source_train": {}, "source_dev": {}}
    prev_elapsed_sec = 0.0
    if manifest_path.is_file():
        prev_doc = verify_cache_manifest(
            cache_dir=cdir,
            expected_manifest_digest=manifest_digest,
            verify_shard_hashes=True,
            require_completed=False,
        )
        for role in ("source_train", "source_dev"):
            for entry in prev_doc.get("shards", {}).get(role, []):
                existing_shards[role][int(entry["shard_index"])] = entry
        prev_elapsed_sec = float((prev_doc.get("timing") or {}).get("total_elapsed_sec", 0.0))

    # Estimate remaining views and verify E: free disk space
    remaining_views = 0
    for role, plan in plans_by_role.items():
        num_shards = (len(plan) + shard_size - 1) // shard_size
        for s_idx in range(num_shards):
            if s_idx not in existing_shards[role]:
                s_start = s_idx * shard_size
                s_end = min(len(plan), s_start + shard_size)
                remaining_views += s_end - s_start

    disk_info = check_disk_space_for_cache(cdir, remaining_views)

    software_info = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "cuda": str(torch.version.cuda),
        "device_name": torch.cuda.get_device_name(wrapper.device) if wrapper.device.type == "cuda" else "cpu",
    }

    if wrapper.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(wrapper.device)

    dummy_logits = torch.zeros((1, policy_contract.num_actions), dtype=torch.float32)
    t_run_start = time.perf_counter()
    newly_processed_views = 0

    completed_shards_by_role: Dict[str, List[Dict]] = {
        role: [existing_shards[role][k] for k in sorted(existing_shards[role])]
        for role in ("source_train", "source_dev")
    }

    def _persist_manifest(is_completed: bool) -> Dict:
        now_elapsed = prev_elapsed_sec + (time.perf_counter() - t_run_start)
        train_stats = _summarize_shard_stats(completed_shards_by_role["source_train"], horizon_window_ids)
        dev_stats = _summarize_shard_stats(completed_shards_by_role["source_dev"], horizon_window_ids)
        total_samples_done = train_stats["num_samples"] + dev_stats["num_samples"]
        total_bytes_done = train_stats["size_bytes"] + dev_stats["size_bytes"]
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
        doc_out = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "manifest_digest": manifest_digest,
            "completed": bool(is_completed),
            "completed_seal_digest": (
                compute_completed_seal_digest(manifest_digest, completed_shards_by_role)
                if is_completed
                else None
            ),
            "manifest": manifest_payload,
            "shards": completed_shards_by_role,
            "stats": {
                "source_train": train_stats,
                "source_dev": dev_stats,
                "total_samples": total_samples_done,
                "total_size_bytes": total_bytes_done,
                "total_size_mib": round(total_bytes_done / (1024**2), 2),
                "total_size_gib": round(total_bytes_done / (1024**3), 3),
            },
            "timing": {
                "total_elapsed_sec": round(now_elapsed, 2),
                "session_elapsed_sec": round(time.perf_counter() - t_run_start, 2),
                "session_processed_views": newly_processed_views,
                "mean_sec_per_view": round(
                    (time.perf_counter() - t_run_start) / max(newly_processed_views, 1), 4
                )
                if newly_processed_views > 0
                else round(now_elapsed / max(total_samples_done, 1), 4),
            },
            "gpu_memory_gib": {
                "peak_allocated_gib": peak_alloc_gib,
                "peak_reserved_gib": peak_res_gib,
            },
            "disk_check": disk_info,
            "software": software_info,
        }
        atomic_write_text(manifest_path, json.dumps(doc_out, indent=2, ensure_ascii=False))
        return doc_out

    # Persist initial manifest (completed=False) if not already present so digest is anchored early
    if not manifest_path.is_file():
        _persist_manifest(is_completed=False)

    # Cache parsed GT & loaded PIL image across the 2+ views of the same image within a shard
    for role in ("source_train", "source_dev"):
        plan = plans_by_role[role]
        num_shards = (len(plan) + shard_size - 1) // shard_size

        for s_idx in range(num_shards):
            s_start = s_idx * shard_size
            s_end = min(len(plan), s_start + shard_size)
            chunk_items = plan[s_start:s_end]
            rel_file = f"shards/{role}_shard_{s_idx:04d}.pt"
            shard_path = cdir / rel_file

            if s_idx in existing_shards[role]:
                entry = existing_shards[role][s_idx]
                if int(entry["num_samples"]) == len(chunk_items) and entry["file"] == rel_file:
                    print(
                        f"[SFT Cache] Resuming: verified intact shard {rel_file} "
                        f"({len(chunk_items)} samples), skipping.",
                        flush=True,
                    )
                    continue
                raise RuntimeError(f"Existing shard metadata mismatch for {rel_file}: {entry}")

            c4_list: List[Tensor] = []
            cue_list: List[Tensor] = []
            style_list: List[Tensor] = []
            score_list: List[Tensor] = []
            mhot_list: List[Tensor] = []
            orig_sizes: List[Tuple[int, int]] = []
            resized_sizes: List[Tuple[int, int]] = []
            canvas_sizes: List[Tuple[int, int]] = []
            image_ids_out: List[str] = []
            split_roles_out: List[str] = []
            deg_tags_out: List[str] = []
            deg_seeds_out: List[int] = []
            oracle_actions_out: List[ActionTrajectory] = []

            cached_img_id: Optional[str] = None
            cached_pil: Optional[Image.Image] = None
            cached_gt: Optional[Tuple[Tensor, Tensor, Tensor]] = None

            desc = f"Cache [{role} shard {s_idx + 1}/{num_shards}]"
            iterator = tqdm(chunk_items, desc=desc, leave=True) if show_progress else chunk_items

            for item in iterator:
                if item.image_id != cached_img_id:
                    cached_pil = loader.load_image(item.image_id)
                    objs = loader.parse_xml(item.image_id)
                    cached_gt = record_to_0based_gt(objs, wrapper.class_names)
                    cached_img_id = item.image_id

                assert cached_pil is not None and cached_gt is not None
                gt_boxes, gt_labels, gt_diff = cached_gt

                view_pil = apply_physical_degradation(
                    cached_pil,
                    image_id=item.image_id,
                    view_tag=item.degradation_tag,
                    seed=item.degradation_seed,
                )

                perc = wrapper.forward_global(view_pil)
                state = state_encoder.encode_state(perc=perc, device="cpu", image_id=item.image_id)

                # Strict OracleSelector.for_policy pseudo-label mining
                sel_wids, win_scores = oracle.select(perc, gt_boxes, gt_labels, gt_diff)
                oracle_action: ActionTrajectory = tuple(int(w) for w in sel_wids)

                # Hard legality assertion via ActiveZoomPolicyHead.log_prob_from_logits before writing!
                lp_check = policy_contract.log_prob_from_logits(dummy_logits, [oracle_action])
                if not bool(torch.isfinite(lp_check).all().item()):
                    raise RuntimeError(
                        f"Non-finite policy log_prob for mined trajectory {oracle_action} on {item.image_id}"
                    )

                multi_hot = torch.zeros((policy_contract.num_actions,), dtype=torch.float32)
                if len(oracle_action) == 0:
                    multi_hot[policy_contract.no_crop_action] = 1.0
                else:
                    for wid in oracle_action:
                        multi_hot[wid] = 1.0

                cache_dict = state.to_cache_dict(storage_dtype=torch.float16)
                c4_list.append(cache_dict["c4_pooled"][0])  # [C, H_g, W_g] FP16
                cue_list.append(cache_dict["cue_maps"][0])  # [5, H_g, W_g] FP16
                style_list.append(cache_dict["style_stats"][0])  # [2*C] FP32
                score_list.append(win_scores.detach().cpu().to(dtype=torch.float32))  # [18] FP32
                mhot_list.append(multi_hot)  # [19] FP32
                orig_sizes.append(tuple(int(v) for v in cache_dict["orig_sizes"][0]))
                resized_sizes.append(tuple(int(v) for v in cache_dict["resized_sizes"][0]))
                canvas_sizes.append(tuple(int(v) for v in cache_dict["canvas_sizes"][0]))
                image_ids_out.append(item.image_id)
                split_roles_out.append(item.split_role)
                deg_tags_out.append(item.degradation_tag)
                deg_seeds_out.append(int(item.degradation_seed))
                oracle_actions_out.append(oracle_action)
                newly_processed_views += 1

            shard_payload = {
                "schema_version": CACHE_SCHEMA_VERSION,
                "manifest_digest": manifest_digest,
                "split_role": role,
                "shard_index": s_idx,
                "num_samples": len(chunk_items),
                "c4_pooled": torch.stack(c4_list, dim=0).contiguous(),
                "cue_maps": torch.stack(cue_list, dim=0).contiguous(),
                "style_stats": torch.stack(style_list, dim=0).contiguous(),
                "oracle_window_scores": torch.stack(score_list, dim=0).contiguous(),
                "sft_multi_hot": torch.stack(mhot_list, dim=0).contiguous(),
                "orig_sizes": orig_sizes,
                "resized_sizes": resized_sizes,
                "canvas_sizes": canvas_sizes,
                "image_ids": image_ids_out,
                "split_roles": split_roles_out,
                "degradation_tags": deg_tags_out,
                "degradation_seeds": deg_seeds_out,
                "oracle_actions": oracle_actions_out,
                "coordinate_system": "orig_0based_xyxy",
            }
            atomic_torch_save(shard_payload, shard_path)

            shard_entry = {
                "file": rel_file,
                "shard_index": s_idx,
                "num_samples": len(chunk_items),
                "size_bytes": int(shard_path.stat().st_size),
                "sha256": sha256_file(shard_path),
                "stats": _compute_single_shard_stats(deg_tags_out, oracle_actions_out, horizon_window_ids),
            }
            existing_shards[role][s_idx] = shard_entry
            completed_shards_by_role[role] = [
                existing_shards[role][k] for k in sorted(existing_shards[role])
            ]
            doc_snap = _persist_manifest(is_completed=False)
            sh_st = shard_entry["stats"]
            elapsed_now = doc_snap["timing"]["total_elapsed_sec"]
            spv = doc_snap["timing"]["mean_sec_per_view"]
            print(
                f"[SFT Cache] Saved {rel_file} ({len(chunk_items)} samples, "
                f"{shard_entry['size_bytes'] / (1024**2):.1f} MiB) | "
                f"K0/1/2={sh_st['k_counts']['0']}/{sh_st['k_counts']['1']}/{sh_st['k_counts']['2']} | "
                f"done={doc_snap['stats']['total_samples']} views | elapsed={elapsed_now:.1f}s ({spv:.3f}s/view)",
                flush=True,
            )
            del shard_payload, c4_list, cue_list, style_list, score_list, mhot_list
            if wrapper.device.type == "cuda":
                torch.cuda.empty_cache()

    final_doc = _persist_manifest(is_completed=True)
    # Final self-audit of written manifest & all shard hashes
    verify_cache_manifest(
        cache_dir=cdir,
        expected_manifest_digest=manifest_digest,
        verify_shard_hashes=True,
        require_completed=True,
    )
    return final_doc


# =============================================================================
# Sharded Offline Cache Reader & AgentBatchState Collator
# =============================================================================


def extract_sample_record_from_shard(
    shard_data: Dict,
    local_idx: int,
    device: torch.device | str = "cpu",
) -> Dict:
    """Restore a single cached sample using ``AgentBatchState.from_cache_dict``."""
    state_cache_dict = {
        "c4_pooled": shard_data["c4_pooled"][local_idx : local_idx + 1],
        "cue_maps": shard_data["cue_maps"][local_idx : local_idx + 1],
        "style_stats": shard_data["style_stats"][local_idx : local_idx + 1],
        "orig_sizes": [shard_data["orig_sizes"][local_idx]],
        "resized_sizes": [shard_data["resized_sizes"][local_idx]],
        "canvas_sizes": [shard_data["canvas_sizes"][local_idx]],
        "image_ids": [shard_data["image_ids"][local_idx]],
        "valid_action_mask": None,
        "coordinate_system": shard_data.get("coordinate_system", "orig_0based_xyxy"),
    }
    state = AgentBatchState.from_cache_dict(state_cache_dict, device=device)
    return {
        "state": state,
        "orig_size": tuple(int(v) for v in shard_data["orig_sizes"][local_idx]),
        "image_id": str(shard_data["image_ids"][local_idx]),
        "split_role": str(shard_data["split_roles"][local_idx]),
        "degradation_tag": str(shard_data["degradation_tags"][local_idx]),
        "degradation_seed": int(shard_data["degradation_seeds"][local_idx]),
        "oracle_action": tuple(int(w) for w in shard_data["oracle_actions"][local_idx]),
        "oracle_window_scores": shard_data["oracle_window_scores"][local_idx].detach().float().clone(),
        "sft_multi_hot": shard_data["sft_multi_hot"][local_idx].detach().float().clone(),
    }


def collate_sft_samples(
    samples: Sequence[Dict],
    device: torch.device | str = "cpu",
) -> Dict:
    """Collate single-sample cache records via ``AgentBatchState.collate``."""
    if not samples:
        raise ValueError("Cannot collate an empty sample list.")
    target = torch.device(device)
    batch_state = AgentBatchState.collate([s["state"] for s in samples], device=target)
    with torch.inference_mode(False):
        sft_multi_hot = torch.stack([s["sft_multi_hot"] for s in samples], dim=0).to(
            device=target, dtype=torch.float32
        )
        oracle_window_scores = torch.stack([s["oracle_window_scores"] for s in samples], dim=0).to(
            device=target, dtype=torch.float32
        )
    return {
        "state": batch_state,
        "sft_multi_hot": sft_multi_hot,
        "oracle_window_scores": oracle_window_scores,
        "oracle_actions": [s["oracle_action"] for s in samples],
        "degradation_tags": [s["degradation_tag"] for s in samples],
        "degradation_seeds": [s["degradation_seed"] for s in samples],
        "image_ids": [s["image_id"] for s in samples],
        "split_roles": [s["split_role"] for s in samples],
        "orig_sizes": [s["orig_size"] for s in samples],
    }


class ShardedSFTCacheReader:
    """Memory-bounded sharded reader with manifest verification & AgentBatchState batching."""

    def __init__(
        self,
        cache_dir: str | Path,
        split_role: str,
        verify_shard_hashes: bool = True,
        require_completed: bool = True,
        max_cached_shards_in_ram: int = 4,
    ) -> None:
        if split_role not in {"source_train", "source_dev"}:
            raise ValueError(f"split_role must be 'source_train' or 'source_dev', got {split_role!r}")
        self.cache_dir = Path(cache_dir).resolve()
        self.split_role = split_role
        self.manifest_doc = verify_cache_manifest(
            cache_dir=self.cache_dir,
            expected_manifest_digest=None,
            verify_shard_hashes=verify_shard_hashes,
            require_completed=require_completed,
        )
        self.manifest_digest: str = self.manifest_doc["manifest_digest"]
        self.manifest: Dict = self.manifest_doc["manifest"]
        self.shard_entries: List[Dict] = list(self.manifest_doc["shards"][split_role])
        if not self.shard_entries:
            raise RuntimeError(f"No shards found for split_role={split_role!r} in {self.cache_dir}")

        self.max_cached_shards_in_ram = max(1, int(max_cached_shards_in_ram))
        self._shard_ram_cache: Dict[int, Dict] = {}
        self._shard_lru_order: List[int] = []

        self.shard_offsets: List[Tuple[int, int]] = []
        offset = 0
        for entry in self.shard_entries:
            n = int(entry["num_samples"])
            self.shard_offsets.append((offset, offset + n))
            offset += n
        self.total_samples = offset

    def __len__(self) -> int:
        return self.total_samples

    def load_shard(self, list_idx: int) -> Dict:
        if list_idx in self._shard_ram_cache:
            if list_idx in self._shard_lru_order:
                self._shard_lru_order.remove(list_idx)
            self._shard_lru_order.append(list_idx)
            return self._shard_ram_cache[list_idx]

        entry = self.shard_entries[list_idx]
        shard_path = self.cache_dir / entry["file"]
        data = torch.load(str(shard_path), map_location="cpu", weights_only=False)
        if data.get("manifest_digest") != self.manifest_digest:
            raise RuntimeError(
                f"Shard {shard_path} internal manifest_digest={data.get('manifest_digest')} "
                f"!= cache manifest_digest={self.manifest_digest}"
            )
        if data.get("split_role") != self.split_role:
            raise RuntimeError(
                f"Shard {shard_path} split_role={data.get('split_role')} != expected {self.split_role}"
            )
        if int(data.get("num_samples", -1)) != int(entry["num_samples"]):
            raise RuntimeError(
                f"Shard {shard_path} num_samples={data.get('num_samples')} != manifest {entry['num_samples']}"
            )

        while len(self._shard_ram_cache) >= self.max_cached_shards_in_ram:
            evict_idx = self._shard_lru_order.pop(0)
            self._shard_ram_cache.pop(evict_idx, None)
        self._shard_ram_cache[list_idx] = data
        self._shard_lru_order.append(list_idx)
        return data

    def get_sample(self, global_idx: int, device: torch.device | str = "cpu") -> Dict:
        if global_idx < 0 or global_idx >= self.total_samples:
            raise IndexError(f"Index {global_idx} out of range [0, {self.total_samples})")
        for list_idx, (start, end) in enumerate(self.shard_offsets):
            if start <= global_idx < end:
                shard_data = self.load_shard(list_idx)
                return extract_sample_record_from_shard(shard_data, global_idx - start, device=device)
        raise IndexError(f"Failed to locate global_idx={global_idx}")

    def compute_label_positive_counts(self) -> Tuple[Tensor, int]:
        """Compute exact per-action (19-dim) positive counts across this split (for train pos_weight)."""
        pos_counts = torch.zeros(19, dtype=torch.float64)
        total = 0
        for list_idx in range(len(self.shard_entries)):
            shard = self.load_shard(list_idx)
            mhot = shard["sft_multi_hot"].to(dtype=torch.float64)
            pos_counts += mhot.sum(dim=0)
            total += int(mhot.shape[0])
        return pos_counts.to(dtype=torch.float32), total

    def iter_batches(
        self,
        batch_size: int = 32,
        shuffle: bool = False,
        seed: int = 0,
        device: torch.device | str = "cpu",
        drop_last: bool = False,
        shard_group_size: int = 4,
    ) -> Iterator[Dict]:
        """Stream batches via ``AgentBatchState.from_cache_dict`` + ``AgentBatchState.collate``.

        Loads at most ``shard_group_size`` shards at a time to keep CPU RAM bounded even on
        the full 38,790-sample dataset while achieving strong epoch-level shuffling.
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        rng = np.random.default_rng(seed)
        shard_order = list(range(len(self.shard_entries)))
        if shuffle:
            rng.shuffle(shard_order)

        grp = max(1, min(int(shard_group_size), len(shard_order)))
        carry: List[Dict] = []

        for g_start in range(0, len(shard_order), grp):
            group_indices = shard_order[g_start : g_start + grp]
            loaded = [self.load_shard(si) for si in group_indices]

            # Build index pairs (pos_in_group, local_idx)
            pairs: List[Tuple[int, int]] = []
            for p_i, sd in enumerate(loaded):
                n_s = int(sd["num_samples"])
                pairs.extend((p_i, li) for li in range(n_s))
            if shuffle:
                rng.shuffle(pairs)

            for p_i, li in pairs:
                rec = extract_sample_record_from_shard(loaded[p_i], li, device="cpu")
                carry.append(rec)
                if len(carry) == batch_size:
                    yield collate_sft_samples(carry, device=device)
                    carry = []

        if carry and not drop_last:
            yield collate_sft_samples(carry, device=device)
