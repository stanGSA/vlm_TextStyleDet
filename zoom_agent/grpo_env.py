"""Isolated GRPO Detection Environment, Symmetric Crop Cache & Smooth Rewarder (Phase 3).

Design & Red-Team Audit Guarantees:
  - **Strict Single-DGOD Source-Only Guard**: Refuses ANY domain != ``daytime_clear`` or
    split != ``train`` (raises ``ValueError``), and validates ``image_id`` against path traversal.
  - **P0-01 Isolated Single-Crop Evaluation**: Every missing crop window ``wid in [0..17]`` is
    computed strictly via ``FrozenTextStyleDetWrapper.forward_single_crop`` (isolated
    ``batch_size = 1`` canvas, zero cross-window padding coupling, zero GT influence).
  - **Exact Dual-Window Symmetry**: Actions ``(w1, w2)`` and ``(w2, w1)`` canonicalize to
    ``tuple(sorted(action))`` when fetching unordered ``wid`` cache entries and calling
    ``DualViewWBFFusion.fuse``, guaranteeing bitwise-identical fused detections and rewards
    with zero duplicate inference.
  - **P0-02 Cryptographic Cache Manifest & Seal**: ``env_cache_manifest.json`` hashes detector/
    CLIP SHA256, detector/grid/fuser/degradation configs, source-only lock, and code hashes,
    plus per-bucket SHA256 digests and a top-level ``seal_digest``. Any tampering, mismatch, or
    corruption raises ``RuntimeError``.
  - **Smooth Multi-Objective Incremental Reward**:
      R(a) = lambda_recall * delta_Recall_small_weak(a)
           + lambda_loc_ap * delta_LocAP_smooth(a)
           - lambda_fp     * delta_FP_pen(a)
           - lambda_cost   * |a|
    with strict empty-action anchor ``R(()) == 0.0`` and full telemetry breakdown.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import hashlib
import json
import os
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torchvision
from PIL import Image
from torch import Tensor
from torchvision.ops import box_iou
from tqdm import tqdm

from .candidate_grid import CandidateGrid
from .detector_wrapper import FrozenTextStyleDetWrapper
from .eval_utils import atomic_write_text, record_to_0based_gt, sha256_file, sha256_json
from .policy_head import ActionTrajectory
from .sft_cache import atomic_torch_save
from .source_degradation import (
    ALLOWED_SOURCE_DOMAIN,
    ALLOWED_SOURCE_SPLIT,
    DEGRADATION_TAGS,
    SourceOnlyDatasetLoader,
    apply_physical_degradation,
    degradation_recipe_config,
    validate_source_only_access,
)
from .state_encoder import DEFAULT_WEAK_CLASS_IDS
from .wbf_fusion import DualViewWBFFusion

ENV_CACHE_SCHEMA_VERSION: int = 1
DEFAULT_NUM_BUCKETS: int = 64

ViewKey = Tuple[str, str, int]  # (image_id, degradation_tag, degradation_seed)


def canonicalize_action_windows(
    action: Sequence[int],
    num_windows: int = 18,
    k_max: int = 2,
) -> Tuple[int, ...]:
    """Validate an action trajectory and return its sorted canonical unordered window IDs."""
    wids = [int(w) for w in action]
    if len(wids) > k_max:
        raise ValueError(f"Action {tuple(wids)} exceeds k_max={k_max}")
    if len(set(wids)) != len(wids):
        raise ValueError(f"Action {tuple(wids)} contains duplicate window IDs")
    for w in wids:
        if w < 0 or w >= num_windows:
            raise ValueError(f"Window ID {w} out of range [0, {num_windows - 1}]")
    return tuple(sorted(wids))


def view_key_to_str(view_key: ViewKey) -> str:
    img_id, deg_tag, deg_seed = view_key
    return f"{img_id}|{deg_tag}|{int(deg_seed)}"


def bucket_id_for_view(view_key: ViewKey, num_buckets: int = DEFAULT_NUM_BUCKETS) -> int:
    key_str = view_key_to_str(view_key)
    h = hashlib.sha256(key_str.encode("utf-8")).digest()
    return int.from_bytes(h[:4], byteorder="little", signed=False) % int(num_buckets)


def _clone_global_det_cpu(det: Dict[str, Tensor]) -> Dict[str, Tensor]:
    return {
        "boxes": det["boxes"].detach().cpu().to(dtype=torch.float32).contiguous().clone(),
        "scores": det["scores"].detach().cpu().to(dtype=torch.float32).contiguous().clone(),
        "labels": det["labels"].detach().cpu().to(dtype=torch.long).contiguous().clone(),
    }


def _clone_crop_det_cpu(det: Dict[str, Tensor]) -> Dict[str, Tensor]:
    return {
        "boxes_in_crop": det["boxes_in_crop"].detach().cpu().to(dtype=torch.float32).contiguous().clone(),
        "boxes_in_orig": det["boxes_in_orig"].detach().cpu().to(dtype=torch.float32).contiguous().clone(),
        "scores": det["scores"].detach().cpu().to(dtype=torch.float32).contiguous().clone(),
        "labels": det["labels"].detach().cpu().to(dtype=torch.long).contiguous().clone(),
        "crop_xyxy": tuple(int(v) for v in det["crop_xyxy"]),
        "zoom_size": tuple(int(v) for v in det["zoom_size"]),
        "orig_size": tuple(int(v) for v in det["orig_size"]),
    }


# =============================================================================
# Cryptographic Manifest & Persistent Bucketed Detection Cache (P0-01 / P0-02)
# =============================================================================


def build_env_cache_manifest_payload(
    wrapper: FrozenTextStyleDetWrapper,
    grid: CandidateGrid,
    fuser: DualViewWBFFusion,
    degradation_cfg: Dict,
    workspace_root: Path,
    source_domain: str = ALLOWED_SOURCE_DOMAIN,
    source_split: str = ALLOWED_SOURCE_SPLIT,
    num_buckets: int = DEFAULT_NUM_BUCKETS,
    sft_cache_manifest_digest: Optional[str] = None,
) -> Dict:
    """Build immutable canonical manifest payload for the Phase 3 detection environment cache."""
    validate_source_only_access(source_domain, source_split)
    det_cfg = wrapper.config_dict()
    if det_cfg.get("crop_isolation_mode") != "strict_single_crop_batch1":
        raise RuntimeError(
            f"Refusing non-isolated crop execution mode: {det_cfg.get('crop_isolation_mode')!r}"
        )

    code_paths = [
        workspace_root / "zoom_agent" / "candidate_grid.py",
        workspace_root / "zoom_agent" / "detector_wrapper.py",
        workspace_root / "zoom_agent" / "wbf_fusion.py",
        workspace_root / "zoom_agent" / "source_degradation.py",
        workspace_root / "zoom_agent" / "grpo_env.py",
    ]
    code_hashes = {
        str(p.relative_to(workspace_root)).replace("\\", "/"): sha256_file(p)
        for p in code_paths
        if p.is_file()
    }

    return {
        "schema_version": ENV_CACHE_SCHEMA_VERSION,
        "source_domain": source_domain,
        "source_split": source_split,
        "checkpoint_path": str(wrapper.checkpoint_path),
        "checkpoint_sha256": sha256_file(wrapper.checkpoint_path),
        "clip_roi_weights_path": str(wrapper.clip_roi_weights_path),
        "clip_roi_weights_sha256": sha256_file(wrapper.clip_roi_weights_path),
        "detector_config": det_cfg,
        "candidate_grid": grid.config_dict(),
        "fuser_config": fuser.config_dict(),
        "degradation_recipe": degradation_cfg,
        "sft_cache_manifest_digest": str(sft_cache_manifest_digest) if sft_cache_manifest_digest else None,
        "num_buckets": int(num_buckets),
        "coordinate_system": "orig_0based_xyxy",
        "code_sha256": code_hashes,
    }


def compute_env_cache_seal_digest(
    manifest_digest: str,
    buckets: Dict[str, Dict],
) -> str:
    """Compute tamper-evident seal over ``manifest_digest`` and all persisted bucket digests."""
    canonical_buckets = [
        {
            "bucket_id": int(k),
            "file": str(v["file"]),
            "num_views": int(v["num_views"]),
            "num_crops": int(v["num_crops"]),
            "size_bytes": int(v["size_bytes"]),
            "sha256": str(v["sha256"]),
        }
        for k, v in sorted(buckets.items(), key=lambda kv: int(kv[0]))
    ]
    payload = {
        "schema_version": ENV_CACHE_SCHEMA_VERSION,
        "manifest_digest": str(manifest_digest),
        "buckets": canonical_buckets,
    }
    return sha256_json(payload)


def verify_env_cache_manifest(
    cache_dir: str | Path,
    expected_manifest_digest: Optional[str] = None,
    verify_bucket_hashes: bool = True,
    check_unregistered_buckets: bool = True,
    verify_disk_artifact_hashes: bool = False,
    workspace_root: Optional[Path] = None,
) -> Dict:
    """Strictly verify ``env_cache_manifest.json`` integrity, source-only lock, and bucket seals.

    Raises ``RuntimeError`` on ANY manifest tampering, digest mismatch, source-domain violation,
    non-isolated crop mode, missing/corrupted bucket file, unregistered orphan bucket file,
    broken ``seal_digest``, or (when enabled) on-disk checkpoint/CLIP/code hash mismatch.
    """
    cdir = Path(cache_dir)
    manifest_path = cdir / "env_cache_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Env cache manifest not found: {manifest_path}")

    try:
        doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to parse env_cache_manifest.json at {manifest_path}: {exc}") from exc

    if not isinstance(doc, dict) or "manifest" not in doc or "manifest_digest" not in doc:
        raise RuntimeError(f"Malformed env_cache_manifest.json at {manifest_path}: missing required keys.")

    manifest = doc["manifest"]
    recorded_digest = str(doc["manifest_digest"])
    recomputed_digest = sha256_json(manifest)
    if recorded_digest != recomputed_digest:
        raise RuntimeError(
            f"Tampered env_cache_manifest.json at {manifest_path}: "
            f"recorded manifest_digest={recorded_digest} != recomputed={recomputed_digest}."
        )

    if expected_manifest_digest is not None and recorded_digest != expected_manifest_digest:
        raise RuntimeError(
            f"Env cache manifest mismatch at {manifest_path}: "
            f"existing={recorded_digest} != expected={expected_manifest_digest}. "
            "Refusing to reuse incompatible detection cache; use a clean --env_cache_dir."
        )

    # Single-DGOD source-only invariant
    try:
        validate_source_only_access(manifest.get("source_domain", ""), manifest.get("source_split", ""))
    except ValueError as exc:
        raise RuntimeError(f"Env cache manifest failed source-only validation: {exc}") from exc

    # P0-01 strict single-crop isolation invariant
    det_cfg = manifest.get("detector_config") or {}
    if det_cfg.get("crop_isolation_mode") != "strict_single_crop_batch1":
        raise RuntimeError(
            f"Env cache manifest violated P0-01 isolation mode: {det_cfg.get('crop_isolation_mode')!r}"
        )

    buckets = doc.get("buckets")
    if not isinstance(buckets, dict):
        raise RuntimeError(f"Malformed 'buckets' section in {manifest_path}.")

    for b_key, entry in buckets.items():
        rel_file = entry.get("file")
        if not rel_file:
            raise RuntimeError(f"Missing 'file' in env cache bucket entry {b_key}: {entry}")
        b_path = cdir / rel_file
        if not b_path.is_file():
            raise RuntimeError(f"Missing env cache bucket file listed in manifest: {b_path}")
        actual_size = int(b_path.stat().st_size)
        expected_size = int(entry.get("size_bytes", -1))
        if actual_size != expected_size:
            raise RuntimeError(
                f"Env cache bucket size mismatch for {b_path}: actual={actual_size} != manifest={expected_size}."
            )
        if verify_bucket_hashes:
            actual_sha = sha256_file(b_path)
            expected_sha = str(entry.get("sha256", ""))
            if actual_sha != expected_sha:
                raise RuntimeError(
                    f"Env cache bucket SHA256 corruption/tamper detected for {b_path}: "
                    f"actual={actual_sha} != manifest={expected_sha}."
                )

    recorded_seal = doc.get("seal_digest")
    if not recorded_seal:
        raise RuntimeError(f"Env cache manifest at {manifest_path} is missing required 'seal_digest'.")
    recomputed_seal = compute_env_cache_seal_digest(recorded_digest, buckets)
    if str(recorded_seal) != recomputed_seal:
        raise RuntimeError(
            f"Broken env cache seal_digest at {manifest_path}: "
            f"recorded={recorded_seal} != recomputed={recomputed_seal}."
        )

    # P3-01: Reject unregistered orphan .pt files in buckets/ directory
    if check_unregistered_buckets:
        buckets_dir = cdir / "buckets"
        if buckets_dir.is_dir():
            registered_files = {
                str(entry["file"]).replace("\\", "/")
                for entry in buckets.values()
                if entry.get("file")
            }
            for pt_file in sorted(buckets_dir.glob("*.pt")):
                rel_pt = str(pt_file.relative_to(cdir)).replace("\\", "/")
                if rel_pt not in registered_files:
                    raise RuntimeError(
                        f"Unregistered orphan env cache bucket file detected on disk: {pt_file} "
                        f"(not present in env_cache_manifest.json)."
                    )

    # P3-01: Optional independent on-disk detector/CLIP weights & source code SHA256 verification
    if verify_disk_artifact_hashes:
        for path_key, sha_key in (
            ("checkpoint_path", "checkpoint_sha256"),
            ("clip_roi_weights_path", "clip_roi_weights_sha256"),
        ):
            art_path = Path(str(manifest.get(path_key, "")))
            if not art_path.is_file():
                raise RuntimeError(f"Recorded artifact missing on disk ({path_key}): {art_path}")
            actual_sha = sha256_file(art_path)
            if actual_sha != manifest.get(sha_key):
                raise RuntimeError(
                    f"On-disk artifact hash mismatch for {art_path}: "
                    f"actual={actual_sha} != manifest={manifest.get(sha_key)}."
                )
        if workspace_root is not None:
            ws = Path(workspace_root).resolve()
            for rel_code, exp_code_sha in (manifest.get("code_sha256") or {}).items():
                code_p = ws / rel_code
                if not code_p.is_file():
                    raise RuntimeError(f"Recorded source file missing on disk: {code_p}")
                if sha256_file(code_p) != exp_code_sha:
                    raise RuntimeError(
                        f"On-disk source file hash mismatch for {code_p}: "
                        f"actual={sha256_file(code_p)} != manifest={exp_code_sha}."
                    )

    return doc


class PersistentDetectionEnvCache:
    """Bucket-sharded persistent cache for ``global_detections`` and unordered ``crop_detections[wid]``."""

    def __init__(
        self,
        cache_dir: str | Path,
        manifest_payload: Optional[Dict] = None,
        verify_bucket_hashes: bool = True,
        read_only: bool = False,
    ) -> None:
        self.cache_dir = Path(cache_dir).resolve()
        self.buckets_dir = self.cache_dir / "buckets"
        self.manifest_path = self.cache_dir / "env_cache_manifest.json"
        self.read_only = bool(read_only)

        self._loaded_buckets: Dict[int, Dict[str, Dict]] = {}
        self._dirty_buckets: set[int] = set()
        self.bucket_entries: Dict[str, Dict] = {}

        if self.manifest_path.is_file():
            expected_digest = sha256_json(manifest_payload) if manifest_payload is not None else None
            doc = verify_env_cache_manifest(
                cache_dir=self.cache_dir,
                expected_manifest_digest=expected_digest,
                verify_bucket_hashes=verify_bucket_hashes,
            )
            self.manifest: Dict = doc["manifest"]
            self.manifest_digest: str = str(doc["manifest_digest"])
            self.num_buckets: int = int(self.manifest["num_buckets"])
            self.bucket_entries = dict(doc.get("buckets") or {})
            self._software_info = dict(doc.get("software") or {})
        else:
            if manifest_payload is None:
                raise FileNotFoundError(
                    f"Env cache does not exist at {self.cache_dir} and no manifest_payload was supplied to initialize it."
                )
            if self.read_only:
                raise FileNotFoundError(f"Env cache missing in read-only mode: {self.manifest_path}")
            validate_source_only_access(
                manifest_payload.get("source_domain", ""),
                manifest_payload.get("source_split", ""),
            )
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.buckets_dir.mkdir(parents=True, exist_ok=True)
            self.manifest = manifest_payload
            self.manifest_digest = sha256_json(manifest_payload)
            self.num_buckets = int(manifest_payload["num_buckets"])
            self._software_info = {
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "torchvision": str(torchvision.__version__),
                "cuda": str(torch.version.cuda),
            }
            self._persist_manifest()

    def _bucket_rel_path(self, bucket_id: int) -> str:
        return f"buckets/bucket_{int(bucket_id):03d}.pt"

    def _ensure_bucket_loaded(self, bucket_id: int) -> Dict[str, Dict]:
        if bucket_id in self._loaded_buckets:
            return self._loaded_buckets[bucket_id]

        b_key = str(bucket_id)
        if b_key in self.bucket_entries:
            rel_file = self.bucket_entries[b_key]["file"]
            b_path = self.cache_dir / rel_file
            if not b_path.is_file():
                raise RuntimeError(f"Env cache bucket file missing on disk: {b_path}")
            data = torch.load(str(b_path), map_location="cpu", weights_only=False)
            if data.get("manifest_digest") != self.manifest_digest:
                raise RuntimeError(
                    f"Env cache bucket {b_path} internal manifest_digest={data.get('manifest_digest')} "
                    f"!= expected={self.manifest_digest}"
                )
            views = data.get("views")
            if not isinstance(views, dict):
                raise RuntimeError(f"Malformed bucket payload in {b_path}: 'views' must be dict.")
            self._loaded_buckets[bucket_id] = views
        else:
            self._loaded_buckets[bucket_id] = {}
        return self._loaded_buckets[bucket_id]

    def has_global(self, view_key: ViewKey) -> bool:
        bid = bucket_id_for_view(view_key, self.num_buckets)
        views = self._ensure_bucket_loaded(bid)
        v_entry = views.get(view_key_to_str(view_key))
        return v_entry is not None and ("global_det" in v_entry)

    def get_global(self, view_key: ViewKey) -> Optional[Tuple[Dict[str, Tensor], Tuple[int, int]]]:
        bid = bucket_id_for_view(view_key, self.num_buckets)
        views = self._ensure_bucket_loaded(bid)
        v_entry = views.get(view_key_to_str(view_key))
        if v_entry is None or "global_det" not in v_entry:
            return None
        return v_entry["global_det"], tuple(int(x) for x in v_entry["orig_size"])

    def put_global(
        self,
        view_key: ViewKey,
        global_det: Dict[str, Tensor],
        orig_size: Tuple[int, int],
    ) -> None:
        if self.read_only:
            raise RuntimeError("Cannot write to PersistentDetectionEnvCache in read-only mode.")
        bid = bucket_id_for_view(view_key, self.num_buckets)
        views = self._ensure_bucket_loaded(bid)
        k_str = view_key_to_str(view_key)
        entry = views.get(k_str)
        if entry is None:
            entry = {
                "image_id": str(view_key[0]),
                "degradation_tag": str(view_key[1]),
                "degradation_seed": int(view_key[2]),
                "orig_size": (int(orig_size[0]), int(orig_size[1])),
                "global_det": _clone_global_det_cpu(global_det),
                "crops": {},
            }
            views[k_str] = entry
        else:
            entry["orig_size"] = (int(orig_size[0]), int(orig_size[1]))
            entry["global_det"] = _clone_global_det_cpu(global_det)
        self._dirty_buckets.add(bid)

    def has_crop(self, view_key: ViewKey, wid: int) -> bool:
        bid = bucket_id_for_view(view_key, self.num_buckets)
        views = self._ensure_bucket_loaded(bid)
        v_entry = views.get(view_key_to_str(view_key))
        if v_entry is None:
            return False
        return int(wid) in v_entry.get("crops", {})

    def get_crop(self, view_key: ViewKey, wid: int) -> Optional[Dict[str, Tensor]]:
        bid = bucket_id_for_view(view_key, self.num_buckets)
        views = self._ensure_bucket_loaded(bid)
        v_entry = views.get(view_key_to_str(view_key))
        if v_entry is None:
            return None
        return v_entry.get("crops", {}).get(int(wid))

    def put_crop(
        self,
        view_key: ViewKey,
        wid: int,
        crop_det: Dict[str, Tensor],
    ) -> None:
        if self.read_only:
            raise RuntimeError("Cannot write to PersistentDetectionEnvCache in read-only mode.")
        bid = bucket_id_for_view(view_key, self.num_buckets)
        views = self._ensure_bucket_loaded(bid)
        k_str = view_key_to_str(view_key)
        entry = views.get(k_str)
        if entry is None:
            entry = {
                "image_id": str(view_key[0]),
                "degradation_tag": str(view_key[1]),
                "degradation_seed": int(view_key[2]),
                "orig_size": tuple(int(v) for v in crop_det["orig_size"]),
                "crops": {},
            }
            views[k_str] = entry
        entry.setdefault("crops", {})[int(wid)] = _clone_crop_det_cpu(crop_det)
        self._dirty_buckets.add(bid)

    def _persist_manifest(self) -> Dict:
        total_views = sum(int(e["num_views"]) for e in self.bucket_entries.values())
        total_crops = sum(int(e["num_crops"]) for e in self.bucket_entries.values())
        total_bytes = sum(int(e["size_bytes"]) for e in self.bucket_entries.values())
        seal_digest = compute_env_cache_seal_digest(self.manifest_digest, self.bucket_entries)
        doc = {
            "schema_version": ENV_CACHE_SCHEMA_VERSION,
            "manifest_digest": self.manifest_digest,
            "seal_digest": seal_digest,
            "manifest": self.manifest,
            "buckets": self.bucket_entries,
            "stats": {
                "num_persisted_buckets": len(self.bucket_entries),
                "total_cached_views": total_views,
                "total_cached_crops": total_crops,
                "total_size_bytes": total_bytes,
                "total_size_mib": round(total_bytes / (1024**2), 2),
            },
            "software": self._software_info,
        }
        atomic_write_text(self.manifest_path, json.dumps(doc, indent=2, ensure_ascii=False))
        return doc

    def flush(self) -> Dict:
        """Atomically persist all dirty buckets and update ``env_cache_manifest.json``."""
        if self.read_only or not self._dirty_buckets:
            return self.stats()

        self.buckets_dir.mkdir(parents=True, exist_ok=True)
        for bid in sorted(self._dirty_buckets):
            views = self._loaded_buckets[bid]
            rel_file = self._bucket_rel_path(bid)
            b_path = self.cache_dir / rel_file
            payload = {
                "schema_version": ENV_CACHE_SCHEMA_VERSION,
                "manifest_digest": self.manifest_digest,
                "bucket_id": int(bid),
                "views": views,
            }
            atomic_torch_save(payload, b_path)
            n_views = sum(1 for v in views.values() if "global_det" in v)
            n_crops = sum(len(v.get("crops", {})) for v in views.values())
            self.bucket_entries[str(bid)] = {
                "bucket_id": int(bid),
                "file": rel_file,
                "num_views": int(n_views),
                "num_crops": int(n_crops),
                "size_bytes": int(b_path.stat().st_size),
                "sha256": sha256_file(b_path),
            }
        self._dirty_buckets.clear()
        doc = self._persist_manifest()
        return doc["stats"]

    def stats(self) -> Dict:
        total_views = sum(int(e["num_views"]) for e in self.bucket_entries.values())
        total_crops = sum(int(e["num_crops"]) for e in self.bucket_entries.values())
        total_bytes = sum(int(e["size_bytes"]) for e in self.bucket_entries.values())
        return {
            "manifest_digest": self.manifest_digest,
            "num_persisted_buckets": len(self.bucket_entries),
            "dirty_buckets": len(self._dirty_buckets),
            "total_cached_views": total_views,
            "total_cached_crops": total_crops,
            "total_size_bytes": total_bytes,
            "total_size_mib": round(total_bytes / (1024**2), 2),
        }


# =============================================================================
# Smooth Multi-Objective Incremental Rewarder (SmolVLM.md Module 5 & Sec. 7)
# =============================================================================


@dataclass(frozen=True)
class GRPORewardConfig:
    """Hyperparameters for the smooth incremental GRPO detection rewarder (Phase 3 sealed defaults)."""

    lambda_recall: float = 2.0  # Weight on delta small/weak-object 1-to-1 recall
    lambda_loc_ap: float = 3.5  # Weight on delta LocAP_smooth (AP50:95 + AP75 + matched IoU)
    lambda_fp: float = 0.90  # Weight on delta high-confidence background FP penalty
    lambda_cost: float = 0.045  # Per-crop budget cost |a_i|

    # Sub-weights inside LocAP_smooth (sum to 1.0 so scale stays in [0, 1])
    loc_w_ap50_95: float = 0.45
    loc_w_ap75: float = 0.30
    loc_w_matched_iou: float = 0.25

    # Sub-weights inside Recall_small/weak (balances R@50 discovery & R@75 tight recall)
    recall_w_50: float = 0.60
    recall_w_75: float = 0.40

    # Detection & FP matching thresholds (sealed Phase 3 uses fp_score_thr=0.05;
    # P2-03 note: for future new retraining ablations, aligning fp_score_thr=0.04 closes
    # the [0.042, 0.050) absent-class local-only FP threshold gap)
    ap_score_thr: float = 0.03
    recall_score_thr: float = 0.04
    fp_score_thr: float = 0.05
    fp_score_power: float = 1.5
    fp_absent_class_mult: float = 1.2
    small_area_ratio: float = 0.015
    weak_class_ids: Tuple[int, ...] = DEFAULT_WEAK_CLASS_IDS

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["weak_class_ids"] = list(self.weak_class_ids)
        return d


@dataclass(frozen=True)
class ImageDetectionScore:
    """Absolute per-image detection quality components for a single detection set."""

    weak_small_recall: float
    weak_small_r50: float
    weak_small_r75: float
    ap50_95: float
    ap75: float
    matched_iou: float
    loc_ap_smooth: float
    fp_pen: float


@dataclass(frozen=True)
class RewardBreakdown:
    """Complete incremental reward & telemetry breakdown for one action trajectory."""

    reward: float
    delta_weak_small_recall: float
    delta_loc_ap_smooth: float
    delta_ap50_95: float
    delta_ap75: float
    delta_matched_iou: float
    delta_fp_pen: float
    cost_pen: float
    num_crops: int

    def to_dict(self) -> Dict[str, float]:
        return {
            "reward": float(self.reward),
            "delta_weak_small_recall": float(self.delta_weak_small_recall),
            "delta_loc_ap_smooth": float(self.delta_loc_ap_smooth),
            "delta_ap50_95": float(self.delta_ap50_95),
            "delta_ap75": float(self.delta_ap75),
            "delta_matched_iou": float(self.delta_matched_iou),
            "delta_fp_pen": float(self.delta_fp_pen),
            "cost_pen": float(self.cost_pen),
            "num_crops": int(self.num_crops),
        }


_AP_IOU_THRESHOLDS: Tuple[float, ...] = (
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
)


def _compute_single_class_all_point_ap(tp_flags: np.ndarray, num_gt: int) -> float:
    """Compute smooth continuous all-point AP for one class on one image."""
    if num_gt <= 0 or tp_flags.size == 0:
        return 0.0
    tp_cum = np.cumsum(tp_flags, dtype=np.float64)
    fp_cum = np.cumsum(1.0 - tp_flags, dtype=np.float64)
    prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    # Monotonic precision envelope from right to left
    prec_env = np.maximum.accumulate(prec[::-1])[::-1]
    return float(np.sum(prec_env[tp_flags > 0.5]) / float(num_gt))


class SmoothDetectionRewarder:
    """Parameter-free smooth incremental rewarder aligning GRPO with AP50/AP75/Weak-Recall/FP/Cost."""

    def __init__(self, config: Optional[GRPORewardConfig] = None) -> None:
        self.config = config if config is not None else GRPORewardConfig()
        self._weak_set = set(int(c) for c in self.config.weak_class_ids)

    def config_dict(self) -> Dict:
        return self.config.to_dict()

    def score_detections(
        self,
        det: Dict[str, Tensor],
        gt_boxes: Tensor,
        gt_labels: Tensor,
        gt_difficult: Tensor,
        orig_size: Tuple[int, int],
    ) -> ImageDetectionScore:
        """Compute smooth per-image absolute quality metrics (all in bounded interpretable scales)."""
        cfg = self.config
        orig_h, orig_w = int(orig_size[0]), int(orig_size[1])
        img_area = float(max(orig_h * orig_w, 1))

        diff_mask = gt_difficult.bool() if gt_difficult.numel() > 0 else torch.zeros((0,), dtype=torch.bool)
        valid_mask = ~diff_mask
        v_boxes = gt_boxes[valid_mask] if gt_boxes.numel() > 0 else torch.zeros((0, 4), dtype=torch.float32)
        v_labels = gt_labels[valid_mask] if gt_labels.numel() > 0 else torch.zeros((0,), dtype=torch.long)
        d_boxes = gt_boxes[diff_mask] if gt_boxes.numel() > 0 else torch.zeros((0, 4), dtype=torch.float32)
        d_labels = gt_labels[diff_mask] if gt_labels.numel() > 0 else torch.zeros((0,), dtype=torch.long)

        n_gt = int(v_boxes.shape[0])
        fp_norm_denom = float(max(n_gt, 2))

        p_boxes_all = det["boxes"].detach().cpu().float()
        p_scores_all = det["scores"].detach().cpu().float()
        p_labels_all = det["labels"].detach().cpu().long()

        # Filter to minimum active threshold
        min_thr = min(cfg.ap_score_thr, cfg.recall_score_thr, cfg.fp_score_thr)
        if p_boxes_all.numel() > 0 and min_thr > 0.0:
            keep = p_scores_all >= min_thr
            p_boxes_all = p_boxes_all[keep]
            p_scores_all = p_scores_all[keep]
            p_labels_all = p_labels_all[keep]

        # Empty-GT image: only FP penalty applies (encourages No-Crop on background images)
        if n_gt == 0:
            fp_sum = 0.0
            if p_boxes_all.numel() > 0:
                for pi in range(p_boxes_all.shape[0]):
                    s = float(p_scores_all[pi].item())
                    if s < cfg.fp_score_thr:
                        continue
                    # Exempt predictions overlapping difficult GTs of same class
                    if d_boxes.numel() > 0:
                        d_same = d_labels == p_labels_all[pi]
                        if d_same.any():
                            if float(box_iou(p_boxes_all[pi : pi + 1], d_boxes[d_same])[0].max().item()) >= 0.50:
                                continue
                    fp_sum += cfg.fp_absent_class_mult * (s ** cfg.fp_score_power)
            return ImageDetectionScore(
                weak_small_recall=0.0,
                weak_small_r50=0.0,
                weak_small_r75=0.0,
                ap50_95=0.0,
                ap75=0.0,
                matched_iou=0.0,
                loc_ap_smooth=0.0,
                fp_pen=fp_sum / fp_norm_denom,
            )

        # No predictions at all
        if p_boxes_all.numel() == 0:
            return ImageDetectionScore(
                weak_small_recall=0.0,
                weak_small_r50=0.0,
                weak_small_r75=0.0,
                ap50_95=0.0,
                ap75=0.0,
                matched_iou=0.0,
                loc_ap_smooth=0.0,
                fp_pen=0.0,
            )

        # Precompute small/weak flags for valid GTs
        gt_areas = ((v_boxes[:, 2] - v_boxes[:, 0]).clamp(min=0.0) * (v_boxes[:, 3] - v_boxes[:, 1]).clamp(min=0.0))
        is_sw_gt = [
            (int(v_labels[gi].item()) in self._weak_set) or (float(gt_areas[gi].item()) / img_area < cfg.small_area_ratio)
            for gi in range(n_gt)
        ]
        total_sw_gt = sum(1 for f in is_sw_gt if f)

        present_classes = sorted(set(int(c) for c in v_labels.tolist()))
        present_cls_set = set(present_classes)

        class_ap_50_95: List[float] = []
        class_ap_75: List[float] = []
        matched_iou_sum = 0.0
        sw_hit_50 = 0
        sw_hit_75 = 0
        fp_pen_raw = 0.0

        for cls_id in present_classes:
            gt_idx_list = [gi for gi in range(n_gt) if int(v_labels[gi].item()) == cls_id]
            c_gt = v_boxes[gt_idx_list]
            n_c_gt = len(gt_idx_list)

            c_diff = d_boxes[d_labels == cls_id] if d_boxes.numel() > 0 else torch.zeros((0, 4), dtype=torch.float32)

            p_mask = p_labels_all == cls_id
            if not p_mask.any():
                class_ap_50_95.append(0.0)
                class_ap_75.append(0.0)
                continue

            c_pb = p_boxes_all[p_mask]
            c_ps = p_scores_all[p_mask]
            order = torch.argsort(c_ps, descending=True, stable=True)
            c_pb = c_pb[order]
            c_ps = c_ps[order]
            n_pred = int(c_pb.shape[0])

            iou_mat = box_iou(c_pb, c_gt).numpy()  # [n_pred, n_c_gt]
            diff_iou_max = (
                box_iou(c_pb, c_diff).max(dim=1).values.numpy()
                if c_diff.numel() > 0
                else np.zeros((n_pred,), dtype=np.float32)
            )
            ps_np = c_ps.numpy()

            # 1) Multi-IoU (0.50:0.05:0.95) 1-to-1 matched all-point AP
            ap_mask = ps_np >= cfg.ap_score_thr
            if np.any(ap_mask):
                iou_ap = iou_mat[ap_mask]
                diff_ap = diff_iou_max[ap_mask]
                n_ap = int(iou_ap.shape[0])
                aps_t: List[float] = []
                for thr in _AP_IOU_THRESHOLDS:
                    matched_g = np.zeros(n_c_gt, dtype=bool)
                    tp_list: List[float] = []
                    for pi in range(n_ap):
                        row = iou_ap[pi]
                        # Choose highest-IoU unmatched GT
                        best_gi = -1
                        best_v = -1.0
                        for gi in range(n_c_gt):
                            if (not matched_g[gi]) and row[gi] > best_v:
                                best_v = float(row[gi])
                                best_gi = gi
                        if best_v >= thr and best_gi >= 0:
                            matched_g[best_gi] = True
                            tp_list.append(1.0)
                        elif diff_ap[pi] >= thr:
                            # VOC ignore difficult GT (neither TP nor FP)
                            continue
                        else:
                            tp_list.append(0.0)
                    ap_val = _compute_single_class_all_point_ap(np.asarray(tp_list, dtype=np.float64), n_c_gt)
                    aps_t.append(ap_val)
                class_ap_50_95.append(float(np.mean(aps_t)))
                class_ap_75.append(float(aps_t[5]))  # index 5 corresponds to IoU=0.75
            else:
                class_ap_50_95.append(0.0)
                class_ap_75.append(0.0)

            # 2) Continuous 1-to-1 matched IoU (IoU >= 0.50), Weak/Small Recall, and present-class FP penalty
            matched_g50 = np.zeros(n_c_gt, dtype=bool)
            for pi in range(n_pred):
                s = float(ps_np[pi])
                row = iou_mat[pi]
                best_gi = -1
                best_v = -1.0
                for gi in range(n_c_gt):
                    if (not matched_g50[gi]) and row[gi] > best_v:
                        best_v = float(row[gi])
                        best_gi = gi

                if s >= cfg.recall_score_thr and best_v >= 0.50 and best_gi >= 0:
                    matched_g50[best_gi] = True
                    matched_iou_sum += best_v
                    orig_gi = gt_idx_list[best_gi]
                    if is_sw_gt[orig_gi]:
                        sw_hit_50 += 1
                        if best_v >= 0.75:
                            sw_hit_75 += 1
                else:
                    # Unmatched prediction: penalize if high-enough confidence and not on a difficult GT
                    if s >= cfg.fp_score_thr and diff_iou_max[pi] < 0.50:
                        fp_pen_raw += s ** cfg.fp_score_power

        # 3) Absent-class FP penalty (predictions for classes with 0 valid GTs in this image)
        for pi in range(p_boxes_all.shape[0]):
            cid = int(p_labels_all[pi].item())
            if cid in present_cls_set:
                continue
            s = float(p_scores_all[pi].item())
            if s < cfg.fp_score_thr:
                continue
            if d_boxes.numel() > 0:
                d_same = d_labels == cid
                if d_same.any():
                    if float(box_iou(p_boxes_all[pi : pi + 1], d_boxes[d_same])[0].max().item()) >= 0.50:
                        continue
            fp_pen_raw += cfg.fp_absent_class_mult * (s ** cfg.fp_score_power)

        ap50_95 = float(np.mean(class_ap_50_95)) if class_ap_50_95 else 0.0
        ap75 = float(np.mean(class_ap_75)) if class_ap_75 else 0.0
        matched_iou = matched_iou_sum / float(n_gt)
        loc_ap_smooth = (
            cfg.loc_w_ap50_95 * ap50_95
            + cfg.loc_w_ap75 * ap75
            + cfg.loc_w_matched_iou * matched_iou
        )

        if total_sw_gt > 0:
            sw_r50 = sw_hit_50 / float(total_sw_gt)
            sw_r75 = sw_hit_75 / float(total_sw_gt)
            sw_rec = cfg.recall_w_50 * sw_r50 + cfg.recall_w_75 * sw_r75
        else:
            sw_r50 = 0.0
            sw_r75 = 0.0
            sw_rec = 0.0

        return ImageDetectionScore(
            weak_small_recall=sw_rec,
            weak_small_r50=sw_r50,
            weak_small_r75=sw_r75,
            ap50_95=ap50_95,
            ap75=ap75,
            matched_iou=matched_iou,
            loc_ap_smooth=loc_ap_smooth,
            fp_pen=fp_pen_raw / fp_norm_denom,
        )

    def compute_incremental_reward(
        self,
        fused_det: Dict[str, Tensor],
        global_score: ImageDetectionScore,
        action: Sequence[int],
        gt_boxes: Tensor,
        gt_labels: Tensor,
        gt_difficult: Tensor,
        orig_size: Tuple[int, int],
    ) -> RewardBreakdown:
        """Compute incremental reward R(a) relative to No-Crop baseline (strictly R(()) == 0.0)."""
        num_crops = len(action)
        if num_crops == 0:
            return RewardBreakdown(
                reward=0.0,
                delta_weak_small_recall=0.0,
                delta_loc_ap_smooth=0.0,
                delta_ap50_95=0.0,
                delta_ap75=0.0,
                delta_matched_iou=0.0,
                delta_fp_pen=0.0,
                cost_pen=0.0,
                num_crops=0,
            )

        cfg = self.config
        act_score = self.score_detections(fused_det, gt_boxes, gt_labels, gt_difficult, orig_size)
        d_rec = act_score.weak_small_recall - global_score.weak_small_recall
        d_loc = act_score.loc_ap_smooth - global_score.loc_ap_smooth
        d_ap = act_score.ap50_95 - global_score.ap50_95
        d_ap75 = act_score.ap75 - global_score.ap75
        d_iou = act_score.matched_iou - global_score.matched_iou
        d_fp = act_score.fp_pen - global_score.fp_pen
        c_pen = cfg.lambda_cost * float(num_crops)

        reward = (
            cfg.lambda_recall * d_rec
            + cfg.lambda_loc_ap * d_loc
            - cfg.lambda_fp * d_fp
            - c_pen
        )
        return RewardBreakdown(
            reward=float(reward),
            delta_weak_small_recall=float(d_rec),
            delta_loc_ap_smooth=float(d_loc),
            delta_ap50_95=float(d_ap),
            delta_ap75=float(d_ap75),
            delta_matched_iou=float(d_iou),
            delta_fp_pen=float(d_fp),
            cost_pen=float(c_pen),
            num_crops=num_crops,
        )


# =============================================================================
# Isolated GRPO Interactive Detection Environment
# =============================================================================


class GRPODetectionEnvironment:
    """P0-01 isolated, symmetry-aware, persistently cached GRPO interaction environment.

    Strictly enforces Single-DGOD source-only data access (``daytime_clear/train``).
    Lazily executes and persists missing ``global_detections`` and isolated ``batch_size=1``
    ``crop_detections[wid]``, and reuses them across all epochs and hyperparameter sweeps
    with zero duplicate detector inference.
    """

    def __init__(
        self,
        loader: SourceOnlyDatasetLoader,
        env_cache: PersistentDetectionEnvCache,
        grid: CandidateGrid,
        fuser: DualViewWBFFusion,
        rewarder: Optional[SmoothDetectionRewarder] = None,
        wrapper: Optional[FrozenTextStyleDetWrapper] = None,
        class_names: Optional[Sequence[str]] = None,
        k_max: int = 2,
    ) -> None:
        validate_source_only_access(loader.domain, loader.split)
        self.loader = loader
        self.env_cache = env_cache
        self.grid = grid
        self.fuser = fuser
        self.rewarder = rewarder if rewarder is not None else SmoothDetectionRewarder()
        self.wrapper = wrapper
        self.k_max = int(k_max)

        if class_names is not None:
            self.class_names: Tuple[str, ...] = tuple(class_names)
        elif wrapper is not None:
            self.class_names = tuple(wrapper.class_names)
        else:
            # Default SKY7 profile class order (1-indexed: car, person, truck, bike, bus, rider, motor)
            self.class_names = ("car", "person", "truck", "bike", "bus", "rider", "motor")

        # Per-process lightweight RAM caches for parsed GT, baseline scores, and canonical fused outputs
        self._gt_cache: Dict[str, Tuple[Tensor, Tensor, Tensor]] = {}
        self._global_score_cache: Dict[ViewKey, ImageDetectionScore] = {}
        self._reward_memo: Dict[Tuple[ViewKey, Tuple[int, ...]], RewardBreakdown] = {}
        self._last_pil_key: Optional[ViewKey] = None
        self._last_pil_img: Optional[Image.Image] = None

        # Telemetry counters (current session + prior resumed cumulative baseline for P1-01)
        self.global_cache_hits: int = 0
        self.global_cache_misses: int = 0
        self.crop_cache_hits: int = 0
        self.crop_cache_misses: int = 0
        self.detector_compute_sec: float = 0.0
        self._prior_global_hits: int = 0
        self._prior_global_misses: int = 0
        self._prior_crop_hits: int = 0
        self._prior_crop_misses: int = 0
        self._prior_detector_compute_sec: float = 0.0

    def load_prior_telemetry(self, counters: Optional[Dict]) -> None:
        """Restore cumulative telemetry counters from a previous resumed training checkpoint."""
        if not counters:
            return
        self._prior_global_hits = int(counters.get("global_cache_hits", 0))
        self._prior_global_misses = int(counters.get("global_cache_misses", 0))
        self._prior_crop_hits = int(counters.get("crop_cache_hits", 0))
        self._prior_crop_misses = int(counters.get("crop_cache_misses", 0))
        self._prior_detector_compute_sec = float(counters.get("detector_compute_sec", 0.0))

    def export_cumulative_counters(self) -> Dict:
        """Return exact cumulative (prior + session) telemetry counters for atomic checkpointing."""
        return {
            "global_cache_hits": int(self._prior_global_hits + self.global_cache_hits),
            "global_cache_misses": int(self._prior_global_misses + self.global_cache_misses),
            "crop_cache_hits": int(self._prior_crop_hits + self.crop_cache_hits),
            "crop_cache_misses": int(self._prior_crop_misses + self.crop_cache_misses),
            "detector_compute_sec": float(self._prior_detector_compute_sec + self.detector_compute_sec),
        }

    def clear_reward_memo(self) -> None:
        """Clear in-memory reward memo (call if rewarder config is swapped)."""
        self._global_score_cache.clear()
        self._reward_memo.clear()

    def _load_view_pil(self, view_key: ViewKey) -> Image.Image:
        if self._last_pil_key == view_key and self._last_pil_img is not None:
            return self._last_pil_img
        img_id, deg_tag, deg_seed = view_key
        if deg_tag not in DEGRADATION_TAGS:
            raise ValueError(f"Invalid degradation tag: {deg_tag!r}")
        base_pil = self.loader.load_image(img_id)
        view_pil = apply_physical_degradation(
            base_pil,
            image_id=img_id,
            view_tag=deg_tag,
            seed=int(deg_seed),
        )
        self._last_pil_key = view_key
        self._last_pil_img = view_pil
        return view_pil

    def get_gt(self, image_id: str) -> Tuple[Tensor, Tensor, Tensor]:
        cached = self._gt_cache.get(image_id)
        if cached is not None:
            return cached
        objs = self.loader.parse_xml(image_id)
        gt = record_to_0based_gt(objs, self.class_names)
        self._gt_cache[image_id] = gt
        return gt

    def get_or_compute_global(self, view_key: ViewKey) -> Tuple[Dict[str, Tensor], Tuple[int, int]]:
        cached = self.env_cache.get_global(view_key)
        if cached is not None:
            self.global_cache_hits += 1
            return cached

        if self.wrapper is None:
            raise RuntimeError(
                f"Global detection cache miss for {view_key} and no FrozenTextStyleDetWrapper is attached."
            )
        if self.wrapper.model.training:
            raise RuntimeError("FrozenTextStyleDetWrapper.model MUST remain in eval() mode!")

        self.global_cache_misses += 1
        view_pil = self._load_view_pil(view_key)
        t0 = time.perf_counter()
        with torch.no_grad():
            perc = self.wrapper.forward_global(view_pil)
        self.detector_compute_sec += time.perf_counter() - t0
        self.env_cache.put_global(view_key, perc.detections, perc.orig_size)
        return perc.detections, perc.orig_size

    def get_or_compute_crop(
        self,
        view_key: ViewKey,
        wid: int,
        orig_size: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Tensor]:
        wid_i = int(wid)
        if wid_i < 0 or wid_i >= self.grid.num_windows:
            raise ValueError(f"Window ID {wid_i} out of range [0, {self.grid.num_windows - 1}]")

        cached = self.env_cache.get_crop(view_key, wid_i)
        if cached is not None:
            self.crop_cache_hits += 1
            return cached

        if self.wrapper is None:
            raise RuntimeError(
                f"Crop detection cache miss for ({view_key}, wid={wid_i}) and no wrapper is attached."
            )
        if self.wrapper.model.training:
            raise RuntimeError("FrozenTextStyleDetWrapper.model MUST remain in eval() mode!")

        self.crop_cache_misses += 1
        view_pil = self._load_view_pil(view_key)
        orig_w, orig_h = view_pil.size
        if orig_size is not None and (orig_h, orig_w) != (int(orig_size[0]), int(orig_size[1])):
            raise RuntimeError(
                f"Image orig_size mismatch for {view_key}: loaded {(orig_h, orig_w)} != expected {orig_size}"
            )
        rect = self.grid.windows[wid_i].to_pixel_xyxy(orig_w, orig_h)
        t0 = time.perf_counter()
        with torch.no_grad():
            # Strictly isolated batch_size=1 single-crop forward (P0-01 compliant)
            crop_det = self.wrapper.forward_single_crop(view_pil, rect)
        self.detector_compute_sec += time.perf_counter() - t0
        self.env_cache.put_crop(view_key, wid_i, crop_det)
        return crop_det

    def fuse_action(
        self,
        view_key: ViewKey,
        action: Sequence[int],
    ) -> Tuple[Dict[str, Tensor], Tuple[int, int]]:
        """Fuse global + canonical unordered crop detections for ``action`` (exact (w1,w2)==(w2,w1) symmetry)."""
        canon_wids = canonicalize_action_windows(action, num_windows=self.grid.num_windows, k_max=self.k_max)
        global_det, orig_size = self.get_or_compute_global(view_key)
        if len(canon_wids) == 0:
            return self.fuser.fuse(global_det, [], orig_size), orig_size

        crop_dets = [self.get_or_compute_crop(view_key, wid, orig_size=orig_size) for wid in canon_wids]
        fused = self.fuser.fuse(global_det, crop_dets, orig_size)
        return fused, orig_size

    def get_global_score(self, view_key: ViewKey) -> ImageDetectionScore:
        cached = self._global_score_cache.get(view_key)
        if cached is not None:
            return cached
        global_det, orig_size = self.get_or_compute_global(view_key)
        base_det = self.fuser.fuse(global_det, [], orig_size)
        gt_boxes, gt_labels, gt_diff = self.get_gt(view_key[0])
        score = self.rewarder.score_detections(base_det, gt_boxes, gt_labels, gt_diff, orig_size)
        self._global_score_cache[view_key] = score
        return score

    def evaluate_action_reward(
        self,
        view_key: ViewKey,
        action: Sequence[int],
    ) -> RewardBreakdown:
        """Evaluate smooth incremental reward R(action) for one (view_key, action) pair."""
        canon_wids = canonicalize_action_windows(action, num_windows=self.grid.num_windows, k_max=self.k_max)
        if len(canon_wids) == 0:
            return RewardBreakdown(
                reward=0.0,
                delta_weak_small_recall=0.0,
                delta_loc_ap_smooth=0.0,
                delta_ap50_95=0.0,
                delta_ap75=0.0,
                delta_matched_iou=0.0,
                delta_fp_pen=0.0,
                cost_pen=0.0,
                num_crops=0,
            )

        memo_key = (view_key, canon_wids)
        cached_rb = self._reward_memo.get(memo_key)
        if cached_rb is not None:
            return cached_rb

        global_score = self.get_global_score(view_key)
        fused_det, orig_size = self.fuse_action(view_key, canon_wids)
        gt_boxes, gt_labels, gt_diff = self.get_gt(view_key[0])
        rb = self.rewarder.compute_incremental_reward(
            fused_det=fused_det,
            global_score=global_score,
            action=canon_wids,
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            gt_difficult=gt_diff,
            orig_size=orig_size,
        )
        self._reward_memo[memo_key] = rb
        return rb

    def evaluate_group_rewards(
        self,
        view_keys: Sequence[ViewKey],
        grouped_actions: Sequence[Sequence[ActionTrajectory]],
    ) -> Tuple[Tensor, List[List[RewardBreakdown]]]:
        """Evaluate rewards [B, G] for a GRPO rollout batch, deduplicating identical canonical sets."""
        bsz = len(view_keys)
        if len(grouped_actions) != bsz:
            raise ValueError(f"grouped_actions length {len(grouped_actions)} != view_keys length {bsz}")
        if bsz == 0:
            raise ValueError("Empty batch passed to evaluate_group_rewards.")
        g_size = len(grouped_actions[0])

        rewards_mat = torch.zeros((bsz, g_size), dtype=torch.float32)
        breakdowns: List[List[RewardBreakdown]] = []

        for b in range(bsz):
            vk = view_keys[b]
            grp = grouped_actions[b]
            if len(grp) != g_size:
                raise ValueError(f"Inconsistent group size at b={b}: {len(grp)} != {g_size}")
            row_bd: List[RewardBreakdown] = []
            for g, act in enumerate(grp):
                rb = self.evaluate_action_reward(vk, act)
                rewards_mat[b, g] = rb.reward
                row_bd.append(rb)
            breakdowns.append(row_bd)

        return rewards_mat, breakdowns

    def precache_global_views(
        self,
        view_keys: Sequence[ViewKey],
        flush_every: int = 256,
        show_progress: bool = True,
    ) -> Dict:
        """Offline helper to pre-populate ``global_detections`` for a list of source views."""
        t0 = time.perf_counter()
        computed = 0
        iterator = tqdm(view_keys, desc="Env Pre-Cache (Global)", leave=True) if show_progress else view_keys
        for i, vk in enumerate(iterator, start=1):
            if not self.env_cache.has_global(vk):
                self.get_or_compute_global(vk)
                computed += 1
            if flush_every > 0 and (i % flush_every == 0):
                self.env_cache.flush()
        stats = self.env_cache.flush()
        return {
            "requested_views": len(view_keys),
            "newly_computed_globals": computed,
            "elapsed_sec": round(time.perf_counter() - t0, 2),
            "cache_stats": stats,
        }

    def telemetry_dict(self) -> Dict:
        cum = self.export_cumulative_counters()
        cg_h, cg_m = cum["global_cache_hits"], cum["global_cache_misses"]
        cc_h, cc_m = cum["crop_cache_hits"], cum["crop_cache_misses"]
        return {
            "telemetry_scope": "full_training_cumulative",
            "global_cache_hits": cg_h,
            "global_cache_misses": cg_m,
            "crop_cache_hits": cc_h,
            "crop_cache_misses": cc_m,
            "global_hit_rate_pct": round(100.0 * cg_h / max(cg_h + cg_m, 1), 2),
            "crop_hit_rate_pct": round(100.0 * cc_h / max(cc_h + cc_m, 1), 2),
            "detector_compute_sec": round(cum["detector_compute_sec"], 2),
            "session_telemetry": {
                "global_cache_hits": int(self.global_cache_hits),
                "global_cache_misses": int(self.global_cache_misses),
                "crop_cache_hits": int(self.crop_cache_hits),
                "crop_cache_misses": int(self.crop_cache_misses),
                "global_hit_rate_pct": round(
                    100.0 * self.global_cache_hits / max(self.global_cache_hits + self.global_cache_misses, 1), 2
                ),
                "crop_hit_rate_pct": round(
                    100.0 * self.crop_cache_hits / max(self.crop_cache_hits + self.crop_cache_misses, 1), 2
                ),
                "detector_compute_sec": round(self.detector_compute_sec, 2),
            },
            "env_cache_stats": self.env_cache.stats(),
        }
