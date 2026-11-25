"""Strict Source-Only Data Loader & Deterministic Physical Synthetic Degradation Pipeline.

Enforces Single-DGOD zero-leakage compliance for Phase 2 (SFT) & Phase 3 (GRPO):
  - Hard-locks data access strictly to ``domain == "daytime_clear"`` and ``split == "train"``.
    Passing ANY target domain (``daytime_foggy``, ``dusk_rainy``, ``Night-Sunny``,
    ``night_rainy``) or any non-``train`` split (including ``daytime_clear/test``) immediately
    raises ``ValueError``.
  - Deterministically partitions ``daytime_clear/train`` (19,395 images) into strictly
    disjoint ``source_train`` and ``source_dev`` splits via cryptographic SHA256 image-ID
    ordering, emitting SHA256 split fingerprints.
  - Parses VOC XML annotations strictly on demand in memory for only the requested IDs
    (zero ``.pkl`` cache writes to ``E:\\textstyleDet\\data\\sky``; avoids scanning all
    19,395 XML files during subsampling or debugging).
  - Provides 5 purely physics-based, target-domain-free, 100% reproducible synthetic
    degradation operators keyed on ``(image_id, view_tag, seed)`` that preserve exact
    ``(orig_w, orig_h)`` pixel dimensions and bounding-box geometry:
      1. ``clean``            : identity pass-through (No-Crop calibration views)
      2. ``low_light``        : gamma darkening [1.8, 3.2] + gain drop + cool tint + sensor noise
      3. ``fog``              : Koschmieder atmospheric light veil + vertical haze gradient + low contrast
      4. ``rain``             : multi-layer slanted rain streaks + overcast contrast attenuation
      5. ``compound_adverse`` : composite adverse weather (low_light+rain OR fog+low_contrast)
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .eval_utils import resolve_sky_domain_root
from .runtime_shim import DEFAULT_RN_ROOT, ensure_textstyle_rn_importable

ALLOWED_SOURCE_DOMAIN: str = "daytime_clear"
ALLOWED_SOURCE_SPLIT: str = "train"

DEGRADATION_TAGS: Tuple[str, ...] = (
    "clean",
    "low_light",
    "fog",
    "rain",
    "compound_adverse",
)
ADVERSE_DEGRADATION_TAGS: Tuple[str, ...] = (
    "low_light",
    "fog",
    "rain",
    "compound_adverse",
)

DEGRADATION_RECIPE_VERSION: int = 1


def validate_source_only_access(domain: str, split: str) -> None:
    """Hard Single-DGOD guard: allow ONLY ``daytime_clear`` + ``train``."""
    if not isinstance(domain, str) or domain != ALLOWED_SOURCE_DOMAIN:
        raise ValueError(
            f"Single-DGOD source-only violation: Phase 2/3 training & dev data MUST use "
            f"domain={ALLOWED_SOURCE_DOMAIN!r}, got domain={domain!r}. "
            "Target domains are strictly forbidden."
        )
    if not isinstance(split, str) or split != ALLOWED_SOURCE_SPLIT:
        raise ValueError(
            f"Single-DGOD source-only violation: Phase 2/3 training & dev data MUST use "
            f"split={ALLOWED_SOURCE_SPLIT!r}, got split={split!r}. "
            "Hold-out test splits (including daytime_clear/test) are strictly forbidden."
        )


def _sha256_id_list(image_ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(image_ids).encode("utf-8")).hexdigest()


def _hash_rank_key(image_id: str, split_seed: int) -> Tuple[str, str]:
    digest = hashlib.sha256(f"split|{split_seed}|{image_id}".encode("utf-8")).hexdigest()
    return (digest, image_id)


def derive_view_seed(image_id: str, view_tag: str, base_seed: int) -> int:
    """Derive a deterministic 63-bit integer seed from ``(image_id, view_tag, base_seed)``."""
    payload = f"deg_v{DEGRADATION_RECIPE_VERSION}|{int(base_seed)}|{image_id}|{view_tag}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) & ((1 << 63) - 1)


def select_adverse_tag_for_view(
    image_id: str,
    view_index: int,
    base_seed: int,
    adverse_tags: Sequence[str] = ADVERSE_DEGRADATION_TAGS,
) -> str:
    """Deterministically pick an adverse degradation tag for ``(image_id, view_index)``.

    Uses a cryptographic permutation of ``adverse_tags`` keyed on ``(base_seed, image_id)``
    so multiple degraded views of the same image get distinct adverse weather types and
    the corpus-wide distribution across tags is closely balanced (~25% each).
    """
    tags = tuple(adverse_tags)
    if not tags:
        raise ValueError("adverse_tags must be non-empty.")
    for t in tags:
        if t not in ADVERSE_DEGRADATION_TAGS:
            raise ValueError(f"Unknown adverse degradation tag: {t!r}")
    if view_index < 0:
        raise ValueError(f"view_index must be >= 0, got {view_index}")

    perm_seed = derive_view_seed(image_id=image_id, view_tag="__adverse_perm__", base_seed=base_seed)
    rng = np.random.default_rng(perm_seed)
    perm = rng.permutation(len(tags)).tolist()
    return tags[perm[view_index % len(tags)]]


def degradation_recipe_config(
    base_seed: int = 20260420,
    include_clean: bool = True,
    degraded_views_per_image: int = 1,
) -> Dict:
    """Return canonical immutable degradation recipe spec for cache manifest hashing."""
    return {
        "recipe_version": DEGRADATION_RECIPE_VERSION,
        "base_seed": int(base_seed),
        "include_clean": bool(include_clean),
        "degraded_views_per_image": int(degraded_views_per_image),
        "degradation_tags": list(DEGRADATION_TAGS),
        "adverse_tags": list(ADVERSE_DEGRADATION_TAGS),
        "operators": {
            "clean": {"mode": "identity"},
            "low_light": {
                "gamma_range": [1.8, 3.2],
                "gain_range": [0.55, 0.85],
                "cool_tint_rgb_ranges": [[0.84, 0.94], [0.92, 1.00], [1.04, 1.16]],
                "read_noise_sigma_range": [0.008, 0.022],
                "shot_noise_sigma_range": [0.005, 0.015],
            },
            "fog": {
                "beta_range": [0.85, 2.15],
                "airlight_range": [0.72, 0.92],
                "min_transmission_range": [0.24, 0.46],
                "vertical_horizon_weight": [0.35, 1.0],
                "contrast_atten_range": [0.82, 0.94],
            },
            "rain": {
                "overcast_gain_range": [0.80, 0.95],
                "contrast_atten_range": [0.80, 0.94],
                "overcast_tint_rgb_ranges": [[0.92, 0.98], [0.95, 1.00], [1.01, 1.07]],
                "far_streak_count_range": [450, 900],
                "near_streak_count_range": [180, 420],
                "angle_deg_range": [-18.0, 18.0],
                "far_streak_length_px_range": [12.0, 24.0],
                "near_streak_length_px_range": [20.0, 36.0],
                "streak_alpha_range": [0.16, 0.34],
                "streak_blur_radius": 0.75,
                "streak_color_rgb": [0.82, 0.86, 0.92],
            },
            "compound_adverse": {
                "sub_modes": ["low_light_rain", "fog_low_contrast"],
                "low_light_rain": {"gamma_range": [1.7, 2.8], "gain_range": [0.58, 0.84]},
                "fog_low_contrast": {
                    "gamma_range": [1.15, 1.55],
                    "gain_range": [0.76, 0.92],
                    "beta_range": [0.95, 2.25],
                    "contrast_atten_range": [0.72, 0.86],
                },
            },
        },
        "target_domain_stats_used": False,
    }


@dataclass(frozen=True)
class SourceSplitInfo:
    """Deterministic source-only train/dev split specification & SHA256 fingerprints."""

    domain: str
    split: str
    voc_root: Path
    dev_ratio: float
    split_seed: int
    total_valid_images: int
    full_source_train_ids: Tuple[str, ...]
    full_source_dev_ids: Tuple[str, ...]
    selected_train_ids: Tuple[str, ...]
    selected_dev_ids: Tuple[str, ...]
    full_source_train_sha256: str
    full_source_dev_sha256: str
    selected_train_sha256: str
    selected_dev_sha256: str

    def config_dict(self) -> Dict:
        return {
            "domain": self.domain,
            "split": self.split,
            "voc_root": str(self.voc_root),
            "dev_ratio": self.dev_ratio,
            "split_seed": self.split_seed,
            "total_valid_images": self.total_valid_images,
            "full_source_train_count": len(self.full_source_train_ids),
            "full_source_dev_count": len(self.full_source_dev_ids),
            "selected_train_count": len(self.selected_train_ids),
            "selected_dev_count": len(self.selected_dev_ids),
            "full_source_train_sha256": self.full_source_train_sha256,
            "full_source_dev_sha256": self.full_source_dev_sha256,
            "selected_train_sha256": self.selected_train_sha256,
            "selected_dev_sha256": self.selected_dev_sha256,
        }


class SourceOnlyDatasetLoader:
    """Strictly source-only (daytime_clear/train) splitter and on-demand XML loader."""

    def __init__(
        self,
        data_dir: str | Path,
        domain: str = ALLOWED_SOURCE_DOMAIN,
        split: str = ALLOWED_SOURCE_SPLIT,
        dev_ratio: float = 0.10,
        split_seed: int = 202604,
        rn_root: str | Path = DEFAULT_RN_ROOT,
    ) -> None:
        validate_source_only_access(domain, split)
        if not (0.01 <= float(dev_ratio) <= 0.50):
            raise ValueError(f"dev_ratio must be in [0.01, 0.50], got {dev_ratio}")

        self.data_dir = Path(data_dir).resolve()
        self.domain = domain
        self.split = split
        self.dev_ratio = float(dev_ratio)
        self.split_seed = int(split_seed)
        self.rn_root = ensure_textstyle_rn_importable(rn_root)

        self.voc_root = resolve_sky_domain_root(self.data_dir, self.domain).resolve()
        expected_domain_dir = (self.data_dir / ALLOWED_SOURCE_DOMAIN).resolve()
        if expected_domain_dir not in (self.voc_root, *self.voc_root.parents):
            raise ValueError(
                f"Resolved VOC root {self.voc_root} is outside {expected_domain_dir}"
            )

        self.img_dir = self.voc_root / "JPEGImages"
        self.anno_dir = self.voc_root / "Annotations"
        self.split_file = self.voc_root / "ImageSets" / "Main" / f"{self.split}.txt"
        if not self.split_file.is_file():
            raise FileNotFoundError(f"Source split file missing: {self.split_file}")

    def build_splits(
        self,
        max_train_images: int = 0,
        max_dev_images: int = 0,
    ) -> SourceSplitInfo:
        """Deterministically partition ``daytime_clear/train`` into disjoint train/dev IDs.

        Does NOT parse any XML files; verifies file existence fast and hashes IDs deterministically.
        """
        validate_source_only_access(self.domain, self.split)
        if max_train_images < 0 or max_dev_images < 0:
            raise ValueError(
                f"max_train_images and max_dev_images must be >= 0, got {(max_train_images, max_dev_images)}"
            )

        raw_ids = [
            line.strip()
            for line in self.split_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        # Preserve unique valid IDs with existing JPG & XML
        seen = set()
        valid_ids: List[str] = []
        for img_id in raw_ids:
            if img_id in seen:
                continue
            seen.add(img_id)
            if (self.img_dir / f"{img_id}.jpg").is_file() and (self.anno_dir / f"{img_id}.xml").is_file():
                valid_ids.append(img_id)

        if len(valid_ids) < 2:
            raise RuntimeError(f"Insufficient valid images in {self.split_file}: {len(valid_ids)}")

        # Cryptographic deterministic permutation by SHA256(split_seed, image_id)
        ranked_ids = sorted(valid_ids, key=lambda iid: _hash_rank_key(iid, self.split_seed))
        n_dev = max(1, int(round(len(ranked_ids) * self.dev_ratio)))
        if n_dev >= len(ranked_ids):
            n_dev = len(ranked_ids) - 1

        full_dev = tuple(ranked_ids[:n_dev])
        full_train = tuple(ranked_ids[n_dev:])

        # Hard zero-overlap invariant
        if not set(full_train).isdisjoint(set(full_dev)):
            raise AssertionError("FATAL: source_train and source_dev overlap!")

        sel_train = full_train if max_train_images == 0 else full_train[: min(max_train_images, len(full_train))]
        sel_dev = full_dev if max_dev_images == 0 else full_dev[: min(max_dev_images, len(full_dev))]

        if not set(sel_train).isdisjoint(set(sel_dev)):
            raise AssertionError("FATAL: selected source_train and source_dev overlap!")

        return SourceSplitInfo(
            domain=self.domain,
            split=self.split,
            voc_root=self.voc_root,
            dev_ratio=self.dev_ratio,
            split_seed=self.split_seed,
            total_valid_images=len(valid_ids),
            full_source_train_ids=full_train,
            full_source_dev_ids=full_dev,
            selected_train_ids=sel_train,
            selected_dev_ids=sel_dev,
            full_source_train_sha256=_sha256_id_list(full_train),
            full_source_dev_sha256=_sha256_id_list(full_dev),
            selected_train_sha256=_sha256_id_list(sel_train),
            selected_dev_sha256=_sha256_id_list(sel_dev),
        )

    @staticmethod
    def _validate_image_id(image_id: str) -> str:
        """Reject path separators, parent traversal, or malformed image IDs (P3-01)."""
        if (
            not isinstance(image_id, str)
            or not image_id
            or "/" in image_id
            or "\\" in image_id
            or ".." in image_id
            or Path(image_id).name != image_id
        ):
            raise ValueError(f"Unsafe or malformed image_id (path traversal forbidden): {image_id!r}")
        return image_id

    def image_path(self, image_id: str) -> Path:
        safe_id = self._validate_image_id(image_id)
        p = (self.img_dir / f"{safe_id}.jpg").resolve()
        if self.img_dir.resolve() not in p.parents:
            raise ValueError(f"Resolved image path escapes source img_dir: {p}")
        return p

    def xml_path(self, image_id: str) -> Path:
        safe_id = self._validate_image_id(image_id)
        p = (self.anno_dir / f"{safe_id}.xml").resolve()
        if self.anno_dir.resolve() not in p.parents:
            raise ValueError(f"Resolved annotation path escapes source anno_dir: {p}")
        return p

    def load_image(self, image_id: str) -> Image.Image:
        path = self.image_path(image_id)
        if not path.is_file():
            raise FileNotFoundError(f"Source image missing: {path}")
        with Image.open(path) as img:
            return img.convert("RGB")

    def parse_xml(self, image_id: str) -> List[Dict]:
        """Parse a single source XML annotation in memory (zero .pkl written)."""
        from eval.voc_eval import parse_rec

        path = self.xml_path(image_id)
        if not path.is_file():
            raise FileNotFoundError(f"Source annotation missing: {path}")
        return parse_rec(str(path))

    def parse_records_on_demand(self, image_ids: Sequence[str]) -> Dict[str, List[Dict]]:
        """Parse VOC XML annotations ONLY for the requested ``image_ids`` (in memory, zero .pkl)."""
        validate_source_only_access(self.domain, self.split)
        from eval.voc_eval import parse_rec

        records: Dict[str, List[Dict]] = {}
        for img_id in image_ids:
            path = self.xml_path(img_id)
            if not path.is_file():
                raise FileNotFoundError(f"Source annotation missing: {path}")
            records[img_id] = parse_rec(str(path))
        return records


# =============================================================================
# Deterministic Physics-Based Synthetic Degradation Operators
# =============================================================================


def _apply_low_light_np(
    arr_f: np.ndarray,
    rng: np.random.Generator,
    gamma_range: Tuple[float, float] = (1.8, 3.2),
    gain_range: Tuple[float, float] = (0.55, 0.85),
) -> Tuple[np.ndarray, Dict]:
    """Gamma darkening + attenuation + cool color-temperature shift + mild sensor noise."""
    gamma = float(rng.uniform(gamma_range[0], gamma_range[1]))
    gain = float(rng.uniform(gain_range[0], gain_range[1]))
    tint = np.array(
        [
            float(rng.uniform(0.84, 0.94)),
            float(rng.uniform(0.92, 1.00)),
            float(rng.uniform(1.04, 1.16)),
        ],
        dtype=np.float32,
    )
    sigma_read = float(rng.uniform(0.008, 0.022))
    sigma_shot = float(rng.uniform(0.005, 0.015))

    out = np.power(np.clip(arr_f, 0.0, 1.0), gamma, dtype=np.float32)
    out = out * (gain * tint[None, None, :])

    # Mild heteroscedastic shot + read sensor noise
    noise_std = np.sqrt(sigma_read * sigma_read + (sigma_shot * sigma_shot) * np.clip(out, 0.0, 1.0))
    noise = rng.standard_normal(size=out.shape, dtype=np.float32) * noise_std
    out = np.clip(out + noise, 0.0, 1.0)
    params = {
        "gamma": round(gamma, 4),
        "gain": round(gain, 4),
        "tint": [round(float(x), 4) for x in tint.tolist()],
        "sigma_read": round(sigma_read, 5),
        "sigma_shot": round(sigma_shot, 5),
    }
    return out, params


def _apply_fog_np(
    arr_f: np.ndarray,
    rng: np.random.Generator,
    beta_range: Tuple[float, float] = (0.85, 2.15),
    contrast_range: Tuple[float, float] = (0.82, 0.94),
) -> Tuple[np.ndarray, Dict]:
    """Koschmieder atmospheric light veil + vertical horizon fog gradient + contrast attenuation."""
    h, w, _ = arr_f.shape
    beta = float(rng.uniform(beta_range[0], beta_range[1]))
    airlight_base = float(rng.uniform(0.72, 0.92))
    airlight_tint = np.array(
        [
            float(rng.uniform(0.96, 1.02)),
            float(rng.uniform(0.98, 1.02)),
            float(rng.uniform(0.99, 1.05)),
        ],
        dtype=np.float32,
    )
    airlight = np.clip(airlight_base * airlight_tint, 0.65, 0.98)
    t_min = float(rng.uniform(0.24, 0.46))
    contrast_alpha = float(rng.uniform(contrast_range[0], contrast_range[1]))
    horizon_y = float(rng.uniform(0.32, 0.52))

    # Smooth vertical pseudo-depth map: highest fog thickness near sky/horizon, lighter near foreground road
    y_norm = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]  # [H, 1]
    # Above horizon: dense (0.85..1.0); below horizon: decays smoothly toward bottom (0.35)
    below = np.clip((y_norm - horizon_y) / max(1.0 - horizon_y, 1e-4), 0.0, 1.0)
    depth_v = 1.0 - 0.65 * np.power(below, 1.35)  # [H, 1] in [0.35, 1.0]

    # Smooth low-frequency spatial cloudiness field via PIL bilinear upsample (exact & fast)
    low_noise = rng.uniform(0.82, 1.18, size=(12, 20)).astype(np.float32)
    low_pil = Image.fromarray(np.clip((low_noise - 0.82) / 0.36 * 255.0, 0, 255).astype(np.uint8), mode="L")
    try:
        _bilinear = Image.Resampling.BILINEAR
    except AttributeError:
        _bilinear = Image.BILINEAR
    spatial_mod = np.asarray(low_pil.resize((w, h), _bilinear), dtype=np.float32) / 255.0
    spatial_mod = 0.84 + 0.32 * spatial_mod  # [H, W] in [0.84, 1.16]

    depth_map = np.clip(depth_v * spatial_mod, 0.20, 1.25)
    transmission = np.clip(np.exp(-beta * depth_map), t_min, 0.96).astype(np.float32)

    fogged = arr_f * transmission[:, :, None] + airlight[None, None, :] * (1.0 - transmission[:, :, None])
    mean_lum = float(fogged.mean())
    fogged = np.clip((fogged - mean_lum) * contrast_alpha + mean_lum, 0.0, 1.0)

    params = {
        "beta": round(beta, 4),
        "airlight": [round(float(x), 4) for x in airlight.tolist()],
        "t_min": round(t_min, 4),
        "horizon_y": round(horizon_y, 4),
        "contrast_alpha": round(contrast_alpha, 4),
    }
    return fogged, params


def _apply_rain_np(
    arr_f: np.ndarray,
    rng: np.random.Generator,
    apply_overcast: bool = True,
) -> Tuple[np.ndarray, Dict]:
    """Slanted two-layer physical rain streaks + optional overcast contrast attenuation."""
    h, w, _ = arr_f.shape
    out = arr_f.copy()
    overcast_gain = 1.0
    contrast_alpha = 1.0

    if apply_overcast:
        overcast_gain = float(rng.uniform(0.80, 0.95))
        contrast_alpha = float(rng.uniform(0.80, 0.94))
        tint = np.array(
            [
                float(rng.uniform(0.92, 0.98)),
                float(rng.uniform(0.95, 1.00)),
                float(rng.uniform(1.01, 1.07)),
            ],
            dtype=np.float32,
        )
        mean_val = float(out.mean())
        out = (out - mean_val) * contrast_alpha + mean_val
        out = np.clip(out * (overcast_gain * tint[None, None, :]), 0.0, 1.0)

    angle_deg = float(rng.uniform(-18.0, 18.0))
    angle_rad = math.radians(angle_deg)
    sin_a = math.sin(angle_rad)
    cos_a = math.cos(angle_rad)

    area_scale = (h * w) / (720.0 * 1280.0)
    n_far = max(80, int(round(rng.integers(450, 901) * area_scale)))
    n_near = max(40, int(round(rng.integers(180, 421) * area_scale)))
    streak_alpha = float(rng.uniform(0.16, 0.34))

    # Draw two-depth slanted rain streaks on grayscale PIL canvas + light optical blur
    canvas = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(canvas)

    # Far thin drizzle streaks
    xs_f = rng.uniform(0.0, float(w), size=n_far)
    ys_f = rng.uniform(0.0, float(h), size=n_far)
    lens_f = rng.uniform(12.0, 24.0, size=n_far)
    vals_f = rng.integers(85, 165, size=n_far)
    for x0, y0, ln, val in zip(xs_f, ys_f, lens_f, vals_f):
        x1 = x0 + ln * sin_a
        y1 = y0 + ln * cos_a
        draw.line((float(x0), float(y0), float(x1), float(y1)), fill=int(val), width=1)

    # Near slightly longer/brighter streaks
    xs_n = rng.uniform(0.0, float(w), size=n_near)
    ys_n = rng.uniform(0.0, float(h), size=n_near)
    lens_n = rng.uniform(20.0, 36.0, size=n_near)
    vals_n = rng.integers(140, 235, size=n_near)
    for x0, y0, ln, val in zip(xs_n, ys_n, lens_n, vals_n):
        x1 = x0 + ln * sin_a
        y1 = y0 + ln * cos_a
        draw.line((float(x0), float(y0), float(x1), float(y1)), fill=int(val), width=1)

    blurred = canvas.filter(ImageFilter.GaussianBlur(radius=0.75))
    streak_mask = (np.asarray(blurred, dtype=np.float32) / 255.0) * streak_alpha  # [H, W]

    # Cool-neutral raindrop specular tint
    streak_color = np.array([0.82, 0.86, 0.92], dtype=np.float32)
    out = out * (1.0 - streak_mask[:, :, None]) + streak_color[None, None, :] * streak_mask[:, :, None]
    out = np.clip(out, 0.0, 1.0)

    params = {
        "overcast_gain": round(overcast_gain, 4),
        "contrast_alpha": round(contrast_alpha, 4),
        "angle_deg": round(angle_deg, 2),
        "n_far": int(n_far),
        "n_near": int(n_near),
        "streak_alpha": round(streak_alpha, 4),
    }
    return out, params


def apply_physical_degradation(
    pil_image: Image.Image,
    image_id: str,
    view_tag: str,
    seed: int = 20260420,
    return_params: bool = False,
) -> Image.Image | Tuple[Image.Image, Dict]:
    """Apply a deterministic physics-based synthetic degradation preserving exact ``(W, H)``.

    Given identical ``(pil_image, image_id, view_tag, seed)``, output pixels are 100%
    bitwise deterministic and strictly valid RGB uint8 in ``[0, 255]`` with zero
    geometric distortion.
    """
    if view_tag not in DEGRADATION_TAGS:
        raise ValueError(f"Unsupported degradation view_tag={view_tag!r}; expected one of {DEGRADATION_TAGS}")

    rgb = pil_image.convert("RGB")
    orig_w, orig_h = rgb.size
    if orig_w < 1 or orig_h < 1:
        raise ValueError(f"Invalid image size: {(orig_w, orig_h)}")

    derived_seed = derive_view_seed(image_id=image_id, view_tag=view_tag, base_seed=seed)

    if view_tag == "clean":
        out_img = rgb.copy()
        meta = {"view_tag": "clean", "derived_seed": derived_seed}
        return (out_img, meta) if return_params else out_img

    rng = np.random.default_rng(derived_seed)
    arr_f = np.asarray(rgb, dtype=np.float32) / 255.0

    if view_tag == "low_light":
        out_f, op_params = _apply_low_light_np(arr_f, rng)
    elif view_tag == "fog":
        out_f, op_params = _apply_fog_np(arr_f, rng)
    elif view_tag == "rain":
        out_f, op_params = _apply_rain_np(arr_f, rng, apply_overcast=True)
    elif view_tag == "compound_adverse":
        sub_mode = "low_light_rain" if int(rng.integers(0, 2)) == 0 else "fog_low_contrast"
        if sub_mode == "low_light_rain":
            tmp_f, p1 = _apply_low_light_np(arr_f, rng, gamma_range=(1.7, 2.8), gain_range=(0.58, 0.84))
            out_f, p2 = _apply_rain_np(tmp_f, rng, apply_overcast=False)
            op_params = {"sub_mode": sub_mode, "low_light": p1, "rain": p2}
        else:
            # Mild dusk dimming + heavier atmospheric fog + strong contrast attenuation
            gamma = float(rng.uniform(1.15, 1.55))
            gain = float(rng.uniform(0.76, 0.92))
            tmp_f = np.clip(np.power(np.clip(arr_f, 0.0, 1.0), gamma, dtype=np.float32) * gain, 0.0, 1.0)
            out_f, p_fog = _apply_fog_np(tmp_f, rng, beta_range=(0.95, 2.25), contrast_range=(0.72, 0.86))
            op_params = {
                "sub_mode": sub_mode,
                "dim_gamma": round(gamma, 4),
                "dim_gain": round(gain, 4),
                "fog": p_fog,
            }
    else:
        raise ValueError(f"Unhandled view_tag={view_tag!r}")

    out_u8 = np.rint(np.clip(out_f, 0.0, 1.0) * 255.0).astype(np.uint8)
    out_img = Image.fromarray(out_u8, mode="RGB")

    # Strict geometric & pixel-validity invariants
    assert out_img.size == (orig_w, orig_h), f"Degradation altered size: {out_img.size} != {(orig_w, orig_h)}"
    assert out_img.mode == "RGB", f"Degradation altered mode: {out_img.mode}"

    meta = {"view_tag": view_tag, "derived_seed": derived_seed, "params": op_params}
    return (out_img, meta) if return_params else out_img
