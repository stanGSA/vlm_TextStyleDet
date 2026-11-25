"""Dataset, Manifest Hashing & VOC2007 Multi-IoU AP Evaluation Utilities.

Uses ``voc_eval`` from ``textstyleDet_rn`` with pre-parsed in-memory ``annotation_records``
(zero disk writes to ``E:\\textstyleDet\\data\\sky``).
Note: ``AP`` denotes the mean of VOC2007 11-point AP over IoU thresholds 0.50:0.05:0.95
(matching SE-COT / StyleDetV6 evaluation protocol, not pycocotools 101-point AP).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

sys.dont_write_bytecode = True

import numpy as np
import torch
from torch import Tensor

from .runtime_shim import DEFAULT_RN_ROOT, ensure_textstyle_rn_importable
from .state_encoder import DEFAULT_WEAK_CLASS_IDS

SKY_DEFAULT_SPLITS: Dict[str, str] = {
    "daytime_clear": "test",
    "daytime_foggy": "test",
    "dusk_rainy": "train",
    "Night-Sunny": "test",
    "night_rainy": "train",
}

SKY_TARGET_DOMAINS: Tuple[str, ...] = (
    "daytime_foggy",
    "dusk_rainy",
    "Night-Sunny",
    "night_rainy",
)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj) -> str:
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def largest_remainder_round_pct(counts: Sequence[int], total: int, decimals: int = 2) -> List[float]:
    """Round mutually exclusive counts to percentages whose display sum equals 100.00%."""
    if total <= 0:
        return [0.0 for _ in counts]
    scale = 10 ** decimals
    target_units = 100 * scale
    raw_units = [c * target_units / total for c in counts]
    floors = [int(np.floor(u)) for u in raw_units]
    remainders = [u - f for u, f in zip(raw_units, floors)]
    deficit = target_units - sum(floors)
    order = sorted(range(len(counts)), key=lambda i: (-remainders[i], i))
    for idx in order[: max(0, deficit)]:
        floors[idx] += 1
    return [round(f / scale, decimals) for f in floors]


def resolve_sky_domain_root(data_dir: str | Path, domain: str) -> Path:
    base = Path(data_dir)
    if domain == "Night-Sunny":
        voc_root = base / domain
    else:
        voc_root = base / domain / "VOC2007"
    if not voc_root.exists():
        voc_root = base / domain
    if not (voc_root / "Annotations").is_dir():
        raise FileNotFoundError(f"Cannot locate VOC root for domain={domain} under {base}")
    return voc_root


def load_domain_records(
    data_dir: str | Path,
    domain: str,
    split: str | None = None,
    max_images: int = 0,
    rn_root: str | Path = DEFAULT_RN_ROOT,
) -> Tuple[Path, str, List[str], Dict[str, List[Dict]], str]:
    """Load split image IDs, parse XML annotations in memory, and return split SHA256."""
    ensure_textstyle_rn_importable(rn_root)
    from eval.voc_eval import parse_rec

    voc_root = resolve_sky_domain_root(data_dir, domain)
    resolved_split = split or SKY_DEFAULT_SPLITS.get(domain, "test")
    split_file = voc_root / "ImageSets" / "Main" / f"{resolved_split}.txt"
    if not split_file.is_file():
        raise FileNotFoundError(f"Split file missing: {split_file}")

    raw_ids = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    anno_dir = voc_root / "Annotations"
    img_dir = voc_root / "JPEGImages"

    image_ids: List[str] = []
    records: Dict[str, List[Dict]] = {}
    for img_id in raw_ids:
        xml_path = anno_dir / f"{img_id}.xml"
        jpg_path = img_dir / f"{img_id}.jpg"
        if xml_path.is_file() and jpg_path.is_file():
            image_ids.append(img_id)
            records[img_id] = parse_rec(str(xml_path))
            if max_images > 0 and len(image_ids) >= max_images:
                break

    split_ids_sha256 = hashlib.sha256("\n".join(image_ids).encode("utf-8")).hexdigest()
    return voc_root, resolved_split, image_ids, records, split_ids_sha256


def record_to_0based_gt(
    objs: Sequence[Dict],
    class_names: Sequence[str],
) -> Tuple[Tensor, Tensor, Tensor]:
    """Convert parsed VOC XML objects (1-based inclusive) to 0-based continuous tensors."""
    cls_to_idx = {name: i + 1 for i, name in enumerate(class_names)}
    boxes: List[List[float]] = []
    labels: List[int] = []
    difficult: List[bool] = []

    for obj in objs:
        name = obj["name"]
        if name not in cls_to_idx:
            continue
        x1, y1, x2, y2 = obj["bbox"]
        x1, x2 = min(float(x1), float(x2)), max(float(x1), float(x2))
        y1, y2 = min(float(y1), float(y2)), max(float(y1), float(y2))
        x1_0 = max(x1 - 1.0, 0.0)
        y1_0 = max(y1 - 1.0, 0.0)
        if (x2 - x1_0) <= 0.0 or (y2 - y1_0) <= 0.0:
            continue
        boxes.append([x1_0, y1_0, x2, y2])
        labels.append(cls_to_idx[name])
        difficult.append(bool(obj.get("difficult", 0)))

    if not boxes:
        return (
            torch.zeros((0, 4), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.long),
            torch.zeros((0,), dtype=torch.bool),
        )
    return (
        torch.tensor(boxes, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(difficult, dtype=torch.bool),
    )


def format_voc_prediction_lines(
    image_id: str,
    det_0based: Dict[str, Tensor],
    orig_size: Tuple[int, int],
    num_classes: int,
    voc_1_based: bool = True,
) -> Dict[int, List[str]]:
    """Return per-class formatted VOC lines for a single image (identical to ``train.py:evaluate()``)."""
    out: Dict[int, List[str]] = defaultdict(list)
    boxes = det_0based["boxes"].detach().cpu().clone()
    scores = det_0based["scores"].detach().cpu().clone()
    labels = det_0based["labels"].detach().cpu().clone()

    if boxes.numel() == 0:
        return out

    orig_h, orig_w = int(orig_size[0]), int(orig_size[1])
    if voc_1_based:
        boxes[:, 0] += 1.0
        boxes[:, 1] += 1.0
        boxes[:, 0] = boxes[:, 0].clamp(1.0, float(orig_w))
        boxes[:, 1] = boxes[:, 1].clamp(1.0, float(orig_h))
        boxes[:, 2] = boxes[:, 2].clamp(1.0, float(orig_w))
        boxes[:, 3] = boxes[:, 3].clamp(1.0, float(orig_h))
    else:
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0.0, float(orig_w))
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0.0, float(orig_h))

    for box, score, label in zip(boxes, scores, labels):
        cls_id = int(label.item())
        if cls_id < 1 or cls_id > num_classes:
            continue
        line = (
            f"{image_id} {float(score):.6f} "
            f"{float(box[0]):.1f} {float(box[1]):.1f} "
            f"{float(box[2]):.1f} {float(box[3]):.1f}"
        )
        out[cls_id].append(line)
    return out


def append_voc_prediction_lines(
    pred_lines_by_class: Dict[int, List[str]],
    image_id: str,
    det_0based: Dict[str, Tensor],
    orig_size: Tuple[int, int],
    num_classes: int,
    voc_1_based: bool = True,
) -> None:
    single = format_voc_prediction_lines(image_id, det_0based, orig_size, num_classes, voc_1_based)
    for cls_id, lines in single.items():
        pred_lines_by_class[cls_id].extend(lines)


def evaluate_voc_predictions(
    pred_lines_by_class: Dict[int, List[str]],
    class_names: Sequence[str],
    annotation_records: Dict[str, List[Dict]],
    use_07_metric: bool = True,
    weak_class_ids: Sequence[int] = DEFAULT_WEAK_CLASS_IDS,
    iou_thresholds: Iterable[int] = range(50, 100, 5),
    rn_root: str | Path = DEFAULT_RN_ROOT,
) -> Dict:
    """Compute exact SE-COT VOC07 11-point AP (0.50:0.05:0.95), AP50, AP75, Weak-AP, and max recall."""
    ensure_textstyle_rn_importable(rn_root)
    from eval.voc_eval import voc_eval

    iou_list = list(iou_thresholds)
    aps_by_iou: Dict[int, List[float]] = defaultdict(list)
    max_rec50_by_cls: List[float] = []
    max_rec75_by_cls: List[float] = []

    with tempfile.TemporaryDirectory(prefix="az_agent_voc_eval_") as dirname:
        res_file_template = os.path.join(dirname, "{}.txt")
        for cls_id, cls_name in enumerate(class_names, start=1):
            lines = pred_lines_by_class.get(cls_id, [])
            with open(res_file_template.format(cls_name), "w", encoding="utf-8") as f:
                if lines:
                    f.write("\n".join(lines))
                else:
                    f.write("")

            for thresh in iou_list:
                rec, _, ap = voc_eval(
                    res_file_template,
                    "",
                    "",
                    cls_name,
                    ovthresh=thresh / 100.0,
                    use_07_metric=use_07_metric,
                    annotation_records=annotation_records,
                )
                aps_by_iou[thresh].append(float(ap * 100.0))
                max_r = float(rec[-1] * 100.0) if rec.size > 0 else 0.0
                if thresh == 50:
                    max_rec50_by_cls.append(max_r)
                elif thresh == 75:
                    max_rec75_by_cls.append(max_r)

    mAP_by_iou = {iou: float(np.mean(vals)) if vals else 0.0 for iou, vals in aps_by_iou.items()}
    cls_ap50 = aps_by_iou.get(50, [0.0] * len(class_names))
    cls_ap75 = aps_by_iou.get(75, [0.0] * len(class_names))
    cls_ap_mean = [
        float(np.mean([aps_by_iou[iou][c_i] for iou in iou_list]))
        for c_i in range(len(class_names))
    ]

    weak_indices = [cid - 1 for cid in weak_class_ids if 1 <= cid <= len(class_names)]
    weak_ap50 = float(np.mean([cls_ap50[i] for i in weak_indices])) if weak_indices else 0.0
    weak_ap75 = float(np.mean([cls_ap75[i] for i in weak_indices])) if weak_indices else 0.0
    weak_ap = float(np.mean([cls_ap_mean[i] for i in weak_indices])) if weak_indices else 0.0

    ret = OrderedDict()
    ret["metric_protocol"] = "voc07_11pt_multi_iou_50_95" if use_07_metric else "voc10_continuous_multi_iou_50_95"
    ret["AP"] = float(np.mean(list(mAP_by_iou.values()))) if mAP_by_iou else 0.0
    ret["AP50"] = mAP_by_iou.get(50, 0.0)
    ret["AP75"] = mAP_by_iou.get(75, 0.0)
    ret["weak_AP"] = weak_ap
    ret["weak_AP50"] = weak_ap50
    ret["weak_AP75"] = weak_ap75
    ret["voc_1to1_max_recall50"] = float(np.mean(max_rec50_by_cls)) if max_rec50_by_cls else 0.0
    ret["voc_1to1_max_recall75"] = float(np.mean(max_rec75_by_cls)) if max_rec75_by_cls else 0.0
    ret["classwise_ap50"] = {name: round(val, 4) for name, val in zip(class_names, cls_ap50)}
    ret["classwise_ap75"] = {name: round(val, 4) for name, val in zip(class_names, cls_ap75)}
    ret["classwise_ap"] = {name: round(val, 4) for name, val in zip(class_names, cls_ap_mean)}
    ret["classwise_voc_max_recall50"] = {name: round(val, 4) for name, val in zip(class_names, max_rec50_by_cls)}
    ret["mAP_by_iou"] = {str(k): round(v, 4) for k, v in mAP_by_iou.items()}
    return ret
