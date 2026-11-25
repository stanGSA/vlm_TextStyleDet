"""Zero-intrusion frozen wrapper around TextStyleDet (StyleDetV6).

Audit-hardened guarantees:
  - ``sys.dont_write_bytecode = True`` enforced prior to upstream imports.
  - Strictly frozen detector: overriding any outer ``.train()`` risks via explicit
    ``eval()`` + ``requires_grad_(False)`` + ``torch.no_grad()``.
  - Chunked ROI evaluation is structurally equivalent to ``C4ClipROIHead.forward``
    in eval mode within standard floating-point reduction tolerance while keeping
    8 GB GPU VRAM well below 2 GiB even in full FP32.
  - ``forward_single_crop`` & ``forward_crops`` execute each crop window strictly
    at ``batch_size = 1`` on its own deterministic 32-aligned canvas, guaranteeing
    100% independence from what other windows (e.g. GT-Oracle windows) are evaluated
    on the same image (resolving Red-Team P0-01).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.dont_write_bytecode = True

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torch import Tensor
from torchvision.ops import roi_align

from .runtime_shim import DEFAULT_RN_ROOT, ensure_textstyle_rn_importable

try:
    _BILINEAR = Image.Resampling.BILINEAR
except AttributeError:
    _BILINEAR = Image.BILINEAR


@dataclass
class GlobalPerceptionOutput:
    """All intermediate and final outputs from a single frozen global forward pass."""

    orig_size: Tuple[int, int]  # (orig_h, orig_w)
    resized_size: Tuple[int, int]  # (img_h, img_w) before 32-align pad
    canvas_size: Tuple[int, int]  # (pad_h, pad_w)
    detections: Dict[str, Tensor]  # 'boxes' [M,4], 'scores' [M], 'labels' [M] in orig 0-based coords
    proposals: Tensor  # [P, 4] in orig 0-based coords
    proposal_probs: Tensor  # [P] in [0, 1] (sigmoid of RPN objectness score)
    c4_feature: Tensor  # [1, C, H_f_valid, W_f_valid] detached FP32
    rpn_prob_map: Tensor  # [1, A, H_f_valid, W_f_valid] in [0, 1] detached FP32
    channel_mu: Tensor  # [1, C] detached FP32
    channel_log_sigma: Tensor  # [1, C] detached FP32


class FrozenTextStyleDetWrapper:
    """Frozen inference & intermediate-state extractor for StyleDetV6."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        clip_roi_weights_path: str | Path,
        rn_root: str | Path = DEFAULT_RN_ROOT,
        device: str | torch.device = "cuda",
        per_class_score_thresh: Optional[Dict[str, float] | Dict[int, float] | str] = "auto",
        score_thresh: Optional[float] = None,
        use_amp_bf16: bool = False,
        roi_chunk_size: int = 256,
        crop_max_proposals: int = 300,
    ) -> None:
        if roi_chunk_size < 16:
            raise ValueError(f"roi_chunk_size must be >= 16, got {roi_chunk_size}")
        if crop_max_proposals < 0:
            raise ValueError(f"crop_max_proposals must be >= 0, got {crop_max_proposals}")

        self.rn_root = ensure_textstyle_rn_importable(rn_root)
        from data.task_profiles import get_task_profile
        from models.detector import RAW_C4_INDEX, StyleDetV6

        self._raw_c4_index = RAW_C4_INDEX
        self.roi_chunk_size = int(roi_chunk_size)
        self.crop_max_proposals = int(crop_max_proposals)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_amp_bf16 = bool(use_amp_bf16 and self.device.type == "cuda")
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.clip_roi_weights_path = Path(clip_roi_weights_path).resolve()

        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Detector checkpoint not found: {self.checkpoint_path}")
        if not self.clip_roi_weights_path.is_file():
            raise FileNotFoundError(f"CLIP ROI weights not found: {self.clip_roi_weights_path}")

        ckpt = torch.load(str(self.checkpoint_path), map_location="cpu", weights_only=False)
        arch = dict(ckpt.get("arch") or {})
        saved_cfg = dict(ckpt.get("resolved_config") or {})

        self.dataset_profile = saved_cfg.get("dataset_profile") or arch.get("dataset_profile") or "sky7"
        profile = get_task_profile(self.dataset_profile)
        self.class_names: Tuple[str, ...] = tuple(profile.class_names)
        self.num_classes: int = len(self.class_names)

        self.detector_head: str = (saved_cfg.get("detector_head") or arch.get("detector_head") or "native_rn").lower()
        self.image_backbone: str = (saved_cfg.get("image_backbone") or arch.get("image_backbone") or "clip_rn101").lower()
        self.clip_roi_model_name: str = saved_cfg.get("clip_roi_model_name") or arch.get("clip_roi_model_name") or "RN101"
        self.native_padding_contract: str = (
            saved_cfg.get("native_padding_contract") or arch.get("native_padding_contract") or "valid_mask_v1"
        )

        native_min, native_max = 864, 1536
        if arch.get("native_geometry"):
            parts = str(arch["native_geometry"]).split("x")
            if len(parts) >= 2:
                native_min, native_max = int(parts[0]), int(parts[1])
        self.native_min_size = int(saved_cfg.get("native_min_size", native_min))
        self.native_max_size = int(saved_cfg.get("native_max_size", native_max))

        min_test, max_test = 700, 1100
        if arch.get("native_runtime_eval_geometry"):
            kv = dict(
                item.split("=", 1)
                for item in str(arch["native_runtime_eval_geometry"]).split(";")
                if "=" in item
            )
            min_test = int(kv.get("short", min_test))
            max_test = int(kv.get("max", max_test))
        self.min_size_test = int(saved_cfg.get("min_size_test", min_test))
        self.max_size_test = int(saved_cfg.get("max_size_test", max_test))

        thresh_map: Optional[Dict[int, float]] = None
        if per_class_score_thresh == "auto":
            raw_pc = saved_cfg.get("per_class_score_thresh")
            if raw_pc is None and self.dataset_profile == "sky7" and self.detector_head == "native_rn":
                # Matches SKY7 native_launch.py training/eval recipe
                raw_pc = {"motor": 0.01, "bike": 0.01, "rider": 0.01}
            thresh_map = self._normalize_thresh_map(raw_pc)
        elif per_class_score_thresh is not None and per_class_score_thresh != "none":
            thresh_map = self._normalize_thresh_map(per_class_score_thresh)
        self.per_class_score_thresh = thresh_map
        self.score_thresh = float(score_thresh if score_thresh is not None else saved_cfg.get("score_thresh", 0.03))

        logit_scale = 50.0
        if "roi_head.logit_scale" in ckpt["model_state_dict"]:
            logit_scale = float(ckpt["model_state_dict"]["roi_head.logit_scale"].item())
        self.roi_clip_logit_scale = logit_scale

        prompt_file = self.rn_root / "data" / (
            "art_prompts.json" if self.dataset_profile == "art6" else "weather_prompts.txt"
        )

        self.model = StyleDetV6(
            num_classes=self.num_classes,
            class_names=self.class_names,
            dataset_profile=self.dataset_profile,
            score_thresh=self.score_thresh,
            native_task_contract=ckpt.get("run_contract"),
            native_train_text2style=False,
            native_text2style_teacher_weight=0.0,
            pretrained_backbone=False,
            image_backbone=self.image_backbone,
            prompt_file=str(prompt_file) if prompt_file.is_file() else None,
            clip_model_name=self.clip_roi_model_name,
            clip_weights_path=str(self.clip_roi_weights_path),
            native_raw_c4_clip_model_name=self.clip_roi_model_name,
            native_raw_c4_clip_weights_path=str(self.clip_roi_weights_path),
            native_min_size=self.native_min_size,
            native_max_size=self.native_max_size,
            roi_clip_logit_scale=logit_scale,
            per_class_score_thresh=self.per_class_score_thresh,
            focal_class_weights=saved_cfg.get("default_class_weights_vec"),
            detector_head=self.detector_head,
            clip_roi_model_name=self.clip_roi_model_name,
            clip_roi_weights_path=str(self.clip_roi_weights_path),
        )

        incompatible = self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
        strict_missing = [k for k in incompatible.missing_keys if not self._is_style_key(k)]
        strict_unexpected = [k for k in incompatible.unexpected_keys if not self._is_style_key(k)]
        if strict_missing or strict_unexpected:
            raise RuntimeError(
                f"Detector strict weight mismatch: missing={strict_missing[:8]}, unexpected={strict_unexpected[:8]}"
            )

        self.model.to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.checkpoint_epoch = ckpt.get("epoch")
        self.checkpoint_best_metric = ckpt.get("best_metric")

    def config_dict(self) -> Dict:
        return {
            "dataset_profile": self.dataset_profile,
            "detector_head": self.detector_head,
            "image_backbone": self.image_backbone,
            "clip_roi_model_name": self.clip_roi_model_name,
            "native_padding_contract": self.native_padding_contract,
            "native_min_size": self.native_min_size,
            "native_max_size": self.native_max_size,
            "min_size_test": self.min_size_test,
            "max_size_test": self.max_size_test,
            "score_thresh": self.score_thresh,
            "per_class_score_thresh": self.per_class_score_thresh,
            "roi_clip_logit_scale": self.roi_clip_logit_scale,
            "detections_per_img": int(self.model.roi_head.detections_per_img),
            "use_amp_bf16": self.use_amp_bf16,
            "roi_chunk_size": self.roi_chunk_size,
            "crop_max_proposals": self.crop_max_proposals,
            "crop_isolation_mode": "strict_single_crop_batch1",
        }

    @staticmethod
    def _is_style_key(key: str) -> bool:
        return (
            "._clip_encoder." in key
            or "._text2style." in key
            or "._normalizer." in key
            or key.startswith("native_rn_style_augmentor.")
            or key.endswith("._clip_encoder")
            or key.endswith("._text2style")
        )

    def _normalize_thresh_map(self, raw_map) -> Optional[Dict[int, float]]:
        if not raw_map:
            return None
        if isinstance(raw_map, str):
            raw_map = json.loads(raw_map)
        out: Dict[int, float] = {}
        for k, v in raw_map.items():
            if isinstance(k, int) or str(k).isdigit():
                idx = int(k)
            else:
                idx = self.class_names.index(str(k).lower()) + 1
            out[idx] = float(v)
        return out

    def resize_pil_for_eval(self, image: Image.Image) -> Tuple[Tensor, Tuple[int, int]]:
        """Resize PIL image matching StyleDetV6 eval DetectionAugmentation."""
        w, h = image.size
        if w < 1 or h < 1:
            raise ValueError(f"Invalid PIL image size: {(w, h)}")
        min_s = self.min_size_test
        max_s = self.max_size_test
        scale = min_s / min(h, w)
        if max(h, w) * scale > max_s:
            scale = max_s / max(h, w)
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        if (new_w, new_h) != (w, h):
            resized = image.resize((new_w, new_h), _BILINEAR)
        else:
            resized = image
        tensor = TF.to_tensor(resized)  # [3, new_h, new_w] in [0, 1]
        return tensor, (new_h, new_w)

    @staticmethod
    def collate_tensors(tensors: Sequence[Tensor], device: torch.device) -> Tuple[Tensor, List[Tuple[int, int]], Tensor]:
        """Pad list of [3, H_i, W_i] tensors to 32-divisible batch canvas + valid mask."""
        image_sizes = [(int(t.shape[1]), int(t.shape[2])) for t in tensors]
        max_h = ((max(h for h, _ in image_sizes) + 31) // 32) * 32
        max_w = ((max(w for _, w in image_sizes) + 31) // 32) * 32

        padded = torch.zeros((len(tensors), 3, max_h, max_w), dtype=torch.float32)
        valid_mask = torch.zeros((len(tensors), 1, max_h, max_w), dtype=torch.bool)
        for i, t in enumerate(tensors):
            h, w = image_sizes[i]
            padded[i, :, :h, :w] = t
            valid_mask[i, :, :h, :w] = True
        return padded.to(device, non_blocking=True), image_sizes, valid_mask.to(device, non_blocking=True)

    @torch.no_grad()
    def _chunked_roi_head_eval(
        self,
        c4_feature: Tensor,
        proposals_list: List[Tensor],
        image_sizes: List[Tuple[int, int]],
    ) -> List[Dict[str, Tensor]]:
        """Evaluate C4ClipROIHead in proposal chunks (structurally equivalent in eval mode; keeps VRAM < 2 GiB)."""
        self.model.eval()
        roi_head = self.model.roi_head
        rois_parts: List[Tensor] = []
        for b, props in enumerate(proposals_list):
            if props.numel() == 0:
                continue
            batch_idx = torch.full((props.shape[0], 1), float(b), dtype=props.dtype, device=props.device)
            rois_parts.append(torch.cat([batch_idx, props], dim=1))

        if not rois_parts:
            return [
                {
                    "boxes": torch.zeros((0, 4), dtype=torch.float32, device=c4_feature.device),
                    "labels": torch.zeros((0,), dtype=torch.long, device=c4_feature.device),
                    "scores": torch.zeros((0,), dtype=torch.float32, device=c4_feature.device),
                }
                for _ in image_sizes
            ]

        all_rois = torch.cat(rois_parts, dim=0)
        total_rois = all_rois.shape[0]
        chunk = max(16, self.roi_chunk_size)

        text_features = roi_head.text_features.to(device=c4_feature.device)
        logit_scale = roi_head.logit_scale.to(device=c4_feature.device)

        cls_logits_chunks: List[Tensor] = []
        bbox_pred_chunks: List[Tensor] = []

        for start in range(0, total_rois, chunk):
            sub_rois = all_rois[start : start + chunk]
            roi_feats = roi_align(
                c4_feature,
                sub_rois,
                output_size=roi_head.pool_size,
                spatial_scale=roi_head.spatial_scale,
                sampling_ratio=roi_head.pool_sampling_ratio,
                aligned=True,
            )
            res5 = roi_head.roi_res5(roi_feats)
            img_feats = roi_head.roi_attnpool(res5)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True).clamp_min(1e-6)

            tf = text_features.to(dtype=img_feats.dtype)
            ls = logit_scale.to(dtype=img_feats.dtype)
            fg_logits = ls * (img_feats @ tf.t())
            bg_logits = roi_head.bg_score(img_feats)
            cls_logits_chunks.append(torch.cat([bg_logits, fg_logits], dim=1))
            bbox_pred_chunks.append(roi_head.bbox_pred(res5.mean(dim=(2, 3))))

        cls_logits = torch.cat(cls_logits_chunks, dim=0)
        bbox_pred = torch.cat(bbox_pred_chunks, dim=0)
        return roi_head._postprocess(cls_logits, bbox_pred, proposals_list, image_sizes)

    @torch.no_grad()
    def forward_global(self, pil_image: Image.Image) -> GlobalPerceptionOutput:
        """Run single global forward pass and extract detections + RPN/C4 intermediates."""
        self.model.eval()
        orig_w, orig_h = pil_image.size
        img_tensor, (img_h, img_w) = self.resize_pil_for_eval(pil_image)
        images, image_sizes, image_valid_mask = self.collate_tensors([img_tensor], self.device)
        canvas_size = (int(images.shape[2]), int(images.shape[3]))

        autocast_ctx = (
            torch.amp.autocast("cuda", dtype=torch.bfloat16)
            if self.use_amp_bf16
            else torch.amp.autocast("cuda", enabled=False)
        )
        with autocast_ctx:
            if self.detector_head == "native_rn":
                raw_feats = self.model.backbone.forward_raw_features(images)
                feature_masks = self.model._build_native_feature_masks(
                    image_valid_mask, image_sizes, raw_feats, canvas_size
                )
                c4_feature = raw_feats[self._raw_c4_index]
                feat_mask = feature_masks[self._raw_c4_index]
            elif self.detector_head == "raw_c4_adain_clip":
                c4_feature = self.model.backbone.forward_c4(images)
                feat_mask = F.interpolate(
                    image_valid_mask.float(), size=c4_feature.shape[-2:], mode="nearest"
                ).to(torch.bool)
            else:
                raise NotImplementedError(
                    f"FrozenTextStyleDetWrapper currently supports native_rn/raw_c4_adain_clip, got {self.detector_head}"
                )

            rpn = self.model.rpn
            features_list = [c4_feature]
            anchors_per_level = rpn.anchor_generator.grid_anchors(features_list, [rpn.stride])
            t_rpn = F.relu(rpn.head.conv(c4_feature))
            cls_logits_spatial = rpn.head.cls_logits(t_rpn)  # [1, A, H_f, W_f]
            reg_spatial = rpn.head.bbox_pred(t_rpn)  # [1, A*4, H_f, W_f]

            bsz, num_a, hf, wf = cls_logits_spatial.shape
            obj_flat = [cls_logits_spatial.permute(0, 2, 3, 1).reshape(bsz, hf * wf * num_a)]
            reg_flat = [reg_spatial.permute(0, 2, 3, 1).reshape(bsz, hf * wf * num_a, 4)]

            valid_anchor_masks = None
            if self.detector_head == "native_rn" and feat_mask is not None:
                valid_anchor_masks = [
                    feat_mask[:, 0].reshape(bsz, -1).repeat_interleave(num_a, dim=1)
                ]

            proposals_list, proposal_scores_list = rpn.proposal_layer.batch_forward(
                obj_flat,
                reg_flat,
                anchors_per_level,
                image_sizes,
                training=False,
                valid_masks=valid_anchor_masks,
            )
            detections_list = self._chunked_roi_head_eval(c4_feature, proposals_list, image_sizes)

        # Slice out pure nearest-downsampled padding rows/columns per upstream valid_mask contract
        valid_fh = max(1, int(feat_mask[0, 0, :, 0].sum().item()))
        valid_fw = max(1, int(feat_mask[0, 0, 0, :].sum().item()))
        c4_valid = c4_feature[:, :, :valid_fh, :valid_fw].detach().float()
        rpn_prob_valid = torch.sigmoid(cls_logits_spatial[:, :, :valid_fh, :valid_fw].detach().float())

        flat = c4_valid.flatten(2)
        mu = flat.mean(dim=-1).detach()
        sigma = flat.std(dim=-1, unbiased=False).clamp(min=1e-5).detach()
        log_sigma = torch.log(sigma).detach()

        scale_x = orig_w / max(img_w, 1)
        scale_y = orig_h / max(img_h, 1)

        det = detections_list[0]
        boxes = det["boxes"].detach().float().cpu().clone()
        scores = det["scores"].detach().float().cpu().clone()
        labels = det["labels"].detach().long().cpu().clone()
        if boxes.numel() > 0:
            boxes[:, 0::2] *= scale_x
            boxes[:, 1::2] *= scale_y
            boxes[:, 0::2] = boxes[:, 0::2].clamp(0.0, float(orig_w))
            boxes[:, 1::2] = boxes[:, 1::2].clamp(0.0, float(orig_h))

        props = proposals_list[0].detach().float().cpu().clone()
        prop_probs = torch.sigmoid(proposal_scores_list[0].detach().float()).cpu()
        if props.numel() > 0:
            props[:, 0::2] *= scale_x
            props[:, 1::2] *= scale_y
            props[:, 0::2] = props[:, 0::2].clamp(0.0, float(orig_w))
            props[:, 1::2] = props[:, 1::2].clamp(0.0, float(orig_h))

        return GlobalPerceptionOutput(
            orig_size=(orig_h, orig_w),
            resized_size=(img_h, img_w),
            canvas_size=canvas_size,
            detections={"boxes": boxes, "scores": scores, "labels": labels},
            proposals=props,
            proposal_probs=prop_probs,
            c4_feature=c4_valid,
            rpn_prob_map=rpn_prob_valid,
            channel_mu=mu,
            channel_log_sigma=log_sigma,
        )

    @torch.no_grad()
    def forward_single_crop(
        self,
        pil_image: Image.Image,
        crop_xyxy: Tuple[int, int, int, int],
    ) -> Dict[str, Tensor]:
        """Evaluate a single crop strictly at batch_size=1 on its own deterministic canvas.

        Guarantees zero cross-window padding, convolution border, or ROI-chunk interference.
        """
        self.model.eval()
        orig_w, orig_h = pil_image.size
        x1, y1, x2, y2 = crop_xyxy
        x1 = max(0, min(int(x1), orig_w - 2))
        y1 = max(0, min(int(y1), orig_h - 2))
        x2 = max(x1 + 2, min(int(x2), orig_w))
        y2 = max(y1 + 2, min(int(y2), orig_h))

        crop_pil = pil_image.crop((x1, y1, x2, y2))
        t, (zoom_h, zoom_w) = self.resize_pil_for_eval(crop_pil)
        images, image_sizes, image_valid_mask = self.collate_tensors([t], self.device)

        autocast_ctx = (
            torch.amp.autocast("cuda", dtype=torch.bfloat16)
            if self.use_amp_bf16
            else torch.amp.autocast("cuda", enabled=False)
        )
        with autocast_ctx:
            if self.detector_head == "native_rn":
                raw_feats = self.model.backbone.forward_raw_features(images)
                feature_masks = self.model._build_native_feature_masks(
                    image_valid_mask, image_sizes, raw_feats, tuple(images.shape[-2:])
                )
                c4_feature = raw_feats[self._raw_c4_index]
                proposals_list, _, _ = self.model.rpn(
                    c4_feature, image_sizes, targets=None, feature_mask=feature_masks[self._raw_c4_index]
                )
                if self.crop_max_proposals > 0:
                    proposals_list = [p[: self.crop_max_proposals] for p in proposals_list]
                detections_list = self._chunked_roi_head_eval(c4_feature, proposals_list, image_sizes)
            elif self.detector_head == "raw_c4_adain_clip":
                c4_feature = self.model.backbone.forward_c4(images)
                proposals_list, _, _ = self.model.rpn(c4_feature, image_sizes, targets=None)
                if self.crop_max_proposals > 0:
                    proposals_list = [p[: self.crop_max_proposals] for p in proposals_list]
                detections_list = self._chunked_roi_head_eval(c4_feature, proposals_list, image_sizes)
            else:
                detections_list, _ = self.model(
                    images,
                    image_sizes,
                    targets=None,
                    style_prob=0.0,
                    image_valid_mask=image_valid_mask,
                )

        det = detections_list[0]
        crop_w = x2 - x1
        crop_h = y2 - y1
        sx = crop_w / max(zoom_w, 1)
        sy = crop_h / max(zoom_h, 1)

        boxes_crop = det["boxes"].detach().float().cpu().clone()
        scores = det["scores"].detach().float().cpu().clone()
        labels = det["labels"].detach().long().cpu().clone()

        if boxes_crop.numel() > 0:
            boxes_crop[:, 0::2] *= sx
            boxes_crop[:, 1::2] *= sy
            boxes_crop[:, 0::2] = boxes_crop[:, 0::2].clamp(0.0, float(crop_w))
            boxes_crop[:, 1::2] = boxes_crop[:, 1::2].clamp(0.0, float(crop_h))

            boxes_orig = boxes_crop.clone()
            boxes_orig[:, 0::2] += float(x1)
            boxes_orig[:, 1::2] += float(y1)
            boxes_orig[:, 0::2] = boxes_orig[:, 0::2].clamp(0.0, float(orig_w))
            boxes_orig[:, 1::2] = boxes_orig[:, 1::2].clamp(0.0, float(orig_h))
        else:
            boxes_orig = boxes_crop.clone()

        return {
            "boxes_in_crop": boxes_crop,
            "boxes_in_orig": boxes_orig,
            "scores": scores,
            "labels": labels,
            "crop_xyxy": (x1, y1, x2, y2),
            "zoom_size": (zoom_h, zoom_w),
            "orig_size": (orig_h, orig_w),
        }

    @torch.no_grad()
    def forward_crops(
        self,
        pil_image: Image.Image,
        crop_boxes_xyxy: Sequence[Tuple[int, int, int, int]],
    ) -> List[Dict[str, Tensor]]:
        """Evaluate each crop strictly in isolated single-crop mode (zero batch-partner coupling)."""
        return [self.forward_single_crop(pil_image, rect) for rect in crop_boxes_xyxy]
