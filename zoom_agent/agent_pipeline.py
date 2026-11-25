"""End-to-End AZ-Agent Detector Pipeline (Fast/Slow Dual-Path & Strict Frozen Isolation).

Connects:
  1. ``FrozenTextStyleDetWrapper`` (permanently frozen StyleDetV6 in eval mode)
  2. ``StateEncoder`` (parameter-free cues + grad-safe ``AgentBatchState`` assembly)
  3. ``ActiveZoomPolicyHead`` (sole trainable ~2M-param active-zoom policy network)
  4. ``DualViewWBFFusion`` (edge-truncation-filtered same-view-exclusive WBF / NMS)

Guarantees:
  - Calling ``agent.train(True)`` or ``agent.requires_grad_(True)`` ONLY affects
    ``policy_head``; ``detector_wrapper.model`` remains strictly ``eval()`` with
    ``requires_grad = False`` (zero BN stat drift, zero detector gradients).
  - Fast path (``action == ()``, No-Crop): returns global detections immediately
    with zero extra crop forward passes.
  - Slow path (``1 <= len(action) <= K_max``): executes each chosen window strictly
    via Phase-0 audited ``forward_single_crop`` (isolated ``batch_size=1`` canvas)
    and fuses via ``DualViewWBFFusion``.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

sys.dont_write_bytecode = True

import torch
import torch.nn as nn
from PIL import Image
from torch import Tensor

from .candidate_grid import CandidateGrid
from .detector_wrapper import FrozenTextStyleDetWrapper, GlobalPerceptionOutput
from .policy_head import ActionTrajectory, ActiveZoomPolicyHead
from .state_encoder import AgentBatchState, StateEncoder
from .wbf_fusion import DualViewWBFFusion


@dataclass
class AgentInferenceOutput:
    """Structured end-to-end detection & decision telemetry for a single image."""

    detections: Dict[str, Tensor]  # Final refined 0-based detections {'boxes', 'scores', 'labels'}
    global_detections: Dict[str, Tensor]  # Global-only baseline detections (capped to max_detections)
    action: ActionTrajectory  # Selected window IDs: (), (w1,), or (w1, w2)
    used_slow_path: bool  # False if No-Crop fast path; True if local crops ran
    action_log_prob: float  # Exact log pi_theta(action | state)
    action_prob: float  # Exact pi_theta(action | state)
    no_crop_prob: float  # Step-1 probability of No-Crop (index 18)
    window_logits: Tensor  # [19] detached CPU FP32
    step1_probs: Tensor  # [19] detached CPU FP32
    selected_crop_boxes: List[Tuple[int, int, int, int]]  # Executed crop rects in orig pixel xyxy
    timings_ms: Dict[str, float]  # Per-stage latency breakdown in milliseconds
    orig_size: Tuple[int, int]  # (orig_h, orig_w)


class AZAgentDetector(nn.Module):
    """End-to-end Active Zoom Agent Detector wrapping frozen TextStyleDet + trainable policy."""

    def __init__(
        self,
        detector_wrapper: FrozenTextStyleDetWrapper,
        policy_head: Optional[ActiveZoomPolicyHead] = None,
        state_encoder: Optional[StateEncoder] = None,
        fuser: Optional[DualViewWBFFusion] = None,
        move_policy_to_detector_device: bool = True,
    ) -> None:
        super().__init__()
        # Store frozen detector wrapper as a non-Module attribute so `agent.parameters()`
        # exposes ONLY trainable `policy_head` parameters, never frozen StyleDetV6 weights.
        self.detector_wrapper = detector_wrapper
        self.policy_head = policy_head if policy_head is not None else ActiveZoomPolicyHead()
        self.grid: CandidateGrid = self.policy_head.grid
        self.state_encoder = state_encoder if state_encoder is not None else StateEncoder(grid_size=(12, 20))
        self.fuser = (
            fuser
            if fuser is not None
            else DualViewWBFFusion(
                global_weight=1.0,
                local_weight=0.85,
                iou_thr=0.55,
                trunc_margin_px=5.0,
                max_detections=100,
                fusion_mode="wbf",
            )
        )

        if move_policy_to_detector_device:
            self.policy_head.to(self.detector_wrapper.device)

        self._enforce_detector_frozen()

    def _enforce_detector_frozen(self) -> None:
        """Guarantee frozen StyleDetV6 stays in eval() with requires_grad=False."""
        self.detector_wrapper.model.eval()
        for param in self.detector_wrapper.model.parameters():
            param.requires_grad = False

    @property
    def policy_device(self) -> torch.device:
        return next(self.policy_head.parameters()).device

    def train(self, mode: bool = True) -> "AZAgentDetector":
        """Toggle training mode ONLY on ``policy_head``; keep detector permanently frozen/eval."""
        super().train(mode)
        self.policy_head.train(mode)
        self._enforce_detector_frozen()
        return self

    def requires_grad_(self, requires_grad: bool = True) -> "AZAgentDetector":
        """Toggle ``requires_grad`` ONLY on ``policy_head``; keep detector permanently frozen."""
        self.policy_head.requires_grad_(requires_grad)
        self._enforce_detector_frozen()
        return self

    def config_dict(self) -> Dict:
        return {
            "detector": self.detector_wrapper.config_dict(),
            "candidate_grid": self.grid.config_dict(),
            "state_encoder": self.state_encoder.config_dict(),
            "policy_head": self.policy_head.config_dict(),
            "fuser": self.fuser.config_dict(),
        }

    def _maybe_sync(self, sync_cuda: bool) -> None:
        if sync_cuda and self.detector_wrapper.device.type == "cuda":
            torch.cuda.synchronize(self.detector_wrapper.device)

    @torch.no_grad()
    def extract_state(
        self,
        pil_image: Image.Image,
        image_id: Optional[str] = None,
        valid_action_mask: Optional[Tensor] = None,
    ) -> Tuple[GlobalPerceptionOutput, AgentBatchState]:
        """Run frozen global perception and build a grad-safe ``AgentBatchState`` (B=1)."""
        self._enforce_detector_frozen()
        perc = self.detector_wrapper.forward_global(pil_image)
        state = self.state_encoder.encode_state(
            perc=perc,
            device=self.policy_device,
            image_id=image_id,
            valid_action_mask=valid_action_mask,
        )
        return perc, state

    @torch.no_grad()
    def extract_batch_state(
        self,
        pil_images: Sequence[Image.Image],
        image_ids: Optional[Sequence[str]] = None,
        valid_action_mask: Optional[Tensor] = None,
    ) -> Tuple[List[GlobalPerceptionOutput], AgentBatchState]:
        """Run frozen global perception on a list of images and assemble a batched ``AgentBatchState``."""
        if not pil_images:
            raise ValueError("pil_images must be non-empty.")
        self._enforce_detector_frozen()
        percs = [self.detector_wrapper.forward_global(img) for img in pil_images]
        batch_state = self.state_encoder.encode_batch_state(
            perceptions=percs,
            device=self.policy_device,
            image_ids=image_ids,
            valid_action_mask=valid_action_mask,
        )
        return percs, batch_state

    @torch.no_grad()
    def predict(
        self,
        pil_image: Image.Image,
        mode: str = "greedy",
        k_max: Optional[int] = None,
        temperature: float = 1.0,
        stop_logit_bias: float = 0.0,
        valid_action_mask: Optional[Tensor] = None,
        forced_action: Optional[Sequence[int]] = None,
        sync_cuda_for_timing: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> AgentInferenceOutput:
        """Run end-to-end dual-path AZ-Agent inference on a single PIL image.

        Args:
            pil_image: Input RGB PIL image.
            mode: ``"greedy"`` (deterministic) or ``"sample"`` (stochastic policy sampling).
            k_max: Override max crops in ``{0, 1, 2}``; defaults to ``policy_head.k_max``.
            temperature: Softmax temperature (used when ``mode="sample"``).
            stop_logit_bias: Optional additive bias on the ``No-Crop`` logit for speed-accuracy tuning.
            valid_action_mask: Optional ``[1, 19]`` or ``[19]`` bool mask of allowed actions.
            forced_action: Optional explicit action tuple (e.g. ``()`` or ``(14,)`` or ``(3, 14)``)
                to override selection while still scoring it under ``policy_head``.
            sync_cuda_for_timing: Synchronize CUDA around timed stages for accurate latency breakdown.
            generator: Optional ``torch.Generator`` for reproducible stochastic sampling.
        """
        if mode not in {"greedy", "sample"}:
            raise ValueError(f"mode must be 'greedy' or 'sample', got {mode!r}")

        self._enforce_detector_frozen()
        policy_was_training = self.policy_head.training
        self.policy_head.eval()

        try:
            self._maybe_sync(sync_cuda_for_timing)
            t_start = time.perf_counter()

            # Stage 1: Frozen global perception
            t0 = time.perf_counter()
            perc = self.detector_wrapper.forward_global(pil_image)
            self._maybe_sync(sync_cuda_for_timing)
            global_ms = (time.perf_counter() - t0) * 1000.0

            orig_h, orig_w = perc.orig_size

            # Stage 2: Parameter-free state encoding -> GPU-ready detached FP32 state
            t1 = time.perf_counter()
            state = self.state_encoder.encode_state(
                perc=perc,
                device=self.policy_device,
                valid_action_mask=valid_action_mask,
            )
            self._maybe_sync(sync_cuda_for_timing)
            state_ms = (time.perf_counter() - t1) * 1000.0

            # Stage 3: Lightweight active-zoom policy forward + action decision
            t2 = time.perf_counter()
            logits = self.policy_head(state)  # [1, 19]
            if forced_action is not None:
                action: ActionTrajectory = tuple(int(w) for w in forced_action)
                lp_t = self.policy_head.log_prob_from_logits(
                    logits=logits,
                    actions=[action],
                    valid_action_mask=state.valid_action_mask,
                    k_max=k_max,
                    temperature=temperature if mode == "sample" else 1.0,
                    stop_logit_bias=stop_logit_bias,
                )
                log_p1, mask1 = self.policy_head.step1_log_probs(
                    logits=logits,
                    valid_action_mask=state.valid_action_mask,
                    temperature=temperature if mode == "sample" else 1.0,
                    stop_logit_bias=stop_logit_bias,
                )
                step1_probs = torch.where(mask1, log_p1.exp(), torch.zeros_like(log_p1))
                action_log_prob = float(lp_t[0].item())
            elif mode == "greedy":
                act_out = self.policy_head.act_greedy_from_logits(
                    logits=logits,
                    valid_action_mask=state.valid_action_mask,
                    k_max=k_max,
                    stop_logit_bias=stop_logit_bias,
                )
                action = act_out.actions[0]
                step1_probs = act_out.step1_probs
                action_log_prob = float(act_out.log_probs[0].item())
            else:
                samp_out = self.policy_head.sample_group_from_logits(
                    logits=logits,
                    G=1,
                    valid_action_mask=state.valid_action_mask,
                    k_max=k_max,
                    temperature=temperature,
                    stop_logit_bias=stop_logit_bias,
                    generator=generator,
                )
                action = samp_out.actions[0][0]
                step1_probs = samp_out.step1_probs
                action_log_prob = float(samp_out.log_probs[0, 0].item())

            self._maybe_sync(sync_cuda_for_timing)
            policy_ms = (time.perf_counter() - t2) * 1000.0

            global_det = self.fuser.fuse(perc.detections, [], perc.orig_size)

            # Stage 4: Fast path (No-Crop) vs Slow path (Isolated batch=1 single-crop + Dual-View WBF)
            selected_crop_boxes: List[Tuple[int, int, int, int]] = []
            if len(action) == 0:
                used_slow_path = False
                local_crop_ms = 0.0
                fusion_ms = 0.0
                final_det = global_det
            else:
                used_slow_path = True
                t3 = time.perf_counter()
                crop_dets: List[Dict[str, Tensor]] = []
                for wid in action:
                    rect = self.grid.windows[wid].to_pixel_xyxy(orig_w, orig_h)
                    selected_crop_boxes.append(rect)
                    crop_dets.append(self.detector_wrapper.forward_single_crop(pil_image, rect))
                self._maybe_sync(sync_cuda_for_timing)
                local_crop_ms = (time.perf_counter() - t3) * 1000.0

                t4 = time.perf_counter()
                final_det = self.fuser.fuse(perc.detections, crop_dets, perc.orig_size)
                fusion_ms = (time.perf_counter() - t4) * 1000.0

            self._maybe_sync(sync_cuda_for_timing)
            total_ms = (time.perf_counter() - t_start) * 1000.0

            return AgentInferenceOutput(
                detections=final_det,
                global_detections=global_det,
                action=action,
                used_slow_path=used_slow_path,
                action_log_prob=action_log_prob,
                action_prob=float(torch.exp(torch.tensor(action_log_prob)).item()),
                no_crop_prob=float(step1_probs[0, self.policy_head.no_crop_action].item()),
                window_logits=logits[0].detach().float().cpu().clone(),
                step1_probs=step1_probs[0].detach().float().cpu().clone(),
                selected_crop_boxes=selected_crop_boxes,
                timings_ms={
                    "global_perception_ms": round(global_ms, 3),
                    "state_encode_ms": round(state_ms, 3),
                    "policy_ms": round(policy_ms, 3),
                    "local_crop_det_ms": round(local_crop_ms, 3),
                    "fusion_ms": round(fusion_ms, 3),
                    "total_ms": round(total_ms, 3),
                },
                orig_size=perc.orig_size,
            )
        finally:
            if policy_was_training:
                self.policy_head.train(True)
            self._enforce_detector_frozen()

    def forward(
        self,
        inputs: Union[AgentBatchState, Image.Image],
        **kwargs,
    ) -> Union[Tensor, AgentInferenceOutput]:
        """Dispatch ``AgentBatchState`` -> trainable policy logits (with grad) or ``PIL.Image`` -> ``predict``."""
        self._enforce_detector_frozen()
        if isinstance(inputs, AgentBatchState):
            return self.policy_head(inputs)
        if isinstance(inputs, Image.Image):
            return self.predict(inputs, **kwargs)
        raise TypeError(f"Expected AgentBatchState or PIL.Image.Image, got {type(inputs).__name__}")
