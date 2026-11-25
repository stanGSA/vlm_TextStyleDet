"""AZ-Agent (Active Zoom Agent) modular package for TextStyleDet."""

from __future__ import annotations

import sys

# Prevent Python from writing .pyc bytecode into read-only upstream repositories
sys.dont_write_bytecode = True

from .agent_pipeline import AZAgentDetector, AgentInferenceOutput
from .candidate_grid import CandidateGrid, CropWindow
from .detector_wrapper import FrozenTextStyleDetWrapper, GlobalPerceptionOutput
from .policy_head import ActiveZoomPolicyHead, PolicyActionOutput, PolicyGroupSampleOutput
from .selectors import OracleSelector, RPNHeuristicSelector
from .sft_cache import (
    ShardedSFTCacheReader,
    build_sharded_sft_cache,
    compute_completed_seal_digest,
    seal_completed_cache_manifest,
    verify_cache_manifest,
)
from .sft_trainer import SFTTrainConfig, evaluate_sft_e2e_on_source_dev, train_sft_policy_offline
from .grpo_env import (
    GRPODetectionEnvironment,
    GRPORewardConfig,
    PersistentDetectionEnvCache,
    RewardBreakdown,
    SmoothDetectionRewarder,
    build_env_cache_manifest_payload,
    canonicalize_action_windows,
    verify_env_cache_manifest,
)
from .grpo_trainer import (
    GRPOTrainConfig,
    SubsetSFTCacheView,
    compute_group_advantages,
    compute_grpo_loss_components,
    evaluate_phase3_e2e_on_source_dev,
    generate_phase3_markdown_report,
    train_grpo_policy,
    verify_grpo_step0_parity,
)
from .source_degradation import (
    ADVERSE_DEGRADATION_TAGS,
    ALLOWED_SOURCE_DOMAIN,
    ALLOWED_SOURCE_SPLIT,
    DEGRADATION_TAGS,
    SourceOnlyDatasetLoader,
    SourceSplitInfo,
    apply_physical_degradation,
    validate_source_only_access,
)
from .state_encoder import AgentBatchState, StateEncoder, ensure_grad_safe_fp32
from .wbf_fusion import DualViewWBFFusion

__all__ = [
    "CandidateGrid",
    "CropWindow",
    "FrozenTextStyleDetWrapper",
    "GlobalPerceptionOutput",
    "StateEncoder",
    "AgentBatchState",
    "ensure_grad_safe_fp32",
    "ActiveZoomPolicyHead",
    "PolicyActionOutput",
    "PolicyGroupSampleOutput",
    "AZAgentDetector",
    "AgentInferenceOutput",
    "DualViewWBFFusion",
    "RPNHeuristicSelector",
    "OracleSelector",
    "ALLOWED_SOURCE_DOMAIN",
    "ALLOWED_SOURCE_SPLIT",
    "DEGRADATION_TAGS",
    "ADVERSE_DEGRADATION_TAGS",
    "SourceOnlyDatasetLoader",
    "SourceSplitInfo",
    "validate_source_only_access",
    "apply_physical_degradation",
    "ShardedSFTCacheReader",
    "build_sharded_sft_cache",
    "compute_completed_seal_digest",
    "seal_completed_cache_manifest",
    "verify_cache_manifest",
    "SFTTrainConfig",
    "train_sft_policy_offline",
    "evaluate_sft_e2e_on_source_dev",
    "GRPORewardConfig",
    "RewardBreakdown",
    "SmoothDetectionRewarder",
    "PersistentDetectionEnvCache",
    "GRPODetectionEnvironment",
    "build_env_cache_manifest_payload",
    "verify_env_cache_manifest",
    "canonicalize_action_windows",
    "GRPOTrainConfig",
    "SubsetSFTCacheView",
    "compute_group_advantages",
    "compute_grpo_loss_components",
    "verify_grpo_step0_parity",
    "train_grpo_policy",
    "evaluate_phase3_e2e_on_source_dev",
    "generate_phase3_markdown_report",
]
