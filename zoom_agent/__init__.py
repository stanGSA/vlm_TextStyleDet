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
]
