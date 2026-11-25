"""AZ-Agent (Active Zoom Agent) modular package for TextStyleDet."""

from __future__ import annotations

import sys

# Prevent Python from writing .pyc bytecode into read-only upstream repositories
sys.dont_write_bytecode = True

from .candidate_grid import CandidateGrid, CropWindow
from .detector_wrapper import FrozenTextStyleDetWrapper
from .selectors import OracleSelector, RPNHeuristicSelector
from .state_encoder import StateEncoder
from .wbf_fusion import DualViewWBFFusion

__all__ = [
    "CandidateGrid",
    "CropWindow",
    "FrozenTextStyleDetWrapper",
    "StateEncoder",
    "DualViewWBFFusion",
    "RPNHeuristicSelector",
    "OracleSelector",
]
