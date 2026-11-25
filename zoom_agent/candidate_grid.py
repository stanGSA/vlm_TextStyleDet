"""Multi-scale overlapping discrete candidate crop pool (Module 2).

Generates N=18 deterministic overlapping crop windows:
  - 12 standard overlapping grid windows (3 rows x 4 cols, width=0.35 x height=0.42,
    horizontal adjacent overlap 38.10%, vertical adjacent overlap 30.95%, ~6.80x area zoom).
  - 6 horizon-band reinforcement windows covering vertical [35%, 73%] driving far-field
    small-object hotspot zone:
      * 4 fine far-field windows (0.32 x 0.34, y in [0.36, 0.70], ~9.19x area zoom)
      * 2 central-horizon windows (0.46 x 0.38, y in [0.35, 0.73], ~5.72x area zoom)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class CropWindow:
    """Normalized and pixel-level crop window descriptor."""

    window_id: int
    name: str
    group: str  # "grid_3x4" or "horizon_band"
    norm_xyxy: Tuple[float, float, float, float]  # (x1, y1, x2, y2) in [0, 1]

    def to_pixel_xyxy(self, orig_w: int, orig_h: int) -> Tuple[int, int, int, int]:
        if orig_w < 4 or orig_h < 4:
            raise ValueError(f"Image too small for cropping: {(orig_w, orig_h)}")
        nx1, ny1, nx2, ny2 = self.norm_xyxy
        x1 = max(0, min(int(round(nx1 * orig_w)), orig_w - 2))
        y1 = max(0, min(int(round(ny1 * orig_h)), orig_h - 2))
        x2 = max(x1 + 2, min(int(round(nx2 * orig_w)), orig_w))
        y2 = max(y1 + 2, min(int(round(ny2 * orig_h)), orig_h))
        return x1, y1, x2, y2


class CandidateGrid:
    """N=18 overlapping discrete crop window pool + No-Crop index (N)."""

    def __init__(
        self,
        grid_rows: int = 3,
        grid_cols: int = 4,
        grid_win_w: float = 0.35,
        grid_win_h: float = 0.42,
    ) -> None:
        if grid_rows < 1 or grid_cols < 1:
            raise ValueError(f"Invalid grid_rows={grid_rows}, grid_cols={grid_cols}")
        if not (0.05 < grid_win_w < 1.0 and 0.05 < grid_win_h < 1.0):
            raise ValueError(f"Invalid window size: {(grid_win_w, grid_win_h)}")
        self.grid_rows = int(grid_rows)
        self.grid_cols = int(grid_cols)
        self.grid_win_w = float(grid_win_w)
        self.grid_win_h = float(grid_win_h)
        self.windows: List[CropWindow] = self._build_windows()
        self.num_windows: int = len(self.windows)
        self.no_crop_action: int = self.num_windows  # index 18

    def config_dict(self) -> Dict:
        return {
            "grid_rows": self.grid_rows,
            "grid_cols": self.grid_cols,
            "grid_win_w": self.grid_win_w,
            "grid_win_h": self.grid_win_h,
            "num_windows": self.num_windows,
            "windows": [
                {"id": w.window_id, "name": w.name, "group": w.group, "norm_xyxy": list(w.norm_xyxy)}
                for w in self.windows
            ],
        }

    def _build_windows(self) -> List[CropWindow]:
        windows: List[CropWindow] = []
        wid = 0

        xs = np.linspace(0.0, 1.0 - self.grid_win_w, self.grid_cols) if self.grid_cols > 1 else [0.0]
        ys = np.linspace(0.0, 1.0 - self.grid_win_h, self.grid_rows) if self.grid_rows > 1 else [0.0]
        for r, y1 in enumerate(ys):
            for c, x1 in enumerate(xs):
                windows.append(
                    CropWindow(
                        window_id=wid,
                        name=f"grid_r{r}_c{c}",
                        group="grid_3x4",
                        norm_xyxy=(
                            round(float(x1), 4),
                            round(float(y1), 4),
                            round(float(x1 + self.grid_win_w), 4),
                            round(float(y1 + self.grid_win_h), 4),
                        ),
                    )
                )
                wid += 1

        fine_w, fine_h, fine_y1 = 0.32, 0.34, 0.36
        fine_xs = np.linspace(0.0, 1.0 - fine_w, 4)
        for c, x1 in enumerate(fine_xs):
            windows.append(
                CropWindow(
                    window_id=wid,
                    name=f"horizon_fine_c{c}",
                    group="horizon_band",
                    norm_xyxy=(
                        round(float(x1), 4),
                        fine_y1,
                        round(float(x1 + fine_w), 4),
                        round(fine_y1 + fine_h, 4),
                    ),
                )
            )
            wid += 1

        mid_specs = [
            ("horizon_mid_left", (0.08, 0.35, 0.54, 0.73)),
            ("horizon_mid_right", (0.46, 0.35, 0.92, 0.73)),
        ]
        for name, coords in mid_specs:
            windows.append(
                CropWindow(
                    window_id=wid,
                    name=name,
                    group="horizon_band",
                    norm_xyxy=coords,
                )
            )
            wid += 1

        return windows

    def get_pixel_boxes(self, orig_w: int, orig_h: int) -> Tensor:
        coords = [w.to_pixel_xyxy(orig_w, orig_h) for w in self.windows]
        return torch.tensor(coords, dtype=torch.float32)

    def get_normalized_boxes(self) -> Tensor:
        coords = [w.norm_xyxy for w in self.windows]
        return torch.tensor(coords, dtype=torch.float32)

    def window_iou_matrix(self) -> Tensor:
        boxes = self.get_normalized_boxes()
        x1 = torch.max(boxes[:, None, 0], boxes[None, :, 0])
        y1 = torch.max(boxes[:, None, 1], boxes[None, :, 1])
        x2 = torch.min(boxes[:, None, 2], boxes[None, :, 2])
        y2 = torch.min(boxes[:, None, 3], boxes[None, :, 3])
        inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = area[:, None] + area[None, :] - inter
        return inter / union.clamp(min=1e-6)
