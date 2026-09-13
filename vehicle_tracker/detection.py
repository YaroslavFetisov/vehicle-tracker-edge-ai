from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Detection:
    track_id: int
    bbox: tuple[int, int, int, int]
    class_id: int
    confidence: float


def crop_to_bbox(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray | None:
    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    # a tracker can predict a box that has left the frame, and a negative slice bound
    # would wrap around to the opposite edge instead of yielding nothing
    left, top = max(x1, 0), max(y1, 0)
    right, bottom = min(x2, frame_width), min(y2, frame_height)
    if right <= left or bottom <= top:
        return None
    return frame[top:bottom, left:right]
