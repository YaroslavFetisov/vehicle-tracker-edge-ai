from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    track_id: int
    bbox: tuple[int, int, int, int]
    class_id: int
    confidence: float
