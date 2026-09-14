from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# body panel only: glass above reflects the sky, the road shows below
SAMPLE_LEFT = 0.25
SAMPLE_RIGHT = 0.75
SAMPLE_TOP = 0.40
SAMPLE_BOTTOM = 0.85
MIN_SAMPLE_PIXELS = 40

HUE_RANGE = 180
HUE_SMOOTHING = 10

ACHROMATIC_SATURATION = 60
# dark paint still has saturation, and below this black cars come out blue
BLACK_VALUE = 110
WHITE_VALUE = 175

# cyan and magenta readings come from sky reflections on dark paint, so they fold into blue and red
HUE_NAMES = (
    (8, "red"),
    (20, "orange"),
    (33, "yellow"),
    (78, "green"),
    (150, "blue"),
)

PALETTE = {
    "black": (50, 50, 50),
    "grey": (160, 160, 160),
    "white": (255, 255, 255),
    "red": (0, 0, 255),
    "orange": (0, 140, 255),
    "yellow": (0, 220, 220),
    "green": (0, 200, 0),
    "blue": (255, 80, 0),
}


@dataclass(frozen=True)
class VehicleColor:
    name: str
    bgr: tuple[int, int, int]


def color_from_name(name: str) -> VehicleColor:
    return VehicleColor(name, PALETTE[name])


def sample_patch(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray | None:
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        return None

    frame_height, frame_width = frame.shape[:2]
    left = max(x1 + int(width * SAMPLE_LEFT), 0)
    right = min(x1 + int(width * SAMPLE_RIGHT), frame_width)
    top = max(y1 + int(height * SAMPLE_TOP), 0)
    bottom = min(y1 + int(height * SAMPLE_BOTTOM), frame_height)

    if (right - left) * (bottom - top) < MIN_SAMPLE_PIXELS:
        return None
    return frame[top:bottom, left:right]


def dominant_hue(hues: np.ndarray) -> int:
    counts = np.bincount(hues.ravel(), minlength=HUE_RANGE).astype(np.float64)
    # hue wraps around; a flat window would leave plateaus and argmax would pick their edge
    offsets = np.arange(-HUE_SMOOTHING, HUE_SMOOTHING + 1)
    weights = HUE_SMOOTHING + 1 - np.abs(offsets)
    smoothed = sum(
        weight * np.roll(counts, int(offset))
        for offset, weight in zip(offsets, weights, strict=True)
    )
    return int(np.argmax(smoothed))


def name_for_hue(hue: int) -> str:
    for upper, name in HUE_NAMES:
        if hue < upper:
            return name
    return "red"


def name_for_brightness(value: float) -> str:
    if value < BLACK_VALUE:
        return "black"
    if value < WHITE_VALUE:
        return "grey"
    return "white"


def estimate_color(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> VehicleColor | None:
    patch = sample_patch(frame, bbox)
    if patch is None:
        return None

    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hue, saturation, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    level = float(np.median(value))
    # hue carries no information in near darkness, where sensor noise inflates saturation
    if level < BLACK_VALUE:
        return color_from_name("black")

    if float(np.median(saturation)) < ACHROMATIC_SATURATION:
        return color_from_name(name_for_brightness(level))

    colored = saturation >= ACHROMATIC_SATURATION
    return color_from_name(name_for_hue(dominant_hue(hue[colored])))
