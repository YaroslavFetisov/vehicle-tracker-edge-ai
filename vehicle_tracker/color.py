from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# The body panel is sampled instead of the whole box: the upper part of a vehicle is glass
# reflecting the sky, and the bottom edge is road surface. The band reaches lower than the
# windows suggest, because the body of a bus or a van sits below where a car's does.
SAMPLE_LEFT = 0.25
SAMPLE_RIGHT = 0.75
SAMPLE_TOP = 0.40
SAMPLE_BOTTOM = 0.85
MIN_SAMPLE_PIXELS = 40

HUE_RANGE = 180
HUE_SMOOTHING = 10

ACHROMATIC_SATURATION = 60
# Raised from 70 after measuring on labelled vehicles: dark paint keeps enough saturation to
# be read as a hue, so cars that are plainly black came out blue. Below this brightness the
# colour of a vehicle is not recoverable and black is the honest answer. Higher still starts
# calling a yellow machine black, which is why the threshold sits here.
BLACK_VALUE = 110
WHITE_VALUE = 175

# Cyan, purple and magenta are not production car colours: such readings come from dark
# red paint under a blue sky rather than from the vehicle, so they fold into blue and red.
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
    # Hue wraps around, so the histogram is smoothed with a window that wraps too.
    # The window is triangular rather than flat: a flat one leaves plateaus of equal
    # value around a peak, and argmax then reports their left edge instead of the peak.
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
