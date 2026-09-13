from __future__ import annotations

import numpy as np
import pytest

from vehicle_tracker.color import (
    SAMPLE_BOTTOM,
    SAMPLE_TOP,
    dominant_hue,
    estimate_color,
    name_for_hue,
)

FRAME_SIZE = 200
BBOX = (40, 40, 160, 160)


def solid_frame(bgr: tuple[int, int, int]) -> np.ndarray:
    frame = np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8)
    frame[:, :] = bgr
    return frame


@pytest.mark.parametrize(
    ("bgr", "expected"),
    [
        ((0, 0, 200), "red"),
        ((0, 140, 230), "orange"),
        ((0, 200, 200), "yellow"),
        ((0, 180, 0), "green"),
        ((200, 0, 0), "blue"),
        ((255, 255, 255), "white"),
        ((130, 130, 130), "grey"),
        ((20, 20, 20), "black"),
    ],
)
def test_solid_colors_are_recognised(bgr, expected):
    color = estimate_color(solid_frame(bgr), BBOX)
    assert color is not None
    assert color.name == expected


def test_only_the_body_band_is_sampled():
    frame = np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8)
    x1, y1, x2, y2 = BBOX
    height = y2 - y1
    frame[y1 : y1 + int(height * SAMPLE_TOP)] = (0, 0, 200)
    frame[y1 + int(height * SAMPLE_TOP) : y1 + int(height * SAMPLE_BOTTOM)] = (200, 0, 0)
    frame[y1 + int(height * SAMPLE_BOTTOM) : y2] = (0, 200, 0)

    color = estimate_color(frame, BBOX)
    assert color is not None
    assert color.name == "blue"


def test_dark_paint_reads_as_black_rather_than_as_its_hue():
    # dark blue paint keeps enough saturation to be classified by hue, and a car that is
    # plainly black then comes out blue
    color = estimate_color(solid_frame((100, 40, 35)), BBOX)
    assert color is not None
    assert color.name == "black"


def test_a_lit_colour_is_still_read_as_a_colour():
    # the guard above must not swallow vehicles that are genuinely coloured
    color = estimate_color(solid_frame((200, 70, 60)), BBOX)
    assert color is not None
    assert color.name == "blue"


def test_box_reaching_past_the_frame_edge_is_clamped():
    color = estimate_color(solid_frame((200, 0, 0)), (100, 100, 260, 260))
    assert color is not None
    assert color.name == "blue"


def test_box_too_small_to_sample_is_rejected():
    assert estimate_color(solid_frame((0, 0, 200)), (10, 10, 13, 13)) is None


def test_empty_box_is_rejected():
    assert estimate_color(solid_frame((0, 0, 200)), (50, 50, 50, 50)) is None


def test_dark_paint_reads_as_black_despite_its_hue():
    color = estimate_color(solid_frame((60, 15, 15)), BBOX)
    assert color is not None
    assert color.name == "black"


def test_magenta_reading_folds_into_red():
    color = estimate_color(solid_frame((120, 20, 160)), BBOX)
    assert color is not None
    assert color.name == "red"


def test_cyan_reading_folds_into_blue():
    color = estimate_color(solid_frame((180, 180, 20)), BBOX)
    assert color is not None
    assert color.name == "blue"


def test_dominant_hue_handles_the_red_wrap_around():
    hues = np.array([178, 179, 0, 1, 2] * 20, dtype=np.uint8)
    assert name_for_hue(dominant_hue(hues)) == "red"
