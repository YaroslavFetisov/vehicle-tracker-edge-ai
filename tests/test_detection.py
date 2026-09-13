from __future__ import annotations

import numpy as np

from vehicle_tracker.detection import crop_to_bbox


def frame() -> np.ndarray:
    return np.zeros((100, 200, 3), dtype=np.uint8)


def test_a_box_inside_the_frame_is_cropped_as_is():
    crop = crop_to_bbox(frame(), (10, 20, 60, 50))

    assert crop is not None
    assert crop.shape[:2] == (30, 50)


def test_a_box_running_off_the_right_edge_is_clamped():
    crop = crop_to_bbox(frame(), (150, 10, 260, 40))

    assert crop is not None
    assert crop.shape[:2] == (30, 50)


def test_a_box_starting_outside_the_frame_is_clamped():
    crop = crop_to_bbox(frame(), (-40, -10, 60, 50))

    assert crop is not None
    assert crop.shape[:2] == (50, 60)


def test_a_box_fully_outside_the_frame_yields_nothing():
    # a negative slice bound would wrap around and return the opposite edge instead
    assert crop_to_bbox(frame(), (-80, -60, -20, -10)) is None
    assert crop_to_bbox(frame(), (220, 120, 300, 160)) is None


def test_an_empty_box_yields_nothing():
    assert crop_to_bbox(frame(), (30, 40, 30, 40)) is None
