from __future__ import annotations

import numpy as np

from vehicle_tracker.detection import Detection
from vehicle_tracker.overlay import draw_detections

FRAME_HEIGHT = 120
FRAME_WIDTH = 160
BOX = (20, 40, 100, 90)


def make_frame() -> np.ndarray:
    return np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)


def make_detection() -> Detection:
    return Detection(track_id=3, bbox=BOX, class_id=2, confidence=0.9)


def test_box_edge_is_drawn():
    frame = make_frame()
    draw_detections(frame, [make_detection()])
    assert frame[40, 60].any()


def test_area_outside_the_box_is_untouched():
    frame = make_frame()
    draw_detections(frame, [make_detection()])
    assert not frame[FRAME_HEIGHT - 1, FRAME_WIDTH - 1].any()


def test_frame_shape_is_preserved():
    frame = make_frame()
    draw_detections(frame, [make_detection()])
    assert frame.shape == (FRAME_HEIGHT, FRAME_WIDTH, 3)


def test_without_detections_the_frame_is_untouched():
    frame = make_frame()
    before = frame.copy()
    draw_detections(frame, [])
    assert np.array_equal(frame, before)
