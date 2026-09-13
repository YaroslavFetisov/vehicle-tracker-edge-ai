from __future__ import annotations

import numpy as np

from vehicle_tracker.color import color_from_name
from vehicle_tracker.detection import Detection
from vehicle_tracker.overlay import DARK_TEXT, LIGHT_TEXT, draw_detections, text_color
from vehicle_tracker.tracks import TrackState

FRAME_HEIGHT = 120
FRAME_WIDTH = 160
BOX = (20, 40, 100, 90)


def make_frame() -> np.ndarray:
    return np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)


def make_detection() -> Detection:
    return Detection(track_id=3, bbox=BOX, class_id=2, confidence=0.9)


def state_with_color(name: str) -> dict[int, TrackState]:
    state = TrackState(track_id=3, last_seen=0)
    state.color_votes[color_from_name(name)] += 1
    return {3: state}


def test_box_edge_is_drawn():
    frame = make_frame()
    draw_detections(frame, [make_detection()], {})
    assert frame[40, 60].any()


def test_area_outside_the_box_is_untouched():
    frame = make_frame()
    draw_detections(frame, [make_detection()], {})
    assert not frame[FRAME_HEIGHT - 1, FRAME_WIDTH - 1].any()


def test_frame_shape_is_preserved():
    frame = make_frame()
    draw_detections(frame, [make_detection()], {})
    assert frame.shape == (FRAME_HEIGHT, FRAME_WIDTH, 3)


def test_without_detections_the_frame_is_untouched():
    frame = make_frame()
    before = frame.copy()
    draw_detections(frame, [], {})
    assert np.array_equal(frame, before)


def test_box_takes_the_vehicle_color():
    frame = make_frame()
    draw_detections(frame, [make_detection()], state_with_color("red"))
    assert tuple(int(channel) for channel in frame[40, 60]) == color_from_name("red").bgr


def test_label_stays_readable_on_light_and_dark_boxes():
    assert text_color(color_from_name("white").bgr) == DARK_TEXT
    assert text_color(color_from_name("black").bgr) == LIGHT_TEXT
