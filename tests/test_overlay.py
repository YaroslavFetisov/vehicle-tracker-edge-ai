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


def make_state(track_id: int, color: str) -> TrackState:
    state = TrackState(track_id=track_id, last_seen=0, min_plate_score=2.0)
    state.color_votes[color_from_name(color)] += 1
    return state


def state_with_color(name: str) -> dict[int, TrackState]:
    return {3: make_state(3, name)}


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


def test_nearer_vehicles_are_drawn_over_distant_ones():
    frame = make_frame()
    far = Detection(track_id=1, bbox=(20, 40, 40, 55), class_id=2, confidence=0.9)
    near = Detection(track_id=2, bbox=(10, 40, 120, 100), class_id=2, confidence=0.9)
    states = {1: make_state(1, "red"), 2: make_state(2, "white")}

    # the nearer vehicle is passed first, so list order alone would draw it underneath
    draw_detections(frame, [near, far], states)

    # Both labels cover this point, so it shows which of them was drawn last. The blue
    # channel separates the white label from the red one even under the antialiased text.
    blue = int(frame[30, 60][0])
    assert blue > 200, "the distant vehicle's label was drawn over the nearer one"


def test_label_of_a_vehicle_at_the_frame_edge_stays_visible():
    frame = make_frame()
    at_edge = Detection(
        track_id=3, bbox=(FRAME_WIDTH - 15, 40, FRAME_WIDTH - 5, 60), class_id=2, confidence=0.9
    )
    state = make_state(3, "white")
    state.plate_votes["AA1234BB"] = 5.0

    draw_detections(frame, [at_edge], {3: state})

    assert frame[:40, : FRAME_WIDTH // 2].any(), "the label was clipped off the frame"


def test_label_stays_readable_on_light_and_dark_boxes():
    assert text_color(color_from_name("white").bgr) == DARK_TEXT
    assert text_color(color_from_name("black").bgr) == LIGHT_TEXT
