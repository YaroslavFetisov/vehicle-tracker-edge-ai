from __future__ import annotations

import numpy as np

from vehicle_tracker.detection import Detection
from vehicle_tracker.tracks import TrackRegistry

FRAME_SIZE = 200
BBOX = (40, 40, 160, 160)
RED = (0, 0, 200)
BLUE = (200, 0, 0)


def solid_frame(bgr: tuple[int, int, int]) -> np.ndarray:
    frame = np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8)
    frame[:, :] = bgr
    return frame


def detection(track_id: int = 1) -> Detection:
    return Detection(track_id=track_id, bbox=BBOX, class_id=2, confidence=0.9)


def test_majority_vote_survives_a_few_odd_frames():
    registry = TrackRegistry(color_interval=1)
    for frame_index in range(8):
        registry.update(frame_index, solid_frame(RED), [detection()])
    for frame_index in range(8, 11):
        states = registry.update(frame_index, solid_frame(BLUE), [detection()])

    assert states[1].color is not None
    assert states[1].color.name == "red"


def test_color_is_not_sampled_on_every_frame():
    registry = TrackRegistry(color_interval=5)
    for frame_index in range(5):
        states = registry.update(frame_index, solid_frame(RED), [detection()])

    assert states[1].color_samples == 1


def test_sampling_stops_at_the_limit():
    registry = TrackRegistry(color_interval=1, max_color_samples=3)
    for frame_index in range(20):
        states = registry.update(frame_index, solid_frame(RED), [detection()])

    assert states[1].color_samples == 3


def test_tracks_are_kept_apart():
    registry = TrackRegistry(color_interval=1)
    registry.update(0, solid_frame(RED), [detection(track_id=1)])
    states = registry.update(1, solid_frame(BLUE), [detection(track_id=2)])

    assert states[2].color is not None
    assert states[2].color.name == "blue"


def test_vehicles_that_left_the_scene_are_forgotten():
    registry = TrackRegistry(expiry_frames=10)
    registry.update(0, solid_frame(RED), [detection(track_id=1)])
    registry.update(50, solid_frame(RED), [detection(track_id=2)])

    assert len(registry) == 1


def test_state_without_samples_reports_no_color():
    registry = TrackRegistry(max_color_samples=0)
    states = registry.update(0, solid_frame(RED), [detection()])

    assert states[1].color is None
