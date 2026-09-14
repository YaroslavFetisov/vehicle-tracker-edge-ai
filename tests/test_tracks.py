from __future__ import annotations

import numpy as np

from vehicle_tracker.detection import Detection
from vehicle_tracker.plate import PlateReading
from vehicle_tracker.tracks import (
    MIN_REPORTED_PLATE_SCORE,
    TrackRegistry,
    carries_a_plate,
    newly_confirmed,
)

FRAME_SIZE = 200
BBOX = (40, 40, 160, 160)
RED = (0, 0, 200)
BLUE = (200, 0, 0)


def solid_frame(bgr: tuple[int, int, int]) -> np.ndarray:
    frame = np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8)
    frame[:, :] = bgr
    return frame


def detection(track_id: int = 1, shift: int = 0) -> Detection:
    x1, y1, x2, y2 = BBOX
    return Detection(
        track_id=track_id, bbox=(x1 + shift, y1, x2 + shift, y2), class_id=2, confidence=0.9
    )


# BBOX is 120 pixels tall, so a tenth of its height is twelve and this clears the bar
TRAVELLED = 20


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


def reading(text: str, confidence: float = 0.9) -> PlateReading:
    return PlateReading(text=text, ocr_confidence=confidence)


def test_large_vehicles_are_queued_for_plate_reading():
    registry = TrackRegistry()
    registry.update(0, solid_frame(RED), [detection()])

    assert registry.due_for_plate(10, [detection()]) == [detection()]


def test_vehicles_too_small_for_the_frame_are_not_worth_an_ocr_pass():
    registry = TrackRegistry()
    small = Detection(track_id=1, bbox=(0, 0, 50, 10), class_id=2, confidence=0.9)
    registry.update(0, solid_frame(RED), [small])

    assert registry.due_for_plate(10, [small]) == []


def test_vehicle_size_is_measured_against_the_frame_not_in_pixels():
    vehicle = Detection(track_id=1, bbox=(0, 0, 60, 40), class_id=2, confidence=0.9)
    small_frame = np.zeros((200, 200, 3), dtype=np.uint8)
    large_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    on_small = TrackRegistry()
    on_small.update(0, small_frame, [vehicle])
    on_large = TrackRegistry()
    on_large.update(0, large_frame, [vehicle])

    assert on_small.due_for_plate(10, [vehicle]) == [vehicle]
    assert on_large.due_for_plate(10, [vehicle]) == []


def test_vehicles_appearing_together_do_not_queue_on_the_same_frame():
    registry = TrackRegistry(plate_interval=5, max_plate_reads_per_frame=10)
    together = [detection(track_id) for track_id in (1, 2, 3)]

    per_frame = []
    for frame_index in range(6):
        registry.update(frame_index, solid_frame(RED), together)
        due = registry.due_for_plate(frame_index, together)
        per_frame.append(len(due))
        for served in due:
            registry.record_plate(served.track_id, frame_index, None)

    assert max(per_frame) == 1
    assert sum(per_frame) == len(together)


def test_plate_reads_are_capped_per_frame():
    registry = TrackRegistry(plate_interval=1, max_plate_reads_per_frame=2)
    crowd = [detection(track_id) for track_id in range(1, 6)]
    registry.update(0, solid_frame(RED), crowd)

    assert len(registry.due_for_plate(10, crowd)) == 2


def test_the_longest_waiting_vehicle_is_served_first():
    registry = TrackRegistry(plate_interval=1, max_plate_reads_per_frame=1)
    waiting = [detection(1), detection(2)]
    registry.update(0, solid_frame(RED), waiting)
    registry.record_plate(1, 5, None)

    assert [served.track_id for served in registry.due_for_plate(10, waiting)] == [2]


def test_ocr_is_not_retried_on_every_frame():
    registry = TrackRegistry(plate_interval=5)
    registry.update(0, solid_frame(RED), [detection()])
    registry.record_plate(1, 0, reading("AA1234BB"))

    assert registry.due_for_plate(1, [detection()]) == []


def test_ocr_stops_once_the_answer_is_settled():
    registry = TrackRegistry(plate_interval=1, confident_plate_score=2.0)
    registry.update(0, solid_frame(RED), [detection()])
    for frame_index in range(4):
        registry.record_plate(1, frame_index, reading("AA1234BB"))

    assert registry.due_for_plate(10, [detection()]) == []


def test_ocr_gives_up_after_enough_failed_attempts():
    registry = TrackRegistry(plate_interval=1, max_plate_attempts=3)
    registry.update(0, solid_frame(RED), [detection()])
    for frame_index in range(3):
        registry.record_plate(1, frame_index, None)

    assert registry.due_for_plate(10, [detection()]) == []


def approaching(height: int, track_id: int = 1) -> Detection:
    return Detection(track_id=track_id, bbox=(40, 0, 160, height), class_id=2, confidence=0.9)


def exhaust_budget(registry: TrackRegistry, vehicle: Detection, attempts: int) -> None:
    registry.update(0, solid_frame(RED), [vehicle])
    for frame_index in range(attempts):
        registry.record_plate(vehicle.track_id, frame_index, None)


def test_a_vehicle_that_has_not_come_closer_is_not_retried():
    registry = TrackRegistry(plate_interval=1, max_plate_attempts=3)
    far = approaching(40)
    exhaust_budget(registry, far, 3)
    registry.update(4, solid_frame(RED), [far])

    assert registry.due_for_plate(10, [far]) == []


def test_a_vehicle_that_has_come_closer_gets_another_chance():
    registry = TrackRegistry(plate_interval=1, max_plate_attempts=3, retry_growth=1.5)
    far, near = approaching(40), approaching(60)
    exhaust_budget(registry, far, 3)
    states = registry.update(4, solid_frame(RED), [near])

    assert states[1].plate_attempts == 0
    assert registry.due_for_plate(10, [near]) == [near]


def test_coming_a_little_closer_is_not_enough_to_retry():
    registry = TrackRegistry(plate_interval=1, max_plate_attempts=3, retry_growth=1.5)
    far, slightly_nearer = approaching(40), approaching(55)
    exhaust_budget(registry, far, 3)
    registry.update(4, solid_frame(RED), [slightly_nearer])

    assert registry.due_for_plate(10, [slightly_nearer]) == []


def test_every_further_retry_costs_another_approach():
    registry = TrackRegistry(plate_interval=1, max_plate_attempts=3, retry_growth=1.5)
    exhaust_budget(registry, approaching(40), 3)

    registry.update(4, solid_frame(RED), [approaching(60)])
    for frame_index in range(5, 8):
        registry.record_plate(1, frame_index, None)

    registry.update(9, solid_frame(RED), [approaching(80)])
    assert registry.due_for_plate(10, [approaching(80)]) == []

    registry.update(10, solid_frame(RED), [approaching(90)])
    assert registry.due_for_plate(11, [approaching(90)]) == [approaching(90)]


def test_a_settled_plate_is_not_reread_when_the_vehicle_comes_closer():
    registry = TrackRegistry(plate_interval=1, max_plate_attempts=3, confident_plate_score=2.0)
    far, near = approaching(40), approaching(120)
    registry.update(0, solid_frame(RED), [far])
    for frame_index in range(3):
        registry.record_plate(1, frame_index, reading("AA1234BB"))
    registry.update(4, solid_frame(RED), [near])

    assert registry.due_for_plate(10, [near]) == []


def test_plate_majority_wins_over_scattered_misreadings():
    registry = TrackRegistry()
    registry.update(0, solid_frame(RED), [detection()])
    for frame_index, text in enumerate(["CF5775", "CF5775", "CF5775", "EF5775", "CF5715"]):
        registry.record_plate(1, frame_index, reading(text))

    states = registry.update(1, solid_frame(RED), [detection(shift=TRAVELLED)])
    assert states[1].plate == "CF5775"


def test_local_format_breaks_a_tie():
    registry = TrackRegistry(min_plate_score=0.0, min_plate_lead=0.0, min_agreeing_readings=1)
    registry.update(0, solid_frame(RED), [detection()])
    registry.record_plate(1, 0, reading("AA1234BB"))
    registry.record_plate(1, 1, reading("CF57751"))

    states = registry.update(2, solid_frame(RED), [detection(shift=TRAVELLED)])
    assert states[1].plate == "AA1234BB"


def test_a_single_reading_is_not_reported_yet():
    registry = TrackRegistry(min_plate_score=2.0)
    registry.update(0, solid_frame(RED), [detection()])
    registry.record_plate(1, 0, reading("CF5775"))

    states = registry.update(1, solid_frame(RED), [detection(shift=TRAVELLED)])
    assert states[1].plate is None


def test_a_plate_is_reported_once_frames_agree():
    registry = TrackRegistry(min_plate_score=2.0)
    registry.update(0, solid_frame(RED), [detection()])
    for frame_index in range(3):
        registry.record_plate(1, frame_index, reading("CF5775"))

    states = registry.update(3, solid_frame(RED), [detection(shift=TRAVELLED)])
    assert states[1].plate == "CF5775"


def test_a_box_that_never_travels_reports_no_plate():
    # the identification number on a road sign reads perfectly on every frame, so without
    # this the only plate the highway clip yields is street furniture
    registry = TrackRegistry()
    registry.update(0, solid_frame(RED), [detection()])
    for frame_index in range(4):
        registry.record_plate(1, frame_index, reading("9448A"))

    states = registry.update(1, solid_frame(RED), [detection()])
    assert states[1].plate_score > 0
    assert states[1].plate is None


def test_a_plate_is_reported_once_the_vehicle_has_travelled():
    registry = TrackRegistry()
    registry.update(0, solid_frame(RED), [detection()])
    for frame_index in range(4):
        registry.record_plate(1, frame_index, reading("CF5775"))

    assert registry.update(1, solid_frame(RED), [detection()])[1].plate is None
    assert registry.update(2, solid_frame(RED), [detection(shift=TRAVELLED)])[1].plate == "CF5775"


def test_a_vehicle_that_stops_keeps_the_plate_it_earned():
    # a car waiting at a barrier has still arrived under its own power
    registry = TrackRegistry()
    registry.update(0, solid_frame(RED), [detection()])
    for frame_index in range(4):
        registry.record_plate(1, frame_index, reading("CF5775"))
    registry.update(1, solid_frame(RED), [detection(shift=TRAVELLED)])

    states = registry.update(2, solid_frame(RED), [detection()])
    assert states[1].plate == "CF5775"


def test_travel_is_judged_against_the_vehicle_own_size():
    frame = solid_frame(RED)
    nudge = 5
    distant = Detection(track_id=1, bbox=(0, 0, 30, 20), class_id=2, confidence=0.9)
    near = Detection(track_id=1, bbox=(0, 0, 300, 200), class_id=2, confidence=0.9)

    for vehicle, expected in ((distant, True), (near, False)):
        x1, y1, x2, y2 = vehicle.bbox
        later = Detection(
            track_id=1, bbox=(x1 + nudge, y1, x2 + nudge, y2), class_id=2, confidence=0.9
        )
        registry = TrackRegistry()
        registry.update(0, frame, [vehicle])

        assert registry.update(1, frame, [later])[1].moved is expected


def test_a_failed_reading_still_counts_as_an_attempt():
    registry = TrackRegistry()
    registry.update(0, solid_frame(RED), [detection()])
    registry.record_plate(1, 0, None)

    states = registry.update(1, solid_frame(RED), [detection()])
    assert states[1].plate_attempts == 1
    assert states[1].plate is None


def test_plate_is_unknown_before_any_reading():
    registry = TrackRegistry()
    states = registry.update(0, solid_frame(RED), [detection()])

    assert states[1].plate is None


def test_a_confirmed_plate_is_reported_once_and_not_again():
    registry = TrackRegistry(min_plate_score=2.0)
    registry.update(0, solid_frame(RED), [detection()])
    for frame_index in range(3):
        registry.record_plate(1, frame_index, reading("CF5775"))
    states = registry.update(3, solid_frame(RED), [detection(shift=TRAVELLED)])

    assert [state.track_id for state, _ in newly_confirmed(states)] == [1]
    assert newly_confirmed(states) == []


def test_one_crisp_reading_in_the_local_format_is_still_only_one_frame():
    # 1.5 for the format times a confidence of exactly 1.0 used to clear the score on its own
    registry = TrackRegistry()
    registry.update(0, solid_frame(RED), [detection()])
    registry.record_plate(1, 0, reading("AA1234BB", confidence=1.0))

    states = registry.update(1, solid_frame(RED), [detection(shift=TRAVELLED)])
    assert states[1].plate_score >= MIN_REPORTED_PLATE_SCORE
    assert states[1].plate is None

    registry.record_plate(1, 1, reading("AA1234BB", confidence=1.0))
    states = registry.update(2, solid_frame(RED), [detection(shift=TRAVELLED)])
    assert states[1].plate == "AA1234BB"


def test_a_winner_that_a_rival_is_shadowing_is_not_reported():
    # the rival readings of a plate differ from the winner by one character, so a pair of
    # agreeing frames means nothing while another text is holding a pair of its own
    registry = TrackRegistry(min_plate_score=1.5, min_plate_lead=0.9)
    registry.update(0, solid_frame(RED), [detection()])
    for frame_index, text in enumerate(["CF5775", "CF5775", "CF5715", "CF5715"]):
        registry.record_plate(1, frame_index, reading(text))

    states = registry.update(1, solid_frame(RED), [detection(shift=TRAVELLED)])
    assert states[1].plate_score >= 1.5
    assert states[1].plate is None


def test_a_winner_that_pulls_clear_is_reported():
    registry = TrackRegistry(min_plate_score=1.5, min_plate_lead=0.9)
    registry.update(0, solid_frame(RED), [detection()])
    for frame_index, text in enumerate(["CF5775", "CF5775", "CF5715", "CF5715", "CF5775"]):
        registry.record_plate(1, frame_index, reading(text))

    states = registry.update(1, solid_frame(RED), [detection(shift=TRAVELLED)])
    assert states[1].plate == "CF5775"


def test_two_readings_tied_at_the_top_report_nothing():
    registry = TrackRegistry(min_plate_score=0.0, min_plate_lead=0.9)
    registry.update(0, solid_frame(RED), [detection()])
    registry.record_plate(1, 0, reading("CF5775"))
    registry.record_plate(1, 1, reading("CF5715"))

    states = registry.update(1, solid_frame(RED), [detection(shift=TRAVELLED)])
    assert states[1].plate is None


def test_a_plate_beaten_by_a_closer_reading_is_announced_as_a_correction():
    registry = TrackRegistry(min_plate_score=1.5)
    registry.update(0, solid_frame(RED), [detection()])
    for frame_index in range(2):
        registry.record_plate(1, frame_index, reading("CF5795"))
    states = registry.update(1, solid_frame(RED), [detection(shift=TRAVELLED)])
    assert [state.plate for state, _ in newly_confirmed(states)] == ["CF5795"]

    for frame_index in range(2, 6):
        registry.record_plate(1, frame_index, reading("CF5775"))

    states = registry.update(2, solid_frame(RED), [detection(shift=TRAVELLED)])
    assert [(state.plate, previous) for state, previous in newly_confirmed(states)] == [
        ("CF5775", "CF5795")
    ]


def test_a_plate_that_keeps_winning_is_not_announced_again():
    registry = TrackRegistry(min_plate_score=1.5)
    registry.update(0, solid_frame(RED), [detection()])
    for frame_index in range(2):
        registry.record_plate(1, frame_index, reading("CF5775"))
    states = registry.update(1, solid_frame(RED), [detection(shift=TRAVELLED)])
    newly_confirmed(states)

    registry.record_plate(1, 3, reading("CF5775"))
    states = registry.update(2, solid_frame(RED), [detection(shift=TRAVELLED)])
    assert newly_confirmed(states) == []


def test_a_frame_with_a_known_plate_passes_the_stream_filter():
    registry = TrackRegistry(min_plate_score=2.0)
    registry.update(0, solid_frame(RED), [detection()])
    for frame_index in range(3):
        registry.record_plate(1, frame_index, reading("CF5775"))
    states = registry.update(3, solid_frame(RED), [detection(shift=TRAVELLED)])

    assert carries_a_plate(states)


def test_a_frame_with_vehicles_but_no_plate_does_not_pass_the_stream_filter():
    registry = TrackRegistry(min_plate_score=2.0)
    states = registry.update(0, solid_frame(RED), [detection()])

    assert not carries_a_plate(states)


def test_an_empty_frame_does_not_pass_the_stream_filter():
    assert not carries_a_plate({})


def test_a_vehicle_without_a_settled_plate_is_not_reported():
    registry = TrackRegistry(min_plate_score=2.0)
    registry.update(0, solid_frame(RED), [detection()])
    registry.record_plate(1, 0, reading("CF5775"))
    states = registry.update(1, solid_frame(RED), [detection(shift=TRAVELLED)])

    assert newly_confirmed(states) == []


def test_vehicles_that_left_the_scene_are_still_counted():
    registry = TrackRegistry(expiry_frames=10)
    registry.update(0, solid_frame(RED), [detection(track_id=1)])
    registry.update(50, solid_frame(RED), [detection(track_id=2)])

    assert len(registry) == 1
    assert registry.total_tracks == 2


def test_a_vehicle_is_counted_once_however_long_it_stays():
    registry = TrackRegistry()
    for frame_index in range(10):
        registry.update(frame_index, solid_frame(RED), [detection()])

    assert registry.total_tracks == 1


def test_recording_a_plate_for_an_unknown_track_is_ignored():
    registry = TrackRegistry()
    registry.record_plate(99, 0, reading("AA1234BB"))

    assert len(registry) == 0
