from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from vehicle_tracker.color import VehicleColor, estimate_color
from vehicle_tracker.detection import Detection
from vehicle_tracker.plate import PlateReading, vote_weight

COLOR_SAMPLE_INTERVAL = 5
MAX_COLOR_SAMPLES = 15
TRACK_EXPIRY_FRAMES = 90

# Plates get easier to read as a vehicle approaches, so the fixed OCR budget per vehicle is
# spread across the whole track rather than spent as fast as possible. Sampling every second
# frame instead of every fifth exhausts the budget while the vehicle is still far away and
# loses plates that a slower schedule reads correctly.
PLATE_SAMPLE_INTERVAL = 5
MAX_PLATE_ATTEMPTS = 40

# A vehicle that has doubled in height gets a fresh allowance, because its plate has roughly
# doubled in width too, and that is exactly the span over which the recognizer becomes usable:
# characters are 13% correct between 20 and 39 pixels of plate width and 82% correct above it.
# Without this a long lived track spends its whole budget while still far away - on the sample
# clip a bus is in view for 377 frames and only becomes readable in the last 40 of them. A
# smaller factor renews too early and wastes the new budget on frames that are still hopeless;
# a larger one never triggers. A vehicle that does not approach never renews, so dense traffic
# pays nothing for this.
PLATE_RETRY_GROWTH = 2.0

# Expressed against the frame rather than in pixels: an absolute threshold tuned at 1080p
# rejects every vehicle on a 480p camera, which is exactly the kind of source this runs on.
MIN_VEHICLE_HEIGHT_FRACTION = 0.08

# A vehicle travels through the scene and a road sign does not. The detector boxes a matrix
# sign on the sample footage as a bus, and the identification number printed on its frame is
# then read as a plate - crisply and identically on every frame, because the sign never moves,
# so it outscores every real plate in the clip. Displacement is measured from where the track
# was first seen rather than summed frame by frame, so box jitter cannot accumulate into
# movement, and it is expressed against the vehicle's own height so that the same rule holds
# at any distance. Measured on both clips: the false positives reach 0.03 of their height and
# every vehicle that lives long enough to collect a plate reaches at least 0.27.
MIN_DRIFT_FRACTION = 0.10

# Plate reading is the one stage that can outgrow a frame's time budget, so a burst of
# vehicles arriving together is served over several frames instead of all at once.
MAX_PLATE_READS_PER_FRAME = 2

# A single reading of a distant or motion blurred plate is close to a random guess, so a
# plate is only reported once several frames agree on it, and retried until they strongly do.
# Each vote is weighted by the recognizer's own confidence, which runs at 0.8 to 1.0 on a
# readable plate, so this has to sit below twice that or two agreeing frames never clear it:
# at 2.0 the one legible plate of the highway clip scored 1.87 and was never reported.
MIN_REPORTED_PLATE_SCORE = 1.5

# Two frames agreeing is not enough on its own, because the rival readings of a plate differ
# from the winner by a single character and collect their own pairs. Measured on the dashcam
# clip, where the followed car holds its plate at 40 pixels for six seconds: three wrong texts
# reach a pair before the right one does, and each leads its nearest rival by at most 0.66,
# while every text that turns out to be correct pulls at least 0.92 clear. A reading of a
# legible plate scores 0.9 to 1.0, so this asks the winner to be one whole reading ahead.
PLATE_LEAD = 0.9

# Stated outright rather than implied by the score: a local-format reading weighs 1.5 and the
# recognizer does return a confidence of exactly 1.0, so one frame alone could clear the bar.
MIN_AGREEING_READINGS = 2

PLATE_CONFIDENT_SCORE = 5.0


def center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


@dataclass
class TrackState:
    track_id: int
    last_seen: int
    min_plate_score: float
    min_plate_lead: float
    min_agreeing_readings: int
    last_plate_frame: int
    plate_budget_height: int
    first_center: tuple[float, float]
    color_votes: Counter[VehicleColor] = field(default_factory=Counter)
    last_color_frame: int | None = None
    plate_votes: dict[str, float] = field(default_factory=dict)
    plate_hits: Counter[str] = field(default_factory=Counter)
    plate_attempts: int = 0
    # the text this vehicle was last reported under, so that a plate is announced once and
    # not on every frame the vehicle stays in view
    announced_plate: str | None = None
    # latched rather than recomputed per frame: a vehicle that stops at a barrier has still
    # arrived under its own power, and its plate stays reportable while it waits
    moved: bool = False

    @property
    def color(self) -> VehicleColor | None:
        if not self.color_votes:
            return None
        return self.color_votes.most_common(1)[0][0]

    @property
    def color_samples(self) -> int:
        return sum(self.color_votes.values())

    @property
    def plate(self) -> str | None:
        # whatever text a box that has never travelled carries, it is not a vehicle's plate
        if not self.moved or not self.plate_votes:
            return None
        best = max(self.plate_votes, key=lambda text: self.plate_votes[text])
        if self.plate_hits[best] < self.min_agreeing_readings:
            return None
        if self.plate_votes[best] < self.min_plate_score:
            return None
        if self.plate_votes[best] - self.runner_up_score < self.min_plate_lead:
            return None
        return best

    @property
    def runner_up_score(self) -> float:
        """The best score among the readings that are not currently winning."""
        ranked = sorted(self.plate_votes.values(), reverse=True)
        return ranked[1] if len(ranked) > 1 else 0.0

    @property
    def plate_score(self) -> float:
        if not self.plate_votes:
            return 0.0
        return max(self.plate_votes.values())


def carries_a_plate(states: dict[int, TrackState]) -> bool:
    """Whether this frame shows a vehicle whose plate is known."""
    return any(state.plate is not None for state in states.values())


def newly_confirmed(states: dict[int, TrackState]) -> list[tuple[TrackState, str | None]]:
    """Vehicles whose plate has just been confirmed, or has changed since it was announced.

    The vote keeps improving while a vehicle approaches, so the text that first clears the
    threshold can be beaten later by a reading taken from closer up. Each vehicle is returned
    with the text it was last announced under, so a correction can be reported as a correction
    rather than as a second vehicle.
    """
    announcements = []
    for state in states.values():
        plate = state.plate
        if plate is None or plate == state.announced_plate:
            continue
        announcements.append((state, state.announced_plate))
        state.announced_plate = plate
    return announcements


class TrackRegistry:
    """Accumulates what is known about each tracked vehicle.

    Colour and the plate are properties of the vehicle, not of a single frame, so both are
    sampled on a few frames per track and decided by majority vote. A blurred or shadowed
    frame then costs one vote instead of changing the answer. The registry also keeps track
    of whether a box has ever travelled, because a plate is only credible on something that
    moves through the scene.
    """

    def __init__(
        self,
        *,
        color_interval: int = COLOR_SAMPLE_INTERVAL,
        max_color_samples: int = MAX_COLOR_SAMPLES,
        expiry_frames: int = TRACK_EXPIRY_FRAMES,
        plate_interval: int = PLATE_SAMPLE_INTERVAL,
        max_plate_attempts: int = MAX_PLATE_ATTEMPTS,
        min_vehicle_height_fraction: float = MIN_VEHICLE_HEIGHT_FRACTION,
        max_plate_reads_per_frame: int = MAX_PLATE_READS_PER_FRAME,
        confident_plate_score: float = PLATE_CONFIDENT_SCORE,
        min_plate_score: float = MIN_REPORTED_PLATE_SCORE,
        min_plate_lead: float = PLATE_LEAD,
        min_agreeing_readings: int = MIN_AGREEING_READINGS,
        retry_growth: float = PLATE_RETRY_GROWTH,
        min_drift_fraction: float = MIN_DRIFT_FRACTION,
    ) -> None:
        self._states: dict[int, TrackState] = {}
        self._color_interval = color_interval
        self._max_color_samples = max_color_samples
        self._expiry_frames = expiry_frames
        self._plate_interval = plate_interval
        self._max_plate_attempts = max_plate_attempts
        self._min_vehicle_height_fraction = min_vehicle_height_fraction
        self._max_plate_reads_per_frame = max_plate_reads_per_frame
        self._confident_plate_score = confident_plate_score
        self._min_plate_score = min_plate_score
        self._min_plate_lead = min_plate_lead
        self._min_agreeing_readings = min_agreeing_readings
        self._retry_growth = retry_growth
        self._min_drift_fraction = min_drift_fraction
        self._frame_height = 0
        self._total_tracks = 0

    def __len__(self) -> int:
        return len(self._states)

    @property
    def total_tracks(self) -> int:
        """Vehicles seen since the start, including those that have left the scene."""
        return self._total_tracks

    def update(
        self,
        frame_index: int,
        frame: np.ndarray,
        detections: list[Detection],
    ) -> dict[int, TrackState]:
        self._frame_height = frame.shape[0]
        for detection in detections:
            _, y1, _, y2 = detection.bbox
            state = self._states.get(detection.track_id)
            if state is None:
                state = TrackState(
                    track_id=detection.track_id,
                    last_seen=frame_index,
                    min_plate_score=self._min_plate_score,
                    min_plate_lead=self._min_plate_lead,
                    min_agreeing_readings=self._min_agreeing_readings,
                    last_plate_frame=self._staggered_start(detection.track_id, frame_index),
                    plate_budget_height=y2 - y1,
                    first_center=center(detection.bbox),
                )
                self._states[detection.track_id] = state
                self._total_tracks += 1
            state.last_seen = frame_index
            self._note_movement(state, detection.bbox)
            self._renew_plate_budget(state, y2 - y1)

            if self._needs_color(state, frame_index):
                color = estimate_color(frame, detection.bbox)
                if color is not None:
                    state.color_votes[color] += 1
                    state.last_color_frame = frame_index

        self._expire(frame_index)
        return {detection.track_id: self._states[detection.track_id] for detection in detections}

    def due_for_plate(self, frame_index: int, detections: list[Detection]) -> list[Detection]:
        """Vehicles worth spending a plate detection and an OCR pass on this frame."""
        due = [
            detection
            for detection in detections
            if detection.track_id in self._states
            and self._needs_plate(self._states[detection.track_id], detection, frame_index)
        ]
        if len(due) <= self._max_plate_reads_per_frame:
            return due

        due.sort(key=lambda detection: self._states[detection.track_id].last_plate_frame)
        return due[: self._max_plate_reads_per_frame]

    def _note_movement(self, state: TrackState, bbox: tuple[int, int, int, int]) -> None:
        if state.moved:
            return
        x, y = center(bbox)
        origin_x, origin_y = state.first_center
        drift = max(abs(x - origin_x), abs(y - origin_y))
        _, y1, _, y2 = bbox
        state.moved = drift >= (y2 - y1) * self._min_drift_fraction

    def _renew_plate_budget(self, state: TrackState, height: int) -> None:
        if state.plate_attempts < self._max_plate_attempts:
            return
        if height < state.plate_budget_height * self._retry_growth:
            return
        state.plate_attempts = 0
        state.plate_budget_height = height

    def _staggered_start(self, track_id: int, frame_index: int) -> int:
        # vehicles entering the scene together would otherwise queue their first pass on
        # the same frame and then stay in lockstep for as long as they are tracked
        return frame_index + track_id % self._plate_interval - self._plate_interval

    def record_plate(self, track_id: int, frame_index: int, reading: PlateReading | None) -> None:
        state = self._states.get(track_id)
        if state is None:
            return

        state.plate_attempts += 1
        state.last_plate_frame = frame_index
        if reading is not None:
            weight = vote_weight(reading.text) * reading.ocr_confidence
            state.plate_votes[reading.text] = state.plate_votes.get(reading.text, 0.0) + weight
            state.plate_hits[reading.text] += 1

    def _needs_plate(self, state: TrackState, detection: Detection, frame_index: int) -> bool:
        if state.plate_attempts >= self._max_plate_attempts:
            return False
        if state.plate_score >= self._confident_plate_score:
            return False

        _, y1, _, y2 = detection.bbox
        if y2 - y1 < self._frame_height * self._min_vehicle_height_fraction:
            return False

        return frame_index - state.last_plate_frame >= self._plate_interval

    def _needs_color(self, state: TrackState, frame_index: int) -> bool:
        if state.color_samples >= self._max_color_samples:
            return False
        if state.last_color_frame is None:
            return True
        return frame_index - state.last_color_frame >= self._color_interval

    def _expire(self, frame_index: int) -> None:
        # a stream can run for days, so vehicles that left the scene must not pile up
        stale = [
            track_id
            for track_id, state in self._states.items()
            if frame_index - state.last_seen > self._expiry_frames
        ]
        for track_id in stale:
            del self._states[track_id]
