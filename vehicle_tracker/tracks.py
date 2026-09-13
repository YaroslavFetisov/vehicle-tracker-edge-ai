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

# Plate reading is the one stage that can outgrow a frame's time budget, so a burst of
# vehicles arriving together is served over several frames instead of all at once.
MAX_PLATE_READS_PER_FRAME = 2

# A single reading of a distant or motion blurred plate is close to a random guess, so a
# plate is only reported once several frames agree on it, and retried until they strongly do.
MIN_REPORTED_PLATE_SCORE = 2.0
PLATE_CONFIDENT_SCORE = 5.0


@dataclass
class TrackState:
    track_id: int
    last_seen: int
    min_plate_score: float
    last_plate_frame: int
    plate_budget_height: int
    color_votes: Counter[VehicleColor] = field(default_factory=Counter)
    last_color_frame: int | None = None
    plate_votes: dict[str, float] = field(default_factory=dict)
    plate_attempts: int = 0
    # so that a confirmed plate is reported once and not on every frame the vehicle stays in
    announced: bool = False

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
        if not self.plate_votes:
            return None
        best = max(self.plate_votes, key=lambda text: self.plate_votes[text])
        if self.plate_votes[best] < self.min_plate_score:
            return None
        return best

    @property
    def plate_score(self) -> float:
        if not self.plate_votes:
            return 0.0
        return max(self.plate_votes.values())


def carries_a_plate(states: dict[int, TrackState]) -> bool:
    """Whether this frame shows a vehicle whose plate is known."""
    return any(state.plate is not None for state in states.values())


def newly_confirmed(states: dict[int, TrackState]) -> list[TrackState]:
    """Vehicles whose plate has just been confirmed, each one returned only once."""
    confirmed = []
    for state in states.values():
        if state.plate is not None and not state.announced:
            state.announced = True
            confirmed.append(state)
    return confirmed


class TrackRegistry:
    """Accumulates what is known about each tracked vehicle.

    Colour is a property of the vehicle, not of a single frame, so it is sampled on a
    few frames per track and decided by majority vote. A blurred or shadowed frame then
    costs one vote instead of changing the answer.
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
        retry_growth: float = PLATE_RETRY_GROWTH,
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
        self._retry_growth = retry_growth
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
                    last_plate_frame=self._staggered_start(detection.track_id, frame_index),
                    plate_budget_height=y2 - y1,
                )
                self._states[detection.track_id] = state
                self._total_tracks += 1
            state.last_seen = frame_index
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
