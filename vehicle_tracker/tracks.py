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

# plates get easier to read as a vehicle approaches, so the budget is spread over the track
PLATE_SAMPLE_INTERVAL = 5
MAX_PLATE_ATTEMPTS = 40

# twice the height means roughly twice the plate width, enough to go from unreadable to readable
PLATE_RETRY_GROWTH = 2.0

# relative to the frame, so the same value works for a 480p and a 1080p camera
MIN_VEHICLE_HEIGHT_FRACTION = 0.08

# a road sign detected as a bus carries crisp text too, but it never moves
MIN_DRIFT_FRACTION = 0.10

MAX_PLATE_READS_PER_FRAME = 2

# votes are weighted by OCR confidence, a plate needs agreeing frames and a clear lead over
# rival readings, which usually differ from it by a single character
MIN_REPORTED_PLATE_SCORE = 1.5
PLATE_LEAD = 0.9
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
    announced_plate: str | None = None
    # latched, so a car waiting at a barrier keeps its plate
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
    """Vehicles whose plate was just confirmed or changed, with the text announced before."""
    announcements = []
    for state in states.values():
        plate = state.plate
        if plate is None or plate == state.announced_plate:
            continue
        announcements.append((state, state.announced_plate))
        state.announced_plate = plate
    return announcements


class TrackRegistry:
    """Colour and plate belong to the vehicle, not the frame, so both are voted on per track."""

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
        # otherwise vehicles that appear together are read on the same frames for good
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
        # a high score alone is not settled while a one character rival is close behind it
        if (
            state.plate_score >= self._confident_plate_score
            and state.plate_score - state.runner_up_score >= self._min_plate_lead
        ):
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
        stale = [
            track_id
            for track_id, state in self._states.items()
            if frame_index - state.last_seen > self._expiry_frames
        ]
        for track_id in stale:
            del self._states[track_id]
