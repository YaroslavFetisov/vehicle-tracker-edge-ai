from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from vehicle_tracker.color import VehicleColor, estimate_color
from vehicle_tracker.detection import Detection

COLOR_SAMPLE_INTERVAL = 5
MAX_COLOR_SAMPLES = 15
TRACK_EXPIRY_FRAMES = 90


@dataclass
class TrackState:
    track_id: int
    last_seen: int
    color_votes: Counter[VehicleColor] = field(default_factory=Counter)
    last_color_frame: int | None = None

    @property
    def color(self) -> VehicleColor | None:
        if not self.color_votes:
            return None
        return self.color_votes.most_common(1)[0][0]

    @property
    def color_samples(self) -> int:
        return sum(self.color_votes.values())


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
    ) -> None:
        self._states: dict[int, TrackState] = {}
        self._color_interval = color_interval
        self._max_color_samples = max_color_samples
        self._expiry_frames = expiry_frames

    def __len__(self) -> int:
        return len(self._states)

    def update(
        self,
        frame_index: int,
        frame: np.ndarray,
        detections: list[Detection],
    ) -> dict[int, TrackState]:
        for detection in detections:
            state = self._states.get(detection.track_id)
            if state is None:
                state = TrackState(track_id=detection.track_id, last_seen=frame_index)
                self._states[detection.track_id] = state
            state.last_seen = frame_index

            if self._needs_color(state, frame_index):
                color = estimate_color(frame, detection.bbox)
                if color is not None:
                    state.color_votes[color] += 1
                    state.last_color_frame = frame_index

        self._expire(frame_index)
        return {detection.track_id: self._states[detection.track_id] for detection in detections}

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
