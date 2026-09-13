from __future__ import annotations

import logging
import threading
from collections import deque
from collections.abc import Iterator

import cv2
import numpy as np

from vehicle_tracker.config import STREAM_SCHEMES

logger = logging.getLogger(__name__)

INITIAL_RECONNECT_DELAY = 1.0
MAX_RECONNECT_DELAY = 8.0
MAX_RECONNECT_ATTEMPTS = 5
WAIT_TIMEOUT = 0.1


def is_stream(source: str | int) -> bool:
    if isinstance(source, int):
        return True
    return source.startswith(STREAM_SCHEMES)


class VideoSource:
    """Reads frames in a background thread.

    Live streams keep only the newest frame: a camera that outruns inference would
    otherwise build an ever growing backlog and the displayed frame would drift
    further into the past. File sources keep every frame instead, so repeated runs
    stay reproducible and benchmark numbers remain comparable.
    """

    def __init__(
        self,
        source: str | int,
        *,
        max_reconnect_attempts: int = MAX_RECONNECT_ATTEMPTS,
    ) -> None:
        self._source = source
        self._is_stream = is_stream(source)
        self._max_reconnect_attempts = max_reconnect_attempts
        self._buffer: deque[np.ndarray] = deque(maxlen=1)
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._failure: Exception | None = None
        self._fps = 0.0

    def __enter__(self) -> VideoSource:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def start(self) -> None:
        capture = self._open()
        self._thread = threading.Thread(target=self._read_loop, args=(capture,), daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    @property
    def fps(self) -> float:
        """Frame rate the source reports, or zero when it reports none or nonsense."""
        return self._fps if self._fps > 0 else 0.0

    def __iter__(self) -> Iterator[np.ndarray]:
        if self._thread is None:
            raise RuntimeError("video source was not started")

        while True:
            with self._condition:
                while not self._buffer and self._thread.is_alive():
                    self._condition.wait(WAIT_TIMEOUT)
                if not self._buffer:
                    break
                frame = self._buffer.popleft()
                self._condition.notify()
            yield frame

        if self._failure is not None:
            raise self._failure

    def _open(self) -> cv2.VideoCapture:
        capture = cv2.VideoCapture(self._source)
        if not capture.isOpened():
            raise RuntimeError(f"cannot open video source: {self._source}")
        # advisory only, most backends ignore it - hence the explicit dropping below
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._fps = capture.get(cv2.CAP_PROP_FPS)
        return capture

    def _read_loop(self, capture: cv2.VideoCapture) -> None:
        try:
            while not self._stop.is_set():
                ok, frame = capture.read()
                if ok:
                    self._publish(frame)
                    continue
                if not self._is_stream:
                    break
                capture.release()
                reconnected = self._reconnect()
                if reconnected is None:
                    break
                capture = reconnected
        finally:
            capture.release()
            with self._condition:
                self._condition.notify_all()

    def _reconnect(self) -> cv2.VideoCapture | None:
        for attempt in range(1, self._max_reconnect_attempts + 1):
            delay = min(INITIAL_RECONNECT_DELAY * 2 ** (attempt - 1), MAX_RECONNECT_DELAY)
            logger.warning(
                "stream read failed, reconnecting in %.0fs (attempt %d/%d)",
                delay,
                attempt,
                self._max_reconnect_attempts,
            )
            if self._stop.wait(delay):
                return None
            try:
                return self._open()
            except RuntimeError as exc:
                logger.warning("reconnect failed: %s", exc)

        self._failure = RuntimeError(
            f"video stream lost after {self._max_reconnect_attempts} reconnect attempts"
        )
        return None

    def _publish(self, frame: np.ndarray) -> None:
        with self._condition:
            if not self._is_stream:
                while self._buffer and not self._stop.is_set():
                    self._condition.wait(WAIT_TIMEOUT)
            self._buffer.append(frame)
            self._condition.notify()
