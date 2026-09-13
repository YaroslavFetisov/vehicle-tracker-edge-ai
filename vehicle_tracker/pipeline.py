from __future__ import annotations

import logging
import time

import cv2
import numpy as np

from vehicle_tracker.config import Config
from vehicle_tracker.detection import Detection, crop_to_bbox
from vehicle_tracker.detector import VehicleDetector
from vehicle_tracker.metrics import Metrics
from vehicle_tracker.overlay import draw_detections, draw_status
from vehicle_tracker.plate_reader import PlateReader
from vehicle_tracker.tracks import TrackRegistry, TrackState
from vehicle_tracker.video_source import VideoSource

logger = logging.getLogger(__name__)

WINDOW_NAME = "vehicle tracker"
QUIT_KEYS = (ord("q"), 27)


class Pipeline:
    """Runs every stage on one frame and annotates it in place.

    Kept separate from the display loop so that the benchmark measures the same code that
    the application runs, rather than a copy of it that can drift.
    """

    def __init__(
        self,
        detector: VehicleDetector,
        plate_reader: PlateReader,
        registry: TrackRegistry,
        metrics: Metrics,
    ) -> None:
        self._detector = detector
        self._plate_reader = plate_reader
        self._registry = registry
        self._metrics = metrics

    def process(self, frame_index: int, frame: np.ndarray) -> dict[int, TrackState]:
        """Analyse one frame and return the state of every vehicle visible on it."""
        with self._metrics.stage("detect"):
            detections = self._detector.track(frame)

        # nothing found means nothing to analyse: the remaining stages are the expensive
        # ones, and a frame without vehicles never reaches them
        if not detections:
            return {}

        with self._metrics.stage("color"):
            states = self._registry.update(frame_index, frame, detections)
        with self._metrics.stage("plate"):
            self._read_plates(frame, frame_index, detections)
        with self._metrics.stage("draw"):
            draw_detections(frame, detections, states)
        return states

    def _read_plates(
        self, frame: np.ndarray, frame_index: int, detections: list[Detection]
    ) -> None:
        for detection in self._registry.due_for_plate(frame_index, detections):
            crop = crop_to_bbox(frame, detection.bbox)
            if crop is None:
                self._registry.record_plate(detection.track_id, frame_index, None)
                continue
            with self._metrics.stage("plate read", nested=True):
                reading = self._plate_reader.read(crop)
            self._registry.record_plate(detection.track_id, frame_index, reading)


def build_pipeline(config: Config, *, registry: TrackRegistry, metrics: Metrics) -> Pipeline:
    detector = VehicleDetector(
        config.weights,
        device=config.device,
        confidence=config.confidence,
        image_size=config.image_size,
    )
    return Pipeline(detector, PlateReader(device=config.device), registry, metrics)


def run(config: Config) -> None:
    metrics = Metrics()
    pipeline = build_pipeline(config, registry=TrackRegistry(), metrics=metrics)

    with VideoSource(config.source) as source:
        for frame_index, frame in enumerate(source):
            frame_started = time.perf_counter()
            states = pipeline.process(frame_index, frame)

            draw_status(frame, metrics.status_line())
            cv2.imshow(WINDOW_NAME, frame)
            metrics.frame_done(time.perf_counter() - frame_started, had_vehicles=bool(states))

            if cv2.waitKey(1) & 0xFF in QUIT_KEYS:
                logger.info("stopped by user")
                break

    cv2.destroyAllWindows()
    for line in metrics.summary():
        logger.info("%s", line)
