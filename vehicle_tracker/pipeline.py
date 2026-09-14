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
from vehicle_tracker.tracks import TrackRegistry, TrackState, carries_a_plate, newly_confirmed
from vehicle_tracker.video_source import VideoSource
from vehicle_tracker.video_writer import VideoWriter

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
    registry = TrackRegistry()
    pipeline = build_pipeline(config, registry=registry, metrics=metrics)
    writer = None
    kept = 0

    try:
        with VideoSource(config.source) as source:
            if config.output is not None:
                writer = VideoWriter(config.output, source.fps)
            for frame_index, frame in enumerate(source):
                frame_started = time.perf_counter()
                states = pipeline.process(frame_index, frame)
                report(states)

                # the optional stream filter: a frame that carries no vehicle with a
                # readable plate is analysed but never shown or stored
                if not config.filter_stream or carries_a_plate(states):
                    kept += 1
                    draw_status(frame, metrics.status_line())
                    if writer is not None:
                        writer.write(frame)
                    if config.display:
                        cv2.imshow(WINDOW_NAME, frame)
                metrics.frame_done(time.perf_counter() - frame_started, had_vehicles=bool(states))

                if config.display and cv2.waitKey(1) & 0xFF in QUIT_KEYS:
                    logger.info("stopped by user")
                    break
    except KeyboardInterrupt:
        # the usual way to stop a container, and the output file still has to be closed
        logger.info("interrupted")
    finally:
        # also runs when the stream is lost for good, so a failed run still reports what it
        # saw and leaves a playable file behind rather than only a stack trace
        if writer is not None:
            writer.close()
        if config.display:
            cv2.destroyAllWindows()
        logger.info("tracks seen: %d", registry.total_tracks)
        if config.filter_stream:
            logger.info("frames kept by the filter: %d of %d", kept, metrics.frames)
        for line in metrics.summary():
            logger.info("%s", line)


def report(states: dict[int, TrackState]) -> None:
    for state, previous in newly_confirmed(states):
        if previous is not None:
            logger.info(
                "vehicle %d: plate %s corrected to %s", state.track_id, previous, state.plate
            )
            continue
        color = state.color.name if state.color is not None else "unknown"
        logger.info("vehicle %d: %s, plate %s", state.track_id, color, state.plate)
