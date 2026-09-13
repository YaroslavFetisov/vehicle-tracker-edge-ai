from __future__ import annotations

import logging
import time

import cv2
import numpy as np

from vehicle_tracker.config import Config
from vehicle_tracker.detection import Detection
from vehicle_tracker.detector import VehicleDetector
from vehicle_tracker.metrics import Metrics
from vehicle_tracker.overlay import draw_detections, draw_status
from vehicle_tracker.plate_reader import PlateReader
from vehicle_tracker.tracks import TrackRegistry
from vehicle_tracker.video_source import VideoSource

logger = logging.getLogger(__name__)

WINDOW_NAME = "vehicle tracker"
QUIT_KEYS = (ord("q"), 27)


def run(config: Config) -> None:
    detector = VehicleDetector(
        config.weights,
        device=config.device,
        confidence=config.confidence,
        image_size=config.image_size,
    )

    plate_reader = PlateReader(device=config.device)
    registry = TrackRegistry()
    metrics = Metrics()

    with VideoSource(config.source) as source:
        for frame_index, frame in enumerate(source):
            frame_started = time.perf_counter()

            with metrics.stage("detect"):
                detections = detector.track(frame)

            # nothing found means nothing to analyse: the remaining stages are the
            # expensive ones, and a frame without vehicles never reaches them
            if detections:
                with metrics.stage("color"):
                    states = registry.update(frame_index, frame, detections)
                with metrics.stage("plate"):
                    read_plates(frame, frame_index, detections, registry, plate_reader, metrics)
                with metrics.stage("draw"):
                    draw_detections(frame, detections, states)

            draw_status(frame, metrics.status_line())
            cv2.imshow(WINDOW_NAME, frame)
            metrics.frame_done(time.perf_counter() - frame_started, had_vehicles=bool(detections))

            if cv2.waitKey(1) & 0xFF in QUIT_KEYS:
                logger.info("stopped by user")
                break

    cv2.destroyAllWindows()
    for line in metrics.summary():
        logger.info("%s", line)


def read_plates(
    frame: np.ndarray,
    frame_index: int,
    detections: list[Detection],
    registry: TrackRegistry,
    plate_reader: PlateReader,
    metrics: Metrics,
) -> None:
    for detection in registry.due_for_plate(frame_index, detections):
        crop = crop_vehicle(frame, detection.bbox)
        if crop is None:
            registry.record_plate(detection.track_id, frame_index, None)
            continue
        with metrics.stage("plate read", nested=True):
            reading = plate_reader.read(crop)
        registry.record_plate(detection.track_id, frame_index, reading)


def crop_vehicle(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray | None:
    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    # a tracker can predict a box that has left the frame, and a negative slice bound
    # would wrap around to the opposite edge instead of yielding nothing
    left, top = max(x1, 0), max(y1, 0)
    right, bottom = min(x2, frame_width), min(y2, frame_height)
    if right <= left or bottom <= top:
        return None
    return frame[top:bottom, left:right]
