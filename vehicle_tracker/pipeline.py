from __future__ import annotations

import logging

import cv2
import numpy as np

from vehicle_tracker.config import Config
from vehicle_tracker.detection import Detection
from vehicle_tracker.detector import VehicleDetector
from vehicle_tracker.overlay import draw_detections
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

    plate_reader = PlateReader()
    registry = TrackRegistry()

    with VideoSource(config.source) as source:
        for frame_index, frame in enumerate(source):
            detections = detector.track(frame)
            states = registry.update(frame_index, frame, detections)
            read_plates(frame, frame_index, detections, registry, plate_reader)
            draw_detections(frame, detections, states)
            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF in QUIT_KEYS:
                logger.info("stopped by user")
                break

    cv2.destroyAllWindows()


def read_plates(
    frame: np.ndarray,
    frame_index: int,
    detections: list[Detection],
    registry: TrackRegistry,
    plate_reader: PlateReader,
) -> None:
    for detection in registry.due_for_plate(frame_index, detections):
        x1, y1, x2, y2 = detection.bbox
        crop = frame[max(y1, 0) : y2, max(x1, 0) : x2]
        registry.record_plate(detection.track_id, frame_index, plate_reader.read(crop))
