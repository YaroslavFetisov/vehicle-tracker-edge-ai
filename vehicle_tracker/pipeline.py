from __future__ import annotations

import logging

import cv2

from vehicle_tracker.config import Config
from vehicle_tracker.detector import VehicleDetector
from vehicle_tracker.overlay import draw_detections
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

    registry = TrackRegistry()

    with VideoSource(config.source) as source:
        for frame_index, frame in enumerate(source):
            detections = detector.track(frame)
            states = registry.update(frame_index, frame, detections)
            draw_detections(frame, detections, states)
            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF in QUIT_KEYS:
                logger.info("stopped by user")
                break

    cv2.destroyAllWindows()
