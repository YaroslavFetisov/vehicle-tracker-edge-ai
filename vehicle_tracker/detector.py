from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from vehicle_tracker.detection import Detection

logger = logging.getLogger(__name__)

# COCO classes: car, motorcycle, bus, truck
VEHICLE_CLASS_IDS = (2, 3, 5, 7)

TRACKER_CONFIG = "bytetrack.yaml"
FP16 = 16


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


class VehicleDetector:
    """Detects vehicles and assigns a stable track id to each of them."""

    def __init__(self, weights: Path, *, device: str, confidence: float, image_size: int) -> None:
        if not weights.is_file():
            raise FileNotFoundError(
                f"detection weights not found: {weights} - run scripts/fetch_assets.py"
            )
        self._device = resolve_device(device)
        self._confidence = confidence
        self._image_size = image_size
        self._model = YOLO(str(weights))
        logger.info("vehicle detector: %s on %s", weights.name, self._device)

    @property
    def device(self) -> str:
        return self._device

    def track(self, frame: np.ndarray) -> list[Detection]:
        results = self._model.track(
            frame,
            persist=True,
            tracker=TRACKER_CONFIG,
            classes=list(VEHICLE_CLASS_IDS),
            conf=self._confidence,
            imgsz=self._image_size,
            device=self._device,
            quantize=FP16 if self._device == "cuda" else None,
            verbose=False,
        )
        boxes = results[0].boxes
        if boxes is None or boxes.id is None:
            return []

        return [
            Detection(
                track_id=int(track_id),
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                class_id=int(class_id),
                confidence=float(confidence),
            )
            for (x1, y1, x2, y2), track_id, class_id, confidence in zip(
                boxes.xyxy.tolist(),
                boxes.id.tolist(),
                boxes.cls.tolist(),
                boxes.conf.tolist(),
                strict=True,
            )
        ]
