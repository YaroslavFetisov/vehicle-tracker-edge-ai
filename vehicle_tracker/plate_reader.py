from __future__ import annotations

import logging

import cv2
import numpy as np
from fast_plate_ocr import LicensePlateRecognizer
from open_image_models import create_detector

from vehicle_tracker.plate import PlateReading, normalize

logger = logging.getLogger(__name__)

# Measured on the sample footage: 384 detects plates on 92% of vehicle crops against 84%
# at 256 and 72% at 640. A larger input is worse, not better, because upscaling a small
# crop adds no detail while moving it away from the scale the model was trained on.
DETECTION_MODEL = "yolo-v9-t-384-license-plate-end2end"

# cct-xs matches the accuracy of the larger cct-s on this footage at a sixth of the cost.
OCR_MODEL = "cct-xs-v2-global-model"

# Below this width the recognizer returns confident nonsense, so size is the only
# usable filter: at 40 px and above characters are 82% correct, below 40 px almost none.
MIN_PLATE_WIDTH = 40
MIN_PLATE_HEIGHT = 10


class PlateReader:
    def __init__(self, *, device: str = "auto", min_plate_width: int = MIN_PLATE_WIDTH) -> None:
        self._detector = create_detector(DETECTION_MODEL)
        self._recognizer = LicensePlateRecognizer(hub_ocr_model=OCR_MODEL, device=device)
        self._expects_grayscale = self._recognizer.config.image_color_mode == "grayscale"
        self._min_plate_width = min_plate_width
        logger.info("plate reader: %s + %s", DETECTION_MODEL, OCR_MODEL)

    def read(self, vehicle_crop: np.ndarray) -> PlateReading | None:
        plate_crop, confidence = self._locate(vehicle_crop)
        if plate_crop is None:
            return None

        prediction = self._recognizer.run(self._to_model_input(plate_crop), return_confidence=True)[
            0
        ]
        text = normalize(prediction.plate)
        if text is None:
            return None

        probabilities = prediction.char_probs
        return PlateReading(
            text=text,
            detection_confidence=confidence,
            ocr_confidence=float(np.mean(probabilities)) if probabilities is not None else 0.0,
        )

    def _locate(self, vehicle_crop: np.ndarray) -> tuple[np.ndarray | None, float]:
        if vehicle_crop.size == 0:
            return None, 0.0

        plates = self._detector.predict(vehicle_crop)
        if not plates:
            return None, 0.0

        best = max(plates, key=lambda plate: plate.confidence)
        box = best.bounding_box
        crop = vehicle_crop[max(box.y1, 0) : box.y2, max(box.x1, 0) : box.x2]
        if (
            crop.size == 0
            or crop.shape[1] < self._min_plate_width
            or crop.shape[0] < MIN_PLATE_HEIGHT
        ):
            return None, 0.0
        return crop, float(best.confidence)

    def _to_model_input(self, plate_crop: np.ndarray) -> np.ndarray:
        if self._expects_grayscale:
            return cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)
