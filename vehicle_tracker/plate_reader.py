from __future__ import annotations

import logging
import os
from typing import Any

import cv2
import numpy as np
import onnxruntime
from fast_plate_ocr import LicensePlateRecognizer
from open_image_models import create_detector

from vehicle_tracker.plate import PlateReading, normalize
from vehicle_tracker.runtime import CUDA_PROVIDER, plate_model_threads, providers_for

logger = logging.getLogger(__name__)

# input sizes and model variants were picked on the sample footage, see README
DETECTION_MODEL = "yolo-v9-t-384-license-plate-end2end"
OCR_MODEL = "cct-xs-v2-global-model"

# below this the recognizer returns confident nonsense
MIN_PLATE_WIDTH = 40
MIN_PLATE_HEIGHT = 10

# hides the shape inference warnings both models print on every start
ONNX_ERROR_SEVERITY = 3

NEUTRAL_OCR_CONFIDENCE = 1.0

ALLOW_SPINNING = "session.intra_op.allow_spinning"


def session_options() -> onnxruntime.SessionOptions:
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = plate_model_threads(os.cpu_count())
    # idle threads busy-wait by default and take cores away from the detector
    options.add_session_config_entry(ALLOW_SPINNING, "0")
    return options


class PlateReader:
    def __init__(self, *, device: str = "auto") -> None:
        onnxruntime.set_default_logger_severity(ONNX_ERROR_SEVERITY)
        available = onnxruntime.get_available_providers()
        if device == "cuda" and CUDA_PROVIDER not in available:
            logger.warning("no CUDA execution provider available, plate models run on the CPU")
        providers = providers_for(device, available)
        options = session_options()
        self._detector = create_detector(DETECTION_MODEL, providers=providers, sess_options=options)
        self._recognizer = LicensePlateRecognizer(
            hub_ocr_model=OCR_MODEL, device=device, providers=providers, sess_options=options
        )
        self._expects_grayscale = self._recognizer.config.image_color_mode == "grayscale"
        logger.info(
            "plate reader: %s + %s on %s, %d threads each",
            DETECTION_MODEL,
            OCR_MODEL,
            device,
            options.intra_op_num_threads,
        )

    def read(self, vehicle_crop: np.ndarray) -> PlateReading | None:
        plate_crop = self._locate(vehicle_crop)
        if plate_crop is None:
            return None

        prediction = self._recognizer.run(self._to_model_input(plate_crop), return_confidence=True)[
            0
        ]
        text = normalize(prediction.plate)
        if text is None:
            return None
        return PlateReading(text=text, ocr_confidence=confidence_of(prediction, len(text)))

    def _locate(self, vehicle_crop: np.ndarray) -> np.ndarray | None:
        if vehicle_crop.size == 0:
            return None

        plates = self._detector.predict(vehicle_crop)
        if not plates:
            return None

        box = max(plates, key=lambda plate: plate.confidence).bounding_box
        crop = vehicle_crop[max(box.y1, 0) : box.y2, max(box.x1, 0) : box.x2]
        if crop.size == 0 or crop.shape[1] < MIN_PLATE_WIDTH or crop.shape[0] < MIN_PLATE_HEIGHT:
            return None
        return crop

    def _to_model_input(self, plate_crop: np.ndarray) -> np.ndarray:
        if self._expects_grayscale:
            return cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)


def confidence_of(prediction: Any, length: int) -> float:
    probabilities = prediction.char_probs
    if probabilities is None:
        return NEUTRAL_OCR_CONFIDENCE
    # unused trailing slots score 1.0 and would inflate short readings
    return float(np.mean(probabilities[:length]))
