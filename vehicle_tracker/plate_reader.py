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
from vehicle_tracker.runtime import plate_model_threads, providers_for

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

# both plate models emit harmless shape-inference warnings on every session start
ONNX_ERROR_SEVERITY = 3

NEUTRAL_OCR_CONFIDENCE = 1.0

ALLOW_SPINNING = "session.intra_op.allow_spinning"


def session_options() -> onnxruntime.SessionOptions:
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = plate_model_threads(os.cpu_count())
    # pool threads busy-wait after every run by default, taking the cores the detector needs:
    # on four cores 5.2 fps with spinning and 13.9 without, reading the same plates
    options.add_session_config_entry(ALLOW_SPINNING, "0")
    return options


class PlateReader:
    def __init__(self, *, device: str = "auto") -> None:
        onnxruntime.set_default_logger_severity(ONNX_ERROR_SEVERITY)
        # the severity above also hides the runtime's own fallback warning, so the one
        # case that matters - asking for a GPU and silently getting a CPU - is checked here
        if (
            device == "cuda"
            and "CUDAExecutionProvider" not in onnxruntime.get_available_providers()
        ):
            logger.warning("no CUDA execution provider available, plate models run on the CPU")
        options = session_options()
        self._detector = create_detector(
            DETECTION_MODEL, providers=providers_for(device), sess_options=options
        )
        self._recognizer = LicensePlateRecognizer(
            hub_ocr_model=OCR_MODEL, device=device, sess_options=options
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
    # The model scores every plate slot it has, and the unused trailing slots come back at
    # 1.0, so averaging all of them rates a short reading higher than a long correct one.
    return float(np.mean(probabilities[:length]))
