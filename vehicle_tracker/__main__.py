from __future__ import annotations

import logging

import onnxruntime

from vehicle_tracker.config import parse_args
from vehicle_tracker.pipeline import run

# the plate models trigger harmless shape-inference warnings on every session start
ONNX_ERROR_SEVERITY = 3


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    onnxruntime.set_default_logger_severity(ONNX_ERROR_SEVERITY)
    run(parse_args())


if __name__ == "__main__":
    main()
