from __future__ import annotations

from pathlib import Path

import cv2
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = PROJECT_ROOT / "weights" / "yolo11n.pt"
SAMPLE_VIDEO = PROJECT_ROOT / "data" / "dashcam_highway.mp4"

# the silver hatchback leading the camera car carries this plate
KNOWN_PLATE = "CF5775"
FRAMES_TO_READ = 400


@pytest.mark.integration
def test_the_known_plate_is_read_from_the_sample_clip():
    if not WEIGHTS.is_file() or not SAMPLE_VIDEO.is_file():
        pytest.skip("weights or sample footage missing - run scripts/fetch_assets.py")

    # imported here so that the unit test run does not need the ML stack installed
    from vehicle_tracker.detector import VehicleDetector
    from vehicle_tracker.plate_reader import PlateReader
    from vehicle_tracker.tracks import TrackRegistry

    detector = VehicleDetector(WEIGHTS, device="cpu", confidence=0.5, image_size=640)
    plate_reader = PlateReader(device="cpu")
    registry = TrackRegistry()

    capture = cv2.VideoCapture(str(SAMPLE_VIDEO))
    plates = set()
    try:
        for frame_index in range(FRAMES_TO_READ):
            ok, frame = capture.read()
            if not ok:
                break
            detections = detector.track(frame)
            states = registry.update(frame_index, frame, detections)
            for detection in registry.due_for_plate(frame_index, detections):
                x1, y1, x2, y2 = detection.bbox
                crop = frame[max(y1, 0) : y2, max(x1, 0) : x2]
                registry.record_plate(detection.track_id, frame_index, plate_reader.read(crop))
            plates.update(state.plate for state in states.values() if state.plate is not None)
    finally:
        capture.release()

    assert KNOWN_PLATE in plates, f"expected {KNOWN_PLATE} among the readings, got {sorted(plates)}"
