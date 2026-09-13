from __future__ import annotations

from collections import Counter
from pathlib import Path

import cv2
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = PROJECT_ROOT / "weights" / "yolo11n.pt"
SAMPLE_VIDEO = PROJECT_ROOT / "data" / "highway_traffic.mp4"

FRAMES_TO_TRACK = 10


def read_frames(video: Path, count: int):
    capture = cv2.VideoCapture(str(video))
    try:
        for _ in range(count):
            ok, frame = capture.read()
            if not ok:
                break
            yield frame
    finally:
        capture.release()


@pytest.mark.integration
def test_vehicles_keep_their_id_across_frames():
    if not WEIGHTS.is_file() or not SAMPLE_VIDEO.is_file():
        pytest.skip("weights or sample footage missing - run scripts/fetch_assets.py")

    # imported here so that the unit test run does not need the ML stack installed
    from vehicle_tracker.detector import VehicleDetector

    detector = VehicleDetector(WEIGHTS, device="cpu", confidence=0.35, image_size=640)

    seen = Counter()
    for frame in read_frames(SAMPLE_VIDEO, FRAMES_TO_TRACK):
        for detection in detector.track(frame):
            seen[detection.track_id] += 1

    assert seen, "no vehicles detected in the sample footage"
    assert max(seen.values()) >= FRAMES_TO_TRACK // 2, (
        f"no track survived half of the frames: {dict(seen)}"
    )
