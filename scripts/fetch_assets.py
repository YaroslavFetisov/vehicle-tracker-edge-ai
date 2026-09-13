"""Download model weights and sample footage used for local runs and benchmarks."""

from __future__ import annotations

import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
WEIGHTS_DIR = PROJECT_ROOT / "weights"

# the stock footage CDN rejects the default urllib user agent
USER_AGENT = "vehicle-tracker-edge-ai/0.1"

DETECTION_MODEL = "yolo11n.pt"

# Pexels footage, free to use without attribution.
SAMPLE_VIDEOS = {
    "highway_traffic.mp4": (
        "https://videos.pexels.com/video-files/854671/854671-hd_1920_1080_25fps.mp4"
    ),
    "dashcam_highway.mp4": (
        "https://videos.pexels.com/video-files/5915075/5915075-hd_1920_1080_30fps.mp4"
    ),
}


def download(url: str, target: Path) -> None:
    if target.exists():
        print(f"{target.name}: already present")
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    print(f"{target.name}: downloading")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request) as response, partial.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    except urllib.error.URLError as exc:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"failed to download {url}: {exc.reason}") from exc
    partial.replace(target)
    print(f"{target.name}: {target.stat().st_size / 1e6:.1f} MB")


def fetch_detection_weights() -> None:
    target = WEIGHTS_DIR / DETECTION_MODEL
    if target.exists():
        print(f"{DETECTION_MODEL}: already present")
        return

    from ultralytics import YOLO

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"{DETECTION_MODEL}: downloading")
    YOLO(str(target))
    print(f"{DETECTION_MODEL}: {target.stat().st_size / 1e6:.1f} MB")


def fetch_plate_models() -> None:
    # the plate detector and the recognizer pull their own weights from a model hub on
    # first use, which would otherwise happen on an edge device that is already offline
    sys.path.insert(0, str(PROJECT_ROOT))
    from vehicle_tracker.plate_reader import PlateReader

    print("plate models: downloading if missing")
    PlateReader(device="cpu")
    print("plate models: ready")


def main() -> None:
    fetch_detection_weights()
    fetch_plate_models()
    for name, url in SAMPLE_VIDEOS.items():
        download(url, DATA_DIR / name)


if __name__ == "__main__":
    main()
