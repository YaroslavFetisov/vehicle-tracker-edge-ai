"""Download model weights and sample footage used for local runs and benchmarks."""

from __future__ import annotations

import argparse
import json
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

# OpenALPR end to end benchmark: vehicle photos with the plate text as ground truth, used by
# scripts/eval_ocr.py. Only the european half is fetched, the repository itself is ~190 MB.
OCR_BENCHMARK_DIR = DATA_DIR / "openalpr-eu"
OCR_BENCHMARK_LISTING = "https://api.github.com/repos/openalpr/benchmarks/contents/endtoend/eu"


def read_url(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request) as response:
            return response.read()
    except urllib.error.URLError as exc:
        raise RuntimeError(f"failed to fetch {url}: {exc.reason}") from exc


def download(url: str, target: Path, *, announce: bool = True) -> None:
    if target.exists():
        if announce:
            print(f"{target.name}: already present")
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    if announce:
        print(f"{target.name}: downloading")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request) as response, partial.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    except urllib.error.URLError as exc:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"failed to download {url}: {exc.reason}") from exc
    partial.replace(target)
    if announce:
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


def fetch_ocr_benchmark() -> None:
    entries = json.loads(read_url(OCR_BENCHMARK_LISTING))
    files = [entry for entry in entries if entry["type"] == "file"]
    missing = [entry for entry in files if not (OCR_BENCHMARK_DIR / entry["name"]).exists()]
    if not missing:
        print(f"ocr benchmark: {len(files)} files already present")
        return

    print(f"ocr benchmark: downloading {len(missing)} of {len(files)} files")
    for entry in missing:
        download(entry["download_url"], OCR_BENCHMARK_DIR / entry["name"], announce=False)
    print(f"ocr benchmark: ready in {OCR_BENCHMARK_DIR}")


TARGETS = ("models", "videos", "ocr-benchmark")
DEFAULT_TARGETS = ("models", "videos")


def main() -> None:
    parser = argparse.ArgumentParser(prog="fetch-assets", description=__doc__)
    parser.add_argument(
        "targets",
        nargs="*",
        default=list(DEFAULT_TARGETS),
        help=(
            "what to download: models (detector and plate models), videos (sample footage), "
            "ocr-benchmark (labelled plates for scripts/eval_ocr.py). "
            f"Default: {' '.join(DEFAULT_TARGETS)}"
        ),
    )
    args = parser.parse_args()
    unknown = sorted(set(args.targets) - set(TARGETS))
    if unknown:
        parser.error(f"unknown target: {', '.join(unknown)}")

    if "models" in args.targets:
        fetch_detection_weights()
        fetch_plate_models()
    if "videos" in args.targets:
        for name, url in SAMPLE_VIDEOS.items():
            download(url, DATA_DIR / name)
    if "ocr-benchmark" in args.targets:
        fetch_ocr_benchmark()


if __name__ == "__main__":
    main()
