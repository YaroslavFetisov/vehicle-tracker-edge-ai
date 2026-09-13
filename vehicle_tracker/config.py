from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

STREAM_SCHEMES = ("rtsp://", "rtmp://", "http://", "https://")

DEFAULT_WEIGHTS = Path("weights/yolo11n.pt")
# below 0.5 the detector starts flickering on distant vehicles and roadside furniture,
# which fragments tracks into short lived ids - see the measurements in README.md
DEFAULT_CONFIDENCE = 0.5
DEFAULT_IMAGE_SIZE = 640


@dataclass(frozen=True)
class Config:
    source: str | int
    device: str
    weights: Path
    confidence: float
    image_size: int
    display: bool = True
    output: Path | None = None
    filter_stream: bool = False


def parse_source(value: str) -> str | int:
    """Resolve a CLI source into a stream URL, a webcam index or a video file path."""
    if value.startswith(STREAM_SCHEMES):
        return value
    if value.isdigit():
        return int(value)
    path = Path(value)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"video source not found: {value}")
    return str(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vehicle-tracker",
        description="Real-time vehicle tracking with licence plate and colour recognition",
    )
    parser.add_argument(
        "--source",
        required=True,
        type=parse_source,
        help="RTSP URL, path to a video file, or webcam index",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="inference device (default: auto)",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help=f"vehicle detection weights (default: {DEFAULT_WEIGHTS})",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONFIDENCE,
        help=f"detection confidence threshold (default: {DEFAULT_CONFIDENCE})",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help=f"inference image size (default: {DEFAULT_IMAGE_SIZE})",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="do not open a window, for servers and containers",
    )
    parser.add_argument(
        "--save-video",
        type=Path,
        metavar="PATH",
        help="write the annotated video to this file",
    )
    parser.add_argument(
        "--filter-stream",
        action="store_true",
        help="only show and save frames that carry a vehicle with a readable plate",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> Config:
    args = build_parser().parse_args(argv)
    return Config(
        source=args.source,
        device=args.device,
        weights=args.weights,
        confidence=args.conf,
        image_size=args.imgsz,
        display=not args.no_display,
        output=args.save_video,
        filter_stream=args.filter_stream,
    )
