from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

STREAM_SCHEMES = ("rtsp://", "rtmp://", "http://", "https://")


@dataclass(frozen=True)
class Config:
    source: str | int
    device: str


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
    return parser


def parse_args(argv: Sequence[str] | None = None) -> Config:
    args = build_parser().parse_args(argv)
    return Config(source=args.source, device=args.device)
