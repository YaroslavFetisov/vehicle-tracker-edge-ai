"""Measure processing throughput and per stage latency, with and without the per track cache."""

from __future__ import annotations

import argparse
import dataclasses
import logging
import math
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import onnxruntime
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from vehicle_tracker.config import (
    DEFAULT_CONFIDENCE,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_WEIGHTS,
    Config,
    parse_source,
)
from vehicle_tracker.metrics import Metrics
from vehicle_tracker.pipeline import build_pipeline
from vehicle_tracker.tracks import TrackRegistry
from vehicle_tracker.video_source import VideoSource

WARMUP_FRAMES = 20
DEFAULT_FRAMES = 400

# the naive baseline: colour and plate recomputed for every vehicle on every frame
UNCACHED_POLICY = {
    "color_interval": 1,
    "max_color_samples": sys.maxsize,
    "plate_interval": 1,
    "max_plate_attempts": sys.maxsize,
    "max_plate_reads_per_frame": sys.maxsize,
    "min_vehicle_height_fraction": 0.0,
    "confident_plate_score": math.inf,
}
POLICIES: dict[str, dict[str, object]] = {"cached": {}, "uncached": UNCACHED_POLICY}


@dataclass
class Run:
    device: str
    policy: str
    metrics: Metrics
    tracks: set[int] = field(default_factory=set)
    plates: dict[int, str] = field(default_factory=dict)

    @property
    def frame_ms(self) -> float:
        return 0.0 if self.metrics.average_fps == 0 else 1000 / self.metrics.average_fps


def measure(config: Config, policy: str, *, frames: int, warmup: int) -> Run:
    metrics = Metrics()
    registry = TrackRegistry(**POLICIES[policy])
    pipeline = build_pipeline(config, registry=registry, metrics=metrics)
    run = Run(device=config.device, policy=policy, metrics=metrics)

    with VideoSource(config.source) as source:
        for frame_index, frame in enumerate(source):
            # the first frames pay for lazy kernel loading and cuDNN autotuning
            if frame_index == warmup:
                metrics.reset()

            started = time.perf_counter()
            states = pipeline.process(frame_index, frame)
            metrics.frame_done(time.perf_counter() - started, had_vehicles=bool(states))

            run.tracks.update(states)
            for track_id, state in states.items():
                if state.plate is not None:
                    run.plates[track_id] = state.plate

            if frame_index + 1 >= warmup + frames:
                break

    if metrics.frames < frames:
        raise RuntimeError(
            f"video ran out after {metrics.frames + warmup} frames, asked for {warmup + frames}"
        )
    return run


def describe_source(source: str | int) -> str:
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video source: {source}")
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS)
    finally:
        capture.release()
    return f"{source} {width}x{height} at {fps:.0f} fps"


def print_environment(config: Config, *, frames: int, warmup: int) -> None:
    print("environment")
    print(
        f"  python {platform.python_version()}, torch {torch.__version__}, "
        f"onnxruntime {onnxruntime.__version__}"
    )
    if torch.cuda.is_available():
        print(f"  gpu {torch.cuda.get_device_name(0)}, cuda {torch.version.cuda}")
    print(f"  onnxruntime providers {', '.join(onnxruntime.get_available_providers())}")
    print(f"  source {describe_source(config.source)}")
    print(f"  weights {config.weights}, conf {config.confidence}, imgsz {config.image_size}")
    print(f"  {frames} measured frames after {warmup} warm up frames")
    print("  fps is processing throughput: decoding runs in the reader thread, display is off")


def print_run(run: Run) -> None:
    print(f"\n{run.device} / {run.policy}")
    for line in run.metrics.summary():
        print(f"  {line}")
    plates = ", ".join(f"{track_id}:{plate}" for track_id, plate in sorted(run.plates.items()))
    print(f"  tracks seen: {len(run.tracks)}")
    print(f"  plates read: {plates or 'none'}")


def print_table(runs: list[Run]) -> None:
    print("\n| device | policy | fps | ms/frame | detect | colour | plate | tracks | plates |")
    print("|---|---|---|---|---|---|---|---|---|")
    for run in runs:
        metrics = run.metrics
        print(
            f"| {run.device} | {run.policy} | {metrics.average_fps:.1f} | {run.frame_ms:.1f} "
            f"| {metrics.ms_per_frame('detect'):.1f} | {metrics.ms_per_frame('color'):.1f} "
            f"| {metrics.ms_per_frame('plate'):.1f} | {len(run.tracks)} | {len(run.plates)} |"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="benchmark", description=__doc__)
    parser.add_argument("--source", required=True, type=parse_source, help="video file or stream")
    parser.add_argument(
        "--device",
        nargs="+",
        default=["cpu"],
        choices=("cpu", "cuda"),
        help="devices to measure, in order (default: cpu)",
    )
    parser.add_argument(
        "--policy",
        nargs="+",
        default=list(POLICIES),
        choices=tuple(POLICIES),
        help="cached uses the per track cache, uncached recomputes everything every frame",
    )
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES, help="frames to measure")
    parser.add_argument("--warmup", type=int, default=WARMUP_FRAMES, help="frames to discard first")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--conf", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMAGE_SIZE)
    return parser.parse_args(argv)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    if "cuda" in args.device and not torch.cuda.is_available():
        raise SystemExit("cuda was requested but torch reports no available device")

    config = Config(
        source=args.source,
        device=args.device[0],
        weights=args.weights,
        confidence=args.conf,
        image_size=args.imgsz,
    )
    print_environment(config, frames=args.frames, warmup=args.warmup)

    runs = []
    for device in args.device:
        for policy in args.policy:
            run = measure(
                dataclasses.replace(config, device=device),
                policy,
                frames=args.frames,
                warmup=args.warmup,
            )
            print_run(run)
            runs.append(run)

    print_table(runs)


if __name__ == "__main__":
    main()
