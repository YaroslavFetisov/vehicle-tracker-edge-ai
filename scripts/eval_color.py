"""Measure colour accuracy on hand labelled vehicles from the sample footage.

The labels hold up to six crops per vehicle spread over its track, the way the pipeline samples
a vehicle from every distance. A silver car is accepted as grey or white and a dark one as black
or grey, since a person would accept either.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from vehicle_tracker.color import PALETTE, estimate_color
from vehicle_tracker.evaluation import ColorLabel, ColorScore, parse_color_labels, score_colors

DEFAULT_LABELS = PROJECT_ROOT / "labels" / "vehicle_colors.csv"
DEFAULT_DATA = PROJECT_ROOT / "data"


def predict_clip(video: Path, labels: list[ColorLabel]) -> list[tuple[ColorLabel, str | None]]:
    by_frame: dict[int, list[ColorLabel]] = defaultdict(list)
    for label in labels:
        by_frame[label.frame].append(label)
    last_frame = max(by_frame)

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {video} - run: python scripts/fetch_assets.py videos")

    predictions = []
    try:
        # decoded in order: seeking in an mp4 can land on a neighbouring frame
        for index in range(last_frame + 1):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"{video} ended at frame {index}, labels need {last_frame}")
            for label in by_frame.get(index, []):
                color = estimate_color(frame, label.bbox)
                predictions.append((label, color.name if color is not None else None))
    finally:
        capture.release()
    return predictions


def print_report(result: ColorScore, accepted: dict[str, frozenset[str]]) -> None:
    print(f"per crop: {result.correct_crops}/{result.crops} ({result.crop_accuracy:.1%})")
    print(
        f"per vehicle, majority of its crops: {result.correct_vehicles}/{result.vehicles} "
        f"({result.vehicle_accuracy:.1%})"
    )

    print("\nmisjudged vehicles, accepted -> voted")
    for vehicle, voted in sorted(result.misjudged.items()):
        print(f"  {vehicle:<13} {'/'.join(sorted(accepted[vehicle])):<12} -> {voted}")

    print("\n| vehicles | crops | per crop | per vehicle |")
    print("|---|---|---|---|")
    print(
        f"| {result.vehicles} | {result.crops} | {result.crop_accuracy:.1%} "
        f"| {result.vehicle_accuracy:.1%} |"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="eval-color", description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="folder with the clips")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    labels = parse_color_labels(args.labels.read_text(encoding="utf-8"), PALETTE)

    by_clip: dict[str, list[ColorLabel]] = defaultdict(list)
    for label in labels:
        by_clip[label.clip].append(label)
    vehicles = {label.vehicle for label in labels}
    print(f"labels: {args.labels.name}, {len(vehicles)} vehicles, {len(labels)} crops")

    predictions = []
    for clip, clip_labels in sorted(by_clip.items()):
        predictions.extend(predict_clip(args.data / clip, clip_labels))

    accepted = {label.vehicle: label.accepted for label in labels}
    print_report(score_colors(predictions), accepted)


if __name__ == "__main__":
    main()
