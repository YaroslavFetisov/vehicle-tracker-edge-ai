"""Measure plate recognition accuracy on the labelled OpenALPR EU benchmark."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from vehicle_tracker.evaluation import Comparison, OcrScore, closest, parse_annotation, score
from vehicle_tracker.plate import NON_PLATE_CHARACTERS, normalize
from vehicle_tracker.plate_reader import PlateReader

DEFAULT_DATASET = Path("data/openalpr-eu")
WORST_READINGS_SHOWN = 15


def ground_truth(annotation: Path) -> list[str]:
    plates = parse_annotation(annotation.read_text(encoding="utf-8"))
    if not plates:
        raise RuntimeError(f"no plate text in {annotation}")
    return [NON_PLATE_CHARACTERS.sub("", plate.upper()) for plate in plates]


def evaluate(images: list[Path], reader: PlateReader) -> tuple[list[Comparison], float]:
    comparisons = []
    seconds = 0.0

    for image_path in images:
        annotation = image_path.with_suffix(".txt")
        if not annotation.is_file():
            raise RuntimeError(f"missing annotation for {image_path.name}")
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"cannot decode {image_path}")

        started = time.perf_counter()
        reading = reader.read(image)
        seconds += time.perf_counter() - started

        expected = ground_truth(annotation)
        actual = reading.text if reading is not None else None
        comparisons.append(Comparison(image_path.name, closest(expected, actual), actual))

    return comparisons, seconds


def print_report(comparisons: list[Comparison], seconds: float) -> None:
    strict = score(comparisons)
    folded_comparisons = [comparison.folded() for comparison in comparisons]
    folded = score(folded_comparisons)
    rejected = [
        comparison.expected for comparison in comparisons if normalize(comparison.expected) is None
    ]
    latency = seconds / max(strict.images, 1) * 1000

    print(f"\nimages: {strict.images}")
    print(f"ground truth the letter plus digit rule rejects: {len(rejected)} {rejected}")
    print(f"readings produced: {strict.readings} ({strict.reading_rate:.1%})")
    print(f"exact match: {strict.exact} ({strict.exact_match:.1%})")
    print(f"exact match with O/0 and I/1 folded: {folded.exact} ({folded.exact_match:.1%})")
    print(f"character error rate: {strict.character_error_rate:.3f}")
    print(f"character error rate folded: {folded.character_error_rate:.3f}")
    print(f"latency: {latency:.1f} ms per image")

    print("\nreadings still wrong after folding, expected -> read")
    wrong = sorted(
        (item for item in folded_comparisons if not item.correct),
        key=lambda item: item.errors,
        reverse=True,
    )
    for item in wrong[:WORST_READINGS_SHOWN]:
        print(f"  {item.image:<13} {item.expected:<10} -> {item.actual}")

    print_table(strict, folded, latency)


def print_table(strict: OcrScore, folded: OcrScore, latency: float) -> None:
    print("\n| images | readings | exact | exact folded | cer | cer folded | ms per image |")
    print("|---|---|---|---|---|---|---|")
    print(
        f"| {strict.images} | {strict.reading_rate:.1%} | {strict.exact_match:.1%} "
        f"| {folded.exact_match:.1%} | {strict.character_error_rate:.3f} "
        f"| {folded.character_error_rate:.3f} | {latency:.1f} |"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="eval-ocr", description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--device", default="cpu", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--limit", type=int, help="evaluate only the first N images")
    return parser.parse_args(argv)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()

    images = sorted(args.dataset.glob("*.jpg"))
    if not images:
        raise SystemExit(
            f"no images in {args.dataset} - run: python scripts/fetch_assets.py ocr-benchmark"
        )
    if args.limit:
        images = images[: args.limit]

    print(f"dataset: {args.dataset}, {len(images)} images, device {args.device}")
    print("the reader gets the whole photo here, in the pipeline it gets a vehicle crop")
    comparisons, seconds = evaluate(images, PlateReader(device=args.device))
    print_report(comparisons, seconds)


if __name__ == "__main__":
    main()
