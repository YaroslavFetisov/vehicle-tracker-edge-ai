from __future__ import annotations

import csv
import io
from collections import Counter
from collections.abc import Collection, Iterable
from dataclasses import dataclass, field

# OpenALPR annotation line: image name, plate box (x, y, width, height), plate text
ANNOTATION_FIELDS = 6

# nearly identical in plate fonts and labelled inconsistently, so also scored folded
AMBIGUOUS_CHARACTERS = str.maketrans({"O": "0", "I": "1"})


def parse_annotation(content: str) -> list[str]:
    """Read the plate texts out of one OpenALPR benchmark annotation file."""
    plates = []
    for line in content.splitlines():
        if not line.strip():
            continue
        # the text is taken as the whole remainder, so a plate written with a space survives
        fields = line.split(None, ANNOTATION_FIELDS - 1)
        if len(fields) != ANNOTATION_FIELDS:
            raise ValueError(f"malformed annotation line: {line!r}")
        plates.append(fields[-1].strip())
    return plates


def fold_ambiguous(text: str) -> str:
    return text.translate(AMBIGUOUS_CHARACTERS)


def edit_distance(expected: str, actual: str) -> int:
    """Number of single character insertions, deletions and substitutions between two strings."""
    previous = list(range(len(actual) + 1))
    for row, expected_char in enumerate(expected, start=1):
        current = [row]
        for column, actual_char in enumerate(actual, start=1):
            current.append(
                min(
                    previous[column] + 1,
                    current[column - 1] + 1,
                    previous[column - 1] + (expected_char != actual_char),
                )
            )
        previous = current
    return previous[-1]


def closest(expected: list[str], actual: str | None) -> str:
    """Pick the labelled plate a reading should be scored against."""
    # an image can carry several labelled plates while the reader returns one
    if actual is None or len(expected) == 1:
        return expected[0]
    return min(expected, key=lambda plate: edit_distance(plate, actual))


@dataclass(frozen=True)
class Comparison:
    """One labelled image and what the reader made of it."""

    image: str
    expected: str
    actual: str | None

    @property
    def errors(self) -> int:
        if self.actual is None:
            return len(self.expected)
        return edit_distance(self.expected, self.actual)

    @property
    def correct(self) -> bool:
        return self.actual == self.expected

    def folded(self) -> Comparison:
        actual = None if self.actual is None else fold_ambiguous(self.actual)
        return Comparison(self.image, fold_ambiguous(self.expected), actual)


@dataclass
class OcrScore:
    """Recognition accuracy over a labelled set, accumulated one image at a time."""

    images: int = 0
    readings: int = 0
    exact: int = 0
    character_errors: int = 0
    characters: int = 0

    def add(self, expected: str, actual: str | None) -> None:
        self.images += 1
        self.characters += len(expected)
        # a missed plate costs every character, or skipping hard images would look accurate
        if actual is None:
            self.character_errors += len(expected)
            return
        self.readings += 1
        self.character_errors += edit_distance(expected, actual)
        if actual == expected:
            self.exact += 1

    @property
    def exact_match(self) -> float:
        return 0.0 if self.images == 0 else self.exact / self.images

    @property
    def reading_rate(self) -> float:
        return 0.0 if self.images == 0 else self.readings / self.images

    @property
    def character_error_rate(self) -> float:
        return 0.0 if self.characters == 0 else self.character_errors / self.characters


def score(comparisons: Iterable[Comparison]) -> OcrScore:
    result = OcrScore()
    for comparison in comparisons:
        result.add(comparison.expected, comparison.actual)
    return result


COLOR_LABEL_FIELDS = ("vehicle", "clip", "frame", "x1", "y1", "x2", "y2", "accepted")
ACCEPTED_SEPARATOR = "|"


@dataclass(frozen=True)
class ColorLabel:
    """One crop of a labelled vehicle, with every colour name a person would accept for it."""

    vehicle: str
    clip: str
    frame: int
    bbox: tuple[int, int, int, int]
    accepted: frozenset[str]


def parse_color_labels(content: str, known_colors: Collection[str]) -> list[ColorLabel]:
    reader = csv.DictReader(io.StringIO(content))
    if tuple(reader.fieldnames or ()) != COLOR_LABEL_FIELDS:
        raise ValueError(f"expected the columns {', '.join(COLOR_LABEL_FIELDS)}")

    labels = []
    for row in reader:
        accepted = frozenset(name for name in row["accepted"].split(ACCEPTED_SEPARATOR) if name)
        # a misspelt name would quietly count every crop of that vehicle as wrong
        unknown = sorted(accepted - set(known_colors))
        if not accepted or unknown:
            raise ValueError(f"bad accepted colours for {row['vehicle']}: {row['accepted']!r}")
        bbox = (int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"]))
        labels.append(ColorLabel(row["vehicle"], row["clip"], int(row["frame"]), bbox, accepted))
    return labels


@dataclass
class ColorScore:
    """Accuracy per crop, and per vehicle after the majority vote the tracker takes."""

    crops: int = 0
    correct_crops: int = 0
    vehicles: int = 0
    correct_vehicles: int = 0
    # vehicle -> the colour its crops voted for, None when no crop could be classified
    misjudged: dict[str, str | None] = field(default_factory=dict)

    @property
    def crop_accuracy(self) -> float:
        return 0.0 if self.crops == 0 else self.correct_crops / self.crops

    @property
    def vehicle_accuracy(self) -> float:
        return 0.0 if self.vehicles == 0 else self.correct_vehicles / self.vehicles


def score_colors(predictions: Iterable[tuple[ColorLabel, str | None]]) -> ColorScore:
    result = ColorScore()
    votes: dict[str, Counter[str]] = {}
    accepted: dict[str, frozenset[str]] = {}

    for label, predicted in predictions:
        result.crops += 1
        if predicted in label.accepted:
            result.correct_crops += 1
        accepted[label.vehicle] = label.accepted
        vehicle_votes = votes.setdefault(label.vehicle, Counter())
        if predicted is not None:
            vehicle_votes[predicted] += 1

    for vehicle, vehicle_votes in votes.items():
        result.vehicles += 1
        majority = vehicle_votes.most_common(1)[0][0] if vehicle_votes else None
        if majority in accepted[vehicle]:
            result.correct_vehicles += 1
        else:
            result.misjudged[vehicle] = majority
    return result
