from __future__ import annotations

import re
from dataclasses import dataclass

UKRAINIAN_FORMAT = re.compile(r"^[A-Z]{2}\d{4}[A-Z]{2}$")
NON_PLATE_CHARACTERS = re.compile(r"[^A-Z0-9]")

MIN_PLATE_LENGTH = 4
MAX_PLATE_LENGTH = 10

# A reading in the local plate format is more likely to be right than one that is not,
# but foreign vehicles are read here too, so the format only weights the vote.
LOCAL_FORMAT_WEIGHT = 1.5
FOREIGN_FORMAT_WEIGHT = 1.0


@dataclass(frozen=True)
class PlateReading:
    text: str
    detection_confidence: float
    ocr_confidence: float


def normalize(text: str) -> str | None:
    cleaned = NON_PLATE_CHARACTERS.sub("", text.upper())
    if not MIN_PLATE_LENGTH <= len(cleaned) <= MAX_PLATE_LENGTH:
        return None
    return cleaned


def looks_local(text: str) -> bool:
    return bool(UKRAINIAN_FORMAT.match(text))


def vote_weight(text: str) -> float:
    return LOCAL_FORMAT_WEIGHT if looks_local(text) else FOREIGN_FORMAT_WEIGHT
