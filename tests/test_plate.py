from __future__ import annotations

import pytest

from vehicle_tracker.plate import looks_local, normalize, vote_weight


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("AA1234BB", "AA1234BB"),
        ("aa1234bb", "AA1234BB"),
        ("AA 1234 BB", "AA1234BB"),
        ("AA-1234-BB", "AA1234BB"),
        ("CF5775", "CF5775"),
    ],
)
def test_readings_are_normalised(raw, expected):
    assert normalize(raw) == expected


@pytest.mark.parametrize("raw", ["", "AB", "A1", "ABCDEFGHIJK", "!!!"])
def test_implausible_readings_are_rejected(raw):
    assert normalize(raw) is None


def test_local_format_is_recognised():
    assert looks_local("AA1234BB")


@pytest.mark.parametrize("text", ["CF5775", "AA123BB", "1234ABCD", "AAAAAAAA"])
def test_other_formats_are_not_local(text):
    assert not looks_local(text)


def test_local_format_outweighs_a_foreign_one():
    assert vote_weight("AA1234BB") > vote_weight("CF5775")


def test_foreign_plates_still_carry_a_vote():
    assert vote_weight("CF5775") > 0
